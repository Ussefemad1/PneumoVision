"""Round 3 threshold sweep: the binned capture (tools/pv/r3_cache.at_threshold).

The 0.75 groups must be bit-for-bit those of the plain capture (so the main
run means the same), every swept threshold must match a capture made at that
threshold, and cached must equal live at a swept threshold too. Real
CMSMAFusion with the stub readers (and medpatch's real LSTM where noted).
SYNTHETIC data only.
"""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from r3_stubs import r3_model
from test_round3_cache import StayDataset
from test_round3_tools import lineage, opts  # noqa: F401 -- pytest fixture

from tools.pv import manifest, r3_cache, round3

TOL = 1e-5
SWEEPS = {
    "phenotyping": [0.65, 0.75, 0.85],
    "in-hospital-mortality": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
}
CPU = torch.device("cpu")


def setup(task, real_ehr=False, scale=2.5, **extra):
    """A model whose confidences spread over [0.5, 1], so every threshold splits differently."""
    from datasets.DataFusion import my_collate  # noqa: PLC0415

    args, model = r3_model(task, real_ehr=real_ehr, **extra)
    fm = model.fusion_model
    with torch.no_grad():
        for m in args.modalities.split("-"):
            getattr(fm, f"{m.lower()}_confidence_predictor").confidence_layer.weight.mul_(scale)
    model.equalize()
    return args, model, partial(my_collate, args=args)


def batch(task, collate, n=6, seed=0):
    ds = StayDataset(n, task, seed)
    return collate([ds[i] for i in range(n)])


def plain_at(model, b, t):
    """A plain capture made by a model whose own threshold is t."""
    m = copy.deepcopy(model)
    r3_cache.set_threshold(m.fusion_model.args, t)
    return r3_cache.capture_batch(m, b, CPU)


# ── 1. the 0.75 split is the current capture, bit for bit ────────────────────


@pytest.mark.parametrize("task", list(SWEEPS))
@pytest.mark.parametrize("real_ehr", [False, True])
def test_main_groups_are_bit_identical_with_the_sweep(task, real_ehr):
    args, model, collate = setup(task, real_ehr=real_ehr)
    b = batch(task, collate)
    plain = r3_cache.capture_batch(model, b, CPU)
    swept = r3_cache.capture_batch(model, b, CPU, SWEEPS[task])
    assert set(plain) < set(swept)  # the sweep only adds keys
    for key, value in plain.items():
        assert torch.equal(swept[key], value), key
    at_main = r3_cache.at_threshold(swept, 0.75, 0.75)
    assert at_main is swept  # the stored main groups, untouched


@pytest.mark.parametrize("task", list(SWEEPS))
def test_each_swept_threshold_matches_a_capture_at_it(task):
    args, model, collate = setup(task)
    b = batch(task, collate)
    swept = r3_cache.capture_batch(model, b, CPU, SWEEPS[task])
    counts = set()
    for t in SWEEPS[task]:
        derived = r3_cache.at_threshold(swept, t, 0.75)
        reference = plain_at(model, b, t)
        for key, value in reference.items():
            if key.endswith(("_n_high", "_n_low")):
                assert torch.equal(derived[key], value), (t, key)  # counts exact
            else:
                torch.testing.assert_close(derived[key], value, atol=1e-6, rtol=1e-5,
                                           msg=f"{t} {key}")  # fmt: skip
        counts.add(float(derived["rr_n_high"].sum()))
    assert len(counts) > 1, "the thresholds must split the tokens differently"


# ── 2. cached == live at a swept threshold ───────────────────────────────────


@pytest.mark.parametrize("task", list(SWEEPS))
@pytest.mark.parametrize("real_ehr", [False, True])
def test_cached_equals_live_at_swept_thresholds(task, real_ehr):
    args, model, collate = setup(task, real_ehr=real_ehr)
    b = batch(task, collate)
    for t in SWEEPS[task]:
        report = r3_cache.compare_live_cached(model, b, CPU, threshold=t, sweep=SWEEPS[task])
        worst = {k: v for k, v in report.items() if k.startswith("max_abs")}
        assert max(worst.values()) <= TOL, (t, worst)


def test_equivalence_step_checks_main_and_one_swept_threshold(lineage, monkeypatch):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping", "--sweep-thresholds", "0.65", "0.75", "0.85")
    run = round3.plan(o)
    args, model, collate = setup("phenotyping")
    b = batch("phenotyping", collate)
    monkeypatch.setattr(round3, "trainer_args", lambda run_, *a, **k: args)
    monkeypatch.setattr(round3, "live_loaders", lambda a: (None, [b], None))
    monkeypatch.setattr(round3, "build_trainer",
                        lambda a, loaders: SimpleNamespace(model=model, device=CPU))  # fmt: skip
    round3.step_equivalence(run, o)
    assert run["equivalence"]["threshold"] == 0.75
    assert run["equivalence_sweep"]["threshold"] in (0.65, 0.85)
    assert run["equivalence_sweep"]["max_abs_diff_late"] <= TOL


# ── cache, loader, fingerprint, separate directory ───────────────────────────


def test_loader_serves_batches_at_a_threshold(tmp_path):
    args, model, collate = setup("in-hospital-mortality")
    ds = StayDataset(8, "in-hospital-mortality", 1)
    sweep = SWEEPS["in-hospital-mortality"]
    r3_cache.build_split(model, ds, "val", tmp_path, batch_size=4, collate_fn=collate,
                         device=CPU, shard_size=4, thresholds=sweep)  # fmt: skip
    kw = {"shuffle": False, "drop_last": False, "classes": ["mortality"], "main": 0.75}
    plain = list(r3_cache.CacheLoader(tmp_path, "val", 4, **kw))
    at_main = list(r3_cache.CacheLoader(tmp_path, "val", 4, threshold=0.75, **kw))
    at_60 = list(r3_cache.CacheLoader(tmp_path, "val", 4, threshold=0.60, **kw))
    for a, b in zip(plain, at_main, strict=True):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(plain[0]["rr_n_high"], at_60[0]["rr_n_high"])
    live = collate([ds[i] for i in range(4)])
    m = copy.deepcopy(model)
    r3_cache.set_threshold(m.fusion_model.args, 0.60)
    m.eval()
    with torch.no_grad():
        x, seq, img, pairs, rr, dn, _ = r3_cache.live_inputs(live, CPU)
        out_live = m(x, seq, img, pairs, rr, dn)
        out_cached = model.fusion_model.forward_cached(at_60[0])
    assert (out_live["late"] - out_cached["late"]).abs().max() <= TOL


def test_old_plain_cache_still_matches_and_a_sweep_cache_does_not():
    args = SimpleNamespace(
        task="in-hospital-mortality",
        modalities="EHR-CXR-RR",
        num_classes=1,
        data_pairs="partial",
        ehr_data_dir="/nowhere",
        timestep=1.0,
        bert_model_name="dmis-lab/biobert-v1.1",
        **{f"{m}_confidence_threshold": 0.75 for m in ("ehr", "cxr", "rr", "dn")},
    )
    parents = {"ehr": {"sha256": "a"}, "cxr": {"sha256": "b"}, "rr": {"sha256": "c"}}
    old = r3_cache.fingerprint(args, parents)
    for key in ("sweep_thresholds", "capture_semantics"):
        del old[key]  # a meta.json written before these existed
    old["capture_code_commit"] = "7f2d783-and-older"
    assert r3_cache.fingerprint_mismatch(old, r3_cache.fingerprint(args, parents)) == []
    swept = r3_cache.fingerprint(args, parents, [0.6, 0.75])
    assert any("sweep_thresholds" in p for p in r3_cache.fingerprint_mismatch(old, swept))


def test_sweep_gets_its_own_cache_dir(lineage):  # noqa: F811
    tmp_path, _ = lineage
    plain = round3.plan(round3.build_parser().parse_args(
        ["--task", "mortality", "--data", "virtual", "--data-root",
         str(tmp_path / "smoke-r3-mortality")]))  # fmt: skip
    swept = round3.plan(round3.build_parser().parse_args(
        ["--task", "mortality", "--data", "virtual", "--data-root",
         str(tmp_path / "smoke-r3-mortality"), "--sweep-thresholds",
         "0.60", "0.65", "0.70", "0.75", "0.80", "0.85"]))  # fmt: skip
    assert plain["cache_dir"] != swept["cache_dir"]
    assert swept["cache_dir"].endswith("in-hospital-mortality-sweep-060-065-070-075-080-085")
    assert swept["sweep_thresholds"] == SWEEPS["in-hospital-mortality"]


# ── 3. the sweep step ────────────────────────────────────────────────────────


def _m(x):
    return {"macro_auroc": x, "macro_auprc": x / 2, "target_auroc": x, "target_auprc": x / 3,
            "target_auroc_ci": [x - 0.01, x + 0.01], "target_auprc_ci": [x / 4, x / 2]}  # fmt: skip


def test_sweep_step_trains_the_other_thresholds_and_leaves_the_manifest(lineage, monkeypatch):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(
        tmp_path, "mortality", "--sweep-thresholds", *map(str, SWEEPS["in-hospital-mortality"])
    )
    run = round3.plan(o)
    with pytest.raises(SystemExit, match="not recorded yet"):
        round3.step_sweep(run, o)
    run["scores"] = {"main": {"head": "late", "val": _m(0.86), "test": _m(0.865)}}
    manifest.append_row({"id": run["id"], "stage": "r3", "reader": "fusion",
                         "task": "in-hospital-mortality", "data": "virtual"})  # fmt: skip
    before = manifest.manifest_path().read_bytes()
    trained = {}

    def fake_train(run_, name, args, o_, threshold=None):
        trained[name] = (threshold, args.rr_confidence_threshold)
        return name

    monkeypatch.setattr(round3, "train_variant", fake_train)
    monkeypatch.setattr(round3, "score_variant", lambda run_, args, o_, threshold=None: {
        "head": "high", "val": _m(threshold), "test": _m(threshold)})  # fmt: skip
    path = round3.step_sweep(run, o)
    others = [t for t in SWEEPS["in-hospital-mortality"] if t != 0.75]
    assert trained == {f"theta_{t:.2f}": (t, t) for t in others}
    assert manifest.manifest_path().read_bytes() == before
    lines = path.read_text().splitlines()
    assert path.name == "r3_threshold_sweep.csv" and "threshold sweep (ablations)" in lines[0]
    names = [ln.split(",")[0] for ln in lines[2::2]]
    assert names == [
        "theta_0.75 main run (recorded)" if t == 0.75 else f"theta_{t:.2f}"
        for t in SWEEPS["in-hospital-mortality"]
    ]


# ── 5. disk check ────────────────────────────────────────────────────────────


def test_cache_refuses_to_start_without_disk(tmp_path, monkeypatch):
    import shutil  # noqa: PLC0415

    args, model, collate = setup("in-hospital-mortality")
    b = batch("in-hospital-mortality", collate, n=4)

    class Val(list):
        dataset = []

    loaders = [SimpleNamespace(dataset=list(range(1000))), Val([b]), SimpleNamespace(dataset=[])]
    trainer = SimpleNamespace(model=model, device=CPU)
    o = SimpleNamespace(disk_margin_gb=20.0)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=25 * 2**30))
    round3.check_disk(trainer, loaders, tmp_path / "c", o, SWEEPS["in-hospital-mortality"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=10 * 2**30))
    with pytest.raises(SystemExit, match="not enough disk"):
        round3.check_disk(trainer, loaders, tmp_path / "c", o, SWEEPS["in-hospital-mortality"])


# ── the shared copy is the main run only ─────────────────────────────────────


def test_share_copies_the_main_run_and_no_variant(lineage):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "mortality", "--step", "share", "--share-to", str(tmp_path / "shared"))
    run = round3.plan(o)
    save_dir = Path(run["save_dir"])
    best = save_dir / "in-hospital-mortality" / "c-msma" / "best_checkpoint_main.pth.tar"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"main")
    for variant in ("theta_0.60", "theta_0.85", "without_missingness_module"):
        v = save_dir / variant / "in-hospital-mortality" / "c-msma" / "best_checkpoint_main.pth.tar"
        v.parent.mkdir(parents=True)
        v.write_bytes(variant.encode())
    for name in ("run.json", "r3_results.csv", "r3_threshold_sweep.csv", "train_main.log"):
        (save_dir / name).write_text(name)
    with pytest.raises(SystemExit, match="not recorded"):
        round3.step_share(run, o)
    manifest.append_row(
        {
            "id": run["id"],
            "stage": "r3",
            "reader": "fusion",
            "task": "in-hospital-mortality",
            "data": "virtual",
            "file_path": best.as_posix(),
            "sha256": manifest.sha256_file(best),
        }
    )
    target = round3.step_share(run, o)
    copied = sorted(p.name for p in target.iterdir())
    assert copied == sorted(["best_checkpoint_main.pth.tar", "run.json", "r3_results.csv",
                             "r3_threshold_sweep.csv", "train_main.log"])  # fmt: skip
    assert (target / "best_checkpoint_main.pth.tar").read_bytes() == b"main"
    assert not any(p.is_dir() for p in target.iterdir())
