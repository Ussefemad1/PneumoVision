"""Round 3 training cost: needed keys only, no per-epoch bootstrap, fast scoring, step logs.

Stub readers, real CMSMAFusion and MSMA_Trainer, SYNTHETIC tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_round3_cache import ABLATIONS, StayDataset, _cached_trainer  # noqa: F401
from test_round3_tools import lineage, opts  # noqa: F401 -- pytest fixture
from test_threshold_sweep import SWEEPS, setup

from tools.pv import manifest, r3_cache, round3

CPU = torch.device("cpu")


def _sweep_cache(tmp_path, task="phenotyping"):
    args, model, collate = setup(task)
    r3_cache.build_split(model, StayDataset(8, task, 1), "train", tmp_path, batch_size=4,
                         collate_fn=collate, device=CPU, shard_size=4,
                         thresholds=SWEEPS[task])  # fmt: skip
    return args, model


# ── A2: a variant reads only its own keys ────────────────────────────────────


def test_main_loader_reads_no_sweep_keys_and_a_sweep_loader_only_its_own(tmp_path):
    _sweep_cache(tmp_path)
    kw = {"shuffle": False, "drop_last": False, "classes": ["c"] * 25, "main": 0.75}
    main = r3_cache.CacheLoader(tmp_path, "train", 4, **kw)
    at_65 = r3_cache.CacheLoader(tmp_path, "train", 4, threshold=0.65, **kw)
    main_keys, keys_65 = set(main._shard(0)), set(at_65._shard(0))
    assert not any("_t0" in k for k in main_keys)
    assert {k for k in keys_65 if "_t0" in k} and all("_t085" not in k for k in keys_65)
    on_disk = torch.load(main.paths[0], weights_only=True)
    every = sum(v.numel() * v.element_size() for k, v in on_disk.items()
                if k not in r3_cache.INDEX_KEYS)  # fmt: skip
    assert main.nbytes() < at_65.nbytes() < every * len(main.paths)
    preloaded = r3_cache.CacheLoader(tmp_path, "train", 4, preload=True, **kw)
    assert set(preloaded._kept[0]) == main_keys and preloaded.nbytes() == main.nbytes()
    for a, b in zip(main, preloaded, strict=True):  # same batches, preloaded or mapped
        assert all(torch.equal(a[k], b[k]) for k in a)


# ── A: the per-epoch bootstrap changes nothing that is kept ──────────────────


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
def test_training_without_the_bootstrap_keeps_the_same_checkpoint(task, tmp_path, monkeypatch):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    best = []
    for iters in (0, 5):
        trainer = _cached_trainer(tmp_path / f"b{iters}", task)
        trainer.args.bootstrap_iters = iters
        trainer.train()
        best.append(torch.load(trainer.checkpoint_path(), weights_only=False))
    a, b = best
    assert (a["epoch"], a["best_head"], a["best_auroc"]) == (b["epoch"], b["best_head"],
                                                             b["best_auroc"])  # fmt: skip
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])


# ── A: fast scoring gives validate()'s point metrics ─────────────────────────


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
def test_head_metrics_match_validate(task, tmp_path, monkeypatch):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    trainer = _cached_trainer(tmp_path, task)
    trainer.args.bootstrap_iters = 0
    trainer.model.eval()
    with torch.no_grad():
        rets = dict(zip(trainer.HEADS, trainer.validate(trainer.val_dl, False)[:5], strict=True))
        labels, preds = round3.predict_heads(trainer, trainer.val_dl)
    target = 21 if task == "phenotyping" else 0
    for head in trainer.HEADS:
        m = round3.head_metrics(labels, preds[head], target, 20)
        assert m["macro_auroc"] == pytest.approx(rets[head]["auroc_mean"], abs=1e-12)
        assert m["macro_auprc"] == pytest.approx(rets[head]["auprc_mean"], abs=1e-12)
        expected = np.ravel(rets[head]["auc_scores"])[target]
        if not np.isnan(expected):
            assert m["target_auroc"] == pytest.approx(expected, abs=1e-12)


# ── B6: every step logs fully and ends with one OK line ──────────────────────


def test_each_step_writes_its_log_and_an_ok_line(lineage, monkeypatch, capsys):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping")
    run = round3.plan(o)
    monkeypatch.setattr(round3, "step_train", lambda run_, o_: print("training output"))
    round3.run_step(run, o, "train")
    out = capsys.readouterr().out
    assert any(line.startswith("[r3] train OK (") for line in out.splitlines())
    log = (Path(run["save_dir"]) / "train.log").read_text(encoding="utf-8")
    assert "training output" in log and "[r3] train OK (" in log

    def boom(run_, o_):
        raise SystemExit("[r3] equivalence FAILED")

    monkeypatch.setattr(round3, "step_equivalence", boom)
    with pytest.raises(SystemExit):
        round3.run_step(run, o, "equivalence")
    log = (Path(run["save_dir"]) / "equivalence.log").read_text(encoding="utf-8")
    assert "equivalence FAILED" in log and "[r3] equivalence OK" not in log


# ── B5: share carries the extra tables, still only the main checkpoint ───────


def test_share_includes_the_extra_tables(lineage):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping", "--share-to", str(tmp_path / "shared"))
    run = round3.plan(o)
    save_dir = Path(run["save_dir"])
    best = save_dir / "phenotyping" / "c-msma" / "best.pth.tar"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"main")
    manifest.append_row({"id": run["id"], "stage": "r3", "reader": "fusion",
                         "task": "phenotyping", "data": "virtual", "file_path": best.as_posix(),
                         "sha256": manifest.sha256_file(best)})  # fmt: skip
    (save_dir / "r3_results.csv").write_text("main")
    round3.step_share(run, o)
    assert not (tmp_path / "shared" / "r3_ablations.csv").exists()
    for name in ("r3_ablations.csv", "r3_threshold_sweep.csv"):
        (save_dir / name).write_text(name)
    (save_dir / "theta_0.65").mkdir()
    (save_dir / "theta_0.65" / "best.pth.tar").write_bytes(b"variant")
    round3.step_share(run, o)  # rerun after the extras: the tables follow
    shared = sorted(p.name for p in (tmp_path / "shared").iterdir())
    assert shared == ["best.pth.tar", "r3_ablations.csv", "r3_results.csv",
                      "r3_threshold_sweep.csv"]  # fmt: skip
    assert (tmp_path / "shared" / "best.pth.tar").read_bytes() == b"main"
