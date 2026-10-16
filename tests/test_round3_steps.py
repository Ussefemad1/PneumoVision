"""run r3: exit codes, the cache copy, requested variants, --step ablations.

No model is built: the training / scoring functions are replaced by stand-ins,
so these check the plumbing around them. SYNTHETIC files only.
"""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

import pytest
from test_round3_tools import lineage, opts  # noqa: F401 -- pytest fixture

from tools.pv import manifest, round3


def _argv(tmp_path, *extra):
    return [
        "--task",
        "phenotyping",
        "--data",
        "virtual",
        "--data-root",
        str(tmp_path / "smoke-r3"),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]


# ── 1. exit codes ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("step", ["cache", "equivalence", "train", "ablations"])
def test_successful_steps_exit_zero(lineage, monkeypatch, step):  # noqa: F811
    tmp_path, _ = lineage
    called = []
    for name in ("step_cache", "step_equivalence", "step_train", "step_ablations"):
        monkeypatch.setattr(round3, name, lambda run, o, _n=name: called.append(_n))
    assert round3.cli(_argv(tmp_path, "--step", step, "--run-id", "r3-fusion-001")) == 0
    assert called == [f"step_{step}"]


def test_run_py_dispatches_r3_and_exits_zero(lineage, monkeypatch):  # noqa: F811
    tmp_path, _ = lineage
    monkeypatch.setattr(round3, "step_train", lambda run, o: None)  # success, returns None
    monkeypatch.setattr(sys, "argv", ["run.py", "r3", *_argv(tmp_path, "--step", "train")])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("tools.pv.run", run_name="__main__")
    assert exit_.value.code == 0


def test_a_failing_step_exits_non_zero(lineage, monkeypatch):  # noqa: F811
    tmp_path, _ = lineage

    def boom(run, o):
        raise SystemExit("[r3] something real went wrong")

    monkeypatch.setattr(round3, "step_train", boom)
    monkeypatch.setattr(sys, "argv", ["run.py", "r3", *_argv(tmp_path, "--step", "train")])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("tools.pv.run", run_name="__main__")
    assert exit_.value.code not in (0, None)


# ── 2. cache copy ────────────────────────────────────────────────────────────


def _cache(tmp_path):
    src = tmp_path / "src" / "phenotyping"
    for rel in ("meta.json", "val/shard_00000.pt", "val/shard_00001.pt", "train/shard_00000.pt"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(rel.encode() * 10)
    return src


def test_copy_uses_copyfile_only_and_counts_both_sides(tmp_path, monkeypatch, capsys):
    src = _cache(tmp_path)

    def no_copystat(*a, **k):
        raise PermissionError("FUSE: operation not permitted")

    monkeypatch.setattr(shutil, "copystat", no_copystat)  # as on the Drive mount
    monkeypatch.setattr(shutil, "copy2", no_copystat)
    target = tmp_path / "drive" / "phenotyping"
    assert round3.copy_cache(src, target) is True
    out = capsys.readouterr().out
    assert "source 4 files, target 4 files, 4 identical in size; 4 copied now, 0 failed" in out
    assert round3.copy_cache(src, target) is True  # rerun: nothing to copy
    assert "0 copied now" in capsys.readouterr().out


def test_copy_errors_are_reported_not_raised(tmp_path, monkeypatch, capsys):
    src = _cache(tmp_path)
    real = shutil.copyfile

    def flaky(a, b, *k, **kw):
        if str(a).endswith("shard_00001.pt"):
            raise OSError(5, "Input/output error")
        return real(a, b)

    monkeypatch.setattr(shutil, "copyfile", flaky)
    assert round3.copy_cache(src, tmp_path / "drive" / "phenotyping") is False
    out = capsys.readouterr().out
    assert "WARNING could not copy val" in out and "3 copied now, 1 failed" in out


# ── 3. requested variants ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "flags, expected",
    [
        ((), ["main", round3.MAIN_ABLATION]),
        (("--no-ablation",), ["main"]),
        (("--extra-ablations",), ["main", round3.MAIN_ABLATION, *round3.EXTRA_ABLATIONS]),
    ],
)
def test_train_records_exactly_the_requested_variants(lineage, monkeypatch, flags, expected):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping", *flags)
    run = round3.plan(o)
    monkeypatch.setattr(round3, "train_variant", lambda run_, name, args, o_: f"{name}.pth.tar")
    round3.step_train(run, o)
    assert run["requested_variants"] == expected
    assert list(run["trained"]) == expected


def test_variant_args_set_flags_and_folders(lineage):  # noqa: F811
    tmp_path, _ = lineage
    run = round3.plan(opts(tmp_path, "phenotyping"))
    main = round3.variant_args(run, "main")
    assert main.save_dir == run["save_dir"] and main.ablation is None
    late = round3.variant_args(run, "late_weights_per_class")
    assert late.late_weights_per_class is True and late.ablation is None
    assert Path(late.save_dir) == Path(run["save_dir"]) / "late_weights_per_class"
    joint = round3.variant_args(run, "without_joint_module")
    assert joint.ablation == "without_joint_module" and not joint.mask_missing_tokens


# ── 4. ablations after the row exists ────────────────────────────────────────


def _m(x):
    return {
        "macro_auroc": x,
        "macro_auprc": x / 2,
        "target_auroc": x,
        "target_auprc": x / 3,
        "target_auroc_ci": [x - 0.01, x + 0.01],
        "target_auprc_ci": [x / 4, x / 2],
    }


def test_ablations_step_scores_variants_and_leaves_the_manifest(lineage, monkeypatch):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping", "--step", "ablations")
    run = round3.plan(o)
    run["scores"] = {"main": {"head": "late", "val": _m(0.86), "test": _m(0.865)}}
    manifest.append_row(
        {
            "id": run["id"],
            "stage": "r3",
            "reader": "fusion",
            "task": "phenotyping",
            "data": "virtual",
            "notes": "main",
        }
    )
    before = manifest.manifest_path().read_bytes()
    trained = []
    monkeypatch.setattr(
        round3, "train_variant", lambda run_, name, args, o_: trained.append((name, args)) or name
    )
    monkeypatch.setattr(
        round3,
        "score_variant",
        lambda run_, args, o_: {"head": "high", "val": _m(0.8), "test": _m(0.81)},
    )
    path = round3.step_ablations(run, o)
    assert [n for n, _ in trained] == list(round3.DEFAULT_ABLATION_STEP)
    assert dict(trained)["mask_missing_tokens"].mask_missing_tokens is True
    assert manifest.manifest_path().read_bytes() == before  # row untouched
    lines = path.read_text().splitlines()
    assert path.name == "r3_ablations.csv" and "the manifest row is the main run" in lines[0]
    assert [ln.split(",")[0] for ln in lines[2:]] == ["main (recorded)"] * 2 + [
        n for n in round3.DEFAULT_ABLATION_STEP for _ in (0, 1)
    ]
    assert run["requested_ablations"] == sorted(round3.DEFAULT_ABLATION_STEP)


def test_ablations_step_can_pick_variants(lineage, monkeypatch):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping", "--variants", "late_weights_per_class")
    run = round3.plan(o)
    monkeypatch.setattr(round3, "train_variant", lambda run_, name, args, o_: name)
    monkeypatch.setattr(
        round3,
        "score_variant",
        lambda run_, args, o_: {"head": "late", "val": _m(0.7), "test": _m(0.7)},
    )
    round3.step_ablations(run, o)
    assert list(run["ablation_scores"]) == ["late_weights_per_class"]
