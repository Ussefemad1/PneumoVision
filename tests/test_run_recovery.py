"""tools/pv/run.py: stray outputs move across filesystems; --record-only. No training."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import pytest

from tools.pv import manifest, run
from tools.pv.evaluate import Scores, checkpoint_path
from tools.pv.medpatch_bridge import parse_args


def _exdev(*_args, **_kwargs):
    raise OSError(errno.EXDEV, "Invalid cross-device link")


@pytest.fixture
def fake_medpatch(tmp_path, monkeypatch):
    folder = tmp_path / "medpatch"
    folder.mkdir()
    monkeypatch.setattr(run, "MEDPATCH", folder)
    return folder


# ── 1. cross-device move ─────────────────────────────────────────────────────


def test_collect_moves_files_when_rename_is_cross_device(fake_medpatch, tmp_path, monkeypatch):
    (fake_medpatch / "fusion_main.py").write_text("source")
    before = {p.name for p in fake_medpatch.iterdir()}
    for c in range(3):
        (fake_medpatch / f"calibration_curve_class_{c}.png").write_bytes(b"png")
    monkeypatch.setattr(os, "replace", _exdev)
    monkeypatch.setattr(os, "rename", _exdev)  # what shutil.move tries first
    monkeypatch.setattr(Path, "replace", _exdev)
    monkeypatch.setattr(Path, "rename", _exdev)

    dest = tmp_path / "drive" / "medpatch_outputs"
    run._collect_stray_outputs(before, dest)

    assert sorted(p.name for p in dest.iterdir()) == [
        f"calibration_curve_class_{c}.png" for c in range(3)
    ]
    assert [p.name for p in fake_medpatch.iterdir()] == ["fusion_main.py"]


def test_collect_falls_back_to_copy_and_never_raises(fake_medpatch, tmp_path, monkeypatch, capsys):
    before: set[str] = set()
    (fake_medpatch / "a_ece_table_epoch_1.csv").write_text("x")
    (fake_medpatch / "stuck.csv").write_text("y")
    monkeypatch.setattr(shutil, "move", _exdev)
    real_copy2 = shutil.copy2

    def copy2(src, dst, **kw):
        if Path(src).name == "stuck.csv":
            raise PermissionError(errno.EACCES, "denied")
        return real_copy2(src, dst, **kw)

    monkeypatch.setattr(shutil, "copy2", copy2)
    dest = tmp_path / "out"
    run._collect_stray_outputs(before, dest)  # must not raise

    out = capsys.readouterr().out
    assert (dest / "a_ece_table_epoch_1.csv").is_file()
    assert not (fake_medpatch / "a_ece_table_epoch_1.csv").exists()
    assert "WARNING could not move stuck.csv" in out
    assert "moved 1 file(s)" in out


def test_execute_keeps_the_training_exit_code_if_collecting_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_run_fusion_main", lambda *a: 3)

    def boom(*_a):
        raise RuntimeError("collect failed")

    monkeypatch.setattr(run, "_collect_stray_outputs", boom)
    plan = {"id": "r2-cxr-001", "script": "s", "save_dir": str(tmp_path / "r"), "argv": []}
    with pytest.raises(SystemExit, match="exit 3"):
        run.execute(plan)


# ── 2. --record-only ─────────────────────────────────────────────────────────


@pytest.fixture
def virtual_r1(tmp_path, monkeypatch, fake_medpatch):
    monkeypatch.setenv("PV_MANIFEST", str(tmp_path / "manifest.csv"))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "virtual"
    root.mkdir()
    (root / "README.md").write_text("SYNTHETIC")
    parent = tmp_path / "r1-cxr.pth.tar"
    parent.write_bytes(b"r1")
    manifest.append_row(
        {
            "id": "r1-cxr-001",
            "stage": "r1",
            "reader": "cxr",
            "data": "virtual",
            "file_path": parent.as_posix(),
            "sha256": manifest.sha256_file(parent),
        }
    )
    monkeypatch.setattr(
        run,
        "score_checkpoint",
        lambda *a, **k: Scores(None, None, auroc=0.61, auprc=0.42, confidence=None),
    )
    argv = ["r2", "--reader", "cxr", "--data", "virtual", "--data-root", str(root)]
    return argv + ["--run-id", "r2-cxr-001", "--who", "test"]


def _best(argv):
    plan = run.plan(run.build_parser().parse_args(argv))
    return checkpoint_path(parse_args(plan["argv"])), Path(plan["save_dir"])


def test_record_only_records_once_then_refuses(virtual_r1, fake_medpatch, monkeypatch):
    monkeypatch.setattr(run, "_run_fusion_main", lambda *a: pytest.fail("trained"))
    best, save_dir = _best(virtual_r1)
    best.parent.mkdir(parents=True)
    best.write_bytes(b"finished r2")
    (save_dir / "run.json").write_text("{}")
    (fake_medpatch / "calibration_curve_class_0.png").write_bytes(b"png")

    row = run.main([*virtual_r1, "--record-only"])
    assert row["id"] == "r2-cxr-001" and row["parent_id"] == "r1-cxr-001"
    assert row["sha256"] == manifest.sha256_file(best)
    assert row["val_auroc"] == "0.6100" and "record-only" in row["notes"]
    assert (save_dir / "medpatch_outputs" / "calibration_curve_class_0.png").is_file()

    with pytest.raises(SystemExit, match="already in"):
        run.main([*virtual_r1, "--record-only"])


def test_record_only_refuses_a_missing_checkpoint(virtual_r1):
    with pytest.raises(SystemExit, match="no best checkpoint"):
        run.main([*virtual_r1, "--record-only"])
    assert manifest.find("r2-cxr-001") is None


def test_record_only_needs_a_run_id(virtual_r1):
    argv = [a for a in virtual_r1 if a not in ("--run-id", "r2-cxr-001")]
    with pytest.raises(SystemExit):
        run.main([*argv, "--record-only"])


def test_record_only_dry_run_is_unchanged(virtual_r1, capsys):
    assert run.main([*virtual_r1, "--record-only", "--dry-run"]) is None
    assert "[dry-run] r2-cxr-001" in capsys.readouterr().out
    assert manifest.find("r2-cxr-001") is None
