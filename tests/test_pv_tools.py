"""Unit tests for the run tooling (tools/pv). No models, no data."""

from __future__ import annotations

import argparse

import pytest
import torch

from tools.pv import manifest, run
from tools.pv.import_checkpoint import check_round1
from tools.pv.paper_scripts import SCRIPT_FOR, build_argv, read_script_args, script_settings


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    monkeypatch.setenv("PV_MANIFEST", str(path))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    return path


# ── paper scripts ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["r1", "r2"])
@pytest.mark.parametrize("reader", ["ehr", "cxr", "rr", "dn"])
def test_every_paper_script_parses(stage, reader):
    settings = script_settings(stage, reader)
    assert settings["--task"] == "phenotyping"
    assert settings["--num_classes"] == "25"
    expected = f"{'c-' if stage == 'r2' else ''}unimodal_{reader}"
    assert settings["--fusion_type"] == expected


def test_overrides_keep_every_paper_setting():
    argv = build_argv("r2", "cxr", {"--save_dir": "X", "--load_cxr": "P", "--resume": None})
    paper = read_script_args(SCRIPT_FOR["r2"]("cxr"))
    for flag in (
        "--lr",
        "--epochs",
        "--batch_size",
        "--cxr_encoder",
        "--use_cls_token",
        "--output_dim_cxr",
        "--data_pairs",
        "--classifier",
        "--loss",
    ):
        assert argv[argv.index(flag) + 1] == paper[paper.index(flag) + 1]
    assert argv[argv.index("--save_dir") + 1] == "X"
    assert argv.count("--save_dir") == 1 and "--resume" in argv


def test_dn_loads_with_load_dn_not_the_scripts_load_rr():
    assert script_settings("r2", "dn").get("--load_rr") is not None  # the upstream slip
    argv = build_argv("r2", "dn", {"--load_dn": "parent.pth.tar"})
    assert "--load_rr" not in argv
    assert argv[argv.index("--load_dn") + 1] == "parent.pth.tar"


# ── manifest lineage ─────────────────────────────────────────────────────────


def _r1(tmp_path, reader="cxr", data="virtual", stage="r1"):
    f = tmp_path / f"{reader}-{stage}.pth.tar"
    f.write_bytes(b"checkpoint-bytes")
    return manifest.append_row(
        {
            "id": manifest.next_id(stage, reader),
            "stage": stage,
            "reader": reader,
            "data": data,
            "file_path": f.as_posix(),
            "sha256": manifest.sha256_file(f),
        }
    )


def test_verify_parent_accepts_a_matching_r1(tmp_manifest, tmp_path):
    row = _r1(tmp_path)
    assert manifest.verify_parent(row, "cxr", "virtual").is_file()


@pytest.mark.parametrize(
    "case, message",
    [
        ("missing", "No r1 row"),
        ("stage", "not r1"),
        ("reader", "checkpoint; this run is"),
        ("data", "Mixing virtual and real"),
        ("sha", "sha256 mismatch"),
        ("file", "file is missing"),
    ],
)
def test_verify_parent_refuses(tmp_manifest, tmp_path, case, message):
    row = (
        None
        if case == "missing"
        else _r1(
            tmp_path,
            reader="ehr" if case == "reader" else "cxr",
            stage="r2" if case == "stage" else "r1",
        )
    )
    if case == "sha":
        with open(row["file_path"], "ab") as f:
            f.write(b"tampered")
    if case == "file":
        import os

        os.remove(row["file_path"])
    data = "real" if case == "data" else "virtual"
    with pytest.raises(manifest.ManifestError, match=message):
        manifest.verify_parent(row, "cxr", data)


def test_run_plan_refuses_a_tampered_parent(tmp_manifest, tmp_path):
    row = _r1(tmp_path)
    with open(row["file_path"], "ab") as f:
        f.write(b"x")
    args = argparse.Namespace(
        stage="r2",
        reader="cxr",
        data="virtual",
        parent=None,
        run_id=None,
        preset="smoke",
        data_root=None,
        epochs=None,
        batch_size=None,
        bootstrap_iters=None,
        num_workers=None,
    )
    with pytest.raises(manifest.ManifestError, match="sha256 mismatch"):
        run.plan(args)


def test_ids_increment_per_stage_and_reader(tmp_manifest, tmp_path):
    assert _r1(tmp_path)["id"] == "r1-cxr-001"
    assert _r1(tmp_path)["id"] == "r1-cxr-002"
    assert _r1(tmp_path, reader="ehr")["id"] == "r1-ehr-001"


def test_manifest_rejects_unknown_values(tmp_manifest):
    with pytest.raises(ValueError):
        manifest.append_row({"id": "x", "stage": "r3", "data": "virtual"})
    with pytest.raises(ValueError):
        manifest.append_row({"id": "x", "stage": "r1", "data": "mimic"})


# ── import_checkpoint ────────────────────────────────────────────────────────


def _ckpt(*keys):
    return {"epoch": 3, "state_dict": {k: torch.zeros(1) for k in keys}}


def test_import_accepts_a_round1_file():
    good = _ckpt("cxr_model.feature_extractor.x", "fusion_model.cxr_classifier.w")
    assert check_round1(good, "cxr") == []


def test_import_rejects_rr_file_offered_as_dn():
    rr = _ckpt("text_model.bert.x", "text_model.fc_rr.w", "fusion_model.rr_classifier.w")
    assert check_round1(rr, "dn")


def test_import_rejects_a_round2_file():
    r2 = _ckpt(
        "ehr_model.l",
        "fusion_model.ehr_classifier.w",
        "fusion_model.ehr_confidence_predictor.confidence_layer.weight",
    )
    assert any("Round 2" in p for p in check_round1(r2, "ehr"))


def test_import_rejects_non_checkpoints():
    assert check_round1({"weights": 1}, "ehr")


# ── normalizer resolution (real runs pass no --normalizer_state) ──────────────


def test_bridge_resolves_the_default_normalizer_like_fusion_main():
    from tools.pv.medpatch_bridge import MEDPATCH, normalizer_state_path, parse_args

    real = parse_args(build_argv("r2", "ehr", {"--ehr_data_dir": "/content/data/ehr"}))
    assert real.normalizer_state is None
    expected = MEDPATCH / "normalizers" / "ph_ts1.0.input_str_previous.start_time_zero.normalizer"
    assert normalizer_state_path(real) == expected
    assert expected.is_file()  # the bundled code resource fusion_main.py loads

    virtual = parse_args(["--normalizer_state", "/tmp/virtual.normalizer"])
    assert str(normalizer_state_path(virtual)).replace("\\", "/") == "/tmp/virtual.normalizer"
