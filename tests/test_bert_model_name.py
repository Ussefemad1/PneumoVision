"""--bert_model_name: medpatch flag, manifest lineage, import and run rules.

Non-training, and no model download: from_pretrained is stubbed. Real-data
paths here are placeholders; no data file is opened.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.pv import manifest, run
from tools.pv.import_checkpoint import resolve_bert_model_name as import_bert
from tools.pv.medpatch_bridge import parse_args

BIOBERT = "dmis-lab/biobert-v1.1"


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    monkeypatch.setenv("PV_MANIFEST", str(path))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    return path


def _row(tmp_path, stage, reader="rr", data="real", bert="", parent_id=""):
    f = tmp_path / f"{manifest.next_id(stage, reader)}.pth.tar"
    f.write_bytes(f"{stage}-{reader}-{bert}".encode())
    return manifest.append_row(
        {
            "id": manifest.next_id(stage, reader),
            "stage": stage,
            "reader": reader,
            "data": data,
            "file_path": f.as_posix(),
            "sha256": manifest.sha256_file(f),
            "parent_id": parent_id,
            "bert_model_name": bert,
        }
    )


# ── medpatch ─────────────────────────────────────────────────────────────────


def test_flag_defaults_to_the_paper_bert():
    assert parse_args([]).bert_model_name == "emilyalsentzer/Bio_ClinicalBERT"
    assert parse_args(["--bert_model_name", BIOBERT]).bert_model_name == BIOBERT


def test_text_encoder_uses_the_flag_for_model_and_tokenizer(monkeypatch):
    from models import text_models  # noqa: PLC0415

    loaded = {}

    class FakeBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = argparse.Namespace(hidden_size=8)
            self.w = nn.Linear(1, 1)

    def fake_model(name):
        loaded["model"] = name
        return FakeBert()

    def fake_tokenizer(name):
        loaded["tokenizer"] = name
        return object()

    monkeypatch.setattr(text_models.BertModel, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(
        text_models.BertTokenizerFast, "from_pretrained", staticmethod(fake_tokenizer)
    )
    args = parse_args(
        ["--modalities", "EHR-RR", "--fusion_type", "c-unimodal_rr", "--bert_model_name", BIOBERT]
    )
    text_models.Text_encoder(args, torch.device("cpu"))
    assert loaded == {"model": BIOBERT, "tokenizer": BIOBERT}


# ── manifest ─────────────────────────────────────────────────────────────────


def test_old_manifests_are_upgraded_on_append(tmp_manifest, tmp_path):
    old = manifest.COLUMNS[:-1]  # the header before bert_model_name existed
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow(
            {c: "" for c in old}
            | {"id": "r1-rr-001", "stage": "r1", "reader": "rr", "data": "virtual"}
        )
    assert manifest.read_rows()[0]["bert_model_name"] == ""
    _row(tmp_path, "r1", bert=BIOBERT)
    with open(tmp_manifest, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == manifest.COLUMNS
    rows = manifest.read_rows()
    assert [r["id"] for r in rows] == ["r1-rr-001", "r1-rr-002"]
    assert rows[1]["bert_model_name"] == BIOBERT


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"reader": "rr", "data": "virtual", "bert_model_name": ""}, manifest.DEFAULT_BERT),
        ({"reader": "dn", "data": "real", "bert_model_name": ""}, ""),
        ({"reader": "dn", "data": "real", "bert_model_name": BIOBERT}, BIOBERT),
        ({"reader": "cxr", "data": "real", "bert_model_name": ""}, ""),
    ],
)
def test_bert_of_a_row(row, expected):
    assert manifest.bert_model_name_of(row) == expected


# ── import_checkpoint ────────────────────────────────────────────────────────


@pytest.mark.parametrize("reader", ["rr", "dn"])
def test_import_requires_the_name_for_real_text_readers(reader):
    with pytest.raises(SystemExit, match="required"):
        import_bert(reader, "real", None)
    assert import_bert(reader, "real", BIOBERT) == BIOBERT


def test_import_defaults_only_for_virtual_and_rejects_it_for_ehr_cxr():
    assert import_bert("rr", "virtual", None) == manifest.DEFAULT_BERT
    assert import_bert("cxr", "real", None) == ""
    with pytest.raises(SystemExit, match="only for rr/dn"):
        import_bert("ehr", "real", BIOBERT)


# ── run.py ───────────────────────────────────────────────────────────────────


def _real_args(tmp_path, stage="r2", reader="rr", **kw):
    defaults = dict(
        stage=stage,
        reader=reader,
        data="real",
        parent=None,
        run_id=None,
        preset="smoke",
        data_root=None,
        epochs=None,
        batch_size=None,
        bootstrap_iters=None,
        num_workers=None,
        bert_model_name=None,
        ehr_data_dir="/content/data/ehr",
        cxr_data_dir="/content/data/cxr",
        notes_data_dir="/content/data/notes",
    )
    return argparse.Namespace(**{**defaults, **kw})


@pytest.mark.parametrize("stage, parent_stage", [("r2", "r1"), ("r2b", "r2")])
def test_r2_and_r2b_inherit_the_parent_bert(tmp_manifest, tmp_path, stage, parent_stage):
    _row(tmp_path, parent_stage, bert=BIOBERT)
    plan = run.plan(_real_args(tmp_path, stage=stage))
    argv = plan["argv"]
    assert argv[argv.index("--bert_model_name") + 1] == BIOBERT
    assert plan["bert_model_name"] == BIOBERT


def test_explicit_mismatch_is_refused(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", bert=BIOBERT)
    with pytest.raises(manifest.ManifestError, match="differs from parent"):
        run.plan(_real_args(tmp_path, bert_model_name=manifest.DEFAULT_BERT))


def test_explicit_match_is_accepted(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", bert=BIOBERT)
    assert run.plan(_real_args(tmp_path, bert_model_name=BIOBERT))["bert_model_name"] == BIOBERT


def test_real_text_parent_without_a_name_is_refused(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", bert="")
    with pytest.raises(manifest.ManifestError, match="no bert_model_name recorded"):
        run.plan(_real_args(tmp_path))


def test_ehr_and_cxr_get_no_bert_and_refuse_the_flag(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", reader="cxr")
    plan = run.plan(_real_args(tmp_path, reader="cxr"))
    assert "--bert_model_name" not in plan["argv"] and plan["bert_model_name"] == ""
    with pytest.raises(manifest.ManifestError, match="only for rr/dn"):
        run.plan(_real_args(tmp_path, reader="cxr", bert_model_name=BIOBERT))


def test_virtual_legacy_text_parent_keeps_the_default(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", data="virtual", bert="")
    data_root = tmp_path / "virtual"
    data_root.mkdir()
    (data_root / "README.md").write_text("SYNTHETIC")
    plan = run.plan(_real_args(tmp_path, data="virtual", data_root=str(data_root)))
    assert plan["bert_model_name"] == manifest.DEFAULT_BERT


def test_dry_run_prints_command_and_parent_check_without_training(tmp_manifest, tmp_path, capsys):
    r1 = _row(tmp_path, "r1", bert=BIOBERT)
    result = run.main(
        [
            "r2",
            "--reader",
            "rr",
            "--data",
            "real",
            "--ehr-data-dir",
            "/content/data/ehr",
            "--cxr-data-dir",
            "/content/data/cxr",
            "--notes-data-dir",
            "/content/data/notes",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert result is None
    assert f"parent        : {r1['id']}" in out and "matches the file: OK" in out
    assert f"--bert_model_name {BIOBERT}" in out
    assert "--load_rr" in out and "--fusion_type c-unimodal_rr" in out
    assert not (tmp_path / "runs").exists(), "a dry run must not create the run folder"
    assert Path(tmp_manifest).read_text().count("\n") == 2  # header + the r1 row only
