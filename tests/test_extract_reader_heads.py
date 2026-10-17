"""tools/pv/extract_reader_heads.py on fake Round 1 readers (synthetic, random weights).

The fake readers share one tiny encoder set and are written in the readers
layout; the tool must find all seven, write reader_heads.pt + .json, record
each source's sha256, and refuse a folder with a reader missing.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


def _fake_reader_dir(tmp_path):
    """Seven state_dicts named and shaped like the readers', minus the encoders' bulk."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))
    from app.readers import READERS

    prefixes = {"ehr": "ehr_model.", "cxr": "cxr_model.", "rr": "text_model.bert.", "dn": "text_model.bert."}
    for spec in READERS:
        classes = 25 if spec.task == "pneumonia" else 1
        state = {
            f"{prefixes[spec.modality]}layer.weight": torch.ones(4, 4),
            f"fusion_model.{spec.modality}_classifier.classifier.fc.weight": torch.randn(classes, 4),
            f"fusion_model.{spec.modality}_classifier.classifier.fc.bias": torch.randn(classes),
        }
        path = tmp_path / spec.checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state}, path)
    return tmp_path, READERS


def test_extract_writes_heads_with_provenance(tmp_path):
    from tools.pv.extract_reader_heads import main

    readers_dir, readers = _fake_reader_dir(tmp_path / "r1")
    out = tmp_path / "round3"
    assert main(["--checkpoint-dir", str(readers_dir), "--out", str(out)]) == 0
    manifest = json.loads((out / "reader_heads.json").read_text())
    assert {s["reader"] for s in manifest["sources"]} == {s.key for s in readers}
    assert all(len(s["sha256"]) == 64 for s in manifest["sources"])

    from app import reader_heads

    payload = reader_heads.load(out)
    entry = payload["readers"]["dn-pneumonia"]
    assert set(entry["head"]) == {"classifier.fc.weight", "classifier.fc.bias"}
    assert entry["output_index"] == 21 and entry["modality"] == "dn"
    assert list(entry["encoder_digests"]) == ["text_model.bert.layer.weight"]
    assert payload["readers"]["ehr-mortality"]["normalizer"].startswith("ihm_")


def test_extract_refuses_a_missing_reader(tmp_path, capsys):
    from tools.pv.extract_reader_heads import main

    readers_dir, readers = _fake_reader_dir(tmp_path / "r1")
    (readers_dir / readers[0].checkpoint).unlink()
    assert main(["--checkpoint-dir", str(readers_dir), "--out", str(tmp_path / "out")]) == 1
    assert "MISSING" in capsys.readouterr().out
    assert not (tmp_path / "out" / "reader_heads.pt").exists()
