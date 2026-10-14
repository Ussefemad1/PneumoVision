"""tools.pv.parent_keys: which reader prefixes a Round 2b file carries (F2 report).

Synthetic parent files built around the stub readers; counts only.
"""

from __future__ import annotations

import torch
from r3_stubs import D, r3_model
from test_round3_fixes import _parents


def test_parent_keys_reports_taken_and_ignored_prefixes(tmp_path, monkeypatch, capsys):
    from tools.pv import manifest, parent_keys  # noqa: PLC0415

    monkeypatch.setenv("PV_MANIFEST", str(tmp_path / "manifest.csv"))
    _, model = r3_model("in-hospital-mortality")
    paths = _parents(model, tmp_path, "in-hospital-mortality")
    rr = torch.load(paths["rr"])
    rr["state_dict"]["ehr_model.lstm.weight"] = torch.zeros(D, 76)  # overlap
    torch.save(rr, paths["rr"])
    for reader, path in paths.items():
        manifest.append_row(
            {
                "id": f"r2b-{reader}-001",
                "stage": "r2b",
                "reader": reader,
                "task": "in-hospital-mortality",
                "data": "real",
                "file_path": path.as_posix(),
                "sha256": manifest.sha256_file(path),
            }
        )
    assert parent_keys.main(["--task", "mortality", "--trust-pickle"]) == 0
    out = capsys.readouterr().out
    assert "r2b-rr-001  sha256 matches the manifest" in out
    assert "IGNORED       1  ehr_model" in out
    assert "taken" in out and "fusion_model.rr_classifier" in out
