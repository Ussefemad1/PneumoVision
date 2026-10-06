"""--cache_frozen_logits on the real trainer and virtual data: cached == uncached.

    python -m pytest -m round2b_cache -s                       # phenotyping rr and dn
    PV_CHECK_TASK=in-hospital-mortality python -m pytest -m round2b_cache -s   # mortality rr

TRAINS (two 2-epoch Round 2b runs per reader, BERT forward passes), so it is
deselected by default: run it on Colab, not on a machine reserved for
non-training checks. It needs a virtual r2 row per reader in the manifest
(``python -m tools.pv.smoke_round2`` first).

Each reader is calibrated twice from the same r2 parent through tools.pv.run --
once with --no-cache-frozen-logits (the released loop: BERT over val every
epoch) and once with the cache -- into a scratch manifest and RUNS_ROOT, so the
team manifest is never written. Then the two runs must match: temperatures
(atol 1e-6), best epoch, checkpoint file names and every ECE table.

On virtual data every number here is SYNTHETIC -- not a scientific result.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from tools.pv import manifest, run

pytestmark = pytest.mark.round2b_cache

TASK = os.environ.get("PV_CHECK_TASK") or "phenotyping"
READERS = ["rr"] if TASK == "in-hospital-mortality" else ["rr", "dn"]
EPOCHS = "2"


@pytest.fixture
def scratch_manifest(tmp_path, monkeypatch):
    source = manifest.manifest_path()
    if not source.is_file():
        pytest.skip(f"no manifest at {source}; run python -m tools.pv.smoke_round2 first")
    copy = tmp_path / "manifest.csv"
    shutil.copy2(source, copy)
    monkeypatch.setenv("PV_MANIFEST", str(copy))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    return copy


def calibrate(reader: str, cache: bool) -> tuple[Path, Path]:
    run_id = f"r2b-{reader}-{'cache' if cache else 'plain'}"
    row = run.main(
        [
            "r2b",
            "--reader",
            reader,
            "--data",
            "virtual",
            "--task",
            TASK,
            "--epochs",
            EPOCHS,
            "--run-id",
            run_id,
            "--who",
            "round2b_cache",
            "--cache-frozen-logits" if cache else "--no-cache-frozen-logits",
        ]
    )
    best = Path(row["file_path"])
    return best, best.parents[2] / "medpatch_outputs"


@pytest.mark.parametrize("reader", READERS)
def test_cached_run_equals_the_released_loop(scratch_manifest, reader):
    if manifest.latest("r2", reader, "virtual", task=TASK) is None:
        pytest.skip(f"no virtual r2 row for {reader} ({TASK})")
    plain_best, plain_out = calibrate(reader, cache=False)
    cached_best, cached_out = calibrate(reader, cache=True)

    assert cached_best.name == plain_best.name
    a = torch.load(plain_best, map_location="cpu", weights_only=False)
    b = torch.load(cached_best, map_location="cpu", weights_only=False)
    assert a["epoch"] == b["epoch"], "different best epoch"
    key = f"fusion_model.{reader}_temperature"
    diff = (a["state_dict"][key] - b["state_dict"][key]).abs().max().item()
    print(f"\n{reader}: best epoch {a['epoch']}, max |temperature diff| {diff:.2e}")
    torch.testing.assert_close(b["state_dict"][key], a["state_dict"][key], atol=1e-6, rtol=0)
    for name in a["state_dict"]:
        if name != key:
            assert torch.equal(a["state_dict"][name], b["state_dict"][name]), name

    plain_tables = sorted(p.name for p in plain_out.glob("*.csv"))
    assert plain_tables == sorted(p.name for p in cached_out.glob("*.csv"))
    for name in plain_tables:
        pd.testing.assert_frame_equal(
            pd.read_csv(cached_out / name, index_col=0),
            pd.read_csv(plain_out / name, index_col=0),
            atol=1e-6,
            rtol=0,
        )
