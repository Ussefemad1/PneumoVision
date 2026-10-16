"""Round 3 acceptance checks on the latest r3 run (python -m pytest -m round3 -s).

Run by ``python -m tools.pv.smoke_round3`` after a virtual Round 3, or on
Colab after a real one (set PV_CHECK_TASK). Builds medpatch's model, so it
needs the readers' pretrained weights (BERT, ViT): Colab, not a laptop. On
virtual data every number is SYNTHETIC -- not a scientific result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from tools.pv import manifest, r3_cache, round3
from tools.pv.paper_scripts import READERS_FOR_TASK, canonical_task

pytestmark = pytest.mark.round3

TASK = canonical_task(os.environ.get("PV_CHECK_TASK") or "phenotyping")
TOL = 1e-5


@pytest.fixture(scope="module")
def run():
    row = manifest.latest("r3", "fusion", task=TASK)
    if row is None:
        pytest.skip(f"no r3 row for {TASK} -- run python -m tools.pv.smoke_round3 first")
    run_json = Path(row["file_path"]).parents[2] / "run.json"
    return {**json.loads(run_json.read_text(encoding="utf-8")), "row": row}


@pytest.fixture(scope="module")
def live(run):
    args = round3.trainer_args(run)
    args.resume = False
    loaders = round3.live_loaders(args)
    return args, loaders, round3.build_trainer(args, loaders)


def test_a_manifest_row_and_parents(run):
    row = run["row"]
    assert row["stage"] == "r3" and row["reader"] == "fusion"
    ids = row["parent_id"].split(";")
    assert len(ids) == len(READERS_FOR_TASK[TASK])
    for reader, parent_id in zip(READERS_FOR_TASK[TASK], ids, strict=True):
        parent = manifest.find(parent_id)
        assert parent["stage"] == "r2b" and parent["reader"] == reader
        assert manifest.sha256_file(Path(parent["file_path"])) == parent["sha256"]
    assert manifest.sha256_file(Path(row["file_path"])) == row["sha256"]
    print(f"\n{row['id']} <- {row['parent_id']}: {row['notes'][:300]}")


def test_b_model_builds_and_f2_bit_identical(run, live):
    lines = run.get("f2") or []
    print("\n" + "\n".join(lines))
    assert len(lines) == len(READERS_FOR_TASK[TASK]), "every reader must report bit-identical"
    _, _, trainer = live  # rebuilding loads the parents strictly again; it raises otherwise
    assert trainer.model.fusion_model.cxr_high_proj.out_features == trainer.args.patch_output_dim


def test_c_f4_absent_note_gets_no_late_weight(run, live):
    args, loaders, trainer = live
    trainer.load_state(run["trained"]["main"])
    trainer.model.eval()
    batch = list(next(iter(loaders[1])))
    batch[3] = [""] * len(batch[3])  # every radiology report absent
    x, seq, img, pairs, rr, dn, _y = r3_cache.live_inputs(batch, trainer.device)
    with torch.no_grad():
        trainer.model(x, seq, img, pairs, rr, dn)
    weights = trainer.model.fusion_model.last_late_weights
    mods = args.modalities.split("-")
    assert torch.all(weights[:, mods.index("RR")] == 0)
    if "DN" in mods:
        present = torch.tensor([bool(n.strip()) for n in dn])
        assert torch.all(weights[present, mods.index("DN")] > 0), "DN lost its weight (F4)"


def test_d_cached_equals_live(run, live):
    e = run.get("equivalence") or {}
    print(f"\nrecorded: {e}")
    assert e and all(v <= TOL for k, v in e.items() if k.startswith("max_abs")), e
    args, loaders, trainer = live
    report = r3_cache.compare_live_cached(trainer.model, next(iter(loaders[1])), trainer.device)
    print(f"again: {report}")
    assert all(v <= TOL for k, v in report.items() if k.startswith("max_abs")), report


def test_e_trained_the_requested_variants(run):
    """Exactly the variants run.json says were asked for (train step, then any ablations)."""
    trained = run.get("trained") or {}
    requested = set(run.get("requested_variants") or ["main"])
    requested |= set(run.get("requested_ablations") or [])
    assert requested <= set(trained), (sorted(requested), sorted(trained))
    for name in sorted(requested):
        ckpt = torch.load(trained[name], map_location="cpu", weights_only=False)
        assert ckpt["best_head"] in ("high", "low", "late", "combined", "less_combined"), name
        print(f"{name}: best epoch {ckpt['epoch']}, head {ckpt['best_head']}")
    if round3.MAIN_ABLATION in (run.get("requested_variants") or []):
        assert f"ablation {round3.MAIN_ABLATION}: head=" in run["row"]["notes"]
    if run.get("requested_ablations"):
        table = (Path(run["save_dir"]) / "r3_ablations.csv").read_text(encoding="utf-8")
        for name in run["requested_ablations"]:
            assert any(line.startswith(f"{name},") for line in table.splitlines()), name


def test_f_last_validation_batch_of_one_and_missing_image(run):
    cache_dir = Path(run["cache_dir"])
    args = round3.trainer_args(run)
    val = r3_cache.CacheLoader(cache_dir, "val", args.batch_size, shuffle=False,
                               drop_last=False, classes=["c"] * args.num_classes)  # fmt: skip
    sizes = [b["y"].shape[0] for b in val]
    print(f"\nval batches: {sizes}")
    if run["data"] == "virtual":
        assert sizes[-1] == 1
        assert "X-rays missing on disk treated as absent" in run["row"]["notes"]
