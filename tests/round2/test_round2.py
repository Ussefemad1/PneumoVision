"""Round 2 (confidence predictors) acceptance checks.

    python -m pytest -m round2 -s        # -s shows the printed tables

Reads the latest r2 row per reader from the manifest (runs/manifest.csv, or
$PV_MANIFEST) and checks the saved checkpoints with medpatch's own model code.
Skips when no Round 2 runs exist yet -- create them with
``python -m tools.pv.smoke_round2``.

On virtual data every number here is SYNTHETIC -- not a scientific result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.pv import manifest
from tools.pv.evaluate import load_run, model_output, run_args, score_checkpoint
from tools.pv.medpatch_bridge import build_loaders, build_trainer
from tools.pv.paper_scripts import READERS

pytestmark = pytest.mark.round2

READER_PREFIX = {"ehr": "ehr_model.", "cxr": "cxr_model.", "rr": "text_model.", "dn": "text_model."}
AUROC_FLOOR = 0.55


def r2_row(reader: str) -> dict[str, str]:
    row = manifest.latest("r2", reader)
    if row is None:
        pytest.skip(
            f"no r2 row for {reader} in {manifest.manifest_path()} -- "
            "run `python -m tools.pv.smoke_round2` first"
        )
    return row


def run_dir(row: dict[str, str]) -> Path:
    # <save_dir>/phenotyping/<fusion_type>/best_checkpoint_....pth.tar
    return Path(row["file_path"]).parents[2]


def state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def kind(key: str, reader: str) -> str:
    if "confidence_predictor" in key:
        return "confidence"
    if "_classifier" in key:
        return "classifier"
    if key.startswith(READER_PREFIX[reader]) or f".{READER_PREFIX[reader]}" in key:
        return "reader"
    return "other"


@pytest.fixture(scope="module")
def loaders_cache():
    cache: dict[str, tuple] = {}

    def get(reader: str, args):
        if reader not in cache:
            cache[reader] = build_loaders(args)
        return cache[reader]

    return get


# ── a) the reader is frozen ──────────────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_a_reader_frozen_only_confidence_trained(reader, loaders_cache):
    row = r2_row(reader)
    run = load_run(run_dir(row))
    args = run_args(run)  # with --load_<reader> = the r1 parent
    args.resume = False
    before_trainer = build_trainer(args, loaders_cache(reader, args))
    before = {k: v.detach().cpu() for k, v in before_trainer.model.state_dict().items()}
    after = state_dict(row["file_path"])
    parent = state_dict(manifest.find(row["parent_id"])["file_path"])

    trainable = [
        (n, p.numel()) for n, p in before_trainer.model.named_parameters() if p.requires_grad
    ]
    print(f"\n[{reader}] trainable parameters in Round 2:")
    for name, count in trainable:
        print(f"    {count:>9,}  {name}")
    print(
        f"    {sum(c for _, c in trainable):>9,}  total "
        f"(of {sum(p.numel() for p in before_trainer.model.parameters()):,})"
    )

    assert set(before) == set(after), "checkpoint keys differ from the model's"
    changed = sorted(k for k in after if not torch.equal(before[k], after[k]))
    reader_keys = [k for k in after if kind(k, reader) == "reader"]
    assert reader_keys, "no reader weights found"

    for key in reader_keys:
        assert torch.equal(after[key], before[key]), f"reader weight changed: {key}"
        if key in parent:
            assert torch.equal(after[key], parent[key]), f"reader differs from r1 parent: {key}"
    assert changed, "nothing changed -- the best checkpoint is the untrained start"
    assert all(kind(k, reader) == "confidence" for k in changed), (
        f"non-confidence weights changed: {[k for k in changed if kind(k, reader) != 'confidence']}"
    )
    assert all(kind(n, reader) in ("confidence", "classifier") for n, _ in trainable), (
        "a reader parameter is trainable"
    )
    print(
        f"[{reader}] {len(reader_keys)} reader tensors bit-identical to r1 parent "
        f"{row['parent_id']}; changed: {changed}"
    )


# ── b) confidence values ─────────────────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_b_confidence_in_half_to_one(reader, loaders_cache):
    row = r2_row(reader)
    args = run_args(load_run(run_dir(row)), with_parent=False)
    stats = score_checkpoint(args, Path(row["file_path"]), loaders_cache(reader, args)).confidence
    print(
        f"\n[{reader}] gamma = max(sigmoid(l), 1 - sigmoid(l)) over {stats['n']:,} "
        f"token-class values: min {stats['min']:.4f}  mean {stats['mean']:.4f}  "
        f"max {stats['max']:.4f}  fraction >= 0.75: {stats['frac_ge_0.75']:.3f}"
    )
    assert 0.5 <= stats["min"] <= stats["mean"] <= stats["max"] <= 1.0


# ── c) checkpoint round-trip ─────────────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_c_checkpoint_reload_reproduces_outputs(reader, loaders_cache):
    row = r2_row(reader)
    args = run_args(load_run(run_dir(row)), with_parent=False)
    args.resume = False
    loaders = loaders_cache(reader, args)
    batch = next(iter(loaders[1]))
    outputs = []
    for _ in range(2):
        trainer = build_trainer(args, loaders)
        trainer.load_state(row["file_path"])
        trainer.model.eval()
        with torch.no_grad():
            outputs.append(model_output(trainer, batch)[0].cpu())
    assert outputs[0].shape[0] == len(batch[4])
    assert torch.equal(outputs[0], outputs[1]), "reloaded checkpoint gives different outputs"
    print(f"\n[{reader}] reload x2 -> identical outputs, shape {tuple(outputs[0].shape)}")


# ── d) lineage ───────────────────────────────────────────────────────────────


def test_d_lineage():
    rows = [r for r in manifest.read_rows() if r["stage"] == "r2"]
    if not rows:
        pytest.skip("no r2 rows yet")
    for row in rows:
        parent = manifest.find(row["parent_id"])
        assert parent is not None, f"{row['id']}: parent {row['parent_id']!r} not in manifest"
        assert parent["stage"] == "r1", f"{row['id']}: parent is {parent['stage']}"
        assert parent["reader"] == row["reader"], f"{row['id']}: parent reader differs"
        assert parent["data"] == row["data"], f"{row['id']}: parent data kind differs"
        assert manifest.sha256_file(Path(parent["file_path"])) == parent["sha256"], (
            f"{row['id']}: parent file no longer matches its sha256"
        )
        assert manifest.sha256_file(Path(row["file_path"])) == row["sha256"], (
            f"{row['id']}: own file no longer matches its sha256"
        )
        print(f"{row['id']} <- {parent['id']}  ({row['reader']}, sha256 ok)")


# ── e) the plumbing learns ───────────────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_e_confidence_heads_carry_pneumonia_signal(reader, loaders_cache):
    row = r2_row(reader)
    if row["data"] != "virtual":
        pytest.skip("signal sanity is defined on the virtual data's planted signal")
    args = run_args(load_run(run_dir(row)), with_parent=False)
    scores = score_checkpoint(args, Path(row["file_path"]), loaders_cache(reader, args))
    print(
        f"\n[{reader}] SYNTHETIC -- not a scientific result: val pneumonia "
        f"AUROC {scores.auroc:.3f}  AUPRC {scores.auprc:.3f}  "
        f"(n={len(scores.labels)}, positives={int(scores.labels.sum())})"
    )
    assert scores.auroc > AUROC_FLOOR
