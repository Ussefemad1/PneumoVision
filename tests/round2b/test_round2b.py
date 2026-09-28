"""Round 2b (temperature calibration) acceptance checks.

    python -m pytest -m round2b -s       # -s shows the printed tables

Reads the latest r2b row per reader from the manifest (runs/manifest.csv, or
$PV_MANIFEST) and checks the saved checkpoints with medpatch's own model code.
Skips when no Round 2b runs exist yet -- create them with
``python -m tools.pv.smoke_round2b``.

Unlike the Round 2 checks, check f) trains: it runs one calibration epoch to
prove the trainer only ever reads the validation split. Run this suite where
training is allowed (Colab), not on a machine reserved for non-training checks.

On virtual data every number here is SYNTHETIC -- not a scientific result.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from tools.pv import manifest
from tools.pv.evaluate import load_run, model_output, run_args, score_checkpoint
from tools.pv.medpatch_bridge import build_loaders, build_trainer, medpatch_cwd
from tools.pv.paper_scripts import LOAD_FLAG, READERS

pytestmark = pytest.mark.round2b

#: "Not obviously worse": after calibration, the ECE on the split it was fitted
#: on may not rise by more than this. A sanity check, not a scientific claim.
ECE_TOLERANCE = 0.02


def r2b_row(reader: str) -> dict[str, str]:
    row = manifest.latest("r2b", reader)
    if row is None:
        pytest.skip(
            f"no r2b row for {reader} in {manifest.manifest_path()} -- "
            "run `python -m tools.pv.smoke_round2b` first"
        )
    return row


def run_dir(row: dict[str, str]) -> Path:
    # <save_dir>/phenotyping/<fusion_type>/best_checkpoint_....pth.tar
    return Path(row["file_path"]).parents[2]


def state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def is_temperature(key: str, reader: str) -> bool:
    return key.endswith(f"{reader}_temperature")


def parent_file(run: dict) -> Path:
    argv = run["argv"]
    return Path(argv[argv.index(LOAD_FLAG[run["reader"]]) + 1])


@pytest.fixture(scope="module")
def loaders_cache():
    cache: dict[str, tuple] = {}

    def get(reader: str, args):
        if reader not in cache:
            cache[reader] = build_loaders(args)
        return cache[reader]

    return get


# ── a) everything but the temperature is frozen ─────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_a_only_temperature_trained(reader, loaders_cache):
    row = r2b_row(reader)
    run = load_run(run_dir(row))
    args = run_args(run)  # with --load_<reader> = the r2 parent
    args.resume = False
    before_trainer = build_trainer(args, loaders_cache(reader, args))
    before = {k: v.detach().cpu() for k, v in before_trainer.model.state_dict().items()}
    after = state_dict(row["file_path"])
    parent = state_dict(manifest.find(row["parent_id"])["file_path"])

    trainable = [
        (n, p.numel()) for n, p in before_trainer.model.named_parameters() if p.requires_grad
    ]
    print(f"\n[{reader}] trainable parameters in Round 2b:")
    for name, count in trainable:
        print(f"    {count:>9,}  {name}")

    assert set(before) == set(after), "checkpoint keys differ from the model's"
    temperature = [k for k in after if is_temperature(k, reader)]
    assert temperature, f"no {reader}_temperature in the checkpoint"
    assert all(is_temperature(n, reader) for n, _ in trainable), (
        "something other than the temperature is trainable"
    )

    frozen = [k for k in after if not is_temperature(k, reader)]
    for key in frozen:
        assert torch.equal(after[key], before[key]), f"frozen weight changed: {key}"
        assert key in parent, f"{key} is not in the r2 parent -- load_state could not restore it"
        assert torch.equal(after[key], parent[key]), f"differs from r2 parent: {key}"
    changed = [k for k in temperature if not torch.equal(after[k], before[k])]
    assert changed, "temperature unchanged -- the best checkpoint is the uncalibrated start"
    print(
        f"[{reader}] {len(frozen)} tensors (reader, classifier, Round 2 confidence head) "
        f"bit-identical to r2 parent {row['parent_id']}; changed: {changed}"
    )


# ── b) the temperature stays positive ────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_b_temperature_positive(reader):
    row = r2b_row(reader)
    sd = state_dict(row["file_path"])
    temps = torch.cat([sd[k].flatten() for k in sd if is_temperature(k, reader)])
    print(
        f"\n[{reader}] temperature: {temps.numel():,} values, min {temps.min():.6f}  "
        f"mean {temps.mean():.6f}  max {temps.max():.6f}  "
        f"moved from 1.0: {(temps != 1).float().mean():.1%}"
    )
    assert torch.isfinite(temps).all()
    # The forward divides by temperature.clamp_min(1e-9); a raw value at or below
    # that floor means the clamp, not the learned value, is doing the scaling.
    assert (temps > 1e-9).all(), f"temperature at or below the 1e-9 clamp: {temps.min()}"


# ── c) checkpoint round-trip ─────────────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_c_checkpoint_reload_reproduces_outputs(reader, loaders_cache):
    row = r2b_row(reader)
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
    rows = [r for r in manifest.read_rows() if r["stage"] == "r2b"]
    if not rows:
        pytest.skip("no r2b rows yet")
    for row in rows:
        parent = manifest.find(row["parent_id"])
        assert parent is not None, f"{row['id']}: parent {row['parent_id']!r} not in manifest"
        assert parent["stage"] == "r2", f"{row['id']}: parent is {parent['stage']}, not r2"
        assert parent["reader"] == row["reader"], f"{row['id']}: parent reader differs"
        assert parent["data"] == row["data"], f"{row['id']}: parent data kind differs"
        assert manifest.sha256_file(Path(parent["file_path"])) == parent["sha256"], (
            f"{row['id']}: parent file no longer matches its sha256"
        )
        assert manifest.sha256_file(Path(row["file_path"])) == row["sha256"], (
            f"{row['id']}: own file no longer matches its sha256"
        )
        grandparent = manifest.find(parent["parent_id"])
        assert grandparent is not None and grandparent["stage"] == "r1", (
            f"{row['id']}: its r2 parent {parent['id']} has no r1 parent"
        )
        print(f"{row['id']} <- {parent['id']} <- {grandparent['id']}  ({row['reader']}, sha256 ok)")


# ── e) calibration sanity (ECE before vs after) ──────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_e_ece_not_obviously_worse(reader, loaders_cache):
    row = r2b_row(reader)
    run = load_run(run_dir(row))
    args = run_args(run)
    loaders = loaders_cache(reader, args)
    ece = {}
    for split in ("val", "test"):
        before = score_checkpoint(args, parent_file(run), loaders, split=split).ece
        after = score_checkpoint(args, Path(row["file_path"]), loaders, split=split).ece
        ece[split] = (before, after)
    print(
        f"\n[{reader}] SYNTHETIC -- not a scientific result: ECE (mean of 25 classes, all "
        f"real tokens) val {ece['val'][0]:.4f} -> {ece['val'][1]:.4f} (fitted here)   "
        f"test {ece['test'][0]:.4f} -> {ece['test'][1]:.4f} (held out)"
    )
    before_val, after_val = ece["val"]
    assert after_val <= before_val + ECE_TOLERANCE, (
        f"ECE on the calibration split got worse: {before_val:.4f} -> {after_val:.4f}"
    )


# ── f) calibration only ever reads the validation split ──────────────────────


class Tripwire:
    """A dataloader stand-in that fails the test if anything iterates it."""

    def __init__(self, name: str, real):
        self.name = name
        self.dataset = real.dataset
        self._len = len(real)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        raise AssertionError(f"Round 2b touched the {self.name} split")


@pytest.mark.parametrize("reader", READERS)
def test_f_calibration_reads_only_val(reader, loaders_cache):
    from trainers.Calibration import calibration  # noqa: PLC0415

    row = r2b_row(reader)
    args = copy.copy(run_args(load_run(run_dir(row))))
    args.resume = False
    train_dl, val_dl, test_dl = loaders_cache(reader, args)

    iterated = {"val": 0}

    class CountingVal:
        dataset = val_dl.dataset

        def __len__(self):
            return len(val_dl)

        def __iter__(self):
            iterated["val"] += 1
            return iter(val_dl)

    with medpatch_cwd():
        trainer = calibration(
            Tripwire("train", train_dl), CountingVal(), args, Tripwire("test", test_dl)
        )
        trainer.train_epoch(inference=True)  # the pre-training ECE pass
        trainer.train_epoch()  # one real calibration epoch
    assert iterated["val"] == 2
    print(f"\n[{reader}] calibration read val twice; train and test never iterated")
