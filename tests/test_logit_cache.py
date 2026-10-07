"""--cache_frozen_logits (Round 2b): replaying cached logits equals re-running the reader.

Runs medpatch's real ``calibration.train()`` -- the Round 2b loop, ECE tables,
checkpoints -- on a real TempCUnimodalRR whose text encoder is a tiny frozen
stub (a fixed embedding per note, a linear layer and an active-by-default
dropout). Random tensors only; a few seconds on CPU, nothing downloaded.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from tools.pv import manifest, run
from tools.pv.medpatch_bridge import parse_args
from tools.pv.paper_scripts import build_argv

TOKENS, DIM, BATCH, SAMPLES, EPOCHS = 6, 8, 4, 10, 3  # 10 samples: last batch is short


class StubText(nn.Module):
    """Text_encoder's interface: a frozen fixed feature sequence per note."""

    feats_dim_rr = feats_dim_dn = DIM
    full_feats_dim_rr = full_feats_dim_dn = DIM

    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(SAMPLES, TOKENS * DIM)
        self.bert = nn.Linear(DIM, DIM)
        self.dropout = nn.Dropout(0.5)  # must be off (frozen reader in eval)
        self.calls = 0

    def forward(self, dn_notes=None, rr_notes=None):
        self.calls += 1
        notes = rr_notes if rr_notes is not None else dn_notes
        idx = torch.tensor([int(n.split("-")[1]) for n in notes])
        feats = self.dropout(self.bert(self.table(idx).view(len(notes), TOKENS, DIM)))
        return None, feats, None, feats


class ValDL:
    """val_dl's batch layout (DataFusion collate), fixed order like shuffle=False."""

    def __init__(self, classes: int, mortality: bool):
        gen = np.random.default_rng(0)
        shape = (SAMPLES,) if mortality else (SAMPLES, classes)
        labels = gen.integers(0, 2, size=shape).astype(np.float32)
        self.batches = []
        for start in range(0, SAMPLES, BATCH):
            ids = range(start, min(start + BATCH, SAMPLES))
            n = len(ids)
            self.batches.append(
                (
                    np.zeros((n, 2, 76), dtype=np.float32),
                    torch.zeros(n, 3, 4, 4),
                    [""] * n,
                    [f"note-{i}" for i in ids],
                    labels[start : start + n],
                    None,
                    [2] * n,
                    [False] * n,
                )
            )
        self.dataset = argparse.Namespace(CLASSES=[f"class {c}" for c in range(classes)])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def make_args(task: str, cache: bool, frozen_eval: bool = True, save_dir: str = "."):
    classes = "1" if task == "in-hospital-mortality" else "3"
    overrides: dict[str, str | None] = {
        "--num_classes": classes,
        "--lr": "0.05",
        "--epochs": str(EPOCHS),
        "--save_dir": save_dir,
    }
    if frozen_eval:
        overrides["--frozen_readers_eval"] = None
    if cache:
        overrides["--cache_frozen_logits"] = None
    return parse_args(build_argv("r2b", "rr", overrides, task=task))


def make_trainer(args, base_state, monkeypatch):
    import trainers.Calibration as calibration_module  # noqa: PLC0415
    from models.fusion import Fusion  # noqa: PLC0415
    from models.loss_set import Loss  # noqa: PLC0415
    from trainers.trainer import Trainer  # noqa: PLC0415

    monkeypatch.setattr(calibration_module.wandb, "log", lambda *a, **k: None)
    trainer = calibration_module.calibration.__new__(calibration_module.calibration)
    Trainer.__init__(trainer, args)  # skips wandb.init and the real encoders
    trainer.epoch = trainer.start_epoch = 0
    trainer.device = torch.device("cpu")
    trainer.val_dl = ValDL(args.num_classes, args.task == "in-hospital-mortality")
    trainer.model = Fusion(args, None, None, StubText())
    trainer.model.load_state_dict(base_state)
    trainer.best_auroc = float("inf")
    trainer.loss = Loss(args)
    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(), args.lr, betas=(0.9, args.beta_1)
    )
    trainer.logit_cache = None
    return trainer


def base_state(task: str):
    from models.fusion import Fusion  # noqa: PLC0415

    torch.manual_seed(0)
    model = Fusion(make_args(task, cache=False), None, None, StubText())
    with torch.no_grad():  # overconfident logits, so the temperature has work to do
        model.fusion_model.rr_confidence_predictor.confidence_layer.weight.mul_(8)
    return copy.deepcopy(model.state_dict())


def run_once(tmp_path, task, cache, state, monkeypatch):
    folder = tmp_path / ("cached" if cache else "uncached")
    folder.mkdir()
    monkeypatch.chdir(folder)  # Calibration.py writes its tables to the cwd
    matplotlib.use("Agg", force=True)  # it also saves calibration-curve PNGs; no GUI
    trainer = make_trainer(make_args(task, cache, save_dir=str(folder)), state, monkeypatch)
    trainer.train()
    best = Path(trainer.checkpoint_path())
    tables = {p.name: pd.read_csv(p, index_col=0) for p in sorted(folder.glob("*.csv"))}
    return trainer, best, tables


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
def test_cached_equals_uncached(tmp_path, task, monkeypatch, capsys):
    state = base_state(task)
    plain, plain_best, plain_tables = run_once(tmp_path, task, False, state, monkeypatch)
    cached, cached_best, cached_tables = run_once(tmp_path, task, True, state, monkeypatch)
    out = capsys.readouterr().out

    t_plain = plain.model.fusion_model.rr_temperature.detach()
    t_cached = cached.model.fusion_model.rr_temperature.detach()
    assert not torch.allclose(t_plain, torch.ones_like(t_plain))  # it actually trained
    torch.testing.assert_close(t_cached, t_plain, atol=1e-6, rtol=0)

    # Same checkpoint: file name, best epoch, best ECE, every tensor.
    assert plain_best.is_file() and cached_best.name == plain_best.name
    a = torch.load(plain_best, weights_only=False)
    b = torch.load(cached_best, weights_only=False)
    assert a["epoch"] == b["epoch"]
    assert float(a["best_auroc"]) == pytest.approx(float(b["best_auroc"]), abs=1e-6)
    assert a["state_dict"].keys() == b["state_dict"].keys()
    for key in a["state_dict"]:
        torch.testing.assert_close(b["state_dict"][key], a["state_dict"][key], atol=1e-6, rtol=0)

    # Same ECE tables and per-token probability files, same names.
    assert plain_tables.keys() == cached_tables.keys()
    assert any(name.startswith("final_") for name in plain_tables)
    for name, table in plain_tables.items():
        pd.testing.assert_frame_equal(cached_tables[name], table, atol=1e-6, rtol=0)

    # The point of it: the reader ran once over val instead of once per epoch.
    batches = len(plain.val_dl)
    assert plain.model.text_model.calls == batches * (1 + EPOCHS)
    assert cached.model.text_model.calls == batches
    assert f"cached frozen logits: {batches} batches, {SAMPLES} samples" in out
    assert "epoch 0 train step 0/" in out and "progress " in out


def test_refused_without_frozen_readers_eval(tmp_path, monkeypatch):
    state = base_state("phenotyping")
    args = make_args("phenotyping", cache=True, frozen_eval=False, save_dir=str(tmp_path))
    trainer = make_trainer(args, state, monkeypatch)
    with pytest.raises(SystemExit, match="needs --frozen_readers_eval"):
        trainer.train()
    assert trainer.model.text_model.calls == 0


def test_refused_when_another_parameter_trains(tmp_path, monkeypatch):
    state = base_state("phenotyping")
    trainer = make_trainer(
        make_args("phenotyping", cache=True, save_dir=str(tmp_path)), state, monkeypatch
    )
    trainer.model.fusion_model.rr_confidence_predictor.confidence_layer.weight.requires_grad_(True)
    with pytest.raises(SystemExit, match="would receive gradients"):
        trainer.train()


def test_refused_when_dropout_stays_in_train_mode(tmp_path, monkeypatch):
    state = base_state("phenotyping")
    trainer = make_trainer(
        make_args("phenotyping", cache=True, save_dir=str(tmp_path)), state, monkeypatch
    )
    trainer.model.fusion_model.head_dropout = nn.Dropout(0.1)  # outside the frozen reader
    with pytest.raises(SystemExit, match="dropout / batch-norm"):
        trainer.train()


def test_default_is_off():
    assert make_args("phenotyping", cache=False).cache_frozen_logits is False


# ── tools/pv plans ───────────────────────────────────────────────────────────


@pytest.fixture
def parents(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MANIFEST", str(tmp_path / "manifest.csv"))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "virtual"
    root.mkdir()
    (root / "README.md").write_text("SYNTHETIC")
    for reader in ("ehr", "rr"):
        for stage in ("r1", "r2"):
            f = tmp_path / f"{stage}-{reader}.pth.tar"
            f.write_bytes(stage.encode())
            manifest.append_row(
                {
                    "id": manifest.next_id(stage, reader),
                    "stage": stage,
                    "reader": reader,
                    "data": "virtual",
                    "file_path": f.as_posix(),
                    "sha256": manifest.sha256_file(f),
                    "bert_model_name": manifest.DEFAULT_BERT if reader == "rr" else "",
                }
            )
    return ["--data", "virtual", "--data-root", str(root)]


def _plan(argv):
    return run.plan(run.build_parser().parse_args(argv))


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["r2b", "--reader", "rr"], True),
        (["r2b", "--reader", "ehr"], False),
        (["r2b", "--reader", "ehr", "--cache-frozen-logits"], True),
        (["r2b", "--reader", "rr", "--no-cache-frozen-logits"], False),
        (["r2", "--reader", "rr"], False),
    ],
)
def test_plans_pass_the_flag(parents, argv, expected):
    plan = _plan([*argv, *parents])
    assert ("--cache_frozen_logits" in plan["argv"]) is expected
    note = "cached frozen logits (numerically equivalent)"
    assert (note in plan["deviations"]) is expected


def test_flag_refused_outside_r2b(parents):
    with pytest.raises(manifest.ManifestError, match="r2b runs only"):
        _plan(["r2", "--reader", "rr", "--cache-frozen-logits", *parents])
