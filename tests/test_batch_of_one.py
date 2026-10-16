"""A last batch of one stay, with medpatch's REAL EHR encoder (LSTM, hidden 512).

Val and test loaders keep the last partial batch, and mortality validation has
2,161 = 135 x 16 + 1 stays. LSTM.forward used ``ht.squeeze()``: with one stay
the features lost their batch axis ([512]), so CMSMAFusion took
batch_size = 512 and the joint classifier crashed (Colab smoke, 2026-10-16).
These tests run the real LSTM (nothing to download) with batches of 3 then 1,
live and cached, both tasks, and check that nothing changes for batch > 1.
Text and CXR readers are the stubs from r3_stubs.py. SYNTHETIC numbers.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from r3_stubs import r3_args
from test_round3_cache import StayDataset, setup
from torch import nn

from tools.pv import r3_cache
from tools.pv.medpatch_bridge import parse_args

TASKS = ["phenotyping", "in-hospital-mortality"]
TOL = 1e-5


def lstm_args(task, fusion_type, modalities="EHR"):
    classes = "25" if task == "phenotyping" else "1"
    return parse_args(["--fusion_type", fusion_type, "--modalities", modalities, "--task", task,
                       "--num_classes", classes, "--ehr_encoder", "lstm", "--classifier", "mlp",
                       "--loss", "bce"])  # fmt: skip


@pytest.mark.parametrize("fusion_type", ["unimodal_ehr", "c-unimodal_ehr", "c-msma"])
@pytest.mark.parametrize("batch", [1, 3])
def test_lstm_features_keep_the_batch_axis(fusion_type, batch):
    from models.ehr_models import EHR_encoder  # noqa: PLC0415

    torch.manual_seed(0)
    enc = EHR_encoder(lstm_args("phenotyping", fusion_type)).eval()
    x = torch.randn(batch, 6, 76)
    lengths = [6 - i % 3 for i in range(batch)]
    with torch.no_grad():
        out = enc(x, lengths)
        feats = out[0] if isinstance(out, tuple) else out
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (ht, _) = enc.ehr_encoder.layer0(packed)
    assert tuple(feats.shape) == (batch, 512)
    if batch > 1:  # the old ht.squeeze() gave exactly this: Rounds 1-2b are unaffected
        assert torch.equal(feats, ht.squeeze())


def _loader(task, collate, sizes, seed=2):
    ds = StayDataset(sum(sizes), task, seed)
    out, start = [], 0
    for size in sizes:
        out.append(collate([ds[i] for i in range(start, start + size)]))
        start += size
    return ds, out


@pytest.mark.parametrize("task", TASKS)
def test_round3_live_and_cached_batches_of_3_then_1(task, tmp_path):
    args, model, collate = setup(task, real_ehr=True)
    assert type(model.fusion_model.ehr_model).__name__ == "EHR_encoder"
    ds, live = _loader(task, collate, [3, 1])
    for batch in live:  # live forward and cached replay agree, including the batch of one
        report = r3_cache.compare_live_cached(model, batch, torch.device("cpu"))
        assert all(v <= TOL for k, v in report.items() if k.startswith("max_abs")), report
        x, seq, img, pairs, rr, dn, _ = r3_cache.live_inputs(batch, torch.device("cpu"))
        with torch.no_grad():
            out = model(x, seq, img, pairs, rr, dn)
        assert out["late"].shape[0] == len(seq)

    r3_cache.build_split(model, ds, "val", tmp_path, batch_size=3, collate_fn=collate,
                         device=torch.device("cpu"), shard_size=3)  # fmt: skip
    cached = r3_cache.CacheLoader(tmp_path, "val", 3, shuffle=False, drop_last=False,
                                  classes=["c"] * args.num_classes)  # fmt: skip
    assert [b["y"].shape[0] for b in cached] == [3, 1]
    model.eval()
    for live_batch, cached_batch in zip(live, cached, strict=True):
        x, seq, img, pairs, rr, dn, _ = r3_cache.live_inputs(live_batch, torch.device("cpu"))
        with torch.no_grad():
            a = model(x, seq, img, pairs, rr, dn)
            b = model.fusion_model.forward_cached(cached_batch)
        for key in r3_cache.HEAD_KEYS:
            assert (a[key] - b[key]).abs().max() <= TOL, key


def _msma(args, model, monkeypatch):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415
    from models.loss_set import Loss  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    trainer = msma.MSMA_Trainer.__new__(msma.MSMA_Trainer)
    msma.Trainer.__init__(trainer, args)
    trainer.args, trainer.model, trainer.device = args, model, torch.device("cpu")
    trainer.epoch, trainer.loss = 0, Loss(args)
    trainer.weights = nn.Parameter(torch.ones(3))
    trainer.scheduler = SimpleNamespace(step=lambda *_: None)
    return trainer


def _labels_both_classes(batches):
    for n, batch in enumerate(batches):
        y = np.arange(np.asarray(batch[4]).size).reshape(np.asarray(batch[4]).shape) + n
        batch[4] = (y % 2).astype(np.float32)
    return batches


@pytest.mark.parametrize("task", TASKS)
def test_round3_validate_live_and_cached_end_on_a_batch_of_one(task, tmp_path, monkeypatch):
    args, model, collate = setup(task, real_ehr=True, bootstrap_iters=3, save_dir=str(tmp_path))
    ds, live = _loader(task, collate, [3, 1])
    trainer = _msma(args, model, monkeypatch)
    trainer.val_dl = _labels_both_classes(live)
    *live_rets, live_loss = trainer.validate(trainer.val_dl, False)
    r3_cache.build_split(model, ds, "val", tmp_path / "c", batch_size=3, collate_fn=collate,
                         device=torch.device("cpu"), shard_size=3)  # fmt: skip
    cached = r3_cache.CacheLoader(tmp_path / "c", "val", 3, shuffle=False, drop_last=False,
                                  classes=["c"] * args.num_classes)  # fmt: skip
    trainer.val_dl = cached
    *cached_rets, cached_loss = trainer.validate(cached, False)
    assert np.isfinite(live_loss) and np.isfinite(cached_loss)
    assert len(live_rets) == len(cached_rets) == 5


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fusion_type", ["unimodal_ehr", "c-unimodal_ehr", "temp_c-unimodal_ehr"])
def test_rounds_1_to_2b_forward_a_batch_of_one(task, fusion_type):
    from models.ehr_models import EHR_encoder  # noqa: PLC0415
    from models.fusion import Fusion  # noqa: PLC0415

    args = lstm_args(task, fusion_type)
    torch.manual_seed(0)
    model = Fusion(args, EHR_encoder(args), None, None).eval()
    for batch in (3, 1):
        x = torch.randn(batch, 5, 76)
        with torch.no_grad():
            out = model(x, [5] * batch)[fusion_type]
        assert out.shape[0] == batch, (fusion_type, tuple(out.shape))


@pytest.mark.parametrize("task", TASKS)
def test_round1_validate_ends_on_a_batch_of_one(task, tmp_path, monkeypatch):
    from datasets.DataFusion import my_collate  # noqa: PLC0415
    from models.ehr_models import EHR_encoder  # noqa: PLC0415
    from models.fusion import Fusion  # noqa: PLC0415

    args = lstm_args(task, "unimodal_ehr")
    args.save_dir, args.bootstrap_iters = str(tmp_path), 3
    args.labels_set = "pheno" if task == "phenotyping" else "mortality"
    torch.manual_seed(0)
    model = Fusion(args, EHR_encoder(args), None, None)
    collate = partial(my_collate, args=args)
    _, live = _loader(task, collate, [3, 1])
    trainer = _msma(args, model, monkeypatch)
    trainer.val_dl = _labels_both_classes(live)
    ret, loss = trainer.validate(trainer.val_dl, False)
    assert np.isfinite(loss) and ret is not None


def test_r3_args_unchanged_for_stub_runs():
    assert r3_args().patch_output_dim == 16  # the stub suite keeps its small widths
