"""A1 (frozen readers stay in eval mode) and A2 (one-class ECE over all tokens).

Non-training: modules are built with stub encoders, nothing is downloaded, no
optimizer step is taken.
"""

from __future__ import annotations

import argparse
import inspect

import pytest
import torch
from torch import nn

from tools.pv.medpatch_bridge import parse_args


class StubText(nn.Module):
    """Text_encoder's interface, with a frozen weight and an active dropout."""

    feats_dim_rr = feats_dim_dn = 8
    full_feats_dim_rr = full_feats_dim_dn = 8

    def __init__(self):
        super().__init__()
        self.bert = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, dn_notes=None, rr_notes=None):
        feats = self.dropout(self.bert(torch.ones(2, 5, 8)))
        return None, feats, None, feats


def _rr_model(flag: bool, fusion_type: str = "c-unimodal_rr"):
    from models.fusion import UnimodalRRConfidence  # noqa: PLC0415

    argv = [
        "--fusion_type",
        fusion_type,
        "--modalities",
        "EHR-RR",
        "--num_classes",
        "25",
        "--classifier",
        "mlp",
    ]
    args = parse_args(argv + (["--frozen_readers_eval"] if flag else []))
    model = UnimodalRRConfidence(args, StubText())  # freezes text_model
    return argparse.Namespace(args=args, model=model)


def _keep(holder):
    from trainers.trainer import Trainer  # noqa: PLC0415

    holder.model.train()
    return Trainer.keep_frozen_readers_in_eval(holder)


def test_flag_off_keeps_the_released_behaviour():
    holder = _rr_model(flag=False)
    assert _keep(holder) == []
    assert holder.model.text_model.training and holder.model.text_model.dropout.training


def test_flag_on_puts_only_the_frozen_reader_in_eval():
    holder = _rr_model(flag=True)
    switched = _keep(holder)
    assert "text_model" in switched
    assert not holder.model.text_model.training
    assert not holder.model.text_model.dropout.training
    assert holder.model.rr_confidence_predictor.training  # the trained head keeps train mode
    # Effect: the frozen reader's features are deterministic in a training step.
    out = [holder.model(rr=["note"])["c-unimodal_rr"] for _ in range(2)]
    assert torch.equal(out[0], out[1])


def test_without_the_flag_dropout_changes_reader_features():
    holder = _rr_model(flag=False)
    _keep(holder)
    torch.manual_seed(0)
    out = [holder.model(rr=["note"])["c-unimodal_rr"] for _ in range(2)]
    assert not torch.equal(out[0], out[1])


def test_round2b_is_covered_and_round3_never_touched():
    from models.fusion import TempCUnimodalRR  # noqa: PLC0415

    args = parse_args(
        [
            "--fusion_type",
            "temp_c-unimodal_rr",
            "--modalities",
            "EHR-RR",
            "--num_classes",
            "25",
            "--classifier",
            "mlp",
            "--frozen_readers_eval",
        ]
    )
    r2b = argparse.Namespace(args=args, model=TempCUnimodalRR(args, StubText()))
    switched = _keep(r2b)
    # Everything but the temperature is frozen in Round 2b, so all of it stays in eval.
    assert {"text_model", "rr_classifier", "rr_confidence_predictor"} <= set(switched)
    assert r2b.model.training  # the module holding the trainable temperature

    holder = argparse.Namespace(
        args=argparse.Namespace(frozen_readers_eval=True, fusion_type="c-msma"),
        model=nn.Sequential(nn.Dropout(0.5)),
    )
    assert _keep(holder) == []


def test_both_trainers_call_it_right_after_train_mode():
    from trainers.Calibration import calibration  # noqa: PLC0415
    from trainers.MSMA_trainer import MSMA_Trainer  # noqa: PLC0415

    msma = inspect.getsource(MSMA_Trainer.train)
    assert "self.model.train()" in msma and "self.keep_frozen_readers_in_eval()" in msma
    assert msma.index("self.keep_frozen_readers_in_eval()") > msma.index("self.model.train()")
    cal = inspect.getsource(calibration.train_epoch)
    assert "self.keep_frozen_readers_in_eval()" in cal


# ── A2: one-class (mortality) ECE covers every token ─────────────────────────


@pytest.mark.parametrize("tokens", [1, 48, 512])  # CXR default, EHR, RR
def test_one_class_ece_is_over_all_tokens_not_token_0(tokens):
    from trainers.Calibration import calibration  # noqa: PLC0415

    holder = argparse.Namespace(args=argparse.Namespace(num_classes=1))
    holder.compute_ece = lambda p, lab, n_bins=10: calibration.compute_ece(holder, p, lab, n_bins)
    labels = torch.zeros(20, tokens)
    labels[:10] = 1.0
    probs = torch.full((20, tokens), 0.9)  # wrong and over-confident for half the stays...
    probs[:, 0] = labels[:, 0]  # ...except token 0, which is perfectly calibrated
    token0_only = calibration.compute_ece(holder, probs, labels)  # released train(): [N, L]
    all_tokens = calibration.flat_ece(holder, probs, labels)  # what train() uses now
    assert token0_only.item() == pytest.approx(0.0)
    if tokens == 1:
        assert all_tokens.item() == pytest.approx(0.0)
    else:
        assert all_tokens.item() > 0.3
