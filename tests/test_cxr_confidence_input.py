"""The CXR confidence predictor sees the same tensor shape in Round 2, 2b and 3.

Training a head on one representation and using it on another would not crash
(CLS and patch tokens have the same width), so this is checked directly: each
stage's `cxr_confidence_predictor` is hooked and its input shape recorded, for
both values of --cxr_token_confidence.

The CXR encoder is a stub with CXRTransformer's output contract
(tokens [B, 577, D], cls [B, D]) plus the `cxr_encoder.projection_layer` that
Round 3 reads -- the real encoder lacks it (known upstream defect, CLAUDE.md),
so Round 3 cannot be built with it. Round 3's forward may still fail later for
unrelated reasons; the hook fires first, which is all this test needs.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tools.pv.medpatch_bridge import parse_args

B, TOKENS, D = 2, 577, 384


class StubCXR(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats_dim = D
        self.full_feats_dim = D
        self.cxr_encoder = nn.Module()
        self.cxr_encoder.projection_layer = nn.Linear(D, 512)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, img):
        return torch.randn(img.shape[0], TOKENS, D), torch.randn(img.shape[0], D)


def make_args(token_confidence: bool, fusion_type: str, modalities: str):
    argv = [
        "--fusion_type",
        fusion_type,
        "--modalities",
        modalities,
        "--task",
        "phenotyping",
        "--num_classes",
        "25",
        "--classifier",
        "mlp",
        "--ehr_encoder",
        "lstm",
    ]
    if token_confidence:
        argv.append("--cxr_token_confidence")
    return parse_args(argv)


def predictor_input_shape(model: nn.Module, run) -> tuple[int, ...]:
    seen: list[tuple[int, ...]] = []
    model.cxr_confidence_predictor.register_forward_pre_hook(
        lambda _m, inputs: seen.append(tuple(inputs[0].shape))
    )
    try:
        with torch.no_grad():
            run()
    except Exception:  # noqa: BLE001 -- only the hooked call matters (see module doc)
        pass
    assert seen, "cxr_confidence_predictor was never called"
    return seen[0]


def stage_shapes(token_confidence: bool) -> dict[str, tuple[int, ...]]:
    from models.ehr_models import LSTM  # noqa: PLC0415
    from models.fusion import (  # noqa: PLC0415
        CMSMAFusion,
        EMSMAFusion,
        TempCUnimodalCXR,
        UnimodalCXRConfidence,
    )

    img = torch.zeros(B, 3, 384, 384)
    shapes = {}

    r2 = UnimodalCXRConfidence(make_args(token_confidence, "c-unimodal_cxr", "EHR-CXR"), StubCXR())
    shapes["r2"] = predictor_input_shape(r2, lambda: r2(img=img))

    r2b = TempCUnimodalCXR(make_args(token_confidence, "temp_c-unimodal_cxr", "EHR-CXR"), StubCXR())
    shapes["r2b"] = predictor_input_shape(r2b, lambda: r2b(img=img))

    for name, cls, fusion_type in (
        ("r3 c-msma", CMSMAFusion, "c-msma"),
        ("r3 c-e-msma", EMSMAFusion, "c-e-msma"),
    ):
        args = make_args(token_confidence, fusion_type, "EHR-CXR")
        model = cls(args, LSTM(args), StubCXR(), None)
        x = torch.zeros(B, 10, 76)
        shapes[name] = predictor_input_shape(
            model, lambda m=model, x=x: m(x=x, seq_lengths=[10] * B, img=img, pairs=[True] * B)
        )
    return shapes


@pytest.mark.parametrize("token_confidence, expected", [(False, (B, 1, D)), (True, (B, TOKENS, D))])
def test_every_stage_feeds_the_predictor_the_same_shape(token_confidence, expected):
    shapes = stage_shapes(token_confidence)
    print(f"\n--cxr_token_confidence={token_confidence}: {shapes}")
    assert set(shapes.values()) == {expected}, shapes


def test_default_is_cls_per_image():
    assert make_args(False, "c-unimodal_cxr", "EHR-CXR").cxr_token_confidence is False


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((4, 1, 25), (4, 1, 25)),  # CXR default: CLS as one token -- token axis kept
        ((4, 577, 25), (4, 577, 25)),  # patch tokens / text / EHR, phenotyping
        ((4, 48, 1), (4, 48)),  # mortality: class axis dropped, as before
    ],
)
def test_round2_trainer_keeps_the_token_axis(shape, expected):
    from types import SimpleNamespace  # noqa: PLC0415

    from trainers.MSMA_trainer import MSMA_Trainer  # noqa: PLC0415

    fake = SimpleNamespace(args=SimpleNamespace(fusion_type="c-unimodal_cxr"))
    out = MSMA_Trainer.confidence_logits(fake, {"c-unimodal_cxr": torch.zeros(shape)})
    assert tuple(out.shape) == expected
