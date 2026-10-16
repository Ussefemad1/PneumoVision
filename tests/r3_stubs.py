"""Tiny stand-ins for the Round 3 readers (unit tests only; nothing downloaded).

Each stub has the real encoder's interface -- the attributes CMSMAFusion reads
and the outputs it unpacks -- at small widths, so a whole c-msma / c-e-msma
model builds and runs on CPU in milliseconds:

- StubEHR:  EHR_encoder (LSTM) -> (feats [B, D], full [B, T, D])
- StubCXR:  CXRTransformer -> (tokens [B, N, C], cls [B, C]); like the real
            one, it has NO cxr_encoder.projection_layer (F1)
- StubText: Text_encoder -> (dn_proj, dn_full, rr_proj, rr_full); a note's
            tokens are a deterministic function of its text, and an empty note
            still yields tokens, as BERT's [CLS]/[SEP] do

Numbers are random and SYNTHETIC.
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import torch
from torch import nn

from tools.pv.medpatch_bridge import parse_args

D = 16  # --patch_output_dim: EHR pool width, text/CXR projection output
CXR_D = 12  # CXR token width (384 for vit_small_patch16_384)
H = 24  # BERT hidden size (768)
TXT_LEN = 7  # text tokens per note (512)
CXR_TOKENS = 4  # CXR tokens incl. CLS (577)


class StubEHR(nn.Module):
    feats_dim = full_feats_dim = D

    def __init__(self):
        super().__init__()
        self.lstm = nn.Linear(76, D)

    def forward(self, x, seq_lengths):
        full = torch.tanh(self.lstm(x))
        return full[:, -1], full


class StubCXR(nn.Module):
    feats_dim = full_feats_dim = CXR_D

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(3, CXR_D)
        self.offsets = nn.Parameter(torch.randn(CXR_TOKENS, CXR_D) * 0.1)

    def forward(self, img):
        pooled = img.mean(dim=(2, 3))
        tokens = torch.tanh(self.feature_extractor(pooled)).unsqueeze(1) + self.offsets
        return tokens, tokens[:, 0]


class StubText(nn.Module):
    def __init__(self, modalities: str, out_dim: int = D):
        super().__init__()
        self.modalities = modalities.split("-")
        self.feats_dim_rr = self.feats_dim_dn = out_dim
        self.full_feats_dim_rr = self.full_feats_dim_dn = H
        self.bert = nn.Linear(H, H)
        self.bert.config = SimpleNamespace(hidden_size=H)
        if "DN" in self.modalities:
            self.fc_dn = nn.Linear(H, out_dim)
        if "RR" in self.modalities:
            self.fc_rr = nn.Linear(H, out_dim)

    def tokens(self, notes):
        out = []
        for note in notes:
            g = torch.Generator().manual_seed(zlib.crc32(note.strip().encode()))
            out.append(torch.randn(TXT_LEN, H, generator=g))
        return torch.stack(out).to(self.bert.weight.device)

    def forward(self, dn_notes=None, rr_notes=None):
        dn_proj = dn_full = rr_proj = rr_full = None
        if dn_notes is not None and "DN" in self.modalities:
            dn_full = torch.tanh(self.bert(self.tokens(dn_notes)))
            dn_proj = self.fc_dn(dn_full)
        if rr_notes is not None and "RR" in self.modalities:
            rr_full = torch.tanh(self.bert(self.tokens(rr_notes)))
            rr_proj = self.fc_rr(rr_full)
        return dn_proj, dn_full, rr_proj, rr_full


MODALITIES = {"phenotyping": "EHR-CXR-RR-DN", "in-hospital-mortality": "EHR-CXR-RR"}


def r3_args(task: str = "phenotyping", fusion_type: str = "c-msma", **extra):
    argv = [
        "--fusion_type", fusion_type,
        "--modalities", MODALITIES[task],
        "--task", task,
        "--num_classes", "25" if task == "phenotyping" else "1",
        "--classifier", "mlp",
        "--loss", "bce",
        "--labels_set", "pheno" if task == "phenotyping" else "mortality",
        "--patch_output_dim", str(D),
        "--ehr_confidence_threshold", "0.75",
        "--cxr_confidence_threshold", "0.75",
        "--rr_confidence_threshold", "0.75",
        "--dn_confidence_threshold", "0.75",
    ]  # fmt: skip
    for flag, value in extra.items():
        argv += [f"--{flag}"] + ([] if value is None else [str(value)])
    return parse_args(argv)


def r3_model(
    task: str = "phenotyping",
    fusion_type: str = "c-msma",
    seed: int = 0,
    real_ehr: bool = False,
    **extra,
):
    """`real_ehr`: medpatch's own EHR_encoder (the 1-layer LSTM, hidden 512; nothing
    to download) instead of StubEHR, with the pools widened to 512 to match."""
    from models.ehr_models import EHR_encoder  # noqa: PLC0415
    from models.fusion import Fusion  # noqa: PLC0415

    if real_ehr:
        extra = {"ehr_encoder": "lstm", "patch_output_dim": 512, **extra}
    args = r3_args(task, fusion_type, **extra)
    torch.manual_seed(seed)
    ehr = EHR_encoder(args) if real_ehr else StubEHR()
    text = StubText(args.modalities, out_dim=512 if real_ehr else D)
    model = Fusion(args, ehr, StubCXR(), text)
    return args, model


def r3_batch(batch: int, task: str = "phenotyping", missing=(), seed: int = 0):
    """(x, seq_lengths, img, pairs, rr, dn, y) like DataFusion's collate.

    `missing`: (index, modality) pairs -- a missing CXR is an all-zero
    [3, 384, 384] image (what detect_missingness_batch tests for), a missing
    note is "".
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 5, 76, generator=g)
    img = torch.rand(batch, 3, 384, 384, generator=g) + 0.1
    rr = [f"report {seed}-{i}" for i in range(batch)]
    dn = [f"discharge {seed}-{i}" for i in range(batch)]
    for i, modality in missing:
        if modality == "CXR":
            img[i] = 0
        elif modality == "RR":
            rr[i] = ""
        elif modality == "DN":
            dn[i] = ""
    classes = 25 if task == "phenotyping" else 1
    y = (torch.rand(batch, classes, generator=g) > 0.5).float()
    if classes == 1:
        y = y[:, 0]
    pairs = [bool(img[i].any()) for i in range(batch)]
    return x, [5] * batch, img, pairs, rr, (dn if task == "phenotyping" else None), y
