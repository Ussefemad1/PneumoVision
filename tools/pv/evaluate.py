"""Scoring a checkpoint on the validation split, with medpatch's own model.

Metrics are AUROC and AUPRC only (never accuracy), for the platform's task:
pneumonia, phenotype class 21.

- r1 (``unimodal_*``): the reader's classifier logit for class 21.
- r2 (``c-unimodal_*``) and r2b (``temp_c-unimodal_*``, the same head with
  its logits divided by the learned temperature): each token predicts the
  label; the stay's score is the mean over its tokens of sigmoid(logit[21]).
  EHR tokens past a stay's real length are LSTM padding and are excluded.

For token models the expected calibration error is also reported: medpatch's
own ``Calibration.compute_ece`` (10 bins, confidence max(p, 1-p)) per class over
every real token, averaged over the 25 classes.

Round 2 in medpatch computes no AUROC of its own (its best checkpoint is picked
by validation loss), so this is the only place r2 AUROC/AUPRC come from.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .medpatch_bridge import build_loaders, build_trainer, parse_args

PNEUMONIA = 21


def checkpoint_path(args, prefix: str = "best") -> Path:
    """Where medpatch's Trainer.checkpoint_path() writes (same f-string)."""
    return Path(
        f"{args.save_dir}/{args.task}/{args.fusion_type}/{prefix}_checkpoint_{args.lr}_"
        f"{args.task}_{args.fusion_type}_{args.modalities}_{args.data_pairs}.pth.tar"
    )


def load_run(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def run_args(run: dict, *, with_parent: bool = True):
    """The run's fusion_main args; optionally without its --load_* parent."""
    argv = list(run["argv"])
    if not with_parent:
        for flag in ("--load_ehr", "--load_cxr", "--load_rr", "--load_dn"):
            if flag in argv:
                i = argv.index(flag)
                del argv[i : i + 2]
    return parse_args(argv)


def model_output(trainer, batch):
    """(output tensor for this fusion type, seq_lengths) for one batch."""
    x, img, dn, rr, _y_ehr, _y_cxr, seq_lengths, pairs, *_ = batch
    x = torch.from_numpy(x).float().to(trainer.device)
    img = img.to(trainer.device)
    out = trainer.model(x, seq_lengths, img, pairs, rr, dn)
    return out[trainer.args.fusion_type], seq_lengths


def token_mask(trainer, logits: torch.Tensor, seq_lengths) -> torch.Tensor:
    """[B, T] mask of real tokens (EHR padding excluded; other readers: all)."""
    mask = torch.ones(logits.shape[:2], dtype=torch.bool)
    if trainer.args.fusion_type.endswith("_ehr"):
        steps = torch.arange(logits.shape[1])[None, :]
        mask = steps < torch.as_tensor(seq_lengths)[:, None]
    return mask


@dataclass
class Scores:
    labels: np.ndarray
    scores: np.ndarray
    auroc: float
    auprc: float
    confidence: dict | None  # token models (r2/r2b) only: confidence statistics
    ece: float = math.nan  # token models only: mean per-class ECE over real tokens


def is_token_model(fusion_type: str) -> bool:
    """r2 (c-unimodal_*) and r2b (temp_c-unimodal_*) score per token."""
    return "c-unimodal" in fusion_type


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """medpatch's Calibration.compute_ece on [n, classes], averaged over classes."""
    from trainers.Calibration import calibration  # noqa: PLC0415

    holder = SimpleNamespace(args=SimpleNamespace(num_classes=probs.shape[-1]))
    return float(calibration.compute_ece(holder, probs.float(), labels.float()).mean())


def _metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: PLC0415

    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


@torch.no_grad()
def score_checkpoint(args, checkpoint: Path, loaders=None, split: str = "val") -> Scores:
    """Load ``checkpoint`` into a freshly built model and score one split."""
    args = copy.copy(args)
    args.resume = False  # scoring loads `checkpoint` itself, never a run's training state
    loaders = loaders or build_loaders(args)
    trainer = build_trainer(args, loaders)
    trainer.load_state(str(checkpoint))
    trainer.model.eval()
    dl = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]

    is_token = is_token_model(trainer.args.fusion_type)
    labels, scores, gammas = [], [], []
    token_probs, token_labels = [], []
    for batch in dl:
        logits, seq_lengths = model_output(trainer, batch)
        y_all = torch.as_tensor(np.asarray(batch[4]), dtype=torch.float32)  # [B, 25]
        y = y_all[:, PNEUMONIA].numpy()
        if is_token:
            mask = token_mask(trainer, logits, seq_lengths).to(logits.device)
            prob = torch.sigmoid(logits[..., PNEUMONIA])
            per_stay = (prob * mask).sum(1) / mask.sum(1).clamp(min=1)
            scores.append(per_stay.cpu().numpy())
            p_all = torch.sigmoid(logits)[mask]  # [n_tokens, 25]
            gammas.append(torch.maximum(p_all, 1 - p_all).flatten().cpu().numpy())
            token_probs.append(p_all.cpu())
            y_tok = y_all[:, None, :].expand(-1, logits.shape[1], -1)
            token_labels.append(y_tok[mask.cpu()])
        else:
            scores.append(logits.reshape(len(y), -1)[:, PNEUMONIA].cpu().numpy())
        labels.append(y)

    labels_np, scores_np = np.concatenate(labels), np.concatenate(scores)
    auroc, auprc = _metrics(labels_np, scores_np)
    confidence = None
    ece = math.nan
    if is_token:
        g = np.concatenate(gammas)
        confidence = {
            "min": float(g.min()),
            "mean": float(g.mean()),
            "max": float(g.max()),
            "frac_ge_0.75": float((g >= 0.75).mean()),
            "n": int(g.size),
        }
        ece = expected_calibration_error(torch.cat(token_probs), torch.cat(token_labels))
    return Scores(labels_np, scores_np, auroc, auprc, confidence, ece)
