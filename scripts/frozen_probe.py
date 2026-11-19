"""Frozen-encoder probe: is a chest-X-ray backbone better than an ImageNet one?

Phase 3, change 5 of the CXR track (docs/cxr_progress_notes.md). Changes 1 and 2
both came back null -- the augmentation fix did nothing and the split learning
rate removed the overfitting without moving accuracy. Two changes to *how* the
model is trained, two null results. What is left is *what the backbone already
knows*: the paper's encoder is a ViT-Small pretrained on ImageNet, which has
never seen a lung.

Fine-tuning RAD-DINO end to end is not viable here: ViT-Base/14 at 384px is
about 4x the parameters and 1.3x the tokens of vit_small_patch16_384, which put
one run near 29 hours on a Colab T4. So this measures the *representation*
instead, which is what the question is actually about:

    freeze the encoder -> extract one CLS vector per image, once
                       -> train the paper's own head (LayerNorm + Linear)
                       -> compare

Both encoders see the identical cohort, the identical images, the identical
preprocessing and the identical head. The only variable is the backbone. Head
training runs on cached vectors, so it takes seconds per epoch rather than ten
minutes, and the whole comparison fits in about an hour.

This is a *linear probe*, not a fine-tune, so its absolute numbers are not
comparable to the fine-tuned baseline of 0.703. Compare the two rows this
script produces against **each other**. That is the controlled comparison:

    python scripts/frozen_probe.py --encoder imagenet  ...   # control
    python scripts/frozen_probe.py --encoder rad-dino  ...   # treatment

LEAKAGE WARNING (read before reporting the rad-dino number)
-----------------------------------------------------------
RAD-DINO was pretrained on MIMIC-CXR's official *training* split, which covers
almost every MIMIC-CXR patient. This cohort's test stays are selected by ICU
admission, independently of that split, so most test images were very likely in
RAD-DINO's pretraining. Report its score as an upper bound with the caveat
stated, never as a clean result. The imagenet row has no such problem.

Augmentation is switched off for extraction (train split included) so each
image yields one deterministic vector. That is intentional: a cached random
crop would be one arbitrary view, which is worse than no augmentation.

Usage (Colab):

    python scripts/frozen_probe.py \
        --encoder rad-dino \
        --medpatch-dir /content/PneumoVision/medpatch \
        --ehr-data-dir /content/ehr --cxr-data-dir /content/cxr \
        --task pneumonia-48h --cache-dir /content/probe_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# CCS phenotype 21 of 25. docs: CLAUDE.md, "Tasks".
PNEUMONIA_INDEX = 21

ENCODERS = {
    # The paper's own encoder, frozen. The control row.
    "imagenet": "timm vit_small_patch16_384 (ImageNet-21k/1k)",
    # Chest-X-ray pretrained. The treatment row. See LEAKAGE WARNING above.
    "rad-dino": "microsoft/rad-dino (DINOv2 ViT-B/14, chest radiographs)",
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_dataloaders(medpatch_dir: Path, argv: list[str]):
    """Return fusion_main.py's own train/val/test loaders and parsed args.

    Runs the real file up to its `print("data gone")` marker rather than
    reimplementing the cohort selection, so the split is bit-identical to the
    baseline run. Same trick the notebook's diagnostic cell uses.
    """
    os.chdir(medpatch_dir)
    sys.path.insert(0, str(medpatch_dir))

    # Patch before fusion_main imports anything: get_cxr_datasets() looks
    # get_transforms up on the module at call time, so replacing the module
    # attribute is enough.
    import datasets.cxr_dataset as cxr_dataset

    original_get_transforms = cxr_dataset.get_transforms

    def deterministic_transforms(args):
        _train, test = original_get_transforms(args)
        return test, test  # no flip, no affine -- one fixed view per image

    cxr_dataset.get_transforms = deterministic_transforms

    sys.argv = ["fusion_main.py", *argv]
    source = Path("fusion_main.py").read_text().split('print("data gone")')[0]
    scope = {"__file__": str(medpatch_dir / "fusion_main.py"), "__name__": "__main__"}
    exec(compile(source, "fusion_main.py", "exec"), scope)

    missing = [k for k in ("train_dl", "val_dl", "test_dl", "args") if k not in scope]
    if missing:
        raise RuntimeError(f"fusion_main.py did not define {missing} -- it changed upstream")
    return scope["train_dl"], scope["val_dl"], scope["test_dl"], scope["args"]


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------

def build_encoder(kind: str, device: torch.device):
    """(module, forward_fn -> [B, D], width). Frozen and in eval mode."""
    if kind == "imagenet":
        import timm

        model = timm.create_model(
            "vit_small_patch16_384", pretrained=True, in_chans=3,
            img_size=384, num_classes=0,
        )
        width = model.num_features

        def forward(img):
            return model.forward_features(img)[:, 0]

    elif kind == "rad-dino":
        from transformers import AutoModel

        model = AutoModel.from_pretrained("microsoft/rad-dino")
        width = model.config.hidden_size

        def forward(img):
            # Dinov2 interpolates its position embeddings, so a 384px input
            # (27x27 patches of 14, + CLS) is accepted even though the
            # checkpoint was trained at 518.
            return model(pixel_values=img).last_hidden_state[:, 0]

    else:
        raise SystemExit(f"unknown --encoder {kind!r}; choose from {sorted(ENCODERS)}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"[encoder] {kind}: {ENCODERS[kind]}")
    print(f"[encoder] {n/1e6:.1f}M parameters, frozen; feature width {width}")
    return model, forward, width


@torch.no_grad()
def extract(dl, forward, device, split: str):
    """One CLS vector and one label row per stay."""
    feats, labels = [], []
    started = time.time()
    for i, batch in enumerate(dl, 1):
        # Same unpacking as MSMA_Trainer.batch_outputs.
        _x, img, _dn, _rr, y_ehr, _y_cxr, _seq, _pairs = batch[:8]
        feats.append(forward(img.to(device)).float().cpu())
        labels.append(torch.from_numpy(np.asarray(y_ehr)).float())
        if i % 50 == 0:
            print(f"  {split}: {i} batches, {time.time()-started:.0f}s", flush=True)
    f = torch.cat(feats)
    y = torch.cat(labels)
    print(f"[extract] {split}: {tuple(f.shape)} features, {tuple(y.shape)} labels "
          f"in {time.time()-started:.0f}s")
    return f, y


# --------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------

def mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    scores = [
        roc_auc_score(y[:, c], p[:, c])
        for c in range(y.shape[1])
        if len(np.unique(y[:, c])) > 1
    ]
    return float(np.mean(scores)) if scores else float("nan")


def train_head(f_tr, y_tr, f_va, y_va, *, num_classes, epochs, lr, device, seed):
    """Train LayerNorm+Linear on cached features; keep the best val epoch.

    The head is medpatch's own MLPClassifier, so the only thing that differs
    from the baseline is which encoder produced the input vector.
    """
    from models.classifier import MLPClassifier

    torch.manual_seed(seed)
    head = MLPClassifier(input_dim=f_tr.shape[1], output_dim=num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, betas=(0.9, 0.9))
    loss_fn = nn.BCEWithLogitsLoss()

    f_tr, y_tr = f_tr.to(device), y_tr.to(device)
    f_va_d = f_va.to(device)
    y_va_np = y_va.numpy()

    best_score, best_state, best_epoch = -1.0, None, -1
    n = f_tr.shape[0]
    batch = 256

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            opt.zero_grad()
            loss_fn(head(f_tr[idx]), y_tr[idx]).backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(head(f_va_d)).cpu().numpy()
        score = mean_auroc(y_va_np, val_pred)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"  head epoch {epoch:3d}  val mean auroc {score:.4f}"
                  f"   (best {best_score:.4f} @ {best_epoch})", flush=True)

    head.load_state_dict(best_state)
    print(f"[head] best val mean auroc {best_score:.4f} at epoch {best_epoch}")
    return head, best_epoch


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def bootstrap_ci(y: np.ndarray, p: np.ndarray, iters: int, seed: int):
    """(AUROC, lo, hi, AUPRC) with a percentile bootstrap over test rows."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"),) * 2
    return (
        float(roc_auc_score(y, p)),
        float(lo),
        float(hi),
        float(average_precision_score(y, p)),
    )


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--medpatch-dir", type=Path, default=Path("/content/PneumoVision/medpatch"))
    ap.add_argument("--ehr-data-dir", default="/content/ehr")
    ap.add_argument("--cxr-data-dir", default="/content/cxr")
    ap.add_argument("--task", default="pneumonia-48h")
    ap.add_argument("--labels-set", default="pheno")
    ap.add_argument("--num-classes", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cache-dir", type=Path, default=Path("/content/probe_cache"))
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--pneumonia-index", type=int, default=PNEUMONIA_INDEX)
    ap.add_argument("--force-extract", action="store_true",
                    help="ignore cached features and recompute them")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device {device}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / f"{args.encoder}_{args.task}.pt"

    if cache.exists() and not args.force_extract:
        print(f"[cache] reusing {cache}")
        blob = torch.load(cache)
    else:
        argv = [
            "--mode", "train", "--epochs", "1",
            "--batch_size", str(args.batch_size),
            "--lr", "3e-05",
            "--num_classes", str(args.num_classes),
            "--modalities", "EHR-CXR",
            "--fusion_type", "unimodal_cxr",
            "--cxr_encoder", "vit_small_patch16_384",
            "--classifier", "mlp", "--loss", "bce",
            "--task", args.task, "--labels_set", args.labels_set,
            "--output_dim_cxr", "512", "--data_pairs", "paired",
            "--use_cls_token", "cls",
            "--num_workers", str(args.num_workers),
            "--ehr_data_dir", args.ehr_data_dir,
            "--cxr_data_dir", args.cxr_data_dir,
            "--save_dir", str(args.cache_dir / "_unused"),
        ]
        (args.cache_dir / "_unused").mkdir(parents=True, exist_ok=True)
        train_dl, val_dl, test_dl, _ = build_dataloaders(args.medpatch_dir, argv)

        model, forward, width = build_encoder(args.encoder, device)
        blob = {"width": width, "encoder": args.encoder}
        for split, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
            f, y = extract(dl, forward, device, split)
            blob[f"{split}_f"], blob[f"{split}_y"] = f, y
        del model
        torch.save(blob, cache)
        print(f"[cache] wrote {cache}")

    head, best_epoch = train_head(
        blob["train_f"], blob["train_y"], blob["val_f"], blob["val_y"],
        num_classes=args.num_classes, epochs=args.head_epochs,
        lr=args.head_lr, device=device, seed=args.seed,
    )

    head.eval()
    with torch.no_grad():
        test_pred = torch.sigmoid(head(blob["test_f"].to(device))).cpu().numpy()
    test_y = blob["test_y"].numpy()

    c = args.pneumonia_index
    auroc, lo, hi, auprc = bootstrap_ci(
        test_y[:, c], test_pred[:, c], args.bootstrap_iters, args.seed)
    mean = mean_auroc(test_y, test_pred)

    print("\n" + "=" * 66)
    print(f"FROZEN PROBE -- {args.encoder}  ({ENCODERS[args.encoder]})")
    print(f"  task                : {args.task}")
    print(f"  test stays          : {len(test_y):,}")
    print(f"  feature width       : {blob['width']}")
    print(f"  head epoch chosen   : {best_epoch} (on validation, 25-class mean)")
    print(f"  PNEUMONIA AUROC     : {auroc:.3f}  ({lo:.3f}, {hi:.3f})")
    print(f"  PNEUMONIA AUPRC     : {auprc:.3f}")
    print(f"  25-class mean AUROC : {mean:.3f}")
    print("=" * 66)
    if args.encoder == "rad-dino":
        print("NOTE: RAD-DINO was pretrained on MIMIC-CXR's official training split,")
        print("      which very likely contains these test patients. Report this as an")
        print("      upper bound with that caveat, not as a clean comparison.")
    print("Compare against the `imagenet` row of this same script, not against the")
    print("fine-tuned baseline -- a frozen probe and a fine-tune are different setups.")

    out = args.cache_dir / f"result_{args.encoder}_{args.task}.json"
    out.write_text(json.dumps({
        "encoder": args.encoder, "task": args.task,
        "pneumonia_auroc": auroc, "ci_low": lo, "ci_high": hi,
        "pneumonia_auprc": auprc, "mean_auroc_25": mean,
        "feature_width": blob["width"], "head_epoch": best_epoch,
        "n_test": int(len(test_y)),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
