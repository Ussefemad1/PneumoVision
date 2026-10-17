"""Write SYNTHETIC, randomly initialised Round 3 runs for exercising the stack.

    python scripts/make_fake_runs.py <out-dir>        # e.g. checkpoints/round3

Builds one CMSMAFusion per task with the real architecture (BioBERT, ViT-small,
LSTM; from the pretrained caches) and random fusion weights, saves it as
<out-dir>/{mortality,phenotyping}/fake.pth.tar with a run.json in the model
track's shape (argv, manifest_row.sha256). The service loads them exactly as
the real runs, so memory and latency are representative -- the numbers they
produce are meaningless. Never copy these to a demo server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BERT = "dmis-lab/biobert-v1.1"

_COMMON = [
    "--mode", "train", "--batch_size", "16", "--lr", "0.001",
    "--ehr_encoder", "lstm", "--cxr_encoder", "vit_small_patch16_384", "--text_encoder", "biobert",
    "--classifier", "mlp", "--loss", "bce", "--output_dim", "512", "--output_dim_cxr", "512",
    "--output_dim_rr", "512", "--fusion_type", "c-msma", "--data_pairs", "partial",
    "--ehr_confidence_threshold", "0.75", "--cxr_confidence_threshold", "0.75",
    "--rr_confidence_threshold", "0.75", "--bert_model_name", BERT,
]  # fmt: skip
#: Shaped like the real r3 runs' argv, parents and save_dir included.
ARGV = {
    "mortality": [
        *_COMMON, "--num_classes", "1", "--modalities", "EHR-CXR-RR",
        "--task", "in-hospital-mortality", "--labels_set", "mortality",
        "--load_ehr", "/content/drive/r2b-ehr.pth.tar", "--load_cxr", "/content/drive/r2b-cxr.pth.tar",
        "--load_rr", "/content/drive/r2b-rr.pth.tar", "--ehr_data_dir", "/content/ehr",
        "--save_dir", "/content/drive/r3/r3-fusion-901", "--resume", "--epochs", "100",
    ],
    "pneumonia": [
        *_COMMON, "--num_classes", "25", "--modalities", "EHR-CXR-RR-DN",
        "--task", "phenotyping", "--labels_set", "pheno", "--output_dim_dn", "512",
        "--dn_confidence_threshold", "0.75",
        "--load_ehr", "/content/drive/a.pth.tar", "--load_cxr", "/content/drive/b.pth.tar",
        "--load_rr", "/content/drive/c.pth.tar", "--load_dn", "/content/drive/d.pth.tar",
        "--save_dir", "/content/drive/r3/r3-fusion-902", "--resume", "--epochs", "100",
    ],
}  # fmt: skip
FOLDER = {"mortality": "mortality", "pneumonia": "phenotyping"}
RUN_ID = {"mortality": "r3-fusion-901", "pneumonia": "r3-fusion-902"}


def build(task: str, device: str = "cpu"):
    """(args, model) for a fake run, built as MSMA_Trainer builds it."""
    import torch
    from arguments import args_parser  # type: ignore[import-not-found]
    from models.cxr_models import CXR_encoder  # type: ignore[import-not-found]
    from models.ehr_models import EHR_encoder  # type: ignore[import-not-found]
    from models.fusion import Fusion  # type: ignore[import-not-found]
    from models.text_models import Text_encoder  # type: ignore[import-not-found]

    from app.fusion import clean_argv

    args = args_parser().parse_args(clean_argv(ARGV[task]))
    torch.manual_seed(7 if task == "mortality" else 11)
    model = Fusion(args, EHR_encoder(args), CXR_encoder(args), Text_encoder(args, device)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("_temperature"):
                p.uniform_(0.5, 1.5)  # learned per-position temperatures
            elif name.endswith("weights"):
                p.normal_(0.0, 1.0)  # late-fusion α logits
    return args, model


def write_run(task: str, args, model, out: Path) -> Path:
    import torch

    from app.fusion import sha256_file

    folder = out / FOLDER[task]
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = folder / f"best_checkpoint_0.001_{args.task}_c-msma_{args.modalities}_partial.pth.tar"
    torch.save({"epoch": 3, "state_dict": model.state_dict(), "best_head": "late"}, ckpt)
    run = {
        "id": RUN_ID[task],
        "stage": "r3",
        "fusion_type": "c-msma",
        "data": "synthetic-random-weights",
        "bert_model_name": BERT,
        "argv": ARGV[task],
        "manifest_row": {"id": RUN_ID[task], "sha256": sha256_file(ckpt), "bert_model_name": BERT},
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return ckpt


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    here = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here))
    import os

    env_root = os.environ.get("MEDPATCH_ROOT")
    medpatch = Path(env_root) if env_root else here.parent.parent / "medpatch"
    sys.path.insert(0, str(medpatch))
    out = Path(sys.argv[1])
    for task in ("mortality", "pneumonia"):
        args, model = build(task)
        ckpt = write_run(task, args, model, out)
        print(f"{task}: {ckpt} ({ckpt.stat().st_size / 2**20:.0f} MiB, random weights)")
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
