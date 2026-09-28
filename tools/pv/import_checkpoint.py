"""Register an external (teammate) Round 1 checkpoint in the manifest.

    python -m tools.pv.import_checkpoint --reader cxr --who Mariam \\
        --file "/content/drive/MyDrive/pv/r1/cxr/best_checkpoint_...pth.tar" \\
        --val-auroc 0.71 --val-auprc 0.22

This is how real Round 1 files replace the virtual stand-ins: once a file is
registered as an r1 row, ``python -m tools.pv.run r2 --reader cxr --data real``
loads it (the latest r1 row for that reader and data kind, or ``--parent ID``).
No code changes, only a manifest row.

Before registering, the file is opened and checked to really be a Round 1
checkpoint of that reader: a medpatch checkpoint dict with a state_dict holding
the reader's encoder and classifier weights, and no Round 2 confidence head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Run as a file (`python tools/pv/import_checkpoint.py`), the repo root is not
# on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import manifest  # noqa: E402
from tools.pv.paper_scripts import READERS  # noqa: E402

#: state_dict key prefixes a Round 1 checkpoint of each reader must contain.
REQUIRED_PREFIXES = {
    "ehr": ("ehr_model.", "fusion_model.ehr_classifier."),
    "cxr": ("cxr_model.", "fusion_model.cxr_classifier."),
    "rr": ("text_model.bert.", "text_model.fc_rr.", "fusion_model.rr_classifier."),
    "dn": ("text_model.bert.", "text_model.fc_dn.", "fusion_model.dn_classifier."),
}


def load_checkpoint(path: Path, trust_pickle: bool = False) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=not trust_pickle)
    except Exception as exc:  # noqa: BLE001 -- surface any unpickling failure clearly
        if trust_pickle:
            raise
        raise SystemExit(
            f"Could not load {path} with weights_only=True ({type(exc).__name__}: {exc}).\n"
            "If the file comes from a teammate you trust, rerun with --trust-pickle."
        ) from exc


def check_round1(checkpoint: dict, reader: str) -> list[str]:
    """Problems that make this file unusable as the reader's r1 parent (empty = ok)."""
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        return ["not a medpatch checkpoint (no 'state_dict' key)"]
    keys = list(checkpoint["state_dict"])
    problems = [
        f"no '{p}*' weights -- is this really the {reader.upper()} Round 1 file?"
        for p in REQUIRED_PREFIXES[reader]
        if not any(k.startswith(p) for k in keys)
    ]
    if any("confidence_predictor" in k for k in keys):
        problems.append("contains a confidence_predictor -- this is a Round 2 file, not Round 1")
    return problems


def main(argv: list[str] | None = None) -> dict[str, str]:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.import_checkpoint", description=__doc__.splitlines()[0]
    )
    p.add_argument("--reader", choices=READERS, required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--who", required=True, help="who trained it (as in the team Sheet)")
    p.add_argument("--data", choices=manifest.DATA_KINDS, default="real")
    p.add_argument("--task", default="phenotyping")
    p.add_argument("--seed", default="")
    p.add_argument("--val-auroc", dest="val_auroc", default="")
    p.add_argument("--val-auprc", dest="val_auprc", default="")
    p.add_argument("--notes", default="")
    p.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow full unpickling (only for files from people you trust)",
    )
    args = p.parse_args(argv)

    path = args.file.resolve()
    if not path.is_file():
        raise SystemExit(f"No such file: {path}")
    problems = check_round1(load_checkpoint(path, args.trust_pickle), args.reader)
    if problems:
        raise SystemExit(f"Refusing to register {path}:\n  - " + "\n  - ".join(problems))

    digest = manifest.sha256_file(path)
    duplicate = next((r for r in manifest.read_rows() if r["sha256"] == digest), None)
    if duplicate:
        raise SystemExit(f"Already registered as {duplicate['id']} ({duplicate['file_path']}).")

    row = manifest.append_row(
        {
            "id": manifest.next_id("r1", args.reader),
            "who": args.who,
            "stage": "r1",
            "reader": args.reader,
            "task": args.task,
            "file_path": path.as_posix(),
            "sha256": digest,
            "parent_id": "",
            "data": args.data,
            "seed": args.seed,
            "val_auroc": args.val_auroc,
            "val_auprc": args.val_auprc,
            "notes": "; ".join(filter(None, ["imported", args.notes])),
        }
    )
    print(
        f"Registered {row['id']}  {args.reader.upper()}  sha256 {digest[:12]}...  "
        f"-> {manifest.manifest_path()}"
    )
    return row


if __name__ == "__main__":
    main()
