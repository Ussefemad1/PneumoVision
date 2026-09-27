"""Run one training stage for one reader and record it in the manifest.

    python -m tools.pv.run r2 --reader cxr --data virtual
    python -m tools.pv.run r1 --reader ehr --data virtual      # virtual stand-in

The fusion_main.py command is built from the paper's own script
(Unimodal/<READER>.sh for r1, Confidence/Confidence-<READER>.sh for r2); every
setting in it is kept. Replaced, and recorded in run.json and the manifest:

- data directories, --save_dir (under RUNS_ROOT) and --normalizer_state
  (virtual data ships its own; real data keeps medpatch's default);
- --load_<reader> for r2, taken from the manifest -- never hard-coded;
- --resume (always: rerunning with the same --run-id continues an interrupted run);
- --num_workers when set, and the smoke-test --epochs / --batch_size overrides.

r2 refuses to start unless its parent is an r1 row for the same reader and the
same kind of data, whose file still matches the recorded sha256.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import manifest
from .evaluate import checkpoint_path, score_checkpoint
from .medpatch_bridge import MEDPATCH, REPO_ROOT, fusion_main_argv, parse_args
from .paper_scripts import LOAD_FLAG, READERS, SCRIPT_FOR, build_argv, script_settings

#: Settings a virtual smoke run may change, and to what. Everything else is the paper's.
VIRTUAL_DEFAULTS = {
    "r1": {"epochs": 2, "batch_size": 4, "bootstrap_iters": 20},
    "r2": {"epochs": 5, "batch_size": 4},
}
#: Round 2 epochs per reader on virtual data. The confidence head is one linear
#: layer and the smoke set is tiny, so it needs more passes than 5 to learn;
#: EHR and CXR epochs are cheap on CPU, BERT (RR/DN) epochs are not.
VIRTUAL_R2_EPOCHS = {"ehr": 20, "cxr": 20, "rr": 5, "dn": 5}
#: Paper default for flags the scripts do not set explicitly.
PAPER_DEFAULTS = {"--bootstrap_iters": "1000"}


def data_dirs(args) -> dict[str, str]:
    if args.data == "virtual":
        root = Path(args.data_root or REPO_ROOT / "data" / "virtual" / args.preset).resolve()
        if not (root / "README.md").is_file():
            raise SystemExit(
                f"No virtual dataset at {root}. Generate it with:\n"
                f"  python -m tools.synthetic.make_virtual_dataset --preset {args.preset}"
            )
        return {
            "--ehr_data_dir": str(root / "ehr"),
            "--cxr_data_dir": str(root / "cxr"),
            "--notes_data_dir": str(root / "notes"),
            "--normalizer_state": str(root / "ehr" / "ph_ts1.0.virtual.normalizer"),
        }
    missing = [
        n for n in ("ehr_data_dir", "cxr_data_dir", "notes_data_dir") if not getattr(args, n)
    ]
    if missing:
        raise SystemExit(
            "--data real needs " + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )
    return {
        "--ehr_data_dir": args.ehr_data_dir,
        "--cxr_data_dir": args.cxr_data_dir,
        "--notes_data_dir": args.notes_data_dir,
    }


def data_seed(args) -> str:
    if args.data != "virtual":
        return ""
    root = Path(args.data_root or REPO_ROOT / "data" / "virtual" / args.preset)
    try:
        return str(json.loads((root / "summary.json").read_text())["seed"])
    except (OSError, KeyError, ValueError):
        return ""


def plan(args) -> dict:
    """Everything about the run except executing it."""
    stage, reader = args.stage, args.reader
    parent = None
    overrides: dict[str, str | None] = {}
    if stage == "r2":
        parent = (
            manifest.find(args.parent) if args.parent else manifest.latest("r1", reader, args.data)
        )
        if args.parent and parent is None:
            raise manifest.ManifestError(
                f"--parent {args.parent} is not in {manifest.manifest_path()}"
            )
        parent_file = manifest.verify_parent(parent, reader, args.data)
        overrides[LOAD_FLAG[reader]] = str(parent_file)

    run_id = args.run_id or manifest.next_id(stage, reader)
    save_dir = (manifest.runs_root() / stage / reader / run_id).resolve()
    overrides.update(data_dirs(args))
    overrides["--save_dir"] = str(save_dir)
    overrides["--resume"] = None

    deviations = []
    defaults = dict(VIRTUAL_DEFAULTS[stage]) if args.data == "virtual" else {}
    if args.data == "virtual" and stage == "r2":
        defaults["epochs"] = VIRTUAL_R2_EPOCHS[reader]
    paper = script_settings(stage, reader)
    for name in ("epochs", "batch_size", "bootstrap_iters"):
        value = getattr(args, name, None) or defaults.get(name)
        if value is not None:
            flag = f"--{name}"
            overrides[flag] = str(value)
            baseline = paper.get(flag, PAPER_DEFAULTS.get(flag))
            if str(value) != baseline:
                deviations.append(f"{name}={value} (paper {baseline})")
    if args.num_workers is not None:
        overrides["--num_workers"] = str(args.num_workers)
    if stage == "r2" and reader == "dn" and paper.get("--load_rr") is not None:
        deviations.append("--load_dn instead of the script's --load_rr (see model_track_notes)")

    argv = build_argv(stage, reader, overrides)
    return {
        "id": run_id,
        "stage": stage,
        "reader": reader,
        "data": args.data,
        "parent_id": parent["id"] if parent else "",
        "save_dir": str(save_dir),
        "script": str(SCRIPT_FOR[stage](reader).relative_to(REPO_ROOT)).replace("\\", "/"),
        "argv": argv,
        "deviations": deviations,
        "seed": data_seed(args),
    }


def execute(run: dict) -> Path:
    save_dir = Path(run["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    log_path = save_dir / "train.log"
    print(f"[pv] {run['id']}: {run['script']} -> {save_dir}")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            fusion_main_argv(run["argv"]),
            cwd=MEDPATCH,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if any(
                k in line
                for k in (
                    "val [",
                    "checkpoint",
                    "Error",
                    "error",
                    "Traceback",
                    "loaded",
                    "Loaded",
                    "RESUM",
                    "size:",
                )
            ):
                print(f"  {line.rstrip()[:160]}")
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"[pv] fusion_main.py failed (exit {code}); see {log_path}")
    best = checkpoint_path(parse_args(run["argv"]))
    if not best.is_file():
        raise SystemExit(f"[pv] run finished but {best} was not written; see {log_path}")
    return best


def record(run: dict, best: Path, who: str) -> dict[str, str]:
    scores = score_checkpoint(parse_args(run["argv"]), best, split="val")
    notes = [
        f"script={run['script']}",
        "trainer seeds fixed in medpatch (1002/379647)",
        "metric=pneumonia (class 21), val split",
    ]
    if run["stage"] == "r2":
        notes.append("r2 score=mean token confidence-head prob")
    if run["data"] == "virtual":
        notes.insert(0, "VIRTUAL stand-in, SYNTHETIC - not a scientific result")
    notes += run["deviations"]
    row = manifest.append_row(
        {
            "id": run["id"],
            "who": who,
            "stage": run["stage"],
            "reader": run["reader"],
            "task": "phenotyping",
            "file_path": best.as_posix(),
            "sha256": manifest.sha256_file(best),
            "parent_id": run["parent_id"],
            "data": run["data"],
            "seed": run["seed"],
            "val_auroc": "" if scores.auroc != scores.auroc else f"{scores.auroc:.4f}",
            "val_auprc": "" if scores.auprc != scores.auprc else f"{scores.auprc:.4f}",
            "notes": "; ".join(notes),
        }
    )
    run["manifest_row"] = row
    Path(run["save_dir"], "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.pv.run", description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=("r1", "r2"))
    p.add_argument("--reader", choices=READERS, required=True)
    p.add_argument("--data", choices=manifest.DATA_KINDS, required=True)
    p.add_argument(
        "--parent", help="r1 manifest id to load (default: latest r1 row for the reader)"
    )
    p.add_argument("--run-id", help="reuse an existing id to resume an interrupted run")
    p.add_argument("--who", default=os.environ.get("PV_WHO", os.environ.get("USERNAME", "unknown")))
    p.add_argument("--preset", default="smoke", help="virtual dataset preset (default smoke)")
    p.add_argument("--data-root", help="virtual dataset folder (default data/virtual/<preset>)")
    p.add_argument("--ehr-data-dir", dest="ehr_data_dir")
    p.add_argument("--cxr-data-dir", dest="cxr_data_dir")
    p.add_argument("--notes-data-dir", dest="notes_data_dir")
    p.add_argument("--epochs", type=int, help="override the paper's epochs (recorded)")
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="override the paper's batch size (recorded)",
    )
    p.add_argument(
        "--bootstrap-iters",
        dest="bootstrap_iters",
        type=int,
        help="AUROC CI resamples (paper 1000; virtual r1 default 20, recorded)",
    )
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--dry-run", action="store_true", help="print the command and exit")
    return p


def main(argv: list[str] | None = None) -> dict[str, str] | None:
    args = build_parser().parse_args(argv)
    run = plan(args)
    if args.dry_run:
        print(json.dumps(run, indent=2))
        return None
    started = time.time()
    best = execute(run)
    row = record(run, best, args.who)
    print(
        f"[pv] {row['id']} done in {time.time() - started:.0f}s  "
        f"val AUROC {row['val_auroc'] or 'n/a'}  AUPRC {row['val_auprc'] or 'n/a'}  "
        f"-> {manifest.manifest_path()}"
    )
    return row


if __name__ == "__main__":
    sys.exit(0 if main() is not None or "--dry-run" in sys.argv else 1)
