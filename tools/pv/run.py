"""Run one training stage for one reader and record it in the manifest.

    python -m tools.pv.run r2 --reader cxr --data virtual
    python -m tools.pv.run r2b --reader cxr --data virtual     # Round 2b: calibration
    python -m tools.pv.run r1 --reader ehr --data virtual      # virtual stand-in

The fusion_main.py command is built from the paper's own script
(Unimodal/<READER>.sh for r1, Confidence/Confidence-<READER>.sh for r2,
Calibrate/Calibrate-<READER>.sh for r2b); every setting in it is kept.
Replaced, and recorded in run.json and the manifest:

- data directories, --save_dir (under RUNS_ROOT) and --normalizer_state
  (virtual data ships its own; real data keeps medpatch's default);
- --load_<reader> for r2 / r2b, taken from the manifest -- never hard-coded;
- --resume (always: rerunning with the same --run-id continues an interrupted run);
- --num_workers when set, and the smoke-test --epochs / --batch_size overrides.

r2 refuses to start unless its parent is an r1 row, and r2b unless its parent is
an r2 row, for the same reader and the same kind of data, whose file still
matches the recorded sha256 (manifest.PARENT_STAGE).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Run as a file (`python tools/pv/run.py`), the repo root is not on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import manifest  # noqa: E402
from tools.pv.evaluate import checkpoint_path, score_checkpoint  # noqa: E402
from tools.pv.medpatch_bridge import (  # noqa: E402
    MEDPATCH,
    REPO_ROOT,
    fusion_main_argv,
    parse_args,
)
from tools.pv.paper_scripts import (  # noqa: E402
    LOAD_FLAG,
    READERS,
    SCRIPT_FOR,
    build_argv,
    script_settings,
)

#: Settings a virtual smoke run may change, and to what. Everything else is the paper's.
VIRTUAL_DEFAULTS = {
    "r1": {"epochs": 2, "batch_size": 4, "bootstrap_iters": 20},
    "r2": {"epochs": 5, "batch_size": 4},
    "r2b": {"batch_size": 4},
}
#: Round 2 epochs per reader on virtual data. The confidence head is one linear
#: layer and the smoke set is tiny, so it needs more passes than 5 to learn;
#: EHR and CXR epochs are cheap on CPU, BERT (RR/DN) epochs are not.
VIRTUAL_R2_EPOCHS = {"ehr": 20, "cxr": 20, "rr": 5, "dn": 5}
#: Round 2b epochs per reader on virtual data. Only the temperature trains, and
#: only on the validation split (~15 stays, ~4 batches per epoch at batch 4), so
#: an epoch is cheap except for BERT's forward pass. At the paper's lr (0.001,
#: Adam) each temperature moves by at most ~0.001 per step, so 30 epochs x ~4
#: steps move it by at most ~0.1: enough to show calibration working, far from
#: converged. RR/DN get fewer epochs because their cost is the BERT forward.
VIRTUAL_R2B_EPOCHS = {"ehr": 30, "cxr": 30, "rr": 10, "dn": 10}
VIRTUAL_EPOCHS_BY_STAGE = {"r2": VIRTUAL_R2_EPOCHS, "r2b": VIRTUAL_R2B_EPOCHS}
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
    if stage in manifest.PARENT_STAGE:
        # r2 loads its r1 reader; r2b loads the r2 reader + confidence head
        # (TempCUnimodal* uses the same submodule names, so load_state restores
        # them and only <reader>_temperature starts fresh).
        parent_stage = manifest.PARENT_STAGE[stage]
        parent = (
            manifest.find(args.parent)
            if args.parent
            else manifest.latest(parent_stage, reader, args.data)
        )
        if args.parent and parent is None:
            raise manifest.ManifestError(
                f"--parent {args.parent} is not in {manifest.manifest_path()}"
            )
        parent_file = manifest.verify_parent(parent, reader, args.data, expected_stage=parent_stage)
        overrides[LOAD_FLAG[reader]] = str(parent_file)

    run_id = args.run_id or manifest.next_id(stage, reader)
    save_dir = (manifest.runs_root() / stage / reader / run_id).resolve()
    overrides.update(data_dirs(args))
    overrides["--save_dir"] = str(save_dir)
    overrides["--resume"] = None

    deviations = []
    defaults = dict(VIRTUAL_DEFAULTS[stage]) if args.data == "virtual" else {}
    if args.data == "virtual" and stage in VIRTUAL_EPOCHS_BY_STAGE:
        defaults["epochs"] = VIRTUAL_EPOCHS_BY_STAGE[stage][reader]
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
    before = {p.name for p in MEDPATCH.iterdir()}
    try:
        code = _run_fusion_main(run, env, log_path)
    finally:
        _collect_stray_outputs(before, save_dir / "medpatch_outputs")
    if code != 0:
        raise SystemExit(f"[pv] fusion_main.py failed (exit {code}); see {log_path}")
    best = checkpoint_path(parse_args(run["argv"]))
    if not best.is_file():
        raise SystemExit(f"[pv] run finished but {best} was not written; see {log_path}")
    return best


def _collect_stray_outputs(before: set[str], dest: Path) -> None:
    """Move files a trainer wrote into medpatch/ (its cwd) into the run folder.

    Calibration.py writes ECE tables, per-token probability CSVs and 25
    calibration-curve PNGs to the working directory, which must be medpatch/
    (the discretizer config path is relative). They belong to the run, not the
    source tree.
    """
    new = [p for p in MEDPATCH.iterdir() if p.name not in before and p.is_file()]
    if not new:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in new:
        path.replace(dest / path.name)
    print(f"  moved {len(new)} file(s) written into medpatch/ to {dest}")


def _run_fusion_main(run: dict, env: dict[str, str], log_path: Path) -> int:
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
        return proc.wait()


def record(run: dict, best: Path, who: str) -> dict[str, str]:
    scores = score_checkpoint(parse_args(run["argv"]), best, split="val")
    notes = [
        f"script={run['script']}",
        "trainer seeds fixed in medpatch (1002/379647)",
        "metric=pneumonia (class 21), val split",
    ]
    if run["stage"] == "r2":
        notes.append("r2 score=mean token confidence-head prob")
    if run["stage"] == "r2b":
        notes.append("r2b score=mean calibrated token confidence-head prob")
        notes += calibration_notes(run, best, scores)
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


def calibration_notes(run: dict, best: Path, after_val) -> list[str]:
    """ECE (mean over the 25 classes) before and after Round 2b, val and test.

    "Before" is the r2 parent loaded into the same model with every temperature
    at its initial 1.0, i.e. the uncalibrated head. Val is the split the
    temperatures were fitted on (in-sample); test is held out.
    """
    args = parse_args(run["argv"])
    parent_file = Path(run["argv"][run["argv"].index(LOAD_FLAG[run["reader"]]) + 1])
    before_val = score_checkpoint(args, parent_file, split="val")
    before_test = score_checkpoint(args, parent_file, split="test")
    after_test = score_checkpoint(args, best, split="test")
    return [
        f"ECE val {before_val.ece:.4f}->{after_val.ece:.4f} (fit split)",
        f"ECE test {before_test.ece:.4f}->{after_test.ece:.4f} (held out)",
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.pv.run", description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=manifest.STAGES)
    p.add_argument("--reader", choices=READERS, required=True)
    p.add_argument("--data", choices=manifest.DATA_KINDS, required=True)
    p.add_argument(
        "--parent",
        help="manifest id to load: an r1 row for r2, an r2 row for r2b "
        "(default: the latest such row for the reader)",
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
