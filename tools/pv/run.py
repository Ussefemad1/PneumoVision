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
import shlex
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
    DEFAULT_TASK,
    LOAD_FLAG,
    READERS,
    SCRIPT_FOR,
    TASK_ALIASES,
    build_argv,
    canonical_task,
    refuse_reader,
    script_settings,
    to_pairs,
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


#: The normalizer the virtual dataset generator fits on its own train split.
VIRTUAL_NORMALIZER = {
    "phenotyping": "ph_ts1.0.virtual.normalizer",
    "in-hospital-mortality": "ihm_ts1.0.virtual.normalizer",
}


def task_of(args) -> str:
    return canonical_task(getattr(args, "task", None))


def virtual_root(args) -> Path:
    """data/virtual/<preset> (phenotyping) or data/virtual/<preset>-mortality."""
    if args.data_root:
        return Path(args.data_root)
    suffix = "" if task_of(args) == DEFAULT_TASK else "-mortality"
    return REPO_ROOT / "data" / "virtual" / f"{args.preset}{suffix}"


def data_dirs(args) -> dict[str, str]:
    task = task_of(args)
    if args.data == "virtual":
        root = virtual_root(args).resolve()
        if not (root / "README.md").is_file():
            task_flag = "" if task == DEFAULT_TASK else " --task mortality"
            raise SystemExit(
                f"No virtual dataset at {root}. Generate it with:\n"
                f"  python -m tools.synthetic.make_virtual_dataset --preset {args.preset}"
                f"{task_flag}"
            )
        return {
            "--ehr_data_dir": str(root / "ehr"),
            "--cxr_data_dir": str(root / "cxr"),
            "--notes_data_dir": str(root / "notes"),
            "--normalizer_state": str(root / "ehr" / VIRTUAL_NORMALIZER[task]),
        }
    missing = [
        n for n in ("ehr_data_dir", "cxr_data_dir", "notes_data_dir") if not getattr(args, n)
    ]
    if missing:
        raise SystemExit(
            "--data real needs " + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )
    dirs = {
        "--ehr_data_dir": args.ehr_data_dir,
        "--cxr_data_dir": args.cxr_data_dir,
        "--notes_data_dir": args.notes_data_dir,
    }
    # Only when given: without it fusion_main.py loads its bundled default (the
    # phenotyping file, for every task) -- what a Round 1 run with
    # normalizer_state None used. See docs/REAL_RUNS.md, "EHR normalizer".
    if getattr(args, "normalizer_state", None):
        dirs["--normalizer_state"] = args.normalizer_state
    return dirs


def data_seed(args) -> str:
    if args.data != "virtual":
        return ""
    root = virtual_root(args)
    try:
        return str(json.loads((root / "summary.json").read_text())["seed"])
    except (OSError, KeyError, ValueError):
        return ""


def resolve_bert_model_name(
    reader: str, explicit: str | None, parent: dict[str, str] | None
) -> str:
    """The --bert_model_name a run must use ("" for ehr/cxr), or ManifestError.

    rr/dn checkpoints embed a BERT, and a different model name loads without
    error but tokenizes with the wrong vocabulary. So r2/r2b inherit the
    parent's name, and an explicit value that disagrees is refused rather than
    silently overriding it. A run with no parent (r1) uses the explicit value or
    medpatch's default, and records it.
    """
    if reader not in manifest.TEXT_READERS:
        if explicit:
            raise manifest.ManifestError(f"--bert-model-name is only for rr/dn, not {reader}.")
        return ""
    if parent is None:
        return explicit or manifest.DEFAULT_BERT
    inherited = manifest.bert_model_name_of(parent)
    if not inherited:
        raise manifest.ManifestError(
            f"Parent {parent['id']} has no bert_model_name recorded, so the BERT its weights "
            "were trained with is unknown. Re-register the r1 file with "
            "`python -m tools.pv.import_checkpoint ... --bert-model-name <model>`."
        )
    if explicit and explicit != inherited:
        raise manifest.ManifestError(
            f"--bert-model-name {explicit} differs from parent {parent['id']}'s {inherited}. "
            "A text checkpoint only works with the BERT it was trained with; omit the flag "
            "to inherit it."
        )
    return inherited


def plan(args) -> dict:
    """Everything about the run except executing it."""
    stage, reader = args.stage, args.reader
    task = task_of(args)
    try:
        refuse_reader(reader, task)  # e.g. DN for mortality: discharge notes leak the outcome
    except ValueError as exc:
        raise manifest.ManifestError(str(exc)) from exc
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
            else manifest.latest(parent_stage, reader, args.data, task=task)
        )
        if args.parent and parent is None:
            raise manifest.ManifestError(
                f"--parent {args.parent} is not in {manifest.manifest_path()}"
            )
        parent_file = manifest.verify_parent(
            parent, reader, args.data, expected_stage=parent_stage, task=task
        )
        overrides[LOAD_FLAG[reader]] = str(parent_file)

    bert_model_name = resolve_bert_model_name(
        reader, getattr(args, "bert_model_name", None), parent
    )
    if bert_model_name:
        overrides["--bert_model_name"] = bert_model_name

    run_id = args.run_id or manifest.next_id(stage, reader)
    # Phenotyping keeps its original layout (<RUNS_ROOT>/<stage>/...) so existing
    # folders and --run-id resumes still resolve; other tasks get their own
    # subfolder, so the two tasks can never write to the same place.
    task_dir = manifest.runs_root() if task == DEFAULT_TASK else manifest.runs_root() / task
    save_dir = (task_dir / stage / reader / run_id).resolve()
    overrides.update(data_dirs(args))
    overrides["--save_dir"] = str(save_dir)
    overrides["--resume"] = None
    if stage in manifest.PARENT_STAGE:
        # Round 2 / 2b train a head on a frozen reader. Keep the reader in eval
        # mode (BERT dropout off), as it was when the reader was trained and
        # evaluated; the released code leaves it on. No effect on EHR/CXR.
        overrides["--frozen_readers_eval"] = None

    deviations = []
    defaults = dict(VIRTUAL_DEFAULTS[stage]) if args.data == "virtual" else {}
    if args.data == "virtual" and stage in VIRTUAL_EPOCHS_BY_STAGE:
        defaults["epochs"] = VIRTUAL_EPOCHS_BY_STAGE[stage][reader]
    paper = script_settings(stage, reader, task)
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
    if "--frozen_readers_eval" in overrides:
        deviations.append("frozen_readers_eval (frozen reader in eval: BERT dropout off)")
    if stage == "r2" and reader == "dn" and paper.get("--load_rr") is not None:
        deviations.append("--load_dn instead of the script's --load_rr (see model_track_notes)")

    argv = build_argv(stage, reader, overrides, task=task)
    return {
        "id": run_id,
        "stage": stage,
        "reader": reader,
        "task": task,
        "data": args.data,
        "parent_id": parent["id"] if parent else "",
        "parent_file": overrides.get(LOAD_FLAG[reader]) or "",
        "parent_sha256": parent["sha256"] if parent else "",
        "bert_model_name": bert_model_name,
        "save_dir": str(save_dir),
        "script": str(SCRIPT_FOR[stage](reader, task).relative_to(REPO_ROOT)).replace("\\", "/"),
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
        METRIC_NOTE[run.get("task", DEFAULT_TASK)],
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
            "task": run.get("task", DEFAULT_TASK),
            "file_path": best.as_posix(),
            "sha256": manifest.sha256_file(best),
            "parent_id": run["parent_id"],
            "data": run["data"],
            "seed": run["seed"],
            "val_auroc": "" if scores.auroc != scores.auroc else f"{scores.auroc:.4f}",
            "val_auprc": "" if scores.auprc != scores.auprc else f"{scores.auprc:.4f}",
            "notes": "; ".join(notes),
            "bert_model_name": run.get("bert_model_name", ""),
        }
    )
    run["manifest_row"] = row
    Path(run["save_dir"], "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return row


#: Manifest note naming the metric, per task (phenotyping text unchanged).
METRIC_NOTE = {
    "phenotyping": "metric=pneumonia (class 21), val split",
    "in-hospital-mortality": "metric=in-hospital mortality (1 class), val split",
}


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


def describe(run: dict) -> str:
    """Human-readable dry-run report: the parent check and the exact command.

    By the time this runs, plan() has already verified the parent (stage,
    reader, data kind, file present, sha256), so reaching here means it passed.
    """
    lines = [
        f"[dry-run] {run['id']}  stage {run['stage']}  reader {run['reader']}  "
        f"task {run.get('task', DEFAULT_TASK)}  data {run['data']}  (nothing will be trained)",
        f"  script        : {run['script']}",
    ]
    if run["parent_id"]:
        parent = manifest.find(run["parent_id"]) or {}
        lines += [
            f"  parent        : {run['parent_id']}  (stage {parent.get('stage', '?')}, "
            f"{parent.get('task', '?')}, {parent.get('data', '?')} data, "
            f"by {parent.get('who') or '?'})",
            f"  parent file   : {run['parent_file']}",
            f"  parent sha256 : {run['parent_sha256']}  -- matches the file: OK",
        ]
    else:
        lines.append("  parent        : none (r1)")
    if run["bert_model_name"]:
        source = "inherited from parent" if run["parent_id"] else "recorded for this r1"
        lines.append(f"  BERT          : {run['bert_model_name']}  ({source})")
    lines.append(f"  save_dir      : {run['save_dir']}")
    if run["deviations"]:
        lines.append(f"  deviations    : {'; '.join(run['deviations'])}")
    lines.append("  argv (cwd = medpatch/):")
    argv = ["python", "fusion_main.py", *run["argv"]]
    pairs = to_pairs(run["argv"])
    lines.append("    python fusion_main.py \\")
    for i, (flag, value) in enumerate(pairs):
        end = " \\" if i < len(pairs) - 1 else ""
        lines.append(f"      {flag}{'' if value is None else ' ' + shlex.quote(value)}{end}")
    lines.append(f"  as one line   : {shlex.join(argv)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.pv.run", description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=manifest.STAGES)
    p.add_argument("--reader", choices=READERS, required=True)
    p.add_argument(
        "--task",
        choices=sorted(TASK_ALIASES),
        default=DEFAULT_TASK,
        help="phenotyping (default) or mortality (= in-hospital-mortality; readers ehr, cxr, "
        "rr -- dn is refused: discharge notes leak the outcome)",
    )
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
    p.add_argument(
        "--normalizer-state",
        dest="normalizer_state",
        help="real data only: EHR normalizer file to pass to fusion_main.py. Omit to use "
        "fusion_main's default (what a Round 1 run with normalizer_state None used); set it "
        "only to match a Round 1 run that set one (check its args.txt).",
    )
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
    p.add_argument(
        "--bert-model-name",
        dest="bert_model_name",
        help="rr/dn only. r2/r2b inherit the parent's BERT; a different value is refused. "
        "For r1 it is recorded (default: medpatch's Bio_ClinicalBERT).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="verify the parent and print the exact fusion_main command, then exit (no training)",
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, str] | None:
    args = build_parser().parse_args(argv)
    run = plan(args)
    if args.dry_run:
        print(describe(run))
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
