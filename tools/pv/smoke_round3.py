"""Round 3 end to end on virtual data: pre-flight, cache, equivalence, train, record, checks.

    python -m tools.pv.smoke_round3 --task phenotyping
    python -m tools.pv.smoke_round3 --task mortality
    python -m tools.pv.smoke_round3 --task phenotyping --skip-parents   # require r2b rows

Steps:
  1. the virtual r2b parents of the task (runs smoke_round2b for the missing
     ones unless --skip-parents);
  2. the Round 3 virtual dataset (data/virtual/smoke-r3[-mortality]): some
     stays without radiology reports / discharge notes, one validation X-ray
     missing on disk;
  3. r3_preflight (with --missing-image-as-absent);
  4. run r3, every step, 3 epochs, a batch size that leaves a final validation
     batch of one stay, the without-missingness ablation and the two others;
  5. the checks: pytest -m round3.

Everything it produces is SYNTHETIC -- not a scientific result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import logtee, manifest, r3_preflight, round3  # noqa: E402
from tools.pv.medpatch_bridge import REPO_ROOT  # noqa: E402
from tools.pv.paper_scripts import READERS_FOR_TASK, canonical_task  # noqa: E402


def batch_size_for(n: int) -> int:
    """A batch size that leaves a last batch of exactly one stay (n = k * b + 1)."""
    for b in range(4, 17):
        if n > b and (n - 1) % b == 0:
            return b
    return max(2, n - 1)


def _run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.smoke_round3", description=__doc__.splitlines()[0]
    )
    p.add_argument("--task", choices=("phenotyping", "mortality"), default="phenotyping")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--skip-parents", action="store_true", help="require existing virtual r2b rows")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args(argv)
    task = canonical_task(args.task)
    started, timings = time.time(), []

    def step(name, fn):
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        result = fn()
        timings.append((name, time.time() - t))
        return result

    missing = [
        r for r in READERS_FOR_TASK[task] if not manifest.latest("r2b", r, "virtual", task=task)
    ]
    if missing:
        if args.skip_parents:
            raise SystemExit(f"no virtual r2b row for {missing} ({task}); run smoke_round2b first")
        from tools.pv import smoke_round2b  # noqa: PLC0415

        step("parents: smoke_round2b", lambda: smoke_round2b.main(
            ["--task", args.task, "--skip-r1", "--skip-r2", "--readers", *missing]))  # fmt: skip

    suffix = "" if task == "phenotyping" else "-mortality"
    root = REPO_ROOT / "data" / "virtual" / f"smoke-r3{suffix}"
    if args.regenerate or not (root / "README.md").is_file():
        from tools.synthetic.make_virtual_dataset import (  # noqa: PLC0415
            generate,
            generate_mortality,
            make_round3_variant,
        )

        build = generate if task == "phenotyping" else generate_mortality
        step("generate Round 3 virtual dataset", lambda: (
            build(root, "smoke", args.seed),
            make_round3_variant(root, task, args.seed)))  # fmt: skip
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    batch = batch_size_for(summary["split_sizes"]["val"])
    common = [
        "--task", args.task, "--data", "virtual", "--data-root", str(root),
        "--batch-size", str(batch), "--missing-image-as-absent",
    ]  # fmt: skip
    code = step("r3_preflight", lambda: r3_preflight._run([*common, "--sample", "32"]))
    if code:
        return code
    row = step("run r3 (cache, equivalence, train + ablations, record)", lambda: round3.main([
        *common, "--epochs", str(args.epochs), "--shard-size", str(batch * 2),
        "--equivalence-batches", "2", "--extra-ablations", "--who", "smoke_round3",
        "--cache-dir", str(REPO_ROOT / "runs" / "r3_cache" / "virtual" / task),
    ]))  # fmt: skip
    step("run r3 --step ablations (after the row exists)", lambda: round3.main([
        *common, "--epochs", str(args.epochs), "--run-id", row["id"], "--step", "ablations",
        "--variants", "late_weights_per_class", "mask_missing_tokens",
        "--cache-dir", str(REPO_ROOT / "runs" / "r3_cache" / "virtual" / task),
    ]))  # fmt: skip
    code = step("checks: pytest -m round3", lambda: subprocess.call(
        [sys.executable, "-m", "pytest", "-m", "round3", "-s", "-p", "no:warnings", "tests/round3"],
        cwd=REPO_ROOT, env={**os.environ, "PV_CHECK_TASK": task},
    ))  # fmt: skip

    print("\n=== summary (SYNTHETIC -- not a scientific result) ===")
    print(f"  {row['id']}  parents {row['parent_id']}  val AUROC {row['val_auroc']}  "
          f"AUPRC {row['val_auprc']}  (batch {batch}: last val batch of one)")  # fmt: skip
    for name, seconds in timings:
        print(f"  {int(seconds // 60)}m{int(seconds % 60):02d}s  {name}")
    print(f"  total {int((time.time() - started) // 60)} min   checks "
          f"{'PASSED' if code == 0 else 'FAILED'}   "
          f"manifest: {manifest.manifest_path()}")  # fmt: skip
    return code


NAME = "smoke_round3"


def _log_path(argv: list[str], name: str) -> tuple[Path, list[str]]:
    """--log PATH (default <RUNS_ROOT>/logs/<name>_<task>_<time>.log) and the other args."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--log", type=Path)
    known, rest = pre.parse_known_args(argv)
    task = argv[argv.index("--task") + 1] if "--task" in argv[:-1] else "phenotyping"
    return known.log or logtee.default_log(f"{name}_{task}"), rest


def main(argv: list[str] | None = None) -> int:
    """_run, with the whole output (subprocesses and any traceback too) in a log file."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path, rest = _log_path(argv, NAME)
    with logtee.tee_to_log(path), mock.patch.object(subprocess, "call", logtee.run_logged):
        return _run(rest)


if __name__ == "__main__":
    sys.exit(main())
