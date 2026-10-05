"""The whole Round 2 chain on virtual data, in one command, on CPU.

    python -m tools.pv.smoke_round2              # generate -> r1 x4 -> r2 x4 -> checks
    python -m tools.pv.smoke_round2 --skip-r1    # reuse existing r1 stand-ins
    python tools/pv/smoke_round2.py              # same, run as a file path

Steps:
  1. generate the smoke virtual dataset (data/virtual/smoke), unless present;
  2. train the four Round 1 stand-ins (paper Unimodal scripts, short);
  3. run Round 2 for all four readers, each loading its r1 parent from the manifest;
  4. run the acceptance checks: pytest -m round2.

Everything it produces is SYNTHETIC -- not a scientific result.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Run as a file (`python tools/pv/smoke_round2.py`), Python puts tools/pv/ on
# sys.path instead of the repo root, so `tools` is not importable. Add the root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import manifest, run  # noqa: E402
from tools.pv.medpatch_bridge import REPO_ROOT  # noqa: E402
from tools.pv.paper_scripts import (  # noqa: E402
    DEFAULT_TASK,
    READERS,
    READERS_FOR_TASK,
    canonical_task,
)


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.smoke_round2", description=__doc__.splitlines()[0]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regenerate", action="store_true", help="rebuild the virtual dataset")
    p.add_argument(
        "--skip-r1",
        action="store_true",
        help="reuse the latest virtual r1 rows instead of retraining them",
    )
    p.add_argument(
        "--task",
        choices=("phenotyping", "mortality"),
        default="phenotyping",
        help="mortality: readers ehr, cxr, rr on data/virtual/smoke-mortality",
    )
    p.add_argument("--readers", nargs="+", choices=READERS, default=None)
    args = p.parse_args(argv)
    task = canonical_task(args.task)
    readers = args.readers or list(READERS_FOR_TASK[task])
    if task != DEFAULT_TASK and "dn" in readers:
        p.error("dn is not a mortality reader: discharge notes leak the outcome")

    started = time.time()
    timings: list[tuple[str, float]] = []

    def step(name: str, fn):
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        result = fn()
        timings.append((name, time.time() - t))
        return result

    suffix = "" if task == DEFAULT_TASK else "-mortality"
    data_root = REPO_ROOT / "data" / "virtual" / f"smoke{suffix}"
    if args.regenerate or not (data_root / "README.md").is_file():
        from tools.synthetic.make_virtual_dataset import (  # noqa: PLC0415
            generate,
            generate_mortality,
        )

        build = generate if task == DEFAULT_TASK else generate_mortality

        step("generate virtual smoke dataset", lambda: print(build(data_root, "smoke", args.seed)))
    else:
        print(f"using existing virtual dataset {data_root} (--regenerate to rebuild)")

    common = ["--data", "virtual", "--preset", "smoke", "--who", "smoke_round2", "--task", task]
    rows = {}
    for reader in readers:
        if args.skip_r1 and manifest.latest("r1", reader, "virtual", task=task):
            continue
        step(f"r1 stand-in: {reader}", lambda r=reader: run.main(["r1", "--reader", r, *common]))
    for reader in readers:
        rows[reader] = step(
            f"r2: {reader}", lambda r=reader: run.main(["r2", "--reader", r, *common])
        )

    code = step(
        "checks: pytest -m round2",
        lambda: subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "round2",
                "-s",
                "-p",
                "no:warnings",
                "tests/round2",
            ],
            cwd=REPO_ROOT,
            # Check this task's rows only (tests/round2: PV_CHECK_TASK).
            env={**os.environ, "PV_CHECK_TASK": task},
        ),
    )

    print("\n=== summary (SYNTHETIC -- not a scientific result) ===")
    for row in rows.values():
        print(
            f"  {row['id']:<12} parent {row['parent_id']:<12} "
            f"val AUROC {row['val_auroc'] or 'n/a':<7} AUPRC {row['val_auprc'] or 'n/a'}"
        )
    for name, seconds in timings:
        print(f"  {_fmt(seconds):>7}  {name}")
    print(
        f"  total time {_fmt(time.time() - started)}   "
        f"checks {'PASSED' if code == 0 else 'FAILED'}   manifest: {manifest.manifest_path()}"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
