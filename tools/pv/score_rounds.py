"""Score finished Round 2 / 2b runs on test, comparable with the paper's tables.

    python -m tools.pv.score_rounds --task phenotyping --data real
    python -m tools.pv.score_rounds --task mortality --data real --stages r2b --out scores.csv

For the latest row of each stage and reader of the task (or ``--ids``), loads
the run's own fusion_main arguments (run.json) and scores the best checkpoint
with medpatch's model on the chosen split: macro AUROC / AUPRC over the classes
with both labels present (25 phenotypes; 1 for mortality), and the target
(pneumonia, class 21 / mortality) AUROC / AUPRC with 95% bootstrap CIs
(medpatch's get_model_performance). Token models (r2/r2b) score a stay as the
mean over its real tokens of sigmoid(logit), as tools/pv/evaluate.py does.
AUROC / AUPRC only. Prints numbers, never data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import manifest  # noqa: E402
from tools.pv.evaluate import TARGET_CLASS, load_run, run_args, score_checkpoint  # noqa: E402
from tools.pv.paper_scripts import READERS_FOR_TASK, TASK_ALIASES, canonical_task  # noqa: E402

COLUMNS = ["id", "stage", "reader", "split", "n", "macro_auroc", "macro_auprc", "classes",
           "target_auroc", "target_auroc_lo", "target_auroc_hi", "target_auprc",
           "target_auprc_lo", "target_auprc_hi"]  # fmt: skip


def bootstrap_ci(labels, scores, iters: int) -> tuple[float, float, float, float]:
    """(auroc_lo, auroc_hi, auprc_lo, auprc_hi) from medpatch's get_model_performance."""
    import pandas as pd  # noqa: PLC0415
    from trainers.utils import get_model_performance  # noqa: PLC0415

    df = pd.DataFrame({"y_truth": labels, "y_pred": scores})
    (_, prc_a, prc_b), (_, roc_a, roc_b) = get_model_performance(
        df, label="target", num_iter=iters, verbose=False
    )
    return min(roc_a, roc_b), max(roc_a, roc_b), min(prc_a, prc_b), max(prc_a, prc_b)


def score_row(row: dict[str, str], split: str, iters: int) -> dict:
    run_dir = Path(row["file_path"]).parents[2]
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"{row['id']}: no run.json in {run_dir} (only runs made by tools.pv.run)")
    args = run_args(load_run(run_dir))
    s = score_checkpoint(args, Path(row["file_path"]), split=split)
    lo_roc, hi_roc, lo_prc, hi_prc = bootstrap_ci(s.labels, s.scores, iters)
    return {
        "id": row["id"], "stage": row["stage"], "reader": row["reader"], "split": split,
        "n": len(s.labels), "macro_auroc": s.macro_auroc, "macro_auprc": s.macro_auprc,
        "classes": s.classes_scored, "target_auroc": s.auroc, "target_auroc_lo": lo_roc,
        "target_auroc_hi": hi_roc, "target_auprc": s.auprc, "target_auprc_lo": lo_prc,
        "target_auprc_hi": hi_prc,
    }  # fmt: skip


def table(results: list[dict], task: str) -> str:
    target = "pneumonia" if TARGET_CLASS[task] == 21 else "mortality"
    lines = [f"{'id':<12} {'reader':<6} {'n':>6}  macro AUROC  macro AUPRC  "
             f"{target} AUROC [95% CI]     {target} AUPRC [95% CI]"]  # fmt: skip
    for r in results:
        lines.append(
            f"{r['id']:<12} {r['reader']:<6} {r['n']:>6}  {r['macro_auroc']:>11.3f}  "
            f"{r['macro_auprc']:>11.3f}  {r['target_auroc']:.3f} [{r['target_auroc_lo']:.3f}, "
            f"{r['target_auroc_hi']:.3f}]   {r['target_auprc']:.3f} [{r['target_auprc_lo']:.3f}, "
            f"{r['target_auprc_hi']:.3f}]"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.score_rounds", description=__doc__.splitlines()[0]
    )
    p.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    p.add_argument("--data", choices=manifest.DATA_KINDS, default="real")
    p.add_argument("--stages", nargs="+", choices=("r2", "r2b"), default=["r2", "r2b"])
    p.add_argument("--ids", nargs="+", help="manifest ids instead of the latest per reader")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--bootstrap-iters", dest="iters", type=int, default=1000)
    p.add_argument("--out", type=Path, help="also write the table as CSV")
    args = p.parse_args(argv)
    task = canonical_task(args.task)
    if args.ids:
        rows = [manifest.find(i) for i in args.ids]
        if None in rows:
            raise SystemExit(f"not in {manifest.manifest_path()}: {args.ids}")
    else:
        rows = [
            manifest.latest(stage, reader, args.data, task=task)
            for stage in args.stages
            for reader in READERS_FOR_TASK[task]
        ]
        rows = [r for r in rows if r is not None]
    results = [score_row(r, args.split, args.iters) for r in rows]
    print(f"\n{task}, {args.split} split, AUROC / AUPRC only")
    print(table(results, task))
    if args.out:
        lines = [",".join(COLUMNS)]
        lines += [",".join(str(r[c]) for c in COLUMNS) for r in results]
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return results


if __name__ == "__main__":
    main()
