"""Which readers' weights each Round 2b parent file carries (key counts only).

    python -m tools.pv.parent_keys --task phenotyping       # r2b ehr, cxr, rr, dn
    python -m tools.pv.parent_keys --task mortality         # r2b ehr, cxr, rr
    python -m tools.pv.parent_keys --task phenotyping --parents r2b-ehr-001 r2b-cxr-001 ...

For each parent (default: the latest real r2b row per reader of the task) it
checks the sha256 against the manifest, then prints, per key prefix, how many
state_dict entries the file holds, and how many of them the Round 3 loader
takes as that reader's own keys (medpatch Trainer.READER_KEYS, F2) versus
ignores. Overlap -- a file carrying another reader's encoder -- shows up as
ignored prefixes such as ``ehr_model``. Prints names and counts, never values.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import (
    manifest,  # noqa: E402
    medpatch_bridge,  # noqa: E402,F401 -- puts medpatch/ on sys.path
)
from tools.pv.import_checkpoint import load_checkpoint  # noqa: E402
from tools.pv.paper_scripts import READERS_FOR_TASK, TASK_ALIASES, canonical_task  # noqa: E402


def prefix(key: str) -> str:
    """The module a key belongs to: 'fusion_model.rr_classifier.w' -> 'fusion_model.rr_classifier',
    'ehr_model.w' -> 'ehr_model'."""
    parts = key.split(".")
    return ".".join(parts[:2]) if parts[0] == "fusion_model" and len(parts) > 1 else parts[0]


def describe(reader: str, row_id: str, state: dict) -> tuple[str, int, int]:
    from trainers.trainer import Trainer  # noqa: PLC0415

    taken = Counter(prefix(k) for k in state if Trainer.reader_key(k, reader))
    ignored = Counter(prefix(k) for k in state if not Trainer.reader_key(k, reader))
    lines = [f"[{reader}] {row_id}: {len(state)} keys"]
    for name, count in sorted(taken.items()):
        lines.append(f"    taken    {count:>6}  {name}")
    for name, count in sorted(ignored.items()):
        lines.append(f"    IGNORED  {count:>6}  {name}")
    return "\n".join(lines), sum(taken.values()), sum(ignored.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.pv.parent_keys", description=__doc__)
    p.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    p.add_argument("--data", choices=manifest.DATA_KINDS, default="real")
    p.add_argument("--parents", nargs="+", help="r2b manifest ids (default: latest per reader)")
    p.add_argument("--trust-pickle", action="store_true")
    args = p.parse_args(argv)
    task = canonical_task(args.task)
    if args.parents:
        rows = [manifest.find(i) for i in args.parents]
        missing = [i for i, r in zip(args.parents, rows, strict=True) if r is None]
        if missing:
            raise SystemExit(f"not in {manifest.manifest_path()}: {missing}")
    else:
        rows = [manifest.latest("r2b", r, args.data, task=task) for r in READERS_FOR_TASK[task]]
        if None in rows:
            raise SystemExit(f"no {args.data} r2b row for every reader of {task}")
    for row in rows:
        if row["stage"] != "r2b" or row["reader"] not in READERS_FOR_TASK[task]:
            raise SystemExit(f"{row['id']} is not an r2b row of a {task} reader")
        path = Path(row["file_path"])
        sha_ok = manifest.sha256_file(path) == row["sha256"]
        print(f"{row['id']}  sha256 {'matches the manifest' if sha_ok else 'MISMATCH'}")
        if not sha_ok:
            return 1
        state = load_checkpoint(path, args.trust_pickle)["state_dict"]
        text, _, _ = describe(row["reader"], row["id"], state)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
