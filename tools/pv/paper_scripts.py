"""Read the paper's SLURM scripts and turn them into fusion_main.py arguments.

The commands are *derived* from ``medpatch/scripts/phenotyping/{Unimodal,
Confidence}/*.sh`` rather than retyped, so every hyper-parameter the authors set
(lr, classes, encoders, output dims, data_pairs, use_cls_token, ...) is kept.
Only machine-specific values -- data directories, save_dir, the checkpoint to
load -- and explicitly recorded smoke-test overrides are replaced.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from .medpatch_bridge import MEDPATCH

SCRIPTS = MEDPATCH / "scripts" / "phenotyping"

READERS = ("ehr", "cxr", "rr", "dn")

#: Round -> folder and filename pattern of the paper script for each reader.
SCRIPT_FOR = {
    "r1": lambda reader: SCRIPTS / "Unimodal" / f"{reader.upper()}.sh",
    "r2": lambda reader: SCRIPTS / "Confidence" / f"Confidence-{reader.upper()}.sh",
    # Round 2b: temperature calibration. fusion_main.py routes temp_c-unimodal_*
    # to trainers/Calibration.py. Calibrate-DN.sh already uses --load_dn.
    "r2b": lambda reader: SCRIPTS / "Calibrate" / f"Calibrate-{reader.upper()}.sh",
}

#: The flag Round 2 must use to load its Round 1 parent. Confidence-DN.sh passes
#: --load_rr for the DN reader (a copy-paste slip in the upstream script: the
#: architecture being loaded is DN's). --load_dn exists in arguments.py and is
#: what we pass. See docs/model_track_notes.md, entry "DN load flag".
LOAD_FLAG = {"ehr": "--load_ehr", "cxr": "--load_cxr", "rr": "--load_rr", "dn": "--load_dn"}
_ALL_LOAD_FLAGS = set(LOAD_FLAG.values())


def read_script_args(path: Path) -> list[str]:
    """The argument list after ``python fusion_main.py`` in a paper script."""
    text = path.read_text(encoding="utf-8")
    # Join backslash continuations, then find the fusion_main invocation.
    joined = text.replace("\\\r\n", " ").replace("\\\n", " ")
    for line in joined.splitlines():
        tokens = shlex.split(line, comments=True)
        if "fusion_main.py" in tokens:
            return tokens[tokens.index("fusion_main.py") + 1 :]
    raise ValueError(f"no fusion_main.py invocation in {path}")


def to_pairs(args: list[str]) -> list[tuple[str, str | None]]:
    """Split ``--flag value`` / bare ``--flag`` tokens into ordered pairs."""
    pairs: list[tuple[str, str | None]] = []
    i = 0
    while i < len(args):
        flag = args[i]
        if not flag.startswith("--"):
            raise ValueError(f"unexpected token {flag!r}")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            pairs.append((flag, args[i + 1]))
            i += 2
        else:
            pairs.append((flag, None))
            i += 1
    return pairs


def build_argv(
    stage: str, reader: str, overrides: dict[str, str | None], drop: set[str] | None = None
) -> list[str]:
    """Paper script arguments with ``overrides`` applied.

    ``overrides`` maps a flag to its new value (None for a bare switch such as
    --resume). Flags in ``drop`` are removed. Any ``--load_*`` in the script is
    always removed; the caller supplies the correct one via ``overrides``.
    """
    pairs = to_pairs(read_script_args(SCRIPT_FOR[stage](reader)))
    drop = (drop or set()) | (_ALL_LOAD_FLAGS - set(overrides))
    kept = [(f, v) for f, v in pairs if f not in drop and f not in overrides]
    argv: list[str] = []
    for flag, value in [*kept, *overrides.items()]:
        argv.append(flag)
        if value is not None:
            argv.append(value)
    return argv


def script_settings(stage: str, reader: str) -> dict[str, str | None]:
    return dict(to_pairs(read_script_args(SCRIPT_FOR[stage](reader))))
