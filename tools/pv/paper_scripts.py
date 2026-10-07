"""Read the paper's SLURM scripts and turn them into fusion_main.py arguments.

The commands are *derived* from ``medpatch/scripts/<task>/{Unimodal,
Confidence,Calibrate}/*.sh`` rather than retyped, so every hyper-parameter the
authors set (lr, classes, encoders, output dims, data_pairs, use_cls_token, ...)
is kept -- per task: the mortality scripts set their own num_classes (1), task
and labels_set. Only machine-specific values -- data directories, save_dir, the
checkpoint to load -- and explicitly recorded smoke-test overrides are replaced.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from .medpatch_bridge import MEDPATCH

SCRIPTS_ROOT = MEDPATCH / "scripts"
#: Kept for callers that predate tasks; the phenotyping scripts.
SCRIPTS = SCRIPTS_ROOT / "phenotyping"

READERS = ("ehr", "cxr", "rr", "dn")

#: medpatch task name (as passed to --task and stored in the manifest) for each
#: name accepted on the command line.
TASK_ALIASES = {
    "phenotyping": "phenotyping",
    "mortality": "in-hospital-mortality",
    "in-hospital-mortality": "in-hospital-mortality",
}
TASKS = ("phenotyping", "in-hospital-mortality")
DEFAULT_TASK = "phenotyping"
#: medpatch/scripts/<folder> holding each task's paper scripts.
SCRIPT_DIR = {"phenotyping": "phenotyping", "in-hospital-mortality": "mortality"}
#: Readers each task trains. Mortality has no DN reader (see refuse_reader).
READERS_FOR_TASK = {"phenotyping": READERS, "in-hospital-mortality": ("ehr", "cxr", "rr")}

DN_LEAKS_OUTCOME = (
    "DN (discharge notes) is not a mortality reader: discharge notes leak the outcome "
    "(they are written after death or discharge). Mortality readers are ehr, cxr and rr."
)


def canonical_task(task: str | None) -> str:
    """medpatch's name for a task given on the command line ("mortality" ok)."""
    name = TASK_ALIASES.get(task or DEFAULT_TASK)
    if name is None:
        raise ValueError(f"unknown task {task!r}; use one of {sorted(TASK_ALIASES)}")
    return name


def refuse_reader(reader: str, task: str) -> None:
    """Raise ValueError if ``reader`` is not trained for ``task``."""
    task = canonical_task(task)
    if reader == "dn" and task == "in-hospital-mortality":
        raise ValueError(DN_LEAKS_OUTCOME)
    if reader not in READERS_FOR_TASK[task]:
        raise ValueError(f"reader {reader!r} is not trained for task {task!r}")


def script_path(stage: str, reader: str, task: str | None = None) -> Path:
    """The paper script for one stage, reader and task."""
    task = canonical_task(task)
    refuse_reader(reader, task)
    base = SCRIPTS_ROOT / SCRIPT_DIR[task]
    name = {
        "r1": ("Unimodal", f"{reader.upper()}.sh"),
        "r2": ("Confidence", f"Confidence-{reader.upper()}.sh"),
        # Round 2b: temperature calibration. fusion_main.py routes temp_c-unimodal_*
        # to trainers/Calibration.py. Calibrate-DN.sh already uses --load_dn.
        "r2b": ("Calibrate", f"Calibrate-{reader.upper()}.sh"),
    }[stage]
    return base / name[0] / name[1]


#: Round -> the paper script for a reader (and optionally a task; default
#: phenotyping, so existing callers are unchanged).
SCRIPT_FOR = {
    stage: (lambda reader, task=None, _stage=stage: script_path(_stage, reader, task))
    for stage in ("r1", "r2", "r2b")
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
    stage: str,
    reader: str,
    overrides: dict[str, str | None],
    drop: set[str] | None = None,
    task: str | None = None,
) -> list[str]:
    """Paper script arguments with ``overrides`` applied.

    ``overrides`` maps a flag to its new value (None for a bare switch such as
    --resume). Flags in ``drop`` are removed. Any ``--load_*`` in the script is
    always removed; the caller supplies the correct one via ``overrides``.
    """
    pairs = to_pairs(read_script_args(script_path(stage, reader, task)))
    drop = (drop or set()) | (_ALL_LOAD_FLAGS - set(overrides))
    kept = [(f, v) for f, v in pairs if f not in drop and f not in overrides]
    argv: list[str] = []
    for flag, value in [*kept, *overrides.items()]:
        argv.append(flag)
        if value is not None:
            argv.append(value)
    return argv


def script_settings(stage: str, reader: str, task: str | None = None) -> dict[str, str | None]:
    return dict(to_pairs(read_script_args(script_path(stage, reader, task))))
