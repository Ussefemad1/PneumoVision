"""Write a command's whole output to a log file as well as the screen.

A Colab cell truncates long output, so a failure can scroll out of reach. With
``tee_to_log(path)`` everything printed (stdout and stderr), the traceback of a
crash, and -- through ``run_logged`` -- the output of subprocesses, also goes
to ``path``, whose location is printed at the start and at the end.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

from tools.pv import manifest


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ...
        return getattr(self.stream, name)


def default_log(name: str) -> Path:
    """<RUNS_ROOT>/logs/<name>_<UTC time>.log (RUNS_ROOT is on Drive on Colab)."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return manifest.runs_root() / "logs" / f"{name}_{stamp}.log"


@contextlib.contextmanager
def tee_to_log(path: Path) -> Iterator[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out, err = sys.stdout, sys.stderr
    with open(path, "a", encoding="utf-8", errors="replace") as log:
        sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
        print(f"[log] full output also in {path}", flush=True)
        try:
            yield path
        except BaseException:
            traceback.print_exc()  # into the log as well as the cell
            raise
        finally:
            sys.stdout.flush()
            sys.stdout, sys.stderr = out, err
            print(f"[log] full output in {path}", flush=True)


def run_logged(cmd, cwd=None, env=None, **_ignored) -> int:
    """subprocess.call, with the child's output passed through sys.stdout (and the log)."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    return proc.wait()
