"""In-process access to medpatch's own data pipeline and trainer.

``fusion_main.py`` is a script (all top-level code), so it cannot be imported.
This module repeats its *wiring* -- discretizer, normalizer, dataset and
dataloader construction, MSMA_Trainer -- by calling the same medpatch
functions, so tests and checks see exactly what a training run sees. No
preprocessing is reimplemented here.

medpatch resolves some resources relative to the working directory
(``ehr_utils/resources/discretizer_config.json``), so everything that touches
it runs inside :func:`medpatch_cwd`.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDPATCH = REPO_ROOT / "medpatch"

if str(MEDPATCH) not in sys.path:
    sys.path.insert(0, str(MEDPATCH))

# medpatch's trainers call wandb.init unconditionally; never phone home from tooling.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")


@contextlib.contextmanager
def medpatch_cwd() -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(MEDPATCH)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args(argv: list[str]):
    """Parse a fusion_main.py argument list with medpatch's own parser."""
    from arguments import args_parser  # noqa: PLC0415

    return args_parser().parse_args(argv)


def normalizer_state_path(args) -> Path:
    """The normalizer file a run uses -- the same resolution as fusion_main.py.

    fusion_main.py (lines 105-108): an explicit --normalizer_state, else the
    bundled phenotyping file for the timestep,
    medpatch/normalizers/ph_ts{timestep}.input_str_previous.start_time_zero.normalizer.
    Real runs pass no --normalizer_state, so scoring must resolve the default
    the same way; loading `None` raised TypeError after training had finished.
    """
    if args.normalizer_state:
        return Path(args.normalizer_state)
    return (
        MEDPATCH
        / "normalizers"
        / f"ph_ts{args.timestep}.input_str_previous.start_time_zero.normalizer"
    )


def build_loaders(args):
    """(train_dl, val_dl, test_dl) exactly as fusion_main.py builds them."""
    import numpy as np  # noqa: PLC0415
    from datasets.cxr_dataset import get_cxr_datasets  # noqa: PLC0415
    from datasets.DataFusion import load_cxr_ehr_rr_dn  # noqa: PLC0415
    from datasets.ehr_dataset import get_datasets  # noqa: PLC0415
    from ehr_utils.preprocessing import Discretizer, Normalizer  # noqa: PLC0415

    with medpatch_cwd():
        discretizer = Discretizer(
            timestep=float(args.timestep),
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
        )
        train_dir = Path(args.ehr_data_dir) / args.task / "train"
        sample = sorted(train_dir.glob("*_timeseries.csv"))[0]
        with open(sample) as f:
            assert f.readline().strip().split(",")[0] == "Hours"
            rows = np.stack([np.array(line.strip().split(",")) for line in f])
        header = discretizer.transform(rows)[1].split(",")
        cont = [i for i, x in enumerate(header) if x.find("->") == -1]
        normalizer = Normalizer(fields=cont)
        normalizer.load_params(str(normalizer_state_path(args)))

        ehr_train, ehr_val, ehr_test = get_datasets(discretizer, normalizer, args)
        cxr_train, cxr_val, cxr_test = get_cxr_datasets(args)
        return load_cxr_ehr_rr_dn(args, ehr_train, ehr_val, cxr_train, cxr_val, ehr_test, cxr_test)


def build_trainer(args, loaders=None):
    """An MSMA_Trainer (the class fusion_main.py uses for these fusion types).

    Construction seeds torch and applies ``--load_*`` exactly as a real run
    does, so the model is bit-identical to a run's starting point.
    """
    from trainers.MSMA_trainer import MSMA_Trainer  # noqa: PLC0415

    train_dl, val_dl, test_dl = loaders or build_loaders(args)
    with medpatch_cwd():
        return MSMA_Trainer(train_dl, val_dl, args, test_dl)


def fusion_main_argv(argv: list[str]) -> list[str]:
    """The command line for running fusion_main.py itself (cwd must be medpatch/)."""
    return [sys.executable, "fusion_main.py", *argv]
