"""Smoke tests: run `pytest` after setup to confirm the environment is usable.

These do not touch MIMIC data -- they only check that the pinned dependency set
imports cleanly and that the project's own modules load.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDPATCH = REPO_ROOT / "medpatch"

# The project modules import each other by top-level name (`from trainers... `),
# so `medpatch/` itself has to be on sys.path.
if str(MEDPATCH) not in sys.path:
    sys.path.insert(0, str(MEDPATCH))


def test_python_is_311():
    assert sys.version_info[:2] == (3, 11), (
        f"Project targets Python 3.11 (see .python-version); found {sys.version.split()[0]}"
    )


@pytest.mark.parametrize(
    "module",
    ["torch", "torchvision", "transformers", "timm", "numpy", "pandas",
     "sklearn", "scipy", "matplotlib", "PIL", "yaml", "einops"],
)
def test_dependency_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    ["arguments",
     "models.cxr_encoder", "models.ehr_encoder", "models.rr_encoder",
     "models.dn_encoder", "models.fusion", "models.text_models",
     "datasets.cxr_dataset", "datasets.ehr_dataset", "datasets.DataFusion",
     "trainers.MSMA_trainer"],
)
def test_project_imports(module):
    importlib.import_module(module)


def test_argument_parser_builds():
    from arguments import args_parser

    args = args_parser().parse_args([])
    assert args.num_workers >= 0
    assert args.task
    assert hasattr(args, "notes_data_dir")


def test_default_normalizer_file_exists():
    """fusion_main.py falls back to this file when --normalizer_state is unset."""
    for timestep in ("1.0", "0.8"):
        target = MEDPATCH / "normalizers" / (
            f"ph_ts{timestep}.input_str_previous.start_time_zero.normalizer"
        )
        assert target.is_file(), f"missing bundled normalizer: {target}"


def test_collate_is_picklable():
    """DataLoader workers on Windows/macOS pickle collate_fn; a lambda would fail."""
    import pickle
    from functools import partial
    from datasets.DataFusion import my_collate

    pickle.loads(pickle.dumps(partial(my_collate, args=None)))


def test_torch_runs_a_forward_pass():
    import torch

    model = torch.nn.Linear(4, 2)
    out = model(torch.zeros(3, 4))
    assert out.shape == (3, 2)
