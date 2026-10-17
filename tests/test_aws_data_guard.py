"""infra/aws/data-guard.sh: the deploy refuses a server holding MIMIC-derived data.

Runs the real script on a synthetic directory tree (empty placeholder files,
named like the off-limits ones). No real data is created or read.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "infra" / "aws" / "data-guard.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

DEMO_CSVS = [
    "mortality_high_risk_vitals.csv",
    "mortality_low_risk_vitals.csv",
    "pneumonia_control_vitals.csv",
    "pneumonia_sick_vitals.csv",
]


def tree(root: Path, extra: list[str] = ()) -> Path:
    """A clean server checkout: code, the 4 demo CSVs, the Wikimedia JPGs, weights."""
    files = [
        "medpatch/models/fusion.py",
        "medpatch/ehr_utils/resources/discretizer_config.json",
        "medpatch/normalizers/ph_ts1.0.input_str_previous.start_time_zero.normalizer",
        "docs/demo-examples/cxr_pneumonia_lobar.jpg",
        "apps/web/e2e/fixtures/synthetic-cxr.png",
        "data/.gitkeep",
        "checkpoints/.gitkeep",
        "checkpoints/round3/mortality/run.json",
        "checkpoints/round3/mortality/best_checkpoint_0.0001_in-hospital-mortality_c-msma_EHR-CXR-RR_partial.pth.tar",
        "node_modules/some-package/fixture.csv",  # dependency trees are skipped
        *[f"docs/demo-examples/{name}" for name in DEMO_CSVS],
        *extra,
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


def guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(GUARD), str(root)], capture_output=True, text=True, timeout=60
    )


def test_a_clean_synthetic_checkout_passes(tmp_path):
    res = guard(tree(tmp_path))
    assert res.returncode == 0, res.stderr
    assert "data-guard: OK" in res.stdout


@pytest.mark.parametrize(
    "bad",
    [
        "medpatch/mimic4extract/mimic3benchmark/resources/itemid_to_variable_map.csv",
        "medpatch/ehr_utils/mimic-cxr-ehr-split.csv",
        "medpatch/ehr_utils/resources/valset.csv",
        "data/root/train_listfile.csv",
        "phenotyping/test_listfile.csv",
        "notes/mimic-cxr-2.0.0-metadata.csv",
        "notes/discharge.csv",
        "notes/radiology.csv",
        "tools/cache/stays.pkl",
        "some/other/table.csv",
        "exports/ehr.parquet",
        "docs/demo-examples/sick cxr.png",
        "checkpoints/round3/phenotyping/r3_results.csv",
        "checkpoints/round3/phenotyping/train.log",
        "data/anything.txt",
    ],
)
def test_off_limits_files_are_refused(tmp_path, bad):
    res = guard(tree(tmp_path, [bad]))
    assert res.returncode == 1
    assert "REFUSED" in res.stderr
    assert bad in res.stderr


def test_a_full_clone_is_refused(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("needs git")
    root = tree(tmp_path)
    subprocess.run([git, "init", "-q", str(root)], check=True)
    res = guard(root)
    assert res.returncode == 1
    assert "not a sparse checkout" in res.stderr

    subprocess.run([git, "-C", str(root), "sparse-checkout", "init", "--no-cone"], check=True)
    res = guard(root)
    assert res.returncode == 0, res.stderr
