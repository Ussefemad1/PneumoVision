"""The virtual dataset loads through medpatch's own dataset classes.

Synthetic data only -- generated into a temp folder by
tools/synthetic/make_virtual_dataset.py. No MIMIC file is opened.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools.pv.medpatch_bridge import build_loaders, parse_args
from tools.pv.paper_scripts import READERS, build_argv
from tools.synthetic.make_virtual_dataset import PHENOTYPES, PNEUMONIA, generate


@pytest.fixture(scope="module")
def virtual(tmp_path_factory):
    out = tmp_path_factory.mktemp("virtual") / "smoke"
    summary = generate(out, "smoke", seed=0)
    return out, summary


def data_overrides(root) -> dict[str, str]:
    return {
        "--ehr_data_dir": str(root / "ehr"),
        "--cxr_data_dir": str(root / "cxr"),
        "--notes_data_dir": str(root / "notes"),
        "--normalizer_state": str(root / "ehr" / "ph_ts1.0.virtual.normalizer"),
        "--save_dir": str(root / "_unused"),
        "--num_workers": "0",
        "--batch_size": "4",
    }


def test_layout_and_stamp(virtual):
    root, summary = virtual
    for rel in [
        "ehr/root/all_stays.csv",
        "ehr/phenotyping/train_listfile.csv",
        "ehr/phenotyping/val_listfile.csv",
        "ehr/phenotyping/test_listfile.csv",
        "cxr/mimic-cxr-2.0.0-metadata.csv",
        "cxr/mimic-cxr-2.0.0-chexpert.csv",
        "cxr/mimic-cxr-ehr-split.csv",
        "notes/radiology.csv",
        "notes/discharge.csv",
    ]:
        assert (root / rel).is_file(), rel
    assert "SYNTHETIC" in (root / "README.md").read_text(encoding="utf-8")
    assert list(root.glob("cxr/resized/**/*.jpg"))

    listfile = pd.read_csv(root / "ehr/phenotyping/train_listfile.csv")
    assert list(listfile.columns[3:]) == PHENOTYPES
    metadata = pd.read_csv(root / "cxr/mimic-cxr-2.0.0-metadata.csv")
    assert set(metadata.ViewPosition) == {"AP"}
    split = pd.read_csv(root / "cxr/mimic-cxr-ehr-split.csv")
    assert list(split.columns) == ["dicom_id", "study_id", "subject_id", "split"]
    assert set(split.split) <= {"train", "validate", "test"}


def test_presets_match_the_brief(virtual):
    _, summary = virtual
    assert abs(summary["pneumonia_prevalence"] - 0.12) < 0.03
    small = generate.__globals__["PRESETS"]["small"]
    assert small.cxr_rate == pytest.approx(0.18)


def test_studies_fall_inside_their_stays(virtual):
    root, _ = virtual
    stays = pd.read_csv(root / "ehr/root/all_stays.csv", parse_dates=["intime", "outtime"])
    meta = pd.read_csv(root / "cxr/mimic-cxr-2.0.0-metadata.csv")
    taken = pd.to_datetime(
        meta.StudyDate.astype(str) + " " + meta.StudyTime.map(lambda x: f"{int(float(x)):06}"),
        format="%Y%m%d %H%M%S",
    )
    merged = meta.assign(taken=taken).merge(stays, on="subject_id")
    assert ((merged.taken >= merged.intime) & (merged.taken <= merged.outtime)).all()


def test_generation_is_seeded(tmp_path):
    a = generate(tmp_path / "a", "smoke", seed=3)
    b = generate(tmp_path / "b", "smoke", seed=3)
    assert a == b
    for rel in ["ehr/phenotyping/train_listfile.csv", "notes/radiology.csv"]:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


@pytest.mark.parametrize("reader", READERS)
def test_loads_through_medpatch_datasets(virtual, reader):
    root, _ = virtual
    args = parse_args(build_argv("r1", reader, data_overrides(root)))
    train_dl, val_dl, test_dl = build_loaders(args)
    assert len(train_dl.dataset) > 0 and len(val_dl.dataset) > 0

    x, img, dn, rr, y_ehr, _y_cxr, seq_lengths, *_ = next(iter(val_dl))
    assert x.shape[-1] == 76  # discretizer width
    assert y_ehr.shape[-1] == len(PHENOTYPES)
    assert len(seq_lengths) == x.shape[0]
    if reader == "cxr":
        assert tuple(img.shape[1:]) == (3, 384, 384)
    if reader == "rr":
        assert all(isinstance(t, str) and t for t in rr)
    if reader == "dn":
        assert all(isinstance(t, str) and t for t in dn)


def test_pneumonia_is_present_in_every_split(virtual):
    root, _ = virtual
    for split in ("train", "val", "test"):
        labels = pd.read_csv(root / f"ehr/phenotyping/{split}_listfile.csv").iloc[:, 3:]
        assert labels.iloc[:, PNEUMONIA].sum() >= 1, split


def test_summary_is_json(virtual):
    root, _ = virtual
    assert json.loads((root / "summary.json").read_text())["stays"] == 60
    assert np.isfinite(json.loads((root / "summary.json").read_text())["cxr_available"])
