"""In-hospital mortality in tools/pv -- non-training checks.

Script lookup, task lineage in the manifest, DN refusal, num_classes 1 through
the argv builder, --dry-run output, the shared trainer/scoring shape handling,
and the virtual mortality dataset loading through medpatch's own loaders.
No model is trained or downloaded; parent "checkpoints" are placeholder bytes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
import torch

from tools.pv import import_checkpoint, manifest, run
from tools.pv.evaluate import TARGET_CLASS, _as_2d
from tools.pv.medpatch_bridge import build_loaders, parse_args
from tools.pv.paper_scripts import (
    READERS_FOR_TASK,
    SCRIPT_FOR,
    build_argv,
    canonical_task,
    script_path,
    script_settings,
)
from tools.synthetic.make_virtual_dataset import generate_mortality

MORT = "in-hospital-mortality"
BIOBERT = "dmis-lab/biobert-v1.1"


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    monkeypatch.setenv("PV_MANIFEST", str(path))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    return path


def _row(tmp_path, stage, reader="rr", task=MORT, data="real", bert=BIOBERT):
    f = tmp_path / f"{manifest.next_id(stage, reader)}-{task}.pth.tar"
    f.write_bytes(f"{stage}{reader}{task}".encode())
    return manifest.append_row(
        {
            "id": manifest.next_id(stage, reader),
            "stage": stage,
            "reader": reader,
            "task": task,
            "data": data,
            "file_path": f.as_posix(),
            "sha256": manifest.sha256_file(f),
            "bert_model_name": bert if reader in ("rr", "dn") else "",
        }
    )


def _args(tmp_path, stage="r2", reader="rr", task="mortality", data="real", **kw):
    vroot = tmp_path / "smoke-mortality"
    if data == "virtual":
        vroot.mkdir(exist_ok=True)
        (vroot / "README.md").write_text("SYNTHETIC")
    defaults = dict(
        stage=stage,
        reader=reader,
        task=task,
        data=data,
        parent=None,
        run_id=None,
        preset="smoke",
        data_root=str(vroot) if data == "virtual" else None,
        epochs=None,
        batch_size=None,
        bootstrap_iters=None,
        num_workers=None,
        bert_model_name=None,
        normalizer_state=None,
        ehr_data_dir="/content/data/ehr",
        cxr_data_dir="/content/data/cxr",
        notes_data_dir="/content/data/notes",
    )
    return argparse.Namespace(**{**defaults, **kw})


# ── 1. task-aware script lookup ──────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["r1", "r2", "r2b"])
@pytest.mark.parametrize("reader", ["ehr", "cxr", "rr"])
def test_mortality_scripts_are_read_from_their_own_files(stage, reader):
    path = script_path(stage, reader, "mortality")
    assert path.parent.parent.name == "mortality" and path.is_file()
    settings = script_settings(stage, reader, "mortality")
    # The mortality scripts' own values -- not phenotyping's.
    assert settings["--task"] == MORT
    assert settings["--labels_set"] == "mortality"
    assert settings["--num_classes"] == "1"
    assert settings["--data_pairs"] == "paired"
    assert (settings["--epochs"], settings["--batch_size"], settings["--lr"]) == (
        "100",
        "16",
        "0.001",
    )
    assert script_settings(stage, reader)["--num_classes"] == "25"  # phenotyping default


def test_task_names():
    assert canonical_task("mortality") == canonical_task(MORT) == MORT
    assert canonical_task(None) == "phenotyping"
    assert READERS_FOR_TASK[MORT] == ("ehr", "cxr", "rr")
    assert SCRIPT_FOR["r2"]("cxr") == script_path("r2", "cxr", "phenotyping")


# ── 2. DN is refused for mortality everywhere ────────────────────────────────


@pytest.mark.parametrize("stage", ["r1", "r2", "r2b"])
def test_dn_script_lookup_refused(stage):
    with pytest.raises(ValueError, match="discharge notes leak the outcome"):
        script_path(stage, "dn", "mortality")


def test_dn_import_refused(tmp_manifest, tmp_path):
    with pytest.raises(SystemExit, match="discharge notes leak the outcome"):
        import_checkpoint.main(
            [
                "--reader",
                "dn",
                "--task",
                "mortality",
                "--who",
                "x",
                "--file",
                str(tmp_path / "never-opened.pth.tar"),
                "--bert-model-name",
                BIOBERT,
            ]
        )


@pytest.mark.parametrize("stage", ["r1", "r2", "r2b"])
def test_dn_run_refused(tmp_manifest, tmp_path, stage):
    with pytest.raises(manifest.ManifestError, match="discharge notes leak the outcome"):
        run.plan(_args(tmp_path, stage=stage, reader="dn"))


# ── 3. manifest task column ──────────────────────────────────────────────────


def test_rows_without_a_task_read_as_phenotyping_and_survive_an_upgrade(tmp_manifest, tmp_path):
    old = [c for c in manifest.COLUMNS if c != "bert_model_name"]
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow(
            {c: "" for c in old}
            | {"id": "r1-ehr-001", "stage": "r1", "reader": "ehr", "data": "virtual"}
        )
    assert manifest.read_rows()[0]["task"] == "phenotyping"
    assert manifest.latest("r1", "ehr", task="phenotyping")["id"] == "r1-ehr-001"
    assert manifest.latest("r1", "ehr", task=MORT) is None
    _row(tmp_path, "r1", reader="ehr")  # appends -> header upgrade
    rows = manifest.read_rows()
    assert [(r["id"], r["task"]) for r in rows] == [
        ("r1-ehr-001", "phenotyping"),
        ("r1-ehr-002", MORT),
    ]
    with open(tmp_manifest, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == manifest.COLUMNS


# ── 4. a parent of the other task is refused ─────────────────────────────────


@pytest.mark.parametrize("parent_task, run_task", [("phenotyping", MORT), (MORT, "phenotyping")])
def test_parent_task_mismatch_refused(tmp_manifest, tmp_path, parent_task, run_task):
    parent = _row(tmp_path, "r1", task=parent_task)
    with pytest.raises(manifest.ManifestError, match="Mixing tasks is refused"):
        manifest.verify_parent(parent, "rr", "real", task=run_task)
    with pytest.raises(manifest.ManifestError, match="Mixing tasks is refused"):
        run.plan(_args(tmp_path, task=run_task, parent=parent["id"]))


def test_mortality_run_never_picks_a_phenotyping_parent(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", task="phenotyping")
    with pytest.raises(manifest.ManifestError, match="No r1 row .* task in-hospital-mortality"):
        run.plan(_args(tmp_path))


# ── 5. num_classes 1 through the argv builder; separate run folders ─────────


def test_mortality_plan_argv(tmp_manifest, tmp_path):
    r1 = _row(tmp_path, "r1", reader="cxr", data="virtual")
    plan = run.plan(_args(tmp_path, reader="cxr", data="virtual"))
    argv = plan["argv"]
    flag = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert flag("--num_classes") == "1"
    assert flag("--task") == MORT and flag("--labels_set") == "mortality"
    assert flag("--fusion_type") == "c-unimodal_cxr" and flag("--modalities") == "EHR-CXR"
    assert flag("--normalizer_state").endswith("ihm_ts1.0.virtual.normalizer")
    assert Path(flag("--load_cxr")) == Path(r1["file_path"])
    assert plan["task"] == MORT and plan["parent_id"] == r1["id"]
    assert plan["script"] == "medpatch/scripts/mortality/Confidence/Confidence-CXR.sh"
    assert Path(plan["save_dir"]).parts[-4:] == (MORT, "r2", "cxr", plan["id"])


def test_phenotyping_run_folder_layout_is_unchanged(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", reader="cxr", task="phenotyping", data="real")
    plan = run.plan(_args(tmp_path, reader="cxr", task="phenotyping"))
    assert Path(plan["save_dir"]).parts[-4:] == ("runs", "r2", "cxr", plan["id"])
    assert "--normalizer_state" not in plan["argv"]


def test_normalizer_state_only_when_given(tmp_manifest, tmp_path):
    _row(tmp_path, "r1", reader="ehr")
    argv = run.plan(_args(tmp_path, reader="ehr", normalizer_state="/content/ihm.normalizer"))[
        "argv"
    ]
    assert argv[argv.index("--normalizer_state") + 1] == "/content/ihm.normalizer"


# ── 6. --dry-run for mortality r2 and r2b ────────────────────────────────────


@pytest.mark.parametrize("stage, parent_stage", [("r2", "r1"), ("r2b", "r2")])
def test_mortality_dry_run(tmp_manifest, tmp_path, capsys, stage, parent_stage):
    parent = _row(tmp_path, parent_stage)
    result = run.main(
        [
            stage,
            "--reader",
            "rr",
            "--task",
            "mortality",
            "--data",
            "real",
            "--ehr-data-dir",
            "/content/data/ehr",
            "--cxr-data-dir",
            "/content/data/cxr",
            "--notes-data-dir",
            "/content/data/notes",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert result is None
    assert f"task {MORT}" in out
    assert f"parent        : {parent['id']}  (stage {parent_stage}, {MORT}" in out
    assert "matches the file: OK" in out and f"BERT          : {BIOBERT}" in out
    assert "--num_classes 1" in out and f"--task {MORT}" in out and "--labels_set mortality" in out
    fusion = "c-unimodal_rr" if stage == "r2" else "temp_c-unimodal_rr"
    assert f"--fusion_type {fusion}" in out
    assert not (tmp_path / "runs").exists()


# ── 7. shared trainer / scoring shape handling with one class ────────────────


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((4, 1), (4, 1)),  # mortality CXR: one CLS token, class axis already squeezed
        ((4, 48), (4, 48)),  # mortality EHR
        ((4, 512), (4, 512)),  # mortality RR
        ((4, 1, 25), (4, 1, 25)),  # phenotyping CXR -- unchanged
        ((4, 512, 25), (4, 512, 25)),  # phenotyping text -- unchanged
        ((4, 48, 1), (4, 48)),  # a 3-D single-class output still drops its class axis
    ],
)
def test_confidence_logits_handles_one_class(shape, expected):
    from trainers.trainer import Trainer  # noqa: PLC0415

    holder = argparse.Namespace(args=argparse.Namespace(fusion_type="c-unimodal_cxr"))
    out = Trainer.confidence_logits(holder, {"c-unimodal_cxr": torch.zeros(shape)})
    assert tuple(out.shape) == expected


def test_scoring_targets_and_label_shapes():
    assert TARGET_CLASS == {"phenotyping": 21, MORT: 0}
    assert tuple(_as_2d([0, 1, 1]).shape) == (3, 1)
    assert tuple(_as_2d([[0] * 25, [1] * 25]).shape) == (2, 25)


def test_calibration_ece_with_one_class():
    from trainers.Calibration import calibration  # noqa: PLC0415

    holder = argparse.Namespace(args=argparse.Namespace(num_classes=1))
    holder.compute_ece = lambda p, lab, n_bins=10: calibration.compute_ece(holder, p, lab, n_bins)
    for shape in [(6, 1), (6, 48), (6, 512)]:  # mortality probs/labels as train_epoch makes them
        probs, labels = torch.rand(shape), (torch.rand(shape) > 0.5).float()
        assert calibration.flat_ece(holder, probs, labels).shape == (1,)


# ── 8. the virtual mortality dataset ─────────────────────────────────────────


@pytest.fixture(scope="module")
def mortality_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("virtual") / "smoke-mortality"
    return out, generate_mortality(out, "smoke", seed=0)


def test_mortality_dataset_layout(mortality_data):
    root, summary = mortality_data
    assert summary["task"] == MORT and "SYNTHETIC" in (root / "README.md").read_text("utf-8")
    listfile = pd.read_csv(root / "ehr" / MORT / "train_listfile.csv")
    assert list(listfile.columns) == ["stay", "period_length", "stay_id", "y_true"]
    assert set(listfile.y_true) <= {0, 1} and (listfile.period_length == 0).all()
    assert pd.read_csv(root / "notes" / "discharge.csv").empty  # no DN text at all
    stays = pd.read_csv(root / "ehr/root/all_stays.csv", parse_dates=["intime"])
    rad = pd.read_csv(root / "notes/radiology.csv", parse_dates=["charttime"]).merge(
        stays, on="subject_id"
    )
    assert (rad.charttime - rad.intime <= timedelta(hours=48)).all()
    assert (root / "ehr" / "ihm_ts1.0.virtual.normalizer").is_file()


@pytest.mark.parametrize("reader", ["ehr", "cxr", "rr"])
def test_mortality_dataset_loads_through_medpatch(mortality_data, reader):
    root, _ = mortality_data
    overrides = {
        "--ehr_data_dir": str(root / "ehr"),
        "--cxr_data_dir": str(root / "cxr"),
        "--notes_data_dir": str(root / "notes"),
        "--normalizer_state": str(root / "ehr" / "ihm_ts1.0.virtual.normalizer"),
        "--save_dir": str(root / "_unused"),
        "--num_workers": "0",
        "--batch_size": "4",
    }
    args = parse_args(build_argv("r2", reader, overrides, task="mortality"))
    train_dl, val_dl, _ = build_loaders(args)
    assert len(train_dl.dataset) > 0 and len(val_dl.dataset) > 0
    x, img, _dn, rr, y_ehr, *_ = next(iter(val_dl))
    assert tuple(x.shape[1:]) == (48, 76)  # the 48-hour window, 76 discretized columns
    assert y_ehr.ndim == 1  # one mortality label per stay
    if reader == "cxr":
        assert tuple(img.shape[1:]) == (3, 384, 384)
    if reader == "rr":
        assert all(isinstance(t, str) and t for t in rr)
