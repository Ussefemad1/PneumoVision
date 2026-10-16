"""Round 3 tooling (tools/pv/round3.py, r3_preflight, score_rounds, generator variant).

No model is built here (that needs BERT/ViT: Colab, tests/round3); planning,
refusals, recording and counting are tested on a scratch manifest and
SYNTHETIC files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools.pv import manifest, r3_cache, r3_preflight, round3, run
from tools.pv.paper_scripts import READERS_FOR_TASK

BIOBERT = "dmis-lab/biobert-v1.1"


@pytest.fixture
def lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MANIFEST", str(tmp_path / "manifest.csv"))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    rows = {}
    for task in ("phenotyping", "in-hospital-mortality"):
        for reader in READERS_FOR_TASK[task]:
            f = tmp_path / f"{task}-{reader}.pth.tar"
            f.write_bytes(f"{task}{reader}".encode())
            rows[(task, reader)] = manifest.append_row({
                "id": manifest.next_id("r2b", reader), "stage": "r2b", "reader": reader,
                "task": task, "data": "virtual", "file_path": f.as_posix(),
                "sha256": manifest.sha256_file(f),
                "bert_model_name": BIOBERT if reader in ("rr", "dn") else "",
            })  # fmt: skip
    for suffix in ("", "-mortality"):
        root = tmp_path / f"smoke-r3{suffix}"
        (root / "ehr").mkdir(parents=True)
        (root / "README.md").write_text("SYNTHETIC")
    return tmp_path, rows


def opts(tmp_path, task="phenotyping", *extra):
    suffix = "" if task == "phenotyping" else "-mortality"
    argv = ["--task", task, "--data", "virtual", "--data-root", str(tmp_path / f"smoke-r3{suffix}"),
            "--cache-dir", str(tmp_path / "cache"), *extra]  # fmt: skip
    return round3.build_parser().parse_args(argv)


def flag(argv, name):
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("task", ["phenotyping", "mortality"])
def test_plan_builds_the_round3_command(lineage, task):
    tmp_path, rows = lineage
    plan = round3.plan(opts(tmp_path, task))
    canonical = "phenotyping" if task == "phenotyping" else "in-hospital-mortality"
    argv = plan["argv"]
    assert flag(argv, "--fusion_type") == "c-msma" and argv.count("--fusion_type") == 1
    assert flag(argv, "--data_pairs") == "partial"
    assert flag(argv, "--bert_model_name") == BIOBERT
    readers = READERS_FOR_TASK[canonical]
    for reader in readers:
        assert Path(flag(argv, f"--load_{reader}")).name == f"{canonical}-{reader}.pth.tar"
        assert flag(argv, f"--{reader}_confidence_threshold") == "0.75"
    assert ("--load_dn" in argv) is (canonical == "phenotyping")
    assert plan["parent_id"] == ";".join(rows[(canonical, r)]["id"] for r in readers)
    assert (
        Path(plan["save_dir"]) == (tmp_path / "runs" / canonical / "r3" / "r3-fusion-001").resolve()
    )
    assert any("cached Round 3" in d for d in plan["deviations"])
    assert any("F5" in d for d in plan["deviations"])


def test_run_dispatches_r3_dry_run(lineage, capsys):
    tmp_path, _ = lineage
    argv = ["r3", "--task", "phenotyping", "--data", "virtual",
            "--data-root", str(tmp_path / "smoke-r3"), "--dry-run"]  # fmt: skip
    assert run.main(argv) is None
    out = capsys.readouterr().out
    assert "[dry-run] r3-fusion-001  stage r3" in out and "fusion c-msma (cached)" in out


def test_entropy_is_refused_unless_allowed(lineage):
    tmp_path, _ = lineage
    with pytest.raises(SystemExit, match="allow-entropy"):
        round3.plan(opts(tmp_path, "phenotyping", "--fusion-type", "c-e-msma"))
    plan = round3.plan(
        opts(tmp_path, "phenotyping", "--fusion-type", "c-e-msma", "--allow-entropy")
    )
    assert flag(plan["argv"], "--fusion_type") == "c-e-msma" and not plan["cached"]


def test_mortality_refuses_a_dn_parent(lineage):
    tmp_path, rows = lineage
    ids = [rows[("in-hospital-mortality", r)]["id"] for r in ("ehr", "cxr", "rr")]
    with pytest.raises(SystemExit, match="leak the outcome"):
        round3.plan(
            opts(tmp_path, "mortality", "--parents", *ids, rows[("phenotyping", "dn")]["id"])
        )


def test_text_parents_must_share_one_bert(lineage):
    tmp_path, rows = lineage
    dn = rows[("phenotyping", "dn")]
    manifest.append_row(
        {**dn, "id": "r2b-dn-099", "bert_model_name": "emilyalsentzer/Bio_ClinicalBERT"}
    )
    ids = [rows[("phenotyping", r)]["id"] for r in ("ehr", "cxr", "rr")] + ["r2b-dn-099"]
    with pytest.raises(SystemExit, match="one bert_model_name"):
        round3.plan(opts(tmp_path, "phenotyping", "--parents", *ids))


def test_flags_reach_the_command_and_the_deviations(lineage):
    tmp_path, _ = lineage
    plan = round3.plan(opts(tmp_path, "phenotyping", "--late-weights-per-class",
                            "--mask-missing-tokens", "--missing-image-as-absent"))  # fmt: skip
    for name in ("--late_weights_per_class", "--mask_missing_tokens", "--missing_image_as_absent"):
        assert name in plan["argv"]
    assert len(plan["deviations"]) >= 6


# ── record ───────────────────────────────────────────────────────────────────


def _metrics(base):
    return {"macro_auroc": base, "macro_auprc": base / 2, "target_auroc": base + 0.1,
            "target_auprc": base / 3, "target_auroc_ci": [base, base + 0.2],
            "target_auprc_ci": [base / 4, base / 2]}  # fmt: skip


def test_record_writes_the_row_notes_and_results(lineage, monkeypatch):
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping")
    plan = round3.plan(o)
    best = Path(plan["save_dir"]) / "best.pth.tar"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"r3")
    plan["cache"] = {"splits": {"val": {"stays": 15, "present": {"EHR": 15, "CXR": 7, "RR": 12,
                     "DN": 12}, "share_high": {"EHR": 0.4, "CXR": 0.9, "RR": 0.2, "DN": 0.3},
                     "images_missing_on_disk": 1}}}  # fmt: skip
    plan["equivalence"] = {"stays": 14, "max_abs_diff_late": 2e-7}

    def fake_score(run_, args, _opts):
        main = args.ablation is None
        return {"checkpoint": str(best), "head": "late" if main else "high",
                "val": _metrics(0.6 if main else 0.5),
                "test": _metrics(0.7 if main else 0.55)}  # fmt: skip

    monkeypatch.setattr(round3, "score_variant", fake_score)
    monkeypatch.setattr(round3, "_best", lambda args: best)
    row = round3.step_record(plan, o, "test")
    assert (row["stage"], row["reader"], row["val_auroc"]) == ("r3", "fusion", "0.7000")
    assert row["parent_id"] == plan["parent_id"] and row["sha256"] == manifest.sha256_file(best)
    for text in ("main: head=late", "ablation without_missingness_module: head=high",
                 "test pneumonia AUROC 0.800 [0.700, 0.900]", "val: 15 stays; present EHR 15",
                 "share>=0.75 EHR 0.400", "1 X-rays missing on disk treated as absent",
                 "cached vs live: max abs diff late 2.0e-07", "cached Round 3"):  # fmt: skip
        assert text in row["notes"], text
    table = (Path(plan["save_dir"]) / "r3_results.csv").read_text().splitlines()
    assert table[1].startswith("variant,chosen_head,split") and len(table) == 2 + 4
    rows_before = manifest.read_rows()
    assert round3.step_record(plan, o, "test")["id"] == row["id"]  # rerun: nothing to do
    assert manifest.read_rows() == rows_before
    best.write_bytes(b"a different checkpoint")
    with pytest.raises(SystemExit, match="different checkpoint"):
        round3.step_record(plan, o, "test")


def test_training_refuses_a_cache_that_does_not_match(lineage, monkeypatch):
    tmp_path, _ = lineage
    plan = round3.plan(opts(tmp_path, "phenotyping"))
    args = SimpleNamespace(task="phenotyping", modalities="EHR-CXR-RR-DN", num_classes=25,
                           data_pairs="partial", ehr_data_dir=str(tmp_path), timestep=1.0,
                           bert_model_name=BIOBERT, batch_size=4,
                           **{f"{m}_confidence_threshold": 0.75
                              for m in ("ehr", "cxr", "rr", "dn")})  # fmt: skip
    with pytest.raises(SystemExit, match="no cache"):
        round3.cache_loaders(plan, args, False)
    saved = r3_cache.fingerprint(args, plan["parents"])
    r3_cache.write_meta(Path(plan["cache_dir"]), {"fingerprint": {**saved, "timestep": 2.0},
                                                   "splits": {}})  # fmt: skip
    with pytest.raises(SystemExit, match="timestep: cache 2.0, now 1.0"):
        round3.cache_loaders(plan, args, False)
    r3_cache.write_meta(Path(plan["cache_dir"]), {"fingerprint": saved, "splits": {}})
    with pytest.raises(SystemExit, match="incomplete"):
        round3.cache_loaders(plan, args, False)


# ── score_rounds / evaluate ──────────────────────────────────────────────────


def test_macro_metrics_skip_classes_without_both_labels():
    from tools.pv.evaluate import macro_metrics  # noqa: PLC0415

    labels = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0]])
    scores = np.array([[0.1, 0.5, 0.2], [0.9, 0.4, 0.3], [0.2, 0.6, 0.1], [0.8, 0.7, 0.4]])
    auroc, auprc, scored = macro_metrics(labels, scores)
    assert (auroc, auprc, scored) == (1.0, 1.0, 1)


def test_score_rounds_bootstrap_ci_brackets_the_estimate():
    from tools.pv.score_rounds import bootstrap_ci  # noqa: PLC0415

    g = np.random.default_rng(0)
    labels = g.integers(0, 2, 200)
    scores = labels * 0.3 + g.random(200)
    lo, hi, plo, phi = bootstrap_ci(labels, scores, 50)
    assert 0.5 < lo < hi <= 1.0 and 0 < plo < phi <= 1.0


# ── generator variant, pre-flight counts, --missing_image_as_absent ──────────


@pytest.mark.parametrize("task", ["phenotyping", "mortality"])
def test_round3_virtual_variant(task, tmp_path):
    from tools.synthetic.make_virtual_dataset import (  # noqa: PLC0415
        generate,
        generate_mortality,
        make_round3_variant,
    )

    (generate if task == "phenotyping" else generate_mortality)(tmp_path, "smoke", 0)
    summary = make_round3_variant(tmp_path, task, 0)
    assert summary["round3_variant"] and summary["val_images_missing_on_disk"] == 1
    assert summary["stays_without_rr"] >= 2
    assert "SYNTHETIC" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_preflight_counts(tmp_path):
    meta = pd.DataFrame({
        "dicom_id": ["a", np.nan, "b", "c", ""],
        "radiology_text": ["x", np.nan, "  ", "y", "z"],
        "discharge_text": [np.nan, "d", "d", "", "d"],
    })  # fmt: skip
    cxr = SimpleNamespace(filenames_to_path={"a": 1, "c": 1}, filesnames_to_labels={"a": 1, "b": 1})

    class DS:
        metadata_with_labels, cxr_ds = meta, cxr

        def __len__(self):
            return 5

    counts = r3_preflight.split_counts(DS(), "phenotyping")
    assert counts == {"stays": 5, "metadata_rows": 5, "cxr_selected": 3, "cxr_missing_on_disk": 1,
                      "cxr_without_chexpert_label": 1, "rr": 3, "dn": 3}  # fmt: skip


@pytest.mark.parametrize("flag_on", [False, True])
def test_missing_image_as_absent(flag_on):
    from datasets.DataFusion import MIMIC_CXR_EHR_RR_DN  # noqa: PLC0415

    meta = pd.DataFrame({"stay": ["s1"], "dicom_id": ["gone"], "age": [50], "gender": ["F"],
                         "ethnicity": ["WHITE"], "hadm_id": [1]})  # fmt: skip

    class CXR:
        filenames_to_path, filesnames_to_labels = {}, {"gone": [0]}

        def __getitem__(self, name):
            raise KeyError(name)

    class EHR:
        def __getitem__(self, name):
            return np.zeros((2, 76)), np.zeros(25)

    fake = SimpleNamespace(
        metadata_with_labels=meta, discharge_notes_paired=["d"], radiology_notes_paired=["r"],
        ehr_ds=EHR(), cxr_ds=CXR(),
        args=SimpleNamespace(data_pairs="partial", num_classes=25, missing_image_as_absent=flag_on),
    )  # fmt: skip
    if flag_on:
        item = MIMIC_CXR_EHR_RR_DN.__getitem__(fake, 0)
        assert item[1] is None
    else:
        with pytest.raises(KeyError):
            MIMIC_CXR_EHR_RR_DN.__getitem__(fake, 0)


@pytest.mark.parametrize("epoch, patience, expected", [(2, 0, True), (0, 15, True), (0, 3, False)])
def test_finished_variants_are_skipped(tmp_path, monkeypatch, epoch, patience, expected):
    import torch  # noqa: PLC0415

    from tools.pv.evaluate import checkpoint_path  # noqa: PLC0415

    args = SimpleNamespace(save_dir=str(tmp_path), task="in-hospital-mortality",
                           fusion_type="c-msma", lr=0.001, modalities="EHR-CXR-RR",
                           data_pairs="partial", epochs=3, patience=15)  # fmt: skip
    assert round3.finished(args) is False
    for prefix in ("last", "best"):
        path = checkpoint_path(args, prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "patience": patience}, path)
    assert round3.finished(args) is expected
