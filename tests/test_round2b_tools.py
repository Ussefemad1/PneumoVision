"""Round 2b tooling and trainer fixes -- checks that do not train.

No optimizer step and no real encoder runs here: script parsing, manifest
lineage, run planning, the Calibration trainer's shape handling on synthetic
tensors, and the r2 -> r2b state_dict key match on modules built with stubs.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.pv import manifest, run
from tools.pv.medpatch_bridge import parse_args
from tools.pv.paper_scripts import LOAD_FLAG, READERS, build_argv, script_settings


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    monkeypatch.setenv("PV_MANIFEST", str(path))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    return path


def _row(tmp_path, stage, reader="cxr", data="virtual", parent_id=""):
    f = tmp_path / f"{reader}-{stage}-{manifest.next_id(stage, reader)}.pth.tar"
    f.write_bytes(f"{stage}-{reader}".encode())
    return manifest.append_row(
        {
            "id": manifest.next_id(stage, reader),
            "stage": stage,
            "reader": reader,
            "data": data,
            "file_path": f.as_posix(),
            "sha256": manifest.sha256_file(f),
            "parent_id": parent_id,
        }
    )


# ── paper scripts ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reader", READERS)
def test_calibrate_scripts_parse(reader):
    settings = script_settings("r2b", reader)
    assert settings["--fusion_type"] == f"temp_c-unimodal_{reader}"
    assert settings["--task"] == "phenotyping"
    # Every Calibrate script already names its own reader's load flag (DN included).
    assert settings.get(LOAD_FLAG[reader]) is not None


@pytest.mark.parametrize("reader", READERS)
def test_r2b_argv_loads_only_the_given_parent(reader):
    argv = build_argv("r2b", reader, {LOAD_FLAG[reader]: "r2-parent.pth.tar"})
    assert argv[argv.index(LOAD_FLAG[reader]) + 1] == "r2-parent.pth.tar"
    others = set(LOAD_FLAG.values()) - {LOAD_FLAG[reader]}
    assert not others & set(argv)


# ── manifest lineage ─────────────────────────────────────────────────────────


def test_r2b_is_a_stage_with_r2_as_parent():
    assert "r2b" in manifest.STAGES
    assert manifest.PARENT_STAGE == {"r2": "r1", "r2b": "r2"}


def test_verify_r2_parent_accepts_an_r2_row(tmp_manifest, tmp_path):
    r2 = _row(tmp_path, "r2")
    assert manifest.verify_r2_parent(r2, "cxr", "virtual").is_file()


@pytest.mark.parametrize(
    "case, message",
    [
        ("missing", "run r2 --reader cxr --data virtual"),
        ("r1", "not r2"),
        ("reader", "checkpoint; this run is"),
        ("data", "Mixing virtual and real"),
        ("sha", "sha256 mismatch"),
    ],
)
def test_verify_r2_parent_refuses(tmp_manifest, tmp_path, case, message):
    row = None
    if case != "missing":
        row = _row(
            tmp_path, "r1" if case == "r1" else "r2", reader="ehr" if case == "reader" else "cxr"
        )
    if case == "sha":
        with open(row["file_path"], "ab") as f:
            f.write(b"tampered")
    with pytest.raises(manifest.ManifestError, match=message):
        manifest.verify_r2_parent(row, "cxr", "real" if case == "data" else "virtual")


# ── run.plan ─────────────────────────────────────────────────────────────────


def _plan_args(tmp_path, **kw):
    data_root = tmp_path / "virtual"
    data_root.mkdir(exist_ok=True)
    (data_root / "README.md").write_text("SYNTHETIC")
    defaults = dict(
        stage="r2b",
        reader="cxr",
        data="virtual",
        parent=None,
        run_id=None,
        preset="smoke",
        data_root=str(data_root),
        epochs=None,
        batch_size=None,
        bootstrap_iters=None,
        num_workers=None,
    )
    return argparse.Namespace(**{**defaults, **kw})


def test_plan_r2b_loads_the_latest_r2_row(tmp_manifest, tmp_path):
    r1 = _row(tmp_path, "r1")
    r2 = _row(tmp_path, "r2", parent_id=r1["id"])
    plan = run.plan(_plan_args(tmp_path))
    argv = plan["argv"]
    assert plan["id"] == "r2b-cxr-001"
    assert plan["parent_id"] == r2["id"]
    assert plan["script"].endswith("Calibrate/Calibrate-CXR.sh")
    assert Path(argv[argv.index("--load_cxr") + 1]) == Path(r2["file_path"])
    assert argv[argv.index("--fusion_type") + 1] == "temp_c-unimodal_cxr"
    assert argv[argv.index("--epochs") + 1] == str(run.VIRTUAL_R2B_EPOCHS["cxr"])
    assert argv[argv.index("--batch_size") + 1] == "4"


def test_plan_r2b_refuses_an_r1_only_lineage(tmp_manifest, tmp_path):
    _row(tmp_path, "r1")
    with pytest.raises(manifest.ManifestError, match="No r2 row"):
        run.plan(_plan_args(tmp_path))


def test_plan_r2b_refuses_an_explicit_r1_parent(tmp_manifest, tmp_path):
    r1 = _row(tmp_path, "r1")
    with pytest.raises(manifest.ManifestError, match="not r2"):
        run.plan(_plan_args(tmp_path, parent=r1["id"]))


def test_run_cli_accepts_r2b():
    args = run.build_parser().parse_args(["r2b", "--reader", "dn", "--data", "virtual"])
    assert args.stage == "r2b"


# ── Calibration.py fixes (synthetic tensors, no training) ────────────────────


def _calibration():
    from trainers.Calibration import calibration  # noqa: PLC0415

    return calibration


def test_calibration_trains_on_val_only_by_construction():
    """Static guard: the calibration loop may read val_dl and nothing else."""
    cal = _calibration()
    # confidence_batches holds the model call (or the --cache_frozen_logits replay).
    for method in (cal.train_epoch, cal.train, cal.confidence_batches):
        src = inspect.getsource(method)
        assert "self.train_dl" not in src and "self.test_dl" not in src, method.__name__
    assert "self.val_dl" in inspect.getsource(cal.confidence_batches)


def test_calibration_uses_the_shared_token_axis_fix():
    cal = _calibration()
    src = inspect.getsource(cal.train_epoch) + inspect.getsource(cal.confidence_batches)
    code = "\n".join(line.split("#")[0] for line in src.splitlines())  # ignore comments
    assert "self.confidence_logits(output)" in code  # the model path
    assert "self.confidence_logits({self.args.fusion_type: scaled})" in code  # the cache
    assert ".squeeze()" not in code


@pytest.mark.parametrize("length", [30, 48, 71])
def test_pad_to_length_pads_and_truncates(length):
    out = _calibration().pad_to_length(None, torch.ones(2, length, 25), max_len=48)
    assert tuple(out.shape) == (2, 48, 25)
    assert out[:, : min(length, 48)].eq(1).all()


@pytest.mark.parametrize("shape", [(6, 1, 25), (6, 48, 25), (6, 512, 25)])
def test_flat_ece_is_per_class_over_all_tokens(shape):
    cal = _calibration()
    holder = argparse.Namespace(args=argparse.Namespace(num_classes=25))
    holder.compute_ece = lambda p, lab, n_bins=10: cal.compute_ece(holder, p, lab, n_bins)
    probs = torch.rand(shape)
    labels = (torch.rand(shape) > 0.5).float()
    ece = cal.flat_ece(holder, probs, labels)  # CXR's single token used to raise IndexError
    expected = cal.compute_ece(holder, probs.reshape(-1, 25), labels.reshape(-1, 25))
    assert ece.shape == (25,) and torch.equal(ece, expected)


def test_base_trainer_keeps_the_single_cxr_token():
    from trainers.trainer import Trainer  # noqa: PLC0415

    holder = argparse.Namespace(args=argparse.Namespace(fusion_type="temp_c-unimodal_cxr"))
    out = Trainer.confidence_logits(holder, {"temp_c-unimodal_cxr": torch.zeros(4, 1, 25)})
    assert tuple(out.shape) == (4, 1, 25)


# ── r2 checkpoints restore into the r2b model ────────────────────────────────


class StubCXR(nn.Module):
    feats_dim = full_feats_dim = 384

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)


class StubText(nn.Module):
    feats_dim_rr = feats_dim_dn = 512
    full_feats_dim_rr = full_feats_dim_dn = 768

    def __init__(self):
        super().__init__()
        self.bert = nn.Linear(2, 2)


@pytest.mark.parametrize("reader", READERS)
def test_r2_keys_cover_every_r2b_weight_but_temperature(reader):
    from models import fusion  # noqa: PLC0415
    from models.ehr_models import LSTM  # noqa: PLC0415

    r2_cls = {
        "ehr": fusion.UnimodalEHRConfidence,
        "cxr": fusion.UnimodalCXRConfidence,
        "rr": fusion.UnimodalRRConfidence,
        "dn": fusion.UnimodalDNConfidence,
    }[reader]
    r2b_cls = {
        "ehr": fusion.TempCUnimodalEHR,
        "cxr": fusion.TempCUnimodalCXR,
        "rr": fusion.TempCUnimodalRR,
        "dn": fusion.TempCUnimodalDN,
    }[reader]
    modalities = {"ehr": "EHR", "cxr": "EHR-CXR", "rr": "EHR-RR", "dn": "EHR-DN"}[reader]

    def build(cls, fusion_type):
        args = parse_args(
            [
                "--fusion_type",
                fusion_type,
                "--modalities",
                modalities,
                "--task",
                "phenotyping",
                "--num_classes",
                "25",
                "--classifier",
                "mlp",
                "--ehr_encoder",
                "lstm",
            ]
        )
        encoder = {"ehr": lambda: LSTM(args), "cxr": StubCXR, "rr": StubText, "dn": StubText}[
            reader
        ]()
        return cls(args, encoder)

    r2_keys = set(build(r2_cls, f"c-unimodal_{reader}").state_dict())
    r2b_keys = set(build(r2b_cls, f"temp_c-unimodal_{reader}").state_dict())
    assert r2b_keys - r2_keys == {f"{reader}_temperature"}
    assert r2_keys <= r2b_keys
