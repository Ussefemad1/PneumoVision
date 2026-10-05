"""tools.pv.check_weights -- strict parent coverage, both tasks. Non-training.

Models are medpatch's real Fusion / Unimodal* classes; the CXR and text
encoders are deterministic stubs (no ViT or BERT download), the EHR LSTM is
real. "r1 files" are the state_dict of the matching Round 1 model, saved the
way Trainer.save_checkpoint saves it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tools.pv import check_weights, import_checkpoint, manifest

BIOBERT = "dmis-lab/biobert-v1.1"


class StubCXR(nn.Module):
    feats_dim = full_feats_dim = 16

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.feature_extractor = nn.Linear(4, 16)


class StubText(nn.Module):
    """Text_encoder's structure: frozen `bert`, plus fc_rr / fc_dn per modality."""

    def __init__(self, args):
        super().__init__()
        torch.manual_seed(2)
        self.bert = nn.Linear(8, 8)  # stands in for the pretrained BERT
        self.feats_dim_rr, self.feats_dim_dn = args.output_dim_rr, args.output_dim_dn
        self.full_feats_dim_rr = self.full_feats_dim_dn = 8
        if "RR" in args.modalities:
            self.fc_rr = nn.Linear(8, args.output_dim_rr)
        if "DN" in args.modalities:
            self.fc_dn = nn.Linear(8, args.output_dim_dn)


@pytest.fixture(autouse=True)
def stub_encoders(monkeypatch):
    from models.ehr_models import EHR_encoder  # noqa: PLC0415

    def make(args):
        ehr = EHR_encoder(args) if "EHR" in args.modalities and args.ehr_encoder else None
        cxr = StubCXR() if "CXR" in args.modalities and args.cxr_encoder else None
        text = (
            StubText(args)
            if ("RR" in args.modalities or "DN" in args.modalities) and args.text_encoder
            else None
        )
        return ehr, cxr, text

    monkeypatch.setattr(check_weights, "make_encoders", make)


def r1_file(tmp_path, reader, task, edit=None):
    """A Round 1 checkpoint of `reader`, optionally edited, saved like medpatch saves."""
    args = check_weights.model_args(reader, task, "r1", BIOBERT)
    state = check_weights.build_model(args).state_dict()
    if edit:
        edit(state)
    path = tmp_path / f"r1-{reader}-{task}.pth.tar"
    torch.save({"epoch": 3, "state_dict": state, "best_auroc": 0.7}, path)
    return path


CASES = [
    ("ehr", "phenotyping"),
    ("cxr", "phenotyping"),
    ("rr", "phenotyping"),
    ("dn", "phenotyping"),
    ("ehr", "mortality"),
    ("cxr", "mortality"),
    ("rr", "mortality"),
]


@pytest.mark.parametrize("reader, task", CASES)
def test_full_coverage_passes_and_only_the_head_is_new(tmp_path, reader, task):
    rep = check_weights.check(reader, task, r1_file(tmp_path, reader, task), "r2", BIOBERT)
    assert rep.failures == [] and rep.coverage == 100.0 and rep.applied
    new = [k for k in rep.uncovered if check_weights.role(k, reader) == "expected_new"]
    assert new == [
        f"fusion_model.{reader}_confidence_predictor.confidence_layer.weight",
        f"fusion_model.{reader}_confidence_predictor.confidence_layer.bias",
    ]
    assert all(check_weights.role(k, reader) == "expected_new" for k in rep.uncovered)
    assert rep.required()  # encoder and classifier tensors were actually checked


@pytest.mark.parametrize("task", ["phenotyping", "mortality"])
def test_renamed_key_fails(tmp_path, task):
    key = "fusion_model.rr_classifier.classifier.fc.weight"

    def rename(state):
        state[key.replace(".fc.", ".fc_renamed.")] = state.pop(key)

    rep = check_weights.check("rr", task, r1_file(tmp_path, "rr", task, rename), "r2", BIOBERT)
    assert f"not in the file: {key}" in rep.failures
    assert key.replace(".fc.", ".fc_renamed.") in rep.unused


@pytest.mark.parametrize("task", ["phenotyping", "mortality"])
def test_wrong_shape_fails_and_is_not_applied(tmp_path, task):
    key = "fusion_model.ehr_classifier.classifier.fc.weight"

    def reshape(state):
        state[key] = torch.zeros(3, state[key].shape[1])

    rep = check_weights.check("ehr", task, r1_file(tmp_path, "ehr", task, reshape), "r2")
    assert any(f.startswith("shape (3, 512)") and f.endswith(key) for f in rep.failures)
    assert not rep.applied


def test_why_strict_load_state_broadcasts_a_wrong_shape_silently():
    """The hazard check_weights guards against: copy_ broadcasts, no error."""
    target = torch.empty(25, 512)
    target.copy_(torch.ones(1, 512))  # what load_state does with a [1, 512] tensor
    assert torch.equal(target, torch.ones(25, 512))


def test_bert_weights_must_be_the_named_pretrained_ones(tmp_path):
    def retrain_bert(state):
        for k in state:
            if ".bert." in f".{k}":
                state[k] = state[k] + 0.01

    rep = check_weights.check(
        "rr", "phenotyping", r1_file(tmp_path, "rr", "phenotyping", retrain_bert), "r2", BIOBERT
    )
    assert any(f.startswith("BERT weight differs") for f in rep.failures)


def test_r2_file_into_r2b_lists_the_temperature_as_expected_new(tmp_path):
    args = check_weights.model_args("cxr", "mortality", "r2", None)
    path = tmp_path / "r2-cxr.pth.tar"
    torch.save({"state_dict": check_weights.build_model(args).state_dict()}, path)
    rep = check_weights.check("cxr", "mortality", path, "r2b")
    assert rep.failures == []
    assert rep.uncovered == ["fusion_model.cxr_temperature"]


def test_dn_for_mortality_is_refused(tmp_path):
    with pytest.raises(ValueError, match="leak the outcome"):
        check_weights.check("dn", "mortality", tmp_path / "never-read.pth.tar")


def test_cli_exit_code(tmp_path, capsys):
    good = r1_file(tmp_path, "cxr", "phenotyping")
    assert check_weights.main(["--reader", "cxr", "--file", str(good)]).failures == []
    assert "PASS" in capsys.readouterr().out

    def drop(state):
        state.pop("fusion_model.cxr_classifier.classifier.fc.bias")

    bad = r1_file(tmp_path, "cxr", "mortality", drop)
    with pytest.raises(SystemExit) as exc:
        check_weights.main(["--reader", "cxr", "--task", "mortality", "--file", str(bad)])
    assert exc.value.code == 1


# ── import_checkpoint --check-weights ────────────────────────────────────────


@pytest.fixture
def tmp_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MANIFEST", str(tmp_path / "manifest.csv"))
    return tmp_path / "manifest.csv"


def test_import_records_the_coverage(tmp_manifest, tmp_path):
    path = r1_file(tmp_path, "rr", "mortality")
    row = import_checkpoint.main(
        [
            "--reader",
            "rr",
            "--task",
            "mortality",
            "--who",
            "Farida",
            "--file",
            str(path),
            "--bert-model-name",
            BIOBERT,
            "--check-weights",
        ]
    )
    assert "weights: 100% covered" in row["notes"] and BIOBERT in row["notes"]


def test_import_refuses_a_failing_file(tmp_manifest, tmp_path):
    def drop(state):
        state.pop("text_model.fc_dn.weight")

    path = r1_file(tmp_path, "dn", "phenotyping", drop)
    with pytest.raises(SystemExit, match="weight check failed"):
        import_checkpoint.main(
            [
                "--reader",
                "dn",
                "--who",
                "Farida",
                "--file",
                str(path),
                "--bert-model-name",
                BIOBERT,
                "--check-weights",
            ]
        )
    assert manifest.read_rows() == []
