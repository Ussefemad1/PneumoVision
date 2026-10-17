"""Fusion mode (INFERENCE_MODE=fusion). Synthetic inputs only -- no MIMIC data.

The models here are *fake Round 3 runs*: real medpatch CMSMAFusion models
built from a run.json shaped like the model track's, with random weights, a
tiny random BERT (hidden size 32) and an un-pretrained ViT, so nothing is
downloaded. They check that the service rebuilds, verifies and runs a run
exactly as training would. A last test loads the real checkpoints when they
are present under CHECKPOINT_DIR/round3 and is skipped otherwise.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_SECRET = "test-secret-at-least-16-chars"
os.environ.setdefault("INFERENCE_HMAC_SECRET", TEST_SECRET)

REPO = Path(__file__).resolve().parents[3]
MEDPATCH = REPO / "medpatch"
REAL_FUSION_DIR = Path(os.environ.get("CHECKPOINT_DIR", REPO / "checkpoints")) / "round3"

have_ml = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "timm"))
pytestmark = pytest.mark.skipif(not have_ml, reason="needs torch, transformers and timm")

BERT = "dmis-lab/biobert-v1.1"
WORDS = "synthetic warm up report summary right lower lobe consolidation clear lungs pneumonia treated with antibiotics".split()

ARGV = {
    "mortality": [
        "--mode", "train", "--batch_size", "16", "--lr", "0.001", "--num_classes", "1",
        "--modalities", "EHR-CXR-RR", "--ehr_encoder", "lstm",
        "--cxr_encoder", "vit_small_patch16_384", "--text_encoder", "biobert",
        "--classifier", "mlp", "--loss", "bce", "--task", "in-hospital-mortality",
        "--labels_set", "mortality", "--output_dim", "512", "--output_dim_cxr", "512",
        "--output_dim_rr", "512", "--fusion_type", "c-msma", "--data_pairs", "partial",
        "--ehr_confidence_threshold", "0.75", "--cxr_confidence_threshold", "0.75",
        "--rr_confidence_threshold", "0.75", "--bert_model_name", BERT,
        "--load_ehr", "/content/drive/r2b-ehr.pth.tar", "--load_cxr", "/content/drive/r2b-cxr.pth.tar",
        "--load_rr", "/content/drive/r2b-rr.pth.tar", "--ehr_data_dir", "/content/ehr",
        "--cxr_data_dir", "/content/cxr", "--notes_data_dir", "/content/notes",
        "--save_dir", "/content/drive/r3/r3-fusion-901", "--resume", "--epochs", "100",
    ],
    "pneumonia": [
        "--mode", "train", "--batch_size", "16", "--lr", "0.001", "--num_classes", "25",
        "--modalities", "EHR-CXR-RR-DN", "--ehr_encoder", "lstm",
        "--cxr_encoder", "vit_small_patch16_384", "--text_encoder", "biobert",
        "--classifier", "mlp", "--loss", "bce", "--task", "phenotyping", "--labels_set", "pheno",
        "--output_dim", "512", "--output_dim_cxr", "512", "--output_dim_rr", "512",
        "--output_dim_dn", "512", "--fusion_type", "c-msma", "--data_pairs", "partial",
        "--ehr_confidence_threshold", "0.75", "--cxr_confidence_threshold", "0.75",
        "--rr_confidence_threshold", "0.75", "--dn_confidence_threshold", "0.75",
        "--bert_model_name", BERT,
        "--load_ehr", "/content/drive/a.pth.tar", "--load_cxr", "/content/drive/b.pth.tar",
        "--load_rr", "/content/drive/c.pth.tar", "--load_dn", "/content/drive/d.pth.tar",
        "--save_dir", "/content/drive/r3/r3-fusion-902", "--resume", "--epochs", "100",
    ],
}  # fmt: skip
FOLDER = {"mortality": "mortality", "pneumonia": "phenotyping"}
RUN_ID = {"mortality": "r3-fusion-901", "pneumonia": "r3-fusion-902"}


def _install_offline_stubs(mp: pytest.MonkeyPatch, vocab_file: Path) -> None:
    """Tiny random BERT + tokenizer and an un-pretrained ViT: no downloads."""
    import timm
    from transformers import BertConfig, BertModel, BertTokenizerFast

    config = BertConfig(
        vocab_size=len(vocab_file.read_text().split()), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=37, max_position_embeddings=512,
    )  # fmt: skip

    def tiny_bert(cls, name, *args, **kwargs):
        return BertModel(config)

    def tiny_tokenizer(cls, name, *args, **kwargs):
        return BertTokenizerFast(vocab_file=str(vocab_file), name_or_path=name)

    real_create = timm.create_model

    def create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return real_create(*args, **kwargs)

    mp.setattr(BertModel, "from_pretrained", classmethod(tiny_bert))
    mp.setattr(BertTokenizerFast, "from_pretrained", classmethod(tiny_tokenizer))
    mp.setattr(timm, "create_model", create_model)


def _build_reference(task: str):
    """A CMSMAFusion built from the fake run's arguments, as MSMA_Trainer does."""
    import torch
    from arguments import args_parser
    from models.cxr_models import CXR_encoder
    from models.ehr_models import EHR_encoder
    from models.fusion import Fusion
    from models.text_models import Text_encoder

    from app.fusion import clean_argv

    args = args_parser().parse_args(clean_argv(ARGV[task]))
    torch.manual_seed(7 if task == "mortality" else 11)
    model = Fusion(
        args, EHR_encoder(args), CXR_encoder(args), Text_encoder(args, "cpu")
    ).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("_temperature"):
                p.uniform_(0.5, 1.5)  # learned per-position temperatures
            elif name.endswith("weights"):
                p.normal_(0.0, 1.0)  # late-fusion α logits
    return args, model


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory):
    """FUSION_DIR with one fake Round 3 run per task, the stubs active, and
    the reference models the runs were saved from."""
    import torch

    sys.path.insert(0, str(MEDPATCH))
    root = tmp_path_factory.mktemp("round3")
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS, "."]))
    mp = pytest.MonkeyPatch()
    _install_offline_stubs(mp, vocab)

    from app.fusion import sha256_file

    references = {}
    for task in ("mortality", "pneumonia"):
        args, model = _build_reference(task)
        folder = root / FOLDER[task]
        folder.mkdir()
        ckpt = folder / f"best_checkpoint_0.001_{args.task}_c-msma_{args.modalities}_partial.pth.tar"
        torch.save({"epoch": 3, "state_dict": model.state_dict(), "best_head": "late"}, ckpt)
        run = {
            "id": RUN_ID[task],
            "stage": "r3",
            "fusion_type": "c-msma",
            "bert_model_name": BERT,
            "argv": ARGV[task],
            "manifest_row": {"id": RUN_ID[task], "sha256": sha256_file(ckpt), "bert_model_name": BERT},
        }
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        references[task] = (args, model)
    yield root, references
    mp.undo()


@pytest.fixture(scope="module")
def loaded(fake_runs):
    from app.fusion import load_fusion

    root, _ = fake_runs
    return load_fusion(root, MEDPATCH, "cpu", 0.75)


def _png(value: int = 128, size: tuple[int, int] = (600, 700)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", size, value).save(buffer, format="PNG")
    return buffer.getvalue()


def _ehr(hours: int = 48):
    from app.schemas import EhrInput

    variables = ["Heart Rate", "Respiratory rate", "Oxygen saturation", "Glascow coma scale total"]
    values = [[80.0 + h, 18.0 + (h % 5), 97.0 - (h % 3), 15 if h % 7 else None] for h in range(hours)]
    return EhrInput(variables=variables, values=values)


def _request(task: str, *, ehr=True, rr=True, dn=True, hours: int = 48):
    from app.schemas import NoteInput, PredictRequest

    notes = []
    if rr:
        notes += [
            NoteInput(id="r1", type="radiology", text="synthetic right lower lobe consolidation"),
            NoteInput(id="r2", type="radiology", text="synthetic clear lungs"),
        ]
    if dn and task == "pneumonia":
        notes.append(NoteInput(id="d1", type="discharge", text="pneumonia treated with antibiotics"))
    return PredictRequest(task=task, seed="s", ehr=_ehr(hours) if ehr else None, notes=notes)


# ── argv and loading ─────────────────────────────────────────────────────────


def test_clean_argv_drops_parents_resume_and_save_dir():
    from app.fusion import clean_argv

    cleaned = clean_argv(ARGV["pneumonia"])
    assert not any(t.startswith("--load_") or t in ("--resume", "--save_dir") for t in cleaned)
    assert not any(t.startswith("/content/drive/r") for t in cleaned)
    assert cleaned[cleaned.index("--epochs") + 1] == "100"
    assert cleaned[cleaned.index("--modalities") + 1] == "EHR-CXR-RR-DN"


def test_both_tasks_load_with_the_recorded_head(loaded):
    m, p = loaded.by_task["mortality"], loaded.by_task["pneumonia"]
    assert (m.run_id, m.head, m.modalities, m.output_index) == ("r3-fusion-901", "late", ["EHR", "CXR", "RR"], 0)
    assert (p.run_id, p.head, p.modalities, p.output_index) == ("r3-fusion-902", "late", ["EHR", "CXR", "RR", "DN"], 21)
    assert m.bert_model_name == BERT and m.theta == 0.75
    assert loaded.version.startswith("medpatch-r3:r3-fusion-901@")


# ── parity with medpatch, all heads ──────────────────────────────────────────


@pytest.mark.parametrize("task", ["mortality", "pneumonia"])
@pytest.mark.parametrize("with_cxr", [True, False])
def test_every_head_matches_the_medpatch_forward(fake_runs, loaded, task, with_cxr):
    import numpy as np
    import torch
    from datasets.cxr_dataset import get_transforms
    from datasets.DataFusion import pad_zeros
    from ehr_utils.preprocessing import Discretizer, Normalizer

    from app.readers import cxr_image, ehr_rows

    _, references = fake_runs
    args, model = references[task]
    req = _request(task)
    image = _png() if with_cxr else None
    got = loaded.score(req, image)

    # The same stay through medpatch, independently of the service.
    disc = Discretizer(timestep=1.0, store_masks=True, impute_strategy="previous", start_time="zero",
                       config_path=str(MEDPATCH / "ehr_utils/resources/discretizer_config.json"))  # fmt: skip
    data, header = disc.transform(np.array(ehr_rows(req.ehr, disc, 48)), end=48.0)
    norm = Normalizer(fields=[i for i, h in enumerate(header.split(",")) if "->" not in h])
    norm.load_params(str(MEDPATCH / "normalizers/ph_ts1.0.input_str_previous.start_time_zero.normalizer"))
    x, seq = pad_zeros([norm.transform(data)], args=args)
    if task == "pneumonia":
        assert x.shape[1] == 2646  # padded as in training
    img = (get_transforms(args)[1](cxr_image(image))[None] if with_cxr else torch.zeros(1, 3, 384, 384))
    rr = ["synthetic right lower lobe consolidation synthetic clear lungs"]
    dn = ["pneumonia treated with antibiotics"] if task == "pneumonia" else None
    with torch.inference_mode():
        out = model(torch.from_numpy(x).float(), seq, img, [with_cxr], rr, dn)

    k = 21 if task == "pneumonia" else 0
    high, low, late = (out[h].reshape(-1)[k] for h in ("high_conf", "low", "late"))
    expected = {
        "high": high, "low": low, "late": late,
        "combined": (high + low + late) / 3, "less_combined": (high + late) / 2,
    }  # fmt: skip
    for head, logit in expected.items():
        assert abs(got.heads[head] - float(torch.sigmoid(logit))) <= 1e-5, head
    assert got.probability == got.heads["late"]
    assert got.present["cxr"] is with_cxr
    assert abs(sum(got.alphas.values()) - 1.0) < 1e-5


# ── every missing-modality combination returns a result ──────────────────────


@pytest.mark.parametrize("task", ["mortality", "pneumonia"])
def test_every_missing_modality_combination_returns_a_result(loaded, task):
    from app.cxr import DecodedCxr
    from app.fusion import build_fusion_result

    fm = loaded.by_task[task]
    texts = ["rr", "dn"] if task == "pneumonia" else ["rr"]
    for ehr, cxr, *notes in itertools.product([True, False], repeat=2 + len(texts)):
        flags = dict(zip(texts, notes, strict=True))
        req = _request(task, ehr=ehr, rr=flags["rr"], dn=flags.get("dn", False))
        image = _png() if cxr else None
        score = loaded.score(req, image)
        assert score.present["cxr"] is cxr
        assert score.present["rr"] is flags["rr"]
        if task == "pneumonia":
            assert score.present["dn"] is flags["dn"]
        decoded = DecodedCxr(data=image, sha256="0" * 64, width=600, height=700) if cxr else None
        result = build_fusion_result(req, decoded, fm, score)
        assert 0.0 <= result.probability <= 1.0
        assert result.sources.fusion == "trained"
        assert result.received.mode == "model"
        a = result.alphas
        total = a.high + a.low + a.miss + sum(v or 0.0 for v in (a.ehr, a.cxr, a.rr, a.dn))
        assert abs(total - 1.0) < 1e-6
        assert (result.unimodal.cxr is None) is not cxr
        assert result.fusion_model is not None and result.fusion_model.calibrated is False


def test_pneumonia_reads_the_whole_stay_it_is_sent(loaded):
    score = loaded.score(_request("pneumonia", hours=72), None)
    assert any("sent 72 h" in n and "2646" in n for n in score.notes)


# ── refusals ─────────────────────────────────────────────────────────────────


def _copy_run(fake_runs, tmp_path, task: str, edit) -> Path:
    root, _ = fake_runs
    folder = tmp_path / FOLDER[task]
    shutil.copytree(root / FOLDER[task], folder)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    edit(run)
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return folder


def test_a_checkpoint_whose_sha256_differs_is_refused(fake_runs, tmp_path):
    from app.fusion import FusionLoadError, load_task

    def edit(run):
        run["manifest_row"]["sha256"] = "f" * 64

    folder = _copy_run(fake_runs, tmp_path, "mortality", edit)
    with pytest.raises(FusionLoadError, match="sha256"):
        load_task("mortality", folder, MEDPATCH, "cpu", 0.75)


def test_a_threshold_other_than_theta_is_refused(fake_runs, tmp_path):
    from app.fusion import FusionLoadError, load_task

    def edit(run):
        argv = run["argv"]
        argv[argv.index("--rr_confidence_threshold") + 1] = "0.65"

    folder = _copy_run(fake_runs, tmp_path, "mortality", edit)
    with pytest.raises(FusionLoadError, match="threshold"):
        load_task("mortality", folder, MEDPATCH, "cpu", 0.75)
    # And a service configured with another θ refuses the run as recorded.
    with pytest.raises(FusionLoadError, match="DEFAULT_THETA"):
        load_task("mortality", fake_runs[0] / "mortality", MEDPATCH, "cpu", 0.6)


def test_a_bert_mismatch_is_refused(fake_runs, tmp_path):
    from app.fusion import FusionLoadError, load_task

    def edit(run):
        run["manifest_row"]["bert_model_name"] = "emilyalsentzer/Bio_ClinicalBERT"

    folder = _copy_run(fake_runs, tmp_path, "pneumonia", edit)
    with pytest.raises(FusionLoadError, match="BERT"):
        load_task("pneumonia", folder, MEDPATCH, "cpu", 0.75)


def test_a_missing_run_json_or_checkpoint_is_refused(fake_runs, tmp_path):
    from app.fusion import FusionLoadError, load_task

    folder = _copy_run(fake_runs, tmp_path, "mortality", lambda run: None)
    for ckpt in folder.glob("*.pth.tar"):
        ckpt.unlink()
    with pytest.raises(FusionLoadError, match="exactly one"):
        load_task("mortality", folder, MEDPATCH, "cpu", 0.75)
    (folder / "run.json").unlink()
    with pytest.raises(FusionLoadError, match="run.json"):
        load_task("mortality", folder, MEDPATCH, "cpu", 0.75)


# ── through the HTTP service ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fusion_client(fake_runs):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import app

    root, _ = fake_runs
    keys = ("INFERENCE_MODE", "FUSION_DIR", "MEDPATCH_ROOT", "MOCK_MODE")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(INFERENCE_MODE="fusion", FUSION_DIR=str(root), MEDPATCH_ROOT=str(MEDPATCH), MOCK_MODE="false")
    get_settings.cache_clear()
    with TestClient(app) as client:
        yield client
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


def _post(client, body: dict):
    from app.security import compute_signature

    raw = json.dumps(body).encode()
    ts = str(int(time.time()))
    return client.post(
        "/v1/predict",
        content=raw,
        headers={"content-type": "application/json", "X-PV-Timestamp": ts,
                 "X-PV-Signature": compute_signature(TEST_SECRET, ts, raw)},
    )  # fmt: skip


def test_service_reports_the_loaded_fusion_models(fusion_client):
    health = fusion_client.get("/v1/health").json()
    assert health["status"] == "ok" and health["mode"] == "fusion"
    assert health["modelsLoaded"] is True and health["warmedUp"] is True
    info = fusion_client.get("/v1/model-info").json()
    assert info["textModel"] == BERT
    assert info["fusion"]["mortality"]["runId"] == "r3-fusion-901"
    assert info["fusion"]["pneumonia"]["modalities"] == ["EHR", "CXR", "RR", "DN"]
    assert info["fusion"]["pneumonia"]["head"] == "late"
    assert len(info["fusion"]["pneumonia"]["sha256"]) == 12


def test_service_predicts_with_the_fusion_model(fusion_client):
    import base64
    import hashlib

    image = _png()
    body = {
        "task": "pneumonia",
        "seed": "s",
        "ehr": _ehr().model_dump(),
        "cxr": {"contentType": "image/png", "sha256": hashlib.sha256(image).hexdigest(),
                "dataB64": base64.b64encode(image).decode()},
        "notes": [{"id": "r", "type": "radiology", "text": "synthetic clear lungs"},
                  {"id": "p", "type": "progress", "text": "synthetic progress note"},
                  {"id": "d", "type": "discharge", "text": "pneumonia treated"}],
    }  # fmt: skip
    res = _post(fusion_client, body)
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["mock"] is False
    assert out["modelVersion"].startswith("medpatch-r3:")
    result = out["result"]
    assert result["sources"]["fusion"] == "trained"
    assert result["received"]["mode"] == "model"
    fm = result["fusionModel"]
    assert fm["runId"] == "r3-fusion-902" and fm["head"] == "late" and fm["calibrated"] is False
    assert result["probability"] == pytest.approx(fm["heads"]["late"])
    assert any("Progress notes are not read" in n for n in fm["notes"])


def test_service_refuses_discharge_notes_for_mortality(fusion_client):
    res = _post(fusion_client, {"task": "mortality", "seed": "s",
                                "notes": [{"id": "d", "type": "discharge", "text": "synthetic"}]})  # fmt: skip
    assert res.status_code == 400


# ── the real Round 3 checkpoints, when present ───────────────────────────────

have_real = all((REAL_FUSION_DIR / f / "run.json").is_file() for f in ("mortality", "phenotyping"))


@pytest.mark.skipif(not have_real, reason=f"needs the Round 3 checkpoints under {REAL_FUSION_DIR}")
def test_the_real_round3_checkpoints_load_and_score():
    from app.fusion import load_fusion

    models = load_fusion(REAL_FUSION_DIR, MEDPATCH, "cpu", 0.75)
    assert models.by_task["mortality"].run_id == "r3-fusion-001"
    assert models.by_task["mortality"].modalities == ["EHR", "CXR", "RR"]
    assert models.by_task["pneumonia"].modalities == ["EHR", "CXR", "RR", "DN"]
    for task in ("mortality", "pneumonia"):
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            score = models.score(_request(task), _png())
            timings.append(time.perf_counter() - started)
            assert 0.0 < score.probability < 1.0
        timings.sort()
        print(f"\n{task}: p50 {timings[2]:.2f}s max {timings[-1]:.2f}s")
    long_note = copy.deepcopy(_request("pneumonia"))
    long_note.notes[-1].text = " ".join(WORDS * 400)  # ~6,000 tokens: 12 chunks
    models.score(long_note, _png())
