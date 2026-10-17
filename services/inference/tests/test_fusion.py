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

from scripts.make_fake_runs import (  # noqa: E402
    ARGV,
    BERT,
    FOLDER,
    RUN_ID,
    build,
    share_bert,
    write_fake_readers,
    write_run,
)

WORDS = "synthetic warm up report summary right lower lobe consolidation clear lungs pneumonia treated with antibiotics".split()


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

    from app import reader_heads

    references = {task: build(task) for task in ("mortality", "pneumonia")}
    share_bert({task: model for task, (_, model) in references.items()})
    for task, (args, model) in references.items():
        write_run(task, args, model, root)
    # Fake Round 1 readers sharing each run's encoders (as F2 makes the real
    # ones), and their heads extracted as tools/pv/extract_reader_heads.py does.
    r1 = tmp_path_factory.mktemp("r1")
    write_fake_readers({task: model for task, (_, model) in references.items()}, r1)
    reader_heads.extract(r1, root)
    yield root, references, r1
    mp.undo()


@pytest.fixture(scope="module")
def loaded(fake_runs):
    from app.fusion import load_fusion

    root, _, r1 = fake_runs
    return load_fusion(root, MEDPATCH, "cpu", 0.75, checkpoint_dir=r1)


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

    _, references, _ = fake_runs
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
    root, _, _ = fake_runs
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

    root, _, _ = fake_runs
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


# ── "By source": the Round 1 standalone readers ──────────────────────────────


def _full_request(task: str):
    return _request(task, ehr=True, rr=True, dn=True)


@pytest.fixture(scope="module")
def readers_mode(fake_runs):
    """INFERENCE_MODE=readers' own path over the same fake Round 1 readers."""
    from app.readers import load_readers

    _, _, r1 = fake_runs
    return load_readers(MEDPATCH, r1, "cpu", confidence=False)


@pytest.mark.parametrize("task", ["mortality", "pneumonia"])
@pytest.mark.parametrize("with_cxr", [True, False])
def test_by_source_equals_readers_mode(loaded, readers_mode, task, with_cxr):
    req = _full_request(task)
    image = _png() if with_cxr else None
    fusion = loaded.score(req, image)
    readers = readers_mode.score(req, image)
    expected = {m: s.probability for m, s in readers.items()}
    assert set(fusion.standalone) == set(expected)
    assert ("cxr" in expected) is with_cxr
    assert "dn" not in fusion.standalone or task == "pneumonia"
    for m, p in expected.items():
        assert abs(fusion.standalone[m] - p) <= 1e-5, m


def test_standalone_heads_use_the_fusion_encoders(loaded):
    for task, fm in loaded.by_task.items():
        assert fm.standalone_state == "installed"
        assert set(fm.standalone) == {m.lower() for m in fm.modalities}
        assert not fm.fallback_modalities
        assert all("bit-identical" in d for d in fm.standalone_detail.values()), task


def test_by_source_is_not_the_fusion_votes(loaded):
    """The fusion's per-modality votes are a different quantity (trained only
    through the late sum); By source must not be their sigmoid."""
    import math

    score = loaded.score(_full_request("pneumonia"), _png())
    for m, logit in score.votes.items():
        vote = 1 / (1 + math.exp(-logit)) if logit > -50 else 0.0
        assert abs(vote - score.standalone[m]) > 1e-6, m


def test_result_labels_standalone_readers_and_internals(loaded):
    from app.fusion import build_fusion_result

    fm = loaded.by_task["pneumonia"]
    req = _full_request("pneumonia")
    score = loaded.score(req, None)
    result = build_fusion_result(req, None, fm, score)
    assert result.unimodal.ehr == pytest.approx(score.standalone["ehr"])
    assert result.unimodal.cxr is None
    assert "standalone Round 1 reader" in (result.sources.rr.model or "")
    info = result.fusion_model
    assert info.standalone == "installed"
    assert set(info.internals.votes) == {"ehr", "rr", "dn"}
    assert info.internals.high.logit == pytest.approx(score.head_logits["high"])
    assert 0.0 <= info.internals.missingness.probability <= 1.0


def _fusion_copy(fake_runs, tmp_path, heads: bool) -> Path:
    root = fake_runs[0]
    out = tmp_path / "round3"
    for folder in ("mortality", "phenotyping"):
        shutil.copytree(root / folder, out / folder)
    if heads:
        for name in ("reader_heads.pt", "reader_heads.json"):
            shutil.copy(root / name, out / name)
    return out


def test_without_reader_heads_fusion_still_serves(fake_runs, tmp_path):
    from app.fusion import build_fusion_result, load_fusion

    folder = _fusion_copy(fake_runs, tmp_path, heads=False)
    models = load_fusion(folder, MEDPATCH, "cpu", 0.75)
    fm = models.by_task["pneumonia"]
    assert fm.standalone_state == "not-installed"
    req = _full_request("pneumonia")
    score = models.score(req, None)
    assert score.standalone == {}
    result = build_fusion_result(req, None, fm, score)
    assert (result.unimodal.ehr, result.unimodal.rr, result.unimodal.dn) == (None, None, None)
    assert result.fusion_model.standalone == "not-installed"
    assert any("Standalone readers not installed" in n for n in result.fusion_model.notes)


def test_reader_heads_whose_sha256_differs_are_refused(fake_runs, tmp_path):
    from app.fusion import FusionLoadError, load_fusion

    folder = _fusion_copy(fake_runs, tmp_path, heads=True)
    manifest = json.loads((folder / "reader_heads.json").read_text())
    manifest["sha256"] = "0" * 64
    (folder / "reader_heads.json").write_text(json.dumps(manifest))
    with pytest.raises(FusionLoadError, match="reader_heads.pt"):
        load_fusion(folder, MEDPATCH, "cpu", 0.75)


def test_an_encoder_mismatch_falls_back_to_the_readers_own_path(fake_runs, tmp_path):
    """A Round 1 reader whose encoder differs from the fusion's is never
    approximated by its head: it runs its own full path, or is unavailable."""
    import torch

    from app import reader_heads
    from app.fusion import load_fusion
    from app.readers import READERS, load_readers

    _, _, r1 = fake_runs
    r1_copy = tmp_path / "r1"
    shutil.copytree(r1, r1_copy)
    spec = next(s for s in READERS if s.key == "cxr-mortality")
    path = r1_copy / spec.checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key = next(k for k in ckpt["state_dict"] if k.startswith("cxr_model.") and k.endswith("weight"))
    ckpt["state_dict"][key] = ckpt["state_dict"][key] + 1e-3
    for alias in [k for k in ckpt["state_dict"] if k.endswith(key[len("cxr_model."):]) and k != key]:
        ckpt["state_dict"][alias] = ckpt["state_dict"][key]
    torch.save(ckpt, path)

    folder = _fusion_copy(fake_runs, tmp_path, heads=False)
    reader_heads.extract(r1_copy, folder)
    models = load_fusion(folder, MEDPATCH, "cpu", 0.75, checkpoint_dir=r1_copy)
    fm = models.by_task["mortality"]
    assert fm.fallback_modalities == {"cxr"}
    assert "own Round 1 reader path" in fm.standalone_detail["cxr"]
    req, image = _full_request("mortality"), _png()
    own = load_readers(MEDPATCH, r1_copy, "cpu", keys={"cxr-mortality"}, confidence=False)
    assert models.score(req, image).standalone["cxr"] == pytest.approx(
        own.score(req, image)["cxr"].probability, abs=1e-6
    )

    # Without the reader's checkpoint on the server it is unavailable, not guessed.
    path.unlink()
    models = load_fusion(folder, MEDPATCH, "cpu", 0.75, checkpoint_dir=r1_copy)
    fm = models.by_task["mortality"]
    assert fm.standalone_state == "partial"
    assert fm.standalone_detail["cxr"].startswith("unavailable")
    assert "cxr" not in models.score(req, image).standalone


def test_service_model_info_reports_the_standalone_readers(fusion_client):
    info = fusion_client.get("/v1/model-info").json()
    for task in ("mortality", "pneumonia"):
        assert info["fusion"][task]["standalone"] == "installed"
        assert all("bit-identical" in d for d in info["fusion"][task]["standaloneDetail"].values())


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


# ── the four synthetic demo patients, through the real readers (when present) ─

DEMO = REPO / "docs" / "demo-examples"
#: The readers-mode table in docs/demo-examples/README.md (percent, 1 decimal).
DEMO_EXPECTED = {
    ("pneumonia", "pneumonia_sick"): {"ehr": 30.7, "rr": 72.0, "dn": 99.9},
    ("pneumonia", "pneumonia_control"): {"ehr": 7.0, "rr": 0.5, "dn": 0.2},
    ("mortality", "mortality_high_risk"): {"ehr": 89.9, "rr": 68.2},
    ("mortality", "mortality_low_risk"): {"ehr": 4.9, "rr": 3.2},
}
DEMO_CXR = {  # (task, image) -> standalone CXR reader, percent
    ("pneumonia", "cxr_pneumonia_covid_ap.jpg"): 42.7,
    ("mortality", "cxr_pneumonia_covid_ap.jpg"): 23.1,
    ("pneumonia", "cxr_pneumonia_lobar.jpg"): 16.5,
    ("mortality", "cxr_pneumonia_lobar.jpg"): 2.9,
}
have_real_heads = have_real and (REAL_FUSION_DIR / "reader_heads.pt").is_file()


def demo_request(task: str, name: str):
    """A demo patient as the Analyze page sends it: CSV row ``hour = h`` (-47..0)
    lands in grid row 47 + h; empty cells are null; numbers stay numbers."""
    import csv

    from app.schemas import EhrInput, NoteInput, PredictRequest

    with open(DEMO / f"{name}_vitals.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    variables = header[1:]
    grid: list[list[float | str | None]] = [[None] * len(variables) for _ in range(48)]
    for row in body:
        hour = int(float(row[0]))
        for i, cell in enumerate(row[1:]):
            cell = cell.strip()
            if not cell:
                continue
            try:
                grid[47 + hour][i] = float(cell)
            except ValueError:
                grid[47 + hour][i] = cell
    notes = [NoteInput(id="r", type="radiology", text=(DEMO / f"{name}_radiology.txt").read_text(encoding="utf-8"))]
    if task == "pneumonia":
        notes.append(NoteInput(id="d", type="discharge", text=(DEMO / f"{name}_discharge.txt").read_text(encoding="utf-8")))
    return PredictRequest(task=task, seed="demo", ehr=EhrInput(variables=variables, values=grid), notes=notes)


@pytest.mark.skipif(not have_real_heads, reason=f"needs the Round 3 runs and reader_heads.pt under {REAL_FUSION_DIR}")
def test_demo_patients_match_the_readers_mode_table():
    from app.fusion import load_fusion

    models = load_fusion(REAL_FUSION_DIR, MEDPATCH, "cpu", 0.75)
    for (task, name), expected in DEMO_EXPECTED.items():
        score = models.score(demo_request(task, name), None)
        got = {m: round(100 * p, 1) for m, p in score.standalone.items()}
        print(f"\n{name}: by source {got}; fusion late {100 * score.probability:.1f}%")
        for m, value in expected.items():
            assert abs(got[m] - value) <= 0.1, (name, m, got[m], value)
    for (task, image), value in DEMO_CXR.items():
        req = demo_request(task, "pneumonia_sick" if task == "pneumonia" else "mortality_high_risk")
        got = round(100 * models.score(req, (DEMO / image).read_bytes()).standalone["cxr"], 1)
        print(f"{image} ({task}): CXR reader {got}%")
        assert abs(got - value) <= 0.1, (image, task, got, value)
