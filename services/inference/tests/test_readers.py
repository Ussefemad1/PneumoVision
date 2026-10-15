"""Readers mode. Synthetic inputs only — no MIMIC data.

The input-shaping and provenance tests run everywhere. The end-to-end tests
need torch and the trained checkpoints under CHECKPOINT_DIR, and are skipped
where those are absent (CI, the Railway image).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_SECRET = "test-secret-at-least-16-chars"
os.environ.setdefault("INFERENCE_HMAC_SECRET", TEST_SECRET)

from app.mock import ReaderScore, build_mock_result  # noqa: E402
from app.readers import READERS, ReaderInputError, dn_text, ehr_rows, rr_text  # noqa: E402
from app.schemas import EhrInput, NoteInput, PredictRequest  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MEDPATCH = REPO / "medpatch"
CHECKPOINTS = Path(os.environ.get("CHECKPOINT_DIR", REPO / "checkpoints"))


@pytest.fixture(scope="module")
def discretizer():
    sys.path.insert(0, str(MEDPATCH))
    from ehr_utils.preprocessing import Discretizer  # type: ignore[import-not-found]

    return Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(MEDPATCH / "ehr_utils" / "resources" / "discretizer_config.json"),
    )


def window(**cells: dict[int, float | str]) -> EhrInput:
    """A 48-hour grid with values only at the given hours."""
    variables = list(cells)
    values = [[cells[v].get(h) for v in variables] for h in range(48)]
    return EhrInput(variables=variables, values=values)


# ── Inputs are shaped exactly as in training ─────────────────────────────────


def test_ehr_rows_place_each_hour_in_its_own_bin(discretizer):
    import numpy as np

    rows = ehr_rows(window(**{"Heart Rate": {0: 80, 1: 81, 47: 120}}), discretizer)
    assert [r[0] for r in rows] == ["0.5", "1.5", "47.5"]

    data, header = discretizer.transform(np.array(rows), end=48.0)
    assert data.shape == (48, 76)
    hr = header.split(",").index("Heart Rate")
    assert data[0, hr] == 80 and data[1, hr] == 81 and data[47, hr] == 120
    # Hours in between are imputed from the previous value, as in training.
    assert data[2, hr] == 81


def test_categorical_values_map_to_the_trained_strings(discretizer):
    rows = ehr_rows(
        window(
            **{
                "Glascow coma scale total": {0: 15},
                "Capillary refill rate": {0: 1},
                "Glascow coma scale eye opening": {0: "4 Spontaneously"},
            }
        ),
        discretizer,
    )
    header = discretizer._header
    assert rows[0][header.index("Glascow coma scale total")] == "15"
    assert rows[0][header.index("Capillary refill rate")] == "1.0"
    assert rows[0][header.index("Glascow coma scale eye opening")] == "4 Spontaneously"


def test_an_untrained_category_is_rejected(discretizer):
    with pytest.raises(ReaderInputError):
        ehr_rows(window(**{"Glascow coma scale eye opening": {0: "Wide awake"}}), discretizer)


def test_an_unknown_variable_is_rejected(discretizer):
    with pytest.raises(ReaderInputError):
        ehr_rows(window(**{"Shoe size": {0: 42}}), discretizer)


def test_rr_reads_radiology_reports_only_and_dn_discharge_only():
    notes = [
        NoteInput(id="a", type="radiology", text="first report"),
        NoteInput(id="b", type="nursing", text="nursing note"),
        NoteInput(id="c", type="radiology", text="second report"),
        NoteInput(id="d", type="discharge", text="summary"),
    ]
    assert rr_text(notes) == "first report second report"
    assert dn_text(notes) == "summary"
    assert rr_text([NoteInput(id="b", type="progress", text="x")]) is None


def test_every_reader_names_a_test_auroc_and_mortality_never_has_dn():
    assert {(r.task, r.modality) for r in READERS} == {
        ("mortality", "ehr"),
        ("mortality", "rr"),
        ("pneumonia", "ehr"),
        ("pneumonia", "rr"),
        ("pneumonia", "dn"),
    }
    assert all(0.5 < r.test_auroc < 1 for r in READERS)


# ── Provenance ───────────────────────────────────────────────────────────────


def test_reader_scores_replace_the_simulated_ones_and_are_labelled():
    req = PredictRequest(
        task="pneumonia",
        seed="s",
        ehr=window(**{"Heart Rate": {0: 90}}),
        notes=[NoteInput(id="n", type="radiology", text="synthetic")],
    )
    result = build_mock_result(
        req, readers={"rr": ReaderScore(probability=0.42, model="RR reader", test_auroc=0.859)}
    )
    assert result.unimodal.rr == 0.42
    assert result.sources.fusion == "mock"
    assert result.sources.rr.kind == "trained" and result.sources.rr.test_auroc == 0.859
    assert result.sources.ehr.kind == "mock"  # sent, but no reader answered
    assert result.sources.cxr is None and result.sources.dn is None
    assert result.received.mode == "readers"


def test_mock_mode_labels_everything_mock():
    req = PredictRequest(task="mortality", seed="s", notes=[NoteInput(id="n", text="synthetic")])
    result = build_mock_result(req)
    assert result.received.mode == "mock"
    assert result.sources.rr.kind == "mock" and result.sources.ehr is None


# ── End to end with the real checkpoints ─────────────────────────────────────

have_torch = importlib.util.find_spec("torch") is not None
have_checkpoints = all((CHECKPOINTS / r.checkpoint).is_file() for r in READERS)
needs_models = pytest.mark.skipif(
    not (have_torch and have_checkpoints), reason="needs torch and the trained checkpoints"
)


@pytest.fixture(scope="module")
def readers_client():
    from fastapi.testclient import TestClient

    from app.config import get_settings

    saved = {k: os.environ.get(k) for k in ("INFERENCE_MODE", "MEDPATCH_ROOT", "CHECKPOINT_DIR")}
    os.environ.update(
        INFERENCE_MODE="readers", MEDPATCH_ROOT=str(MEDPATCH), CHECKPOINT_DIR=str(CHECKPOINTS)
    )
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as client:
        yield client
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


def predict(client, body: dict) -> dict:
    from app.security import compute_signature

    raw = json.dumps(body).encode()
    ts = str(int(time.time()))
    res = client.post(
        "/v1/predict",
        content=raw,
        headers={
            "content-type": "application/json",
            "X-PV-Timestamp": ts,
            "X-PV-Signature": compute_signature(TEST_SECRET, ts, raw),
        },
    )
    assert res.status_code == 200, res.text
    return res.json()


RADIOLOGY = "SYNTHETIC: new right lower lobe consolidation concerning for pneumonia."
DISCHARGE = "SYNTHETIC: treated for community-acquired pneumonia with antibiotics."


@needs_models
def test_readers_mode_scores_each_modality_with_its_trained_reader(readers_client):
    body = {
        "task": "pneumonia",
        "seed": "s",
        "ehr": window(**{"Heart Rate": {h: 100 + h for h in range(48)}}).model_dump(),
        "notes": [
            {"id": "r", "type": "radiology", "text": RADIOLOGY},
            {"id": "d", "type": "discharge", "text": DISCHARGE},
        ],
    }
    out = predict(readers_client, body)
    assert out["modelVersion"] == "readers-r1+mock-fusion"
    assert out["mock"] is False
    sources = out["result"]["sources"]
    assert sources["fusion"] == "mock"
    for modality in ("ehr", "rr", "dn"):
        assert sources[modality]["kind"] == "trained"
        assert 0 < out["result"]["unimodal"][modality] < 1
    assert out["result"]["received"]["mode"] == "readers"

    # Deterministic: the same input gives the same real score.
    again = predict(readers_client, body)
    assert again["result"]["unimodal"] == out["result"]["unimodal"]


@needs_models
def test_readers_mode_mortality_has_no_discharge_reader(readers_client):
    out = predict(
        readers_client,
        {"task": "mortality", "seed": "s", "notes": [{"id": "r", "type": "radiology", "text": RADIOLOGY}]},
    )
    assert out["result"]["sources"]["rr"]["kind"] == "trained"
    assert out["result"]["sources"]["rr"]["testAuroc"] == 0.73
    assert out["result"]["unimodal"]["dn"] is None


@needs_models
def test_readers_mode_rejects_an_untrained_category_with_400(readers_client):
    from app.security import compute_signature

    body = {
        "task": "mortality",
        "ehr": window(**{"Glascow coma scale eye opening": {0: "Wide awake"}}).model_dump(),
    }
    raw = json.dumps(body).encode()
    ts = str(int(time.time()))
    res = readers_client.post(
        "/v1/predict",
        content=raw,
        headers={
            "content-type": "application/json",
            "X-PV-Timestamp": ts,
            "X-PV-Signature": compute_signature(TEST_SECRET, ts, raw),
        },
    )
    assert res.status_code == 400
