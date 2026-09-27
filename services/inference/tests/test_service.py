"""Inference service tests. No MIMIC data — synthetic fixtures only."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_SECRET = "test-secret-at-least-16-chars"
os.environ.setdefault("INFERENCE_HMAC_SECRET", TEST_SECRET)
os.environ.setdefault("MOCK_MODE", "true")

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import CXR_PATCH_GRID, EHR_WINDOW_HOURS, PredictRequest  # noqa: E402
from app.security import compute_signature, verify_signature  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def signed_headers(body: bytes, *, secret: str = TEST_SECRET, ts: str | None = None) -> dict[str, str]:
    timestamp = ts or str(int(time.time()))
    return {
        "X-PV-Timestamp": timestamp,
        "X-PV-Signature": compute_signature(secret, timestamp, body),
        "Content-Type": "application/json",
    }


def synthetic_ehr() -> dict:
    """A 48x17 window of plausible-but-fabricated values."""
    variables = [
        "Capillary refill rate",
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
        "Glucose",
        "Heart Rate",
        "Height",
        "Mean blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "Weight",
        "pH",
    ]
    rows: list[list[object]] = []
    for hour in range(EHR_WINDOW_HOURS):
        row: list[object] = [None] * 17
        row[8] = 80 + (hour % 11)          # Heart Rate
        row[11] = 97 - (hour % 4)          # Oxygen saturation
        row[12] = 18 + (hour % 5)          # Respiratory rate
        rows.append(row)
    return {"variables": variables, "values": rows}


def synthetic_png(width: int = 20, height: int = 12) -> bytes:
    """A small synthetic greyscale gradient. Never a real radiograph."""
    import io

    from PIL import Image

    img = Image.new("L", (width, height))
    img.putdata([(x * 13 + y * 7) % 256 for y in range(height) for x in range(width)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def synthetic_cxr(data: bytes | None = None, **overrides: object) -> dict:
    import base64
    import hashlib

    data = synthetic_png() if data is None else data
    payload = {
        "imageId": "img-1",
        "contentType": "image/png",
        "sha256": hashlib.sha256(data).hexdigest(),
        "dataB64": base64.b64encode(data).decode("ascii"),
    }
    payload.update(overrides)
    return payload


# ── health / model-info ──────────────────────────────────────────────────────


def test_health_reports_mock_mode(client: TestClient):
    res = client.get("/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["mock"] is True
    assert body["warmedUp"] is True


def test_model_info_matches_the_training_input_spec(client: TestClient):
    body = client.get("/v1/model-info").json()
    assert len(body["ehrVariables"]) == 17
    assert body["ehrVariables"][0] == "Capillary refill rate"
    # The LSTM takes the 76-column discretized tensor, not 17 raw features.
    assert body["ehrDiscretizedWidth"] == 76
    assert body["ehrWindowHours"] == 48
    assert body["cxrPatchGrid"] == 24
    assert body["cxrImageSize"] == 384
    assert body["textModel"] == "emilyalsentzer/Bio_ClinicalBERT"
    assert body["theta"] == 0.75


# ── HMAC verification ────────────────────────────────────────────────────────


def test_predict_rejects_unsigned_request(client: TestClient):
    res = client.post("/v1/predict", json={"task": "mortality"})
    assert res.status_code == 401


def test_predict_rejects_bad_signature(client: TestClient):
    import json

    body = json.dumps({"task": "mortality"}).encode()
    headers = signed_headers(body, secret="a-completely-different-secret")
    res = client.post("/v1/predict", content=body, headers=headers)
    assert res.status_code == 401


def test_predict_rejects_stale_timestamp(client: TestClient):
    import json

    body = json.dumps({"task": "mortality"}).encode()
    stale = str(int(time.time()) - 600)
    res = client.post("/v1/predict", content=body, headers=signed_headers(body, ts=stale))
    assert res.status_code == 401


def test_signature_is_bound_to_the_body():
    """Re-signing a different body must not validate against the original."""
    ts = str(int(time.time()))
    sig = compute_signature(TEST_SECRET, ts, b'{"task":"mortality"}')
    with pytest.raises(Exception):
        verify_signature(
            secret=TEST_SECRET,
            timestamp=ts,
            signature=sig,
            body=b'{"task":"pneumonia"}',
            max_skew_seconds=60,
        )


# ── mock predictions ─────────────────────────────────────────────────────────


def post_predict(client: TestClient, payload: dict):
    import json

    body = json.dumps(payload).encode()
    return client.post("/v1/predict", content=body, headers=signed_headers(body))


def test_predict_returns_every_medpatch_intermediate(client: TestClient):
    res = post_predict(
        client,
        {
            "task": "mortality",
            "seed": "stay-001",
            "ehr": synthetic_ehr(),
            "cxr": synthetic_cxr(),
            "notes": [{"id": "n1", "text": "Synthetic radiology report. " * 40, "type": "radiology"}],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    result = body["result"]

    assert body["mock"] is True
    assert 0.0 <= result["probability"] <= 1.0
    # F1 unimodal, F4 joint high/low, F5 missingness, F6 alphas, F2 confidence.
    assert result["unimodal"]["ehr"] is not None
    assert result["unimodal"]["cxr"] is not None
    assert result["unimodal"]["rr"] is not None
    assert result["unimodal"]["dn"] is None
    assert set(result["joint"]) == {"high", "low"}
    assert result["missingness"]["vector"] == {"ehr": True, "cxr": True, "rr": True, "dn": False}
    assert len(result["confidence"]["ehrTimesteps"]) == EHR_WINDOW_HOURS
    assert len(result["confidence"]["cxrPatchGrid"]) == CXR_PATCH_GRID
    assert all(len(row) == CXR_PATCH_GRID for row in result["confidence"]["cxrPatchGrid"])
    assert result["confidence"]["noteSpans"]


def test_alphas_sum_to_one_over_contributing_branches(client: TestClient):
    res = post_predict(
        client, {"task": "mortality", "seed": "stay-002", "ehr": synthetic_ehr()}
    )
    alphas = res.json()["result"]["alphas"]
    present = [v for v in alphas.values() if v is not None]
    assert sum(present) == pytest.approx(1.0)
    # EHR only: cxr/rr/dn branches are absent, not zero.
    assert alphas["cxr"] is None and alphas["rr"] is None and alphas["dn"] is None


def test_output_is_deterministic_for_a_given_seed(client: TestClient):
    payload = {"task": "pneumonia", "seed": "stay-003", "ehr": synthetic_ehr()}
    first = post_predict(client, payload).json()["result"]
    second = post_predict(client, payload).json()["result"]
    assert first == second


def test_different_seeds_give_different_risk(client: TestClient):
    a = post_predict(client, {"task": "mortality", "seed": "stay-A", "ehr": synthetic_ehr()})
    b = post_predict(client, {"task": "mortality", "seed": "stay-B", "ehr": synthetic_ehr()})
    assert a.json()["result"]["probability"] != b.json()["result"]["probability"]


def test_reduced_input_prediction_still_runs(client: TestClient):
    """F5: a missing modality must not block scoring."""
    res = post_predict(client, {"task": "mortality", "seed": "stay-004", "ehr": synthetic_ehr()})
    assert res.status_code == 200
    vector = res.json()["result"]["missingness"]["vector"]
    assert vector == {"ehr": True, "cxr": False, "rr": False, "dn": False}


def test_fraction_above_respects_theta(client: TestClient):
    """F4: the reported fraction must match the confidence map and θ."""
    res = post_predict(
        client,
        {"task": "mortality", "seed": "stay-005", "ehr": synthetic_ehr(), "theta": 0.6},
    )
    confidence = res.json()["result"]["confidence"]
    steps = confidence["ehrTimesteps"]
    expected = sum(1 for v in steps if v >= 0.6) / len(steps)
    assert confidence["theta"] == 0.6
    assert confidence["fractionAbove"]["ehr"] == pytest.approx(expected)


def test_explanations_can_be_switched_off(client: TestClient):
    res = post_predict(
        client,
        {
            "task": "mortality",
            "seed": "stay-006",
            "ehr": synthetic_ehr(),
            "returnExplanations": False,
        },
    )
    confidence = res.json()["result"]["confidence"]
    assert confidence["ehrTimesteps"] is None
    assert confidence["cxrPatchGrid"] is None


def test_unknown_fields_are_rejected(client: TestClient):
    res = post_predict(client, {"task": "mortality", "seed": "x", "surprise": 1})
    assert res.status_code == 422


def test_mock_never_sees_note_text():
    """Span offsets derive from length alone, so content cannot leak into them."""
    from app.mock import build_mock_result

    long_text = "x" * 900
    other_text = "y" * 900
    a = build_mock_result(
        PredictRequest.model_validate(
            {"task": "mortality", "seed": "s", "notes": [{"id": "n", "text": long_text}]}
        )
    )
    b = build_mock_result(
        PredictRequest.model_validate(
            {"task": "mortality", "seed": "s", "notes": [{"id": "n", "text": other_text}]}
        )
    )
    assert a == b


# ── CXR bytes in the signed body (the image never travels as a URL) ─────────


def test_cxr_bytes_are_decoded_and_verified(client: TestClient):
    import hashlib

    png = synthetic_png(20, 12)
    res = post_predict(client, {"task": "pneumonia", "seed": "s", "cxr": synthetic_cxr(png)})
    assert res.status_code == 200, res.text
    got = res.json()["result"]["received"]["cxr"]
    assert got == {"sha256": hashlib.sha256(png).hexdigest(), "width": 20, "height": 12}


def test_cxr_sha256_mismatch_is_rejected(client: TestClient):
    res = post_predict(
        client, {"task": "pneumonia", "seed": "s", "cxr": synthetic_cxr(sha256="0" * 64)}
    )
    assert res.status_code == 400
    assert "sha256" in res.json()["detail"]


def test_cxr_that_is_not_an_image_is_rejected(client: TestClient):
    res = post_predict(
        client, {"task": "pneumonia", "seed": "s", "cxr": synthetic_cxr(b"not an image at all")}
    )
    assert res.status_code == 400


def test_cxr_content_type_must_match_the_bytes(client: TestClient):
    res = post_predict(
        client,
        {"task": "pneumonia", "seed": "s", "cxr": synthetic_cxr(contentType="image/jpeg")},
    )
    assert res.status_code == 400


def test_cxr_invalid_base64_is_rejected(client: TestClient):
    res = post_predict(
        client, {"task": "pneumonia", "seed": "s", "cxr": synthetic_cxr(dataB64="@@not-base64@@")}
    )
    assert res.status_code == 400


def test_cxr_presigned_url_alone_is_not_accepted_yet(client: TestClient):
    res = post_predict(
        client,
        {"task": "pneumonia", "seed": "s", "cxr": {"presignedUrl": "http://minio:9000/fake"}},
    )
    assert res.status_code == 400
    assert "dataB64" in res.json()["detail"]


# ── Bug C: proof of what the model received ──────────────────────────────────


def test_received_reports_the_actual_request(client: TestClient):
    ehr = synthetic_ehr()
    res = post_predict(
        client,
        {
            "task": "pneumonia",
            "seed": "s",
            "ehr": ehr,
            "notes": [
                {"id": "n1", "text": "Bibasilar opacities, worse on the right.", "type": "radiology"},
                {"id": "n2", "text": "word " * 600, "type": "discharge"},
            ],
        },
    )
    assert res.status_code == 200, res.text
    got = res.json()["result"]["received"]
    assert got["mode"] == "mock"
    assert got["cxr"] is None
    assert got["ehr"] == {
        "hours": EHR_WINDOW_HOURS,
        "variablesPresent": ["Heart Rate", "Oxygen saturation", "Respiratory rate"],
    }
    assert got["notes"] == [
        {"id": "n1", "type": "radiology", "tokens": 10, "chunks": 1},
        {"id": "n2", "type": "discharge", "tokens": 600, "chunks": 2},
    ]


def test_received_is_empty_for_absent_modalities(client: TestClient):
    res = post_predict(client, {"task": "mortality", "seed": "s", "notes": [{"id": "a", "text": "ok"}]})
    got = res.json()["result"]["received"]
    assert got["ehr"] is None and got["cxr"] is None
    assert got["notes"] == [{"id": "a", "type": "radiology", "tokens": 1, "chunks": 1}]


def test_token_estimate_matches_the_typescript_vectors():
    """Same vectors as packages/shared/src/text.test.ts — the two must agree."""
    from app.text import chunk_count, estimate_tokens

    assert estimate_tokens("") == 0
    assert estimate_tokens("Bibasilar opacities, worse on the right.") == 10
    assert estimate_tokens("pneumothorax") == 2
    assert estimate_tokens("SpO2 88% on 4L") == 5
    assert chunk_count(0) == 0
    assert chunk_count(512) == 1
    assert chunk_count(513) == 2
