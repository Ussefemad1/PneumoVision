"""FastAPI application for the PneumoVision inference service.

Internal network only — never published through Nginx. Every request must
carry a valid HMAC signature (``security.py``).

Logging rule (Section 6): never log input content. Only the request id, task,
availability vector and latency.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .cxr import decode_cxr
from .mock import MOCK_MODEL_VERSION, build_mock_result
from .schemas import (
    CXR_PATCH_GRID,
    EHR_DISCRETIZED_WIDTH,
    EHR_WINDOW_HOURS,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
from .security import require_signed_request

logger = logging.getLogger("pneumovision.inference")

# The 17 variables in the exact order medpatch's discretizer emits them.
EHR_VARIABLES = [
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

CXR_IMAGE_SIZE = 384
TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def resolve_device(preference: str) -> str:
    """Pick the torch device without importing torch in mock mode."""
    if preference != "auto":
        return preference
    try:
        import torch  # noqa: PLC0415 — optional in mock mode
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


READERS_MODEL_VERSION = "readers-r1+mock-fusion"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.device = resolve_device(settings.torch_device)
    app.state.readers = None
    if settings.mode == "mock":
        # Nothing to load; mock output is generated per request.
        app.state.models_loaded = False
        app.state.warmed_up = True
        logger.info("inference service started in mock mode (no checkpoints loaded)")
    else:
        # Imported here so mock mode never needs torch installed.
        from .readers import load_readers  # noqa: PLC0415

        app.state.readers = load_readers(
            settings.medpatch_root, settings.checkpoint_dir, app.state.device
        )
        app.state.models_loaded = True
        app.state.warmed_up = True
        logger.info(
            "inference service started in readers mode on %s: %s",
            app.state.device,
            ", ".join(app.state.readers.keys),
        )
    yield


app = FastAPI(
    title="PneumoVision Inference",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/v1/openapi.json",
    lifespan=lifespan,
)

SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.get("/v1/health", response_model=HealthResponse)
async def health(request: Request, settings: SettingsDep) -> HealthResponse:
    """Unsigned liveness probe — carries no patient data."""
    return HealthResponse(
        status="ok",
        mock=settings.mode == "mock",
        models_loaded=bool(getattr(request.app.state, "models_loaded", False)),
        device=str(getattr(request.app.state, "device", "cpu")),
        warmed_up=bool(getattr(request.app.state, "warmed_up", False)),
    )


@app.get("/v1/model-info", response_model=ModelInfo)
async def model_info(request: Request, settings: SettingsDep) -> ModelInfo:
    """Expected input spec, so the API can assert its payloads match."""
    return ModelInfo(
        version=MOCK_MODEL_VERSION if settings.mode == "mock" else READERS_MODEL_VERSION,
        mock=settings.mode == "mock",
        device=str(getattr(request.app.state, "device", "cpu")),
        tasks=["mortality", "pneumonia"],
        theta=settings.default_theta,
        ehr_variables=EHR_VARIABLES,
        ehr_window_hours=EHR_WINDOW_HOURS,
        ehr_discretized_width=EHR_DISCRETIZED_WIDTH,
        cxr_patch_grid=CXR_PATCH_GRID,
        cxr_image_size=CXR_IMAGE_SIZE,
        text_model=TEXT_MODEL,
    )


@app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(require_signed_request)])
async def predict(body: PredictRequest, request: Request, settings: SettingsDep) -> PredictResponse:
    started = time.perf_counter()

    # Decode and verify the radiograph before anything else: a hash mismatch
    # or unreadable image is a 400, never a silently image-less prediction.
    cxr = decode_cxr(body.cxr) if body.cxr is not None else None

    readers = getattr(request.app.state, "readers", None)
    if readers is None:
        result = build_mock_result(body, cxr)
        version = MOCK_MODEL_VERSION
    else:
        from .readers import ReaderInputError  # noqa: PLC0415

        try:
            # Torch work off the event loop, so health checks stay responsive.
            scores = await run_in_threadpool(
                readers.score, body, cxr.data if cxr is not None else None
            )
        except ReaderInputError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        result = build_mock_result(body, cxr, scores)
        version = READERS_MODEL_VERSION

    latency_ms = int((time.perf_counter() - started) * 1000)

    # Content-free log line: availability vector, never the inputs themselves.
    logger.info(
        "predict",
        extra={
            "request_id": request.headers.get("x-request-id", ""),
            "task": body.task,
            "availability": {
                "ehr": body.ehr is not None,
                "cxr": body.cxr is not None,
                "notes": bool(body.notes),
            },
            "latency_ms": latency_ms,
        },
    )

    return PredictResponse(
        result=result,
        model_version=version,
        latency_ms=latency_ms,
        mock=readers is None,
    )
