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
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .cxr import decode_cxr
from .mock import MOCK_MODEL_VERSION, build_mock_result
from .schemas import (
    CXR_PATCH_GRID,
    FusionModelSummary,
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


def configure_torch(num_threads: int | None) -> None:
    """One intra-op pool sized to the machine (2 on the EC2 target), no
    gradient tracking anywhere: the service only ever runs inference."""
    import torch  # noqa: PLC0415

    if num_threads:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:  # already set (a second app in the same process)
            pass
    torch.set_grad_enabled(False)


def rss_mb() -> float | None:
    """Resident memory of this process in MiB (Linux), for /v1/health."""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            pages = int(f.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    import os  # noqa: PLC0415

    return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)


def _load_models(app: FastAPI, settings: Settings) -> None:
    """Load the configured real mode's models into app.state, or raise."""
    # Imported here so mock mode never needs torch installed.
    from .offline import BERT_MODEL, CXR_TIMM_MODEL, check_caches  # noqa: PLC0415

    configure_torch(settings.torch_num_threads)
    check_caches([BERT_MODEL], [CXR_TIMM_MODEL])
    if settings.mode == "fusion":
        from .fusion import load_fusion  # noqa: PLC0415

        app.state.fusion = load_fusion(
            settings.fusion_dir, settings.medpatch_root, app.state.device, settings.default_theta
        )
        app.state.text_model = app.state.fusion.text_model
        loaded = app.state.fusion.version
    else:
        from .readers import load_readers  # noqa: PLC0415

        app.state.readers = load_readers(
            settings.medpatch_root, settings.checkpoint_dir, app.state.device
        )
        app.state.text_model = app.state.readers.text_model
        loaded = ", ".join(app.state.readers.keys)
    logger.info(
        "inference service started in %s mode on %s (%s threads, %s MiB): %s",
        settings.mode,
        app.state.device,
        settings.torch_num_threads or "default",
        rss_mb(),
        loaded,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.device = resolve_device(settings.torch_device)
    app.state.readers = None
    app.state.fusion = None
    app.state.models_loaded = False
    app.state.warmed_up = False
    app.state.load_error = None
    app.state.text_model = None
    if settings.mode == "mock":
        # Nothing to load; mock output is generated per request.
        app.state.warmed_up = True
        logger.info("inference service started in mock mode (no checkpoints loaded)")
    else:
        try:
            # Loading (and the warm-up pass) is torch work: off the event loop.
            await run_in_threadpool(_load_models, app, settings)
        except Exception as exc:  # noqa: BLE001 -- reported by /v1/health, never scored
            # Degraded rather than crashed: /v1/health answers 503 with the
            # reason (so `restart: always` does not hide it in a loop) and
            # /v1/predict refuses, so nothing is ever scored by a model that
            # failed to load.
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("%s mode failed to load: %s", settings.mode, app.state.load_error)
        else:
            app.state.models_loaded = True
            app.state.warmed_up = True
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


@app.get("/v1/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health(request: Request, settings: SettingsDep) -> HealthResponse | JSONResponse:
    """Unsigned readiness probe — carries no patient data.

    503 ("degraded", with the reason) when the configured models did not load.
    """
    error = getattr(request.app.state, "load_error", None)
    body = HealthResponse(
        status="degraded" if error else "ok",
        mode=settings.mode,
        mock=settings.mode == "mock",
        models_loaded=bool(getattr(request.app.state, "models_loaded", False)),
        device=str(getattr(request.app.state, "device", "cpu")),
        warmed_up=bool(getattr(request.app.state, "warmed_up", False)),
        reason=error,
        rss_mb=rss_mb(),
    )
    if error:
        return JSONResponse(status_code=503, content=body.model_dump(by_alias=True))
    return body


@app.get("/v1/model-info", response_model=ModelInfo)
async def model_info(request: Request, settings: SettingsDep) -> ModelInfo:
    """Expected input spec, so the API can assert its payloads match."""
    fusion = getattr(request.app.state, "fusion", None)
    version = {"mock": MOCK_MODEL_VERSION, "readers": READERS_MODEL_VERSION}.get(settings.mode)
    return ModelInfo(
        version=version or (fusion.version if fusion is not None else "medpatch-r3 (not loaded)"),
        mock=settings.mode == "mock",
        device=str(getattr(request.app.state, "device", "cpu")),
        tasks=["mortality", "pneumonia"],
        theta=settings.default_theta,
        ehr_variables=EHR_VARIABLES,
        ehr_window_hours=EHR_WINDOW_HOURS,
        ehr_discretized_width=EHR_DISCRETIZED_WIDTH,
        cxr_patch_grid=CXR_PATCH_GRID,
        cxr_image_size=CXR_IMAGE_SIZE,
        # The BERT actually loaded (from the readers' tokenizer); none in mock.
        text_model=getattr(request.app.state, "text_model", None) or "none (mock mode)",
        fusion=(
            {
                task: FusionModelSummary(
                    run_id=m.run_id,
                    sha256=m.sha256[:12],
                    head=m.head,
                    bert_model_name=m.bert_model_name,
                    theta=m.theta,
                    modalities=m.modalities,
                )
                for task, m in fusion.by_task.items()
            }
            if fusion is not None
            else None
        ),
    )


@app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(require_signed_request)])
async def predict(body: PredictRequest, request: Request, settings: SettingsDep) -> PredictResponse:
    started = time.perf_counter()

    # F8 leakage control: a discharge summary states the outcome, so it is
    # never an input for mortality. The API filters these out already
    # (NOTE_TYPES_BY_TASK); the service refuses them too, in every mode.
    if body.task == "mortality" and any(n.type == "discharge" for n in body.notes or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="discharge notes are never used for mortality (outcome leakage); "
            "send radiology reports only",
        )
    error = getattr(request.app.state, "load_error", None)
    if error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"models not loaded ({settings.mode} mode): {error}",
        )

    # Decode and verify the radiograph before anything else: a hash mismatch
    # or unreadable image is a 400, never a silently image-less prediction.
    cxr = decode_cxr(body.cxr) if body.cxr is not None else None

    readers = getattr(request.app.state, "readers", None)
    fusion = getattr(request.app.state, "fusion", None)
    if fusion is not None:
        from .fusion import FusionInputError, build_fusion_result  # noqa: PLC0415

        try:
            score = await run_in_threadpool(
                fusion.score, body, cxr.data if cxr is not None else None
            )
        except FusionInputError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        result = build_fusion_result(body, cxr, fusion.by_task[body.task], score)
        version = fusion.version
    elif readers is None:
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
        mock=readers is None and fusion is None,
    )
