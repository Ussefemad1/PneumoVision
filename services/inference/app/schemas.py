"""Request/response contracts for the inference service.

These mirror ``packages/shared`` on the TypeScript side. Shapes follow what the
training code in ``medpatch/`` actually produces — see CLAUDE.md § "Ground truth
from the training code".
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Task = Literal["mortality", "pneumonia"]
ModalityName = Literal["ehr", "cxr", "rr", "dn"]

# The EHR window the LSTM was trained on: 48 hourly bins.
EHR_WINDOW_HOURS = 48
# Discretizer output width: 12 continuous + 47 one-hot + 17 mask columns.
EHR_DISCRETIZED_WIDTH = 76
# timm vit_small_patch16_384 -> 24x24 spatial patches (+1 CLS, excluded here).
CXR_PATCH_GRID = 24
# Upload ceiling for a radiograph, raw bytes. Mirrors ANALYZE_LIMITS in packages/shared.
CXR_MAX_BYTES = 20 * 1024 * 1024


#: A calibrated probability in [0, 1].
Prob = Annotated[float, Field(ge=0.0, le=1.0)]


class EhrInput(BaseModel):
    """Hourly EHR window.

    ``values`` is the *raw* 17-variable grid, one row per hour, ``null`` where
    nothing was charted. Discretisation (one-hot + mask + imputation) and
    normalisation happen inside this service using the training code's own
    Discretizer/Normalizer, so the API never has to know the 76-column layout.
    """

    model_config = ConfigDict(extra="forbid")

    variables: list[str] = Field(description="The 17 variable names, in training order")
    values: list[list[float | str | None]] = Field(
        description=f"{EHR_WINDOW_HOURS} rows x 17 columns; null where not charted"
    )
    mask: list[list[bool]] | None = Field(
        default=None, description="Optional explicit observed-mask, same shape as values"
    )


class CxrInput(BaseModel):
    """The radiograph, carried as bytes inside the HMAC-signed body.

    ``presignedUrl`` is kept for a future object-storage path but is not used:
    a request without ``dataB64`` is rejected (see ``cxr.decode_cxr``).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image_id: str | None = Field(default=None, alias="imageId", max_length=64)
    content_type: Literal["image/png", "image/jpeg"] = Field(
        default="image/png", alias="contentType"
    )
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # 4/3 of the raw ceiling, rounded up to a whole base64 quantum.
    data_b64: str | None = Field(
        default=None, alias="dataB64", max_length=-(-CXR_MAX_BYTES // 3) * 4
    )
    presigned_url: str | None = Field(default=None, alias="presignedUrl")


class NoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    text: str
    # radiology/progress/nursing -> RR encoder; discharge -> DN encoder.
    # The API is responsible for never sending discharge notes for mortality.
    type: Literal["radiology", "progress", "nursing", "discharge"] = "radiology"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    task: Task
    # Used only to seed deterministic mock output; never logged.
    seed: str | None = None
    ehr: EhrInput | None = None
    cxr: CxrInput | None = None
    notes: list[NoteInput] | None = None
    theta: Prob = 0.75
    return_explanations: bool = Field(default=True, alias="returnExplanations")


class UnimodalScores(BaseModel):
    """F1: each modality's own prediction, for the modality breakdown cards."""

    model_config = ConfigDict(extra="forbid")

    ehr: Prob | None = None
    cxr: Prob | None = None
    rr: Prob | None = None
    dn: Prob | None = None


class JointScores(BaseModel):
    """F4: the confidence-patched high/low group predictions."""

    model_config = ConfigDict(extra="forbid")

    high: Prob
    low: Prob


class Missingness(BaseModel):
    """F5: indicator vector a ∈ {0,1}^M and the missingness branch prediction."""

    model_config = ConfigDict(extra="forbid")

    vector: dict[str, bool]
    prediction: Prob


class Alphas(BaseModel):
    """F6: softmax-normalised late-fusion weights, one per contributing branch."""

    model_config = ConfigDict(extra="forbid")

    high: float
    low: float
    miss: float
    ehr: float | None = None
    cxr: float | None = None
    rr: float | None = None
    dn: float | None = None


class NoteSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    note_id: str = Field(alias="noteId")
    start: int
    end: int
    confidence: Prob


class FractionAbove(BaseModel):
    """Share of each modality's tokens whose confidence clears θ."""

    model_config = ConfigDict(extra="forbid")

    ehr: Prob | None = None
    cxr: Prob | None = None
    rr: Prob | None = None
    dn: Prob | None = None


class ConfidenceMaps(BaseModel):
    """F2: token-level confidence γ = max(σ(l̂), 1 − σ(l̂))."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    theta: Prob
    fraction_above: FractionAbove = Field(alias="fractionAbove")
    # One value per hourly bin of the 48-hour window.
    ehr_timesteps: list[Prob] | None = Field(default=None, alias="ehrTimesteps")
    # 24x24 ViT patch grid, row-major. CLS token excluded.
    cxr_patch_grid: list[list[Prob]] | None = Field(default=None, alias="cxrPatchGrid")
    note_spans: list[NoteSpan] | None = Field(default=None, alias="noteSpans")


class ReceivedEhr(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Hourly bins (of 48) with at least one charted value.
    hours: int
    variables_present: list[str] = Field(alias="variablesPresent")


class ReceivedCxr(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str
    width: int
    height: int


class ReceivedNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["radiology", "progress", "nursing", "discharge"]
    tokens: int
    chunks: int


class Received(BaseModel):
    """Proof of what reached the model, derived from the decoded request."""

    model_config = ConfigDict(extra="forbid")

    ehr: ReceivedEhr | None
    cxr: ReceivedCxr | None
    notes: list[ReceivedNote]
    mode: Literal["mock", "model"]


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # The final calibrated ŷ_late.
    probability: Prob
    unimodal: UnimodalScores
    joint: JointScores
    missingness: Missingness
    alphas: Alphas
    confidence: ConfidenceMaps
    received: Received


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    result: PredictionResult
    model_version: str = Field(alias="modelVersion")
    latency_ms: int = Field(alias="latencyMs")
    # True when the numbers are synthetic rather than from trained weights.
    mock: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str
    mock: bool
    device: str
    tasks: list[Task]
    theta: Prob
    ehr_variables: list[str] = Field(alias="ehrVariables")
    ehr_window_hours: int = Field(alias="ehrWindowHours")
    ehr_discretized_width: int = Field(alias="ehrDiscretizedWidth")
    cxr_patch_grid: int = Field(alias="cxrPatchGrid")
    cxr_image_size: int = Field(alias="cxrImageSize")
    text_model: str = Field(alias="textModel")


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: Literal["ok", "degraded"]
    mock: bool
    models_loaded: bool = Field(alias="modelsLoaded")
    device: str
    warmed_up: bool = Field(alias="warmedUp")
