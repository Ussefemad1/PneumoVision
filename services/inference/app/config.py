"""Configuration for the inference service, validated at import time."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings.

    The service is reachable only on the internal Docker network; it is never
    published through Nginx. Every request must still carry a valid HMAC
    signature (see ``security.py``).
    """

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # ── Mode ────────────────────────────────────────────────────────────────
    # INFERENCE_MODE:
    #   mock    — every number simulated; no torch needed (Railway default).
    #   readers — the team's trained single-modality readers (EHR, CXR, RR,
    #             DN) score their own modality; the fused score is their plain
    #             average, labelled so. Needs torch and CHECKPOINT_DIR.
    # When unset, MOCK_MODE (the original flag) decides: true means mock.
    # (The Railway comments here are historical: the deployment is AWS EC2,
    # infra/aws/; Railway is no longer used.)
    inference_mode: Literal["mock", "readers"] | None = Field(default=None, alias="INFERENCE_MODE")
    mock_mode: bool = Field(default=True, alias="MOCK_MODE")

    torch_device: Literal["auto", "cpu", "cuda"] = Field(default="auto", alias="TORCH_DEVICE")
    # torch intra-op threads; unset keeps torch's default (all cores). The
    # EC2 target has 2 vCPUs and sets 2.
    torch_num_threads: int | None = Field(default=None, alias="TORCH_NUM_THREADS", ge=1)

    # ── Paths ───────────────────────────────────────────────────────────────
    medpatch_root: Path = Field(default=Path("/app/medpatch"), alias="MEDPATCH_ROOT")
    checkpoint_dir: Path = Field(default=Path("/app/checkpoints"), alias="CHECKPOINT_DIR")
    normalizer_dir: Path = Field(default=Path("/app/medpatch/normalizers"), alias="NORMALIZER_DIR")

    # ── Security ────────────────────────────────────────────────────────────
    hmac_secret: str = Field(default="", alias="INFERENCE_HMAC_SECRET")
    hmac_max_skew_seconds: int = Field(default=60, alias="INFERENCE_HMAC_MAX_SKEW_SECONDS")

    # ── Model behaviour ─────────────────────────────────────────────────────
    # Confidence-based patching threshold θ (MedPatch §3.3). Matches the value
    # in both MedPatch training scripts.
    default_theta: float = Field(default=0.75, alias="DEFAULT_THETA")

    log_level: str = Field(default="info", alias="LOG_LEVEL")

    @property
    def mode(self) -> Literal["mock", "readers"]:
        if self.inference_mode is not None:
            return self.inference_mode
        if self.mock_mode:
            return "mock"
        raise ValueError(
            "MOCK_MODE=false needs INFERENCE_MODE to say which real mode to run "
            "(INFERENCE_MODE=readers)"
        )

    @field_validator("default_theta")
    @classmethod
    def _theta_in_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("DEFAULT_THETA must be strictly between 0 and 1")
        return v

    @field_validator("hmac_secret")
    @classmethod
    def _secret_long_enough(cls, v: str) -> str:
        # Empty is tolerated only so tests can construct Settings directly;
        # main.py refuses to start without one.
        if v and len(v) < 16:
            raise ValueError("INFERENCE_HMAC_SECRET must be at least 16 characters")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()  # type: ignore[call-arg]
