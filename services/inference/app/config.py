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
    # No trained checkpoints exist in this repo yet, so MOCK_MODE defaults to
    # true. Flipping it to false requires CHECKPOINT_DIR to hold real weights.
    mock_mode: bool = Field(default=True, alias="MOCK_MODE")

    torch_device: Literal["auto", "cpu", "cuda"] = Field(default="auto", alias="TORCH_DEVICE")

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
