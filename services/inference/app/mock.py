"""Deterministic mock predictions.

No trained checkpoints exist in this repo yet (``checkpoints/`` is empty), so
MOCK_MODE is the only mode that runs today. Output here is *synthetic* but
schema-valid and reproducible: the same seed and the same available modalities
always produce the same numbers, which keeps the API, the UI and the e2e tests
deterministic while the real weights are still training.

Nothing in this module reads real patient content — only which modalities are
present, and the caller-supplied seed string.
"""

from __future__ import annotations

import hashlib
import random

from .schemas import (
    CXR_PATCH_GRID,
    EHR_WINDOW_HOURS,
    Alphas,
    ConfidenceMaps,
    FractionAbove,
    JointScores,
    Missingness,
    NoteSpan,
    PredictRequest,
    PredictionResult,
    UnimodalScores,
)

MOCK_MODEL_VERSION = "mock-0.1.0"

# Which encoder each note type feeds. Radiology reports and the other in-stay
# notes go to the RR branch; discharge notes to DN. Matches the modality split
# in the MedPatch training scripts.
_DN_NOTE_TYPES = {"discharge"}


def _seeded_rng(*parts: str) -> random.Random:
    """A Random seeded by a stable hash of the given strings.

    Uses SHA-256 rather than ``hash()`` so results are stable across processes
    (Python salts string hashing per-process).
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _jitter(rng: random.Random, base: float, spread: float) -> float:
    """A probability near ``base``, clamped away from the exact 0/1 endpoints."""
    return min(0.999, max(0.001, base + rng.uniform(-spread, spread)))


def build_mock_result(req: PredictRequest) -> PredictionResult:
    """Produce a complete, internally consistent MedPatch-shaped result."""
    notes = req.notes or []
    has_ehr = req.ehr is not None
    has_cxr = req.cxr is not None
    has_rr = any(n.type not in _DN_NOTE_TYPES for n in notes)
    has_dn = any(n.type in _DN_NOTE_TYPES for n in notes)

    availability = {"ehr": has_ehr, "cxr": has_cxr, "rr": has_rr, "dn": has_dn}
    rng = _seeded_rng(
        req.seed or "no-seed",
        req.task,
        "".join(k for k, v in sorted(availability.items()) if v),
    )

    # A per-stay baseline risk, so the same stay keeps a coherent risk level
    # across repeated scorings while still varying between stays.
    base = rng.uniform(0.08, 0.72)

    unimodal = UnimodalScores(
        ehr=_jitter(rng, base, 0.12) if has_ehr else None,
        cxr=_jitter(rng, base, 0.15) if has_cxr else None,
        rr=_jitter(rng, base, 0.14) if has_rr else None,
        dn=_jitter(rng, base, 0.14) if has_dn else None,
    )

    joint = JointScores(high=_jitter(rng, base, 0.08), low=_jitter(rng, base, 0.18))
    missingness = Missingness(vector=availability, prediction=_jitter(rng, base, 0.2))

    # F6: α weights are a softmax, so they must sum to 1 over the branches that
    # actually contributed. Branches for absent modalities are omitted, not
    # zeroed, matching how the fusion masks them out.
    branches: list[str] = ["high", "low", "miss"] + [m for m, present in availability.items() if present]
    raw = {b: rng.uniform(0.5, 2.0) for b in branches}
    total = sum(raw.values())
    weights = {b: v / total for b, v in raw.items()}
    alphas = Alphas(
        high=weights["high"],
        low=weights["low"],
        miss=weights["miss"],
        ehr=weights.get("ehr"),
        cxr=weights.get("cxr"),
        rr=weights.get("rr"),
        dn=weights.get("dn"),
    )

    # The fused score is the α-weighted combination of the branch predictions,
    # so the report's "modality contribution" chart reconciles with the number
    # shown beside it.
    branch_scores = {
        "high": joint.high,
        "low": joint.low,
        "miss": missingness.prediction,
        "ehr": unimodal.ehr,
        "cxr": unimodal.cxr,
        "rr": unimodal.rr,
        "dn": unimodal.dn,
    }
    probability = sum(weights[b] * (branch_scores[b] or 0.0) for b in branches)
    probability = min(0.999, max(0.001, probability))

    confidence = _build_confidence(req, rng, availability, notes)
    return PredictionResult(
        probability=probability,
        unimodal=unimodal,
        joint=joint,
        missingness=missingness,
        alphas=alphas,
        confidence=confidence,
    )


def _build_confidence(
    req: PredictRequest,
    rng: random.Random,
    availability: dict[str, bool],
    notes: list,
) -> ConfidenceMaps:
    """Token-level confidence maps (F2) and the θ-crossing fractions (F4)."""
    theta = req.theta
    explain = req.return_explanations

    ehr_timesteps: list[float] | None = None
    if availability["ehr"] and explain:
        # A slow drift plus noise reads like a plausible per-hour confidence
        # band rather than white noise.
        drift = rng.uniform(-0.2, 0.2)
        start = rng.uniform(0.55, 0.9)
        ehr_timesteps = [
            min(0.999, max(0.001, start + drift * (h / EHR_WINDOW_HOURS) + rng.uniform(-0.06, 0.06)))
            for h in range(EHR_WINDOW_HOURS)
        ]

    cxr_patch_grid: list[list[float]] | None = None
    if availability["cxr"] and explain:
        # Confidence peaks near a randomly placed focus, mimicking a region of
        # interest on the radiograph.
        fy, fx = rng.uniform(0, CXR_PATCH_GRID), rng.uniform(0, CXR_PATCH_GRID)
        cxr_patch_grid = []
        for y in range(CXR_PATCH_GRID):
            row = []
            for x in range(CXR_PATCH_GRID):
                dist = ((y - fy) ** 2 + (x - fx) ** 2) ** 0.5 / CXR_PATCH_GRID
                row.append(min(0.999, max(0.001, 0.92 - 0.55 * dist + rng.uniform(-0.05, 0.05))))
            cxr_patch_grid.append(row)

    note_spans: list[NoteSpan] | None = None
    if notes and explain:
        note_spans = []
        for note in notes:
            length = len(note.text)
            if length == 0:
                continue
            # Carve the note into a handful of contiguous spans. Offsets are
            # derived from the text length only — the content is never read.
            span_count = min(6, max(1, length // 220))
            step = length // span_count
            for i in range(span_count):
                start = i * step
                end = length if i == span_count - 1 else min(length, start + step)
                note_spans.append(
                    NoteSpan(
                        note_id=note.id,
                        start=start,
                        end=end,
                        confidence=_jitter(rng, 0.7, 0.28),
                    )
                )

    def fraction(values: list[float] | None) -> float | None:
        if not values:
            return None
        return sum(1 for v in values if v >= theta) / len(values)

    flat_cxr = [v for row in cxr_patch_grid for v in row] if cxr_patch_grid else None
    rr_conf = [s.confidence for s in (note_spans or []) if _note_type(notes, s.note_id) not in _DN_NOTE_TYPES]
    dn_conf = [s.confidence for s in (note_spans or []) if _note_type(notes, s.note_id) in _DN_NOTE_TYPES]

    return ConfidenceMaps(
        theta=theta,
        fraction_above=FractionAbove(
            ehr=fraction(ehr_timesteps),
            cxr=fraction(flat_cxr),
            rr=fraction(rr_conf or None),
            dn=fraction(dn_conf or None),
        ),
        ehr_timesteps=ehr_timesteps,
        cxr_patch_grid=cxr_patch_grid,
        note_spans=note_spans,
    )


def _note_type(notes: list, note_id: str) -> str:
    for n in notes:
        if n.id == note_id:
            return str(n.type)
    return "radiology"
