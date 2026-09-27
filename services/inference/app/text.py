"""Token and chunk estimates for clinical notes.

Bio_ClinicalBERT is not loaded in MOCK_MODE, so the ``received`` summary uses
a cheap WordPiece-shaped estimate instead of the real tokenizer: every
punctuation character is one token, and every alphanumeric run is one token
per six characters (long clinical words split into several word pieces).

The same rule lives in ``packages/shared/src/text.ts`` so the Analyze page's
live counts match what the service reports. Both sides are tested against the
same vectors — change them together.
"""

from __future__ import annotations

import math
import re

#: Bio_ClinicalBERT's maximum sequence length; longer notes are chunked.
TEXT_CHUNK_TOKENS = 512

_PIECE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def estimate_tokens(text: str) -> int:
    total = 0
    for match in _PIECE.finditer(text):
        piece = match.group(0)
        total += max(1, math.ceil(len(piece) / 6)) if piece[0].isascii() and piece[0].isalnum() else 1
    return total


def chunk_count(tokens: int) -> int:
    return math.ceil(tokens / TEXT_CHUNK_TOKENS) if tokens > 0 else 0
