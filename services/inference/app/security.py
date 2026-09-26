"""HMAC-SHA256 request verification for API → inference calls.

The API signs ``timestamp + "." + body`` with a shared secret and sends:

    X-PV-Timestamp: <unix seconds>
    X-PV-Signature: <hex hmac-sha256>

A request is rejected when the signature does not match or the timestamp is
more than ``hmac_max_skew_seconds`` away from now (replay window).
"""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Depends, Header, HTTPException, Request, status

from .config import Settings, get_settings

TIMESTAMP_HEADER = "X-PV-Timestamp"
SIGNATURE_HEADER = "X-PV-Signature"


def compute_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Return the hex HMAC-SHA256 over ``timestamp + "." + body``."""
    message = timestamp.encode("utf-8") + b"." + body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_signature(
    *,
    secret: str,
    timestamp: str,
    signature: str,
    body: bytes,
    max_skew_seconds: int,
    now: float | None = None,
) -> None:
    """Raise :class:`HTTPException` unless the request is authentic and fresh.

    Kept free of FastAPI plumbing so it can be unit-tested directly.
    """
    try:
        sent_at = int(timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid timestamp"
        ) from exc

    current = time.time() if now is None else now
    if abs(current - sent_at) > max_skew_seconds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="timestamp outside allowed skew"
        )

    expected = compute_signature(secret, timestamp, body)
    # Constant-time comparison: a short-circuiting == would leak the signature
    # one byte at a time.
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature"
        )


async def require_signed_request(
    request: Request,
    x_pv_timestamp: str = Header(default="", alias=TIMESTAMP_HEADER),
    x_pv_signature: str = Header(default="", alias=SIGNATURE_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency enforcing the signature on a route.

    ``settings`` must arrive via ``Depends``; a plain default would make
    FastAPI treat the Settings model as a second request-body field and
    silently switch the endpoint to embedded-body parsing.
    """
    cfg = settings
    if not cfg.hmac_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="inference service has no HMAC secret configured",
        )
    if not x_pv_timestamp or not x_pv_signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing signature headers"
        )

    verify_signature(
        secret=cfg.hmac_secret,
        timestamp=x_pv_timestamp,
        signature=x_pv_signature,
        body=await request.body(),
        max_skew_seconds=cfg.hmac_max_skew_seconds,
    )
