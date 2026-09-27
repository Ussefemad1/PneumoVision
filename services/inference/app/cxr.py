"""Decoding and verifying the chest X-ray carried inside the signed request.

The API sends the image bytes base64-encoded in the HMAC-signed body rather
than a URL: the API's image route needs a clinician cookie the service does
not have, and in single-container deployments the file only exists on the
API's disk. Carrying the bytes also means the signature covers the pixels.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass

from fastapi import HTTPException, status

from .schemas import CXR_MAX_BYTES, CxrInput

_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG"}


@dataclass(frozen=True)
class DecodedCxr:
    data: bytes
    sha256: str
    width: int
    height: int


def _reject(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def decode_cxr(cxr: CxrInput) -> DecodedCxr:
    """Decode, integrity-check and measure the image, or raise a 400."""
    if cxr.data_b64 is None:
        # presignedUrl is reserved for a future object-storage path.
        raise _reject("cxr.dataB64 is required; presignedUrl is not supported yet")

    try:
        data = base64.b64decode(cxr.data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _reject("cxr.dataB64 is not valid base64") from exc

    if len(data) > CXR_MAX_BYTES:
        raise _reject("cxr exceeds the 20 MB limit")

    digest = hashlib.sha256(data).hexdigest()
    if digest != cxr.sha256:
        raise _reject("cxr sha256 does not match the decoded bytes")

    # Imported lazily so importing the app does not require Pillow at build time.
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            width, height = img.size
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise _reject("cxr bytes are not a readable image") from exc

    if fmt != _FORMATS[cxr.content_type]:
        raise _reject(f"cxr contentType {cxr.content_type} does not match the {fmt} bytes")

    return DecodedCxr(data=data, sha256=digest, width=width, height=height)
