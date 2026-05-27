"""Decoding for messages mirrored by the browser observer."""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any


def _inflate(raw: bytes) -> str:
    attempts = (
        lambda data: zlib.decompress(data),
        lambda data: zlib.decompress(data, -zlib.MAX_WBITS),
        lambda data: zlib.decompress(data, zlib.MAX_WBITS | 32),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt(raw).decode("utf-8")
        except zlib.error as exc:
            last_error = exc
    raise ValueError(f"could not inflate binary payload: {last_error}")


def decode_observed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("kind")
    if kind == "text":
        return json.loads(str(payload.get("data", "")))
    if kind == "binary":
        raw = base64.b64decode(str(payload.get("base64", "")))
        return json.loads(_inflate(raw))
    raise ValueError(f"unsupported observed payload kind: {kind!r}")
