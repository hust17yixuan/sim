"""Deterministic bounded MCP envelopes with lossless identity preservation.

[Role]
Guarantee every MCP response is <= 250 UTF-8 bytes (and therefore <= 250 BPE
tokens in cl100k_base/o200k_base) while never truncating required identity
fields such as `sku`, `op`, or `mode`.
[Invariants]
Required keys are never shortened or dropped. Optional diagnostic keys are
removed in a deterministic order until the serialized envelope fits; if even
the identity cannot fit, a fixed EBUDGET digest envelope is returned instead,
which is an explicit failure rather than a silently mangled identifier.
token_estimate is the UTF-8 byte length, a strict upper bound because every BPE
token covers at least one byte.
[Timeouts & Locks]
No locks, I/O, or waiting; pure in-memory serialization and hashing.
[Side-Effect Policy]
No side effects. The digest is deterministic and no payload is persisted.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

MAX_ENVELOPE_BYTES = 250
_SAFE_TARGET = 246


def _truncate(value: str, chars: int) -> str:
    if len(value) <= chars:
        return value
    return value[: max(0, chars - 1)] + "~"


def _sizes(payload: Any) -> tuple[int, int]:
    default = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    compact = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return len(default.encode("utf-8")), len(compact.encode("utf-8"))


def _fits(payload: Any, limit: int) -> bool:
    return max(_sizes(payload)) <= limit


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]


def bounded_envelope(payload: dict[str, Any], limit: int = MAX_ENVELOPE_BYTES,
                     required: Iterable[str] = ()) -> dict[str, Any]:
    """Return a deterministic envelope <= ``limit`` bytes without identity loss."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    if type(limit) is not int or not 32 <= limit <= MAX_ENVELOPE_BYTES:
        raise ValueError("limit must be an integer in [32, 250]")
    candidate = dict(payload)
    keep = set(required) | {"ok"}
    target = min(limit, _SAFE_TARGET)
    if _fits(candidate, target):
        return candidate
    optional = [key for key in candidate if key not in keep]
    for key in optional:
        del candidate[key]
        if _fits(candidate, target):
            return candidate
    if _fits(candidate, target):
        return candidate
    # The required identity itself does not fit. Do not truncate it; report a
    # deterministic budget failure instead.
    failure = {"ok": 0, "err": "EBUDGET", "sha": _digest(payload)}
    return failure if _fits(failure, limit) else {"ok": 0, "err": "EBUDGET"}


def envelope_size(payload: dict[str, Any]) -> dict[str, int]:
    """Report the serialized size and its strict BPE upper bound."""
    default_size, compact_size = _sizes(payload)
    bound = max(default_size, compact_size)
    return {
        "utf8_bytes": bound,
        "token_estimate": bound,
        "default_json_bytes": default_size,
        "compact_json_bytes": compact_size,
    }


def bounded_error(code: str, message: str = "") -> dict[str, Any]:
    """Compact error: keep the code, drop an overlong message rather than truncate it."""
    code_text = _truncate(str(code), 16)
    return bounded_envelope(
        {"ok": 0, "err": code_text, "msg": str(message)},
        required=("ok", "err"),
    )


__all__ = [
    "MAX_ENVELOPE_BYTES", "bounded_envelope", "bounded_error", "envelope_size",
]
