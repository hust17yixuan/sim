"""Small JSON/shape helpers shared by the public extension surface.

[Role] Convert public declaration data into deterministic, immutable values.
[Invariants] Only finite JSON values are accepted; mapping keys become strings.
  Unknown objects are rejected instead of being stringified into unstable code.
[Timeouts & Locks] Pure in-memory conversion; no I/O or locks.
[Side-Effect Policy] No persistent state and no mutation of caller values.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any

from .errors import ExtensionError


def text(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtensionError(reason)
    return value.strip()


def name(value: Any, reason: str) -> str:
    value = text(value, reason)
    if not all(ch.isalnum() or ch in "_.:-" for ch in value):
        raise ExtensionError(reason + ":invalid_characters")
    return value


def finite_number(value: Any, reason: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ExtensionError(reason)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExtensionError(reason) from exc
    if not math.isfinite(result) or result < minimum:
        raise ExtensionError(reason)
    return result


def json_value(value: Any, reason: str) -> Any:
    """Return a private JSON-compatible copy; reject non-finite/unknown values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtensionError(reason + ":non_finite_number")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_value(item, reason) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item, reason) for item in value]
    raise ExtensionError(reason + ":unsupported_type:" + type(value).__name__)


def canonical_json(value: Any, reason: str) -> str:
    try:
        return json.dumps(json_value(value, reason), sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExtensionError(reason + ":not_serializable") from exc


def digest(value: Any, reason: str) -> str:
    return digest_bytes(canonical_json(value, reason).encode("utf-8"))


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _freeze(value: Any, reason: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtensionError(reason + ":non_finite_number")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item, reason)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, reason) for item in value)
    raise ExtensionError(reason + ":unsupported_type:" + type(value).__name__)


def frozen_mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ExtensionError(reason)
    return _freeze(value, reason)


def frozen_tuple(value: Any, reason: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ExtensionError(reason)
    return tuple(_freeze(item, reason) for item in value)
