"""Physical-provenance admission for declared SOL minima.

[Role]
  Decide whether one declared service-regime completion floor may enter the
  `declared_physical` SOL path, or must stay in `empirical_floor_ms` only.
  Also recognize a measured Stage latency so the caller can strip the matching
  program minimum without weakening genuine declared minima.

[Invariants]
  * Physical admission is affirmative: a non-empty mapping must declare a
    recognized architectural/specification ``basis`` and carry no observation,
    profile, capture, fit, training, calibration or other empirical key.
  * Missing/empty provenance is empirical, never a default physical input.
  * `response_functions.provenance_roles` is consulted when importable; its
    training/validation roles and empirical source tokens veto admission.
  * A non-completion regime, absent rule, or non-finite duration contributes
    nothing; it is not silently coerced to a floor.

[Timeouts & Locks]
  One selected rule lookup and a bounded provenance walk; no locks.

[Side-Effect Policy]
  No I/O, environment lookup, mutation or measurement.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


PHYSICAL_BASES = frozenset({
    "declared", "declared_physical", "declared_minimum", "declared_hardware",
    "declared_hardware_minimum", "architectural", "architectural_spec",
    "architecture_spec", "hardware_spec", "hardware_specification",
    "specification", "spec_sheet",
})

_EMPIRICAL_TOKENS = (
    "observ", "profil", "captur", "dataset", "fitted", "fitting", "fit",
    "train", "valid", "calibrat", "measur", "empiric", "trace_source",
    "csv", "heldout", "ground_truth", "benchmark_result",
)


def _empirical_text(value):
    text = str(value).lower()
    return any(token in text for token in _EMPIRICAL_TOKENS)


def _has_empirical_key(value):
    if isinstance(value, Mapping):
        return any(_empirical_text(key) or _has_empirical_key(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_empirical_key(item) for item in value)
    return _empirical_text(value) if isinstance(value, str) else False


def affirmative_physical_provenance(provenance):
    """True only for an explicit non-empirical architectural/spec basis."""
    if not isinstance(provenance, Mapping) or not provenance:
        return False
    basis = provenance.get("basis")
    if not isinstance(basis, str) or basis.strip().lower() not in PHYSICAL_BASES:
        return False
    if _has_empirical_key(provenance):
        return False
    try:
        from .response_functions import provenance_roles
        roles = provenance_roles(provenance)
    except TimeoutError:
        raise
    except Exception:  # pragma: no cover - defensive import boundary
        roles = {}
    if roles.get("training") or roles.get("validation"):
        return False
    return not any(_empirical_text(token) for token in roles.get("other", ()))


def is_measured_minimum(value, measured_floor):
    """True when a declared minimum is numerically the observed floor."""
    return (measured_floor > 0 and math.isclose(
        float(value), float(measured_floor), rel_tol=1e-9, abs_tol=1e-9))


def service_regime_floor(execution, context):
    """Return ``(empirical_floor_ms, physical_admitted)`` for one lowering."""
    diagnostics = dict(getattr(context, "service_regime", None) or {})
    if diagnostics.get("execution_mode") != "completion_floor":
        return 0.0, True
    rule_id = str(diagnostics.get("rule_id") or "")
    raw = next((raw for raw, _key in getattr(
        getattr(execution, "regimes", None), "rules", ())
        if str(raw.get("id")) == rule_id), {})
    admitted = affirmative_physical_provenance(raw.get("provenance"))
    try:
        duration = float(diagnostics.get("response_duration_ms") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        return 0.0, admitted
    return (0.0, True) if admitted else (duration, False)


__all__ = [
    "PHYSICAL_BASES", "affirmative_physical_provenance", "is_measured_minimum",
    "service_regime_floor",
]
