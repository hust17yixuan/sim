"""Lean MCP envelope with a real, dependency-free BPE token bound.

[Role]
  Build the public ``<= 250``-token MCP summary envelope from a hybrid report.
  The bound is enforced on the *entire compact JSON serialization*, and every
  uncontrolled string is symbolized so no path, home marker or arbitrary
  Unicode can reach the wire.

[Invariants]
  * ``bpe_token_upper_bound`` counts UTF-8 bytes; every byte-level BPE token
    covers at least one byte, so the bound is conservative for cl100k_base and
    o200k_base without a tokenizer, network access or cache file.
  * ``status``, ``errors`` and ``diagnostics`` are mandatory and added first;
    optional summary clauses and facts are added only while the whole envelope
    stays inside ``max_tokens``.
  * A budget smaller than the mandatory status envelope raises the explicit
    ``hybrid_kv_mcp_budget_below_mandatory`` error.
  * No model name, path, file name, connector string or diagnostic text is
    emitted verbatim; only bounded symbols and numbers are allowed.
  * ``json.dumps(..., allow_nan=False)`` succeeds for every emitted value.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] No filesystem, network, environment or global mutation.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    bpe_token_upper_bound, sanitize_code, sanitize_symbol, strict_bool, strict_int,
)
from mindie_sim.schema.common import ModelingError

MCP_MAX_TOKENS = 250
SCHEMA = "nr-s7.mcp/1"
#: A soft upper bound only; every code is emitted while the budget allows.
MAX_CODES = 64


@typed_inputs(errors={"text": "hybrid_kv_summary_not_text"})
def summary_token_count(text: str) -> int:
    """Conservative BPE token upper bound of one summary string."""
    if type(text) is not str:
        raise ModelingError("hybrid_kv_summary_not_text")
    return bpe_token_upper_bound(text)


def _compact(envelope: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ModelingError("hybrid_kv_mcp_payload_invalid") from error


def _codes(values: Any) -> list[str]:
    """Every distinct sanitized code, before any budget/count truncation."""
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = (values,)
    if not isinstance(values, Sequence):
        raise ModelingError("hybrid_kv_mcp_codes_invalid")
    result: list[str] = []
    for value in values:
        code = sanitize_code(value)
        if code not in result:
            result.append(code)
    return result


def _number(value: Any, digits: int = 4) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(value, digits)


def _ascii_int(value: Any, field: str) -> int:
    """One numeric field must already be an exact ``int`` (no Unicode digits)."""
    if value is None:
        return 0
    if type(value) is not int:
        raise ModelingError(f"hybrid_kv_mcp_numeric_invalid:{field}")
    return value


def _ascii_number(value: Any, field: str, digits: int = 4,
                  default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"hybrid_kv_mcp_numeric_invalid:{field}")
    number = _number(value, digits)
    if number is None:
        raise ModelingError(f"hybrid_kv_mcp_numeric_invalid:{field}")
    return float(number)


def _ascii_bytes(value: Any, field: str) -> int:
    """A byte-count field may be an int or a finite float, never text."""
    return int(_ascii_number(value, field, digits=0))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _flag(record: Mapping[str, Any], name: str) -> bool:
    return strict_bool(name, record.get(name, False), reason="hybrid_kv_mcp")


def _summary_clauses(report: Mapping[str, Any]) -> list[str]:
    layout = _mapping(report.get("layout"))
    linear = _mapping(report.get("linear_state"))
    memory = _mapping(report.get("memory"))
    dspark = _mapping(report.get("dspark"))
    decode = _mapping(report.get("decode"))
    nz = _mapping(layout.get("kv_nz"))
    pool = _mapping(layout.get("mixed_pool"))
    pd = _mapping(layout.get("component_major_pd"))
    mem = _mapping(linear.get("memory"))
    replay = _mapping(linear.get("replay"))
    standard = _mapping(_mapping(linear.get("comparison")).get("standard"))
    clauses = [
        "st=" + sanitize_symbol(report.get("status"), limit=12),
        "nz=" + ("1" if nz else "0"),
        "mlapo=" + ("1" if _flag(nz, "mlapo_enabled") else "0"),
        "state=" + sanitize_symbol(mem.get("mode"), limit=12),
        "slots=" + str(_ascii_int(mem.get("state_slots_per_sequence"),
                                  "state_slots_per_sequence"))
        + "of" + str(_ascii_int(standard.get("state_slots_per_sequence"),
                                "standard_state_slots_per_sequence")),
        "ratio=" + f"{_ascii_number(mem.get('matrix_to_record_ratio'), 'ratio', 2)}",
        "dspark=" + ("1" if _flag(dspark, "active") else "0"),
        "step=" + f"{_ascii_number(decode.get('step_overhead_ms'), 'step_ms')}ms",
        "replay=" + str(_ascii_int(replay.get("replay_tokens"), "replay_tokens")),
        "mem=" + sanitize_symbol(memory.get("adjusted_status"), limit=18),
        "pool=" + str(_ascii_bytes(pool.get("physical_pool_bytes"),
                                   "physical_pool_bytes")),
        "pd=" + sanitize_symbol(pd.get("selected"), limit=12),
        "inc=" + str(_ascii_bytes(memory.get("total_increment_bytes"),
                                  "total_increment_bytes")),
        "head=" + str(_ascii_bytes(memory.get("adjusted_headroom_bytes"),
                                   "adjusted_headroom_bytes")),
    ]
    return clauses


def _facts(report: Mapping[str, Any]) -> list[tuple[str, Any]]:
    layout = _mapping(report.get("layout"))
    linear = _mapping(report.get("linear_state"))
    memory = _mapping(report.get("memory"))
    dspark = _mapping(report.get("dspark"))
    nz = _mapping(layout.get("kv_nz"))
    pool = _mapping(layout.get("mixed_pool"))
    pd = _mapping(layout.get("component_major_pd"))
    mem = _mapping(linear.get("memory"))
    step = _mapping(linear.get("step_cost"))
    replay = _mapping(linear.get("replay"))
    facts: list[tuple[str, Any]] = [
        ("nz", bool(nz)),
        ("mlapo", _flag(nz, "mlapo_enabled")),
        ("pool", _ascii_bytes(pool.get("physical_pool_bytes"), "pool")),
        ("pd", sanitize_symbol(pd.get("selected"), limit=12)),
        ("state", sanitize_symbol(mem.get("mode"), limit=12)),
        ("slots", _ascii_int(mem.get("state_slots_per_sequence"), "slots")),
        ("ratio", _ascii_number(mem.get("matrix_to_record_ratio"), "ratio", 3)),
        ("step_ms", _ascii_number(step.get("total_ms"), "step_ms")),
        ("replay", _ascii_int(replay.get("replay_tokens"), "replay")),
        ("dspark", _flag(dspark, "active")),
        ("mem", sanitize_symbol(memory.get("adjusted_status"), limit=18)),
        ("inc", _ascii_bytes(memory.get("total_increment_bytes"), "inc")),
        ("head", _ascii_bytes(memory.get("adjusted_headroom_bytes"), "head")),
        ("peak", _ascii_bytes(memory.get("adjusted_peak_hbm_bytes"), "peak")),
    ]
    return facts


def _fits(envelope: Mapping[str, Any], limit: int) -> bool:
    return len(_compact(envelope)) <= limit


def _ascii_payload(envelope: Mapping[str, Any]) -> bytes:
    """Serialize and require printable ASCII ``[0x20..0x7E]`` everywhere."""
    payload = _compact(envelope)
    if any(byte < 0x20 or byte > 0x7E for byte in payload):
        raise ModelingError("hybrid_kv_mcp_non_ascii_payload")
    return payload


def _code_envelope(envelope: dict[str, Any], errors: list[str],
                   diagnostics: list[str], limit: int) -> None:
    """Fill codes first; count the complete distinct set before truncating."""
    total = len(dict.fromkeys([*errors, *diagnostics]))
    full = dict(envelope)
    full["errors"] = list(errors[:MAX_CODES])
    remaining = max(0, MAX_CODES - len(full["errors"]))
    full["diagnostics"] = list(diagnostics[:remaining])
    if total <= MAX_CODES and _fits(full, limit):
        envelope["errors"] = full["errors"]
        envelope["diagnostics"] = full["diagnostics"]
        return
    truncated = dict(envelope)
    truncated["errors"] = []
    truncated["diagnostics"] = []
    truncated["codes_truncated"] = True
    truncated["total_error_count"] = total
    if not _fits(truncated, limit):
        raise ModelingError(f"hybrid_kv_mcp_budget_below_mandatory:{limit}")
    live = dict(truncated)
    emitted = 0
    for field, values in (("errors", errors), ("diagnostics", diagnostics)):
        for code in values:
            if emitted >= MAX_CODES:
                break
            trial = dict(live)
            trial[field] = [*live[field], code]
            if not _fits(trial, limit):
                break
            live = trial
            emitted += 1
        if emitted >= MAX_CODES:
            break
    envelope.update(live)


@typed_inputs
def hybrid_kv_state_mcp_envelope(report: Any, *,
                                 max_tokens: int = MCP_MAX_TOKENS) -> dict[str, Any]:
    """Build the bounded deterministic MCP envelope from a hybrid report."""
    limit = min(strict_int("max_tokens", max_tokens, minimum=1,
                           reason="hybrid_kv_mcp"), MCP_MAX_TOKENS)
    if not isinstance(report, Mapping) or report.get(
            "schema") != "mindie-sim.hybrid-kv-state/1":
        raise ModelingError("hybrid_kv_report_invalid")
    envelope: dict[str, Any] = {
        "schema": SCHEMA,
        "status": sanitize_symbol(report.get("status"), limit=18),
        "errors": [],
        "diagnostics": [],
        "max_tokens": limit,
        "summary": "",
        "summary_tokens": 0,
        "facts": {},
    }
    if not _fits(envelope, limit):
        raise ModelingError(
            f"hybrid_kv_mcp_budget_below_mandatory:{limit}")
    # Diagnostic codes are mandatory content: they are placed before any prose
    # summary, and a truncation is explicit rather than silent.
    _code_envelope(envelope, _codes(report.get("errors")),
                   _codes(report.get("diagnostics")), limit)
    clauses: list[str] = []
    for clause in _summary_clauses(report):
        candidate = " ".join((*clauses, clause))
        trial = dict(envelope)
        trial["summary"] = candidate
        trial["summary_tokens"] = bpe_token_upper_bound(candidate)
        if _fits(trial, limit):
            clauses.append(clause)
    summary = " ".join(clauses)
    if not summary:
        candidate = sanitize_symbol(report.get("status"), limit=18)
        trial = dict(envelope)
        trial["summary"] = candidate
        trial["summary_tokens"] = bpe_token_upper_bound(candidate)
        summary = candidate if _fits(trial, limit) else ""
    envelope["summary"] = summary
    envelope["summary_tokens"] = bpe_token_upper_bound(summary)
    for key, value in _facts(report):
        trial = dict(envelope)
        trial["facts"] = {**envelope["facts"], key: value}
        if _fits(trial, limit):
            envelope["facts"] = trial["facts"]
    _ascii_payload(envelope)
    return envelope


@typed_inputs
def envelope_token_upper_bound(envelope: Mapping[str, Any]) -> int:
    """Conservative BPE token upper bound of the whole compact envelope."""
    return len(_compact(envelope))


__all__ = ["MCP_MAX_TOKENS", "SCHEMA", "envelope_token_upper_bound",
           "hybrid_kv_state_mcp_envelope", "summary_token_count"]
