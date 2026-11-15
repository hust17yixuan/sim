"""`diff-trace` CLI/MCP surface over the counterfactual replay engine.

[Role]
  Expose :func:`attribute_diff` plus the padding-charge helpers used by other
  simulator components.  The attribution mathematics lives in
  ``diff_trace_replay``; this module only adapts catalog entries, parses CLI
  arguments and writes receipts.

[Invariants]
  * ``effective_operator_rows`` uses the pinned catalog entry, never a regex
    guess; an unknown operator is reported as ``semantic_source=unknown_operator``
    and charged at the full bucket rather than silently guessed.
  * Padding charges preserve the layout/group encoding declared by the entry.
  * The CLI passes an explicit CANN version through to the semantic lookup; no
    release is invented when the caller omits one.
  * Every emitted offender command uses the immutable occurrence index and
    shell-quoted paths, and targets an ``--output-dir`` artifact.
  * A required full-artifact persistence failure is an explicit bounded
    delivery failure, never a reconciled envelope with a null path.

[Timeouts & Locks]
  Pure in-memory arithmetic; no device access.

[Side-Effect Policy]
  Read-only except for the optional ``--output`` file written by ``main``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._diff_support import ProfilerUsageError, load_cli_json, snapshot, snapshot_call

from .diff_trace_replay import (  # re-exported surface
    _attribute_diff_snapshot as _attribute_diff,
)
from .diff_trace_replay import complement, intersect, normalize_ops, replay, total, union
from .diff_trace_replay import num as _num

SCHEMA = "mindie-diff-trace/2"
DEFAULT_CAPTURE_PATH = "capture.json"


def attribute_diff(predicted: Mapping[str, Any], captured: Mapping[str, Any], *,
                   padding: Mapping[str, Any] | None = None,
                   top_n: int = 5, capture_path: str = DEFAULT_CAPTURE_PATH,
                   launch_overhead_us: float | None = None,
                   cann_version: str | None = None) -> dict[str, Any]:
    """Attribution wrapper that optionally attaches validated padding charges."""
    if padding is not None and not isinstance(padding, Mapping):
        raise ValueError("padding must be an object")
    rows: list[tuple[str, Mapping[str, Any], int]] = []
    if isinstance(padding, Mapping):
        for name, spec in padding.items():
            if not isinstance(name, str) or not name:
                raise ValueError("padding operator names must be non-empty strings")
            if not isinstance(spec, Mapping):
                raise ValueError("padding operator specs must be objects")
            if "bucket_tokens" in spec:
                bucket = _strict_tokens(spec["bucket_tokens"])
                if bucket is None:
                    raise ValueError(
                        "bucket_tokens must be a non-negative integer")
            else:
                bucket = 0
            rows.append((name, spec, bucket))
    result = _attribute_diff(predicted, captured, top_n=top_n,
                             capture_path=capture_path,
                             launch_overhead_us=launch_overhead_us)
    if isinstance(padding, Mapping):
        charges = [_effective_operator_rows_snapshot(
            name, bucket, spec.get("logical_tokens"), spec.get("bound_args"),
            cann_version=cann_version)
            for name, spec, bucket in rows]
        invalid = [charge for charge in charges if not charge["valid"]]
        result["padding"] = {"charges": charges, "valid": not invalid,
                             "invalid_operators": [charge["op_name"]
                                                   for charge in invalid]}
        if invalid:
            result.setdefault("issues", []).append({
                "code": "padding_validation_failed",
                "operators": [charge["op_name"] for charge in invalid],
                "violations": [violation for charge in invalid
                               for violation in charge["violations"]]})
            result["status"] = "partial"
            result["reconciliation"] = {
                **(result.get("reconciliation") or {}), "reconciled": False}
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return result


def _semantics(op_name: str, cann_version: str | None
               ) -> tuple[dict[str, Any] | None, str, bool, bool, str | None]:
    try:
        from mindie_profiler.op_api_registry import catalog_operator
    except TimeoutError:
        raise
    except Exception:  # pragma: no cover - profiler package unavailable
        return None, "unknown_operator", False, False, None
    match = catalog_operator(op_name, cann_version=cann_version)
    if match is None:
        return None, "unknown_operator", False, False, None
    supported = bool(match.get("version_supported"))
    qualified = bool(match.get("version_qualified"))
    source = (f"catalog@{cann_version}" if supported else "catalog_unversioned")
    return (match["entry"], source, supported, qualified,
            match.get("qualification_state"))


def _strict_tokens(value: Any) -> int | None:
    """Finite, non-boolean, non-negative integral explicit token count."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() \
            and value >= 0 else None
    return None


def _filter_total(values: Sequence[float], encoding: str | None) -> int:
    body = list(values)
    if encoding == "cumsum" and body and body[0] == 0:
        body = body[1:]
    if encoding == "lengths":
        return max(0, int(round(sum(body))))
    return max(0, int(round(body[-1]))) if body else 0


def _logical_from_entry(entry: Mapping[str, Any], args: Mapping[str, Any],
                        bucket: int) -> tuple[int, str, list[dict[str, Any]]]:
    from mindie_profiler.op_abi_padding import (
        validate_group_list, validate_sequence_lengths,
    )
    mode = entry["padding_semantics"]["mode"]
    spec = entry["padding_semantics"]
    violations: list[dict[str, Any]] = []
    explicit = args.get("effective_tokens")
    if explicit is None:
        explicit = args.get("logical_tokens")
    checked_sequences = []
    if mode == "masked_varlen":
        for name in spec.get("filters") or ():
            values = args.get(name)
            if isinstance(values, (list, tuple)) and values:
                rule = (spec.get("layouts") or {}).get(str(args.get("layout") or "")) or {}
                encoding = rule.get("seq_encoding") or spec.get("sequence_mode")
                checked = validate_sequence_lengths(
                    list(values), encoding=str(encoding or "cumsum"),
                    extent=bucket, leading_zero=bool(rule.get("leading_zero")),
                    label=f"sequence_{name}")
                violations.extend(checked["violations"])
                checked_sequences.append(checked["total"])
    group_values: list[Any] = []
    group_routed: int | None = None
    group_source: str | None = None
    if mode == "grouped_prefix":
        name = spec.get("group_list_arg") or "group_list"
        values = args.get(name)
        if isinstance(values, (list, tuple)) and values:
            group_values = list(values)
            group_type = args.get(spec.get("group_list_type_arg") or "group_list_type", "0")
            checked = validate_group_list(group_values, group_type, bucket)
            violations.extend(checked["violations"])
            group_routed, group_source = checked["routed_tokens"], f"group:{name}"
        else:
            expert = args.get("expert_ids")
            if isinstance(expert, (list, tuple)):
                group_routed = len(expert)
                group_source = "expert_ids"
                numbers = []
                for value in expert:
                    try:
                        number = int(value)
                    except (TypeError, ValueError):
                        violations.append({"code": "expert_ids_invalid",
                                           "detail": repr(value)})
                        continue
                    if number < 0:
                        violations.append({"code": "expert_ids_negative",
                                           "detail": str(list(expert)[:8])})
                    numbers.append(number)
                if group_routed > bucket:
                    violations.append({"code": "routed_rows_exceed_bucket",
                                       "detail": f"{group_routed} > {bucket}"})
    if explicit is not None:
        number = _strict_tokens(explicit)
        if number is None:
            violations.append({"code": "effective_tokens_invalid",
                               "detail": repr(explicit)})
            return 0, "explicit", violations
        if number > bucket:
            violations.append({"code": "effective_tokens_exceeds_bucket",
                               "detail": f"{number} > {bucket}"})
            return 0, "explicit", violations
        return min(number, bucket), "explicit", violations
    if mode == "full_padded_tensor":
        return bucket, "full_bucket", violations
    if mode == "masked_varlen":
        if checked_sequences:
            return (min(bucket, max(0, int(checked_sequences[0]))),
                    f"filter:{spec.get('query_arg')}", violations)
        return bucket, "unfiltered", violations
    if group_routed is not None:
        return min(bucket, max(0, int(group_routed))), group_source, violations
    return bucket, "unfiltered", violations


def effective_operator_rows(
    op_name: str,
    bucket_tokens: int,
    logical_tokens: int | None,
    bound_args: Mapping[str, Any] | None = None,
    *,
    cann_version: str | None = None,
) -> dict[str, Any]:
    """Charge logical rows for masked/grouped modes and bucket rows otherwise."""
    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    bucket = _strict_tokens(bucket_tokens)
    if bucket is None:
        raise ValueError("bucket_tokens must be a non-negative integer")
    if bound_args is not None and not isinstance(bound_args, Mapping):
        raise ValueError("bound_args must be an object")
    args = dict(bound_args or {})
    if logical_tokens is not None:
        args["logical_tokens"] = logical_tokens
    entry, source, version_supported, version_qualified, qualification_state = (
        _semantics(op_name, cann_version))
    if entry is None:
        return {"op_name": op_name, "mode": "unknown", "bucket_tokens": bucket,
                "logical_tokens": bucket, "charged_tokens": bucket,
                "effective_tokens": bucket, "padded_tokens": 0,
                "bucket_padding_tax_pct": 0.0, "filters": [], "group_mode": None,
                "semantic_source": source, "version_supported": False,
                "version_qualified": False, "version_verified": False,
                "qualification_state": None,
                "valid": False,
                "violations": [{"code": "unknown_operator_semantics",
                                "detail": op_name}]}
    spec = entry["padding_semantics"]
    logical, logical_source, violations = _logical_from_entry(entry, args, bucket)
    charged = bucket if spec["mode"] == "full_padded_tensor" else min(logical, bucket)
    if violations:
        charged = 0
    padded = max(0, bucket - logical)
    paid = max(0, charged - logical)
    return {
        "op_name": op_name, "mode": spec["mode"], "bucket_tokens": bucket,
        "logical_tokens": logical, "charged_tokens": charged,
        "effective_tokens": charged, "padded_tokens": padded,
        "bucket_padding_tax_pct": round(100.0 * paid / logical, 4) if logical else 0.0,
        "filters": list(spec.get("filters") or ()),
        "group_mode": (spec.get("group_types") or {}).get(
            str(args.get(spec.get("group_list_type_arg") or "group_list_type", "0"))),
        "logical_source": logical_source,
        "semantic_source": source,
        "version_supported": version_supported,
        "version_qualified": version_qualified,
        "version_verified": version_qualified,
        "qualification_state": qualification_state,
        "valid": not violations, "violations": violations,
    }


def _rounded_cost_ms(value: float) -> float:
    """Round sub-unit cost digits only while float64 can still carry them."""
    if abs(value) >= 1e290:
        return float(value)
    return round(value, 9)


def pad_aware_operator_cost(
    op_name: str,
    duration_ms: float,
    bucket_tokens: int,
    logical_tokens: int | None,
    *,
    basis: str = "bucket",
    bound_args: Mapping[str, Any] | None = None,
    cann_version: str | None = None,
    setup_ms: float | None = None,
    per_row_ms: float | None = None,
) -> dict[str, Any]:
    """Decompose a bucket cost into setup + per-row work where evidence exists.

    The bounded ``charged / bucket`` fraction is formed before the cost is
    scaled, so a finite mathematical product is never lost to an intermediate
    multiplication overflow.  Attention pair work is not a row-scaled
    quantity; without an explicit ``setup_ms``/``per_row_ms`` decomposition the
    bucket cost is returned unchanged and the row-scaled number is exposed only
    as an estimate.  Results that no finite float64 can represent raise
    ``OverflowError`` instead of returning a nonstandard JSON infinity.
    """
    charge = _effective_operator_rows_snapshot(op_name, bucket_tokens, logical_tokens,
                                     bound_args, cann_version=cann_version)
    cost = max(0.0, _num(duration_ms))
    bucket = charge["bucket_tokens"]
    charged = charge["charged_tokens"]
    fraction = (charged / bucket) if bucket else 1.0
    estimate = cost * fraction
    if setup_ms is not None and per_row_ms is not None:
        adjusted = max(0.0, _num(setup_ms)) + max(0.0, _num(per_row_ms)) * charged
        model = "setup_plus_per_row"
    elif charge["mode"] == "grouped_prefix" and basis == "bucket" and bucket:
        adjusted, model = estimate, "routed_row_scaled"
    elif charge["mode"] == "full_padded_tensor" or basis == "logical" or not bucket:
        adjusted, model = cost, "bucket_rows"
    else:
        adjusted, model = cost, "bucket_basis_unresolved"
    if not (math.isfinite(adjusted) and math.isfinite(estimate)):
        raise OverflowError("padding cost exceeds the finite float64 range")
    result = {**charge, "basis": basis, "input_duration_ms": cost,
              "duration_ms": _rounded_cost_ms(adjusted),
              "row_scaled_estimate_ms": _rounded_cost_ms(estimate),
              "work_model": model}
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


MCP_SCHEMA = "mindie-diff-trace-mcp/1"
MCP_TOKEN_LIMIT = 250
_ZERO_BUCKETS = {"graph_ms": 0.0, "host_ms": 0.0, "comm_ms": 0.0,
                 "roofline_ms": 0.0, "residual_ms": 0.0, "total_delta_ms": 0.0}


def _diff_root() -> Path:
    """Same bounded ASCII short-root rules as the reproducer store."""
    try:
        from mindie_profiler.repro_storage import short_root
        return short_root() / "diff"
    except TimeoutError:
        raise
    except Exception:  # pragma: no cover - profiler package unavailable
        return Path("/tmp/mprof/diff")


def _persist_diff(payload: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    """Persist the full attribution under ``<root>/diff/<digest12>/diff.json``."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n"
    data = text.encode("utf-8")
    identifier = "sha256:" + hashlib.sha256(data).hexdigest()
    path = _diff_root() / identifier.split(":", 1)[-1][:12] / "diff.json"
    try:
        from mindie_profiler.repro_atomic_io import (
            path_lexists, read_regular_bytes, safe_write_bytes,
            validate_directory,
        )
        from mindie_profiler.repro_storage import protection_snapshot
        inodes, slots = protection_snapshot()
        if not validate_directory(path.parent, forbidden=slots):
            return identifier, None, "artifact_persistence_failed"
        if read_regular_bytes(path) != data:
            existed = path_lexists(path.parent)
            allowed = None if existed else path.parent
            if not safe_write_bytes(path, data, protected=inodes,
                                    forbidden=slots, allow_slot=allowed):
                return identifier, None, "artifact_persistence_failed"
    except TimeoutError:
        raise
    except OSError:
        return identifier, None, "artifact_persistence_failed"
    return identifier, str(path), None


def _issue_codes(issues: Any) -> list[str]:
    """Exact enum codes; the list may be shortened, a code is never sliced."""
    codes = []
    for issue in issues if isinstance(issues, Sequence) else ():
        if isinstance(issue, Mapping):
            codes.append(str(issue.get("code") or "unspecified"))
        else:
            codes.append(str(issue))
    return codes


def _diff_envelope_fits(envelope: Mapping[str, Any]) -> bool:
    from mindie_profiler.token_budget import _envelope_estimate
    return _envelope_estimate(dict(envelope), MCP_TOKEN_LIMIT)[0] <= MCP_TOKEN_LIMIT


def _shed_diff_issues(envelope: dict[str, Any]) -> dict[str, Any]:
    """Steps 2-3: compact padding, then drop whole issue codes."""
    if _diff_envelope_fits(envelope):
        return envelope
    padding = envelope.get("padding")
    if isinstance(padding, Mapping) and "charges" in padding:
        charges = padding.get("charges")
        count = envelope.get("padding_charges_count")
        if not isinstance(count, int):
            count = len(charges) if isinstance(charges, list) else 0
        compact = dict(envelope)
        compact["padding"] = {"valid": bool(padding.get("valid")),
                              "charges_count": count}
        if _diff_envelope_fits(compact):
            return compact
        envelope = compact
    dropped = dict(envelope)
    dropped.pop("padding", None)
    if _diff_envelope_fits(dropped):
        return dropped
    envelope = dropped
    issues = list(envelope.get("issues") or ())
    for keep in (4, 3, 2, 1):
        if keep >= len(issues):
            continue
        candidate = dict(envelope)
        candidate["issues"] = issues[:keep]
        candidate["issues_omitted"] = len(issues) - keep
        if _diff_envelope_fits(candidate):
            return candidate
    return envelope


def _compact_diff(payload: Mapping[str, Any], identifier: str,
                  artifact_path: str | None, padding: bool,
                  persistence_error: str | None = None) -> dict[str, Any]:
    from mindie_profiler.token_budget import bounded_envelope
    persisted = artifact_path is not None and persistence_error is None
    # The protocol envelope keeps exact delivery/reconciliation identity and
    # compact issue codes; verbose attribution rows stay in diff.json.
    envelope: dict[str, Any] = {
        "schema": MCP_SCHEMA, "status": payload.get("status"),
        "reconciled": bool((payload.get("reconciliation") or {}).get(
            "reconciled")),
        "issues": _issue_codes(payload.get("issues"))[:6],
        "artifact_id": identifier,
        "artifact_path": artifact_path if persisted else None,
        "retrievable": persisted, "persisted": persisted,
    }
    if persistence_error is not None:
        # The mathematics may still reconcile, but the required full artifact
        # was not delivered.  Never shape this as a successful retrieval.
        envelope["status"] = "delivery_failed"
        envelope["error"] = persistence_error
        codes = [code for code in envelope["issues"]
                 if code != persistence_error]
        # Keep the delivery code first; list trimming keeps it intact.
        envelope["issues"] = [persistence_error] + codes[:5]
    full_padding = payload.get("padding")
    if padding and isinstance(full_padding, Mapping):
        # Step 2: keep per-op padding evidence while it fits; the bounded
        # envelope replaces it with scalar counts before touching issues.
        charges = full_padding.get("charges")
        charge_rows = [charge for charge in (
            charges if isinstance(charges, Sequence) else ())
            if isinstance(charge, Mapping)]
        envelope["padding_valid"] = bool(full_padding.get("valid"))
        envelope["padding_charges_count"] = len(charge_rows)
        envelope["padding"] = {
            "valid": envelope["padding_valid"],
            "charges": [
                {"effective_tokens": charge.get("effective_tokens"),
                 "version_qualified": charge.get("version_qualified")}
                for charge in charge_rows[:2]],
        }
    envelope = _shed_diff_issues(envelope)
    return bounded_envelope(envelope, limit=MCP_TOKEN_LIMIT, list_limit=4)


def _error_envelope(code: str, reason: str | None = None) -> dict[str, Any]:
    from mindie_profiler.token_budget import bounded_envelope
    envelope = {
        "schema": MCP_SCHEMA, "status": "invalid_request", "reconciled": False,
        "issues": [code], "artifact_id": None, "artifact_path": None,
        "retrievable": False, "persisted": False,
    }
    if reason:
        envelope["reason"] = reason
    return bounded_envelope(envelope, limit=MCP_TOKEN_LIMIT, list_limit=4)


def resolve_diff_artifact(identifier: Any) -> dict[str, Any]:
    """Resolve and byte-verify a persisted full diff-trace artifact."""
    from mindie_profiler.token_budget import bounded_envelope
    text = str(identifier or "")
    digest = text[7:] if text.startswith("sha256:") else ""
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        return bounded_envelope({
            "schema": MCP_SCHEMA, "status": "invalid_request",
            "resolved": False, "retrievable": False, "persisted": False,
            "artifact_id": None, "error": "invalid_artifact_id"},
            limit=MCP_TOKEN_LIMIT, list_limit=4)
    path = _diff_root() / digest[:12] / "diff.json"
    try:
        from mindie_profiler.repro_atomic_io import (
            path_is_dir, read_regular_bytes, validate_directory,
        )
        from mindie_profiler.repro_storage import protection_snapshot
        _inodes, slots = protection_snapshot()
        if not validate_directory(path.parent, forbidden=slots):
            if path_is_dir(path.parent):
                return bounded_envelope({
                    "schema": MCP_SCHEMA, "status": "delivery_blocked",
                    "resolved": False, "retrievable": False,
                    "persisted": False, "artifact_id": text,
                    "error": "protected_artifact_target"},
                    limit=MCP_TOKEN_LIMIT, list_limit=4)
            return bounded_envelope({
                "schema": MCP_SCHEMA, "status": "not_found",
                "resolved": False, "retrievable": False, "persisted": False,
                "artifact_id": text, "error": "artifact_not_found"},
                limit=MCP_TOKEN_LIMIT, list_limit=4)
        data = read_regular_bytes(path)
        if data is None:
            raise OSError("not a regular unfollowed artifact")
    except TimeoutError:
        raise
    except OSError:
        return bounded_envelope({
            "schema": MCP_SCHEMA, "status": "not_found",
            "resolved": False, "retrievable": False, "persisted": False,
            "artifact_id": text, "error": "artifact_not_found"},
            limit=MCP_TOKEN_LIMIT, list_limit=4)
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != text:
        return bounded_envelope({
            "schema": MCP_SCHEMA, "status": "integrity_mismatch",
            "resolved": False, "retrievable": False, "persisted": False,
            "artifact_id": text, "error": "artifact_integrity_mismatch"},
            limit=MCP_TOKEN_LIMIT, list_limit=4)
    try:
        payload = json.loads(data)
    except ValueError:
        payload = None
    if not isinstance(payload, Mapping):
        return bounded_envelope({
            "schema": MCP_SCHEMA, "status": "integrity_mismatch",
            "resolved": False, "retrievable": False, "persisted": False,
            "artifact_id": text, "error": "artifact_integrity_mismatch"},
            limit=MCP_TOKEN_LIMIT, list_limit=4)
    return bounded_envelope({
        "schema": MCP_SCHEMA, "status": "resolved", "resolved": True,
        "retrievable": True, "persisted": True, "artifact_id": text,
        "artifact_path": str(path),
        "diff_schema": str(payload.get("schema") or "")},
        limit=MCP_TOKEN_LIMIT, list_limit=4)


MCP_TOOL_SPEC = {
    "name": "diff_trace",
    "description": "Attribute sim-vs-capture step time with R0->R4 replay.",
    "inputSchema": {
        "type": "object",
        "properties": {"predicted": {"type": "object"},
                       "captured": {"type": "object"},
                       "padding": {"type": "object"},
                       "cann_version": {"type": "string"},
                       "top_n": {"type": "integer"},
                       "capture_path": {"type": "string"},
                       "action": {"type": "string"},
                       "artifact_id": {"type": "string"},
                       "resolve_diff_artifact_id": {"type": "string"},
                       "resolve_artifact_id": {"type": "string"}},
        "required": [], "additionalProperties": False,
    },
}


def call_mcp_tool(arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a <=250-token envelope plus a retrievable full artifact."""
    try:
        args = snapshot({} if arguments is None else arguments)
        if not isinstance(args, Mapping):
            raise ProfilerUsageError("diff arguments must be an object")
        identifier = args.get("resolve_diff_artifact_id")
        if args.get("action") == "resolve_diff_artifact":
            identifier = (identifier if identifier is not None
                          else args.get("artifact_id")
                          if args.get("artifact_id") is not None
                          else args.get("resolve_artifact_id"))
        elif identifier is None and args.get("resolve_artifact_id") is not None:
            identifier = args.get("resolve_artifact_id")
        if identifier is not None:
            return resolve_diff_artifact(identifier)
        predicted, captured = args.get("predicted"), args.get("captured")
        if not isinstance(predicted, Mapping) or not isinstance(captured, Mapping):
            return _error_envelope("invalid_request")
        padding = args.get("padding")
        if padding is not None and not isinstance(padding, Mapping):
            raise ProfilerUsageError("padding must be an object")
        payload = _attribute_diff_snapshot(
            predicted, captured, padding=padding,
            top_n=int(args.get("top_n", 5)),
            cann_version=args.get("cann_version"),
            capture_path=str(args.get("capture_path") or DEFAULT_CAPTURE_PATH))
        artifact_id, artifact_path, persistence_error = _persist_diff(payload)
    except TimeoutError:
        raise
    except (ProfilerUsageError, TypeError, ValueError, KeyError, OverflowError):
        return _error_envelope("invalid_request", "invalid diff input")
    return _compact_diff(payload, artifact_id, artifact_path,
                         padding is not None, persistence_error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mindie-sim diff-trace", allow_abbrev=False)
    parser.add_argument("--predicted", required=True, help="sim step prediction JSON")
    parser.add_argument("--captured", required=True, help="profiler steady-state JSON")
    parser.add_argument("--padding", default=None, help="optional bucket map")
    parser.add_argument("--cann-version", default=None)
    parser.add_argument("--capture-path", default=DEFAULT_CAPTURE_PATH)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args(snapshot([] if argv is None else argv))
    try:
        predicted = load_cli_json(args.predicted)
        captured = load_cli_json(args.captured)
        padding = load_cli_json(args.padding) if args.padding else None
        result = attribute_diff(predicted, captured, padding=padding, top_n=args.top,
                                cann_version=args.cann_version,
                                capture_path=args.capture_path)
        text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n"
        data = text.encode("utf-8")
    except TimeoutError:
        raise
    except (ProfilerUsageError, TypeError, ValueError, KeyError, OverflowError, OSError):
        parser.exit(2, "diff-trace: invalid input\n")
    if args.output:
        Path(args.output).write_bytes(data)
    else:
        print(text, end="")
    return 0 if result["status"] == "reconciled" else 2


# Internal calls consume detached inputs; public aliases validate ingress.
_attribute_diff_snapshot = attribute_diff
attribute_diff = snapshot_call(attribute_diff, 'predicted', 'captured', 'padding')
_effective_operator_rows_snapshot = effective_operator_rows
effective_operator_rows = snapshot_call(effective_operator_rows, 'bound_args')
_pad_aware_operator_cost_snapshot = pad_aware_operator_cost
pad_aware_operator_cost = snapshot_call(pad_aware_operator_cost, 'bound_args')


__all__ = [
    "DEFAULT_CAPTURE_PATH", "MCP_SCHEMA", "MCP_TOKEN_LIMIT", "MCP_TOOL_SPEC",
    "SCHEMA", "attribute_diff", "call_mcp_tool", "complement",
    "effective_operator_rows", "intersect", "main", "normalize_ops",
    "pad_aware_operator_cost", "replay", "resolve_diff_artifact", "total",
    "union",
]


if __name__ == "__main__":
    raise SystemExit(main())
