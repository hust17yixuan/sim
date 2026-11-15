"""Shared primitives for profile ingestion and the counterfactual advisor.

[Role]
  Define canonical bottleneck classes, bounded heterogeneous-record lookup,
  finite-number coercion, and text classification used by advisor modules.

[Invariants]
  * Missing values stay ``None``; booleans are never numeric measurements.
  * Canonical bottleneck classes are exactly the four public C7-aligned names.
  * Recursive profile lookup is bounded by depth and node budget.

[Timeouts & Locks]
  Pure in-memory stdlib helpers; no I/O, sleeps, or locks.

[Side-Effect Policy]
  No mutation of inputs; every helper returns new records or scalars.
"""
from __future__ import annotations

from dataclasses import fields
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .schema.case import Software
from .schema.parallel import ParallelSpec

CANONICAL_BOTTLENECKS = ("comm_bound", "launch_host_bound", "memory_kv_bound", "compute_bound")
_PROFILE_MAX_BYTES = 50 * 1024**2
_CSV_MAX_ROWS = 2_000_000
_SLA_SOLVER_MAX_CONTEXT = 16_384
_COMM_TOKENS = ("comm", "link", "hccl", "allreduce", "allgather", "reduce_scatter", "alltoall",
                "all_to_all", "network")
_LAUNCH_TOKENS = ("launch", "host", "dispatch", "submission", "idle", "eager",
                  "cpu", "aicpu", "ai_cpu")
_MEMORY_TOKENS = ("memory", "membound", "kv", "hbm", "mte", "cache", "bandwidth", "dram",
                  "copy", "fill", "gather", "scatter")
_COMPUTE_TOKENS = ("compute", "cube", "vector", "aic", "aiv", "math", "matmul", "gemm",
                   "attention", "moe", "ffn", "expert", "softmax", "norm", "swiglu")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _positive(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and result >= 0.0 else None


def _pct(delta: float | None, base: float | None) -> float | None:
    if delta is None or base is None or base <= 0.0:
        return None
    return round(delta / base * 100.0, 6)


def _deep_values(mapping: Any, keys: Sequence[str], depth: int = 4, budget: int = 4096) -> list[Any]:
    """Bounded recursive lookup for heterogeneous profiler/capture records."""
    found: list[Any] = []
    skip_children = {"semantic_block_breakdown", "top_operators", "top_hotspot_kernels",
                     "top_operators_by_device_time"}
    stack: list[tuple[Any, int]] = [(mapping, 0)]
    while stack and budget > 0:
        node, level = stack.pop()
        budget -= 1
        if isinstance(node, Mapping):
            for key in keys:
                if key in node:
                    found.append(node[key])
            if level < depth:
                stack.extend((value, level + 1) for k, value in node.items()
                             if k not in skip_children and isinstance(value, (Mapping, list, tuple)))
        elif isinstance(node, (list, tuple)) and level < depth:
            stack.extend((value, level + 1) for value in node[:256]
                         if isinstance(value, (Mapping, list, tuple)))
    return found


def _first_number(mapping: Any, keys: Sequence[str]) -> float | None:
    for key in keys:
        for value in _deep_values(mapping, (key,)):
            result = _number(value)
            if result is not None:
                return result
    return None


def _first_text(mapping: Any, keys: Sequence[str]) -> str | None:
    for key in keys:
        for value in _deep_values(mapping, (key,)):
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _first_int(mapping: Any, keys: Sequence[str], minimum: int = 0) -> int | None:
    for key in keys:
        for value in _deep_values(mapping, (key,)):
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                try:
                    value = int(float(value.strip()))
                except ValueError:
                    continue
            if isinstance(value, int) and value >= minimum:
                return int(value)
    return None


def _sanitize_case(case: Any) -> Any:
    """Move top-level runtime observed/provenance blocks into runtime.metadata."""
    if not isinstance(case, Mapping) or not isinstance(case.get("runtime"), Mapping):
        return case
    runtime = dict(case["runtime"])
    moved = {key: runtime.pop(key) for key in ("observed_profile", "profiler_provenance")
             if key in runtime}
    if moved:
        meta = dict(runtime.get("metadata") or {})
        for key, value in moved.items():
            meta.setdefault(key, value)
        runtime["metadata"] = meta
        case = dict(case)
        case["runtime"] = runtime
    return case


def _has_measured_observation(observed: Any) -> bool:
    if not isinstance(observed, Mapping):
        return False
    numeric_keys = ("observed_step_ms", "step_time_ms", "observed_tpot_p50_ms", "tpot_p50_ms",
                    "observed_ttft_p50_ms", "ttft_p50_ms", "observed_operator_sum_ms",
                    "device_activity_sum_ms", "device_busy_union_ms")
    if any(_number(observed.get(key)) is not None for key in numeric_keys):
        return True
    shares = observed.get("observed_share_by_bound") or observed.get("bottleneck_shares")
    if isinstance(shares, Mapping) and any(_number(value) is not None for value in shares.values()):
        return True
    operators = observed.get("top_operators")
    return isinstance(operators, list) and bool(operators)


def _has_observed_measurement(observed: Any) -> bool:
    if not isinstance(observed, Mapping):
        return False
    numeric_keys = ("observed_step_ms", "step_time_ms", "observed_tpot_p50_ms", "tpot_p50_ms",
                    "observed_ttft_p50_ms", "ttft_p50_ms", "observed_operator_sum_ms",
                    "device_activity_sum_ms", "device_busy_union_ms")
    if any(_number(observed.get(key)) is not None for key in numeric_keys):
        return True
    shares = observed.get("observed_share_by_bound") or observed.get("bottleneck_shares")
    if isinstance(shares, Mapping) and any(_number(value) is not None for value in shares.values()):
        return True
    if isinstance(observed.get("observed_dominant_bottleneck")
                  or observed.get("dominant_bottleneck"), str):
        return True
    operators = observed.get("top_operators")
    return isinstance(operators, list) and bool(operators)


def _candidate_mappings(data: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def visit(node: Any, depth: int) -> None:
        if not isinstance(node, Mapping) or id(node) in seen or depth > 3:
            return
        seen.add(id(node))
        result.append(node)
        for key in ("run_meta", "benchmark_summary", "summary", "scene", "capture", "provenance",
                    "metrics", "workload", "parallelism", "parallel", "architecture_override",
                    "design", "config", "case", "attribution", "accounting", "step_accounting",
                    "window_accounting", "summary_document", "attribution_document", "runtime",
                    "metadata", "observed_profile"):
            visit(node.get(key), depth + 1)

    visit(data, 0)
    return result


def _canonical_bottleneck(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("dominant_bottleneck", "dominant_bound", "bottleneck", "bound", "primary_bound"):
            if key in value:
                result = _canonical_bottleneck(value[key])
                if result:
                    return result
        return None
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if raw in CANONICAL_BOTTLENECKS:
        return raw
    if any(token in raw for token in _COMM_TOKENS):
        return "comm_bound"
    if any(token in raw for token in _LAUNCH_TOKENS):
        return "launch_host_bound"
    if any(token in raw for token in _MEMORY_TOKENS):
        return "memory_kv_bound"
    if any(token in raw for token in _COMPUTE_TOKENS):
        return "compute_bound"
    return None


def _classify_name(name: str) -> str:
    lowered = str(name).lower()
    if any(token in lowered for token in _COMM_TOKENS):
        return "comm_bound"
    if any(token in lowered for token in _LAUNCH_TOKENS):
        return "launch_host_bound"
    if any(token in lowered for token in _MEMORY_TOKENS):
        return "memory_kv_bound"
    return "compute_bound"


def _normalize_shares(shares: Mapping[str, Any]) -> dict[str, float]:
    totals = {name: 0.0 for name in CANONICAL_BOTTLENECKS}
    for key, raw in shares.items():
        value = _positive(raw)
        canonical = _canonical_bottleneck(key) if value is not None else None
        if canonical:
            totals[canonical] += value
    total = sum(totals.values())
    if total <= 0.0:
        return {}
    scale = total if total > 1.000001 else 1.0
    return {name: round(value / scale, 9) for name, value in totals.items() if value > 0.0}


def memory_stage_facts(case, *, phase: str, prefix_tokens: int, tokens: int,
                       context: int, sequences: int, output_tokens: int):
    """Pair per-stage supply with the executing and complete-service windows.

    The executing window, delayed-release hold, workspace, peak and both
    headroom values come from the same ``capacity_window`` projection used by
    the public SLA solver and ``capacity_feedback``.  Active-window headroom is
    retained under its explicit name; guardrails consume the complete peak.
    """
    from .prediction.capacity import Memory
    from .prediction.efficiency import _resolve_speculative_tokens, capacity_window
    memory = Memory(case, workspace_tokens=max(1, sequences * tokens))
    window_context = prefix_tokens if phase == "prefill" else context
    if phase == "decode":
        _, emitted = _resolve_speculative_tokens(
            case, output_budget=output_tokens, tokens_per_seq=tokens,
            emitted_tokens_per_seq=None)
    else:
        _, emitted = _resolve_speculative_tokens(
            case, output_budget=output_tokens, tokens_per_seq=None,
            emitted_tokens_per_seq=None)
    current = capacity_window(
        case, phase=phase, context=window_context, tokens_per_seq=tokens,
        sequences=sequences, output_budget=output_tokens,
        emitted_tokens_per_seq=emitted, memory=memory)
    if phase == "prefill":
        decode_q, decode_emit = _resolve_speculative_tokens(
            case, output_budget=output_tokens, tokens_per_seq=None,
            emitted_tokens_per_seq=None)
        life_memory = Memory(case, workspace_tokens=max(1, sequences * decode_q))
        lifecycle = capacity_window(
            case, phase="decode", context=prefix_tokens + tokens,
            tokens_per_seq=decode_q, sequences=sequences,
            output_budget=output_tokens, emitted_tokens_per_seq=decode_emit,
            memory=life_memory)
    else:
        lifecycle = current

    def _legacy_stages(window):
        count = max(1, int(window.get("sequences") or 1))
        rows = []
        for row in window.get("stages", ()):
            active = float(row.get("active_kv_bytes", 0.0) or 0.0)
            external = float(row.get("external_held_bytes", 0.0) or 0.0)
            rows.append({
                "stage": int(row["stage"]),
                "available_bytes": round(float(row.get("supply_bytes", 0.0) or 0.0), 3),
                "kv_bytes_per_sequence": round(active / count, 3),
                "headroom_bytes": round(float(row.get("active_headroom_bytes", 0.0) or 0.0), 3),
                "used_bytes": round(float(row.get("fixed_bytes", 0.0) or 0.0) + external + active, 3),
                "hbm_bytes_per_die": round(float(row.get("hbm_bytes_per_die", 0.0) or 0.0), 3),
                "hbm_utilization_ratio": round(float(row.get("hbm_utilization_ratio", 0.0) or 0.0), 9),
                "workspace_bytes": round(float(row.get("workspace_bytes", 0.0) or 0.0), 3),
                "external_held_bytes": round(external, 3),
                "active_kv_bytes": round(active, 3),
                "retiring_kv_bytes": round(float(row.get("retiring_kv_bytes", 0.0) or 0.0), 3),
                "peak_bytes": round(float(row.get("peak_bytes", 0.0) or 0.0), 3),
                "full_headroom_bytes": round(float(row.get("headroom_bytes", 0.0) or 0.0), 3),
            })
        return rows

    stages = _legacy_stages(current)
    lifecycle_stages = _legacy_stages(lifecycle)
    hbm_by_stage = [float(row["hbm_bytes_per_die"]) for row in stages]
    hbm = min(hbm_by_stage, default=min(float(die.hbm_bytes) for die in case.machine.dies))
    return {
        "memory": memory, "hbm_bytes": hbm, "hbm_bytes_by_stage": hbm_by_stage,
        "kv_tokens": int(current["window"]["active_tokens_per_sequence"]),
        "available_by_stage": [float(row["available_bytes"]) for row in stages],
        "kv_by_stage": [float(row["kv_bytes_per_sequence"]) for row in stages],
        "headroom_by_stage": [float(row["headroom_bytes"]) for row in stages],
        "used_by_stage": [float(row["used_bytes"]) for row in stages],
        "min_available_bytes": min((float(row["available_bytes"]) for row in stages), default=0.0),
        "max_kv_bytes_per_sequence": max((float(row["kv_bytes_per_sequence"]) for row in stages), default=0.0),
        "min_headroom_bytes": min((float(row["headroom_bytes"]) for row in stages), default=0.0),
        "max_used_bytes": max((float(row["used_bytes"]) for row in stages), default=0.0),
        "max_workspace_bytes": float(current.get("workspace_bytes", 0.0) or 0.0),
        "lifecycle_kv_bytes_per_sequence": max((float(row["kv_bytes_per_sequence"]) for row in lifecycle_stages), default=0.0),
        "lifecycle_min_headroom_bytes": float(lifecycle.get("headroom_bytes", 0.0) or 0.0),
        "joint_min_headroom_bytes": min(
            float(current.get("active_headroom_bytes", 0.0) or 0.0),
            float(lifecycle.get("headroom_bytes", 0.0) or 0.0)),
        "headroom_basis": "active_window_kv_only; joint is complete_peak_after_workspace_and_fixed_reserve",
        "capacity_window": current,
        "lifecycle_capacity_window": lifecycle,
        "lifecycle_stages": lifecycle_stages if phase == "prefill" else None,
        "stages": stages,
    }


def capacity_metrics(case, cfg, b_sync: int, b_async: int,
                     b_lifecycle_sync: int | None = None,
                     b_lifecycle_async: int | None = None,
                     b_joint_sync: int | None = None,
                     b_joint_async: int | None = None):
    facts = memory_stage_facts(case, phase=cfg.phase, prefix_tokens=cfg.prefix_tokens,
                               tokens=cfg.tokens, context=cfg.context,
                               sequences=cfg.sequences, output_tokens=cfg.output_tokens)
    headroom = float(facts["min_headroom_bytes"])
    b_eff = b_async if case.runtime.async_scheduling else b_sync
    b_life = (b_lifecycle_async if case.runtime.async_scheduling else b_lifecycle_sync)
    b_joint = (b_joint_async if case.runtime.async_scheduling else b_joint_sync)
    return {
        "vram_max_sequences_per_dp_sync": int(b_sync),
        "vram_max_sequences_per_dp_async": int(b_async),
        "vram_max_sequences_per_dp_lifecycle_sync": int(b_lifecycle_sync) if b_lifecycle_sync is not None else None,
        "vram_max_sequences_per_dp_lifecycle_async": int(b_lifecycle_async) if b_lifecycle_async is not None else None,
        "vram_max_sequences_per_dp_joint_sync": int(b_joint_sync) if b_joint_sync is not None else None,
        "vram_max_sequences_per_dp_joint_async": int(b_joint_async) if b_joint_async is not None else None,
        "weight_bytes_per_die": round(float(max(facts["memory"].weights, default=0.0)), 3),
        "kv_bytes_per_seq": round(float(facts["max_kv_bytes_per_sequence"]), 3),
        "workspace_bytes": round(float(facts["max_workspace_bytes"]), 3),
        "available_kv_capacity_bytes_per_die": round(float(facts["min_available_bytes"]), 3),
        "hbm_utilization_ratio": round(max((stage["hbm_utilization_ratio"] for stage in facts["stages"]), default=0.0), 9),
        "capacity_headroom_bytes": round(headroom, 3),
        "capacity_headroom_basis": facts["headroom_basis"],
        "capacity_by_stage": facts["stages"],
        "capacity_by_stage_lifecycle": facts["lifecycle_stages"] or facts["stages"],
        "lifecycle_capacity_headroom_bytes": round(float(facts["lifecycle_min_headroom_bytes"]), 3),
        "joint_capacity_headroom_bytes": round(float(facts["joint_min_headroom_bytes"]), 3),
        "capacity_window": facts["capacity_window"],
        "lifecycle_capacity_window": facts["lifecycle_capacity_window"],
        "is_vram_feasible": bool(b_eff >= cfg.sequences and headroom >= -1e-6),
        "is_lifecycle_feasible": bool(b_life is not None and b_life >= cfg.sequences
                                      and float(facts["lifecycle_min_headroom_bytes"]) >= -1e-6),
        "is_joint_feasible": bool(b_joint is not None and b_joint >= cfg.sequences
                                  and float(facts["joint_min_headroom_bytes"]) >= -1e-6),
    }


__all__ = [name for name in globals() if not name.startswith("__")]
