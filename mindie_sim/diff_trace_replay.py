"""Device-wide interval unions and ordered R0->R4 counterfactual replay.

[Role]
  Turn one predicted and one captured step into R0..R4 using per-``(rank,
  stream)`` FIFO causal replay, captured DAG edges, symmetric start floors and
  critical-path interval arithmetic; ``diff_trace`` only formats this engine.

[Invariants]
  * Every stage is a schedule, not ``captured_total + extra idle``.  An identical
    trace replays to the same makespan at R0..R4 with zero buckets/residual.
  * FIFO is keyed by ``(rank, stream)``, so equal stream ids on different ranks
    never serialize against each other.
  * A matched follower may start earlier than predicted when its dependency ends
    earlier; a predicted start is an earlier-of floor, never a one-way floor.
    An explicit captured host gap may pin the follower's captured start.
  * Idle requires gap evidence; otherwise it is an unexplained residual that
    blocks ``reconciled`` and is never promoted to a host bubble.
  * ``deps`` may be indices/names/self/out-of-range; cycles are reported, never
    raised.  Topology changes produce ``added``/``removed`` buckets.
  * Clock derating is reported as ``clock_ratio_below_nominal``; thermal state is
    never inferred without temperature telemetry.
  * Every delta is signed; the four buckets plus residual sum to ``O - R0``.

[Timeouts & Locks]
  Pure in-memory interval arithmetic proportional to ops + gaps.

[Side-Effect Policy]
  No I/O, device access or simulation imports.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from ._diff_support import snapshot_call

from .diff_trace_release import (
    captured_total_explicit, complement, gap_evidence, intersect, interval,
    launch_cost as _launch_cost, normalize_gaps, op_uid, release_delays,
    tail_gap, terminal_gap_evidence, total, union, window_of,
)
from .diff_trace_release import (
    _union_snapshot,
    _total_snapshot,
    _complement_snapshot,
    _intersect_snapshot,
    _normalize_gaps_snapshot,
    _window_of_snapshot,
    _captured_total_explicit_snapshot,
    _terminal_gap_evidence_snapshot,
    _launch_cost_snapshot,
    _release_delays_snapshot,
    _tail_gap_snapshot,
    _gap_evidence_snapshot,
    _op_uid_snapshot,
)

EPS = 1e-9
COLLECTIVE_TOKENS = ("hccl", "allreduce", "allgather", "reducescatter", "alltoall",
                     "dispatch", "combine", "sendrecv", "moedistribute")
COVERAGE_THRESHOLD = 0.9
_RAW_FIELDS = ("ops", "kernels", "occurrences", "operator_rows", "operators", "rows")


def num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def key(name: Any) -> str:
    text = str(name or "").strip().lower().replace("-", "_")
    for prefix in ("torch_npu.", "torch.", "aclnn"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("_kernel", "kernel"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    return text


def op_name(op: Mapping[str, Any]) -> str:
    return str(op.get("name") or op.get("op_type") or op.get("kernel") or "")


def duration(op: Mapping[str, Any]) -> float:
    for field, scale in (("duration_ms", 1.0), ("ms", 1.0), ("end_ms", 0.0),
                         ("duration_us", 1e-3), ("duration_ns", 1e-6)):
        if field in op and op[field] is not None:
            if field == "end_ms":
                start = op.get("start_ms", op.get("start_us", 0))
                start = num(start) / 1000.0 if "start_us" in op else num(start)
                return max(0.0, num(op[field]) - start)
            return max(0.0, num(op[field]) * scale)
    if "start_ms" in op and "end_ms" in op:
        return max(0.0, num(op["end_ms"]) - num(op["start_ms"]))
    return 0.0


def start(op: Mapping[str, Any], fallback: float = 0.0) -> float:
    if "start_ms" in op:
        return num(op["start_ms"], fallback)
    if "start_us" in op:
        return num(op["start_us"], fallback * 1000.0) / 1000.0
    return fallback


def is_collective(op: Mapping[str, Any]) -> bool:
    kind = str(op.get("kind") or "").lower()
    if kind in {"collective", "comm", "hccl", "dispatch", "combine"}:
        return True
    if str(op.get("pipe") or "") == "hccl_rdma_sio":
        return True
    return any(token in key(_op_name_snapshot(op)) for token in COLLECTIVE_TOKENS)


def is_host(op: Mapping[str, Any]) -> bool:
    return str(op.get("kind") or "").lower() in {"host", "launch", "dispatch"}


def is_fallback(op: Mapping[str, Any]) -> bool:
    return bool(op.get("fallback") or op.get("unexpected") or op.get("lowering_fallback"))


def _deps_of(op: Mapping[str, Any]) -> list[Any]:
    raw = op.get("deps") or op.get("dependencies") or ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return [raw] if raw not in (None, ()) else []
    return list(raw)


def normalize_ops(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = None
    for field in _RAW_FIELDS:
        rows = packet.get(field)
        if rows is not None:
            if not isinstance(rows, list):
                raise ValueError("diff row wrappers must contain arrays")
            if raw is None:
                raw = rows
    ops: list[dict[str, Any]] = []
    run_id = str(packet.get("run_id") or packet.get("capture_id") or "run")
    occurrence: dict[tuple[int, int, int, str], int] = {}
    for index, item in enumerate(raw or ()):
        if not isinstance(item, Mapping) or not _op_name_snapshot(item):
            continue
        stream = int(num(item.get("stream", item.get("stream_id", 0))))
        step = int(num(item.get("step", item.get("step_id", 0))))
        rank = int(num(item.get("rank", item.get("device_id", 0))))
        name = _op_name_snapshot(item)
        occurrence_key = (step, rank, stream, key(name))
        ordinal = occurrence.get(occurrence_key, 0)
        occurrence[occurrence_key] = ordinal + 1
        abi = str(item.get("abi_sha256") or item.get("abi_digest") or item.get("abi_id") or "")
        ops.append({
            "index": len(ops), "raw_op_index": index, "name": name, "key": key(name),
            "stream": stream, "step": step, "rank": rank, "run_id": run_id,
            "start": _start_snapshot(item), "duration": _duration_snapshot(item),
            "kind": item.get("kind"), "pipe": item.get("pipe"),
            "deps": _deps_of(item),
            "collective": _is_collective_snapshot(item), "host": _is_host_snapshot(item),
            "fallback": _is_fallback_snapshot(item),
            "abi_digest": abi, "stream_occurrence_index": ordinal,
            "op_uid": str(item.get("op_uid") or ""),
            "raw": dict(item),
        })
    return ops


def _resolve_deps(ops: Sequence[Mapping[str, Any]], op: Mapping[str, Any],
                  position: int, by_name: Mapping[str, list[int]]) -> list[int]:
    """Resolve index/name dependencies, dropping self and out-of-range edges."""
    deps: list[int] = []
    for dep in op.get("deps") or ():
        if isinstance(dep, bool):
            continue
        if isinstance(dep, int):
            if 0 <= dep < len(ops) and dep != position:
                deps.append(dep)
        elif isinstance(dep, str) and dep in by_name:
            candidates = [index for index in by_name[dep] if index < position]
            if candidates:
                deps.append(candidates[-1])
    return list(dict.fromkeys(deps))


def _order(ops: Sequence[Mapping[str, Any]], floors: Sequence[float],
           by_name: Mapping[str, list[int]], *, fifo: bool = True
           ) -> tuple[list[int], bool]:
    """Topological order with per-(rank, stream) FIFO edges; cycles fall back."""
    deps = {index: set(_resolve_deps(ops, op, index, by_name))
            for index, op in enumerate(ops)}
    if fifo:
        streams: dict[tuple[int, int], list[int]] = {}
        for index, op in enumerate(ops):
            streams.setdefault((int(op.get("rank", 0)), int(op["stream"])), []).append(index)
        for indices in streams.values():
            # Stream FIFO follows enqueue (raw list) order, never timestamp order.
            ordered = sorted(indices)
            for left, right in zip(ordered, ordered[1:]):
                deps[right].add(left)
    pending = {index: set(values) for index, values in deps.items()}
    ready = [index for index, values in pending.items() if not values]
    order: list[int] = []
    while ready:
        index = min(ready, key=lambda item: (floors[item], item))
        ready.remove(index)
        order.append(index)
        for other, values in pending.items():
            if index in values:
                values.discard(index)
                if not values and other not in order and other not in ready:
                    ready.append(other)
    cyclic = len(order) != len(ops)
    if cyclic:
        remaining = [index for index in range(len(ops)) if index not in order]
        order.extend(sorted(remaining, key=lambda item: (floors[item], item)))
    return order, cyclic


def replay(ops: Sequence[Mapping[str, Any]], durations: Sequence[float],
           *, window: tuple[float, float] | None = None,
           starts: Sequence[float] | None = None,
           release_delays: Sequence[float] | None = None,
           stream_fifo: bool = True) -> dict[str, Any]:
    """Replay ops with explicit start floors, DAG edges and stream FIFO."""
    if not ops:
        return {"wall_ms": 0.0, "busy_ms": 0.0, "idle_ms": 0.0,
                "intervals": [], "has_deps": False, "starts": [], "ends": [],
                "order": [], "cycle": False}
    floors = [num(item) for item in (starts if starts is not None
                                     else [op["start"] for op in ops])]
    by_name: dict[str, list[int]] = {}
    for index, op in enumerate(ops):
        by_name.setdefault(op["key"], []).append(index)
    order, cycle = _order(ops, floors, by_name, fifo=stream_fifo)
    ends: dict[int, float] = {}
    begins: dict[int, float] = {}
    cursor: dict[tuple[int, int], float] = {}
    intervals: list[tuple[float, float] | None] = []
    has_deps = any(op.get("deps") for op in ops)
    for index in order:
        op = ops[index]
        stream_key = (int(op.get("rank", 0)), int(op["stream"]))
        begin = floors[index]
        for dep in _resolve_deps(ops, op, index, by_name):
            begin = max(begin, ends.get(dep, begin))
        if stream_fifo:
            begin = max(begin, cursor.get(stream_key, 0.0))
        if release_delays is not None:
            begin += max(0.0, num(release_delays[index]))
        end = begin + max(0.0, num(durations[index]))
        begins[index], ends[index] = begin, end
        cursor[stream_key] = max(cursor.get(stream_key, 0.0), end)
        intervals.append((begin, end))
    unioned = _union_snapshot(intervals, window=window)
    low, high = min(begins.values()), max(ends.values())
    wall = max(0.0, high - low)
    busy = _total_snapshot(unioned)
    return {"wall_ms": wall, "busy_ms": busy,
            "idle_ms": max(0.0, wall - busy), "intervals": unioned,
            "has_deps": has_deps, "cycle": cycle,
            "starts": [begins[index] for index in range(len(ops))],
            "ends": [ends[index] for index in range(len(ops))],
            "order": order}


def _abi_digest(op: Mapping[str, Any]) -> str:
    return str(op.get("abi_digest") or "")


def match_key(op: Mapping[str, Any]) -> tuple[Any, ...]:
    return (op["step"], op["rank"], op["stream"], op["key"], _abi_digest(op),
            op["stream_occurrence_index"])


def _match(captured: Sequence[Mapping[str, Any]],
           predicted: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]],
                                                             list[Mapping[str, Any]],
                                                             list[Mapping[str, Any]]]:
    queue: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for op in predicted:
        queue.setdefault(_match_key_snapshot(op), []).append(op)
    matched, unmatched = [], []
    for op in captured:
        candidates = queue.get(_match_key_snapshot(op))
        if candidates:
            matched.append({"captured": op, "predicted": candidates.pop(0)})
        else:
            unmatched.append(op)
    used = {id(item["predicted"]) for item in matched}
    removed = [op for op in predicted if id(op) not in used]
    return matched, unmatched, removed


def _durations(captured: Sequence[Mapping[str, Any]],
               matched: Sequence[Mapping[str, Any]],
               phase: str) -> list[float]:
    """Duration map for one counterfactual phase."""
    predicted_for = {id(item["captured"]): item["predicted"] for item in matched}
    out = []
    for op in captured:
        source = predicted_for.get(id(op))
        if source is None:
            if phase == "structure" and op["collective"]:
                out.append(0.0)
            else:
                out.append(op["duration"])
        elif op["fallback"] or phase == "captured":
            out.append(op["duration"])
        elif phase == "collective" and op["collective"]:
            out.append(op["duration"])
        else:
            out.append(source["duration"])
    return out


def _release_delays(captured, gaps, origin=0.0):
    """Resolve deps once, then derive release delays from earliest readiness."""
    by_name: dict[str, list[int]] = {}
    for index, op in enumerate(captured):
        by_name.setdefault(op["key"], []).append(index)
    deps = [_resolve_deps(captured, op, index, by_name)
            for index, op in enumerate(captured)]
    return _release_delays_snapshot(captured, gaps, origin, deps)


def _floors(captured, matched, releases=()):
    """Earlier-of start floor with the host release component removed."""
    predicted_for = {id(item["captured"]): item["predicted"] for item in matched}
    floors = []
    for index, op in enumerate(captured):
        source = predicted_for.get(id(op))
        delay = releases[index] if index < len(releases) else 0.0
        floor = op["start"] - max(0.0, num(delay))
        floors.append(min(num(source["start"]), floor) if source else floor)
    return floors


_gap_evidence = _gap_evidence_snapshot
_op_uid = _op_uid_snapshot


def _compute_diff(predicted: Mapping[str, Any], captured: Mapping[str, Any], *,
                   top_n: int = 5, capture_path: str = "capture.json",
                   launch_overhead_us: float | None = None) -> dict[str, Any]:
    """Attribute ``captured - predicted`` with ordered R0 -> R4 replay."""
    if not isinstance(predicted, Mapping) or not isinstance(captured, Mapping):
        raise ValueError("diff-trace requires predicted and captured objects")
    pred_ops, cap_ops = _normalize_ops_snapshot(predicted), _normalize_ops_snapshot(captured)
    window_pred, window_cap = _window_of_snapshot(predicted, pred_ops), _window_of_snapshot(captured, cap_ops)
    if window_pred is None or window_cap is None:
        raise ValueError("diff-trace needs total_ms/window_ms or timed ops on both sides")
    predicted_total = num(predicted.get("total_ms", predicted.get("step_ms")),
                          window_pred[1] - window_pred[0])
    captured_total = num(captured.get("total_ms", captured.get("step_ms")),
                         window_cap[1] - window_cap[0])
    issues: list[dict[str, Any]] = []

    matched, added, removed = _match(cap_ops, pred_ops)
    gaps_cap, gaps_pred = _normalize_gaps_snapshot(captured), _normalize_gaps_snapshot(predicted)
    cap_end = max((op["start"] + op["duration"] for op in cap_ops),
                  default=window_cap[0])
    pred_end = max((op["start"] + op["duration"] for op in pred_ops),
                   default=window_pred[0])
    pred_tail = _tail_gap_snapshot(gaps_pred, window_pred, pred_end)
    explicit_cap_tail = _tail_gap_snapshot(gaps_cap, window_cap, cap_end, implicit=False)
    # Replace, never add.  A predicted implicit tail is the baseline; explicit
    # captured terminal coverage replaces it in EITHER direction.  Without such
    # evidence the predicted tail stays and any captured excess is residual.
    terminal_evidence, explicit_terminal_ms = _terminal_gap_evidence_snapshot(
        captured, window_cap, cap_end)
    cap_tail = explicit_terminal_ms if terminal_evidence else pred_tail
    total_explicit = _captured_total_explicit_snapshot(captured)
    release_cap = _release_delays(cap_ops, gaps_cap, window_cap[0])
    release_pred = _release_delays(pred_ops, gaps_pred, window_pred[0])
    floors_cap = _floors(cap_ops, matched, release_cap)
    floors_pred = [op["start"] for op in pred_ops]
    pred_base_floor = [max(0.0, floors_pred[index] - release_pred[index])
                       for index in range(len(pred_ops))]
    # R1 keeps the predicted graph but replaces each matched occurrence's host
    # release with the captured release for THAT captured index.  The captured
    # release is never added to a start floor that already contains it, and an
    # unmatched predicted occurrence keeps its own floor.
    r1_floors = list(floors_pred)
    for item in matched:
        cap_index = item["captured"]["index"]
        pred_index = item["predicted"]["index"]
        r1_floors[pred_index] = (pred_base_floor[pred_index]
                                 + release_cap[cap_index])
    pred_replay = _replay_snapshot(pred_ops, [op["duration"] for op in pred_ops],
                         window=window_pred, starts=floors_pred)
    predicted_total = max(predicted_total, pred_replay["wall_ms"])
    r0 = predicted_total
    # R0 already includes the predicted tail; R1..R4 REPLACE it with the
    # captured tail instead of adding both.  Any leading window offset is
    # carried once so a trace with a nonzero origin also replays identically.
    baseline_head = max(0.0, predicted_total - pred_tail - pred_replay["wall_ms"])
    base_extra = baseline_head + cap_tail

    structure_durations = _durations(cap_ops, matched, "structure")
    # Gaps already represented by a release delay are not counted again, and
    # terminal evidence is priced once by ``cap_tail`` rather than as a host gap.
    host_window = (window_cap[0], max(window_cap[0],
                                     min(window_cap[1], cap_end)))
    host_evidence, _host_idle, _host_gaps = _gap_evidence(
        gaps_cap, host_window,
        _replay_snapshot(cap_ops, structure_durations, window=window_cap,
               starts=_floors(cap_ops, matched)).get("starts", ()),
        structure_durations,
        exclude_ends={round(cap_ops[index]["start"], 9)
                      for index in range(len(cap_ops)) if release_cap[index] > 0})
    overhead, modeled_launch, launch_count = _launch_cost_snapshot(captured, launch_overhead_us)
    pred_overhead, pred_modeled_launch, _ = _launch_cost_snapshot(predicted, launch_overhead_us)
    host_extra = max(0.0, host_evidence - modeled_launch)
    host_replay = _replay_snapshot(pred_ops, [op["duration"] for op in pred_ops],
                         window=window_pred, starts=r1_floors)
    r1 = host_replay["wall_ms"] + base_extra + host_extra
    d_host = r1 - r0
    host_bubble = max(0.0, d_host - modeled_launch)
    structure = _replay_snapshot(cap_ops, structure_durations, window=window_cap,
                       starts=floors_cap, release_delays=release_cap)
    r2 = structure["wall_ms"] + base_extra + host_extra
    d_graph = r2 - r1

    evidence_cap, idle_cap, gap_count = _gap_evidence(
        gaps_cap, window_cap,
        [op["start"] for op in cap_ops],
        [op["duration"] for op in cap_ops])
    evidence_pred, idle_pred, pred_gap_count = _gap_evidence(
        gaps_pred, window_pred, floors_pred,
        [op["duration"] for op in pred_ops])
    collective = _replay_snapshot(cap_ops, _durations(cap_ops, matched, "collective"),
                        window=window_cap, starts=floors_cap,
                        release_delays=release_cap)
    r3 = collective["wall_ms"] + base_extra + host_extra
    d_comm = r3 - r2

    captured_phase = _replay_snapshot(cap_ops, _durations(cap_ops, matched, "captured"),
                            window=window_cap, starts=floors_cap,
                            release_delays=release_cap)
    roofline = captured_phase["wall_ms"] - collective["wall_ms"]
    r4 = captured_phase["wall_ms"] + base_extra + host_extra
    d_op = r4 - r3
    observed = captured_total
    residual = observed - r4

    buckets = {"graph_lowering_delta_ms": d_graph,
               "host_dispatch_bubble_delta_ms": d_host,
               "collective_and_skew_delta_ms": d_comm,
               "kernel_roofline_and_clock_delta_ms": roofline,
               "unexplained_residual_delta_ms": residual}
    bucket_sum = math.fsum(buckets.values())
    total_delta = observed - r0

    matched_pred_ms = math.fsum(item["predicted"]["duration"] for item in matched)
    predicted_ms = math.fsum(op["duration"] for op in pred_ops)
    coverage_ratio = (matched_pred_ms / predicted_ms) if predicted_ms > 0 else (
        1.0 if not pred_ops and not cap_ops else 0.0)
    explicit_alignment = captured.get("alignment") if isinstance(
        captured.get("alignment"), Mapping) else {}
    if not total_explicit:
        issues.append({"code": "captured_total_unreported"})
    if not cap_ops:
        issues.append({"code": "no_captured_op_evidence"})
    if not pred_ops:
        issues.append({"code": "no_predicted_op_evidence"})
    if pred_ops and coverage_ratio < COVERAGE_THRESHOLD:
        issues.append({"code": "alignment_coverage_incomplete",
                       "coverage_ratio": round(coverage_ratio, 6)})
    if any(op["duration"] <= 0 for op in (*cap_ops, *pred_ops)):
        issues.append({"code": "zero_duration_op"})
    if explicit_alignment.get("coverage_complete") is False:
        issues.append({"code": "declared_alignment_incomplete"})
    if added or removed:
        issues.append({"code": "unmatched_occurrences",
                       "added": [op["name"] for op in added],
                       "removed": [op["name"] for op in removed]})
    if structure["cycle"] or collective["cycle"] or captured_phase["cycle"]:
        issues.append({"code": "dependency_cycle"})
    if pred_replay["cycle"]:
        issues.append({"code": "predicted_dependency_cycle"})
    conservation_exact = abs(bucket_sum - total_delta) <= 1e-9
    if not conservation_exact:
        issues.append({"code": "conservation_mismatch",
                       "abs_error_ms": abs(bucket_sum - total_delta)})
    tolerance = max(1e-6, 1e-4 * abs(observed))
    reconciled_residual = abs(residual) <= tolerance
    if not reconciled_residual:
        issues.append({"code": "unexplained_residual",
                       "residual_ms": round(residual, 9),
                       "tolerance_ms": tolerance})
    reconciled = (not issues) and conservation_exact and reconciled_residual
    status = "reconciled" if reconciled else "partial"

    aic = captured.get("aic_mhz")
    nominal = captured.get("nominal_aic_mhz")
    telemetry = captured.get("clock_telemetry")
    if isinstance(telemetry, Mapping):
        aic = aic if aic is not None else telemetry.get("aic_mhz")
        nominal = nominal if nominal is not None else telemetry.get("nominal_mhz")
    aic_mhz, nominal_mhz = num(aic), num(nominal)
    measured = aic_mhz > 0 and nominal_mhz > 0
    derate = aic_mhz / nominal_mhz if measured else None
    clock_ratio_below_nominal = bool(measured and derate < 0.98)

    straggler = num(captured.get("straggler_ratio") or captured.get("routing_skew"), 1.0)
    collected_ms = math.fsum(op["duration"] for op in cap_ops if op["collective"])
    baseline_ms = math.fsum(op["duration"] for op in pred_ops if op["collective"])
    skew = captured.get("skew_excess_ms")
    if skew is None:
        skew = d_comm * (1.0 - 1.0 / max(1.0, straggler)) if straggler > 1 else 0.0
    congestion = d_comm - num(skew)

    offenders = []
    for item in matched:
        cap_op, pred_op = item["captured"], item["predicted"]
        delta = cap_op["duration"] - pred_op["duration"]
        if abs(delta) < EPS:
            continue
        uid = cap_op.get("op_uid") or _op_uid(cap_op["run_id"], cap_op)
        offenders.append({
            "name": cap_op["name"], "stream": cap_op["stream"],
            "occurrence_index": cap_op["raw_op_index"],
            "raw_op_index": cap_op["raw_op_index"], "op_uid": uid,
            "selector": {"run_id": cap_op["run_id"], "rank": cap_op["rank"],
                         "step": cap_op["step"], "stream": cap_op["stream"],
                         "raw_op_index": cap_op["raw_op_index"],
                         "abi_sha256": _abi_digest(cap_op) or None},
            "captured_ms": round(cap_op["duration"], 9),
            "predicted_ms": round(pred_op["duration"], 9),
            "delta_ms": round(delta, 9), "fallback": cap_op["fallback"],
            "repro_command": (
                f"mindie-profiler repro-op --capture {_quote(capture_path)} "
                f"--occurrence-index {cap_op['raw_op_index']} "
                f"--op-uid {_quote(uid)} "
                f"--output-dir {_quote(_artifact_dir(capture_path, uid))}"),
        })
    offenders.sort(key=lambda item: item["delta_ms"], reverse=True)

    host_ratio = (host_bubble / observed) if observed > 0 else 0.0
    capture_version = str(captured.get("cann_version")
                          or captured.get("cann_release")
                          or captured.get("cann_build") or "").strip()
    version_arg = f" --version {_quote(capture_version)}" if capture_version else ""
    for offender in offenders:
        offender["repro_command"] = offender["repro_command"] + version_arg
    return {
        "schema": "mindie-diff-trace/2", "status": status,
        "predicted_total_ms": predicted_total, "captured_total_ms": observed,
        "total_delta_ms": total_delta,
        "stages": {"R0": r0, "R1": r1, "R2": r2, "R3": r3,
                   "R4_model": r4, "O": observed},
        "buckets": buckets,
        "attribution": {
            "graph_delta_ms": d_graph, "sched_delta_ms": d_host,
            "comm_delta_ms": d_comm, "op_delta_ms": roofline,
            "residual_ms": residual, "total_delta_ms": total_delta,
            "sum_ms": bucket_sum,
        },
        "reconciliation": {
            "reconciled": reconciled, "residual_ms": residual,
            "tolerance_ms": tolerance, "captured_total_explicit": total_explicit,
            "captured_terminal_evidence": terminal_evidence,
            "criterion": "abs(residual) <= max(1e-6, 1e-4 * observed_ms)",
        },
        "host_bound_ratio": round(min(1.0, max(0.0, host_ratio)), 9),
        "graph_lowering": {
            "delta_ms": d_graph,
            "added_ops": [{"name": op["name"], "duration_ms": op["duration"],
                           "stream": op["stream"],
                           "raw_op_index": op["raw_op_index"]} for op in added],
            "removed_ops": [{"name": op["name"], "duration_ms": op["duration"],
                             "stream": op["stream"],
                             "raw_op_index": op["raw_op_index"]} for op in removed],
            "fallback_ops": [item["captured"]["name"] for item in matched
                             if item["captured"]["fallback"]],
            "r1_wall_ms": structure["wall_ms"],
        },
        "host_dispatch": {
            "delta_ms": d_host, "host_bound_ratio": round(min(1.0, max(0.0, host_ratio)), 9),
            "device_idle_ms": idle_cap, "gap_evidence_ms": evidence_cap,
            "host_release_evidence_ms": d_host,
            "predicted_idle_ms": idle_pred, "predicted_gap_evidence_ms": evidence_pred,
            "modeled_launch_ms": modeled_launch, "predicted_modeled_launch_ms":
            pred_modeled_launch,
            "launch_overhead_us": overhead, "launch_count": launch_count,
            "predicted_launch_overhead_us": pred_overhead,
            "gap_count": gap_count, "predicted_gap_count": pred_gap_count,
            "gap_union_ms": _total_snapshot(_union_snapshot(
                [[gap["start"], gap["start"] + gap["duration"]]
                 for gap in _normalize_gaps_snapshot(captured)], window=window_cap)),
        },
        "collective_and_skew": {
            "delta_ms": d_comm, "congestion_delta_ms": congestion,
            "routing_skew_delta_ms": num(skew), "straggler_ratio": straggler,
            "captured_collective_ms": collected_ms,
            "predicted_collective_ms": baseline_ms,
            "r3_collective_wall_ms": collective["wall_ms"],
        },
        "kernel_roofline_and_clock": {
            "delta_ms": d_op, "roofline_clock_explained_ms": roofline,
            "unexplained_residual_ms": residual,
            "clock_status": ("clock_ratio_below_nominal" if measured
                             and clock_ratio_below_nominal else
                             "nominal" if measured else "unavailable"),
            "aic_mhz": aic_mhz if measured else None,
            "nominal_aic_mhz": nominal_mhz if measured else None,
            "clock_derate_factor": round(derate, 6) if measured else None,
            "clock_ratio_below_nominal": (clock_ratio_below_nominal
                                          if measured else None),
            "thermal_attribution": "not_measured",
            "temperature_telemetry": False,
            "thermal_throttled": None,
            "top_offenders": offenders[:max(0, int(top_n))],
        },
        "coverage": {
            "coverage_ratio": round(coverage_ratio, 6),
            "matched_ops": len(matched), "captured_ops": len(cap_ops),
            "predicted_ops": len(pred_ops), "added_ops": len(added),
            "removed_ops": len(removed),
            "match_key": ["step", "rank", "stream", "op_name", "abi_digest",
                          "stream_occurrence_index"],
            "alignment": dict(explicit_alignment),
            "deps_replayed": structure["has_deps"],
            "stream_fifo": True,
            "window_captured_ms": list(window_cap),
            "window_predicted_ms": list(window_pred),
        },
        "issues": issues,
        "conservation": {
            "sum_delta_ms": bucket_sum, "captured_minus_predicted_ms": total_delta,
            "abs_error_ms": abs(bucket_sum - total_delta),
            "exact": conservation_exact,
            "signed_buckets": True,
        },
    }


def attribute_diff(predicted: Mapping[str, Any], captured: Mapping[str, Any], *,
                   top_n: int = 5, capture_path: str = "capture.json",
                   launch_overhead_us: float | None = None) -> dict[str, Any]:
    """Return an attribution only when its computed payload is finite JSON."""
    try:
        result = _compute_diff(predicted, captured, top_n=top_n,
                               capture_path=capture_path,
                               launch_overhead_us=launch_overhead_us)
    except TimeoutError:
        raise
    except OverflowError as exc:
        raise ValueError("diff-trace arithmetic exceeds the finite range") from exc
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return result


def _quote(text: str) -> str:
    import shlex
    return shlex.quote(str(text))


def _artifact_dir(capture_path: str, uid: str) -> str:
    from pathlib import Path
    token = str(uid).split(":")[-1][:16]
    return str(Path(str(capture_path)).with_suffix("")) + f".{token}"


# Internal calls consume detached inputs; public aliases validate ingress.
_op_name_snapshot = op_name
op_name = snapshot_call(op_name, 'op')
_duration_snapshot = duration
duration = snapshot_call(duration, 'op')
_start_snapshot = start
start = snapshot_call(start, 'op')
_is_collective_snapshot = is_collective
is_collective = snapshot_call(is_collective, 'op')
_is_host_snapshot = is_host
is_host = snapshot_call(is_host, 'op')
_is_fallback_snapshot = is_fallback
is_fallback = snapshot_call(is_fallback, 'op')
_normalize_ops_snapshot = normalize_ops
normalize_ops = snapshot_call(normalize_ops, 'packet')
_replay_snapshot = replay
replay = snapshot_call(replay, 'ops', 'durations', 'window', 'starts', 'release_delays')
_match_key_snapshot = match_key
match_key = snapshot_call(match_key, 'op')
_attribute_diff_snapshot = attribute_diff
attribute_diff = snapshot_call(attribute_diff, 'predicted', 'captured')


__all__ = [
    "COVERAGE_THRESHOLD", "attribute_diff", "complement", "intersect",
    "match_key", "normalize_gaps", "normalize_ops", "replay", "total", "union",
    "window_of",
]
