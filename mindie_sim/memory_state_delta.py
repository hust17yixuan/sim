"""Reconcile a modeled hybrid state/pool delta into a memory breakdown.

[Role]
  Apply one net-once signed stage vector plus optional transient capture/
  staging bytes to an existing ``MemoryBreakdown``, recomputing the per-stage
  component sums, the limiting rank, the capacity verdict and the status from
  the post-delta rank rows.

[Invariants]
  * Measurement qualification (activation/graph/draft-weight evidence) is
    independent of current capacity feasibility: a signed release can
    requalify an over-budget baseline and a positive delta can disqualify a
    fitting one.
  * One stage's delta is charged to that stage's ranks only; a limiting-rank
    change also updates ``budget_bytes``/``device_hbm_bytes``.
  * Transient capture/transport bytes enter the transient-workspace component,
    never the persistent KV-and-state pool.
  * The six component bytes still sum to the peak after the delta.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Returns a fresh immutable record; the input is not mutated.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.memory_ledger_types import MemoryBreakdown
from mindie_sim.memory_eplb import _non_negative
from mindie_sim.nr_s7_validate import strict_bool, strict_number
from mindie_sim.schema.common import ModelingError


@typed_inputs
def apply_state_pool_delta(
    breakdown: MemoryBreakdown, *, delta_bytes: float = 0.0,
    per_stage_bytes: Sequence[float] | None = None,
    replaced_per_stage_bytes: Sequence[float] | None = None,
    transient_per_stage_bytes: Sequence[float] | None = None,
    allow_negative: bool = False,
    detail_key: str = "hybrid_state_pool_bytes",
) -> MemoryBreakdown:
    """Reconcile a modeled pool replacement into one existing breakdown.

    ``per_stage_bytes`` is charged to its own PP stage only (a ``(0, 1000)``
    vector adds zero to stage 0 and 1000 to stage 1).  When
    ``replaced_per_stage_bytes`` is supplied, the net stage delta is
    ``modeled - replaced``, which is how an already-priced baseline allocation
    is replaced net-once instead of being double counted.

    ``transient_per_stage_bytes`` books live capture/transport scratch under
    the transient-workspace component, never inside the persistent
    KV-and-state pool.  Each stage's component sum, the limiting rank, the
    capacity verdict and ``status`` are recomputed from the post-delta rank
    rows, so a signed release can requalify an over-budget baseline and a
    positive delta can disqualify a fitting one.
    """
    if not isinstance(breakdown, MemoryBreakdown):
        raise ModelingError("memory_breakdown_state_delta_target_invalid")
    strict_bool("allow_negative", allow_negative, reason="memory_breakdown")
    strict_number("delta_bytes", delta_bytes, reason="memory_breakdown")
    for values in (per_stage_bytes, replaced_per_stage_bytes, transient_per_stage_bytes):
        if values is not None and (isinstance(values, (str, bytes))
                                   or not isinstance(values, Sequence)):
            raise ModelingError("memory_breakdown_stage_vector_invalid")
    stages = len(breakdown.stage_peak_bytes)
    if per_stage_bytes is None:
        modeled = (_non_negative("hybrid_state_pool_bytes", delta_bytes),) * stages
    else:
        modeled = tuple(_signed_state_value("hybrid_state_stage_bytes", value)
                        for value in per_stage_bytes)
        if len(modeled) != stages:
            raise ModelingError(
                "memory_breakdown_state_delta_stage_count:"
                f"{len(modeled)}!={stages}")
    if replaced_per_stage_bytes is None:
        replaced = (0.0,) * stages
    else:
        replaced = tuple(_non_negative("hybrid_state_replaced_bytes", value)
                         for value in replaced_per_stage_bytes)
        if len(replaced) != stages:
            raise ModelingError(
                "memory_breakdown_state_replaced_stage_count:"
                f"{len(replaced)}!={stages}")
    if transient_per_stage_bytes is None:
        transient = (0.0,) * stages
    else:
        transient = tuple(_non_negative("hybrid_transient_stage_bytes", value)
                          for value in transient_per_stage_bytes)
        if len(transient) != stages:
            raise ModelingError(
                "memory_breakdown_transient_stage_count:"
                f"{len(transient)}!={stages}")
    net = tuple(modeled[index] - replaced[index] for index in range(stages))
    if not allow_negative and any(value < 0 for value in net):
        raise ModelingError(
            "memory_breakdown_state_delta_negative:"
            + ",".join(str(value) for value in net))
    if all(value == 0.0 for value in net) and all(
            value == 0.0 for value in transient):
        return breakdown
    components = list(breakdown.components)
    target = components[2]
    detail = dict(target.detail)
    detail[detail_key] = detail.get(detail_key, 0.0) + sum(net)
    detail["hybrid_state_charged_bytes"] = sum(
        value for value in net if value > 0)
    for index, value in enumerate(net):
        detail[f"hybrid_state_stage{index}_bytes"] = value
    scratch_component = components[5]
    scratch_detail = dict(scratch_component.detail)
    scratch_detail["aux_scratch_per_stage_bytes"] = list(transient)
    # Stage scratch is a simultaneous peak per rank, not additive across
    # stages; the total live extent is the largest stage's bytes.
    scratch_detail["aux_scratch_stage_peak_bytes"] = max(transient)
    components[5] = replace(
        scratch_component,
        per_stage_bytes=tuple(
            float(scratch_component.per_stage_bytes[index]
                  if len(scratch_component.per_stage_bytes) > index
                  else scratch_component.bytes) + transient[index]
            for index in range(stages)),
        detail=tuple(sorted(scratch_detail.items())))
    rows = tuple((dp_idx, stage, rank,
                  required + net[int(stage)] + transient[int(stage)], capacity)
                 for dp_idx, stage, rank, required, capacity
                 in breakdown.per_rank_bytes)
    stage_peak = tuple(
        max((row[3] for row in rows if int(row[1]) == stage),
            default=float(breakdown.stage_peak_bytes[stage]))
        for stage in range(stages))
    worst_rank = min(range(len(rows)), key=lambda index: (
        rows[index][4] - rows[index][3], rows[index][3], rows[index][2]))
    peak_row = rows[worst_rank]
    peak = float(peak_row[3])
    budget = float(peak_row[4])
    worst_stage = int(peak_row[1])
    other_total = 0.0
    for index, component in enumerate(components):
        if index == 2:
            continue
        stage_value = (component.per_stage_bytes[worst_stage]
                       if len(component.per_stage_bytes) > worst_stage
                       else component.bytes)
        other_total += float(stage_value)
        components[index] = replace(
            component, bytes=float(stage_value),
            per_stage_bytes=tuple(float(value)
                                  for value in component.per_stage_bytes))
    other_by_stage = [
        sum(float(component.per_stage_bytes[index])
            if len(component.per_stage_bytes) > index else float(component.bytes)
            for position, component in enumerate(components)
            if position != 2)
        for index in range(stages)]
    kv_stage_bytes = tuple(
        max(0.0, stage_peak[index] - other_by_stage[index])
        for index in range(stages))
    kv_bytes = kv_stage_bytes[worst_stage]
    components[2] = replace(target, bytes=kv_bytes,
                            per_stage_bytes=kv_stage_bytes)
    if abs(sum(component.bytes for component in components) - peak) > 1e-6:
        raise ModelingError("memory_breakdown_state_delta_sum_not_conserved")
    measured = bool(breakdown.activation_peak_qualified
                    and breakdown.graph_workspace_qualified
                    and breakdown.draft_weight_qualified)
    oom = bool(measured and all(row[3] <= row[4] for row in rows))
    if measured:
        status = "qualified" if oom else "memory_infeasible"
        reason = None if oom else "rank_capacity_exceeded"
    else:
        status, reason = breakdown.status, breakdown.reason
    top_detail = dict(breakdown.detail)
    top_detail[detail_key] = sum(net)
    # Preserve the declared utilization while following the new limiting rank.
    old_budget = float(breakdown.budget_bytes)
    utilization = (old_budget / float(breakdown.device_hbm_bytes)
                   if float(breakdown.device_hbm_bytes) > 0 else 1.0)
    return replace(
        breakdown, components=tuple(components), stage_peak_bytes=stage_peak,
        peak_hbm_bytes=peak, headroom_bytes=round(budget - peak, 6),
        budget_bytes=budget,
        device_hbm_bytes=budget / max(1e-12, utilization),
        oom_feasible=oom, memory_qualified=measured,
        dominant_assassin=max(components, key=lambda item: item.bytes).name,
        hidden_assassin_bytes=peak - breakdown.naive_weights_kv_bytes,
        status=status, reason=reason, per_rank_bytes=rows,
        worst_stage=worst_stage, worst_rank=worst_rank,
        worst_replica=int(peak_row[0]),
        detail=tuple(sorted(top_detail.items())))


def _signed_state_value(name: str, value: Any) -> float:
    return strict_number(name, value, minimum=-math.inf)



__all__ = ["apply_state_pool_delta"]
