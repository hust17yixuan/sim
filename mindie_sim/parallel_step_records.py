"""Shared parallel step records, scalar validation and physical axis probing.

[Role]
  Hold the immutable ``StepBreakdown`` record and the small numeric/endpoint
  validators and communicator probes used by the decode, prefill, draft and
  EP-knee models.  Keeping them here lets each pricing module stay focused on
  its own executed work.

[Invariants]
  * Every active axis is qualified with the collective it actually issues:
    AllReduce for TP/ETP/EDP/DCP, AllGather for PCP/PP, AllToAllV for EP.
  * An endpoint map is complete, duplicate-free, non-negative and known to the
    supplied fabric before any timing or zero-work shortcut uses it.
  * Scalar validators reject bools, non-finite values and out-of-range values
    before any coercion.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable records; caller Cases and trees untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import (
    _validate_record_domains,
    strict_int,
    strict_number,
    typed_inputs,
    typed_sequence,
)
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError

#: Every active communicator axis and the collective it actually issues.
AXIS_COLLECTIVES = (
    ("tp_groups", "all_reduce"),
    ("etp_groups", "all_reduce"),
    ("ep_groups", "all_to_all_v"),
    ("edp_groups", "all_reduce"),
    ("pp_groups", "all_gather"),
    ("pcp_groups", "all_gather"),
    ("dcp_groups", "all_reduce"),
)

SCALING_MODES = ("fixed_per_dp_batch", "fixed_global_batch")


def _positive(name: str, value: Any, *, strict: bool = True) -> float:
    number = strict_number(name, value, reason="parallel_explorer")
    assert number is not None
    if strict and number <= 0.0:
        raise ModelingError(f"parallel_explorer_{name}_invalid:{value}")
    return number


def _int(name: str, value: Any, minimum: int = 1) -> int:
    return strict_int(name, value, minimum=minimum, reason="parallel_explorer")


def _interconnect_bw_gbps(machine: Any) -> float:
    """Slowest declared link capacity in GB/s, used for EPLB rebalance cost."""
    values = [
        float(link.capacity_bytes_per_ms) * float(link.bw_utilization) / 1e6
        for link in getattr(machine, "links", ())
        if len(link.members) > 1 and float(link.capacity_bytes_per_ms) > 0
    ]
    return min(values) if values else 1.0


@typed_inputs
def iter_axis_groups(
    plan: Mapping[str, Any],
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Every multi-rank group on every active axis, not just ``ep_groups[0]``."""
    rows = []
    for axis, kind in AXIS_COLLECTIVES:
        for group in typed_sequence(axis, plan.get(axis, ()), Sequence):
            values = typed_sequence("rank_ids", group, int)
            for rank in values:
                _int("rank", rank, minimum=0)
            if len(values) > 1:
                rows.append((axis, kind, values))
    return tuple(rows)


@typed_inputs
def check_route_qualified(
    tree: Any,
    plan: Mapping[str, Any],
    rank_offset: int = 0,
    *,
    dtype: str | None = None,
    topology: str | None = None,
) -> tuple[bool, str]:
    """Route-qualify every active group on every axis at its physical devices."""
    if not callable(getattr(tree, "collective", None)):
        raise ModelingError("parallel_explorer_tree_invalid")
    _int("rank_offset", rank_offset, minimum=0)
    evidence = "single_rank"
    checked = 0
    for axis, kind, group in iter_axis_groups(plan):
        endpoints = tuple(int(rank_offset) + rank for rank in group)
        route = tree.collective(
            kind, 1.0, len(endpoints), endpoints=endpoints, dtype=dtype, topology=topology
        )
        checked += 1
        if not route.route_qualified:
            return False, f"{axis}:{route.evidence}"
        evidence = route.evidence
    return True, (evidence if checked else "single_rank")


@typed_inputs
def execution_context(case: CaseSpec) -> dict[str, Any]:
    """Actual dtype/topology evidence forwarded to a calibrated profile."""
    return {
        "dtype": str(case.model.activation_dtype),
        "topology": str(
            dict(case.machine.topology_options).get("topology_preset", case.machine.sku or "") or ""
        )
        or None,
    }


def validate_endpoint_map(
    endpoints: Sequence[int] | None, world: int, *, tree: Any = None
) -> tuple[int, ...]:
    """Strict physical endpoint map before any timing/filtering."""
    count = int(world)
    if count < 1:
        raise ModelingError(f"parallel_explorer_world_invalid:{count}")
    if endpoints is None:
        return tuple(range(count))
    if isinstance(endpoints, (str, bytes)) or not isinstance(endpoints, Sequence):
        raise ModelingError(f"parallel_explorer_endpoints_invalid:{endpoints}")
    values = tuple(endpoints)
    if len(values) != count:
        raise ModelingError(f"parallel_explorer_endpoints_count:{len(values)}!={count}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"parallel_explorer_endpoint_invalid:{value}")
    if len(set(values)) != len(values):
        raise ModelingError(f"parallel_explorer_endpoint_duplicate:{values}")
    known = {int(item.rank) for item in getattr(tree, "endpoints", ())} if tree is not None else set()
    if known:
        unknown = tuple(value for value in values if value not in known)
        if unknown:
            raise ModelingError(f"parallel_explorer_endpoint_unknown:{unknown}")
    return values


@typed_inputs
@dataclass(frozen=True)
class StepBreakdown:
    """One decode step's useful-work ledger and executed communication record."""

    batch_tokens: int
    step_ms: float
    t_attn_ms: float
    t_expert_compute_ms: float
    t_expert_straggler_ms: float
    t_dispatch_combine_ms: float
    t_non_moe_ms: float
    expert_weight_ms: float
    active_expert_fraction: float
    straggler_ratio: float
    bottleneck: str
    moe_layers: int = 0
    layers: int = 0
    raw_comm_ms: float = 0.0
    host_exposed_ms: float = 0.0
    host_bound_ratio: float = 0.0
    pp_fill_drain_ms: float = 0.0
    tp_allreduce_ms: float = 0.0
    dcp_reduce_ms: float = 0.0
    active_replicas: int = 1
    global_batch: int = 0
    per_replica_batch: int = 0
    useful_tokens_per_step: int = 0
    expert_assignments: float = 0.0
    expert_assignments_per_owner: float = 0.0
    scaling_mode: str = "fixed_per_dp_batch"
    mc2: bool = False
    schedule: tuple[tuple[str, float], ...] = ()
    tps: float = 0.0
    tps_per_die: float = 0.0
    expert_flops: float = 0.0
    expert_flops_per_owner: float = 0.0
    tokens_entering_ep_group: float = 0.0
    tile_tokens: int = 0
    tile_wave_ms: float = 0.0
    uncached_fallback_ms: float = 0.0
    mlapo_prefetch: bool = True
    pp_microbatches: int = 1
    overlap_mode: str = "serial"
    eplb_rebalance_ms: float = 0.0
    pp_stage_ms: tuple[float, ...] = ()
    pp_max_stage_ms: float = 0.0
    two_stream_exposed_ms: float = 0.0
    pp_pipeline_bubble_ms: float = 0.0
    pp_pipeline_makespan_ms: float = 0.0
    pp_serial_stage_ms: float = 0.0
    pp_p2p_ms: float = 0.0
    route_qualified: bool = True
    route_evidence: str = "declared"
    speculative_draft_ms: float = 0.0
    comm_ledger: Any = None
    shared_cut_demands: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _validate_record_domains(self)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            result[name] = value.to_dict() if hasattr(value, "to_dict") else value
        return result


for _exported in (iter_axis_groups, check_route_qualified, execution_context, StepBreakdown):
    _exported.__module__ = "mindie_sim.parallel_decode_cost"

__all__ = [
    "AXIS_COLLECTIVES",
    "SCALING_MODES",
    "StepBreakdown",
    "_int",
    "_interconnect_bw_gbps",
    "_positive",
    "check_route_qualified",
    "execution_context",
    "iter_axis_groups",
    "validate_endpoint_map",
]
