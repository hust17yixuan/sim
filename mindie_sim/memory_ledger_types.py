"""Memory ledger component records and construction helpers.

[Role]
  Define the six-component per-rank memory record and construct its weight,
  draft, communicator-arena and ACLGraph sub-ledgers from declared placement.

[Invariants]
  * Exactly six ordered components sum to ``peak_hbm_bytes``.
  * Explicit reserves are additive; communicator arenas are owned per rank.
  * An explicit graph capture reserve is persistent and added on top of the
    operator workspace, never absorbed by ``max``.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable records; caller Cases untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import stage_weight_terms
from mindie_sim.memory_eplb import _align, _non_negative
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes


MIB = 1024.0 * 1024.0
COMPONENT_NAMES = (
    "sharded_weights_bytes", "aux_weights_bytes", "kv_and_state_pool_bytes",
    "collective_static_buffers_bytes", "graph_capture_pool_bytes",
    "transient_workspace_peak_bytes",
)
_AXIS_FIELDS = ("tp", "ep", "pcp", "cp", "pp", "dp")
_ALIGNMENT = 512




@dataclass(frozen=True)
class MemoryComponent:
    """One named assassin at the worst rank, plus its per-stage vector."""

    name: str
    bytes: float
    per_stage_bytes: tuple[float, ...]
    detail: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bytes": self.bytes,
                "per_stage_bytes": list(self.per_stage_bytes),
                "detail": dict(self.detail)}


@dataclass(frozen=True)
class MemoryBreakdown:
    """Complete feasibility record for one exact batch and layout."""

    components: tuple[MemoryComponent, ...]
    stage_peak_bytes: tuple[float, ...]
    peak_hbm_bytes: float
    device_hbm_bytes: float
    budget_bytes: float
    headroom_bytes: float
    oom_feasible: bool
    dominant_assassin: str
    worst_stage: int
    naive_weights_kv_bytes: float
    hidden_assassin_bytes: float
    sequence_tokens: int
    sequences: int
    parallel: tuple[tuple[str, int], ...]
    detail: tuple[tuple[str, float], ...] = ()
    worst_rank: int = 0
    worst_replica: int = 0
    per_rank_bytes: tuple[tuple[int, int, int, float, float], ...] = ()
    status: str = "qualified"
    reason: str | None = None
    fragmentation_margin_bytes: float = 0.0
    runtime_reserve_bytes: float = 0.0
    memory_qualified: bool = True
    activation_peak_qualified: bool = True
    graph_workspace_qualified: bool = True
    comm_persistent_qualified: bool = False
    comm_persistent_bytes: float = 0.0
    comm_persistent_basis: str = "heuristic_arena_sizing"
    workspace_bytes: float = 0.0
    draft_weight_qualified: bool = True
    draft_weight_bytes: float = 0.0
    kv_transfer_staging_bytes: float = 0.0
    kv_transfer_fields: tuple[tuple[str, float], ...] = ()
    kv_transfer_page_owner_ranks: tuple[int, ...] = ()
    per_rank_comm_bytes: tuple[float, ...] = ()
    per_rank_capacity_bytes: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if tuple(component.name for component in self.components) != COMPONENT_NAMES:
            raise ModelingError("memory_breakdown_components_out_of_order")
        total = sum(component.bytes for component in self.components)
        if abs(total - self.peak_hbm_bytes) > max(1.0, self.peak_hbm_bytes * 1e-9):
            raise ModelingError(f"memory_breakdown_sum_mismatch:{total}!={self.peak_hbm_bytes}")

    @property
    def qualified(self) -> bool:
        """Public alias: both capacity and every dynamic-pool evidence pass."""
        return bool(self.oom_feasible and self.memory_qualified)

    @property
    def feasible(self) -> bool:
        return self.qualified

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mindie-sim.memory-breakdown/2",
            "components": [component.to_dict() for component in self.components],
            "stage_peak_bytes": list(self.stage_peak_bytes),
            "peak_hbm_bytes": self.peak_hbm_bytes,
            "device_hbm_bytes": self.device_hbm_bytes,
            "budget_bytes": self.budget_bytes,
            "headroom_bytes": self.headroom_bytes,
            "oom_feasible": self.oom_feasible,
            "qualified": self.qualified, "feasible": self.feasible,
            "status": self.status, "reason": self.reason,
            "dominant_assassin": self.dominant_assassin,
            "worst_stage": self.worst_stage, "worst_rank": self.worst_rank,
            "worst_replica": self.worst_replica,
            "naive_weights_kv_bytes": self.naive_weights_kv_bytes,
            "hidden_assassin_bytes": self.hidden_assassin_bytes,
            "sequence_tokens": self.sequence_tokens, "sequences": self.sequences,
            "parallel": dict(self.parallel), "detail": dict(self.detail),
            "per_rank_bytes": [list(row) for row in self.per_rank_bytes],
            "fragmentation_margin_bytes": self.fragmentation_margin_bytes,
            "runtime_reserve_bytes": self.runtime_reserve_bytes,
            "memory_qualified": self.memory_qualified,
            "activation_peak_qualified": self.activation_peak_qualified,
            "graph_workspace_qualified": self.graph_workspace_qualified,
            "comm_persistent_qualified": self.comm_persistent_qualified,
            "comm_persistent_bytes": self.comm_persistent_bytes,
            "comm_persistent_basis": self.comm_persistent_basis,
            "workspace_bytes": self.workspace_bytes,
            "draft_weight_qualified": self.draft_weight_qualified,
            "draft_weight_bytes": self.draft_weight_bytes,
            "kv_transfer_staging_bytes": self.kv_transfer_staging_bytes,
            "kv_transfer_fields": dict(self.kv_transfer_fields),
            "kv_transfer_page_owner_ranks": list(
                self.kv_transfer_page_owner_ranks),
            "per_rank_comm_bytes": list(self.per_rank_comm_bytes),
            "per_rank_capacity_bytes": list(self.per_rank_capacity_bytes),
        }


def _stage_weights(case: CaseSpec, mem: Memory, *, include_draft: bool):
    model, p = case.model, case.runtime.parallel
    quant = model.quant or QuantSpec("bf16")
    tables = not case.runtime.engram_cpu_offload
    stages = stage_weight_terms(
        model, p, include_draft=include_draft, include_engram_tables=tables,
        layer_partition=mem.layers, draft_kind=mem.draft_kind if include_draft else None,
    )
    base = [0.0] * p.pp
    categories = [dict() for _ in range(p.pp)]
    for stage, terms in enumerate(stages):
        for term in terms:
            placed = p
            if term.category == "shared_expert" and dict(case.runtime.metadata).get(
                    "enable_shared_expert_dp", False):
                placed = replace(p, tp=1)
            value = float(term_bytes(term, quant, placed, case.machine.ranks_per_node))
            base[stage] += value
            categories[stage][term.category] = categories[stage].get(term.category, 0.0) + value
    return base, categories


def _draft_bytes(case: CaseSpec, mem: Memory) -> tuple[list[float], dict[str, float]]:
    if not mem.draft_kind:
        return [0.0] * case.runtime.parallel.pp, {}
    with_draft, _ = _stage_weights(case, mem, include_draft=True)
    without, _ = _stage_weights(case, mem, include_draft=False)
    auxiliary = getattr(case.model, "auxiliary", None)
    draft_layers = sum(
        int(getattr(draft, "layers", 0) or 0)
        for draft in ((getattr(auxiliary, "mtp", None), getattr(auxiliary, "dspark", None))
                      if auxiliary is not None else ())
        if draft is not None)
    return ([a - b for a, b in zip(with_draft, without)],
            {"draft_layers": float(draft_layers),
             "mtp_present": float(mem.draft_kind == "mtp")})


def   _communicator_groups(case: CaseSpec, plan, buffsize_mb: float,
                         mc2_multiplier: float, recipe: str,
                         arena_sharing: Mapping[str, str] | None = None,
                         explicit_domains: Mapping[str, float] | None = None):
    """Arenas owned by each physical rank, built from actual placement groups.

    A rank is charged once per communicator whose endpoint set contains it.
    TP, expert-TP, PCP, DCP, PP P2P, EP AllToAllV and MC2 are distinct
    persistent HCCL arenas; axis cardinality is not arena count.
    """
    p = case.runtime.parallel
    recipe = str(recipe or "alltoall").strip().lower()
    if recipe not in {"alltoall", "mc2", "tp_join_reduce", "allgather_reducescatter"}:
        raise ModelingError(f"memory_breakdown_moe_recipe_unsupported:{recipe}")
    mc2 = recipe == "mc2" and int(p.ep) > 1
    world = int(p.world_size)
    sharing = dict(arena_sharing or {})
    rank_raw: list[dict[str, float]] = [dict() for _ in range(world)]
    seen_groups: list[dict[str, Any]] = []

    domains = {str(key): float(value)
               for key, value in dict(explicit_domains or {}).items()}
    applied_domains: set[str] = set()
    heuristic_arenas: set[str] = set()

    def add(name: str, groups, directions: int, slots: int, protocol: str,
            metadata: float = 4096.0) -> None:
        storage = sharing.get(name, name)
        explicit = domains.get(storage, domains.get(name))
        if explicit is not None:
            # Versioned, caller-supplied domain arena: the declared backing
            # bytes are the resource, not a heuristic direction/slot product.
            raw = float(explicit) + float(metadata)
        else:
            raw = (buffsize_mb * MIB * int(directions) * int(slots)
                   * (mc2_multiplier if protocol == "mc2" else 1.0)
                   + float(metadata))
        value = _align(raw)
        unique = tuple(dict.fromkeys(tuple(int(rank) for rank in group)
                                     for group in groups))
        used = False
        for group in unique:
            if len(group) <= 1:
                continue
            used = True
            seen_groups.append({"name": name, "size": len(group),
                                "protocol": protocol, "storage": storage})
            for rank in group:
                if 0 <= int(rank) < world:
                    rank_raw[int(rank)][storage] = max(
                        rank_raw[int(rank)].get(storage, 0.0), value)
        if used:
            if explicit is not None:
                applied_domains.add(storage if storage in domains else name)
            else:
                heuristic_arenas.add(storage)

    add("tp", plan.tp_groups, 2, 1, "hccl_ring")
    add("etp", plan.etp_groups, 2, 1, "hccl_ring")
    add("pcp", plan.pcp_groups, 2, 1, "hccl_ring")
    add("dcp", plan.dcp_groups, 2, 1, "hccl_ring")
    add("pp", plan.pp_groups, 2, 1, "hccl_p2p")
    if mc2:
        add("ep_mc2", plan.ep_groups, 2, 2, "mc2", 8192.0)
    else:
        add("ep_alltoall", plan.ep_groups, 2, 1, "alltoall")
    per_rank = tuple(float(sum(row.values())) for row in rank_raw)
    total = float(max(per_rank, default=0.0))
    storage_max: dict[str, float] = {}
    for row in rank_raw:
        for storage, value in row.items():
            storage_max[storage] = max(storage_max.get(storage, 0.0), value)
    detail: dict[str, float] = {
        f"arena_{storage}_bytes": value for storage, value in storage_max.items()}
    detail.update({"arena_count": float(len(storage_max)),
                   "group_count": float(len(seen_groups)),
                   "max_rank_arena_count": float(max(
                       (len(row) for row in rank_raw), default=0)),
                   "mc2_recipe": float(mc2), "recipe_" + recipe: 1.0,
                   "explicit_domain_count": float(len(applied_domains)),
                   "heuristic_arena_count": float(len(heuristic_arenas))})
    return total, per_rank, detail, applied_domains, heuristic_arenas

def  _graph_pool(case: CaseSpec, *, buckets, activation_per_token, fixed_bytes,
                pool_mode: str, analytical_bound: bool = False):
    """Return ``(bytes, activation_unqualified, workspace_unqualified, detail)``.

    A declared fixed workspace (even one byte) qualifies only that fixed
    allocation.  It can never qualify an unmeasured activation extent.
    """
    if activation_per_token is not None:
        activation_per_token = _non_negative("graph_activation_bytes_per_token",
                                             activation_per_token)
    if fixed_bytes is not None:
        fixed_bytes = _non_negative("graph_workspace_bytes", fixed_bytes)
    mode = str(case.runtime.graph_capture_mode or "none")
    if mode in {"none", ""}:
        return 0.0, 0.0, 0.0, {"graph_buckets": 0.0, "capture_mode_numeric": 0.0,
                               "graph_pool_shared": 0.0,
                               "graph_activation_qualified": 1.0,
                               "graph_workspace_qualified": 1.0}
    declared = (case.runtime.piecewise_capture_keys if mode == "piecewise"
                else case.runtime.graph_capture_keys or case.runtime.graph_buckets)
    raw_buckets = tuple(buckets if buckets is not None else
                        (declared or (case.runtime.max_num_batched_tokens,)))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in raw_buckets):
        raise ModelingError(f"memory_breakdown_graph_buckets_invalid:{raw_buckets}")
    sizes = tuple(sorted({value for value in raw_buckets if value > 0}))
    activation_declared = activation_per_token is not None
    workspace_declared = fixed_bytes is not None
    if activation_per_token is None:
        hidden = case.model.hidden_size
        activation_per_token = float(hidden * dtype_bytes(case.model.activation_dtype) * 4)
    if fixed_bytes is None:
        fixed_bytes = float(case.model.hidden_size * dtype_bytes(
            case.model.activation_dtype) * 4)
    extents = [size * float(activation_per_token) + float(fixed_bytes) for size in sizes]
    if pool_mode not in {"independent", "shared"}:
        raise ModelingError(f"memory_breakdown_graph_pool_mode_invalid:{pool_mode}")
    shared = pool_mode == "shared"
    total = max(extents, default=0.0) if shared else float(sum(extents))
    activation_qualified = bool(activation_declared or analytical_bound)
    workspace_qualified = bool(workspace_declared or analytical_bound)
    activation_unqualified = 0.0 if activation_qualified else 1.0
    workspace_unqualified = 0.0 if workspace_qualified else 1.0
    return total, activation_unqualified, workspace_unqualified, {
        "graph_buckets": float(len(sizes)), "bucket_tokens_total": float(sum(sizes)),
        "activation_bytes_per_token": float(activation_per_token),
        "fixed_bytes_per_bucket": float(fixed_bytes),
        "graph_pool_shared": float(shared),
        "max_bucket_arena_bytes": float(max(extents, default=0.0)),
        "graph_estimate_unqualified": max(activation_unqualified,
                                          workspace_unqualified),
        "graph_activation_qualified": 1.0 - activation_unqualified,
        "graph_workspace_qualified": 1.0 - workspace_unqualified,
    }


