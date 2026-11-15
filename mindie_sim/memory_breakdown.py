"""HBM hidden-assassins ledger with six explicit per-rank components.

[Role]
  Decompose one parallel plan into the resident per-rank HBM footprint a serving
  engine holds: sharded weights, auxiliary (MTP/EPLB) weights, paged KV plus
  sparse/recurrent state, communicator arenas, ACLGraph pools and workspace.

[Invariants]
  * Exactly six components are emitted in order and sum to ``peak_hbm_bytes``.
  * Explicit ``MemoryPolicy`` reserves are real allocations; runtime/workspace
    are added, and persistent arenas/graphs are additive unless an explicit
    backing handle proves reuse.
  * Each rank uses its own die capacity, KV ownership and communicator
    memberships; the limiting rank is the maximum ``required/capacity`` ratio.
  * Paged KV is evaluated at the exact admitted batch and full ``ISL+OSL`` span.
  * Arenas come from owned groups/protocol/slots; axis count is not arena count
    and MC2 exists only for the declared MC2 recipe.  Heuristic sizing is never
    persistent-qualified; only explicit versioned HCCL bytes or qualified
    domain arenas qualify, otherwise strict memory qualification fails.
  * An ACLGraph pool is independent per bucket or one shared max-extent arena
    across buckets; captured activations are persistent and additive unless a
    named backing handle proves the alias, and a fixed workspace never
    qualifies an unmeasured extent.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Immutable outputs; caller Cases are not mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any

from mindie_sim.catalog.types import AuxiliarySpec
from mindie_sim.memory_eplb import (
    _cache_detail,
    _eplb_extra_bytes,
    _non_negative,
    _resolve_int,
    resolve_eplb_epoch_config,
)
from mindie_sim.memory_ledger_types import (
    COMPONENT_NAMES,
    MIB,
    MemoryBreakdown,
    MemoryComponent,
    _AXIS_FIELDS,
    _communicator_groups,
    _draft_bytes,
    _graph_pool,
    _stage_weights,
)
from mindie_sim.nr_s7_validate import _validate_nested, strict_int, typed_inputs
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.hardware import Device


@typed_inputs
@dataclass
class MemoryLedger:
    """Reusable per-plan ledger; ``at`` evaluates exact batches cheaply."""

    case: CaseSpec
    mem: Memory
    pp: int
    sharded: list[float]
    aux: list[float]
    comm_bytes: float
    comm_detail: tuple[tuple[str, float], ...]
    graph_bytes: float
    graph_unqualified: float
    graph_detail: tuple[tuple[str, float], ...]
    graph_shared: bool
    transient_base: list[float]
    workspace: list[float]
    logits_per_sequence: float
    cache_one: list[float]
    cache_detail: list[dict[str, float]]
    stage_capacity: list[float]
    stage_dies: list[list[Device]]
    rank_layout: list[list[list[Device]]]
    rank_cache: list[list[list[float]]]
    runtime_reserve: float
    fragmentation: float
    fusion_detail: tuple[tuple[str, float], ...] = ()
    rank_map: tuple[tuple[int, ...], ...] = ()
    comm_rank: tuple[float, ...] = ()
    aux_rank: tuple[tuple[float, ...], ...] = ()
    graph_rank: tuple[float, ...] = ()
    capacity_rank: tuple[float, ...] = ()
    activation_peak_qualified: bool = True
    graph_workspace_qualified: bool = True
    draft_weight_qualified: bool = True
    draft_weight_bytes: float = 0.0
    kv_transfer_staging: float = 0.0
    kv_transfer_fields: tuple[tuple[str, float], ...] = ()
    kv_transfer_page_owner_ranks: tuple[int, ...] = ()
    comm_persistent_qualified: bool = False
    comm_persistent_basis: str = "heuristic_arena_sizing"
    strict_comm_qualification: bool = False
    transfer_rank: tuple[float, ...] = ()
    cache_mem: Memory | None = None
    _cache_cache: dict[int, list[list[float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = self.case.runtime.parallel
        if self.pp < 1 or self.pp != p.pp:
            raise ModelingError("memory_ledger_stage_shape_invalid")
        for name in (
            "sharded", "aux", "transient_base", "workspace", "cache_one",
            "cache_detail", "stage_capacity", "stage_dies",
        ):
            if len(getattr(self, name)) != self.pp:
                raise ModelingError(f"memory_ledger_{name}_stage_shape_invalid")
        for name in ("rank_map", "comm_rank", "aux_rank", "graph_rank", "capacity_rank", "transfer_rank"):
            values = getattr(self, name)
            if values and len(values) != p.world_size:
                raise ModelingError(f"memory_ledger_{name}_rank_shape_invalid")
        limits = (p.pp, p.dp, p.pcp, p.dcp, p.tp, p.ep)
        for coordinate in self.rank_map:
            if len(coordinate) != 6 or any(
                not 0 <= value < limit for value, limit in zip(coordinate, limits)
            ):
                raise ModelingError("memory_ledger_rank_coordinate_invalid")
        if any(len(row) != self.pp for row in self.aux_rank):
            raise ModelingError("memory_ledger_aux_rank_stage_shape_invalid")
        if self.rank_layout and (
            len(self.rank_layout) != p.dp or any(len(row) != self.pp for row in self.rank_layout)
        ):
            raise ModelingError("memory_ledger_rank_layout_stage_shape_invalid")

    def _rank_cache_at(self, tokens: int) -> list[list[float]]:
        cached = self._cache_cache.get(tokens)
        if cached is not None:
            return cached
        p = self.case.runtime.parallel
        rows: list[list[float]] = []
        cm = self.cache_mem or self.mem
        for rank in range(int(p.world_size)):
            coordinate = (
                self.rank_map[rank]
                if self.rank_map
                else (rank // max(1, p.dp * p.tp), 0, 0, 0, 0, 0)
            )
            stage, _dp, pcp, _dcp, tp, _ep = coordinate
            cache_rank = int(tp) * int(p.pcp) + int(pcp)
            try:
                values = cm._rank_cache_bytes(tokens, padded=True, rank=cache_rank)
                value = float(values[int(stage)])
            except (ValueError, IndexError):
                value = float(self.cache_one[int(stage)])
            rows.append([value if int(stage) == index else 0.0 for index in range(self.pp)])
        self._cache_cache[tokens] = rows
        return rows

    def at(self, sequences: int, sequence_tokens: int) -> MemoryBreakdown:
        seqs = _resolve_int("sequences", sequences, 0)
        tokens = _resolve_int("sequence_tokens", sequence_tokens, 0)
        if seqs < 1 or tokens < 1:
            raise ModelingError("memory_breakdown_requires_positive_tokens_and_sequences")
        p = self.case.runtime.parallel
        world = int(p.world_size)
        if self.rank_map and len(self.rank_map) != world:
            raise ModelingError("memory_ledger_rank_map_shape_invalid")
        cm = self.cache_mem or self.mem
        paged = [float(value) * seqs for value in cm.cache_bytes(tokens, padded=True)]
        rank_cache = self._rank_cache_at(tokens)
        logits = min(max(1, int(self.case.runtime.max_num_batched_tokens)), seqs) * self.logits_per_sequence
        stage_peak = [0.0] * self.pp
        rank_rows: list[tuple[int, int, int, float, float]] = []
        kv_by_stage = [0.0] * self.pp
        transient_by_stage = list(self.transient_base)
        rank_parts: list[tuple[float, float, float, float, float, float]] = []
        for rank in range(world):
            coordinate = (
                self.rank_map[rank]
                if self.rank_map
                else (rank // max(1, p.dp * p.tp), 0, 0, 0, 0, 0)
            )
            stage, dp_idx = int(coordinate[0]), int(coordinate[1])
            value = float(rank_cache[rank][stage])
            kv_value = value * seqs if value > 0 else paged[stage]
            aux_value = float(self.aux_rank[rank][stage]) if self.aux_rank else float(self.aux[stage])
            comm_value = float(self.comm_rank[rank]) if self.comm_rank else float(self.comm_bytes)
            graph_value = float(self.graph_rank[rank]) if self.graph_rank else float(self.graph_bytes)
            capacity = float(self.capacity_rank[rank]) if self.capacity_rank else float(self.stage_capacity[stage])
            kv_by_stage[stage] = max(kv_by_stage[stage], kv_value)
            transient = self.transient_base[stage] + (self.transfer_rank[rank] if self.transfer_rank else 0.0)
            transient_by_stage[stage] = max(transient_by_stage[stage], transient)
            required = (
                self.sharded[stage] + aux_value + kv_value + comm_value
                + graph_value + transient + logits + self.fragmentation
            )
            stage_peak[stage] = max(stage_peak[stage], required)
            rank_rows.append((dp_idx, stage, rank, required, capacity))
            rank_parts.append((self.sharded[stage], aux_value, kv_value, comm_value, graph_value, transient))
        worst_rank = min(
            range(world),
            key=lambda index: (
                rank_rows[index][4] - rank_rows[index][3],
                rank_rows[index][3],
                rank_rows[index][2],
            ),
        )
        peak_row = rank_rows[worst_rank]
        peak, budget = float(peak_row[3]), float(peak_row[4])
        worst_stage, worst_replica = int(peak_row[1]), int(peak_row[0])
        sharded, aux_value, kv_value, comm_value, graph_value, transient = rank_parts[worst_rank]
        kv_detail = self.cache_detail[worst_stage]
        detail = {
            "paged_kv": float(
                kv_detail.get("gqa_kv", 0.0) + kv_detail.get("mla_kv", 0.0)
                + kv_detail.get("compressed_kv", 0.0) + kv_detail.get("window", 0.0)
                + kv_detail.get("draft_kv", 0.0)
            ),
            "index_k": float(kv_detail.get("index_k", 0.0) + kv_detail.get("index_pool_tail", 0.0)),
            "recurrent_state": float(
                kv_detail.get("recurrent_state", 0.0) + kv_detail.get("conv_state", 0.0)
                + kv_detail.get("compressor_state", 0.0) + kv_detail.get("index_compressor_state", 0.0)
            ),
            "page_group_or_rounding_reserve": max(0.0, paged[worst_stage] - float(sum(kv_detail.values()))),
        }
        components = (
            MemoryComponent("sharded_weights_bytes", sharded, tuple(self.sharded), ()),
            MemoryComponent("aux_weights_bytes", aux_value, tuple(self.aux), ()),
            MemoryComponent("kv_and_state_pool_bytes", kv_value, tuple(kv_by_stage), tuple(sorted(detail.items()))),
            MemoryComponent(
                "collective_static_buffers_bytes", comm_value,
                (float(max(self.comm_rank, default=self.comm_bytes)),) * self.pp, self.comm_detail,
            ),
            MemoryComponent(
                "graph_capture_pool_bytes", graph_value,
                (float(max(self.graph_rank, default=self.graph_bytes)),) * self.pp, self.graph_detail,
            ),
            MemoryComponent(
                "transient_workspace_peak_bytes", transient + logits + self.fragmentation,
                tuple(transient_by_stage[i] + logits + self.fragmentation for i in range(self.pp)),
                (
                    ("logits_bytes", logits),
                    ("runtime_reserve_bytes", self.runtime_reserve),
                    ("fragmentation_margin_bytes", self.fragmentation),
                    ("operator_workspace_bytes", self.workspace[worst_stage]),
                    ("temporary_dequant_bytes", float(self.mem.temporary_dequant[worst_stage])),
                ),
            ),
        )
        status, reason = "qualified", None
        if not self.activation_peak_qualified:
            status, reason = "memory_unqualified", "activation_pool_unmeasured"
        elif not self.graph_workspace_qualified:
            status, reason = "memory_unqualified", "graph_workspace_unmeasured"
        elif not self.draft_weight_qualified:
            status, reason = "memory_unqualified", "draft_weight_unmeasured"
        elif self.strict_comm_qualification and not self.comm_persistent_qualified:
            status, reason = "memory_unqualified", "comm_persistent_unmeasured"
        elif any(row[3] > row[4] for row in rank_rows):
            status, reason = "memory_infeasible", "rank_capacity_exceeded"
        naive = sharded + aux_value + kv_value
        # Measurement qualification is independent of current capacity overflow;
        # oom_feasible combines qualification with per-rank capacity fit.
        memory_qualified = (
            self.activation_peak_qualified
            and self.graph_workspace_qualified
            and self.draft_weight_qualified
            and (self.comm_persistent_qualified or not self.strict_comm_qualification)
        )
        return MemoryBreakdown(
            components=components, stage_peak_bytes=tuple(stage_peak),
            peak_hbm_bytes=peak,
            device_hbm_bytes=budget / max(1e-12, float(self.case.runtime.gpu_memory_utilization)),
            budget_bytes=budget, headroom_bytes=round(budget - peak, 6),
            oom_feasible=bool(memory_qualified and status == "qualified" and all(row[3] <= row[4] for row in rank_rows)),
            dominant_assassin=max(components, key=lambda item: item.bytes).name,
            worst_stage=worst_stage, naive_weights_kv_bytes=naive,
            hidden_assassin_bytes=peak - naive, sequence_tokens=tokens, sequences=seqs,
            parallel=tuple((name, int(getattr(p, name))) for name in _AXIS_FIELDS),
            detail=(
                ("hccl_buffer_bytes", comm_value), ("graph_pool_bytes", graph_value),
                ("resident_weight_bytes", sharded + aux_value),
                ("runtime_reserve_bytes", self.runtime_reserve),
                ("fragmentation_margin_bytes", self.fragmentation),
            ),
            worst_rank=worst_rank, worst_replica=worst_replica,
            per_rank_bytes=tuple(rank_rows), status=status, reason=reason,
            fragmentation_margin_bytes=self.fragmentation,
            runtime_reserve_bytes=self.runtime_reserve,
            memory_qualified=memory_qualified,
            activation_peak_qualified=self.activation_peak_qualified,
            graph_workspace_qualified=self.graph_workspace_qualified,
            comm_persistent_qualified=self.comm_persistent_qualified,
            comm_persistent_bytes=comm_value,
            comm_persistent_basis=self.comm_persistent_basis,
            workspace_bytes=self.workspace[worst_stage],
            draft_weight_qualified=self.draft_weight_qualified,
            draft_weight_bytes=self.draft_weight_bytes,
            kv_transfer_staging_bytes=self.kv_transfer_staging,
            kv_transfer_fields=self.kv_transfer_fields,
            kv_transfer_page_owner_ranks=self.kv_transfer_page_owner_ranks,
            per_rank_comm_bytes=tuple(self.comm_rank),
            per_rank_capacity_bytes=tuple(self.capacity_rank),
        )


@typed_inputs
def build_memory_ledger(
    case: CaseSpec | Mapping[str, Any], *, sequence_tokens: int | None = None,
    workspace_tokens: int | None = None,
    hccl_buffsize_mb: float | None = None, mc2_buffer_multiplier: float = 1.0,
    enable_eplb: bool | None = None, eplb_redundant_experts: int | None = None,
    eplb_redundant_slots_per_rank: Sequence[int] | None = None,
    eplb_slots_per_rank: Sequence[int] | None = None,
    eplb_rebalance_staging_bytes_per_rank: Sequence[float] | None = None,
    kv_transfer_staging_bytes: float = 0.0,
    kv_transfer_staging_per_stage_bytes: Sequence[float] | None = None,
    kv_transfer_rank_bytes: Sequence[float] = (),
    kv_transfer_fields: Mapping[str, float] | None = None,
    kv_transfer_page_owner_ranks: Sequence[int] | None = None,
    graph_buckets: Sequence[int] | None = None,
    graph_activation_bytes_per_token: float | None = None,
    graph_workspace_bytes: float | None = None,
    graph_pool_mode: str | None = None,
    graph_analytical_bound: bool = False,
    graph_operator_backing_pool: str | None = None,
    fragmentation_margin_bytes: float = 0.0,
    fragmentation_margin_ratio: float = 0.0,
    moe_recipe: str | None = None,
    arena_sharing: Mapping[str, str] | None = None,
    hccl_buffsize_bytes: int | None = None,
    comm_arena_qualified: bool = False,
    comm_arena_domains: Mapping[str, float] | None = None,
    strict_memory_qualification: bool = False,
    resolved_draft: Any = None,
) -> MemoryLedger:
    """Prepare a reusable ledger at one token span; ``at`` varies the batch."""
    from mindie_sim.parallel_placement import build_placement
    from mindie_sim.prediction.structure import parse_case
    parsed = parse_case(case)
    _validate_nested(parsed, "case")
    p = parsed.runtime.parallel
    pp, world = p.pp, int(p.world_size)
    valid_slot_lens = {max(1, int(p.ep)), world}
    for name, values in (
        ("eplb_slots_per_rank", eplb_slots_per_rank),
        ("eplb_redundant_slots_per_rank", eplb_redundant_slots_per_rank),
        ("graph_buckets", graph_buckets),
    ):
        if values is not None:
            for value in values:
                strict_int(name, value, minimum=0)
            if name != "graph_buckets" and len(values) not in valid_slot_lens:
                raise ModelingError(f"memory_breakdown_{name}_length_invalid:{len(values)}")
    plan = build_placement(parsed, p)
    meta = dict(parsed.runtime.metadata)
    pool_mode = (
        graph_pool_mode
        if graph_pool_mode is not None
        else meta.get("graph_pool_mode", "shared" if meta.get("shared_graph_pool") else "independent")
    )
    if type(pool_mode) is not str or pool_mode not in {"independent", "shared"}:
        raise ModelingError(f"memory_breakdown_graph_pool_mode_invalid:{pool_mode}")
    if not parsed.workload.requests:
        raise ModelingError("memory_breakdown_workload_requests_empty")
    if sequence_tokens is None:
        request = parsed.workload.requests[0]
        sequence_tokens = min(parsed.runtime.max_model_len, request.prompt_tokens + request.output_tokens)
    tokens = _resolve_int("sequence_tokens", sequence_tokens, parsed.runtime.max_model_len)
    if tokens < 1:
        raise ModelingError("memory_breakdown_requires_positive_tokens_and_sequences")
    ws_tokens = _resolve_int("workspace_tokens", workspace_tokens, int(parsed.runtime.max_num_batched_tokens))
    margin = _non_negative("fragmentation_margin_bytes", fragmentation_margin_bytes)
    ratio = _non_negative("fragmentation_margin_ratio", fragmentation_margin_ratio)
    if ratio > 1.0:
        raise ModelingError(f"memory_breakdown_fragmentation_margin_ratio_invalid:{ratio}")
    multiplier = _non_negative("mc2_buffer_multiplier", mc2_buffer_multiplier)
    policy = parsed.runtime.memory
    mem = Memory(parsed, workspace_tokens=ws_tokens)
    cache_mem = mem
    if parsed.model.auxiliary is None:
        from mindie_sim.xpyd_draft import resolve_draft_geometry
        if resolved_draft is None:
            resolved_draft, reason = resolve_draft_geometry(parsed)
            if reason is not None:
                raise ModelingError(reason)
        if resolved_draft is not None:
            cache_mem = copy(mem)
            cache_mem.draft_kind = mem.draft_kind or "mtp"
            cache_mem.case = replace(
                parsed,
                model=replace(parsed.model, auxiliary=AuxiliarySpec(**{mem.draft_kind or "mtp": resolved_draft})),
            )
    base, _base_cat = _stage_weights(parsed, mem, include_draft=False)
    draft, _draft_detail = _draft_bytes(parsed, mem)
    raw_redundant = meta.get("num_redundant_experts", 0) if eplb_redundant_experts is None else eplb_redundant_experts
    raw_slots = eplb_redundant_slots_per_rank if eplb_redundant_slots_per_rank is not None else eplb_slots_per_rank
    ep_coords = tuple(int(row[5]) for row in plan.rank_map)
    epoch = resolve_eplb_epoch_config(
        enable_eplb=enable_eplb, redundant_experts=raw_redundant,
        slot_vector=raw_slots, ep=max(1, int(p.ep)), world=world,
        num_experts=int(parsed.model.num_experts or 0), ep_coords=ep_coords,
        top_k=max(1, int(parsed.model.top_k or 1)),
    )
    eplb_on, redundant, slots = epoch.enable_eplb, epoch.redundant, epoch.slots_rank
    if redundant == 0 and slots is not None:
        slots = None
    staging_values = (
        tuple(_non_negative("eplb_staging", value) for value in eplb_rebalance_staging_bytes_per_rank)
        if eplb_rebalance_staging_bytes_per_rank is not None
        else None
    )
    if staging_values is not None and len(staging_values) != world:
        raise ModelingError("memory_breakdown_eplb_staging_length_invalid")
    eplb, eplb_rank, _eplb_detail = _eplb_extra_bytes(
        parsed, mem, redundant, slots_per_rank=slots,
        staging_bytes_per_rank=(staging_values if eplb_on else None),
        rank_stage=tuple(int(row[0]) for row in plan.rank_map),
    )
    if isinstance(comm_arena_qualified, bool) is False:
        raise ModelingError(f"memory_breakdown_comm_arena_qualified_invalid:{comm_arena_qualified}")
    explicit_bytes = None
    if hccl_buffsize_bytes is not None:
        raw_bytes = _non_negative("hccl_buffsize_bytes", hccl_buffsize_bytes)
        if not float(raw_bytes).is_integer():
            raise ModelingError(f"memory_breakdown_hccl_buffsize_bytes_invalid:{hccl_buffsize_bytes}")
        explicit_bytes = int(raw_bytes)
        if hccl_buffsize_mb is not None and abs(explicit_bytes / MIB - float(hccl_buffsize_mb)) > 1e-9:
            raise ModelingError(f"memory_breakdown_hccl_buffsize_conflict:{explicit_bytes}!={hccl_buffsize_mb}MiB")
    domains: dict[str, float] = {}
    if comm_arena_domains is not None:
        if not isinstance(comm_arena_domains, Mapping):
            raise ModelingError(f"memory_breakdown_comm_arena_domains_invalid:{type(comm_arena_domains).__name__}")
        for name, value in comm_arena_domains.items():
            domains[str(name)] = _non_negative("comm_arena_domain", value)
    if explicit_bytes is not None:
        buffsize = float(explicit_bytes) / MIB
    else:
        buffsize = (
            float(meta.get("hccl_buffsize_mb", 200.0))
            if hccl_buffsize_mb is None
            else _non_negative("hccl_buffsize_mb", hccl_buffsize_mb)
        )
    recipe = str(moe_recipe if moe_recipe is not None else parsed.runtime.moe_recipe)
    if recipe.strip().lower() not in {"alltoall", "mc2", "tp_join_reduce", "allgather_reducescatter"}:
        raise ModelingError(f"memory_breakdown_moe_recipe_unsupported:{recipe}")
    comm_bytes, comm_rank, comm_detail, applied_domains, heuristic_arenas = _communicator_groups(
        parsed, plan, buffsize, multiplier, recipe, arena_sharing, domains
    )
    unused_domains = set(domains) - applied_domains
    if unused_domains:
        raise ModelingError(f"memory_breakdown_comm_arena_domain_unused:{tuple(sorted(unused_domains))}")
    comm_persistent_qualified = bool(
        explicit_bytes is not None or (comm_arena_qualified and bool(domains) and not heuristic_arenas)
    )
    comm_persistent_basis = (
        "versioned_hccl_buffsize_bytes" if explicit_bytes is not None
        else "explicit_comm_arena_domains" if comm_persistent_qualified
        else "heuristic_arena_sizing"
    )
    explicit_comm = _non_negative("communication_buffer_bytes", float(policy.get("communication_buffer_bytes", 0.0) or 0.0))
    if explicit_comm > 0:
        comm_rank = tuple(value + explicit_comm for value in comm_rank)
        comm_bytes = float(max(comm_rank, default=0.0))
        comm_detail = dict(comm_detail)
        comm_detail["explicit_communication_reserve_bytes"] = explicit_comm
    graph_bytes, activation_unqualified, workspace_unqualified, graph_detail = _graph_pool(
        parsed, buckets=graph_buckets, activation_per_token=graph_activation_bytes_per_token,
        fixed_bytes=graph_workspace_bytes, pool_mode=pool_mode, analytical_bound=graph_analytical_bound,
    )
    explicit_graph = _non_negative("graph_pool_bytes", policy.get("graph_pool_bytes", 0.0))
    if explicit_graph > 0:
        graph_detail = dict(graph_detail)
        graph_detail["explicit_graph_pool_bytes"] = explicit_graph
        workspace_unqualified = 0.0
    graph_shared = pool_mode == "shared"
    backing_pool = str(graph_operator_backing_pool or "").strip()
    stage_dies: list[list[Device]] = [[] for _ in range(pp)]
    capacity_rank: list[float] = []
    for rank in range(world):
        stage = int(plan.rank_map[rank][0])
        if rank >= len(parsed.machine.dies):
            raise ModelingError(f"memory_rank_plan_exceeds_devices:{rank + 1}>{len(parsed.machine.dies)}")
        die = parsed.machine.dies[rank]
        stage_dies[stage].append(die)
        capacity_rank.append(die.hbm_bytes * float(parsed.runtime.gpu_memory_utilization))
    stage_capacity = [
        min((die.hbm_bytes * float(parsed.runtime.gpu_memory_utilization) for die in stage_dies[stage]), default=0.0)
        for stage in range(pp)
    ]
    rank_layout: list[list[list[Device]]] = []
    for replica in range(int(p.dp)):
        row: list[list[Device]] = [[] for _ in range(pp)]
        for rank, coordinate in enumerate(plan.rank_map):
            if int(coordinate[1]) == replica:
                row[int(coordinate[0])].append(parsed.machine.dies[rank])
        rank_layout.append(row)
    margin_value = margin + ratio * min(stage_capacity, default=0.0)
    workspace = [float(mem.workspace[stage]) for stage in range(pp)]
    base_ws = _non_negative("workspace_bytes", float(policy.get("workspace_bytes", 0.0) or 0.0))
    runtime_reserve = _non_negative("runtime_bytes", float(policy.get("runtime_bytes", 0.0) or 0.0))
    temp = [float(mem.temporary_dequant[stage]) for stage in range(pp)]
    if graph_shared and backing_pool:
        graph_bytes = max(graph_bytes, max(workspace, default=0.0))
        transient_base = [base_ws + runtime_reserve + temp[stage] for stage in range(pp)]
    else:
        transient_base = [base_ws + runtime_reserve + temp[stage] + workspace[stage] for stage in range(pp)]
    if explicit_graph > 0:
        graph_bytes = float(graph_bytes) + explicit_graph
    transfer_staging = _non_negative("kv_transfer_staging_bytes", kv_transfer_staging_bytes)
    transfer_fields = {str(key): float(value) for key, value in dict(kv_transfer_fields or {}).items()}
    eplb_peak = _non_negative("kv_transfer_eplb_peak", transfer_fields.get("eplb_rebalance_peak_bytes", 0.0))
    cache_one = [float(value) for value in cache_mem.cache_bytes(tokens, padded=True)]
    transfer_rank = tuple(_non_negative("kv_transfer_rank_bytes", v) for v in kv_transfer_rank_bytes)
    if transfer_rank and len(transfer_rank) != world:
        raise ModelingError("memory_transfer_rank_shape_invalid")
    if kv_transfer_staging_per_stage_bytes is not None:
        per_stage = tuple(
            _non_negative("kv_transfer_staging_stage_bytes", value)
            for value in kv_transfer_staging_per_stage_bytes
        )
        if len(per_stage) != pp:
            raise ModelingError(f"memory_breakdown_kv_transfer_staging_stage_count:{len(per_stage)}!={pp}")
        for stage in range(pp):
            transient_base[stage] += per_stage[stage]
        transfer_staging = sum(per_stage)
    else:
        kv_staging = max(0.0, transfer_staging - eplb_peak)
        if kv_staging and not transfer_rank:
            weights = [max(0.0, float(value)) for value in cache_one]
            total_weight = float(sum(weights))
            if total_weight <= 0.0:
                weights, total_weight = [1.0] * pp, float(pp)
            for stage in range(pp):
                transient_base[stage] += kv_staging * weights[stage] / total_weight
        if eplb_peak > 0.0 and not transfer_rank:
            for stage in range(pp):
                transient_base[stage] += eplb_peak
    detail_stages = _cache_detail(parsed, mem, tokens, 1)
    if cache_mem is not mem:
        detail_stages[-1]["draft_kv"] = cache_mem._cache_component_breakdown(
            tokens, padded=True, skip_covered=True
        )[-1]["components"]["draft_kv"]["total"]
    sharded = [max(0.0, base[i] - float(mem.expert_hbm_saved[i])) for i in range(pp)]
    aux = [draft[i] + eplb[i] for i in range(pp)]
    aux_rank = tuple(
        tuple(float(draft[stage]) + float(eplb_rank[rank][stage]) for stage in range(pp))
        for rank in range(world)
    )
    draft_total = float(sum(draft))
    shared_draft = bool(
        meta.get("mtp_shared_weights") or meta.get("draft_shared_weights") or meta.get("eagle_shared_weights")
    )
    declared_shared_cost = float(meta.get("mtp_draft_layer_cost_ms", meta.get("mtp_draft_cost_ms", 0.0)) or 0.0) > 0.0
    from mindie_sim.xpyd_draft import shared_target_draft
    draft_qualified = bool(
        draft_total > 0 or not mem.draft_kind
        or (shared_draft and (declared_shared_cost or shared_target_draft(parsed.model) is not None))
    )
    return MemoryLedger(
        parsed, mem, pp, sharded, aux, comm_bytes, tuple(sorted(comm_detail.items())),
        float(graph_bytes), float(max(activation_unqualified, workspace_unqualified)),
        tuple(sorted(graph_detail.items())), graph_shared, transient_base, workspace,
        float(parsed.model.vocab_size * dtype_bytes(parsed.model.activation_dtype)),
        cache_one, detail_stages, stage_capacity, stage_dies, rank_layout, [],
        runtime_reserve, margin_value,
        (
            ("mc2_buffer_multiplier", multiplier),
            ("graph_pool_mode_shared", float(graph_shared)),
            ("graph_operator_backing_pool", 1.0 if backing_pool else 0.0),
        ),
        rank_map=tuple(plan.rank_map), comm_rank=tuple(comm_rank),
        aux_rank=aux_rank, graph_rank=(float(graph_bytes),) * world,
        capacity_rank=tuple(capacity_rank),
        activation_peak_qualified=(activation_unqualified == 0.0),
        graph_workspace_qualified=(workspace_unqualified == 0.0),
        draft_weight_qualified=draft_qualified, draft_weight_bytes=draft_total,
        kv_transfer_staging=transfer_staging,
        kv_transfer_fields=tuple(sorted(transfer_fields.items())),
        kv_transfer_page_owner_ranks=tuple(int(rank) for rank in (kv_transfer_page_owner_ranks or ())),
        comm_persistent_qualified=comm_persistent_qualified,
        comm_persistent_basis=comm_persistent_basis,
        strict_comm_qualification=bool(strict_memory_qualification),
        transfer_rank=transfer_rank, cache_mem=cache_mem,
    )


@typed_inputs
def analyze_memory_breakdown(
    case: CaseSpec | Mapping[str, Any], *, sequence_tokens: int | None = None,
    sequences: int | None = None, workspace_tokens: int | None = None,
    hccl_buffsize_mb: float | None = None, mc2_buffer_multiplier: float = 1.0,
    enable_eplb: bool | None = None, eplb_redundant_experts: int | None = None,
    eplb_redundant_slots_per_rank: Sequence[int] | None = None,
    eplb_slots_per_rank: Sequence[int] | None = None,
    eplb_rebalance_staging_bytes_per_rank: Sequence[float] | None = None,
    kv_transfer_staging_bytes: float = 0.0,
    kv_transfer_staging_per_stage_bytes: Sequence[float] | None = None,
    kv_transfer_rank_bytes: Sequence[float] = (),
    kv_transfer_fields: Mapping[str, float] | None = None,
    kv_transfer_page_owner_ranks: Sequence[int] | None = None,
    graph_buckets: Sequence[int] | None = None,
    graph_activation_bytes_per_token: float | None = None,
    graph_workspace_bytes: float | None = None,
    graph_pool_mode: str | None = None,
    graph_analytical_bound: bool = False,
    graph_operator_backing_pool: str | None = None,
    fragmentation_margin_bytes: float = 0.0,
    fragmentation_margin_ratio: float = 0.0,
    moe_recipe: str | None = None,
    arena_sharing: Mapping[str, str] | None = None,
    hccl_buffsize_bytes: int | None = None,
    comm_arena_qualified: bool = False,
    comm_arena_domains: Mapping[str, float] | None = None,
    strict_memory_qualification: bool = False,
) -> MemoryBreakdown:
    """Compute the six-component per-rank ledger at the exact batch."""
    from mindie_sim.prediction.structure import parse_case
    parsed = parse_case(case)
    _validate_nested(parsed, "case")
    if not parsed.workload.requests:
        raise ModelingError("memory_breakdown_workload_requests_empty")
    seqs = _resolve_int("sequences", sequences, int(parsed.runtime.max_num_seqs))
    if seqs < 1:
        raise ModelingError("memory_breakdown_requires_positive_tokens_and_sequences")
    ledger = build_memory_ledger(
        parsed, sequence_tokens=sequence_tokens, workspace_tokens=workspace_tokens,
        hccl_buffsize_mb=hccl_buffsize_mb, mc2_buffer_multiplier=mc2_buffer_multiplier,
        enable_eplb=enable_eplb, eplb_redundant_experts=eplb_redundant_experts,
        eplb_redundant_slots_per_rank=eplb_redundant_slots_per_rank,
        eplb_slots_per_rank=eplb_slots_per_rank,
        eplb_rebalance_staging_bytes_per_rank=eplb_rebalance_staging_bytes_per_rank,
        kv_transfer_staging_bytes=kv_transfer_staging_bytes,
        kv_transfer_staging_per_stage_bytes=kv_transfer_staging_per_stage_bytes,
        kv_transfer_rank_bytes=kv_transfer_rank_bytes,
        kv_transfer_fields=kv_transfer_fields,
        kv_transfer_page_owner_ranks=kv_transfer_page_owner_ranks,
        graph_buckets=graph_buckets,
        graph_activation_bytes_per_token=graph_activation_bytes_per_token,
        graph_workspace_bytes=graph_workspace_bytes, graph_pool_mode=graph_pool_mode,
        graph_analytical_bound=graph_analytical_bound,
        graph_operator_backing_pool=graph_operator_backing_pool,
        fragmentation_margin_bytes=fragmentation_margin_bytes,
        fragmentation_margin_ratio=fragmentation_margin_ratio,
        moe_recipe=moe_recipe, arena_sharing=arena_sharing,
        hccl_buffsize_bytes=hccl_buffsize_bytes,
        comm_arena_qualified=comm_arena_qualified,
        comm_arena_domains=comm_arena_domains,
        strict_memory_qualification=strict_memory_qualification,
    )
    if sequence_tokens is None:
        request = parsed.workload.requests[0]
        tokens = min(parsed.runtime.max_model_len, request.prompt_tokens + request.output_tokens)
    else:
        tokens = _resolve_int("sequence_tokens", sequence_tokens, parsed.runtime.max_model_len)
    return ledger.at(seqs, tokens)


@typed_inputs
def memory_ledger(case: CaseSpec | Mapping[str, Any], **kwargs: Any) -> MemoryLedger:
    """Prepare a reusable ledger (public form used by xPyD batch search)."""
    return build_memory_ledger(case, **kwargs)


@typed_inputs
def memory_breakdown(case: CaseSpec | Mapping[str, Any], **kwargs: Any) -> MemoryBreakdown:
    """Short alias used by the parallel explorer and agent tools."""
    return analyze_memory_breakdown(case, **kwargs)


@typed_inputs
def analyze_hybrid_state_breakdown(
    case: CaseSpec | Mapping[str, Any], *, state_pool_bytes: float,
    per_stage_state_bytes: Sequence[float] | None = None, **kwargs: Any,
) -> MemoryBreakdown:
    """Baseline six-component ledger plus a modeled hybrid-state pool delta."""
    base = analyze_memory_breakdown(case, **kwargs)
    return apply_state_pool_delta(base, delta_bytes=state_pool_bytes, per_stage_bytes=per_stage_state_bytes)


@typed_inputs
def apply_state_pool_delta(*args: Any, **kwargs: Any) -> MemoryBreakdown:
    """Late re-export; the delta module owns the post-delta feasibility math."""
    from mindie_sim.memory_state_delta import apply_state_pool_delta as impl
    return impl(*args, **kwargs)


__all__ = [
    "COMPONENT_NAMES", "MemoryBreakdown", "MemoryComponent", "MemoryLedger",
    "analyze_hybrid_state_breakdown", "analyze_memory_breakdown",
    "apply_state_pool_delta", "build_memory_ledger", "memory_breakdown", "memory_ledger",
]
