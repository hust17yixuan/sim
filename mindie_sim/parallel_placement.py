"""Homogeneous placement normalization for one parallel plan.

[Role]
  Turn a ``ParallelSpec`` into the concrete rank-coordinate plan used by the
  memory ledger, the EP knee and xPyD solver: ``W = PP*PCP*DP*TP``, EP confined
  to each ``(pp, pcp_lane)`` plane, explicit expert TP/DP factorization, DCP as
  a TP subdivision, and an explicit PCP lane policy.

[Invariants]
  * ``EP * expert_tp * expert_dp == DP * TP`` inside every lane plane.  An EP
    group that would span PCP lanes is refused unless the caller supplies an
    explicit rank/group table.
  * Every physical rank has one coordinate tuple ``(pp, pcp_lane, dp, tp)`` and
    no two owners share it.  Fabricated dies are never created to satisfy a
    candidate; a world larger than the declared device list is unsupported.
  * DCP adds no world axis; ``DCP <= TP`` and ``TP % DCP == 0``.
  * ``park_extra_lanes`` advances decode on one lane only; ``replica_lanes``
    activates every lane and must pay an explicit KV/index/state remap.
  * The PP layer partition is normalized to the candidate PP before the base
    schema validates it, so a valid PP=2 plan is not judged by a stale PP=1
    partition.

[Timeouts & Locks] Pure arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Returns fresh immutable records; caller Cases untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec


def normalize_partition(case: CaseSpec, pp: int) -> tuple[int, ...]:
    """Return a complete candidate PP split, preserving a valid declared one."""
    declared = tuple(int(value) for value in case.runtime.pp_layer_partition)
    if len(declared) == pp and sum(declared) == case.model.layers and all(declared):
        return declared
    layers = int(case.model.layers)
    if layers < pp:
        raise ModelingError(f"pp_exceeds_layers:{pp}>{layers}")
    return tuple(layers // pp + (1 if index < layers % pp else 0) for index in range(pp))


def with_world(case: CaseSpec, parallel: ParallelSpec) -> CaseSpec:
    """Bind the plan to the declared physical endpoints, never fabricate."""
    if parallel.world_size > len(case.machine.dies):
        raise ModelingError(
            f"placement_exceeds_declared_devices:{parallel.world_size}>{len(case.machine.dies)}")
    return replace(case, runtime=replace(
        case.runtime, parallel=parallel,
        pp_layer_partition=normalize_partition(case, parallel.pp)))


def validate_placement(case: CaseSpec, parallel: ParallelSpec
                       ) -> tuple[CaseSpec | None, dict[str, Any] | None, str | None]:
    """Normalize then validate; return ``(case, plan, reason)``."""
    try:
        candidate = with_world(case, parallel)
        parallel.validate(case.model, layer_partition=candidate.runtime.pp_layer_partition or None)
        plan = build_placement(candidate, parallel)
        return candidate, plan.to_dict(), None
    except (ModelingError, ValueError) as exc:
        return None, None, str(exc)


@dataclass(frozen=True)
class PlacementPlan:
    pp: int
    pcp: int
    dp: int
    tp: int
    ep: int
    dcp: int
    expert_tp: int
    expert_dp: int
    pcp_lane_policy: str
    world: int
    lane_plane: int
    active_lanes: int
    parked_lanes: int
    active_replicas: int
    rank_table: tuple[tuple[int, ...], ...] = ()
    ep_groups: tuple[tuple[int, ...], ...] = ()
    etp_groups: tuple[tuple[int, ...], ...] = ()
    edp_groups: tuple[tuple[int, ...], ...] = ()
    tp_groups: tuple[tuple[int, ...], ...] = ()
    dcp_groups: tuple[tuple[int, ...], ...] = ()
    pcp_groups: tuple[tuple[int, ...], ...] = ()
    pp_groups: tuple[tuple[int, ...], ...] = ()
    dp_groups: tuple[tuple[int, ...], ...] = ()
    #: ``rank -> (pp_stage, dp_idx, pcp_idx, dcp_idx, tp_idx, ep_idx)``.
    rank_map: tuple[tuple[int, int, int, int, int, int], ...] = ()
    #: ``stage -> expert -> ep_idx``; every expert has exactly one owner.
    expert_ownership: tuple[tuple[int, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"pp": self.pp, "pcp": self.pcp, "dp": self.dp, "tp": self.tp,
                "ep": self.ep, "dcp": self.dcp, "expert_tp": self.expert_tp,
                "expert_dp": self.expert_dp,
                "pcp_lane_policy": self.pcp_lane_policy,
                "world": self.world, "lane_plane": self.lane_plane,
                "active_lanes": self.active_lanes, "parked_lanes": self.parked_lanes,
                "active_replicas": self.active_replicas,
                "rank_table": [list(group) for group in self.rank_table],
                "ep_groups": [list(group) for group in self.ep_groups],
                "etp_groups": [list(group) for group in self.etp_groups],
                "edp_groups": [list(group) for group in self.edp_groups],
                "tp_groups": [list(group) for group in self.tp_groups],
                "dcp_groups": [list(group) for group in self.dcp_groups],
                "pcp_groups": [list(group) for group in self.pcp_groups],
                "pp_groups": [list(group) for group in self.pp_groups],
                "dp_groups": [list(group) for group in self.dp_groups],
                "rank_map": [list(row) for row in self.rank_map],
                "expert_ownership": [list(row) for row in self.expert_ownership]}


def _coordinate(pp: int, lane: int, dp: int, tp: int, p: ParallelSpec) -> int:
    return ((pp * p.pcp + lane) * p.dp + dp) * p.tp + tp


def expert_owner_slot(expert: int, experts: int, ep: int) -> int:
    """Canonical contiguous expert ownership: ``expert // (E // EP)``.

    The placement, the memory ledger and ``place_eplb`` all import this one
    function, so a captured routing schedule can never be compared against a
    different ownership map.  When ``E`` is not divisible by ``EP`` the same
    contiguous partition is expressed as ``floor(expert*EP/E)``.
    """
    if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
        raise ModelingError(f"expert_owner_expert_invalid:{expert}")
    if isinstance(experts, bool) or not isinstance(experts, int) or experts < 1:
        raise ModelingError(f"expert_owner_count_invalid:{experts}")
    if isinstance(ep, bool) or not isinstance(ep, int) or ep < 1:
        raise ModelingError(f"expert_owner_ep_invalid:{ep}")
    if expert >= experts:
        raise ModelingError(f"expert_owner_expert_out_of_range:{expert}>={experts}")
    if experts % ep == 0:
        return expert // (experts // ep)
    return min(ep - 1, (expert * ep) // experts)


def _lane_factorization(p: ParallelSpec) -> tuple[int, int, int]:
    """Return ``(EP, ETP, EDP)`` with ``EP*ETP*EDP == DP*TP``.

    The lane plane is decomposed orthogonally: ``etp`` varies fastest (each
    expert is sharded across a contiguous ETP group), ``ep`` is the next axis
    (an EP group varies across shards), and ``expert_dp`` is the outer axis.
    With ``TP=4, DP=1, EP=2, ETP=2`` this yields ETP groups ``(0,1),(2,3)``
    and EP groups ``(0,2),(1,3)`` rather than two identical partitions.
    """
    lane_plane = max(1, p.dp * p.tp)
    ep = max(1, int(p.ep))
    etp = max(1, int(p.etp))
    if lane_plane % (ep * etp) != 0:
        raise ModelingError(
            f"ep_expert_factorization_invalid:{ep}*{etp}!|{lane_plane}")
    return ep, etp, lane_plane // (ep * etp)


def rank_parts(rank: int, p: ParallelSpec) -> tuple[int, int, int, int, int, int]:
    """Physical rank -> ``(pp, dp, pcp, dcp, tp, ep)`` for one homogeneous plan.

    The flattening order is exactly ``((pp*PCP + pcp)*DP + dp)*TP + tp``.
    Inside a lane plane the orthogonal order is ``lane = (edp*EP + ep)*ETP+etp``;
    the returned EP coordinate is therefore the owner slot used by
    ``expert_owner_slot`` and never a constant for ``TP=1``.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ModelingError(f"rank_coordinate_invalid:{rank}")
    plane = max(1, p.dp * p.tp)
    stage_plane = max(1, p.pcp * plane)
    pp, remainder = divmod(int(rank), stage_plane)
    pcp, remainder = divmod(remainder, plane)
    dp, tp = divmod(remainder, p.tp)
    dcp_width = max(1, p.tp // max(1, p.dcp))
    dcp = tp // dcp_width if p.dcp > 1 else 0
    ep, etp, _edp = _lane_factorization(p)
    lane = dp * p.tp + tp
    ep_idx = (lane // etp) % ep
    return (pp, dp, pcp, dcp, tp, ep_idx)


def canonical_coordinate(rank: int, p: ParallelSpec
                         ) -> tuple[int, int, int, int, int, int]:
    """Backward-compatible public alias of :func:`rank_parts`."""
    return rank_parts(rank, p)


def rank_placement_map(plan: PlacementPlan) -> dict[int, tuple[int, int, int, int, int, int]]:
    """Public ``rank -> (pp, dp, pcp, dcp, tp, ep)`` view of one plan."""
    if not plan.rank_map or len(plan.rank_map) != plan.world:
        raise ModelingError("placement_rank_map_incomplete")
    return {rank: tuple(plan.rank_map[rank]) for rank in range(plan.world)}


def _expert_ownership(case: CaseSpec, p: ParallelSpec, ep_groups: Sequence[Sequence[int]],
                      stage_of: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Prove ``expert -> ep_idx`` is the canonical contiguous bijection."""
    experts = int(case.model.num_experts or 0)
    if experts <= 0:
        return ()
    if experts % max(1, p.ep) != 0:
        raise ModelingError(f"experts_not_divisible_by_ep:{experts}%{p.ep}")
    result: list[tuple[int, ...]] = []
    for stage in range(p.pp):
        groups = [tuple(group) for group in ep_groups
                  if len(group) > 0 and stage_of[group[0]] == stage]
        if not groups:
            raise ModelingError(f"expert_ownership_missing_stage:{stage}")
        group = groups[0]
        if len(group) != p.ep:
            raise ModelingError(f"expert_ownership_group_width:{len(group)}!={p.ep}")
        owners = [expert_owner_slot(expert, experts, p.ep)
                  for expert in range(experts)]
        if set(owners) != set(range(p.ep)):
            raise ModelingError(f"expert_ownership_incomplete:{stage}")
        seen = [0] * p.ep
        for owner in owners:
            seen[owner] += 1
        per_owner = experts // p.ep
        if any(count != per_owner for count in seen):
            raise ModelingError(f"expert_ownership_imbalanced:{tuple(seen)}")
        result.append(tuple(owners))
    return tuple(result)


def _explicit_rank_table(
    p: ParallelSpec, ep_groups: tuple[tuple[int, ...], ...],
    expert_tp: int, expert_dp: int,
) -> tuple[tuple[tuple[int, int, int, int], ...], list[tuple[int, ...]],
           list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Factor an explicit EP table into orthogonal stage/EDP/EP/ETP axes."""
    world = int(p.world_size)
    stage_of = [canonical_coordinate(rank, p)[0] for rank in range(world)]
    coordinates: dict[int, tuple[int, int, int, int]] = {}
    etp_groups: list[tuple[int, ...]] = []
    edp_groups: list[tuple[int, ...]] = []
    per_stage = max(1, int(p.pcp)) * int(expert_dp) * int(p.ep) * int(expert_tp)
    for stage in range(int(p.pp)):
        expected = {rank for rank in range(world) if stage_of[rank] == stage}
        if len(expected) != per_stage:
            raise ModelingError("explicit_rank_table_non_orthogonal")
        stage_rows = []
        for row in ep_groups:
            stages = {stage_of[int(rank)] for rank in row}
            if stages == {stage}:
                stage_rows.append(row)
            elif stage in stages:
                raise ModelingError(
                    f"rank_table_group_crosses_pp_stage:{tuple(sorted(stages))}")
        if len(stage_rows) != max(1, int(p.pcp)) * int(expert_dp) * int(expert_tp):
            raise ModelingError("explicit_rank_table_non_orthogonal")
        covered: set[int] = set()
        stage_coords: dict[int, tuple[int, int, int, int]] = {}
        lane_of: dict[int, int] = {}
        for position, row in enumerate(stage_rows):
            if len(row) != int(p.ep):
                raise ModelingError("explicit_rank_table_non_orthogonal")
            lane, remainder = divmod(position, int(expert_dp) * int(expert_tp))
            edp_coord, etp_coord = divmod(remainder, int(expert_tp))
            for ep_coord, rank in enumerate(row):
                value = int(rank)
                if value in covered or value not in expected:
                    raise ModelingError("explicit_rank_table_non_orthogonal")
                covered.add(value)
                stage_coords[value] = (stage, edp_coord, ep_coord, etp_coord)
                lane_of[value] = lane
        if covered != expected:
            raise ModelingError("explicit_rank_table_non_orthogonal")
        # Group construction is stage- and lane-local: iterating every
        # accumulated stage coordinate re-appended earlier stages, and a key
        # without the PCP lane merged independent lane planes.
        for key in sorted({(lane_of[rank], c[1], c[2])
                           for rank, c in stage_coords.items()}):
            members = sorted((c[3], rank) for rank, c in stage_coords.items()
                             if (lane_of[rank], c[1], c[2]) == key)
            etp_groups.append(tuple(rank for _coord, rank in members))
        for key in sorted({(lane_of[rank], c[2], c[3])
                           for rank, c in stage_coords.items()}):
            members = sorted((c[1], rank) for rank, c in stage_coords.items()
                             if (lane_of[rank], c[2], c[3]) == key)
            if len(members) > 1:
                edp_groups.append(tuple(rank for _coord, rank in members))
        coordinates.update(stage_coords)
    if len(coordinates) != world:
        raise ModelingError("explicit_rank_table_non_orthogonal")
    ordered = tuple(coordinates[rank] for rank in range(world))
    # Pairwise orthogonality: one EP group and one ETP/EDP group share at most
    # one physical rank.  A table whose ETP group spans two EP owners is not a
    # tensor-shard of one owner.
    for left in ep_groups:
        for right in etp_groups:
            if len(set(left) & set(right)) > 1:
                raise ModelingError("explicit_rank_table_non_orthogonal")
        for right in edp_groups:
            if len(set(left) & set(right)) > 1:
                raise ModelingError("explicit_rank_table_non_orthogonal")
    for left in etp_groups:
        for right in edp_groups:
            if len(set(left) & set(right)) > 1:
                raise ModelingError("explicit_rank_table_non_orthogonal")
    return ordered, etp_groups, edp_groups


def _assert_axis_partition(name: str, groups: Sequence[Sequence[int]],
                           world: int, width: int) -> None:
    """Every active axis partitions ``0..world-1`` exactly once."""
    if int(width) <= 1:
        return
    flat = [int(rank) for group in groups for rank in group]
    if len(flat) != int(world) or set(flat) != set(range(int(world))):
        raise ModelingError(
            f"placement_axis_partition:{name}:{len(set(flat))}!={world}")
    if len(set(flat)) != len(flat):
        raise ModelingError(f"placement_axis_duplicate_membership:{name}")


def build_placement(case: CaseSpec, parallel: ParallelSpec) -> PlacementPlan:
    """Construct explicit rank groups and prove the homogeneous factorization."""
    p = parallel
    explicit_coords = None
    if p.rank_table is not None:
        flat = tuple(rank for group in p.rank_table for rank in group)
        if min(flat, default=0) < 0 or max(flat, default=-1) >= p.world_size:
            raise ModelingError("rank_table_rank_out_of_world")
        if len(set(flat)) != len(flat):
            raise ModelingError("rank_table_duplicate_rank")
        rank_table = tuple(p.rank_table)
        ep_groups = tuple(tuple(int(rank) for rank in group) for group in rank_table)
        expert_tp = int(p.expert_tp or 1)
        expert_dp = int(p.expert_dp or 1)
        if int(p.ep) * expert_tp * expert_dp != p.lane_plane:
            raise ModelingError("rank_table_factorization_unproven")
        explicit_coords, etp_groups, edp_groups = _explicit_rank_table(
            p, ep_groups, expert_tp, expert_dp)
        # Attention coordinates come from the attention axes.  Expert
        # EDP/ETP coordinates never overwrite attention DP/TP, and PCP/DCP are
        # never forced to zero by an explicit expert table.
        rank_map = []
        for rank in range(p.world_size):
            attention = canonical_coordinate(rank, p)
            rank_map.append((attention[0], attention[1], attention[2],
                             attention[3], attention[4],
                             explicit_coords[rank][2]))
        rank_map = tuple(rank_map)
    else:
        if p.ep > p.lane_plane:
            raise ModelingError(f"ep_exceeds_lane_plane:{p.ep}>{p.lane_plane}")
        expert_tp = int(p.etp)
        expert_dp = int(p.expert_dp if p.expert_dp is not None
                        else p.lane_plane // (p.ep * expert_tp))
        if p.ep * expert_tp * expert_dp != p.lane_plane:
            raise ModelingError(
                f"ep_expert_factorization_invalid:{p.ep}*{expert_tp}*{expert_dp}"
                f"!={p.lane_plane}")
        rank_table = tuple(
            tuple(_coordinate(stage, lane, replica, tp, p)
                  for replica in range(p.dp) for tp in range(p.tp))
            for stage in range(p.pp) for lane in range(p.pcp))
        # Orthogonal lane decomposition: etp varies fastest, ep next, edp outer.
        ep_groups = []
        etp_groups = []
        edp_groups = []
        for stage in range(p.pp):
            for lane in range(p.pcp):
                base = _coordinate(stage, lane, 0, 0, p)
                for edp_idx in range(expert_dp):
                    for etp_idx in range(expert_tp):
                        ep_groups.append(tuple(
                            base + (edp_idx * p.ep + ep_idx) * expert_tp + etp_idx
                            for ep_idx in range(p.ep)))
                    for ep_idx in range(p.ep):
                        stride = (edp_idx * p.ep + ep_idx) * expert_tp
                        etp_groups.append(tuple(base + stride + shard
                                                for shard in range(expert_tp)))
                for ep_idx in range(p.ep):
                    for etp_idx in range(expert_tp):
                        edp_groups.append(tuple(
                            base + (edp_idx * p.ep + ep_idx) * expert_tp + etp_idx
                            for edp_idx in range(expert_dp)))
        rank_map = tuple(canonical_coordinate(rank, p) for rank in range(p.world_size))
    tp_groups = tuple(
        tuple(_coordinate(stage, lane, replica, tp, p) for tp in range(p.tp))
        for stage in range(p.pp) for lane in range(p.pcp) for replica in range(p.dp))
    dcp_groups = tuple(
        group[start:start + p.dcp]
        for group in tp_groups for start in range(0, p.tp, max(1, p.dcp))) \
        if p.dcp > 1 else ()
    pcp_groups = tuple(
        tuple(_coordinate(stage, lane, replica, tp, p) for lane in range(p.pcp))
        for stage in range(p.pp) for replica in range(p.dp) for tp in range(p.tp)) \
        if p.pcp > 1 else ()
    pp_groups = tuple(
        tuple(_coordinate(stage, lane, replica, tp, p) for stage in range(p.pp))
        for lane in range(p.pcp) for replica in range(p.dp)
        for tp in range(p.tp)) if p.pp > 1 else ()
    all_ranks = tuple(rank for group in rank_table for rank in group)
    if len(set(all_ranks)) != p.world_size:
        raise ModelingError(f"placement_rank_coverage:{len(set(all_ranks))}!={p.world_size}")
    stage_of = [canonical_coordinate(rank, p)[0] for rank in range(p.world_size)]
    for group in ep_groups:
        if p.ep > 1 and len(group) != p.ep:
            raise ModelingError(f"rank_table_group_width:{len(group)}!={p.ep}")
        stages = {stage_of[int(rank)] for rank in group}
        if len(stages) != 1:
            raise ModelingError(
                f"rank_table_group_crosses_pp_stage:{tuple(sorted(stages))}")
        if p.rank_table is None:
            lanes = {canonical_coordinate(int(rank), p)[2] for rank in group}
            if len(lanes) != 1:
                raise ModelingError("rank_table_group_crosses_pcp_lane")
    dp_groups = tuple(
        tuple(_coordinate(stage, lane, replica, tp, p)
              for replica in range(p.dp))
        for stage in range(p.pp) for lane in range(p.pcp)
        for tp in range(p.tp)) if p.dp > 1 else ()
    _assert_axis_partition("tp", tp_groups, p.world_size, p.tp)
    _assert_axis_partition("dcp", dcp_groups, p.world_size, p.dcp)
    _assert_axis_partition("pcp", pcp_groups, p.world_size, p.pcp)
    _assert_axis_partition("dp", dp_groups, p.world_size, p.dp)
    _assert_axis_partition("ep", tuple(ep_groups), p.world_size, p.ep)
    _assert_axis_partition("etp", tuple(etp_groups), p.world_size, expert_tp)
    _assert_axis_partition("edp", tuple(edp_groups), p.world_size, expert_dp)
    attention_coords = tuple(tuple(row[:5]) for row in rank_map)
    if len(set(attention_coords)) != p.world_size:
        raise ModelingError("placement_attention_coordinates_not_distinct")
    ownership = _expert_ownership(case, p, tuple(ep_groups), stage_of)
    return PlacementPlan(
        pp=p.pp, pcp=p.pcp, dp=p.dp, tp=p.tp, ep=p.ep, dcp=p.dcp,
        expert_tp=expert_tp, expert_dp=expert_dp,
        pcp_lane_policy=p.pcp_lane_policy, world=p.world_size,
        lane_plane=p.lane_plane, active_lanes=p.active_pcp_lanes,
        parked_lanes=p.parked_pcp_lanes, active_replicas=p.active_replicas,
        rank_table=rank_table, ep_groups=tuple(ep_groups),
        etp_groups=tuple(etp_groups), edp_groups=tuple(edp_groups),
        tp_groups=tp_groups,
        dcp_groups=dcp_groups, pcp_groups=pcp_groups, pp_groups=pp_groups,
        dp_groups=dp_groups, rank_map=tuple(rank_map),
        expert_ownership=ownership)

def lane_remap_bytes(kv_bytes: float, index_bytes: float, state_bytes: float,
                     plan: PlacementPlan) -> dict[str, float]:
    """Unique state a replica-lane activation must move into parked lanes."""
    kv = float(max(0.0, kv_bytes))
    index = float(max(0.0, index_bytes))
    state = float(max(0.0, state_bytes))
    targets = max(0, int(plan.pcp) - 1) if plan.pcp_lane_policy == "replica_lanes" else 0
    payload = (kv + index + state) * targets
    metadata = 4096.0 * targets
    return {"target_lanes": float(targets), "kv_bytes": kv * targets,
            "index_bytes": index * targets, "state_bytes": state * targets,
            "metadata_bytes": metadata, "total_bytes": payload + metadata}


def lane_remap_ms(remap: dict[str, float], bandwidth_gbps: float) -> float:
    """Time to move the remap payload over one declared lane link."""
    if bandwidth_gbps <= 0 or not math.isfinite(float(bandwidth_gbps)):
        raise ModelingError(f"lane_remap_bandwidth_invalid:{bandwidth_gbps}")
    return float(remap.get("total_bytes", 0.0)) / (float(bandwidth_gbps) * 1e9) * 1000.0


__all__ = ["PlacementPlan", "build_placement", "canonical_coordinate",
           "expert_owner_slot", "lane_remap_bytes", "lane_remap_ms",
           "normalize_partition", "rank_parts", "rank_placement_map",
           "validate_placement", "with_world"]
