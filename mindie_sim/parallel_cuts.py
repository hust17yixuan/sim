"""Pool-wide shared physical-cut accounting for inference and handoff traffic.

[Role]
  Enumerate every physical shared cut (declared switch or shared/mesh tier),
  compose the per-request byte demand placed on it by P/D inference
  collectives, PP activation hops, KV handoff and layout remap, and derive the
  pool-wide admitted-rate ceiling from one conserved resource identity.

[Invariants]
  * A cut is identified by its physical tier name, never by a per-instance
    capacity.  Concurrent logical instances reuse the same identity.
  * A collective's cut demand is the total physical wire it moves, not one
    sender's private share: AllReduce ``2*(g-1)*payload``, AllGather/
    ReduceScatter/AllToAllV ``(g-1)*payload`` and MC2 exactly the sum of its
    dispatch and combine AllToAllV wires, ``2*(g-1)*payload``.
  * Every participant's ingress and egress are distinct per-endpoint port
    resources; routed traffic is charged to the exact endpoint that carries it,
    never to an unweighted ``N * port_rate`` aggregate.
  * Multiple synchronous groups on the same switch aggregate their bytes in one
    step; they never each own 100% of the switch.
  * Per-request demand is independent of the number of P/D instances: one
    request traverses one P plan and one D plan.  Pool traffic is
    ``admitted_rps * demand_per_request``.
  * A PP activation hop moves its payload once on the physical route of its
    own adjacent stage endpoints; sequential collectives on one communicator
    are summed, never deduplicated or overlapped.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable dicts; caller records untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.common import ModelingError

_AXIS_KIND = (
    ("tp_groups", "all_reduce"),
    ("etp_groups", "all_reduce"),
    ("ep_groups", "all_to_all_v"),
    ("edp_groups", "all_reduce"),
    ("pcp_groups", "all_gather"),
    ("dcp_groups", "all_reduce"),
)


def collective_wire(kind: str, group: int, payload: float) -> float:
    """Total physical wire bytes a collective moves across one cut."""
    size = max(1, int(group))
    value = max(0.0, float(payload))
    if size <= 1 or value <= 0.0:
        return 0.0
    if kind == "all_reduce":
        return 2.0 * float(size - 1) * value
    if kind in {"all_gather", "reduce_scatter", "all_to_all_v"}:
        return float(size - 1) * value
    if kind == "mc2":
        # MC2 is one dispatch plus one combine: exactly the sum of the two
        # AllToAllV wire amounts, matching ``fabric_topology``'s recipe.
        return 2.0 * float(size - 1) * value
    raise ModelingError(f"parallel_cuts_collective_unknown:{kind}")


def _per_rank_wire(kind: str, group: int, payload: float) -> float:
    """Physical bytes one participant sends (and receives) per collective."""
    size = max(1, int(group))
    value = max(0.0, float(payload))
    if size <= 1 or value <= 0.0:
        return 0.0
    share = float(size - 1) / float(size)
    if kind in {"all_reduce", "mc2"}:
        return 2.0 * value * share
    if kind in {"all_gather", "reduce_scatter", "all_to_all_v"}:
        return value * share
    raise ModelingError(f"parallel_cuts_collective_unknown:{kind}")


def collective_port_demands(tree: Any, groups: Sequence[Sequence[int]],
                            kind: str, payload: float) -> dict[str, float]:
    """Per-endpoint ingress/egress bytes of every group in one call.

    Each participant uses its own full-duplex port; a fast destination can
    never borrow another endpoint's idle port.  The byte amount is the
    participant's own send and receive share, not an unweighted pool aggregate.
    """
    result: dict[str, float] = {}
    for group in groups:
        ids = tuple(int(rank) for rank in group)
        if len(ids) < 2:
            continue
        wire = _per_rank_wire(kind, len(ids), payload)
        if wire <= 0.0:
            continue
        for rank in ids:
            for direction in ("egress", "ingress"):
                key = f"port:{rank}:{direction}"
                result[key] = result.get(key, 0.0) + wire
    return result


def peer_port_demands(tree: Any, endpoints: Sequence[int],
                      payload: float) -> dict[str, float]:
    """Per-endpoint ingress/egress bytes of one point-to-point transfer."""
    del tree
    value = max(0.0, float(payload))
    if value <= 0.0:
        return {}
    result: dict[str, float] = {}
    for rank in {int(rank) for rank in endpoints}:
        for direction in ("egress", "ingress"):
            key = f"port:{rank}:{direction}"
            result[key] = result.get(key, 0.0) + value
    return result


def _tier_members(tier: Any, ids: Sequence[int]) -> bool:
    members = getattr(tier, "link_members", ())
    if not members:
        return len(ids) > 1
    return any(set(ids) <= set(group) for group in members)


def cut_inventory(tree: Any) -> dict[str, float]:
    """Every declared shared cut, per-endpoint port and capacity in bytes/s."""
    inventory: dict[str, float] = {}

    def add(key: str, capacity: float) -> None:
        if capacity <= 0.0:
            return
        existing = inventory.get(key)
        inventory[key] = (capacity if existing is None
                          else min(float(existing), float(capacity)))

    for tier in getattr(tree, "tiers", ()):
        switch = getattr(tier, "switch_capacity_bytes_per_ms", None)
        if switch is not None:
            add(f"switch:{tier.name}", float(switch) * 1e3)
        semantics = str(getattr(tier, "semantics", "shared"))
        if semantics in {"shared", "mesh"}:
            add(f"tier:{tier.name}", float(tier.bw_gbps) * 1e9)
        # Physical full-duplex ports: one independent resource per (rank,
        # direction) for every *declared* per-rank endpoint rate (bytes/ms).
        # An aggregate tier bandwidth is not evidence of an individual port.
        for rank, rate in getattr(tier, "endpoint_rates", ()):
            add(f"port:{int(rank)}:egress", float(rate) * 1e3)
            add(f"port:{int(rank)}:ingress", float(rate) * 1e3)
        ports = getattr(tier, "num_ports_per_rank", None)
        if not getattr(tier, "endpoint_rates", ()) and ports is not None:
            # A declared per-port link count is evidence of one physical
            # full-duplex link per port; an undeclared aggregate is not.
            capacity = float(tier.bw_gbps) * 1e9 / float(max(1, int(ports)))
            for members in getattr(tier, "link_members", ()):
                for rank in members:
                    add(f"port:{int(rank)}:egress", capacity)
                    add(f"port:{int(rank)}:ingress", capacity)
    return inventory


def collective_cut_demands(tree: Any, groups: Sequence[Sequence[int]],
                           kind: str, payload: float) -> dict[str, float]:
    """Per-cut physical bytes of every group in one synchronous step."""
    result: dict[str, float] = {}
    for group in groups:
        ids = tuple(int(rank) for rank in group)
        if len(ids) < 2:
            continue
        wire = collective_wire(kind, len(ids), payload)
        if wire <= 0.0:
            continue
        for tier in getattr(tree, "tiers", ()):
            if not _tier_members(tier, ids):
                continue
            switch = getattr(tier, "switch_capacity_bytes_per_ms", None)
            if switch is not None:
                key = f"switch:{tier.name}"
                result[key] = result.get(key, 0.0) + wire
            if str(getattr(tier, "semantics", "shared")) in {"shared", "mesh"}:
                key = f"tier:{tier.name}"
                result[key] = result.get(key, 0.0) + wire
    return result


def unique_cut_demands(tree: Any, rows: Sequence[
        tuple[Sequence[Sequence[int]], str, float]]) -> dict[str, float]:
    """Aggregate cut bytes once per distinct physical communicator."""
    seen: set[tuple[tuple[int, ...], str, float]] = set()
    result: dict[str, float] = {}
    for groups, kind, payload in rows:
        for group in groups:
            ids = tuple(int(rank) for rank in group)
            if len(ids) < 2:
                continue
            key = (ids, str(kind), float(payload))
            if key in seen:
                continue
            seen.add(key)
            result = merge_demands(
                result, collective_cut_demands(tree, (ids,), kind, payload))
    return result


def aggregate_switch_bound(tree: Any, groups: Sequence[Sequence[int]],
                           kind: str, payload: float) -> float:
    """Concurrent switch service in microseconds, aggregated by identity."""
    demands = collective_cut_demands(tree, groups, kind, payload)
    bound = 0.0
    for tier in getattr(tree, "tiers", ()):
        key = f"switch:{tier.name}"
        if key in demands and tier.switch_capacity_bytes_per_ms:
            bound = max(bound, demands[key] * 1000.0
                        / float(tier.switch_capacity_bytes_per_ms))
    return bound


def switch_bound_from_demands(tree: Any,
                              demands: Mapping[str, float]) -> float:
    """Microseconds of the slowest declared switch for one concurrent step."""
    bound = 0.0
    for tier in getattr(tree, "tiers", ()):
        switch = getattr(tier, "switch_capacity_bytes_per_ms", None)
        if switch is None:
            continue
        value = demands.get(f"switch:{tier.name}", 0.0)
        if value > 0.0:
            bound = max(bound, value * 1000.0 / float(switch))
    return bound


def peer_cut_demands(tree: Any, endpoints: Sequence[int], payload: float,
                     *, dtype: str | None = None,
                     topology: str | None = None) -> dict[str, float]:
    """Bytes a point-to-point transfer moves over each shared physical cut.

    A PP activation hop or KV handoff moves ``payload`` once, unlike a
    collective that repeats a ring round per participant.  The fabric's own
    peer route is authoritative; the fallback keeps the same tier membership
    rule for injected trees that expose only ``tiers``.
    """
    ids = tuple(int(rank) for rank in endpoints)
    value = max(0.0, float(payload))
    if len(ids) < 2 or value <= 0.0:
        return {}
    route = getattr(tree, "peer_cut_demands", None)
    if callable(route):
        return {str(key): float(bytes_)
                for key, bytes_ in route(value, ids, dtype=dtype,
                                         topology=topology).items()}
    result: dict[str, float] = {}
    for tier in getattr(tree, "tiers", ()):
        if not _tier_members(tier, ids):
            continue
        switch = getattr(tier, "switch_capacity_bytes_per_ms", None)
        if switch is not None:
            key = f"switch:{tier.name}"
            result[key] = result.get(key, 0.0) + value
        if str(getattr(tier, "semantics", "shared")) in {"shared", "mesh"}:
            key = f"tier:{tier.name}"
            result[key] = result.get(key, 0.0) + value
    return result


def sequential_cut_demands(tree: Any, rows: Sequence[
        tuple[Sequence[Sequence[int]], str, float]]) -> dict[str, float]:
    """Sum every issued collective; identical calls are distinct events."""
    result: dict[str, float] = {}
    for groups, kind, payload in rows:
        result = merge_demands(
            result, collective_cut_demands(tree, groups, kind, payload))
    return result


def _stage_rank_blocks(plan: Mapping[str, Any], pp: int,
                       endpoint_ids: Sequence[int]) -> list[tuple[int, ...]]:
    """Physical endpoint blocks of every PP stage in the declared placement."""
    rank_map = plan.get("rank_map") or ()
    world = len(endpoint_ids)
    blocks: list[list[int]] = [[] for _ in range(max(0, int(pp)))]
    if rank_map:
        for rank, coordinate in enumerate(rank_map):
            if rank >= world:
                break
            stage = int(coordinate[0])
            if 0 <= stage < len(blocks):
                blocks[stage].append(int(endpoint_ids[rank]))
    else:
        width = max(1, world // max(1, int(pp)))
        for stage in range(len(blocks)):
            lo = stage * width
            hi = world if stage == len(blocks) - 1 else (stage + 1) * width
            blocks[stage].extend(int(value) for value in endpoint_ids[lo:hi])
    return [tuple(block) for block in blocks]


def _pp_hop_demands(tree: Any, plan: Mapping[str, Any],
                    endpoint_ids: Sequence[int], *, tokens: float,
                    hidden: int, element_bytes: float, pp: int
                    ) -> dict[str, float]:
    """Every adjacent PP hop, on the physical route it actually traverses."""
    boundary = float(tokens) * float(hidden) * float(element_bytes) \
        / float(max(1, int(plan.get("tp", 1) or 1)))
    if int(pp) <= 1 or boundary <= 0.0:
        return {}
    blocks = _stage_rank_blocks(plan, int(pp), endpoint_ids)
    result: dict[str, float] = {}
    for stage in range(len(blocks) - 1):
        endpoints = tuple(blocks[stage]) + tuple(blocks[stage + 1])
        result = merge_demands(
            result, peer_cut_demands(tree, endpoints, boundary))
    return result


def _step_demands(tree: Any, plan: Mapping[str, Any],
                  endpoint_ids: Sequence[int], *, tokens: float,
                  hidden: int, element_bytes: float, layers: int,
                  top_k: int, pp: int) -> dict[str, float]:
    """Inference collective + PP-hop demand for one request and one step."""
    unit = float(tokens) * float(hidden) * float(element_bytes)
    rows: list[tuple[Sequence[Sequence[int]], str, float]] = []
    for axis, kind in _AXIS_KIND:
        groups = []
        for group in plan.get(axis) or ():
            ids = []
            for rank in group:
                index = int(rank)
                if index < 0 or index >= len(endpoint_ids):
                    raise ModelingError(
                        f"parallel_explorer_group_rank_invalid:{index}")
                ids.append(endpoint_ids[index])
            if len(ids) >= 2:
                groups.append(tuple(ids))
        if not groups:
            continue
        payload = unit * (max(0, int(top_k)) if axis == "ep_groups" else 1.0)
        rows.append((tuple(groups), kind, payload))
    demands = {key: value * float(layers)
               for key, value in unique_cut_demands(tree, rows).items()}
    # PP activation: one payload per token per adjacent stage boundary, once
    # per decode step rather than once per layer.  Each boundary is routed on
    # its own physical stage endpoints, never collapsed into one synthetic
    # all-to-all over the whole world.
    demands = merge_demands(demands, _pp_hop_demands(
        tree, plan, endpoint_ids, tokens=tokens, hidden=hidden,
        element_bytes=element_bytes, pp=pp))
    return demands


def request_phase_demands(tree: Any, plan: Mapping[str, Any],
                          endpoint_ids: Sequence[int], *,
                          tokens: float, hidden: int, element_bytes: float,
                          layers: int, top_k: int, pp: int
                          ) -> dict[str, float]:
    """Per-request demand of one inference phase on every shared cut."""
    return _step_demands(
        tree, plan, endpoint_ids, tokens=tokens, hidden=hidden,
        element_bytes=element_bytes, layers=layers, top_k=top_k, pp=pp)


def merge_demands(*parts: Mapping[str, float]) -> dict[str, float]:
    """Conserve byte sums by physical cut identity."""
    result: dict[str, float] = {}
    for part in parts:
        for key, value in part.items():
            result[key] = result.get(key, 0.0) + float(value)
    return result


def cut_ceiling(tree: Any, demands_per_request: Mapping[str, float]
                ) -> tuple[float, str | None]:
    """Minimum ``capacity / demand`` over every demanded shared cut."""
    inventory = cut_inventory(tree)
    ceiling = float("inf")
    limiting: str | None = None
    for key, demand in demands_per_request.items():
        if demand <= 0.0:
            continue
        capacity = inventory.get(key)
        if capacity is None or capacity <= 0.0:
            continue
        rate = capacity / demand
        if rate < ceiling:
            ceiling = rate
            limiting = key
    return ceiling, limiting


__all__ = ["aggregate_switch_bound", "collective_cut_demands",
           "collective_port_demands", "collective_wire",
           "cut_ceiling", "cut_inventory", "merge_demands",
           "peer_cut_demands", "peer_port_demands",
           "request_phase_demands", "sequential_cut_demands",
           "switch_bound_from_demands", "unique_cut_demands"]
