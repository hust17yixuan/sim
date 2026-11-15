"""Pool cut composition from actual placed P/D instances and routed shares.

[Role]
  Turn the per-instance, per-request operation ledgers of the placed prefill and
  decode instances into one pool cut ledger under capacity-proportional request
  routing, including the actual P->D handoff and destination remap routes.

[Invariants]
  * Each instance contributes its *own* emitted ledger; a preliminary rank-zero
    template is never substituted for a placed instance.
  * Routing shares are capacity-proportional: ``alpha_i = mu_i / sum(mu)`` and
    ``beta_j = nu_j / sum(nu)`` with non-negative sums of one.
  * The handoff term is the expectation over independent P/D choices,
    ``sum_ij alpha_i beta_j * handoff(i, j)``, never an unweighted replica sum.
  * The destination remap is owned by the destination, so it is weighted by
    ``beta_j`` once, not once per P/D pair.
  * A zero-capacity instance rejects composition instead of dividing by zero.
  * Routed instance rates ``alpha_i * rate`` and ``beta_j * rate`` never exceed
    the corresponding instance capacity, and an instance's own share of a
    shared cut never exceeds that cut's declared physical capacity.
  * Every routed byte class is also charged to each physical source/destination
    ingress and egress port (``alpha_i * beta_j`` handoff, ``alpha_i``/``beta_j``
    inference, ``beta_j`` remap); a fast destination never borrows another
    endpoint's idle NIC port through an unweighted aggregate.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh dicts; caller instance rows are untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from mindie_sim.parallel_cuts import (cut_inventory, merge_demands,
                                      peer_cut_demands)
from mindie_sim.schema.common import ModelingError
from mindie_sim.xpyd_bridge_ownership import physical_bridge_edges


def _weighted_sum(pairs: Sequence[tuple[float, float]]) -> float:
    """Weighted sum that preserves an exactly uniform per-instance demand."""
    if not pairs:
        return 0.0
    first = float(pairs[0][0])
    if all(float(value) == first for value, _weight in pairs):
        return first
    return math.fsum(float(value) * float(weight)
                     for value, weight in pairs)


def capacity_shares(instances: Sequence[Mapping[str, Any]]) -> list[float] | None:
    """Capacity-proportional request shares, or ``None`` for zero capacity."""
    caps = [max(0.0, float(item.get("mu_requests_per_s", 0.0) or 0.0))
            for item in instances]
    total = float(sum(caps))
    if not instances or total <= 0.0:
        return None
    shares = [value / total for value in caps]
    # One pool request is routed exactly once; remove the final ULPS so the
    # shares conserve one and an equal-capacity witness reproduces its exact
    # per-instance demand.
    if shares:
        shares[-1] += 1.0 - math.fsum(shares)
    return shares


def _weighted_instance_field(
        instances: Sequence[Mapping[str, Any]], shares: Sequence[float],
        field: str) -> dict[str, float]:
    keys = {str(key) for item in instances
            for key in item.get(field, {})}
    return {key: _weighted_sum([
        (float(item.get(field, {}).get(key, 0.0)), float(share))
        for item, share in zip(instances, shares)]) for key in keys}


def _weighted_instance_demands(
        instances: Sequence[Mapping[str, Any]], shares: Sequence[float]
) -> dict[str, float]:
    return _weighted_instance_field(instances, shares, "shared_cut_demands")


def _edge_demands(tree: Any, edges: Mapping[tuple[int, int], float]):
    demands, local = {}, 0.0
    for (source, destination), value in edges.items():
        if source == destination:
            local += value
            continue
        demands = merge_demands(demands, peer_cut_demands(tree, (source, destination), value),
                                {f"port:{source}:egress": value,
                                 f"port:{destination}:ingress": value})
    return demands, local


def _conserved_edges(edges, expected):
    if not math.isclose(math.fsum(edges.values()), expected, rel_tol=1e-12, abs_tol=1e-6):
        raise ModelingError("bridge_physical_wire_not_conserved")


def compose_pool_cut_demands(
    tree: Any, p_inst: Sequence[Mapping[str, Any]],
    d_inst: Sequence[Mapping[str, Any]], *, transfer_bytes: float,
    remap_bytes: float, bridge: Any = None, transfer: Mapping[str, Any] | None = None,
    case: Any = None, edge_cache: dict | None = None,
) -> dict[str, Any]:
    """Compose one pool request's physical cut and port ledger."""
    alpha = capacity_shares(p_inst)
    beta = capacity_shares(d_inst)
    if alpha is None or beta is None:
        return {"qualified": False, "reason": "zero_instance_capacity",
                "demands": {}, "alpha": [], "beta": []}
    handoff_edges, remap_edges = {}, {}
    totals = {"unique_logical_bytes": 0.0, "destination_replica_bytes": 0.0,
              "handoff_header_bytes": 0.0, "handoff_total_bytes": 0.0,
              "remap_total_bytes": 0.0}
    if (transfer_bytes > 0.0 or remap_bytes > 0.0) and (
            bridge is None or transfer is None or case is None):
        return {"qualified": False, "reason": "bridge_ownership_missing",
                "demands": {}, "alpha": alpha, "beta": beta}
    def weighted_add(target, edges, weight):
        for key, value in edges.items():
            target[key] = target.get(key, 0.0) + weight * value
    cache = {} if edge_cache is None else edge_cache
    def placed_edges(left, right, remap=False):
        key = (id(case), id(bridge), id(transfer), remap,
               tuple(tuple(row) for row in left["plan"]["rank_map"]),
               tuple(tuple(row) for row in right["plan"]["rank_map"]),
               right["plan"]["active_lanes"])
        # Ownership in local coordinates is invariant under the physical
        # endpoint map.  Reuse it across pool counts, then map every edge to
        # the *current* placed instances before applying their routing shares.
        source_ids, dest_ids = tuple(left["device_ids"]), tuple(right["device_ids"])
        physical = dest_ids if remap else source_ids + dest_ids
        if key not in cache:
            start = 0 if remap else len(source_ids)
            local_left = dict(left, device_ids=tuple(range(len(source_ids))))
            local_right = dict(right, device_ids=tuple(range(start, start + len(dest_ids))))
            cache[key] = physical_bridge_edges(case, bridge, transfer, local_left,
                                               local_right, remap=remap)
        record = cache[key]
        return ({(physical[s], physical[d]): value for (s, d), value in record["edges"].items()}, record)
    try:
        for left, left_share in zip(p_inst, alpha):
            for right, right_share in zip(d_inst, beta):
                pair = float(left_share) * float(right_share)
                if pair <= 0.0 or transfer_bytes <= 0.0:
                    continue
                edges, record = placed_edges(left, right)
                _conserved_edges(edges, record["destination_data_bytes"] + record["header_bytes"])
                for field in ("unique_logical_bytes", "destination_replica_bytes"):
                    totals[field] += pair * record[field]
                totals["handoff_header_bytes"] += pair * record["header_bytes"]
                totals["handoff_total_bytes"] += pair * record["total_bytes"]
                weighted_add(handoff_edges, edges, pair)
        for right, share in zip(d_inst, beta):
            if share > 0.0 and remap_bytes > 0.0:
                edges, record = placed_edges(right, right, remap=True)
                _conserved_edges(edges, record["destination_data_bytes"] + record["header_bytes"])
                totals["remap_total_bytes"] += share * record["total_bytes"]
                weighted_add(remap_edges, edges, share)
    except ModelingError as exc:
        return {"qualified": False, "reason": str(exc), "demands": {},
                "alpha": alpha, "beta": beta}
    handoff, handoff_local = _edge_demands(tree, handoff_edges)
    remap, remap_local = _edge_demands(tree, remap_edges)
    instance_demands = merge_demands(
        _weighted_instance_demands(p_inst, alpha),
        _weighted_instance_demands(d_inst, beta))
    instance_ports = merge_demands(
        _weighted_instance_field(p_inst, alpha, "port_demands"),
        _weighted_instance_field(d_inst, beta, "port_demands"))
    composed = merge_demands(
        instance_demands, instance_ports, handoff, remap)
    # Only declared per-endpoint capacities are physical resources.  A port
    # key without an inventory entry has no declared capacity and is not
    # charged; this keeps aggregate tier bandwidth out of the port ledger.
    inventory = cut_inventory(tree)
    port_demands = {str(key): value for key, value in composed.items()
                    if str(key).startswith("port:") and str(key) in inventory}
    composed = {str(key): value for key, value in composed.items()
                if not str(key).startswith("port:") or str(key) in inventory}
    return {"qualified": True, "reason": None, "demands": composed,
            **totals,
            "wire_bytes": totals["handoff_total_bytes"] + totals["remap_total_bytes"] - handoff_local - remap_local,
            "alpha": list(alpha), "beta": list(beta),
            "handoff_demands": handoff, "remap_demands": remap,
            "handoff_rank_edges": tuple((s, d, b) for (s, d), b in sorted(handoff_edges.items())),
            "remap_rank_edges": tuple((s, d, b) for (s, d), b in sorted(remap_edges.items())),
            "handoff_local_bytes": handoff_local, "remap_local_bytes": remap_local,
            "port_demands": port_demands}


def routed_instance_rates(
    p_inst: Sequence[Mapping[str, Any]], d_inst: Sequence[Mapping[str, Any]],
    *, admitted_rps: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """P and D instance rates implied by capacity-proportional routing."""
    alpha = capacity_shares(p_inst) or []
    beta = capacity_shares(d_inst) or []
    return (tuple(float(value) * float(admitted_rps) for value in alpha),
            tuple(float(value) * float(admitted_rps) for value in beta))


def routed_rates_within_capacity(
    tree: Any, p_inst: Sequence[Mapping[str, Any]],
    d_inst: Sequence[Mapping[str, Any]], *, admitted_rps: float,
) -> tuple[bool, str | None]:
    """Reject a rate that exceeds any routed instance or its own cut share."""
    p_rates, d_rates = routed_instance_rates(
        p_inst, d_inst, admitted_rps=admitted_rps)
    for label, instances, rates in (("prefill", p_inst, p_rates),
                                    ("decode", d_inst, d_rates)):
        for index, (item, rate) in enumerate(zip(instances, rates)):
            capacity = max(0.0, float(item.get("mu_requests_per_s", 0.0) or 0.0))
            if rate > capacity + 1e-9:
                return False, f"{label}_instance_rate_exceeded:{index}"
    inventory = cut_inventory(tree)
    for label, instances, rates in (("prefill", p_inst, p_rates),
                                    ("decode", d_inst, d_rates)):
        for index, (item, rate) in enumerate(zip(instances, rates)):
            for key, demand in item.get("shared_cut_demands", {}).items():
                capacity = inventory.get(str(key))
                if capacity is None or capacity <= 0.0:
                    continue
                if float(rate) * float(demand) > float(capacity) + 1e-6:
                    return False, (f"{label}_instance_cut_exceeded:"
                                   f"{index}:{key}")
            for key, demand in item.get("port_demands", {}).items():
                capacity = inventory.get(str(key))
                if capacity is None or capacity <= 0.0:
                    continue
                if float(rate) * float(demand) > float(capacity) + 1e-6:
                    return False, (f"{label}_instance_port_exceeded:"
                                   f"{index}:{key}")
    return True, None


__all__ = ["capacity_shares", "compose_pool_cut_demands",
           "routed_instance_rates", "routed_rates_within_capacity"]
