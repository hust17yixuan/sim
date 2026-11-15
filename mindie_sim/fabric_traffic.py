"""AllToAllV traffic matrix and endpoint/cut reduction for fabric timing.

[Role]
  Build one canonical ``R x R`` communicator traffic matrix, map its local rows
  and columns onto physical endpoint identities, and reduce every per-sender,
  per-receiver, switch and shared/port cut to one deterministic service time.

[Invariants]
  * ``endpoints[i]`` is the physical endpoint of communicator-local rank ``i``.
    Traffic arrays are indexed by ``i``; endpoint rate limits are looked up by
    ``endpoints[i].rank``.  Physical IDs never index local arrays.
  * Uniform/hotspot semantics are derived solely from the off-diagonal matrix
    entries (``1e-9`` tolerance), never from whether a vector was supplied.
  * Per-sender egress, per-receiver ingress and total cut conserve the same
    matrix: a uniform implicit buffer and an explicit uniform vector are
    byte-identical.
  * No endpoint for a declared rate is silently dropped, and no rate is applied
    to a different physical rank after a permutation.

[Timeouts & Locks] Pure bounded arithmetic over one finite matrix; no I/O.
[Side-Effect Policy] Fresh immutable result; caller records untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from mindie_sim.schema.common import ModelingError


def _positive(name: str, value: Any, *, minimum: float = 0.0,
              strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"fabric_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ModelingError(f"fabric_{name}_invalid:{value}")
    return number


def _effective_bw_gbps(payload_bytes: float, total_us: float) -> float:
    """Useful payload GB/s: bytes/us * 1e6 (bytes/s) / 1e9."""
    return payload_bytes / total_us / 1000.0 if total_us > 0 else 0.0


def _coerce_endpoints(values: Sequence[Any] | None):
    if values is None:
        return ()
    from mindie_sim.fabric_topology import RankEndpoint
    result = []
    for value in values:
        if isinstance(value, RankEndpoint):
            result.append(value)
        elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            # A bare integer names a physical rank, not a package/node
            # ancestry: explicit IDs in one communicator are one flat domain.
            result.append(RankEndpoint(value, value, 0, 0, 0, 0))
        else:
            raise ModelingError(f"fabric_endpoint_invalid:{value}")
    ranks = [item.rank for item in result]
    if len(set(ranks)) != len(ranks):
        raise ModelingError("fabric_endpoint_duplicate_rank")
    return tuple(result)


def _traffic_matrix(payload: float, ranks: int,
                    payloads_by_destination: Sequence[float] | None
                    ) -> dict[str, Any]:
    """One canonical ``R x R`` AllToAllV traffic matrix.

    ``payload_bytes`` is the *total per-rank buffer*; without an explicit
    destination split it is divided evenly over all ``ranks`` slots, so only
    ``payload*(R-1)/R`` bytes leave each sender.  ``payloads_by_destination``
    is that same per-destination split: ``M[s][d] = vec[d]`` for ``s != d`` and
    zero on the diagonal (self bytes never touch the wire).  The implicit
    default and an explicit uniform vector are therefore bit-identical.
    """
    if payloads_by_destination is None:
        per_destination = payload / float(ranks)
        values = (per_destination,) * ranks
    else:
        values = tuple(_positive("destination_payload", value, strict=False)
                       for value in payloads_by_destination)
        if len(values) != ranks:
            raise ModelingError(
                f"fabric_all_to_all_destination_count:{len(values)}!={ranks}")
    matrix = tuple(tuple(0.0 if source == dest else values[dest]
                         for dest in range(ranks))
                   for source in range(ranks))
    egress = tuple(sum(row) for row in matrix)
    ingress = tuple(sum(matrix[source][dest] for source in range(ranks))
                    for dest in range(ranks))
    total = float(sum(egress))
    off_diagonal = [matrix[source][dest] for source in range(ranks)
                    for dest in range(ranks) if source != dest]
    uniform = (not off_diagonal
               or all(abs(value - off_diagonal[0]) <= 1e-9
                      for value in off_diagonal))
    return {"matrix": matrix, "egress": egress, "ingress": ingress,
            "max_sender": max(egress, default=0.0),
            "max_receiver": max(ingress, default=0.0),
            "cut": total, "uniform": uniform, "values": values}


def _timing(kind: str, payload: float, endpoints: Sequence[Any] | None,
            hops: list[tuple[Any, int, int, int]], *, method: str,
            payloads_by_destination: Sequence[float] | None = None,
            hotspot: bool = False,
            qualified: bool = True,
            evidence: str = "declared_endpoint_route"):
    from mindie_sim.fabric_topology import CollectiveTiming, TIER_ORDER
    ranks = len(endpoints) if endpoints else (hops[-1][3] if hops else 1)
    flat_share = (ranks - 1) / float(ranks) if ranks > 1 else 0.0
    traffic = None
    if kind == "all_to_all_v":
        traffic = _traffic_matrix(payload, ranks, payloads_by_destination)
        flat_wire = traffic["max_sender"]
    elif kind in {"all_reduce", "reduce_scatter"}:
        flat_wire = 2.0 * payload * flat_share if kind == "all_reduce" else payload * flat_share
    else:
        flat_wire = payload * flat_share
    per_tier, rounds, total_rounds = [], [], 0
    wire_total = 0.0
    shared_cut_bound = 0.0
    for tier, groups, before, after in hops:
        share = max(0.0, (after - before) / float(max(1, ranks - 1)))
        wire = flat_wire * share
        if kind in {"all_reduce", "reduce_scatter"}:
            count = 2 * (groups - 1) if kind == "all_reduce" else (groups - 1)
        else:
            count = groups - 1
        os_factor = tier.oversubscription if tier.semantics in {"shared", "mesh"} else 1.0
        us = (tier.alpha_us * count + tier.per_peer_us * (groups - 1)
              + wire * 1.0 / tier.bw_gbps * os_factor / 1000.0)
        if kind == "all_to_all_v" and tier.semantics in {"shared", "mesh"}:
            # A shared port carries every rank's remote bytes: the physical
            # port demand is the whole cut crossing that tier, not one sender's
            # private share.
            shared_cut_bound = max(
                shared_cut_bound,
                traffic["cut"] * share * os_factor / (tier.bw_gbps * 1000.0))
        if us > 0:
            per_tier.append((tier.name, us))
        rounds.append((tier.name, count))
        total_rounds += count
        wire_total += wire
    del total_rounds
    base_total = float(sum(us for _, us in per_tier))
    bottleneck = max(per_tier, key=lambda item: (item[1], TIER_ORDER.index(item[0]))) \
        if per_tier else ("none", 0.0)
    path = tuple(tier.name for tier, *_ in hops)
    participants = tuple(entry.rank for entry in endpoints) if endpoints \
        else tuple(range(ranks))
    participant_set = set(participants)
    # Link membership is physical evidence: a participant outside every
    # declared member set of a used tier is disconnected, never silently
    # priced as an intra-node transfer.
    for tier, _groups, _before, _after in hops:
        if not tier.link_members:
            continue
        if participant_set and not any(participant_set <= set(members)
                                       for members in tier.link_members):
            qualified, evidence = False, "disconnected_participants"
            break
    # One byte ledger drives every bound: per-sender egress, per-receiver
    # ingress and the total cut.  Endpoint rates apply to the direction that
    # actually lands on that endpoint.  Local matrix index ``i`` maps to
    # physical ``endpoints[i].rank``; a physical ID must never index the
    # communicator-local arrays.
    local_of_rank = {int(entry.rank): index
                     for index, entry in enumerate(endpoints or ())}
    if kind == "all_to_all_v":
        per_sender = traffic["max_sender"]
        per_receiver = traffic["max_receiver"]
        cut = traffic["cut"]
        # Uniform/hotspot is a matrix property only.
        hotspot = bool(not traffic["uniform"])
    else:
        per_sender = per_receiver = flat_wire
        cut = wire_total
    endpoint_bound = 0.0
    switch_bound = 0.0
    sender_bound = 0.0
    receiver_bound = 0.0
    receiver_endpoint_bound = 0.0
    for tier, _groups, _before, _after in hops:
        os_factor = (tier.oversubscription
                     if tier.semantics in {"shared", "mesh"} else 1.0)
        for rank, rate in tier.endpoint_rates:
            physical = int(rank)
            if physical in participant_set:
                local = local_of_rank.get(physical, physical)
                if kind == "all_to_all_v":
                    egress = traffic["egress"][local]
                    ingress = traffic["ingress"][local]
                else:
                    egress = ingress = per_sender
                endpoint_bound = max(endpoint_bound,
                                     egress * 1000.0 / max(1e-12, rate))
                receiver_endpoint_bound = max(
                    receiver_endpoint_bound,
                    ingress * 1000.0 / max(1e-12, rate))
        if tier.switch_capacity_bytes_per_ms is not None:
            switch_bound = max(
                switch_bound,
                cut * 1000.0 / max(1e-12, tier.switch_capacity_bytes_per_ms))
        if kind == "all_to_all_v":
            per_port = tier.semantics in {"per_port", "per_pair", "mesh"}
            ports = (tier.num_ports_per_rank
                     if tier.num_ports_per_rank is not None
                     else (ranks if per_port else 1))
            sender_bound = max(
                sender_bound,
                per_sender * os_factor
                / max(1e-12, ports * tier.bw_gbps * 1000.0))
            # A declared hotspot still lands on the receiver's single NIC
            # ingress port; uniform peer traffic may use independent ports.
            ingress_ports = 1 if hotspot else ports
            receiver_bound = max(
                receiver_bound,
                per_receiver * os_factor
                / max(1e-12, ingress_ports * tier.bw_gbps * 1000.0))
    if kind == "all_to_all_v" and any(
            tier.semantics in {"shared", "mesh"} for tier, *_ in hops):
        receiver_bound = max(receiver_bound, shared_cut_bound)
    port_bound = max(sender_bound, receiver_bound)
    total = max(base_total, endpoint_bound, receiver_endpoint_bound,
                switch_bound, port_bound)
    return CollectiveTiming(kind, ranks, payload, total, tuple(per_tier),
                            bottleneck[0], float(bottleneck[1]), wire_total,
                            _effective_bw_gbps(payload, total), method,
                            tuple(rounds), path, qualified, evidence,
                            per_sender, per_receiver, cut,
                            max(receiver_endpoint_bound, receiver_bound,
                                switch_bound))
