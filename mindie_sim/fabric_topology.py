"""Two-mode communication fabric and counterfactual hardware scaling.
[Role]
  Price hierarchical collectives over explicit rank endpoints and die/package/
  node ancestry (Mode A) or measured piecewise curves (Mode B); project a Case
  onto another SKU or scale one physical supply immutably.

[Invariants]
  * A communicator is an explicit endpoint set.  Equal cardinality on different
    ancestry is not the same collective; a missing die link is unqualified
    evidence, never silently priced as an intra-node transfer.
  * Ring startup is ``g-1`` rounds for AllGather/ReduceScatter and ``2*(g-1)``
    for AllReduce.  Incomplete groups use actual uneven sizes; non-power-of-two
    communicators are legal and per-tier wire conserves the flat ring budget.
  * ``effective_bw_gbps`` is payload bytes/us/1000; MC2 doubles traversal time.
  * AllToAllV is priced from one ``R x R`` byte matrix: ``payload_bytes`` is
    the total per-rank buffer (uniformly split over destinations), sender
    egress, receiver ingress and total cut are conserved, and implicit and
    explicit uniform vectors are byte-identical.
  * Mode B defaults to typed ``out_of_range``; extrapolation is explicit and an
    effective-under-load beta cannot receive a second congestion penalty.
  * ``scale_hardware`` scales every declared affected link/topology bandwidth
    together and floors integer capacities after scaling.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] No mutation of caller records or global state.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

from mindie_sim.fabric_traffic import (_coerce_endpoints, _effective_bw_gbps,
                                      _timing)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device, Hardware, Link

TIER_ORDER = ("intra_package_die", "intra_node_hccs", "superpod_hccs",
              "pod_roce", "spine_roce")
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all_v", "mc2")
_LOCALITY = {"intra_package_die": 1, "intra_node_hccs": 2, "superpod_hccs": 3,
             "pod_roce": 4, "spine_roce": 5}
_EXTRAPOLATION_POLICIES = frozenset({"out_of_range", "extrapolate"})
def _positive(name: str, value: Any, *, minimum: float = 0.0, strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"fabric_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ModelingError(f"fabric_{name}_invalid:{value}")
    return number
class FabricRangeError(ModelingError):
    """Typed Mode B range classification (not an extrapolated value)."""
@dataclass(frozen=True)
class RankEndpoint:
    """Absolute rank identity and its explicit physical ancestry."""

    rank: int
    die: int
    package: int
    node: int
    superpod: int = 0
    pod: int = 0

    def __post_init__(self) -> None:
        for name in ("rank", "die", "package", "node", "superpod", "pod"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"fabric_endpoint_{name}_invalid:{value}")

    def to_dict(self) -> dict[str, int]:
        return dict(vars(self))
@dataclass(frozen=True)
class FabricTier:
    """One nested locality domain: four physical parameters plus affinity."""

    name: str
    domain_ranks: int
    alpha_us: float
    bw_gbps: float
    oversubscription: float = 1.0
    per_peer_us: float = 0.0
    semantics: str = "shared"
    die_pair: bool = False
    evidence: str = "declared"
    #: Link member sets copied from the physical records.  A collective whose
    #: participants are not contained in one declared link is disconnected.
    link_members: tuple[tuple[int, ...], ...] = ()
    #: Declared per-rank send/receive port service, bytes per millisecond.
    endpoint_rates: tuple[tuple[int, float], ...] = ()
    switch_capacity_bytes_per_ms: float | None = None
    #: Independent full-duplex ports per rank for ``per_port`` semantics.
    num_ports_per_rank: int | None = None

    def __post_init__(self) -> None:
        if self.name not in TIER_ORDER:
            raise ModelingError(f"fabric_tier_unknown:{self.name}")
        if isinstance(self.domain_ranks, bool) or not isinstance(self.domain_ranks, int) \
                or self.domain_ranks < 2:
            raise ModelingError(f"fabric_tier_domain_invalid:{self.name}:{self.domain_ranks}")
        object.__setattr__(self, "alpha_us", _positive("alpha_us", self.alpha_us, strict=False))
        object.__setattr__(self, "bw_gbps", _positive("bw_gbps", self.bw_gbps))
        object.__setattr__(self, "oversubscription", _positive(
            "oversubscription", self.oversubscription, minimum=1.0, strict=False))
        object.__setattr__(self, "per_peer_us", _positive("per_peer_us", self.per_peer_us, strict=False))
        if self.semantics not in {"shared", "per_port", "per_pair", "mesh"}:
            raise ModelingError(f"fabric_tier_semantics_invalid:{self.semantics}")
        if self.die_pair and self.domain_ranks != 2:
            raise ModelingError("fabric_die_pair_requires_domain_two")
        if not str(self.evidence or "declared"):
            raise ModelingError("fabric_tier_evidence_missing")
        members = []
        for group in self.link_members:
            values = tuple(int(value) for value in group)
            if not values or any(value < 0 for value in values):
                raise ModelingError(f"fabric_tier_link_members_invalid:{self.name}")
            members.append(tuple(sorted(set(values))))
        object.__setattr__(self, "link_members", tuple(members))
        rates = tuple((int(rank), _positive("endpoint_rate", rate))
                      for rank, rate in self.endpoint_rates)
        if len({rank for rank, _ in rates}) != len(rates):
            raise ModelingError(f"fabric_tier_endpoint_rate_duplicate:{self.name}")
        object.__setattr__(self, "endpoint_rates", rates)
        if self.switch_capacity_bytes_per_ms is not None:
            object.__setattr__(self, "switch_capacity_bytes_per_ms", _positive(
                "switch_capacity", self.switch_capacity_bytes_per_ms))
        if self.num_ports_per_rank is not None and (
                isinstance(self.num_ports_per_rank, bool)
                or not isinstance(self.num_ports_per_rank, int)
                or self.num_ports_per_rank < 1):
            raise ModelingError(f"fabric_tier_ports_invalid:{self.name}")

    @property
    def locality(self) -> int:
        return _LOCALITY[self.name]

    def to_dict(self) -> dict[str, Any]:
        data = dict(vars(self))
        data["link_members"] = [list(group) for group in self.link_members]
        data["endpoint_rates"] = dict(self.endpoint_rates)
        return data

@dataclass(frozen=True)
class CollectiveTiming:
    """Deterministic completion time of one collective on explicit endpoints."""
    kind: str
    ranks: int
    payload_bytes: float
    total_us: float
    tier_us: tuple[tuple[str, float], ...]
    bottleneck_tier: str
    bottleneck_us: float
    wire_bytes_per_rank: float
    effective_bw_gbps: float
    method: str
    rounds: tuple[tuple[str, int], ...] = ()
    endpoint_path: tuple[str, ...] = ()
    route_qualified: bool = True
    evidence: str = "declared"
    max_sender_egress_bytes: float = 0.0
    max_receiver_ingress_bytes: float = 0.0
    cut_bytes: float = 0.0
    receiver_bound_us: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = dict(vars(self))
        data["tier_us"] = dict(self.tier_us)
        data["rounds"] = dict(self.rounds)
        data["endpoint_path"] = list(self.endpoint_path)
        return data

def _ancestor_key(endpoint: RankEndpoint, level: int) -> tuple[int, ...]:
    """Physical grouping key at a locality level (0=die, 1=package, ...)."""
    if level <= 0:
        return (endpoint.die,)
    if level == 1:
        return (endpoint.package,)
    if level == 2:
        return (endpoint.node,)
    if level == 3:
        return (endpoint.superpod,)
    if level == 4:
        return (endpoint.pod,)
    return (0,)

def _tier_key(tier: FabricTier, endpoint: RankEndpoint) -> tuple[int, ...]:
    """Grouping key of one tier.

    The RoCE pod tier is an *inter-node* cut: two dies with different ``node``
    ancestry are separated by it even when the synthetic pod id is uniform.
    Without this, a cross-node collective produced no hop at all and a
    per-node switch/endpoint cap could never apply.
    """
    if tier.name == "pod_roce":
        return (endpoint.node,)
    if tier.name == "spine_roce":
        return (endpoint.pod,)
    return _ancestor_key(endpoint, tier.locality)

def _count_hops(tiers: Sequence[FabricTier], ranks: int) -> list[tuple[FabricTier, int, int, int]]:
    """Count mode: never overshoots a tier domain."""
    result: list[tuple[FabricTier, int, int, int]] = []
    before = 1
    for tier in tiers:
        if before >= ranks:
            break
        capacity_groups = max(0, tier.domain_ranks // before)
        groups = min(math.ceil(ranks / before), capacity_groups)
        if groups < 2:
            continue
        after = min(ranks, before * groups)
        if after > before:
            result.append((tier, groups, before, after))
        before = after
    if before < ranks:
        raise ModelingError(f"fabric_tiers_do_not_cover_ranks:{ranks}:{before}")
    return result
def _endpoint_hops(tiers: Sequence[FabricTier], endpoints: Sequence[RankEndpoint],
                   required: int) -> tuple[list[tuple[FabricTier, int, int, int]], bool, str]:
    """Endpoint mode: uneven groups plus route qualification."""
    ranks = len(endpoints)
    result: list[tuple[FabricTier, int, int, int]] = []
    previous_count = 1
    unmeasured = False
    for tier in tiers:
        if tier.locality < required:
            continue
        if tier.evidence == "unmeasured_die_link":
            unmeasured = True
            continue
        counts: dict[tuple[int, ...], int] = {}
        for endpoint in endpoints:
            key = _tier_key(tier, endpoint)
            counts[key] = counts.get(key, 0) + 1
        group_count = len(counts)
        if group_count > 1:
            # Endpoints span several ancestor groups: this tier is the cut
            # that joins them, so every remote byte crosses it once.  Using
            # only the largest group size left an evenly split two-node
            # collective with no hop at all.
            after = ranks
            groups = group_count
        else:
            after = max(counts.values(), default=1)
            groups = max(2, math.ceil(after / max(1, previous_count)))
        before = previous_count
        if after > before:
            result.append((tier, groups, before, after))
        previous_count = after
        if after >= ranks:
            break
    if not result or max(item[3] for item in result) < ranks:
        raise ModelingError(f"fabric_no_route_for_endpoints:{required}:{ranks}")
    if unmeasured:
        return result, False, "cross_die_absorbed_unmeasured_die_link"
    return result, True, "declared_endpoint_route"

@dataclass(frozen=True)
class HierarchicalFabricTree:
    """Mode A: explicit endpoint locality tree with physical tier parameters."""

    tiers: tuple[FabricTier, ...]
    rank_placement: tuple[str, ...] = ("tp", "dcp", "pcp", "ep", "pp")
    source: str = "declared"
    assumptions: tuple[tuple[str, str], ...] = field(default=())
    endpoints: tuple[RankEndpoint, ...] = field(default=())

    def __post_init__(self) -> None:
        domains = [tier.domain_ranks for tier in self.tiers]
        if not domains:
            raise ModelingError("fabric_tree_requires_tiers")
        if domains != sorted(set(domains)):
            raise ModelingError(f"fabric_tier_domains_not_increasing:{domains}")
        if len({tier.name for tier in self.tiers}) != len(self.tiers):
            raise ModelingError("fabric_tier_names_not_unique")
        if tuple(self.rank_placement) != ("tp", "dcp", "pcp", "ep", "pp"):
            raise ModelingError("fabric_rank_placement_unsupported")
        ranks = [endpoint.rank for endpoint in self.endpoints]
        if ranks and len(set(ranks)) != len(ranks):
            raise ModelingError("fabric_endpoint_duplicate_rank")

    @property
    def dual_die_affinity(self) -> bool:
        return any(tier.die_pair for tier in self.tiers) or (
            bool(self.endpoints) and len({item.die for item in self.endpoints}) > 1)

    @property
    def member_ranks(self) -> frozenset[int]:
        """Physical ranks that appear in at least one declared link member set."""
        return frozenset(
            rank for tier in self.tiers for members in tier.link_members
            for rank in members)

    def _endpoint_route(self, ranks: int, endpoints: Sequence[int] | None):
        if endpoints is None:
            if not self.endpoints:
                return None, None, True, "count_span"
            chosen = tuple(item for item in self.endpoints if item.rank < ranks)
            if len(chosen) < ranks:
                raise ModelingError(f"fabric_endpoints_missing_for_span:{len(chosen)}!={ranks}")
        else:
            known = {item.rank: item for item in self.endpoints}
            if not self.endpoints:
                chosen = _coerce_endpoints(list(endpoints))
            else:
                chosen = tuple(known[rank] for rank in endpoints if rank in known)
                if len(chosen) != len(endpoints):
                    raise ModelingError("fabric_endpoint_rank_unknown")
        if len(chosen) < 2:
            return chosen, 0, True, "single_rank"
        required = 0
        for left in range(len(chosen)):
            for right in range(left + 1, len(chosen)):
                a, b = chosen[left], chosen[right]
                required = max(required,
                              1 if a.die != b.die else 0,
                              2 if a.package != b.package else 0,
                              3 if a.node != b.node else 0,
                              4 if a.superpod != b.superpod else 0,
                              5 if a.pod != b.pod else 0)
        return chosen, required, True, "explicit_endpoint_route"

    def cut_bandwidth_bytes_per_ms(
            self, endpoints: Sequence[int]) -> float | None:
        """Slowest physical service on the cut joining ``endpoints`` (B/ms).

        Used for point-to-point PP activation hops: the hop traverses the same
        declared tiers and endpoint ports as a collective but moves one
        payload once, so it must not borrow an all-gather round count.
        """
        values = tuple(int(value) for value in endpoints)
        if len(values) < 2:
            return None
        chosen, required, _ok, _mode = self._endpoint_route(len(values), values)
        if not chosen or len(chosen) < 2:
            return None
        hops, _qualified, _evidence = _endpoint_hops(
            self.tiers, chosen, required or 0)
        participant_set = {int(entry.rank) for entry in chosen}
        rates = [float(tier.bw_gbps) * 1e6 for tier, *_ in hops]
        for tier, _groups, _before, _after in hops:
            for rank, rate in tier.endpoint_rates:
                if int(rank) in participant_set:
                    rates.append(float(rate))
        return min(rates) if rates else None

    def peer_transfer(self, payload_bytes: float,
                      endpoints: Sequence[int | RankEndpoint], *,
                      dtype: str | None = None,
                      topology: str | None = None) -> CollectiveTiming:
        """Price one point-to-point activation transfer over physical endpoints.

        A PP activation hop moves one payload once: it pays every traversed
        tier's startup latency, its bandwidth/oversubscription service, the
        declared shared switch capacity and every participant endpoint port.
        Route qualification and link membership are those of the same physical
        path a collective would use -- an absent/unmeasured route is propagated
        as ``route_qualified=False``, never replaced by unrelated hardware
        bandwidth.
        """
        del dtype, topology
        payload = _positive("payload_bytes", payload_bytes, strict=False)
        requested = [item.rank if isinstance(item, RankEndpoint) else item
                     for item in endpoints]
        values = tuple(int(value) for value in requested)
        if len(values) < 2 or payload <= 0.0:
            return CollectiveTiming("peer_transfer", max(1, len(values)), payload,
                                    0.0, (), "none", 0.0, 0.0, 0.0, "peer",
                                    (), (), True, "trivial")
        chosen, required, _ok, _mode = self._endpoint_route(len(values), values)
        if not chosen or len(chosen) < 2:
            return CollectiveTiming("peer_transfer", len(values), payload, 0.0,
                                    (), "none", 0.0, 0.0, 0.0, "peer", (), (),
                                    False, "no_declared_route")
        participant_set = {entry.rank for entry in chosen}
        try:
            hops, qualified, evidence = _endpoint_hops(
                self.tiers, chosen, required or 0)
        except ModelingError as exc:
            return CollectiveTiming("peer_transfer", len(values), payload, 0.0,
                                    (), "none", 0.0, 0.0, 0.0, "peer", (), (),
                                    False, f"no_declared_route:{exc}")
        per_tier: list[tuple[str, float]] = []
        alpha_total = 0.0
        service_total = 0.0
        switch_bound = 0.0
        endpoint_bound = 0.0
        for tier, _groups, _before, _after in hops:
            if tier.link_members and not any(
                    participant_set <= set(members)
                    for members in tier.link_members):
                qualified = False
                evidence = "disconnected_participants"
            os_factor = (tier.oversubscription
                         if tier.semantics in {"shared", "mesh"} else 1.0)
            service = payload / (tier.bw_gbps * 1000.0) * os_factor
            alpha_total += tier.alpha_us
            service_total += service
            per_tier.append((tier.name, tier.alpha_us + service))
            if tier.switch_capacity_bytes_per_ms is not None:
                switch_bound = max(
                    switch_bound,
                    payload * 1000.0 / tier.switch_capacity_bytes_per_ms)
            for rank, rate in tier.endpoint_rates:
                if int(rank) in participant_set:
                    endpoint_bound = max(endpoint_bound,
                                         payload * 1000.0 / max(1e-12, rate))
        # Startup latency precedes wire/switch serialization: a slow shared cut
        # can never erase the fixed ``alpha`` of the traversed path.
        total = alpha_total + max(service_total, switch_bound, endpoint_bound)
        bottleneck = max(per_tier, key=lambda item: (item[1],
                                                     TIER_ORDER.index(item[0]))) \
            if per_tier else ("none", 0.0)
        return CollectiveTiming(
            "peer_transfer", len(values), payload, total, tuple(per_tier),
            bottleneck[0], float(bottleneck[1]), payload, 0.0, "peer_point_to_point",
            (), tuple(tier.name for tier, *_ in hops), qualified, evidence,
            max_sender_egress_bytes=payload,
            max_receiver_ingress_bytes=payload, cut_bytes=payload,
            receiver_bound_us=max(endpoint_bound, switch_bound))

    def peer_resources(self, payload_bytes: float,
                       endpoints: Sequence[int | RankEndpoint], *,
                       dtype: str | None = None,
                       topology: str | None = None,
                       ) -> tuple[tuple[str, float], ...]:
        """Physical resources occupied by one peer transfer, in microseconds.

        The same route, membership and startup/switch rules as
        :meth:`peer_transfer` apply.  Each traversed shared switch/cut and every
        participant's egress/ingress port is a distinct resource identity, so a
        PP scheduler can serialize transfers that share a physical switch even
        when they belong to different logical stage boundaries.
        """
        del dtype, topology
        payload = _positive("payload_bytes", payload_bytes, strict=False)
        requested = [item.rank if isinstance(item, RankEndpoint) else item
                     for item in endpoints]
        values = tuple(int(value) for value in requested)
        if len(values) < 2 or payload <= 0.0:
            return ()
        chosen, required, _ok, _mode = self._endpoint_route(len(values), values)
        if not chosen or len(chosen) < 2:
            return ()
        participant_set = {entry.rank for entry in chosen}
        try:
            hops, _qualified, _evidence = _endpoint_hops(
                self.tiers, chosen, required or 0)
        except ModelingError:
            return ()
        services: dict[str, float] = {}
        for tier, _groups, _before, _after in hops:
            os_factor = (tier.oversubscription
                         if tier.semantics in {"shared", "mesh"} else 1.0)
            service = payload / (tier.bw_gbps * 1000.0) * os_factor
            if tier.switch_capacity_bytes_per_ms is not None:
                key = f"switch:{tier.name}"
                services[key] = max(
                    services.get(key, 0.0),
                    payload * 1000.0 / tier.switch_capacity_bytes_per_ms)
            if tier.semantics in {"shared", "mesh"}:
                key = f"tier:{tier.name}"
                services[key] = max(services.get(key, 0.0), service)
            for rank in participant_set:
                for direction in ("egress", "ingress"):
                    key = f"port:{rank}:{direction}"
                    services[key] = max(services.get(key, 0.0), service)
            for rank, rate in tier.endpoint_rates:
                if int(rank) in participant_set:
                    port_us = payload * 1000.0 / max(1e-12, rate)
                    for direction in ("egress", "ingress"):
                        key = f"port:{int(rank)}:{direction}"
                        services[key] = max(services.get(key, 0.0), port_us)
        return tuple(sorted(services.items()))

    def peer_cut_demands(self, payload_bytes: float,
                         endpoints: Sequence[int | RankEndpoint], *,
                         dtype: str | None = None,
                         topology: str | None = None) -> dict[str, float]:
        """Per-request bytes a peer transfer places on every shared cut."""
        del dtype, topology
        payload = _positive("payload_bytes", payload_bytes, strict=False)
        requested = [item.rank if isinstance(item, RankEndpoint) else item
                     for item in endpoints]
        values = tuple(int(value) for value in requested)
        if len(values) < 2 or payload <= 0.0:
            return {}
        chosen, required, _ok, _mode = self._endpoint_route(len(values), values)
        if not chosen or len(chosen) < 2:
            return {}
        try:
            hops, _qualified, _evidence = _endpoint_hops(
                self.tiers, chosen, required or 0)
        except ModelingError:
            return {}
        result: dict[str, float] = {}
        for tier, _groups, _before, _after in hops:
            if tier.switch_capacity_bytes_per_ms is not None:
                key = f"switch:{tier.name}"
                result[key] = result.get(key, 0.0) + payload
            if tier.semantics in {"shared", "mesh"}:
                key = f"tier:{tier.name}"
                result[key] = result.get(key, 0.0) + payload
        return result

    def collective(self, kind: str, payload_bytes: float, ranks: int, *,
                   endpoints: Sequence[int | RankEndpoint] | None = None,
                   payloads_by_destination: Sequence[float] | None = None,
                   dtype: str | None = None, topology: str | None = None,
                   **kwargs: Any) -> CollectiveTiming:
        """Price one collective over explicit endpoints (or a fallback span).

        ``dtype``/``topology`` are accepted so a caller can forward its actual
        fixture evidence to either mode; Mode A has no fitted conditions and
        therefore ignores them.
        """
        del dtype, topology, kwargs
        payload = _positive("payload_bytes", payload_bytes, strict=False)
        if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
            raise ModelingError(f"fabric_collective_ranks_invalid:{ranks}")
        if kind not in COLLECTIVES:
            raise ModelingError(f"fabric_collective_unknown:{kind}")
        routed = payloads_by_destination if kind == "all_to_all_v" else None
        if kind == "mc2":
            one = self.collective("all_to_all_v", payload, ranks, endpoints=endpoints,
                                  payloads_by_destination=payloads_by_destination)
            return replace(one, kind="mc2", total_us=2.0 * one.total_us,
                           bottleneck_us=2.0 * one.bottleneck_us,
                           wire_bytes_per_rank=2.0 * one.wire_bytes_per_rank,
                           tier_us=tuple((name, 2.0 * value) for name, value in one.tier_us),
                           rounds=tuple((name, 2 * count) for name, count in one.rounds),
                           effective_bw_gbps=one.effective_bw_gbps / 2.0,
                           method="tree_mc2_dispatch_combine")
        if ranks == 1:
            return CollectiveTiming(kind, ranks, payload, 0.0, (), "none", 0.0, 0.0,
                                    0.0, "trivial", (), (), True, "trivial")
        requested = None if endpoints is None else [
            item.rank if isinstance(item, RankEndpoint) else item for item in endpoints]
        chosen, required, _ok, _mode = self._endpoint_route(ranks, requested)
        if chosen:
            hops, qualified, evidence = _endpoint_hops(self.tiers, chosen, required or 0)
            unmeasured_die = (
                dict(self.assumptions).get("die_link_evidence")
                == "unmeasured_cross_die_absorbed"
                or any(tier.evidence == "unmeasured_die_link" for tier in self.tiers))
            if ((required or 0) >= 1 and unmeasured_die
                    and not any(tier.locality == 1 for tier in self.tiers)):
                qualified, evidence = False, "cross_die_absorbed_unmeasured_die_link"
        else:
            hops = _count_hops(self.tiers, ranks)
            qualified, evidence = True, "count_span"
        return _timing(kind, payload, chosen, hops, method="tree_hierarchical_ring",
                       payloads_by_destination=routed,
                       qualified=qualified, evidence=evidence)

    def mc2_dispatch_combine(self, payload_bytes: float, ranks: int, *,
                             endpoints: Sequence[int | RankEndpoint] | None = None) -> dict[str, Any]:
        dispatch = self.collective("all_to_all_v", payload_bytes, ranks, endpoints=endpoints)
        combine = self.collective("all_to_all_v", payload_bytes, ranks, endpoints=endpoints)
        return {"dispatch": dispatch.to_dict(), "combine": combine.to_dict(),
                "total_us": dispatch.total_us + combine.total_us,
                "bottleneck_tier": max((dispatch, combine),
                                       key=lambda item: item.total_us).bottleneck_tier,
                "method": "tree_mc2_dispatch_combine"}

    @classmethod
    def from_hardware(cls, hardware: Hardware, *, spine_bw_gbps: float | None = None,
                      spine_oversubscription: float = 1.0,
                      pod_roce_oversubscription: float = 1.0) -> "HierarchicalFabricTree":
        """Derive tiers and explicit rank endpoints from physical records."""
        options = dict(hardware.topology_options)
        rpn = max(2, int(hardware.ranks_per_node))
        endpoints = tuple(RankEndpoint(
            rank=index, die=index % max(1, int(hardware.dies_per_package)),
            package=int(die.package), node=int(die.node),
            superpod=int(die.node) // max(1, int(options.get("superpod_size", 0) or
                                                  len(hardware.dies)) // rpn),
            pod=0) for index, die in enumerate(hardware.dies))

        def channel(channel_names: Sequence[str]) -> dict[str, Any]:
            candidates = [link for link in hardware.links if link.channel in channel_names
                          and len(link.members) > 1]
            if not candidates:
                return {"bw": 0.0, "alpha": 0.0, "evidence": "missing",
                        "members": (), "endpoint_rates": (), "switch": None,
                        "semantics": "shared"}
            slowest = min(candidates, key=lambda link: (
                float(link.capacity_bytes_per_ms) * float(link.bw_utilization),
                link.latency_us))
            capacity = float(slowest.capacity_bytes_per_ms) * float(slowest.bw_utilization)
            evidence = "declared_link" if len(candidates) == 1 else "slowest_declared_link"
            members = tuple(tuple(int(rank) for rank in link.members)
                            for link in candidates)
            rates: dict[int, float] = {}
            for link in candidates:
                for rank, rate in link.endpoint_rates:
                    rates[int(rank)] = min(rates.get(int(rank), rate), float(rate))
            switches = [float(link.switch_capacity_bytes_per_ms)
                        for link in candidates
                        if link.switch_capacity_bytes_per_ms is not None]
            semantics = {link.semantics for link in candidates}
            return {"bw": capacity / 1e6, "alpha": float(slowest.latency_us),
                    "evidence": evidence, "members": members,
                    "endpoint_rates": tuple(sorted(rates.items())),
                    "switch": min(switches) if switches else None,
                    "semantics": semantics.pop() if len(semantics) == 1 else "shared"}

        tiers: list[FabricTier] = []
        if hardware.dies_per_package > 1:
            info = channel(("d2d_intra_package",))
            if info["bw"] > 0:
                tiers.append(FabricTier(
                    "intra_package_die", 2, info["alpha"], info["bw"],
                    semantics=info["semantics"], die_pair=True,
                    evidence=info["evidence"], link_members=info["members"],
                    endpoint_rates=info["endpoint_rates"],
                    switch_capacity_bytes_per_ms=info["switch"]))
        info = channel(("d2d_intra_node", "d2d_intra_8p", "d2d_intra_4p"))
        if info["bw"] > 0:
            tiers.append(FabricTier(
                "intra_node_hccs", rpn, info["alpha"], info["bw"],
                semantics=info["semantics"], evidence=info["evidence"],
                link_members=info["members"],
                endpoint_rates=info["endpoint_rates"],
                switch_capacity_bytes_per_ms=info["switch"]))
        superpod_size = int(options.get("superpod_size", 0) or 0)
        superpod_bw = float(options.get("superpod_bw_gbps", 0.0) or 0.0)
        if superpod_size > rpn and superpod_bw > 0:
            alpha = float(options.get("superpod_latency_us",
                                      options.get("inter_node_latency_us", 0.0)) or 0.0)
            tiers.append(FabricTier("superpod_hccs", superpod_size, alpha, superpod_bw,
                                    evidence="declared_topology_option"))
        pod_bw = float(options.get("inter_pod_bw_gbps", 0.0) or 0.0)
        pod_alpha = float(options.get("inter_pod_latency_us", 0.0) or 0.0)
        pod_evidence = "declared_topology_option"
        pod_members: tuple[tuple[int, ...], ...] = ()
        pod_rates: tuple[tuple[int, float], ...] = ()
        pod_switch: float | None = None
        pod_semantics = "shared"
        if pod_bw <= 0:
            fallback = [link for link in hardware.links
                        if link.channel == "d2d_inter_node" and len(link.members) > 1]
            if fallback and len({item.node for item in endpoints}) > 1:
                slowest = min(fallback, key=lambda link: link.capacity_bytes_per_ms)
                pod_bw = float(slowest.capacity_bytes_per_ms) * float(slowest.bw_utilization) / 1e6
                pod_alpha = float(slowest.latency_us)
                # Preserve every physical cut of the inter-node tier: a
                # converted tier that drops members/endpoint rates/switch
                # capacity prices a 3-hop RoCE path as a 1-us local wire.
                pod_members = tuple(tuple(int(rank) for rank in link.members)
                                    for link in fallback)
                rates: dict[int, float] = {}
                for link in fallback:
                    for rank, rate in link.endpoint_rates:
                        rates[int(rank)] = min(rates.get(int(rank), rate), float(rate))
                pod_rates = tuple(sorted(rates.items()))
                switches = [float(link.switch_capacity_bytes_per_ms)
                            for link in fallback
                            if link.switch_capacity_bytes_per_ms is not None]
                pod_switch = min(switches) if switches else None
                semantics = {link.semantics for link in fallback}
                pod_semantics = (semantics.pop() if len(semantics) == 1
                                 else "shared")
                pod_evidence = "declared_inter_node_link"
        max_endpoint = max((item.superpod + 1 for item in endpoints), default=1)
        if pod_bw > 0:
            last = max((tier.domain_ranks for tier in tiers), default=rpn)
            domain = max(last * 2, superpod_size or len(hardware.dies), rpn * max_endpoint)
            tiers.append(FabricTier("pod_roce", domain, pod_alpha, pod_bw,
                                    oversubscription=pod_roce_oversubscription,
                                    evidence=pod_evidence,
                                    link_members=pod_members,
                                    endpoint_rates=pod_rates,
                                    switch_capacity_bytes_per_ms=pod_switch,
                                    semantics=pod_semantics))
        if spine_bw_gbps is not None:
            tiers.append(FabricTier("spine_roce", max(
                (tier.domain_ranks for tier in tiers), default=1) * 2, 10.0,
                spine_bw_gbps, oversubscription=spine_oversubscription,
                evidence="declared_counterfactual"))
        tiers.sort(key=lambda tier: tier.domain_ranks)
        if not tiers:
            raise ModelingError("fabric_hardware_has_no_declared_tier")
        notes = [("source", "hardware_links_and_topology_options"),
                 ("pod_roce_oversubscription", str(pod_roce_oversubscription))]
        if hardware.dies_per_package > 1 and not any(
                tier.name == "intra_package_die" and tier.evidence != "unmeasured_die_link"
                for tier in tiers):
            notes.append(("die_link_evidence", "unmeasured_cross_die_absorbed"))
        return cls(tuple(tiers), endpoints=endpoints, assumptions=tuple(notes))

    def to_dict(self) -> dict[str, Any]:
        return {"mode": "hierarchical_tree", "source": self.source,
                "rank_placement": list(self.rank_placement),
                "dual_die_affinity": self.dual_die_affinity,
                "tiers": [tier.to_dict() for tier in self.tiers],
                "endpoints": [item.to_dict() for item in self.endpoints],
                "assumptions": dict(self.assumptions)}

_CALIBRATED_NAMES = frozenset({"CalibratedFabricProfile", "CalibratedTier",
                               "ProbePoint"})

def __getattr__(name: str):
    """Lazy Mode B export avoids a circular import when it is imported first."""
    if name in _CALIBRATED_NAMES:
        from mindie_sim.fabric_calibrated import (CalibratedFabricProfile,
                                                  CalibratedTier, ProbePoint)
        value = {"CalibratedFabricProfile": CalibratedFabricProfile,
                 "CalibratedTier": CalibratedTier, "ProbePoint": ProbePoint}[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from mindie_sim.fabric_scaling import _floor, project_case, scale_hardware  # noqa: E402

__all__ = ["COLLECTIVES", "FabricRangeError", "TIER_ORDER", "CalibratedFabricProfile",
           "CalibratedTier", "CollectiveTiming", "FabricTier", "HierarchicalFabricTree",
           "ProbePoint", "RankEndpoint", "project_case", "scale_hardware"]
