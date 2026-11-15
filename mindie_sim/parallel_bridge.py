"""Executable PCP-prefill to DCP-decode ownership bridge.

[Role]
  Partition logical KV/index/state pages and bytes across source lanes and
  destination shards, or replicate one conserved copy into every destination
  replica, and convert the resulting edge plan into the unique remap payload.

[Invariants]
  * Canonical post-gather PCP producers and destination DCP buckets use
    absolute block IDs; automatic buckets may legitimately be empty.
  * Replicated mode sends each source's owned slice to every destination
    replica: each replica receives exactly one logical payload and owns every
    logical page.  It never multiplies one replica's payload by ``S``.
  * Sharded mode's edge intersections conserve the payload exactly.
  * A supplied destination page list that does not cover the source pages is a
    typed rejection, never a certified zero-transfer hit.  ``None`` is absent
    evidence; an explicit empty tuple is a positive no-page claim and rejects
    when the logical universe is non-empty.
  * The lifetime plan derives source retention, in-flight handoff staging,
    destination TP/DCP replication and remap peaks from those exact edges, so
    one pool cannot admit a transfer larger than its HBM headroom.
  * ``eplb_rebalance_peak_bytes`` is a per-rank bound.  It is additive to the
    simultaneously live handoff/retention/remap storage and is never divided by
    PP stage weights downstream.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable records; caller specs untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec


def _page_tuple(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"bridge_page_id_invalid:{value}")
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ModelingError(f"bridge_page_id_duplicate:{result}")
    return tuple(result)


def validate_page_plan(source_pages: Sequence[int] | None,
                       destination_pages: Sequence[int] | None, *,
                       required_pages: int | None = None) -> tuple[
                           tuple[int, ...], tuple[int, ...]]:
    """Validate page ownership evidence without collapsing ``None``/``()``.

    ``None`` is absent evidence (a caller may supply a default plan); an
    explicit empty tuple is a positive claim that the owner has no pages.  When
    the logical universe is known, every supplied id must be inside it and an
    explicit empty destination with a non-empty universe is rejected.  The
    zero-transfer shortcut additionally requires *complete* destination
    coverage; that check belongs to the caller that certifies the hit.
    """
    source = _page_tuple(source_pages)
    destination = _page_tuple(destination_pages)
    if required_pages is not None:
        if isinstance(required_pages, bool) or not isinstance(required_pages, int) \
                or required_pages < 0:
            raise ModelingError(
                f"bridge_page_plan_required_invalid:{required_pages}")
        count = required_pages
        universe = set(range(count))
        for label, values in (("source", source), ("destination", destination)):
            unknown = tuple(sorted(set(values) - universe))
            if unknown:
                raise ModelingError(
                    f"bridge_page_plan_out_of_range:{label}:{unknown}")
        if count > 0 and destination_pages is not None and not destination:
            raise ModelingError("bridge_page_plan_destination_empty")
    if source_pages is not None and destination_pages is not None:
        missing = tuple(sorted(set(source) - set(destination)))
        if missing:
            raise ModelingError(f"bridge_page_plan_mismatch:{missing}")
    return source, destination


@dataclass(frozen=True)
class BridgeLifetimePlan:
    """Peak HBM a bridge's page/edge lifetime imposes on one instance rank.

    The plan is derived from the actual ownership edges, not a scalar reserve:
    source pages retained until handoff acknowledgement, in-flight handoff
    staging, destination retention/replication, and the remap working set.
    """

    ownership_mode: str
    page_allocation_bytes: float
    handoff_staging_bytes: float
    source_retention_bytes: float
    destination_retention_bytes: float
    replication_peak_bytes: float
    remap_peak_bytes: float
    eplb_rebalance_peak_bytes: float = 0.0
    page_owner_ranks: tuple[int, ...] = ()
    prefill_rank_peaks: tuple[float, ...] = ()
    decode_rank_peaks: tuple[float, ...] = ()

    def phase_rank_peak_bytes(self, phase: str) -> tuple[float, ...]:
        values = self.prefill_rank_peaks if phase == "prefill" else self.decode_rank_peaks
        return tuple(value + self.eplb_rebalance_peak_bytes for value in values)

    def phase_peak_bytes(self, phase: str) -> float:
        exact = self.phase_rank_peak_bytes(phase)
        if exact:
            return max(exact)
        if phase == "prefill":
            retained = self.source_retention_bytes
        elif phase == "decode":
            retained = self.destination_retention_bytes
        else:
            retained = max(self.source_retention_bytes,
                           self.destination_retention_bytes)
        live = self.handoff_staging_bytes + retained
        # Remap is a phased relocation of the same retained page extents, so it
        # shares storage with the live source/destination copies.  The EPLB
        # rebalance buffer is an *independent*, simultaneously live allocation
        # on every participating rank: it is added in full, never hidden below
        # ``max`` and never divided across PP stages.
        return float(max(live, self.remap_peak_bytes)
                     + self.eplb_rebalance_peak_bytes)

    def peak_bytes(self) -> float:
        return float(max(self.phase_peak_bytes("prefill"),
                         self.phase_peak_bytes("decode")))

    def to_dict(self) -> dict[str, Any]:
        return {"ownership_mode": self.ownership_mode,
                "page_allocation_bytes": self.page_allocation_bytes,
                "handoff_staging_bytes": self.handoff_staging_bytes,
                "source_retention_bytes": self.source_retention_bytes,
                "destination_retention_bytes": self.destination_retention_bytes,
                "replication_peak_bytes": self.replication_peak_bytes,
                "remap_peak_bytes": self.remap_peak_bytes,
                "eplb_rebalance_peak_bytes": self.eplb_rebalance_peak_bytes,
                "prefill_peak_bytes": self.phase_peak_bytes("prefill"),
                "decode_peak_bytes": self.phase_peak_bytes("decode"),
                "peak_bytes": self.peak_bytes(),
                "page_owner_ranks": list(self.page_owner_ranks),
                "prefill_rank_peaks": self.phase_rank_peak_bytes("prefill"),
                "decode_rank_peaks": self.phase_rank_peak_bytes("decode")}


def kv_rank_fraction(model: Any, parallel: ParallelSpec) -> float:
    """Fraction of the unique logical payload one physical rank retains.

    GQA/MQA keeps ``max(1, ceil(kv_heads/tp))`` local heads, so replicated
    heads are charged in full (never divided by ``tp``).  MLA latent rows are
    replicated across TP, while recurrent/conv state is TP-sharded.  An unknown
    or absent KV geometry returns the conservative full-payload factor 1.0.
    """
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    if not plans:
        return 1.0
    tp = max(1, int(parallel.tp))
    fractions: list[float] = []
    for plan in plans:
        attention = str(getattr(plan, "attention", "") or "").lower()
        if attention in {"kda", "gdn"}:
            fractions.append(1.0 / float(tp))
        elif attention in {"gqa", "projection_only"}:
            heads = max(0, int(getattr(plan, "num_kv_heads", 0) or 0))
            if heads <= 0:
                fractions.append(1.0)
            else:
                local = max(1, int(math.ceil(heads / float(tp))))
                fractions.append(min(1.0, local / float(heads)))
        elif attention == "mla":
            fractions.append(1.0)
        else:
            fractions.append(1.0)
    return max(fractions, default=1.0)


def bridge_lifetime_plan(bridge: PhaseBridge, *,
                         eplb_rebalance_peak_bytes: float = 0.0,
                         case: Any = None, transfer: Any = None,
                         ) -> BridgeLifetimePlan:
    """Exact peak bound of the bridge's allocation/handoff/remap lifetimes."""
    metadata = max(0.0, float(bridge.metadata_bytes))
    # The replication/sharding factor is declared by the caller from the actual
    # KV head geometry (GQA/MQA/MLA).  Its default is the conservative full
    # logical payload per rank; the bridge never optimistically divides by TP.
    source_factor = max(0.0, float(bridge.source_rank_fraction))
    destination_factor = max(0.0, float(bridge.destination_rank_fraction))
    source = ((max(bridge.source_lane_bytes, default=0.0) * source_factor)
              if bridge.source_lane_bytes else 0.0)
    destination = ((max(bridge.destination_lane_bytes, default=0.0)
                    * destination_factor)
                   if bridge.destination_lane_bytes else 0.0)
    edge = ((max((float(row[2]) for row in bridge.edge_bytes), default=0.0)
             * destination_factor) if bridge.edge_bytes else 0.0)
    remap_total = float(bridge_remap_bytes(bridge)["total_bytes"])
    remap = max(0.0, remap_total - metadata) * destination_factor + metadata
    source_peak = source + metadata
    destination_peak = destination + metadata
    handoff = edge + metadata
    replication = (destination_peak
                   if bridge.ownership_mode == "replicated" else 0.0)
    exact = ((), ())
    owners = tuple(int(rank) for rank in bridge.page_owner_ranks)
    if bridge.layer_extents and case is None:
        raise ModelingError("bridge_lifetime_requires_model_geometry")
    if case is not None and transfer is None and bridge.layer_extents:
        transfer = {"shard_bytes_aggregate": bridge.kv_bytes + bridge.index_bytes,
                    "fixed_state_transfer_bytes": bridge.state_bytes,
                    "layer_extents": bridge.layer_extents}
    if case is not None and transfer is not None:
        from mindie_sim.xpyd_bridge_ownership import rank_lifetime_peaks
        ranks = rank_lifetime_peaks(case, bridge, transfer)
        exact = ranks["prefill"], ranks["decode"]
        owners = tuple(sorted(ranks["owners"]))
        source_peak, destination_peak = ranks["source_retention"], ranks["destination_retention"]
        handoff = ranks["handoff_staging"]
        replication = destination_peak if bridge.ownership_mode == "replicated" else 0.0
        remap = max(*exact[0], *exact[1])
    return BridgeLifetimePlan(
        bridge.ownership_mode,
        max(source_peak, destination_peak), handoff, source_peak,
        destination_peak, replication,
        remap if exact[0] else max(remap, source_peak + destination_peak),
        max(0.0, float(eplb_rebalance_peak_bytes)),
        owners, *exact)


def _partition_pages(pages: tuple[int, ...], lanes: int
                     ) -> tuple[tuple[int, ...], ...]:
    if not pages or lanes < 1:
        return tuple(() for _ in range(max(1, lanes)))
    buckets: list[list[int]] = [[] for _ in range(lanes)]
    for page in pages:
        buckets[page % lanes].append(page)
    return tuple(tuple(bucket) for bucket in buckets)


@dataclass(frozen=True)
class PhaseBridge:
    prefill_plan: str
    decode_plan: str
    pcp: int
    dcp: int
    epoch: int
    kv_bytes: float
    index_bytes: float
    state_bytes: float
    metadata_bytes: float
    barrier: bool
    source_lane: int
    destination_lanes: tuple[int, ...]
    source_lanes: tuple[int, ...] = ()
    lane_map: tuple[tuple[int, int], ...] = ()
    ownership_mode: str = "sharded"
    edge_bytes: tuple[tuple[int, int, float], ...] = ()
    source_lane_bytes: tuple[float, ...] = ()
    destination_lane_bytes: tuple[float, ...] = ()
    destination_pages: tuple[tuple[int, ...], ...] = ()
    source_pages: tuple[tuple[int, ...], ...] = ()
    source_tp: int = 1
    destination_tp: int = 1
    #: Fraction of the unique logical payload one physical rank retains.  The
    #: default is the conservative full payload (never an optimistic ``1/tp``).
    source_rank_fraction: float = 1.0
    destination_rank_fraction: float = 1.0
    page_owner_ranks: tuple[int, ...] = ()
    source_spec: ParallelSpec | None = None
    destination_spec: ParallelSpec | None = None
    layer_extents: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {key: (value.to_dict() if isinstance(value, ParallelSpec) else value)
                for key, value in vars(self).items()}


def phase_bridge(source: ParallelSpec, destination: ParallelSpec, *,
                 kv_bytes: float, index_bytes: float = 0.0,
                 state_bytes: float = 0.0, epoch: int = 0,
                 lane_bw_gbps: float = 100.0,
                 page_ids: Sequence[int] | None = None,
                 replicated: bool = False,
                 source_page_ids: Sequence[int] | None = None,
                 destination_page_ids: Sequence[int] | None = None,
                 source_rank_fraction: float = 1.0,
                 destination_rank_fraction: float = 1.0,
                 page_owner_ranks: Sequence[int] | None = None,
                 cached_page_ids: Sequence[int] | None = None,
                 layer_extents: Sequence[Any] = ()) -> PhaseBridge:
    """One conserved ownership plan for sharded or replicated bridges.

    ``page_ids`` is the authoritative required *missing-page* universe.  An
    explicit source or destination ownership list is reconciled against that
    universe; it can never replace it.  ``cached_page_ids`` names pages already
    resident on the destination (a verified cache prefix), which are accepted
    in a destination list but carry no handoff bytes.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ModelingError(f"bridge_epoch_invalid:{epoch}")
    pcp = max(1, int(source.pcp))
    # A single active PCP lane has no replica remap: DCP still owns page % CP
    # within lane zero. Keep the legacy explicit scalar replication contract
    # for callers that do not select the PCP replica policy.
    if destination.pcp_lane_policy == "replica_lanes" and destination.active_pcp_lanes == 1:
        replicated = False
    dcp = max(1, int(destination.cp or 0),
              int(getattr(destination, "active_pcp_lanes", 1) or 1))
    if replicated and destination.active_pcp_lanes > 1:
        dcp = int(destination.active_pcp_lanes)
    destination_lanes = tuple(range(dcp))
    payload = (max(0.0, float(kv_bytes)) + max(0.0, float(index_bytes))
               + max(0.0, float(state_bytes)))
    page_values = _page_tuple(page_ids)
    source_explicit, destination_explicit = validate_page_plan(
        source_page_ids, destination_page_ids)
    if page_ids is not None:
        universe = tuple(page_values)
    else:
        universe = tuple(dict.fromkeys(source_explicit + destination_explicit))
    if not universe and payload > 0.0:
        if page_ids is not None:
            raise ModelingError("bridge_page_plan_universe_empty")
        universe = tuple(range(max(1, pcp * dcp)))
    required = set(universe)
    cached = set(_page_tuple(cached_page_ids))
    if page_ids is not None and cached & required:
        raise ModelingError(
            f"bridge_page_plan_cached_overlap:{tuple(sorted(cached & required))}")
    # Already-cached destination pages are legal ownership evidence but are not
    # part of the required missing-page universe and carry no transfer edge.
    destination_universe = (required | cached if page_ids is not None
                            else set(required))
    if payload > 0.0:
        if source_page_ids is not None and not source_explicit:
            raise ModelingError("bridge_page_plan_source_empty")
        if destination_page_ids is not None and not destination_explicit:
            raise ModelingError("bridge_page_plan_destination_empty")
    if destination_page_ids is not None and required and not destination_explicit:
        raise ModelingError("bridge_page_plan_destination_empty")
    for label, declared, values, allowed in (
            ("source", source_page_ids, source_explicit, required),
            ("destination", destination_page_ids, destination_explicit,
             destination_universe)):
        if declared is None:
            continue
        unknown = tuple(sorted(set(values) - allowed))
        if unknown:
            raise ModelingError(
                f"bridge_page_plan_out_of_range:{label}:{unknown}")
        if (label == "source" and payload <= 0.0):
            continue
        missing = tuple(sorted(required - set(values)))
        if missing:
            raise ModelingError(
                f"bridge_page_plan_{label}_incomplete:{missing}")
    # ``page_ids`` is the required universe.  An explicit destination list may
    # also mention the verified cached prefix, but only its required pages are
    # partitioned into transfer edges.
    source_order = (source_explicit if source_page_ids is not None
                    else universe)
    destination_order = (tuple(page for page in destination_explicit
                               if page in required)
                         if destination_page_ids is not None else universe)
    if not source_order and source_page_ids is None:
        source_order = universe
    if not destination_order and destination_page_ids is None:
        destination_order = universe
    count = len(universe)
    page_bytes = ({page: payload * ((index + 1) / float(count)
                                    - index / float(count))
                   for index, page in enumerate(universe)}
                  if count > 0 else {})
    if layer_extents and all(row.get("page_extents") for row in layer_extents):
        page_bytes = {page: 0.0 for page in universe}
        for layer in layer_extents:
            for row in layer["page_extents"]:
                page_bytes[row["page"]] += sum(float(row[key]) for key in ("kv_bytes", "index_bytes", "state_bytes"))
        if not math.isclose(sum(page_bytes.values()), payload, rel_tol=1e-12, abs_tol=1e-6):
            raise ModelingError("bridge_page_extent_not_conserved")
    source_buckets = _partition_pages(tuple(source_order), pcp)
    destination_buckets = _partition_pages(tuple(destination_order), dcp)
    source_lane_bytes = [float(sum(page_bytes.get(page, 0.0)
                                   for page in bucket))
                         for bucket in source_buckets]
    destination_lane_bytes = [0.0] * dcp
    edges: list[tuple[int, int, float]] = []
    if replicated:
        for lane, bucket in enumerate(source_buckets):
            edge = source_lane_bytes[lane]
            if edge <= 0.0:
                continue
            for dest in range(dcp):
                edges.append((lane, dest, edge))
                destination_lane_bytes[dest] += edge
        destination_pages = tuple(universe for _ in range(dcp))
    else:
        for lane, bucket in enumerate(source_buckets):
            owned = set(bucket)
            for dest, target in enumerate(destination_buckets):
                shared = owned & set(target)
                edge = float(sum(page_bytes.get(page, 0.0)
                                 for page in shared))
                if edge <= 0.0:
                    continue
                edges.append((lane, dest, edge))
                destination_lane_bytes[dest] += edge
        destination_pages = tuple(tuple(bucket)
                                  for bucket in destination_buckets)
    lane_map = tuple((lane, dest) for lane, dest, _bytes in edges)
    source_pages = tuple(tuple(bucket) for bucket in source_buckets)
    for label, value in (("source_rank_fraction", source_rank_fraction),
                         ("destination_rank_fraction",
                          destination_rank_fraction)):
        number = float(value)
        if not math.isfinite(number) or number <= 0.0 or number > 1.0:
            raise ModelingError(f"bridge_rank_fraction_invalid:{label}:{value}")
    owners = (tuple(int(rank) for rank in destination_lanes)
              if page_owner_ranks is None
              else tuple(int(rank) for rank in page_owner_ranks))
    if any(rank < 0 for rank in owners):
        raise ModelingError("bridge_page_owner_invalid")
    del lane_bw_gbps
    return PhaseBridge(
        str(source.cp_plan or "pcp"), str(destination.cp_plan or "dcp"),
        pcp, dcp, int(epoch), float(kv_bytes), float(index_bytes),
        float(state_bytes), 4096.0 * max(1, len(destination_lanes)),
        True, 0, destination_lanes, tuple(range(pcp)), lane_map,
        "replicated" if replicated else "sharded",
        tuple(edges), tuple(source_lane_bytes), tuple(destination_lane_bytes),
        destination_pages, source_pages,
        max(1, int(source.tp)), max(1, int(destination.tp)),
        float(source_rank_fraction), float(destination_rank_fraction), owners,
        source, destination, tuple(layer_extents))


def bridge_remap_bytes(bridge: PhaseBridge) -> dict[str, float]:
    """Unique payload a replica-lane activation must move (metadata included)."""
    lanes = max(1, len(bridge.destination_lanes))
    payload = (max(0.0, bridge.kv_bytes) + max(0.0, bridge.index_bytes)
               + max(0.0, bridge.state_bytes))
    moved = (float(sum(edge[2] for edge in bridge.edge_bytes))
             if bridge.edge_bytes else payload)
    return {"destination_lanes": float(lanes),
            "kv_bytes": max(0.0, bridge.kv_bytes),
            "index_bytes": max(0.0, bridge.index_bytes),
            "state_bytes": max(0.0, bridge.state_bytes),
            "metadata_bytes": float(bridge.metadata_bytes),
            "mode": bridge.ownership_mode,
            "total_bytes": moved + float(bridge.metadata_bytes)}


__all__ = ["BridgeLifetimePlan", "PhaseBridge", "bridge_lifetime_plan",
           "bridge_remap_bytes", "kv_rank_fraction", "phase_bridge",
           "validate_page_plan"]
