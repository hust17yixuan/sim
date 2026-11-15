"""Physical connector service for a complete P/D cache handoff.

[Role] Bind declared connector framing, propagation and capacity to topology.
[Invariants] Endpoints 0/1 denote prefill/decode pools, not inferred model ranks.
  One handoff is one logical message; wire bytes exclude duplicated capacity
  demands. HBM work is supplied separately by each pool's representative shard.
  Optional chunk/credit/startup fields are explicit already-normalized T06
  services; absent fields keep the legacy whole-message path. A finite receiver
  credit is an Engine transport pool shared by plans on the same Engine, never
  extra wire work. Source readiness, destination grant and consumer release are
  distinct events; both transport branches consume the same common readiness
  boundary and neither branch drops a caller-provided event.
[Timeouts & Locks] Constant-sized connector graph and aggregate counters only.
[Side-Effect Policy] Submits tasks only to the caller's in-memory engine.
"""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from .communication import MessageFlow
from .topology import Topology, TransportOptions


class _AliasedTopology:
    """Read view of a connector topology with its historical public names."""

    def __init__(self, topology, alias):
        self._topology = topology
        self._alias = alias
        self._resources = tuple(
            replace(resource, name=alias.get(resource.name, resource.name))
            for resource in topology.resources())
        self._resource_rates = {resource.name: resource.rate
                                for resource in self._resources}

    def resources(self):
        return self._resources

    def __getattr__(self, name):
        return getattr(self._topology, name)


def _non_negative_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"{label}_invalid:{value}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return result


@dataclass(frozen=True)
class PDTransfer:
    """One directed physical-rank shard inside a request handoff."""

    source_rank: int
    destination_rank: int
    payload_bytes: float
    source_stage: int = 0
    destination_stage: int = 0
    source_layers: tuple[int, ...] = ()
    destination_layers: tuple[int, ...] = ()
    component: str = "kv"
    source_version: int = 0
    destination_version: int = 0
    source_heads: tuple[int, int] = ()
    destination_heads: tuple[int, int] = ()
    source_token_span: tuple[int, int] = ()
    destination_token_span: tuple[int, int] = ()
    producer: str = ""
    output: str = ""
    #: Physical destination domain.  ``hbm`` is a decode NPU receive buffer;
    #: ``host`` is the shared Host DRAM copy made once by the SFA RD2H
    #: connector.  A host transfer still names the decode rank whose replica
    #: owns that host buffer, but it does not consume that rank's HBM footprint.
    destination_tier: str = "hbm"
    source_storage: str = ""
    destination_storage: str = ""
    #: Half-open byte ranges inside the named source/destination storage.
    source_byte_range: tuple[float, float] = ()
    destination_byte_range: tuple[float, float] = ()
    source_scope: str = ""
    destination_scope: str = ""

    def __post_init__(self):
        if (isinstance(self.source_rank, bool)
                or not isinstance(self.source_rank, int)
                or isinstance(self.destination_rank, bool)
                or not isinstance(self.destination_rank, int)):
            raise ModelingError("pd_transfer_requires_integer_ranks")
        value = _non_negative_number(self.payload_bytes, "pd_transfer_payload")
        if value <= 0:
            raise ModelingError("pd_transfer_payload_requires_positive_bytes")
        for name in ("source_stage", "destination_stage"):
            if (isinstance(getattr(self, name), bool)
                    or not isinstance(getattr(self, name), int)
                    or getattr(self, name) < 0):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("source_layers", "destination_layers"):
            value = getattr(self, name)
            if (not isinstance(value, tuple)
                    or any(isinstance(layer, bool) or not isinstance(layer, int)
                           or layer < 0 for layer in value)):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        if not isinstance(self.component, str) or not self.component:
            raise ModelingError("pd_transfer_component_invalid")
        for name in ("source_version", "destination_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("source_heads", "destination_heads",
                     "source_token_span", "destination_token_span"):
            value = getattr(self, name)
            if value and (not isinstance(value, tuple) or len(value) != 2
                          or any(isinstance(item, bool) or not isinstance(item, int)
                                 or item < 0 for item in value)
                          or value[0] > value[1]):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("producer", "output"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        if self.destination_tier not in {"hbm", "host"}:
            raise ModelingError(
                f"pd_transfer_destination_tier_invalid:{self.destination_tier}")
        for name in ("source_storage", "destination_storage",
                     "source_scope", "destination_scope"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ModelingError(f"pd_transfer_{name}_invalid")
        for name in ("source_byte_range", "destination_byte_range"):
            value = getattr(self, name)
            if not value:
                continue
            if (not isinstance(value, tuple) or len(value) != 2
                    or any(isinstance(item, bool)
                           or not isinstance(item, (int, float))
                           or not math.isfinite(float(item)) or item < 0
                           for item in value)
                    or value[0] > value[1]):
                raise ModelingError(f"pd_transfer_{name}_invalid")


@dataclass(frozen=True)
class PDHandoff:
    """Actual request/rank/stage/chunk/destination-receive binding.

    ``transfers`` own the directed physical shards.  ``receive_capacity_bytes``
    and ``receive_buffer_ids`` bind each destination rank to the decode pool's
    physical receive allocation; no global endpoint is invented.  ``memory_work``
    carries the prefill/decode stage HBM demands separately from wire payload.
    A transfer with ``destination_tier == "host"`` is the single SFA RD2H copy
    into shared Host DRAM: it is real wire/source work but not that rank's HBM
    receive footprint.  C (pool budget) and F (this call's footprint) therefore
    stay separate for both HBM and host domains.
    """

    request_id: str
    transfers: tuple[PDTransfer, ...]
    memory_work: tuple[tuple[str, float], ...] = ()
    receive_capacity_bytes: tuple[tuple[int, float], ...] = ()
    receive_buffer_ids: tuple[tuple[int, str], ...] = ()
    receive_pool_capacity_bytes: tuple[tuple[int, float], ...] = ()
    receive_host_bytes: tuple[tuple[int, float], ...] = ()

    def __post_init__(self):
        if not self.request_id:
            raise ModelingError("pd_handoff_requires_request_id")
        if not self.transfers:
            return
        if any(not isinstance(item, PDTransfer) for item in self.transfers):
            raise ModelingError("pd_handoff_transfer_record_invalid")
        totals = defaultdict(float)
        host_totals = defaultdict(float)
        for item in self.transfers:
            if item.destination_tier == "host":
                host_totals[item.destination_rank] += float(item.payload_bytes)
            else:
                totals[item.destination_rank] += float(item.payload_bytes)
        capacities = dict(self.receive_capacity_bytes)
        if len(capacities) != len(self.receive_capacity_bytes):
            raise ModelingError("pd_handoff_duplicate_receive_capacity")
        for rank, value in capacities.items():
            if (isinstance(rank, bool) or not isinstance(rank, int)
                    or not math.isfinite(float(value)) or float(value) <= 0):
                raise ModelingError(
                    f"pd_handoff_receive_capacity_invalid:{rank}:{value}")
        buffers = dict(self.receive_buffer_ids)
        if len(buffers) != len(self.receive_buffer_ids):
            raise ModelingError("pd_handoff_duplicate_receive_buffer")
        for rank, value in buffers.items():
            if (isinstance(rank, bool) or not isinstance(rank, int)
                    or not isinstance(value, str) or not value):
                raise ModelingError(
                    f"pd_handoff_receive_buffer_invalid:{rank}:{value}")
        for rank, value in capacities.items():
            if rank not in totals:
                raise ModelingError(f"pd_handoff_capacity_without_destination:{rank}")
            if value + 1e-9 < totals[rank]:
                raise ModelingError(
                    "pd_handoff_receive_capacity_below_physical_shard:"
                    f"{rank}:{value}<{totals[rank]}")
        pool_capacities = dict(self.receive_pool_capacity_bytes)
        if len(pool_capacities) != len(self.receive_pool_capacity_bytes):
            raise ModelingError("pd_handoff_duplicate_pool_capacity")
        for rank, value in pool_capacities.items():
            if rank not in capacities:
                raise ModelingError(
                    f"pd_handoff_pool_capacity_without_destination:{rank}")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ModelingError(
                    f"pd_handoff_pool_capacity_invalid:{rank}:{value}")
            # C may legitimately be smaller than this call's F: that is a
            # real backpressure fact, not a record error.  The whole-message
            # submit refuses it with zero issue in submit_handoff().
        for key, value in self.memory_work:
            if not isinstance(key, str) or not key:
                raise ModelingError("pd_handoff_memory_resource_invalid")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ModelingError(
                    f"pd_handoff_memory_work_invalid:{key}:{value}")
        host_capacity = dict(self.receive_host_bytes)
        if len(host_capacity) != len(self.receive_host_bytes):
            raise ModelingError("pd_handoff_duplicate_host_capacity")
        for rank, value in host_capacity.items():
            if (isinstance(rank, bool) or not isinstance(rank, int)
                    or not math.isfinite(float(value)) or float(value) <= 0):
                raise ModelingError(
                    f"pd_handoff_host_capacity_invalid:{rank}:{value}")
            if rank not in host_totals:
                raise ModelingError(
                    f"pd_handoff_host_capacity_without_host_transfer:{rank}")
            if float(value) + 1e-9 < host_totals[rank]:
                raise ModelingError(
                    "pd_handoff_host_capacity_below_host_shard:"
                    f"{rank}:{value}<{host_totals[rank]}")

    @classmethod
    def build(cls, request_id, transfers, *, memory_work=(),
              receive_capacity_bytes=None, receive_buffer_ids=None,
              receive_pool_capacity_bytes=None, receive_host_bytes=None):
        transfers = tuple(transfers)
        totals = defaultdict(float)
        host_totals = defaultdict(float)
        for item in transfers:
            if item.destination_tier == "host":
                host_totals[item.destination_rank] += float(item.payload_bytes)
            else:
                totals[item.destination_rank] += float(item.payload_bytes)
        if receive_capacity_bytes is None:
            capacity = tuple(sorted(totals.items()))
        elif isinstance(receive_capacity_bytes, Mapping):
            capacity = tuple(sorted(
                (int(rank), float(value))
                for rank, value in receive_capacity_bytes.items()))
        else:
            capacity = tuple(sorted(
                (int(rank), float(value)) for rank, value in receive_capacity_bytes))
        if receive_buffer_ids is None:
            buffers = tuple((rank, f"pd:{request_id}:rank{rank}")
                            for rank in sorted(totals))
        elif isinstance(receive_buffer_ids, Mapping):
            buffers = tuple(sorted(
                (int(rank), str(value)) for rank, value in receive_buffer_ids.items()))
        else:
            buffers = tuple(sorted(
                (int(rank), str(value)) for rank, value in receive_buffer_ids))
        memory = (tuple(memory_work.items()) if isinstance(memory_work, Mapping)
                  else tuple(memory_work))
        if receive_pool_capacity_bytes is None:
            pool_capacity = capacity
        elif isinstance(receive_pool_capacity_bytes, Mapping):
            pool_capacity = tuple(sorted(
                (int(rank), float(value))
                for rank, value in receive_pool_capacity_bytes.items()))
        else:
            pool_capacity = tuple(sorted(
                (int(rank), float(value))
                for rank, value in receive_pool_capacity_bytes))
        if receive_host_bytes is None:
            host = tuple(sorted(host_totals.items()))
        elif isinstance(receive_host_bytes, Mapping):
            host = tuple(sorted(
                (int(rank), float(value))
                for rank, value in receive_host_bytes.items()))
        else:
            host = tuple(sorted(
                (int(rank), float(value)) for rank, value in receive_host_bytes))
        return cls(str(request_id), transfers, memory, capacity, buffers,
                   pool_capacity, host)

    @property
    def ranks(self):
        return tuple(sorted({item.source_rank for item in self.transfers}
                            | {item.destination_rank for item in self.transfers}))

    def flows(self):
        return tuple(MessageFlow(item.source_rank, item.destination_rank,
                                 float(item.payload_bytes))
                     for item in self.transfers)

    def hbm_transfers(self):
        return tuple(item for item in self.transfers
                     if item.destination_tier == "hbm")

    def host_transfers(self):
        return tuple(item for item in self.transfers
                     if item.destination_tier == "host")

    def capacity_for(self, rank):
        for destination, value in self.receive_capacity_bytes:
            if destination == rank:
                return float(value)
        return None


def _normalize_releases(release_events):
    """Return destination release events keyed by transport endpoint rank."""
    if release_events is None:
        return None
    if isinstance(release_events, Mapping):
        items = release_events.items()
    else:
        items = ((1, release_events),)
    result = {}
    for rank, value in items:
        result[int(rank)] = (tuple(value) if isinstance(value, (tuple, list))
                             else (value,))
    return result


class PDConnector:
    def __init__(self, raw):
        explicit = "link" in raw
        declaration = dict(raw["link"] if explicit else raw)
        if not explicit:
            declaration["capacity_bytes_per_ms"] = declaration.pop("bytes_per_s") / 1000
        declaration.setdefault("name", "connector")
        declaration.setdefault("members", (0, 1))
        declaration["members"] = tuple(declaration["members"])
        # Optional transport declaration lives beside the link record.  It is
        # not part of Link and never changes the declared connector identity.
        self.chunk_bytes = raw.get("chunk_bytes")
        if self.chunk_bytes is not None:
            self.chunk_bytes = _non_negative_number(
                self.chunk_bytes, "pd_connector_chunk_bytes")
            if not self.chunk_bytes:
                raise ModelingError("pd_connector_chunk_bytes_invalid:0")
        self.receive_capacity_bytes = raw.get("receive_capacity_bytes")
        if self.receive_capacity_bytes is not None:
            self.receive_capacity_bytes = _non_negative_number(
                self.receive_capacity_bytes, "pd_connector_receive_capacity_bytes")
            if not self.receive_capacity_bytes:
                raise ModelingError("pd_connector_receive_capacity_bytes_invalid:0")
        self.hop_startup_us = _non_negative_number(
            raw.get("hop_startup_us", 0.0), "pd_connector_hop_startup_us")
        self.receive_buffer_id = str(raw.get("receive_buffer_id", "connector"))
        self.shared_receive_buffer = bool(raw.get("shared_receive_buffer", False))
        for key in ("chunk_bytes", "receive_capacity_bytes", "hop_startup_us",
                    "receive_buffer_id", "shared_receive_buffer"):
            declaration.pop(key, None)
        self.transport_declared = (
            self.chunk_bytes is not None
            or self.receive_capacity_bytes is not None
            or bool(self.hop_startup_us))
        self.link = Link(**declaration)
        if set(self.link.members) != {0, 1}:
            raise ModelingError("pd_connector_members_require_pool_endpoints_0_1")
        if self.link.semantics not in {
                "shared", "direct", "per_pair", "per_port", "fully_connected", "mesh"}:
            raise ModelingError(f"link_semantics_unknown:{self.link.semantics}")
        self.topology = Topology(Hardware("pd_connector", (), (self.link,)), None, "pd:")
        # Existing flat connector inputs keep their public resource identity.
        self.alias = ({f"pd:link:{self.link.name}": "pd_link"}
                      if not explicit else {})
        self.wire_bytes = 0.0
        self.transactions = 0 if self.link.transaction_bytes is not None else None
        # Aggregate bounded counters; a request's TransportRun is released as
        # soon as its completion folds these values, so request count cannot
        # grow live memory.
        self._transport = {
            "chunks": 0, "credit_waits": 0,
            "reservation_waits": 0, "reservation_wait_ms": 0.0,
            "resident": {}, "peak_resident": {}, "peak_reserved": {},
            "unknowns": set(),
        }
        self._tracked_residency = False
        self._engine = None
        self._handoff_count = 0
        self._handoff_flows = 0
        self._buffer_rank = {}
        self._handoff_ranks = set()
        self._last_receive_capacity = {}
        self._last_receive_buffers = {}

    def resources(self):
        return tuple(replace(resource, name=self.alias.get(resource.name, resource.name))
                     for resource in self.topology.resources())

    def _alias_plan(self, plan):
        """Keep the public resource identity while using the shared planner."""
        if not self.alias:
            return plan
        flows = []
        for path in plan.flows:
            chunks = []
            for chunk in path.chunks:
                hops = tuple(
                    replace(hop,
                            resource=self.alias.get(hop.resource, hop.resource),
                            additional_resources=tuple(
                                self.alias.get(name, name)
                                for name in hop.additional_resources))
                    for hop in chunk.hops)
                chunks.append(replace(chunk, hops=hops))
            flows.append(replace(path, chunks=tuple(chunks)))
        proxy = _AliasedTopology(self.topology, self.alias)
        return replace(plan, topology=proxy, flows=tuple(flows))

    def _topology_for_ranks(self, ranks):
        wanted = set(ranks)
        if wanted <= set(self.link.members):
            return self.topology
        link = replace(self.link, members=tuple(sorted(
            set(self.link.members) | wanted)))
        return Topology(Hardware("pd_connector", (), (link,)), None, "pd:")

    def _ensure_engine_resources(self, engine, topology):
        missing = [resource for resource in topology.resources()
                   if resource.name not in engine.resources]
        if missing:
            engine.extend(resources=missing)

    def _expand_handoff_releases(self, handoff, release_events):
        normalized = _normalize_releases(release_events) or {}
        chunk = self.chunk_bytes
        counts = defaultdict(int)
        for transfer in handoff.transfers:
            count = (1 if chunk is None else max(1, math.ceil(
                float(transfer.payload_bytes) / float(chunk) - 1e-12)))
            counts[transfer.destination_rank] += count
        result = {}
        for rank, count in counts.items():
            events = tuple(normalized.get(rank, ()))
            if not events:
                continue
            if len(events) == 1:
                events = events * count
            elif len(events) < count:
                raise ModelingError(
                    "pd_handoff_release_event_shortage:"
                    f"{rank}:{len(events)}<{count}")
            result[rank] = events
        return result

    def submit_handoff(self, engine, handoff, *, source_ready_by_rank,
                       destination_ready_by_rank, release_events=None,
                       source_last_use_by_rank=None,
                       source_release_callbacks=None,
                       release_callbacks_by_rank=None,
                       name=None):
        """Submit an actual request/rank/stage handoff binding.

        Every source and destination rank in ``handoff`` must have its own real
        readiness event.  The T09 transport plan receives the actual directed
        flows and per-destination physical receive buffer ids; T08 owns the
        storage/generation/range facts already carried by each transfer.  The
        receive footprint F is reserved once for the complete handoff in the
        declared pool budget C.  A whole-message policy is refused *before any
        submission* when F cannot fit C, so it cannot issue partial grants or
        wire work.

        ``source_last_use_by_rank`` are optional real local-reader events for a
        source rank; the source handle is returned exactly at the last of those
        events and that source's final transport read (never at a fabricated
        time).  ``release_events`` are real destination consumer events; the
        receive handle is returned only when their actual coverage equals the
        full footprint.  ``source_release_callbacks`` and
        ``release_callbacks_by_rank`` are fixed at call time with their
        pool/rank identity and invoked once at the real boundary.
        """
        if not isinstance(handoff, PDHandoff):
            raise ModelingError("pd_submit_handoff_requires_PDHandoff")
        name = name or f"pd:{handoff.request_id}"
        if not handoff.transfers:
            return engine.submit(
                name + ":empty", owner=f"request:{handoff.request_id}",
                metadata={"boundary": "pd_empty_handoff",
                          "request_id": handoff.request_id})
        if any(item.source_rank == item.destination_rank
               for item in handoff.transfers):
            raise ModelingError(
                "pd_handoff_self_flow_requires_distinct_pool_ranks")
        sources = {int(rank): event for rank, event in
                   dict(source_ready_by_rank).items()}
        destinations = {int(rank): event for rank, event in
                        dict(destination_ready_by_rank).items()}
        missing_sources = sorted({item.source_rank for item in handoff.transfers}
                                 - set(sources))
        missing_destinations = sorted({item.destination_rank for item in handoff.transfers}
                                      - set(destinations))
        if missing_sources or missing_destinations:
            raise ModelingError(
                "pd_handoff_missing_rank_ready:"
                f"source={missing_sources}:destination={missing_destinations}")
        capacity_by_rank = dict(handoff.receive_capacity_bytes)
        pool_capacity_by_rank = dict(handoff.receive_pool_capacity_bytes)
        buffer_by_rank = dict(handoff.receive_buffer_ids)
        # C and F are distinct declarations.  A whole-message receiver needs
        # the complete footprint in one allocation; if C < F it can never be
        # authorized, so refuse with zero Engine submissions.
        for destination, capacity in sorted(capacity_by_rank.items()):
            pool_capacity = pool_capacity_by_rank.get(destination, capacity)
            if float(pool_capacity) + 1e-9 < float(capacity):
                raise ModelingError(
                    "pd_whole_message_footprint_exceeds_pool:"
                    f"{destination}:{capacity}>{pool_capacity}")
        source_last_use = {}
        for rank, value in dict(source_last_use_by_rank or {}).items():
            source_last_use[int(rank)] = (
                tuple(value) if isinstance(value, (tuple, list)) else (value,))
        source_callbacks = {int(rank): callback for rank, callback
                            in dict(source_release_callbacks or {}).items()}
        release_callbacks = {int(rank): callback for rank, callback
                            in dict(release_callbacks_by_rank or {}).items()}

        topology = self._topology_for_ranks(handoff.ranks)
        self._ensure_engine_resources(engine, topology)
        # The receive footprint is reserved once for the complete handoff.
        # T09 still owns route/chunk/framing; its per-chunk credit is not used
        # here because the consumer is a whole-message consumer.
        transport = TransportOptions(
            chunk_bytes=self.chunk_bytes,
            hop_startup_us=self.hop_startup_us or 0.0,
            receive_buffer_ids=handoff.receive_buffer_ids,
            max_physical_chunks=4096)
        plan = topology.transport_plan(handoff.flows(), transport=transport)
        plan = self._alias_plan(plan)

        pair_ready = defaultdict(list)
        for transfer in handoff.transfers:
            pair = engine.submit(
                f"{name}:pair{transfer.source_rank}>{transfer.destination_rank}:ready",
                after=(sources[transfer.source_rank],
                       destinations[transfer.destination_rank]),
                owner=f"request:{handoff.request_id}")
            pair_ready[transfer.source_rank].append(pair)
            pair_ready[transfer.destination_rank].append(pair)
        from .engine import Allocation
        reservations = {}
        allocations = []
        for destination, capacity in sorted(capacity_by_rank.items()):
            buffer_id = buffer_by_rank.get(
                destination, f"{handoff.request_id}:rank{destination}")
            pool = f"transport:rx:{buffer_id}"
            pool_capacity = pool_capacity_by_rank.get(destination, capacity)
            if pool in engine.pools:
                if not math.isclose(engine.pools[pool], pool_capacity,
                                    rel_tol=1e-12, abs_tol=1e-9):
                    raise ModelingError(
                        "pd_receive_pool_capacity_conflict:"
                        f"{destination}:{engine.pools[pool]}!={pool_capacity}")
            else:
                engine.extend(pools={pool: pool_capacity})
            allocation = Allocation(
                f"{name}:rx:r{destination}", pool, float(capacity), (0,))
            reservations[destination] = allocation
            allocations.append(allocation)
        if allocations:
            reserve_ready = engine.submit(
                f"{name}:receive_ready",
                after=tuple(event for destination in sorted(capacity_by_rank)
                            for event in pair_ready[destination]),
                owner=f"request:{handoff.request_id}")
            grant = engine.submit(
                f"{name}:receive_grant", after=(reserve_ready,),
                acquire=tuple(allocations),
                owner=f"request:{handoff.request_id}",
                on_complete=lambda _event: self._track_reservation(reserve_ready),
                metadata={"boundary": "pd_whole_message_receive_grant",
                          "request_id": handoff.request_id,
                          "destination_ranks": sorted(capacity_by_rank),
                          "receive_buffer_ids": {
                              str(rank): buffer_by_rank.get(rank)
                              for rank in sorted(capacity_by_rank)},
                          "footprint_bytes": {
                              str(rank): capacity
                              for rank, capacity in capacity_by_rank.items()},
                          "pool_capacity_bytes": {
                              str(rank): pool_capacity_by_rank.get(rank, capacity)
                              for rank in sorted(capacity_by_rank)}})
            for rank in pair_ready:
                pair_ready[rank].append(grant)
        else:
            grant = None
        rank_ready = {
            rank: engine.submit(f"{name}:ready:r{rank}",
                                after=tuple(pair_ready[rank]),
                                owner=f"request:{handoff.request_id}")
            for rank in sorted(pair_ready)}
        boundary = engine.submit(
            name + ":ready", after=tuple(rank_ready.values()),
            owner=f"request:{handoff.request_id}",
            metadata={"boundary": "pd_handoff_all_rank_ready",
                      "request_id": handoff.request_id,
                      "rank_count": len(rank_ready)})
        memory = engine.submit(
            name + ":memory", dict(handoff.memory_work), after=(boundary,),
            owner=f"request:{handoff.request_id}")
        run = plan.start(engine, name + ":transport", rank_ready,
                         owner=f"request:{handoff.request_id}")
        release_map = _normalize_releases(release_events) or {}
        flow_index_by_destination = defaultdict(list)
        source_flow_events = defaultdict(list)
        for index, (event, path) in enumerate(zip(run.completions,
                                                  run.plan.flows)):
            transfer = handoff.transfers[index]
            source_flow_events[transfer.source_rank].append(event)
            if transfer.destination_tier == "hbm":
                flow_index_by_destination[path.flow.destination].append(index)
        release_events_by_rank = {}
        for destination, allocation in reservations.items():
            indices = flow_index_by_destination.get(destination, ())
            flow_events = tuple(run.completions[index] for index in indices)
            consumer_events = tuple(release_map.get(destination, ()))
            amount = sum(float(handoff.transfers[index].payload_bytes)
                         for index in indices)
            release_event = self._schedule_residency(
                engine, f"{name}:r{destination}", destination, amount,
                flow_events, consumer_events, allocation=allocation,
                release_name=f"{name}:receive_release:r{destination}",
                explicit_consumer=True,
                release_callback=release_callbacks.get(destination))
            release_events_by_rank[destination] = release_event
        source_release_events = {}
        for rank in sorted(source_flow_events):
            explicit = source_last_use.get(rank, ())
            release = engine.submit(
                f"{name}:source_last_use:r{rank}",
                after=(sources[rank], *source_flow_events[rank], *explicit),
                owner=f"request:{handoff.request_id}",
                metadata={"boundary": "pd_source_last_use_release",
                          "request_id": handoff.request_id,
                          "source_rank": rank,
                          "explicit_local_readers": len(explicit)},
                on_complete=(
                    (lambda _event, callback=source_callbacks.get(rank):
                     callback() if callback is not None else None)
                    if rank in source_callbacks else None))
            source_release_events[rank] = release
        self._engine = engine
        wire = sum(hop.wire_bytes for path in run.plan.flows
                   for chunk in path.chunks for hop in chunk.hops)
        transactions = (sum(hop.transactions or 0 for path in run.plan.flows
                            for chunk in path.chunks for hop in chunk.hops)
                        if self.transactions is not None else None)
        self.wire_bytes += wire
        if self.transactions is not None:
            self.transactions += transactions
        chunks = sum(len(path.chunks) for path in run.plan.flows)
        self._transport["chunks"] += chunks
        self._transport["unknowns"].update(key for key, _ in run.plan.unknowns)
        self._handoff_count += 1
        self._handoff_flows += len(handoff.transfers)
        self._handoff_ranks.update(handoff.ranks)
        self._last_receive_capacity = dict(handoff.receive_capacity_bytes)
        self._last_receive_buffers = dict(handoff.receive_buffer_ids)
        for rank, buffer_id in run.plan.receive_buffer_ids:
            self._buffer_rank[buffer_id] = int(rank)
        return engine.submit(
            name, after=(memory, *run.completions),
            owner=f"request:{handoff.request_id}",
            metadata={
                "boundary": "pd_pool_connector_handoff",
                "request_id": handoff.request_id,
                "useful_bytes": sum(float(item.payload_bytes)
                                    for item in handoff.transfers),
                "host_bytes": sum(float(item.payload_bytes)
                                  for item in handoff.host_transfers()),
                "hbm_bytes": sum(float(item.payload_bytes)
                                 for item in handoff.hbm_transfers()),
                "wire_bytes": wire,
                "transactions": transactions,
                "chunks": chunks,
                "flows": len(handoff.transfers),
                "receive_buffer_ids": dict(handoff.receive_buffer_ids),
                "source_release_ranks": sorted(source_release_events),
                "transport": "common_topology_transport_plan",
            }, on_complete=lambda _event: self._fold_transport(run))

    def submit(self, engine, name, volume, memory_work, *,
               ready=None, destination_ready=None, release_events=None):
        """Submit one handoff under one common source/destination boundary.

        ``ready`` is the source KV availability event; ``destination_ready`` is
        the actual decode admission/grant event.  Both branches receive one
        boundary event that is the join of those two events.  A legacy
        whole-message transfer always pays its propagation stage after that
        boundary; specifying an equivalent zero-time ready cannot delete work.
        ``release_events`` are consumer events, not an additional call duration.
        """
        releases = _normalize_releases(release_events)
        declared_consumer = release_events is not None
        self._engine = engine
        source_ready = ready if ready is not None else engine.submit(
            name + ":source_ready")
        receiver_ready = (destination_ready if destination_ready is not None
                          else source_ready)
        boundary = engine.submit(
            name + ":ready", after=(source_ready, receiver_ready),
            metadata={"boundary": "pd_common_source_destination_ready",
                      "source_ready": source_ready.name,
                      "destination_ready": receiver_ready.name})
        if not self.transport_declared:
            return self._submit_legacy(
                engine, name, volume, memory_work, boundary=boundary,
                releases=releases)
        if (declared_consumer and self.receive_capacity_bytes is not None
                and float(self.receive_capacity_bytes) >= float(volume)):
            return self._submit_reserved_single(
                engine, name, volume, memory_work, boundary=boundary,
                releases=releases, declared_consumer=True)
        if (declared_consumer and self.receive_capacity_bytes is not None
                and float(self.receive_capacity_bytes) < float(volume)):
            # A single consumer release after the complete handoff cannot
            # safely free a smaller per-chunk receive pool: the next chunk's
            # grant would wait for a consumer that waits for the complete
            # handoff.  Refuse to fabricate a deadlock; a chunked consumer
            # protocol is a separate explicit capability.
            raise ModelingError(
                "pd_finite_receive_credit_requires_per_chunk_consumer:"
                f"capacity={self.receive_capacity_bytes}:volume={volume}")
        plan = self.topology.transport_plan(
            [MessageFlow(0, 1, volume)],
            chunk_bytes=self.chunk_bytes,
            receive_capacity_bytes=self.receive_capacity_bytes,
            hop_startup_us=self.hop_startup_us or 0.0)
        plan = self._alias_plan(plan)
        run = plan.start(engine, name + ":transport",
                         {0: boundary, 1: boundary},
                         release_events=releases)
        memory = engine.submit(name + ":memory", memory_work, after=(boundary,))
        self.wire_bytes += sum(
            hop.wire_bytes for path in run.plan.flows
            for chunk in path.chunks for hop in chunk.hops)
        if self.transactions is not None:
            self.transactions += sum(
                hop.transactions or 0 for path in run.plan.flows
                for chunk in path.chunks for hop in chunk.hops)
        self._transport["chunks"] += sum(len(path.chunks) for path in run.plan.flows)
        self._transport["unknowns"].update(key for key, _ in run.plan.unknowns)
        for rank, buffer_id in run.plan.receive_buffer_ids:
            self._buffer_rank[buffer_id] = int(rank)
        return engine.submit(
            name, after=(memory, *run.completions), metadata={
                "boundary": "pd_pool_connector",
                "useful_bytes": volume,
                "wire_bytes": sum(hop.wire_bytes for path in run.plan.flows
                                  for chunk in path.chunks for hop in chunk.hops),
                "chunks": sum(len(path.chunks) for path in run.plan.flows),
                "transactions": (sum(
                    hop.transactions or 0 for path in run.plan.flows
                    for chunk in path.chunks for hop in chunk.hops)
                    if self.transactions is not None else None),
                "transport": "common_topology_transport_plan",
            }, on_complete=lambda _event: self._fold_transport(run))

    def _submit_reserved_single(self, engine, name, volume, memory_work, *,
                                boundary, releases, declared_consumer=False):
        """Hold one complete receive footprint for a whole-message consumer."""
        from .engine import Allocation
        capacity = float(self.receive_capacity_bytes)
        pool = f"transport:rx:{self.receive_buffer_id}"
        if pool in engine.pools:
            if not math.isclose(engine.pools[pool], capacity,
                                rel_tol=1e-12, abs_tol=1e-9):
                raise ModelingError(
                    "pd_receive_pool_capacity_conflict:"
                    f"{self.receive_buffer_id}:{engine.pools[pool]}!={capacity}")
        else:
            engine.extend(pools={pool: capacity})
        allocation = Allocation(f"{name}:rx", pool, float(volume), (0,))
        grant = engine.submit(
            name + ":receive_grant", after=(boundary,), acquire=(allocation,),
            metadata={"boundary": "pd_whole_message_receive_grant",
                      "receive_buffer_id": self.receive_buffer_id,
                      "pool_capacity_bytes": capacity,
                      "footprint_bytes": volume},
            on_complete=lambda _event: self._track_reservation(boundary))
        transport = TransportOptions(
            chunk_bytes=self.chunk_bytes,
            hop_startup_us=self.hop_startup_us or 0.0,
            receive_buffer_ids=((1, self.receive_buffer_id),),
            max_physical_chunks=4096)
        plan = self.topology.transport_plan(
            [MessageFlow(0, 1, volume)], transport=transport)
        plan = self._alias_plan(plan)
        run = plan.start(engine, name + ":transport", {0: grant, 1: grant})
        memory = engine.submit(name + ":memory", memory_work, after=(grant,))
        consumer_events = tuple((releases or {}).get(1, ()))
        self._schedule_residency(
            engine, name, 1, float(volume), run.completions, consumer_events,
            allocation=allocation,
            release_name=f"{name}:receive_release:r1",
            explicit_consumer=declared_consumer)
        self._engine = engine
        wire = sum(hop.wire_bytes for path in run.plan.flows
                   for chunk in path.chunks for hop in chunk.hops)
        transactions = (sum(hop.transactions or 0 for path in run.plan.flows
                            for chunk in path.chunks for hop in chunk.hops)
                        if self.transactions is not None else None)
        self.wire_bytes += wire
        if self.transactions is not None:
            self.transactions += transactions
        chunks = sum(len(path.chunks) for path in run.plan.flows)
        self._transport["chunks"] += chunks
        self._transport["unknowns"].update(key for key, _ in run.plan.unknowns)
        self._handoff_count += 1
        self._handoff_flows += 1
        self._handoff_ranks.update((0, 1))
        self._last_receive_capacity = {1: capacity}
        self._last_receive_buffers = {1: self.receive_buffer_id}
        for rank, buffer_id in run.plan.receive_buffer_ids:
            self._buffer_rank[buffer_id] = int(rank)
        return engine.submit(
            name, after=(memory, *run.completions), metadata={
                "boundary": "pd_pool_connector_reserved",
                "useful_bytes": volume, "wire_bytes": wire,
                "transactions": transactions, "chunks": chunks,
                "receive_buffer_id": self.receive_buffer_id,
                "transport": "whole_message_receive_reservation",
            }, on_complete=lambda _event: self._fold_transport(run))

    def _submit_legacy(self, engine, name, volume, memory_work, *,
                       boundary, releases=None):
        hop, = self.topology.route_message(MessageFlow(0, 1, volume)).hops
        # Propagation is a physical stage after the common boundary, never a
        # branch-local shortcut that disappears when an explicit ready is given.
        propagation = engine.submit(
            name + ":latency", after=(boundary,), latency=hop.latency_ms,
            metadata={"boundary": "pd_link_propagation_once"})
        demand = {self.alias.get(key, key): value for key, value in hop.demand.items()}
        self.wire_bytes += hop.wire_bytes
        if self.transactions is not None:
            self.transactions += hop.transactions
        memory = engine.submit(name + ":memory", memory_work, after=(propagation,))
        wire = engine.submit(name + ":wire", demand, after=(propagation,), coupled=True)
        complete = engine.submit(name, after=(memory, wire), metadata={
            "boundary": "pd_pool_connector", "useful_bytes": volume,
            "wire_bytes": hop.wire_bytes, "transactions": hop.transactions,
        })
        destination_releases = (releases or {}).get(1, ())
        if destination_releases:
            # Consumer release is a separate zero-work boundary.  It never
            # extends the handoff completion, but it is not silently dropped.
            engine.submit(name + ":consume", after=(wire, *destination_releases),
                          metadata={"boundary": "pd_consumer_release",
                                    "consumer_events": len(destination_releases)})
        return complete

    def summary(self):
        result = {
            "boundary": "pd_pool_connector", "endpoints": {"0": "prefill", "1": "decode"},
            "message_boundary": "one_complete_destination_sharded_handoff",
            "memory_boundary": "one_local_shard_per_pool_replica_stage",
            "link": self.link.to_dict(), "wire_bytes": self.wire_bytes,
            "transactions": self.transactions,
            "chunk_bytes": self.chunk_bytes,
            "receive_capacity_bytes": self.receive_capacity_bytes,
            "hop_startup_us": self.hop_startup_us,
            "handoffs": self._handoff_count,
            "handoff_flows": self._handoff_flows,
            "handoff_ranks": sorted(self._handoff_ranks),
            "handoff_receive_capacity_bytes": {
                str(rank): value for rank, value in
                sorted(self._last_receive_capacity.items())},
            "handoff_receive_buffer_ids": {
                str(rank): value for rank, value in
                sorted(self._last_receive_buffers.items())},
        }
        if self._transport["chunks"] or self._transport["credit_waits"]:
            shared_peak = {}
            if self._engine is not None:
                for pool, value in self._engine.peak_allocated.items():
                    if str(pool).startswith("transport:rx:"):
                        buffer_id = str(pool)[len("transport:rx:"):]
                        rank = self._buffer_rank.get(buffer_id, buffer_id)
                        shared_peak[str(rank)] = value
            reserved = (shared_peak if shared_peak else
                        {str(rank): value for rank, value in
                         self._transport["peak_reserved"].items()})
            result["transport"] = {
                "boundary": "common_topology_transport_plan",
                "capacity_scope": ("engine_pool_transport:rx:<destination>; "
                                   "shared by plans on the same Engine"),
                "reserved_peak_basis": ("engine_pool_peak_granted" if shared_peak
                                        else "max_single_handoff_granted"),
                "residency_peak_basis": (
                    "pd_receive_arrival_minus_consumer"
                    if self._tracked_residency else
                    "max_single_handoff_not_concurrent_sum"),
                "chunks": self._transport["chunks"],
                "credit_waits": self._transport["credit_waits"],
                "reservation_waits": self._transport["reservation_waits"],
                "reservation_wait_ms": self._transport["reservation_wait_ms"],
                "resident_bytes": dict(sorted(
                    (str(rank), value) for rank, value in
                    self._transport["resident"].items())),
                "peak_resident_bytes": dict(sorted(
                    (str(rank), value) for rank, value in
                    self._transport["peak_resident"].items())),
                "peak_reserved_bytes": dict(sorted(reserved.items())),
                "unknowns": sorted(self._transport["unknowns"]),
            }
        return result

    def _track_arrival(self, destination, amount):
        key = int(destination)
        value = self._transport["resident"].get(key, 0.0) + float(amount)
        self._transport["resident"][key] = value
        if value > self._transport["peak_resident"].get(key, 0.0):
            self._transport["peak_resident"][key] = value

    def _track_consumption(self, destination, amount):
        key = int(destination)
        self._transport["resident"][key] = max(
            0.0, self._transport["resident"].get(key, 0.0) - float(amount))

    def _schedule_residency(self, engine, name, destination, amount,
                            arrivals, consumer_events, allocation=None,
                            release_name=None, explicit_consumer=False,
                            release_callback=None):
        """Bind arrival, real consumer times and the actual release decision.

        Consumer events are not a count multiplier.  Each consumer's coverage
        is decided from that consumer's *original* event time, never from the
        synthetic callback time: a consumer before full arrival may drain at
        most one physical chunk share, while a consumer at/after full arrival
        may drain the remaining tail.  If the declared sequence does not cover
        the whole footprint, the allocation stays live, so a successor cannot
        be admitted by an exhausted event list.  An explicitly empty sequence
        is a hold, not arrival-time consumption.
        """
        self._tracked_residency = True
        events = tuple(consumer_events)

        def track_arrival(_event):
            self._track_arrival(destination, amount)

        arrival_event = engine.submit(
            f"{name}:arrival", after=tuple(arrivals),
            metadata={"boundary": "pd_receive_arrival",
                      "destination_rank": destination,
                      "bytes": amount},
            on_complete=track_arrival)
        if explicit_consumer and not events:
            return None
        consumed = [0.0]
        count = (1 if self.chunk_bytes is None else max(
            1, math.ceil(float(amount) / float(self.chunk_bytes) - 1e-12)))
        shares = []
        if not events:
            shares = []
        elif len(events) == 1:
            shares = [float(amount)]
        elif len(events) >= count:
            share = float(self.chunk_bytes or amount)
            shares = [share] * count
            shares[-1] = float(amount) - share * (count - 1)
            shares.extend([0.0] * (len(events) - count))
        else:
            shares = [float(self.chunk_bytes or (amount / len(events)))] * len(events)

        def apply(index, value):
            value = min(max(0.0, float(value)),
                        max(0.0, float(amount) - consumed[0]))
            consumed[0] += value
            self._track_consumption(destination, value)

        def consume(_task, index):
            original = events[index]
            base = shares[index] if index < len(shares) else 0.0
            if not arrival_event.done:
                # The consumer ran before the complete message arrived.  Its
                # real declaration still covers its bounded share; account it
                # at the real full-arrival boundary instead of inventing an
                # earlier arrival.
                engine.submit(
                    f"{name}:consume{index}:deferred", after=(arrival_event,),
                    metadata={"boundary": "pd_receive_consumer_share",
                              "destination_rank": destination,
                              "consumer_index": index,
                              "bytes": float(base)},
                    on_complete=lambda _event, i=index:
                        apply(i, shares[i] if i < len(shares) else 0.0))
                return
            original_time = (original.time if original.time is not None
                             else float(arrival_event.time or 0.0))
            if original_time >= float(arrival_event.time or 0.0):
                value = max(0.0, float(amount) - consumed[0])
            else:
                value = float(base)
            apply(index, value)

        for index, event in enumerate(events):
            engine.submit(
                f"{name}:consume{index}", after=(event,),
                metadata={"boundary": "pd_receive_consumer_share",
                          "destination_rank": destination,
                          "consumer_index": index,
                          "bytes": shares[index] if index < len(shares) else 0.0},
                on_complete=lambda _event, i=index: consume(_event, i))
        if allocation is None:
            return None

        def decide_release(_event):
            if consumed[0] + 1e-9 < float(amount) and events:
                last = events[-1]
                if (last.time is not None and arrival_event.time is not None
                        and last.time >= arrival_event.time):
                    # The last original consumer is a whole-message tail; its
                    # callback may have been deferred behind the same arrival.
                    consumed[0] = float(amount)
            if consumed[0] + 1e-9 < float(amount):
                return
            def released(_event):
                if release_callback is not None:
                    release_callback()
            engine.submit(
                release_name or f"{name}:receive_release", free=(allocation,),
                metadata={"boundary": "pd_whole_message_receive_release",
                          "destination_rank": destination,
                          "bytes": amount},
                on_complete=released)

        return engine.submit(
            f"{name}:release_decision", after=(arrival_event, *events),
            metadata={"boundary": "pd_receive_consumer_coverage",
                      "destination_rank": destination,
                      "bytes": amount},
            on_complete=decide_release)

    def _track_reservation(self, ready_event):
        ready_ms = ready_event.time or 0.0
        wait = max(0.0, float(self._engine.now) - float(ready_ms))
        if wait > 1e-12:
            self._transport["reservation_waits"] += 1
            self._transport["reservation_wait_ms"] += wait

    def _fold_transport(self, run):
        report = run.report()
        self._transport["credit_waits"] += report["credit_waits"]
        if self._tracked_residency:
            return
        for key, value in report["receiver_resident_peak_bytes"].items():
            rank = str(key)
            if value > self._transport["peak_resident"].get(rank, 0.0):
                self._transport["peak_resident"][rank] = value
        for key, value in report["receiver_reserved_peak_bytes"].items():
            rank = str(key)
            if value > self._transport["peak_reserved"].get(rank, 0.0):
                self._transport["peak_reserved"][rank] = value
        self._transport["unknowns"].update(report.get("unknowns", {}))
