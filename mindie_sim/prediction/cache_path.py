"""Logical, GM, on-chip and HBM access resolution with explicit cache state.

[Role] Turn an ordered sequence of data-identity accesses into physical path
  work: logical payload, on-chip copy service, staging occupancy, cache fill,
  eviction/writeback and next-level line transactions.
[Invariants] Logical bytes are not HBM bytes. A hit removes only the next-level
  fill for the retained lines; on-chip service, staging occupancy and a declared
  ready delay remain. Bypass never fills or evicts. A written line that cannot
  be retained is materialized in the same access. A dirty line is written back
  at most once, on eviction, invalidation or explicit flush. HBM write work
  always satisfies direct_write + writeback; residency is reported per identity
  and never added to a capacity allocation here. Every non-empty access is held
  by one reservation; zero-byte accesses hold nothing.
[Timeouts & Locks] O(resident identities) per access, no I/O, threads or locks.
[Side-Effect Policy] Mutates only the CacheState owned by the caller. Program
  submission uses the caller's Engine and transient staging allocations only;
  retained residency is reported, never reserved here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from mindie_sim.schema.common import ModelingError
from .cache_geometry import (
    CacheAccessGeometry, Span, TransactionSummary, byte_spans_of_line_spans,
    drop_last_spans, full_line_spans, intersect_spans, lines_of_byte_spans,
    normalize_spans, range_count, subtract_spans,
    take_first_spans, union_spans,
)
from .engine import Allocation, Engine, Event
from .program import KernelProgram, Stage
from .transfer import FiniteTransfer

# A5 external GM/HBM transaction width. SIMT external accesses are 128-byte
# granular (npu_arch_3510.md:28); no cache hit/miss policy is claimed by it.
A5_LINE_BYTES = 128
# A5 GM copy / UB transaction granule already used by the existing binder.
A5_TRANSFER_BYTES = 32


def _align_up(value: int, base: int) -> int:
    return (value + base - 1) // base * base


@dataclass(frozen=True)
class CacheGeometry:
    """One declarative cache level between GM requests and next-level lines.

    ``capacity_bytes`` is the competition scope shared by every identity.
    ``line_bytes`` is the next-level transaction width. ``transfer_bytes`` is
    the GM/on-chip copy granule. ``staging_lines`` bounds in-flight occupancy;
    an access never creates one object per byte or per payload line.
    """

    capacity_bytes: int = 0
    line_bytes: int = A5_LINE_BYTES
    transfer_bytes: int = A5_TRANSFER_BYTES
    staging_lines: int = 1

    def __post_init__(self):
        for name in ("capacity_bytes", "line_bytes", "transfer_bytes", "staging_lines"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ModelingError(f"cache_geometry_requires_integer:{name}")
        if self.capacity_bytes < 0:
            raise ModelingError("cache_geometry_capacity_must_be_nonnegative")
        if self.line_bytes < 1 or self.transfer_bytes < 1 or self.staging_lines < 1:
            raise ModelingError("cache_geometry_granules_must_be_positive")

    @property
    def capacity_lines(self) -> int:
        return self.capacity_bytes // self.line_bytes


@dataclass(frozen=True)
class CachePolicy:
    """Access behavior declared by the selected implementation path."""

    bypass: bool = False
    # A cold path has no reusable identity. It never looks up, fills or
    # evicts; state remains untouched even when a long-lived state is passed.
    cold: bool = False
    write_policy: str = "write_through"
    ready_ms: float = 0.0

    def __post_init__(self):
        if type(self.bypass) is not bool or type(self.cold) is not bool:
            raise ModelingError("cache_policy_bypass_and_cold_must_be_boolean")
        if self.write_policy not in {"write_through", "write_back"}:
            raise ModelingError("cache_policy_unknown_write_policy")
        if not isinstance(self.ready_ms, (int, float)) or isinstance(self.ready_ms, bool):
            raise ModelingError("cache_policy_ready_must_be_numeric")
        if not (self.ready_ms >= 0.0):
            raise ModelingError("cache_policy_ready_must_be_nonnegative")


@dataclass(frozen=True)
class CacheAccessResult:
    """Physical quantities resolved for one access against current state."""

    identity: str
    mode: str
    logical_bytes: int
    gm_bytes: int
    onchip_bytes: int
    staging_bytes: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    hbm_read_lines: int
    hbm_write_lines: int
    cache_read_bytes: int
    fill_bytes: int
    evicted_bytes: int
    writeback_bytes: int
    # New bytes written through now because they were not retained. Dirty
    # writebacks remain separate; hbm_write_bytes == direct + writeback.
    direct_write_bytes: int
    hit_lines: int
    miss_lines: int
    resident_bytes: int
    bypassed: bool
    cold_path: bool
    ready_ms: float

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def transactions(self) -> int:
        return self.hbm_read_lines + self.hbm_write_lines

    def to_dict(self) -> dict:
        return {
            "identity": self.identity, "mode": self.mode,
            "logical_bytes": self.logical_bytes, "gm_bytes": self.gm_bytes,
            "onchip_bytes": self.onchip_bytes, "staging_bytes": self.staging_bytes,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_bytes": self.hbm_bytes,
            "hbm_read_lines": self.hbm_read_lines,
            "hbm_write_lines": self.hbm_write_lines,
            "cache_read_bytes": self.cache_read_bytes,
            "fill_bytes": self.fill_bytes, "evicted_bytes": self.evicted_bytes,
            "writeback_bytes": self.writeback_bytes,
            "direct_write_bytes": self.direct_write_bytes,
            "hit_lines": self.hit_lines, "miss_lines": self.miss_lines,
            "resident_bytes": self.resident_bytes, "bypassed": self.bypassed,
            "cold_path": self.cold_path,
            "ready_ms": self.ready_ms,
            "transactions": self.transactions,
        }


def _byte_spans_of_lines(ranges, line):
    return byte_spans_of_line_spans(ranges, line)


@dataclass
class CacheAccessJournal:
    """One access's physical responsibility and its real terminal history.

    Prepared capacity, actual issued stages, publication and retirement live in
    this single record.  A failed issuance therefore cannot silently free a
    reservation or staging slot: the journal still says which work was already
    submitted and whether its physical effect was published.
    """

    identity: str
    holder: int | None = None
    target_lines: int = 0
    reserved_bytes: int = 0
    dirty_lines: int = 0
    dirty_bytes: int = 0
    drain_bytes: int = 0
    direct_bytes: int = 0
    staged_bytes: int = 0
    issued: list = field(default_factory=list)
    published: bool = False
    drained: bool = False
    retired: bool = False
    failure: str = ""

    def note_issue(self, label: str, resource: str | None, amount: float) -> None:
        self.issued.append((label, resource, float(amount)))

    def note_publish(self) -> None:
        self.published = True

    def note_drain(self) -> None:
        self.drained = True

    def note_retire(self) -> None:
        self.retired = True

    def snapshot(self) -> dict:
        return {
            "identity": self.identity, "holder": self.holder,
            "target_lines": self.target_lines,
            "reserved_bytes": self.reserved_bytes,
            "dirty_lines": self.dirty_lines, "dirty_bytes": self.dirty_bytes,
            "drain_bytes": self.drain_bytes,
            "direct_bytes": self.direct_bytes,
            "staged_bytes": self.staged_bytes,
            "issued": [list(item) for item in self.issued],
            "published": self.published, "drained": self.drained,
            "retired": self.retired, "failure": self.failure,
        }



@dataclass
class _Resident:
    lines: int
    # Counts remain the compatibility view. Ranges are actual resident/dirty
    # lines; byte ranges distinguish a resident line from a valid byte span.
    dirty_lines: int = 0
    ranges: tuple = ()
    dirty_ranges: tuple = ()
    valid_bytes: tuple = ()
    dirty_bytes_ranges: tuple = ()


class CacheState:
    """Mutable per-level residency, recency and dirty-prefix state.

    Access order is the call order. Each identity denotes one reusable data
    object; repeated accesses to it reuse the retained line prefix. Different
    identities compete for one capacity by least-recent use. A caller resolving
    speculative work should pass ``clone()`` and commit only the selected path.
    """

    def __init__(self, geometry: CacheGeometry):
        if not isinstance(geometry, CacheGeometry):
            raise ModelingError("cache_state_requires_geometry")
        self.geometry = geometry
        self._resident: dict[str, _Resident] = {}
        self._resident_bytes = 0
        # Monotonic last-use ticket per resident identity. Dict insertion order
        # is only a display order; eviction and rollback use this stable rank so
        # overlapping reservations cannot reorder the LRU list.
        self._recency: dict[str, int] = {}
        self._clock = 0
        # Lines reserved by in-flight accesses but not yet published (fill or
        # dirty write). They are not hittable and cannot be evicted.
        self._reserved_bytes = 0
        # identity -> set of unique reservation holders. A holder releases only
        # the identities it actually claimed; a zero-byte reservation claims
        # nothing and can never release another access's responsibility.
        self._inflight: dict[str, set[int]] = {}
        self._holder_sequence = 0
        self._version = 0
        # A pending writeback owns the dirty set. No access/invalidate/reset
        # may change it until the executable writeback completes.
        self._pending_writeback = None
        # One runtime root may own this mutable cache domain. A detached state
        # stays usable standalone; adopting it registers the same object in
        # RuntimeState.cache_domains and never copies residency.
        self.runtime_uid: str | None = None
        self.cache_domain_id: str | None = None
        self.counters = {
            "accesses": 0, "reads": 0, "writes": 0, "hits": 0, "misses": 0,
            "fills_bytes": 0, "evictions": 0, "writeback_bytes": 0,
            "direct_write_bytes": 0, "logical_bytes": 0, "gm_bytes": 0,
            "onchip_bytes": 0, "hbm_read_bytes": 0, "hbm_write_bytes": 0,
        }

    @property
    def version(self) -> int:
        return self._version

    def bind_runtime(self, state, *, domain_id: str | None = None) -> str | None:
        """Adopt this residency state into the one RuntimeState cache domain."""
        uid = getattr(state, "uid", None)
        if not isinstance(uid, str) or not uid:
            raise ModelingError("cache_state_requires_runtime_uid")
        if self.runtime_uid is not None and self.runtime_uid != uid:
            raise ModelingError("cache_runtime_conflict")
        domains = getattr(state, "cache_domains", None)
        if domains is None:
            raise ModelingError("cache_state_runtime_missing_cache_domains")
        key = domain_id or self.cache_domain_id or f"cache:{id(self):x}"
        existing = domains.get(key)
        if existing is not None and existing is not self:
            raise ModelingError(f"cache_domain_conflict:{key}")
        domains[key] = self
        self.runtime_uid = uid
        self.cache_domain_id = key
        return key

    def clone(self) -> "CacheState":
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        if self.dirty_bytes:
            # Two independently flushed copies would materialize the same
            # dirty bytes twice. Reject rather than invent alias tracking.
            raise ModelingError("cache_clone_dirty_requires_flush")
        result = CacheState(self.geometry)
        result._resident = {
            key: _Resident(value.lines, value.dirty_lines, value.ranges,
                           value.dirty_ranges, value.valid_bytes,
                           value.dirty_bytes_ranges)
            for key, value in self._resident.items()}
        result._resident_bytes = self._resident_bytes
        result._recency = dict(self._recency)
        result._clock = self._clock
        result._version = self._version
        result.counters = dict(self.counters)
        return result

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def resident_identities(self) -> int:
        return len(self._resident)

    @property
    def dirty_bytes(self) -> int:
        return sum(range_count(value.dirty_bytes_ranges)
                   for value in self._resident.values())

    @property
    def inflight_identities(self) -> int:
        return len(self._inflight)

    def resident(self, identity: str) -> tuple[int, int]:
        """Return retained line count and dirty line count."""
        value = self._resident.get(identity)
        return (0, 0) if value is None else (value.lines, value.dirty_lines)

    def resident_ranges(self, identity: str):
        """Return actual retained and dirty line spans."""
        value = self._resident.get(identity)
        return ((), ()) if value is None else (value.ranges, value.dirty_ranges)

    def _require_no_inflight(self) -> None:
        if self._inflight:
            raise ModelingError("cache_access_in_flight")

    def _new_holder(self) -> int:
        self._holder_sequence += 1
        return self._holder_sequence

    def _next_recency(self) -> int:
        self._clock += 1
        return self._clock

    def _touch(self, identity: str) -> None:
        self._recency[identity] = self._next_recency()
        self._reorder_resident()

    def _reorder_resident(self) -> None:
        if len(self._resident) > 1:
            self._resident = dict(sorted(
                self._resident.items(),
                key=lambda item: (self._recency.get(item[0], 0),
                                  item[0])))

    def _drop_recency(self, identity: str) -> None:
        self._recency.pop(identity, None)

    def _claim(self, holder: int, identities) -> None:
        for identity in identities:
            self._inflight.setdefault(identity, set()).add(holder)

    def _release(self, holder: int) -> None:
        for identity in tuple(self._inflight):
            holders = self._inflight[identity]
            holders.discard(holder)
            if not holders:
                del self._inflight[identity]

    def _identity_held(self, identity: str) -> bool:
        return identity in self._inflight

    def reset(self) -> None:
        """Forget all residency explicitly, discarding dirty data."""
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        self._resident.clear()
        self._resident_bytes = 0
        self._recency.clear()
        self._version += 1

    def flush(self) -> "CacheWriteback":
        """Return the executable writeback of the current dirty prefix.

        The dirty set is not cleared and counters do not change here. A
        ``CacheWriteback`` owns the dirty set until its Engine task completes;
        while it is pending, accesses/invalidate/reset/clone are rejected.
        """
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        lines = sum(value.dirty_lines for value in self._resident.values())
        if not lines:
            return CacheWriteback(self, 0)
        plan = CacheWriteback(self, lines)
        self._pending_writeback = plan
        return plan

    def _finish_writeback(self, plan: "CacheWriteback") -> None:
        if self._pending_writeback is not plan:
            raise ModelingError("cache_writeback_completion_without_pending_plan")
        written = plan.bytes
        for value in self._resident.values():
            value.dirty_lines = 0
            value.dirty_ranges = ()
            value.dirty_bytes_ranges = ()
        self.counters["writeback_bytes"] += written
        self.counters["hbm_write_bytes"] += written
        self._pending_writeback = None
        self._version += 1

    def invalidate(self, identity: str | None = None) -> int:
        """Drop clean residency; reject dirty data until it is flushed.

        Dirty invalidation would silently lose an external effect, so it is
        rejected rather than returned as a counter. Flush the dirty prefix
        first, then invalidate the clean resident object.
        """
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        line = self.geometry.line_bytes
        keys = (list(self._resident) if identity is None else
                ([identity] if identity in self._resident else []))
        if any(self._resident[key].dirty_lines for key in keys):
            raise ModelingError("cache_invalidate_dirty_requires_flush")
        removed = 0
        for key in keys:
            value = self._resident.pop(key)
            self._drop_recency(key)
            self._resident_bytes -= value.lines * line
            removed += value.lines
        if keys:
            self._version += 1
        return removed

    def access(self, identity: str, logical_bytes: int, *, write: bool = False,
               policy: CachePolicy | None = None) -> CacheAccessResult:
        """Resolve and commit one contiguous access through the range resolver.

        This is the synchronous adapter for the same byte/line projection used
        by ``prepare_access`` and the executable path; it is not a second
        install mechanism.
        """
        policy = policy or CachePolicy()
        if not isinstance(identity, str) or not identity:
            raise ModelingError("cache_access_identity_must_be_nonempty_string")
        if type(logical_bytes) is not int or logical_bytes < 0:
            raise ModelingError("cache_access_bytes_must_be_nonnegative_integer")
        geometry = (CacheAccessGeometry(rows=1, row_bytes=logical_bytes)
                    if logical_bytes else CacheAccessGeometry(rows=0,
                                                              row_bytes=0))
        reservation = self.prepare_access(identity, geometry, write=write,
                                          policy=policy)
        # The synchronous adapter is one atomic access: its result already
        # accounts the victim/preflush HBM write, so the slot handoff is part
        # of this access rather than a separately executable drain stage.
        reservation.mark_drained()
        reservation.publish()
        reservation.finish()
        return reservation.result

    def _evict_ranges(self, needed_lines, keep):
        """Build a read-only eviction plan; no resident value is changed here."""
        line = self.geometry.line_bytes
        plans = []
        remaining = needed_lines
        ordered = sorted(self._resident.items(),
                         key=lambda item: (self._recency.get(item[0], 0),
                                           item[0]))
        for key, value in ordered:
            if key == keep or self._identity_held(key):
                continue
            if remaining <= 0:
                break
            drop = min(value.lines, remaining)
            if not drop:
                continue
            remaining -= drop
            snapshot = replace(value)
            recency = self._recency.get(key, 0)
            remaining_ranges, dropped = drop_last_spans(value.ranges, drop)
            dirty_dropped = range_count(intersect_spans(
                dropped, value.dirty_ranges))
            dropped_bytes = _byte_spans_of_lines(dropped, line)
            updated = _Resident(
                value.lines - drop, value.dirty_lines - dirty_dropped,
                remaining_ranges,
                subtract_spans(value.dirty_ranges, dropped),
                subtract_spans(value.valid_bytes, dropped_bytes),
                subtract_spans(value.dirty_bytes_ranges, dropped_bytes))
            plans.append((key, recency, snapshot,
                          None if updated.lines == 0 else updated, drop,
                          dirty_dropped))
        if remaining > 0:
            if self._reserved_bytes:
                raise ModelingError("cache_capacity_reserved_exhausted")
            if any(self._identity_held(key) for key in self._resident):
                raise ModelingError("cache_capacity_inflight_held")
            raise ModelingError("cache_state_capacity_accounting_inconsistent")
        evicted = sum(item[4] for item in plans)
        written = sum(item[5] * line for item in plans)
        return evicted, written, tuple(plans)

    def _apply_evictions(self, plans) -> None:
        line = self.geometry.line_bytes
        for key, _recency, _snapshot, updated, drop, _dirty in plans:
            if updated is None:
                self._resident.pop(key, None)
                self._drop_recency(key)
            else:
                self._resident[key] = updated
            self._resident_bytes -= drop * line

    @staticmethod
    def _restore_resident_snapshots(state, restorations) -> None:
        """Restore snapshots with their stable last-use rank, not dict index."""
        line = state.geometry.line_bytes
        for recency, key, snapshot in sorted(
                restorations, key=lambda item: (item[0], str(item[1]))):
            current = state._resident.pop(key, None)
            if current is not None:
                state._resident_bytes -= current.lines * line
            state._resident[key] = snapshot
            state._recency[key] = recency
            state._resident_bytes += snapshot.lines * line
        state._reorder_resident()

    @staticmethod
    def _restore_victims(state, victims):
        CacheState._restore_resident_snapshots(
            state, ((recency, key, snapshot)
                    for key, recency, snapshot, _updated, _drop, _dirty
                    in victims))

    def prepare_access(self, identity: str, geometry: "CacheAccessGeometry", *,
                       write: bool = False,
                       policy: CachePolicy | None = None) -> "CacheAccessReservation":
        """Reserve an access against current state for event-time execution.

        Existing lines of the same identity move to the reserved set. The
        actual requested line spans (offset/stride holes included) determine
        hit, miss, fill, dirty and RMW work. Victims and their dirty
        responsibility are held until the reservation terminal event; cancel
        restores them if no physical work was issued.
        """
        policy = policy or CachePolicy()
        if not isinstance(identity, str) or not identity:
            raise ModelingError("cache_access_identity_must_be_nonempty_string")
        if not isinstance(geometry, CacheAccessGeometry):
            raise ModelingError("cache_access_requires_typed_geometry")
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        if self._identity_held(identity):
            raise ModelingError("cache_identity_access_in_flight")
        line = self.geometry.line_bytes
        summary = geometry.summary(
            line, write=write, transfer_bytes=self.geometry.transfer_bytes)
        line_count = summary.line_count
        request_ranges = summary.line_segments
        transfer_bytes = summary.gm_bytes if summary.logical_bytes else 0
        staging = (min(transfer_bytes, line * self.geometry.staging_lines)
                   if summary.logical_bytes else 0)
        capacity = self.geometry.capacity_lines
        bypass = policy.bypass or policy.cold or capacity == 0
        existing = self._resident.get(identity)
        if line_count == 0:
            result = CacheAccessResult(
                identity=identity, mode="write" if write else "read",
                logical_bytes=0, gm_bytes=0, onchip_bytes=0, staging_bytes=0,
                hbm_read_bytes=0, hbm_write_bytes=0, hbm_read_lines=0,
                hbm_write_lines=0, cache_read_bytes=0, fill_bytes=0,
                evicted_bytes=0, writeback_bytes=0, direct_write_bytes=0,
                hit_lines=0, miss_lines=0, resident_bytes=self._resident_bytes,
                bypassed=bypass, cold_path=policy.cold,
                ready_ms=float(policy.ready_ms),
            )
            return CacheAccessReservation(
                self, identity, 0, 0, target_ranges=(), dirty_ranges=(),
                target_valid_bytes=(), target_dirty_bytes=(),
                existing_ranges=(), existing_dirty=(), existing_valid=(),
                existing_dirty_bytes=(), victims=(),
                result=result, bypassed=True, holder=None, claimed=())
        bypass_existing = None
        existing_recency = -1
        existing_reserved_lines = 0
        if bypass:
            if existing is not None:
                if existing.dirty_lines:
                    raise ModelingError("cache_bypass_conflicts_with_dirty_identity")
                if write:
                    # Keep the clean old row as a rollback reservation until
                    # this access is irreversibly published or cancelled.
                    bypass_existing = replace(existing)
                    existing_recency = self._recency.get(identity, 0)
                    existing_reserved_lines = existing.lines
            if bypass_existing is not None:
                self._resident.pop(identity)
                self._drop_recency(identity)
                self._resident_bytes -= existing_reserved_lines * line
                self._reserved_bytes += existing_reserved_lines * line
                existing = None
            hbm_read_lines = hbm_write_lines = direct_lines = 0
            if write:
                hbm_write_lines = line_count
                direct_lines = line_count
            else:
                hbm_read_lines = line_count
            hbm_read_lines += summary.rmw_read_lines
            result = CacheAccessResult(
                identity=identity, mode="write" if write else "read",
                logical_bytes=summary.logical_bytes,
                gm_bytes=transfer_bytes, onchip_bytes=transfer_bytes,
                staging_bytes=staging,
                hbm_read_bytes=hbm_read_lines * line,
                hbm_write_bytes=hbm_write_lines * line,
                hbm_read_lines=hbm_read_lines,
                hbm_write_lines=hbm_write_lines,
                cache_read_bytes=0, fill_bytes=0, evicted_bytes=0,
                writeback_bytes=0, direct_write_bytes=direct_lines * line,
                hit_lines=0, miss_lines=line_count,
                resident_bytes=self._resident_bytes + self._reserved_bytes,
                bypassed=True, cold_path=policy.cold,
                ready_ms=float(policy.ready_ms),
            )
            holder = self._new_holder()
            held = (identity,)
            self._claim(holder, held)
            return CacheAccessReservation(
                self, identity, 0, 0, target_ranges=(), dirty_ranges=(),
                target_valid_bytes=(), target_dirty_bytes=(),
                existing_ranges=(() if bypass_existing is None
                                 else bypass_existing.ranges),
                existing_dirty=(() if bypass_existing is None
                                else bypass_existing.dirty_ranges),
                existing_valid=(() if bypass_existing is None
                                else bypass_existing.valid_bytes),
                existing_dirty_bytes=(() if bypass_existing is None
                                      else bypass_existing.dirty_bytes_ranges),
                victims=(),
                result=result, bypassed=True, holder=holder, claimed=held,
                existing_recency=existing_recency,
                existing_reserved_lines=existing_reserved_lines)
        resident_ranges = () if existing is None else existing.ranges
        resident_valid = () if existing is None else existing.valid_bytes
        dirty_lines = () if existing is None else existing.dirty_ranges
        dirty_bytes = () if existing is None else existing.dirty_bytes_ranges
        existing_snapshot = None if existing is None else replace(existing)
        existing_recency = (-1 if existing is None
                            else self._recency.get(identity, 0))
        request_bytes = summary.byte_segments
        request_lines = request_ranges
        resident_total = (self._resident_bytes + self._reserved_bytes) // line
        existing_lines = range_count(resident_ranges)
        free = capacity - resident_total
        max_new = max(0, capacity - existing_lines)
        candidates = subtract_spans(request_lines, resident_ranges)
        need = max(0, min(range_count(candidates), max_new) - max(0, free))
        evicted_lines = victim_writeback = 0
        victims = ()
        if need:
            evicted_lines, victim_writeback, victims = self._evict_ranges(
                need, identity)
        free_after = max(
            0, capacity
            - ((self._resident_bytes - evicted_lines * line
                + self._reserved_bytes) // line))
        retain = take_first_spans(
            candidates, min(range_count(candidates), free_after))
        target_ranges = union_spans(resident_ranges, retain)
        target_lines = range_count(target_ranges)
        target_byte_ranges = _byte_spans_of_lines(target_ranges, line)
        hit_bytes = intersect_spans(request_bytes, resident_valid)
        miss_bytes = subtract_spans(request_bytes, resident_valid)
        hit_lines_ranges = lines_of_byte_spans(hit_bytes, line)
        miss_lines_ranges = lines_of_byte_spans(miss_bytes, line)
        rmw_lines = ()
        if write and geometry.partial_line == "read_modify_write":
            covered = union_spans(request_bytes, resident_valid)
            rmw_lines = subtract_spans(
                request_lines, full_line_spans(covered, line))
        valid_bytes_new = resident_valid
        dirty_bytes_new = dirty_bytes
        hbm_read_lines = hbm_write_lines = direct_lines = 0
        writeback_bytes = victim_writeback
        if write and policy.write_policy == "write_through":
            preflush_lines = subtract_spans(dirty_lines, request_lines)
            preflush = range_count(preflush_lines)
            hbm_write_lines = line_count + preflush
            direct_lines = line_count
            writeback_bytes += preflush * line
            valid_bytes_new = union_spans(
                resident_valid,
                intersect_spans(request_bytes, target_byte_ranges),
                _byte_spans_of_lines(
                    intersect_spans(rmw_lines, target_ranges), line))
            dirty_bytes_new = ()
        elif write:
            retained_bytes = intersect_spans(request_bytes,
                                             target_byte_ranges)
            direct_ranges = subtract_spans(request_lines, target_ranges)
            direct_lines = range_count(direct_ranges)
            hbm_write_lines = direct_lines
            valid_bytes_new = union_spans(
                resident_valid, retained_bytes,
                _byte_spans_of_lines(
                    intersect_spans(rmw_lines, target_ranges), line))
            dirty_bytes_new = union_spans(
                subtract_spans(dirty_bytes, request_bytes), retained_bytes)
        else:
            hbm_read_lines = range_count(miss_lines_ranges)
            fetched_resident = intersect_spans(miss_lines_ranges,
                                               resident_ranges)
            valid_bytes_new = union_spans(
                resident_valid,
                _byte_spans_of_lines(
                    union_spans(retain, fetched_resident), line))
        hbm_read_lines += range_count(rmw_lines)
        if write and policy.write_policy == "write_through":
            hbm_write_lines += victim_writeback // line
        elif write:
            hbm_write_lines += victim_writeback // line
        else:
            hbm_write_lines = victim_writeback // line
        direct_write_bytes = direct_lines * line
        if hbm_write_lines * line != direct_write_bytes + writeback_bytes:
            raise ModelingError("cache_write_conservation_failed")
        drain_bytes = writeback_bytes
        dirty_lines_new = lines_of_byte_spans(dirty_bytes_new, line)
        # All arithmetic above was read-only. Apply the complete plan at once.
        reserved_added = False
        try:
            self._apply_evictions(victims)
            if existing is not None:
                self._resident.pop(identity)
                self._drop_recency(identity)
                self._resident_bytes -= existing.lines * line
            self._reserved_bytes += target_lines * line
            reserved_added = True
            if (self._resident_bytes + self._reserved_bytes
                    > self.geometry.capacity_bytes):
                raise ModelingError(
                    "cache_state_capacity_accounting_inconsistent")
        except Exception:
            if reserved_added:
                self._reserved_bytes -= target_lines * line
            restorations = [(recency, key, snapshot)
                            for key, recency, snapshot, _updated, _drop, _dirty
                            in victims]
            if existing_snapshot is not None:
                restorations.append(
                    (existing_recency, identity, existing_snapshot))
            self._restore_resident_snapshots(self, restorations)
            raise
        holder = self._new_holder()
        held = (identity,) + tuple(item[0] for item in victims)
        self._claim(holder, held)
        result = CacheAccessResult(
            identity=identity, mode="write" if write else "read",
            logical_bytes=summary.logical_bytes,
            gm_bytes=summary.gm_bytes, onchip_bytes=summary.gm_bytes,
            staging_bytes=staging,
            hbm_read_bytes=hbm_read_lines * line,
            hbm_write_bytes=hbm_write_lines * line,
            hbm_read_lines=hbm_read_lines, hbm_write_lines=hbm_write_lines,
            cache_read_bytes=(0 if write else range_count(hit_bytes)),
            fill_bytes=range_count(retain) * line,
            evicted_bytes=evicted_lines * line,
            writeback_bytes=writeback_bytes,
            direct_write_bytes=direct_write_bytes,
            hit_lines=range_count(hit_lines_ranges),
            miss_lines=range_count(miss_lines_ranges),
            resident_bytes=self._resident_bytes + self._reserved_bytes,
            bypassed=False, cold_path=False, ready_ms=float(policy.ready_ms),
        )
        return CacheAccessReservation(
            self, identity, target_lines, range_count(dirty_lines_new),
            target_ranges, dirty_lines_new, valid_bytes_new, dirty_bytes_new,
            resident_ranges, dirty_lines, resident_valid, dirty_bytes,
            victims, result, False, holder, held,
            existing_recency=existing_recency,
            drain_bytes=drain_bytes, drain_completed=not drain_bytes)

    def _account(self, result: CacheAccessResult) -> None:
        counters = self.counters
        counters["accesses"] += 1
        counters["reads" if result.mode == "read" else "writes"] += 1
        if result.hit_lines or result.miss_lines:
            counters["hits" if result.hit_lines else "misses"] += 1
        counters["fills_bytes"] += result.fill_bytes
        counters["evictions"] += result.evicted_bytes
        counters["writeback_bytes"] += result.writeback_bytes
        counters["direct_write_bytes"] += result.direct_write_bytes
        counters["logical_bytes"] += result.logical_bytes
        counters["gm_bytes"] += result.gm_bytes
        counters["onchip_bytes"] += result.onchip_bytes
        counters["hbm_read_bytes"] += result.hbm_read_bytes
        counters["hbm_write_bytes"] += result.hbm_write_bytes

    def snapshot(self) -> dict:
        """Bounded export: totals and identity count, never one row per line."""
        return {
            "geometry": {
                "capacity_bytes": self.geometry.capacity_bytes,
                "line_bytes": self.geometry.line_bytes,
                "transfer_bytes": self.geometry.transfer_bytes,
                "staging_lines": self.geometry.staging_lines,
            },
            "resident_bytes": self._resident_bytes,
            "reserved_bytes": self._reserved_bytes,
            "resident_identities": len(self._resident),
            "dirty_bytes": self.dirty_bytes,
            "inflight_identities": len(self._inflight),
            "writeback_in_flight": self._pending_writeback is not None,
            "version": self._version,
            "counters": dict(self.counters),
        }


@dataclass(eq=False)
class CacheAccessReservation:
    """Lines/victims held by one in-flight access; publish installs the target."""

    state: "CacheState"
    identity: str
    target_lines: int
    dirty_lines: int
    target_ranges: tuple
    dirty_ranges: tuple
    target_valid_bytes: tuple
    target_dirty_bytes: tuple
    existing_ranges: tuple
    existing_dirty: tuple
    existing_valid: tuple
    existing_dirty_bytes: tuple
    victims: tuple
    result: "CacheAccessResult"
    bypassed: bool
    holder: int | None = None
    claimed: tuple = ()
    published: bool = False
    finished: bool = False
    # Stable last-use ticket and lines moved from resident to the reversible
    # reservation (clean bypass write).
    existing_recency: int = -1
    existing_reserved_lines: int = 0
    # Dirty victim/preflush bytes that must reach HBM before this target may
    # occupy the same physical slot. The sync adapter and an executable drain
    # stage both close this explicitly.
    drain_bytes: int = 0
    drain_completed: bool = True
    journal: CacheAccessJournal | None = None

    def __post_init__(self):
        if self.journal is None:
            line = self.state.geometry.line_bytes
            self.journal = CacheAccessJournal(
                identity=self.identity, holder=self.holder,
                target_lines=self.target_lines,
                reserved_bytes=(self.target_lines * line
                                + self.existing_reserved_lines * line),
                dirty_lines=self.dirty_lines,
                dirty_bytes=range_count(self.target_dirty_bytes),
                drain_bytes=self.drain_bytes,
                direct_bytes=self.result.direct_write_bytes,
                staged_bytes=self.result.staging_bytes)

    @property
    def held_identities(self):
        """Identities claimed by this reservation (zero-byte holds none)."""
        return self.claimed

    def note_issue(self, label: str, resource: str | None,
                   amount: float) -> None:
        self.journal.note_issue(label, resource, amount)

    def mark_drained(self) -> None:
        """Mark the victim/preflush physical handoff as really completed."""
        if not self.drain_completed:
            self.drain_completed = True
            self.journal.note_drain()

    def fail(self, error: Exception) -> None:
        self.journal.failure = f"{type(error).__name__}:{error}"

    def publish(self) -> None:
        if self.finished:
            raise ModelingError("cache_reservation_already_finished")
        if self.published:
            raise ModelingError("cache_reservation_already_published")
        if not self.drain_completed:
            raise ModelingError("cache_reservation_drain_incomplete")
        line = self.state.geometry.line_bytes
        if self.bypassed:
            if self.existing_reserved_lines:
                self.state._reserved_bytes -= (
                    self.existing_reserved_lines * line)
                self.state._version += 1
            self.published = True
            self.journal.reserved_bytes = 0
            self.journal.note_publish()
            return
        if not self.target_lines:
            self.published = True
            self.journal.reserved_bytes = 0
            self.journal.note_publish()
            return
        self.state._reserved_bytes -= self.target_lines * line
        self.state._resident_bytes += self.target_lines * line
        self.state._resident[self.identity] = _Resident(
            self.target_lines, self.dirty_lines, self.target_ranges,
            self.dirty_ranges, self.target_valid_bytes, self.target_dirty_bytes)
        self.state._touch(self.identity)
        self.state._version += 1
        self.published = True
        self.journal.reserved_bytes = 0
        self.journal.note_publish()

    def finish(self) -> None:
        if self.finished:
            raise ModelingError("cache_reservation_already_finished")
        if not self.published:
            raise ModelingError("cache_reservation_finished_before_publish")
        self.state._account(self.result)
        if self.holder is not None:
            self.state._release(self.holder)
        self.state._version += 1
        self.finished = True
        self.journal.note_retire()

    def cancel(self) -> None:
        if self.published or self.finished:
            raise ModelingError("cache_reservation_already_published")
        line = self.state.geometry.line_bytes
        if self.target_lines:
            self.state._reserved_bytes -= self.target_lines * line
        if self.existing_reserved_lines:
            self.state._reserved_bytes -= self.existing_reserved_lines * line
        restorations = []
        if self.existing_ranges:
            # If a preflush already reached HBM, the old prefix is durable.
            # Restore it as clean rather than re-creating a dirty writeback.
            dirty_ranges = (() if self.drain_completed and self.drain_bytes
                            else self.existing_dirty)
            dirty_bytes = (() if self.drain_completed and self.drain_bytes
                           else self.existing_dirty_bytes)
            restorations.append((
                self.existing_recency, self.identity,
                _Resident(range_count(self.existing_ranges),
                          range_count(dirty_ranges),
                          self.existing_ranges, dirty_ranges,
                          self.existing_valid, dirty_bytes)))
        if not (self.drain_completed and self.drain_bytes):
            restorations.extend(
                (recency, key, snapshot)
                for key, recency, snapshot, _updated, _drop, _dirty
                in self.victims)
        CacheState._restore_resident_snapshots(self.state, restorations)
        if self.holder is not None:
            self.state._release(self.holder)
        self.state._version += 1
        self.finished = True
        self.journal.reserved_bytes = 0
        self.journal.note_retire()


@dataclass(eq=False)
class CacheWriteback:
    """Executable writeback of one CacheState's dirty prefix.

    Created by ``CacheState.flush()``. It owns the dirty set while pending;
    ``start`` submits real HBM work and clears the dirty prefix/counters only
    on task completion. A plan that is never started leaves the state blocked,
    which is an explicit incomplete external effect rather than a fake clear.
    """

    state: CacheState
    lines: int
    started: bool = False

    @property
    def bytes(self) -> int:
        return self.lines * self.state.geometry.line_bytes

    def start(self, engine: Engine, name: str, ready: Event, *,
              hbm_resource: str = "hbm", owner: str = "") -> Event:
        if not isinstance(engine, Engine):
            raise ModelingError("cache_writeback_requires_engine")
        if self.started:
            raise ModelingError("cache_writeback_already_started")
        if not self.bytes:
            self.started = True
            return engine.submit(name + ":empty", after=(ready,), owner=owner)
        if self.state._pending_writeback is not self:
            raise ModelingError("cache_writeback_is_not_pending")
        if hbm_resource not in engine.resources:
            raise ModelingError(f"cache_writeback_hbm_resource_missing:{hbm_resource}")
        self.started = True
        try:
            event = engine.submit(
                name, {hbm_resource: float(self.bytes)}, after=(ready,),
                on_complete=lambda _: self.state._finish_writeback(self),
                owner=owner,
            )
        except Exception:
            self.started = False
            raise
        return event

    def cancel(self) -> None:
        """Release the pending state without writing or clearing dirty data."""
        if self.started:
            raise ModelingError("cache_writeback_already_started")
        if self.state._pending_writeback is self:
            self.state._pending_writeback = None


@dataclass
class AccessSummary:
    """Aggregated access work for compute and capacity consumers."""

    accesses: int = 0
    logical_bytes: int = 0
    gm_bytes: int = 0
    onchip_bytes: int = 0
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    direct_write_bytes: int = 0
    cache_read_bytes: int = 0
    staging_peak_bytes: int = 0
    resident_peak_bytes: int = 0

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    def add(self, result: CacheAccessResult) -> "AccessSummary":
        self.accesses += 1
        self.logical_bytes += result.logical_bytes
        self.gm_bytes += result.gm_bytes
        self.onchip_bytes += result.onchip_bytes
        self.hbm_read_bytes += result.hbm_read_bytes
        self.hbm_write_bytes += result.hbm_write_bytes
        self.direct_write_bytes += result.direct_write_bytes
        self.cache_read_bytes += result.cache_read_bytes
        self.staging_peak_bytes = max(self.staging_peak_bytes, result.staging_bytes)
        self.resident_peak_bytes = max(self.resident_peak_bytes, result.resident_bytes)
        return self

    def to_dict(self) -> dict:
        return {
            "accesses": self.accesses, "logical_bytes": self.logical_bytes,
            "gm_bytes": self.gm_bytes, "onchip_bytes": self.onchip_bytes,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes, "hbm_bytes": self.hbm_bytes,
            "direct_write_bytes": self.direct_write_bytes,
            "cache_read_bytes": self.cache_read_bytes,
            "staging_peak_bytes": self.staging_peak_bytes,
            "resident_peak_bytes": self.resident_peak_bytes,
        }


@dataclass(frozen=True)
class CacheStage:
    """One shared physical stage of the resolved cache data path."""

    label: str
    resource: str | None = None
    amount: float = 0.0
    after: tuple[str, ...] = ()
    publish: bool = False
    terminal: bool = False


def cache_work_stages(result: CacheAccessResult, *,
                      hbm_resource: str = "hbm",
                      onchip_resource: str = "mte2",
                      write_resource: str | None = None,
                      vector_resource: str = "vector",
                      source_bytes: int = 0,
                      source_residency: str = "on_chip",
                      convert_ops: float = 0.0) -> tuple[CacheStage, ...]:
    """One explicit immutable phase DAG consumed by both adapters.

    Every stage names its real predecessors; there is no implicit ``ready``
    sibling. The chain is source -> conversion -> request read/RMW -> physical
    drain (when a dirty slot is reused) -> publish/stage -> direct writeback ->
    retire. Only absent phases are collapsed.
    """
    stages: list[CacheStage] = []
    labels: set[str] = set()

    def add(label, resource=None, amount=0.0, *, after, publish=False,
            terminal=False):
        predecessors = tuple(after)
        if not predecessors:
            raise ModelingError(f"cache_stage_requires_predecessor:{label}")
        if label in labels:
            raise ModelingError(f"cache_stage_duplicate:{label}")
        for predecessor in predecessors:
            if predecessor != "ready" and predecessor not in labels:
                raise ModelingError(
                    f"cache_stage_predecessor_missing:{label}:{predecessor}")
        labels.add(label)
        stages.append(CacheStage(
            label, resource, float(amount) if amount else 0.0,
            predecessors, publish, terminal))
        return label

    basis = ("ready",)
    if source_bytes:
        if source_residency != "on_chip":
            label = add("source_hbm", hbm_resource, source_bytes, after=basis)
            basis = (label,)
        label = add("source_onchip", onchip_resource, source_bytes, after=basis)
        basis = (label,)
    if convert_ops:
        label = add("convert", vector_resource, convert_ops, after=basis)
        basis = (label,)
    drain_bytes = result.writeback_bytes
    direct_bytes = result.direct_write_bytes
    if result.mode == "read":
        if drain_bytes:
            # A miss may replace a dirty slot. The old bytes leave HBM before
            # the fill overwrites that physical slot.
            label = add("drain", hbm_resource, drain_bytes, after=basis)
            basis = (label,)
        if result.hbm_read_bytes:
            fill = add("fill", hbm_resource, result.hbm_read_bytes, after=basis)
            basis = (add("publish", after=(fill,), publish=True),)
        deliver = add("deliver", onchip_resource, result.onchip_bytes,
                      after=basis)
        add("retire", after=(deliver,), terminal=True)
    else:
        if result.hbm_read_bytes:
            label = add("rmw", hbm_resource, result.hbm_read_bytes, after=basis)
            basis = (label,)
        if drain_bytes:
            label = add("drain", hbm_resource, drain_bytes, after=basis)
            basis = (label,)
        stage = add("stage", write_resource or onchip_resource,
                    result.onchip_bytes, after=basis, publish=True)
        terminal = (stage,)
        if direct_bytes:
            label = add("writeback", hbm_resource, direct_bytes,
                        after=terminal)
            terminal = (label,)
        add("retire", after=terminal, terminal=True)

    # The graph is closed, acyclic in declaration order, has exactly one
    # terminal runner and every stage is reachable from the adapter's ready.
    reachable = {"ready"}
    for stage in stages:
        if any(predecessor not in reachable for predecessor in stage.after):
            raise ModelingError(f"cache_stage_graph_open:{stage.label}")
        reachable.add(stage.label)
    terminal_stages = [stage for stage in stages if stage.terminal]
    if len(terminal_stages) != 1:
        raise ModelingError(
            "cache_stage_graph_terminal_count:" + str(len(terminal_stages)))
    terminal_label = terminal_stages[0].label
    successors = {stage.label: 0 for stage in stages}
    for stage in stages:
        for predecessor in stage.after:
            if predecessor in successors:
                successors[predecessor] += 1
    if any(count == 0 and label != terminal_label
           for label, count in successors.items()):
        raise ModelingError("cache_stage_graph_orphan")
    return tuple(stages)


@dataclass(frozen=True)
class CacheTransfer:
    """Finite byte transfer with the same credit/response parameters for both
    cache adapters. This is a transport declaration, not a T08 grant."""

    work: tuple[tuple[str, float], ...]
    transaction_bytes: int = A5_LINE_BYTES
    outstanding: int | None = None
    response_ms: float = 0.0

    def __post_init__(self):
        if type(self.transaction_bytes) is not int or self.transaction_bytes < 1:
            raise ModelingError("cache_transfer_transaction_bytes_invalid")
        if self.outstanding is not None and (
                type(self.outstanding) is not int or self.outstanding < 1):
            raise ModelingError("cache_transfer_outstanding_invalid")
        if (not isinstance(self.response_ms, (int, float))
                or isinstance(self.response_ms, bool) or self.response_ms < 0):
            raise ModelingError("cache_transfer_response_invalid")

    def start(self, engine: Engine, name: str, ready: Event, *,
              owner: str = "") -> Event:
        return FiniteTransfer(
            dict(self.work), transaction_bytes=self.transaction_bytes,
            outstanding=self.outstanding, response_ms=self.response_ms,
        ).start(engine, name, ready, owner=owner)


@dataclass(frozen=True)
class CacheAccessProgram:
    """Executable access path for callers that own a Stage/Program boundary.

    The cache resolves state when the access is built; this program submits the
    resolved physical path. It is the runnable form that includes ready delay,
    source read/conversion, staging allocation and all HBM work (read fill plus
    any eviction writeback), which a graph envelope cannot express by bytes
    alone. ``outstanding``/``response_ms`` are the request's real transfer
    parameters and are applied to every HBM phase by the same ``CacheTransfer``
    used by the event-time adapter.
    """

    result: CacheAccessResult
    hbm_resource: str = "hbm"
    onchip_resource: str = "mte2"
    pool: str | None = None
    source_required: bool = False
    source: Event | None = None
    source_bytes: int = 0
    convert_ops: float = 0.0
    source_residency: str = "on_chip"
    vector_resource: str = "vector"
    write_resource: str | None = None
    outstanding: int = 0
    response_ms: float = 0.0
    transaction_bytes: int = A5_LINE_BYTES

    def __post_init__(self):
        if type(self.outstanding) is not int or self.outstanding < 0:
            raise ModelingError("cache_access_program_outstanding_invalid")
        if (not isinstance(self.response_ms, (int, float))
                or isinstance(self.response_ms, bool) or self.response_ms < 0):
            raise ModelingError("cache_access_program_response_invalid")
        if type(self.transaction_bytes) is not int or self.transaction_bytes < 1:
            raise ModelingError("cache_access_program_transaction_bytes_invalid")

    def start(self, engine: Engine, name: str, ready: Event, *,
              source: Event | None = None, owner: str = "") -> Event:
        source = self.source if source is None else source
        if self.source_required and source is None:
            raise ModelingError("cache_access_program_source_missing")
        if source is not None and not isinstance(source, Event):
            raise ModelingError("cache_access_program_source_invalid")
        if not isinstance(engine, Engine):
            raise ModelingError("cache_access_program_requires_engine")
        if self.result.staging_bytes and not self.pool:
            raise ModelingError("cache_access_program_requires_staging_pool")
        if (self.result.hbm_read_bytes or self.result.hbm_write_bytes) \
                and self.hbm_resource not in engine.resources:
            raise ModelingError(f"cache_access_hbm_resource_missing:{self.hbm_resource}")
        if self.result.onchip_bytes and self.onchip_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_onchip_resource_missing:{self.onchip_resource}")
        if self.source_bytes and self.source_residency == "unresolved" \
                and self.hbm_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_hbm_resource_missing:{self.hbm_resource}")
        if (self.source_bytes or self.convert_ops) \
                and self.onchip_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_onchip_resource_missing:{self.onchip_resource}")
        if self.convert_ops and self.vector_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_vector_resource_missing:{self.vector_resource}")
        staging = ()
        if self.result.staging_bytes:
            if self.pool not in engine.pools:
                raise ModelingError(f"cache_access_pool_missing:{self.pool}")
            staging = (Allocation(name + ":staging", self.pool,
                                  self.result.staging_bytes),)
        # Build and preflight the complete phase graph before creating the
        # source-join task or any Stage. No Engine mutation precedes this.
        items = cache_work_stages(
            self.result, hbm_resource=self.hbm_resource,
            onchip_resource=self.onchip_resource,
            write_resource=(self.write_resource
                            or self.onchip_resource),
            vector_resource=self.vector_resource,
            source_bytes=self.source_bytes,
            source_residency=self.source_residency,
            convert_ops=self.convert_ops)
        for item in items:
            if item.resource is not None and item.resource not in engine.resources:
                raise ModelingError(
                    f"cache_access_resource_missing:{item.resource}")
            if (item.resource == self.hbm_resource
                    and (self.outstanding or self.response_ms)
                    and engine.resources[item.resource].unit != "bytes"):
                raise ModelingError(
                    f"cache_access_hbm_resource_not_bytes:{self.hbm_resource}")
        start_ready = ready
        if source is not None:
            start_ready = engine.submit(name + ":source_join",
                                        after=(ready, source), owner=owner)
        stages = [Stage("ready", latency=self.result.ready_ms,
                        acquire=staging)]
        for item in items:
            work = ({item.resource: item.amount} if item.resource else {})
            if (item.resource == self.hbm_resource
                    and (self.outstanding or self.response_ms)):
                work = {}
                transfer = CacheTransfer(
                    ((item.resource, item.amount),),
                    transaction_bytes=self.transaction_bytes,
                    outstanding=self.outstanding or None,
                    response_ms=self.response_ms)
                stages.append(Stage(item.label, after=item.after,
                                    program=transfer))
            else:
                stages.append(Stage(item.label, work, after=item.after))
        if staging:
            # Hold the staging slot through the last physical effect, not just
            # through the visible copy.
            stages[-1] = replace(stages[-1], free=staging)
        return KernelProgram(tuple(stages)).start(engine, name, start_ready,
                                                  owner=owner)


__all__ = [
    "A5_LINE_BYTES", "A5_TRANSFER_BYTES", "AccessSummary",
    "CacheAccessGeometry", "CacheAccessProgram", "CacheAccessReservation",
    "CacheAccessResult", "CacheGeometry", "CachePolicy", "CacheStage",
    "CacheState", "CacheTransfer", "cache_work_stages",
    "CacheWriteback", "TransactionSummary",
]
