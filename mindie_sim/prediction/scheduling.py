"""Shared fragment execution on the one causal Engine.

[Role] Lower ``ProgramFragment`` declarations into ordinary Engine tasks:
  ordered repeats, conditional/program/slot allocations, stream order, event
  ports, state visibility and bounded online aggregation.
[Invariants] One virtual clock and one Engine. A fragment keeps its complete
  executable semantics; unsupported executable semantics are rejected before
  work is admitted, never silently projected away. Repeats are sequential on
  the fragment stream and batch only when homogeneous. State reads bind at
  start and writes/commit/rollback apply once at completion; same-time commits
  need an explicit arbiter. Observations never feed durations back.
[Timeouts & Locks] One task per fragment unless repeats must expand, bounded by
  ``repeat_limit``. Homogeneous repeats use one aggregated task. No I/O or sleep.
[Side-Effect Policy] Mutates only the caller's Engine, the state ledger owned
  by that runtime and this conductor's bounded observation caches.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import math
from collections.abc import Mapping as _AbcMapping, Sequence as _AbcSequence
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import AllocationEffect, ProgramFragment, StateEffect
from .engine import Allocation, Engine, Event


class FragmentContractError(ModelingError):
    """The caller supplied a fragment whose executable meaning is unresolved."""


class UnsupportedFragmentSemantics(FragmentContractError):
    """A declared fragment feature is not executable by this slice."""


_MISSING = object()
_STATE_ACTIONS_WRITE = frozenset({"write", "commit", "rollback"})


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FragmentContractError(f"{name}_invalid:{value!r}")
    return value


def _version(value: Any, target: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FragmentContractError(f"state_version_invalid:{target}:{value!r}")
    return value


class StateLedger:
    """Visible committed versions plus speculative live buffers.

    ``write`` adds a candidate; ``commit`` changes the visible version;
    ``rollback`` discards only an uncommitted candidate. A commit retains the
    previous committed version for already-issued readers plus versions at or
    ahead of the new commitment; older superseded candidates are reclaimed so a
    long request history cannot grow this ledger without bound.
    """

    def __init__(self, initial: Mapping[str, int] | None = None, *,
                 version_limit: int = 64):
        self._committed: dict[str, int] = {}
        self._live: dict[str, set[int]] = {}
        #: Runtime root that adopted this ledger; a second root is an error.
        self.runtime_uid: str | None = None
        self.version_limit = max(2, _nonnegative_int(
            "state_version_limit", version_limit))
        # Same-timestamp commits are grouped before selecting a winner. The
        # winner is the largest (arbiter, sequence) pair, so container order is
        # never the authority.
        self._pending: dict[tuple[str, float], dict[int, tuple[str, int]]] = {}
        self._commit_base: dict[tuple[str, float], int | None] = {}
        for target, version in dict(initial or {}).items():
            if not isinstance(target, str) or not target:
                raise FragmentContractError("state_target_invalid")
            value = _version(version, target)
            self._committed[target] = value
            self._live[target] = {value}

    def snapshot(self) -> dict[str, Any]:
        return {
            "committed": dict(sorted(self._committed.items())),
            "live": {target: tuple(sorted(versions))
                     for target, versions in sorted(self._live.items())},
        }

    def initialize(self, initial: Mapping[str, int] | None = None) -> None:
        """Seed targets that are not yet known; a different version conflicts."""
        for target, version in dict(initial or {}).items():
            if not isinstance(target, str) or not target:
                raise FragmentContractError("state_target_invalid")
            value = _version(version, target)
            current = self._committed.get(target)
            if current is None:
                self._committed[target] = value
                self._live.setdefault(target, set()).add(value)
            elif current != value:
                raise FragmentContractError(
                    f"state_ledger_conflict:{target}:{current}:{value}")

    def committed_version(self, target: str) -> int | None:
        return self._committed.get(target)

    def live_versions(self, target: str) -> tuple[int, ...]:
        return tuple(sorted(self._live.get(target, ())))

    def check_read(self, target: str, version: int | None, *, at: float) -> int:
        current = self._committed.get(target)
        if version is None:
            if current is None:
                raise FragmentContractError(f"state_target_uninitialized:{target}")
            return current
        value = _version(version, target)
        if value == current or value in self._live.get(target, ()):
            return value
        raise FragmentContractError(
            f"state_version_unavailable:{target}:{value}:committed={current}")

    def apply(self, effects: Sequence[StateEffect], *, at: float,
              sequence: int) -> dict[str, Any]:
        self._prune(at)
        for effect in effects:
            action = effect.action
            if action == "read":
                continue
            target = effect.target
            live = self._live.setdefault(target, set())
            if action == "write":
                current = self._committed.get(target)
                version = effect.version
                if version is None:
                    version = 1 if current is None else current + 1
                version = _version(version, target)
                if current is not None and version < current:
                    raise FragmentContractError(
                        f"state_write_behind_committed:{target}:{version}:{current}")
                live.add(version)
                if len(live) > self.version_limit:
                    raise FragmentContractError(
                        f"state_live_versions_exceeded:{target}:"
                        f"{len(live)}:{self.version_limit}")
                continue
            version = effect.version
            if version is None:
                raise FragmentContractError(f"state_effect_version_missing:{target}")
            version = _version(version, target)
            if version not in live:
                raise FragmentContractError(
                    f"state_version_not_live:{target}:{version}")
            if action == "rollback":
                if self._committed.get(target) == version:
                    raise FragmentContractError(
                        f"state_rollback_committed:{target}:{version}")
                live.discard(version)
                continue
            self._commit(target, version, effect.arbiter, sequence, at)
        return self.snapshot()

    def _commit(self, target: str, version: int, arbiter: str,
                sequence: int, at: float) -> None:
        key = target, at
        if key not in self._commit_base:
            self._commit_base[key] = self._committed.get(target)
        previous = self._commit_base[key]
        if previous is not None and version < previous:
            raise FragmentContractError(
                f"state_commit_behind_committed:{target}:{version}:{previous}")
        records = self._pending.setdefault(key, {})
        records.setdefault(version, (arbiter, sequence))
        if len(records) == 1:
            self._committed[target] = version
            self._retain(target, version, previous)
            return
        if any(not item[0] for item in records.values()):
            raise FragmentContractError(
                f"state_effect_conflict_requires_arbiter:{target}")
        winner = max(records.items(),
                     key=lambda item: (item[1][0], item[1][1]))
        self._committed[target] = winner[0]
        self._retain(target, winner[0], previous)

    def _retain(self, target: str, version: int, previous: int | None) -> None:
        live = self._live.setdefault(target, set())
        # Keep every version at or ahead of the pre-commit base until a later
        # timestamp. A same-time competing commit may still arrive after this
        # one, and dropping its candidate first would turn a declared arbiter
        # conflict into a spurious "not live" error.
        floor = previous if previous is not None else version
        keep = {version}
        if previous is not None:
            keep.add(previous)
        keep.update(item for item in live if item >= floor)
        if len(keep) > self.version_limit:
            raise FragmentContractError(
                f"state_live_versions_exceeded:{target}:"
                f"{len(keep)}:{self.version_limit}")
        self._live[target] = keep

    def _prune(self, at: float) -> None:
        stale = [key for key in self._pending if key[1] < at]
        for key in stale:
            del self._pending[key]
        stale = [key for key in self._commit_base if key[1] < at]
        for key in stale:
            del self._commit_base[key]


@dataclass(frozen=True)
class FragmentHandle:
    """Completion handle for one fragment occurrence, not a retained history."""

    fragment: ProgramFragment
    event: Event
    release_events: tuple[Event, ...] = ()
    grant_by_name: tuple[tuple[str, Event], ...] = ()
    release_by_name: tuple[tuple[str, Event], ...] = ()
    batched: bool = False
    repeat_count: int = 1
    sequence: int = 0


@dataclass
class _LiveRecord:
    fragment: ProgramFragment
    sequence: int
    handle: FragmentHandle | None = None
    started_ms: float | None = None


@dataclass(frozen=True)
class _PlannedAllocation:
    """One acquire/release lifetime; release boundaries stay per allocation."""

    name: str
    allocation: Allocation
    lifetime: str
    external_hold: Event | None = None
    local_release: bool = False
    #: Original producer-owned buffer identity/binding (owner, storage,
    #: ranges, generation, event, rank, stage, host_bytes, unit). It stays on
    #: this allocation until its real free; it is never applied globally to a
    #: different K/V entry.
    memory: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _AllocationPlan:
    stage: tuple[_PlannedAllocation, ...] = ()
    program: tuple[_PlannedAllocation, ...] = ()


class FragmentConductor:
    """Submit fragments into one shared Engine and bounded aggregate their results."""

    def __init__(
        self, engine: Engine, *,
        resource_bindings: Mapping[str, str] | None = None,
        pool_bindings: Mapping[str, str] | None = None,
        unit_conversions: Mapping[tuple[str, str], float] | None = None,
        state: Mapping[str, int] | None = None,
        state_version_limit: int = 64,
        detail: bool = False, cache_limit: int = 64,
        repeat_limit: int = 1_000_000, stream_limit: int = 1024,
        event_limit: int = 4096, prefix: str = "fragment:",
        memory=None, memory_replica: int = 0, memory_stage: int = 0,
        memory_pools: Sequence[str] = ("hbm",),
        memory_owner=None, memory_storage=None, memory_ranges=None,
        memory_generation=None, memory_event=None, memory_host_bytes: float = 0.0,
        memory_unit: str | None = None,
        memory_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        if not isinstance(engine, Engine):
            raise FragmentContractError("fragment_conductor_requires_engine")
        self.engine = engine
        self.memory = memory if memory is not None else engine.memory
        self.memory_replica = int(memory_replica)
        self.memory_stage = int(memory_stage)
        self.memory_pools = frozenset(str(item) for item in memory_pools)
        self.memory_owner = memory_owner
        self.memory_storage = memory_storage
        self.memory_ranges = memory_ranges
        self.memory_generation = memory_generation
        self.memory_event = memory_event
        self.memory_host_bytes = float(memory_host_bytes)
        self.memory_unit = memory_unit
        if memory is not None and engine.memory is not None and engine.memory is not memory:
            # PD endpoints can share one Engine with different ledgers. Each
            # task carries its own ledger; a second conductor must not rewrite
            # the first endpoint's default pool set.
            pass
        elif memory is not None:
            engine.bind_memory(memory, pools=self.memory_pools)
        self.memory_bindings = {
            str(name): dict(binding)
            for name, binding in dict(memory_bindings or {}).items()}
        self.resource_bindings = {str(key): str(value)
                                  for key, value in dict(resource_bindings or {}).items()}
        self.pool_bindings = {str(key): str(value)
                              for key, value in dict(pool_bindings or {}).items()}
        self.unit_conversions: dict[tuple[str, str], float] = {}
        for key, factor in dict(unit_conversions or {}).items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or not all(isinstance(item, str) and item for item in key)):
                raise FragmentContractError(
                    f"fragment_unit_conversion_key_invalid:{key!r}")
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) \
                    or not math.isfinite(float(factor)) or float(factor) <= 0:
                raise FragmentContractError(
                    f"fragment_unit_conversion_factor_invalid:{key}")
            self.unit_conversions[(key[0], key[1])] = float(factor)
        # The typed state ledger belongs to the shared RuntimeState. An
        # explicitly supplied StateLedger is adopted as-is; a plain mapping is
        # an initial-version seed for that one runtime-owned ledger. A
        # conductor never owns a second, independent version map.
        self.state = engine.ensure_state_ledger(
            state, version_limit=state_version_limit)
        self.detail = bool(detail)
        self.cache_limit = max(1, _nonnegative_int("fragment_cache_limit", cache_limit))
        self.repeat_limit = max(1, _nonnegative_int("fragment_repeat_limit", repeat_limit))
        self.stream_limit = max(1, _nonnegative_int("fragment_stream_limit", stream_limit))
        self.event_limit = max(1, _nonnegative_int("fragment_event_limit", event_limit))
        self.prefix = str(prefix)
        self._streams: dict[str, Event] = {}
        self._events: OrderedDict[tuple[str, int], Event] = OrderedDict()
        self._events_evicted = 0
        self._event_evicted_names: set[str] = set()
        self._event_history_ambiguous = False
        self._generations: dict[str, int] = {}
        self._live: OrderedDict[int, _LiveRecord] = OrderedDict()
        self._recent: deque[dict[str, Any]] | None = (
            deque() if self.detail else None)
        self._recent_dropped = 0
        self._aggregates: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._aggregates_evicted = 0
        self._unsupported: OrderedDict[str, int] = OrderedDict()
        self._unsupported_overflow = 0
        self._sequence = 0
        self._totals = {
            "fragments": 0, "repeats": 0, "batched": 0, "expanded": 0,
            "release_events": 0,
        }
        self._total_service: dict[str, float] = {}

    # -- public surface -------------------------------------------------
    def check(self, fragment: ProgramFragment, *, holds: Mapping[str, Any] | None = None
              ) -> tuple[str, ...]:
        """Return unsupported executable-semantics reasons without submitting."""
        if not isinstance(fragment, ProgramFragment):
            raise FragmentContractError("fragment_requires_program_fragment")
        reasons: list[str] = []
        for effect in fragment.allocations:
            if effect.lifetime == "event":
                reasons.append(f"allocation_scope_unsupported:event:{effect.name}")
        if fragment.repeats > 1 and any(
                effect.action in _STATE_ACTIONS_WRITE
                for effect in fragment.state_effects):
            reasons.append("repeat_state_writes_unsupported")
        if fragment.repeats > 1 and holds:
            reasons.append("repeat_external_holds_unsupported")
        used = set(fragment.service_work)
        priorities = {policy.priority for resource, policy in fragment.policies
                      if resource in used and policy.priority}
        if len(priorities) > 1:
            reasons.append("fragment_priority_ambiguous")
        elif priorities and fragment.priority and fragment.priority not in priorities:
            reasons.append(
                f"fragment_priority_conflict:{fragment.priority}:{sorted(priorities)}")
        reasons = list(dict.fromkeys(reasons))
        for reason in reasons:
            self._record_unsupported(reason)
        return tuple(reasons)

    def _unit_factor(self, declared_unit: str, physical: str,
                     declared_resource: str) -> float:
        expected = self.engine.resources[physical].unit
        if declared_unit == expected:
            return 1.0
        factor = self.unit_conversions.get((declared_unit, expected))
        if factor is None:
            raise FragmentContractError(
                f"fragment_work_unit_mismatch:{declared_resource}:"
                f"{declared_unit}:{expected}")
        return factor

    def _validate_binding(self, fragment: ProgramFragment) -> None:
        declared = {name: policy for name, policy in fragment.policies}
        for item in fragment.work.items:
            physical = self._resource(item.resource)
            resource = self.engine.resources[physical]
            policy = declared.get(item.resource)
            if policy is not None and policy.sharing != resource.policy:
                raise FragmentContractError(
                    f"fragment_resource_policy_mismatch:{item.resource}:"
                    f"{physical}:{policy.sharing}:{resource.policy}")
            self._unit_factor(item.unit, physical, item.resource)

    def _batchable(self, fragment: ProgramFragment,
                   holds: Mapping[str, Any]) -> bool:
        if (fragment.repeats <= 1 or fragment.allocations
                or fragment.state_effects or holds):
            return False
        used = {self._resource(item.resource)
                for item in fragment.work.issued() if item.amount}
        if not used:
            return True
        if any(self.engine.resources[name].policy != "fair" for name in used):
            # FIFO arbitration and per-repeat start times are observable.
            return False
        # A single fair flow is fungible across repeats. Coupled resources
        # progress together, so one larger coupled flow is equivalent. A
        # non-coupled multi-resource repeat can finish one resource early and
        # release its share before the next repeat; batching would hide that.
        return len(used) == 1 or fragment.coupled

    def submit(
        self, fragment: ProgramFragment, *,
        after: Sequence[Event] = (), inputs: Mapping[str, Any] | None = None,
        conditions: Mapping[str, bool] | None = None,
        holds: Mapping[str, Any] | None = None,
        event_bindings: Mapping[str, Any] | None = None,
        owner: str = "",
    ) -> FragmentHandle:
        """Admit one fragment. Work advances only through ``engine.advance``."""
        reasons = self.check(fragment, holds=holds)
        if reasons:
            raise UnsupportedFragmentSemantics(";".join(reasons))
        conditions = dict(conditions or {})
        holds = dict(holds or {})
        event_bindings = dict(event_bindings or {})
        self._validate_binding(fragment)
        stream = fragment.occurrence.stream if fragment.occurrence else ""
        if stream and stream not in self._streams \
                and len(self._streams) >= self.stream_limit:
            raise FragmentContractError(
                f"fragment_stream_limit_exceeded:{stream}")
        # Every rejection that depends on the event registry, allocation plan,
        # bindings or repeat bound happens before the first Engine submission
        # and before the conductor's sequence/live handle changes. The whole
        # event_out batch is checked against old names plus all new names at
        # once; no port is registered (or evicted) until the batch is known to
        # be publishable.
        output_generations = self._preflight_event_publication(
            fragment, event_bindings)
        base = self._base_dependencies(fragment, after, inputs)
        priority = self._priority(fragment)
        writes = self._active_effects(fragment, conditions, _STATE_ACTIONS_WRITE)
        reads = self._active_effects(fragment, conditions, {"read"})
        prospective_sequence = self._sequence + 1
        plan = self._plan_allocations(
            fragment, conditions, holds, prospective_sequence)
        output_plan = self._plan_event_outputs(
            fragment, plan, conditions, event_bindings)
        work = self._work(fragment)
        batchable = self._batchable(fragment, holds)
        if not batchable and fragment.repeats > self.repeat_limit:
            raise UnsupportedFragmentSemantics(
                "fragment_repeats_exceed_expansion_bound:"
                f"{fragment.repeats}:{self.repeat_limit}")
        self._sequence = prospective_sequence
        sequence = prospective_sequence
        name = f"{self.prefix}{fragment.name}#{sequence}"
        record = _LiveRecord(fragment, sequence)

        def check_reads(event: Event) -> None:
            at = event.started if event.started is not None else self.engine.now
            if record.started_ms is None:
                record.started_ms = at
            for effect in reads:
                self.state.check_read(effect.target, effect.version, at=at)

        def apply_writes(event: Event) -> None:
            at = event.time if event.time is not None else self.engine.now
            if writes:
                self.state.apply(writes, at=at, sequence=event.sequence)

        grant_by_name: dict[str, Event] = {}
        release_by_name: dict[str, Event] = {}
        release_events: list[Event] = []
        if batchable:
            def finish(event: Event) -> None:
                apply_writes(event)
                self._finish(record, event)

            work_complete = self._submit_task(
                name, self._work(fragment, fragment.repeats), base,
                (), (), priority, fragment.coupled, check_reads, finish, owner)
            batched = True
        else:
            batched = False
            work_complete = self.engine.future(name + ":work_complete")

            def finalize(event: Event) -> None:
                apply_writes(event)
                work_complete.started = (
                    record.started_ms if record.started_ms is not None
                    else event.started)
                self.engine.resolve(work_complete)
                self._finish(record, event)

            def issue(index: int, predecessors: Sequence[Event]) -> Event:
                first = index == 0
                last = index == fragment.repeats - 1
                # Program allocations are part of the first real admission
                # boundary, not a separate earlier grant. This gives all-or-
                # wait admission on one Engine: an unavailable Q cannot leave P
                # held by a half-admitted fragment while another fragment takes
                # Q. Program allocations stay live because the first repeat
                # does not free them; each is still released independently.
                acquire = (
                    tuple(item.allocation for item in plan.program)
                    if first else ()) + tuple(
                        item.allocation for item in plan.stage)
                free = tuple(item.allocation for item in plan.stage
                             if item.local_release)

                def completed(event: Event) -> None:
                    if last:
                        finalize(event)
                    else:
                        issue(index + 1, (event,))

                allocations = (plan.program if first else ()) + tuple(plan.stage)
                bindings = {
                    planned.allocation.name: planned.memory
                    for planned in allocations
                    if planned.memory is not None}
                task = self._submit_task(
                    f"{name}:r{index}", work, predecessors,
                    acquire, free, priority, fragment.coupled,
                    check_reads, completed, owner,
                    memory_bindings=bindings)
                if first:
                    for grant_index, planned in enumerate(plan.program):
                        grant_by_name[planned.name] = self.engine.start_boundary(
                            task, f"{name}:program_grant{grant_index}")
                    if fragment.repeats == 1:
                        for grant_index, planned in enumerate(plan.stage):
                            grant_by_name[planned.name] = (
                                self.engine.start_boundary(
                                    task, f"{name}:stage_grant{grant_index}"))
                return task

            issue(0, base)

            # One release task per allocation. Independent holds never merge
            # into one max-release; a local program release is not delayed by
            # an unrelated external hold.
            for index, planned in enumerate(plan.program):
                dependencies = ((work_complete,) if planned.external_hold is None
                                else (work_complete, planned.external_hold))
                release = self.engine.submit(
                    f"{name}:release{index}", after=dependencies,
                    free=(planned.allocation,), owner=owner)
                release_by_name[planned.name] = release
                release_events.append(release)
            for index, planned in enumerate(plan.stage):
                if planned.external_hold is None:
                    if planned.local_release and fragment.repeats == 1:
                        release_by_name.setdefault(planned.name, work_complete)
                    continue
                release = self.engine.submit(
                    f"{name}:stage_release{index}",
                    after=(work_complete, planned.external_hold),
                    free=(planned.allocation,), owner=owner)
                release_by_name[planned.name] = release
                release_events.append(release)

        handle = FragmentHandle(
            fragment=fragment, event=work_complete,
            release_events=tuple(release_events),
            grant_by_name=tuple(grant_by_name.items()),
            release_by_name=tuple(release_by_name.items()),
            batched=batched, repeat_count=fragment.repeats, sequence=sequence)
        record.handle = handle
        self._live[work_complete.sequence] = record
        output_events = self._resolve_output_events(
            fragment, output_plan, work_complete, grant_by_name,
            release_by_name, event_bindings)
        self._record_outputs(fragment, output_events, output_generations)
        if stream:
            self._streams[stream] = work_complete
        return handle

    def advance(
        self, *, until: Sequence[Event] = (), horizon: float = math.inf,
        first: bool = False, max_steps: int | None = None,
    ) -> tuple[Event, ...]:
        """Advance the shared Engine; default targets all live fragments."""
        if not until:
            until = tuple(record.handle.event for record in self._live.values()
                          if record.handle is not None)
        return self.engine.advance(
            until=until, horizon=horizon, first=first, max_steps=max_steps)

    def event(self, name: str, generation: int | None = None) -> Event:
        """Look up one recorded event without guessing across generations."""
        event = self._find_event(name, generation)
        if event is None:
            raise FragmentContractError(
                f"fragment_event_unbound:{name}"
                + ("" if generation is None else f":{generation}"))
        return event

    def record_event(self, name: str, event: Event,
                     generation: int | None = None) -> int:
        """Publish a real recorded event for event-port consumers.

        The registry is an LRU of completed/recorded handles. Once it reaches
        ``event_limit`` the oldest record is reclaimed; a later consumer that
        asks for that exact generation receives an explicit unbound/generation
        error rather than a different same-named event.
        """
        if not isinstance(name, str) or not name:
            raise FragmentContractError("fragment_event_record_name_invalid")
        if not isinstance(event, Event):
            raise FragmentContractError(f"fragment_event_record_invalid:{name}")
        if generation is None:
            generation = self._generations.get(name, -1) + 1
        generation = _nonnegative_int("fragment_event_generation", generation)
        key = str(name), generation
        if key in self._events:
            raise FragmentContractError(
                f"fragment_event_generation_already_recorded:{name}:{generation}")
        # A name-limit rejection must not evict a healthy old record first.
        if name not in self._generations \
                and len(self._generations) >= self.event_limit:
            raise FragmentContractError(
                f"fragment_event_name_limit_exceeded:{self.event_limit}")
        while len(self._events) >= self.event_limit:
            evicted, _ = self._events.popitem(last=False)
            self._events_evicted += 1
            if len(self._event_evicted_names) < self.event_limit:
                self._event_evicted_names.add(evicted[0])
            else:
                # The per-name miss set itself must stay bounded. Once history
                # is ambiguous at this scale, require explicit generations for
                # every implicit lookup rather than guessing across old names.
                self._event_evicted_names.clear()
                self._event_history_ambiguous = True
        self._events[key] = event
        self._generations[str(name)] = max(
            generation, self._generations.get(str(name), 0))
        return generation

    def summary(self, detail: bool | None = None) -> dict[str, Any]:
        """Bounded aggregate; ``detail`` only retains recent observations."""
        if detail is None:
            detail = self.detail
        result = {
            "engine": self.engine.summary(),
            "live_fragments": len(self._live),
            "totals": dict(self._totals),
            "service_work": dict(sorted(self._total_service.items())),
            "fragments": {name: dict(value)
                          for name, value in self._aggregates.items()},
            "aggregates_evicted": self._aggregates_evicted,
            "unsupported": dict(self._unsupported),
            "unsupported_overflow": self._unsupported_overflow,
            "state": self.state.snapshot(),
            "streams": len(self._streams),
            "recorded_events": len(self._events),
            "events_evicted": self._events_evicted,
            "event_history_ambiguous": self._event_history_ambiguous,
            "recent": [],
        }
        if detail and self._recent is not None:
            result["recent"] = list(self._recent)
            result["recent_dropped"] = self._recent_dropped
        return result

    # -- lowering -------------------------------------------------------
    def _resource(self, declared: str) -> str:
        name = self.resource_bindings.get(str(declared), str(declared))
        if name not in self.engine.resources:
            raise FragmentContractError(
                f"fragment_resource_unbound:{declared}->{name}")
        return name

    def _pool(self, declared: str) -> str:
        name = self.pool_bindings.get(str(declared), str(declared))
        if name not in self.engine.pools:
            raise FragmentContractError(f"fragment_pool_unbound:{declared}->{name}")
        return name

    def _priority(self, fragment: ProgramFragment) -> int:
        used = set(fragment.service_work)
        priorities = {policy.priority for resource, policy in fragment.policies
                      if resource in used and policy.priority}
        if len(priorities) > 1:
            raise UnsupportedFragmentSemantics(
                f"fragment_priority_ambiguous:{fragment.name}")
        if priorities:
            value = next(iter(priorities))
            if fragment.priority and fragment.priority != value:
                raise UnsupportedFragmentSemantics(
                    f"fragment_priority_conflict:{fragment.priority}:{value}")
            return value
        return fragment.priority

    def _work(self, fragment: ProgramFragment, multiply: int = 1) -> dict[str, float]:
        result: dict[str, float] = {}
        for item in fragment.work.issued():
            if not item.amount:
                continue
            name = self._resource(item.resource)
            factor = self._unit_factor(item.unit, name, item.resource)
            try:
                scaled = (float(item.amount) / factor) * multiply
            except OverflowError as exc:
                raise FragmentContractError(
                    f"fragment_batched_work_overflow:{fragment.name}") from exc
            result[name] = result.get(name, 0.0) + scaled
        for value in result.values():
            if not math.isfinite(value):
                raise FragmentContractError(
                    f"fragment_batched_work_overflow:{fragment.name}")
        return result

    def _active(self, condition: str, conditions: Mapping[str, bool],
                owner: str) -> bool:
        if not condition:
            return True
        if condition not in conditions:
            raise FragmentContractError(
                f"conditional_effect_unbound:{owner}:{condition}")
        return bool(conditions[condition])

    def _active_effects(self, fragment: ProgramFragment,
                        conditions: Mapping[str, bool],
                        actions: set[str]) -> tuple[StateEffect, ...]:
        result = []
        for effect in fragment.state_effects:
            if effect.action not in actions:
                continue
            if self._active(effect.condition, conditions, effect.target):
                result.append(effect)
        return tuple(result)

    def _as_event(self, value: Any, label: str) -> Event:
        if isinstance(value, FragmentHandle):
            return value.event
        if isinstance(value, Event):
            return value
        raise FragmentContractError(f"fragment_event_binding_invalid:{label}")

    def _as_events(self, value: Any, label: str) -> tuple[Event, ...]:
        if isinstance(value, (FragmentHandle, Event)):
            return (self._as_event(value, label),)
        if isinstance(value, (tuple, list)):
            return tuple(self._as_event(item, label) for item in value)
        raise FragmentContractError(f"fragment_event_binding_invalid:{label}")

    def _base_dependencies(self, fragment: ProgramFragment,
                           after: Sequence[Event],
                           inputs: Mapping[str, Any] | None) -> tuple[Event, ...]:
        provided = dict(inputs or {})
        events: list[Event] = [self._as_event(item, "after") for item in after]
        stream = fragment.occurrence.stream if fragment.occurrence else ""
        if stream and stream in self._streams:
            events.append(self._streams[stream])
        for group in ("tensor_in", "state_in", "event_in"):
            for port in getattr(fragment.ports, group):
                value = provided.get(port.name, _MISSING)
                if value is _MISSING:
                    if group == "event_in":
                        recorded = self._find_event(port.name, port.generation)
                        if recorded is not None:
                            events.append(recorded)
                            continue
                    if port.optional:
                        continue
                    raise FragmentContractError(
                        f"fragment_input_unbound:{port.name}")
                if value is None:
                    raise FragmentContractError(
                        f"fragment_input_unbound:{port.name}")
                if group == "event_in":
                    events.extend(self._as_events(value, port.name))
        return tuple(dict.fromkeys(events))

    def _find_event(self, name: str, generation: int | None) -> Event | None:
        if generation is not None:
            return self._events.get((name, generation))
        if self._event_history_ambiguous or name in self._event_evicted_names:
            raise FragmentContractError(
                f"fragment_event_generation_required:{name}")
        matches = [event for (recorded, _), event in self._events.items()
                   if recorded == name]
        if len(matches) > 1:
            raise FragmentContractError(
                f"fragment_event_generation_required:{name}")
        return matches[0] if matches else None

    def hold_async_reader(self, request: str, consumer: str, *,
                          storage=None, ranges=None, generation=None,
                          event=None) -> bool:
        """Bind a real async reader before its submit; no synthetic hold."""
        if self.memory is None:
            raise FragmentContractError("fragment_memory_not_attached")
        return self.memory.hold_for_async_consumer(
            self.memory_replica, str(request), str(consumer),
            storage=storage if storage is not None else self.memory_storage,
            ranges=ranges if ranges is not None else self.memory_ranges,
            generation=(generation if generation is not None
                        else self.memory_generation),
            event=event if event is not None else self.memory_event)

    def release_async_reader(self, request: str, consumer: str) -> bool:
        if self.memory is None:
            raise FragmentContractError("fragment_memory_not_attached")
        return self.memory.release_async_consumer(
            self.memory_replica, str(request), str(consumer))

    def _plan_allocations(self, fragment: ProgramFragment,
                          conditions: Mapping[str, bool],
                          holds: Mapping[str, Any],
                          sequence: int) -> _AllocationPlan:
        acquires: dict[str, tuple[AllocationEffect, Allocation]] = {}
        releases: dict[str, tuple[AllocationEffect, Allocation]] = {}
        ready_holds = {str(name): self._as_event(event, f"hold:{name}")
                       for name, event in holds.items()}
        for effect in fragment.allocations:
            if not self._active(effect.condition, conditions, effect.name):
                continue
            pool = self._pool(effect.pool)
            if effect.size > self.engine.pools[pool]:
                raise FragmentContractError(
                    f"fragment_allocation_exceeds_capacity:"
                    f"{effect.name}:{effect.size}:{pool}")
            if any(unit >= self.engine.pool_units[pool] for unit in effect.units):
                raise FragmentContractError(
                    f"fragment_allocation_unit_out_of_range:"
                    f"{effect.name}:{pool}")
            allocation = Allocation(
                f"{self.prefix}{fragment.name}#{sequence}:{effect.name}", pool,
                effect.size, effect.units)
            target = acquires if effect.action == "acquire" else releases
            if effect.name in target:
                raise FragmentContractError(
                    f"fragment_allocation_duplicate:{effect.name}")
            target[effect.name] = effect, allocation
        unknown_holds = set(ready_holds) - set(acquires)
        if unknown_holds:
            raise FragmentContractError(
                f"fragment_hold_unknown:{sorted(unknown_holds)[0]}")
        stage: list[_PlannedAllocation] = []
        program: list[_PlannedAllocation] = []
        for name, (effect, allocation) in acquires.items():
            release = releases.get(name)
            if release is not None:
                other = release[0]
                if (other.size != effect.size or other.units != effect.units
                        or other.pool != effect.pool
                        or other.lifetime != effect.lifetime):
                    raise FragmentContractError(
                        f"fragment_allocation_release_mismatch:{name}")
                if name in ready_holds:
                    raise FragmentContractError(
                        f"fragment_allocation_hold_conflict:{name}")
            elif name not in ready_holds:
                raise FragmentContractError(
                    f"fragment_allocation_release_missing:{name}")
            planned = _PlannedAllocation(
                name=name, allocation=allocation, lifetime=effect.lifetime,
                external_hold=ready_holds.get(name),
                local_release=release is not None,
                memory=self._allocation_binding(name))
            if effect.lifetime in {"stage", "slot"}:
                stage.append(planned)
            elif effect.lifetime == "program":
                program.append(planned)
        missing_grant = set(releases) - set(acquires)
        if missing_grant:
            raise FragmentContractError(
                f"fragment_allocation_grant_missing:{sorted(missing_grant)[0]}")
        # Program allocations stay live across repeats while the current repeat
        # also holds its stage/slot allocations. If that concurrent set cannot
        # fit, the fragment can never be admitted; reject instead of stalling.
        concurrent: dict[tuple[str, int], float] = {}
        for planned in tuple(program) + tuple(stage):
            allocation = planned.allocation
            for unit in allocation.units:
                key = allocation.pool, unit
                concurrent[key] = concurrent.get(key, 0.0) + allocation.size
        for (pool, unit), amount in concurrent.items():
            if amount > self.engine.pools[pool] + 4 * math.ulp(
                    float(self.engine.pools[pool])):
                raise FragmentContractError(
                    f"fragment_allocation_concurrent_exceeds_capacity:"
                    f"{pool}:unit{unit}:{amount}:{self.engine.pools[pool]}")
        return _AllocationPlan(tuple(stage), tuple(program))

    def _allocation_binding(self, name: str) -> dict | None:
        """Merge the conductor default with one real allocation identity."""
        defaults = {
            "owner": self.memory_owner, "storage": self.memory_storage,
            "ranges": self.memory_ranges,
            "generation": self.memory_generation, "event": self.memory_event,
            "host_bytes": self.memory_host_bytes, "stage": self.memory_stage,
        }
        if self.memory_unit is not None:
            defaults["unit"] = self.memory_unit
        result = {key: value for key, value in defaults.items()
                  if value is not None}
        override = self.memory_bindings.get(name)
        if override is None:
            matches = [(len(str(key)), str(key))
                       for key in self.memory_bindings
                       if name.endswith(":" + str(key))
                       or name.endswith("#" + str(key))]
            if matches:
                matches.sort(reverse=True)
                if len(matches) > 1 and matches[0][0] == matches[1][0]:
                    raise FragmentContractError(
                        f"fragment_memory_binding_ambiguous:{name}")
                override = self.memory_bindings[matches[0][1]]
        if override is not None:
            result.update({str(key): value
                           for key, value in dict(override).items()
                           if value is not None})
        return result or None

    def _submit_task(
        self, name: str, work: Mapping[str, float], after: Sequence[Event],
        acquire: Sequence[Allocation], free: Sequence[Allocation],
        priority: int, coupled: bool, on_start, on_complete, owner: str,
        memory_bindings: Mapping[str, Mapping] | None = None,
    ) -> Event:
        metadata = {"memory_replica": self.memory_replica,
                    "memory_stage": self.memory_stage,
                    "memory_owner": self.memory_owner,
                    "memory_storage": self.memory_storage,
                    "memory_ranges": self.memory_ranges,
                    "memory_generation": self.memory_generation,
                    "memory_event": self.memory_event,
                    "memory_host_bytes": self.memory_host_bytes,
                    "memory_unit": self.memory_unit}
        return self.engine.submit(
            name, work, after=tuple(after), acquire=tuple(acquire),
            free=tuple(free), priority=priority, coupled=coupled,
            on_start=on_start, on_complete=on_complete, owner=owner,
            metadata={key: value for key, value in metadata.items()
                      if value is not None},
            memory=self.memory, memory_pools=self.memory_pools,
            memory_bindings=memory_bindings)

    def _preflight_event_publication(
        self, fragment: ProgramFragment,
        event_bindings: Mapping[str, Any],
    ) -> dict[str, int]:
        """Validate the whole event_out batch without touching the registry.

        The union of existing names and every name this fragment will publish
        must fit ``event_limit``. Every proposed generation must be unused, and
        every explicit binding must resolve. A rejection therefore happens
        before an Engine task, a live handle, an eviction or a sequence bump;
        exactly-fitting batches still publish normally.
        """
        ports = fragment.ports.event_out
        output_names = {port.name for port in ports}
        unknown = set(event_bindings) - output_names
        if unknown:
            raise FragmentContractError(
                f"fragment_event_binding_unknown:{sorted(unknown)[0]}")
        existing_names = set(self._generations)
        if len(existing_names | output_names) > self.event_limit:
            raise FragmentContractError(
                f"fragment_event_name_limit_exceeded:{self.event_limit}")
        existing_keys = set(self._events)
        proposed_keys: set[tuple[str, int]] = set()
        result: dict[str, int] = {}
        for port in ports:
            generation = port.generation
            if generation is None:
                generation = self._generations.get(port.name, -1) + 1
            key = port.name, generation
            if key in existing_keys or key in proposed_keys:
                raise FragmentContractError(
                    f"fragment_event_generation_already_recorded:"
                    f"{port.name}:{generation}")
            if port.name in event_bindings:
                self._as_event(event_bindings[port.name],
                               f"event:{port.name}")
            result[port.name] = generation
            proposed_keys.add(key)
        return result

    def _unique_allocation_event(self, kind: str, port_name: str,
                                 table: Mapping[str, int]) -> tuple[str, str]:
        if not table:
            raise UnsupportedFragmentSemantics(
                f"fragment_event_port_unbound:{kind}:{port_name}")
        if port_name in table:
            if table[port_name] != 1:
                raise UnsupportedFragmentSemantics(
                    f"fragment_event_port_ambiguous:{kind}:{port_name}")
            return kind, port_name
        if len(table) == 1:
            name, count = next(iter(table.items()))
            if count == 1:
                return kind, name
        raise UnsupportedFragmentSemantics(
            f"fragment_event_port_ambiguous:{kind}:{port_name}")

    def _plan_event_outputs(
        self, fragment: ProgramFragment, plan: _AllocationPlan,
        conditions: Mapping[str, bool],
        event_bindings: Mapping[str, Any],
    ) -> dict[str, tuple[str, ...]]:
        output_names = {port.name for port in fragment.ports.event_out}
        unknown = set(event_bindings) - output_names
        if unknown:
            raise FragmentContractError(
                f"fragment_event_binding_unknown:{sorted(unknown)[0]}")
        grants: dict[str, int] = {}
        releases: dict[str, int] = {}
        for planned in plan.program:
            grants[planned.name] = 1
            releases[planned.name] = 1
        for planned in plan.stage:
            grants[planned.name] = fragment.repeats
            if planned.external_hold is not None:
                releases[planned.name] = 1
            elif planned.local_release:
                releases[planned.name] = fragment.repeats
        result: dict[str, tuple[str, ...]] = {}
        for port in fragment.ports.event_out:
            if port.name in event_bindings:
                self._as_event(event_bindings[port.name],
                               f"event:{port.name}")
                result[port.name] = ("explicit", port.name)
                continue
            kind = port.event
            if kind in {"complete", "ready"}:
                result[port.name] = ("work",)
            elif kind == "grant":
                result[port.name] = self._unique_allocation_event(
                    kind, port.name, grants)
            elif kind == "release":
                result[port.name] = self._unique_allocation_event(
                    kind, port.name, releases)
            elif kind == "commit":
                if not any(
                        effect.action == "commit"
                        and self._active(effect.condition, conditions,
                                         effect.target)
                        for effect in fragment.state_effects):
                    raise UnsupportedFragmentSemantics(
                        f"fragment_event_port_without_commit:{port.name}")
                result[port.name] = ("work",)
            else:
                raise UnsupportedFragmentSemantics(
                    f"fragment_event_port_requires_explicit_binding:"
                    f"{kind}:{port.name}")
        return result

    def _resolve_output_events(
        self, fragment: ProgramFragment, plan: Mapping[str, tuple[str, ...]],
        work_complete: Event, grant_by_name: Mapping[str, Event],
        release_by_name: Mapping[str, Event],
        event_bindings: Mapping[str, Any],
    ) -> dict[str, Event]:
        result: dict[str, Event] = {}
        for port in fragment.ports.event_out:
            key = plan[port.name]
            if key[0] == "explicit":
                result[port.name] = self._as_event(
                    event_bindings[port.name], f"event:{port.name}")
            elif key[0] == "work":
                result[port.name] = work_complete
            elif key[0] == "grant":
                event = grant_by_name.get(key[1])
                if event is None:
                    raise FragmentContractError(
                        f"fragment_event_grant_missing:{key[1]}")
                result[port.name] = event
            elif key[0] == "release":
                event = release_by_name.get(key[1])
                if event is None:
                    raise FragmentContractError(
                        f"fragment_event_release_missing:{key[1]}")
                result[port.name] = event
        return result

    def _record_outputs(self, fragment: ProgramFragment,
                        events: Mapping[str, Event],
                        generations: Mapping[str, int]) -> None:
        for port in fragment.ports.event_out:
            self.record_event(port.name, events[port.name],
                              generations[port.name])

    def _record_unsupported(self, reason: str) -> None:
        if reason in self._unsupported:
            self._unsupported[reason] += 1
            return
        if len(self._unsupported) >= 32:
            self._unsupported_overflow += 1
            return
        self._unsupported[reason] = 1

    def _finish(self, record: _LiveRecord, event: Event) -> None:
        handle = record.handle
        completion_event = handle.event if handle is not None else event
        self._live.pop(completion_event.sequence, None)
        fragment = record.fragment
        release_boundaries = len(handle.release_by_name) if handle is not None else 0
        grants = tuple(handle.grant_by_name) if handle is not None else ()
        service_work = self._work(fragment, fragment.repeats)
        start_ms = record.started_ms if record.started_ms is not None else event.started
        observation = {
            "name": fragment.name,
            "sequence": record.sequence,
            "completion_ms": completion_event.time,
            "start_ms": start_ms,
            "repeats": fragment.repeats,
            "batched": bool(handle and handle.batched),
            "service_work": dict(service_work),
            "useful_work": fragment.work.by_resource("useful"),
            "grant_events": len(grants),
            "release_events": release_boundaries,
            "state": self.state.snapshot(),
        }
        self._totals["fragments"] += 1
        self._totals["repeats"] += fragment.repeats
        if handle and handle.batched:
            self._totals["batched"] += 1
        else:
            self._totals["expanded"] += 1
        self._totals["release_events"] += release_boundaries
        for resource, amount in service_work.items():
            self._total_service[resource] = (
                self._total_service.get(resource, 0.0) + float(amount))
        if self._recent is not None:
            if len(self._recent) >= self.cache_limit:
                self._recent.popleft()
                self._recent_dropped += 1
            self._recent.append(observation)
        aggregate = self._aggregates.pop(fragment.name, None)
        if aggregate is None:
            aggregate = {
                "count": 0, "repeats": 0, "batched": 0,
                "first_ms": start_ms, "last_ms": completion_event.time,
                "min_span_ms": None, "max_span_ms": None,
            }
        aggregate["count"] += 1
        aggregate["repeats"] += fragment.repeats
        aggregate["batched"] += int(bool(handle and handle.batched))
        aggregate["last_ms"] = completion_event.time
        if start_ms is not None and completion_event.time is not None:
            span = completion_event.time - start_ms
            previous = aggregate["min_span_ms"]
            aggregate["min_span_ms"] = span if previous is None else min(previous, span)
            previous = aggregate["max_span_ms"]
            aggregate["max_span_ms"] = span if previous is None else max(previous, span)
        self._aggregates[fragment.name] = aggregate
        while len(self._aggregates) > self.cache_limit:
            self._aggregates.popitem(last=False)
            self._aggregates_evicted += 1


def _scoped_allocations(
    label: str, items: Sequence[Allocation], *,
    root: str | None = None, bindings: Mapping[str, str] | None = None,
    persistent: frozenset[str] = frozenset(),
) -> tuple[Allocation, ...]:
    """Bind allocations by the template's invocation/output/panel scope.

    ``bindings`` comes from ``PipelineTemplate.allocation_bindings``. The
    invocation scope is the explicit template root (never derived from a display
    task name); output allocations span panels of one output; panel allocations
    remain tile-local. Setup/body/drain use this same function.
    """
    scope_root = root or _pipeline_scope(label)
    table = dict(bindings or {})
    result = []
    for item in items:
        scope = table.get(item.name)
        if scope == "invocation":
            prefix = scope_root
        elif scope == "output":
            digits = "".join(ch for ch in item.name if ch.isdigit())
            prefix = f"{scope_root}:output{digits or 0}"
        elif scope == "panel":
            prefix = label
        elif item.name in persistent:
            prefix = scope_root
        else:
            prefix = label
        result.append(Allocation(f"{prefix}:{item.name}", item.pool,
                                 item.size, item.units))
    return tuple(result)


def _indexed_scoped_allocations(
    items: Sequence[Allocation], *, launch_uid: str, owner: int, index: int,
    output: int | None, panel: int | None, slot: int | None,
    bindings: Mapping[str, Mapping],
    persistent: frozenset[str],
) -> tuple[Allocation, ...]:
    """Bind one indexed tile's allocations by real scope ordinal.

    The producer's binding row is authoritative.  A row with an explicit
    physical identity is used verbatim; otherwise the local key is
    ``launch_uid/owner/scope/ordinal/symbol``.  Indexed symbols are stable
    names such as ``panel``/``accumulator``, so no digit parsing is possible.
    """
    result = []
    for item in items:
        row = bindings.get(item.name)
        row = row if isinstance(row, _AbcMapping) else {}
        scope = row.get("scope")
        explicit = None
        for key in ("identity", "physical", "physical_name", "external"):
            value = row.get(key)
            if isinstance(value, str) and value:
                explicit = value
                break
        if explicit is not None:
            name = explicit
        elif scope == "invocation":
            name = f"{launch_uid}:owner{owner}:invocation0:{item.name}"
        elif scope == "output":
            ordinal = output if output is not None else int(
                row.get("ordinal", 0))
            name = f"{launch_uid}:owner{owner}:output{ordinal}:{item.name}"
        elif scope == "slot":
            ordinal = slot if slot is not None else int(row.get("ordinal", 0))
            name = f"{launch_uid}:owner{owner}:slot{ordinal}:{item.name}"
        elif scope == "panel" or item.name not in persistent:
            # A panel is one concrete absolute panel, not a recurring wave
            # residue: using the absolute index keeps overlapping slots apart.
            ordinal = index if index is not None else int(row.get("ordinal", 0))
            name = f"{launch_uid}:owner{owner}:panel{ordinal}:{item.name}"
        else:
            name = f"{launch_uid}:owner{owner}:invocation0:{item.name}"
        result.append(Allocation(name, item.pool, item.size, item.units))
    return tuple(result)


def _pipeline_scope(label: str) -> str:
    # ``label`` is a tile task name (name:tileN) or a setup/drain label.
    marker = ":tile"
    index = label.find(marker)
    return label[:index] if index >= 0 else label.split(":setup", 1)[0].split(":drain", 1)[0]


def _drop_periodic_run(engine: Engine, run_id: int) -> None:
    engine._periodic_runs.pop(run_id, None)
    for sequence, metadata in tuple(engine._periodic_tasks.items()):
        if metadata and metadata[0] == run_id:
            engine._periodic_tasks.pop(sequence, None)


def start_pipeline_template(
    engine: Engine, template, name: str, ready: Event, *, owner: str = "",
    quiescent: bool = False,
) -> Event:
    """Submit a finite ``PipelineTemplate`` lazily on the caller's Engine.

    This is the T04 side of the T10 handoff. It consumes the duck-typed
    ``PipelinePhase(stages, count)`` / ``PipelineTemplate(slots, phases, setup,
    drain, setup_ms, minimum_ms)`` contract without importing ``program.py``.
    At most ``slots`` tiles are in flight; each retire issues the next real tile.
    ``Stage.after``/``previous``, per-stage work/caps/placement/priority,
    coupled progress and per-tile scoped allocations are preserved. No phase is
    averaged into one body and no whole-phase allocation maximum is reserved.
    An indexed template is selected by absolute ``index_of(i)`` and
    ``phases[phase_id].stages_for(i)``; it never materializes every panel.
    """
    if not isinstance(engine, Engine):
        raise FragmentContractError("pipeline_template_requires_engine")
    phases = tuple(getattr(template, "phases", ()))
    slots = int(getattr(template, "slots", 0))
    if slots < 1:
        raise FragmentContractError("pipeline_template_requires_positive_slots")
    indexed_panels = getattr(template, "indexed_panels", None)
    index_of = getattr(template, "index_of", None)
    has_indexed_panels = indexed_panels is not None
    has_index_of = index_of is not None
    if has_indexed_panels != has_index_of:
        raise FragmentContractError(
            "pipeline_indexed_template_requires_both_fields")
    indexed = has_indexed_panels and has_index_of
    if indexed:
        if (type(indexed_panels) is not int or indexed_panels < 0
                or not callable(index_of)):
            raise FragmentContractError("pipeline_indexed_panels_invalid")
        if any(type(getattr(phase, "count", 0)) is not int
               or getattr(phase, "count") != 1 for phase in phases):
            raise FragmentContractError(
                "pipeline_indexed_phase_count_must_be_one")
        if any(getattr(phase, "end", None) is not None for phase in phases):
            raise FragmentContractError(
                "pipeline_indexed_phase_end_must_be_none")
        if indexed_panels and not phases:
            raise FragmentContractError(
                "pipeline_indexed_template_requires_catalog")
        total = indexed_panels
    else:
        total = sum(int(getattr(phase, "count", 0)) for phase in phases)
        if total < 0:
            raise FragmentContractError("pipeline_template_count_invalid")
    for phase in phases:
        if not tuple(getattr(phase, "stages", ())):
            raise FragmentContractError("pipeline_template_phase_requires_stages")
        if int(getattr(phase, "count", 0)) < 1:
            raise FragmentContractError("pipeline_template_phase_requires_positive_count")
    # Names acquired and freed at different phase boundaries are resident
    # across tiles (accumulator/KV/workspace). Names acquired and freed inside
    # the same phase are tile-local.
    acquire_phase: dict[str, set[int]] = {}
    free_phase: dict[str, set[int]] = {}
    for phase_index, phase in enumerate(phases):
        for stage in tuple(getattr(phase, "stages", ())):
            for allocation in tuple(getattr(stage, "acquire", ())):
                acquire_phase.setdefault(allocation.name, set()).add(phase_index)
            for allocation in tuple(getattr(stage, "free", ())):
                free_phase.setdefault(allocation.name, set()).add(phase_index)
    persistent_names = frozenset(
        name for name, acquired in acquire_phase.items()
        if not free_phase.get(name)
        or not free_phase[name].issubset(acquired))
    binding_rows = tuple(getattr(template, "allocation_bindings", ()) or ())
    allocation_bindings = {
        str(row.get("name")): str(row.get("scope"))
        for row in binding_rows if isinstance(row, _AbcMapping) and row.get("name")}
    indexed_binding_rows = {
        str(row.get("name")): dict(row)
        for row in binding_rows
        if isinstance(row, _AbcMapping) and row.get("name")}
    # Immutable, allocation-free Stage validation can be shared by every
    # repeat. Acquire/free stages still bind each occurrence's allocation id and
    # therefore keep Stage.prepare dynamic.
    prepared_cache: dict[int, object] = {}

    def prepared_for(stage):
        if stage.acquire or stage.free or stage.program is not None:
            return None
        key = (
            stage.name, tuple(sorted(stage.work.items())), stage.latency,
            tuple(sorted(stage.caps.items())),
            tuple(sorted((name, tuple(units))
                         for name, units in stage.placement.items())),
            stage.coupled, stage.priority,
            tuple(sorted(stage.work_units.items())),
        )
        prepared = prepared_cache.get(key)
        if prepared is None:
            prepared = stage.prepare(engine, f"{name}:prepare", lambda items: ())
            prepared_cache[key] = prepared
        return prepared

    setup = tuple(getattr(template, "setup", ()))
    drain = tuple(getattr(template, "drain", ()))
    # Setup/body/drain share the same invocation symbols. The producer's
    # allocation_bindings scans body phases; any allocation first visible in
    # setup/drain is invocation-scoped here rather than getting a new label.
    template_owner = getattr(template, "owner", 0)
    if indexed and (type(template_owner) is not int or template_owner < 0):
        raise FragmentContractError("pipeline_template_owner_invalid")
    for stage in (*setup, *drain):
        for allocation in (*stage.acquire, *stage.free):
            allocation_bindings.setdefault(allocation.name, "invocation")
            indexed_binding_rows.setdefault(allocation.name, {
                "name": allocation.name, "scope": "invocation",
                "owner": template_owner,
            })

    def scoped_at(label: str, index: int, output: int | None,
                  panel: int | None, slot: int | None):
        if indexed:
            return lambda items: _indexed_scoped_allocations(
                items, launch_uid=name, owner=template_owner, index=index,
                output=output, panel=panel, slot=slot,
                bindings=indexed_binding_rows, persistent=persistent_names)
        return lambda items: _scoped_allocations(
            label, items, root=name, bindings=allocation_bindings,
            persistent=persistent_names)

    setup_ms = float(getattr(template, "setup_ms", 0.0))
    minimum_ms = float(getattr(template, "minimum_ms", 0.0))
    if setup_ms < 0 or minimum_ms < 0:
        raise FragmentContractError("pipeline_template_negative_fixed_time")

    def indexed_mapping(index: int):
        mapped = index_of(index)
        if not isinstance(mapped, tuple) or len(mapped) != 5:
            raise FragmentContractError("pipeline_index_mapping_invalid")
        phase_id, end, output, panel, slot = mapped
        if (type(phase_id) is not int or not 0 <= phase_id < len(phases)
                or type(end) is not int or not index < end <= total
                or type(output) is not int or output < 0
                or type(panel) is not int or panel < 0
                or type(slot) is not int or not 0 <= slot < slots):
            raise FragmentContractError("pipeline_index_mapping_invalid")
        return phase_id, end, output, panel, slot

    drain_index = total - 1 if total else 0
    if indexed and total:
        (_drain_phase, _drain_end, drain_output,
         drain_panel, drain_slot) = indexed_mapping(drain_index)
    else:
        drain_output = drain_panel = drain_slot = 0

    current = ready
    if setup_ms:
        current = engine.submit(name + ":setup", after=(ready,),
                                latency=setup_ms, owner=owner)
    if setup:
        local: dict[str, Event] = {}
        for stage in setup:
            deps = tuple(local[key] for key in stage.after)
            scoped = scoped_at(name + ":setup:" + stage.name, 0, 0, 0, 0)
            local[stage.name] = stage.submit(
                engine, f"{name}:setup:{stage.name}", deps or (current,),
                scoped, owner, prepared=prepared_for(stage))
        current = engine.submit(name + ":setup_complete",
                                after=tuple(local.values()) or (current,),
                                owner=owner)
    observation = (engine.submit(name + ":span", after=(current,),
                                 latency=minimum_ms, owner=owner)
                   if minimum_ms > 0 else current)
    complete = engine.future(name + ":complete")
    run_id = complete.sequence
    periodic_run = {
        "run_id": run_id, "template": template, "name": name, "owner": owner,
        "phase_ordinal": 0,
        "phase_end": int(getattr(phases[0], "count", 0)) if phases else 0,
        "next": 0, "finished": 0, "slots": (),
        "live_tiles": set(), "retire_tiles": {}, "stage_tail": {}, "retire": None,
        "indexed": indexed,
    }
    engine._periodic_runs[run_id] = periodic_run
    if (quiescent and not indexed and phases
            and all(tuple(phase.stages) == tuple(phases[0].stages)
                    for phase in phases)):
        # A fully quiescent, dependency-free, allocation-free homogeneous phase
        # run can be integrated as one meta-cycle: every stage carries K times
        # its real work, the barrier preserves the tile boundary, and the
        # caller guarantees no external submit until completion. Anything with
        # after/previous/alloc/latency/tail falls back to the detailed lazy
        # path below.
        batch_count = total
        batch_stages = tuple(getattr(phases[0], "stages", ()))
        batchable = (
            batch_count > 1
            and all(not stage.after and not stage.previous
                    and not stage.acquire and not stage.free
                    and stage.program is None and stage.latency == 0
                    for stage in batch_stages))
        if batchable:
            barrier = []
            for stage in batch_stages:
                work = {resource: amount * batch_count
                        for resource, amount in stage.work.items()}
                batched = replace(stage, work=work)
                barrier.append(batched.submit(
                    engine, f"{name}:batch:{stage.name}",
                    (current,), lambda items: _scoped_allocations(
                        name, items, root=name,
                        bindings=allocation_bindings), owner,
                    prepared=prepared_for(batched)))
            periodic_run.update({
                "next": total, "finished": total, "phase_ordinal": 0,
                "phase_end": total, "stage_tail": {},
                "last_events": tuple(barrier) or (current,),
                "total": total, "persistent_names": persistent_names,
                "batch_mode": True,
            })
            batch_complete = engine.submit(
                name + ":batch_complete", after=tuple(barrier) or (current,),
                on_complete=lambda _: drain_and_resolve(), owner=owner)
            engine._periodic_tasks[batch_complete.sequence] = (
                run_id, total, len(batch_stages), "batch")
            periodic_run["slots"] = ()
            batch_mode = True
        else:
            batch_mode = False
    else:
        batch_mode = False
    if not batch_mode:
        periodic_run.update({
            "next": 0, "finished": 0, "phase_ordinal": 0,
            "phase_end": int(getattr(phases[0], "count", 0)) if phases else 0,
            "stage_tail": {}, "last_events": (current,), "retire": None,
            "total": total, "persistent_names": persistent_names,
        })

    def drop_run() -> None:
        _drop_periodic_run(engine, run_id)

    def select_stages(index: int):
        if indexed:
            phase_id, end, output, panel, slot = indexed_mapping(index)
            # The producer's end is an absolute exclusive phase bound. Keep it
            # for observation, but choose the class afresh for every absolute
            # index: class ids are catalog ids, not an ordered cursor.
            periodic_run["phase_ordinal"] = phase_id
            periodic_run["phase_end"] = end
            phase = phases[phase_id]
            selector = getattr(phase, "stages_for", None)
            stages = (selector(index) if callable(selector)
                      else getattr(phase, "stages"))
            selected = tuple(stages)
            if not selected:
                raise FragmentContractError(
                    "pipeline_indexed_phase_requires_stages")
            return selected, (output, panel, slot)
        while (periodic_run["phase_ordinal"] + 1 < len(phases)
               and index >= periodic_run["phase_end"]):
            ordinal = periodic_run["phase_ordinal"] + 1
            periodic_run["phase_ordinal"] = ordinal
            periodic_run["phase_end"] += int(
                getattr(phases[ordinal], "count", 0))
        selected = tuple(getattr(phases[periodic_run["phase_ordinal"]],
                                 "stages"))
        if not selected:
            raise FragmentContractError("pipeline_template_phase_requires_stages")
        return selected, None

    def drain_and_resolve() -> None:
        if not drain:
            engine.resolve(complete)
            drop_run()
            return
        local: dict[str, Event] = {}
        for stage in drain:
            deps = [local[key] for key in stage.after]
            deps.extend(periodic_run["stage_tail"][key]
                        for key in stage.previous
                        if key in periodic_run["stage_tail"])
            scoped = scoped_at(name + ":drain:" + stage.name, drain_index,
                               drain_output, drain_panel, drain_slot)
            local[stage.name] = stage.submit(
                engine, f"{name}:drain:{stage.name}",
                tuple(deps) or periodic_run["last_events"], scoped, owner,
                prepared=prepared_for(stage))

        def finished(_event: Event) -> None:
            engine.resolve(complete)
            drop_run()

        engine.submit(name + ":drain_complete", after=tuple(local.values()),
                      on_complete=finished, owner=owner)

    def retire(_event: Event) -> None:
        finished = periodic_run["finished"] + 1
        periodic_run["finished"] = finished
        tile_index = periodic_run["retire_tiles"].pop(_event.sequence, None)
        if tile_index is not None:
            periodic_run["live_tiles"].discard(tile_index)
        periodic_run["slots"] = tuple(sorted(
            tile - finished for tile in periodic_run["live_tiles"]))
        if finished == total:
            drain_and_resolve()
            return
        issue()

    def issue() -> None:
        index = periodic_run["next"]
        if index >= total:
            return
        stages, coordinates = select_stages(index)
        periodic_run["next"] = index + 1
        local: dict[str, Event] = {}
        # One immutable snapshot per panel: a same-panel ``after`` edge never
        # masquerades as the previous panel's stream tail.
        previous = dict(periodic_run["stage_tail"])
        label = f"{name}:tile{index}"
        if indexed:
            output, panel, slot = coordinates
            scoped = scoped_at(label, index, output, panel, slot)
        else:
            scoped = scoped_at(label, index, None, None, None)
        for stage in stages:
            deps = [local[key] for key in stage.after]
            deps.extend(previous[key] for key in stage.previous
                        if key in previous)
            if stage.name in previous:
                deps.append(previous[stage.name])
            event = stage.submit(
                engine, f"{label}:{stage.name}",
                tuple(deps) or (current,), scoped, owner,
                prepared=prepared_for(stage))
            local[stage.name] = event
            periodic_run["stage_tail"][stage.name] = event
            engine._periodic_tasks[event.sequence] = (
                run_id, index, len(local) - 1, "stage")
        periodic_run["last_events"] = tuple(local.values())
        retire_event = engine.submit(f"{label}:retire",
                                     after=periodic_run["last_events"],
                                     on_complete=retire, owner=owner)
        periodic_run["retire"] = retire_event
        engine._periodic_tasks[retire_event.sequence] = (
            run_id, index, len(stages), "retire")
        periodic_run["retire_tiles"][retire_event.sequence] = index
        periodic_run["live_tiles"].add(index)
        periodic_run["slots"] = tuple(sorted(
            tile - periodic_run["finished"]
            for tile in periodic_run["live_tiles"]))

    def begin(_event: Event) -> None:
        if not total:
            drain_and_resolve()
            return
        for _ in range(min(slots, total)):
            issue()

    if not batch_mode:
        engine.submit(name + ":begin", after=(current,), on_complete=begin,
                      owner=owner)
    return engine.submit(name + ":complete_outer",
                         after=(complete, observation), owner=owner)


# Capability evidence is the real callable's integer marker, never a producer
# self-report.  The indexed reader above is connected before this is published;
# old readers still see no marker and keep the exact legacy encoding.
start_pipeline_template.indexed_panels_abi = 1
