"""Focused M ledger tests: exact union, atomic grants, readers and deferral.

These tests exercise ``runtime.ledger`` through its public frozen protocol on
the documented RuntimeState surface (``uid`` + ``capacity_domains``).  The
byte-union oracle expands ranges into integer byte sets independently; no
production counter is used to compute the expected bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mindie_sim.prediction.runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    BackingKey,
    CapacityDomainState,
    EventKey,
    OccurrenceKey,
    PoolKey,
)
from mindie_sim.prediction.runtime.ledger import (
    Ledger,
    LedgerError,
    normalise_ranges,
    ranges_bytes,
)


@dataclass
class LedgerState:
    """Only the canonical mutable root surface the ledger is allowed to touch."""

    uid: str = "runtime-test"
    capacity_domains: dict[str, CapacityDomainState] = field(default_factory=dict)


def _pool(name="hbm", *, domain="memory", rank=0, capacity=8192):
    pool = PoolKey(physical_domain=domain, rank=rank, tier="hbm", name=name)
    return pool, capacity


def _occurrence(ordinal=0, *, launch=1):
    return OccurrenceKey(runtime_uid="runtime-test", launch=launch, ordinal=ordinal)


def _allocation(ordinal=0, *, symbol=0, generation=0, launch=1):
    return AllocationKey(
        occurrence=_occurrence(ordinal, launch=launch),
        scope_kind="invocation", scope_ordinal=ordinal, symbol=symbol,
        generation=generation,
    )


def _binding(
    key, pool, amount, *, backing=None, ranges=(), acquire=True,
    parent=None, owner="", generation=0,
):
    return AllocationBinding(
        key=key, pool=pool, amount=float(amount),
        backing=backing, byte_ranges=tuple(ranges), reservation_parent=parent,
        owner=owner, acquire=acquire,
    )


def _event(sequence, *, done=False, generation=0):
    class _E:
        def __init__(self):
            self.sequence = sequence
            self.time = 0.0 if done else None
            self.generation = generation
            self.done = done

    return _E()


def _bytes(ranges):
    return sum(end - start for start, end in ranges)


def _byte_set(ranges):
    result = set()
    for start, end in ranges:
        result.update(range(start, end))
    return result


def _fresh(capacity=8192, *, domain="memory", rank=0):
    state = LedgerState()
    pool, cap = _pool(domain=domain, rank=rank, capacity=capacity)
    Ledger.ensure_domain(
        state, domain, capacities={pool: float(capacity)})
    return state, pool


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([(0, 10), (5, 15), (20, 20)], ((0, 15),)),
        ([(8, 12), (0, 4), (4, 8)], ((0, 12),)),
        ([(3, 4), (9, 10)], ((3, 4), (9, 10))),
        (None, ((0, 7),)),
    ],
)
def test_range_normalisation_is_canonical(raw, expected):
    extent = 7 if raw is None else 0
    assert normalise_ranges(raw, extent=extent) == expected
    assert ranges_bytes(normalise_ranges(raw, extent=extent)) == ranges_bytes(expected)


def test_overlapping_backings_charge_union_not_sum():
    state, pool = _fresh()
    key = BackingKey(physical_domain="memory", rank=0, tier="hbm",
                     storage_id="kv", generation=0)
    first = _allocation(0, symbol=1)
    second = _allocation(1, symbol=2)
    grant = Ledger.try_acquire(
        state,
        (_binding(first, pool, 4096, backing=key, ranges=((0, 4096),)),
         _binding(second, pool, 4096, backing=key, ranges=((2048, 6144),))),
        holder=_occurrence(0), at=_event(1, done=True))
    assert grant.granted
    assert state.capacity_domains["memory"].current[pool] == 6144
    assert state.capacity_domains["memory"].epoch == 1

    result = Ledger.release(state, first, after=_event(2, done=True))
    assert result.released and result.freed_by_pool == ((pool, 2048.0),)
    assert state.capacity_domains["memory"].current[pool] == 4096
    assert _bytes(state.capacity_domains["memory"].backings[key].segments) == 4096


def test_rejected_batch_changes_nothing():
    state, pool = _fresh(8192)
    first = _allocation(0, symbol=1)
    second = _allocation(1, symbol=2)
    domain = state.capacity_domains["memory"]
    before = dict(domain.current)
    epoch = domain.epoch
    grant = Ledger.try_acquire(
        state,
        (_binding(first, pool, 4096), _binding(second, pool, 8192)),
        holder=_occurrence(0), at=_event(1, done=True))
    assert not grant.granted
    assert grant.blocked_pools == (pool,)
    assert domain.current == before
    assert domain.epoch == epoch
    assert not domain.grants and not domain.backings and not domain.peaks


def test_exact_capacity_boundary_and_release():
    state, pool = _fresh(8192)
    key = _allocation(0, symbol=1)
    assert Ledger.try_acquire(
        state, (_binding(key, pool, 8192),), holder=_occurrence(0),
        at=_event(1, done=True)).granted
    assert state.capacity_domains["memory"].current[pool] == 8192
    result = Ledger.release(state, key, after=_event(2, done=True))
    assert result.released
    assert state.capacity_domains["memory"].current[pool] == 0


def test_reader_lease_defers_release_until_real_last_use():
    state, pool = _fresh()
    key = _allocation(0, symbol=1)
    Ledger.try_acquire(state, (_binding(key, pool, 4096),),
                       holder=_occurrence(0), at=_event(1, done=True))
    reader = Ledger.retain_reader(state, key, reader=_event(3, done=True))
    deferred = Ledger.release(state, key, after=_event(4, done=True))
    assert deferred.deferred and not deferred.released
    assert state.capacity_domains["memory"].current[pool] == 4096
    result = Ledger.retire_reader(state, reader)
    assert result.released
    assert state.capacity_domains["memory"].current[pool] == 0


def test_stale_generation_cannot_read_or_release():
    state, pool = _fresh()
    key = _allocation(0, symbol=1, generation=2)
    Ledger.try_acquire(state, (_binding(key, pool, 1024),),
                       holder=_occurrence(0), at=_event(1, done=True))
    with pytest.raises(LedgerError, match="generation_mismatch"):
        Ledger.retain_reader(state, key, reader=_event(2, done=True),
                             expected_generation=1)
    with pytest.raises(LedgerError, match="generation_mismatch"):
        Ledger.release(state, key, after=_event(2, done=True),
                       expected_generation=1)
    assert state.capacity_domains["memory"].current[pool] == 1024


def test_external_reference_never_reacquires_bytes():
    state, pool = _fresh()
    key = _allocation(0, symbol=1)
    grant = Ledger.try_acquire(
        state, (_binding(key, pool, 4096, acquire=False),),
        holder=_occurrence(0), at=_event(1, done=True))
    assert grant.granted
    assert state.capacity_domains["memory"].current.get(pool, 0.0) == 0.0
    assert Ledger.release(state, key, after=_event(2, done=True)).released
    assert state.capacity_domains["memory"].current.get(pool, 0.0) == 0.0


def test_parent_reservation_slice_not_double_charged():
    state, pool = _fresh(8192)
    parent_key = _allocation(0, symbol=1)
    child_key = _allocation(1, symbol=2)
    backing = BackingKey(physical_domain="memory", rank=0, tier="hbm",
                         storage_id="workspace", generation=0)
    assert Ledger.try_acquire(
        state,
        (_binding(parent_key, pool, 8192, backing=backing,
                  ranges=((0, 8192),)),),
        holder=_occurrence(0), at=_event(1, done=True)).granted
    assert state.capacity_domains["memory"].current[pool] == 8192
    assert Ledger.try_acquire(
        state,
        (_binding(child_key, pool, 2048, backing=backing,
                  ranges=((4096, 6144),), parent=parent_key),),
        holder=_occurrence(1), at=_event(2, done=True)).granted
    assert state.capacity_domains["memory"].current[pool] == 8192
    child_release = Ledger.release(state, child_key, after=_event(3, done=True))
    assert child_release.released
    assert state.capacity_domains["memory"].current[pool] == 8192
    parent_release = Ledger.release(state, parent_key, after=_event(4, done=True))
    assert parent_release.released
    assert state.capacity_domains["memory"].current[pool] == 0


def test_parent_release_deferred_while_child_live():
    state, pool = _fresh(4096)
    parent_key = _allocation(0, symbol=1)
    child_key = _allocation(1, symbol=2)
    backing = BackingKey(physical_domain="memory", rank=0, tier="hbm",
                         storage_id="workspace", generation=0)
    Ledger.try_acquire(
        state,
        (_binding(parent_key, pool, 4096, backing=backing,
                  ranges=((0, 4096),)),),
        holder=_occurrence(0), at=_event(1, done=True))
    Ledger.try_acquire(
        state,
        (_binding(child_key, pool, 1024, backing=backing,
                  ranges=((0, 1024),), parent=parent_key),),
        holder=_occurrence(1), at=_event(2, done=True))
    deferred = Ledger.release(state, parent_key, after=_event(3, done=True))
    assert deferred.deferred and not deferred.released
    assert state.capacity_domains["memory"].grants[parent_key].released is False
    Ledger.release(state, child_key, after=_event(4, done=True))
    again = Ledger.release(state, parent_key, after=_event(5, done=True))
    assert again.released
    assert state.capacity_domains["memory"].current[pool] == 0


def test_pending_last_use_defers_until_settle():
    state, pool = _fresh()
    key = _allocation(0, symbol=1)
    Ledger.try_acquire(state, (_binding(key, pool, 1024),),
                       holder=_occurrence(0), at=_event(1, done=True))
    pending = _event(9, done=False)
    result = Ledger.release(state, key, after=pending)
    assert result.deferred and not result.released
    assert Ledger.settle(state) == ()
    pending.done = True
    pending.time = 9.0
    settled = Ledger.settle(state, (pending,))
    assert len(settled) == 1 and settled[0].released
    assert state.capacity_domains["memory"].current[pool] == 0


def test_peak_and_epoch_accounting():
    state, pool = _fresh(4096)
    keys = [_allocation(i, symbol=i) for i in range(2)]
    Ledger.try_acquire(state, (_binding(keys[0], pool, 3072),),
                       holder=_occurrence(0), at=_event(1, done=True))
    Ledger.try_acquire(state, (_binding(keys[1], pool, 1024),),
                       holder=_occurrence(1), at=_event(2, done=True))
    domain = state.capacity_domains["memory"]
    assert domain.current[pool] == 4096
    assert domain.peaks[pool] == 4096
    assert domain.epoch == 2
    rejected = Ledger.try_acquire(
        state, (_binding(_allocation(2, symbol=3), pool, 1),),
        holder=_occurrence(2), at=_event(3, done=True))
    assert not rejected.granted
    assert domain.epoch == 2
    Ledger.release(state, keys[1], after=_event(4, done=True))
    assert domain.current[pool] == 3072
    assert domain.peaks[pool] == 4096


def test_duplicate_and_malformed_batches_are_rejected_without_mutation():
    state, pool = _fresh()
    key = _allocation(0, symbol=1)
    with pytest.raises(LedgerError, match="allocation_duplicate"):
        Ledger.try_acquire(
            state, (_binding(key, pool, 1), _binding(key, pool, 1)),
            holder=_occurrence(0), at=_event(1, done=True))
    with pytest.raises(LedgerError, match="allocation_amount_invalid"):
        Ledger.try_acquire(
            state, (_binding(key, pool, float("nan")),),
            holder=_occurrence(0), at=_event(1, done=True))
    domain = state.capacity_domains["memory"]
    assert not domain.grants and not domain.current and domain.epoch == 0


def test_byte_set_oracle_over_overlapping_transactions():
    """Independent O3 oracle: union cardinality over a 16-byte universe."""
    state, pool = _fresh(16)
    spans = [(0, 4), (3, 9), (8, 12), (14, 16), (1, 2)]
    keys = []
    expected = set()
    backing = BackingKey(physical_domain="memory", rank=0, tier="hbm",
                         storage_id="shared-slot", generation=0)
    for index, span in enumerate(spans):
        key = _allocation(index, symbol=index)
        keys.append((key, span))
        assert Ledger.try_acquire(
            state, (_binding(key, pool, span[1] - span[0], backing=backing,
                             ranges=(span,)),),
            holder=_occurrence(index), at=_event(index, done=True)).granted
        expected |= _byte_set((span,))
        assert state.capacity_domains["memory"].current[pool] == len(expected)
        assert state.capacity_domains["memory"].current[pool] <= 16
    remaining = list(spans)
    for key, span in keys:
        Ledger.release(state, key, after=_event(20 + span[0], done=True))
        remaining.remove(span)
        expected = set().union(*(_byte_set((item,)) for item in remaining)) \
            if remaining else set()
        assert state.capacity_domains["memory"].current[pool] == len(expected)


def test_distinct_domains_are_independent():
    state = LedgerState()
    pool_a, cap_a = _pool(domain="pd-a", rank=0, capacity=1024)
    pool_b, cap_b = _pool(domain="pd-b", rank=0, capacity=2048)
    Ledger.ensure_domain(state, "pd-a", capacities={pool_a: float(cap_a)})
    Ledger.ensure_domain(state, "pd-b", capacities={pool_b: float(cap_b)})
    key_a = _allocation(0, symbol=1)
    key_b = _allocation(0, symbol=1, launch=2)
    assert Ledger.try_acquire(
        state, (_binding(key_a, pool_a, 1000),), holder=_occurrence(0),
        at=_event(1, done=True)).granted
    assert Ledger.try_acquire(
        state, (_binding(key_b, pool_b, 2000),), holder=_occurrence(1),
        at=_event(2, done=True)).granted
    assert state.capacity_domains["pd-a"].current[pool_a] == 1000
    assert state.capacity_domains["pd-b"].current[pool_b] == 2000
    assert set(Ledger.usage(state)) == {"pd-a", "pd-b"}


def test_domain_cannot_be_bound_to_two_runtimes():
    state = LedgerState()
    pool, _cap = _pool()
    domain = Ledger.ensure_domain(state, "memory", capacities={pool: 1024})
    other = LedgerState(uid="runtime-other")
    with pytest.raises(LedgerError, match="memory_runtime_conflict"):
        Ledger.adopt(other, domain)


# --------------------------------------------------------------- adoption
def test_memory_adopts_its_capacity_domain_into_real_runtime_state():
    """Memory's detached domain is claimed by the one real RuntimeState."""
    from test_prediction_system import case as prediction_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.runtime.state import RuntimeState
    from mindie_sim.prediction.structure import parse_case

    memory = Memory(parse_case(prediction_case()))
    state = RuntimeState(resources={})
    domain = memory.bind_runtime(state)
    assert domain is state.capacity_domains["memory"]
    assert domain.owner_runtime_uid == state.uid

    pool = memory.ledger_pool(0, 0, 0)
    assert pool in domain.capacities
    binding = memory.typed_binding(
        0, "typed", stage=0, rank=0, amount=1, scope_ordinal=1,
        backing_key="typed-backing", ranges=((0, 1),))
    assert memory.grant_typed(
        (binding,), holder=_occurrence(0), at=_event(1, done=True)).granted
    assert domain.current[pool] == 1
    reader = memory.retain_reader_typed(
        binding.key, reader=_event(2, done=True))
    deferred = memory.release_typed(binding.key, after=_event(3, done=True))
    assert deferred.deferred and domain.current[pool] == 1
    assert memory.runtime_state is state
    from mindie_sim.prediction.runtime.ledger import Ledger
    assert Ledger.retire_reader(state, reader).released
    assert domain.current[pool] == 0


def test_memory_domain_cannot_bind_two_runtimes():
    from test_prediction_system import case as prediction_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.runtime.state import RuntimeState
    from mindie_sim.prediction.structure import parse_case

    memory = Memory(parse_case(prediction_case()))
    memory.bind_runtime(RuntimeState(resources={}))
    with pytest.raises(ValueError, match="memory_runtime_conflict"):
        memory.bind_runtime(RuntimeState(resources={}))


def test_cache_state_adopts_one_runtime_cache_domain():
    from mindie_sim.prediction.cache_path import CacheGeometry, CacheState
    from mindie_sim.prediction.runtime.state import RuntimeState

    cache = CacheState(CacheGeometry(capacity_bytes=4096))
    state = RuntimeState(resources={})
    key = cache.bind_runtime(state)
    assert state.cache_domains[key] is cache
    assert cache.runtime_uid == state.uid
    other = RuntimeState(resources={})
    with pytest.raises(Exception, match="cache_runtime_conflict"):
        cache.bind_runtime(other)


def test_legacy_public_grant_batch_records_canonical_ledger_state():
    from test_prediction_system import case as prediction_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.runtime.state import RuntimeState
    from mindie_sim.prediction.structure import parse_case

    memory = Memory(parse_case(prediction_case()))
    state = RuntimeState(resources={})
    domain = memory.bind_runtime(state)
    assert memory.grant_batch(0, [{
        "name": "kv-a", "bytes": 256, "rank": 0, "stage": 0,
        "generation": 1, "physical_key": "kv-a"}])
    pool = memory.ledger_pool(0, 0, 0)
    assert domain.current[pool] == 256
    assert memory.release_grant(0, "kv-a")
    assert domain.current[pool] == 0


def test_new_engine_admits_through_adopted_memory_ledger():
    from test_prediction_system import case as prediction_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.runtime.engine import Allocation, Engine, Resource
    from mindie_sim.prediction.structure import parse_case

    memory = Memory(parse_case(prediction_case()))
    engine = Engine([Resource(name="hbm", rate=1.0)],
                    pools={"hbm": 1 << 30})
    memory.bind_runtime(engine)
    assert engine.bind_memory(memory)
    task = engine.submit(
        "work", work={"hbm": 1.0},
        acquire=(Allocation("kv", "hbm", 256.0),),
        memory=memory, memory_pools=("hbm",))
    engine.advance(until=(task,))
    assert task.done
    pool = memory.ledger_pool(0, 0, 0)
    assert memory.ledger_domain.peaks[pool] >= 256.0


def test_legacy_async_grant_reader_keeps_canonical_bytes_until_last_use():
    from test_prediction_system import case as prediction_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.runtime.state import RuntimeState
    from mindie_sim.prediction.structure import parse_case

    memory = Memory(parse_case(prediction_case()))
    state = RuntimeState(resources={})
    memory.bind_runtime(state)
    assert memory.grant_batch(0, [{
        "name": "kv", "bytes": 256, "rank": 0, "stage": 0,
        "generation": 1, "physical_key": "kv"}])
    last_use = _event(7, done=True)
    assert memory.hold_for_async_consumer(
        0, "kv", "reader", grant="kv", generation=1, event=last_use)
    pool = memory.ledger_pool(0, 0, 0)
    assert memory.release_grant(0, "kv")
    assert memory.ledger_domain.current[pool] == 256
    assert memory.release_async_consumer(
        0, "kv", "reader", generation=1, event=last_use)
    assert memory.ledger_domain.current[pool] == 0


def test_random_multi_backing_union_matches_byte_set_oracle():
    """O3 over overlapping grants on three backings with releases."""
    import random

    for seed in range(25):
        rng = random.Random(seed)
        state, pool = _fresh(32)
        backings = [
            BackingKey(physical_domain="memory", rank=0, tier="hbm",
                       storage_id=f"b{index}", generation=0)
            for index in range(3)
        ]
        live: dict[int, tuple[BackingKey, tuple[int, int]]] = {}
        spans_by_backing: dict[BackingKey, list[tuple[int, int]]] = {
            key: [] for key in backings}

        def union_oracle():
            return {
                key: set().union(*(
                    set(range(*span)) for span in spans_by_backing[key]))
                if spans_by_backing[key] else set()
                for key in backings
            }

        oracle = union_oracle()
        for index in range(8):
            backing = rng.choice(backings)
            start = rng.randrange(0, 16)
            end = start + rng.randrange(1, 9)
            span = (start, end)
            key = _allocation(index, symbol=index)
            before = {item: set(value) for item, value in oracle.items()}
            before_spans = {item: list(value) for item, value in spans_by_backing.items()}
            before_usage = sum(len(value) for value in oracle.values())
            attempt = _binding(
                key, pool, end - start, backing=backing, ranges=(span,))
            result = Ledger.try_acquire(
                state, (attempt,), holder=_occurrence(index),
                at=_event(index, done=True))
            candidate_spans = {
                item: list(value) for item, value in spans_by_backing.items()}
            candidate_spans[backing].append(span)
            candidate_usage = sum(
                len(set().union(*(set(range(*part)) for part in parts)))
                if parts else 0
                for parts in candidate_spans.values())
            if result.granted:
                spans_by_backing = candidate_spans
                oracle = union_oracle()
                live[index] = (backing, span)
            else:
                assert candidate_usage > 32
                spans_by_backing = {item: list(value) for item, value in before_spans.items()}
                oracle = before
                assert state.capacity_domains["memory"].current[pool] == \
                    before_usage
            assert state.capacity_domains["memory"].current[pool] == \
                sum(len(value) for value in oracle.values())
        order = list(live)
        rng.shuffle(order)
        for index in order:
            backing, span = live.pop(index)
            Ledger.release(state, _allocation(index, symbol=index),
                           after=_event(100 + index, done=True))
            spans_by_backing[backing].remove(span)
            oracle = union_oracle()
            assert state.capacity_domains["memory"].current[pool] == \
                sum(len(value) for value in oracle.values())
