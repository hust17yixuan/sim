"""Focused tests for the private exact-period Engine helper.

These tests establish the detector's structural contract directly and also
exercise it against the real ``Engine`` on a finite 32-owner, two-slot,
phase/tail, persistent-accumulator pipeline.  The Engine-side adapter used by
the second group is test-local; T04 owns the production adapter and registry.
"""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass, field
import math

import pytest

from mindie_sim.prediction.engine import (
    Allocation,
    Engine,
    ExecutionError,
    Resource,
)
from mindie_sim.prediction.engine_periodic import (
    CanonicalState,
    CycleDetector,
    IntervalDelta,
    PeriodicContractError,
    SettledFrontier,
    StateInterner,
    TemplateRef,
    Cursor,
    metrics,
    reset_state_caches,
    structural_difference,
    structural_equal,
)


STATE = (
    ("schema", "periodic-test-v1"),
    (
        "resources",
        (("mte2", 100.0, "work", "fair", 1), ("cube", 100.0, "work", "fair", 1)),
    ),
    ("pools", (("l1", 128.0, 32), ("l0c", 256.0, 32))),
    (
        "tasks",
        (
            ("run", 1, "tile@0", "load", (("mte2", 8.0),), 0, ()),
            ("run", 1, "tile@0", "compute", (("cube", 16.0),), 1, ("tile@0:load",)),
            ("run", 1, "tile@1", "load", (("mte2", 8.0),), 0, ("tile@0:load",)),
        ),
    ),
    (
        "live_allocations",
        (
            ("run", 1, "tile@0", "panel", "l1", 4.0, (0,)),
            ("run", 1, None, "accumulator0", "l0c", 8.0, (0,)),
        ),
    ),
    ("pool_occupied", (("l1", 4.0), ("l0c", 8.0))),
    ("unit_occupied", ((("l1", 0), 4.0), (("l0c", 0), 8.0))),
    ("heaps", (("ready", ((0, 1),)), ("release", ()), ("completable", ()))),
    ("promises", (("complete",),)),
)


def cursor(
    run_id=1,
    *,
    phase_ordinal=0,
    phase_end=67,
    next_issue=3,
    retired=1,
    slots=(0, 1),
):
    return Cursor(run_id, phase_ordinal, phase_end, next_issue, retired, slots)


def frame(now, *, state=STATE, cursors=None, sequence=100, epoch=7):
    if cursors is None:
        cursors = (cursor(next_issue=2, retired=0),)
    return SettledFrontier(
        now=now,
        epoch=epoch,
        normalized_state=state,
        cursors=tuple(cursors),
        sequence=sequence,
    )


def delta(
    dt=1.0,
    *,
    issued=None,
    sequence_count=4,
    service=None,
    busy=None,
    allocation_event_count=2,
    peaks=None,
    logical_completions=1,
    crossed=False,
    allocation_grants=0,
    allocation_last_uses=0,
):
    if issued is None:
        issued = {1: (1, 1)}
    if service is None:
        service = {"mte2": 8.0, "cube": 16.0}
    if busy is None:
        busy = {"mte2": 1.0, "cube": 1.0}
    if peaks is None:
        peaks = {(("l1", 0)): 4.0, (("l0c", 0)): 8.0}
    return IntervalDelta(
        dt=dt,
        issued_retired_by_run=issued,
        sequence_count=sequence_count,
        service_by_resource=service,
        busy_by_resource=busy,
        allocation_event_count=allocation_event_count,
        per_unit_peaks=peaks,
        logical_completions=logical_completions,
        crossed_external_or_opaque=crossed,
        allocation_grants=allocation_grants,
        allocation_last_uses=allocation_last_uses,
    )


def feed(detector, observations, *, stop_at=math.inf):
    """Feed ``(frame, interval)`` pairs and return each observe result."""
    results = []
    for item, interval in observations:
        results.append(detector.observe(item, interval, stop_at=stop_at))
    return results


def advancing_frames(count, *, start=0.0, period=1.0, first_next=2, phase_end=67, seq_stride=4, run_id=1):
    """A clean one-run steady cycle: one issue and one retire per period."""
    result = []
    for index in range(count):
        result.append(
            (
                frame(
                    start + index * period,
                    cursors=(
                        cursor(
                            run_id,
                            phase_end=phase_end,
                            next_issue=first_next + index,
                            retired=first_next - 2 + index,
                        ),
                    ),
                    sequence=100 + index * seq_stride,
                ),
                None if index == 0 else delta(period, issued={run_id: (1, 1)}, sequence_count=seq_stride),
            )
        )
    return result


def test_two_equal_intervals_return_a_bounded_exact_skip():
    detector = CycleDetector()
    observations = advancing_frames(5)
    results = feed(detector, observations)

    assert results[0] is None
    assert results[1] is None  # one matching interval
    skip = results[2]  # two measured equal intervals
    assert skip is not None
    assert skip.cycles == 63  # 67 - next_issue(4)
    assert skip.period == 1.0
    assert dict(skip.tile_strides_by_run) == {1: 1}
    assert skip.seq_stride == 4
    assert skip.delta.dt == 1.0
    assert skip.delta.logical_completions == 1

    run = skip.correspondence.run(1)
    assert (run.next_issue_before, run.next_issue_after) == (4, 67)
    assert (run.retired_before, run.retired_after) == (2, 65)
    assert run.tile_offset == 63
    assert run.finished_offset == 63
    assert [(tile.old_tile, tile.new_tile) for tile in skip.correspondence.window] == [
        (2, 65),
        (3, 66),
    ]
    assert skip.correspondence.event_sequence_offset == 252

    total = skip.total_delta()
    assert total.dt == 63.0
    assert dict(total.issued_retired_by_run) == {1: (63, 63)}
    assert total.sequence_count == 252
    assert total.service_by_resource["mte2"] == 504.0
    assert total.busy_by_resource["cube"] == 63.0
    assert total.allocation_event_count == 126
    assert total.per_unit_peaks == skip.delta.per_unit_peaks


def test_structural_identity_is_deep_and_never_uses_a_digest():
    # Same top-level tuple shape as STATE, different deep remaining work.
    tasks = list(STATE[3][1])
    tasks[0] = ("run", 1, "tile@0", "load", (("mte2", 8.0000001),), 0, ())
    changed = STATE[:3] + (("tasks", tuple(tasks)),) + STATE[4:]
    assert not structural_equal(STATE, changed)
    assert structural_equal(STATE, tuple(STATE))
    assert structural_difference(STATE, changed) == ("state[3][1][0][4][0][1]:8.0!=8.0000001")

    detector = CycleDetector()
    # f0 baseline; f1 differs deeply and therefore cannot match.
    f0 = frame(0.0)
    f1 = frame(1.0, state=changed, cursors=(cursor(1, next_issue=3, retired=1),), sequence=104)
    first = detector.observe(f0, None, stop_at=math.inf)
    second = detector.observe(f1, delta(), stop_at=math.inf)
    assert first is None and second is None
    assert detector.last_rejection in {"baseline", "no_repeated_span"}

    # Once the changed shape is stable, it becomes its own exact cycle.
    f2 = frame(2.0, state=changed, cursors=(cursor(1, next_issue=4, retired=2),), sequence=108)
    f3 = frame(3.0, state=changed, cursors=(cursor(1, next_issue=5, retired=3),), sequence=112)
    assert detector.observe(f2, delta(), stop_at=math.inf) is None
    skip = detector.observe(f3, delta(), stop_at=math.inf)
    assert skip is not None and skip.cycles == 62


def test_order_is_structural_but_mapping_order_is_not():
    reordered = STATE[:5] + (tuple(reversed(STATE[5])),) + STATE[6:]
    assert not structural_equal(STATE, reordered)

    mapping_a = (("m", {"a": 1.0, "b": 2.0}),)
    mapping_b = (("m", {"b": 2.0, "a": 1.0}),)
    assert structural_equal(mapping_a, mapping_b)


def test_cursor_progress_is_not_the_key_but_phase_class_is():
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(
        0.0,
        cursors=(cursor(1, phase_ordinal=2, phase_end=67, next_issue=3, retired=1),),
    )
    assert detector.observe(f0, None, stop_at=math.inf) is None
    # next_issue/retired moved; phase_ordinal/end/slots did not.
    f1 = frame(
        1.0,
        cursors=(cursor(1, phase_ordinal=2, phase_end=67, next_issue=4, retired=2),),
        sequence=104,
    )
    skip = detector.observe(f1, delta(), stop_at=math.inf)
    assert skip is not None

    detector = CycleDetector(evidence_periods=1)
    detector.observe(f0, None, stop_at=math.inf)
    moved_phase = frame(
        1.0,
        cursors=(cursor(1, phase_ordinal=3, phase_end=68, next_issue=4, retired=2),),
        sequence=104,
    )
    assert detector.observe(moved_phase, delta(), stop_at=math.inf) is None
    assert detector.last_rejection == "phase_changed"


def test_phase_remaining_only_bounds_cycles():
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, cursors=(cursor(1, phase_end=8, next_issue=2, retired=0),))
    detector.observe(f0, None, stop_at=math.inf)
    f1 = frame(1.0, cursors=(cursor(1, phase_end=8, next_issue=3, retired=1),), sequence=104)
    skip = detector.observe(f1, delta(), stop_at=math.inf)
    assert skip is not None and skip.cycles == 5  # 8 - 3

    # At the phase boundary no new tile can be issued.
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, cursors=(cursor(1, phase_end=8, next_issue=7, retired=5),))
    detector.observe(f0, None, stop_at=math.inf)
    f1 = frame(1.0, cursors=(cursor(1, phase_end=8, next_issue=8, retired=6),), sequence=104)
    assert detector.observe(f1, delta(), stop_at=math.inf) is None
    assert detector.last_rejection == "boundary"


def test_stop_at_is_an_actual_boundary_not_a_target():
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, cursors=(cursor(1, phase_end=100, next_issue=2, retired=0),))
    detector.observe(f0, None, stop_at=math.inf)
    f1 = frame(1.0, cursors=(cursor(1, phase_end=100, next_issue=3, retired=1),), sequence=104)
    skip = detector.observe(f1, delta(), stop_at=4.5)
    assert skip is not None and skip.cycles == 3

    # An exact boundary is not skipped: leave the tie to detailed advance.
    detector = CycleDetector(evidence_periods=1)
    detector.observe(f0, None, stop_at=math.inf)
    skip = detector.observe(f1, delta(), stop_at=5.0)
    assert skip is not None and skip.cycles == 3

    detector = CycleDetector(evidence_periods=1)
    detector.observe(f0, None, stop_at=math.inf)
    assert detector.observe(f1, delta(), stop_at=0.5) is None


def test_external_or_opaque_interval_never_becomes_evidence():
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0)
    detector.observe(f0, None, stop_at=math.inf)
    f1 = frame(1.0, sequence=104)
    assert detector.observe(f1, delta(crossed=True), stop_at=math.inf) is None
    assert detector.last_rejection == "external_or_opaque"


def test_interval_mismatch_resets_the_candidate_period():
    detector = CycleDetector(evidence_periods=2)
    normal = delta()
    other = delta(service={"mte2": 9.0, "cube": 16.0})
    frames = advancing_frames(5)
    # Replace the third measured interval with a different real delta.
    observations = [
        frames[0],
        (frames[1][0], normal),
        (frames[2][0], other),
        (frames[3][0], other),
    ]
    assert detector.observe(observations[0][0], None, stop_at=math.inf) is None
    assert detector.observe(observations[1][0], normal, stop_at=math.inf) is None
    assert detector.observe(observations[2][0], other, stop_at=math.inf) is None
    assert detector.last_rejection == "no_repeated_span"
    skip = detector.observe(observations[3][0], other, stop_at=math.inf)
    assert skip is not None
    assert dict(skip.delta.service_by_resource)["mte2"] == 9.0


def test_epoch_change_forces_a_new_baseline():
    detector = CycleDetector(evidence_periods=1)
    detector.observe(frame(0.0, epoch=7), None, stop_at=math.inf)
    assert detector.observe(frame(1.0, epoch=8, sequence=104), delta(), stop_at=math.inf) is None
    assert detector.last_rejection == "epoch_changed"


def test_joint_cohort_is_not_per_owner_exclusive():
    detector = CycleDetector(evidence_periods=1)
    runs = []
    for run_id in range(1, 33):
        phase_end = 10 if run_id == 1 else 68
        runs.append(cursor(run_id, phase_ordinal=1, phase_end=phase_end, next_issue=4, retired=2))
    # A shared unit occupancy from all owners is part of the state.
    shared = STATE + (("shared_units", ((("l1", 0), 64.0),)),)
    f0 = frame(0.0, state=shared, cursors=runs, sequence=1000)
    detector.observe(f0, None, stop_at=math.inf)
    advanced = tuple(
        cursor(
            item.run_id,
            phase_ordinal=item.phase_ordinal,
            phase_end=item.phase_end_exclusive,
            next_issue=item.next_issue + 1,
            retired=item.retired + 1,
        )
        for item in runs
    )
    f1 = frame(1.0, state=shared, cursors=advanced, sequence=1000 + 32 * 4)
    skip = detector.observe(
        f1,
        delta(
            issued={item.run_id: (1, 1) for item in runs},
            sequence_count=32 * 4,
            logical_completions=32,
        ),
        stop_at=math.inf,
    )
    assert skip is not None
    # Owner 1 is the limiting phase, not a per-owner independent skip.
    assert skip.cycles == 5  # phase_end 10 - next_issue 5
    assert set(skip.tile_strides_by_run) == set(range(1, 33))
    assert {run.run_id: run.tile_offset for run in skip.correspondence.runs}[1] == 5
    assert {run.run_id: run.tile_offset for run in skip.correspondence.runs}[2] == 5


def test_slot_count_and_cross_phase_cursor_fail_closed():
    # The production adapter may report the declared slot count rather than a
    # tuple of live-window offsets.  It is still part of the cursor class.
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, cursors=(cursor(1, phase_end=4, next_issue=5, retired=3, slots=2),))
    assert detector.observe(f0, None, stop_at=math.inf) is None
    f1 = frame(
        1.0,
        cursors=(cursor(1, phase_end=4, next_issue=6, retired=4, slots=2),),
        sequence=104,
    )
    assert detector.observe(f1, delta(), stop_at=math.inf) is None
    assert detector.last_rejection == "boundary"


def test_future_release_offset_change_breaks_identity():
    detector = CycleDetector(evidence_periods=1)
    with_release = STATE + (("release_offset", 2.0),)
    later_release = STATE + (("release_offset", 1.0),)
    f0 = frame(0.0, state=with_release, cursors=(cursor(next_issue=2, retired=0),))
    f1 = frame(
        1.0,
        state=later_release,
        cursors=(cursor(next_issue=3, retired=1),),
        sequence=104,
    )
    assert detector.observe(f0, None, stop_at=3.0) is None
    assert detector.observe(f1, delta(), stop_at=3.0) is None
    assert detector.last_rejection in {"baseline", "no_repeated_span"}

    # Once the offset is stable, the same state is periodic again.
    f2 = frame(
        2.0,
        state=later_release,
        cursors=(cursor(next_issue=4, retired=2),),
        sequence=108,
    )
    skip = detector.observe(f2, delta(), stop_at=math.inf)
    assert skip is not None


def test_per_run_strides_are_reported_and_bound_the_joint_skip():
    detector = CycleDetector(evidence_periods=1)
    runs0 = (
        cursor(1, phase_end=10, next_issue=3, retired=1),
        cursor(2, phase_end=12, next_issue=4, retired=0),
    )
    runs1 = (
        cursor(1, phase_end=10, next_issue=4, retired=2),
        cursor(2, phase_end=12, next_issue=6, retired=2),
    )
    f0 = frame(0.0, cursors=runs0, sequence=100)
    f1 = frame(
        2.0,
        cursors=runs1,
        sequence=106,
    )
    interval = delta(
        dt=2.0,
        issued={1: (1, 1), 2: (2, 2)},
        sequence_count=6,
    )
    detector.observe(f0, None, stop_at=math.inf)
    skip = detector.observe(f1, interval, stop_at=math.inf)
    assert skip is not None
    # run 2 allows (12-6)//2=3; run 1 allows (10-4)//1=6; time is 6.
    assert dict(skip.tile_strides_by_run) == {1: 1, 2: 2}
    assert skip.cycles == 3
    assert skip.correspondence.run(1).tile_offset == 3
    assert skip.correspondence.run(2).tile_offset == 6
    assert skip.correspondence.event_sequence_offset == 18


def test_same_hashable_shape_with_different_values_is_not_isomorphic():
    detector = CycleDetector(evidence_periods=1)
    detector.observe(frame(0.0), None, stop_at=math.inf)
    changed = STATE[:-1] + (("promises", (("other",),)),)
    assert detector.observe(frame(1.0, state=changed, sequence=104), delta(), stop_at=math.inf) is None


def test_unmeasured_or_opaque_state_is_rejected_not_approximated():
    with pytest.raises(PeriodicContractError):
        SettledFrontier(now=0.0, epoch=1, normalized_state=(object(),), cursors=(), sequence=0)
    with pytest.raises(PeriodicContractError):
        SettledFrontier(now=float("nan"), epoch=1, normalized_state=(), cursors=(), sequence=0)
    with pytest.raises(PeriodicContractError):
        IntervalDelta(
            dt=-1.0,
            issued_retired_by_run={1: (1, 1)},
            sequence_count=1,
            service_by_resource={},
            busy_by_resource={},
            allocation_event_count=0,
            per_unit_peaks={},
            logical_completions=0,
            crossed_external_or_opaque=False,
        )
    # A transition observation may have next_issue past the current phase
    # end; the bound is then zero and the movement stays detailed.
    assert Cursor(1, 0, 4, 5, 2, 2).slots == 2
    with pytest.raises(PeriodicContractError):
        Cursor(1, 0, 4, 1, 2, (0, 1))  # retired > next_issue


def engine_state(*, remaining=8.0, caps=(("cube", 1.0),), tasks=1):
    """The current T04 5-tuple shape: tasks, runs, allocations, pools, releases."""
    task_rows = tuple(
        (("periodic", 1000, 0, index, "stage"), (("cube", remaining),), 0, 0.0, 0.0, 0, caps, (), False, "owner")
        for index in range(tasks)
    )
    runs = ((1000, 0, 67, (0, 1), (("l0c", 8.0, (0,)),)),)
    return (
        task_rows,
        runs,
        (("l1", 4.0, (0,), 1),),
        (("l1", 4.0, ((0, 4.0),)),),
        (),
    )


def test_engine_shape_interns_static_template_and_validates_only_dynamic():
    reset_state_caches()
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, state=engine_state(), cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=engine_state(), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    assert f0.state_form == "engine"
    assert f0.canonical_state is not None
    assert detector.observe(f0, None, stop_at=math.inf) is None
    skip = detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf)
    assert skip is not None and skip.cycles == 64
    stats = detector.stats()
    assert stats["interner"]["misses"] == 1
    assert stats["interner"]["hits"] >= 1
    assert stats["state_metrics"]["template_validations"] == 1
    assert stats["state_metrics"]["dynamic_validations"] >= 2
    assert stats["fast_state_compares"] >= 1
    assert stats["skips"] == 1
    assert stats["skipped_cycles"] == 64
    assert stats["rejections"].get("skip") == 1
    assert f0.canonical_state.template is f1.canonical_state.template


def test_engine_static_or_dynamic_change_is_not_a_cycle():
    reset_state_caches()
    detector = CycleDetector(evidence_periods=1)
    base = frame(0.0, state=engine_state(), cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    assert detector.observe(base, None, stop_at=math.inf) is None

    changed_caps = frame(
        1.0, state=engine_state(caps=(("cube", 2.0),)), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104
    )
    assert detector.observe(changed_caps, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_difference.startswith("canonical.template")

    reset_state_caches()
    detector = CycleDetector(evidence_periods=1)
    detector.observe(base, None, stop_at=math.inf)
    changed_remaining = frame(
        1.0, state=engine_state(remaining=7.5), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104
    )
    assert detector.observe(changed_remaining, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_difference.startswith("canonical.dynamic")


def test_fingerprint_collision_is_rejected_by_exact_template():
    interner = StateInterner(limit=4)
    left = interner.intern((("left",),), generation=7, fingerprint=123)
    right = interner.intern((("right",),), generation=7, fingerprint=123)
    assert left.template_id != right.template_id
    assert left.fingerprint == right.fingerprint
    left_state = CanonicalState(left, (("d", 1.0),))
    right_state = CanonicalState(right, (("d", 1.0),))
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, state=left_state, cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=right_state, cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    detector.observe(f0, None, stop_at=math.inf)
    assert detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_difference.startswith("canonical.template")


def test_template_ids_are_local_to_an_interner_not_global_identity():
    left_interner = StateInterner(limit=2)
    right_interner = StateInterner(limit=2)
    left = left_interner.intern((("left",),), generation=7, fingerprint=5)
    right = right_interner.intern((("right",),), generation=7, fingerprint=5)
    assert left.template_id == right.template_id == 1
    left_state = CanonicalState(left, (("d", 1.0),))
    right_state = CanonicalState(right, (("d", 1.0),))
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, state=left_state, cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=right_state, cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    detector.observe(f0, None, stop_at=math.inf)
    assert detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None


def test_mutable_external_state_keeps_the_full_validation_path():
    reset_state_caches()
    f0 = frame(
        0.0, state=engine_state(caps=[("cube", 1.0)]), cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100
    )
    assert f0.state_form == "raw"
    assert metrics()["raw_nodes"] > 0
    f1 = frame(
        1.0, state=engine_state(caps=[("cube", 2.0)]), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104
    )
    assert f1.state_form == "raw"
    detector = CycleDetector(evidence_periods=1)
    detector.observe(f0, None, stop_at=math.inf)
    # A mutable object is revalidated; a changed value is not skipped.
    assert detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_rejection == "no_repeated_span"


def test_placement_change_is_part_of_the_static_template():
    reset_state_caches()

    def placed(placement):
        state = engine_state(tasks=2)
        tasks = list(state[0])
        first = list(tasks[0])
        first[7] = placement
        tasks[0] = tuple(first)
        return (tuple(tasks), *state[1:])

    f0 = frame(0.0, state=placed(((0,),)), cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=placed(((1,),)), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    detector = CycleDetector(evidence_periods=1)
    detector.observe(f0, None, stop_at=math.inf)
    assert detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_difference.startswith("canonical.template")


def test_interner_is_bounded_and_evicts_old_generations():
    interner = StateInterner(limit=2)
    for index in range(4):
        interner.intern((("t", index),), generation=1)
    assert interner.templates == 2
    assert interner.evictions >= 2
    interner.intern((("t", 99),), generation=2)
    assert all(ref.generation == 2 for ref in interner._refs.values())
    assert interner.templates == 1


def test_static_validation_scales_with_unique_templates_not_frames():
    reset_state_caches()
    for index in range(20):
        frame(
            index,
            state=engine_state(tasks=100),
            cursors=(cursor(1000, next_issue=2 + index, retired=index),),
            sequence=100 + index * 4,
        )
    state_metrics = metrics()
    assert state_metrics["template_validations"] == 1
    assert state_metrics["dynamic_validations"] == 20
    assert state_metrics["template_nodes"] > 0
    assert state_metrics["dynamic_nodes"] > 0

    reset_state_caches()
    raw = (engine_state(tasks=100), "not-the-engine-form")
    for index in range(20):
        frame(index, state=raw, cursors=(cursor(1000, next_issue=2 + index, retired=index),), sequence=100 + index * 4)
    assert metrics()["raw_frames"] == 20
    assert metrics()["template_validations"] == 0


def test_engine_static_hits_reuse_the_fixed_tree_and_charge_probe_cost():
    reset_state_caches()
    frames = [
        frame(
            float(index),
            state=engine_state(tasks=3),
            cursors=(cursor(1000, next_issue=2 + index, retired=index),),
            sequence=100 + index * 4,
        )
        for index in range(4)
    ]
    assert all(item.state_form == "engine" for item in frames)
    assert frames[0].canonical_state.template is frames[1].canonical_state.template
    state_metrics = metrics()
    assert state_metrics["template_builds"] == 1
    assert state_metrics["template_validations"] == 1
    assert state_metrics["template_hits"] == 3
    assert state_metrics["template_hit_nodes"] > 0
    assert state_metrics["template_probes"] >= 3
    # A hit must not rebuild the canonical tree.
    assert state_metrics["template_nodes"] == state_metrics["template_build_nodes"]


def test_engine_waiting_int_subclass_is_rejected():
    class _Int(int):
        pass

    state = engine_state()
    rows = list(state[0])
    task = list(rows[0])
    task[2] = _Int(0)  # waiting was declared as a builtin int
    rows[0] = tuple(task)
    changed = (tuple(rows),) + state[1:]
    with pytest.raises(PeriodicContractError):
        frame(
            0.0,
            state=changed,
            cursors=(cursor(1000, next_issue=2, retired=0),),
            sequence=100,
        )


def test_grants_and_last_use_are_aggregated_and_scaled():
    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, state=engine_state(), cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=engine_state(), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    detector.observe(f0, None, stop_at=math.inf)
    skip = detector.observe(
        f1,
        delta(issued={1000: (1, 1)}, sequence_count=4, allocation_grants=2, allocation_last_uses=1),
        stop_at=math.inf,
    )
    assert skip is not None
    total = skip.total_delta()
    assert total.allocation_grants == 2 * skip.cycles
    assert total.allocation_last_uses == skip.cycles
    assert skip.delta.allocation_grants == 2


def test_shared_mutable_remaining_is_snapshotted_per_frontier():
    # Astra r2 closure: the first three changing values must not be skipped;
    # after 6.0 is stable for two real periods, the fifth frontier yields 61.
    reset_state_caches()
    interner = StateInterner(limit=64)
    template = interner.intern((("fixed",),), generation=7)
    remaining = [("cube", 8.0)]
    detector = CycleDetector()  # default two-period evidence
    frames = []
    results = []
    for index, value in enumerate((8.0, 7.0, 6.0, 6.0, 6.0)):
        remaining[0] = ("cube", value)
        state = CanonicalState(template, (("remaining", remaining),), version=index, origin="external")
        current = frame(
            float(index),
            state=state,
            cursors=(cursor(1000, phase_end=67, next_issue=2 + index, retired=index),),
            sequence=100 + 4 * index,
        )
        interval = None if index == 0 else delta(issued={1000: (1, 1)}, sequence_count=4)
        frames.append(current)
        results.append(detector.observe(current, interval, stop_at=math.inf))

    # All three changing values were observed; no historical snapshot was
    # rewritten by the shared list.
    assert not structural_equal(frames[0].canonical_state.dynamic, frames[2].canonical_state.dynamic)
    assert b"8.0" in repr(frames[0].canonical_state.dynamic).encode()
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert results[3] is None
    skip = results[4]
    assert skip is not None
    assert skip.cycles == 61
    assert skip.period == 1.0
    run = skip.correspondence.run(1000)
    assert (run.next_issue_before, run.next_issue_after) == (6, 67)
    assert (run.retired_before, run.retired_after) == (4, 65)


def test_typed_values_never_merge_through_python_equality():
    reset_state_caches()
    interner = StateInterner(limit=8)
    boolean = interner.intern((("flag", True),), generation=7)
    integer = interner.intern((("flag", 1),), generation=7)
    real = interner.intern((("flag", 1.0),), generation=7)
    assert len({boolean.template_id, integer.template_id, real.template_id}) == 3
    assert not structural_equal((("flag", True),), (("flag", 1),))
    assert not structural_equal((("flag", 1),), (("flag", 1.0),))
    assert structural_equal((("flag", True),), (("flag", True),))

    left = CanonicalState(boolean, (("flag", True),))
    middle = CanonicalState(integer, (("flag", 1),))
    right = CanonicalState(real, (("flag", 1.0),))
    assert not structural_equal(left.dynamic, middle.dynamic)
    assert not structural_equal(middle.dynamic, right.dynamic)

    detector = CycleDetector(evidence_periods=1)
    f0 = frame(0.0, state=left, cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=middle, cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    detector.observe(f0, None, stop_at=math.inf)
    assert detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_difference.startswith("canonical.template")


def test_opaque_new_input_is_rejected_at_both_entry_points():
    interner = StateInterner(limit=4)
    template = interner.intern((("fixed",),), generation=7)
    with pytest.raises(PeriodicContractError):
        CanonicalState(template, (object(),))
    with pytest.raises(PeriodicContractError):
        SettledFrontier(
            now=0.0,
            epoch=7,
            normalized_state=(object(),),
            cursors=(cursor(1000, next_issue=2, retired=0),),
            sequence=100,
        )


def test_dynamic_wait_order_change_breaks_identity():
    reset_state_caches()
    interner = StateInterner(limit=4)
    template = interner.intern((("fixed",),), generation=7)
    detector = CycleDetector(evidence_periods=1)

    def observed(waiters):
        return CanonicalState(template, (("waiters", waiters),), origin="external")

    f0 = frame(0.0, state=observed((1, 2)), cursors=(cursor(1000, next_issue=2, retired=0),), sequence=100)
    f1 = frame(1.0, state=observed((2, 1)), cursors=(cursor(1000, next_issue=3, retired=1),), sequence=104)
    detector.observe(f0, None, stop_at=math.inf)
    assert detector.observe(f1, delta(issued={1000: (1, 1)}, sequence_count=4), stop_at=math.inf) is None
    assert detector.last_difference.startswith(
        ("canonical.dynamic", "canonical.external"))


def test_raw_shared_mutable_remaining_uses_frozen_evidence_view():
    # Astra r3 PER3-P1-1: every raw frontier must compare its own frozen
    # observation, never the shared input list that later frames rewrite.
    reset_state_caches()
    remaining = [("cube", 8.0)]
    detector = CycleDetector()  # default two-period evidence
    frames = []
    results = []
    for index, value in enumerate((8.0, 7.0, 6.0, 6.0, 6.0)):
        remaining[0] = ("cube", value)
        current = SettledFrontier(
            now=float(index),
            epoch=7,
            normalized_state=(("remaining", remaining),),
            cursors=(cursor(1000, phase_end=67, next_issue=2 + index, retired=index),),
            sequence=100 + 4 * index,
        )
        interval = None if index == 0 else delta(
            issued={1000: (1, 1)}, sequence_count=4)
        frames.append(current)
        results.append(detector.observe(current, interval, stop_at=math.inf))
        if index == 1:
            # The first history still says 8.0 and the 8->7 movement is
            # diagnostically locatable; display text is not an identity key.
            assert detector.last_difference is not None
            assert "8.0" in detector.last_difference
            assert "7.0" in detector.last_difference

    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert results[3] is None
    skip = results[4]
    assert skip is not None
    assert skip.cycles == 61
    # The public input was mutated in place; history used its frozen snapshot.
    assert frames[0].normalized_state[0][1][0][1] == 6.0
    assert frames[0]._raw_snapshot != frames[2]._raw_snapshot
    run = skip.correspondence.run(1000)
    assert (run.next_issue_before, run.next_issue_after) == (6, 67)
    assert (run.retired_before, run.retired_after) == (4, 65)


def test_interner_keys_are_type_exact_not_repr_based():
    interner = StateInterner(limit=8)
    plain = interner.intern((({"x": 1}),), generation=7, fingerprint=4242)
    user = interner.intern(
        ((UserDict({"x": 1})),), generation=7, fingerprint=4242)
    assert repr(plain.template) != repr(user.template)
    assert plain.template_id != user.template_id

    # Ordinary mapping insertion order is not structural.
    first = interner.intern(
        (({"a": 1, "b": 2}),), generation=7, fingerprint=4243)
    second = interner.intern(
        (({"b": 2, "a": 1}),), generation=7, fingerprint=4243)
    assert first.template_id == second.template_id
    assert interner.intern(
        (({"x": 1}),), generation=7, fingerprint=4242) is plain

    # The authenticated UserDict key hits its fixed tree; the schema/key cost
    # is still charged and no extra canonical tree is built.
    before = metrics()
    repeated = interner.intern(
        ((UserDict({"x": 1})),), generation=7, fingerprint=4242)
    assert repeated is user
    after = metrics()
    assert after["template_builds"] == before["template_builds"]
    assert after["template_hits"] == before["template_hits"] + 1
    assert after["template_hit_nodes"] > before["template_hit_nodes"]

    # A real three-frame type change is never a 63-cycle proposal.
    detector = CycleDetector()  # default two-period evidence
    states = (
        CanonicalState(plain, (("d", 1.0),), origin="external"),
        CanonicalState(user, (("d", 1.0),), origin="external"),
        CanonicalState(plain, (("d", 1.0),), origin="external"),
    )
    results = []
    for index, state in enumerate(states):
        current = frame(
            float(index),
            state=state,
            cursors=(cursor(1000, next_issue=2 + index, retired=index),),
            sequence=100 + 4 * index,
        )
        interval = None if index == 0 else delta(
            issued={1000: (1, 1)}, sequence_count=4)
        results.append(detector.observe(current, interval, stop_at=math.inf))
    assert results == [None, None, None]
    assert detector.last_rejection != "skip"


def test_forged_mapping_tags_are_rejected_but_real_mapping_classes_are_kept():
    with pytest.raises(PeriodicContractError):
        TemplateRef.from_validated(
            1, 7, 4242, ("mapping", dict, ()), "", 0)
    with pytest.raises(PeriodicContractError):
        TemplateRef.from_validated(
            1, 7, 4242, ("mapping", int, ()), "", 0)
    accepted = TemplateRef.from_validated(
        1, 7, 4242, ("mapping", UserDict, ()), "", 0)
    assert accepted.retained_bytes > 0


def test_interner_byte_budget_keeps_oversized_live_ref_out_of_lru():
    interner = StateInterner(limit=64, retained_bytes_limit=4096)
    small = interner.intern((("small",),), generation=7)
    payload = b"x" * 8192
    large = interner.intern((("large", payload),), generation=7)
    assert large.template_id != small.template_id
    assert interner.templates == 1
    assert interner.retained_bytes <= 4096
    assert all(ref is not large for ref in interner._refs.values())
    stats = interner.stats()
    assert stats["uncached"] == 1
    assert stats["oversized"] == 1
    assert stats["live_refs"] >= 2
    assert stats["live_bytes"] >= 8192

    # Repeated large observations stay out of the index; the small admission
    # still leaves the budget intact.
    for _ in range(3):
        repeated = interner.intern((("large", payload),), generation=7)
        assert repeated.template_id != small.template_id
        assert interner.templates == 1
        assert interner.retained_bytes <= 4096
    assert stats["retained_bytes"] <= 4096

    # Returning to a small proven shape still gives the normal three-frontier
    # two-period proposal.  Full observation accounting now charges the real
    # interval maps as well, so this fixture needs a larger history envelope.
    detector = CycleDetector(history_bytes_limit=16384)
    state = CanonicalState(small, (("d", 1.0),))
    results = []
    for index in range(3):
        current = frame(
            float(index),
            state=state,
            cursors=(cursor(1000, next_issue=2 + index, retired=index),),
            sequence=100 + 4 * index,
        )
        interval = None if index == 0 else delta(
            issued={1000: (1, 1)}, sequence_count=4)
        results.append(detector.observe(current, interval, stop_at=math.inf))
    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None
    assert detector.stats()["history_bytes"] <= 16384


def test_detector_history_budget_drops_an_oversized_baseline():
    interner = StateInterner(limit=8, retained_bytes_limit=4096)
    payload = b"y" * 8192
    large_template = interner.intern((("large", payload),), generation=7)
    large_state = CanonicalState(large_template, (("d", 1.0),))
    detector = CycleDetector(history_bytes_limit=4096)
    current = frame(
        0.0,
        state=large_state,
        cursors=(cursor(1000, next_issue=2, retired=0),),
        sequence=100,
    )
    assert detector.observe(current, None, stop_at=math.inf) is None
    assert detector.last_rejection == "history_budget"
    assert detector.observations == 0
    assert detector.stats()["history_bytes"] <= 4096

    # After dropping the unaffordable baseline, the small exact shape works.
    small_template = interner.intern((("small",),), generation=7)
    small_state = CanonicalState(small_template, (("d", 1.0),))
    small_detector = CycleDetector(history_bytes_limit=16384)
    results = []
    for index in range(3):
        current = frame(
            float(index),
            state=small_state,
            cursors=(cursor(1000, next_issue=2 + index, retired=index),),
            sequence=100 + 4 * index,
        )
        interval = None if index == 0 else delta(
            issued={1000: (1, 1)}, sequence_count=4)
        results.append(
            small_detector.observe(current, interval, stop_at=math.inf))
    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None
    assert small_detector.stats()["history_bytes"] <= 16384


def test_observation_budget_charges_interval_identifiers_and_public_template_cost():
    # Same 4 KiB fixture: a legal interval whose single resource key is 8 KiB
    # cannot leave an over-budget observation behind.
    small = TemplateRef(1, 7, 1, (("small",),), "", 0, 0)
    small_state = CanonicalState(small, ())
    detector = CycleDetector(history_bytes_limit=4096)
    baseline = frame(
        0.0,
        state=small_state,
        cursors=(cursor(1000, next_issue=2, retired=0, slots=1),),
        sequence=100,
    )
    assert detector.observe(baseline, None, stop_at=math.inf) is None
    assert detector.observations == 1
    huge_resource = "r" * 8192
    current = frame(
        1.0,
        state=small_state,
        cursors=(cursor(1000, next_issue=3, retired=1, slots=1),),
        sequence=104,
    )
    assert detector.observe(
        current,
        delta(service={huge_resource: 1.0}, issued={1000: (1, 1)},
              sequence_count=4, busy={}, peaks={}),
        stop_at=math.inf,
    ) is None
    assert detector.last_rejection == "history_budget"
    assert detector.observations == 0
    assert detector.stats()["history_bytes"] <= 4096

    # The public template constructor owns its cost; a supplied zero cannot
    # under-report an 8 KiB tree into the history envelope.
    payload = b"z" * 8192
    spoofed = TemplateRef(
        1, 7, 4242, (("large", payload),), "", 0, 0)
    assert spoofed.retained_bytes > 8192
    large_state = CanonicalState(spoofed, (("d", 1.0),))
    large_frame = frame(
        0.0,
        state=large_state,
        cursors=(cursor(1000, next_issue=2, retired=0),),
        sequence=100,
    )
    detector = CycleDetector(history_bytes_limit=4096)
    assert detector.observe(large_frame, None, stop_at=math.inf) is None
    assert detector.last_rejection == "history_budget"
    assert detector.observations == 0
    assert detector.stats()["history_bytes"] <= 4096

    # A small exact template fits again; three compact frontiers produce the
    # normal two-period proposal under the same 4 KiB budget.
    detector = CycleDetector(history_bytes_limit=4096)
    results = []
    for index in range(3):
        current = frame(
            float(index),
            state=small_state,
            cursors=(cursor(
                1000, next_issue=2 + index, retired=index, slots=1),),
            sequence=100 + 4 * index,
        )
        interval = None if index == 0 else delta(
            issued={1000: (1, 1)}, sequence_count=4,
            service={}, busy={}, peaks={})
        results.append(detector.observe(current, interval, stop_at=math.inf))
    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None
    assert detector.stats()["history_bytes"] <= 4096


def test_engine_fast_path_does_not_accept_scalar_subclasses():
    class _Float(float):
        pass

    with pytest.raises(PeriodicContractError):
        frame(
            0.0,
            state=engine_state(remaining=_Float(8.0)),
            cursors=(cursor(1000, next_issue=2, retired=0),),
            sequence=100,
        )
    template = StateInterner(limit=4).intern((("fixed",),), generation=7)
    with pytest.raises(PeriodicContractError):
        CanonicalState(
            template,
            ((((("cube", _Float(1.0)),), 0, None, None),), (), (), (), ()),
            origin="engine",
        )


# ---------------------------------------------------------------------------
# Real-Engine periodic pipeline: 32 owners, 2 slots, first/middle/last phases,
# a cross-panel persistent L0C accumulator, shared L1 panel buffers and the
# M4 tile count (8 owners x 136 + 24 owners x 68 = 2720 tiles).
#
# The adapter below is test-local on purpose.  T04 owns the production
# registry/hook/apply path; this harness proves the helper contract against the
# real Engine and gives T04 a concrete normalization reference.
# ---------------------------------------------------------------------------


@dataclass
class _M4Run:
    owner: int
    run_id: int
    total: int
    panels_per_output: int
    ready: object
    complete: object
    phases: tuple
    next_issue: int = 0
    retired: int = 0
    finished: int = 0
    stage_tail: dict = field(default_factory=dict)


class _M4Harness:
    def __init__(self, owners: int = 32) -> None:
        self.engine = Engine(
            (Resource("mte2", 320.0), Resource("cube", 320.0), Resource("mte3", 320.0)),
            pools={"l1": 128.0, "l0c": 128.0},
            pool_units={"l1": 32, "l0c": 32},
        )
        self.runs: dict[int, _M4Run] = {}
        self.tasks: dict[int, tuple] = {}
        self.alloc_meta: dict[str, tuple] = {}
        for owner in range(owners):
            panels = 68
            outputs = 2 if owner < 8 else 1
            total = outputs * panels
            run_id = 1000 + owner
            ready = self.engine.boundary(f"run{owner}:ready")
            complete = self.engine.future(f"run{owner}:complete")
            phases = []
            for wave in range(outputs):
                base = wave * panels
                phases.extend(
                    (
                        (base, base + 1),
                        (base + 1, base + panels - 1),
                        (base + panels - 1, base + panels),
                    )
                )
            run = _M4Run(owner, run_id, total, panels, ready, complete, tuple(phases))
            self.runs[run_id] = run
            begin = self.engine.submit(
                f"run{owner}:begin",
                after=(ready,),
                on_complete=lambda _event, run=run: self._begin(run),
                owner=f"run{owner}",
            )
            self.tasks[ready.sequence] = (run_id, -1, 0, "ready")
            self.tasks[begin.sequence] = (run_id, -1, 0, "begin")

    def _begin(self, run: _M4Run) -> None:
        for _ in range(min(2, run.total)):
            self._issue(run)

    def _issue(self, run: _M4Run) -> None:
        index = run.next_issue
        if index >= run.total:
            return
        local = {}
        tail = run.stage_tail
        for ordinal, (name, work, acquire, free, after) in enumerate(self._stages(run, index)):
            deps = [local[key] for key in after]
            if name in tail:
                deps.append(tail[name])
            event = self.engine.submit(
                f"run{run.owner}:tile{index}:{name}",
                work,
                after=tuple(deps) or (run.ready,),
                acquire=acquire,
                free=free,
                owner=f"run{run.owner}",
            )
            self.tasks[event.sequence] = (run.run_id, index, ordinal, "stage")
            local[name] = tail[name] = event
        retire = self.engine.submit(
            f"run{run.owner}:tile{index}:retire",
            after=tuple(local.values()),
            on_complete=lambda _event, run=run: self._retire(run),
            owner=f"run{run.owner}",
        )
        self.tasks[retire.sequence] = (run.run_id, index, len(local), "retire")
        run.next_issue += 1

    def _stages(self, run: _M4Run, index: int):
        wave, panel = divmod(index, run.panels_per_output)
        first = panel == 0
        last = panel == run.panels_per_output - 1
        panel_alloc = Allocation(f"run{run.owner}:tile{index}:panel", "l1", 4.0, (run.owner,))
        accumulator = Allocation(f"run{run.owner}:wave{wave}:accumulator", "l0c", 8.0, (run.owner,))
        self.alloc_meta[panel_alloc.name] = (
            "tile",
            run.run_id,
            index,
            "panel",
            "l1",
            4.0,
            (run.owner,),
        )
        self.alloc_meta[accumulator.name] = (
            "persistent",
            run.run_id,
            wave,
            "accumulator",
            "l0c",
            8.0,
            (run.owner,),
        )
        return (
            ("load", {"mte2": 8.0}, (panel_alloc,), (), ()),
            (
                "compute",
                {"cube": 16.0},
                (accumulator,) if first else (),
                (panel_alloc,),
                ("load",),
            ),
            ("store", {"mte3": 4.0}, (), (accumulator,) if last else (), ("compute",)),
        )

    def _retire(self, run: _M4Run) -> None:
        run.retired += 1
        run.finished += 1
        if run.finished == run.total:
            self.engine.resolve(run.complete)
            return
        self._issue(run)

    def _phase_for(self, run: _M4Run, anchor: int):
        for ordinal, (start, end) in enumerate(run.phases):
            if start <= anchor < end:
                return ordinal, end
        if anchor >= run.total:
            ordinal = len(run.phases) - 1
            return ordinal, run.phases[ordinal][1]
        raise AssertionError((run.owner, run.total, anchor))

    def _cursors(self):
        result = []
        for run in sorted(self.runs.values(), key=lambda item: item.run_id):
            if run.complete.done:
                continue
            anchor = run.next_issue - 1 if run.next_issue > run.retired else run.next_issue
            ordinal, end = self._phase_for(run, anchor)
            result.append(
                Cursor(
                    run.run_id,
                    ordinal,
                    end,
                    run.next_issue,
                    run.retired,
                    tuple(range(run.retired, run.next_issue)).__class__(
                        (slot - run.retired for slot in range(run.retired, run.next_issue))
                    ),
                )
            )
        return tuple(result)

    def _norm_alloc(self, item: Allocation):
        scope, run_id, anchor, base, pool, size, units = self.alloc_meta[item.name]
        run = self.runs[run_id]
        if scope == "tile":
            anchor -= run.retired
        return (scope, run_id, anchor, base, pool, size, tuple(units))

    def normalized_state(self):
        engine = self.engine
        retired = {run_id: run.retired for run_id, run in self.runs.items()}

        def task_key(sequence):
            run_id, tile, ordinal, kind = self.tasks[sequence]
            return (run_id, tile - retired[run_id], ordinal, kind)

        records = []
        for sequence, task in engine._tasks.items():
            key = task_key(sequence)
            records.append(
                (
                    key,
                    tuple(sorted(task.remaining.items())),
                    tuple(sorted(task.demand.items())),
                    task.waiting,
                    max(0.0, task.release - engine.now),
                    task.latency,
                    None if task.start is None else max(0.0, task.start + task.latency - engine.now),
                    tuple(sorted(task.caps.items())),
                    tuple(self._norm_alloc(item) for item in task.acquire),
                    tuple(self._norm_alloc(item) for item in task.free),
                    tuple(sorted((name, tuple(units)) for name, units in task.placement.items())),
                    tuple(sorted((pair, amount) for pair, amount in task.allocation_demands)),
                    task.priority,
                    task.coupled,
                    tuple(task_key(item) for item in task.event._waiters),
                    sequence in engine._active,
                    sequence in engine._ready_seen,
                    sequence in engine._completable_seen,
                )
            )
        records.sort(key=lambda item: item[0])
        ready = tuple(
            (engine._tasks[sequence].priority, task_key(sequence))
            for _priority, sequence in engine._ready_heap
            if sequence in engine._tasks
        )
        releases = tuple(
            (time - engine.now, task_key(sequence)) for time, sequence in engine._releases if sequence in engine._tasks
        )
        completable = tuple(task_key(sequence) for sequence in engine._completable if sequence in engine._tasks)
        live_allocations = tuple(sorted(self._norm_alloc(engine.allocations[name]) for name in engine.allocations))
        return (
            ("schema", "engine-periodic-engine-test-v1"),
            (
                "resources",
                tuple(
                    sorted(
                        (name, item.rate, item.unit, item.policy, item.units) for name, item in engine.resources.items()
                    )
                ),
            ),
            (
                "pools",
                tuple(sorted((name, capacity, engine.pool_units[name]) for name, capacity in engine.pools.items())),
            ),
            ("records", tuple(records)),
            ("ready_heap", ready),
            ("releases", releases),
            ("completable", completable),
            ("live_allocations", live_allocations),
            ("allocated", tuple(sorted(engine.allocated.items()))),
            (
                "unit_allocated",
                tuple(sorted(((pool, unit), amount) for (pool, unit), amount in engine.unit_allocated.items())),
            ),
            (
                "promises",
                tuple(sorted((event.name, event.done) for event in engine._promises.values())),
            ),
        )

    def frame(self):
        return SettledFrontier(
            now=self.engine.now,
            epoch=1,
            normalized_state=self.normalized_state(),
            cursors=self._cursors(),
            sequence=self.engine._sequence,
        )

    def snapshot(self):
        return {
            "now": self.engine.now,
            "sequence": self.engine._sequence,
            "serviced": dict(self.engine.serviced),
            "busy": dict(self.engine.busy_ms),
            "allocations": self.engine.allocation_event_count,
            "completed": self.engine.completed_count,
            "peaks": dict(self.engine.peak_unit_allocated),
            "next": {run_id: run.next_issue for run_id, run in self.runs.items()},
            "retired": {run_id: run.retired for run_id, run in self.runs.items()},
        }

    def interval(self, previous):
        if self.engine.now < previous["now"]:
            raise AssertionError("time moved backwards")
        active = {run_id: run for run_id, run in self.runs.items() if not run.complete.done}
        return IntervalDelta(
            dt=self.engine.now - previous["now"],
            issued_retired_by_run={
                run_id: (
                    run.next_issue - previous["next"][run_id],
                    run.retired - previous["retired"][run_id],
                )
                for run_id, run in active.items()
            },
            sequence_count=self.engine._sequence - previous["sequence"],
            service_by_resource={
                name: value - previous["serviced"][name] for name, value in self.engine.serviced.items()
            },
            busy_by_resource={name: value - previous["busy"][name] for name, value in self.engine.busy_ms.items()},
            allocation_event_count=(self.engine.allocation_event_count - previous["allocations"]),
            per_unit_peaks=dict(self.engine.peak_unit_allocated),
            logical_completions=(self.engine.completed_count - previous["completed"]),
            crossed_external_or_opaque=False,
        )

    def apply_skip(self, skip):
        total = skip.total_delta()
        self.engine.now += skip.period * skip.cycles
        self.engine._sequence += skip.correspondence.event_sequence_offset
        for name, amount in total.service_by_resource.items():
            self.engine.serviced[name] += amount
        for name, amount in total.busy_by_resource.items():
            self.engine.busy_ms[name] += amount
        self.engine.allocation_event_count += total.allocation_event_count
        self.engine.completed_count += total.logical_completions
        for (pool, unit), value in total.per_unit_peaks.items():
            self.engine.peak_unit_allocated[(pool, unit)] = max(
                self.engine.peak_unit_allocated.get((pool, unit), 0.0), value
            )
        for pool in self.engine.pools:
            values = [value for (item, _unit), value in self.engine.peak_unit_allocated.items() if item == pool]
            self.engine.peak_allocated[pool] = max([self.engine.peak_allocated[pool], *values])
        for run_id, shift in skip.correspondence.tile_offset_by_run.items():
            if not shift:
                continue
            before_issue = skip.correspondence.run(run_id).next_issue_before
            for sequence, (task_run, tile, ordinal, kind) in list(self.tasks.items()):
                if task_run == run_id and tile < before_issue:
                    self.tasks[sequence] = (task_run, tile + shift, ordinal, kind)
            for name, meta in list(self.alloc_meta.items()):
                scope, task_run, anchor, base, pool, size, units = meta
                if scope == "tile" and task_run == run_id and anchor < before_issue:
                    self.alloc_meta[name] = (
                        scope,
                        task_run,
                        anchor + shift,
                        base,
                        pool,
                        size,
                        units,
                    )
            corr = skip.correspondence.run(run_id)
            run = self.runs[run_id]
            run.next_issue = corr.next_issue_after
            run.retired = corr.retired_after
            run.finished += corr.finished_offset


def _run_detailed(harness: _M4Harness) -> None:
    while not all(run.complete.done for run in harness.runs.values()):
        try:
            harness.engine.advance(max_steps=1)
        except ExecutionError as exc:
            if str(exc) != "execution_step_budget_exhausted":
                raise


def _run_accelerated(harness: _M4Harness):
    detector = CycleDetector()
    previous = harness.snapshot()
    skips = cycles = 0
    while not all(run.complete.done for run in harness.runs.values()):
        frame = harness.frame()
        skip = detector.observe(frame, harness.interval(previous), stop_at=math.inf)
        previous = harness.snapshot()
        if skip is not None:
            for run in skip.correspondence.runs:
                assert run.next_issue_after <= run.phase_end_exclusive
            harness.apply_skip(skip)
            previous = harness.snapshot()
            skips += 1
            cycles += skip.cycles
            continue
        try:
            harness.engine.advance(max_steps=1)
        except ExecutionError as exc:
            if str(exc) != "execution_step_budget_exhausted":
                raise
    return skips, cycles


def test_real_engine_m4_periodic_cohort_matches_detailed_reference():
    # Test-local simplified pipeline: its two middle phases total 125 skipped
    # cycles.  The actual T18 M4 ordinary entry is verified separately at
    # 2 skips / 113 cycles in logs/r3/m4-latest.json.
    detailed = _M4Harness()
    _run_detailed(detailed)

    accelerated = _M4Harness()
    skips, cycles = _run_accelerated(accelerated)

    # The target production shape is actually present.
    assert sum(run.total for run in detailed.runs.values()) == 2720
    assert detailed.engine.completed_count == 10_944
    assert detailed.engine.allocation_event_count == 5_520
    assert detailed.engine.peak_allocated["l1"] == 8.0
    assert detailed.engine.peak_allocated["l0c"] == 16.0

    assert skips >= 2
    assert cycles >= 125
    assert accelerated.engine.advance_steps < detailed.engine.advance_steps
    assert accelerated.engine.completed_count == detailed.engine.completed_count
    assert accelerated.engine._sequence == detailed.engine._sequence
    assert accelerated.engine.allocation_event_count == detailed.engine.allocation_event_count
    assert accelerated.engine.now == pytest.approx(detailed.engine.now, rel=1e-12)
    for name in detailed.engine.serviced:
        assert accelerated.engine.serviced[name] == pytest.approx(detailed.engine.serviced[name], rel=1e-12)
        assert accelerated.engine.busy_ms[name] == pytest.approx(detailed.engine.busy_ms[name], rel=1e-12)
    assert accelerated.engine.peak_unit_allocated == detailed.engine.peak_unit_allocated
    assert accelerated.engine.allocated == {"l1": 0.0, "l0c": 0.0}
    assert accelerated.engine.unit_allocated == {}
    assert all(run.complete.done for run in accelerated.runs.values())
    assert all(run.finished == run.total for run in accelerated.runs.values())
