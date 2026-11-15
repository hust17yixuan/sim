"""Round-4 boundary and Round-5/6 closure counterexamples for the accelerator.

[Role] Reproduce each Astra Round-4 counterexample against the canonical Python
  Engine and the native ``c``/``auto`` backends: delayed-release adoption,
  exhausted-key hand-back, solver capacity, late completion observers, an
  arithmetic prefix in a multi-step epoch and rejected link-batch atomicity.
  R5 adds effective default-placement capacity and the unified failure prefix;
  R6 adds same-instant output-buffer continuation at every public boundary.
[Invariants] Every fixture asserts backend parity on the public clock, starts,
  completions and service/busy totals; a kernel fault is never replayed.  The
  tests use the session's isolated cache fixture only.
[Timeouts & Locks] No sleeps or threads; subprocess compilation belongs to the
  parity module, so these tests only load the already-built library.
[Side-Effect Policy] Writes only the temporary cache directory owned by
  ``tests/conftest.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "mindie_sim" / "prediction" / "runtime"
sys.path.insert(0, str(ROOT))

from mindie_sim.prediction.runtime import accel_bridge, accel_build  # noqa: E402
from mindie_sim.prediction.runtime.accel_tables import (  # noqa: E402
    Arena, KernelFault,
)
from mindie_sim.prediction.runtime.engine import (  # noqa: E402
    Engine, ExecutionError, Resource,
)

BACKENDS = ("python", "c", "auto")


def _completed_times(events):
    return tuple(None if event.time is None else float(event.time)
                 for event in events)


def _delayed_fixture(backend, *, warm_work, warm_count=62):
    engine = Engine([Resource("r", 64.0)], backend=backend)
    warm = [engine.submit(f"w{index}", dict(warm_work))
            for index in range(warm_count)]
    root = engine.submit("A", {"r": 1.0}, release=2.0)
    follower = engine.submit("B", {"r": 1.0}, after=(root,))
    return engine, warm, root, follower


def test_r4_1_delayed_release_adoption_matches_python():
    """R4-1: adoption transfers the delayed root's release ownership to C."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r", 1.0)], backend=backend)
        root = engine.submit("A", {"r": 1.0}, release=2.0)
        follower = engine.submit("B", {"r": 1.0}, after=(root,))
        warm = [engine.submit(f"w{index}", {}) for index in range(62)]
        engine.advance()
        assert all(event.done for event in (root, follower, *warm)), backend
        status = engine.accel_status()
        results[backend] = (
            engine.now, root.started, root.time, follower.started,
            follower.time, engine.completed_count, status["fallback_reason"])
        assert engine._releases == [], backend
    assert results["python"][:6] == (4.0, 2.0, 3.0, 3.0, 4.0, 64)
    for backend in ("c", "auto"):
        assert results[backend][:6] == results["python"][:6], backend
        assert results[backend][6] is None


def test_r4_1_positive_burst_survives_horizon_continuation():
    """R4-1: a delayed root keeps exact parity across 1.0/2.5 horizon steps."""
    results = {}
    for backend in BACKENDS:
        engine, warm, root, follower = _delayed_fixture(
            backend, warm_work={"r": 1.0})
        engine.advance(horizon=1.0)
        first = (engine.now, root.started, follower.started)
        engine.advance(horizon=2.5)
        second = (engine.now, root.time, follower.time)
        engine.advance()
        results[backend] = (
            first, second, engine.now, _completed_times(warm),
            _completed_times((root, follower)),
            engine.serviced["r"], engine.accel_status()["fallback_reason"])
    for backend in ("c", "auto"):
        assert results[backend][:6] == results["python"][:6], backend
        assert results[backend][6] is None, backend
    assert results["python"][0][0] == 1.0
    assert results["python"][2] == 2.03125
    assert results["python"][4] == (2.015625, 2.03125)


def test_r4_1_native_delayed_root_matches_adopted_root():
    """R4-1: a delayed root submitted after attachment owns its release in C."""
    results = {}
    for backend in BACKENDS:
        engine, warm, root, follower = _delayed_fixture(
            backend, warm_work={"r": 1.0}, warm_count=64)
        engine.advance()
        results[backend] = (
            engine.now, _completed_times(warm), root.started, root.time,
            follower.started, follower.time, len(engine._releases),
            engine.completed_count)
    assert results["python"] == (
        2.03125, (1.0,) * 64, 2.0, 2.015625, 2.015625, 2.03125, 0, 66)
    for backend in ("c", "auto"):
        assert results[backend] == results["python"], backend


def test_r4_1_delayed_root_hands_back_before_its_release():
    """R4-1: a mixed frontier at t=1.0 restores A's future release once."""
    results = {}
    for backend in BACKENDS:
        engine, warm, root, follower = _delayed_fixture(
            backend, warm_work={})
        engine.advance(horizon=1.0)
        extra = engine.submit("D", {"r": 1.0},
                              on_complete=lambda event: None)
        engine.advance()
        results[backend] = (
            engine.now, root.started, root.time, follower.started,
            follower.time, extra.started, extra.time,
            engine.completed_count, len(engine._releases),
            engine.accel_status()["fallback_reason"])
    assert results["python"][:9] == (
        2.03125, 2.0, 2.015625, 2.015625, 2.03125, 1.0, 1.015625, 65, 0)
    for backend in ("c", "auto"):
        assert results[backend][:9] == results["python"][:9], backend
    assert results["c"][9] == "mixed_frontier"
    assert results["auto"][9] == "mixed_frontier"


def test_r4_2_exhausted_key_handback_keeps_latency_task_alive():
    """R4-2: work-free latency residue hands back with no zero-valued key."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r", 1.0)], backend=backend)
        for index in range(63):
            engine.submit(f"w{index}", {})
        sleep = engine.submit("sleep", {"r": 1.0}, latency=10.0, coupled=True)
        engine.advance(horizon=2.0)
        extra = engine.submit("D", {"r": 1.0},
                              on_complete=lambda event: None)
        engine.advance()
        results[backend] = (engine.now, sleep.time, extra.time,
                            engine.completed_count, dict(engine.serviced),
                            engine.accel_status()["native_steps"])
        assert engine._tasks == {}, backend
    assert results["python"][:5] == (10.0, 10.0, 3.0, 65, {"r": 2.0})
    for backend in ("c", "auto"):
        assert results[backend][:5] == results["python"][:5], backend
        assert results[backend][5] > 0


def test_r4_2_partial_multiresource_handback_is_canonical():
    """R4-2: r1 is exhausted while r2 is live; only positive keys persist."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r1", 100.0), Resource("r2", 1.0)],
                        backend=backend)
        for index in range(63):
            engine.submit(f"w{index}", {})
        mixed = engine.submit("M", {"r1": 1.0, "r2": 10.0})
        engine.advance(horizon=0.5)
        extra = engine.submit("D", {"r1": 1.0},
                              on_complete=lambda event: None)
        engine.advance(horizon=0.5001)
        task = engine._tasks[mixed.sequence]
        assert all(amount > 0.0 for amount in task.remaining.values()), backend
        snapshot = (engine.now, dict(task.remaining), dict(task.service))
        engine.advance()
        results[backend] = (snapshot, engine.now, mixed.time, extra.time,
                            dict(engine.serviced))
    assert results["python"][0] == (
        0.5001, {"r2": 9.4999}, {"r1": 1.0, "r2": 0.5001})
    for backend in ("c", "auto"):
        assert results[backend] == results["python"], backend


def test_r4_3_overlapping_unit_membership_runs_natively():
    """R4-3: 65,599 overlapping membership entries execute in the kernel."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
        events = []
        for index in range(1025):
            units = tuple(range(63 if index == 1024 else 64))
            events.append(engine.submit(str(index), {"r": 1.0},
                                        placement={"r": units}))
        engine.advance()
        status = engine.accel_status()
        results[backend] = (
            engine.now, engine.completed_count,
            tuple(event.time for event in events), dict(engine.serviced),
            status["native_steps"], status["fallback_reason"])
    assert results["python"][0] == 16.015873015873016
    assert results["python"][1] == 1025
    assert results["python"][3] == {"r": 1025.0}
    for backend in ("c", "auto"):
        assert results[backend][:4] == results["python"][:4], backend
        assert results[backend][4] > 0 and results[backend][5] is None


def test_r4_3_wide_coupled_frontier_runs_natively():
    """R4-3: 8,193 coupled tasks exceed the former 8,192-flow limit."""
    results = {}
    units = tuple(range(64))
    for backend in BACKENDS:
        engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
        events = [engine.submit(str(index), {"r": 1.0},
                                placement={"r": units}, coupled=True)
                  for index in range(8193)]
        engine.advance()
        status = engine.accel_status()
        results[backend] = (
            engine.now, engine.completed_count, events[0].time, events[-1].time,
            dict(engine.serviced), status["native_steps"],
            status["fallback_reason"])
    assert results["python"][:5] == (
        128.015625, 8193, 128.015625, 128.015625, {"r": 8193.0})
    for backend in ("c", "auto"):
        assert results[backend][:5] == results["python"][:5], backend
        assert results[backend][5] > 0 and results[backend][6] is None


def test_r4_3_solver_capacity_declines_before_native_execution():
    """R4-3: every remaining static solver cap is preflighted, never faulted."""
    resources = [Resource(f"r{index}", 100.0) for index in range(32)]
    work = {f"r{index}": 1.0 for index in range(32)}
    engine = Engine(resources, backend="c")
    for index in range(4096):
        engine.submit(f"c{index}", work, coupled=True)
    status = engine.accel_status()
    assert status["fallback_reason"] == "capacity_exceeded"
    assert status["native_steps"] == 0
    engine.advance(horizon=0.0)
    assert status["accel_backend"] == "python_fallback"

    engine = Engine([Resource("r", 64.0, units=64)], backend="c")
    units = tuple(range(64))
    for index in range(32769):
        engine.submit(f"p{index}", {"r": 1.0}, placement={"r": units})
    status = engine.accel_status()
    assert status["fallback_reason"] == "capacity_exceeded"
    assert status["native_steps"] == 0
    engine.advance(horizon=0.0)
    assert status["fallback_reason"] == "capacity_exceeded"


def test_r4_4_late_completion_callback_hands_back_before_the_step():
    """R4-4: an observer registered after admission fires at the true instant."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r", 1.0)], backend=backend)
        for index in range(63):
            engine.submit(f"w{index}", {})
        first = engine.submit("a", {"r": 1.0})
        second = engine.submit("b", {"r": 2.0})
        engine.advance(horizon=0.5)
        seen, extra = [], []

        def callback(event):
            seen.append((event.time, engine.now))
            extra.append(engine.submit("extra", {"r": 1.0}))

        engine.on_complete(first, callback)
        engine.advance()
        results[backend] = (
            tuple(seen), second.time, extra[0].started, extra[0].time,
            engine.now, engine.accel_status()["native_steps"],
            engine.accel_status()["fallback_reason"])
    assert results["python"] == (((2.0, 2.0),), 4.0, 2.0, 4.0, 4.0, 0,
                                 "requested_python")
    for backend in ("c", "auto"):
        assert results[backend][:5] == results["python"][:5], backend
        assert results[backend][5] > 0


def test_r4_5_late_underflow_publishes_the_valid_prefix_once():
    """R4-5: a later E_ARITH keeps the already accepted completions/starts."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("tiny", 5e-324), Resource("r", 1.0)],
                        backend=backend)
        for index in range(63):
            engine.submit(f"w{index}", {})
        good = engine.submit("good", {"r": 0.5})
        engine.submit("a", {"tiny": 5e-323})
        late = engine.submit("b", {"tiny": 5e-323}, release=1.0)
        with pytest.raises(ExecutionError,
                           match="service_rate_underflow:tiny"):
            engine.advance()
        results[backend] = (
            engine.now, good.time, late.started, dict(engine.serviced),
            dict(engine.busy_ms), engine.completed_count,
            engine.advance_steps, engine.accel_status()["native_steps"])
    assert results["python"] == (
        1.0, 0.5, 1.0, {"tiny": 0.0, "r": 0.5},
        {"tiny": 1.0, "r": 0.5}, 64, 2, 0)
    for backend in ("c", "auto"):
        assert results[backend] == (
            1.0, 0.5, 1.0, {"tiny": 0.0, "r": 0.5},
            {"tiny": 1.0, "r": 0.5}, 64, 2, 2), backend


def _default_placement_frontier(backend, coupled):
    """R5-1: 32,768 implicit plus one explicit 63-unit placement."""
    engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
    events = [engine.submit(str(index), {"r": 1.0}, coupled=coupled)
              for index in range(32768)]
    pinned = engine.submit("pinned", {"r": 1.0},
                           placement={"r": tuple(range(63))},
                           coupled=coupled)
    engine.advance()
    status = engine.accel_status()
    return (engine.now, engine.completed_count, dict(engine.serviced),
            pinned.time, status["fallback_reason"], status["native_steps"],
            all(event.done for event in events))


def test_r5_1_effective_default_placement_capacity_declines_cleanly():
    """R5-1: implicit placements count as all units before native execution."""
    for coupled in (False, True):
        results = {backend: _default_placement_frontier(backend, coupled)
                   for backend in BACKENDS}
        expected = (512.015873015873, 32769, {"r": 32769.0},
                    512.015873015873, None, 0, True)
        assert results["python"][:4] == expected[:4]
        assert results["python"][6] is True
        for backend in ("c", "auto"):
            assert results[backend][:4] == results["python"][:4], (backend, coupled)
            assert results[backend][4] == "capacity_exceeded", (backend, coupled)
            assert results[backend][5] == 0, (backend, coupled)
            assert results[backend][6] is True, (backend, coupled)


def test_r5_1_post_progress_handback_counts_effective_units():
    """R5-1: a mixed default batch hands prior native progress back cleanly."""
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
        warm = [engine.submit(f"w{index}", {"r": 1.0}) for index in range(64)]
        engine.advance(horizon=0.5)
        midpoint = (engine.now, engine.advance_steps,
                    engine.accel_status()["native_steps"])
        batch = [engine.submit(f"d{index}", {"r": 1.0})
                 for index in range(32704)]
        pinned = engine.submit("pinned", {"r": 1.0},
                               placement={"r": tuple(range(63))})
        engine.advance()
        status = engine.accel_status()
        results[backend] = (
            midpoint, engine.now, engine.completed_count,
            dict(engine.serviced), dict(engine.busy_ms), engine.advance_steps,
            warm[0].time, pinned.time, batch[0].time, batch[-1].time,
            status["fallback_reason"])
    assert results["python"][:6] == (
        (0.5, 1, 0), 512.015873015873, 32769, {"r": 32769.0},
        {"r": 512.015873015873}, 3)
    for backend in ("c", "auto"):
        assert results[backend][0][:2] == results["python"][0][:2], backend
        assert results[backend][0][2] > 0, backend
        assert results[backend][1:10] == results["python"][1:10], backend
        assert results[backend][1] == 512.015873015873, backend
        assert results[backend][10] == "capacity_exceeded", backend


def _compute_rates_failure(backend):
    """R5-2: a zero coupled scale after 64 accepted completions."""
    engine = Engine([Resource("r", 1.0), Resource("huge", 1e308)],
                    backend=backend)
    for index in range(63):
        engine.submit(str(index), {})
    good = engine.submit("good", {"r": 0.5})
    late = engine.submit("late", {"huge": 5e-324}, release=1.0, coupled=True)
    expected = ZeroDivisionError if backend == "python" else KernelFault
    with pytest.raises(expected) as info:
        engine.advance()
    return engine, good, late, info.value


def test_r5_2_compute_rates_prefix_is_published_once():
    """R5-2: every failure after accepted work publishes one coherent prefix."""
    results = {}
    for backend in BACKENDS:
        engine, good, late, error = _compute_rates_failure(backend)
        if backend != "python":
            assert error.has_prefix is True, backend
            assert error.prefix == (1.0, 0, 2, 1, 1), backend
        results[backend] = (
            engine.now, good.time, late.started, engine.completed_count,
            engine.advance_steps, dict(engine.serviced), dict(engine.busy_ms),
            engine.accel_status()["native_steps"])
    expected = (1.0, 0.5, 1.0, 64, 2, {"r": 0.5, "huge": 0.0},
                {"r": 0.5, "huge": 0.0})
    assert results["python"][:7] == expected
    for backend in ("c", "auto"):
        assert results[backend][:7] == expected, backend
        assert results[backend][7] == 2, backend


def test_r5_2_step_zero_failure_exposes_no_prefix():
    """R5-2: an initial compute_rates failure after admission has no prefix."""
    for backend in BACKENDS:
        engine = Engine([Resource("r", 1.0), Resource("huge", 1e308)],
                        backend=backend)
        for index in range(63):
            engine.submit(str(index), {})
        late = engine.submit("late", {"huge": 5e-324}, release=0.0,
                             coupled=True)
        expected = ZeroDivisionError if backend == "python" else KernelFault
        with pytest.raises(expected) as info:
            engine.advance()
        if backend != "python":
            assert info.value.has_prefix is False, backend
            assert info.value.prefix is None, backend
        assert engine.now == 0.0, backend
        assert engine.advance_steps == 0, backend
        assert dict(engine.serviced) == {"r": 0.0, "huge": 0.0}, backend
        assert late.started == 0.0, backend


def _link_arena(rate=1.0):
    status = accel_build.build_accel()
    assert status["ok"], status.get("detail") or status.get("reason")
    library = accel_bridge.load_library(status)
    arena = Arena(library)
    arena.configure([rate], [1], [0])
    rows = [
        (1, 0, 0, 1, 0, 0.0, 0.0, 0.0, [(0, 1.0, float("inf"), ())]),
        (2, 0, 0, 1, 2, 0.0, 0.0, 0.0, [(0, 1.0, float("inf"), ())]),
    ]
    return arena, arena.submit(rows)


def test_r4_6_rejected_link_batch_commits_no_prefix_edge():
    """R4-6: [valid, invalid] link batches are atomic, like submit."""
    try:
        control, control_slots = _link_arena()
        poisoned, poisoned_slots = _link_arena()
        with pytest.raises(KernelFault) as info:
            poisoned.link([(poisoned_slots[0], poisoned_slots[1], 2),
                           (poisoned_slots[0], 999999, 3)])
        assert info.value.code == 1
        control.link([(control_slots[0], control_slots[1], 2)])
        poisoned.link([(poisoned_slots[0], poisoned_slots[1], 2)])
        observed = []
        for arena in (control, poisoned):
            now, _boundary, steps, ncomp, _nstart = arena.run(
                0.0, float("inf"), float("inf"), 100)
            observed.append((now, steps, ncomp,
                             tuple(arena.completions(ncomp))))
        assert observed[0] == observed[1]
        assert observed[0][0] == 1.0 and observed[0][2] == 1
        assert observed[0][3][0][0] == 1 and observed[0][3][0][1] == 1.0
    finally:
        for arena in (control, poisoned):
            arena.close()


def _r6_frontier(backend, count, units=1):
    """R6-1: ``count`` identical work-1 tasks on one resource."""
    engine = Engine([Resource("r", float(count), units=units)], backend=backend)
    events = [engine.submit(f"t{index}", {"r": 1.0}) for index in range(count)]
    return engine, events


@pytest.mark.parametrize(("count", "units"), (
    (4097, 1), (8192, 1), (16384, 1),
    (4097, 64), (8192, 64), (16384, 64)))
def test_r6_1_horizon_drains_every_simultaneous_completion(count, units):
    """R6-1: a full completion buffer continues at ``now``, not a budget stop."""
    results = {}
    for backend in BACKENDS:
        engine, events = _r6_frontier(backend, count, units)
        engine.advance(horizon=1.0)
        first = (engine.now, engine.completed_count, engine.serviced["r"],
                 engine.advance_steps, events[-1].started, events[-1].time,
                 all(event.started == 0.0 and event.time == 1.0
                     for event in events))
        engine.advance(horizon=1.0)
        results[backend] = first + (
            engine.completed_count, events[-1].time, engine.now)
    expected = (1.0, count, float(count), 1, 0.0, 1.0, True,
                count, 1.0, 1.0)
    assert results["python"] == expected
    for backend in ("c", "auto"):
        assert results[backend] == expected, backend


def test_r6_1_zero_horizon_drains_every_admission_start():
    """R6-1: ``horizon=now`` publishes all 4,097 starts with zero steps."""
    for backend in BACKENDS:
        engine, events = _r6_frontier(backend, 4097)
        engine.advance(horizon=0.0)
        assert engine.now == 0.0 and engine.advance_steps == 0, backend
        assert events[-1].started == 0.0, backend
        assert all(event.started == 0.0 and not event.done
                   for event in events), backend
        if backend != "python":
            assert engine.accel_status()["native_calls"] > 0, backend


def test_r6_1_zero_work_completions_drain_inside_admission():
    """R6-1: zero-work tasks publish, complete and reclaim at ``now``."""
    for backend in BACKENDS:
        engine = Engine([Resource("r", 4097.0)], backend=backend)
        events = [engine.submit(f"z{index}", {}) for index in range(4097)]
        engine.advance(horizon=0.0)
        assert engine.now == 0.0 and engine.advance_steps == 0, backend
        assert engine.completed_count == 4097, backend
        assert events[-1].started == events[-1].time == 0.0, backend
        assert engine._tasks == {}, backend


def test_r6_1_until_waits_for_all_simultaneous_completions():
    """R6-1: an observed target cannot return before its 4,096 peers settle."""
    for backend in BACKENDS:
        engine, events = _r6_frontier(backend, 4097)
        completed = engine.advance(until=(events[0],))
        assert completed == (events[0],), backend
        assert engine.completed_count == 4097, backend
        assert events[0].time == events[-1].time == 1.0, backend


def test_r6_1_until_settles_successors_at_the_same_instant():
    """R6-1: an observed boundary closes the instant before handing back."""
    for backend in BACKENDS:
        engine, wave = _r6_frontier(backend, 64)
        tail = engine.submit("b", {"r": 1.0}, after=(wave[0],))
        engine.advance(until=(wave[0],))
        assert engine.now == 1.0 and engine.completed_count == 64, backend
        assert tail.started == 1.0 and tail.time is None, backend


def test_r6_1_max_steps_one_never_reports_a_settled_budget():
    """R6-1: one numerical step completes all 4,097 tasks without an error."""
    for backend in BACKENDS:
        engine, events = _r6_frontier(backend, 4097)
        engine.advance(max_steps=1)
        assert engine.now == 1.0 and engine.advance_steps == 1, backend
        assert engine.completed_count == 4097, backend
        assert events[-1].time == 1.0, backend


def test_r6_1_step_budget_still_bounds_the_next_wave():
    """R6-1: output draining must not consume the caller's next-wave budget."""
    for backend in BACKENDS:
        engine, wave = _r6_frontier(backend, 4097)
        tail = engine.submit("b", {"r": 1.0}, after=(wave[0],))
        with pytest.raises(ExecutionError,
                           match="execution_step_budget_exhausted"):
            engine.advance(max_steps=1)
        assert engine.completed_count == 4097, backend
        assert all(event.time == 1.0 for event in wave), backend
        assert tail.started == 1.0 and tail.time is None, backend


def test_r6_1_service_buffer_overflow_drains_all_totals():
    """R6-1: 32,800 service records settle 1,025 completions on the first call."""
    names = [f"r{index}" for index in range(32)]
    work = {name: 1.0 for name in names}
    for backend in BACKENDS:
        engine = Engine([Resource(name, 1025.0) for name in names],
                        backend=backend)
        events = [engine.submit(f"t{index}", work) for index in range(1025)]
        engine.advance(horizon=1.0)
        assert engine.now == 1.0 and engine.advance_steps == 1, backend
        assert engine.completed_count == 1025, backend
        assert events[-1].time == 1.0, backend
        assert dict(engine.serviced) == {
            name: 1025.0 for name in names}, backend


@pytest.mark.parametrize("target", (0, 4096))
def test_r6_1_callback_after_a_start_output_boundary(target):
    """R6-1: a completion observer hands back after a 4,097-start drain."""
    results = {}
    for backend in BACKENDS:
        engine, events = _r6_frontier(backend, 4097)
        engine.advance(horizon=0.0)
        seen = []

        def callback(event):
            seen.append((event.name, event.time, engine.now))
            engine.submit("c", {"r": 1.0})

        engine.on_complete(events[target], callback)
        engine.advance(horizon=1.0)
        follow = next(task.event for task in engine._tasks.values()
                      if task.event.name == "c")
        first = (engine.now, engine.completed_count, events[-1].time,
                 tuple(seen), follow.started, follow.time)
        engine.advance(horizon=2.0)
        results[backend] = first + (
            follow.time, engine.now, engine.accel_status()["native_calls"])
    expected = (1.0, 4097, 1.0, ((f"t{target}", 1.0, 1.0),), 1.0, None,
                1.0 + 1.0 / 4097.0, 1.0 + 1.0 / 4097.0)
    assert results["python"][:8] == expected
    for backend in ("c", "auto"):
        assert results[backend][:8] == expected, backend
        assert results[backend][8] > 0, backend
