"""Round-13 closure: finalizer metadata declines and wire holds survive.

[Role] Reproduce ``R12-1`` (the admission hold lost across a completion or
  service output split) and ``R12-2`` (the reference's last-consumer metadata
  lifetime), then exercise the engine-arena/2 ``pending_hold`` continuation at
  the raw wire and through the Python bridge with inert Engine tasks.
[Invariants] A finalizer-bearing metadata leaf is never native-eligible, so
  those fixtures execute through the unchanged Python reference with exact
  ``child.started``, derived work, clock and service parity.  The native
  bridge fixtures keep every Engine task inert and flag exactly one row at the
  wire, so eligibility and scheduler cells are untouched while the
  ``F_CALLBACK`` hold path is really reached.
[Timeouts & Locks] No threads or sleeps; one bounded library memo and one
  bounded native call per output window.
[Side-Effect Policy] Compiles only into the isolated test cache and writes only
  test-owned temporary state.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import weakref

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime import (
    accel_build, accel_bridge, accel_sync, accel_tables as tables,
)
from mindie_sim.prediction.runtime.accel_tables import _stable_task_payloads

BACKENDS = ("python", "c", "auto")
NATIVE = ("c", "auto")
ROOT = Path(__file__).resolve().parents[1]


class _Token:
    """A weakrefable custom leaf: never an inert built-in."""


class _Dict(dict):
    """A dict subclass is not one exact private metadata mapping."""


def _close(engine) -> None:
    if engine._accel is not None:
        engine._accel.close()


def _same(left, right, label) -> None:
    assert left == right, (label, left, right)


def _last_consumer(backend, derived):
    """R12-2: the token-bearing root is the reference's last consumer."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    for index, amount in enumerate([2.0] + [3.0] * 62):
        engine.submit(f"root{index}", {"r": amount}, caps={"r": 1.0})
    token = _Token()
    root = engine.submit("root63", {"r": 1.0}, caps={"r": 1.0},
                         metadata={"token": token})
    child = engine.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(root,))
    seen, extra = [], []

    def release():
        seen.append((engine.now, root.done, child.started, child.done,
                     engine.completed_count, engine.serviced["r"]))
        if derived:
            work = 1.0 if child.started is None else 2.0
            extra.append((work, engine.submit(
                "extra", {"r": work}, on_complete=lambda _event: None)))

    finalizer = weakref.finalize(token, release)
    del token
    engine.advance()
    result = {
        "seen": list(seen), "extra": [(work, event.started, event.time)
                                      for work, event in extra],
        "child": (child.started, child.time), "now": engine.now,
        "service": engine.serviced["r"], "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "finished": not finalizer.alive,
    }
    _close(engine)
    return result


@pytest.mark.parametrize("derived", (False, True))
def test_r12_2_last_consumer_metadata_matches_the_reference(derived):
    """The last integrated consumer's token releases at the reference instant."""
    results = {backend: _last_consumer(backend, derived) for backend in BACKENDS}
    control = results["python"]
    expected = ([(1.0, True, 1.0, False, 1, 64.0)],
                [(2.0, 2.0, 3.0)] if derived else [],
                (1.0, 2.0), 3.0, 192.0 if derived else 190.0,
                66 if derived else 65)
    assert (control["seen"], control["extra"], control["child"], control["now"],
            control["service"], control["completed"]) == expected
    assert control["finished"] is True
    for backend in NATIVE:
        result = results[backend]
        assert result["steps"] == 0, backend
        for key in ("seen", "extra", "child", "now", "service", "completed",
                    "finished"):
            _same(result[key], control[key], (backend, derived, key))


def _completion_overflow(backend, limit, derived):
    """R12-1: a 4,096-completion split interrupts a trace-retained snapshot."""
    count = 4097
    engine = Engine([Resource("r", float(count))], backend=backend,
                    trace_limit=limit)
    token = _Token()
    roots = [engine.submit(f"root{index}", {"r": 1.0},
                           metadata={"token": token}
                           if index == 4096 - limit else None)
             for index in range(count)]
    trigger = 4096 - limit
    child = engine.submit("child", {"r": 1.0}, after=(roots[trigger],))
    seen, extra = [], []

    def release():
        seen.append((engine.now, engine.completed_count, roots[-1].done,
                     child.started, child.done, engine.serviced["r"]))
        if derived:
            work = 1.0 if child.started is None else 2.0
            extra.append((work, engine.submit(
                "extra", {"r": work}, on_complete=lambda _event: None)))

    finalizer = weakref.finalize(token, release)
    del token
    engine.advance()
    result = {
        "seen": list(seen), "extra": [(work, event.started, event.time)
                                      for work, event in extra],
        "child": (child.started, child.time), "now": engine.now,
        "service": engine.serviced["r"], "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "finished": not finalizer.alive,
    }
    _close(engine)
    return result


@pytest.mark.parametrize("limit", (1, 2, 3))
@pytest.mark.parametrize("derived", (False, True))
def test_r12_1_completion_overflow_with_trace_retention(limit, derived):
    """Observation and derived work match Python across every trace limit."""
    results = {backend: _completion_overflow(backend, limit, derived)
               for backend in BACKENDS}
    control = results["python"]
    expected = ([(1.0, 4097, True, None, False, 4097.0)],
                ([(1.0, 1.0, 1.0004881620698072)] if derived else []),
                (1.0, 1.0004881620698072 if derived else 1.0002440810349036),
                1.0004881620698072 if derived else 1.0002440810349036,
                4099.0 if derived else 4098.0, 4099 if derived else 4098)
    assert (control["seen"], control["extra"], control["child"], control["now"],
            control["service"], control["completed"]) == expected
    assert control["finished"] is True
    for backend in NATIVE:
        result = results[backend]
        assert result["steps"] == 0, (backend, limit, derived)
        for key in ("seen", "extra", "child", "now", "service", "completed",
                    "finished"):
            _same(result[key], control[key], (backend, limit, derived, key))


def _service_overflow(backend, limit, derived):
    """R12-1: a 32,768-service split interrupts a trace-retained snapshot."""
    count, nres = 1025, 32
    names = [f"r{index}" for index in range(nres)]
    work = dict.fromkeys(names, 1.0)
    engine = Engine([Resource(name, float(count)) for name in names],
                    backend=backend, trace_limit=limit)
    token = _Token()
    trigger = 1024 - limit
    roots = [engine.submit(f"root{index}", work,
                           metadata={"token": token}
                           if index == trigger else None)
             for index in range(count)]
    child = engine.submit("child", work, after=(roots[trigger],))
    seen, extra = [], []

    def release():
        seen.append((engine.now, engine.completed_count, roots[-1].done,
                     child.started, child.done, engine.serviced[names[0]]))
        if derived:
            amount = 1.0 if child.started is None else 2.0
            extra.append((amount, engine.submit(
                "extra", dict.fromkeys(names, amount),
                on_complete=lambda _event: None)))

    finalizer = weakref.finalize(token, release)
    del token
    engine.advance()
    result = {
        "seen": list(seen), "extra": [(amount, event.started, event.time)
                                      for amount, event in extra],
        "child": (child.started, child.time), "now": engine.now,
        "service": [engine.serviced[name] for name in names],
        "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "finished": not finalizer.alive,
    }
    _close(engine)
    return result


@pytest.mark.parametrize("limit", (1, 2, 3))
@pytest.mark.parametrize("derived", (False, True))
def test_r12_1_service_overflow_with_trace_retention(limit, derived):
    """The per-link service buffer split publishes the trace release exactly."""
    results = {backend: _service_overflow(backend, limit, derived)
               for backend in BACKENDS}
    control = results["python"]
    assert control["seen"] == [(1.0, 1025, True, None, False, 1025.0)]
    assert control["finished"] is True
    assert control["extra"] == (
        [(1.0, 1.0, 1.0019512195121951)] if derived else [])
    assert control["child"] == (1.0, 1.0019512195121951 if derived
                                else 1.0009756097560976)
    for backend in NATIVE:
        result = results[backend]
        assert result["steps"] == 0, (backend, limit, derived)
        for key in ("seen", "extra", "child", "now", "service", "completed",
                    "finished"):
            _same(result[key], control[key], (backend, limit, derived, key))


def test_r12_2_metadata_eligibility_is_one_all_inert_exact_dict():
    """Keys and values must be recursively inert; None stays eligible."""
    engine = Engine([Resource("r", 64.0)], backend="python")
    event = engine.submit(
        "inert", {"r": 1.0},
        metadata={"layer": 0, "op": "qkv", "ratio": 0.5, "flag": True,
                  "raw": b"x", "none": None, "shape": (1, (2, 3))})
    task = engine._tasks[event.sequence]
    assert _stable_task_payloads(task, engine)
    assert _stable_task_payloads(dataclasses.replace(task, metadata=None),
                                 engine)
    reject = (
        {"token": _Token()}, {_Token(): 1}, {"nested": {"inner": 1}},
        {"payload": [1]}, {"payload": (1, bytearray(b"x"))},
        {"tags": frozenset({"a", "b"})}, {"tags": (frozenset({"a"}),)},
        _Dict({"layer": 0}),
    )
    for metadata in reject:
        assert not _stable_task_payloads(
            dataclasses.replace(task, metadata=metadata), engine), metadata
    nested = 1
    for _ in range(8):
        nested = (nested,)
    assert not _stable_task_payloads(
        dataclasses.replace(task, metadata={"deep": nested}), engine)
    nested = 1
    for _ in range(7):
        nested = (nested,)
    assert _stable_task_payloads(
        dataclasses.replace(task, metadata={"deep": nested}), engine)
    _close(engine)


def _wire_arena():
    status = accel_build.build_accel()
    assert status.get("ok"), status
    return tables.Arena(accel_bridge.load_library(status))


def _wire_case(count, nres, trigger):
    """One raw engine-arena/2 run; returns the observed call sequence."""
    arena = _wire_arena()
    try:
        arena.configure([float(count)] * nres, [1] * nres, [0] * nres)
        links = [(index, 1.0, math.inf, ()) for index in range(nres)]
        rows = [(index + 1,
                 tables.F_CALLBACK if index == trigger else 0,
                 0, 1, 0, 0.0, 0.0, 0.0, links)
                for index in range(count)]
        rows.append((count + 1, 0, 0, 1, 1, 0.0, 0.0, 0.0, links))
        slots = arena.submit(rows)
        arena.link([(slots[trigger], slots[-1], count + 1)])
        calls = []

        def run(now, horizon, budget):
            answer = arena.run(now, horizon, math.inf, budget)
            calls.append({
                "boundary": answer[1], "hold": int(arena.hdr[3]),
                "completions": answer[3], "starts": answer[4],
                "start_seqs": [start[0] for start in arena.starts(answer[4])],
            })
            return answer

        answer = run(0.0, 0.0, 0)
        while answer[1] == tables.B_OUTPUT:
            answer = run(0.0, 0.0, 0)
        answer = run(0.0, 1.0, 1)
        assert answer[1] == tables.B_OUTPUT, calls
        answer = run(1.0, 1.0, 0)
        assert (answer[1] in (tables.B_CALLBACK, tables.B_HORIZON)
                and answer[4] == 0 and answer[3] == 1
                and int(arena.hdr[3]) == 1), calls
        answer = run(1.0, 1.0, 0)
        assert int(arena.hdr[3]) == 0 and (count + 1, 1.0) in arena.starts(
            answer[4]), calls
        return calls
    finally:
        arena.close()


@pytest.mark.parametrize(("count", "nres", "trigger"), ((4097, 1, 4095),
                                                       (1025, 32, 1023)))
def test_r12_1_pending_hold_survives_the_raw_wire_split(count, nres, trigger):
    """The interrupted snapshot publishes its final row before any admission."""
    calls = _wire_case(count, nres, trigger)
    first_completion = next(index for index, call in enumerate(calls)
                            if call["completions"] > 1)
    split = calls[first_completion]
    continuation = calls[first_completion + 1]
    admission = calls[first_completion + 2]
    assert split["boundary"] == tables.B_OUTPUT and split["hold"] == 1
    assert split["completions"] == (4096 if nres == 1 else 1024)
    assert split["starts"] == 0
    # R12-1 regression: the continuation still holds admission, so the final
    # completion is published alone and the child is not started yet.  Either
    # settled boundary is acceptable; only the hold and publication matter.
    assert continuation["boundary"] in (tables.B_CALLBACK, tables.B_HORIZON)
    assert continuation["hold"] == 1
    assert continuation["completions"] == 1 and continuation["starts"] == 0
    assert admission["hold"] == 0 and admission["completions"] == 0
    assert admission["start_seqs"] == [count + 1]


def _native_hold(backend, count, nres, trigger):
    """R12-1 bridge: inert Engine tasks with one test-local wire callback flag."""
    names = [f"r{index}" for index in range(nres)]
    work = dict.fromkeys(names, 1.0)
    engine = Engine([Resource(name, float(count)) for name in names],
                    backend=backend, trace_limit=0)
    roots = [engine.submit(f"root{index}", work) for index in range(count)]
    child = engine.submit("child", work, after=(roots[trigger],))
    seen_completions = []
    original = engine._complete_native

    def traced(task, moment):
        if task.event.name in (f"root{trigger}", f"root{trigger + 1}"):
            seen_completions.append((task.event.name, child.started, moment))
        return original(task, moment)

    engine._complete_native = traced
    engine.advance()
    result = {
        "seen_completions": list(seen_completions),
        "child": (child.started, child.time), "now": engine.now,
        "service": [engine.serviced[name] for name in names],
        "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "ownership": (len(engine._tasks), len(engine._native_seqs),
                      len(engine._native_slots)),
    }
    _close(engine)
    return result


@pytest.mark.parametrize(("count", "nres", "trigger", "name"),
                         ((4097, 1, 4095, "root4095"),
                          (1025, 32, 1023, "root1023")))
def test_r12_1_native_bridge_holds_admission_before_the_child_starts(
        monkeypatch, count, nres, trigger, name):
    """A wire-flagged inert row proves the bridge publishes, then admits."""
    control = _native_hold("python", count, nres, trigger)
    original = accel_sync.task_flags

    def flagged(task, engine=None):
        flags = original(task, engine)
        return flags | tables.F_CALLBACK if task.event.name == name else flags

    monkeypatch.setattr(accel_sync, "task_flags", flagged)
    results = {backend: _native_hold(backend, count, nres, trigger)
               for backend in NATIVE}
    for backend in NATIVE:
        result = results[backend]
        assert result["steps"] > 0, backend
        assert result["seen_completions"] == [
            (name, None, 1.0), (f"root{trigger + 1}", None, 1.0)], backend
        for key in ("child", "now", "service", "completed"):
            _same(result[key], control[key], (backend, key))
        assert result["ownership"] == (0, 0, 0), backend


def test_r12_accelerator_modules_stay_below_the_reviewed_line_caps():
    """Every accelerator source stays <735 lines and this module <450."""
    runtime = ROOT / "mindie_sim" / "prediction" / "runtime"
    files = (sorted(runtime.glob("accel_*.py"))
             + sorted(runtime.glob("_engine_accel*")))
    assert files
    for path in files:
        assert len(path.read_bytes().splitlines()) < 735, path.name
    assert len(Path(__file__).read_bytes().splitlines()) < 450
