"""Round-10 closure regressions: identity, mutation, trace eviction and abort.

[Role] Reproduce the three remaining ``R9-2`` causal-finalizer gaps with public
  ``Engine`` APIs: a metaclass whose equality says every type is equal, a
  caller-aliased mutable metadata container mutated between two completed
  ``advance`` calls, and a bounded ``engine.trace`` that retains a finalizable
  object beyond its own completion instant.  Also prove that an exception raised
  while publishing an accepted native batch is never masked by a secondary
  ``stale_slot`` hand-back error.
[Invariants] Every fixture uses the unchanged production library and public
  Engine APIs.  Registration is a deterministic CPython ``weakref.finalize``
  trigger; no fabricated ``KernelFault``, allocator change, private scheduler
  mutation or second thread exists.  The Python reference runs the identical
  workload in every comparison.
[Timeouts & Locks] No threads or sleeps; native calls stay inside the bounded
  per-epoch budget and the isolated build memo.
[Side-Effect Policy] Compiles only into the isolated temporary cache; never
  writes the workspace, package tree or user cache.
"""
from __future__ import annotations

from pathlib import Path
import weakref

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime.accel_tables import (
    _is_inert_builtin, metadata_is_stable,
)

BACKENDS = ("python", "c", "auto")
NATIVE = ("c", "auto")
LIMITS = (0, 1, 2, 3, None)
WORK = [1.0, 2.0] + [3.0] * 62


class _EqualMeta(type):
    """A user metaclass whose equality claims every other type is equal."""

    calls: list = []

    def __eq__(cls, other):
        _EqualMeta.calls.append(other)
        return True

    def __hash__(cls):
        return id(cls)


class _Token(metaclass=_EqualMeta):
    """Weakrefable custom object: never an inert built-in."""


class _MyStr(str):
    pass


class _MyInt(int):
    pass


class _MyFloat(float):
    pass


def _close(engine) -> None:
    if engine._accel is not None:
        engine._accel.close()


def _same(left, right, label):
    assert left == right, (label, left, right)


def test_r10_3_identity_types_never_dispatch_user_equality():
    """``type(obj) is X`` only: metaclass equality is never consulted."""
    _EqualMeta.calls.clear()
    assert _is_inert_builtin(_Token()) is False
    assert _EqualMeta.calls == []
    assert not _is_inert_builtin(_MyStr("x"))
    assert not _is_inert_builtin(_MyInt(1))
    assert not _is_inert_builtin(_MyFloat(1.5))
    assert _EqualMeta.calls == []
    assert _is_inert_builtin((1, (2.5, (3, 4))))
    assert not _is_inert_builtin((1, (2.5, frozenset({3, 4}))))
    assert not _is_inert_builtin([1, 2])
    assert not _is_inert_builtin({"a": 1})
    # A non-inert *leaf* is an allowed metadata value (it becomes F_CALLBACK);
    # a nested mutable container is not accepted on the native path.
    assert metadata_is_stable({"token": _Token()})
    assert metadata_is_stable({"layer": 0, "op": "qkv", "shape": (1, 128)})
    assert not metadata_is_stable({"payload": []})
    assert not metadata_is_stable({"payload": {"inner": 1}})
    assert not metadata_is_stable({"payload": (1, bytearray(b"x"))})
    assert not metadata_is_stable({_Token(): 1})


def _metaclass_lifetime(backend):
    """R10-3: the token finalizer must run at its own completion instant."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    token = _Token()
    roots = [engine.submit(f"root{index}", {"r": amount}, caps={"r": 1.0},
                           metadata={"token": token} if index == 0 else None)
             for index, amount in enumerate(WORK)]
    seen = []
    finalizer = weakref.finalize(
        token, lambda: seen.append((engine.now, engine.serviced["r"])))
    del token
    engine.advance()
    assert not finalizer.alive
    result = {
        "seen": list(seen), "now": engine.now,
        "busy": engine.busy_ms["r"], "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "roots": [(event.started, event.time) for event in roots],
    }
    _close(engine)
    return result


def test_r10_3_metaclass_token_finalizes_at_its_completion_instant():
    """A weakrefable custom object is a callback boundary in python/c/auto."""
    results = {backend: _metaclass_lifetime(backend) for backend in BACKENDS}
    control = results["python"]
    assert control["seen"] == [(1.0, 64.0)]
    assert control["now"] == 3.0
    assert control["busy"] == 3.0
    assert control["service"] == 189.0
    assert control["completed"] == 64
    assert control["roots"] == [(0.0, 1.0)] + [(0.0, 2.0)] + [(0.0, 3.0)] * 62
    for backend in NATIVE:
        result = results[backend]
        # R12-2: finalizer-bearing metadata declines native before integration.
        assert result["steps"] == 0, backend
        for key in ("seen", "now", "busy", "service", "completed", "roots"):
            _same(result[key], control[key], (backend, key))


def _mutable_metadata(backend):
    """R10-2: admission, caller mutation, then a second completed advance."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    payload = []
    roots = [engine.submit(f"root{index}", {"r": amount}, caps={"r": 1.0},
                           metadata={"payload": payload} if index == 0 else None)
             for index, amount in enumerate(WORK)]
    engine.advance(horizon=0.0)
    token = _Token()
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        extra.append(engine.submit(
            "extra", {"r": engine.serviced["r"]},
            on_complete=lambda _event: None))

    def register():
        registration.append((engine.now, roots[1].done))
        engine.on_complete(roots[1], follow)

    finalizer = weakref.finalize(token, register)
    payload.append(token)
    del token, payload
    error = None
    try:
        engine.advance()
    except Exception as exc:  # noqa: BLE001 - the contract is "no exception"
        error = (type(exc).__name__, str(exc))
    result = {
        "error": error, "registration": list(registration),
        "observed": list(observed),
        "extra": [(event.started, event.time) for event in extra],
        "now": engine.now, "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "finished": not finalizer.alive,
    }
    _close(engine)
    return result


def test_r10_2_mutable_metadata_stays_on_python_and_matches_exactly():
    """A nested mutable container declines native execution before ``run``."""
    results = {backend: _mutable_metadata(backend) for backend in BACKENDS}
    control = results["python"]
    assert control["error"] is None
    assert control["registration"] == [(1.0, False)]
    assert control["observed"] == [(2.0, 2.0, 2.0, 127.0)]
    assert control["extra"] == [(2.0, 4.953125)]
    assert control["now"] == 4.953125
    assert control["service"] == 316.0
    assert control["completed"] == 65
    assert control["finished"] is True
    for backend in NATIVE:
        result = results[backend]
        assert result["steps"] == 0, (backend, result["steps"])
        for key in ("error", "registration", "observed", "extra", "now",
                    "service", "completed", "finished"):
            _same(result[key], control[key], (backend, key))


def _flat_metadata(backend):
    """An exact scalar/tuple metadata dict still takes the multi-step path."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    roots = [engine.submit(
        f"root{index}", {"r": amount}, caps={"r": 1.0},
        metadata={"layer": 0, "op": "qkv", "shape": (1, 128)})
        for index, amount in enumerate(WORK)]
    engine.advance()
    result = {
        "now": engine.now, "busy": engine.busy_ms["r"],
        "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "roots": [(event.started, event.time) for event in roots],
        "steps": engine.accel_status()["native_steps"],
    }
    _close(engine)
    return result


def test_r10_2_flat_immutable_metadata_keeps_native_fast_path():
    """The R10-2 decline does not slow ordinary flat metadata dicts."""
    results = {backend: _flat_metadata(backend) for backend in BACKENDS}
    control = results["python"]
    assert control["now"] == 3.0
    assert control["busy"] == 3.0
    assert control["service"] == 189.0
    assert control["completed"] == 64
    for backend in NATIVE:
        result = results[backend]
        assert result["steps"] > 0, backend
        for key in ("now", "busy", "service", "completed", "roots"):
            _same(result[key], control[key], (backend, key))


def test_r10_2_aborted_publication_preserves_the_primary_exception(monkeypatch):
    """A raising native publication never becomes a secondary stale export.

    R12-2 keeps finalizer-bearing metadata on Python, so this fixture drives
    the exact ``_apply`` abort path with a test-local completion observer.  The
    second observer registers a public callback on a still-live mirrored row and
    then raises, exercising the deferred hand-back and primary-error contract.
    """
    engine = Engine([Resource("r", 128.0)], backend="c", trace_limit=0)
    first = engine.submit("root0", {"r": 1.0}, caps={"r": 1.0})
    second = engine.submit("root1", {"r": 1.0}, caps={"r": 1.0})
    roots = [engine.submit(f"root{index}", {"r": 1.0}, caps={"r": 1.0})
             for index in range(2, 10)]
    tail = engine.submit("tail", {"r": 2.0}, caps={"r": 1.0})
    original = engine._complete_native

    def boom(task, moment):
        if task.event.name == "root1":
            engine.on_complete(roots[-1], lambda _event: None)
            raise RuntimeError("primary-callback-boom")
        return original(task, moment)

    monkeypatch.setattr(engine, "_complete_native", boom)
    with pytest.raises(RuntimeError, match="primary-callback-boom") as caught:
        engine.advance()
    assert caught.value.__context__ is None
    assert engine._accel.aborted_publication is True
    assert engine._accel.pending_handback is True
    assert engine.completed_count == 1
    assert len(engine._native_slots) == 9
    assert first.time == 1.0
    assert second.time is None
    assert tail.time is None
    assert roots[-1].time is None
    _close(engine)


def _trace_lifetime(backend, limit):
    """R10-1: a finalizer retained by ``engine.trace`` must evict causally."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=limit)
    token = _Token()
    roots = [engine.submit(f"root{index}", {"r": amount}, caps={"r": 1.0},
                           metadata={"token": token} if index == 0 else None)
             for index, amount in enumerate(WORK)]
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        extra.append(engine.submit(
            "extra", {"r": engine.serviced["r"]},
            on_complete=lambda _event: None))

    def register():
        registration.append((engine.now, roots[1].done))
        engine.on_complete(roots[1], follow)

    finalizer = weakref.finalize(token, register)
    del token
    error = None
    try:
        engine.advance()
    except Exception as exc:  # noqa: BLE001 - equality below must be exact
        error = (type(exc).__name__, str(exc))
    status = engine.accel_status()
    result = {
        "error": error, "registration": list(registration),
        "observed": list(observed),
        "extra": [(event.started, event.time) for event in extra],
        "now": engine.now, "service": engine.serviced["r"],
        "busy": engine.busy_ms["r"], "completed": engine.completed_count,
        "steps": status["native_steps"],
        "trace": [row["task"] for row in engine.trace],
        "finished": not finalizer.alive,
    }
    _close(engine)
    return result


_TRACE_EXPECTED = {
    0: ([(1.0, False)], [(2.0, 2.0, 2.0, 127.0)], [(2.0, 4.953125)],
        4.953125, 316.0, []),
    1: ([(2.0, True)], [(2.0, 2.0, 2.0, 127.0)], [(2.0, 4.953125)],
        4.953125, 316.0, ["extra"]),
    2: ([(3.0, True)], [(2.0, 3.0, 3.0, 189.0)], [(3.0, 5.953125)],
        5.953125, 378.0, ["root63", "extra"]),
    3: ([(3.0, True)], [(2.0, 3.0, 3.0, 189.0)], [(3.0, 5.953125)],
        5.953125, 378.0, ["root62", "root63", "extra"]),
    None: ([(1.0, False)], [(2.0, 2.0, 2.0, 127.0)], [(2.0, 4.953125)],
           4.953125, 316.0, []),
}


@pytest.mark.parametrize("limit", LIMITS)
def test_r10_1_bounded_trace_eviction_is_causally_exact(limit):
    """Every bounded/unbounded trace limit matches Python in c and auto."""
    results = {backend: _trace_lifetime(backend, limit) for backend in BACKENDS}
    control = results["python"]
    registration, observed, extra, now, service, trace = _TRACE_EXPECTED[limit]
    assert control["error"] is None
    assert control["registration"] == registration
    assert control["observed"] == observed
    assert control["extra"] == extra
    assert control["now"] == now
    assert control["busy"] == now
    assert control["service"] == service
    assert control["completed"] == 65
    assert control["trace"] == trace
    for backend in NATIVE:
        result = results[backend]
        # R12-2: finalizer-bearing metadata declines native before integration.
        assert result["steps"] == 0, (backend, limit)
        for key in ("error", "registration", "observed", "extra", "now",
                    "service", "busy", "completed", "trace", "finished"):
            _same(result[key], control[key], (backend, limit, key))


def test_r10_accelerator_modules_stay_below_the_reviewed_line_caps():
    """R10 keeps every accelerator module <735 lines and this test <450."""
    runtime = (Path(__file__).resolve().parents[1]
               / "mindie_sim" / "prediction" / "runtime")
    files = (sorted(runtime.glob("accel_*.py"))
             + sorted(runtime.glob("_engine_accel*")))
    assert files
    for path in files:
        assert len(path.read_bytes().splitlines()) < 735, path.name
    assert len(Path(__file__).read_bytes().splitlines()) < 450
