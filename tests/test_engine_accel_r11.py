"""Round-11 closure regressions: field ownership and causal ordering.

[Role] Reproduce ``R11-1`` (release/latency/coupled/auxiliary payloads and the
  fail-closed metadata depth budget) and ``R11-2`` (a callback-capable
  completion published before its awakened successor starts) with public
  ``Engine`` APIs in all three backends, plus the installed-wheel path.
[Invariants] The production kernel is unchanged; registration is a
  deterministic CPython ``weakref.finalize`` trigger.  Native results are
  compared field-by-field with Python, never inferred from the final clock
  alone.  The wheel test builds, verifies and installs a real PEP 517 artefact.
[Timeouts & Locks] No threads or sleeps; native calls stay inside the bounded
  per-epoch budget and the isolated build memo.
[Side-Effect Policy] Compiles only into a temporary cache and writes only
  test-owned temporary trees; never writes the workspace or user cache.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import weakref

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime.accel_tables import (
    _stable_task_payloads, metadata_is_stable,
)

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("python", "c", "auto")
NATIVE = ("c", "auto")
WORK = [1.0, 2.0] + [3.0] * 62


class _Float(float):
    pass


class _Int(int):
    pass


class _Str(str):
    pass


class _Token:
    pass


class _Pool(str):
    pass


def _close(engine):
    if engine._accel is not None:
        engine._accel.close()


def _same(left, right, label):
    assert left == right, (label, left, right)


def _roots(engine, **first):
    """The review's 64-root capped frontier with an optional root0 payload."""
    roots = [engine.submit("root0", {"r": 1.0}, caps={"r": 1.0}, **first)]
    roots += [engine.submit(f"root{i}", {"r": amount}, caps={"r": 1.0})
              for i, amount in enumerate(WORK[1:], 1)]
    return roots


def _observer(engine, roots, target):
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        extra.append(engine.submit("extra", {"r": engine.serviced["r"]},
                                   on_complete=lambda _event: None))

    def register():
        registration.append((engine.now, roots[target].done))
        engine.on_complete(roots[target], follow)
    return register, registration, observed, extra


def _omitted(backend, field, target):
    """R11-1: a scalar subclass in release/latency must decline native."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    payload = _Float(1.0 if field == "latency" else 0.0)
    roots = _roots(engine, **{field: payload})
    register, registration, observed, extra = _observer(engine, roots, target)
    finalizer = weakref.finalize(payload, register)
    del payload
    error = None
    try:
        engine.advance()
    except Exception as exc:  # noqa: BLE001 - equality below must be exact
        error = (type(exc).__name__, str(exc))
    result = {
        "error": error, "registration": registration, "observed": observed,
        "extra": [(event.started, event.time) for event in extra],
        "now": engine.now, "busy": engine.busy_ms["r"],
        "service": engine.serviced["r"], "completed": engine.completed_count,
        "steps": engine.accel_status()["native_steps"],
        "finished": not finalizer.alive,
    }
    _close(engine)
    return result


@pytest.mark.parametrize("field", ("release", "latency"))
def test_r11_1_numeric_subclasses_decline_native(field):
    """Both omitted scalar fields match Python for root1 and done root0."""
    rows = {(target, backend): _omitted(backend, field, target)
            for target in (1, 0) for backend in BACKENDS}
    for target in (1, 0):
        control = rows[target, "python"]
        expected = ([(1.0, False)], [(2.0, 2.0, 2.0, 127.0)],
                    [(2.0, 4.953125)], 4.953125, 316.0) if target == 1 else (
                    [(1.0, True)], [(1.0, 1.0, 1.0, 64.0)],
                    [(2.0, 3.96875)], 3.96875, 253.0)
        assert control["error"] is None and control["steps"] == 0
        assert (control["registration"], control["observed"], control["extra"],
                control["now"], control["service"]) == expected
        assert control["completed"] == 65 and control["finished"] is True
        for backend in NATIVE:
            result = rows[target, backend]
            for key in ("error", "registration", "observed", "extra", "now",
                        "busy", "service", "completed", "finished"):
                _same(result[key], control[key], (field, target, backend, key))
            assert result["steps"] == 0, (field, target, backend)


def _auxiliary(backend, field):
    """R11-1: an unused retained payload may still own a caller finalizer."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    payload = _Pool("p") if field == "memory_pools" else _Token()
    first = ({"memory_bindings": {"unused": {"owner": payload}}}
             if field == "memory_bindings"
             else {"memory_pools": frozenset({payload})})
    roots = _roots(engine, **first)
    del first
    register, registration, observed, extra = _observer(engine, roots, 1)
    finalizer = weakref.finalize(payload, register)
    del payload
    error = None
    try:
        engine.advance()
    except Exception as exc:  # noqa: BLE001
        error = (type(exc).__name__, str(exc))
    result = {"error": error, "registration": registration,
              "observed": observed,
              "extra": [(event.started, event.time) for event in extra],
              "now": engine.now, "service": engine.serviced["r"],
              "completed": engine.completed_count,
              "steps": engine.accel_status()["native_steps"],
              "finished": not finalizer.alive}
    _close(engine)
    return result


@pytest.mark.parametrize("field", ("memory_bindings", "memory_pools"))
def test_r11_1_auxiliary_payloads_decline_native(field):
    results = {backend: _auxiliary(backend, field) for backend in BACKENDS}
    control = results["python"]
    assert control["error"] is None
    assert control["registration"] == [(1.0, False)]
    assert control["observed"] == [(2.0, 2.0, 2.0, 127.0)]
    assert control["extra"] == [(2.0, 4.953125)]
    assert (control["now"], control["service"], control["completed"]) == (
        4.953125, 316.0, 65)
    for backend in NATIVE:
        result = results[backend]
        for key in ("error", "registration", "observed", "extra", "now",
                    "service", "completed", "finished"):
            _same(result[key], control[key], (backend, field, key))
        assert result["steps"] == 0, (backend, field)


def _coupled(backend):
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    seen = []

    class _CoupledInt(int):
        def __del__(self):
            seen.append((engine.now, engine.serviced["r"],
                         engine.completed_count))

    roots = [engine.submit("root0", {"r": 1.0}, caps={"r": 1.0},
                           coupled=_CoupledInt(1))]
    roots += [engine.submit(f"root{i}", {"r": amount}, caps={"r": 1.0})
              for i, amount in enumerate(WORK[1:], 1)]
    engine.advance()
    result = {"seen": list(seen), "now": engine.now,
              "service": engine.serviced["r"],
              "completed": engine.completed_count,
              "steps": engine.accel_status()["native_steps"]}
    _close(engine)
    return result


def test_r11_1_coupled_int_subclass_declines_native():
    results = {backend: _coupled(backend) for backend in BACKENDS}
    control = results["python"]
    assert control["seen"] == [(1.0, 64.0, 1)]
    assert (control["now"], control["service"], control["completed"]) == (
        3.0, 189.0, 64)
    for backend in NATIVE:
        result = results[backend]
        for key in ("seen", "now", "service", "completed"):
            _same(result[key], control[key], (backend, key))
        assert result["steps"] == 0, backend


def _depth(backend, kind):
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    payload = {"list": list, "dict": dict, "set": set,
               "bytearray": bytearray}[kind]()
    nested = payload
    for _ in range(9):
        nested = (nested,)
    stable = metadata_is_stable({"payload": nested})
    roots = _roots(engine, metadata={"payload": nested})
    del nested
    engine.advance(horizon=0.0)
    register, registration, observed, extra = _observer(engine, roots, 1)
    if kind == "bytearray":
        payload.extend(b"changed")
    else:
        token = _Token()
        finalizer = weakref.finalize(token, register)
        if kind == "list":
            payload.append(token)
        elif kind == "dict":
            payload["token"] = token
        else:
            payload.add(token)
        del token, finalizer
    del payload
    engine.advance()
    result = {"stable": stable, "registration": registration,
              "observed": observed,
              "extra": [(event.started, event.time) for event in extra],
              "now": engine.now, "service": engine.serviced["r"],
              "completed": engine.completed_count,
              "steps": engine.accel_status()["native_steps"]}
    _close(engine)
    return result


@pytest.mark.parametrize("kind", ("list", "dict", "set", "bytearray"))
def test_r11_depth_budget_fails_closed_for_nine_wrappers(kind):
    """A depth-budget exhaustion is unproven: it must decline, not accept."""
    results = {backend: _depth(backend, kind) for backend in BACKENDS}
    control = results["python"]
    assert control["stable"] is False
    for backend in NATIVE:
        result = results[backend]
        assert result["stable"] is False and result["steps"] == 0
        for key in ("registration", "observed", "extra", "now", "service",
                    "completed"):
            _same(result[key], control[key], (backend, kind, key))


def test_r11_1_every_retained_task_field_is_classified():
    """The ownership audit covers the complete ``_Task`` slot set."""
    engine = Engine([Resource("r", 64.0)], backend="python")
    cases = {
        "event.name": ("name", _Str("root")),
        "owner": ("owner", _Str("owner")),
        "demand key": ("work", {_Str("r"): 1.0}),
        "caps key": ("caps", {_Str("r"): 1.0}),
        "caps value": ("caps", {"r": _Float(1.0)}),
        "placement key": ("placement", {_Str("r"): (0,)}),
        "placement unit": ("placement", {"r": (_Int(0),)}),
        "release": ("release", _Float(0.0)),
        "latency": ("latency", _Float(1.0)),
        "coupled": ("coupled", _Int(1)),
        "priority": ("priority", _Int(1)),
        "cohort": ("cohort_size", _Int(2)),
        "metadata key": ("metadata", {_Str("k"): 1}),
        "memory ledger": ("memory", object()),
        "memory bindings": ("memory_bindings", {"unused": {"owner": _Token()}}),
        "memory pools": ("memory_pools", frozenset({_Pool("p")})),
    }
    for label, (field, value) in cases.items():
        kwargs = ({"name": value} if field == "name" else
                  {"name": "root0", field: value})
        name, work = kwargs.pop("name"), kwargs.pop("work", {"r": 1.0})
        kwargs.setdefault("caps", {"r": 1.0})
        event = engine.submit(name, work, **kwargs)
        assert not _stable_task_payloads(engine._tasks[event.sequence]), label
    event = engine.submit("ok", {"r": 1.0}, caps={"r": 1.0})
    assert _stable_task_payloads(engine._tasks[event.sequence])
    _close(engine)


def _descendant(backend, derived):
    """R11-2: root0's finalizer must see the awakened child unstarted."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    token = _Token()
    root = engine.submit("root0", {"r": 1.0}, caps={"r": 1.0},
                         metadata={"token": token})
    child = engine.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(root,))
    roots = [root] + [engine.submit(f"root{i}", {"r": amount},
                                    caps={"r": 1.0})
                      for i, amount in enumerate(WORK[1:], 1)]
    seen, extra = [], []

    def release():
        seen.append((engine.now, root.done, child.started, child.done,
                     engine.completed_count, engine.serviced["r"]))
        if derived:
            extra.append(engine.submit(
                "extra", {"r": 1.0 if child.started is None else 2.0},
                on_complete=lambda _event: None))

    finalizer = weakref.finalize(token, release)
    del token
    engine.advance()
    result = {"seen": seen,
              "extra": [(event.started, event.time) for event in extra],
              "child": (child.started, child.time), "now": engine.now,
              "service": engine.serviced["r"],
              "completed": engine.completed_count,
              "steps": engine.accel_status()["native_steps"],
              "finished": not finalizer.alive}
    _close(engine)
    return result


@pytest.mark.parametrize("derived", (False, True))
def test_r11_2_descendant_start_is_published_after_the_completion(derived):
    """Observation-only and derived-work variants match Python exactly."""
    results = {backend: _descendant(backend, derived) for backend in BACKENDS}
    control = results["python"]
    expected = ([(1.0, True, None, False, 1, 64.0)],
                [(1.0, 2.015625)] if derived else [],
                (1.0, 2.015625) if derived else (1.0, 2.0),
                3.015625 if derived else 3.0, 191.0 if derived else 190.0,
                66 if derived else 65)
    assert control["seen"] == expected[0]
    assert (control["extra"], control["child"], control["now"],
            control["service"], control["completed"]) == expected[1:]
    for backend in NATIVE:
        result = results[backend]
        # R12-2: finalizer-bearing metadata declines native before integration.
        assert result["steps"] == 0, backend
        for key in ("seen", "extra", "child", "now", "service", "completed",
                    "finished"):
            _same(result[key], control[key], (backend, derived, key))


def _zero_chain(backend):
    """R11-2: a zero-time chained callback hands back at every completion."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    seen, handles, tokens = [], [], []

    def metadata(label, index):
        token = _Token()

        def record():
            successor = None if index is None else handles[index]
            seen.append((label, engine.now, engine.completed_count,
                         engine.serviced["r"],
                         None if successor is None else successor.started))
        tokens.append(weakref.finalize(token, record))
        return {"token": token}

    root = engine.submit("root0", {}, metadata=metadata("root", 1))
    child = engine.submit("child", {}, after=(root,),
                          metadata=metadata("child", 2))
    grand = engine.submit("grand", {"r": 1.0}, after=(child,),
                          metadata=metadata("grand", None))
    handles.extend([root, child, grand])
    engine.advance()
    result = {"seen": list(seen), "now": engine.now,
              "service": engine.serviced["r"],
              "completed": engine.completed_count,
              "events": [(root.started, root.time), (child.started, child.time),
                         (grand.started, grand.time)],
              "steps": engine.accel_status()["native_steps"]}
    _close(engine)
    return result


def test_r11_2_zero_latency_chain_is_causally_ordered():
    results = {backend: _zero_chain(backend) for backend in BACKENDS}
    control = results["python"]
    assert control["seen"] == [
        ("root", 0.0, 1, 0.0, None), ("child", 0.0, 2, 0.0, None),
        ("grand", 0.015625, 3, 1.0, None)]
    assert (control["now"], control["service"], control["completed"]) == (
        0.015625, 1.0, 3)
    for backend in NATIVE:
        for key in ("seen", "now", "service", "completed", "events"):
            _same(results[backend][key], control[key], (backend, key))


def _zero_horizon(backend):
    """R11-2: same-instant admission also runs when the horizon is reached."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    token = _Token()
    root = engine.submit("root0", {}, metadata={"token": token})
    child = engine.submit("child", {"r": 1.0}, after=(root,))
    seen = []
    finalizer = weakref.finalize(
        token, lambda: seen.append((engine.now, child.started, root.done)))
    del token
    engine.advance(horizon=0.0)
    result = {"seen": seen, "child": (child.started, child.time),
              "now": engine.now, "steps": engine.accel_status()["native_steps"]}
    _close(engine)
    return result


def test_r11_2_zero_horizon_admission_is_after_the_completion():
    results = {backend: _zero_horizon(backend) for backend in BACKENDS}
    control = results["python"]
    assert control["seen"] == [(0.0, None, True)]
    assert control["child"] == (0.0, None) and control["now"] == 0.0
    for backend in NATIVE:
        for key in ("seen", "child", "now"):
            _same(results[backend][key], control[key], (backend, key))


def test_r11_accelerator_modules_stay_below_the_reviewed_line_caps():
    """R11 keeps every accelerator module <735 lines and this test <450."""
    runtime = ROOT / "mindie_sim" / "prediction" / "runtime"
    files = (sorted(runtime.glob("accel_*.py"))
             + sorted(runtime.glob("_engine_accel*")))
    assert files
    for path in files:
        assert len(path.read_bytes().splitlines()) < 735, path.name
    assert len(Path(__file__).read_bytes().splitlines()) < 450
