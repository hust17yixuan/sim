"""Round-15 closure: caller frozensets decline; retained sentinels stay safe.

[Role] Prove that the inert built-in whitelist admits only non-weakrefable
  exact types, that ``memory_pools`` is accepted only through an Engine-retained
  sentinel, and that every caller-owned frozenset declines to the unchanged
  Python reference before native integration.
[Invariants] A caller-supplied exact ``frozenset`` -- metadata key, value,
  nested tuple member, trace record or ``memory_pools`` -- is never certified
  inert, including a finalizer registered after ``advance(horizon=0.0)``.  The
  caller-owned cases take zero native steps.  Sentinel classification depends
  on attachment phase; strong retention preserves exact Python parity and
  empty ownership in ``python``, ``c`` and ``auto``.
[Timeouts & Locks] No threads, sleeps or I/O; bounded native calls only.
[Side-Effect Policy] Compiles only into the isolated test cache and writes only
  test-owned temporary state.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import weakref

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime import accel_tables as tables
from mindie_sim.prediction.runtime.engine import (
    _EMPTY_FROZENSET as ENGINE_EMPTY_FROZENSET,
)

BACKENDS = ("python", "c", "auto")
NATIVE = ("c", "auto")
ROOT = Path(__file__).resolve().parents[1]

_PLAIN_TRACE = {
    "task": "plain", "owner": "", "start_ms": 0.0, "end_ms": 1.0,
    "demand": {"r": 1.0}, "serviced": {"r": 1.0}, "metadata": {},
    "placement": {}, "coupled": False,
}

_SHAPES = (
    frozenset(),
    frozenset({"p"}),
    frozenset({1, 2.5}),
    (frozenset({"p"}),),
    ((1, (frozenset({"p"}),)),),
    (1, (2.0, (frozenset(),))),
)

_FROZEN_METADATA = (
    {"payload": frozenset({"p"})},
    {"payload": frozenset()},
    {frozenset({"p"}): 1},
    {frozenset(): 1},
    {(frozenset({"p"}),): 1},
    {"payload": (1, (frozenset({"p"}),))},
    {"payload": ((frozenset(),), 2.0)},
    {"payload": (1, 2.0, frozenset({"a", "b"}))},
)

def _close(engine) -> None:
    if engine._accel is not None:
        engine._accel.close()

def _same(left, right, label) -> None:
    assert left == right, (label, left, right)

def _engine(backend="python", trace_limit=0):
    return Engine([Resource("r", 64.0)], backend=backend,
                  trace_limit=trace_limit)

class _Str(str):
    """An exact-``str`` subclass: never an inert pool name."""

def _task(engine, **kwargs):
    event = engine.submit("plain", {"r": 1.0}, **kwargs)
    return engine._tasks[event.sequence]

def test_r13_1_whitelisted_builtins_are_not_weakrefable():
    """The whitelist has ``tp_weaklistoffset == 0``; frozenset does not."""
    for value in (None, True, 1, 1.0, "x", b"x", (), (1, (2.0, None))):
        with pytest.raises(TypeError):
            weakref.ref(value)
    for value in (frozenset(), frozenset({"p"})):
        assert weakref.ref(value)() is value
        finalizer = weakref.finalize(value, lambda: None)
        assert weakref.getweakrefcount(value) == 1
        finalizer.detach()
        assert weakref.getweakrefcount(value) == 0

def test_r13_1_inert_builtin_excludes_every_exact_frozenset():
    """Only None/bool/int/float/str/bytes and exact tuple (<8) are inert."""
    for value in (None, True, 1, 1.0, "x", b"x", (), (1, (2.0, None))):
        assert tables._is_inert_builtin(value), value
    for value in _SHAPES:
        assert not tables._is_inert_builtin(value), value
    assert tables._is_inert_builtin((1, (2.0, (3,))))
    deep = 1
    for _ in range(7):
        deep = (deep,)
    assert tables._is_inert_builtin(deep)
    assert not tables._is_inert_builtin((deep,))
    # Exact type identity: a frozenset subclass is declined too.
    class FrozenSub(frozenset):
        pass

    assert not tables._is_inert_builtin(FrozenSub({"p"}))

@pytest.mark.parametrize("metadata", _FROZEN_METADATA)
def test_r13_1_frozenset_metadata_is_never_eligible(metadata):
    """Keys, values and nested wrappers all decline before integration."""
    engine = _engine()
    task = _task(engine)
    assert task.memory_pools is ENGINE_EMPTY_FROZENSET
    assert tables._stable_task_payloads(task)
    assert tables._stable_task_payloads(task, engine)
    assert tables.pure_task(engine, task)
    assert not tables.callback_capable(task, engine)
    changed = dataclasses.replace(task, metadata=metadata)
    assert not tables._eligible_metadata(metadata)
    assert not tables._stable_task_payloads(changed)
    assert not tables._stable_task_payloads(changed, engine)
    assert not tables.pure_task(engine, changed)
    assert tables.callback_capable(changed, engine)
    assert tables.task_flags(changed, engine) & tables.F_CALLBACK
    record = dict(_PLAIN_TRACE, metadata=metadata)
    assert not tables.is_inert_trace_entry(record)
    _close(engine)

def test_r13_1_inert_metadata_and_trace_records_stay_eligible():
    """The accepted whitelist still covers ordinary inert records."""
    engine = _engine()
    task = _task(engine)
    metadata = {"layer": 0, "op": "qkv", "ratio": 0.5, "flag": True,
                "raw": b"x", "none": None, "shape": (1, (2, 3))}
    changed = dataclasses.replace(task, metadata=metadata)
    assert tables._eligible_metadata(metadata)
    assert tables._stable_task_payloads(changed, engine)
    assert tables.pure_task(engine, changed)
    assert not tables.callback_capable(changed, engine)
    assert tables.is_inert_trace_entry(dict(_PLAIN_TRACE, metadata=metadata))
    _close(engine)

def test_r13_1_pool_classifier_accepts_only_engine_retained_sentinels():
    """``_inert_pool_set`` proves identity, str members and zero weakrefs."""
    engine = _engine()
    frozen = engine._memory_pools_frozen
    assert tables._inert_pool_set(ENGINE_EMPTY_FROZENSET)
    assert tables._inert_pool_set(ENGINE_EMPTY_FROZENSET, engine)
    assert tables._inert_pool_set(frozen, engine)
    assert not tables._inert_pool_set(frozen)
    assert not tables._inert_pool_set(frozenset(), engine)
    assert not tables._inert_pool_set(frozenset({"p"}), engine)
    assert not tables._inert_pool_set(frozenset({"p"}))
    assert not tables._inert_pool_set(("p",), engine)
    assert not tables._inert_pool_set(frozenset({1}), engine)
    assert not tables._inert_pool_set(frozenset({_Str("p")}), engine)
    # Attachment changes classification, not already-mirrored admission.
    finalizer = weakref.finalize(ENGINE_EMPTY_FROZENSET, lambda: None)
    try:
        assert not tables._inert_pool_set(ENGINE_EMPTY_FROZENSET)
        assert not tables._inert_pool_set(ENGINE_EMPTY_FROZENSET, engine)
    finally:
        finalizer.detach()
    assert tables._inert_pool_set(ENGINE_EMPTY_FROZENSET)
    _close(engine)

def test_r13_1_engine_default_pools_stay_eligible():
    """Bare and Engine-snapshot tasks keep their proven native eligibility."""
    engine = _engine()
    plain = _task(engine)
    assert plain.memory_pools is ENGINE_EMPTY_FROZENSET
    assert tables._stable_task_payloads(plain)
    assert tables._stable_task_payloads(plain, engine)
    assert tables.pure_task(engine, plain)
    assert not tables.callback_capable(plain)
    assert not tables.callback_capable(plain, engine)
    snapshot = _task(engine, metadata={"layer": 0})
    assert snapshot.memory_pools is engine._memory_pools_frozen
    assert not tables._stable_task_payloads(snapshot)
    assert tables._stable_task_payloads(snapshot, engine)
    assert tables.pure_task(engine, snapshot)
    # A real bind_memory publishes a fresh retained snapshot.
    engine.bind_memory(object(), pools=("hbm",))
    assert tables._inert_pool_set(engine._memory_pools_frozen, engine)
    assert not tables._inert_pool_set(engine._memory_pools_frozen)
    assert not tables._inert_pool_set(frozenset({"hbm"}), engine)
    _close(engine)

@pytest.mark.parametrize("value", (frozenset({"p"}), frozenset()))
def test_r13_1_caller_memory_pools_decline_before_integration(value):
    """``Engine.submit`` keeps the caller's exact identity; it is unproven."""
    engine = _engine()
    task = _task(engine, memory_pools=value)
    assert task.memory_pools is value
    assert not tables._stable_task_payloads(task)
    assert not tables._stable_task_payloads(task, engine)
    assert not tables.pure_task(engine, task)
    assert tables.callback_capable(task)
    assert tables.callback_capable(task, engine)
    assert tables.task_flags(task, engine) & tables.F_CALLBACK
    _close(engine)

def _last_consumer(backend, kind, nonempty, late, derived):
    """The 64-root last-consumer witness with one exact frozenset payload."""
    engine = _engine(backend)
    for index, amount in enumerate([2.0] + [3.0] * 62):
        engine.submit(f"root{index}", {"r": amount}, caps={"r": 1.0})
    payload = frozenset({"p"}) if nonempty else frozenset()
    kwargs = ({"metadata": {"payload": payload}} if kind == "metadata"
              else {"memory_pools": payload})
    root = engine.submit("root63", {"r": 1.0}, caps={"r": 1.0}, **kwargs)
    child = engine.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(root,))
    task = engine._tasks[root.sequence]
    classification = (tables._stable_task_payloads(task),
                      tables._stable_task_payloads(task, engine),
                      tables.pure_task(engine, task),
                      tables.callback_capable(task, engine))
    del task, kwargs
    seen, extra = [], []

    def release():
        seen.append((engine.now, root.done, child.started, child.done,
                     engine.completed_count, engine.serviced["r"]))
        if derived == 1:
            work = 1.0 if child.started is None else 2.0
        elif derived == 2:
            work = 1.0 if engine.now == 1.0 else 2.0
        else:
            return
        extra.append((work, engine.submit(
            "extra", {"r": work}, on_complete=lambda _event: None)))

    if late:
        engine.advance(horizon=0.0)
    finalizer = weakref.finalize(payload, release)
    del payload
    engine.advance()
    result = {
        "observed": [(work, event.started, event.time)
                     for work, event in extra],
        "seen": list(seen), "child": (child.started, child.time),
        "now": engine.now, "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "finished": not finalizer.alive,
        "classification": classification,
        "steps": engine.accel_status()["native_steps"],
        "ownership": (len(engine._tasks), len(engine._native_seqs),
                      len(engine._native_slots)),
    }
    _close(engine)
    return result

_LAST_EXPECTED = {
    0: {"observed": [], "child": (1.0, 2.0), "now": 3.0,
        "service": 190.0, "completed": 65},
    1: {"observed": [(2.0, 2.0, 3.0)], "child": (1.0, 2.0), "now": 3.0,
        "service": 192.0, "completed": 66},
    2: {"observed": [(1.0, 2.0, 2.5)], "child": (1.0, 2.0), "now": 3.0,
        "service": 191.0, "completed": 66},
}

@pytest.mark.parametrize("kind", ("metadata", "pools"))
@pytest.mark.parametrize("nonempty", (True, False))
@pytest.mark.parametrize("late", (False, True))
@pytest.mark.parametrize("derived", (0, 1, 2))
def test_r13_1_last_consumer_frozenset_matches_python(kind, nonempty, late,
                                                      derived):
    """Early and late registration never change the reference observation."""
    control = _last_consumer("python", kind, nonempty, late, derived)
    assert control["classification"] == (False, False, False, True)
    for key, value in _LAST_EXPECTED[derived].items():
        _same(control[key], value, (kind, nonempty, late, derived, key))
    _same(control["seen"],
          [(1.0, True, 1.0, False, 1, 64.0)], (kind, nonempty, late, derived))
    assert control["steps"] == 0 and control["finished"]
    assert control["ownership"] == (0, 0, 0)
    for backend in NATIVE:
        result = _last_consumer(backend, kind, nonempty, late, derived)
        assert result["classification"] == (False, False, False, True)
        assert result["steps"] == 0, (backend, kind, nonempty, late, derived)
        for key in ("seen", "observed", "child", "now", "service",
                    "completed", "finished", "ownership"):
            _same(result[key], control[key], (backend, key))

_OVERFLOW = {
    (4097, 1, 4095): {"seen": (1.0, 4097, None, False, 4097.0),
                      "child": 1.0002440810349036,
                      "clock": 1.0004881620698072, "service": 4099.0,
                      "completed": 4099},
    (8193, 1, 8191): {"seen": (1.0, 8193, None, False, 8193.0),
                      "child": 1.0001220554131576,
                      "clock": 1.0002441108263151, "service": 8195.0,
                      "completed": 8195},
    (1025, 32, 1023): {"seen": (1.0, 1025, None, False, 1025.0),
                       "child": 1.0009756097560976,
                       "clock": 1.0019512195121951, "service": 1027.0,
                       "completed": 1027},
}

def _overflow(backend, count, nres, trigger, derived):
    """Public overflow shape: one trace-bearing frozenset root at limit 1."""
    names = [f"r{index}" for index in range(nres)]
    work = dict.fromkeys(names, 1.0)
    engine = Engine([Resource(name, float(count)) for name in names],
                    backend=backend, trace_limit=1)
    payload = frozenset({"p"})
    roots = [engine.submit(f"root{index}", work,
                           metadata={"token": payload} if index == trigger
                           else None) for index in range(count)]
    child = engine.submit("child", work, after=(roots[trigger],))
    task = engine._tasks[roots[trigger].sequence]
    classification = (tables._stable_task_payloads(task, engine),
                      tables.pure_task(engine, task),
                      tables.callback_capable(task, engine))
    del task, roots
    seen, extra = [], []

    def release():
        seen.append((engine.now, engine.completed_count, child.started,
                     child.done, engine.serviced[names[0]]))
        work_amount = 1.0 if child.started is None else 2.0
        if derived:
            extra.append((work_amount, engine.submit(
                "extra", dict.fromkeys(names, work_amount),
                on_complete=lambda _event: None)))

    finalizer = weakref.finalize(payload, release)
    del payload
    engine.advance()
    result = {
        "seen": list(seen), "observed": [(amount, event.started, event.time)
                                         for amount, event in extra],
        "child": (child.started, child.time), "now": engine.now,
        "service": list(engine.serviced.values()),
        "completed": engine.completed_count,
        "finished": not finalizer.alive, "classification": classification,
        "steps": engine.accel_status()["native_steps"],
        "ownership": (len(engine._tasks), len(engine._native_seqs),
                      len(engine._native_slots)),
    }
    _close(engine)
    return result

@pytest.mark.parametrize("case", tuple(_OVERFLOW))
@pytest.mark.parametrize("derived", (False, True))
def test_r13_1_public_overflow_frozenset_matches_python(case, derived):
    """The repaired wire hold cannot be reached through an unsound classifier."""
    count, nres, trigger = case
    expected = _OVERFLOW[case]
    control = _overflow("python", count, nres, trigger, derived)
    assert control["classification"] == (False, False, True)
    _same(control["seen"], [expected["seen"]], (case, derived))
    _same(control["child"],
          (1.0, expected["clock"] if derived else expected["child"]),
          (case, derived))
    _same(control["now"], expected["clock"] if derived else expected["child"],
          (case, derived))
    _same(control["service"], [expected["service"] - (0.0 if derived else 1.0)]
          * nres, (case, derived))
    _same(control["completed"],
          expected["completed"] - (0 if derived else 1), (case, derived))
    _same(control["observed"],
          [(1.0, 1.0, expected["clock"])] if derived else [], (case, derived))
    assert control["steps"] == 0 and control["finished"]
    assert control["ownership"] == (0, 0, 0)
    for backend in NATIVE:
        result = _overflow(backend, count, nres, trigger, derived)
        assert result["classification"] == (False, False, True)
        assert result["steps"] == 0, (backend, case, derived)
        for key in ("seen", "observed", "child", "now", "service",
                    "completed", "finished", "ownership"):
            _same(result[key], control[key], (backend, case, derived, key))

@pytest.mark.parametrize("snapshot", (False, True), ids=("module", "snapshot"))
@pytest.mark.parametrize("phase,horizon,steps", (
    ("before_submit", None, 0), ("after_submit_before_advance", None, 3),
    ("after_zero_horizon", 0.0, 3), ("after_partial_advance", 0.5, 4)))
def test_r14_1_sentinel_attachment_phases(monkeypatch, snapshot, phase, horizon, steps):
    """Observe real packing/holds; retain no local pool or Task at completion."""
    submit, run = tables.Arena.submit, tables.Arena.run
    control = None
    for backend in BACKENDS:
        engine = _engine(backend)
        pool = engine._memory_pools_frozen if snapshot else ENGINE_EMPTY_FROZENSET
        flags, calls, seen = [], [], []
        finalizer = None
        def packed(arena, rows, deps=()):
            flags.extend(row[1] for row in rows)
            return submit(arena, rows, deps)
        def ran(arena, *args):
            answer = run(arena, *args)
            calls.append((answer[1], arena.last_held))
            return answer
        monkeypatch.setattr(tables.Arena, "submit", packed)
        monkeypatch.setattr(tables.Arena, "run", ran)
        try:
            assert weakref.getweakrefcount(pool) == 0
            assert tables._inert_pool_set(pool, engine)
            if phase == "before_submit":
                finalizer = weakref.finalize(pool, seen.append, "released")
            kwargs = {"metadata": {"inert": 1}} if snapshot else {}
            roots = [engine.submit(f"root{i}", {"r": work}, caps={"r": 1.0}, **kwargs)
                     for i, work in enumerate([2.0] + [3.0] * 62 + [1.0])]
            child = engine.submit("child", {"r": 1.0}, caps={"r": 1.0},
                                  after=(roots[-1],), **kwargs)
            if horizon is not None:
                engine.advance(horizon=horizon)
            if finalizer is None:
                finalizer = weakref.finalize(pool, seen.append, "released")
            task = engine._tasks[roots[-1].sequence]
            assert task.memory_pools is pool and weakref.getweakrefcount(pool) == 1
            assert not tables._inert_pool_set(pool, engine)
            assert not tables.pure_task(engine, task)
            assert tables.callback_capable(task, engine)
            del task, pool
            engine.advance()
            result = (engine.now, engine.serviced["r"], engine.completed_count,
                      (child.started, child.time), (len(engine._tasks),
                      len(engine._native_seqs), len(engine._native_slots)))
            assert result == (3.0, 190.0, 65, (1.0, 2.0), (0, 0, 0))
            semantic = (result, [(r.started, r.time) for r in roots], dict(engine.busy_ms))
            if backend == "python":
                control = semantic
            assert semantic == control and finalizer.alive and not seen
            native = backend != "python" and phase != "before_submit"
            callback = native and phase == "after_submit_before_advance"
            assert flags == ([tables.F_CALLBACK if callback else 0] * 65 if native else [])
            assert sum(held for _, held in calls) == (3 if callback else 0)
            assert sum(code == tables.B_CALLBACK for code, _ in calls) == (3 if callback else 0)
            assert engine.accel_status()["native_steps"] == (steps if native else 0)
        finally:
            if finalizer is not None:
                finalizer.detach()
            pool = engine._memory_pools_frozen if snapshot else ENGINE_EMPTY_FROZENSET
            _close(engine)
            assert weakref.getweakrefcount(pool) == 0
            assert tables._inert_pool_set(pool, engine)

def test_r13_module_stays_below_its_reviewed_line_cap():
    """This regression module stays <450 physical lines, LF-only and headed."""
    data = Path(__file__).read_bytes()
    assert len(data.splitlines()) < 450
    assert b"\r" not in data and data.endswith(b"\n")
    scoped = (sorted((ROOT / "mindie_sim" / "prediction" / "runtime").glob(
        "accel_*.py")) + sorted((ROOT / "tests").glob(
            "test_engine_accel_*.py")))
    for path in scoped:
        assert len(path.read_bytes().splitlines()) < 735, path.name
