"""Execution-spine invariants: one runtime, one state ledger, bounded cursors.

[Role]
  Exercise the public ``Execution.submit_catalog`` seam and the repeated
  catalog cursor against observable causal behaviour: real start/done events,
  one typed state ledger owned by ``RuntimeState``, a live window bounded by
  declared slots, inherited nested readiness and real release symbols.
[Invariants]
  * A catalog submitted through ``Execution`` is the same ``Engine``
    submission; the returned ``Submission`` aliases that runtime's events.
  * ``Execution.fragment_state`` and every ``FragmentConductor.state`` are the
    one ledger at ``engine.state_ledger``; a catalog write/commit is visible
    there exactly once and never in a second version map.
  * A finite catalog with a large ``total`` holds O(slots * stages) live tasks
    and publishes only the declared window before the first advance.
  * Nested catalogs wait for the parent stage's real dependencies.
  * A release symbol inside a finite class frees the grant acquired by an
    earlier stage of the same panel.
[Timeouts & Locks]
  Pure in-memory virtual time; no sleeps, threads, devices or I/O.
[Side-Effect Policy]
  Constructs local Engines/Executions only; no shared state is mutated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime.contracts import (
    AllocationSpec, BindingFrame, EdgeSpec, FiniteProgramSpec, ProgramCatalog,
    ReleaseSpec, StageClass, StageSpec, WorkTerm,
)
from mindie_sim.prediction.scheduling import FragmentConductor
from mindie_sim.schema.operators import StateEffect


def _catalog(stages=(), *, finite=None, **kwargs):
    return ProgramCatalog(
        catalog_id="spine", implementation_id="spine-test",
        stages=tuple(stages), finite=finite, **kwargs)


def _stage(local_id, label, role, work=8.0, **kwargs):
    return StageSpec(
        local_id=local_id, label=label,
        work=(WorkTerm(physical_resource_role=role, issued=work),), **kwargs)


def _execution():
    import json

    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen35-a5.json").read_text())
    case = parse_case(raw)
    return Execution(case, CostModel(case), include_submission=False)


def _fair_resource(engine):
    return next(
        name for name, resource in engine.resources.items()
        if resource.policy == "fair" and resource.rate > 0)


def test_execution_submit_catalog_is_one_runtime_submission():
    execution = _execution()
    engine = execution.engine
    role = _fair_resource(engine)
    catalog = _catalog(
        (_stage(1, "op", role, 4.0, output_ports=("x",)),))
    occurrence = engine.new_occurrence()
    submission = execution.submit_catalog(
        catalog, BindingFrame(occurrence=occurrence, resources={role: role}),
        name="one")
    assert submission.occurrence == occurrence
    assert submission.start.sequence in engine._tasks
    assert submission.done.sequence in engine._tasks
    engine.advance(until=(submission.done,))
    assert submission.done.done
    assert submission.ports["x"].ready.done
    receipt = engine.snapshot_submission(submission)
    assert receipt["metrics"]["completed"] == 1
    assert receipt["metrics"]["service"][role] == pytest.approx(4.0)
    # The execution facade is a projection of this exact runtime object.
    assert execution.engine is engine


def test_fragment_state_and_catalog_effects_share_one_runtime_ledger():
    # A write-only catalog stage projects the exact handoff snapshot shape that
    # a typed fragment expects: speculative live version, nothing committed.
    written = Engine([Resource("cpu", 8.0)])
    write_catalog = _catalog((
        _stage(1, "store", "cpu", 8.0,
               state_effects=(
                   StateEffect("kv_cache:layer:0", "write", 1),)),))
    write_submission = written.submit_catalog(
        write_catalog, BindingFrame(occurrence=written.new_occurrence()),
        name="write")
    written.advance(until=(write_submission.done,))
    assert written.ensure_state_ledger().snapshot() == {
        "committed": {}, "live": {"kv_cache:layer:0": (1,)}}

    engine = Engine([Resource("cpu", 8.0)])
    ledger = engine.ensure_state_ledger()
    catalog = _catalog((
        _stage(1, "write", "cpu", 8.0,
               state_effects=(StateEffect("kv", "write", 1),)),
        _stage(2, "commit", "cpu", 8.0,
               relative_after=(EdgeSpec(source_node=1),),
               state_effects=(StateEffect("kv", "commit", 1),)),
    ))
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()),
        name="state")
    engine.advance(until=(submission.done,))
    assert engine.state_ledger is ledger
    snapshot = ledger.snapshot()
    assert snapshot["committed"] == {"kv": 1}
    assert snapshot["live"] == {"kv": (1,)}
    # One write plus one commit, each at its real completion boundary.
    assert engine.state_versions == {"kv": [1]}
    # A conductor on the same root adopts this ledger instead of a second one.
    conductor = FragmentConductor(engine)
    assert conductor.state is ledger
    assert conductor.state.check_read("kv", 1, at=engine.now) == 1


def test_finite_cursor_publishes_only_the_declared_live_window():
    engine = Engine([Resource("cpu", 8.0)])
    total, slots = 50_000, 4
    calls = {"n": 0}

    def index_of(index):
        calls["n"] += 1
        if not 0 <= index < total:
            raise IndexError(index)
        return (0, index + 1, index, index, index % slots)

    catalog = _catalog(finite=FiniteProgramSpec(
        total=total, slots=slots,
        classes=(StageClass(class_id=0, stages=(
            _stage(1, "tile", "cpu", 2.0),)),),
        index_of=index_of))
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()), name="f")
    # Submission is O(window): preflight never walks all 50k absolute panels.
    assert calls["n"] <= 2 * slots + 8
    assert len(engine._tasks) <= 2 * slots + 2
    engine.advance(until=(submission.done,))
    assert submission.done.done
    assert engine.serviced["cpu"] == pytest.approx(2.0 * total)
    assert engine.now == pytest.approx(2.0 * total / 8.0)
    assert engine.summary()["peak_live_tasks"] <= 2 * slots + 2
    assert engine.summary()["live_tasks"] == 0
    assert not engine._promises


def test_finite_nested_catalog_inherits_parent_readiness():
    engine = Engine([Resource("cpu", 8.0)])
    ready = engine.submit("external:ready", latency=10.0)
    nested = _catalog((_stage(1, "inner", "cpu", 8.0),))
    catalog = _catalog(finite=FiniteProgramSpec(
        total=2, slots=1,
        classes=(StageClass(class_id=0, stages=(
            StageSpec(local_id=1, label="outer", nested_catalog=nested),)),),
        index_of=lambda index: (0, index + 1, index, index, 0)))
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()),
        name="nested", after=(ready,))
    engine.advance(until=(submission.done,))
    # Two sequential inner calls may not begin before the external readiness
    # bound, so the makespan is exactly the ready boundary plus their service.
    assert engine.now == pytest.approx(10.0 + 2.0)
    assert engine.serviced["cpu"] == pytest.approx(16.0)


def test_flat_nested_catalog_inherits_parent_after_dependencies():
    engine = Engine([Resource("cpu", 8.0)])
    ready = engine.submit("external:ready", latency=10.0)
    nested = _catalog((_stage(1, "inner", "cpu", 8.0),))
    catalog = _catalog((
        StageSpec(local_id=1, label="outer", nested_catalog=nested),))
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()),
        name="flat", after=(ready,))
    engine.advance(until=(submission.done,))
    assert engine.now == pytest.approx(11.0)
    assert engine.serviced["cpu"] == pytest.approx(8.0)


def test_finite_release_symbol_frees_the_grant_of_its_panel():
    engine = Engine([Resource("cpu", 8.0)], pools={"l1": 16}, pool_units={"l1": 2})
    total = 3
    allocation = AllocationSpec(
        symbol=0, label="buf", pool_role="l1", amount=16.0, units=(0,))
    catalog = _catalog(finite=FiniteProgramSpec(
        total=total, slots=2,
        classes=(StageClass(class_id=0, stages=(
            _stage(1, "load", "cpu", 8.0, acquire=(allocation,)),
            _stage(2, "free", "cpu", 8.0,
                   relative_after=(EdgeSpec(source_node=1),),
                   release=(ReleaseSpec(symbol=0),)),)),),
        index_of=lambda index: (0, index + 1, index, index, index % 2)))
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()), name="alloc")
    engine.advance(until=(submission.done,))
    assert submission.done.done
    assert engine.allocated == {"l1": 0.0}
    assert engine.peak_allocated == {"l1": 16.0}
    acquires = [row for row in engine.allocation_trace if row["action"] == "acquire"]
    releases = [row for row in engine.allocation_trace if row["action"] == "release"]
    assert len(acquires) == len(releases) == total
    # Each panel owns a distinct live grant, so a later panel can never free it.
    assert len({row["name"] for row in acquires}) == total
    assert {row["name"] for row in acquires} == {row["name"] for row in releases}


def test_allocation_evidence_is_independent_of_the_verbose_trace():
    def run(trace_limit):
        engine = Engine(
            [Resource("cpu", 8.0)], pools={"l1": 8}, trace_limit=trace_limit)
        allocation = AllocationSpec(
            symbol=0, label="buf", pool_role="l1", amount=8.0, units=(0,))
        catalog = _catalog((
            _stage(1, "load", "cpu", 8.0, acquire=(allocation,)),
            _stage(2, "free", "cpu", 8.0,
                   relative_after=(EdgeSpec(source_node=1),),
                   release=(ReleaseSpec(symbol=0),)),
        ))
        submission = engine.submit_catalog(
            catalog, BindingFrame(occurrence=engine.new_occurrence()),
            name="trace")
        engine.advance(until=(submission.done,))
        return engine.summary()

    traced = run(64)
    quiet = run(0)
    for key in ("retained_trace_events", "trace_dropped"):
        traced.pop(key)
        quiet.pop(key)
    assert traced == quiet
    assert traced["retained_allocation_events"] == 2
    assert traced["allocation_events"] == quiet["allocation_events"] == 2


def test_a_state_ledger_owned_by_another_runtime_is_rejected():
    first = Engine([Resource("cpu", 1.0)])
    second = Engine([Resource("cpu", 1.0)])
    ledger = first.ensure_state_ledger({"kv": 1})
    assert FragmentConductor(first, state=ledger).state is ledger
    with pytest.raises(ValueError, match="state_ledger_runtime_conflict"):
        FragmentConductor(second, state=ledger)
    # The failed adoption must not leave a foreign ledger installed.
    assert second.state_ledger is None
    second.ensure_state_ledger({"kv": 1})
    assert second.state_ledger is not ledger
