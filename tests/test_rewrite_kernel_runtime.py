"""Executable invariants for the T04 unique causal runtime.

These tests establish behavior and independent mathematics, not static shape:
one mutable root, conserved service, causal ordering, finite-cursor work,
host-FIFO independence, frozen binding snapshots and exact max-min fairness.
"""
from __future__ import annotations

import math

import pytest

from mindie_sim.prediction.engine import (
    Allocation, Engine, Event, ExecutionError, Resource,
)
from mindie_sim.prediction.runtime.arbiter import fair_rates
from mindie_sim.prediction.runtime.contracts import (
    BindingFrame, CapTerm, FiniteProgramSpec, HostProgramSpec, OccurrenceKey,
    PortSpec, ProgramCatalog, StageClass, StageSpec, WorkTerm,
)
from mindie_sim.prediction.runtime.host import HostCursor
from mindie_sim.prediction.runtime.periodic import IntervalSummary
from mindie_sim.prediction.runtime.state import RuntimeState
from mindie_sim.prediction import runtime


def _catalog(*stages, finite=None, host_spec=None, **kwargs):
    return ProgramCatalog(
        catalog_id="c", implementation_id="i", stages=tuple(stages),
        finite=finite, host_spec=host_spec, **kwargs)


def test_unique_root_and_single_engine_identity():
    engine = Engine([Resource("cpu", 8.0)])
    assert isinstance(engine, RuntimeState)
    assert runtime.Engine is Engine
    assert runtime.Event is Event
    snapshot = engine.snapshot()
    assert snapshot["uid"] == engine.uid
    assert snapshot["now_ms"] == 0.0
    other = Engine([Resource("cpu", 8.0)])
    assert other.uid != engine.uid


def test_service_is_conserved_and_order_is_causal():
    engine = Engine([Resource("cpu", 4.0)])
    first = engine.submit("first", {"cpu": 8.0})
    second = engine.submit("second", {"cpu": 4.0}, after=(first,))
    engine.advance(until=(second,))
    # 12 work at 4 work/ms, serialized by the dependency.
    assert engine.now == pytest.approx(3.0)
    assert second.started >= first.time
    assert engine.serviced["cpu"] == pytest.approx(12.0)
    assert sum(engine.summary()["serviced"].values()) == pytest.approx(12.0)


def test_fifo_is_nonpreemptive_and_priority_only_orders_admission():
    engine = Engine([Resource("wire", 1.0, policy="fifo")])
    low = engine.submit("low", {"wire": 2.0}, priority=0)
    high = engine.submit("high", {"wire": 1.0}, priority=9)
    engine.advance(until=(high,))
    # Low started first and keeps the wire; priority cannot preempt it.
    assert low.done and high.done
    # Both may be admitted, but the first owner keeps service until done.
    assert low.time < high.time


def test_catalog_stage_edges_and_ports_are_real_events():
    engine = Engine([Resource("cpu", 8.0)])
    catalog = _catalog(
        StageSpec(local_id=1, label="a",
                  work=(WorkTerm(physical_resource_role="cpu", issued=8.0),),
                  output_ports=("x",)),
        StageSpec(local_id=2, label="b",
                  work=(WorkTerm(physical_resource_role="cpu", issued=8.0),),
                  relative_after=()),
    )
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()), name="p")
    engine.advance(until=(submission.done,))
    assert submission.done.done and submission.start.done
    assert submission.ports["x"].ready.done
    assert engine.serviced["cpu"] == pytest.approx(16.0)
    receipt = engine.snapshot_submission(submission)
    assert "serviced" not in receipt
    assert receipt["metrics"]["completed"] >= 2
    assert receipt["metrics"]["service"]["cpu"] == pytest.approx(16.0)


def test_finite_cursor_counts_every_tile_and_coordinates():
    engine = Engine([Resource("cpu", 8.0)])
    classes = (StageClass(class_id=0, stages=(
        StageSpec(local_id=1, label="tile",
                  work=(WorkTerm(physical_resource_role="cpu", issued=8.0),)),)),)
    finite = FiniteProgramSpec(
        total=3, slots=2, classes=classes,
        index_of=lambda index: (0, index + 1, index, 0, index // 2))
    submission = engine.submit_catalog(
        _catalog(finite=finite), BindingFrame(occurrence=engine.new_occurrence()),
        name="f")
    engine.advance(until=(submission.done,))
    assert engine.serviced["cpu"] == pytest.approx(24.0)
    assert engine.now == pytest.approx(3.0)
    assert submission.done.done


def test_finite_minimum_and_setup_drain_execute():
    engine = Engine([Resource("cpu", 8.0)])
    classes = (StageClass(class_id=0, stages=(
        StageSpec(local_id=1, label="t",
                  work=(WorkTerm(physical_resource_role="cpu", issued=8.0),)),)),)
    finite = FiniteProgramSpec(
        total=1, slots=1, classes=classes,
        index_of=lambda index: (0, 1, 0, 0, 0),
        setup=(StageSpec(local_id=90, label="setup",
                         work=(WorkTerm(physical_resource_role="cpu", issued=8.0),)),),
        drain=(StageSpec(local_id=91, label="drain",
                         work=(WorkTerm(physical_resource_role="cpu", issued=8.0),)),),
        minimum_ms=5.0)
    submission = engine.submit_catalog(
        _catalog(finite=finite), BindingFrame(occurrence=engine.new_occurrence()),
        name="fd")
    engine.advance(until=(submission.done,))
    assert engine.now == pytest.approx(5.0)
    assert engine.serviced["cpu"] == pytest.approx(24.0)


def test_host_fifo_progress_is_independent_of_device_and_bounded():
    cursor = HostCursor(fifo_role="host", occurrence_uid="u", queue_capacity=1)
    assert cursor.can_submit()
    cursor.note_submit()
    assert not cursor.can_submit()
    with pytest.raises(ValueError):
        cursor.note_submit()
    cursor.note_host_done(0)
    assert cursor.can_submit()
    assert cursor.host_completed == 1


def test_binding_frame_snapshots_nested_mappings():
    engine = Engine([Resource("cpu", 1.0)])
    occurrence = engine.new_occurrence()
    geometry = {"shape": (2, 3)}
    frame = BindingFrame(occurrence=occurrence, geometry=geometry,
                         coordinates={"unit": 1})
    geometry["shape"] = (99, 99)
    assert frame.geometry["shape"] == (2, 3)
    with pytest.raises(TypeError):
        frame.coordinates["unit"] = 2  # type: ignore[index]


def test_capped_max_min_fairness():
    assert fair_rates(10.0, {1: 3.0, 2: 100.0}) == pytest.approx({1: 3.0, 2: 7.0})
    assert fair_rates(9.0, {1: 100.0, 2: 100.0, 3: 100.0}) == pytest.approx(
        {1: 3.0, 2: 3.0, 3: 3.0})


def test_periodic_summary_peaks_do_not_multiply():
    first = IntervalSummary(dt=2.0, completed=2, service={"cpu": 4.0},
                            peak_allocated={"hbm": 8.0})
    second = IntervalSummary(dt=1.0, completed=1, service={"cpu": 2.0},
                             peak_allocated={"hbm": 5.0})
    composed = first.plus(second)
    assert composed.dt == 3.0 and composed.completed == 3
    assert composed.service["cpu"] == 6.0
    assert composed.peak_allocated["hbm"] == 8.0
    assert first.scaled(3).service["cpu"] == 12.0
    assert first.scaled(3).peak_allocated["hbm"] == 8.0


def test_catalog_validation_publishes_nothing_on_failure():
    engine = Engine([Resource("cpu", 1.0)])
    bad = _catalog(StageSpec(
        local_id=1, label="bad",
        work=(WorkTerm(physical_resource_role="missing", issued=1.0),)))
    before = engine.snapshot()
    with pytest.raises(Exception):
        engine.submit_catalog(
            bad, BindingFrame(occurrence=engine.new_occurrence()), name="bad")
    after = engine.snapshot()
    assert after["live_tasks"] == before["live_tasks"] == 0
    assert after["next_sequence"] == before["next_sequence"]


def test_catalog_host_fifo_serializes_submit_and_precedes_device():
    engine = Engine([
        Resource("host", 1.0, policy="fifo"), Resource("cpu", 8.0)])
    catalog = _catalog(
        StageSpec(local_id=1, label="dev",
                  work=(WorkTerm(physical_resource_role="cpu", issued=8.0),)),
        host_spec=HostProgramSpec(fifo_role="host", submit_work_ms=2.0,
                                  queue_capacity=1))
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()), name="h")
    # A second launch cannot enter the bounded host queue before the first
    # host submit completes.
    with pytest.raises(ExecutionError):
        engine.submit_catalog(
            catalog, BindingFrame(occurrence=engine.new_occurrence()), name="h2")
    engine.advance(until=(submission.done,))
    assert engine.now == pytest.approx(3.0)
    assert engine.serviced["host"] == pytest.approx(2.0)
    assert engine.serviced["cpu"] == pytest.approx(8.0)
    assert engine.host_fifos["host"].host_completed == 1
    # Host progress is independent: a new launch is admissible after host
    # completion, not after an imaginary global drain.
    second = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()), name="h3")
    engine.advance(until=(second.done,))
    assert second.done.done
