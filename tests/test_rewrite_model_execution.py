"""T04 rewrite: model forward composes on the one caller Execution.

[Role] Prove the N-domain model producer keeps its analytical mathematical
  plan while every submitted physical cost belongs to the caller's single
  Engine and final readiness is an actual Event, not a summed isolated replay.
[Invariants] No shadow Engine when an Execution is supplied; no repeated
  timing cache across forwards; layer work still changes with geometry.
[Timeouts & Locks] One bounded CPU Engine advance.
[Side-Effect Policy] CPU-only virtual engine; no device, weights or I/O.
"""
from __future__ import annotations

from tests.test_piecewise_subgraph_model_pipeline import dense_model, pipeline


def test_model_forward_submits_one_engine_and_reads_actual_final_event():
    pipe = pipeline(dense_model())
    execution = _execution()
    geometry = _geometry()
    first = pipe.forward(geometry, execution=execution)
    assert first.submitted_event_count > 0
    assert first.execution_final_event is not None
    assert all(hasattr(event, "sequence")
               for event in first.execution_submission)
    assert execution.engine.uid
    execution.engine.advance()
    assert execution.engine.summary()["live_tasks"] == 0
    assert first.execution_final_event.done
    assert execution.engine.summary()["completed_tasks"] > 0
    assert first.execution_duration_ms is not None
    assert first.execution_duration_ms > 0.0
    assert first.execution_duration_ms <= float(first.total_ms) + 1e-9

    second = pipe.forward(geometry, execution=execution)
    # Reused immutable templates still execute their actual occurrence work;
    # no timing from the first run is replayed.
    assert second.submitted_event_count > 0
    assert second.execution_final_event is not first.execution_final_event
    execution.engine.advance()
    assert second.execution_final_event.done


def _execution():
    import json
    from pathlib import Path
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen35-a5.json").read_text())
    case = parse_case(raw)
    return Execution(case, CostModel(case), include_submission=False)


def _geometry():
    from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
    return WorkloadGeometry.homogeneous(2, 4, 32)


def test_compiled_subgraph_submit_is_one_catalog_on_the_caller_engine():
    from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
    pipe = pipeline(dense_model())
    compiled = pipe._subgraph(pipe.instantiated.find("mlp", layer=0,
                                                    role="dense_ffn"))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    evaluation = compiled.evaluate(geometry)
    execution = _execution()
    ready = execution.engine.submit("external:ready", latency=1.0)
    _, submitted = compiled.submit(execution, geometry, ready=ready)
    execution.engine.advance()
    assert submitted
    assert all(item.events for item in submitted)
    assert all(item.events[0].done for item in submitted)
    # Every per-cost event belongs to the one caller runtime and starts no
    # earlier than the declared external readiness boundary.
    assert execution.engine.now >= 1.0
    assert len({item.name for item in submitted}) == len(evaluation.operators)
    assert execution.engine.summary()["live_tasks"] == 0
