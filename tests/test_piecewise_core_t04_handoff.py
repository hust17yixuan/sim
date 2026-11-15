"""Compile-core -> T01/T04 handoff: typed fragments and multi-stage segments.

[Role]
  Prove the two real public exits of the piecewise compile core:

  1. a single-stage, typed family record reaches T01 ``ProgramFragment`` and
     T04 ``FragmentConductor`` with its attrs, ports and state effect intact;
  2. a multi-stage GEMM is never flattened into one total-work fragment; its
     five stage ledgers stay explicit on ``OperatorSegment`` and are executed
     by the same ``Engine`` through ``run_engine_work(chains=...)``.

[Invariants]
  * The fragment preserves exact issued work, declared state effect and typed
    attrs; ``stage_projection_blockers`` reports the semantics that prevent a
    static single-Stage projection instead of silently dropping them.
  * Multi-stage ``family_work().fragment()`` raises
    ``family_work_fragment_requires_single_stage``; it does not substitute an
    aggregate duration or work total.
  * ``operator_segment`` keeps all stage ledgers and ``run_engine_work``
    advances one shared Engine to a positive makespan.

[Timeouts & Locks]
  Pure CPU arithmetic on repository fixtures; no device or network.

[Side-Effect Policy]
  Read-only test code; immutable records only.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.piecewise.composition import (
    operator_segment, run_engine_work,
)
from mindie_sim.prediction.piecewise.families_bridge import family_work
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.scheduling import FragmentConductor
from mindie_sim.schema.common import ModelingError
from tests.test_piecewise_subgraph_model_pipeline import (
    SUPPLY, dense_model, dense_plan, make_model, pipeline,
)


def _compiled(pipe, kind: str, role: str):
    instance = pipe.instantiated.find(kind, layer=0, role=role)
    return pipe._subgraph(instance)


def test_single_stage_typed_fragment_reaches_t04_conductor() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    compiled = _compiled(pipe, "kv_cache", "kv")
    operator = next(item for item in compiled.operators if item.name == "kv_store")
    work = family_work(operator, geometry)
    fragment = work.fragment()

    assert fragment.name == "kv_access:kv_store"
    issued = {(item.resource, item.unit, item.kind): item.amount
              for item in fragment.work.items}
    assert issued[("hbm", "bytes", "issued")] == 1024.0
    assert issued[("mte3", "bytes", "issued")] == 1024.0
    effects = [(effect.target, effect.action) for effect in fragment.state_effects]
    assert len(effects) == 1
    target, action = effects[0]
    # The instance-bound state effect carries the actual owner, not ``self``.
    assert target.startswith("kv_cache:")
    assert "layer:0" in target
    assert action == "write"
    assert {attr.name for attr in fragment.attrs} >= {
        "row_bytes", "valid_rows_mode", "cache_layout", "kv_dtype"}
    assert fragment.ports.tensor_out and fragment.ports.state_out
    # The full fragment carries semantics a single static Stage cannot.
    assert {"attrs", "state_effects"} <= set(fragment.stage_projection_blockers)

    engine = Engine([Resource("hbm", 1000.0, "bytes", "fair"),
                     Resource("mte3", 1000.0, "bytes", "fair")])
    conductor = FragmentConductor(engine)
    ready = engine.submit("test:ready", work={})
    handle = conductor.submit(fragment, inputs={"stage0_ready": ready})
    engine.advance()
    assert handle.event.time == pytest.approx(1.024)
    assert engine.summary()["live_tasks"] == 0


def test_multi_stage_work_is_not_flattened_and_uses_the_shared_engine() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    compiled = _compiled(pipe, "gqa_attention", "attention")
    operator = next(item for item in compiled.operators
                    if item.name == "qkv_projection")
    cost = operator.evaluate(geometry)
    work = family_work(operator, geometry)
    assert len(work.stages) == 5
    with pytest.raises(ModelingError, match="family_work_fragment_requires_single_stage"):
        work.fragment()

    segment = operator_segment(cost, compiled.supply)
    assert len(segment.stages) == 5
    assert all(isinstance(amount, Fraction)
               for stage in segment.stages for _pipe, amount in stage)
    result = run_engine_work({}, compiled.supply, chains={"main": [segment]})
    assert result.makespan_ms > 0
    assert dict(result.resources)["cube"] > 0


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


def test_model_forward_submits_actual_costs_to_passed_execution() -> None:
    from mindie_sim.prediction.piecewise.composition import compose_sequence
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    execution = _execution()
    prediction = pipe.forward(geometry, execution=execution)
    assert prediction.submitted_event_count > 0
    assert prediction.execution_final_event is not None
    execution.engine.advance()
    summary = execution.engine.summary()
    assert summary["live_tasks"] == 0
    assert summary["resource_busy_ms"]["matrix@r0:s0"] > 0
    assert summary["resource_busy_ms"]["memory@r0:s0"] > 0
    assert "kv_cache:layer:0" in execution.fragment_state.snapshot()["live"]

    instance = pipe.instantiated.find("mlp", layer=0, role="dense_ffn")
    compiled = pipe._subgraph(instance)
    evaluation, submitted = compiled.submit(
        _execution(), geometry, replica=0, stage=0)
    assert evaluation.operators
    assert submitted and all(item.events for item in submitted)

    composed = compose_sequence(
        (compiled,), geometry, execution=_execution(), replica=0, stage=0)
    assert composed.submission
    assert composed.to_dict()["submitted"] == len(composed.submission)


def test_predict_framework_execution_uses_actual_events_for_dp1() -> None:
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, predict_framework)
    pipe = pipeline(dense_model())
    execution = _execution()
    result = predict_framework(
        pipe,
        [dict(id="a", prompt_tokens=2, output_tokens=1),
         dict(id="b", prompt_tokens=3, output_tokens=1)],
        FrameworkConfig(supply=SUPPLY, max_batch=2, max_steps=8),
        execution=execution,
    )
    assert result.steps and result.steps[0]["body_ms"] > 0
    assert execution.engine.summary()["live_tasks"] == 0
    assert execution.engine.busy_ms["matrix@r0:s0"] > 0
    assert execution.engine.busy_ms["memory@r0:s0"] > 0


def test_predict_framework_execution_rejects_multi_dp_without_service_driver() -> None:
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, predict_framework)
    from mindie_sim.schema.common import ModelingError
    pipe = pipeline(dense_model(), feature_overrides={"dp": 2})
    with pytest.raises(ModelingError,
                       match="framework_execution_requires_single_dp"):
        predict_framework(
            pipe, [dict(id="a", prompt_tokens=2, output_tokens=1)],
            FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=4),
            execution=_execution())


def _controlled_execution():
    import json
    from pathlib import Path
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen35-a5.json").read_text())
    raw["machine"]["device"]["hbm_bw_bytes_per_s"] = 1_000_000
    raw["machine"]["device"]["mte_bw_bytes_per_s"] = 1_000_000
    case = parse_case(raw)
    return Execution(case, CostModel(case), include_submission=False)


def _pp_execution():
    import json
    from pathlib import Path
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen35-a5.json").read_text())
    raw["runtime"]["pp"] = 2
    raw["runtime"]["tp"] = 2
    case = parse_case(raw)
    return Execution(case, CostModel(case), include_submission=False)


def test_k0_typed_fragment_keeps_one_write_and_input_readiness() -> None:
    from mindie_sim.prediction.subgraphs.state import StateView
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    execution = _controlled_execution()
    ready = execution.engine.submit("external:ready", latency=2.0)
    compiled = _compiled(pipe, "kv_cache", "kv")
    evaluation, submitted = compiled.submit(
        execution, geometry, StateView(), ready=ready)
    execution.engine.advance()
    kv = next(item for item in submitted if item.name.startswith("kv_store"))
    assert kv.work == (("hbm", 1024.0), ("mte3", 1024.0))
    assert kv.events[0].time == pytest.approx(3.024)
    state = execution.fragment_state.snapshot()
    assert state["committed"] == {}
    assert state["live"] == {"kv_cache:layer:0": (1,)}


def test_f0_readiness_is_a_real_submission_dependency() -> None:
    from mindie_sim.prediction.piecewise.composition import compose_sequence
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    compiled = _compiled(pipe, "mlp", "dense_ffn")

    base_execution = _execution()
    base = compose_sequence((compiled,), geometry, execution=base_execution)
    base_execution.engine.advance()
    waited_execution = _execution()
    waited = compose_sequence(
        (compiled,), geometry, readiness={"gate_up": 3},
        execution=waited_execution)
    waited_execution.engine.advance()

    base_event = next(item.events[0] for item in base.submission
                      if item.name.startswith("gate_up"))
    waited_event = next(item.events[0] for item in waited.submission
                        if item.name.startswith("gate_up"))
    assert waited_event.started >= 3.0
    assert waited_event.time - base_event.time == pytest.approx(3.0)


def test_model_forward_submits_layers_beta_pp_and_globals() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    one = pipe.forward(geometry, execution=_execution())
    names = [event.name for event in one.execution_submission]
    for kind in ("gqa_attention", "kv_cache", "mlp", "vocab_parallel_embedding", "lm_head_sampler"):
        assert any(kind in name for name in names)
    # Fused members are replaced by the fused occurrence, not submitted.
    assert not any(":qk_norm:" in name for name in names)
    assert not any(":rotary:" in name for name in names)

    two = pipeline(dense_model(layers=3))
    reused = two.forward(geometry, execution=_execution())
    assert reused.layers[2].reused is True
    assert any("L2" in event.name for event in reused.execution_submission)

    plan = dense_plan(0, attention="kda", linear_heads=4, linear_head_dim=8,
                      linear_key_heads=4, linear_key_head_dim=8,
                      linear_value_heads=4, conv_kernel=4)
    kda = pipeline(make_model(plans=[plan]),
                   feature_overrides={"metadata": {"kda_use_beta_sigmoid": 0}})
    beta = kda.forward(
        WorkloadGeometry((RequestGeometry(1, 4, phase="decode"),),
                         phase="decode"), execution=_execution())
    beta_names = [event.name for event in beta.execution_submission]
    assert any("linear_beta_sigmoid" in name for name in beta_names)
    beta_index = next(index for index, name in enumerate(beta_names)
                      if "linear_beta_sigmoid" in name)
    recurrent_index = next(index for index, name in enumerate(beta_names)
                           if "kda_recurrent" in name)
    assert beta_index < recurrent_index

    parallel = pipeline(dense_model(layers=3), feature_overrides={"pp": 2})
    pp = parallel.forward(geometry, execution=_pp_execution())
    names = [event.name for event in pp.execution_submission]
    pp_index = next(index for index, name in enumerate(names)
                    if "PP" in name or "p2p" in name)
    before_pp = max(index for index, name in enumerate(names)
                    if ("L1" in name and "PP" not in name
                        and "p2p" not in name))
    after_pp = min(index for index, name in enumerate(names)
                   if ":L2:" in name or ":L2" in name)
    assert before_pp < pp_index < after_pp


def test_predict_framework_future_arrival_advances_and_finishes() -> None:
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, predict_framework)
    pipe = pipeline(dense_model())
    completions = {}
    ttfts = {}
    for arrival in (0, 1):
        execution = _execution()
        result = predict_framework(
            pipe, [dict(id="a", prompt_tokens=2, output_tokens=1,
                        arrival_ms=arrival)],
            FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=1),
            execution=execution)
        ttfts[arrival] = result.ttft_ms
        completions[arrival] = execution.engine.now
        assert len(result.steps) == 1
    assert ttfts[1] == pytest.approx(ttfts[0])
    assert completions[1] == pytest.approx(completions[0] + 1.0)


def _cohort_case():
    import json
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen35-a5.json").read_text())
    raw["runtime"]["serving_io"] = {"host_result_ms": 0.1,
                                    "client_delivery_ms": 0.2}
    return parse_case(raw)


def _cohort_context(case, requests):
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.model_pipeline import _coerce_requests
    from mindie_sim.prediction.scheduler import Scheduler
    specs = list(_coerce_requests(requests))
    memory = Memory(case)
    scheduler = Scheduler(case, specs, memory, 0)
    # Use the explicit scheduler frontier for this finite control; T04's
    # optional lazy path is a separate owner's concurrent implementation.
    execution = Execution(case, CostModel(case), include_submission=False,
                          include_serving_io=True, lazy_frontier=False)
    return specs, memory, scheduler, execution


def test_predict_framework_cohort_uses_t14_run_cohort_and_same_memory() -> None:
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, predict_framework)
    case = _cohort_case()
    pipe = pipeline(dense_model())
    specs, memory, scheduler, execution = _cohort_context(
        case, [dict(id="a", prompt_tokens=2, output_tokens=1)])
    result = predict_framework(
        pipe, specs,
        FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=4),
        case=case, execution=execution, schedulers=(scheduler,),
        memory=memory)
    assert scheduler.memory is memory
    assert execution.memory is memory
    assert execution.memory_ledger is memory
    assert result.stop_reason == "drained"
    assert result.requests[0].ttft_ms > 0
    assert result.scheduler_feedback[0]["host_tokens"] == 1
    assert result.scheduler_feedback[0]["delivered_tokens"] == 1
    assert execution.engine.now == pytest.approx(result.body_duration_ms)


def test_predict_framework_cohort_arrival_and_bounded_window() -> None:
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, predict_framework)
    case = _cohort_case()
    pipe = pipeline(dense_model())
    completed = {}
    ttft = {}
    for arrival in (0, 1):
        specs, memory, scheduler, execution = _cohort_context(
            case, [dict(id="a", prompt_tokens=2, output_tokens=1,
                        arrival_ms=arrival)])
        result = predict_framework(
            pipe, specs,
            FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=4),
            case=case, execution=execution, schedulers=(scheduler,),
            memory=memory)
        completed[arrival] = execution.engine.now
        ttft[arrival] = result.requests[0].ttft_ms
    assert ttft[1] == pytest.approx(ttft[0])
    assert completed[1] == pytest.approx(completed[0] + 1.0)

    specs, memory, scheduler, execution = _cohort_context(
        case, [dict(id="a", prompt_tokens=2, output_tokens=3)])
    bounded = predict_framework(
        pipe, specs,
        FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=4),
        case=case, execution=execution, schedulers=(scheduler,),
        memory=memory, max_model_steps=1)
    assert bounded.stop_reason == "max_model_steps"
    assert bounded.scheduler_feedback[0]["done"] is False
    assert bounded.scheduler_feedback[0]["generated"] == 1


def test_predict_framework_cohort_preserves_dp2_replicas() -> None:
    import json
    from pathlib import Path
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, _coerce_requests, predict_framework)
    from mindie_sim.prediction.scheduler import Scheduler
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen35-a5.json").read_text())
    raw["runtime"]["dp"] = 2
    raw["runtime"]["tp"] = 2
    raw["runtime"]["serving_io"] = {"host_result_ms": 0.1,
                                    "client_delivery_ms": 0.2}
    case = parse_case(raw)
    specs = list(_coerce_requests([
        dict(id="a", prompt_tokens=2, output_tokens=1),
        dict(id="b", prompt_tokens=2, output_tokens=1),
    ]))
    memory = Memory(case)
    schedulers = tuple(Scheduler(case, [spec], memory, rank)
                       for rank, spec in enumerate(specs))
    execution = Execution(case, CostModel(case), include_submission=False,
                          include_serving_io=True, lazy_frontier=False)
    result = predict_framework(
        pipeline(dense_model()), specs,
        FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=4),
        case=case, execution=execution, schedulers=schedulers,
        memory=memory)
    assert result.stop_reason == "drained"
    assert len(result.requests) == 2
    assert {item["replica"] for item in result.scheduler_feedback} == {0, 1}
    assert execution.memory is memory


def test_compose_sequence_joins_terminal_and_absolute_ready() -> None:
    from mindie_sim.prediction.piecewise.composition import compose_sequence
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    compiled = _compiled(pipe, "mlp", "dense_ffn")

    one_execution = _execution()
    compose_sequence((compiled,), geometry, execution=one_execution)
    one_execution.engine.advance()

    two_execution = _execution()
    two = compose_sequence((compiled, compiled), geometry,
                           execution=two_execution)
    two_execution.engine.advance()
    first_end = two.submission[2].events[-1].time
    second_start = two.submission[3].events[0].started
    assert second_start >= first_end
    assert two_execution.engine.now == pytest.approx(
        2 * one_execution.engine.now)

    waited_execution = _execution()
    waited = compose_sequence(
        (compiled,), geometry, readiness={"activation": 3},
        execution=waited_execution)
    waited_execution.engine.advance()
    activation = next(item for item in waited.submission
                      if item.name.startswith("activation"))
    assert activation.events[0].started == pytest.approx(3.0)


def test_predict_framework_graph_replay_advances_real_engine() -> None:
    import json
    from pathlib import Path
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.model_pipeline import (
        FrameworkConfig, _coerce_requests, predict_framework)
    from mindie_sim.prediction.scheduler import Scheduler
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).resolve().parents[1]
    case = parse_case(json.loads(
        (root / "examples/cases/qwen35-a5.json").read_text()))

    def run(replay):
        execution = Execution(case, CostModel(case), include_submission=False,
                              lazy_frontier=False)
        memory = Memory(case)
        specs = list(_coerce_requests([
            dict(id="a", prompt_tokens=2, output_tokens=2)]))
        scheduler = Scheduler(case, specs, memory, 0)
        result = predict_framework(
            pipeline(dense_model()), specs,
            FrameworkConfig(supply=SUPPLY, max_batch=1, max_steps=4,
                            graph_replay_ms=replay),
            case=case, execution=execution, schedulers=(scheduler,),
            memory=memory)
        return result

    baseline = run(0)
    replayed = run(3)
    assert replayed.body_duration_ms - baseline.body_duration_ms == pytest.approx(6.0)
    assert replayed.ttft_ms - baseline.ttft_ms == pytest.approx(3.0)
    assert replayed.tpot_ms - baseline.tpot_ms == pytest.approx(3.0)


def test_compiled_submit_joins_two_external_ready_inputs_once() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    execution = _execution()
    first = execution.engine.submit("external:a", latency=3.0)
    second = execution.engine.submit("external:b", latency=3.0)
    compiled = _compiled(pipe, "mlp", "dense_ffn")
    _, submitted = compiled.submit(execution, geometry,
                                   ready=(first, second))
    execution.engine.advance()
    gate = next(item for item in submitted
                if item.name.startswith("gate_up"))
    assert gate.events[0].started == pytest.approx(3.0)
    assert execution.engine.now > 3.0


def test_compile_stats_and_native_operator_records() -> None:
    pipe = pipeline(dense_model(layers=3))
    prediction = pipe.forward(
        WorkloadGeometry.homogeneous(2, 4, 32), execution=_execution())
    stats = pipe.compile_stats()
    assert stats["compiled_subgraphs"] > 0
    assert stats["dynamic_cache_high_water"] >= 0
    assert prediction.compiled_count == stats["compiled_subgraphs"]
    diagnostics = pipe.diagnostics(prediction)
    assert diagnostics["compiled"]["compiled_subgraphs"] < diagnostics["compiled"]["bound_subgraphs"]
    assert diagnostics["compiled"]["compiled_by_kind"]["gqa_attention"] == 1
    assert diagnostics["occurrence_stage_order"]
    assert diagnostics["state_last_use"]["kv_ready"] >= 0
    assert prediction.submitted_operators
    assert any(item.native for item in prediction.submitted_operators) or all(
        item.work for item in prediction.submitted_operators)


def test_prepared_subgraph_and_selected_native_program_are_used_once() -> None:
    from dataclasses import replace
    from mindie_sim.prediction.piecewise.compiler import NativeProgram
    from mindie_sim.prediction.piecewise.composition import submit_cost

    class _Program:
        def __init__(self):
            self.calls = 0
            self.peer = None

        def start(self, engine, name, ready, **kwargs):
            self.calls += 1
            self.peer = kwargs.get("peer_ready")
            after = (ready,) if self.peer is None else (ready, self.peer)
            return engine.submit(name, work={}, after=after)

    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    compiled = _compiled(pipe, "kv_cache", "kv")
    evaluation = compiled.prepare(geometry)
    execution = _execution()
    _, submitted = compiled.submit(
        execution, geometry, prepared=evaluation)
    assert submitted and len(submitted) == len(evaluation.operators)

    native_execution = _execution()
    cost = evaluation.operators[0]
    program = _Program()
    native = NativeProgram(program, work=tuple(cost.work),
                           duration=cost.duration_ms)
    native_cost = replace(cost, native_program=native)
    ready = native_execution.engine.submit("native:ready", latency=3.0)
    peer = native_execution.engine.submit("native:peer", latency=5.0)
    submitted_native = submit_cost(
        native_cost, native_execution, name="selected_native", ready=ready,
        native_kwargs={"peer_ready": peer})
    native_execution.engine.advance()
    assert program.calls == 1
    assert program.peer is peer
    assert submitted_native.native is True
    assert submitted_native.events[0].started >= 5.0


def test_materialized_rows_mode_only_changes_retention() -> None:
    pipe = pipeline(dense_model(layers=3))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    detailed = pipe.forward(geometry)
    compact = pipe.forward(geometry, retain_rows=False)
    assert compact.rows == ()
    assert compact.total_ms == detailed.total_ms
    assert compact.state_out == detailed.state_out
    assert compact.compiled_count == detailed.compiled_count


def test_submit_serial_result_carries_each_entering_state() -> None:
    from mindie_sim.prediction.piecewise.composition import (
        SerialResult, submit_serial_result)
    from mindie_sim.prediction.subgraphs.state import StateView
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    compiled = _compiled(pipe, "mlp", "dense_ffn")
    evaluation = compiled.prepare(geometry)
    states = (StateView(iteration=0), StateView(iteration=1),
              StateView(iteration=2))
    execution = _execution()
    submitted = submit_serial_result(
        SerialResult(evaluation.operators, evaluation.total_ms,
                     evaluation.state_out),
        execution, compiled=compiled, geometry=geometry,
        states=states, supply=pipe.supply)
    assert [item.entering_state.iteration for item in submitted] == [0, 1, 2]
    assert [item.entering_state for item in submitted] == list(states)
