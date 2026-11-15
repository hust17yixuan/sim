"""Assembly-scope behaviour: model, framework and service in one context.

These tests exercise event-derived boundaries and resource work. They do not
assert mere type/string presence: shared forward work and service-only nodes
are compared, and every framework occurrence is read from actual events.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import assemble
from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.frontend import (
    Frontend, SERVICE_FAMILIES, _ports_for_work,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Event
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.step_metrics import StepSummary
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


ROOT = Path(__file__).parents[1]


def a5_case():
    return json.loads((ROOT / "examples/cases/qwen35-a5.json").read_text())


def a5_plan(proposal=0):
    return IterationPlan((RequestView(
        "r0", 1, 8, True, "decode",
        draft_verify=bool(proposal), draft_proposal_tokens=proposal,
        draft_advanced_tokens=proposal or 0),))


def io_policy():
    return {
        "host_prepare_ms": .5,
        "host_result_ms": .5,
        "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "latency_us": 250},
        "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4, "latency_us": 250},
        "client_delivery_ms": 2,
    }


def test_a5_model_scope_excludes_service_and_keeps_same_forward_work():
    raw = a5_case()
    parsed = parse_case(raw)
    frontend = Frontend()
    model = frontend.program(a5_plan(), parsed.model, parsed.runtime, parsed.machine,
                             assembly="model")
    service = frontend.program(a5_plan(), parsed.model, parsed.runtime, parsed.machine,
                               assembly="service")
    model_ops = {op.op_id: op for op in model.work.ops}
    service_ops = {op.op_id: op for op in service.work.ops}
    assert model_ops.keys() < service_ops.keys()
    for op_id, op in model_ops.items():
        other = service_ops[op_id]
        assert (op.flops, op.hbm_bytes, op.vector_ops, op.link_bytes) == (
            other.flops, other.hbm_bytes, other.vector_ops, other.link_bytes)
    assert not ({op.family for op in model.work.ops} & SERVICE_FAMILIES)
    assert {"sampling", "token_commit"} <= {
        op.family for op in service.work.ops}
    assert model.layer_scope == "per_model_layer"
    assert not model.ports.state_out
    assert any(port.name == "state.token_committed" for port in service.ports.state_out)
    # iteration_input is a payload-free barrier: readiness only, no tensor.
    assert model.ports.tensor_in == ()
    assert [trigger[0] for trigger in model.triggers] == []
    assert {trigger[0] for trigger in service.triggers} == {
        "sampling", "token_commit"}


def test_int8_compute_logits_port_uses_declared_bf16_output_not_compute_dtype():
    raw = case()
    parsed = parse_case(raw)
    quant = QuantSpec(
        "int8-lm", weight_format="int8", activation_format="int8",
        expert_weight_format="int8", scale_format="fp32",
        native_format="int8", output_format="bf16", exclusions=(),
    )
    model = replace(parsed.model, quant=quant)
    plan = IterationPlan((RequestView("r", 1, 4, True, "decode"),))
    program = Frontend().program(plan, model, parsed.runtime, parsed.machine,
                                 assembly="model")
    logits = next(op for op in program.work.ops if op.family == "logits")
    assert logits.dtype == "int8"
    assert dict(logits.metadata)["compute_format"] == "int8"
    assert dict(logits.metadata)["output_format"] == "bf16"
    output = {port.name: port for port in program.ports.tensor_out}
    assert output["output.logits"].dtype == "bf16"
    assert program.ports.tensor_in == ()
    assert program.to_dict()["unbound_tensor_ports"] == []


def test_unknown_tensor_boundary_stays_explicitly_unbound():
    root = OpSpec("custom_root", "vector", "decode", 0, -1, ("r",),
                  (1,), (1,), (1,), (1,), dtype="", recipe="custom")
    work = IterationWork(
        (root,), IterationPlan((RequestView("r", 1, 0, True, "decode"),)),
        1, 1, 1, None, "decode", "test",
    )
    ports = _ports_for_work(work)
    assert len(ports.tensor_in) == 1
    assert ports.tensor_in[0].dtype == ""  # no activation-dtype fallback


def test_generic_graph_accepts_service_nodes_and_model_scope_has_none():
    raw = case()
    parsed = parse_case(raw)
    view = RequestView("r", 1, 4, True, "decode", draft_prepare=False)
    frontend = Frontend()
    plan = IterationPlan((view,))
    service = frontend.program(plan, parsed.model, parsed.runtime, parsed.machine,
                               assembly="service")
    model = frontend.program(plan, parsed.model, parsed.runtime, parsed.machine,
                             assembly="model")
    report = graph_costs(raw, service.work.ops, isolated=True)
    families = {row["family"] for row in report["timeline"]}
    assert {"sampling", "token_commit"} <= families
    assert report["duration_ms"] > 0
    pure = graph_costs(raw, model.work.ops)
    assert not ({row["family"] for row in pure["timeline"]} & SERVICE_FAMILIES)
    standalone = OpSpec(
        "sample", "sampling", "decode", 0, -1, ("r",), (1,), (1,), (1,), (1,),
        vector_ops=8, dtype="fp32", recipe="sample_logits",
    )
    commit = OpSpec(
        "commit", "token_commit", "decode", 0, -1, ("r",), (1,), (1,), (1,), (1,),
        deps=("sample",), recipe="token_commit",
    )
    accept = OpSpec(
        "accept", "acceptance", "decode", 0, -1, ("r",), (1,), (1,), (1,), (1,),
        vector_ops=4, dtype="fp32", deps=("commit",),
        recipe="speculative_acceptance",
    )
    accepted = graph_costs(raw, (standalone, commit, accept))
    assert [row["family"] for row in accepted["timeline"]] == [
        "sampling", "token_commit", "acceptance"]


def test_service_assembly_uses_one_context_and_event_feedback():
    raw = case(requests=[
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 4},
        {"id": "r1", "prompt_tokens": 3, "output_tokens": 4},
    ])
    raw["runtime"]["serving_io"] = io_policy()
    result = assemble(raw, scope="service", iterations=3, tokens=1, diagnostic_window=True)
    assert result["same_context"] is True
    assert result["independent_model_invocations"] == 0
    assert result["iterations_executed"] == 3
    assert result["triggers"]["model_invocation"] == 3
    assert result["triggers"]["batch_prepare"] == 3
    assert result["triggers"]["request_commit_callbacks"] == 6
    assert result["triggers"]["token_commit_nodes"] == 3
    assert result["triggers"]["sampling"] == 3
    rows = result["steps"]
    assert rows[0]["submitted_basis"] == "actual_step_host_submissions"
    first_host = rows[0]["host_submit_first"]
    last_host = rows[0]["host_submit_last"]
    assert first_host is not None and rows[0]["host_submit_event_count"] > 0
    assert rows[0]["submitted_ms"] == first_host["submit_ms"]
    assert rows[0]["submitted_last_ms"] == last_host["submit_ms"]
    assert rows[0]["submitted_ms"] >= rows[0]["host_prepare_done_ms"]
    assert rows[0]["submitted_ms"] > rows[0]["release_ms"]
    assert rows[0]["submitted_last_ms"] >= rows[0]["submitted_first_ms"]
    assert "host_submit" in first_host["event"]
    assert "host_submit" in last_host["event"]
    assert rows[0]["step_ready_ms"] > rows[0]["device_done_ms"]
    assert max(rows[0]["delivery_ms"].values()) > rows[0]["step_ready_ms"]
    # The next invocation starts from the actual delivery edge, not a constant.
    assert rows[1]["release_ms"] == pytest.approx(
        max(rows[0]["delivery_ms"].values()))
    assert rows[2]["release_ms"] == pytest.approx(
        max(rows[1]["delivery_ms"].values()))
    for request in result["requests"]:
        commits = [row["commit_ms"][request["request_id"]] for row in rows]
        assert request["generated_tokens"] == 3
        assert request["first_commit_ms"] == pytest.approx(commits[0])
        assert request["last_commit_ms"] == pytest.approx(commits[-1])
        assert request["ttft_ms"] == pytest.approx(
            commits[0] - request["arrival_ms"])
        assert request["tpot_ms"] == pytest.approx(
            (commits[-1] - commits[0]) / 2)
        assert request["output_tps"] > 0
    assert result["metrics"]["output_tokens"] == 6
    assert result["metrics"]["commit_events"] == 6
    assert result["metrics"]["delivery_events"] == 6
    assert result["metrics"]["client_propagation_events"] == 3
    assert result["framework_occurrences"]["client_delivery"] == 3
    assert result["framework_occurrences"]["client_order"] == 6
    assert result["triggers"]["client_delivery_batches"] == 3
    assert result["triggers"]["client_order_events"] == 6
    assert result["execution"]["live_tasks"] == 0
    assert result["execution"]["live_programs"] == 0
    assert result["execution"]["serviced"]["scheduler@r0"] > 0
    assert result["execution"]["serviced"]["h2d@r0"] == 3 * (2 * 8)
    assert result["execution"]["serviced"]["d2h@r0"] == 3 * ((1 + 1) * 4)
    events = result["execution"]["step_summary"]["total"]["request_events"]
    assert events["commits"] == 6
    assert events["fallback_completions"] == 0
    assert result["steps"][0]["declared_commit_ms"]
    assert result["steps"][0]["fallback_commit_ms"] == {}
    assert result["metrics"]["commit_events"] == 6
    assert result["metrics"]["fallback_completion_events"] == 0


def test_decode_token_budget_expands_to_single_token_steps_with_nonzero_tpot():
    raw = case(requests=[
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 6},
        {"id": "r1", "prompt_tokens": 3, "output_tokens": 6},
    ])
    raw["runtime"]["serving_io"] = io_policy()
    result = assemble(raw, scope="service", iterations=2, tokens=2, diagnostic_window=True)
    assert result["iterations_requested"] == 2
    assert result["decode_tokens_per_iteration"] == 2
    assert result["steps_requested"] == 4
    assert result["iterations_executed"] == result["steps_executed"] == 4
    assert result["triggers"]["model_invocation"] == 4
    assert result["triggers"]["request_commit_callbacks"] == 8
    contexts = [row["context_tokens"] for row in result["steps"]]
    assert [item["r0"] for item in contexts] == [4, 5, 6, 7]
    assert [item["r1"] for item in contexts] == [3, 4, 5, 6]
    assert result["state_contract"]["decode_step"] == "one_token_per_active_request"
    for row in result["steps"]:
        assert row["valid_tokens"] == 2
        assert row["emitted_tokens"] == {"r0": 1, "r1": 1}
        assert len(row["commit_ms"]) == 2
        assert row["submitted_basis"] == "actual_step_host_submissions"
    for request in result["requests"]:
        assert request["generated_tokens"] == 4
        assert request["tpot_ms"] > 0
        assert request["last_commit_ms"] > request["first_commit_ms"]
    assert result["execution"]["serviced"]["h2d@r0"] == 4 * (2 * 8)
    assert result["execution"]["serviced"]["d2h@r0"] == 4 * (2 * 4)


def test_replay_enqueue_and_batch_propagation_bind_real_events():
    raw = case(requests=[
        {"id": "r0", "prompt_tokens": 2, "output_tokens": 2},
        {"id": "r1", "prompt_tokens": 2, "output_tokens": 2},
    ])
    raw["runtime"].update(
        graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full",
        submission={"graph_replay_ms": .25},
        serving_io={
            "host_prepare_ms": .5, "host_result_ms": .5,
            "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8},
            "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4},
            "client_delivery_ms": 1,
        },
    )
    result = assemble(raw, scope="service", iterations=1, diagnostic_window=True)
    step = result["steps"][0]
    assert result["triggers"]["graph_replay_submissions"] == 1
    assert step["graph_replay_submissions"] == 1
    assert step["eager_submissions"] == 0
    assert step["submitted_basis"] == "actual_step_host_submissions"
    assert step["release_ms"] < step["host_prepare_done_ms"] <= step["submitted_ms"]
    assert step["submitted_ms"] < min(step["commit_ms"].values())
    assert step["client_propagation_ms"] is not None
    assert step["client_propagation_ms"] <= min(step["delivery_ms"].values())
    assert result["triggers"]["client_delivery_batches"] == 1
    assert result["triggers"]["client_order_events"] == 2
    assert len(step["delivery_ms"]) == 2
    assert result["execution"]["submission"]["graph_replays"] == 1


def test_single_two_token_budget_matches_two_real_one_token_steps():
    raw = case(requests=[{"id": "r0", "prompt_tokens": 4, "output_tokens": 2}])
    raw["runtime"]["serving_io"] = io_policy()
    bulk = assemble(raw, scope="service", iterations=1, tokens=2, diagnostic_window=True)
    stepped = assemble(raw, scope="service", iterations=2, tokens=1, diagnostic_window=True)
    assert bulk["steps_executed"] == stepped["steps_executed"] == 2
    assert bulk["triggers"]["model_invocation"] == 2
    assert stepped["triggers"]["model_invocation"] == 2
    assert bulk["triggers"]["request_commit_callbacks"] == 2
    assert stepped["triggers"]["request_commit_callbacks"] == 2
    assert bulk["execution"]["serviced"]["d2h@r0"] == 8
    assert stepped["execution"]["serviced"]["d2h@r0"] == 8
    for left, right in zip(bulk["steps"], stepped["steps"]):
        assert left["valid_tokens"] == right["valid_tokens"] == 1
        assert left["emitted_tokens"] == right["emitted_tokens"] == {"r0": 1}
        assert left["context_tokens"] == right["context_tokens"]
    assert bulk["requests"][0]["generated_tokens"] == 2
    assert bulk["requests"][0]["tpot_ms"] > 0
    assert bulk["requests"][0]["tpot_ms"] == pytest.approx(
        stepped["requests"][0]["tpot_ms"])


def test_framework_scope_has_real_no_layer_occurrences_and_no_model():
    raw = case(requests=[
        {"id": "r0", "prompt_tokens": 1, "output_tokens": 2},
        {"id": "r1", "prompt_tokens": 1, "output_tokens": 2},
    ])
    raw["runtime"]["serving_io"] = io_policy()
    result = assemble(raw, scope="framework", iterations=2,
                      emitted_tokens={"r0": 1, "r1": 1})
    assert result["model_fragment"] is None
    assert result["triggers"]["model_invocation"] == 0
    assert result["triggers"]["batch_prepare"] == 2
    assert result["triggers"]["request_commit_callbacks"] == 0
    assert result["execution"]["work"]["flops"] == 0
    assert result["execution"]["work"]["vector_ops"] == 0
    assert result["execution"]["serviced"]["scheduler@r0"] > 0
    assert result["execution"]["serviced"]["h2d@r0"] > 0
    assert result["execution"]["serviced"]["d2h@r0"] > 0
    assert all(row["submitted_ms"] is None for row in result["steps"])
    assert all(row["submitted_basis"] == "no_engine_launch"
               for row in result["steps"])
    assert all(fragment["layer"] is None for fragment in result["framework_fragments"])
    by_name = {fragment["name"]: fragment for fragment in result["framework_fragments"]}
    assert by_name["host_prepare"]["trigger"] == "per_batch"
    assert by_name["client_delivery"]["trigger"] == "per_batch_with_emissions"
    assert by_name["client_order"]["trigger"] == "per_request_per_batch"
    assert by_name["host_prepare"]["occurrences"] == 2
    # One propagation per batch; two per-request order/delivery events.
    assert by_name["client_delivery"]["occurrences"] == 2
    assert by_name["client_order"]["occurrences"] == 4
    assert result["metrics"]["client_propagation_events"] == 2
    assert result["metrics"]["delivery_events"] == 4
    events = result["execution"]["step_summary"]["total"]["request_events"]
    assert events["commits"] == 0
    assert events["fallback_completions"] == 4
    assert result["steps"][1]["declared_commit_ms"] == {}
    assert set(result["steps"][1]["fallback_commit_ms"]) == set(
        result["steps"][1]["active_requests"])
    assert result["metrics"]["fallback_completion_events"] == 4
    assert all(port["direction"] in {"in", "out"}
               for fragment in result["framework_fragments"]
               for group in fragment["ports"].values() for port in group)
    assert all(row["device_done_ms"] > 0 for row in result["steps"])


def test_prefill_then_decode_uses_actual_computed_kv_not_emitted_token():
    raw = case(requests=[{"id": "r", "prompt_tokens": 4, "output_tokens": 2}])
    result = assemble(raw, scope="service", phase="prefill", iterations=2, diagnostic_window=True)
    assert [row["mode"] for row in result["steps"]] == ["prefill", "decode"]
    assert [row["context_tokens"]["r"] for row in result["steps"]] == [0, 4]
    assert [row["computed_kv_tokens_after"]["r"]
            for row in result["steps"]] == [4, 5]
    assert result["requests"][0]["generated_tokens"] == 2


def test_explicit_decode_context_advances_by_written_kv():
    raw = case(requests=[{"id": "r", "prompt_tokens": 4, "output_tokens": 2}])
    result = assemble(raw, scope="service", phase="decode", iterations=1,
                      tokens=2, context=4, diagnostic_window=True)
    assert [row["context_tokens"]["r"] for row in result["steps"]] == [4, 5]
    assert [row["computed_kv_tokens_after"]["r"]
            for row in result["steps"]] == [5, 6]
    assert result["requests"][0]["tpot_ms"] > 0


def test_framework_mixed_short_long_batch_only_reports_active_request():
    raw = case(requests=[
        {"id": "short", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "long", "prompt_tokens": 2, "output_tokens": 3},
    ])
    raw["runtime"]["serving_io"] = io_policy()
    result = assemble(raw, scope="framework", iterations=2,
                      emitted_tokens={"short": 1, "long": 1})
    assert result["steps"][0]["active_requests"] == ["long", "short"]
    assert result["steps"][1]["active_requests"] == ["long"]
    assert result["steps"][1]["emitted_tokens"] == {"long": 1}
    assert result["steps"][1]["delivery_ms"].keys() == {"long"}
    assert result["triggers"]["client_delivery_batches"] == 2
    assert result["triggers"]["client_order_events"] == 3
    assert result["requests"][0]["generated_tokens"] == 1


def test_kernel_group_commit_is_declared_not_guessed_from_event_name():
    commit = Event(1, "r0:i0:token_commit:kernel:7:complete", time=4.0)
    completion = Event(2, "r0:i0:step_ready", time=5.0)
    step = SimpleNamespace(
        replica=0, iteration=0, mode="decode", replay_scope="none",
        graph_bucket=None, implementation_selections=(),
        completion=completion, start_ms=0.0, valid_tokens=1, padded_tokens=1,
        operators=2, device_complete=completion,
        declared_commits={"r": commit}, fallback_commits={},
        commits={"r": commit}, host_results={}, client_deliveries={},
    )
    declared = StepSummary()
    declared.record(step)
    events = declared.to_dict()["total"]["request_events"]
    assert events["commits"] == 1
    assert events["fallback_completions"] == 0
    assert events["unclassified_commits"] == 0

    fallback = StepSummary()
    fallback.record(SimpleNamespace(**{**vars(step), "declared_commits": {},
                                       "fallback_commits": {"r": commit}}))
    events = fallback.to_dict()["total"]["request_events"]
    assert events["commits"] == 0
    assert events["fallback_completions"] == 1
    assert events["unclassified_commits"] == 0


def test_real_execution_mixed_declared_and_fallback_commit_maps():
    parsed = parse_case(case())
    plan = IterationPlan((
        RequestView("r0", 1, 4, True, "decode"),
        RequestView("r1", 1, 4, True, "decode"),
    ))
    commit = OpSpec(
        "commit", "token_commit", "decode", 0, -1, ("r0",),
        (1,), (1,), (1,), (1,), recipe="token_commit",
    )
    work = IterationWork((commit,), plan, 1, 2, 2, None, "decode", "test")
    execution = Execution(
        parsed, CostModel(parsed), include_serving_io=False,
        include_submission=False,
    )
    step = execution.add(work, 0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    assert set(step.declared_commits) == {"r0"}
    assert set(step.fallback_commits) == {"r1"}
    assert set(step.commits) == {"r0", "r1"}
    public_step = step.to_dict()
    assert set(public_step["declared_commit_ms"]) == {"r0"}
    assert set(public_step["fallback_commit_ms"]) == {"r1"}
    events = execution.summary()["step_summary"]["total"]["request_events"]
    assert events["commits"] == 1
    assert events["fallback_completions"] == 1
    assert events["unclassified_commits"] == 0
    assert events["merged_commits"] == 2


def test_pure_model_scope_ignores_declared_serving_io():
    raw = case()
    raw["runtime"]["serving_io"] = io_policy()
    result = assemble(raw, scope="model", iterations=1)
    assert result["model_fragment"]["service_families_present"] == []
    assert result["framework_fragments"] == []
    assert result["triggers"]["request_commit_callbacks"] == 0
    assert result["triggers"]["batch_prepare"] == 0
    assert result["execution"]["serviced"].get("h2d@r0", 0) == 0
    assert result["execution"]["serviced"].get("d2h@r0", 0) == 0
    assert result["execution"]["serviced"].get("scheduler@r0", 0) == 0
    assert result["steps"][0]["submitted_ms"] is None
    assert result["steps"][0]["submitted_basis"] == "submission_not_included"
    events = result["execution"]["step_summary"]["total"]["request_events"]
    assert events["commits"] == 0
    assert events["fallback_completions"] == 1
    assert events["host_results"] == 0
    assert result["steps"][0]["declared_commit_ms"] == {}
    assert result["steps"][0]["fallback_commit_ms"].keys() == {"r"}
    assert result["metrics"]["commit_events"] == 0
    assert result["metrics"]["fallback_completion_events"] == 1


def test_invalid_scope_and_empty_framework_declaration_fail_loudly():
    raw = case()
    with pytest.raises(ModelingError, match="assembly_scope_unknown"):
        assemble(raw, scope="everything")
    with pytest.raises(ModelingError, match="framework_assembly_requires_declared_serving_io"):
        assemble(raw, scope="framework")
    multi = assemble(case(dp=2), scope="service")
    assert multi["assemble_backend"] == "cohort_driver"
