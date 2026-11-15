"""Round-6 regressions for the four open `astra_review_r6.md` items.

[Role] Turn each Round-6 counterexample and coordinate check into a public
  launcher, SOL projection, Engine-parity and TPOT-gate assertion.
[Invariants] Expected values come from the review's declared work, an actual
  Engine ledger or the report's own validator; no host defaults and no
  substring-only filler where an exact coordinate/value exists.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] No file, network or device writes; fixtures are in-memory.
"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from mindie_sim import compose_case
from mindie_sim.feature_reports import (
    Context, SourceIdentity, canonical_json, launch_markdown,
)
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_documents import (
    DocumentInputError, finite_json_loads, parse_document,
)
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.invocation import InvocationProgram
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineTemplate, PipelineTemplateProgram, Stage,
)
from mindie_sim.prediction.sol import bound, compare_sol
from mindie_sim.prediction.sol_work import Peak, project_ops, project_program
from mindie_sim.schema.case import Case
from mindie_sim.schema.invocation import InvocationTiming
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
CANARY = "R6_CANARY"
VERSION = "vllm-ascend@0.30.0"


def _op(name, deps=(), hbm=0.0, vec=0.0, family="matmul"):
    return OpSpec(op_id=name, family=family, phase="decode", stage=0, layer=0,
                  requests=("r0",), semantic_shape=(1, 1), local_shape=(1, 1),
                  padded_shape=(1, 1), launch_shape=(1, 1), hbm_bytes=hbm,
                  vector_ops=vec, deps=tuple(deps), dtype="bf16")


def _case():
    return Case.from_mapping(compose_case(
        "qwen3-30b-a3b", hardware="a3", layers=1, graph_capture_mode="none"))


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION, **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _context(plan_id):
    return Context("c", plan_id, "model_step", "decode", 1, 2, 128,
                   "b", "complete", (SourceIdentity("fixture"),))


@pytest.fixture
def lower_program():
    saved = Execution.__dict__["lower_program"]

    def install(program):
        Execution.lower_program = staticmethod(lambda context, recipe: program)

    yield install
    Execution.lower_program = saved


# ---------------------------------------------------------------- P1-01
def test_p1_01_early_duplicate_key_and_literal_cli_taint():
    payload = ('{"env":{"API_KEY":"' + CANARY + '"},"' + CANARY + '":1,"'
               + CANARY + '":2,"command":["vllm","serve","/m"]}')
    report = _parse(payload, format="json")
    assert report.status == "failed"
    assert report.diagnostics[-1].code == "duplicate_json_key"
    for rendered in (canonical_json(report), launch_markdown(report)):
        assert CANARY not in rendered
    with pytest.raises(DocumentInputError) as caught:
        finite_json_loads(payload)
    assert caught.value.code == "duplicate_json_key"
    assert CANARY in caught.value.secrets
    assert CANARY not in caught.value.message
    with pytest.raises(DocumentInputError) as document:
        parse_document(payload, format="json")
    assert CANARY in document.value.secrets

    literal_text = ("vllm serve /m --api-key " + CANARY
                    + " --served-model-name " + CANARY)
    assert CANARY in parse_document(literal_text, format="shell").secret_values
    literal = _parse(literal_text)
    assert literal.status == "completed"
    assert _setting(literal, "--served-model-name").value == "[REDACTED]"
    for rendered in (canonical_json(literal), launch_markdown(literal)):
        assert CANARY not in rendered

    nested = _parse("vllm serve /m --api-key " + CANARY
                    + " --additional-config '{\"" + CANARY + "\":1}'")
    assert nested.status == "partial"
    for rendered in (canonical_json(nested), launch_markdown(nested)):
        assert CANARY not in rendered
    for item in _settings(nested):
        for field in (item.value, item.name, item.pointer, item.reason, item.id):
            assert CANARY not in str(field or "")


def test_p1_01_json_flag_credentials_join_global_taint():
    report = _parse("vllm serve /m --additional-config "
                    "'{\"api_key\":\"R6_JSON_SECRET\"}'")
    assert report.status == "partial"
    for rendered in (canonical_json(report), launch_markdown(report)):
        assert "R6_JSON_SECRET" not in rendered


# ---------------------------------------------------------------- P1-11
def _setup_chain():
    return (Stage("a", {"memory@r0:s0": 1.6e9}),
            Stage("b", {"vector@r0:s0": 2.7648e9}, after=("a",)))


def test_p1_11_template_minimum_is_device_root_anchored(lower_program):
    case = _case()
    for work in (0.0, 5.5296e9):
        template = PipelineTemplate(
            0, 1, (PipelinePhase("body", (Stage("z", {"vector@r0:s0": work}),), 1),),
            setup=_setup_chain(), minimum_ms=10)
        lower_program(PipelineTemplateProgram(template))
        plan = project_ops(case, (_op("x"),))
        predicted = graph_costs(case, (_op("x"),), compact=True,
                                isolated=False)["duration_ms"]
        assert plan.coverage == "complete" and not plan.holes
        assert plan.terms[0].ideal_phase_precedence_ms == pytest.approx(
            2.0 + (work / 2.7648e9))
        assert bound(plan).sol_ms == pytest.approx(2.0 + (work / 2.7648e9))
        assert bound(plan, "declared_physical").sol_ms == pytest.approx(10.0)
        assert predicted == pytest.approx(10.0)


def _invocation_case(provenance):
    base = _case()
    rule = {"match": {"family": "matmul"}, "minimum_ms": 100}
    if provenance is not None:
        rule["provenance"] = provenance
    return replace(base, runtime=replace(base.runtime, kernels={
        **dict(base.runtime.kernels), "invocations": [rule]}))


def test_p1_11_invocation_floor_requires_affirmative_provenance():
    op = _op("matmul", hbm=1.6e9)
    for provenance in ({"observation_source": "measured100.csv"},
                       {"basis": "profiled"}, None, {}):
        case = _invocation_case(provenance)
        plan = project_ops(case, (op,))
        ideal = bound(plan)
        physical = bound(plan, "declared_physical")
        predicted = graph_costs(case, (op,), compact=True,
                                isolated=False)["duration_ms"]
        assert ideal.sol_ms == pytest.approx(1.0)
        assert physical.sol_ms == pytest.approx(1.0)
        assert physical.empirical_floor_ms == pytest.approx(100.0)
        assert not physical.floor_bases
        assert predicted == pytest.approx(100.0)
    admitted = project_ops(_invocation_case({"basis": "hardware_spec"}), (op,))
    physical = bound(admitted, "declared_physical")
    assert physical.sol_ms == pytest.approx(100.0)
    assert physical.empirical_floor_ms is None
    assert "declared_invocation_minimum" in physical.floor_bases


def test_p1_11_nested_invocation_timing_is_projected(lower_program):
    case = _case()
    program = KernelProgram((
        Stage("pre", {"memory@r0:s0": 1.6e9}),
        Stage("nested", after=("pre",), program=InvocationProgram(
            KernelProgram((Stage("v", {"vector@r0:s0": 2.7648e9}),)),
            InvocationTiming(startup_ms=2, minimum_ms=10,
                             provenance={"basis": "hardware_spec"})))))
    lower_program(program)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True,
                            isolated=False)["duration_ms"]
    assert plan.coverage == "complete" and not plan.holes
    assert bound(plan).sol_ms == pytest.approx(2.0)
    physical = bound(plan, "declared_physical")
    assert physical.sol_ms == pytest.approx(11.0)
    assert "declared_invocation_minimum" in physical.floor_bases
    assert "declared_invocation_startup" in physical.floor_bases
    assert predicted == pytest.approx(11.0)

    empirical = KernelProgram((
        Stage("pre", {"memory@r0:s0": 1.6e9}),
        Stage("nested", after=("pre",), program=InvocationProgram(
            KernelProgram((Stage("v", {"vector@r0:s0": 2.7648e9}),)),
            InvocationTiming(startup_ms=2, minimum_ms=10,
                             provenance={"basis": "profiled"})))))
    lower_program(empirical)
    plan = project_ops(case, (_op("x"),))
    physical = bound(plan, "declared_physical")
    assert physical.sol_ms == pytest.approx(2.0)
    assert physical.empirical_floor_ms == pytest.approx(10.0)
    assert not physical.floor_bases


# ---------------------------------------------------------------- P1-15
def _tpot(trace, steps=1, commits=2.8):
    plan = project_program(KernelProgram((Stage("a", {"hbm": 1}),)),
                           (Peak("hbm", "bytes", 1.0, "declared", "fixture"),))
    report = compare_sol(plan, 2.0, context=_context(plan.plan_id),
                         request_metrics_enabled=True, commits=commits,
                         request_trace={"decode_steps": steps, **trace})
    assert report.bound.sol_ms == pytest.approx(1.0)
    return report.request_metrics[1]


def test_p1_15_event_identity_and_all_alias_validation():
    unidentified = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                          "commit_events": [{"time_ms": 0}, {"time_ms": 10}]})
    assert unidentified.status == "unavailable"
    assert "request" in unidentified.reason

    numbered = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                      "commit_events": [0, 10]})
    assert numbered.status == "unavailable"
    assert "objects" in numbered.reason

    contradictory = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                           "commit_times": [0, 10], "commit_events": [
                               {"request_id": "rA", "time_ms": 0},
                               {"request_id": "rB", "time_ms": 10}]})
    assert contradictory.status == "unavailable"
    assert contradictory.reason

    bound_trace = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                         "request_id": "rA", "commit_events": [
                             {"time_ms": 0}, {"time_ms": 10}]})
    assert bound_trace.status == "available"
    assert bound_trace.predicted_ms == pytest.approx(10.0 / 2.8)
    bound_workload = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                            "workload": {"request_id": "rA"},
                            "commit_events": [{"time_ms": 0}, {"time_ms": 10}]})
    assert bound_workload.status == "available"
    assert bound_workload.request_id == "rA"

    agreeing = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                      "commit_times": [0, 10], "commit_events": [
                          {"request_id": "rA", "time_ms": 0},
                          {"request_id": "rA", "time_ms": 10}]})
    assert agreeing.status == "available"
    assert agreeing.predicted_ms == pytest.approx(10.0 / 2.8)

    disagreeing = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                         "commit_times": [0, 11], "commit_events": [
                             {"request_id": "rA", "time_ms": 0},
                             {"request_id": "rA", "time_ms": 10}]})
    assert disagreeing.status == "unavailable"
    assert "disagree" in disagreeing.reason

    numeric = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                     "commit_times": [0, 10]})
    assert numeric.status == "available"
    mapped_without_id = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                               "commit_times": [{"time_ms": 0},
                                                {"time_ms": 10}]})
    assert mapped_without_id.status == "unavailable"
    mixed = _tpot({"arrival_ms": 0, "first_commit_ms": 0, "commit_events": [
        {"request_id": "rA", "time_ms": 0},
        {"request_id": "rB", "time_ms": 10}]})
    assert mixed.status == "unavailable"


# ---------------------------------------------------------------- P2-23
def test_p2_23_json_shell_env_and_escaped_leaf_coordinates():
    payload = json.dumps({
        "notes": "decoy",
        "command": "export VLLM_ALL2ALL_BACKEND=mc2\nvllm serve /m",
    }, indent=2)
    report = _parse(payload, format="json")
    assert report.status == "completed"
    env = _setting(report, "VLLM_ALL2ALL_BACKEND")
    assert (env.span.line_start, env.span.column_start) == (3, 22)

    escaped = json.dumps({
        "command": "vllm serve /m --compilation-config "
                   "'{\"cudagraph_mode\":\"FULL\"}'",
    }, indent=2)
    report = _parse(escaped, format="json")
    leaf = _setting(report, "cudagraph_mode")
    assert (leaf.span.line_start, leaf.span.column_start) == (2, 53)
    assert escaped.splitlines()[1][leaf.span.column_start - 1] == '"'


def test_p2_23_yaml_block_and_folded_scalar_coordinates():
    try:
        import yaml  # noqa: F401
    except ImportError:
        unavailable = _parse("command: |\n  vllm serve /m\n", format="yaml")
        assert unavailable.diagnostics[-1].code == "yaml_parser_unavailable"
        return
    block = ("command: |\n"
             "  vllm serve /m --compilation-config "
             "'{\"cudagraph_mode\":\"FULL\"}'\n")
    report = _parse(block, format="yaml")
    flag = _setting(report, "--compilation-config")
    key = _setting(report, "cudagraph_mode")
    assert (flag.span.line_start, flag.span.column_start) == (2, 17)
    assert (key.span.line_start, key.span.column_start) == (2, 40)

    folded = ("command: >\n"
              "  vllm serve /m\n"
              "  --compilation-config\n"
              "  '{\"cudagraph_mode\":\"FULL\"}'\n")
    report = _parse(folded, format="yaml")
    flag = _setting(report, "--compilation-config")
    key = _setting(report, "cudagraph_mode")
    assert (flag.span.line_start, flag.span.column_start) == (3, 3)
    assert (key.span.line_start, key.span.column_start) == (4, 5)


def test_p1_11_non_flat_template_uses_setup_relative_branch(lower_program):
    template = PipelineTemplate(
        0, 1, (PipelinePhase("body", (Stage("z", program=KernelProgram(
            (Stage("v", {"vector@r0:s0": 2.7648e9}),))),), 1),),
        setup=(Stage("a", {"memory@r0:s0": 1.6e9}),), minimum_ms=10)
    lower_program(PipelineTemplateProgram(template))
    case = _case()
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True,
                            isolated=False)["duration_ms"]
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(11.0)
    assert predicted == pytest.approx(11.0)
