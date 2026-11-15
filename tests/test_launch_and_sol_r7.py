"""Round-7 regressions for the five open `astra_review_r7.md` findings.

[Role] Turn every Round-7 counterexample and its structural variants into a
  public launcher, SOL projection, Engine-parity, trace-gate, coordinate and
  bounded-work assertion.
[Invariants] Expected values come from the review's declared work, an actual
  Engine ledger or the report's own validator; no host default and no
  guessed coordinate is accepted as exact provenance.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] No file, network or device writes; fixtures are in-memory.
"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from mindie_sim import compose_case
from mindie_sim.feature_reports import (
    Context, SourceIdentity, canonical_json, launch_markdown, validate_launch_report,
)
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_documents import DocumentInputError, parse_document
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.invocation import InvocationProgram
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineTemplate, PipelineTemplateProgram, Stage,
)
from mindie_sim.prediction.sol import bound, compare_sol
from mindie_sim.prediction.sol_work import project_ops
from mindie_sim.schema.case import Case
from mindie_sim.schema.invocation import InvocationTiming
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
VERSION = "vllm-ascend@0.30.0"
CANARY = "R7_CANARY"
try:  # The retained suite exercises both the installed and absent branches.
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _op(name="x", hbm=0.0, vec=0.0):
    return OpSpec(op_id=name, family="matmul", phase="decode", stage=0, layer=0,
                  requests=("r0",), semantic_shape=(1, 1), local_shape=(1, 1),
                  padded_shape=(1, 1), launch_shape=(1, 1), hbm_bytes=hbm,
                  vector_ops=vec, dtype="bf16")


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


def _project(program, install):
    case = _case()
    install(program)
    plan = project_ops(case, (_op(),))
    predicted = graph_costs(case, (_op(),), compact=True,
                            isolated=False)["duration_ms"]
    return plan, predicted


def _compare(plan, predicted):
    return compare_sol(plan, predicted, context=_context(plan.plan_id),
                       dispatch="declared_physical")


# ---------------------------------------------------------------- P1-01
def test_p1_01_decoded_escaped_credentials_never_echo():
    payloads = [
        ('{"env":{"API_\\u004bEY":"' + CANARY + '"},"' + CANARY + '":1,"'
         + CANARY + '":2,"command":["vllm","serve","/m"]}'),
        ('{"env":{"API_KEY":"R7_\\u0043ANARY"},"' + CANARY + '":1,"'
         + CANARY + '":2,"command":["vllm","serve","/m"]}'),
    ]
    for payload in payloads:
        report = _parse(payload, format="json")
        assert report.status == "failed"
        assert report.diagnostics[-1].code == "duplicate_json_key"
        assert CANARY not in canonical_json(report)
        assert CANARY not in launch_markdown(report)
        with pytest.raises(DocumentInputError) as caught:
            parse_document(payload, format="json")
        assert caught.value.code == "duplicate_json_key"
        assert CANARY in caught.value.secrets
        assert CANARY not in caught.value.message
        assert "[REDACTED]" in caught.value.message

    yaml_text = ('env:\n  "API_\\u004bEY": "R7_\\u0043ANARY"\n'
                 + CANARY + ': 1\n' + CANARY + ': 2\n'
                 'command: ["vllm", "serve", "/m"]\n')
    report = _parse(yaml_text, format="yaml")
    assert report.status == "failed"
    if not HAS_YAML:
        assert report.diagnostics[-1].code == "yaml_parser_unavailable"
        return
    assert report.diagnostics[-1].code == "yaml_duplicate_key"
    with pytest.raises(DocumentInputError) as caught:
        parse_document(yaml_text, format="yaml")
    assert caught.value.code == "yaml_duplicate_key"
    assert CANARY in caught.value.secrets
    assert "R7_\\u0043ANARY" in caught.value.secrets
    assert CANARY not in canonical_json(report)
    assert CANARY not in launch_markdown(report)


def test_p1_01_digit_credentials_are_tainted_type_safely():
    literal = "vllm serve /m --api-key 8675309123 --served-model-name 8675309123"
    report = _parse(literal)
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").value == "[REDACTED]"
    assert "8675309123" not in canonical_json(report)
    assert "8675309123" not in launch_markdown(report)

    variant = "vllm serve /m --api-key 8000 --port 8000 --served-model-name 8000"
    report = _parse(variant)
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").value == "[REDACTED]"
    rendered = canonical_json(report)
    assert json.loads(rendered)["status"] == "completed"
    assert "8000" not in json.dumps(_settings_to_dict(report))

    short = _parse("vllm serve /m --api-key 7 --served-model-name 7")
    assert short.status == "completed"
    assert _setting(short, "--served-model-name").value == "[REDACTED]"
    assert validate_launch_report(short) == ()


def _settings_to_dict(report):
    return {item.id: item.value for item in _settings(report)}


# ---------------------------------------------------------------- P1-11
def _inner_flat():
    return PipelineTemplate(
        0, 1, (PipelinePhase("body", (Stage("z"),), 1),),
        setup=(Stage("a", {"memory@r0:s0": 1.6e9}),
               Stage("b", {"vector@r0:s0": 2.7648e9}, after=("a",))),
        minimum_ms=10)


def _nested_template(where):
    inner = PipelineTemplateProgram(_inner_flat())
    if where == "body":
        return PipelineTemplateProgram(PipelineTemplate(
            0, 1, (PipelinePhase("outer", (Stage("nested", program=inner),), 1),)))
    if where == "setup":
        return PipelineTemplateProgram(PipelineTemplate(
            0, 1, (PipelinePhase("outer", (Stage("z"),), 1),),
            setup=(Stage("nested", program=inner),)))
    return PipelineTemplateProgram(PipelineTemplate(
        0, 1, (PipelinePhase("outer", (Stage("z"),), 1),),
        drain=(Stage("nested", program=inner),)))


def test_p1_11_nested_template_uses_its_own_catalog_branch(lower_program):
    for where in ("body", "setup", "drain"):
        plan, predicted = _project(_nested_template(where), lower_program)
        assert plan.coverage == "complete" and not plan.holes
        assert bound(plan).sol_ms == pytest.approx(2.0)
        assert bound(plan, "declared_physical").sol_ms == pytest.approx(10.0)
        assert predicted == pytest.approx(10.0)
        assert _compare(plan, predicted).status == "completed"


def _nested_invocation(startup=2, minimum=0, body=2.7648e9):
    return InvocationProgram(
        KernelProgram((Stage("v", {"vector@r0:s0": body}),)),
        InvocationTiming(startup_ms=startup, minimum_ms=minimum,
                         provenance={"basis": "hardware_spec"}))


def test_p1_11_nested_startup_is_scoped_to_its_stage(lower_program):
    parallel = KernelProgram((
        Stage("slow", {"memory@r0:s0": 16e9}),
        Stage("nested", program=_nested_invocation()),
    ))
    plan, predicted = _project(parallel, lower_program)
    physical = bound(plan, "declared_physical")
    assert bound(plan).sol_ms == pytest.approx(10.0)
    assert physical.sol_ms == pytest.approx(10.0)
    assert predicted == pytest.approx(10.0)
    assert "declared_invocation_startup" in physical.floor_bases
    assert _compare(plan, predicted).status == "completed"

    serial = KernelProgram((
        Stage("pre", {"memory@r0:s0": 1.6e9}),
        Stage("nested", after=("pre",), program=_nested_invocation(minimum=10)),
    ))
    plan, predicted = _project(serial, lower_program)
    physical = bound(plan, "declared_physical")
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert physical.sol_ms == pytest.approx(11.0)
    assert predicted == pytest.approx(11.0)
    assert {"declared_invocation_startup",
            "declared_invocation_minimum"} <= set(physical.floor_bases)
    assert _compare(plan, predicted).status == "completed"

    diamond = KernelProgram((
        Stage("pre", {"memory@r0:s0": 1.6e9}),
        Stage("slow", {"memory@r0:s0": 16e9}, after=("pre",)),
        Stage("nested", after=("pre",), program=_nested_invocation()),
        Stage("join", after=("slow", "nested")),
    ))
    plan, predicted = _project(diamond, lower_program)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(11.0)
    assert predicted == pytest.approx(11.0)
    assert _compare(plan, predicted).status == "completed"

    doubly = PipelineTemplateProgram(PipelineTemplate(
        0, 1, (PipelinePhase("o", (Stage("n", program=_nested_invocation()),), 1),),
        minimum_ms=1))
    plan, predicted = _project(doubly, lower_program)
    assert _compare(plan, predicted).status == "completed"


def test_p1_11_per_source_floor_provenance_is_not_laundered():
    base = _case()
    kernels = {
        **dict(base.runtime.kernels),
        "service_regimes": [{
            "id": "measured100", "match": {"family": "matmul"},
            "response": {"kind": "affine", "fixed_us": 100000, "terms": []},
            "provenance": {"observation_source": "measured100.csv"},
        }],
        "invocations": [{
            "match": {"family": "matmul"}, "minimum_ms": 10,
            "provenance": {"basis": "hardware_spec"},
        }],
    }
    case = replace(base, runtime=replace(base.runtime, kernels=kernels))
    op = _op("matmul", hbm=1.6e9)
    plan = project_ops(case, (op,))
    predicted = graph_costs(case, (op,), compact=True,
                            isolated=False)["duration_ms"]
    ideal, physical = bound(plan), bound(plan, "declared_physical")
    assert ideal.sol_ms == pytest.approx(1.0)
    assert physical.sol_ms == pytest.approx(10.0)
    assert physical.empirical_floor_ms == pytest.approx(100.0)
    assert physical.floor_bases == ("declared_invocation_minimum",)
    assert predicted == pytest.approx(100.0)
    report = compare_sol(plan, predicted, context=_context(plan.plan_id),
                         dispatch="declared_physical")
    assert report.status == "completed"
    assert not [item for item in report.diagnostics
                if item.code == "bound_exceeds_prediction"]

    declared = {**kernels, "service_regimes": [{
        **kernels["service_regimes"][0], "provenance": {"basis": "declared"}}]}
    other = replace(base, runtime=replace(base.runtime, kernels=declared))
    physical = bound(project_ops(other, (op,)), "declared_physical")
    assert physical.sol_ms == pytest.approx(100.0)
    assert physical.empirical_floor_ms is None


# ---------------------------------------------------------------- P1-15
def _tpot(trace, steps=1, commits=2.8):
    plan = project_ops(_case(), (_op("a", hbm=1.6e9),))
    report = compare_sol(plan, 2.0, context=_context(plan.plan_id),
                         request_metrics_enabled=True, commits=commits,
                         request_trace={"decode_steps": steps, **trace})
    return report.request_metrics[1]


def test_p1_15_every_identity_container_must_agree():
    conflict = _tpot({
        "request_id": "rA", "workload": {"request_id": "rB"},
        "commit_events": [{"time_ms": 0}, {"time_ms": 10}],
    })
    assert conflict.status == "unavailable"
    assert conflict.reason == "conflicting_request_identity"
    assert conflict.request_id is None

    same_container = _tpot({
        "request_id": "rA", "id": "rB",
        "commit_events": [{"time_ms": 0}, {"time_ms": 10}],
    })
    assert same_container.status == "unavailable"
    assert same_container.reason == "conflicting_request_identity"

    event_conflict = _tpot({
        "request_id": "rA",
        "commit_events": [
            {"request_id": "rA", "id": "rB", "time_ms": 0},
            {"request_id": "rA", "time_ms": 10},
        ],
    })
    assert event_conflict.status == "unavailable"
    assert event_conflict.reason == "conflicting_request_identity"

    invalid = _tpot({
        "request_id": {"request": "rA"},
        "commit_events": [{"time_ms": 0}, {"time_ms": 10}],
    })
    assert invalid.status == "unavailable"
    assert invalid.request_id is None
    assert "non-empty string" in invalid.reason

    for identity in (1, True, "", [], {}):
        rejected = _tpot({
            "request_id": identity,
            "commit_events": [{"time_ms": 0}, {"time_ms": 10}],
        })
        assert rejected.status == "unavailable", identity

    agreeing = _tpot({
        "request_id": "rA", "id": "rA",
        "workload": {"request_id": "rA"},
        "request": {"id": "rA"},
        "commit_events": [
            {"request_id": "rA", "id": "rA", "time_ms": 0},
            {"request_id": "rA", "id": "rA", "time_ms": 10},
        ],
        "commit_times": [0, 10],
    })
    assert agreeing.status == "available"
    assert agreeing.request_id == "rA"
    assert agreeing.predicted_ms == pytest.approx(10.0 / 2.8)


def test_p1_15_every_event_timestamp_alias_must_agree():
    conflict = _tpot({
        "commit_events": [
            {"request_id": "rA", "time_ms": 0, "commit_ms": 100},
            {"request_id": "rA", "time_ms": 10, "commit_ms": 5},
        ],
    })
    assert conflict.status == "unavailable"
    assert conflict.reason == "conflicting_event_timestamps"

    invalid = _tpot({"commit_events": [
        {"request_id": "rA", "time_ms": 0, "at_ms": True},
        {"request_id": "rA", "time_ms": 10},
    ]})
    assert invalid.status == "unavailable"

    agreeing = _tpot({"commit_events": [
        {"request_id": "rA", "time_ms": 0, "commit_ms": 0, "time": 0.0,
         "at_ms": 0, "timestamp_ms": 0},
        {"request_id": "rA", "time_ms": 10, "commit_ms": 10.0, "time": 10,
         "at_ms": 10, "timestamp_ms": 10},
    ]})
    assert agreeing.status == "available"
    assert agreeing.predicted_ms == pytest.approx(10.0 / 2.8)


# ---------------------------------------------------------------- P2-23
def test_p2_23_yaml_multiline_quotes_and_explicit_indentation():
    cases = {
        ("multiline-double",
         'command: "vllm serve /m\n  --compilation-config '
         '\'{\\"cudagraph_mode\\":\\"FULL\\"}\'"\n'): (2, 3, 2, 27),
        ("multiline-single",
         "command: 'vllm serve /m\n  --compilation-config "
         "''{\"cudagraph_mode\":\"FULL\"}'''\n"): (2, 3, 2, 27),
        ("explicit-indent",
         'command: |2\n    vllm serve /m --compilation-config '
         '\'{"cudagraph_mode":"FULL"}\'\n'): (2, 19, 2, 42),
        ("folded-blank",
         'command: >\n  vllm serve /m --compilation-config \'{\n\n'
         '  "cudagraph_mode":"FULL"}\'\n'): (2, 17, 4, 3),
    }
    for (label, text), expected in cases.items():
        report = _parse(text, format="yaml")
        if not HAS_YAML:
            assert report.diagnostics[-1].code == "yaml_parser_unavailable"
            return
        assert report.status == "completed", label
        flag = _setting(report, "--compilation-config")
        key = _setting(report, "cudagraph_mode")
        assert (flag.span.line_start, flag.span.column_start) == expected[:2], label
        assert (key.span.line_start, key.span.column_start) == expected[2:], label


def test_p2_23_crlf_docker_and_shell_unquoting_compose():
    crlf = json.dumps({
        "notes": "decoy",
        "command": "export VLLM_ALL2ALL_BACKEND=mc2\r\n"
                   "vllm serve /m --compilation-config "
                   "'{\"cudagraph_mode\":\"FULL\"}'",
    }, indent=2)
    report = _parse(crlf, format="json")
    env = _setting(report, "VLLM_ALL2ALL_BACKEND")
    key = _setting(report, "cudagraph_mode")
    assert (env.span.line_start, env.span.column_start) == (3, 22)
    assert (key.span.line_start, key.span.column_start) == (3, 88)

    docker = json.dumps({"command": [
        "docker", "run", "image", "sh", "-c",
        "export VLLM_ALL2ALL_BACKEND=mc2\n"
        "vllm serve /m --compilation-config '{\"cudagraph_mode\":\"FULL\"}'",
    ]}, indent=2)
    report = _parse(docker, format="json")
    flag = _setting(report, "--compilation-config")
    key = _setting(report, "cudagraph_mode")
    assert (flag.span.line_start, flag.span.column_start) == (8, 53)
    assert (key.span.line_start, key.span.column_start) == (8, 77)

    quoted = json.dumps({"command": (
        r'vllm serve /m --compilation-config '
        r'"{\"cudagraph_mode\":\"FULL\"}"')}, indent=2)
    report = _parse(quoted, format="json")
    key = _setting(report, "cudagraph_mode")
    assert (key.span.line_start, key.span.column_start) == (2, 56)
    assert quoted.splitlines()[1][key.span.column_start - 1] == '"'


# ---------------------------------------------------------------- P2-19
def test_p2_19_one_line_index_per_document_and_linear_scan():
    import mindie_sim.launch_spans as spans

    original = spans._Lines.__init__
    calls = [0, 0]

    def counting(self, text):
        calls[0] += 1
        calls[1] += len(text)
        original(self, text)

    spans._Lines.__init__ = counting
    try:
        payload = json.dumps({
            "notes": ["abcdefghij"] * 4000,
            "command": ["vllm", "serve", "/m"],
        })
        document = parse_document(payload, format="json")
    finally:
        spans._Lines.__init__ = original
    assert document.argv == ("vllm", "serve", "/m")
    assert calls[0] <= 2
    assert calls[1] <= 2 * len(payload)
