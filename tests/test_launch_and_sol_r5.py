"""Round-5 regressions for every open `astra_review_r5.md` item.

[Role] Turn each Round-5 counterexample and its nearby variants into a check
  of the public launcher, SOL projection, Engine parity and report contracts.
[Invariants] Expected values come from declared work, an actual Engine ledger
  or the report's own validator; no substring-grep filler and no host defaults.
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
from mindie_sim.launch_documents import parse_document
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineTemplate, PipelineTemplateProgram, Stage,
)
from mindie_sim.prediction.sol import bound, compare_sol
from mindie_sim.prediction.sol_work import Peak, project_ops, project_program
from mindie_sim.schema.case import Case
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
CANARY = "R5_CANARY"
VERSION = "vllm-ascend@0.30.0"


def _yaml_available():
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    for name in ("HOME", "MINDIE_CACHE_HOME", "MINDIE_DATA_HOME", "MINDIE_SIM_HOME",
                 "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    for name in ("VLLM_ASCEND_ENABLE_MLAPO", "HCCL_BUFFSIZE", "VLLM_ALL2ALL_BACKEND",
                 "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _op(name, deps=(), flops=0.0, hbm=0.0, vec=0.0, link=0.0, stage=0,
        stream="", metadata=(), family="matmul"):
    return OpSpec(op_id=name, family=family, phase="decode", stage=stage, layer=0,
                  requests=("r0",), semantic_shape=(1, 1), local_shape=(1, 1),
                  padded_shape=(1, 1), launch_shape=(1, 1), flops=flops,
                  hbm_bytes=hbm, vector_ops=vec, link_bytes=link, deps=tuple(deps),
                  dtype="bf16", stream=stream, metadata=tuple(metadata))


def _case(model="qwen3-30b-a3b", **kwargs):
    kwargs.setdefault("layers", 1)
    kwargs.setdefault("graph_capture_mode", "none")
    return Case.from_mapping(compose_case(model, hardware="a3", **kwargs))


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION, **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _context(plan_id, phase="decode"):
    return Context("c", plan_id, "model_step", phase, 1, 2, 128,
                   "b", "complete", (SourceIdentity("fixture"),))


@pytest.fixture
def lower_program():
    saved = Execution.__dict__["lower_program"]

    def install(program):
        Execution.lower_program = staticmethod(lambda context, recipe: program)

    yield install
    Execution.lower_program = saved


# ---------------------------------------------------------------- P1-01
def test_p1_01_scanner_errors_and_structured_shell_keep_secrets():
    script = (f"API_KEY={CANARY}\n"
              "vllm serve /m --x ${R5_CANARY-bad}\n")
    failed = _parse(script)
    assert failed.status == "failed"
    assert failed.diagnostics[-1].message == "parameter_expansion_not_supported"
    for rendered in (canonical_json(failed), launch_markdown(failed)):
        assert CANARY not in rendered

    payload = json.dumps({"command": (
        f"API_KEY={CANARY}\n"
        "vllm serve /m --additional-config "
        "'{\"R5_CANARY_api_key\":1}'")})
    structured = _parse(payload, format="json")
    assert structured.status == "partial"
    for rendered in (canonical_json(structured), launch_markdown(structured)):
        assert CANARY not in rendered
    document = parse_document(payload, format="json")
    assert CANARY in document.secret_values

    # Variant: structured env, plus a secret-named nested argument value.
    variant = _parse(json.dumps({
        "env": {"API_KEY": CANARY},
        "command": "vllm serve /m --x ${API_KEY}"}), format="json")
    assert CANARY not in canonical_json(variant)
    assert CANARY not in launch_markdown(variant)


# ---------------------------------------------------------------- P1-08
def _template(**kwargs):
    return PipelineTemplateProgram(PipelineTemplate(
        0, 1, (PipelinePhase("body", (Stage("z"),), 1),), **kwargs))


def test_p1_08_setup_and_drain_follow_after_dependencies(lower_program):
    case = _case()
    chain = (Stage("a", {"memory@r0:s0": 1.6e9}),
             Stage("b", {"vector@r0:s0": 2.7648e9}, after=("a",)))
    for field in ("setup", "drain"):
        lower_program(_template(**{field: chain}))
        plan = project_ops(case, (_op("x"),))
        predicted = graph_costs(case, (_op("x"),), compact=True,
                                isolated=False)["duration_ms"]
        assert plan.coverage == "complete" and not plan.holes
        assert plan.terms[0].phase_precedence_ms == pytest.approx(2.0)
        assert plan.terms[0].ideal_phase_precedence_ms == pytest.approx(2.0)
        assert bound(plan).sol_ms == pytest.approx(2.0)
        assert bound(plan, "declared_physical").sol_ms == pytest.approx(2.0)
        assert predicted == pytest.approx(2.0)

    # Unresolved intra-phase references are holes, exactly like KernelProgram.
    lower_program(_template(setup=(Stage("b", {"vector@r0:s0": 2.7648e9},
                                           after=("missing",)),)))
    plan = project_ops(case, (_op("x"),))
    assert plan.coverage != "complete"
    assert any("program_dependency_unresolved:b:missing" in hole
               for hole in plan.holes)


# ---------------------------------------------------------------- P1-11
def test_p1_11_ideal_zero_excludes_declared_minimum(lower_program):
    case = _case()
    program = KernelProgram((Stage("a", {"memory@r0:s0": 1.6e9}),),
                            minimum_ms=100)
    lower_program(program)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True,
                            isolated=False)["duration_ms"]
    assert bound(plan).sol_ms == pytest.approx(1.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(100.0)
    assert predicted == pytest.approx(100.0)

    peaks = (Peak("hbm", "bytes", 1.6e9, "", "fixture"),)
    direct = project_program(program, peaks)
    assert bound(direct).sol_ms == pytest.approx(1.0)
    assert bound(direct, "declared_physical").sol_ms == pytest.approx(100.0)

    nested = KernelProgram((
        Stage("pre", {"memory@r0:s0": 3.2e9}),
        Stage("nested", program=KernelProgram((), minimum_ms=1),
              after=("pre",))),)
    lower_program(nested)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True,
                            isolated=False)["duration_ms"]
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(3.0)
    assert predicted == pytest.approx(3.0)


def _regime_case(provenance, *, include=True):
    case = _case()
    rule = {"id": "measured100", "match": {"family": "matmul"},
            "response": {"kind": "affine", "fixed_us": 100000, "terms": []}}
    if include:
        rule["provenance"] = provenance
    return replace(case, runtime=replace(case.runtime, kernels={
        **dict(case.runtime.kernels), "service_regimes": [rule]}))


def test_p1_11_physical_admission_requires_affirmative_provenance():
    op = _op("matmul", hbm=1.6e9)
    for provenance, include in (
            ({"observation_source": "measured100.csv"}, True),
            ({"profile_source": "measured100.csv"}, True),
            ({"basis": "measured"}, True),
            ({"basis": "declared_physical", "observation_source": "x.csv"}, True),
            ({}, True),
            ({}, False)):
        case = _regime_case(provenance, include=include)
        plan = project_ops(case, (op,))
        ideal = bound(plan)
        physical = bound(plan, "declared_physical")
        predicted = graph_costs(case, (op,), compact=True,
                                isolated=False)["duration_ms"]
        assert ideal.sol_ms == pytest.approx(1.0)
        assert physical.sol_ms == pytest.approx(1.0)
        assert physical.empirical_floor_ms == pytest.approx(100.0)
        assert "declared_program_minimum" not in physical.floor_bases
        assert predicted == pytest.approx(100.0)

    for basis in ("declared", "architectural_spec", "hardware_spec"):
        case = _regime_case({"basis": basis})
        plan = project_ops(case, (op,))
        assert bound(plan).sol_ms == pytest.approx(1.0)
        assert bound(plan, "declared_physical").sol_ms == pytest.approx(100.0)
        assert "declared_program_minimum" in bound(
            plan, "declared_physical").floor_bases


# ---------------------------------------------------------------- P1-15
def _tpot(trace, steps=1, commits=2.8):
    plan = project_program(KernelProgram((Stage("a", {"hbm": 1}),)),
                           (Peak("hbm", "bytes", 1.0, "declared", "fixture"),))
    report = compare_sol(plan, 2.0, context=_context(plan.plan_id),
                         request_metrics_enabled=True, commits=commits,
                         request_trace={"decode_steps": steps, **trace})
    assert report.bound.sol_ms == pytest.approx(1.0)
    return report.request_metrics[1]


def test_p1_15_commit_boundaries_are_one_fail_closed_gate():
    non_increasing = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                            "commit_times": [0, 10, 5]}, steps=2)
    assert non_increasing.status == "unavailable"
    assert "increase" in non_increasing.reason

    equal = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                   "commit_times": [0, 0, 10]}, steps=2)
    assert equal.status == "unavailable"
    assert "increase" in equal.reason

    for field in ("commit_times", "step_commit_times", "commit_events"):
        mixed = _tpot({"arrival_ms": 0, "first_commit_ms": 0, field: [
            {"request_id": "rA", "time_ms": 0},
            {"request_id": "rB", "time_ms": 10}]}, steps=1)
        assert mixed.status == "unavailable", field
        assert mixed.reason

    valid = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                   "step_commit_times": [
                       {"request_id": "rA", "time_ms": 0},
                       {"request_id": "rA", "time_ms": 10}]}, steps=1)
    assert valid.status == "available"
    assert valid.predicted_ms == pytest.approx(10.0 / 2.8)


# ---------------------------------------------------------------- P2-18
def test_p2_18_explicit_zero_visibility_is_not_undeclared():
    for mask in ("2", "2,3", ""):
        script = ("docker run --device /dev/davinci0 --device /dev/davinci1 "
                  f"-e ASCEND_RT_VISIBLE_DEVICES={mask} image vllm serve /m "
                  "--tensor-parallel-size 2")
        report = _parse(script)
        assert report.status == "partial", mask
        assert any(item.name == "visible_device_count"
                   for item in report.unmodeled_or_unsupported), mask

    document = parse_document(
        "docker run --device /dev/davinci0 --device /dev/davinci1 "
        "-e ASCEND_RT_VISIBLE_DEVICES=2 image vllm serve /m",
        format="shell")
    assert document.device_visibility_declared is True
    assert document.visible_devices == ()

    # Variant: the same explicit mask outside Docker still resolves its rank.
    outside = _parse("vllm serve /m --tensor-parallel-size 2", env={
        "ASCEND_RT_VISIBLE_DEVICES": "0"})
    assert outside.status == "partial"
    assert any(item.name == "visible_device_count"
               for item in outside.unmodeled_or_unsupported)


# ---------------------------------------------------------------- P2-20
def test_p2_20_requirements_survive_all_document_failures():
    cases = (
        ("docker run -e MISSING -e", "needs_inputs"),
        ("docker run -e MISSING image unknown", "failed"),
        ("docker run -e MISSING --unknown image vllm serve /m", "failed"),
    )
    for script, status in cases:
        report = _parse(script)
        assert report.status == status, script
        assert "env:MISSING" in {item.name for item in report.required_inputs}, script

    without_geometry = _parse("docker run -e MISSING image unknown",
                              machine=None, model=None, workload=None)
    names = {item.name for item in without_geometry.required_inputs}
    assert {"env:MISSING", "machine", "model", "workload"} <= names


# ---------------------------------------------------------------- P2-23
def test_p2_23_structural_spans_ignore_decoys_and_field_order():
    command = ["vllm", "serve", "/m", "--compilation-config",
               '{"cudagraph_mode":"FULL"}']
    payload = json.dumps({"notes": command, "containers": [
        {"command": command}, {"command": command}]}, indent=2)
    first = _parse(payload, format="json", container=0)
    second = _parse(payload, format="json", container=1)
    assert first.status == "completed" and second.status == "completed"
    assert _setting(first, "cudagraph_mode").span.line_start == 16
    assert _setting(second, "cudagraph_mode").span.line_start == 25

    reordered = json.dumps({"args": command[3:], "command": command[:3]},
                           indent=2)
    report = _parse(reordered, format="json")
    assert report.status == "completed"
    leaf = _setting(report, "cudagraph_mode")
    assert leaf.span.line_start == 4

    # Nested JSON flag offsets come from the selected scalar, not a text find.
    argv = ["vllm", "serve", "/m",
            '--compilation-config={"cudagraph_mode":"FULL"}']
    attached = json.dumps({"command": argv})
    report = _parse(attached, format="json")
    leaf = _setting(report, "cudagraph_mode")
    assert leaf.span.column_start == 61
    assert attached[leaf.span.column_start - 1] == '"'
    assert attached[leaf.span.column_start - 2] == "\\"

    yaml_text = ("containers:\n"
                 "  - command: [\"vllm\", \"serve\", \"/m\", "
                 "\"--compilation-config\", '{\"cudagraph_mode\":\"FULL\"}']\n"
                 "  - command: [\"vllm\", \"serve\", \"/m\", "
                 "\"--compilation-config\", '{\"cudagraph_mode\":\"FULL\"}']\n")
    yaml_report = _parse(yaml_text, format="yaml", container=1)
    if not _yaml_available():
        assert yaml_report.diagnostics[-1].code == "yaml_parser_unavailable"
        return
    assert yaml_report.status == "completed"
    assert _setting(yaml_report, "cudagraph_mode").span.line_start == 3
