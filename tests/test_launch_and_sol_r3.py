"""Round-3 behavioral regressions for every open `astra_review_r3.md` item.

[Role] Turn each exact Round-3 counterexample into a check of the public
  parser, bindings, SOL projection, Engine conservation and report contracts.
[Invariants] Isolated HOME/MINDIE state, no host env defaults, no network, no
  string-grep filler: expected values come from declared work, an actual
  Engine ledger or the report's own semantic validator.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] Writes only under pytest's tmp_path.
"""
from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import json

import pytest

from mindie_sim import compose_case
from mindie_sim.feature_cli import main as cli_main
from mindie_sim.feature_reports import (
    Context, ReportValidationError, RequestMetric, SourceIdentity, canonical_json,
    launch_markdown, validate_sol_report,
)
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_documents import parse_document
from mindie_sim.launch_parser import LaunchSyntaxError, parse_shell, scan, selected_command
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineTemplate, PipelineTemplateProgram, Stage,
)
from mindie_sim.prediction.sol import bound, compare_sol
from mindie_sim.prediction.sol_work import (
    Peak, PlanWork, WorkTerm, _program_work, engine_peaks, project_ops,
    project_program,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
CANARY = "R3_CANARY"
VERSION = "vllm-ascend@0.30.0"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    for name in ("HOME", "MINDIE_CACHE_HOME", "MINDIE_DATA_HOME", "MINDIE_SIM_HOME",
                 "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    for name in ("VLLM_ASCEND_ENABLE_MLAPO", "HCCL_BUFFSIZE", "VLLM_ALL2ALL_BACKEND"):
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
    return parse_launch(text, **{**BASE, **kwargs})


def _engine(program):
    engine = Engine([Resource("hbm", 1), Resource("vector", 1)])
    ready = engine.submit("ready")
    end = program.start(engine, "program", ready)
    engine.advance(until=(end,))
    return end.time, dict(engine.serviced)


def _context(plan_id="p", phase="decode"):
    return Context("c", plan_id, "model_step", phase, 1, 2, 128,
                   "b", "complete", (SourceIdentity("fixture"),))


# ---------------------------------------------------------------- P1-01
def test_p1_01_secret_values_are_scrubbed_from_every_report_surface():
    cases = {
        "shell_local": "API_KEY=" + CANARY + "\n$API_KEY",
        "docker_option": "export API_KEY=" + CANARY
                         + "\ndocker run --$API_KEY image vllm serve /m",
        "required_name": 'docker run -e "$API_KEY" image vllm serve /m',
        "json_key": 'vllm serve /m --additional-config \'{"%s":1}\'' % CANARY,
    }
    for name, text in cases.items():
        report = _parse(text, env={"API_KEY": CANARY}, framework_version=VERSION)
        encoded = canonical_json(report)
        markdown = launch_markdown(report)
        assert CANARY not in encoded, name
        assert CANARY not in markdown, name
        assert "[REDACTED]" in encoded or report.status == "failed", name
    required = _parse(cases["required_name"], env={"API_KEY": CANARY})
    assert required.status == "needs_inputs"
    assert [item.name for item in required.required_inputs] == ["env:[REDACTED]"]
    keyed = _parse(cases["json_key"], env={"API_KEY": CANARY},
                   framework_version=VERSION)
    assert keyed.status == "partial"
    assert all(CANARY not in str(item.id) and CANARY not in str(item.name)
               and CANARY not in str(item.pointer)
               for item in keyed.unmodeled_or_unsupported)
    local = _parse(cases["shell_local"])
    assert local.status == "failed" and "[REDACTED]" in local.diagnostics[0].message
    docker = _parse(cases["docker_option"])
    assert docker.status == "failed" and "[REDACTED]" in docker.diagnostics[0].message


# ---------------------------------------------------------------- P1-03
def test_p1_03_explicit_env_is_outer_scope_for_docker():
    kept = parse_launch("docker run image vllm serve /m", **BASE,
                        env={"VLLM_ALL2ALL_BACKEND": "mc2"})
    assert kept.case["runtime"]["moe_recipe"] != "mc2"
    assert not any(item.name == "VLLM_ALL2ALL_BACKEND"
                   for item in (*kept.applied, *kept.neutral_operational,
                                *kept.unmodeled_or_unsupported))
    for script, kwargs in (
            ("docker run -e VLLM_ALL2ALL_BACKEND image vllm serve /m", {}),
            ("docker run -e VLLM_ALL2ALL_BACKEND=mc2 image vllm serve /m", {}),
            ("docker run --env-file values.env image vllm serve /m",
             {"env_files": {"values.env": {"VLLM_ALL2ALL_BACKEND": "mc2"}}}),
    ):
        report = parse_launch(script, **BASE,
                              env={"VLLM_ALL2ALL_BACKEND": "mc2"}, **kwargs)
        assert report.status == "completed"
        assert report.case["runtime"]["moe_recipe"] == "mc2"
    direct = parse_launch("vllm serve /m", **BASE,
                          env={"VLLM_ALL2ALL_BACKEND": "mc2"})
    assert direct.case["runtime"]["moe_recipe"] == "mc2"


# ---------------------------------------------------------------- P1-04
def test_p1_04_enforce_eager_shadows_compilation_independent_of_order():
    for version in ("vllm-ascend@0.18.0", VERSION):
        for text in (
                "vllm serve /m --enforce-eager --compilation-config "
                "'{\"cudagraph_mode\":\"FULL\"}'",
                "vllm serve /m --compilation-config '{\"cudagraph_mode\":\"FULL\"}' "
                "--enforce-eager",
        ):
            report = parse_launch(text, **BASE, framework_version=version)
            assert report.status == "completed"
            runtime = report.case["runtime"]
            assert runtime["graph_mode"] == "none"
            assert runtime["graph_capture_mode"] == "none"
            assert tuple(runtime["graph_capture_keys"]) == ()
            eager = next(item for item in report.applied
                         if item.name == "--enforce-eager")
            assert eager.effective
            compilation = next(item for item in (*report.applied,
                                                 *report.unmodeled_or_unsupported)
                               if item.name == "--compilation-config")
            assert not compilation.effective
            assert compilation.shadowed_by == eager.id
            child = next(item for item in (*report.applied,
                                           *report.unmodeled_or_unsupported)
                         if item.name == "cudagraph_mode")
            assert not child.effective and child.shadowed_by == eager.id
            assert any(item.code == "enforce_eager_overrides_compilation"
                       for item in report.diagnostics)


# ---------------------------------------------------------------- P1-08
@pytest.fixture
def lower_program():
    """Install a static lowered program without corrupting the descriptor."""
    saved = Execution.__dict__["lower_program"]

    def install(program):
        Execution.lower_program = staticmethod(
            lambda context, recipe: program)

    yield install
    Execution.lower_program = saved


def test_p1_08_template_tiles_residues_and_nested_programs_are_conserved(lower_program):
    nested = PipelineTemplateProgram(PipelineTemplate(0, 1, (
        PipelinePhase("body", (Stage("outer", program=KernelProgram((
            Stage("inner", {"hbm": 1024}),))),), 2),)))
    work, holes, _floors = _program_work(nested, engine=Engine([Resource("hbm", 1),
                                                                Resource("vector", 1)]))
    assert dict(work)[("hbm", "")] == 2048.0
    assert not holes
    residue = PipelineTemplateProgram(PipelineTemplate(0, 1, (
        PipelinePhase("body", (Stage("a", {"hbm": 1}),), 2,
                      residue_stages=((Stage("a", {"hbm": 1}),),
                                      (Stage("a", {"hbm": 9}),))),)))
    work, holes, _floors = _program_work(residue, engine=Engine([
        Resource("hbm", 1), Resource("vector", 1)]))
    assert dict(work)[("hbm", "")] == 10.0
    assert not holes

    case = _case()
    ops = (_op("a"),)
    nested_public = PipelineTemplateProgram(PipelineTemplate(0, 1, (
        PipelinePhase("body", (Stage("outer", program=KernelProgram((
            Stage("inner", {"memory@r0:s0": 1024}),))),), 2),)))
    lower_program(nested_public)
    plan = project_ops(case, ops)
    assert plan.coverage == "complete"
    assert sum(amount for term in plan.terms for _r, amount, _u, _m in term.work) == 2048.0
    predicted = graph_costs(case, ops, compact=True, isolated=False)["duration_ms"]
    assert bound(plan).sol_ms == pytest.approx(1.28e-06)
    assert bound(plan).sol_ms == pytest.approx(predicted)

    serial = KernelProgram((Stage("a", {"memory@r0:s0": 1.6e9}),
                            Stage("b", {"vector@r0:s0": 2.7648e9}, after=("a",))))
    lower_program(serial)
    plan = project_ops(case, ops)
    predicted = graph_costs(case, ops, compact=True, isolated=False)["duration_ms"]
    assert plan.coverage == "complete"
    assert plan.terms[0].phase_precedence_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(predicted)


def test_p1_08_replay_scope_matches_engine_and_derating_provenance(monkeypatch):
    case = _case(graph_capture_mode="full")
    scopes = []
    original = Execution._context

    def capture(self, op, replica, scope, *, recipe=None):
        scopes.append(scope)
        return original(self, op, replica, scope, recipe=recipe)

    monkeypatch.setattr(Execution, "_context", capture)
    ops = (_op("a", flops=1e9,
               metadata=(("graph_replay_scope", "none"),)),)
    scopes.clear()
    predicted = graph_costs(case, ops, compact=True, isolated=False)
    engine_scopes = list(scopes)
    scopes.clear()
    plan = project_ops(case, ops)
    assert engine_scopes == ["none"]
    assert scopes == ["none"]
    assert plan.coverage == "complete"
    assert bound(plan).sol_ms <= predicted["duration_ms"] + 1e-12

    case = _case(tp=2, num_devices=4)
    machine = replace(case.machine, links=(
        Link("fabric", 1e8, (0, 1), bw_utilization=0.1,
             peak_capacity_bytes_per_ms=1e9),))
    case = replace(case, machine=machine)
    collective = _op("collective", link=1e9, family="collective",
                     metadata=(("group_axis", "tp"),))
    nominal = project_ops(case, (collective,))
    peaks = [item for item in nominal.peaks if item.resource_id == "link:fabric"]
    assert peaks and peaks[0].peak_per_ms == pytest.approx(1e9)
    assert peaks[0].source == "declared_uncalibrated_nominal_link_peak"
    assert bound(nominal).sol_ms == pytest.approx(2.0)
    assert bound(nominal).uses_empirical_derating is False
    derated = project_ops(case, (collective,), empirical_derating=True)
    peaks = [item for item in derated.peaks if item.resource_id == "link:fabric"]
    assert peaks and peaks[0].peak_per_ms == pytest.approx(1e8)
    assert peaks[0].source == "declared_empirically_derated_capacity"
    assert bound(derated).sol_ms == pytest.approx(20.0)
    assert bound(derated).uses_empirical_derating is True


# ---------------------------------------------------------------- P1-09
def test_p1_09_previous_is_not_a_kernel_edge_and_empty_nested_forwards():
    peaks = (Peak("hbm", "bytes", 1.0, "", "fixture"),
             Peak("vector", "ops", 1.0, "", "fixture"))
    previous = KernelProgram((Stage("a", {"hbm": 1}),
                              Stage("b", {"vector": 1}, previous=("a",))))
    plan = project_program(previous, peaks)
    assert plan.coverage == "complete"
    assert bound(plan).sol_ms == pytest.approx(1.0)
    assert bound(plan).sol_ms == pytest.approx(_engine(previous)[0])

    empty = KernelProgram((
        Stage("outer", {"hbm": 1}, program=KernelProgram(())),
        Stage("b", {"vector": 1}, after=("outer",))))
    plan = project_program(empty, peaks)
    assert plan.coverage == "complete"
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(_engine(empty)[0])


# ---------------------------------------------------------------- P1-11
def test_p1_11_measured_stage_latency_is_reported_not_inflated():
    peaks = (Peak("hbm", "bytes", 1.0, "", "fixture"),)
    measured = KernelProgram((Stage("a", {"hbm": 1}, latency=100),))
    plan = project_program(measured, peaks)
    assert bound(plan).sol_ms == pytest.approx(1.0)
    physical = bound(plan, "declared_physical")
    assert physical.sol_ms == pytest.approx(1.0)
    assert "measured_stage_latency" not in physical.floor_bases
    assert physical.empirical_floor_ms == pytest.approx(100.0)


# ---------------------------------------------------------------- P2-21
def test_p2_21_executed_fused_child_is_counted_until_elided(monkeypatch):
    import mindie_sim.prediction.graph_analysis as analysis
    seen = []
    original = analysis.Execution

    class Observed(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)

    case = _case()
    parent = _op("parent", hbm=4096)
    child = _op("child", hbm=1024, metadata=(
        ("fused_into", "parent"), ("fused_work_conserved", True)))
    consumer = _op("consumer", deps=("child",), hbm=1024)
    with monkeypatch.context() as patch:
        patch.setattr(analysis, "Execution", Observed)
        graph_costs(case, (parent, child, consumer), compact=True,
                    trace_limit=0, isolated=False)
    serviced = {name: amount for name, amount in seen[0].engine.serviced.items()
                if name.startswith("memory@")}
    plan = project_ops(case, (parent, child, consumer))
    projected = defaultdict(float)
    for term in plan.terms:
        for resource, amount, _unit, _mode in term.work:
            if resource.startswith("memory@"):
                projected[resource] += amount
    assert projected == pytest.approx(serviced)
    assert sum(projected.values()) == pytest.approx(6144.0)
    assert plan.coverage != "complete"
    assert any("fused_work_unproven" in hole for hole in plan.holes)
    assert bound(plan).status == "unavailable"


# ---------------------------------------------------------------- P1-15
def test_p1_15_tpot_requires_commit_boundaries_and_integer_steps():
    plan = project_program(KernelProgram((Stage("a", {"hbm": 1}),)),
                           (Peak("hbm", "bytes", 1.0, "", "fixture"),))
    context = _context(plan.plan_id)
    no_boundaries = compare_sol(
        plan, 2.0, context=context, request_metrics_enabled=True, commits=2.8,
        request_trace={"decode_steps": 10})
    assert no_boundaries.request_metrics[1].status == "unavailable"
    fractional = compare_sol(
        plan, 2.0, context=context, request_metrics_enabled=True, commits=2.8,
        request_trace={"decode_steps": 0.5,
                       "commit_times": [0, 1]})
    assert fractional.request_metrics[1].status == "unavailable"
    valid = compare_sol(
        plan, 2.0, context=context, request_metrics_enabled=True, commits=2.8,
        request_trace={"decode_steps": 10,
                       "commit_times": [float(index) for index in range(11)]})
    assert valid.request_metrics[1].status == "available"
    assert valid.request_metrics[1].predicted_ms == pytest.approx(10.0 / (10 * 2.8))
    negative = replace(valid, request_metrics=(
        RequestMetric("tpot", "available", "r", "fixed_trace",
                      "per_committed_token", -2.0, -1.0, True, None),))
    with pytest.raises(ReportValidationError) as excinfo:
        validate_sol_report(negative)
    assert "nonpositive_latency" in str(excinfo.value)


# ---------------------------------------------------------------- P2-17
def test_p2_17_double_quoted_ansi_literal_is_not_rejected():
    command = selected_command(parse_shell(
        'vllm serve /m --served-model-name "$' + "'literal'\""))
    assert command.argv[-1] == "$'literal'"
    report = _parse('vllm serve /m --served-model-name "$' + "'literal'\"")
    assert report.status == "completed"
    assert next(item.value for item in report.neutral_operational
                if item.name == "--served-model-name") == "$'literal'"
    for script in ("vllm serve /m --x $'ab\\ncd'", 'vllm serve /m --x $"ab"'):
        failed = _parse(script)
        assert failed.status == "failed"


# ---------------------------------------------------------------- P2-18
def test_p2_18_npu_mappings_dedupe_targets_and_unidentified_devices():
    deduped = _parse("docker run --device /dev/davinci0 "
                     "-e ASCEND_RT_VISIBLE_DEVICES=0 image vllm serve /m "
                     "--tensor-parallel-size 2")
    assert deduped.status != "completed"
    assert any(item.name == "visible_device_count"
               for item in deduped.unmodeled_or_unsupported)
    one = _parse("docker run --device /dev/davinci0 "
                 "-e ASCEND_RT_VISIBLE_DEVICES=0 image vllm serve /m")
    assert one.status == "completed"
    mapped = _parse("docker run --device /host/davinci2:/dev/davinci0 "
                    "image vllm serve /m")
    assert mapped.status == "completed"
    enough = _parse("docker run --device /host/davinci2:/dev/davinci0 "
                    "--device /host/davinci3:/dev/davinci1 image vllm serve /m "
                    "--tensor-parallel-size 2")
    assert enough.status == "completed"
    bare = _parse("docker run --device /dev/npu image vllm serve /m")
    assert bare.status == "partial"
    assert any(item.rule_id == "visible_device_identity_unverified"
               for item in bare.unmodeled_or_unsupported)


# ---------------------------------------------------------------- P2-19
def test_p2_19_newline_tokens_obey_the_scanner_cap():
    with pytest.raises(LaunchSyntaxError) as excinfo:
        scan("vllm serve /m\n" + "\n" * 100_001)
    assert excinfo.value.code == "input_too_large"
    report = _parse("vllm serve /m\n" + "\n" * 100_001)
    assert report.status == "failed"
    assert report.diagnostics[0].code == "input_too_large"


# ---------------------------------------------------------------- P2-20
def test_p2_20_terminal_env_requirements_and_accumulated_inputs(tmp_path):
    for script, expected in (("docker run -e", "env"),
                             ("docker run --env-file", "env_file")):
        report = _parse(script)
        assert report.status == "needs_inputs"
        assert expected in {item.name.split(":")[1]
                            for item in report.required_inputs
                            if item.name.startswith("env")}
    malformed = parse_launch("docker run -e MISSING image vllm serve /m "
                             "--max-num-seqs nope")
    assert malformed.status == "needs_inputs"
    names = {item.name for item in malformed.required_inputs}
    assert {"machine", "model", "workload", "env:MISSING"} <= names
    assert any(item.code == "invalid_declaration" for item in malformed.diagnostics)
    broken_json = parse_launch('vllm serve /m --additional-config "{bad}"')
    assert {"machine", "model", "workload"} <= {
        item.name for item in broken_json.required_inputs}
    invalid_override = parse_launch(
        "docker run -e MISSING image vllm serve /m", base_case=_case(),
        workload=WORKLOAD, overrides={"/runtime/parallel/tp": 999})
    assert "env:MISSING" in {item.name for item in invalid_override.required_inputs}
    script = tmp_path / "deploy.sh"
    script.write_text("docker run -e\n", encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD), encoding="utf-8")
    assert cli_main(["inspect-launch", str(script), "--machine", "a3",
                     "--model", "qwen3-30b-a3b", "--workload", str(workload)]) == 3


# ---------------------------------------------------------------- P2-23
def test_p2_23_attached_json_flag_has_exact_argv_and_json_columns():
    argv = ["vllm", "serve", "/m", '--compilation-config={"cudagraph_mode":"FULL"}']
    report = parse_launch(argv, format="argv", **BASE, framework_version=VERSION)
    leaf = next(item for item in report.applied if item.name == "cudagraph_mode")
    assert leaf.span.column_start == 37
    payload = json.dumps({"command": argv})
    report = parse_launch(payload, format="json", **BASE,
                          framework_version=VERSION)
    leaf = next(item for item in report.applied if item.name == "cudagraph_mode")
    assert leaf.span.column_start == 61
    assert payload[leaf.span.column_start - 1] == '"'
    assert payload[leaf.span.column_start - 2] == "\\"
