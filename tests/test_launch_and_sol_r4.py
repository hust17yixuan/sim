"""Round-4 regressions for every open `astra_review_r4.md` item.

[Role] Turn each exact Round-4 counterexample plus a nearby variant into a
  check of the public launcher, SOL projection, Engine parity and report
  contracts; keep the fifteen previously closed findings untouched.
[Invariants] Expected values come from declared work, an actual Engine ledger
  or the report's own semantic validator; no substring-grep filler and no host
  environment defaults.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] Writes only under pytest's tmp_path; no network.
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
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineTemplate, PipelineTemplateProgram, Stage,
)
from mindie_sim.prediction.sol import bound, compare_sol
from mindie_sim.prediction.sol_work import Peak, project_ops, project_program
from mindie_sim.schema.case import Case
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
CANARY = "R4_CANARY"
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
    return parse_launch(text, **{**BASE, "framework_version": VERSION, **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _context(plan_id="p", phase="decode"):
    return Context("c", plan_id, "model_step", phase, 1, 2, 128,
                   "b", "complete", (SourceIdentity("fixture"),))


@pytest.fixture
def lower_program():
    """Install one static lowered program without corrupting the descriptor."""
    saved = Execution.__dict__["lower_program"]

    def install(program):
        Execution.lower_program = staticmethod(lambda context, recipe: program)

    yield install
    Execution.lower_program = saved


# ---------------------------------------------------------------- P1-01
def test_p1_01_secret_substrings_in_keys_and_export_errors():
    report = _parse("vllm serve /m --additional-config "
                    "'{\"R4_CANARY_api_key\":1}'", env={"API_KEY": CANARY})
    assert report.status == "partial"
    for rendered in (canonical_json(report), launch_markdown(report)):
        assert CANARY not in rendered

    script = (f"API_KEY={CANARY}\n"
              "export $API_KEY\n"
              "vllm serve /m\n")
    failed = _parse(script)
    assert failed.status == "failed"
    assert CANARY not in canonical_json(failed)
    assert CANARY not in launch_markdown(failed)
    assert failed.diagnostics[-1].message == "export_unknown_variable:[REDACTED]"

    # Variant: the secret came from explicit `env=` and the failing statement
    # is an export expansion rather than a shell-local assignment.
    explicit = _parse("export $API_KEY\nvllm serve /m\n",
                      env={"API_KEY": CANARY})
    assert explicit.status == "failed"
    assert CANARY not in canonical_json(explicit)
    assert explicit.diagnostics[-1].message == "export_unknown_variable:[REDACTED]"

    # Variant: a nested, non-secret-named key still carries the substring.
    nested = _parse("vllm serve /m --additional-config "
                    "'{\"nested\":{\"R4_CANARY_api_key\":1}}'",
                    env={"API_KEY": CANARY})
    assert nested.status == "partial"
    assert CANARY not in canonical_json(nested)
    assert CANARY not in launch_markdown(nested)


# ---------------------------------------------------------------- P1-04
def test_p1_04_boolean_eager_spellings_share_one_shadow_key():
    report = _parse("vllm serve /m --enforce-eager --enforce-eager=false "
                    "--compilation-config '{\"cudagraph_mode\":\"FULL\"}'")
    assert report.status == "completed"
    eager = [item for item in _settings(report) if item.name == "--enforce-eager"]
    assert {(item.value, item.effective) for item in eager} == {
        (True, False), (False, True)}
    disabled = next(item for item in eager if item.value is False)
    shadowed = next(item for item in eager if item.value is True)
    assert shadowed.shadowed_by == disabled.id
    runtime = report.case["runtime"]
    assert runtime["graph_mode"] == "full"
    assert runtime["graph_capture_mode"] == "full"
    mode = _setting(report, "cudagraph_mode")
    assert mode.value == "FULL" and mode.effective
    assert not any(item.code == "enforce_eager_overrides_compilation"
                   for item in report.diagnostics)

    # Variant: the later true occurrence still wins under the shared key.
    reverse = _parse("vllm serve /m --enforce-eager=false --enforce-eager "
                     "--compilation-config '{\"cudagraph_mode\":\"FULL\"}'")
    assert reverse.case["runtime"]["graph_mode"] == "none"
    eager = [item for item in _settings(reverse) if item.name == "--enforce-eager"]
    assert {(item.value, item.effective) for item in eager} == {
        (True, True), (False, False)}
    assert not _setting(reverse, "cudagraph_mode").effective

    # Variant: a lone disabled spelling never forces eager graph mode.
    alone = _parse("vllm serve /m --enforce-eager=false "
                   "--compilation-config '{\"cudagraph_mode\":\"FULL\"}'")
    assert alone.case["runtime"]["graph_mode"] == "full"
    assert _setting(alone, "cudagraph_mode").effective


# ---------------------------------------------------------------- P1-08
def _template(stages, tiles=1):
    return PipelineTemplateProgram(PipelineTemplate(
        0, 1, (PipelinePhase("body", stages, tiles),)))


def test_p1_08_template_after_edges_conserve_serial_phase_path(lower_program):
    case = _case()
    program = _template((Stage("a", {"memory@r0:s0": 1.6e9}),
                         Stage("b", {"vector@r0:s0": 2.7648e9},
                               after=("a",))))
    lower_program(program)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True, isolated=False)["duration_ms"]
    assert plan.coverage == "complete" and not plan.holes
    assert plan.terms[0].phase_precedence_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(predicted)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(2.0)

    # Variant: a three-stage serial chain is a 3 ms path, not a max of 1 ms.
    chain = _template((Stage("a", {"memory@r0:s0": 1.6e9}),
                       Stage("b", {"vector@r0:s0": 2.7648e9}, after=("a",)),
                       Stage("c", {"memory@r0:s0": 1.6e9}, after=("b",))))
    lower_program(chain)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True, isolated=False)["duration_ms"]
    assert plan.terms[0].phase_precedence_ms == pytest.approx(3.0)
    assert bound(plan).sol_ms == pytest.approx(3.0)
    assert bound(plan).sol_ms == pytest.approx(predicted)


def test_p1_08_replay_scope_and_derated_link_without_nominal_peak(monkeypatch):
    case = _case(graph_capture_mode="full")
    scopes = []
    original = Execution._context

    def capture(self, op, replica, scope, *, recipe=None):
        scopes.append(scope)
        return original(self, op, replica, scope, recipe=recipe)

    monkeypatch.setattr(Execution, "_context", capture)
    for declared, expected in ((None, "none"), ("full", "full"), ("none", "none")):
        metadata = (() if declared is None
                    else (("graph_replay_scope", declared),))
        ops = (_op("custom", flops=1e9, metadata=metadata),)
        scopes.clear()
        graph_costs(case, ops, compact=True, isolated=False)
        engine_scopes = list(scopes)
        scopes.clear()
        project_ops(case, ops)
        assert engine_scopes == scopes == [expected]

    # A derated link with no declared nominal peak has no physical capability
    # to price; it must not fabricate an available 20 ms bound.
    linked = _case(tp=2, num_devices=4)
    linked = replace(linked, machine=replace(
        linked.machine, links=(Link("fabric", 1e8, (0, 1), bw_utilization=0.1),)))
    collective = _op("collective", link=2e9, family="collective",
                     metadata=(("group_axis", "tp"),))
    plan = project_ops(linked, (collective,))
    peaks = [item for item in plan.peaks if item.resource_id == "link:fabric"]
    assert peaks and peaks[0].peak_per_ms == 0.0
    assert peaks[0].source == "missing_nominal_link_peak"
    physical = bound(plan, "declared_physical")
    assert physical.status == "unavailable"
    assert any("missing_nominal_link_peak" in reason
               for reason in physical.unavailable_reasons)
    assert physical.uses_empirical_derating is False

    # Variant: an explicit nominal peak still yields the declared 2 ms.
    declared = Link("fabric", 1e8, (0, 1), bw_utilization=0.1,
                    peak_capacity_bytes_per_ms=1e9)
    case = _case(tp=2, num_devices=4)
    case = replace(case, machine=replace(case.machine, links=(declared,)))
    nominal = project_ops(case, (_op("collective", link=1e9, family="collective",
                                     metadata=(("group_axis", "tp"),)),))
    assert bound(nominal, "declared_physical").sol_ms == pytest.approx(2.0)


# ---------------------------------------------------------------- P1-11
def _regime_case(basis, fixed_us=100000):
    case = _case()
    rule = {"id": "measured100", "match": {"family": "matmul"},
            "response": {"kind": "affine", "fixed_us": fixed_us, "terms": []},
            "provenance": {"basis": basis}}
    return replace(case, runtime=replace(case.runtime, kernels={
        **dict(case.runtime.kernels), "service_regimes": [rule]}))


def test_p1_11_measured_service_regime_floor_is_not_physical():
    op = _op("matmul", hbm=1.6e9)
    measured = _regime_case("measured")
    plan = project_ops(measured, (op,))
    physical = bound(plan, "declared_physical")
    assert plan.coverage == "complete" and not plan.holes
    assert bound(plan).sol_ms == pytest.approx(1.0)
    assert physical.sol_ms == pytest.approx(1.0)
    assert physical.empirical_floor_ms == pytest.approx(100.0)
    assert "declared_program_minimum" not in physical.floor_bases
    predicted = graph_costs(measured, (op,), compact=True, isolated=False)["duration_ms"]
    assert predicted == pytest.approx(100.0)

    profiled = project_ops(_regime_case("profiled"), (op,))
    assert bound(profiled, "declared_physical").sol_ms == pytest.approx(1.0)
    assert bound(profiled, "declared_physical").empirical_floor_ms == pytest.approx(100.0)

    # Variant: a declared (non-empirical) response remains physical.
    declared = project_ops(_regime_case("declared"), (op,))
    physical = bound(declared, "declared_physical")
    assert physical.sol_ms == pytest.approx(100.0)
    assert "declared_program_minimum" in physical.floor_bases


def test_p1_11_nested_program_minimum_keeps_admission_predecessor(lower_program):
    case = _case()
    program = KernelProgram((
        Stage("pre", {"memory@r0:s0": 3.2e9}),
        Stage("nested", program=KernelProgram((), minimum_ms=1), after=("pre",))))
    lower_program(program)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True, isolated=False)["duration_ms"]
    assert plan.coverage == "complete" and not plan.holes
    assert plan.terms[0].phase_precedence_ms == pytest.approx(3.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(3.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(predicted)

    # Variant: a top-level minimum is admission-relative (parallel with work),
    # not terminal, so 2 ms of work plus a 1 ms observation stays 2 ms.
    top = KernelProgram((Stage("a", {"memory@r0:s0": 3.2e9}),), minimum_ms=1)
    lower_program(top)
    plan = project_ops(case, (_op("x"),))
    predicted = graph_costs(case, (_op("x"),), compact=True, isolated=False)["duration_ms"]
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(2.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(predicted)


# ---------------------------------------------------------------- P1-15
def _tpot(trace, steps=1, commits=2.8):
    plan = project_program(KernelProgram((Stage("a", {"hbm": 1}),)),
                           (Peak("hbm", "bytes", 1.0, "declared", "fixture"),))
    report = compare_sol(plan, 2.0, context=_context(plan.plan_id),
                         request_metrics_enabled=True, commits=commits,
                         request_trace={"decode_steps": steps, **trace})
    assert report.bound.sol_ms == pytest.approx(1.0)
    return report.request_metrics[1]


def test_p1_15_trace_chronology_and_single_request_identity():
    before_arrival = _tpot({"arrival_ms": 0, "first_commit_ms": 5,
                            "commit_times": [-20, -10]})
    assert before_arrival.status == "unavailable"
    assert "arrival" in before_arrival.reason

    mixed = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                   "commit_events": [{"request_id": "rA", "time_ms": 0},
                                     {"request_id": "rB", "time_ms": 10}]})
    assert mixed.status == "unavailable"
    assert "request" in mixed.reason

    mismatch = _tpot({"arrival_ms": 0, "first_commit_ms": 5,
                      "commit_times": [4, 6]})
    assert mismatch.status == "unavailable"
    assert "first_commit_ms" in mismatch.reason

    malformed = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                       "commit_events": [{"request_id": "rA", "time_ms": 0},
                                         {"request_id": "rA"}]})
    assert malformed.status == "unavailable"

    partial_identity = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                              "commit_events": [{"request_id": "rA", "time_ms": 0},
                                                {"time_ms": 10}]})
    assert partial_identity.status == "unavailable"

    null_identity = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                           "commit_events": [{"request_id": None, "time_ms": 0},
                                             {"request_id": None, "time_ms": 10}]})
    assert null_identity.status == "unavailable"

    early = _tpot({"arrival_ms": 15, "first_commit_ms": 15,
                   "commit_times": [10, 20]})
    assert early.status == "unavailable"
    assert "arrival" in early.reason

    valid_events = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                          "commit_events": [{"request_id": "rA", "time_ms": 0},
                                            {"request_id": "rA", "time_ms": 10}]})
    assert valid_events.status == "available"
    assert valid_events.predicted_ms == pytest.approx(10.0 / 2.8)

    valid_times = _tpot({"arrival_ms": 0, "first_commit_ms": 0,
                         "commit_times": [0, 10]})
    assert valid_times.status == "available"
    assert valid_times.predicted_ms == pytest.approx(10.0 / 2.8)


# ---------------------------------------------------------------- P2-18
def test_p2_18_runtime_mask_intersects_mounted_container_devices():
    two_mounts = ("docker run --device /dev/davinci0 --device /dev/davinci1 "
                  "-e ASCEND_RT_VISIBLE_DEVICES=0 image vllm serve /m "
                  "--tensor-parallel-size 2")
    report = _parse(two_mounts)
    assert report.status == "partial"
    assert any(item.name == "visible_device_count"
               for item in report.unmodeled_or_unsupported)

    for mask in ("0,00", "00"):
        masked = _parse("docker run --device /dev/davinci0 "
                        f"-e ASCEND_RT_VISIBLE_DEVICES={mask} image vllm serve /m "
                        "--tensor-parallel-size 2")
        assert masked.status == "partial", mask
        assert any(item.name == "visible_device_count"
                   for item in masked.unmodeled_or_unsupported), mask

    complete = _parse("docker run --device /dev/davinci0 --device /dev/davinci1 "
                      "-e ASCEND_RT_VISIBLE_DEVICES=0,1 image vllm serve /m "
                      "--tensor-parallel-size 2")
    assert complete.status == "completed"

    unmounted = _parse("docker run -e ASCEND_RT_VISIBLE_DEVICES=0,00 "
                       "image vllm serve /m")
    assert unmounted.status == "completed"

    unidentified = _parse("docker run --device /dev/davinci0 "
                          "--device /dev/davinci1 "
                          "-e ASCEND_RT_VISIBLE_DEVICES=-1 image vllm serve /m")
    assert unidentified.status == "partial"
    assert any(item.name == "visible_device_unidentified"
               for item in unidentified.unmodeled_or_unsupported)


# ---------------------------------------------------------------- P2-20
def test_p2_20_requirements_accumulate_before_scalar_failure():
    report = _parse("docker run -e MISSING image vllm serve /m "
                    "--enable-expert-parallel=invalid")
    assert report.status == "failed"
    assert "env:MISSING" in {item.name for item in report.required_inputs}
    assert any("invalid_boolean_value" in item.message
               for item in report.diagnostics)

    scalar = _parse("docker run -e MISSING image vllm serve /m "
                    "--tensor-parallel-size abc")
    assert scalar.status == "needs_inputs"
    assert "env:MISSING" in {item.name for item in scalar.required_inputs}

    env_file = _parse("docker run --env-file secrets.env image vllm serve /m "
                      "--enable-expert-parallel=invalid")
    assert env_file.status == "failed"
    assert "env:secrets.env" in {item.name for item in env_file.required_inputs}

    without_geometry = _parse(
        "docker run -e MISSING image vllm serve /m "
        "--enable-expert-parallel=invalid", machine=None, model=None, workload=None)
    names = {item.name for item in without_geometry.required_inputs}
    assert {"env:MISSING", "machine", "model", "workload"} <= names


# ---------------------------------------------------------------- P2-23
def test_p2_23_structural_key_and_container_occurrence_spans():
    config = '{"splitting_ops":["cudagraph_mode"],"cudagraph_mode":"FULL"}'
    report = _parse(f"vllm serve /m --compilation-config '{config}'")
    mode = _setting(report, "cudagraph_mode")
    assert mode.span.column_start == 73
    assert mode.effective and mode.value == "FULL"

    # Variant: the object key is first, the array element afterwards; the
    # structural scan must still choose the object key, never the array value.
    reversed_config = '{"cudagraph_mode":"FULL","splitting_ops":["cudagraph_mode"]}'
    reversed_script = (f"vllm serve /m --compilation-config "
                       f"'{reversed_config}'")
    reversed_report = _parse(reversed_script)
    key = _setting(reversed_report, "cudagraph_mode")
    assert key.span.column_start == reversed_script.index('"cudagraph_mode"') + 1

    command = ["vllm", "serve", "/m",
               "--compilation-config", '{"cudagraph_mode":"FULL"}']
    document = json.dumps({"containers": [{"command": command},
                                          {"command": command}]}, indent=2)
    second = _parse(document, format="json", container=1)
    assert _setting(second, "cudagraph_mode").span.line_start == 18
    first = _parse(document, format="json", container=0)
    assert _setting(first, "cudagraph_mode").span.line_start == 9

    # Variant: named containers with distinct commands keep their own lines.
    named = json.dumps({"containers": [
        {"name": "first", "command": command},
        {"name": "second", "command": command}]}, indent=2)
    by_name = _parse(named, format="json", container="second")
    assert _setting(by_name, "cudagraph_mode").span.line_start == 20
