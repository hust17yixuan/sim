"""Adversarial regressions for the NR-S1 launch/SOL review findings.

[Role] Convert the P1-01..P2-23 reproductions into behavioral checks of the
  public parser, bindings, SOL, analysis and CLI boundaries.
[Invariants] Isolated HOME/MINDIE state, no host env defaults, no network, no
  string-grep filler; every expected value comes from a declared contract or
  an independently computed engine/work quantity.
[Timeouts & Locks] Entire module stays under five seconds; no locks.
[Side-Effect Policy] Writes only under pytest's tmp_path.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path

import pytest

from mindie_sim import compose_case
from mindie_sim.feature_analysis import compare_launch
from mindie_sim.feature_cli import main as cli_main
from mindie_sim.feature_reports import (
    Context, Counterfactual, ReportValidationError, SourceIdentity, canonical_json,
    launch_markdown, validate_sol_report,
)
from mindie_sim.launch_bindings import adapter_for, parse_launch
from mindie_sim.launch_documents import (
    DocumentInputError, finite_json_loads, parse_document,
)
from mindie_sim.launch_parser import LaunchSyntaxError, parse_argv, parse_shell, selected_command
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.sol import bound, compare_sol, critical_path
from mindie_sim.prediction.sol_work import (
    Peak, PlanWork, WorkTerm, project_ops, project_program,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8, "prefix_tokens": 0}
SPEC = {"draft_len": 3, "acceptance": [0.8, 0.6, 0.4]}
CANARY = "REVIEW_CANARY_9f3c"
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    for name in ("HOME", "MINDIE_CACHE_HOME", "MINDIE_DATA_HOME", "MINDIE_SIM_HOME",
                 "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    for name in ("VLLM_ASCEND_ENABLE_MLAPO", "HCCL_BUFFSIZE", "VLLM_ALL2ALL_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _op(name, deps=(), flops=0.0, hbm=0.0, vec=0.0, link=0.0, stage=0,
        stream="", metadata=(), family="matmul", dtype="bf16"):
    return OpSpec(op_id=name, family=family, phase="decode", stage=stage, layer=0,
                  requests=("r0",), semantic_shape=(1, 1), local_shape=(1, 1),
                  padded_shape=(1, 1), launch_shape=(1, 1), flops=flops,
                  hbm_bytes=hbm, vector_ops=vec, link_bytes=link, deps=tuple(deps),
                  dtype=dtype, stream=stream, metadata=tuple(metadata))


def _case(model="qwen3-30b-a3b", **kwargs):
    kwargs.setdefault("layers", 1)
    kwargs.setdefault("graph_capture_mode", "none")
    return compose_case(model, hardware="a3", **kwargs)


def _parse(text, **kwargs):
    options = {**BASE, **kwargs}
    return parse_launch(text, **options)


def _deepseek_base():
    return _case("deepseek-v3.2", layers=4, tp=8, dp=2, ep=16)


# ---------------------------------------------------------------- P1-01
@pytest.mark.parametrize("script", [
    f"vllm serve /m --new-config '{{\"api_key\":\"{CANARY}\"}}'",
    f"export API_KEY={CANARY}\nvllm serve /m --served-model-name \"$API_KEY\"",
    f"export MODE={CANARY}\nvllm serve /m --compilation-config \"{{\\\"cudagraph_mode\\\":\\\"$MODE\\\"}}\"",
])
def test_p1_01_redaction_covers_unknown_json_variables_and_diagnostics(script):
    report = _parse(script, framework_version="vllm-ascend@0.30.0")
    assert CANARY not in canonical_json(report)
    assert CANARY not in launch_markdown(report)
    assert all(CANARY not in item.message for item in report.diagnostics)
    assert report.status in {"partial", "failed", "completed"}
    if report.status == "failed":
        assert report.case is None


# ---------------------------------------------------------------- P1-02
def test_p1_02_unsupported_does_not_suppress_supported_patches_or_overrides():
    script = (
        "export HCCL_BUFFSIZE=200\n"
        "vllm serve /m --tensor-parallel-size 8 --data-parallel-size 2 "
        "--enable-expert-parallel --max-num-seqs 7 --async-scheduling "
        "--compilation-config '{\"cudagraph_mode\":\"NONE\"}' "
        "--additional-config '{\"enable_mlapo\":true}' "
        "--speculative-config '{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":3}'\n"
    )
    report = parse_launch(script, framework_version="vllm-ascend@0.30.0",
                          machine="a3", model="deepseek-v3.2", workload=WORKLOAD,
                          speculative=SPEC, base_case=_deepseek_base(),
                          overrides={"/runtime/kv_block_size": 64})
    assert report.status == "partial"
    runtime = report.case["runtime"]
    assert runtime["max_num_seqs"] == 7 and runtime["async_scheduling"] is True
    assert runtime["graph_mode"] == "none"
    assert runtime["kernels"]["lowerings"]["mla_prolog"]
    assert dict(runtime["metadata"])["draft_mode"] == "mtp"
    assert runtime["kv_block_size"] == 64
    assert report.effective_overrides[0].after == 64
    assert any(item.name == "HCCL_BUFFSIZE"
               for item in report.unmodeled_or_unsupported)


def test_p1_02_cli_keeps_exit_three_for_partial(tmp_path):
    script = tmp_path / "partial.sh"
    script.write_text("export HCCL_BUFFSIZE=200\nvllm serve /m --max-num-seqs 7\n",
                      encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD), encoding="utf-8")
    assert cli_main(["inspect-launch", str(script), "--machine", "a3",
                     "--model", "qwen3-30b-a3b", "--workload", str(workload)]) == 3


# ---------------------------------------------------------------- P1-03
def test_p1_03_environment_occurrences_are_command_scoped():
    late = _parse("vllm serve /m\nexport VLLM_ASCEND_ENABLE_MLAPO=1\n",
                  framework_version="vllm-ascend@0.18.0")
    assert "mla_prolog" not in late.case["runtime"]["kernels"].get("lowerings", {})
    outer = _parse("export VLLM_ASCEND_ENABLE_MLAPO=1\ndocker run image vllm serve /m\n",
                   framework_version="vllm-ascend@0.18.0")
    assert "mla_prolog" not in outer.case["runtime"]["kernels"].get("lowerings", {})


def test_p1_03_docker_pass_through_uses_outer_snapshot_not_inheritance():
    passed = _parse("export HCCL_BUFFSIZE=200\n"
                    "docker run -e HCCL_BUFFSIZE image vllm serve /m\n")
    assert passed.status == "partial"
    assert any(item.name == "HCCL_BUFFSIZE" and item.value == "200"
               for item in passed.unmodeled_or_unsupported)
    substituted = _parse("export BACKEND=alltoall\n"
                         "export VLLM_ALL2ALL_BACKEND=$BACKEND\nvllm serve /m\n")
    assert substituted.case["runtime"]["moe_recipe"] == "alltoall"
    script = ("export VLLM_ALL2ALL_BACKEND=mc2\nvllm serve /a --tensor-parallel-size 2\n"
              "export VLLM_ALL2ALL_BACKEND=alltoall\nvllm serve /b --tensor-parallel-size 4\n")
    assert _parse(script, command_index=0).case["runtime"]["moe_recipe"] == "mc2"
    assert _parse(script, command_index=1).case["runtime"]["moe_recipe"] == "alltoall"


# ---------------------------------------------------------------- P1-04
def test_p1_04_boolean_actions_and_explicit_disabling():
    report = _parse("vllm serve /m --tensor-parallel-size 2 "
                    "--async-scheduling=false --enable-chunked-prefill=false "
                    "--enable-expert-parallel=false\n")
    runtime = report.case["runtime"]
    assert runtime["async_scheduling"] is False
    assert runtime["enable_chunked_prefill"] is False
    assert runtime["parallel"]["ep"] == 1
    assert _parse("vllm serve /m --async-scheduling=maybe\n").status == "failed"
    wrong_json = _parse("vllm serve /m --additional-config '{\"enable_mlapo\":\"false\"}'\n",
                        framework_version="vllm-ascend@0.30.0")
    assert wrong_json.status == "failed"


def test_p1_04_enabled_base_can_be_disabled():
    base = _deepseek_base()
    enabled = parse_launch(
        "vllm serve /m --additional-config '{\"enable_mlapo\":true}'\n",
        framework_version="vllm-ascend@0.30.0", base_case=base, workload=WORKLOAD,
        speculative=SPEC)
    disabled = parse_launch(
        "vllm serve /m --additional-config '{\"enable_mlapo\":false}'\n",
        framework_version="vllm-ascend@0.30.0", base_case=enabled.case,
        workload=WORKLOAD, speculative=SPEC)
    assert "mla_prolog" in enabled.case["runtime"]["kernels"]["lowerings"]
    assert "mla_prolog" not in disabled.case["runtime"]["kernels"].get("lowerings", {})
    no_capture = parse_launch(
        "vllm serve /m --compilation-config '{\"cudagraph_mode\":\"NONE\"}'\n",
        framework_version="vllm-ascend@0.30.0", base_case=_case(graph_capture_mode="full"),
        workload=WORKLOAD)
    assert no_capture.case["runtime"]["graph_capture_mode"] == "none"
    assert tuple(no_capture.case["runtime"]["graph_capture_keys"]) == ()


def test_p1_04_prefix_caching_disable_clears_declared_hits():
    base = _case(prefix_tokens=64)
    report = parse_launch("vllm serve /m --no-enable-prefix-caching\n",
                          base_case=base, workload={**WORKLOAD, "prefix_tokens": 64})
    assert [request["prefix_tokens"] for request in report.case["workload"]["requests"]] == [0, 0]
    assert dict(report.case["runtime"]["metadata"])["enable_prefix_caching"] is False


# ---------------------------------------------------------------- P1-05
def test_p1_05_shadowed_json_parent_invalidates_children():
    base = _deepseek_base()
    report = parse_launch(
        "vllm serve /m --additional-config '{\"enable_mlapo\":true}' "
        "--additional-config '{\"multistream_overlap_shared_expert\":false}'\n",
        framework_version="vllm-ascend@0.30.0", base_case=base, workload=WORKLOAD,
        speculative=SPEC)
    parents = [item for item in report.applied if item.name == "--additional-config"]
    assert len(parents) == 2
    first, second = parents
    assert (first.effective, second.effective) == (False, True)
    assert first.shadowed_by == second.id
    child = next(item for item in report.applied if item.name == "enable_mlapo")
    assert child.parent_id == first.id and child.shadowed_by == second.id
    assert child.effective is False
    assert "mla_prolog" not in report.case["runtime"]["kernels"].get("lowerings", {})
    assert report.case["runtime"]["multistream_overlap_shared_expert"] is False


# ---------------------------------------------------------------- P1-06
def test_p1_06_ep_uses_inherited_and_overridden_effective_axes():
    base = _case(tp=8, dp=2, ep=1)
    inherited = parse_launch("vllm serve /m --enable-expert-parallel\n",
                             base_case=base, workload=WORKLOAD)
    assert inherited.case["runtime"]["parallel"]["ep"] == 16
    overridden = parse_launch(
        "vllm serve /m --enable-expert-parallel --tensor-parallel-size 4\n",
        base_case=base, workload=WORKLOAD)
    assert overridden.case["runtime"]["parallel"]["ep"] == 8
    pcp_base = _case(tp=4, dp=2, pcp=2, ep=1)
    pcp = parse_launch("vllm serve /m --enable-expert-parallel\n",
                       base_case=pcp_base, workload=WORKLOAD)
    assert pcp.case["runtime"]["parallel"]["ep"] == 16


# ---------------------------------------------------------------- P1-07
def test_p1_07_replica_and_stage_owners_are_distinct():
    case = _case(layers=2, pp=2, tp=1)
    ops = {0: (_op("stage0", flops=1e9, stage=0), _op("stage1", flops=1e9, stage=1))}
    predicted = graph_costs(case, ops, compact=True, trace_limit=0)["duration_ms"]
    plan = project_ops(case, ops, plan_id="pp")
    assert plan.coverage == "complete"
    resources = {row.physical_resource_id for row in bound(plan).resources}
    assert "matrix@r0:s0" in resources and "matrix@r0:s1" in resources
    assert bound(plan).sol_ms <= predicted + 1e-12
    assert bound(plan).sol_ms < 2.0 * (1e9 / 3.5e11)


# ---------------------------------------------------------------- P1-08
def test_p1_08_nested_program_work_is_conserved_not_zero():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "fixture"),)
    nested = KernelProgram((Stage("outer", program=KernelProgram(
        (Stage("inner", {"hbm": 1e9}),))),))
    plan = project_program(nested, peaks)
    assert plan.coverage == "complete" and not plan.holes
    assert bound(plan).sol_ms == pytest.approx(1.0)
    class Opaque:
        pass
    opaque = project_program(Opaque(), peaks)
    assert opaque.coverage != "complete"
    assert any("opaque_program_unconserved" in hole for hole in opaque.holes)


def test_p1_08_unknown_precision_is_a_hole_not_bf16_fallback():
    case = _case()
    plan = project_ops(case, (_op("fp64", flops=1e9, dtype="fp64"),), plan_id="fp64")
    assert plan.coverage != "complete"
    assert any("lowering_unavailable" in hole or "compute_precision" in hole
               for hole in plan.holes)


# ---------------------------------------------------------------- P1-09
def test_p1_09_program_dag_respects_after_not_tuple_order():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "fixture"),
             Peak("vector", "ops", 1e9, "declared", "fixture"))
    independent = project_program(KernelProgram(
        (Stage("a", {"hbm": 1e9}), Stage("b", {"vector": 1e9}))), peaks)
    assert bound(independent).sol_ms == pytest.approx(1.0)
    serial = project_program(KernelProgram(
        (Stage("a", {"hbm": 1e9}),
         Stage("b", {"vector": 1e9}, after=("a",)))), peaks)
    assert bound(serial).sol_ms == pytest.approx(2.0)


# ---------------------------------------------------------------- P1-10
def test_p1_10_peak_ms_units_and_exact_service_modes():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "fixture"),
             Peak("cube", "peak_ms", 2.0, "", "fixture"))
    stage = KernelProgram((Stage("a", {"matrix": 1.0},
                                 work_units={"matrix": "peak_ms"}),))
    plan = project_program(stage, peaks)
    assert bound(plan).sol_ms == pytest.approx(0.5)
    unknown = project_program(KernelProgram((
        Stage("ghost", {"unknown": 1.0}, work_units={"unknown": "peak_ms"}),)), ())
    assert bound(unknown).status == "unavailable"
    missing = PlanWork("missing", (WorkTerm("a", ("a",),
                                            (("cube", 1e9, "flops", "fp8"),)),), peaks)
    assert bound(missing).status == "unavailable"
    mismatch = WorkTerm("m", ("m",), (("hbm", 1.0, "ops", ""),))
    assert bound(PlanWork("m", (mismatch,), peaks)).status == "unavailable"


# ---------------------------------------------------------------- P1-11
def test_p1_11_inclusive_floors_do_not_add_to_service():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "fixture"),)
    inclusive = project_program(KernelProgram(
        (Stage("a", {"hbm": 1e9}, latency=1.0),)), peaks)
    assert bound(inclusive, "declared_physical").sol_ms == pytest.approx(1.0)
    grouped = project_program(KernelProgram(
        (Stage("a", {"hbm": 0.25e9}), Stage("b", {"hbm": 0.25e9})),
        minimum_ms=0.5), peaks)
    assert bound(grouped, "declared_physical").sol_ms == pytest.approx(0.5)
    startup = WorkTerm("a", ("a",), (("hbm", 1e9, "bytes", ""),),
                       dispatch_min_ms=0.1)
    assert bound(PlanWork("s", (startup,), peaks), "declared_physical").sol_ms == pytest.approx(1.1)


# ---------------------------------------------------------------- P1-12
def test_p1_12_collective_uses_the_traversed_slow_domain():
    case = Case.from_mapping(_case(tp=2, num_devices=4))
    machine = replace(case.machine, links=(
        Link("used_slow", 1e6, (0, 1)), Link("unrelated_fast", 1e9, (2, 3))))
    case = replace(case, machine=machine)
    op = _op("collective", link=1e9, family="collective",
             metadata=(("group_axis", "tp"),))
    plan = project_ops(case, {0: (op,)}, plan_id="link")
    rows = bound(plan).resources
    assert plan.coverage == "complete"
    assert {row.physical_resource_id for row in rows} == {"link:used_slow"}
    assert rows[0].load_bound_ms == pytest.approx(2000.0)
    assert "unrelated_fast" not in canonical_json([row.physical_resource_id for row in rows])


# ---------------------------------------------------------------- P1-13
def test_p1_13_counterfactual_is_relowered_and_records_coupled_patches():
    case = _case(tp=2, graph_capture_mode="full")
    report = parse_launch(
        "vllm serve /m --tensor-parallel-size 2 "
        "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'\n",
        framework_version="vllm-ascend@0.30.0", base_case=case, workload=WORKLOAD)
    compared = compare_launch(report, counterfactuals=("ACLGraph",),
                              tokens=1, context=128, sequences=2)
    candidate = compared.counterfactuals[0]
    assert candidate.status == "evaluated"
    assert candidate.candidate_plan_id != compared.context.plan_id
    assert any(patch.path == "/runtime/graph_mode" and patch.after == "none"
               for patch in candidate.patches)
    assert not any("_saved_bytes" in patch.path for patch in candidate.patches)
    assert candidate.sol_ms <= candidate.predicted_ms + 1e-12


def test_p1_13_missing_repartition_is_inapplicable_with_null_deltas():
    case = _case(tp=2, pcp=2)
    report = parse_launch("vllm serve /m --tensor-parallel-size 2 "
                          "--prefill-context-parallel-size 2\n",
                          base_case=case, workload=WORKLOAD)
    compared = compare_launch(report, plan_ops=(_op("a", flops=1e9),),
                              counterfactuals=("PCP/DCP",))
    candidate = compared.counterfactuals[0]
    assert candidate.status == "inapplicable"
    assert candidate.predicted_ms is None and candidate.saved_ms is None
    assert candidate.reason


# ---------------------------------------------------------------- P1-14
def test_p1_14_candidate_sol_violations_are_rejected():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "fixture"),)
    plan = PlanWork("p", (WorkTerm("a", ("a",), (("hbm", 1e9, "bytes", ""),)),), peaks)
    ctx = Context("c", "p", "model_step", "decode", 1, 2, 128, "b", "complete",
                  (SourceIdentity("fixture"),))
    baseline = compare_sol(plan, 2.0, context=ctx)
    bad = Counterfactual("cf", "Feature", "evaluated", "p", "p:cf", (), (),
                         0.1, 100.0, 1.9, -98.0, None)
    with pytest.raises(ReportValidationError) as excinfo:
        validate_sol_report(replace(baseline, counterfactuals=(bad,)))
    assert "candidate_bound_exceeds_prediction" in str(excinfo.value)


# ---------------------------------------------------------------- P1-15
def test_p1_15_request_metrics_require_trace_and_fractional_commits():
    report = _parse("vllm serve /m --tensor-parallel-size 2 "
                    "--speculative-config '{\"method\":\"deepseek_mtp\","
                    "\"num_speculative_tokens\":3}'\n", speculative=SPEC)
    ops = (_op("a", flops=1e9, hbm=1024),)
    bare = compare_launch(report, plan_ops=ops, predicted_ms=2.0,
                          request_metrics=True)
    assert [metric.status for metric in bare.request_metrics] == ["unavailable", "unavailable"]
    assert bare.request_metrics[0].predicted_ms is None
    traced = compare_launch(
        report, plan_ops=ops, prefill_ops=ops, predicted_ms=2.0,
        request_metrics=True, commits=2.8,
        request_trace={"arrival_ms": 0, "first_commit_ms": 50,
                       "decode_steps": 1, "commit_times": [50, 52]})
    ttft, tpot = traced.request_metrics
    assert (ttft.status, tpot.status) == ("available", "available")
    assert ttft.predicted_ms == pytest.approx(50.0)
    assert tpot.predicted_ms == pytest.approx(2.0 / 2.8)
    missing_prefill = compare_launch(
        report, plan_ops=ops, predicted_ms=2.0, request_metrics=True,
        request_trace={"arrival_ms": 0, "first_commit_ms": 50, "decode_steps": 1})
    assert missing_prefill.request_metrics[0].status == "unavailable"


# ---------------------------------------------------------------- P2-16
def test_p2_16_versions_are_bounded_and_capture_defaults_are_explicit():
    assert adapter_for("vllm-ascend@9.99.0") is None
    assert adapter_for("vllm-ascend@0.30.7").id == "vllm-ascend/0.30-static"
    absent = _parse("vllm serve /m --compilation-config '{\"cudagraph_mode\":\"FULL\"}'\n")
    assert absent.status == "needs_inputs"
    assert [item.name for item in absent.required_inputs] == ["framework_version"]
    future = _parse("vllm serve /m --compilation-config '{\"cudagraph_mode\":\"FULL\"}'\n",
                    framework_version="vllm-ascend@9.99.0")
    assert future.status == "needs_inputs"
    known = _parse("vllm serve /m --compilation-config '{\"cudagraph_mode\":\"FULL\"}'\n",
                   framework_version="vllm-ascend@0.30.7")
    assert known.status == "completed" and known.case["runtime"]["graph_capture_keys"]


# ---------------------------------------------------------------- P2-17
def test_p2_17_shell_literals_and_assignment_positions():
    command = selected_command(parse_shell('vllm serve /m --x "a\\nb" --y ""'))
    assert command.argv[3:5] == ("--x", "a\\nb")
    assert command.argv[-1] == ""
    command = selected_command(parse_shell("A=2 B=$A vllm serve /m"))
    assert dict(command.env)["A"] == "2" and dict(command.env)["B"] == "2"
    command = selected_command(parse_shell("vllm serve /m --x 'a'\"b\""))
    assert command.argv[-1] == "ab"
    with pytest.raises(LaunchSyntaxError):
        parse_shell('"VLLM=1" vllm serve /m')
    with pytest.raises(LaunchSyntaxError):
        parse_shell("MODEL=~/weights vllm serve $MODEL")


# ---------------------------------------------------------------- P2-18
def test_p2_18_structured_command_args_and_container_selection():
    document = parse_document(
        json.dumps({"command": ["vllm", "serve", "/m"],
                    "args": ["--tensor-parallel-size", "2"]}),
        format="json")
    assert document.argv == ("vllm", "serve", "/m", "--tensor-parallel-size", "2")
    flat = parse_document(json.dumps(["vllm", "serve", "/m"]), format="json")
    assert flat.argv == ("vllm", "serve", "/m")
    containers = json.dumps({"containers": [
        {"name": "a", "command": ["vllm", "serve", "/a"]},
        {"name": "b", "command": ["vllm", "serve", "/b"]}]})
    assert parse_document(containers, format="json", container="b").argv == \
        ("vllm", "serve", "/b")
    with pytest.raises(LaunchSyntaxError):
        parse_document("docker run --entrypoint echo image vllm serve /m", format="shell")
    assert parse_document("docker run -- image vllm serve /m", format="shell").image == "image"


def _yaml_available():
    """Availability is fixed by the process's initial import environment."""
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def test_p2_18_yaml_installed_and_absent_paths():
    if not _yaml_available():
        report = _parse("a: 1", format="yaml")
        assert report.status == "failed"
        assert report.diagnostics[-1].code == "yaml_parser_unavailable"
        return
    document = parse_document("command: [vllm, serve, /m]\n"
                              "args: [--tensor-parallel-size, '2']\n", format="yaml")
    assert document.argv == ("vllm", "serve", "/m", "--tensor-parallel-size", "2")
    for text in ("a: &x 1\nb: *x\n", "a:\n  <<: {b: 1}\n", "a: 1\na: 2\n"):
        with pytest.raises(DocumentInputError):
            parse_document(text, format="yaml")


def test_p2_18_unverified_device_visibility_is_not_completed():
    report = _parse("docker run --device /dev/davinci999 image vllm serve /m\n")
    assert report.status != "completed"
    assert any(item.name == "visible_device"
               for item in report.unmodeled_or_unsupported)


# ---------------------------------------------------------------- P2-19
def test_p2_19_limits_finiteness_and_single_argv_validation(monkeypatch):
    with pytest.raises(DocumentInputError):
        finite_json_loads('{"value":1e999}')
    with pytest.raises(DocumentInputError):
        finite_json_loads("[" * 1100 + "0" + "]" * 1100)
    with pytest.raises(DocumentInputError):
        finite_json_loads(str(2 ** 5000))
    with pytest.raises(DocumentInputError):
        finite_json_loads(" " * (8 * 1024 * 1024 + 1))
    import mindie_sim.launch_parser as parser
    calls = {"n": 0}
    original = parser._validate_command
    def counted(argv):
        calls["n"] += 1
        return original(argv)
    monkeypatch.setattr(parser, "_validate_command", counted)
    argv = ["vllm", "serve", "/m"] + [value for i in range(50)
                                      for value in (f"--x{i}", "1")]
    parse_argv(argv)
    assert calls["n"] == 1


# ---------------------------------------------------------------- P2-20
def test_p2_20_independent_diagnostics_and_required_env_inputs():
    combined = parse_launch("vllm serve /m --max-num-seqs nope\n")
    assert combined.status == "failed"
    assert any(item.code == "invalid_declaration" for item in combined.diagnostics)
    assert {item.name for item in combined.required_inputs} >= {"machine", "model"}
    both = _parse("export HCCL_BUFFSIZE=200\nvllm serve /m --max-num-seqs nope\n")
    assert both.status == "failed"
    assert any(item.code == "invalid_declaration" for item in both.diagnostics)
    assert any(item.name == "HCCL_BUFFSIZE"
               for item in both.unmodeled_or_unsupported)
    needed = _parse("docker run -e UNDECLARED image vllm serve /m\n")
    assert needed.status == "needs_inputs"
    assert [item.name for item in needed.required_inputs] == ["env:UNDECLARED"]
    env_file = _parse("docker run --env-file values.env image vllm serve /m\n")
    assert env_file.status == "needs_inputs"
    assert [item.name for item in env_file.required_inputs] == ["env:values.env"]


# ---------------------------------------------------------------- P2-21
def test_p2_21_same_pipe_modes_sum_before_resource_maxima():
    peaks = (Peak("cube", "peak_ms", 1.0, "bf16", "fixture"),
             Peak("hbm", "bytes", 1e9, "declared", "fixture"))
    term = WorkTerm("mixed", ("mixed",),
                    (("cube", 1.0, "peak_ms", ""), ("cube", 1.0, "peak_ms", ""),
                     ("hbm", 1e9, "bytes", "")))
    plan = PlanWork("mixed", (term,), peaks)
    result = bound(plan)
    assert result.critical_path_ms == pytest.approx(2.0)
    assert result.resource_load_ms == pytest.approx(2.0)
    assert result.sol_ms == pytest.approx(2.0)


def test_p2_21_stream_order_and_fused_alias_conservation():
    case = _case()
    ops = (_op("a", flops=1e9, hbm=1024),
           _op("b", hbm=2048))
    plan = project_ops(case, ops, plan_id="stream")
    assert plan.terms[1].predecessors == (plan.terms[0].id,)
    parent = _op("parent", flops=1e9, hbm=1024)
    child = _op("child", hbm=1024, metadata=(("fused_into", "parent"),))
    consumer = _op("consumer", deps=("child",), hbm=1024)
    unproven = project_ops(case, (parent, child, consumer), plan_id="unproven")
    assert unproven.coverage != "complete"
    proven_child = _op("child", hbm=1024,
                       metadata=(("fused_into", "parent"),
                                 ("fused_work_conserved", True)))
    proven = project_ops(case, (parent, proven_child, consumer), plan_id="proven")
    assert proven.coverage != "complete"
    assert [term.member_ops for term in proven.terms] == [
        ("parent",), ("child",), ("consumer",)]
    assert proven.terms[2].predecessors == (proven.terms[1].id,)


# ---------------------------------------------------------------- P2-22
def test_p2_22_generated_body_bound_binds_external_input():
    case = _case(tp=2, graph_capture_mode="full")
    report = parse_launch("vllm serve /m --tensor-parallel-size 2\n",
                          base_case=case, workload=WORKLOAD)
    compared = compare_launch(report, scope="body", tokens=1, context=128,
                              sequences=2)
    assert compared.status == "completed"
    assert compared.bound.status == "available"
    assert compared.context.scope == "body_step"
    assert compared.bound.sol_ms <= compared.prediction.predicted_ms + 1e-12


def test_p2_22_cli_generated_body(tmp_path):
    case = _case(tp=2)
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps(Case.from_mapping(case).to_dict()), encoding="utf-8")
    script = tmp_path / "deploy.sh"
    script.write_text("vllm serve /m --tensor-parallel-size 2\n", encoding="utf-8")
    assert cli_main(["sol", str(script), "--case", str(case_file),
                     "--scope", "body", "--tokens", "1", "--context", "128",
                     "--sequences", "2", "--format", "json"]) == 0


# ---------------------------------------------------------------- P2-23
def test_p2_23_occurrence_ids_and_spans_identify_the_source():
    script = ("vllm serve /m \\\n"
              "  --max-num-seqs 7 \\\n"
              "  --tensor-parallel-size 2\n")
    report = _parse(script)
    settings = {item.name: item for item in report.applied}
    assert settings["--max-num-seqs"].span.line_start == 2
    assert settings["--max-num-seqs"].span.column_start == 3
    assert settings["--tensor-parallel-size"].span.line_start == 3
    assert len(report.setting_ids) == len(set(report.setting_ids))
    source = _parse(script, source=str(Path("/tmp/deploy.sh")))
    assert source.source.name == "/tmp/deploy.sh"
    assert source.source.source_kind == "file"
    assert len(source.source.sha256) == 64
    document = parse_document("docker run image sh -c 'vllm serve /m \\\n"
                              "  --tensor-parallel-size 2'\n", format="shell")
    flag = next(word for word in document.words
                if word.text == "--tensor-parallel-size")
    assert (flag.span.line_start, flag.span.column_start) == (2, 3)
    json_report = _parse(
        "vllm serve /m --compilation-config "
        "'{\"cudagraph_mode\":\"FULL\",\"capture_sizes\":[1,2]}'\n",
        framework_version="vllm-ascend@0.30.0")
    leaf = next(item for item in json_report.applied
                if item.name == "cudagraph_mode")
    assert leaf.span.line_start == 1 and leaf.span.column_start > 30
