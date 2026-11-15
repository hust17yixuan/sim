"""Round-2 adversarial regressions for every open NR-S1 review item.

[Role] Turn each exact counterexample in `astra_review_r2.md` into a behavioral
  check of the public parser, bindings, SOL, counterfactual and CLI boundaries.
[Invariants] Isolated HOME/MINDIE state, no host env defaults, no network and no
  string-grep filler: every expected quantity comes from a declared contract, a
  lowered program, or an independently captured Engine serviced ledger.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] Writes only under pytest's tmp_path.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math

import pytest

from mindie_sim import compose_case
from mindie_sim.feature_analysis import compare_launch
from mindie_sim.feature_cli import main as cli_main
from mindie_sim.feature_reports import (
    Context, ReportValidationError, RequestMetric, SourceIdentity, canonical_json,
    launch_markdown, validate_sol_report,
)
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_documents import DocumentInputError, finite_json_loads, parse_document
from mindie_sim.launch_parser import LaunchSyntaxError, parse_argv, parse_shell, selected_command
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.sol import bound, compare_sol, request_metrics
from mindie_sim.prediction.sol_work import (
    Peak, PlanWork, WorkTerm, engine_peaks, project_ops, project_program,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
SPEC = {"draft_len": 3, "acceptance": [0.8, 0.6, 0.4]}
CANARY = "R2_CANARY"
BASE = {"machine": "a3", "model": "deepseek-v3.2", "workload": WORKLOAD}


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
    return parse_launch(text, **{**BASE, **kwargs})


def _yaml_available():
    """Availability is fixed by the process's initial import environment."""
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------ _draft() TypeError
def test_additional_and_kv_transfer_configs_are_structured_api_and_cli(tmp_path, capsys):
    options = {"machine": "a3", "model": "deepseek-v3.2", "workload": WORKLOAD,
               "framework_version": "vllm-ascend@0.30.0"}
    unknown = parse_launch(
        "vllm serve /m --additional-config '{\"unknown\":1}'", **options)
    assert unknown.status == "partial" and unknown.case is not None
    assert {(item.name, item.effect)
            for item in unknown.unmodeled_or_unsupported} == {
        ("--additional-config", "unmodeled"), ("unknown", "unmodeled")}
    peer = parse_launch(
        "vllm serve /m --kv-transfer-config "
        "'{\"kv_connector\":\"x\",\"kv_role\":\"kv_both\"}'", **options)
    assert peer.status == "needs_inputs" and peer.case is None
    assert [item.name for item in peer.required_inputs] == ["kv_transfer_peer"]

    script = tmp_path / "deploy.sh"
    script.write_text("vllm serve /m --additional-config '{\"unknown\":1}'\n",
                      encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD), encoding="utf-8")
    common = ["--machine", "a3", "--model", "deepseek-v3.2",
              "--workload", str(workload), "--framework-version", "vllm-ascend@0.30.0"]
    assert cli_main(["inspect-launch", str(script), *common]) == 3
    assert "multiple values" not in capsys.readouterr().err
    script.write_text("vllm serve /m --kv-transfer-config "
                      "'{\"kv_connector\":\"x\",\"kv_role\":\"kv_both\"}'\n",
                      encoding="utf-8")
    assert cli_main(["inspect-launch", str(script), *common]) == 3
    assert "multiple values" not in capsys.readouterr().err
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op_id": "a", "flops": 1e9}]), encoding="utf-8")
    script.write_text("vllm serve /m --additional-config '{\"unknown\":1}'\n",
                      encoding="utf-8")
    assert cli_main(["sol", str(script), "--ops-json", str(ops), "--allow-partial",
                     *common]) == 3


# ------------------------------------------------------ P1-01
def test_p1_01_every_secret_taint_path_is_redacted():
    script = ('export API_KEY=' + CANARY + '\n'
              'docker run image vllm serve /m --served-model-name "$API_KEY"')
    forwarded = _parse(script, framework_version="vllm-ascend@0.30.0")
    assert forwarded.status == "completed"
    assert CANARY not in canonical_json(forwarded)
    assert CANARY not in launch_markdown(forwarded)
    model_name = next(item for item in forwarded.neutral_operational
                      if item.name == "--served-model-name")
    assert model_name.value == "[REDACTED]" and model_name.redacted

    failed = _parse("export API_KEY=" + CANARY + "\n$API_KEY")
    assert failed.status == "failed"
    assert CANARY not in canonical_json(failed)
    assert CANARY not in launch_markdown(failed)
    assert "[REDACTED]" in failed.diagnostics[0].message

    overridden = _parse("vllm serve /m",
                        overrides={"/runtime/metadata/api_key": CANARY})
    assert CANARY not in canonical_json(overridden)
    assert overridden.effective_overrides[0].after == "[REDACTED]"
    assert dict(overridden.case["runtime"]["metadata"])["api_key"] == "[REDACTED]"

    duplicate = _parse('{"' + CANARY + '":1,"' + CANARY + '":2}', format="json",
                       env={"API_KEY": CANARY},
                       framework_version="vllm-ascend@0.30.0")
    assert duplicate.status == "failed"
    assert duplicate.diagnostics[-1].code == "duplicate_json_key"
    assert CANARY not in canonical_json(duplicate)
    assert CANARY not in launch_markdown(duplicate)


# ------------------------------------------------------ P1-02
def test_p1_02_partial_comparison_and_cli_exit_are_three(tmp_path, capsys):
    base = _case(tp=2)
    report = parse_launch(
        "export HCCL_BUFFSIZE=200\nvllm serve /m --max-num-seqs 7",
        base_case=base, workload=WORKLOAD)
    assert report.status == "partial"
    compared = compare_launch(report, plan_ops=(_op("a", flops=1e9),),
                              allow_partial=True)
    assert compared.status == "partial"
    assert compared.context.coverage == "conditional_subset"
    unsupported = parse_launch("vllm serve /m --quantization fp8",
                               base_case=base, workload=WORKLOAD)
    assert unsupported.status == "partial"
    other = compare_launch(unsupported, plan_ops=(_op("a", flops=1e9),),
                           allow_partial=True)
    assert other.status == "partial"

    script = tmp_path / "partial.sh"
    script.write_text("export HCCL_BUFFSIZE=200\n"
                      "vllm serve /m --max-num-seqs 7\n", encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD), encoding="utf-8")
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op_id": "a", "flops": 1e9}]), encoding="utf-8")
    code = cli_main(["sol", str(script), "--machine", "a3",
                     "--model", "qwen3-30b-a3b", "--workload", str(workload),
                     "--ops-json", str(ops), "--allow-partial"])
    text = capsys.readouterr().out
    assert code == 3
    assert json.loads(text)["status"] == "partial"


# ------------------------------------------------------ P1-03
def test_p1_03_export_scope_inline_assignments_and_explicit_env():
    local = _parse("VLLM_ALL2ALL_BACKEND=mc2\nvllm serve /m")
    assert local.status == "completed"
    assert not any(item.name == "VLLM_ALL2ALL_BACKEND"
                   for item in (*local.applied, *local.neutral_operational,
                                *local.unmodeled_or_unsupported))
    assert local.case["runtime"]["moe_recipe"] != "mc2"

    inline = _parse("VLLM_ALL2ALL_BACKEND=mc2 vllm serve /m")
    assert inline.case["runtime"]["moe_recipe"] == "mc2"
    exported = _parse("export VLLM_ALL2ALL_BACKEND=mc2\nvllm serve /m")
    assert exported.case["runtime"]["moe_recipe"] == "mc2"

    not_inherited = _parse(
        "FOO=bar\ndocker run -e FOO image vllm serve /m")
    assert not_inherited.status == "needs_inputs"
    assert [item.name for item in not_inherited.required_inputs] == ["env:FOO"]
    for script in ("export FOO=bar\ndocker run -e FOO image vllm serve /m",
                   "FOO=bar docker run -e FOO image vllm serve /m"):
        passed = _parse(script)
        assert passed.status == "completed"
        assert any(item.name == "FOO" and item.value == "bar"
                   for item in passed.neutral_operational)

    explicit = parse_launch("vllm serve /m", machine="a3",
                            model="qwen3-30b-a3b", workload=WORKLOAD,
                            env={"VLLM_ALL2ALL_BACKEND": "mc2"})
    assert explicit.case["runtime"]["moe_recipe"] == "mc2"
    assert any(item.kind == "env" and item.name == "VLLM_ALL2ALL_BACKEND"
               for item in explicit.applied)


# ------------------------------------------------------ P1-04
def test_p1_04_030_legacy_disable_and_enforce_eager():
    base = _case("deepseek-v3.2", layers=4, tp=8, dp=2, ep=16)
    enabled = parse_launch(
        "vllm serve /m --additional-config '{\"enable_mlapo\":true}'",
        framework_version="vllm-ascend@0.30.0", base_case=base,
        workload=WORKLOAD, speculative=SPEC)
    assert "mla_prolog" in enabled.case["runtime"]["kernels"]["lowerings"]
    legacy = parse_launch(
        "VLLM_ASCEND_ENABLE_MLAPO=0 vllm serve /m",
        framework_version="vllm-ascend@0.30.0", base_case=enabled.case,
        workload=WORKLOAD, speculative=SPEC)
    assert legacy.status == "partial"
    rule = next(item for item in legacy.unmodeled_or_unsupported
                if item.name == "VLLM_ASCEND_ENABLE_MLAPO")
    assert rule.rule_id == "legacy_mlapo_env_not_interpreted"
    assert "mla_prolog" in legacy.case["runtime"]["kernels"]["lowerings"]
    old = parse_launch(
        "VLLM_ASCEND_ENABLE_MLAPO=0 vllm serve /m",
        framework_version="vllm-ascend@0.18.0", base_case=enabled.case,
        workload=WORKLOAD, speculative=SPEC)
    assert old.status == "completed"
    assert "mla_prolog" not in old.case["runtime"]["kernels"].get("lowerings", {})

    capture = _case(tp=2, graph_capture_mode="full")
    eager = parse_launch("vllm serve /m --enforce-eager",
                         base_case=capture, workload=WORKLOAD)
    runtime = eager.case["runtime"]
    assert eager.status == "completed"
    assert runtime["graph_mode"] == "none"
    assert runtime["graph_capture_mode"] == "none"
    assert tuple(runtime["graph_capture_keys"]) == ()
    assert tuple(runtime["piecewise_capture_keys"]) == ()
    assert runtime["mixed_mode"] == "none"
    assert dict(runtime["metadata"])["captured_execution_mode"] == "NONE"


# ------------------------------------------------------ P1-06
def test_p1_06_simulator_parallel_override_precedes_ep_derivation():
    base = _case(tp=4, dp=2, num_devices=16, graph_capture_mode="none")
    report = parse_launch(
        "vllm serve /m --enable-expert-parallel", base_case=base,
        workload=WORKLOAD, overrides={"/runtime/parallel/tp": 8})
    assert report.status == "completed"
    assert report.case["runtime"]["parallel"]["tp"] == 8
    assert report.case["runtime"]["parallel"]["dp"] == 2
    assert report.case["runtime"]["parallel"]["ep"] == 16


# ------------------------------------------------------ P1-08
def test_p1_08_lowered_program_work_is_conserved_against_engine(monkeypatch):
    import mindie_sim.prediction.graph_analysis as analysis
    from mindie_sim.prediction.operator_analysis import target_graphs
    seen = []
    original = analysis.Execution

    class Observed(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)

    case = _case(tp=2, graph_capture_mode="full")
    ops = target_graphs(case, phase="decode", scope="model", tokens=1,
                        context=128, sequences=2)
    with monkeypatch.context() as patch:
        patch.setattr(analysis, "Execution", Observed)
        graph_costs(case, ops, compact=True, trace_limit=0, isolated=False)
    serviced = seen[0].engine.serviced
    plan = project_ops(case, ops)
    assert plan.coverage == "complete" and not plan.holes
    projected = defaultdict(float)
    for term in plan.terms:
        for resource, amount, _unit, _mode in term.work:
            projected[resource] += amount
    assert projected["mte1@r0:s0"] == pytest.approx(76_677_120)
    assert projected["mte2_aic@r0:s0"] == pytest.approx(409_264_128, rel=1e-9)
    for resource, amount in serviced.items():
        if resource.startswith(("host@", "scheduler@", "serving@", "client@")):
            continue
        assert projected[resource] == pytest.approx(
            amount, rel=1e-9, abs=1e-6), resource
    peaks = engine_peaks(case)
    assert peaks and all(item.source.startswith("declared_uncalibrated")
                         for item in peaks)
    assert all(item.calibrated is False for item in peaks)


# ------------------------------------------------------ P1-09 / P1-11
def _engine_time(program):
    engine = Engine([Resource("hbm", 1), Resource("vector", 1)])
    ready = engine.submit("ready")
    end = program.start(engine, "program", ready)
    engine.advance(until=(end,))
    return end.time


def test_p1_09_nested_prefix_sequential_work_and_minimum_floor():
    peaks = (Peak("hbm", "bytes", 1.0, "", "fixture"),
             Peak("vector", "ops", 1.0, "", "fixture"))
    nested_after = KernelProgram((Stage("outer", program=KernelProgram((
        Stage("a", {"hbm": 1}), Stage("b", {"vector": 1}, after=("a",))))),))
    plan = project_program(nested_after, peaks)
    assert plan.coverage == "complete"
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(_engine_time(nested_after))

    own_then_nested = KernelProgram((Stage(
        "outer", {"hbm": 1},
        program=KernelProgram((Stage("b", {"vector": 1}),))),))
    plan = project_program(own_then_nested, peaks)
    assert bound(plan).sol_ms == pytest.approx(2.0)
    assert bound(plan).sol_ms == pytest.approx(_engine_time(own_then_nested))

    nested_minimum = KernelProgram((
        Stage("a", {"hbm": 2}),
        Stage("outer", after=("a",), program=KernelProgram(
            (Stage("empty"),), minimum_ms=1))))
    plan = project_program(nested_minimum, peaks)
    assert plan.coverage == "complete"
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(3.0)
    assert bound(plan, "declared_physical").sol_ms == pytest.approx(
        _engine_time(nested_minimum))
    assert "declared_program_minimum" in bound(plan, "declared_physical").floor_bases

    missing = KernelProgram((Stage("a", {"hbm": 1}, after=("absent",)),))
    broken = project_program(missing, peaks)
    assert broken.coverage != "complete"
    assert any("program_dependency_unresolved" in hole for hole in broken.holes)
    assert bound(broken).status == "unavailable"


# ------------------------------------------------------ P1-10
def test_p1_10_peak_ms_rate_is_checked_and_unknown_is_unavailable():
    declared = project_program(
        KernelProgram((Stage("a", {"matrix": 1.0},
                             work_units={"matrix": "peak_ms"}),)),
        (Peak("cube", "peak_ms", 2.0, "", "fixture"),))
    assert bound(declared).sol_ms == pytest.approx(0.5)
    ghost = project_program(
        KernelProgram((Stage("a", {"unknown": 1.0},
                             work_units={"unknown": "peak_ms"}),)), ())
    assert ghost.coverage != "complete"
    assert bound(ghost).status == "unavailable"
    mismatch = project_program(
        KernelProgram((Stage("a", {"hbm": 1.0},
                             work_units={"hbm": "ms"}),)),
        (Peak("hbm", "bytes", 1e9, "", "fixture"),))
    assert bound(mismatch).status == "unavailable"


# ------------------------------------------------------ P1-12
def test_p1_12_switch_plane_keeps_its_own_cut_capacity():
    case = Case.from_mapping(_case(tp=2, num_devices=4))
    machine = replace(case.machine, links=(
        Link("fabric", 1e9, (0, 1), switch_capacity_bytes_per_ms=1e6,
             switch_hops=1),))
    case = replace(case, machine=machine)
    op = _op("collective", link=1e9, family="collective",
             metadata=(("group_axis", "tp"),))
    plan = project_ops(case, {0: (op,)}, plan_id="switch")
    assert plan.coverage == "complete"
    peaks = {item.resource_id: item for item in plan.peaks}
    assert peaks["link:fabric"].peak_per_ms == pytest.approx(1e9)
    assert peaks["link:fabric:switch_plane"].peak_per_ms == pytest.approx(1e6)
    result = bound(plan)
    rows = {row.physical_resource_id: row for row in result.resources}
    switch = rows["link:fabric:switch_plane"]
    link = rows["link:fabric"]
    assert switch.load_bound_ms == pytest.approx(
        sum(term.work for term in switch.terms) / 1e6)
    assert link.load_bound_ms == pytest.approx(
        sum(term.work for term in link.terms) / 1e9)
    assert switch.load_bound_ms == pytest.approx(2000.0)
    assert link.load_bound_ms == pytest.approx(2.0)
    assert switch.load_bound_ms > link.load_bound_ms
    assert result.sol_ms == pytest.approx(switch.load_bound_ms)
    predicted = graph_costs(case, ops={0: (op,)}, compact=True,
                            trace_limit=0)["duration_ms"]
    assert predicted == pytest.approx(2000.0)
    assert result.sol_ms <= predicted + 1e-12


# ------------------------------------------------------ P1-13
def test_p1_13_prefix_and_mtp_cannot_reuse_stale_geometry():
    base = _case(tp=2, prefix_tokens=64)
    report = parse_launch("vllm serve /m --tensor-parallel-size 2",
                          base_case=base, workload={**WORKLOAD, "prefix_tokens": 64})
    prefix = compare_launch(report, counterfactuals=("Prefix Caching",),
                            phase="prefill", tokens=64, context=0, sequences=2)
    candidate = prefix.counterfactuals[0]
    assert candidate.status == "inapplicable"
    assert candidate.saved_ms is None and candidate.sol_saved_ms is None
    assert any(patch.path == "/workload/requests" for patch in candidate.patches)

    deepseek = _case("deepseek-v3.2", layers=4, tp=8, dp=2, ep=16)
    mtp_report = parse_launch(
        "vllm serve /m --tensor-parallel-size 8 --data-parallel-size 2 "
        "--enable-expert-parallel --speculative-config "
        "'{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":3}'",
        base_case=deepseek, workload=WORKLOAD, speculative=SPEC,
        framework_version="vllm-ascend@0.30.0")
    mtp_ops = (_op("verify", flops=2 * 16 * 4096 * 4096, hbm=16 * 4096 * 2),
               _op("draft", deps=("verify",), flops=2 * 16 * 4096 * 4096,
                   hbm=16 * 4096 * 2, metadata=(("mtp_draft", True),),
                   family="mtp_draft"))
    mtp = compare_launch(mtp_report, plan_ops=mtp_ops, counterfactuals=("MTP",))
    candidate = mtp.counterfactuals[0]
    assert candidate.status == "inapplicable"
    assert candidate.saved_ms is None
    assert any(patch.path == "/runtime/mtp_acceptance_length"
               and patch.after == 1.0 for patch in candidate.patches)


# ------------------------------------------------------ P1-15
def test_p1_15_tpot_divides_total_by_steps_and_commits():
    report = _parse(
        "vllm serve /m --tensor-parallel-size 2 --speculative-config "
        "'{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":3}'",
        speculative=SPEC)
    ops = (_op("a", flops=1e9, hbm=1024),)
    traced = compare_launch(
        report, plan_ops=ops, predicted_ms=2.0, request_metrics=True,
        commits=2.8, request_trace={"arrival_ms": 0, "first_commit_ms": 50,
                                    "decode_steps": 10,
                                    "commit_times": [50.0 + 2.0 * index
                                                     for index in range(11)]})
    tpot = traced.request_metrics[1]
    assert tpot.status == "available"
    assert tpot.predicted_ms == pytest.approx(10 * 2.0 / (10 * 2.8))
    assert tpot.predicted_ms == pytest.approx(2.0 / 2.8)
    assert tpot.sol_ms == pytest.approx(traced.bound.sol_ms / 2.8)
    no_commits = compare_launch(
        report, plan_ops=ops, predicted_ms=2.0, request_metrics=True,
        request_trace={"arrival_ms": 0, "first_commit_ms": 50,
                       "decode_steps": 10})
    assert [metric.status for metric in no_commits.request_metrics] == [
        "unavailable", "unavailable"]
    for steps in (0, -5, 0.5):
        bad = compare_launch(
            report, plan_ops=ops, predicted_ms=2.0, request_metrics=True,
            commits=2.8, request_trace={"arrival_ms": 0, "first_commit_ms": 50,
                                        "decode_steps": steps})
        assert bad.request_metrics[1].status == "unavailable"
    zero_ttft = compare_launch(
        report, plan_ops=ops, prefill_ops=ops, predicted_ms=2.0,
        request_metrics=True, commits=2.8,
        request_trace={"arrival_ms": 0, "first_commit_ms": 0,
                       "decode_steps": 1})
    assert zero_ttft.request_metrics[0].status == "unavailable"
    assert "prefill SOL" in zero_ttft.request_metrics[0].reason


def test_p1_15_s05_rejects_ttft_and_tpot_below_their_bounds():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "fixture"),)
    plan = PlanWork("p", (WorkTerm("a", ("a",),
                                   (("hbm", 1e9, "bytes", ""),)),), peaks)
    context = Context("c", "p", "model_step", "decode", 1, 2, 128,
                      "b", "complete", (SourceIdentity("fixture"),))
    baseline = compare_sol(plan, 2.0, context=context)
    bad_ttft = replace(baseline, request_metrics=(
        RequestMetric("ttft", "available", "r", "fixed_trace", "per_request",
                      1.0, 0.0, True, None),))
    with pytest.raises(ReportValidationError) as nonpositive:
        validate_sol_report(bad_ttft)
    assert "nonpositive_latency" in str(nonpositive.value)
    bad_ttft = replace(baseline, request_metrics=(
        RequestMetric("ttft", "available", "r", "fixed_trace", "per_request",
                      2.0, 1.0, True, None),))
    with pytest.raises(ReportValidationError) as ttft:
        validate_sol_report(bad_ttft)
    assert "ttft_below_prefill_sol" in str(ttft.value)
    bad_tpot = replace(baseline, request_metrics=(
        RequestMetric("tpot", "available", "r", "fixed_trace",
                      "per_committed_token", 2.0, 0.5, True, None),))
    with pytest.raises(ReportValidationError) as tpot:
        validate_sol_report(bad_tpot)
    assert "tpot_below_decode_sol" in str(tpot.value)


# ------------------------------------------------------ P2-17
def test_p2_17_escape_provenance_and_ansi_c_quoting():
    escaped = _parse("VLLM\\_ALL2ALL_BACKEND=mc2 vllm serve /m")
    assert escaped.status == "failed" and escaped.case is None
    assert any("unsupported_launch_command" in item.message
               for item in escaped.diagnostics)
    ansi = _parse("vllm serve /m --x $'ab\\ncd'")
    assert ansi.status == "failed"
    assert any("ansi_c_quoting_not_supported" in item.message
               for item in ansi.diagnostics)
    command = selected_command(parse_shell("VLLM_ALL2ALL_BACKEND=mc2 vllm serve /m"))
    assert dict(command.inline_env)["VLLM_ALL2ALL_BACKEND"] == "mc2"
    applied = _parse("VLLM_ALL2ALL_BACKEND=mc2 vllm serve /m")
    assert applied.case["runtime"]["moe_recipe"] == "mc2"


# ------------------------------------------------------ P2-18
def test_p2_18_visible_npu_count_must_cover_local_world():
    report = _parse("docker run --device /dev/davinci0 image vllm serve /m "
                    "--tensor-parallel-size 2")
    assert report.status != "completed"
    assert any(item.name == "visible_device_count"
               for item in report.unmodeled_or_unsupported)
    enough = _parse("docker run --device /dev/davinci0 --device /dev/davinci1 "
                    "image vllm serve /m --tensor-parallel-size 2")
    assert enough.status == "completed"


def test_p2_18_yaml_complex_key_is_structured():
    if not _yaml_available():
        with pytest.raises(DocumentInputError) as excinfo:
            parse_document("? [a, b]\n: 1\n", format="yaml")
        assert excinfo.value.code == "yaml_parser_unavailable"
        return
    with pytest.raises(DocumentInputError) as excinfo:
        parse_document("? [a, b]\n: 1\n", format="yaml")
    assert excinfo.value.code == "yaml_unhashable_key"


# ------------------------------------------------------ P2-19
def test_p2_19_aggregate_argv_bytes_and_structural_node_cap():
    argv = ["vllm", "serve", "/m"] + ["x" * 100_000] * 84
    with pytest.raises(LaunchSyntaxError) as excinfo:
        parse_argv(argv)
    assert excinfo.value.code == "input_too_large"
    with pytest.raises(DocumentInputError) as excinfo:
        finite_json_loads("[" + ",".join(["1"] * 100_001) + "]")
    assert excinfo.value.code == "input_exceeds_100000_tokens"
    assert len(finite_json_loads("[" + ",".join(["1"] * 99_999) + "]")) == 99_999
    assert len(finite_json_loads("[" + ",".join(["1"] * 100_000) + "]")) == 100_000


# ------------------------------------------------------ P2-20
def test_p2_20_missing_env_and_malformed_scalar_are_both_preserved():
    report = _parse("docker run -e MISSING image vllm serve /m "
                    "--max-num-seqs nope")
    assert report.status == "needs_inputs"
    assert "env:MISSING" in {item.name for item in report.required_inputs}
    assert any(item.code == "invalid_declaration" for item in report.diagnostics)
    assert any(item.name == "--max-num-seqs"
               for item in report.unmodeled_or_unsupported)

    no_base = parse_launch("docker run -e MISSING image vllm serve /m "
                           "--max-num-seqs nope")
    names = {item.name for item in no_base.required_inputs}
    assert {"env:MISSING", "machine", "model", "workload"} <= names
    assert any(item.code == "invalid_declaration" for item in no_base.diagnostics)


def test_p2_20_late_case_error_keeps_accumulated_drafts():
    base = _case(tp=2)
    report = parse_launch(
        "export HCCL_BUFFSIZE=200\nvllm serve /m --max-num-seqs 7",
        base_case=base, workload=WORKLOAD,
        overrides={"/runtime/parallel/tp": 999})
    assert report.status == "failed" and report.case is None
    assert any(item.code.startswith("invalid_base_case")
               for item in report.diagnostics)
    assert any(item.name == "--max-num-seqs" for item in report.applied)
    assert any(item.name == "HCCL_BUFFSIZE"
               for item in report.unmodeled_or_unsupported)


# ------------------------------------------------------ P2-21
def test_p2_21_fused_child_needs_parent_work_conservation():
    case = Case.from_mapping(_case())
    parent = _op("parent", hbm=1024)
    child = _op("child", hbm=4096, metadata=(
        ("fused_into", "parent"), ("fused_work_conserved", True)))
    consumer = _op("consumer", deps=("child",), hbm=1024)
    plan = project_ops(case, (parent, child, consumer), plan_id="fused")
    assert plan.coverage != "complete"
    assert any("fused_work_unproven" in hole for hole in plan.holes)
    assert bound(plan).status == "unavailable"
    covered = _op("child", hbm=1024, metadata=(
        ("fused_into", "parent"), ("fused_work_conserved", True)))
    proven = project_ops(case, (_op("parent", hbm=4096), covered, consumer),
                         plan_id="proven")
    # The Engine still executes the child; its work is included, and the
    # parent-magnitude claim alone cannot certify elision.
    assert proven.coverage != "complete"
    assert any("fused_work_unproven" in hole for hole in proven.holes)
    assert [term.member_ops for term in proven.terms] == [
        ("parent",), ("child",), ("consumer",)]


# ------------------------------------------------------ P2-23
def test_p2_23_json_leaf_spans_use_raw_source_offsets():
    single = ("vllm serve /m --compilation-config "
              "'{\"cudagraph_mode\":\"FULL\",\"capture_sizes\":[1,2]}'")
    report = parse_launch(single, machine="a3", model="qwen3-30b-a3b",
                          workload=WORKLOAD,
                          framework_version="vllm-ascend@0.30.0")
    leaf = next(item for item in report.applied if item.name == "cudagraph_mode")
    assert (leaf.span.line_start, leaf.span.column_start) == (1, 38)
    multi = ("vllm serve /m --compilation-config '{\n"
             "  \"cudagraph_mode\": \"FULL\"\n}'")
    report = parse_launch(multi, machine="a3", model="qwen3-30b-a3b",
                          workload=WORKLOAD,
                          framework_version="vllm-ascend@0.30.0")
    leaf = next(item for item in report.applied if item.name == "cudagraph_mode")
    assert (leaf.span.line_start, leaf.span.column_start) == (2, 3)
    attached = ("vllm serve /m --compilation-config="
                "'{\"cudagraph_mode\":\"FULL\"}'")
    report = parse_launch(attached, machine="a3", model="qwen3-30b-a3b",
                          workload=WORKLOAD,
                          framework_version="vllm-ascend@0.30.0")
    leaf = next(item for item in report.applied if item.name == "cudagraph_mode")
    assert (leaf.span.line_start, leaf.span.column_start) == (1, 38)
    escaped = ("vllm serve /m --compilation-config "
               "\"{\\\"cudagraph_mode\\\":\\\"FULL\\\"}\"")
    report = parse_launch(escaped, machine="a3", model="qwen3-30b-a3b",
                          workload=WORKLOAD,
                          framework_version="vllm-ascend@0.30.0")
    leaf = next(item for item in report.applied if item.name == "cudagraph_mode")
    assert (leaf.span.line_start, leaf.span.column_start) == (1, 38)
