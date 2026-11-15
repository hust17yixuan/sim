"""Tier-0 behavioral checks for launch ingestion and selected-plan SOL.

[Role] Exercise the public launch/SOL/CLI behavior with pure in-memory work.
[Invariants] Isolated HOME/MINDIE_* state, no host env defaults, no network,
  no string-grep assertions; physics is checked from declared work/peaks.
[Timeouts & Locks] Entire module stays well under five seconds; no locks.
[Side-Effect Policy] Writes only under pytest's tmp_path.
"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from mindie_sim import compose_case
from mindie_sim.feature_analysis import compare_launch
from mindie_sim.feature_cli import main as cli_main
from mindie_sim.feature_reports import (
    Context, ReportValidationError, SourceIdentity, canonical_json, to_dict,
    validate_launch_report, validate_sol_report,
)
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.sol import bound, compare_sol, resource_loads
from mindie_sim.prediction.sol_work import Peak, PlanWork, WorkTerm, project_ops
from mindie_sim.schema.operators import OpSpec

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8, "prefix_tokens": 0}
SPEC = {"draft_len": 3, "acceptance": [0.8, 0.6, 0.4]}
BASE = dict(machine="a3", model="deepseek-v3.2", workload=WORKLOAD, speculative=SPEC)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    for name in ("HOME", "MINDIE_CACHE_HOME", "MINDIE_DATA_HOME", "MINDIE_SIM_HOME",
                 "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_MLAPO", raising=False)
    monkeypatch.delenv("HCCL_BUFFSIZE", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    return tmp_path


def _op(name, deps=(), flops=0.0, hbm=0.0, vec=0.0, mte=0.0, link=0.0,
        stream="", metadata=(), family="matmul"):
    return OpSpec(
        op_id=name, family=family, phase="decode", stage=0, layer=0, requests=("r0",),
        semantic_shape=(16, 4096), local_shape=(16, 4096), padded_shape=(16, 4096),
        launch_shape=(16, 4096), flops=flops, hbm_bytes=hbm, vector_ops=vec,
        mte2_bytes=mte, link_bytes=link, deps=tuple(deps), dtype="bf16", stream=stream,
        metadata=tuple(metadata))


def _case_and_prediction(ops, model="qwen3-30b-a3b", **kwargs):
    case = compose_case(model, hardware="a3", graph_capture_mode="full", **kwargs)
    predicted = graph_costs(case, ops, compact=True, trace_limit=0)["duration_ms"]
    return case, predicted


def _context(plan_id, phase="decode", sequences=2):
    return Context("case", plan_id, "model_step", phase, 1, sequences, 128,
                   "selected_execution_plan", "complete", (SourceIdentity("fixture"),))


DS_SCRIPT = """MODEL=/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8
export VLLM_USE_V1=1
vllm serve "$MODEL" \\
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \\
  --max-model-len 8192 --max-num-batched-tokens 4096 --max-num-seqs 16 \\
  --gpu-memory-utilization 0.92 --no-enable-prefix-caching --seed 1024 \\
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \\
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":3}' \\
  --additional-config '{"enable_mlapo":true}' \\
  --api-key sk-super-secret --host 0.0.0.0 --port 8000 \\
  --served-model-name deepseek_v3_2 --trust-remote-code
"""


def test_real_ds_script_continuations_redaction_ep_and_mlapo():
    report = parse_launch(DS_SCRIPT, framework_version="vllm-ascend@0.30.0", **BASE)
    assert report.status == "completed"
    assert report.adapter_id == "vllm-ascend/0.30-static"
    runtime = report.case["runtime"]
    assert runtime["parallel"]["tp"] == 8
    assert runtime["parallel"]["dp"] == 2
    assert runtime["parallel"]["ep"] == 16
    assert runtime["graph_mode"] == "full"
    assert tuple(runtime["piecewise_capture_keys"]) == ()
    assert runtime["graph_capture_keys"]
    assert runtime["kernels"]["lowerings"]["mla_prolog"] == "vllm_ascend_mla_prolog_v3"
    assert runtime["mtp_acceptance_length"] == pytest.approx(2.8)
    assert dict(runtime["metadata"])["draft_mode"] == "mtp"
    assert dict(runtime["metadata"])["draft_proposal_tokens"] == 3
    redacted = [item for item in report.neutral_operational if item.redacted]
    assert [item.value for item in redacted] == ["[REDACTED]"]
    assert "sk-super-secret" not in canonical_json(report)
    validate_launch_report(report)
    assert len(report.setting_ids) == len(set(report.setting_ids))
    for item in (*report.applied, *report.neutral_operational, *report.unmodeled_or_unsupported):
        assert item.span.line_start >= 1 and item.span.line_end >= item.span.line_start


def test_versioned_mlapo_spellings_are_not_reinterpreted():
    env_script = "export VLLM_ASCEND_ENABLE_MLAPO=1\nvllm serve /m --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel\n"
    old = parse_launch(env_script, framework_version="vllm-ascend@0.18.0", **BASE)
    assert old.status == "completed"
    assert old.case["runtime"]["kernels"]["lowerings"]["mla_prolog"]
    new = parse_launch(env_script, framework_version="vllm-ascend@0.30.0", **BASE)
    assert new.status == "partial"
    legacy = [item for item in new.unmodeled_or_unsupported
              if item.name == "VLLM_ASCEND_ENABLE_MLAPO"]
    assert legacy and legacy[0].rule_id == "legacy_mlapo_env_not_interpreted"
    config_script = "vllm serve /m --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel --additional-config '{\"enable_mlapo\":true}'\n"
    new_config = parse_launch(config_script, framework_version="vllm-ascend@0.30.0", **BASE)
    assert new_config.status == "completed"
    old_config = parse_launch(config_script, framework_version="vllm-ascend@0.18.0", **BASE)
    assert old_config.status == "partial"
    assert any(item.name == "--additional-config"
               for item in old_config.unmodeled_or_unsupported)


def test_docker_envelope_env_devices_and_single_sh_unwrap():
    script = """docker run --rm -it --device=/dev/davinci0 --device /dev/davinci3 \\
  -e VLLM_USE_V1=1 --env HCCL_BUFFSIZE=200 -e API_KEY=sk-docker \\
  vllm-ascend:0.18 sh -c 'vllm serve /models/ds --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel'
"""
    report = parse_launch(script, framework_version="vllm-ascend@0.18.0", **BASE)
    assert report.status == "partial"
    assert "container_image_tag_is_provenance_not_a_verified_runtime_version" in report.assumptions
    assert "visible_devices_establish_visibility_only" in report.assumptions
    assert report.case["runtime"]["parallel"]["ep"] == 16
    redacted = [item for item in report.neutral_operational if item.redacted]
    assert redacted and redacted[0].value == "[REDACTED]"
    assert "sk-docker" not in canonical_json(report)


@pytest.mark.parametrize("script,needle", [
    ("vllm serve /m $(whoami)", "command_or_arithmetic_substitution"),
    ("vllm serve /m | tee /tmp/x", "operator_or_expansion"),
    ("vllm serve $UNKNOWN", "unknown_variable"),
    ("vllm serve /m `id`", "command_substitution"),
])
def test_unsafe_shell_syntax_is_rejected_whole(script, needle):
    report = parse_launch(script, **BASE)
    assert report.status == "failed"
    assert report.case is None
    assert any(needle in item.message for item in report.diagnostics)


def test_duplicate_cli_and_env_occurrences_keep_visible_shadowing():
    script = """export HCCL_BUFFSIZE=100
export HCCL_BUFFSIZE=200
vllm serve /m --tensor-parallel-size 2 --tensor-parallel-size 4 --port 1 --port 2
"""
    report = parse_launch(script, **BASE)
    assert report.status == "partial"
    tp = [item for item in report.applied if item.name == "--tensor-parallel-size"]
    assert [(item.value, item.effective) for item in tp] == [(2, False), (4, True)]
    shadowed = [item for item in report.applied + report.neutral_operational
                if item.shadowed_by is not None]
    assert shadowed and all(not item.effective for item in shadowed)
    hccl = [item for item in report.unmodeled_or_unsupported if item.name == "HCCL_BUFFSIZE"]
    assert [(item.value, item.effective) for item in hccl] == [("100", False), ("200", True)]
    assert report.case["runtime"]["parallel"]["tp"] == 4
    prefix_script = ("vllm serve /m -tp 2 --tensor-parallel-size 4 "
                     "--enable-prefix-caching --no-enable-prefix-caching\n")
    prefix = parse_launch(prefix_script, machine="a3", model="qwen3-30b-a3b", workload=WORKLOAD)
    states = [(item.name, item.effective) for item in prefix.applied if "prefix" in item.name]
    assert states == [("--enable-prefix-caching", False), ("--no-enable-prefix-caching", True)]


def test_multiple_commands_require_and_select_command_index():
    script = ("vllm serve /a --tensor-parallel-size 2\n"
              "vllm serve /b --tensor-parallel-size 4\n")
    ambiguous = parse_launch(script, **BASE)
    assert ambiguous.status == "failed"
    selected = parse_launch(script, command_index=1, **BASE)
    assert selected.status == "completed"
    assert selected.command_index == 1
    assert selected.case["runtime"]["parallel"]["tp"] == 4
    out_of_range = parse_launch("vllm serve /m --tensor-parallel-size 2\n",
                                command_index=1, **BASE)
    assert out_of_range.status == "failed"


def test_structured_json_duplicate_keys_and_optional_yaml():
    payload = {"command": ["vllm", "serve", "/m"],
               "env": {"VLLM_USE_V1": "1", "OMP_NUM_THREADS": "10"},
               "args": ["vllm", "serve", "/m", "--tensor-parallel-size", "2"]}
    report = parse_launch(json.dumps(payload), format="json", **BASE)
    assert report.status == "partial"
    assert report.input_format == "json"
    duplicate = parse_launch('{"command": ["vllm","serve","/m"], "command": []}',
                             format="json", **BASE)
    assert duplicate.status == "failed"
    assert duplicate.diagnostics[-1].code == "duplicate_json_key"
    yaml = parse_launch("a: 1", format="yaml", **BASE)
    assert yaml.status == "failed"
    try:
        import yaml as _yaml  # noqa: F401
    except ImportError:
        # The optional extra is absent: the adapter must say so explicitly.
        assert yaml.diagnostics[-1].code == "yaml_parser_unavailable"
    else:
        # With PyYAML installed the document is valid YAML but declares no
        # command; the structured adapter rejects the shape, not the parser.
        assert yaml.diagnostics[-1].code == "structured_command_required"


def test_strict_partial_and_needs_inputs_statuses():
    partial = parse_launch("export HCCL_BUFFSIZE=200\nvllm serve /m --tensor-parallel-size 2\n", **BASE)
    assert partial.status == "partial" and partial.case is not None
    strict = parse_launch("export HCCL_BUFFSIZE=200\nvllm serve /m --tensor-parallel-size 2\n",
                          strict=True, **BASE)
    assert strict.status == "failed" and strict.case is None
    needs = parse_launch("vllm serve /m --tensor-parallel-size 2\n", machine="a3", model="deepseek-v3.2")
    assert needs.status == "needs_inputs" and needs.case is None
    assert [item.name for item in needs.required_inputs] == ["workload"]


def test_worked_example_resource_and_critical_path_bounds():
    peaks = (Peak("fixture:cube", "flops", 1e12, "declared", "explicit_synthetic_capacity"),
             Peak("fixture:vector", "ops", 1e9, "declared", "explicit_synthetic_capacity"),
             Peak("fixture:hbm", "bytes", 1e9, "declared", "explicit_synthetic_capacity"))
    terms = (WorkTerm("cube_branch", ("cube_branch",),
                      (("fixture:cube", 1.2e12, "flops", "declared"),
                       ("fixture:hbm", 0.8e9, "bytes", "declared"))),
             WorkTerm("vector_branch", ("vector_branch",),
                      (("fixture:vector", 0.6e9, "ops", "declared"),
                       ("fixture:hbm", 0.8e9, "bytes", "declared"))))
    plan = PlanWork("two-branches.serial", terms, peaks)
    result = bound(plan)
    assert result.resource_load_ms == pytest.approx(1.6)
    assert result.critical_path_ms == pytest.approx(1.2)
    assert result.sol_ms == pytest.approx(1.6)
    cp_plan = PlanWork("cp", (WorkTerm("a", ("a",),
                                       (("fixture:cube", 2.0e12, "flops", "declared"),
                                        ("fixture:hbm", 0.2e9, "bytes", "declared"))),), peaks)
    assert bound(cp_plan).sol_ms == pytest.approx(2.0)


def test_dense_same_plan_conservation_and_sol_le_engine():
    ops = (_op("a", flops=2 * 16 * 4096 * 4096, hbm=16 * 4096 * 2),
           _op("b", deps=("a",), hbm=32 * 4096 * 2, vec=16 * 4096),
           _op("c", deps=("b",), flops=2 * 16 * 4096 * 4096, hbm=16 * 4096 * 2))
    case, predicted = _case_and_prediction(ops, tp=2, dp=1)
    plan = project_ops(case, ops, plan_id="dense")
    assert plan.coverage == "complete" and not plan.holes
    rows = resource_loads(plan)
    assert sum(row.load_bound_ms for row in rows) >= max(row.load_bound_ms for row in rows)
    result = compare_sol(plan, predicted, context=_context("dense"), request_metrics_enabled=True)
    assert result.status == "completed"
    assert result.bound.sol_ms <= predicted + 1e-12
    assert 0 <= result.efficiency_vs_sol_pct <= 100
    assert result.request_metrics[1].name == "tpot"
    assert result.request_metrics[1].status == "unavailable"
    assert result.request_metrics[0].status == "unavailable"
    assert result.request_metrics[0].reason and result.request_metrics[1].reason
    assert result.throughput_tokens_per_s() == pytest.approx(2 * 1000.0 / predicted)
    traced = compare_sol(plan, predicted, context=_context("dense"),
                         request_metrics_enabled=True, commits=2.8,
                         prefill_predicted_ms=predicted, prefill_sol_ms=result.bound.sol_ms,
                         request_trace={"arrival_ms": 0, "first_commit_ms": predicted,
                                        "decode_steps": 1,
                                        "commit_times": [predicted, predicted * 2]})
    assert [item.status for item in traced.request_metrics] == ["available", "available"]
    assert traced.request_metrics[1].predicted_ms == pytest.approx(predicted / 2.8)


def test_moe_ep_mc2_link_and_pcp_dcp_bounds():
    mc2 = parse_launch(
        "vllm serve /m --tensor-parallel-size 2 --data-parallel-size 2 "
        "--enable-expert-parallel --all2all-backend mc2\n", **BASE)
    assert mc2.status == "completed"
    assert mc2.case["runtime"]["parallel"]["ep"] == 4
    assert mc2.case["runtime"]["moe_recipe"] == "mc2"
    ops = (_op("dispatch", hbm=64 * 4096 * 2, link=64 * 4096 * 2,
               metadata=(("moe_collective", True),)),
           _op("experts", deps=("dispatch",), flops=2 * 32 * 1024 * 4096, hbm=32 * 4096 * 2),
           _op("combine", deps=("experts",), hbm=32 * 4096 * 2, link=32 * 4096 * 2))
    predicted = graph_costs(mc2.case, ops, compact=True, trace_limit=0)["duration_ms"]
    plan = project_ops(mc2.case, ops, plan_id="moe")
    assert any(row.physical_resource_id.startswith("link:") for row in resource_loads(plan))
    assert bound(plan).sol_ms <= predicted + 1e-12

    pcp = parse_launch("vllm serve /m --tensor-parallel-size 2 --prefill-context-parallel-size 2\n",
                       **BASE)
    assert pcp.status == "completed"
    assert pcp.case["runtime"]["parallel"]["pcp"] == 2
    pcp_ops = (_op("qkv", hbm=16 * 4096 * 2, flops=2 * 16 * 4096 * 4096),
               _op("attn", deps=("qkv",), hbm=16 * 4096 * 2, vec=16 * 4096))
    pcp_predicted = graph_costs(pcp.case, pcp_ops, compact=True, trace_limit=0)["duration_ms"]
    pcp_plan = project_ops(pcp.case, pcp_ops, plan_id="pcp")
    assert bound(pcp_plan).sol_ms <= pcp_predicted + 1e-12


def test_declared_physical_floor_and_unavailable_peak():
    peaks = (Peak("cube", "flops", 1e9, "declared", "synthetic"),)
    term = WorkTerm("a", ("a",), (("cube", 1e9, "flops", "declared"),), dispatch_min_ms=0.003)
    plan = PlanWork("floor", (term,), peaks)
    ideal = bound(plan, "ideal_zero")
    physical = bound(plan, "declared_physical")
    assert physical.sol_ms == pytest.approx(ideal.sol_ms + 0.003)
    missing = PlanWork("missing", (WorkTerm("a", ("a",), (("cube2", 1e9, "flops", "declared"),)),),
                       peaks)
    unavailable = bound(missing)
    assert unavailable.status == "unavailable" and unavailable.sol_ms is None
    assert "missing_or_zero_peak" in "".join(unavailable.unavailable_reasons)


def test_fused_child_is_counted_unless_engine_elides_it():
    parent = _op("fused_parent", flops=1e9, hbm=1024)
    child = _op("fused_child", hbm=1024,
                metadata=(("fused_into", "fused_parent"), ("fused_work_conserved", True)))
    consumer = _op("consumer", deps=("fused_child",), hbm=1024)
    case, _ = _case_and_prediction((parent,), tp=2)
    plan = project_ops(case, (parent, child, consumer), plan_id="fused")
    # Execution still services the child op; parent magnitude is not proof of
    # elision, so the child term is retained and the fusion claim is partial.
    assert [term.member_ops for term in plan.terms] == [
        ("fused_parent",), ("fused_child",), ("consumer",)]
    assert plan.terms[2].predecessors == (plan.terms[1].id,)
    assert any("fused_work_unproven" in hole for hole in plan.holes)
    assert plan.coverage != "complete"


def test_counterfactual_signs_and_leave_one_out_semantics():
    report = parse_launch("vllm serve /m --tensor-parallel-size 2 --max-num-seqs 4\n",
                          machine="a3", model="qwen3-30b-a3b", workload=WORKLOAD)
    ops = (_op("cube", flops=4 * 16 * 4096 * 4096, hbm=16 * 4096 * 2),
           _op("vec", hbm=16 * 4096 * 2, vec=4 * 16 * 4096, stream="aux"))
    compared = compare_launch(report, plan_ops=ops, sequences=2,
                              counterfactuals=("Multi-Stream Overlap",))
    delta = compared.counterfactuals[0]
    assert delta.status == "evaluated"
    assert delta.saved_ms < 0  # disabling overlap is slower, negative saving retained
    assert delta.sol_saved_ms <= 0.0  # stream merge can only add a serial edge

    mlapo_report = parse_launch(
        "vllm serve /m --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel "
        "--additional-config '{\"enable_mlapo\":true}'\n",
        framework_version="vllm-ascend@0.30.0", **BASE)
    mlapo_ops = (_op("mla", hbm=1024 * 1024, flops=2 * 16 * 4096 * 128, family="mla_prolog",
                     metadata=(("lowering", "mla_prolog"), ("mlapo_hbm_saved_bytes", 4 * 1024 * 1024))),
                 _op("proj", deps=("mla",), hbm=1024 * 1024, flops=2 * 16 * 4096 * 4096))
    mlapo = compare_launch(mlapo_report, plan_ops=mlapo_ops, sequences=2,
                           counterfactuals=("MLAPO",))
    assert mlapo.counterfactuals[0].status in {"evaluated", "inapplicable"}
    if mlapo.counterfactuals[0].status == "evaluated":
        assert mlapo.counterfactuals[0].saved_ms <= 0.0

    mtp_report = parse_launch(
        "vllm serve /m --tensor-parallel-size 2 --speculative-config "
        "'{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":3}'\n",
        machine="a3", model="deepseek-v3.2", workload=WORKLOAD, speculative=SPEC)
    mtp_ops = (_op("verify", flops=2 * 16 * 4096 * 4096, hbm=16 * 4096 * 2),
               _op("draft", deps=("verify",), flops=2 * 16 * 4096 * 4096, hbm=16 * 4096 * 2,
                   metadata=(("mtp_draft", True),), family="mtp_draft"))
    mtp = compare_launch(mtp_report, plan_ops=mtp_ops, counterfactuals=("MTP",))
    # A caller-supplied four-position verify graph cannot be re-materialized
    # as B*1 without explicit trace geometry, so deleting draft work may not
    # be rewarded as a positive saving.
    assert mtp.counterfactuals[0].status == "inapplicable"
    assert mtp.counterfactuals[0].saved_ms is None
    assert any(patch.path == "/runtime/mtp_acceptance_length" and patch.after == 1.0
               for patch in mtp.counterfactuals[0].patches)


def test_sol_report_rejects_clipped_efficiency():
    peaks = (Peak("hbm", "bytes", 1e9, "declared", "synthetic"),)
    plan = PlanWork("clipped", (WorkTerm("a", ("a",), (("hbm", 1e9, "bytes", "declared"),)),), peaks)
    with pytest.raises(ReportValidationError) as excinfo:
        compare_sol(plan, 0.5, context=_context("clipped"))
    assert "bound_exceeds_prediction" in str(excinfo.value)


def test_cli_exit_codes_and_clean_io(tmp_path, capsys):
    script = tmp_path / "deploy.sh"
    script.write_text("vllm serve /m --tensor-parallel-size 2 --max-num-seqs 4\n", encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD), encoding="utf-8")
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([
        {"op_id": "a", "flops": 536870912, "hbm_bytes": 131072},
        {"op_id": "b", "deps": ["a"], "vector_ops": 65536, "hbm_bytes": 131072},
    ]), encoding="utf-8")
    common = ["--machine", "a3", "--model", "qwen3-30b-a3b", "--workload", str(workload),
              "--format", "markdown"]
    assert cli_main(["inspect-launch", str(script), *common]) == 0
    assert "Launch inspection: completed" in capsys.readouterr().out
    assert cli_main(["sol", str(script), "--ops-json", str(ops), *common]) == 0
    assert "SOL comparison: completed" in capsys.readouterr().out
    assert cli_main(["feature-deltas", str(script), "--ops-json", str(ops),
                     "--features", "Multi-Stream Overlap", *common]) == 0
    assert "Feature counterfactuals" in capsys.readouterr().out
    strict = tmp_path / "strict.sh"
    strict.write_text("export HCCL_BUFFSIZE=200\nvllm serve /m --tensor-parallel-size 2\n",
                      encoding="utf-8")
    assert cli_main(["inspect-launch", str(strict), "--strict", *common]) == 2
    assert cli_main(["inspect-launch", str(script), "--model", "qwen3-30b-a3b",
                     "--machine", "a3"]) == 3


def test_reports_are_canonical_and_partitioned():
    report = parse_launch("export HCCL_BUFFSIZE=200\nvllm serve /m --tensor-parallel-size 2\n", **BASE)
    text = canonical_json(report)
    assert "\r" not in text
    decoded = json.loads(text)
    assert decoded["schema_version"] == "launch-parse/1"
    ids = [item["id"] for bucket in ("applied", "neutral_operational",
                                     "unmodeled_or_unsupported") for item in decoded[bucket]]
    assert sorted(ids) == sorted(decoded["setting_ids"])
    assert len(ids) == len(set(ids))
    validation = validate_launch_report(report)
    assert validation == ()
    assert to_dict(report)["status"] == "partial"


