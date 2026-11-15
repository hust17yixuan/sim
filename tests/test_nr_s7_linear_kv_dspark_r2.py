"""NR-S7 public evaluation, reconciliation and MCP envelope tests.

[Role]
  Prove the model-derived NZ/MLAPO/PD gates, the pre-padded and per-stage
  memory reconciliation, the active-only DSpark retention and auxiliary
  scratch integration, the shared prefix restore rejection, the separate
  one-time prefix replay and the complete bounded MCP envelope.

[Invariants]
  Assertions are exact conservation/status identities or bounded-token
  properties; the HOME observation only proves no file is created.

[Timeouts & Locks]
  Pure in-memory arithmetic plus one empty temporary HOME directory; no
  device, subprocess or network access.

[Side-Effect Policy]
  Builds immutable records; creates nothing outside pytest's temporary root.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim.hybrid_kv_state_mcp import (
    MCP_MAX_TOKENS, envelope_token_upper_bound, hybrid_kv_state_mcp_envelope,
    summary_token_count)
from mindie_sim.hybrid_kv_state_options import HybridKvStateOptions
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

FORBIDDEN = ("/Users/", "/Volumes/", "$HOME", "~/", "\\", "龘")


def fix12_test_evidence(log, counts, passed):
    from collections import Counter
    from xml.etree import ElementTree as ET
    root = ET.parse(log.parent.parent / "pytest.xml").getroot()
    cases = root.findall(".//testcase")
    assert len(cases) == passed
    assert all(root.find(f".//{tag}") is None for tag in ("failure", "error", "skipped"))
    per_file = Counter(case.attrib["classname"].replace(".", "/") + ".py" for case in cases)
    assert per_file == counts, (per_file, counts)
    suites = root.findall(".//testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == passed
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    return {"tests": passed, "failures": 0, "errors": 0, "skips": 0,
            "seconds": sum(float(s.attrib["time"]) for s in suites)}


def fix12_source_evidence(artifact_dir):
    import hashlib
    names = ("r12_minimal_extension_results.json", "r12_extension_compatibility_results.json",
             "r12_fresh_contracts_results.json", "r12_fresh_contracts_inventory_adapter.json")
    minimal, extension, fresh, adapter = (json.loads((artifact_dir / name).read_text()) for name in names)
    assert minimal["outcome"] == "accepted" and minimal["base_fields_unchanged"]
    for key, count in (("direct", 84), ("consumers", 360), ("controls", 60)):
        assert len(extension[key]) == count and all(r["outcome"] == "accepted" for r in extension[key])
    assert all(r["body_entries"] for r in extension["consumers"])
    counts = fresh["counts"]
    assert counts == {"base_field_inventory_checks": 8000, "valid_subtypes": 352,
        "invalid_nested": 704, "missing_base_fields": 500, "invalid_consumers": 5280, "workload_first": 80}
    assert not fresh["failures"] and not fresh["consumer_body_entries"]
    assert len(fresh["families"]) == 22 and adapter["production_helper_unchanged"]
    return {"minimal": minimal, "extension": extension["summary"], "contracts": counts,
        "contract_seconds": fresh["seconds"], "consumer_body_entries": 0,
        "inventory_adapter": adapter,
        "artifacts": {name: {"bytes": len((artifact_dir / name).read_bytes()),
            "sha256": hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()} for name in names}}


def fix12_report(data):
    evidence = data["fix11"]["field_ownership"]
    extension, contracts = evidence["extension"], evidence["contracts"]
    hashes = "\n".join(f"| `{name}` | {row['bytes']} | `{row['sha256']}` |"
                       for name, row in evidence["artifacts"].items())
    junit = data["tests"]["junit"]
    return f"""**R12-P2-01 — Base rules apply only to base-owned fields**

Each authoritative (name, annotation) entry now retains its registered base owner
as a frozenset. Overlapping fields keep separate entries, so even a metadata
exception in one base cannot weaken another base's contract. Subclass additions
have no base owners: their own types are checked without inherited numeric or
special-field rules. Any/object payloads remain opaque; Union and typed containers
still recurse. Missing values reject, Case.workload remains first, and authoritative
base-domain checks still run after field validation.

The adapted ordinary-constructor reproduction passes. The independent extension
matrix accepts **{extension['direct_valid_type_calls']}/84** direct additions,
**{extension['consumer_valid_type_calls']}/360** valid public calls and
**{extension['controls']}/60** controls, with zero unexpected ModelingError.
Permanent tests cover all six affected families with seven signed/zero/positive/
Any/object/Union variants, one/four inheritance levels and all ten consumer forms
in three capture modes. Each typed carrier includes all six extended records.
Additional controls cover unrelated field-name collisions, multiple registered
bases, overlapping metadata contracts, missing additions, and opaque versus typed
nested dataclasses/tuples/lists/dicts.

The fresh R12 contracts retain **{contracts['invalid_consumers']}/5,280** pre-body
ModelingErrors with zero body entries; **{contracts['valid_subtypes']}/352** valid
subtypes; **{contracts['invalid_nested']}/704** invalid direct rejections;
**{contracts['missing_base_fields']}/500** missing-property rejections; and
**{contracts['workload_first']}/80** workload-first checks. Its recorded elapsed
time is {evidence['contract_seconds']:.6f} seconds. The R12 probe source is byte
identical to the audit copy. The scratch launcher projects only the probe's private
inventory import from triples to legacy pairs; all assertions and production
validators remain unchanged. It also checks every triple's base ownership.
The R11 redeclaration probe uses the same inventory-only adapter.

Collection, pytest summary and JUnit reconcile at **{junit['tests']}** tests,
**0 failures, 0 errors, 0 skips**. JUnit records {junit['seconds']:.6f} seconds;
its timing includes reporting overhead and can differ from pytest's summary.
Fresh evidence is nested inside the retained `fix11` group, preserving 16 groups.

| FIX12 evidence artifact | Bytes | SHA-256 |
|---|---|---|
{hashes}

"""


def _case(key="glm-5.3-flash", *, tp=8, pp=1, pcp=1, dcp=1,
          max_model_len=4096, max_seqs=8, memory=None, prompt=1024,
          sku="Ascend910_9362", topology="a3_16p"):
    runtime = {"tp": tp, "dp": 1, "ep": 1, "pp": pp, "pcp": pcp, "cp": dcp,
               "max_model_len": max_model_len, "max_num_seqs": max_seqs,
               "max_num_batched_tokens": 16384, "kv_block_size": 128}
    if memory is not None:
        runtime["memory"] = memory
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku,
                    "num_devices": 64, "ranks_per_node": 16,
                    "topology_preset": topology},
        "model": {"catalog_key": key},
        "runtime": runtime,
        "workload": {"requests": [{"request_id": "r",
                                   "prompt_tokens": prompt,
                                   "output_tokens": 16}]}})


def _glm(**overrides):
    values = dict(use_replayssm=True, num_spec=4, mamba_cache_mode="none",
                  enable_prefix_caching=False, batch=8, sequence_tokens=2048,
                  accept_rate=0.8, kernel_launch_ms=0.02,
                  graph_analytical_bound=True)
    values.update(overrides)
    return HybridKvStateOptions(**values)


def _kimi(**overrides):
    values = dict(use_replayssm=False, num_spec=3, mamba_cache_mode="align",
                  enable_prefix_caching=True, batch=2, sequence_tokens=4096,
                  prefix_hit_tokens=1024, kernel_launch_ms=0.02,
                  graph_analytical_bound=True)
    values.update(overrides)
    return HybridKvStateOptions(**values)


def _ds41(**overrides):
    values = dict(use_replayssm=False, num_spec=3, mamba_cache_mode="align",
                  enable_prefix_caching=True, batch=2, sequence_tokens=4096,
                  prefix_hit_tokens=1024, swa_bounded_replay=True,
                  kernel_launch_ms=0.02, dspark_active=True,
                  graph_analytical_bound=True)
    values.update(overrides)
    return HybridKvStateOptions(**values)


# ------------------------------------------------------- model-derived gates


def test_sparse_sfa_catalog_case_rejects_enable_kv_nz():
    report = evaluate_hybrid_kv_state(_case("deepseek-v3.2"), HybridKvStateOptions(
        enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
        kv_connector="MooncakeConnectorV2", batch=2, sequence_tokens=2048))
    assert report["status"] == "rejected"
    assert "kv_nz_rejects_sparse_sfa" in report["errors"]
    assert report["model"]["sparse_sfa"] is True
    assert report["model"]["mlapo_eligible"] is False
    assert report["layout"].get("kv_nz") is None


def test_glm_nz_reports_a_real_mlapo_eligible_to_disabled_transition():
    # A5 exposes MLAPO_NATIVE_WEIGHTS, so unquantized BF16 GLM is eligible.
    case = _case(sku="Ascend950DT_9581", topology="a5_pod")
    report = evaluate_hybrid_kv_state(case, HybridKvStateOptions(
        enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
        kv_connector="MooncakeConnectorV2", batch=2, sequence_tokens=2048))
    assert report["errors"] == []
    nz = report["layout"]["kv_nz"]
    assert report["model"]["hardware_arch"] == "A5"
    assert report["model"]["mlapo_quant"] == "native"
    assert report["model"]["mlapo_eligible"] is True
    assert nz["mlapo_eligible"] is True
    assert nz["mlapo_enabled"] is False
    assert nz["mlapo_disabled_by_nz"] is True
    assert nz["layout"]["planes"][0]["padding_bytes"] == 0
    validation = report["layout"]["kv_nz_validation"]
    assert validation["pd_decode_node"] is True


def test_a3_bf16_glm_is_not_mlapo_eligible_so_nz_does_not_claim_it():
    report = evaluate_hybrid_kv_state(_case(), HybridKvStateOptions(
        enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
        kv_connector="MooncakeConnectorV2", batch=2, sequence_tokens=2048))
    assert report["errors"] == []
    assert report["model"]["hardware_arch"] == "A3"
    assert report["model"]["mlapo_quant"] == "native"
    assert report["model"]["mlapo_eligible"] is False
    nz = report["layout"]["kv_nz"]
    assert nz["mlapo_eligible"] is False
    assert nz["mlapo_enabled"] is False
    assert nz["mlapo_disabled_by_nz"] is False


def test_kimi_fused_qkv_a_gate_disables_mlapo_even_on_a5():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192, sku="Ascend950DT_9581",
              topology="a5_pod"),
        _kimi(enable_kv_nz=True, pd_disaggregated=True,
              is_kv_consumer=True, kv_connector="MooncakeConnectorV2"))
    assert report["model"]["hardware_arch"] == "A5"
    assert report["model"]["mlapo_eligible"] is False
    assert report["layout"]["kv_nz"]["mlapo_disabled_by_nz"] is False


def test_supported_projection_quantization_makes_mlapo_eligible_on_a3():
    from mindie_sim.hybrid_kv_state_options import (
        model_mlapo_eligible, projection_quant_class)
    import dataclasses as dc
    model = _case().model
    facts = dict(model.catalog_facts)
    facts["quantization"] = {**dict(facts.get("quantization") or {}),
                             "name": "w8a8_mxfp8", "weight_format": "fp8_e4m3"}
    quantized = dc.replace(model, catalog_facts=tuple(facts.items()))
    assert projection_quant_class(quantized) == "w8a8_mxfp8"
    assert model_mlapo_eligible(quantized, hardware_arch="A3",
                                decode_instance=True) is True
    assert model_mlapo_eligible(quantized, hardware_arch="A3",
                                decode_instance=False) is False


def test_pd_producer_and_non_pd_nz_are_rejected():
    producer = evaluate_hybrid_kv_state(_case(), HybridKvStateOptions(
        enable_kv_nz=True, pd_disaggregated=True, kv_role="kv_producer",
        kv_connector="MooncakeConnectorV2", batch=1, sequence_tokens=128))
    assert "kv_nz_rejected_on_prefill_node" in producer["errors"]
    non_pd = evaluate_hybrid_kv_state(_case(), HybridKvStateOptions(
        enable_kv_nz=True, batch=1, sequence_tokens=128))
    assert "kv_nz_requires_pd_deployment" in non_pd["errors"]


def test_contradictory_mlapo_override_is_rejected():
    with pytest.raises(ModelingError, match="override_contradicts_model"):
        evaluate_hybrid_kv_state(_case("deepseek-v3.2"), HybridKvStateOptions(
            batch=1, sequence_tokens=128, is_sparse_sfa=True,
            mlapo_eligible=True))


# --------------------------------------------------- pre-padded baseline


def _kda_indices(model):
    return tuple(plan.index for plan in model.layer_plans
                 if plan.attention in {"kda", "gdn"})


def test_already_priced_state_pages_are_not_double_counted():
    case = _case(tp=64, memory=None)
    kda = _kda_indices(case.model)
    groups = ({"layers": kda, "tokens_per_page": 128,
               "bytes_per_layer_page": 131_072,
               "components": ("recurrent_state", "conv_state"),
               "fixed_pages": 1},)
    padded = _case(tp=64, memory={"cache_page_groups": groups},
                   prompt=128)
    report = evaluate_hybrid_kv_state(padded, _glm(
        batch=1, sequence_tokens=128))
    pool = report["layout"]["mixed_pool"]
    # The declared per-layer state pages are already priced; the net-once
    # delta is only the modeled pool minus that declared baseline.
    assert pool["main_class_baseline_per_stage"] == [5_898_240.0]
    assert pool["baseline_replaced_per_stage"] == [6_033_408.0]
    assert report["memory"]["pool_replacement_delta_bytes"] == (
        pool["physical_pool_bytes"] - 6_033_408)
    assert report["memory"]["total_increment_bytes"] == 9_039_872
    unpadded = evaluate_hybrid_kv_state(
        _case(tp=8), _glm(batch=8, sequence_tokens=2048))
    assert unpadded["memory"]["total_increment_bytes"] > 0
    assert unpadded["memory"]["total_increment_bytes"] > report["memory"][
        "total_increment_bytes"]


def test_glm_pool_prices_slots_not_the_layer_sum():
    report = evaluate_hybrid_kv_state(_case(tp=8), _glm(
        batch=8, sequence_tokens=2048))
    pool = report["layout"]["mixed_pool"]
    assert pool["main_slot_count"] == 12
    assert pool["global_recurrent_group_sizes"] == [12, 11, 11]
    assert pool["main_page_bytes"] == 608_256
    assert pool["main_slot_count"] * pool["main_page_bytes"] == 7_299_072
    # Attention demand 128 + one RecoverSSM page per group per sequence.
    assert pool["attention_blocks"] == 128
    assert pool["recurrent_blocks"] == 24
    # The GLM compressor-tail group adds one global ID per request.
    assert pool["tail_blocks"] == 8
    assert pool["blocks"] == 160
    assert pool["physical_pool_bytes"] == 160 * (12 * 608_256 + 11 * 131_072)
    assert pool["physical_pool_bytes"] >= report["linear_state"][
        "memory"]["total_bytes"]
    assert pool["state_pages_per_sequence"] == 1
    assert pool["state_pages_total"] == 8


# ------------------------------------------------------------- PP ownership


def test_pp_partition_owns_state_and_net_delta_per_stage():
    case = parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 64, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"catalog_key": "glm-5.3-flash"},
        "runtime": {"tp": 8, "dp": 1, "ep": 1, "pp": 2, "pcp": 1, "cp": 1,
                    "max_model_len": 4096, "max_num_seqs": 8,
                    "max_num_batched_tokens": 16384, "kv_block_size": 128,
                    "pp_layer_partition": [1, 44]},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16}]}})
    report = evaluate_hybrid_kv_state(case, HybridKvStateOptions(
        use_replayssm=False, num_spec=4, mamba_cache_mode="none", batch=2,
        sequence_tokens=2048, graph_analytical_bound=True))
    stages = report["linear_state"]["stage_state_bytes"]
    assert stages == [567_296.0, 33 * 567_296.0]
    deltas = report["memory"]["net_delta_per_stage"]
    assert len(deltas) == 2
    assert deltas[0] > 0 and deltas[1] > 0
    assert report["memory"]["total_increment_bytes"] == pytest.approx(
        deltas[1])


# --------------------------------------------------------- DSpark activity


def test_dspark_modeling_is_inactive_without_active_speculation():
    report = evaluate_hybrid_kv_state(_case("kimi-k3", max_model_len=8192),
                                      _kimi())
    assert report["dspark"]["present"] is True
    assert report["dspark"]["active"] is False
    assert report["dspark"]["prefix"] is None
    assert report["memory"]["dspark_delta_per_stage"] == [0.0]


def test_bounded_draft_retention_changes_the_adjusted_peak():
    peaks = []
    for extra in (0, 8, 256):
        report = evaluate_hybrid_kv_state(
            _case("deepseek-v4.1-flash", max_model_len=8192),
            _ds41(extra_retained_tokens=extra))
        assert report["errors"] == []
        assert report["dspark"]["active"] is True
        assert report["dspark"]["swa"]["retained_tokens"] == (
            128, 256, 384)[(0, 8, 256).index(extra)]
        assert report["dspark"]["modeled_draft_kv_bytes"] == (
            884_736, 1_769_472, 2_654_208)[(0, 8, 256).index(extra)]
        peaks.append(report["memory"]["adjusted_peak_hbm_bytes"])
    # One added 128-token block costs 442,368 B per sequence, for two
    # active sequences.
    assert peaks[1] - peaks[0] == pytest.approx(884_736)
    assert peaks[2] - peaks[1] == pytest.approx(884_736)


def test_gqa_draft_row_includes_k_and_v_and_local_tp_heads():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(dspark_active=True))
    assert report["dspark"]["active"] is True
    assert report["dspark"]["modeled_draft_kv_bytes"] == 2 * 4096 * 5 * 512
    assert report["dspark"]["modeled_draft_kv_bytes"] == 20_971_520


def test_active_aux_scratch_changes_the_peak_deterministically():
    one = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(dspark_active=True, dspark_aux_layer_ids=(7,)))
    two = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(dspark_active=True, dspark_aux_layer_ids=(7, 23)))
    assert two["memory"]["aux_scratch_bytes"] == (
        2 * one["memory"]["aux_scratch_bytes"])
    assert two["memory"]["adjusted_peak_hbm_bytes"] - one["memory"][
        "adjusted_peak_hbm_bytes"] == pytest.approx(
            one["memory"]["aux_scratch_bytes"])


def test_prefix_plan_reports_per_rank_and_aggregate_scope():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(dspark_active=True, dspark_aux_layer_ids=(7,)))
    prefix = report["dspark"]["prefix"]
    assert prefix["scope"] == "per_rank"
    assert prefix["aggregate_storage_bytes"] == 8 * prefix[
        "per_rank_storage_bytes"]
    assert prefix["aux_scratch_persistent"] is False
    assert "per_rank" in prefix["transfer_scope"]


def test_prefix_restore_beyond_prompt_extent_is_rejected():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(dspark_active=True, prefix_hit_tokens=4096,
              sequence_tokens=128, batch=1))
    assert report["status"] == "rejected"
    assert any("hit_exceeds_prompt" in code for code in report["errors"])
    assert report["transfer"]["dspark_prefix_per_rank_bytes"] == 0.0


# ------------------------------------------------------ replay vs decode


def test_prefix_replay_is_reported_apart_from_steady_state_decode():
    report = evaluate_hybrid_kv_state(_case(tp=8), _glm(
        enable_prefix_caching=True, mamba_cache_mode="align",
        prefix_hit_tokens=1000, batch=2))
    assert report["layout"]["mixed_pool"]["state_pages_per_sequence"] == 16
    decode = report["decode"]
    assert decode["prefix_replay_ms"] > 0
    assert decode["step_overhead_ms"] == pytest.approx(
        decode["linear_ms"] + decode["precopy_ms"] + decode["dspark_block_ms"])
    assert decode["prefix_replay_ms"] not in (
        decode["linear_ms"], decode["precopy_ms"], decode["dspark_block_ms"])


# ----------------------------------------------------------- MCP envelope


def _compact(envelope):
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True,
                      allow_nan=False)


def test_mcp_envelope_is_whole_json_bounded_and_path_free():
    for case, options in (
            (_case(tp=16), _glm()),
            (_case(tp=64), _glm()),
            (_case("kimi-k3", max_model_len=8192),
             _kimi(enable_kv_nz=True, pd_disaggregated=True,
                   is_kv_consumer=True,
                   kv_connector="MooncakeConnectorV2")),
            (_case("deepseek-v4.1-flash", max_model_len=8192), _ds41())):
        report = evaluate_hybrid_kv_state(case, options)
        envelope = hybrid_kv_state_mcp_envelope(report)
        compact = _compact(envelope)
        assert envelope_token_upper_bound(envelope) == len(compact.encode())
        assert len(compact.encode()) <= MCP_MAX_TOKENS
        assert summary_token_count(envelope["summary"]) <= MCP_MAX_TOKENS
        assert envelope["status"] == report["status"]
        for forbidden in FORBIDDEN:
            assert forbidden not in compact


def test_mcp_envelope_sanitizes_pathological_unicode_and_paths():
    report = evaluate_hybrid_kv_state(_case(), _glm())
    report["model"]["model_type"] = "龘" * 2000
    report["model"]["name"] = "/Users/private/model"
    report["warnings"] = ["/Users/private/secret", "龘" * 500]
    report["diagnostics"] = ["/Users/private/secret", "a_b"]
    envelope = hybrid_kv_state_mcp_envelope(report)
    compact = _compact(envelope)
    assert len(compact.encode()) <= MCP_MAX_TOKENS
    assert "龘" not in compact
    assert "/Users" not in compact
    assert "private" not in compact


def test_mcp_envelope_keeps_errors_and_diagnostics_first():
    report = evaluate_hybrid_kv_state(_case("deepseek-v3.2"),
                                      HybridKvStateOptions(
        enable_kv_nz=True, pd_disaggregated=True, is_kv_consumer=True,
        kv_connector="MooncakeConnectorV2", batch=1, sequence_tokens=128))
    envelope = hybrid_kv_state_mcp_envelope(report)
    assert envelope["status"] == "rejected"
    assert "kv_nz_rejects_sparse_sfa" in envelope["errors"]
    assert envelope["diagnostics"][0] == "kv_nz_rejects_sparse_sfa"
    assert len(_compact(envelope).encode()) <= MCP_MAX_TOKENS


def test_mcp_budget_below_mandatory_raises_and_large_budget_clamps():
    report = evaluate_hybrid_kv_state(_case(), _glm())
    with pytest.raises(ModelingError, match="budget_below_mandatory"):
        hybrid_kv_state_mcp_envelope(report, max_tokens=1)
    oversized = hybrid_kv_state_mcp_envelope(report, max_tokens=10_000)
    assert oversized["max_tokens"] == MCP_MAX_TOKENS
    with pytest.raises(ModelingError, match="summary_not_text"):
        summary_token_count(None)
    assert summary_token_count("") == 0


def test_mcp_real_bpe_encodings_stay_within_250():
    tiktoken = pytest.importorskip("tiktoken")
    report = evaluate_hybrid_kv_state(_case(tp=16), _glm())
    compact = _compact(hybrid_kv_state_mcp_envelope(report))
    for name in ("cl100k_base", "o200k_base"):
        encoding = tiktoken.get_encoding(name)
        assert len(encoding.encode(compact)) <= MCP_MAX_TOKENS


# ------------------------------------------------------------- strict input


def test_options_false_is_not_silently_replaced():
    with pytest.raises(ModelingError, match="options_invalid"):
        evaluate_hybrid_kv_state(_case(), False)


@pytest.mark.parametrize("value", [True, 1.9, float("nan"), float("inf")])
def test_runtime_max_num_seqs_is_validated_before_cast(value):
    with pytest.raises(ModelingError):
        evaluate_hybrid_kv_state(
            _case(max_seqs=value),
            HybridKvStateOptions(sequence_tokens=128,
                                 graph_analytical_bound=True))


def test_home_is_not_polluted_by_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    report = evaluate_hybrid_kv_state(_case(), _glm(batch=2))
    hybrid_kv_state_mcp_envelope(report)
    assert list(tmp_path.iterdir()) == []


def test_bounded_draft_replaces_a_priced_baseline_allocation_once():
    case = parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 64, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"catalog_key": "deepseek-v4.1-flash"},
        "runtime": {"tp": 8, "dp": 1, "ep": 1, "pp": 1,
                    "max_model_len": 8192, "max_num_seqs": 8,
                    "max_num_batched_tokens": 16384, "kv_block_size": 128,
                    "metadata": {"draft_mode": "dspark"}},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16}]}})
    report = evaluate_hybrid_kv_state(case, _ds41(extra_retained_tokens=8))
    baseline = report["dspark"]["baseline_draft_kv_per_stage"][-1]
    modeled = report["dspark"]["modeled_draft_kv_bytes"]
    assert baseline > modeled
    assert report["memory"]["dspark_delta_per_stage"] == [-(
        baseline - modeled)]
    assert report["memory"]["net_delta_per_stage"][-1] == pytest.approx(
        report["memory"]["pool_delta_per_stage"][-1] + modeled - baseline)
    assert report["memory"]["aux_scratch_per_stage"][-1] == pytest.approx(
        report["memory"]["aux_scratch_bytes"])
    from mindie_sim.memory_breakdown import (
        analyze_memory_breakdown, apply_state_pool_delta)
    ledger = analyze_memory_breakdown(
        case, sequence_tokens=4096, sequences=2,
        graph_analytical_bound=True, kv_transfer_staging_per_stage_bytes=(0.0,))
    adjusted = apply_state_pool_delta(
        ledger,
        per_stage_bytes=tuple(report["memory"]["net_delta_per_stage"]),
        transient_per_stage_bytes=tuple(
            report["memory"]["aux_scratch_per_stage"]),
        allow_negative=True)
    assert adjusted.peak_hbm_bytes == report["memory"][
        "adjusted_peak_hbm_bytes"]


def test_aux_layer_ids_outside_the_model_are_rejected():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192),
        _kimi(dspark_active=True, dspark_aux_layer_ids=(999,)))
    assert report["status"] == "rejected"
    assert "dspark_aux_layer_out_of_range:999" in report["errors"]


def test_plan_aux_layers_must_equal_the_key_aux_layers():
    from mindie_sim.dspark_prefix_cache import (
        DSparkPrefixPage, ModelIdentity, dspark_prefix_keys,
        dspark_prefix_page_bytes, dspark_prefix_plan)
    identity = ModelIdentity("m", "bf16", "bf16")
    keys = dspark_prefix_keys(
        target=identity, draft=identity, aux_layers=(7,), tp_size=1,
        pp_size=1, block_size=128, prefix_match_unit=128, draft_width=5)
    geometry = dspark_prefix_page_bytes(
        pages=[DSparkPrefixPage("d", 0, 8, 128, 128, 1, 128, 1)],
        num_blocks=8, block_size=128, hash_block_size=128, group_ids=(0,),
        block_sizes=(128,))
    with pytest.raises(ModelingError, match="aux_layers_mismatch_key"):
        dspark_prefix_plan(
            keys=keys, page_geometry=geometry, aux_layer_ids=(7, 23),
            target_hidden_size=128, target_dtype="bf16", tokens=128,
            prompt_tokens=128, block_hash_count=1)


def test_dcp_block_sharding_changes_pool_blocks_per_rank():
    single = evaluate_hybrid_kv_state(_case(tp=8), HybridKvStateOptions(
        use_replayssm=False, num_spec=4, batch=8, sequence_tokens=2048,
        graph_analytical_bound=True))
    sharded = evaluate_hybrid_kv_state(
        _case(tp=8, dcp=2), HybridKvStateOptions(
            use_replayssm=False, num_spec=4, batch=8, sequence_tokens=2048,
            graph_analytical_bound=True))
    assert single["layout"]["mixed_pool"]["attention_blocks"] == 128
    # Standard mode keeps K + 1 = 5 rollback state slots per sequence and
    # group: 3 groups * 5 * batch 8.
    assert single["layout"]["mixed_pool"]["recurrent_blocks"] == 120
    assert single["layout"]["mixed_pool"]["blocks"] == 256
    assert sharded["layout"]["mixed_pool"]["attention_blocks"] == 64
    assert sharded["layout"]["mixed_pool"]["recurrent_blocks"] == 120
    assert sharded["layout"]["mixed_pool"]["blocks"] == 192


def test_explicit_inactive_dspark_cannot_contradict_runtime():
    case = parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 64, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"catalog_key": "deepseek-v4.1-flash"},
        "runtime": {"tp": 8, "dp": 1, "ep": 1, "pp": 1,
                    "max_model_len": 8192, "max_num_seqs": 8,
                    "max_num_batched_tokens": 16384, "kv_block_size": 128,
                    "dspark": True},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16}]}})
    with pytest.raises(ModelingError, match="dspark_override_contradicts"):
        evaluate_hybrid_kv_state(case, _ds41(dspark_active=False))


def test_dcp_report_cost_scales_with_batch_not_just_a_block():
    one = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192, dcp=2),
        _kimi(batch=1, sequence_tokens=2048, dspark_active=True,
              dspark_block_size=7))
    eight = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192, dcp=2),
        _kimi(batch=8, sequence_tokens=2048, dspark_active=True,
              dspark_block_size=7))
    assert one["dspark"]["dcp_block"] is not None
    assert one["dspark"]["dcp_block"]["gather_bytes"] > 0
    assert eight["dspark"]["dcp_block"]["gather_bytes"] == (
        8 * one["dspark"]["dcp_block"]["gather_bytes"])
    assert eight["dspark"]["dcp_block"]["total_ms"] > (
        one["dspark"]["dcp_block"]["total_ms"])


def test_dcp_report_window_bounded_index_and_batch_scaling():
    one = evaluate_hybrid_kv_state(
        _case("deepseek-v4.1-flash", max_model_len=8192, dcp=2),
        _ds41(batch=1, sequence_tokens=2048, dspark_block_size=5))
    eight = evaluate_hybrid_kv_state(
        _case("deepseek-v4.1-flash", max_model_len=8192, dcp=2),
        _ds41(batch=8, sequence_tokens=2048, dspark_block_size=5))
    left, right = one["dspark"]["dcp_block"], eight["dspark"]["dcp_block"]
    assert left["index_width"] == 256 and left["visible_tokens"] == 133
    assert left["gathered_heads"] == 16
    assert left["gather_bytes"] == 138_240
    assert left["combine_bytes"] == 123_840
    assert left["index_bytes"] == 15_420
    assert left["index_bytes"] == 15 * (256 * 4 + 4)
    assert right["gather_bytes"] == 8 * left["gather_bytes"]
    assert right["index_bytes"] == 8 * left["index_bytes"]
    assert right["total_ms"] > left["total_ms"]
