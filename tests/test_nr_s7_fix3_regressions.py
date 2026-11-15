"""Permanent Round-3 counterexamples for the NR-S7 source-exact closure.

[Role]
  One regression module per adversarial finding: F01-F09, F11, F14-F16 in
  ``astra_review_r3.md``.  Each test reproduces the review's concrete
  counterexample and asserts the source-derived result, not merely that an
  error is raised.

[Invariants]
  * No test trusts a scalar the implementation could forge: F09 reads the
    immutable registration evidence, F04 checks the global ID equation.
  * F11 checks read/write/FLOP/timing consistency from one trace record and
    the public A5/strided launch geometry.
  * F05 handles a limiting-rank change followed by a second delta.
  * F07 distinguishes a labeled analytical split from a source grouping in
    both strict and non-strict modes.

[Timeouts & Locks]
  Pure in-memory arithmetic plus the real cached tokenizer; no device or
  network access.

[Side-Effect Policy]
  Builds fresh records; creates no files.
"""
from __future__ import annotations

import dataclasses as dc
import json
from types import SimpleNamespace as NS

import pytest

from nr_s7_fix2_support import (
    _base_specs, _case, _config, _dcp, _ds41, _geometry, _glm, _kimi,
    _layer, _keys)
from mindie_sim.dspark_dcp_cost import DcpDraftBlockCost
from mindie_sim.dspark_kv_physics import kimi_k3_dspark_grouping
from mindie_sim.dspark_prefix_cache import (
    DSparkPrefixPlan, dspark_prefix_plan)
from mindie_sim.hybrid_kv_state_blocks import _pd_state
from mindie_sim.hybrid_kv_state_mcp import hybrid_kv_state_mcp_envelope
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, _machine_limits, model_is_sparse_sfa,
    model_mlapo_eligible, projection_quant_class)
from mindie_sim.hybrid_kv_state_pool import BaselinePricing, stage_deltas
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state as E
from mindie_sim.linear_state_replay import verify_commit_cost
from mindie_sim.memory_ledger_types import (
    COMPONENT_NAMES, MemoryBreakdown, MemoryComponent)
from mindie_sim.memory_state_delta import apply_state_pool_delta
from mindie_sim.recoverssm_config import RecoverSsmConfig
from mindie_sim.schema.common import ModelingError
from mindie_sim.xpyd_transfer import component_major_pd_transfer
from mindie_sim.kv_pd_descriptors import component_major_pd_comparison


def _raw(*args, **kwargs):
    return NS(*args, **kwargs)


# ---------------------------------------------------------------- F01


def test_f01_glm_kda_token_cap_does_not_restrict_generic_standard_math():
    config = RecoverSsmConfig(
        use_replayssm=False, speculative_k=8, enable_prefix_caching=False,
        mamba_cache_mode="none", model_type="generic_linear")
    assert config.speculative_k == 8
    with pytest.raises(ModelingError, match="spec_tokens_exceed_kda_limit"):
        RecoverSsmConfig(use_replayssm=True, speculative_k=8,
                         enable_prefix_caching=False,
                         mamba_cache_mode="none")
    with pytest.raises(ModelingError, match="spec_tokens_exceed_kda_limit"):
        RecoverSsmConfig(use_replayssm=False, speculative_k=8,
                         enable_prefix_caching=False,
                         mamba_cache_mode="none",
                         model_type="glm5_next")


# ---------------------------------------------------------------- F02


def test_f02_xpyd_wrapper_propagates_the_c8_scale_plane():
    kwargs = dict(pages=1, layers=24, nope_bytes=65_536,
                  rope_bytes=16_384)
    typed = component_major_pd_comparison(scale_bytes=512, **kwargs)
    wrapper = component_major_pd_transfer(
        scale_bytes=512, kv_connector="MooncakeConnectorV2",
        pd_disaggregated=True, **kwargs)
    assert typed["physical_page_bytes"] == 82_432
    assert typed["physical_payload_bytes"] == 1_978_368
    assert wrapper["component_major"]["payload_bytes"] == 1_978_368
    assert wrapper["split"]["payload_bytes"] == 1_978_368
    assert wrapper["component_major"]["scale_bytes"] == 512
    assert wrapper["payload_conserved"] is True
    assert wrapper["selected"] == "component_major"
    for route in (wrapper["component_major"], wrapper["split"]):
        assert route["payload_bytes"] == wrapper["physical_payload_bytes"]


# ---------------------------------------------------------------- F03


def test_f03_pd_role_string_and_boolean_are_normalized_once():
    case = _case()
    first = E(case, _glm(
        use_replayssm=False, kv_connector="MooncakeConnectorV2",
        pd_disaggregated=True, is_kv_consumer=True))
    second = E(case, _glm(
        use_replayssm=False, kv_connector="MooncakeConnectorV2",
        pd_disaggregated=True, is_kv_consumer=False,
        kv_role="kv_consumer"))
    assert first["model"]["decode_instance"] is True
    assert second["model"]["decode_instance"] is True
    assert first["model"]["mlapo_eligible"] == second["model"][
        "mlapo_eligible"]
    options = HybridKvStateOptions(
        pd_disaggregated=True, kv_role="kv_consumer",
        kv_connector="MooncakeConnectorV2")
    assert _pd_state(options) == (True, True, False)


def test_f03_sparse_predicate_uses_attribute_presence():
    values = []
    for raw in (_raw(index_topk=0),
                _raw(index_topk=8, index_kpool=1),
                _raw(index_topk=8, compress_ratios=[0])):
        catalog = NS(hf_text_config=raw, hf_config=NS())
        values.append(model_is_sparse_sfa(catalog))
    assert values == [True, False, False]
    assert model_is_sparse_sfa(_case("deepseek-v3.2").model) is True
    assert model_is_sparse_sfa(_case().model) is False


def test_f03_dynamic_int8_is_distinct_and_rejected_on_a5():
    facts = dict(_case().model.catalog_facts)
    facts["quantization"] = {"name": "w8a8_dynamic_int8",
                             "weight_format": "int8"}
    model = dc.replace(_case().model, catalog_facts=tuple(facts.items()))
    assert projection_quant_class(model) == "w8a8_dynamic_int8"
    assert model_mlapo_eligible(model, hardware_arch="A3",
                                decode_instance=True) is True
    assert model_mlapo_eligible(model, hardware_arch="A5",
                                decode_instance=True) is False
    facts["quantization"] = {"name": "w8a8_int8", "weight_format": "int8"}
    static = dc.replace(_case().model, catalog_facts=tuple(facts.items()))
    assert projection_quant_class(static) == "w8a8_int8"
    assert model_mlapo_eligible(static, hardware_arch="A5",
                                decode_instance=True) is True


# ---------------------------------------------------------------- F04


def test_f04_global_domain_includes_the_compressor_tail():
    report = E(_case(tp=8), _glm(batch=8, sequence_tokens=2048))
    pool = report["layout"]["mixed_pool"]
    assert pool["global_recurrent_group_sizes"] == [12, 11, 11]
    assert pool["attention_blocks"] == 128
    assert pool["recurrent_blocks"] == 24
    assert pool["tail_blocks"] == 8
    assert pool["blocks"] == 160
    assert pool["main_class_bytes"] == 160 * 12 * 608_256
    assert pool["physical_pool_bytes"] == 160 * (12 * 608_256 + 11 * 131_072)
    assert pool["small_page_bytes"] > 0 and pool["small_slot_count"] == 11


def test_f04_pp2_standard_projects_local_slots_on_the_global_domain():
    case = dc.replace(_case(pp=2), runtime=dc.replace(
        _case(pp=2).runtime, pp_layer_partition=(1, 44)))
    report = E(case, _glm(batch=8, sequence_tokens=2048,
                          use_replayssm=False))
    assert report["errors"] == []
    pool = report["layout"]["mixed_pool"]
    assert pool["projected_group_sizes_per_stage"] == [[1, 0, 0],
                                                       [11, 11, 11]]
    assert [stage["blocks"] for stage in pool["stages"]] == [40, 248]
    assert [stage["main_slot_count"] for stage in pool["stages"]] == [1, 11]
    assert pool["global_block_domain"] == 256
    assert [stage["physical_pool_bytes"] for stage in pool["stages"]] == [
        145_227_776, 1_597_505_536 + 256 * 11 * 131_072]
    assert pool["stages"][0]["physical_pool_bytes"] >= (
        256 * 1 * pool["global_main_page_bytes"])


def test_f04_recover_pp2_is_pp1_only_but_standard_pp2_is_valid():
    case = dc.replace(_case(pp=2), runtime=dc.replace(
        _case(pp=2).runtime, pp_layer_partition=(1, 44)))
    recover = E(case, _glm(batch=8, sequence_tokens=2048))
    assert "recoverssm_requires_pp1" in recover["errors"]
    assert recover["linear_state"]["stage_state_bytes"] == [0.0, 0.0]
    standard = E(case, _glm(batch=8, sequence_tokens=2048,
                            use_replayssm=False))
    assert standard["errors"] == []
    assert standard["layout"]["mixed_pool"]["blocks"] == 256


# ---------------------------------------------------------------- F05


def _ledger():
    components = tuple(
        MemoryComponent(name, 100 if index == 0 else 0,
                        (100, 110) if index == 0 else (0, 0))
        for index, name in enumerate(COMPONENT_NAMES))
    return MemoryBreakdown(
        components=components, stage_peak_bytes=(100, 110),
        peak_hbm_bytes=100, device_hbm_bytes=200, budget_bytes=200,
        headroom_bytes=100, oom_feasible=True,
        dominant_assassin=COMPONENT_NAMES[0], worst_stage=0,
        naive_weights_kv_bytes=100, hidden_assassin_bytes=0,
        sequence_tokens=128, sequences=1, parallel=(("pp", 2),),
        per_rank_bytes=((0, 0, 0, 100, 200), (0, 1, 1, 110, 300)))


def test_f05_limiting_rank_change_updates_budget_and_stays_stable():
    first = apply_state_pool_delta(_ledger(), per_stage_bytes=(0, 200))
    assert first.worst_stage == 1
    assert first.budget_bytes == 300
    assert first.device_hbm_bytes == 300
    second = apply_state_pool_delta(first, per_stage_bytes=(0, 1))
    assert second.worst_stage == 1
    assert second.budget_bytes == 300
    assert second.device_hbm_bytes == 300


# ---------------------------------------------------------------- F06


def test_f06_inclusive_final_aux_boundary_is_a_valid_capture_point():
    case = _case("deepseek-v4.1-flash")
    report = E(case, _ds41(dspark_aux_layer_ids=(case.model.layers,)))
    assert report["errors"] == []
    assert report["status"] != "rejected"
    assert report["memory"]["aux_scratch_per_stage"] == [81_920.0]


def test_f06_rejected_aux_boundary_allocates_no_scratch():
    case = _case("deepseek-v4.1-flash")
    report = E(case, _ds41(dspark_aux_layer_ids=(case.model.layers + 1,)))
    assert any("dspark_aux_layer_out_of_range" in code
               for code in report["errors"])
    assert report["memory"]["aux_scratch_per_stage"] == [0.0]


# ---------------------------------------------------------------- F07


def test_f07_noncausal_split_is_analytical_and_strict_rejects():
    specs = _base_specs()
    specs = [dc.replace(item, kind="mla_attention")
             if item.is_attention else item for item in specs]
    specs[2] = dc.replace(specs[2], noncausal_query_block=True)
    grouping = kimi_k3_dspark_grouping(specs)
    assert grouping.ok is False
    assert grouping.source_compatible is False
    assert grouping.mode == "analytical_noncausal_split"
    assert grouping.fallback_reason == (
        "noncausal_split_is_not_source_uniform_grouping")
    assert grouping.scheduler_block_tables == 0
    assert grouping.analytical_block_tables == 2
    assert grouping.group_sizes == (2, 1, 2, 1)
    with pytest.raises(ModelingError, match="noncausal_split_is_not_source"):
        kimi_k3_dspark_grouping(specs, strict=True)


# ---------------------------------------------------------------- F08


def test_f08_gathered_heads_follow_one_cp_contract():
    rows = {}
    for pcp, dcp in ((1, 2), (2, 2), (2, 16)):
        cost = DcpDraftBlockCost(**_dcp(pcp=pcp, dcp=dcp))
        rows[(pcp, dcp)] = (cost.gathered_heads, cost.merge_payload_bytes,
                            cost.attention_flops)
        if dcp == pcp:
            assert cost.gather_peers == 0 and cost.gather_bytes == 0
    assert rows[(1, 2)][0] == 16
    assert rows[(2, 2)][0] == 8
    assert rows[(2, 16)][0] == 64
    assert rows[(1, 2)][1] == 25 * 16 * (512 * 2 + 4)
    assert rows[(1, 2)][2] == 116_633_600
    assert rows[(2, 2)][2] == 58_316_800
    assert rows[(2, 16)][2] == 62_668_800


def test_f08_merge_phases_use_the_resulting_contribution_tensor():
    cost = DcpDraftBlockCost(**_dcp(pcp=2, dcp=16))
    packed = 25 * 64 * (512 + 4) * 2
    assert cost.combine_tp_bytes == pytest.approx(7 / 8 * packed)
    assert cost.combine_pcp_bytes == pytest.approx(
        packed)
    assert cost.combine_bytes == pytest.approx(
        cost.combine_tp_bytes + cost.combine_pcp_bytes)
    assert cost.combine_bytes != pytest.approx(
        15 / 16 * cost.merge_payload_bytes)


def test_f08_swa_branch_has_no_extra_pairs():
    windowed = DcpDraftBlockCost(**_dcp())
    assert windowed.swa_window > 0
    assert windowed.extra_pairs == 0
    unwindowed = DcpDraftBlockCost(**_dcp(swa_window=0))
    assert unwindowed.extra_pairs == 5 * 4 // 2


# ---------------------------------------------------------------- F09


def test_f09_plan_rejects_every_forged_registration_scalar():
    original = _geometry()
    mutations = (
        ({"num_blocks": 2}, 256, 2, "num_blocks"),
        ({"page_bytes": 1}, 128, 1, "page_bytes"),
        ({"require_hash_equals_block": False, "hash_block_size": 64},
         128, 2, "hash_block_size"))
    for changes, tokens, hashes, field in mutations:
        geometry = dict(original)
        geometry.update(changes)
        with pytest.raises(ModelingError, match=f"geometry_mutated:{field}"):
            dspark_prefix_plan(
                keys=_keys(), page_geometry=geometry, aux_layer_ids=(1,),
                target_hidden_size=8, target_dtype="bf16", tokens=tokens,
                prompt_tokens=tokens, block_hash_count=hashes)


def test_f09_registration_evidence_is_carried_and_bounded():
    plan = dspark_prefix_plan(
        keys=_keys(), page_geometry=_geometry(), aux_layer_ids=(1,),
        target_hidden_size=8, target_dtype="bf16", tokens=128,
        prompt_tokens=128, block_hash_count=1)
    assert isinstance(plan, DSparkPrefixPlan)
    payload = plan.to_dict()
    registration = payload["registration"]
    assert registration["num_blocks"] == 1
    assert registration["page_bytes"] == 128
    assert registration["require_hash_equals_block"] is True
    assert registration["coverage"]["groups"] == [0]
    assert registration["coverage"]["pages"][0]["region_size"] == 128
    with pytest.raises(ModelingError, match="block_id_out_of_range"):
        plan.validate_block_table({0: [1]}, [0])


# ---------------------------------------------------------------- F11


def test_f11_boundary_branch_is_exclusive_and_timing_is_consistent():
    align = _config(mamba_cache_mode="align", enable_prefix_caching=True)
    kwargs = dict(batch=2, hbm_bytes_per_s=1e12, kernel_launch_ms=0.0)
    crossing = verify_commit_cost(align, [_layer()], sequence_tokens=126,
                                  accepted_tokens=5, **kwargs)
    inside = verify_commit_cost(align, [_layer()], sequence_tokens=120,
                                accepted_tokens=5, **kwargs)
    page = 542_720
    assert crossing["boundary_read_bytes"] == 2 * 18_432
    assert crossing["boundary_write_bytes"] == 2 * page
    assert crossing["boundary_extra_bytes"] == 1_122_304
    assert crossing["write_bytes"] - inside["write_bytes"] == 2 * page
    assert crossing["read_bytes"] - inside["read_bytes"] == 2 * 18_432 + 2 * 5 * 8_192
    assert crossing["stage2_matrix_flops"] - inside["stage2_matrix_flops"] == 2 * 2 * 131_072 * 6
    assert crossing["total_ms"] == pytest.approx(
        crossing["hbm_bytes"] / 1e12 * 1000.0)
    assert crossing["stage2_timing"]["hbm_ms"] == pytest.approx(
        (crossing["stage2_read_bytes"] + crossing["stage2_write_bytes"])
        / 1e12 * 1000.0)
    assert inside["boundary_read_bytes"] == 0.0
    assert inside["boundary_write_bytes"] == 0.0


def test_f11_record_reads_follow_the_actual_value_tiling():
    cost = verify_commit_cost(
        _config(mamba_cache_mode="align", enable_prefix_caching=True),
        [_layer()], batch=1, sequence_tokens=128,
        hbm_bytes_per_s=1e12)
    assert cost["commit_block_v"] == 64
    # correction 8 x 128 x 4 = 4,096; key/gate read twice = 8,192.
    assert cost["record_read_bytes_per_token"] == 12_288
    assert cost["record_read_bytes_per_token"] > 8_192


def test_f11_public_a5_geometry_matches_the_a5_helper():
    a5 = _case(sku="Ascend950DT_9581", topology="a5_pod")
    report = E(a5, _glm(batch=1))
    step = report["linear_state"]["step_cost"]
    assert report["model"]["hardware_arch"] == "A5"
    assert step["commit_block_v"] == 128
    assert step["launch_geometry"]["commit_programs"] == 272
    assert step["launch_geometry"]["program_count"] == 307
    assert step["launch_geometry"]["conv_tile"] == 4096
    pr = E(_case(sku="Ascend950PR_9579", topology="a5_pod"), _glm(batch=1))
    assert pr["linear_state"]["step_cost"]["commit_block_v"] == 128


def test_f11_registered_conv_stride_reaches_the_public_report():
    facts = dict(_case().model.catalog_facts)
    facts["conv_dim_stride"] = 2
    case = dc.replace(_case(), model=dc.replace(
        _case().model, catalog_facts=tuple(facts.items())))
    report = E(case, _glm(batch=1, sequence_tokens=128))
    step = report["linear_state"]["step_cost"]
    assert report["linear_state"]["commit_hardware"]["conv_dim_stride"] == 2
    assert step["launch_geometry"]["conv_tile"] == 256
    assert step["launch_geometry"]["conv_programs"] == 12 * 34


# ---------------------------------------------------------------- F14


def test_f14_bool_and_fraction_baselines_reject_before_arithmetic():
    with pytest.raises(ModelingError, match="stages_invalid"):
        BaselinePricing(True, (True,), (True,), True, True, True)
    with pytest.raises(ModelingError, match="replaced_per_stage_invalid"):
        BaselinePricing(1, (__import__("fractions").Fraction(1, 2),), (0,),
                        0, 0, 0)


def test_f14_nan_delta_and_nondivisible_heads_reject():
    with pytest.raises(ModelingError, match="modeled_bytes_invalid"):
        stage_deltas(modeled=(float("nan"),), replaced=(0,),
                     conservative=True)
    with pytest.raises(ModelingError, match="not_divisible_by_tp"):
        DcpDraftBlockCost(**_dcp(num_q_heads=10, tp=3))


def test_f14_zero_fp32_peak_is_not_an_omitted_compute_term():
    case = _case()
    dies = []
    for device in case.machine.dies:
        peaks = dict(device.compute_peaks)
        peaks["fp32"] = 0
        dies.append(dc.replace(device, compute_peaks=tuple(peaks.items())))
    broken = dc.replace(case, machine=dc.replace(case.machine,
                                                dies=tuple(dies)))
    with pytest.raises(ModelingError, match="fp32_flops_invalid"):
        _machine_limits(broken)


def test_f14_valid_staging_reserve_is_supported_and_malformed_is_typed():
    case = _case()
    runtime = dc.replace(case.runtime,
                         memory={"kv_transfer_staging_bytes": 40_000_000})
    report = E(dc.replace(case, runtime=runtime), _glm(
        use_replayssm=False, enable_kv_nz=True, pd_disaggregated=True,
        is_kv_consumer=True, kv_connector="MooncakeConnectorV2"))
    assert report["errors"] == []
    assert report["memory"]["nz_staging_per_stage"] == [40_000_000.0]
    for value in (True, __import__("fractions").Fraction(1, 2), "1024"):
        broken = dc.replace(case, runtime=dc.replace(
            case.runtime, memory={"kv_transfer_staging_bytes": value}))
        with pytest.raises(ModelingError,
                           match="kv_transfer_staging_bytes_invalid"):
            E(broken, _glm(use_replayssm=False))


# ---------------------------------------------------------------- F15


def test_f15_total_error_count_precedes_the_budgeted_prefix():
    tiktoken = pytest.importorskip("tiktoken")
    encodings = [tiktoken.get_encoding(name)
                 for name in ("cl100k_base", "o200k_base")]
    report = {"schema": "mindie-sim.hybrid-kv-state/1",
              "status": "rejected",
              "errors": [f"e{index}" for index in range(100)]}
    envelope = hybrid_kv_state_mcp_envelope(report)
    text = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    assert envelope["total_error_count"] == 100
    assert envelope["codes_truncated"] is True
    assert len(text.encode()) <= 250
    assert all(len(encoding.encode(text)) <= 250 for encoding in encodings)
    assert all(32 <= ord(char) <= 126 for char in text)


# ---------------------------------------------------------------- F16


def test_f16_documented_pool_and_boundary_numbers_are_current():
    glm = E(_case(tp=8), _glm(batch=8, sequence_tokens=2048))
    pool = glm["layout"]["mixed_pool"]
    assert pool["blocks"] == 160
    assert pool["physical_pool_bytes"] == 1_398_538_240
    crossing = verify_commit_cost(
        _config(mamba_cache_mode="align", enable_prefix_caching=True),
        [_layer()], batch=2, sequence_tokens=126, accepted_tokens=5,
        hbm_bytes_per_s=1e12)
    assert crossing["hbm_bytes"] == 4_685_824
    assert crossing["total_ms"] == pytest.approx(0.004685824)
    assert crossing["boundary_read_bytes"] == 36_864


def fix11_domain_witnesses():
    from mindie_sim.catalog.types import DraftSpec
    from mindie_sim.dspark_kv_physics import DsparkSwaFootprint, KimiK3DsparkGrouping
    from mindie_sim.kv_nz_read_model import Mte2ReadModel
    from mindie_sim.memory_breakdown import build_memory_ledger, memory_breakdown
    from mindie_sim.nr_s7_validate import _record_kinds, _RECORD_MODULES, _validate_nested
    from mindie_sim.parallel_decode_cost import DecodeCostModel
    from mindie_sim.recoverssm_config import GlmKdaExecution, validate_recoverssm
    from mindie_sim.schema.model import SloSpec, WorkloadSpec
    from test_nr_s7_linear_kv_dspark_r1 import FIX11_FORMS, fix11_subtype
    case = _case()
    decode, memory = DecodeCostModel.from_case(case), memory_breakdown(case)
    records = [
        (case, "workload", WorkloadSpec._pure_model_sentinel()),
        (case.model, "num_heads", 0), (case.model.mla, "kv_lora_rank", 0),
        (case.model.layer_plans[0], "head_dim", -1),
        (DraftSpec("mtp", 1, 128, 256, 1, 1, 128, 128, 128), "layers", 0),
        (case.runtime, "max_num_seqs", 0), (case.runtime.parallel, "tp", 0),
        (case.machine, "dies", ()), (case.machine.dies[0], "memory_bytes", 0),
        (case.workload.requests[0], "output_tokens", 0), (case.workload, "requests", ()),
        (SloSpec(ttft_ms=1.0), "ttft_ms", 0), (decode, "hidden", 0),
        (decode.step(1), "batch_tokens", 0),
        (GlmKdaExecution("kda", "bf16", 128, 128, 4, 4, "none", 128, -5.0, 128, ()), "key_dim", 0),
        (validate_recoverssm(_config()), "expected_cache_mode", "invalid"),
        (KimiK3DsparkGrouping((), None, 0, 0, 0), "mamba_layers", -1),
        (DsparkSwaFootprint(128, 128, 0, 128, 1, 1.0), "row_bytes_per_token", 0),
        (Mte2ReadModel(*([1] * 12)), "head_bytes", 0),
        (build_memory_ledger(case), "pp", 0), (memory, "stage_peak_bytes", ()),
        (memory.components[0], "per_stage_bytes", ("invalid",)),
    ]
    covered, rejections, valid = set(), 0, 0
    for record, name, bad in records:
        for form in FIX11_FORMS:
            for depth in (1, 3):
                control = fix11_subtype(record, name, getattr(record, name), form, depth, forge=True)
                _validate_nested(control, "valid")
                covered.update(_record_kinds(type(control)))
                valid += 1
                for value in (bad, object()):
                    broken = fix11_subtype(record, name, value, form, depth, forge=True)
                    with pytest.raises(ModelingError):
                        _validate_nested(broken, "broken")
                    rejections += 1
    expected = {name for names in _RECORD_MODULES.values() for name in names}
    assert covered == expected and len(covered) == 22
    # Base metadata/diagnostic exceptions remain valid after redeclaration.
    extras = ((case.model, "metadata", {"nested": [1, "ok"]}),
              (case.model, "catalog_facts", {"nested": [1, "ok"]}),
              (case.runtime, "metadata", {"nested": [1, "ok"]}),
              (memory, "detail", (("nested", [1, "ok"]),)),
              (memory.components[0], "detail", (("nested", [1, "ok"]),)))
    for record, name, value in extras:
        _validate_nested(fix11_subtype(record, name, value, "classvar", 3, forge=True), "metadata")
    return {"families": sorted(covered), "rejections": rejections,
            "valid_subtypes": valid, "metadata_controls": len(extras)}


def test_fix11_all_registered_base_fields_survive_redeclaration():
    fix11_domain_witnesses()


def test_fix11_multiple_registered_bases_and_missing_optional_property():
    from typing import Any, ClassVar
    from mindie_sim.nr_s7_validate import _validate_nested
    from mindie_sim.schema.model import SloSpec
    from mindie_sim.schema.parallel import ParallelSpec
    Combined = dc.make_dataclass("Combined", [("tp", ClassVar[int], 0),
        ("ttft_ms", Any, 1.0)], bases=(SloSpec, ParallelSpec), frozen=True)
    with pytest.raises(ModelingError):
        _validate_nested(Combined(), "both")
    # A valid first contract must not hide an invalid field in the other one.
    Valid = dc.make_dataclass("Valid", [("tp", ClassVar[int], 1),
        ("ttft_ms", Any, 0.0)], bases=(SloSpec, ParallelSpec), frozen=True)
    with pytest.raises(ModelingError):
        _validate_nested(Valid(), "both")
    _validate_nested(dc.replace(Valid(), ttft_ms=1.0), "valid")
    def absent(self):
        raise AttributeError("missing optional base field")
    Missing = dc.make_dataclass("Missing", [("ttft_ms", ClassVar[object], property(absent))],
                               bases=(SloSpec,), frozen=True)
    with pytest.raises(ModelingError):
        _validate_nested(Missing(), "missing")


def fix11_source_evidence(artifact_dir):
    load = lambda name: json.loads((artifact_dir / name).read_text())
    redeclared = load("r11_redeclared_fields_results.json")
    rows = redeclared["rows"]
    assert len(rows) == 300 and len(redeclared["base_controls"]) == 150
    assert all(r["outcome"] == "ModelingError" and not r["body_entries"]
               for r in rows + redeclared["base_controls"] + redeclared["nested"])
    fresh = load("r11_fresh_subtypes_results.json")
    assert fresh["total_rejections"] == 12032 and not fresh["failures"] and not fresh["consumer_body_entries"]
    assert len(fresh["registered_domain_families"]) == 22
    return {"redeclared": {"consumer_rejections": len(rows),
        "base_controls": len(redeclared["base_controls"]), "nested_rejections": len(redeclared["nested"]),
        "body_entries": 0, "invalid_accepts": 0, "raw_type_errors": 0}, "inherited": fresh}


def fix11_report(data):
    evidence = data["fix11"]
    return f"""**R11-P2-01 — Registered base field contracts**

The validator builds required fields and resolved types from every registered base
class in the MRO. Leaf ClassVar, Any, object, widened Union, unannotated overrides
and properties cannot remove or weaken those base checks or nested traversal.
SLO and scalar domain loops use the registered base fields too. Case.workload is
fetched with getattr and a missing-field sentinel, and traversed first. Additional
subclass fields are checked; documented metadata/diagnostics and hashable str IDs
remain supported. Pure-model parsing retains its separate empty sentinel factory.

Permanent matrix: `{evidence['permanent']}`.
All invalid consumer calls require ModelingError and profile the original bodies;
zero accepts, raw TypeErrors or body entries pass. The matrix includes single and
three-level inheritance, six declaration forms, all ten consumer forms and all
three capture modes. Additional tests cover intermediate redeclarations, multiple
registered bases, missing optional properties, workload-first ordering and typed
or recursive subclass additions.

The adapted R11 defect probe now requires **{evidence['redeclared']['consumer_rejections']}/300**
pre-body rejections, **{evidence['redeclared']['base_controls']}/150** same-value base controls
and **{evidence['redeclared']['nested_rejections']}/10** direct rejections.
The unchanged inherited-field matrix passes **{evidence['inherited']['total_rejections']}/12,032**,
including 10,560 invalid consumers with zero body entries, 120 valid consumers and
six pure-model parses. These are executed finite matrices, not a proof over
arbitrary Python code or stateful descriptors that change values between reads.
"""
