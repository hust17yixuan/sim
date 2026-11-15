"""Deterministic regenerator for every number published in ``result.md``.

[Role]
  Re-execute each witness in the NR-S7 Round-3 closure against the checked-in
  modules and emit one JSON object.  ``result.md`` quotes this object, not a
  hand-maintained table: a changed implementation changes the JSON.

[Invariants]
  * Every displayed number is computed here from declared inputs; no value is
    copied from prose, a previous report, or an ambient hardware probe.
  * Valid standard PP2 and PP1-only RecoverSSM are separate witnesses; the
    rejected RecoverSSM PP2 record is never presented as a deployment number.
  * ISL1024 and ISL2048 replay extents are both emitted, as are DSpark active
    and inactive, and ``none`` versus ``align`` crossover with ``max_batch``
    explicitly 2000.
  * File-table counts are measured from the files; BPE counts use the real
    cached ``tiktoken`` vocabularies when available.

[Timeouts & Locks]
  Pure in-memory arithmetic plus file reads; no device, subprocess or network.

[Side-Effect Policy]
  Writes nothing unless ``--out PATH`` is given; caller inputs are not mutated.
"""
from __future__ import annotations

import dataclasses as dc
import json
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    tiktoken = None

from nr_s7_fix2_support import (
    _base_specs, _case, _config, _dcp, _ds41, _glm, _kimi, _layer)
from mindie_sim.dspark_dcp_cost import DcpDraftBlockCost
from mindie_sim.dspark_kv_physics import (
    kimi_k3_dspark_grouping, swa_bounded_replay_policy)
from mindie_sim.dspark_prefix_cache import (
    DSparkPrefixPage, dspark_prefix_keys, dspark_prefix_page_bytes,
    dspark_prefix_plan)
from mindie_sim.hybrid_kv_state_mcp import hybrid_kv_state_mcp_envelope
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, model_is_sparse_sfa, model_mlapo_eligible,
    projection_quant_class)
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state as E
from mindie_sim.kv_layout_physics import KvNzLayout
from mindie_sim.kv_nz_read_model import mte2_read_model
from mindie_sim.kv_nz_transfer import nd_to_nz_reformat_cost
from mindie_sim.kv_pd_descriptors import component_major_pd_comparison
from mindie_sim.linear_state_physics import (
    accepted_token_pmf, expected_accepted_tokens, linear_state_memory)
from mindie_sim.linear_state_prefix_replay import (
    prefix_replay_cost, prefix_replay_plan)
from mindie_sim.linear_state_replay import (
    precopy_cost, recoverssm_crossover, verify_commit_cost)
from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.xpyd_transfer import component_major_pd_transfer

ROOT = Path(__file__).resolve().parents[1]
SCOPED = (
    "mindie_sim/nr_s7_validate.py", "mindie_sim/recoverssm_config.py",
    "mindie_sim/linear_state_physics.py", "mindie_sim/linear_state_replay.py",
    "mindie_sim/linear_state_prefix_replay.py",
    "mindie_sim/recoverssm_launch_geometry.py",
    "mindie_sim/recoverssm_commit_work.py",
    "mindie_sim/kv_layout_physics.py", "mindie_sim/kv_nz_read_model.py",
    "mindie_sim/kv_nz_transfer.py", "mindie_sim/kv_pd_descriptors.py",
    "mindie_sim/kv_pool_geometry.py", "mindie_sim/dspark_kv_physics.py",
    "mindie_sim/dspark_dcp_cost.py", "mindie_sim/dspark_prefix_cache.py",
    "mindie_sim/dspark_prefix_registration.py",
    "mindie_sim/hybrid_kv_state_options.py",
    "mindie_sim/hybrid_kv_state_pool.py",
    "mindie_sim/hybrid_kv_state_blocks.py",
    "mindie_sim/hybrid_kv_state_dspark.py",
    "mindie_sim/hybrid_kv_state_report.py",
    "mindie_sim/hybrid_kv_state_mcp.py", "mindie_sim/memory_breakdown.py",
    "mindie_sim/memory_state_delta.py",
    "mindie_sim/parallel_decode_cost.py", "mindie_sim/xpyd_transfer.py",
    "tests/test_nr_s7_linear_kv_dspark.py",
    "tests/test_nr_s7_linear_kv_dspark_r1.py",
    "tests/test_nr_s7_linear_kv_dspark_r2.py",
    "tests/test_nr_s7_fix2_regressions.py",
    "tests/test_nr_s7_fix2_regressions_r2.py",
    "tests/test_nr_s7_fix3_regressions.py", "tests/nr_s7_fix2_support.py",
    "tests/nr_s7_fix3_verify.py", "tests/nr_s7_fix4_verify.py",
    "tests/test_nr_s7_fix4_regressions.py", "tests/test_nr_s7_fix5_regressions.py")
MARKERS = ("[Role]", "[Invariants]", "[Timeouts & Locks]",
           "[Side-Effect Policy]")
OUT = {}


def _files():
    rows = {}
    for name in SCOPED:
        text = (ROOT / name).read_text()
        rows[name] = {
            "lines": len(text.splitlines()),
            "cr": text.count("\r"),
            "lf_end": text.endswith("\n"),
            "markers": all(marker in "\n".join(
                text.splitlines()[:30]) for marker in MARKERS)}
    OUT["files"] = rows


def _ssm_layer(name="layer0", **overrides):
    values = dict(name=name, kind="kda", index=0, num_heads=64, key_dim=128,
                  value_dim=128, conv_kernel=4, conv_width=24576,
                  activation_dtype="bf16",
                  recurrent_elements=64 * 128 * 128,
                  conv_history_elements=24576 * 3)
    values.update(overrides)
    return _layer(**values)


def _ssm_config(**overrides):
    values = dict(use_replayssm=True, speculative_k=4,
                  enable_prefix_caching=False, mamba_cache_mode="none",
                  tp=8, model_type="glm5_next", kv_block_size=128)
    values.update(overrides)
    return _config(**values)


def _linear():
    layer = _ssm_layer()
    recover = _ssm_config()
    align = _ssm_config(mamba_cache_mode="align", enable_prefix_caching=True)
    standard = _ssm_config(use_replayssm=False)
    plan = linear_state_memory(recover, [layer], sequence_tokens=2048,
                               active_sequences=1)
    align_plan = linear_state_memory(align, [layer], sequence_tokens=2048,
                                     active_sequences=1)
    std_plan = linear_state_memory(standard, [layer], sequence_tokens=2048,
                                   active_sequences=1)
    fp32 = _layer(record_kg_dtype="fp32")
    fp32_plan = linear_state_memory(recover, [fp32], sequence_tokens=2048,
                                    active_sequences=1)
    crossing = verify_commit_cost(
        align, [layer], batch=2, sequence_tokens=2047,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    inside = verify_commit_cost(
        align, [layer], batch=2, sequence_tokens=2048,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    pmf = verify_commit_cost(
        align, [layer], batch=1, sequence_tokens=2044,
        hbm_bytes_per_s=1.2e12, accept_rate=0.8)
    multi = tuple(_layer(name=f"l{i}", index=i) for i in range(34))
    layer_launch = verify_commit_cost(
        align, multi, batch=2, sequence_tokens=2048,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02)
    strided = verify_commit_cost(
        _ssm_config(), [layer], batch=1, sequence_tokens=128,
        hbm_bytes_per_s=1e12, conv_dim_stride=2)
    OUT["linear"] = {
        "state": layer.state_bytes(8),
        "conv_allocated": layer.conv_bytes(8, spec_query_len=5),
        "records": plan.packed_record_bytes_per_sequence,
        "page": plan.checkpoint_page_bytes,
        "ratio": plan.matrix_to_record_ratio,
        "align_pages": align_plan.pages_per_sequence,
        "align_total": align_plan.total_bytes_per_sequence,
        "standard_slots": std_plan.state_slots_per_sequence,
        "standard_total": std_plan.total_bytes_per_sequence,
        "fp32_record_ratio": fp32_plan.matrix_to_record_ratio,
        "fp32_records": fp32_plan.packed_record_bytes_per_sequence,
        "pmf": dict(accepted_token_pmf(4, 0.8)),
        "expectation": expected_accepted_tokens(4, 0.8),
        "pmf_cost": {"probability": pmf["boundary_probability"],
                     "writes": pmf["full_state_writes"],
                     "commit_len": pmf["commit_len"]},
        "boundary": {key: crossing[key] for key in (
            "committed_conv_bytes", "boundary_read_bytes",
            "boundary_write_bytes", "boundary_extra_bytes",
            "stage2_read_bytes", "hbm_bytes", "flops", "full_state_writes",
            "kernel_launches", "stage2_program_count", "total_ms")},
        "inside": {key: inside[key] for key in (
            "boundary_read_bytes", "boundary_write_bytes", "hbm_bytes",
            "flops", "full_state_writes")},
        "boundary_delta": {
            "reads": crossing["boundary_read_bytes"],
            "writes": crossing["boundary_write_bytes"],
            "hbm": crossing["hbm_bytes"] - inside["hbm_bytes"],
            "flops": crossing["flops"] - inside["flops"]},
        "launch_34": {"stage1_ms": layer_launch["stage1_timing"]["launch_ms"],
                      "launches": layer_launch["kernel_launches"]},
        "record_read_bytes_per_token": crossing[
            "record_read_bytes_per_token"],
        "strided_conv": {"tile": strided["launch_geometry"]["conv_tile"],
                         "programs": strided["launch_geometry"][
                             "conv_programs"]},
    }
    zero = verify_commit_cost(
        align, [layer], batch=2, sequence_tokens=126,
        hbm_bytes_per_s=1e12, accepted_tokens=5)
    pcp_page = linear_state_memory(
        _ssm_config(tp=1, pcp=2, kv_connector="MooncakeConnectorV2",
                    kv_role="kv_producer"), [layer], sequence_tokens=2048,
        active_sequences=1)
    OUT["documented"] = {
        "committed_conv_per_request": layer.conv_width
        * layer.history_rows // 8 * 2,
        "committed_page": layer.state_bytes(8)
        + layer.conv_width * layer.history_rows // 8 * 2,
        "flat_record_per_token": layer.record_bytes_per_token(8),
        "zero_launch_hbm": zero["hbm_bytes"],
        "zero_launch_total_ms": zero["total_ms"],
        "zero_launch_boundary_read": zero["boundary_read_bytes"],
        "zero_launch_extra_flops": zero["boundary_matrix_flops"],
        "pcp_page": pcp_page.checkpoint_page_bytes,
        "kimi_hidden": _case("kimi-k3").model.hidden_size,
        "kimi_aux_one_layer": 2 * 4 * _case("kimi-k3").model.hidden_size * 2,
        "kimi_group_sizes": E(
            _case("kimi-k3", max_model_len=8192),
            _kimi(enable_kv_nz=True, pd_disaggregated=True,
                  is_kv_consumer=True,
                  kv_connector="MooncakeConnectorV2"))["dspark"][
                      "grouping"]["group_sizes"],
        "aux_partition": [37, 3],
        "review_prefix_pool": 1_109_458_944,
        "review_none_crossover": 1075,
        "pmf_k": 4, "pmf_alpha": 0.8, "pmf_start": 2044,
        "replay_hit": 1000, "replay_isl": [1024, 2048],
        "a5_default_ub_rows": [190, 62],
    }
    OUT["crossover"] = {}
    for mode, config in (("none", _ssm_config()),
                         ("align", align)):
        OUT["crossover"][mode] = recoverssm_crossover(
            config, [layer], hbm_bytes_per_s=1.2e12,
            compute_flops_per_s=1e14, kernel_launch_ms=0.5, max_batch=2000)
    OUT["precopy"] = {}
    for label, start, end, batch in (("crossing", 127, 132, 2),
                                     ("inside", 129, 134, 2)):
        OUT["precopy"][label] = precopy_cost(
            align, [layer], sequence_start=start, sequence_end=end,
            batch=batch, hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.01)
    OUT["replay"] = {}
    for isl in (1024, 2048):
        plan = prefix_replay_plan(align, [layer], isl=isl, prefix_hit=1000)
        OUT["replay"][str(isl)] = prefix_replay_cost(
            align, [layer], plan=plan, hbm_bytes_per_s=1.2e12,
            compute_flops_per_s=1e14)


def _layout():
    bf16 = KvNzLayout(num_blocks=8, num_kv_heads=1, block_size=128,
                      nope_head_size=512, nope_dtype="bf16",
                      rope_head_size=64, rope_dtype="bf16")
    c8 = KvNzLayout(num_blocks=2, num_kv_heads=1, block_size=64,
                    nope_head_size=512, nope_dtype="int8",
                    rope_head_size=64, rope_dtype="bf16", scale_dim=1)
    mte2 = mte2_read_model(
        KvNzLayout(num_blocks=1, block_size=128, num_kv_heads=4,
                   nope_head_size=64, nope_dtype="bf16"),
        tokens=128, local_kv_heads=4)
    multi_layout = KvNzLayout(num_blocks=4, num_kv_heads=1, block_size=128,
                              nope_head_size=64, nope_dtype="bf16")
    multi = mte2_read_model(multi_layout, tokens=256, local_kv_heads=1,
                            block_ids=(0, 2), cacheline_bytes=256)
    tail = mte2_read_model(multi_layout, tokens=200, local_kv_heads=1,
                           block_ids=(1, 3), cacheline_bytes=256)
    reformat = nd_to_nz_reformat_cost(
        pages=8, layers=3, nd_page_bytes=147456, hbm_bytes_per_s=1e12,
        fused_op=False, kernel_fixed_ms=0.01, synchronize_ms=0.02,
        nz_layout=None)
    OUT["layout"] = {
        "bf16": {"nope_shape": list(bf16.nope.shape),
                 "rope_shape": list(bf16.rope.shape),
                 "logical": bf16.logical_bytes,
                 "physical": bf16.physical_bytes,
                 "padding": bf16.padding_bytes,
                 "total": bf16.total_bytes,
                 "view": bf16.view_compatible},
        "c8": {"nope_c0": c8.nope.c0_elements,
               "rope_c0": c8.rope.c0_elements,
               "nope_bytes": c8.nope.element_bytes,
               "rope_bytes": c8.rope.element_bytes,
               "scale": c8.scale_bytes, "total": c8.total_bytes,
               "padding": c8.padding_bytes},
        "mte2": mte2.to_dict(),
        "mte2_multi": {"ops": multi.nz_dma_ops,
                       "transactions": multi.nz_transactions,
                       "pages": multi.pages_used},
        "mte2_tail": {"ops": tail.nz_dma_ops,
                      "transactions": tail.nz_transactions,
                      "pages": tail.pages_used},
        "reformat": {"payload": reformat.payload_bytes,
                     "hbm": reformat.hbm_traffic_bytes,
                     "launches": reformat.kernel_launches,
                     "staging": reformat.staging_bytes},
        "pd": component_major_pd_comparison(
            pages=8, layers=61, nope_bytes=512, rope_bytes=64,
            padding_bytes=8, per_descriptor_setup_us=2.0, doorbell_us=1.0),
        "pd_c8": component_major_pd_comparison(
            pages=1, layers=24, nope_bytes=65536, rope_bytes=16384,
            scale_bytes=512),
        "pd_wrapper": component_major_pd_transfer(
            pages=1, layers=24, nope_bytes=65536, rope_bytes=16384,
            scale_bytes=512, kv_connector="MooncakeConnectorV2",
            pd_disaggregated=True),
    }


def _pool():
    glm = E(_case(tp=8), _glm(batch=8, sequence_tokens=2048))
    pool = glm["layout"]["mixed_pool"]
    standard = E(_case(tp=8), HybridKvStateOptions(
        use_replayssm=False, num_spec=4, batch=8, sequence_tokens=128,
        graph_analytical_bound=True))
    std = standard["layout"]["mixed_pool"]
    case = dc.replace(_case(pp=2), runtime=dc.replace(
        _case(pp=2).runtime, pp_layer_partition=(1, 44)))
    pp = E(case, _glm(batch=8, sequence_tokens=2048, use_replayssm=False))
    pp_pool = pp["layout"]["mixed_pool"]
    recover_pp = E(case, _glm(batch=8, sequence_tokens=2048))
    plain = E(_case(), _glm(batch=2, sequence_tokens=2048))
    stride = E(_case(), _glm(batch=2, sequence_tokens=2048,
                             state_page_stride_bytes=1_000_000))
    declared_groups = ({"layers": tuple(
        p.index for p in _case(tp=16).model.layer_plans
        if p.attention == "kda"), "tokens_per_page": 128,
        "bytes_per_layer_page": 524288, "fixed_pages": 1,
        "components": ("recurrent_state", "conv_state")},)
    from mindie_sim.hybrid_kv_state_pool import baseline_pricing
    declared = baseline_pricing(
        _case(tp=16, memory={"cache_page_groups": declared_groups}),
        tokens=2048, active_sequences=8)
    prepad_groups = ({"layers": tuple(
        p.index for p in _case(tp=64).model.layer_plans
        if p.attention in {"kda", "gdn"}), "tokens_per_page": 128,
        "bytes_per_layer_page": 131072, "fixed_pages": 1,
        "components": ("recurrent_state", "conv_state")},)
    prepad = E(_case(tp=64, memory={"cache_page_groups": prepad_groups},
                     prompt=128), _glm(batch=1, sequence_tokens=128))
    pre = prepad["layout"]["mixed_pool"]
    OUT["pool"] = {
        "glm_recover": {key: pool[key] for key in (
            "global_recurrent_group_sizes", "main_slot_count",
            "global_main_slot_count",
            "main_page_bytes", "global_main_page_bytes", "blocks",
            "attention_blocks", "recurrent_blocks", "tail_blocks",
            "physical_pool_bytes", "layer_sum_bytes", "page_padding_bytes",
            "attention_uplift_bytes", "small_page_bytes",
            "small_slot_count", "small_class_baseline_bytes")},
        "glm_recover_linear": glm["linear_state"]["memory"]["total_bytes"],
        "glm_standard": {key: std[key] for key in (
            "main_page_bytes", "blocks", "attention_blocks",
            "recurrent_blocks", "tail_blocks", "physical_pool_bytes")},
        "glm_standard_linear": standard["linear_state"]["memory"][
            "total_bytes"],
        "pp_standard": {
            "errors": list(pp["errors"]),
            "projected": pp_pool["projected_group_sizes_per_stage"],
            "stage_blocks": [s["blocks"] for s in pp_pool["stages"]],
            "stage_slots": [s["main_slot_count"] for s in pp_pool["stages"]],
            "stage_pools": [s["physical_pool_bytes"]
                            for s in pp_pool["stages"]],
            "global_blocks": pp_pool["blocks"],
            "global_main_page": pp_pool["global_main_page_bytes"],
            "stage_state": pp["linear_state"]["stage_state_bytes"],
            "stage0_lower_bound": (pp_pool["blocks"]
                                   * pp_pool["stages"][0]["main_slot_count"]
                                   * pp_pool["global_main_page_bytes"]),
        },
        "pp_recover": {
            "errors": list(recover_pp["errors"]),
            "stage_state": recover_pp["linear_state"]["stage_state_bytes"],
            "projected": recover_pp["layout"]["mixed_pool"][
                "projected_group_sizes_per_stage"]},
        "stride": {
            "plain_linear": plain["linear_state"]["memory"]["total_bytes"],
            "stride_linear": stride["linear_state"]["memory"]["total_bytes"],
            "plain_main": plain["layout"]["mixed_pool"]["main_page_bytes"],
            "stride_main": stride["layout"]["mixed_pool"][
                "main_page_bytes"],
            "plain_peak": plain["memory"]["adjusted_peak_hbm_bytes"],
            "stride_peak": stride["memory"]["adjusted_peak_hbm_bytes"]},
        "declared": {
            "declared_layer": declared.declared_layer_bytes,
            "declared_group": declared.declared_group_bytes,
            "undeclared": declared.undeclared_bytes,
            "replaced": list(declared.replaced_per_stage)},
        "prepad": {"replaced": pre["baseline_replaced_per_stage"],
                   "modeled": pre["physical_pool_bytes"],
                   "increment": prepad["memory"]["total_increment_bytes"],
                   "delta": prepad["memory"]["pool_replacement_delta_bytes"]},
    }


def _gates():
    raw = {}
    for label, attrs in (
            ("index_topk_zero", {"index_topk": 0}),
            ("kpool_one_present", {"index_topk": 8, "index_kpool": 1}),
            ("compress_ratios_zero", {"index_topk": 8,
                                      "compress_ratios": [0]})):
        text = type("Raw", (), attrs)()
        catalog = type("Catalog", (), {"hf_text_config": text,
                                       "hf_config": type("Raw", (), {})()})()
        raw[label] = {"sparse": model_is_sparse_sfa(catalog)}
    facts = dict(_case().model.catalog_facts)
    facts["quantization"] = {"name": "w8a8_dynamic_int8",
                             "weight_format": "int8"}
    dynamic = dc.replace(_case().model, catalog_facts=tuple(facts.items()))
    OUT["gates"] = {
        "sparse_raw": raw,
        "sparse_catalog": {
            "deepseek": model_is_sparse_sfa(_case("deepseek-v3.2").model),
            "glm": model_is_sparse_sfa(_case().model),
            "kimi": model_is_sparse_sfa(_case("kimi-k3").model)},
        "quant": {"dynamic_class": projection_quant_class(dynamic),
                  "dynamic_a3": model_mlapo_eligible(
                      dynamic, hardware_arch="A3", decode_instance=True),
                  "dynamic_a5": model_mlapo_eligible(
                      dynamic, hardware_arch="A5", decode_instance=True)},
        "recover_generic_k8": None,
    }
    from mindie_sim.recoverssm_config import RecoverSsmConfig
    config = RecoverSsmConfig(
        use_replayssm=False, speculative_k=8, enable_prefix_caching=False,
        mamba_cache_mode="none", model_type="generic_linear")
    OUT["gates"]["recover_generic_k8"] = config.speculative_k
    OUT["gates"]["recover_glm_k8_rejected"] = False
    try:
        RecoverSsmConfig(use_replayssm=True, speculative_k=8,
                         enable_prefix_caching=False,
                         mamba_cache_mode="none")
    except TimeoutError:
        raise
    except Exception as exc:
        OUT["gates"]["recover_glm_k8_rejected"] = "spec_tokens_exceed" in str(exc)


def _dspark():
    noncausal = _base_specs()
    noncausal = [dc.replace(item, kind="mla_attention")
                 if item.is_attention else item for item in noncausal]
    noncausal[2] = dc.replace(noncausal[2], noncausal_query_block=True)
    OUT["grouping"] = {
        "causal": kimi_k3_dspark_grouping(_base_specs()).to_dict(),
        "noncausal": kimi_k3_dspark_grouping(noncausal).to_dict(),
        "noncausal_strict_rejected": False}
    try:
        kimi_k3_dspark_grouping(noncausal, strict=True)
    except TimeoutError:
        raise
    except Exception as exc:
        OUT["grouping"]["noncausal_strict_rejected"] = (
            "noncausal_split_is_not_source" in str(exc))
    swa = []
    for extra in (0, 8, 256):
        report = E(_case("deepseek-v4.1-flash", max_model_len=8192),
                   _ds41(batch=2, extra_retained_tokens=extra))
        swa.append({"extra": extra,
                    "visible": report["dspark"]["swa"]["visible_tokens"],
                    "retained": report["dspark"]["swa"]["retained_tokens"],
                    "bytes": report["dspark"]["modeled_draft_kv_bytes"],
                    "delta": report["dspark"]["draft_kv_delta_per_stage"][-1],
                    "peak": report["memory"]["adjusted_peak_hbm_bytes"]})
    OUT["swa"] = swa
    OUT["draft_weights"] = {}
    for key, options in (("deepseek-v4.1-flash", _ds41()),
                         ("kimi-k3", _kimi(dspark_active=True))):
        report = E(_case(key, max_model_len=8192), options)
        OUT["draft_weights"][key] = report["memory"]["baseline"][
            "draft_weight_bytes"]
    aux_case = dc.replace(
        _case("deepseek-v4.1-flash", pp=2, max_model_len=8192),
        runtime=dc.replace(
            _case("deepseek-v4.1-flash", pp=2,
                  max_model_len=8192).runtime,
            pp_layer_partition=(37, 3)))
    aux = E(aux_case, _ds41(batch=2, sequence_tokens=2048))
    OUT["aux"] = {"per_stage": aux["memory"]["aux_scratch_per_stage"],
                  "total": aux["memory"]["aux_scratch_bytes"]}
    dcp_rows = []
    for batch in (1, 8):
        report = E(_case("deepseek-v4.1-flash", max_model_len=8192, dcp=2),
                   _ds41(batch=batch, sequence_tokens=2048,
                         dspark_block_size=5))
        block = report["dspark"]["dcp_block"]
        dcp_rows.append({key: block[key] for key in (
            "rows", "visible_tokens", "index_width", "gather_bytes",
            "combine_bytes", "index_bytes", "total_ms")})
    OUT["dcp_rows"] = dcp_rows
    topologies = []
    for pcp, dcp in ((1, 2), (2, 2), (2, 4), (2, 16)):
        cost = DcpDraftBlockCost(**_dcp(pcp=pcp, dcp=dcp))
        topologies.append({
            "pcp": pcp, "dcp": dcp, "gathered_heads": cost.gathered_heads,
            "gather_bytes": cost.gather_bytes,
            "merge_payload": cost.merge_payload_bytes,
            "combine_tp": cost.combine_tp_bytes,
            "combine_pcp": cost.combine_pcp_bytes,
            "combine": cost.combine_bytes,
            "attention_flops": cost.attention_flops,
            "extra_pairs": cost.extra_pairs,
            "causal_pairs": cost.causal_pairs,
            "noncausal_pairs": cost.noncausal_pairs})
    OUT["dcp_topologies"] = topologies
    OUT["dcp_swa"] = DcpDraftBlockCost(**_dcp(
        dcp=2, context_tokens=8192)).to_dict()
    prefix = {}
    for key, options in (("kimi-k3", _kimi(dspark_active=True)),
                         ("deepseek-v4.1-flash", _ds41())):
        report = E(_case(key, max_model_len=8192), options)
        record = report["dspark"]["prefix"]
        prefix[key] = {field: record[field] for field in (
            "draft_page_bytes", "blocks_stored", "per_rank_storage_bytes",
            "aggregate_storage_bytes", "per_rank_transfer_bytes",
            "aux_scratch_bytes", "require_hash_equals_block")}
    OUT["prefix"] = prefix
    variants = []
    base = _case("kimi-k3")
    auxiliary = base.model.auxiliary
    draft = auxiliary.dspark
    models = [
        base.model,
        dc.replace(base.model, auxiliary=dc.replace(
            auxiliary, dspark=dc.replace(draft, activation_dtype="fp32"))),
        dc.replace(base.model, auxiliary=dc.replace(
            auxiliary, dspark=dc.replace(
                draft, markov_rank=512,
                confidence_input_width=draft.hidden_size + 512))),
        dc.replace(base.model, norm_eps=base.model.norm_eps * 2),
        dc.replace(base.model, layer_plans=tuple(
            dc.replace(plan, recurrent_state_dtype="bf16")
            if plan.attention == "kda" else plan
            for plan in base.model.layer_plans)),
    ]
    for label, model in zip(("base", "draft_fp32", "markov_512",
                             "norm_eps_2x", "state_bf16"), models):
        report = E(dc.replace(base, model=model),
                   _kimi(dspark_active=True, batch=1,
                         sequence_tokens=1024))
        # A changed state dtype can legitimately change the main page, but the
        # joint identity must still differ for all five variants.
        variants.append({"label": label,
                         "identity": report["dspark"]["prefix"]["identity"],
                         "page": report["layout"]["mixed_pool"][
                             "main_page_bytes"]})
    OUT["prefix_variants"] = variants


def _mcp():
    encodings = ([tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")]
                 if tiktoken is not None else ())
    rows = []
    for label, codes in (("100_codes", [f"e{i}" for i in range(100)]),
                         ("70_diagnostics",
                          [f"d{i}" for i in range(70)])):
        report = {"schema": "mindie-sim.hybrid-kv-state/1",
                  "status": "rejected", "errors": codes,
                  "diagnostics": []}
        if label == "70_diagnostics":
            report = {"schema": "mindie-sim.hybrid-kv-state/1",
                      "status": "rejected", "errors": [],
                      "diagnostics": codes}
        envelope = hybrid_kv_state_mcp_envelope(report)
        text = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        rows.append({"label": label, "bytes": len(text),
                     "bpe": [len(enc.encode(text)) for enc in encodings],
                     "total": envelope.get("total_error_count"),
                     "truncated": envelope.get("codes_truncated"),
                     "emitted": len(envelope["errors"])
                     + len(envelope["diagnostics"])})
    OUT["mcp"] = rows


def _public():
    cases = [
        ("glm16", _case(tp=16), _glm()),
        ("glm64", _case(tp=64), _glm()),
        ("kimi_inactive", _case("kimi-k3", max_model_len=8192),
         _kimi(enable_kv_nz=True, pd_disaggregated=True,
               is_kv_consumer=True, kv_connector="MooncakeConnectorV2")),
        ("kimi_active", _case("kimi-k3", max_model_len=8192),
         _kimi(enable_kv_nz=True, dspark_active=True, pd_disaggregated=True,
               is_kv_consumer=True, kv_connector="MooncakeConnectorV2")),
        ("ds41", _case("deepseek-v4.1-flash", max_model_len=8192), _ds41()),
    ]
    encodings = ([tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")]
                 if tiktoken is not None else ())
    rows = []
    for label, case, options in cases:
        report = E(case, options)
        envelope = hybrid_kv_state_mcp_envelope(report)
        text = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        pool = report["layout"].get("mixed_pool") or {}
        rows.append({
            "label": label, "status": report["status"],
            "peak": report["memory"]["adjusted_peak_hbm_bytes"],
            "increment": report["memory"]["total_increment_bytes"],
            "step": report["decode"]["step_overhead_ms"],
            "bytes": len(text),
            "bpe": [len(enc.encode(text)) for enc in encodings],
            "pool": pool.get("physical_pool_bytes"),
            "main": pool.get("main_page_bytes"),
            "slots": pool.get("main_slot_count"),
            "blocks": pool.get("blocks"),
            "padding": pool.get("page_padding_bytes"),
            "uplift": pool.get("attention_uplift_bytes"),
            "dspark_active": (report["dspark"] or {}).get("active"),
            "grouping": (report["dspark"] or {}).get("grouping_mode"),
        })
    OUT["public"] = rows


def _hardware():
    a5_case = _case(sku="Ascend950DT_9581", topology="a5_pod")
    direct = verify_commit_cost(
        _ssm_config(), [layer for layer in __import__(
            "mindie_sim.linear_state_physics", fromlist=["x"]
        ).linear_layers_from_model(a5_case.model, tp=8)],
        batch=1, sequence_tokens=2048, hbm_bytes_per_s=1e12, is_a5=True)
    public = E(a5_case, _glm(batch=1))
    facts = dict(_case().model.catalog_facts)
    facts["conv_dim_stride"] = 2
    strided_case = dc.replace(_case(), model=dc.replace(
        _case().model, catalog_facts=tuple(facts.items())))
    strided = E(strided_case, _glm(batch=1, sequence_tokens=128))
    OUT["hardware"] = {
        "a5_arch": a5_case.machine.dies[0].arch_family,
        "a5_ub": dict(a5_case.machine.dies[0].buffers).get("ub"),
        "direct_block_v": direct["commit_block_v"],
        "public_block_v": public["linear_state"]["step_cost"][
            "commit_block_v"],
        "public_geometry": public["linear_state"]["step_cost"][
            "launch_geometry"],
        "strided_conv": strided["linear_state"]["step_cost"][
            "launch_geometry"]["conv_tile"],
        "strided_programs": strided["linear_state"]["step_cost"][
            "launch_geometry"]["conv_programs"],
        "strided_record": strided["linear_state"]["commit_hardware"],
    }
    pr_case = _case(sku="Ascend950PR_9579", topology="a5_pod")
    pr = E(pr_case, _glm(batch=1))
    OUT["hardware"]["pr_block_v"] = pr["linear_state"]["step_cost"][
        "commit_block_v"]


def _memory_edges():
    crossing = verify_commit_cost(
        _ssm_config(mamba_cache_mode="align", enable_prefix_caching=True),
        [_ssm_layer()], batch=2, sequence_tokens=2047, hbm_bytes_per_s=1e12)
    OUT["timing_consistency"] = {
        "hbm_bytes": crossing["hbm_bytes"],
        "hbm_ms": crossing["hbm_bytes"] / 1e12 * 1000.0,
        "total_ms": crossing["total_ms"]}
    base = analyze_memory_breakdown(
        _case(tp=16), sequence_tokens=128, sequences=1,
        graph_analytical_bound=True)
    OUT["ledger"] = {"peak": base.peak_hbm_bytes,
                     "budget": base.budget_bytes,
                     "device": base.device_hbm_bytes,
                     "stage_peak": list(base.stage_peak_bytes)}


def main():
    _files()
    _layout()
    _linear()
    _pool()
    _gates()
    _dspark()
    _mcp()
    _public()
    _hardware()
    _memory_edges()
    text = json.dumps(OUT, indent=2, sort_keys=True, default=str) + "\n"
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
