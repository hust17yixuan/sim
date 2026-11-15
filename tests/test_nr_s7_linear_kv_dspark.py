"""NR-S7 KV layout, Fractal-NZ, grouped pool, DSpark and prefix tests.

[Role]
  Prove the typed NoPE/RoPE/scale NZ planes, the multi-page MTE2 runs, the
  two-pass transfer staging, the grouped main/small page-class pool, the
  resolved Kimi-K3 grouping, the bounded DSpark SWA window, the DCP draft-block
  tensors and the draft-only DSpark prefix companion.

[Invariants]
  Assertions are exact byte/shape/pair identities, typed rejections or
  independently derived constants; no assertion repeats a private helper.
  Every zero is a real no-work result, never a hidden default.

[Timeouts & Locks]
  Pure in-memory arithmetic; no device, subprocess or network access.

[Side-Effect Policy]
  Builds immutable records only; writes nothing.
"""
from __future__ import annotations

import pytest

from mindie_sim.dspark_dcp_cost import non_causal_block_drafter_cost
from mindie_sim.dspark_kv_physics import (
    KvSpecSignature, draft_row_bytes_per_token, dspark_swa_footprint,
    kimi_k3_dspark_grouping, swa_bounded_replay_policy)
from mindie_sim.dspark_prefix_cache import (
    DSparkPrefixPage, ModelIdentity, dspark_prefix_keys,
    dspark_prefix_page_bytes, dspark_prefix_plan)
from mindie_sim.kv_layout_physics import (
    FRACTAL_NZ_C0_BYTES, KvNzLayout, KvNzPlane, KvNzPreconditions,
    component_major_pd_comparison, mte2_read_model, nd_to_nz_reformat_cost,
    pd_page_descriptors, validate_kv_nz)
from mindie_sim.kv_pool_geometry import grouped_pool_layout, pool_blocks
from mindie_sim.nr_s7_validate import RestoreExtent
from mindie_sim.schema.common import ModelingError
from mindie_sim.xpyd_transfer import dspark_prefix_transfer_bytes


def _target(index):
    return f"language_model.model.layers.{index}.cache"


def _draft(index):
    return f"model.layers.{index}.cache"


def _spec(name, kind="mamba", block=128, page=147456, scope="target",
          *, noncausal=False, shapes=((8, 128, 128),), dtypes=("fp32",),
          slots=4):
    attention = kind != "mamba"
    return KvSpecSignature(name, kind, block, page, scope, "kda",
                           noncausal_query_block=noncausal,
                           state_shapes=shapes if not attention else (),
                           state_dtypes=dtypes if not attention else (),
                           speculative_slots=slots if not attention else 0,
                           num_heads=64 if attention else 0,
                           num_kv_heads=8 if attention else 0,
                           head_dim=128 if attention else 0,
                           head_dtype="bf16")


# --------------------------------------------------------------- NZ planes


def test_bf16_mla_planes_are_exactly_16_element_aligned():
    layout = KvNzLayout(num_blocks=8, num_kv_heads=1, block_size=128,
                        nope_head_size=512, nope_dtype="bf16",
                        rope_head_size=64, rope_dtype="bf16")
    assert FRACTAL_NZ_C0_BYTES == 32
    assert layout.nope.c0_elements == 16
    assert layout.rope.c0_elements == 16
    assert layout.nope.tiles_per_head == 32
    assert layout.rope.tiles_per_head == 4
    assert layout.padding_bytes == 0
    assert layout.nope.shape == (8, 1, 32, 128, 16)
    assert layout.rope.shape == (8, 1, 4, 128, 16)
    assert layout.logical_bytes == 8 * 128 * (512 + 64) * 2
    assert layout.physical_bytes == layout.logical_bytes
    assert layout.view_compatible is True


def test_c8_nope_uses_32_int8_elements_and_rope_16_bf16():
    layout = KvNzLayout(num_blocks=2, num_kv_heads=1, block_size=64,
                        nope_head_size=512, nope_dtype="int8",
                        rope_head_size=64, rope_dtype="bf16",
                        scale_dim=1, scale_dtype="fp32")
    assert layout.nope.c0_elements == 32
    assert layout.rope.c0_elements == 16
    assert layout.nope.tiles_per_head == 16
    assert layout.rope.tiles_per_head == 4
    assert layout.padding_bytes == 0
    assert layout.scale_bytes == 2 * 64 * 1 * 1 * 4
    assert layout.total_bytes == 2 * 64 * (512 + 128) + 512


def test_nz_view_incompatibility_is_not_ceil_padded_silently():
    with pytest.raises(ModelingError, match="view_incompatible"):
        KvNzPlane("nope", 100, "bf16", 1, 1, 8)
    padded = KvNzPlane("nope", 100, "bf16", 1, 1, 8, pad_to_tile=True)
    assert padded.padded_head_size == 112
    assert padded.padding_bytes == 8 * 12 * 2
    with pytest.raises(ModelingError, match="view_incompatible"):
        KvNzLayout(num_blocks=1, num_kv_heads=1, block_size=8,
                   nope_head_size=100, nope_dtype="bf16")


def test_scale_plane_requires_one_byte_payload_and_integral_dim():
    with pytest.raises(ModelingError, match="one_byte_payload"):
        KvNzPlane("nope", 512, "bf16", 1, 1, 8, scale_dim=1)
    with pytest.raises(ModelingError, match="scale_dim_invalid"):
        KvNzLayout(num_blocks=1, num_kv_heads=1, block_size=8,
                   nope_head_size=512, nope_dtype="int8", scale_dim=1.0)


# ----------------------------------------------------------- NZ preconditions


def test_nz_sparse_sfa_is_rejected_even_for_mla():
    report = validate_kv_nz(KvNzPreconditions(
        enable_kv_nz=True, is_mla=True, is_sparse_sfa=True,
        pd_disaggregated=True, is_kv_consumer=True))
    assert "kv_nz_rejects_sparse_sfa" in report.codes
    assert not report.ok


def test_nz_mlapo_transition_is_model_derived():
    eligible = validate_kv_nz(KvNzPreconditions(
        enable_kv_nz=True, is_mla=True, pd_disaggregated=True,
        is_kv_consumer=True, mlapo_eligible=True))
    assert eligible.mlapo_eligible and not eligible.mlapo_enabled
    assert eligible.mlapo_disabled_by_nz
    supported = validate_kv_nz(KvNzPreconditions(
        enable_kv_nz=True, is_mla=True, pd_disaggregated=True,
        is_kv_consumer=True, mlapo_eligible=True, nz_supports_mlapo=True))
    assert supported.mlapo_enabled and not supported.mlapo_disabled_by_nz


@pytest.mark.parametrize("overrides,code", [
    ({"is_mla": False}, "kv_nz_requires_mla"),
    ({"is_sparse_sfa": True}, "kv_nz_rejects_sparse_sfa"),
    ({"pd_disaggregated": False, "is_kv_consumer": False},
     "kv_nz_requires_pd_deployment"),
    ({"is_kv_consumer": False}, "kv_nz_rejected_on_prefill_node"),
])
def test_nz_topology_rejections(overrides, code):
    config = dict(enable_kv_nz=True, is_mla=True, pd_disaggregated=True,
                  is_kv_consumer=True)
    config.update(overrides)
    report = validate_kv_nz(KvNzPreconditions(**config))
    assert code in report.codes


# ----------------------------------------------------------------- MTE2 pages


def test_mte2_multi_page_runs_never_cross_a_page_boundary():
    layout = KvNzLayout(num_blocks=4, num_kv_heads=1, block_size=128,
                        nope_head_size=64, nope_dtype="bf16")
    model = mte2_read_model(layout, tokens=256, local_kv_heads=1,
                            block_ids=(0, 2), cacheline_bytes=256)
    assert model.pages_used == 2
    # One head x four tiles x two separate pages.
    assert model.nz_dma_ops == 1 * 4 * 2 == 8
    assert model.nz_transactions == 2 * 4 * (4096 // 256) == 128
    tail = mte2_read_model(layout, tokens=200, local_kv_heads=1,
                           block_ids=(1, 3), cacheline_bytes=256)
    assert tail.pages_used == 2
    assert tail.nz_transactions == 4 * (4096 // 256) + 4 * 9


def test_mte2_rejects_tokens_beyond_the_allocated_pages():
    layout = KvNzLayout(num_blocks=1, num_kv_heads=1, block_size=8,
                        nope_head_size=64, nope_dtype="bf16")
    with pytest.raises(ModelingError, match="tokens_exceed_capacity"):
        mte2_read_model(layout, tokens=9, local_kv_heads=1)
    with pytest.raises(ModelingError, match="block_ids_invalid"):
        mte2_read_model(layout, tokens=8, local_kv_heads=1, block_ids=(0, 1))


# ------------------------------------------------------------ NZ reformat


def test_nz_reformat_is_two_pass_with_one_reusable_staging_set():
    layout = KvNzLayout(num_blocks=8, num_kv_heads=1, block_size=128,
                        nope_head_size=512, rope_head_size=64)
    page = 128 * 576 * 2
    cost = nd_to_nz_reformat_cost(
        pages=8, layers=3, nd_page_bytes=page, hbm_bytes_per_s=1.0e12,
        kernel_fixed_ms=0.01, synchronize_ms=0.02, nz_layout=layout)
    assert cost.path == "two_pass"
    assert cost.hbm_traffic_bytes == 4 * 8 * 3 * page
    assert cost.kernel_launches == 2 * 3 + 1
    assert cost.staging_bytes == 8 * page
    assert cost.staging_reconciled_bytes == 8 * page
    assert cost.nz_padding_bytes == 0
    deduplicated = nd_to_nz_reformat_cost(
        pages=8, layers=3, nd_page_bytes=page, hbm_bytes_per_s=1.0e12,
        configured_staging_bytes=8 * page, nz_layout=layout)
    assert deduplicated.staging_reconciled_bytes == 0


def test_nz_fused_path_and_negative_padding_are_rejected():
    layout = KvNzLayout(num_blocks=1, num_kv_heads=1, block_size=8,
                        nope_head_size=64)
    with pytest.raises(ModelingError, match="fused_nz_reformat_unsupported"):
        nd_to_nz_reformat_cost(pages=1, layers=1, nd_page_bytes=1024,
                               hbm_bytes_per_s=1.0e12, fused_op=True,
                               nz_layout=layout)
    with pytest.raises(ModelingError, match="extent_too_small"):
        nd_to_nz_reformat_cost(pages=1, layers=1, nd_page_bytes=1_000_000,
                               hbm_bytes_per_s=1.0e12, nz_layout=layout)


# ------------------------------------------------------------ PD descriptors


def test_component_major_descriptor_startup_is_exact():
    comparison = component_major_pd_comparison(
        pages=8, layers=61, nope_bytes=512, rope_bytes=64, padding_bytes=8,
        per_descriptor_setup_us=2.0, doorbell_us=1.0)
    assert comparison["component_major"]["descriptors"] == 488
    assert comparison["split"]["descriptors"] == 976
    assert comparison["component_major"]["startup_ms"] == pytest.approx(1.464)
    assert comparison["split"]["startup_ms"] == pytest.approx(2.928)
    assert comparison["payload_conserved"]
    assert pd_page_descriptors(
        pages=2, layers=3, nope_bytes=100, rope_bytes=0,
        component_major=False).descriptors == 6


# ------------------------------------------------------------- grouped pools


def test_grouped_pool_uses_max_group_slots_not_layer_sum():
    groups = ((67_840,) * 12, (67_840,) * 11, (67_840,) * 11)
    pool = grouped_pool_layout(
        attn_group_size=11, attn_page_bytes=147_456,
        recurrent_groups=groups, attention_blocks=100,
        recurrent_blocks=(8, 8, 8))
    assert pool.recurrent_group_count == 3
    assert pool.recurrent_group_sizes == (12, 11, 11)
    assert pool.main_slot_count == 12
    assert pool.main_page_bytes == 147_456
    assert pool.blocks == 100 + 24
    assert pool.physical_pool_bytes == 124 * 12 * 147_456
    bare = 100 * 11 * 147_456 + 8 * 34 * 67_840
    assert pool.layer_sum_bytes == bare
    assert pool.bare_slot_bytes == bare
    assert pool.page_padding_bytes == (
        pool.physical_pool_bytes - pool.bare_slot_bytes)
    assert pool.slot_sharing_saving_bytes == 0


def test_grouped_pool_pads_attention_up_when_state_dominates():
    pool = grouped_pool_layout(
        attn_group_size=2, attn_page_bytes=100_000,
        recurrent_groups=((200_000, 150_000),),
        attention_blocks=3, recurrent_blocks=(3,))
    assert pool.main_page_bytes == 200_000
    assert pool.blocks == 6
    assert pool.attention_uplift_bytes == 3 * 2 * 100_000
    assert pool.page_padding_bytes == (
        pool.physical_pool_bytes - pool.bare_slot_bytes)
    assert pool.physical_pool_bytes == 6 * 2 * 200_000


def test_grouped_pool_reconciles_a_padded_baseline_net_once():
    pool = grouped_pool_layout(
        attn_group_size=1, attn_page_bytes=100_000,
        recurrent_groups=((50_000,),), attention_blocks=2,
        recurrent_blocks=(2,))
    # Attention and recurrent groups hold independent block IDs.
    padded_baseline = (2 + 2) * 100_000
    result = pool.reconcile(padded_baseline)
    assert result["modeled_bytes"] == 400_000
    assert result["delta_bytes"] == 0
    assert result["retained_baseline_bytes"] == 0
    small = pool.reconcile(10_000)
    assert small["delta_bytes"] == 390_000


def test_pool_blocks_are_attention_blocks_times_batch():
    assert pool_blocks(tokens=128, block_size=128, batch=1) == 1
    assert pool_blocks(tokens=2048, block_size=128, batch=8) == 128


# ------------------------------------------------------------ Kimi grouping


def test_kimi_grouping_rejects_duplicate_layer_names():
    specs = [_spec("same", "mla_attention", page=147_456),
             _spec("same", "mla_attention", page=147_456)]
    with pytest.raises(ModelingError, match="duplicate_layer_name"):
        kimi_k3_dspark_grouping(specs)


def test_kimi_grouping_rejects_same_byte_different_state_shape():
    left = _spec(_target(10), shapes=((8, 128, 128),))
    right = _spec(_target(11), shapes=((8, 64, 256),))
    grouping = kimi_k3_dspark_grouping([
        _spec(_target(0), "mla_attention", page=147_456),
        _spec(_draft(0), "mla_attention", page=147_456, scope="draft"),
        left, right])
    assert grouping.fallback_reason == "mamba_specs_not_uniform"


def test_kimi_grouping_rejects_different_mamba_dtype():
    left = _spec(_target(10), dtypes=("fp32",))
    right = _spec(_target(11), dtypes=("bf16",))
    grouping = kimi_k3_dspark_grouping([
        _spec(_target(0), "mla_attention", page=147_456),
        _spec(_draft(0), "mla_attention", page=147_456, scope="draft"),
        left, right])
    assert grouping.fallback_reason == "mamba_specs_not_uniform"


def test_kimi_grouping_labels_a_non_causal_draft_split_as_analytical():
    specs = [_spec(_target(i), "mla_attention", page=147_456)
             for i in range(2)]
    specs += [_spec(_draft(0), "full_attention", page=147_456, scope="draft",
                    noncausal=True)]
    specs += [_spec(_target(10 + i)) for i in range(7)]
    grouping = kimi_k3_dspark_grouping(specs)
    # The source's uniform merge has no causal-plus-non-causal branch; the
    # useful two-table alternative is labelled analytical and is not a source
    # scheduler grouping.
    assert not grouping.ok
    assert grouping.source_compatible is False
    assert grouping.mode == "analytical_noncausal_split"
    assert grouping.fallback_reason == (
        "noncausal_split_is_not_source_uniform_grouping")
    assert grouping.noncausal_split
    assert grouping.scheduler_block_tables == 0
    assert grouping.analytical_block_tables == 2
    assert grouping.group_sizes == (2, 1, 2, 2, 2, 1)
    with pytest.raises(ModelingError, match="noncausal_split_is_not_source"):
        kimi_k3_dspark_grouping(specs, strict=True)


def test_kimi_grouping_balances_compatible_recurrent_groups():
    specs = [_spec(_target(i), "mla_attention", page=147_456)
             for i in range(24)]
    specs += [_spec(_draft(i), "mla_attention", page=147_456, scope="draft")
              for i in range(5)]
    specs += [_spec(_target(100 + i)) for i in range(69)]
    grouping = kimi_k3_dspark_grouping(specs, strict=True)
    assert grouping.group_sizes == (29, 23, 23, 23)
    assert grouping.scheduler_block_tables == 1
    assert grouping.mamba_group_count == 3


# ------------------------------------------------------------------ SWA bound


def test_swa_footprint_uses_local_k_plus_v_rows():
    row = draft_row_bytes_per_token(kv_heads=16, head_dim=64, tp=8,
                                    dtype="bf16")
    assert row == 2 * 2 * 64 * 2 == 512
    footprint = dspark_swa_footprint(
        sequence_tokens=4096, swa_window_tokens=128,
        extra_retained_tokens=8, kv_block_size=128, layers=5,
        kv_heads=16, head_dim=64, tp=8, dtype="bf16")
    assert footprint.visible_tokens == 136
    assert footprint.retained_tokens == 256
    assert footprint.bounded_bytes == 256 * 5 * 512 == 655_360


def test_draft_swa_never_replays_but_target_bounded_replay_does():
    target = swa_bounded_replay_policy(
        architecture="DeepseekV41ForCausalLM", swa_bounded_replay=True,
        prefix_hit=1000, swa_window=128)
    assert target["replay_tokens"] == 128
    draft = swa_bounded_replay_policy(
        architecture="DeepseekV41ForCausalLM", swa_bounded_replay=True,
        prefix_hit=1000, swa_window=128,
        draft_cache_class="DeepseekV41DSparkSWACache", is_draft_layer=True)
    assert draft["replay_tokens"] == 0
    assert "dspark_swa_bounded_replay_excluded" in draft["diagnostics"]


# --------------------------------------------------------------- DCP block


def _dcp(**overrides):
    values = dict(draft_block_size=5, context_tokens=4096, dcp=2, pcp=1,
                  tp=8, num_q_heads=64, q_nope_dim=512, q_rope_dim=64,
                  kv_heads=16, head_dim=64, latent_width=512,
                  num_draft_layers=5, batch=1, swa_window=128,
                  collective_bytes_per_s=1.0e12, hbm_bytes_per_s=1.0e12,
                  compute_flops_per_s=4.0e14, index_entries_per_us=1000.0)
    values.update(overrides)
    return values


def test_dcp_gather_moves_query_heads_not_draft_kv():
    cost = non_causal_block_drafter_cost(**_dcp())
    local_heads = 64 // 8
    per_token = (512 + 64) * local_heads * 2
    assert cost["local_q_heads"] == local_heads
    assert cost["gather_peers"] == 1
    assert cost["gather_bytes"] == 1 * 5 * 5 * per_token
    assert cost["gather_bytes"] != 5 * 16 * 64 * 2


def test_dcp_merge_includes_fp32_lse_and_sums_the_dependency_chain():
    cost = non_causal_block_drafter_cost(**_dcp())
    # 25 rows, 16 gathered heads; FP32 LSE packs into four BF16 elements.
    output = 25 * 16 * 512 * 2
    lse = 25 * 16 * 4 * 2
    assert cost["gathered_heads"] == 16
    assert cost["combine_bytes"] == pytest.approx(0.5 * (output + lse))
    assert cost["combine_bytes"] == 206_400
    assert cost["total_ms"] == pytest.approx(
        cost["index_prep_ms"] + cost["gather_ms"] + cost["attention_ms"]
        + cost["combine_ms"])
    assert cost["total_ms"] > cost["gather_ms"] + cost["combine_ms"]
    assert cost["dependency_chain"] == ["gather", "attention", "combine"]


def test_dcp_index_is_one_plane_plus_one_length_bounded_by_window():
    cost = non_causal_block_drafter_cost(**_dcp(batch=8))
    assert cost["visible_tokens"] == 133
    assert cost["index_width"] == 256
    assert cost["index_entries"] == 8 * 5 * 5 * 256
    assert cost["index_bytes"] == 8 * 5 * 5 * (256 * 4 + 4)
    assert cost["index_entries"] < 2 * 8 * 5 * (4096 + 5)


def test_dcp_scales_by_batch_and_draft_layers():
    one = non_causal_block_drafter_cost(**_dcp(batch=1, num_draft_layers=1))
    many = non_causal_block_drafter_cost(**_dcp(batch=8, num_draft_layers=5))
    assert many["gather_bytes"] == pytest.approx(40 * one["gather_bytes"])
    assert many["index_bytes"] == 40 * one["index_bytes"]


def test_pcp_gathers_distinct_tp_heads():
    cost = non_causal_block_drafter_cost(**_dcp(pcp=2, dcp=16))
    assert cost["gather_peers"] == 7


# ------------------------------------------------------------ prefix identity


def _identity(dtype="bf16"):
    return ModelIdentity(
        name="kimi", activation_dtype=dtype, kv_dtype="bf16",
        revision="r1", code_revision="c1", quantization="",
        config=(("hidden_size", "7168"),))


def test_prefix_identity_changes_when_model_dtype_changes():
    bf16 = dspark_prefix_keys(
        target=_identity("bf16"), draft=_identity("bf16"), aux_layers=(7,),
        tp_size=8, pp_size=1, block_size=128, prefix_match_unit=128,
        draft_width=5)
    fp32 = dspark_prefix_keys(
        target=_identity("fp32"), draft=_identity("bf16"), aux_layers=(7,),
        tp_size=8, pp_size=1, block_size=128, prefix_match_unit=128,
        draft_width=5)
    assert bf16.identity != fp32.identity
    assert bf16.target_hash != fp32.target_hash


# -------------------------------------------------------------- prefix pages


def _page(name="l0", group=0, shape0=16, shape1=64, numel=64, element=2,
          stride=64, row_bytes=2):
    return DSparkPrefixPage(name, group, shape0, shape1, numel, element,
                            stride, row_bytes)


def test_prefix_page_requires_physical_capacity_and_group_coverage():
    geometry = dspark_prefix_page_bytes(
        pages=[_page()], num_blocks=8, block_size=128,
        hash_block_size=128, group_ids=(0,), block_sizes=(128,),
        draft_group_ids=(0,))
    assert geometry["page_bytes"] == 256
    with pytest.raises(ModelingError, match="capacity_too_small"):
        dspark_prefix_page_bytes(
            pages=[_page(numel=1)], num_blocks=8, block_size=128,
            hash_block_size=128, group_ids=(0,), block_sizes=(128,),
            draft_group_ids=(0,))
    with pytest.raises(ModelingError, match="draft_group_uncovered"):
        dspark_prefix_page_bytes(
            pages=[_page()], num_blocks=8, block_size=128,
            hash_block_size=128, group_ids=(0,), block_sizes=(128,),
            draft_group_ids=(0, 1))
    with pytest.raises(ModelingError, match="hash_block_must_equal"):
        dspark_prefix_page_bytes(
            pages=[_page()], num_blocks=8, block_size=128,
            hash_block_size=64, group_ids=(0,), block_sizes=(128,),
            require_hash_equals_block=True)


def test_prefix_plan_is_draft_only_and_reports_rank_scope():
    keys = dspark_prefix_keys(
        target=_identity(), draft=_identity(), aux_layers=(7, 23),
        tp_size=8, pp_size=1, block_size=128, prefix_match_unit=128,
        draft_width=5)
    geometry = dspark_prefix_page_bytes(
        pages=[_page()], num_blocks=8, block_size=128,
        hash_block_size=128, group_ids=(0,), block_sizes=(128,))
    plan = dspark_prefix_plan(
        keys=keys, page_geometry=geometry, aux_layer_ids=(7, 23),
        target_hidden_size=7168, target_dtype="bf16", tokens=1024,
        prompt_tokens=1024, block_hash_count=8,
        metadata_bytes_per_object=64, aux_scratch_tokens=10)
    assert plan.companion_page_bytes == 256
    assert plan.aux_scratch_bytes == 2 * 10 * 7168 * 2
    assert plan.per_rank_storage_bytes == 8 * 256 + 8 * 64
    assert plan.aggregate_storage_bytes == 8 * plan.per_rank_storage_bytes
    payload = plan.to_dict()
    assert payload["aux_scratch_persistent"] is False
    assert payload["scope"] == "per_rank"


def test_shared_restore_validator_rejects_partial_and_over_prompt():
    with pytest.raises(ModelingError, match="requires_full_blocks"):
        RestoreExtent(prefix_hit_tokens=129, prompt_tokens=256,
                      block_size=128, hash_block_size=128,
                      block_hash_count=2)
    with pytest.raises(ModelingError, match="hit_exceeds_prompt"):
        RestoreExtent(prefix_hit_tokens=512, prompt_tokens=256,
                      block_size=128, hash_block_size=128,
                      block_hash_count=2)
    with pytest.raises(ModelingError, match="missing_block_hashes"):
        RestoreExtent(prefix_hit_tokens=256, prompt_tokens=256,
                      block_size=128, hash_block_size=128,
                      block_hash_count=1).validated_blocks()


def test_xpyd_prefix_transfer_uses_the_same_validator():
    with pytest.raises(ModelingError, match="requires_full_blocks"):
        dspark_prefix_transfer_bytes(
            prompt_tokens=129, prefix_hit_tokens=129, block_size=128,
            draft_page_bytes=256, aux_layer_ids=(7,), hidden_size=128,
            dtype="bf16")
    with pytest.raises(ModelingError, match="hit_exceeds_prompt"):
        dspark_prefix_transfer_bytes(
            prompt_tokens=128, prefix_hit_tokens=256, block_size=128,
            draft_page_bytes=256, aux_layer_ids=(7,), hidden_size=128,
            dtype="bf16")
    with pytest.raises(ModelingError, match="draft_page_bytes_invalid"):
        dspark_prefix_transfer_bytes(
            prompt_tokens=128, prefix_hit_tokens=128, block_size=128,
            draft_page_bytes=0.5, aux_layer_ids=(7,), hidden_size=128,
            dtype="bf16")
    valid = dspark_prefix_transfer_bytes(
        prompt_tokens=256, prefix_hit_tokens=256, block_size=128,
        draft_page_bytes=256, aux_layer_ids=(7,), hidden_size=128,
        dtype="bf16", tp_size=8, metadata_bytes_per_object=8)
    assert valid["per_rank_transfer_bytes"] == 2 * (256 + 8)
    assert valid["aggregate_transfer_bytes"] == 8 * 2 * 264
    assert valid["aux_scratch_persistent"] is False


# ------------------------------------------------------- pre-coercion (F14)


@pytest.mark.parametrize("kwargs", [
    dict(hbm_bytes_per_s=True), dict(hbm_bytes_per_s="1000"),
    dict(hbm_bytes_per_s=0), dict(kernel_fixed_ms=True),
    dict(synchronize_ms=float("nan")),
])
def test_reformat_rejects_non_numeric_or_non_positive_rates(kwargs):
    layout = KvNzLayout(num_blocks=1, num_kv_heads=1, block_size=8,
                        nope_head_size=64)
    base = dict(pages=1, layers=1, nd_page_bytes=1024,
                hbm_bytes_per_s=1.0e12, nz_layout=layout)
    base.update(kwargs)
    with pytest.raises(ModelingError):
        nd_to_nz_reformat_cost(**base)


def test_mte2_zero_or_boolean_bandwidth_is_rejected():
    layout = KvNzLayout(num_blocks=1, num_kv_heads=1, block_size=8,
                        nope_head_size=64)
    for value in (0, True, "1000", float("nan")):
        with pytest.raises(ModelingError):
            mte2_read_model(layout, tokens=8, local_kv_heads=1,
                            mte2_bytes_per_us=value)
