"""Composed layout, DSpark and prefix blocks of the hybrid evaluation.

[Role]
  Build the NZ and grouped-pool sub-records of one hybrid evaluation,
  including their per-stage replacement deltas, from the catalog model and the
  declared options; the DSpark/prefix sub-record lives in
  ``hybrid_kv_state_dspark`` and is re-exported here.

[Invariants]
  * Sparse-SFA and MLAPO facts are supplied by the caller from model ground
    truth; an NZ rejection yields an explicit code, never a silent layout.
  * The grouped pool models main/small page classes with the largest scheduler
    group as the slot count, and reconciles against the baseline's already
    priced components per stage.
  * The DSpark sub-record is a re-export; its module owns draft SWA extent,
    prefix companion pages and the transient aux capture workspace.
  * Every public value is a strict integer/finite number; malformed model
    geometry is a typed ``ModelingError``.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.schema.case import Case
from mindie_sim.schema.model import ModelSpec, LayerPlan
from mindie_sim.hybrid_kv_state_dspark import _dspark_block, _model_identity
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, _attention_layer_count, _attention_layers_for_stage,
    _attention_row, _linear_layers_for_stage, _model_type, _stage_partition,
    pd_role)
from mindie_sim.hybrid_kv_state_pool import (
    REPLACED_COMPONENTS, SMALL_CLASS_COMPONENTS, baseline_pricing, stage_deltas)
from mindie_sim.kv_layout_physics import (
    KvNzLayout, KvNzPreconditions, mte2_read_model, nd_to_nz_reformat_cost,
    validate_kv_nz)
from mindie_sim.kv_pool_geometry import grouped_pool_layout, pool_blocks
from mindie_sim.linear_state_physics import LinearLayer
from mindie_sim.nr_s7_validate import strict_int, strict_number, positive_rate
from mindie_sim.schema.common import ModelingError, dtype_bytes

SHARED_POOL_MODEL_TYPES = frozenset({
    "glm5_next", "glm5_next_text", "deepseek_v41_text", "qwen4_exp_text"})
BOUNDED_REPLAY_ARCHITECTURES = frozenset({
    "DeepseekV41ForCausalLM", "DeepseekV41ForConditionalGeneration"})


def _runtime_int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="hybrid_kv_runtime")


@typed_inputs
def _typed_mla_planes(model: ModelSpec, plan: LayerPlan) -> tuple[int, int, str, str, int]:
    """Exact ``(nope, rope, nope_dtype, rope_dtype, scale_dim)`` for one MLA page.

    Under a one-byte KV dtype (``c8``) only NoPE is INT8/FP8 (``C0 = 32``);
    RoPE keeps the activation/RoPE dtype (``C0 = 16`` for BF16) and a per-token
    FP32 scale plane is added.  A non-c8 page keeps both planes in ``kv_dtype``.
    """
    rope = _runtime_int("qk_rope_head_dim",
                        getattr(plan, "qk_rope_head_dim", 0))
    nope = _runtime_int("nope_head_dim", plan.cache_row_elements - rope,
                        minimum=1)
    kv_dtype = str(getattr(model, "kv_dtype", "bf16"))
    c8 = dtype_bytes(kv_dtype) == 1.0
    rope_dtype = str(getattr(model, "activation_dtype", "bf16")) if c8 \
        else kv_dtype
    return nope, rope, kv_dtype, rope_dtype, (1 if c8 else 0)


@typed_inputs
def _attention_page_bytes(model: ModelSpec, plan: LayerPlan, block_size: int,
                          kv_elem: float, tp: int) -> int:
    """Per-rank page bytes: typed MLA planes or TP-sharded GQA K+V."""
    _runtime_int("block_size", block_size, minimum=1)
    _runtime_int("tp", tp, minimum=1)
    positive_rate("kv_elem", kv_elem)
    if plan.attention == "mla":
        nope, rope, nope_dtype, rope_dtype, scale_dim = _typed_mla_planes(
            model, plan)
        page = (block_size * nope * dtype_bytes(nope_dtype)
                + block_size * rope * dtype_bytes(rope_dtype))
        return int(page + (block_size * scale_dim * 4 if scale_dim else 0))
    if plan.attention == "gqa":
        declared = _runtime_int("num_kv_heads", plan.num_kv_heads, minimum=1)
        shards = _runtime_int("tp", tp, minimum=1)
        if declared % shards and shards % declared:
            raise ModelingError(
                f"hybrid_kv_gqa_heads_not_divisible_by_tp:{declared}%{shards}")
        heads = max(1, declared // shards)
        return int(block_size * 2 * heads * plan.head_dim * kv_elem)
    return int(block_size * plan.cache_row_elements * kv_elem)


@typed_inputs
def _pd_state(options: HybridKvStateOptions) -> tuple[bool, bool, bool]:
    consumer, producer = pd_role(options)
    return options.pd_disaggregated, consumer, producer


@typed_inputs
def _nz_block(parsed: Case, options: HybridKvStateOptions, *, attn_kind: str,
              attn_row: int, block_size: int, batch: int, tokens: int,
              sparse: bool, mlapo: bool, hbm: float, launch_ms: float,
              staging_bytes: int, errors: list[str]) -> tuple[dict[str, Any],
                                                              list[float]]:
    for name, value in (("attn_row", attn_row), ("block_size", block_size), ("batch", batch), ("tokens", tokens)):
        _runtime_int(name, value, minimum=1)
    _runtime_int("staging_bytes", staging_bytes)
    positive_rate("hbm", hbm)
    strict_number("launch_ms", launch_ms)
    if attn_kind not in {"mla", "gqa"}:
        raise ModelingError("hybrid_kv_attention_kind_invalid")
    model = parsed.model
    pd, consumer, producer = _pd_state(options)
    partition = _stage_partition(parsed.runtime, model)
    layout: dict[str, Any] = {"enable_kv_nz": options.enable_kv_nz}
    staging = [0.0] * len(partition)
    if not options.enable_kv_nz:
        return layout, staging
    validation = validate_kv_nz(KvNzPreconditions(
        enable_kv_nz=True, is_mla=attn_kind == "mla", is_sparse_sfa=sparse,
        kv_connector=options.kv_connector,
        kv_connector_module_path=options.kv_connector_module_path,
        is_kv_consumer=consumer, is_kv_producer=producer,
        pd_disaggregated=pd, kv_role=options.kv_role,
        mlapo_eligible=mlapo), strict=False)
    layout["kv_nz_validation"] = validation.to_dict()
    errors.extend(validation.codes)
    if not validation.ok:
        return layout, [0.0] * len(partition)
    mla = getattr(model, "mla", None)
    rope = _runtime_int("qk_rope_head_dim",
                        getattr(mla, "qk_rope_head_dim", 0) if mla else 0)
    nope = _runtime_int("nope_head_dim", attn_row - rope)
    c8 = (attn_kind == "mla" and dtype_bytes(model.kv_dtype) == 1.0)
    nope_dtype = str(model.kv_dtype)
    rope_dtype = (str(getattr(model, "activation_dtype", "bf16"))
                  if c8 else nope_dtype)
    scale_dim = 1 if c8 else 0
    blocks = pool_blocks(tokens=tokens, block_size=block_size, batch=batch,
                         cp=int(parsed.runtime.parallel.cp))
    nz_layout = KvNzLayout(
        num_blocks=blocks, num_kv_heads=1, block_size=block_size,
        nope_head_size=nope, nope_dtype=nope_dtype, rope_head_size=rope,
        rope_dtype=rope_dtype, scale_dim=scale_dim)
    mte2 = mte2_read_model(
        nz_layout, tokens=block_size, local_kv_heads=1,
        mte2_bytes_per_us=hbm / 1e6, dma_setup_us=launch_ms * 1000.0)
    nd_page = int(block_size * nope * dtype_bytes(nope_dtype)
                  + block_size * rope * dtype_bytes(rope_dtype))
    scale_bytes = int(block_size * scale_dim * 4) if c8 else 0
    configured = float(staging_bytes)
    reformat = nd_to_nz_reformat_cost(
        pages=blocks, layers=_attention_layer_count(model),
        nd_page_bytes=nd_page, hbm_bytes_per_s=hbm,
        fused_op=False, kernel_fixed_ms=launch_ms,
        synchronize_ms=launch_ms, nz_layout=nz_layout,
        nd_scale_bytes=scale_bytes,
        configured_staging_bytes=int(configured))
    # The reusable two-pass staging is live only on PP stages that own at
    # least one MLA layer participating in the NZ reformat.  A stage that owns
    # only KDA layers allocates zero transfer staging.
    buffer_bytes = max(configured, reformat.staging_bytes)
    for stage in range(len(partition)):
        own_mla = [plan for plan in _attention_layers_for_stage(
            model, stage, partition) if plan.attention == "mla"]
        if own_mla:
            staging[stage] = buffer_bytes
    layout["kv_nz"] = {"layout": nz_layout.to_dict(),
                       "mte2": mte2.to_dict(),
                       "reformat": reformat.to_dict(),
                       "staging_per_stage_bytes": list(staging),
                       "mlapo_eligible": validation.mlapo_eligible,
                       "mlapo_enabled": validation.mlapo_enabled,
                       "mlapo_disabled_by_nz": validation.mlapo_disabled_by_nz}
    return layout, staging


def _recurrent_page_bytes(layer: LinearLayer, tp: int, spec: int,
                          recoverssm: bool) -> int:
    """One layer's physical recurrent page (state + widened conv + records)."""
    if recoverssm:
        return int(layer.page_geometry(tp, spec).page_bytes)
    return int(layer.state_bytes(tp)
               + layer.conv_bytes(tp, spec_query_len=spec))


@typed_inputs
def _pool_block(parsed: Case, options: HybridKvStateOptions, layers: tuple[LinearLayer, ...],
                *, tokens: int, batch: int, block_size: int, recoverssm: bool,
                sparse: bool, errors: list[str],
                warnings: list[str]) -> tuple[dict[str, Any] | None, list[float],
                                              list[float]]:
    """Per-stage grouped pool with source-exact grouping and block demand.

    The global page class and the recurrent group assignment are derived from
    the full model layer sequence first; each PP stage then only projects its
    owned layers onto those groups.  Independent scheduler groups allocate
    independent physical block IDs, so attention and recurrent demands add.
    """
    from mindie_sim.kv_pool_geometry import (
        balanced_recurrent_groups, longest_run_recurrent_groups, project_groups)
    for name, value in (("tokens", tokens), ("batch", batch), ("block_size", block_size)):
        _runtime_int(name, value, minimum=1)
    model, runtime = parsed.model, parsed.runtime
    partition = _stage_partition(runtime, model)
    stages = len(partition)
    tp = int(runtime.parallel.tp)
    # ``get_kda_parallel_size``: PCP shards KDA heads only when TP is one.
    shard = tp if tp != 1 else int(runtime.parallel.pcp)
    spec = options.num_spec + 1
    kv_elem = dtype_bytes(model.kv_dtype)
    model_type = _model_type(model)
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    attention_plans = tuple(plan for plan in plans
                            if plan.attention in {"mla", "gqa"})
    page_by_layer: dict[int, int] = {}
    for plan in attention_plans:
        page_by_layer[plan.index] = _attention_page_bytes(
            model, plan, block_size, kv_elem, tp)
    raw_pages = {layer.index: _recurrent_page_bytes(
        layer, shard, spec, recoverssm) for layer in layers}
    raw_state_page = max(raw_pages.values(), default=0)
    stride = options.state_page_stride_bytes if recoverssm else None
    if stride is not None:
        too_small = [index for index, page in raw_pages.items()
                     if stride < page]
        if too_small:
            raise ModelingError(
                "kv_pool_state_page_stride_smaller_than_page:"
                f"{stride}<{max(raw_pages[index] for index in too_small)}")
        raw_pages = {index: max(page, stride)
                     for index, page in raw_pages.items()}
    page_by_layer.update(raw_pages)
    global_main_page = max(page_by_layer.values(), default=0)
    if not global_main_page:
        return None, [0.0] * stages, [0.0] * stages
    recurrent_indices = [layer.index for layer in layers]
    glm_family = model_type in {"glm5_next", "glm5_next_text"}
    if glm_family:
        global_groups = longest_run_recurrent_groups(
            [(plan.index, plan.attention) for plan in plans])
    else:
        global_groups = balanced_recurrent_groups(
            recurrent_indices,
            attention_group_size=max(1, len(attention_plans)))
    config_block = (options.mamba_block_size or block_size)
    if recoverssm:
        state_slots = (1 if options.mamba_cache_mode == "none"
                       else -(-tokens // config_block))
    elif options.mamba_cache_mode == "align":
        # A standard-path align cache still checkpoints block-aligned state;
        # demand the larger of that page count and its K + 1 rollback slots
        # so the pool always covers the linear-state plan.
        state_slots = max(spec, -(-tokens // config_block))
    else:
        state_slots = spec
    attention_blocks = pool_blocks(tokens=tokens, block_size=block_size,
                                   batch=batch,
                                   cp=int(runtime.parallel.cp))
    # GLM-Next keeps one compressor-tail scheduler group.  Its one block per
    # request draws IDs from the same global pool, so it consumes main-slot
    # address space even though its own payload belongs to the small class.
    indexed_attention = [plan for plan in attention_plans
                         if getattr(plan, "indexer", False)]
    tail_blocks = (batch if glm_family and indexed_attention else 0)
    index_dim = max((_runtime_int("index_dim", getattr(plan, "index_dim", 0))
                     for plan in indexed_attention), default=0)
    small_page = (int(2 * block_size * index_dim * 4)
                  if tail_blocks and index_dim else 0)
    small_slots = len(indexed_attention) if tail_blocks else 0
    # Global scheduler-group ID demand, computed once on the full model.  A PP
    # stage projects slot ownership onto this domain; it never recomputes (and
    # therefore never shrinks) the number of global block IDs.
    global_group_blocks = tuple(int(state_slots * batch)
                                for _ in global_groups)
    global_blocks = attention_blocks + tail_blocks + sum(global_group_blocks)
    global_main_slots = max((len(attention_plans),
                             *(len(group) for group in global_groups)))
    baseline = baseline_pricing(parsed, tokens=tokens,
                                active_sequences=batch)
    main_baseline = baseline.replaced_per_stage
    if glm_family:
        baseline = baseline_pricing(
            parsed, tokens=tokens, active_sequences=batch,
            components=tuple(sorted(REPLACED_COMPONENTS | SMALL_CLASS_COMPONENTS)))
    small_baseline = [total - main for total, main in zip(
        baseline.replaced_per_stage, main_baseline)]
    modeled: list[float] = []
    baselines: list[float] = []
    per_stage_dicts: list[dict[str, Any]] = []
    projected_sizes: list[list[int]] = []
    for stage in range(stages):
        attention = _attention_layers_for_stage(model, stage, partition)
        recurrent = _linear_layers_for_stage(model, layers, stage, partition)
        attn_page = max((page_by_layer[plan.index] for plan in attention),
                        default=0)
        owned = [layer.index for layer in recurrent]
        projected = project_groups(global_groups, owned)
        group_pages = tuple(tuple(raw_pages[index] for index in group)
                            for group in projected)
        group_blocks = tuple(int(state_slots * batch) if group else 0
                             for group in projected)
        pool = grouped_pool_layout(
            attn_group_size=len(attention), attn_page_bytes=attn_page,
            recurrent_groups=group_pages,
            attention_blocks=attention_blocks if attention else 0,
            recurrent_blocks=group_blocks,
            indexes_kv_by_block_stride=True,
            small_slot_count=(sum(plan.indexer for plan in attention)
                              if glm_family else 0),
            small_page_bytes=(small_page if glm_family and any(
                plan.indexer for plan in attention) else 0),
            declared_page_size_padded=global_main_page,
            block_domain=global_blocks)
        modeled.append(float(pool.physical_pool_bytes))
        baselines.append(float(baseline.replaced_per_stage[stage]))
        per_stage_dicts.append({**pool.to_dict(),
                               "main_class_bytes": pool.main_slot_bytes,
                               "small_class_bytes": pool.small_class_bytes})
        projected_sizes.append(list(pool.recurrent_group_sizes))
    deltas = list(stage_deltas(modeled=modeled, replaced=baselines,
                               conservative=not glm_family))
    aggregate = {
        "blocks": global_blocks,
        "global_block_domain": global_blocks,
        "stage_blocks": [record["blocks"] for record in per_stage_dicts],
        "attention_blocks": attention_blocks,
        "recurrent_blocks": sum(global_group_blocks),
        "tail_blocks": tail_blocks,
        "stages": per_stage_dicts,
        "main_page_bytes": max((record["main_page_bytes"]
                                for record in per_stage_dicts), default=0),
        "main_slot_count": max((record["main_slot_count"]
                                for record in per_stage_dicts), default=0),
        "global_main_page_bytes": global_main_page,
        "global_main_slot_count": global_main_slots,
        "small_page_bytes": small_page,
        "small_slot_count": small_slots,
        "small_class_bytes": sum(r["small_class_bytes"] for r in per_stage_dicts),
        "main_class_bytes": sum(r["main_class_bytes"] for r in per_stage_dicts),
        "small_class_baseline_bytes": sum(small_baseline),
        "small_class_baseline_per_stage": small_baseline,
        "main_class_baseline_per_stage": list(main_baseline),
        "global_recurrent_group_sizes": [len(group)
                                         for group in global_groups],
        "projected_group_sizes_per_stage": projected_sizes,
        "physical_pool_bytes": sum(record["physical_pool_bytes"]
                                   for record in per_stage_dicts),
        "bare_slot_bytes": sum(record["bare_slot_bytes"]
                               for record in per_stage_dicts),
        "layer_sum_bytes": sum(record["layer_sum_bytes"]
                               for record in per_stage_dicts),
        "page_padding_bytes": sum(record["page_padding_bytes"]
                                  for record in per_stage_dicts),
        "attention_uplift_bytes": sum(record["attention_uplift_bytes"]
                                      for record in per_stage_dicts),
        "slot_sharing_saving_bytes": sum(record["slot_sharing_saving_bytes"]
                                         for record in per_stage_dicts),
        "requires_padded_page_layout": any(
            record["requires_padded_page_layout"]
            for record in per_stage_dicts),
        "baseline_replaced_per_stage": baselines,
        "modeled_per_stage": modeled,
        "delta_per_stage": deltas,
        "replacement_delta_bytes": sum(deltas),
        "shared_pool_family": model_type in SHARED_POOL_MODEL_TYPES,
        "reconciliation": "signed_source_allocation" if glm_family else "conservative",
        "missing_total_bytes": sum(modeled) - sum(
            base + delta for base, delta in zip(baselines, deltas)),
        "mamba_cache_mode": options.mamba_cache_mode,
        "state_page_stride_bytes": stride,
        "raw_state_page_bytes": raw_state_page,
        "state_slots_per_sequence": state_slots,
        "state_slots_total": state_slots * batch,
        "state_pages_per_sequence": state_slots,
        "state_pages_total": state_slots * batch,
    }
    return aggregate, modeled, deltas


@typed_inputs
def _pd_supported(options: HybridKvStateOptions) -> bool:
    from mindie_sim.kv_pd_descriptors import supports_component_major_mla_pd
    return supports_component_major_mla_pd(
        kv_connector=options.kv_connector,
        kv_connector_module_path=options.kv_connector_module_path,
        pd_disaggregated=options.pd_disaggregated)


__all__ = ["BOUNDED_REPLAY_ARCHITECTURES", "SHARED_POOL_MODEL_TYPES",
           "_attention_row", "_dspark_block", "_model_identity", "_nz_block",
           "_pd_state", "_pd_supported", "_pool_block"]
