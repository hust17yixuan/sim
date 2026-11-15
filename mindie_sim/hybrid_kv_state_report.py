"""Deterministic hybrid KV/state evaluation and public report record.

[Role]
  Compose the three physical models (KV layout, DSpark coupling, linear-state
  rollback/replay) with the existing memory, decode and xPyD ledgers, returning
  one deterministic public record.

[Invariants]
  * The evaluation is a pure function of declared inputs: no clock, host,
    environment or filesystem value enters a number or the summary.
  * Sparse-SFA, MLA/GQA, MLAPO eligibility and the PD role are derived from
    catalog structure and the connector contract; an option override that
    contradicts the model ground truth is rejected.
  * Pool and DSpark deltas are per-stage and applied net-once: a baseline
    state/attention page (including a declared cache-page group) is replaced,
    never charged twice.
  * DSpark KV/companion modeling activates only when speculation is active;
    the bounded draft SWA extent replaces a full-sequence draft allocation
    once, and aux scratch is a separate transient workspace.
  * ``state_page_stride_bytes`` is propagated only to the RecoverSSM side of
    the counterfactual comparison.
  * One-time prefix replay is reported separately from steady-state decode.
  * ``json.dumps(..., allow_nan=False)`` succeeds for every returned record.
  * A rejected topology yields no fabricated layout/state numbers; strict mode
    raises with the diagnostic codes and non-strict mode returns ``rejected``.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] No filesystem, network, environment or global mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs, _validate_nested
from mindie_sim.schema.case import Case
from mindie_sim.hybrid_kv_state_blocks import (
    BOUNDED_REPLAY_ARCHITECTURES, SHARED_POOL_MODEL_TYPES, _dspark_block,
    _nz_block, _pd_supported, _pool_block, _typed_mla_planes)
from mindie_sim.hybrid_kv_state_mcp import (
    MCP_MAX_TOKENS, hybrid_kv_state_mcp_envelope, summary_token_count)
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, _architecture, _attention_layer_count,
    _attention_row, _draft_spec, _json_safe, _linear_layers_for_stage,
    _machine_limits, _model_type, _stage_partition, model_is_sparse_sfa,
    model_mlapo_eligible, pd_role, projection_quant_class)
from mindie_sim.hybrid_kv_state_pool import baseline_pricing
from mindie_sim.kv_layout_physics import component_major_pd_comparison
from mindie_sim.kv_pool_geometry import pool_blocks
from mindie_sim.linear_state_physics import (
    LinearLayer, RecoverSsmConfig, compare_linear_state_memory,
    linear_layers_from_model, linear_state_memory, validate_glm_kda_execution,
    validate_recoverssm)
from mindie_sim.linear_state_prefix_replay import (
    prefix_replay_cost, prefix_replay_plan)
from mindie_sim.linear_state_replay import precopy_cost, verify_commit_cost
from mindie_sim.recoverssm_launch_geometry import DEFAULT_UB_BYTES
from mindie_sim.memory_breakdown import (analyze_memory_breakdown,
                                         apply_state_pool_delta)
from mindie_sim.nr_s7_validate import strict_int
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError, dtype_bytes

SHARED_POOL_MODEL_TYPES = frozenset({
    "glm5_next", "glm5_next_text", "deepseek_v41_text", "qwen4_exp_text"})
BOUNDED_REPLAY_ARCHITECTURES = frozenset({
    "DeepseekV41ForCausalLM", "DeepseekV41ForConditionalGeneration"})


def _runtime_int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="hybrid_kv_runtime")


def _options_tokens(parsed: Any, options: HybridKvStateOptions) -> tuple[int, int]:
    if not parsed.workload.requests:
        raise ModelingError("hybrid_kv_workload_requests_empty")
    runtime = parsed.runtime
    batch = (options.batch if options.batch is not None
             else _runtime_int("max_num_seqs", runtime.max_num_seqs, minimum=1))
    if options.sequence_tokens is not None:
        tokens = options.sequence_tokens
    else:
        request = parsed.workload.requests[0]
        tokens = min(_runtime_int("max_model_len", runtime.max_model_len,
                                  minimum=1),
                     _runtime_int("prompt_tokens", request.prompt_tokens)
                     + _runtime_int("output_tokens", request.output_tokens))
    return batch, tokens


def _commit_hardware(parsed: Any, options: HybridKvStateOptions) -> tuple[bool, int, int]:
    """``(is_a5, ub_bytes, conv_dim_stride)`` for the commit helper.

    The public report must pass the same device and registered-conv facts the
    launch-geometry helper consumes; otherwise an A5/DAV_3510 report keeps the
    A3 value tile and a strided 256-channel conv keeps the 4096 tile.
    """
    dies = tuple(parsed.machine.dies)
    arch = str(getattr(dies[0], "arch_family", "") or "").strip().upper()
    is_a5 = arch in {"A5", "DAV_3510"}
    buffers = dict(getattr(dies[0], "buffers", ()) or ())
    ub_bytes = strict_int("ub_bytes", buffers.get("ub", DEFAULT_UB_BYTES),
                          minimum=1, reason="hybrid_kv_runtime")
    facts = dict(getattr(parsed.model, "catalog_facts", ()) or ())
    stride_value = facts.get("conv_dim_stride", options.conv_dim_stride)
    conv_stride = strict_int("conv_dim_stride", stride_value, minimum=1,
                             reason="hybrid_kv_runtime")
    return is_a5, ub_bytes, conv_stride


def _linear_block(parsed: Any, options: HybridKvStateOptions,
                  layers: tuple[LinearLayer, ...], *, tokens: int, batch: int,
                  block_size: int, hbm: float, flops: float, launch: float,
                  errors: list[str], warnings: list[str]) -> tuple[dict[str, Any],
                                                                   list[float],
                                                                   float]:
    model, runtime = parsed.model, parsed.runtime
    parallel = runtime.parallel
    partition = _stage_partition(runtime, model)
    linear: dict[str, Any] = {"layer_count": len(layers)}
    state_deltas = [0.0] * len(partition)
    state_increment = 0.0
    config = RecoverSsmConfig(
        use_replayssm=options.use_replayssm,
        speculative_k=options.num_spec,
        enable_prefix_caching=options.enable_prefix_caching,
        mamba_cache_mode=options.mamba_cache_mode,
        enable_stochastic_rounding=options.enable_stochastic_rounding,
        pp=int(parallel.pp), dcp=int(parallel.dcp), pcp=int(parallel.pcp),
        tp=int(parallel.tp), kv_connector=options.kv_connector,
        kv_role=options.kv_role, model_type=_model_type(model),
        mamba_block_size=options.mamba_block_size or block_size,
        kv_block_size=block_size, swa_bounded_replay=options.swa_bounded_replay)
    validation = validate_recoverssm(config, strict=False)
    linear["config"] = config.to_dict() if hasattr(config, "to_dict") else {
        "use_replayssm": config.use_replayssm,
        "speculative_k": config.speculative_k,
        "mamba_cache_mode": config.mamba_cache_mode}
    linear["validation"] = validation.to_dict()
    errors.extend(validation.codes)
    # GLM KDA execution is a model contract: every executing KDA layer is
    # validated in both the standard and RecoverSSM modes.  Generic non-GLM
    # linear-state arithmetic stays open to arbitrary layers.
    glm_execution = None
    if layers and _model_type(model) in {"glm5_next", "glm5_next_text"}:
        executions = [validate_glm_kda_execution(
            kind=layer.kind, activation_dtype=layer.activation_dtype,
            key_dim=layer.key_dim, value_dim=layer.value_dim,
            conv_kernel=layer.conv_kernel,
            speculative_k=options.num_spec,
            cache_mode=options.mamba_cache_mode,
            block_size=block_size,
            kda_gate_lower_bound=layer.kda_gate_lower_bound,
            mamba_block_size=config.block_size) for layer in layers]
        codes = list(dict.fromkeys(
            code for execution in executions for code in execution.codes))
        glm_execution = {
            "ok": not codes, "codes": codes,
            "layers": [execution.to_dict() for execution in executions],
            "layer_count": len(executions),
        }
        linear["glm_execution"] = glm_execution
        errors.extend("glm_kda_execution:" + code for code in codes)
    if not layers or not validation.ok:
        linear.update({"memory": None, "comparison": None, "step_cost": None,
                       "precopy": None, "replay": None, "replay_cost": None,
                       "stage_state_bytes": [0.0] * len(partition)})
        return linear, state_deltas, 0.0
    head_shard = config.kda_head_shard
    stride = options.state_page_stride_bytes if config.use_replayssm else None
    chosen = linear_state_memory(
        config, layers, sequence_tokens=tokens, active_sequences=batch,
        page_stride_bytes=stride)
    comparison = compare_linear_state_memory(
        config, layers, sequence_tokens=tokens, active_sequences=batch,
        page_stride_bytes=stride)
    linear["memory"] = chosen.to_dict()
    linear["comparison"] = comparison
    is_a5, ub_bytes, conv_stride = _commit_hardware(parsed, options)
    linear["commit_hardware"] = {"is_a5": is_a5, "ub_bytes": ub_bytes,
                                 "conv_dim_stride": conv_stride}
    linear["step_cost"] = verify_commit_cost(
        config, layers, batch=batch, sequence_tokens=tokens,
        hbm_bytes_per_s=hbm, compute_flops_per_s=flops if flops > 0 else None,
        kernel_launch_ms=launch, accepted_tokens=options.accepted_tokens,
        accept_rate=options.accept_rate,
        program_dispatch_us=options.program_dispatch_us,
        ub_bytes=ub_bytes, is_a5=is_a5, conv_dim_stride=conv_stride)
    if config.mamba_cache_mode == "align":
        linear["precopy"] = precopy_cost(
            config, layers, sequence_start=tokens,
            sequence_end=tokens + config.spec_query_len, batch=batch,
            hbm_bytes_per_s=hbm, kernel_launch_ms=launch,
            layerwise=options.align_layerwise_copy,
            overlap_fraction=(options.align_copy_overlap
                              if options.align_layerwise_copy else 0.0))
    else:
        linear["precopy"] = None
    target_window = (max(0, int(getattr(model.mla, "sliding_window", 0) or 0))
                     if getattr(model, "mla", None) else 0)
    try:
        replay = prefix_replay_plan(
            config, layers, isl=tokens, prefix_hit=options.prefix_hit_tokens,
            swa_window=target_window,
            swa_bounded_replay=bool(options.swa_bounded_replay
                                    and target_window > 0),
            swa_architecture_supported=_architecture(model)
            in BOUNDED_REPLAY_ARCHITECTURES, draft_swa=False)
        linear["replay"] = replay
        linear["replay_cost"] = prefix_replay_cost(
            config, layers, plan=replay, hbm_bytes_per_s=hbm,
            compute_flops_per_s=flops if flops > 0 else None)
    except ModelingError as error:
        errors.append(str(error))
        linear["replay"] = None
        linear["replay_cost"] = None
    stride_bytes = (options.state_page_stride_bytes
                    if config.use_replayssm else None)
    linear["stage_state_bytes"] = [
        sum(max(layer.state_bytes(head_shard)
                + layer.conv_bytes(head_shard,
                                   spec_query_len=config.spec_query_len),
                stride_bytes or 0)
            for layer in _linear_layers_for_stage(model, layers, stage,
                                                 partition))
        for stage in range(len(partition))]
    modeled = [0.0] * len(partition)
    for stage in range(len(partition)):
        for layer in _linear_layers_for_stage(model, layers, stage, partition):
            page = layer.page_geometry(head_shard,
                                       config.spec_query_len)
            if config.use_replayssm:
                modeled[stage] += (float(chosen.pages_per_sequence)
                                   * max(page.page_bytes, stride_bytes or 0))
            else:
                modeled[stage] += float(config.spec_query_len) * (
                    page.state_conv_bytes)
    linear["modeled_state_per_stage"] = modeled
    linear["ep_independent"] = True
    linear["replication"] = "dp_replicated"
    linear["baseline_state_per_stage"] = list(
        baseline_pricing(parsed, tokens=tokens,
                         active_sequences=batch).replaced_per_stage)
    return linear, state_deltas, state_increment


_RAW_RUNTIME_INTS = ("max_num_seqs", "max_model_len", "kv_block_size", "tp",
                     "pp", "dp", "ep", "pcp", "cp")


def _validate_raw_case(case: Any) -> None:
    """Reject non-integral runtime fields before ``parse_case`` coerces them."""
    if not isinstance(case, Mapping):
        return
    runtime = case.get("runtime")
    if not isinstance(runtime, Mapping):
        return
    for key in _RAW_RUNTIME_INTS:
        if key in runtime:
            strict_int(f"runtime_{key}", runtime[key], minimum=1,
                       reason="hybrid_kv_runtime")


@typed_inputs
def evaluate_hybrid_kv_state(case: Case | Mapping[str, Any],
                             options: HybridKvStateOptions | None = None) -> dict[str, Any]:
    """Return one deterministic hybrid layout/state/DSpark evaluation record."""
    _validate_raw_case(case)
    parsed = parse_case(case)
    _validate_nested(parsed, "case")
    if options is None:
        options = HybridKvStateOptions()
    elif not isinstance(options, HybridKvStateOptions):
        raise ModelingError("hybrid_kv_options_invalid")
    model, runtime = parsed.model, parsed.runtime
    parallel = runtime.parallel
    partition = _stage_partition(runtime, model)
    stages = len(partition)
    model_type = _model_type(model)
    architecture = _architecture(model)
    attn_row, attn_kind = _attention_row(model)
    draft = _draft_spec(model)
    draft_aux = tuple(getattr(draft, "target_layer_ids", ()) or ()) if draft else ()
    options = options.with_dspark_aux(draft_aux)
    batch, tokens = _options_tokens(parsed, options)
    block_size = _runtime_int("kv_block_size", runtime.kv_block_size, minimum=1)
    hbm, flops, launch, collective = _machine_limits(parsed)
    hbm = options.hbm_bytes_per_s if options.hbm_bytes_per_s is not None else hbm
    flops = (options.compute_flops_per_s
             if options.compute_flops_per_s is not None else flops)
    launch = (options.kernel_launch_ms
              if options.kernel_launch_ms is not None else launch)
    kv_elem = dtype_bytes(model.kv_dtype)
    tp = int(parallel.tp)
    layers = linear_layers_from_model(model, tp=tp)
    derived_sparse = model_is_sparse_sfa(model)
    hardware_arch = str(getattr(parsed.machine.dies[0], "arch_family", "") or "")
    consumer, _producer = pd_role(options)
    decode_instance = bool(options.pd_disaggregated and consumer)
    derived_mlapo = model_mlapo_eligible(
        model, hardware_arch=hardware_arch,
        enable_mlapo=options.enable_mlapo,
        decode_instance=decode_instance)
    if (options.is_sparse_sfa is not None
            and options.is_sparse_sfa != derived_sparse):
        raise ModelingError(
            "hybrid_kv_sparse_sfa_override_contradicts_model")
    if (options.mlapo_eligible is not None
            and options.mlapo_eligible != derived_mlapo):
        raise ModelingError(
            "hybrid_kv_mlapo_override_contradicts_model")
    sparse = derived_sparse
    mlapo = derived_mlapo
    errors: list[str] = []
    warnings: list[str] = []
    if sparse and mlapo:
        raise ModelingError("hybrid_kv_mlapo_contradicts_sparse_sfa")
    if options.dspark_active is False and runtime.dspark:
        raise ModelingError("hybrid_kv_dspark_override_contradicts_runtime")
    dspark_active = (options.dspark_active if options.dspark_active is not None
                     else bool(runtime.dspark))
    # An explicitly active DSpark must price the loaded draft: synthesize the
    # runtime flag on the effective Case used by the baseline ledger.
    effective = parsed
    if dspark_active and draft is not None and not runtime.dspark:
        effective = replace(
            parsed, runtime=replace(parsed.runtime, dspark=True))
    request = parsed.workload.requests[0]
    prompt_tokens = _runtime_int("prompt_tokens", request.prompt_tokens)
    available_hashes = (
        options.prefix_available_hashes
        if options.prefix_available_hashes is not None
        else prompt_tokens // block_size)
    # A configured transfer-staging reserve is validated before any arithmetic
    # and removed from the raw policy mapping: it is a declared staging fact,
    # not a ``MemoryPolicy`` constructor keyword.
    raw_memory = dict(runtime.memory)
    raw_staging = raw_memory.get("kv_transfer_staging_bytes", 0)
    if raw_staging is None:
        raw_staging = 0
    staging_bytes = strict_int("kv_transfer_staging_bytes", raw_staging,
                               minimum=0, reason="hybrid_kv_runtime")
    analysis = effective
    if "kv_transfer_staging_bytes" in raw_memory:
        cleaned = {key: value for key, value in raw_memory.items()
                   if key != "kv_transfer_staging_bytes"}
        analysis = replace(
            effective, runtime=replace(effective.runtime, memory=cleaned))
    layout, staging = _nz_block(
        parsed, options, attn_kind=attn_kind, attn_row=attn_row,
        block_size=block_size, batch=batch, tokens=tokens, sparse=sparse,
        mlapo=mlapo, hbm=hbm, launch_ms=launch,
        staging_bytes=staging_bytes, errors=errors)
    if attn_kind == "mla":
        mla_plan = next((plan for plan in model.layer_plans
                         if plan.attention == "mla"), None)
        if mla_plan is None:
            raise ModelingError("hybrid_kv_mla_layer_plan_missing")
        nope, rope, nope_dtype, rope_dtype, scale_dim = _typed_mla_planes(
            model, mla_plan)
        nope_bytes = int(block_size * nope * dtype_bytes(nope_dtype))
        rope_bytes = int(block_size * rope * dtype_bytes(rope_dtype))
        scale_bytes = int(block_size * scale_dim * 4)
        padding = 0
        if options.pd_page_alignment > 0:
            padding = (-nope_bytes) % options.pd_page_alignment
        layout["component_major_pd"] = {
            "supported": _pd_supported(options),
            "selected": ("component_major" if _pd_supported(options)
                         else "split"),
            "scale_bytes": scale_bytes,
            "nope_dtype": nope_dtype, "rope_dtype": rope_dtype,
            **component_major_pd_comparison(
                pages=pool_blocks(tokens=tokens, block_size=block_size,
                                  batch=batch, cp=int(parallel.cp)),
                layers=_attention_layer_count(model),
                nope_bytes=nope_bytes, rope_bytes=rope_bytes,
                padding_bytes=padding, scale_bytes=scale_bytes,
                per_descriptor_setup_us=options.descriptor_setup_us)}
    else:
        layout["component_major_pd"] = None
    pool, _modeled_pool, pool_deltas = _pool_block(
        effective, options, layers, tokens=tokens, batch=batch,
        block_size=block_size, recoverssm=options.use_replayssm,
        sparse=sparse, errors=errors, warnings=warnings)
    layout["mixed_pool"] = pool
    dspark, dspark_deltas, scratch = _dspark_block(
        effective, options, active=dspark_active, tokens=tokens,
        prompt_tokens=prompt_tokens, available_hashes=available_hashes,
        batch=batch, block_size=block_size, kv_elem=kv_elem,
        errors=errors, warnings=warnings)
    linear, state_deltas, _state_increment = _linear_block(
        parsed, options, layers, tokens=tokens, batch=batch,
        block_size=block_size, hbm=hbm, flops=flops, launch=launch,
        errors=errors, warnings=warnings)
    # The grouped page-class delta already replaces attention + state pages;
    # it is the single net-once delta for that ownership domain.
    net = [pool_deltas[i] + dspark_deltas[i] for i in range(stages)]
    aux_stage = list(dspark.get("aux_scratch_per_stage") or [0.0] * stages)
    if len(aux_stage) != stages:
        raise ModelingError("hybrid_kv_aux_scratch_stage_count_mismatch")
    baseline = analyze_memory_breakdown(
        analysis, sequence_tokens=tokens, sequences=batch,
        graph_analytical_bound=options.graph_analytical_bound,
        kv_transfer_staging_per_stage_bytes=tuple(staging))
    adjusted = apply_state_pool_delta(
        baseline, per_stage_bytes=tuple(net),
        transient_per_stage_bytes=tuple(aux_stage),
        allow_negative=True)
    memory = {
        "baseline": baseline.to_dict(),
        "adjusted": adjusted.to_dict(),
        "pool_replacement_delta_bytes": sum(pool_deltas),
        "state_ownership_per_stage": linear.get("stage_state_bytes", []),
        "state_delta_per_stage": list(state_deltas),
        "pool_delta_per_stage": pool_deltas,
        "dspark_delta_per_stage": dspark_deltas,
        "aux_scratch_bytes": scratch,
        "aux_scratch_per_stage": aux_stage,
        "nz_staging_bytes": sum(staging),
        "nz_staging_per_stage": list(staging),
        "net_delta_per_stage": net,
        "total_increment_bytes": adjusted.peak_hbm_bytes
        - baseline.peak_hbm_bytes,
        "adjusted_peak_hbm_bytes": adjusted.peak_hbm_bytes,
        "adjusted_headroom_bytes": adjusted.headroom_bytes,
        "adjusted_status": adjusted.status,
        "adjusted_reason": adjusted.reason,
        "qualified": bool(adjusted.memory_qualified and adjusted.oom_feasible),
    }
    replay_ms = 0.0
    if linear.get("replay_cost"):
        replay_ms = linear["replay_cost"]["replay_ms"]
    decode = {"step_overhead_ms": 0.0, "linear_ms": 0.0,
              "dspark_block_ms": 0.0, "precopy_ms": 0.0,
              "prefix_replay_ms": replay_ms, "timing_model": "serial_estimate"}
    if linear.get("step_cost") is not None:
        decode["linear_ms"] = linear["step_cost"]["total_ms"]
        if linear.get("precopy") is not None:
            decode["precopy_ms"] = (
                linear["precopy"]["layerwise_overlapped_ms"]
                if options.align_layerwise_copy
                else linear["precopy"]["bulk_ms"])
    if dspark.get("dcp_block") is not None:
        decode["dspark_block_ms"] = dspark["dcp_block"]["total_ms"]
    decode["step_overhead_ms"] = (decode["linear_ms"] + decode["precopy_ms"]
                                  + decode["dspark_block_ms"])
    prefix = dspark.get("prefix") or {}
    transfer = {"nz_reformat": layout.get("kv_nz", {}).get("reformat"),
                "component_major_pd": layout["component_major_pd"],
                "dspark_prefix_per_rank_bytes": prefix.get(
                    "transfer_bytes_at_hit", 0.0),
                "dspark_prefix_aggregate_bytes": (
                    prefix.get("transfer_bytes_at_hit", 0.0) * tp)}
    rejected = bool(errors)
    if rejected:
        status = "rejected"
    elif memory["adjusted_status"] == "memory_infeasible":
        status = "memory_pressure"
    elif memory["adjusted_status"] == "memory_unqualified":
        status = "memory_unqualified"
    else:
        status = "qualified"
    diagnostics = list(dict.fromkeys([*errors, *warnings]))
    if options.strict and rejected:
        raise ModelingError("hybrid_kv_state_rejected:" + ",".join(
            dict.fromkeys(errors)))
    record = {
        "schema": "mindie-sim.hybrid-kv-state/1",
        "status": status,
        "diagnostics": diagnostics,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "model": {"name": model.name, "model_type": model_type,
                  "architecture": architecture, "layers": int(model.layers),
                  "hidden_size": int(model.hidden_size),
                  "attention": attn_kind, "kv_dtype": model.kv_dtype,
                  "sparse_sfa": sparse, "mlapo_eligible": mlapo,
                  "mlapo_quant": projection_quant_class(model),
                  "hardware_arch": hardware_arch,
                  "decode_instance": decode_instance},
        "parallel": {"tp": tp, "pp": int(parallel.pp),
                     "dp": int(parallel.dp), "dcp": int(parallel.dcp),
                     "pcp": int(parallel.pcp)},
        "batch": batch, "sequence_tokens": tokens,
        "block_size": block_size, "prefix_hit_tokens": options.prefix_hit_tokens,
        "layout": layout, "dspark": dspark, "linear_state": linear,
        "memory": memory, "decode": decode, "transfer": transfer,
        "limits": {"hbm_bytes_per_s": hbm, "fp32_flops_per_s": flops,
                   "kernel_launch_ms": launch,
                   "collective_bytes_per_s": collective},
        "deterministic": True,
    }
    return _json_safe(record)
