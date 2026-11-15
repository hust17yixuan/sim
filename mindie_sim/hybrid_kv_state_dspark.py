"""DSpark draft-KV, prefix-companion and stage-ownership blocks.

[Role]
  Build the DSpark sub-record of one hybrid evaluation: the resolved cache
  grouping, the bounded draft SWA extent, the non-causal DCP block, the joint
  target/draft prefix identity and the per-stage transient auxiliary capture
  workspace.

[Invariants]
  * The joint identity serializes both complete resolved model configs and the
    draft's OWN activation dtype/quantization/name/revisions.
  * Prefix extents use the actual workload prompt (or explicit hashes), never
    the sequence capacity; the plan is an immutable validated record.
  * Active SWA bytes and the replaced draft-KV baseline are both batch-scaled.
  * Aux capture/transport scratch is transient and attributed to the PP stages
    that produce each boundary state plus the final consumer.
  * An explicitly active DSpark is priced against an effective runtime that has
    DSpark enabled, so draft weights are never omitted.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.schema.case import Case
from mindie_sim.schema.model import ModelSpec
from mindie_sim.catalog.types import DraftSpec
from mindie_sim.dspark_dcp_cost import non_causal_block_drafter_cost
from mindie_sim.dspark_kv_physics import (
    BOUNDED_REPLAY_ARCHITECTURES, KvSpecSignature, dspark_swa_footprint,
    draft_row_bytes_per_token, kimi_k3_dspark_grouping,
    swa_bounded_replay_policy)
from mindie_sim.dspark_prefix_cache import (
    DSparkPrefixPage, ModelIdentity, dspark_prefix_keys,
    dspark_prefix_page_bytes, dspark_prefix_plan)
from mindie_sim.hybrid_kv_state_options import (
    HybridKvStateOptions, _architecture, _attention_row, _facts,
    _draft_local_kv_heads, _draft_row_elements, _draft_spec, _layer_stage,
    _machine_limits, _stage_partition)
from mindie_sim.hybrid_kv_state_pool import baseline_pricing, stage_deltas
from mindie_sim.kv_pool_geometry import pool_blocks
from mindie_sim.linear_state_physics import linear_layers_from_model
from mindie_sim.nr_s7_validate import strict_int, strict_text
from mindie_sim.schema.common import ModelingError, dtype_bytes


def _runtime_int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="hybrid_kv_runtime")


def _project(value: Any) -> Any:
    """Deterministic JSON projection of a resolved dataclass graph."""
    import dataclasses
    import enum
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _project(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_project(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flat_config(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a resolved config into sorted dotted ``path -> JSON`` pairs."""
    import json
    flat: dict[str, str] = {}
    projected = _project(value)
    if not isinstance(projected, dict):
        return {prefix or "value": json.dumps(projected, sort_keys=True,
                                              default=str)}
    for key, item in sorted(projected.items()):
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(_flat_config(item, path))
        else:
            flat[path] = json.dumps(item, sort_keys=True, default=str)
    return flat


def _quantization_identity(model: Any) -> str:
    """The resolved weight recipe plus the declared external quantization block."""
    import hashlib
    import json
    quantization = _facts(model).get("quantization", {})
    if not isinstance(quantization, Mapping):
        raise ModelingError("hybrid_kv_quantization_invalid")
    name = strict_text("quantization_name", quantization.get("name", "none"))
    metadata = dict(getattr(model, "metadata", ()) or {})
    declared = metadata.get("declared_quantization")
    if declared:
        digest = hashlib.sha256(json.dumps(
            _project(declared), sort_keys=True, default=str).encode()).hexdigest()
        name = f"{name}|{digest}"
    return name


def _metadata_text(model: Any, key: str, default: str = "") -> str:
    value = dict(getattr(model, "metadata", ()) or {}).get(key, default)
    return value if type(value) is str else default


@typed_inputs
def _model_identity(model: ModelSpec, draft: DraftSpec | None = None) -> ModelIdentity:
    """One complete model identity: target or the draft's OWN resolved record.

    The source hashes each model's full HF config, dtype, quantization, name and
    revisions.  The projection below serializes every resolved architectural,
    numerical, dtype and rollup field instead of a handpicked subset, so an
    epsilon or a per-layer state dtype changes the digest.
    """
    if draft is None:
        config = _flat_config(model)
        config.pop("auxiliary", None)
        return ModelIdentity(
            name=strict_text("model_name", model.name),
            activation_dtype=model.activation_dtype,
            kv_dtype=model.kv_dtype,
            revision=_metadata_text(model, "revision"),
            code_revision=_metadata_text(model, "code_revision"),
            quantization=_quantization_identity(model),
            config=tuple(sorted(config.items())))
    auxiliary = getattr(model, "auxiliary", None)
    draft_name = str(getattr(auxiliary, "external_draft_key", "")
                     or f"{getattr(model, 'name', 'model')}:{getattr(draft, 'kind', 'draft')}")
    config = _flat_config(draft, prefix="draft")
    config["draft.source_name"] = _flat_config(
        getattr(auxiliary, "notes", ()), prefix="notes").get("notes", "[]")
    return ModelIdentity(
        name=draft_name,
        activation_dtype=draft.activation_dtype,
        kv_dtype=model.kv_dtype,
        revision=_metadata_text(model, "draft_revision"),
        code_revision=_metadata_text(model, "draft_code_revision"),
        quantization=str(getattr(draft, "weight_category", "")
                         or _quantization_identity(model)),
        config=tuple(sorted(config.items())))


@typed_inputs
def _dspark_block(parsed: Case, options: HybridKvStateOptions, *,
                  active: bool, tokens: int, prompt_tokens: int,
                  available_hashes: int, batch: int, block_size: int,
                  kv_elem: float, errors: list[str],
                  warnings: list[str]) -> tuple[dict[str, Any], list[float], float]:
    from mindie_sim.nr_s7_validate import positive_rate
    for name, value in (("tokens", tokens), ("batch", batch), ("block_size", block_size)):
        _runtime_int(name, value, minimum=1)
    _runtime_int("prompt_tokens", prompt_tokens)
    _runtime_int("available_hashes", available_hashes)
    positive_rate("kv_elem", kv_elem)
    model, runtime = parsed.model, parsed.runtime
    draft = _draft_spec(model)
    partition = _stage_partition(runtime, model)
    stages = len(partition)
    deltas = [0.0] * stages
    scratch = 0.0
    if draft is None:
        return ({"present": False, "active": False, "grouping": None,
                 "swa": None, "swa_replay": None, "target_swa_replay": None,
                 "dcp_block": None, "prefix": None}, deltas, 0.0)
    mla = getattr(model, "mla", None)
    draft_row_full = _draft_row_elements(draft)
    draft_page = int(block_size * draft_row_full * kv_elem)
    attn_row, attn_kind = _attention_row(model)
    target_page = int(block_size * attn_row * kv_elem)
    tp = int(runtime.parallel.tp)
    kda_shard = tp if tp != 1 else int(runtime.parallel.pcp)
    target_head_dim = (_runtime_int(
        "v_head_dim", getattr(mla, "v_head_dim", 0) if mla else 0,
        minimum=1) if attn_kind == "mla" else _runtime_int(
            "head_dim", model.head_dim, minimum=1))
    target_kv_heads = (1 if attn_kind == "mla" else _runtime_int(
        "num_kv_heads", model.num_kv_heads, minimum=1))
    # Authentic Kimi-K3 layer-name prefixes and fully resolved attention
    # heads/dtypes.  A deliberately different draft geometry is reported as
    # the labeled analytical fallback, never as the source's uniform grouping.
    specs = [KvSpecSignature(
        name=f"language_model.model.layers.{layer.index}.cache",
        kind="mla_attention" if attn_kind == "mla" else "full_attention",
        block_size=block_size, page_size_bytes=target_page, scope="target",
        num_heads=_runtime_int("num_heads", model.num_heads, minimum=1),
        num_kv_heads=target_kv_heads, head_dim=target_head_dim,
        head_dtype=str(model.activation_dtype))
        for layer in (getattr(model, "layer_plans", ()) or ())
        if layer.attention in {"mla", "gqa"}]
    specs += [KvSpecSignature(
        name=f"model.layers.{index}.cache",
        kind=("mla_attention" if getattr(draft, "attention_kind", "") == "mla"
              else "full_attention"),
        block_size=block_size, page_size_bytes=draft_page, scope="draft",
        noncausal_query_block=bool(draft.noncausal_query_block),
        num_heads=_runtime_int("draft_num_heads", draft.num_heads, minimum=1),
        num_kv_heads=_runtime_int("draft_num_kv_heads", draft.num_kv_heads,
                                  minimum=1),
        head_dim=_runtime_int("draft_head_dim", draft.head_dim, minimum=1),
        head_dtype=str(getattr(draft, "activation_dtype", "bf16")))
        for index in range(_runtime_int("draft_layers", draft.layers, minimum=1))]
    spec_q = options.num_spec + 1
    for layer in linear_layers_from_model(model, tp=kda_shard):
        page = layer.page_geometry(kda_shard, spec_q).page_bytes
        specs.append(KvSpecSignature(
            name=f"language_model.model.layers.{layer.index}.cache",
            kind="mamba", block_size=block_size,
            page_size_bytes=int(page), scope="target", mamba_type=layer.kind,
            state_shapes=((layer.local_heads(kda_shard), layer.key_dim,
                           layer.value_dim),
                          (layer.conv_width, layer.conv_rows(spec_q))),
            state_dtypes=(layer.recurrent_dtype, layer.conv_dtype),
            speculative_slots=options.num_spec))
    grouping = kimi_k3_dspark_grouping(specs, strict=False,
                                       source_strict=True)
    record_mode = grouping.mode
    if grouping.fallback_reason:
        warnings.append(
            f"kimi_k3_dspark_grouping:{grouping.mode}:"
            f"{grouping.fallback_reason}")
    window = _runtime_int("draft_window",
                          getattr(draft, "window", 0) or 0)
    if window <= 0:
        window = _runtime_int("target_window",
                              getattr(mla, "sliding_window", 0) if mla else 0)
    record: dict[str, Any] = {
        "present": True, "active": active,
        "grouping": grouping.to_dict(),
        "grouping_mode": record_mode,
        "window": window, "swa": None, "swa_replay": None,
        "target_swa_replay": None, "dcp_block": None, "prefix": None,
        "aux_scratch_per_stage": [0.0] * stages,
        "draft_kv_delta_per_stage": list(deltas)}
    if not active:
        warnings.append("dspark_configured_but_inactive")
        return record, deltas, scratch
    local_heads = _draft_local_kv_heads(draft, tp)
    head_dim = _runtime_int("draft_head_dim", draft.head_dim, minimum=1)
    draft_layers = _runtime_int("draft_layers", draft.layers, minimum=1)
    if window > 0:
        mla_row = None
        if getattr(draft, "attention_kind", "") == "mla":
            mla_row = _runtime_int("draft_mla_row", draft_row_full, minimum=1)
        footprint = dspark_swa_footprint(
            sequence_tokens=tokens, swa_window_tokens=window,
            extra_retained_tokens=options.extra_retained_tokens,
            kv_block_size=block_size, layers=draft_layers,
            kv_heads=_runtime_int("draft_kv_heads", draft.num_kv_heads,
                                  minimum=1),
            head_dim=head_dim, tp=tp, dtype=model.kv_dtype,
            mla_row_elements=mla_row)
        record["swa"] = footprint.to_dict()
        baseline = baseline_pricing(parsed, tokens=tokens,
                                    active_sequences=batch).draft_kv_per_stage
        modeled = [0.0] * stages
        # The active SWA extent is per sequence; the replaced baseline is
        # priced for every active sequence, so the modeled side must match.
        modeled[-1] = footprint.bounded_bytes * batch
        deltas = list(stage_deltas(modeled=modeled, replaced=baseline,
                                   conservative=False))
        record["draft_kv_delta_per_stage"] = deltas
        record["baseline_draft_kv_per_stage"] = list(baseline)
        record["modeled_draft_kv_bytes"] = modeled[-1]
    else:
        row = (float(draft_row_full) * kv_elem if
               getattr(draft, "attention_kind", "") == "mla"
               else draft_row_bytes_per_token(
                   kv_heads=int(draft.num_kv_heads), head_dim=head_dim,
                   tp=tp, dtype=model.kv_dtype))
        modeled = [0.0] * stages
        modeled[-1] = float(batch * tokens) * draft_layers * row
        baseline = baseline_pricing(parsed, tokens=tokens,
                                    active_sequences=batch).draft_kv_per_stage
        deltas = list(stage_deltas(modeled=modeled, replaced=baseline,
                                   conservative=False))
        record["draft_kv_delta_per_stage"] = deltas
        record["modeled_draft_kv_bytes"] = modeled[-1]
    if window > 0:
        record["swa_replay"] = swa_bounded_replay_policy(
            architecture=_architecture(model),
            swa_bounded_replay=bool(options.swa_bounded_replay),
            prefix_hit=options.prefix_hit_tokens, swa_window=window,
            extra_retained=options.extra_retained_tokens,
            draft_cache_class="DeepseekV41DSparkSWACache",
            is_draft_layer=True)
        record["target_swa_replay"] = swa_bounded_replay_policy(
            architecture=_architecture(model),
            swa_bounded_replay=bool(options.swa_bounded_replay
                                    and _architecture(model)
                                    in BOUNDED_REPLAY_ARCHITECTURES),
            prefix_hit=options.prefix_hit_tokens, swa_window=window,
            is_draft_layer=False)
        warnings.extend(record["swa_replay"]["diagnostics"])
    if int(runtime.parallel.dcp) > 1 and bool(draft.noncausal_query_block):
        # MLA exchanges an already-projected latent query: ``kv_lora_rank``
        # NoPE plus its own RoPE fragment, with the draft's own head counts --
        # never the target's original ``qk_nope_head_dim``.
        draft_is_mla = getattr(draft, "attention_kind", "") == "mla"
        draft_latent = (_runtime_int(
            "draft_kv_lora_rank", getattr(draft, "kv_lora_rank", 0),
            minimum=1) if draft_is_mla else _runtime_int(
                "draft_kv_width", draft.kv_width, minimum=1))
        draft_rope = (_runtime_int(
            "draft_qk_rope_head_dim",
            getattr(draft, "qk_rope_head_dim", 0)) if draft_is_mla else 0)
        record["dcp_block"] = non_causal_block_drafter_cost(
            draft_block_size=max(1, options.dspark_block_size
                                 or int(draft.block_size)),
            context_tokens=tokens, dcp=int(runtime.parallel.dcp),
            pcp=int(runtime.parallel.pcp), tp=tp,
            num_q_heads=_runtime_int("draft_num_heads", draft.num_heads,
                                     minimum=1),
            q_nope_dim=draft_latent, q_rope_dim=draft_rope,
            kv_heads=_runtime_int("draft_kv_heads", draft.num_kv_heads,
                                  minimum=1),
            head_dim=head_dim,
            latent_width=draft_latent,
            num_draft_layers=draft_layers, batch=batch, swa_window=window,
            kv_dtype=model.kv_dtype,
            q_dtype=str(getattr(draft, "activation_dtype", "bf16")),
            index_entry_bytes=4, index_length_bytes=4,
            index_entries_per_us=max(1.0, _machine_limits(parsed)[0] / 1e9),
            collective_bytes_per_s=_machine_limits(parsed)[3],
            kernel_fixed_ms=_machine_limits(parsed)[2])
    aux_ids = options.dspark_aux_layer_ids
    if aux_ids:
        record["prefix"] = _prefix_block(
            parsed, options, draft=draft, tokens=tokens,
            prompt_tokens=prompt_tokens,
            available_hashes=available_hashes, batch=batch,
            block_size=block_size, tp=tp, kv_elem=kv_elem,
            errors=errors, warnings=warnings)
        scratch = record["prefix"].get("aux_scratch_bytes", 0.0)
        record["aux_scratch_per_stage"] = list(
            record["prefix"].get("aux_scratch_per_stage",
                                 [0.0] * stages))
    return record, deltas, scratch


def _prefix_block(parsed: Any, options: HybridKvStateOptions, *, draft: Any,
                  tokens: int, prompt_tokens: int, available_hashes: int,
                  batch: int, block_size: int, tp: int,
                  kv_elem: float, errors: list[str],
                  warnings: list[str]) -> dict[str, Any]:
    model = parsed.model
    aux_ids = options.dspark_aux_layer_ids
    layers = int(model.layers)
    invalid_aux = [layer_id for layer_id in aux_ids
                   if layer_id > layers]
    for layer_id in invalid_aux:
        errors.append(f"dspark_aux_layer_out_of_range:{layer_id}")
    target_identity = _model_identity(model)
    draft_identity = _model_identity(model, draft=draft)
    partition = _stage_partition(parsed.runtime, model)
    num_blocks = max(
        pool_blocks(tokens=tokens, block_size=block_size, batch=batch,
                    cp=int(parsed.runtime.parallel.cp)),
        -(-int(prompt_tokens) // block_size))
    spec_q = options.num_spec + 1
    draft_layers = _runtime_int("draft_layers", draft.layers, minimum=1)
    if getattr(draft, "attention_kind", "") == "mla":
        row_bytes = float(_draft_row_elements(draft)) * kv_elem
    else:
        row_bytes = draft_row_bytes_per_token(
            kv_heads=int(draft.num_kv_heads), head_dim=int(draft.head_dim),
            tp=tp, dtype=model.kv_dtype)
    pages = [DSparkPrefixPage(
        name=f"draft_{index}", group_id=index, shape0=num_blocks,
        shape1=block_size, numel=int(block_size * row_bytes),
        element_size=1, stride0_elements=int(block_size * row_bytes),
        row_bytes=int(row_bytes)) for index in range(draft_layers)]
    geometry = dspark_prefix_page_bytes(
        pages=pages, num_blocks=num_blocks, block_size=block_size,
        hash_block_size=block_size,
        group_ids=tuple(range(draft_layers)),
        block_sizes=(block_size,) * draft_layers,
        draft_group_ids=tuple(range(draft_layers)),
        require_hash_equals_block=True)
    keys = dspark_prefix_keys(
        target=target_identity, draft=draft_identity, aux_layers=aux_ids,
        tp_size=tp, pp_size=int(len(partition)), block_size=block_size,
        prefix_match_unit=block_size,
        draft_width=max(1, options.dspark_block_size
                        or int(draft.block_size)),
        dcp_size=int(parsed.runtime.parallel.dcp),
        include_dcp_in_key=False)
    try:
        storage = dspark_prefix_plan(
            keys=keys, page_geometry=geometry, aux_layer_ids=aux_ids,
            target_hidden_size=int(model.hidden_size),
            target_dtype=str(model.activation_dtype),
            tokens=prompt_tokens,
            prompt_tokens=prompt_tokens,
            block_hash_count=available_hashes,
            metadata_bytes_per_object=64, aux_scratch_tokens=batch * spec_q,
            exact=False)
    except ModelingError as error:
        errors.append(str(error))
        return {"identity": keys.identity, "plan": None,
                "transfer_bytes_at_hit": 0.0}
    transfer_bytes = 0.0
    if options.prefix_hit_tokens:
        try:
            transfer = dspark_prefix_plan(
                keys=keys, page_geometry=geometry, aux_layer_ids=aux_ids,
                target_hidden_size=int(model.hidden_size),
                target_dtype=str(model.activation_dtype),
                tokens=options.prefix_hit_tokens,
                prompt_tokens=prompt_tokens,
                block_hash_count=available_hashes,
                metadata_bytes_per_object=64, aux_scratch_tokens=0,
                exact=True)
            transfer_bytes = transfer.per_rank_transfer_bytes
        except ModelingError as error:
            text = str(error)
            if ("hit_exceeds_prompt" in text
                    or "missing_block_hashes" in text):
                errors.append(text)
            else:
                warnings.append("dspark_prefix_transfer_rejected:" + text)
    payload = storage.to_dict()
    # ``aux_hidden_state`` capture/transport is live on the PP stages that
    # produce each boundary state; the final rank also receives every relayed
    # state.  It is transient workspace, never a persistent companion page.
    spec_q = options.num_spec + 1
    per_layer_scratch = (float(batch * spec_q * int(model.hidden_size))
                         * dtype_bytes(model.activation_dtype))
    # Capture boundaries are inclusive over ``0 <= id <= num_layers``.  A
    # rejected configuration allocates no scratch: a partially priced invalid
    # boundary would understate the transient workspace.
    valid_aux = ([] if invalid_aux else [
        layer_id for layer_id in aux_ids if layer_id <= layers])
    counts = [0] * len(partition)
    for layer_id in valid_aux:
        if layer_id == 0:
            stage = 0
        else:
            stage = _layer_stage(partition, min(layer_id - 1, layers - 1))
        counts[stage] += 1
    aux_per_stage = [count * per_layer_scratch for count in counts]
    if aux_per_stage:
        aux_per_stage[-1] = float(len(valid_aux)) * per_layer_scratch
    payload.update({
        "aux_scratch_bytes": float(len(valid_aux)) * per_layer_scratch,
        "aux_scratch_per_stage": aux_per_stage,
        "prefix_hit_tokens": options.prefix_hit_tokens,
        "transfer_bytes_at_hit": transfer_bytes,
        "per_rank_storage_bytes": storage.per_rank_storage_bytes,
        "aggregate_storage_bytes": storage.aggregate_storage_bytes,
        "transfer_scope": "per_rank (TP peer objects aggregate by tp_size)",
    })
    return payload



__all__ = ["_dspark_block", "_model_identity"]
