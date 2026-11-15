"""Draft generation as a program preceding target verification.

[Role] Derive proposal work from a declared draft structure.
[Invariants] Proposals are computed, verified and then committed. Draft heads
  have their own dimensions and weights; acceptance never discounts target work.
[Timeouts & Locks] Bounded by the scheduled proposal count and draft layers.
[Side-Effect Policy] Constructs operator records only.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.draft import optional_draft_spec, draft_quant
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op
from mindie_sim.schema.model import LayerPlan
from .structure import stage_map
from .implementation_selection import resolve, LoweringRegion


def _tag_draft_ops(ops, rows_by_request, *, step=0):
    """Attach the actual draft producer identity and materialized row counts.

    A target layer and a draft layer can share a numeric index.  Offload and
    state consumers must use this scope, not the layer number alone, to choose
    the producer and completion handle.
    """
    rows = tuple(sorted((str(request), int(count)) for request, count in rows_by_request.items()))
    return tuple(replace(op, metadata=tuple({
        **dict(op.metadata),
        "layer_scope": "draft",
        "producer_scope": "draft",
        "draft_step": int(step),
        "draft_rows_by_request": rows,
    }.items())) for op in ops)


def _reference_attention_plan(model, draft, index: int) -> LayerPlan | None:
    if model.auxiliary is not None and model.auxiliary.standalone_draft:
        return None
    matching = [p for p in model.layer_plans if p.attention == draft.attention_kind]
    if not matching:
        return None
    ref = None
    if index < len(draft.target_layer_ids):
        target_id = int(draft.target_layer_ids[index])
        if 0 <= target_id < len(model.layer_plans):
            candidate = model.layer_plans[target_id]
            if candidate.attention == draft.attention_kind:
                ref = candidate
    if ref is None:
        owners = [
            p for p in matching
            if (not p.indexer and p.topk_owner_layer_id < 0 and p.kv_owner_layer_id in (-1, p.index))
            or (p.owns_index_k and p.owns_topk)
        ]
        ref = owners[-1] if owners else matching[-1]
    if not ref.indexer and 0 <= ref.topk_owner_layer_id < len(model.layer_plans):
        donor = model.layer_plans[ref.topk_owner_layer_id]
        if donor.attention == draft.attention_kind and donor.indexer:
            ref = replace(
                ref, indexer=True, indexer_type=donor.indexer_type,
                owns_index_k=True, owns_topk=True,
                index_query_per_layer=donor.index_query_per_layer,
                index_heads=donor.index_heads, index_dim=donor.index_dim,
                index_topk=donor.index_topk, index_kpool=donor.index_kpool,
                index_kpool_compress=donor.index_kpool_compress,
                index_kpool_always_select_tail=donor.index_kpool_always_select_tail,
                index_input=donor.index_input,
            )
    return ref


def _draft_layer_plan(model, draft, index: int, plan_fields: dict) -> LayerPlan:
    ref = _reference_attention_plan(model, draft, index)
    explicit_ratio = (
        int(draft.draft_compress_ratios[index])
        if index < len(draft.draft_compress_ratios) else None
    )
    if explicit_ratio is not None and explicit_ratio > 0 and (
        ref is None or ref.compress_ratio != explicit_ratio
    ):
        same_ratio = [
            p for p in model.layer_plans
            if p.attention == draft.attention_kind and p.compress_ratio == explicit_ratio and p.owns_long_kv
        ]
        if same_ratio:
            ref = same_ratio[-1]
        elif ref is not None:
            ref = replace(ref, cache_kind="window_compressed", compress_ratio=explicit_ratio, owns_long_kv=True)

    mhc = int(draft.mhc_mult or (ref.hc_mult if ref and ref.hyper_connections else 0))
    mhc_iters = int(draft.mhc_sinkhorn_iters or (ref.hc_sinkhorn_iters if ref and ref.hyper_connections else 0))
    kv_elems = int(draft.kv_cache_elements or (ref.kv_cache_elements if ref else 0))
    qk_norm = bool(draft.qk_norm or (ref.qk_norm if ref else False))
    q_gate = bool(draft.q_output_gate or (ref.q_output_gate if ref else False))
    mla_gate = bool(ref.mla_output_gate) if ref else False

    swa_only = (
        explicit_ratio == 0
        or (explicit_ratio is None and ref is not None and not ref.owns_long_kv and ref.window > 0)
    )
    if swa_only:
        window = int(draft.sliding_window or (ref.window if ref else 0))
        cache_kind = ref.cache_kind if (ref and ref.cache_kind in {"dense", "window_compressed"}) else "window_compressed"
        return LayerPlan(
            index=index, attention=draft.attention_kind, architecture="swa",
            ffn="moe" if draft.num_experts else "dense",
            moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
            noncausal_attention=draft.noncausal_query_block,
            mla_variant=draft.mla_variant, mla_output_gate=mla_gate,
            kv_cache_elements=kv_elems, qk_norm=qk_norm, q_output_gate=q_gate,
            cache_kind=cache_kind, window=window, kv_owner_layer_id=-1, owns_long_kv=False,
            compress_ratio=0, compressor_kind="none", compressor_width=1, compressor_state_rows=0,
            indexer=False, indexer_type="none", owns_index_k=False, index_k_owner_layer_id=-1,
            owns_topk=False, topk_owner_layer_id=-1, index_query_per_layer=False,
            hyper_connections=bool(mhc), hc_mult=mhc, hc_sinkhorn_iters=mhc_iters,
            **plan_fields,
        )

    if ref is None:
        return LayerPlan(
            index=index, attention=draft.attention_kind, ffn="moe" if draft.num_experts else "dense",
            moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
            noncausal_attention=draft.noncausal_query_block,
            mla_variant=draft.mla_variant, kv_cache_elements=kv_elems,
            qk_norm=qk_norm, q_output_gate=q_gate,
            kv_owner_layer_id=index, owns_long_kv=True,
            hyper_connections=bool(mhc), hc_mult=mhc, hc_sinkhorn_iters=mhc_iters,
            **plan_fields,
        )

    has_indexer = bool((ref.indexer or ref.owns_topk) and ref.index_topk > 0)
    share_index_k = bool(has_indexer and draft.index_share_for_mtp_iteration and draft.attention_kind == "mla")
    indexer_type = (
        "shared" if share_index_k
        else (ref.indexer_type if ref.indexer_type in {"full", "c4"} else "full")
    ) if has_indexer else "none"
    owns_long = True if ref.cache_kind == "dense" else bool(ref.owns_long_kv)
    window = int(draft.sliding_window or ref.window)
    return LayerPlan(
        index=index, attention=draft.attention_kind, architecture=ref.architecture,
        ffn="moe" if draft.num_experts else "dense",
        moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
        noncausal_attention=draft.noncausal_query_block,
        mla_variant=draft.mla_variant, mla_output_gate=mla_gate,
        kv_cache_elements=kv_elems, qk_norm=qk_norm, q_output_gate=q_gate,
        cache_kind=ref.cache_kind, window=window,
        kv_owner_layer_id=index if owns_long else -1, owns_long_kv=owns_long,
        compress_ratio=ref.compress_ratio, compressor_kind=ref.compressor_kind,
        compressor_width=ref.compressor_width, compressor_state_rows=ref.compressor_state_rows,
        indexer=has_indexer, indexer_type=indexer_type,
        owns_index_k=bool(has_indexer and not share_index_k),
        index_k_owner_layer_id=(index if (has_indexer and not share_index_k) else -1),
        owns_topk=has_indexer, topk_owner_layer_id=(index if has_indexer else -1),
        index_query_per_layer=bool(ref.index_query_per_layer) if has_indexer else False,
        index_heads=ref.index_heads if has_indexer else 0,
        index_dim=ref.index_dim if has_indexer else 0,
        index_topk=ref.index_topk if has_indexer else 0,
        index_kpool=ref.index_kpool if has_indexer else 1,
        index_kpool_compress=bool(ref.index_kpool_compress) if has_indexer else False,
        index_kpool_always_select_tail=bool(ref.index_kpool_always_select_tail) if has_indexer else False,
        index_input=ref.index_input,
        candidate_filter=bool(ref.candidate_filter) if has_indexer else False,
        candidate_block_size=ref.candidate_block_size if has_indexer else 0,
        candidate_topk_blocks=ref.candidate_topk_blocks if has_indexer else 0,
        candidate_source_layer_id=-1,
        hyper_connections=bool(mhc), hc_mult=mhc, hc_sinkhorn_iters=mhc_iters,
        **plan_fields,
    )


def draft_layer_plans(model, kind="mtp"):
    draft = optional_draft_spec(model, kind)
    plan_fields = {
        key: getattr(draft, key) for key in (
            "num_heads", "num_kv_heads", "head_dim", "q_lora_rank", "kv_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "o_lora_rank",
            "o_groups", "num_experts", "top_k", "num_shared_experts", "intermediate_size",
        )
    }
    return tuple(_draft_layer_plan(model, draft, index, plan_fields) for index in range(draft.layers))


def draft_program(frontend, model, runtime, views, ready, *, global_views=None, prefill=False,
                  replica=0, replay_scope="none"):
    kind = dict(runtime.metadata).get("draft_mode", "mtp")
    draft = optional_draft_spec(model, kind)
    selected_view = lambda view: view.phase == "prefill" if prefill else bool(view.draft_proposal_tokens)
    active = tuple(view for view in views if selected_view(view))
    all_active = tuple(view for view in (global_views or views) if selected_view(view))
    # Autoregressive MTP computes one proposal at a time. A parallel block
    # drafter computes its declared query block in one transformer pass.
    serial = not prefill and not draft.noncausal_query_block and draft.kind == "mtp"
    steps = max(view.draft_proposal_tokens for view in all_active) if serial else 1
    plans = draft_layer_plans(model, kind)
    draft_model = replace(
        model, name="draft", layers=draft.layers, hidden_size=draft.hidden_size,
        num_heads=draft.num_heads, num_kv_heads=draft.num_kv_heads, head_dim=draft.head_dim,
        intermediate_size=draft.intermediate_size, layer_plans=plans, auxiliary=None,
        expert_routing=(), num_experts=draft.num_experts, top_k=draft.top_k,
        vocab_size=draft.vocab_size or model.vocab_size,
        quant=draft_quant(model, draft), activation_dtype=draft.activation_dtype,
        compute_dtype=draft.activation_dtype,
    )
    regions, previous = [], ready
    stage = runtime.parallel.pp - 1
    # Feature projection shards its output; the draft body consumes a complete
    # hidden vector, so TP must gather before entering the first layer.
    tp = runtime.parallel.tp
    for step in range(steps):
        ops = []
        selected = tuple(view for view in active if not serial or view.draft_proposal_tokens > step)

        def proposal(view):
            return replace(
                view, q_len=view.q_len if prefill else 1 if serial else view.draft_proposal_tokens,
                samples=not prefill, phase="prefill" if prefill else "decode",
                context=view.context + (step if serial else 0),
                draft_verify=False, draft_proposal_tokens=0,
            )
        proposal_views = tuple(proposal(view) for view in selected)
        global_proposals = tuple(proposal(view) for view in all_active
                                 if not serial or view.draft_proposal_tokens > step)
        rows = sum(view.q_len for view in proposal_views)
        selected_runtime, selections = resolve(
            runtime, proposal_views, padded_tokens=rows, replay_scope=replay_scope,
            global_views=global_proposals, model_scope="draft")
        # Sequence-parallel lowering is body-scoped; draft_program manages TP
        # feature gather explicitly before entering the draft body.
        if "sequence_parallel" in selected_runtime.kernels:
            selected_runtime = replace(
                selected_runtime,
                kernels={k: v for k, v in selected_runtime.kernels.items()
                         if k != "sequence_parallel"})
        requests = tuple(view.request_id for view in proposal_views)
        prefix = "draft_prefill." if prefill else f"draft{step}."
        feature_deps = [previous]
        targets = stage_map(model, runtime)
        for layer_id in draft.target_layer_ids:
            source_stage = targets[layer_id]
            if source_stage != stage:
                transfer = comm_op(
                    op_id=prefix + f"features_from{layer_id}", deps=(previous,), stage=stage,
                    layer=-1, requests=requests, family="pp_p2p", phase="draft",
                    recipe="draft_feature_transfer", link_bytes=rows * model.hidden_size * 2,
                    shape=(rows, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("source_stage", source_stage),
                              ("destination_stage", stage)))
                ops.append(transfer)
                feature_deps.append(transfer.op_id)
        projection = gemm_op(
            op_id=prefix + "features", deps=tuple(feature_deps), stage=stage, layer=-1, requests=requests,
            m=rows, n=(draft.hidden_size + tp - 1) // tp, k=draft.input_width, quant=draft_model.quant,
            category=draft.input_projection_category, phase="draft", family="gemm",
            recipe="draft_features",
        )
        if rows and draft.input_mode == "dual_normalized":
            projected = []
            for component in ("embedding", "target_hidden"):
                normalized = vector_op(
                    op_id=prefix + component + "_norm", deps=tuple(feature_deps), stage=stage,
                    layer=-1, requests=requests, elements=rows * draft.hidden_size,
                    ops_per_element=5, dtype=draft.activation_dtype, family="rmsnorm",
                    phase="draft", recipe="draft_input_norm")
                projection = gemm_op(
                    op_id=prefix + component + "_projection", deps=(normalized.op_id,), stage=stage,
                    layer=-1, requests=requests, m=rows, n=draft.hidden_size, k=draft.hidden_size,
                    quant=draft_model.quant, category="linear_param_bf16", phase="draft",
                    family="gemm", recipe="draft_feature_projection")
                ops.extend((normalized, projection))
                projected.append(projection.op_id)
            projection = vector_op(
                op_id=prefix + "features", deps=tuple(projected), stage=stage, layer=-1,
                requests=requests, elements=rows * draft.hidden_size, ops_per_element=1,
                dtype=draft.activation_dtype, family="elementwise", phase="draft",
                recipe="draft_feature_add")
            ops.append(projection)
        elif rows:
            ops.append(projection)
            if tp > 1:
                gathered = comm_op(
                    op_id=prefix + "features_gather", deps=(projection.op_id,), stage=stage,
                    layer=-1, requests=requests, family="allgather", phase="draft",
                    recipe="draft_features_gather", link_bytes=rows * draft.hidden_size * 2 * (tp - 1) / tp,
                    shape=(rows, draft.hidden_size), metadata=(("group_axis", "tp"), ("steps", tp - 1)))
                ops.append(gathered)
                projection = gathered
        global_tokens = sum(view.q_len if prefill else 1 if serial else view.draft_proposal_tokens
                            for view in all_active if not serial or view.draft_proposal_tokens > step)
        body, previous = frontend.body(
            draft_model, selected_runtime, proposal_views, prefix=prefix,
            ready=projection.op_id if rows else previous,
            tokens=rows, mode="draft_prefill" if prefill else "draft",
            stages=(stage,) * draft.layers, global_tokens=global_tokens, replica=replica,
        )
        ops.extend(body)
        if prefill:
            rows_by_request = {view.request_id: view.q_len for view in proposal_views}
            regions.append(LoweringRegion(
                _tag_draft_ops(ops, rows_by_request),
                selected_runtime, draft_model.activation_dtype, rows, selections))
            continue
        heads, previous = frontend.logits(draft_model, selected_runtime, proposal_views, previous,
                                         prefix=prefix, mode="draft", stage=stage, sample=False)
        ops.extend(heads)
        if draft.markov_rank and rows:
            lookup = vector_op(
                op_id=prefix + "markov_lookup", deps=(previous,), stage=stage, layer=-1,
                requests=requests, elements=rows * draft.markov_rank, ops_per_element=1,
                dtype=draft.activation_dtype, family="embedding", phase="draft",
                recipe="markov_embedding")
            markov = gemm_op(
                op_id=prefix + "markov", deps=(lookup.op_id,), stage=stage, layer=-1, requests=requests,
                m=rows, n=draft.vocab_size, k=draft.markov_rank, quant=draft_model.quant,
                category="lm_head", phase="draft", family="logits", recipe="draft_markov_head",
            )
            correction = vector_op(
                op_id=prefix + "markov_bias", deps=(previous, markov.op_id), stage=stage,
                layer=-1, requests=requests, elements=rows * draft.vocab_size,
                ops_per_element=1, dtype="fp32", family="elementwise", phase="draft",
                recipe="markov_bias")
            ops.extend((lookup, markov, correction))
            previous = correction.op_id
        if rows and (draft.confidence_executes or (runtime.dspark and runtime.dspark.confidence)):
            confidence = vector_op(
                op_id=prefix + "confidence", deps=(previous,), stage=stage, layer=-1, requests=requests,
                elements=rows * (draft.hidden_size + draft.markov_rank), ops_per_element=2,
                dtype=model.activation_dtype, family="confidence", phase="draft",
                recipe="draft_confidence",
            )
            ops.append(confidence)
            previous = confidence.op_id
        if rows:
            sample = vector_op(
                op_id=prefix + "sample", deps=(previous,), stage=stage, layer=-1,
                requests=requests, elements=rows * draft_model.vocab_size,
                ops_per_element=5, dtype="fp32", family="sampling", phase="draft",
                recipe="draft_sample")
            ops.append(sample)
            previous = sample.op_id
        rows_by_request = {
            view.request_id: (view.q_len if prefill else 1 if serial else view.draft_proposal_tokens)
            for view in selected
        }
        regions.append(LoweringRegion(
            _tag_draft_ops(ops, rows_by_request, step=step),
            selected_runtime, draft_model.activation_dtype, rows, selections))
    # Draft runs on the last pipeline stage. Send accepted proposal ids back
    # to stage 0 so the target schedule can consume them.
    if not prefill and runtime.parallel.pp > 1 and regions and rows > 0:
        stage = runtime.parallel.pp - 1
        active = [v for v in views if v.draft_proposal_tokens > 0] or list(views)
        xfer = comm_op(
            op_id="draft_proposal_transfer",
            deps=(previous,),
            stage=0,
            layer=-1,
            requests=tuple(v.request_id for v in active),
            family="p2p",
            phase="draft",
            recipe="draft_proposal_transfer",
            link_bytes=len(active) * max(
                1, max((v.draft_proposal_tokens for v in active), default=1)) * 8,
            shape=(len(active),),
            metadata=(
                ("group_axis", "pp"),
                ("source_stage", stage),
                ("destination_stage", 0),
            ),
        )
        xfer = _tag_draft_ops((xfer,), {
            view.request_id: view.draft_proposal_tokens for view in active
        })[0]
        last = regions[-1]
        regions[-1] = replace(last, ops=last.ops + (xfer,))
        previous = xfer.op_id
    return regions, previous
