"""Canonical operator names, aliases, strict parameter validation and dispatch.

[Role]
  Normalize a native or user-supplied spelling to one canonical key, validate
  every parameter through the strict scalar validators *before* any coercion,
  fill documented ABI defaults, and call the matching builder.

[Invariants]
  * Every canonical operator maps to one builder; alias resolution is stable.
  * No parameter is coerced with ``int(...)``/``float(...)``: integers must be
    ``int`` (never bool/float/str), booleans must be ``bool``.
  * A missing dimension raises ``FormulaValidationError``; no shape is invented.
  * ``composite_chain`` is resolved by the engine, never a leaf builder.

[Timeouts & Locks]
  Pure dictionary lookup and arithmetic; no I/O.

[Side-Effect Policy]
  No mutation of caller params or global state.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .errors import FormulaValidationError
from .hardware import HardwareProfile, dtype_bytes
from .ops_attention import (
    attention_work, lightning_indexer_work, mla_prolog_work, sparse_mla_work,
)
from .ops_comm import collective_work, mc2_work
from .ops_linear_attn import (
    causal_conv1d_work, chunk_linear_attention_work, recover_ssm_work,
    recurrent_linear_attention_work,
)
from .ops_matmul import (
    argmax_work, embedding_work, grouped_matmul_work, lm_head_work,
    matmul_work, top_p_sample_work, validate_quant_mode,
)
from .ops_vector import (
    dynamic_quant_work, moe_routing_work, rms_norm_work, rope_work,
    slice_work, swiglu_work,
)
from .primitives import (
    decode_sequence_lengths, strict_bool, strict_finite_float, strict_int,
    strict_int_list, strict_pos_int,
)
from .work import OpWork

from .registry_names import (
    COMPOSITE, _CATALOG, canonical_name, supported_operators,
    _get, _int_param, _bool_param, _str_param, _layout_param, _decoded_seq_param,
)


def _build_attention(canonical: str, params: Mapping[str, Any],
                     hardware: HardwareProfile, steady: bool) -> OpWork:
    q_len = _int_param(params, ("q_len", "S_q", "seq_q", "tokens"), default=1,
                       operator=canonical)
    kv_len = _int_param(params, ("kv_len", "S_kv", "seq_kv", "context_len"),
                        default=q_len, operator=canonical)
    spec_slots = _int_param(params, ("spec_slots", "num_spec_tokens"), default=1,
                            operator=canonical)
    layout = _layout_param(params, operator=canonical, default="TND")
    batch = _int_param(params, ("batch", "B"), default=1, operator=canonical)
    encoding = _get(params, "actual_seq_lengths_type", "seq_len_encoding",
                    default="lengths", operator=canonical)
    total_q = batch * q_len
    total_kv = batch * kv_len
    q_raw = _get(params, "actual_seq_lengths", "actual_seq_lens",
                 "actual_seq_lens_q", "actual_seq_lengths_q", "seq_lens_q",
                 "seq_lens", default=None, operator=canonical)
    kv_raw = _get(params, "actual_seq_lengths_kv", "actual_seq_lens_kv",
                  "seq_lens_kv", default=None, operator=canonical)
    if kv_raw is None and q_raw is not None and params.get("seq_lens") is not None:
        kv_raw = params["seq_lens"]
    q_lens = None if q_raw is None else decode_sequence_lengths(
        q_raw, encoding=encoding, total_tokens=total_q,
        expected_count=batch, label=f"{canonical}.actual_seq_lens_q")
    kv_lens = None if kv_raw is None else decode_sequence_lengths(
        kv_raw, encoding=encoding, total_tokens=total_kv,
        expected_count=batch, label=f"{canonical}.actual_seq_lens_kv")
    if q_lens is not None and len(q_lens) == 1 and spec_slots > 1:
        q_lens = q_lens + (q_lens[0],) * (spec_slots - 1)
    phase = _str_param(params, ("phase",), default=("decode" if q_len <= 4 else "prefill"),
                       operator=canonical)
    paged = canonical == "paged_attention" or _bool_param(
        params, ("paged",), default=False, operator=canonical)
    mask_shape = _get(params, "attn_mask_shape", "mask_shape", default=None,
                      operator=canonical)
    if mask_shape is not None:
        mask_shape = strict_int_list(mask_shape, f"{canonical}.attn_mask_shape")
        if any(int(dim) < 1 for dim in mask_shape):
            raise FormulaValidationError(
                f"{canonical}.attn_mask_shape dimensions must be positive")
    mask_flag = None
    if params.get("has_attn_mask") is not None:
        mask_flag = strict_bool(params["has_attn_mask"],
                                f"{canonical}.has_attn_mask")
    elif type(params.get("attn_mask")) is bool:
        mask_flag = params["attn_mask"]
    return attention_work(
        q_len=q_len, kv_len=kv_len,
        hq=_int_param(params, ("hq", "num_heads", "num_query_heads"), default=32,
                      operator=canonical),
        hkv=_int_param(params, ("hkv", "num_kv_heads", "num_key_value_heads"),
                       default=1, operator=canonical),
        head_dim=_int_param(params, ("head_dim", "D", "d"), default=128,
                            operator=canonical),
        head_dim_v=_get(params, "head_dim_v", "Dv", default=None, operator=canonical),
        batch=batch,
        causal=_bool_param(params, ("causal",), default=False, operator=canonical),
        causal_offset=_get(params, "causal_offset", "pos_offset", default=None,
                           operator=canonical),
        phase=phase.lower(), layout=layout,
        dtype=_str_param(params, ("dtype",), default="bf16", operator=canonical),
        kv_dtype=_get(params, "kv_dtype", "kv_quant", default=None),
        q_lens=q_lens, kv_lens=kv_lens,
        has_attn_mask=mask_flag,
        attn_mask_shape=mask_shape,
        mask_dtype=_str_param(params, ("mask_dtype",), default="uint8",
                              operator=canonical),
        storage_q=_get(params, "storage_q", "storage_m", default=None,
                       operator=canonical),
        storage_kv=_get(params, "storage_kv", default=None, operator=canonical),
        paged=paged,
        block_size=_int_param(params, ("block_size",), default=128,
                              operator=canonical),
        paged_blocks=_int_param(params, ("paged_blocks",), default=0,
                                operator=canonical, minimum=0),
        steady_l2_warm=steady, hardware=hardware,
        family=("paged_attention" if paged else canonical),
        padding_skipped=_bool_param(params, ("padding_skipped",), default=False,
                                    operator=canonical),
    )


def _build_norm(canonical: str, params: Mapping[str, Any],
                hardware: HardwareProfile, steady: bool) -> OpWork:
    rows = _int_param(params, ("rows", "tokens", "M"), default=1, operator=canonical)
    hidden = _int_param(params, ("hidden", "H", "N"), operator=canonical)
    residual = canonical == "add_rms_norm" or _bool_param(
        params, ("residual",), default=False, operator=canonical)
    qk = canonical == "qk_rms_norm" or _bool_param(
        params, ("qk",), default=False, operator=canonical)
    return rms_norm_work(
        rows=rows, hidden=hidden,
        dtype=_str_param(params, ("dtype",), default="bf16", operator=canonical),
        gamma=_bool_param(params, ("gamma",), default=True, operator=canonical),
        residual=residual,
        inplace=_bool_param(params, ("inplace",), default=False, operator=canonical),
        qk=qk,
        groups=_int_param(params, ("groups",), default=(2 if qk else 1),
                          operator=canonical),
        gamma_dtype=_get(params, "gamma_dtype", default=None),
        gamma_elems=_get(params, "gamma_elems", default=None),
        out_dtype=_get(params, "out_dtype", default=None),
        residual_dtype=_get(params, "residual_dtype", default=None),
        x_out_dtype=_get(params, "x_out_dtype", "residual_out_dtype", default=None),
        rstd_elems=_int_param(params, ("rstd_elems",), default=0,
                              operator=canonical, minimum=0),
        rstd_dtype=_str_param(params, ("rstd_dtype",), default="fp32",
                              operator=canonical),
        steady_l2_warm=steady, hardware=hardware,
    )


def _quant_param(params: Mapping[str, Any], *, operator: str,
                 default: str | None = None) -> str | None:
    """Validate an explicitly supplied quant mode; never truthiness-coerce."""
    value: Any = None
    present = False
    for key in ("quant", "quant_mode"):
        if key in params and params[key] is not None:
            value = params[key]
            present = True
            break
    if not present:
        return default
    if not isinstance(value, str) or not value.strip():
        raise FormulaValidationError(
            f"{operator}.quant must be a non-empty string, got {value!r}")
    return validate_quant_mode(value)


def _matmul(canonical: str, params: Mapping[str, Any], hw: HardwareProfile,
            steady: bool) -> OpWork:
    quant = _quant_param(
        params, operator=canonical,
        default="w8a8" if canonical == "quant_batch_matmul" else None)
    group_size = _get(params, "group_size", default=None)
    return matmul_work(
        m=_int_param(params, ("M", "m", "rows", "tokens"), operator=canonical),
        n=_int_param(params, ("N", "n"), operator=canonical),
        k=_int_param(params, ("K", "k"), operator=canonical),
        dtype=_str_param(params, ("dtype",), default="bf16", operator=canonical),
        weight_dtype=_get(params, "weight_dtype", default=None),
        out_dtype=_get(params, "out_dtype", default=None),
        layout=_layout_param(params, operator=canonical, default="ND"),
        quant=quant,
        group_size=(strict_int(group_size, "group_size", minimum=1)
                    if group_size is not None else None),
        bias=_bool_param(params, ("bias",), default=False, operator=canonical),
        zero_points=_bool_param(params, ("zero_points",), default=False,
                                operator=canonical),
        batch=_int_param(params, ("batch", "B"), default=1, operator=canonical),
        weight_broadcast=_bool_param(params, ("weight_broadcast",), default=True,
                                     operator=canonical),
        act_batch=_get(params, "act_batch", default=None),
        weight_batch=_get(params, "weight_batch", default=None),
        scale_bytes=_get(params, "scale_bytes", default=None),
        act_scale_bytes=_get(params, "act_scale_bytes", default=None),
        steady_l2_warm=steady, hardware=hw,
        family=("quant_batch_matmul" if canonical == "quant_batch_matmul"
                else "matmul"),
        variant=str(quant or "dense"),
    )


def _grouped(canonical: str, params: Mapping[str, Any], hw: HardwareProfile,
             steady: bool) -> OpWork:
    e_total = _int_param(params, ("experts_total", "E", "experts", "num_experts"),
                         operator=canonical)
    ep = _int_param(params, ("ep", "ep_world_size"), default=1, operator=canonical)
    e_local = _get(params, "experts_local", "E_local", default=None)
    e_local_i = (-(-e_total // ep) if e_local is None else
                 strict_int(e_local, "experts_local", minimum=1))
    group_list = _get(params, "group_list", "expert_token_nums", default=None)
    if group_list is not None:
        group_list = strict_int_list(group_list, "group_list")
    routed = _get(params, "routed_rows", default=None)
    group_size = _get(params, "group_size", default=None)
    k_dim = _get(params, "k_dim", "explicit_k", default=None)
    n_dim = _get(params, "n_dim", "explicit_n", default=None)
    quant = _quant_param(params, operator=canonical)
    return grouped_matmul_work(
        tokens=_int_param(params, ("tokens", "T", "rows"), operator=canonical),
        hidden=_int_param(params, ("hidden", "H"), operator=canonical),
        inter=_int_param(params, ("inter", "moe_inter", "I"), operator=canonical),
        k_dim=(strict_int(k_dim, "k_dim", minimum=1)
               if k_dim is not None else None),
        n_dim=(strict_int(n_dim, "n_dim", minimum=1)
               if n_dim is not None else None),
        experts_total=e_total, experts_local=e_local_i,
        topk=_int_param(params, ("topk", "K"), operator=canonical),
        ep=ep, mode=_str_param(params, ("mode",), default="up_gate",
                               operator=canonical),
        dtype=_str_param(params, ("dtype",), default="bf16", operator=canonical),
        weight_dtype=_get(params, "weight_dtype", default=None),
        out_dtype=_get(params, "out_dtype", default=None),
        quant=quant,
        group_list=group_list,
        group_list_type=_int_param(params, ("group_list_type",), default=1,
                                   operator=canonical, minimum=0),
        routed_rows=(strict_int(routed, "routed_rows", minimum=0)
                     if routed is not None else None),
        group_size=(strict_int(group_size, "group_size", minimum=1)
                    if group_size is not None else None),
        zero_points=_bool_param(params, ("zero_points",), default=False,
                                operator=canonical),
        scale_bytes=_get(params, "scale_bytes", default=None),
        steady_l2_warm=steady, hardware=hw,
    )


def build_work(canonical: str, params: Mapping[str, Any], hardware: HardwareProfile,
               *, steady_l2_warm: bool = False) -> OpWork:
    """Dispatch one canonical operator to its builder with strict validation."""
    if canonical not in _CATALOG:
        raise FormulaValidationError(f"unsupported operator {canonical!r}")
    family, kind, mode = _CATALOG[canonical]
    if not isinstance(params, Mapping):
        raise FormulaValidationError("params must be a mapping")
    p = params
    if kind == "matmul":
        return _matmul(canonical, p, hardware, steady_l2_warm)
    if kind == "grouped":
        return _grouped(canonical, p, hardware, steady_l2_warm)
    if kind == "attention":
        return _build_attention(canonical, p, hardware, steady_l2_warm)
    if kind == "mla":
        return mla_prolog_work(
            tokens=_int_param(p, ("tokens", "T"), operator=canonical),
            hidden=_int_param(p, ("hidden", "H"), operator=canonical),
            num_heads=_int_param(p, ("heads", "num_heads", "hq"), operator=canonical),
            head_dim=_int_param(p, ("head_dim", "D"), default=128, operator=canonical),
            head_dim_v=_get(p, "head_dim_v", default=None),
            q_lora_rank=_int_param(p, ("q_lora_rank", "q_lora"), operator=canonical),
            kv_lora_rank=_int_param(p, ("kv_lora_rank", "kv_lora"), operator=canonical),
            rope_dim=_int_param(p, ("rope_dim",), default=64, operator=canonical,
                                minimum=0),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            fused=_bool_param(p, ("fused",), default=True, operator=canonical),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "sparse":
        tokens = _int_param(p, ("tokens", "T"), operator=canonical)
        kv_len = _int_param(p, ("kv_len", "S_kv"), operator=canonical)
        q_lens = _decoded_seq_param(
            p, ("actual_seq_lens_q", "q_lens", "actual_seq_lengths"),
            operator=canonical, total_tokens=tokens,
            expected_count=_get(p, "batch", default=None))
        kv_lens = _decoded_seq_param(
            p, ("actual_seq_lens_kv", "kv_lens", "actual_seq_lengths_kv"),
            operator=canonical, total_tokens=None,
            expected_count=_get(p, "batch", default=None))
        return sparse_mla_work(
            tokens=tokens,
            num_heads=_int_param(p, ("heads", "num_heads", "hq"), operator=canonical),
            head_dim=_int_param(p, ("head_dim", "D"), default=128, operator=canonical),
            head_dim_v=_get(p, "head_dim_v", default=None),
            kv_len=kv_len,
            topk=_int_param(p, ("topk",), default=2048, operator=canonical),
            q_lens=q_lens, kv_lens=kv_lens,
            causal=_bool_param(p, ("causal",), default=False, operator=canonical),
            causal_offset=_get(p, "causal_offset", default=None),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            kv_dtype=_get(p, "kv_dtype", default=None),
            has_attn_mask=_bool_param(p, ("has_attn_mask",), default=False,
                                      operator=canonical),
            unique_selected_kv=_get(p, "unique_selected_kv", default=None),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "indexer":
        tokens = _int_param(p, ("tokens", "T"), operator=canonical)
        kv_len = _int_param(p, ("kv_len", "S_kv"), operator=canonical)
        q_lens = _decoded_seq_param(
            p, ("actual_seq_lens_q", "q_lens", "actual_seq_lengths"),
            operator=canonical, total_tokens=tokens,
            expected_count=_get(p, "batch", default=None))
        kv_lens = _decoded_seq_param(
            p, ("actual_seq_lens_kv", "kv_lens", "actual_seq_lengths_kv"),
            operator=canonical, total_tokens=None,
            expected_count=_get(p, "batch", default=None))
        return lightning_indexer_work(
            tokens=_int_param(p, ("tokens", "T"), operator=canonical),
            index_heads=_int_param(p, ("index_heads", "heads", "hq"),
                                   operator=canonical),
            index_dim=_int_param(p, ("index_dim",), default=128,
                                 operator=canonical),
            kv_len=_int_param(p, ("kv_len", "S_kv"), operator=canonical),
            topk=_int_param(p, ("topk",), default=2048, operator=canonical),
            q_lens=q_lens, kv_lens=kv_lens,
            output_index_heads=_get(p, "output_index_heads", default=None),
            causal=_bool_param(p, ("causal",), default=False, operator=canonical),
            causal_offset=_get(p, "causal_offset", default=None),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            kv_dtype=_get(p, "kv_dtype", default=None),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "chunk":
        tokens = _int_param(p, ("tokens", "T"), operator=canonical)
        lens = _decoded_seq_param(
            p, ("actual_seq_lengths", "actual_seq_lens", "cu_seqlens",
                "actual_seq_lens_q"),
            operator=canonical, total_tokens=tokens)
        return chunk_linear_attention_work(
            tokens=tokens,
            num_heads=_int_param(p, ("heads", "num_heads", "H"),
                                 operator=canonical),
            head_dim_k=_int_param(p, ("head_dim_k", "head_dim", "D"),
                                  default=128, operator=canonical),
            head_dim_v=_get(p, "head_dim_v", default=None),
            batch=_get(p, "batch", "num_sequences", "B", default=None),
            actual_seq_lens=lens,
            chunk_size=_int_param(p, ("chunk_size", "chunk"), default=64,
                                  operator=canonical),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            state_dtype=_str_param(p, ("state_dtype",), default="fp32",
                                   operator=canonical),
            beta_elems=_get(p, "beta_elems", default=None),
            beta_dtype=_get(p, "beta_dtype", default=None),
            gate_elems=_get(p, "gate_elems", default=None),
            gate_dtype=_get(p, "gate_dtype", default=None),
            out_dtype=_get(p, "out_dtype", default=None),
            prefix_replay=_bool_param(p, ("prefix_replay",), default=False,
                                      operator=canonical),
            hardware=hardware, family=family,
        )
    if kind == "recurrent":
        return recurrent_linear_attention_work(
            tokens=_int_param(p, ("tokens", "B", "batch"), operator=canonical),
            num_heads=_int_param(p, ("heads", "num_heads", "H"),
                                 operator=canonical),
            head_dim_k=_int_param(p, ("head_dim_k", "head_dim", "D"),
                                  default=128, operator=canonical),
            head_dim_v=_get(p, "head_dim_v", default=None),
            batch=_get(p, "batch", "B", default=None),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            out_dtype=_get(p, "out_dtype", default=None),
            beta_elems=_get(p, "beta_elems", default=None),
            beta_dtype=_get(p, "beta_dtype", default=None),
            gate_elems=_get(p, "gate_elems", default=None),
            gate_dtype=_get(p, "gate_dtype", default=None),
            state_dtype=_str_param(p, ("state_dtype",), default="fp32",
                                   operator=canonical),
            spec_slots=_int_param(p, ("spec_slots", "num_spec_tokens"),
                                  default=1, operator=canonical),
            write_per_slot_states=_bool_param(
                p, ("write_per_slot_states",), default=False,
                operator=canonical),
            hardware=hardware, family=family,
        )
    if kind == "recover":
        return recover_ssm_work(
            tokens=_int_param(p, ("tokens", "B", "batch"), operator=canonical),
            num_heads=_int_param(p, ("heads", "num_heads", "H"),
                                 operator=canonical),
            head_dim_k=_int_param(p, ("head_dim_k", "head_dim", "D"),
                                  default=128, operator=canonical),
            head_dim_v=_get(p, "head_dim_v", default=None),
            spec_slots=_int_param(p, ("spec_slots", "num_spec_tokens"),
                                  default=1, operator=canonical),
            accepted_tokens=_get(p, "accepted_tokens", "m", default=None),
            phase=mode,
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            state_dtype=_str_param(p, ("state_dtype",), default="fp32",
                                   operator=canonical),
            align_boundary_crossings=_int_param(
                p, ("align_boundary_crossings",), default=0,
                operator=canonical, minimum=0),
            hardware=hardware,
        )
    if kind == "conv":
        tokens = _int_param(p, ("tokens", "T"), operator=canonical)
        run_mode = _int_param(p, ("run_mode",), default=0, operator=canonical,
                              minimum=0)
        lens = _decoded_seq_param(
            p, ("actual_seq_lengths", "actual_seq_lens", "query_start_loc",
                "actual_seq_lens_q"),
            operator=canonical, total_tokens=tokens)
        return causal_conv1d_work(
            tokens=tokens,
            channels=_int_param(p, ("channels", "C", "hidden"),
                                operator=canonical),
            width=_int_param(p, ("width",), default=4, operator=canonical),
            run_mode=run_mode,
            batch=_get(p, "batch", "B", "num_sequences", default=None),
            actual_seq_lens=lens,
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "norm":
        return _build_norm(canonical, p, hardware, steady_l2_warm)
    if kind == "slice":
        return slice_work(
            rows=_int_param(p, ("rows", "tokens", "M"), operator=canonical),
            in_hidden=_int_param(p, ("in_hidden", "input_hidden", "H"),
                                 operator=canonical),
            out_hidden=_int_param(p, ("out_hidden", "hidden", "N"),
                                  operator=canonical),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "swiglu":
        return swiglu_work(
            rows=_int_param(p, ("rows", "tokens", "M"), operator=canonical),
            hidden=_int_param(p, ("hidden", "H", "inter"), operator=canonical),
            mode=_str_param(p, ("mode",), default=mode or "swiglu",
                            operator=canonical),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            inplace=_bool_param(p, ("inplace",), default=False, operator=canonical),
            two_input=_bool_param(p, ("two_input",), default=False,
                                  operator=canonical),
            out_dtype=_get(p, "out_dtype", default=None),
            first_elems=_get(p, "first_elems", default=None),
            second_elems=_get(p, "second_elems", default=None),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "rope":
        return rope_work(
            tokens=_int_param(p, ("tokens", "T"), operator=canonical),
            heads=_int_param(p, ("heads", "num_heads", "H"), operator=canonical),
            head_dim=_int_param(p, ("head_dim", "D"), default=128,
                                operator=canonical),
            rotary_dim=_get(p, "rotary_dim", default=None),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            qk=_bool_param(p, ("qk",), default=True, operator=canonical),
            fused=_bool_param(p, ("fused",), default=True, operator=canonical),
            pass_through_copy=_get(p, "pass_through_copy", default=None),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "quant":
        return dynamic_quant_work(
            rows=_int_param(p, ("rows", "tokens", "M"), operator=canonical),
            hidden=_int_param(p, ("hidden", "H"), operator=canonical),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            out_dtype=_str_param(p, ("out_dtype",), default="int8",
                                 operator=canonical),
            dequant_swiglu=(mode == "dequant_swiglu"),
            has_smooth_scales=_bool_param(
                p, ("has_smooth_scales", "smooth_scale_present"), default=False,
                operator=canonical),
            smooth_scale_bytes=_get(p, "smooth_scale_bytes", default=None),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "routing":
        return moe_routing_work(
            tokens=_int_param(p, ("tokens", "T"), operator=canonical),
            experts=_int_param(p, ("experts", "E", "experts_total"),
                               operator=canonical),
            hidden=_int_param(p, ("hidden", "H"), operator=canonical),
            topk=_int_param(p, ("topk",), default=8, operator=canonical),
            mode=mode,
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "mc2":
        e_total = _int_param(p, ("experts_total", "E"), operator=canonical)
        ep = _int_param(p, ("ep", "ep_world_size"), operator=canonical)
        e_local = _get(p, "experts_local", default=None)
        routed = _get(p, "routed_rows", default=None)
        remote = _get(p, "remote_routed_rows", "remote", default=None)
        return mc2_work(
            tokens=_int_param(p, ("tokens", "T"), operator=canonical),
            hidden=_int_param(p, ("hidden", "H"), operator=canonical),
            experts_total=e_total,
            experts_local=(-(-e_total // ep) if e_local is None else
                           strict_int(e_local, "experts_local", minimum=0)),
            topk=_int_param(p, ("topk", "K"), operator=canonical),
            ep=ep, stage=mode,
            has_scales=_bool_param(p, ("has_scales",), default=True, operator=canonical),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            quant=_get(p, "quant", default=None),
            use_measured_fit=_bool_param(p, ("use_measured_fit",), default=True,
                                         operator=canonical),
            transport=_get(p, "transport", default=None),
            routed_rows=(strict_int(routed, "routed_rows", minimum=0)
                         if routed is not None else None),
            remote_routed_rows=(strict_int(remote, "remote_routed_rows",
                                           minimum=0)
                                if remote is not None else None),
            sent_rows=_get(p, "sent_rows", "sent_routed_rows", "sent", default=None),
            received_rows=_get(p, "received_rows", "received_routed_rows", "received", default=None),
            local_rows=_get(p, "local_rows", "local_routed_rows", "local", default=None),
            skew=strict_finite_float(_get(p, "skew", default=1.0), "skew",
                                     minimum=0.0),
            hardware=hardware,
        )
    if kind == "collective":
        send = _get(p, "send_counts", default=None)
        recv = _get(p, "recv_counts", default=None)
        op_mode = ("all_to_all_v" if mode == "all_to_all_single"
                   and (send is not None or recv is not None) else mode)
        return collective_work(
            op=op_mode,
            size_bytes=_int_param(p, ("size_bytes", "bytes", "B"),
                                  operator=canonical),
            world_size=_int_param(p, ("world_size", "p", "ws"),
                                  operator=canonical),
            algorithm=_str_param(p, ("algorithm", "algo"), default="ring",
                                 operator=canonical),
            dtype=_str_param(p, ("dtype",), default="bf16", operator=canonical),
            transport=_get(p, "transport", default=None),
            send_counts=(strict_int_list(send, "send_counts")
                         if send is not None else None),
            recv_counts=(strict_int_list(recv, "recv_counts")
                         if recv is not None else None),
            self_rank=_int_param(p, ("self_rank", "root_rank"), default=0,
                                 operator=canonical, minimum=0),
            hardware=hardware,
        )
    if kind == "embedding":
        tokens = _int_param(p, ("tokens", "T"), operator=canonical)
        hidden = _int_param(p, ("hidden", "H"), default=1, operator=canonical)
        vocab = _int_param(p, ("vocab", "V", "vocab_size"), operator=canonical)
        tp = _int_param(p, ("tp",), default=1, operator=canonical)
        dtype = _str_param(p, ("dtype",), default="bf16", operator=canonical)
        logits_dtype = _str_param(p, ("logits_dtype",), default="fp32",
                                  operator=canonical)
        if mode == "embedding":
            return embedding_work(tokens=tokens, hidden=hidden, vocab=vocab,
                                  tp=tp, dtype=dtype,
                                  steady_l2_warm=steady_l2_warm,
                                  hardware=hardware)
        if mode == "argmax":
            return argmax_work(tokens=tokens, vocab=vocab,
                               logits_dtype=logits_dtype, dtype=dtype,
                               hardware=hardware)
        if mode == "top_p_sample":
            return top_p_sample_work(tokens=tokens, vocab=vocab,
                                     logits_dtype=logits_dtype, dtype=dtype,
                                     hardware=hardware)
        return lm_head_work(
            tokens=tokens, hidden=hidden, vocab=vocab, tp=tp, dtype=dtype,
            logits_dtype=logits_dtype,
            include_sampler=_bool_param(p, ("include_sampler",), default=False,
                                        operator=canonical),
            steady_l2_warm=steady_l2_warm, hardware=hardware,
        )
    if kind == "composite":
        raise FormulaValidationError(
            "composite_chain is resolved by the engine, not a leaf builder")
    raise FormulaValidationError(f"no builder for {canonical!r}")


__all__ = ["COMPOSITE", "build_work", "canonical_name", "supported_operators"]
