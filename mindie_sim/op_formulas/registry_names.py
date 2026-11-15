"""Stable canonical operator catalogue and alias normalization.

[Role]
  Own the canonical operator table and deterministic alias resolution used by
  ``registry.build_work`` and the public ``supported_operators``/canonical-name
  contract.  Keeping the table in its own module holds every source file below
  the project line ceiling while preserving one source of truth.

[Invariants]
  * Every canonical operator maps to exactly one ``(family, kind, mode)`` tuple.
  * Alias resolution is deterministic and independent of insertion order.
  * Normalization strips framework prefixes and version suffixes only after an
    exact or compact-name match is attempted.

[Timeouts & Locks]
  Pure dictionary lookup and regular-expression work; no I/O.

[Side-Effect Policy]
  No mutation of caller state; the catalogue and alias maps are module constants.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .errors import FormulaValidationError
from .primitives import decode_sequence_lengths, strict_bool, strict_int

COMPOSITE = "composite_chain"

# canonical -> (family, builder kind, extra mode)
_CATALOG: dict[str, tuple[str, str, str]] = {
    "matmul": ("matmul", "matmul", ""),
    "quant_batch_matmul": ("quant_batch_matmul", "matmul", ""),
    "grouped_matmul": ("grouped_matmul", "grouped", ""),
    "flash_attention": ("flash_attention", "attention", ""),
    "incre_flash_attention": ("incre_flash_attention", "attention", "decode"),
    "paged_attention": ("paged_attention", "attention", "paged"),
    "fused_infer_attention_score": ("fused_infer_attention_score", "attention", "decode"),
    "mla_prolog": ("mla_prolog", "mla", ""),
    "sparse_flash_mla": ("sparse_flash_mla", "sparse", ""),
    "lightning_indexer": ("lightning_indexer", "indexer", ""),
    "chunk_kda": ("chunk_kda", "chunk", ""),
    "chunk_gated_delta_rule": ("chunk_gated_delta_rule", "chunk", ""),
    "recurrent_kda": ("recurrent_kda", "recurrent", ""),
    "fused_recurrent_gdn": ("fused_recurrent_gdn", "recurrent", ""),
    "kda_recover_ssm_verify": ("kda_recover_ssm", "recover", "verify"),
    "kda_recover_ssm_commit": ("kda_recover_ssm", "recover", "commit"),
    "causal_conv1d": ("causal_conv1d", "conv", ""),
    "rms_norm": ("rms_norm", "norm", ""),
    "add_rms_norm": ("rms_norm", "norm", "add"),
    "qk_rms_norm": ("rms_norm", "norm", "qk"),
    "swiglu": ("swiglu", "swiglu", ""),
    "silu_and_mul": ("swiglu", "swiglu", "silu_and_mul"),
    "geglu": ("swiglu", "swiglu", "geglu"),
    "rotary_position_embedding": ("rotary_embedding", "rope", ""),
    "slice": ("slice", "slice", ""),
    "dynamic_quant": ("dynamic_quant", "quant", ""),
    "dequant_swiglu_quant": ("dynamic_quant", "quant", "dequant_swiglu"),
    "moe_gating_topk_softmax": ("moe_gating", "routing", "gating"),
    "moe_init_routing": ("moe_gating", "routing", "init_routing"),
    "moe_finalize_routing": ("moe_gating", "routing", "finalize"),
    "moe_distribute_dispatch": ("moe_distribute", "mc2", "dispatch"),
    "moe_distribute_combine": ("moe_distribute", "mc2", "combine"),
    "hccl_all_reduce": ("collective", "collective", "all_reduce"),
    "hccl_reduce_scatter": ("collective", "collective", "reduce_scatter"),
    "hccl_all_gather": ("collective", "collective", "all_gather"),
    "hccl_all_to_all": ("collective", "collective", "all_to_all_single"),
    "hccl_broadcast": ("collective", "collective", "broadcast"),
    "embedding": ("embedding", "embedding", "embedding"),
    "vocab_parallel_embedding": ("embedding", "embedding", "embedding"),
    "lm_head": ("lm_head", "embedding", "lm_head"),
    "logits_matmul": ("lm_head", "embedding", "lm_head"),
    "argmax": ("argmax", "embedding", "argmax"),
    "top_p_sample": ("top_p_sample", "embedding", "top_p_sample"),
    COMPOSITE: ("composite", "composite", ""),
}

_ALIASES: dict[str, str] = {
    "matmulv3": "matmul", "mat_mul": "matmul", "linear": "matmul",
    "gemm": "matmul", "mm": "matmul", "addmm": "matmul",
    "quantmatmul": "quant_batch_matmul", "qbmm": "quant_batch_matmul",
    "w8a8": "quant_batch_matmul", "w4a16": "quant_batch_matmul",
    "w4a8": "quant_batch_matmul", "mxfp8": "quant_batch_matmul",
    "mxfp4": "quant_batch_matmul",
    "groupedmatmul": "grouped_matmul", "groupedmatmulv4": "grouped_matmul",
    "groupedmatmulv5": "grouped_matmul", "gmm": "grouped_matmul",
    "flashattentionscore": "flash_attention", "flash_attention_score": "flash_attention",
    "fia": "flash_attention", "fusedinferattentionscore": "fused_infer_attention_score",
    "fusedinferattentionscorev2": "fused_infer_attention_score",
    "prefillattention": "flash_attention",
    "increflashattention": "incre_flash_attention", "pagedattention": "paged_attention",
    "mlapo": "mla_prolog", "mlaprologue": "mla_prolog",
    "sparseflashmla": "sparse_flash_mla", "lightningindexer": "lightning_indexer",
    "dsa": "sparse_flash_mla",
    "chunkgateddeltarule": "chunk_gated_delta_rule", "gdnprefill": "chunk_kda",
    "fusedrecurrentgdn": "fused_recurrent_gdn", "gdndecode": "recurrent_kda",
    "kdarecoverssmverify": "kda_recover_ssm_verify",
    "kdarecoverssmcommit": "kda_recover_ssm_commit",
    "recover_ssm": "kda_recover_ssm_verify",
    "causalconv1d": "causal_conv1d", "conv1d": "causal_conv1d",
    "rmsnorm": "rms_norm", "layernorm": "rms_norm", "add_rmsnorm": "add_rms_norm",
    "addrmsnorm": "add_rms_norm", "qkrmsnorm": "qk_rms_norm",
    "siluandmul": "silu_and_mul", "swish": "swiglu", "geglu": "geglu",
    "rope": "rotary_position_embedding", "fusedrope": "rotary_position_embedding",
    "rotaryembedding": "rotary_position_embedding",
    "dynamicquant": "dynamic_quant", "dequantswigluquant": "dequant_swiglu_quant",
    "moegatingtopksoftmax": "moe_gating_topk_softmax",
    "moeinitrouting": "moe_init_routing", "moeinitroutingv2": "moe_init_routing",
    "moefinalizerouting": "moe_finalize_routing",
    "moedistributedispatch": "moe_distribute_dispatch",
    "moedistributedispatchv2": "moe_distribute_dispatch",
    "moedistributedispatchv3": "moe_distribute_dispatch",
    "moedistributecombine": "moe_distribute_combine",
    "moedistributecombinev2": "moe_distribute_combine",
    "moedistributecombinev3": "moe_distribute_combine",
    "all_reduce": "hccl_all_reduce", "allreduce": "hccl_all_reduce",
    "hcclallreduce": "hccl_all_reduce", "hcclallgather": "hccl_all_gather",
    "hcclreducescatter": "hccl_reduce_scatter", "hcclalltoall": "hccl_all_to_all",
    "hcclalltoallv": "hccl_all_to_all", "alltoallv": "hccl_all_to_all",
    "reduce_scatter": "hccl_reduce_scatter", "reducescatter": "hccl_reduce_scatter",
    "all_gather": "hccl_all_gather", "allgather": "hccl_all_gather",
    "all_to_all": "hccl_all_to_all", "alltoall": "hccl_all_to_all",
    "all_to_all_single": "hccl_all_to_all", "alltoallsingle": "hccl_all_to_all",
    "hccl_all_to_allv": "hccl_all_to_all", "broadcast": "hccl_broadcast",
    "hcclbroadcast": "hccl_broadcast",
    "vocabparallel_embedding": "vocab_parallel_embedding",
    "vocabparallelembedding": "vocab_parallel_embedding",
    "logitsmatmul": "lm_head", "argmax": "argmax", "toppsample": "top_p_sample",
    "top_p": "top_p_sample", "composite": COMPOSITE,
}

_COMPACT_CATALOG = {name.replace("_", ""): name for name in _CATALOG}


def _norm(name: str) -> str:
    text = str(name or "").strip().lower()
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    for prefix in ("aclnn", "aclop", "npu_"):
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix):]
            break
    if "_" in text:
        head = text.split("_", 1)[0]
        head_no_version = re.sub(r"v\d+$", "", head)
        if head in _ALIASES or head in _CATALOG:
            text = head
        elif (head_no_version in _ALIASES or head_no_version in _CATALOG
              or head_no_version in _COMPACT_CATALOG):
            text = head_no_version
    return "".join(char for char in text if char.isalnum())


def canonical_name(name: str) -> str | None:
    """Return the canonical operator key or ``None`` for an unknown spelling."""
    text = str(name or "").strip()
    raw = text.lower()
    if raw in _CATALOG:
        return raw
    key = _norm(text)
    if key in _CATALOG:
        return key
    if key in _COMPACT_CATALOG:
        return _COMPACT_CATALOG[key]
    return _ALIASES.get(key)


def supported_operators() -> tuple[str, ...]:
    return tuple(sorted(_CATALOG))




__all__ = [
    "COMPOSITE", "canonical_name", "supported_operators",
]


def _get(params: Mapping[str, Any], *names: str, default: Any = None,
         required: bool = False, operator: str = "") -> Any:
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
    if required:
        raise FormulaValidationError(
            f"{operator}: missing required parameter {'/'.join(names)}")
    return default


def _int_param(params: Mapping[str, Any], names: tuple[str, ...], *,
               default: int | None = None, operator: str,
               minimum: int = 1) -> int:
    value = _get(params, *names, default=default, required=default is None,
                 operator=operator)
    return strict_int(value, f"{operator}.{names[0]}", minimum=minimum)


def _bool_param(params: Mapping[str, Any], names: tuple[str, ...], *,
                default: bool, operator: str) -> bool:
    value = _get(params, *names, default=default, required=False, operator=operator)
    return strict_bool(value, f"{operator}.{names[0]}")


def _str_param(params: Mapping[str, Any], names: tuple[str, ...], *,
               default: str, operator: str) -> str:
    value = _get(params, *names, default=default, required=False, operator=operator)
    if not isinstance(value, str) or not value.strip():
        raise FormulaValidationError(
            f"{operator}.{names[0]} must be a non-empty string")
    return value.strip()


_LAYOUTS = {"ND", "TND", "BSND", "BNSD", "BSH", "SBH", "FRACTAL_NZ", "NZ",
            "NCHW", "SCALAR"}


def _layout_param(params: Mapping[str, Any], *, operator: str,
                  default: str = "ND") -> str:
    value = _str_param(params, ("layout", "input_layout"), default=default,
                       operator=operator)
    key = value.strip().upper()
    if key not in _LAYOUTS:
        raise FormulaValidationError(f"{operator}.layout unknown: {value!r}")
    return key


def _decoded_seq_param(params: Mapping[str, Any], names: tuple[str, ...],
                       *, operator: str, total_tokens: int | None = None,
                       expected_count: int | None = None
                       ) -> tuple[int, ...] | None:
    raw = _get(params, *names, default=None, operator=operator)
    if raw is None:
        return None
    encoding = _get(params, "actual_seq_lengths_type", "seq_len_encoding",
                    default="lengths", operator=operator)
    count = (None if expected_count is None else
             strict_int(expected_count, f"{operator}.batch", minimum=1))
    return decode_sequence_lengths(
        raw, encoding=encoding, total_tokens=total_tokens,
        expected_count=count, label=f"{operator}.{names[0]}")
