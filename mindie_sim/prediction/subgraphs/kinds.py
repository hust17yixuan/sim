"""Paper-aligned attention subgraph kind resolution.

[Role] Map one declared ``LayerPlan`` attention mechanism onto exactly one
  canonical English subgraph kind from ``mindie_sim.schema.subgraphs``.  The
  layer slot planner and the independent builders share this resolver so a
  declared plan cannot produce two different kind names for one layer.

[Invariants]
  * Resolution reads declared plan fields only: ``attention``, head counts,
    ``window``, ``compress_ratio``, ``cache_kind``, ``architecture`` and the
    indexer/top-k/KV owner ids.  No model name, layer id, measurement,
    hardware tier or request shape takes part.
  * A "window_compressed" plan with an unrecognised declared architecture is
    refused instead of being folded into one of the compressed families.
  * The resolver is a pure function: the same plan always yields the same
    vocabulary string, and a kind outside the vocabulary is a modeling error.

[Timeouts & Locks] Bounded branching; no waits, I/O or locks.

[Side-Effect Policy] Returns an immutable vocabulary string; mutates nothing.
"""
from __future__ import annotations

from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import (
    LATENT_ATTENTION_KINDS, STANDARD_ATTENTION_KINDS,
    WINDOW_SPARSE_ATTENTION_KINDS,
)

MHA_ATTENTION = "mha_attention"
GQA_ATTENTION = "gqa_attention"
MQA_ATTENTION = "mqa_attention"
ATTENTION = "attention"
MLA_ATTENTION = "mla_attention"
DSA_ATTENTION = "dsa_attention"
SWA_ATTENTION = "swa_attention"
CSA_ATTENTION = "csa_attention"
CSA2_ATTENTION = "csa2_attention"
HCA_ATTENTION = "hca_attention"
GDN_LINEAR_ATTN = "gdn_linear_attn"
KDA_LINEAR_ATTN = "kda_linear_attn"

#: Declared execution variant of each standard attention kind.
STANDARD_VARIANTS = {
    MHA_ATTENTION: "mha",
    GQA_ATTENTION: "gqa",
    MQA_ATTENTION: "mqa",
    ATTENTION: "default",
    SWA_ATTENTION: "window_only",
}

#: Declared ``LayerPlan.architecture`` value of each window/compressed family.
_WINDOW_COMPRESSED_ARCHITECTURES = {
    "swa": SWA_ATTENTION,
    "csa": CSA_ATTENTION,
    "csa2": CSA2_ATTENTION,
    "hca": HCA_ATTENTION,
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def standard_attention_kind(layer: Any) -> str:
    """MHA/GQA/MQA from head geometry, ``attention`` when shape-free.

    A declared ``window > 0`` with no compression is a sliding-window
    schedule and resolves to ``swa_attention`` even though it is built by the
    standard attention macro.
    """
    heads = _int(getattr(layer, "num_heads", 0))
    kv_heads = _int(getattr(layer, "num_kv_heads", 0))
    declared = str(getattr(layer, "attention", "") or "").strip().lower()
    window = _int(getattr(layer, "window", 0))
    ratio = _int(getattr(layer, "compress_ratio", 0))
    if heads > 0 and kv_heads > 0:
        if kv_heads == 1 and heads > 1:
            family = MQA_ATTENTION
        elif heads == kv_heads:
            family = MHA_ATTENTION
        else:
            family = GQA_ATTENTION
    elif declared == "mha":
        family = MHA_ATTENTION
    elif declared == "mqa":
        family = MQA_ATTENTION
    elif declared == "gqa":
        family = GQA_ATTENTION
    else:
        family = ATTENTION
    if family != ATTENTION and window > 0 and ratio == 0:
        return SWA_ATTENTION
    return family


def latent_attention_kind(layer: Any) -> str:
    """``dsa_attention`` when a sparse indexer/selection is declared."""
    if str(getattr(layer, "cache_kind", "") or "") == "window_compressed":
        raise ModelingError(
            f"subgraph_latent_kind_on_window_compressed:layer={_int(getattr(layer, 'index', -1))}")
    sparse = (
        bool(getattr(layer, "indexer", False))
        or bool(getattr(layer, "owns_topk", False))
        or _int(getattr(layer, "topk_owner_layer_id", -1), -1) >= 0
        or _int(getattr(layer, "index_k_owner_layer_id", -1), -1) >= 0
    )
    return DSA_ATTENTION if sparse else MLA_ATTENTION


def _compressed_architecture(layer: Any) -> str:
    declared = str(getattr(layer, "architecture", "") or "").strip().lower()
    if declared in _WINDOW_COMPRESSED_ARCHITECTURES:
        return declared
    return ""


def window_compressed_attention_kind(layer: Any) -> str:
    """SWA / CSA / CSA2 / HCA from the declared compression schedule.

    ``compress_ratio == 0`` is the window-only member.  A positive ratio uses
    the declared architecture name first (``swa``/``csa``/``csa2``/``hca``);
    when a source only declares compression geometry, the structural fallback
    reads the declared owner/candidate contract: a shared compressed KV or a
    decoupled index/top-k owner with candidate-block filtering is CSA v2, a
    ratio above the single-stage range without selection is HCA, and the
    remaining per-layer single-stage schedule is CSA.
    """
    ratio = _int(getattr(layer, "compress_ratio", 0))
    if ratio <= 0:
        return SWA_ATTENTION
    declared = _compressed_architecture(layer)
    if declared:
        return _WINDOW_COMPRESSED_ARCHITECTURES[declared]
    compressor = str(getattr(layer, "compressor_kind", "") or "").strip().lower()
    if compressor in {"", "none"}:
        # A ratio without a declared compressor is not a compressed branch.
        return SWA_ATTENTION
    index_owner = _int(getattr(layer, "index_k_owner_layer_id", -1), -1)
    topk_owner = _int(getattr(layer, "topk_owner_layer_id", -1), -1)
    index_source = _int(getattr(layer, "index_source_layer_id", -1), -1)
    decoupled = index_source >= 0 and index_source != topk_owner
    selects = (bool(getattr(layer, "indexer", False))
               or bool(getattr(layer, "owns_topk", False))
               or topk_owner >= 0)
    if bool(getattr(layer, "candidate_filter", False)) or decoupled:
        return CSA2_ATTENTION
    if selects:
        # A per-layer single-stage lightning-indexer Top-K selection.
        return CSA_ATTENTION
    # No selection branch: the full compressed branch is the hierarchical
    # schedule regardless of the declared ratio magnitude.
    return HCA_ATTENTION


def attention_kind(layer: Any) -> str:
    """The single canonical kind of one layer's attention core."""
    if str(getattr(layer, "cache_kind", "") or "") == "window_compressed":
        return window_compressed_attention_kind(layer)
    declared = str(getattr(layer, "attention", "") or "").strip().lower()
    if declared == "mla":
        return latent_attention_kind(layer)
    if declared == "gdn":
        return GDN_LINEAR_ATTN
    if declared in {"kda", "mamba", "gla", "deltanet", "lightning", "retnet"}:
        return KDA_LINEAR_ATTN
    return standard_attention_kind(layer)


__all__ = [
    "ATTENTION", "CSA2_ATTENTION", "CSA_ATTENTION", "DSA_ATTENTION",
    "GDN_LINEAR_ATTN", "GQA_ATTENTION", "HCA_ATTENTION", "KDA_LINEAR_ATTN",
    "LATENT_ATTENTION_KINDS", "MHA_ATTENTION", "MLA_ATTENTION",
    "MQA_ATTENTION", "STANDARD_ATTENTION_KINDS", "STANDARD_VARIANTS",
    "SWA_ATTENTION",
    "WINDOW_SPARSE_ATTENTION_KINDS", "attention_kind", "latent_attention_kind",
    "standard_attention_kind", "window_compressed_attention_kind",
]
