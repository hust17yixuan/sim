"""Finite canonical-subgraph vocabulary, builder registry and legality.

[Role]
  Register the 29 typed standard subgraph families, the per-kind builder
  functions supplied by ``instantiate``, an interning cache keyed by semantic
  structure, and the declared combination legality of the vocabulary.

[Invariants]
  * The vocabulary is finite and explicit; a missing mechanism is a coverage
    hole, never an unknown silent kind.
  * Interning keys contain kind/variant/operator structure and declared
    contracts, never model names, layer ids or request identities.
  * The cache is bounded (LRU) so a long-lived process cannot accumulate
    arbitrary template history.
  * A combination is rejected only for declared structural reasons (for
    example an indexer consumer without a producer), never because a model
    label or measurement is absent.

[Timeouts & Locks]
  Bounded in-memory dictionaries; no waits, I/O, locks or device probes.

[Side-Effect Policy]
  Registers builders in a module-level table and mutates only the caller-owned
  bounded cache; no filesystem or network access.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import (
    ALL_ATTENTION_KINDS, CoverageHole, OperatorTemplate, StatePortSpec,
    SubgraphKind, SubgraphTemplate, SUBGRAPH_KINDS, SUBGRAPH_KIND_SET,
    validate_kind,
)


#: One standard attention macro: QKV projection through TP reduction.  The
#: MHA/GQA/MQA/attention kinds share this operator surface and differ only in
#: the declared head grouping executed by the attention core.
_STANDARD_ATTENTION_OPS = (
    "qkv_projection", "qk_norm", "rotary", "kv_store", "kv_read",
    "attention", "output_gate", "o_projection", "tp_allreduce",
    "tp_all_gather", "tp_reduce_scatter",
)


def _kind(kind, title, variants, operators, inputs=(), outputs=(), coverage="structural",
          notes=""):
    return SubgraphKind(kind, title, tuple(variants), tuple(operators),
                        tuple(inputs), tuple(outputs), coverage, notes)


#: The complete canonical LLM vocabulary.  ``C``-like mechanisms that are only
#: partially covered keep an explicit ``partial`` label and a note.
VOCABULARY: tuple[SubgraphKind, ...] = (
    _kind("vocab_parallel_embedding", "Input Embeddings", ("token", "media"),
          ("embedding", "embedding_allreduce", "input_copy", "media_splice"),
          outputs=("hidden",),
          notes="media splice requires mm_encoder_projector geometry"),
    _kind("layernorm", "Normalization and Residual Addition",
          ("rmsnorm", "layernorm", "residual", "qk_norm", "fused_add_rmsnorm"),
          ("rmsnorm", "layernorm", "residual_add", "qk_norm", "cast"),
          inputs=("hidden", "residual"), outputs=("hidden", "residual")),
    _kind("kv_cache", "KV Cache Read, Write and Transfer",
          ("store", "read", "layout", "transfer"),
          ("kv_store", "kv_read", "layout_convert", "stage_transfer"),
          inputs=("kv_cache",), outputs=("kv_cache", "kv_ready")),
    _kind("kv_compressor", "KV Cache Compression and Key Pooling",
          ("weighted_pool", "gated_ring", "kpool", "none"),
          ("compress", "kpool", "tail_store", "state_store"),
          inputs=("kv_cache", "compress_state"),
          outputs=("compressed_kv", "compress_state")),
    _kind("indexer_prolog", "Attention Indexer Projections and Cache Update",
          ("c4", "full", "shared", "none"),
          ("index_q_projection", "index_k_projection", "index_rope",
           "index_quant", "index_cache_store"),
          inputs=("hidden", "index_cache"), outputs=("index_cache", "index_ready")),
    _kind("lightning_indexer", "Lightning Indexer Scoring and Top-K Selection",
          ("topk", "pooled_tail", "candidate"),
          ("index_score", "topk", "candidate_filter", "cp_gather"),
          inputs=("index_cache", "index_selection"),
          outputs=("index_selection",), coverage="partial",
          notes="candidate reuse and CP gather stay separate contracts"),
    _kind("mlp", "Dense Feed-Forward Network",
          ("silu", "gelu", "gelu_tanh", "situ", "ungated"),
          ("gate_up", "activation", "down", "tp_allreduce",
           "tp_all_gather", "tp_reduce_scatter"),
          inputs=("hidden",), outputs=("hidden",)),
    _kind("moe_router", "MoE Routing and Token Permutation",
          ("softmax", "sigmoid", "sqrtsoftplus", "hash", "grouped"),
          ("router_gemm", "topk", "histogram", "sort"),
          inputs=("hidden",), outputs=("routing_ledger", "expert_rows"),
          coverage="partial", notes="native router variants remain explicit gaps"),
    _kind("moe_token_dispatcher", "MoE Token Dispatch and Combine",
          ("alltoall", "mc2", "allgather", "local",
           "tp_join_reduce", "local_expert_reduce"),
          ("dispatch", "combine", "quantize", "notify"),
          inputs=("hidden", "expert_rows"), outputs=("hidden", "peer_traffic"),
          coverage="partial"),
    _kind("routed_experts", "Routed Expert FFN", ("packed", "latent", "mxfp4"),
          ("expert_gate_up", "expert_activation", "expert_down"),
          inputs=("hidden", "expert_rows"), outputs=("hidden",),
          coverage="partial",
          notes="expected and integer execution domains stay split"),
    _kind("shared_experts", "Shared Expert FFN and Output Merge",
          ("plain", "gated", "sp_gather", "join"),
          ("shared_gate_up", "shared_activation", "shared_down", "join"),
          inputs=("hidden",), outputs=("hidden",)),
    _kind("hyper_connection", "Hyper-Connections",
          ("hc", "hc_arch35", "attn_pre", "attn_post", "ffn_pre", "ffn_post"),
          ("hc_projection", "inv_rms", "affine", "sinkhorn", "mix"),
          inputs=("hidden",), outputs=("hidden",)),
    _kind("attn_residual", "Attention Residuals", ("block_stack", "slot_mix"),
          ("slot_push", "score", "softmax_mix", "cast"),
          inputs=("hidden", "residual_slots"), outputs=("hidden", "residual_slots")),
    _kind("per_layer_embedding", "Engram and Per-Layer Embeddings",
          ("engram", "ple"),
          ("table_lookup", "gate", "projection"),
          inputs=("hidden", "lookup_tokens"), outputs=("hidden",),
          coverage="partial", notes="Engram and Gemma PLE are distinct contracts"),
    _kind("lm_head_sampler", "Final Normalization, LM Head and Sampling",
          ("all", "selected"),
          ("final_norm", "hidden_select", "lm_head", "logits_gather",
           "tp_all_gather", "sampling"),
          inputs=("hidden",), outputs=("logits", "sampled_tokens")),
    _kind("spec_decode", "Speculative Decoding Draft and Verify",
          ("mtp", "dspark", "standalone"),
          ("draft_projection", "draft_heads", "target_verify", "accept_commit"),
          inputs=("hidden", "draft_state", "accept_state"),
          outputs=("hidden", "accept_state"), coverage="partial",
          notes="target verify is never discounted by acceptance"),
    _kind("mm_encoder_projector", "Multimodal Encoder and Projector",
          ("vision", "audio", "unified"),
          ("vision_encoder", "projector", "splice"),
          inputs=("media_tokens",), outputs=("hidden",), coverage="partial"),
    _kind("mha_attention", "Multi-Head Attention",
          ("mha", "shared_consumer"),
          _STANDARD_ATTENTION_OPS,
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache"),
          notes="shared_consumer omits KV projection and pays reads only"),
    _kind("gqa_attention", "Grouped-Query Attention",
          ("gqa", "shared_consumer"),
          _STANDARD_ATTENTION_OPS,
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache"),
          notes="shared_consumer omits KV projection and pays reads only"),
    _kind("mqa_attention", "Multi-Query Attention",
          ("mqa", "shared_consumer"),
          _STANDARD_ATTENTION_OPS,
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache"),
          notes="shared_consumer omits KV projection and pays reads only"),
    _kind("attention", "Scaled Dot-Product Attention",
          ("default", "shared_consumer"),
          _STANDARD_ATTENTION_OPS,
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache"),
          notes="shape-free fallback when a plan declares no head geometry"),
    _kind("mla_attention", "Multi-head Latent Attention",
          ("latent_mla", "latent_mqa", "direct_q"),
          ("q_down", "q_up", "kv_down", "rotary", "mla_attention",
           "v_expand", "o_projection", "tp_allreduce"),
          inputs=("hidden", "kv_cache"), outputs=("hidden", "kv_cache")),
    _kind("dsa_attention", "DeepSeek Sparse Attention",
          ("latent_mla", "latent_mqa", "direct_q"),
          ("q_down", "q_up", "kv_down", "rotary", "mla_attention",
           "v_expand", "o_projection", "tp_allreduce"),
          inputs=("hidden", "kv_cache", "index_selection"),
          outputs=("hidden", "kv_cache"), coverage="partial",
          notes="lightning_indexer Top-K selects the attended prefix"),
    _kind("swa_attention", "Sliding-Window Attention",
          ("window_only", "window_compressed", "shared_consumer"),
          ("window_attention", "compressed_attention", "merge", "grouped_o",
           "output_gate"),
          inputs=("hidden", "kv_cache"), outputs=("hidden",), coverage="partial",
          notes="window branch only; no compressed-KV attention"),
    _kind("csa_attention", "Compressed Sparse Attention",
          ("window_compressed",),
          ("window_attention", "compressed_attention", "merge", "grouped_o",
           "output_gate"),
          inputs=("hidden", "compressed_kv", "index_selection"),
          outputs=("hidden",), coverage="partial",
          notes="per-layer compressor, index_k and single-stage Top-K"),
    _kind("csa2_attention", "Compressed Sparse Attention v2",
          ("window_compressed",),
          ("window_attention", "compressed_attention", "merge", "grouped_o",
           "output_gate"),
          inputs=("hidden", "compressed_kv", "index_selection"),
          outputs=("hidden",), coverage="partial",
          notes="cross-layer shared compressed KV with candidate-block filtering"),
    _kind("hca_attention", "Hierarchically Compressed Attention",
          ("window_compressed",),
          ("window_attention", "compressed_attention", "merge", "grouped_o",
           "output_gate"),
          inputs=("hidden", "compressed_kv"),
          outputs=("hidden",), coverage="partial",
          notes="128x compressed branch without lightning_indexer Top-K"),
    _kind("gdn_linear_attn", "Gated DeltaNet Linear Attention",
          ("chunked_prefill", "recurrent_decode"),
          ("qkvz_projection", "ba_projection", "conv", "prepare",
           "gated_delta", "gated_norm", "o_projection", "tp_allreduce",
           "tp_all_gather", "tp_reduce_scatter"),
          inputs=("hidden", "recurrent_state", "conv_state"),
          outputs=("hidden", "recurrent_state", "conv_state"),
          coverage="partial", notes="implementation domain constraints stay explicit"),
    _kind("kda_linear_attn", "Kimi Delta Attention",
          ("chunked_prefill", "recurrent_decode"),
          ("qkv_projection", "gate_projection", "conv", "prepare",
           "kda_recurrent", "output_norm", "o_projection"),
          inputs=("hidden", "recurrent_state", "conv_state", "accepted_state"),
          outputs=("hidden", "recurrent_state", "conv_state"),
          coverage="partial", notes="accepted-state contract is explicit"),
)

VOCABULARY_BY_KIND = {entry.kind: entry for entry in VOCABULARY}

if tuple(VOCABULARY_BY_KIND) != SUBGRAPH_KINDS:
    raise RuntimeError("subgraph_vocabulary_order")


@dataclass(frozen=True)
class CombinationRule:
    kind: str
    requires: tuple[str, ...] = ()
    requires_any: tuple[str, ...] = ()
    reason: str = ""

    def validate(self, kinds: set[str]) -> None:
        if self.kind not in kinds:
            return
        missing = [item for item in self.requires if item not in kinds]
        if missing:
            raise ModelingError(
                f"subgraph_combination_invalid:{self.kind}:missing={sorted(missing)}")
        if self.requires_any and not (set(self.requires_any) & kinds):
            raise ModelingError(
                f"subgraph_combination_invalid:{self.kind}:requires_any="
                f"{sorted(self.requires_any)}")


#: Structural legality only.  It says nothing about which implementation is
#: available for a (kind, variant, hardware) triple.
COMBINATION_RULES: tuple[CombinationRule, ...] = (
    CombinationRule("lightning_indexer", requires=("indexer_prolog",),
                    reason="selection needs a produced index cache"),
    CombinationRule("kv_compressor", requires=("kv_cache",),
                    reason="compression needs a declared KV cache"),
    CombinationRule("csa_attention", requires_any=("kv_compressor", "kv_cache"),
                    reason="compressed attention needs a cache producer"),
    CombinationRule("csa2_attention", requires_any=("kv_compressor", "kv_cache"),
                    reason="compressed attention needs a cache producer"),
    CombinationRule("hca_attention", requires_any=("kv_compressor", "kv_cache"),
                    reason="compressed attention needs a cache producer"),
    CombinationRule("swa_attention", requires=("kv_cache",),
                    reason="window attention reads a declared KV cache"),
    CombinationRule("shared_experts", requires_any=("routed_experts", "mlp"),
                    reason="shared branch joins a routed or dense branch"),
    CombinationRule("spec_decode", requires_any=tuple(sorted(ALL_ATTENTION_KINDS)),
                    reason="draft/verify needs an attention macro"),
)


def validate_combination(kinds: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(kinds))
    unknown = set(ordered) - SUBGRAPH_KIND_SET
    if unknown:
        raise ModelingError(f"subgraph_kind_unknown:{sorted(unknown)}")
    present = set(ordered)
    for rule in COMBINATION_RULES:
        rule.validate(present)
    return ordered


def vocabulary() -> tuple[SubgraphKind, ...]:
    return VOCABULARY


def kind_spec(kind: str) -> SubgraphKind:
    validate_kind(kind)
    return VOCABULARY_BY_KIND[kind]


def port(name: str, direction: str, representation: str, **kwargs) -> StatePortSpec:
    return StatePortSpec(name=name, direction=direction,
                         representation=representation, **kwargs)


def operator(name: str, family: str, role: str, **kwargs) -> OperatorTemplate:
    return OperatorTemplate(name=name, family=family, role=role, **kwargs)


def template(kind: str, variant: str, title: str,
             operators: Iterable[OperatorTemplate], **kwargs) -> SubgraphTemplate:
    return SubgraphTemplate(kind=kind, variant=variant, title=title,
                            operators=tuple(operators), **kwargs)


class TemplateRegistry:
    """Bounded LRU cache of immutable subgraph templates."""

    def __init__(self, max_entries: int = 256):
        if not isinstance(max_entries, int) or max_entries < 1:
            raise ModelingError("template_registry_capacity_invalid")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, SubgraphTemplate] = OrderedDict()
        self.evictions = 0

    def intern(self, value: SubgraphTemplate) -> SubgraphTemplate:
        if not isinstance(value, SubgraphTemplate):
            raise ModelingError("template_registry_value_invalid")
        key = value.key()
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return existing
        self._entries[key] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        return value

    def get(self, key: tuple) -> SubgraphTemplate | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


KIND_BUILDERS: dict[str, Callable[..., SubgraphTemplate]] = {}


def register_builder(kind: str, builder: Callable[..., SubgraphTemplate]) -> None:
    validate_kind(kind)
    if not callable(builder):
        raise ModelingError(f"subgraph_builder_invalid:{kind}")
    if kind in KIND_BUILDERS:
        raise ModelingError(f"subgraph_builder_duplicate:{kind}")
    KIND_BUILDERS[kind] = builder


def build(kind: str, context: Any) -> Any:
    """Invoke the registered builder; it may return a template or a binding."""
    validate_kind(kind)
    builder = KIND_BUILDERS.get(kind)
    if builder is None:
        raise ModelingError(f"subgraph_builder_missing:{kind}")
    return builder(context)


def build_template(kind: str, context: Any) -> SubgraphTemplate:
    value = build(kind, context)
    template = getattr(value, "template", value)
    if not isinstance(template, SubgraphTemplate):
        raise ModelingError(f"subgraph_builder_return_invalid:{kind}")
    return template


def coverage_holes(kinds: Iterable[str], *, reason: str = "declared_partial") -> tuple[CoverageHole, ...]:
    holes = []
    for kind in kinds:
        spec = kind_spec(kind)
        if spec.coverage != "structural":
            holes.append(CoverageHole(kind, reason, spec.notes))
    return tuple(holes)
