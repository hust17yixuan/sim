"""DSpark draft-KV coupling: resolved grouping, bounded SWA and DCP blocks.

[Role]
  Model the DSpark draft cache as a first-class client of the target server:
  the Kimi-K3 mixed scheduler grouping over fully resolved cache signatures,
  the DeepSeek-V4.1 draft SWA window bound (with its replay exclusion), and
  the non-causal draft block under DCP/PCP with the source's query gather,
  latent-output/LSE merge and index-plane geometry.

[Invariants]
  * Every cache signature carries a unique layer name, its causal flag, its
    Ascend layout (ND or FRACTAL_NZ), full Mamba state shapes/dtypes and its
    speculative slot count.  Duplicate names are a typed rejection.
  * A causal target and non-causal draft never share one UniformType group.
    The source has no two-attention-group branch, so a useful split is labelled
    ``analytical_noncausal_split`` with ``source_compatible=False``.
  * Recurrent specs must be exactly equal on every resolved field, including
    same-byte/different-shape and different-dtype cases.
  * The draft SWA footprint is ``min(seq, W + extra)`` rounded to the KV block
    size, never the full ``ISL + OSL`` span, and the
    ``DeepseekV41DSparkSWACache`` class never replays.
  * DCP gathers query NoPE + RoPE head fragments (distinct TP heads under
    PCP), merges latent attention output plus FP32 softmax LSE, allocates one
    aligned index plane plus one per-row length vector, scales by batch and
    draft-layer count, and sums gather + attention + combine sequentially.
  * All dimensions are strict integers; NaN, booleans and negatives reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    ceil_div, dtype_elem_bytes, strict_bool, strict_dtype, strict_int,
    strict_number, strict_text,
)
from mindie_sim.schema.common import ModelingError

KIMI_K3_TARGET_PREFIX = "language_model.model.layers."
KIMI_K3_DRAFT_PREFIX = "model.layers."
DSPARK_DRAFT_SWA_CLASS = "DeepseekV41DSparkSWACache"
BOUNDED_REPLAY_ARCHITECTURES = frozenset({
    "DeepseekV41ForCausalLM", "DeepseekV41ForConditionalGeneration"})
SPEC_KINDS = frozenset({"full_attention", "mla_attention", "sliding_attention",
                        "mamba"})
SPEC_LAYOUTS = frozenset({"nd", "fractal_nz"})


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="dspark")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="dspark")


def _text(name: str, value: Any) -> str:
    return strict_text(name, value, reason="dspark")


@typed_inputs
@dataclass(frozen=True)
class KvSpecSignature:
    """A fully resolved cache spec, not a page-size proxy."""

    name: str
    kind: str
    block_size: int
    page_size_bytes: int
    scope: str
    mamba_type: str = "kda"
    noncausal_query_block: bool = False
    layout: str = "nd"
    state_shapes: tuple[tuple[int, ...], ...] = ()
    state_dtypes: tuple[str, ...] = ()
    speculative_slots: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    head_dtype: str = "bf16"
    sliding_window: int = 0

    def __post_init__(self) -> None:
        _text("layer_name", self.name)
        if not isinstance(self.kind, str) or self.kind not in SPEC_KINDS:
            raise ModelingError(f"dspark_spec_kind_invalid:{self.kind}")
        _int("block_size", self.block_size, minimum=1)
        _int("page_size_bytes", self.page_size_bytes, minimum=1)
        if not isinstance(self.scope, str) or self.scope not in {"target", "draft"}:
            raise ModelingError(f"dspark_spec_scope_invalid:{self.scope}")
        _text("mamba_type", self.mamba_type)
        _bool("noncausal_query_block", self.noncausal_query_block)
        if not isinstance(self.layout, str) or self.layout not in SPEC_LAYOUTS:
            raise ModelingError(f"dspark_spec_layout_invalid:{self.layout}")
        _int("speculative_slots", self.speculative_slots, minimum=0)
        if isinstance(self.state_shapes, (str, bytes)) or not isinstance(
                self.state_shapes, tuple):
            raise ModelingError("dspark_state_shapes_invalid")
        for shape in self.state_shapes:
            if isinstance(shape, (str, bytes)) or not isinstance(shape, tuple):
                raise ModelingError("dspark_state_shapes_invalid")
            for extent in shape:
                _int("state_extent", extent, minimum=1)
        if isinstance(self.state_dtypes, (str, bytes)) or not isinstance(
                self.state_dtypes, tuple):
            raise ModelingError("dspark_state_dtypes_invalid")
        for dtype in self.state_dtypes:
            strict_dtype("state_dtype", dtype, reason="dspark")
        if len(self.state_dtypes) != len(self.state_shapes):
            raise ModelingError("dspark_state_dtypes_not_aligned")
        if self.kind != "mamba" and (self.state_shapes or self.state_dtypes):
            raise ModelingError("dspark_attention_spec_has_state")
        for name in ("num_heads", "num_kv_heads", "head_dim", "sliding_window"):
            _int(name, getattr(self, name), minimum=0)
        strict_dtype("head_dtype", self.head_dtype, reason="dspark")

    @property
    def is_attention(self) -> bool:
        return self.kind in {"full_attention", "mla_attention",
                             "sliding_attention"}

    @property
    def is_full_attention(self) -> bool:
        """Mirror the source's ``isinstance(spec, FullAttentionSpec)``.

        A sliding-attention spec (or a full spec carrying a finite window) is
        not a ``FullAttentionSpec`` and cannot join the uniform merge.
        """
        return (self.kind in {"full_attention", "mla_attention"}
                and self.sliding_window == 0)

    @property
    def head_resolved(self) -> bool:
        return (self.num_heads > 0 and self.num_kv_heads > 0
                and self.head_dim > 0)

    @property
    def identity(self) -> tuple[Any, ...]:
        """Resolved fields that must match inside one uniform group."""
        return (self.kind, self.block_size, self.page_size_bytes,
                self.scope, self.mamba_type, self.noncausal_query_block,
                self.layout, self.state_shapes, self.state_dtypes,
                self.speculative_slots, self.num_heads, self.num_kv_heads,
                self.head_dim, self.head_dtype, self.sliding_window)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "block_size": self.block_size,
                "page_size_bytes": self.page_size_bytes,
                "scope": self.scope, "mamba_type": self.mamba_type,
                "noncausal_query_block": self.noncausal_query_block,
                "layout": self.layout,
                "state_shapes": [list(shape) for shape in self.state_shapes],
                "state_dtypes": list(self.state_dtypes),
                "speculative_slots": self.speculative_slots,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim, "head_dtype": self.head_dtype,
                "sliding_window": self.sliding_window}


@typed_inputs
@dataclass(frozen=True)
class KimiK3DsparkGrouping:
    """Attention scheduler groups plus balanced recurrent groups."""

    groups: tuple[tuple[str, ...], ...]
    fallback_reason: str | None
    target_attention_layers: int
    draft_attention_layers: int
    mamba_layers: int
    attention_groups: int = 0
    mode: str = "source_uniform_grouping"
    source_compatible: bool = True

    @property
    def ok(self) -> bool:
        return self.fallback_reason is None and self.source_compatible

    @property
    def scheduler_block_tables(self) -> int:
        return self.attention_groups if self.ok else 0

    @property
    def analytical_block_tables(self) -> int:
        """Alternative split's block tables; never claimed as a source result."""
        return self.attention_groups

    @property
    def attention_layer_count(self) -> int:
        return self.target_attention_layers + self.draft_attention_layers

    @property
    def mamba_group_count(self) -> int:
        if not self.groups:
            return 0
        return max(0, len(self.groups) - self.attention_groups)

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.groups)

    @property
    def noncausal_split(self) -> bool:
        return self.attention_groups == 2

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "fallback_reason": self.fallback_reason,
                "mode": self.mode,
                "source_compatible": self.source_compatible,
                "groups": [list(group) for group in self.groups],
                "group_sizes": list(self.group_sizes),
                "scheduler_block_tables": self.scheduler_block_tables,
                "analytical_block_tables": self.analytical_block_tables,
                "attention_groups": self.attention_groups,
                "noncausal_split": self.noncausal_split,
                "target_attention_layers": self.target_attention_layers,
                "draft_attention_layers": self.draft_attention_layers,
                "mamba_layers": self.mamba_layers,
                "mamba_group_count": self.mamba_group_count}


def _uniform_attention(specs: Sequence[KvSpecSignature]) -> bool:
    return len({(spec.block_size, spec.page_size_bytes, spec.layout,
                 spec.noncausal_query_block, spec.kind, spec.num_heads,
                 spec.num_kv_heads, spec.head_dim, spec.head_dtype,
                 spec.sliding_window)
                for spec in specs}) == 1


def _uniform_mamba(specs: Sequence[KvSpecSignature]) -> bool:
    return len({spec.identity for spec in specs}) == 1


def _analytical_groups(target: Sequence[KvSpecSignature],
                       draft: Sequence[KvSpecSignature],
                       mamba: Sequence[KvSpecSignature]) -> tuple[tuple[str, ...], ...]:
    """Labeled alternative split; not the source's uniform grouping."""
    groups: list[tuple[str, ...]] = [tuple(item.name for item in target)]
    if draft:
        groups.append(tuple(item.name for item in draft))
    names = [item.name for item in mamba]
    size = max(1, len(groups[0]))
    count = ceil_div(len(names), size) if names else 0
    for index in range(count):
        groups.append(tuple(names[index::count]))
    return tuple(groups)


@typed_inputs
def kimi_k3_dspark_grouping(specs: Sequence[KvSpecSignature], *,
                            strict: bool = False,
                            source_strict: bool = True) -> KimiK3DsparkGrouping:
    """Reproduce ``_get_kimi_k3_dspark_mixed_kv_cache_groups`` over specs.

    ``source_strict`` requires the authentic target/draft layer-name prefixes
    and fully resolved attention/Mamba specs.  An unrecognized signature yields
    an explicitly labeled ``analytical_fallback_split`` instead of a fabricated
    source-compatible grouping.
    """
    _bool("strict", strict)
    _bool("source_strict", source_strict)
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise ModelingError("dspark_specs_invalid")
    records = tuple(specs)
    if not records or not all(isinstance(item, KvSpecSignature)
                              for item in records):
        raise ModelingError("dspark_specs_invalid")
    names = [item.name for item in records]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ModelingError("dspark_specs_duplicate_layer_name")
    target = [item for item in records
              if item.scope == "target" and item.is_attention]
    draft = [item for item in records
             if item.scope == "draft" and item.is_attention]
    mamba = [item for item in records
             if item.kind == "mamba" and item.scope == "target"]
    matched = len(target) + len(draft) + len(mamba)
    fallback = _analytical_groups(target, draft, mamba)

    def fail(reason: str) -> KimiK3DsparkGrouping:
        if strict:
            raise ModelingError(f"kimi_k3_dspark_grouping_invalid:{reason}")
        return KimiK3DsparkGrouping(
            fallback, reason, len(target), len(draft), len(mamba),
            1 + (1 if draft else 0),
            mode="analytical_fallback_split", source_compatible=False)

    reason: str | None = None
    attention_groups = 0
    if not target or not draft or not mamba:
        reason = "not_kimi_k3_dspark_signature"
    elif matched != len(records):
        reason = "unrecognized_layer_signature"
    elif source_strict and not all(
            item.name.startswith(KIMI_K3_TARGET_PREFIX) for item in target):
        reason = "unrecognized_target_layer_prefix"
    elif source_strict and not all(
            item.name.startswith(KIMI_K3_DRAFT_PREFIX) for item in draft):
        reason = "unrecognized_draft_layer_prefix"
    elif source_strict and not all(
            item.name.startswith(KIMI_K3_TARGET_PREFIX) for item in mamba):
        reason = "unrecognized_mamba_layer_prefix"
    elif any(not item.is_full_attention for item in (*target, *draft)):
        reason = "attention_spec_is_not_full_attention"
    elif source_strict and any(not item.head_resolved
                               for item in (*target, *draft)):
        reason = "attention_spec_not_fully_resolved"
    elif any(not item.state_shapes or not item.state_dtypes
             for item in mamba):
        reason = "mamba_spec_not_fully_resolved"
    elif not _uniform_mamba(mamba):
        reason = "mamba_specs_not_uniform"
    elif len({item.page_size_bytes for item in records}) != 1:
        reason = "page_size_mismatch"
    elif len({item.block_size for item in (*target, *draft)}) != 1:
        reason = "attention_block_size_mismatch"
    else:
        mixed = [*target, *draft]
        causal = {item.noncausal_query_block for item in mixed}
        if len(causal) == 1:
            if not _uniform_attention(mixed):
                reason = "attention_spec_mismatch"
            else:
                attention_groups = 1
        else:
            # The source's uniform merge tries exactly one attention group;
            # it has no branch producing a causal target table plus a
            # non-causal draft table.  A useful two-table split therefore
            # stays an explicitly labeled analytical alternative.
            if not _uniform_attention(target):
                reason = "target_attention_specs_not_uniform"
            elif not _uniform_attention(draft):
                reason = "draft_attention_specs_not_uniform"
            else:
                if strict:
                    raise ModelingError(
                        "kimi_k3_dspark_grouping_invalid:"
                        "noncausal_split_is_not_source_uniform_grouping")
                return KimiK3DsparkGrouping(
                    fallback,
                    "noncausal_split_is_not_source_uniform_grouping",
                    len(target), len(draft), len(mamba), 2,
                    mode="analytical_noncausal_split", source_compatible=False)
    if reason is not None:
        return fail(reason)
    groups: list[tuple[str, ...]] = []
    if attention_groups == 1:
        groups.append(tuple(item.name for item in (*target, *draft)))
    else:
        groups.append(tuple(item.name for item in target))
        groups.append(tuple(item.name for item in draft))
    attention_size = len(groups[0])
    mamba_names = [item.name for item in mamba]
    group_count = ceil_div(len(mamba_names), attention_size)
    for index in range(group_count):
        groups.append(tuple(mamba_names[index::group_count]))
    return KimiK3DsparkGrouping(tuple(groups), None, len(target), len(draft),
                                len(mamba), attention_groups)


@typed_inputs
@dataclass(frozen=True)
class DsparkSwaFootprint:
    """Bounded draft SWA KV, rounded to KV blocks."""

    sequence_tokens: int
    swa_window_tokens: int
    extra_retained_tokens: int
    kv_block_size: int
    layers: int
    row_bytes_per_token: float
    capacity_padded: bool = True

    def __post_init__(self) -> None:
        _int("sequence_tokens", self.sequence_tokens, minimum=1)
        _int("swa_window_tokens", self.swa_window_tokens, minimum=1)
        _int("extra_retained_tokens", self.extra_retained_tokens, minimum=0)
        _int("kv_block_size", self.kv_block_size, minimum=1)
        _int("layers", self.layers, minimum=1)
        if strict_number("row_bytes_per_token", self.row_bytes_per_token,
                         reason="dspark") is None:
            raise ModelingError("dspark_row_bytes_invalid")
        _bool("capacity_padded", self.capacity_padded)

    @property
    def visible_tokens(self) -> int:
        return min(self.sequence_tokens,
                   self.swa_window_tokens + self.extra_retained_tokens)

    @property
    def retained_tokens(self) -> int:
        if not self.capacity_padded:
            return self.visible_tokens
        return ceil_div(self.visible_tokens, self.kv_block_size) * self.kv_block_size

    @property
    def bounded_bytes(self) -> float:
        return (float(self.retained_tokens * self.layers)
                * self.row_bytes_per_token)

    @property
    def full_bytes(self) -> float:
        tokens = (ceil_div(self.sequence_tokens, self.kv_block_size)
                  * self.kv_block_size if self.capacity_padded
                  else self.sequence_tokens)
        return float(tokens * self.layers) * self.row_bytes_per_token

    @property
    def saving_bytes(self) -> float:
        return self.full_bytes - self.bounded_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"visible_tokens": self.visible_tokens,
                "retained_tokens": self.retained_tokens,
                "bounded_bytes": self.bounded_bytes,
                "full_bytes": self.full_bytes,
                "saving_bytes": self.saving_bytes,
                "saving_ratio": (self.saving_bytes / self.full_bytes
                                 if self.full_bytes else 0.0)}


@typed_inputs
def draft_row_bytes_per_token(*, kv_heads: int, head_dim: int, tp: int,
                              dtype: str) -> float:
    """Local K+V row bytes: TP-sharded KV heads, both K and V planes."""
    _int("kv_heads", kv_heads, minimum=1)
    _int("head_dim", head_dim, minimum=1)
    _int("tp", tp, minimum=1)
    if kv_heads % tp and tp % kv_heads:
        # Neither a clean partition nor a replicated head count: a floored
        # partition is not a shard, so reject before arithmetic.
        raise ModelingError(
            f"dspark_draft_kv_heads_not_divisible_by_tp:{kv_heads}%{tp}")
    local = max(1, kv_heads // tp)
    return float(2 * local * head_dim) * dtype_elem_bytes(
        "dtype", dtype, reason="dspark")


@typed_inputs
def dspark_swa_footprint(*, sequence_tokens: int, swa_window_tokens: int,
                         extra_retained_tokens: int, kv_block_size: int,
                         layers: int, kv_heads: int, head_dim: int, tp: int,
                         dtype: str, mla_row_elements: int | None = None,
                         capacity_padded: bool = True) -> DsparkSwaFootprint:
    """Bound the draft SWA cache by its window, not by ``ISL + OSL``.

    An MLA draft row is its latent width (``kv_lora + rope``); a GQA draft row
    is ``2 * local_kv_heads * head_dim`` (both K and V, TP-sharded).
    """
    for name, value in (("kv_heads", kv_heads), ("head_dim", head_dim), ("tp", tp)):
        _int(name, value, minimum=1)
    if mla_row_elements is not None:
        _int("mla_row_elements", mla_row_elements, minimum=1)
        row = (float(mla_row_elements)
               * dtype_elem_bytes("dtype", dtype, reason="dspark"))
    else:
        row = draft_row_bytes_per_token(
            kv_heads=kv_heads, head_dim=head_dim, tp=tp, dtype=dtype)
    return DsparkSwaFootprint(
        sequence_tokens=sequence_tokens, swa_window_tokens=swa_window_tokens,
        extra_retained_tokens=extra_retained_tokens,
        kv_block_size=kv_block_size, layers=layers,
        row_bytes_per_token=row, capacity_padded=capacity_padded)


@typed_inputs
@dataclass(frozen=True)
class SwaBoundedReplayPolicy:
    """Target bounded replay versus the excluded DSpark draft cache."""

    architecture: str
    swa_bounded_replay: bool
    prefix_hit: int
    swa_window: int
    extra_retained: int = 0
    draft_cache_class: str | None = None
    is_draft_layer: bool = False

    def __post_init__(self) -> None:
        _text("architecture", self.architecture)
        _bool("swa_bounded_replay", self.swa_bounded_replay)
        _bool("is_draft_layer", self.is_draft_layer)
        _int("prefix_hit", self.prefix_hit, minimum=0)
        _int("swa_window", self.swa_window, minimum=0)
        _int("extra_retained", self.extra_retained, minimum=0)
        if self.draft_cache_class is not None:
            _text("draft_cache_class", self.draft_cache_class)
        if self.swa_bounded_replay and self.swa_window <= 0:
            raise ModelingError("dspark_bounded_replay_requires_window")

    @property
    def architecture_supported(self) -> bool:
        return self.architecture in BOUNDED_REPLAY_ARCHITECTURES

    @property
    def draft_excluded(self) -> bool:
        return bool(self.is_draft_layer
                    or self.draft_cache_class == DSPARK_DRAFT_SWA_CLASS)

    @property
    def replay_tokens(self) -> int:
        if (not self.swa_bounded_replay or self.draft_excluded
                or not self.architecture_supported):
            return 0
        return min(self.prefix_hit, self.swa_window)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        codes: list[str] = []
        if self.swa_bounded_replay and self.draft_excluded:
            codes.append("dspark_swa_bounded_replay_excluded")
        if self.swa_bounded_replay and not self.architecture_supported:
            codes.append("swa_bounded_replay_unsupported_architecture")
        return tuple(codes)

    def to_dict(self) -> dict[str, Any]:
        return {"architecture_supported": self.architecture_supported,
                "draft_excluded": self.draft_excluded,
                "replay_tokens": self.replay_tokens,
                "diagnostics": list(self.diagnostics)}


@typed_inputs
def swa_bounded_replay_policy(*, architecture: str, swa_bounded_replay: bool,
                              prefix_hit: int, swa_window: int,
                              extra_retained: int = 0,
                              draft_cache_class: str | None = None,
                              is_draft_layer: bool = False) -> dict[str, Any]:
    """Target replays ``min(L_hit, W)``; the DSpark draft never replays."""
    policy = SwaBoundedReplayPolicy(
        architecture=architecture, swa_bounded_replay=swa_bounded_replay,
        prefix_hit=prefix_hit, swa_window=swa_window,
        extra_retained=extra_retained, draft_cache_class=draft_cache_class,
        is_draft_layer=is_draft_layer)
    return policy.to_dict()


@typed_inputs
@dataclass(frozen=True)
class DeepseekV41DSparkSWACache:
    """Marker for the DSpark draft SWA cache; bounded replay is excluded."""

    bounded_replay: bool = False

    def supports_bounded_replay(self) -> bool:
        return False


get_kimi_k3_dspark_mixed_kv_cache_groups = kimi_k3_dspark_grouping

_DCP_EXPORTS = frozenset({"DcpDraftBlockCost",
                          "non_causal_block_drafter_cost"})
_PREFIX_EXPORTS = frozenset({
    "DSparkPrefixKeys", "DSparkPrefixPage", "DSparkPrefixPlan",
    "aux_feature_bytes", "dspark_prefix_keys", "dspark_prefix_page_bytes",
    "dspark_prefix_plan"})


def __getattr__(name: str) -> Any:
    """Lazily re-export the companion DCP-cost and prefix-page records."""
    if name in _DCP_EXPORTS:
        from mindie_sim import dspark_dcp_cost
        return getattr(dspark_dcp_cost, name)
    if name in _PREFIX_EXPORTS:
        from mindie_sim import dspark_prefix_cache
        return getattr(dspark_prefix_cache, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BOUNDED_REPLAY_ARCHITECTURES", "DSPARK_DRAFT_SWA_CLASS",
           "DSparkPrefixKeys", "DSparkPrefixPage", "DSparkPrefixPlan",
           "DeepseekV41DSparkSWACache", "DcpDraftBlockCost",
           "DsparkSwaFootprint", "KIMI_K3_DRAFT_PREFIX",
           "KIMI_K3_TARGET_PREFIX", "KimiK3DsparkGrouping", "KvSpecSignature",
           "SwaBoundedReplayPolicy", "aux_feature_bytes",
           "draft_row_bytes_per_token", "dspark_prefix_keys",
           "dspark_prefix_page_bytes", "dspark_prefix_plan",
           "dspark_swa_footprint", "get_kimi_k3_dspark_mixed_kv_cache_groups",
           "kimi_k3_dspark_grouping", "non_causal_block_drafter_cost",
           "swa_bounded_replay_policy"]
