"""Grouped main/small page-class pool ownership and net reconciliation.

[Role]
  Model the ``vllm-ascend`` hybrid cache-group allocation: one main page class
  shared by attention and recurrent scheduler groups, an optional small class
  (indexer/compressor-tail pages), the source's longest-consecutive-run
  recurrent grouping (GLM-Next) and the independent physical block demand of
  every scheduler group.  Baseline ledgers that already price state or
  attention pages are reconciled net-once.

[Invariants]
  * ``main_page_bytes = max(attn_page, every recurrent page)``; attention
    pages are padded up when state dominates, not only the other way round.
  * The GLM-Next recurrent groups are recovered from the longest consecutive
    recurrent run over the full model layer sequence (``12, 11, 11`` for the
    45-layer catalog), then projected onto a PP stage's owned layers.
  * Independent scheduler groups allocate independent physical block IDs.  The
    global ID domain is ``attention_blocks + tail_blocks +
    sum(recurrent_blocks)`` (GLM's compressor tail is one block per request);
    a PP stage projects only local slot ownership and charges
    ``local_slot_count * global_block_domain * page``.
  * The recurrent demand is ``(K + 1) * batch`` (none) or
    ``ceil(L / B) * batch`` (align), covering the linear-state plan.
  * ``physical_pool_bytes`` charges every global block ID's locally owned
    slots at the class-padded main page, never a per-layer sum.
  * Reconciliation never double counts: a replacement delta is
    ``max(0, modeled - baseline)`` for a conservative baseline.
  * All dimensions are strict integers; booleans, NaN and negatives reject.
[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    ceil_div, strict_bool, strict_int, strict_number,
)
from mindie_sim.schema.common import ModelingError

RECURRENT_KINDS = frozenset({"kda", "gdn", "mamba"})


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="kv_pool")


@typed_inputs
def pool_blocks(*, tokens: int, block_size: int, batch: int,
                cp: int = 1) -> int:
    """Per-rank attention pool blocks for every active sequence (DCP shard)."""
    _int("tokens", tokens, minimum=1)
    _int("block_size", block_size, minimum=1)
    _int("batch", batch, minimum=1)
    _int("cp", cp, minimum=1)
    return ceil_div(ceil_div(tokens, block_size), cp) * batch


@typed_inputs
def balanced_recurrent_groups(layer_indices: Sequence[int], *,
                              attention_group_size: int) -> tuple[tuple[int, ...], ...]:
    """Fewest balanced recurrent groups no larger than the attention group."""
    _int("attention_group_size", attention_group_size, minimum=1)
    indices = tuple(_int("recurrent_layer", index) for index in layer_indices)
    if len(set(indices)) != len(indices):
        raise ModelingError("kv_pool_duplicate_recurrent_layer")
    if not indices:
        return ()
    count = ceil_div(len(indices), attention_group_size)
    return tuple(indices[offset::count] for offset in range(count))


@typed_inputs
def longest_run_recurrent_groups(layer_kinds: Sequence[tuple[int, str]]) -> tuple[tuple[int, ...], ...]:
    """Source ``_group_glm5_next_mamba_layer_names`` over the full model.

    The longest consecutive recurrent run defines the stride at which the
    sorted recurrent layers are distributed; the 45-layer GLM catalog (three
    recurrent layers, one MLA layer, ...) yields groups ``(12, 11, 11)``.
    """
    if isinstance(layer_kinds, (str, bytes)) or not isinstance(
            layer_kinds, Sequence) or not layer_kinds:
        raise ModelingError("kv_pool_layer_sequence_invalid")
    is_recurrent: dict[int, bool] = {}
    for index, kind in layer_kinds:
        _int("layer_index", index)
        if type(kind) is not str or kind not in {"mla", "gqa", *RECURRENT_KINDS}:
            raise ModelingError("kv_pool_layer_kind_invalid")
        flag = kind in RECURRENT_KINDS
        previous = is_recurrent.setdefault(index, flag)
        if previous != flag:
            raise ModelingError(
                f"kv_pool_layer_cannot_be_both:layer{index}")
    max_run = 0
    run = 0
    previous: int | None = None
    for index in sorted(is_recurrent):
        consecutive = previous is not None and index == previous + 1
        if is_recurrent[index]:
            run = run + 1 if consecutive else 1
            max_run = max(max_run, run)
        else:
            run = 0
        previous = index
    if max_run == 0:
        raise ModelingError("kv_pool_no_recurrent_layer")
    recurrent = [index for index in sorted(is_recurrent) if is_recurrent[index]]
    return tuple(tuple(recurrent[offset::max_run])
                 for offset in range(max_run))


@typed_inputs
def project_groups(groups: Sequence[Sequence[int]],
                   owned: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Project global recurrent groups onto one PP stage's owned layer set."""
    owned_tuple = tuple(owned)
    owned_set = {_int("owned_layer", index) for index in owned_tuple}
    if len(owned_set) != len(owned_tuple):
        raise ModelingError("kv_pool_duplicate_owned_layer")
    projected: list[tuple[int, ...]] = []
    for group in groups:
        indices = tuple(_int("group_layer", index) for index in group)
        projected.append(tuple(index for index in indices
                               if index in owned_set))
    covered = {index for group in projected for index in group}
    if covered != owned_set:
        missing = sorted(owned_set - covered)
        raise ModelingError(f"kv_pool_recurrent_layer_not_grouped:{missing}")
    return tuple(projected)


@typed_inputs
@dataclass(frozen=True)
class GroupedPoolLayout:
    """One PP stage's main/small page-class physical pool.

    ``recurrent_groups`` holds one tuple of face bytes per recurrent scheduler
    group; ``recurrent_blocks`` holds that group's independent block demand.
    """

    attn_group_size: int
    attn_page_bytes: int
    recurrent_groups: tuple[tuple[int, ...], ...] = ()
    attention_blocks: int = 0
    recurrent_blocks: tuple[int, ...] = ()
    indexes_kv_by_block_stride: bool = True
    small_page_bytes: int = 0
    small_slot_count: int = 0
    declared_page_size_padded: int | None = None
    block_domain: int | None = None

    def __post_init__(self) -> None:
        _int("attn_group_size", self.attn_group_size, minimum=0)
        _int("attn_page_bytes", self.attn_page_bytes, minimum=0)
        _int("attention_blocks", self.attention_blocks, minimum=0)
        _int("small_page_bytes", self.small_page_bytes, minimum=0)
        _int("small_slot_count", self.small_slot_count, minimum=0)
        strict_bool("indexes_kv_by_block_stride",
                    self.indexes_kv_by_block_stride)
        if isinstance(self.recurrent_groups, (str, bytes)) or not isinstance(
                self.recurrent_groups, tuple):
            raise ModelingError("kv_pool_recurrent_groups_invalid")
        for group in self.recurrent_groups:
            if isinstance(group, (str, bytes)) or not isinstance(group, tuple):
                raise ModelingError("kv_pool_recurrent_groups_invalid")
            for page in group:
                _int("recurrent_page", page, minimum=0)
        if isinstance(self.recurrent_blocks, (str, bytes)) or not isinstance(
                self.recurrent_blocks, tuple):
            raise ModelingError("kv_pool_recurrent_blocks_invalid")
        for blocks in self.recurrent_blocks:
            _int("recurrent_blocks", blocks, minimum=0)
        if len(self.recurrent_blocks) != len(self.recurrent_groups):
            raise ModelingError("kv_pool_recurrent_block_group_mismatch")
        if self.attn_group_size == 0 and self.attn_page_bytes:
            raise ModelingError("kv_pool_attention_page_without_group")
        if self.attn_group_size and not self.attn_page_bytes:
            raise ModelingError("kv_pool_attention_group_without_page")
        if self.attn_group_size == 0 and not self.recurrent_groups \
                and self.small_slot_count == 0:
            raise ModelingError("kv_pool_has_no_layer")
        if self.small_slot_count and not self.small_page_bytes:
            raise ModelingError("kv_pool_small_slots_without_page")
        if self.small_page_bytes and not self.small_slot_count:
            raise ModelingError("kv_pool_small_page_without_slots")
        if self.declared_page_size_padded is not None:
            _int("declared_page_size_padded",
                 self.declared_page_size_padded, minimum=1)
            if self.declared_page_size_padded < self.local_main_page_bytes:
                raise ModelingError(
                    "kv_pool_page_size_padded_mismatch:"
                    f"{self.declared_page_size_padded}<"
                    f"{self.local_main_page_bytes}")
        if self.block_domain is not None:
            _int("block_domain", self.block_domain, minimum=0)

    @property
    def recurrent_group_count(self) -> int:
        return len(self.recurrent_groups)

    @property
    def recurrent_group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.recurrent_groups)

    @property
    def recurrent_layer_count(self) -> int:
        return sum(self.recurrent_group_sizes)

    @property
    def total_recurrent_blocks(self) -> int:
        return sum(self.recurrent_blocks)

    @property
    def blocks(self) -> int:
        """This stage's independent local block ownership (no tail group)."""
        return self.attention_blocks + self.total_recurrent_blocks

    @property
    def effective_blocks(self) -> int:
        """Global scheduler block-ID domain every owned slot is charged to.

        The source's ``num_blocks`` is global; a PP stage only projects slot
        ownership.  Recomputing it from the stage's own demands would shrink
        the address space and drop the compressor-tail group's global IDs.
        """
        return self.blocks if self.block_domain is None else self.block_domain

    @property
    def local_main_page_bytes(self) -> int:
        pages = [page for group in self.recurrent_groups for page in group]
        if self.attn_page_bytes:
            pages.append(self.attn_page_bytes)
        return max(pages) if pages else 0

    @property
    def main_page_bytes(self) -> int:
        """The shared page class; a projected stage keeps the global class."""
        if self.declared_page_size_padded is not None:
            return self.declared_page_size_padded
        return self.local_main_page_bytes

    @property
    def main_slot_count(self) -> int:
        sizes = self.recurrent_group_sizes
        return max((self.attn_group_size, *sizes)) if sizes else self.attn_group_size

    @property
    def slot_payloads(self) -> tuple[int, ...]:
        """Unpadded payload of every main slot (max of its co-located pages)."""
        payloads: list[int] = []
        for slot in range(self.main_slot_count):
            page = self.attn_page_bytes if slot < self.attn_group_size else 0
            for group in self.recurrent_groups:
                if slot < len(group):
                    page = max(page, group[slot])
            payloads.append(page)
        return tuple(payloads)

    @property
    def main_slot_bytes(self) -> int:
        """Class-padded main bytes for every global block ID."""
        return (self.effective_blocks * self.main_slot_count
                * self.main_page_bytes)

    @property
    def small_class_bytes(self) -> int:
        return (self.effective_blocks * self.small_slot_count
                * self.small_page_bytes)

    @property
    def physical_pool_bytes(self) -> int:
        return self.main_slot_bytes + self.small_class_bytes

    @property
    def attention_domain(self) -> int:
        """Attention block IDs charged at this stage (global when projected)."""
        return (self.attention_blocks if self.block_domain is None
                else self.block_domain)

    @property
    def bare_slot_bytes(self) -> int:
        attention = (self.attention_domain * self.attn_group_size
                     * self.attn_page_bytes)
        if self.block_domain is None:
            recurrent = sum(
                self.recurrent_blocks[index] * sum(group)
                for index, group in enumerate(self.recurrent_groups))
            return attention + recurrent + self.small_class_bytes
        return (self.block_domain * sum(self.slot_payloads)
                + self.small_class_bytes)

    @property
    def page_padding_bytes(self) -> int:
        """Bytes added by padding every main slot to the class maximum."""
        return self.physical_pool_bytes - self.bare_slot_bytes

    @property
    def layer_sum_bytes(self) -> int:
        """Per-layer sum at the same independent allocation domains.

        With a projected global block domain this is the no-slot-sharing
        layer sum; the difference from ``bare_slot_bytes`` is the source's
        cross-group slot sharing.  The legacy direct helper keeps them equal.
        """
        if self.block_domain is None:
            return self.bare_slot_bytes
        attention = (self.attention_domain * self.attn_group_size
                     * self.attn_page_bytes)
        recurrent = self.block_domain * sum(
            sum(group) for group in self.recurrent_groups)
        return attention + recurrent + self.small_class_bytes

    @property
    def slot_sharing_saving_bytes(self) -> int:
        """No cross-group reuse exists once block IDs are independent."""
        return max(0, self.layer_sum_bytes - self.bare_slot_bytes)

    @property
    def attention_uplift_bytes(self) -> int:
        """Attention pages raised to a state-dominated main class."""
        if self.main_page_bytes <= self.attn_page_bytes:
            return 0
        return (self.attention_domain * self.attn_group_size
                * (self.main_page_bytes - self.attn_page_bytes))

    @property
    def requires_padded_page_layout(self) -> bool:
        return bool(self.indexes_kv_by_block_stride and self.recurrent_groups
                    and self.main_page_bytes > 0)

    def reconcile(self, baseline_bytes: float) -> dict[str, float]:
        """Net-once replacement against a conservative baseline.

        The shared pool is a *replacement*: when the modeled pool is smaller
        than the per-layer baseline, the baseline is retained as a
        conservative reserve and the additive delta is zero.  Whatever is
        kept is reported explicitly so no byte is counted twice.
        """
        # Raw inputs only: booleans, strings, NaN and infinities are not a
        # baseline allocation and must not reach the delta arithmetic.
        declared = strict_number("baseline_bytes", baseline_bytes,
                                 minimum=0.0, reason="kv_pool")
        assert declared is not None
        baseline = declared
        modeled = float(self.physical_pool_bytes)
        return {"modeled_bytes": modeled, "baseline_bytes": baseline,
                "delta_bytes": max(0.0, modeled - baseline),
                "retained_baseline_bytes": max(0.0, baseline - modeled)}

    def to_dict(self) -> dict[str, Any]:
        return {"attn_group_size": self.attn_group_size,
                "attn_page_bytes": self.attn_page_bytes,
                "recurrent_layers": self.recurrent_layer_count,
                "recurrent_group_count": self.recurrent_group_count,
                "recurrent_group_sizes": list(self.recurrent_group_sizes),
                "recurrent_blocks": list(self.recurrent_blocks),
                "attention_blocks": self.attention_blocks,
                "main_page_bytes": self.main_page_bytes,
                "main_slot_count": self.main_slot_count,
                "small_page_bytes": self.small_page_bytes,
                "small_slot_count": self.small_slot_count,
                "blocks": self.blocks,
                "block_domain": self.effective_blocks,
                "local_blocks": self.blocks,
                "slot_payloads": list(self.slot_payloads),
                "physical_pool_bytes": self.physical_pool_bytes,
                "bare_slot_bytes": self.bare_slot_bytes,
                "layer_sum_bytes": self.layer_sum_bytes,
                "page_padding_bytes": self.page_padding_bytes,
                "attention_uplift_bytes": self.attention_uplift_bytes,
                "slot_sharing_saving_bytes": self.slot_sharing_saving_bytes,
                "requires_padded_page_layout":
                    self.requires_padded_page_layout,
                "indexes_kv_by_block_stride": self.indexes_kv_by_block_stride}


@typed_inputs
def grouped_pool_layout(*, attn_group_size: int, attn_page_bytes: int,
                        recurrent_groups: Sequence[Sequence[int]] | None = None,
                        attention_blocks: int = 0,
                        recurrent_blocks: Sequence[int] | None = None,
                        recurrent_pages: Sequence[int] | None = None,
                        blocks: int | None = None,
                        indexes_kv_by_block_stride: bool = True,
                        small_page_bytes: int = 0,
                        small_slot_count: int = 0,
                        declared_page_size_padded: int | None = None,
                        block_domain: int | None = None) -> GroupedPoolLayout:
    """Construct and validate one stage's grouped page-class pool.

    ``recurrent_groups`` is the source-exact grouping.  The legacy
    ``recurrent_pages``/``blocks`` form derives balanced groups and charges the
    same block demand to every group (kept only for direct helper callers).
    """
    if blocks is not None:
        _int("blocks", blocks, minimum=0)
    if recurrent_pages is not None:
        for page in recurrent_pages:
            _int("recurrent_page", page, minimum=0)
    if recurrent_groups is None:
        if recurrent_pages is None:
            recurrent_groups = ()
        else:
            if isinstance(recurrent_pages, (str, bytes)) or not isinstance(
                    recurrent_pages, Sequence):
                raise ModelingError("kv_pool_recurrent_pages_invalid")
            pages = tuple(_int("recurrent_page", page, minimum=0)
                          for page in recurrent_pages)
            if pages and attn_group_size <= 0:
                raise ModelingError("kv_pool_attention_group_required")
            recurrent_groups = (tuple(pages),) if pages else ()
            if blocks is not None and recurrent_blocks is None:
                recurrent_blocks = (blocks,) * len(recurrent_groups)
    if isinstance(recurrent_groups, (str, bytes)) or not isinstance(
            recurrent_groups, Sequence):
        raise ModelingError("kv_pool_recurrent_groups_invalid")
    if any(isinstance(group, (str, bytes)) or not isinstance(group, Sequence)
           for group in recurrent_groups):
        raise ModelingError("kv_pool_recurrent_groups_invalid")
    groups = tuple(tuple(_int("recurrent_page", page, minimum=0)
                         for page in group) for group in recurrent_groups)
    if recurrent_blocks is None:
        declared_blocks = (0 if blocks is None
                           else _int("blocks", blocks, minimum=0))
        recurrent_blocks = (declared_blocks,) * len(groups)
    if isinstance(recurrent_blocks, (str, bytes)) or not isinstance(
            recurrent_blocks, Sequence):
        raise ModelingError("kv_pool_recurrent_blocks_invalid")
    return GroupedPoolLayout(
        attn_group_size=attn_group_size, attn_page_bytes=attn_page_bytes,
        recurrent_groups=groups, attention_blocks=attention_blocks,
        recurrent_blocks=tuple(recurrent_blocks),
        indexes_kv_by_block_stride=indexes_kv_by_block_stride,
        small_page_bytes=small_page_bytes,
        small_slot_count=small_slot_count,
        declared_page_size_padded=declared_page_size_padded,
        block_domain=block_domain)


__all__ = ["GroupedPoolLayout", "RECURRENT_KINDS", "balanced_recurrent_groups",
           "grouped_pool_layout", "longest_run_recurrent_groups", "pool_blocks",
           "project_groups"]
