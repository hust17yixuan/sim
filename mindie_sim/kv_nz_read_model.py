"""MTE2 ND-vs-NZ cacheline and DMA accounting for Fractal-NZ KV pages.

[Role]
  Price the strided per-head ND read and the tile-wise NZ read across whole
  physical pages, including partial tail pages and non-contiguous block IDs.

[Invariants]
  * One contiguous tile run per (head, tile, physical page): a run never
    crosses a page boundary, so two pages cost two separate runs.
  * Tokens must fit the layout's allocated pages, and selected block IDs must
    be in range and distinct.
  * ND issues one DMA per (token, head); NZ issues one per (head, tile, page).
  * All arithmetic is integer-exact; booleans and NaN reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.kv_layout_physics import FRACTAL_NZ_C0_BYTES, KvNzLayout
from mindie_sim.nr_s7_validate import (ceil_div, positive_rate, strict_int,
                                       strict_number)
from mindie_sim.schema.common import ModelingError


def _int(name, value, *, minimum=0):
    from mindie_sim.nr_s7_validate import strict_int
    return strict_int(name, value, minimum=minimum, reason="kv_layout")

DEFAULT_CACHELINE_BYTES = 256


@typed_inputs
@dataclass(frozen=True)
class Mte2ReadModel:
    """One strided head-wise ND read versus per-page tile-wise NZ runs."""

    tokens: int
    logical_heads: int
    head_bytes: float
    cacheline_bytes: int
    pages_used: int
    nd_transactions: int
    nz_transactions: int
    nd_dma_ops: int
    nz_dma_ops: int
    useful_bytes: float
    nd_fetched_bytes: float
    nz_fetched_bytes: float
    nd_us: float | None = None
    nz_us: float | None = None

    @property
    def nd_efficiency(self) -> float:
        return self.useful_bytes / self.nd_fetched_bytes if self.nd_fetched_bytes else 0.0

    @property
    def nz_efficiency(self) -> float:
        return self.useful_bytes / self.nz_fetched_bytes if self.nz_fetched_bytes else 0.0

    @property
    def fetched_reduction(self) -> float:
        return (self.nd_fetched_bytes / self.nz_fetched_bytes
                if self.nz_fetched_bytes else 0.0)

    @property
    def dma_op_reduction(self) -> float:
        return (self.nd_dma_ops / self.nz_dma_ops
                if self.nz_dma_ops else 0.0)

    @property
    def speedup(self) -> float | None:
        if self.nd_us is None or self.nz_us is None or self.nz_us <= 0:
            return None
        return self.nd_us / self.nz_us

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "logical_heads": self.logical_heads,
                "head_bytes": self.head_bytes,
                "cacheline_bytes": self.cacheline_bytes,
                "pages_used": self.pages_used,
                "nd_transactions": self.nd_transactions,
                "nz_transactions": self.nz_transactions,
                "nd_dma_ops": self.nd_dma_ops,
                "nz_dma_ops": self.nz_dma_ops,
                "useful_bytes": self.useful_bytes,
                "nd_fetched_bytes": self.nd_fetched_bytes,
                "nz_fetched_bytes": self.nz_fetched_bytes,
                "nd_efficiency": self.nd_efficiency,
                "nz_efficiency": self.nz_efficiency,
                "fetched_reduction": self.fetched_reduction,
                "dma_op_reduction": self.dma_op_reduction,
                "nd_us": self.nd_us, "nz_us": self.nz_us,
                "speedup": self.speedup}


@typed_inputs
def mte2_read_model(layout: KvNzLayout, *, tokens: int, local_kv_heads: int,
                    cacheline_bytes: int = DEFAULT_CACHELINE_BYTES,
                    block_ids: Sequence[int] | None = None,
                    mte2_bytes_per_us: float | None = None,
                    dma_setup_us: float = 0.0) -> Mte2ReadModel:
    """Model cacheline waste and DMA ops across whole physical pages."""
    if not isinstance(layout, KvNzLayout):
        raise ModelingError("kv_layout_mte2_layout_invalid")
    _int("tokens", tokens, minimum=1)
    _int("local_kv_heads", local_kv_heads, minimum=1)
    _int("cacheline_bytes", cacheline_bytes, minimum=1)
    if local_kv_heads > layout.num_kv_heads:
        raise ModelingError(
            "kv_layout_local_heads_exceed_global:"
            f"{local_kv_heads}>{layout.num_kv_heads}")
    setup = strict_number("dma_setup_us", dma_setup_us, reason="kv_layout")
    assert setup is not None
    bandwidth = (positive_rate("mte2_bytes_per_us", mte2_bytes_per_us,
                               reason="kv_layout")
                 if mte2_bytes_per_us is not None else None)
    pages_used = ceil_div(tokens, layout.block_size)
    if pages_used > layout.num_blocks:
        raise ModelingError(
            "kv_layout_mte2_tokens_exceed_capacity:"
            f"{tokens}>{(layout.num_blocks * layout.block_size)}")
    if block_ids is not None:
        if (not isinstance(block_ids, Sequence)
                or isinstance(block_ids, (str, bytes))
                or len(block_ids) != pages_used):
            raise ModelingError("kv_layout_mte2_block_ids_invalid")
        seen: set[int] = set()
        for block in block_ids:
            _int("block_id", block, minimum=0)
            if block >= layout.num_blocks:
                raise ModelingError(f"kv_layout_mte2_block_id_out_of_range:{block}")
            seen.add(block)
        if len(seen) != pages_used:
            raise ModelingError("kv_layout_mte2_block_ids_not_distinct")
    plane = layout.nope
    tile_bytes = FRACTAL_NZ_C0_BYTES
    useful = (float(tokens * local_kv_heads * plane.head_size)
              * plane.element_bytes)
    nd_ops = tokens * local_kv_heads
    nd_transactions = nd_ops * ceil_div(int(plane.head_size
                                            * plane.element_bytes),
                                        cacheline_bytes)
    nd_fetched = float(nd_transactions * cacheline_bytes)
    nz_transactions = 0
    for page in range(pages_used):
        run = min(layout.block_size, tokens - page * layout.block_size)
        run_bytes = run * tile_bytes
        nz_transactions += (local_kv_heads * plane.tiles_per_head
                            * ceil_div(run_bytes, cacheline_bytes))
    nz_ops = local_kv_heads * plane.tiles_per_head * pages_used
    nz_fetched = float(nz_transactions * cacheline_bytes)
    nd_us = nz_us = None
    if bandwidth is not None:
        nd_us = max(nd_fetched / bandwidth, nd_ops * setup)
        nz_us = max(nz_fetched / bandwidth, nz_ops * setup)
    return Mte2ReadModel(
        tokens=tokens, logical_heads=local_kv_heads,
        head_bytes=plane.head_size * plane.element_bytes,
        cacheline_bytes=cacheline_bytes, pages_used=pages_used,
        nd_transactions=nd_transactions, nz_transactions=nz_transactions,
        nd_dma_ops=nd_ops, nz_dma_ops=nz_ops, useful_bytes=useful,
        nd_fetched_bytes=nd_fetched, nz_fetched_bytes=nz_fetched,
        nd_us=nd_us, nz_us=nz_us)



__all__ = ["Mte2ReadModel", "mte2_read_model"]
