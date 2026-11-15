"""Execution-occupancy geometry for one physical native kernel launch.

[Role]
  Turn a normalized ABI invocation (declared tensor shapes/dtypes) plus the
  target die's declared resources into the *space* a kernel actually runs on:
  how many independent output blocks exist, how many on-chip units can serve
  them, how many columns of a row fit one UB segment, and how many rows one
  UB-resident tile can hold.  A native catalog needs this because a launch
  with ``B_active`` blocks cannot borrow the idle cores' bandwidth: the
  per-block service rate is ``die_rate / B_active``, not the whole-die peak.

[Invariants]
  * Geometry is derived only from declared shapes, dtypes, the declared UB
    capacity and the declared unit counts.  No shape, request id or measured
    duration selects a branch.
  * A block count is never widened past the physical unit count, and never
    below one.
  * On-chip tile rows are bounded by ``UB_BYTES / (live_buffers*row_bytes)``;
    a row wider than one vector segment is charged a whole extra segment, the
    same way ``ceil`` charges a partial Cube wave.
  * Request traffic that crosses HBM is reported in 128 B cachelines so a
    paged KV page tail pays its real burst, not a byte-rounded fraction.

[Timeouts & Locks]
  Pure bounded arithmetic; no waits, no locks, no caches.

[Side-Effect Policy]
  Returns immutable records.  No I/O, no device work, no global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

#: arch3510 UB capacity when a profile does not declare ``ub_bytes``.
ARCH3510_UB_BYTES = 253_952

#: One arch3510 AIV pair per AIC; the declared die carries 64 AIV / 32 AIC.
ARCH3510_AIV = 64

#: HBM/MTE burst granularity: a metadata or page access pays whole cachelines.
HBM_CACHELINE_BYTES = 128

#: Declared two-slot software pipeline (``SOFTWARE_PIPELINE_SLOTS``).  Rows
#: below this depth cannot hide a fill; rows above it reach steady state.
PIPELINE_SLOTS = 2

#: A row that exceeds one 12 KiB vector segment is processed as two segments;
#: the second one is a whole extra pass over the row, exactly as a K tail is a
#: whole extra Cube wave.  12 KiB is one 6144-element BF16 register segment.
VECTOR_SEGMENT_BYTES = 12_288

#: Elementwise owners split a row into 512-element column tiles.
COLUMN_TILE_ELEMENTS = 512


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def cacheline_bytes(amount: float) -> int:
    """Round a request traffic amount up to whole 128 B HBM cachelines."""
    if amount <= 0:
        return 0
    lines = int(math.ceil(float(amount) / HBM_CACHELINE_BYTES))
    return lines * HBM_CACHELINE_BYTES


def occupancy_fraction(blocks: int, units: int) -> float:
    """Fraction of the die's on-chip owners that can run one block each.

    ``1.0`` means every owner is busy and the whole-die rate is available;
    ``1/64`` means one owner runs while 63 idle owners' bandwidth is *not*
    borrowable by the active block.
    """
    blocks = _positive_int(blocks)
    units = _positive_int(units)
    if units < 1 or blocks < 1:
        return 1.0
    return min(1.0, float(blocks) / float(units))


def active_rate(rate: float, blocks: int, units: int) -> float:
    """Service rate actually reachable by ``blocks`` concurrent owners."""
    rate = float(rate)
    if rate <= 0:
        return 0.0
    fraction = occupancy_fraction(blocks, units)
    if fraction <= 0:
        return rate
    return rate * fraction


def vector_segments(row_bytes: float) -> int:
    """Whole vector register segments one physical row occupies."""
    row_bytes = float(row_bytes)
    if row_bytes <= 0:
        return 1
    return max(1, int(math.ceil(row_bytes / VECTOR_SEGMENT_BYTES)))


def gamma_preload_bytes(gamma_bytes: int, blocks: int) -> int:
    """MTE2 broadcast traffic of one resident vector every active block reads.

    A norm kernel keeps ``gamma`` in UB for its whole row tile, so each *active*
    block issues its own broadcast load of the vector in whole 128 B
    cachelines.  With ``B`` owners that is ``B * ceil128B(gamma)`` bytes, not a
    single per-call copy: on two-row blocks the vector is a third of the MTE2
    stream, while a 32-row block amortizes the same load over its tile.
    """
    gamma = _positive_int(gamma_bytes)
    blocks = _positive_int(blocks)
    if gamma < 1 or blocks < 1:
        return 0
    return blocks * cacheline_bytes(gamma)


@dataclass(frozen=True)
class KernelOccupancy:
    """The declared launch space of one physical native kernel."""

    schema: str = "mindie-sim/native-kernel-occupancy/1"
    rows: int = 0
    width: int = 0
    dtype_bytes: int = 2
    row_bytes: int = 0
    blocks: int = 0
    units: int = 0
    occupancy: float = 1.0
    rows_per_block: int = 0
    ub_bytes: int = ARCH3510_UB_BYTES
    live_buffers: int = 0
    ub_rows: int = 0
    tiles: int = 0
    columns_split: bool = False
    segments: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "rows": self.rows, "width": self.width,
            "dtype_bytes": self.dtype_bytes, "row_bytes": self.row_bytes,
            "blocks": self.blocks, "units": self.units,
            "occupancy": self.occupancy,
            "rows_per_block": self.rows_per_block,
            "ub_bytes": self.ub_bytes, "live_buffers": self.live_buffers,
            "ub_rows": self.ub_rows, "tiles": self.tiles,
            "columns_split": self.columns_split,
            "segments": self.segments,
        }


def row_norm_occupancy(
        rows: int, width: int, *, dtype_bytes: int = 2, live_buffers: int = 5,
        units: int = ARCH3510_AIV, ub_bytes: int = ARCH3510_UB_BYTES,
        columns_split: bool = False,
        column_tile: int = COLUMN_TILE_ELEMENTS) -> KernelOccupancy:
    """Launch geometry of one row-parallel or column-split elementwise kernel.

    Row-parallel (norm/fused) kernels parallelize over rows: one block owns
    whole rows, so ``blocks = min(units, rows)``.  Column-split elementwise
    kernels parallelize over ``column_tile``-element strips, so a single row
    still fills several owners.
    """
    rows = _positive_int(rows)
    width = _positive_int(width)
    dtype_bytes = max(1, _positive_int(dtype_bytes))
    if rows < 1 or width < 1:
        raise ValueError("native_occupancy_geometry_invalid")
    units = max(1, _positive_int(units))
    ub_bytes = max(1, _positive_int(ub_bytes))
    row_bytes = width * dtype_bytes
    if columns_split:
        column_tile = max(1, _positive_int(column_tile))
        strips = int(math.ceil(float(rows * width) / float(column_tile)))
        blocks = max(1, min(units, strips))
        rows_per_block = rows
    else:
        blocks = max(1, min(units, rows))
        rows_per_block = int(math.ceil(float(rows) / float(blocks)))
    live = max(1, _positive_int(live_buffers))
    ub_rows = max(1, ub_bytes // (live * row_bytes))
    tiles = max(1, int(math.ceil(float(rows_per_block) / float(ub_rows))))
    return KernelOccupancy(
        rows=rows, width=width, dtype_bytes=dtype_bytes,
        row_bytes=row_bytes, blocks=blocks, units=units,
        occupancy=occupancy_fraction(blocks, units),
        rows_per_block=rows_per_block, ub_bytes=ub_bytes,
        live_buffers=live, ub_rows=ub_rows, tiles=tiles,
        columns_split=bool(columns_split),
        segments=vector_segments(row_bytes))


def matrix_occupancy(
        m: int, n: int, *, tile_m: int, tile_n: int, units: int = 32,
        dtype_bytes: int = 2) -> KernelOccupancy:
    """Launch geometry of a tiled Cube kernel: one block per (M,N) tile."""
    m, n = _positive_int(m), _positive_int(n)
    tile_m = max(1, _positive_int(tile_m))
    tile_n = max(1, _positive_int(tile_n))
    if m < 1 or n < 1:
        raise ValueError("native_occupancy_matrix_geometry_invalid")
    units = max(1, _positive_int(units))
    tiles_m = int(math.ceil(float(m) / float(tile_m)))
    tiles_n = int(math.ceil(float(n) / float(tile_n)))
    blocks = max(1, min(units, tiles_m * tiles_n))
    return KernelOccupancy(
        rows=m, width=n, dtype_bytes=max(1, _positive_int(dtype_bytes)),
        row_bytes=n * max(1, _positive_int(dtype_bytes)), blocks=blocks,
        units=units, occupancy=occupancy_fraction(blocks, units),
        rows_per_block=tiles_m, ub_rows=tiles_m,
        tiles=tiles_m * tiles_n)


def attention_occupancy(
        batch: int, kv_heads: int, *, units: int = 32, visible_kv: int = 0,
        page_size: int = 128, head_dim: int = 128,
        dtype_bytes: int = 2) -> KernelOccupancy:
    """Launch geometry of one paged-FIA S1 kernel (one block per B x KV head)."""
    batch, kv_heads = _positive_int(batch), _positive_int(kv_heads)
    if batch < 1 or kv_heads < 1:
        raise ValueError("native_occupancy_attention_geometry_invalid")
    units = max(1, _positive_int(units))
    blocks = max(1, min(units, batch * kv_heads))
    page_size = max(1, _positive_int(page_size))
    head_dim = max(1, _positive_int(head_dim))
    visible = _positive_int(visible_kv)
    pages = max(1, int(math.ceil(float(visible) / page_size))) if visible else 0
    return KernelOccupancy(
        rows=batch, width=visible, dtype_bytes=max(1, _positive_int(dtype_bytes)),
        row_bytes=page_size * head_dim * max(1, _positive_int(dtype_bytes)),
        blocks=blocks, units=units,
        occupancy=occupancy_fraction(blocks, units),
        rows_per_block=kv_heads, ub_rows=pages, tiles=max(1, pages))


def occupancy_from_dict(payload: Mapping[str, Any] | None) -> KernelOccupancy | None:
    """Rebuild a geometry record without trusting foreign fields."""
    if not isinstance(payload, Mapping):
        return None
    try:
        return KernelOccupancy(**{
            key: payload[key] for key in KernelOccupancy.__dataclass_fields__
            if key in payload})
    except TypeError:
        return None
