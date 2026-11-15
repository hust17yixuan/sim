"""Declared native-kernel clock physics for every Ascend die (A2/A3/A5).

[Role] Turn declared tensor geometry and pipe work into elapsed pipe time for
  the native ACLNN leaves and catalogs.  Cube kernel laws live in
  ``native_cube_physics`` and are re-exported here alongside the AIV kernel laws
  (``RmsNorm``/``DynamicMxQuant``/``SwiGlu``/elementwise/vector/routing).
  Every pipe time is a declared byte/op quantity divided by the *selected die's*
  sustained rate from ``native_clock_rates``.
[Invariants]
  * Continuous/tile-quantised geometry laws only.  No shape list, ``case_id`` /
    ``cohort_id`` selector, piecewise knot table, ``fixed_us`` or observed
    duration is stored or read.
  * Fewer active owners never lend their bandwidth; private L1/L2 port streams
    scale with active owners, and shared HBM/DMA streams ramp with concurrency.
[Timeouts & Locks] Pure bounded arithmetic; no waits or locks.
[Side-Effect Policy] Read-only pure functions; no I/O, global mutation or
  device access.
"""
from __future__ import annotations

from typing import Any, Mapping

from .native_clock_rates import (  # noqa: F401
    A5_RATES, CUBE_BF16, CUBE_FP8, CUBE_FP32, CUBE_INT8, CUBE_SUSTAINED,
    DISPATCH_MS, EXTRAPOLATED_FLOOR_MS, FIXPIPE_RATE, HBM_RATE, L2_BYTES,
    LAUNCH_FLOOR_MS, MTE2_AIV_RATE, MTE2_EFFICIENCY, MTE2_RATE, MTE3_RATE,
    NativeRates, SCALAR_ISSUE_RATE, UB_BYTES, VECTOR_RATE, _field,
    declared_clock_hz, dtype_bytes, rates_from_supply,
)
from .native_cube_physics import (  # noqa: F401
    AIC_STREAM_HALF, CACHELINE, CUBE_K_BF16, CUBE_K_INT8,
    EXECUTION_SCOPE_FACTORS, FIXPIPE_TILE_MS, FRACTAL_MN, HBM_STREAM_SHARE,
    KernelTimes, L1_BYTES, L2_PORT_BYTES_PER_CYCLE, L2_REREAD_GAIN, MACRO_K,
    MACRO_M, MACRO_N, MATRIX_LAUNCH_HANDOFF_MS, MATRIX_QUANT_SCALAR_SHARE,
    MATRIX_SCALAR_BASE_MS, MATRIX_SCALAR_EXPOSED, MATRIX_SCALAR_K_CHUNK_MS,
    MATRIX_SCALAR_PANEL_LOG_MS, MATRIX_SCALAR_PANEL_MS, MTE2_STREAM_KNEE_BYTES,
    SPLIT_K_MAX, SPLIT_K_MAX_OUTPUT_BYTES, SPLIT_K_MIN_K, _ceil_div, _log2,
    _matrix_l1_and_hbm_passes, matrix_kind, matrix_mte2_bytes, matrix_tile_mn,
    matrix_times, split_k_for,
)
from .native_occupancy import PIPELINE_SLOTS, row_norm_occupancy

AIV_MTE2_START_MS = 0.00022
AIV_MTE3_START_MS = 0.00012
AIV_VECTOR_START_MS = 0.00022
AIV_MTE3_START_BY_CLASS = {"row_norm": 0.00020, "fused_row_norm": 0.00052}
AIV_SCALAR_BASE_MS = {
    "row_norm": 0.00048, "fused_row_norm": 0.00120, "elementwise": 0.00008,
    "strided_slice": 0.00068, "quantize": 0.00012, "swiglu": 0.00125,
    "vector": 0.00088, "hc_post": 0.00050, "routing": 0.00230,
    "gating": 0.00148, "finalize": 0.00210, "topk": 0.00112,
    "quant_row": 0.00360, "recurrent": 0.00300, "convolution": 0.00045,
}
AIV_SCALAR_BATCH_LOG_MS = 0.00026
AIV_BLOCK_DRAIN_MS = 0.00000
AIV_SCATTER_SLOT_MS = 0.0000082
AIV_STREAM_EFFICIENCY = {
    "recurrent": 0.420, "convolution": 0.195, "swiglu": 0.68, "vector": 0.68,
    "hc_post": 0.40, "strided_slice": 0.83, "quantize": 0.34,
}
DMA_HALF_BLOCKS = 9.0
VECTOR_OPS_PER_ELEMENT = {
    "row_norm": 7.0, "fused_row_norm": 9.0, "elementwise": 5.0,
    "quantize": 9.0, "vector": 7.5,
}
AIV_VECTOR_EXPOSED = 0.92
AIV_SCALAR_HIDDEN_BY_LOAD = 0.565

GROUP_SCALAR_MS = 0.0000128
GROUP_SCALAR_WALK_MS = 0.000140


def catalog_body_share(times: KernelTimes, rates: NativeRates) -> KernelTimes:
    """Express a local-rate body in the Engine's shared copy-owner time base."""
    KernelTimes.__post_init__(times)
    local, shared = float(rates.mte2_aic), float(rates.mte2_aiv)
    if local <= 0 or shared <= 0 or abs(local - shared) < 1.0:
        return times
    ratio = shared / local
    body = tuple((pipe, float(value) * ratio if pipe == "mte2" else float(value))
                 for pipe, value in times.body)
    return KernelTimes(prologue_ms=float(times.prologue_ms), body=body,
                       tail=tuple((pipe, float(value)) for pipe, value in times.tail),
                       split_k=times.split_k)


def aiv_times(*, kernel_class: str, rows: int, width: int,
              read_bytes: float, write_bytes: float, blocks: int,
              units: int, vector_ops: float | None = None,
              execution_scope: str = "serving",
              scatter_slots: int = 0,
              dtype_bytes: int = 2, ub_bytes: int | None = None,
              rates: NativeRates | None = None) -> KernelTimes:
    """Declared AIV launch: dispatch + scalar prologue + body + store drain."""
    r = rates or A5_RATES
    rows, width = max(1, int(rows)), max(1, int(width))
    units = max(1, int(units))
    blocks = max(1, min(units, int(blocks)))
    vec_share = float(blocks) / float(units)
    dma_share = float(blocks) / (float(blocks) + DMA_HALF_BLOCKS)
    ops = (float(vector_ops) if vector_ops is not None else
           float(rows) * float(width) * VECTOR_OPS_PER_ELEMENT.get(kernel_class, 7.0))
    batch = max(1, _ceil_div(rows, blocks))
    scalar_ms = (AIV_SCALAR_BASE_MS.get(kernel_class, AIV_SCALAR_BASE_MS["vector"])
                 + AIV_SCALAR_BATCH_LOG_MS * _log2(1.0 + blocks))
    if kernel_class in {"swiglu", "vector"} and rows == 1 and blocks <= 4:
        scalar_ms += 0.00110 if blocks <= 2 else 0.00055
    if kernel_class == "quantize":
        scalar_ms = AIV_SCALAR_BASE_MS["quantize"] + 0.00040 * _log2(1.0 + blocks)
        if blocks > 8:
            scalar_ms += 0.00012 * _log2(float(blocks) / 8.0)
    elif kernel_class == "row_norm":
        if width > 512 and width % 1024 != 0:
            scalar_ms += 0.00138
        if blocks > 8:
            scalar_ms += 0.00050 * _log2(float(blocks) / 8.0)
        if execution_scope == "isolated" and batch == 1 and blocks > 4:
            scalar_ms += 0.00048 * min(1.0, float(blocks - 4) / 12.0)
        if blocks > 48:
            scalar_ms += (0.00150 if execution_scope == "isolated" else 0.00250) * float(blocks - 48) / 16.0
            if execution_scope == "isolated" and batch == 1:
                scalar_ms += 0.00120 * float(blocks - 48) / 16.0
    elif kernel_class == "elementwise" and execution_scope == "isolated":
        if blocks > 32 and not r.measured_floors:
            scalar_ms += 0.00105 * float(blocks - 32) / 32.0
        elif r.measured_floors and float(read_bytes) <= 1.2 * float(write_bytes):
            scalar_ms += (0.00105 * float(blocks) / 64.0
                          + 0.00105 * max(0.0, float(blocks - 32)) / 32.0)
    row_loop_ms = 0.0
    if kernel_class in {"elementwise", "row_norm"} and batch > 1:
        row_loop_ms = 0.00009 * _log2(float(batch))
        scalar_ms += row_loop_ms
    elif kernel_class == "topk" and batch > 1:
        scalar_ms += 0.00036 + (0.00050 + 0.00011 * _log2(max(16.0, float(width)) / 16.0)) * float(batch - 1)
    if scatter_slots:
        scalar_ms += float(scatter_slots) * AIV_SCATTER_SLOT_MS
    scope = EXECUTION_SCOPE_FACTORS.get(execution_scope, 1.0)
    scalar_ms *= r.scalar_scale() * scope
    efficiency = float(AIV_STREAM_EFFICIENCY.get(kernel_class, 1.0))
    if r.mte2_aiv > r.hbm * 1.15 and kernel_class in {"swiglu", "vector"}:
        efficiency = 1.0
    total_rw = float(read_bytes) + float(write_bytes)
    l2_hot = execution_scope == "isolated" and total_rw <= 0.72 * float(r.l2_bytes)
    hbm_share = 0.83 if (float(read_bytes) >= 1.8 * float(write_bytes) > 0.0 and kernel_class == "elementwise") else 0.88
    hbm_stream_ms = 0.0 if l2_hot else (
        total_rw / max(1.0, r.hbm * hbm_share * dma_share * efficiency) * 1000.0)
    shared_l2 = l2_hot and (not r.measured_floors) and total_rw > 16.0e6
    a5_large_l2 = l2_hot and r.measured_floors and total_rw > 20.0e6
    l2_read_bytes = total_rw if shared_l2 else float(read_bytes)
    mte2_ms = AIV_MTE2_START_MS + max(
        l2_read_bytes / max(1.0, r.mte2_aiv * dma_share * efficiency) * 1000.0,
        hbm_stream_ms)
    vector_ms = AIV_VECTOR_START_MS + ops / max(1.0, r.vector * vec_share) * 1000.0
    store_raw_ms = float(write_bytes) / max(1.0, r.mte3 * dma_share * efficiency) * 1000.0
    drain_waves = float(batch) if batch > 1 and kernel_class not in {"recurrent", "convolution"} else 1.0
    store_body = (0.0 if hbm_stream_ms > 0 and drain_waves > 1.0
                  else (0.75 * store_raw_ms if shared_l2
                        else (0.65 * store_raw_ms if a5_large_l2 else store_raw_ms / drain_waves)))
    store_ms = AIV_MTE3_START_BY_CLASS.get(kernel_class, AIV_MTE3_START_MS) + store_body
    if kernel_class == "row_norm" and batch > 1:
        occupancy = row_norm_occupancy(
            rows, width, dtype_bytes=dtype_bytes, live_buffers=3,
            units=units, ub_bytes=ub_bytes or r.ub_bytes)
        if min(PIPELINE_SLOTS, occupancy.ub_rows) > 1:
            # Continuation bookkeeping overlaps earlier rows' DMA/vector
            # service, less the load window already credited to scalar work.
            # Launch/scalar-base and pipe fill/drain remain once per kernel.
            # Using the body window also prevents a faster vector pipe from
            # losing overlap while the same DMA work still limits that row.
            load_ms = mte2_ms - AIV_MTE2_START_MS
            compute_ms = vector_ms - AIV_VECTOR_START_MS
            preceding_body_ms = (max(load_ms, compute_ms)
                                 - AIV_SCALAR_HIDDEN_BY_LOAD * load_ms) * (
                                     float(batch - 1) / float(batch))
            scalar_ms -= min(row_loop_ms * r.scalar_scale() * scope,
                             preceding_body_ms)
    if execution_scope == "isolated":
        if not r.measured_floors:
            is_binary_ew = kernel_class == "elementwise" and float(read_bytes) >= 1.5 * float(write_bytes)
            launch_scale = ((0.68 if blocks >= 3 else 0.82)
                            if kernel_class in {"swiglu", "vector"} and rows <= 4
                            else (0.38 if (is_binary_ew and blocks <= 4)
                                  else (0.275 if blocks <= 8 else 0.38)))
        elif kernel_class == "elementwise":
            unary_bump = 0.13 if float(read_bytes) <= 1.2 * float(write_bytes) else 0.0
            launch_scale = 0.56 + unary_bump + 0.12 * min(1.0, float(blocks) / 64.0)
        elif kernel_class == "row_norm":
            launch_scale = 0.84 + 0.26 * min(1.0, float(blocks) / 16.0)
        else:
            launch_scale = 0.82 + 0.18 * min(1.0, float(blocks) / 24.0)
    else:
        launch_scale = 1.0
    launch_ms = r.launch(kernel_class) * launch_scale
    grid_sync_ms = (0.000105 * float(blocks) / 16.0 * r.scalar_scale()
                    if execution_scope == "isolated" and blocks > 1 else 0.0)
    if execution_scope == "isolated" and r.arch == "A3" and blocks > r.aiv // 2:
        grid_sync_ms += 0.00038 * r.scalar_scale()
    return KernelTimes(
        prologue_ms=(launch_ms + grid_sync_ms + AIV_BLOCK_DRAIN_MS + max(
            0.0, scalar_ms - AIV_SCALAR_HIDDEN_BY_LOAD * mte2_ms)),
        body=(("mte2", max(1e-6, mte2_ms - AIV_MTE2_START_MS)), ("vector", vector_ms)),
        tail=(("mte3", store_ms),))


def aiv_launch_us(*, kernel_class: str, rows: int, width: int,
                  read_bytes: float, write_bytes: float,
                  execution_scope: str = "serving",
                  dtype_bytes: int = 2, ub_bytes: int | None = None,
                  block_num: int | None = None,
                  vector_ops: float | None = None,
                  scatter_slots: int = 0,
                  rates: NativeRates | None = None) -> float:
    """Elapsed AIV launch from the declared die law (microseconds)."""
    r = rates or A5_RATES
    live = {"row_norm": 3, "fused_row_norm": 5, "elementwise": 3}.get(
        kernel_class, 3)
    occupancy = row_norm_occupancy(
        max(1, int(rows)), max(1, int(width)),
        dtype_bytes=max(1, int(dtype_bytes)), live_buffers=live,
        units=r.aiv, ub_bytes=int(ub_bytes or r.ub_bytes),
        columns_split=kernel_class == "elementwise")
    blocks = occupancy.blocks
    if isinstance(block_num, int) and block_num > 0:
        blocks = min(occupancy.units, block_num)
    times = aiv_times(
        kernel_class=kernel_class, rows=rows, width=width,
        read_bytes=read_bytes, write_bytes=write_bytes, blocks=blocks,
        units=occupancy.units, execution_scope=execution_scope, rates=r,
        vector_ops=vector_ops, scatter_slots=scatter_slots,
        dtype_bytes=dtype_bytes, ub_bytes=ub_bytes)
    return times.total_ms() * 1000.0


def group_scalar_ms(*, groups: int, rates: NativeRates | None = None) -> float:
    """Declared per-group AIC tiling setup of one grouped matmul launch."""
    r = rates or A5_RATES
    return (max(0, int(groups)) * GROUP_SCALAR_MS
            * (r.scalar_scale() ** 0.5))


def declared_block_num(conditions: Mapping[str, Any] | None) -> int | None:
    """The captured launch grid, when the caller declares it."""
    try:
        value = int((conditions or {}).get("block_num"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def declared_execution_scope(conditions: Mapping[str, Any] | None) -> str:
    value = str((conditions or {}).get("execution_scope") or "serving")
    return value if value in EXECUTION_SCOPE_FACTORS else "serving"
