"""Physical stage builders of one native kernel launch (per selected die).

[Role] Translate a declared kernel geometry into the ordered physical stages a
  device launch executes: the Cube macro-tile body with its scalar prologue and
  Split-K reduction, the paged-attention walk/stream/Cube body in its declared
  occupancy regime, and the AIV dispatch/scalar/body/store chain.
[Invariants]  Every time is a declared byte/op quantity divided by the selected
  die's sustained rate (``KernelSupply``); no shape list, cohort id, fitted
  per-shape factor or observed duration is stored.  Idle owners never lend
  their bandwidth.
[Timeouts & Locks]  Pure bounded arithmetic; no waits or locks.
[Side-Effect Policy]  Returns stage drafts only; no I/O or global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError

from . import native_clock_physics as _clock
from .native_catalog_builder import (
    LIVE_BUFFERS, KernelSupply, StageDraft, clock_stage, matrix_occupancy,
    row_norm_occupancy,
)

#: Declared S1 scalar page walk of one paged-attention launch: one fixed
#: page-table/softmax prologue plus one walk per visible KV token.
ATTENTION_WALK_BASE_MS = 0.0065
ATTENTION_WALK_MS_PER_KV_TOKEN = 3.3e-7
#: Visible page group below which a paged launch is latency-bound (its walk,
#: load and Cube region do not overlap) rather than throughput-bound.
ATTENTION_LATENCY_PAGES = 8


def _validated_catalog_times(times: _clock.KernelTimes,
                             rates: _clock.NativeRates) -> _clock.KernelTimes:
    """Enforce elapsed-service validity before and after shared-rate conversion."""
    _clock.KernelTimes.__post_init__(times)
    times = _clock.catalog_body_share(times, rates)
    _clock.KernelTimes.__post_init__(times)
    return times


def matrix_drafts(
        *, m: int, n: int, k: int, tile: Sequence[int], supply: KernelSupply,
        quantized: bool, read_bytes: float, write_bytes: float,
        dtype_bytes: int = 2, orientation_hbm: float = 0.0,
        vector_ops: float = 0.0, kind: str | None = None,
        weight_layout: str = "ND", out_dtype_bytes: int = 2,
        execution_scope: str = "serving",
        block_num: int | None = None
) -> list[StageDraft]:
    """Cube launch: one concurrent body + the declared launch hand-off.

    A real Cube kernel issues its MTE2 operand streams, Cube MMAD waves and
    Fixpipe drain concurrently inside one launch, and its AIC scalar tiling
    loop issues tile descriptors while those pipes run.  The captured launch
    grid (``block_num``) is the declared number of resident AIC owners; the
    output-panel grid is only a lower bound.  A narrow output panel that leaves
    AIC owners idle splits the K loop and adds one workspace reduction launch.
    All rates come from the selected die's supply.
    """
    if len(tuple(tile)) != 3:
        raise ModelingError("native_catalog_tile_invalid")
    if min(m, n, k) < 1:
        raise ModelingError("native_catalog_geometry_invalid")
    material = _clock.matrix_kind(
        quantized=quantized, dtype_bytes=dtype_bytes, weight_dtype=kind)
    width = _clock.dtype_bytes(material, default=max(1, int(dtype_bytes)))
    act_bytes = float(m) * float(k) * float(width)
    weight_bytes = max(float(n) * float(k) * float(width),
                       float(read_bytes) - act_bytes)
    units = supply.units_of("fixpipe") or supply.units_of("vector") or 32
    rates = _clock.rates_from_supply(supply)
    tile_m, tile_n = _clock.matrix_tile_mn(
        m=m, n=n, units=units, kind=material, weight_layout=weight_layout,
        arch=rates.arch, k=k)
    occupancy = matrix_occupancy(
        m, n, tile_m=tile_m, tile_n=tile_n, units=units,
        dtype_bytes=width)
    # ``block_num`` is the captured AIC grid; when declared it bounds the
    # owners available for spatial panels and Split-K reduction.
    declared = _clock.declared_block_num({"block_num": block_num})
    avail_owners = (max(1, min(units, declared))
                    if declared is not None else max(1, min(units, occupancy.blocks
                    if k < _clock.SPLIT_K_MIN_K else units)))
    split_k = _clock.split_k_for(
        m=m, n=n, k=k, owners=avail_owners, kind=material,
        weight_layout=weight_layout, out_dtype_bytes=out_dtype_bytes,
        units=units, arch=rates.arch)
    eff_tiles = occupancy.tiles * split_k
    if (material == "bf16" and split_k == 1 and k >= 1024
            and occupancy.tiles < max(1, avail_owners // 2) and tile_n > 64):
        sub_n = (128 if (rates.arch == "A3" and n >= 768)
                 else (64 if n <= 1536 else 128))
        eff_tiles = max(eff_tiles, _clock._ceil_div(m, tile_m) * _clock._ceil_div(n, sub_n))
    owners = (avail_owners if declared is not None
              else max(1, min(avail_owners, max(occupancy.blocks, eff_tiles))))
    times = _validated_catalog_times(_clock.matrix_times(
        m=m, n=n, k=k, read_bytes=float(read_bytes),
        write_bytes=float(write_bytes), weight_bytes=weight_bytes,
        act_bytes=act_bytes, out_bytes=float(write_bytes),
        owners=owners, units=units, kind=material, rates=rates,
        split_k=split_k, weight_layout=weight_layout,
        out_dtype_bytes=out_dtype_bytes,
        execution_scope=execution_scope,
        tile=tuple(int(x) for x in tile)), rates)
    drafts = [StageDraft(label="native:cube_prologue",
                         latency_ms=float(times.prologue_ms), role="prologue"),
              clock_stage(supply, {pipe: float(time) for pipe, time in times.body},
                          label="native:cube_body",
                          role="compute")]
    # Tail entries are serial stages, even when consecutive entries use one pipe.
    for index, (pipe, milliseconds) in enumerate(times.tail):
        milliseconds = float(milliseconds)
        if milliseconds == 0:
            continue
        if index == 0 and times.split_k > 1:
            label, role = "native:splitk_reduce", "reduce"
        elif pipe in {"fixpipe", "mte3"}:
            label, role = f"native:cube_drain:{index}", "store"
        else:
            label, role = f"native:cube_epilogue:{index}", "epilogue"
        drafts.append(clock_stage(
            supply, {pipe: milliseconds}, label=label, role=role))
    return drafts


def paged_attention_drafts(
        *, batch: int, kv_heads: int, visible_kv: int, page_size: int,
        head_dim: int, read_bytes: float, write_bytes: float,
        supply: KernelSupply, dtype_bytes: int = 2,
        orientation_hbm: float = 0.0, vector_ops: float = 0.0,
        head_token_pairs: int = 0
) -> list[StageDraft]:
    """Paged-attention launch in one of two declared occupancy regimes.

    * A **paged tail** (a handful of visible pages) is latency-bound: the page
      walk, the per-page KV load and the Cube region are serialised around one
      launch, so their declared times add.
    * A **full-capacity paged launch** is throughput-bound: each owner walks one
      sequence's page table and streams its own KV slice while the Cube
      accumulates QK/PV, so the body is the maximum of the walk, the per-owner
      stream and the Cube MAC waves.

    The regime boundary is the declared page group (8 pages); it is an
    occupancy statement, not a shape selector.
    """
    units = max(1, int(supply.units_of("mte2_aic") or 32))
    blocks = max(1, min(units, int(batch) * int(kv_heads)))
    mte2_rate = float(supply.rate("mte2_aic") or 4.224e9)
    mte3_rate = float(supply.rate("mte3") or 1.35168e10)
    vector_rate = float(supply.rate("vector") or 6.7584e9)
    cube_rate = float(supply.rate("cube") or 4.0e14)
    head_dim = max(1, int(head_dim))
    stream_ms = float(read_bytes) / max(1.0, mte2_rate)
    owner_ms = float(read_bytes) / max(1.0, mte2_rate * blocks)
    store_ms = float(write_bytes) / max(1.0, mte3_rate)
    vector_ms = float(vector_ops) / max(1.0, vector_rate)
    scalar_ms = (ATTENTION_WALK_BASE_MS
                 + float(visible_kv) * ATTENTION_WALK_MS_PER_KV_TOKEN)
    pairs = int(head_token_pairs or 0)
    if pairs <= 0:
        pairs = max(1, int(batch) * int(visible_kv))
    cube_ms = 2.0 * float(pairs) * float(head_dim) / max(1.0, cube_rate)
    pages = max(1, int(visible_kv) // max(1, int(page_size)))
    if pages <= ATTENTION_LATENCY_PAGES:
        return [
            StageDraft(label="native:pa_dispatch",
                       latency_ms=float(_clock.DISPATCH_MS), role="prologue"),
            clock_stage(supply, {"mte2": stream_ms},
                        label="native:pa_load", role="load"),
            clock_stage(supply, {"vector": scalar_ms + vector_ms},
                        label="native:pa_walk", role="compute"),
            clock_stage(supply, {"matrix": cube_ms},
                        label="native:pa_kv", role="compute"),
            clock_stage(supply, {"mte3": store_ms},
                        label="native:pa_store", role="store"),
        ]
    body_max = max(owner_ms, cube_ms, vector_ms, store_ms)
    exposed = (scalar_ms if scalar_ms >= body_max
               else min(scalar_ms, 0.002))
    body = clock_stage(
        supply,
        {"mte2": owner_ms, "matrix": cube_ms, "vector": vector_ms,
         "mte3": store_ms},
        label="native:pa_body", role="compute")
    return [
        StageDraft(label="native:pa_prologue",
                   latency_ms=float(_clock.DISPATCH_MS) + exposed,
                   role="prologue"),
        body]


def _a5_aiv_drafts(
        kernel_class: str, orientation: Mapping[str, Any],
        supply: KernelSupply) -> list[StageDraft]:
    """AIV kernel: dispatch + scalar prologue, concurrent body, MTE3 store.

    The AIV owner issues its MTE2 read and vector body concurrently, so the
    body is their maximum; the MTE3 store drains after the last computed row.
    The reduction/scalar loop is a separate serial pass.  Occupancy, rates and
    start-up costs all come from the selected die's declared supply.
    """
    rows = int(orientation.get("rows") or 0)
    width = int(orientation.get("width") or 0)
    dtype_bytes = int(orientation.get("dtype_bytes") or 2)
    live = LIVE_BUFFERS.get(kernel_class, LIVE_BUFFERS["elementwise"])
    rates = _clock.rates_from_supply(supply)
    occupancy = row_norm_occupancy(
        rows, width, dtype_bytes=dtype_bytes, live_buffers=live,
        units=supply.units_of("vector") or rates.aiv,
        ub_bytes=supply.ub_bytes,
        columns_split=bool(orientation.get("columns_split")))
    blocks = occupancy.blocks
    declared_blocks = _clock.declared_block_num(
        {"block_num": orientation.get("block_num")})
    if declared_blocks is not None:
        blocks = min(occupancy.units, declared_blocks)
    times = _validated_catalog_times(_clock.aiv_times(
        kernel_class=kernel_class, rows=rows, width=width,
        read_bytes=float(orientation.get("read_bytes") or 0.0),
        write_bytes=float(orientation.get("write_bytes") or 0.0),
        blocks=blocks, units=occupancy.units,
        vector_ops=orientation.get("vector_ops"),
        execution_scope=str(orientation.get("execution_scope") or "serving"),
        dtype_bytes=dtype_bytes, ub_bytes=supply.ub_bytes,
        rates=rates), rates)
    drafts = [
        StageDraft(label="native:aiv_prologue", latency_ms=float(times.prologue_ms),
                   role="prologue"),
        clock_stage(supply, {pipe: float(time) for pipe, time in times.body},
                    label="native:aiv_body",
                    role="compute")]
    for index, (pipe, milliseconds) in enumerate(times.tail):
        milliseconds = float(milliseconds)
        if milliseconds == 0:
            continue
        if pipe == "mte3":
            label = "native:aiv_store" if index == 0 else f"native:aiv_store:{index}"
            role = "store"
        else:
            label, role = f"native:aiv_epilogue:{index}", "epilogue"
        drafts.append(clock_stage(
            supply, {pipe: milliseconds}, label=label, role=role))
    return drafts
