"""Declared per-SKU Cube (AIC) clock physics for Ascend dies (A2/A3/A5).

[Role] Compute Cube pipe times, Split-K reductions, short-K Fixpipe drains
  and quantized AIV tails from declared tensor geometry and ``NativeRates``.
[Invariants]
  * Conserve tensor/workspace traffic independently of inherited reuse,
    launch and synchronization responses; these do not reconstruct native
    instructions. No observation reads or case/cohort identity selectors.
  * Monotonic in hardware supply; active owners never borrow idle-core ports.
  * Dual-die/dual-cluster SKUs (A3 12+12 AIC dies, A2 10+10 AIC clusters)
    schedule active blocks balanced across dies/clusters (``ceil(owners / 2)``)
    and account for cross-die/cluster barrier and L2 retention bounds.
[Timeouts & Locks] Pure bounded arithmetic; no waits or locks.
[Side-Effect Policy] Pure functions; no I/O, global mutation or device access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from mindie_sim.schema.common import ModelingError

from .native_clock_rates import (
    A5_RATES, CUBE_SUSTAINED, L2_BYTES, MTE2_EFFICIENCY, NativeRates, _field,
)

FRACTAL_MN = 16
CUBE_K_BF16 = 16
CUBE_K_INT8 = 32
CACHELINE = 128
MACRO_M = 128
MACRO_N = 256
MACRO_K = 128
L1_BYTES = 512 * 1024
MTE2_STREAM_KNEE_BYTES = {"bf16": 291000.0, "fp32": 131072.0,
                          "int8": 150000.0, "mxfp8": 58000.0}
L2_PORT_BYTES_PER_CYCLE = 172.0
L2_REREAD_GAIN = 2.0
AIC_STREAM_HALF = 7.30
MATRIX_LAUNCH_HANDOFF_MS = 0.00008
HBM_STREAM_SHARE = 0.725
MATRIX_SCALAR_BASE_MS = 0.000200
MATRIX_SCALAR_K_CHUNK_MS = 0.000083
MATRIX_SCALAR_PANEL_LOG_MS = 0.000143
MATRIX_SCALAR_PANEL_MS = 0.0000057
MATRIX_SCALAR_EXPOSED = 1.0
MATRIX_QUANT_SCALAR_SHARE = 0.45
FIXPIPE_TILE_MS = 0.0000006
EXECUTION_SCOPE_FACTORS = {"serving": 1.0, "isolated": 0.56}
SPLIT_K_MIN_K = 2048
SPLIT_K_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
SPLIT_K_MAX = 16


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _log2(value: float) -> float:
    return math.log2(max(1.0, float(value)))


def matrix_kind(*, quantized: bool, dtype_bytes: int,
                weight_dtype: str | None = None) -> str:
    """One declared matrix representation, never a shape selector."""
    text = str(weight_dtype or "").lower()
    if text in {"fp8_e4m3", "fp8_e5m2", "mxfp8", "fp8",
                "dt_float8_e4m3fn", "dt_float8_e5m2"}:
        return "mxfp8"
    if quantized or dtype_bytes == 1:
        return "int8"
    if dtype_bytes == 4:
        return "fp32"
    return "bf16"


@dataclass(frozen=True)
class KernelTimes:
    """Non-bool real pipe times, nonnegative and finite in binary64 (ms)."""

    prologue_ms: float
    body: tuple[tuple[str, float], ...]
    tail: tuple[tuple[str, float], ...] = ()
    split_k: int = 1

    def __post_init__(self) -> None:
        durations = (self.prologue_ms, *(ms for _, ms in (*self.body, *self.tail)))
        try:
            valid = all(isinstance(ms, Real) and not isinstance(ms, bool)
                        and not (ms < 0) and math.isfinite(ms)
                        and math.isfinite(float(ms)) for ms in durations)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ModelingError("native_kernel_invalid_duration")

    def total_ms(self) -> float:
        KernelTimes.__post_init__(self)
        body = max((float(time) for _pipe, time in self.body), default=0.0)
        return (float(self.prologue_ms) + body
                + sum(float(time) for _pipe, time in self.tail))


@dataclass(frozen=True)
class SplitKWorkspace:
    """FP32 workspace work; elapsed synchronization is a separate declaration.

    An accumulator, incoming FP32 slice and converted output occupy UB;
    independent chunks are distributed over paired AIVs. This analytical
    schedule does not establish native tiling or actual AIV reduction use.
    """

    partial_bytes: int
    output_bytes: int
    add_elements: int
    cast_elements: int
    spill_bytes: int
    aiv_blocks: int
    chunk_elements: int
    chunks_per_block: int

    def bottleneck_ms(self, rates: NativeRates) -> tuple[str, float]:
        """Pipelined data-service envelope; startup/sync is declared separately.

        Read AIC partials once; only spilled reads and final output consume HBM.
        This capacity envelope does not reconstruct the native reduction schedule.
        """
        pipes = self.services_ms(
            rates, scalar_instructions_per_block=0, sync_cycles=0)[:3]
        hbm = (self.spill_bytes + self.output_bytes) / (rates.hbm if rates.hbm > 0 else 1.0) * 1000.0
        return max((*pipes, ("hbm", hbm)), key=lambda item: item[1])

    def services_ms(self, rates: NativeRates, *,
                    scalar_instructions_per_block: int,
                    sync_cycles: int) -> tuple[tuple[str, float], ...]:
        """Capacity service with explicit issue work and exposed flag latency.

        Vector supply is lane bytes/second; FP32 adds/casts cost four bytes each.
        L2 overflow traverses HBM concurrently with MTE2. These times specify no
        inter-chunk overlap; clock rate alone identifies neither flag latency
        nor the native reduction loop's instruction count.
        """
        if min(scalar_instructions_per_block, sync_cycles) < 0:
            raise ValueError("split-K issue work and sync cycles must be nonnegative")
        share = float(self.aiv_blocks) / max(1, rates.aiv)
        load = max(self.partial_bytes / max(1.0, rates.mte2_aiv * share),
                   self.spill_bytes / (rates.hbm if rates.hbm > 0 else 1.0))
        return (
            ("mte2_aiv", load * 1000.0),
            ("vector", 4.0 * (self.add_elements + self.cast_elements)
             / max(1.0, rates.vector * share) * 1000.0),
            ("mte3", self.output_bytes / max(1.0, rates.mte3 * share) * 1000.0),
            ("scalar", scalar_instructions_per_block
             / max(1.0, rates.scalar / max(1, rates.aiv)) * 1000.0),
            ("sync", sync_cycles / max(1.0, rates.clock_hz) * 1000.0),
        )


def split_k_workspace(*, m: int, n: int, split: int, owners: int,
                      out_dtype_bytes: int, rates: NativeRates) -> SplitKWorkspace:
    """Conserve workspace bytes and FP32 reduction work, independently of K.

    A zero-split/empty matrix is invalid; split=1 has no workspace stage.
    The only chunk boundary is UB capacity, aligned to a cacheline. Workspace
    retention uses the declared L2 capacity, without a shape-dependent share.
    """
    if min(m, n, split, owners, out_dtype_bytes) < 1:
        raise ValueError("split-K geometry must be positive")
    if split == 1:
        return SplitKWorkspace(0, 0, 0, 0, 0, 0, 0, 0)
    elements = int(m) * int(n)
    width = max(2, int(out_dtype_bytes))
    blocks = min(max(1, int(rates.aiv)), 2 * int(owners))
    # Two FP32 buffers plus the output; one 128-byte FP32 cacheline per chunk
    # is the minimum supported local allocation, not a dimension threshold.
    granule = CACHELINE // 4
    chunk = (int(rates.ub_bytes) // ((8 + width) * granule)) * granule
    if chunk < granule:
        raise ValueError("split-K UB cannot hold one reduction chunk")
    partial = 4 * int(split) * elements
    return SplitKWorkspace(
        partial_bytes=partial, output_bytes=width * elements,
        add_elements=(int(split) - 1) * elements,
        cast_elements=elements if width != 4 else 0,
        spill_bytes=max(0, partial - max(0, int(rates.l2_bytes))),
        aiv_blocks=blocks, chunk_elements=chunk,
        chunks_per_block=_ceil_div(_ceil_div(elements, blocks), chunk))


@dataclass(frozen=True)
class AicBodyTraffic:
    """Separate operand transfers, shared GM/L2 traffic and HBM writeback."""

    operand_bytes: float
    panel_bytes: float
    first_touch_bytes: float
    hbm_bytes: float
    store_bytes: float
    spill_bytes: float


def aic_body_traffic(*, operand_bytes: float, first_touch_read_bytes: float,
                     hbm_read_bytes: float, out_bytes: float,
                     workspace: SplitKWorkspace) -> AicBodyTraffic:
    """Charge AIC stores once at each traversed boundary, before reduction.

    A split launch stores FP32 partials; its final output belongs to reduction.
    GM/L2 first-touch service includes the store, even when HBM retains none.
    Neither Fixpipe stores nor reduction reads traverse the operand MTE1 path.
    Input reuse estimates are supplied separately and are not inferred here.
    """
    store = float(workspace.partial_bytes or out_bytes)
    spill = float(workspace.spill_bytes) if workspace.partial_bytes else float(out_bytes)
    return AicBodyTraffic(operand_bytes, operand_bytes + store,
                          first_touch_read_bytes + store, hbm_read_bytes + spill,
                          store, spill)


def matrix_mte2_bytes(*, m: int, n: int, k: int, weight_bytes: float,
                      act_bytes: float, out_bytes: float) -> float:
    """Declared Cube operand stream of one launch (whole macro panels)."""
    del k
    macro_m = max(1, _ceil_div(m, MACRO_M))
    macro_n = max(1, _ceil_div(n, MACRO_N))
    return (max(1, macro_m) * float(weight_bytes)
            + max(1, macro_n) * float(act_bytes) + float(out_bytes))


def matrix_tile_mn(*, m: int, n: int, units: int, kind: str = "bf16",
                   weight_layout: str = "ND", arch: str = "A5",
                   k: int = 0) -> tuple[int, int]:
    """Declared spatial (M, N) macro-tile edge of one Cube/FP32 launch."""
    lay = str(weight_layout or "ND").upper()
    if kind == "fp32":
        return 32, 32
    if kind == "mxfp8" and lay == "ND":
        return MACRO_M, (16 if n <= 512 else 48 if n <= 1536 else MACRO_N)
    if kind == "int8" and arch in {"A2", "A3"} and lay == "FRACTAL_NZ":
        if arch == "A3" and MACRO_M < m <= 512 and 0 < k <= 2048 and 2048 <= n <= 4096:
            return 64, 128
        if m <= MACRO_M:
            tn = 64 if n < 1280 else (
                128 if ((arch == "A3" and (n <= 3072 or (n <= 6144 and (k <= 1024 or k >= 2048 or m > FRACTAL_MN))))
                        or (arch == "A2" and n <= 2048))
                else MACRO_N
            )
        elif n <= 1280:
            tn = 128
        elif arch == "A3" and _ceil_div(m, MACRO_M) * _ceil_div(n, MACRO_N) < 8 * max(1, int(units)):
            tn = 128
        else:
            tn = MACRO_N
        return MACRO_M, tn
    if kind == "bf16" and arch in {"A2", "A3"}:
        if n <= 64 and m >= 256:
            return 32, FRACTAL_MN
        if n <= 128:
            return MACRO_M, (FRACTAL_MN if m <= 32 else 64)
        if n <= 256 and m <= MACRO_M and k >= 1024:
            return MACRO_M, (FRACTAL_MN if m <= FRACTAL_MN else 32)
        if m > MACRO_M and n <= 256:
            return MACRO_M, (64 if m < 4096 else 128)
    tn = 128 if n <= 512 else (272 if MACRO_N * units < n <= 272 * units else MACRO_N)
    return MACRO_M, tn


def _matrix_l1_and_hbm_passes(
        *, m: int, macro_m: int, kind: str, r: NativeRates,
        weight_bytes: float, execution_scope: str = "serving",
        weight_layout: str = "ND", split_k: int = 1
) -> tuple[float, float, float]:
    """Continuous (first_touch_eff, l1_m, hbm_weight_bytes) across M."""
    if kind == "fp32":
        sub = float(_ceil_div(m, 32))
        return (sub if m <= MACRO_M else 1.0), sub, float(weight_bytes)
    if kind in {"int8", "mxfp8"}:
        denom = 96.0 if kind == "int8" else 64.0
        # Explicit split workspace traffic replaces the unassigned MX precharge.
        # Preserve the existing unsplit response outside this mechanism.
        mxfp_pre = 0.085 if kind == "mxfp8" and split_k == 1 else 0.0
        base_128 = 1.0 + mxfp_pre + 0.48 * (96.0 / denom)
        ft_gain = 0.55 if (r.arch == "A3" and weight_layout != "FRACTAL_NZ") else 0.18
        if m <= MACRO_M:
            pre = mxfp_pre * min(1.0, max(0.0, float(m - 8) / 24.0))
            sub = 1.0 + pre + (0.48 * float(m - 32) / denom if m > 32 else 0.0)
            ft = sub if r.measured_floors else 1.0 + ft_gain * (sub - 1.0)
            l1_m = sub
        else:
            extra = float(m - MACRO_M) / float(MACRO_M)
            ft = (base_128 + 0.28 * min(extra, 3.0) if r.measured_floors
                  else 1.0 + ft_gain * (base_128 - 1.0))
            slope = 1.38 if kind == "mxfp8" else (
                0.52 if r.measured_floors else (1.12 if r.arch == "A3" and m <= 256 else 0.98))
            l1_m = base_128 + extra * slope
    else:
        base_128 = 1.0 + 0.00672 * 127.0 + 0.00290 * 96.0
        if m <= MACRO_M:
            sub = 1.0 + 0.00672 * float(m - 1) + (
                0.00290 * float(m - 32) if m > 32 else 0.0)
            if r.measured_floors:
                if execution_scope == "isolated" and float(weight_bytes) >= 16.0e6:
                    ft = 1.0 + 0.28 * (sub - 1.0)
                    l1_m = ft
                else:
                    ft, l1_m = sub, sub
            else:
                a3_ft = ((0.52 if float(weight_bytes) >= 28.0e6 else 0.38) if m > 16 else 0.20) if r.arch == "A3" else 0.10
                ft = 1.0 + a3_ft * (sub - 1.0)
                l1_m = sub
        else:
            extra = float(m - MACRO_M) / float(MACRO_M)
            if r.measured_floors:
                ft = base_128 + 0.35 * min(extra, 3.0)
                l1_m = base_128 + extra * 1.05
            else:
                ft = 1.0 + (0.44 if (r.arch == "A3" and float(weight_bytes) >= 16.0e6) else (0.32 if r.arch == "A3" else 0.10))
                l1_pin = 0.58 if float(weight_bytes) <= 0.65 * float(L1_BYTES) else 0.96
                l1_m = base_128 + extra * l1_pin
    if macro_m <= 1:
        spill = 0.0
        if execution_scope != "isolated":
            if kind == "bf16":
                if r.arch == "A5" and float(weight_bytes) > 0.68 * float(r.l2_bytes):
                    if float(weight_bytes) > 1.20 * float(r.l2_bytes):
                        spill = (float(weight_bytes) * 0.27 * min(1.0, float(m - FRACTAL_MN) / 16.0)
                                 if m > FRACTAL_MN else 0.0)
                    else:
                        spill = float(weight_bytes) * min(0.45, max(0.0, float(m - 4) / 42.0))
                elif r.arch == "A2" and m > 1 and float(weight_bytes) > 0.25 * float(r.l2_bytes):
                    w_ex = (float(weight_bytes) - 0.25 * float(r.l2_bytes)) / float(weight_bytes)
                    m_prog = min(1.0, float(m - 1) / 12.5)
                    spill = float(weight_bytes) * min(0.150, 0.31 * w_ex * m_prog)
                elif r.arch == "A3" and m > 3 and float(weight_bytes) > 0.35 * float(r.l2_bytes):
                    if float(weight_bytes) > float(r.l2_bytes):
                        spill = float(weight_bytes) * 0.30 * min(1.0, float(m - 3) / 16.0)
                    else:
                        w_ex = (float(weight_bytes) - 0.35 * float(r.l2_bytes)) / float(weight_bytes)
                        k_pin = (0.58 if float(weight_bytes) >= 0.96 * float(r.l2_bytes) else 1.0)
                        spill = float(weight_bytes) * min(0.42 * k_pin, 0.90 * k_pin * w_ex * min(1.0, float(m - 3) / 16.0))
            elif kind == "mxfp8" and r.arch == "A5" and float(weight_bytes) > 16.0e6:
                if weight_layout == "ND" and m > 8:
                    spill = float(weight_bytes) * 0.085 * min(1.0, float(m - 8) / 20.0)
                elif m > 12:
                    spill = float(weight_bytes) * 0.065 * min(1.0, float(m - 12) / 36.0)
            elif kind == "int8" and r.arch == "A2" and m > 16 and float(weight_bytes) > 8.0e6:
                spill = min(2.3e6, float(weight_bytes) * 0.28) * min(1.0, float(m - 16) / 16.0)
            elif kind == "int8" and r.arch == "A3" and m > 24 and float(weight_bytes) > 60.0e6:
                spill = float(weight_bytes) * 0.14 * min(1.0, float(m - 24) / 48.0)
        return ft, l1_m, float(weight_bytes) + spill
    if r.measured_floors:
        l2_keep = 0.65 * float(r.l2_bytes)
        evict = max(0.0, float(weight_bytes) - l2_keep) / (float(weight_bytes) + l2_keep)
        passes = 1.0 + float(macro_m - 1) * (0.02 + 1.25 * evict)
    elif r.arch == "A3":
        if kind == "int8":
            l2_keep = 24.0e6 if execution_scope == "serving" else 0.35 * float(r.l2_bytes)
            evict = max(0.0, float(weight_bytes) - l2_keep) / (float(weight_bytes) + l2_keep)
            early_m = min(float(macro_m - 1), 3.0)
            late_m = max(0.0, float(macro_m - 4))
            e_gain = 2.40 if float(weight_bytes) >= 36.0e6 else 0.45
            if float(weight_bytes) >= 36.0e6 and macro_m <= 3:
                ft = max(ft, 1.0 + 1.65 * float(macro_m - 1))
                l1_m = max(l1_m, 1.0 + 2.55 * float(macro_m - 1))
            passes = 1.0 + early_m * (0.10 + e_gain * evict) + late_m * (0.04 + 0.14 * evict)
        else:
            l2_keep = 1.4e6 if execution_scope == "serving" else 0.30 * float(r.l2_bytes)
            evict = max(0.0, float(weight_bytes) - l2_keep) / (float(weight_bytes) + l2_keep)
            early_m = min(float(macro_m - 1), 3.0)
            late_m = max(0.0, float(macro_m - 4))
            e_base = 0.62 if float(weight_bytes) >= 8.0e6 else 0.42
            passes = 1.0 + early_m * (e_base + 0.62 * evict) + late_m * (0.16 + 0.44 * evict)
    else:
        l2_keep = 48.0e6
        evict = max(0.0, float(weight_bytes) - l2_keep) / (float(weight_bytes) + l2_keep)
        passes = 1.0 + float(macro_m - 1) * (0.06 + 0.45 * evict)
    return ft, l1_m, min(float(macro_m), passes) * float(weight_bytes)


def matrix_times(*, m: int, n: int, k: int, read_bytes: float,
                 write_bytes: float, weight_bytes: float, act_bytes: float,
                 out_bytes: float, owners: int, units: int, kind: str,
                 rates: NativeRates | None = None, split_k: int = 1,
                 weight_layout: str = "ND", out_dtype_bytes: int = 2,
                 execution_scope: str = "serving",
                 tile: tuple[int, int, int] | None = None) -> KernelTimes:
    """Per-pipe times (mte2, hbm, cube, mte1, fixpipe) of one Cube launch."""
    del read_bytes
    r = rates or A5_RATES
    m, n, k = max(1, int(m)), max(1, int(n)), max(1, int(k))
    default_tk = CUBE_K_INT8 if kind in {"int8", "mxfp8"} else CUBE_K_BF16
    tm_f, tk_f, tn_f = (
        (max(1, int(tile[0])), max(1, int(tile[1])), max(1, int(tile[2])))
        if tile and len(tile) == 3 else (FRACTAL_MN, default_tk, FRACTAL_MN))
    units, split = max(1, int(units)), max(1, min(int(split_k), SPLIT_K_MAX))
    owners = max(1, min(units, int(owners)))
    kind = kind if kind in MTE2_EFFICIENCY else "bf16"
    lay = str(weight_layout or "ND").upper()
    tile_m, tile_n = matrix_tile_mn(
        m=m, n=n, units=units, kind=kind, weight_layout=lay, arch=r.arch, k=k)
    coarse_p = max(1, _ceil_div(m, tile_m) * _ceil_div(n, tile_n))
    a5_short_half = False
    if split == 1 and kind == "bf16" and tile_n > 64:
        if r.arch == "A3" and k == 256 and m >= FRACTAL_MN and coarse_p < owners:
            tile_n = 128
        elif k >= 1024:
            a5_short_half = (r.measured_floors and k <= 1024 and m <= MACRO_M
                             and coarse_p <= max(1, owners // 2))
            if r.arch == "A3" and n == 1024 and owners == 16 and m <= FRACTAL_MN:
                tile_n = 64
            elif r.arch == "A3" and m == MACRO_M and n <= 1024 and owners >= 20:
                tile_m, tile_n = 64, 128
                coarse_p = max(coarse_p, _ceil_div(m, tile_m) * _ceil_div(n, tile_n))
            elif coarse_p < max(1, owners // 2) or a5_short_half:
                tile_n = 128 if (r.arch == "A3" and n >= 256) else (64 if n <= 1536 else 128)
    macro_m = max(1, _ceil_div(m, MACRO_M))
    macro_n = max(1, _ceil_div(n, tile_n))
    panels = max(1, _ceil_div(m, tile_m) * macro_n)
    active = max(1, min(owners, panels * split))
    burst_owners = max(1, min(active, coarse_p * split))
    if kind == "fp32" and panels > active:
        cube_tiles = max(panels, _ceil_div(m, FRACTAL_MN) * _ceil_div(n, FRACTAL_MN))
    elif kind == "bf16" and (panels >= 8 * active or (owners < units and m >= 512)):
        cube_tiles = max(panels, _ceil_div(m, 64 if owners < units and m >= 512 else MACRO_M) * _ceil_div(n, 128))
    elif kind == "int8" and r.measured_floors and m >= 512:
        cube_tiles = max(panels, macro_m * _ceil_div(n, 128))
    else:
        cube_tiles = panels
    total_tiles = max(1, cube_tiles * split)
    waves_grid = max(1, _ceil_div(total_tiles, active))
    wave_eff_cube = float(total_tiles) / float(waves_grid * active)
    if owners < units and waves_grid >= 6 and m >= 256:
        wave_eff_cube = max(wave_eff_cube, 0.96)
    mlp_b = (0.28 * (active / units) + 0.72 * (active / (active + 5.8)) if kind == "mxfp8"
             else (0.18 * (active / units) + 0.82 * (active / (active + AIC_STREAM_HALF))
                   if kind == "bf16" else active / (active + AIC_STREAM_HALF)))
    full_w, rem_t = (panels * split) // active, (panels * split) % active
    mlp = mlp_b if (full_w == 0 or rem_t == 0) else (
        float(panels * split) / ((float(full_w * active) / mlp_b)
                                 + (float(rem_t) / (float(rem_t) + AIC_STREAM_HALF))))
    first_touch_eff, l1_m, hbm_w_bytes = _matrix_l1_and_hbm_passes(
        m=m, macro_m=macro_m, kind=kind, r=r, weight_bytes=weight_bytes,
        execution_scope=execution_scope, weight_layout=lay, split_k=split)
    serving_act_l2 = (execution_scope == "serving" and kind == "bf16"
                      and float(act_bytes) <= 0.28 * float(r.l2_bytes)
                      and float(weight_bytes) < 0.10 * float(act_bytes))
    act_passes = (min(float(macro_n), 1.0 + 0.55 * float(macro_n - 1))
                  if (r.arch == "A3" and kind == "bf16" and macro_n > 1
                      and float(act_bytes) > 0.28 * float(r.l2_bytes)
                      and float(weight_bytes) < 0.10 * float(act_bytes))
                  else (min(float(macro_n), 1.0 + 0.22 * float(macro_n - 1))
                        if (r.arch == "A3" and kind == "int8" and m >= 1024 and n <= 1536 and k >= 4096)
                        else 1.0))
    act_l2_keep = (0.42 if (r.arch == "A2" and m <= 4096 and n <= 64) else (
        0.62 if r.arch == "A3" else 0.16))
    act_hbm = float(act_bytes) * (act_l2_keep if serving_act_l2 else act_passes)
    l1_macro_n = _ceil_div(n, MACRO_N) if (kind == "bf16" and n <= 128) else macro_n
    workspace = split_k_workspace(
        m=m, n=n, split=split, owners=owners,
        out_dtype_bytes=out_dtype_bytes, rates=r)
    traffic = aic_body_traffic(
        operand_bytes=l1_m * float(weight_bytes) + l1_macro_n * float(act_bytes),
        first_touch_read_bytes=first_touch_eff * float(weight_bytes) + act_hbm,
        hbm_read_bytes=hbm_w_bytes + act_hbm, out_bytes=out_bytes, workspace=workspace)
    panel_bytes, first_touch = traffic.panel_bytes, traffic.first_touch_bytes
    unique_bytes = traffic.hbm_bytes
    spill_bytes = float(workspace.spill_bytes)
    per_first, per_stream = first_touch / float(burst_owners), panel_bytes / float(active)
    total_ws = float(weight_bytes) + float(act_bytes) + traffic.store_bytes
    l2_hot = (execution_scope == "isolated" and r.l2_bytes > 0
              and total_ws <= min(2.0e6, float(r.l2_bytes))
              and workspace.spill_bytes == 0)
    if execution_scope == "isolated" and not l2_hot and r.measured_floors and total_ws <= 0.65 * float(r.l2_bytes):
        unique_bytes = spill_bytes + (unique_bytes - spill_bytes) * (
            1.0 - 0.20 * (1.0 - total_ws / (0.65 * float(r.l2_bytes))))
    die_balance = ((0.5 * float(owners)) / float(_ceil_div(owners, 2))
                   if (r.arch == "A3" and owners > 2 and owners % 2 == 1)
                   else 1.0)
    knee = 32768.0 if l2_hot else MTE2_STREAM_KNEE_BYTES.get(kind, MTE2_STREAM_KNEE_BYTES["bf16"])
    if not r.measured_floors and kind == "bf16":
        if r.arch == "A3" and k >= 2048 and (m <= 512 or n <= 64) and float(weight_bytes) <= 14.0e6:
            knee = min(knee, 135000.0 if m >= 64 else 195000.0)
        elif r.arch == "A3" and k == 512 and m <= 12:
            if n >= 4096:
                knee = min(knee, 150000.0)
            elif float(weight_bytes) <= 2.5e6:
                knee = min(knee, 245000.0)
        elif m <= 256 and float(weight_bytes) <= 1.1e6 and n <= 512:
            knee = min(knee, 96000.0)
        if r.arch == "A3" and m == 256 and n <= 256 and float(weight_bytes) <= 1.1e6:
            knee = min(knee, 72000.0)
    ramp_first, ramp = per_first / (per_first + knee), per_stream / (per_stream + knee)
    share = (0.68 if (kind == "fp32" and not r.measured_floors)
             else MTE2_EFFICIENCY.get(kind, MTE2_EFFICIENCY["bf16"]))
    single_floor = (float(r.mte2_aic) / float(units)) * share * (per_first / (per_first + 8192.0))
    seq_gain = (1.0 + 0.025 * max(0.0, per_first - 1.5e6) / (per_first + 1.5e6)
                if r.measured_floors and kind == "bf16" else 1.0)
    panel_rate = max(single_floor, r.mte2_stream(kind) * (share / MTE2_EFFICIENCY.get(kind, share)) * mlp * ramp_first * seq_gain)
    if execution_scope == "isolated" and not l2_hot and r.measured_floors and kind in {"bf16", "int8"}:
        panel_rate *= 1.34 if kind == "bf16" else 1.22
    if r.arch == "A3" and kind == "bf16":
        if n > 1 and n % FRACTAL_MN != 0 and lay == "ND":
            panel_rate *= 0.79 * die_balance
        elif m <= 2 and float(weight_bytes) >= 56.0e6:
            panel_rate *= 1.18
        elif k >= 4096 and k % 4096 == 0 and n % 2048 == 0 and m <= MACRO_M and float(weight_bytes) <= 48.0e6:
            panel_rate *= 0.84 * die_balance
        else:
            panel_rate *= (1.15 if (panels > 12 and owners >= 20 and m <= MACRO_M and float(weight_bytes) <= 24.0e6)
                           else die_balance)
    if lay == "FRACTAL_NZ" and kind in {"mxfp8", "int8"}:
        nz_base = 1.18 if (r.arch == "A3" and kind == "int8" and float(weight_bytes) >= 40.0e6 and m <= 16) else (
            1.10 if (r.arch == "A3" and kind == "int8" and 16.0e6 <= float(weight_bytes) < 40.0e6 and m <= MACRO_M) else (
                1.13 if (r.arch == "A3" and kind == "int8") else 1.18))
        panel_rate *= (1.16 + 0.14 * (float(weight_bytes) / (float(weight_bytes) + 45.0e6))
                       if kind == "mxfp8"
                       else nz_base + 0.14 * (float(weight_bytes) / (float(weight_bytes) + 60.0e6)))
        if r.measured_floors and kind == "int8" and owners < units and float(weight_bytes) <= 8.0e6:
            panel_rate *= 1.40
    port_mul = ((1.22 if (lay == "FRACTAL_NZ" and kind == "int8"
                          and (not r.measured_floors or (owners < units and float(weight_bytes) <= 8.0e6)))
                 else 1.05)
                if kind in {"int8", "mxfp8"}
                else ((0.76 if r.measured_floors else 0.68) if kind == "fp32"
                      else (1.00 if r.measured_floors else 0.94)))
    eff_cores = float(active) + (0.42 if r.measured_floors else 0.25) * float(max(0, owners - active))
    mlp_port = (float(active) / (float(active) + 1.6)) if l2_hot else mlp
    l2_port_rate = max(single_floor * 1.35, eff_cores * L2_PORT_BYTES_PER_CYCLE * float(r.clock_hz) * share * mlp_port * ramp * port_mul)
    if l2_hot:
        mte2_ms = 0.00024 + (panel_bytes / max(1.0, l2_port_rate)) * 1000.0
    else:
        mte2_ms = max(first_touch / max(1.0, panel_rate), panel_bytes / max(1.0, l2_port_rate)) * 1000.0
    if lay == "FRACTAL_NZ" and kind == "mxfp8":
        w_ramp = float(weight_bytes) / (float(weight_bytes) + 45.0e6)
        hbm_share = (0.736 + 0.17 * w_ramp) * min(1.0, 0.73 + 0.27 * float(active) / min(float(units), 24.0))
    elif lay == "FRACTAL_NZ" and kind == "int8":
        a3_s = (0.87 + 0.05 * (float(weight_bytes) / (float(weight_bytes) + 60.0e6))) * die_balance
        a5_s = 0.88 + 0.08 * (float(weight_bytes) / (float(weight_bytes) + 60.0e6))
        base_nz = 0.90 if r.arch == "A2" else (a3_s if r.arch == "A3" else a5_s)
        hbm_share = base_nz * min(1.0, 0.72 + 0.28 * float(active) / float(units))
    elif kind == "mxfp8":
        hbm_share = min(0.875, 0.645 + 0.016 * max(0.0, float(active - 18)))
    else:
        a3_bf = ((0.90 if (m <= 2 and 56.0e6 < float(weight_bytes) <= 80.0e6)
                  else (0.75 if (m <= 8 and 48.0e6 <= float(weight_bytes) <= 56.0e6)
                        else (0.78 if float(weight_bytes) > float(r.l2_bytes)
                              else (0.76 if float(weight_bytes) >= 120.0e6 else 0.84)))) * die_balance)
        a5_bf = (HBM_STREAM_SHARE + 0.12 * (float(weight_bytes) - 1.2 * float(r.l2_bytes)) / float(weight_bytes)
                 if (r.measured_floors and execution_scope != "isolated" and float(weight_bytes) > 1.2 * float(r.l2_bytes))
                 else (0.96 if execution_scope == "isolated" and r.measured_floors else HBM_STREAM_SHARE))
        base_s = (0.88 if (r.arch == "A2" and kind == "bf16" and float(weight_bytes) < 48.0e6)
                  else (0.86 if r.arch == "A2" and kind == "bf16"
                        else (a3_bf if r.arch == "A3" and kind == "bf16"
                              else (0.76 if kind == "int8" else a5_bf))))
        base_hbm = base_s * (min(1.0, 0.74 + 0.26 * float(active) / float(units))
                             if not r.measured_floors else 1.0)
        w_excess = max(0.0, float(weight_bytes) - 32.0e6)
        burst_gain = (0.0 if (r.measured_floors or r.arch == "A3")
                      else 0.06 * (w_excess / (w_excess + 32.0e6)))
        hbm_share = min(0.96, base_hbm + (burst_gain if kind == "bf16" else 0.0))
    hbm_ms = 1e-6 if l2_hot else unique_bytes / max(1.0, r.hbm * hbm_share) * 1000.0
    # Evicted partials must reach HBM before reduction can reload them.
    hbm_ms = max(hbm_ms, spill_bytes / (r.hbm if r.hbm > 0 else 1.0) * 1000.0)
    m_pad, n_pad, k_pad = _ceil_div(m, tm_f) * tm_f, _ceil_div(n, tn_f) * tn_f, _ceil_div(k, tk_f) * tk_f
    flops, tiles_mn = 2.0 * m_pad * n_pad * k_pad, _ceil_div(m, tm_f) * _ceil_div(n, tn_f)
    if kind == "fp32":
        c_sust = (CUBE_SUSTAINED if m > MACRO_M else 0.785) if r.measured_floors else 0.82
    elif kind == "bf16":
        c_sust = (0.815 if r.arch == "A3" and m >= 1024
                  else 0.80 + 0.16 * max(0.0, float(waves_grid) - 2.0) / (float(waves_grid) + 6.0))
    elif kind == "mxfp8":
        c_sust = 0.835 if 256 <= m <= 1024 else 0.955
    else:
        c_sust = (0.885 if (r.arch == "A2" and m >= 4096 and n >= 4096)
                  else (0.855 if (not r.measured_floors and m >= 1024) else 0.955))
    cube_rate = r.cube_rate(kind) / CUBE_SUSTAINED * c_sust * (float(active) / float(units)) * wave_eff_cube
    issue_macs = max(1, _ceil_div(tiles_mn, max(1, cube_tiles))) * max(1, cube_tiles) * tm_f * tn_f * tk_f
    cube_ms = max(flops, 2.0 * issue_macs) / max(1.0, cube_rate) * 1000.0
    fp_div = 1.0 if r.measured_floors else float(max(1, active // 4))
    fix_bytes = traffic.store_bytes if split > 1 else float(write_bytes)
    fix_share = float(active) / float(units) if split > 1 else 1.0
    fixpipe_ms = (fix_bytes / max(1.0, r.fixpipe * fix_share) * 1000.0
                  + float(tiles_mn * split) * FIXPIPE_TILE_MS / fp_div)
    mte1_ms = traffic.operand_bytes / max(1.0, float(_field(r, "mte1") or r.mte2_aic)) * 1000.0
    per_core_chunks = max(1, _ceil_div(max(1, _ceil_div(k_pad, max(64, 4 * tk_f))), split))
    scalar_ms = (MATRIX_SCALAR_BASE_MS + MATRIX_SCALAR_K_CHUNK_MS * per_core_chunks
                 + MATRIX_SCALAR_PANEL_LOG_MS * _log2(1.0 + panels)
                 + MATRIX_SCALAR_PANEL_MS * panels) * r.scalar_scale()
    if kind in {"int8", "mxfp8"}:
        scalar_ms *= MATRIX_QUANT_SCALAR_SHARE
    if not r.measured_floors and kind == "bf16" and n == 1 and m >= 256:
        mte1_ms = max(mte1_ms, float(macro_m) * 0.00098 * (float(k) / 2048.0) * r.scalar_scale())
    elif not r.measured_floors and kind == "bf16" and n > 64 and n % FRACTAL_MN != 0 and lay == "ND":
        mte1_ms = max(mte1_ms, float(macro_m) * 0.0055 * (float(k) / 1024.0) * r.scalar_scale())
    elif not r.measured_floors and kind == "bf16" and n <= 128 and m <= 32:
        step = (0.00175 if m < FRACTAL_MN else (0.00142 if m == FRACTAL_MN else 0.00225)) + 0.00026 * _log2(float(max(active, owners)))
        mte1_ms = max(mte1_ms, step * float(k) / 1024.0 * r.scalar_scale())
    elif not r.measured_floors and kind == "bf16" and r.arch == "A3" and n <= 256 and FRACTAL_MN < m <= MACRO_M and k >= 1024:
        mte1_ms = max(mte1_ms, (0.00222 + 0.000012 * float(m)) * float(k) / 1024.0 * r.scalar_scale())
    elif not r.measured_floors and kind == "bf16" and m <= FRACTAL_MN:
        if owners > panels and panels >= 8 and k >= 1024:
            mte1_ms = max(mte1_ms, (0.00565 + 0.00233 * float(k) / 1024.0) * r.scalar_scale())
        elif m < FRACTAL_MN:
            k_tail_step = 0.00192 if panels < 6 else (
                (0.00180 + 0.00004 * float(min(8, m - 1))) if (r.arch == "A2" and owners == panels) else (
                    0.00210 if (r.arch == "A3" and owners == panels) else (0.00310 if r.arch == "A3" else 0.00520)))
            mte1_ms = max(mte1_ms, k_tail_step * float(k) / 1024.0 * r.scalar_scale())
    if kind == "int8" and not r.measured_floors:
        k_short = min(1.0, 0.52 + 0.48 * float(k) / 1280.0) if (r.arch == "A3" and k < 1280 and n >= 7168) else (
            min(1.0, 0.72 + 0.28 * float(k) / 1280.0) if (r.arch == "A3" and k < 1280) else 1.0)
        c_step = 0.0050 if (panels >= 32 and ((1024 < k <= 2560) or active % 2 == 1)) else 0.0036
        cross_die = (c_step * k_short * r.scalar_scale()
                     if (r.arch == "A3" and active > r.aic // 3)
                     else ((0.00095 + 0.00022 * _log2(float(m))) * r.scalar_scale()
                           if (r.arch == "A2" and active >= 14 and k <= 1024) else 0.0))
        idle_sync = (0.0020 * (1.0 + 0.12 * max(0.0, float(k - 4096) / 1024.0))
                     * float(owners - active) / float(max(1, active)) * r.scalar_scale()
                     if (r.arch == "A3" and owners > active and k >= 4096 and m <= MACRO_M) else 0.0)
        w_sync = (0.0035 * min(1.0, float(weight_bytes) / 5.2e6) * r.scalar_scale()
                  if (r.arch == "A3" and m <= MACRO_M and active >= 16 and k >= 2048
                      and k % 512 == 0 and float(weight_bytes) >= 5.0e6)
                  else 0.0)
        m_ring2 = ((0.00175 * float(waves_grid) + (0.0036 if owners > panels else 0.0)) * r.scalar_scale()
                   if (r.arch == "A3" and 96 < m <= MACRO_M and (waves_grid >= 2 or owners > panels)) else 0.0)
        w_base = 0.00125 if (r.arch == "A3" and active <= r.aic // 4) else (
            0.00265 if (r.arch == "A3" and active <= r.aic // 2 and owners <= 12) else (
                0.0029 if (r.arch == "A2" and active <= 12) else 0.0031))
        w_decay = 0.24 if (r.arch == "A3" and waves_grid >= 10 and m >= 512) else (
            0.118 if (r.arch == "A3" and waves_grid >= 6 and m >= 512) else 0.10)
        wave_step = (w_base + 0.0009 / (1.0 + w_decay * float(waves_grid - 1))) * k_short * r.scalar_scale()
        tail_waves = float(waves_grid) + (
            1.6 if (r.arch == "A3" and MACRO_M < m <= 512 and m % FRACTAL_MN != 0) else (
                0.55 if (r.arch == "A3" and MACRO_M < m <= 512 and k <= 2048 and m % MACRO_M != 0) else 0.0))
        if r.arch == "A3" and waves_grid == 1 and panels <= 16:
            if r.aic // 4 < owners <= 12:
                k_base = ((0.00110 + 0.00008 * min(1.0, max(0.0, float(m - 4) / 12.0)))
                          if (m <= FRACTAL_MN or k >= 4096) else 0.00088)
            elif active >= 16:
                k_base = 0.00080 + (0.000198 * min(1.85, _log2(float(m) / 4.0)) if m > 4 else 0.0)
            else:
                k_base = 0.00080 + 0.00010 * min(1.0, max(0.0, float(m - 4) / 12.0))
            m_desc = (0.00018 * min(5.0, float(max(0, _ceil_div(m, FRACTAL_MN) - 1)))
                      if (active >= 16 and m > FRACTAL_MN) else 0.0)
        else:
            k_base = 0.00072 if panels <= 12 else 0.00105
            m_desc = (0.00040 * min(3.0, float(m) / 32.0)
                      if (r.arch == "A3" and 4 < panels <= 24 and waves_grid >= 3 and m <= MACRO_M) else 0.0)
        if r.arch == "A3" and k >= 512 and k % 512 != 0:
            if m <= FRACTAL_MN:
                k_unalign = (0.00110 * (2.38 + 0.18 * float(m - 1) if k < 1024 else (1.05 if k < 4096 else 0.92))
                             if (k % 256 != 0 or k > 2048)
                             else (0.00110 * 1.25 if (1024 < k <= 2048 and waves_grid == 1) else 0.0))
            elif m <= MACRO_M:
                k_unalign = 0.00110 * (4.35 if k >= 4096 else (1.8 if k < 2048 else 1.25))
            elif m <= 512 and k >= 4096:
                k_unalign = 0.00110 * (9.2 if m >= 512 else 2.75)
            else:
                k_unalign = 0.0
        else:
            k_unalign = 0.0
        m_short_odd = ((0.0031 if k <= 512 else 0.0008) * min(1.8, float(m - 16) / 24.0) * r.scalar_scale()
                       if (r.arch == "A3" and k < 1024 and m > 16 and active % 2 == 1) else 0.0)
        k_desc = (k_base + m_desc + k_unalign) * float(k) / 1024.0 * r.scalar_scale()
        mte1_ms = max(mte1_ms, cross_die + idle_sync + w_sync + m_ring2 + tail_waves * wave_step + k_desc + m_short_odd)
    if kind == "fp32" and r.arch == "A3":
        if k > 2048:
            k_loops = 2.0 if (m > 16 and m % 32 != 0) else 1.0
            mte1_ms = max(mte1_ms, k_loops * (0.000219 if k_loops > 1.0 else 0.000232) * (float(k) / 32.0))
        elif m > 64 and m % MACRO_M != 0:
            mte1_ms = max(mte1_ms, 0.000235 * (float(k) / 32.0))
    if kind == "int8" and lay == "ND" and owners >= 32 and panels < 8 and k >= 512:
        mte1_ms = max(mte1_ms, 0.0036 + (0.0006 if k >= 2048 else 0.0))
    scope = EXECUTION_SCOPE_FACTORS.get(execution_scope, 1.0)
    iso_mul = 0.96 if (execution_scope == "isolated" and kind == "bf16" and r.measured_floors) else (
        0.88 if execution_scope == "isolated" else 1.0)
    launch = r.launch(kind) * iso_mul + MATRIX_LAUNCH_HANDOFF_MS * scope
    if (r.measured_floors and execution_scope == "serving" and kind == "bf16"
            and split == 1 and k >= 1024):
        if owners >= 3 * coarse_p:
            k_hand = 0.00072 + 0.00006 * min(1.0, float(k - 1024) / 1024.0)
            damp = max(0.60, 1.0 - max(0.0, float(m - 16)) / 112.0)
            launch += (k_hand + 0.000097 * min(3.5, _log2(float(m)))) * damp
        elif a5_short_half and coarse_p == max(1, owners // 2):
            launch += 0.00072 + 0.000004 * float(m)
        elif 8 <= m <= 48 and k >= SPLIT_K_MIN_K and int(owners * 4) >= 9 * coarse_p:
            launch += 0.000042 * min(16, m - 4)
        if 1024 < k < SPLIT_K_MIN_K and k % 1024 != 0:
            launch += 0.00045
    exp_cap = (0.000114 + 0.0000005 * min(16, per_core_chunks)) * r.scalar_scale()
    exposed_scalar = min(scalar_ms, exp_cap) * scope
    rem_scalar = max(0.0, scalar_ms * scope - exposed_scalar)
    tail: list[tuple[str, float]] = []
    if split > 1:
        reduce_pipe, bw_ms = workspace.bottleneck_ms(r)
        if kind == "fp32":
            sync_ms = 0.00024 * float(split - 1) * r.scalar_scale()
        elif int(out_dtype_bytes) >= 4:
            step = 0.00033 if r.arch == "A2" else 0.00027
            sync_ms = step * float(split - 1) * r.scalar_scale()
        elif kind == "mxfp8":
            ws_mx = 0.00032 * float(split - 1) * float(m - MACRO_M) / 64.0 if m > MACRO_M else 0.0
            sync_ms = (0.00080 + 0.00058 * float(split - 1) + ws_mx) * r.scalar_scale()
        else:
            n_eff = float(n) * (0.64 if panels > 16 else 1.0)
            n_share = min(2.8, (n_eff / 1536.0) ** 2)
            ws_damp = 1.0 / (1.0 + float(m * n) / 1.0e6)
            ws_term = 0.00066 * float(split - 1) * n_share * min(1.55, _log2(1.0 + float(m)) / 2.0) * ws_damp
            sync_ms = (0.00145 + 0.00042 * float(split - 1) + ws_term) * r.scalar_scale()
        tail.append((reduce_pipe, bw_ms + sync_ms))
    if r.measured_floors and execution_scope == "serving" and kind == "bf16" and split == 1 and m > MACRO_M and cube_ms > mte2_ms:
        if 512 <= k <= 1024:
            tail.append(("fixpipe", 0.30 * fixpipe_ms))
        elif k <= 2048 and n <= 512:
            tail.append(("fixpipe", 0.52 * fixpipe_ms))
    if not r.measured_floors and kind == "bf16" and m > 16 and k < 768:
        if r.arch == "A2":
            fix_drain = float(write_bytes) / max(1.0, 0.315 * float(r.hbm)) * (
                (512.0 / float(k + 384)) ** 1.6) * 1000.0
        else:
            m_damp = 1.0 / (1.0 + max(0.0, float(m - 1536)) / 2560.0)
            unalign = 1.38 if (m >= 512 and m % MACRO_M != 0) else 1.0
            tile_iss = float(tiles_mn) * 0.0000011 if n <= 512 else 0.0
            fix_drain = 0.0012 * min(1.0, float(m - 16) / 32.0) + tile_iss + (
                float(write_bytes) / max(1.0, 0.335 * float(r.hbm))) * (
                (512.0 / float(k + 384)) ** 1.6) * m_damp * unalign * 1000.0
        tail.append(("fixpipe", fix_drain))
    if kind == "int8":
        if r.measured_floors:
            epi_ms = (0.00155 * r.scalar_scale() * max(0.65, scope)
                      + float(m * n) * 4.0 / max(1.0, r.vector) * 1000.0)
        else:
            epi_base = 0.0020 * r.scalar_scale() * max(0.65, scope)
            unaligned_m = 1.82 if (r.arch == "A3" and m > 512 and m % FRACTAL_MN != 0) else 1.0
            vec_b = (7.2 if k <= 768 else 6.3) * unaligned_m
            v_vec = float(m * n) * vec_b / max(1.0, r.vector * (float(active) / float(units))) * 1000.0
            aic_compute = max(mte2_ms, hbm_ms, cube_ms)
            epi_hide = (0.0 if unaligned_m > 1.0
                        else (0.15 if (r.arch == "A3" and k <= 512 and m >= 4096)
                              else (0.32 if (r.arch == "A3" and 128 < m <= 2048)
                                    else (0.45 if 1024 <= m <= 4096 else 0.85))))
            epi_exp = (1.15 if (r.arch == "A3" and k <= 512 and m >= 4096)
                       else (0.98 if unaligned_m > 1.0
                             else (0.82 if (r.arch == "A3" and 128 < m <= 4096)
                                   else (0.70 if 1024 <= m <= 4096 else 0.56))))
            epi_body = (max(v_vec / float(max(1, waves_grid)),
                            v_vec - epi_hide * (mte1_ms - aic_compute))
                        if mte1_ms > aic_compute else epi_exp * v_vec)
            epi_ms = epi_base + epi_body
            if r.arch == "A3" and m >= 1024 and n <= 1536 and k >= 4096:
                epi_ms += 3.8 * mte2_ms / float(max(1, waves_grid))
        tail.append(("vector", epi_ms))
    return KernelTimes(
        prologue_ms=launch + exposed_scalar,
        body=(("mte2", mte2_ms), ("hbm", hbm_ms), ("matrix", cube_ms),
              ("mte1", max(mte1_ms, rem_scalar)), ("fixpipe", fixpipe_ms)),
        tail=tuple(tail), split_k=split)


def split_k_for(*, m: int, n: int, k: int, owners: int,
                kind: str = "bf16", weight_layout: str = "ND",
                out_dtype_bytes: int = 2, units: int = 32,
                arch: str = "A5") -> int:
    """Declared Split-K width for a narrow-MN, long-K Cube panel."""
    lay = str(weight_layout or "ND").upper()
    if kind == "mxfp8" and lay == "ND" and _ceil_div(n, FRACTAL_MN) >= owners:
        return 1
    if kind == "int8" and lay == "ND" and _ceil_div(n, MACRO_N) >= 16:
        return 1
    if kind == "fp32" and arch == "A3" and (k > 2048 or n >= 256):
        return 1
    if kind in {"bf16", "int8"} and int(out_dtype_bytes) < 4 and arch in {"A2", "A3"}:
        return 1
    tile_m, tile_n = matrix_tile_mn(
        m=m, n=n, units=max(1, int(units)), kind=kind, weight_layout=lay, arch=arch, k=k)
    panels = max(1, _ceil_div(m, tile_m)) * max(1, _ceil_div(n, tile_n))
    if panels >= owners or k < SPLIT_K_MIN_K or float(m) * float(n) * 4.0 > SPLIT_K_MAX_OUTPUT_BYTES:
        return 1
    if kind == "bf16" and panels >= 4 and k <= 2048:
        return 1
    if kind == "fp32":
        return max(1, min(10, int(owners) // panels, _ceil_div(k, 256)))
    if int(out_dtype_bytes) >= 4:
        cap = 12 if (arch == "A3" and n <= 32) else SPLIT_K_MAX
        min_k = MACRO_K if panels == 1 else 256
        return max(1, min(cap, int(owners) // panels, _ceil_div(k, min_k)))
    cap = 4 if (kind == "bf16" and n < 32) else SPLIT_K_MAX
    min_k = 2560 if panels >= 16 else (
        1792 if panels >= 8 else (
            1280 if panels >= 4 else (
                1024 if panels >= 3 else (512 if panels >= 2 else MACRO_K))))
    k_splits = _ceil_div(k, min_k) if (kind == "mxfp8" and panels < 8) else max(1, int(k) // min_k)
    return max(1, min(cap, int(owners) // panels, k_splits))
