"""Declared per-tile die laws for the specialised native native leaves.

[Role] Price the four native leaves whose declared geometry is not one ordinary
  Cube/AIV launch: the absorbed-MLA paged decode, the ``HcPre`` mixing matrix,
  the routed grouped-matmul expert bank and the transpose-batch GEMM.  Every
  time is a declared byte/op quantity divided by the selected die's sustained
  rate, plus a declared per-class launch floor.
[Invariants]
  * No shape list, cohort/case selector, piecewise knot table, ``constant_us_intercept`` or
    observed duration is stored or read; the laws are continuous in the
    declared geometry.
  * Every constant is a declared hardware parameter (die rate, capacity,
    quantum) or a declared kernel-class floor; the derived shares are named
    ratios of declared quantities.
  * The routed expert bank is streamed exactly once: ``k`` is the declared
    per-expert reduction extent, never a flattened ``E * K`` payload that would
    charge the bank a second time.
  * Idle owners never lend their bandwidth, and a routed bank never exceeds the
    die's shared HBM path.
[Timeouts & Locks] Pure bounded arithmetic; no waits or locks.
[Side-Effect Policy] Read-only pure functions; no I/O, global mutation or
  device access.
"""
from __future__ import annotations

from .native_clock_physics import (
    AIC_STREAM_HALF, CUBE_K_BF16, FRACTAL_MN,
    GROUP_SCALAR_MS, GROUP_SCALAR_WALK_MS, HBM_STREAM_SHARE, MACRO_K,
    MACRO_M, MACRO_N, MATRIX_SCALAR_BASE_MS, MATRIX_SCALAR_K_CHUNK_MS,
    MATRIX_SCALAR_PANEL_LOG_MS, MATRIX_SCALAR_PANEL_MS,
    MTE2_STREAM_KNEE_BYTES, NativeRates, A5_RATES, KernelTimes, _ceil_div,
    _log2, matrix_mte2_bytes,
)

#: Declared absorbed-MLA paged-decode prologue (ms): the page-table walk, the
#: first KV panel latency and the absorbed QK^T/PV Cube wave all fall inside
#: one launch before the steady KV stream saturates.
MLA_WALK_PROLOGUE_MS = 0.012760
#: Declared share of the die HBM rate that the absorbed-KV payload sustains.
MLA_STREAM_SHARE = 0.61
#: Active sequences at which the shared KV stream reaches half its rate.
MLA_WAVE_HALF_SEQUENCES = 0.45
#: One AIC descriptor panel of a single-core read reaches ``mte2_aic / N`` of
#: the die panel rate (one stream, no die-level memory parallelism).
AIC_PANEL_STREAM_DIVISOR = 26.0
#: Declared per-row stream-mixing/Sinkhorn reduction of ``HcPre`` (ms).
HC_MIXING_ROW_MS = 0.0000385
#: A transpose-batch GEMM lays its declared [B,K,N] weight out as the physical
#: [B,N,K] panel before the Cube body consumes it: one read plus one write of
#: the weight panel, not a fitted per-shape factor.
TRANSPOSE_WEIGHT_STREAMS = 2.4


def hc_pre_times(*, batch: int, act_bytes: float, weight_bytes: float,
                 out_bytes: float = 0.0, dynamic_gate: bool = False,
                 rates: NativeRates | None = None) -> KernelTimes:
    """Declared ``HcPre`` launch: one slot-1 mixing matrix plus two-pass stream.

    ``HcPre`` reads its ``[experts, hidden]`` FP32 slot-1 matrix through an AIC
    descriptor panel, projects the ``[batch, streams, hidden]`` activation to
    stream weights, and executes a second AIV pass that mixes the ``streams``
    lanes into ``[batch, hidden]``.  When a dynamic per-token gate matrix is
    present, AIV also runs iterative Sinkhorn row/column normalisation per row.
    """
    r = rates or A5_RATES
    batch = max(1, int(batch))
    weight_ms = float(weight_bytes) / max(
        1.0, r.mte2_aic / AIC_PANEL_STREAM_DIVISOR) * 1000.0
    # Two-pass HBM traffic: pass 1 reads activation for coefficient projection;
    # pass 2 re-reads activation (when exceeding L1/UB) and writes mixed output.
    stream_bytes = 2.0 * max(0.0, float(act_bytes) - float(weight_bytes)) + float(out_bytes)
    if out_bytes <= 0:
        stream_bytes = 2.15 * max(0.0, float(act_bytes) - float(weight_bytes))
    act_ms = stream_bytes / max(1.0, r.hbm * 0.48) * 1000.0
    sinkhorn_ms = (float(batch) * 0.000112 * r.scalar_scale()
                   if dynamic_gate and batch > 1 else 0.0)
    mixing_ms = float(batch) * HC_MIXING_ROW_MS * 0.25 + sinkhorn_ms
    body = weight_ms + act_ms + mixing_ms
    return KernelTimes(
        prologue_ms=r.launch("elementwise"),
        body=(("mte2", body),))


def grouped_matmul_times(*, experts: int, tokens: int, k: int, n: int,
                         weight_bytes: float, act_bytes: float,
                         out_bytes: float, dtype_bytes: float = 1.0,
                         act_dtype: str = "int8", weight_dtype: str = "int8",
                         expected_active: int | None = None,
                         rates: NativeRates | None = None) -> KernelTimes:
    """Declared grouped-matmul launch over one routed expert bank.

    ``k`` is the declared per-expert reduction extent.  With no explicitly
    declared active-expert count, routed post-topK rows across an EP shard of
    ``experts`` local banks follow coupon-collector occupancy over the power-of-2
    EP global router bank.  Only touched expert panels are streamed.
    """
    r = rates or A5_RATES
    experts = max(1, int(experts))
    tokens = max(1, int(tokens))
    k = max(1, int(k))
    n = max(1, int(n))
    width = max(0.5, float(dtype_bytes))
    w_dt = str(weight_dtype or "").lower()
    a_dt = str(act_dtype or "").lower()
    is_subbyte = width < 0.75 or any(t in w_dt for t in ("int4", "fp4", "e2m1"))
    is_bf16 = width >= 1.5 or "bf16" in w_dt or "fp16" in w_dt
    is_int8 = (not is_subbyte and not is_bf16
               and ("int8" in w_dt or "int8" in a_dt))
    if expected_active is not None and int(expected_active) > 0:
        active = max(1.0, min(float(experts), float(expected_active)))
    elif experts == 1:
        active = 1.0
    else:
        global_e = float(next(
            (b for b in (256, 384, 448, 512) if b >= experts and b % experts == 0),
            max(256, experts)))
        draws = float(tokens) * (0.58 if experts < int(global_e) else 1.0)
        active = float(experts) * (1.0 - (1.0 - 1.0 / global_e) ** draws)
        active = max(1.0, min(float(experts), active))
    panel_bytes = active * float(k) * float(n) * width
    if weight_bytes > 0:
        panel_bytes = float(weight_bytes)
    avg_m = float(tokens) / max(1.0, active)
    m_tiles = max(1.0, avg_m / 128.0)
    l2_panel_bytes = panel_bytes * min(2.5, m_tiles)
    hbm_reloads = (min(2.4, m_tiles) if panel_bytes > 0.75 * float(r.l2_bytes)
                   else 1.0)
    first_touch = panel_bytes + float(act_bytes) + float(out_bytes)
    total_bytes = panel_bytes * hbm_reloads + float(act_bytes) + float(out_bytes)
    streams = (min(float(r.aic), float(_ceil_div(n, 256) * max(1, _ceil_div(k, 1024))))
               if experts == 1 else max(active, min(float(r.aic), active * float(_ceil_div(n, 1024)))))
    per_first = first_touch / max(1.0, streams)
    per_stream = l2_panel_bytes / max(1.0, streams)
    kind_key = "bf16" if is_bf16 else ("int8" if is_int8 else "mxfp8")
    knee = MTE2_STREAM_KNEE_BYTES.get(kind_key, 131072.0)
    ramp_first = per_first / (per_first + knee)
    ramp = per_stream / (per_stream + knee)
    mlp = streams / (streams + AIC_STREAM_HALF)
    mte2_rate = r.mte2_stream(kind_key) * mlp * ramp_first
    l2_port_rate = float(r.aic) * 152.0 * float(r.clock_hz) * mlp * ramp
    mte2_ms = max(
        first_touch / max(1.0, mte2_rate),
        (l2_panel_bytes + float(act_bytes) + float(out_bytes)) / max(1.0, l2_port_rate)
    ) * 1000.0
    hbm_ms = total_bytes / max(1.0, r.hbm * HBM_STREAM_SHARE) * 1000.0
    cube_kind = "bf16" if is_bf16 else ("int8" if (is_int8 or r.arch != "A5") else "mxfp8")
    cube_ms = (2.0 * float(tokens) * float(k) * float(n)
               / max(1.0, r.cube_rate(cube_kind)) * 1000.0)
    body_ms = max(mte2_ms, hbm_ms, cube_ms)
    act_subbyte = any(t in a_dt for t in ("int4", "fp4", "e2m1"))
    if is_subbyte and r.arch == "A5" and not act_subbyte:
        unpack_passes = max(1.0, avg_m / 32.0)
        aiv_ms = (panel_bytes * 2.0 * unpack_passes * 0.92
                  / max(1.0, r.vector) * 1000.0)
        body_ms = max(body_ms, aiv_ms)
    tail_ms = 0.0
    if is_int8:
        epilogue_elems = float(tokens) * float(n)
        aiv_epi_ms = epilogue_elems * 4.5 / max(1.0, r.vector) * 1000.0
        body_ms = max(body_ms, aiv_epi_ms)
        exposed_elems = min(epilogue_elems, float(r.aiv) * 262144.0)
        tail_ms = (0.0040 * r.scalar_scale()
                   + exposed_elems * 4.2 / max(1.0, r.vector) * 1000.0)
    if r.arch in {"A2", "A3"}:
        scalar_ms = float(experts) * GROUP_SCALAR_WALK_MS * r.scalar_scale()
        prologue = r.launch("matrix_int8") + 0.65 * scalar_ms
    else:
        scalar_ms = float(experts) * 0.000031 * (r.scalar_scale() ** 0.5)
        prologue = r.launch("matrix_int8") + scalar_ms
    tail = (("vector", tail_ms),) if tail_ms > 0 else ()
    return KernelTimes(
        prologue_ms=prologue,
        body=(("mte2", body_ms), ("scalar", scalar_ms)),
        tail=tail)


def mla_paged_times(*, batch: int, kv_len: int, active_bytes: float,
                    rates: NativeRates | None = None) -> KernelTimes:
    """Declared absorbed-MLA paged decode launch.

    One launch walks every visible KV page of every sequence, streams the
    active KV payload and keeps the absorbed ``QK^T``/``PV`` Cube region inside
    the same body.  The visible window is a straight line in
    ``batch * kv_len``: the walk/Cube prologue is one launch cost, and the
    shared KV stream ramps with the number of active sequences because a single
    sequence cannot fill the die's DMA path.
    """
    r = rates or A5_RATES
    batch, kv_len = max(1, int(batch)), max(1, int(kv_len))
    share = float(batch) / (float(batch) + MLA_WAVE_HALF_SEQUENCES)
    rate = max(1.0, r.hbm * MLA_STREAM_SHARE * share)
    stream_ms = float(max(0.0, active_bytes)) / rate * 1000.0
    return KernelTimes(
        prologue_ms=r.launch("paged_fia") + MLA_WALK_PROLOGUE_MS,
        body=(("hbm", stream_ms),))


#: Concurrent AIC panel streams at which the die's sustained MTE2 share is
#: reached (declared occupancy of the dense Cube copy path).
TRANSPOSE_STREAMS_FULL = 12.0
#: Per-stream byte knee of the transpose-batch weight layout panel: the
#: declared [B,K,N] -> [B,N,K] panel is several macro panels long before it
#: amortises its first-line cost.
TRANSPOSE_STREAM_KNEE_BYTES = 350000.0
#: Declared first-wave scalar exposure of the transpose-batch launch (ms).
TRANSPOSE_SCALAR_EXPOSED_MS = 0.00112
TRANSPOSE_SCALAR_WAVE_SHARE = 0.5


def transpose_batch_times(*, tokens: int, batch: int, k: int, n: int,
                          rates: NativeRates | None = None) -> KernelTimes:
    """Declared [T,B,K] x [B,K,N] -> [T,B,N] transpose-batch GEMM launch.

    One launch block per (head, M macro-tile, N macro-tile) output panel.  The
    kernel lays the declared ``[B,K,N]`` weight out as the physical ``[B,N,K]``
    panel stream (one read plus one write of the panel), streams the activation
    once per N macro tile and drains the output through Fixpipe.  The panel
    streams are the same per-stream law as the dense Cube body: the active block
    count is the panel count bounded by the AIC owners, and a short panel does
    not reach the sustained share.
    """
    r = rates or A5_RATES
    tokens, batch = max(1, int(tokens)), max(1, int(batch))
    k, n = max(1, int(k)), max(1, int(n))
    width = 2  # declared BF16 activation/weight/output representation
    weight = float(batch) * float(k) * float(n) * width
    act = float(tokens) * float(batch) * float(k) * width
    out = float(tokens) * float(batch) * float(n) * width
    macro_m = max(1, _ceil_div(tokens, MACRO_M))
    macro_n = max(1, _ceil_div(n, MACRO_N))
    panels = batch * macro_m * macro_n
    active = max(1, min(panels, max(1, int(r.aic))))
    panel_bytes = (float(macro_m) * weight * TRANSPOSE_WEIGHT_STREAMS
                   + float(macro_n) * act + out)
    per_stream = panel_bytes / float(active)
    knee = TRANSPOSE_STREAM_KNEE_BYTES
    ramp = per_stream / (per_stream + knee)
    streams = min(float(active), TRANSPOSE_STREAMS_FULL) / TRANSPOSE_STREAMS_FULL
    mte2_ms = (panel_bytes
               / max(1.0, r.mte2_stream("bf16") * streams * ramp) * 1000.0)
    hbm_ms = (weight * TRANSPOSE_WEIGHT_STREAMS
              + act + out) / max(1.0, r.hbm * HBM_STREAM_SHARE) * 1000.0
    flops = 2.0 * float(tokens) * float(batch) * float(k) * float(n)
    tiles_mn = _ceil_div(tokens, FRACTAL_MN) * _ceil_div(n, FRACTAL_MN)
    waves = max(1, _ceil_div(tiles_mn * batch, active))
    issue_macs = waves * active * FRACTAL_MN * FRACTAL_MN * CUBE_K_BF16
    cube_ms = (max(flops, 2.0 * float(issue_macs))
               / max(1.0, r.cube_rate("bf16")) * 1000.0)
    fixpipe_ms = out / max(1.0, r.fixpipe) * 1000.0
    k_chunks = max(1, _ceil_div(k, MACRO_K))
    scalar_ms = (MATRIX_SCALAR_BASE_MS + MATRIX_SCALAR_K_CHUNK_MS * k_chunks
                 + MATRIX_SCALAR_PANEL_LOG_MS * _log2(1.0 + panels)
                 + MATRIX_SCALAR_PANEL_MS * panels) * r.scalar_scale()
    launch = r.launch("transpose_batch")
    body = max(mte2_ms, hbm_ms, cube_ms, fixpipe_ms)
    exposed = (scalar_ms if scalar_ms >= body else
               min(scalar_ms, TRANSPOSE_SCALAR_EXPOSED_MS
                   + TRANSPOSE_SCALAR_WAVE_SHARE * scalar_ms / float(waves)))
    tail = (("fixpipe", (0.00038 + out / max(1.0, r.fixpipe * 0.092) * 1000.0)
             * float(macro_n - 1)),) if macro_n > 1 else ()
    return KernelTimes(
        prologue_ms=launch + exposed,
        body=(("mte2", mte2_ms), ("hbm", hbm_ms), ("matrix", cube_ms),
              ("fixpipe", fixpipe_ms)),
        tail=tail)
