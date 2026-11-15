"""Formula builders for normalization, activation, RoPE, quant and MoE routing.

[Role]
  Cover the Vector-pipe and routing families with exact per-output-element lane
  operation counts: RMSNorm/AddRmsNorm/QkRmsNorm, SwiGLU/SiluAndMul/GeGLU,
  RotaryPositionEmbedding with an explicit ``rotary_dim``, DynamicQuant /
  DequantSwigluQuant, and MoeGatingTopKSoftmax / MoeInitRoutingV2 /
  MoeFinalizeRouting.

[Invariants]
  * Every operator here is Vector-only: zero Cube MACs and zero tiled Cube FLOPs.
  * Vector work is an explicit element-operation count derived from the math
    (5 per RMSNorm element, 5 per SwiGLU output, 8 per GeGLU output, 3 per
    rotated RoPE element, 4 per quant element); it is never a fudge coefficient.
  * Every GM buffer is aligned independently, so ``granule`` equals the sum of
    ``align_up(buffer, granule)`` and never a clamped aggregate.
  * RoPE validates ``0 < rotary_dim <= head_dim`` and evenness; rotation work
    scales with ``rotary_dim`` while pass-through traffic scales with ``head_dim``.
  * Routing uses ``log2_ceil(E)`` (exact ``ceil(log2(E))``), reads logits once and
    validates ``1 <= topk <= experts``.

[Timeouts & Locks]
  Pure arithmetic; bounded by declared dimensions, no I/O.

[Side-Effect Policy]
  No mutation or device work.
"""

from __future__ import annotations

from typing import Any

from .errors import FormulaValidationError
from .expr_ast import ZERO, Const, log2_ceil
from .hardware import HardwareProfile, dtype_bytes
from .primitives import (
    align_up, ceil_div, coalesce_efficiency, maybe_l2_hot, row_bytes,
    row_bytes_expr, strict_bool, strict_int, tensor_bytes, wave_efficiency,
)
from .work import Algebra, ByteAccount, OpWork


def _hot(hardware: HardwareProfile, steady: bool, sizes: tuple[int, ...]) -> int:
    if not steady:
        return 0
    budget = hardware.l2_hot_budget()
    total = 0
    for size in sizes:
        grant = maybe_l2_hot(True, int(size), budget)
        total += grant
        budget -= grant
    return total


def _tiles(hardware: HardwareProfile, elements: int, minimum: int = 1,
           per_tile: int = 8192) -> tuple[int, int, float]:
    tiles = max(minimum, ceil_div(max(1, elements), per_tile))
    waves, wave_eff = wave_efficiency(tiles, hardware.cores)
    return tiles, waves, wave_eff


def rms_norm_work(*, rows: int, hidden: int, dtype: str = "bf16",
                  gamma: bool = True, residual: bool = False,
                  inplace: bool = False, qk: bool = False, groups: int = 1,
                  gamma_dtype: str | None = None,
                  gamma_elems: int | None = None,
                  out_dtype: str | None = None,
                  residual_dtype: str | None = None,
                  x_out_dtype: str | None = None,
                  rstd_elems: int = 0, rstd_dtype: str = "fp32",
                  steady_l2_warm: bool = False,
                  hardware: HardwareProfile) -> OpWork:
    """RmsNorm, AddRmsNorm (fused residual) and QkRmsNorm."""
    r_i = strict_int(rows, "rows", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    g_i = strict_int(groups, "groups", minimum=1)
    residual_b = strict_bool(residual, "residual")
    qk_b = strict_bool(qk, "qk")
    gamma_b = strict_bool(gamma, "gamma")
    bpe_i = dtype_bytes(dtype)
    gamma_bpe = dtype_bytes(gamma_dtype or dtype)
    out_bpe = dtype_bytes(out_dtype or dtype)
    residual_bpe = dtype_bytes(residual_dtype or dtype)
    x_out_bpe = dtype_bytes(x_out_dtype or dtype)
    rstd_bpe = dtype_bytes(rstd_dtype)
    rstd_i = strict_int(rstd_elems, "rstd_elems", minimum=0)
    gamma_count = (h_i * g_i if gamma_elems is None
                   else strict_int(gamma_elems, "gamma_elems", minimum=1))
    ops_per_elem = 5 + (2 if residual_b else 0)
    granule = hardware.granule_for("ND")
    al = Algebra()
    R = al.sym("rows", r_i)
    H = al.sym("hidden", h_i)
    G = al.sym("groups", g_i)
    OPS = al.sym("ops_per_element", ops_per_elem)
    BPE = al.real("dtype_bytes", bpe_i)
    GB = al.real("gamma_bytes", gamma_bpe)
    OB = al.real("out_bytes", out_bpe)
    RB = al.real("rstd_bytes", rstd_bpe)
    RESB = al.real("residual_bytes", residual_bpe)
    XB = al.real("x_out_bytes", x_out_bpe)
    width = H * G
    account = ByteAccount(granule)
    account.read_buffer(R * width * BPE, 1.0, rows=R,
                        width=row_bytes_expr(width, bpe_i))
    if residual_b:
        account.read_buffer(R * width * RESB, 1.0, rows=R,
                            width=row_bytes_expr(width, residual_bpe))
    if gamma_b:
        account.read_buffer(gamma_count * GB, 1.0)
    account.write_buffer(R * width * OB, 1.0,
                         rows=R, width=row_bytes_expr(width, out_bpe))
    if residual_b:
        account.write_buffer(R * width * XB, 1.0, rows=R,
                             width=row_bytes_expr(width, x_out_bpe))
    if rstd_i:
        account.write_buffer(rstd_i * RB, 1.0)
    vector = OPS * R * width
    elements = r_i * h_i * g_i
    tiles, waves, wave_eff = _tiles(hardware, elements, minimum=r_i)
    hot = _hot(hardware, steady_l2_warm,
               (r_i * h_i * g_i * bpe_i,
                int(tensor_bytes((h_i * g_i,), bpe_i)) if gamma_b else 0))
    variant = ("add_rms_norm" if residual_b else
               "qk_rms_norm" if qk_b else "rms_norm")
    notes = [f"{ops_per_elem} Vector ops per element",
             "gamma is a single aligned row" if gamma_b else "no gamma tensor"]
    if residual_b:
        notes.append("fused residual: residual_in read and residual_out written")
    if inplace:
        notes.append("in-place destination still charges the required output write")
    return OpWork(
        family="rms_norm", variant=variant,
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=tiles,
        n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(h_i * g_i, bpe_i),
                                               granule),
        m_tiled=align_up(r_i, 16), n_tiled=align_up(h_i * g_i, 16),
        k_tiled=1, tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
        notes=tuple(notes),
        bindings={**al.merged(), "elements": elements, "n_tiles": tiles,
                  "gamma_elems": gamma_count, "rstd_elems": rstd_i,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


def swiglu_work(*, rows: int, hidden: int, mode: str = "swiglu",
                dtype: str = "bf16", inplace: bool = False,
                two_input: bool = False, out_dtype: str | None = None,
                first_elems: int | None = None,
                second_elems: int | None = None,
                steady_l2_warm: bool = False,
                hardware: HardwareProfile) -> OpWork:
    """SwiGLU / SiluAndMul / GeGLU activation and gated multiply."""
    r_i = strict_int(rows, "rows", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    if mode not in {"swiglu", "packed", "silu_and_mul", "geglu", "silu",
                    "gate_mul"}:
        raise FormulaValidationError(
            "mode must be swiglu/packed/silu_and_mul/geglu/silu/gate_mul")
    bpe_i = dtype_bytes(dtype)
    out_bpe = dtype_bytes(out_dtype or dtype)
    arity2 = strict_bool(two_input, "two_input") or mode == "gate_mul"
    if mode == "geglu":
        ops_per_elem = 8
    elif mode == "gate_mul":
        ops_per_elem = 1
    else:
        ops_per_elem = 5
    gated = mode != "silu"
    granule = hardware.granule_for("ND")
    al = Algebra()
    R = al.sym("rows", r_i)
    H = al.sym("hidden", h_i)
    OPS = al.sym("ops_per_element", ops_per_elem)
    BPE = al.real("dtype_bytes", bpe_i)
    OB = al.real("out_bytes", out_bpe)
    read_w = 2 * H if gated else H
    read_w_num = 2 * h_i if (gated or arity2) else h_i
    account = ByteAccount(granule)
    if arity2:
        # Two independent input tensors (gate and up) each own their granules.
        first_n = (r_i * h_i if first_elems is None
                   else strict_int(first_elems, "first_elems", minimum=0))
        second_n = (r_i * h_i if second_elems is None
                    else strict_int(second_elems, "second_elems", minimum=0))
        account.read_buffer(first_n * BPE, 1.0)
        account.read_buffer(second_n * BPE, 1.0)
    else:
        account.read_buffer(R * read_w * BPE, 1.0, rows=R,
                            width=row_bytes_expr(read_w, bpe_i))
    account.write_buffer(R * H * OB, 1.0, rows=R,
                         width=row_bytes_expr(H, out_bpe))
    vector = OPS * R * H
    elements = r_i * h_i
    tiles, waves, wave_eff = _tiles(hardware, elements)
    hot = _hot(hardware, steady_l2_warm,
               (int(tensor_bytes((r_i, read_w_num), bpe_i)),
                int(tensor_bytes((r_i, h_i), bpe_i))))
    return OpWork(
        family="swiglu", variant=mode,
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=tiles,
        n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(read_w_num, bpe_i), granule),
        m_tiled=align_up(r_i, 16), n_tiled=align_up(h_i, 16), k_tiled=1,
        tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
        notes=(f"{ops_per_elem} Vector ops per gated output element",
               "in-place gated output" if inplace else "out-of-place gated output"),
        bindings={**al.merged(), "elements": elements, "n_tiles": tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff, "mode": mode,
                  "two_input": arity2},
    )


def rope_work(*, tokens: int, heads: int, head_dim: int,
              rotary_dim: int | None = None, dtype: str = "bf16",
              qk: bool = True, fused: bool = True,
              pass_through_copy: bool | None = None,
              steady_l2_warm: bool = False,
              hardware: HardwareProfile) -> OpWork:
    """RotaryPositionEmbedding / FusedRoPE with explicit ``rotary_dim``."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(heads, "heads", minimum=1)
    d_i = strict_int(head_dim, "head_dim", minimum=1)
    rot_i = d_i if rotary_dim is None else strict_int(rotary_dim, "rotary_dim",
                                                      minimum=1)
    if rot_i > d_i:
        raise FormulaValidationError("rotary_dim cannot exceed head_dim")
    if rot_i % 2:
        raise FormulaValidationError("rotary_dim must be even")
    qk_b = strict_bool(qk, "qk")
    fused_b = strict_bool(fused, "fused")
    copy_b = (not fused_b) if pass_through_copy is None else strict_bool(
        pass_through_copy, "pass_through_copy")
    bpe_i = dtype_bytes(dtype)
    tensors = 2 if qk_b else 1
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    HQ = al.sym("heads_q", h_i)
    HK = al.sym("heads_k", h_i if qk_b else 0)
    RD = al.sym("rotary_dim", rot_i)
    BPE = al.real("dtype_bytes", bpe_i)
    account = ByteAccount(granule)
    traffic_d = d_i if copy_b else rot_i
    rows = t_i * h_i * tensors
    account.read_buffer(T * (HQ + HK) * traffic_d * BPE, 1.0, rows=rows,
                        width=row_bytes_expr(traffic_d, bpe_i))
    account.read_buffer(T * rot_i * 4 * 2 // 2, 1.0)
    account.write_buffer(T * (HQ + HK) * traffic_d * BPE, 1.0, rows=rows,
                         width=row_bytes_expr(traffic_d, bpe_i))
    vector = 3 * T * (HQ + HK) * RD
    elements = rows * traffic_d
    tiles, waves, wave_eff = _tiles(hardware, elements, minimum=t_i * h_i)
    cos_sin = int(tensor_bytes((t_i, rot_i), 4))
    hot = _hot(hardware, steady_l2_warm, (cos_sin,))
    return OpWork(
        family="rotary_embedding", variant="fused_qk" if qk_b else "q_only",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=tiles,
        n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(traffic_d, bpe_i),
                                               granule),
        m_tiled=align_up(t_i, 16), n_tiled=align_up(h_i * traffic_d, 16),
        k_tiled=1, tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
        notes=(f"rotation work scales with rotary_dim={rot_i}",
               "pass-through copies the full head_dim" if copy_b
               else "only the rotary slice is read/written"),
        bindings={**al.merged(), "head_dim": d_i, "cos_sin_bytes": cos_sin,
                  "n_tiles": tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


def slice_work(*, rows: int, in_hidden: int, out_hidden: int,
               dtype: str = "bf16", steady_l2_warm: bool = False,
               hardware: HardwareProfile) -> OpWork:
    """Pure data-copy/view slice: read the source, write the narrowed view."""
    r_i = strict_int(rows, "rows", minimum=1)
    in_i = strict_int(in_hidden, "in_hidden", minimum=1)
    out_i = strict_int(out_hidden, "out_hidden", minimum=1)
    if out_i > in_i:
        raise FormulaValidationError("out_hidden cannot exceed in_hidden")
    bpe_i = dtype_bytes(dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    R = al.sym("rows", r_i)
    IN = al.sym("in_hidden", in_i)
    OUT = al.sym("out_hidden", out_i)
    BPE = al.real("dtype_bytes", bpe_i)
    account = ByteAccount(granule)
    # A slice/view only touches the selected elements: it reads and writes the
    # narrowed view, not the whole source row.
    account.read_buffer(R * OUT * BPE, 1.0, rows=R,
                        width=row_bytes_expr(OUT, bpe_i))
    account.write_buffer(R * OUT * BPE, 1.0, rows=R,
                         width=row_bytes_expr(OUT, bpe_i))
    tiles, waves, wave_eff = _tiles(hardware, r_i * in_i, minimum=1)
    return OpWork(
        family="slice", variant="view_copy",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=ZERO,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=tiles, n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(in_i, bpe_i),
                                               granule),
        m_tiled=align_up(r_i, 16), n_tiled=align_up(in_i, 16), k_tiled=1,
        tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
        notes=("pure slice/view data movement",
               "no activation Vector arithmetic is fabricated"),
        bindings={**al.merged(), "n_tiles": tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


def dynamic_quant_work(*, rows: int, hidden: int, dtype: str = "bf16",
                       out_dtype: str = "int8", dequant_swiglu: bool = False,
                       has_smooth_scales: bool = False,
                       smooth_scale_bytes: int | None = None,
                       steady_l2_warm: bool = False,
                       hardware: HardwareProfile) -> OpWork:
    """DynamicQuant or fused DequantSwigluQuant."""
    r_i = strict_int(rows, "rows", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    dequant = strict_bool(dequant_swiglu, "dequant_swiglu")
    smooth = strict_bool(has_smooth_scales, "has_smooth_scales")
    if smooth_scale_bytes is None:
        smooth_bytes = r_i * 4 if smooth else 0
    else:
        smooth_bytes = strict_int(smooth_scale_bytes, "smooth_scale_bytes",
                                  minimum=0)
        if not smooth and smooth_bytes:
            raise FormulaValidationError(
                "smooth_scale_bytes requires has_smooth_scales=True")
    bpe_i = dtype_bytes(dtype)
    out_bpe = dtype_bytes(out_dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    R = al.sym("rows", r_i)
    H = al.sym("hidden", h_i)
    OPS = al.sym("ops_per_element", 8 if dequant else 4)
    BPE = al.real("dtype_bytes", bpe_i)
    OB = al.real("out_bytes", out_bpe)
    in_w = 2 * H if dequant else H
    in_w_num = 2 * h_i if dequant else h_i
    account = ByteAccount(granule)
    account.read_buffer(R * in_w * BPE, 1.0, rows=R,
                        width=row_bytes_expr(in_w, bpe_i))
    if smooth and smooth_bytes:
        # Optional smoothing operand only; output per-row scales are produced
        # by the reduction and are never invented as an input tensor.
        account.read_buffer(smooth_bytes, 1.0)
    account.write_buffer(R * H * OB, 1.0, rows=R, width=row_bytes_expr(H, out_bpe))
    account.write_buffer(R * 4, 1.0)
    vector = OPS * R * H
    elements = r_i * h_i
    tiles, waves, wave_eff = _tiles(hardware, elements)
    hot = _hot(hardware, steady_l2_warm,
               (int(tensor_bytes((r_i, in_w_num), bpe_i)),
                int(tensor_bytes((r_i, h_i), out_bpe))))
    return OpWork(
        family="dynamic_quant",
        variant="dequant_swiglu_quant" if dequant else "dynamic_quant",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=tiles,
        n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(in_w_num, bpe_i), granule),
        m_tiled=align_up(r_i, 16), n_tiled=align_up(h_i, 16), k_tiled=1,
        tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
        notes=("per-token abs-max dynamic scale" if not dequant else
               "dequant + SwiGLU + re-quant in one pass",),
        bindings={**al.merged(), "elements": elements, "n_tiles": tiles,
                  "has_smooth_scales": smooth,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


def moe_routing_work(*, tokens: int, experts: int, hidden: int,
                     topk: int = 8, mode: str = "gating",
                     dtype: str = "bf16", steady_l2_warm: bool = False,
                     hardware: HardwareProfile) -> OpWork:
    """MoeGatingTopKSoftmax / MoeInitRoutingV2 / MoeFinalizeRouting."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    e_i = strict_int(experts, "experts", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    k_i = strict_int(topk, "topk", minimum=1)
    if k_i > e_i:
        raise FormulaValidationError(
            f"topk={k_i} cannot exceed experts={e_i}")
    bpe_i = dtype_bytes(dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    E = al.sym("experts", e_i)
    H = al.sym("hidden", h_i)
    K = al.sym("topk", k_i)
    BPE = al.real("dtype_bytes", bpe_i)
    if mode == "gating":
        account = ByteAccount(granule)
        account.read_buffer(T * E * BPE, 1.0, rows=T, width=row_bytes_expr(E, bpe_i))
        account.write_buffer(T * K * BPE, 1.0, rows=T, width=row_bytes_expr(K, bpe_i))
        account.write_buffer(T * K * 4, 1.0, rows=T, width=K * 4)
        vector = 5 * T * E + T * E * log2_ceil(E)
        elements = t_i * e_i
        tiles, waves, wave_eff = _tiles(hardware, elements, minimum=t_i)
        hot = _hot(hardware, steady_l2_warm, (int(tensor_bytes((t_i, e_i), bpe_i)),))
        return OpWork(
            family="moe_gating", variant="topk_softmax",
            cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
            read_bytes=account.read, write_bytes=account.write,
            granule_read_bytes=account.granule_read,
            granule_write_bytes=account.granule_write,
            l2_hot_bytes=Const(hot), n_tiles=tiles,
            n_cores=hardware.cores,
            coalesce_efficiency=coalesce_efficiency(row_bytes(e_i, bpe_i), granule),
            m_tiled=align_up(t_i, 16), n_tiled=align_up(e_i, 16), k_tiled=1,
            tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
            notes=("router logits read once; log2_ceil(E) exact top-k scan",
                   "indices are int32"),
            bindings={**al.merged(), "n_tiles": tiles, "n_waves": waves,
                      "wave_efficiency": wave_eff},
        )
    if mode == "init_routing":
        account = ByteAccount(granule)
        account.read_buffer(T * H * BPE, 1.0, rows=T, width=row_bytes_expr(H, bpe_i))
        account.read_buffer(T * K * 4, 1.0, rows=T, width=K * 4)
        account.write_buffer(T * K * H * BPE, 1.0, rows=T * K,
                             width=row_bytes_expr(H, bpe_i))
        account.write_buffer(E * 4, 1.0)
        vector = 2 * T * K * H
        elements = t_i * k_i * h_i
        tiles, waves, wave_eff = _tiles(hardware, elements, minimum=t_i * k_i)
        hot = _hot(hardware, steady_l2_warm, (int(tensor_bytes((t_i, h_i), bpe_i)),))
        return OpWork(
            family="moe_gating", variant="init_routing",
            cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
            read_bytes=account.read, write_bytes=account.write,
            granule_read_bytes=account.granule_read,
            granule_write_bytes=account.granule_write,
            l2_hot_bytes=Const(hot), n_tiles=tiles,
            n_cores=hardware.cores,
            coalesce_efficiency=coalesce_efficiency(row_bytes(h_i, bpe_i), granule),
            m_tiled=align_up(t_i, 16), n_tiled=align_up(k_i * h_i, 16),
            k_tiled=1, tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
            notes=("expanded buffer is T*K*H, not T*E*H",),
            bindings={**al.merged(), "n_tiles": tiles, "n_waves": waves,
                      "wave_efficiency": wave_eff},
        )
    if mode == "finalize":
        account = ByteAccount(granule)
        account.read_buffer(T * K * H * BPE, 1.0, rows=T * K,
                            width=row_bytes_expr(H, bpe_i))
        account.read_buffer(T * K * BPE, 1.0, rows=T, width=row_bytes_expr(K, bpe_i))
        account.write_buffer(T * H * BPE, 1.0, rows=T,
                             width=row_bytes_expr(H, bpe_i))
        vector = 3 * T * K * H
        elements = t_i * k_i * h_i
        tiles, waves, wave_eff = _tiles(hardware, elements, minimum=t_i)
        hot = _hot(hardware, steady_l2_warm,
                   (int(tensor_bytes((t_i, k_i, h_i), bpe_i)),))
        return OpWork(
            family="moe_gating", variant="finalize_routing",
            cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
            read_bytes=account.read, write_bytes=account.write,
            granule_read_bytes=account.granule_read,
            granule_write_bytes=account.granule_write,
            l2_hot_bytes=Const(hot), n_tiles=tiles,
            n_cores=hardware.cores, coalesce_efficiency=1.0,
            m_tiled=align_up(t_i, 16), n_tiled=align_up(k_i * h_i, 16),
            k_tiled=1, tile_m=1, tile_n=128, tile_k=1, structure="vector_only",
            notes=("top-k weighted reduction over unique tokens",),
            bindings={**al.merged(), "n_tiles": tiles, "n_waves": waves,
                      "wave_efficiency": wave_eff},
        )
    raise FormulaValidationError(
        "moe routing mode must be gating/init_routing/finalize")


__all__ = [
    "dynamic_quant_work", "moe_routing_work", "rms_norm_work", "rope_work",
    "slice_work", "swiglu_work",
]
