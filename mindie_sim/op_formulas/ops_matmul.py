"""Formula builders for dense/quant GEMM, GroupedMatmul and LMHead/samplers.

[Role]
  Translate the tensor contract of dense, quantized and grouped matrix operators
  into exact MAC counts, 16x16-tiled Cube work, declared-precision Cube tiers,
  quantization-scale tensors and per-buffer GM traffic.  GroupedMatmul keeps a
  certified minimum-active-expert floor separate from the uniform-routing
  reference, and honours an exact ``group_list`` histogram (cumulative or
  counts).  LMHead is a pure projection; ArgMax/TopPSample are standalone
  Vector/reduction samplers.

[Invariants]
  * The Cube tier is the real compute precision (FP32 GEMM uses the FP32 peak,
    not BF16), and an unknown dtype/quant mode raises.
  * Activation-quantization Vector work is charged only to quantized GEMMs; a
    dense BF16 GEMM charges only an explicit bias add.
  * Without ``group_list`` the certified floor uses ``min_active_experts`` (a
    guaranteed lower bound over all legal routings), while the reference uses the
    uniform-routing expectation; an exact histogram overrides both.
  * ``group_list_type`` 0 is a non-decreasing cumulative vector, 1 is per-expert
    counts; sums are validated against the routed-row count when it is known.
  * Group sizes are validated per format (MX block fixed at 32).
  * ``lm_head(include_sampler=False)`` charges zero sampler work.

[Timeouts & Locks]
  Pure arithmetic proportional to declared experts; no I/O.

[Side-Effect Policy]
  No mutation; returns a fresh ``OpWork``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .errors import FormulaValidationError
from .expr_ast import (
    ZERO, Const, align_up, ceil_div, log2_ceil, max_expr, sum_align_up,
)
from .hardware import HardwareProfile, cube_tier, dtype_bytes
from .ops_quant import (
    QuantSpec, _KNOWN_QUANT_KEYS, _check_quant_storage, _inferred_quant,
    _norm, _quant_spec, _quant_vector_ops, _weight_scale_expr,
)

from .primitives import (
    align_up as align_up_int, ceil_div as ceil_div_int, coalesce_efficiency,
    row_bytes, row_bytes_expr, strict_bool, strict_int, tensor_bytes, tile_geometry,
    wave_efficiency,
)
from .work import Algebra, ByteAccount, OpWork


def validate_quant_mode(quant: str | None) -> str | None:
    """Validate a declared quant mode against the explicit whitelist."""
    if quant is None:
        return None
    if not isinstance(quant, str) or not quant.strip():
        raise FormulaValidationError("quant mode must be a non-empty string")
    key = _norm(quant)
    if key not in _KNOWN_QUANT_KEYS:
        raise FormulaValidationError(f"unknown quant_mode {quant!r}")
    return key


def validate_dtypes(*values: Any, names: tuple[str, ...] = ()) -> None:
    """Reject every unknown dtype spelling before any quant branch."""
    for index, value in enumerate(values):
        if value is None:
            continue
        label = names[index] if index < len(names) else f"dtype[{index}]"
        dtype_bytes(value if isinstance(value, str) else str(value))
        strict_str = value
        if not isinstance(strict_str, str) or not strict_str.strip():
            raise FormulaValidationError(f"{label} must be a dtype name")


def matmul_work(*, m: int, n: int, k: int, dtype: str = "bf16",
                weight_dtype: str | None = None, out_dtype: str | None = None,
                layout: str = "ND", quant: str | None = None,
                group_size: int | None = None, bias: bool = False,
                zero_points: bool = False, batch: int = 1,
                weight_broadcast: bool = True,
                act_batch: int | None = None,
                weight_batch: int | None = None,
                scale_bytes: int | None = None,
                act_scale_bytes: int | None = None,
                steady_l2_warm: bool = False, hardware: HardwareProfile,
                family: str = "matmul", variant: str = "dense") -> OpWork:
    """Build one dense/quantized matrix-multiply work record."""
    m_i = strict_int(m, "M", minimum=1)
    n_i = strict_int(n, "N", minimum=1)
    k_i = strict_int(k, "K", minimum=1)
    b_i = strict_int(batch, "batch", minimum=1)
    a_batches = (b_i if act_batch is None
                 else strict_int(act_batch, "act_batch", minimum=1))
    broadcast_b = strict_bool(weight_broadcast, "weight_broadcast")
    w_batches_i = (1 if broadcast_b else b_i) if weight_batch is None else \
        strict_int(weight_batch, "weight_batch", minimum=1)
    quant_key = validate_quant_mode(quant)
    validate_dtypes(dtype, weight_dtype, out_dtype,
                    names=("dtype", "weight_dtype", "out_dtype"))
    _check_quant_storage(quant_key, dtype, weight_dtype)
    if quant_key:
        spec = _quant_spec(quant_key, out_dtype, group_size, k_i)
        a_bpe, w_bpe, o_bpe = spec.act_bytes, spec.weight_bytes, spec.out_bytes
        tier = spec.tier
        variant = str(quant)
    else:
        a_bpe = dtype_bytes(dtype)
        w_bpe = dtype_bytes(weight_dtype) if weight_dtype else a_bpe
        o_bpe = dtype_bytes(out_dtype) if out_dtype else a_bpe
        tier = cube_tier(dtype)
        spec = QuantSpec(a_bpe, w_bpe, o_bpe, tier, 0, "none", None, 0)
        variant = str(dtype)
    geom = tile_geometry(m_i, n_i, k_i)
    granule = hardware.granule_for(layout)
    al = Algebra()
    M = al.sym("M", m_i)
    N = al.sym("N", n_i)
    K = al.sym("K", k_i)
    BATCH = al.sym("batch", b_i)
    A_BATCH = al.sym("act_batch", a_batches)
    W_BATCH = al.sym("weight_batch", w_batches_i)
    AB = al.real("dtype_bytes", a_bpe)
    WB = al.real("weight_bytes", w_bpe)
    OB = al.real("out_bytes", o_bpe)
    account = ByteAccount(granule)
    rows_b = BATCH * M
    a_rows = A_BATCH * M
    w_batches = W_BATCH
    account.read_buffer(a_rows * K, a_bpe, rows=a_rows, width=K)
    account.read_buffer(w_batches * K * N, w_bpe,
                        rows=w_batches * K, width=N)
    scale_expr = ZERO
    if quant_key:
        if act_scale_bytes is None:
            act_scale = rows_b * spec.activation_scale_bytes
        else:
            act_scale = strict_int(act_scale_bytes, "act_scale_bytes", minimum=0)
        if act_scale:
            account.read_buffer(act_scale, 1.0)
        if scale_bytes is None:
            scale_part, _ = _weight_scale_expr(al, K, N, spec, zero_points)
            scale_total = w_batches * scale_part
        else:
            scale_total = strict_int(scale_bytes, "scale_bytes", minimum=0)
        account.read_buffer(scale_total, 1.0)
        scale_expr = scale_total
    if bias:
        account.read_buffer(w_batches * N, o_bpe)
    account.write_buffer(rows_b * N, o_bpe, rows=rows_b, width=N)
    macs = BATCH * M * N * K
    tiled = 2 * BATCH * align_up(M, 16) * align_up(N, 16) * align_up(K, 16)
    if quant_key:
        vector = _quant_vector_ops(spec, quant_key, rows=rows_b, k_sym=K,
                                   n_sym=N, weight_batches=w_batches,
                                   bias=bias)
    elif bias:
        vector = rows_b * N
    else:
        vector = ZERO
    n_tiles = max(1, b_i * geom["n_tiles"])
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    hot = 0
    if steady_l2_warm:
        budget = hardware.l2_hot_budget()
        for size in (b_i * m_i * k_i * a_bpe, b_i * m_i * n_i * o_bpe,
                     w_batches_i * k_i * n_i * w_bpe):
            if 0 < int(size) <= budget:
                hot += int(size)
                budget -= int(size)
    row_min = min(int(row_bytes_expr(K, a_bpe).value(al.bindings)),
                  int(row_bytes_expr(N, w_bpe).value(al.bindings)),
                  max(1, int(row_bytes_expr(N, o_bpe).value(al.bindings))))
    coalesce = max(1e-6, min(1.0, row_min / granule))
    bindings = al.merged()
    bindings.update({
        "m_tiled": geom["m_tiled"], "n_tiled": geom["n_tiled"],
        "k_tiled": geom["k_tiled"], "tile_m": geom["tile_m"],
        "tile_n": geom["tile_n"], "tile_k": geom["tile_k"],
        "n_tiles": n_tiles, "n_waves": waves, "n_cores": hardware.cores,
        "wave_efficiency": wave_eff, "coalesce_efficiency": coalesce,
        "compute_tier": tier, "layout": layout, "dtype": str(dtype),
        "weight_broadcast": broadcast_b,
        "act_batch": a_batches, "weight_batch": w_batches_i,
        "group_size": spec.group_size or 0,
        "weight_scale_mode": spec.weight_scale_mode,
    })
    notes = [f"cube tier {tier} from quant={quant or dtype}",
             f"batched GEMM: batch={b_i}, weight_broadcast={broadcast_b}",
             f"block tiles {geom['tile_m']}x{geom['tile_n']}x{geom['tile_k']}"]
    if quant:
        notes.append("activation/weight quantization scale tensors charged")
    return OpWork(
        family=family, variant=variant,
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=n_tiles,
        n_cores=hardware.cores, coalesce_efficiency=coalesce,
        m_tiled=geom["m_tiled"], n_tiled=geom["n_tiled"],
        k_tiled=geom["k_tiled"], tile_m=geom["tile_m"], tile_n=geom["tile_n"],
        tile_k=geom["tile_k"], cube_tier=tier, notes=tuple(notes),
        bindings=bindings,
    )


def decode_group_list(values: Sequence[int], group_list_type: int,
                      experts_local: int,
                      expected_rows: int | None) -> tuple[int, ...]:
    """Decode a cumulative/count histogram and validate it exactly."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FormulaValidationError("group_list must be a sequence of integers")
    numbers = [strict_int(item, "group_list entry", minimum=0) for item in values]
    kind = strict_int(group_list_type, "group_list_type", minimum=0, maximum=1)
    if kind == 0:
        if len(numbers) == experts_local + 1 and numbers and numbers[0] == 0:
            numbers = numbers[1:]
        if len(numbers) != experts_local:
            raise FormulaValidationError(
                f"group_list must have {experts_local} cumulative entries")
        if any(right < left for left, right in zip(numbers, numbers[1:])):
            raise FormulaValidationError("cumulative group_list must be non-decreasing")
        counts = tuple(numbers[0] if index == 0 else numbers[index] - numbers[index - 1]
                       for index in range(len(numbers)))
    else:
        if len(numbers) != experts_local:
            raise FormulaValidationError(
                f"group_list must have {experts_local} count entries")
        counts = tuple(numbers)
    if expected_rows is not None and sum(counts) != expected_rows:
        raise FormulaValidationError(
            f"group_list sums to {sum(counts)} but routed rows are {expected_rows}")
    return counts


def grouped_matmul_work(*, tokens: int, hidden: int, inter: int,
                        experts_total: int, experts_local: int, topk: int,
                        ep: int = 1, mode: str = "up_gate",
                        dtype: str = "bf16", weight_dtype: str | None = None,
                        out_dtype: str | None = None,
                        quant: str | None = None,
                        group_list: Sequence[int] | None = None,
                        group_list_type: int = 1,
                        routed_rows: int | None = None,
                        group_size: int | None = None,
                        zero_points: bool = False,
                        scale_bytes: int | None = None,
                        k_dim: int | None = None,
                        n_dim: int | None = None,
                        steady_l2_warm: bool = False,
                        hardware: HardwareProfile) -> OpWork:
    """MoE Up/Gate or Down grouped GEMM with certified minimum active experts."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    mi_i = strict_int(inter, "inter", minimum=1)
    e_total = strict_int(experts_total, "experts_total", minimum=1)
    e_local = strict_int(experts_local, "experts_local", minimum=1)
    k_i = strict_int(topk, "topk", minimum=1)
    ep_i = strict_int(ep, "ep", minimum=1)
    quant_key = validate_quant_mode(quant)
    validate_dtypes(dtype, weight_dtype, out_dtype,
                    names=("dtype", "weight_dtype", "out_dtype"))
    inferred = _inferred_quant(dtype, weight_dtype)
    if quant_key is None:
        quant_key = inferred
    elif weight_dtype is not None and inferred is not None \
            and inferred != quant_key:
        raise FormulaValidationError(
            f"quant_mode {quant_key!r} contradicts weight_dtype "
            f"{weight_dtype!r}")
    _check_quant_storage(quant_key, dtype, weight_dtype)
    if e_local > e_total:
        raise FormulaValidationError("experts_local cannot exceed experts_total")
    if k_i > e_total:
        raise FormulaValidationError(
            f"topk={k_i} cannot exceed experts_total={e_total}")
    if mode not in {"up_gate", "down", "generic"}:
        raise FormulaValidationError("mode must be up_gate, down or generic")
    if k_dim is None or n_dim is None:
        if mode == "generic":
            raise FormulaValidationError("generic mode requires explicit K and N")
        k_dim = h_i if mode == "up_gate" else mi_i
        n_dim = 2 * mi_i if mode == "up_gate" else h_i
    k_dim = strict_int(k_dim, "K", minimum=1)
    n_dim = strict_int(n_dim, "N", minimum=1)
    explicit_rows = (None if routed_rows is None else
                     strict_int(routed_rows, "routed_rows", minimum=0))
    # Every destination rank can receive the global source population of all
    # EP ranks: G = tokens * ep.  A per-rank ceiling would reject legal skewed
    # arrivals (each source rank contributes its own top-k routes).
    source_tokens = t_i * ep_i
    local_arrival_floor = source_tokens * max(0, k_i - (e_total - e_local))
    local_arrival_ceiling = source_tokens * min(k_i, e_local)
    if ep_i == 1 and group_list is None and explicit_rows is None \
            and e_local != e_total:
        raise FormulaValidationError(
            "ep=1 requires experts_local == experts_total without a subset list")
    counts: tuple[int, ...] | None = None
    if group_list is not None:
        counts = decode_group_list(group_list, group_list_type, e_local, None)
        if explicit_rows is not None and sum(counts) != explicit_rows:
            raise FormulaValidationError(
                f"group_list sums to {sum(counts)} but routed_rows is "
                f"{explicit_rows}")
        for index, value in enumerate(counts):
            if value > source_tokens:
                raise FormulaValidationError(
                    f"group_list[{index}]={value} exceeds the {source_tokens} "
                    "global source tokens that can select distinct experts")
        if not local_arrival_floor <= sum(counts) <= local_arrival_ceiling:
            raise FormulaValidationError(
                f"group_list sums to {sum(counts)} outside the feasible local "
                f"arrival interval [{local_arrival_floor}, "
                f"{local_arrival_ceiling}]")
        floor_rows = sum(counts)
        min_active = sum(1 for value in counts if value > 0)
        expected_rows = floor_rows
        expected_active = float(min_active)
    elif explicit_rows is not None:
        if not local_arrival_floor <= explicit_rows <= local_arrival_ceiling:
            raise FormulaValidationError(
                f"routed_rows={explicit_rows} outside the feasible local "
                f"arrival interval [{local_arrival_floor}, "
                f"{local_arrival_ceiling}]")
        floor_rows = explicit_rows
        expected_rows = explicit_rows
        min_active = (0 if floor_rows == 0 else
                      ceil_div_int(floor_rows, source_tokens))
        expected_active = _expected_active(e_local, k_i, e_total, source_tokens,
                                           floor_rows)
        expected_active = max(float(min_active), expected_active)
    else:
        remote = max(0, e_total - e_local)
        floor_rows = local_arrival_floor
        min_active = (0 if floor_rows == 0 else
                      ceil_div_int(floor_rows, source_tokens))
        expected_rows = int(round(source_tokens * k_i
                                  * (float(e_local) / float(e_total))))
        expected_active = _expected_active(e_local, k_i, e_total, source_tokens,
                                           expected_rows)
        expected_active = max(float(min_active), expected_active)
    expected_active_i = int(round(expected_active))
    granule_active = max(expected_active_i, min_active)
    if quant_key:
        spec = _quant_spec(quant_key, out_dtype, group_size, k_dim)
        ibpe, wbpe, obpe = spec.act_bytes, spec.weight_bytes, spec.out_bytes
        tier = spec.tier
    else:
        spec = QuantSpec(0, 0, 0, cube_tier(dtype), 0, "none", None, 0)
        ibpe = dtype_bytes(dtype)
        wbpe = dtype_bytes(weight_dtype) if weight_dtype else ibpe
        obpe = dtype_bytes(out_dtype) if out_dtype else ibpe
        tier = spec.tier
    uniform_rows = (ceil_div_int(expected_rows, expected_active_i)
                    if expected_active_i else 0)
    granule = hardware.granule_for("FRACTAL_NZ")
    al = Algebra()
    ROWS = al.sym("floor_routed_rows", floor_rows)
    EROWS = al.sym("expected_routed_rows", expected_rows)
    KL = al.sym("K", k_dim)
    NL = al.sym("N", n_dim)
    MIN_ACTIVE = al.sym("min_active_experts", min_active)
    EXPECTED = al.sym("expected_active_experts", expected_active_i)
    GRANULE_ACTIVE = al.sym("granule_active_experts", granule_active)
    IB = al.real("dtype_bytes", ibpe)
    WB = al.real("weight_bytes", wbpe)
    OB = al.real("out_bytes", obpe)
    account = ByteAccount(granule)
    # Compulsory bytes use the certified floor; aligned traffic uses expectation.
    account.read_buffer(ROWS * KL, ibpe, rows=EROWS, width=KL)
    account.read_buffer(MIN_ACTIVE * KL * NL, wbpe,
                        rows=GRANULE_ACTIVE * KL, width=NL)
    if quant_key:
        if spec.activation_scale_bytes:
            account.read_buffer(ROWS * spec.activation_scale_bytes, 1.0)
            account.granule_read = (account.granule_read
                                    + max_expr(EROWS - ROWS, 0)
                                    * spec.activation_scale_bytes)
        if scale_bytes is None:
            scale_part, _ = _weight_scale_expr(al, KL, NL, spec, zero_points)
            account.read_buffer(MIN_ACTIVE * scale_part, 1.0)
            account.granule_read = (account.granule_read
                                    + max_expr(GRANULE_ACTIVE - MIN_ACTIVE, 0)
                                    * scale_part)
        else:
            declared_scale = strict_int(scale_bytes, "scale_bytes", minimum=0)
            account.read_buffer(declared_scale, 1.0)
    account.write_buffer(ROWS * NL, obpe, rows=EROWS, width=NL)
    macs = ROWS * KL * NL
    if counts is not None:
        # Exact per-expert reference tiling: each active expert pads its own
        # row count to the M tile; averaging the histogram is not a reference.
        exact_tiles = sum(align_up_int(value, 16) for value in counts)
        tiled = Const(2 * exact_tiles * align_up_int(k_dim, 16)
                      * align_up_int(n_dim, 16))
    else:
        tiled = (2 * EXPECTED * align_up(max(1, uniform_rows), 16)
                 * align_up(KL, 16) * align_up(NL, 16))
    if quant_key:
        vector = _quant_vector_ops(spec, quant_key, rows=ROWS, k_sym=KL,
                                   n_sym=NL, weight_batches=MIN_ACTIVE,
                                   bias=False)
        vector_ref = _quant_vector_ops(spec, quant_key, rows=EROWS, k_sym=KL,
                                       n_sym=NL, weight_batches=GRANULE_ACTIVE,
                                       bias=False)
    else:
        vector = ZERO
        vector_ref = ZERO
    n_tiles = max(1, max(1, expected_active_i) * ceil_div_int(n_dim, 256))
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    hot = 0
    if steady_l2_warm:
        budget = hardware.l2_hot_budget()
        for size in (floor_rows * k_dim * ibpe, floor_rows * n_dim * obpe,
                     granule_active * k_dim * n_dim * wbpe):
            if 0 < int(size) <= budget:
                hot += int(size)
                budget -= int(size)
    if counts is not None:
        al.list("group_counts", counts)
    bindings = al.merged()
    bindings.update({
        "tokens": t_i, "hidden": h_i, "inter": mi_i,
        "experts_total": e_total, "experts_local": e_local, "topk": k_i,
        "ep": ep_i, "rows": floor_rows, "floor_routed_rows": floor_rows,
        "expected_routed_rows": expected_rows,
        "E_active": min_active, "min_active_experts": min_active,
        "E_active_expected": round(expected_active, 3),
        "expected_active_experts": expected_active_i,
        "uniform_rows_per_expert": uniform_rows,
        "group_list_exact": counts is not None,
        "weight_bytes_per_expert": int(tensor_bytes((k_dim, n_dim), wbpe)),
        "n_tiles": n_tiles, "n_waves": waves, "wave_efficiency": wave_eff,
        "compute_tier": tier, "layout": "FRACTAL_NZ", "dtype": str(dtype),
        "weight_dtype": str(weight_dtype or dtype),
        "mode": mode, "K": k_dim, "N": n_dim,
    })
    notes = ["certified floor uses floor_routed_rows/min_active_experts",
             ("exact group_list histogram" if counts is not None
              else "reference uses expected_routed_rows/expected_active_experts")]
    if counts is None and ep_i > 1 and explicit_rows is None:
        notes.append("EP>1 without local arrivals: certified floor is zero")
    return OpWork(
        family="grouped_matmul", variant=mode,
        cube_macs=macs, cube_flops_tiled=tiled,
        vector_ops=vector, vector_ref_ops=vector_ref,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=n_tiles,
        n_cores=hardware.cores, coalesce_efficiency=1.0,
        m_tiled=align_up_int(floor_rows, 16), n_tiled=align_up_int(n_dim, 16),
        k_tiled=align_up_int(k_dim, 16), tile_m=16, tile_n=256, tile_k=128,
        cube_tier=tier, notes=tuple(notes), bindings=bindings,
    )


def _expected_active(e_local: int, topk: int, e_total: int,
                     source_tokens: int, rows: int) -> float:
    """Balanced statistical distinct-active-expert expectation."""
    if rows <= 0:
        return 0.0
    probability = min(1.0, float(topk) / float(e_total))
    value = e_local * (1.0 - (1.0 - probability) ** source_tokens)
    return min(float(e_local), round(value, 6))


def embedding_work(*, tokens: int, hidden: int, vocab: int, tp: int = 1,
                   dtype: str = "bf16", steady_l2_warm: bool = False,
                   hardware: HardwareProfile) -> OpWork:
    """Vocabulary-parallel embedding gather (only requested rows are charged)."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    v_i = strict_int(vocab, "vocab", minimum=1)
    tp_i = strict_int(tp, "tp", minimum=1)
    bpe_i = dtype_bytes(dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    H = al.sym("hidden", h_i)
    V = al.sym("vocab", v_i)
    TP = al.sym("tp", tp_i)
    BPE = al.real("dtype_bytes", bpe_i)
    account = ByteAccount(granule)
    account.read_buffer(T * H * BPE, 1.0, rows=T, width=row_bytes_expr(H, bpe_i))
    account.read_buffer(T * 4, 1.0)
    account.write_buffer(T * H * BPE, 1.0, rows=T, width=row_bytes_expr(H, bpe_i))
    hot = 0
    if steady_l2_warm:
        size = int(tensor_bytes((t_i, h_i), bpe_i) + tensor_bytes((t_i,), 4))
        if 0 < size <= hardware.l2_hot_budget():
            hot = size
    n_tiles = max(1, t_i)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family="embedding", variant="gather",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=2 * T * H,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=n_tiles,
        n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(h_i, bpe_i), granule),
        tile_m=1, tile_n=16, tile_k=1, structure="vector_only",
        notes=("only requested embedding rows are compulsory",),
        bindings={**al.merged(), "n_tiles": n_tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


def lm_head_work(*, tokens: int, hidden: int, vocab: int, tp: int = 1,
                 dtype: str = "bf16", logits_dtype: str = "fp32",
                 include_sampler: bool = False,
                 steady_l2_warm: bool = False,
                 hardware: HardwareProfile) -> OpWork:
    """Logits projection; sampler work is charged only when requested."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    v_i = strict_int(vocab, "vocab", minimum=1)
    tp_i = strict_int(tp, "tp", minimum=1)
    sampler = strict_bool(include_sampler, "include_sampler")
    v_local = ceil_div_int(v_i, tp_i)
    bpe_i = dtype_bytes(dtype)
    logit_bpe = dtype_bytes(logits_dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    H = al.sym("hidden", h_i)
    VL = al.sym("vocab_local", v_local)
    BPE = al.real("dtype_bytes", bpe_i)
    LB = al.real("logit_bytes", logit_bpe)
    account = ByteAccount(granule)
    account.read_buffer(T * H * BPE, 1.0, rows=T, width=row_bytes_expr(H, bpe_i))
    account.read_buffer(H * VL * BPE, 1.0, rows=H, width=row_bytes_expr(VL, bpe_i))
    account.write_buffer(T * VL * LB, 1.0, rows=T, width=row_bytes_expr(VL, logit_bpe))
    macs = T * VL * H
    tiled = 2 * align_up(T, 16) * align_up(VL, 16) * align_up(H, 16)
    vector = 5 * T * VL if sampler else ZERO
    geom = tile_geometry(t_i, v_local, h_i)
    n_tiles = max(1, geom["n_tiles"])
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    hot = 0
    if steady_l2_warm:
        budget = hardware.l2_hot_budget()
        for size in (t_i * h_i * bpe_i, t_i * v_local * logit_bpe,
                     h_i * v_local * bpe_i):
            if 0 < int(size) <= budget:
                hot += int(size)
                budget -= int(size)
    return OpWork(
        family="lm_head", variant="projection",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=n_tiles,
        n_cores=hardware.cores, coalesce_efficiency=1.0,
        m_tiled=geom["m_tiled"], n_tiled=geom["n_tiled"],
        k_tiled=geom["k_tiled"], tile_m=geom["tile_m"], tile_n=geom["tile_n"],
        tile_k=geom["tile_k"], cube_tier=cube_tier(dtype),
        notes=("pure projection: include_sampler=False charges zero sampler ops"
               if not sampler else
               "include_sampler=True adds 5*T*V_local softmax/top-p ops"),
        bindings={**al.merged(), "tp": tp_i, "vocab": v_i,
                  "include_sampler": sampler, "n_tiles": n_tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


def argmax_work(*, tokens: int, vocab: int, logits_dtype: str = "fp32",
                dtype: str = "bf16", hardware: HardwareProfile) -> OpWork:
    """Standalone ArgMax over ``[T, V]`` logits (no projection)."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    v_i = strict_int(vocab, "vocab", minimum=1)
    logit_bpe = dtype_bytes(logits_dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    V = al.sym("vocab", v_i)
    LB = al.real("logit_bytes", logit_bpe)
    account = ByteAccount(granule)
    account.read_buffer(T * V * LB, 1.0, rows=T, width=row_bytes_expr(V, logit_bpe))
    account.write_buffer(T * 4, 1.0)
    n_tiles = max(1, t_i)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family="argmax", variant="reduce_max",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=T * V,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(v_i, logit_bpe), granule),
        tile_m=1, tile_n=128, tile_k=1, structure="sampler",
        notes=("standalone sampler: reads [T,V] logits, writes T token ids",),
        bindings={**al.merged(), "n_tiles": n_tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


def top_p_sample_work(*, tokens: int, vocab: int, logits_dtype: str = "fp32",
                      dtype: str = "bf16",
                      hardware: HardwareProfile) -> OpWork:
    """Standalone TopPSample: softmax + sort/prefix-scan over ``[T, V]``."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    v_i = strict_int(vocab, "vocab", minimum=1)
    logit_bpe = dtype_bytes(logits_dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    V = al.sym("vocab", v_i)
    LB = al.real("logit_bytes", logit_bpe)
    account = ByteAccount(granule)
    account.read_buffer(T * V * LB, 1.0, rows=T, width=row_bytes_expr(V, logit_bpe))
    account.write_buffer(T * 4, 1.0)
    vector = 5 * T * V + T * V * log2_ceil(V)
    n_tiles = max(1, t_i)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family="top_p_sample", variant="softmax_top_p",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(v_i, logit_bpe), granule),
        tile_m=1, tile_n=128, tile_k=1, structure="sampler",
        notes=("standalone sampler: softmax + top-p prefix scan, no GEMM",),
        bindings={**al.merged(), "n_tiles": n_tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


__all__ = [
    "argmax_work", "decode_group_list", "embedding_work", "grouped_matmul_work",
    "lm_head_work", "matmul_work", "top_p_sample_work",
]
