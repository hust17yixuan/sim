"""Matrix, memory, indexer and elementwise piecewise family models.

[Role]
  Derive dense/quantised GEMM, KV access, norm/activation/copy and embedding
  resource demands from declared structure and physical tiles.  MoE grouped
  GEMM, collective and expert transport live in the sibling
  ``families_moe_collective`` module and are re-exported here for the registry.
  The refined indexer/compression programs live in
  ``families_recurrent_indexer``.

[Invariants]
  * Cube work uses 16/32-aligned fractal extents and a capacity-screened
    ``(M_t, N_t, K_t)`` step.  Short M charges ``align(M, 16)`` rows.  HBM
    weights move once per N-tile/split; MTE1 may replay the resident panel.
  * One output tile is serial load/compute/store; many tiles use
    ``fill + (q-1) * max_pipe``.  MTE2 bytes round to the transaction granule.
  * Elementwise, norm, quant and copy pay at least one AIV repeat per core
    per token wave.  A dynamic quant on the row axis launches
    min(AIV, rows) blocks from the machine vector owners.  A row that
    passes the host full-load check is one GM load; the scale and cast
    reread that UB line as vector work.  A large-shape split adds a
    second GM load of each full chunk only.  UB rows still round to
    32 bytes.  No model name, shape-keyed lookup or measurement
    discount selects a tile or a latency.

[Timeouts & Locks]
  Pure bounded expression construction.  No I/O, waits or locks.

[Side-Effect Policy]
  Returns immutable ``FamilyTemplate`` records; no global mutation.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from mindie_sim.prediction.dynamic_quant_schedule import (
    OCCUPANCY, ROW_SOURCE, VECTOR_SPLIT, host_source_copies,
    quant_program_result, quant_row_demand,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import SemanticAttr
from mindie_sim.schema.tensors import TensorLayout
from .compiler import (
    CompiledOperator, FamilyTemplate, SegmentAxis, _demand, _frac, _out_name,
    _zero,
)
from .expressions import (
    Add, CeilDiv, Const, Div, Expr, FloorDiv, Max, Min, Mul, Sub, Var, as_fraction,
)
from .guards import SplitKGuard, StepGuard
from .families_bridge import (
    FamilyDeclaration, TypedOperand, register_family_operands,
)
from .families_moe_collective import (
    DEFAULT_L0A_BYTES, DEFAULT_L0B_BYTES, DEFAULT_L0C_BYTES,
    DEFAULT_L1_BYTES, DEFAULT_MTE2_GRANULE, FRACTAL, UB_ALIGN,
    _compile_collective, _compile_grouped_gemm, _compile_router,
    _compile_transport, _counts_for, _is_quant, _merge_groups, _moe_comm_active,
    _moe_comm_bindings, _owner_capacity, _round_expr, _scale, _stage, _width,
    invocation_floor, tile_plan,
)

# One vector repeat is eight 32-byte UB datablocks.  Counted as element-ops
# on the vector pipe, never as a fitted microsecond.
AIV_REPEAT_BYTES = UB_ALIGN * 8


def _theta_int(theta: Any, name: str, default: int) -> int:
    value = theta.find(name, None)
    return default if value is None else int(value)


def _ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


def _wave_schedule(tiles: int, cores: int) -> tuple[int, Fraction]:
    """Pinned-tile wave count and the active-core elapsed rate fraction.

    A right-sized wave leaves no idle active cores (fraction 1).  When the
    tile count is not a multiple of the declared cores, the last wave is a
    partial wave: the work is unchanged, but only ``tiles`` slots can sustain
    the device rate, so the cube/fixpipe clocks are capped to
    ``tiles / (cores * waves)``.  The same ratio is the single-tile case
    ``1 / cores``; idle cores never become tensor work.
    """
    cores = max(1, int(cores))
    tiles = max(0, int(tiles))
    if tiles <= 0:
        return 0, Fraction(1)
    waves = _ceil_div(tiles, cores)
    return waves, Fraction(tiles, cores * waves)


def _theta_frac(theta: Any, name: str, default: Any) -> Any:
    value = theta.find(name, None)
    return as_fraction(default) if value is None else as_fraction(value)


def _row_bytes(value: Any) -> int:
    exact = as_fraction(value)
    return int(math.ceil(exact / UB_ALIGN)) * UB_ALIGN


def _aiv_quantum(elem_bytes: Any) -> int:
    width = as_fraction(elem_bytes)
    if width <= 0:
        return AIV_REPEAT_BYTES
    return max(1, int(math.ceil(AIV_REPEAT_BYTES / width)))


def _vector_owners(ctx: CompiledOperator, theta: Any, cores: int) -> int:
    """AIV block count from the machine, not the AIC policy cap."""
    units = dict(getattr(ctx.supply, "resource_units", ()) or ())
    for name in ("vector", "mte2_aiv"):
        value = int(units.get(name, 0) or 0)
        if value > 0:
            return value
    return max(1, int(cores))


def _machine_ub(ctx: CompiledOperator, theta: Any) -> int:
    """Per-core UB bytes.  An operator override wins; else the device record."""
    explicit = theta.find("ub_bytes", None)
    if explicit not in (None, ""):
        return int(explicit)
    raw = dict(getattr(ctx.supply, "invocations", ()) or ()).get("ub_bytes", 0)
    return int(raw or 0)


def _aiv_floor(rows: Expr, tile: int, cores: int, elem_bytes: Any) -> Expr:
    waves = CeilDiv(rows, Const(max(1, int(tile))))
    c = min(max(1, int(cores)), 32)
    return Mul((waves, Const(c * _aiv_quantum(elem_bytes))))


_MX_FP8 = {"mxfp8", "mx_fp8", "mxfp8_e4m3"}
_MX_FP4 = {"mxfp4", "mx_fp4", "mxfp4_e2m1"}


def _payload_width(fmt: Any) -> float:
    """Packed element width.  Group scales stay on their own tensor."""
    text = str(fmt or "").strip().lower().replace("-", "_")
    if text in _MX_FP8:
        return 1.0
    if text in _MX_FP4:
        return 0.5
    return float(_width(fmt))


def payload_width(fmt: Any) -> float:
    """Public payload width used by the typed projection checks.

    MX payloads keep their block scale as a separate tensor, matching the
    family work formulas; plain formats use the shared dtype width table.
    """
    return _payload_width(fmt)


def _quant_output_width(theta: Any) -> Any:
    out_format = str(theta.find("output_format", "") or "").strip()
    if out_format:
        return _payload_width(out_format)
    raw = theta.find("quant_bytes", None)
    return as_fraction(1 if raw is None else raw)


def _dynamic_quant_vector(rows: Expr, cols: int) -> Expr:
    """Group reduction, scale encode, rescale, cast and pack.

    ``round(H*59/8)`` is the per-row arithmetic inventory.  The 1024-term is
    one helper constant per 256-wide row tile, never a rate-selected floor.
    """
    width = max(1, int(cols))
    arithmetic = round(width * 59.0 / 8.0)
    tiles = math.ceil(width / 256)
    return Add((
        Mul((rows, Const(arithmetic))),
        Mul((Max((rows, Const(1))), Const(1024 * tiles))),
    ))


def _transpose_view(theta: Any) -> dict | None:
    """The declared transposed-batch view, if this operator carries one.

    ``npu_transpose_batchmatmul`` is a real launch even when the source-level
    operator is named ``v_expand``; the metadata view, not the display name,
    is the authoritative physical identity.
    """
    raw = theta.find("native_tensor_view", None)
    if isinstance(raw, dict):
        view = dict(raw)
    elif isinstance(raw, (list, tuple)):
        try:
            view = dict(raw)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return view if str(view.get("kind") or "") == "transpose_batchmatmul" else None


def _gemm_mode(name: str, theta: Any) -> str:
    text = name.lower().replace("_", "").replace("-", "")
    weight_format = str(theta.find("weight_format", "bf16") or "bf16").lower()
    if (_transpose_view(theta) is not None or "transpose" in text
            or bool(int(theta.find("transpose_b", 0) or 0))):
        return "transpose_batch_matmul"
    # One authoritative method from the resolved quant declaration.  Do not
    # re-infer MX from scale_group/weight_scale/format heuristics here.
    method = str(theta.find("linear_method", "") or "").lower()
    if method == "mx_fp8":
        return "quant_batch_matmul_v3_mxfp8"
    if method in {"fp8", "w8a8", "w4a8"}:
        return "quant_batch_matmul_v3"
    if method == "mx_fp4":
        # Dense MXFP4 has no separate measured recipe yet; keep the generic
        # quant row explicitly instead of labeling it MXFP8.
        return "quant_batch_matmul_v3"
    # Legacy callers that declare the MX format name explicitly.
    if weight_format in {"mxfp8", "mx_fp8", "mxfp8_e4m3"}:
        return "quant_batch_matmul_v3_mxfp8"
    if "weightquant" in text:
        return "weight_quant_batch_matmul"
    if "quantbatch" in text or weight_format in {
            "w8a8", "w4a8", "int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2"}:
        return "quant_batch_matmul_v3"
    return "matmul_v2"


def _require_mx_scale_producer(operator: Any, siblings: Any, scale_row: float,
                               producer_name: str = "") -> Any:
    """Require the separate E8M0 producer port for an MX consumer.

    The GEMM consumes a payload plus the producer's activation-scale tensor.
    A missing, later, or mismatched producer means the graph is incomplete and
    must not issue an ordinary QuantBatchMatmulV3.
    """
    expected = str(producer_name or f"{operator.name}_QuantV2")
    producer = None
    for sibling in siblings:
        if sibling.name == expected:
            producer = sibling
            break
    if producer is None:
        raise ModelingError("mx_gemm_requires_dynamic_mx_quant_scale_producer:" + operator.name)
    if (producer.family != "quantize"
            or str(producer.theta.find("native_kind", "")) != "DynamicMxQuant"
            or str(producer.theta.find("activation_scale_format", "")) != "fp8_e8m0"
            or float(producer.theta.find("activation_scale_bytes", 0.0) or 0.0) != float(scale_row)):
        raise ModelingError("mx_gemm_activation_scale_producer_mismatch:" + operator.name)
    return producer


def _compile_gemm(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    k, n = int(theta.require("K")), int(theta.require("N"))
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    nt = _theta_int(theta, "nt", ctx.policy.nt)
    kt = _theta_int(theta, "kt", ctx.policy.kt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    act_bytes = _theta_frac(theta, "act_bytes", 2.0)
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    weight_format = str(theta.find("weight_format", "bf16") or "bf16")
    weight_bytes = _payload_width(weight_format)
    out_format = str(theta.find("output_format", act_format) or act_format)
    out_bytes = _payload_width(out_format)
    weight_total = _theta_frac(theta, "weight_total", n * k * weight_bytes)
    weight_payload = _theta_frac(theta, "weight_payload", weight_total)
    weight_scale = _theta_frac(theta, "weight_scale", 0.0)
    weight_zero = _theta_frac(theta, "weight_zero", 0.0)
    granule = max(1, _theta_int(theta, "mte2_granule", DEFAULT_MTE2_GRANULE))
    slots = max(1, _theta_int(theta, "pipeline_slots",
                             _theta_int(theta, "slots", ctx.policy.pipeline_slots)))
    plan = tile_plan(
        mt, n, k, mt=mt, nt=nt, kt=kt, act_bytes=float(act_bytes),
        weight_bytes=weight_bytes, out_bytes=out_bytes,
        l1_bytes=_owner_capacity(ctx, theta, "l1", DEFAULT_L1_BYTES),
        l0a_bytes=_owner_capacity(ctx, theta, "l0a", DEFAULT_L0A_BYTES),
        l0b_bytes=_owner_capacity(ctx, theta, "l0b", DEFAULT_L0B_BYTES),
        l0c_bytes=_owner_capacity(ctx, theta, "l0c", DEFAULT_L0C_BYTES),
        slots=slots, compute_slots=max(1, _theta_int(theta, "compute_slots", 2)))
    if not plan.fits:
        raise ModelingError("tile_plan_capacity_exceeded")
    mode = _gemm_mode(operator.name, theta)
    transpose_view = _transpose_view(theta)
    batch = max(1, _theta_int(theta, "batch", 1))
    if transpose_view is not None:
        # The source-level v_expand carries its physical head-batch in the
        # native view; theta.batch is the same quantity when present.
        batch = max(batch, int(transpose_view.get("batch") or 1))
    m_tile, n_tile, k_tile = plan.m_tile, plan.n_tile, plan.k_tile
    n_tiles, k_panels = plan.n_tiles, plan.k_panels
    split_policy = ctx.policy.split_policy(
        target=_theta_int(theta, "split_target", ctx.policy.split_target),
        max_splits=_theta_int(theta, "max_splits", ctx.policy.max_splits))
    selected_rows = bool(int(theta.find("selected_rows", 0) or 0))
    axes = (
        SegmentAxis("M_TILES", "gemm_tokens",
                    StepGuard("gemm_m_tiles", "gemm_tokens", m_tile),
                    "max_tokens"),
        SegmentAxis("SPLITS", "gemm_work",
                    SplitKGuard(split_policy, coordinate="gemm_work"), "max_work"),
    )

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        if selected_rows:
            # Same row rule the logits collective binds to.
            tokens = geometry.producing_lm_head_rows()
        else:
            tokens = geometry.effective_tokens
        # Split-K beyond the physical K panel count duplicates work without
        # shortening a panel.  Cap the guard coordinate on that lattice.
        panel_cap = max(1, int(as_fraction(split_policy.target) * plan.k_panels))
        m_tiles_count = _ceil_div(tokens, m_tile) if tokens > 0 else 0
        # Split-K only supplies parallelism that the MxN grid cannot already
        # fill.  When enough independent output tiles exist, the ordinary
        # kernel accumulates the full K panel locally; an external partial
        # workspace is not a physical requirement.
        base_tiles = m_tiles_count * n_tiles * batch
        work = max(1, min(tokens * n * k, panel_cap))
        # The SPLITS axis is selected from gemm_work.  When the output grid
        # already covers the cores, give the guard zero split work so it
        # selects one local-K owner; do not merely compute a dead local count.
        # Wave saturation suppresses Split-K only when there is a wave to
        # saturate: a single-owner die has no idle AIC neighbour that could
        # stream a K slice, so its MxN grid can never cover the cores and the
        # guard stays armed.  This is the declared-core occupancy rule, not a
        # per-shape knob.
        saturated = cores > 1 and base_tiles >= cores
        guard_work = 0 if saturated else work
        work = guard_work
        split_count = 1 if saturated else split_policy.splits(work)
        tiles_count = base_tiles * split_count
        waves_count, schedule_scale = _wave_schedule(tiles_count, cores)
        # Cube, vector and fixpipe keep the output-wave cap.  MTE is pinned
        # per output owner on the declared hardware units, not on split-K
        # j_tiles and not on the policy core cap.  HBM stays on the full chip.
        from mindie_sim.prediction.matrix_mapping import (
            critical_output_owner_rate_fraction, gemm_fractal_operand_bytes,
            mte_owner_unit_count, owner_tiles,
        )
        fractal_m_val = _ceil_div(tokens, FRACTAL) * FRACTAL if tokens > 0 else 0
        area_scale = schedule_scale
        if fractal_m_val > 0 and plan.n_pad > 0 and cores > 0:
            area_owners = owner_tiles(
                fractal_m_val, plan.n_pad, m_tile, n_tile, cores,
                batch * split_count)
            peak_area = max((area for _, _, _, area in area_owners), default=0)
            total_area = fractal_m_val * plan.n_pad * batch * split_count
            if peak_area > 0:
                area_scale = min(Fraction(1),
                                 Fraction(total_area, peak_area * cores))
        bound = {
            "gemm_work": work, "gemm_tokens": tokens,
            "GEMM_TILES": tiles_count, "GEMM_WAVES": waves_count,
            "GEMM_SCHEDULE_SCALE": schedule_scale,
            "HW_RATE_CUBE": area_scale,
            "HW_RATE_VECTOR": area_scale,
            "HW_RATE_FIXPIPE": area_scale,
            "transpose_blocks": 0,
        }
        unit_count = mte_owner_unit_count(getattr(ctx.supply, "resource_units", ()))
        if unit_count > 0 and tokens > 0:
            m_t_cnt = max(1, _ceil_div(tokens, m_tile))
            tile_rows_v = Fraction(fractal_m_val, m_t_cnt)
            n_issue_v = Fraction(plan.n_pad, max(1, n_tiles))
            k_issue_v = Fraction(plan.k_pad, max(1, k_panels))
            act_f = as_fraction(act_bytes)
            w_f = as_fraction(weight_bytes)
            meta_tot_f = as_fraction(weight_scale + weight_zero)
            meta_p = (meta_tot_f * k_issue_v * n_issue_v / max(1, k * n)
                      if meta_tot_f else Fraction(0))
            act_p = tile_rows_v * k_issue_v * act_f
            comb_p = act_p + k_issue_v * n_issue_v * w_f + meta_p
            comb_g = Fraction(math.ceil(comb_p / granule) * granule)
            act_g = Fraction(math.ceil(act_p / granule) * granule)
            charged_panels = (batch * n_tiles * k_panels
                              * (comb_g + (m_t_cnt - 1) * act_g))
            xscale_b = batch * tokens * _theta_frac(
                theta, "activation_scale_bytes", 0.0)
            partial_r = Fraction(max(0, split_count - 1) * batch * tokens * n * 4)
            owners_u = owner_tiles(tokens, n, m_tile, n_tile, unit_count, batch)
            peak_panel = Fraction(0)
            for cnt, em, en, area in owners_u:
                cnt_v = max(1, cnt)
                o_meta = (meta_tot_f * k_issue_v * en / max(1, k * n)
                          if meta_tot_f else Fraction(0))
                o_pan = (act_f * em * k_issue_v
                         + w_f * en * k_issue_v / m_t_cnt + o_meta) / cnt_v
                o_tot = (cnt_v * Fraction(math.ceil(o_pan / granule) * granule)
                         * k_panels
                         + max(0, split_count - 1) * area * 4 + xscale_b)
                if o_tot > peak_panel:
                    peak_panel = o_tot
            if peak_panel > 0:
                share = min(Fraction(1),
                            (charged_panels + xscale_b + partial_r)
                            / (peak_panel * unit_count))
            else:
                charged = gemm_fractal_operand_bytes(
                    tokens, n_tiles, plan.n_pad, plan.k_pad, act_bytes,
                    weight_bytes, fractal=FRACTAL)
                share = critical_output_owner_rate_fraction(
                    tokens, n, k, m_tile, n_tile, unit_count,
                    act_bytes=act_bytes, weight_bytes=weight_bytes,
                    charged_bytes=charged)
            rates_map = dict(ctx.supply.rates)
            mte2_boost = Fraction(1)
            if ("mte2_aic" in rates_map and "mte2_aiv" in rates_map
                    and rates_map["mte2_aic"] > rates_map["mte2_aiv"]
                    and rates_map.get("mte2") == rates_map["mte2_aiv"]
                    and rates_map["mte2"] > 0):
                mte2_boost = (as_fraction(rates_map["mte2_aic"])
                              / as_fraction(rates_map["mte2"]))
            units_map = dict(getattr(ctx.supply, "resource_units", ()) or ())
            mte3_units = int(units_map.get("mte3", 0) or unit_count)
            mte3_owners = owner_tiles(
                tokens, n, m_tile, n_tile, mte3_units, batch)
            mte3_pad_owners = (
                owner_tiles(fractal_m_val, plan.n_pad, m_tile, n_tile,
                            mte3_units, batch)
                if split_count > 1 and fractal_m_val > 0 and plan.n_pad > 0
                else mte3_owners
            )
            out_f = as_fraction(out_bytes)
            part_mult = split_count * 4 if split_count > 1 else 0
            peak_mte3 = Fraction(0)
            for idx, (_cnt, _em, _en, area) in enumerate(mte3_owners):
                pad_area = (mte3_pad_owners[idx][3]
                            if idx < len(mte3_pad_owners) else area)
                o_mte3 = area * out_f + part_mult * pad_area
                if o_mte3 > peak_mte3:
                    peak_mte3 = o_mte3
            total_mte3 = (tokens * n * batch * out_f
                          + batch * part_mult * fractal_m_val * plan.n_pad)
            mte3_share = (min(Fraction(1),
                              total_mte3 / (peak_mte3 * mte3_units))
                          if peak_mte3 > 0 and mte3_units > 0 else share)
            bound["HW_RATE_MTE2"] = share * mte2_boost
            bound["HW_RATE_MTE1"] = share
            bound["HW_RATE_MTE3"] = mte3_share
        if mode == "transpose_batch_matmul":
            # One launch block per (head, M-tile, N-tile) pair.  A decode
            # batch of 1..32 heads on a 20/32-owner die leaves most MTE2
            # owners idle; an active block cannot borrow an idle owner's
            # share, so the transpose panel moves at blocks/units of the die
            # rate.  This is the partial-wave occupancy, not a fitted factor.
            blocks = (batch * max(1, _ceil_div(tokens, m_tile)) * n_tiles
                      if tokens > 0 else 0)
            bound["transpose_blocks"] = blocks
            if blocks > 0:
                units = max(1, unit_count)
                share = Fraction(min(blocks, units), units)
                bound["HW_RATE_MTE2"] = share
                bound["HW_RATE_MTE1"] = share
                bound["HW_RATE_MTE3"] = share
                bound["HW_RATE_FIXPIPE"] = share
            # Kernel invocation prologue (the declared cube completion floor)
            # plus the empty-AIC block drain is serial with the transpose
            # panel work; it is not a lower bound that hides that work.
            inv = dict(ctx.supply.invocations)
            prologue = (Fraction(str(inv.get("cube", 0.0) or 0.0))
                        + Fraction(str(inv.get("empty_aic", 0.0) or 0.0)))
            if prologue > 0:
                bound["HW_DEPENDENCY_MS"] = prologue
        return bound

    m_tiles, splits = Var("M_TILES"), Var("SPLITS")
    tokens = Var("gemm_tokens")
    m_pad = Mul((m_tiles, Const(m_tile)))
    fractal_m = Mul((CeilDiv(tokens, Const(FRACTAL)), Const(FRACTAL)))
    # Tail tiles issue only their aligned remainder.  Averages conserve
    # align(M), n_pad and k_pad without charging a full macro-tile per tail.
    tile_rows = Div(fractal_m, m_tiles)
    n_issue = Div(Const(plan.n_pad), Const(n_tiles))
    k_issue = Div(Const(plan.k_pad), Const(k_panels))
    k_split = Div(Const(k_panels), splits)
    k_chunk = Mul((k_split, k_issue))
    tiles = Mul((m_tiles, Const(n_tiles), splits, Const(batch)))
    waves = CeilDiv(tiles, Const(cores))
    gate = Min((tiles, Const(1)))
    active = Min((Const(cores), tiles))
    rest_tiles = Sub(tiles, active)
    internal = Const(1 if slots >= 2 else 0)
    metadata_total = weight_scale + weight_zero
    metadata_panel = (Div(Mul((Const(metadata_total), k_issue, n_issue)),
                          Const(max(1, k * n)))
                      if metadata_total else Const(0))
    act_scale_row = _theta_frac(theta, "activation_scale_bytes", 0.0)
    if act_scale_row and str(theta.find("category", "")) == "attn.o":
        _require_mx_scale_producer(operator, ctx.siblings, act_scale_row,
                                   str(theta.find("scale_producer", "") or ""))
    act_panel = Mul((tile_rows, k_issue, Const(act_bytes)))
    weight_panel = Add((
        Mul((k_issue, n_issue, Const(weight_bytes))),
        metadata_panel,
    ))
    combined_panel = Add((act_panel, weight_panel))
    combined_granule = _round_expr(combined_panel, granule)
    act_granule = _round_expr(act_panel, granule)
    extra_tiles = Sub(m_tiles, Const(1))
    # First scheduled M tile of an N-tile/split pays the weight granule.
    # Later M tiles pay the activation granule only.  m_tiles == 1 is the
    # combined granule, so single-tile rounding stays exact.
    mte2_panel = Div(Add((combined_granule, Mul((extra_tiles, act_granule)))),
                     m_tiles)
    hbm_panel = Div(Add((combined_panel, Mul((extra_tiles, act_panel)))),
                    m_tiles)
    mte1_panel = combined_granule if slots == 1 else mte2_panel
    act_chunk = Mul((tile_rows, k_chunk, Const(act_bytes)))
    weight_chunk = Add((
        Mul((k_chunk, n_issue, Const(weight_bytes))),
        Mul((metadata_panel, k_split)),
    ))
    load_tile = Div(Add((Add((act_chunk, weight_chunk)),
                         Mul((extra_tiles, act_chunk)))), m_tiles)
    load_granule = Mul((k_split, mte2_panel))
    panel_compute = Mul((Const(2), tile_rows, n_issue, k_issue))
    compute_tile = Mul((Const(2), tile_rows, n_issue, k_chunk))
    partial_tile = Mul((tile_rows, n_issue, Const(4),
                        Min((Sub(splits, Const(1)), Const(1)))))
    fixpipe = Mul((tile_rows, n_issue, Const(4)))
    act_quant = Const(int(_is_quant(act_format)))
    weight_quant = Const(int(_is_quant(weight_format)))
    # QuantBatchMatmulV3 on r64 is an AI_CORE-only quantized matrix: its
    # trace has AIV time 0 and no separate AIV epilogue.  The quant producer,
    # its scales and the BF16 store are already separate rows; charging a
    # generic AIV quant/dequant pass here double-counts and serializes it.
    quant_aic_only = mode in {"quant_batch_matmul_v3",
                              "quant_batch_matmul_v3_mxfp8"}
    panel_vector = (Const(0) if quant_aic_only else Add((
        Mul((tile_rows, k_issue, act_quant)),
        Mul((k_issue, n_issue, weight_quant)),
    )))
    # A transposed batch product reads the source weight panel and re-reads
    # the staged transposed panel; the MTE2 side therefore moves two copies
    # while HBM and MTE3 each move one.  ``transpose_bytes`` is the same
    # logical two-copy traffic for the declaration ledger.
    transpose_bytes = (batch * n * k * weight_bytes * 2
                       if mode == "transpose_batch_matmul" else 0.0)
    transpose_mte2_bytes = transpose_bytes
    transpose_store_bytes = (batch * n * k * weight_bytes
                             if mode == "transpose_batch_matmul" else 0.0)
    # Double buffer: one panel fills, the rest overlap, one panel drains.
    # Never replay every panel just because a later output wave exists.
    if slots >= 2:
        fill_panels = Min((Const(1), k_split))
        steady_panels = Max((Const(0), Sub(k_split, Const(1))))
    else:
        fill_panels = k_split
        steady_panels = Const(0)
    col_groups = Min((tiles, Mul((Const(n_tiles * batch), splits))))
    first_cols = Min((active, col_groups))
    first_extra = Sub(active, first_cols)
    rest_cols = Sub(col_groups, first_cols)
    rest_extra = Sub(rest_tiles, rest_cols)
    wave0_mte2 = Add((Mul((first_cols, combined_granule)),
                      Mul((first_extra, act_granule))))
    wave0_hbm = Add((Mul((first_cols, combined_panel)),
                     Mul((first_extra, act_panel))))
    wave0_mte1 = Mul((active, combined_granule)) if slots == 1 else wave0_mte2
    rest_mte2 = Add((Mul((rest_cols, combined_granule)),
                     Mul((rest_extra, act_granule))))
    rest_hbm_panel = Add((Mul((rest_cols, combined_panel)),
                          Mul((rest_extra, act_panel))))
    rest_mte1 = Mul((rest_tiles, combined_granule)) if slots == 1 else rest_mte2
    wave0_mem = (("mte2", wave0_mte2), ("mte1", wave0_mte1))
    wave0_hbm_g = (("hbm", wave0_hbm),)
    rest_mem_g = (("mte2", rest_mte2), ("mte1", rest_mte1))
    rest_hbm_g = (("hbm", rest_hbm_panel),)
    compute = (("cube", panel_compute), ("vector", panel_vector))
    fixpipe_group = (("fixpipe", fixpipe),)
    store = (("mte3", partial_tile), ("hbm", partial_tile))
    xscale_total = (Mul((Const(batch), tokens, Const(act_scale_row)))
                    if act_scale_row else Const(0))
    xscale_group = ((("mte2", xscale_total), ("hbm", xscale_total))
                    if act_scale_row else ())
    load_stage = _stage(ctx, _merge_groups(
        _scale(wave0_mem, Mul((gate, fill_panels))),
        _scale(wave0_hbm_g, Mul((gate, fill_panels))),
        xscale_group))
    compute_stage = _stage(ctx, _merge_groups(
        _scale(compute, Mul((gate, active, fill_panels))),
        _scale(fixpipe_group, Mul((gate, active)))))
    first_store = _stage(ctx, _scale(store, Mul((gate, active))))
    rest_mem = _scale(rest_mem_g, Mul((gate, k_split)))
    rest_hbm = _scale(rest_hbm_g, Mul((gate, k_split)))
    rest_compute = _scale(compute, Mul((gate, rest_tiles, k_split)))
    rest_fixpipe = _scale(fixpipe_group, Mul((gate, rest_tiles)))
    rest_store = _scale(store, Mul((gate, rest_tiles)))
    internal_factor = Mul((gate, internal, steady_panels))
    internal_stage = _stage(ctx, _merge_groups(
        _scale(wave0_mem, internal_factor),
        _scale(wave0_hbm_g, internal_factor),
        _scale(compute, Mul((active, internal_factor)))))
    if slots >= 2:
        if k_panels > 1:
            # Every split accumulates all of its K panels before its partial
            # tile is published; only then does the merge consume it.
            stages = [
                load_stage, compute_stage,
                _stage(ctx, _merge_groups(rest_mem, rest_hbm, rest_compute,
                                          rest_fixpipe)),
                internal_stage,
                _stage(ctx, _scale(store, Mul((gate, Add((active,
                                                          rest_tiles)))))),
            ]
        else:
            stages = [
                load_stage, compute_stage, first_store,
                _stage(ctx, _merge_groups(rest_mem, rest_hbm, rest_compute,
                                          rest_fixpipe, rest_store)),
            ]
    else:
        stages = [
            load_stage, compute_stage, first_store,
            _stage(ctx, _merge_groups(rest_mem, rest_hbm)),
            _stage(ctx, _merge_groups(rest_compute, rest_fixpipe)),
            _stage(ctx, rest_store),
        ]
    if transpose_bytes:
        stages.insert(0, (
            _demand(ctx, "mte2", Const(transpose_mte2_bytes)),
            _demand(ctx, "hbm", Const(transpose_store_bytes)),
            _demand(ctx, "mte3", Const(transpose_store_bytes)),
        ))
    out = Mul((Const(batch), tokens, Const(n), Const(out_bytes)))
    partial = Mul((Sub(splits, Const(1)), Const(batch),
                   tokens, Const(n), Const(4)))
    partial_write = Mul((tiles, partial_tile))
    merge = (
        _demand(ctx, "mte2", partial),
        _demand(ctx, "mte3", out),
        _demand(ctx, "hbm", Add((partial, out))),
        _demand(ctx, "vector", Mul((Sub(splits, Const(1)), Const(batch),
                                    tokens, Const(n)))),
    )
    stages.append(merge)
    useful = Mul((tiles, load_tile))
    padded = Mul((tiles, load_granule))
    footprint = plan.footprint(float(act_bytes), weight_bytes, out_bytes)
    derived = (
        ("flops", Mul((Const(2 * batch), tokens, Const(n), Const(k)))),
        ("padded_flops", Mul((Const(2 * batch), m_pad, Const(plan.n_pad),
                              Const(plan.k_pad)))),
        ("fractal_flops", Mul((Const(2 * batch), fractal_m, Const(plan.n_pad),
                               Const(plan.k_pad)))),
        ("fractal_m", fractal_m), ("fractal_n", Const(plan.n_pad)),
        ("fractal_k", Const(plan.k_pad)), ("k_align", Const(plan.k_align)),
        ("tile_padded_m", m_pad), ("charged_m", tile_rows),
        ("tile_m", Const(m_tile)), ("tile_n", Const(n_tile)),
        ("tile_k", Const(k_tile)), ("k_inner", Const(plan.k_inner)),
        ("tile_n_tiles", Const(n_tiles)), ("n_tiles", Const(n_tiles)),
        ("k_panels", Const(k_panels)), ("k_split_panels", k_split),
        ("k_chunk", k_chunk), ("splits", splits),
        ("j_tiles", tiles), ("m_tiles", m_tiles), ("waves", waves),
        ("gemm_schedule_tiles", Var("GEMM_TILES")),
        ("gemm_schedule_waves", Var("GEMM_WAVES")),
        ("gemm_schedule_scale", Var("GEMM_SCHEDULE_SCALE")),
        ("active_cores", active),
        ("tail_rows", Sub(tokens, Mul((Sub(m_tiles, Const(1)),
                                       Const(m_tile))))),
        ("weight_payload", Const(weight_payload)),
        ("weight_scale", Const(weight_scale)),
        ("weight_zero", Const(weight_zero)),
        ("weight_total", Const(weight_total)),
        ("rows", tokens),
        ("activation_payload_bytes", Mul((Const(batch), tokens, Const(k),
                                           Const(act_bytes)))),
        ("output_bytes", out), ("padded_output_bytes",
                                Mul((Const(batch), m_pad, Const(plan.n_pad),
                                     Const(out_bytes)))),
        ("partial_bytes", Add((partial_write, partial))),
        ("partial_write_bytes", partial_write),
        ("partial_read_bytes", partial),
        ("transpose_bytes", Const(transpose_bytes)),
        ("transpose_mte2_bytes", Const(transpose_mte2_bytes)),
        ("transpose_store_bytes", Const(transpose_store_bytes)),
        ("transpose_blocks", Var("transpose_blocks")),
        ("tile_load_bytes", load_tile), ("tile_compute_flops", compute_tile),
        ("tile_load_granule_bytes", load_granule),
        ("tile_store_bytes", partial_tile), ("tile_fixpipe_bytes", fixpipe),
        ("mte2_granule_bytes", Const(granule)),
        ("mte2_useful_bytes", useful), ("mte2_padded_bytes", padded),
        ("mte2_efficiency", Div(useful, Max((padded, Const(1))))),
        ("l1_tile_bytes", Const(footprint["l1"])),
        ("l0a_tile_bytes", Const(footprint["l0a"])),
        ("l0b_tile_bytes", Const(footprint["l0b"])),
        ("l0c_tile_bytes", Const(footprint["l0c"])),
        ("pipeline_slots", Const(slots)), ("pipeline_q", waves),
        ("pipeline_fill_tiles", Const(1)), ("pipeline_drain_tiles", Const(0)),
        ("tile_capacity_fits", Const(int(plan.fits))), ("batch", Const(batch)),
        ("activation_scale_row_bytes", Const(act_scale_row)),
        ("activation_scale_bytes", Mul((Const(batch), tokens,
                                        Const(act_scale_row))) if act_scale_row else Const(0)),
        ("quantized", Const(int(_is_quant(weight_format)
                                 or _is_quant(act_format)))),
    )
    return FamilyTemplate(
        axes=axes, stages=tuple(stages),
        materialization=((_out_name(ctx), out),), internal=(),
        residency=Const(weight_total), derived=derived,
        regime_exprs=(), binder=binder,
        annotations=(("family", "gemm"), ("mode", mode), ("K", str(k)),
                     ("N", str(n)), ("weight_format", weight_format),
                     ("act_format", act_format)),
    )


def _gemm_typed_declaration(compiled: CompiledOperator,
                            cost: Any, context: Any = None) -> FamilyDeclaration:
    """Declare the dense GEMM operands and attrs actually used by its work.

    The three main tensors are exactly the payload/result quantities the
    family formulas charge: ``activation_payload_bytes``, ``weight_payload``
    and ``output_bytes``.  Their dtype/layout come from the same theta fields
    the compiler reads, so a declared byte width and a dtype cannot disagree
    silently.
    """
    operator = compiled.operator
    theta = operator.theta
    k, n = int(theta.require("K")), int(theta.require("N"))
    batch = max(1, _theta_int(theta, "batch", 1))
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    weight_format = str(theta.find("weight_format", "bf16") or "bf16")
    out_format = str(theta.find("output_format", act_format) or act_format)
    act_bytes = _theta_frac(theta, "act_bytes", 2.0)
    rows = int(cost.get("rows", 0) or 0)
    quant_scheme = str(theta.find("linear_method", "") or "")

    def layout(*keys: str) -> str:
        for key in keys:
            value = theta.find(key, None)
            if isinstance(value, str) and value:
                return value
        return "strided"

    operands = (
        TypedOperand(
            "activation", "in", "activation", act_format,
            layout("activation_layout", "act_layout", "input_layout"),
            (batch, "rows", k), values=(("rows", rows),),
            bytes_per_element=act_bytes,
            bytes_key="activation_payload_bytes",
            quant=quant_scheme,
            source="theta.quant:act_format,act_bytes;theta.matrix:K"),
        TypedOperand(
            "weight", "in", "weight", weight_format,
            layout("weight_layout", "b_layout"),
            (k, n), bytes_per_element=_payload_width(weight_format),
            bytes_key="weight_payload",
            quant=quant_scheme,
            source="theta.quant:weight_format;theta.matrix:K,N"),
        TypedOperand(
            "output", "out", "output", out_format,
            layout("output_layout", "out_layout"),
            (batch, "rows", n), values=(("rows", rows),),
            bytes_per_element=_payload_width(out_format),
            bytes_key="output_bytes",
            source=("theta.quant:output_format"
                    if theta.find("output_format", None) is not None
                    else "gemm:result_format_fallback_act_format")),
    )
    attrs: list[SemanticAttr] = [
        SemanticAttr("K", k, "elements", "theta.matrix"),
        SemanticAttr("N", n, "elements", "theta.matrix"),
        SemanticAttr("batch", batch, "elements",
                     "theta:batch" if theta.find("batch", None) is not None
                     else "default:batch=1"),
        SemanticAttr("mode", _gemm_mode(operator.name, theta), "",
                     "families_matrix:_gemm_mode"),
        SemanticAttr("act_format", act_format, "", "theta.quant:act_format"),
        SemanticAttr("weight_format", weight_format, "",
                     "theta.quant:weight_format"),
        SemanticAttr("output_format", out_format, "",
                     "theta.quant:output_format"
                     if theta.find("output_format", None) is not None
                     else "gemm:result_format_fallback_act_format"),
        SemanticAttr("quantized", int(_is_quant(weight_format)
                                      or _is_quant(act_format)), "",
                     "theta.quant:quantized"
                     if theta.find("quantized", None) is not None
                     else "gemm:format_quantization"),
    ]
    for name, default, unit in (
            ("mt", compiled.policy.mt, "elements"),
            ("nt", compiled.policy.nt, "elements"),
            ("kt", compiled.policy.kt, "elements"),
            ("cores", compiled.policy.cores, "units"),
            ("split_target", compiled.policy.split_target, "flops"),
            ("max_splits", compiled.policy.max_splits, "count")):
        declared = theta.find(name, None)
        attrs.append(SemanticAttr(
            name, int(declared if declared is not None else default), unit,
            f"theta.tiles:{name}" if declared is not None
            else f"policy:{name}"))
    declared_slots = theta.find("pipeline_slots", theta.find("slots", None))
    attrs.append(SemanticAttr(
        "pipeline_slots",
        int(declared_slots if declared_slots is not None
            else compiled.policy.pipeline_slots), "slots",
        "theta.tiles:pipeline_slots" if declared_slots is not None
        else "policy:pipeline_slots"))
    granule = theta.find("mte2_granule", None)
    attrs.append(SemanticAttr(
        "mte2_granule", int(granule if granule is not None
                            else DEFAULT_MTE2_GRANULE), "bytes",
        "theta.tiles:mte2_granule" if granule is not None
        else "default:mte2_granule"))
    for key, unit in (("scale_group", "elements"),
                      ("weight_scale", "bytes"), ("weight_zero", "bytes"),
                      ("weight_scale_bytes", "bytes"),
                      ("weight_scale_format", ""), ("category", ""),
                      ("linear_method", ""), ("prequant_format", ""),
                      ("selected_rows", "flag")):
        value = theta.find(key, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            attrs.append(SemanticAttr(key, value, unit, f"theta.quant:{key}"))
    notes: list[str] = []
    quantized = _is_quant(weight_format) or _is_quant(act_format)
    prequant = str(theta.find("prequant_format", "") or "")
    if (quantized and theta.find("output_format", None) is None
            and prequant and prequant != out_format):
        notes.append(
            f"result dtype falls back to act_format={act_format} while the "
            f"declared pre-quant stream is {prequant}; a selected native "
            "quant wrapper may declare its own output dtype, so the result "
            "representation remains an open T12/quant contract")
    if _theta_frac(theta, "weight_scale", 0.0) > 0:
        notes.append(
            "weight scale/zero bytes stay in weight_payload/weight_total "
            "metadata; their per-block tensor layout is not declared here")
    return FamilyDeclaration(operands, tuple(attrs), tuple(notes))


def _kv_access_typed_declaration(compiled: CompiledOperator,
                                 cost: Any, context: Any = None) -> FamilyDeclaration:
    """Declare the KV/cache row representation and the readiness attributes.

    A pure readiness read (``kv_read`` with row_bytes=0) declares no tensor
    payload: its state read and zero work are the whole contract.  A store
    declares its row payload as byte-addressed cache storage and checks the
    exact ``kv_bytes`` traffic.
    """
    operator = compiled.operator
    theta = operator.theta
    row_bytes = _theta_frac(theta, "row_bytes", 0.0)
    # ROWS is the physical access range actually bound by the family axis:
    # ``kv_positions`` for a read and ``kv_rows`` for a store.  valid_rows is
    # a distinct write/query dimension and must not replace it.
    regime = dict(cost.regime)
    rows = int(regime.get("ROWS", cost.get("valid_rows", 0) or 0))
    read = "read" in operator.name.lower()
    declared_dtype = theta.find("kv_dtype", None)
    dtype = str(declared_dtype or "uint8")
    operands: tuple[TypedOperand, ...] = ()
    derived = {str(key): int(value) for key, value in cost.derived}
    if (operator.name == "index_cache_store"
            and derived.get("index_scale_bytes", 0) > 0):
        for name, key, row_width in (
                ("index_k_cache", "index_data_bytes",
                 int(theta.find("row_bytes", 0) or 0)
                 - int(theta.find("scale_bytes", 0) or 0)),
                ("index_scale_cache", "index_scale_bytes",
                 int(theta.find("scale_bytes", 0) or 0))):
            amount = derived.get(key, 0)
            if amount > 0 and row_width > 0:
                operands += (TypedOperand(
                    name, "out", "kv_cache", "uint8", "row_major",
                    ("rows", "row_bytes"),
                    values=(("rows", rows), ("row_bytes", row_width)),
                    bytes_per_element=Fraction(1), bytes_key=key,
                    storage="device",
                    source=f"cost.derived:{key};cost.regime:ROWS",
                    layout_spec=TensorLayout.contiguous((rows, row_width))),)
    elif row_bytes > 0:
        # The cache row is byte-addressed storage; its element format stays a
        # separate attr instead of being guessed as a whole-row element type.
        shape = (rows, int(row_bytes))
        operands = (TypedOperand(
            "kv_cache_row", "in" if read else "out", "kv_cache", "uint8",
            "row_major", ("rows", "row_bytes"),
            values=(("rows", rows), ("row_bytes", int(row_bytes))),
            bytes_per_element=Fraction(1),
            bytes_key="kv_bytes",
            storage="device",
            source="theta.matrix:row_bytes;cost.regime:ROWS",
            layout_spec=TensorLayout.contiguous(shape)),)
    attrs: list[SemanticAttr] = [
        SemanticAttr("row_bytes", float(row_bytes), "bytes", "theta.matrix"),
        SemanticAttr("valid_rows_mode",
                     str(theta.find("valid_rows_mode", "dynamic")), "",
                     "theta.extras:valid_rows_mode"),
    ]
    owner = theta.find("owner_layer", None)
    if owner is not None:
        attrs.append(SemanticAttr("owner_layer", int(owner), "layer",
                                  "theta.extras:owner_layer"))
    cache_layout = theta.find("layout", theta.find("cache_layout", None))
    if isinstance(cache_layout, str) and cache_layout:
        attrs.append(SemanticAttr("cache_layout", cache_layout, "",
                                  "theta.extras:layout"))
    attrs.append(SemanticAttr(
        "kv_dtype", dtype, "",
        "theta.extras:kv_dtype" if declared_dtype is not None
        else "default:uint8"))
    return FamilyDeclaration(operands, tuple(attrs))


def _compile_kv_access(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    row_bytes = _frac(theta.require("row_bytes"))
    coordinate = "kv_positions" if operator.name == "kv_read" else "kv_rows"
    scale_bytes = _frac(theta.find("scale_bytes", 0) or 0)
    data_bytes = max(Fraction(0), row_bytes - scale_bytes)
    rows_mode = str(theta.find("valid_rows_mode", "dynamic"))
    explicit_rows = (operator.name == "index_cache_store" and rows_mode == "explicit")
    unresolved_rows = (operator.name == "index_cache_store" and rows_mode == "unresolved")
    axes = (SegmentAxis("ROWS", coordinate,
                        StepGuard("kv_rows", coordinate, 1,
                                  lower_bound=0 if (explicit_rows or unresolved_rows) else 1),
                        "max_s_kv" if operator.name == "kv_read" else "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        positions = max(1, sum(request.kv_len for request in geometry.requests))
        eligible = max(0, geometry.effective_tokens)
        if unresolved_rows:
            rows = 0
        elif explicit_rows:
            declared = int(theta.require("rows"))
            if declared < 0:
                raise ModelingError("index_cache_valid_rows_negative")
            if declared > eligible:
                raise ModelingError(
                    f"index_cache_valid_rows_exceed_eligible_query:{declared}>{eligible}")
            rows = declared
        else:
            rows = eligible
        owner = theta.find("owner_layer", -1)
        if operator.name == "kv_read" and owner is not None and int(owner) >= 0:
            state.require_kv(int(owner))
        values = {"kv_positions": positions, "kv_rows": rows,
                  "valid_rows": rows, "eligible_query_rows": eligible}
        if operator.name != "kv_read":
            # A scattered cache store/reload is a launched AIV copy: its
            # prologue is serial with the row traffic and cannot be hidden by
            # the declared inclusive completion floor.
            floor = invocation_floor(ctx, "vector")
            if floor > 0:
                values["HW_DEPENDENCY_MS"] = floor
        return values

    rows = Var("ROWS")
    data_traffic = Mul((rows, Const(data_bytes)))
    scale_traffic = Mul((rows, Const(scale_bytes)))
    traffic = Add((data_traffic, scale_traffic))
    effects = (("kv", operator.name),)
    stages = None
    if operator.name == "kv_read":
        stage = (_demand(ctx, "mte2", traffic), _demand(ctx, "hbm", traffic),
                 _demand(ctx, "link", _zero()))
        materialization = ()
    elif operator.name == "index_cache_store" and scale_bytes > 0:
        # Two child kernels read update payload + indices and write their own
        # cache slot.  The fused store is one selected boundary and counts its
        # combined source once instead.
        index_dtype = str(theta.find("index_dtype", "int32") or "int32")
        index_bytes = Fraction(8 if index_dtype == "int64" else 4)
        index_traffic = Mul((rows, Const(index_bytes)))
        fused = bool(int(theta.find("fused_store", 0) or 0))
        if fused:
            combined = Add((data_traffic, scale_traffic))
            stages = ((_demand(ctx, "mte2", Add((combined, index_traffic))),
                       _demand(ctx, "mte3", combined),
                       _demand(ctx, "hbm", Add((Mul((Const(2), combined)),
                                                index_traffic)))),)
            materialization = (("index_cache", combined),)
            effects = (("index_k_cache", "index_store"),
                       ("index_scale_cache", "index_store"))
        else:
            k_read = Add((data_traffic, index_traffic))
            scale_read = Add((scale_traffic, index_traffic))
            stages = (
                (_demand(ctx, "mte2", k_read),
                 _demand(ctx, "mte3", data_traffic),
                 _demand(ctx, "hbm", Add((k_read, data_traffic)))),
                (_demand(ctx, "mte2", scale_read),
                 _demand(ctx, "mte3", scale_traffic),
                 _demand(ctx, "hbm", Add((scale_read, scale_traffic)))),
            )
            materialization = (("index_k_cache", data_traffic),
                               ("index_scale_cache", scale_traffic))
            effects = (("index_k_cache", "index_store"),
                       ("index_scale_cache", "index_store"))
    else:
        # Packed valid-slot stores are separate data and scale transactions;
        # zero valid rows produce exactly zero work.
        stage = (_demand(ctx, "mte3", traffic),
                 _demand(ctx, "hbm", traffic))
        materialization = (("kv_cache", traffic),)
    if stages is None:
        stages = (stage,)
    return FamilyTemplate(
        axes=axes, stages=stages, materialization=materialization,
        internal=(), residency=Const(0),
        derived=(("kv_positions", rows), ("kv_rows", Var("kv_rows")),
                 ("kv_bytes", traffic),
                 ("index_data_bytes", data_traffic),
                 ("index_scale_bytes", scale_traffic),
                 ("valid_rows", Var("valid_rows")),
                 ("eligible_query_rows", Var("eligible_query_rows")),
                 ("data_bytes", data_traffic), ("scale_bytes", scale_traffic)),
        regime_exprs=(), binder=binder,
        state_effects=effects,
        annotations=(("family", "kv_access"), ("row_bytes", str(row_bytes)),
                     ("valid_rows_mode", rows_mode)),
    )


def _compile_elementwise(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    elements = int(theta.require("elements"))
    ops = int(theta.find("ops_per_element", 1) or 1)
    act_bytes = _theta_frac(theta, "act_bytes", 2.0)
    read = _theta_frac(theta, "read_bytes", elements * act_bytes)
    write = _theta_frac(theta, "write_bytes", elements * act_bytes)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    moe_top_k = max(1, _theta_int(theta, "k",
                                  _theta_int(theta, "top_k", 1)))
    moe_local = max(1, _theta_int(theta, "local",
                                  _theta_int(theta, "E_local", 1)))
    text = (operator.name + " " + operator.role).lower().replace("_", "")
    logical_elements = elements
    l2_derived: tuple[tuple[str, Expr], ...] = ()
    l2_extra_derived: tuple[tuple[str, Expr], ...] = ()
    l2_annotations: tuple[tuple[str, str], ...] = ()
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    bd_value = int(theta.find("BD", 0) or 0)
    mblock_value = int(theta.find("MBLOCK", 0) or 0)
    l2_active = (bd_value > 0 and str(theta.find("source_kernel", "")) ==
                 "l2norm_fwd_kernel2" and logical_elements > 0)
    l2_n = logical_elements
    l2_out_format = str(theta.find(
        "output_dtype", theta.find("resolved_output_dtype", act_format))
        or act_format)
    l2_out_bytes = Fraction(str(payload_width(l2_out_format)))
    if l2_active:
        # L2Norm keeps the logical M x N value; MBLOCK/BD only pad the issued
        # reduction tiles.  Source/destination ranges are masked by M/N, and
        # the compute lane count is the padded MBLOCK x BD tile, separately.
        elements = l2_n
        ops = 5
        read = Fraction(l2_n) * act_bytes
        write = Fraction(l2_n) * l2_out_bytes
        l2_derived = (
            ("BD", Const(bd_value)), ("MBLOCK", Const(mblock_value)),
            ("l2_N", Const(l2_n)),
            ("issued_elements", Const(l2_n)),
        )
        l2_annotations = (
            ("source_kernel", "l2norm_fwd_kernel2"),
            ("BD", str(bd_value)), ("MBLOCK", str(mblock_value)),
            ("l2_input_dtype", str(theta.find("act_format", "bf16"))),
            ("l2_output_dtype", l2_out_format),
            ("output_alias", "none"),
        )
        l2_extra_derived = (
            ("l2_logical_math", Mul((Var("L2_M"), Const(l2_n * 5)))),
            ("l2_issued_math", Mul((Var("L2_TILE_ROWS"),
                                    Const(bd_value * 5)))),
        )
    rope_derived: tuple[tuple[str, Expr], ...] = ()
    rope_annotations: tuple[tuple[str, str], ...] = ()
    rope_dim = int(theta.find("qk_rope_head_dim", 0) or 0)
    if rope_dim > 0 and "rope" in text:
        # Native SISO rope touches only R elements per head: it reads two R/2
        # qk halves plus cos/sin rows and writes R elements per head.  The
        # untouched [R,D) suffix is a logical alias range, not issued work.
        rope_heads = max(1, _theta_int(
            theta, "index_heads", _theta_int(theta, "heads", 1)))
        rope_dtype = str(theta.find("rope_dtype",
                                    theta.find("act_format", "bf16"))
                         or "bf16")
        cos_dtype = str(theta.find("cos_dtype", rope_dtype) or rope_dtype)
        sin_dtype = str(theta.find("sin_dtype", rope_dtype) or rope_dtype)
        cos_bytes = as_fraction(dtype_bytes(cos_dtype))
        sin_bytes = as_fraction(dtype_bytes(sin_dtype))
        half = max(1, rope_dim // 2)
        elements = rope_heads * rope_dim
        ops = 3
        read = (Fraction(rope_heads) * rope_dim * act_bytes
                + half * cos_bytes + half * sin_bytes)
        write = Fraction(rope_heads) * rope_dim * act_bytes
        rope_derived = (
            ("rope_dim", Const(rope_dim)),
            ("rope_heads", Const(rope_heads)),
            ("issued_elements", Const(elements)),
            ("untouched_elements",
             Const(max(0, logical_elements - elements))),
            ("rope_cos_sin_bytes", Const(half * cos_bytes
                                          + half * sin_bytes)),
        )
        rope_annotations = (
            ("rope_dim", str(rope_dim)),
            ("rope_dtype", rope_dtype),
            ("cos_dtype", cos_dtype),
            ("sin_dtype", sin_dtype),
            ("is_neox_style", str(theta.find("is_neox_style", 0))),
            ("index_rope_role", str(theta.find("index_rope_role", ""))),
            ("output_alias", "qk_in_place"),
        )
    router_op = any(word in text for word in ("topk", "histogram", "sort",
                                              "router"))
    f3_domain = (ctx.kind == "routed_experts"
                 or theta.find("expert_width", None) is not None
                 or (theta.find("k", None) is not None and not router_op))
    # decode_only is the same contract q/k L2-norm already publishes. View and
    # fill launches that set it count decode rows only and stay inactive when
    # the geometry has none, so a prefill request does not pay their floor.
    decode_scoped = int(theta.find("decode_only", 0) or 0) == 1
    prep_scoped = operator.name == "prepare"
    rows_var = "DECODE_ROWS" if decode_scoped else "T"
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("token_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        rows, explicit_routing = geometry.effective_tokens, 0
        decode_rows = sum(1 for request in geometry.requests
                          if request.phase == "decode" and request.q_len > 0)
        if f3_domain:
            # Routed activation lives on the same local-expert domain as the
            # adjacent grouped GEMM: an explicit local shard is consumed as
            # delivered, otherwise the global uniform distribution is
            # localized to this EP shard.
            experts = max(moe_local, _theta_int(theta, "E", moe_local))
            counts = geometry.routing_for(ctx.layer)
            if counts is None:
                try:
                    counts = state.routing_state(ctx.layer).counts
                except ModelingError:
                    counts = None
            if counts is None:
                counts = _counts_for(ctx, geometry, state, moe_local,
                                     moe_top_k, experts)
            else:
                explicit_routing = 1
                if len(counts) == experts and experts != moe_local:
                    counts = tuple(counts)[:moe_local]
            rows = sum(int(value) for value in counts)
        entry_unsharded = (str(ctx.kind or "") == "layernorm"
                           and ctx.layer == 0
                           and "residual_add" not in {
                               getattr(s, "name", "") for s in ctx.siblings})
        if sequence_rows == "shard" and sequence_ranks > 1 and not entry_unsharded:
            bound_rows = int(geometry.effective_tokens)
            if not int(getattr(geometry, "operand_tokens", 0) or 0):
                bound_rows = _ceil_div(bound_rows, sequence_ranks)
            bound = {"SEQUENCE_ROWS": bound_rows}
        else:
            bound = {}
        if l2_active and mblock_value > 0:
            valid_rows = max(0, int(theta.find("M", rows) or rows))
            tile_rows = _ceil_div(valid_rows, mblock_value) * mblock_value
            bound["L2_M"] = valid_rows
            bound["L2_TILE_ROWS"] = tile_rows
            bound["L2_VALID_ROWS"] = valid_rows
            bound["T"] = valid_rows
        bound.update({"elements_per_token": elements, "routed_rows": rows,
                 "ROUTED_ROWS": rows, "EXPLICIT_ROUTING": explicit_routing,
                 "vector_ops_per_token": elements * ops,
                 "read_per_token": read, "write_per_token": write,
                 "DECODE_ROWS": decode_rows,
                 "NORM_SELECTED_ROWS": geometry.producing_lm_head_rows(),
                 "DECODE_ACTIVE": int(decode_rows > 0) if decode_scoped else 1,
                 "PREPARE_ACTIVE": (int(any(request.q_len > 1
                                            for request in geometry.requests))
                                    if prep_scoped else 1)})
        if int(theta.find("quant_passes", 0) or 0) == 2:
            demand = quant_row_demand(
                geometry, kind=str(ctx.kind or ""),
                tp=_theta_int(theta, "tp", 1))
            bound["QUANT_ROWS"] = int(demand.rows)
            bound["QUANT_VALID"] = int(demand.valid)
        bound.update(_moe_comm_bindings(theta, geometry))
        return bound

    sequence_rows = str(theta.find("sequence_rows", "") or "")
    if sequence_rows not in {"", "shard", "selected"}:
        raise ModelingError(f"elementwise_sequence_rows_invalid:{sequence_rows}")
    if sequence_rows and (f3_domain or decode_scoped):
        raise ModelingError("elementwise_sequence_rows_excludes_routed_or_decode_scope")
    sequence_ranks = max(1, _theta_int(theta, "sequence_ranks", 1))
    tokens = (Var("ROUTED_ROWS") if f3_domain else Var(rows_var))
    # One distribution. ``shard`` is ceil(T / ranks) of the geometry's own
    # sequence, the exit gather's input. ``selected`` is the producing row
    # count and is not divided again. Rank 1 leaves T unchanged.
    entry_unsharded = (str(ctx.kind or "") == "layernorm"
                       and ctx.layer == 0
                       and "residual_add" not in {
                           getattr(s, "name", "") for s in ctx.siblings})
    if sequence_rows == "selected":
        tokens = Var("NORM_SELECTED_ROWS")
    elif sequence_rows == "shard" and sequence_ranks > 1 and not entry_unsharded:
        # operand_tokens is the already projected local shard; do not ceil it
        # again.  The binder supplies the same one scalar for legacy views.
        tokens = Var("SEQUENCE_ROWS")
    quant_passes = int(theta.find("quant_passes", 0) or 0)
    if quant_passes == 2:
        # The quant tensor's physical rows, not the graph bucket and not T.
        tokens = Var("QUANT_ROWS")
    quantized = (bool(int(theta.find("quantized", 0) or 0)) or "quant" in text
                 or operator.family in {"quantize", "quant"})
    read_row = _row_bytes(read)
    gamma = _theta_frac(theta, "gamma_bytes", 0.0)
    if gamma and ("norm" in text or "rms" in text or "layer" in text):
        # ``read`` already includes the resident gamma vector.  Separate the
        # per-token activation/residual row here; gamma is added once below.
        read_row = _row_bytes(max(Fraction(0), read - gamma))
    gamma_storage = _theta_frac(theta, "gamma_storage_bytes", gamma)
    gamma_format = str(theta.find("gamma_format", "") or "")
    gamma_storage_format = str(theta.find("gamma_storage_format", "") or "")
    gamma_role = str(theta.find("gamma_role", "") or "")
    mx_native = str(theta.find("native_kind", "")) == "DynamicMxQuant"
    if quantized:
        out_fmt = str(theta.find("out_format", theta.find("output_format", "")) or "").lower()
        mx_scale = math.ceil(elements / 32) if out_fmt in (_MX_FP8 | _MX_FP4) else 0
        scale_row = _theta_frac(theta, "row_scale_bytes", 0)
        write_row = _row_bytes(
            as_fraction(elements) * as_fraction(_quant_output_width(theta))
            + mx_scale + scale_row)
        if mx_native:
            # DynamicMxQuant writes the true packed payload and the padded
            # E8M0 lanes; the tail's reduction padding is not a GM write.
            write_row = _theta_frac(theta, "write_bytes", write_row)
    else:
        write_row = _row_bytes(write)
    gated = 2 if ("glu" in text or "geglu" in text or "swiglu" in text) else 1
    reduction = int(math.ceil(elements / 64))
    quant_vector = (_dynamic_quant_vector(tokens, elements) if quantized
                    else Const(0))
    if "norm" in text or "rms" in text or "layer" in text:
        # A row norm is two passes over the UB row: sum(x^2)/rstd then the
        # normalized write.  The gamma vector is broadcast-preloaded once per
        # block by the native occupancy pass, so the declared ``ops`` already
        # carries the per-element vector work; do not add a second pass here.
        vector = Add((Mul((tokens, Const(elements * ops))),
                      Mul((tokens, Const(reduction)))))
        if quantized:
            vector = Add((vector, quant_vector))
    elif "histogram" in text or "sort" in text or "topk" in text:
        vector = Mul((tokens, Const(elements * max(1, ops))))
    elif quantized and gated == 1:
        vector = quant_vector
    else:
        vector = Mul((tokens, Const(elements * ops * gated)))
        if quantized:
            vector = Add((vector, quant_vector))
    mx_row_ops = _theta_frac(theta, "mx_vector_ops_per_row", 0.0)
    if mx_row_ops:
        # Raw DynamicMxQuant: max-exponent/scale construction/clamp-cast are
        # per occupied row, not the generic per-token integer reduction.
        vector = Mul((tokens, Const(mx_row_ops)))
    if "layoutconvert" in text or ("cast" in text and "broadcast" not in text):
        vector = Add((vector, Mul((tokens, Const(elements * 2)))))
    if mx_native:
        read_row = _theta_frac(theta, "read_bytes", read_row)
    reads = Mul((tokens, Const(read_row)))
    if gamma and ("norm" in text or "rms" in text or "layer" in text):
        # The gamma vector is resident once and read once per launch, not once
        # per token.  Activation read, output write and vector/cube work stay
        # on their existing paths.
        reads = Add((reads, Const(gamma)))
    writes = Mul((tokens, Const(write_row)))
    gather = "gather" in text
    if gather:
        reads = Add((reads, Mul((tokens, Const(
            _theta_frac(theta, "index_bytes", 8.0))))))
    gamma_derived = ()
    gamma_annotations = ()
    if gamma:
        gamma_derived = (
            ("gamma_bytes", Const(gamma)),
            ("gamma_storage_bytes", Const(gamma_storage)),
        )
        if gamma_format:
            gamma_annotations += (("gamma_format", gamma_format),)
        if gamma_storage_format:
            gamma_annotations += (("gamma_storage_format", gamma_storage_format),)
        if gamma_role:
            gamma_annotations += (("gamma_role", gamma_role),)
    quant_derived: tuple[tuple[str, Expr], ...] = ()
    quant_notes: tuple[tuple[str, str], ...] = ()
    duration_fn = None
    if quant_passes == 2:
        # pass_read_bytes is one UB-resident copy of x, not a second GM load.
        source_row = _row_bytes(_theta_frac(theta, "pass_read_bytes", read))
        read_row = source_row
        source_bytes = _theta_frac(theta, "source_bytes", act_bytes)
        plan = host_source_copies(
            elements, _machine_ub(ctx, theta),
            has_smooth=int(theta.find("has_smooth", 0) or 0) == 1)
        gm_row = sum(
            _row_bytes(as_fraction(count) * as_fraction(source_bytes))
            for count in plan["copies"])
        reads = Mul((tokens, Const(gm_row)))
        if gamma and ("norm" in text or "rms" in text or "layer" in text):
            reads = Add((reads, Const(gamma)))
        if gather:
            reads = Add((reads, Mul((tokens, Const(
                _theta_frac(theta, "index_bytes", 8.0))))))
        aiv = _vector_owners(ctx, theta, cores)
        quant_derived = (
            ("quant_passes", Const(quant_passes)),
            ("row_scale_bytes", Const(_theta_frac(theta, "row_scale_bytes", 0))),
            ("offset_gm_bytes", Const(_theta_frac(theta, "offset_gm_bytes", 0))),
            ("launched_blocks", Min((tokens, Const(aiv)))),
            ("aiv_owners", Const(aiv)),
            ("rows_per_head_core", CeilDiv(tokens, Const(aiv))),
            ("rows_per_tail_core", FloorDiv(tokens, Const(aiv))),
            ("gm_source_reads", Const(int(plan["gm_source_reads"]))),
            ("gm_read_row_bytes", Const(gm_row)),
            ("use_db", Const(1 if plan["use_db"] else 0)),
            ("ub_mode", Const(int(plan["mode_code"]))),
        )
        quant_notes = (("ub_mode", str(plan["mode"])),
                       ("symmetrical", str(int(theta.find("symmetrical", 1) or 1))),
                       ("row_source", ROW_SOURCE),
                       ("vector_split", VECTOR_SPLIT),
                       ("occupancy", OCCUPANCY))
        # DataCopyPad writes y and one FP32 scale. The aligned combined
        # footprint stays on ub_write_row_bytes; it is not the GM store.
        out_fmt = str(theta.find("out_format", "") or "")
        payload = (as_fraction(_payload_width(out_fmt)) if out_fmt
                   else as_fraction(_quant_output_width(theta)))
        y_gm = as_fraction(elements) * payload
        scale_gm = _theta_frac(theta, "row_scale_bytes", 0)
        writes = Mul((tokens, Const(y_gm + scale_gm)))
        quant_derived = quant_derived + (
            ("y_gm_row_bytes", Const(y_gm)),
            ("scale_gm_row_bytes", Const(scale_gm)),
            ("physical_rows", tokens),
            ("valid_rows", Var("QUANT_VALID")),
        )
        source_width = float(source_bytes)
        y_width = float(payload)
        scale_width = float(scale_gm)
        smooth = int(theta.find("has_smooth", 0) or 0) == 1
        ub_bytes = _machine_ub(ctx, theta)
        kind_name = str(ctx.kind or "")
        tp_width = _theta_int(theta, "tp", 1)
        rate_key = tuple(ctx.supply.rates)
        unit_key = tuple(ctx.supply.resource_units or ())
        row_len = int(elements)

        def duration_fn(geometry, _state, *, _kind=kind_name, _tp=tp_width,
                        _rates=rate_key, _units=unit_key, _ub=ub_bytes,
                        _src=source_width, _y=y_width, _scale=scale_width,
                        _smooth=smooth, _aiv=aiv, _row=row_len):
            demand = quant_row_demand(geometry, kind=_kind, tp=_tp)
            return quant_program_result(
                rates=dict(_rates), units=dict(_units), rows=int(demand.rows),
                row_len=_row, source_bytes=_src, y_bytes=_y,
                scale_bytes=_scale, ub_bytes=_ub, has_smooth=_smooth, aiv=_aiv)
    if mx_row_ops:
        quant_derived = (
            ("mx_vector_ops_per_row", Const(int(mx_row_ops))),
            ("mx_groups_per_row", Const(int(_theta_frac(theta, "mx_groups_per_row", 0)))),
            ("mx_scale_row_bytes", Const(int(_theta_frac(theta, "mx_scale_row_bytes", 0)))),
            ("dynamic_mx_quant", Const(1)),
            ("mx_register_calls", Const(int(_theta_frac(theta, "mx_register_calls", 0)))),
            ("mx_issue_cycles_known", Const(int(_theta_frac(theta, "mx_issue_cycles_known", 0)))),
        )
        quant_notes = (("quant_op", "DynamicMxQuant"),
                       ("activation_scale_format", "fp8_e8m0"),
                       ("scale_group", "32"))
    block_axis = str(theta.find("block_axis", "") or "")
    if block_axis == "row":
        # Occupied rows pay the vector repeat.  The repeat is not a GM load,
        # and the AIC cap is not the AIV block count.
        source_bytes = _theta_frac(theta, "source_bytes", act_bytes)
        vector_pipe = Max((vector, Mul((tokens, Const(_aiv_quantum(source_bytes))))))
    else:
        vector_pipe = Max((vector, _aiv_floor(tokens, mt, cores, act_bytes)))
    if l2_active:
        m = Var("L2_M")
        reads = Mul((m, Const(l2_n * act_bytes)))
        writes = Mul((m, Const(l2_n * l2_out_bytes)))
        tile_rows = Var("L2_TILE_ROWS")
        vector_pipe = Mul((tile_rows, Const(bd_value * 5)))
    main = (
        _demand(ctx, "vector", vector_pipe), _demand(ctx, "mte2", reads),
        _demand(ctx, "mte3", writes), _demand(ctx, "hbm", Add((reads, writes))),
    )
    stages = (main,)
    extra_derived = quant_derived
    sequence_derived = (("norm_rows", tokens),) if sequence_rows else ()
    sequence_notes = (("sequence_rows", sequence_rows),) if sequence_rows else ()
    materialization = ((_out_name(ctx), writes),)
    rstd_row = _theta_frac(theta, "rstd_bytes", 0.0)
    residual_row = _theta_frac(theta, "residual_out_bytes", 0.0)
    if rstd_row or residual_row:
        # The write ledger is padded, but the three logical outputs are not.
        # Keep issued mte3/hbm separate from the actual materialized roles.
        norm_logical = max(Fraction(0), write - rstd_row - residual_row)
        norm_out = Mul((tokens, Const(norm_logical)))
        materialization = ((_out_name(ctx), norm_out),
                           ("residual", Mul((tokens, Const(residual_row)))),
                           ("rstd", Mul((tokens, Const(rstd_row)))))
    if str(theta.find("native_kind", "")) == "DynamicMxQuant":
        base = _out_name(ctx)
        scale_row = _theta_frac(theta, "row_scale_bytes", 0.0)
        materialization = (
            (f"{base}_payload", Mul((tokens, Const(elements)))),
            (f"{base}_scale", Mul((tokens, Const(scale_row)))),
        )
    if l2_active:
        materialization = ((_out_name(ctx), writes),)
    return FamilyTemplate(
        axes=axes, stages=stages, materialization=materialization,
        internal=(), residency=Const(0),
        derived=(("elements", Mul((tokens, Const(elements)))),
                 ("read_bytes", reads), ("write_bytes", writes),
                 ("vector_ops", vector),
                 ("ub_row_bytes", Const(read_row)),
                 ("ub_write_row_bytes", Const(write_row)),
                 ("ub_padded_bytes", reads),
                 ("vector_reduction_ops", Mul((tokens, Const(reduction)))),
                 ("gated_inputs", Mul((tokens, Const(elements * gated)))),
                 ("gather_indices", tokens if gather else Const(0)),
                 ("quantized_output", Const(int(quantized))),
                 ("logical_elements",
                  Mul((tokens, Const(logical_elements)))),
                 *l2_derived, *l2_extra_derived, *rope_derived,
                 *((( "rope_math_ops",
                     Mul((tokens, Const(3 * elements)))),)
                   if rope_dim > 0 and "rope" in text else ()),
                 *gamma_derived, *extra_derived,
                 *sequence_derived),
        regime_exprs=(), binder=binder,
        annotations=(("family", "elementwise"), ("ops_per_element", str(ops)),
                     *l2_annotations, *rope_annotations, *gamma_annotations,
                     *quant_notes,
                     *sequence_notes),
        active=((_moe_comm_active(theta) if theta.find("moe_comm", None) is not None
                 else None) or
                (Var("DECODE_ACTIVE") if decode_scoped else
                 Var("PREPARE_ACTIVE") if prep_scoped else None)),
        program_duration=duration_fn,
    )


def _phase_tokens(geometry: Any) -> tuple[int, int, int]:
    """Forward, decode and prefill query rows. Bucket padding is not a row."""
    forward = decode = prefill = 0
    for request in geometry.requests:
        rows = int(request.q_len)
        forward += rows
        if request.phase == "decode":
            decode += rows
        elif request.phase == "prefill":
            prefill += rows
    return forward, decode, prefill


def _materialized_rows(source: str, forward: int, decode: int, prefill: int) -> int:
    """Rows a v0.30.0 contiguous copy or zero fill actually allocates.

    A last-dimension split of one row is already contiguous. A multi-row
    split is not, so ``contiguous`` and a non-contiguous ``reshape`` copy.
    The zero fill is ``torch.zeros`` and allocates at any row count.
    """
    if source == "forward_fill":
        return forward
    if source == "forward_multirow":
        return forward if forward > 1 else 0
    if source == "decode_multirow":
        return decode if decode > 1 else 0
    if source == "conv_qkv":
        # Each side is copied only when that side itself has more than one
        # row. One decode token beside a prefill copies the prefill rows only.
        rows = 0
        if decode > 1:
            rows += decode
        if prefill > 1:
            rows += prefill
        return rows
    raise ModelingError(f"gdn_transfer_row_source_invalid:{source}")


def _compile_materialized_transfer(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    """MTE copy into an independent buffer, or a native zero fill.

    Slices observed with ``aiv_vec_time`` of 0 declare no vector work. The
    kept bytes move on MTE2 and MTE3 and land in a dense buffer on HBM.
    A fill reads nothing, writes the buffer, and does have vector work, so
    the declared vector completion floor still applies. No new latency
    constant is introduced, and a one-row view stays inactive.
    """
    theta = operator.theta
    elements = int(theta.require("elements"))
    source = str(theta.require("row_source"))
    transfer = str(theta.require("transfer"))
    if transfer not in {"mte", "fill"}:
        raise ModelingError(f"gdn_transfer_invalid:{transfer}")
    # read_bytes and write_bytes are the captured buffer width. quant.act_bytes
    # is the W8 activation format and must not price these DT_BF16 copies:
    # StaticSignature.find keeps the later quant entry when both are set.
    read_bytes = _theta_frac(theta, "read_bytes", elements * 2)
    write_bytes = _theta_frac(theta, "write_bytes", elements * 2)
    declared = write_bytes if transfer == "fill" else read_bytes
    elem_bytes = (declared / elements) if elements and declared else as_fraction(2)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    ops = int(theta.find("ops_per_element", 1) or 1)
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("copy_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        del state
        # Static dense-SP decode feeds the same contiguous rule a captured
        # width. Logical phases, including a one-row view, stay on request q_len.
        forward, decode, prefill = geometry.physical_gdn_rows()
        return {"COPY_ROWS": _materialized_rows(source, forward, decode, prefill)}

    rows = Var("COPY_ROWS")
    read = Mul((rows, Const(read_bytes)))
    write = Mul((rows, Const(write_bytes)))
    if transfer == "fill":
        vector = Max((Mul((rows, Const(elements * ops))),
                      _aiv_floor(rows, mt, cores, elem_bytes)))
        main = (
            _demand(ctx, "vector", vector),
            _demand(ctx, "mte3", write),
            _demand(ctx, "hbm", write),
        )
        derived_vector = vector
    else:
        main = (
            _demand(ctx, "mte2", read),
            _demand(ctx, "mte3", write),
            _demand(ctx, "hbm", Add((read, write))),
        )
        derived_vector = Const(0)
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=((_out_name(ctx), write),),
        internal=(), residency=Const(0),
        derived=(("copy_rows", rows),
                 ("copy_bytes", write),
                 ("vector_ops", derived_vector)),
        regime_exprs=(), binder=binder,
        annotations=(("family", "copy"), ("transfer", transfer)),
        active=rows,
    )


def _flat_typed_operand(
    name: str, direction: str, representation: str, dtype: str,
    key: str, value: Fraction, *, tokens: int = 0, width_elements: int = 0,
) -> TypedOperand | None:
    """Build a typed operand from an exact derived byte total."""
    if value <= 0:
        return None
    try:
        width = Fraction(str(payload_width(dtype)))
    except (TypeError, ValueError):
        width = Fraction(0)
    if width <= 0:
        return None
    if width_elements > 0 and tokens > 0 and value == (
            tokens * width_elements * width):
        shape = (tokens, width_elements)
        dims: tuple[str | int, ...] = shape
        values: tuple[tuple[str, int], ...] = ()
        layout = TensorLayout.contiguous(shape)
    elif value % width == 0:
        elements = int(value / width)
        dims = ("elements",)
        values = (("elements", elements),)
        shape = (elements,)
        layout = TensorLayout.contiguous(shape)
    else:
        dims = ("bytes",)
        values = (("bytes", int(value)),)
        shape = (int(value),)
        dtype = "uint8"
        width = Fraction(1)
        layout = TensorLayout.contiguous(shape)
    return TypedOperand(
        name, direction, representation, dtype, "strided", dims,
        values=values, bytes_per_element=width, bytes_key=key,
        quant=dtype if representation in {"activation", "quantized"} else "",
        source=f"cost.derived:{key}", layout_spec=layout)


def _elementwise_typed_declaration(compiled: CompiledOperator,
                                   cost: Any, context: Any = None) -> FamilyDeclaration:
    """Per-value elementwise/norm declaration.

    The aggregate ``read_bytes`` is not a tensor shape.  Norm-style operators
    publish activation, gamma, residual, rstd and the real output separately;
    issued/padded traffic stays in the family work ledger.
    """
    theta = compiled.operator.theta
    derived = dict(cost.derived)
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    out_format = str(theta.find(
        "output_format", theta.find(
            "out_format", theta.find("output_dtype", act_format)))
        or act_format)
    base_elements = int(theta.find("elements", 0) or 0)
    total_elements = int(derived.get("elements", 0) or 0)
    tokens = (total_elements // base_elements
              if base_elements > 0 and total_elements % base_elements == 0
              else 0)
    width = base_elements
    act_width = Fraction(str(payload_width(act_format)))
    operands: list[TypedOperand] = []
    missing: list[str] = []

    def make(name: str, direction: str, representation: str, dtype: str,
             amount: Fraction, shape: tuple[int, ...],
             key: str = "", quant: str = "") -> TypedOperand | None:
        value = as_fraction(amount)
        if value <= 0:
            return None
        try:
            elem_width = Fraction(str(payload_width(dtype)))
        except (TypeError, ValueError):
            elem_width = Fraction(0)
        if elem_width <= 0:
            missing.append(f"{name}:element_dtype")
            return None
        if (all(dim >= 0 for dim in shape)
                and math.prod(shape) * elem_width == value):
            dims: tuple[str | int, ...] = tuple(shape)
            values: tuple[tuple[str, int], ...] = ()
            layout = TensorLayout.contiguous(shape)
        elif value % elem_width == 0:
            elements = int(value / elem_width)
            dims = ("elements",)
            values = (("elements", elements),)
            layout = TensorLayout.contiguous((elements,))
        else:
            dims = ("bytes",)
            values = (("bytes", int(value)),)
            elem_width = Fraction(1)
            dtype = "uint8"
            key = ""
            layout = TensorLayout.contiguous((int(value),))
            missing.append(f"{name}:element_dtype")
        return TypedOperand(
            name, direction, representation, dtype, "strided", dims,
            values=values, bytes_per_element=elem_width, bytes_key=key,
            quant=quant, source="theta.extras / cost.materialization",
            layout_spec=layout)

    rope_dim = int(theta.find("qk_rope_head_dim", 0) or 0)
    rope_text = (compiled.name + " " + str(
        getattr(compiled, "role", "") or theta.find("role", ""))).lower()
    if rope_dim > 0 and "rope" in rope_text:
        logical = int(derived.get("logical_elements", total_elements) or 0)
        if logical > 0 and base_elements > 0:
            tokens = logical // base_elements
        heads = max(1, int(theta.find(
            "index_heads", theta.find("heads", 1)) or 1))
        dim = max(1, base_elements // heads)
        half = max(1, rope_dim // 2)
        rope_dtype = str(theta.find("rope_dtype", act_format) or act_format)
        cos_dtype = str(theta.find("cos_dtype", rope_dtype) or rope_dtype)
        sin_dtype = str(theta.find("sin_dtype", rope_dtype) or rope_dtype)
        qk_shape = (tokens, heads, dim)
        qk_bytes = Fraction(logical or (tokens * heads * dim)) * act_width
        qk = make("qk", "in", "activation", act_format, qk_bytes,
                  qk_shape, "", act_format)
        if qk is not None:
            operands.append(qk)
        for name, dtype in (("cos", cos_dtype), ("sin", sin_dtype)):
            try:
                width = Fraction(str(payload_width(dtype)))
            except (TypeError, ValueError):
                missing.append(f"{name}:dtype")
                continue
            amount = Fraction(tokens * half) * width
            operand = make(name, "in", "rope_angle", dtype, amount,
                           (tokens, half), "", dtype)
            if operand is not None:
                operands.append(operand)
            else:
                missing.append(f"{name}:producer")
        out_shape = (tokens, heads, dim)
        out_bytes = Fraction(tokens * heads * dim) * Fraction(
            str(payload_width(out_format)))
        output = make("output", "out", "activation", out_format, out_bytes,
                      out_shape, "", out_format)
        if output is not None:
            operands.append(output)
        attrs = [SemanticAttr("family", compiled.family, "", "compiled.family"),
                 SemanticAttr("rope_dim", rope_dim, "elements",
                              "theta.extras:qk_rope_head_dim"),
                 SemanticAttr("rope_heads", heads, "", "theta.extras:index_heads"),
                 SemanticAttr("rope_dtype", rope_dtype, "",
                              "theta.extras:rope_dtype"),
                 SemanticAttr("cos_dtype", cos_dtype, "",
                              "theta.extras:cos_dtype|rope_dtype"),
                 SemanticAttr("sin_dtype", sin_dtype, "",
                              "theta.extras:sin_dtype|rope_dtype"),
                 SemanticAttr("is_neox_style", int(theta.find(
                     "is_neox_style", 0) or 0), "",
                     "theta.extras:is_neox_style"),
                 SemanticAttr("output_alias", "qk_in_place", "",
                              "NATIVE SISO rope in-place output")]
        unique_missing = tuple(dict.fromkeys(missing))
        return FamilyDeclaration(tuple(operands), tuple(attrs),
                                 producer_complete=not unique_missing,
                                 missing=unique_missing)

    # The evaluated ledger is the actual row domain.  Theta read_bytes is a
    # static per-row/template value and must not override a 2-row instance.
    read_total = as_fraction(derived.get(
        "read_bytes", theta.find("read_bytes", 0)))
    gamma_bytes = as_fraction(theta.find("gamma_bytes", 0))
    gamma_format = str(theta.find("gamma_format", "") or "")
    residual_add = int(theta.find("residual_add", 0) or 0)
    residual_in_bytes = (tokens * width * act_width
                         if residual_add and tokens and width else Fraction(0))
    activation_bytes = read_total - gamma_bytes - residual_in_bytes
    if activation_bytes <= 0:
        activation_bytes = read_total - gamma_bytes
    if activation_bytes <= 0:
        activation_bytes = read_total
    activation = make("activation", "in", "activation", act_format,
                      activation_bytes, (tokens, width), "", act_format)
    if activation is not None:
        operands.append(activation)
    if gamma_bytes > 0 and gamma_format:
        try:
            gamma_width = Fraction(str(payload_width(gamma_format)))
        except (TypeError, ValueError):
            gamma_width = Fraction(0)
        if gamma_width > 0:
            gamma_elements = int(gamma_bytes / gamma_width)
            gamma = make("gamma", "in", "norm_weight", gamma_format,
                         gamma_bytes, (gamma_elements,), "gamma_bytes")
            if gamma is not None:
                operands.append(gamma)
            else:
                missing.append("gamma_producer")
        else:
            missing.append("gamma_dtype")
    elif gamma_bytes > 0:
        missing.append("gamma_format")
    if residual_in_bytes > 0:
        residual = make("residual_in", "in", "activation", act_format,
                        residual_in_bytes, (tokens, width), "", act_format)
        if residual is not None:
            operands.append(residual)
        else:
            missing.append("residual_producer")

    emitted: set[str] = set()
    for name, amount in cost.materialization:
        label = str(name)
        value = as_fraction(amount)
        if value <= 0 or label in emitted:
            continue
        emitted.add(label)
        lower = label.lower()
        if "rstd" in lower:
            element_dtype, representation = "fp32", "rstd"
            shape = (tokens,)
        elif "residual" in lower:
            element_dtype, representation = act_format, "activation"
            shape = (tokens, width)
        elif "scale" in lower:
            element_dtype = str(theta.find(
                "activation_scale_format", "fp8_e8m0") or "fp8_e8m0")
            representation, shape = "scale", (0,)
        elif "gamma" in lower:
            continue
        else:
            element_dtype, representation = out_format, "activation"
            shape = (tokens, width)
        operand = make(label, "out", representation, element_dtype, value,
                       shape, label, act_format if representation == "activation" else "")
        if operand is not None:
            operands.append(operand)
        else:
            missing.append(f"{label}:producer")

    attrs = [SemanticAttr("family", compiled.family, "", "compiled.family"),
             SemanticAttr("act_format", act_format, "", "theta.quant:act_format"),
             SemanticAttr("output_format", out_format, "",
                          "theta.quant:output_format|act_format"),
             SemanticAttr("elements", total_elements, "elements",
                          "cost.derived:elements")]
    for key in ("ops_per_element", "quantized", "gamma_format",
                "gamma_bytes", "gamma_storage_bytes", "residual_add",
                "rstd_bytes", "activation_scale_format"):
        value = theta.find(key, None)
        if value is not None and isinstance(value, (int, float, str, bool)):
            attrs.append(SemanticAttr(key, value, "", f"theta.quant:{key}"))
    if activation is None:
        missing.append("logical_input_producer")
    if any(operand.dtype == "uint8" for operand in operands):
        missing.append("element_dtype_or_shape")
    unique_missing = tuple(dict.fromkeys(missing))
    return FamilyDeclaration(
        tuple(operands), tuple(attrs),
        producer_complete=not unique_missing,
        missing=unique_missing)

def _copy_typed_declaration(compiled: CompiledOperator,
                            cost: Any, context: Any = None) -> FamilyDeclaration:
    theta = compiled.operator.theta
    derived = dict(cost.derived)
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    base_elements = int(theta.find("elements", 0) or 0)
    tokens = int(derived.get("copy_rows", 0) or 0)
    total_elements = tokens * base_elements
    operands: list[TypedOperand] = []
    read_key = "copy_read_bytes" if "copy_read_bytes" in derived else "copy_bytes"
    read = _flat_typed_operand(
        "input", "in", "activation", act_format, read_key,
        as_fraction(derived.get(read_key, 0)),
        tokens=tokens, width_elements=base_elements)
    if read is not None:
        operands.append(read)
    write_key = ("copy_write_bytes" if "copy_write_bytes" in derived
                 else "copy_bytes")
    write = _flat_typed_operand(
        "output", "out", "activation", act_format, write_key,
        as_fraction(derived.get(write_key, 0)),
        tokens=tokens, width_elements=base_elements)
    if write is not None:
        operands.append(write)
    attrs = [SemanticAttr("family", compiled.family, "", "compiled.family"),
             SemanticAttr("act_format", act_format, "", "theta.quant:act_format"),
             SemanticAttr("elements", total_elements, "elements", "cost.derived:elements")]
    missing = [] if len(operands) == 2 else ["logical_value_producer"]
    return FamilyDeclaration(tuple(operands), tuple(attrs),
                             producer_complete=not missing,
                             missing=tuple(missing))


def _embedding_typed_declaration(compiled: CompiledOperator,
                                 cost: Any, context: Any = None) -> FamilyDeclaration:
    theta = compiled.operator.theta
    derived = dict(cost.derived)
    act_format = str(theta.find("act_format", "bf16") or "bf16")
    lookup_rows = int(derived.get("lookup_rows", derived.get("gather_indices", 0)) or 0)
    operands: list[TypedOperand] = []
    index_bytes = as_fraction(derived.get("index_bytes", 0))
    if lookup_rows > 0 and index_bytes > 0:
        index_width = index_bytes / lookup_rows
        index_dtype = "int64" if index_width == 8 else "int32"
        operands.append(TypedOperand(
            "token_ids", "in", "index", index_dtype, "strided",
            ("rows",), values=(("rows", lookup_rows),),
            bytes_per_element=index_bytes / lookup_rows,
            bytes_key="index_bytes", source="cost.derived:index_bytes",
            layout_spec=TensorLayout.contiguous((lookup_rows,))))
    for name, amount in cost.materialization:
        label = str(name)
        value = as_fraction(amount)
        width = payload_width(act_format)
        if lookup_rows > 0 and value % (lookup_rows * Fraction(str(width))) == 0:
            width_elements = int(value / (lookup_rows * Fraction(str(width))))
            operand = TypedOperand(
                label, "out", "activation", act_format, "strided",
                ("rows", "width"), values=(("rows", lookup_rows),
                                          ("width", width_elements)),
                bytes_per_element=Fraction(str(width)), bytes_key=label,
                source="cost.materialization",
                layout_spec=TensorLayout.contiguous((lookup_rows, width_elements)))
        else:
            operand = _flat_typed_operand(
                label, "out", "activation", act_format,
                label, value, tokens=lookup_rows,
                width_elements=int(theta.find("table_cols", 0) or 0))
        if operand is not None:
            operands.append(operand)
    attrs = [SemanticAttr("family", compiled.family, "", "compiled.family"),
             SemanticAttr("act_format", act_format, "", "theta.quant:act_format"),
             SemanticAttr("table_rows", int(theta.find("table_rows", 0) or 0),
                          "elements", "theta.matrix:table_rows"),
             SemanticAttr("table_cols", int(theta.find("table_cols", 0) or 0),
                          "elements", "theta.matrix:table_cols")]
    missing = [] if operands and any(o.name == "token_ids" for o in operands) else ["logical_index_producer"]
    return FamilyDeclaration(tuple(operands), tuple(attrs),
                             producer_complete=not missing,
                             missing=tuple(missing))


def _compile_copy(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    if theta.find("row_source", None) is not None:
        return _compile_materialized_transfer(ctx, operator)
    elements = int(theta.require("elements"))
    elem_bytes = _theta_frac(theta, "act_bytes", 2.0)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    cores = max(1, _theta_int(theta, "cores", ctx.policy.cores))
    text = (operator.name + " " + operator.role).lower()
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("copy_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        values = {"copy_elements": elements,
                  "copy_bytes": elements * elem_bytes}
        # A slice/scatter copy issues whole 128 B HBM cachelines per row; a
        # non-contiguous tail still pays its last line.  The copy kernel also
        # pays the declared vector prologue, which is serial with the copy.
        floor = invocation_floor(ctx, "vector")
        if floor > 0:
            values["HW_DEPENDENCY_MS"] = floor
        return values

    # A producing+selected gather keeps the producing token extent in the
    # geometry but prices/activates only the legitimate selected rows.
    row_var = ("T_SELECTED"
               if str(theta.find("row_binding", "") or "") == "selected"
               else "T")
    rows = Var(row_var)
    read_bytes = _theta_frac(theta, "read_bytes", elements * elem_bytes)
    write_bytes = _theta_frac(theta, "write_bytes", elements * elem_bytes)
    # Non-contiguous rows are fetched/written as whole 128 B cachelines.
    read_bytes = Fraction(math.ceil(read_bytes / 128) * 128)
    write_bytes = Fraction(math.ceil(write_bytes / 128) * 128)
    traffic = Mul((rows, Const(write_bytes)))
    read = Mul((rows, Const(read_bytes)))
    if "gather" in text:
        index_bytes = _theta_frac(theta, "index_bytes", 8.0)
        read = Add((read, Mul((rows, Const(index_bytes)))))
    lane = Mul((rows, Const(elements * 2)))
    main = (
        _demand(ctx, "vector", Max((lane, _aiv_floor(
            rows, mt, cores, elem_bytes)))),
        _demand(ctx, "mte2", read), _demand(ctx, "mte3", traffic),
        _demand(ctx, "hbm", Add((read, traffic))),
    )
    effects = (("residual", "push"),) if ctx.kind == "attn_residual" and ctx.name == "slot_push" else ()
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=((_out_name(ctx), traffic),),
        internal=(), residency=Const(0),
        derived=(("copy_bytes", traffic), ("copy_read_bytes", read),
                 ("copy_write_bytes", traffic),
                 ("gather_indices", rows if "gather" in text else Const(0))),
        regime_exprs=(), binder=binder, state_effects=effects,
        annotations=(("family", "copy"),),
    )


def _compile_vocab_lookup(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    """Gather live token rows from the resident table.

    The table stays resident. A step reads one stored row per live id
    (``T_VALID``), never the graph bucket and never the vocabulary. A sharded
    table zeros out-of-range ids on those same rows. That mask is not a
    second hidden tensor and not a collective.
    """
    theta = operator.theta
    table_rows = int(theta.require("table_rows"))
    width = int(theta.require("table_cols"))
    if table_rows < 0 or width < 0:
        raise ModelingError(f"embedding_table_invalid:{table_rows}:{width}")
    elem_bytes = _theta_frac(theta, "act_bytes", 2.0)
    weight_total = _theta_frac(theta, "weight_total", 0.0)
    masked = int(theta.find("masked", 0) or 0) == 1
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    if table_rows > 0 and weight_total:
        read_row = weight_total / table_rows
    else:
        read_row = as_fraction(width) * elem_bytes
    write_row = as_fraction(width) * elem_bytes
    index_row = _theta_frac(theta, "index_bytes", 8.0)
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("embed_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        del geometry, state
        return {"embedding_width": width, "table_rows": table_rows}

    live = Var("T_VALID")
    row_read = Mul((live, Const(read_row)))
    index_read = Mul((live, Const(index_row)))
    read = Add((row_read, index_read))
    write = Mul((live, Const(write_row)))
    vector = Mul((live, Const(width * (2 if masked else 1))))
    main = (
        _demand(ctx, "mte2", read), _demand(ctx, "mte3", write),
        _demand(ctx, "hbm", Add((read, write))),
        _demand(ctx, "vector", vector),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=(("hidden", write),),
        internal=(), residency=Const(weight_total),
        derived=(("gather_bytes", row_read), ("gather_indices", live),
                 ("lookup_rows", live), ("index_bytes", index_read),
                 ("table_rows", Const(table_rows)), ("table_cols", Const(width)),
                 ("masked_fill", Const(int(masked))),
                 ("embedding_width", Const(width))),
        regime_exprs=(), binder=binder,
        annotations=(("family", "embedding"), ("lookup", "1"),
                     ("table_rows", str(table_rows)),
                     ("table_cols", str(width)),
                     ("masked", "1" if masked else "0")),
    )


def _compile_embedding(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    if theta.find("table_rows", None) is not None:
        return _compile_vocab_lookup(ctx, operator)
    width = int(theta.require("K")) if theta.find("K", None) is not None else int(theta.require("width"))
    elem_bytes = _theta_frac(theta, "act_bytes", 2.0)
    weight_total = _theta_frac(theta, "weight_total", 0.0)
    mt = _theta_int(theta, "mt", ctx.policy.mt)
    axes = (SegmentAxis("TOKEN_TILES", "tokens",
                        StepGuard("embed_tiles", "tokens", mt), "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        return {"embedding_width": width, "embedding_bytes": width * elem_bytes}

    traffic = Mul((Var("T"), Const(width * elem_bytes)))
    main = (
        _demand(ctx, "mte2", traffic), _demand(ctx, "mte3", traffic),
        _demand(ctx, "hbm", Add((traffic, traffic))),
        _demand(ctx, "vector", Mul((Var("T"), Const(width)))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=(("hidden", traffic),),
        internal=(), residency=Const(weight_total),
        derived=(("gather_bytes", traffic), ("gather_indices", Var("T")),
                 ("embedding_width", Const(width))),
        regime_exprs=(), binder=binder,
        annotations=(("family", "embedding"),),
    )


# Typed family projections registered with ``families_bridge``.  These are
# declarations over the same theta/work fields above; they do not add a
# second numerical path, and only a registered family is reported as projected.
register_family_operands(
    "gemm", _gemm_typed_declaration,
    source="families_matrix:_gemm_typed_declaration")
register_family_operands(
    "kv_access", _kv_access_typed_declaration,
    source="families_matrix:_kv_access_typed_declaration")


register_family_operands(
    "elementwise", _elementwise_typed_declaration,
    source="families_matrix:_elementwise_typed_declaration")
register_family_operands(
    "quantize", _elementwise_typed_declaration,
    source="families_matrix:_elementwise_typed_declaration")
register_family_operands(
    "hc_post", _elementwise_typed_declaration,
    source="families_matrix:_elementwise_typed_declaration")
register_family_operands(
    "copy", _copy_typed_declaration,
    source="families_matrix:_copy_typed_declaration")
register_family_operands(
    "embedding", _embedding_typed_declaration,
    source="families_matrix:_embedding_typed_declaration")
