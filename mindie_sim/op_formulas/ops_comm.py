"""Formula builders for HCCL collectives and MC2 MoE dispatch/combine.

[Role]
  Model ring and halving-doubling collectives with their real wire volume and
  step counts, variable-split AllToAllV, and MC2 dispatch/combine with an
  explicit locality proof: ``world_size == 1`` and ``ep == 1`` remove wire
  traffic entirely, while EP>1 keeps a guaranteed remote floor separate from the
  balanced reference and qualifies the measured A5 fit domain.

[Invariants]
  * ``world_size == 1`` collectives and ``ep == 1`` MC2 have ``comm_bytes == 0``.
  * Ring/Halving-Doubling differ in step count; ``hd`` on a non-power-of-two
    world size is rejected instead of silently degraded.
  * AllToAllV validates non-negative split vectors and subtracts the local
    self-transfer from wire bytes.
  * MC2 bounds local arrivals by the global population ``G=T_rank*ep``;
    ``expanded_tokens = T*topk*ep`` (global), and the floor charges only the
    minimum guaranteed remote rows while the reference uses the balanced share.
  * A measured fit is used only for ``t_ref`` and only on A5 + UB inside its
    declared domain; A2/A3 and RoCE never see ``a5_*_measured``.

[Timeouts & Locks]
  Pure arithmetic; no HCCL calls, sockets, ranks or device work.

[Side-Effect Policy]
  No network, no subprocess, no I/O; returns a fresh ``OpWork``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .errors import FormulaValidationError
from .expr_ast import ZERO, ceil_div as ceil_div_ast, log2_ceil, max_expr
from .hardware import (
    HardwareProfile, collective_step_alpha_us, cube_tier, dtype_bytes, fabric_fit,
    mc2_fit, mc2_fit_domain,
)
from .primitives import (
    align_up, ceil_div, coalesce_efficiency, row_bytes_expr, strict_bool,
    strict_finite_float, strict_int, strict_int_list, tensor_bytes,
    wave_efficiency,
)
from .work import Algebra, ByteAccount, OpWork

_COLLECTIVES = {
    "all_reduce": "all_reduce", "hccl_all_reduce": "all_reduce",
    "all_gather": "all_gather", "hccl_all_gather": "all_gather",
    "reduce_scatter": "reduce_scatter", "hccl_reduce_scatter": "reduce_scatter",
    "all_to_all": "all_to_all_single", "all_to_all_v": "all_to_all_v",
    "all_to_all_single": "all_to_all_single",
    "hccl_all_to_all": "all_to_all_single", "broadcast": "broadcast",
}


def _wire_factor(op: str, world_size: int) -> float:
    p = max(1, world_size)
    if p == 1:
        return 0.0
    if op == "all_reduce":
        return 2.0 * (p - 1) / p
    if op in {"all_gather", "reduce_scatter", "all_to_all_single"}:
        return (p - 1) / p
    if op == "broadcast":
        return 1.0
    return 1.0


def _steps_for(op: str, algo: str, p: int) -> int:
    """Declared per-algorithm synchronization step count."""
    if p <= 1:
        return 0
    if algo == "mesh":
        return 2 if op == "all_reduce" else 1
    if algo == "hd":
        bits = int(log2_ceil(max(2, p)).value({}))
        return 2 * bits if op == "all_reduce" else bits
    return 2 * (p - 1) if op == "all_reduce" else (p - 1)


def collective_work(*, op: str, size_bytes: int, world_size: int,
                    algorithm: str = "ring", dtype: str = "bf16",
                    transport: str | None = None,
                    send_counts: Sequence[int] | None = None,
                    recv_counts: Sequence[int] | None = None,
                    self_rank: int = 0,
                    hardware: HardwareProfile) -> OpWork:
    """Build one HCCL collective work record with a qualified A5 fit."""
    key = str(op).strip().lower().replace("-", "_")
    canonical = _COLLECTIVES.get(key)
    if canonical is None:
        raise FormulaValidationError(f"unknown collective op {op!r}")
    size = strict_int(size_bytes, "size_bytes", minimum=1)
    p = strict_int(world_size, "world_size", minimum=1)
    rank = strict_int(self_rank, "self_rank", minimum=0)
    if rank >= p:
        raise FormulaValidationError(f"self_rank {rank} must be < world_size {p}")
    algo = str(algorithm or "ring").strip().lower()
    if algo in {"halving_doubling", "halvingdoubling"}:
        algo = "hd"
    if algo in {"tree"}:
        algo = "hd"
    if algo == "auto":
        algo = "ring"
    if algo not in {"ring", "hd", "mesh"}:
        raise FormulaValidationError(
            "algorithm must be 'ring', 'hd' (tree) or 'mesh'")
    if algo == "hd" and p > 1 and (p & (p - 1)):
        raise FormulaValidationError("hd requires a power-of-two world_size")
    elem_bytes = dtype_bytes(dtype)
    split_read_expr = None
    split_write_expr = None
    if canonical == "all_to_all_v":
        if send_counts is None or recv_counts is None:
            raise FormulaValidationError(
                "AllToAllV requires send_counts and recv_counts")
        send = strict_int_list(send_counts, "send_counts", length=p)
        recv = strict_int_list(recv_counts, "recv_counts", length=p)
        if any(value < 0 for value in send) or any(value < 0 for value in recv):
            raise FormulaValidationError(
                "AllToAllV split counts must be non-negative")
        wire_elems = max(sum(send) - send[rank], sum(recv) - recv[rank])
        wire = int(wire_elems * elem_bytes)
        # Split vectors are the authoritative whole-buffer sizes.
        split_read_expr = sum(send) * elem_bytes
        split_write_expr = sum(recv) * elem_bytes
        factor = (wire / float(split_read_expr)
                  if split_read_expr else 0.0)
        factor_note = "AllToAllV split-vector wire = max(send-self, recv-self)"
    else:
        factor = _wire_factor(canonical, p)
        wire = 0 if p == 1 else int(round(factor * size))
        factor_note = f"{canonical} wire factor {factor:.4f} with p={p}"
    steps = _steps_for(canonical, algo, p)
    base_steps = _steps_for(canonical, "ring", p)
    step_alpha = collective_step_alpha_us(
        hardware.transport_label(p, transport))
    link = hardware.link_bps(p, transport)
    label = hardware.transport_label(p, transport)
    fit = fabric_fit(canonical, p, size, hardware, transport)
    alpha = (0.0 if fit is None else float(fit[0])) + steps * step_alpha
    granule = hardware.granule_for("ND")
    al = Algebra()
    B = al.sym("bytes", size)
    P = al.sym("world_size", p)
    WIRE = al.sym("wire_bytes", wire)
    WIRE_FACTOR = al.real("wire_factor", round(float(factor), 6))
    LINK = al.real("link_bps", link)
    STEPS = al.sym("alpha_steps", int(steps))
    account = ByteAccount(granule)
    if split_read_expr is not None:
        account.read_buffer(split_read_expr, 1.0)
        account.write_buffer(split_write_expr, 1.0)
    else:
        read_expr = B if canonical != "all_gather" else ceil_div_ast(B, P)
        write_expr = B if canonical != "reduce_scatter" else ceil_div_ast(B, P)
        account.read_buffer(read_expr, 1.0)
        account.write_buffer(write_expr, 1.0)
    n_tiles = max(1, p)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    bindings: dict[str, Any] = {
        **al.merged(), "op": canonical, "algorithm": algo,
        "transport": label, "topology": label, "ep": 1, "p": p,
        "send_counts": list(send_counts) if send_counts is not None else [],
        "recv_counts": list(recv_counts) if recv_counts is not None else [],
        "alpha_us": alpha, "step_alpha_us": step_alpha,
        "alpha_steps": int(steps), "base_alpha_steps": int(base_steps),
        "n_tiles": n_tiles, "n_waves": waves,
        "n_cores": hardware.cores, "wave_efficiency": wave_eff,
        "halving_doubling_words": (1 << (p - 1).bit_length()) if p > 1 else 1,
        "wire_factor": round(float(factor), 6),
        "fit_domain_status": ("in_calibrated_ref_domain" if fit is not None
                              else "analytical_alpha_beta"),
        "dtype": str(dtype),
    }
    if split_read_expr is not None:
        bindings["send_bytes"] = int(split_read_expr)
        bindings["recv_bytes"] = int(split_write_expr)
    if fit is not None:
        bindings["fit_beta_ns_per_byte"] = float(fit[1])
        bindings["fit_source"] = "a5_fabric_measured"
        # Explicit algorithm deltas shift the fitted alpha by step count.
        alpha = float(fit[0]) + (steps - base_steps) * step_alpha
        bindings["alpha_us"] = alpha
    notes = (factor_note,
             "measured A5 UB fit used only for t_ref, never t_floor",
             f"transport={label}",
             "world_size=1: no inter-rank wire traffic" if p == 1 else
             f"{algo} alpha steps={steps} at {step_alpha:.3f} us/step")
    return OpWork(
        family="collective", variant=f"{canonical}_{algo}",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=ZERO,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        comm_bytes=WIRE, comm_ref_bytes=WIRE,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=1.0, m_tiled=1, n_tiled=1, k_tiled=1,
        tile_m=16, tile_n=16, tile_k=16,
        comm_bw_bps=link, comm_alpha_us=alpha,
        comm_measured_fit=fit,
        structure="comm_only",
        notes=notes, bindings=bindings,
    )


def mc2_ownership(t: int, ep: int, experts: int, local_experts: int, k: int,
                  stage: str, routed: int | None, remote: int | None,
                  sent: int | None, received: int | None, local: int | None
                  ) -> tuple[int, int, bool, int]:
    """Local routes L: dispatch sends T*K-L, receives R-L; combine reverses.

    ``remote`` counts the expert-side remote rows R-L in either stage.
    Bounds apply independently to this rank's source and all other sources.
    Shared by the native ABI and formula after validating integer dimensions.
    """
    if stage not in {"dispatch", "combine"}:
        raise FormulaValidationError("MC2 stage must be dispatch or combine")
    if local_experts > experts:
        raise FormulaValidationError("experts_local cannot exceed experts_total")
    if k > experts:
        raise FormulaValidationError(f"topk={k} cannot exceed experts_total={experts}")
    lo, hi = t * max(0, k - (experts - local_experts)), t * min(k, local_experts)
    rows = (min(ep * hi, max(ep * lo, t * k)) if routed is None
            else strict_int(routed, "routed_rows", minimum=0))
    if not ep * lo <= rows <= ep * hi:
        raise FormulaValidationError(f"routed_rows={rows} outside [{ep*lo}, {ep*hi}]")
    local_min, local_max = max(lo, rows - (ep - 1) * hi), min(hi, rows - (ep - 1) * lo)
    candidates = []
    for value, label, total in ((local, "local_rows", None),
                                (remote, "remote_routed_rows", rows),
                                (sent, "sent_rows", t*k if stage == "dispatch" else rows),
                                (received, "received_rows", rows if stage == "dispatch" else t*k)):
        if value is not None:
            count = strict_int(value, label, minimum=0)
            candidates.append(count if total is None else total - count)
    if candidates and (len(set(candidates)) != 1
                       or not local_min <= candidates[0] <= local_max):
        raise FormulaValidationError("MC2 sent/received/local/remote ownership is inconsistent")
    if ep == 1 and candidates and candidates[0] != t * k:
        raise FormulaValidationError("ep=1 cannot send or receive remote routes")
    exact = bool(candidates) or ep == 1
    remote_rows = rows - (candidates[0] if candidates else local_max)
    balanced = min(rows - local_min, max(rows - local_max, round(rows * (ep-1) / ep)))
    return rows, remote_rows, exact, balanced


def mc2_work(*, tokens: int, hidden: int, experts_total: int,
             experts_local: int, topk: int, ep: int,
             stage: str = "dispatch", dtype: str = "bf16",
             has_scales: bool = True,
             quant: str | None = None, use_measured_fit: bool = True,
             transport: str | None = None,
             routed_rows: int | None = None,
             remote_routed_rows: int | None = None,
             sent_rows: int | None = None, received_rows: int | None = None,
             local_rows: int | None = None,
             skew: float = 1.0,
             tokens_per_rank_scope: str = "per_rank",
             hardware: HardwareProfile) -> OpWork:
    """MoeDistributeDispatch / MoeDistributeCombine (MC2) work record."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    e_i = strict_int(experts_total, "experts_total", minimum=1)
    el_i = strict_int(experts_local, "experts_local", minimum=0)
    k_i = strict_int(topk, "topk", minimum=1)
    p = strict_int(ep, "ep", minimum=1)
    use_fit = strict_bool(use_measured_fit, "use_measured_fit")
    scales = strict_bool(has_scales, "has_scales")
    if stage == "combine" and not scales:
        raise FormulaValidationError("MC2 combine requires expert scales")
    skew_f = strict_finite_float(skew, "skew", minimum=0.0)
    if quant not in (None, ""):
        cube_tier(dtype=dtype, quant=str(quant))
    if tokens_per_rank_scope != "per_rank":
        raise FormulaValidationError(
            "MC2 token scope is per_rank; convert global tokens first")
    rows, remote, exact_remote, balanced_remote = mc2_ownership(
        t_i, p, e_i, el_i, k_i, stage, routed_rows, remote_routed_rows,
        sent_rows, received_rows, local_rows)
    bpe_i = dtype_bytes(dtype)
    row_bytes = h_i * bpe_i
    # Full-duplex communication must cover the larger send/receive direction.
    source_excess = max(0, t_i * k_i - rows)
    if p == 1:
        wire_floor = wire_ref = 0
    elif exact_remote:
        # Exact locality disables the balanced statistical wire expectation.
        wire_floor = wire_ref = int((remote + source_excess) * row_bytes)
    else:
        wire_floor = int((remote + source_excess) * row_bytes)
        wire_ref = int((balanced_remote + source_excess) * row_bytes)
    expanded = t_i * k_i * p
    link = hardware.link_bps(p, transport)
    label = hardware.transport_label(p, transport)
    fit = (mc2_fit(p, stage, hardware, transport, tokens_per_rank=t_i,
                   skew=skew_f, hidden=h_i, experts_total=e_i, topk=k_i,
                   dtype=dtype, quant=quant)
           if use_fit and p > 1 and not exact_remote else None)
    domain = mc2_fit_domain(t_i, skew_f, hardware, transport, hidden=h_i,
                            experts_total=e_i, topk=k_i, dtype=dtype,
                            quant=quant)
    alpha = 0.0 if fit is None else float(fit[0])
    granule = hardware.granule_for("FRACTAL_NZ")
    al = Algebra()
    T = al.sym("tokens", t_i)
    H = al.sym("hidden", h_i)
    K = al.sym("topk", k_i)
    EP = al.sym("ep", p)
    E = al.sym("experts_total", e_i)
    ROWS = al.sym("routed_rows", rows)
    REMOTE = al.sym("remote_routed_rows", remote)
    BPE = al.real("dtype_bytes", bpe_i)
    LINK = al.real("link_bps", link)
    local_rows = rows
    account = ByteAccount(granule)
    if stage == "dispatch":
        account.read_buffer(T * H * BPE, 1.0, rows=T, width=row_bytes_expr(H, bpe_i))
        account.read_buffer(T * K * 4, 1.0, rows=T, width=K * 4)
        if scales:
            account.read_buffer(T * K * 4, 1.0, rows=T, width=K * 4)
        account.write_buffer(local_rows * H * BPE, 1.0, rows=local_rows,
                             width=row_bytes_expr(H, bpe_i))
        account.write_buffer(E * 4, 1.0)
        read_note = "T*H tokens + T*K expert ids + T*K scales"
        write_note = "local expanded rows + per-expert counts"
    else:
        account.read_buffer(local_rows * H * BPE, 1.0, rows=local_rows,
                            width=row_bytes_expr(H, bpe_i))
        account.read_buffer(T * K * 4, 1.0, rows=T, width=K * 4)
        account.write_buffer(T * H * BPE, 1.0, rows=T,
                             width=row_bytes_expr(H, bpe_i))
        read_note = "local expert outputs + T*K weights"
        write_note = "T*H combined tokens"
    macs = ZERO
    tiled = ZERO
    vector = T * K
    n_tiles = max(1, p)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    floor_sym = al.sym("wire_bytes_floor", wire_floor)
    ref_sym = al.sym("wire_bytes_ref", wire_ref)
    bindings: dict[str, Any] = {
        **al.merged(), "expanded_tokens": expanded,
        **({"has_scales": False} if not scales else {}),
        "global_source_tokens": t_i * p,
        **({"local_rows": rows - remote} if exact_remote else {}),
        "remote_routed_rows": remote, "remote_rows_known": exact_remote,
        "balanced_remote_rows": balanced_remote, "skew": skew_f,
        "transport": label, "topology": label,
        "alpha_us": alpha, "n_tiles": n_tiles, "n_waves": waves,
        "n_cores": hardware.cores, "wave_efficiency": wave_eff,
        "layout": "FRACTAL_NZ", "dtype": str(dtype),
        "quant": "" if quant in (None, "") else str(quant),
        "fit_domain_status": domain if p > 1 else "no_wire_ep1",
    }
    if fit is not None:
        bindings["fit_slope_us_per_expanded_token"] = float(fit[1])
        bindings["fit_source"] = "a5_mc2_measured"
    notes = [
        f"expanded_tokens={expanded} (T*K*EP, global), routed_rows={rows} "
        f"(local expert arrivals)",
        "measured MC2 fit used only for t_ref, never t_floor",
        f"transport={label}; fit domain={bindings['fit_domain_status']}",
    ]
    if p == 1:
        notes.append("ep=1: zero inter-rank wire traffic")
    elif not exact_remote:
        notes.append("wire covers max(sent, received); self routes obey distinct-expert bounds")
    return OpWork(
        family="moe_distribute", variant=stage,
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        comm_bytes=floor_sym, comm_ref_bytes=ref_sym,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=1.0, m_tiled=align_up(t_i, 16),
        n_tiled=align_up(h_i, 16), k_tiled=align_up(k_i, 16),
        tile_m=16, tile_n=256, tile_k=128,
        comm_bw_bps=link, comm_alpha_us=alpha, comm_measured_fit=fit,
        structure="comm_only",
        notes=tuple(notes), bindings=bindings,
    )


__all__ = ["collective_work", "mc2_ownership", "mc2_work"]
