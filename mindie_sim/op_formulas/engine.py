"""Dual-bound latency engine, composite semantics and diagnostics.

[Role]
  Turn one typed ``OpWork`` into the certified physical floor and the explainable
  structural reference.  Concurrent pipes are combined with ``max``; the raw
  Vector floor divides lane operations by the declared operations/second rate
  (never a byte rate); the GM floor is a concurrent HBM/L2 bound and never a
  serial sum; wave quantization and burst coalescing are decomposed into
  inspectable factors.  A composite chain is a sum of serialized kernel floors,
  with exact identity for a one-component chain.

[Invariants]
  * ``0 < tau_hw_floor_us <= t_floor_us <= t_ref_us`` for every evaluation,
    including composites.
  * The floor uses raw SKU peaks and never a measured fit or derated bandwidth.
  * ``t_ref_us`` includes one explicit launch term and uses ``max`` over Cube,
    Vector, GM and fabric pipes, never their sum.
  * ``composite_chain([op]).t_floor_us == op.t_floor_us`` exactly; for k >= 2 the
    floor is ``max(k * tau_hw, sum(component device floors))``.
  * Every card is verified (render -> safe parse -> value) before it is returned.
  * Measured values are reported, never clamped; a sub-floor measurement is
    surfaced for diagnostics instead of being hidden.

[Timeouts & Locks]
  Pure arithmetic; no I/O, device access, locks or sleeps.

[Side-Effect Policy]
  Returns fresh immutable records; no mutation of caller params or globals.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .errors import FormulaError, FormulaValidationError
from .expr_ast import ZERO
from .hardware import (
    HardwareProfile, mc2_fit_domain, resolve_sku,
)
from .model import (
    SCHEMA, DeratingBreakdown, FormulaCard, LatencyBounds,
    OperatorFormulaEvaluation,
)
from .primitives import (
    l2_split, strict_bool, strict_finite_float, strict_mapping, strict_tol,
    us_from_work, wave_efficiency,
)
from .registry import build_work, canonical_name
from .work import OpWork

_ETA_BY_ARCH = {
    "A5": (0.90, 0.92, 0.90),
    "A3": (0.86, 0.88, 0.88),
    "A2": (0.84, 0.86, 0.87),
}

_MAX_COMPOSITE_DEPTH = 8

_HOST_KEYS = {
    "flash_attention": "attention",
    "incre_flash_attention": "attention",
    "paged_attention": "attention",
    "fused_infer_attention_score": "attention",
    "mla_prolog": "mla",
    "sparse_flash_mla": "linear_attention",
    "lightning_indexer": "linear_attention",
    "chunk_kda": "linear_attention",
    "chunk_gated_delta_rule": "linear_attention",
    "recurrent_kda": "linear_attention",
    "fused_recurrent_gdn": "linear_attention",
    "kda_recover_ssm": "linear_attention",
    "causal_conv1d": "conv",
    "grouped_matmul": "grouped_matmul",
    "quant_batch_matmul": "quant",
    "dynamic_quant": "quant",
    "moe_gating": "moe_routing",
    "moe_distribute": "moe_distribute",
    "collective": "collective",
    "embedding": "embedding",
    "swiglu": "swiglu",
    "rotary_embedding": "rope",
    "rms_norm": "rms_norm",
    "argmax": "sampler",
    "top_p_sample": "sampler",
}


def _etas(hardware: HardwareProfile) -> tuple[float, float, float]:
    return _ETA_BY_ARCH.get(hardware.arch, _ETA_BY_ARCH["A3"])


def _host_key(family: str) -> str:
    return _HOST_KEYS.get(family, family)


def _card(work: OpWork, hardware: HardwareProfile, launch_mode: str,
          hbm_bytes: int, l2_bytes: int, values: Mapping[str, Any],
          extra_bindings: Mapping[str, Any] | None = None) -> FormulaCard:
    bindings = dict(work.bindings)
    bindings.update({
        "sku": hardware.sku, "launch_mode": launch_mode,
        "n_cores": hardware.cores, "n_tiles": work.n_tiles,
        "hbm_bytes": int(hbm_bytes), "l2_hot_bytes": int(l2_bytes),
        "l2_hot_budget_bytes": hardware.l2_hot_budget(),
        "compulsory_gm_bytes": int(values["compulsory_gm_bytes"]),
        "granule_gm_bytes": int(values["granule_gm_bytes"]),
    })
    bindings.update(dict(extra_bindings or {}))
    card = FormulaCard(
        cube_macs_expr=work.cube_macs.render(),
        cube_macs=float(values["cube_macs"]),
        alg_flops_expr=work.alg_flops.render(),
        alg_flops=float(values["alg_flops"]),
        cube_flops_tiled_expr=work.cube_flops_tiled.render(),
        cube_flops_tiled=float(values["cube_flops_tiled"]),
        vector_ops_expr=work.vector_ops.render(),
        vector_ops=float(values["vector_ops"]),
        compulsory_read_bytes_expr=work.read_bytes.render(),
        compulsory_read_bytes=int(values["read_bytes"]),
        compulsory_write_bytes_expr=work.write_bytes.render(),
        compulsory_write_bytes=int(values["write_bytes"]),
        compulsory_gm_bytes_expr=work.compulsory_bytes.render(),
        compulsory_gm_bytes=int(values["compulsory_gm_bytes"]),
        read_bytes_compulsory=int(values["read_bytes"]),
        write_bytes_compulsory=int(values["write_bytes"]),
        l2_hot_eligible_bytes=int(values["l2_hot_bytes"]),
        granule_read_bytes_expr=work.granule_read_bytes.render(),
        granule_read_bytes=int(values["granule_read_bytes"]),
        granule_write_bytes_expr=work.granule_write_bytes.render(),
        granule_write_bytes=int(values["granule_write_bytes"]),
        granule_gm_bytes_expr=work.granule_bytes.render(),
        granule_gm_bytes=int(values["granule_gm_bytes"]),
        comm_bytes_expr=work.comm_bytes.render(),
        comm_bytes=int(values["comm_bytes"]),
        comm_ref_bytes_expr=work.comm_ref_bytes.render(),
        comm_ref_bytes=int(values["comm_ref_bytes"]),
        bindings=bindings,
        notes=work.notes,
    )
    try:
        card.verify()
    except FormulaValidationError as exc:
        raise FormulaError(f"{work.family}: card invariant failed: {exc}") from exc
    return card


def _evaluate_work(work: OpWork, *, operator: str, family: str, variant: str,
                   hardware: HardwareProfile, launch_mode: str,
                   steady_l2_warm: bool,
                   measured_us: float | None,
                   extra_bindings: Mapping[str, Any] | None = None
                   ) -> OperatorFormulaEvaluation:
    work = work.capped()
    work.self_check()
    values = work.values()
    n_cores = work.n_cores or hardware.cores
    waves, wave_eff = wave_efficiency(work.n_tiles, n_cores)
    coalesce = max(1e-6, min(1.0, float(work.coalesce_efficiency)))
    eta_cube, eta_vec, eta_bw = _etas(hardware)
    eta_cube_geom = eta_cube * wave_eff
    eta_vec_geom = eta_vec * wave_eff
    compulsory = int(values["compulsory_gm_bytes"])
    hbm_bytes, l2_bytes = l2_split(compulsory, int(values["l2_hot_bytes"]))
    bw_wave_eff = (max(wave_eff, min(1.0, 4.0 / float(n_cores)))
                   if compulsory >= 4 * 1024 * 1024 else 1.0)
    eta_bw_geom = eta_bw * bw_wave_eff * coalesce

    tau_hw = hardware.launch_floor(launch_mode)
    # Cube-owned algebra is 2*MACs; an algebra override priced on Vector must
    # never inflate the Cube floor.
    cube_floor = us_from_work(2.0 * values["cube_macs"],
                              hardware.cube_peak(work.cube_tier))
    vector_floor = us_from_work(values["vector_ops"],
                                hardware.vector_ops_per_s)
    # Concurrent cold/warm lower bound: HBM refill and L2 service overlap.
    gm_floor = max(us_from_work(hbm_bytes, hardware.hbm_bps),
                   us_from_work(l2_bytes, hardware.l2_bps))
    comm_floor = (us_from_work(values["comm_bytes"], work.comm_bw_bps)
                  if values["comm_bytes"] > 0 else 0.0)
    t_floor = max(tau_hw, cube_floor, vector_floor, gm_floor, comm_floor)

    host = hardware.host_dispatch(_host_key(family), launch_mode)
    fit_drives_ref = (work.comm_measured_fit is not None
                      and (values["comm_bytes"] > 0
                           or values["comm_ref_bytes"] > 0))
    tau_launch = (max(tau_hw, float(work.extra_host_us)) if fit_drives_ref
                  else max(tau_hw, host, float(work.extra_host_us)))
    t_cube = us_from_work(values["cube_flops_tiled"],
                          eta_cube_geom * hardware.cube_peak(work.cube_tier))
    t_vec = us_from_work(values["vector_ref_ops"],
                         eta_vec_geom * hardware.vector_ops_per_s)
    granule = int(values["granule_gm_bytes"])
    if compulsory > 0:
        ratio_hbm = hbm_bytes / compulsory
        ratio_l2 = l2_bytes / compulsory
        t_bw = (us_from_work(granule * ratio_hbm,
                             eta_bw_geom * hardware.hbm_bps)
                + us_from_work(granule * ratio_l2,
                               eta_bw_geom * hardware.l2_bps))
    elif granule > 0:
        # A zero certified floor with positive structural reference traffic:
        # split the reference bytes directly instead of dividing by zero.
        ref_hbm, ref_l2 = l2_split(granule, int(values["l2_hot_bytes"]))
        t_bw = (us_from_work(ref_hbm, eta_bw_geom * hardware.hbm_bps)
                + us_from_work(ref_l2, eta_bw_geom * hardware.l2_bps))
    else:
        t_bw = 0.0

    t_comm = 0.0
    if (values["comm_bytes"] > 0 or values["comm_ref_bytes"] > 0
            or fit_drives_ref):
        fit = work.comm_measured_fit
        if fit is not None and work.family == "collective":
            # The fabric fit is calibrated per byte of the caller's tensor, not
            # per wire-inflated byte: use the declared size binding.  The
            # algorithm delta is already folded into comm_alpha_us.
            size = float(work.bindings.get("bytes",
                                           values["comm_ref_bytes"] or values["comm_bytes"]))
            t_comm = float(work.comm_alpha_us) + float(fit[1]) * size / 1000.0
        elif fit is not None and work.family == "moe_distribute":
            expanded = float(work.bindings.get("expanded_tokens", 1))
            t_comm = float(fit[0]) + float(fit[1]) * expanded
        else:
            link_eta = eta_bw * max(1e-6, wave_eff)
            t_comm = float(work.comm_alpha_us) + us_from_work(
                values["comm_ref_bytes"], link_eta * work.comm_bw_bps)
        t_comm = max(comm_floor, t_comm)

    t_ref_device = max(t_cube, t_vec, t_bw, t_comm) + float(work.tau_epilogue_us)
    if fit_drives_ref:
        t_ref = max(t_floor, t_ref_device)
    else:
        t_ref = max(t_floor, tau_launch + t_ref_device)
    notes = list(work.notes)
    if tau_launch + t_ref_device < t_floor:
        notes.append("reference raised to the certified floor")

    pipes = {"launch": tau_launch, "cube": t_cube, "vector": t_vec,
             "hbm": t_bw, "fabric": t_comm}
    bottleneck = max(pipes, key=lambda key: pipes[key])
    if bottleneck == "hbm" and l2_bytes > hbm_bytes:
        bottleneck = "l2"
    latency = LatencyBounds(
        tau_hw_floor_us=tau_hw, tau_launch_us=tau_launch,
        tau_epilogue_us=float(work.tau_epilogue_us),
        cube_floor_us=cube_floor, vector_floor_us=vector_floor,
        gm_floor_us=gm_floor, comm_floor_us=comm_floor, t_floor_us=t_floor,
        cube_ref_us=t_cube, vector_ref_us=t_vec, bw_ref_us=t_bw,
        comm_ref_us=t_comm, t_ref_us=t_ref, bottleneck_pipe=bottleneck,
    )
    derating = DeratingBreakdown(
        eta_asymp_cube=eta_cube, eta_asymp_vector=eta_vec, eta_asymp_bw=eta_bw,
        n_tiles=work.n_tiles, n_cores=n_cores, n_waves=waves,
        wave_efficiency=wave_eff, bw_wave_efficiency=bw_wave_eff,
        coalesce_efficiency=coalesce,
        cube_geom_efficiency=eta_cube_geom, vector_geom_efficiency=eta_vec_geom,
        bw_geom_efficiency=eta_bw_geom,
        m_tiled=work.m_tiled, n_tiled=work.n_tiled, k_tiled=work.k_tiled,
        tile_m=work.tile_m, tile_n=work.tile_n, tile_k=work.tile_k,
        hbm_bytes=hbm_bytes, l2_bytes=l2_bytes,
        l2_hot_budget_bytes=hardware.l2_hot_budget(),
        l2_peak_bps=hardware.l2_bps,
        capacity_reload_factor=float(work.capacity_reload_factor),
    )
    card = _card(work, hardware, launch_mode, hbm_bytes, l2_bytes, values,
                 extra_bindings)
    measured = None if measured_us is None else float(measured_us)
    sol = 100.0 * t_floor / measured if measured and measured > 0 else None
    ref_ratio = measured / t_ref if measured and measured > 0 else None
    gap = (t_ref - t_floor) if measured is not None else None
    headroom = (max(0.0, measured - t_ref)
                if measured is not None else None)
    return OperatorFormulaEvaluation(
        operator=operator, variant=variant, family=family,
        sku=hardware.sku, arch=hardware.arch, launch_mode=launch_mode,
        steady_l2_warm=bool(steady_l2_warm), formula_card=card,
        derating=derating, latency=latency,
        assumptions=tuple(notes), measured_us=measured,
        sol_floor_pct=sol, ref_ratio=ref_ratio,
        structural_gap_us=gap, impl_headroom_us=headroom,
    )


def _evaluate_leaf(operator: str, params: Mapping[str, Any], *,
                   sku: str, launch_mode: str, steady_l2_warm: bool,
                   measured_us: float | None
                   ) -> tuple[OpWork, OperatorFormulaEvaluation]:
    canonical = canonical_name(operator)
    if canonical is None:
        raise FormulaError(f"unsupported operator {operator!r}")
    if canonical == "composite_chain":
        raise FormulaError("composite_chain cannot be a leaf component")
    hardware = resolve_sku(sku)
    work = build_work(canonical, params, hardware, steady_l2_warm=steady_l2_warm)
    extra = {key: value for key, value in work.bindings.items()
             if key in {"transport", "fit_domain_status", "fit_source",
                        "topology", "ep", "world_size"}}
    evaluation = _evaluate_work(
        work, operator=operator, family=work.family,
        variant=work.variant, hardware=hardware, launch_mode=launch_mode,
        steady_l2_warm=steady_l2_warm, measured_us=measured_us,
        extra_bindings=extra)
    return work, evaluation


def _combine(parts: Sequence[tuple[OpWork, OperatorFormulaEvaluation]],
             operator: str, hardware: HardwareProfile, launch_mode: str,
             steady_l2_warm: bool, measured_us: float | None
             ) -> OperatorFormulaEvaluation:
    if len(parts) == 1:
        child = parts[0][1]
        card = child.formula_card
        measured = None if measured_us is None else float(measured_us)
        floor = child.t_floor_us
        ref = child.t_ref_us
        positive = measured is not None and measured > 0
        return OperatorFormulaEvaluation(
            operator=operator, variant="composite", family="composite",
            sku=hardware.sku, arch=hardware.arch, launch_mode=launch_mode,
            steady_l2_warm=bool(steady_l2_warm), formula_card=card,
            derating=child.derating, latency=child.latency,
            assumptions=child.assumptions + (
                "single-component composite: identical to the child evaluation",),
            measured_us=measured,
            sol_floor_pct=(100.0 * floor / measured) if positive else None,
            ref_ratio=(measured / ref) if positive else None,
            structural_gap_us=((ref - floor) if measured is not None else None),
            impl_headroom_us=(max(0.0, measured - ref)
                              if measured is not None else None))
    tau_hw = hardware.launch_floor(launch_mode)
    works = [item[0].prefixed(f"c{index}_") for index, item in enumerate(parts)]
    evals = [item[1] for item in parts]
    bindings: dict[str, Any] = {"components": len(parts), "sku": hardware.sku}
    for index, work in enumerate(works):
        bindings[f"c{index}_operator"] = str(evals[index].operator)
        for key, value in work.bindings.items():
            bindings[key] = value
    cube_macs = sum((item.cube_macs for item in works), ZERO)
    tiled = sum((item.cube_flops_tiled for item in works), ZERO)
    vector = sum((item.vector_ops for item in works), ZERO)
    read = sum((item.read_bytes for item in works), ZERO)
    write = sum((item.write_bytes for item in works), ZERO)
    g_read = sum((item.granule_read_bytes for item in works), ZERO)
    g_write = sum((item.granule_write_bytes for item in works), ZERO)
    comm = sum((item.comm_bytes for item in works), ZERO)
    comm_ref = sum((item.comm_ref_bytes for item in works), ZERO)
    l2 = sum((item.l2_hot_bytes for item in works), ZERO)
    composite = OpWork(
        family="composite", variant="composite",
        cube_macs=cube_macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=read, write_bytes=write,
        granule_read_bytes=g_read, granule_write_bytes=g_write,
        comm_bytes=comm, comm_ref_bytes=comm_ref, l2_hot_bytes=l2,
        n_tiles=max(item.n_tiles for item in works),
        n_cores=hardware.cores,
        coalesce_efficiency=min(item.coalesce_efficiency for item in works),
        m_tiled=max(item.m_tiled for item in works),
        n_tiled=max(item.n_tiled for item in works),
        k_tiled=max(item.k_tiled for item in works),
        tile_m=max(item.tile_m for item in works),
        tile_n=max(item.tile_n for item in works),
        tile_k=max(item.tile_k for item in works),
        cube_tier=works[0].cube_tier,
        structure="composite",
        tau_epilogue_us=sum(item.tau_epilogue_us for item in works),
        notes=tuple(note for item in works for note in item.notes) + (
            f"composite chain: {len(parts)} serialized kernels",
            "single-component identity: no extra dispatch outside max()",
        ),
        bindings=bindings,
    )
    composite = composite.capped()
    composite.self_check()
    values = composite.values()
    n_cores = hardware.cores
    waves, wave_eff = wave_efficiency(composite.n_tiles, n_cores)
    coalesce = max(1e-6, min(1.0, composite.coalesce_efficiency))
    eta_cube, eta_vec, eta_bw = _etas(hardware)
    compulsory = int(values["compulsory_gm_bytes"])
    hbm_bytes, l2_bytes = l2_split(compulsory, int(values["l2_hot_bytes"]))
    bw_wave_eff = (max(wave_eff, min(1.0, 4.0 / float(n_cores)))
                   if compulsory >= 4 * 1024 * 1024 else 1.0)
    t_floor = max(float(tau_hw) * len(parts),
                  sum(item.t_floor_us for item in evals))
    total_ref = sum(item.t_ref_us for item in evals)
    t_ref = max(t_floor, total_ref)
    card = _card(composite, hardware, launch_mode, hbm_bytes, l2_bytes, values,
                 {"components": len(parts)})
    derating = DeratingBreakdown(
        eta_asymp_cube=eta_cube, eta_asymp_vector=eta_vec, eta_asymp_bw=eta_bw,
        n_tiles=composite.n_tiles, n_cores=n_cores, n_waves=waves,
        wave_efficiency=wave_eff, bw_wave_efficiency=bw_wave_eff,
        coalesce_efficiency=coalesce,
        cube_geom_efficiency=eta_cube * wave_eff,
        vector_geom_efficiency=eta_vec * wave_eff,
        bw_geom_efficiency=eta_bw * bw_wave_eff * coalesce,
        m_tiled=composite.m_tiled, n_tiled=composite.n_tiled,
        k_tiled=composite.k_tiled, tile_m=composite.tile_m,
        tile_n=composite.tile_n, tile_k=composite.tile_k,
        hbm_bytes=hbm_bytes, l2_bytes=l2_bytes,
        l2_hot_budget_bytes=hardware.l2_hot_budget(),
        l2_peak_bps=hardware.l2_bps,
        capacity_reload_factor=max(item.derating.capacity_reload_factor
                                   for item in evals),
    )
    pipes = {
        "launch": max(item.latency.tau_launch_us for item in evals),
        "cube": sum(item.latency.cube_ref_us for item in evals),
        "vector": sum(item.latency.vector_ref_us for item in evals),
        "hbm": sum(item.latency.bw_ref_us for item in evals),
        "fabric": sum(item.latency.comm_ref_us for item in evals),
    }
    bottleneck = max(pipes, key=lambda key: pipes[key])
    latency = LatencyBounds(
        tau_hw_floor_us=tau_hw,
        tau_launch_us=max(item.latency.tau_launch_us for item in evals),
        tau_epilogue_us=sum(item.latency.tau_epilogue_us for item in evals),
        cube_floor_us=sum(item.latency.cube_floor_us for item in evals),
        vector_floor_us=sum(item.latency.vector_floor_us for item in evals),
        gm_floor_us=sum(item.latency.gm_floor_us for item in evals),
        comm_floor_us=sum(item.latency.comm_floor_us for item in evals),
        t_floor_us=t_floor,
        cube_ref_us=sum(item.latency.cube_ref_us for item in evals),
        vector_ref_us=sum(item.latency.vector_ref_us for item in evals),
        bw_ref_us=sum(item.latency.bw_ref_us for item in evals),
        comm_ref_us=sum(item.latency.comm_ref_us for item in evals),
        t_ref_us=t_ref, bottleneck_pipe=bottleneck,
    )
    measured = None if measured_us is None else float(measured_us)
    sol = 100.0 * t_floor / measured if measured and measured > 0 else None
    ref_ratio = measured / t_ref if measured and measured > 0 else None
    return OperatorFormulaEvaluation(
        operator=operator, variant="composite", family="composite",
        sku=hardware.sku, arch=hardware.arch, launch_mode=launch_mode,
        steady_l2_warm=bool(steady_l2_warm), formula_card=card,
        derating=derating, latency=latency,
        assumptions=(
            f"composite of {len(evals)} serialized kernels",
            "each component keeps its own launch and pipe-max reference",
        ), measured_us=measured, sol_floor_pct=sol, ref_ratio=ref_ratio,
        structural_gap_us=(t_ref - t_floor) if measured is not None else None,
        impl_headroom_us=(max(0.0, measured - t_ref) if measured is not None else None),
    )


def _composite_evaluation(operator: str, params: Mapping[str, Any], *,
                          sku: str, launch_mode: str, steady_l2_warm: bool,
                          measured_us: float | None, depth: int
                          ) -> OperatorFormulaEvaluation:
    if depth > _MAX_COMPOSITE_DEPTH:
        raise FormulaError("composite_chain nesting exceeds 8 levels")
    components = params.get("components")
    if (not isinstance(components, Sequence) or isinstance(components, (str, bytes))
            or not components):
        raise FormulaError("composite_chain requires a non-empty components list")
    parts: list[tuple[OpWork, OperatorFormulaEvaluation]] = []
    for index, item in enumerate(components):
        if not isinstance(item, Mapping):
            raise FormulaError(f"composite component {index} must be a mapping")
        child_name = item.get("operator") or item.get("op")
        if not child_name:
            raise FormulaError(f"composite component {index} is missing operator")
        child_params = (strict_mapping(item["params"],
                                       f"composite component {index}.params")
                        if "params" in item else {})
        if child_params.get("components"):
            raise FormulaError("composite components cannot nest another composite")
        child_warm = (strict_bool(item["steady_l2_warm"],
                                  f"composite component {index}.steady_l2_warm")
                      if "steady_l2_warm" in item else steady_l2_warm)
        child_mode = (str(item["launch_mode"]) if "launch_mode" in item
                      else launch_mode)
        if child_mode not in {"superkernel_subtask", "aclgraph", "eager",
                              "python_eager"}:
            raise FormulaValidationError(
                f"composite component {index}: unknown launch_mode {child_mode!r}")
        parts.append(_evaluate_leaf(
            str(child_name), child_params, sku=sku,
            launch_mode=child_mode,
            steady_l2_warm=child_warm,
            measured_us=None))
    return _combine(parts, operator, resolve_sku(sku), launch_mode,
                    steady_l2_warm, measured_us)


def _evaluate_request(operator: str, params: Mapping[str, Any], *,
                      sku: str, launch_mode: str, steady_l2_warm: bool,
                      measured_us: float | None,
                      depth: int = 0) -> OperatorFormulaEvaluation:
    canonical = canonical_name(operator)
    if canonical is None:
        raise FormulaError(f"unsupported operator {operator!r}")
    if canonical == "composite_chain":
        return _composite_evaluation(
            operator=operator, params=params, sku=sku, launch_mode=launch_mode,
            steady_l2_warm=steady_l2_warm, measured_us=measured_us, depth=depth)
    _work, evaluation = _evaluate_leaf(
        operator, params, sku=sku, launch_mode=launch_mode,
        steady_l2_warm=steady_l2_warm, measured_us=measured_us)
    return evaluation


def evaluate_operator(operator: str, params: Mapping[str, Any] | None = None, *,
                      sku: str = "Ascend950DT_9581",
                      launch_mode: str = "eager",
                      steady_l2_warm: bool = False,
                      measured_us: float | None = None
                      ) -> OperatorFormulaEvaluation:
    """Evaluate one operator into an immutable dual-bound formula record."""
    if launch_mode not in {"superkernel_subtask", "aclgraph", "eager", "python_eager"}:
        raise FormulaValidationError(f"unknown launch_mode {launch_mode!r}")
    measured = None if measured_us is None else strict_finite_float(
        measured_us, "measured_us", minimum=0.0)
    warm = strict_bool(steady_l2_warm, "steady_l2_warm")
    if params is None:
        params = {}
    body = dict(strict_mapping(params, "params"))
    return _evaluate_request(operator, body, sku=sku,
                             launch_mode=launch_mode,
                             steady_l2_warm=warm,
                             measured_us=measured)


def evaluate_with_diagnostics(operator: str, params: Mapping[str, Any] | None = None,
                              *, sku: str = "Ascend950DT_9581",
                              launch_mode: str = "eager",
                              steady_l2_warm: bool = False,
                              measured_us: float | None = None,
                              tol: float = 0.02,
                              flags: Mapping[str, Any] | None = None
                              ) -> tuple[OperatorFormulaEvaluation, dict[str, Any]]:
    """Evaluate and classify the measured row without clamping the bound."""
    evaluation = evaluate_operator(
        operator, params, sku=sku, launch_mode=launch_mode,
        steady_l2_warm=steady_l2_warm, measured_us=measured_us)
    return evaluation, diagnose(evaluation, measured_us, tol=tol, flags=flags)


def _candidates(evaluation: OperatorFormulaEvaluation, flags: Mapping[str, Any],
                measured_us: float | None) -> list[str]:
    out: list[str] = []
    if flags.get("wrong_sku_spec"):
        out.append("wrong_sku_spec")
    if flags.get("unfiltered_padding_shape_bound"):
        out.append("unfiltered_padding_shape_bound")
    if flags.get("async_overlap_shadowed"):
        out.append("async_overlap_shadowed")
    if not evaluation.steady_l2_warm and evaluation.formula_card.l2_hot_eligible_bytes > 0:
        out.append("cross_op_l2_hit")
    if not evaluation.steady_l2_warm and not out:
        out.append("cross_op_l2_hit")
    if not out:
        out.append("async_overlap_shadowed")
    return out


def diagnose(evaluation: OperatorFormulaEvaluation,
             measured_us: float | None, *, tol: float = 0.02,
             flags: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the explicit clean/violation classification for one row."""
    flag_map = dict(flags or {})
    tolerance = strict_tol(tol)
    if measured_us is None:
        return {
            "schema": SCHEMA + "/diagnostic",
            "status": "unmeasured",
            "measured_us": None,
            "bound_violation_warning": False,
            "root_cause_candidates": [],
            "structural_gap_us": None,
            "impl_headroom_us": None,
        }
    measured = strict_finite_float(measured_us, "measured_us", minimum=0.0)
    if measured <= 0:
        return {
            "schema": SCHEMA + "/diagnostic",
            "status": "unmeasured",
            "measured_us": measured,
            "bound_violation_warning": False,
            "root_cause_candidates": [],
            "structural_gap_us": None,
            "impl_headroom_us": None,
        }
    floor = evaluation.latency.t_floor_us
    ref = evaluation.latency.t_ref_us
    sol = 100.0 * floor / measured
    ratio = measured / ref
    violation = measured < floor * (1.0 - tolerance)
    return {
        "schema": SCHEMA + "/diagnostic",
        "status": "bound_violation" if violation else "clean",
        "measured_us": measured,
        "t_floor_us": floor,
        "t_ref_us": ref,
        "sol_floor_pct": sol,
        "ref_ratio": ratio,
        "structural_gap_us": ref - floor,
        "impl_headroom_us": max(0.0, measured - ref),
        "bound_violation_warning": bool(violation),
        "root_cause_candidates": _candidates(evaluation, flag_map, measured) if violation else [],
    }


__all__ = [
    "diagnose", "evaluate_operator", "evaluate_with_diagnostics",
]
