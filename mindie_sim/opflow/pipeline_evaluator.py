"""Intra-op pipeline, tiling, recurrence, wave-tail, and launch-floor evaluator.

[Role]
Turn one declarative OpFlowTemplate into a non-double-counted timing
decomposition: critical-chain first item, steady recurrence tau, exact wave
tails, lane-partitioned Vector service, SRAM liveness, and launch floors.
[Invariants]
Steady time is a bottleneck, never a pipe sum. The first item pays its full
stage chain (MTE2->MTE1->CUBE->FIX or MTE2->VEC->MTE3). Single-buffer flows
serialize stage chains across items. tau >= every pipe load and every
recurrence bound with distance >= 1. GM service is capped by the chip bandwidth
and a per-owner port that does not shrink when physical core count grows.
Full tiles and tail tiles are never averaged. Stateless retile pays zero spill;
only algorithm-owned workspace adds traffic and unique live bytes.
[Timeouts & Locks]
Bounded in-memory arithmetic; no sleeps, locks, I/O, or subprocesses.
[Side-Effect Policy]
Pure evaluator. No filesystem or device effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .flow_schema import (
    AuthoringBackend, BufferPool, BufferSpec, ExecutionArchetype, ExecutionMode,
    FixpipePath, FlowSchemaError, OpFlowTemplate, VectorMode, buffer_capacity,
    check_flow_fixpipe, require_int,
)
from .hardware_profiles import HardwareProfile, ProfileError
from .schedule_math import pipeline_chain, pipeline_tail
from .schedule_eval import (
    _vector_lanes, _pipe_tile_us, _gm_bytes, _gm_tile_us, _stage_services,
    _first_item_chain, _sram_plans, _adapt_triton, owner_schedule, _wave_services,
    _merge_triton_features,
)

from .tiling import TilingError, TilingPlan, derive_tiling
from .transfer_accounting import plan_buffer_pool
from .triton_lowering import (
    TritonFeatures, TritonLoweringError, TritonLoweringResult,
    analyze_triton_lowering,
)

LAUNCH_FLOORS_US: Mapping[ExecutionMode, float] = {
    ExecutionMode.EAGER: 5.0,
    ExecutionMode.ACLGRAPH_PIECEWISE: 1.6,
    ExecutionMode.ACLGRAPH_FULL: 1.35,
    ExecutionMode.ACLGRAPH_SUPERKERNEL: 0.32,
}
LAUNCH_FLOORS = {mode.value: value for mode, value in LAUNCH_FLOORS_US.items()}


class ExecutionModeError(ValueError):
    """Execution mode is invalid for the requested phase."""


def _as_mode(mode: ExecutionMode | str) -> ExecutionMode:
    try:
        return mode if isinstance(mode, ExecutionMode) else ExecutionMode(str(mode))
    except ValueError as exc:
        raise ExecutionModeError(f"unknown execution mode: {mode!r}") from exc


def mode_launch_floor_us(mode: ExecutionMode | str, *, static_decode: bool = True,
                         superkernel_eligible: bool = True) -> float:
    for name, value in (("static_decode", static_decode), ("superkernel_eligible", superkernel_eligible)):
        if not isinstance(value, bool):
            raise ExecutionModeError(f"{name} must be bool")
    chosen = _as_mode(mode)
    if chosen is ExecutionMode.ACLGRAPH_SUPERKERNEL:
        if not static_decode:
            raise ExecutionModeError(
                "aclgraph_superkernel is valid only on static-shape decode; "
                "dynamic prefill must use aclgraph_piecewise/full"
            )
        if not superkernel_eligible:
            raise ExecutionModeError("aclgraph_superkernel requires capturable static ops")
    return LAUNCH_FLOORS_US[chosen]


def effective_gm_bandwidth(profile: HardwareProfile, working_set_bytes: int) -> float:
    """L2/HBM interpolation on the unique live set only (<L2 L2, >2L2 HBM)."""
    if isinstance(working_set_bytes, bool) or not isinstance(working_set_bytes, int):
        raise ValueError("working set must be an integer")
    if working_set_bytes < 0:
        raise ValueError("working set must be non-negative")
    l2 = float(profile.memory.l2)
    low = float(profile.rates.l2_bytes_s)
    high = float(profile.rates.hbm_bytes_s)
    if working_set_bytes <= l2:
        return low
    if working_set_bytes >= 2.0 * l2:
        return high
    frac = (working_set_bytes - l2) / l2
    return max(high, low + (high - low) * frac)


@dataclass(frozen=True, slots=True)
class BubbleLedger:
    fill_us: float
    steady_us: float
    drain_us: float
    tail_wave_quantization_us: float
    intra_group_1c2v_sync_us: float

    @property
    def wall_components_us(self) -> float:
        return self.fill_us + self.steady_us + self.drain_us + self.intra_group_1c2v_sync_us


@dataclass(frozen=True, slots=True)
class FlowEvaluation:
    name: str
    archetype: ExecutionArchetype
    mode: ExecutionMode
    profile: str
    tiling: TilingPlan
    active_units: int
    waves: int
    per_tile_us: float
    pipe_us: Mapping[str, float]
    gm_us: float
    launch_floor_us: float
    t_pipeline_us: float
    t_est_us: float
    bubble: BubbleLedger
    effective_gm_bytes_s: float
    working_set_bytes: int
    l2_resident: bool
    fixpipe_path: str
    fixpipe_lane: int | None
    warnings: tuple[str, ...] = ()
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    authoring_backend: str = "ascendc"
    tau_steady_us: float = 0.0
    recurrence_interval_us: float = 0.0
    t_unit_us: float = 0.0
    t_setup_us: float = 0.0
    t_first_item_us: float = 0.0
    t_drain_finish_us: float = 0.0
    max_pipe_load_us: float = 0.0
    launched: bool = True
    estimate_kind: str = "analytic_prior"
    calibrated: bool = False

    def __post_init__(self) -> None:
        for name in ("l2_resident", "launched", "calibrated"):
            if not isinstance(getattr(self, name), bool):
                raise FlowSchemaError(f"{name} must be bool")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "op": self.name, "arch": self.archetype.value, "mode": self.mode.value,
            "profile": self.profile, "backend": self.authoring_backend,
            "t_est_us": round(self.t_est_us, 6),
            "t_pipeline_us": round(self.t_pipeline_us, 6),
            "t_unit_us": round(self.t_unit_us, 6),
            "t_first_item_us": round(self.t_first_item_us, 6),
            "max_pipe_load_us": round(self.max_pipe_load_us, 6),
            "tau_steady_us": round(self.tau_steady_us, 6),
            "recurrence_us": round(self.recurrence_interval_us, 6),
            "launch_floor_us": round(self.launch_floor_us, 6),
            "tiles": self.tiling.tile_count, "passes": self.tiling.passes,
            "waves": self.waves, "active_units": self.active_units,
            "gm_us": round(self.gm_us, 6),
            "pipe_us": {key: round(value, 6) for key, value in self.pipe_us.items()},
            "working_set_bytes": self.working_set_bytes,
            "l2_resident": self.l2_resident, "fixpipe": self.fixpipe_path,
            "launched": self.launched, "est_kind": self.estimate_kind,
            "calibrated": self.calibrated, "warnings": list(self.warnings),
            "triton": dict(self.extra.get("triton", {})),
        }
        return payload


def evaluate_flow(flow: OpFlowTemplate, profile: HardwareProfile,
                  execution_mode: ExecutionMode | str = ExecutionMode.ACLGRAPH_FULL, *,
                  requested_tile_units: float | None = None,
                  double_buffer: bool = True,
                  active_units: int | None = None,
                  static_decode: bool = True,
                  triton_features: Mapping[str, Any] | None = None,
                  capturable_ops: bool = True,
                  dynamic_tiling_host_sync: bool = False,
                  allow_structural_prior: bool = False) -> FlowEvaluation:
    """Evaluate one concrete flow with dependencies, tails, and SRAM liveness."""
    if not isinstance(flow, OpFlowTemplate):
        raise FlowSchemaError("flow must be an OpFlowTemplate")
    if not isinstance(profile, HardwareProfile):
        raise ProfileError("profile must be a HardwareProfile")
    for name, value in (("double_buffer", double_buffer), ("static_decode", static_decode),
                        ("capturable_ops", capturable_ops), ("allow_structural_prior", allow_structural_prior),
                        ("dynamic_tiling_host_sync", dynamic_tiling_host_sync)):
        if not isinstance(value, bool):
            raise FlowSchemaError(f"{name} must be bool")
    triton_features = _merge_triton_features(flow.triton_features, triton_features)
    mode = _as_mode(execution_mode)
    profile.require_prediction_ready()
    if flow.estimate_kind in ("structural_schema_only", "conditional_ledger_required") and not allow_structural_prior:
        from .paradigm_catalog import UncalibratedOperatorError
        raise UncalibratedOperatorError(
            f"{flow.name!r} is {flow.estimate_kind}; pass "
            "allow_structural_prior=True to obtain an explicitly uncalibrated prior"
        )
    eligible = capturable_ops and not dynamic_tiling_host_sync and static_decode
    floor = mode_launch_floor_us(mode, static_decode=static_decode, superkernel_eligible=eligible)
    fixpipe = check_flow_fixpipe(flow, profile)
    if flow.no_launch:
        zero = BubbleLedger(0.0, 0.0, 0.0, 0.0, 0.0)
        return FlowEvaluation(
            name=flow.name, archetype=flow.archetype, mode=mode, profile=profile.sku,
            tiling=TilingPlan(0, 1, 0, 0, 1, 0, 1, 0, 0, flow.tiling_pool.value),
            active_units=0, waves=0, per_tile_us=0.0, pipe_us={}, gm_us=0.0,
            launch_floor_us=0.0, t_pipeline_us=0.0, t_est_us=0.0, bubble=zero,
            effective_gm_bytes_s=0.0, working_set_bytes=0, l2_resident=True,
            fixpipe_path=fixpipe.path.value, fixpipe_lane=fixpipe.lane,
            warnings=("zero_launch_noop",), notes=flow.notes,
            extra={"launched": False}, authoring_backend=flow.authoring_backend.value,
            launched=False, estimate_kind=flow.estimate_kind, calibrated=flow.calibrated,
        )
    physical_units = profile.active_units(flow.archetype.value)
    requested_active = physical_units if active_units is None else require_int(
        "active_units", active_units, minimum=1
    )
    active = min(requested_active, physical_units)
    simt = flow.vector_mode == VectorMode.SIMT.value or any(w.pipe == "SIMT" for w in flow.pipe_work)
    if simt and profile.arch_code != 3510:
        raise FlowSchemaError("SIMT/SSBUF vector mode is unavailable on 2201")
    if flow.tiling_pool in (BufferPool.UB_LANE0, BufferPool.UB_LANE1):
        capacity = profile.ub_usable_bytes("simt" if simt else "simd")
    elif flow.tiling_pool is BufferPool.SSBUF:
        capacity = buffer_capacity(profile, flow.tiling_pool, archetype=flow.archetype)
    else:
        capacity = buffer_capacity(profile, flow.tiling_pool)
    slots = 2 if double_buffer else 1
    tiling = derive_tiling(
        flow.total_units, flow.base_tile_units, flow.live_bytes_per_unit, capacity,
        slots=slots, pool=flow.tiling_pool.value,
        requested_tile_units=requested_tile_units, spill_policy=flow.spill_policy,
    )
    sram_peaks, sram_warnings = _sram_plans(flow, profile, tiling, simt)
    total = int(round(flow.total_units))
    tile_count = tiling.tile_count
    full_units = max(1, int(round(tiling.tile_units)))
    tail_units = tiling.tail_units or (total - (tile_count - 1) * full_units)
    if tail_units <= 0:
        tail_units = full_units
    has_tail = tail_units != full_units
    lanes = _vector_lanes(flow)
    unique = flow.unique_gm_bytes + flow.spill_workspace_bytes
    working_set = int(unique)
    bandwidth = effective_gm_bandwidth(profile, working_set)
    alpha = profile.rates.rw_contention_alpha
    per_lane_cap = profile.rates.per_owner_port_bw

    pipe_full, pipe_warnings = _pipe_tile_us(flow, profile, full_units, total, lanes)
    pipe_tail, _ = _pipe_tile_us(flow, profile, tail_units, total, lanes)
    read_full, write_full = _gm_bytes(flow, full_units, total, False)
    read_tail, write_tail = _gm_bytes(flow, tail_units, total, False)
    first_read, first_write = _gm_bytes(flow, full_units, total, True)
    raw_pipe_full = dict(pipe_full)
    gm_total_us = _gm_tile_us(float(flow.gm_read_bytes), float(flow.gm_write_bytes), bandwidth, alpha)
    gm_step_us = _gm_tile_us(first_read, first_write, min(per_lane_cap, bandwidth), alpha)
    triton = _adapt_triton(flow, profile, raw_pipe_full, gm_step_us, working_set,
                           active, triton_features, items=tile_count * flow.recurrence_steps)
    if triton is not None:
        factor = triton.vector_degradation_factor
        for pipes in (pipe_full, pipe_tail):
            for key in ("VEC", "SIMT"):
                if key in pipes:
                    pipes[key] *= factor
    stage_full = _stage_services(
        flow, profile, pipe_full,
        first_read / per_lane_cap * 1e6, first_write / per_lane_cap * 1e6,
    )
    first_chain = _first_item_chain(flow, stage_full)
    compute_full = max([*pipe_full.values(), 0.0])
    compute_tail = max([*pipe_tail.values(), 0.0])
    if triton is not None:
        compute_full = max(compute_full, triton.cv_wall_us)
        compute_tail = max(compute_tail, triton.cv_wall_us * tail_units / full_units)
    gm_pipe_loads = {
        "MTE2_AIC": first_read / profile.rates.mte2_aic_bytes_s * 1e6,
        "MTE2_AIV": first_read / profile.rates.mte2_aiv_bytes_s * 1e6,
        "MTE3": first_write / profile.rates.mte3_bytes_s * 1e6,
    }
    if flow.archetype in (ExecutionArchetype.PURE_VECTOR, ExecutionArchetype.COMMUNICATION_HYBRID):
        gm_pipe_load = max(gm_pipe_loads["MTE2_AIV"], gm_pipe_loads["MTE3"])
    elif flow.archetype is ExecutionArchetype.PURE_CUBE:
        gm_pipe_load = max(gm_pipe_loads["MTE2_AIC"], gm_pipe_loads["MTE3"])
    else:
        gm_pipe_load = max(gm_pipe_loads.values())
    gm_pair_service = _gm_tile_us(first_read, first_write, min(per_lane_cap, bandwidth), alpha)
    tau_resource = max([*stage_full.values(), compute_full, gm_pipe_load, gm_pair_service, 0.0])
    recurrence_bound = 0.0
    if flow.recurrence is not None:
        latency = flow.recurrence.latency_us_at(profile.freq_mhz)
        recurrence_bound = max(latency / flow.recurrence.distance_items, flow.recurrence.bound_us)
    tau = max(tau_resource, recurrence_bound)
    credits = flow.pipeline_slots if double_buffer else 1
    if credits is not None:
        tau = max(tau, first_chain / credits)

    wave_count = int(math.ceil(tile_count / active))
    last_n = tile_count - (wave_count - 1) * active
    sync_each = (flow.sync_pairs * 0.45) if flow.archetype in (
        ExecutionArchetype.CV_COUPLED_1C2V,
        ExecutionArchetype.COMMUNICATION_HYBRID,
    ) else 0.0
    aggregate_full = min(active * per_lane_cap, bandwidth)
    gm_full_wave = _gm_tile_us(active * read_full, active * write_full, aggregate_full, alpha)
    if has_tail:
        gm_last = _gm_tile_us((last_n - 1) * read_full + read_tail,
                              (last_n - 1) * write_full + write_tail,
                              min(last_n * per_lane_cap, bandwidth), alpha)
        last_compute = max(compute_full, compute_tail) if last_n > 1 else compute_tail
    else:
        gm_last = _gm_tile_us(last_n * read_full, last_n * write_full,
                              min(last_n * per_lane_cap, bandwidth), alpha)
        last_compute = compute_full
    if wave_count == 1:
        compute_total_us = last_compute * flow.recurrence_steps
        waves_services = [max(last_compute, gm_last) * flow.recurrence_steps + sync_each]
        gm_waves = [gm_last * flow.recurrence_steps]
    else:
        compute_total_us = ((wave_count - 1) * compute_full + last_compute) * flow.recurrence_steps
        first_wall = max(compute_full, gm_full_wave) * flow.recurrence_steps + sync_each
        last_wall = max(last_compute, gm_last) * flow.recurrence_steps + sync_each
        if wave_count <= 4096:
            waves_services = [first_wall] * (wave_count - 1) + [last_wall]
            gm_waves = [gm_full_wave * flow.recurrence_steps] * (wave_count - 1) + [
                gm_last * flow.recurrence_steps]
        else:
            waves_services = [first_wall, last_wall]
            gm_waves = [gm_full_wave * flow.recurrence_steps, gm_last * flow.recurrence_steps]
    waves = wave_count
    sync_total = wave_count * sync_each
    gm_total_us = _gm_tile_us(float(flow.gm_read_bytes), float(flow.gm_write_bytes), bandwidth, alpha)
    t_setup = 0.0
    drain = 0.0
    steps = max(1, waves * flow.recurrence_steps)
    # Single declared chains have an exact finite credit schedule. A coupled
    # DAG remains a tagged service bound unless an explicit stage ledger exists.
    if len(flow.stage_order) == 1:
        durations = [stage_full.get(pipe, 0.0) for pipe in flow.stage_order[0]]
        last_stage = (_stage_services(flow, profile, pipe_tail,
                                       read_tail / per_lane_cap * 1e6, write_tail / per_lane_cap * 1e6)
                      if has_tail and last_n == 1 else stage_full)
        scheduled = pipeline_tail(durations, [last_stage.get(pipe, 0.0) for pipe in flow.stage_order[0]],
                                   steps - flow.recurrence_steps, flow.recurrence_steps, credits,
                                   recurrence_latency=recurrence_bound * (
                                       flow.recurrence.distance_items if flow.recurrence else 1),
                                   recurrence_distance=flow.recurrence.distance_items if flow.recurrence else 1)
        t_unit = float(scheduled["exact_makespan"])
        tau = max(tau, float(scheduled["tau_lower_bound"]))
    else:
        t_unit = t_setup + first_chain + max(0, steps - 1) * tau + drain
    barrier_sum = ((wave_count - 1) * waves_services[0] + waves_services[-1])
    wave_sum = (barrier_sum if flow.schedule_policy == "barrier_waves"
                else max(compute_total_us, gm_total_us) + sync_total)
    t_pipeline = max(wave_sum, t_unit)
    owner_result = None
    if flow.owner_work:
        if tiling.tile_units != 1:
            raise FlowSchemaError("per-owner recurrence cannot be merged into a larger tile")
        owner_result = owner_schedule(flow, profile, active, credits,
                                      triton.vector_degradation_factor if triton else 1.0)
        t_unit = owner_result["wall_us"]
        waves_services = owner_result["waves"]
        barrier_sum = sum(waves_services)
        compute_total_us = owner_result["compute_schedule_us"]
        first_chain = owner_result["first_item_us"]
        tau_resource = owner_result["resource_us"]
        pipe_full["VEC"] = owner_result["max_step_compute_us"]
        tau = max(tau, tau_resource, pipe_full["VEC"])
        t_pipeline = max(t_unit, gm_total_us) + sync_total
    if triton is not None:
        t_pipeline += triton.gather_us + triton.spill_penalty_us + triton.block_qo_penalty_us
        triton = _adapt_triton(flow, profile, raw_pipe_full, gm_step_us, working_set,
                               active, triton_features, wall_us_hint=max(t_pipeline, 1e-12),
                               items=tile_count * flow.recurrence_steps)
        pipe_warnings.extend(triton.warnings)
    t_est = max(t_pipeline, floor)
    tail = max(0.0, waves_services[-1] - min(waves_services)) if waves_services else 0.0
    if triton is not None:
        from .triton_lowering import evaluate_mfu_gates
        report = evaluate_mfu_gates(profile, wall_us=t_est,
                                    features=replace(TritonFeatures.from_mapping(
                                        dict(flow.triton_features) | dict(triton_features or {})),
                                        flops=sum(w.total_work * (2 if w.work_kind == "macs" else 1)
                                                  for w in flow.pipe_work if w.pipe == "CUBE")),
                                    working_set_bytes=working_set,
                                    total_bytes=flow.gm_read_bytes + flow.gm_write_bytes, dtype=flow.dtype)
        whole_busy = (raw_pipe_full.get("CUBE", 0) + raw_pipe_full.get("VEC", 0)
                      + .2 * raw_pipe_full.get("SIMT", 0)) * total / full_units * flow.recurrence_steps
        triton = replace(triton, mfu=report, extra=dict(triton.extra,
                          observed_overlap_ratio=whole_busy / max(t_est, 1e-12),
                          wall_us_hint=t_est, busy_scope="whole_flow_aggregate",
                          modeled_scope="one_full_tile_recurrence_step",
                          call_cost_scope="gather_spill_block_qo_once"))
    bubble = BubbleLedger(0.0, max(0.0, t_pipeline - sync_total), 0.0, tail, sync_total)
    l2_resident = working_set <= profile.memory.l2
    warnings = list(dict.fromkeys(sram_warnings + pipe_warnings))
    if tiling.retiled:
        warnings.append("stateless_retile_zero_spill")
    if tiling.passes > 1:
        warnings.append(f"algorithm_workspace_{tiling.passes}pass")
    if not l2_resident:
        warnings.append("working_set_spills_l2")
    if simt:
        warnings.append("simt_ub_budget_enforced")
    if flow.fixpipe_path is FixpipePath.L0C_TO_UB:
        warnings.append("fixpipe_l0c_to_ub_lane_selected")
    if flow.archetype is ExecutionArchetype.CV_COUPLED_1C2V and flow.sync_pairs == 0:
        warnings.append("1c2v_without_explicit_sync")
    if not double_buffer and len(pipe_full) > 1:
        warnings.append("single_buffer_serializes_stage_chain")
    if flow.estimate_kind == "structural_schema_only":
        warnings.append("structural_schema_only_uncalibrated_prior")
    extra: dict[str, Any] = {
        "requested_tile_units": tiling.requested_tile_units,
        "service_overlap_us": max(0.0, sum(pipe_full.values()) - compute_full),
        "gm_read_tile_us": read_full / max(per_lane_cap, 1.0) * 1e6,
        "gm_write_tile_us": write_full / max(per_lane_cap, 1.0) * 1e6,
        "wave_services_us": list(waves_services),
        "gm_wave_us": list(gm_waves),
        "gm_total_us": gm_total_us,
        "compute_wave_sum_us": compute_total_us,
        "wave_group_sum_us": barrier_sum,
        "schedule_policy": flow.schedule_policy,
        "latency_confidence": "selected_owner_chains_and_service_bound" if owner_result else
                              "analytic_service_bound" if len(flow.stage_order) > 1
                              else "selected_chain_and_service_bound",
        "core_monotonicity": "not_guaranteed_with_barriers_or_shared_contention",
        "per_lane_dma_cap_gbps": per_lane_cap / 1e9,
        "tau_floor_us": tau_resource,
        "gm_pipe_load_us": gm_pipe_load,
        "max_pipe_load_us": tau_resource,
        "requested_active_units": requested_active,
        "physical_active_units": physical_units,
        "recurrence_distance": flow.recurrence.distance_items if flow.recurrence else 0,
        "recurrence_steps": flow.recurrence_steps,
        "advance_steps": steps,
        "vector_lanes": lanes,
        "sram_peak_bytes_by_pool": sram_peaks,
        "t_unit_law": {"setup": t_setup, "first": first_chain, "N": steps,
                       "tau": tau, "finish": drain, "value": t_unit},
        "lane_vector_ops": list(flow.vector_ops_by_lane) if flow.vector_ops_by_lane else [],
        "cache_assumption": "warm_l2_declared" if l2_resident else "hbm_pressure",
    }
    if owner_result is not None:
        extra["owner_schedule"] = owner_result
    if triton is not None:
        extra["triton"] = triton.as_dict()
        extra["triton_detail"] = triton
        extra["mfu"] = triton.mfu.as_dict()
    if flow.recurrence is not None:
        extra["recurrence_bound_active"] = recurrence_bound > tau_resource
    return FlowEvaluation(
        name=flow.name, archetype=flow.archetype, mode=mode, profile=profile.sku,
        tiling=tiling, active_units=active, waves=waves,
        per_tile_us=max([*pipe_full.values(), 0.0]), pipe_us=dict(sorted(pipe_full.items())),
        gm_us=gm_total_us, launch_floor_us=floor, t_pipeline_us=t_pipeline,
        t_est_us=t_est, bubble=bubble, effective_gm_bytes_s=bandwidth,
        working_set_bytes=working_set, l2_resident=l2_resident,
        fixpipe_path=fixpipe.path.value, fixpipe_lane=fixpipe.lane,
        warnings=tuple(sorted(set(warnings))), notes=flow.notes, extra=extra,
        authoring_backend=flow.authoring_backend.value, tau_steady_us=tau,
        recurrence_interval_us=recurrence_bound, t_unit_us=t_unit, t_setup_us=t_setup,
        t_first_item_us=first_chain, t_drain_finish_us=drain,
        max_pipe_load_us=tau_resource, estimate_kind=flow.estimate_kind,
        calibrated=flow.calibrated,
    )


def launch_floor_ordering(profile: HardwareProfile, flow: OpFlowTemplate, *,
                          static_decode: bool = True) -> dict[str, float]:
    if not isinstance(static_decode, bool):
        raise ExecutionModeError("static_decode must be bool")
    return {
        mode.value: evaluate_flow(flow, profile, mode, static_decode=static_decode).t_est_us
        for mode in ExecutionMode
    }


__all__ = [
    "BubbleLedger", "ExecutionModeError", "FlowEvaluation", "LAUNCH_FLOORS",
    "LAUNCH_FLOORS_US", "TilingError", "TilingPlan", "derive_tiling",
    "effective_gm_bandwidth", "evaluate_flow", "launch_floor_ordering",
    "mode_launch_floor_us",
]
