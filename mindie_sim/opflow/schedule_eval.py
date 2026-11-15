"""Service construction and finite schedule integration.

[Role]
Build per-stage service ledgers and preserve owner and tail distributions.
[Invariants]
Tile work never becomes a call-level cost; finite credits use actual durations.
Owner task counts, work and traffic are conserved before timing.
[Timeouts & Locks]
Finite DAG expansion has a one-million-node budget; no waits or locks.
[Side-Effect Policy]
Pure in-memory evaluation; no device or filesystem changes.
"""

import math
from dataclasses import replace
from typing import Any, Mapping

from .flow_schema import (AuthoringBackend, BufferPool, ExecutionArchetype,
                          FixpipePath, FlowSchemaError, OpFlowTemplate, buffer_capacity)
from .hardware_profiles import HardwareProfile
from .tiling import TilingError, TilingPlan
from .transfer_accounting import plan_buffer_pool
from .triton_lowering import (TritonFeatures, TritonLoweringError, TritonLoweringResult,
                              analyze_triton_lowering)
from .schedule_math import pipeline_chain


def _tile_units_list(tiling: TilingPlan, total: int) -> list[int]:
    if tiling.tile_units_exact:
        return list(tiling.tile_units_exact)
    return [int(tiling.tile_units)] * tiling.tile_count


def _vector_lanes(flow: OpFlowTemplate) -> int:
    if flow.archetype is not ExecutionArchetype.CV_COUPLED_1C2V:
        return 1
    # Two-lane service is credited only when both lanes own work, the dualDst
    # route is explicit, and a producer/consumer handshake is declared.
    if flow.sync_pairs < 1:
        return 1
    if flow.dual_dst_ctl in (0, 1, 2) and len(flow.lanes_receiving) == 2:
        return 2
    return 1


def _pipe_tile_us(flow: OpFlowTemplate, profile: HardwareProfile, units: float,
                  total: float, lanes: int) -> tuple[dict[str, float], list[str]]:
    """Per-item or per-recurrence-step pipe service, never whole-op work."""
    pipe_us: dict[str, float] = {}
    warnings: list[str] = []
    if total <= 0:
        return pipe_us, warnings
    steps = max(1, flow.recurrence_steps)
    for work in flow.pipe_work:
        if work.total_work <= 0:
            continue
        amount = work.total_work * (units / total) / steps
        if work.pipe == "CUBE" and work.work_kind == "ops":
            amount /= 2  # rate_for(CUBE) is MAC/s, while an ops field counts FLOPs
        rate = profile.rate_for(work.pipe, flow.dtype)
        lane_scale = lanes
        if flow.vector_ops_by_lane and lanes == 2:
            lane_scale = sum(flow.vector_ops_by_lane) / max(flow.vector_ops_by_lane)
        divisor = rate * (lane_scale if work.pipe in ("VEC", "VEC_TRANS") else 1.0)
        pipe_us[work.pipe] = pipe_us.get(work.pipe, 0.0) + amount / divisor * 1e6
    if "VEC" in pipe_us and "VEC_TRANS" in pipe_us:
        vec, trans = pipe_us.pop("VEC"), pipe_us.pop("VEC_TRANS")
        if profile.arch_code == 3510:
            pipe_us["VEC"] = max(vec, trans) + 0.15 * min(vec, trans)
            warnings.append("regbase_dual_issue_overlap")
        else:
            pipe_us["VEC"] = vec + trans
            warnings.append("regbase_dual_issue_unavailable_on_mem_3201")
    return pipe_us, warnings


def _gm_bytes(flow: OpFlowTemplate, units: float, total: float,
              weight_done: bool) -> tuple[float, float]:
    """Per-item or per-recurrence-step GM bytes."""
    if not isinstance(weight_done, bool):
        raise FlowSchemaError("weight_done must be bool")
    steps = max(1, flow.recurrence_steps)
    read_total = max(0, flow.gm_read_bytes - flow.weight_bytes_read_once)
    read = read_total * (units / total) / steps if total else 0.0
    write = flow.gm_write_bytes * (units / total) / steps if total else 0.0
    if weight_done:
        read += flow.weight_bytes_read_once / steps
    return float(read), float(write)


def _gm_tile_us(read: float, write: float, lane_bw: float, alpha: float) -> float:
    if lane_bw <= 0:
        return 0.0
    read_us = read / lane_bw * 1e6
    write_us = write / lane_bw * 1e6
    if read_us > 0 and write_us > 0:
        return max(read_us, write_us) + alpha * min(read_us, write_us)
    return read_us + write_us


def _chain_us(stage_us: Mapping[str, float], chain: tuple[str, ...]) -> float:
    return sum(stage_us.get(stage, 0.0) for stage in chain)


def _stage_services(flow: OpFlowTemplate, profile: HardwareProfile, pipe_us: Mapping[str, float],
                    read_us: float, write_us: float) -> dict[str, float]:
    stage = dict(pipe_us)
    reader = "MTE2_AIV" if flow.archetype in (
        ExecutionArchetype.PURE_VECTOR, ExecutionArchetype.COMMUNICATION_HYBRID) else "MTE2_AIC"
    writer = "FIX" if flow.archetype is ExecutionArchetype.PURE_CUBE else "MTE3"
    stage[reader] = max(stage.get(reader, 0.0), read_us)
    stage[writer] = max(stage.get(writer, 0.0), write_us)
    return stage


def _first_item_chain(flow: OpFlowTemplate, stage_us: Mapping[str, float]) -> float:
    chains = flow.stage_order
    sums = [_chain_us(stage_us, chain) for chain in chains]
    if flow.archetype is ExecutionArchetype.CV_COUPLED_1C2V and flow.fixpipe_path is FixpipePath.L0C_TO_UB:
        # The L0C->UB handoff makes the Vector chain depend on the Cube chain.
        return max(max(sums, default=0.0), sums[0] + sum(
            stage_us.get(pipe, 0.0) for pipe in ("VEC", "MTE3")
        ))
    return max(sums, default=0.0)


def _wave_services(*, tile_count: int, active_units: int, compute_tile_us: float,
                   read_per_tile: float, write_per_tile: float, bandwidth: float,
                   per_lane_cap: float, alpha: float
                   ) -> tuple[list[float], list[float], float]:
    """Compatibility helper: equal tiles, lane-concurrent wave walls."""
    if tile_count < 1 or active_units < 1:
        raise FlowSchemaError("tile_count and active_units must be positive")
    waves = int(math.ceil(tile_count / active_units))
    services: list[float] = []
    gm_waves: list[float] = []
    total_gm = 0.0
    for wave in range(waves):
        lanes = min(active_units, tile_count - wave * active_units)
        lane_bw = min(per_lane_cap, bandwidth / lanes)
        gm = _gm_tile_us(lanes * read_per_tile, lanes * write_per_tile, lane_bw * lanes, alpha)
        services.append(max(compute_tile_us, gm))
        gm_waves.append(gm)
        total_gm += gm
    return services, gm_waves, total_gm


def _sram_plans(flow: OpFlowTemplate, profile: HardwareProfile, tiling: TilingPlan,
                simt: bool) -> tuple[dict[str, int], list[str]]:
    if not isinstance(simt, bool):
        raise FlowSchemaError("simt must be bool")
    warnings: list[str] = []
    peaks: dict[str, int] = {}
    for pool in (BufferPool.UB_LANE0, BufferPool.UB_LANE1, BufferPool.L1,
                 BufferPool.L0A, BufferPool.L0B, BufferPool.L0C, BufferPool.SSBUF, BufferPool.BIAS):
        plan = plan_buffer_pool(flow.buffers, pool, profile)
        if plan.peak_bytes == 0:
            continue
        if pool is BufferPool.SSBUF:
            capacity = buffer_capacity(profile, pool, archetype=flow.archetype)
        elif pool in (BufferPool.UB_LANE0, BufferPool.UB_LANE1):
            capacity = profile.ub_usable_bytes("simt" if simt else "simd")
        else:
            capacity = profile.memory.capacity(pool.value)
        if plan.peak_bytes > capacity:
            raise TilingError(
                f"{pool.value} liveness {plan.peak_bytes} B exceeds {capacity} B"
            )
        peaks[pool.value] = plan.peak_bytes
    return peaks, warnings


def _merge_triton_features(stored: Mapping[str, Any],
                           overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate each input before merging so overrides cannot hide invalid fields."""
    if not isinstance(stored, Mapping):
        raise TritonLoweringError("flow.triton_features must be a mapping")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise TritonLoweringError("triton_features must be a mapping")
    TritonFeatures.from_mapping(stored)
    TritonFeatures.from_mapping(overrides)
    return dict(stored) | (dict(overrides) if overrides is not None else {})


def _adapt_triton(flow: OpFlowTemplate, profile: HardwareProfile,
                  pipe_us: Mapping[str, float], gm_tile_us: float,
                  working_set_bytes: int, active_units: int,
                  features: Mapping[str, Any] | None,
                  wall_us_hint: float | None = None, items: int = 1) -> TritonLoweringResult | None:
    merged = _merge_triton_features(flow.triton_features, features)
    requested = flow.authoring_backend is AuthoringBackend.TRITON_ASCEND or bool(merged)
    if not requested:
        return None
    cube_flops = sum(w.total_work * (2 if w.work_kind == "macs" else 1)
                     for w in flow.pipe_work if w.pipe == "CUBE")
    if "flops" in merged and merged["flops"] != cube_flops:
        raise TritonLoweringError("MFU FLOPs must equal this whole flow's Cube FLOPs")
    merged["flops"] = cube_flops
    spec = TritonFeatures.from_mapping(merged)
    aic = pipe_us.get("CUBE", 0.0)
    aiv = pipe_us.get("VEC", 0.0) + pipe_us.get("SIMT", 0.0) * 0.2
    per_core = spec.per_core_working_set_bytes or max(1, working_set_bytes // max(1, active_units))
    return analyze_triton_lowering(
        profile, aic_us=aic, aiv_us=aiv, gm_us=gm_tile_us, features=spec,
        working_set_bytes=working_set_bytes,
        per_core_working_set_bytes=per_core,
        total_bytes=flow.gm_read_bytes + flow.gm_write_bytes,
        wall_us_hint=wall_us_hint, dtype=flow.dtype, items=items,
    )


def owner_schedule(flow, profile, active, credits, factor=1.0):
    """Fixed-order independent owner queues; recurrent steps never migrate.

    Per-step traffic is the selected amortized service prior. Whole-owner
    compute is mandatory even when other owners have zero accepted tokens.
    """
    import heapq
    queues = [(0.0, i) for i in range(active)]
    walls, computes, steps, first_items, resources = [], [], [], [], []
    compute_queues = [0.0] * active
    if sum(x.multiplicity for x in flow.owner_work) > 1_000_000:
        raise FlowSchemaError("owner schedule exceeds task budget")
    for owner in flow.owner_work:
        n = max(1, owner.steps)
        read = owner.read_bytes / n / profile.rates.per_owner_port_bw * 1e6
        write = owner.write_bytes / n / profile.rates.per_owner_port_bw * 1e6
        compute = owner.vector_ops / n / profile.rate_for("VEC", flow.dtype) * 1e6 * factor
        feedback = flow.recurrence
        distance = feedback.distance_items if feedback else 1
        latency = max(feedback.latency_us_at(profile.freq_mhz), feedback.bound_us*distance) if feedback else 0
        time = float(pipeline_chain((read, compute, write), n, credits,
                                   recurrence_latency=latency, recurrence_distance=distance)["exact_makespan"])
        first_items.append(read + compute + write)
        resources.append(max(read, compute, write, latency/distance))
        walls.append(time)
        computes.append(compute * n)
        steps.append(compute)
        for _ in range(owner.multiplicity):
            finish, unit = heapq.heappop(queues)
            heapq.heappush(queues, (finish + time, unit))
            compute_queues[unit] += compute * n
    durations = [t for t, o in zip(walls, flow.owner_work) for _ in range(o.multiplicity)]
    waves = [max(durations[i:i+active]) for i in range(0, len(durations), active)]
    wall = sum(waves) if flow.schedule_policy == "barrier_waves" else max(t for t, _ in queues)
    compute_jobs = [t for t, o in zip(computes, flow.owner_work) for _ in range(o.multiplicity)]
    compute_wall = (sum(max(compute_jobs[i:i+active]) for i in range(0, len(compute_jobs), active))
                    if flow.schedule_policy == "barrier_waves" else max(compute_queues))
    return {"compute_schedule_us": compute_wall, "first_item_us": max(first_items),
            "resource_us": max(resources), "wall_us": wall, "waves": waves, "owner_service_us": walls,
            "owner_compute_us": computes, "max_step_compute_us": max(steps, default=0),
            "assignment": "fixed_order_earliest_free", "traffic_model": "amortized_per_step"}
