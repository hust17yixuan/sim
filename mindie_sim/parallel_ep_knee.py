"""Exact discrete EP knee with causal two-lane scheduling.

[Role]
  Price one decode or prefill step from model geometry, the fabric and the MoE
  straggler model; sweep EP scale-out in an explicit batch/world mode; return
  latency, throughput and efficiency optima plus every discrete knee.

[Invariants]
  * Global and per-replica batch are distinct declarations and are never mixed.
    ``EP*EDP`` owner groups split ``T_global*top_k`` assignments and every owner
    executes its share on a ``1/ETP`` slice, conserving useful FLOPs.
  * Every MoE layer repeats Dispatch/Combine causally; only a declared dual
    microbatch may overlap them.  PP uses the physical ``(m, s)`` trace.
  * An OOM or route-unqualified row is never a winner.  With no feasible row,
    ``ep_star`` is ``None`` and status is ``all_candidates_infeasible``.
  * Every active communicator group on every axis is route-qualified on its
    actual physical devices, and service timing uses those same endpoints.
  * Fractional EPLB slots/redundancy are rejected before any ``int`` coercion.
  * Latency/throughput/efficiency optima stay distinct with deterministic ties.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh records; caller Cases are not mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.memory_eplb import resolve_eplb_epoch_config
from mindie_sim.moe_skew_eplb import eplb_effect, max_rank_tokens
from mindie_sim.parallel_context import (model_families, pcp_attention_pairs,
                                         pcp_kv_gather_bytes)
from mindie_sim.parallel_placement import validate_placement
from mindie_sim.parallel_prefill_cost import prefill_step_ms
from mindie_sim.parallel_decode_cost import (AXIS_COLLECTIVES, SCALING_MODES,
                                           DecodeCostModel, StepBreakdown,
                                           _int, _interconnect_bw_gbps,
                                           _positive, check_route_qualified,
                                           execution_context, iter_axis_groups)
from mindie_sim.parallel_schedule import (layer_partition, pp_bubble_ms,
                                          pp_microbatch_schedule_ms,
                                          stage_durations)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec

DEFAULT_EP_CANDIDATES = (2, 4, 8, 16, 32, 64, 128)

def settle_optima(rows: Sequence[Mapping[str, Any]], *,
                  allow_unqualified: bool = False) -> dict[str, Any]:
    """Separate latency/throughput/efficiency optima and every discrete knee."""
    feasible = [row for row in rows
                if row.get("feasible")
                and (allow_unqualified or row.get("route_qualified", True))]
    if not feasible:
        return {"status": "all_candidates_infeasible", "ep_star": None,
                "optimal_ep": None, "ep_star_latency": None,
                "ep_star_throughput": None, "ep_star_efficiency": None,
                "peak_ep": None, "efficiency_ep": None, "knee_ep": None,
                "knees": (), "marginal_knees": (),
                "allow_unqualified": bool(allow_unqualified),
                "qualified": False}
    def tie(value: float, row: Mapping[str, Any]) -> tuple[float, int, int]:
        return (value, -int(row["world_size"]), -int(row["ep"]))
    latency = max(feasible, key=lambda row: tie(-float(row["step_ms"]), row))
    throughput = max(feasible, key=lambda row: tie(float(row["tps"]), row))
    efficiency = max(feasible, key=lambda row: tie(float(row["tps_per_die"]), row))
    ordered = sorted(feasible, key=lambda row: (int(row["ep"]), int(row["world_size"])))
    knees, marginal_knees = [], []
    previous = None
    for row in ordered:
        if previous is not None:
            delta_width = int(row["world_size"]) - int(previous["world_size"])
            delta_ep = int(row["ep"]) - int(previous["ep"])
            delta_q = float(row["tps"]) - float(previous["tps"])
            denominator = delta_width if delta_width > 0 else delta_ep
            if denominator > 0:
                marginal = delta_q / float(denominator)
                basis = "per_npu" if delta_width > 0 else "per_ep"
                sign_change = (previous.get("_marginal") is not None
                               and (previous["_marginal"] > 0.0) != (marginal > 0.0))
                if marginal <= 0.0 or sign_change:
                    knees.append(int(row["ep"]))
                row["marginal_basis"] = basis
                previous["marginal_gain_per_npu"] = previous.get("_marginal")
                marginal_knees.append((int(previous["ep"]), int(row["ep"]),
                                       marginal))
                row["_marginal"] = marginal
            previous = row
        else:
            previous = row
    for row in feasible:
        row.pop("_marginal", None)
    return {"status": "feasible", "ep_star": int(latency["ep"]),
            "optimal_ep": int(latency["ep"]),
            "ep_star_latency": int(latency["ep"]),
            "ep_star_throughput": int(throughput["ep"]),
            "ep_star_efficiency": int(efficiency["ep"]),
            "peak_ep": int(throughput["ep"]), "efficiency_ep": int(efficiency["ep"]),
            "knee_ep": knees[0] if knees else None, "knees": tuple(knees),
            "marginal_knees": tuple(marginal_knees),
            "allow_unqualified": bool(allow_unqualified),
            "qualified": (not allow_unqualified
                          and all(row.get("route_qualified", True)
                                  for row in feasible))}

def validate_slot_vector(values: Sequence[int] | None) -> tuple[int, ...] | None:
    """Reject a fractional slot vector *before* any ``int`` coercion."""
    if values is None:
        return None
    slots = []
    for value in values:
        if isinstance(value, bool):
            raise ModelingError(f"eplb_slots_invalid:{value}")
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer() or value < 0:
                raise ModelingError(f"eplb_slots_invalid:{value}")
            slots.append(int(value))
        elif isinstance(value, int):
            if value < 0:
                raise ModelingError(f"eplb_slots_invalid:{value}")
            slots.append(value)
        else:
            raise ModelingError(f"eplb_slots_invalid:{value}")
    return tuple(slots)

def _tree(case: CaseSpec, fabric: Any = None) -> Any:
    if fabric is not None:
        if not hasattr(fabric, "collective"):
            raise ModelingError(f"parallel_explorer_fabric_invalid:{type(fabric).__name__}")
        return fabric
    return HierarchicalFabricTree.from_hardware(case.machine)

def  _eplb_extra_bytes(case: CaseSpec, redundant: int, slots=None,
                      staging=None) -> float:
    from mindie_sim.moe_skew_eplb import replicated_expert_bytes
    return float(replicated_expert_bytes(
        case, redundant, redundant_slots_per_rank=slots,
        rebalance_staging_bytes_per_rank=staging)["bytes_per_rank"])

def  sweep_ep_knee(case: CaseSpec | Mapping[str, Any], *,
                  ep_candidates: Sequence[int] = DEFAULT_EP_CANDIDATES,
                  batch_sequences: int | None = None, context_tokens: int | None = None,
                  routing_skew: float = 1.0, enable_eplb: bool | None = None,
                  eplb_redundant_experts: int | None = None,
                  eplb_redundant_slots_per_rank: Sequence[int] | None = None,
                  fabric: Any = None,
                  mc2: bool | None = None, hccl_buffsize_mb: float | None = None,
                  marginal_threshold: float = 0.02,
                  mc2_buffer_multiplier: float = 1.0,
                  scaling_mode: str = "fixed_per_dp_batch",
                  fixed_world_npus: int | None = None,
                  host_overlap: float = 0.5,
                  tp_override: int | None = None,
                  overlap_mode: str = "serial",
                  microbatches: int = 1,
                  allow_unqualified: bool = False) -> dict[str, Any]:
    """Sweep EP in one declared batch/world mode over the feasible set."""
    if scaling_mode not in SCALING_MODES:
        raise ModelingError(f"parallel_explorer_scaling_mode_invalid:{scaling_mode}")
    parsed = parse_case(case)
    base = parsed.runtime.parallel
    meta = dict(parsed.runtime.metadata)
    skew = _positive("routing_skew", routing_skew, strict=False)
    threshold = _positive("marginal_threshold", marginal_threshold, strict=False)
    if skew < 1.0:
        raise ModelingError(f"parallel_explorer_routing_skew_invalid:{skew}")
    slot_vector = validate_slot_vector(eplb_redundant_slots_per_rank)
    raw_redundant = (meta.get("num_redundant_experts", 0)
                     if eplb_redundant_experts is None else eplb_redundant_experts)
    if raw_redundant is None:
        raw_redundant = 0
    raw_redundant = _int("eplb_redundant_experts", raw_redundant, minimum=0)
    # Enablement/count/slot agreement is resolved once per EP candidate by the
    # canonical resolver -- never by ``enable and R > 0``.
    eplb_on = False
    redundant = 0
    batch_value = (batch_sequences if batch_sequences is not None
                   else min(int(parsed.runtime.max_num_seqs), 64))
    batch = _int("batch_sequences", batch_value)
    if context_tokens is None:
        request = parsed.workload.requests[0]
        context_tokens = min(parsed.runtime.max_model_len,
                             request.prompt_tokens + request.output_tokens)
    context = _int("context_tokens", context_tokens)
    tree = _tree(parsed, fabric)
    evidence = execution_context(parsed)
    rows: list[dict[str, Any]] = []
    world_mode = "fixed_world" if fixed_world_npus is not None else "marginal_world"
    fixed_world = (_int("fixed_world_npus", fixed_world_npus)
                   if fixed_world_npus is not None else None)
    sweep_tp = _int("tp_override", tp_override) if tp_override is not None else int(base.tp)
    for ep in sorted({_int("ep_candidate", value, minimum=1)
                      for value in ep_candidates}):
        if ep < 2 or (parsed.model.num_experts and parsed.model.num_experts % ep != 0):
            rows.append({"ep": ep, "world_size": None, "feasible": False,
                         "oom": False, "reason": "invalid_ep_partition",
                         "tps": 0.0, "tps_per_die": 0.0, "step_ms": None})
            continue
        try:
            if fixed_world is not None:
                denominator = sweep_tp * base.pp * base.pcp
                if fixed_world % denominator != 0:
                    raise ModelingError("fixed_world_not_divisible")
                dp = fixed_world // denominator
                if ep > dp * sweep_tp:
                    raise ModelingError("ep_exceeds_fixed_world_plane")
                parallel = ParallelSpec(
                    dp=dp, tp=sweep_tp, ep=ep, pp=base.pp, cp=base.cp, pcp=base.pcp,
                    cp_plan=base.cp_plan, enable_dsa_cp=base.enable_dsa_cp,
                    pcp_lane_policy=base.pcp_lane_policy)
                expected_world = fixed_world
            else:
                dp = ep
                parallel = ParallelSpec(
                    dp=dp, tp=sweep_tp, ep=ep, pp=base.pp, cp=base.cp, pcp=base.pcp,
                    cp_plan=base.cp_plan, enable_dsa_cp=base.enable_dsa_cp,
                    pcp_lane_policy=base.pcp_lane_policy)
                expected_world = parallel.world_size
            candidate, plan, reason = validate_placement(parsed, parallel)
            if candidate is None:
                raise ModelingError(reason or "invalid_placement")
            replicas = int(plan["active_replicas"])
            if scaling_mode == "fixed_global_batch":
                if batch % replicas != 0:
                    raise ModelingError(f"global_batch_not_divisible:{batch}%{replicas}")
                per_rep = batch // replicas
            else:
                per_rep = batch
            model = DecodeCostModel.from_case(candidate, context_tokens=context)
            placement = None
            slots = None
            staging = None
            epoch_mismatch = False
            world = int(plan["world"])
            ep_coords = tuple(int(row[5]) for row in plan["rank_map"])
            epoch = resolve_eplb_epoch_config(
                enable_eplb=enable_eplb, redundant_experts=raw_redundant,
                slot_vector=slot_vector, ep=ep, world=world,
                num_experts=int(model.num_experts or 0), ep_coords=ep_coords,
                top_k=max(1, int(model.top_k or 1)), routing_skew=skew)
            eplb_on = epoch.enable_eplb
            redundant = epoch.redundant
            if epoch.active:
                from mindie_sim.moe_skew_eplb import (place_eplb,
                                                     replicated_expert_bytes)
                # The EP domain receives DP*B tokens; EP-slot destinations are
                # mapped onto every PP/PCP/ETP/EDP rank, never left as an
                # EP-length vector when the world is DP*TP.
                tokens_ep = float(int(plan["dp"])) * float(per_rep)
                placement = place_eplb(int(round(tokens_ep)), model.top_k, ep, skew,
                                       num_experts=model.num_experts,
                                       redundant_experts=redundant, per_token_ms=1.0)
                executed_ep = tuple(int(value) for value in
                                    placement.redundant_slots_per_rank)
                if len(executed_ep) != int(ep):
                    raise ModelingError("parallel_explorer_eplb_slots_length")
                mapped_executed = tuple(executed_ep[ep_coords[rank]]
                                        for rank in range(world))
                supplied = epoch.slots_rank
                slots = supplied if supplied is not None else mapped_executed
                # Unconditional epoch binding: the resident/staging ledger and
                # the execution service MUST name the same placement epoch,
                # whether the vector was supplied by the caller or derived.
                if tuple(int(value) for value in slots) != mapped_executed:
                    epoch_mismatch = True
                if epoch_mismatch:
                    rows.append({"ep": ep, "world_size": expected_world,
                                 "feasible": False, "oom": False,
                                 "route_qualified": False,
                                 "fabric_route_qualified": False,
                                 "route_evidence": "eplb_epoch_mismatch",
                                 "reason": "eplb_epoch_mismatch",
                                 "tps": 0.0, "tps_per_die": 0.0,
                                 "step_ms": None})
                    continue
                replica = replicated_expert_bytes(
                    candidate, redundant, redundant_slots_per_rank=slots)
                replica_rank = tuple(float(value) for value in
                                     replica["per_rank_bytes"])
                bandwidth = _interconnect_bw_gbps(parsed.machine)
                placement = place_eplb(
                    int(round(tokens_ep)), model.top_k, ep, skew,
                    num_experts=model.num_experts, redundant_experts=redundant,
                    per_token_ms=1.0,
                    extra_weight_bytes_per_rank=max(replica_rank, default=0.0),
                    rebalance_bw_gbps=bandwidth)
                executed_ep = tuple(int(value) for value in
                                    placement.redundant_slots_per_rank)
                mapped_executed = tuple(executed_ep[ep_coords[rank]]
                                        for rank in range(world))
                if tuple(int(value) for value in slots) != mapped_executed:
                    rows.append({"ep": ep, "world_size": expected_world,
                                 "feasible": False, "oom": False,
                                 "route_qualified": False,
                                 "fabric_route_qualified": False,
                                 "route_evidence": "eplb_epoch_mismatch",
                                 "reason": "eplb_epoch_mismatch",
                                 "tps": 0.0, "tps_per_die": 0.0,
                                 "step_ms": None})
                    continue
                staging = (replica_rank if replica_rank
                           else (float(placement.rebalance_staging_bytes_per_rank),)
                           * expected_world)
            memory = analyze_memory_breakdown(
                candidate, sequences=per_rep, sequence_tokens=context,
                hccl_buffsize_mb=hccl_buffsize_mb, enable_eplb=eplb_on,
                eplb_redundant_experts=redundant,
                eplb_redundant_slots_per_rank=slots,
                eplb_rebalance_staging_bytes_per_rank=staging,
                mc2_buffer_multiplier=mc2_buffer_multiplier,
                moe_recipe=("mc2" if mc2 is True else
                            "alltoall" if mc2 is False else None))
            route_ok, route_evidence = check_route_qualified(
                tree, plan, 0, **evidence)
            route_qualified = bool(route_ok)
            if not memory.oom_feasible:
                rows.append({"ep": ep, "world_size": expected_world, "feasible": False,
                             "oom": memory.status != "memory_unqualified",
                             "route_qualified": route_qualified,
                             "fabric_route_qualified": route_qualified,
                             "route_evidence": route_evidence,
                             "reason": f"{memory.status}:{memory.reason}",
                             "peak_hbm_bytes": memory.peak_hbm_bytes,
                             "budget_bytes": memory.budget_bytes,
                             "headroom_bytes": memory.headroom_bytes,
                             "tps": 0.0, "tps_per_die": 0.0, "step_ms": None})
                continue
            if not route_qualified and not allow_unqualified:
                rows.append({"ep": ep, "world_size": expected_world, "feasible": False,
                             "oom": False, "route_qualified": False,
                             "fabric_route_qualified": False,
                             "route_evidence": route_evidence,
                             "reason": "fabric_route_unqualified",
                             "peak_hbm_bytes": memory.peak_hbm_bytes,
                             "budget_bytes": memory.budget_bytes,
                             "headroom_bytes": memory.headroom_bytes,
                             "tps": 0.0, "tps_per_die": 0.0, "step_ms": None})
                continue
            step = model.step(
                per_rep, context=context, fabric=tree, routing_skew=skew,
                enable_eplb=eplb_on, eplb_redundant_experts=redundant, mc2=mc2,
                scaling_mode=scaling_mode, global_batch=batch,
                per_replica_batch=per_rep, active_replicas=replicas,
                host_overlap=host_overlap, eplb_placement=placement,
                # Rebalance staging is charged once through the dedicated
                # ``eplb_rebalance_ms`` term; re-reading it as per-step FFN
                # weight bytes double-charged the same resource service.
                microbatches=microbatches, overlap_mode=overlap_mode,
                eplb_rebalance_ms=(placement.rebalance_ms if placement else 0.0),
                endpoints=tuple(range(int(expected_world))),
                dtype=evidence["dtype"], topology=evidence["topology"])
            if not step.route_qualified and not allow_unqualified:
                rows.append({
                    "ep": ep, "world_size": expected_world, "feasible": False,
                    "oom": False, "route_qualified": False,
                    "fabric_route_qualified": False,
                    "route_evidence": step.route_evidence,
                    "reason": "fabric_route_unqualified",
                    "peak_hbm_bytes": memory.peak_hbm_bytes,
                    "budget_bytes": memory.budget_bytes,
                    "headroom_bytes": memory.headroom_bytes,
                    "tps": 0.0, "tps_per_die": 0.0, "step_ms": None})
                continue
            row = {
                "ep": ep, "world_size": expected_world, "feasible": True, "oom": False,
                "route_qualified": bool(route_qualified and step.route_qualified),
                "fabric_route_qualified": bool(step.route_qualified),
                "route_evidence": (step.route_evidence
                                   if not step.route_qualified else route_evidence),
                "world_mode": world_mode, "scaling_mode": scaling_mode,
                "global_batch": step.global_batch, "per_replica_batch": per_rep,
                "active_replicas": replicas, "useful_tokens_per_step": step.useful_tokens_per_step,
                "expert_assignments": step.expert_assignments,
                "expert_assignments_per_owner": step.expert_assignments_per_owner,
                "expert_flops": step.expert_flops,
                "expert_flops_per_owner": step.expert_flops_per_owner,
                "tokens_entering_ep_group": step.tokens_entering_ep_group,
                "tile_tokens": step.tile_tokens,
                "tile_wave_ms": step.tile_wave_ms,
                "uncached_fallback_ms": step.uncached_fallback_ms,
                "mlapo_prefetch": step.mlapo_prefetch,
                "t_attn_ms": step.t_attn_ms, "t_expert_compute_ms": step.t_expert_compute_ms,
                "t_expert_straggler_ms": step.t_expert_straggler_ms,
                "t_dispatch_combine_ms": step.t_dispatch_combine_ms,
                "t_non_moe_ms": step.t_non_moe_ms, "step_ms": step.step_ms,
                "expert_weight_ms": step.expert_weight_ms,
                "expert_amortization_ms": (model.expert_bytes * 1000.0 / model.hbm_bw
                                           if model.expert_bytes else 0.0),
                "straggler_ratio": step.straggler_ratio, "bottleneck": step.bottleneck,
                "tps": step.tps, "tps_per_die": step.tps_per_die,
                "marginal_gain": None, "peak_hbm_bytes": memory.peak_hbm_bytes,
                "budget_bytes": memory.budget_bytes,
                "headroom_bytes": memory.headroom_bytes,
                "host_bound_ratio": step.host_bound_ratio,
                "raw_comm_ms": step.raw_comm_ms, "schedule": dict(step.schedule),
                "mc2": step.mc2, "overlap_mode": step.overlap_mode,
                "pp_microbatches": step.pp_microbatches,
                "eplb_rebalance_ms": step.eplb_rebalance_ms,
            }
            previous = next((item for item in reversed(rows) if item.get("feasible")), None)
            if previous is not None and previous.get("tps", 0.0) > 0:
                delta_q = float(step.tps) - float(previous["tps"])
                delta_world = (int(expected_world) - int(previous.get("world_size")
                                                        or expected_world))
                delta_ep = int(ep) - int(previous.get("ep", ep))
                denominator = delta_world if delta_world > 0 else max(1, delta_ep)
                row["marginal_gain"] = delta_q / float(denominator)
                row["marginal_relative"] = step.tps / float(previous["tps"]) - 1.0
            row["threshold_knee"] = bool(
                row.get("marginal_relative") is not None
                and float(row["marginal_relative"]) < threshold)
            rows.append(row)
        except (ModelingError, ValueError) as exc:
            rows.append({"ep": ep, "world_size": None, "feasible": False, "oom": False,
                         "reason": str(exc), "tps": 0.0, "tps_per_die": 0.0,
                         "step_ms": None})
    for row in rows:
        row.setdefault("world_mode", world_mode)
        row.setdefault("scaling_mode", scaling_mode)
        row.pop("_marginal", None)
    optima = settle_optima(rows, allow_unqualified=allow_unqualified)
    feasible_rows = [row for row in rows if row.get("feasible")]
    threshold_knee = next((int(row["ep"]) for row in rows if row.get("threshold_knee")), None)
    return {
        "schema": "mindie-sim.ep-knee/2", **optima,
        "threshold_knee_ep": threshold_knee,
        "batch_sequences": batch, "context_tokens": context,
        "routing_skew": skew, "enable_eplb": eplb_on, "redundant_experts": redundant,
        "mc2": (bool(mc2) if mc2 is not None else parsed.runtime.moe_recipe == "mc2"),
        "marginal_threshold": threshold, "fabric": tree.to_dict(),
        "rows": rows, "feasible_rows": feasible_rows,
        "selection": ("feasible_set_only; ep_star=argmin_latency; "
                      "peak_ep=argmax_throughput; efficiency_ep=argmax_Q_per_NPU; "
                      "knee_ep=first_marginal_efficiency_sign_change"),
    }

def   decode_step_ms(case: CaseSpec | Mapping[str, Any], batch_tokens: int,
                   **kwargs: Any) -> float:
    """Convenience closure used by the MTP reversal solver."""
    parsed = parse_case(case)
    return DecodeCostModel.from_case(parsed).step_ms(batch_tokens, **kwargs)

def  sweep_tp_ep_tradeoff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy proxy avoids an import cycle with ``parallel_tradeoff``."""
    from mindie_sim.parallel_tradeoff import sweep_tp_ep_tradeoff as _sweep
    return _sweep(*args, **kwargs)

__all__ = ["AXIS_COLLECTIVES", "DEFAULT_EP_CANDIDATES", "SCALING_MODES",
           "DecodeCostModel", "StepBreakdown", "decode_step_ms",
           "check_route_qualified", "execution_context",
           "iter_axis_groups", "prefill_step_ms", "settle_optima", "sweep_ep_knee",
           "sweep_tp_ep_tradeoff", "validate_slot_vector"]
