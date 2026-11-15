"""Stable façade for the split parallel-physics modules.

[Role]
  Re-export the placement, context-parallel, EP-knee and xPyD solver APIs under
  the original ``mindie_sim.parallel_explorer`` import path while keeping each
  implementation module focused and below the repository line cap.

[Invariants]
  * Every public name resolves to exactly one implementation; no duplicate
    formula is maintained in this file.
  * All heavy invariants (HBM gate, EP factorization, causal schedule, stable
    request-rate cuts) live in the focused modules and are imported unchanged.

[Timeouts & Locks] Import-only; no I/O, device context or locks.
[Side-Effect Policy] No mutation of caller records or global state.
"""
from __future__ import annotations

from mindie_sim.parallel_context import (ContextParallelPlan, PhaseBridge,
                                        dcp_merge_payloads, model_families,
                                        pcp_index_k_gather_bytes,
                                        pcp_kv_gather_bytes, plan_context_parallel,
                                        register_pcp_builder, registered_builders,
                                        valid_pairs)
from mindie_sim.parallel_ep_knee import (DEFAULT_EP_CANDIDATES, SCALING_MODES,
                                        DecodeCostModel, StepBreakdown,
                                        decode_step_ms, prefill_step_ms,
                                        settle_optima, sweep_ep_knee,
                                        sweep_tp_ep_tradeoff)
from mindie_sim.parallel_placement import (PlacementPlan, build_placement,
                                          canonical_coordinate, lane_remap_bytes,
                                          lane_remap_ms, normalize_partition,
                                          rank_placement_map, validate_placement,
                                          with_world)
from mindie_sim.xpyd_pool_solver import (DEFAULT_RHO, Flow,
                                        completed_requests_per_s,
                                        kv_transfer_contention, logical_transfer_bytes,
                                        pool_config_facts, progressive_max_min,
                                        queue_wait_ms, solve_xpyd_pools)

__all__ = [
    "ContextParallelPlan", "DEFAULT_EP_CANDIDATES", "DEFAULT_RHO",
    "completed_requests_per_s", "DecodeCostModel",
    "Flow", "PhaseBridge", "PlacementPlan", "SCALING_MODES", "StepBreakdown",
    "build_placement", "canonical_coordinate", "dcp_merge_payloads",
    "decode_step_ms", "kv_transfer_contention",
    "lane_remap_bytes", "lane_remap_ms", "logical_transfer_bytes", "model_families",
    "normalize_partition", "pcp_index_k_gather_bytes", "pcp_kv_gather_bytes",
    "pool_config_facts",
    "plan_context_parallel", "prefill_step_ms", "progressive_max_min",
    "rank_placement_map", "sweep_tp_ep_tradeoff",
    "queue_wait_ms", "register_pcp_builder", "registered_builders", "settle_optima",
    "solve_xpyd_pools", "sweep_ep_knee", "valid_pairs", "validate_placement",
    "with_world",
]
