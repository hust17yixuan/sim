"""Deterministic Day-0 graph optimizer entry point.

[Role] Dispatch a canonical Case/OpSpec tuple to the ``graph_costs`` scoring
  path and a typed projection to the public Engine scoring path.
[Invariants] One fixed pass order, one shared per-pass budget and one monotone
  waterfall; a canonical run never uses the projection as its timing authority.
[Timeouts & Locks] Caller-visible wall/evaluation budgets; no locks.
[Side-Effect Policy] Never mutates the caller's Case, ModelSpec or OpSpec tuple.
"""
from __future__ import annotations

from mindie_sim.prediction.engine_plan import HardwareProfile
from mindie_sim.prediction.fusion_bridge import canonical_plan
from mindie_sim.prediction.fusion_graph import TypedGraph

from .canonical_optimizer import _run_canonical
from .graph_optimizer_engine import _run_analytical
from .optimizer_report import (
    MAX_PEAK_BYTES, Budget, OptimizationReport, PassReport, Rejection,
)

__all__ = ["run_optimizer", "HardwareProfile", "OptimizationReport", "PassReport",
           "Rejection", "Budget", "MAX_PEAK_BYTES"]


def run_optimizer(graph_or_case, hardware_or_ops=None, *,
                  evaluation_budget: int = 128, max_streams: int | None = None,
                  wall_budget_s: float = 30.0,
                  max_candidates_per_pass: int = 32,
                  allow_relaxed_numerics: bool = False) -> OptimizationReport:
    if isinstance(graph_or_case, TypedGraph):
        hardware = hardware_or_ops
        if not isinstance(hardware, HardwareProfile):
            raise TypeError("typed_projection_requires_HardwareProfile")
        return _run_analytical(graph_or_case, hardware,
                               evaluation_budget=evaluation_budget,
                               per_pass=max_candidates_per_pass,
                               max_streams=max_streams,
                               wall_budget_s=wall_budget_s,
                               allow_relaxed_numerics=allow_relaxed_numerics)
    if hardware_or_ops is None:
        raise TypeError("canonical_optimizer_requires_selected_ops")
    plan = canonical_plan(graph_or_case, hardware_or_ops)
    return _run_canonical(plan, evaluation_budget=evaluation_budget,
                          per_pass=max_candidates_per_pass,
                          max_streams=max_streams, wall_budget_s=wall_budget_s,
                          allow_relaxed_numerics=allow_relaxed_numerics)
