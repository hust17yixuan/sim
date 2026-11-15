"""Stable flow cuts and KV handoff contention for xPyD admission.

[Role] Allocate simultaneous P-HBM/P-NIC/RoCE/D-NIC/D-HBM/staging resources with
  weighted progressive max-min filling, and price one page-granular KV handoff
  including decode write interference and finite queueing.
[Invariants] Utilization is strictly below one; a zero-KV full-destination-cache
  handoff is a finite zero-work transaction; a saturated cut is never admitted.
[Timeouts & Locks] Pure bounded arithmetic over a finite flow list.
[Side-Effect Policy] Fresh immutable outputs; no I/O or global mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.schema.common import ModelingError


def _positive(name: str, value: Any, *, strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"xpyd_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or (number <= 0.0 if strict else number < 0.0):
        raise ModelingError(f"xpyd_{name}_invalid:{value}")
    return number


def _int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError(f"xpyd_{name}_invalid:{value}")
    return value


def _fraction(name: str, value: Any) -> float:
    number = _positive(name, value, strict=False)
    if number > 1.0:
        raise ModelingError(f"xpyd_{name}_invalid:{value}")
    return number


@dataclass(frozen=True)
class Flow:
    id: str
    coefficients: tuple[tuple[str, float], ...]
    weight: float = 1.0
    demand: float = float("inf")
    priority: int = 0

    def __post_init__(self) -> None:
        _positive("flow_weight", self.weight)
        if not math.isfinite(self.demand) and self.demand != float("inf"):
            raise ModelingError(f"xpyd_flow_demand_invalid:{self.demand}")
        if self.demand < 0:
            raise ModelingError(f"xpyd_flow_demand_invalid:{self.demand}")
        for _, value in self.coefficients:
            _positive("flow_coefficient", value, strict=False)


def progressive_max_min(resources: Mapping[str, float],
                        flows: Sequence[Flow]) -> dict[str, Any]:
    """Weighted progressive filling with priority classes and deadlock state."""
    capacities = {str(name): _positive("resource", value, strict=False)
                  for name, value in resources.items()}
    rates = {flow.id: 0.0 for flow in flows}
    saturated: set[str] = set()
    deadlock = False
    committed = {name: 0.0 for name in capacities}
    for priority in sorted({flow.priority for flow in flows}):
        group = [flow for flow in flows if flow.priority == priority]
        used = {name: committed[name] for name in capacities}
        for flow in group:
            for name, value in flow.coefficients:
                used[name] += value * rates[flow.id]
        active = set(flow.id for flow in group if rates[flow.id] < flow.demand)
        while active:
            best = None
            for name, capacity in capacities.items():
                remaining = capacity - used[name]
                denominator = sum(
                    dict(flow.coefficients).get(name, 0.0) * flow.weight
                    for flow in group if flow.id in active)
                if denominator <= 0:
                    continue
                delta = max(0.0, remaining / denominator)
                best = delta if best is None else min(best, delta)
            if best is None:
                deadlock = True
                break
            if best <= 1e-15:
                blocked = set()
                for flow in group:
                    if flow.id not in active:
                        continue
                    touches = [name for name, value in flow.coefficients if value > 0]
                    if any(used[name] >= capacities[name] - 1e-9 for name in touches):
                        blocked.add(flow.id)
                        saturated.update(name for name in touches
                                         if used[name] >= capacities[name] - 1e-9)
                    else:
                        deadlock = True
                active -= blocked
                if deadlock:
                    break
                continue
            progressed = False
            frozen = set()
            for flow in group:
                if flow.id not in active:
                    continue
                old = rates[flow.id]
                new = min(flow.demand, old + best * flow.weight)
                increase = max(0.0, new - old)
                for name, value in flow.coefficients:
                    used[name] += value * increase
                rates[flow.id] = new
                if increase > 1e-15:
                    progressed = True
                if rates[flow.id] >= flow.demand - 1e-12:
                    active.discard(flow.id)
                for name, value in flow.coefficients:
                    if value > 0 and used[name] >= capacities[name] - 1e-9:
                        saturated.add(name)
                        frozen.add(flow.id)
            if not progressed:
                blocked = set()
                for flow in group:
                    if flow.id not in active:
                        continue
                    touches = [name for name, value in flow.coefficients if value > 0]
                    if any(used[name] >= capacities[name] - 1e-9 for name in touches):
                        blocked.add(flow.id)
                    else:
                        deadlock = True
                active -= blocked
                if deadlock:
                    break
            active -= frozen
        for flow in group:
            for name, value in flow.coefficients:
                committed[name] += value * rates[flow.id]
    return {"rates": rates, "saturated": tuple(sorted(saturated)),
            "deadlock": deadlock, "feasible": not deadlock,
            "resources": dict(sorted(capacities.items()))}



__all__ = ["Flow", "kv_transfer_contention", "progressive_max_min", "queue_wait_ms"]


def queue_wait_ms(utilization: float, service_ms: float) -> float:
    """Declared M/D/1 mean wait; ``inf`` at a non-stable utilization."""
    if utilization <= 0.0:
        return 0.0
    if utilization >= 1.0:
        return float("inf")
    return utilization * service_ms / (2.0 * (1.0 - utilization))


