"""Causal phase lowering for a contracted physical plan node.

[Role] Derive the executed phase work and the destination-buffer acquisition
  points of one fused ``PlanNode`` from the same conserved service that its
  reported totals use.
[Invariants] Every resource reported by ``PlanNode.work`` is executed by some
  phase, so ``Engine.summary()['serviced']`` matches the report.  A destination
  buffer is acquired by the earliest phase that writes into it: L1 by the
  Vector/Cube operand handoff, L0C by the Cube body.
[Timeouts & Locks] Finite dictionary scanning; no I/O, clocks or locks.
[Side-Effect Policy] Pure functions over immutable candidate/node records.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

_PHASE_ORDER = ("operand_read", "prologue", "prologue_handoff", "cube",
                "handoff", "epilogue", "vector", "reduce", "merge",
                "finalize", "output_write")


def _work(totals: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), float(value)) for key, value in totals.items()
                        if abs(float(value)) > 1e-12))


def _traffic_split(candidate) -> tuple[float, float]:
    reads = sum(value.candidate_read_bytes for value in candidate.analysis.traffic.values)
    writes = sum(value.candidate_write_bytes + value.new_spill_bytes
                 for value in candidate.analysis.traffic.values)
    return float(reads), float(writes)


def conserved_fusion_work(member_nodes: tuple,
                          candidate) -> tuple[tuple[str, float], ...]:
    """Conserve every declared resource except the proven eliminated HBM."""
    declared_hbm = sum(node.work_dict().get("hbm", 0.0) for node in member_nodes)
    if declared_hbm + 1e-6 < float(candidate.hbm_before):
        raise ValueError("fusion_ledger_baseline_mismatch")
    # Additive metadata HBM work is executed service, not an eliminable
    # intermediate access; only the typed ledger's proven bytes may vanish.
    extra_hbm = max(0.0, declared_hbm - float(candidate.hbm_before))
    totals: dict[str, float] = {}
    for node in member_nodes:
        for resource, amount in node.work:
            if resource != "hbm":
                totals[resource] = totals.get(resource, 0.0) + float(amount)
    # Phase-only work (priced Cube/Vector handoff, cross-tile merge) is real
    # executed service; the aggregate plan node must report it too.
    phase_totals: dict[str, float] = {}
    for _phase, phase_demand in candidate.analysis.phase_work:
        for resource, amount in phase_demand.items():
            phase_totals[resource] = phase_totals.get(resource, 0.0) + float(amount)
    for resource, amount in phase_totals.items():
        if resource == "hbm":
            continue
        if amount > totals.get(resource, 0.0) + 1e-9:
            totals[resource] = float(amount)
    totals["hbm"] = float(candidate.hbm_after) + extra_hbm
    if abs(declared_hbm - totals["hbm"] - candidate.hbm_saved) > 1e-6:
        raise ValueError("fusion_hbm_delta_not_conserved")
    return _work(totals)


def _present(names: Iterable[str]) -> set[str]:
    return set(names)


def _resource_phase(resource: str, names: set[str], last: str) -> str:
    candidates = {
        "vector": ("epilogue", "vector", "merge", "prologue"),
        "cube": ("cube",),
        "mte1": ("prologue_handoff", "handoff", "operand_read", "cube", "merge"),
        "mte2": ("operand_read", "prologue_handoff", "cube"),
        "mte3": ("output_write", "epilogue"),
        "fixpipe": ("handoff", "epilogue", "cube"),
        "host": ("prologue", "vector", "epilogue"),
    }.get(resource, ())
    for name in candidates:
        if name in names:
            return name
    return last


def _ensure(phases: list[list[Any]], phase: str) -> dict[str, float]:
    for name, demand in phases:
        if name == phase:
            return demand
    demand: dict[str, float] = {}
    phases.append([phase, demand])
    return demand


def reconciled_phase_work(candidate,
                          conserved: tuple[tuple[str, float], ...]
                          ) -> tuple[tuple[str, dict[str, float]], ...]:
    """Distribute the conserved report work into the causal executed phases."""
    raw_phases = [(phase, dict(demand))
                  for phase, demand in getattr(candidate.analysis, "phase_work", ())]
    if not raw_phases:
        raw_phases = [("vector", {})]
    phases: list[list[Any]] = [[name, demand] for name, demand in raw_phases]
    names = {name for name, _ in phases}
    last = _PHASE_ORDER[max(
        (index for index, name in enumerate(_PHASE_ORDER) if name in names),
        default=len(_PHASE_ORDER) - 1)]
    totals: dict[str, float] = {}
    for _phase, demand in phases:
        for resource, amount in demand.items():
            totals[resource] = totals.get(resource, 0.0) + float(amount)
    reads, writes = _traffic_split(candidate)
    for resource, amount in conserved:
        delta = float(amount) - totals.get(resource, 0.0)
        if delta <= 1e-9:
            continue
        if resource == "hbm" and "operand_read" in names and "output_write" in names:
            denominator = reads + writes
            read_part = delta * reads / denominator if denominator > 0 else 0.0
            _ensure(phases, "operand_read")["hbm"] = (
                _ensure(phases, "operand_read").get("hbm", 0.0) + read_part)
            _ensure(phases, "output_write")["hbm"] = (
                _ensure(phases, "output_write").get("hbm", 0.0)
                + (delta - read_part))
        else:
            phase = _resource_phase(resource, names, last)
            demand = _ensure(phases, phase)
            demand[resource] = demand.get(resource, 0.0) + delta
        totals[resource] = float(amount)
    cleaned = []
    for name, demand in phases:
        kept = {resource: float(value) for resource, value in demand.items()
                if abs(float(value)) > 1e-12}
        if kept:
            cleaned.append((name, kept))
    return tuple(cleaned)


def phase_acquisitions(candidate) -> tuple[tuple[str, tuple[
        tuple[str, str, int, tuple[int, ...]], ...]], ...]:
    """Acquire each declared Cube buffer at the phase that first writes it."""
    allocations = tuple(getattr(candidate.analysis.worksheet, "cube_allocations", ()))
    if not allocations:
        return ()
    names = {phase for phase, _ in getattr(candidate.analysis, "phase_work", ())}
    grouped: dict[str, list[tuple[str, str, int, tuple[int, ...]]]] = {}
    for allocation in allocations:
        if allocation.pool == "l1":
            phase = ("prologue_handoff" if "prologue_handoff" in names else
                     "operand_read" if "operand_read" in names else "cube")
        elif allocation.pool == "ub":
            phase = ("handoff" if "handoff" in names else
                     "epilogue" if "epilogue" in names else "cube")
        else:
            phase = "cube"
        grouped.setdefault(phase, []).append(
            (allocation.name, allocation.pool, allocation.total_bytes,
             tuple(allocation.units)))
    return tuple((phase, tuple(items)) for phase, items in grouped.items())
