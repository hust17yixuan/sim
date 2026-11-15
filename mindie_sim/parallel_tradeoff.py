"""Fixed-world TP <-> EP trade-off sweep with an explicit dQ/dEP knee.

[Role] Re-run the EP knee for each declared TP split at one physical world size
  and report the marginal throughput gain per added EP rank before it drops.
[Invariants] Every row is a fixed-world candidate; unqualified routes and
  infeasible rows never become the winner; marginal = dQ/dEP.
[Timeouts & Locks] Pure bounded arithmetic; no I/O or device context.
[Side-Effect Policy] Fresh report; caller Case untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mindie_sim.parallel_ep_knee import DEFAULT_EP_CANDIDATES, sweep_ep_knee
from mindie_sim.schema.common import ModelingError


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"parallel_explorer_{name}_invalid:{value}")
    return value


def sweep_tp_ep_tradeoff(
    case: CaseSpec | Mapping[str, Any], *, world_npus: int,
    tp_candidates: Sequence[int] = (1, 2, 4, 8),
    ep_candidates: Sequence[int] = DEFAULT_EP_CANDIDATES,
    batch_sequences: int | None = None, context_tokens: int | None = None,
    routing_skew: float = 1.0, fabric: Any = None, mc2: bool | None = None,
    hccl_buffsize_mb: float | None = None, scaling_mode: str = "fixed_per_dp_batch",
    host_overlap: float = 0.5, overlap_mode: str = "serial",
    microbatches: int = 1, allow_unqualified: bool = False) -> dict[str, Any]:
    """Fixed-world ``TP <-> EP`` trade-off with an explicit ``dQ/dEP`` knee."""
    world = _int("world_npus", world_npus)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for tp in sorted({_int("tp_candidate", value) for value in tp_candidates}):
        try:
            report = sweep_ep_knee(
                case, ep_candidates=ep_candidates, batch_sequences=batch_sequences,
                context_tokens=context_tokens, routing_skew=routing_skew,
                fabric=fabric, mc2=mc2, hccl_buffsize_mb=hccl_buffsize_mb,
                scaling_mode=scaling_mode, fixed_world_npus=world,
                host_overlap=host_overlap, tp_override=tp,
                overlap_mode=overlap_mode, microbatches=microbatches,
                allow_unqualified=allow_unqualified)
        except (ModelingError, ValueError) as exc:
            reports.append({"tp": tp, "status": "invalid", "reason": str(exc),
                            "rows": []})
            continue
        for row in report["rows"]:
            row["tradeoff_tp"] = tp
            row["tradeoff_world"] = world
        rows.extend(report["rows"])
        reports.append({"tp": tp, "status": report["status"],
                        "ep_star": report["ep_star"],
                        "knee_ep": report["knee_ep"], "rows": report["rows"]})
    marginal: list[tuple[int, int, float]] = []
    for tp in sorted({int(row["tradeoff_tp"]) for row in rows}):
        ordered = sorted((row for row in rows
                          if int(row["tradeoff_tp"]) == tp and row.get("feasible")),
                         key=lambda row: int(row["ep"]))
        previous = None
        for row in ordered:
            if previous is not None:
                delta_ep = int(row["ep"]) - int(previous["ep"])
                if delta_ep > 0:
                    gain = (float(row["tps"]) - float(previous["tps"])) / delta_ep
                    row["marginal_gain_per_ep"] = gain
                    marginal.append((tp, int(row["ep"]), gain))
            previous = row
    feasible = [row for row in rows if row.get("feasible")
                and row.get("route_qualified", True)]
    best = max(feasible, key=lambda row: (float(row["tps"]), -int(row["ep"])),
               default=None)
    knee = None
    if best is not None:
        same_tp = sorted((row for row in feasible
                          if int(row["tradeoff_tp"]) == int(best["tradeoff_tp"])),
                         key=lambda row: int(row["ep"]))
        for previous, row in zip(same_tp, same_tp[1:]):
            if float(row["tps"]) < float(previous["tps"]):
                knee = int(row["ep"])
                break
    return {"schema": "mindie-sim.tp-ep-tradeoff/1", "world_npus": world,
            "tp_candidates": tuple(sorted({int(row["tradeoff_tp"]) for row in rows})),
            "rows": rows, "reports": reports,
            "feasible_rows": feasible,
            "best_configuration": best,
            "ep_star_throughput": (int(best["ep"]) if best is not None else None),
            "tradeoff_knee_ep": knee,
            "marginal_gains_per_ep": tuple(marginal),
            "status": "feasible" if best is not None else "all_candidates_infeasible",
            "selection": "max_tps_before_first_decrease; dQ/dEP per TP lane"}



__all__ = ["sweep_tp_ep_tradeoff"]
