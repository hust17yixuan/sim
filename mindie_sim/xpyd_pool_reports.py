"""Typed pool rejection records and placed-instance memory evidence.

[Role] Build reviewable failure rows without terminating candidate search.
[Invariants] Invalid geometry is unqualified; pair failures retain both specs.
[Timeouts & Locks] Pure bounded record construction; no locks or I/O.
[Side-Effect Policy] Fresh dictionaries only; inputs are not mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def geometry_rejection(reason: str, total: int, isl: int, osl: int) -> dict[str, Any]:
    """Public result when raw shared geometry cannot safely be normalized."""
    return {"schema": "mindie-sim.xpyd-pools/3", "total_npus": total,
            "isl": isl, "osl": osl, "kv_transfer": None,
            "prefill_configs": [], "decode_configs": [], "rows": [],
            "feasible_rows": [], "pareto_frontier": [],
            "best_configuration": None, "qualified": False,
            "route_qualified": False, "route_evidence": reason,
            "reason": reason, "qualification_reason": reason,
            "status": "no_feasible_configuration",
            "slo_qualification_mode": "analytic_md1_mean",
            "tail_quantile_certified": False}


def instance_evidence(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        memory = item.get("memory")
        result.append({
            "device_ids": list(item.get("device_ids", ())),
            "feasible": bool(item.get("feasible")),
            "reason": item.get("reason"),
            "per_rank_bytes": [list(row) for row in
                               getattr(memory, "per_rank_bytes", ())],
        })
    return result

def pool_failure(total: int, n_p: int, n_d: int, wp: int, wd: int, reason: str,
             p_ids: Sequence[int] = (), d_ids: Sequence[int] = (),
             p_inst: Sequence[Mapping[str, Any]] = (),
             d_inst: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return {
        "prefill_instance_rows": instance_evidence(p_inst),
        "decode_instance_rows": instance_evidence(d_inst),
        "n_prefill": n_p, "n_decode": n_d, "pd_ratio": f"{n_p}P:{n_d}D",
        "used_npus": n_p * wp + n_d * wd,
        "idle_npus": total - n_p * wp - n_d * wd,
        "prefill_world": wp, "decode_world": wd,
        "prefill_device_ids": list(p_ids), "decode_device_ids": list(d_ids),
        "admitted_rps": 0.0, "goodput_rps": 0.0,
        "goodput_output_tps": 0.0, "goodput_total_tps": 0.0,
        "goodput_tps": 0.0, "goodput_tps_per_npu": 0.0,
        "ttft_ms": float("inf"), "tpot_ms": float("inf"),
        "utilization": 0.0, "rho_prefill": 0.0, "rho_decode": 0.0,
        "decode_effective_rps": 0.0, "d_hbm_jitter": float("inf"),
        "feasible": False, "slo_feasible": False, "qualified": False,
        "reason": reason,
    }

