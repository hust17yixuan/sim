"""Closed-world counterfactual advisor for MindIE simulation Cases.

[Role]
  Public facade for ``load_case_from_profile`` and ``advise_case``: ingest
  measured profiling artifacts, evaluate the baseline causal simulator, expose
  four-dimensional and resource attribution plus calibration residuals, then rank
  guarded counterfactual runtime mutations and optional fixed-plan resource or
  capacity interventions.

[Invariants]
  * Every counterfactual is validated and evaluated by the causal engine; no
    model-name multiplier or observed-latency leakage enters prediction.
  * Resource interventions re-evaluate the same lowered plan; a replanned
    candidate is reported through its own complete evaluation.
  * Caller Cases, mappings, and profile records are never mutated in place.
  * Missing observations stay ``None``; emitted report floats are finite.

[Timeouts & Locks]
  Bounded local profile reads and in-memory graph evaluation; no network, NPU,
  subprocess, sleep, or file locks.

[Side-Effect Policy]
  Pure in-memory report construction; the CLI writes only an explicit ``-o``
  path through the shared JSON emitter.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from .advise_catalog import _evaluate_intervention, candidate_specs, rank_key
from .advise_core import _baseline_with_plan, _calibration, _config_for, _emitted_tokens
from .prediction.efficiency import evaluate_resource_counterfactual
from .prediction.structure import parse_case
from .advise_util import _has_observed_measurement
from .profile_ingest import _merge_observed_profiles, load_case_from_profile
from .schema.case import Case as CaseSpec

ADVISE_SCHEMA = "mindie-sim.advise/1"


def _normalize_resource_counterfactuals(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, (str, Mapping)):
        entries = [raw]
    elif isinstance(raw, Sequence):
        entries = list(raw)
    else:
        raise ValueError(f"counterfactual_resource_invalid:{type(raw).__name__}")
    records = []
    for entry in entries:
        if isinstance(entry, str):
            text = entry.strip()
            resource, _, factor_text = text.partition(":")
            factor = float(factor_text) if factor_text else 2.0
        elif isinstance(entry, Mapping):
            unknown = set(entry) - {"resource", "factor"}
            if unknown:
                raise ValueError("counterfactual_resource_unknown_fields:" + ",".join(sorted(unknown)))
            resource = str(entry.get("resource", "")).strip()
            raw_factor = entry.get("factor", 2.0)
            if isinstance(raw_factor, bool):
                raise ValueError(f"counterfactual_resource_factor_must_be_positive:{raw_factor}")
            factor = float(raw_factor)
        else:
            raise ValueError(f"counterfactual_resource_invalid:{type(entry).__name__}")
        if not resource:
            raise ValueError("counterfactual_resource_requires_resource")
        if factor <= 0.0:
            raise ValueError(f"counterfactual_resource_factor_must_be_positive:{factor}")
        records.append({"resource": resource, "factor": factor})
    return records


def _capacity_consistency(advise_capacity: Mapping[str, Any],
                          counterfactual_capacity: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {
        "kv_bytes": (advise_capacity.get("kv_bytes_per_seq"),
                     counterfactual_capacity.get("kv_bytes_per_sequence")),
        "capacity_headroom_bytes": (advise_capacity.get("capacity_headroom_bytes"),
                                    counterfactual_capacity.get("capacity_headroom_bytes")),
        "lifecycle_headroom_bytes": (advise_capacity.get("lifecycle_capacity_headroom_bytes"),
                                     counterfactual_capacity.get("lifecycle_capacity_headroom_bytes")),
        "joint_headroom_bytes": (advise_capacity.get("joint_capacity_headroom_bytes"),
                                 counterfactual_capacity.get("joint_capacity_headroom_bytes")),
        "b_vram_max_sync": (advise_capacity.get("vram_max_sequences_per_dp_sync"),
                            counterfactual_capacity.get("b_vram_max_sync")),
        "b_vram_max_async": (advise_capacity.get("vram_max_sequences_per_dp_async"),
                             counterfactual_capacity.get("b_vram_max_async")),
        "b_lifecycle_sync": (advise_capacity.get("vram_max_sequences_per_dp_lifecycle_sync"),
                             counterfactual_capacity.get("b_vram_max_lifecycle_sync")),
        "b_lifecycle_async": (advise_capacity.get("vram_max_sequences_per_dp_lifecycle_async"),
                              counterfactual_capacity.get("b_vram_max_lifecycle_async")),
        "joint_sync": (advise_capacity.get("vram_max_sequences_per_dp_joint_sync"),
                       counterfactual_capacity.get("b_vram_max_joint_sync")),
        "joint_async": (advise_capacity.get("vram_max_sequences_per_dp_joint_async"),
                        counterfactual_capacity.get("b_vram_max_joint_async")),
    }
    matched = all(
        left is not None and right is not None and math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
        for left, right in pairs.values()
    )
    advise_stage_kv = [row.get("kv_bytes_per_sequence") for row in advise_capacity.get("capacity_by_stage", ())]
    counter_stage_kv = [row.get("kv_bytes_per_sequence") for row in counterfactual_capacity.get("capacity_by_stage", ())]
    stage_matched = (len(advise_stage_kv) == len(counter_stage_kv) and all(
        left is not None and right is not None and math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
        for left, right in zip(advise_stage_kv, counter_stage_kv)))
    return {"fields": {name: {"advise": left, "counterfactual": right}
                       for name, (left, right) in pairs.items()},
            "stage_kv_bytes": {"advise": advise_stage_kv, "counterfactual": counter_stage_kv},
            "matched": bool(matched and stage_matched)}


def _resource_counterfactuals(case: CaseSpec, baseline: Mapping[str, Any],
                              plan: Mapping[str, Any],
                              specs: Sequence[Mapping[str, Any]], *,
                              output_budget: int,
                              emitted_tokens_per_seq: float) -> list[dict[str, Any]]:
    records = []
    advise_capacity = dict((baseline.get("attribution") or {}).get("capacity") or {})
    for spec in specs:
        result = evaluate_resource_counterfactual(
            case, resource=str(spec["resource"]), factor=float(spec["factor"]),
            plan=plan, include_submission=True, overlap_async_scheduler=True,
            include_capacity=True, output_budget=output_budget,
            emitted_tokens_per_seq=emitted_tokens_per_seq,
        )
        common_step = float(result["baseline"]["graph_duration_ms"])
        advise_step = float(baseline["step_duration_ms"])
        result["evaluation_consistency"]["advise_baseline_step_ms"] = advise_step
        result["evaluation_consistency"]["matched_advise_baseline"] = math.isclose(
            common_step, advise_step, rel_tol=1e-12, abs_tol=1e-12)
        capacity_consistency = _capacity_consistency(
            advise_capacity, result["baseline"].get("capacity") or {})
        result["evaluation_consistency"]["capacity"] = capacity_consistency
        if float(spec["factor"]) == 1.0:
            base_capacity = result["baseline"].get("capacity") or {}
            changed_capacity = result["counterfactual"].get("capacity") or {}
            same_capacity = all(
                base_capacity.get(field) is not None and changed_capacity.get(field) is not None
                and math.isclose(float(base_capacity[field]), float(changed_capacity[field]),
                                 rel_tol=1e-12, abs_tol=1e-9)
                for field in ("kv_bytes_per_sequence", "capacity_headroom_bytes",
                              "lifecycle_capacity_headroom_bytes",
                              "joint_capacity_headroom_bytes",
                              "b_vram_max_sync", "b_vram_max_async",
                              "b_vram_max_lifecycle_sync", "b_vram_max_lifecycle_async",
                              "b_vram_max_joint_sync", "b_vram_max_joint_async"))
            result["evaluation_consistency"]["factor_one_capacity_matches"] = bool(
                capacity_consistency["matched"] and same_capacity)
        result["evaluation_consistency"]["note"] = (
            "baseline and counterfactual both use the same lowered fixed plan; "
            "capacity fields use the advise output budget and admission window")
        records.append(result)
    return records


def _merge_observed(case_dict: dict[str, Any], observed: Mapping[str, Any]) -> None:
    if not observed or not _has_observed_measurement(observed):
        return
    runtime = case_dict.setdefault("runtime", {})
    raw_meta = runtime.get("metadata") or {}
    meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else dict(raw_meta)
    merged = _merge_observed_profiles(meta.get("observed_profile"), observed)
    if merged:
        meta["observed_profile"] = merged
    runtime["metadata"] = meta


def advise_case(
    case_or_path: CaseSpec | Mapping[str, Any] | str | Path | None = None,
    *,
    from_profile: str | Path | Mapping[str, Any] | None = None,
    phase: str = "decode",
    scope: str = "body",
    context: int | None = None,
    sequences: int | None = None,
    tokens: int | None = None,
    sla_tpot_ms: float | None = None,
    sla_ttft_ms: float | None = None,
    max_interventions: int = 10,
    counterfactual_resources: str | Mapping[str, Any] | Sequence[str | Mapping[str, Any]] | None = None,
    **profile_overrides: Any,
) -> dict[str, Any]:
    """Evaluate a baseline Case and rank closed-world counterfactual interventions."""
    if isinstance(max_interventions, bool) or not isinstance(max_interventions, int) or max_interventions < 0:
        raise ValueError(f"max_interventions_must_be_nonnegative_int:{max_interventions}")
    resource_specs = _normalize_resource_counterfactuals(counterfactual_resources)
    if case_or_path is None and from_profile is None:
        raise ValueError("advise_requires_case_or_from_profile")
    profile_case = None
    if from_profile is not None:
        profile_case = load_case_from_profile(from_profile, **profile_overrides)
    if case_or_path is None:
        source = profile_case
    elif isinstance(case_or_path, (str, Path)):
        source = load_case_from_profile(case_or_path, **profile_overrides)
    elif isinstance(case_or_path, CaseSpec):
        source = (load_case_from_profile(case_or_path, **profile_overrides)
                  if profile_overrides else case_or_path.to_dict())
    elif isinstance(case_or_path, Mapping):
        if {"machine", "model", "workload"} <= set(case_or_path) and not profile_overrides:
            source = deepcopy(dict(case_or_path))
        else:
            source = load_case_from_profile(case_or_path, **profile_overrides)
    else:
        raise ValueError(f"advise_invalid_case_source:{type(case_or_path).__name__}")
    if source is None:
        raise ValueError("advise_requires_case_or_from_profile")
    case_dict = deepcopy(dict(source)) if isinstance(source, Mapping) else dict(source)
    if profile_case is not None and source is not profile_case:
        raw_meta = dict(profile_case.get("runtime", {}).get("metadata") or ())
        observed = raw_meta.get("observed_profile")
        if isinstance(observed, Mapping):
            _merge_observed(case_dict, observed)
        prof_prov = raw_meta.get("profiler_provenance")
        if isinstance(prof_prov, Mapping) and prof_prov:
            rt_meta = dict(case_dict.setdefault("runtime", {}).get("metadata") or {})
            existing_prov = rt_meta.get("profiler_provenance")
            rt_meta["profiler_provenance"] = {
                **(dict(existing_prov) if isinstance(existing_prov, Mapping) else {}),
                **dict(prof_prov),
            }
            case_dict["runtime"]["metadata"] = rt_meta
    case = parse_case(case_dict)
    cfg = _config_for(case, phase, scope, context, sequences, tokens, sla_tpot_ms, sla_ttft_ms)
    baseline, baseline_plan = _baseline_with_plan(case, cfg, include_plan=True)
    raw_meta = dict(case.runtime.metadata)
    observed = raw_meta.get("observed_profile")
    observed_vs_predicted = _calibration(
        baseline, dict(observed), cfg=cfg, dp=int(case.runtime.parallel.dp), case=case,
    ) if isinstance(observed, Mapping) and observed else {
        "available": False, "reason": "no_observed_profile_metadata"}
    inventory = candidate_specs(case, baseline, cfg, case_dict)[:max_interventions]
    recommendations = [_evaluate_intervention(spec, case_dict, baseline, case, cfg)
                       for spec in inventory]
    recommendations.sort(key=rank_key)
    top = next((record for record in recommendations
                if record.get("applicable_and_recommended")), None)
    if top is None and recommendations:
        top = recommendations[0]
    public_baseline = {key: baseline[key] for key in (
        "step_duration_ms", "operator_sum_ms", "tpot_p50_ms", "ttft_p50_ms",
        "sla_goodput_tok_s", "b_vram_max_sync", "b_vram_max_async", "dominant_bottleneck")}
    resource_records = _resource_counterfactuals(
        case, baseline, baseline_plan, resource_specs,
        output_budget=cfg.output_tokens,
        emitted_tokens_per_seq=_emitted_tokens(case, cfg))
    provenance = raw_meta.get("profiler_provenance")
    return {
        "schema": ADVISE_SCHEMA,
        "status": "completed",
        "geometry": {"phase": cfg.phase, "scope": cfg.scope, "context": cfg.context,
                     "sequences": cfg.sequences, "tokens": cfg.tokens,
                     "output_tokens": cfg.output_tokens, "prefix_tokens": cfg.prefix_tokens,
                     "sla_tpot_ms": cfg.sla_tpot_ms, "sla_ttft_ms": cfg.sla_ttft_ms},
        "case": {"model": case.model.name,
                 "hardware": case.machine.sku or case.machine.name,
                 "parallel": {"tp": case.runtime.parallel.tp, "dp": case.runtime.parallel.dp,
                              "ep": case.runtime.parallel.ep, "pp": case.runtime.parallel.pp}},
        "baseline": public_baseline,
        "attribution_4d": baseline["attribution"],
        "resource_attribution": baseline["analysis"].get("attribution"),
        "legacy_service_regime_diagnostics": baseline["analysis"].get("legacy_service_regime_diagnostics"),
        "resource_counterfactuals": resource_records,
        "evaluation_boundary": {
            "recommendations": "each_candidate_replanned_and_fully_evaluated",
            "resource_counterfactuals": "same_lowered_plan_baseline_and_intervention",
        },
        "observed_vs_predicted": observed_vs_predicted,
        "recommendations": recommendations,
        "top_recommendation": top,
        "profile_provenance": dict(provenance) if isinstance(provenance, Mapping) else {},
        "max_interventions": max_interventions,
        "limitations": [
            "counterfactual timing is causal-simulation output, not a measured hardware claim",
            "SLA goodput uses VRAM capacity and step latency; the exact solver is capped for large contexts",
        ],
    }


__all__ = ["advise_case", "load_case_from_profile"]
