"""Same-plan SOL comparison and bounded single-feature counterfactuals.

[Role]
  Reuse one selected plan for the Engine prediction and the SOL bound, then
  evaluate at most one feature change at a time from that same baseline.  A
  candidate Case is re-materialized and re-lowered from scratch; no stale
  pre-mutation plan object is reused.  It never refits calibration, re-solves
  SLA, or merges unrelated plans.

[Invariants]
  * `compare_launch` refuses failed/needs-input reports; a partial Case
    requires `allow_partial` and carries `conditional_subset` coverage.
  * The Engine prediction and SOL consume the same target graph, external
    input boundary and recipe expansion.  A body step binds its declared
    `analysis_input` in both paths.
  * Counterfactuals keep workload/hardware/quantization/implementation fixed,
    change one named feature, revalidate, and re-run lowering and the Engine.
    A true repartition that is unavailable makes the candidate `inapplicable`,
    never a stale hand-edited plan.  `saved_ms` and `sol_saved_ms` are baseline
    minus candidate; negative deltas are retained.
  * Every evaluated candidate satisfies finite nonnegative times, matching
    scope/plan identity, and `candidate_SOL <= candidate_prediction`; a
    violation becomes a failed candidate with null deltas and a reason.
    Custom `plan_ops` are still re-lowered through the same implementation path.
  * Request metrics need an explicit arrival/first-commit/decode trace and use
    fractional expected commits.  At most 16 candidates run; deltas are not
    additive.
[Timeouts & Locks]
  Finite in-memory work; the caller's Engine budget applies.
[Side-Effect Policy]
  No I/O, devices, environment reads or input mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math

from .feature_reports import (Context, Counterfactual, Patch, ReportValidationError,
                              validate_sol_report)
from .prediction.graph_analysis import graph_costs
from .prediction.sol import compare_sol
from .prediction.sol_work import project_ops
from .prediction.operator_analysis import target_graphs
from .schema.case import Case
from .schema.common import ModelingError

FEATURES = ("ACLGraph", "MLAPO", "MC2", "MTP", "Prefix Caching", "PCP/DCP",
            "Multi-Stream Overlap")
_FEATURE_ID = {"ACLGraph": "aclgraph", "MLAPO": "mlapo", "MC2": "mc2", "MTP": "mtp",
               "Prefix Caching": "prefix_caching", "PCP/DCP": "pcp_dcp",
               "Multi-Stream Overlap": "multi_stream"}
_MTP_METADATA = {"draft_mode", "draft_proposal_tokens", "speculative"}


class LaunchComparisonError(ValueError):
    """The launch report cannot be compared under the requested options."""


class _Inapplicable(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _graphs(ops):
    return ops if isinstance(ops, Mapping) else {0: tuple(ops)}


def _external(scope):
    return {"analysis_input": 0} if scope == "body" else None


def _predict(case, ops, include_submission=False, scope="model"):
    return graph_costs(case, _graphs(ops), compact=True, trace_limit=0,
                       include_submission=include_submission,
                       external_inputs=_external(scope))["duration_ms"]


def _runtime(case, **changes):
    return replace(case, runtime=replace(case.runtime, **changes))


def _strip(ops, keys):
    return {owner: tuple(replace(op, metadata=tuple(
        (key, value) for key, value in dict(op.metadata).items() if key not in keys))
        for op in items) for owner, items in _graphs(ops).items()}


def _meta(op, key, default=None):
    return dict(op.metadata).get(key, default)


def _tagged(ops, predicate):
    return [op for items in _graphs(ops).values() for op in items if predicate(op)]


def _replace_ops(ops, transform):
    return {owner: tuple(transform(op) for op in items)
            for owner, items in _graphs(ops).items()}


def _aclgraph(case, ops):
    if (str(case.runtime.graph_mode) == "none"
            and str(case.runtime.graph_capture_mode) == "none"
            and not case.runtime.graph_capture_keys):
        raise _Inapplicable("baseline has no capture/replay to disable")
    patches = [
        Patch("/runtime/graph_mode", case.runtime.graph_mode, "none",
              "Disable ACLGraph capture."),
        Patch("/runtime/graph_capture_mode", case.runtime.graph_capture_mode, "none",
              "Disable ACLGraph capture and its capture-key set."),
    ]
    metadata = tuple((key, value) for key, value in case.runtime.metadata
                     if key != "captured_execution_mode")
    candidate = _runtime(case, graph_mode="none", graph_capture_mode="none",
                         graph_capture_keys=(), piecewise_capture_keys=(),
                         mixed_mode="none", metadata=metadata)
    return candidate, _strip(ops, {"graph_replay_scope", "capture_keys",
                                   "graph_capture_mode"}), patches


def _mlapo(case, ops):
    lowerings = dict(case.runtime.kernels.get("lowerings", {}))
    if "mla_prolog" not in lowerings:
        raise _Inapplicable("baseline does not select an mla_prolog lowering")
    selected = _tagged(ops, lambda op: op.family == "mla_prolog"
                       or "mla_prolog" in str(_meta(op, "lowering", ""))
                       or "mla_prolog" in op.op_id)
    if not selected:
        raise _Inapplicable("selected plan has no mla_prolog program")
    kernels = {**case.runtime.kernels,
               "lowerings": {key: value for key, value in lowerings.items()
                             if key != "mla_prolog"}}
    patches = [Patch("/runtime/kernels/lowerings/mla_prolog",
                     lowerings["mla_prolog"], None,
                     "Disable the version-pinned MLAPO lowering.")]
    candidate = _runtime(case, kernels=kernels)
    return candidate, _strip(ops, {"lowering", "mla_prolog", "mlapo_hbm_saved_bytes",
                                   "mlapo_link_saved_bytes", "mlapo_mte_saved_bytes"}), patches


def _mc2(case, ops):
    if case.runtime.moe_recipe != "mc2":
        raise _Inapplicable("baseline does not select the MC2 recipe")
    selected = _tagged(ops, lambda op: _meta(op, "moe_collective") is True
                       or op.family in {"dispatch", "combine", "alltoall"})
    if not selected:
        raise _Inapplicable("selected plan has no MoE collective to re-price")
    candidate = _runtime(case, moe_recipe="alltoall", deepep_high_throughput=False)
    patches = [Patch("/runtime/moe_recipe", "mc2", "alltoall", "Disable MC2.")]
    return candidate, _strip(ops, {"mc2_link_saved_bytes", "mc2_hbm_saved_bytes",
                                   "mc2_mte_saved_bytes", "mc2"}), patches


def _mtp(case, ops):
    draft = _tagged(ops, lambda op: _meta(op, "mtp_draft") is True
                    or op.family in {"mtp_draft", "draft"})
    if not draft:
        raise _Inapplicable("selected plan has no draft program")
    metadata = tuple((key, value) for key, value in case.runtime.metadata
                     if key not in _MTP_METADATA)
    candidate = _runtime(case, mtp_acceptance_length=1.0,
                         mtp_acceptance_by_output_tokens=(), metadata=metadata)
    patches = [Patch("/runtime/mtp_acceptance_length",
                     case.runtime.mtp_acceptance_length, 1.0,
                     "Remove speculative draft geometry and acceptance.")]
    if not draft:
        return candidate, ops, patches
    drop = {op.op_id for op in draft}
    kept = {owner: tuple(replace(op, deps=tuple(dep for dep in op.deps if dep not in drop))
                         for op in items if op.op_id not in drop)
            for owner, items in _graphs(ops).items()}
    patches.extend(Patch(f"/ops/{op.op_id}", "selected", "excluded",
                         "Remove MTP draft work.") for op in draft)
    return candidate, kept, patches


def _prefix(case, ops):
    hits = sum(request.prefix_tokens for request in case.workload.requests)
    candidates = _tagged(ops, lambda op: _meta(op, "prefix_cache") is True
                         or op.family == "prefix_cache")
    if hits <= 0 and not candidates:
        raise _Inapplicable("workload declares no prefix hits")
    requests = tuple(replace(request, prefix_tokens=0)
                     for request in case.workload.requests)
    metadata = tuple((key, value) for key, value in case.runtime.metadata
                     if key != "enable_prefix_caching") + (
        ("enable_prefix_caching", False),)
    candidate = replace(case, workload=replace(case.workload, requests=requests),
                        runtime=replace(case.runtime, metadata=metadata))
    patches = [Patch("/workload/requests", [r.prefix_tokens for r in case.workload.requests],
                     [0 for _ in case.workload.requests],
                     "Disable prefix caching and declared prefix hits.")]
    return candidate, ops, patches


def _pcp_dcp(case, ops):
    parallel = case.runtime.parallel
    if parallel.pcp <= 1 and parallel.cp <= 1:
        raise _Inapplicable("baseline has no PCP/DCP sharding")
    candidate = _runtime(case, parallel=replace(parallel, pcp=1, cp=1, dcp=1,
                                                         cp_plan="none"))
    patches = [Patch("/runtime/parallel/pcp", parallel.pcp, 1, "Disable PCP."),
               Patch("/runtime/parallel/cp", parallel.cp, 1, "Disable DCP.")]
    return candidate, ops, patches


def _multi_stream(case, ops):
    aux = _tagged(ops, lambda op: op.stream == "aux")
    flags = ("multistream_overlap_shared_expert", "multistream_kda_proj",
             "multistream_attention_indexer", "multistream_window_compressed")
    if not aux and not any(getattr(case.runtime, flag) for flag in flags):
        raise _Inapplicable("baseline has no auxiliary stream")
    candidate = _runtime(case, **{flag: False for flag in flags})
    patches = [Patch(f"/runtime/{flag}", getattr(case.runtime, flag), False,
                     "Disable auxiliary-stream overlap.") for flag in flags]
    return candidate, _replace_ops(
        ops, lambda op: replace(op, stream="") if op.stream == "aux" else op), patches


_TRANSFORMS = {"ACLGraph": _aclgraph, "MLAPO": _mlapo, "MC2": _mc2, "MTP": _mtp,
               "Prefix Caching": _prefix, "PCP/DCP": _pcp_dcp,
               "Multi-Stream Overlap": _multi_stream}
_REGENERATED = frozenset({"ACLGraph", "MLAPO", "MC2", "MTP", "Prefix Caching",
                          "PCP/DCP", "Multi-Stream Overlap"})


def _candidate_ops(feature, case, base_ops, *, generated, phase, scope,
                   tokens, context, sequences):
    if not generated:
        if feature == "PCP/DCP":
            raise _Inapplicable("repartition_requires_generated_plan")
        return base_ops
    return target_graphs(case, phase=phase, tokens=tokens, context=context,
                         sequences=sequences, scope=scope)


def _counterfactual(feature, case, ops, baseline, *, generated, phase, scope,
                    tokens, context, sequences, include_submission, scope_name):
    identifier = _FEATURE_ID[feature]
    patches = ()
    try:
        candidate_case, candidate_ops, patches = _TRANSFORMS[feature](case, ops)
        candidate_case = Case.from_mapping(candidate_case.to_dict())
        candidate_ops = _candidate_ops(feature, candidate_case, candidate_ops,
                                       generated=generated, phase=phase, scope=scope,
                                       tokens=tokens, context=context,
                                       sequences=sequences)
        if feature in {"Prefix Caching", "MTP"}:
            base_graph = {owner: tuple(items) for owner, items in _graphs(ops).items()}
            candidate_graph = {owner: tuple(items)
                               for owner, items in _graphs(candidate_ops).items()}
            if candidate_graph == base_graph:
                raise _Inapplicable(
                    f"{_FEATURE_ID[feature]} geometry is unchanged: explicit "
                    "prefill/verify token and commit-boundary inputs are required.")
            if feature == "MTP" and not generated:
                raise _Inapplicable(
                    "custom selected-plan MTP cannot be re-materialized without "
                    "explicit verify-token and commit-boundary inputs")
        predicted = _predict(candidate_case, candidate_ops, include_submission, scope)
        if not math.isfinite(predicted) or predicted < 0:
            raise _Inapplicable("candidate prediction is not finite and nonnegative")
        plan_id = f"{baseline.context.plan_id}:cf:{identifier}"
        plan = project_ops(candidate_case, candidate_ops, plan_id=plan_id,
                          external_inputs=tuple(_external(scope) or ()))
        if plan.coverage != "complete":
            raise _Inapplicable("candidate SOL coverage is incomplete: "
                                + ",".join(plan.holes[:3]))
        candidate_context = replace(baseline.context, plan_id=plan_id,
                                    coverage=baseline.context.coverage)
        report = compare_sol(plan, predicted, context=candidate_context,
                             dispatch=baseline.bound.dispatch_policy,
                             includes_host_submission=include_submission,
                             validate=True)
        if report.status not in {"completed"} or report.bound.status != "available":
            raise _Inapplicable("candidate SOL consistency validation failed")
        sol_ms = report.bound.sol_ms
        if sol_ms is None or sol_ms > predicted + 1e-12:
            raise _Inapplicable("candidate SOL exceeds candidate prediction")
    except _Inapplicable as exc:
        recorded = tuple(patches) + (
            Patch(f"/features/{identifier}", None, None, str(exc)),)
        return Counterfactual(identifier, feature, "inapplicable",
                              baseline.context.plan_id, None, recorded,
                              (), None, None, None, None, str(exc))
    except (ModelingError, ValueError, ReportValidationError) as exc:
        reason = f"{type(exc).__name__}:{str(exc)[:220]}"
        return Counterfactual(identifier, feature, "failed", baseline.context.plan_id,
                              None, (Patch(f"/features/{identifier}", None, None, reason),),
                              (), None, None, None, None, reason)
    return Counterfactual(
        identifier, feature, "evaluated", baseline.context.plan_id, plan.plan_id,
        tuple(patches),
        ("same_workload", "same_hardware", "same_quantization",
         "same_implementation", "same_declared_boundary"),
        predicted, sol_ms, baseline.prediction.predicted_ms - predicted,
        baseline.bound.sol_ms - sol_ms, None)


def compare_launch(report, *, phase="decode", scope="model", tokens=1, context=4096,
                   sequences=16, dispatch="ideal_zero", request_metrics=False,
                   counterfactuals=(), backend="auto", max_candidates=16,
                   allow_partial=False, plan_ops=None, prefill_ops=None,
                   predicted_ms=None, request_trace=None, commits=None):
    """Compare one launch report's selected plan with the Engine prediction."""
    if report.status in {"failed", "needs_inputs"}:
        raise LaunchComparisonError(f"launch_status_not_comparable:{report.status}")
    if report.status == "partial" and not allow_partial:
        raise LaunchComparisonError("launch_partial_requires_allow_partial")
    if report.case is None:
        raise LaunchComparisonError("launch_case_required")
    case = Case.from_mapping(report.case)
    generated = plan_ops is None
    ops = plan_ops if plan_ops is not None else target_graphs(
        case, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope=scope, pure_model=False)
    include_submission = bool(case.runtime.submission)
    prediction = (float(predicted_ms) if predicted_ms is not None
                  else _predict(case, ops, include_submission, scope))
    if not math.isfinite(prediction) or prediction < 0:
        raise LaunchComparisonError("prediction_must_be_finite_and_nonnegative")
    plan_id = f"{case.model.name}:{phase}:{scope}:selected"
    external = tuple(_external(scope) or ())
    plan = project_ops(case, ops, plan_id=plan_id, external_inputs=external)
    coverage = "conditional_subset" if report.status == "partial" else "complete"
    if plan.coverage != "complete":
        coverage = "conditional_subset"
    ctx = Context(case.model.name, plan_id,
                  "body_step" if scope == "body" else "model_step",
                  phase, int(tokens), int(sequences), int(context),
                  "selected_execution_plan", coverage, (report.source,))
    backend_data = {"requested": backend, "engine_backend": "python_reference",
                    "reason": "same_selected_plan", "native_service_steps": 0,
                    "native_template_tiles": 0, "python_boundaries": 0,
                    "unaccelerated_reasons": []}
    prefill_pred = prefill_sol = None
    if request_metrics and prefill_ops is not None:
        prefill_pred = _predict(case, prefill_ops, include_submission, scope)
        prefill_plan = project_ops(case, prefill_ops, plan_id=f"{plan_id}:prefill",
                                  external_inputs=external)
        prefill_bound = compare_sol(prefill_plan, prefill_pred,
                                    context=replace(ctx, plan_id=f"{plan_id}:prefill",
                                                    phase="prefill"),
                                    dispatch=dispatch, validate=False)
        if prefill_bound.bound.status == "available":
            prefill_sol = prefill_bound.bound.sol_ms
    if commits is not None and (
            not math.isfinite(float(commits)) or float(commits) <= 0):
        raise LaunchComparisonError("commits_must_be_finite_and_positive")
    baseline = compare_sol(plan, prediction, context=ctx, backend=backend_data,
                           includes_host_submission=include_submission, dispatch=dispatch,
                           request_metrics_enabled=request_metrics,
                           commits=commits, request_trace=request_trace,
                           prefill_predicted_ms=prefill_pred, prefill_sol_ms=prefill_sol)
    if coverage != "complete" and baseline.status == "completed":
        baseline = replace(baseline, status="partial")
    aliases = {**_FEATURE_ID, **{key.lower(): key for key in FEATURES},
               "capture": "ACLGraph", "shared_expert": "Multi-Stream Overlap",
               "prefix": "Prefix Caching", "pcp": "PCP/DCP"}
    reverse = {value: key for key, value in _FEATURE_ID.items()}
    requested = [reverse.get(aliases.get(str(item), str(item)),
                             aliases.get(str(item), str(item))) for item in counterfactuals]
    requested = [feature for feature in FEATURES if feature in requested]
    limit = max(0, min(16, int(max_candidates)))
    evaluated = []
    for feature in requested[:limit]:
        evaluated.append(_counterfactual(
            feature, case, ops, baseline, generated=generated, phase=phase,
            scope=scope, tokens=tokens, context=context, sequences=sequences,
            include_submission=include_submission, scope_name=scope))
    for feature in requested[limit:]:
        evaluated.append(Counterfactual(
            _FEATURE_ID[feature], feature, "unevaluated", plan_id, None,
            (Patch(f"/features/{_FEATURE_ID[feature]}", None, None,
                   "budget exhausted"),), (), None, None, None, None,
            "max_candidates budget exhausted"))
    evaluated.sort(key=lambda item: (item.saved_ms is None, -(item.saved_ms or 0.0),
                                     item.id))
    final = replace(baseline, counterfactuals=tuple(evaluated))
    validate_sol_report(final)
    return final


__all__ = ["FEATURES", "LaunchComparisonError", "compare_launch"]
