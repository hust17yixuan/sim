"""Selected-work SOL bounds and same-plan comparison arithmetic.

[Role]
  Consume one `PlanWork` projection and compute both declared dispatch
  policies: `ideal_zero` (no host/device launch floors) and
  `declared_physical` (only provenance-labeled startup/inclusive minima).
  Report resource loads, the dependency critical path, local bounds,
  efficiency and request-metric projections; never solve a second schedule.

[Invariants]
  * Service is `work / peak_per_ms`; `peak_ms` work is already service.
  * Within one local term, service modes on the same physical capacity domain
    sum before the maxima across distinct domains are taken.  HBM and MTE are
    distinct constraints, so they combine by max, not by adding bytes.
  * A declared invocation/dispatch minimum is inclusive; a serial intra-recipe
    phase path is a separate local lower bound.  Measured Stage latency is
    reported as an empirical floor, never added to physical SOL.
  * Global SOL is `max(critical_path, resource_load)`.  An unresolved
    dependency is unavailable, never a silently dropped edge.
  * `SOL <= predicted + 1e-12 ms` for the same plan; a violation emits
    `bound_exceeds_prediction` and fails validation, never clipping
    efficiency above 100.  A missing/zero peak makes the bound unavailable.
  * TPOT requires explicit, chronologically ordered single-request boundaries
    and a positive integer decode step count; S05 rejects invalid latencies.

[Timeouts & Locks]
  O(terms + edges + demands), finite in-memory; no locks.

[Side-Effect Policy]
  No I/O, devices, measurement or input mutation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
import math

from ..feature_reports import (
    Bound, Context, Counterfactual, Diagnostic, GapEvidence, LocalBound, Prediction,
    ReportValidationError, RequestMetric, ResourceLoad, SolComparisonReport,
    SOL_TOLERANCE_MS, validate_sol_report,
)
from .sol_work import PlanWork


def _grouped_service(demands):
    grouped = defaultdict(float)
    for demand in demands:
        grouped[demand.physical_resource_id] += demand.service_ms
    return grouped


def _local_ms(term, demands, dispatch):
    grouped = _grouped_service(demands)
    service = max(grouped.values(), default=0.0)
    if dispatch != "declared_physical":
        phase = float(getattr(term, "ideal_phase_precedence_ms", 0.0) or 0.0)
        return max(service, phase)
    phase = float(getattr(term, "phase_precedence_ms", 0.0) or 0.0)
    return max(service + float(getattr(term, "dispatch_min_ms", 0.0) or 0.0),
               float(getattr(term, "inclusive_min_ms", 0.0) or 0.0), phase)


def critical_path(plan: PlanWork, dispatch="ideal_zero"):
    """Longest dependency path over selected terms, including releases."""
    indexed = {term.id: term for term in plan.terms}
    waiting = {term.id: 0 for term in plan.terms}
    successors = defaultdict(list)
    for term in plan.terms:
        for predecessor in term.predecessors:
            if predecessor not in indexed:
                raise ValueError(f"sol_dependency_unresolved:{term.id}->{predecessor}")
            waiting[term.id] += 1
            successors[predecessor].append(term.id)
    ready = deque(sorted(key for key, count in waiting.items() if not count))
    ends, parents, order = {}, {}, []
    while ready:
        key = ready.popleft()
        term = indexed[key]
        demands, holes = plan.demands(term, dispatch)
        if holes:
            raise ValueError(";".join(holes))
        local = _local_ms(term, demands, dispatch)
        upstream = max((ends[item] for item in term.predecessors), default=0.0)
        start = max(float(term.release_ms), upstream)
        ends[key] = start + local
        parents[key] = (max(term.predecessors, key=ends.get)
                        if term.predecessors else None)
        order.append(key)
        for successor in successors.get(key, ()):
            waiting[successor] -= 1
            if not waiting[successor]:
                ready.append(successor)
    if len(order) != len(plan.terms):
        raise ValueError("sol_dependency_cycle")
    leaf = max(indexed, key=ends.get, default=None)
    path = []
    cursor = leaf
    while cursor is not None:
        path.append(cursor)
        cursor = parents.get(cursor)
    ops = []
    for key in reversed(path):
        for op in indexed[key].member_ops:
            if op not in ops:
                ops.append(op)
    return (ends[leaf] if leaf is not None else 0.0), tuple(ops)


def resource_loads(plan: PlanWork, dispatch="ideal_zero"):
    """One row per physical constraint; service demands sum, never max."""
    grouped = defaultdict(list)
    for term in plan.terms:
        demands, holes = plan.demands(term, dispatch)
        if holes:
            raise ValueError(";".join(holes))
        for demand in demands:
            grouped[demand.physical_resource_id].append(demand)
    rows = []
    for resource_id in sorted(grouped):
        terms = tuple(grouped[resource_id])
        rows.append(ResourceLoad(resource_id, terms,
                                 math.fsum(item.service_ms for item in terms)))
    return tuple(rows)


def local_bounds(plan: PlanWork, dispatch="ideal_zero"):
    bounds = []
    for term in plan.terms:
        demands, holes = plan.demands(term, dispatch)
        if holes:
            raise ValueError(";".join(holes))
        resource = max(_grouped_service(demands).values(), default=0.0)
        local = _local_ms(term, demands, dispatch)
        bounds.append(LocalBound(term.id, term.member_ops, demands, resource, local, local))
    return tuple(bounds)


def _floor_bases(plan, dispatch):
    if dispatch != "declared_physical":
        return ()
    bases = []
    for term in plan.terms:
        # A scoped nested floor reports its basis even when it does not
        # promote either whole-operator dispatch or inclusive minimum.
        bases.extend(getattr(term, "floor_bases", ()) or (
            (term.floor_basis,) if term.floor_basis else ()))
    return tuple(dict.fromkeys(item for item in bases if item))


def _empirical_floor(plan):
    """Measured service is reported separately, never added to physical SOL."""
    measured = [float(getattr(term, "measured_latency_ms", 0.0) or 0.0)
                for term in plan.terms]
    measured = [value for value in measured if value > 0]
    return max(measured) if measured else None


def bound(plan: PlanWork, dispatch="ideal_zero"):
    """Compute one necessary lower bound; unavailable on a coverage hole."""
    reasons = list(plan.holes)
    if plan.coverage != "complete":
        reasons.append(f"coverage:{plan.coverage}")
    empirical = _empirical_floor(plan)
    empirical_flag = bool(getattr(plan, "uses_empirical_derating", False))
    try:
        resources = resource_loads(plan, dispatch)
        path_ms, path_ops = critical_path(plan, dispatch)
        locals_ = local_bounds(plan, dispatch)
    except ValueError as exc:
        return Bound("unavailable", dispatch, None, (), None, None, (), (), tuple(
            dict.fromkeys(reasons + [str(exc)])),
            uses_empirical_derating=empirical_flag,
            floor_bases=_floor_bases(plan, dispatch),
            empirical_floor_ms=empirical)
    if reasons:
        return Bound("unavailable", dispatch, None, (), None, None, resources, locals_,
                     tuple(dict.fromkeys(reasons)),
                     uses_empirical_derating=empirical_flag,
                     floor_bases=_floor_bases(plan, dispatch),
                     empirical_floor_ms=empirical)
    load = max((row.load_bound_ms for row in resources), default=0.0)
    sol = max(path_ms, load)
    return Bound("available", dispatch, path_ms, path_ops, load, sol, resources, locals_,
                 uses_empirical_derating=empirical_flag,
                 floor_bases=_floor_bases(plan, dispatch),
                 empirical_floor_ms=empirical)


def _unavailable(name, reason, statistic="per_request"):
    return RequestMetric(name, "unavailable", None, "not_requested", statistic, None,
                         None, False, reason)


_COMMIT_FIELDS = ("commit_times", "commit_events", "step_commit_times")
_COMMIT_TIME_KEYS = ("time_ms", "commit_ms", "time", "at_ms", "timestamp_ms")
_IDENTITY_KEYS = ("request_id", "id")
_IDENTITY_CONTAINERS = ("workload", "request")


def _identity_of(container):
    """One validated identity from a supported container, or a failure."""
    values = []
    for name in _IDENTITY_KEYS:
        if name in container and container[name] is not None:
            value = container[name]
            if not isinstance(value, str) or not value.strip():
                return None, "Request identity must be a non-empty string."
            if value not in values:
                values.append(value)
    if len(values) > 1:
        return None, "conflicting_request_identity"
    return (values[0] if values else None), None


def _request_identity(trace):
    """Reconcile every supported identity container and event declaration."""
    values, mapped, identified = [], 0, 0
    for container in (trace, *(trace.get(name) for name in _IDENTITY_CONTAINERS)):
        if not isinstance(container, Mapping):
            continue
        value, failure = _identity_of(container)
        if failure is not None:
            return None, failure
        if value is not None:
            values.append(value)
    for key in _COMMIT_FIELDS:
        raw = trace.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            mapped += 1
            value, failure = _identity_of(entry)
            if failure is not None:
                return None, failure
            if value is not None:
                identified += 1
                values.append(value)
    unique = set(values)
    if len(unique) > 1:
        return None, "conflicting_request_identity"
    if mapped and not values:
        return None, "missing_request_identity"
    if mapped and 0 < identified < mapped:
        return None, "missing_request_identity"
    return (values[0] if values else None), None


def _trace_request_id(trace):
    """Single-request identity after full reconciliation (``str | None``)."""
    return _request_identity(trace)[0]


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _event_time(entry):
    """One event mapping's agreed timestamp; every alias participates.

    A present alias is never discarded: ``None``, ``bool``, non-numeric,
    ``NaN`` and ``Inf`` all fail.  Agreement is the full range
    ``max(values) - min(values) <= 1e-9`` rather than an adjacent-pair chain,
    so a slow tolerance walk cannot pass.
    """
    present = [(name, entry[name]) for name in _COMMIT_TIME_KEYS if name in entry]
    if not present:
        return None, ("Commit events must be objects carrying a "
                      "time_ms timestamp.")
    values = []
    for _name, raw in present:
        value = _timestamp(raw)
        if value is None:
            return None, "Commit boundaries must be finite timestamps."
        values.append(value)
    if max(values) - min(values) > 1e-9:
        return None, "conflicting_event_timestamps"
    return values[0], None


def _boundary_field(trace, key, steps):
    """Parse one commit-boundary spelling under its own structure rules.

    ``steps=None`` still performs every timestamp/identity/chronology check;
    only the decode-step count relation is skipped.  The shared admission gate
    uses that mode when a trace carries events but no usable step count, so
    TTFT cannot certify contradictory evidence.
    """
    raw, events = trace.get(key), key == "commit_events"
    if not isinstance(raw, (list, tuple)) or not raw:
        return (), ("Commit boundary count must equal decode_steps or "
                    "decode_steps+1.")
    if steps is not None and len(raw) < 2:
        return (), ("Commit boundary count must equal decode_steps or "
                    "decode_steps+1.")
    times, mappings = [], []
    for entry in raw:
        if isinstance(entry, Mapping):
            value, failure = _event_time(entry)
            if failure is not None:
                return (), failure
            mappings.append(True)
        else:
            if events:
                return (), ("Commit events must be objects carrying a "
                            "time_ms timestamp.")
            value = _timestamp(entry)
            if value is None:
                return (), "Commit boundaries must be finite timestamps."
            mappings.append(False)
        times.append(value)
    if any(mappings) and not all(mappings):
        return (), "Commit boundary representations cannot be mixed."
    arrival = trace.get("arrival_ms")
    if arrival is not None:
        arrival = _timestamp(arrival)
        if arrival is None:
            return (), "Arrival timestamp must be finite."
        if any(value < arrival for value in times):
            return (), "Commit boundaries precede the declared arrival."
    first = trace.get("first_commit_ms")
    if first is not None:
        first = _timestamp(first)
        if first is None:
            return (), "First commit timestamp must be finite."
        if abs(times[0] - first) > 1e-9:
            return (), "First commit boundary must equal first_commit_ms."
        if arrival is not None and first < arrival:
            return (), "First commit precedes the declared arrival."
    if any(right <= left for left, right in zip(times, times[1:])):
        return (), "Commit boundaries must strictly increase."
    if steps is not None and len(times) not in {steps, steps + 1}:
        return (), ("Commit boundary count must equal decode_steps or "
                    "decode_steps+1.")
    return tuple(times), None


def _parse_commit_boundaries(trace, key, steps):
    """Validate every supplied boundary alias, not only the first one."""
    supplied = [name for name in _COMMIT_FIELDS if trace.get(name) is not None]
    times = None
    for name in supplied:
        field_times, failure = _boundary_field(trace, name, steps)
        if failure is not None:
            return (), failure
        if times is None:
            times = field_times
            continue
        if (len(field_times) != len(times) or any(
                abs(left - right) > 1e-9
                for left, right in zip(times, field_times))):
            return (), "Commit boundary fields disagree on timestamps."
    return (times, None) if times is not None else (
        (), "Commit boundary count must equal decode_steps or decode_steps+1.")


def request_metrics(plan, predicted_ms, sol_ms, *, phase="decode", commits=None,
                    prefill_predicted_ms=None, prefill_sol_ms=None, trace=None):
    """TTFT/TPOT from explicit boundaries and fractional expected commits.

    For ``N`` decode steps totaling ``T_total`` (or ``N * T_step``) with
    ``E[commits] = g`` per step, ``TPOT = T_step / g = T_total / (N*g)``.
    A step count is never multiplied into a per-commit interval.
    """
    trace = dict(trace or {})
    identity, identity_failure = _request_identity(trace)
    if commits is None:
        commits = trace.get("commits")
    steps_raw = trace.get("decode_steps")
    steps = None
    if isinstance(steps_raw, (int, float)) and not isinstance(steps_raw, bool):
        if math.isfinite(float(steps_raw)) and float(steps_raw) == int(steps_raw) \
                and int(steps_raw) > 0:
            steps = int(steps_raw)
    supplied = [key for key in _COMMIT_FIELDS if trace.get(key) is not None]
    # Shared admission gate: every supplied boundary/event spelling is
    # reconciled once, before TTFT or TPOT can use the same evidence.  It runs
    # even when TPOT is not requested or has no other decode inputs.
    boundary_times, boundary_failure = (
        _parse_commit_boundaries(trace, supplied[0], steps)
        if supplied else ((), None))
    if boundary_failure is not None and supplied:
        ttft_reason = "conflicting_event_timestamps"
    else:
        ttft_reason = None
    ttft = _unavailable("ttft", "Arrival/first-commit trace and a prefill plan are required.")
    if ttft_reason is not None and identity_failure is None:
        ttft = _unavailable("ttft", ttft_reason)
    if (trace.get("arrival_ms") is not None and trace.get("first_commit_ms") is not None
            and prefill_predicted_ms is not None and prefill_sol_ms is not None):
        arrival = _timestamp(trace["arrival_ms"])
        commit = _timestamp(trace["first_commit_ms"])
        try:
            prefill = float(prefill_sol_ms)
        except (TypeError, ValueError):
            prefill = float("nan")
        if identity_failure is not None:
            ttft = _unavailable("ttft", identity_failure)
        elif ttft_reason is not None:
            ttft = _unavailable("ttft", ttft_reason)
        elif arrival is None or commit is None or not (
                math.isfinite(prefill) and prefill > 0):
            ttft = _unavailable("ttft", "Arrival/first-commit/prefill values must be positive and finite.")
        elif commit < arrival:
            ttft = _unavailable("ttft", "First commit precedes arrival.")
        elif commit - arrival < prefill - SOL_TOLERANCE_MS:
            ttft = _unavailable(
                "ttft", "Observed TTFT precedes the declared prefill SOL bound.")
        elif commit - arrival <= 0:
            ttft = _unavailable("ttft", "First commit must follow arrival.")
        else:
            ttft = RequestMetric("ttft", "available", identity,
                                 trace.get("scope", "fixed_trace"), "per_request",
                                 prefill, commit - arrival, True, None)
    tpot = _unavailable(
        "tpot", "Decode trace, explicit commit boundary and acceptance are required.")
    if (phase == "decode" and predicted_ms is not None and sol_ms is not None
            and steps is not None and commits is not None and supplied):
        if identity_failure is not None:
            tpot = _unavailable("tpot", identity_failure)
        elif boundary_failure is not None:
            tpot = _unavailable("tpot", boundary_failure)
        else:
            times = boundary_times
            intervals = steps if len(times) == steps + 1 else steps - 1
            try:
                commits_value = float(commits)
            except (TypeError, ValueError):
                commits_value = 0.0
            total = times[-1] - times[0]
            per_step = float(sol_ms)
            if (math.isfinite(commits_value) and commits_value > 0
                    and math.isfinite(total) and total > 0
                    and math.isfinite(per_step) and per_step > 0):
                tpot = RequestMetric(
                    "tpot", "available", identity,
                    trace.get("scope", "fixed_trace"), "per_committed_token",
                    per_step / commits_value,
                    total / (intervals * commits_value), True, None)
            else:
                tpot = _unavailable(
                    "tpot", "Decode totals and expected commits must be positive and finite.")
    elif steps_raw is not None and steps is None:
        tpot = _unavailable(
            "tpot", "decode_steps must be a positive integer step count.")
    return ttft, tpot


def _gap(predicted_ms, sol_ms, evidence=()):
    delta = predicted_ms - sol_ms
    if abs(delta) <= SOL_TOLERANCE_MS:
        return (GapEvidence("none", "zero_gap", (), "Equal to the lower bound within 1e-12 ms.", 0.0),)
    if evidence:
        return tuple(evidence)
    return (GapEvidence("unknown", "not_identified", (),
                        "No labeled engine accounting or isolated intervention identifies the gap.",
                        delta),)


def compare_sol(plan: PlanWork, predicted_ms, *, context: Context, backend=None,
                includes_host_submission=False, timing_basis="engine_selected_plan",
                dispatch="ideal_zero", request_metrics_enabled=False, commits=None,
                prefill_predicted_ms=None, prefill_sol_ms=None, request_trace=None,
                counterfactuals=(), gap_evidence=(), diagnostics=(), status="completed",
                validate=True):
    """Compare the same plan's SOL with its Engine prediction and record gaps."""
    if not math.isfinite(predicted_ms) or predicted_ms < 0:
        raise ValueError("predicted_ms_must_be_finite_and_nonnegative")
    ideal = bound(plan, "ideal_zero")
    physical = bound(plan, "declared_physical") if (
        dispatch == "declared_physical" or any(
            getattr(term, "inclusive_min_ms", 0.0) or term.dispatch_min_ms
            for term in plan.terms)) else None
    selected = physical if dispatch == "declared_physical" and physical is not None else ideal
    backend = backend or {"requested": "python", "engine_backend": "python_reference",
                          "reason": "selected_plan_engine", "native_service_steps": 0,
                          "native_template_tiles": 0, "python_boundaries": 0,
                          "unaccelerated_reasons": []}
    prediction = Prediction(plan.plan_id, float(predicted_ms), backend,
                            bool(includes_host_submission), timing_basis)
    sol_ms = selected.sol_ms
    notes = list(diagnostics)
    if sol_ms is not None and sol_ms > predicted_ms + SOL_TOLERANCE_MS:
        notes.append(Diagnostic("bound_exceeds_prediction", "error",
                                "Same-plan SOL exceeds the Engine prediction; consistency failed.",
                                None, ()))
        status = "failed"
    efficiency = (100.0 * sol_ms / predicted_ms
                  if sol_ms is not None and predicted_ms > 0 else None)
    metrics = request_metrics(plan, predicted_ms, sol_ms, phase=context.phase,
                              commits=commits, prefill_predicted_ms=prefill_predicted_ms,
                              prefill_sol_ms=prefill_sol_ms,
                              trace=request_trace) if request_metrics_enabled else ()
    if selected.status != "available" and status == "completed":
        status = "partial"
    report = SolComparisonReport(
        "sol-comparison/1", status, context, selected, physical, prediction, efficiency,
        _gap(predicted_ms, sol_ms, gap_evidence) if sol_ms is not None else (
            GapEvidence("unknown", "not_identified", (), "SOL bound is unavailable.", None),),
        tuple(metrics), tuple(counterfactuals), tuple(notes))
    if validate:
        validate_sol_report(report)
    return report


__all__ = ["bound", "compare_sol", "critical_path", "local_bounds", "request_metrics",
           "resource_loads"]
