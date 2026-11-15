"""Static service-regime domain compilation and coverage validation.

[Role] Normalize declared response domains into physical intervals and prove
  that every continuity group covers its announced support exactly once.
[Invariants] Coverage is geometric: interior gaps, fractional gaps such as
  (128, 129), multidimensional holes, ambiguous overlaps, uncovered faces and
  knot support that cannot reach the declared domain are rejected. Tolerance
  (atol=1e-6 us, rtol=1e-9) compares response values only, never geometry.
  Shared-face limits must agree; rate-dependent or roofline seams require a
  provably equivalent expression. Group members agree on categorical
  selectors, coordinate definitions, execution mode and native boundary.
  Validation work is bounded and fails explicitly when the budget is exceeded.
[Timeouts & Locks] Bounded interval subdivisions; no waits, I/O or locks.
[Side-Effect Policy] Pure validation: reads declarations and returns reports.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import json
import math

from mindie_sim.schema.common import ModelingError
from .response_functions import (ResponseFunction, nonnegative, positive,
                                 provenance_roles, same_declared_value)

ATOL_US = 1e-6
RTOL = 1e-9
MAX_PROBES = 100_000
MAX_FACE_VERTICES = 4_096
EXECUTION_MODES = frozenset({"completion_floor", "joint_native", "mac_service"})
OUTSIDE_POLICIES = frozenset({"analytic", "linear", "error"})
_BOUND_FIELDS = frozenset({"min", "max", "min_inclusive", "max_inclusive"})
_CONTINUITY_FIELDS = frozenset({"group", "support", "outside"})


@dataclass(frozen=True)
class Interval:
    """One closed (or explicitly half-open) physical interval."""

    name: str
    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @property
    def empty(self):
        if self.lower > self.upper:
            return True
        return self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive)

    def intersect(self, other):
        lower, lower_inclusive = self.lower, self.lower_inclusive
        if other.lower > lower:
            lower, lower_inclusive = other.lower, other.lower_inclusive
        elif other.lower == lower:
            lower_inclusive = lower_inclusive and other.lower_inclusive
        upper, upper_inclusive = self.upper, self.upper_inclusive
        if other.upper < upper:
            upper, upper_inclusive = other.upper, other.upper_inclusive
        elif other.upper == upper:
            upper_inclusive = upper_inclusive and other.upper_inclusive
        return Interval(self.name, lower, upper, lower_inclusive, upper_inclusive)

    def contains(self, value):
        if value < self.lower or value > self.upper:
            return False
        if value == self.lower and not self.lower_inclusive:
            return False
        if value == self.upper and not self.upper_inclusive:
            return False
        return True

    def to_dict(self):
        return {
            "min": self.lower if math.isfinite(self.lower) else None,
            "max": self.upper if math.isfinite(self.upper) else None,
            "min_inclusive": self.lower_inclusive,
            "max_inclusive": self.upper_inclusive,
        }


@dataclass(frozen=True)
class RuleDomain:
    """Normalized applicability, response and execution of one rule."""

    rule_id: str
    index: int
    match: Mapping
    coordinates: Mapping
    execution_mode: str
    execution: Mapping | None
    group: str
    support: Mapping[str, Interval]
    function: ResponseFunction
    response: Mapping
    outside: str
    coverage: Mapping[str, Interval]
    guards: Mapping[str, Interval]

    def to_dict(self):
        return {"id": self.rule_id, "index": self.index,
                "execution_mode": self.execution_mode,
                "response_basis": self.response.get("kind", "affine"),
                "outside": self.outside,
                "coverage": {name: interval.to_dict()
                             for name, interval in self.coverage.items()}}


def _canonical(value):
    return json.dumps(value, sort_keys=True, default=repr, allow_nan=False,
                      separators=(",", ":"))


def _number(value, code):
    try:
        return nonnegative(value)
    except ModelingError as exc:
        raise ModelingError(code) from exc


def _flag(bounds, field):
    value = bounds.get(field, True)
    if not isinstance(value, bool):
        raise ModelingError("service_regime_domain_inclusive_flag_requires_boolean:" + field)
    return value


def _interval(name, bounds, *, require_max=False):
    if not isinstance(bounds, Mapping):
        raise ModelingError("service_regime_domain_bounds_require_mapping:" + name)
    if set(bounds) - _BOUND_FIELDS:
        raise ModelingError("service_regime_domain_unknown_fields:" + name)
    if require_max and ({"min", "max"} - set(bounds)):
        raise ModelingError("service_regime_continuity_support_requires_bounds:" + name)
    lower = _number(bounds.get("min", 0), "service_regime_domain_requires_finite_bounds:" + name)
    if "max" in bounds:
        upper = _number(bounds["max"], "service_regime_domain_requires_finite_bounds:" + name)
    else:
        upper = math.inf
    lower_inclusive = _flag(bounds, "min_inclusive")
    upper_inclusive = _flag(bounds, "max_inclusive")
    if upper < lower:
        raise ModelingError("service_regime_domain_reversed:" + name)
    return Interval(name, lower, upper, lower_inclusive, upper_inclusive)


def _execution(rule):
    execution = rule.get("execution")
    if execution is None:
        return "completion_floor", None
    if not isinstance(execution, Mapping):
        raise ModelingError("service_regime_execution_requires_mapping")
    kind = execution.get("kind")
    if kind == "completion_floor":
        if set(execution) != {"kind"}:
            raise ModelingError("service_regime_completion_floor_execution_requires_only_kind")
        return "completion_floor", None
    if kind == "mac_service":
        if (set(execution) != {"kind", "resource", "native_kernel",
                               "tile_m", "tile_n", "tile_k", "slots"}
                or execution.get("resource") != "cube"
                or not execution.get("native_kernel")):
            raise ModelingError("mac_service_requires_cube_and_native_kernel")
        for key in ("tile_m", "tile_n", "tile_k", "slots"):
            if (isinstance(execution.get(key), bool)
                    or not isinstance(execution.get(key), int)
                    or execution[key] <= 0):
                raise ModelingError("mac_service_requires_positive_tile_geometry")
        return "mac_service", dict(execution)
    if kind != "joint_native":
        raise ModelingError("service_regime_execution_kind_unknown")
    if not execution.get("resource") or not execution.get("native_kernel"):
        raise ModelingError("joint_native_service_requires_queue_and_native_boundary")
    if _number(execution.get("reference_rate", 0),
               "service_regime_reference_rate_must_be_positive") == 0:
        raise ModelingError("service_regime_reference_rate_must_be_positive")
    return "joint_native", dict(execution)


def _continuity(rule):
    raw = rule.get("continuity")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - _CONTINUITY_FIELDS:
        raise ModelingError("service_regime_continuity_unknown_fields")
    group = raw.get("group")
    if not isinstance(group, str) or not group.strip():
        raise ModelingError("service_regime_continuity_requires_group")
    support = raw.get("support")
    if not isinstance(support, Mapping) or not support:
        raise ModelingError("service_regime_continuity_requires_support")
    outside = raw.get("outside", "analytic")
    if outside not in OUTSIDE_POLICIES:
        raise ModelingError("service_regime_continuity_outside_requires_analytic_linear_or_error")
    return {"group": group,
            "support": {name: _interval(name, bounds, require_max=True)
                        for name, bounds in support.items()},
            "outside": outside}


def _response(rule, coordinates):
    response = rule.get("response")
    if not isinstance(response, Mapping):
        raise ModelingError("service_regime_response_requires_mapping")
    function = ResponseFunction.from_mapping(response, coordinates)
    outside = response.get("outside", "analytic") if function.kind in {
        "piecewise_linear", "multilinear"} else "analytic"
    if function.kind in {"piecewise_linear", "multilinear"} and outside not in OUTSIDE_POLICIES:
        raise ModelingError("service_regime_curve_outside_requires_analytic_linear_or_error")
    return response, function, outside


def _coverage(rule, domain, function, support, outside):
    declared = rule.get("domain") or {}
    if not isinstance(declared, Mapping):
        raise ModelingError("service_regime_domain_requires_mapping")
    if set(declared) - set(domain):
        raise ModelingError("service_regime_domain_requires_declared_coordinate")
    if not set(function.varying_coordinates) <= set(support):
        missing = sorted(set(function.varying_coordinates) - set(support))
        raise ModelingError("service_regime_domain_support_requires_response_coordinate:"
                            + ",".join(missing))
    coverage = {}
    for name, base in support.items():
        bounds = declared.get(name)
        interval = _interval(name, bounds) if bounds is not None else base
        interval = interval.intersect(base)
        limits = function.knot_support.get(name)
        if limits is not None and outside != "linear":
            interval = interval.intersect(Interval(name, limits[0], limits[1]))
        if interval.empty:
            raise ModelingError("service_regime_domain_rule_outside_support:"
                                + str(rule.get("id")) + ":" + name)
        coverage[name] = interval
    # A non-support coordinate is an applicability guard, not a coverage axis:
    # a bounded guard asserts applicability throughout its range, while the
    # support contract covers only the response coordinates.
    guards = {}
    for name in set(domain) - set(support):
        bounds = declared.get(name)
        if bounds is not None:
            guards[name] = _interval(name, bounds)
    return coverage, guards


def _normalize(declarations, index):
    rule = declarations[index]
    if not isinstance(rule, Mapping):
        raise ModelingError("service_regime_requires_mapping")
    if not rule.get("id"):
        raise ModelingError("service_regime_requires_unique_id")
    coordinates = rule.get("coordinates") or {}
    if not isinstance(coordinates, Mapping):
        raise ModelingError("service_regime_coordinates_require_mapping")
    response, function, outside = _response(rule, coordinates)
    execution_mode, execution = _execution(rule)
    continuity = _continuity(rule)
    if continuity is None:
        return None
    if continuity["support"].keys() - coordinates.keys():
        raise ModelingError("service_regime_continuity_support_requires_declared_coordinate")
    if outside != continuity["outside"]:
        raise ModelingError("service_regime_continuity_outside_must_match_response")
    if function.kind in {"affine", "roofline"} and continuity["outside"] != "analytic":
        raise ModelingError("service_regime_continuity_outside_must_match_response")
    coverage, guards = _coverage(rule, coordinates, function, continuity["support"], continuity["outside"])
    return RuleDomain(
        rule_id=str(rule["id"]), index=index, match=rule.get("match") or {},
        coordinates=coordinates, execution_mode=execution_mode, execution=execution,
        group=continuity["group"], support=continuity["support"],
        function=function, response=response, outside=continuity["outside"],
        coverage=coverage, guards=guards)


def _agree(group, domains):
    first = domains[0]
    for domain in domains[1:]:
        for field, value in (("match", domain.match), ("coordinates", domain.coordinates)):
            expected = first.match if field == "match" else first.coordinates
            if _canonical(value) != _canonical(expected):
                raise ModelingError("service_regime_domain_members_differ:"
                                    + group + ":" + field)
        if domain.execution_mode != first.execution_mode:
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":execution_mode")
        if domain.execution_mode == "joint_native" and (
                _canonical(domain.execution) != _canonical(first.execution)):
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":native_boundary")
        if domain.outside != first.outside:
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":outside")
        if dict(domain.support) != dict(first.support):
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":support")
        if dict(domain.guards) != dict(first.guards):
            raise ModelingError("service_regime_domain_members_differ:"
                                + group + ":guards")


def _dimension_probes(intervals, support):
    lower, upper = support.lower, support.upper
    boundaries = {lower, upper}
    for interval in intervals:
        boundaries.add(min(max(interval.lower, lower), upper))
        boundaries.add(min(max(interval.upper, lower), upper))
    ordered = sorted(boundaries)
    probes = set()
    for position, value in enumerate(ordered):
        previous = ordered[position - 1] if position else None
        following = ordered[position + 1] if position + 1 < len(ordered) else None
        inside_support = support.lower <= value <= support.upper
        if inside_support and support.contains(value):
            probes.add(value)
        left = math.nextafter(value, -math.inf)
        if left > (previous if previous is not None else -math.inf) and left >= lower:
            if support.contains(left) or (left == lower and support.lower_inclusive):
                probes.add(left)
        right = math.nextafter(value, math.inf)
        if right < (following if following is not None else math.inf) and right <= upper:
            if support.contains(right) or (right == upper and support.upper_inclusive):
                probes.add(right)
    return tuple(sorted(probes))


def _contains(domain, point):
    return all(interval.contains(point[name])
               for name, interval in domain.coverage.items())


def _owns(domain, point, members):
    if not _contains(domain, point):
        return False
    for name, interval in domain.coverage.items():
        value = point[name]
        if value != interval.upper or interval.lower == interval.upper:
            continue
        for other in members:
            if other is domain:
                continue
            other_interval = other.coverage[name]
            if (other_interval.lower == value and other_interval.lower_inclusive
                    and _contains(other, point)):
                return False
    return True


def _owners(point, members):
    return [domain for domain in members if _owns(domain, point, members)]


def _probe_coverage(group, domains, support):
    dimensions = tuple(sorted(support))
    per_dimension = [_dimension_probes([domain.coverage[name] for domain in domains],
                                       support[name]) for name in dimensions]
    total = math.prod(len(probes) for probes in per_dimension)
    if total > MAX_PROBES:
        raise ModelingError("service_regime_domain_validation_budget_exceeded:"
                            + group + ":" + str(total))
    for combination in product(*per_dimension):
        point = dict(zip(dimensions, combination))
        owners = _owners(point, domains)
        if not owners:
            raise ModelingError("service_regime_domain_interior_gap:"
                                + group + ":" + _canonical(point))
        if len(owners) > 1:
            raise ModelingError("service_regime_domain_interior_overlap:"
                                + group + ":" + ",".join(d.rule_id for d in owners))
    return total


def _shared_face(first, second, support):
    ranges = {}
    touching = False
    for name in support:
        left, right = first.coverage[name], second.coverage[name]
        lower, upper = max(left.lower, right.lower), min(left.upper, right.upper)
        if lower > upper:
            return None
        if lower == upper and (left.lower < left.upper or right.lower < right.upper):
            touching = True
        ranges[name] = (lower, upper)
    return ranges if touching else None


def _face_samples(domains, name, lower, upper):
    samples = {lower, upper}
    for domain in domains:
        for value in domain.function.breakpoints.get(name, ()):
            if lower <= value <= upper:
                samples.add(value)
    ordered = sorted(samples)
    samples.update((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    return sorted(samples)


def _rate_dependent(response):
    if response.get("kind") == "roofline":
        return True
    return any(term.get("scale_with") for term in response.get("terms", ()))


def _limit(domain, values):
    try:
        result = domain.function.evaluate(values)
    except ModelingError as exc:
        raise ModelingError("service_regime_domain_seam_response_invalid:"
                            + domain.rule_id) from exc
    if result.duration_us is None:
        raise ModelingError("service_regime_domain_seam_uncovered:" + domain.rule_id)
    return result.duration_us


def _check_seams(group, domains, support):
    checked = 0
    for position, first in enumerate(domains):
        for second in domains[position + 1:]:
            ranges = _shared_face(first, second, support)
            if ranges is None:
                continue
            if _canonical(first.response) == _canonical(second.response):
                continue
            if _rate_dependent(first.response) or _rate_dependent(second.response):
                raise ModelingError("service_regime_domain_seam_requires_provable_equivalence:"
                                    + group + ":" + first.rule_id + ":" + second.rule_id)
            axes = {name: _face_samples((first, second), name, *ranges[name])
                    for name in ranges}
            vertices = math.prod(len(samples) for samples in axes.values())
            if vertices > MAX_FACE_VERTICES:
                raise ModelingError("service_regime_domain_validation_budget_exceeded:"
                                    + group + ":seam")
            names = tuple(axes)
            for combination in product(*(axes[name] for name in names)):
                values = dict(zip(names, combination))
                left, right = _limit(first, values), _limit(second, values)
                scale = max(abs(left), abs(right))
                if abs(left - right) > ATOL_US + RTOL * scale:
                    raise ModelingError("service_regime_domain_seam_limit_mismatch:"
                                        + group + ":" + first.rule_id + ":" + second.rule_id)
                checked += 1
    return checked


def _overlaps_support(rule, member_raw, domain):
    if not rules_overlap(rule, member_raw):
        return False
    declared = rule.get("domain") or {}
    for name, interval in domain.support.items():
        bounds = declared.get(name) if isinstance(declared, Mapping) else None
        if bounds is None:
            continue
        if _interval(name, bounds).intersect(interval).empty:
            return False
    return True


def compile_service_regime_domains(declarations):
    """Compile every declared continuity group and return bounded reports.

    Raises ModelingError when a group is inconsistent, when its support is not
    partitioned exactly once by member domains and knot support, or when a
    shared-face limit is not provably continuous.
    """
    if isinstance(declarations, Mapping) or not isinstance(declarations, (list, tuple)):
        raise ModelingError("service_regime_declarations_require_sequence")
    domains, legacy, identifiers = {}, [], set()
    for index, rule in enumerate(declarations):
        if isinstance(rule, Mapping):
            identifier = rule.get("id")
            if identifier in identifiers:
                raise ModelingError("service_regime_requires_unique_id")
            identifiers.add(identifier)
        domain = _normalize(declarations, index)
        if domain is None:
            legacy.append(rule)
        else:
            domains.setdefault(domain.group, []).append(domain)
    groups = []
    for group, members in domains.items():
        _agree(group, members)
        support = members[0].support
        probes = _probe_coverage(group, members, support)
        seams = _check_seams(group, members, support)
        groups.append({
            "group": group,
            "execution_mode": members[0].execution_mode,
            "response_basis": members[0].response.get("kind", "affine"),
            "outside": members[0].outside,
            "support": {name: interval.to_dict() for name, interval in support.items()},
            "guards": {name: interval.to_dict() for name, interval in members[0].guards.items()},
            "members": [domain.rule_id for domain in members],
            "coverage": {domain.rule_id: {name: interval.to_dict()
                                          for name, interval in domain.coverage.items()}
                         for domain in members},
            "probes": probes, "seam_vertices": seams,
        })
    for rule in legacy:
        if not isinstance(rule, Mapping):
            continue
        match = _canonical(rule.get("match") or {})
        for group, members in domains.items():
            if match != _canonical(members[0].match):
                continue
            if any(_overlaps_support(rule, declarations[domain.index], domain)
                   for domain in members):
                raise ModelingError("service_regime_domain_requires_coverage_contract:"
                                    + group + ":" + str(rule.get("id")))
    return tuple(groups)


def validate_service_regime_domains(declarations):
    """Static compatibility/coverage checker exported for profiler importers."""
    return compile_service_regime_domains(declarations)


def rules_overlap(first, second):
    """Conservative static overlap of two declarations on shared coordinates."""
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ModelingError("service_regime_requires_mapping")
    if _canonical(first.get("match") or {}) != _canonical(second.get("match") or {}):
        return False
    left = first.get("coordinates") or {}
    right = second.get("coordinates") or {}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ModelingError("service_regime_coordinates_require_mapping")
    for name in set(left) & set(right):
        first_bounds = (first.get("domain") or {}).get(name)
        second_bounds = (second.get("domain") or {}).get(name)
        if first_bounds is None or second_bounds is None:
            continue
        if _interval(name, first_bounds).intersect(_interval(name, second_bounds)).empty:
            return False
    return True
_MISSING = object()
_FIXED_EVIDENCE_KINDS = frozenset({
    "legacy_whole_call", "whole_call", "whole-call", "observed_replay",
    "legacy_observation"})


def _attr(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _condition_check(conditions, context):
    """Use T06's declared-condition semantics; never a weaker local matcher."""
    if not conditions:
        return {"applies": True, "checks": [], "policy": "no_declared_conditions"}
    from mindie_sim.observations.calibration import service_conditions_match
    return service_conditions_match({"conditions": dict(conditions)}, dict(context or {}))


def _stage(plan, name):
    for stage in plan.get("stages", ()):
        if getattr(stage, "name", None) == name:
            return stage
    return None


def _fragment_for(plan, stage):
    if stage is None:
        return None
    indexed = (plan.get("fragments") or {}).get(getattr(stage, "name", ""))
    if indexed is not None:
        return indexed
    return (getattr(stage, "fragment", None)
            or (plan.get("occurrences") or {}).get(getattr(stage, "name", "")))


def _resource_identity(plan, resource):
    """Resolve one logical role or physical key through the actual context.

    A service or binding may name the logical role (``hbm``) or the physical
    resource key produced by ``RecipeContext.placement`` (``hbm@r0:s0``).  Both
    spellings must resolve to the same physical occurrence; otherwise the same
    work, owner partition or fixed component would be judged once per spelling.
    """
    if resource is None:
        return None, None, frozenset()
    text = str(resource)
    context = plan.get("context")
    names = dict(_attr(context, "resource_names", {}) or {})
    modes = dict(_attr(context, "service_modes", {}) or {})
    reverse = {str(name): str(role) for role, name in names.items()}
    if text in modes:
        role = str(modes[text][0])
    elif text in reverse:
        role = reverse[text]
    else:
        role = text
    physical = str(names.get(role, text))
    return role, physical, frozenset({text, role, physical})


def _stage_resource_aliases(plan, stage):
    """Logical and physical names actually issued by one selected stage."""
    resources = set()
    fragment = _fragment_for(plan, stage)
    if fragment is not None and hasattr(fragment, "work"):
        for item in fragment.work.issued():
            resources.update(_resource_identity(plan, item.resource)[2])
    for key in (getattr(stage, "work", {}) or {}):
        resources.update(_resource_identity(plan, key)[2])
    return resources


def _service_stage_overlap(plan, stage, resources):
    aliases = _stage_resource_aliases(plan, stage)
    return any(_resource_identity(plan, resource)[2] & aliases
               for resource in resources)


def _resolve_actual_work(plan):
    """One dereference of the selected program's stage/fragment/context work.

    The returned instances are the only authority for work amount, native unit,
    repeat count, physical resource and actual owner partition.  Term bindings,
    external bindings, coverage and fixed checks all consume this same map.
    """
    instances, fragments = {}, {}
    for stage in plan.get("stages", ()):
        name = getattr(stage, "name", "")
        fragment = _fragment_for(plan, stage)
        fragments[name] = fragment
        if fragment is None or not hasattr(fragment, "work"):
            continue
        repeats = int(getattr(fragment, "repeats", 1))
        for item in fragment.work.issued():
            role, physical, _aliases = _resource_identity(plan, item.resource)
            key = (name, physical)
            entry = {"stage": name, "resource": item.resource,
                     "logical_resource": role, "physical_resource": physical,
                     "unit": item.unit, "repeats": repeats,
                     "amount": float(item.amount) * repeats,
                     "owner_units": _owner_units(plan, stage, physical),
                     "fragment": getattr(fragment, "name", "")}
            if key in instances:
                instances[key]["ambiguous"] = True
            else:
                instances[key] = entry
    return {"instances": instances, "fragments": fragments}


def _stage_work(plan, stage, resource, multiplicity, work_unit=None,
                resolved=None):
    """Read one work reference from the already resolved actual instance."""
    role, physical, aliases = _resource_identity(plan, resource)
    result = {"amount": None, "unit": None,
              "basis": "actual_work_instance_missing",
              "repeats": None, "owner_units": None, "instance": None,
              "delivered": False, "logical_resource": role,
              "resource": physical, "fragment": None}
    if stage is None:
        result["basis"] = "stage_missing"
        return result
    resolved = resolved if resolved is not None else _resolve_actual_work(plan)
    stage_name = getattr(stage, "name", "")
    fragment = resolved["fragments"].get(stage_name)
    result["fragment"] = getattr(fragment, "name", None)
    matched = []
    for key, entry in resolved["instances"].items():
        if key[0] != stage_name:
            continue
        item_role, item_physical, _item_aliases = _resource_identity(
            plan, entry["resource"])
        if (key[1] == physical or entry["resource"] in aliases
                or item_role == role):
            matched.append(entry)
    if len(matched) != 1:
        result["basis"] = ("actual_work_instance_ambiguous" if matched
                           else "actual_work_instance_missing")
        result["repeats"] = int(getattr(fragment, "repeats", 1))
        return result
    entry = matched[0]
    if entry.get("ambiguous"):
        result["basis"] = "actual_work_instance_ambiguous"
        result["repeats"] = entry["repeats"]
        return result
    if not entry["unit"]:
        result["basis"] = "actual_native_unit_missing"
        result["repeats"] = entry["repeats"]
        result["owner_units"] = entry["owner_units"]
        return result
    basis = ("program_fragment_real_zero" if entry["amount"] == 0
             else "program_fragment_work")
    if work_unit is not None and work_unit != entry["unit"]:
        basis = "declared_unit_conflict"
    result.update({
        "amount": entry["amount"], "unit": entry["unit"], "basis": basis,
        "repeats": entry["repeats"],
        "multiplicity_matches": int(multiplicity) == entry["repeats"],
        "owner_units": entry["owner_units"],
        "instance": (stage_name, entry["physical_resource"]),
        "logical_resource": entry["logical_resource"],
        "resource": entry["physical_resource"],
        "fragment": entry["fragment"], "delivered": True,
    })
    return result


def _owner_units(plan, stage, resource):
    """Actual owner partition keyed by either physical key or logical role."""
    role, physical, aliases = _resource_identity(plan, resource)
    placement = dict(getattr(stage, "placement", {}) or {})
    for key in (physical, role):
        if key in placement and placement[key]:
            return len(tuple(placement[key]))
    for key, units in placement.items():
        if key in aliases and units:
            return len(tuple(units))
    context = plan.get("context")
    units = _attr(context, "resource_units", {}) or {}
    for key in (role, physical):
        if key in units and int(units[key] or 0):
            return int(units[key])
    return None


def _actual_native(plan, rule_execution, context):
    values = [plan.get("native_boundary")]
    identity = _attr(context, "service_identity", {}) or {}
    values.append(identity.get("kernel") if isinstance(identity, Mapping) else None)
    if isinstance(rule_execution, Mapping):
        values.append(rule_execution.get("native_kernel"))
    present = sorted({str(value).strip() for value in values if value})
    if len(present) > 1:
        return None, present
    return (present[0] if present else None), present


def _service_problem(record, kind, native):
    problems = []
    if record["boundary"] not in {"resource_phase", "device_kernel"}:
        problems.append("service_boundary_must_be_resource_phase_or_device_kernel")
    if not record["phase"]:
        problems.append("service_requires_phase")
    if not record["work_unit"]:
        problems.append("service_requires_work_unit")
    try:
        positive(record["effective_rate"])
    except (ModelingError, TypeError):
        problems.append("service_effective_rate_requires_positive_finite")
    ident = record["identification"]
    if kind == "independent_service":
        if record["scope"] != "independent_resource":
            problems.append("service_scope_does_not_match_replacement_kind")
        if (ident.get("status") != "identified"
                or ident.get("resource_identified") is not True):
            problems.append("service_does_not_declare_identified_resource")
        if len(record["resources"]) != 1:
            problems.append("independent_service_requires_exactly_one_resource")
    else:
        if record["scope"] != "joint_phase":
            problems.append("service_scope_does_not_match_replacement_kind")
        if ident.get("status") != "joint_identified":
            problems.append("joint_service_does_not_declare_joint_identified_status")
        if not record["resources"]:
            problems.append("joint_service_requires_resources")
        if not record["native_boundary"]:
            problems.append("joint_service_requires_native_boundary")
    if native and record["native_boundary"] and record["native_boundary"] != native:
        problems.append("service_native_boundary_does_not_match_actual_program")
    return problems


def _fixed_effect_text(record):
    identification = record.get("identification") or {}
    provenance = record.get("provenance") or {}
    fit = provenance.get("fit") if isinstance(provenance.get("fit"), Mapping) else {}
    return (identification.get("included_fixed_effects")
            or provenance.get("included_effects")
            or (fit or {}).get("included_fixed_effects"))


def _fixed_occurrence_stage(plan, stage_name):
    """Actual stage carrying the fixed latency for one declared fixed row.

    T04's real resource-service lowering puts the resolved startup on a
    latency-only ``setup`` stage before the work ``service`` stage.  A fixed
    component therefore belongs to that actual occurrence, not to every work
    term that shares its stage.
    """
    by_name = {getattr(stage, "name", None): stage
               for stage in plan.get("stages", ())}
    stage = by_name.get(stage_name)
    if stage is None:
        return None
    if not (getattr(stage, "work", {}) or {}) and getattr(stage, "latency", 0) > 0:
        return stage_name
    for name in getattr(stage, "after", ()) or ():
        prior = by_name.get(name)
        if (prior is not None and not (getattr(prior, "work", {}) or {})
                and getattr(prior, "latency", 0) > 0):
            return name
    return stage_name


def _first_invocation_stage(plan):
    for stage in plan.get("stages", ()):
        if (not (getattr(stage, "work", {}) or {})
                and getattr(stage, "latency", 0) > 0
                and getattr(stage, "fragment", None) is None):
            return getattr(stage, "name", None)
    stages = tuple(plan.get("stages", ()) or ())
    return getattr(stages[0], "name", None) if stages else None


def _invocation_evidence(plan):
    """Read T04's actual invocation timing from the selected context."""
    context = plan.get("context")
    timing = _attr(context, "invocation_timing")
    if timing is None:
        return None
    try:
        startup_us = nonnegative(_attr(timing, "startup_ms", 0)) * 1000.0
        minimum_us = nonnegative(_attr(timing, "minimum_ms", 0)) * 1000.0
    except ModelingError as exc:
        raise ModelingError("migration_actual_invocation_timing_invalid") from exc
    provenance = _attr(timing, "provenance", {}) or {}
    return {"startup_us": startup_us, "minimum_us": minimum_us,
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else {}}


def _fixed_check_stage(plan, stage_name):
    """Work stage whose phase/resources own a fixed component.

    A fixed row may name the latency-only invocation stage (the actual fixed
    occurrence) or the work stage.  Resource and phase checks always use the
    associated work stage, never the latency-only setup itself.
    """
    stage = _stage(plan, stage_name)
    if stage is None:
        return None
    if (getattr(stage, "work", {}) or {}) or _fragment_for(plan, stage) is not None:
        return stage
    for candidate in plan.get("stages", ()):
        if stage_name in (getattr(candidate, "after", ()) or ()) and (
                (getattr(candidate, "work", {}) or {})
                or _fragment_for(plan, candidate) is not None):
            return candidate
    return stage


def _invocation_fixed_occurrence(plan, invocation):
    """One independently declared T04 invocation startup, charged once."""
    if not invocation or invocation["startup_us"] <= 0:
        return None
    occurrence_stage = _first_invocation_stage(plan)
    return {
        "kind": "invocation_startup", "service_id": None,
        "stage": occurrence_stage, "actual_stage": occurrence_stage,
        "phase": None, "fixed_us": invocation["startup_us"],
        "native_boundary": "",
        "included_fixed_effects": (
            "T04 actual invocation startup declared separately from the "
            "selected service formula and charged once"),
        "fixed_components": "invocation_startup",
        "covers_invocation_startup": True,
        "invocation_startup_us": invocation["startup_us"],
    }


def _fixed_rows(plan, services, stage_work_phases, legacy_fixed, invocation=None):
    rows, occurrences, conflicts, notes = [], [], [], []
    names = {stage["name"] for stage in plan.get("program_view") or ()}
    if not plan.get("fixed"):
        if legacy_fixed > 0 or any(
                term.get("kind") == "zero_slope"
                for term in plan.get("decomposition", {}).get("terms", ())):
            conflicts.append("legacy_fixed_component_unbound")
        else:
            occurrence = _invocation_fixed_occurrence(plan, invocation)
            if occurrence is not None:
                occurrences.append(occurrence)
        return rows, occurrences, conflicts, notes
    if not names:
        conflicts.append("fixed_binding_requires_actual_stage")
        return rows, occurrences, conflicts, notes
    seen = set()
    for item in plan["fixed"]:
        sid = item["service_id"]
        record = services.get(sid)
        stage = item.get("stage")
        if stage is None and len(names) == 1:
            stage = next(iter(names))
        if stage is None or stage not in names:
            conflicts.append("fixed_binding_requires_actual_stage:" + sid)
            continue
        if record is None:
            conflicts.append("unknown_fixed_service:" + sid)
            continue
        actual_stage = _fixed_occurrence_stage(plan, stage)
        check_stage = _fixed_check_stage(plan, stage)
        phases = stage_work_phases.get(getattr(check_stage, "name", stage), set())
        if phases and record["phase"] not in phases:
            conflicts.append("fixed_phase_does_not_match_stage_work:" + sid)
        if record["resources"] and not _service_stage_overlap(
                plan, check_stage, record["resources"]):
            conflicts.append("fixed_service_resource_not_on_actual_stage:" + sid)
        if (actual_stage, sid) in seen:
            conflicts.append("fixed_boundary_duplicated:" + str(actual_stage) + ":" + sid)
            continue
        seen.add((actual_stage, sid))
        covers_invocation = bool(
            invocation and invocation["startup_us"] > 0
            and record["fixed_us"] >= invocation["startup_us"] - 1e-9)
        rows.append({"service_id": sid, "stage": stage,
                     "phase": record["phase"], "new_fixed_us": record["fixed_us"],
                     "native_boundary": record["native_boundary"]})
        if record["fixed_us"] > 0:
            occurrences.append({
                "kind": "service_fixed", "service_id": sid, "stage": stage,
                "actual_stage": actual_stage, "phase": record["phase"],
                "fixed_us": record["fixed_us"],
                "native_boundary": record["native_boundary"],
                "included_fixed_effects": _fixed_effect_text(record),
                "fixed_components": record["identification"].get("fixed_components"),
                "covers_invocation_startup": covers_invocation,
                "invocation_startup_us": (invocation or {}).get("startup_us"),
            })
    by_stage = {}
    for occurrence in occurrences:
        if occurrence["kind"] != "service_fixed":
            continue
        by_stage.setdefault(occurrence["actual_stage"], []).append(
            occurrence["service_id"])
    for stage, ids in by_stage.items():
        if len(ids) > 1:
            conflicts.append("multiple_fixed_components_on_stage:" + str(stage))
    if invocation and invocation["startup_us"] > 0 and not any(
            occurrence["covers_invocation_startup"] for occurrence in occurrences):
        occurrence = _invocation_fixed_occurrence(plan, invocation)
        if occurrence is not None:
            occurrences.append(occurrence)
    if legacy_fixed == 0 and rows:
        notes.append("new_fixed_without_legacy_fixed")
    return rows, occurrences, conflicts, notes


def _identification_problems(record):
    ident = record.get("identification") or {}
    fit = (record.get("provenance") or {}).get("fit") or {}
    rate_source = ident.get("rate_source")
    fixed_source = ident.get("fixed_source")
    train_work = (ident.get("train_work") or fit.get("training_work")
                  or fit.get("train_work") or [])
    try:
        distinct = len({float(work) for work in train_work})
    except (TypeError, ValueError):
        distinct = 0
    problems = []
    if rate_source == "fitted_with_declared_fixed":
        # Fixed is known/declared; one work point identifies the remaining rate.
        rate_ok = distinct >= 1
        fixed_ok = fixed_source == "declared"
    elif rate_source == "declared":
        # Rate is known/declared; one work point identifies the remaining fixed.
        rate_ok = True
        fixed_ok = fixed_source == "fitted" and ident.get("fixed_identified") is True
    elif rate_source == "fitted":
        # Rate and fixed are both unknown: two work sizes are required.
        rate_ok = distinct >= 2
        fixed_ok = fixed_source == "fitted" and ident.get("fixed_identified") is True
    else:
        rate_ok = False
        fixed_ok = False
    if not rate_ok:
        problems.append("replacement_rate_not_independently_identified")
    if not fixed_ok:
        problems.append("replacement_fixed_not_independently_identified")
    return problems


def _undelivered_stage_work(plan, resolved=None):
    """Enumerate positive work whose authoritative fragment/unit is absent."""
    resolved = resolved if resolved is not None else _resolve_actual_work(plan)
    undelivered = []
    for stage in plan.get("stages", ()):
        fragment = resolved["fragments"].get(getattr(stage, "name", ""))
        issued = set()
        if fragment is not None and hasattr(fragment, "work"):
            for item in fragment.work.issued():
                issued |= _resource_identity(plan, item.resource)[2]
        for resource, amount in (getattr(stage, "work", {}) or {}).items():
            try:
                positive_work = float(amount) > 0
            except (TypeError, ValueError):
                positive_work = False
            _role, physical, aliases = _resource_identity(plan, resource)
            if positive_work and not (issued & aliases):
                undelivered.append({
                    "stage": getattr(stage, "name", ""),
                    "resource": resource,
                    "physical_resource": physical,
                    "declared_engine_work": float(amount),
                    "declared_unit": (getattr(stage, "work_units", {}) or {}).get(resource),
                    "reason": "authoritative_fragment_work_instance_or_native_unit_missing",
                })
    return undelivered


def _evaluate_work_reference(plan, item, stage_map, services, service_evidence,
                             context, native, resolved):
    """Resolve one work row against the same actual instance used by coverage.

    Delivery is decided before any rate arithmetic: a missing fragment, a
    missing/ambiguous work instance or a missing native unit returns an
    undelivered result with no amount.  Every rate scope consumes this result.
    """
    result = {"item": item, "problems": [], "undelivered": False,
              "stage_missing": False, "projected": None, "instance": None,
              "rate_row": None, "bound": False, "phase": None,
              "record": None, "service_id": item["service_id"]}
    if item["stage"] is None:
        result["stage_missing"] = True
        return result
    stage = stage_map.get(item["stage"])
    if stage is None:
        result["problems"].append("work_stage_not_in_program:" + str(item["stage"]))
        return result
    record = services.get(item["service_id"])
    if record is None:
        result["problems"].append("unknown_service:" + item["service_id"])
        return result
    result["record"] = record
    evidence = service_evidence.get(item["service_id"], {})
    if not evidence.get("valid", False):
        result["problems"].append("binding_service_invalid")
    elif not evidence.get("identification_ready", False):
        result["problems"].append("binding_service_identification_pending")
    projected = _stage_work(plan, stage, item["resource"], item["multiplicity"],
                            item["work_unit"], resolved=resolved)
    result["projected"] = projected
    if not projected["delivered"]:
        result["undelivered"] = True
        result["problems"].append(projected["basis"])
        return result
    result["instance"] = projected["instance"]
    if projected["basis"] == "declared_unit_conflict":
        result["problems"].append("binding_work_unit_does_not_match_actual_stage")
    binding_role, _binding_physical, binding_aliases = _resource_identity(
        plan, item["resource"])
    _actual_role, _actual_physical, actual_aliases = _resource_identity(
        plan, projected["resource"])
    service_aliases = set()
    for resource in record["resources"]:
        service_aliases.update(_resource_identity(plan, resource)[2])
    if service_aliases:
        if not (service_aliases & actual_aliases):
            result["problems"].append("actual_work_resource_not_covered_by_service")
        if not (service_aliases & binding_aliases):
            result["problems"].append("actual_stage_resource_not_covered_by_service")
    if projected["basis"] == "program_fragment_real_zero":
        result["problems"].append("service_bound_for_real_zero_work")
    if projected.get("multiplicity_matches") is False:
        result["problems"].append("multiplicity_does_not_match_actual_occurrence")
    if projected["unit"] != record["work_unit"]:
        result["problems"].append("stage_work_unit_does_not_match_service")
    if item["phase"] != record["phase"]:
        result["problems"].append("binding_phase_does_not_match_service")
    if item["native_boundary"]:
        if native and item["native_boundary"] != native:
            result["problems"].append(
                "binding_native_boundary_does_not_match_actual_program")
        elif record["native_boundary"] and item["native_boundary"] != record["native_boundary"]:
            result["problems"].append(
                "binding_native_boundary_does_not_match_service")
    work_scope = item["work_scope"] or record["rate_scope"]
    rate = (record.get("aggregate_rate") if work_scope == "aggregate"
            else record.get("per_owner_rate"))
    if rate is None:
        result["problems"].append("service_scope_rate_missing")
    owner_units = projected.get("owner_units")
    if work_scope == "per_owner":
        if owner_units is None:
            result["problems"].append("per_owner_work_requires_actual_owner_units")
        elif int(record.get("active_units") or 0) != int(owner_units):
            result["problems"].append(
                "per_owner_active_units_do_not_match_actual_owner_units")
    work_amount = projected["amount"]
    if work_scope == "per_owner":
        work_amount = (work_amount / owner_units) if owner_units else None
    native_work_map = _attr(context, "native_work", {}) or {}
    native_work = native_work_map.get(item["resource"])
    if native_work is None:
        native_work = native_work_map.get(binding_role)
    result["rate_row"] = {
        "term_index": item["term_index"], "service_id": item["service_id"],
        "stage": item["stage"], "work_scope": work_scope,
        "actual_work": work_amount, "aggregate_work": projected["amount"],
        "actual_unit": projected["unit"], "owner_units": owner_units,
        "rate_used": rate, "amount_basis": projected["basis"],
        "multiplicity": item["multiplicity"],
        "actual_repeats": projected.get("repeats"),
        "native_work": native_work,
        "native_work_difference": (
            None if native_work is None
            else projected["amount"] - native_work),
        "instance": list(projected["instance"]), "charged_once": True,
        "difference_is_diagnostic": True}
    result["phase"] = item["phase"]
    result["bound"] = not result["problems"]
    return result


def evaluate_migration_binding(decomposition, rule_match, rule_id, plan, services, *,
                               execution=None, old_provenance_negative=False,
                               old_training_tokens=(), coordinate_declarations=None):
    """One evaluation of actual stage/work/fixed ownership and source roles.

    ``plan`` carries read-only ``KernelProgram``/``Stage``/``RecipeContext``
    references. Old term indices and formulas are diagnostics; actual stage
    work, unit, owner and boundary are the authority. Missing program/work or
    training evidence is pending/replay, never ready.
    """
    execution = execution if isinstance(execution, Mapping) else {}
    if not plan.get("program_view") and (plan["work"] or plan["fixed"]):
        return {"status": "awaiting_program_binding",
                "blockers": ["migration_program_binding_required"],
                "conflicts": [], "unresolved_conditions": [],
                "coverage": {"complete": False, "program": None,
                             "work_terms": [], "external_work": [], "fixed": []},
                "service_evidence": {}, "conditions": {},
                "diagnostics": {"notes": ["actual_program_stage_reference_required"]},
                "scope": {"kind": plan["scope"],
                          "slice_conditions": dict(plan.get("slice_conditions") or {})},
                "replacement_scope": None}
    context = plan.get("context")
    native, native_values = _actual_native(plan, execution, context)
    kind, scope = plan["replacement_kind"], plan["scope"]
    context_values = plan.get("condition_context") or {}
    slice_conditions = plan.get("slice_conditions") or {}
    conflicts, unresolved, notes, rate_rows = [], [], [], []
    identification_pending, service_evidence = [], {}
    referenced = {item["service_id"] for item in plan["work"]}
    referenced |= {item["service_id"] for item in plan["fixed"]}
    for sid in sorted(referenced):
        record = services.get(sid)
        if record is None:
            conflicts.append("unknown_service:" + sid)
            continue
        problems = _service_problem(record, kind, native)
        provenance = record["provenance"]
        roles = provenance_roles(provenance)
        training = roles["training"] or roles["other"]
        source_problems = []
        if not training:
            source_problems.append("replacement_service_requires_training_sources")
        source_problems.extend(_identification_problems(record))
        legacy_overlap = sorted(set(training) & set(old_training_tokens))
        if old_provenance_negative and legacy_overlap:
            notes.append("legacy_training_point_shared_with_independent_identification")
        if str(provenance.get("kind") or "").lower() in _FIXED_EVIDENCE_KINDS:
            source_problems.append("replacement_service_is_not_independent_evidence")
        checks = _condition_check(record["conditions"], context_values)
        missing_paths = {check["path"] for check in checks["checks"]
                         if not check["present"]}
        for check in checks["checks"]:
            if not check["present"]:
                unresolved.append({"scope": "service", "service_id": sid, **check})
            elif not check["matched"]:
                conflicts.append("service_condition_mismatch:" + sid + ":" + check["path"])
        for path, expected in record["conditions"].items():
            if path == "regime_rule_id":
                if str(expected) != str(rule_id):
                    conflicts.append("service_regime_rule_id_mismatch:" + sid)
                continue
            if path in {"native_boundary", "native_kernel"} or path in missing_paths:
                continue
            rule_expected = rule_match.get(path, _MISSING)
            slice_expected = slice_conditions.get(path, _MISSING)
            if rule_expected is not _MISSING and same_declared_value(rule_expected, expected):
                continue
            if scope == "slice" and slice_expected is not _MISSING and same_declared_value(
                    slice_expected, expected):
                continue
            conflicts.append("service_condition_outside_rule_scope:" + sid + ":" + path)
        service_evidence[sid] = {
            "service_id": sid, "scope": record["scope"],
            "resources": list(record["resources"]), "phase": record["phase"],
            "work_unit": record["work_unit"], "boundary": record["boundary"],
            "effective_rate": record["effective_rate"],
            "aggregate_rate": record.get("aggregate_rate"),
            "per_owner_rate": record.get("per_owner_rate"),
            "rate_scope": record["rate_scope"], "active_units": record.get("active_units"),
            "fixed_us": record["fixed_us"], "native_boundary": record["native_boundary"],
            "identification": dict(record["identification"]),
            "rate_source": record["identification"].get("rate_source"),
            "fixed_source": record["identification"].get("fixed_source"),
            "train_work": record["identification"].get("train_work"),
            "fixed_components": record["identification"].get("fixed_components"),
            "included_fixed_effects": (
                record["identification"].get("included_fixed_effects")
                or (record["provenance"] or {}).get("included_effects")
                or (record["provenance"].get("fit") or {}).get("included_fixed_effects")),
            "provenance": dict(provenance), "source_roles": roles,
            "legacy_training_overlap": legacy_overlap,
            "conditions": checks, "problems": sorted(set(problems)),
            "source_problems": sorted(set(source_problems)),
            "valid": not problems, "identification_ready": not source_problems}
        conflicts.extend(sid + ":" + problem for problem in problems)
        identification_pending.extend(sid + ":" + problem for problem in source_problems)
    rule_check = _condition_check(rule_match or {}, context_values)
    slice_check = (_condition_check(slice_conditions, context_values) if slice_conditions
                   else {"applies": True, "checks": [], "policy": "no_slice_conditions"})
    for scope_name, check in (("rule", rule_check), ("slice", slice_check)):
        for item in check["checks"]:
            if not item["present"]:
                unresolved.append({"scope": scope_name, **item})
            elif not item["matched"]:
                conflicts.append(scope_name + "_condition_mismatch:" + item["path"])
    if len(native_values) > 1:
        conflicts.append("actual_native_boundary_conflict:" + ",".join(native_values))
    terms = [term for term in decomposition.get("terms", ())
             if term.get("kind") in {"resource_rate", "unidentified_slope",
                                     "declared_resource_supply"}]
    legacy_fixed = float(decomposition.get("fixed_us", 0))
    stage_map = {getattr(stage, "name", None): stage
                 for stage in plan.get("stages", ())}
    resolved = _resolve_actual_work(plan)
    actual_instances = resolved["instances"]
    invocation = _invocation_evidence(plan)
    undelivered_work = _undelivered_stage_work(plan, resolved)
    evaluations, bound_stages, instance_services = [], {}, {}
    for item in plan["work"]:
        evaluation = _evaluate_work_reference(
            plan, item, stage_map, services, service_evidence, context, native,
            resolved)
        evaluations.append(evaluation)
        if evaluation["instance"] is not None:
            instance_services.setdefault(evaluation["instance"], set()).add(
                item["service_id"])
            bound_stages.setdefault(item["stage"], set()).add(item["phase"])
        if evaluation["undelivered"]:
            projected = evaluation["projected"] or {}
            undelivered_work.append({
                "stage": item["stage"], "resource": item["resource"],
                "physical_resource": projected.get("resource"),
                "reason": projected.get("basis", "actual_work_instance_missing")})
        prefix = ("term" + str(item["term_index"]) + ":"
                  if item["term_index"] is not None
                  else "external:" + str(item["stage"]) + ":")
        for problem in evaluation["problems"]:
            if problem == "binding_service_identification_pending":
                identification_pending.append(prefix + item["service_id"])
            elif problem in {"actual_work_instance_missing",
                             "actual_work_instance_ambiguous",
                             "actual_native_unit_missing"}:
                continue
            else:
                conflicts.append(prefix + problem)
    deduplicated, seen_undelivered = [], set()
    for entry in undelivered_work:
        key = (entry.get("stage"),
               entry.get("physical_resource") or entry.get("resource"))
        if key in seen_undelivered:
            continue
        seen_undelivered.add(key)
        deduplicated.append(entry)
    undelivered_work = deduplicated
    by_term = {}
    for evaluation in evaluations:
        index = evaluation["item"]["term_index"]
        if index is not None:
            by_term.setdefault(index, []).append(evaluation)
    work_rows, missing_terms, unbound_terms = [], [], []
    for term in terms:
        index = term["index"]
        bindings = by_term.get(index, [])
        row = {"term_index": index, "coordinate": term["coordinate"],
               "coordinate_declaration": dict(
                   (coordinate_declarations or {}).get(term["coordinate"], {})),
               "legacy_resource": term.get("resource"),
               "legacy_dtype": term.get("dtype"),
               "legacy_us_per_unit": term.get("us_per_unit"),
               "bound": False, "unbound": not bindings, "bindings": [],
               "problems": []}
        if len(bindings) > 1:
            row["problems"].append("work_term_bound_multiple_times")
            conflicts.append("term" + str(index) + ":work_term_bound_multiple_times")
        for evaluation in bindings:
            row["bindings"].append(dict(evaluation["item"]))
            row["problems"].extend(evaluation["problems"])
        if bindings:
            row["bound"] = (len(bindings) == 1 and not row["problems"]
                            and not bindings[0]["undelivered"]
                            and not bindings[0]["stage_missing"])
        else:
            term_aliases = _resource_identity(plan, term.get("resource"))[2]
            matching = [entry for entry in actual_instances.values()
                        if (_resource_identity(plan, entry["resource"])[2]
                            & term_aliases)]
            covered = any(
                instance_services.get(key)
                for key, entry in actual_instances.items()
                if (_resource_identity(plan, entry["resource"])[2]
                    & term_aliases))
            row["actual_resource_covered"] = bool(covered)
            if covered:
                row["diagnostic_only"] = True
            elif matching and any(entry["amount"] > 0 for entry in matching):
                missing_terms.append(index)
            else:
                row["diagnostic_only"] = True
                unbound_terms.append(index)
        work_rows.append(row)
    external_rows = []
    for evaluation in evaluations:
        item = evaluation["item"]
        if item["term_index"] is not None:
            continue
        row = {"stage": item["stage"], "service_id": item["service_id"],
               "resource": item["resource"], "bound": bool(evaluation["bound"]),
               "problems": list(evaluation["problems"])}
        if evaluation["undelivered"]:
            row["undelivered"] = True
        external_rows.append(row)
    rate_rows = []
    for evaluation in evaluations:
        if evaluation["rate_row"] is None:
            continue
        instance = evaluation["instance"]
        evaluation["rate_row"]["charged_once"] = len(
            instance_services.get(instance, set())) <= 1
        rate_rows.append(evaluation["rate_row"])
    uncovered_actual_work, real_zero_work, zero_work_with_service = [], [], []
    for key, instance in sorted(actual_instances.items()):
        label = str(key[0]) + ":" + str(instance["resource"])
        bound_services = instance_services.get(key, set())
        if instance.get("ambiguous"):
            conflicts.append("actual_work_instance_ambiguous:" + label)
            continue
        if instance["amount"] > 0 and not bound_services:
            uncovered_actual_work.append({
                "stage": key[0], "resource": instance["resource"],
                "physical_resource": instance["physical_resource"],
                "amount": instance["amount"], "unit": instance["unit"],
                "repeats": instance["repeats"],
                "blocker": "actual_work_uncovered:" + label})
        if instance["amount"] == 0:
            if bound_services:
                conflicts.append("service_bound_for_real_zero_work:" + label)
                zero_work_with_service.append(label)
            else:
                real_zero_work.append({"stage": key[0],
                                       "resource": instance["resource"],
                                       "unit": instance["unit"]})
        if len(bound_services) > 1:
            conflicts.append("actual_work_bound_to_multiple_services:" + label)
    plan_for_fixed = dict(plan, decomposition=decomposition)
    fixed_rows, fixed_occurrences, fixed_conflicts, fixed_notes = _fixed_rows(
        plan_for_fixed, services, bound_stages, legacy_fixed,
        invocation=invocation)
    conflicts.extend(fixed_conflicts)
    notes.extend(fixed_notes)
    for row in fixed_rows:
        evidence = service_evidence.get(row["service_id"], {})
        if not evidence.get("identification_ready", True):
            identification_pending.append(
                "fixed:" + row["stage"] + ":" + row["service_id"])
    owned_fixed = {(occurrence["actual_stage"], occurrence["service_id"])
                   for occurrence in fixed_occurrences
                   if occurrence["kind"] == "service_fixed"}
    for item in plan["work"]:
        record = services.get(item["service_id"])
        if record and record["fixed_us"] > 0 and item["stage"]:
            occurrence_stage = _fixed_occurrence_stage(plan, item["stage"])
            if (occurrence_stage, item["service_id"]) not in owned_fixed:
                conflicts.append("stage_fixed_component_not_owned:"
                                 + item["stage"] + ":" + item["service_id"])
    complete = (not conflicts and not unresolved and not identification_pending
                and not uncovered_actual_work
                and not zero_work_with_service and not undelivered_work
                and all(evaluation["bound"] for evaluation in evaluations)
                and (legacy_fixed <= 0 or bool(fixed_rows)))
    diagnostic_combination = ("max_by_resource"
                              if decomposition.get("response_class") == "resource_phase_supply"
                              else "sum")
    coverage = {
        "combination": "actual_program_dependencies",
        "diagnostic_combination": diagnostic_combination,
        "program": plan.get("program_view"),
        "work_terms": work_rows, "external_work": external_rows,
        "actual_work_instances": list(actual_instances.values()),
        "uncovered_actual_work": uncovered_actual_work,
        "undelivered_work": undelivered_work,
        "real_zero_work": real_zero_work,
        "zero_work_with_service": zero_work_with_service,
        "fixed": fixed_rows, "fixed_occurrences": fixed_occurrences,
        "invocation": (None if invocation is None else {
            **invocation,
            "fixed_occurrence_count": len(fixed_occurrences),
            "service_fixed_covers_startup": any(
                occurrence["kind"] == "service_fixed"
                and occurrence["covers_invocation_startup"]
                for occurrence in fixed_occurrences)}),
        "complete": complete,
        "fixed_total_us": sum(occurrence["fixed_us"]
                              for occurrence in fixed_occurrences),
        "legacy_fixed_us": legacy_fixed,
    }
    stage_missing = any(evaluation["stage_missing"] for evaluation in evaluations)
    if conflicts:
        status = "conflict"
    elif unresolved:
        status = "awaiting_conditions"
    elif identification_pending:
        status = "observation_replay"
    elif stage_missing:
        status = "awaiting_program_binding"
    elif complete and scope == "slice":
        status = "slice_ready"
    elif complete:
        status = "ready"
    elif (any(evaluation["instance"] is not None for evaluation in evaluations)
          or external_rows or fixed_rows):
        status = "partial"
    else:
        status = "no_binding"
    blockers = sorted(set(conflicts))
    blockers.extend(sorted(set(identification_pending)))
    blockers.extend(sorted({"condition_context_missing:" + str(item.get("path"))
                            for item in unresolved}))
    blockers.extend("legacy_work_term_unbound:" + str(index)
                    for index in missing_terms)
    blockers.extend(item["blocker"] for item in uncovered_actual_work)
    blockers.extend("actual_work_undelivered:" + str(item["stage"]) + ":"
                    + str(item["resource"]) for item in undelivered_work)
    if status == "awaiting_program_binding":
        blockers.append("migration_stage_binding_required")
    if legacy_fixed > 0 and not fixed_rows:
        blockers.append("legacy_fixed_component_unbound")
    if status == "no_binding":
        blockers.append("migration_binding_plan_requires_actual_work")
    return {
        "status": status, "blockers": sorted(set(blockers)),
        "conflicts": sorted(set(conflicts)),
        "unresolved_conditions": unresolved,
        "coverage": coverage, "service_evidence": service_evidence,
        "conditions": {"rule": rule_check, "slice": slice_check,
                       "services": {sid: item["conditions"]
                                    for sid, item in service_evidence.items()}},
        "diagnostics": {
            "old_provenance_negative": bool(old_provenance_negative),
            "old_training_tokens": sorted(old_training_tokens),
            "actual_native_boundary": native,
            "native_boundary_sources": native_values,
            "actual_resources": dict(
                _attr(context, "resource_names", {}) or {}),
            "actual_invocation": invocation,
            "rate_comparison": rate_rows,
            "fixed_comparison": {
                "legacy_fixed_us": legacy_fixed,
                "service_fixed_total_us": sum(
                    occurrence["fixed_us"] for occurrence in fixed_occurrences),
                "difference_is_diagnostic": True},
            "fixed_occurrence_count": len(fixed_occurrences),
            "unbound_legacy_terms": unbound_terms,
            "undelivered_work_instances": undelivered_work,
            "notes": sorted(set(notes + [
                "old_term_indexes_and_formula_are_diagnostic",
                "old_rate_and_fixed_differences_are_diagnostic",
                "actual_work_owner_and_fixed_boundary_are_shared_projection",
                "validation_or_diagnostic_references_do_not_veto_new_training"])),
        },
        "scope": {"kind": scope, "slice_conditions": dict(slice_conditions)},
        "replacement_scope": ("slice" if scope == "slice" else
                              ("independent_resource" if kind == "independent_service"
                               else "joint_phase")),
    }
