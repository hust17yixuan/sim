"""Pure physical-coordinate response functions shared by execution and scoring.

[Role] Evaluate physical supply responses and declared interpolation surfaces.
  Also expose the declared coefficient structure needed to decide whether a
  legacy response can be replaced by a resource/phase service.
[Invariants] Knots are supplied parameters, never learned from query timings.
  Every coordinate, scaled coordinate, coefficient, knot, grid value and rate
  is finite and nonnegative; booleans are never numbers and rates are strictly
  positive before division. A grid requires every corner. Extrapolation never
  clamps an invalid duration. Response dependencies, knot support and cell
  breakpoints are exposed for static coverage compilation; continuity never
  implies global monotonicity. Coefficient decomposition is diagnostic
  evidence; it never evaluates a query or changes the response formula.
[Timeouts & Locks] Bounded parameter tables and binary search; no waits or locks.
[Side-Effect Policy] No I/O, fitting, global mutation or device execution.
"""
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import math

from mindie_sim.schema.common import ModelingError


def nonnegative(value):
    """Return a finite nonnegative float; booleans and non-numbers are invalid."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ModelingError("service_regime_requires_finite_nonnegative_parameters")
    return float(value)


def positive(value):
    """Return a finite strictly positive rate; zero and negatives are invalid."""
    result = nonnegative(value)
    if result == 0:
        raise ModelingError("service_regime_requires_positive_rate")
    return result


def scaled_coordinate(value, scale=1):
    """Apply an explicit recipe scale without silently overflowing to infinity."""
    result = nonnegative(value) * nonnegative(scale)
    if not math.isfinite(result):
        raise ModelingError("service_regime_coordinate_scaling_overflow")
    return result


def _duration(value):
    result = nonnegative(value)
    if not math.isfinite(result):
        raise ModelingError("service_regime_response_not_finite")
    return result


@dataclass(frozen=True)
class ResponseValue:
    duration_us: float | None
    basis: str
    issued_work: float | None = None
    rate_flops_per_s: float | None = None
    active_owners: int | None = None
    stage: str | None = None


@dataclass(frozen=True)
class ResponseFunction:
    kind: str
    fixed_us: float = 0
    terms: tuple[Mapping, ...] = ()
    coordinate: str = ""
    knots: tuple[tuple[float, float], ...] = ()
    outside: str = "analytic"
    coordinates: tuple[float, ...] = ()
    grid: tuple[tuple[str, tuple[float, ...]], ...] = ()
    values_us: tuple[float, ...] = ()
    strides: tuple[int, ...] = ()
    loads: tuple[Mapping, ...] = ()
    resource: str = ""
    dtype: str = ""
    scope: str = ""
    issue: Mapping | None = None
    owner_grid: Mapping | None = None
    owner_coordinate: bool = False

    @property
    def varying_coordinates(self):
        if self.kind == "piecewise_linear":
            return {self.coordinate}
        if self.kind == "multilinear":
            return {name for name, _ in self.grid}
        if self.kind == "measured_dual_load":
            return {"m", "n", "k"} | ({"owners"} if self.owner_coordinate else set())
        return {term["coordinate"] for term in self.terms}

    @property
    def dependencies(self):
        """Physical coordinates the response actually consumes."""
        return frozenset(self.varying_coordinates)

    @property
    def breakpoints(self):
        """Declared cell breakpoints per varying coordinate (empty if analytic)."""
        if self.kind == "piecewise_linear":
            return {self.coordinate: self.coordinates}
        if self.kind == "multilinear":
            return {name: points for name, points in self.grid}
        return {}

    @property
    def knot_support(self):
        """Calibrated knot interval per coordinate; analytic kinds are unbounded."""
        return {name: (points[0], points[-1])
                for name, points in self.breakpoints.items() if points}

    def service_decomposition(self):
        """Return declared resource/phase structure without evaluating a query.

        This is migration evidence only. ``resource_rate`` terms state a work
        coordinate and an effective rate at the declared reference supply;
        ``unidentified_slope`` terms are whole-call slopes with no physical
        resource. A response is decomposable only when every term is a positive
        rate on the same work unit, so no implicit unit conversion or residual
        pricing is introduced.
        """
        fixed = float(self.fixed_us)
        if self.kind == "affine":
            terms = []
            for index, term in enumerate(self.terms):
                coordinate, us_per_unit = term["coordinate"], float(term["us_per_unit"])
                scaling = term.get("scale_with")
                if scaling is None:
                    terms.append({"index": index, "kind": "unidentified_slope",
                                  "coordinate": coordinate,
                                  "us_per_unit": us_per_unit})
                    continue
                reference = float(scaling["reference_rate"])
                if us_per_unit == 0:
                    terms.append({"index": index, "kind": "zero_slope",
                                  "coordinate": coordinate,
                                  "resource": str(scaling["resource"]),
                                  "dtype": str(scaling.get("dtype") or ""),
                                  "reference_rate": reference})
                    continue
                terms.append({"index": index, "kind": "resource_rate",
                              "coordinate": coordinate,
                              "resource": str(scaling["resource"]),
                              "dtype": str(scaling.get("dtype") or ""),
                              "reference_rate": reference,
                              "us_per_unit": us_per_unit,
                              "effective_rate": 1e6 / us_per_unit})
            resources = sorted({term["resource"] for term in terms
                                if term["kind"] == "resource_rate"})
            units = sorted({term["coordinate"] for term in terms
                            if term["kind"] == "resource_rate"})
            blockers = []
            if not terms:
                blockers.append("fixed_only_response_requires_an_existing_service_rate")
            if any(term["kind"] == "unidentified_slope" for term in terms):
                blockers.append("declared_slope_not_scaled_by_a_physical_resource")
            if any(term["kind"] == "zero_slope" for term in terms):
                blockers.append("zero_slope_term_is_a_fixed_response_not_a_rate")
            if len(units) > 1:
                blockers.append("multiple_declared_work_units_require_explicit_conversion")
            decomposable = bool(terms) and not blockers
            return {
                "response_class": ("resource_phase_candidate" if decomposable
                                   else "whole_call_response"),
                "decomposable": decomposable,
                "fixed_us": fixed,
                "terms": terms,
                "resources": resources,
                "work_units": units,
                "service_scope": (
                    "independent_resource" if decomposable and len(resources) == 1
                    else "joint_phase" if decomposable else None),
                "blockers": blockers,
            }
        if self.kind == "roofline":
            terms = [{
                "index": index,
                "kind": "declared_resource_supply",
                "coordinate": term["coordinate"],
                "resource": str(term["resource"]),
                "dtype": str(term.get("dtype") or ""),
                "utilization": float(term["utilization"]),
                "requires_physical_rate": True,
            } for index, term in enumerate(self.terms)]
            resources = sorted({term["resource"] for term in terms})
            units = sorted({term["coordinate"] for term in terms})
            return {
                "response_class": "resource_phase_supply",
                "decomposable": True,
                "fixed_us": fixed,
                "terms": terms,
                "resources": resources,
                "work_units": units,
                "service_scope": ("independent_resource" if len(resources) == 1
                                  and len(units) == 1 else "joint_phase"),
                "blockers": [],
            }
        if self.kind == "measured_dual_load":
            terms = [{
                "index": index,
                "kind": "measured_issue_class_rate",
                "load": str(load["id"]),
                "active_owners": int(load["active_owners"]),
                "effective_rate": float(load["rate_flops_per_s"]),
                "measured_issued_flops": float(load["measured_issued_flops"]),
            } for index, load in enumerate(self.loads)]
            return {
                "response_class": "measured_issue_class",
                "decomposable": False,
                "fixed_us": fixed,
                "terms": terms,
                "resources": [str(self.resource)] if self.resource else [],
                "work_units": ["flops"],
                "service_scope": None,
                "blockers": [
                    "measured_issue_class_is_not_a_shape_generalization",
                    "two_measured_load_classes_cannot_be_replaced_by_one_rate",
                ],
            }
        return {
            "response_class": "whole_kernel_surface",
            "decomposable": False,
            "fixed_us": fixed,
            "terms": [],
            "resources": [],
            "work_units": [],
            "service_scope": None,
            "blockers": ["whole_kernel_shape_surface_requires_observation_replay"],
        }

    @property
    def migration_class(self):
        """Causal class used by T20 readiness; never a production selector."""
        return self.service_decomposition()["response_class"]

    @classmethod
    def from_mapping(cls, raw, coordinate_names):
        """Validate declared parameters without consuming any observation rows."""
        if not isinstance(raw, Mapping):
            raise ModelingError("service_regime_response_requires_mapping")
        if not isinstance(coordinate_names, Mapping):
            raise ModelingError("service_regime_coordinates_require_mapping")
        kind = raw.get("kind", "affine")
        if not isinstance(kind, str):
            raise ModelingError("service_regime_response_kind_unknown")
        if kind == "measured_dual_load":
            if set(raw) - {"kind", "fixed_us", "resource", "dtype", "loads",
                           "scope", "issue", "owner_grid"}:
                raise ModelingError("service_regime_response_unknown_fields")
            for name in ("m", "n", "k"):
                if name not in coordinate_names:
                    raise ModelingError(
                        "measured_dual_load_requires_declared_m_n_k_coordinates")
            owner_coordinate = "owners" in coordinate_names
            owner_grid = raw.get("owner_grid")
            if owner_grid is not None and (not isinstance(owner_grid, Mapping)
                  or set(owner_grid) != {"tile_m", "tile_n", "cores"}
                  or any(isinstance(owner_grid.get(key), bool)
                         or not isinstance(owner_grid.get(key), int)
                         or owner_grid[key] <= 0
                         for key in ("tile_m", "tile_n", "cores"))):
                raise ModelingError("measured_dual_load_invalid_owner_grid")
            resource = raw.get("resource")
            if not isinstance(resource, str) or not resource:
                raise ModelingError("measured_dual_load_requires_resource")
            dtype = raw.get("dtype")
            if resource == "cube" and not dtype:
                raise ModelingError("measured_dual_load_cube_requires_dtype")
            issue = raw.get("issue")
            if (not isinstance(issue, Mapping)
                    or set(issue) - {"native_type", "dtype", "output_dtype",
                                     "m_alignment", "alignment_basis", "source_path",
                                     "source_sha256", "source_revision", "scope", "n", "k",
                                     "weight_layout", "wrapper_source_path",
                                     "wrapper_source_sha256", "wrapper_source_revision",
                                     "wrapper_symbol", "deployment_image_id",
                                     "source_scope", "source_basis"}
                    or not isinstance(issue.get("m_alignment"), int)
                    or isinstance(issue.get("m_alignment"), bool)
                    or issue["m_alignment"] <= 0
                    or not isinstance(issue.get("n"), int)
                    or not isinstance(issue.get("k"), int)
                    or issue["n"] <= 0 or issue["k"] <= 0):
                raise ModelingError("measured_dual_load_requires_certified_issue_rule")
            declared = raw.get("loads")
            if not isinstance(declared, (list, tuple)) or len(declared) != 2:
                raise ModelingError("measured_dual_load_requires_two_measured_loads")
            loads, ids = [], set()
            for load in declared:
                if (not isinstance(load, Mapping)
                        or set(load) - {"id", "active_owners", "rate_flops_per_s",
                                        "measured_issued_flops", "measured_logical_flops",
                                        "event_count", "median_aic_mac_time_ns",
                                        "median_aic_mac_ratio", "median_mte2_time_ns",
                                        "median_elapsed_ns", "cube_utilization_pct",
                                        "limiting_pipe", "provenance"}):
                    raise ModelingError("measured_dual_load_invalid_load_record")
                load_id = load.get("id")
                if load_id not in {"saturated", "underfilled"} or load_id in ids:
                    raise ModelingError("measured_dual_load_requires_saturated_and_underfilled")
                ids.add(load_id)
                owners = load.get("active_owners")
                if (isinstance(owners, bool) or not isinstance(owners, int)
                        or owners <= 0):
                    raise ModelingError("measured_dual_load_requires_positive_active_owners")
                rate = positive(load.get("rate_flops_per_s"))
                issued = positive(load.get("measured_issued_flops"))
                loads.append({**load, "active_owners": owners,
                              "rate_flops_per_s": rate,
                              "measured_issued_flops": issued})
            if ids != {"saturated", "underfilled"}:
                raise ModelingError("measured_dual_load_requires_saturated_and_underfilled")
            if int(loads[0]["active_owners"]) == int(loads[1]["active_owners"]):
                raise ModelingError("measured_dual_load_requires_distinct_owner_counts")
            return cls(kind, nonnegative(raw.get("fixed_us", 0)),
                       loads=tuple(loads), resource=str(resource),
                       dtype=str(dtype or ""), scope=str(raw.get("scope", "")),
                       issue=dict(issue),
                       owner_grid=dict(owner_grid) if owner_grid is not None else None,
                       owner_coordinate=owner_coordinate)
        if kind == "roofline":
            if set(raw) - {"kind", "fixed_us", "services"}:
                raise ModelingError("service_regime_response_unknown_fields")
            services = raw.get("services")
            if not isinstance(services, (list, tuple)) or not services:
                raise ModelingError("service_regime_roofline_requires_service")
            resolved = []
            for term in services:
                if not isinstance(term, Mapping):
                    raise ModelingError("service_regime_roofline_requires_coordinate_and_resource")
                if (set(term) - {"coordinate", "resource", "dtype", "utilization"}
                        or term.get("coordinate") not in coordinate_names
                        or not term.get("resource")):
                    raise ModelingError("service_regime_roofline_requires_coordinate_and_resource")
                utilization = nonnegative(term.get("utilization", 1))
                if not 0 < utilization <= 1:
                    raise ModelingError("service_regime_roofline_requires_positive_utilization_and_precision")
                if term["resource"] == "cube" and not term.get("dtype"):
                    raise ModelingError("service_regime_roofline_requires_positive_utilization_and_precision")
                resolved.append({**term, "utilization": utilization})
            return cls(kind, nonnegative(raw.get("fixed_us", 0)), tuple(resolved))
        if kind == "affine":
            if set(raw) - {"kind", "fixed_us", "terms"}:
                raise ModelingError("service_regime_response_unknown_fields")
            fixed = nonnegative(raw.get("fixed_us", 0))
            declared = raw.get("terms", ())
            if not isinstance(declared, (list, tuple)):
                raise ModelingError("service_regime_affine_terms_require_sequence")
            terms = []
            for term in declared:
                if not isinstance(term, Mapping) or set(term) - {"coordinate", "us_per_unit", "scale_with"}:
                    raise ModelingError("service_regime_term_unknown_fields")
                if term.get("coordinate") not in coordinate_names:
                    raise ModelingError("service_regime_term_requires_declared_coordinate")
                nonnegative(term.get("us_per_unit"))
                scaling = term.get("scale_with")
                if scaling:
                    if (not isinstance(scaling, Mapping)
                            or set(scaling) - {"resource", "reference_rate", "dtype"}
                            or not scaling.get("resource")
                            or "reference_rate" not in scaling):
                        raise ModelingError("service_regime_invalid_rate_scaling")
                    if positive(scaling["reference_rate"]) == 0:
                        raise ModelingError("service_regime_reference_rate_must_be_positive")
                terms.append({**term, **({"scale_with": dict(scaling)} if scaling else {})})
            return cls(kind, fixed, tuple(terms))
        if kind == "multilinear":
            if set(raw) - {"kind", "grid", "values_us", "outside"}:
                raise ModelingError("service_regime_response_unknown_fields")
            declared = raw.get("grid")
            if not isinstance(declared, (list, tuple)) or not declared:
                raise ModelingError("service_regime_grid_requires_declared_coordinates")
            grid = []
            for axis in declared:
                if (not isinstance(axis, Mapping)
                        or set(axis) != {"coordinate", "knots"}
                        or axis.get("coordinate") not in coordinate_names):
                    raise ModelingError("service_regime_grid_requires_declared_coordinates")
                grid.append((axis["coordinate"], _ordered_knots(axis["knots"])))
            grid = tuple(grid)
            if len({name for name, _ in grid}) != len(grid):
                raise ModelingError("service_regime_grid_requires_declared_coordinates")
            raw_values = raw.get("values_us")
            if not isinstance(raw_values, (list, tuple)):
                raise ModelingError("service_regime_grid_requires_every_corner")
            values = tuple(nonnegative(value) for value in raw_values)
            shape = tuple(len(points) for _, points in grid)
            if len(values) != math.prod(shape):
                raise ModelingError("service_regime_grid_requires_every_corner")
            strides = tuple(math.prod(shape[index + 1:]) for index in range(len(shape)))
            return cls(kind, outside=_outside(raw), grid=grid, values_us=values, strides=strides)
        if kind != "piecewise_linear":
            raise ModelingError("service_regime_response_kind_unknown:" + kind)
        if set(raw) - {"kind", "coordinate", "knots", "outside"}:
            raise ModelingError("service_regime_response_unknown_fields")
        coordinate = raw.get("coordinate")
        if coordinate not in coordinate_names:
            raise ModelingError("service_regime_curve_requires_declared_coordinate")
        knots = _pairs(raw.get("knots"))
        xs = _ordered_knots(pair[0] for pair in knots)
        return cls(kind, coordinate=coordinate, knots=knots, outside=_outside(raw), coordinates=xs)

    def evaluate(self, values, *, rate_for=None, resource_for=None):
        """Return the numerical response, not an Engine completion prediction."""
        if not isinstance(values, Mapping):
            raise ModelingError("service_regime_response_requires_coordinate_values")
        if self.kind == "measured_dual_load":
            m = _value(values, "m")
            n = _value(values, "n")
            k = _value(values, "k")
            explicit_owners = None
            if "owners" in values:
                owners_value = values.get("owners")
                if (isinstance(owners_value, bool)
                        or not isinstance(owners_value, (int, float))
                        or not math.isfinite(float(owners_value))
                        or float(owners_value) != int(owners_value)
                        or int(owners_value) <= 0):
                    raise ModelingError("measured_dual_load_requires_positive_active_owners")
                explicit_owners = int(owners_value)
            issue = self.issue or {}
            if float(n) != float(issue.get("n")) or float(k) != float(issue.get("k")):
                return ResponseValue(None, "measured_load_geometry_mismatch")
            alignment = int(issue["m_alignment"])
            aligned_m = math.ceil(float(m) / alignment) * alignment
            issued = 2.0 * aligned_m * float(n) * float(k)
            if not math.isfinite(issued) or issued <= 0:
                raise ModelingError("measured_dual_load_issued_work_invalid")
            under = next((item for item in self.loads
                          if item["id"] == "underfilled"), None)
            saturated = next((item for item in self.loads
                              if item["id"] == "saturated"), None)
            load = None
            if (under is not None
                    and int(round(issued)) == int(under["measured_issued_flops"])):
                load = under
            else:
                full_work = (int(saturated["measured_issued_flops"])
                             if saturated is not None else 0)
                issued_int = int(round(issued))
                if (full_work > 0 and issued_int >= full_work
                        and issued_int % full_work == 0):
                    load = saturated
            if load is None:
                return ResponseValue(None, "measured_load_unknown_issue_class")
            owners = int(load["active_owners"])
            if explicit_owners is not None and explicit_owners != owners:
                return ResponseValue(None, "measured_load_unknown_active_owners")
            duration = self.fixed_us + issued * 1e6 / float(load["rate_flops_per_s"])
            if not math.isfinite(duration):
                raise ModelingError("measured_dual_load_service_overflow")
            return ResponseValue(
                _duration(duration), "measured_" + str(load["id"]),
                issued_work=issued, rate_flops_per_s=float(load["rate_flops_per_s"]),
                active_owners=owners, stage="mac",
            )
        if self.kind == "roofline":
            if rate_for is None:
                raise ModelingError("service_regime_scaled_response_requires_physical_rates")
            service = {}
            for term in self.terms:
                resource = resource_for(term) if resource_for is not None else term["resource"]
                rate = positive(rate_for(term))
                duration = _value(values, term["coordinate"]) * 1e6 / (rate * term["utilization"])
                if not math.isfinite(duration):
                    raise ModelingError("service_regime_roofline_service_overflow")
                service[resource] = service.get(resource, 0) + duration
            # Distinct instruction modes and compute precisions can consume
            # the same physical pipe; their service adds before pipe overlap.
            duration = self.fixed_us + max(service.values())
            return ResponseValue(_duration(duration), "physical_supply")
        if self.kind == "affine":
            duration = self.fixed_us
            for term in self.terms:
                time = _value(values, term["coordinate"]) * term["us_per_unit"]
                if term.get("scale_with"):
                    if rate_for is None:
                        raise ModelingError("service_regime_scaled_response_requires_physical_rates")
                    rate = positive(rate_for(term["scale_with"]))
                    time *= term["scale_with"]["reference_rate"] / rate
                if not math.isfinite(time) or time < 0:
                    raise ModelingError("service_regime_affine_response_invalid_term")
                duration += time
                if not math.isfinite(duration):
                    raise ModelingError("service_regime_affine_response_overflow")
            return ResponseValue(_duration(duration), "affine")
        if self.kind == "multilinear":
            brackets, outside, anchor = [], False, True
            for name, points in self.grid:
                x = nonnegative(_value(values, name))
                right = bisect_left(points, x)
                if right < len(points) and points[right] == x:
                    brackets.append(((right, 1.0),))
                    continue
                anchor = False
                outside |= right == 0 or right == len(points)
                right = min(max(right, 1), len(points) - 1)
                weight = (x - points[right - 1]) / (points[right] - points[right - 1])
                if not math.isfinite(weight):
                    raise ModelingError("service_regime_grid_cell_weight_invalid")
                brackets.append(((right - 1, 1 - weight), (right, weight)))
            if outside and self.outside == "analytic":
                return ResponseValue(None, "analytic_outside_knots")
            if outside and self.outside == "error":
                raise ModelingError("service_regime_grid_coordinate_outside_knots")
            duration = math.fsum(
                self.values_us[sum(index * stride for (index, _), stride in zip(corner, self.strides))]
                * math.prod(weight for _, weight in corner) for corner in product(*brackets))
            if duration < 0 or not math.isfinite(duration):
                raise ModelingError("service_regime_curve_extrapolation_invalid_duration")
            return ResponseValue(duration, "extrapolated" if outside else "anchor" if anchor else "interpolated")
        x = nonnegative(_value(values, self.coordinate))
        right = bisect_left(self.coordinates, x)
        if right < len(self.knots) and self.coordinates[right] == x:
            return ResponseValue(self.knots[right][1], "anchor")
        outside = right == 0 or right == len(self.knots)
        if outside and self.outside == "analytic":
            return ResponseValue(None, "analytic_outside_knots")
        if outside and self.outside == "error":
            raise ModelingError("service_regime_curve_coordinate_outside_knots:" + str(x))
        right = min(max(right, 1), len(self.knots) - 1)
        x0, y0 = self.knots[right - 1]
        x1, y1 = self.knots[right]
        duration = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        if duration < 0 or not math.isfinite(duration):
            raise ModelingError("service_regime_curve_extrapolation_invalid_duration")
        return ResponseValue(duration, "extrapolated" if outside else "interpolated")


def _value(values, name):
    if name not in values:
        raise ModelingError("service_regime_response_missing_coordinate:" + name)
    return nonnegative(values[name])


def _pairs(values):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ModelingError("service_regime_curve_requires_coordinate_duration_pairs")
    pairs = []
    for pair in values:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ModelingError("service_regime_curve_requires_coordinate_duration_pairs")
        pairs.append((nonnegative(pair[0]), nonnegative(pair[1])))
    return tuple(pairs)


def _ordered_knots(values):
    if values is None or isinstance(values, (str, bytes)):
        raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates")
    try:
        points = tuple(nonnegative(value) for value in values)
    except TypeError as exc:
        raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates") from exc
    if len(points) < 2 or any(right <= left for left, right in zip(points, points[1:])):
        raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates")
    return points


def _outside(raw):
    outside = raw.get("outside", "analytic")
    if outside not in {"analytic", "linear", "error"}:
        raise ModelingError("service_regime_curve_outside_requires_analytic_linear_or_error")
    return outside


MIGRATION_PLAN_FIELDS = frozenset({
    "replacement_kind", "scope", "condition_context", "slice_conditions",
    "fixed", "work", "program", "context", "occurrences", "native_boundary"})
MIGRATION_WORK_FIELDS = frozenset({
    "term_index", "service_id", "stage", "resource", "work_unit",
    "work_scale", "multiplicity", "phase", "native_boundary", "work_scope"})
MIGRATION_FIXED_FIELDS = frozenset({"service_id", "stage", "native_boundary"})
MIGRATION_REPLACEMENT_KINDS = frozenset({
    "independent_service", "joint_phase_service"})
MIGRATION_SCOPES = frozenset({"complete_rule", "slice"})
_WORK_SCOPES = frozenset({"aggregate", "per_owner"})
_TRAINING_FIELDS = ("training_sources", "training", "train_sources")
_VALIDATION_FIELDS = ("validation_sources", "validation", "heldout",
                      "heldout_sources", "diagnostic_sources", "diagnostic")
_OTHER_SOURCE_FIELDS = ("source", "sources", "source_path", "sha256",
                        "source_sha256", "profile_source", "observation_source",
                        "calibration_source", "train_source")


def _plan_text(value, code):
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _role_tokens(value):
    tokens = set()

    def walk(node):
        if isinstance(node, str):
            if node.strip():
                tokens.add(node.strip())
        elif isinstance(node, Mapping):
            for item in node.values():
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    return tokens


def provenance_roles(provenance):
    """Read T06 production source roles without merging training with validation."""
    if not isinstance(provenance, Mapping):
        provenance = {}
    fit = provenance.get("fit") if isinstance(provenance.get("fit"), Mapping) else {}
    roles = {}
    for role, fields in (("training", _TRAINING_FIELDS),
                         ("validation", _VALIDATION_FIELDS),
                         ("other", _OTHER_SOURCE_FIELDS)):
        tokens = set()
        for field in fields:
            if field in provenance:
                tokens |= _role_tokens(provenance[field])
            if field in fit:
                tokens |= _role_tokens(fit[field])
        roles[role] = sorted(tokens)
    roles["training_or_other"] = sorted(set(roles["training"]) | set(roles["other"]))
    return roles


def same_declared_value(left, right):
    """Compare declared condition values with numeric tolerance and structure."""
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (left.keys() == right.keys()
                and all(same_declared_value(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(same_declared_value(a, b) for a, b in zip(left, right)))
    return left == right


def _program_view(program):
    if program is None:
        return None, (), None
    stages = getattr(program, "stages", None)
    if stages is None and isinstance(program, (list, tuple)):
        stages = program
    if stages is None:
        raise ModelingError("migration_plan_program_requires_stages")
    stages = tuple(stages)
    names, view = set(), []
    for stage in stages:
        name = getattr(stage, "name", None)
        work = getattr(stage, "work", None)
        if not isinstance(name, str) or not name or name in names or not isinstance(work, Mapping):
            raise ModelingError("migration_plan_program_stage_invalid:" + str(name))
        names.add(name)
        view.append({
            "name": name, "work": dict(work),
            "work_units": dict(getattr(stage, "work_units", {}) or {}),
            "after": list(getattr(stage, "after", ()) or ()),
            "previous": list(getattr(stage, "previous", ()) or ()),
            "acquire": [getattr(item, "name", str(item))
                        for item in getattr(stage, "acquire", ()) or ()],
            "free": [getattr(item, "name", str(item))
                     for item in getattr(stage, "free", ()) or ()],
            "placement": {key: list(value) for key, value in dict(
                getattr(stage, "placement", {}) or {}).items()},
            "fragment": getattr(getattr(stage, "fragment", None), "name", None),
            "repeats": getattr(getattr(stage, "fragment", None), "repeats", None),
        })
    return program, stages, view


def _fragment_evidence(fragment):
    work = getattr(fragment, "work", None)
    items = tuple(work.issued()) if work is not None and hasattr(work, "issued") else ()
    return (tuple((item.resource, float(item.amount), item.unit, item.kind)
                  for item in items),
            int(getattr(fragment, "repeats", 1)))


def normalize_migration_plan(plan, decomposition):
    """Validate a migration plan against real program/stage/context references.

    ``program``/``context`` are read-only references to the actual selected
    execution objects. Old ``term_index`` remains as a diagnostic link to the
    legacy formula; ``stage`` is the actual occurrence being replaced.
    """
    if not isinstance(plan, Mapping):
        raise ModelingError("migration_plan_requires_mapping")
    unknown = set(plan) - MIGRATION_PLAN_FIELDS
    if unknown:
        raise ModelingError("migration_plan_unknown_fields:"
                            + ",".join(sorted(unknown)))
    kind = _plan_text(plan.get("replacement_kind"), "migration_plan_requires_replacement_kind")
    if kind not in MIGRATION_REPLACEMENT_KINDS:
        raise ModelingError("migration_plan_replacement_kind_unknown:" + kind)
    scope = _plan_text(plan.get("scope"), "migration_plan_requires_scope")
    if scope not in MIGRATION_SCOPES:
        raise ModelingError("migration_plan_scope_unknown:" + scope)
    context = plan.get("condition_context")
    if context is not None and not isinstance(context, Mapping):
        raise ModelingError("migration_plan_condition_context_requires_mapping")
    slice_conditions = plan.get("slice_conditions") or {}
    if not isinstance(slice_conditions, Mapping):
        raise ModelingError("migration_plan_slice_conditions_requires_mapping")
    if scope == "slice" and not slice_conditions:
        raise ModelingError("migration_plan_slice_requires_conditions")
    if scope == "complete_rule" and slice_conditions:
        raise ModelingError("migration_plan_complete_rule_does_not_accept_slice_conditions")
    program, stages, view = _program_view(plan.get("program"))
    stage_names = {stage["name"] for stage in view} if view else set()
    occurrences = plan.get("occurrences") or {}
    if not isinstance(occurrences, Mapping):
        raise ModelingError("migration_plan_occurrences_requires_mapping")
    for name, fragment in occurrences.items():
        if not isinstance(name, str) or not name:
            raise ModelingError("migration_plan_occurrence_stage_requires_name")
        if view and name not in stage_names:
            raise ModelingError("migration_plan_occurrence_stage_not_in_program:" + name)
        if not hasattr(fragment, "work") or not hasattr(fragment.work, "issued"):
            raise ModelingError("migration_plan_occurrence_requires_program_fragment")
    fragments = {}
    for stage in stages:
        builtin = getattr(stage, "fragment", None)
        external = occurrences.get(getattr(stage, "name", ""))
        if builtin is not None and external is not None:
            if _fragment_evidence(builtin) != _fragment_evidence(external):
                raise ModelingError(
                    "migration_plan_occurrence_conflicts_with_builtin_fragment:"
                    + getattr(stage, "name", ""))
        fragments[getattr(stage, "name", "")] = external or builtin
    actual_context = plan.get("context")
    if actual_context is not None and not any(
            hasattr(actual_context, name) for name in (
                "native_work", "work_units", "rates", "resource_names")):
        raise ModelingError("migration_plan_context_requires_recipe_context_view")
    fixed_raw = plan.get("fixed")
    if fixed_raw is None:
        fixed_items = []
    elif isinstance(fixed_raw, Mapping):
        fixed_items = [fixed_raw]
    elif isinstance(fixed_raw, (list, tuple)):
        fixed_items = list(fixed_raw)
    else:
        raise ModelingError("migration_plan_fixed_requires_mapping_or_sequence")
    fixed = []
    for item in fixed_items:
        if not isinstance(item, Mapping) or set(item) - MIGRATION_FIXED_FIELDS:
            raise ModelingError("migration_plan_fixed_item_unknown_fields")
        sid = _plan_text(item.get("service_id"), "migration_plan_fixed_requires_service_id")
        stage = item.get("stage")
        if stage is not None:
            stage = _plan_text(stage, "migration_plan_fixed_stage_requires_name")
            if view and stage not in stage_names:
                raise ModelingError("migration_plan_fixed_stage_not_in_program:" + stage)
        native = item.get("native_boundary")
        if native is not None:
            native = _plan_text(native, "migration_plan_fixed_native_boundary_requires_text")
        fixed.append({"service_id": sid, "stage": stage, "native_boundary": native})
    declared = {term["index"]: term for term in decomposition.get("terms", ())}
    work = plan.get("work") or ()
    if not isinstance(work, (list, tuple)):
        raise ModelingError("migration_plan_work_requires_sequence")
    normalized = []
    for item in work:
        if not isinstance(item, Mapping):
            raise ModelingError("migration_plan_work_item_requires_mapping")
        unknown = set(item) - MIGRATION_WORK_FIELDS
        if unknown:
            raise ModelingError("migration_plan_work_item_unknown_fields:"
                                + ",".join(sorted(unknown)))
        stage = item.get("stage")
        if stage is not None:
            stage = _plan_text(stage, "migration_plan_work_stage_requires_name")
            if view and stage not in stage_names:
                raise ModelingError("migration_plan_work_stage_not_in_program:" + stage)
        index = item.get("term_index")
        if index is not None:
            if isinstance(index, bool) or not isinstance(index, int) or index not in declared:
                raise ModelingError("migration_plan_work_item_requires_declared_term_index")
            if declared[index].get("kind") == "zero_slope":
                raise ModelingError("migration_plan_cannot_bind_zero_slope_term:" + str(index))
        elif stage is None:
            raise ModelingError("migration_plan_work_item_requires_term_or_stage")
        resource = item.get("resource")
        if resource is None and index is not None:
            resource = declared[index].get("resource")
        if resource is not None:
            resource = _plan_text(resource, "migration_plan_work_resource_requires_text")
        work_unit = item.get("work_unit")
        if work_unit is not None:
            work_unit = _plan_text(work_unit, "migration_plan_work_unit_requires_text")
        phase = _plan_text(item.get("phase"), "migration_plan_work_item_requires_phase")
        multiplicity = item.get("multiplicity", 1)
        if (isinstance(multiplicity, bool) or not isinstance(multiplicity, int)
                or multiplicity < 1):
            raise ModelingError("migration_plan_work_item_requires_positive_multiplicity")
        scale = item.get("work_scale")
        if scale is not None:
            scale = positive(scale)
        elif stage is None:
            raise ModelingError("migration_plan_term_binding_requires_work_scale")
        else:
            scale = 1.0
        work_scope = item.get("work_scope")
        if work_scope is not None:
            work_scope = _plan_text(work_scope, "migration_plan_work_scope_unknown")
            if work_scope not in _WORK_SCOPES:
                raise ModelingError("migration_plan_work_scope_unknown:" + work_scope)
        boundary = item.get("native_boundary")
        if boundary is not None:
            boundary = _plan_text(boundary, "migration_plan_native_boundary_requires_text")
        normalized.append({
            "term_index": index, "service_id": _plan_text(
                item.get("service_id"), "migration_plan_work_item_requires_service_id"),
            "stage": stage, "resource": resource, "work_unit": work_unit,
            "work_scale": scale, "multiplicity": multiplicity, "phase": phase,
            "native_boundary": boundary, "work_scope": work_scope})
    plan_native = plan.get("native_boundary")
    if plan_native is not None:
        plan_native = _plan_text(plan_native, "migration_plan_native_boundary_requires_text")
    return {
        "replacement_kind": kind, "scope": scope,
        "condition_context": dict(context) if context is not None else None,
        "slice_conditions": dict(slice_conditions), "fixed": fixed, "work": normalized,
        "program": program, "stages": stages, "program_view": view,
        "occurrences": dict(occurrences), "fragments": fragments,
        "context": actual_context, "native_boundary": plan_native,
    }
