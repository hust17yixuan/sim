"""A5 communication engine binding, recipe construction and streaming helpers.

[Role] Bind calibrated A5 communication curves to engine service phases, build
  explicit link/HBM/vector stage programs, and stream very large JSON captures
  without materializing the whole document.
[Invariants] Physical link work is never deleted, shrunk or clamped.  Captured
  per-rank payload knots are converted to wire coordinates only through an
  explicit per-collective factor, and only A5 hardware identities may bind.
  A bound measurement keeps its measured phase (AI_CPU child, phase, complete
  call, directed p2p) separate from the consumer role; a child is never
  relabelled as a whole collective or a link transit price.  A bound
  measurement is a completion floor, never invented vector demand.
  TimeoutError always propagates unchanged.
[Timeouts & Locks] No timers, locks or subprocesses.  Files above the ingest
  threshold are consumed in 64 KiB chunks; no whole-file json.load is used and
  a pure-stdlib event stream replaces ijson when it is unavailable.
[Side-Effect Policy] Pure computation and read-only streaming.  Never launches a
  device, writes calibration or mutates the primitive store.
"""
from __future__ import annotations

import codecs
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import math
import re
from typing import Mapping

from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.services import (
    ResourceService, ServiceCurve, service_identity,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from .a5_comm_golden import A5_HARDWARE_KEY
from .a5_comm_ingest import (
    _collective, _dtype, _explicit_pair, _explicit_scope, _mode,
    _observed_mode, _protocol_key, _require_a5, _selectors, _split,
    _validate_row,
)

__all__ = [
    "_A5CommPlan", "_stream_json", "_wire_factor", "bind_service_curve",
    "build_a5_comm_recipe", "build_normalized_service_recipe",
    "build_normalized_service_stage", "normalized_service_duration_ms",
    "NormalizedServiceStage", "service_curve_from_normalized",
    "to_service_regime",
]

_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_BYTE_RESOURCES = frozenset({"link", "hbm", "mte1", "mte2", "mte3", "fixpipe"})
_TIMING_KEYS = frozenset({
    "per_op_ms", "median_ms", "duration_ms", "ms", "per_op_ms_summary",
    "raw_iterations_us"})
_SUPPORTED_COLLECTIVES = frozenset({
    "all_reduce", "reduce_scatter", "all_gather", "all_to_all", "all_to_allv",
    "p2p_send_recv"})
_COLLECTIVE_DIRECTIONS = {
    "all_gather": "read", "all_reduce": "send", "reduce_scatter": "send",
    "all_to_all": "send", "all_to_allv": "send", "p2p_send_recv": "directed"}
_SAMPLE_KEYS = frozenset({
    "split", "run_id", "repeat_index", "status", "state", "canary_ok",
    "canary_passed", "canary", "correctness", "hardware_key", "sku",
    "arch_family", "hardware", "single", "boundary", "observed_expansion_mode",
    "observed_execution", "requested_expansion_mode", "profiler_evidence",
    "stack", "cann", "algorithm", "protocol", "machine", "stack_binding",
    "payload_bytes", "size_bytes", "work", "dtype",
    "measurement", "transport_window", "direction_observed",
    "algorithm_identified"})
_STREAM_SKIP = frozenset({
    "per_op_ms_by_trial", "per_rank_median_ms", "clock_samples",
    "clock_summary", "metadata_before", "metadata_after", "rank_map",
    "devices", "env"})


def _wire_factor(collective, world_size, curve=None):
    """Captured payload per rank -> the declared collective's physical work.

    This is a coordinate conversion for the ring/tree recipe, not a hop price.
    A whole-collective completion envelope is never converted into a per-hop
    link rate by this factor; its provenance records
    ``link_transit_identified=False``.
    """
    rank_count = max(1, int(world_size or 1))
    name = _collective(collective)
    if rank_count <= 1:
        return 1.0
    if name in {"all_gather", "reduce_scatter"}:
        # The worker's payload_bytes is per-rank ag_in / rs_out while the model
        # accounts the complete ring traffic, (r - 1) * local_bytes.
        return float(rank_count - 1)
    if name == "all_reduce":
        return 2.0 * (rank_count - 1) / rank_count
    if name == "all_to_all":
        return float(rank_count - 1) / rank_count
    if name == "all_to_allv":
        provenance = getattr(curve, "provenance", None) or {}
        identity = getattr(curve, "identity", None) or {}
        if (dict(provenance).get("derived_from") == "all_to_all"
                or dict(identity).get("derived_from") == "all_to_all"):
            return float(rank_count - 1) / rank_count
        # A measured mc2-quant comm-lane row already recorded link_bytes.
        return 1.0
    return 1.0


@dataclass(frozen=True)
class _A5CommPlan(OperatorPlan):
    """A physical communication program with a measured completion floor.

    ``metadata`` remains the ordinary engine selector record.  The measured
    phase and consumer role are typed attributes so a consumer can require a
    complete collective call without parsing free text or a filename.
    """

    minimum_ms: float = 0.0
    measurement_boundary: Mapping = field(default_factory=dict)
    consumer_role: str = ""
    source_service_semantics: str = ""
    complete_collective_call_response: bool = False
    complete_directed_call_response: bool = False
    direction_observed: bool = False
    direction_basis: str = ""
    algorithm_identified: bool = False
    link_transit_identified: bool = False
    hop_scaling: str = "not_derived"
    diagnostic_only: bool = False
    whole_call_floor: bool = False
    production_eligible: bool = False

    def lower(self, context):
        if context.service_boundary != "device_kernel":
            raise ModelingError("service_curve_boundary_mismatch")
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",),
                  program=self.program(replace(context, setup_ms=0, minimum_ms=0))),
        ), minimum_ms=max(context.minimum_ms, self.minimum_ms))


def _fit_bundle(ingested=None):
    from .a5_comm_calibration import fit_a5_service_curves
    return fit_a5_service_curves(ingested)


def _resolve_bundle(bundle_or_case, case_or_op, op):
    if op is None:
        bundle, case, operation = _fit_bundle(), bundle_or_case, case_or_op
    else:
        bundle, case, operation = bundle_or_case, case_or_op, op
    if not hasattr(bundle, "lookup_curve"):
        bundle = _fit_bundle(bundle)
    return bundle, case, operation


def _validate_hardware(source, *, identity=False):
    source = dict(source or {})
    if identity:
        keys = (source.get("key"), source.get("sku"))
        architecture = source.get("architecture")
    else:
        keys = (source.get("hardware_key"), source.get("sku"))
        architecture = source.get("arch_family")
    for key in keys:
        if key or architecture:
            _require_a5(key or A5_HARDWARE_KEY, architecture)


def _require_case_a5(parsed):
    machine = parsed.machine
    _require_a5(machine.hardware_key or machine.sku, machine.arch_family)
    if machine.hardware_key:
        _require_a5(machine.hardware_key, machine.arch_family)
    if machine.sku:
        _require_a5(machine.sku, machine.arch_family)


def _world_size(case, op):
    metadata = dict(op.metadata)
    for key in ("world_size", "group_size", "ranks", "group_ranks"):
        if metadata.get(key):
            return int(metadata[key])
    axis = metadata.get("group_axis")
    parallel = case.runtime.parallel
    if axis and hasattr(parallel, axis):
        return int(getattr(parallel, axis))
    return int(parallel.tp * parallel.cp or len(case.machine.dies) or 1)


def _placement_for(case, op, collective, world_size, pair):
    metadata = dict(op.metadata)
    if "placement_scope" in metadata and metadata["placement_scope"] is not None:
        explicit = _explicit_scope(metadata["placement_scope"])
        if explicit is None:
            raise ModelingError(
                f"a5_placement_scope_unknown:{metadata['placement_scope']}")
        return explicit
    if pair == "0->24":
        return "cross_node_urma_far"
    if pair == "0->1":
        return "intra_node_ub"
    if pair and int(world_size) > 8:
        return "cross_node_urma"
    if int(world_size) > 8:
        return "cross_node_urma"
    if case is not None:
        for link in case.machine.links:
            if link.members and max(link.members) >= 8:
                return "cross_node_urma"
    return "intra_node_ub"


def _resource_for(op, metadata, collective):
    explicit = metadata.get("a5_service_resource") or metadata.get("service_resource")
    if explicit:
        return str(explicit)
    if op.link_bytes or collective in _SUPPORTED_COLLECTIVES:
        return "link"
    return "hbm" if op.hbm_bytes else "link"


def _measurement_boundary(curve, collective):
    """Preserve the measured phase; never relabel it as another boundary."""
    declared = curve.provenance.get("measurement_boundary")
    descriptor = dict(declared) if isinstance(declared, Mapping) else {}
    semantics = curve.provenance.get("service_semantics")
    if not descriptor:
        if semantics == "aicpu_child_collective_phase" or curve.resource == "aicpu_child":
            phase, complete = "aicpu_child", False
        elif curve.boundary == "resource_phase" or curve.resource == "collective_phase":
            phase, complete = "collective_phase", False
        elif collective == "p2p_send_recv":
            phase, complete = "directed_p2p_call", False
        else:
            phase, complete = "collective_call", True
        descriptor = {
            "kind": "native_target_call" if phase == "aicpu_child" else "device_call",
            "phase": phase, "resource": curve.resource,
            "work_coordinate": "per_rank_input_bytes" if phase == "aicpu_child"
            else "captured_per_rank_payload",
            "duration_unit": "ms",
            "source_kind": ("native_ai_cpu_child" if phase == "aicpu_child"
                            else "campaign_device_call"),
            "service_semantics": semantics or (
                "aicpu_child_collective_phase" if phase == "aicpu_child" else
                "directed_p2p_call" if phase == "directed_p2p_call" else
                "collective_phase_service" if phase == "collective_phase" else
                "whole_collective_call_envelope"),
            "complete_collective_call": complete,
            "complete_directed_call": phase == "directed_p2p_call",
            "link_transit_identified": False, "hop_scaling": "not_derived",
            "algorithm_identified": False,
            "direction_observed": phase == "directed_p2p_call",
        }
    descriptor.setdefault("complete_collective_call", False)
    descriptor.setdefault("complete_directed_call", False)
    descriptor.setdefault("link_transit_identified", False)
    descriptor.setdefault("hop_scaling", "not_derived")
    descriptor.setdefault("direction_observed", False)
    descriptor.setdefault("algorithm_identified", False)
    descriptor.setdefault("work_coordinate", "captured_per_rank_payload")
    descriptor.setdefault("phase", "unknown")
    descriptor.setdefault("service_semantics", semantics)
    return descriptor


def _consumer_role(descriptor):
    phase = descriptor.get("phase")
    if phase == "aicpu_child":
        return "aicpu_child_completion_floor"
    if phase == "collective_phase":
        return "resource_phase_envelope"
    if phase == "directed_p2p_call":
        return "directed_p2p_call_envelope"
    if phase == "collective_call":
        return "collective_call_completion_envelope"
    return "measured_call_envelope"


def _require_complete_call(descriptor, required):
    if required and not descriptor.get("complete_collective_call"):
        raise ModelingError(
            "a5_comm_measurement_boundary_cannot_serve_complete_collective_call:"
            + str(descriptor.get("phase") or "unknown"))


def _bind_to_bundle(bundle, case, op, scope="none", *, device=None,
                    resource=None, boundary=None, active_units=None,
                    require_complete_call=False, production=False,
                    **lookup_kwargs):
    if production:
        raise ModelingError("measured_a5_call_binding_is_diagnostic_only")
    parsed = (case if hasattr(case, "machine") and not isinstance(case, Mapping)
              else parse_case(case))
    _require_case_a5(parsed)
    metadata = dict(op.metadata)
    selectors = dict(lookup_kwargs)
    explicit = {key: selectors.get(key) for key in (
        "world_size", "expansion_mode", "placement_scope", "pair",
        "collective", "dtype", "stack", "cann", "algorithm",
        "protocol") if selectors.get(key) is not None}
    hardware_key, sku = selectors.pop("hardware_key", None), selectors.pop("sku", None)
    arch_family = selectors.pop("arch_family", None) or parsed.machine.arch_family
    _validate_hardware({"hardware_key": hardware_key, "sku": sku, "arch_family": arch_family})
    hardware_key = hardware_key or sku or parsed.machine.hardware_key or parsed.machine.sku
    placement = selectors.pop("placement_scope", None)
    world = selectors.pop("world_size", None) or _world_size(parsed, op)
    mode = selectors.pop("expansion_mode", None) or metadata.get(
        "expansion_mode", metadata.get("expansion_regime", "unspecified"))
    dtype = selectors.pop("dtype", None) or _dtype(op.dtype or parsed.model.compute_dtype)
    pair = selectors.pop("pair", None) or _explicit_pair(metadata.get("p2p_pair"))
    collective = _collective(selectors.pop("collective", None) or metadata.get("collective")
                             or metadata.get("a5_collective") or op.family)
    if collective not in _SUPPORTED_COLLECTIVES:
        collective = _collective(metadata.get("kernel") or op.recipe)
    if collective not in _SUPPORTED_COLLECTIVES:
        raise ModelingError(f"a5_curve_unavailable:{collective}")
    placement = placement or _placement_for(parsed, op, collective, world, pair)
    boundary = (boundary or selectors.pop("boundary", None)
                or ("resource_phase" if metadata.get("device_kernel_group") else "device_kernel"))
    for key in ("stack", "cann", "algorithm", "protocol"):
        if selectors.get(key) is None and metadata.get(key) is not None:
            selectors[key] = metadata[key]
    curve = bundle.lookup_curve(
        collective, world, placement_scope=placement, dtype=dtype,
        expansion_mode=mode, pair=pair, hardware_key=hardware_key,
        arch_family=arch_family, boundary=boundary, **selectors)
    if device is None:
        device = parsed.machine.dies[op.stage * parsed.runtime.parallel.tp * parsed.runtime.parallel.pcp]
    bound_metadata = dict(metadata)
    if "world_size" in explicit:
        bound_metadata["world_size"] = int(world)
        for alias in ("group_size", "ranks", "group_ranks", "steps"):
            bound_metadata.pop(alias, None)
    if "expansion_mode" in explicit:
        bound_metadata["expansion_mode"] = mode
        if "expansion_regime" in bound_metadata:
            bound_metadata["expansion_regime"] = mode
    for k, v in (("placement_scope", placement), ("pair", pair), ("collective", collective)):
        if k in explicit:
            bound_metadata["p2p_pair" if k == "pair" else k] = v
    if "collective" in explicit and "a5_collective" in bound_metadata:
        bound_metadata["a5_collective"] = collective
    for key in ("algorithm", "protocol", "stack", "cann"):
        if key in explicit:
            bound_metadata[key] = selectors[key]
    bound_op = op
    if bound_metadata != metadata:
        bound_op = replace(op, metadata=tuple(sorted(bound_metadata.items())))
    if "dtype" in explicit and dtype != bound_op.dtype:
        bound_op = replace(bound_op, dtype=dtype)
    identity = service_identity(parsed, bound_op, scope, device=device)
    resource = resource or _resource_for(op, metadata, collective)
    if resource not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{resource}")
    unit = "bytes"
    units = active_units or device.units(resource)
    measurement = _measurement_boundary(curve, collective)
    _require_complete_call(measurement, require_complete_call)
    wire_factor = _wire_factor(collective, world, curve) if resource == "link" else 1.0
    knots = curve.knots
    source_semantics = (curve.provenance.get("service_semantics")
                        or measurement.get("service_semantics")
                        or ("directed_p2p_call" if collective == "p2p_send_recv"
                            else "whole_collective_call_envelope"))
    provenance = {**curve.provenance, "bound_a5_curve": curve.model_id,
                  "boundary": boundary, "bound_stack": curve.identity.get("stack"),
                  "bound_metadata": dict(bound_metadata),
                  "measurement_boundary": measurement,
                  "measured_resource": measurement.get("resource"),
                  "consumption_resource": resource,
                  "source_service_semantics": source_semantics,
                  "service_semantics": source_semantics,
                  "consumer_role": _consumer_role(measurement),
                  "complete_collective_call_response": bool(
                      measurement.get("complete_collective_call")),
                  "complete_directed_call_response": bool(
                      measurement.get("complete_directed_call")),
                  "direction": curve.provenance.get(
                      "direction", _COLLECTIVE_DIRECTIONS.get(collective,
                                                              "unspecified")),
                  "direction_observed": bool(
                      measurement.get("direction_observed", False)),
                  "algorithm_identified": bool(
                      measurement.get("algorithm_identified", False)),
                  "shared_domain": curve.provenance.get("shared_domain"),
                  "transport_window": curve.provenance.get("transport_window"),
                  "work_scope": measurement.get(
                      "work_coordinate", "captured_per_rank_payload"),
                  # A measured phase/call is a completion envelope, not a
                  # per-hop transit price and not a licence to compose an
                  # invented hop count.
                  "link_transit_identified": bool(
                      measurement.get("link_transit_identified", False)),
                  "hop_scaling": measurement.get("hop_scaling", "not_derived")}
    if wire_factor != 1.0:
        knots = tuple((float(work) * wire_factor, float(duration))
                      for work, duration in curve.knots)
        provenance["raw_knots"] = curve.knots
        provenance["wire_factor"] = wire_factor
        provenance["wire_coordinate_semantics"] = (
            "captured per-rank payload converted to the declared collective's "
            "physical work coordinate; this is not a hop price")
    return ServiceCurve(resource, unit, identity, boundary, units,
                        curve.startup_ms, knots, provenance)


_NORMALIZED_PHYSICAL_RESOURCES = frozenset({
    "link", "hbm", "mte1", "mte2", "mte3", "fixpipe"})


@dataclass(frozen=True)
class NormalizedServiceStage:
    """One selected public stage bound to a T06 normalized service.

    ``kind`` is ``independent_resource`` or ``joint_phase``.  A joint phase
    keeps its resource set and never becomes one link cap.  A child/phase
    resource that has no physical byte mapping is rejected before this record.
    """

    kind: str
    service_id: str
    resources: tuple[str, ...]
    physical_resource: str | None
    phase: str
    work_unit: str
    boundary: str
    work: float
    work_scope: str
    active_units: int
    fixed_us: float
    duration_us: float
    owner_private_rate: float
    aggregate_rate: float
    stage_cap_work_per_ms: float
    aggregate_cap_work_per_ms: float
    work_total_aggregate: float
    conditions_bound: bool
    production_eligible: bool
    resource_service: ResourceService | None = None
    conditions: Mapping = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    def to_record(self):
        return {
            "kind": self.kind, "service_id": self.service_id,
            "resources": list(self.resources),
            "physical_resource": self.physical_resource,
            "phase": self.phase, "work_unit": self.work_unit,
            "boundary": self.boundary, "work": self.work,
            "work_scope": self.work_scope,
            "active_units": self.active_units,
            "fixed_us": self.fixed_us, "duration_us": self.duration_us,
            "owner_private_rate": self.owner_private_rate,
            "aggregate_rate": self.aggregate_rate,
            "stage_cap_work_per_ms": self.stage_cap_work_per_ms,
            "aggregate_cap_work_per_ms": self.aggregate_cap_work_per_ms,
            "work_total_aggregate": self.work_total_aggregate,
            "conditions_bound": self.conditions_bound,
            "production_eligible": self.production_eligible,
            "conditions": deepcopy(self.conditions),
            "provenance": deepcopy(self.provenance),
        }


def _normalized_service(service):
    from mindie_sim.observations.resource_services import (
        normalize_resource_service,
    )
    if hasattr(service, "duration_us") and hasattr(service, "effective_rate"):
        return service
    return normalize_resource_service(service)


def _normalized_work_scope(normalized, work_scope):
    # The default production adapter consumes an aggregate stage work amount.
    # A per-owner representation is only used when the real split work/owners
    # are supplied explicitly.  Stored rate_scope decides normalization, not
    # the stage work coordinate.
    scope = work_scope or "aggregate"
    if scope not in {"aggregate", "per_owner"}:
        raise ModelingError("normalized_service_work_scope_unknown:" + str(scope))
    return scope


def _normalized_rate_for_scope(normalized, work_scope):
    rate = (normalized.aggregate_rate if work_scope == "aggregate"
            else normalized.per_owner_rate)
    if rate <= 0 or not math.isfinite(rate):
        raise ModelingError("normalized_service_rate_not_finite")
    return rate


def _normalized_conditions_bound(normalized, conditions_context):
    if conditions_context is None:
        return False
    from mindie_sim.observations.calibration import service_conditions_match
    matched = service_conditions_match(normalized, conditions_context)
    if not matched["applies"]:
        failed = [check["path"] for check in matched["checks"]
                  if not check["matched"]]
        raise ModelingError(
            "normalized_service_conditions_mismatch:" + ",".join(failed))
    return True


def normalized_service_duration_ms(service, work, *, work_scope=None):
    """Apply T06's owner-aware formula exactly once."""
    normalized = _normalized_service(service)
    scope = _normalized_work_scope(normalized, work_scope)
    return normalized.duration_us(work, work_scope=scope) / 1000.0


def _normalized_service_curve(service, *, identity, resource, work_scope,
                              work_points, provenance, conditions_bound,
                              phase=None):
    normalized = _normalized_service(service)
    scope = _normalized_work_scope(normalized, work_scope)
    rate = _normalized_rate_for_scope(normalized, scope)
    points = tuple(float(point) for point in work_points)
    if (len(points) < 2 or any(point <= 0 or not math.isfinite(point)
                               for point in points)
            or any(right <= left for left, right in zip(points, points[1:]))):
        raise ValueError("normalized_service_requires_ordered_positive_work_points")
    knots = tuple(
        (point, (normalized.fixed_us + point * 1e6 / rate) / 1000.0)
        for point in points)
    details = {
        "normalized_service_id": normalized.model_id,
        "normalized_service_kind": normalized.kind,
        "normalized_service_scope": normalized.scope,
        "normalized_phase": normalized.phase,
        "normalized_work_unit": normalized.work_unit,
        "normalized_rate_scope": normalized.rate_scope,
        "service_work_scope": scope,
        "normalized_effective_rate": normalized.effective_rate,
        "owner_private_rate": normalized.per_owner_rate,
        "aggregate_rate": normalized.aggregate_rate,
        "aggregate_cap_work_per_ms": normalized.aggregate_rate / 1000.0,
        "normalized_fixed_us": normalized.fixed_us,
        "fixed_source": (normalized.identification or {}).get("fixed_source"),
        "fixed_components": (normalized.identification or {}).get(
            "fixed_components"),
        "fixed_charged": "once_in_ResourceService_setup",
        "normalized_conditions": dict(normalized.conditions),
        "normalized_provenance": dict(normalized.provenance),
        "conditions_bound": bool(conditions_bound),
        "whole_call_floor": False,
        "production_stage": bool(conditions_bound),
        "diagnostic_only": not bool(conditions_bound),
        "formula": "T_us = fixed_us + work * 1e6 / scoped_rate",
    }
    if phase is not None and phase != normalized.phase:
        raise ModelingError("normalized_service_phase_mismatch")
    if provenance:
        details.update(provenance)
    return ServiceCurve(
        resource, normalized.work_unit, identity, normalized.boundary,
        normalized.active_units, normalized.fixed_us / 1000.0, knots, details)


def service_curve_from_normalized(service, *, identity, resource=None,
                                  boundary=None, active_units=None,
                                  work_points=(1.0, 1e15), provenance=None,
                                  conditions_context=None, work_scope=None):
    """Adapt one T06 independent service to ServiceCurve.

    A joint phase or a child/phase resource without a real physical byte
    mapping is rejected; renaming it to ``link`` is not accepted.  Aggregate
    vs per-owner work is selected explicitly and evaluated by T06.
    """
    normalized = _normalized_service(service)
    if normalized.scope != "independent_resource":
        raise ModelingError("normalized_service_joint_phase_requires_joint_stage")
    if len(normalized.resources) != 1:
        raise ModelingError("normalized_service_independent_requires_one_resource")
    expected = normalized.resources[0]
    resource = resource or expected
    if resource != expected:
        raise ModelingError("normalized_service_resource_mismatch")
    if resource not in _NORMALIZED_PHYSICAL_RESOURCES:
        raise ModelingError("normalized_service_resource_not_physical")
    if boundary is not None and boundary != normalized.boundary:
        raise ModelingError("normalized_service_boundary_mismatch")
    units = normalized.active_units if active_units is None else int(active_units)
    if units != normalized.active_units:
        raise ModelingError("normalized_service_active_units_mismatch")
    scope = _normalized_work_scope(normalized, work_scope)
    conditions_bound = _normalized_conditions_bound(normalized, conditions_context)
    return _normalized_service_curve(
        normalized, identity=identity, resource=resource, work_scope=scope,
        work_points=work_points, provenance=provenance,
        conditions_bound=conditions_bound, phase=normalized.phase)


def build_normalized_service_stage(service, *, identity, stage,
                                   conditions_context=None,
                                   work_points=(1.0, 1e15), provenance=None,
                                   require_conditions=False):
    """Bind one selected public Stage's work/owner/role to a T06 service."""
    normalized = _normalized_service(service)
    stage = dict(stage or {})
    for key in ("work", "work_scope", "active_units", "phase", "work_unit",
                "boundary"):
        if key not in stage:
            raise ValueError("normalized_service_stage_requires_" + key)
    work = float(stage["work"])
    if not math.isfinite(work) or work <= 0:
        raise ValueError("normalized_service_stage_work_positive")
    scope = _normalized_work_scope(normalized, stage["work_scope"])
    active = int(stage["active_units"])
    if active != normalized.active_units:
        raise ModelingError("normalized_service_active_units_mismatch")
    if str(stage["phase"]) != normalized.phase:
        raise ModelingError("normalized_service_phase_mismatch")
    if str(stage["work_unit"]) != normalized.work_unit:
        raise ModelingError("normalized_service_work_unit_mismatch")
    if str(stage["boundary"]) != normalized.boundary:
        raise ModelingError("normalized_service_boundary_mismatch")
    conditions_bound = _normalized_conditions_bound(normalized, conditions_context)
    if require_conditions and not conditions_bound:
        raise ModelingError("normalized_service_conditions_required_for_production")
    duration_us = normalized.duration_us(work, work_scope=scope)
    fixed_us = normalized.fixed_us
    service_ms = duration_us - fixed_us
    if service_ms <= 0 or not math.isfinite(service_ms):
        raise ModelingError("normalized_service_stage_requires_positive_service")
    stage_cap = work * 1000.0 / service_ms
    aggregate_cap = normalized.aggregate_rate / 1000.0
    work_total = work if scope == "aggregate" else work * active
    common = dict(
        service_id=normalized.service_id, phase=normalized.phase,
        work_unit=normalized.work_unit, boundary=normalized.boundary,
        work=work, work_scope=scope, active_units=active,
        fixed_us=fixed_us, duration_us=duration_us,
        owner_private_rate=normalized.per_owner_rate,
        aggregate_rate=normalized.aggregate_rate,
        stage_cap_work_per_ms=stage_cap,
        aggregate_cap_work_per_ms=aggregate_cap,
        work_total_aggregate=work_total,
        conditions_bound=conditions_bound,
        conditions=dict(normalized.conditions),
        provenance={
            "normalized_service_id": normalized.model_id,
            "normalized_service_kind": normalized.kind,
            "normalized_service_scope": normalized.scope,
            "normalized_fixed_us": fixed_us,
            "fixed_source": (normalized.identification or {}).get("fixed_source"),
            "fixed_components": (normalized.identification or {}).get(
                "fixed_components"),
            "fixed_charged": "once_in_ResourceService_setup",
            "scoped_rate": _normalized_rate_for_scope(normalized, scope),
            "whole_call_floor": False,
            "selected_stage": stage,
            **(dict(provenance) if provenance else {}),
        },
    )
    if normalized.scope == "joint_phase":
        resources = tuple(str(item) for item in (stage.get("resources") or ()))
        if set(resources) != set(normalized.resources):
            raise ModelingError("normalized_service_joint_phase_resource_mismatch")
        if any(item not in _NORMALIZED_PHYSICAL_RESOURCES for item in resources):
            raise ModelingError(
                "normalized_service_joint_phase_resource_not_physical")
        return NormalizedServiceStage(
            kind="joint_phase", resources=resources, physical_resource=None,
            resource_service=None,
            production_eligible=bool(conditions_bound), **common)
    if normalized.scope != "independent_resource":
        raise ModelingError("normalized_service_scope_unknown")
    resource = stage.get("resource")
    if resource != normalized.resources[0]:
        raise ModelingError("normalized_service_resource_mismatch")
    if resource not in _NORMALIZED_PHYSICAL_RESOURCES:
        raise ModelingError("normalized_service_resource_not_physical")
    if "resources" in stage and tuple(stage["resources"]) != normalized.resources:
        raise ModelingError("normalized_service_resource_mismatch")
    curve = _normalized_service_curve(
        normalized, identity=identity, resource=resource, work_scope=scope,
        work_points=work_points, provenance=common["provenance"],
        conditions_bound=conditions_bound, phase=normalized.phase)
    return NormalizedServiceStage(
        kind="independent_resource", resources=normalized.resources,
        physical_resource=resource, resource_service=ResourceService(curve),
        production_eligible=bool(conditions_bound), **common)


def build_normalized_service_recipe(service, *, identity, resource=None,
                                    boundary=None, active_units=None,
                                    work_points=(1.0, 1e15), provenance=None,
                                    conditions_context=None, work_scope=None,
                                    require_conditions=True, stage=None):
    """Return ResourceService for one independent T06 service stage.

    When ``stage`` is supplied it is the same selected Stage as the public
    execution binding; the returned recipe is its ``resource_service``.
    """
    if stage is not None:
        bound = build_normalized_service_stage(
            service, identity=identity, stage=stage,
            conditions_context=conditions_context, work_points=work_points,
            provenance=provenance, require_conditions=require_conditions)
        if bound.resource_service is None:
            raise ModelingError("normalized_service_joint_phase_requires_joint_stage")
        return bound.resource_service
    if require_conditions and conditions_context is None:
        raise ModelingError("normalized_service_conditions_required_for_production")
    curve = service_curve_from_normalized(
        service, identity=identity, resource=resource, boundary=boundary,
        active_units=active_units, work_points=work_points,
        provenance=provenance, conditions_context=conditions_context,
        work_scope=work_scope)
    return ResourceService(curve)



def bind_service_curve(bundle_or_case, case_or_op, op=None, scope="none",
                       production=False, **kwargs):
    """Bind one A5 curve to a case/operator; the bundle argument is optional."""
    bundle, case, operation = _resolve_bundle(bundle_or_case, case_or_op, op)
    return _bind_to_bundle(bundle, case, operation, scope,
                           production=production, **kwargs)


def _regime_from_bundle(bundle, case, op, scope="none", *, regime_id=None,
                        coordinate="link", outside="analytic",
                        require_complete_call=False, production=False,
                        **lookup_kwargs):
    if production:
        raise ModelingError("to_service_regime_is_diagnostic_only")
    if coordinate not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{coordinate}")
    if (lookup_kwargs.get("boundary") not in (None, "device_kernel")
            or dict(op.metadata).get("device_kernel_group")):
        raise ModelingError("service_regime_requires_device_kernel_boundary")
    curve = _bind_to_bundle(bundle, case, op, scope,
                            require_complete_call=require_complete_call,
                            **lookup_kwargs)
    if curve.boundary != "device_kernel":
        raise ModelingError("service_regime_requires_device_kernel_boundary")
    identity = curve.identity
    match = {"hardware": identity["hardware"]["key"], "family": op.family,
             **{key: identity[key] for key in ("runtime", "revision", "format",
                                               "dtype", "layout", "replay_scope",
                                               "implementation")}}
    if "transport" in identity:
        match["transport"] = deepcopy(identity["transport"])
    if op.recipe:
        match["recipe"] = op.recipe
    else:
        match["kernel"] = identity["kernel"]
    metadata = dict(curve.provenance.get("bound_metadata") or op.metadata)
    for key in ("collective", "a5_collective", "group_axis", "steps", "world_size",
                "group_size", "ranks", "group_ranks", "placement_scope",
                "p2p_pair", "expansion_mode", "expansion_regime",
                "algorithm", "protocol", "stack", "cann"):
        if key in metadata:
            match[f"metadata.{key}"] = metadata[key]
    raw_knots = curve.provenance.get("raw_knots") or curve.knots
    wire_factor = float(curve.provenance.get("wire_factor") or 1.0)
    knots = [[float(work), float(duration * 1000.0)]
             for work, duration in raw_knots]
    coord_spec = {"path": f"work.{coordinate}"}
    if coordinate == "link" and wire_factor != 1.0:
        coord_spec["scale"] = 1.0 / wire_factor
    return {
        "id": regime_id or (f"a5_comm:{op.family}:{op.recipe or identity['kernel']}:"
                            f"{curve.model_id[:12]}"),
        "match": match,
        "coordinates": {"bytes": coord_spec},
        "domain": {"bytes": {"min": knots[0][0], "max": knots[-1][0]}},
        "response": {"kind": "piecewise_linear", "coordinate": "bytes",
                     "knots": knots, "outside": outside},
        "provenance": {
            "boundary": ("measured A5 "
                         + str(curve.provenance.get("measurement_boundary", {}).get(
                             "phase", "unknown")) + " response"),
            # T20 C07: this rule is a source-bearing diagnostic projection of
            # one measured curve.  Ordinary production pricing uses physical
            # work + normalized services, not an injected old whole-call rule.
            "diagnostic_only": True,
            "source_bearing": True,
            "production_eligible": False,
            "model_id": curve.model_id,
            "stack": curve.provenance.get("bound_stack"),
            "measurement_boundary": curve.provenance.get("measurement_boundary"),
            "measured_resource": curve.provenance.get("measured_resource"),
            "consumption_resource": curve.provenance.get("consumption_resource"),
            "source_service_semantics": curve.provenance.get(
                "source_service_semantics"),
            "service_semantics": curve.provenance.get("service_semantics"),
            "consumer_role": curve.provenance.get("consumer_role"),
            "complete_collective_call_response": bool(curve.provenance.get(
                "complete_collective_call_response", False)),
            "complete_directed_call_response": bool(curve.provenance.get(
                "complete_directed_call_response", False)),
            "direction": curve.provenance.get("direction"),
            "direction_observed": bool(curve.provenance.get(
                "direction_observed", False)),
            "direction_basis": curve.provenance.get("direction_basis"),
            "algorithm_identified": bool(curve.provenance.get(
                "algorithm_identified", False)),
            "shared_domain": curve.provenance.get("shared_domain"),
            "transport_window": curve.provenance.get("transport_window"),
            "work_scope": curve.provenance.get("work_scope"),
            "link_transit_identified": bool(curve.provenance.get(
                "link_transit_identified", False)),
            "hop_scaling": curve.provenance.get("hop_scaling", "not_derived"),
            "coordinate_basis": ("captured_per_rank_payload_not_link_transit"
                                 if coordinate == "link" else
                                 f"captured_{coordinate}_work"),
        },
    }


def to_service_regime(bundle_or_case, case_or_op, op=None, scope="none",
                      require_complete_call=False, production=False, **kwargs):
    """Return a service_regimes rule bound to the calibrated A5 response."""
    bundle, case, operation = _resolve_bundle(bundle_or_case, case_or_op, op)
    return _regime_from_bundle(
        bundle, case, operation, scope,
        require_complete_call=require_complete_call,
        production=production, **kwargs)


def _duration_for(curve, lookup_bytes):
    estimate = curve.estimate(lookup_bytes)
    if estimate is not None:
        return estimate.duration_ms
    if lookup_bytes < curve.knots[0][0]:
        x0, t0 = curve.knots[0]
        return curve.startup_ms + (t0 - curve.startup_ms) * max(
            0.0, lookup_bytes) / x0
    x0, t0 = curve.knots[-2]
    x1, t1 = curve.knots[-1]
    slope = max((t1 - t0) / (x1 - x0), (t1 - curve.startup_ms) / x1)
    return t1 + slope * (lookup_bytes - x1)


def _recipe_from_bundle(bundle, collective=None, *, world_size=None,
                        expansion_mode="unspecified", active_aiv_blocks=16,
                        allow_unspecified_fallback=False,
                        require_complete_call=False, production=False,
                        **lookup_kwargs):
    if production:
        raise ModelingError("measured_call_recipe_is_diagnostic_only")
    mode = _mode(expansion_mode)
    if mode not in _EXPANSION_MODES and mode != "unspecified":
        raise ModelingError(f"a5_expansion_mode_unknown:{expansion_mode}")
    spec = dict(lookup_kwargs)
    coordinate = spec.pop("coordinate", "link")
    if coordinate not in _BYTE_RESOURCES:
        raise ModelingError(f"a5_comm_curve_requires_byte_resource:{coordinate}")
    if spec.pop("boundary", None) not in (None, "device_kernel"):
        raise ModelingError("service_curve_boundary_mismatch")
    placement = spec.pop("placement_scope", None)
    hardware_key, sku, arch_family = spec.pop("hardware_key", None), spec.pop("sku", None), spec.pop("arch_family", None)
    _validate_hardware({"hardware_key": hardware_key, "sku": sku, "arch_family": arch_family})
    hardware_key = hardware_key or sku

    def build(op, geometry):
        meta = dict(op.metadata)
        if meta.get("device_kernel_group"):
            raise ModelingError("service_curve_boundary_mismatch")
        _validate_hardware(meta)
        op_mode = mode
        if op_mode == "unspecified" and (meta.get("expansion_mode") or meta.get("expansion_regime")):
            op_mode = _mode(meta.get("expansion_mode") or meta.get("expansion_regime"))
            if op_mode not in _EXPANSION_MODES and op_mode != "unspecified":
                raise ModelingError(f"a5_expansion_mode_unknown:{op_mode}")
        name = _collective(collective or meta.get("collective") or meta.get("a5_collective") or op.family)
        world = world_size or meta.get("world_size") or meta.get("group_size") or meta.get("ranks") or 8
        dtype = _dtype(spec.get("dtype") or getattr(op, "dtype", None) or meta.get("dtype") or "bf16")
        pair = spec.get("pair") or _explicit_pair(meta.get("p2p_pair"))
        scope = placement or _placement_for(None, op, name, world, pair)
        software = {k: spec[k] if k in spec else meta.get(k) for k in ("stack", "cann", "algorithm", "protocol")}
        curve = bundle.lookup_curve(
            name, world, placement_scope=scope, dtype=dtype, expansion_mode=op_mode,
            allow_unspecified_fallback=allow_unspecified_fallback, pair=pair,
            hardware_key=hardware_key or A5_HARDWARE_KEY, arch_family=arch_family, boundary="device_kernel", **software,
            **{k: v for k, v in spec.items() if k not in {"dtype", "pair", "stack", "cann", "algorithm", "protocol"}})
        if curve.boundary != "device_kernel":
            raise ModelingError("service_curve_boundary_mismatch")
        measurement = _measurement_boundary(curve, name)
        _require_complete_call(measurement, require_complete_call)
        curve_mode = curve.identity.get("expansion_regime") or curve.identity.get("expansion_mode") or "unspecified"
        metadata = {
            "a5_collective": name, "world_size": int(world), "placement_scope": scope, "dtype": dtype,
            "requested_expansion_mode": op_mode, "curve_expansion_mode": curve_mode,
            "unspecified_fallback": bool(curve_mode != op_mode), "model_id": curve.model_id,
            **{k: software[k] if software[k] is not None else curve.identity.get(k)
               for k in ("stack", "cann", "algorithm", "protocol")}}
        wire_factor = _wire_factor(name, world, curve) if coordinate == "link" else 1.0
        raw_amount = float(getattr(op, coordinate + "_bytes", 0) or getattr(op, "link_bytes", 0)
                           or getattr(op, "hbm_bytes", 0) or 1.0)
        link_bytes = float(getattr(op, "link_bytes", 0)
                           or (getattr(op, coordinate + "_bytes", 0) if coordinate == "link" else 0) or 0.0)
        lookup_bytes = raw_amount / wire_factor if coordinate == "link" else raw_amount
        duration = float(_duration_for(curve, lookup_bytes) or 0.0)
        hbm, mte2, mte3 = (float(getattr(op, k, 0) or 0) for k in ("hbm_bytes", "mte2_bytes", "mte3_bytes"))
        elem_bytes = {"int8": 1.0, "fp8-e4m3": 1.0, "fp32": 4.0, "int32": 4.0}.get(dtype, 2.0)
        vector = ((float(getattr(op, "vector_ops", 0) or 0) or max(1.0, raw_amount / elem_bytes))
                  if op_mode == "aiv" else float(getattr(op, "vector_ops", 0) or 0))
        if op_mode != "aiv":
            hbm = max(1.0, hbm or raw_amount)
        resources = {"hbm": hbm, **{k: v for k, v in (("vector", vector), ("link", link_bytes), ("mte2", mte2), ("mte3", mte3)) if v > 0}}
        work = Work(resources=resources)

        def program(context):
            _validate_hardware(context.service_identity.get("hardware", {}), identity=True)
            link_work = {res: val for res, val in context.work.items()
                         if res not in context.resource_names.values()}
            stages = [Stage("link", link_work) if link_work else Stage("link", latency=duration)]
            for res_name, amount in (("hbm", hbm), ("mte2", mte2), ("mte3", mte3)):
                if amount > 0:
                    stages.append(Stage(res_name, context.bind(Work(resources={res_name: amount}))))
            if vector > 0:
                place = {}
                if op_mode == "aiv":
                    count = context.resource_units.get("vector", context.core_count)
                    units = tuple(range(min(max(1, int(active_aiv_blocks)), count)))
                    place = {context.resource_names["vector"]: units}
                stages.append(Stage("vector", context.bind(Work(resources={"vector": vector})),
                                    placement=place))
            return KernelProgram(tuple(stages))
        return _A5CommPlan(
            work, program, minimum_ms=duration, metadata=metadata,
            measurement_boundary=measurement,
            consumer_role=_consumer_role(measurement),
            source_service_semantics=(curve.provenance.get("service_semantics")
                                      or measurement.get("service_semantics") or ""),
            complete_collective_call_response=bool(
                measurement.get("complete_collective_call")),
            complete_directed_call_response=bool(
                measurement.get("complete_directed_call")),
            direction_observed=bool(measurement.get("direction_observed", False)),
            direction_basis=str(curve.provenance.get("direction_basis") or ""),
            algorithm_identified=bool(measurement.get("algorithm_identified", False)),
            link_transit_identified=bool(
                measurement.get("link_transit_identified", False)),
            hop_scaling=measurement.get("hop_scaling", "not_derived"),
            diagnostic_only=True, whole_call_floor=True,
            production_eligible=False)

    return PlannedRecipe(build)


def build_a5_comm_recipe(bundle_or_collective=None, collective=None,
                         allow_unspecified_fallback=False,
                         require_complete_call=False, production=False,
                         **kwargs):
    """Build a PlannedRecipe from the calibrated A5 curve; bundle is optional."""
    if hasattr(bundle_or_collective, "lookup_curve"):
        bundle, name = bundle_or_collective, collective
    else:
        bundle, name = _fit_bundle(), (collective or bundle_or_collective)
    return _recipe_from_bundle(
        bundle, name,
        allow_unspecified_fallback=allow_unspecified_fallback,
        require_complete_call=require_complete_call,
        production=production, **kwargs)


# ---------------------------------------------------------------------------
# Streaming JSON events.  Large captures are read in bounded 64 KiB chunks.
# ---------------------------------------------------------------------------

_JSON_TOKEN = re.compile(
    r'\s*(?:'
    r'(?P<lbrace>\{)|(?P<rbrace>\})|(?P<lbracket>\[)|(?P<rbracket>\])|'
    r'(?P<colon>:)|(?P<comma>,)|'
    r'(?P<string>"(?:[^"\\]|\\.)*")|'
    r'(?P<number>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|'
    r'(?P<literal>true|false|null)'
    r')', re.DOTALL)
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}


def _decode_json_string(text):
    if "\\" not in text:
        return text
    result, index = [], 0
    while index < len(text):
        character = text[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        escape = text[index + 1]
        index += 2
        if escape in _ESCAPES:
            result.append(_ESCAPES[escape])
        elif escape == "u":
            code = int(text[index:index + 4], 16)
            index += 4
            if (0xD800 <= code <= 0xDBFF and text[index:index + 2] == "\\u"):
                low = int(text[index + 2:index + 6], 16)
                index += 6
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
            result.append(chr(code))
        else:
            raise ValueError("invalid json string escape")
    return "".join(result)


def _iter_json_tokens(handle, chunk_size=65536):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer, position, eof = "", 0, False
    while True:
        match = _JSON_TOKEN.match(buffer, position)
        if match is None:
            if eof:
                if buffer[position:].strip():
                    raise ValueError("invalid trailing json stream data")
                return
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += decoder.decode(chunk)
            else:
                eof = True
                buffer += decoder.decode(b"", final=True)
            continue
        if match.lastgroup == "number" and not eof:
            end = match.end()
            # A number may continue in the next chunk; never commit a prefix
            # such as "1" while the stream still holds ".5".
            if end == len(buffer) or buffer[end] not in " \t\r\n,]}":
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += decoder.decode(chunk)
                else:
                    eof = True
                    buffer += decoder.decode(b"", final=True)
                continue
        position = match.end()
        kind = match.lastgroup
        text = match.group(kind)
        if kind == "string":
            yield ("string", _decode_json_string(text[1:-1]))
        elif kind == "number":
            yield ("number", float(text))
        elif kind == "literal":
            yield ("literal", {"true": True, "false": False, "null": None}[text])
        else:
            yield ("punct", text)
        if position > 1_048_576:
            buffer = buffer[position:]
            position = 0


def _parse_json_value(token, tokens):
    kind, value = token
    if kind == "punct" and value == "{":
        yield ("start_map", None)
        token = next(tokens)
        if token == ("punct", "}"):
            yield ("end_map", None)
            return
        while True:
            if token[0] != "string":
                raise ValueError("json object key must be a string")
            yield ("map_key", token[1])
            if next(tokens) != ("punct", ":"):
                raise ValueError("json object key requires a colon")
            yield from _parse_json_value(next(tokens), tokens)
            separator = next(tokens)
            if separator == ("punct", "}"):
                yield ("end_map", None)
                return
            if separator != ("punct", ","):
                raise ValueError("json object requires a comma or closing brace")
            token = next(tokens)
    elif kind == "punct" and value == "[":
        yield ("start_array", None)
        token = next(tokens)
        if token == ("punct", "]"):
            yield ("end_array", None)
            return
        while True:
            yield from _parse_json_value(token, tokens)
            separator = next(tokens)
            if separator == ("punct", "]"):
                yield ("end_array", None)
                return
            if separator != ("punct", ","):
                raise ValueError("json array requires a comma or closing bracket")
            token = next(tokens)
    elif kind == "string":
        yield ("string", value)
    elif kind == "number":
        yield ("number", value)
    elif kind == "literal":
        yield ("boolean", value) if isinstance(value, bool) else ("null", None)
    else:
        raise ValueError("json value expected")


def _iter_json_events(handle, chunk_size=65536):
    tokens = _iter_json_tokens(handle, chunk_size)
    yield from _parse_json_value(next(tokens), tokens)

def _skip_stream_value(events, item=None):
    event = (item or next(events))[0]
    if event not in {"start_map", "start_array"}:
        return
    depth = 1
    while depth:
        event = next(events)[0]
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1

def _sample_ms(sample):
    for key in ("median_ms", "per_op_ms", "duration_ms", "ms", "per_op_ms_summary"):
        val = sample.get(key)
        if isinstance(val, Mapping):
            val = val.get("median")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    raw = [(float(x) / 1000.0, 1) for x in (sample.get("raw_iterations_us") or ())
           if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return _weighted_median(raw) if raw else None

def _fold_timings(values, item, weight=1):
    if item is None:
        return
    values.append((float(item), int(weight)))
    if len(values) > 256:
        values.sort(key=lambda pair: pair[0])
        merged = []
        for val, wt in values:
            if merged and merged[-1][0] == val:
                merged[-1] = (val, merged[-1][1] + wt)
            else:
                merged.append((val, wt))
        if len(merged) > 4096:
            raise ValueError("a5_stream_timing_quantile_limit_exceeded")
        values[:] = merged

def _weighted_median(values):
    if not values:
        return None
    pairs = sorted(values, key=lambda pair: pair[0])
    total = sum(wt for _, wt in pairs)
    lo, hi, acc, lo_val = (total - 1) // 2, total // 2, 0, None
    for val, wt in pairs:
        nxt = acc + wt
        if lo_val is None and nxt > lo:
            lo_val = val
        if nxt > hi:
            return (lo_val + val) / 2.0
        acc = nxt
    return pairs[-1][0]

def _semantic_key(value):
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _bucket_add(buckets, sample, index):
    pay = sample.get("payload_bytes", sample.get("size_bytes", sample.get("work")))
    obs = sample.get("observed_expansion_mode", sample.get("observed_execution"))
    req, ev, sel = sample.get("requested_expansion_mode"), sample.get("profiler_evidence"), _selectors({}, sample)
    sp_ex = _split(sample.get("split"))
    key = ("repeat_index" in sample, "run_id" in sample, sample.get("run_id") or None,
           _dtype(sample.get("dtype")) if sample.get("dtype") else None,
           int(pay) if pay is not None else None, bool(sample["single"]) if sample.get("single") is not None else None,
           "boundary" in sample, sample.get("boundary"), "requested_expansion_mode" in sample,
           _mode(req) if req is not None else None, obs is not None, _observed_mode(obs) if obs is not None else None,
           "profiler_evidence" in sample, _protocol_key(ev), "stack" in sample, sel["stack"],
           sel["cann"], sel["algorithm"], "protocol" in sample, _protocol_key(sel["protocol"]),
           _semantic_key(sample.get("measurement")),
           _semantic_key(sample.get("transport_window")),
           sample.get("direction_observed"), sample.get("algorithm_identified"))
    entry = buckets.get(key)
    if entry is None:
        if len(buckets) >= 4096:
            raise ValueError("a5_stream_sample_identity_limit_exceeded")
        entry = {}
        buckets[key] = entry
    ms, r = _sample_ms(sample), int(sample.get("repeat_index", index) or 0)
    if ms is not None:
        for o_sp, (_, _, o_seen, o_runs) in entry.items():
            hit, p_ms = (True, o_seen[r]) if r in o_seen else next(
                ((True, v) for lo, hi, v in o_runs if v is not Ellipsis and lo <= r <= hi), (False, None))
            if hit:
                if p_ms is None or p_ms != ms or (sp_ex is not None and o_sp is not None and o_sp != sp_ex):
                    raise ValueError("conflicting_primitive_repeat_or_split")
                if o_sp == sp_ex:
                    return
    part = entry.get(sp_ex)
    if part is None:
        tmpl = {k: v for k, v in sample.items() if k in _SAMPLE_KEYS and k not in _TIMING_KEYS and k != "split"}
        if sp_ex is not None:
            tmpl["split"] = sp_ex
        part = [tmpl, [], {}, []]
        entry[sp_ex] = part
    _, vals, seen, runs = part
    if ms is not None:
        if not runs:
            runs.append([r, r, ms])
        elif runs[-1][2] is Ellipsis:
            runs[-1][0], runs[-1][1] = min(runs[-1][0], r), max(runs[-1][1], r)
        elif r == runs[-1][1] + 1:
            if ms == runs[-1][2] or runs[-1][2] is None:
                runs[-1][1] = r
            elif len(runs) < 256:
                runs.append([r, r, ms])
            else:
                runs[-1][1], runs[-1][2] = r, None
                while len(runs) > 1 and runs[-2][1] + 1 == runs[-1][0]:
                    runs[-2][1], runs[-2][2] = runs[-1][1], None
                    runs.pop()
        else:
            seen[r] = ms
            if len(seen) >= 4096:
                m = []
                for lo, hi, v in sorted(runs + [[k, k, w] for k, w in seen.items()]):
                    if m and m[-1][1] + 1 == lo:
                        m[-1][1], m[-1][2] = hi, v if m[-1][2] == v else None
                    else:
                        m.append([lo, hi, v])
                runs[:] = [[m[0][0], m[-1][1], Ellipsis]] if len(m) >= 4096 else m
                seen.clear()
    _fold_timings(vals, ms)


def _read_stream_samples(events):
    """Consume a samples array in bounded O(1) working memory."""
    item = next(events)
    raw_samples, buckets, timings, has_evidence, idx = [], {}, [], False, 0

    def _consume(sample):
        nonlocal raw_samples, has_evidence, idx
        cur = idx
        idx += 1
        if not isinstance(sample, Mapping):
            return
        _validate_row(sample, f"stream:sample[{cur}]")
        keys = set(sample)
        if not (keys & (_SAMPLE_KEYS | _TIMING_KEYS)):
            return
        if keys & _SAMPLE_KEYS:
            has_evidence = True
        _fold_timings(timings, _sample_ms(sample))
        clean = {k: sample[k] for k in keys & (_SAMPLE_KEYS | _TIMING_KEYS)}
        if raw_samples is not None:
            raw_samples.append((cur, clean))
            if len(raw_samples) > 256:
                for p_idx, prev in raw_samples:
                    _bucket_add(buckets, prev, p_idx)
                raw_samples = None
        else:
            _bucket_add(buckets, clean, cur)

    if item[0] != "start_array":
        value = _read_stream_value(events, item)
        for sample in (value if isinstance(value, list) else ()):
            _consume(sample)
    else:
        while True:
            entry = next(events)
            if entry[0] == "end_array":
                break
            _consume(_read_stream_value(events, entry))
    if not has_evidence:
        return [], _weighted_median(timings)
    if raw_samples is not None:
        return [s for _, s in raw_samples], None
    return [{**tmpl, **({"per_op_ms": _weighted_median(vals)} if vals else {}),
             "_repeats": (runs, seen), "_weights": list(vals)}
            for entry in buckets.values()
            for tmpl, vals, seen, runs in entry.values()], None


def _read_stream_value(events, item=None):
    item = item or next(events)
    if item[0] == "start_map":
        result, s_med, r_med = {}, None, None
        while True:
            entry = next(events)
            if entry[0] == "end_map":
                fb = s_med if s_med is not None else r_med
                cur = result.get("per_op_ms_summary")
                if fb is not None and not (isinstance(cur, Mapping) and isinstance(cur.get("median"), (int, float))):
                    result["per_op_ms_summary"] = {"median": fb}
                samples = result.get("samples")
                if isinstance(samples, list):
                    inherited = {key: result[key] for key in (
                        "measurement", "transport_window", "direction_observed",
                        "algorithm_identified") if key in result}
                    for sample in samples:
                        if isinstance(sample, dict):
                            for key, value in inherited.items():
                                sample.setdefault(key, deepcopy(value))
                return result
            key = entry[1]
            if key in _STREAM_SKIP:
                _skip_stream_value(events)
            elif key == "samples":
                filtered, s_med = _read_stream_samples(events)
                if filtered:
                    result["samples"] = filtered
            elif key == "raw_iterations_us":
                raw_entry = next(events)
                if raw_entry[0] != "start_array":
                    _skip_stream_value(events, raw_entry)
                else:
                    vals = []
                    while (elem := next(events))[0] != "end_array":
                        val = _read_stream_value(events, elem)
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            _fold_timings(vals, float(val) / 1000.0)
                    r_med = _weighted_median(vals)
            else:
                result[key] = _read_stream_value(events)
    if item[0] == "start_array":
        result = []
        while True:
            entry = next(events)
            if entry[0] == "end_array":
                return result
            if len(result) >= 65536:
                raise ValueError("a5_stream_array_limit_exceeded")
            result.append(_read_stream_value(events, entry))
    return item[1]


def _stream_json(path):
    """Stream one JSON document into the bounded ingest projection."""
    with open(path, "rb") as handle:
        try:
            import ijson
        except ImportError:
            events = _iter_json_events(handle)
        else:
            events = ijson.basic_parse(handle, use_float=True)
        return _read_stream_value(events)
