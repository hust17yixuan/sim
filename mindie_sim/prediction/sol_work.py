"""Read-only projection of selected-plan issued work onto physical constraints.

[Role]
  Walk the same Case/OpSpec/CostModel lowering the Engine executes and the
  existing Stage/KernelProgram records, then expose per-owner work envelopes
  (`PlanWork`) for `sol.py`; never schedule or mutate the selected plan.
[Invariants]
  * Work is every actual lowered Stage: nested programs, indexed tiles and all
    residue groups are walked; an opaque structure is a hole, never zero work.
  * Owner identities are Engine resources (`cube@r0:s0`), never replica ids.
  * `after` sequences KernelProgram and template tiles; `previous` is not an edge.
  * Serial intra-recipe phase service is a separate critical-path lower bound.
  * A missing precision/service mode or unknown/zero-rate resource is a hole.
  * Measured Stage latency and measured service-regime floors are reported
    separately, never added to physical SOL.
  * Nominal link peaks price physical SOL; a derated link without a nominal
    peak is unavailable; empirical derating carries explicit provenance.
  * A `fused_into` child is counted until the Engine actually elides it.
  * Stream order is `(replica, stage, stream)`; unknown dependencies are holes.
  * Engine capacities are declared, nominal and uncalibrated, never measured.
[Timeouts & Locks]
  O(selected ops + lowered stages + tiles), finite in-memory; no locks.
[Side-Effect Policy]
  No I/O, devices, environment reads or input mutation.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
from ..schema.case import Case
from ..feature_reports import Demand
from . import sol_invocation as _inv
from .sol_provenance import (
    affirmative_physical_provenance, is_measured_minimum, service_regime_floor,
)
_SKIP_PREFIXES = ("host@", "scheduler@", "serving@", "client@")
_native = "declared_uncalibrated_engine_resource_capacity"
_nominal = "declared_uncalibrated_nominal_link_peak"
_switch = "declared_uncalibrated_switch_plane_capacity"
_DERATED = "declared_empirically_derated_capacity"
@dataclass(frozen=True)
class Peak:
    """One declared uncalibrated physical capacity constraint."""
    resource_id: str
    unit: str
    peak_per_ms: float
    mode: str
    source: str
    total_units: int = 1
    owner_units: tuple[int, ...] = ()
    calibrated: bool = False
    empirical: bool = False
@dataclass(frozen=True)
class WorkTerm:
    """One selected program/phase with issued work on declared constraints."""
    id: str
    member_ops: tuple[str, ...]
    work: tuple[tuple[str, float, str, str], ...]
    predecessors: tuple[str, ...] = ()
    release_ms: float = 0.0
    dispatch_min_ms: float = 0.0
    parent_id: str | None = None
    owner: int = 0
    inclusive_min_ms: float = 0.0
    floor_basis: str = ""
    #: Necessary serial lower bound for a declared ``HBM -> Vector`` order.
    phase_precedence_ms: float = 0.0
    #: Empirical service kept out of the physical bound (reported separately).
    measured_latency_ms: float = 0.0
    #: Pure device/resource path with every invocation/program minimum zeroed.
    ideal_phase_precedence_ms: float = 0.0
    #: Every declared basis that contributed to this term (both startup and
    #: minimum for one nested invocation).
    floor_bases: tuple[str, ...] = ()
@dataclass(frozen=True)
class PlanWork:
    """Same-plan projection shared by both SOL policies."""
    plan_id: str
    terms: tuple[WorkTerm, ...]
    peaks: tuple[Peak, ...]
    coverage: str = "complete"
    holes: tuple[str, ...] = ()
    owner: int = 0
    uses_empirical_derating: bool = False
    def peak(self, resource_id, mode=None):
        exact = [item for item in self.peaks if item.resource_id == resource_id]
        matches = exact or [item for item in self.peaks
                            if _inv.same_family(item.resource_id, resource_id)]
        if mode is not None:
            matches = [item for item in matches if not item.mode or item.mode == mode]
        return matches[0] if matches else None
    def demands(self, term, dispatch="ideal_zero"):
        """Expand one term's work into contract Demand rows for its owner(s)."""
        rows, holes = [], []
        for resource_id, amount, unit, mode in term.work:
            if amount <= 0:
                continue
            peak = self.peak(resource_id, mode or None)
            if peak is None or peak.peak_per_ms <= 0:
                source = f":{peak.source}" if peak is not None and peak.source else ""
                holes.append(f"missing_or_zero_peak:{resource_id}:{mode or unit}{source}")
                continue
            if unit in {"peak_ms", "ms"}:
                if peak.unit and peak.unit != unit:
                    holes.append(f"unit_mismatch:{resource_id}:{unit}!={peak.unit}")
                    continue
            elif unit and peak.unit and unit != peak.unit:
                holes.append(f"unit_mismatch:{resource_id}:{unit}!={peak.unit}")
                continue
            rows.extend(_expand(peak.resource_id, amount, unit or peak.unit, peak, term))
        return tuple(rows), tuple(holes)
def _expand(resource_id, amount, unit, peak, term):
    """Uniform owner-pinning rule: W/|U| each at rate P/n."""
    if peak.owner_units and peak.total_units > len(peak.owner_units):
        count = peak.total_units
        per_work = amount / len(peak.owner_units)
        per_peak = peak.peak_per_ms / count
        return [Demand(f"{resource_id}:u{unit_id}", (unit_id,), peak.mode, per_work, unit,
                       per_peak, peak.source, per_work / per_peak)
                for unit_id in peak.owner_units]
    service = amount / peak.peak_per_ms
    units = peak.owner_units or tuple(range(max(1, peak.total_units)))
    return [Demand(resource_id, units, peak.mode, amount, unit, peak.peak_per_ms,
                   peak.source, service)]
def engine_peaks(case, *, empirical_derating=False):
    """Exact declared engine resource capacities for the selected Case.
    A link declaring ``peak=1e9`` and ``bw_utilization=0.1`` serializes an
    effective ``capacity=1e8``.  Pure physical SOL prices the nominal peak
    (``_nominal``) unless empirical derating is explicitly requested, in which
    case the row says ``declared_empirically_derated_capacity`` and the bound
    reports ``uses_empirical_derating=True``.  Every traversed resource
    (including a switch plane) keeps its own declared capacity; the enclosing
    link peak is never substituted for a narrower bottleneck.
    """
    from .execution import CostModel, Execution
    from .program import RecipeBook
    parsed = case if isinstance(case, Case) else Case.from_mapping(case)
    execution = Execution(parsed, CostModel(parsed),
                          recipes=RecipeBook.from_mapping(parsed.runtime.kernels))
    links = {str(link.name): link for link in parsed.machine.links}
    peaks = []
    for name, resource in execution.engine.resources.items():
        if name.startswith(_SKIP_PREFIXES) or float(resource.rate) <= 0:
            continue
        rate = float(resource.rate)
        source = _switch if ":switch_plane" in name else _native
        empirical = False
        if name.startswith("link:") and ":switch_plane" not in name:
            link = links.get(name.split(":", 2)[1])
            if link is not None:
                explicit_peak = link.peak_capacity_bytes_per_ms is not None
                nominal = float(link.effective_peak_capacity_bytes_per_ms)
                derated = nominal > rate + 1e-12 * max(1.0, nominal)
                shared = ":tx:" not in name and ":rx:" not in name
                if (not explicit_peak and not math.isclose(
                        float(link.bw_utilization), 1.0,
                        rel_tol=1e-12, abs_tol=1e-12)):
                    # A derated link with no declared nominal peak cannot
                    # price a physical capability: the empirical effective
                    # rate is not a nominal input.  Keep a zero-rate row with
                    # explicit provenance so ``PlanWork.demands`` reports
                    # ``missing_or_zero_peak`` instead of an available SOL.
                    peaks.append(Peak(name, resource.unit, 0.0, "",
                                      "missing_nominal_link_peak",
                                      int(getattr(resource, "units", 1) or 1),
                                      (), False, False))
                    continue
                if empirical_derating:
                    if derated and (shared or math.isclose(
                            rate, float(link.capacity_bytes_per_ms),
                            rel_tol=1e-12, abs_tol=1e-12)):
                        source, empirical = _DERATED, True
                elif derated and math.isclose(
                        rate, float(link.capacity_bytes_per_ms),
                        rel_tol=1e-12, abs_tol=1e-12):
                    rate, source = nominal, _nominal
        peaks.append(Peak(name, resource.unit, rate, "", source,
                          int(getattr(resource, "units", 1) or 1), (), False,
                          empirical))
    return tuple(peaks)
def _graphs(ops, owner):
    return ops if isinstance(ops, Mapping) else {owner: ops}
def _replay_scope(case, ops):
    """Engine parity: explicit per-op scope metadata, else ``none``.
    ``graph_analysis.graph_costs`` never derives replay scope from a Case
    capture policy.  Delegating to that exact rule keeps SOL and Engine from
    disagreeing; a conflicting declaration becomes a conservative hole.
    """
    from .operator_analysis import _replay_scope as engine_replay_scope
    try:
        value = str(engine_replay_scope(ops, None))
    except ValueError:
        return None
    return value if value in {"none", "full", "piecewise"} else None
def project_ops(case, ops, *, plan_id=None, owner=0, external_inputs=(),
                empirical_derating=False):
    """Project the exact lowered selected work for each replica/op occurrence."""
    from .execution import CostModel, Execution
    from .graph_analysis import bind_cohort_traffic
    from .program import RecipeBook
    from .body_defaults import consume_target_declaration
    parsed = case if isinstance(case, Case) else Case.from_mapping(case)
    parsed = consume_target_declaration(parsed, ops)
    graphs = {int(replica): tuple(sequence)
              for replica, sequence in _graphs(ops, owner).items()}
    try:
        graphs = bind_cohort_traffic(parsed, graphs)
    except (ValueError, KeyError):
        # The lowerer will diagnose a genuinely incomplete cohort per op.
        pass
    book = RecipeBook.from_mapping(parsed.runtime.kernels)
    execution = Execution(parsed, CostModel(parsed), recipes=book)
    peaks = {item.resource_id: item
             for item in engine_peaks(parsed, empirical_derating=empirical_derating)}
    catalog_engine = callable(getattr(execution.engine, "submit_catalog", None))
    external = set(external_inputs or ())
    terms, holes = [], []
    for replica in sorted(graphs):
        sequence = tuple(book.expand(graphs[replica]))
        ids = {}
        for index, op in enumerate(sequence):
            ids.setdefault(op.op_id, f"r{replica}:op:{op.op_id}")
        scope = _replay_scope(parsed, sequence)
        if scope is None:
            holes.append("replay_scope_conflict")
            scope = "none"
        last_stream = {}
        for index, op in enumerate(sequence):
            metadata = dict(op.metadata)
            fused_into = metadata.get("fused_into")
            positive = (float(op.flops or 0) + float(op.hbm_bytes or 0)
                        + float(op.vector_ops or 0) + float(op.mte2_bytes or 0)
                        + float(op.link_bytes or 0)) > 0
            if fused_into and positive:
                # ``Execution`` schedules every selected op; ``fused_into`` is
                # annotation, not an elision instruction.  Only an actually
                # skipped op may drop its issued work, so the executed child is
                # counted and the fusion claim stays conditional.
                if fused_into in ids:
                    holes.append(f"fused_work_unproven:{op.op_id}->{fused_into}")
                else:
                    holes.append(f"fused_parent_not_selected:{op.op_id}:{fused_into}")
            try:
                prepared, recipe = execution.prepare(op, replica)
                context = execution._context(prepared, replica, scope, recipe=recipe)
                program = execution.lower_program(context, recipe)
                measured_floor, physical_admitted = service_regime_floor(
                    execution, context)
            except TimeoutError:
                raise
            except Exception as exc:  # precision/mode/cohort domain rejection
                code = f"{type(exc).__name__}:{exc}".replace("\n", " ")[:160]
                holes.append(f"lowering_unavailable:{op.op_id}:{code}")
                continue
            try:
                work_map, program_holes, floors = _inv.program_work(
                    program, engine=execution.engine)
                phase_path, physical_holes = _inv.program_phase_path(
                    program, peaks, measured_floor, include_minimum=True,
                    catalog_engine=catalog_engine)
                ideal_path, ideal_holes = _inv.program_phase_path(
                    program, peaks, measured_floor, include_minimum=False,
                    catalog_engine=catalog_engine)
            except TimeoutError:
                raise
            except Exception as exc:  # an unwalkable structure is a hole
                holes.append(f"work_projection_failed:{op.op_id}:"
                             f"{type(exc).__name__}")
                continue
            holes.extend(f"{op.op_id}:{hole}" for hole in program_holes)
            holes.extend(f"{op.op_id}:{hole}" for hole in dict.fromkeys(
                (*physical_holes, *ideal_holes)))
            if metadata.get("resource_service") is not None:
                # Per-owner caps/fixed setup are applied after lowering.
                holes.append(f"resource_service_binding_unprojected:{op.op_id}")
            work = []
            for (resource, unit), amount in sorted(work_map.items()):
                if resource.startswith(_SKIP_PREFIXES):
                    continue
                peak = peaks.get(resource)
                if peak is None:
                    hole = next((item for item in execution.engine.resources
                                 if _inv.same_family(item, resource)), None)
                    peak = (Peak(hole, execution.engine.resources[hole].unit,
                                 float(execution.engine.resources[hole].rate), "",
                                 _native, int(getattr(
                                     execution.engine.resources[hole], "units", 1) or 1))
                            if hole is not None else None)
                if peak is None:
                    holes.append(f"resource_not_declared:{op.op_id}:{resource}")
                    continue
                work.append((peak.resource_id, float(amount), unit or peak.unit, peak.mode))
            native = dict(getattr(context, "native_work", {}) or {})
            if not work and any(float(value or 0) > 0 for value in native.values()):
                holes.append(f"unpriced_selected_work:{op.op_id}")
            predecessors, missing = [], []
            for dep in op.deps:
                if dep in external:
                    continue
                target = ids.get(dep)
                if target is None:
                    missing.append(dep)
                else:
                    predecessors.append(target)
            if missing:
                holes.extend(f"dependency_not_selected:{op.op_id}:{dep}"
                             for dep in missing)
            stream = (op.stage, op.stream or "main")
            if stream in last_stream:
                predecessors.append(last_stream[stream])
            timing = getattr(context, "invocation_timing", None)
            startup = float(getattr(timing, "startup_ms", 0.0) or 0.0) if timing else 0.0
            inclusive = float(getattr(timing, "minimum_ms", 0.0) or 0.0) if timing else 0.0
            admitted = timing is None or affirmative_physical_provenance(
                getattr(timing, "provenance", None))
            measured = (0.0 if physical_admitted else
                        max(float(getattr(context, "minimum_ms", 0.0) or 0.0), 0.0))
            if not admitted:
                # Invocation timing is an architectural floor only with
                # affirmative physical provenance; observation is empirical.
                measured = max(measured, startup, inclusive)
                startup = inclusive = 0.0
            declared_floors = [item for item in floors
                               if item[2] != "measured_stage_latency"]
            if measured_floor > 0:
                # Observation-derived response floors are empirical, never
                # declared minima, and are excluded from both policies.
                declared_floors = [
                    item for item in declared_floors
                    if not (item[2] == "declared_program_minimum"
                            and is_measured_minimum(item[1], measured_floor))]
                if is_measured_minimum(inclusive, measured_floor):
                    inclusive = 0.0
            measured = max(measured, measured_floor)
            measured = max(measured, max((item[1] for item in floors
                                          if item[2] == "measured_stage_latency"),
                                         default=0.0))
            # A nested floor keeps its basis for reporting but is scoped to
            # its own leaf: it cannot delay unrelated parallel work.
            scoped = [item for item in declared_floors
                      if len(item) > 3 and item[3]]
            aggregating = [item for item in declared_floors if item not in scoped]
            if aggregating:
                dispatch = max(item[0] for item in aggregating)
                floor_value = max(item[1] for item in aggregating)
                basis = max(aggregating,
                            key=lambda item: (item[0] + item[1]))[2]
                startup = max(startup, dispatch)
                inclusive = max(inclusive, floor_value)
            else:
                basis = ("declared_hardware_dispatch_minimum" if startup > 0 else
                         "declared_invocation_minimum" if inclusive > 0 else "")
            bases = tuple(dict.fromkeys(item[2] for item in declared_floors))
            term_id = ids[op.op_id]
            terms.append(WorkTerm(
                term_id, (op.op_id,), tuple(work),
                tuple(dict.fromkeys(predecessors)),
                float(metadata.get("release_ms", 0.0) or 0.0),
                startup, metadata.get("fused_into"), int(replica), inclusive, basis,
                float(phase_path), float(measured), float(ideal_path), bases))
            last_stream[stream] = term_id
    if not terms:
        holes.append("no_selected_work")
    coverage = "complete" if not holes else "conditional_subset"
    return PlanWork(plan_id or f"selected-plan:{parsed.model.name}", tuple(terms),
                    tuple(peaks.values()), coverage,
                    tuple(dict.fromkeys(holes)), int(owner),
                    any(item.empirical for item in peaks.values()))
def _peak_for(peaks, resource, unit, mode=""):
    exact = [item for item in peaks if item.resource_id == resource]
    candidates = exact or [item for item in peaks
                           if _inv.same_family(item.resource_id, resource)]
    if mode:
        candidates = [item for item in candidates if item.mode == mode] or candidates
    if unit:
        candidates = [item for item in candidates if item.unit == unit]
    return candidates[0] if candidates else None
def project_program(program, peaks, *, plan_id="lowered-program", owner=0):
    """Read-only projection of a declared Stage/KernelProgram DAG."""
    terms, holes, counter = [], [], [0]
    def new_id():
        counter[0] += 1
        return f"r{owner}:p{counter[0]}"
    def add_term(stage, predecessors):
        identifier = new_id()
        work = []
        units = dict(getattr(stage, "work_units", {}) or {})
        for resource, amount in dict(getattr(stage, "work", {}) or {}).items():
            if float(amount) <= 0:
                continue
            unit = units.get(resource, "")
            peak = _peak_for(peaks, resource, unit)
            if peak is None:
                holes.append(f"missing_program_peak:{resource}:{unit or 'native'}")
                continue
            if unit in {"ms", "peak_ms"} and peak.unit and peak.unit != unit:
                holes.append(f"unit_mismatch:{resource}:{unit}!={peak.unit}")
                continue
            work.append((peak.resource_id, float(amount), unit or peak.unit, peak.mode))
        dispatch = float(getattr(stage, "dispatch_min_ms", 0.0) or 0.0)
        latency = float(getattr(stage, "latency", 0.0) or 0.0)
        # Measured stage latency is empirical, never in the physical bound.
        basis = "declared_hardware_dispatch_minimum" if dispatch > 0 else ""
        terms.append(WorkTerm(
            identifier, (str(getattr(stage, "name", identifier)),), tuple(work),
            tuple(dict.fromkeys(predecessors)),
            float(getattr(stage, "release_ms", 0.0) or 0.0),
            dispatch, owner=owner, inclusive_min_ms=0.0, floor_basis=basis,
            measured_latency_ms=latency))
        return identifier
    def walk(node, ready, prefix=""):
        from .program import KernelProgram
        stages = tuple(getattr(node, "stages", ()) or ())
        if not stages:
            minimum = float(getattr(node, "minimum_ms", 0.0) or 0.0)
            if isinstance(node, KernelProgram):
                if minimum <= 0:
                    return tuple(ready), ()
                floor = WorkTerm(new_id(), ("program_minimum",), (),
                                 tuple(dict.fromkeys(ready)), 0.0, 0.0,
                                 owner=owner, inclusive_min_ms=minimum,
                                 floor_basis="declared_program_minimum")
                terms.append(floor)
                return (floor.id,), (floor.id,)
            holes.append(f"opaque_program_unconserved:{type(node).__name__}")
            return tuple(ready), ()
        names, exits, consumed = {}, [], set()
        for stage in stages:
            key = prefix + str(getattr(stage, "name", len(names)))
            preds, missing = [], []
            # ``KernelProgram.start`` sequences only ``Stage.after``.
            for reference in tuple(getattr(stage, "after", ()) or ()):
                target = names.get(prefix + str(reference))
                if target is None:
                    missing.append(reference)
                else:
                    preds.extend(target)
                consumed.add(prefix + str(reference))
            for reference in missing:
                holes.append(f"program_dependency_unresolved:{key}:{reference}")
            if missing:
                preds.extend(ready)
            elif not getattr(stage, "after", ()):
                preds.extend(ready)
            identifier = None
            if (dict(getattr(stage, "work", {}) or {})
                    or float(getattr(stage, "latency", 0.0) or 0.0) > 0
                    or float(getattr(stage, "dispatch_min_ms", 0.0) or 0.0) > 0
                    or getattr(stage, "keep_empty", False)):
                identifier = add_term(stage, tuple(dict.fromkeys(preds)))
            own_ready = (identifier,) if identifier else tuple(dict.fromkeys(preds))
            stage_exits = []
            nested = getattr(stage, "program", None)
            if nested is not None:
                nested_exits, nested_sinks = walk(nested, own_ready, key + ":")
                # An empty nested program forwards its incoming completion.
                stage_exits.extend(nested_sinks or nested_exits)
            elif identifier:
                stage_exits.append(identifier)
            if not stage_exits:
                stage_exits = list(preds)
            names[key] = tuple(dict.fromkeys(stage_exits))
            exits.extend(stage_exits)
        minimum = float(getattr(node, "minimum_ms", 0.0) or 0.0)
        if minimum > 0:
            floor = WorkTerm(new_id(), ("program_minimum",), (),
                             tuple(dict.fromkeys(ready)), 0.0, 0.0, owner=owner,
                             inclusive_min_ms=minimum,
                             floor_basis="declared_program_minimum")
            terms.append(floor)
            exits.append(floor.id)
        sinks = tuple(item for item in dict.fromkeys(exits)
                      if not any(item in names.get(name, ()) for name in consumed))
        return tuple(dict.fromkeys(exits)), tuple(sinks or exits)
    walk(program, ())
    if not terms and not holes:
        holes.append("no_program_stages")
    return PlanWork(plan_id, tuple(terms), tuple(peaks),
                    "complete" if not holes else "conditional_subset",
                    tuple(dict.fromkeys(holes)), owner)
_program_work = _inv.program_work
_program_phase_path = _inv.program_phase_path
__all__ = ["Peak", "PlanWork", "WorkTerm", "engine_peaks", "project_ops",
           "project_program"]
