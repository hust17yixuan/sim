"""Invocation floors, template Engine branches and SOL work/path walkers.

[Role]
  Keep the nested `InvocationProgram` / `_FloorJoinProgram` rules, the
  flat-template Engine-branch predicate and every selected-work/phase walker in
  one bounded helper so `sol_work` projects exactly the completion constraints
  the selected Engine submits.

[Invariants]
  * Only `InvocationProgram.timing`/`.constraints` and
    `_FloorJoinProgram.provenance`/`.constraints` are invocation floors; a
    generic `body`/`minimum_ms` pair is not reclassified.
  * Physical admission is per constraint:
    `sol_provenance.affirmative_physical_provenance`.  One admitted constraint
    never launders an empirical sibling's larger duration.
  * A root-dominating invocation contributes startup to the whole operator;
    a stage-scoped nested invocation contributes only to that stage's node
    duration and floor reporting, never to unrelated parallel work.
  * `Engine` branches are selected independently for every
    `PipelineTemplateProgram`: a flat template's minimum is anchored at that
    template's own entry (`device_root`), never inherited from an outer
    template's consumer; every other shape uses `setup_end`.
  * Physical duration is `max(ready + minimum, ready + startup + body)` with
    each admitted source kept separate; `ideal_zero` zeroes both.

[Timeouts & Locks]
  Finite attribute/class checks and bounded stage walks; no locks, I/O or
  measurement.

[Side-Effect Policy]
  Read-only; never mutates a program, Engine or Case.
"""
from __future__ import annotations

from .sol_provenance import affirmative_physical_provenance, is_measured_minimum

_RESOURCE_ALIASES = {"matrix": "cube", "memory": "hbm", "cube": "cube",
                     "vector": "vector", "hbm": "hbm", "mte2": "mte2",
                     "mte2_aic": "mte2", "mte2_aiv": "mte2", "mte3": "mte3",
                     "mte1": "mte1", "fixpipe": "fixpipe", "link": "link"}
_MAX_TEMPLATE_TILES = 200_000


def _family(name):
    text = str(name)
    return text if text.startswith("link:") else text.split("@", 1)[0]


def same_family(left, right):
    a, b = _family(left), _family(right)
    return (_RESOURCE_ALIASES.get(a, a) == _RESOURCE_ALIASES.get(b, b)
            or a == b)


def _constraints(fields):
    startup, minimum, provenance, constraints = fields
    if constraints:
        return constraints
    from mindie_sim.schema.invocation import FloorConstraint
    if startup or minimum:
        return (FloorConstraint(startup, minimum, provenance, "invocation"),)
    return ()


def invocation_fields(node):
    """``(startup, minimum, provenance, constraints)`` for one floor wrapper."""
    from .invocation import InvocationProgram
    if isinstance(node, InvocationProgram):
        timing = node.timing
        return (float(timing.startup_ms), float(timing.minimum_ms),
                dict(timing.provenance), tuple(node.constraints))
    from .invocation_binding import _FloorJoinProgram
    if isinstance(node, _FloorJoinProgram):
        return (0.0, float(node.minimum_ms),
                dict(getattr(node, "provenance", {}) or {}),
                tuple(getattr(node, "constraints", ())))
    return None


def admitted(fields):
    return any(affirmative_physical_provenance(item.provenance)
               for item in _constraints(fields))


def declared_startup(fields):
    return max((item.startup_ms for item in _constraints(fields)
                if affirmative_physical_provenance(item.provenance)), default=0.0)


def declared_minimum(fields):
    return max((item.minimum_ms for item in _constraints(fields)
                if affirmative_physical_provenance(item.provenance)), default=0.0)


def body_start(ready, fields, include_minimum):
    """The body starts after admitted startup only when it is physical."""
    startup = declared_startup(fields) if include_minimum else 0.0
    return ready + startup if startup > 0 else ready


def completion(body_end, ready, fields, include_minimum, measured_floor):
    """Engine completion for one invocation wrapper, per admitted constraint."""
    if not include_minimum:
        return body_end
    minimum = declared_minimum(fields)
    if minimum > 0 and not is_measured_minimum(minimum, measured_floor):
        return max(body_end, ready + minimum)
    return body_end


def floor_rows(fields, scoped=False):
    """Declared or empirical floor rows for `project_ops` aggregation.

    ``scoped`` rows still report their basis but never promote a nested floor
    into the whole operator's dispatch/inclusive minimum.
    """
    rows = []
    for item in _constraints(fields):
        if affirmative_physical_provenance(item.provenance):
            if item.startup_ms > 0:
                rows.append((item.startup_ms, 0.0,
                             "declared_invocation_startup", scoped))
            if item.minimum_ms > 0:
                rows.append((0.0, item.minimum_ms,
                             "declared_invocation_minimum", scoped))
        else:
            empirical = max(item.startup_ms, item.minimum_ms)
            if empirical > 0:
                rows.append((0.0, empirical, "measured_stage_latency", scoped))
    return tuple(rows)


def template_catalog_selected(template):
    """True when `PipelineTemplateProgram.start` selects `submit_template`."""
    if getattr(template, "indexed_panels", None) is not None:
        return False
    stages = [*getattr(template, "setup", ()), *getattr(template, "drain", ())]
    try:
        for group in template.reference_stages():
            stages.extend(group)
    except TimeoutError:
        raise
    except Exception:
        return False
    for stage in stages:
        if (getattr(stage, "program", None) is not None
                or getattr(stage, "acquire", ())
                or getattr(stage, "free", ())):
            return False
    return True


def _add_stage(work, stage, mult=1):
    units = dict(getattr(stage, "work_units", {}) or {})
    for resource, amount in dict(getattr(stage, "work", {}) or {}).items():
        value = float(amount) * mult
        if value <= 0:
            continue
        key = (str(resource), units.get(resource, ""))
        work[key] = work.get(key, 0.0) + value


def _stage_floor(stage, floors):
    dispatch = float(getattr(stage, "dispatch_min_ms", 0.0) or 0.0)
    latency = float(getattr(stage, "latency", 0.0) or 0.0)
    if dispatch > 0:
        floors.append((dispatch, 0.0, "declared_hardware_dispatch_minimum"))
    if latency > 0:
        floors.append((0.0, latency, "measured_stage_latency"))


def _walk_template_work(template, factor, level, work, holes, floors, engine=None):
    """Every tile, residue group and nested Stage.program, exactly once."""
    tiles = int(getattr(template, "tiles", 0) or 0)
    if tiles > _MAX_TEMPLATE_TILES:
        holes.append(f"opaque_program_work:{type(template).__name__}:{tiles}")
        return
    try:
        reference = template.reference_stages()
        for stages in reference:
            for stage in stages:
                _add_stage(work, stage, factor)
                _stage_floor(stage, floors)
                if getattr(stage, "program", None) is not None:
                    _walk_work(stage.program, factor, level + 1, work, holes,
                               floors, engine)
    except TimeoutError:
        raise
    except Exception as exc:
        holes.append(f"opaque_template_stages:{type(exc).__name__}")
        return
    for stage in (*template.setup, *template.drain):
        _add_stage(work, stage, factor)
        _stage_floor(stage, floors)
        if getattr(stage, "program", None) is not None:
            _walk_work(stage.program, factor, level + 1, work, holes, floors, engine)
    minimum = float(getattr(template, "minimum_ms", 0.0) or 0.0)
    if minimum > 0:
        floors.append((0.0, minimum, "declared_program_minimum", level > 0))


def _scratch_work(program, engine):
    """Service one non-stage program on a throwaway Engine copy."""
    from .engine import Engine
    scratch = Engine(list(engine.resources.values()), pools=dict(engine.pools),
                     pool_units=dict(engine.pool_units))
    ready = scratch.submit("sol_probe_ready")
    end = program.start(scratch, "sol_probe", ready, owner="sol_probe")
    scratch.advance(until=(end,))
    return {name: float(amount) for name, amount in scratch.serviced.items()
            if float(amount) > 0}


def _walk_work(node, factor, level, work, holes, floors, engine=None):
    from .program import (KernelProgram, ParallelProgram, PipelineProgram,
                          PipelineTemplateProgram)
    if level > 12:
        holes.append(f"lowered_program_depth_exceeded:{type(node).__name__}")
        return
    if isinstance(node, KernelProgram):
        for stage in node.stages:
            _add_stage(work, stage, factor)
            _stage_floor(stage, floors)
            if getattr(stage, "program", None) is not None:
                _walk_work(stage.program, factor, level + 1, work, holes, floors,
                           engine)
        if float(node.minimum_ms or 0.0) > 0:
            floors.append((0.0, float(node.minimum_ms),
                           "declared_program_minimum", level > 0))
        return
    if isinstance(node, ParallelProgram):
        for child in node.programs:
            _walk_work(child, factor, level + 1, work, holes, floors, engine)
        return
    if isinstance(node, PipelineTemplateProgram):
        _walk_template_work(node.template, factor, level, work, holes, floors,
                            engine)
        return
    if isinstance(node, PipelineProgram):
        count = int(getattr(node, "count", 0) or 0)
        if count > _MAX_TEMPLATE_TILES:
            holes.append(f"opaque_program_work:{type(node).__name__}:{count}")
            return
        if node.homogeneous:
            try:
                template = node.template()
            except TimeoutError:
                raise
            except Exception as exc:
                holes.append(f"opaque_program_work:{type(node).__name__}:"
                             f"{type(exc).__name__}")
                return
            _walk_template_work(template, factor, level, work, holes, floors,
                                engine)
            return
        holes.append(f"opaque_program_work:{type(node).__name__}")
        return
    fields = invocation_fields(node)
    if fields is not None:
        _walk_work(node.body, factor, level + 1, work, holes, floors, engine)
        floors.extend(floor_rows(fields, scoped=level > 0))
        return
    body = getattr(node, "body", None)
    if body is not None:
        _walk_work(body, factor, level + 1, work, holes, floors, engine)
        minimum = float(getattr(node, "minimum_ms", 0.0) or 0.0)
        if minimum > 0:
            floors.append((0.0, minimum, "declared_invocation_minimum",
                           level > 0))
        return
    if engine is not None and callable(getattr(node, "start", None)):
        try:
            for name, amount in _scratch_work(node, engine).items():
                key = (name, "")
                work[key] = work.get(key, 0.0) + amount * factor
            return
        except TimeoutError:
            raise
        except Exception as exc:
            holes.append(f"opaque_program_work:{type(node).__name__}:"
                         f"{type(exc).__name__}")
            return
    holes.append(f"opaque_program_work:{type(node).__name__}")


def program_work(program, mult=1, depth=0, engine=None):
    """Aggregate every Stage work of one lowered program; holes on opacity."""
    work, holes, floors = {}, [], []
    _walk_work(program, mult, depth, work, holes, floors, engine)
    return work, holes, floors


def _peak_value(peaks, resource, unit):
    values = list(peaks.values()) if isinstance(peaks, dict) else list(peaks or ())
    exact = [item for item in values if item.resource_id == resource]
    candidates = exact or [item for item in values
                           if same_family(item.resource_id, resource)]
    if unit:
        candidates = [item for item in candidates if item.unit == unit]
    return candidates[0] if candidates else None


def _stage_service(stage, factor, peaks, holes):
    service = 0.0
    units = dict(getattr(stage, "work_units", {}) or {})
    for resource, amount in dict(getattr(stage, "work", {}) or {}).items():
        value = float(amount) * factor
        if value <= 0:
            continue
        unit = units.get(resource, "")
        peak = _peak_value(peaks, resource, unit)
        if peak is None or peak.peak_per_ms <= 0:
            source = f":{peak.source}" if peak is not None and peak.source else ""
            holes.append(f"missing_program_peak:{resource}:{unit or 'native'}{source}")
            continue
        service = max(service, value / peak.peak_per_ms)
    return service


def walk_stage_set(stages, peaks, holes, level, ready, floor, include_minimum=True,
                   catalog_engine=False):
    """Longest intra-phase path over one Stage set, honoring ``Stage.after``."""
    events, end = {}, ready
    for stage in stages:
        after = tuple(getattr(stage, "after", ()) or ())
        holes.extend(f"program_dependency_unresolved:{stage.name}:{ref}"
                     for ref in after if ref not in events)
        base = max((events[ref] for ref in after if ref in events), default=ready)
        own = _stage_service(stage, 1.0, peaks, holes)
        nested = getattr(stage, "program", None)
        completion = (walk_path(nested, peaks, holes, level + 1, base + own, floor,
                                include_minimum, catalog_engine)
                      if nested is not None else base + own)
        events[stage.name] = completion
        end = max(end, completion)
    return end


def walk_template_path(template, peaks, holes, level=0, ready=0.0, floor=0.0,
                       include_minimum=True, catalog_engine=False):
    """Walk one template, selecting its own Engine branch from the template."""
    if int(getattr(template, "tiles", 0) or 0) > _MAX_TEMPLATE_TILES:
        holes.append(f"opaque_program_work:{type(template).__name__}:{template.tiles}")
        return ready
    setup_end = (walk_stage_set(template.setup, peaks, holes, level, ready, floor,
                                include_minimum, catalog_engine)
                 if template.setup else ready)
    body_end = setup_end
    try:
        for stages in template.reference_stages():
            body_end = max(body_end, walk_stage_set(
                stages, peaks, holes, level, setup_end, floor, include_minimum,
                catalog_engine))
    except TimeoutError:
        raise
    except Exception:
        holes.append(f"opaque_template_stages:{type(template).__name__}")
        return setup_end
    drain_end = (walk_stage_set(template.drain, peaks, holes, level, body_end,
                                floor, include_minimum, catalog_engine)
                 if template.drain else body_end)
    best = max(setup_end, body_end, drain_end)
    minimum = float(getattr(template, "minimum_ms", 0.0) or 0.0)
    if (include_minimum and minimum > 0
            and not is_measured_minimum(minimum, floor)):
        # Decide for *this* template, never inherit a parent consumer's branch.
        catalog_branch = bool(catalog_engine
                              and template_catalog_selected(template))
        anchor = ready if catalog_branch else setup_end
        best = max(best, anchor + minimum)
    return best


def walk_path(node, peaks, holes, level=0, ready=0.0, floor=0.0,
              include_minimum=True, catalog_engine=False):
    from .program import (KernelProgram, ParallelProgram, PipelineProgram,
                          PipelineTemplateProgram)
    if level > 12:
        holes.append(f"lowered_program_depth_exceeded:{type(node).__name__}")
        return ready
    if isinstance(node, KernelProgram):
        end = walk_stage_set(node.stages, peaks, holes, level, ready, floor,
                             include_minimum, catalog_engine)
        minimum = float(getattr(node, "minimum_ms", 0.0) or 0.0)
        if (include_minimum and minimum > 0
                and not is_measured_minimum(minimum, floor)):
            end = max(end, ready + minimum)
        return end
    if isinstance(node, ParallelProgram):
        return max((walk_path(child, peaks, holes, level + 1, ready, floor,
                              include_minimum, catalog_engine)
                    for child in node.programs), default=ready)
    if isinstance(node, PipelineTemplateProgram):
        return walk_template_path(node.template, peaks, holes, level, ready, floor,
                                  include_minimum, catalog_engine)
    if isinstance(node, PipelineProgram):
        if int(getattr(node, "count", 0) or 0) > _MAX_TEMPLATE_TILES:
            return ready
        if node.homogeneous:
            try:
                return walk_template_path(node.template(), peaks, holes, level,
                                          ready, floor, include_minimum,
                                          catalog_engine)
            except TimeoutError:
                raise
            except Exception:
                holes.append(f"opaque_program_work:{type(node).__name__}")
        return ready
    fields = invocation_fields(node)
    if fields is not None:
        start = body_start(ready, fields, include_minimum)
        end = walk_path(node.body, peaks, holes, level + 1, start, floor,
                        include_minimum, catalog_engine)
        return completion(end, ready, fields, include_minimum, floor)
    body = getattr(node, "body", None)
    if body is not None:
        end = walk_path(body, peaks, holes, level + 1, ready, floor,
                        include_minimum, catalog_engine)
        minimum = float(getattr(node, "minimum_ms", 0.0) or 0.0)
        if (include_minimum and minimum > 0
                and not is_measured_minimum(minimum, floor)):
            end = max(end, ready + minimum)
        return end
    return ready


def program_phase_path(program, peaks, floor=0.0, include_minimum=True,
                       catalog_engine=False):
    holes = []
    return (walk_path(program, peaks, holes, 0, 0.0, floor, include_minimum,
                      catalog_engine), holes)


__all__ = [
    "admitted", "body_start", "completion", "declared_minimum",
    "declared_startup", "floor_rows", "invocation_fields", "program_phase_path",
    "program_work", "same_family", "template_catalog_selected", "walk_path",
    "walk_stage_set", "walk_template_path",
]
