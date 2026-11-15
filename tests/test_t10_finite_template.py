"""Finite indexed template producer for TiledMatmul and T04 compatibility.

[Role]
  Exercise the frozen ``PipelineTemplate.indexed_panels``/``index_of``
  interface from the producer side: bounded catalog construction, absolute
  index selection, stable allocation symbols, and capability-gated selection.

[Invariants]
  * The indexed catalog is built from quotient/remainder geometry, never by
    enumerating real waves and de-duplicating them.
  * Both sizes construct 6 body Phase classes, 18 body Stages and 6 builder
    calls, while ``tiles`` remains 20/80 logical panels.
  * A reader without the exact ``indexed_panels_abi == 1`` marker receives the
    original exact ordered template or detailed program, never the catalog.
  * Indexed Stage work, dependencies, allocations, timing and peaks equal the
    detailed reference for equal groups, tails, single K and quant epilogues.
  * T04's real reader is not replaced here; the executable harness below is a
    producer-side control and is labelled as such.
"""
from dataclasses import replace
from unittest.mock import patch

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.program import (
    PipelinePhase, PipelineTemplate, PipelineTemplateProgram, RecipeContext,
    Stage,
)
from mindie_sim.schema.common import ModelingError
import mindie_sim.prediction.scheduling as scheduling


def _context(m, n, k, *, cores=2, groups=1, quant=None, epilogue=False):
    op = gemm_op(
        op_id="matrix", deps=(), stage=0, layer=0, requests=(),
        m=m, n=n, k=k, weight_count=groups,
        quant=quant or QuantSpec("bf16"), category="dense_mlp",
        phase="prefill",
    )
    if epilogue:
        rows, columns = int(op.local_shape[-3]), int(op.local_shape[-2])
        operations = 4
        metadata = dict(op.metadata)
        metadata["matrix_epilogue"] = {
            "kind": "row_column_scale",
            "row_scale_bytes": rows * 4,
            "column_scale_bytes": columns * 8,
            "operations_per_output": operations,
        }
        op = replace(op, metadata=tuple(metadata.items()),
                     vector_ops=float(operations * rows * columns))
    rates = {"cube": 128, "hbm": 1024, "mte2": 16, "mte1": 1024,
             "mte3": 1024, "vector": 1024}
    pools = {"l1": 4096, "l0c": 4096}
    work = {"cube": op.flops, "hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
            "mte3": op.mte3_bytes}
    if op.vector_ops:
        work["vector"] = op.vector_ops
    return RecipeContext(
        op, work, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools},
        pool_capacities=pools,
        pool_units={name: cores for name in pools},
        resource_units={name: 1 if name == "hbm" else cores
                        for name in rates},
    )


def _count_construction(build):
    counts = {"stage": 0, "phase": 0}
    original_stage = Stage.__init__
    original_phase = PipelinePhase.__init__

    def stage_init(self, *args, **kwargs):
        counts["stage"] += 1
        return original_stage(self, *args, **kwargs)

    def phase_init(self, *args, **kwargs):
        counts["phase"] += 1
        return original_phase(self, *args, **kwargs)

    Stage.__init__ = stage_init
    PipelinePhase.__init__ = phase_init
    try:
        return build(), counts
    finally:
        Stage.__init__ = original_stage
        PipelinePhase.__init__ = original_phase


def _body_stages(template):
    return sum(len(phase.stages) for phase in template.phases)


# ---------------------------------------------------------------------------
# A bounded producer-side reader control.
#
# T04 owns the real reader and has not published its ABI marker in this
# candidate.  This harness is deliberately local: it verifies that the catalog
# is executable and equivalent to the detailed program, not that T04 consumed
# it.  It uses only the frozen public fields (tiles, index_of, stages_for,
# bindings, setup/drain) and the existing Stage/Engine contract.
# ---------------------------------------------------------------------------


def _indexed_consumer(engine, template, name, ready, *, owner="",
                      quiescent=False):
    if template.indexed_panels is None:
        raise AssertionError("producer harness requires an indexed template")
    bindings = {row["name"]: row["scope"] for row in template.bindings}
    complete = engine.future(name + ":complete")
    current = ready
    if template.setup_ms:
        current = engine.submit(name + ":setup_ms", after=(current,),
                                latency=template.setup_ms, owner=owner)

    def scoped_for(index, wave):
        def scoped(items):
            result = []
            for item in items:
                scope = bindings.get(item.name)
                if scope == "invocation":
                    prefix = name
                elif scope == "output":
                    prefix = f"{name}:output{wave}"
                elif scope == "panel":
                    prefix = f"{name}:tile{index}"
                elif scope == "slot":
                    unit = item.units[0] if item.units else 0
                    prefix = f"{name}:slot{unit}"
                else:
                    raise AssertionError(f"unbound:{item.name}")
                result.append(Allocation(f"{prefix}:{item.name}", item.pool,
                                         item.size, item.units))
            return tuple(result)
        return scoped

    setup_local = {}
    for stage in template.setup:
        deps = tuple(setup_local[key] for key in stage.after)
        setup_local[stage.name] = stage.submit(
            engine, f"{name}:setup:{stage.name}", deps or (current,),
            scoped_for(0, 0), owner)
    if setup_local:
        current = engine.submit(name + ":setup_complete",
                                after=tuple(setup_local.values()),
                                owner=owner)
    total = template.tiles
    next_index = 0
    finished = 0
    stage_tail = {}

    def issue():
        nonlocal next_index
        if next_index >= total:
            return
        index = next_index
        next_index += 1
        phase_id, end, wave, panel, slot = template.index_of(index)
        assert 0 <= phase_id < len(template.phases)
        assert index < end <= total
        local = {}
        previous = dict(stage_tail)
        scoped = scoped_for(index, wave)
        for stage in template.phases[phase_id].stages_for(index):
            deps = [local[key] for key in stage.after]
            deps.extend(previous[key] for key in stage.previous
                        if key in previous)
            if stage.name in previous:
                deps.append(previous[stage.name])
            event = stage.submit(
                engine, f"{name}:tile{index}:{stage.name}",
                tuple(deps) or (current,), scoped, owner)
            local[stage.name] = stage_tail[stage.name] = event
        engine.submit(f"{name}:tile{index}:retire",
                      after=tuple(local.values()),
                      on_complete=retire_done, owner=owner)

    def drain():
        if not template.drain:
            engine.resolve(complete)
            return
        local = {}
        for stage in template.drain:
            deps = [local[key] for key in stage.after]
            deps.extend(stage_tail[key] for key in stage.previous
                        if key in stage_tail)
            local[stage.name] = stage.submit(
                engine, f"{name}:drain:{stage.name}",
                tuple(deps) or (current,), scoped_for(0, 0), owner)
        engine.submit(name + ":drain_complete", after=tuple(local.values()),
                      on_complete=lambda _event: engine.resolve(complete),
                      owner=owner)

    def retire_done(_event):
        nonlocal finished
        finished += 1
        if finished == total:
            drain()
        else:
            issue()

    def begin(_event):
        if not total:
            drain()
            return
        for _ in range(min(template.slots, total)):
            issue()

    engine.submit(name + ":begin", after=(current,), on_complete=begin,
                  owner=owner)
    return engine.submit(name + ":complete_outer", after=(complete,),
                         owner=owner)


_indexed_consumer.indexed_panels_abi = 1


def _run(recipe, context):
    engine = Engine(
        [Resource(name, float(rate),
                  units=1 if name == "hbm" else context.core_count)
         for name, rate in context.rates.items()],
        pools=dict(context.pool_capacities),
        pool_units={name: context.core_count
                    for name in context.pool_capacities},
    )
    program = recipe.lower(context)
    done = program.start(engine, "m", engine.submit("ready"))
    engine.advance(until=(done,))
    summary = engine.summary()
    return (done.time, dict(summary["serviced"]),
            dict(summary["peak_allocated"]),
            dict(engine.peak_unit_allocated))


def _no_marker(*args, **kwargs):
    raise AssertionError("old reader must not execute an indexed catalog")


# ---------------------------------------------------------------------------
# Interface and construction.
# ---------------------------------------------------------------------------


def test_indexed_fields_are_both_present_and_validated():
    stage = (Stage("work", {"r": 1.0}),)
    mapping = lambda index: (0, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="requires_both_fields"):
        PipelineTemplate(0, 1, (PipelinePhase("body", stage, 1),),
                         indexed_panels=1)
    with pytest.raises(ValueError, match="requires_both_fields"):
        PipelineTemplate(0, 1, (PipelinePhase("body", stage, 1),),
                         index_of=mapping)
    for invalid in (True, -1, 1.0):
        with pytest.raises(ValueError, match="indexed_panels_invalid"):
            PipelineTemplate(0, 1, (PipelinePhase("body", stage, 1),),
                             indexed_panels=invalid, index_of=mapping)
    with pytest.raises(ValueError, match="phase_count_must_be_one"):
        PipelineTemplate(0, 1, (PipelinePhase("body", stage, 2),),
                         indexed_panels=2, index_of=mapping)
    with pytest.raises(ValueError, match="phase_end_must_be_none"):
        PipelineTemplate(
            0, 1, (PipelinePhase("body", stage, 1, end=2),),
            indexed_panels=1, index_of=mapping)
    with pytest.raises(ValueError, match="allocation_symbol_unbound:x"):
        PipelineTemplate(
            0, 1, (PipelinePhase("body", (Stage(
                "work", acquire=(Allocation("x", "p", 1, (0,)),)),), 1),),
            indexed_panels=1, index_of=mapping)
    with pytest.raises(ValueError, match="binding_scope_conflict:x"):
        PipelineTemplate(
            0, 1, (PipelinePhase("body", (Stage(
                "work", acquire=(Allocation("x", "p", 1, (0,)),)),), 1),),
            indexed_panels=1, index_of=mapping,
            bindings=({"name": "x", "scope": "panel", "owner": 0},
                      {"name": "x", "scope": "output", "owner": 0}))
    with pytest.raises(ValueError, match="binding_owner_mismatch:x"):
        PipelineTemplate(
            0, 1, (PipelinePhase("body", (Stage(
                "work", acquire=(Allocation("x", "p", 1, (0,)),)),), 1),),
            indexed_panels=1, index_of=mapping,
            bindings=({"name": "x", "scope": "panel", "owner": 1},))
    valid = PipelineTemplate(
        0, 1, (PipelinePhase("body", (Stage(
            "work", acquire=(Allocation("x", "p", 1, (0,)),)),), 1),),
        indexed_panels=1, index_of=mapping,
        bindings=({"name": "x", "scope": "panel", "owner": 0},))
    assert valid.tiles == 1
    assert valid.stage_for(0)[0].name == "work"


def test_finite_catalog_construction_is_six_phase_eighteen_stage_six_builder():
    for m, n, expected_tiles in ((64, 64, 20), (128, 128, 80)):
        context = _context(m, n, 160)
        templates, counts = _count_construction(
            lambda: TiledMatmul(32, 32, 32, slots=2).finite_templates(context))
        assert len(templates) == 2
        assert sum(template.tiles for template in templates) == expected_tiles
        assert sum(template.indexed_panels for template in templates) == \
            expected_tiles
        # Real constructor/factory observations, not self-reported fields.
        assert counts == {"stage": 18, "phase": 6}
        assert sum(template.phase_objects for template in templates) == 6
        assert sum(_body_stages(template) for template in templates) == 18
        assert sum(template.tile_factory_calls for template in templates) == 6
        assert sum(template.static_phase_classes for template in templates) == 6
        assert sum(template.dynamic_instances for template in templates) == \
            expected_tiles
        assert all(template.motif_phases == () for template in templates)
        assert all(template.motif_repeats == 1 for template in templates)
        for owner, template in enumerate(templates):
            assert template.owner == owner
            assert len(template.phases) == 3
            assert [phase.count for phase in template.phases] == [1, 1, 1]
            assert all(phase.end is None for phase in template.phases)
            rows = {(row["name"], row["scope"]) for row in template.bindings}
            assert rows == {("panel", "panel"), ("accumulator", "output")}
            assert all(row["owner"] == owner for row in template.bindings)


def test_absolute_index_controls_for_small_and_asymmetric_tail():
    small = TiledMatmul(32, 32, 32, slots=2).finite_templates(
        _context(64, 64, 160))[0]
    expected = {
        0: ((32, 32, 32, True, False), 1, 0, 0, 0),
        1: ((32, 32, 32, False, False), 4, 0, 1, 1),
        3: ((32, 32, 32, False, False), 4, 0, 3, 1),
        4: ((32, 32, 32, False, True), 5, 0, 4, 0),
        5: ((32, 32, 32, True, False), 6, 1, 0, 1),
    }
    for index, (geometry, end, wave, panel, slot) in expected.items():
        phase_id, actual_end, actual_wave, actual_panel, actual_slot = \
            small.index_of(index)
        assert small.phases[phase_id].key == geometry
        assert (actual_end, actual_wave, actual_panel, actual_slot) == (
            end, wave, panel, slot)
        assert small.index_of(index) == small.index_of(index)
    with pytest.raises(IndexError):
        small.index_of(-1)
    with pytest.raises(IndexError):
        small.index_of(small.indexed_panels)
    with pytest.raises(TypeError):
        small.index_of(True)

    templates = TiledMatmul(32, 32, 32, slots=2).finite_templates(
        _context(196, 65, 70, groups=2))
    assert [template.tiles for template in templates] == [36, 36]
    assert sum(template.tiles for template in templates) == 72
    owner = templates[1]
    phase_id, end, wave, panel, slot = owner.index_of(17)
    assert owner.phases[phase_id].key == (2, 1, 6, False, True)
    assert (end, wave, panel, slot) == (18, 5, 2, 1)
    phase_id, end, wave, panel, slot = owner.index_of(18)
    assert owner.phases[phase_id].key == (32, 32, 32, True, False)
    assert (end, wave, panel, slot) == (19, 6, 0, 0)
    # The same absolute index maps to a different owner-local class because
    # the strided owner placement is real, not a shared nominal sequence.
    owner_zero = templates[0]
    phase_id, _end, _wave, _panel, _slot = owner_zero.index_of(17)
    assert owner_zero.phases[phase_id].key == (2, 32, 6, False, True)


def test_single_k_panel_is_simultaneously_first_and_last():
    context = _context(70, 65, 17)
    templates = TiledMatmul(32, 32, 32, slots=2).finite_templates(context)
    assert sum(template.tiles for template in templates) == 9
    for template in templates:
        assert len(template.phases) == 4
        for index in range(template.tiles):
            phase_id, end, _wave, panel, _slot = template.index_of(index)
            key = template.phases[phase_id].key
            assert key[3] is True and key[4] is True
            assert key[2] == 17
            assert panel == 0
            assert end == index + 1
            stages = template.stage_for(index)
            compute = next(stage for stage in stages if stage.name == "compute")
            store = next(stage for stage in stages if stage.name == "store")
            assert compute.acquire
            assert store.free


def test_zero_k_and_zero_owner_geometry_never_divide_or_emit_panels():
    recipe = TiledMatmul(32, 32, 32, slots=2)
    zero_k = _context(64, 64, 32)
    zero_k = replace(zero_k, op=replace(zero_k.op,
                                       local_shape=(64, 64, 0)))
    templates = recipe.finite_templates(zero_k)
    assert len(templates) == 2
    assert all(template.tiles == 0 for template in templates)
    assert all(template.phases == () for template in templates)
    assert all(template.indexed_panels == 0 for template in templates)
    with pytest.raises(IndexError):
        templates[0].index_of(0)
    with patch.object(scheduling, "start_pipeline_template",
                      _indexed_consumer):
        reduced = recipe.lower(zero_k)
    cores = next(stage.program for stage in reduced.stages
                 if stage.name == "cores")
    assert len(cores.programs) == 2
    assert all(program.template.tiles == 0 for program in cores.programs)

    for total_m, n in ((0, 64), (64, 0)):
        context = _context(64, 64, 32)
        context = replace(context, op=replace(
            context.op, local_shape=(total_m, n, 32)))
        assert recipe.finite_templates(context) == ()
        with patch.object(scheduling, "start_pipeline_template",
                          _indexed_consumer):
            program = recipe.lower(context)
        cores = next(stage.program for stage in program.stages
                     if stage.name == "cores")
        assert cores.programs == ()


def test_indexed_catalog_matches_detailed_work_timing_and_peaks():
    cases = (
        (64, 64, 160, 1, False, 20),
        (128, 128, 160, 1, False, 80),
        (196, 65, 70, 2, False, 72),
        (70, 33, 17, 1, False, 6),
        (70, 33, 17, 1, True, 6),
        (67, 35, 129, 1, False, 30),
    )
    for m, n, k, groups, epilogue, expected_tiles in cases:
        context = _context(m, n, k, groups=groups, epilogue=epilogue,
                           quant=(quant_preset("w8a8_int8")
                                  if epilogue else None))
        detailed = _run(TiledMatmul(32, 32, 32, slots=2,
                                    template_execution=False), context)
        calls = []

        def consumer(engine, template, name, ready, *, owner="",
                     quiescent=False):
            calls.append(template.tiles)
            return _indexed_consumer(engine, template, name, ready,
                                     owner=owner, quiescent=quiescent)

        consumer.indexed_panels_abi = 1
        with patch.object(scheduling, "start_pipeline_template", consumer):
            indexed = _run(TiledMatmul(32, 32, 32, slots=2), context)
        assert indexed == detailed
        # The default lower really entered the indexed producer, and the
        # per-owner logical counts add up to the finite total.
        assert sum(calls) == expected_tiles
        assert all(tiles > 0 for tiles in calls)


def test_lower_and_pipeline_templates_share_the_same_indexed_catalog():
    context = _context(128, 128, 160)
    recipe = TiledMatmul(32, 32, 32, slots=2)
    with patch.object(scheduling, "start_pipeline_template",
                      _indexed_consumer):
        templates = recipe.pipeline_templates(context)
        program = recipe.lower(context)
    cores = next(stage.program for stage in program.stages
                 if stage.name == "cores")
    lowered = tuple(program.template for program in cores.programs)
    assert len(templates) == len(lowered) == 2
    for left, right in zip(templates, lowered):
        assert left.tiles == right.tiles
        assert left.indexed_panels == right.indexed_panels
        assert left.phase_objects == right.phase_objects == 3
        assert left.stage_objects == right.stage_objects == 9
        assert left.bindings == right.bindings
        assert [phase.key for phase in left.phases] == \
            [phase.key for phase in right.phases]
        for index in range(left.tiles):
            assert left.index_of(index) == right.index_of(index)
            assert left.stage_for(index)[0].work == \
                right.stage_for(index)[0].work


def test_reference_stages_selects_by_absolute_index_without_copying_catalog():
    template = TiledMatmul(32, 32, 32, slots=2).finite_templates(
        _context(64, 64, 160))[0]
    references = list(template.reference_stages())
    assert len(references) == template.tiles == 10
    for index, stages in enumerate(references):
        phase_id, _end, _wave, _panel, _slot = template.index_of(index)
        catalog = template.phases[phase_id].stages_for(index)
        assert len(stages) == len(catalog)
        assert all(left is right for left, right in zip(stages, catalog))
    # No second full table is retained for the old motif fields.
    assert template.motif_phases == ()
    assert template.motif_repeats == 1


def test_no_marker_keeps_exact_legacy_and_refuses_explicit_indexed():
    context = _context(64, 64, 160)
    recipe = TiledMatmul(32, 32, 32, slots=2)
    with patch.object(scheduling, "start_pipeline_template", _no_marker):
        templates = recipe.pipeline_templates(context)
        program = recipe.lower(context)
    assert sum(template.tiles for template in templates) == 20
    assert all(template.indexed_panels is None for template in templates)
    assert sum(template.phase_objects for template in templates) == 12
    assert sum(template.stage_objects for template in templates) == 36
    assert sum(template.tile_factory_calls for template in templates) == 12
    cores = next(stage.program for stage in program.stages
                 if stage.name == "cores")
    lower_templates = tuple(program.template for program in cores.programs)
    assert all(template.indexed_panels is None for template in lower_templates)
    assert sum(template.tiles for template in lower_templates) == 20

    explicit = recipe.finite_templates(context)
    assert sum(template.tiles for template in explicit) == 20
    engine = Engine([Resource("r", 1.0)])
    with patch.object(scheduling, "start_pipeline_template", _no_marker):
        with pytest.raises(ModelingError,
                           match="pipeline_indexed_consumer_unavailable"):
            PipelineTemplateProgram(explicit[0]).start(
                engine, "m", engine.submit("ready"))


def test_default_mode_tracks_the_actual_consumer_marker():
    from mindie_sim.prediction.program import (
        indexed_pipeline_consumer_available,
    )

    context = _context(64, 64, 160)
    recipe = TiledMatmul(32, 32, 32, slots=2)
    available = indexed_pipeline_consumer_available()
    templates = recipe.pipeline_templates(context)
    assert templates
    assert all((template.indexed_panels is not None) == available
               for template in templates)
    if available:
        assert sum(template.tiles for template in templates) == 20
        assert sum(template.phase_objects for template in templates) == 6
        assert sum(template.stage_objects for template in templates) == 18
    else:
        assert sum(template.tiles for template in templates) == 20
        assert sum(template.phase_objects for template in templates) == 12
        assert sum(template.stage_objects for template in templates) == 36
    program = recipe.lower(context)
    cores = next(stage.program for stage in program.stages
                 if stage.name == "cores")
    lowered = tuple(item.template for item in cores.programs)
    assert all((template.indexed_panels is not None) == available
               for template in lowered)


def test_two_owners_and_two_invocations_build_independent_catalogs():
    context = _context(196, 65, 70, groups=2)
    recipe = TiledMatmul(32, 32, 32, slots=2)
    first = recipe.finite_templates(context)
    second = recipe.finite_templates(context)
    assert first is not second
    assert [template.owner for template in first] == [0, 1]
    assert all(template.bindings[0]["owner"] == template.owner
               for template in first)
    assert first[0].index_of(17) != first[1].index_of(17)
    for left, right in zip(first, second):
        assert left is not right
        assert left.index_of(0) == right.index_of(0)
        assert left.phases[0].stages[0] is not right.phases[0].stages[0]
