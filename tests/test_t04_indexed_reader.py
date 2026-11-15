"""A5-T04 G1: the real indexed reader on the ordinary public path.

[Role]
  Exercise ``scheduling.start_pipeline_template`` itself, not a local harness:
  the normal ``TiledMatmul.lower`` entry must select T10's finite catalog once
  the actual callable publishes ``indexed_panels_abi == 1``, then execute the
  absolute index/end/class mapping through the existing Engine.

[Invariants]
  * The marker belongs to the real callable and stays an exact ``int`` 1.
  * 20/80 logical panels are consumed lazily through ``tiles``/``index_of``;
    construction stays 6 Phase/18 body Stage/6 builder calls.
  * Indexed public execution equals the detailed reference in completion time,
    service work and peak allocations for 20/80, tails and slots 1/2/3.
  * ``previous`` is a per-panel snapshot, separate from same-panel ``after``.
  * Allocation identity is launch/owner/scope/ordinal/symbol; two owners with
    the same display name do not collide.
  * A zero-panel catalog still runs setup/drain, and an explicit indexed-only
    template is rejected before mutation when the real callable has no marker.
"""
from dataclasses import replace

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.program import (
    PipelinePhase, PipelineTemplate, PipelineTemplateProgram, PipelineProgram,
    RecipeContext, Stage,
)
from mindie_sim.prediction.scheduling import start_pipeline_template
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


def _stage_shape(stage):
    return (
        stage.name, tuple(sorted(stage.work.items())), stage.after,
        stage.previous, stage.latency, stage.coupled, stage.priority,
        tuple(sorted((item.pool, item.size, item.units)
                     for item in stage.acquire)),
        tuple(sorted((item.pool, item.size, item.units)
                     for item in stage.free)),
        tuple(sorted((name, tuple(units))
                     for name, units in stage.placement.items())),
    )


def test_real_callable_marker_and_normal_producer_select_indexed():
    assert type(scheduling.start_pipeline_template.indexed_panels_abi) is int
    assert scheduling.start_pipeline_template.indexed_panels_abi == 1

    context = _context(64, 64, 160)
    recipe = TiledMatmul(32, 32, 32, slots=2)
    templates = recipe.pipeline_templates(context)
    assert len(templates) == 2
    assert all(template.indexed_panels is not None for template in templates)
    assert sum(template.tiles for template in templates) == 20
    assert sum(len(template.phases) for template in templates) == 6
    assert sum(len(phase.stages) for template in templates
               for phase in template.phases) == 18

    program = recipe.lower(context)
    cores = next(stage.program for stage in program.stages
                 if stage.name == "cores")
    lowered = tuple(item.template for item in cores.programs)
    assert len(lowered) == 2
    assert all(template.indexed_panels is not None for template in lowered)
    assert sum(template.tiles for template in lowered) == 20

    # No patched consumer: the ordinary public entry really executes.
    assert _run(recipe, context)[0] is not None


def test_finite_catalog_construction_stays_six_eighteen_six():
    for m, n, expected_tiles in ((64, 64, 20), (128, 128, 80)):
        context = _context(m, n, 160)
        templates, counts = _count_construction(
            lambda: TiledMatmul(32, 32, 32, slots=2).finite_templates(context))
        assert counts == {"stage": 18, "phase": 6}
        assert sum(template.tiles for template in templates) == expected_tiles
        assert sum(template.phase_objects for template in templates) == 6
        assert sum(template.stage_objects for template in templates) == 18
        assert sum(template.tile_factory_calls for template in templates) == 6
        assert sum(len(phase.stages) for template in templates
                   for phase in template.phases) == 18


def test_public_indexed_matches_detailed_for_20_and_80():
    for m, n, expected_tiles in ((64, 64, 20), (128, 128, 80)):
        context = _context(m, n, 160)
        detailed = _run(TiledMatmul(32, 32, 32, slots=2,
                                    template_execution=False), context)
        indexed = _run(TiledMatmul(32, 32, 32, slots=2), context)
        assert indexed == detailed
        templates = TiledMatmul(32, 32, 32, slots=2).finite_templates(context)
        assert sum(template.tiles for template in templates) == expected_tiles


def test_public_indexed_tails_and_slot_counts_match_detailed():
    for slots in (1, 2, 3):
        for m, n, k in ((64, 64, 160), (128, 128, 160), (70, 33, 17)):
            context = _context(m, n, k)
            detailed = _run(TiledMatmul(32, 32, 32, slots=slots,
                                        template_execution=False), context)
            indexed = _run(TiledMatmul(32, 32, 32, slots=slots), context)
            assert indexed == detailed


def test_absolute_index_end_class_and_slot_controls():
    context = _context(128, 128, 160)
    for slots in (1, 2, 3):
        template = TiledMatmul(32, 32, 32, slots=slots).finite_templates(
            context)[0]
        assert template.tiles == 40
        for index in range(template.tiles):
            phase_id, end, wave, panel, slot = template.index_of(index)
            assert 0 <= phase_id < len(template.phases)
            assert index < end <= template.tiles
            assert wave == index // 5
            assert panel == index % 5
            assert slot == index % slots
            expected_end = (index + 1 if panel in (0, 4)
                            else wave * 5 + 4)
            assert end == expected_end
            assert template.stage_for(index) == \
                template.phases[phase_id].stages_for(index)


def test_indexed_catalog_stage_shapes_match_expanded_reference():
    context = _context(128, 128, 160)
    detailed_program = TiledMatmul(
        32, 32, 32, slots=2, template_execution=False).lower(context)
    detailed_cores = next(stage.program for stage in detailed_program.stages
                          if stage.name == "cores")
    detailed = tuple(
        pipe.template(owner=owner, panels_per_output=5, k_tail=False)
        for owner, pipe in enumerate(detailed_cores.programs))
    indexed = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(context)
    for left, right in zip(indexed, detailed):
        assert left.indexed_panels is not None
        assert right.indexed_panels is None
        assert left.tiles == right.tiles == 40
        for index in range(left.tiles):
            assert tuple(map(_stage_shape, left.stage_for(index))) == \
                tuple(map(_stage_shape, right.stage_for(index)))


def test_indexed_reader_uses_previous_snapshot_and_local_after_separately(
        monkeypatch):
    captured = {}
    original = Stage.submit

    def recording(self, engine, name, predecessors, scoped, owner,
                  prepared=None):
        if name.endswith((":producer", ":consumer", ":local")):
            captured[name] = tuple(event.name for event in predecessors)
        return original(self, engine, name, predecessors, scoped, owner,
                        prepared=prepared)

    stages = (
        Stage("producer", {"r": 1.0}),
        Stage("consumer", {"r": 0.0}, previous=("producer",)),
        Stage("local", {"r": 0.0}, after=("producer",)),
    )
    template = PipelineTemplate(
        0, 2, (PipelinePhase("body", stages, 1),),
        indexed_panels=2,
        index_of=lambda index: (0, index + 1, 0, index, index % 2),
    )
    engine = Engine([Resource("r", 1.0)])
    monkeypatch.setattr(Stage, "submit", recording)
    done = start_pipeline_template(engine, template, "p",
                                   engine.submit("ready"))
    engine.advance(until=(done,))
    # The explicit previous edge is one panel behind; the local after edge is
    # the current panel's producer.  The same-stage stream tail from the
    # previous panel is a third, real edge that the detailed reference also
    # keeps, so both are asserted exactly rather than dropped.
    assert captured["p:tile1:consumer"] == (
        "p:tile0:producer", "p:tile0:consumer")
    assert captured["p:tile1:local"] == (
        "p:tile1:producer", "p:tile0:local")


def test_indexed_allocation_identity_separates_owners_with_same_display_name():
    def template(owner):
        body = (
            Stage("work", {"r": 1.0}, acquire=(
                Allocation("accumulator", "p", 1, (0,)),)),
            Stage("retire", after=("work",), free=(
                Allocation("accumulator", "p", 1, (0,)),)),
        )
        return PipelineTemplate(
            owner, 1, (PipelinePhase("body", body, 1),),
            bindings=({"name": "accumulator", "scope": "output",
                       "owner": owner},),
            indexed_panels=1,
            index_of=lambda index: (0, 1, 0, 0, 0),
        )

    engine = Engine([Resource("r", 2.0)], pools={"p": 2},
                    pool_units={"p": 1}, trace_limit=16)
    ready = engine.submit("ready")
    one = start_pipeline_template(engine, template(0), "same", ready,
                                  owner="same")
    two = start_pipeline_template(engine, template(1), "same", ready,
                                  owner="same")
    engine.advance(until=(one, two), first=False)
    assert one.time == two.time == 1
    assert engine.allocated == {"p": 0.0}
    assert engine.peak_allocated["p"] == 2
    names = {row["name"] for row in engine.allocation_trace}
    assert any("owner0:output0:accumulator" in name for name in names)
    assert any("owner1:output0:accumulator" in name for name in names)


def test_indexed_zero_panels_still_runs_setup_and_drain():
    setup = (Stage("setup", acquire=(Allocation("w", "p", 1, (0,)),)),)
    drain = (Stage("drain", {"r": 1.0},
                   free=(Allocation("w", "p", 1, (0,)),)),)
    template = PipelineTemplate(
        0, 1, (), setup=setup, drain=drain, setup_ms=2.0,
        bindings=({"name": "w", "scope": "invocation", "owner": 0},),
        indexed_panels=0,
        index_of=lambda index: (0, 1, 0, 0, 0),
    )
    engine = Engine([Resource("r", 1.0)], pools={"p": 1},
                    pool_units={"p": 1})
    done = start_pipeline_template(engine, template, "p",
                                   engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 3
    assert engine.serviced == {"r": 1.0}
    assert engine.allocated == {"p": 0.0}
    assert engine.peak_allocated["p"] == 1


def test_explicit_indexed_only_without_marker_rejects_before_mutation(
        monkeypatch):
    context = _context(64, 64, 160)
    template = TiledMatmul(32, 32, 32, slots=2).finite_templates(context)[0]
    engine = Engine([Resource("r", 1.0)])
    ready = engine.submit("ready")
    before = engine.summary()
    monkeypatch.delattr(scheduling.start_pipeline_template,
                        "indexed_panels_abi")
    with pytest.raises(ModelingError,
                       match="pipeline_indexed_consumer_unavailable"):
        PipelineTemplateProgram(template).start(engine, "m", ready)
    after = engine.summary()
    assert after["live_tasks"] == before["live_tasks"] == 1
    assert after["live_programs"] == before["live_programs"] == 0


def test_legacy_ordered_template_remains_exact_on_real_reader():
    def tile(index):
        name = f"panel{index}"
        return (
            Stage("load", {"r": 1.0},
                  acquire=(Allocation(name, "p", 1, (0,)),)),
            Stage("compute", {"r": 1.0}, after=("load",),
                  free=(Allocation(name, "p", 1, (0,)),)),
        )

    engine = Engine([Resource("r", 1.0)], pools={"p": 2},
                    pool_units={"p": 1})
    template = PipelineProgram(8, 2, tile, homogeneous=True).template()
    assert template.indexed_panels is None
    done = start_pipeline_template(engine, template, "legacy",
                                   engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 16
    assert engine.serviced == {"r": 16.0}
    assert engine.allocated == {"p": 0.0}
