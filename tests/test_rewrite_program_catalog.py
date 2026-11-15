"""T04 P-domain: pure catalog compilation and fresh binding invariants.

All expected values come from the declared Stage/PipelineTemplate work and
from quotient/remainder indexing, not from production counters.
"""
from __future__ import annotations

import time

import pytest

from mindie_sim.prediction.program import (
    Allocation,
    KernelProgram,
    PipelinePhase,
    PipelineProgram,
    PipelineTemplate,
    PipelineTemplateProgram,
    Stage,
)
from mindie_sim.prediction.program_catalog import (
    bind_program,
    catalog_resource_roles,
    compile_pipeline_template,
    compile_program,
    submit_program,
    submit_program_catalog,
)
from mindie_sim.prediction.runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    CatalogGeneration,
    OccurrenceKey,
    PoolKey,
    Submission,
)


GENERATION = CatalogGeneration(
    registry_generation="test-registry",
    implementation_generation="test-program-catalog",
    hardware_fingerprint="ascend-test",
)


def _alloc(name="buf", pool="l1", size=1024.0, units=(0,)):
    return Allocation(name, pool, size, tuple(units))


def test_kernel_catalog_preserves_declared_graph_and_work():
    kernel = KernelProgram((
        Stage("load", {"hbm": 2.5}, acquire=(_alloc(),)),
        Stage("compute", {"cube": 4.0}, after=("load",),
              caps={"cube": 8.0}, placement={"cube": (0,)},
              free=(_alloc(),)),
    ), minimum_ms=0.5)

    catalog = compile_program(kernel, generation=GENERATION)

    assert catalog.finite is None
    labels = [stage.label for stage in catalog.stages]
    assert labels == ["load", "compute", "__observation__", "__complete__"]
    load, compute, observation, complete = catalog.stages
    assert load.local_id == 0 and compute.local_id == 1
    assert compute.relative_after[0].source_node == 0
    assert [term.issued for term in load.work] == [2.5]
    assert [term.unit for term in load.work] == ["ms"]
    assert compute.caps[0].physical_resource_role == "cube"
    assert compute.placement[0].units == (0,)
    assert observation.latency_expr == 0.5
    assert {edge.source_node for edge in complete.relative_after} == {0, 1, 2}

    symbol = compute.release[0].symbol
    acquired = [spec for spec in catalog.allocation_symbols
                if spec.symbol == symbol]
    assert len(acquired) == 1
    assert acquired[0].label == "buf"
    assert acquired[0].scope_kind == "invocation"
    assert acquired[0].amount == 1024.0
    assert catalog.selected_call_count == 3
    assert catalog.semantic_sources == ("load", "compute")
    assert set(catalog_resource_roles(catalog)) == {"hbm", "cube"}


def test_catalog_identity_is_deterministic_content_not_engine_state():
    kernel = KernelProgram((Stage("s", {"cube": 1.0}),))
    first = compile_program(kernel, generation=GENERATION)
    second = compile_program(kernel, generation=GENERATION)
    assert first.catalog_id == second.catalog_id
    assert first.implementation_id == second.implementation_id
    assert first == second

    changed = compile_program(
        KernelProgram((Stage("s", {"cube": 2.0}),)),
        generation=GENERATION)
    assert changed.catalog_id != first.catalog_id
    assert first.generation == GENERATION


def test_bind_creates_fresh_occurrence_and_allocation_keys():
    catalog = compile_program(
        KernelProgram((Stage("s", {"cube": 1.0},
                            acquire=(_alloc(),), free=(_alloc(),)),)),
        generation=GENERATION)
    one = OccurrenceKey(runtime_uid="rt", launch=1)
    two = OccurrenceKey(runtime_uid="rt", launch=2)

    first = bind_program(
        catalog, occurrence=one, geometry={"m": 64}, values={},
        resources={"cube": "cube"})
    second = bind_program(
        catalog, occurrence=two, geometry={"m": 64}, values={},
        resources={"cube": "cube"})

    assert first.occurrence == one and second.occurrence == two
    assert set(first.allocations) == {0}
    left = first.allocations[0]
    right = second.allocations[0]
    assert left.key.occurrence == one and right.key.occurrence == two
    assert left.key != right.key
    assert left.pool.name == "l1" and left.pool.rank == 0
    assert left.amount == 1024.0 and left.acquire is True


def test_supplied_allocation_binding_wins_over_derived_default():
    catalog = compile_program(
        KernelProgram((Stage("s", {}, acquire=(_alloc(size=64.0),)),)),
        generation=GENERATION)
    occurrence = OccurrenceKey(runtime_uid="rt", launch=7)
    pool = PoolKey(physical_domain="pd0", rank=0, tier="hbm", name="hbm",
                   unit=0, quantity_unit="bytes")
    key = AllocationKey(occurrence=occurrence, scope_kind="invocation",
                        scope_ordinal=0, symbol=0, generation=0)
    supplied = AllocationBinding(
        key=key, pool=pool, amount=4096.0, backing=None, byte_ranges=((0, 4096),),
        owner="native", storage="kv", access="read", acquire=False)

    frame = bind_program(
        catalog, occurrence=occurrence, geometry={}, values={},
        resources={}, allocations=(supplied,))
    assert frame.allocations[0] is supplied
    assert frame.allocations[0].amount == 4096.0
    assert frame.allocations[0].acquire is False


def _indexed_template(total=1_000_000, slots=2):
    phases = (
        PipelinePhase("first", (
            Stage("load", {"hbm": 2.0}),
            Stage("compute", {"cube": 1.0}, after=("load",)),
        ), 1),
        PipelinePhase("middle", (
            Stage("load", {"hbm": 2.0}),
            Stage("compute", {"cube": 1.0}, after=("load",)),
        ), 1),
        PipelinePhase("last", (
            Stage("load", {"hbm": 2.0}),
            Stage("compute", {"cube": 1.0}, after=("load",)),
            Stage("store", {"hbm": 3.0}, after=("compute",)),
        ), 1),
    )

    def producer(index):
        if index == 0:
            return (0, 1, 0, 0, index % slots)
        if index == total - 1:
            return (2, total, 0, index, index % slots)
        return (1, index + 1, 0, index, index % slots)

    return PipelineTemplate(0, slots, phases, indexed_panels=total,
                            index_of=producer, setup_ms=0.25,
                            minimum_ms=0.5)


def test_indexed_template_compiles_bounded_classes_and_o1_lookup():
    template = _indexed_template()
    calls = {"n": 0}
    producer = template.index_of

    def counted(index):
        calls["n"] += 1
        return producer(index)

    template = PipelineTemplate(
        0, template.slots, template.phases, indexed_panels=template.indexed_panels,
        index_of=counted, setup_ms=template.setup_ms,
        minimum_ms=template.minimum_ms)
    started = time.perf_counter()
    catalog = compile_pipeline_template(template, generation=GENERATION)
    elapsed = time.perf_counter() - started

    finite = catalog.finite
    assert finite is not None and finite.total == 1_000_000
    assert len(finite.classes) == 3
    assert len(finite.classes[0].stages) == 2
    assert len(finite.classes[2].stages) == 3
    assert finite.setup_ms == 0.25 and finite.minimum_ms == 0.5
    assert catalog.stages == ()
    assert calls["n"] == 0
    assert elapsed < 1.0

    for absolute, expected_class in ((0, 0), (1, 1), (999_998, 1),
                                     (999_999, 2)):
        class_id, end, output, panel, slot = finite.index_of(absolute)
        assert class_id == expected_class
        assert end > absolute
        assert 0 <= slot < template.slots
    with pytest.raises(IndexError):
        finite.index_of(1_000_000)
    with pytest.raises(TypeError):
        finite.index_of(True)
    assert calls["n"] == 4


def test_pipeline_program_template_preserves_each_wave_geometry():
    built: list[int] = []

    def tile(index):
        built.append(index)
        return (Stage("load", {"hbm": float(index + 1)}),
                Stage("compute", {"cube": 2.0}, after=("load",)))

    pipe = PipelineProgram(12, 2, tile, homogeneous=True)
    template = pipe.template()

    assert template.indexed_panels is None
    assert built == list(range(12))
    assert template.tile_factory_calls == 12
    for index in range(12):
        stages = template.stage_for(index)
        assert stages[0].work["hbm"] == float(index + 1)
        assert stages[1].work["cube"] == 2.0


def test_pipeline_template_work_conserves_declared_tail():
    def tile(index):
        stages = [Stage("load", {"hbm": 2.0}),
                  Stage("compute", {"cube": 1.0}, after=("load",))]
        if index == 4:
            stages.append(Stage("store", {"hbm": 3.0}, after=("compute",)))
        return tuple(stages)

    pipe = PipelineProgram(5, 2, tile, homogeneous=True)
    template = pipe.template()
    hbm = 0.0
    for index in range(5):
        hbm += sum(stage.work.get("hbm", 0.0)
                   for stage in template.stage_for(index))
    assert hbm == 5 * 2.0 + 3.0


class _PreparedEngine:
    def __init__(self):
        self.prepared = None
        self.submitted = []

    def prepare_stage(self, name, work=None, **options):
        self.prepared = (name, dict(work or {}), options)
        return "PREPARED"

    def submit_prepared(self, prepared, name, *, after=(), owner=""):
        self.submitted.append((prepared, name, tuple(after), owner))
        return "EVENT"

    def submit(self, *args, **kwargs):
        raise AssertionError("prepared path must not fall back to submit")


def test_stage_uses_public_prepare_and_submit_prepared_only():
    engine = _PreparedEngine()
    stage = Stage("compute", {"cube": 1.0}, caps={"cube": 4.0})
    scoped = lambda items: tuple(items)
    prepared = stage.prepare(engine, "task", scoped)
    event = stage.submit(engine, "task", (), scoped, "owner",
                         prepared=prepared)
    assert prepared == "PREPARED"
    assert event == "EVENT"
    assert engine.submitted[0][0] == "PREPARED"
    assert engine.prepared[2]["caps"] == {"cube": 4.0}


class _CatalogEngine:
    uid = "runtime-test"

    def __init__(self):
        self.calls = []

    def new_occurrence(self):
        return OccurrenceKey(runtime_uid=self.uid, launch=len(self.calls) + 1)

    def submit_catalog(self, catalog, binding, *, name, after=(), owner=""):
        done = object()
        self.calls.append((catalog, binding, name, tuple(after), owner, done))
        return Submission(occurrence=binding.occurrence, start=object(),
                          done=done, ports={}, allocations={},
                          selected_program_id=catalog.catalog_id)


def test_submit_program_submits_one_fresh_catalog_to_the_caller_engine():
    engine = _CatalogEngine()
    kernel = KernelProgram((
        Stage("load", {"hbm": 1.0}, acquire=(_alloc(size=32.0),)),
        Stage("compute", {"cube": 2.0}, after=("load",)),
    ), minimum_ms=0.25)
    ready = object()

    result = submit_program(
        engine, kernel, name="matmul", ready=ready, owner="tile0").done

    assert len(engine.calls) == 1
    catalog, binding, name, after, owner, done = engine.calls[0]
    assert result is done
    assert name == "matmul" and owner == "tile0" and after == (ready,)
    assert catalog.finite is None
    assert binding.occurrence.runtime_uid == "runtime-test"
    assert binding.allocations[0].amount == 32.0


def test_template_program_submits_one_bounded_catalog():
    engine = _CatalogEngine()
    template = PipelineTemplate(0, 2, (
        PipelinePhase("only", (Stage("work", {"cube": 1.0}),), 5),))
    program = PipelineTemplateProgram(template)
    ready = object()

    result = program.start(engine, "panels", ready, owner="owner")

    assert len(engine.calls) == 1
    catalog, binding, name, after, owner, done = engine.calls[0]
    assert result is done
    assert catalog.finite.total == 5
    assert len(catalog.finite.classes) == 1
    assert binding.occurrence.launch == 1


def test_large_indexed_template_uses_bounded_repeating_consumer(monkeypatch):
    # The indexed producer must not hand a million absolute indices to a
    # materializing finite loop; it selects the repeating consumer instead.
    engine = _CatalogEngine()
    template = _indexed_template(total=1_000_000)
    calls = []

    def consume(engine, template, name, ready, owner=""):
        calls.append((engine, template, name, owner))
        return object()
    consume.indexed_panels_abi = 1

    import mindie_sim.prediction.scheduling as scheduling
    monkeypatch.setattr(scheduling, "start_pipeline_template", consume)
    result = PipelineTemplateProgram(template).start(engine, "panels", object())
    assert result is not None
    assert calls and calls[0][1] is template
    assert engine.calls == []
