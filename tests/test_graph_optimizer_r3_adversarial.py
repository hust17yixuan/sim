"""NR-S2-FIX3: executed counterexamples for every Round-3 open finding.

[Role] Reproduce each exact witness in ``astra_review_r3.md`` through the
  canonical selected-``OpSpec`` bridge, the typed projection, real ``Engine``
  pools and the exported effective DAG.
[Invariants] Every assertion is an executed work/timeline/edge fact.  Pool
  leases belong to one evaluation, per-unit capacity is conserved across
  priority tiers, declared output shapes are never rewritten and failed
  evaluations remain charged.
[Timeouts & Locks] Pure in-process simulation; discovery on a two-million-row
  graph is bounded by the caller's wall budget; no I/O, clocks or locks.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction import canonical_optimizer
from mindie_sim.prediction.canonical_optimizer import _run_canonical
from mindie_sim.prediction.coupled import coupled_rates
from mindie_sim.prediction.engine_plan import (
    HardwareProfile, PlanNode, apply_fusion, baseline_plan, evaluate_plan,
)
from mindie_sim.prediction.fusion_bridge import (
    add_prefetch, apply_rewrite, canonical_plan, canonical_stream_move,
    insert_stream_events, project_graph, rewrite_fusion,
)
from mindie_sim.prediction.fusion_contract import (
    CubeResources, ReductionMerge, _contract, tile_bytes,
)
from mindie_sim.prediction.fusion_feasibility import (
    analyze_region, region_legality, single_op_ub,
)
from mindie_sim.prediction.fusion_graph import TypedGraph
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_cube_epilogue, synthesize_vector,
)
from mindie_sim.prediction.graph_optimizer import run_optimizer
from mindie_sim.prediction.graph_optimizer_engine import _candidate_specs
from mindie_sim.prediction.optimizer_report import Budget, _canonical_dag
from mindie_sim.prediction.runtime.arbiter import component_rates
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.weight_prefetch import PrefetchPhysics, plan_prefetch
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import (
    cast_chain_graph, hw, quant_graph, scalar_cube_graph, val,
)

_CASES: dict[int, object] = {}


def case(tp: int = 1):
    if tp not in _CASES:
        _CASES[tp] = compose_case("Qwen3.8-27B", hardware="a3", tp=tp,
                                  layers=1)
    return _CASES[tp]


def _add(op_id: str, deps=(), *, vector: float = 2.0,
         hbm: float = 1_000_000.0, shape=(1, 4), metadata=()) -> OpSpec:
    return OpSpec(
        op_id=op_id, family="add", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, hbm_bytes=hbm,
        vector_ops=vector, dtype="bf16", deps=tuple(deps),
        metadata=tuple(metadata))


def _matmul(op_id: str = "mm", *, shape=(1, 512, 2048), hbm: float = 1e6,
            metadata=()) -> OpSpec:
    m, n, k = shape
    return OpSpec(
        op_id=op_id, family="gemm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=2.0 * m * n * k, hbm_bytes=float(hbm), dtype="bf16",
        metadata=tuple(metadata))


def _fuse_first(plan):
    specs, _ = _candidate_specs("vector_fusion", plan.graph, plan.hardware,
                                Budget(per_pass=32, total=64, wall_s=30))
    for spec in specs:
        if not isinstance(spec, RejectedFusion):
            return apply_rewrite(plan, rewrite_fusion(plan, spec)), spec
    raise AssertionError("no admissible vector fusion candidate")


def _vector_work(result: dict) -> float:
    return sum(float(row["work"]) for row in result["resources"]
               if str(row["resource"]).startswith("vector"))


# ---------------------------------------------------------------------------
# New P1: a shared recipe closure never shares mutable allocator state
# ---------------------------------------------------------------------------
def test_new_p1_interleaved_engines_keep_their_own_l1_lease():
    plan = canonical_plan(case(), (_matmul(),))
    decision = plan_prefetch(
        plan.graph, "mm", window_ms=50.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(plan.hardware.hbm_bytes_per_ms,
                                plan.hardware.onchip_bytes_per_ms,
                                plan.hardware.prefetch_capacity_bytes),
        capacity_bytes=plan.hardware.prefetch_capacity_bytes)
    assert decision.allowed
    trial, prefetch_id = add_prefetch(plan, decision, plan.hardware)
    prefetch_recipe = trial.book.recipes[prefetch_id]
    release_recipe = trial.book.recipes["release_W_mm"]
    units = int(plan.hardware.prefetch_units)
    capacity = float(plan.hardware.prefetch_capacity_bytes) / units
    context = SimpleNamespace(
        resource_names={"hbm": "hbm"}, pool_units={"l1": units},
        pool_names={"l1": "l1"})
    copy_program = prefetch_recipe.build(None, None).program(context)
    free_program = release_recipe.build(None, None).program(context)

    first = Engine([Resource("hbm", 1e9)], pools={"l1": capacity},
                   pool_units={"l1": units})
    second = Engine([Resource("hbm", 1e9)], pools={"l1": capacity},
                    pool_units={"l1": units})
    first_copy = copy_program.start(first, "first", first.submit("entry:first"))
    second_copy = copy_program.start(second, "second",
                                     second.submit("entry:second"))
    second_free = free_program.start(second, "second:release", second_copy)
    first.advance(until=(first_copy,))
    second.advance(until=(second_free,))
    # Finishing the second evaluation cannot consume the first evaluation's
    # lease: the first still owns exactly its own 8,388,608 B.
    assert first.allocated.get("l1") == pytest.approx(decision.target_bytes)
    assert second.allocated.get("l1") == pytest.approx(0.0)
    first_free = free_program.start(first, "first:release", first_copy)
    first.advance(until=(first_free,))
    assert first.allocated.get("l1") == pytest.approx(0.0)
    assert second.allocated.get("l1") == pytest.approx(0.0)
    first_actions = [(row["action"], row["pool"], row["size"])
                     for row in first.allocation_trace]
    assert ("acquire", "l1", decision.target_bytes / units) in first_actions
    assert ("release", "l1", decision.target_bytes / units) in first_actions


# ---------------------------------------------------------------------------
# R01: declared output shapes are authoritative
# ---------------------------------------------------------------------------
def test_r01_declared_output_shape_is_never_overwritten():
    ops = (_add("a", ("analysis_input",), shape=(1, 4)),
           _add("b", ("a",), shape=(1, 8)))
    plan = canonical_plan(case(), ops)
    assert plan.graph.values["b:out"].shape == (1, 8)
    node = plan.graph.ops["b"]
    assert node.semantic_complete is False
    assert node.param("declared_shape_mismatch") is True
    legality = region_legality(plan.graph, ("a", "b"))
    assert legality.ok is False
    assert "declared_output_shape_mismatch" in legality.reasons
    report = run_optimizer(case(), ops, evaluation_budget=8,
                           max_candidates_per_pass=4)
    assert not any(item["pass"] == "vector_fusion" for item in report.applied)
    assert any("declared_output_shape_mismatch" in rejection.reasons
               for rejection in report.rejections)
    # A consistent pair still fuses and keeps both declared shapes.
    good = canonical_plan(case(), (
        _add("x", ("analysis_input",), shape=(1, 8)),
        _add("y", ("x",), shape=(1, 8))))
    fused, _spec = _fuse_first(good)
    assert next(op for op in fused.ops
                if op.family == "native_fusion").local_shape == (1, 8)


# ---------------------------------------------------------------------------
# R02: oversized single-op UB demand is rejected, never silently erased
# ---------------------------------------------------------------------------
def test_r02_oversized_single_op_ub_is_rejected_not_erased():
    values, ops = cast_chain_graph("t")
    graph = TypedGraph(1, values, ops, ("tz",))
    hardware = hw(ub_bytes_per_aiv=4_000, usable_ub_bytes=4_000,
                  prefetch_capacity_bytes=0)
    for op_id in ("ta", "tb"):
        allocations = single_op_ub(graph, op_id, tile_rows=2,
                                   capacity_per_owner=4_000)
        assert allocations
        assert sum(item.total_bytes for item in allocations) > 4_000
    plan = baseline_plan(graph, hardware)
    assert all(node.ub_allocations for node in plan)
    measured = evaluate_plan(plan, hardware)
    assert measured.status == "failed"
    assert "single_op_ub_capacity_exceeded" in measured.reason
    report = run_optimizer(graph, hardware, evaluation_budget=4,
                           max_candidates_per_pass=2)
    assert report.baseline.status == "failed"
    assert not any(item["pass"] == "vector_fusion" for item in report.applied)


# ---------------------------------------------------------------------------
# R03: prefetch and Cube/group holders share the single L1 pool
# ---------------------------------------------------------------------------
def test_r03_prefetch_and_holder_share_one_l1_pool():
    hardware = HardwareProfile(
        1024, 1e9, 1e9, 1e9, 1 << 20, 1024, max_streams=1, launch_ms=0.0,
        device_min_ms=0.0, prefetch_units=1,
        cube_capacities=CubeResources(65536, 65536, 65536, 65536))
    plan = (
        PlanNode("hold", (("host", 10.0),), kind="kernel", stream="main",
                 ub_allocations=(("hold", "l1", 1024, (0,)),)),
        PlanNode("prefetch_W_hold", (("hbm", 1024.0),), kind="prefetch",
                 prefetch_for="hold", stream="prefetch", priority=10,
                 prefetch_alloc=("prefetch_W_hold", "l1", 1024.0, (0,))),
    )
    measured = evaluate_plan(plan, hardware)
    assert measured.status == "completed", measured.reason
    assert measured.start("hold") == pytest.approx(0.0)
    assert measured.complete("hold") == pytest.approx(10.0)
    # The 1 KiB prefetch cannot start while the 1 KiB L1 holder is live.
    assert measured.start("prefetch_W_hold") >= measured.complete("hold") - 1e-12
    assert measured.complete("prefetch_W_hold") == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# R04: per-unit owner residual survives every priority tier
# ---------------------------------------------------------------------------
def _task(priority=0, placement=None, coupled=False, remaining=None,
          start=0.0):
    return SimpleNamespace(priority=priority, placement=placement or {},
                           coupled=coupled, remaining=remaining or {},
                           caps={}, start=start, cohort_size=1)


def test_r04_per_unit_owner_residual_across_priority_tiers():
    resources = {"hbm": Resource("hbm", 1024, "bytes", units=2),
                 "vector": Resource("vector", 1e9, "ops", units=2)}
    foreground = _task(0, {"hbm": (0,)}, remaining={"hbm": 1024})
    prefetch = _task(10, {"hbm": (0,)}, remaining={"hbm": 1024})
    rates = component_rates(resources, {0: foreground, 1: prefetch}, [0, 1],
                            coupled_active=0, placed_active=2)
    assert rates["hbm"][0] == pytest.approx(512.0)
    assert rates["hbm"].get(1, 0.0) == pytest.approx(0.0)

    coupled = _task(0, {"hbm": (0,), "vector": (0,)}, coupled=True,
                    remaining={"hbm": 1024, "vector": 1024})
    rates = component_rates(resources, {0: coupled, 1: prefetch}, [0, 1],
                            coupled_solver=coupled_rates, coupled_active=1,
                            placed_active=1)
    assert rates["hbm"][0] == pytest.approx(512.0)
    assert rates["hbm"].get(1, 0.0) == pytest.approx(0.0)

    engine = Engine([Resource("hbm", 1024, "bytes", units=2)])
    first = engine.submit("foreground", {"hbm": 1024},
                          placement={"hbm": (0,)}, priority=0)
    second = engine.submit("prefetch", {"hbm": 1024},
                           placement={"hbm": (0,)}, priority=10)
    engine.advance(until=(first, second))
    engine.advance(until=(second,))
    assert first.time == pytest.approx(2.0)
    assert second.time == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# R05: local_shape already carries the rank-local weight identity
# ---------------------------------------------------------------------------
def test_r05_local_shape_weights_are_not_divided_by_tp():
    storage = []
    for tp in (1, 4, 8):
        plan = canonical_plan(case(tp), (_matmul(),))
        weights = [value for value in plan.graph.ops["mm"].inputs
                   if plan.graph.values[value].role == "weight"]
        assert len(weights) == 1
        storage.append(plan.graph.values[weights[0]].storage_bytes)
    assert storage == [2048 * 512 * 2] * 3


# ---------------------------------------------------------------------------
# R06: the Cube/Vector boundary has a priced, bandwidth-sensitive handoff
# ---------------------------------------------------------------------------
def test_r06_onchip_handoff_is_priced_and_bandwidth_sensitive():
    graph = scalar_cube_graph()
    durations = []
    for rate in (1e9, 1e-6):
        hardware = hw(vector_ops_per_ms=0.1, cube_flops_per_ms=0.2,
                      hbm_bytes_per_ms=1e9, onchip_bytes_per_ms=rate,
                      ub_bytes_per_aiv=65536, max_streams=1)
        candidate = synthesize_cube_epilogue(
            graph, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
            capability=True, cube_resources=hardware.cube_capacities)
        assert not isinstance(candidate, RejectedFusion)
        phases = dict(candidate.analysis.phase_work)
        assert "handoff" in phases and phases["handoff"]["mte1"] > 0
        assert candidate.analysis.handoff_bytes > 0
        plan = apply_fusion(baseline_plan(graph, hardware), candidate)
        measured = evaluate_plan(plan, hardware)
        assert measured.status == "completed", measured.reason
        durations.append(measured.duration_ms)
    assert durations[1] > durations[0] + 1.0
    assert durations[0] == pytest.approx(20.0, abs=1e-3)


# ---------------------------------------------------------------------------
# R07: metadata.resource_work is additive through canonical fusion
# ---------------------------------------------------------------------------
def test_r07_resource_work_is_additive_through_canonical_fusion():
    declared = (("resource_work", {"vector": 100_000_000.0}),)
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",), metadata=declared),
        _add("b", ("a",))))
    assert _vector_work(plan.evaluate()) == pytest.approx(100_000_004.0)
    fused, _spec = _fuse_first(plan)
    assert _vector_work(fused.evaluate()) == pytest.approx(100_000_004.0)
    fused_op = next(op for op in fused.ops if op.family == "native_fusion")
    assert fused_op.vector_ops == pytest.approx(100_000_004.0)


# ---------------------------------------------------------------------------
# R08: read-only state effects stay read; versions are never synthesized
# ---------------------------------------------------------------------------
def _state_effect_op(op_id, deps, effects):
    return OpSpec(
        op_id=op_id, family="elementwise", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
        padded_shape=(1, 4), launch_shape=(1, 4), hbm_bytes=8.0,
        vector_ops=2.0, dtype="bf16", deps=tuple(deps),
        metadata=(("gdn_state_effects", effects),))


def test_r08_read_only_state_and_unversioned_writes_are_strict():
    reads = ({"target": "k", "action": "read", "version": 7},)
    plan = canonical_plan(case(), (
        _state_effect_op("r1", ("analysis_input",), reads),
        _state_effect_op("r2", ("r1",), reads)))
    plan.graph.validate()
    for op_id in ("r1", "r2"):
        node = plan.graph.ops[op_id]
        assert node.state_effect == "read"
        versions = [plan.graph.values[v].state_version for v in node.inputs
                    if plan.graph.values[v].role == "state"]
        assert versions == [7]
    assert plan.evaluate()["duration_ms"] > 0

    unversioned = ({"target": "k", "action": "write"},)
    plan = canonical_plan(case(), (
        _state_effect_op("w1", ("analysis_input",), unversioned),
        _state_effect_op("w2", ("w1",), ())))
    plan.graph.validate()
    node = plan.graph.ops["w1"]
    assert node.state_effect == "write"
    assert node.param("unversioned_state") is True
    assert node.opaque is True
    assert not [value for value in plan.graph.values.values()
                if value.role == "state" and value.state_version in {0, 1}]
    legality = region_legality(plan.graph, ("w1", "w2"))
    assert legality.ok is False
    assert "state_version_barrier" in legality.reasons


# ---------------------------------------------------------------------------
# R09: negative quantization axes are normalized before shape checks
# ---------------------------------------------------------------------------
def test_r09_negative_quant_axis_is_normalized():
    graph = quant_graph(axis=-1, group=8)
    exact_values = dict(graph.values)
    exact_values["s"] = replace(exact_values["s"], shape=(16, 8),
                                storage_bytes=512)
    exact_graph = TypedGraph(16, exact_values, graph.ops, ("y",))
    exact = _contract(exact_graph, exact_graph.ops["quant"])
    assert exact.reasons == ()
    assert exact.numerical_policy == "exact"
    assert exact.reduction_width == 64

    wrong_values = dict(graph.values)
    wrong_values["s"] = replace(wrong_values["s"], shape=(16, 64),
                                storage_bytes=4096)
    wrong_graph = TypedGraph(16, wrong_values, graph.ops, ("y",))
    wrong = _contract(wrong_graph, wrong_graph.ops["quant"])
    assert "quant_scale_shape_mismatch" in wrong.reasons
    assert wrong.numerical_policy != "exact"


# ---------------------------------------------------------------------------
# R10: capture blockers cross the fusion boundary
# ---------------------------------------------------------------------------
def test_r10_capture_blockers_cross_fusion():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",),
             metadata=(("dynamic_shape", True), ("pointer_stable", False))),
        _add("b", ("a",))))
    capabilities = tuple(dict(plan.hardware.capabilities).items())
    capabilities = tuple(sorted({**dict(capabilities),
                                 "aclgraph_capture": True,
                                 "aclgraph_prefetch": True}.items()))
    plan = replace(plan, hardware=replace(plan.hardware,
                                          capabilities=capabilities))
    report = _run_canonical(plan, evaluation_budget=8, per_pass=4,
                            max_streams=None, wall_budget_s=10.0)
    assert any(item["pass"] == "vector_fusion" for item in report.applied)
    assert report.capture["status"] == "blocked"
    assert report.capture["evaluated"] is False
    blockers = tuple(report.capture["blockers"])
    assert any(item.startswith("dynamic_shape:") for item in blockers)
    assert any(item.startswith("unstable_pointer:") for item in blockers)


# ---------------------------------------------------------------------------
# R11: only optimizer-generated crossings are stripped
# ---------------------------------------------------------------------------
def test_r11_caller_events_survive_optimizer_event_stripping():
    native = OpSpec(
        op_id="native_event", family="event", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1,), local_shape=(1,), padded_shape=(1,),
        launch_shape=(1,), deps=("a",), dtype="bf16", stream="main",
        metadata=(("event_kind", "caller"),))
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)), native, _add("b", ("native_event",))))
    assert plan.evaluate()["duration_ms"] > 0
    switched = replace(plan.hardware, stream_switch_ms=0.5)
    moved = canonical_stream_move(plan, ("b",), switched, "aux")
    kept = [op for op in moved if op.op_id == "native_event"]
    assert len(kept) == 1 and kept[0].family == "event"
    assert dict(kept[0].metadata).get("optimizer_generated_event") is None
    assert kept[0].deps == ("a",)
    # The consumer may now wait on a generated crossing event, but that chain
    # must still terminate at the caller-authored event.
    by_id = {op.op_id: op for op in moved}
    pending = list(by_id["b"].deps)
    seen: set[str] = set()
    while pending:
        dep = pending.pop()
        if dep in seen:
            continue
        seen.add(dep)
        pending.extend(by_id[dep].deps if dep in by_id else ())
    assert "native_event" in seen
    generated = [op for op in moved if op.family == "event"
                 and dict(op.metadata).get("optimizer_generated_event")]
    assert generated
    twice = insert_stream_events(moved, switched)
    thrice = insert_stream_events(twice, switched)
    assert len([op for op in twice if op.op_id == "native_event"]) == 1
    assert len([op for op in twice if op.family == "event"]) == len(
        [op for op in thrice if op.family == "event"])
    trial = replace(plan, ops=twice,
                    graph=project_graph(plan.case, twice,
                                        opaque_ops=plan.opaque_ops,
                                        external_data=plan.external_data))
    timeline = [row["op_id"] for row in trial.evaluate()["timeline"]]
    assert "native_event" in timeline
    assert timeline.index("native_event") > timeline.index("a")
    assert timeline.index("b") > timeline.index("native_event")


# ---------------------------------------------------------------------------
# R12: coherent merge partition, executed work and reconciled ledger
# ---------------------------------------------------------------------------
def test_r12_reduction_merge_partition_work_and_ledger():
    graph = quant_graph()
    incoherent = synthesize_vector(
        graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100_000,
        channel_width=1, reduction_merge=ReductionMerge(128, 0, 2))
    assert isinstance(incoherent, RejectedFusion)
    assert "reduction_merge_tiles_mismatch" in incoherent.reasons
    assert "reduction_merge_work_missing" in incoherent.reasons

    candidate = synthesize_vector(
        graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100_000,
        channel_width=1, reduction_merge=ReductionMerge(128, 1_000_000, 64))
    assert not isinstance(candidate, RejectedFusion)
    analysis = candidate.analysis
    merge = dict(analysis.phase_work)["merge"]
    assert merge["vector"] == pytest.approx(1_000_000.0)
    assert merge["mte1"] == pytest.approx(256.0)
    ledger = analysis.traffic
    assert ledger.merge_overhead_bytes == 256
    assert ledger.saved_bytes == ledger.baseline_bytes - ledger.candidate_bytes
    assert sum(value.saved_bytes for value in ledger.values) == ledger.saved_bytes

    plan = apply_fusion(baseline_plan(graph, hw()), candidate)
    fused = next(node for node in plan if node.node_id.startswith("fused:"))
    work = fused.work_dict()
    assert work["vector"] == pytest.approx(1_000_002.0)
    assert work["mte1"] == pytest.approx(256.0)
    measured = evaluate_plan(plan, hw())
    assert measured.status == "completed", measured.reason
    assert dict(measured.total_work)["vector"] == pytest.approx(1_000_002.0)
    assert dict(measured.total_work)["mte1"] == pytest.approx(256.0)


# ---------------------------------------------------------------------------
# R13: a rank-1 value has one implicit row, not one element per graph row
# ---------------------------------------------------------------------------
def test_r13_rank_one_row_axis_conserves_bytes():
    value = val("v", "bf16", (64,))
    assert tile_bytes(value, 1, 1) == 128
    values, ops = cast_chain_graph("t")
    values = {key: replace(item, shape=(64,), storage_bytes=None)
              for key, item in values.items()}
    graph = TypedGraph(1, values, ops, ("tz",))
    analysis = analyze_region(graph, ("ta", "tb"), tile_rows=1,
                              capacity_per_owner=200)
    assert analysis.traffic.baseline_bytes == 512
    assert analysis.traffic.candidate_bytes == 256
    assert analysis.traffic.saved_bytes == 256
    assert analysis.worksheet.fits is False
    assert analysis.worksheet.ub_tile_bytes > 200


# ---------------------------------------------------------------------------
# R15: canonical export follows the declared execution order
# ---------------------------------------------------------------------------
def test_r15_canonical_effective_dag_matches_declaration_order():
    prefix = _add(
        "fused:pre", ("analysis_input",),
        metadata=(("device_kernel_group", {
            "id": "r3_group", "kernel": "R3Kernel", "entry": True,
            "exit": True, "buffers": ()}),))
    plan = canonical_plan(case(), (
        prefix, _add("aa", ("fused:pre",)), _add("bb", ("aa",)),
        _add("cc", ())))
    edges = [(edge["from"], edge["to"]) for edge in _canonical_dag(plan)
             if edge["kind"] == "stream_order"]
    assert ("aa", "bb") in edges
    assert ("bb", "cc") in edges
    assert ("aa", "cc") not in edges

    lazy_plan = canonical_plan(case(), (
        _add("aa", ("analysis_input",)), _add("bb", ("aa",)), _add("cc", ())))
    lazy_edges = [(edge["from"], edge["to"]) for edge in _canonical_dag(lazy_plan)
                  if edge["kind"] == "stream_order"]
    assert ("aa", "cc") in lazy_edges
    assert ("cc", "bb") in lazy_edges
    assert ("aa", "bb") not in lazy_edges


# ---------------------------------------------------------------------------
# R16: O(1) tile accounting, bounded discovery and unconditional charges
# ---------------------------------------------------------------------------
def test_r16_huge_row_discovery_is_bounded_and_failed_eval_charges(
        monkeypatch):
    values, ops = cast_chain_graph("t")
    rows = 2_000_000
    values = {key: replace(item, shape=(rows, 1), storage_bytes=None)
              for key, item in values.items()}
    graph = TypedGraph(rows, values, ops, ("tz",))
    budget = Budget(per_pass=32, total=64, wall_s=0.01)
    started = time.monotonic()
    specs, unevaluated = _candidate_specs("vector_fusion", graph, hw(), budget)
    elapsed = time.monotonic() - started
    assert specs or unevaluated
    assert elapsed < 0.05
    report = run_optimizer(graph, hw(), evaluation_budget=8,
                           max_candidates_per_pass=4, wall_budget_s=0.01)
    assert report.evaluations <= 8

    def fail(*_args, **_kwargs):
        raise ValueError("synthetic_evaluation_failure")

    monkeypatch.setattr(canonical_optimizer, "evaluate_rewrite", fail)
    failing = canonical_plan(case(), (
        _add("a", ("analysis_input",)), _add("b", ("a",))))
    charged = _run_canonical(failing, evaluation_budget=8, per_pass=4,
                             max_streams=None, wall_budget_s=10.0)
    assert charged.evaluations >= 3
    assert any("canonical_rewrite_rejected" in rejection.reasons
               for rejection in charged.rejections)
