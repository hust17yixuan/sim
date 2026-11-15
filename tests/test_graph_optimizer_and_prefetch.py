"""R01-R06: canonical bridge, UB, prefetch pool/priority and causal fusion.

[Role] Permanent behavioral regressions for the P0 bridge and the first five
  P1 physics findings.
[Invariants] Every claim is an executed Engine/graph-cost result; no assertion
  reads source strings. Canonical tests use the real Case/ModelSpec fixtures.
[Timeouts & Locks] Pure in-process simulation; total wall time stays under 5 s.
[Side-Effect Policy] No filesystem or environment writes.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.engine_plan import (
    apply_fusion, apply_stream_move, baseline_plan, evaluate_plan, settle_prefetch,
    apply_prefetch,
)
from mindie_sim.prediction.fusion_bridge import canonical_plan
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_cube_epilogue, synthesize_vector,
)
from mindie_sim.prediction.weight_prefetch import (
    PrefetchPhysics, apply_to_work, plan_prefetch,
)

from graph_opt_fixtures import (
    CUBE, canonical_mlapo, canonical_qwen, external_graph, hw, prefetch_graph,
    scalar_cube_graph, ub_contention_graph,
)


def _two_chain_plan(hardware):
    graph = ub_contention_graph()
    first = synthesize_vector(graph, ("aa", "ab"), tile_rows=1,
                              capacity_per_owner=hardware.ub_bytes_per_aiv)
    second = synthesize_vector(graph, ("ba", "bb"), tile_rows=1,
                               capacity_per_owner=hardware.ub_bytes_per_aiv)
    assert not isinstance(first, RejectedFusion)
    assert not isinstance(second, RejectedFusion)
    plan = apply_fusion(apply_fusion(baseline_plan(graph, hardware), first),
                        second)
    return graph, first, second, plan


def test_r01_canonical_qwen_layer_scores_through_graph_costs():
    raw, ops = canonical_qwen()
    plan = canonical_plan(raw, ops)
    assert plan.provenance == "selected_opspec_projection"
    direct = plan.evaluate()
    assert direct["duration_ms"] > 0
    assert plan.graph.validate()[0].status == "satisfied"
    assert any(op.op_id == "s0.l0.norm2" for op in plan.ops)

    from mindie_sim.prediction.graph_optimizer import run_optimizer
    report = run_optimizer(raw, ops, evaluation_budget=24,
                           max_candidates_per_pass=16)
    assert report.provenance == "canonical_graph_costs"
    assert report.baseline.duration_ms == pytest.approx(direct["duration_ms"])
    assert report.final.duration_ms <= report.baseline.duration_ms + 1e-9
    assert report.evaluations <= 24
    assert report.applied
    assert any(item["pass"] == "vector_fusion" for item in report.applied)
    assert all(op_id in {op.op_id for op in ops}
               for item in report.applied if item["pass"] == "streams"
               for op_id in item["members"])
    fused = [node for node in report.final_plan if "fused:" in node.node_id]
    assert fused and fused[0].recipe_id


def test_r01_mlapo_decode_keeps_source_bound_native_and_model_baseline():
    raw, ops = canonical_mlapo()
    plan = canonical_plan(raw, ops)
    direct = plan.evaluate()
    native = [op for op in ops if op.op_id == "s0.l0.mla_prolog.native"]
    assert native and native[0].source_boundary in {"kernel", "call"}
    assert native[0].flops > 0
    assert plan.graph.ops["s0.l0.mla_prolog.native"].work
    ready = canonical_plan(raw, ops, options={
        "external_ready_ms": {"analysis_input": 0.5}})
    assert ready.evaluate()["duration_ms"] == pytest.approx(
        direct["duration_ms"] + 0.5, abs=1e-6)

    from mindie_sim.prediction.graph_optimizer import run_optimizer
    report = run_optimizer(raw, ops, evaluation_budget=16,
                           max_candidates_per_pass=8)
    assert report.provenance == "canonical_graph_costs"
    assert report.baseline.duration_ms == pytest.approx(direct["duration_ms"])
    assert any(node.node_id == "s0.l0.mla_prolog.native"
               for node in report.final_plan)
    assert not any("mla_prolog.native" in item["candidate_id"]
                   for item in report.applied)
    assert report.final.duration_ms <= report.baseline.duration_ms + 1e-9


def test_r02_concurrent_ub_is_admitted_per_owner_not_report_only():
    small = hw(ub_bytes_per_aiv=30000, ub_units=1, max_streams=2,
               vector_ops_per_ms=1.0, hbm_bytes_per_ms=4096,
               onchip_bytes_per_ms=4096, cube_flops_per_ms=1e9)
    graph, first, second, plan = _two_chain_plan(small)
    assert first.analysis.worksheet.ub_tile_bytes == 24576
    assert second.analysis.worksheet.ub_tile_bytes == 24576
    plan = apply_stream_move(plan, ("fused:ba+bb",), "aux", hardware=small)
    measured = evaluate_plan(plan, small)
    assert measured.status == "completed", measured.reason
    main_done = measured.complete("fused:aa+ab")
    other_start = measured.start("fused:ba+bb")
    assert other_start >= main_done - 1e-9, (main_done, other_start)
    assert measured.complete("fused:ba+bb") > other_start + 1.0
    assert measured.duration_ms > main_done

    large = hw(ub_bytes_per_aiv=50000, ub_units=1, max_streams=2,
               vector_ops_per_ms=1.0, hbm_bytes_per_ms=4096,
               onchip_bytes_per_ms=4096, cube_flops_per_ms=1e9)
    graph2, first2, second2, plan2 = _two_chain_plan(large)
    plan2 = apply_stream_move(plan2, ("fused:ba+bb",), "aux", hardware=large)
    measured2 = evaluate_plan(plan2, large)
    assert measured2.status == "completed", measured2.reason
    assert measured2.start("fused:ba+bb") <= 1e-6
    assert measured2.complete("fused:aa+ab") > measured2.start("fused:aa+ab")
    assert measured2.duration_ms < 16.0


def test_r03_prefetch_pool_is_shared_across_targets():
    graph = prefetch_graph(weight_bytes=1024, predecessor=10.0)
    physics = PrefetchPhysics(1e9, 4e9, 1024)
    first = plan_prefetch(graph, "mm", window_ms=10.0, foreground_hbm_bytes=0.0,
                          physics=physics, capacity_bytes=1024)
    assert first.allowed and first.bytes_prefetched == 1024
    second = plan_prefetch(graph, "mm", window_ms=10.0, foreground_hbm_bytes=0.0,
                           physics=physics, capacity_bytes=1024,
                           reserved_bytes=1024)
    assert second.allowed is False
    assert second.reason == "prefetch_pool_saturated"
    assert second.bytes_prefetched == 0
    assert first.bytes_prefetched + second.bytes_prefetched <= 1024

    small = prefetch_graph(weight_bytes=8192, predecessor=10.0)
    limited = plan_prefetch(small, "mm", window_ms=10.0,
                            foreground_hbm_bytes=0.0,
                            physics=PrefetchPhysics(1e9, 4e9, 8192),
                            capacity_bytes=8192)
    assert limited.allowed
    assert limited.target_bytes <= 8192
    assert limited.bytes_prefetched + limited.hbm_read_bytes == limited.weight_bytes


def test_r04_prefetch_lower_priority_never_slows_foreground():
    graph = prefetch_graph(weight_bytes=8192, predecessor=10.0)
    hardware = hw(hbm_bytes_per_ms=1024, onchip_bytes_per_ms=4096,
                  vector_ops_per_ms=1.0, cube_flops_per_ms=1e9,
                  prefetch_capacity_bytes=1 << 20, max_streams=1)
    base = baseline_plan(graph, hardware)
    normal = evaluate_plan(base, hardware)
    assert normal.status == "completed", normal.reason
    decision = plan_prefetch(
        graph, "mm", window_ms=normal.start("mm"),
        foreground_hbm_bytes=16.0,
        physics=PrefetchPhysics(1024, 4096, 1 << 20))
    assert decision.allowed and decision.bytes_prefetched == 8192

    trial = apply_prefetch(base, decision, base[1], hardware)
    measured = evaluate_plan(trial, hardware)
    assert measured.status == "completed", measured.reason
    assert measured.complete("pre") == pytest.approx(normal.complete("pre"))
    assert measured.start("mm") == pytest.approx(normal.start("mm"))
    assert measured.prefetched("prefetch_W_mm") == 8192.0

    final = settle_prefetch(trial, "prefetch_W_mm", 8192)
    settled = evaluate_plan(final, hardware)
    assert settled.status == "completed", settled.reason
    mm = next(node for node in final if node.node_id == "mm")
    assert mm.work_dict()["hbm"] == 16.0
    assert mm.work_dict()["mte1"] == 8192.0
    assert settled.duration_ms < normal.duration_ms

    saturated = plan_prefetch(
        graph, "mm", window_ms=normal.start("mm"),
        foreground_hbm_bytes=1024 * normal.start("mm"),
        physics=PrefetchPhysics(1024, 4096, 1 << 20))
    assert saturated.allowed is False
    assert saturated.reason == "hbm_saturated_no_overlap_gain"
    assert saturated.estimated_gain_ms == 0.0
    assert decision.settled(0).allowed is False
    assert decision.settled(4096).bytes_prefetched == 4096
    assert decision.settled(4096).conserved is True
    assert decision.settled(999999).bytes_prefetched == 8192


def test_r05_tp_shard_is_identical_in_baseline_and_decision():
    weight = 32 * 1024 * 1024
    for tp in (1, 4, 8):
        values = {
            "x": replace(external_graph().values["x1"], value_id="x"),
            "W": replace(external_graph().values["g"], value_id="W",
                         shape=(16, 16), storage_bytes=weight),
            "y": replace(external_graph().values["h"], value_id="y",
                         shape=(1, 16), storage_bytes=32),
        }
        from mindie_sim.prediction.fusion_graph import OpNode, TypedGraph
        ops = {"mm": OpNode("mm", "matmul", "cube", ("x", "W"), ("y",),
                            work=(("cube", 2.0),))}
        graph = TypedGraph(1, values, ops, ("y",))
        hardware = hw(tensor_parallel=tp, prefetch_capacity_bytes=64 * 1024 * 1024,
                      hbm_bytes_per_ms=1e9, onchip_bytes_per_ms=4e9,
                      max_streams=1)
        plan = baseline_plan(graph, hardware)
        decision = plan_prefetch(
            graph, "mm", window_ms=10.0, foreground_hbm_bytes=0.0,
            physics=PrefetchPhysics(1e9, 4e9, 64 * 1024 * 1024,
                                    tensor_parallel=tp),
            capacity_bytes=64 * 1024 * 1024)
        expected = weight // tp
        assert decision.weight_bytes == expected
        assert decision.bytes_prefetched + decision.hbm_read_bytes == expected
        moved = apply_to_work(plan[0].work, decision)
        assert dict(moved)["hbm"] == pytest.approx(plan[0].work_dict()["hbm"] - expected)
        assert dict(moved)["mte1"] == expected

    sharded = {
        "x": replace(external_graph().values["x1"], value_id="x"),
        "W": replace(external_graph().values["g"], value_id="W",
                     shape=(16, 16), storage_bytes=weight // 4, sharded=True),
        "y": replace(external_graph().values["h"], value_id="y",
                     shape=(1, 16), storage_bytes=32),
    }
    from mindie_sim.prediction.fusion_graph import OpNode, TypedGraph
    graph = TypedGraph(1, sharded,
                       {"mm": OpNode("mm", "matmul", "cube", ("x", "W"), ("y",),
                                     work=(("cube", 2.0),))}, ("y",))
    decision = plan_prefetch(
        graph, "mm", window_ms=10.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(1e9, 4e9, 64 * 1024 * 1024, tensor_parallel=4),
        capacity_bytes=64 * 1024 * 1024)
    assert decision.weight_bytes == weight // 4
    assert decision.shard_basis == "declared_local"


def test_r06_cube_boundary_is_causally_ordered_and_capability_gated():
    graph = scalar_cube_graph()
    hardware = hw(vector_ops_per_ms=0.1, cube_flops_per_ms=0.2,
                  hbm_bytes_per_ms=1e9, onchip_bytes_per_ms=1e9,
                  ub_bytes_per_aiv=65536, max_streams=1)
    normal = evaluate_plan(baseline_plan(graph, hardware), hardware)
    assert normal.duration_ms == pytest.approx(20.0, abs=1e-6)
    candidate = synthesize_cube_epilogue(
        graph, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=hardware.cube_capacities)
    assert not isinstance(candidate, RejectedFusion)
    fused = apply_fusion(baseline_plan(graph, hardware), candidate)
    measured = evaluate_plan(fused, hardware)
    assert measured.duration_ms == pytest.approx(20.0, abs=1e-5)
    assert measured.duration_ms > 10.0

    no_resources = synthesize_cube_epilogue(
        graph, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=None)
    assert isinstance(no_resources, RejectedFusion)
    assert "cube_resource_constraints_missing" in no_resources.reasons
    no_capability = synthesize_cube_epilogue(
        graph, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
        capability=False, cube_resources=CUBE)
    assert isinstance(no_capability, RejectedFusion)
    assert "cube_epilogue_capability_missing" in no_capability.reasons

    from mindie_sim.prediction.fusion_graph import TypedGraph
    live_values = dict(graph.values)
    live_values["reader"] = replace(live_values["c"], value_id="reader",
                                    producer="reader", shape=(1, 1))
    live_ops = dict(graph.ops)
    live_ops["reader"] = replace(graph.ops["cast"], op_id="reader",
                                 deps=("cast",), inputs=("p",),
                                 outputs=("reader",), work=())
    live = TypedGraph(1, live_values, live_ops, ("c", "reader"))
    blocked = synthesize_cube_epilogue(
        live, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=CUBE)
    assert isinstance(blocked, RejectedFusion)
    assert "single_use_violated" in blocked.reasons


def test_r06_unknown_capability_is_not_admitted_by_default():
    from mindie_sim.prediction.engine_plan import HardwareProfile
    profile = HardwareProfile(1e9, 1e9, 1e12, 4e9, 262144, 0,
                              capabilities=(("fixpipe_epilogue", False),),
                              cube_capacities=CUBE)
    assert profile.capability("fixpipe_epilogue") is False
    assert profile.capability("cube_prologue") is False
    with pytest.raises(ValueError, match="hardware_capability_undeclared"):
        profile.require_capability("cube_prologue")
