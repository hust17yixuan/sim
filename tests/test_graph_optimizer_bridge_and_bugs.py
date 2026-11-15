"""R07-R16: service preservation, state, numerics, capture, streams, tails, DAG.

[Role] Permanent behavioral regressions for the remaining P1/P2 findings.
[Invariants] Every reproduced counterexample fails the new invariant; exported
  DAGs and ledgers are projections of executed plans, not source strings.
[Timeouts & Locks] Pure in-process simulation; total wall time stays under 5 s.
[Side-Effect Policy] No filesystem or environment writes.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.engine_plan import (
    apply_fusion, apply_stream_move, audit_capture, baseline_plan,
    capture_counterfactual, evaluate_plan, necessary_sol,
)
from mindie_sim.prediction.fusion_contract import _contract
from mindie_sim.prediction.fusion_feasibility import (
    ReductionMerge, analyze_region, tile_bytes, traffic_ledger,
)
from mindie_sim.prediction.fusion_graph import (
    OpNode, TypedGraph, is_convex, is_connected, region_causal_order,
)
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_vector,
)
from mindie_sim.prediction.fusion_bridge import fused_op_spec
from mindie_sim.prediction.graph_optimizer import run_optimizer

from graph_opt_fixtures import (
    budget_graph, external_graph, hw, prefetch_graph, scalar_cube_graph,
    state_graph, stream_graph, val,
)


def test_r07_fusion_preserves_every_declared_service_field():
    graph = external_graph(mte1=1_000_000.0)
    hardware = hw()
    base = baseline_plan(graph, hardware)
    add = next(node for node in base if node.node_id == "add")
    assert add.work_dict()["mte1"] == 1_000_000.0
    candidate = synthesize_vector(graph, ("add", "norm"), tile_rows=2,
                                  capacity_per_owner=hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion)
    fused = apply_fusion(base, candidate)[0]
    assert fused.work_dict()["mte1"] == 1_000_000.0
    assert fused.work_dict()["hbm"] == candidate.hbm_after
    assert fused.work_dict()["vector"] == add.work_dict()["vector"] + 200.0

    declared = replace(graph.ops["add"], work=(("hbm", 1234.0),))
    declared_graph = TypedGraph(graph.rows, graph.values,
                                {**graph.ops, "add": declared}, graph.public_outputs)
    plan = baseline_plan(declared_graph, hardware)
    assert next(n for n in plan if n.node_id == "add").work_dict()["hbm"] == 1234.0

    canonical = fused_op_spec(candidate)
    metadata = dict(canonical.metadata)
    assert metadata["resource_work"]["mte1"] == 1_000_000.0
    assert canonical.vector_ops == 300.0
    assert canonical.hbm_bytes == candidate.hbm_after
    assert metadata["device_kernel_group"]["buffers"]


def test_r08_state_effects_versions_and_hazards_fail_closed():
    from mindie_sim.prediction.fusion_graph import FeasibilityError
    missing = state_graph(missing_effect=True)
    with pytest.raises(FeasibilityError, match="state_effect_missing"):
        missing.validate()

    values = dict(state_graph().values)
    values["s1"] = replace(values["s1"], state_version=None)
    broken = TypedGraph(16, values, state_graph().ops, ("z", "s2"))
    with pytest.raises(FeasibilityError, match="state_write_version_missing"):
        broken.validate()

    values = {
        "s0": val("s0", "fp32", (1,), role="state", storage=4, version=0,
                  state_key="k"),
        "s1": val("s1", "fp32", (1,), role="state", producer="one", storage=4,
                  version=1, state_key="k"),
        "s2": val("s2", "fp32", (1,), role="state", producer="two", storage=4,
                  version=2, state_key="k"),
        "x": val("x", "bf16", (1,), storage=2),
    }
    ops = {
        "one": OpNode("one", "state_update", "vector", ("x", "s0"), ("s1",),
                      state_effect="read_write", params=(("state_key", "k"),)),
        "two": OpNode("two", "state_update", "vector", ("x", "s0"), ("s2",),
                      state_effect="read_write", params=(("state_key", "k"),)),
    }
    unordered = TypedGraph(1, values, ops, ("s1", "s2"))
    with pytest.raises(FeasibilityError, match="state_hazard_unordered"):
        unordered.validate()

    graph = state_graph()
    from mindie_sim.prediction.fusion_feasibility import region_legality
    legality = region_legality(graph, ("upd", "upd2"))
    assert legality.ok is False
    assert "state_version_barrier" in legality.reasons
    hardware = hw()
    with pytest.raises(ValueError, match="stream_move_preserves_barrier"):
        apply_stream_move(baseline_plan(graph, hardware), ("upd",), "aux",
                          hardware=hardware)


def test_r09_numerical_contracts_are_never_assumed_exact():
    from mindie_sim.prediction.fusion_feasibility import region_legality
    from graph_opt_fixtures import quant_graph
    for axis, group in ((99, 8), (-1, 0)):
        graph = quant_graph(axis=axis, group=group)
        legality = region_legality(graph, ("quant", "cast"))
        assert legality.ok is False
        assert any(reason.startswith("quant_") for reason in legality.reasons)

    mismatch_graph = quant_graph(target="fp16")
    mismatch = replace(mismatch_graph.ops["cast"],
                       params=(("target_dtype", "bf16"), ("rounding", "rne")))
    graph = TypedGraph(16, mismatch_graph.values,
                       {**mismatch_graph.ops, "cast": mismatch}, ("y",))
    assert "cast_target_dtype_mismatch" in _contract(graph, graph.ops["cast"]).reasons

    relaxed = replace(quant_graph().ops["cast"], params=(("target_dtype", "bf16"),))
    graph = TypedGraph(16, quant_graph().values,
                       {**quant_graph().ops, "cast": relaxed}, ("y",))
    check = _contract(graph, graph.ops["cast"])
    assert check.numerical_policy == "relaxed"
    assert not check.reasons

    exact = _contract(quant_graph(), quant_graph().ops["quant"])
    assert exact.numerical_policy == "exact"

    grouped_values = {
        "x": val("x", "bf16", (16, 64), storage=2048),
        "w": val("w", "bf16", (64, 64), role="weight", storage=8192),
        "y": val("y", "bf16", (16, 64), producer="gmm", storage=2048),
    }
    grouped = TypedGraph(16, grouped_values,
                         {"gmm": OpNode("gmm", "grouped_matmul", "cube",
                                        ("x", "w"), ("y",))}, ("y",))
    assert "grouped_matmul_group_contract_missing" in _contract(
        grouped, grouped.ops["gmm"]).reasons

    from mindie_sim.prediction.fusion_recipe import tensor_record
    record = tensor_record(replace(quant_graph().values["y"], layout="nz",
                                   alias_of="q"))
    assert record["layout"] == "nz"
    assert record["alias_of"] == "q"


def test_r10_capture_uses_a_real_counterfactual_not_an_estimate():
    values = {
        "x": val("x", "bf16", (1,), storage=8),
        "a": val("a", "bf16", (1,), producer="a", storage=8),
        "b": val("b", "bf16", (1,), producer="b", storage=8),
    }
    ops = {
        "a": OpNode("a", "cast", "vector", ("x",), ("a",),
                    work=(("vector", 10.0),),
                    params=(("target_dtype", "bf16"), ("rounding", "rne"))),
        "b": OpNode("b", "cast", "vector", ("a",), ("b",),
                    work=(("vector", 10.0),),
                    params=(("target_dtype", "bf16"), ("rounding", "rne"))),
    }
    graph = TypedGraph(1, values, ops, ("b",))
    hardware = hw(vector_ops_per_ms=1.0, launch_ms=1.0, max_streams=1)
    plan = baseline_plan(graph, hardware)
    normal = evaluate_plan(plan, hardware)
    assert normal.duration_ms == pytest.approx(21.0, abs=1e-6)
    assert audit_capture(plan, hardware)["evaluated"] is False
    assert audit_capture(plan, hardware)["host_bubble_saved_ms"] is None

    captured, audit = capture_counterfactual(plan, hardware, normal)
    assert audit["evaluated"] is True
    assert audit["host_bubble_saved_ms"] == pytest.approx(0.0, abs=1e-6)
    assert audit["host_bubble_saved_ms"] != pytest.approx(
        (normal.launches - 1) * hardware.launch_ms)

    host = (replace(plan[0], node_id="host", work=(("host", 1.0),),
                    args=("host",)),)
    blocked = audit_capture(host, hardware)
    assert blocked["status"] == "blocked"
    assert any("host_barrier" in item for item in blocked["blockers"])

    prefill = (replace(plan[0], phase="prefill"),)
    blocked = audit_capture(prefill, hardware)
    assert blocked["status"] == "blocked"
    assert any("prefill_capture_unsupported" in item for item in blocked["blockers"])


def test_r11_event_work_is_charged_and_stream_cap_cannot_be_bypassed():
    graph = stream_graph(True)
    hardware = hw(hbm_bytes_per_ms=1e9, vector_ops_per_ms=1e9,
                  cube_flops_per_ms=1e12, onchip_bytes_per_ms=4e9,
                  max_streams=2, launch_ms=0.0,
                  ub_bytes_per_aiv=1 << 40)
    base = baseline_plan(graph, hardware)
    no_cost = evaluate_plan(apply_stream_move(base, ("b",), "aux",
                                              hardware=hardware), hardware)
    assert no_cost.duration_ms == pytest.approx(1.6, abs=1e-6)
    switched = replace(hardware, stream_switch_ms=0.5)
    with_cost = evaluate_plan(apply_stream_move(
        base, ("b",), "aux", hardware=switched), switched)
    assert with_cost.duration_ms == pytest.approx(2.1, abs=1e-6)
    assert with_cost.duration_ms > no_cost.duration_ms + 0.49

    report = run_optimizer(graph, hw(
        hbm_bytes_per_ms=1e9, vector_ops_per_ms=1e9, cube_flops_per_ms=1e12,
        onchip_bytes_per_ms=4e9, max_streams=1, launch_ms=0.0,
        ub_bytes_per_aiv=1 << 40),
        evaluation_budget=16, max_candidates_per_pass=8, max_streams=2)
    assert {node.stream for node in report.final_plan} == {"main"}
    assert any("requested_streams_exceed_declared" in item.reasons
               for item in report.rejections)


def test_r12_reduction_channel_cut_requires_a_priced_merge():
    from graph_opt_fixtures import val
    values = {
        "x1": val("x1", "bf16", (1, 4096), storage=8192),
        "x2": val("x2", "bf16", (1, 4096), storage=8192),
        "h": val("h", "bf16", (1, 4096), producer="add", storage=8192),
        "g": val("g", "bf16", (4096,), role="weight", storage=8192),
        "n": val("n", "bf16", (1, 4096), producer="norm", storage=8192),
    }
    ops = {
        "add": OpNode("add", "add", "vector", ("x1", "x2"), ("h",),
                      work=(("vector", 1.0),)),
        "norm": OpNode("norm", "rms_norm", "vector", ("h", "g"), ("n",),
                       work=(("vector", 2.0),),
                       params=(("epsilon", 1e-6), ("reduction_order", "declared"))),
    }
    graph = TypedGraph(1, values, ops, ("n",))
    cut = analyze_region(graph, ("add", "norm"), tile_rows=1, channel_width=1,
                         capacity_per_owner=10000)
    assert cut.legality.ok is False
    assert "whole_width_reduction_requires_merge" in cut.legality.reasons
    full = analyze_region(graph, ("add", "norm"), tile_rows=1,
                          capacity_per_owner=10000)
    assert full.fits is False
    assert full.worksheet.ub_tile_bytes > 10000

    merged = analyze_region(graph, ("add", "norm"), tile_rows=1,
                            channel_width=1, capacity_per_owner=10000,
                            reduction_merge=ReductionMerge(4096, 1.0, 4096))
    assert merged.legality.ok is True
    assert merged.legality.reasons == ()


def test_r13_tail_tiles_conserve_physical_bytes_and_padding_is_separate():
    value = val("v", "bf16", (3, 1), storage=20, producer="p")
    assert tile_bytes(value, 2, 3) + tile_bytes(value, 1, 3, start_row=2) == 20
    assert tile_bytes(value, 2, 3) != 14

    from graph_opt_fixtures import cast_chain_graph
    values, ops = cast_chain_graph("t")
    values = {key: replace(item, shape=(3, 1), storage_bytes=20)
              for key, item in values.items()}
    graph = TypedGraph(3, values, ops, ("tz",))
    ledger = traffic_ledger(graph, ("ta", "tb"), tile_rows=2)
    ledger.assert_conserved()
    assert ledger.baseline_bytes == sum(
        item.baseline_read_bytes + item.baseline_write_bytes
        for item in ledger.values)
    assert ledger.candidate_bytes == sum(
        item.candidate_read_bytes + item.candidate_write_bytes
        + item.new_spill_bytes for item in ledger.values)
    assert ledger.saved_bytes == ledger.baseline_bytes - ledger.candidate_bytes
    assert ledger.padding_overhead_bytes >= 0

    hardware = hw()
    candidate = synthesize_vector(graph, ("ta", "tb"), tile_rows=2,
                                  capacity_per_owner=hardware.ub_bytes_per_aiv)
    if not isinstance(candidate, RejectedFusion):
        plan = apply_fusion(baseline_plan(graph, hardware), candidate)
        fused = plan[0]
        baseline_hbm = sum(node.work_dict().get("hbm", 0.0)
                           for node in baseline_plan(graph, hardware))
        assert baseline_hbm - fused.work_dict()["hbm"] == candidate.hbm_saved


def test_r14_successors_predecessors_and_convexity_follow_forward_edges():
    values = {
        "a": val("a", "bf16", (1, 1), storage=2),
        "b": val("b", "bf16", (1, 1), producer="b", storage=2),
        "c": val("c", "bf16", (1, 1), producer="c", storage=2),
        "d": val("d", "bf16", (1, 1), producer="d", storage=2),
    }
    def cast(name, inputs, deps=()):
        return OpNode(name, "cast", "vector", inputs, (name,), deps=deps,
                      params=(("target_dtype", "bf16"), ("rounding", "rne")))
    graph = TypedGraph(1, values, {
        "b": cast("b", ("a",)), "c": cast("c", ("b",)),
        "d": cast("d", ("b",), deps=("c",)),
    }, ("c", "d"))
    assert graph.successors("b") == ("c", "d")
    assert graph.predecessors("d") == ("c", "b")
    assert graph.successors("c") == ("d",)
    assert is_connected(graph, ("b", "d")) is True
    assert is_convex(graph, ("b", "d")) is False
    assert region_causal_order(graph, ("b", "c", "d")) == ("b", "c", "d")

    data_only = TypedGraph(1, values, {
        "b": cast("b", ("a",)), "c": cast("c", ("b",)),
        "d": cast("d", ("c",)),
    }, ("d",))
    assert data_only.successors("b") == ("c",)
    assert is_convex(data_only, ("b", "c")) is True


def test_r15_exported_dag_includes_prefetch_join_and_release_edges():
    graph = prefetch_graph(weight_bytes=8192, predecessor=10.0)
    hardware = hw(hbm_bytes_per_ms=1024, onchip_bytes_per_ms=4096,
                  vector_ops_per_ms=1.0, cube_flops_per_ms=1e9,
                  prefetch_capacity_bytes=1 << 20, max_streams=1)
    report = run_optimizer(graph, hardware, evaluation_budget=16,
                           max_candidates_per_pass=8)
    assert any(item["pass"] == "weight_prefetch" for item in report.applied)
    edges = report.dag()["edges"]
    assert any(edge["from"] == "prefetch_W_mm" and edge["to"] == "mm"
               and edge["kind"] in {"prefetch_completion", "data"}
               for edge in edges)
    assert any(edge["kind"] == "reservation_release" for edge in edges)
    prefetch_nodes = [node for node in report.final_plan
                      if node.node_id == "prefetch_W_mm"]
    assert prefetch_nodes and prefetch_nodes[0].priority > 0
    sol = necessary_sol(report.final_plan, hardware)
    assert sol["critical_path_ms"] >= 8.0


def test_r16_per_pass_candidate_and_engine_budget_is_shared():
    graph = budget_graph(12)
    hardware = hw()
    report = run_optimizer(graph, hardware, evaluation_budget=24,
                           max_candidates_per_pass=8)
    assert report.evaluations <= 24
    assert report.passes[0].per_pass_used <= 8
    assert all(row.per_pass_used <= 8 for row in report.passes)
    assert report.unevaluated
    assert any(item["reason"] == "per_pass_candidate_budget_exhausted"
               for item in report.unevaluated)
    assert report.budget_exhausted is True


def test_r13_scalar_cube_rejection_does_not_invent_parallelism():
    graph = scalar_cube_graph()
    hardware = hw(vector_ops_per_ms=0.1, cube_flops_per_ms=0.2,
                  hbm_bytes_per_ms=1e9, onchip_bytes_per_ms=1e9,
                  ub_bytes_per_aiv=65536, max_streams=1)
    report = run_optimizer(graph, hardware, evaluation_budget=16,
                           max_candidates_per_pass=8)
    assert report.final.duration_ms == pytest.approx(20.0, abs=1e-6)
    assert not any("mm" in item["members"] for item in report.applied
                   if item["pass"] == "cube_epilogue")
