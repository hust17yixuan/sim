"""Permanent FIX3 regressions for all sixteen R2 closure contracts.

[Role]
Test owner work conservation, finite tails, DAG causality, ABI capacities,
public validation, prior consent, normalization and bounded integer descent.
[Invariants]
Expected values come from explicit arithmetic or independently expanded DAGs.
No expected failures, skipped tests, fixture rewrites or calibration claims.
[Timeouts & Locks]
Small deterministic cases only; no waits, locks, network or subprocesses.
[Side-Effect Policy]
Read-only tests; pytest artifacts are placed in caller-selected scratch space.
"""

import inspect
import random
import sys
from dataclasses import replace
from fractions import Fraction

import pytest

from mindie_sim.opflow import build_flow, evaluate_flow, get_profile
from mindie_sim.opflow.exact_math import floor_sum
from mindie_sim.opflow.flow_schema import (
    BufferPool, BufferSpec, ExecutionArchetype, OpFlowTemplate, PipeWork, buffer_capacity,
)
from mindie_sim.opflow.opflow_mcp import opflow_evaluate, opflow_hardware, opflow_triton
from mindie_sim.opflow.paradigm_catalog import evaluate_operator, list_operators
from mindie_sim.opflow.schedule_math import pipeline_items, pipeline_tail
from mindie_sim.opflow.transfer_accounting import (
    GmStrides, TransferRequest, account_transfer, check_pool_capacity, plan_buffer_pool,
)
from mindie_sim.opflow.triton_lowering import analyze_triton_lowering
from mindie_sim.opflow.v4_math import evaluate_math
pytest.importorskip("mindie_profiler")
from mindie_profiler.aclgraph_multistream import ParallelRegion, StreamInterval, diagnose_region
from mindie_profiler.aclgraph_mcp import aclgraph_diagnose
from mindie_profiler.memory_evidence import region_memory_service
from mindie_knowledge.distiller_text import normalize_text

P = get_profile("Ascend950PR")


def vector(total=11, tile=10, **kwargs):
    return OpFlowTemplate("closure", ExecutionArchetype.PURE_VECTOR, total, tile, 0,
                          (PipeWork("VEC", total * P.rate_for("VEC", "fp32") / 1e6),),
                          dtype="fp32", **kwargs)


@pytest.mark.parametrize("accepted", [(100, 0), (50, 50), (100,), (0, 100, 1), (0, 0)])
@pytest.mark.parametrize("partitions", [1, 2, 4])
def test_owner_service_and_exact_conservation(accepted, partitions):
    f = build_flow("commit_recurrent_kda", batch=len(accepted), heads=1,
                   partitions=partitions, accepted_tokens=accepted)
    assert [x.steps for x in f.owner_work] == list(accepted)
    assert sum(x.vector_ops*x.multiplicity for x in f.owner_work) == sum(accepted)*3*128**2
    assert sum(x.read_bytes*x.multiplicity for x in f.owner_work) == f.gm_read_bytes
    r = evaluate_flow(f, P, active_units=len(accepted)*partitions)
    bound = max(accepted)*3*128**2/partitions/P.rate_for("VEC", "fp32")*1e6
    assert r.t_pipeline_us >= bound
    if partitions == 1 and accepted in ((100, 0), (100,), (50, 50)):
        expected = 5.857317818181818 if accepted == (50, 50) else 11.662391636363637
        assert r.t_pipeline_us == pytest.approx(expected)
    serial = evaluate_flow(f, P, active_units=1)
    owners = serial.extra["owner_schedule"]["owner_service_us"]
    assert serial.t_pipeline_us == pytest.approx(sum(t*x.multiplicity for t, x in zip(owners, f.owner_work)))


@pytest.mark.parametrize("total,active,expected", [(11, 1, 11), (21, 2, 11), (31, 2, 20),
                                                  (41, 2, 21), (61, 3, 21), (1, 1, 1)])
@pytest.mark.parametrize("policy", ["independent_service_bound", "barrier_waves"])
def test_actual_final_wave(total, active, expected, policy):
    r = evaluate_flow(vector(total, schedule_policy=policy), P, active_units=active)
    assert r.t_pipeline_us == pytest.approx(expected)
    assert r.t_unit_us == pytest.approx(expected)


def expanded_chain(items, slots, latency, distance):
    # Explicit max-plus recurrence, independently expanded by item and stage.
    ends, starts = {}, {}
    for i, stages in enumerate(items):
        for j, t in enumerate(stages):
            predecessors = [Fraction()]
            if i:
                predecessors.append(ends[i-1, j])
            if j:
                predecessors.append(ends[i, j-1])
            if j == 0:
                if slots and i >= slots:
                    predecessors.append(ends[i-slots, len(stages)-1])
                if i >= distance:
                    predecessors.append(starts[i-distance, 0]+latency)
            starts[i, j] = max(predecessors)
            ends[i, j] = starts[i, j]+t
    return ends[len(items)-1, len(items[0])-1]


def test_finite_tail_matches_independent_oracle():
    assert pipeline_tail((1, 1, 1), (0, 1, 0), 10**12, slots=1)["exact_makespan"] == 3*10**12+1
    rng = random.Random(34011)
    for _ in range(120):
        full = [Fraction(rng.randrange(1, 12), 3) for _ in range(3)]
        tail = [x*Fraction(rng.randrange(1, 5), 5) for x in full]
        count, suffix = rng.randrange(8), rng.randrange(1, 4)
        slots, distance = rng.choice((None, 1, 2, 3)), rng.randrange(1, 4)
        latency = Fraction(rng.randrange(8), 2)
        items = [full]*count+[tail]*suffix
        expected = expanded_chain(items, slots, latency, distance)
        assert pipeline_tail(full, tail, count, suffix, slots,
                             recurrence_latency=latency, recurrence_distance=distance)["exact_makespan"] == expected
        assert pipeline_items(items, slots, recurrence_latency=latency,
                              recurrence_distance=distance)["exact_makespan"] == expected


@pytest.mark.parametrize("total,active", [(1000, 1), (1001, 1), (21, 2), (31, 2)])
def test_triton_penalty_propagates_through_schedule(total, active):
    f = vector(total)
    base = evaluate_flow(f, P, active_units=active)
    penalized = evaluate_flow(f, P, active_units=active,
                              triton_features={"has_i64_mask_or_noncontig_reduce": True})
    assert penalized.t_pipeline_us == pytest.approx(base.t_pipeline_us*3.3)


def test_triton_observed_and_modeled_scopes():
    f = vector(11)
    integrated = evaluate_flow(f, P, active_units=1, triton_features={"has_cv_pingpong": True})
    assert integrated.extra["triton_detail"].observed_overlap_ratio == pytest.approx(1)
    call_cost = evaluate_flow(vector(1000), P, active_units=1, triton_features={
        "random_gather_to_dot_mode": "device_gather_workspace", "gather_bytes": 0, "launch_us": 7})
    assert call_cost.t_pipeline_us == pytest.approx(1007)
    r = analyze_triton_lowering(P, aic_us=5, aiv_us=2, wall_us_hint=100)
    assert r.observed_overlap_ratio == pytest.approx(.07)
    assert r.modeled_overlap_ratio == pytest.approx(7/5.1)
    r = analyze_triton_lowering(P, aic_us=5, aiv_us=2, items=10, wall_us_hint=100,
                                features={"has_i64_mask_or_noncontig_reduce": True})
    assert r.observed_overlap_ratio == pytest.approx(.7)


def fork_graph():
    return dict(name="fork", capture_root_stream_id=0, nodes=[
        dict(id="fork", stream_id=0, duration_us=2, records=["enter"]),
        dict(id="a", stream_id=1, duration_us=10, waits=["enter"], records=["ae"]),
        dict(id="b", stream_id=2, duration_us=6, waits=["enter"], records=["be"]),
        dict(id="join", stream_id=0, duration_us=3, waits=["ae", "be"]),
    ])


def test_multistream_dag_critical_path_and_root_closure():
    region = ParallelRegion.from_mapping(fork_graph())
    result = diagnose_region(region)
    assert result.code == "OK"
    assert result.span_us == 15
    assert result.wall_us == 18
    assert result.net_speedup_us == 3
    assert result.details["event_us"] == 3  # three events, only two on the critical path
    assert result.details["event_count"] == 3
    assert result.details["overlap_us"] == 6
    assert result.details["capture_hazards"] == ()
    mcp = aclgraph_diagnose(fork_graph())
    assert mcp["ok"] == 1 and mcp["wall"] == 18 and mcp["net"] == 3


def test_records_waits_and_event_forms_are_equivalent():
    base = fork_graph()
    variants = []
    for style in ("events", "records"):
        payload = fork_graph()
        for node in payload["nodes"]:
            node.pop("records", None)
            node.pop("waits", None)
        if style == "events":
            payload["events"] = [dict(id="enter", record="fork", wait=["a", "b"]),
                                 dict(id="ae", record="a", wait="join"),
                                 dict(id="be", record="b", wait="join")]
        else:
            payload["records"] = [dict(id=e, node=n) for e, n in (("enter", "fork"), ("ae", "a"), ("be", "b"))]
            payload["waits"] = [dict(id=e, node=n) for e, n in (("enter", "a"), ("enter", "b"), ("ae", "join"), ("be", "join"))]
        variants.append(payload)
    for payload in variants:
        assert diagnose_region(payload).wall_us == diagnose_region(base).wall_us
        assert diagnose_region(payload).details["edges"] == diagnose_region(base).details["edges"]


@pytest.mark.parametrize("bad", ["region_key", "node_key", "missing_record", "orphan_record",
                                  "duplicate_record", "duplicate_wait", "duplicate_node", "dangling",
                                  "cycle", "stream_cycle", "event_key", "record_key", "wait_key",
                                  "dependency_key", "wrong_bool", "wrong_sequence"])
def test_invalid_graph_evidence_fails_closed(bad):
    g = fork_graph()
    if bad == "region_key": g["surprise"] = 1
    if bad == "node_key": g["nodes"][0]["surprise"] = 1
    if bad == "missing_record": g["nodes"][0]["records"] = []
    if bad == "orphan_record": g["nodes"][0]["records"].append("unused")
    if bad == "duplicate_record": g["nodes"][1]["records"].append("enter")
    if bad == "duplicate_wait": g["nodes"][1]["waits"].append("enter")
    if bad == "duplicate_node": g["nodes"].append(g["nodes"][0])
    if bad == "dangling": g["dependencies"] = [["absent", "a"]]
    if bad == "cycle": g["dependencies"] = [["a", "b"], ["b", "a"]]
    if bad == "stream_cycle": g["dependencies"] = [["join", "fork"]]
    if bad == "event_key": g["events"] = [dict(id="z", record="fork", wait="a", typo=0)]
    if bad == "record_key": g["records"] = [dict(id="z", node="fork", typo=0)]
    if bad == "wait_key": g["waits"] = [dict(id="z", node="a", typo=0)]
    if bad == "dependency_key": g["dependencies"] = [dict(source="a", target="b", typo=0)]
    if bad == "wrong_bool": g["piece_boundary_between"] = "false"
    if bad == "wrong_sequence": g["events"] = False
    with pytest.raises(ValueError):
        ParallelRegion.from_mapping(g)
    assert aclgraph_diagnose(g)["ok"] == 0


def test_dependency_serialization_and_release_intervals():
    g = dict(name="serial", intervals=[dict(id="a", stream_id=0, start_us=20, end_us=23),
                                       dict(id="b", stream_id=1, start_us=20, end_us=24),
                                       dict(id="c", stream_id=2, start_us=20, end_us=25)],
             dependencies=[["a", "b"], ["b", "c"]])
    result = diagnose_region(g)
    assert result.span_us == result.wall_us == 12
    assert result.net_speedup_us == result.overlap_pct == 0
    assert result.details["starts_us"] == {"a": 0, "b": 3, "c": 7}


def test_generated_dags_match_independent_longest_paths():
    rng = random.Random(902)
    for _ in range(60):
        nodes, edges, previous = [], set(), {}
        finish = {}
        for i in range(12):
            stream, duration = rng.randrange(3), rng.randrange(1, 10)
            preds = {j for j in range(i) if rng.random() < .12}
            if stream in previous: preds.add(previous[stream])
            previous[stream] = i
            edges.update((j, i) for j in preds)
            finish[i] = max((finish[j] for j in preds), default=0)+duration
            nodes.append(dict(id=str(i), stream_id=stream, duration_us=duration))
        result = diagnose_region(dict(name="random", nodes=nodes,
                                      dependencies=[[str(a), str(b)] for a, b in sorted(edges)]))
        assert result.wall_us == max(finish.values())
        assert result.net_speedup_us == sum(x["duration_us"] for x in nodes)-result.wall_us


def test_capture_root_exemption_and_nonroot_join():
    root = StreamInterval("root", 0, 0, 10, waits_on_capture_root=False, joined_at_exit=False)
    other = StreamInterval("other", 1, 0, 10)
    assert diagnose_region(ParallelRegion("r", root, other, capture_root_stream_id=0)).code != "A5"
    for key in ("waits_on_capture_root", "joined_at_exit"):
        assert diagnose_region(ParallelRegion("r", root, replace(other, **{key: False}),
                                              capture_root_stream_id=0)).code == "A5"
    g = fork_graph()
    g["nodes"][-1]["waits"].remove("be")
    g["nodes"][2].pop("records")
    assert diagnose_region(g).code == "A5"


def test_mixed_memory_preserves_known_and_unknown_scopes():
    a = StreamInterval("a", 0, 0, 10, read_bytes=10**9, write_bytes=10**9)
    b = StreamInterval("b", 1, 0, 10, io_bytes=64)
    service, detail = region_memory_service(a, b, ParallelRegion("r", a, b))
    assert detail["read_service_bytes"] == detail["write_service_bytes"] == 10**9
    assert detail["unknown_direction_bytes"] == 64
    assert service == pytest.approx(812.50004)


@pytest.mark.parametrize("shape,strides", [((20, 3), (8, 2048)), ((3, 20), (2048, 8))])
def test_nddma_outer_physical_axis(shape, strides):
    r = account_transfer(TransferRequest("nddma", nd_shape=shape, stride_bits=strides,
                                         bits_per_element=8, capacity_policy="full_pitch"), P)
    assert r.actual_read_bytes == 60
    assert r.address_span_bytes == 532
    assert r.raw_source_capacity_bytes == r.source_capacity_bytes == 768


@pytest.mark.parametrize("policy,capacity", [("minimum_span", 544), ("full_pitch", 768)])
def test_nz2dn_target_pitch(policy, capacity):
    r = account_transfer(TransferRequest("nz2dn", rows=3, cols=16, element_bytes=2,
                                         src_pool="L0C", dst_pool="GM", gm_strides=GmStrides(3, 32, 256),
                                         capacity_policy=policy), P)
    assert r.destination_capacity_bytes == capacity
    assert r.actual_write_bytes == 96


def test_replay_partition_four_abi_and_guard():
    f = build_flow("fused_recurrent_kda_replayssm", batch=1, heads=96, seq=1, partitions=4)
    assert f.params["count_bindings"]["r"] == 32
    assert f.params["count_bindings"]["v"] == 64
    assert sum(x.bytes_per_tile*x.slots for x in f.buffers) == 71936
    with pytest.raises(ValueError):
        evaluate_math(f.name, dict(f.params["count_bindings"], v=32))


@pytest.mark.parametrize("archetype", list(ExecutionArchetype))
def test_ssbuf_capacity_uses_kernel_ownership(archetype):
    plan = plan_buffer_pool((BufferSpec("s", BufferPool.SSBUF, 2048),), BufferPool.SSBUF, P)
    mix = archetype is ExecutionArchetype.CV_COUPLED_1C2V
    assert buffer_capacity(P, BufferPool.SSBUF, archetype=archetype) == (3072 if mix else 1024)
    if mix:
        assert check_pool_capacity(plan, P, archetype=archetype) == 1024
    else:
        with pytest.raises(ValueError): check_pool_capacity(plan, P, archetype=archetype)
    with pytest.raises(ValueError): check_pool_capacity(plan, P)


def test_pure_simd_ssbuf_and_lane_route():
    with pytest.raises(ValueError):
        evaluate_flow(vector(buffers=(BufferSpec("s", BufferPool.SSBUF, 2048),)), P)
    f = build_flow("flash_attn_score")
    with pytest.raises(ValueError):
        replace(f, lanes_receiving=(0,), dual_dst_ctl=0, fixpipe_calls=1)
    with pytest.raises(ValueError):
        evaluate_flow(replace(f, dual_dst_ctl=0, fixpipe_calls=1), P)
    total = sum(w.total_work for w in f.pipe_work if w.pipe == "VEC")
    wrong = replace(f, lanes_receiving=(1,), vector_ops_by_lane=(0, total), fixpipe_sub_block=0)
    with pytest.raises(ValueError): evaluate_flow(wrong, P)


@pytest.mark.parametrize("name", ["batch_matmul", "transpose_batch_matmul"])
def test_batched_matrices_repeat_m_and_traffic(name):
    a = build_flow(name, m=16, n=16, k=16, batch=1)
    b = build_flow(name, m=16, n=16, k=16, batch=8)
    assert b.params["group_sizes"] == [16]*8
    assert next(x.total_work for x in b.pipe_work if x.pipe == "CUBE") == 32768
    assert b.gm_read_bytes == a.gm_read_bytes*8
    assert b.gm_write_bytes == a.gm_write_bytes*8
    assert b.unique_gm_bytes == a.unique_gm_bytes*8
    grouped = build_flow("grouped_matmul", m=16, n=16, k=16, groups=8)
    assert sum(grouped.params["group_sizes"]) == 16


@pytest.mark.parametrize("name", ["attn_prologue", "flash_kda", "mega_mla"])
@pytest.mark.parametrize("allow", [None, 0, 1, "false", "true", False])
def test_all_prediction_entrypoints_require_explicit_boolean(name, allow):
    assert opflow_triton(name, allow_structural_prior=allow)["ok"] == 0
    assert opflow_evaluate(name, allow_structural_prior=allow)["ok"] == 0
    with pytest.raises((ValueError, TypeError)):
        evaluate_operator(name, P, allow_structural_prior=allow)
    with pytest.raises((ValueError, TypeError)):
        evaluate_flow(build_flow(name), P, allow_structural_prior=allow)


@pytest.mark.parametrize("name", ["attn_prologue", "flash_kda", "mega_mla"])
def test_explicit_prior_optin_succeeds(name):
    assert opflow_triton(name)["ok"] == 0
    assert opflow_triton(name, allow_structural_prior=True)["ok"] == 1
    assert not evaluate_operator(name, P, allow_structural_prior=True).calibrated


def test_all_builders_reject_unknown_dtype():
    for name in list_operators():
        with pytest.raises(ValueError): build_flow(name, dtype="not-a-dtype")
        assert opflow_evaluate(name, dtype="not-a-dtype", allow_structural_prior=True)["ok"] == 0


@pytest.mark.parametrize("name", ["quant_matmul", "dual_level_quant_matmul", "weight_quant_batchmatmul",
                                  "quant_batch_matmul_inplace_add", "flash_attn_score", "sparse_flash_mla"])
def test_fixed_dtype_rejects_unsupported_valid_dtype(name):
    with pytest.raises(ValueError): build_flow(name, dtype="fp32")


@pytest.mark.parametrize("text", ["supports only ge_graph", "only available in ge_graph", "ge_graph-only",
                                  "does not support aclgraph", "not supported in aclgraph", "仅在 ge_graph 下可用",
                                  "只支持 ge_graph", "仅限 ge_graph", "不支持 aclgraph", "TorchAir-only"])
def test_restrictive_legacy_text_preserved(text):
    r = normalize_text("SuperKernel " + text)
    assert r.preserved_legacy and r.text == r.source_text and not r.applied


def test_incomplete_hardware_is_typed_error():
    assert opflow_hardware("Ascend950PR_9579")["ok"] == 0
    assert opflow_hardware()["ok"] == 1


def test_floor_sum_value_and_iteration_bound():
    lines, start = inspect.getsourcelines(floor_sum)
    loop_line = start + next(i for i, line in enumerate(lines) if "qa, a = divmod" in line)
    for k in (10, 20, 40, 80, 160, 1000):
        a, m = 0, 1
        for _ in range(k): a, m = m, a+m
        for n in (0, 1, 2, 3, 5, 10, 100):
            counter = [0]
            def trace(frame, event, arg):
                if frame.f_code is floor_sum.__code__ and event == "line" and frame.f_lineno == loop_line:
                    counter[0] += 1
                return trace
            old = sys.gettrace()
            try:
                sys.settrace(trace)
                result = floor_sum(n, m, a, m-1)
            finally:
                sys.settrace(old)
            assert result == sum((a*i+m-1)//m for i in range(n))
            assert counter[0] <= 8*(min(n, m)+1).bit_length()
            if n <= 1: assert counter[0] == 0
    assert floor_sum(100, 2**10000, 2**10000-1, 2**10000-1) == 4950
    assert floor_sum(10**20, 13, 0, -14) == -2*10**20
    assert floor_sum(0, 1, -1, -1) == 0
