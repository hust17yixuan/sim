"""Exact pipeline, lane, wave, cache, and MFU scope contracts from v4.

[Role]
Distinguish finite schedules from cycle bounds and independent queues from
barrier waves, and validate public flow/lowering integration.
[Invariants]
All math-pack expectations use integer/Fraction equality. No universal core
monotonicity assertion; cache fills and unique bytes have distinct scopes.
[Timeouts & Locks]
Bounded deterministic arithmetic, no waits or locks.
[Side-Effect Policy]
Read fixtures only; no files or devices are modified.
"""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from mindie_sim.opflow import (
    ExecutionArchetype, FixpipePath, FlowSchemaError, OpFlowTemplate, PipeWork,
    TritonFeatures, build_flow, evaluate_flow, evaluate_mfu_gates, get_profile,
)
from mindie_sim.opflow.exact_math import (
    MathDomainError, cache_service, lane_service, memory_seconds, one_c_two_v, replay_u_promotion,
)
from mindie_sim.opflow.schedule_math import cycle_bound, independent_makespan, pipeline_chain, wave_makespan
from mindie_sim.opflow.triton_lowering import TritonLoweringError

V = json.loads((Path(__file__).parent / "fixtures/v4_exact_vectors.json").read_text())["exact_test_tables"]
P = get_profile("Ascend950PR")


@pytest.mark.parametrize("v", V["pipeline_vectors"], ids=lambda v: str(v["slots"]))
def test_finite_pipeline_vectors(v):
    result = pipeline_chain((100,100,100),v["N"],v["slots"])
    assert result == {key:value for key,value in v.items() if key not in ("N","slots")}
    assert all(isinstance(x,Fraction) for x in result.values())
    flow = OpFlowTemplate("chain",ExecutionArchetype.PURE_CUBE,4,1,0,
        (PipeWork("MTE1",P.rates.mte1_bytes_s*4/10000,"bytes"),
         PipeWork("CUBE",P.rates.cube_macs_s*4/10000,"macs"),
         PipeWork("FIX",P.rates.fix_bytes_s*4/10000,"bytes")),pipeline_slots=v["slots"])
    public = evaluate_flow(flow,P,active_units=1)
    assert public.t_first_item_us == v["first_item"]
    assert public.t_unit_us == public.t_est_us == v["exact_makespan"]


def test_recurrence_fraction_and_serial_fallback():
    assert pipeline_chain((1,1,1),4,1)["exact_makespan"] == 12
    assert pipeline_chain((Fraction(1,7),Fraction(2,7)),100,3)["exact_makespan"] == Fraction(201,7)
    assert pipeline_chain((1,),4,recurrence_latency=10,recurrence_distance=2)["exact_makespan"] == 12
    assert cycle_bound((10,20),(0,2)) == Fraction(15)
    with pytest.raises(MathDomainError):
        cycle_bound((0,),(0,))
    assert max(V["resource_vector"].values()) == 250


@pytest.mark.parametrize("v",V["wave_vectors"])
def test_independent_tiles_keep_full_tile_cost(v):
    count, remainder = divmod(v["hidden"],8192)
    sizes = [8192]*count + ([remainder] if remainder else [])
    assert len(sizes) == v["tiles"]
    assert (count,remainder) == (v["full_tiles"],v["remainder_elements"])
    assert independent_makespan(sizes,v["U"]) == v["exact_independent_tile_makespan"]


def test_barrier_counterexample_and_variable_sequences():
    v = V["core_counterexample"]
    durations = [v["local_time"]]*v["Mtiles"]
    assert [wave_makespan(durations,u,v["shared_time_per_tile"]) for u in v["cores"]] == v["barrier_makespans"]
    assert [independent_makespan(durations,u) for u in v["cores"]] == v["independent_makespans"]
    v = V["multiwave_sequence_vector"]
    assert wave_makespan([s*v["step_time"] for s in v["S"]],v["U"]) == v["exact_time"]


@pytest.mark.parametrize("v",V["memory_vectors"])
def test_memory_contention_exact_rational(v):
    alpha = Fraction(**v["alpha"]) if isinstance(v["alpha"],dict) else Fraction(v["alpha"])
    result = memory_seconds(v["read"],v["write"],v["read_bps"],v["write_bps"],alpha)
    expected = Fraction(**v["seconds"]) if isinstance(v["seconds"],dict) else Fraction(v["seconds"])
    assert result == expected


def test_one_c_two_v_conservation_and_legality():
    # v4 names this invariant in check_counts, not as a standalone vector.
    assert V["check_counts"]["one_C_two_V_conservation"] == 1
    assert lane_service((30,70),(10,10)) == 7
    assert one_c_two_v({"work":100,"read":64,"write":0},
                       ({"work":30,"read":10,"write":16},{"work":70,"read":22,"write":48})) == {
                           "work":200,"read":96,"write":64}
    assert one_c_two_v({"work":1,"read":32,"write":0},
                       ({"work":3,"read":32,"write":32},{"work":5,"read":32,"write":32})) == {
                           "work":9,"read":96,"write":64}
    base = build_flow("mhc_hcpre_cv_splitk")
    total = sum(w.total_work for w in base.pipe_work if w.pipe == "VEC")
    split = replace(base,vector_ops_by_lane=(total/4,total*3/4))
    result = evaluate_flow(split,P,allow_structural_prior=True)
    assert sum(result.extra["lane_vector_ops"]) == total
    assert result.pipe_us["VEC"] == total*3/4/P.rate_for("VEC",split.dtype)/split.total_units*1e6
    for mode,m,n in ((1,16,16),(2,16,32),(0,16,16)):
        routed = replace(base,dual_dst_ctl=mode,fixpipe_m=m,fixpipe_n=n,fixpipe_calls=2)
        assert evaluate_flow(routed,P,allow_structural_prior=True).extra["vector_lanes"] == 2
    for changes in ({"no_launch":"false"},{"calibrated":True},{"fixpipe_sub_block":True},{"dual_dst_ctl":True},{"dual_dst_ctl":1,"fixpipe_m":3},
                    {"dual_dst_ctl":2,"fixpipe_n":16},
                    {"dual_dst_ctl":2,"fixpipe_n":32,"relu_enabled":True},
                    {"dual_dst_ctl":1,"fixpipe_m":16,"fixpipe_layout":"NZ"},
                    {"dual_dst_ctl":1,"fixpipe_m":16,"lanes_receiving":(0,)},
                    {"dual_dst_ctl":0,"fixpipe_calls":1}):
        with pytest.raises(FlowSchemaError):
            evaluate_flow(replace(base,**changes),P,allow_structural_prior=True)


def test_cache_service_and_u_promotion():
    assert cache_service(64,4,1) == V["cache_vector"]
    v = V["U_promotion_vector"]
    assert replay_u_promotion(v["N"],v["A"],v["D"]) == {k:v[k] for k in ("read","write","casts")}
    with pytest.raises(ValueError):
        build_flow("commit_recurrent_kda",u_dtype="fp16")


def test_mfu_undefined_and_exact_whole_flow_scope():
    zero = evaluate_mfu_gates(P,wall_us=1,features=TritonFeatures(flops=0),working_set_bytes=64)
    assert zero.mfu is None and zero.as_dict()["mfu"] is None and not zero.valid
    impossible = evaluate_mfu_gates(P,wall_us=1,features=TritonFeatures(flops=10**15),working_set_bytes=64)
    assert impossible.mfu is None and not impossible.valid
    flow = build_flow("matmul_bf16",m=16,n=16,k=16)
    r = evaluate_flow(flow,P,triton_features={"has_cv_pingpong":True})
    flops = sum(2*w.total_work for w in flow.pipe_work if w.pipe=="CUBE")
    report = r.extra["triton_detail"].mfu
    assert report.mfu == (Fraction(str(flops))*10**6/Fraction(str(r.t_est_us))
                          /(Fraction(str(P.cube_peak_tflops(flow.dtype)))*10**12))
    with pytest.raises(TritonLoweringError):
        evaluate_flow(flow,P,triton_features={"flops":flops+1})
