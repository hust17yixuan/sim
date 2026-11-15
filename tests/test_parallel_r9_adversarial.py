"""Round-9 adversarial regressions R9-01..R9-08.

[Role]
  Re-execute every Round-9 counterexample against its architectural root cause:
  prefill MoE dispatch plus combine emission, per-endpoint routed port
  admission, exact per-chunk prefill ledgers, all-axis draft service and strict
  draft geometry, the authoritative transfer page universe, additive per-rank
  EPLB lifetime, actual-payload prefill DCP qualification, and exact MC2 wire.
[Invariants]
  Assertions are conservation identities, physical lower bounds or typed
  rejections; no assertion reproduces an implementation intermediate.
[Timeouts & Locks]
  Pure in-memory arithmetic; no fixture, tmp_path, subprocess or device access.
[Side-Effect Policy]
  Builds immutable Cases only; writes nothing.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from test_parallel_r7_adversarial import (_dense, _draft, _link,
                                          _machine_with, _profile,
                                          _speculative)
from test_parallel_r8_adversarial import _facts, _moe_case, _sparse_case

from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.parallel_bridge import bridge_lifetime_plan, phase_bridge
from mindie_sim.parallel_context import dcp_merge_payloads
from mindie_sim.parallel_cuts import collective_wire
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_prefill_cost import prefill_step_ms
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_instances import priced_draft_step_ms
from mindie_sim.xpyd_pool_solver import logical_transfer_bytes, solve_xpyd_pools

MIB = 1024 * 1024


# R9-01: prefill emits and prices both executed MoE exchanges at the actual
# MoE-layer count; admission consumes the resulting per-request bytes.
def test_r9_01_prefill_moe_emits_dispatch_and_combine():
    case = _moe_case()
    top_k = int(case.model.top_k)
    hidden = int(case.model.hidden_size)
    layers = int(case.model.layers)
    step = prefill_step_ms(case, isl=256, prefix_hit_rate=0.0, batch=8)
    roles = step["comm_ledger"].role_payloads()
    tokens = float(step["tokens_global"])
    payload = tokens * top_k * hidden * 2
    assert roles["ep_dispatch"] == pytest.approx(payload * layers)
    assert roles["ep_combine"] == pytest.approx(payload * layers)
    assert "ep_mc2_dispatch_combine" not in roles
    row = _facts(case, "prefill", isl=256, osl=1, hit=0.0)
    expected_request = (2 * (2 - 1) * top_k * hidden * 2 * layers * 256)
    assert row["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(float(expected_request), rel=1e-12))
    report = solve_xpyd_pools(
        case, total_npus=15, isl=256, osl=1, prefix_hit_rate=0.0,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec(tp=1, dp=2, ep=2)],
        decode_configs=[ParallelSpec()], max_pool_instances=7,
        hccl_buffsize_mb=8)
    best = report["best_configuration"]
    assert best is not None and report["qualified"] is True
    lower_bound = float(expected_request) + report["kv_bytes_per_request"]
    assert best["admitted_rps"] * lower_bound <= 1_000_000.0 + 1e-6


# R9-02: routed handoff bytes are bounded by each physical destination port,
# never by an unweighted N * nic aggregate that a fast destination can borrow.
def test_r9_02_weighted_handoff_is_bound_by_each_destination_port():
    case = _dense(num_devices=4, max_seqs=1, kv_heads=1)
    link = replace(_link("one", range(4)),
                   endpoint_rates=tuple((index, 1000.0) for index in range(4)))
    dies = tuple(replace(die, compute_peaks=tuple(
        (key, value * 1e-9) for key, value in die.compute_peaks))
        if index == 3 else die for index, die in enumerate(case.machine.dies))
    case = replace(case, machine=replace(
        _machine_with(case, [link]), dies=dies))
    report = solve_xpyd_pools(
        case, total_npus=4, isl=1024, osl=1, prefix_hit_rate=0.0,
        ttft_slo_ms=1e15, tpot_slo_ms=1e15,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        max_pool_instances=3, hccl_buffsize_mb=8)
    row = next(item for item in report["rows"] if item["pd_ratio"] == "2P:2D")
    assert row["qualified"] is True
    assert row["decode_instance_share"][0] > 0.999
    fast = row["decode_device_ids"][0]
    demand = row["shared_cut_demands"][f"port:{fast}:ingress"]
    capacity = 1_000_000.0
    assert row["admitted_rps"] * demand <= capacity + 1e-6
    assert row["admitted_rps"] * demand >= 0.89 * capacity
    assert row["shared_cut_rps"] == pytest.approx(capacity / demand,
                                                  rel=1e-12)


# R9-03: the final partial chunk executes its own full-context index gather;
# no fractional ``q_tokens/chunk_tokens`` scaling of fixed gathers.
def test_r9_03_partial_final_chunk_keeps_every_full_context_gather():
    sparse = _sparse_case()
    case = replace(sparse, runtime=replace(sparse.runtime,
                                           max_num_batched_tokens=2))
    hit = 1021 / 1024
    step = prefill_step_ms(case, isl=1024, prefix_hit_rate=hit, batch=1)
    assert (step["q_tokens"], step["chunk_tokens"], step["steps"]) == (3, 2, 2)
    gathers = [operation for operation in step["comm_ledger"].operations
               if operation.payload_bytes == 131_072.0]
    assert len(gathers) == 2
    assert step["shared_cut_demands"]["switch:intra_node_hccs"] >= 2 * 131_072
    row = _facts(case, "prefill", isl=1024, osl=1, hit=hit)
    assert row["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(step["shared_cut_demands"]["switch:intra_node_hccs"],
                      rel=1e-12))
    report = solve_xpyd_pools(
        case, total_npus=15, isl=1024, osl=1, prefix_hit_rate=hit,
        destination_cached_tokens=1021, prefix_digest="d" * 64,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec(pcp=2)],
        decode_configs=[ParallelSpec()], max_pool_instances=7,
        hccl_buffsize_mb=8)
    best = report["best_configuration"]
    assert best is not None and report["qualified"] is True
    lower_bound = 2 * 131_072 + report["kv_bytes_per_request"]
    assert best["admitted_rps"] * lower_bound <= 1_000_000.0 + 1e-6


# R9-04: every executed draft axis contributes service time, and an incomplete
# public draft geometry fails with a typed missing-geometry result.
def test_r9_04_draft_pcp_service_and_dispatch_combine_are_priced():
    case = _dense(layers=1, hidden=256, heads=2, kv_heads=1, pcp=2,
                  num_devices=3, max_seqs=1)
    case = replace(case, machine=_machine_with(
        case, [_link("one", range(3), switch=1000.0)]))
    case = _speculative(case, _draft(), k=3, cost=0)
    priced = priced_draft_step_ms(case, _draft(), batch=1,
                                  context_tokens=257)
    assert priced is not None and priced.route_qualified is True
    assert priced.step_ms >= 2.048 - 1e-9
    report = solve_xpyd_pools(
        case, total_npus=3, isl=256, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=256, prefix_digest="d" * 64,
        ttft_slo_ms=1e9, tpot_slo_ms=5.0, prefill_configs=[ParallelSpec()],
        decode_configs=[ParallelSpec(pcp=2)], max_pool_instances=1,
        hccl_buffsize_mb=8)
    assert report["qualified"] is False
    # EP dispatch plus combine are two sequential services on the same switch.
    moe = _dense(experts=32, top_k=4, moe_inter=256, tp=1, dp=2, ep=2,
                 num_devices=2, max_seqs=8)
    moe = replace(moe, machine=_machine_with(
        moe, [_link("one", range(2), switch=1.0)]))
    draft = replace(_draft(), num_experts=8, top_k=2,
                    expert_intermediate_size=2048)
    moe = _speculative(moe, draft, k=1, cost=0)
    ep_priced = priced_draft_step_ms(moe, draft, batch=1, context_tokens=257)
    assert ep_priced is not None and ep_priced.route_qualified is True
    # 1024 hidden * 2 B * top-k 2 = 4,096 B per call on a 1 B/ms switch.
    assert ep_priced.step_ms >= 2 * 4_096.0


def test_r9_04_incomplete_public_draft_geometry_is_typed():
    for changes in ({"head_dim": 0},
                    {"num_experts": 32, "top_k": 4,
                     "expert_intermediate_size": 0},
                    {"attention_kind": "mla", "kv_lora_rank": 0,
                     "qk_nope_head_dim": 0, "qk_rope_head_dim": 0,
                     "v_head_dim": 0}):
        draft = replace(_draft(), **changes)
        case = _speculative(_dense(tp=1, pcp=2, num_devices=4, max_seqs=8),
                            draft)
        priced = priced_draft_step_ms(case, draft, batch=8,
                                      context_tokens=272)
        assert priced is not None
        assert priced.step_ms == 0.0
        assert priced.route_qualified is False
        assert priced.route_evidence.startswith("missing_draft_geometry:")
        assert priced.comm_demands == {}


# R9-05: transfer['page_ids'] remains the required universe; ownership evidence
# is reconciled against it and cached destination pages carry no bytes.
def test_r9_05_required_page_universe_is_authoritative():
    case = _dense(num_devices=4, max_seqs=1)
    common = dict(total_npus=3, isl=1024, osl=1, ttft_slo_ms=1e9,
                  tpot_slo_ms=1e9, prefill_configs=[ParallelSpec(pcp=2)],
                  decode_configs=[ParallelSpec()], hccl_buffsize_mb=8,
                  max_pool_instances=1, prefix_hit_rate=0.0)
    partial = solve_xpyd_pools(
        case, destination_cached_tokens=0, destination_page_ids=(0,), **common)
    assert partial["qualified"] is False
    assert partial["best_configuration"] is None
    assert "destination_incomplete" in str(partial["reason"])
    assert tuple(partial["kv_transfer"]["page_ids"]) == tuple(range(8))
    good = solve_xpyd_pools(
        case, destination_cached_tokens=129, prefix_digest="d" * 64,
        destination_page_ids=tuple(range(8)),
        **{**common, "prefix_hit_rate": 129 / 1024})
    assert good["qualified"] is True
    bridge = good["best_configuration"]["remap_bridge"]
    owned = [page for bucket in bridge["destination_pages"] for page in bucket]
    assert 0 not in owned and set(owned) == set(range(1, 8))
    assert sum(edge[2] for edge in bridge["edge_bytes"]) == pytest.approx(
        good["kv_transfer"]["paged_kv_bytes"], rel=1e-12)
    both = solve_xpyd_pools(
        case, destination_cached_tokens=129, prefix_digest="d" * 64,
        source_page_ids=tuple(range(1, 8)),
        destination_page_ids=tuple(range(8)),
        **{**common, "prefix_hit_rate": 129 / 1024})
    assert both["qualified"] is True
    assert tuple(both["best_configuration"]["remap_bridge"]["edge_bytes"]) == (
        tuple(bridge["edge_bytes"]))


# R9-06: the EPLB peak is a per-rank bound (not divided by PP) and is additive
# to simultaneously live handoff/retention storage.
def test_r9_06_eplb_peak_is_additive_and_not_divided_by_pp():
    case = _dense(pp=2, num_devices=4, max_seqs=1)
    lifetime = bridge_lifetime_plan(
        phase_bridge(ParallelSpec(pp=2), ParallelSpec(pp=2),
                     kv_bytes=0, page_ids=()),
        eplb_rebalance_peak_bytes=64 * MIB)
    row = _facts(case, "decode", isl=256, osl=1, hit=0.0,
                 transfer_plan=lifetime)
    base = _facts(case, "decode", isl=256, osl=1, hit=0.0)
    increments = [after[3] - before[3] for after, before in
                  zip(row["memory"].per_rank_bytes,
                      base["memory"].per_rank_bytes)]
    assert lifetime.phase_peak_bytes("decode") >= 64 * MIB
    assert all(value >= 64 * MIB - 1.0 for value in increments)
    # A tight per-rank budget below the declared EPLB peak cannot qualify.
    tight = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=math.ceil(
            base["memory"].per_rank_bytes[index % 2][3] + 48 * MIB) / 0.9)
        for index, die in enumerate(case.machine.dies))))
    report = solve_xpyd_pools(
        tight, total_npus=4, isl=256, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=256, prefix_digest="d" * 64,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec(pp=2)],
        decode_configs=[ParallelSpec(pp=2)], max_pool_instances=1,
        hccl_buffsize_mb=8, eplb_rebalance_peak_bytes=64 * MIB)
    assert report["qualified"] is False


def test_r9_06_simultaneous_eplb_is_summed_with_handoff_and_retention():
    case = _dense(tp=1, kv_heads=1, num_devices=2, max_seqs=1)
    case = replace(case, machine=_machine_with(case, [_link("one", range(2))]))
    transfer = logical_transfer_bytes(case, prompt_tokens=1024)
    lifetime = bridge_lifetime_plan(
        phase_bridge(ParallelSpec(), ParallelSpec(),
                     kv_bytes=transfer["paged_kv_bytes"],
                     page_ids=transfer["page_ids"]),
        eplb_rebalance_peak_bytes=64 * MIB)
    required = (lifetime.handoff_staging_bytes
                + lifetime.destination_retention_bytes
                + lifetime.eplb_rebalance_peak_bytes)
    assert lifetime.phase_peak_bytes("decode") >= required - 1e-9
    base = analyze_memory_breakdown(case, sequences=1, sequence_tokens=1025,
                                    hccl_buffsize_mb=8)
    budget = base.peak_hbm_bytes + 66 * MIB
    tight = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=math.ceil(budget / 0.9))
        for die in case.machine.dies)))
    row = _facts(tight, "decode", isl=1024, osl=1, hit=0.0,
                 transfer_plan=lifetime)
    assert row["feasible"] is False
    assert row["memory"].headroom_bytes < 0.0
    report = solve_xpyd_pools(
        tight, total_npus=2, isl=1024, osl=1, ttft_slo_ms=1e9,
        tpot_slo_ms=1e9, prefill_configs=[ParallelSpec()],
        decode_configs=[ParallelSpec()], hccl_buffsize_mb=8,
        max_pool_instances=1, eplb_rebalance_peak_bytes=64 * MIB)
    assert report["qualified"] is False


# R9-07: the emitted prefill DCP row is priced and qualified at its actual
# payload; a Mode-B unmeasured anchor cannot inherit TP's measured route.
def test_r9_07_prefill_dcp_payload_is_priced_and_qualified():
    case = _dense(tp=2, cp=2, kv_heads=1, num_devices=2, max_seqs=1)
    model = DecodeCostModel.from_case(case)
    payload = dcp_merge_payloads(2, 2, 1, model.query_heads, model.head_dim,
                                 model.element_bytes)["total_bytes"]
    anchors = tuple(sorted(set((0, 1, 100, payload - 1, payload, payload + 1,
                                4096, 8192, 1048576, 1 << 30))))
    tree = _profile(anchors, unmeasured=payload)
    direct = tree.collective("all_reduce", payload, 2, endpoints=(0, 1))
    assert direct.route_qualified is False
    step = prefill_step_ms(case, isl=256, prefix_hit_rate=0.0, batch=1,
                           fabric=tree)
    assert step["route_qualified"] is False
    assert step["route_evidence"] == "unmeasured_profile"
    dcp_ops = [operation for operation in step["comm_ledger"].operations
               if operation.payload_bytes == payload and operation.role == "dcp"]
    assert len(dcp_ops) == 1 and dcp_ops[0].count == 4
    row = _facts(case, "prefill", isl=256, osl=1, hit=0.0, tree=tree)
    assert row["feasible"] is False
    assert row["route_qualified"] is False
    assert row["route_evidence"] == "unmeasured_profile"


# R9-08: MC2 is exactly one dispatch plus one combine, never four exchanges.
def test_r9_08_mc2_wire_equals_dispatch_plus_combine():
    for group in (2, 3, 4, 8):
        for payload in (1.0, 131_072.0, 1 << 20):
            assert collective_wire("mc2", group, payload) == pytest.approx(
                2.0 * collective_wire("all_to_all_v", group, payload),
                rel=1e-12)
    case = replace(_moe_case(), runtime=replace(
        _moe_case().runtime, moe_recipe="mc2"))
    mc2 = DecodeCostModel.from_case(case, context_tokens=272).step(
        4, per_replica_batch=4, active_replicas=2, context=272)
    normal = DecodeCostModel.from_case(_moe_case(),
                                       context_tokens=272).step(
        4, per_replica_batch=4, active_replicas=2, context=272)
    assert mc2.comm_ledger.cut_demands() == normal.comm_ledger.cut_demands()
