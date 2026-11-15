"""Round-2 adversarial regressions R13..R22 and witnesses W4..W7.

[Role] Encode the transfer/fabric/MTP/context contracts and the executed
  Round-2 witnesses W4..W7 as permanent physical assertions.
[Invariants] Assertions are conservation identities, physical inequalities or
  typed rejection outcomes; no golden latency and no string-grep filler.
[Timeouts & Locks] Pure in-memory; no subprocess, network, sleep or device.
[Side-Effect Policy] Builds immutable Cases; writes nothing.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim.fabric_topology import (CalibratedFabricProfile, FabricTier,
                                       HierarchicalFabricTree, scale_hardware)
from mindie_sim.memory_breakdown import analyze_memory_breakdown, memory_ledger
from mindie_sim.moe_skew_eplb import (eplb_effect, eta_skew, expert_demand,
                                      max_rank_tokens, mtp_batch_reversal,
                                      place_eplb, rank_load_cv,
                                      replicated_expert_bytes,
                                      speculative_accept_tokens)
from mindie_sim.parallel_context import (dcp_merge_payloads, pair_conservation,
                                        plan_context_parallel, valid_pairs)
from mindie_sim.parallel_ep_knee import (DecodeCostModel, prefill_step_ms,
                                        settle_optima, sweep_ep_knee,
                                        sweep_tp_ep_tradeoff)
from mindie_sim.parallel_placement import (build_placement, lane_remap_bytes,
                                          validate_placement, with_world)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_pool_solver import (Flow, completed_requests_per_s,
                                        kv_transfer_contention, logical_transfer_bytes,
                                        pool_config_facts, progressive_max_min,
                                        solve_xpyd_pools)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for name, sub in (("HOME", "home"), ("MINDIE_CONFIG_HOME", "config"),
                      ("MINDIE_DATA_HOME", "data"), ("MINDIE_CACHE_HOME", "cache"),
                      ("MINDIE_DIAGNOSTICS_ROOT", "diagnostics")):
        monkeypatch.setenv(name, str(root / sub))


def _case(*, sku="Ascend910_9362", topology="a3_16p", num_devices=64,
          ranks_per_node=16, layers=8, hidden=2048, heads=16, kv_heads=4,
          experts=32, top_k=4, moe_inter=512, tp=2, dp=2, ep=2, pp=1,
          pcp=1, cp=1, max_seqs=64, max_model_len=8192, graph="none",
          inter_node=None):
    machine = {"backend": "ascend", "sku": sku, "num_devices": num_devices,
               "ranks_per_node": ranks_per_node, "topology_preset": topology}
    if inter_node is not None:
        machine.update(inter_node_bw_gbps=inter_node, inter_node_bw_util=1.0,
                       inter_node_latency_us=10.0)
    return parse_case({
        "machine": machine,
        "model": {"name": "adversarial", "layers": layers, "hidden_size": hidden,
                  "num_heads": heads, "num_kv_heads": kv_heads, "head_dim": 128,
                  "intermediate_size": hidden * 2, "vocab_size": 32000,
                  "attention": "gqa", "ffn": "moe" if experts else "dense",
                  "num_experts": experts, "top_k": top_k,
                  "moe_intermediate_size": moe_inter,
                  "activation_dtype": "bf16", "weight_dtype": "bf16",
                  "kv_dtype": "bf16"},
        "runtime": {"tp": tp, "dp": dp, "ep": ep, "pp": pp, "pcp": pcp,
                    "cp": cp, "max_model_len": max_model_len,
                    "max_num_seqs": max_seqs, "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.9, "kv_block_size": 128,
                    "graph_capture_mode": graph},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]},
    })


def _sparse_case(*, families=("dsa",), pcp=2, cp=1):
    layers = len(families)
    plans = []
    for index, family in enumerate(families):
        plans.append({
            "index": index, "attention": "mla", "ffn": "dense",
            "architecture": f"mla_{family}", "indexer": family in {"dsa", "csa", "csa2", "hca"},
            "indexer_type": "c4" if family in {"dsa", "csa", "csa2", "hca"} else "none",
            "index_topk": 512 if family in {"dsa", "csa", "csa2", "hca"} else 0,
            "owns_index_k": family in {"dsa", "csa", "csa2", "hca"},
            "index_dim": 128 if family in {"dsa", "csa", "csa2", "hca"} else 0,
            "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "kv_lora_rank": 512, "v_head_dim": 128,
            "compress_ratio": 4 if family in {"csa", "csa2"} else 0,
            "window": 512 if family == "swa" else 4096,
            "cache_kind": "window_compressed" if family in {"csa", "csa2"} else "dense",
        })
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 32, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "sparse", "layers": layers, "hidden_size": 2048,
                  "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
                  "intermediate_size": 4096, "vocab_size": 32000,
                  "attention": "mla", "ffn": "dense", "layer_plans": plans},
        "runtime": {"tp": 8, "dp": 2, "ep": 1, "pp": 1, "pcp": pcp, "cp": cp,
                    "max_model_len": 8192, "max_num_seqs": 16,
                    "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.9,
                    "kv_block_size": 128},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]},
    })


def _tree() -> HierarchicalFabricTree:
    return HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 8, 1.5, 392.0),
        FabricTier("superpod_hccs", 64, 3.5, 800.0),
    ))


def _components(record):
    return {item.name: item.bytes for item in record.components}


# R13 -----------------------------------------------------------------------
def test_r13_transfer_bytes_are_unique_and_order_independent():
    import dataclasses
    from mindie_sim.parallel_placement import with_world
    case = _case(tp=4, dp=2, ep=1, pp=1, experts=0, top_k=0, moe_inter=0,
                 max_model_len=8192)
    first = logical_transfer_bytes(case, prompt_tokens=4096, output_tokens=16)
    reordered = with_world(case, ParallelSpec(dp=2, tp=4, ep=1, pp=1))
    second = logical_transfer_bytes(reordered, prompt_tokens=4096, output_tokens=16)
    assert first["total_bytes"] == second["total_bytes"]
    pp2 = with_world(_case(tp=2, dp=2, ep=1, pp=2, layers=8, experts=0,
                          top_k=0, moe_inter=0, max_model_len=8192),
                     ParallelSpec(dp=2, tp=2, ep=1, pp=2))
    third = logical_transfer_bytes(pp2, prompt_tokens=4096, output_tokens=16)
    assert third["total_bytes"] == first["total_bytes"]
    sparse = logical_transfer_bytes(_sparse_case(families=("dsa",)),
                                    prompt_tokens=2048, output_tokens=16)
    assert sparse["index_k_bytes"] > 0.0
    assert sparse["unique_pages"] * sparse["block_size"] >= 2048


# R14 -----------------------------------------------------------------------
def test_r14_endpoint_topology_distinguishes_paths():
    a3 = _case()
    tree = HierarchicalFabricTree.from_hardware(a3.machine)
    same_package = tree.collective("all_reduce", 1024, 2, endpoints=(0, 1))
    cross_package = tree.collective("all_reduce", 1024, 2, endpoints=(0, 2))
    assert same_package.endpoint_path == ("intra_package_die",)
    assert cross_package.endpoint_path == ("intra_node_hccs",)
    assert same_package.total_us < cross_package.total_us
    a5 = _case(sku="Ascend950DT_9582", topology="a5_pod", num_devices=64,
               ranks_per_node=64)
    a5_tree = HierarchicalFabricTree.from_hardware(a5.machine)
    die_route = a5_tree.collective("all_reduce", 1024, 2, endpoints=(0, 1))
    assert not die_route.route_qualified
    assert "unmeasured_die_link" in die_route.evidence
    with pytest.raises(ModelingError):
        tree.collective("all_reduce", 1024, 2, endpoints=(0, 99))


# R15 -----------------------------------------------------------------------
def _hierarchy():
    return HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 2, 0.0, 1e9),
        FabricTier("pod_roce", 16, 0.0, 1e9),
    ))


def test_r15_uneven_groups_ring_rounds_and_startup():
    tree = _hierarchy()
    for ranks in (3, 5, 6):
        for kind in ("all_reduce", "all_gather", "all_to_all_v"):
            record = tree.collective(kind, 960, ranks)
            flat = (2 * 960 * (ranks - 1) / ranks if kind == "all_reduce"
                    else 960 * (ranks - 1) / ranks)
            assert record.wire_bytes_per_rank <= flat + 1e-9
            assert record.wire_bytes_per_rank == pytest.approx(flat, rel=1e-9)
    flat8 = HierarchicalFabricTree((FabricTier("intra_node_hccs", 8, 1.0, 1e9),))
    ring = flat8.collective("all_reduce", 8, 8)
    assert dict(ring.rounds)["intra_node_hccs"] == 14
    assert ring.total_us == pytest.approx(14.0, rel=1e-9)
    zero = flat8.collective("all_reduce", 0, 8)
    assert zero.total_us == pytest.approx(14.0, rel=1e-9)
    assert zero.wire_bytes_per_rank == 0.0
    skewed = tree.collective("all_to_all_v", 960, 4,
                             endpoints=(0, 1, 2, 3),
                             payloads_by_destination=(600.0, 100.0, 100.0, 160.0))
    assert skewed.wire_bytes_per_rank == pytest.approx(860.0)


# R16 -----------------------------------------------------------------------
def test_r16_mode_b_range_policy_and_effective_load():
    from mindie_sim.fabric_topology import FabricRangeError, ProbePoint
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 16, "points": [
            {"payload_bytes": 4096, "alpha_us": 3.0, "beta_ns_per_byte": 0.02},
            {"payload_bytes": 1 << 20, "alpha_us": 5.0, "beta_ns_per_byte": 0.03}]}]})
    with pytest.raises(FabricRangeError):
        profile.collective("all_reduce", 1, 8)
    with pytest.raises(FabricRangeError):
        profile.collective("all_reduce", 1 << 30, 8)
    inside = profile.collective("all_reduce", 4096, 8)
    assert inside.total_us > 0.0
    outside = profile.collective("all_reduce", 1 << 30, 8,
                                 extrapolation_policy="extrapolate")
    assert outside.total_us > inside.total_us
    point = profile.tiers[0].at(1 << 30, policy="extrapolate")
    assert point.extrapolated
    with pytest.raises(ModelingError):
        ProbePoint(4096, 1.0, 0.02, congestion_factor=1.2,
                   load_basis="effective_under_load")
    with pytest.raises(ModelingError):
        CalibratedFabricProfile.from_probe({"measurements": [
            {"ranks": 2, "payload_bytes": 4096, "alpha_us": 1.0,
             "beta_ns_per_byte": 0.02}]})


# R17 -----------------------------------------------------------------------
def test_r17_real_mrv2_sparse_builders_and_pair_conservation():
    sparse = _sparse_case(families=("dsa", "swa", "csa", "csa2", "hca"),
                          pcp=2, cp=2)
    parallel = ParallelSpec(dp=2, tp=8, ep=1, pp=1, cp=2, pcp=2, cp_plan="pcp")
    plan = plan_context_parallel(sparse.model, parallel, mrv2_pcp=True,
                                 isl=1024, context_tokens=4096)
    assert plan.compatible, plan.reason
    assert {row["family"] for row in plan.builders} == {"dsa", "swa", "csa",
                                                       "csa2", "hca"}
    assert plan.prefill_collectives[:2] == ("pcp_index_k_gather", "_pcp_kv_gather")
    assert dict(plan.index_gather)["pcp_index_k_gather"] > 0.0
    assert dict(plan.kv_gather)["_pcp_kv_gather"] > 0.0
    for family, row in dict(plan.pair_conservation).items():
        assert row["conserved"] and sum(row["per_rank_pairs"]) == row["total_pairs"]
    reductions = dcp_merge_payloads(8, 2, 1024, 16, 128, 2.0)
    assert reductions["numerator_bytes"] > reductions["max_bytes"] > 0.0
    assert plan.bridge is not None and plan.bridge.barrier
    dense = _case(experts=0, top_k=0, moe_inter=0, ep=1)
    spoof = plan_context_parallel(
        dense.model, ParallelSpec(dp=1, tp=8, ep=1, pcp=1, cp=1),
        mrv2_pcp=True, attention_families=("dsa", "swa"))
    assert not spoof.compatible


# R18 -----------------------------------------------------------------------
def test_r18_feasible_optima_and_all_infeasible_state():
    case = _case(tp=1, dp=2, ep=2, layers=4, experts=32, top_k=2,
                 max_model_len=4096)
    report = sweep_ep_knee(case, ep_candidates=(2, 4, 8, 16, 32),
                           batch_sequences=1, context_tokens=128,
                           hccl_buffsize_mb=1024, mc2=True)
    for key in ("ep_star_latency", "ep_star_throughput", "ep_star_efficiency"):
        value = report[key]
        if value is not None:
            row = next(item for item in report["rows"] if item["ep"] == value)
            assert row["feasible"]
    assert report["ep_star"] == report["ep_star_latency"]
    tiny = sweep_ep_knee(case, ep_candidates=(2, 4, 8),
                         batch_sequences=1, context_tokens=128,
                         hccl_buffsize_mb=1e12, mc2=True)
    assert tiny["ep_star"] is None and tiny["status"] == "all_candidates_infeasible"
    assert tiny["ep_star_latency"] is None and tiny["knees"] == ()


# R19 -----------------------------------------------------------------------
def test_r19_mtp_cycle_equality_and_all_crossings():
    equal = mtp_batch_reversal(lambda batch: float(batch), draft_len=1,
                               accept_rate=1.0, max_batch=4)
    assert equal.crossings == (1, 2, 3, 4) and equal.b_reversal == 1
    costs = {1: 10, 2: 30, 3: 10, 4: 10, 6: 30, 8: 10}
    multiple = mtp_batch_reversal(lambda batch: float(costs[batch]), draft_len=1,
                                  accept_rate=1.0, max_batch=4)
    assert multiple.crossings == (1, 3)
    assert {point[0] for point in multiple.curve} >= {1, 3}
    unqualified = mtp_batch_reversal(lambda batch: float(batch), draft_len=1,
                                     accept_rate=1.0, max_batch=4,
                                     draft_rollback_bytes_per_rank=100,
                                     memory_budget_bytes=50)
    assert not unqualified.memory_qualified


# R20 -----------------------------------------------------------------------
def test_r20_effective_bandwidth_units_and_mc2_recompute():
    tree = HierarchicalFabricTree((FabricTier("intra_node_hccs", 2, 0.0, 1.0),))
    one = tree.collective("all_reduce", 1000, 2)
    assert one.total_us == pytest.approx(1.0, rel=1e-12)
    assert one.effective_bw_gbps == pytest.approx(1.0, rel=1e-12)
    base = tree.collective("all_to_all_v", 1000, 2)
    mc2 = tree.collective("mc2", 1000, 2)
    assert mc2.total_us == pytest.approx(2.0 * base.total_us, rel=1e-12)
    assert mc2.effective_bw_gbps == pytest.approx(base.effective_bw_gbps / 2.0,
                                                  rel=1e-12)


# R21 -----------------------------------------------------------------------
def test_r21_link_scaling_updates_topology_options_and_floors():
    a5 = _case(sku="Ascend950DT_9582", topology="a5_pod", num_devices=64,
               ranks_per_node=64)
    scaled = scale_hardware(a5, link_bw=2.0)
    options = dict(scaled.machine.topology_options)
    original = dict(a5.machine.topology_options)
    assert options["superpod_bw_gbps"] == pytest.approx(2 * original["superpod_bw_gbps"])
    assert options["inter_pod_bw_gbps"] == pytest.approx(2 * original["inter_pod_bw_gbps"])
    before = HierarchicalFabricTree.from_hardware(a5.machine)
    after = HierarchicalFabricTree.from_hardware(scaled.machine)
    for old, new in zip(before.tiers, after.tiers):
        if old.name in {"superpod_hccs", "pod_roce", "intra_node_hccs"}:
            assert new.bw_gbps == pytest.approx(2.0 * old.bw_gbps)
    floored = scale_hardware(a5, hbm_bytes=1.0000001)
    assert floored.machine.dies[0].memory_bytes == int(
        math.floor(a5.machine.dies[0].memory_bytes * 1.0000001))
    a3 = _case(inter_node=6.25)
    a3_scaled = scale_hardware(a3, link_bw=2.0)
    assert dict(a3_scaled.machine.topology_options)["intra_bw_4p_gbps"] == pytest.approx(784.0)


# R22 -----------------------------------------------------------------------
def test_r22_empty_inputs_and_numerical_guards():
    case = _case()
    empty = solve_xpyd_pools(case, total_npus=16, isl=1024, osl=16,
                             ttft_slo_ms=10000.0, tpot_slo_ms=1000.0,
                             decode_configs=[{"tp": 64, "dp": 1, "ep": 1}],
                             hccl_buffsize_mb=64)
    assert empty["best_configuration"] is None
    assert empty["status"] == "no_feasible_configuration"
    assert empty["kv_bytes_per_request"] > 0.0
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    full_hit = pool_config_facts(
        case, ParallelSpec(dp=2, tp=2, ep=2), phase="prefill", isl=2048,
        osl=16, prefix_hit_rate=1.0, tpot_slo_ms=50.0, hccl_buffsize_mb=64,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree, rho=0.9)
    assert full_hit["feasible"] and full_hit["q_tokens"] == 0
    for kwargs in ({"mc2_buffer_multiplier": -1.0},
                   {"graph_activation_bytes_per_token": float("nan")},
                   {"fragmentation_margin_bytes": -1.0}):
        with pytest.raises(ModelingError):
            analyze_memory_breakdown(case, sequences=1, sequence_tokens=1024, **kwargs)
    with pytest.raises(ModelingError):
        eplb_effect(8, 2, 4, 1.0, num_experts=8, redundant_experts=1,
                    extra_weight_bytes_per_rank=1, rebalance_bw_gbps=0.0)
    zero = kv_transfer_contention(
        0, n_prefill=1, n_decode=1, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=1e12, p_world=1, d_world=1,
        decode_step_ms=1.0, osl=1, requests_per_s=1.0)
    assert zero["transfer_ms"] == 0.0 and zero["feasible"]
    with pytest.raises(ModelingError):
        mtp_batch_reversal(lambda batch: float("nan"), draft_len=1, max_batch=2)
    with pytest.raises(ModelingError):
        ParallelSpec(tp=1.5)
    with pytest.raises(ModelingError):
        max_rank_tokens(4, 9, 4, 1.0, num_experts=8)


# ===========================================================================
# Round-2 executed witnesses W4..W7 and residual R-edge assertions.
# ===========================================================================

# W4 / R05 / R17 ------------------------------------------------------------
def test_w4_cross_stage_groups_are_rejected():
    case = _case(layers=4, tp=2, dp=1, ep=2, pp=2, max_model_len=4096,
                 num_devices=8)
    parallel = ParallelSpec(tp=2, dp=1, ep=2, pp=2,
                            rank_table=((0, 2), (1, 3)))
    _, plan, reason = validate_placement(case, parallel)
    assert plan is None and "crosses_pp_stage" in (reason or "")


def test_w4_prefill_dp_replicas_scale_pool_rate():
    dense = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0, max_model_len=4096)
    tree = HierarchicalFabricTree.from_hardware(dense.machine)
    rates = []
    for dp in (1, 2):
        row = pool_config_facts(
            dense, ParallelSpec(tp=2, dp=dp), phase="prefill", isl=256, osl=16,
            prefix_hit_rate=0.0, tpot_slo_ms=1000.0, hccl_buffsize_mb=64,
            routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
            tree=tree, rho=0.9)
        assert row["feasible"] and row["batch"] == 32
        assert row["step_ms"] == pytest.approx(8.970184588744589)
        rates.append(row["mu_requests_per_s"])
    assert rates[0] == pytest.approx(3567.373634668818)
    assert rates[1] == pytest.approx(2.0 * rates[0])


def test_w4_resolved_overrides_and_pair_conservation():
    dense = _case(experts=0, top_k=0, moe_inter=0, ep=1)
    broken = plan_context_parallel(
        dense.model, ParallelSpec(dp=1, tp=8, ep=1, pcp=1, cp=1),
        prefill_plan="dsa_cp")
    assert not broken.compatible and "sparse_indexer" in (broken.reason or "")
    sparse = _sparse_case(families=("dsa",), pcp=1, cp=1)
    dual = plan_context_parallel(
        sparse.model, ParallelSpec(dp=1, tp=8, ep=1, pcp=1, cp=1),
        prefill_plan="dsa_cp", decode_plan="dcp")
    assert not dual.compatible and dual.reason == "dsa_cp_excludes_dcp"
    assert valid_pairs("gqa", 4, 4, 1, 0, causal=False) == 16
    assert valid_pairs("gqa", 4, 4, 1, 0, causal=True) == 10
    pairs = pair_conservation(dense.model, ParallelSpec(), 4, 4, ("gqa",),
                              causal=False)
    assert pairs["gqa"]["total_pairs"] == 16
    assert pairs["gqa"]["expected_pairs"] == 16 and pairs["gqa"]["conserved"]
    bridge_plan = ParallelSpec(dp=1, tp=8, ep=1, pcp=2, cp=2, cp_plan="pcp",
                               pcp_lane_policy="replica_lanes")
    bridged = plan_context_parallel(sparse.model, bridge_plan, isl=1024,
                                    context_tokens=4096, lane_kv_bytes=1 << 20)
    assert bridged.compatible and bridged.bridge is not None
    assert bridged.bridge.destination_lanes == (0, 1)
    assert all(0 <= lane < bridged.bridge.dcp
               for lane in bridged.bridge.destination_lanes)
    assert all(0 <= src < bridged.bridge.pcp
               for src, _dst in bridged.bridge.lane_map)
    assert bridged.remap_bytes > 0.0 and bridged.remap_ms > 0.0
    assert bridged.remap_bytes >= (1 << 19)


# W5 / R10 / R11 ------------------------------------------------------------
def test_w5_top_k_without_replacement_and_exact_slots():
    assert expert_demand(1, 4, 4, 100.0) == (1, 1, 1, 1)
    assert max(expert_demand(1, 4, 4, 100.0)) <= 1
    # Contiguous ownership means a four-token EP working set has four hot
    # owners and spreads the five replicas one per non-owner rank.
    placement = place_eplb(4, 1, 8, 3.0, num_experts=16,
                           redundant_experts=5)
    assert placement.redundant_slots_per_rank == (0, 0, 1, 1, 1, 1, 1, 0)
    canonical = place_eplb(8, 1, 8, 3.0, num_experts=16,
                           redundant_experts=5)
    assert canonical.redundant_slots_per_rank == (0, 0, 0, 0, 2, 1, 1, 1)
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1,
                 max_model_len=4096)
    exact = replicated_expert_bytes(case, 5, redundant_slots_per_rank=(
        0, 0, 0, 0, 2, 1, 1, 1))
    default = replicated_expert_bytes(case, 5)
    assert exact["per_rank_replicas"] == 2
    assert exact["per_rank_bytes"][4] == pytest.approx(2 * exact["per_rank_bytes"][5])
    # Unpinned admission uses the proven tight envelope: min(R, E - base_owner)
    # = 5 replicas on every EP slot for R=5, EP=8, E=16.
    assert default["bytes_per_rank"] == pytest.approx(125_829_120.0)
    record = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8,
        enable_eplb=True, eplb_redundant_experts=5,
        eplb_redundant_slots_per_rank=(0, 0, 0, 0, 2, 1, 1, 1))
    fallback = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8,
        enable_eplb=True, eplb_redundant_experts=5)
    assert record.components[1].bytes == pytest.approx(exact["bytes_per_rank"])
    assert fallback.components[1].bytes == pytest.approx(125_829_120.0)
    with pytest.raises(ModelingError):
        max_rank_tokens(0, 1, 4, 1.0, num_experts=8, captured_rank_counts=())
    with pytest.raises(ModelingError):
        replicated_expert_bytes(case, 5, redundant_slots_per_rank=(0,) * 7)


# W6 / R01 / R19 ------------------------------------------------------------
def test_w6_speculative_tokens_are_never_free():
    base = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=8, max_model_len=4096)
    tree = HierarchicalFabricTree.from_hardware(base.machine)
    common = dict(phase="decode", isl=256, osl=16, prefix_hit_rate=0.0,
                  tpot_slo_ms=1000.0, hccl_buffsize_mb=64, routing_skew=1.0,
                  eplb_redundant_experts=0, enable_eplb=False, tree=tree,
                  rho=0.9)
    baseline = pool_config_facts(base, ParallelSpec(dp=1, tp=2), **common)
    assert baseline["feasible"] and baseline["useful_tokens_per_step"] == 1.0
    spec = replace(base, runtime=replace(base.runtime,
                                         metadata={"speculative": True}))
    speculative = pool_config_facts(spec, ParallelSpec(dp=1, tp=2), **common)
    assert not speculative["feasible"]
    assert speculative["reason"] == "speculative_requires_draft_weight_evidence"
    declared = replace(base, runtime=replace(
        base.runtime, metadata={"mtp_useful_tokens_per_step": 10}))
    non_spec = pool_config_facts(declared, ParallelSpec(dp=1, tp=2), **common)
    assert non_spec["feasible"] and non_spec["useful_tokens_per_step"] == 1.0
    assert non_spec["mu_requests_per_s"] == pytest.approx(
        baseline["mu_requests_per_s"])
    assert speculative_accept_tokens(1.0, 1, bonus_token=True) == 2.0
    assert speculative_accept_tokens(1.0, 1, bonus_token=False) == 1.0
    unqualified = mtp_batch_reversal(lambda batch: float(batch), draft_len=1,
                                     accept_rate=1.0, max_batch=4,
                                     draft_rollback_bytes_per_rank=10)
    assert not unqualified.memory_qualified
    qualified = mtp_batch_reversal(
        lambda batch: float(batch), draft_len=1, accept_rate=1.0, max_batch=4,
        draft_rollback_bytes_per_rank=10, rank_hbm_bytes=90,
        usable_capacity_bytes=100)
    assert qualified.memory_qualified


# W7 / R14 / R15 / R16 ------------------------------------------------------
def test_w7_receiver_incast_and_route_and_device_bounds():
    tree = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="per_port"),))
    incast = tree.collective("all_to_all_v", 1000, 4,
                             payloads_by_destination=(1000, 0, 0, 0))
    assert incast.max_receiver_ingress_bytes == pytest.approx(3000.0)
    assert incast.cut_bytes == pytest.approx(3000.0)
    assert incast.receiver_bound_us == pytest.approx(3.0)
    assert incast.total_us >= 3.0
    a5 = _case(sku="Ascend950DT_9582", topology="a5_pod", num_devices=64,
               ranks_per_node=64, experts=128, top_k=4, tp=2, dp=1, ep=2,
               max_seqs=32, max_model_len=16384)
    sweep = sweep_ep_knee(a5, ep_candidates=(2, 4), batch_sequences=8,
                          context_tokens=1024, hccl_buffsize_mb=64)
    assert sweep["status"] == "all_candidates_infeasible"
    assert all(row["reason"] == "fabric_route_unqualified"
               for row in sweep["rows"])
    tiny = _case(layers=4, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, num_devices=2, max_model_len=4096)
    oversized = solve_xpyd_pools(tiny, total_npus=8, isl=256, osl=16,
                                 ttft_slo_ms=1000.0, tpot_slo_ms=1000.0,
                                 hccl_buffsize_mb=8)
    assert oversized["best_configuration"] is None
    assert oversized["reason"] == "total_npus_exceeds_physical_devices"
    assert any(row.get("route_qualified") is False
               for row in sweep["rows"])


def test_w7_zero_byte_calibration_and_unmeasured_profile():
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 4, "points": [
            {"payload_bytes": 0, "alpha_us": 1.0, "beta_ns_per_byte": 0.0},
            {"payload_bytes": 4096, "alpha_us": 2.0, "beta_ns_per_byte": 0.01}]}]})
    assert profile.collective("all_reduce", 0, 2).total_us >= 0.0
    assert profile.collective("all_reduce", 1, 2).total_us >= 0.0
    unmeasured = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 4, "points": [
            {"payload_bytes": 4096, "alpha_us": 1.0,
             "beta_ns_per_byte": 0.01, "load_basis": "unmeasured"}]}]})
    record = unmeasured.collective("all_reduce", 4096, 2)
    assert record.route_qualified is False
    assert record.evidence == "unmeasured_profile"


# R22 -----------------------------------------------------------------------
def test_r22_type_hygiene_empty_lists_and_zero_transfer():
    case = _case(layers=4, tp=1, dp=4, ep=4)
    ledger = memory_ledger(case, sequence_tokens=1024, hccl_buffsize_mb=0)
    with pytest.raises(ValueError):
        ledger.at(1.9, 1024.9)
    with pytest.raises(ModelingError):
        ParallelSpec.from_mapping({"tp": 2, "ep": 2,
                                   "rank_table": [[0.9, 1.9]]})
    empty = solve_xpyd_pools(case, total_npus=4, isl=256, osl=16,
                             ttft_slo_ms=1000.0, tpot_slo_ms=1000.0,
                             decode_configs=[], hccl_buffsize_mb=8)
    assert empty["best_configuration"] is None
    assert empty["reason"] == "no_feasible_decode_configuration"
    with pytest.raises(ModelingError):
        logical_transfer_bytes(case, prompt_tokens=1024, block_size=-1)
    with pytest.raises(ModelingError):
        logical_transfer_bytes(case, prompt_tokens=1024,
                               metadata_bytes_per_page=-1.0)
    full = logical_transfer_bytes(case, prompt_tokens=1024,
                                  destination_cached_tokens=1024)
    assert full["missing_tokens"] == 0
    assert full["transfer_bytes"] == 0.0 and full["transfer_pages"] == 0
    report = solve_xpyd_pools(case, total_npus=4, isl=256, osl=16,
                              ttft_slo_ms=1000.0, tpot_slo_ms=1000.0,
                              destination_cached_tokens=256,
                              hccl_buffsize_mb=8,
                              prefill_configs=[ParallelSpec()],
                              decode_configs=[ParallelSpec()])
    assert report["kv_transfer"]["missing_tokens"] == 0
    assert report["kv_bytes_per_request"] == 0.0


# R18 -----------------------------------------------------------------------
def test_r18_marginal_is_delta_q_over_delta_n_and_fixed_world_tradeoff():
    rows = [{"ep": ep, "world_size": w, "tps": q, "tps_per_die": q / w,
             "step_ms": ms, "feasible": True}
            for ep, w, q, ms in ((2, 2, 100, 10), (4, 4, 180, 8), (8, 8, 200, 7))]
    optima = settle_optima(rows)
    assert tuple(item[2] for item in optima["marginal_knees"]) == (40.0, 5.0)
    report = sweep_tp_ep_tradeoff(
        _case(layers=4, tp=1, dp=2, ep=2, experts=32, top_k=2,
              max_model_len=4096),
        world_npus=8, tp_candidates=(1, 2), ep_candidates=(2, 4),
        batch_sequences=1, context_tokens=128, hccl_buffsize_mb=64)
    assert report["world_npus"] == 8
    assert all(row.get("tradeoff_world") == 8 for row in report["rows"])


# R21 -----------------------------------------------------------------------
def test_r21_onchip_scaling_reruns_tiling_and_prefetch():
    case = _case(layers=4, tp=1, dp=4, ep=4)
    model = DecodeCostModel.from_case(case, context_tokens=1024)
    zero = DecodeCostModel.from_case(
        scale_hardware(case, l1_bytes=0, ub_bytes=0), context_tokens=1024)
    assert zero.uncached_fallback and zero.tile_tokens == 0
    assert zero.step(32).step_ms != model.step(32).step_ms
    assert zero.step(32).step_ms > model.step(32).step_ms
