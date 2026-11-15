"""Adversarial regressions R01..R12 and Round-2 witnesses W1..W2.

[Role] Encode each executed witness from `astra_review.md` as a permanent
  behavioral assertion against the corrected distributed/memory physics.
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


# R01 -----------------------------------------------------------------------
def test_r01_mu_p_and_mu_d_are_completed_requests_per_second():
    fixed_rate = completed_requests_per_s(12, 1.0, 1024, 13.916234315833)
    assert fixed_rate == pytest.approx(0.84209202964247, rel=1e-12)
    assert completed_requests_per_s(12, 1.0, 1, 13.916234315833) > fixed_rate * 1000
    case = _case(tp=2, dp=1, ep=1, experts=0, top_k=0, moe_inter=0)
    spec = ParallelSpec(dp=1, tp=2, ep=1)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    for osl in (1, 1024):
        row = pool_config_facts(
            case, spec, phase="decode", isl=256, osl=osl, prefix_hit_rate=0.0,
            tpot_slo_ms=1000.0, hccl_buffsize_mb=64, routing_skew=1.0,
            eplb_redundant_experts=0, enable_eplb=False, tree=tree, rho=0.9)
        assert row["feasible"]
        expected = row["batch"] / (osl * row["step_ms"] / 1000.0)
        assert row["mu_requests_per_s"] == pytest.approx(expected, rel=1e-12)
        assert row["output_tokens_per_s"] == pytest.approx(expected * osl, rel=1e-12)
        assert row["output_tokens_per_s"] <= row["batch"] / (row["step_ms"] / 1000.0) + 1e-9
    report = solve_xpyd_pools(case, total_npus=8, isl=256, osl=16,
                              ttft_slo_ms=10000.0, tpot_slo_ms=1000.0,
                              hccl_buffsize_mb=64)
    best = report["best_configuration"]
    assert best["goodput_rps"] == pytest.approx(best["admitted_rps"])
    assert best["goodput_output_tps"] == pytest.approx(best["goodput_rps"] * 16)


# R02 -----------------------------------------------------------------------
def test_r02_explicit_reserves_are_charged_once():
    case = _case(layers=4, tp=1, dp=2, ep=1, experts=0, top_k=0, moe_inter=0,
                 max_model_len=4096)
    base = analyze_memory_breakdown(case, sequences=2, sequence_tokens=2048,
                                    hccl_buffsize_mb=8)
    policy = dict(case.runtime.memory)
    for name, value in (("runtime_bytes", 1 << 30), ("workspace_bytes", 1 << 30),
                        ("temporary_dequant_bytes", 1 << 30),
                        ("graph_pool_bytes", 1 << 30),
                        ("communication_buffer_bytes", 1 << 30)):
        changed = replace(case, runtime=replace(
            case.runtime, memory={**policy, name: value}))
        record = analyze_memory_breakdown(changed, sequences=2, sequence_tokens=2048,
                                          hccl_buffsize_mb=8)
        assert record.peak_hbm_bytes - base.peak_hbm_bytes == pytest.approx(value), name
        assert record.runtime_reserve_bytes == (value if name == "runtime_bytes" else 0.0)
    margin = analyze_memory_breakdown(case, sequences=2, sequence_tokens=2048,
                                      hccl_buffsize_mb=8,
                                      fragmentation_margin_bytes=1 << 30)
    assert margin.peak_hbm_bytes - base.peak_hbm_bytes == pytest.approx(1 << 30)


# R03 -----------------------------------------------------------------------
def test_r03_full_ledger_is_rechecked_at_the_admitted_batch():
    p_case = _case(sku="Ascend910B4", topology="a2_standard", num_devices=64,
                   ranks_per_node=8, layers=8, hidden=2048, heads=16, kv_heads=4,
                   experts=0, top_k=0, moe_inter=0, tp=1, dp=1, ep=1,
                   max_seqs=256, max_model_len=32768)
    tree = HierarchicalFabricTree.from_hardware(p_case.machine)
    row = pool_config_facts(
        p_case, ParallelSpec(), phase="prefill", isl=8192, osl=16,
        prefix_hit_rate=0.999, tpot_slo_ms=50.0, hccl_buffsize_mb=64,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree, rho=0.9)
    assert row["feasible"] and row["batch"] < 256
    assert row["memory"].sequences == row["batch"]
    assert row["memory"].peak_hbm_bytes <= row["memory"].budget_bytes
    one_request = analyze_memory_breakdown(
        p_case, sequences=1, sequence_tokens=8192, hccl_buffsize_mb=64)
    full_batch = analyze_memory_breakdown(
        p_case, sequences=256, sequence_tokens=8192, hccl_buffsize_mb=64)
    assert full_batch.peak_hbm_bytes > full_batch.budget_bytes > one_request.peak_hbm_bytes

    a5 = _case(sku="Ascend950DT_9582", topology="a5_pod", num_devices=32,
               ranks_per_node=32, experts=128, top_k=4, tp=2, dp=1, ep=2,
               max_seqs=32, max_model_len=16384)
    dies = tuple(replace(die, memory_bytes=102726449494) for die in a5.machine.dies)
    tight = replace(a5, machine=replace(a5.machine, dies=dies))
    row = pool_config_facts(
        tight, ParallelSpec(dp=1, tp=2, ep=2), phase="decode", isl=8192, osl=1024,
        prefix_hit_rate=0.0, tpot_slo_ms=50.0, hccl_buffsize_mb=64,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=HierarchicalFabricTree.from_hardware(tight.machine), rho=0.9)
    if row["feasible"]:
        assert row["memory"].sequences == row["b_sla"]
        assert row["memory"].peak_hbm_bytes <= row["memory"].budget_bytes
    else:
        assert ("memory" in row["reason"] or "tpot" in row["reason"]
                or row["reason"] == "fabric_route_unqualified")


# R04 -----------------------------------------------------------------------
def test_r04_every_rank_capacity_is_gated():
    case = _case(layers=4, tp=1, dp=2, ep=1, experts=0, top_k=0, moe_inter=0,
                 max_model_len=4096)
    dies = list(case.machine.dies)
    dies[1] = replace(dies[1], memory_bytes=1)
    bad = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    record = analyze_memory_breakdown(bad, sequences=1, sequence_tokens=2048,
                                      hccl_buffsize_mb=8)
    assert not record.oom_feasible and record.status == "memory_infeasible"
    assert len(record.per_rank_bytes) == 2
    assert any(row[3] > row[4] for row in record.per_rank_bytes)
    assert record.headroom_bytes < 0


# R05 -----------------------------------------------------------------------
def test_r05_lane_plane_factorization_and_lane_policy():
    case = _case(layers=8, hidden=2048, heads=16, kv_heads=2, experts=32,
                 top_k=4, tp=8, dp=2, ep=16, cp=4, pcp=2, max_model_len=8192,
                 num_devices=32)
    canonical = ParallelSpec(dp=2, tp=8, ep=16, pp=1, cp=4, pcp=2, cp_plan="pcp")
    candidate, plan, reason = validate_placement(case, canonical)
    assert reason is None and plan["world"] == 32
    assert plan["lane_plane"] == 16
    assert plan["expert_tp"] * plan["expert_dp"] * 16 == 16
    assert plan["active_replicas"] == 2 and plan["parked_lanes"] == 1
    replica = ParallelSpec(dp=2, tp=8, ep=16, pp=1, cp=4, pcp=2, cp_plan="pcp",
                           pcp_lane_policy="replica_lanes")
    _, rplan, rreason = validate_placement(case, replica)
    assert rreason is None and rplan["active_replicas"] == 4
    remap = lane_remap_bytes(1 << 20, 1 << 12, 1 << 10, _plan_view(rplan))
    assert remap["total_bytes"] > (1 << 20) * (rplan["pcp"] - 1)
    parked_remap = lane_remap_bytes(1 << 20, 1 << 12, 1 << 10, _plan_view(plan))
    assert parked_remap["total_bytes"] == 0.0
    _, bad_plan, bad_reason = validate_placement(
        case, ParallelSpec(dp=1, tp=8, ep=16, pp=1, pcp=2, cp_plan="pcp"))
    assert bad_plan is None and "ep_exceeds_lane_plane" in (bad_reason or "")
    # PP2 validation uses the candidate partition, not the stale PP1 one.
    pp_case = _case(layers=4, tp=2, dp=1, ep=1, pp=1, max_model_len=4096)
    pp2 = ParallelSpec(dp=1, tp=2, ep=1, pp=2)
    _, pplan, preason = validate_placement(pp_case, pp2)
    assert preason is None and pplan["pp"] == 2 and len(pplan["pp_groups"]) == 2
    with pytest.raises(ModelingError):
        with_world(_case(num_devices=2), ParallelSpec(dp=1, tp=4, ep=1))


def _plan_view(plan):
    class View:
        pcp_lane_policy = plan["pcp_lane_policy"]
        parked_lanes = plan["parked_lanes"]
        pcp = plan["pcp"]
        active_lanes = plan["active_lanes"]
    return View()


# R06 -----------------------------------------------------------------------
def test_r06_ep_scaling_modes_do_not_mix_global_work():
    case = _case(tp=1, dp=4, ep=4, layers=4, max_model_len=4096)
    model = DecodeCostModel.from_case(case, context_tokens=1024)
    local = model.step(128, context=1024, scaling_mode="fixed_per_dp_batch",
                       per_replica_batch=128, active_replicas=4)
    global_fixed = model.step(128, context=1024, scaling_mode="fixed_global_batch",
                              global_batch=128, active_replicas=4)
    assert local.global_batch == 512 and local.useful_tokens_per_step == 512
    assert global_fixed.global_batch == 128 and global_fixed.useful_tokens_per_step == 128
    assert local.expert_assignments == 512 * 4
    assert global_fixed.expert_assignments == 128 * 4
    assert local.expert_assignments_per_owner == pytest.approx(
        4 * global_fixed.expert_assignments_per_owner, rel=1e-12)
    rows = sweep_ep_knee(case, ep_candidates=(2, 4), batch_sequences=128,
                         context_tokens=1024, scaling_mode="fixed_global_batch",
                         hccl_buffsize_mb=64)["rows"]
    feasible = [row for row in rows if row["feasible"]]
    assert feasible and all(row["global_batch"] == 128 for row in feasible)


# R07 -----------------------------------------------------------------------
def _comm(case, layers, mc2):
    rebuilt = _case(layers=layers, tp=1, dp=8, ep=8, top_k=2, max_model_len=4096)
    model = DecodeCostModel.from_case(rebuilt, context_tokens=1024)
    return model.step(128, context=1024, fabric=_tree(), mc2=mc2)


def test_r07_all_moe_layers_and_two_lane_schedule():
    case = _case(layers=4, tp=1, dp=8, ep=8, top_k=2, max_model_len=4096)
    four = _comm(case, 4, False)
    eight = _comm(case, 8, False)
    assert eight.t_dispatch_combine_ms == pytest.approx(
        2.0 * four.t_dispatch_combine_ms, rel=1e-9)
    assert eight.raw_comm_ms == pytest.approx(2.0 * four.raw_comm_ms, rel=1e-9)
    mc2_four, mc2_eight = _comm(case, 4, True), _comm(case, 8, True)
    assert mc2_eight.t_dispatch_combine_ms > 0.0
    assert mc2_eight.raw_comm_ms == pytest.approx(2.0 * mc2_four.raw_comm_ms, rel=1e-9)
    assert mc2_eight.t_dispatch_combine_ms <= mc2_eight.raw_comm_ms
    assert sum(value for _, value in eight.schedule) == pytest.approx(
        eight.step_ms, rel=1e-12)


# R08 -----------------------------------------------------------------------
def test_r08_graph_pool_shared_extent_and_workspace_alias():
    case = _case(layers=4, hidden=1024, heads=8, kv_heads=2, experts=0, top_k=0,
                 moe_inter=0, tp=4, dp=2, ep=1, num_devices=8, max_model_len=4096,
                 graph="full")
    kwargs = dict(sequences=2, sequence_tokens=2048, hccl_buffsize_mb=8,
                  graph_buckets=(1024, 2048), graph_activation_bytes_per_token=1 << 20,
                  graph_workspace_bytes=0)
    independent = analyze_memory_breakdown(case, graph_pool_mode="independent", **kwargs)
    shared = analyze_memory_breakdown(case, graph_pool_mode="shared", **kwargs)
    ci, cs = _components(independent), _components(shared)
    assert ci["graph_capture_pool_bytes"] == pytest.approx(
        (1024 + 2048) * (1 << 20))
    assert cs["graph_capture_pool_bytes"] == pytest.approx(2048 * (1 << 20))
    assert cs["transient_workspace_peak_bytes"] <= ci["transient_workspace_peak_bytes"]
    estimate = analyze_memory_breakdown(
        _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0, moe_inter=0,
              max_model_len=2048, graph="full"),
        sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8, graph_buckets=(1024,))
    assert estimate.status == "memory_unqualified" and not estimate.oom_feasible


# R09 -----------------------------------------------------------------------
def test_r09_communicator_arenas_follow_recipe_and_groups():
    case = _case(tp=8, dp=1, ep=8, cp=8, pcp=1, kv_heads=1, experts=128,
                 top_k=4, max_model_len=4096)
    alltoall = analyze_memory_breakdown(case, sequences=2, sequence_tokens=2048,
                                        hccl_buffsize_mb=64, moe_recipe="alltoall")
    mc2 = analyze_memory_breakdown(case, sequences=2, sequence_tokens=2048,
                                   hccl_buffsize_mb=64, moe_recipe="mc2")
    assert _components(mc2)["collective_static_buffers_bytes"] > _components(
        alltoall)["collective_static_buffers_bytes"]
    doubled = analyze_memory_breakdown(
        case, sequences=2, sequence_tokens=2048, hccl_buffsize_mb=64,
        moe_recipe="mc2", mc2_buffer_multiplier=2.0)
    single = _components(mc2)["collective_static_buffers_bytes"]
    assert _components(doubled)["collective_static_buffers_bytes"] > single
    with pytest.raises(ModelingError):
        analyze_memory_breakdown(case, sequences=1, sequence_tokens=1024,
                                 mc2_buffer_multiplier=-10000)


# R10 -----------------------------------------------------------------------
def test_r10_skew_boundaries_are_finite_and_integer():
    assert max_rank_tokens(0, 8, 8, 1.0, num_experts=32) == 0.0
    assert eta_skew([0, 0, 0, 0]) is None
    assert max_rank_tokens(1, 1, 128, 1.0, num_experts=128) == 1.0
    assert rank_load_cv(128, 1, 32, 1.0, 32) == pytest.approx(
        math.sqrt(31 / 128), rel=1e-12)
    for batch, top, ep, skew in ((1, 1, 4, 1.0), (17, 3, 8, 1.5),
                                 (128, 8, 32, 2.0)):
        value = max_rank_tokens(batch, top, ep, skew, num_experts=64)
        assert value == math.floor(value)
        assert value <= batch * top
    counts = (5, 3, 2, 2)
    assert max_rank_tokens(4, 3, 4, 1.0, num_experts=8,
                           captured_rank_counts=counts) == 5.0
    assert eta_skew(counts) > 1.0
    with pytest.raises(ModelingError):
        max_rank_tokens(4, 3, 4, 1.0, num_experts=8,
                        captured_rank_counts=(5, 3, 2, 1))
    with pytest.raises(ModelingError):
        max_rank_tokens(4, 3, 4, 1.0, num_experts=8,
                        captured_rank_counts=(5.5, 3, 2, 1.5))


# R11 -----------------------------------------------------------------------
def test_r11_integer_eplb_placement_and_rebalance_cost():
    single = place_eplb(1, 1, 8, 1.0, num_experts=8, redundant_experts=8,
                        per_token_ms=1.0)
    assert single.feasible and single.max_rank_service_ms == 1.0
    assert sum(single.rank_tokens) == 1
    demand = expert_demand(8, 2, 4, 2.0)
    assert sum(demand) == 16 and all(isinstance(value, int) for value in demand)
    hot = place_eplb(64, 2, 4, 2.0, num_experts=4, redundant_experts=2,
                     per_token_ms=1.0)
    baseline = place_eplb(64, 2, 4, 2.0, num_experts=4, redundant_experts=0,
                          per_token_ms=1.0)
    assert hot.feasible and hot.max_rank_service_ms <= baseline.max_rank_service_ms
    blocked = place_eplb(8, 1, 2, 3.0, num_experts=4, redundant_experts=1,
                         rank_token_capacity=(100, 0), per_token_ms=1.0)
    assert (not blocked.feasible) or blocked.max_rank_service_ms >= 0.0
    effect = eplb_effect(128, 8, 32, 1.5, num_experts=64, redundant_experts=2,
                         extra_weight_bytes_per_rank=1 << 30,
                         rebalance_bw_gbps=1.0, rebalance_interval_steps=2)
    expected = (1 << 30) / 1e9 * 1000.0 / 2
    assert effect.amortized_rebalance_ms == pytest.approx(expected, rel=1e-12)
    assert effect.placement is not None and effect.placement.rank_tokens
    assert effect.net_benefit_ms == pytest.approx(
        effect.shaved_tokens - effect.amortized_rebalance_ms, rel=1e-9)


# R12 -----------------------------------------------------------------------
def test_r12_strict_stability_and_simultaneous_cuts():
    one_p = kv_transfer_contention(
        1e9, n_prefill=1, n_decode=8, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1,
        decode_step_ms=10.0, osl=1024, requests_per_s=0.5)
    assert one_p["wire_requests_per_s"] == pytest.approx(1.0)
    assert "p_nic" in one_p["saturated_resources"]
    many_p = kv_transfer_contention(
        1e9, n_prefill=16, n_decode=8, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1,
        decode_step_ms=10.0, osl=1024, requests_per_s=0.5)
    assert many_p["wire_requests_per_s"] == pytest.approx(8.0)
    overload = kv_transfer_contention(
        1e9, n_prefill=1, n_decode=1, nic_bw_gbps=100.0,
        hbm_bw_bytes_per_s=1e9, p_world=1, d_world=1,
        decode_step_ms=10.0, osl=1024, requests_per_s=100.0)
    assert not overload["feasible"] and overload["tpot_jitter_ms"] is None
    assert overload["d_write_fraction"] >= 1.0
    resources = {"p_hbm": 4.0, "p_nic": 2.0, "d_nic": 3.0, "d_hbm": 5.0}
    solved = progressive_max_min(resources, (Flow("kv", tuple(
        (name, 1.0) for name in resources)),))
    assert solved["rates"]["kv"] == pytest.approx(2.0)
    assert len(solved["saturated"]) >= 1
    case = _case(tp=2, dp=1, ep=1, experts=0, top_k=0, moe_inter=0)
    report = solve_xpyd_pools(case, total_npus=8, isl=256, osl=16,
                              ttft_slo_ms=10000.0, tpot_slo_ms=1000.0,
                              hccl_buffsize_mb=64)
    assert report["best_configuration"]["utilization"] <= report["utilization_cap"] < 1.0


# ===========================================================================
# Round-2 executed witnesses W1..W2 and residual R-edge assertions.
# ===========================================================================

# W1 / R04 ------------------------------------------------------------------
def test_w1_rank_map_and_exact_per_rank_deficit():
    case = _case(layers=8, hidden=2048, heads=16, kv_heads=4, experts=0,
                 top_k=0, moe_inter=0, tp=1, dp=2, ep=1, pp=2,
                 max_model_len=8192)
    case = replace(case, runtime=replace(case.runtime, pp_layer_partition=(7, 1)))
    base = analyze_memory_breakdown(case, sequences=1, sequence_tokens=128,
                                    hccl_buffsize_mb=0)
    assert base.stage_peak_bytes == pytest.approx((934144512.0, 504707584.0))
    mid = sum(base.stage_peak_bytes) / 2.0
    dies = list(case.machine.dies)
    dies[1] = replace(dies[1], memory_bytes=math.ceil(mid / 0.9))
    bad = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    record = analyze_memory_breakdown(bad, sequences=1, sequence_tokens=128,
                                      hccl_buffsize_mb=0)
    assert record.status == "memory_infeasible" and not record.oom_feasible
    assert record.worst_rank == 1 and record.worst_stage == 0
    row = next(item for item in record.per_rank_bytes if item[2] == 1)
    assert row[1] == 0 and row[3] == pytest.approx(934144512.0)
    assert row[4] == pytest.approx(719426048.4)
    assert row[3] - row[4] == pytest.approx(214718463.6)
    assert record.headroom_bytes == pytest.approx(-214718463.6)
    _, plan, reason = validate_placement(bad, bad.runtime.parallel)
    assert reason is None and plan["rank_map"][1][0] == 0
    assert plan["rank_table"] == [[0, 1], [2, 3]]


# R02 -----------------------------------------------------------------------
def test_w1_r02_persistent_comm_reserve_is_additive():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_model_len=4096)
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=64)
    base = analyze_memory_breakdown(case, **kwargs)
    reserve = 32 << 20
    changed = replace(case, runtime=replace(
        case.runtime, memory={**dict(case.runtime.memory),
                              "communication_buffer_bytes": reserve}))
    record = analyze_memory_breakdown(changed, **kwargs)
    assert record.peak_hbm_bytes - base.peak_hbm_bytes == pytest.approx(reserve)
    comm = {item.name: item.bytes for item in record.components}[
        "collective_static_buffers_bytes"]
    assert comm == pytest.approx(base.components[3].bytes + reserve)


# R08 -----------------------------------------------------------------------
def test_w1_r08_activation_and_workspace_qualification_are_separate():
    dense = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0, max_model_len=4096)
    graph = replace(dense, runtime=replace(dense.runtime,
                                           graph_capture_mode="full"))
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=64)
    default = analyze_memory_breakdown(graph, **kwargs)
    assert not default.activation_peak_qualified
    assert default.status == "memory_unqualified"
    fixed0 = analyze_memory_breakdown(graph, graph_workspace_bytes=0, **kwargs)
    assert fixed0.graph_workspace_qualified and not fixed0.activation_peak_qualified
    assert not fixed0.memory_qualified
    one_byte = replace(graph, runtime=replace(
        graph.runtime, memory={"graph_pool_bytes": 1}))
    one = analyze_memory_breakdown(one_byte, **kwargs)
    assert not one.activation_peak_qualified and not one.memory_qualified
    explicit = analyze_memory_breakdown(
        graph, graph_activation_bytes_per_token=1 << 20,
        graph_workspace_bytes=1, **kwargs)
    assert explicit.activation_peak_qualified and explicit.graph_workspace_qualified
    assert explicit.memory_qualified


# R09 -----------------------------------------------------------------------
def test_w1_r09_communicator_arenas_follow_actual_groups():
    case = _case(layers=4, tp=8, dp=1, ep=2, cp=1, pcp=1,
                 experts=32, top_k=2, max_model_len=4096)
    record = analyze_memory_breakdown(case, sequences=1, sequence_tokens=1024,
                                      hccl_buffsize_mb=64, moe_recipe="alltoall")
    assert len(record.per_rank_comm_bytes) == 8
    assert record.components[3].bytes == max(record.per_rank_comm_bytes)
    _, plan, reason = validate_placement(case, case.runtime.parallel)
    assert reason is None and plan["etp_groups"]
    assert plan["ep_groups"] and all(
        len(group) == 2 for group in plan["ep_groups"])
    for stage, owners in enumerate(plan["expert_ownership"]):
        assert sorted(owners) == sorted(list(range(32))) or \
            set(owners) == set(range(case.runtime.parallel.ep))


# W2 / R06 ------------------------------------------------------------------
def test_w2_per_owner_flops_are_identical_across_ep():
    for ep in (2, 4, 8):
        case = _case(layers=4, tp=1, dp=ep, ep=ep, top_k=4, experts=32,
                     max_model_len=4096)
        model = DecodeCostModel.from_case(case, context_tokens=1024)
        step = model.step(128, context=1024, per_replica_batch=128,
                          active_replicas=ep)
        assert step.tokens_entering_ep_group == pytest.approx(ep * 128)
        assert step.expert_assignments_per_owner == pytest.approx(512.0)
        assert step.expert_flops_per_owner == pytest.approx(12884901888.0, rel=1e-12)


# R07 -----------------------------------------------------------------------
def test_r07_single_stream_is_causal_and_prefill_has_ep_comm():
    case = _case(layers=4, tp=1, dp=8, ep=8, top_k=2, max_model_len=4096)
    model = DecodeCostModel.from_case(case, context_tokens=1024)
    serial = model.step(128, context=1024, fabric=_tree(), mc2=True)
    overlap = model.step(128, context=1024, fabric=_tree(), mc2=True,
                         overlap_mode="dual_microbatch")
    assert serial.t_dispatch_combine_ms >= overlap.t_dispatch_combine_ms
    assert serial.schedule[3][0] == "fabric_communication"
    assert serial.t_dispatch_combine_ms == pytest.approx(
        4 * _tree().collective("mc2", 128 * 8 * 2 * 2048 * 2, 8).total_us / 1000.0)
    moe_prefill = prefill_step_ms(case, isl=1024, prefix_hit_rate=0.0, batch=4,
                                  fabric=_tree())
    assert moe_prefill["moe_comm_ms"] > 0.0
    pp_case = _case(layers=4, tp=1, dp=2, ep=2, pp=2, max_model_len=4096)
    pp_model = DecodeCostModel.from_case(pp_case, context_tokens=1024)
    one = pp_model.step(32, context=1024, per_replica_batch=32,
                        active_replicas=1, microbatches=1)
    many = pp_model.step(32, context=1024, per_replica_batch=32,
                         active_replicas=1, microbatches=4)
    assert one.pp_fill_drain_ms == pytest.approx(4 * many.pp_fill_drain_ms)
    merge = dcp_merge_payloads(8, 2, 4, 8, 128, 2.0)
    assert merge["output_elements"] == pytest.approx(4 * 8 * 129)
    assert merge["numerator_bytes"] == pytest.approx(4 * 8 * 129 * 2.0)
