"""Round-3 adversarial regressions for defects A..F and residual R01..R22.

[Role]
  Encode every executed round-3 counterexample as a permanent physical
  assertion: orthogonal rank ownership, MoE work conservation, exact EPLB
  destinations, disjoint xPyD devices, queue-inclusive TTFT, paid speculative
  draft work, physical fabric cuts and conserved context/transfer ownership.

[Invariants]
  Assertions are conservation identities, physical lower bounds or typed
  rejection outcomes.  No golden latency is copied from the review.

[Timeouts & Locks]
  Pure in-memory arithmetic.  No fixture, tmp_path, subprocess, sleep, network
  or device access, so every test also runs when called directly.

[Side-Effect Policy]
  Builds immutable Cases only; writes nothing.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim.fabric_topology import (CalibratedFabricProfile, FabricTier,
                                       HierarchicalFabricTree, scale_hardware)
from mindie_sim.memory_breakdown import analyze_memory_breakdown, memory_ledger
from mindie_sim.moe_skew_eplb import (expert_demand, max_rank_tokens,
                                     mtp_batch_reversal, place_eplb,
                                     replicated_expert_bytes)
from mindie_sim.parallel_context import (pair_conservation, plan_context_parallel,
                                        valid_pairs)
from mindie_sim.parallel_ep_knee import (DecodeCostModel, prefill_step_ms,
                                        sweep_ep_knee)
from mindie_sim.parallel_placement import (build_placement, validate_placement,
                                          with_world)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_flow import queue_wait_ms
from mindie_sim.xpyd_pool_solver import (completed_requests_per_s,
                                        kv_transfer_contention,
                                        logical_transfer_bytes,
                                        pool_config_facts, solve_xpyd_pools)


def _case(*, sku="Ascend910_9362", topology="a3_16p", num_devices=64,
          ranks_per_node=16, layers=8, hidden=2048, heads=16, kv_heads=4,
          experts=32, top_k=4, moe_inter=512, tp=2, dp=2, ep=2, pp=1,
          pcp=1, cp=1, max_seqs=64, max_model_len=8192, graph="none"):
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku, "num_devices": num_devices,
                    "ranks_per_node": ranks_per_node,
                    "topology_preset": topology},
        "model": {"name": "r3-adversarial", "layers": layers,
                  "hidden_size": hidden, "num_heads": heads,
                  "num_kv_heads": kv_heads, "head_dim": 128,
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
    plans = [{
        "index": index, "attention": "mla", "ffn": "dense",
        "architecture": f"mla_{family}",
        "indexer": family in {"dsa", "csa", "csa2", "hca"},
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
    } for index, family in enumerate(families)]
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 32, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "r3-sparse", "layers": len(families),
                  "hidden_size": 2048, "num_heads": 16, "num_kv_heads": 4,
                  "head_dim": 128, "intermediate_size": 4096,
                  "vocab_size": 32000, "attention": "mla", "ffn": "dense",
                  "layer_plans": plans},
        "runtime": {"tp": 8, "dp": 2, "ep": 1, "pp": 1, "pcp": pcp, "cp": cp,
                    "max_model_len": 8192, "max_num_seqs": 16,
                    "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.9, "kv_block_size": 128},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]},
    })


def _fast_tree():
    return HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 8, 1.5, 392.0),
        FabricTier("superpod_hccs", 64, 3.5, 800.0),
    ))


# ---------------------------------------------------------------------------
# Defect A: orthogonal rank coordinates, one ownership map, MoE conservation.
# ---------------------------------------------------------------------------
def test_a_orthogonal_ep_etp_and_pp_pcp_coordinates():
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=32, top_k=4)
    plan = build_placement(case, case.runtime.parallel)
    assert [row[5] for row in plan.rank_map] == list(range(8))
    assert plan.ep_groups == (tuple(range(8)),)

    tp4 = _case(layers=4, tp=4, dp=1, ep=2, experts=32, top_k=4)
    orthogonal = build_placement(tp4, tp4.runtime.parallel)
    assert orthogonal.ep_groups == ((0, 2), (1, 3))
    assert orthogonal.etp_groups == ((0, 1), (2, 3))

    pp2 = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                moe_inter=0, pp=2, pcp=2)
    pp2 = with_world(pp2, ParallelSpec(tp=1, dp=1, pp=2, pcp=2))
    lane_plan = build_placement(pp2, pp2.runtime.parallel)
    assert lane_plan.pp_groups == ((0, 2), (1, 3))


def test_a_single_contiguous_expert_ownership():
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1)
    plan = build_placement(case, case.runtime.parallel)
    placement = place_eplb(4, 1, 8, 3.0, num_experts=16,
                           redundant_experts=0)
    nonowners = [(expert, rank) for row in placement.assignments
                 for expert, rank, _n in row
                 if rank != plan.expert_ownership[0][expert]]
    assert nonowners == []
    assert plan.expert_ownership[0][:4] == (0, 0, 1, 1)
    canonical = place_eplb(8, 1, 8, 3.0, num_experts=16,
                           redundant_experts=5)
    assert canonical.redundant_slots_per_rank == (0, 0, 0, 0, 2, 1, 1, 1)


def test_a_moe_flop_conservation_decode_and_prefill():
    required_global = 128 * 4 * 2 * 3 * 2048 * 512 * 4
    tp2 = _case(layers=4, hidden=2048, heads=16, kv_heads=4, experts=32,
                top_k=4, moe_inter=512, tp=2, dp=1, ep=2)
    model = DecodeCostModel.from_case(tp2, context_tokens=128)
    step = model.step(128, context=128, per_replica_batch=128,
                      active_replicas=1)
    assert step.tokens_entering_ep_group == 128.0
    assert step.expert_assignments_per_owner == 256.0
    assert step.expert_flops_per_owner == pytest.approx(
        required_global / 2, rel=1e-12)
    assert step.expert_flops_per_owner * 2 == pytest.approx(
        required_global, rel=1e-12)

    tp4 = _case(layers=4, hidden=2048, heads=16, kv_heads=4, experts=32,
                top_k=4, moe_inter=512, tp=4, dp=1, ep=2)
    etp2 = DecodeCostModel.from_case(tp4, context_tokens=128)
    step2 = etp2.step(128, context=128, per_replica_batch=128,
                      active_replicas=1)
    assert step2.expert_flops_per_owner == pytest.approx(
        required_global / 4, rel=1e-12)
    assert step2.expert_flops_per_owner * 4 == pytest.approx(
        required_global, rel=1e-12)

    for ep in (2, 4, 8):
        case = _case(layers=4, tp=1, dp=ep, ep=ep, experts=32, top_k=4)
        result = prefill_step_ms(case, isl=128, prefix_hit_rate=0.0, batch=128)
        assert result["expert_flops"] == pytest.approx(
            128 * 128 * 4 * 2 * 3 * 2048 * 512 * 4, rel=1e-12)
        assert result["compute_ms"] > 0.0


def test_a_captured_counts_are_combinatorially_feasible():
    with pytest.raises(ModelingError):
        max_rank_tokens(1, 4, 4, 1.0, num_experts=4,
                        captured_rank_counts=(4, 0, 0, 0))
    assert expert_demand(1, 4, 4, 100.0) == (1, 1, 1, 1)
    assert max_rank_tokens(2, 4, 4, 1.0, num_experts=4,
                           captured_rank_counts=(2, 2, 2, 2)) == 2.0


# ---------------------------------------------------------------------------
# Defect B: exact default EPLB destinations, staging, graph reserve, summary.
# ---------------------------------------------------------------------------
def test_b_default_eplb_ledger_uses_place_eplb():
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1,
                 max_model_len=4096)
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8,
                  enable_eplb=True, eplb_redundant_experts=5)
    default = analyze_memory_breakdown(case, **kwargs)
    exact = analyze_memory_breakdown(
        case, eplb_redundant_slots_per_rank=(0, 0, 0, 0, 2, 1, 1, 1), **kwargs)
    # Pinned canonical epoch: exactly two replicas on rank 4.
    assert exact.per_rank_bytes[4][3] == pytest.approx(665_164_288.0)
    # Unpinned admission: proven tight envelope = 5 replicas (125,829,120 B extra).
    assert default.per_rank_bytes[4][3] == pytest.approx(740_661_760.0)
    assert default.qualified and exact.qualified
    dies = list(case.machine.dies)
    dies[4] = replace(dies[4], memory_bytes=math.floor(652_581_375.3 / 0.9))
    tight = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    rejected = analyze_memory_breakdown(tight, **kwargs)
    assert not rejected.qualified
    assert rejected.status == "memory_infeasible"
    assert rejected.worst_rank == 4


def test_b_sweep_eplb_staging_tp_mapping_and_staging_per_rank():
    rows = {}
    for tp in (1, 2):
        case = _case(layers=4, tp=tp, dp=2, ep=2)
        report = sweep_ep_knee(
            case, ep_candidates=(2,), batch_sequences=8, context_tokens=1024,
            enable_eplb=True, eplb_redundant_experts=2, hccl_buffsize_mb=8)
        assert report["rows"][0]["feasible"]
        assert report["rows"][0]["eplb_rebalance_ms"] > 0.0
        rows[tp] = report["rows"][0]["peak_hbm_bytes"]
    assert rows[1] != rows[2]

    staged_case = _case(layers=4, tp=1, dp=2, ep=2, pp=2)
    base = analyze_memory_breakdown(
        staged_case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=0,
        enable_eplb=True, eplb_redundant_experts=1,
        eplb_redundant_slots_per_rank=(1, 0, 1, 0))
    staged = analyze_memory_breakdown(
        staged_case, eplb_rebalance_staging_bytes_per_rank=(1_000_000,) * 4,
        sequences=1, sequence_tokens=1024, hccl_buffsize_mb=0,
        enable_eplb=True, eplb_redundant_experts=1,
        eplb_redundant_slots_per_rank=(1, 0, 1, 0))
    charged = [after[3] - before[3]
               for before, after in zip(base.per_rank_bytes, staged.per_rank_bytes)]
    assert charged == pytest.approx([1_000_000.0] * 4)


def test_b_graph_reserve_is_additive_and_summary_is_consistent():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, graph="full")
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8,
                  graph_buckets=(1,), graph_activation_bytes_per_token=1,
                  graph_workspace_bytes=0, graph_pool_mode="shared")
    before = analyze_memory_breakdown(case, **kwargs)
    explicit = replace(case, runtime=replace(
        case.runtime, memory={**dict(case.runtime.memory),
                              "graph_pool_bytes": 32 << 20}))
    after = analyze_memory_breakdown(explicit, **kwargs)
    assert after.peak_hbm_bytes - before.peak_hbm_bytes == pytest.approx(32 << 20)

    split = _case(layers=8, tp=1, dp=2, ep=1, experts=0, top_k=0,
                  moe_inter=0, pp=2)
    split = replace(split, runtime=replace(split.runtime,
                                           pp_layer_partition=(7, 1)))
    dies = list(split.machine.dies)
    dies[2] = replace(dies[2], memory_bytes=math.ceil(600_000_000 / 0.9))
    split = replace(split, machine=replace(split.machine, dies=tuple(dies)))
    record = analyze_memory_breakdown(split, sequences=1, sequence_tokens=128,
                                      hccl_buffsize_mb=0)
    assert record.qualified
    assert record.worst_rank == 2 and record.worst_stage == 1
    assert record.headroom_bytes > 0.0
    row = record.per_rank_bytes[record.worst_rank]
    assert record.headroom_bytes == pytest.approx(row[4] - row[3])


def test_b_fractional_and_nonfinite_eplb_inputs_are_rejected():
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1)
    with pytest.raises(ModelingError):
        analyze_memory_breakdown(
            case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=0,
            enable_eplb=True, eplb_redundant_experts=1,
            eplb_redundant_slots_per_rank=(0, 0, 0, 0, 0.9, 0, 0, 0))
    with pytest.raises(ModelingError):
        analyze_memory_breakdown(
            case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=0,
            enable_eplb=True, eplb_redundant_experts=1,
            eplb_redundant_slots_per_rank=(0, 1, 0, 0, 0, 0, 0, 0),
            eplb_rebalance_staging_bytes_per_rank=(float("nan"),) * 8)


# ---------------------------------------------------------------------------
# Defect C: disjoint pool devices, queue-inclusive TTFT, strict saturation.
# ---------------------------------------------------------------------------
def test_c_pool_instances_use_disjoint_physical_devices():
    case = _case(layers=4, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, num_devices=2, max_seqs=8)
    dies = list(case.machine.dies)
    dies[1] = replace(dies[1], memory_bytes=1)
    case = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    report = solve_xpyd_pools(
        case, total_npus=2, isl=256, osl=16, ttft_slo_ms=1000.0,
        tpot_slo_ms=1000.0, prefill_configs=[ParallelSpec()],
        decode_configs=[ParallelSpec()], hccl_buffsize_mb=8)
    assert report["best_configuration"] is None
    assert report["qualified"] is False
    assert report["reason"] == "all_pool_candidates_rejected"
    assert report["rows"]
    assert any(row.get("decode_device_ids") == [1] for row in report["rows"])
    assert all(not row["feasible"] for row in report["rows"])


def test_c_ttft_includes_decode_queue_and_searches_below_cap():
    case = _case(layers=4, hidden=1024, heads=8, kv_heads=2, experts=0,
                 top_k=0, moe_inter=0, tp=1, dp=1, ep=1, max_seqs=64,
                 max_model_len=16384)
    tight = solve_xpyd_pools(
        case, total_npus=2, isl=8192, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=8192, ttft_slo_ms=5.0, tpot_slo_ms=100.0,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        hccl_buffsize_mb=64, max_pool_instances=1)
    assert tight["slo_qualification_mode"] == "analytic_md1_mean"
    assert tight["tail_quantile_certified"] is False
    best = tight["best_configuration"]
    # The declared 0.9 cap point is SLO-infeasible (its decode queue alone is
    # > 8 ms), but ``utilization_cap`` is an upper bound, not a mandatory
    # operating point: F03.2 requires the maximum SLO-feasible rate below it.
    assert best is not None and best["slo_feasible"]
    cap_queue_ms = queue_wait_ms(0.9, best["decode_step_ms"])
    assert cap_queue_ms > 8.0
    assert cap_queue_ms + best["prefill_service_ms"] + best["decode_step_ms"] > 5.0
    assert best["admitted_rps"] < 0.9 * best["raw_capacity_rps"]
    assert best["rho_decode"] < 0.9
    assert best["ttft_ms"] <= 5.0
    assert best["ttft_ms"] == pytest.approx(
        best["queue_prefill_ms"] + best["prefill_service_ms"]
        + best["transfer_ms"] + best["handoff_ms"]
        + best["transfer_queue_ms"] + best["queue_decode_ms"]
        + best["decode_step_ms"] + best["remap_ms"], rel=1e-12)
    loose = solve_xpyd_pools(
        case, total_npus=2, isl=8192, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=0, ttft_slo_ms=1e12, tpot_slo_ms=1e12,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        hccl_buffsize_mb=64, max_pool_instances=1)
    best = loose["best_configuration"]
    assert best is not None and best["feasible"]
    assert best["admitted_rps"] < best["decode_effective_rps"]
    assert best["rho_decode"] == pytest.approx(
        best["admitted_rps"] / best["decode_effective_rps"], rel=1e-12)
    assert best["decode_effective_rps"] == pytest.approx(
        best["decode_rps"] / (1.0 + best["d_hbm_jitter"]), rel=1e-9)
    assert best["queue_decode_ms"] == pytest.approx(
        queue_wait_ms(best["rho_decode"], best["decode_step_ms"]), rel=1e-9)


def test_c_exact_nic_saturation_is_unstable():
    saturated = kv_transfer_contention(
        1e9, n_prefill=1, n_decode=1, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1,
        decode_step_ms=1.0, osl=1, requests_per_s=1.0)
    assert saturated["wire_requests_per_s"] == pytest.approx(1.0)
    assert saturated["feasible"] is False
    assert saturated["overloaded"] is True
    assert math.isinf(saturated["queue_ms"])
    below = kv_transfer_contention(
        1e9, n_prefill=1, n_decode=1, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1,
        decode_step_ms=1.0, osl=1, requests_per_s=0.9)
    assert below["feasible"] is True


# ---------------------------------------------------------------------------
# Defect D: shared-weight speculative draft work is paid.
# ---------------------------------------------------------------------------
def test_d_shared_draft_layer_cost_is_charged():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=8, max_model_len=4096)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    common = dict(phase="decode", isl=256, osl=16, prefix_hit_rate=0.0,
                  tpot_slo_ms=1e9, hccl_buffsize_mb=64, routing_skew=1.0,
                  eplb_redundant_experts=0, enable_eplb=False, tree=tree,
                  rho=0.9)
    steps = []
    for layer_cost in (1.0, 10_000.0):
        declared = replace(case, runtime=replace(
            case.runtime, metadata={
                "speculative": True, "mtp_acceptance_length": 3,
                "mtp_shared_weights": True,
                "mtp_draft_layer_cost_ms": layer_cost}))
        row = pool_config_facts(declared, declared.runtime.parallel, **common)
        assert row["feasible"]
        steps.append(row["step_ms"])
    assert steps[1] - steps[0] == pytest.approx(3 * (10_000.0 - 1.0), rel=1e-12)
    tight = replace(case, runtime=replace(
        case.runtime, metadata={
            "speculative": True, "mtp_acceptance_length": 3,
            "mtp_shared_weights": True, "mtp_draft_layer_cost_ms": 10_000.0}))
    rejected = pool_config_facts(
        tight, tight.runtime.parallel,
        **{**common, "tpot_slo_ms": 1.0})
    assert rejected["feasible"] is False
    assert rejected["reason"] == "decode_tpot_sla_or_memory_exceeded"


# ---------------------------------------------------------------------------
# Defect E: physical cuts, connectivity, per-port AllToAllV, Mode B.
# ---------------------------------------------------------------------------
def _cut_hardware():
    case = _case(layers=4, tp=1, dp=1, ep=1, num_devices=4)
    link = Link("one", 1e6, (0, 1, 2, 3), latency_us=0,
                channel="d2d_intra_node", semantics="per_port",
                endpoint_rates=((0, 1000.0),),
                switch_capacity_bytes_per_ms=1000.0)
    return replace(case.machine, links=(link,), topology_options=(),
                   dies_per_package=1), link


def test_e_endpoint_switch_cut_and_disconnected_participants():
    machine, link = _cut_hardware()
    tree = HierarchicalFabricTree.from_hardware(machine)
    routed = tree.collective("all_reduce", 1000, 2, endpoints=(0, 1))
    assert routed.route_qualified
    assert routed.total_us >= 1000.0
    disconnected = replace(machine, links=(
        replace(link, members=(0, 1), endpoint_rates=()),))
    route = HierarchicalFabricTree.from_hardware(disconnected).collective(
        "all_reduce", 1000, 2, endpoints=(2, 3))
    assert route.route_qualified is False
    assert route.evidence == "disconnected_participants"


def test_e_per_port_alltoallv_and_receiver_incast():
    per_port = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="per_port"),))
    shared = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="shared"),))
    fast = per_port.collective("all_to_all_v", 1000, 4)
    slow = shared.collective("all_to_all_v", 1000, 4)
    assert fast.total_us == pytest.approx(0.75, rel=1e-9)
    assert slow.total_us == pytest.approx(3.0, rel=1e-9)
    incast = per_port.collective(
        "all_to_all_v", 1000, 4, payloads_by_destination=(1000, 0, 0, 0))
    assert incast.max_receiver_ingress_bytes == pytest.approx(3000.0)
    assert incast.cut_bytes == pytest.approx(3000.0)
    assert incast.receiver_bound_us == pytest.approx(3.0)
    assert incast.total_us >= 3.0


def test_e_calibrated_profile_endpoints_fit_and_interpolation():
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "fit_conditions": {"world_size": "4", "collective": "all_reduce"},
        "points": [
            {"payload_bytes": 0, "alpha_us": 1.0, "beta_ns_per_byte": 0.0},
            {"payload_bytes": 4096, "alpha_us": 2.0, "beta_ns_per_byte": 1.0}]}]})
    with pytest.raises(ModelingError):
        profile.collective("all_reduce", 4096, 2)
    assert profile.collective("all_reduce", 4096, 4).total_us > 0.0
    no_fit = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "points": [
            {"payload_bytes": 0, "alpha_us": 1.0, "beta_ns_per_byte": 0.0},
            {"payload_bytes": 1 << 30, "alpha_us": 2.0,
             "beta_ns_per_byte": 1.0}]}]})
    case = _case(layers=4, tp=1, dp=2, ep=2)
    sweep = sweep_ep_knee(case, ep_candidates=(2,), batch_sequences=8,
                          context_tokens=1024, fabric=no_fit,
                          hccl_buffsize_mb=8)
    assert sweep["rows"] and sweep["rows"][0]["feasible"]
    assert profile.tiers[0].at(2048).alpha_us == pytest.approx(1.5)
    assert profile.tiers[0].at(2048).beta_ns_per_byte == pytest.approx(0.5)
    assert 1.99 < profile.tiers[0].at(4095).alpha_us <= 2.0


# ---------------------------------------------------------------------------
# Defect F: DualChunkSwap ownership, bridges, transfer ledger, tiling, PP.
# ---------------------------------------------------------------------------
def test_f_dualchunkswap_pairs_and_bridge_lanes():
    sparse = _sparse_case(families=("dsa",), pcp=2, cp=1)
    assert [valid_pairs("gqa", 8, 8, 2, rank) for rank in range(2)] == [18, 18]
    conservation = pair_conservation(
        sparse.model, ParallelSpec(tp=8, dp=1, ep=1, pcp=2, cp=1),
        8, 8, ("gqa",))
    assert conservation["gqa"]["total_pairs"] == 36
    assert conservation["gqa"]["conserved"]
    bridge_plan = ParallelSpec(tp=8, dp=1, ep=1, pcp=2, cp=4,
                               pcp_lane_policy="replica_lanes")
    bridged = plan_context_parallel(
        sparse.model, bridge_plan, isl=1024, context_tokens=4096,
        lane_kv_bytes=1 << 20)
    assert bridged.compatible and bridged.bridge is not None
    assert bridged.bridge.destination_lanes == (0, 1, 2, 3)
    assert all(0 <= source < bridged.bridge.pcp
               for source, _destination in bridged.bridge.lane_map)
    assert bridged.remap_bytes > 0.0


def test_f_prefill_rate_independent_of_decode_lane_policy():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, pcp=2)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    rates = []
    for policy in ("park_extra_lanes", "replica_lanes"):
        parallel = replace(case.runtime.parallel, pcp_lane_policy=policy)
        row = pool_config_facts(
            case, parallel, phase="prefill", isl=256, osl=16,
            prefix_hit_rate=0.0, tpot_slo_ms=1000.0, hccl_buffsize_mb=8,
            routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
            tree=tree, rho=0.9)
        assert row["feasible"]
        rates.append(row["mu_requests_per_s"])
    assert rates[0] == pytest.approx(rates[1], rel=1e-12)


def test_f_shard_transfer_ledger_conserves_bytes():
    case = _case(layers=4, tp=4, dp=1, ep=1, experts=0, top_k=0, moe_inter=0)
    transfer = logical_transfer_bytes(case, prompt_tokens=1024,
                                      shard_ranks=4, page_size=128)
    assert transfer["missing_tokens_per_shard"] == 1024
    assert transfer["transfer_pages"] == 8
    assert transfer["transfer_bytes"] == pytest.approx(2_097_152.0)
    assert transfer["shard_bytes_aggregate"] == pytest.approx(8_388_608.0)
    assert sum(transfer["shard_bytes"]) == pytest.approx(8_388_608.0)
    assert len(transfer["page_ids"]) == transfer["transfer_pages"]
    assert len(transfer["shard_layout"]) == 4
    assert len(transfer["prefix_hash"]) == 64
    assert transfer["epoch"] == 0


def test_f_prefill_onchip_fallback_and_pp_stage_imbalance():
    dense = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0)
    base = prefill_step_ms(dense, isl=256, prefix_hit_rate=0.0, batch=32)
    zero = prefill_step_ms(scale_hardware(dense, l1_bytes=0, ub_bytes=0),
                           isl=256, prefix_hit_rate=0.0, batch=32)
    assert base["step_ms"] == pytest.approx(8.970184588744589)
    assert zero["step_ms"] > base["step_ms"]
    assert zero["uncached_fallback_ms"] > 0.0

    pp = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
               moe_inter=0, pp=2)
    timings = {}
    for split in ((7, 1), (4, 4)):
        partitioned = replace(pp, runtime=replace(
            pp.runtime, pp_layer_partition=split))
        decode = DecodeCostModel.from_case(partitioned).step(8)
        prefill = prefill_step_ms(partitioned, isl=256, prefix_hit_rate=0.0,
                                  batch=8)
        timings[split] = (decode.step_ms, decode.pp_max_stage_ms,
                          decode.pp_fill_drain_ms, prefill["step_ms"],
                          prefill["pp_max_stage_ms"])
    assert timings[(7, 1)][1] > timings[(4, 4)][1]
    assert timings[(7, 1)][2] > timings[(4, 4)][2]
    assert timings[(7, 1)][3] > timings[(4, 4)][3]
    assert timings[(7, 1)][4] > timings[(4, 4)][4]


# ---------------------------------------------------------------------------
# Residual R01..R22 contracts exercised through the same public entry points.
# ---------------------------------------------------------------------------
def test_r01_r08_residual_contracts():
    rate = completed_requests_per_s(12, 1.0, 1024, 13.916234315833)
    assert rate == pytest.approx(0.84209202964247, rel=1e-12)

    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_model_len=4096)
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=64)
    before = analyze_memory_breakdown(case, **kwargs)
    reserved = replace(case, runtime=replace(
        case.runtime, memory={**dict(case.runtime.memory),
                              "communication_buffer_bytes": 32 << 20}))
    after = analyze_memory_breakdown(reserved, **kwargs)
    assert after.peak_hbm_bytes - before.peak_hbm_bytes == pytest.approx(32 << 20)

    ledger = memory_ledger(case, sequence_tokens=1024, hccl_buffsize_mb=64)
    record = ledger.at(1, 1024)
    assert record.worst_rank in (0, 1)
    assert record.peak_hbm_bytes <= record.budget_bytes
    assert record.headroom_bytes == pytest.approx(
        record.budget_bytes - record.peak_hbm_bytes, rel=1e-12)

    table_case = _case(layers=4, tp=2, dp=1, ep=2, pp=2, num_devices=8)
    _, plan, reason = validate_placement(
        table_case, ParallelSpec(tp=2, dp=1, ep=2, pp=2,
                                 rank_table=((0, 2), (1, 3))))
    assert plan is None and "crosses_pp_stage" in (reason or "")

    moe_case = _case(layers=4, tp=1, dp=4, ep=4, experts=32, top_k=2,
                     max_model_len=4096)
    model = DecodeCostModel.from_case(moe_case, context_tokens=1024)
    per_dp = model.step(128, context=1024, scaling_mode="fixed_per_dp_batch",
                        per_replica_batch=128, active_replicas=4)
    fixed = model.step(128, context=1024, scaling_mode="fixed_global_batch",
                       global_batch=128, active_replicas=4)
    assert per_dp.expert_assignments == 4 * fixed.expert_assignments

    serial = model.step(32, context=1024, fabric=_fast_tree())
    overlap = model.step(32, context=1024, fabric=_fast_tree(),
                         overlap_mode="dual_microbatch")
    assert serial.t_dispatch_combine_ms >= overlap.t_dispatch_combine_ms
    assert sum(value for _name, value in serial.schedule) == pytest.approx(
        serial.step_ms, rel=1e-12)

    graph = replace(case, runtime=replace(case.runtime,
                                          graph_capture_mode="full"))
    unqualified = analyze_memory_breakdown(graph, **kwargs)
    assert unqualified.status == "memory_unqualified"
    fixed_workspace = analyze_memory_breakdown(
        graph, graph_workspace_bytes=0, **kwargs)
    assert not fixed_workspace.memory_qualified


def test_r09_r16_residual_contracts():
    case = _case(layers=4, tp=8, dp=1, ep=2, experts=32, top_k=2,
                 max_model_len=4096)
    record = analyze_memory_breakdown(case, sequences=1, sequence_tokens=1024,
                                      hccl_buffsize_mb=64,
                                      moe_recipe="alltoall")
    assert len(record.per_rank_comm_bytes) == 8
    assert record.components[3].bytes == max(record.per_rank_comm_bytes)
    plan = build_placement(case, case.runtime.parallel)
    assert plan.ep_groups and plan.etp_groups

    with pytest.raises(ModelingError):
        max_rank_tokens(1, 4, 4, 1.0, num_experts=4,
                        captured_rank_counts=(4, 0, 0, 0))

    eplb_case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1)
    default = replicated_expert_bytes(eplb_case, 5)
    assert default["per_rank_replicas"] == 5
    assert default["per_rank_bytes"][4] == pytest.approx(125_829_120.0)

    saturated = kv_transfer_contention(
        1e9, n_prefill=1, n_decode=1, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1,
        decode_step_ms=1.0, osl=1, requests_per_s=1.0)
    assert saturated["feasible"] is False and math.isinf(saturated["queue_ms"])

    transfer_case = _case(tp=4, dp=2, ep=1, experts=0, top_k=0,
                          moe_inter=0, max_model_len=8192)
    first = logical_transfer_bytes(transfer_case, prompt_tokens=4096)
    reordered = with_world(transfer_case,
                           ParallelSpec(dp=2, tp=4, ep=1, pp=1))
    second = logical_transfer_bytes(reordered, prompt_tokens=4096)
    assert first["total_bytes"] == second["total_bytes"]
    assert first["prefix_hash"] == second["prefix_hash"]

    tree = HierarchicalFabricTree.from_hardware(_case().machine)
    same = tree.collective("all_reduce", 1024, 2, endpoints=(0, 1))
    cross = tree.collective("all_reduce", 1024, 2, endpoints=(0, 2))
    assert same.endpoint_path == ("intra_package_die",)
    assert cross.endpoint_path == ("intra_node_hccs",)

    flat = HierarchicalFabricTree((FabricTier("intra_node_hccs", 8, 1.0, 1e9),))
    ring = flat.collective("all_reduce", 8, 8)
    assert dict(ring.rounds)["intra_node_hccs"] == 14

    unmeasured = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 4, "points": [
            {"payload_bytes": 4096, "alpha_us": 1.0,
             "beta_ns_per_byte": 0.01, "load_basis": "unmeasured"}]}]})
    assert unmeasured.collective("all_reduce", 4096, 2).route_qualified is False


def test_r17_r22_residual_contracts():
    dense = _case(experts=0, top_k=0, moe_inter=0, ep=1)
    assert valid_pairs("gqa", 4, 4, 1, 0, causal=False) == 16
    assert valid_pairs("gqa", 4, 4, 1, 0, causal=True) == 10
    pairs = pair_conservation(dense.model, ParallelSpec(), 4, 4, ("gqa",),
                              causal=False)
    assert pairs["gqa"]["total_pairs"] == 16 and pairs["gqa"]["conserved"]

    rows = [{"ep": ep, "world_size": world, "tps": tps,
             "tps_per_die": tps / world, "step_ms": ms, "feasible": True}
            for ep, world, tps, ms in ((2, 2, 100, 10), (4, 4, 180, 8),
                                       (8, 8, 200, 7))]
    from mindie_sim.parallel_ep_knee import settle_optima
    optima = settle_optima(rows)
    assert tuple(item[2] for item in optima["marginal_knees"]) == (40.0, 5.0)

    equal = mtp_batch_reversal(lambda batch: float(batch), draft_len=1,
                               accept_rate=1.0, max_batch=4)
    assert equal.crossings == (1, 2, 3, 4)
    free = mtp_batch_reversal(lambda batch: float(batch), draft_len=1,
                              accept_rate=1.0, max_batch=2,
                              draft_layer_cost_ms=10.0)
    assert free.burst_itl_ms == pytest.approx(2.0 + 10.0)

    tree = HierarchicalFabricTree((FabricTier("intra_node_hccs", 2, 0.0, 1.0),))
    base = tree.collective("all_to_all_v", 1000, 2)
    mc2 = tree.collective("mc2", 1000, 2)
    assert mc2.total_us == pytest.approx(2.0 * base.total_us, rel=1e-12)
    assert mc2.wire_bytes_per_rank == pytest.approx(
        2.0 * base.wire_bytes_per_rank, rel=1e-12)

    a5 = _case(sku="Ascend950DT_9582", topology="a5_pod", num_devices=64,
               ranks_per_node=64)
    scaled = scale_hardware(a5, link_bw=2.0)
    before = HierarchicalFabricTree.from_hardware(a5.machine)
    after = HierarchicalFabricTree.from_hardware(scaled.machine)
    old = next(tier for tier in before.tiers if tier.name == "intra_node_hccs")
    new = next(tier for tier in after.tiers if tier.name == "intra_node_hccs")
    assert new.bw_gbps == pytest.approx(2.0 * old.bw_gbps)

    with pytest.raises(ModelingError):
        logical_transfer_bytes(dense, prompt_tokens=1024, block_size=-1)
    with pytest.raises(ModelingError):
        logical_transfer_bytes(dense, prompt_tokens=1024,
                               metadata_bytes_per_page=-1.0)
    empty = solve_xpyd_pools(
        dense, total_npus=4, isl=256, osl=16, ttft_slo_ms=1000.0,
        tpot_slo_ms=1000.0, decode_configs=[], hccl_buffsize_mb=8)
    assert empty["best_configuration"] is None
    assert empty["reason"] == "no_feasible_decode_configuration"
    full = logical_transfer_bytes(dense, prompt_tokens=1024,
                                  destination_cached_tokens=1024)
    assert full["transfer_bytes"] == 0.0 and full["transfer_pages"] == 0
    with pytest.raises(ValueError):
        memory_ledger(dense, sequence_tokens=1024).at(1.9, 1024.9)
