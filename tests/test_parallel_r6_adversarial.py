"""Round-6 adversarial regressions F01..F08, R03 and R09/R10.

[Role]
  Re-execute every Round-6 counterexample against the mathematical root cause:
  one EPLB resolver, a physical PP compute/link schedule, pool-wide shared cuts
  carrying inference collectives, one speculative PMF, conserved replicated
  bridges and page plans, per-layer sparse prefill, exact multi-stage placement
  partitions, actual-payload PP/draft qualification and strict pre-coercion
  validation.
[Invariants]
  Assertions are conservation identities, proven bounds or typed rejections;
  stochastic laws are checked by their sum/mean and by the renewal they induce,
  never by an implementation intermediate.
[Timeouts & Locks]
  Pure in-memory arithmetic; no fixture, tmp_path, subprocess or device access.
[Side-Effect Policy]
  Builds immutable Cases only; writes nothing.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.fabric_topology import CalibratedFabricProfile, HierarchicalFabricTree
from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.moe_skew_eplb import canonical_eplb_slots, max_rank_tokens
from mindie_sim.moe_mtp import speculative_accept_tokens, speculative_step_pmf
from mindie_sim.parallel_bridge import phase_bridge
from mindie_sim.parallel_context import pcp_attention_pairs, valid_pairs
from mindie_sim.parallel_cuts import unique_cut_demands
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_ep_knee import prefill_step_ms, sweep_ep_knee
from mindie_sim.parallel_placement import build_placement, validate_placement
from mindie_sim.parallel_prefill_cost import _layer_view
from mindie_sim.parallel_schedule import pp_microbatch_schedule_ms
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_instances import pool_config_facts, priced_draft_step_ms
from mindie_sim.xpyd_pool_solver import logical_transfer_bytes, solve_xpyd_pools
from mindie_sim.xpyd_transfer import completed_requests_per_s, expected_decode_cycles


def _case(*, sku="Ascend910_9362", topology="a3_16p", num_devices=64,
          ranks_per_node=16, layers=4, hidden=2048, heads=16, kv_heads=4,
          experts=32, top_k=4, moe_inter=512, tp=1, dp=2, ep=2, pp=1,
          pcp=1, cp=1, max_seqs=8, max_model_len=4096, graph="none"):
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku, "num_devices": num_devices,
                    "ranks_per_node": ranks_per_node,
                    "topology_preset": topology},
        "model": {"name": "r6-adversarial", "layers": layers,
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
                                   "output_tokens": 16, "arrival_ms": 0.0}]}})


def _machine_with(case, links, *, dies_per_package=1):
    return replace(case.machine, links=tuple(links), topology_options=(),
                   dies_per_package=dies_per_package)


def _link(name, members, *, capacity=1e9, latency=0.0, switch=None,
          semantics="per_port", channel="d2d_intra_node"):
    return Link(name, capacity, tuple(members), latency_us=latency,
                channel=channel, semantics=semantics,
                switch_capacity_bytes_per_ms=switch)


def _sparse(*, architecture="mla_dsa", ratio=0, topk=512, layers=1):
    plan = {"attention": "mla", "ffn": "dense", "architecture": architecture,
            "indexer": True, "indexer_type": "c4", "index_topk": topk,
            "owns_index_k": True, "index_dim": 128, "index_heads": 16,
            "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "kv_lora_rank": 512, "q_lora_rank": 512, "v_head_dim": 128,
            "compress_ratio": ratio,
            "window": 4096, "cache_kind": "dense"}
    case = parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 32, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "r6-sparse", "layers": layers, "hidden_size": 2048,
                  "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
                  "intermediate_size": 4096, "vocab_size": 32000,
                  "attention": "mla", "ffn": "dense",
                  "layer_plans": [{**plan, "index": index}
                                  for index in range(layers)]},
        "runtime": {"tp": 8, "dp": 2, "ep": 1, "pp": 1, "pcp": 2, "cp": 1,
                    "max_model_len": 8192, "max_num_seqs": 16,
                    "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.9, "kv_block_size": 128},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]}})
    return replace(case, model=replace(case.model, layer_plans=tuple(
        replace(plan, intermediate_size=4096)
        for plan in case.model.layer_plans)))


# F01: one canonical EPLB resolver across ledger, sweep, pool and solver.
def test_f01_supplied_slots_are_never_ignored_and_impossible_vectors_reject():
    case = _case(tp=1, dp=2, ep=2, experts=32, top_k=4, num_devices=4,
                 max_seqs=8)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    row = pool_config_facts(
        case, ParallelSpec(tp=1, dp=2, ep=2), phase="decode", isl=256,
        osl=16, prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=True,
        tree=tree, rho=0.9, eplb_redundant_slots_per_rank=(1, 0))
    assert row["feasible"] is False
    assert row["reason"] == "eplb_epoch_mismatch"
    impossible = solve_xpyd_pools(
        case, total_npus=4, isl=256, osl=16, ttft_slo_ms=1e9,
        tpot_slo_ms=1e9, prefill_configs=[ParallelSpec(tp=1, dp=1)],
        decode_configs=[ParallelSpec(tp=1, dp=2, ep=2)],
        eplb_redundant_experts=0, eplb_redundant_slots_per_rank=(0, 99),
        hccl_buffsize_mb=8)
    assert impossible["qualified"] is False
    with pytest.raises(ModelingError):
        pool_config_facts(
            case, ParallelSpec(tp=1, dp=2, ep=2), phase="decode", isl=256,
            osl=16, prefix_hit_rate=0.0, tpot_slo_ms=1e9,
            hccl_buffsize_mb=8, routing_skew=1.0,
            eplb_redundant_experts=0, enable_eplb=True, tree=tree, rho=0.9,
            eplb_redundant_slots_per_rank=(0, 99))
    paid = analyze_memory_breakdown(
        _case(tp=1, dp=8, ep=8, experts=16, top_k=1), sequences=1,
        sequence_tokens=1024, hccl_buffsize_mb=0, enable_eplb=True,
        eplb_redundant_experts=0,
        eplb_redundant_slots_per_rank=(0, 0, 0, 0, 0, 0, 0, 1))
    assert paid.per_rank_bytes[7][3] > 0.0
    assert paid.components[1].bytes > 0.0
    with pytest.raises(ModelingError):
        canonical_eplb_slots(1, 2, 4, top_k=1.9)


# F02: one compute-and-link schedule sets total latency; hop resource serializes.
def test_f02_pp_makespan_is_total_latency_and_hop_resource_serializes():
    assert pp_microbatch_schedule_ms(
        ((1, 1), (1, 1)), ((10, 0), (10, 0)))["makespan_ms"] == 22.0
    dense = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0, pp=2)
    model = DecodeCostModel.from_case(dense, context_tokens=1024)
    values = []
    for micro in (1, 2, 4):
        step = model.step(8, context=1024, microbatches=micro)
        assert step.step_ms == pytest.approx(
            step.pp_pipeline_makespan_ms + step.host_exposed_ms
            + step.eplb_rebalance_ms)
        assert step.step_ms < sum(
            float(value) for _name, value in step.schedule) + 1e-12
        values.append(step.step_ms)
    assert values[0] < values[1] < values[2]
    one = model.step(8, context=1024)
    assert one.step_ms == pytest.approx(
        one.pp_pipeline_makespan_ms + one.host_exposed_ms)
    prefill = [prefill_step_ms(dense, isl=256, prefix_hit_rate=0.0, batch=8,
                               microbatches=micro)
               for micro in (1, 2, 4)]
    for step in prefill:
        assert step["step_ms"] == pytest.approx(
            step["pp_pipeline_makespan_ms"] + step["host_ms"])
    assert prefill[0]["step_ms"] > prefill[1]["step_ms"] > prefill[2]["step_ms"]


def test_f02_physical_pp_hop_charges_latency_switch_and_actual_payload():
    dense = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0, pp=2)
    slow = replace(dense, machine=_machine_with(
        dense, [_link("one", (0, 1), latency=1_000_000.0)]))
    step = DecodeCostModel.from_case(slow, context_tokens=1024).step(8, context=1024)
    assert step.pp_p2p_ms >= 1000.0
    assert step.step_ms >= 1000.0
    switch = replace(dense, machine=_machine_with(
        dense, [_link("one", (0, 1), latency=0.0, switch=1.0)]))
    priced = DecodeCostModel.from_case(switch, context_tokens=1024).step(
        8, context=1024)
    assert priced.pp_p2p_ms >= 32_768.0
    row = pool_config_facts(
        slow, ParallelSpec(tp=1, dp=1, ep=1, pp=2), phase="decode", isl=256,
        osl=16, prefix_hit_rate=0.0, tpot_slo_ms=1.0, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=HierarchicalFabricTree.from_hardware(slow.machine), rho=0.9)
    assert row["feasible"] is False


# F04: shared switch carries inference, handoff and remap under one identity.
def test_f04_shared_switch_carries_inference_collectives_not_only_kv():
    case = _case(layers=4, hidden=2048, tp=1, dp=1, ep=1, experts=0,
                 top_k=0, moe_inter=0, num_devices=8, max_seqs=8)
    case = replace(case, machine=_machine_with(
        case, [_link("all", range(8), switch=1000.0)]))
    report = solve_xpyd_pools(
        case, total_npus=5, isl=256, osl=16, prefix_hit_rate=1.0,
        destination_cached_tokens=256, ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec(tp=1)],
        decode_configs=[ParallelSpec(tp=2)], hccl_buffsize_mb=8,
        max_pool_instances=2, prefix_digest="d" * 64)
    best = report["best_configuration"]
    assert best is not None and best["pd_ratio"] == "1P:2D"
    assert report["kv_bytes_per_request"] == 0.0
    assert best["shared_cut_demands"]["switch:intra_node_hccs"] == pytest.approx(
        524_288.0)
    assert best["shared_cut_rps"] == pytest.approx(1.9073486328125)
    assert best["admitted_rps"] <= 1.9073486328125
    assert best["admitted_rps"] != pytest.approx(3.43014572285227)
    for row in report["feasible_rows"]:
        assert row["admitted_rps"] <= 0.9 * 1.9073486328125 + 1e-12
        assert row["shared_cut_utilization"] <= 0.9 + 1e-12


def test_f04_concurrent_groups_on_one_switch_are_aggregated_in_one_step():
    case = _case(layers=1, tp=2, dp=2, ep=1, experts=0, top_k=0,
                 moe_inter=0, num_devices=4, max_seqs=8)
    case = replace(case, machine=_machine_with(
        case, [_link("all", (0, 1, 2, 3), switch=1000.0)]))
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    plan = build_placement(case, case.runtime.parallel).to_dict()
    demands = unique_cut_demands(tree, [(
        ((0, 1), (2, 3)), "all_reduce", 32_768.0)])
    assert demands["switch:intra_node_hccs"] == pytest.approx(131_072.0)
    step = DecodeCostModel.from_case(case, context_tokens=1024).step(
        8, context=1024)
    assert step.step_ms >= 131_072.0 * 1000.0 / 1000.0 / 1000.0


# F06: one canonical PMF derives both yield and finite renewal.
def test_f06_one_pmf_derives_yield_and_finite_osl_renewal():
    for k in (1, 2, 3, 4):
        for alpha in (0.2, 0.5, 0.8, 1.0):
            for bonus in (True, False):
                pmf = speculative_step_pmf(k, alpha, bonus_token=bonus)
                mean = sum((index + 1) * value for index, value in enumerate(pmf))
                assert sum(pmf) == pytest.approx(1.0, rel=1e-12)
                assert speculative_accept_tokens(alpha, k, bonus_token=bonus) == pytest.approx(mean)
                cycles = expected_decode_cycles(64, k, alpha, bonus_token=bonus)
                assert cycles >= math.ceil(64 / (k + (1 if bonus else 0)))
                rate = completed_requests_per_s(8, mean, 64, 1.0,
                                                speculative_k=k,
                                                accept_rate=alpha,
                                                bonus_token=bonus)
                assert rate <= 8 * mean / (64 * 1.0 / 1000.0) + 1e-9
    no_bonus = speculative_step_pmf(2, 0.5, bonus_token=False)
    assert no_bonus == pytest.approx((0.5, 0.5))
    assert speculative_accept_tokens(0.5, 2, bonus_token=False) == pytest.approx(1.5)
    assert expected_decode_cycles(64, 2, 0.5, bonus_token=False) == pytest.approx(
        42.888888888888886, rel=1e-12)
    with pytest.raises(ModelingError):
        expected_decode_cycles(64, 2, 0.5, 1.0, bonus_token=False)
    assert expected_decode_cycles(17, 2, None, 2.5) >= math.ceil(17 / 3)


def test_f06_pool_uses_the_same_stochastic_yield():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=8)
    declared = replace(case, runtime=replace(case.runtime, metadata={
        "speculative": True, "mtp_acceptance_length": 2,
        "mtp_shared_weights": True, "mtp_draft_layer_cost_ms": 0.001,
        "mtp_accept_rate": 0.5, "bonus_token": False}))
    row = pool_config_facts(
        declared, declared.runtime.parallel, phase="decode", isl=256, osl=64,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=HierarchicalFabricTree.from_hardware(declared.machine), rho=0.9)
    assert row["feasible"] and row["useful_tokens_per_step"] == pytest.approx(1.5)
    assert row["decode_cycles"] == pytest.approx(42.888888888888886)
    assert row["mu_requests_per_s"] <= (
        row["batch"] * 1.5 / (64 * row["step_ms"] / 1000.0) + 1e-9)


# F07/R03: conserved replication, page plans, per-layer sparse execution.
def test_f07_replicated_bridge_receives_one_payload_per_replica():
    from mindie_sim.parallel_bridge import bridge_remap_bytes, phase_bridge
    for sources in (1, 2, 4):
        for replicas in (1, 2, 3):
            bridge = phase_bridge(
                ParallelSpec(tp=4, pcp=sources),
                ParallelSpec(tp=4, cp=replicas), kv_bytes=1_000_000,
                index_bytes=1000, state_bytes=2000, replicated=True,
                page_ids=tuple(range(8)))
            assert len(bridge.destination_lanes) == replicas
            for value in bridge.destination_lane_bytes:
                assert value == pytest.approx(1_003_000.0)
            for pages in bridge.destination_pages:
                assert pages == tuple(range(8))
            assert sorted(page for chunk in bridge.source_pages
                          for page in chunk) == list(range(8))
            remap = bridge_remap_bytes(bridge)
            assert remap["total_bytes"] == pytest.approx(
                replicas * (1_003_000.0 + 4096.0))


def test_f07_page_plan_is_checked_and_sparse_prefill_is_per_layer():
    case = _case(experts=0, top_k=0, moe_inter=0, ep=1)
    with pytest.raises(ModelingError):
        logical_transfer_bytes(
            case, prompt_tokens=1024, destination_cached_tokens=1024,
            source_page_ids=(0, 1), destination_page_ids=(99,))
    sparse = _sparse(architecture="mla_csa", ratio=16)
    step = prefill_step_ms(sparse, isl=1024, prefix_hit_rate=0.0, batch=8)
    assert step["critical_pairs"] == 16_640
    assert step["critical_pairs"] == pcp_attention_pairs(
        isl=1024, q_tokens=1024, pcp=2, family="csa", index_topk=512,
        compress_ratio=16)["critical_pairs"]
    assert step["critical_pairs"] == valid_pairs(
        "csa", 1024, 1024, 2, 0, top_k=512, compress_ratio=16)
    assert step["attention_family"] == "csa"


def test_f07_heterogeneous_layers_compose_and_index_gather_is_priced():
    sparse = _sparse(architecture="mla_csa", ratio=16, layers=1)
    plans = [
        replace(sparse.model.layer_plans[0], index=0, architecture="mla_csa",
                compress_ratio=16, index_dim=4096),
        replace(sparse.model.layer_plans[0], index=1, architecture="mla_dsa",
                compress_ratio=0, index_dim=4096),
        replace(sparse.model.layer_plans[0], index=2, architecture="gqa",
                attention="gqa", indexer=False, indexer_type="none",
                index_topk=0, index_dim=0, index_heads=0, compress_ratio=0,
                window=0, cache_kind="dense", intermediate_size=4096,
                owns_index_k=False, owns_topk=False,
                index_query_per_layer=False),
    ]
    mixed = replace(sparse, runtime=replace(sparse.runtime,
                                            pp_layer_partition=()),
                    model=replace(sparse.model, layers=3,
                                  layer_plans=tuple(plans)))
    step = prefill_step_ms(mixed, isl=1024, prefix_hit_rate=0.0, batch=8)
    assert set(step["attention_family"].split("+")) == {"csa", "dsa", "gqa"}
    assert [layer["family"] for layer in
            (_layer_view(plan, {}) for plan in plans)] == ["csa", "dsa", "gqa"]
    no_index = replace(sparse, runtime=replace(sparse.runtime,
                                               pp_layer_partition=()),
                       model=replace(sparse.model, layers=3, layer_plans=tuple(
                           replace(plan, index=index, index_topk=0,
                                   index_dim=0, index_heads=0, indexer=False,
                                   owns_index_k=False, owns_topk=False,
                                   index_query_per_layer=False)
                           for index, plan in enumerate(plans))))
    cheap = prefill_step_ms(no_index, isl=1024, prefix_hit_rate=0.0, batch=8)
    assert step["collective_ms"] > cheap["collective_ms"]


def test_r03_transfer_staging_is_distributed_across_pp_stages():
    pp = _case(layers=8, tp=1, dp=2, ep=1, experts=0, top_k=0,
               moe_inter=0, pp=2, num_devices=4)
    pp = replace(pp, runtime=replace(pp.runtime, pp_layer_partition=(4, 4)))
    base = analyze_memory_breakdown(pp, sequences=1, sequence_tokens=1024,
                                    hccl_buffsize_mb=0)
    staged = analyze_memory_breakdown(pp, sequences=1, sequence_tokens=1024,
                                      hccl_buffsize_mb=0,
                                      kv_transfer_staging_bytes=1_000_000)
    deltas = [after[3] - before[3] for before, after
              in zip(base.per_rank_bytes, staged.per_rank_bytes)]
    assert all(value > 0.0 for value in deltas)


# F08/R09/R10: qualification propagation, exact partition, strict inputs.
def test_f08_pp_and_draft_actual_payload_unqualification_propagates():
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "points": [{"payload_bytes": 0, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001},
                   {"payload_bytes": 1024, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001},
                   {"payload_bytes": 32_768, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001, "load_basis": "unmeasured"},
                   {"payload_bytes": 65_536, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001},
                   {"payload_bytes": 262_144, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001},
                   {"payload_bytes": 1 << 30, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001}]}]})
    pp_case = _case(layers=4, tp=1, dp=2, ep=2, pp=2)
    pp_report = sweep_ep_knee(
        pp_case, ep_candidates=(2,), batch_sequences=8, context_tokens=1024,
        fabric=profile, hccl_buffsize_mb=8)
    assert pp_report["qualified"] is False and pp_report["ep_star"] is None
    assert pp_report["rows"][0]["reason"] == "fabric_route_unqualified"
    base = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=8)
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=1024,
                      intermediate_size=2048, num_heads=8, num_kv_heads=2,
                      head_dim=128, q_width=1024, kv_width=256,
                      vocab_size=32000)
    case = replace(base, model=replace(base.model,
                                       auxiliary=AuxiliarySpec(mtp=draft)))
    declared = replace(case, runtime=replace(case.runtime, metadata={
        "speculative": True, "mtp_acceptance_length": 3}))
    priced = priced_draft_step_ms(
        declared, draft, batch=8, context_tokens=272,
        fabric=profile, endpoints=(0, 1), groups=((0, 1),))
    assert priced is not None and priced.route_qualified is False
    assert priced.route_evidence == "unmeasured_profile"
    row = pool_config_facts(
        declared, declared.runtime.parallel, phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=profile, rho=0.9)
    assert row["feasible"] is False and row["route_qualified"] is False


def test_f08_explicit_multi_stage_groups_partition_and_attention_coords():
    case = _case(num_devices=32, kv_heads=2)
    rows = tuple((2 * index, 2 * index + 1) for index in range(8))
    candidate, plan, reason = validate_placement(
        case, ParallelSpec(tp=4, dp=2, ep=2, pp=2, cp=2, expert_tp=2,
                           expert_dp=2, rank_table=rows))
    assert reason is None and candidate is not None and plan is not None
    widths = {"tp": plan["tp"], "dcp": plan["dcp"], "pcp": plan["pcp"],
              "dp": plan["dp"], "ep": plan["ep"],
              "etp": plan["expert_tp"], "edp": plan["expert_dp"]}
    for axis, width in widths.items():
        groups = plan[axis + "_groups"]
        flat = [rank for group in groups for rank in group]
        if width > 1:
            assert len(flat) == len(set(flat)) == 16
            assert set(flat) == set(range(16))
    assert len({tuple(row[:5]) for row in plan["rank_map"]}) == 16
    attention = _case(num_devices=8, kv_heads=2)
    _, explicit, reason = validate_placement(
        attention, ParallelSpec(tp=4, dp=1, ep=2, expert_tp=2, cp=2))
    assert reason is None and explicit is not None
    assert len({tuple(row[:5]) for row in explicit["rank_map"]}) == 4
    assert {row[3] for row in explicit["rank_map"]} == {0, 1}


def test_f08_strict_pre_coercion_and_owner_captured_cap():
    case = _case(experts=0, top_k=0, moe_inter=0, ep=1)
    with pytest.raises(ModelingError):
        logical_transfer_bytes(case, prompt_tokens=1024,
                               source_epoch=1.1, destination_epoch=1.9)
    with pytest.raises(ModelingError):
        logical_transfer_bytes(case, prompt_tokens=1024,
                               source_epoch=1, destination_epoch=1.9)
    speculative = replace(case, runtime=replace(case.runtime, metadata={
        "speculative": True, "mtp_acceptance_length": 2.9,
        "mtp_shared_weights": True, "mtp_draft_layer_cost_ms": 0.001}))
    with pytest.raises(ModelingError):
        pool_config_facts(
            speculative, speculative.runtime.parallel, phase="decode",
            isl=256, osl=16, prefix_hit_rate=0.0, tpot_slo_ms=1e9,
            hccl_buffsize_mb=8, routing_skew=1.0,
            eplb_redundant_experts=0, enable_eplb=False,
            tree=HierarchicalFabricTree.from_hardware(case.machine), rho=0.9)
    wide = _case(tp=2, dp=2)
    model = DecodeCostModel.from_case(wide)
    with pytest.raises(ModelingError):
        model.step(8, endpoints=(0,))
    with pytest.raises(ModelingError):
        prefill_step_ms(wide, isl=256, prefix_hit_rate=0.0, batch=8,
                        endpoints=(0,))
    with pytest.raises(ModelingError):
        max_rank_tokens(1, 3, 2, 1.0, num_experts=3,
                        captured_rank_counts=(1, 2))
    assert max_rank_tokens(1, 3, 2, 1.0, num_experts=3,
                           captured_rank_counts=(2, 1)) == 2.0


def test_r09_communicator_arenas_consume_partitioned_groups():
    case = _case(layers=4, tp=4, dp=2, ep=2, pp=2, num_devices=32,
                 kv_heads=2)
    plan = build_placement(case, case.runtime.parallel)
    assert plan.dp_groups
    widths = {"tp": plan.tp, "dcp": plan.dcp, "pcp": plan.pcp,
              "dp": plan.dp, "ep": plan.ep, "etp": plan.expert_tp,
              "edp": plan.expert_dp}
    for axis, width in widths.items():
        groups = getattr(plan, axis + "_groups")
        flat = [rank for group in groups for rank in group]
        if width > 1:
            assert len(flat) == len(set(flat)) == plan.world
    record = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=64,
        moe_recipe="alltoall")
    assert len(record.per_rank_comm_bytes) == plan.world
    assert record.components[3].bytes == max(record.per_rank_comm_bytes)


def test_f02_f04_multi_combination_sweeps():
    for pp in (2, 3, 4):
        for micro in (1, 2, 4):
            case = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                         moe_inter=0, pp=pp)
            step = DecodeCostModel.from_case(case, context_tokens=1024).step(
                8, context=1024, microbatches=micro)
            assert step.step_ms == pytest.approx(
                step.pp_pipeline_makespan_ms + step.host_exposed_ms
                + step.eplb_rebalance_ms)
    for sources, replicas in ((1, 3), (2, 2), (4, 1)):
        bridge = phase_bridge(
            ParallelSpec(tp=4, pcp=sources), ParallelSpec(tp=4, cp=replicas),
            kv_bytes=500_000, replicated=True)
        assert all(value == pytest.approx(500_000.0)
                   for value in bridge.destination_lane_bytes)
