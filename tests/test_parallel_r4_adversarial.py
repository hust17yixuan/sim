"""Round-4 adversarial regressions F01..F08.

[Role]
  Encode every executed Round-4 counterexample as a permanent physical
  assertion over memory, execution, fabric, transfer, speculation and public
  validation boundaries.
[Invariants]
  Assertions are conservation identities, physical lower bounds, exact
  component ledgers or typed rejections; every number is recomputed fresh.
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
from mindie_sim.fabric_topology import (CalibratedFabricProfile, FabricTier,
                                       HierarchicalFabricTree)
from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.moe_skew_eplb import (conservative_eplb_slots, place_eplb,
                                      replicated_expert_bytes)
from mindie_sim.moe_mtp import mtp_batch_reversal
from mindie_sim.parallel_context import (bridge_remap_bytes, phase_bridge,
                                         registered_builders)
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_ep_knee import (check_route_qualified,
                                         prefill_step_ms, sweep_ep_knee)
from mindie_sim.parallel_placement import validate_placement
from mindie_sim.parallel_schedule import pp_pipeline_ms, two_stream_schedule
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_instances import p_to_d_path, pool_config_facts
from mindie_sim.xpyd_pool_solver import (kv_transfer_contention,
                                         logical_transfer_bytes, solve_xpyd_pools)
from mindie_sim.xpyd_transfer import completed_requests_per_s
from mindie_sim.xpyd_flow import queue_wait_ms

def _case(*, sku="Ascend910_9362", topology="a3_16p", num_devices=64,
          ranks_per_node=16, layers=4, hidden=2048, heads=16, kv_heads=4,
          experts=32, top_k=4, moe_inter=512, tp=1, dp=2, ep=2, pp=1,
          pcp=1, cp=1, max_seqs=8, max_model_len=4096, graph="none"):
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku, "num_devices": num_devices,
                    "ranks_per_node": ranks_per_node,
                    "topology_preset": topology},
        "model": {"name": "r4-adversarial", "layers": layers,
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

def _sparse_case(*, pcp=2, cp=1):
    plan = {"attention": "mla", "ffn": "dense", "architecture": "mla_dsa",
            "indexer": True, "indexer_type": "c4", "index_topk": 512,
            "owns_index_k": True, "index_dim": 128, "num_heads": 16,
            "num_kv_heads": 4, "head_dim": 128, "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64, "kv_lora_rank": 512, "v_head_dim": 128,
            "window": 4096, "cache_kind": "dense"}
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 32, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "r4-sparse", "layers": 1, "hidden_size": 2048,
                  "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
                  "intermediate_size": 4096, "vocab_size": 32000,
                  "attention": "mla", "ffn": "dense",
                  "layer_plans": [{**plan, "index": 0}]},
        "runtime": {"tp": 8, "dp": 2, "ep": 1, "pp": 1, "pcp": pcp, "cp": cp,
                    "max_model_len": 8192, "max_num_seqs": 16,
                    "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.9, "kv_block_size": 128},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]}})

def _machine_with(case, links, *, dies_per_package=1):
    # One die per package (the Round-4 fixture): a die pair is then a declared
    # intra-node cut rather than a package-local alias.
    return replace(case.machine, links=tuple(links), topology_options=(),
                   dies_per_package=dies_per_package)

def _dp_link(name, members, capacity=1e6, **kwargs):
    return Link(name, capacity, tuple(members), latency_us=0,
                channel="d2d_intra_node", semantics="per_port", **kwargs)


# F01: conservative EPLB envelope and additive graph capture storage.
def test_f01_eplb_default_is_a_conservative_destination_envelope():
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1)
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8,
                  enable_eplb=True, eplb_redundant_experts=5)
    assert place_eplb(8, 1, 8, 100.0, num_experts=16,
                      redundant_experts=5).redundant_slots_per_rank == (
        0, 0, 0, 1, 1, 1, 1, 1)
    default = analyze_memory_breakdown(case, **kwargs)
    exact = analyze_memory_breakdown(
        case, eplb_redundant_slots_per_rank=(0, 0, 0, 1, 1, 1, 1, 1), **kwargs)
    # Rank 3 owns a real destination in a legal epoch: the unpinned ledger may
    # not charge it zero.
    assert default.per_rank_bytes[3][3] >= exact.per_rank_bytes[3][3]
    assert default.per_rank_bytes[3][3] >= 639_998_464.0
    dies = list(case.machine.dies)
    dies[3] = replace(dies[3], memory_bytes=math.floor(627_415_551.9 / 0.9))
    tight = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    rejected = analyze_memory_breakdown(tight, **kwargs)
    assert not rejected.qualified
    assert rejected.status == "memory_infeasible"
    assert rejected.worst_rank == 3
    pinned = analyze_memory_breakdown(
        tight, eplb_redundant_slots_per_rank=(0, 0, 0, 1, 1, 1, 1, 1), **kwargs)
    assert pinned.per_rank_bytes[3][3] == pytest.approx(639_998_464.0)
    envelope = conservative_eplb_slots(5, 8, 16, top_k=1)
    for batch in range(1, 9):
        epoch = place_eplb(batch, 1, 8, 100.0, num_experts=16,
                           redundant_experts=5).redundant_slots_per_rank
        assert all(envelope[rank] >= epoch[rank] for rank in range(8))
    replica = replicated_expert_bytes(case, 5)
    assert replica["per_rank_replicas"] == 5
    assert replica["per_rank_bytes"][4] == pytest.approx(125_829_120.0)

def test_f01_graph_capture_pool_is_additive_without_backing_handle():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, graph="full")
    kwargs = dict(sequences=1, sequence_tokens=1024, hccl_buffsize_mb=8,
                  graph_buckets=(1,), graph_workspace_bytes=0,
                  graph_pool_mode="shared")
    small = analyze_memory_breakdown(
        case, graph_activation_bytes_per_token=1, **kwargs)
    captured = analyze_memory_breakdown(
        case, graph_activation_bytes_per_token=33_554_432, **kwargs)
    assert captured.peak_hbm_bytes == pytest.approx(546_417_152.0)
    assert small.peak_hbm_bytes == pytest.approx(512_862_721.0)
    assert captured.peak_hbm_bytes - small.peak_hbm_bytes == pytest.approx(33_554_431.0)
    dies = tuple(replace(die, memory_bytes=math.floor(530_000_000.1 / 0.9))
                 for die in case.machine.dies)
    tight = replace(case, machine=replace(case.machine, dies=dies))
    standalone = analyze_memory_breakdown(
        tight, graph_activation_bytes_per_token=33_554_432, **kwargs)
    assert standalone.status == "memory_infeasible"
    assert standalone.peak_hbm_bytes == pytest.approx(546_417_152.0)
    aliased = analyze_memory_breakdown(
        tight, graph_activation_bytes_per_token=33_554_432,
        graph_operator_backing_pool="graph_arena_0", **kwargs)
    assert aliased.qualified
    assert aliased.peak_hbm_bytes == pytest.approx(512_862_720.0)


# F02: useful FLOPs, one straggler domain, causal schedule, paid PP/EPLB work.
def test_f02_useful_expert_flops_conserved_when_etp_exceeds_local_experts():
    case = _case(layers=4, tp=8, dp=1, ep=2, experts=2, top_k=2,
                 hidden=2048, moe_inter=512)
    required_global = 128 * 2 * 2 * 3 * 2048 * 512 * 4
    model = DecodeCostModel.from_case(case, context_tokens=1024)
    step = model.step(128, context=1024, per_replica_batch=128,
                      active_replicas=1)
    assert step.expert_assignments_per_owner == pytest.approx(128.0)
    assert step.expert_flops_per_owner == pytest.approx(805_306_368.0)
    assert step.expert_flops_per_owner * 8 == pytest.approx(required_global,
                                                           rel=1e-12)
    assert step.straggler_ratio == pytest.approx(1.0)
    prefill = prefill_step_ms(case, isl=128, prefix_hit_rate=0.0, batch=128)
    assert prefill["expert_flops"] * 8 == pytest.approx(
        128 * 128 * 2 * 2 * 3 * 2048 * 512 * 4, rel=1e-12)

def test_f02_schedule_is_causal_within_each_microbatch():
    record = two_stream_schedule(2, 10.0, 10.0, "dual_microbatch",
                                 attn_per_layer_ms=0.5)
    tasks = {name: (start, finish) for name, start, finish in record["tasks"]}
    for microbatch in (0, 1):
        for layer in (1,):
            dispatch = tasks[f"mb{microbatch}.dispatch:{layer}"]
            combine = tasks[f"mb{microbatch}.combine:{layer - 1}"]
            attn = tasks[f"mb{microbatch}.attn:{layer}"]
            assert dispatch[0] >= combine[1] - 1e-12
            assert dispatch[0] >= attn[1] - 1e-12
    serial = two_stream_schedule(2, 10.0, 10.0, "serial",
                                 attn_per_layer_ms=0.5)
    assert serial["makespan_ms"] == pytest.approx(41.0)
    for name, start, finish in serial["tasks"]:
        assert finish > start, name  # an executable trace, not zero timestamps
    assert serial["tasks"][1][1] >= serial["tasks"][0][2]

def test_f02_one_microbatch_pays_no_pp_fill_and_rebalance_once():
    case = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, pp=2)
    one = DecodeCostModel.from_case(case).step(8, context=1024, microbatches=1)
    assert one.pp_pipeline_bubble_ms == pytest.approx(0.0)
    assert one.pp_pipeline_makespan_ms == pytest.approx(
        sum(one.pp_stage_ms) + one.pp_p2p_ms)
    assert one.pp_p2p_ms > 0.0
    assert one.step_ms == pytest.approx(
        one.t_attn_ms + one.t_expert_compute_ms + one.t_expert_straggler_ms
        + one.t_dispatch_combine_ms + one.t_non_moe_ms + one.host_exposed_ms)
    equal = pp_pipeline_ms((5.0, 5.0), 4)
    assert equal["makespan_ms"] == pytest.approx(25.0)
    assert equal["bubble_ms"] == pytest.approx(15.0)
    uneven = pp_pipeline_ms((35.0, 5.0), 1)
    assert uneven["bubble_ms"] == pytest.approx(0.0)
    assert uneven["makespan_ms"] == pytest.approx(40.0)
    rebalance_case = _case(layers=4, tp=1, dp=2, ep=2)
    model = DecodeCostModel.from_case(rebalance_case, context_tokens=1024)
    base = model.step(8)
    charged = model.step(8, eplb_rebalance_ms=1.0)
    assert charged.step_ms - base.step_ms == pytest.approx(1.0, rel=1e-12)
    assert charged.eplb_rebalance_ms == pytest.approx(1.0)
    assert dict(charged.schedule)["eplb_rebalance"] == pytest.approx(1.0)
    assert dict(charged.schedule)["non_moe"] == pytest.approx(
        dict(base.schedule)["non_moe"])


# F03: KV wire queueing, utilization cap and admitted-rate search.
def test_f03_kv_transfer_wire_queue_and_cap():
    service = 33_570.816
    near = kv_transfer_contention(
        33_570_816, n_prefill=1, n_decode=1, nic_bw_gbps=0.001,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1,
        decode_step_ms=1.0, osl=1, requests_per_s=0.029787747786648975)
    assert near["kv_wire_utilization"] == pytest.approx(0.999999, rel=1e-6)
    assert near["kv_wire_service_ms"] == pytest.approx(service, rel=1e-9)
    assert near["queue_ms"] == pytest.approx(
        16_785_391_214.109325, rel=1e-9)
    assert near["queue_ms"] == pytest.approx(
        queue_wait_ms(near["kv_wire_utilization"], service), rel=1e-9)
    assert near["feasible"] is False
    assert near["reason"] == "kv_wire_utilization_cap_exceeded"


def test_f03_rate_search_finds_feasible_rate_below_the_cap():
    case = _case(layers=4, hidden=1024, heads=8, kv_heads=2, experts=0,
                 top_k=0, moe_inter=0, tp=1, dp=1, ep=1, max_seqs=64,
                 max_model_len=16384)
    common = dict(total_npus=2, isl=8192, osl=1, prefix_hit_rate=1.0,
                  destination_cached_tokens=8192,
                  prefill_configs=[ParallelSpec()],
                  decode_configs=[ParallelSpec()], hccl_buffsize_mb=64,
                  max_pool_instances=1)
    tight = solve_xpyd_pools(case, ttft_slo_ms=5.0, tpot_slo_ms=100.0, **common)
    assert tight["slo_qualification_mode"] == "analytic_md1_mean"
    assert tight["tail_quantile_certified"] is False
    best = tight["best_configuration"]
    assert best is not None and best["slo_feasible"]
    assert best["ttft_ms"] <= 5.0
    assert best["admitted_rps"] < 0.9 * best["raw_capacity_rps"]
    assert best["rho_decode"] < 0.9
    assert best["ttft_ms"] == pytest.approx(
        best["queue_prefill_ms"] + best["prefill_service_ms"]
        + best["transfer_ms"] + best["handoff_ms"]
        + best["transfer_queue_ms"] + best["queue_decode_ms"]
        + best["decode_step_ms"] + best["remap_ms"], rel=1e-12)
    low = solve_xpyd_pools(case, ttft_slo_ms=5.0, tpot_slo_ms=100.0,
                           utilization_cap=0.1, **common)["best_configuration"]
    assert low is not None and low["admitted_rps"] < best["admitted_rps"]
    assert low["utilization"] == pytest.approx(0.1, rel=1e-6)
    high = solve_xpyd_pools(case, ttft_slo_ms=1e12, tpot_slo_ms=100.0,
                            **common)["best_configuration"]
    assert high["rho_decode"] == pytest.approx(0.9, rel=1e-9)
    assert high["ttft_ms"] == pytest.approx(10.832322852851826, rel=1e-9)
    assert high["admitted_rps"] > best["admitted_rps"]


def test_f03_near_saturation_rate_pays_its_queue_or_is_rejected():
    case = _case(layers=4, hidden=1024, heads=8, kv_heads=2, experts=0,
                 top_k=0, moe_inter=0, tp=1, dp=1, ep=1, max_seqs=64,
                 max_model_len=16384)
    report = solve_xpyd_pools(
        case, total_npus=2, isl=8192, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=0, ttft_slo_ms=100_000.0, tpot_slo_ms=100_000.0,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        hccl_buffsize_mb=64, max_pool_instances=1, nic_bw_gbps=0.001)
    best = report["best_configuration"]
    assert best is not None
    assert best["ttft_ms"] <= 100_000.0
    assert best["admitted_rps"] < 0.029787747786648975
    assert best["rho_decode"] <= 0.9
    assert best["transfer_queue_ms"] > 0.0
    assert best["ttft_ms"] == pytest.approx(
        best["queue_prefill_ms"] + best["prefill_service_ms"] + best["transfer_ms"]
        + best["handoff_ms"] + best["transfer_queue_ms"] + best["queue_decode_ms"]
        + best["decode_step_ms"] + best["remap_ms"], rel=1e-12)


# F04: connectivity, every axis, real endpoints, preserved inter-node cuts.
def test_f04_disconnected_p_to_d_pool_is_rejected():
    case = _case(layers=4, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, num_devices=4, max_seqs=8)
    machine = _machine_with(case, [_dp_link("a", (0, 2)),
                                   _dp_link("b", (1, 3))])
    bad = replace(case, machine=machine)
    tree = HierarchicalFabricTree.from_hardware(machine)
    qualified, evidence, _bw = p_to_d_path(tree, (0,), (1,))
    assert qualified is False and "disconnected" in evidence
    report = solve_xpyd_pools(
        bad, total_npus=2, isl=256, osl=16, ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        hccl_buffsize_mb=8)
    assert report["qualified"] is False
    assert report["best_configuration"] is None
    assert any(str(row.get("reason", "")).startswith("p_to_d_route_unqualified")
               for row in report["rows"])
    allowed = solve_xpyd_pools(
        bad, total_npus=2, isl=256, osl=16, ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        hccl_buffsize_mb=8, allow_unqualified=True)
    assert allowed["rows"] and allowed["rows"][0]["route_qualified"] is False


def test_f04_every_communicator_axis_is_route_gated():
    case = _case(layers=4, tp=2, dp=2, ep=2, experts=32, top_k=2,
                 num_devices=4, max_seqs=8)
    machine = _machine_with(case, [_dp_link("only", (0, 1))])
    bad = replace(case, machine=machine)
    report = sweep_ep_knee(bad, ep_candidates=(2,), batch_sequences=8,
                           context_tokens=1024, hccl_buffsize_mb=8)
    assert report["status"] == "all_candidates_infeasible"
    assert report["ep_star"] is None
    assert report["rows"][0]["route_qualified"] is False
    assert "disconnect" in str(report["rows"][0]["route_evidence"]) or \
        "unmeasured" in str(report["rows"][0]["route_evidence"])
    tree = HierarchicalFabricTree.from_hardware(machine)
    _, plan, reason = validate_placement(bad, bad.runtime.parallel)
    assert reason is None and plan is not None
    ok, evidence = check_route_qualified(tree, plan, 0)
    assert ok is False and ("disconnect" in evidence or "unmeasured" in evidence)


def test_f04_instance_offset_prices_the_real_physical_dies():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, num_devices=4, max_seqs=8, ranks_per_node=4)
    link = Link("one", 1e9, (0, 1, 2, 3), latency_us=0,
                channel="d2d_intra_node", semantics="per_port",
                endpoint_rates=((3, 1.0),))
    machine = _machine_with(case, [link])
    case = replace(case, machine=machine)
    tree = HierarchicalFabricTree.from_hardware(machine)
    common = dict(phase="decode", isl=256, osl=16, prefix_hit_rate=0.0,
                  tpot_slo_ms=1e12, hccl_buffsize_mb=8, routing_skew=1.0,
                  eplb_redundant_experts=0, enable_eplb=False, tree=tree,
                  rho=0.9)
    first = pool_config_facts(case, ParallelSpec(tp=2), rank_offset=0, **common)
    second = pool_config_facts(case, ParallelSpec(tp=2), rank_offset=2, **common)
    assert first["feasible"] and second["feasible"]
    assert first["step_ms"] == pytest.approx(0.2357855573333333)
    assert second["step_ms"] > 1e5
    fast = tree.collective("all_reduce", 32_768, 2, endpoints=(0, 1))
    slow = tree.collective("all_reduce", 32_768, 2, endpoints=(2, 3))
    assert slow.total_us == pytest.approx(32_768_000.0)
    assert fast.total_us == pytest.approx(0.032768, rel=1e-6)
    assert second["step_ms"] - first["step_ms"] == pytest.approx(
        2.0 * (slow.total_us - fast.total_us) / 1000.0 * 4, rel=1e-9)


def test_f04_inter_node_tier_preserves_members_rates_and_switch():
    case = _case(layers=4, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, num_devices=4, ranks_per_node=2)
    machine = _machine_with(case, [
        _dp_link("local0", (0, 1)), _dp_link("local1", (2, 3)),
        Link("inter", 1e6, (0, 1, 2, 3), latency_us=0,
             channel="d2d_inter_node", semantics="per_port",
             endpoint_rates=((2, 1000.0),),
             switch_capacity_bytes_per_ms=1000.0)])
    tree = HierarchicalFabricTree.from_hardware(machine)
    tier = next(item for item in tree.tiers if item.name == "pod_roce")
    assert tier.link_members == ((0, 1, 2, 3),)
    assert dict(tier.endpoint_rates)[2] == pytest.approx(1000.0)
    assert tier.switch_capacity_bytes_per_ms == pytest.approx(1000.0)
    cross = tree.collective("all_reduce", 1000, 2, endpoints=(0, 2))
    assert cross.route_qualified
    assert cross.total_us == pytest.approx(1000.0, rel=1e-12)
    local = tree.collective("all_reduce", 1000, 2, endpoints=(0, 1))
    assert local.total_us == pytest.approx(1.0, rel=1e-9)


# F05: one AllToAllV byte ledger, receiver ingress, Mode-B skew rejection.
def test_f05_traffic_ledger_and_receiver_ingress_bound():
    per_port = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="per_port"),))
    shared = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="shared"),))
    implicit = per_port.collective("all_to_all_v", 1000, 4)
    explicit = per_port.collective(
        "all_to_all_v", 1000, 4,
        payloads_by_destination=(250.0, 250.0, 250.0, 250.0))
    for field in ("max_sender_egress_bytes", "max_receiver_ingress_bytes",
                  "cut_bytes", "total_us", "wire_bytes_per_rank"):
        assert getattr(implicit, field) == pytest.approx(
            getattr(explicit, field), rel=1e-12), field
    assert implicit.max_sender_egress_bytes == pytest.approx(750.0)
    assert implicit.max_receiver_ingress_bytes == pytest.approx(750.0)
    assert implicit.cut_bytes == pytest.approx(3000.0)
    assert implicit.total_us == pytest.approx(0.75, rel=1e-9)
    assert shared.collective("all_to_all_v", 1000, 4).total_us == pytest.approx(
        3.0, rel=1e-9)
    incast = per_port.collective(
        "all_to_all_v", 1000, 4, payloads_by_destination=(1000, 0, 0, 0))
    assert incast.max_receiver_ingress_bytes == pytest.approx(3000.0)
    assert incast.max_sender_egress_bytes == pytest.approx(1000.0)
    rated = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="per_port",
                   endpoint_rates=((0, 1000.0),)),))
    slow = rated.collective(
        "all_to_all_v", 1000, 4, payloads_by_destination=(1000, 0, 0, 0))
    assert slow.total_us == pytest.approx(3000.0, rel=1e-12)
    assert slow.receiver_bound_us == pytest.approx(3000.0)


def test_f05_mode_b_skewed_payload_fails_closed():
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "points": [{"payload_bytes": 0, "alpha_us": 0.0,
                    "beta_ns_per_byte": 1.0},
                   {"payload_bytes": 1 << 30, "alpha_us": 2.0,
                    "beta_ns_per_byte": 1.0}]}]})
    skew = profile.collective(
        "all_to_all_v", 1000, 4, payloads_by_destination=(1000, 0, 0, 0))
    assert skew.route_qualified is False
    assert skew.evidence == "calibrated_profile_uniform_only"
    assert profile.collective(
        "all_to_all_v", 1000, 4,
        payloads_by_destination=(250.0, 250.0, 250.0, 250.0)
    ).route_qualified is True
    declared = CalibratedFabricProfile.from_probe({
        "skew_supported": True, "tiers": [{
            "name": "intra_node_hccs", "domain_ranks": 64,
            "points": [{"payload_bytes": 0, "alpha_us": 0.0,
                        "beta_ns_per_byte": 1.0},
                       {"payload_bytes": 1 << 30, "alpha_us": 2.0,
                        "beta_ns_per_byte": 1.0}]}]})
    assert declared.collective(
        "all_to_all_v", 1000, 4,
        payloads_by_destination=(1000, 0, 0, 0)).route_qualified is True


# F06: every draft branch pays; request conservation; per-batch memory.
def _draft_case():
    base = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=8, max_model_len=4096)
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=2048,
                      intermediate_size=4096, num_heads=16, num_kv_heads=4,
                      head_dim=128, q_width=2048, kv_width=512,
                      vocab_size=32000)
    return replace(base, model=replace(
        base.model, auxiliary=AuxiliarySpec(mtp=draft)))


def _decode_row(case, metadata, *, tpot=1000.0, osl=16):
    declared = replace(case, runtime=replace(case.runtime, metadata=metadata))
    tree = HierarchicalFabricTree.from_hardware(declared.machine)
    return pool_config_facts(
        declared, declared.runtime.parallel, phase="decode", isl=256,
        osl=osl, prefix_hit_rate=0.0, tpot_slo_ms=tpot,
        hccl_buffsize_mb=8, routing_skew=1.0, eplb_redundant_experts=0,
        enable_eplb=False, tree=tree, rho=0.9)


def test_f06_auxiliary_weight_draft_execution_is_priced():
    case = _draft_case()
    speculative = {"speculative": True, "mtp_acceptance_length": 3}
    cheap = _decode_row(case, {**speculative, "mtp_draft_layer_cost_ms": 1.0},
                        tpot=1e9)
    pricey = _decode_row(case, {**speculative, "mtp_draft_layer_cost_ms": 10_000.0},
                         tpot=1e9)
    assert cheap["feasible"] and cheap["shared_draft"] is False
    assert pricey["feasible"]
    assert pricey["step_ms"] - cheap["step_ms"] == pytest.approx(
        3 * (10_000.0 - 1.0), rel=1e-12)
    assert cheap["draft_execution_source"] == "declared_layer_cost"
    tight = _decode_row(case, {**speculative,
                               "mtp_draft_layer_cost_ms": 10_000.0}, tpot=1.0)
    assert tight["feasible"] is False
    assert tight["reason"] == "decode_tpot_sla_or_memory_exceeded"
    priced = _decode_row(case, speculative)
    assert priced["feasible"]
    assert priced["draft_execution_source"] == "priced_draft_spec"
    assert priced["draft_step_ms"] > 0.0
    assert cheap["step_ms"] > priced["step_ms"]


def test_f06_short_output_and_per_batch_memory():
    cycle = 0.27430782476190474
    conserved = completed_requests_per_s(
        8, 2.734375, 1, cycle, speculative_k=2)
    assert conserved == pytest.approx(29_164.3156987734, rel=1e-9)
    assert conserved <= 8 / (cycle / 1000.0)
    assert completed_requests_per_s(12, 1.0, 1024, 13.916234315833) == pytest.approx(
        0.84209202964247, rel=1e-12)
    short = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0, max_seqs=8, max_model_len=4096)
    row = _decode_row(
        short, {"speculative": True, "mtp_acceptance_length": 3,
                "mtp_shared_weights": True, "mtp_draft_layer_cost_ms": 0.001},
        osl=1)
    assert row["feasible"] and row["useful_tokens_per_step"] > 1.0
    assert row["mu_requests_per_s"] == pytest.approx(
        row["batch"] / (row["step_ms"] / 1000.0), rel=1e-9)
    assert row["mu_requests_per_s"] < 79_746.17573883351

    def memory(batch):
        return float(batch * 40), 100.0

    record = mtp_batch_reversal(lambda batch: float(batch), draft_len=1,
                                accept_rate=1.0, max_batch=4,
                                memory_at_batch=memory)
    assert record.memory_qualified is False and record.crossings == (1, 2)
    assert [row[0] for row in record.memory_by_batch] == [1, 2, 3, 4]
    assert [row[3] for row in record.memory_by_batch] == [True, True, False, False]
    assert record.itl_basis == "analytic_cycle_mean"
    assert record.modeled_cycle_ms == pytest.approx(record.burst_itl_ms)
    scalar = mtp_batch_reversal(
        lambda batch: float(batch), draft_len=1, accept_rate=1.0, max_batch=4,
        draft_rollback_bytes_per_rank=10, rank_hbm_bytes=90,
        usable_capacity_bytes=100)
    assert scalar.memory_qualified is True


# F07: transfer conservation, page coverage, identity and executed bridges.
def test_f07_sparse_ledger_and_partial_page_coverage():
    transfer = logical_transfer_bytes(_sparse_case(), prompt_tokens=1024,
                                      shard_ranks=4, page_size=128)
    assert transfer["paged_kv_bytes"] == pytest.approx(1_179_648.0)
    assert transfer["index_k_bytes"] == pytest.approx(133_120.0)
    assert transfer["metadata_bytes"] == pytest.approx(512.0)
    assert transfer["total_bytes"] == pytest.approx(1_313_280.0)
    assert transfer["shard_bytes_aggregate"] == pytest.approx(1_312_768.0)
    assert transfer["transfer_total_bytes"] == pytest.approx(1_313_280.0)
    assert transfer["transfer_total_bytes"] == pytest.approx(
        sum(transfer["shard_bytes"])
        + sum(transfer["state_shard_bytes"])
        + transfer["transfer_metadata_bytes"])
    assert transfer["transfer_total_bytes"] != pytest.approx(1_446_400.0)
    case = _case(layers=4, tp=4, dp=1, ep=1, experts=0, top_k=0, moe_inter=0)
    partial = logical_transfer_bytes(case, prompt_tokens=129,
                                     destination_cached_tokens=1,
                                     page_size=128, prefix_digest="c" * 64)
    assert partial["missing_tokens"] == 128
    assert partial["page_ids"] == (0, 1)
    assert partial["transfer_pages"] == 2
    assert partial["transfer_metadata_bytes"] == pytest.approx(2 * 64 * 4)
    logical_missing = partial["paged_kv_bytes"] + partial["index_k_bytes"]
    assert partial["shard_bytes_aggregate"] >= logical_missing
    assert partial["transfer_total_bytes"] == pytest.approx(
        partial["shard_bytes_aggregate"]
        + partial["fixed_state_transfer_bytes"]
        + partial["transfer_metadata_bytes"])


def test_f07_prefix_identity_and_epoch_are_content_evidence():
    case = _case(experts=0, top_k=0, moe_inter=0, ep=1)
    unverified = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=512)
    assert unverified["destination_cached_tokens"] == 0
    assert unverified["prefix_identity_verified"] is False
    assert unverified["prefix_identity_reason"] == "prefix_content_identity_missing"
    verified = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=512,
        prefix_digest="a" * 64)
    assert verified["destination_cached_tokens"] == 512
    assert verified["prefix_identity_verified"] is True
    assert verified["prefix_content_hash"] == "a" * 64
    assert len(verified["prefix_hash"]) == 64
    other = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=512,
        prefix_tokens=(1, 2, 3, 4))
    first = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=512,
        prefix_tokens=(5, 6, 7, 8))
    assert other["prefix_content_hash"] != first["prefix_content_hash"]
    mismatched = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=512,
        source_prefix_hash="a" * 64, destination_prefix_hash="b" * 64)
    assert mismatched["destination_cached_tokens"] == 0
    assert mismatched["prefix_identity_verified"] is False
    stale = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=512,
        prefix_digest="a" * 64, source_epoch=1, destination_epoch=2)
    assert stale["destination_cached_tokens"] == 0
    assert stale["epoch_matched"] is False
    assert stale["prefix_identity_reason"] == "cache_epoch_mismatch"


def test_f07_pcp_shards_query_work_and_bridge_covers_all_source_lanes():
    dense = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                  moe_inter=0, pcp=1)
    one = prefill_step_ms(dense, isl=256, prefix_hit_rate=0.0, batch=32)
    two_case = replace(dense, runtime=replace(
        dense.runtime, parallel=ParallelSpec(tp=2, dp=1, ep=1, pcp=2,
                                             cp_plan="pcp")))
    two = prefill_step_ms(two_case, isl=256, prefix_hit_rate=0.0, batch=32)
    assert two["compute_ms"] < one["compute_ms"]
    assert two["collective_ms"] >= one["collective_ms"]
    assert one["step_ms"] != two["step_ms"]
    assert all(builder.query_owner == "query_dual_chunk_swap"
               for builder in registered_builders()
               if builder.phase == "prefill" and builder.plan == "pcp")
    bridge = phase_bridge(
        ParallelSpec(tp=8, pcp=4),
        ParallelSpec(tp=8, pcp=2, cp=2, pcp_lane_policy="replica_lanes"),
        kv_bytes=1 << 20, index_bytes=1 << 12, state_bytes=1 << 10)
    # Absolute block interleave preserves both source and destination owners.
    # Every source and every destination lane participates.
    assert bridge.lane_map == ((0, 0), (1, 1), (2, 0), (3, 1))
    assert set(bridge.destination_lanes) == {dest for _src, dest in bridge.lane_map}
    remap = bridge_remap_bytes(bridge)
    assert remap["metadata_bytes"] == pytest.approx(8192.0)
    assert remap["state_bytes"] == pytest.approx(1024.0)
    assert remap["total_bytes"] == pytest.approx(
        (1 << 20) + (1 << 12) + (1 << 10) + 8192.0)
    solver_case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                        moe_inter=0, num_devices=6, max_seqs=8)
    best = solve_xpyd_pools(
        solver_case, total_npus=6, isl=256, osl=16, ttft_slo_ms=1e9,
        tpot_slo_ms=1e9, prefill_configs=[ParallelSpec(tp=2, pcp=1)],
        decode_configs=[ParallelSpec(tp=2, pcp=2,
                                     pcp_lane_policy="replica_lanes")],
        hccl_buffsize_mb=8)["best_configuration"]
    assert best is not None and best["remap_ms"] > 0.0
    assert best["remap_bridge"]["lane_map"]


# F08: explicit tables, validation before coercion, Mode-B fit forwarding.
def test_f08_explicit_table_and_pre_coercion_validation():
    case = _case(layers=4, tp=2, dp=1, ep=2, pp=2, num_devices=8)
    candidate, plan, reason = validate_placement(
        case, ParallelSpec(tp=2, dp=1, ep=2, pp=2,
                           rank_table=((0, 1), (2, 3))))
    assert reason is None and candidate is not None and plan is not None
    assert plan["world"] == 4
    assert plan["rank_table"] == [[0, 1], [2, 3]]
    assert plan["edp_groups"] == []
    _, bad_plan, bad_reason = validate_placement(
        case, ParallelSpec(tp=2, dp=1, ep=2, pp=2,
                           rank_table=((0, 2), (1, 3))))
    assert bad_plan is None and "crosses_pp_stage" in (bad_reason or "")
    case = _case(layers=4, tp=1, dp=2, ep=2, experts=32, top_k=2)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    rejecting = (
        lambda slots: sweep_ep_knee(
            case, ep_candidates=(2,), batch_sequences=8, context_tokens=1024,
            hccl_buffsize_mb=8, eplb_redundant_slots_per_rank=slots),
        lambda slots: pool_config_facts(
            case, ParallelSpec(tp=1, dp=2, ep=2), phase="decode", isl=256,
            osl=16, prefix_hit_rate=0.0, tpot_slo_ms=1000.0,
            hccl_buffsize_mb=8, routing_skew=1.0, eplb_redundant_experts=1,
            enable_eplb=True, tree=tree, rho=0.9,
            eplb_redundant_slots_per_rank=slots),
        lambda slots: solve_xpyd_pools(
            case, total_npus=4, isl=256, osl=16, ttft_slo_ms=1000.0,
            tpot_slo_ms=1000.0, hccl_buffsize_mb=8,
            eplb_redundant_experts=1, eplb_redundant_slots_per_rank=slots),
        lambda slots: analyze_memory_breakdown(
            case, sequences=1, sequence_tokens=1024, hccl_buffsize_mb=0,
            enable_eplb=True, eplb_redundant_experts=1,
            eplb_redundant_slots_per_rank=slots))
    for entry in rejecting:
        for bad in ((0.9, 0.9), (0, 1.9)):
            with pytest.raises(ModelingError):
                entry(bad)
    integral = sweep_ep_knee(case, ep_candidates=(2,), batch_sequences=8,
                             context_tokens=1024, hccl_buffsize_mb=8,
                             eplb_redundant_experts=1,
                             eplb_redundant_slots_per_rank=(0.0, 1.0))
    assert integral["rows"] and "eplb" not in str(integral["rows"][0].get("reason"))


def test_f08_mode_b_fit_conditions_and_operation_are_forwarded():
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "fit_conditions": {"dtype": "bf16", "topology": "a3_16p"},
        "points": [{"payload_bytes": 0, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001},
                   {"payload_bytes": 1 << 30, "alpha_us": 2.0,
                    "beta_ns_per_byte": 0.001}]}]})
    case = _case(layers=4, tp=1, dp=2, ep=2)
    matched = sweep_ep_knee(case, ep_candidates=(2,), batch_sequences=8,
                            context_tokens=1024, fabric=profile,
                            hccl_buffsize_mb=8)
    assert matched["status"] == "feasible", matched["rows"]
    assert matched["ep_star"] == 2
    mismatched = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "fit_conditions": {"topology": "a5_pod"},
        "points": [{"payload_bytes": 0, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001},
                   {"payload_bytes": 1 << 30, "alpha_us": 2.0,
                    "beta_ns_per_byte": 0.001}]}]})
    rejected = sweep_ep_knee(case, ep_candidates=(2,), batch_sequences=8,
                             context_tokens=1024, fabric=mismatched,
                             hccl_buffsize_mb=8)
    assert rejected["status"] == "all_candidates_infeasible"
    assert "fit_condition_mismatch" in str(rejected["rows"][0].get("reason"))
