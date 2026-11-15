"""Round-5 adversarial regressions F01..F08.

[Role]
  Re-execute every independent Round-5 counterexample against the mathematical
  root cause: proven EPLB envelopes and epoch binding, conserved PP work,
  limiting-cut queueing, all-group/shared-cut service, local-to-physical
  traffic mapping, finite-request renewal, conserved cache bytes and explicit
  orthogonal expert factorization.
[Invariants]
  Assertions are exact conservation identities, proven tight bounds or typed
  rejections; no assertion repeats the implementation's own intermediate.
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
from mindie_sim.moe_skew_eplb import (conservative_eplb_slots,
                                      max_rank_tokens, place_eplb,
                                      replicated_expert_bytes)
from mindie_sim.moe_mtp import mtp_batch_reversal
from mindie_sim.parallel_context import (bridge_remap_bytes, phase_bridge,
                                         pcp_attention_pairs)
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_ep_knee import (prefill_step_ms, sweep_ep_knee)
from mindie_sim.parallel_placement import validate_placement
from mindie_sim.parallel_schedule import pp_pipeline_ms
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_instances import pool_config_facts
from mindie_sim.xpyd_pool_solver import (kv_transfer_contention,
                                         logical_transfer_bytes,
                                         solve_xpyd_pools)
from mindie_sim.xpyd_transfer import (completed_requests_per_s,
                                      expected_decode_cycles)


def _case(*, sku="Ascend910_9362", topology="a3_16p", num_devices=64,
          ranks_per_node=16, layers=4, hidden=2048, heads=16, kv_heads=4,
          experts=32, top_k=4, moe_inter=512, tp=1, dp=2, ep=2, pp=1,
          pcp=1, cp=1, max_seqs=8, max_model_len=4096, graph="none"):
    return parse_case({
        "machine": {"backend": "ascend", "sku": sku, "num_devices": num_devices,
                    "ranks_per_node": ranks_per_node,
                    "topology_preset": topology},
        "model": {"name": "r5-adversarial", "layers": layers,
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


def _dp_link(name, members, capacity=1e6, **kwargs):
    return Link(name, capacity, tuple(members), latency_us=0,
                channel="d2d_intra_node", semantics="per_port", **kwargs)


def _sparse_case(*, index_topk=512):
    plan = {"attention": "mla", "ffn": "dense", "architecture": "mla_dsa",
            "indexer": True, "indexer_type": "c4", "index_topk": index_topk,
            "owns_index_k": True, "index_dim": 128, "index_heads": 16,
            "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "kv_lora_rank": 512, "v_head_dim": 128, "q_lora_rank": 512,
            "intermediate_size": 4096, "window": 4096, "cache_kind": "dense"}
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 32, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "r5-sparse", "layers": 1, "hidden_size": 2048,
                  "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
                  "intermediate_size": 4096, "vocab_size": 32000,
                  "attention": "mla", "ffn": "dense",
                  "layer_plans": [{**plan, "index": 0}]},
        "runtime": {"tp": 8, "dp": 2, "ep": 1, "pp": 1, "pcp": 2, "cp": 1,
                    "max_model_len": 8192, "max_num_seqs": 16,
                    "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.9, "kv_block_size": 128},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]}})


# F01: proven tight envelope and unconditional epoch equality.
def test_f01_proven_tight_envelope_and_same_batch_rank7_witness():
    assert place_eplb(13, 1, 8, 1.0, num_experts=16,
                      redundant_experts=5).redundant_slots_per_rank == (
        0, 0, 0, 0, 0, 0, 0, 5)
    assert conservative_eplb_slots(5, 8, 16, top_k=1) == (5,) * 8
    case = _case(layers=4, tp=8, dp=1, ep=8, experts=16, top_k=1,
                 max_seqs=16)
    replica = replicated_expert_bytes(case, 5)
    assert replica["per_rank_replicas"] == 5
    assert replica["per_rank_bytes"][7] == pytest.approx(125_829_120.0)
    dies = list(case.machine.dies)
    dies[7] = replace(dies[7], memory_bytes=math.floor(426_074_624 / 0.9))
    tight = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    memory = analyze_memory_breakdown(
        tight, sequences=13, sequence_tokens=1024, hccl_buffsize_mb=8,
        enable_eplb=True, eplb_redundant_experts=5)
    assert memory.per_rank_bytes[7][3] == pytest.approx(476_406_272.0)
    assert memory.budget_bytes == pytest.approx(426_074_623.2)
    assert memory.qualified is False and memory.status == "memory_infeasible"
    assert memory.worst_rank == 7


def test_f01_supplied_epoch_must_match_execution_and_respect_non_local_bound():
    case = _case(layers=4, tp=1, dp=8, ep=8, experts=16, top_k=1,
                 max_seqs=16)
    dies = list(case.machine.dies)
    dies[4] = replace(dies[4], memory_bytes=math.floor(665_164_287.9 / 0.9))
    tight = replace(case, machine=replace(case.machine, dies=tuple(dies)))
    bypass = sweep_ep_knee(
        tight, ep_candidates=(8,), batch_sequences=1, context_tokens=1024,
        hccl_buffsize_mb=8, enable_eplb=True, eplb_redundant_experts=5,
        eplb_redundant_slots_per_rank=(5, 0, 0, 0, 0, 0, 0, 0))
    assert bypass["qualified"] is False and bypass["ep_star"] is None
    assert bypass["rows"][0]["feasible"] is False
    assert bypass["rows"][0]["reason"] == "eplb_epoch_mismatch"
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    pooled = pool_config_facts(
        case, ParallelSpec(tp=1, dp=8, ep=8), phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=5, enable_eplb=True,
        tree=tree, rho=0.9,
        eplb_redundant_slots_per_rank=(5, 0, 0, 0, 0, 0, 0, 0))
    assert pooled["feasible"] is False
    assert pooled["reason"] == "eplb_epoch_mismatch"
    with pytest.raises(ModelingError):
        replicated_expert_bytes(case, 5, redundant_slots_per_rank=(
            15, 0, 0, 0, 0, 0, 0, 0))


# F02: conserved batch/microbatch work and physical PP hops.
def test_f02_prefill_pp_pays_only_stage_sum_plus_physical_hop():
    case = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, pp=2)
    step = prefill_step_ms(case, isl=256, prefix_hit_rate=0.0, batch=8,
                           microbatches=1)
    assert step["pp_fill_ms"] == pytest.approx(0.0)
    assert sum(step["pp_stage_ms"]) == pytest.approx(6.371669333333333)
    assert step["pp_p2p_ms"] > 0.0
    assert step["pp_pipeline_makespan_ms"] == pytest.approx(
        sum(step["pp_stage_ms"]) + step["pp_p2p_ms"])
    assert step["step_ms"] == pytest.approx(step["pp_pipeline_makespan_ms"])
    assert step["step_ms"] != pytest.approx(9.557504)


def test_f02_decode_microbatches_conserve_tokens_and_remove_phantom_bubble():
    case = _case(layers=8, tp=1, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, pp=1)
    model = DecodeCostModel.from_case(case)
    one = model.step(8, context=1024, microbatches=1)
    four = model.step(8, context=1024, microbatches=4)
    assert one.step_ms == pytest.approx(0.80571392)
    assert four.pp_pipeline_bubble_ms == pytest.approx(0.0)
    assert four.pp_pipeline_makespan_ms == pytest.approx(
        four.pp_serial_stage_ms)
    assert four.useful_tokens_per_step == one.useful_tokens_per_step == 8
    assert four.expert_flops == one.expert_flops
    assert four.pp_fill_drain_ms == pytest.approx(one.pp_fill_drain_ms / 4.0)
    assert four.step_ms > one.step_ms
    p2p = model.step(8, context=1024, microbatches=1, pp_p2p_ms=0.25)
    assert p2p.pp_p2p_ms == pytest.approx(one.pp_p2p_ms + 0.25)
    assert p2p.step_ms - one.step_ms == pytest.approx(0.25)
    assert pp_pipeline_ms((5.0, 5.0), 4)["makespan_ms"] == pytest.approx(25.0)
    assert pp_pipeline_ms((5.0, 5.0), 4)["bubble_ms"] == pytest.approx(15.0)
    assert pp_pipeline_ms((35.0, 5.0), 1)["makespan_ms"] == pytest.approx(40.0)


def test_f02_moe_microbatch_work_is_conserved():
    case = _case(layers=4, tp=2, dp=1, ep=2, experts=32, top_k=2, max_seqs=8)
    model = DecodeCostModel.from_case(case, context_tokens=1024)
    one = model.step(8, context=1024, microbatches=1)
    two = model.step(8, context=1024, microbatches=2)
    assert two.expert_flops == pytest.approx(one.expert_flops)
    assert two.useful_tokens_per_step == one.useful_tokens_per_step
    assert two.pp_pipeline_bubble_ms == pytest.approx(0.0)


# F03: queueing uses the limiting cut service, including RoCE.
def test_f03_roce_limiting_service_drives_md1_wait():
    handoff = kv_transfer_contention(
        1e6, n_prefill=1, n_decode=1, nic_bw_gbps=1,
        hbm_bw_bytes_per_s=1e12, p_world=1, d_world=1,
        decode_step_ms=1, osl=1, requests_per_s=0.9, roce_bw_gbps=0.001)
    assert handoff["limiting_cut_service_ms"] == pytest.approx(1000.0)
    assert handoff["kv_wire_service_ms"] == pytest.approx(1000.0)
    assert handoff["queue_ms"] == pytest.approx(4500.0)


def test_f03_rate_search_rejects_false_roce_slo_admission():
    case = _case(layers=4, hidden=1024, heads=8, kv_heads=2, experts=0,
                 top_k=0, moe_inter=0, tp=1, dp=1, ep=1, max_seqs=64,
                 max_model_len=16384)
    report = solve_xpyd_pools(
        case, total_npus=2, isl=8192, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=0, ttft_slo_ms=100_000.0,
        tpot_slo_ms=100_000.0, prefill_configs=[ParallelSpec()],
        decode_configs=[ParallelSpec()], hccl_buffsize_mb=64,
        max_pool_instances=1, nic_bw_gbps=1.0, roce_bw_gbps=0.001)
    best = report["best_configuration"]
    assert best is not None and best["slo_feasible"]
    assert best["ttft_ms"] <= 100_000.0
    assert best["transfer_ms"] == pytest.approx(33_570.816)
    assert best["admitted_rps"] < 0.0268089998170
    # The rate that the old wrong service time falsely admitted:
    false = kv_transfer_contention(
        33_570_816, n_prefill=1, n_decode=1, nic_bw_gbps=1.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=1, decode_step_ms=1.0,
        osl=1, requests_per_s=0.0268089998170, roce_bw_gbps=0.001)
    assert false["queue_ms"] == pytest.approx(151_068.6720015, rel=1e-9)


# F04: every synchronous group and one physical shared cut.
def test_f04_second_tp_group_is_timed_not_only_the_first():
    case = _case(tp=2, dp=2, num_devices=4, max_seqs=8)
    link = _dp_link("wide", (0, 1, 2, 3), capacity=1e9,
                    endpoint_rates=((3, 1.0),))
    case = replace(case, machine=_machine_with(case, [link]))
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    slow = tree.collective("all_reduce", 32_768, 2, endpoints=(2, 3))
    fast = tree.collective("all_reduce", 32_768, 2, endpoints=(0, 1))
    assert slow.total_us >= 32_768_000.0
    assert fast.total_us < 1.0
    row = pool_config_facts(
        case, ParallelSpec(tp=2, dp=2), phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1.0, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree, rho=0.9)
    assert row["feasible"] is False and row.get("step_ms") is None


def test_f04_shared_switch_is_one_cut_across_pool_instances():
    case = _case(layers=4, hidden=1024, heads=8, kv_heads=2, experts=0,
                 top_k=0, moe_inter=0, tp=1, dp=1, ep=1, max_seqs=64,
                 max_model_len=16384, num_devices=4)
    link = Link("shared", 1e9, (0, 1, 2, 3), latency_us=0,
                channel="d2d_intra_node", semantics="shared",
                switch_capacity_bytes_per_ms=1000.0)
    case = replace(case, machine=_machine_with(case, [link]))
    report = solve_xpyd_pools(
        case, total_npus=4, isl=8192, osl=1, prefix_hit_rate=1.0,
        destination_cached_tokens=0, ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec()], decode_configs=[ParallelSpec()],
        hccl_buffsize_mb=8, max_pool_instances=2)
    ceiling = 1_000_000.0 / report["kv_bytes_per_request"]
    assert ceiling == pytest.approx(0.0297877775744, rel=1e-12)
    feasible = [row for row in report["rows"] if row["feasible"]]
    assert feasible
    for row in feasible:
        assert row["kv_wire_rps"] == pytest.approx(ceiling, rel=1e-12)
        assert row["admitted_rps"] <= 0.9 * ceiling + 1e-12
    wide = next(row for row in feasible
                if row["n_prefill"] == 2 and row["n_decode"] == 2)
    assert wide["admitted_rps"] <= 0.9 * ceiling + 1e-12


# F05: local rows map to physical endpoints; matrix-only semantics.
def test_f05_local_index_maps_to_physical_endpoint_rate():
    tree = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="per_port",
                   endpoint_rates=((3, 1000.0),)),))
    clean = tree.collective("all_to_all_v", 1000, 2, endpoints=(2, 3))
    assert clean.route_qualified is True
    rated = HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 4, 0.0, 1.0, semantics="per_port",
                   endpoint_rates=((2, 1000.0),)),))
    incast = rated.collective(
        "all_to_all_v", 1000, 3, endpoints=(2, 0, 1),
        payloads_by_destination=(1000, 0, 0))
    assert incast.max_receiver_ingress_bytes == pytest.approx(2000.0)
    assert incast.total_us == pytest.approx(2000.0)


def test_f05_uniform_matrix_and_hierarchical_equivalence():
    tiered = HierarchicalFabricTree((
        FabricTier("intra_package_die", 2, 0, 0.1, semantics="per_port"),
        FabricTier("intra_node_hccs", 4, 0, 1.0, semantics="per_port")))
    implicit = tiered.collective("all_to_all_v", 1000, 4)
    explicit = tiered.collective("all_to_all_v", 1000, 4,
                                 payloads_by_destination=(250,) * 4)
    assert implicit.total_us == pytest.approx(3.0)
    assert explicit.total_us == pytest.approx(3.0)
    assert implicit.max_receiver_ingress_bytes == pytest.approx(750.0)
    assert explicit.max_receiver_ingress_bytes == pytest.approx(750.0)
    assert explicit.cut_bytes == pytest.approx(3000.0)


# F06: finite-OSL renewal, batch-aware draft price and finite callback capacity.
def test_f06_finite_osl_renewal_completion():
    assert expected_decode_cycles(4, 2, 1.0) == pytest.approx(2.0)
    assert expected_decode_cycles(1) == pytest.approx(1.0)
    assert completed_requests_per_s(
        8, 3.0, 4, 1.0, speculative_k=2, accept_rate=1.0) == pytest.approx(4000.0)
    for g in (1.0, 2.5, 3.0):
        cycles = expected_decode_cycles(17, 2, None, g)
        assert cycles >= math.ceil(17 / math.ceil(g))
    with pytest.raises(ModelingError):
        expected_decode_cycles(17, 2, None, 3.5)


def test_f06_speculative_pool_completion_uses_two_cycles():
    case = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=8, max_model_len=4096)
    declared = replace(case, runtime=replace(case.runtime, metadata={
        "speculative": True, "mtp_acceptance_length": 2,
        "mtp_shared_weights": True, "mtp_draft_layer_cost_ms": 0.001,
        "mtp_accept_rate": 1}))
    tree = HierarchicalFabricTree.from_hardware(declared.machine)
    row = pool_config_facts(
        declared, declared.runtime.parallel, phase="decode", isl=256, osl=4,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree, rho=0.9)
    assert row["feasible"] and row["batch"] == 8
    assert row["step_ms"] == pytest.approx(0.2639506285714286 + 2 * row["draft_step_ms"])
    assert row["mu_requests_per_s"] == pytest.approx(8 * 1000 / (2 * row["step_ms"]))


def test_f06_priced_draft_charges_the_admitted_batch():
    base = _case(layers=4, tp=2, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0, max_seqs=2048, max_model_len=4096)
    draft = DraftSpec(kind="mtp", layers=1, hidden_size=2048,
                      intermediate_size=4096, num_heads=16, num_kv_heads=4,
                      head_dim=128, q_width=2048, kv_width=512,
                      vocab_size=32000)
    case = replace(base, model=replace(
        base.model, auxiliary=AuxiliarySpec(mtp=draft)))
    declared = replace(case, runtime=replace(case.runtime, metadata={
        "speculative": True, "mtp_acceptance_length": 3}))
    tree = HierarchicalFabricTree.from_hardware(declared.machine)
    row = pool_config_facts(
        declared, declared.runtime.parallel, phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree, rho=0.9)
    assert row["feasible"] and row["batch"] == 2048
    assert row["draft_execution_source"] == "priced_draft_spec"
    assert row["draft_step_ms"] >= 0.19065018181818183


def test_f06_rank_callback_honours_finite_capacity():
    record = mtp_batch_reversal(
        lambda batch: float(batch), max_batch=2, batch_grid=(1, 2),
        rank_hbm_bytes_at=lambda batch: 1000 * batch,
        usable_capacity_bytes=1.0)
    assert record.memory_qualified is False
    assert record.crossings == ()
    assert [row[2] for row in record.memory_by_batch] == [1.0, 1.0]
    assert [row[3] for row in record.memory_by_batch] == [False, False]


# F07: conserved partial-cache bytes, identity gate, bridge coverage, sparse PCP.
def test_f07_page_aligned_partial_cache_is_conserved():
    case = _case(layers=4, tp=4, dp=1, ep=1, experts=0, top_k=0,
                 moe_inter=0)
    for cached, shard, metadata in ((512, 4_194_304.0, 1024.0),
                                    (896, 1_048_576.0, 256.0)):
        transfer = logical_transfer_bytes(
            case, prompt_tokens=1024, destination_cached_tokens=cached,
            page_size=128, shard_ranks=4, prefix_digest="d" * 64)
        assert transfer["shard_bytes_aggregate"] == pytest.approx(shard)
        assert transfer["transfer_metadata_bytes"] == pytest.approx(metadata)
        assert transfer["transfer_total_bytes"] == pytest.approx(
            shard + metadata)
        assert transfer["transfer_total_bytes"] == pytest.approx(
            sum(transfer["shard_bytes"]) + sum(transfer["state_shard_bytes"])
            + transfer["transfer_metadata_bytes"])
    unaligned = logical_transfer_bytes(
        case, prompt_tokens=129, destination_cached_tokens=1, page_size=128,
        shard_ranks=4, prefix_digest="d" * 64)
    assert unaligned["page_ids"] == (0, 1)
    assert unaligned["transfer_pages"] == 2
    assert unaligned["transfer_metadata_bytes"] == pytest.approx(512.0)
    assert unaligned["shard_bytes_aggregate"] >= (
        unaligned["paged_kv_bytes"] + unaligned["index_k_bytes"])


def test_f07_full_hit_identity_and_empty_winner_gate_qualification():
    case = _case(tp=1, dp=1, ep=1, experts=0, top_k=0, moe_inter=0,
                 num_devices=2, max_seqs=8)
    kwargs = dict(
        total_npus=2, isl=256, osl=16, prefix_hit_rate=1.0,
        destination_cached_tokens=256, ttft_slo_ms=100.0,
        tpot_slo_ms=100.0, prefill_configs=[ParallelSpec()],
        decode_configs=[ParallelSpec()], hccl_buffsize_mb=8,
        max_pool_instances=1)
    unverified = solve_xpyd_pools(case, **kwargs)
    assert unverified["qualified"] is False
    assert unverified["qualification_reason"] == "kv_cache_identity_unverified"
    best = unverified["best_configuration"]
    assert best is not None and best["slo_feasible"] is True
    assert best["qualified"] is False
    assert best["kv_cache_identity_verified"] is False
    strict = solve_xpyd_pools(case, require_prefix_identity=True, **kwargs)
    assert strict["best_configuration"] is None and strict["qualified"] is False
    assert strict["qualification_reason"] == "kv_cache_identity_unverified"
    verified = solve_xpyd_pools(case, prefix_digest="e" * 64, **kwargs)
    assert verified["qualified"] is True
    assert verified["best_configuration"]["kv_cache_identity_verified"] is True
    empty = solve_xpyd_pools(case, **{**kwargs, "decode_configs": []})
    assert empty["best_configuration"] is None and empty["qualified"] is False


def test_f07_bridge_covers_every_destination_lane_and_sparse_pcp_pairs():
    bridge = phase_bridge(ParallelSpec(tp=4, pcp=1),
                          ParallelSpec(tp=4, cp=4), kv_bytes=1_000_000,
                          page_ids=tuple(range(8)))
    assert bridge.lane_map == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert all(row == pytest.approx(250_000.0)
               for row in bridge.destination_lane_bytes)
    assert sorted(page for bucket in bridge.destination_pages
                  for page in bucket) == list(range(8))
    remap = bridge_remap_bytes(bridge)
    assert remap["total_bytes"] == pytest.approx(1_000_000 + 4 * 4096.0)
    sparse = _sparse_case()
    low = prefill_step_ms(sparse, isl=1024, prefix_hit_rate=0.0, batch=8)
    high_case = _sparse_case(index_topk=1)
    low_index = prefill_step_ms(high_case, isl=1024, prefix_hit_rate=0.0,
                                batch=8)
    assert low_index["compute_ms"] < low["compute_ms"]
    assert low_index["step_ms"] < low["step_ms"]
    assert pcp_attention_pairs(isl=1024, q_tokens=1024, pcp=2, family="dsa",
                               index_topk=512, sliding_window=4096
                               )["critical_pairs"] == 229_504
    assert pcp_attention_pairs(isl=1024, q_tokens=1024, pcp=2, family="dsa",
                               index_topk=1, sliding_window=4096
                               )["critical_pairs"] == 512


# F08: explicit orthogonal table, actual-payload qualification, strict integers.
def test_f08_explicit_table_factorizes_ep_etp_orthogonally():
    case = _case(layers=4, tp=4, dp=1, ep=2, experts=32, top_k=2,
                 num_devices=8)
    candidate, plan, reason = validate_placement(
        case, ParallelSpec(tp=4, dp=1, ep=2, expert_tp=2, expert_dp=1,
                           rank_table=((0, 1), (2, 3))))
    assert reason is None and candidate is not None and plan is not None
    assert plan["ep_groups"] == [[0, 1], [2, 3]]
    assert plan["etp_groups"] == [[0, 2], [1, 3]]
    for ep_group in plan["ep_groups"]:
        for etp_group in plan["etp_groups"]:
            assert len(set(ep_group) & set(etp_group)) <= 1
    bad_case = _case(layers=4, tp=2, dp=1, ep=2, experts=32, top_k=2,
                     pp=2, num_devices=8)
    _, bad, bad_reason = validate_placement(
        bad_case, ParallelSpec(tp=2, dp=1, ep=2, pp=2,
                               rank_table=((0, 2), (1, 3))))
    assert bad is None and "crosses_pp_stage" in (bad_reason or "")


def test_f08_actual_payload_unqualification_rejects_the_winner():
    profile = CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "fit_conditions": {"dtype": "bf16", "topology": "a3_16p"},
        "points": [{"payload_bytes": 0, "alpha_us": 0.0,
                    "beta_ns_per_byte": 1.0},
                   {"payload_bytes": 1024, "alpha_us": 0.0,
                    "beta_ns_per_byte": 1.0},
                   {"payload_bytes": 1 << 30, "alpha_us": 2.0,
                    "beta_ns_per_byte": 1.0, "load_basis": "unmeasured"}]}]})
    case = _case(tp=1, dp=2, ep=2, experts=32, top_k=4)
    report = sweep_ep_knee(
        case, ep_candidates=(2,), batch_sequences=8, context_tokens=1024,
        fabric=profile, hccl_buffsize_mb=8)
    assert report["qualified"] is False and report["ep_star"] is None
    assert report["status"] == "all_candidates_infeasible"
    row = report["rows"][0]
    assert row["feasible"] is False
    assert row["reason"] == "fabric_route_unqualified"
    assert row["route_evidence"] == "unmeasured_profile"


def test_f08_strict_integer_boundaries_and_owner_ceiling():
    with pytest.raises(ModelingError):
        sweep_ep_knee(_case(), ep_candidates=(2,), batch_sequences=8,
                      context_tokens=1024, hccl_buffsize_mb=8,
                      eplb_redundant_experts=0.9)
    assert max_rank_tokens(1, 3, 2, 1.0, num_experts=3) == 2.0
