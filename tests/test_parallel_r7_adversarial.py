"""Round-7 adversarial regressions R7-01..R7-08.

[Role]
  Re-execute every Round-7 counterexample against its architectural root cause:
  an operation-derived speculative traffic ledger, physical-resource PP hop
  serialization with startup, actual-payload draft qualification, per-layer
  sparse index collectives, strict page-plan coverage plus bridge-lifetime HBM
  admission, one canonical speculative PMF, complete endpoint maps and
  versioned communicator-arena qualification.
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

import pytest

from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.fabric_topology import (CalibratedFabricProfile, FabricTier,
                                       HierarchicalFabricTree)
from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.moe_mtp import (mtp_batch_reversal, speculative_accept_tokens,
                                speculative_step_pmf)
from mindie_sim.parallel_bridge import (bridge_lifetime_plan, phase_bridge,
                                        validate_page_plan)
from mindie_sim.parallel_context import pcp_index_k_gather_bytes
from mindie_sim.parallel_cuts import request_phase_demands
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_ep_knee import prefill_step_ms
from mindie_sim.parallel_placement import build_placement, with_world
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_instances import pool_config_facts, priced_draft_step_ms
from mindie_sim.xpyd_pool_solver import logical_transfer_bytes, solve_xpyd_pools
from mindie_sim.xpyd_transfer import expected_decode_cycles

MIB = 1024 * 1024


def _dense(*, layers=4, hidden=2048, heads=16, kv_heads=4, experts=0, top_k=0,
           moe_inter=0, tp=1, dp=1, ep=1, pp=1, pcp=1, cp=1,
           num_devices=4, max_seqs=8, max_model_len=4096) -> object:
    return parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": num_devices, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "r7-adversarial", "layers": layers,
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
                    "graph_capture_mode": "none"},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]}})


def _sparse(*, layers=1, index_dim=128, index_heads=16, topk=512) -> object:
    plan = {"attention": "mla", "ffn": "dense", "architecture": "mla_dsa",
            "indexer": True, "indexer_type": "c4", "index_topk": topk,
            "owns_index_k": True, "index_dim": index_dim,
            "index_heads": index_heads, "num_heads": 16, "num_kv_heads": 4,
            "head_dim": 128, "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64, "kv_lora_rank": 512, "q_lora_rank": 512,
            "v_head_dim": 128, "compress_ratio": 0, "window": 4096,
            "cache_kind": "dense"}
    case = parse_case({
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 32, "ranks_per_node": 16,
                    "topology_preset": "a3_16p"},
        "model": {"name": "r7-sparse", "layers": layers, "hidden_size": 2048,
                  "num_heads": 16, "num_kv_heads": 4, "head_dim": 128,
                  "intermediate_size": 4096, "vocab_size": 32000,
                  "attention": "mla", "ffn": "dense",
                  "layer_plans": [{**plan, "index": index}
                                  for index in range(layers)]},
        "runtime": {"tp": 8, "dp": 2, "ep": 1, "pp": 1, "pcp": 2, "cp": 1,
                    "max_model_len": 8192, "max_num_seqs": 16,
                    "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.9, "kv_block_size": 128,
                    "graph_capture_mode": "none"},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]}})
    return replace(case, model=replace(case.model, layer_plans=tuple(
        replace(layer, intermediate_size=4096)
        for layer in case.model.layer_plans)))


def _link(name, members, *, capacity=1e9, latency=0.0, switch=None):
    return Link(name, capacity, tuple(members), latency_us=latency,
                channel="d2d_intra_node", semantics="per_port",
                switch_capacity_bytes_per_ms=switch)


def _machine_with(case, links):
    return replace(case.machine, links=tuple(links), topology_options=(),
                   dies_per_package=1)


def _profile(anchors, unmeasured=None):
    return CalibratedFabricProfile.from_probe({"tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 64,
        "points": [{"payload_bytes": payload, "alpha_us": 0.0,
                    "beta_ns_per_byte": 0.001,
                    "load_basis": ("unmeasured" if payload == unmeasured
                                   else "isolated")}
                   for payload in anchors]}]})


def _draft(dim=1024):
    return DraftSpec(kind="mtp", layers=1, hidden_size=dim,
                     intermediate_size=2 * dim, num_heads=8, num_kv_heads=2,
                     head_dim=128, q_width=dim, kv_width=dim // 4,
                     vocab_size=32000)


def _speculative(case, draft, *, k=3, rate=1.0, bonus=True, cost=0.001):
    metadata = {"speculative": True, "mtp_acceptance_length": k,
                "mtp_shared_weights": True, "mtp_accept_rate": rate,
                "bonus_token": bonus}
    if cost:
        metadata["mtp_draft_layer_cost_ms"] = cost
    model = (replace(case.model, auxiliary=AuxiliarySpec(mtp=draft))
             if draft is not None else case.model)
    return replace(case, model=model,
                   runtime=replace(case.runtime, metadata=metadata))


# R7-01: the speculative cycle ledger is the executed 1+K target rows and K
# draft steps, normalized by committed tokens per cycle.
def test_r7_01_speculative_switch_demand_matches_executed_rows():
    case = _dense(layers=4, num_devices=16, max_seqs=8, tp=1, dp=1)
    case = replace(case, machine=_machine_with(
        case, [_link("one", range(16), switch=1000.0)]))
    case = _speculative(case, None, k=3, rate=1.0, bonus=True)
    report = solve_xpyd_pools(
        case, total_npus=15, isl=256, osl=16, prefix_hit_rate=1.0,
        destination_cached_tokens=256, prefix_digest="d" * 64,
        prefill_configs=[ParallelSpec(tp=1)],
        decode_configs=[ParallelSpec(tp=2)], max_pool_instances=7,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9, hccl_buffsize_mb=8)
    best = report["best_configuration"]
    assert best is not None and report["qualified"] is True
    demand = best["shared_cut_demands"]["switch:intra_node_hccs"]
    # Target verification plus 4 cycles * 3 one-layer shared draft TP calls.
    assert demand == pytest.approx(524_288.0 + 98_304.0, rel=1e-12)
    assert best["shared_cut_limiting_resource"] == "switch:intra_node_hccs"
    assert best["shared_cut_rps"] == pytest.approx(
        1_000_000.0 / 622_592.0, rel=1e-12)
    assert best["admitted_rps"] <= 1.9073486328125 + 1e-12
    assert best["admitted_rps"] * demand <= 1_000_000.0 + 1e-6
    assert best["admitted_rps"] != pytest.approx(4.005100074533317)
    # Steady-state normalization: bytes per cycle / committed tokens per cycle
    # equals bytes per request / OSL exactly for this complete-cycle witness.
    per_token = best["shared_cut_bytes_per_output_token"][
        "switch:intra_node_hccs"]
    assert per_token == pytest.approx(622_592.0 / 16)
    assert per_token == pytest.approx(demand / 4.0 / 4.0)
    row = report["decode_configs"][0]
    assert row["verification_rows_per_cycle"] == 4
    assert row["draft_steps_per_cycle"] == 3
    assert row["decode_cycles"] == pytest.approx(4.0)


def test_r7_01_draft_collectives_enter_the_same_ledger():
    case = _dense(layers=4, num_devices=2, max_seqs=8, tp=2, dp=1)
    draft = _draft()
    case = _speculative(case, draft, k=2, rate=1.0, bonus=True, cost=0.001)
    profile = _profile((0, 1024, 16_384, 32_768, 65_536, 131_072, 1 << 20,
                        1 << 30))
    row = pool_config_facts(
        case, case.runtime.parallel, phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=profile, rho=0.9)
    assert row["feasible"] is True
    # A declared duration supplies no communication evidence: the priced draft
    # still contributes its actual per-layer cut bytes.
    assert row["draft_execution_source"] == "declared_layer_cost"
    draft_cut = row["draft_comm_demands"]["tier:intra_node_hccs"]
    assert draft_cut > 0.0
    contribution = (draft_cut / float(row["batch"]) * float(row["draft_len"])
                    * float(row["decode_cycles"]))
    assert row["shared_cut_demands"]["tier:intra_node_hccs"] >= (
        contribution - 1e-9)


# R7-02: physical resource identity, not logical stage index, serializes hops;
# startup precedes the slowest wire/switch/endpoint term.
def test_r7_02_shared_switch_serializes_every_pp_boundary():
    case = _dense(layers=6, pp=3, num_devices=3, tp=1, dp=1)
    case = replace(case, machine=_machine_with(
        case, [_link("one", range(3), switch=1000.0)]))
    decode = DecodeCostModel.from_case(case, context_tokens=1024).step(
        8, context=1024, microbatches=4)
    prefill = prefill_step_ms(case, isl=256, prefix_hit_rate=0.0, batch=8,
                              microbatches=4)
    # Four microbatches x two boundaries x 8192 B (decode) on 1000 B/ms.
    decode_lower = 8 * 8192.0 / 1000.0
    prefill_lower = 8 * (2 * 256 * 2048 * 2) / 1000.0
    assert decode.pp_p2p_ms == pytest.approx(decode_lower)
    assert prefill["pp_p2p_ms"] == pytest.approx(prefill_lower)
    assert decode.step_ms >= decode_lower
    assert prefill["step_ms"] >= prefill_lower
    # Both logical boundaries (0->1 and 1->2) share one physical switch.
    assert decode.step_ms >= 65.536
    assert prefill["step_ms"] >= 16_777.216


def test_r7_02_peer_transfer_keeps_startup_above_a_slow_switch():
    tree = HierarchicalFabricTree((FabricTier(
        "intra_node_hccs", 2, 1000.0, 1e6, semantics="per_port",
        switch_capacity_bytes_per_ms=1.0),))
    route = tree.peer_transfer(32_768, (0, 1))
    assert route.total_us == pytest.approx(32_768_000.0 + 1000.0)
    resources = dict(tree.peer_resources(32_768, (0, 1)))
    assert resources["switch:intra_node_hccs"] == pytest.approx(32_768_000.0)
    assert resources["port:0:egress"] > 0.0
    assert resources["port:1:ingress"] > 0.0


def test_r7_02_pp_hop_bytes_are_in_pool_cut_accounting():
    case = _dense(layers=6, pp=3, num_devices=3, tp=1, dp=1, max_seqs=1)
    case = replace(case, machine=_machine_with(
        case, [_link("one", range(3), switch=1000.0)]))
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    plan = build_placement(case, case.runtime.parallel).to_dict()
    demands = request_phase_demands(
        tree, plan, (0, 1, 2), tokens=1.0, hidden=2048, element_bytes=2.0,
        layers=6, top_k=0, pp=3)
    # Two adjacent boundaries, each moving one 2048*2 B activation per token.
    assert demands["switch:intra_node_hccs"] == pytest.approx(2 * 4096.0)
    speculative = request_phase_demands(
        tree, plan, (0, 1, 2), tokens=4.0, hidden=2048, element_bytes=2.0,
        layers=6, top_k=0, pp=3)
    assert speculative["switch:intra_node_hccs"] == pytest.approx(2 * 4 * 4096.0)


# R7-03: a scalar draft duration never bypasses actual-payload qualification.
def test_r7_03_declared_draft_cost_still_requires_measured_route():
    case = _dense(layers=4, num_devices=4, tp=2, dp=1, max_seqs=8)
    draft = _draft()
    profile = _profile((0, 1024, 16_384, 24_576, 32_768, 65_536, 131_072,
                        1 << 30), unmeasured=16_384)
    for cost in (0.0, 0.001):
        case = _speculative(case, draft, k=3, rate=1.0, bonus=True, cost=cost)
        direct = priced_draft_step_ms(
            case, draft, batch=8, context_tokens=272, fabric=profile,
            endpoints=(0, 1), groups=((0, 1),))
        assert direct is not None
        assert direct.payload_bytes == pytest.approx(16_384.0)
        assert direct.route_qualified is False
        assert direct.route_evidence == "unmeasured_profile"
        row = pool_config_facts(
            case, case.runtime.parallel, phase="decode", isl=256, osl=16,
            prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
            routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
            tree=profile, rho=0.9)
        assert row["feasible"] is False
        assert row["reason"] == "fabric_route_unqualified"


# R7-04: each sparse layer issues its own index gather at that layer's payload;
# same-communicator collectives are summed and recorded on the cut.
def test_r7_04_per_layer_index_gathers_qualify_and_sum():
    sparse = with_world(_sparse(), ParallelSpec(tp=1, dp=1, pcp=2))
    plans = tuple(replace(sparse.model.layer_plans[0], index=index,
                          index_dim=dim)
                  for index, dim in enumerate((128, 256)))
    case = replace(sparse, model=replace(sparse.model, layers=2,
                                         layer_plans=plans),
                   runtime=replace(sparse.runtime, pp_layer_partition=(2,),
                                   max_num_seqs=1))
    profile = _profile((0, 1024, 131_072, 196_608, 262_144, 393_216, 524_288,
                        1_048_576, 2_097_152, 4_194_304, 1 << 30),
                       unmeasured=131_072)
    payloads = [
        pcp_index_k_gather_bytes(1024, 2, dim, 2.0, top_k=512)["received_bytes"]
        for dim in (128, 256)]
    assert payloads == [131_072.0, 262_144.0]
    assert profile.collective("all_gather", payloads[0], 2,
                              endpoints=(0, 1)).route_qualified is False
    assert profile.collective("all_gather", payloads[1], 2,
                              endpoints=(0, 1)).route_qualified is True
    step = prefill_step_ms(case, isl=1024, prefix_hit_rate=0.0, batch=1,
                           fabric=profile)
    assert step["route_qualified"] is False
    assert step["route_evidence"] == "unmeasured_profile"
    row = pool_config_facts(
        case, case.runtime.parallel, phase="prefill", isl=1024, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=profile, rho=0.9)
    assert row["feasible"] is False and row["route_qualified"] is False


def test_r7_04_index_service_is_additive_and_on_the_cut():
    class Spy:
        def __init__(self, tree):
            self.tree = tree
            self.calls = []

        def __getattr__(self, name):
            return getattr(self.tree, name)

        def collective(self, kind, payload, ranks, **kwargs):
            timing = self.tree.collective(kind, payload, ranks, **kwargs)
            self.calls.append((kind, payload, timing.total_us))
            return timing

    tree = HierarchicalFabricTree((FabricTier(
        "intra_node_hccs", 64, 0.0, 1.0, semantics="per_port"),))
    sparse = with_world(_sparse(), ParallelSpec(tp=1, dp=1, pcp=2))
    exposed = []
    for dim in (128, 256, 512):
        plan = replace(sparse.model.layer_plans[0], index_dim=dim,
                       index_heads=16)
        case = replace(sparse, model=replace(sparse.model, layer_plans=(plan,)))
        spy = Spy(tree)
        step = prefill_step_ms(case, isl=1024, prefix_hit_rate=0.0, batch=1,
                               fabric=spy)
        exposed.append(step["step_ms"] - step["compute_ms"])
        assert step["collective_ms"] == pytest.approx(exposed[-1], rel=1e-9)
    assert exposed[1] - exposed[0] == pytest.approx(0.065536, rel=1e-9)
    assert exposed[2] - exposed[1] == pytest.approx(0.131072, rel=1e-9)

def test_r7_04_index_bytes_are_recorded_on_the_pcp_cut():
    sparse = with_world(_sparse(), ParallelSpec(tp=1, dp=1, pcp=2))
    sparse = replace(sparse, machine=_machine_with(
        sparse, [_link("one", range(2), switch=1000.0)]))
    one = replace(sparse, model=replace(sparse.model, layer_plans=(
        replace(sparse.model.layer_plans[0], index_dim=128),)))
    no_index = replace(one, model=replace(one.model, layer_plans=(
        replace(one.model.layer_plans[0], index_dim=0, index_topk=0,
                index_heads=0, indexer=False, owns_index_k=False,
                indexer_type="none"),)))
    priced = prefill_step_ms(one, isl=1024, prefix_hit_rate=0.0, batch=1)
    without = prefill_step_ms(no_index, isl=1024, prefix_hit_rate=0.0, batch=1)
    key = "switch:intra_node_hccs"
    assert priced["shared_cut_demands"][key] - without[
        "shared_cut_demands"][key] == pytest.approx(131_072.0)


# R7-05: none/empty ownership evidence is distinguished; zero transfer needs
# complete destination coverage.
def test_r7_05_explicit_empty_and_incomplete_pages_reject():
    case = _dense(tp=1, dp=1, num_devices=4)
    for source, destination in (
            (tuple(range(8)), ()), ((0, 1), (0, 1)), ((), (99,)), ((), ())):
        with pytest.raises(ModelingError):
            logical_transfer_bytes(
                case, prompt_tokens=1024, destination_cached_tokens=1024,
                prefix_digest="d" * 64, source_page_ids=source,
                destination_page_ids=destination)
    with pytest.raises(ModelingError):
        validate_page_plan((0, 1, 2, 3, 4, 5, 6, 7), (), required_pages=8)
    with pytest.raises(ModelingError):
        phase_bridge(ParallelSpec(tp=1), ParallelSpec(tp=1), kv_bytes=1.0,
                     page_ids=tuple(range(8)),
                     source_page_ids=tuple(range(8)),
                     destination_page_ids=())
    implicit = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=1024,
        prefix_digest="d" * 64)
    assert implicit["zero_transfer_qualified"] is True
    explicit = logical_transfer_bytes(
        case, prompt_tokens=1024, destination_cached_tokens=1024,
        prefix_digest="d" * 64, source_page_ids=tuple(range(8)),
        destination_page_ids=tuple(range(8)))
    assert explicit["zero_transfer_qualified"] is True
    report = solve_xpyd_pools(
        case, total_npus=2, isl=1024, osl=16, prefix_hit_rate=1.0,
        destination_cached_tokens=1024, prefix_digest="d" * 64,
        source_page_ids=tuple(range(8)), destination_page_ids=(),
        prefill_configs=[ParallelSpec(tp=1)],
        decode_configs=[ParallelSpec(tp=1)], max_pool_instances=1,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9, hccl_buffsize_mb=8)
    assert report["qualified"] is False
    assert report["best_configuration"] is None
    assert report["kv_bytes_per_request"] == 0.0
    assert report["reason"].startswith("bridge_page_plan_destination_empty")


def test_r7_05_bridge_lifetime_rejects_transfer_above_hbm_headroom():
    case = _dense(layers=4, tp=2, dp=1, num_devices=2, max_seqs=1)
    bridge = phase_bridge(ParallelSpec(tp=1), ParallelSpec(tp=2),
                          kv_bytes=64 * MIB, page_ids=tuple(range(8)))
    lifetime = bridge_lifetime_plan(bridge)
    assert lifetime.phase_peak_bytes("decode") >= 64 * MIB

    def headroom(hccl_mb):
        return analyze_memory_breakdown(
            case, sequences=1, sequence_tokens=257,
            hccl_buffsize_mb=hccl_mb).headroom_bytes

    base_headroom = headroom(0)
    per_mib = base_headroom - headroom(1)
    assert per_mib > 0.0
    hccl = max(0.0, (base_headroom - 32_768.0) / per_mib)
    assert 0.0 <= headroom(hccl) < 65_536.0
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    common = dict(phase="decode", isl=256, osl=1, prefix_hit_rate=0.0,
                  tpot_slo_ms=1e9, hccl_buffsize_mb=hccl, routing_skew=1.0,
                  eplb_redundant_experts=0, enable_eplb=False, tree=tree,
                  rho=0.9)
    baseline = pool_config_facts(case, ParallelSpec(tp=2), **common)
    assert baseline["feasible"] is True
    assert baseline["memory"].headroom_bytes <= 65_536.0
    rejected = pool_config_facts(case, ParallelSpec(tp=2),
                                 transfer_plan=lifetime, **common)
    assert rejected["feasible"] is False
    assert rejected["memory"].headroom_bytes < 0.0
    assert rejected["kv_transfer_staging_bytes"] >= 64 * MIB
    # The same lifetime plan reaches the solver's per-instance admission.
    pool_case = _dense(layers=4, tp=2, dp=1, num_devices=4, max_seqs=1)
    transfer = logical_transfer_bytes(pool_case, prompt_tokens=1024)
    assert transfer["transfer_total_bytes"] > 0.0
    pool_head = analyze_memory_breakdown(
        pool_case, sequences=1, sequence_tokens=1026,
        hccl_buffsize_mb=0).headroom_bytes
    pool_slope = pool_head - analyze_memory_breakdown(
        pool_case, sequences=1, sequence_tokens=1026,
        hccl_buffsize_mb=1).headroom_bytes
    pool_hccl = max(0.0, (pool_head - 32_768.0) / pool_slope)
    report = solve_xpyd_pools(
        pool_case, total_npus=4, isl=1024, osl=1, ttft_slo_ms=1e9,
        tpot_slo_ms=1e9, prefill_configs=[ParallelSpec(tp=2)],
        decode_configs=[ParallelSpec(tp=2)], max_pool_instances=1,
        hccl_buffsize_mb=pool_hccl)
    assert report["qualified"] is False
    assert report["best_configuration"] is None
    assert any(not instance["feasible"]
               for row in report["rows"]
               for instance in row["decode_instance_rows"])


# R7-06: every probability override flows through the one canonical PMF and a
# scalar yield is checked against the known K/bonus support.
def test_r7_06_probability_override_uses_canonical_law():
    for bonus, expected in ((False, 1.5), (True, 1.75)):
        canonical = speculative_accept_tokens(
            0.5, 2, bonus_token=bonus, accept_probabilities=(0.5, 0.25))
        reversal = mtp_batch_reversal(
            lambda batch: 1.0, draft_len=2, accept_rate=0.5,
            accept_probabilities=(0.5, 0.25), bonus_token=bonus, max_batch=1)
        assert canonical == pytest.approx(expected)
        assert reversal.accepted_tokens_per_step == pytest.approx(expected)
    assert speculative_step_pmf(
        2, 0.5, bonus_token=False,
        accept_probabilities=(0.5, 0.25)) == pytest.approx((0.5, 0.5))
    for bad in ((0.5,), (0.25, 0.5), (0.5, 1.5), (0.5, True)):
        with pytest.raises(ModelingError):
            speculative_step_pmf(2, 0.5, accept_probabilities=bad)


def test_r7_06_scalar_yield_is_bounded_by_declared_heads():
    with pytest.raises(ModelingError):
        expected_decode_cycles(64, 1, None, 100.0, bonus_token=False)
    with pytest.raises(ModelingError):
        expected_decode_cycles(64, 1, None, 2.0, bonus_token=False)
    with pytest.raises(ModelingError):
        expected_decode_cycles(64, 0, None, 2.0)
    assert expected_decode_cycles(64, 1, None, 1.0,
                                  bonus_token=False) == pytest.approx(64.0)
    assert expected_decode_cycles(64, 1, None, 2.0,
                                  bonus_token=True) == pytest.approx(32.0)
    case = _dense(layers=4, tp=2, dp=1, num_devices=2, max_seqs=8)
    case = _speculative(case, None, k=1, rate=0.5, bonus=False)
    case = replace(case, runtime=replace(case.runtime, metadata={
        **dict(case.runtime.metadata), "mtp_useful_tokens_per_step": 100.0}))
    row = pool_config_facts(
        case, case.runtime.parallel, phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=HierarchicalFabricTree.from_hardware(case.machine), rho=0.9)
    assert row["feasible"] is False
    assert row["reason"] == "speculative_yield_exceeds_declared_heads"


# R7-07: every public path validates the complete endpoint map before timing or
# a zero-work return, and logical ranks are mapped through that same map.
def test_r7_07_endpoint_maps_validate_before_coercion_or_full_hit():
    case = _dense(layers=4, num_devices=4, tp=2, dp=1, max_seqs=8)
    draft = _draft()
    case = _speculative(case, draft, k=3, rate=1.0, bonus=True, cost=0.001)
    for bad in ((0,), (0, 0), (0, 99), (0.1, 1.9), (True, False)):
        with pytest.raises(ModelingError):
            priced_draft_step_ms(case, draft, batch=8, context_tokens=256,
                                 endpoints=bad, groups=((0, 1),))
        with pytest.raises(ModelingError):
            prefill_step_ms(case, isl=256, prefix_hit_rate=1.0, batch=8,
                            endpoints=bad)
    mapped = priced_draft_step_ms(
        case, draft, batch=8, context_tokens=256, endpoints=(2, 0),
        groups=((0, 1),))
    assert mapped is not None
    assert mapped.groups == ((2, 0),)
    assert mapped.route_qualified is True
    full_hit = prefill_step_ms(case, isl=256, prefix_hit_rate=1.0, batch=8,
                               endpoints=(0, 1))
    assert full_hit["q_tokens"] == 0
    assert full_hit["route_qualified"] is True


# R7-08: persistent communicator arenas are qualified only by explicit,
# versioned backing parameters; the default heuristic is reported unqualified.
def test_r7_08_comm_persistent_qualification_requires_versioned_arenas():
    case = _dense(layers=4, tp=2, dp=1, num_devices=2, max_seqs=1)
    default = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0)
    assert default.comm_persistent_qualified is False
    assert default.comm_persistent_basis == "heuristic_arena_sizing"
    assert default.qualified is True
    strict = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
        strict_memory_qualification=True)
    assert strict.comm_persistent_qualified is False
    assert strict.qualified is False
    assert strict.status == "memory_unqualified"
    assert strict.reason == "comm_persistent_unmeasured"
    versioned = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257,
        hccl_buffsize_bytes=8 * MIB, strict_memory_qualification=True)
    assert versioned.comm_persistent_qualified is True
    assert versioned.comm_persistent_basis == "versioned_hccl_buffsize_bytes"
    assert versioned.qualified is True
    unbacked = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
        comm_arena_qualified=True, strict_memory_qualification=True)
    assert unbacked.comm_persistent_qualified is False
    partial = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
        comm_arena_qualified=True, comm_arena_domains={"tp": MIB},
        strict_memory_qualification=True)
    assert partial.comm_persistent_qualified is False
    asserted = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
        comm_arena_qualified=True,
        comm_arena_domains={"tp": MIB, "etp": MIB},
        strict_memory_qualification=True)
    assert asserted.comm_persistent_qualified is True
    assert asserted.comm_persistent_basis == "explicit_comm_arena_domains"
    assert asserted.qualified is True
    with pytest.raises(ModelingError):
        analyze_memory_breakdown(
            case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
            comm_arena_qualified=True, comm_arena_domains={"bogus": MIB})
    for bad in (True, 1.5, -1):
        with pytest.raises(ModelingError):
            analyze_memory_breakdown(
                case, sequences=1, sequence_tokens=257,
                hccl_buffsize_bytes=bad)
    with pytest.raises(ModelingError):
        analyze_memory_breakdown(
            case, sequences=1, sequence_tokens=257,
            hccl_buffsize_bytes=8 * MIB, hccl_buffsize_mb=7)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    row = pool_config_facts(
        case, case.runtime.parallel, phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree, rho=0.9)
    assert row["feasible"] is True
    assert row["comm_persistent_qualified"] is False
    assert row["comm_persistent_basis"] == "heuristic_arena_sizing"
    pool_case = _dense(layers=4, tp=2, dp=1, num_devices=4, max_seqs=1)
    common = dict(total_npus=4, isl=256, osl=1, ttft_slo_ms=1e9,
                  tpot_slo_ms=1e9,
                  prefill_configs=[ParallelSpec(tp=2)],
                  decode_configs=[ParallelSpec(tp=2)],
                  max_pool_instances=1, hccl_buffsize_mb=0)
    relaxed = solve_xpyd_pools(pool_case, **common)
    assert relaxed["qualified"] is True
    strict_pool = solve_xpyd_pools(
        pool_case, strict_memory_qualification=True, **common)
    assert strict_pool["qualified"] is False
    strict_facts = pool_config_facts(
        pool_case, ParallelSpec(tp=2), phase="decode", isl=256, osl=1,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=0,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=HierarchicalFabricTree.from_hardware(pool_case.machine), rho=0.9,
        strict_memory_qualification=True)
    assert strict_facts["feasible"] is False
    assert strict_facts["reason"] == (
        "memory_unqualified:comm_persistent_unmeasured")
