"""Round-8 adversarial regressions R8-01..R8-06.

[Role]
  Re-execute every Round-8 counterexample against its architectural root cause:
  an operation-derived prefill/decode communication ledger consumed by pool
  admission, cut composition from the placed physical instances under
  capacity-proportional routing, all-axis draft payload qualification, page
  universe/ownership reconciliation, and true per-rank KV lifetime plus the
  complete EPLB bridge peak.
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

from test_parallel_r7_adversarial import (_dense, _draft, _link, _machine_with,
                                          _profile, _sparse, _speculative)

from mindie_sim.fabric_topology import FabricTier, HierarchicalFabricTree
from mindie_sim.memory_breakdown import analyze_memory_breakdown
from mindie_sim.moe_mtp import speculative_accept_tokens
from mindie_sim.parallel_bridge import (bridge_lifetime_plan, kv_rank_fraction,
                                        phase_bridge)
from mindie_sim.parallel_cuts import unique_cut_demands
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_placement import validate_placement, with_world
from mindie_sim.parallel_prefill_cost import prefill_step_ms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_cut_composer import (compose_pool_cut_demands,
                                         routed_rates_within_capacity)
from mindie_sim.xpyd_flow import queue_wait_ms
from mindie_sim.xpyd_instances import pool_config_facts, priced_draft_step_ms
from mindie_sim.xpyd_pool_solver import solve_xpyd_pools
from mindie_sim.xpyd_transfer import logical_transfer_bytes

MIB = 1024 * 1024


def _facts(case, phase, *, isl, osl, hit, tree=None, parallel=None, **kwargs):
    return pool_config_facts(
        case, parallel or case.runtime.parallel, phase=phase, isl=isl, osl=osl,
        prefix_hit_rate=hit, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=False,
        tree=tree or HierarchicalFabricTree.from_hardware(case.machine),
        rho=0.9, **kwargs)


def _moe_case(*, speculative=False):
    case = _dense(experts=32, top_k=4, moe_inter=4096, num_devices=16,
                  max_seqs=8)
    case = replace(case, machine=_machine_with(
        case, [_link("one", range(16), switch=1000.0)]))
    if speculative:
        case = _speculative(case, None, k=3, rate=1.0, bonus=True, cost=0.001)
    return with_world(case, ParallelSpec(tp=1, dp=2, ep=2))


# R8-01: decode MoE ledger contains both executed EP operations and pool
# admission consumes those exact bytes, never a single generic all_to_all_v.
def test_r8_01_decode_moe_ledger_charges_dispatch_and_combine():
    case = _moe_case()
    model = DecodeCostModel.from_case(case, context_tokens=272)
    step = model.step(4, per_replica_batch=4, active_replicas=2, context=272)
    roles = step.comm_ledger.role_payloads()
    assert roles["ep_dispatch"] > 0.0
    assert roles["ep_combine"] == pytest.approx(roles["ep_dispatch"])
    assert step.shared_cut_demands == tuple(sorted(
        step.comm_ledger.cut_demands().items()))
    row = _facts(case, "decode", isl=256, osl=16, hit=1.0)
    assert row["feasible"] is True
    # Per request over the request lifetime: 2*(EP-1)*top_k*H*2B*layers*OSL.
    expected = 2 * (2 - 1) * 4 * 2048 * 2 * 4 * 16
    assert row["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(float(expected), rel=1e-12))
    assert row["decode_cycles"] == pytest.approx(16.0)


def test_r8_01_moe_pool_never_exceeds_dispatch_plus_combine_capacity():
    for speculative in (False, True):
        case = _moe_case(speculative=speculative)
        report = solve_xpyd_pools(
            case, total_npus=15, isl=256, osl=16, prefix_hit_rate=1.0,
            destination_cached_tokens=256, prefix_digest="d" * 64,
            prefill_configs=[ParallelSpec(tp=1)],
            decode_configs=[ParallelSpec(tp=1, dp=2, ep=2)],
            max_pool_instances=7, ttft_slo_ms=1e9, tpot_slo_ms=1e9,
            hccl_buffsize_mb=8)
        best = report["best_configuration"]
        assert best is not None and report["qualified"] is True
        minimum = 2 * (2 - 1) * 4 * 2048 * 2 * 4 * 16
        assert best["shared_cut_demands"]["switch:intra_node_hccs"] >= (
            minimum - 1e-6)
        ceiling = 1_000_000.0 / minimum
        assert best["shared_cut_rps"] <= ceiling + 1e-9
        assert best["admitted_rps"] <= 0.9 * best["shared_cut_rps"] + 1e-9
        assert best["admitted_rps"] * minimum <= 1_000_000.0


def test_r8_01_mc2_recipe_is_recorded_as_one_executed_operation():
    case = _moe_case()
    case = replace(case, runtime=replace(case.runtime, moe_recipe="mc2"))
    model = DecodeCostModel.from_case(case, context_tokens=272)
    step = model.step(4, per_replica_batch=4, active_replicas=2, context=272)
    assert step.mc2 is True
    roles = step.comm_ledger.role_payloads()
    assert roles.get("ep_mc2_dispatch_combine", 0.0) > 0.0
    assert "ep_dispatch" not in roles and "ep_combine" not in roles
    row = _facts(case, "decode", isl=256, osl=16, hit=1.0)
    assert row["feasible"] is True
    non_mc2 = _facts(_moe_case(), "decode", isl=256, osl=16, hit=1.0)
    # One MC2 call is exactly one dispatch plus one combine: its wire bytes are
    # identical to the two separate AllToAllV exchanges, never doubled again.
    assert row["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(
            non_mc2["shared_cut_demands"]["switch:intra_node_hccs"],
            rel=1e-12))
    from mindie_sim.parallel_cuts import collective_wire
    assert collective_wire("mc2", 2, 131_072.0) == pytest.approx(
        collective_wire("all_to_all_v", 2, 131_072.0)
        + collective_wire("all_to_all_v", 2, 131_072.0), rel=1e-12)


# R8-02: the pool cut ledger is composed from the assigned physical instances
# under capacity-proportional routing, never from rank-zero templates.
def _asymmetric_route_case():
    case = _dense(layers=4, num_devices=16, max_seqs=8)
    dies = tuple(replace(die, node=0 if index == 0 else 1, package=index)
                 for index, die in enumerate(case.machine.dies))
    links = (_link("local_slow", range(1, 16), switch=1000.0),
             Link("inter_fast", 1e9, tuple(range(16)), latency_us=0,
                  channel="d2d_inter_node", semantics="per_port"))
    return replace(case, machine=replace(
        case.machine, dies=dies, links=links, topology_options=(),
        dies_per_package=1))


def test_r8_02_pool_uses_placed_instance_ledgers_not_rank_zero_templates():
    case = _asymmetric_route_case()
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    template = _facts(case, "decode", isl=256, osl=16, hit=1.0,
                      parallel=ParallelSpec(tp=2), rank_offset=0)
    placed = _facts(case, "decode", isl=256, osl=16, hit=1.0,
                    parallel=ParallelSpec(tp=2), rank_offset=1)
    assert template["shared_cut_demands"] == {}
    assert placed["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(16 * 4 * 2 * 2048 * 2, rel=1e-12))
    report = solve_xpyd_pools(
        case, total_npus=15, isl=256, osl=16, prefix_hit_rate=1.0,
        destination_cached_tokens=256, prefix_digest="d" * 64,
        prefill_configs=[ParallelSpec()],
        decode_configs=[ParallelSpec(tp=2)], max_pool_instances=7,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9, hccl_buffsize_mb=8)
    best = report["best_configuration"]
    assert best is not None and report["qualified"] is True
    assert best["admitted_rps"] <= 0.9 * 1.9073486328125
    full = next(row for row in report["feasible_rows"]
                if row["pd_ratio"] == "1P:7D")
    assert full["decode_device_ids"] == [1, 3, 5, 7, 9, 11, 13]
    demand = full["shared_cut_demands"]["switch:intra_node_hccs"]
    assert demand == pytest.approx(524_288.0, rel=1e-12)
    assert full["shared_cut_rps"] == pytest.approx(1.9073486328125, rel=1e-12)
    assert full["admitted_rps"] <= 0.9 * 1.9073486328125
    assert all(value == pytest.approx(1.0 / 7.0)
               for value in full["decode_instance_share"])
    assert sum(full["decode_instance_share"]) == pytest.approx(1.0)


def test_r8_02_weighted_instance_demands_and_routed_rates():
    instance = lambda name, rate, demand: {
        "device_ids": (0, 1), "mu_requests_per_s": rate,
        "shared_cut_demands": {"switch:x": demand}}
    left = instance("p0", 1.0, 0.0)
    right = instance("p1", 1.0, 100.0)
    pool = compose_pool_cut_demands(
        _tree_with_switch(), [left], [right],
        transfer_bytes=0.0, remap_bytes=0.0)
    assert pool["demands"] == {"switch:x": 100.0}
    both = compose_pool_cut_demands(
        _tree_with_switch(), [left, instance("p2", 1.0, 200.0)],
        [right], transfer_bytes=0.0, remap_bytes=0.0)
    assert both["demands"]["switch:x"] == pytest.approx(200.0)
    assert both["alpha"] == [pytest.approx(0.5), pytest.approx(0.5)]
    wide = instance("p_wide", 10.0, 0.0)
    ok, reason = routed_rates_within_capacity(
        _tree_with_switch(), [wide], [right], admitted_rps=1.0)
    assert ok is True and reason is None
    bad, why = routed_rates_within_capacity(
        _tree_with_switch(), [wide], [right], admitted_rps=3.0)
    assert bad is False and why == "decode_instance_rate_exceeded:0"


def _tree_with_switch():
    from mindie_sim.fabric_topology import FabricTier
    return HierarchicalFabricTree((FabricTier(
        "intra_node_hccs", 2, 1000.0, 1e6, semantics="per_port",
        switch_capacity_bytes_per_ms=1000.0),))


# R8-03: prefill's per-layer sparse index/PCP ledger reaches pool admission.
def _sparse_case(*, index=True):
    case = with_world(_sparse(), ParallelSpec(tp=1, dp=1, pcp=2))
    case = replace(case, runtime=replace(case.runtime, max_num_seqs=1),
                   machine=_machine_with(
                       case, [_link("one", range(32), switch=1000.0)]))
    if index:
        return case
    return replace(case, model=replace(case.model, layer_plans=(
        replace(case.model.layer_plans[0], index_dim=0, index_topk=0,
                index_heads=0, indexer=False, owns_index_k=False,
                indexer_type="none"),)))


def test_r8_03_prefill_index_and_pcp_ledger_reaches_the_pool():
    for hit, expected in ((0.0, 5_373_952.0), (0.999, 141_312.0)):
        case = _sparse_case()
        step = prefill_step_ms(case, isl=1024, prefix_hit_rate=hit, batch=1)
        row = _facts(case, "prefill", isl=1024, osl=16, hit=hit)
        assert step["shared_cut_demands"]["switch:intra_node_hccs"] == (
            pytest.approx(expected, rel=1e-12))
        assert row["shared_cut_demands"]["switch:intra_node_hccs"] == (
            pytest.approx(expected, rel=1e-12))
    without = _sparse_case(index=False)
    step = prefill_step_ms(without, isl=1024, prefix_hit_rate=0.999, batch=1)
    row = _facts(without, "prefill", isl=1024, osl=16, hit=0.999)
    assert step["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(10_240.0, rel=1e-12))
    assert row["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(10_240.0, rel=1e-12))
    # The actual index gather alone is conserved by the emitted ledger.
    index = prefill_step_ms(_sparse_case(), isl=1024, prefix_hit_rate=0.999,
                            batch=1)["shared_cut_demands"][
                                "switch:intra_node_hccs"]
    assert index - 10_240.0 == pytest.approx(131_072.0)


def test_r8_03_partial_cache_pool_is_bounded_by_index_plus_handoff():
    case = _sparse_case()
    report = solve_xpyd_pools(
        case, total_npus=15, isl=1024, osl=1, prefix_hit_rate=1022 / 1024,
        destination_cached_tokens=1022, prefix_digest="d" * 64,
        prefill_configs=[ParallelSpec(tp=1, pcp=2)],
        decode_configs=[ParallelSpec()], max_pool_instances=7,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9, hccl_buffsize_mb=8)
    best = report["best_configuration"]
    assert best is not None and report["qualified"] is True
    assert report["kv_bytes_per_request"] == pytest.approx(164_160.0)
    demand = best["shared_cut_demands"]["switch:intra_node_hccs"]
    assert demand >= 131_072.0 + 164_160.0 - 1e-9
    assert best["admitted_rps"] * (131_072.0 + 164_160.0) <= 1_000_000.0
    assert best["admitted_rps"] <= 0.9 * best["shared_cut_rps"]


# R8-04: every active draft axis is qualified at actual payload independent of
# the duration source; a PCP-only draft no longer bypasses it.
def test_r8_04_pcp_only_declared_draft_still_requires_actual_payload():
    case = _speculative(_dense(tp=1, pcp=2, num_devices=4, max_seqs=8),
                        _draft(), k=3, cost=0.001)
    profile = _profile((0, 1024, 16_384, 24_576, 32_768, 65_536, 131_072,
                        1 << 30), unmeasured=16_384)
    priced = priced_draft_step_ms(
        case, _draft(), batch=8, context_tokens=272, fabric=profile,
        endpoints=(0, 1))
    assert priced is not None
    assert priced.route_qualified is False
    assert priced.route_evidence == "unmeasured_profile"
    assert priced.comm_demands == {"tier:intra_node_hccs": 16_384.0}
    row = _facts(case, "decode", isl=256, osl=16, hit=0.0, tree=profile)
    assert row["feasible"] is False
    assert row["route_qualified"] is False
    assert row["route_evidence"] == "unmeasured_profile"
    assert row["draft_comm_demands"]["tier:intra_node_hccs"] == 16_384.0


def test_r8_04_declared_duration_always_extracts_draft_payloads():
    case = _speculative(_dense(layers=4, num_devices=2, max_seqs=8, tp=2),
                        _draft(), k=2, cost=0.001)
    profile = _profile((0, 1024, 16_384, 32_768, 65_536, 131_072, 1 << 30))
    row = _facts(case, "decode", isl=256, osl=16, hit=0.0, tree=profile)
    assert row["feasible"] is True
    assert row["draft_execution_source"] == "declared_layer_cost"
    assert row["draft_comm_demands"]["tier:intra_node_hccs"] > 0.0


def test_r8_04_missing_draft_geometry_is_not_a_one_byte_probe(monkeypatch):
    import mindie_sim.xpyd_instances as instances
    case = _speculative(_dense(tp=1, pcp=2, num_devices=4, max_seqs=8),
                        _draft(), k=3, cost=0.001)
    monkeypatch.setattr(instances, "priced_draft_step_ms",
                        lambda *args, **kwargs: None)
    row = _facts(case, "decode", isl=256, osl=16, hit=0.0)
    # An auxiliary draft slot without usable geometry is an unmeasured draft
    # communicator, never a one-byte connectivity probe.
    assert row["feasible"] is False
    assert row["route_evidence"] == "missing_draft_geometry"
    assert row.get("draft_comm_demands", {}) == {}


# R8-05: page universe, source/destination ownership and byte extents are one
# reconciled plan; a lane that owns no transferred page sends nothing.
def test_r8_05_incomplete_and_empty_ownership_rejects():
    for source, destination in (((0, 1), (0, 1)),
                                ((), tuple(range(8)))):
        with pytest.raises(ValueError):
            phase_bridge(ParallelSpec(pcp=2), ParallelSpec(),
                         kv_bytes=8 * 128 * 512, page_ids=tuple(range(8)),
                         source_page_ids=source,
                         destination_page_ids=destination)
    with pytest.raises(ValueError):
        phase_bridge(ParallelSpec(pcp=2), ParallelSpec(), kv_bytes=1.0,
                     page_ids=tuple(range(4)), source_page_ids=(0, 1),
                     destination_page_ids=(0, 1, 9))
    with pytest.raises(ValueError):
        phase_bridge(ParallelSpec(pcp=2), ParallelSpec(), kv_bytes=0.0,
                     page_ids=tuple(range(4)), source_page_ids=(0, 1, 2, 3),
                     destination_page_ids=())


def test_r8_05_zero_transfer_accepts_complete_ownership_plan():
    bridge = phase_bridge(ParallelSpec(pcp=2), ParallelSpec(),
                          kv_bytes=0.0, page_ids=tuple(range(4)),
                          source_page_ids=tuple(range(4)),
                          destination_page_ids=tuple(range(4)))
    assert bridge.edge_bytes == ()
    assert sorted(page for bucket in bridge.source_pages
                  for page in bucket) == list(range(4))
    assert sorted(page for bucket in bridge.destination_pages
                  for page in bucket) == list(range(4))
    empty_source = phase_bridge(
        ParallelSpec(pcp=2), ParallelSpec(), kv_bytes=0.0,
        page_ids=tuple(range(4)), source_page_ids=(),
        destination_page_ids=tuple(range(4)))
    assert empty_source.source_pages == ((), ())
    assert empty_source.edge_bytes == ()


def test_r8_05_edges_are_page_ownership_extents_not_equal_lane_fractions():
    bridge = phase_bridge(ParallelSpec(pcp=2), ParallelSpec(),
                          kv_bytes=65_536, page_ids=(0,))
    assert bridge.source_pages == ((0,), ())
    assert bridge.source_lane_bytes == (65_536.0, 0.0)
    assert bridge.edge_bytes == ((0, 0, 65_536.0),)
    assert bridge.destination_lane_bytes == (65_536.0,)
    full = phase_bridge(ParallelSpec(pcp=2), ParallelSpec(cp=2),
                        kv_bytes=8 * 128 * 512, page_ids=tuple(range(8)))
    assert sum(row[2] for row in full.edge_bytes) == pytest.approx(
        sum(full.source_lane_bytes))
    assert sorted(page for bucket in full.destination_pages
                  for page in bucket) == list(range(8))


# R8-06: per-rank KV retention follows real GQA replication and the complete
# lifetime peak includes a supplied EPLB rebalance buffer.
def test_r8_06_replicated_kv_rank_bound_is_not_divided_by_tp():
    case = _dense(layers=4, tp=8, kv_heads=1, num_devices=16, max_seqs=1)
    case = replace(case, machine=_machine_with(case, [_link("one", range(16))]))
    transfer = logical_transfer_bytes(case, prompt_tokens=1024)
    actual = Memory(case)._rank_cache_bytes(1024, padded=True, rank=0)
    assert transfer["paged_kv_bytes"] == pytest.approx(2_097_152.0)
    assert actual == [pytest.approx(2_097_152.0)]
    assert kv_rank_fraction(case.model, ParallelSpec(tp=8)) == (
        pytest.approx(1.0))
    bridge = phase_bridge(ParallelSpec(tp=8), ParallelSpec(tp=8),
                          kv_bytes=transfer["paged_kv_bytes"],
                          page_ids=transfer["page_ids"])
    lifetime = bridge_lifetime_plan(bridge)
    assert lifetime.source_retention_bytes >= 2_097_152.0
    assert lifetime.destination_retention_bytes >= 2_097_152.0
    assert lifetime.handoff_staging_bytes >= 2_097_152.0
    assert lifetime.phase_peak_bytes("decode") >= 2_097_152.0
    assert kv_rank_fraction(
        replace(case, model=replace(case.model, layer_plans=tuple(
            replace(plan, num_kv_heads=16)
            for plan in case.model.layer_plans))).model,
        ParallelSpec(tp=8)) == pytest.approx(0.125)


def test_r8_06_complete_lifetime_peak_includes_eplb_and_rejects_unfit_pool():
    case = _dense(layers=4, tp=8, kv_heads=1, num_devices=16, max_seqs=1)
    case = replace(case, machine=_machine_with(case, [_link("one", range(16))]))
    transfer = logical_transfer_bytes(case, prompt_tokens=1024)
    bridge = phase_bridge(ParallelSpec(tp=8), ParallelSpec(tp=8),
                          kv_bytes=transfer["paged_kv_bytes"],
                          page_ids=transfer["page_ids"])
    lifetime = bridge_lifetime_plan(bridge)
    base = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=1025, hccl_buffsize_mb=8)
    budget = base.peak_hbm_bytes + MIB
    case = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=math.ceil(budget / 0.9))
        for die in case.machine.dies)))
    row = _facts(case, "decode", isl=1024, osl=1, hit=0.0,
                 transfer_plan=lifetime)
    assert row["feasible"] is False
    assert row["memory"].headroom_bytes < 0.0
    assert row["kv_transfer_staging_bytes"] >= 2_097_152.0
    report = solve_xpyd_pools(
        case, total_npus=16, isl=1024, osl=1, prefix_hit_rate=0.0,
        ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec(tp=8)],
        decode_configs=[ParallelSpec(tp=8)], max_pool_instances=1,
        hccl_buffsize_mb=8)
    assert report["qualified"] is False


def test_r8_06_phase_peak_and_memory_ledger_charge_supplied_eplb_buffer():
    case = _dense(tp=2, num_devices=2, max_seqs=1)
    bridge = phase_bridge(ParallelSpec(tp=2), ParallelSpec(tp=2),
                          kv_bytes=0.0, page_ids=())
    lifetime = bridge_lifetime_plan(bridge,
                                   eplb_rebalance_peak_bytes=64 * MIB)
    assert lifetime.phase_peak_bytes("decode") >= 64 * MIB
    assert lifetime.peak_bytes() >= 64 * MIB
    row = _facts(case, "decode", isl=256, osl=1, hit=0.0,
                 transfer_plan=lifetime)
    assert row["feasible"] is True
    assert row["kv_transfer_staging_bytes"] >= 64 * MIB
    assert dict(row["memory"].kv_transfer_fields)["eplb_rebalance_peak_bytes"] == (
        pytest.approx(float(64 * MIB)))
    assert row["memory"].kv_transfer_page_owner_ranks == (0,)
    baseline = _facts(case, "decode", isl=256, osl=1, hit=0.0)
    assert row["memory"].peak_hbm_bytes - baseline["memory"].peak_hbm_bytes >= (
        64 * MIB - 1.0)


def test_r8_retained_f01_f07_physical_witnesses():
    from mindie_sim.parallel_bridge import bridge_remap_bytes
    tree = HierarchicalFabricTree((FabricTier(
        "intra_node_hccs", 2, 1000.0, 1e6, semantics="per_port",
        switch_capacity_bytes_per_ms=1.0),))
    assert tree.peer_transfer(32_768, (0, 1)).total_us == pytest.approx(
        32_769_000.0)
    demands = unique_cut_demands(tree, [(((0, 1), (2, 3)),
                                         "all_reduce", 32_768.0)])
    assert demands["switch:intra_node_hccs"] == pytest.approx(131_072.0)
    assert speculative_accept_tokens(
        0.5, 2, bonus_token=True,
        accept_probabilities=(0.5, 0.25)) == pytest.approx(1.75)
    replicated = phase_bridge(
        ParallelSpec(tp=4, pcp=2), ParallelSpec(tp=4, cp=3),
        kv_bytes=500_000, replicated=True)
    assert all(value == pytest.approx(500_000.0)
               for value in replicated.destination_lane_bytes)
    assert bridge_remap_bytes(replicated)["total_bytes"] == pytest.approx(
        3 * (500_000.0 + 4096.0))
    eplb = _dense(tp=1, dp=2, ep=2, experts=32, top_k=4, moe_inter=256,
                  num_devices=4, max_seqs=8)
    row = pool_config_facts(
        eplb, eplb.runtime.parallel, phase="decode", isl=256, osl=16,
        prefix_hit_rate=0.0, tpot_slo_ms=1e9, hccl_buffsize_mb=8,
        routing_skew=1.0, eplb_redundant_experts=0, enable_eplb=True,
        eplb_redundant_slots_per_rank=(1, 0),
        tree=HierarchicalFabricTree.from_hardware(eplb.machine), rho=0.9)
    assert row["feasible"] is False
    assert row["reason"] == "eplb_epoch_mismatch"
    assert queue_wait_ms(0.9, 1.0) == pytest.approx(4.5)
    assert queue_wait_ms(1.0, 1.0) == float("inf")
    explicit = _dense(num_devices=8, kv_heads=2, experts=8, top_k=2,
                      moe_inter=256)
    _, plan, reason = validate_placement(
        explicit, ParallelSpec(tp=4, dp=1, ep=2, expert_tp=2, cp=2))
    assert reason is None and plan is not None
    assert len({tuple(row[:5]) for row in plan["rank_map"]}) == 4


# Retained Round-4 / Round-7 invariants that share the repaired dataflow.
def test_r8_retained_dense_and_speculative_ceilings_are_conserved():
    dense = _dense(layers=4, hidden=2048, tp=1, dp=1, ep=1, num_devices=8,
                   max_seqs=8)
    dense = replace(dense, machine=_machine_with(
        dense, [_link("all", range(8), switch=1000.0)]))
    report = solve_xpyd_pools(
        dense, total_npus=5, isl=256, osl=16, prefix_hit_rate=1.0,
        destination_cached_tokens=256, ttft_slo_ms=1e9, tpot_slo_ms=1e9,
        prefill_configs=[ParallelSpec(tp=1)],
        decode_configs=[ParallelSpec(tp=2)], hccl_buffsize_mb=8,
        max_pool_instances=2, prefix_digest="d" * 64)
    best = report["best_configuration"]
    assert best is not None and report["kv_bytes_per_request"] == 0.0
    assert best["shared_cut_demands"]["switch:intra_node_hccs"] == (
        pytest.approx(524_288.0, rel=1e-12))
    assert best["shared_cut_rps"] == pytest.approx(1.9073486328125,
                                                   rel=1e-12)
    assert best["admitted_rps"] <= 0.9 * 1.9073486328125
    speculative = _speculative(dense, None, k=3, rate=1.0, bonus=True)
    row = _facts(speculative, "decode", isl=256, osl=16, hit=1.0)
    assert row["verification_rows_per_cycle"] == 4
    assert row["draft_steps_per_cycle"] == 3
    assert row["decode_cycles"] == pytest.approx(4.0)


def test_r8_retained_r7_08_domain_evidence():
    case = _dense(tp=2, num_devices=2, max_seqs=1)
    partial = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
        comm_arena_qualified=True, comm_arena_domains={"tp": MIB},
        strict_memory_qualification=True)
    assert partial.comm_persistent_qualified is False
    assert partial.comm_persistent_basis == "heuristic_arena_sizing"
    complete = analyze_memory_breakdown(
        case, sequences=1, sequence_tokens=257, hccl_buffsize_mb=0,
        comm_arena_qualified=True,
        comm_arena_domains={"tp": MIB, "etp": MIB},
        strict_memory_qualification=True)
    assert complete.comm_persistent_qualified is True
    assert complete.comm_persistent_basis == "explicit_comm_arena_domains"


def test_r8_retained_pp_hop_service_is_on_the_physical_switch():
    case = _dense(layers=6, pp=3, num_devices=3, tp=1, dp=1)
    case = replace(case, machine=_machine_with(
        case, [_link("one", range(3), switch=1000.0)]))
    decode = DecodeCostModel.from_case(case, context_tokens=1024).step(
        8, context=1024, microbatches=4)
    prefill = prefill_step_ms(case, isl=256, prefix_hit_rate=0.0, batch=8,
                              microbatches=4)
    assert decode.pp_p2p_ms == pytest.approx(65.536)
    assert prefill["pp_p2p_ms"] == pytest.approx(16_777.216)
    assert decode.step_ms >= 65.536
    assert prefill["step_ms"] >= 16_777.216
