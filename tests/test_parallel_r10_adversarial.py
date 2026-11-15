"""Round-10 mathematical conservation and admission counterexamples.

[Role]
  Exercise physical owner ports, draft service lower bounds, transferred page
  extents, stage-local MoE multiplicity, raw draft geometry and worst evidence.
[Invariants]
  Assertions use conserved bytes, independent service bounds, exact owner
  examples and typed rejection through both direct and public pool APIs.
[Timeouts & Locks]
  Bounded in-memory arithmetic; no subprocess, fixture files or device access.
[Side-Effect Policy]
  Immutable Cases and fresh ledgers only; no filesystem writes.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import math

import pytest

from test_parallel_r7_adversarial import (_dense, _draft, _link, _machine_with,
                                          _profile, _sparse, _speculative)
from test_parallel_r8_adversarial import _facts
from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.parallel_bridge import (bridge_lifetime_plan, bridge_remap_bytes,
                                        phase_bridge)
from mindie_sim.parallel_comm_ledger import reduce_route_evidence
from mindie_sim.parallel_placement import build_placement, with_world
from mindie_sim.parallel_prefill_cost import prefill_step_ms
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_cut_composer import compose_pool_cut_demands
from mindie_sim.xpyd_draft import draft_geometry_error, priced_draft_step_ms
from mindie_sim.xpyd_pool_solver import solve_xpyd_pools
from mindie_sim.xpyd_transfer import logical_transfer_bytes, kv_transfer_contention

MIB = 1 << 20


def _ports(case):
    n = len(case.machine.dies)
    link = replace(_link("one", range(n)),
                   endpoint_rates=tuple((rank, 1000.0) for rank in range(n)))
    return replace(case, machine=_machine_with(case, [link]))


def _solve(case, p=ParallelSpec(), d=ParallelSpec(), **kwargs):
    params = dict(total_npus=p.world_size + d.world_size, isl=128, osl=1,
                  prefix_hit_rate=1.0, ttft_slo_ms=1e12, tpot_slo_ms=1e12,
                  prefill_configs=[p], decode_configs=[d],
                  max_pool_instances=1, hccl_buffsize_mb=8)
    params.update(kwargs)
    return solve_xpyd_pools(case, **params)


def _instance(case, spec, ids, rate=1.0):
    return {"plan": build_placement(with_world(case, spec), spec).to_dict(),
            "device_ids": tuple(ids), "mu_requests_per_s": rate,
            "shared_cut_demands": {}, "port_demands": {}}


def _compose(case, p_rows, d_rows, bridge, transfer, remap=0.0):
    return compose_pool_cut_demands(
        HierarchicalFabricTree.from_hardware(case.machine), p_rows, d_rows,
        transfer_bytes=transfer["transfer_total_bytes"], remap_bytes=remap,
        bridge=bridge, transfer=transfer, case=case)


def _bridge(p, d, transfer, **kwargs):
    return phase_bridge(p, d, kv_bytes=transfer["transferred_kv_bytes"],
                        index_bytes=transfer["transferred_index_bytes"],
                        state_bytes=transfer["fixed_state_transfer_bytes"],
                        page_ids=transfer["page_ids"], **kwargs)


@pytest.mark.parametrize("osl", (1, 2, 4, 8, 16, 32, 64))
def test_r10_01_owning_rank_sets_admission_and_md1_service(osl):
    case = _ports(_dense(num_devices=3, max_seqs=1, kv_heads=1))
    report = _solve(case, d=ParallelSpec(tp=2, cp=2), osl=osl)
    best = report["best_configuration"]
    assert report["qualified"] and best is not None
    wire = report["kv_bytes_per_request"]
    assert wire == 262_400
    assert best["remap_bridge"]["edge_bytes"] == ((0, 0, 262_144.0),)
    handoff = best["handoff_cut_demands"]
    assert handoff["port:0:egress"] == handoff["port:1:ingress"] == wire
    assert handoff.get("port:2:ingress", 0) == 0
    assert sum(v for k, v in handoff.items() if k.endswith(":ingress")) == wire
    demands = best["shared_cut_demands"]
    owner_demand = wire + 24_768 * osl
    assert demands["port:1:ingress"] == owner_demand
    for key, demand in demands.items():
        if key.startswith("port:"):
            assert best["admitted_rps"] * demand / 1e6 < 0.9
    service = owner_demand / 1e6 * 1000
    rho = best["admitted_rps"] * owner_demand / 1e6
    assert best["transfer_queue_ms"] == pytest.approx(rho * service / (2 * (1 - rho)))
    saturated = kv_transfer_contention(
        wire, n_prefill=1, n_decode=1, nic_bw_gbps=0.001,
        hbm_bw_bytes_per_s=1e12, p_world=1, d_world=2,
        decode_step_ms=1, osl=osl, requests_per_s=1e6 / owner_demand,
        port_demands={"port:1:ingress": owner_demand},
        port_capacities={"port:1:ingress": 1e6})
    assert not saturated["feasible"] and math.isinf(saturated["queue_ms"])


def test_r10_01_partial_pages_follow_permuted_physical_lane_owners():
    case = _ports(_dense(layers=1, kv_heads=1, num_devices=12))
    p, d = ParallelSpec(pcp=3), ParallelSpec(tp=4, cp=4)
    transfer = logical_transfer_bytes(case, prompt_tokens=257,
                                      destination_cached_tokens=129, prefix_digest="d" * 64)
    bridge = _bridge(p, d, transfer, source_page_ids=(2, 1),
                     destination_page_ids=(0, 1, 2), cached_page_ids=(0,))
    result = _compose(case, [_instance(case, p, (6, 2, 10))],
                      [_instance(case, d, (3, 11, 1, 9))], bridge, transfer)
    assert result["qualified"]
    half = transfer["transfer_total_bytes"] / 2
    assert result["handoff_rank_edges"] == ((2, 11, half), (10, 1, half))
    ports = {k: v for k, v in result["handoff_demands"].items() if k.startswith("port:")}
    assert ports == {"port:2:egress": half, "port:10:egress": half,
                     "port:1:ingress": half, "port:11:ingress": half}


def test_r10_01_layer_head_index_state_and_header_ownership():
    case = _ports(_dense(layers=1, kv_heads=1, num_devices=8))
    p, d = ParallelSpec(tp=2, pcp=2), ParallelSpec(tp=4, cp=2)
    transfer = {"shard_bytes_aggregate": 160.0, "fixed_state_transfer_bytes": 64.0,
                "transfer_total_bytes": 240.0,
                "layer_extents": ({"layer": 0, "kv_bytes": 128.0,
                                   "index_bytes": 32.0, "state_bytes": 64.0,
                                   "metadata_bytes": 16.0},)}
    bridge = phase_bridge(p, d, kv_bytes=128, index_bytes=32,
                          state_bytes=64, page_ids=(5,))
    result = _compose(case, [_instance(case, p, range(4))],
                      [_instance(case, d, range(4, 8))], bridge, transfer)
    # Absolute page 5 selects PCP lane 1 and DCP member 1. Declared index
    # and non-recurrent state replicate across TP, with a header per message.
    assert result["qualified"]
    assert result["handoff_rank_edges"] == ((2, 4, 112.0), (2, 5, 240.0),
                                            (2, 6, 112.0), (2, 7, 240.0))


def test_r10_01_uneven_pp_layer_and_head_owners_conserve_pair_bytes():
    case = _ports(_dense(layers=3, kv_heads=4, num_devices=6))
    case = replace(case, model=replace(case.model, layer_plans=tuple(
        replace(layer, num_kv_heads=heads)
        for layer, heads in zip(case.model.layer_plans, (1, 2, 4)))),
        runtime=replace(case.runtime, parallel=ParallelSpec(tp=2, pp=2),
                        pp_layer_partition=(1, 2)))
    p, d = ParallelSpec(tp=2, pp=2), ParallelSpec(tp=2)
    transfer = logical_transfer_bytes(case, prompt_tokens=128)
    result = _compose(case, [_instance(case, p, range(4))],
                      [_instance(case, d, (4, 5))], _bridge(p, d, transfer), transfer)
    assert result["qualified"]
    # Layer 0 replicates its one head on both consumers; layers 1/2 shard.
    # Each physical layer message carries its own 64-B header.
    assert result["handoff_rank_edges"] == ((0, 4, 65_600.0),
                                            (0, 5, 65_600.0),
                                            (2, 4, 196_736.0),
                                            (3, 5, 196_736.0))
    assert result["wire_bytes"] == transfer["transfer_total_bytes"] + 65_536 + 192


def test_r10_01_replicated_heads_follow_dcp_group_positions():
    case = _ports(_dense(layers=1, kv_heads=2, num_devices=8))
    p, d = ParallelSpec(tp=4), ParallelSpec(tp=4, cp=2)
    t = logical_transfer_bytes(case, prompt_tokens=128)
    result = _compose(case, [_instance(case, p, range(4))],
                      [_instance(case, d, range(4, 8))], _bridge(p, d, t), t)
    # Head 0 replicas are ranks 0/1, head 1 replicas 2/3.  DCP groups
    # (4,5)/(6,7) put this page on members 4 and 6, respectively.
    assert result["qualified"]
    assert result["handoff_rank_edges"] == ((0, 4, 65_600.0), (2, 6, 65_600.0))


def test_r10_01_remap_uses_resident_source_and_each_replica_destination():
    case = _ports(_dense(layers=1, kv_heads=1, num_devices=16))
    p = ParallelSpec(pcp=2)
    d = ParallelSpec(dp=2, pcp=3, pcp_lane_policy="replica_lanes")
    transfer = logical_transfer_bytes(case, prompt_tokens=128)
    bridge = _bridge(p, d, transfer, replicated=True)
    remap = bridge_remap_bytes(bridge)["total_bytes"]
    result = _compose(case, [_instance(case, p, (0, 1), 1),
                             _instance(case, p, (8, 9), 3)],
                      [_instance(case, d, range(2, 8), 4),
                       _instance(case, d, range(10, 16), 1)], bridge, transfer, remap)
    assert result["qualified"]
    wire = transfer["transfer_total_bytes"]
    assert result["handoff_demands"]["port:0:egress"] == pytest.approx(wire / 4)
    assert result["handoff_demands"]["port:8:egress"] == pytest.approx(3 * wire / 4)
    payload = transfer["shard_bytes_aggregate"] + 4096
    ports = result["remap_demands"]
    for source, destinations, beta in ((2, (4, 6), .8), (3, (5, 7), .8),
                                        (10, (12, 14), .2), (11, (13, 15), .2)):
        assert ports[f"port:{source}:egress"] == pytest.approx(beta * payload)
        assert ports.get(f"port:{source}:ingress", 0) == 0
        for dest in destinations:
            assert ports[f"port:{dest}:ingress"] == pytest.approx(beta * payload / 2)
    assert result["remap_local_bytes"] == pytest.approx(payload)
    assert sum(v for k, v in ports.items() if k.endswith(":ingress")) == pytest.approx(2 * payload)
    assert sum(edge[2] for edge in result["remap_rank_edges"]) == pytest.approx(remap)


def test_r10_01_remap_only_ports_still_enter_the_transfer_queue():
    case = _ports(_dense(layers=1, kv_heads=1, num_devices=4, max_seqs=1))
    d = ParallelSpec(pcp=3, pcp_lane_policy="replica_lanes")
    report = _solve(case, d=d, destination_cached_tokens=128, prefix_digest="d" * 64)
    best = report["best_configuration"]
    assert report["kv_bytes_per_request"] == 0 and best is not None
    assert best["remap_cut_demands"]["port:1:egress"] == 8192
    assert best["transfer_queue_ms"] > 0
    assert all(best["admitted_rps"] * value / 1e6 < .9
               for key, value in best["shared_cut_demands"].items() if key.startswith("port:"))


@pytest.mark.parametrize("cost", (0, .001))
def test_r10_02_scalar_cannot_erase_three_sequential_pcp_gathers(cost):
    case = _dense(layers=1, hidden=256, heads=2, kv_heads=1, pcp=2,
                  num_devices=3, max_seqs=1)
    case = replace(case, machine=_machine_with(case, [_link("one", range(3), switch=1000)]))
    case = _speculative(case, _draft(), k=3, cost=cost)
    result = _solve(case, d=ParallelSpec(pcp=2), isl=256,
                    destination_cached_tokens=256, prefix_digest="d" * 64,
                    tpot_slo_ms=5)
    assert 3 * 2048 / 1000 > 5
    assert not result["qualified"] and result["best_configuration"] is None


DRAFT_AXES = (ParallelSpec(tp=2), ParallelSpec(tp=2, expert_tp=1, expert_dp=2),
              ParallelSpec(dp=2, ep=2), ParallelSpec(pcp=2),
              ParallelSpec(tp=2, cp=2), ParallelSpec(pp=2),
              ParallelSpec(tp=2, dp=2, ep=2, expert_tp=2, expert_dp=1,
                           pp=2, pcp=2, cp=2))


@pytest.mark.parametrize("parallel", DRAFT_AXES)
def test_r10_02_override_preserves_every_axis_and_hop(parallel):
    case = _dense(experts=8, top_k=2, moe_inter=2048, kv_heads=1, num_devices=32, max_seqs=1)
    case = with_world(case, parallel)
    case = replace(case, machine=_machine_with(case, [_link("one", range(32),
                                                          capacity=10_000, latency=1)]))
    draft = replace(_draft(), layers=2, num_experts=8, top_k=2, expert_intermediate_size=2048)
    case = _speculative(case, draft, k=3, cost=.001)
    tree = HierarchicalFabricTree.from_hardware(case.machine)
    plan = build_placement(case, parallel).to_dict()
    lower = 0.0
    for axis, kind in (("tp", "all_reduce"), ("etp", "all_reduce"),
                       ("ep", "all_to_all_v"), ("edp", "all_reduce"),
                       ("pcp", "all_gather"), ("dcp", "all_reduce")):
        payload = 4096 if axis == "ep" else 2048
        calls = 2 if axis == "ep" else 1
        times = [tree.collective(kind, payload, len(g), endpoints=g).total_us
                 for g in plan[axis + "_groups"] if len(g) > 1]
        lower += max(times, default=0) * calls * draft.layers / 1000
    blocks = [tuple(i for i, row in enumerate(plan["rank_map"]) if row[0] == stage)
              for stage in range(parallel.pp)]
    lower += sum(tree.peer_transfer(2048 / parallel.tp, a + b).total_us / 1000
                 for a, b in zip(blocks, blocks[1:]))
    priced = priced_draft_step_ms(case, draft, context_tokens=257, fabric=tree)
    assert priced.mandatory_ms >= lower - 1e-9
    facts = _facts(case, "decode", isl=256, osl=1, hit=1, parallel=parallel)
    assert facts["feasible"] and facts["draft_step_ms"] >= lower - 1e-9
    assert facts["step_ms"] >= 3 * lower
    result = _solve(case, d=parallel, isl=256, destination_cached_tokens=256,
                    prefix_digest="d" * 64, tpot_slo_ms=1.5 * lower)
    assert not result["qualified"]
    # A large declared duration is an override, not an additional roofline.
    assert priced.execution_duration(1000) == (1000, "declared_layer_cost")


def test_r10_03_page_boundary_wire_edges_lifetime_and_negative_headroom():
    case = _dense(num_devices=3, max_seqs=1, kv_heads=4)
    args = dict(p=ParallelSpec(pcp=2), isl=129, prefix_hit_rate=1 / 129,
                destination_cached_tokens=1, prefix_digest="d" * 64)
    report = _solve(case, **args)
    t, best = report["kv_transfer"], report["best_configuration"]
    assert t["page_ids"] == (0, 1)
    assert t["paged_kv_bytes"] == MIB
    assert t["shard_bytes_aggregate"] == 2 * MIB
    assert t["transfer_total_bytes"] == 2 * MIB + 512
    assert best["remap_bridge"]["edge_bytes"] == ((0, 0, MIB), (1, 0, MIB))
    assert best["bridge_decode_staging_bytes"] == 3_153_920
    base = _facts(case, "decode", isl=129, osl=1, hit=1 / 129)["memory"]
    budget = base.peak_hbm_bytes + 1.75 * MIB
    tight = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=math.ceil(budget / .9))
        if rank == 2 else die for rank, die in enumerate(case.machine.dies))))
    rejected = _solve(tight, **args)
    assert not rejected["qualified"] and rejected["best_configuration"] is None
    lifetime = bridge_lifetime_plan(_bridge(ParallelSpec(pcp=2), ParallelSpec(), t))
    destination = replace(tight, machine=replace(tight.machine, dies=(tight.machine.dies[2],)))
    facts = _facts(destination, "decode", isl=129, osl=1, hit=1 / 129,
                   transfer_plan=lifetime)
    assert not facts["feasible"]
    assert facts["memory"].headroom_bytes == pytest.approx(-1_318_911.8)


@pytest.mark.parametrize("prompt,cached", ((129, 1), (257, 129), (385, 257), (256, 128)))
def test_r10_03_index_and_fixed_state_transfer_extents(prompt, cached):
    case = _sparse(layers=1)
    case = replace(case, model=replace(case.model, layer_plans=tuple(
        replace(plan, architecture="mla_csa", compress_ratio=16,
                cache_kind="window_compressed", owns_long_kv=True,
                compressor_state_rows=16, compressor_width=1)
        for plan in case.model.layer_plans)), runtime=replace(case.runtime, max_num_seqs=1))
    report = _solve(case, p=ParallelSpec(pcp=2), isl=prompt,
                    prefix_hit_rate=cached / prompt, destination_cached_tokens=cached,
                    prefix_digest="d" * 64)
    t, b = report["kv_transfer"], report["best_configuration"]
    assert b is not None
    data = t["transferred_kv_bytes"] + t["transferred_index_bytes"] + t["fixed_state_transfer_bytes"]
    assert t["transferred_index_bytes"] > 0
    assert t["fixed_state_transfer_bytes"] > 0
    assert data == pytest.approx(t["transfer_total_bytes"] - t["transfer_metadata_bytes"])
    assert sum(row[2] for row in b["remap_bridge"]["edge_bytes"]) == pytest.approx(data)
    assert b["remap_bridge"]["index_bytes"] == t["transferred_index_bytes"]
    assert b["remap_bridge"]["state_bytes"] == t["fixed_state_transfer_bytes"]


@pytest.mark.parametrize("recipe", ("alltoall", "mc2"))
@pytest.mark.parametrize("partition,mask", (((4,), (True, False, True, False)),
                                          ((2, 2), (True, False, True, False)),
                                          ((2, 2), (True, True, False, False)),
                                          ((3, 1), (False, False, True, True)),
                                          ((2, 2, 1), (True, True, False, True, False))))
def test_r10_04_ep_communicators_use_their_own_stage_moe_count(recipe, partition, mask):
    case = _dense(layers=len(mask), experts=32, top_k=4, moe_inter=4096,
                  dp=2, ep=2, pp=len(partition), num_devices=8, max_seqs=1)
    case = replace(case, model=replace(case.model, layer_plans=tuple(
        replace(plan, ffn="moe" if active else "dense")
        for plan, active in zip(case.model.layer_plans, mask))),
        runtime=replace(case.runtime, pp_layer_partition=partition, moe_recipe=recipe),
        machine=_machine_with(case, [_link("one", range(8), switch=1000)]))
    step = prefill_step_ms(case, isl=256, prefix_hit_rate=0, batch=1)
    ops = [op for op in step["comm_ledger"].operations if op.role.startswith("ep_")]
    offset = 0
    for stage, layers in enumerate(partition):
        count = sum(mask[offset:offset + layers])
        stage_ops = [op for op in ops if (2 * stage, 2 * stage + 1) in op.groups]
        assert len(stage_ops) == (0 if count == 0 else 1 if recipe == "mc2" else 2)
        assert all(op.count == count for op in stage_ops)
        offset += layers
    # Two exchanges * one peer * top-k * H * BF16 * MoE layers * prompt.
    expected_ep = 2 * 4 * 2048 * 2 * sum(mask) * 256
    assert sum(op.demands()["switch:intra_node_hccs"] for op in ops) / 2 == expected_ep
    facts = _facts(case, "prefill", isl=256, osl=1, hit=0)
    expected_hops = (len(partition) - 1) * 256 * 2048 * 2
    assert facts["shared_cut_demands"]["switch:intra_node_hccs"] == expected_ep + expected_hops


BAD_GEOMETRY = tuple((field, value) for field in
                     ("num_experts", "top_k", "expert_intermediate_size")
                     for value in (True, False, 1.5, 0, -1)) + (("top_k", 9),) + tuple(
                         ("num_shared_experts", value) for value in (True, False, 1.5, -1))


@pytest.mark.parametrize("field,value", BAD_GEOMETRY)
def test_r10_05_raw_moe_geometry_fails_before_memory_or_pricing(field, value):
    draft = replace(_draft(), num_experts=8, top_k=2, expert_intermediate_size=2048)
    draft = replace(draft, **{field: value})
    case = _speculative(_dense(num_devices=2, max_seqs=1), draft)
    assert draft_geometry_error(draft) == field
    priced = priced_draft_step_ms(case, draft, context_tokens=257)
    assert priced.step_ms == 0 and not priced.route_qualified
    assert priced.route_evidence.endswith("_draft_geometry:" + field)
    assert priced.comm_demands == priced.port_demands == {}
    report = _solve(case, isl=256, destination_cached_tokens=256, prefix_digest="d" * 64)
    assert not report["qualified"] and report["best_configuration"] is None
    assert report["decode_configs"][0]["reason"] == priced.route_evidence


def test_r10_05_dense_and_inclusive_topk_boundary_remain_valid():
    for draft in (_draft(), replace(_draft(), num_experts=8, top_k=8,
                                    expert_intermediate_size=2048)):
        assert draft_geometry_error(draft) is None
        price = priced_draft_step_ms(_speculative(_dense(num_devices=2), draft), draft)
        assert price.route_qualified and price.step_ms > 0


@pytest.mark.parametrize("batch,micro,chunk,isl,pp", ((1, 1, 2, 3, 1),
                                                   (3, 2, 8, 1, 1),
                                                   (3, 2, 2, 3, 1),
                                                   (1, 1, 2, 3, 2)))
def test_r10_06_worst_evidence_survives_chunks_microbatches_and_hops(batch, micro, chunk, isl, pp):
    case = _dense(tp=2 if pp == 1 else 1, pp=pp, num_devices=4, max_seqs=batch)
    case = replace(case, runtime=replace(case.runtime, max_num_batched_tokens=chunk))
    profile = _profile((0, 1, 4095, 4096, 4097, 8191, 8192, 8193,
                        16383, 16384, 16385, 1 << 30), unmeasured=4096)
    step = prefill_step_ms(case, isl=isl, prefix_hit_rate=0, batch=batch,
                           microbatches=micro, fabric=profile)
    assert not step["route_qualified"] and step["route_evidence"] == "unmeasured_profile"
    if micro == 1:
        facts = _facts(case, "prefill", isl=isl, osl=1, hit=0, tree=profile)
        assert not facts["feasible"] and facts["route_evidence"] == "unmeasured_profile"


def test_r10_06_later_pp_hop_and_evidence_lattice_are_monotone():
    case = _dense(layers=3, pp=3, num_devices=3, max_seqs=1)
    profile = _profile((0, 4096, 1 << 30))
    class HopEvidence:
        def __getattr__(self, name):
            return getattr(profile, name)

        def peer_transfer(self, payload, endpoints, **kwargs):
            result = profile.peer_transfer(payload, endpoints, **kwargs)
            return (replace(result, route_qualified=False, evidence="disconnected")
                    if 2 in endpoints else result)

    step = prefill_step_ms(case, isl=1, prefix_hit_rate=0, batch=1, fabric=HopEvidence())
    assert not step["route_qualified"] and step["route_evidence"] == "disconnected"
    rows = ((True, "declared"), (True, "measured_profile"),
            (False, "unmeasured_profile"), (False, "disconnected"))
    for ordered in permutations(rows):
        assert reduce_route_evidence(*ordered) == (False, "disconnected")
        assert reduce_route_evidence(reduce_route_evidence(*ordered[:2]),
                                     reduce_route_evidence(*ordered[2:])) == (False, "disconnected")
    assert reduce_route_evidence((True, "measured_profile"), (False, "custom_failure")) == (
        False, "custom_failure")
