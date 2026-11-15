"""Round-12 shared-draft residency, partial pages and candidate isolation.

[Role] Exercise all four R12 findings through byte oracles and public solves.
[Invariants] Shared weights do not alias KV; page allocations conserve every
  component; PCP replicas and DCP members are separate coordinates; one bad
  candidate cannot erase a feasible sibling; raw geometry is guarded first.
[Timeouts & Locks] Bounded synthetic arithmetic; no subprocesses or locks.
[Side-Effect Policy] Fresh Cases and scoped monkeypatches; no file writes.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import math

import pytest

from test_parallel_r7_adversarial import _dense, _sparse, _speculative, _link, _machine_with
from test_parallel_r8_adversarial import _facts
from test_parallel_r10_adversarial import _bridge, _compose, _instance, _solve
from mindie_sim.catalog.types import AuxiliarySpec
from mindie_sim.memory_breakdown import memory_ledger
from mindie_sim.parallel_bridge import bridge_lifetime_plan, bridge_remap_bytes
from mindie_sim.parallel_decode_cost import DecodeCostModel
from mindie_sim.parallel_placement import with_world
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_draft import resolve_draft_geometry
from mindie_sim.xpyd_transfer import logical_transfer_bytes
import mindie_sim.xpyd_pool_solver as solver


def _shared_case():
    case = _speculative(_dense(layers=4, hidden=256, heads=2, kv_heads=1,
        num_devices=3, max_seqs=1, max_model_len=65536), None, k=3, cost=.001)
    return replace(case, machine=_machine_with(case, [_link("one", range(3))]))


def test_r12_01_tight_shared_draft_charges_cache_with_zero_additional_weights():
    case, spec, tokens = _shared_case(), ParallelSpec(pcp=2), 32769
    direct = with_world(case, spec)
    draft, reason = resolve_draft_geometry(direct)
    assert reason is None and case.model.auxiliary is None
    one = math.ceil(tokens / 128) * 128 * 2 * 128 * 2
    assert one == 16_842_752 and 4 * one == 67_371_008
    target = Memory(direct)
    control = Memory(replace(direct, model=replace(direct.model,
                                                  auxiliary=AuxiliarySpec(mtp=draft))))
    ledger = memory_ledger(direct, sequence_tokens=tokens, hccl_buffsize_mb=8,
                           resolved_draft=draft)
    assert ledger.cache_mem.case.model.auxiliary.mtp is draft
    assert ledger.cache_mem.cache_bytes(tokens) == control.cache_bytes(tokens)
    assert ledger.cache_mem.weights == target.weights
    assert ledger.draft_weight_bytes == ledger.at(4, tokens).draft_weight_bytes == 0
    assert ledger.cache_detail[-1]["draft_kv"] == one
    assert ledger.cache_mem.cache_bytes(tokens)[0] - target.cache_bytes(tokens)[0] == one
    for rank in range(2):
        assert ledger.cache_mem._rank_cache_bytes(tokens, padded=True, rank=rank) == list(control.cache_bytes(tokens))
    # Target execution still reads only target KV. Draft execution is priced
    # separately, so feeding this cache view into DecodeCostModel would fail.
    assert DecodeCostModel.from_case(direct).kv_per_token == 4 * 512
    assert DecodeCostModel.from_case(control.case).kv_per_token == 5 * 512
    facts = _facts(direct, "decode", isl=tokens - 1, osl=1, hit=1)
    assert facts["verification_rows_per_cycle"] == 4
    assert all(row[3] == 429_132_288 for row in facts["memory"].per_rank_bytes)
    # Original audited die budget: 362,809,856.7 usable bytes. Keep it fixed
    # so the regression cannot move the budget with a changed implementation.
    tight = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=403_122_063) if i else die
        for i, die in enumerate(case.machine.dies))))
    result = _solve(tight, d=spec, isl=tokens - 1,
                    destination_cached_tokens=tokens - 1, prefix_digest="d" * 64)
    assert not result["qualified"] and result["best_configuration"] is None
    assert result["reason"] == "all_pool_candidates_rejected"
    rejected = result["decode_configs"][0]
    assert rejected["reason"] == "decode_memory_infeasible"
    assert rejected["memory"].headroom_bytes == pytest.approx(-66_322_431.3)
    assert rejected["memory"].draft_weight_bytes == 0
    assert case.model.auxiliary is direct.model.auxiliary is None


@pytest.mark.parametrize("flag", ("mtp_shared_weights", "draft_shared_weights", "eagle_shared_weights"))
def test_r12_01_final_stage_rank_tp_cache_uses_the_resolved_geometry(flag):
    for tp, pp, pcp in product((1, 2, 4), (1, 2), (1, 2)):
        base = with_world(_dense(layers=4, heads=8, kv_heads=2, num_devices=32,
                                 max_seqs=1), ParallelSpec(tp=tp, pp=pp, pcp=pcp))
        case = _speculative(base, None)
        meta = dict(case.runtime.metadata)
        meta.pop("mtp_shared_weights")
        case = replace(case, runtime=replace(case.runtime, metadata={**meta, flag: True}))
        ledger = memory_ledger(case, sequence_tokens=257, hccl_buffsize_mb=8)
        expected = 384 * 2 * max(1, math.ceil(2 / tp)) * 128 * 2 * 4
        actual = ledger.at(4, 257)
        baseline = memory_ledger(base, sequence_tokens=257, hccl_buffsize_mb=8).at(4, 257)
        for row, original in zip(actual.per_rank_bytes, baseline.per_rank_bytes):
            assert row[3] - original[3] == (expected if row[1] == pp - 1 else 0)
        assert actual.draft_weight_bytes == 0 and case.model.auxiliary is None


def test_r12_01_explicit_rank_token_ownership_is_preserved():
    base = with_world(_dense(layers=4, kv_heads=1, num_devices=8),
                      ParallelSpec(tp=2, pcp=2, pp=2))
    spans = ((0, 128), (128, 256), (256, 384), (384, 512))
    base = replace(base, runtime=replace(base.runtime, memory={"rank_token_owners": spans}))
    ledger = memory_ledger(_speculative(base, None), sequence_tokens=513, hccl_buffsize_mb=8)
    original = memory_ledger(base, sequence_tokens=513, hccl_buffsize_mb=8).at(4, 513)
    for rank, (row, old) in enumerate(zip(ledger.at(4, 513).per_rank_bytes, original.per_rank_bytes)):
        stage, _, lane, _, tp, _ = ledger.rank_map[rank]
        cache_rank = tp * 2 + lane
        owned = (256 if cache_rank == 3 else 128) if stage == 1 else 0
        assert row[3] - old[3] == 4 * owned * 512


TAILS = ((1, 0), (127, 0), (128, 0), (129, 0), (129, 1), (129, 128),
         (255, 128), (256, 128), (257, 256), (4097, 4096))


def _tail_case(kind):
    if kind == "dense":
        return _dense(layers=1, kv_heads=1, num_devices=2, max_seqs=1, max_model_len=8192)
    case = _sparse(layers=1)
    if kind == "compressed":
        case = replace(case, model=replace(case.model, layer_plans=tuple(replace(layer,
            cache_kind="window_compressed", compress_ratio=16, owns_long_kv=True,
            compressor_state_rows=16, compressor_width=1, index_input="hidden")
            for layer in case.model.layer_plans)))
    return replace(case, runtime=replace(case.runtime, max_num_seqs=1))


@pytest.mark.parametrize("kind", ("dense", "sparse", "compressed"))
@pytest.mark.parametrize("prompt,cached", TAILS)
def test_r12_02_partial_pages_conserve_real_rows_and_allocated_blocks(kind, prompt, cached):
    case = _tail_case(kind)
    transfer = logical_transfer_bytes(case, prompt_tokens=prompt,
        destination_cached_tokens=cached, prefix_digest="d" * 64)
    covered = prompt - cached // 128 * 128
    row = transfer["layer_extents"][0]
    if kind == "dense":
        expected = (math.ceil(covered / 128) * 65536, 0, 0)
    elif kind == "sparse":
        expected = (math.ceil(min(covered, 4096) / 128) * 147456, covered * 130, 0)
        for page in row["page_extents"]:
            assert page["index_bytes"] == min(128, prompt - page["page"] * 128) * 130
    else:
        expected = (math.ceil(min(covered, 4096) / 128) * 147456
                    + math.ceil((covered // 16) / 8) * 8 * 1152,
                    (covered // 16) * 130 + min(prompt, 16) * 1024, 16384)
    for key, amount, aggregate in zip(("kv_bytes", "index_bytes", "state_bytes"), expected,
        ("transferred_kv_bytes", "transferred_index_bytes", "fixed_state_transfer_bytes")):
        assert row[key] == amount == sum(page[key] for page in row["page_extents"])
        assert transfer[aggregate] == sum(layer[key] for layer in transfer["layer_extents"])
    assert transfer["page_ids"] == tuple(range(cached // 128, (prompt - 1) // 128 + 1))
    assert sum(transfer["shard_bytes"]) == expected[0] + expected[1]
    assert transfer["transfer_total_bytes"] == sum(expected) + transfer["transfer_metadata_bytes"]
    p = d = ParallelSpec()
    bridge = _bridge(p, d, transfer, layer_extents=transfer["layer_extents"])
    result = _compose(case, [_instance(case, p, (0,))], [_instance(case, d, (1,))], bridge, transfer)
    assert result["qualified"] and result["wire_bytes"] == transfer["transfer_total_bytes"]
    life = bridge_lifetime_plan(bridge, case=case, transfer=transfer)
    assert life.phase_peak_bytes("decode") == 2 * sum(expected) + 8192
    public = _solve(case, isl=prompt, destination_cached_tokens=cached, prefix_digest="d" * 64)
    assert public["qualified"] and public["best_configuration"] is not None


@pytest.mark.parametrize("prompt,cached", ((1, 0), (129, 1), (257, 256)))
def test_r12_02_recurrent_endpoints_transfer_once_on_the_final_absolute_page(prompt, cached):
    case = _dense(layers=1, num_devices=2)
    layer = replace(case.model.layer_plans[0], attention="kda",
                    recurrent_state_elements=256, conv_state_elements=64)
    case = replace(case, model=replace(case.model, layer_plans=(layer,)))
    transfer = logical_transfer_bytes(case, prompt_tokens=prompt,
        destination_cached_tokens=cached, prefix_digest="d" * 64)
    pages = transfer["layer_extents"][0]["page_extents"]
    assert transfer["fixed_state_transfer_bytes"] == 256 * 4 + 64 * 2
    assert [p["state_bytes"] for p in pages] == [0] * (len(pages) - 1) + [1152]
    p = d = ParallelSpec()
    result = _compose(case, [_instance(case, p, (0,))], [_instance(case, d, (1,))],
                      _bridge(p, d, transfer), transfer)
    assert result["qualified"] and result["unique_logical_bytes"] == 1152


@pytest.mark.parametrize("key", ("kv_bytes", "index_bytes", "state_bytes", "metadata_bytes"))
def test_r12_02_inconsistent_external_page_extents_still_reject(key):
    case = _tail_case("compressed")
    transfer = logical_transfer_bytes(case, prompt_tokens=129)
    row = transfer["layer_extents"][0]
    page = row["page_extents"][0]
    altered = dict(transfer, layer_extents=({**row, "page_extents": (
        {**page, key: page[key] + 1}, *row["page_extents"][1:])},))
    p = d = ParallelSpec()
    result = _compose(case, [_instance(case, p, (0,))], [_instance(case, d, (1,))],
                      _bridge(p, d, transfer), altered)
    assert not result["qualified"] and result["reason"] == "bridge_page_extent_not_conserved"


@pytest.mark.parametrize("cp,pcp,policy", tuple(product((1, 2, 4), (1, 2),
    ("park_extra_lanes", "replica_lanes"))))
def test_r12_03_pcp_replication_and_dcp_ownership_are_distinct(cp, pcp, policy):
    case = _dense(layers=1, kv_heads=1, num_devices=16, max_seqs=1)
    p, d = ParallelSpec(), ParallelSpec(tp=4, cp=cp, pcp=pcp, pcp_lane_policy=policy)
    transfer = logical_transfer_bytes(case, prompt_tokens=256,
        destination_cached_tokens=128, prefix_digest="d" * 64)
    bridge = _bridge(p, d, transfer, replicated=policy == "replica_lanes",
                      layer_extents=transfer["layer_extents"])
    replicated = d.active_pcp_lanes > 1
    assert (bridge.ownership_mode == "replicated") == replicated
    result = _compose(case, [_instance(case, p, (0,))],
        [_instance(case, d, range(1, 1 + d.world_size))], bridge, transfer,
        bridge_remap_bytes(bridge)["total_bytes"] if replicated else 0)
    assert result["qualified"]
    assert result["handoff_rank_edges"] == tuple((0, rank + 1, 65_600)
                                                for rank in range(4) if rank % cp == 1 % cp)
    life = bridge_lifetime_plan(bridge, case=case, transfer=transfer)
    for rank, peak in enumerate(life.phase_rank_peak_bytes("decode")):
        owns = rank // 4 < d.active_pcp_lanes and rank % cp == 1 % cp
        assert peak == (2 * 65_536 if owns else 0) + 8192
    public = _solve(case, d=d, isl=256, destination_cached_tokens=128, prefix_digest="d" * 64)
    assert public["qualified"]
    if pcp == 1:
        parked = _solve(case, d=replace(d, pcp_lane_policy="park_extra_lanes"), isl=256,
                         destination_cached_tokens=128, prefix_digest="d" * 64)
        for key in ("admitted_rps", "wire_bytes", "remap_bytes", "bridge_decode_staging_bytes"):
            assert public["best_configuration"][key] == parked["best_configuration"][key]


@pytest.mark.parametrize("reverse", (False, True))
def test_r12_03_original_candidate_orders_select_the_same_good_configuration(reverse):
    case = _dense(layers=1, kv_heads=1, num_devices=16, max_seqs=1)
    good = ParallelSpec(tp=4, cp=2)
    bad = replace(good, pcp_lane_policy="replica_lanes")
    result = _solve(case, d=good, decode_configs=[bad, good] if reverse else [good, bad],
        isl=256, destination_cached_tokens=128, prefix_digest="d" * 64)
    assert result["qualified"] and result["best_configuration"]["decode_parallel"] == good.to_dict()


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("hook", ("phase_bridge", "bridge_lifetime_plan"))
def test_r12_03_candidate_specific_failure_does_not_abort_siblings(monkeypatch, reverse, hook):
    case = _dense(layers=1, kv_heads=1, num_devices=16, max_seqs=1)
    good = ParallelSpec(tp=4, cp=2)
    bad = replace(good, pcp_lane_policy="replica_lanes")
    original = getattr(solver, hook)
    def reject(*args, **kwargs):
        spec = args[1] if hook == "phase_bridge" else args[0].destination_spec
        if spec == bad:
            raise ModelingError("bridge_physical_owner_missing")
        return original(*args, **kwargs)
    monkeypatch.setattr(solver, hook, reject)
    options = dict(isl=256, destination_cached_tokens=128, prefix_digest="d" * 64)
    result = _solve(case, d=good, decode_configs=[bad, good] if reverse else [good, bad], **options)
    assert result["qualified"] and result["best_configuration"]["decode_parallel"] == good.to_dict()
    assert result["candidate_pair_rejections"] == [{"prefill_parallel": ParallelSpec().to_dict(),
        "decode_parallel": bad.to_dict(), "feasible": False, "qualified": False,
        "reason": "bridge_physical_owner_missing"}]
    failed = _solve(case, d=bad, **options)
    assert failed["status"] == "no_feasible_configuration" and failed["best_configuration"] is None
    assert not failed["qualified"] and failed["reason"] == "bridge_physical_owner_missing"


@pytest.mark.parametrize("source,destination,reason", (
    ((0, 1), (0, 1), "bridge_page_plan_source_incomplete:(2, 3, 4, 5, 6, 7)"),
    (None, (), "bridge_page_plan_destination_empty"),
    (None, (0,), "bridge_page_plan_destination_incomplete:(1, 2, 3, 4, 5, 6, 7)")))
def test_r12_03_invalid_explicit_page_plan_returns_typed_public_rejection(source, destination, reason):
    result = _solve(_dense(num_devices=4), isl=1024,
        source_page_ids=source, destination_page_ids=destination,
        decode_configs=[ParallelSpec(), ParallelSpec(tp=2)])
    assert not result["qualified"] and result["best_configuration"] is None
    assert result["status"] == "no_feasible_configuration" and result["reason"] == reason


@pytest.mark.parametrize("field,value", tuple(product(
    ("hidden_size", "intermediate_size", "num_heads", "num_kv_heads", "head_dim"),
    (None, 0, -1, True, 1.5))))
def test_r12_04_raw_shared_geometry_rejects_before_all_placement_paths(field, value):
    case = _speculative(_dense(num_devices=3, pcp=2, max_seqs=1), None)
    case = replace(case, model=replace(case.model, **{field: value}))
    for phase in ("prefill", "decode"):
        facts = _facts(case, phase, isl=256, osl=1, hit=1)
        assert not facts["feasible"] and not facts["route_qualified"]
        assert facts["reason"] == facts["route_evidence"] == "missing_draft_geometry"
    for options in ({}, {"prefill_configs": None, "decode_configs": None}):
        result = _solve(case, d=ParallelSpec(pcp=2), isl=256, **options)
        assert not result["qualified"] and result["best_configuration"] is None
        assert result["status"] == "no_feasible_configuration" and not result["route_qualified"]
        assert result["reason"] == result["route_evidence"] == "missing_draft_geometry"
    raw = case.to_dict()
    result = _solve(raw, d=ParallelSpec(pcp=2), isl=256)
    assert result["reason"] == result["route_evidence"] == "missing_draft_geometry"
