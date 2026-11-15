"""Round-13 absolute compression completion and admission regressions.

[Role] Check completed rows with independent absolute-token byte oracles.
[Invariants] Cached prefixes never restart compression; allocated compressed
  groups partition across pages; fixed endpoints travel once; every summary
  conserves the same bytes; insufficient destination HBM rejects admission.
[Timeouts & Locks] Bounded synthetic arithmetic; no locks or subprocesses.
[Side-Effect Policy] Fresh Cases only; no patches, filesystem or network I/O.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from test_parallel_r10_adversarial import _bridge, _solve
from test_parallel_r12_adversarial import _tail_case
from mindie_sim.parallel_bridge import bridge_lifetime_plan
from mindie_sim.parallel_placement import validate_placement
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_transfer import logical_transfer_bytes


def _case(block, ratio, *, window=4096, index_pool=1):
    case = _tail_case("compressed")
    plan = replace(case.model.layer_plans[0], compress_ratio=ratio,
                   window=window, index_kpool=index_pool)
    return replace(case, model=replace(case.model, layer_plans=(plan,)),
                   runtime=replace(case.runtime, kv_block_size=block))


def _completions(origin, end, ratio):
    # Enumerate real group completion tokens independently of production's
    # floor-difference implementation. Completion at end belongs to this span.
    return [token for token in range(ratio, end + 1, ratio) if token > origin]


def _oracle(case, prompt, cached, page):
    plan = case.model.layer_plans[0]
    block, ratio = case.runtime.kv_block_size, plan.compress_ratio
    capacity = max(1, block // ratio)
    index_ratio = max(ratio, plan.index_kpool)
    origin = cached // page * page
    pages = {p: dict(kv_bytes=0, index_bytes=0, state_bytes=0, metadata_bytes=64)
             for p in range(origin // page, (prompt - 1) // page + 1)}
    # Allocate one compressed capacity group on the page containing its first
    # new completed row. Later rows fill that transferred group. This also
    # handles a cached prefix ending inside an earlier compressed allocation.
    for ordinal, token in enumerate(_completions(origin, prompt, ratio)):
        if ordinal % capacity == 0:
            pages[(token - 1) // page]["kv_bytes"] += capacity * 1152
    for token in _completions(origin, prompt, index_ratio):
        pages[(token - 1) // page]["index_bytes"] += 130
    window = min(prompt - origin, plan.window)
    allocated_window = ((window + block - 1) // block) * block
    if allocated_window:
        assert page == block  # Custom page tests below have no window.
        for p in reversed(pages):
            amount = min(block, allocated_window)
            pages[p]["kv_bytes"] += amount * 1152
            allocated_window -= amount
    endpoint = min(prompt, 16) * 1024
    if plan.index_kpool > 1:
        endpoint += plan.index_kpool * 1024
    pages[max(pages)]["index_bytes"] += endpoint
    pages[max(pages)]["state_bytes"] = 16384
    logical_rows = len(_completions(cached, prompt, ratio))
    logical_window = min(prompt - cached, plan.window)
    logical_kv = (((logical_window + block - 1) // block) * block * 1152
                  + ((logical_rows + capacity - 1) // capacity) * capacity * 1152)
    logical_index = len(_completions(cached, prompt, index_ratio)) * 130 + endpoint
    return pages, logical_kv, logical_index


def _check(case, prompt, cached, *, page_size=None, shards=4):
    page_size = page_size or case.runtime.kv_block_size
    transfer = logical_transfer_bytes(case, prompt_tokens=prompt,
        destination_cached_tokens=cached, prefix_digest="d" * 64,
        page_size=page_size, shard_ranks=shards)
    expected, logical_kv, logical_index = _oracle(case, prompt, cached, page_size)
    assert transfer["page_ids"] == tuple(expected)
    layer, = transfer["layer_extents"]
    for row in layer["page_extents"]:
        assert row == {"page": row["page"], **expected[row["page"]]}
    keys = ("kv_bytes", "index_bytes", "state_bytes", "metadata_bytes")
    sums = {key: sum(row[key] for row in expected.values()) for key in keys}
    for key in keys:
        assert layer[key] == sums[key]
    kv, index, state, metadata = (sums[key] for key in keys)
    assert transfer["transferred_kv_bytes"] == kv
    assert transfer["transferred_index_bytes"] == index
    assert transfer["fixed_state_transfer_bytes"] == state == 16384
    assert transfer["transfer_metadata_bytes"] == metadata
    assert transfer["transfer_total_bytes"] == sum(sums.values())
    assert sum(transfer["shard_bytes"]) == kv + index
    assert sum(row[3] for row in transfer["shard_layout"]) == kv + index
    assert transfer["shard_bytes_aggregate"] == kv + index
    assert transfer["transfer_bytes"] == (kv + index) / shards
    assert transfer["shard_bytes"] == ((kv + index) / shards,) * shards
    assert transfer["shard_layout"] == tuple(
        (rank, min(expected), len(expected), (kv + index) / shards)
        for rank in range(shards))
    assert sum(transfer["state_shard_bytes"]) == state
    assert transfer["transfer_total_bytes"] == (
        sum(transfer["shard_bytes"]) + sum(transfer["state_shard_bytes"]) + metadata)
    assert transfer["paged_kv_bytes"] == logical_kv
    assert transfer["index_k_bytes"] == logical_index
    assert transfer["recurrent_state_bytes"] == state
    assert transfer["metadata_bytes"] == ((prompt + page_size - 1) // page_size) * 64
    assert transfer["total_bytes"] == logical_kv + logical_index + state + transfer["metadata_bytes"]
    return transfer


@pytest.mark.parametrize("block,ratio,prompt,cached,expected", (
    (8, 16, 16, 8, (10368, 16514, 16384, 43330, 94724)),
    (128, 256, 256, 128, (148608, 16514, 16384, 181570, 371204))))
def test_r13_cross_page_completion_has_exact_bytes_and_bridge_peak(block, ratio, prompt, cached, expected):
    case = parse_case(_case(block, ratio).to_dict())
    assert validate_placement(case, ParallelSpec())[2] is None
    transfer = _check(case, prompt, cached, shards=1)
    assert tuple(transfer[key] for key in ("transferred_kv_bytes", "transferred_index_bytes",
        "fixed_state_transfer_bytes", "transfer_total_bytes")) == expected[:4]
    bridge = _bridge(ParallelSpec(), ParallelSpec(), transfer,
                     layer_extents=transfer["layer_extents"])
    life = bridge_lifetime_plan(bridge, case=case, transfer=transfer)
    assert life.phase_peak_bytes("decode") == expected[4]
    public = _solve(case, isl=prompt, destination_cached_tokens=cached, prefix_digest="d" * 64)
    assert public["qualified"]
    assert public["best_configuration"]["wire_bytes"] == expected[3]
    assert public["best_configuration"]["bridge_decode_staging_bytes"] == expected[4]


@pytest.mark.parametrize("block,ratio,prefix_pages,tail", tuple(product(
    (8, 16, 64, 128), (1, 2, 4, 16, 64, 128, 256), (1, 2, 3), (0, 1, 7))))
def test_r13_absolute_compression_grid(block, ratio, prefix_pages, tail):
    cached = block * prefix_pages
    _check(_case(block, ratio), cached + block + tail, cached)


@pytest.mark.parametrize("block,ratio,prompt,cached", (
    (10, 4, 24, 20), (10, 4, 40, 30), (10, 4, 79, 31),
    (128, 16, 32, 17), (128, 16, 129, 1), (128, 16, 144, 129),
    (8, 16, 16, 15), (128, 256, 256, 255)))
def test_r13_unaligned_logical_origin_and_nondivisible_block(block, ratio, prompt, cached):
    _check(_case(block, ratio), prompt, cached)


@pytest.mark.parametrize("block,ratio,page,prompt,cached", (
    (128, 16, 8, 40, 24), (10, 4, 6, 43, 19),
    (10, 4, 4, 43, 13), (8, 16, 32, 79, 35)))
def test_r13_custom_pages_partition_compressed_capacity_after_absolute_completion(block, ratio, page, prompt, cached):
    _check(_case(block, ratio, window=0), prompt, cached, page_size=page)


@pytest.mark.parametrize("pool,prompt,cached", ((7, 28, 20), (32, 32, 24), (32, 65, 33)))
def test_r13_index_pool_uses_its_own_absolute_ratio_and_fixed_tail_once(pool, prompt, cached):
    _check(_case(8, 4, index_pool=pool), prompt, cached)


def test_r13_memory_tight_public_admission_rejects_missing_cross_page_rows():
    case = parse_case(_case(8, 16).to_dict())
    # Fixed audited budget, independent of the current implementation's peak.
    case = replace(case, machine=replace(case.machine, dies=tuple(
        replace(die, memory_bytes=710_178_420) if index == 1 else die
        for index, die in enumerate(case.machine.dies))))
    result = _solve(case, isl=16, destination_cached_tokens=8, prefix_digest="d" * 64)
    assert result["kv_transfer"]["transfer_total_bytes"] == 43_330
    assert not result["qualified"] and result["best_configuration"] is None
    assert result["reason"] == "all_pool_candidates_rejected"
    row, = result["rows"]
    assert row["reason"] == "decode_memory_infeasible"
    rank, = row["decode_instance_rows"][0]["per_rank_bytes"]
    assert tuple(rank[3:]) == (639_162_118, 639_160_578)
    assert rank[4] - rank[3] == -1540
