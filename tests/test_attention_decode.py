"""Independent structure, traffic and ownership controls for staged GQA decode.

The expected ledgers are re-derived from request geometry, causal horizons and
the declared split/tile policy.  Synthetic service rates make trends and
dependencies observable; they do not establish measured hardware accuracy.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim.prediction.attention_decode import (
    RECIPE_ID,
    StagedGQADecode,
    causal_pairs,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def hardware(*, units=4):
    return {
        "machine": {"name": "synthetic-gqa-decode", "num_devices": 1,
                    "device": {
                        "memory_bytes": 1 << 30,
                        "compute_units": units,
                        "compute_peaks": {"bf16": 8e12, "fp32": 4e12},
                        "rates": {"memory": 1e9, "vector": 1e9,
                                  "mte1": 1e9, "mte2": 1e9, "mte3": 1e9},
                        "buffers": {"l1": 1 << 20, "l0a": 1 << 18,
                                    "l0b": 1 << 18, "l0c": 1 << 19,
                                    "ub": 1 << 20},
                        "buffer_units": {"l1": units, "l0a": units,
                                         "l0b": units, "l0c": units,
                                         "ub": 2 * units},
                        "resource_units": {"vector": 2 * units, "mte1": units,
                                           "mte2": units, "mte3": units},
                    }},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def make_op(requests, *, heads=4, kv_heads=2, dim=8,
            recipe=RECIPE_ID, **metadata_overrides):
    metadata = {
        "requests": tuple(requests), "query_heads": heads,
        "kv_heads": kv_heads, "head_dim": dim,
        "query_dtype": "bf16", "kv_dtype": "bf16",
        "causal": True, "window": 0, "cp": 1,
    }
    metadata.update(metadata_overrides)
    tokens = sum(query for query, _ in requests)
    length = max((context + query for query, context in requests), default=0)
    shape = (tokens, length, heads, dim)
    return OpSpec(
        "gqa_decode", "attention", "decode", 0, 0,
        tuple(f"r{index}" for index in range(len(requests))), shape, shape,
        shape, shape, recipe=recipe, dtype="bf16",
        metadata=(("gqa_decode", metadata),),
    )


def split_ranges(tiles, requested):
    count = min(requested, tiles)
    base, extra = divmod(tiles, count)
    ranges, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        ranges.append((start, start + size))
        start += size
    return tuple(ranges)


def expected_traffic(requests, *, heads, kv_heads, dim, key_tile, query_tile,
                     kv_splits, query_bytes=2, kv_bytes=2):
    """Independent HBM/traffic ledger from geometry and causal horizons."""
    group = heads // kv_heads
    q_read = kv_read = partial_write = partial_read = out_write = 0
    flops = vector = 0
    outputs = 0
    for query, context in requests:
        if not query:
            continue
        rows = query * group
        tiles = math.ceil((context + query) / key_tile)
        ranges = split_ranges(tiles, kv_splits)
        for _kv_head in range(kv_heads):
            for row_start in range(0, rows, query_tile):
                row_end = min(rows, row_start + query_tile)
                token_first, token_last = row_start // group, (row_end - 1) // group
                horizon = context + token_last + 1
                cutoff = math.ceil(horizon / key_tile)
                contributors = [span for span in ranges
                                if min(span[1], cutoff) > span[0]]
                assert contributors
                outputs += 1
                pairs = sum(
                    (min(row_end, (token + 1) * group)
                     - max(row_start, token * group)) * (context + token + 1)
                    for token in range(token_first, token_last + 1)
                )
                row_tiles = sum(
                    (min(row_end, (token + 1) * group)
                     - max(row_start, token * group))
                    * math.ceil((context + token + 1) / key_tile)
                    for token in range(token_first, token_last + 1)
                )
                q_read += len(contributors) * (row_end - row_start) * dim * query_bytes
                kv_read += horizon * 2 * dim * kv_bytes
                out_write += (row_end - row_start) * dim * query_bytes
                flops += 4 * dim * pairs
                vector += 7 * pairs + (3 * dim + 10) * row_tiles
                if len(contributors) > 1:
                    partial = (len(contributors) * (row_end - row_start)
                               * (dim * 4 + 2 * 4))
                    partial_write += partial
                    partial_read += partial
                    vector += (len(contributors) * (row_end - row_start)
                               * (3 * dim + 9)
                               + (row_end - row_start) * (dim + 1))
                else:
                    vector += (row_end - row_start) * (dim + 1)
    return {
        "outputs": outputs,
        "q_read": q_read, "kv_read": kv_read,
        "partial_write": partial_write, "partial_read": partial_read,
        "output_write": out_write,
        "hbm": q_read + kv_read + partial_write + partial_read + out_write,
        "mte2": q_read + kv_read + partial_read,
        "mte3": partial_write + out_write,
        "flops": flops, "vector": vector,
    }


def run(op, recipe=None, *, units=4, ready=0.0, external=None, raw_case=None):
    recipe = recipe or StagedGQADecode()
    case = parse_case(raw_case or hardware(units=units))
    execution = Execution(case, CostModel(case),
                          recipes=RecipeBook({RECIPE_ID: recipe}),
                          trace_limit=20000)
    prepared, _ = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    start = execution.engine.submit("input_ready", release=ready)
    if external:
        execution.engine.submit(
            "external", external["work"], placement=external["placement"],
            owner="external",
        )
    end = recipe.lower(context).start(execution.engine, "gqa", start)
    before = {}
    if ready:
        execution.engine.advance(horizon=ready)
        before = {name: value for name, value in execution.engine.serviced.items()
                  if value}
    execution.engine.advance(until=(end,))
    return execution.engine, end, before, context, recipe


def test_qk_and_pv_include_the_causal_triangular_tail():
    query, context = 4, 5
    op = make_op(((query, context),), heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=4)
    work = recipe.work(op, include_mte1=True)
    # 6+7+8+9 valid keys in one query head; QK and PV are both 2*D*pairs.
    assert causal_pairs(query, context) == 30
    assert work.matrix == {"bf16": 4 * 30 * 4 * 8}

    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        expected_traffic(((query, context),), heads=4, kv_heads=2, dim=8,
                         key_tile=4, query_tile=64, kv_splits=1)["hbm"])
    assert engine.serviced["mte1@r0:s0"] == work.resources["mte1"]


@pytest.mark.parametrize("splits", (1, 2))
def test_hbm_for_ragged_batch_and_split_partial_roundtrip(splits):
    requests = ((1, 8), (2, 3))
    op = make_op(requests, heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=4, kv_splits=splits)
    expected = expected_traffic(
        requests, heads=4, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=splits,
    )
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected["hbm"])
    assert engine.serviced["mte2@r0:s0"] == pytest.approx(expected["mte2"])
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(expected["mte3"])
    if splits == 1:
        assert expected["partial_write"] == expected["partial_read"] == 0
    else:
        assert expected["partial_write"] > 0
        assert expected["partial_read"] == expected["partial_write"]
    # Final output is written once, independent of split count.
    assert expected["output_write"] == 4 * 8 * 2 + 2 * 4 * 8 * 2


def test_zero_context_and_gqa_head_reuse_do_not_duplicate_kv():
    zero = ((1, 0), (3, 0))
    op = make_op(zero, heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=2, kv_splits=2)
    expected = expected_traffic(
        zero, heads=4, kv_heads=2, dim=8, key_tile=2,
        query_tile=64, kv_splits=2,
    )
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    # Only the three-token request spans two key tiles; the one-token request
    # keeps a direct final store.  No padded context capacity is charged.
    assert expected["partial_write"] > 0
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected["hbm"])

    requests = ((1, 8),)
    narrow = expected_traffic(
        requests, heads=4, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=2,
    )
    wide = expected_traffic(
        requests, heads=8, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=2,
    )
    assert wide["kv_read"] == narrow["kv_read"]
    assert wide["q_read"] == 2 * narrow["q_read"]
    assert wide["output_write"] == 2 * narrow["output_write"]


def test_no_service_before_input_readiness():
    op = make_op(((1, 9),), heads=2, kv_heads=1, dim=8)
    engine, end, before, _, _ = run(
        op, StagedGQADecode(key_tile=4, kv_splits=2), ready=1.25,
    )
    assert before == {}
    assert end.time >= 1.25


def test_live_tasks_and_allocations_are_released_and_peak_is_bounded():
    short = make_op(((1, 32),), heads=2, kv_heads=1, dim=8)
    long = make_op(((1, 4096),), heads=2, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=16)
    short_engine, short_end, _, _, _ = run(short, recipe)
    long_engine, long_end, _, _, _ = run(long, recipe)
    for engine in (short_engine, long_engine):
        assert not engine._tasks
        assert not engine._promises
        assert not engine.allocations
        assert all(value == 0 for value in engine.allocated.values())
    assert long_end.time > short_end.time
    assert long_engine.peak_tasks <= short_engine.peak_tasks + 2
    for pool, peak in short_engine.peak_allocated.items():
        assert long_engine.peak_allocated[pool] <= peak + 1


def test_owner_pinning_cannot_borrow_idle_per_core_units():
    op = make_op(((1, 16),), heads=1, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=32)
    one_engine, one_end, _, _, _ = run(op, recipe, units=1)
    four_engine, four_end, _, _, _ = run(op, recipe, units=4)
    assert four_end.time > 2.5 * one_end.time

    vector = "vector@r0:s0"
    external = {"work": {vector: 1e7},
                "placement": {vector: (2, 3, 4, 5, 6, 7)}}
    engine, end, _, _, _ = run(op, recipe, units=4, external=external)
    # The external work is still running on the other owners when this task
    # completes; it cannot change the pinned task's completion time.
    assert end.time == pytest.approx(four_end.time, rel=1e-12, abs=1e-12)
    assert engine.serviced[vector] > 0


def test_split_merge_waits_for_every_partial_producer():
    op = make_op(((1, 17),), heads=2, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=4, kv_splits=2)
    engine, end, _, _, _ = run(op, recipe, units=2)
    rows = list(engine.trace)
    partials = [row for row in rows if row["task"].endswith(":partial_store")]
    merges = [row for row in rows if row["task"].endswith(":merge_load")]
    assert len(partials) == 2
    assert len(merges) == 1
    assert merges[0]["demand"]["mte2@r0:s0"] == pytest.approx(160.0)
    assert min(row["start_ms"] for row in merges) >= max(
        row["end_ms"] for row in partials)
    assert not any(row["task"].endswith(":final_store") for row in rows)
    assert end.done

    single = StagedGQADecode(key_tile=4, kv_splits=1)
    engine1, _, _, _, _ = run(op, single, units=2)
    assert not any(row["task"].endswith(":partial_store")
                   for row in engine1.trace)
    assert not any(":merge" in row["task"] for row in engine1.trace)


def test_multiple_query_tiles_keep_one_output_write_per_tile():
    requests = ((4, 6),)
    op = make_op(requests, heads=4, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=4, query_tile=4, kv_splits=2)
    expected = expected_traffic(
        requests, heads=4, kv_heads=1, dim=8, key_tile=4,
        query_tile=4, kv_splits=2,
    )
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert expected["outputs"] == 4
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected["hbm"])
    stores = [row for row in engine.trace
              if row["task"].endswith(":final_store")
              or row["task"].endswith(":merge_store")]
    assert len(stores) == expected["outputs"]
    assert sum(row["demand"].get("memory@r0:s0", 0) for row in stores) == (
        pytest.approx(expected["output_write"])
    )


@pytest.mark.parametrize("overrides", (
    {"window": 1}, {"cp": 2}, {"query_heads": 3},
))
def test_unsupported_domain_is_rejected(overrides):
    op = make_op(((1, 8),), heads=4, kv_heads=2, dim=8, **overrides)
    with pytest.raises(ModelingError):
        StagedGQADecode().work(op)


def test_split_read_pipes_keep_cube_operands_and_vector_merge_separate():
    raw = hardware(units=2)
    device = raw["machine"]["device"]
    device["rates"].pop("mte2")
    device["rates"].update(mte2_aic=1e9, mte2_aiv=1e9)
    device["resource_units"].pop("mte2")
    device["resource_units"].update(mte2_aic=2, mte2_aiv=4)
    op = make_op(((2, 9),), heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=4, kv_splits=2)
    expected = expected_traffic(
        ((2, 9),), heads=4, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=2)
    engine, end, _, _, _ = run(op, recipe, units=2, raw_case=raw)
    assert end.done
    assert engine.serviced["mte2_aic@r0:s0"] == pytest.approx(
        expected["q_read"] + expected["kv_read"])
    assert engine.serviced["mte2_aiv@r0:s0"] == expected["partial_read"]
    probability_bytes = (10 + 11) * 4 * 2
    assert engine.serviced["mte1@r0:s0"] == pytest.approx(
        expected["q_read"] + expected["kv_read"] + probability_bytes)
    for row in engine.trace:
        for resource in ("mte2_aic@r0:s0", "mte2_aiv@r0:s0"):
            if row["demand"].get(resource):
                assert resource in row["placement"]


def with_view(op, **view):
    """Attach the frontend's typed storage view without changing the op shape."""
    return replace(op, metadata=op.metadata + (("attention_tensor_view", view),))


def paged_view(*, page=128, **extra):
    return {
        "kv_storage": "paged", "kv_block_size": page,
        "key_dtype": "bf16", "value_dtype": "bf16", **extra,
    }


def test_paged_kv_reads_active_pages_not_allocated_cache_capacity():
    requests = ((4, 508),)
    heads, kv_heads, dim = 24, 4, 256
    recipe = StagedGQADecode()
    dense = make_op(requests, heads=heads, kv_heads=kv_heads, dim=dim)
    paged = with_view(dense, **paged_view())
    dense_work = recipe.work(dense)
    paged_work = recipe.work(paged)
    # Four KV heads x one query tile, horizon 512, page 128: four active
    # pages per owner and one declared INT32 block-table entry per page.
    expected_pages = 4 * 4
    assert recipe.kv_access(paged).page_size == 128
    assert paged_work.resources["hbm"] == dense_work.resources["hbm"] + expected_pages * 4
    assert paged_work.resources["mte2"] == dense_work.resources["mte2"] + expected_pages * 4
    # Payload and compute geometry do not move with the storage container.
    assert paged_work.matrix == dense_work.matrix
    assert paged_work.resources["vector"] == dense_work.resources["vector"]

    engine, end, _, _, _ = run(paged, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        paged_work.resources["hbm"])


def test_paged_page_size_changes_metadata_bytes_but_not_payload_or_pairs():
    op = make_op(((4, 508),), heads=24, kv_heads=4, dim=256)
    base = StagedGQADecode()
    small = with_view(op, **paged_view(page=64))
    large = with_view(op, **paged_view(page=128))
    # Horizon 512: eight 64-token pages vs four 128-token pages, four owners.
    assert base.work(small).resources["hbm"] - base.work(large).resources["hbm"] == (
        4 * 4 * 4)
    assert base.work(small).matrix == base.work(large).matrix
    assert base.work(small).resources["vector"] == base.work(large).resources["vector"]


def test_paged_page_tail_reads_valid_rows_not_padded_payload():
    op = make_op(((4, 509),), heads=24, kv_heads=4, dim=256)
    recipe = StagedGQADecode()
    paged = with_view(op, **paged_view(page=128))
    dense = make_op(((4, 509),), heads=24, kv_heads=4, dim=256)
    dense_work = recipe.work(dense)
    paged_work = recipe.work(paged)
    # Horizon 513 touches five pages.  The 127 rows past the valid tail are
    # allocated by the pager but are not read as payload; only five block
    # table entries are charged.
    assert paged_work.resources["hbm"] == dense_work.resources["hbm"] + 5 * 4 * 4
    assert paged_work.resources["hbm"] < (
        dense_work.resources["hbm"] + 5 * 128 * 256 * 2 * 4)


def test_quantized_kv_adds_scale_and_conversion_without_rescaling_compute():
    requests = ((1, 512),)
    heads, kv_heads, dim = 24, 4, 256
    recipe = StagedGQADecode()
    bf16 = make_op(requests, heads=heads, kv_heads=kv_heads, dim=dim)
    int8 = make_op(requests, heads=heads, kv_heads=kv_heads, dim=dim,
                   kv_dtype="int8")
    int8 = with_view(
        int8, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8",
        kv_scale_dtype="fp32", kv_dequant=True,
        kv_dequant_ops_per_element=2,
    )
    access = recipe.kv_access(int8)
    assert access.paged and access.page_size == 128
    assert access.page_meta_bytes == 4
    assert access.page_scale_bytes == 8  # one FP32 K and one FP32 V scale
    assert access.dequant_ops_per_element == 2

    bf16_work, int8_work = recipe.work(bf16), recipe.work(int8)
    # Same logical pairs, matrix product and probability vectors.
    assert int8_work.matrix == bf16_work.matrix
    assert int8_work.resources["vector"] == (
        bf16_work.resources["vector"] + 2 * 513 * 256 * 2 * kv_heads)
    # INT8 halves the payload, but page metadata and scales are separate.
    # The difference is not a uniform bitwidth scale on every resource.
    payload = 513 * 2 * 256 * 2 * kv_heads
    scale_and_meta = 5 * (8 + 4) * kv_heads
    expected = bf16_work.resources["hbm"] - payload // 2 + scale_and_meta
    assert int8_work.resources["hbm"] == expected

    engine, end, _, _, _ = run(int8, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        int8_work.resources["hbm"])
    assert engine.serviced["vector@r0:s0"] == pytest.approx(
        int8_work.resources["vector"])
    # BF16 operand UB->L1 conversion store competes with the output MTE3.
    expected_mte3 = 4 * 513 * (2 * 256 * 2) + 4 * 6 * 256 * 2
    assert int8_work.resources["mte3"] == pytest.approx(expected_mte3)
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(expected_mte3)


def test_page_table_bytes_are_declared_not_inferred_from_dtype_width():
    op = make_op(((1, 256),), heads=8, kv_heads=2, dim=64)
    recipe = StagedGQADecode()
    # No tensor view is not a silent contiguous declaration.
    undeclared = recipe.kv_access(op)
    assert not undeclared.declared and undeclared.report()["storage"] == "undeclared"
    default = with_view(op, **paged_view())
    wide = with_view(op, **paged_view(kv_page_table_dtype="int64"))
    explicit = with_view(op, **paged_view(kv_page_table_bytes_per_page=32))
    assert recipe.kv_access(default).page_meta_bytes == 4
    assert recipe.kv_access(wide).page_meta_bytes == 8
    assert recipe.kv_access(explicit).page_meta_bytes == 32
    # Two owners x three touched pages for horizon 257: 2*3 entries.
    assert recipe.work(wide).resources["hbm"] - recipe.work(default).resources["hbm"] == 2 * 3 * 4
    assert recipe.work(explicit).resources["hbm"] - recipe.work(default).resources["hbm"] == 2 * 3 * 28


def test_real_a5_paged_fia_tensor_contract_prices_declared_page_bucket():
    # Retained A5 FIA descriptor:
    # profiler/tests/fixtures/attention_cache_contracts/a5_fia_paged_probe.csv
    #   query (4,24,256), K/V (512,4,128,256), output (4,24,256)
    #   page 128, BF16 payload, one INT32 block-table entry per page.
    # The fixture's 512 is the physical page-pool first dimension (and the
    # page-table capacity), not a recovered valid length.  ((4,508),) is an
    # explicit constructed horizon (512 valid rows) used to exercise the
    # declared page access; this proves contract shapes/page arithmetic, not
    # a captured call length or measured duration.
    op = make_op(((4, 508),), heads=24, kv_heads=4, dim=256)
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="bf16", value_dtype="bf16")
    recipe = StagedGQADecode()
    access = recipe.kv_access(captured)
    assert (access.paged, access.page_size, access.page_meta_bytes) == (True, 128, 4)
    dense = recipe.work(op)
    paged = recipe.work(captured)
    # Four pages x four KV-head owners x one INT32 entry.
    assert paged.resources["hbm"] == dense.resources["hbm"] + 4 * 4 * 4
    # The captured 512-token horizon is entirely valid payload; the 128-row
    # page bucket does not add a padded read.
    assert paged.resources["mte2"] == dense.resources["mte2"] + 4 * 4 * 4
    engine, end, _, _, _ = run(captured, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        paged.resources["hbm"])


def test_paged_access_generalizes_with_heads_context_and_page_geometry():
    recipe = StagedGQADecode()
    cases = (
        (((1, 1000),), 12, 2, 128, 256),
        (((1, 1000),), 12, 2, 256, 256),
        (((1, 1000),), 32, 4, 128, 128),
        (((1, 4095),), 8, 1, 128, 64),
    )
    for requests, heads, kv_heads, dim, page in cases:
        op = make_op(requests, heads=heads, kv_heads=kv_heads, dim=dim)
        paged = with_view(op, **paged_view(page=page))
        access = recipe.kv_access(paged)
        horizon = requests[0][1] + requests[0][0]
        pages = (horizon + page - 1) // page
        assert access.page_size == page
        delta = recipe.work(paged).resources["hbm"] - recipe.work(op).resources["hbm"]
        # Query 1 -> group rows fit one query tile, so one owner per KV head.
        assert delta == kv_heads * pages * 4


@pytest.mark.parametrize("splits,dequant", [(1, False), (2, False), (4, False),
                                            (2, True)])
def test_paged_split_and_dequant_ledgers_match_executed_service(splits, dequant):
    op = make_op(((4, 508),), heads=24, kv_heads=4, dim=256,
                 kv_dtype="int8" if dequant else "bf16")
    view = paged_view()
    if dequant:
        view.update({"key_dtype": "int8", "value_dtype": "int8",
                     "kv_scale_dtype": "fp32", "kv_dequant": True,
                     "kv_dequant_ops_per_element": 2})
    op = with_view(op, **view)
    recipe = StagedGQADecode(key_tile=64, kv_splits=splits)
    work = recipe.work(op)
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        work.resources["hbm"])
    assert engine.serviced["vector@r0:s0"] == pytest.approx(
        work.resources["vector"])


def test_paged_contract_mismatch_is_an_explicit_error_not_a_relabel():
    op = make_op(((1, 256),), heads=8, kv_heads=2, dim=64)
    recipe = StagedGQADecode()
    wrong_dtype = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="fp8_e4m3", value_dtype="fp8_e4m3")
    with pytest.raises(ModelingError, match="disagrees_with_gqa_decode"):
        recipe.kv_access(wrong_dtype)
    missing_page = with_view(
        op, kv_storage="paged", key_dtype="bf16", value_dtype="bf16")
    with pytest.raises(ModelingError, match="block_size_must_be_positive"):
        recipe.kv_access(missing_page)


def test_fp4_packed_payload_uses_cacheformat_row_bytes_and_page_scales():
    # Review counterexample: q=1, context=127, one Q/KV head, dim=128,
    # page=128, BF16 Q, FP4 KV, one FP32 scale per K/V per page, INT32 page.
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="fp4_e2m1")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="fp4_e2m1", value_dtype="fp4_e2m1",
        kv_scale_dtype="fp32")
    recipe = StagedGQADecode()
    access = recipe.kv_access(captured)
    assert access.key_payload_row == access.value_payload_row == 64
    assert access.scale_scope == "page" and access.page_scale_bytes == 8
    assert access.needs_conversion and access.dequant_ops_per_element == 2
    work = recipe.work(captured, include_mte1=True)
    # 2*128*128/2 payload + 2*128 Q/O + 4 page + 8 scale.
    assert work.resources["hbm"] == 16_384 + 512 + 4 + 8
    assert work.resources["mte1"] == 256 + 65_536 + 256
    engine, end, _, _, _ = run(captured, StagedGQADecode())
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        work.resources["hbm"])
    assert engine.serviced["mte1@r0:s0"] == pytest.approx(
        work.resources["mte1"])
    assert engine.serviced["vector@r0:s0"] == pytest.approx(
        work.resources["vector"])


def test_int8_conversion_operand_fails_fixed_tile_capacity_boundary():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="int8")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8", kv_dequant=True,
        kv_dequant_ops_per_element=2)
    raw = hardware(units=1)
    raw["machine"]["device"]["buffers"]["l0b"] = 24_576
    with pytest.raises(ModelingError, match="l0b:k_l0b_s0:32768>24576"):
        run(captured, StagedGQADecode(key_tile=128, slots=1), raw_case=raw)


def test_odd_width_packed_row_rounds_once_not_per_element():
    op = make_op(((1, 1),), heads=1, kv_heads=1, dim=129,
                 kv_dtype="fp4_e2m1")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="fp4_e2m1", value_dtype="fp4_e2m1",
        kv_scale_dtype="fp32")
    recipe = StagedGQADecode()
    access = recipe.kv_access(captured)
    assert access.key_payload_row == access.value_payload_row == 65
    work = recipe.work(captured)
    # horizon=2, payload=2*2*65, Q/O=2*1*129*2, one page + one FP32 K/V scale.
    assert work.resources["hbm"] == 260 + 516 + 4 + 8


def test_missing_view_preserves_known_dtype_conversion_requirement():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="int8")
    recipe = StagedGQADecode()
    access = recipe.kv_access(op)
    assert not access.declared and access.needs_conversion
    assert access.key_payload_row == access.value_payload_row == 128
    assert access.key_operand_row == access.value_operand_row == 256
    work = recipe.work(op)
    # Payload is INT8, but conversion is still charged and materialized.
    # 7*pairs + (3*dim+10)*row_tiles + conversion + output-row update.
    assert work.resources["vector"] == pytest.approx(
        7 * 128 + (3 * 128 + 10) * 1 + 2 * 128 * 128 * 2 + 129)
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        work.resources["hbm"])
    assert engine.serviced["vector@r0:s0"] == pytest.approx(
        work.resources["vector"])


def test_cache_representation_group_scale_uses_shared_cacheformat_rule():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="int8")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8",
        cache_representations={"gqa_kv": {
            "dtype": "int8", "group_size": 64, "scale_dtype": "fp32",
            "suffix_elements": 0, "suffix_dtype": "bf16",
            "packed_elements": 1}})
    recipe = StagedGQADecode()
    access = recipe.kv_access(captured)
    assert access.scale_scope == "row"
    assert access.key_scale_row == access.value_scale_row == 8
    assert access.page_scale_bytes == 0
    work = recipe.work(captured)
    pages = 1
    expected = (2 * 128 * 128) + (2 * 128 * 2) + (2 * 128 * 8) + pages * 4
    assert work.resources["hbm"] == expected


def test_build_plan_uses_split_physical_read_roles():
    op = make_op(((1, 127),), heads=8, kv_heads=2, dim=64)
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="bf16", value_dtype="bf16", kv_scale_dtype="fp32")
    from mindie_sim.prediction.mechanism import Geometry
    geometry = Geometry(
        compute_units=32,
        resource_units={"mte1": 32, "mte2_aic": 32, "mte2_aiv": 64,
                        "mte3": 32, "vector": 64, "cube": 32},
        buffers={"l1": 1 << 20, "l0a": 1 << 18, "l0b": 1 << 18,
                 "l0c": 1 << 19, "ub": 1 << 20},
        buffer_units={"l1": 32, "l0a": 32, "l0b": 32, "l0c": 32, "ub": 64},
    )
    plan = StagedGQADecode().build(captured, geometry)
    resources = plan.work.resources
    assert "mte2" not in resources
    assert resources["mte2_aic"] > 0
    assert resources["mte2_aiv"] > 0
    assert resources["hbm"] == pytest.approx(
        resources["mte2_aic"] + resources["mte2_aiv"] + resources["mte3"])
    assert plan.metadata["kv_cache_access"]["scale_scope"] == "page"


def test_planned_recipe_adapter_applies_work_and_metadata_once():
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.attention_decode import planned_recipe

    op = make_op(((1, 128),), heads=8, kv_heads=2, dim=64)
    captured = with_view(op, **paged_view())
    result = operator_costs(
        hardware(), [captured],
        recipes=RecipeBook({RECIPE_ID: planned_recipe()}),
    )
    row = result["operators"][0]
    assert row["declared_service"] == row["serviced"]
    assert row["service_delta"] == {}
    access = row["operator"]["metadata"]["program_execution"]["kv_cache_access"]
    assert access["storage"] == "paged" and access["page_size"] == 128
    assert row["declared_service"]["memory@r0:s0"] == pytest.approx(
        row["serviced"]["memory@r0:s0"])


def test_planned_recipe_adapter_preserves_split_and_conversion_options():
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.attention_decode import planned_recipe

    op = make_op(((4, 508),), heads=24, kv_heads=4, dim=256,
                 kv_dtype="int8")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8", kv_scale_dtype="fp32",
        kv_dequant=True, kv_dequant_ops_per_element=2)
    recipe = planned_recipe(key_tile=64, kv_splits=2, slots=1)
    result = operator_costs(hardware(), [captured],
                            recipes=RecipeBook({RECIPE_ID: recipe}))
    row = result["operators"][0]
    assert set(row["declared_service"]) == set(row["serviced"])
    for key, value in row["declared_service"].items():
        assert value == pytest.approx(row["serviced"][key], rel=1e-12, abs=1e-12)
    assert all(value == pytest.approx(0, abs=1e-6)
               for value in row["service_delta"].values())
    assert row["declared_service"]["vector@r0:s0"] > 0
    assert row["declared_service"]["memory@r0:s0"] > 0


def test_t10_factory_adapter_applies_split_options_once():
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.prediction.program import RecipeBook

    op = make_op(((4, 508),), heads=24, kv_heads=4, dim=256,
                 recipe="flash_attention_decode")
    captured = with_view(op, **paged_view())
    book = RecipeBook.from_mapping({"recipes": {
        "flash_attention_decode": {"kind": "staged_gqa_decode",
                                   "key_tile": 64, "kv_splits": 2,
                                   "slots": 1}}})
    result = operator_costs(hardware(), [captured], recipes=book)
    row = result["operators"][0]
    assert row["program"] == "StagedGQADecode"
    assert set(row["declared_service"]) == set(row["serviced"])
    for key, value in row["declared_service"].items():
        assert value == pytest.approx(row["serviced"][key], rel=1e-12, abs=1e-12)
    assert row["operator"]["metadata"]["program_execution"]["kv_cache_access"]["storage"] == "paged"


def test_shared_row_suffix_is_counted_once_not_per_side():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="int8")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8",
        local_kv_heads=1, cache_row_width=256, cache_head_width=128,
        cache_representations={"gqa_kv": {
            "dtype": "int8", "group_size": 64, "scale_dtype": "fp32",
            "suffix_elements": 64, "suffix_dtype": "bf16",
            "packed_elements": 1}})
    access = StagedGQADecode().kv_access(captured)
    # 128 B K payload + (64 B V payload + 128 B shared suffix) + 12 B scales.
    assert access.key_payload_row == 128
    assert access.value_payload_row == 192
    assert access.key_scale_row + access.value_scale_row == 12
    assert (access.key_payload_row + access.value_payload_row
            + access.key_scale_row + access.value_scale_row) == 332


def test_independent_kv_formats_only_convert_the_quantized_side():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128)
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="bf16", value_dtype="int8",
        local_kv_heads=1, cache_row_width=256, cache_head_width=128,
        cache_representations={
            "gqa_kv_key": {"dtype": "bf16", "group_size": 0,
                           "scale_dtype": "none", "suffix_elements": 0,
                           "suffix_dtype": "bf16", "packed_elements": 1},
            "gqa_kv_value": {"dtype": "int8", "group_size": 0,
                             "scale_dtype": "none", "suffix_elements": 0,
                             "suffix_dtype": "bf16", "packed_elements": 1}})
    access = StagedGQADecode().kv_access(captured)
    assert not access.key_needs_conversion
    assert access.value_needs_conversion
    assert access.key_conversion_ops(128) == 0
    assert access.value_conversion_ops(128) == 32_768
    assert access.conversion_ops(128) == 32_768


def test_conversion_result_l1_capacity_is_a_fixed_plan_error():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="int8")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8",
        kv_dequant=True, kv_dequant_ops_per_element=2)
    raw = hardware(units=1)
    raw["machine"]["device"]["buffers"]["l1"] = 24_576
    with pytest.raises(ModelingError, match="l1:k_post_s0:32768>24576"):
        run(captured, StagedGQADecode(key_tile=128, slots=1), raw_case=raw)


def test_conversion_mte3_competes_with_external_mte3_work():
    op = make_op(((1, 127),), heads=1, kv_heads=1, dim=128,
                 kv_dtype="int8")
    captured = with_view(
        op, kv_storage="paged", kv_block_size=128,
        key_dtype="int8", value_dtype="int8",
        kv_dequant=True, kv_dequant_ops_per_element=2)
    recipe = StagedGQADecode(key_tile=128, slots=1)
    work = recipe.work(captured)
    external_amount = 1_000.0
    mte3 = "mte3@r0:s0"
    engine, end, _, _, _ = run(
        captured, recipe,
        external={"work": {mte3: external_amount},
                  "placement": {mte3: (0,)}})
    assert end.done
    # The conversion store and the external consumer share one MTE3 pipe.
    assert engine.serviced[mte3] >= work.resources["mte3"]
    assert engine.serviced[mte3] <= work.resources["mte3"] + external_amount + 1e-6


def test_key_tiles_uses_finite_slot_residue_phase_table():
    from collections import Counter
    from mindie_sim.prediction.attention_decode import (
        _attention_pipeline_template, _tile_counts,
    )

    # Two full slot periods plus the tail stay below compact eligibility.
    op = make_op(((1, 512),), heads=8, kv_heads=1, dim=128)
    recipe = StagedGQADecode(key_tile=128, slots=2)
    spec = recipe.spec(op)
    output = spec.outputs[0]
    contribution = output.contributions[0]
    context_len = spec.requests[output.request][1]

    def key_of(index):
        start = (contribution.tile_start + index) * recipe.key_tile
        end = min(start + recipe.key_tile, output.horizon)
        valid = max(0, end - start)
        pairs, active = _tile_counts(
            output, context_len, spec.group, start, end)
        return valid, pairs, active, index % recipe.slots

    def tile(index):
        from types import SimpleNamespace
        valid, pairs, active, slot = key_of(index)
        return (SimpleNamespace(
            name="stage", work={"vector": valid}, caps={}, coupled=False,
            latency=0.0, after=(), previous=(), placement={}, slot=slot,
            acquire=(), free=()),)

    # A stage-set stand-in is enough to test phase segmentation and residue;
    # the real producer supplies the Stage objects.
    template = _attention_pipeline_template(
        tile, key_of, contribution.tile_count,
        rows=output.rows, key_tile=recipe.key_tile,
        slots=recipe.slots, core=0)
    assert template is not None
    assert template.tiles == contribution.tile_count == 5
    full = template.phases[0]
    tail = template.phases[1]
    assert (full.kind, full.count, full.end) == ("full", 4, 4)
    assert len(full.residue_stages) == 2
    assert full.key == (128, 1024, 8, None)
    assert (tail.kind, tail.count, tail.end) == ("tail", 1, 5)
    assert dict(Counter(key_of(i) for i in range(5))) == {
        (128, 1024, 8, 0): 2,
        (128, 1024, 8, 1): 2,
        (1, 8, 8, 0): 1,
    }
    # Slot residue is a stage table, not an averaged body.
    assert full.stages_for(0) != full.stages_for(1)
    assert full.stages_for(0)[0].slot == 0
    assert full.stages_for(1)[0].slot == 1

    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        recipe.work(op).resources["hbm"])
    tasks = [row["task"] for row in engine.trace
             if ":key_tiles:program:" in row["task"]]
    assert any(name.endswith(":complete_outer") for name in tasks)
    assert not any(name.endswith(":pipeline_complete") for name in tasks)
    all_tasks = [row["task"] for row in engine.trace]
    assert sum(":online_update" in name for name in all_tasks) == 5
    assert sum(":final_store" in name for name in all_tasks) == 1
    assert not any(":partial_store" in name for name in all_tasks)


def test_finite_key_phase_preserves_detailed_service_and_peak():
    from unittest.mock import patch

    op = make_op(((1, 512),), heads=8, kv_heads=1, dim=128)
    recipe = StagedGQADecode(key_tile=128, slots=2)
    template_engine, template_end, _, _, _ = run(op, recipe)
    with patch("mindie_sim.prediction.attention_decode."
               "_attention_pipeline_template", return_value=None):
        detailed_engine, detailed_end, _, _, _ = run(op, recipe)
    assert template_end.done and detailed_end.done
    assert template_end.time == pytest.approx(detailed_end.time, rel=1e-12)
    assert set(template_engine.serviced) == set(detailed_engine.serviced)
    for key, value in detailed_engine.serviced.items():
        assert template_engine.serviced[key] == pytest.approx(value, rel=1e-12)
    assert template_engine.peak_allocated == detailed_engine.peak_allocated
    template_tasks = [row["task"] for row in template_engine.trace
                      if ":key_tiles:program:" in row["task"]]
    detailed_tasks = [row["task"] for row in detailed_engine.trace
                      if ":key_tiles:program:" in row["task"]]
    assert any(name.endswith(":complete_outer") for name in template_tasks)
    assert any(name.endswith(":program:complete") for name in detailed_tasks)
    assert not any(name.endswith(":complete_outer") for name in detailed_tasks)
    assert len(template_tasks) <= len(detailed_tasks)


def test_ordinary_paged_fia_native_plan_and_public_execution():
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.attention_decode import planned_recipe

    op = make_op(((1, 126), (1, 129)), heads=4, kv_heads=4, dim=128)
    op = with_view(op, **paged_view(page=128))
    recipe = StagedGQADecode(key_tile=128, slots=2)
    plan = recipe.native_plan(op)
    assert plan is not None
    assert plan["geometry"] == {
        "batch": 2, "query_heads": 4, "kv_heads": 4, "head_dim": 128,
        "group": 1, "hq_equals_hkv": True, "page_size": 128,
        "max_pages": 2, "page_counts": (1, 2),
    }
    assert plan["work"]["head_token_pairs"] == 1028
    assert plan["work"]["qk_mac"] == 131584
    assert plan["work"]["pv_mac"] == 131584
    assert plan["work"]["q_bytes"] == 2048
    assert plan["work"]["out_bytes"] == 2048
    assert plan["work"]["kv_valid_bytes"] == 526336
    assert plan["work"]["page_counts"] == (1, 2)
    assert plan["work"]["pages"] == 3
    assert plan["work"]["lse_bytes"] == 0
    assert plan["work"]["state_writes"] == 0
    assert plan["state"]["read_only_kv"] is True
    assert plan["state"]["lse_materialized"] is False
    assert plan["ports"]["lse"] is None
    assert plan["ports"]["block_table"]["shape"] == (2, 2)
    assert plan["ports"]["query"]["shape"] == (2, 4, 128)
    assert plan["ports"]["key_cache"]["layout"] == "PA_BBND"
    assert plan["ports"]["value_cache"]["layout"] == "PA_BBND"
    assert plan["ports"]["output"]["shape"] == (2, 4, 128)
    assert plan["strategy"]["key_tile"] == 128
    assert plan["strategy"]["query_tile"] == 64
    assert plan["strategy"]["full_tiles"] == 4
    assert plan["strategy"]["tail_tiles"] == 8
    assert plan["strategy"]["key_tiles"] == 12
    assert plan["strategy"]["transactions"] == {
        "page_reads": 3, "key_page_reads": 3, "value_page_reads": 3,
        "static_kv_read": True, "state_writes": 0,
    }
    assert plan["strategy"]["compute"] == {
        "qk": "fia", "softmax": "online", "pv": "fia", "causal": True,
    }
    assert plan["abi"]["entry"] == "torch_npu.npu_fused_infer_attention_score"
    assert plan["limits"] == {
        "d256_specialized_ifa": False, "mla": False, "sfa": False,
        "multiquery_broadcast": False,
    }
    work = recipe.work(op)
    assert work.matrix["bf16"] == 2 * (
        plan["work"]["qk_mac"] + plan["work"]["pv_mac"])

    # The same operator goes through the real public operator path and
    # executes the selected OperatorPlan; no dict-only projection.
    result = operator_costs(
        hardware(units=1), [op],
        recipes=RecipeBook({RECIPE_ID: planned_recipe(key_tile=128, slots=2)}))
    row = result["operators"][0]
    assert row["program"] == "OperatorPlan"
    assert row["declared_service"]["memory@r0:s0"] == pytest.approx(
        recipe.work(op).resources["hbm"])
    assert row["serviced"] == pytest.approx(row["declared_service"], rel=1e-12,
                                            abs=1e-9)
    actual = row["operator"]["metadata"]["program_execution"]["native_paged_fia"]
    assert actual == plan
