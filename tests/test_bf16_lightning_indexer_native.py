"""BF16 LI native-input slice: ABI binding, mechanism program and generalization.

[Role] Exercise the independent operator from real native tensor descriptors and
  the shared Engine, without a model, weights, device or observed duration.
[Invariants] Geometry comes from declared tensors and dynamic lengths; cache
  reload and local TopK exchange are mechanism stages; INT8/FP8 declarations
  never bind the BF16 implementation.
[Timeouts & Locks] Small CPU graphs only; no external locks.
[Side-Effect Policy] Read-only fixture parsing; no device or remote execution.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.family_lightning_indexer import (
    IMPLEMENTATION, RECIPE, make_op, native_op, recipes,
)
from mindie_sim.prediction.indexer_source import (
    BF16_SOURCE_FAMILY, select_indexer_source,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]


class _Bind:
    """Keep raw typed work so stage conservation stays visible."""

    def bind(self, work):
        return {**dict(work.matrix), **dict(work.resources)}


def _case():
    raw = json.loads((ROOT / "examples/cases/high-concurrency-20261030/new17/"
                      "v4flash-a3-tp8-c4.json").read_text())
    raw["machine"]["sku"] = "Ascend950DT_9582"
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
        "lightning_indexer"] = {"implementation": IMPLEMENTATION}
    return raw


def _query(rows=32, heads=4, dim=128):
    return {"shape": (rows, heads, dim), "dtype": "bf16", "layout": "TND"}


def _cache(pages=64, page=128, kv_heads=1, dim=128):
    return {"shape": (pages, page, kv_heads, dim), "dtype": "bf16",
            "layout": "PA_BBND"}


def _weights(rows=32, heads=4, dtype="fp32"):
    return {"shape": (rows, heads), "dtype": dtype, "layout": "ND"}


def _table(requests=1, columns=64):
    return {"shape": (requests, columns), "dtype": "int32", "layout": "ND"}


def _native(**overrides):
    values = dict(query=_query(), key_cache=_cache(), weights=_weights(),
                  block_table=_table(), query_lengths=(32,),
                  context_lengths=(4096,), topk=512, compression=4)
    values.update(overrides)
    return native_op(**values)


def _geometry(*, units=64, compute_units=32, cache_bytes=0, l2_bytes=0,
              ub_bytes=0, gm_bytes=0):
    buffers = {}
    if cache_bytes:
        buffers["l1"] = cache_bytes
    if l2_bytes:
        buffers["l2"] = l2_bytes
    if ub_bytes:
        buffers["ub"] = ub_bytes
    if gm_bytes:
        buffers["gm"] = gm_bytes
    return Geometry(compute_units, {
        "hbm": 1, "vector": units, "mte1": compute_units, "mte2_aic": compute_units,
        "mte2_aiv": 2 * compute_units, "mte3": 2 * compute_units,
        "fixpipe": compute_units,
    }, buffers, {})


def test_native_tensor_inputs_bind_and_execute_independently():
    op = _native()
    binding = dict(op.metadata)["native_input_binding"]
    assert binding["kind"] == "independent_native_tensor_abi"
    assert binding["native_kernel"] == "LightningIndexer"
    assert binding["inputs"]["query"]["shape"] == (32, 4, 128)
    assert binding["inputs"]["key_cache"]["layout"] == "PA_BBND"
    assert binding["inputs"]["block_table"]["dtype"] == "int32"
    assert binding["dynamic_state"] == {"query_lengths": [32],
                                        "key_lengths": [4128]}
    assert binding["cache_policy"]["owner"] == 0
    assert binding["cache_policy"]["l2_capacity_bytes"] is None
    assert binding["outputs"]["indices"] == {
        "shape": (32, 1, 512), "dtype": "int32", "layout": "ND"}
    assert op.recipe == RECIPE and op.source_boundary == "kernel"
    ledger = dict(op.metadata)["lightning_indexer_work"]
    assert ledger["query_key_pairs"] == sum((4096 + i + 1) // 4 for i in range(32))
    assert op.flops == 2 * ledger["query_key_pairs"] * 4 * 128
    result = graph_costs(_case(), (op,), isolated=False)
    assert result["duration_ms"] > 0
    assert result["execution"]["completed_tasks"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_profiler_captured_bf16_lightning_indexer_geometry_binds():
    # Shapes from the retained A3 GLM53 LightningIndexer tensor contract:
    # q [1,32,128], paged k [703,128,1,128], weights [1,32], table [1,576].
    op = native_op(
        query={"shape": (1, 32, 128), "dtype": "bf16", "layout": "TND"},
        key_cache={"shape": (703, 128, 1, 128), "dtype": "bf16", "layout": "ND"},
        weights={"shape": (1, 32), "dtype": "bf16", "layout": "ND"},
        block_table={"shape": (1, 576), "dtype": "int32", "layout": "ND"},
        query_lengths=(1,), context_lengths=(4096,), topk=2048, compression=4,
        declaration={"source_family": BF16_SOURCE_FAMILY, "q_dtype": "bf16",
                     "k_dtype": "bf16", "weight_dtype": "bf16",
                     "layout_query": "TND", "layout_key": "ND"})
    view = dict(op.metadata)["native_tensor_view"]
    assert view["query_shape"] == (1, 32, 128)
    assert view["cache_head_shape"] == (1, 128)
    assert view["weight_dtype"] == "bf16"
    assert view["output_shape"] == (1, 1, 2048)
    ledger = dict(op.metadata)["lightning_indexer_work"]
    assert ledger["query_key_pairs"] == 4097 // 4
    assert op.flops == 2 * (4097 // 4) * 32 * 128
    assert graph_costs(_case(), (op,), isolated=False)["duration_ms"] > 0


def test_native_dtype_aliases_share_one_representation():
    op = _native(query={"shape": (8, 4, 128), "dtype": "bfloat16",
                        "layout": "TND"},
                 weights={"shape": (8, 4), "dtype": "float32", "layout": "ND"},
                 query_lengths=(8,))
    view = dict(op.metadata)["native_tensor_view"]
    assert view["query_dtype"] == "bf16"
    assert view["weight_dtype"] == "fp32"


@pytest.mark.parametrize("declaration,error", [
    ({"source_family": "A5DeviceAdaptor.indexer_select_post_process",
      "image_id": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
      "q_dtype": "int8", "k_dtype": "int8"}, "source_not_bf16"),
    ({"source_family": BF16_SOURCE_FAMILY, "q_dtype": "int8",
      "k_dtype": "int8", "layout_query": "TND", "layout_key": "PA_BBND"},
     "source_not_bf16"),
    ({"source_family": BF16_SOURCE_FAMILY, "q_dtype": "bf16",
      "k_dtype": "bf16", "layout_query": "TND", "layout_key": "PA_BBND",
      "query_scale_present": True}, "source_not_bf16"),
])
def test_int8_or_scale_declarations_never_bind_the_bf16_operator(declaration, error):
    with pytest.raises(ModelingError, match=error):
        _native(declaration=declaration)


def test_explicit_bf16_declaration_selects_the_bf16_representation():
    declaration = {"source_family": BF16_SOURCE_FAMILY, "q_dtype": "bf16",
                   "k_dtype": "bf16", "weight_dtype": "fp32",
                   "layout_query": "TND", "layout_key": "PA_BBND"}
    selected = select_indexer_source(declaration)
    assert selected["status"] == "bf16_li_v2"
    assert selected["query_scale_present"] is False
    op = _native(declaration=declaration)
    assert dict(op.metadata)["native_input_binding"]["source_selection"] == "declared_bf16_li"
    quant = {"source_family": "A5DeviceAdaptor.indexer_select_post_process",
             "image_id": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
             "q_dtype": "fp8_e4m3", "k_dtype": "fp8_e4m3",
             "query_scale_present": True, "query_scale_dtype": "fp32",
             "layout_query": "TND", "layout_key": "PA_BSND",
             "query_quant_mode": 0, "key_quant_mode": 0}
    assert select_indexer_source(quant)["status"] == "c8_r64"
    assert select_indexer_source(None)["status"] == "unknown_source"


@pytest.mark.parametrize("overrides,match", [
    ({"weights": _weights(heads=8)}, "weight_geometry"),
    ({"key_cache": _cache(dim=64)}, "head_dim"),
    ({"block_table": _table(columns=8)}, "block_table_capacity"),
    ({"query_lengths": (31,)}, "query_lengths"),
    ({"block_table": _table(requests=2)}, "block_table_requests"),
    ({"block_table": {"shape": (1, 64), "dtype": "int64", "layout": "ND"}},
     "block_table_dtype"),
    ({"query": {"shape": (32, 4, 128), "dtype": "int8", "layout": "TND"}},
     "query_dtype"),
    ({"page_size": 64}, "page_size"),
])
def test_native_abi_mismatches_are_visible(overrides, match):
    with pytest.raises(ModelingError, match=match):
        _native(**overrides)


def test_multi_request_lengths_are_dynamic_state_not_padding():
    op = _native(query=_query(rows=24), weights=_weights(rows=24),
                 key_cache=_cache(pages=64), block_table=_table(requests=2),
                 query_lengths=(16, 8), context_lengths=(4096, 300))
    ledger = dict(op.metadata)["lightning_indexer_work"]
    assert ledger["sequence_partition"] == ((16, 4096), (8, 300))
    assert ledger["query_key_pairs"] == sum((4096 + i + 1) // 4 for i in range(16)) + \
        sum((300 + i + 1) // 4 for i in range(8))
    assert ledger["selection_output_bytes"] == 24 * 512 * 4
    assert op.local_shape == (24, 4, 128)
    assert graph_costs(_case(), (op,), isolated=False)["duration_ms"] > 0


def _one_token_native(context, *, pages=1, columns=1):
    return native_op(
        query=_query(1, 4, 128), key_cache=_cache(pages=pages),
        weights=_weights(1, 4), block_table=_table(1, columns),
        query_lengths=(1,), context_lengths=(context,), topk=512,
        compression=4)


def test_compressed_k_page_capacity_uses_materialized_rows():
    exact = _one_token_native(511)  # 512 tokens -> 128 compressed rows = 1 page
    ledger = dict(exact.metadata)["lightning_indexer_work"]
    assert ledger["compressed_key_rows"] == 128
    assert ledger["physical_key_pages"] == (1,)
    assert ledger["logical_token_pages"] == (1,)
    assert ledger["block_table_columns"] == 1
    capacity = dict(exact.metadata)["native_input_binding"]["cache_capacity"]
    assert capacity["key_rows_per_page"] == 128
    assert capacity["logical_tokens_per_page"] == 512
    # Logical token pages and physical compressed-K pages are distinct: 513
    # tokens still materialize 128 compressed rows in one physical page.
    split = _one_token_native(512)
    split_ledger = dict(split.metadata)["lightning_indexer_work"]
    assert split_ledger["compressed_key_rows"] == 128
    assert split_ledger["physical_key_pages"] == (1,)
    assert split_ledger["logical_token_pages"] == (2,)
    # A short tail is one page.
    short = _one_token_native(10)
    short_ledger = dict(short.metadata)["lightning_indexer_work"]
    assert short_ledger["compressed_key_rows"] == 2
    assert short_ledger["physical_key_pages"] == (1,)
    # One compressed row over one physical page is rejected by both entries.
    with pytest.raises(ModelingError, match="block_table_capacity"):
        _one_token_native(515)
    with pytest.raises(ModelingError, match="block_table_too_short"):
        make_op(query_rows=1, context=515, compression=4, page_size=128,
                block_table_columns=1)
    two_pages = _one_token_native(515, pages=2, columns=2)
    two_ledger = dict(two_pages.metadata)["lightning_indexer_work"]
    assert two_ledger["compressed_key_rows"] == 129
    assert two_ledger["physical_key_pages"] == (2,)
    assert make_op(query_rows=1, context=511, compression=4, page_size=128,
                   block_table_columns=1).local_shape == (1, 4, 128)


def test_kv_heads_own_independent_topk_exchange_domains():
    two = make_op(query_rows=1, context=8191, compression=4, topk=512,
                  heads=4, kv_heads=2)
    _, _, dual = _mechanism(two, _geometry(units=2, compute_units=1))
    one = make_op(query_rows=1, context=8191, compression=4, topk=512,
                  heads=2, kv_heads=1)
    _, _, single = _mechanism(one, _geometry(units=2, compute_units=1))
    # 2048 candidates, TopK 512, two owners: 1024 exchanged entries per
    # (query row, KV head), 8 B each = 8192 B one-way per head.
    assert single["exchange_bytes"] == 8192
    assert dual["exchange_bytes"] == 16384
    assert dual["local_topk_lists"] == 2 * single["local_topk_lists"]
    assert dual["local_list_merge_ops"] == 2 * single["local_list_merge_ops"]
    assert dual["local_topk_exchange_kv_heads"] == 2
    _, dual_plan, _ = _mechanism(two, _geometry(units=2, compute_units=1))
    _, single_plan, _ = _mechanism(one, _geometry(units=2, compute_units=1))
    dual_delta = (dual_plan.work.resources["hbm"]
                  - int(dict(two.metadata)["lightning_indexer_work"]["unique_key_bytes"]))
    single_delta = (single_plan.work.resources["hbm"]
                    - int(dict(one.metadata)["lightning_indexer_work"]["unique_key_bytes"]))
    assert dual_delta > single_delta


def test_key_access_hierarchy_separates_l1_retention_and_l2_fill():
    # The Astra closure example: two 16-row tiles scan 263,168 and 264,192 B.
    op = make_op(query_rows=32, context=4096, compression=4)
    _, _, retained = _mechanism(
        op, _geometry(units=2, cache_bytes=512 * 1024))
    assert retained["cache_levels"]["l1"]["policy"] == "retain_prefix"
    assert retained["key_gm_bytes"] == 264192
    assert retained["key_hbm_bytes"] == 264192
    assert retained["key_transport_bytes"] == 527360
    assert retained["l1_allocation_bytes"] == 280576
    assert retained["q_allocation_bytes"] == 16384
    assert retained["k_allocation_bytes"] == 264192

    _, _, stream = _mechanism(
        op, _geometry(units=2, cache_bytes=128 * 1024))
    assert stream["cache_levels"]["l1"]["policy"] == "stream_panels"
    assert stream["key_gm_bytes"] == 527360
    assert stream["key_hbm_bytes"] == 527360
    assert stream["key_transport_bytes"] == 527360
    assert stream["l1_allocation_bytes"] == 131072
    assert stream["q_allocation_bytes"] == 16384
    assert stream["k_allocation_bytes"] == 114688

    l2_op = make_op(query_rows=32, context=4096, compression=4,
                    l2_capacity_bytes=8 * 1024 * 1024)
    _, _, l2_hit = _mechanism(
        l2_op, _geometry(units=2, cache_bytes=128 * 1024,
                         l2_bytes=8 * 1024 * 1024))
    assert l2_hit["cache_levels"]["l1"]["policy"] == "stream_panels"
    assert l2_hit["cache_levels"]["l2"]["policy"] == "retain_prefix"
    assert l2_hit["key_gm_bytes"] == 527360
    assert l2_hit["key_hbm_bytes"] == 264192
    assert l2_hit["key_transport_bytes"] == 527360
    assert l2_hit["l1_allocation_bytes"] == 131072
    assert l2_hit["l2_allocation_bytes"] == 264192

    _, _, both = _mechanism(
        l2_op, _geometry(units=2, cache_bytes=512 * 1024,
                         l2_bytes=8 * 1024 * 1024))
    assert both["key_gm_bytes"] == 264192
    assert both["key_hbm_bytes"] == 264192
    assert both["l1_allocation_bytes"] == 280576


def test_panel_plan_reserves_q_and_splits_k_within_l1():
    op = make_op(query_rows=32, context=4096, compression=4)
    _, _, stream = _mechanism(
        op, _geometry(units=2, compute_units=1, cache_bytes=128 * 1024))
    assert stream["cache_levels"]["l1"]["policy"] == "stream_panels"
    assert stream["q_allocation_bytes"] == 16384
    assert stream["k_allocation_bytes"] == 114688
    assert stream["l1_allocation_bytes"] == 131072
    assert stream["key_gm_bytes"] == 527360
    assert stream["key_hbm_bytes"] == 527360
    assert stream["key_transport_bytes"] == 527360
    assert stream["key_access"]["q_bytes"] == 32768
    assert stream["key_access"]["panel_rows"] == 448
    for tile in stream["key_access"]["tiles"]:
        for panel in tile["panels"]:
            assert panel["logical_bytes"] <= stream["k_allocation_bytes"]
        assert len(tile["panels"]) >= 2

    full = make_op(query_rows=32, context=8160, compression=4,
                   cache_owner=16)
    _, _, full_plan = _mechanism(full, _geometry(units=64, cache_bytes=512 * 1024))
    assert full_plan["cache_levels"]["l1"]["policy"] == "stream_panels"
    assert full_plan["q_allocation_bytes"] == 16384
    assert full_plan["k_allocation_bytes"] == 507904
    assert full_plan["l1_allocation_bytes"] == 524288
    assert full_plan["key_access"]["panel_rows"] == 1984
    assert full_plan["aiv_owner_units"] == (32, 33)
    assert [len(tile["panels"]) for tile in full_plan["key_access"]["tiles"]] == [2, 2]

    _, stream_plan = recipes()[RECIPE].prepare(
        op, _geometry(units=2, compute_units=1, cache_bytes=128 * 1024))
    stream_program = stream_plan.program(
        _context(op, _geometry(units=2, compute_units=1), split=True))
    by_name = {stage.name: stage for stage in stream_program.stages}
    load0 = by_name["key_load_0_0"]
    qk0 = by_name["qk_weighted_reduce_0_0"]
    assert "key_load_0_0" in qk0.after and "query_load_0" in qk0.after
    assert "key_load_0_0" in qk0.after
    assert "topk_update_0_0" in by_name["key_load_0_1"].after
    assert "index_release" in by_name
    assert "index_reserve" in by_name


def _context(op, geometry, *, split=True, pools=False):
    resources = {"hbm": 1.0, "vector": 1.0, "mte1": 1.0, "mte3": 1.0,
                 "fixpipe": 1.0, "cube": 1.0}
    names = {"hbm": "hbm@x", "vector": "vector@x", "mte1": "mte1@x",
             "mte3": "mte3@x", "fixpipe": "fixpipe@x", "cube": "cube@x"}
    if split:
        resources.update(mte2_aic=1.0, mte2_aiv=1.0)
        names.update(mte2_aic="mte2_aic@x", mte2_aiv="mte2_aiv@x")
    else:
        resources["mte2"] = 1.0
        names["mte2"] = "mte2@x"
    pool_names = {}
    pool_units = {}
    pool_capacities = {}
    if pools:
        for pool in ("l1", "l2", "ub", "gm"):
            if pool not in geometry.buffers:
                continue
            pool_names[pool] = pool
            pool_units[pool] = (int(geometry.compute_units)
                                if pool in {"l1", "l2"}
                                else int(geometry.resource_units.get("vector", 1)))
            pool_capacities[pool] = float(geometry.buffers[pool])
    return RecipeContext(
        op=op, work={}, resource_names=names, rates=resources,
        core_count=geometry.compute_units,
        compute_peaks={"bf16": 1.0, "int8": 1.0, "fp16": 1.0},
        resource_units=dict(geometry.resource_units),
        pool_names=pool_names, pool_units=pool_units,
        pool_capacities=pool_capacities)


def test_owner_binds_paired_aiv_subblocks_not_double_core_count():
    op = make_op(query_rows=32, context=4096, compression=4, cache_owner=16)
    _, plan = recipes()[RECIPE].prepare(
        op, _geometry(units=64, cache_bytes=128 * 1024))
    program = plan.program(_context(op, _geometry(units=64), split=True))
    qk = next(stage for stage in program.stages
              if stage.name.startswith("qk_weighted_reduce"))
    assert qk.placement["vector@x"] == (32, 33)
    assert qk.placement["cube@x"] == (16,)
    assert qk.placement["mte2_aic@x"] == (16,)
    query = next(stage for stage in program.stages
                 if stage.name == "query_load_0")
    assert query.placement["mte2_aiv@x"] == (32, 33)
    assert query.placement["mte2_aic@x"] == (16,)
    topk = next(stage for stage in program.stages
                if stage.name.startswith("topk_update_"))
    assert topk.placement["vector@x"] == (32, 33)
    merge = next(stage for stage in program.stages
                 if stage.name.startswith("topk_merge_"))
    assert merge.placement["vector@x"] == (32, 33)

    low = make_op(query_rows=32, context=4096, compression=4, cache_owner=1)
    _, low_plan = recipes()[RECIPE].prepare(
        low, _geometry(units=64, cache_bytes=128 * 1024))
    low_program = low_plan.program(_context(low, _geometry(units=64), split=True))
    low_qk = next(stage for stage in low_program.stages
                  if stage.name.startswith("qk_weighted_reduce"))
    assert low_qk.placement["vector@x"] == (2, 3)
    assert low_qk.placement["cube@x"] == (1,)


def test_cache_line_fill_and_small_visible_tail():
    op = make_op(query_rows=64, context=31, compression=32, topk=512,
                 heads=4, head_dim=32, kv_heads=1, cache_owner=0)
    _, _, mechanism = _mechanism(
        op, _geometry(units=2, compute_units=1, cache_bytes=64 * 1024,
                      ub_bytes=196608))
    assert mechanism["key_transport_bytes"] == 384
    assert mechanism["key_gm_bytes"] == 128
    assert mechanism["key_hbm_bytes"] == 128
    assert mechanism["key_access"]["panels"] == 4
    assert mechanism["workspace"]["score_bytes_per_aiv"] + \
        mechanism["workspace"]["list_bytes_per_aiv"] <= 196608


def test_q16_retained_k_graph_respects_ub_peak():
    op = make_op(query_rows=16, context=16384, compression=4, topk=512,
                 heads=4, head_dim=32, kv_heads=1)
    raw = _case()
    raw["machine"]["sku"] = "Ascend910_9362"
    result = graph_costs(raw, (op,), isolated=False)
    peak = result["execution"]["peak_allocated_per_unit"]
    assert peak["l1@r0:s0:unit0"] == 266496
    assert max(value for key, value in peak.items()
               if key.startswith("ub@r0:s0:")) <= 196608
    assert result["execution"]["live_tasks"] == 0


def test_kv_head_output_and_exchange_stage_conservation():
    op = make_op(query_rows=1, context=8191, compression=4, topk=512,
                 heads=4, head_dim=128, kv_heads=2)
    _, plan, mechanism = _mechanism(
        op, _geometry(units=2, compute_units=1, ub_bytes=196608))
    assert mechanism["exchange_bytes"] == 16384
    assert plan.work.resources["mte3"] == 20480
    program = plan.program(_context(
        op, _geometry(units=2, compute_units=1, ub_bytes=196608), split=True))
    summed = {}
    for stage in program.stages:
        for role, amount in dict(stage.work).items():
            summed[role] = summed.get(role, 0) + amount
    assert summed["mte3@x"] == plan.work.resources["mte3"] == 20480
    assert summed["hbm@x"] == plan.work.resources["hbm"]
    names = [stage.name for stage in program.stages]
    write = names.index("exchange_write_0")
    read = names.index("exchange_read_0")
    merge = names.index("topk_merge_0")
    output = names.index("output_store_0")
    assert write < read < merge < output
    assert program.stages[write].work["mte3@x"] == 16384
    assert program.stages[read].work["mte2_aiv@x"] == 16384
    assert program.stages[output].work["mte3@x"] == 4096
    assert program.stages[merge].work["vector@x"] > 0


def test_q16_two_kv_heads_jointly_fits_ub_with_real_exchange():
    op = make_op(query_rows=16, context=16384, compression=4, topk=512,
                 heads=4, head_dim=32, kv_heads=2)
    geometry = _geometry(units=2, compute_units=1, cache_bytes=512 * 1024,
                         ub_bytes=196608, gm_bytes=262144)
    _, plan, mechanism = _mechanism(op, geometry)
    ledger = dict(op.metadata)["lightning_indexer_work"]
    assert ledger["query_key_pairs"] == 65564
    assert ledger["selection_output_bytes"] == 65536
    assert mechanism["key_access"]["pairs"] == 65564
    workspace = mechanism["workspace"]
    assert workspace["score_bytes_per_aiv"] == 65536
    assert workspace["list_bytes_per_aiv"] == 131072
    assert workspace["exchange_receive_bytes_per_aiv"] == 4096
    assert workspace["exchange_receive_additional_bytes_per_aiv"] == 0
    assert workspace["exchange_receive_alias"] == "score_slot"
    assert workspace["resident_bytes_per_aiv"] == 196608
    exchange = mechanism["exchange_storage"]
    assert exchange["total_bytes"] == 262144
    assert exchange["per_tile_bytes"] == 262144
    assert exchange["hbm_round_trip_total"] == 524288
    assert exchange["capacity_feedback"]["capacity_bytes"] == 262144
    assert exchange["capacity_feedback"]["fits"] is True
    assert plan.work.resources["mte3"] == 65536 + 262144
    program = plan.program(_context(
        op, geometry, split=True, pools=True))
    summed = {}
    for stage in program.stages:
        for role, amount in dict(stage.work).items():
            summed[role] = summed.get(role, 0) + amount
    assert summed["mte3@x"] == plan.work.resources["mte3"]
    assert summed["hbm@x"] == plan.work.resources["hbm"]
    names = [stage.name for stage in program.stages]
    assert names.index("exchange_write_0") < names.index("exchange_read_0")
    assert names.index("exchange_read_0") < names.index("topk_merge_0")
    assert names.index("topk_merge_0") < names.index("output_store_0")
    raw = _case()
    raw["machine"]["sku"] = "Ascend910_9362"
    result = graph_costs(raw, (op,), isolated=False)
    peaks = result["execution"]["peak_allocated_per_unit"]
    assert max(value for key, value in peaks.items()
               if key.startswith("ub@r0:s0:")) == 196608
    assert result["execution"]["peak_allocated_per_unit"][
        "l1@r0:s0:unit0"] == 524288
    assert result["execution"]["work"]["hbm_bytes"] == plan.work.resources["hbm"]
    assert result["execution"]["live_tasks"] == 0


def test_exchange_gm_grant_waits_for_one_full_tile_grant():
    from mindie_sim.prediction.engine import Allocation, Engine, Resource

    op = make_op(query_rows=512, context=32768, compression=4, topk=512)
    geometry = _geometry(units=64, compute_units=32, gm_bytes=131072)

    def make_engine():
        context = _context(op, geometry, split=True, pools=True)
        resources = []
        for role, name in context.resource_names.items():
            units = context.resource_units.get(role, context.core_count)
            resources.append(Resource(name, 1e9, "work", "fair", int(units)))
        return context, Engine(resources, pools={"gm": 131072},
                               pool_units={"gm": 1})

    plan = recipes()[RECIPE].prepare(op, geometry)[1]
    context, engine = make_engine()
    ready = engine.submit("ready")
    held = Allocation("held_gm", "gm", 1, (0,))
    blocker = engine.submit(
        "gm_blocker", acquire=(held,), free=(held,), latency=200000.0)
    done = plan.program(context).start(engine, "li", ready)
    engine.advance(until=(done,))
    assert blocker.time == pytest.approx(200000.0)
    assert done.time >= blocker.time
    assert engine.allocated["gm"] == 0

    context, engine = make_engine()
    ready = engine.submit("ready")
    done = plan.program(context).start(engine, "li", ready)
    engine.advance(until=(done,))
    assert done.time < blocker.time


def test_retained_k_still_panels_by_score_capacity():
    op = make_op(query_rows=16, context=16384, compression=4, topk=512,
                 heads=4, head_dim=32, kv_heads=1)
    _, _, mechanism = _mechanism(
        op, _geometry(units=2, compute_units=1, cache_bytes=512 * 1024,
                      ub_bytes=196608))
    assert mechanism["cache_levels"]["l1"]["policy"] == "retain_prefix"
    assert mechanism["key_access"]["pairs"] == 65564
    assert mechanism["key_access"]["panel_rows"] == 2048
    assert [panel["rows"] for tile in mechanism["key_access"]["tiles"]
            for panel in tile["panels"]] == [2048, 2048, 4]
    assert mechanism["workspace"]["score_bytes_per_aiv"] == 131072
    assert mechanism["workspace"]["list_bytes_per_aiv"] == 65536
    assert mechanism["workspace"]["resident_bytes_per_aiv"] == 196608
    assert mechanism["l1_allocation_bytes"] == 266496


def test_top1_panels_include_cross_panel_merge_comparisons():
    op = make_op(query_rows=1, context=4096, compression=4, topk=1,
                 heads=4, head_dim=128, kv_heads=1)
    _, plan, mechanism = _mechanism(
        op, _geometry(units=1, compute_units=1, cache_bytes=128 * 1024,
                      ub_bytes=196608))
    panels = [panel["rows"] for tile in mechanism["key_access"]["tiles"]
              for panel in tile["panels"]]
    assert panels == [448, 448, 128]
    assert plan.work.resources["vector"] == 12287
    assert mechanism["key_access"]["pairs"] == 1024
    program = plan.program(_context(
        op, _geometry(units=1, compute_units=1), split=True))
    summed = sum(dict(stage.work).get("vector@x", 0)
                 for stage in program.stages)
    assert summed == 12287


def test_score_and_list_workspace_is_streaming_bounded():
    op = make_op(query_rows=512, context=32768, compression=4, topk=512)
    geometry = _geometry(units=64, cache_bytes=512 * 1024,
                         ub_bytes=196608, gm_bytes=131072)
    _, plan, mechanism = _mechanism(op, geometry)
    workspace = mechanism["workspace"]
    assert workspace["pool"] == "ub"
    assert workspace["score_bytes_per_aiv"] == 126976
    assert workspace["list_bytes_per_aiv"] == 65536
    assert workspace["exchange_receive_bytes_per_aiv"] == 4096
    assert workspace["exchange_receive_additional_bytes_per_aiv"] == 0
    assert workspace["exchange_receive_alias"] == "score_slot"
    assert workspace["resident_bytes_per_aiv"] == 192512
    assert workspace["full_scores_if_materialized"] == 16907776
    assert workspace["full_lists_if_materialized"] == 4194304
    assert workspace["score_bytes_per_aiv"] + \
        workspace["list_bytes_per_aiv"] <= 196608
    assert mechanism["key_access"]["panel_rows"] == 1984
    exchange = mechanism["exchange_storage"]
    assert exchange["per_tile_bytes"] == 131072
    assert exchange["total_bytes"] == 4194304
    assert exchange["per_aiv_list_bytes"] == 65536
    assert exchange["per_aiv_write_bytes"] == 65536
    assert exchange["receive_bytes"] == 4096
    assert exchange["receive_alias"] == "score_slot"
    assert exchange["grant"] == "index_exchange_gm"
    assert exchange["grant_reused"] is False
    assert exchange["hbm_round_trip_per_tile"] == 262144
    assert exchange["hbm_round_trip_total"] == 8388608
    assert exchange["capacity_feedback"]["capacity_bytes"] == 131072
    assert exchange["capacity_feedback"]["fits"] is True
    assert exchange["capacity_feedback"]["waits_for_grant"] is False
    assert exchange["allocations"] == ("index_lists_ub_0", "index_lists_ub_1")
    program = plan.program(_context(op, geometry, split=True, pools=True))
    names = [stage.name for stage in program.stages]
    update = names.index("topk_update_0_0")
    write = names.index("exchange_write_0")
    read = names.index("exchange_read_0")
    merge = names.index("topk_merge_0")
    output = names.index("output_store_0")
    release = names.index("index_release")
    assert update < write < read < merge < output < release
    assert names.index("topk_update_31_0") < names.index("exchange_write_31")
    assert names.index("exchange_write_31") < names.index("exchange_read_31")
    assert names.index("exchange_read_31") < names.index("topk_merge_31")
    assert names.index("topk_merge_31") < names.index("output_store_31") < release
    # One per-tile GM slot is reused; each next write waits for the prior read.
    assert "exchange_read_0" in program.stages[names.index("exchange_write_1")].after
    acquire = next(stage for stage in program.stages
                   if stage.name == "index_reserve").acquire
    acquired = {allocation.name for allocation in acquire}
    assert {"index_lists_ub_0", "index_lists_ub_1",
            "index_score_ub_0", "index_score_ub_1",
            "index_q_l1", "index_k_l1", "index_exchange_gm"} <= acquired
    gm_allocation = next(allocation for allocation in acquire
                         if allocation.name == "index_exchange_gm")
    assert gm_allocation.size == 131072 and gm_allocation.units == (0,)
    free = next(stage for stage in program.stages
                if stage.name == "index_release").free
    assert "index_exchange_gm" in {allocation.name for allocation in free}


def test_caller_exchange_grant_is_bound_once():
    op = make_op(query_rows=512, context=32768, compression=4, topk=512,
                 exchange_grant={"name": "caller_gm", "bytes": 131072})
    geometry = _geometry(units=64, cache_bytes=512 * 1024,
                         ub_bytes=196608, gm_bytes=131072)
    _, plan, mechanism = _mechanism(op, geometry)
    exchange = mechanism["exchange_storage"]
    assert exchange["grant"] == "caller_gm"
    assert exchange["gm_allocation"] == "caller_gm"
    assert exchange["grant_reused"] is True
    assert exchange["capacity_feedback"]["allocation_bytes"] == 0
    assert exchange["total_bytes"] == 4194304
    assert mechanism["cache_levels"]["gm"]["scope"] == "caller_grant"
    program = plan.program(_context(op, geometry, split=True, pools=True))
    acquire = next(stage for stage in program.stages
                   if stage.name == "index_reserve").acquire
    assert "index_exchange_gm" not in {item.name for item in acquire}
    # The caller's grant still covers the same physical exchange work.
    assert plan.work.resources["mte3"] == 5242880
    assert plan.work.resources["hbm"] >= 2 * 4194304


def test_exchange_gm_capacity_smaller_than_the_tile_is_not_silently_shrunk():
    op = make_op(query_rows=512, context=32768, compression=4, topk=512)
    with pytest.raises(ModelingError, match="exchange_gm_capacity"):
        _mechanism(op, _geometry(units=64, cache_bytes=512 * 1024,
                                 ub_bytes=196608, gm_bytes=131071))


def test_full_k_pool_geometry_executes_with_q_reservation():
    op = make_op(query_rows=32, context=8160, compression=4, cache_owner=16)
    result = graph_costs(_case(), (op,), isolated=False)
    assert result["execution"]["peak_allocated_per_unit"][
        "l1@r0:s0:unit16"] == 524288
    assert result["execution"]["live_allocated"]["l1@r0:s0"] == 0
    assert result["execution"]["live_tasks"] == 0


def test_general_mte2_and_split_use_one_read_port_in_summary_and_stages():
    op = make_op(query_rows=32, context=4096, compression=4)
    split_geometry = _geometry(units=64, cache_bytes=128 * 1024)
    generic_geometry = _geometry(units=64, cache_bytes=128 * 1024)
    generic_geometry = Geometry(
        generic_geometry.compute_units,
        {key: value for key, value in generic_geometry.resource_units.items()
         if key not in {"mte2_aic", "mte2_aiv"}} | {"mte2": 32},
        dict(generic_geometry.buffers), {})
    for geometry, split in ((split_geometry, True), (generic_geometry, False)):
        prepared, plan = recipes()[RECIPE].prepare(op, geometry)
        mechanism = dict(prepared.metadata)["program_execution"]["mechanism"]
        program = plan.program(_context(op, geometry, split=split))
        summed = {}
        for stage in program.stages:
            for role, amount in dict(stage.work).items():
                summed[role] = summed.get(role, 0) + amount
        read = mechanism["read_port"]
        assert mechanism["split_mte2"] is split
        if split:
            assert "mte2_aic@x" in summed and "mte2_aiv@x" in summed
            assert "mte2@x" not in summed
            assert summed["mte2_aic@x"] == plan.work.resources["mte2_aic"]
            assert summed["mte2_aiv@x"] == plan.work.resources["mte2_aiv"]
        else:
            assert "mte2@x" in summed
            assert "mte2_aic@x" not in summed and "mte2_aiv@x" not in summed
            assert summed["mte2@x"] == plan.work.resources["mte2"]
        read_name = "mte2_aic@x" if split else "mte2@x"
        assert read in {"mte2_aic", "mte2"}
        assert read_name in summed
    generic_owner = make_op(query_rows=32, context=4096, compression=4,
                            cache_owner=16)
    generic_owner_geometry = Geometry(
        generic_geometry.compute_units,
        dict(generic_geometry.resource_units),
        dict(generic_geometry.buffers), {})
    _, generic_owner_plan = recipes()[RECIPE].prepare(
        generic_owner, generic_owner_geometry)
    generic_program = generic_owner_plan.program(
        _context(generic_owner, generic_owner_geometry, split=False))
    exchange_read = next(stage for stage in generic_program.stages
                         if stage.name == "exchange_read_0")
    assert exchange_read.placement["mte2@x"] == (16,)


def test_engine_allocation_models_owner_capacity_and_release():
    raw = _case()
    same = (make_op(query_rows=32, context=4096, op_id="k_owner0_a",
                    cache_owner=0, stream="a"),
            make_op(query_rows=32, context=4096, op_id="k_owner0_b",
                    cache_owner=0, stream="b"))
    same_result = graph_costs(raw, same, isolated=False)
    rows = {row["id"]: row for row in same_result["timeline"]}
    first, second = rows["r0:k_owner0_a"], rows["r0:k_owner0_b"]
    assert first["ready_ms"] == 0 and second["ready_ms"] == 0
    assert second["complete_ms"] > first["complete_ms"]
    per_unit = same_result["execution"]["peak_allocated_per_unit"]
    assert per_unit["l1@r0:s0:unit0"] == 280576
    assert same_result["execution"]["allocation_events"] >= 4
    assert same_result["execution"]["live_allocated"]["l1@r0:s0"] == 0

    roomy = _case()
    roomy.setdefault("machine", {}).setdefault("device", {}).setdefault(
        "buffers", {}).update({"l1": 1024 * 1024, "ub": 1024 * 1024})
    roomy_result = graph_costs(roomy, same, isolated=False)
    roomy_rows = {row["id"]: row for row in roomy_result["timeline"]}
    assert roomy_rows["r0:k_owner0_a"]["complete_ms"] == pytest.approx(
        roomy_rows["r0:k_owner0_b"]["complete_ms"])
    assert roomy_result["execution"]["peak_allocated_per_unit"][
        "l1@r0:s0:unit0"] == 561152

    different = (same[0], make_op(query_rows=32, context=4096,
                                  op_id="k_owner1_b", cache_owner=1,
                                  stream="b"))
    different_result = graph_costs(raw, different, isolated=False)
    rows = {row["id"]: row for row in different_result["timeline"]}
    left, right = rows["r0:k_owner0_a"], rows["r0:k_owner1_b"]
    assert left["ready_ms"] == 0 and right["ready_ms"] == 0
    assert left["complete_ms"] == pytest.approx(right["complete_ms"])
    units = different_result["execution"]["peak_allocated_per_unit"]
    assert units["l1@r0:s0:unit0"] == 280576
    assert units["l1@r0:s0:unit1"] == 280576


def test_l2_pool_is_allocated_as_one_shared_scope():
    raw = _case()
    raw.setdefault("machine", {}).setdefault("device", {}).setdefault(
        "buffers", {})["l2"] = 8 * 1024 * 1024
    op = make_op(query_rows=32, context=4096, compression=4,
                 l2_capacity_bytes=8 * 1024 * 1024)
    result = graph_costs(raw, (op,), isolated=False)
    units = result["execution"]["peak_allocated_per_unit"]
    l2_units = {key: value for key, value in units.items()
                if key.startswith("l2@") and value}
    assert l2_units == {"l2@r0:s0:unit0": 264192}
    assert result["execution"]["peak_allocated"]["l2@r0:s0"] == 264192
    assert result["execution"]["live_allocated"]["l2@r0:s0"] == 0


def test_operator_native_and_graph_consume_one_plan():
    native = _native()
    direct = make_op(query_rows=32, context=4096, heads=4, head_dim=128,
                     topk=512, compression=4, page_size=128, kv_heads=1,
                     weight_dtype="fp32", cache_owner=0,
                     block_table_columns=64)
    geometry = _geometry(units=64, cache_bytes=512 * 1024)
    native_prepared, native_plan = recipes()[RECIPE].prepare(native, geometry)
    direct_prepared, direct_plan = recipes()[RECIPE].prepare(direct, geometry)
    assert native_plan.work == direct_plan.work
    assert dict(native_prepared.metadata)["resource_work"] ==         dict(direct_prepared.metadata)["resource_work"]
    native_mechanism = dict(native_prepared.metadata)["program_execution"]["mechanism"]
    direct_mechanism = dict(direct_prepared.metadata)["program_execution"]["mechanism"]
    assert native_mechanism["cache_levels"] == direct_mechanism["cache_levels"]
    assert native_mechanism["key_access"] == direct_mechanism["key_access"]
    result = graph_costs(_case(), (native,), isolated=False)
    assert result["execution"]["work"]["hbm_bytes"] == native_prepared.hbm_bytes
    assert result["execution"]["work"]["vector_ops"] == pytest.approx(
        native_prepared.vector_ops)
    assert result["execution"]["peak_allocated_per_unit"]["l1@r0:s0:unit0"] == \
        native_mechanism["l1_allocation_bytes"]
    assert result["execution"]["allocation_events"] >= 2


def test_top1_selection_is_a_max_reduction_not_zero():
    op = make_op(query_rows=5, context=100, compression=4, topk=1)
    ledger = dict(op.metadata)["lightning_indexer_work"]
    pairs = sum((100 + i + 1) // 4 for i in range(5))
    visible_rows = sum(1 for i in range(5) if (100 + i + 1) // 4 > 0)
    assert ledger["query_key_pairs"] == pairs
    assert ledger["topk_comparison_ops"] == pairs - visible_rows > 0
    prepared, _, _ = _mechanism(op, _geometry())
    assert prepared.vector_ops >= ledger["topk_comparison_ops"]


def _mechanism(op, geometry):
    prepared, plan = recipes()[RECIPE].prepare(op, geometry)
    mechanism = dict(prepared.metadata)["program_execution"]["mechanism"]
    return prepared, plan, mechanism


def test_cache_reload_and_local_topk_exchange_are_program_stages():
    op = make_op(query_rows=512, context=32768, topk=512)
    prepared, plan, mechanism = _mechanism(op, _geometry(units=64))
    assert mechanism["query_tile_rows"] == 16
    ledger = dict(op.metadata)["lightning_indexer_work"]
    assert mechanism["key_transport_bytes"] > ledger["unique_key_bytes"]
    assert mechanism["local_topk_lists"] > 512
    assert mechanism["exchange_bytes"] > 0
    program = plan.program(_Bind())
    names = [stage.name for stage in program.stages]
    assert names[0] == "index_reserve"
    assert "query_load_0" in names
    assert sum(name.startswith("key_load_") for name in names) >= \
        len(mechanism["key_access"]["tiles"])
    assert any(name.startswith("topk_update_") for name in names)
    assert any(name.startswith("topk_merge_") for name in names)
    assert names[-1] == "index_release"
    # Stage decomposition conserves the plan's typed work.
    summed = {}
    for stage in program.stages:
        for role, amount in dict(stage.work).items():
            summed[role] = summed.get(role, 0) + amount
    assert summed["hbm"] == plan.work.resources["hbm"]
    assert summed["mte3"] == plan.work.resources["mte3"]
    assert summed["vector"] == pytest.approx(plan.work.resources["vector"])
    assert summed["bf16"] == plan.work.matrix["bf16"]
    assert prepared.hbm_bytes == (mechanism["key_hbm_bytes"] + ledger["query_bytes"]
                                  + ledger["head_weight_bytes"] + ledger["metadata_read_bytes"]
                                  + ledger["selection_output_bytes"]
                                  + 2 * mechanism["exchange_bytes"])
    # A single vector owner has no cross-owner list to exchange.
    _, _, single = _mechanism(op, _geometry(units=1))
    assert single["exchange_bytes"] == 0
    assert single["local_list_merge_ops"] == 0
    single_names = [stage.name for stage in recipes()[RECIPE].prepare(
        op, _geometry(units=1))[1].program(_Bind()).stages]
    assert any(name.startswith("topk_merge_") for name in single_names)
    assert single_names[-1] == "index_release"


def test_reload_and_exchange_trends_follow_geometry_and_owners():
    short = make_op(query_rows=16, context=32768)
    long = make_op(query_rows=512, context=32768)
    _, _, short_m = _mechanism(short, _geometry(units=64))
    _, _, long_m = _mechanism(long, _geometry(units=64))
    assert long_m["key_transport_bytes"] > 32 * short_m["key_transport_bytes"] / 16
    assert long_m["exchange_bytes"] > short_m["exchange_bytes"]
    wide_context = make_op(query_rows=512, context=65536)
    _, _, wide_m = _mechanism(wide_context, _geometry(units=64))
    assert wide_m["key_transport_bytes"] > 1.9 * long_m["key_transport_bytes"]
    exchanges = []
    for units in (4, 16, 64):
        _, _, mechanism = _mechanism(long, _geometry(units=units))
        exchanges.append(mechanism["exchange_bytes"])
    assert exchanges == sorted(exchanges)
    assert exchanges[0] < exchanges[-1]


def test_bf16_weight_storage_is_upcast_without_changing_qk_work():
    fp32 = make_op(query_rows=64, context=8192, weight_dtype="fp32")
    bf16 = make_op(query_rows=64, context=8192, weight_dtype="bf16")
    fp32_ledger = dict(fp32.metadata)["lightning_indexer_work"]
    bf16_ledger = dict(bf16.metadata)["lightning_indexer_work"]
    assert fp32.flops == bf16.flops
    assert bf16_ledger["head_weight_bytes"] * 2 == fp32_ledger["head_weight_bytes"]
    fp32_prepared, _, _ = _mechanism(fp32, _geometry())
    bf16_prepared, _, _ = _mechanism(bf16, _geometry())
    assert bf16_prepared.vector_ops - fp32_prepared.vector_ops == 64 * 4
    assert bf16_prepared.hbm_bytes < fp32_prepared.hbm_bytes


def test_engine_cost_responds_to_the_declared_pipes():
    op = make_op(query_rows=512, context=32768, topk=512)
    raw = _case()
    base = graph_costs(raw, (op,), isolated=False)["duration_ms"]
    slower = {}
    for role in ("vector", "mte2_aic", "hbm"):
        changed = deepcopy(raw)
        changed.setdefault("machine", {}).setdefault("device", {}).setdefault(
            "rates", {})[role] = 1e3
        slower[role] = graph_costs(changed, (op,), isolated=False)["duration_ms"]
    assert all(value > base * 10 for value in slower.values())
