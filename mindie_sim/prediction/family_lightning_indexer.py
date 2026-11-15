"""BF16 LightningIndexerV2 on the common resource engine.

[Role] Declare one causal compressed-key scan, weighted head reduction and TopK
  as a resource/phase program shared by independent-operator, native-input and
  graph consumers.
[Invariants] BF16 Q/K and FP32 head weights remain distinct from the INT8 QLI
  path; work follows declared tensors, compression, cache and TopK geometry and
  is never selected from shape/measurement tables. Native cache reload and
  partial TopK exchange are analytical mechanisms on the declared call
  boundary, not permanently deferred as an "unresolved lower bound".
[Timeouts & Locks] Closed-form or O(query rows / tile) integer arithmetic; no
  I/O, device contexts or locks.
[Side-Effect Policy] Return immutable operators and pure executable plans only.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import _floor_sum
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from .cache_path import CacheGeometry, CacheState
from .indexer_source import select_indexer_source
from .operators.mixed_core import read_role
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .ownership import unit_group
from .program import KernelProgram, Stage


IMPLEMENTATION = "cann_bf16_lightning_indexer_v2_complete"
RECIPE = "bf16_lightning_indexer_v2"
SOURCE = {
    "caller_repository": "vllm-ascend",
    "caller_revision": "3c1b834495af94efb2af1f1a91da2fc8bd18441e",
    "caller_path": "vllm_ascend/models/qwen4_exp/lightning_indexer.py",
    "caller_lines": [155, 268],
    "binding_sha256": "9af7711bcabc3becc4dc4b97dce81551b1307830e8c1d28ae181c5f6c1880b53",
    "boundary": "one native BF16 LightningIndexerV2 MIX_AIC kernel",
    "dispatch_identity": "caller ABI and profiler boundary; native binary identity not captured",
}

# Native MIX_AIC M tile shared with the quantized sibling.  It bounds how many
# query rows reuse one resident key tile before the next key stream starts.
QUERY_TILE_ROWS = 16
# A local TopK list carries a score and a token id; the exchange moves both.
LOCAL_LIST_ELEMENT_BYTES = 8
CACHE_LINE_BYTES = 128
CACHE_TRANSFER_BYTES = 32
WEIGHT_DTYPES = frozenset({"fp32", "bf16", "fp16"})
_DTYPE_ALIASES = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}
NATIVE_INPUTS = ("query", "key_cache", "weights", "query_lengths",
                 "key_lengths", "block_table")
NATIVE_OUTPUTS = ("indices",)


def recipes():
    return {RECIPE: PlannedRecipe(_build)}


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError("bf16_lightning_indexer_invalid_" + name)
    return value


def _ceil_div(value, divisor):
    return -(-int(value) // int(divisor))


def _compressed_rows(query_rows, context, compression):
    """Materialized compressed K rows for one contiguous span."""
    return (int(context) + int(query_rows)) // int(compression)


def _physical_key_pages(compressed_rows, key_rows_per_page):
    return _ceil_div(compressed_rows, key_rows_per_page)


def causal_pairs(query_rows, context, compression=4):
    """Independent rows see floor((context + row + 1) / compression) keys."""
    _integer(query_rows, "query_rows", minimum=1)
    _integer(context, "context")
    _integer(compression, "compression", minimum=1)
    return _floor_sum(query_rows, compression, 1, context + 1)


def _gather_spans(query_rows, context, query_lengths, context_lengths):
    queries = tuple(query_lengths or ())
    contexts = tuple(context_lengths or ())
    if bool(queries) != bool(contexts):
        raise ModelingError("bf16_lightning_indexer_incomplete_sequence_lengths")
    if queries:
        if len(queries) != len(contexts):
            raise ModelingError("bf16_lightning_indexer_incomplete_sequence_lengths")
        spans = []
        for q, c in zip(queries, contexts):
            _integer(q, "query_length", minimum=1)
            _integer(c, "context_length")
            spans.append((q, c))
        total = sum(q for q, _ in spans)
        if query_rows is not None and _integer(query_rows, "query_rows", minimum=1) != total:
            raise ModelingError("bf16_lightning_indexer_query_rows_disagree_with_lengths")
        if context is not None and _integer(context, "context") != max(
                c for _, c in spans):
            raise ModelingError("bf16_lightning_indexer_context_disagrees_with_lengths")
        return tuple(spans)
    if query_rows is None or context is None:
        raise ModelingError("bf16_lightning_indexer_geometry_required")
    return ((_integer(query_rows, "query_rows", minimum=1),
             _integer(context, "context")),)


def _weighted_vector_ops(pairs, heads, kv_heads):
    """One weight multiply, ReLU and accumulate per head, then one sum."""
    return pairs * (3 * heads - kv_heads)


def _sentinel_elements(spans, compression, topk, kv_heads):
    """Missing TopK slots initialized per query row and KV head."""
    total = 0
    for q, c in spans:
        short = max(0, min(q, topk * compression - c - 1))
        total += short * topk - _floor_sum(short, compression, 1, c + 1)
    return total * kv_heads


def _visible_rows(spans, compression):
    """Rows that own at least one visible compressed key."""
    return sum(max(0, q - max(0, compression - c - 1)) for q, c in spans)


def _topk_comparisons(pairs, topk, kv_heads, spans, compression):
    """Selection comparisons per (query row, KV head) domain.

    A heap selects TopK for K>1; Top1 is a maximum reduction over its visible
    candidates and therefore needs ``visible - 1`` comparisons, not zero.
    """
    if topk == 1:
        return max(0, pairs - _visible_rows(spans, compression)) * kv_heads
    return pairs * kv_heads * math.log2(topk)


def _exchange_counts(spans, compression, topk, units, kv_heads=1):
    """Local TopK lists and their merge cost per (query row, KV head).

    Keys are partitioned over at most ``units`` vector owners.  Every owner
    keeps up to ``topk`` candidates (the correctness capacity of a global
    TopK merge), so a row needs ``min(units, ceil(visible / topk))`` local
    lists.  One list needs no exchange.  Each KV head owns an independent
    selection domain and therefore its own lists and merge.
    """
    if units <= 1:
        # One owner keeps a single local list per row that owns a key; no list
        # is ever exchanged.
        lists = _visible_rows(spans, compression)
        return 0, lists * kv_heads, 0.0
    list_bytes = 0
    list_sum = 0
    merge_ops = 0.0
    for q, c in spans:
        for lists in range(1, units + 1):
            # visible(i) = floor((c + i + 1) / compression); this bucket holds
            # rows whose ceil(visible / topk) is exactly ``lists``.
            first = max(0, ((lists - 1) * topk + 1) * compression - c - 1)
            last = (q if lists == units
                    else min(q, (lists * topk + 1) * compression - c - 1))
            count = max(0, last - first)
            if not count:
                continue
            list_sum += count * lists
            if lists == 1:
                continue
            # Every owner keeps min(topk, its share) candidates, so the union
            # actually exchanged is min(visible, lists * topk).  For interior
            # buckets that is the visible count; the last bucket can saturate.
            if lists == units:
                cut = max(first, min(last, (units * topk + 1) * compression - c - 1))
                entries = _floor_sum(max(0, cut - first), compression, 1, c + first + 1)
                entries += max(0, last - cut) * units * topk
            else:
                entries = _floor_sum(count, compression, 1, c + first + 1)
            list_bytes += entries * LOCAL_LIST_ELEMENT_BYTES
            merge_ops += entries * max(1.0, math.log2(lists))
    return (list_bytes * kv_heads, list_sum * kv_heads, merge_ops * kv_heads)


def _descriptor(value, name, rank, dtypes, layouts):
    if isinstance(value, (tuple, list)) and len(value) == 3:
        shape, dtype, layout = value
    elif isinstance(value, dict):
        shape = value.get("shape")
        dtype = value.get("dtype")
        layout = value.get("layout", "ND")
    else:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_descriptor_invalid")
    if not isinstance(shape, (tuple, list)) or len(shape) != rank:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_rank")
    shape = tuple(shape)
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_shape")
    dtype = str(dtype).strip().lower()
    dtype = _DTYPE_ALIASES.get(dtype, dtype)
    if dtype not in dtypes:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_dtype")
    if str(layout) not in layouts:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_layout")
    return shape, dtype, str(layout)


def native_op(*, query, key_cache, weights, block_table, query_lengths,
              context_lengths, topk, compression=4, page_size=None,
              declaration=None, cache_owner=0, l2_capacity_bytes=None,
              exchange_grant=None,
              op_id="lightning_indexer", deps=(), stage=0,
              layer=0, requests=(), phase="prefill", stream="",
              binding_revision="unknown", runtime_version="unknown"):
    """Build the BF16 LI operator from its real native tensor inputs.

    ``query`` is TND ``[tokens, heads, dim]`` BF16, ``key_cache`` is paged
    ``[pages, page, kv_heads, dim]`` BF16, ``weights`` is
    ``[tokens, heads]`` FP32 (BF16/FP16 storage is accepted and upcast), and
    ``block_table`` is an INT32 ``[requests, columns]`` page table.  Dynamic
    query/key lengths are explicit request state, never inferred from padding.
    """
    q_shape, q_dtype, q_layout = _descriptor(
        query, "query", 3, {"bf16"}, {"TND"})
    k_shape, k_dtype, k_layout = _descriptor(
        key_cache, "key_cache", 4, {"bf16"}, {"ND", "PA_BBND"})
    w_shape, w_dtype, w_layout = _descriptor(
        weights, "weights", 2, WEIGHT_DTYPES, {"ND"})
    b_shape, b_dtype, b_layout = _descriptor(
        block_table, "block_table", 2, {"int32"}, {"ND"})
    if q_shape[1] != w_shape[1] or w_shape[0] != q_shape[0]:
        raise ModelingError("bf16_lightning_indexer_native_weight_geometry")
    if k_shape[3] != q_shape[2]:
        raise ModelingError("bf16_lightning_indexer_native_head_dim")
    compression = _integer(compression, "compression", minimum=1)
    topk = _integer(topk, "topk", minimum=1)
    key_rows_per_page = k_shape[1]
    if page_size is not None and _integer(
            page_size, "page_size", minimum=1) != key_rows_per_page:
        raise ModelingError("bf16_lightning_indexer_native_page_size")
    page_size = key_rows_per_page
    spans = _gather_spans(None, None, query_lengths, context_lengths)
    if len(spans) != b_shape[0]:
        raise ModelingError("bf16_lightning_indexer_native_block_table_requests")
    if q_shape[0] != sum(q for q, _ in spans):
        raise ModelingError("bf16_lightning_indexer_native_query_lengths")
    columns = b_shape[1]
    compressed_rows = tuple(_compressed_rows(q, c, compression) for q, c in spans)
    key_pages = tuple(_physical_key_pages(rows, key_rows_per_page)
                      for rows in compressed_rows)
    logical_pages = tuple(_ceil_div(c + q, key_rows_per_page * compression)
                          for q, c in spans)
    if any(pages > columns or rows > k_shape[0] * key_rows_per_page
           for rows, pages in zip(compressed_rows, key_pages)):
        raise ModelingError("bf16_lightning_indexer_native_block_table_capacity")
    if declaration is not None:
        selected = select_indexer_source(dict(declaration))
        if selected.get("status") != "bf16_li_v2":
            raise ModelingError(
                "bf16_lightning_indexer_native_source_not_bf16:"
                + str(selected.get("status") or "unknown"))
    op = make_op(
        query_rows=q_shape[0], op_id=op_id, deps=deps, stage=stage, layer=layer,
        requests=requests, phase=phase, stream=stream,
        heads=q_shape[1], head_dim=q_shape[2],
        kv_heads=k_shape[2], topk=topk, compression=compression,
        page_size=page_size, query_dtype=q_dtype, cache_dtype=k_dtype,
        weight_dtype=w_dtype, query_lengths=tuple(q for q, _ in spans),
        context_lengths=tuple(c for _, c in spans),
        block_table_columns=columns, cache_owner=cache_owner,
        l2_capacity_bytes=l2_capacity_bytes,
        exchange_grant=exchange_grant,
        binding_revision=binding_revision, runtime_version=runtime_version)
    binding = {
        "kind": "independent_native_tensor_abi",
        "native_kernel": "LightningIndexer",
        "implementation": IMPLEMENTATION,
        "source_boundary": "kernel",
        "inputs": {
            "query": {"shape": q_shape, "dtype": q_dtype, "layout": q_layout},
            "key_cache": {"shape": k_shape, "dtype": k_dtype, "layout": k_layout},
            "weights": {"shape": w_shape, "dtype": w_dtype, "layout": w_layout},
            "block_table": {"shape": b_shape, "dtype": b_dtype, "layout": b_layout},
        },
        "dynamic_state": {
            "query_lengths": [q for q, _ in spans],
            "key_lengths": [c + q for q, c in spans],
        },
        "cache_policy": {
            "owner": cache_owner,
            "l2_capacity_bytes": l2_capacity_bytes,
            "line_bytes": CACHE_LINE_BYTES,
            "transfer_bytes": CACHE_TRANSFER_BYTES,
        },
        "cache_capacity": {
            "key_rows_per_page": key_rows_per_page,
            "compressed_key_rows": list(compressed_rows),
            "physical_key_pages": list(key_pages),
            "logical_token_pages": list(logical_pages),
            "logical_tokens_per_page": key_rows_per_page * compression,
            "block_table_columns": columns,
        },
        "outputs": {"indices": {"shape": (q_shape[0], k_shape[2], int(topk)),
                                "dtype": "int32", "layout": "ND"}},
        "source_selection": ("declared_bf16_li" if declaration is not None
                             else "native_dtypes_bf16"),
        "unresolved_outputs": ("retained tensor contract leaves output[1] "
                               "identity/shape unresolved; index selection consumes output[0]"),
    }
    metadata = dict(op.metadata)
    metadata["native_input_binding"] = binding
    metadata["native_implementation"] = {
        "native_kind": "LightningIndexer", "selection": IMPLEMENTATION,
        "conditions": {"submitted_kernels": 1,
                       "source_boundary": "qk_weighted_head_reduce_topk",
                       "upstream_metadata_generation": "outside"},
    }
    return replace(op, metadata=tuple(sorted(metadata.items())))


def make_op(*, query_rows=None, context=None, op_id="lightning_indexer", deps=(),
            stage=0, layer=0, requests=(), phase="prefill", stream="", heads=4,
            head_dim=128, topk=512, compression=4, page_size=128, kv_heads=1,
            query_dtype="bf16", cache_dtype="bf16", weight_dtype="fp32",
            query_lengths=(), context_lengths=(), block_table_columns=None,
            cache_owner=0, l2_capacity_bytes=None, exchange_grant=None,
            binding_revision="unknown", runtime_version="unknown"):
    """One or more contiguous causal spans with an explicit native geometry.

    Disjoint PCP spans need separate calls because each call rereads its own
    metadata and key range.  The caller supplies an observed binding/runtime
    identity explicitly; the analytical operator also works with unknown or
    different binary identity.
    """
    spans = _gather_spans(query_rows, context, query_lengths, context_lengths)
    for name, value in (("heads", heads), ("head_dim", head_dim),
                        ("topk", topk), ("page_size", page_size),
                        ("kv_heads", kv_heads)):
        _integer(value, name, minimum=1)
    if heads % kv_heads:
        raise ModelingError("bf16_lightning_indexer_kv_heads_must_divide_heads")
    cache_owner = _integer(cache_owner, "cache_owner", minimum=0)
    if l2_capacity_bytes is not None:
        l2_capacity_bytes = _integer(
            l2_capacity_bytes, "l2_capacity_bytes", minimum=0)
    if exchange_grant is not None:
        if isinstance(exchange_grant, dict):
            grant_name = str(exchange_grant.get("name")
                             or exchange_grant.get("grant") or "")
            grant_bytes = exchange_grant.get("bytes")
        else:
            grant_name, grant_bytes = str(exchange_grant), None
        if not grant_name:
            raise ModelingError("bf16_lightning_indexer_exchange_grant_invalid")
        if grant_bytes is not None:
            grant_bytes = _integer(grant_bytes, "exchange_grant_bytes")
        exchange_grant = {"name": grant_name, "bytes": grant_bytes}
    if query_dtype != "bf16" or cache_dtype != "bf16":
        raise ModelingError("bf16_lightning_indexer_v2_domain_mismatch")
    weight_dtype = str(weight_dtype).strip().lower()
    if weight_dtype not in WEIGHT_DTYPES:
        raise ModelingError("bf16_lightning_indexer_v2_domain_mismatch")
    rows = sum(q for q, _ in spans)
    pairs = sum(causal_pairs(q, c, compression) for q, c in spans)
    compressed_rows = tuple(_compressed_rows(q, c, compression) for q, c in spans)
    key_rows = sum(compressed_rows)
    # ``page_size`` is key rows per physical compressed-K page, not logical
    # (uncompressed) tokens.  The caller's token page relation is separate.
    key_pages = tuple(_physical_key_pages(count, page_size)
                      for count in compressed_rows)
    logical_pages = tuple(_ceil_div(c + q, page_size * compression)
                          for q, c in spans)
    required_columns = max(key_pages)
    if block_table_columns is None:
        columns = required_columns
    else:
        columns = _integer(block_table_columns, "block_table_columns", minimum=1)
        if columns < required_columns:
            raise ModelingError("bf16_lightning_indexer_block_table_too_short")
    batch = len(spans)
    # Keep native payload and logical arithmetic distinct.  HBM is the unique
    # external tensor payload; native query-tile reloads and local-list
    # exchange are added by the executable plan from declared mechanisms.
    query_bytes = rows * heads * head_dim * dtype_bytes("bf16")
    key_bytes = key_rows * kv_heads * head_dim * dtype_bytes("bf16")
    weight_bytes = rows * heads * dtype_bytes(weight_dtype)
    metadata_bytes = batch * 8 + batch * columns * 4
    output_bytes = rows * kv_heads * topk * 4
    weighted_ops = _weighted_vector_ops(pairs, heads, kv_heads)
    comparison_ops = _topk_comparisons(pairs, topk, kv_heads, spans, compression)
    sentinel = _sentinel_elements(spans, compression, topk, kv_heads)
    vector_ops = weighted_ops + comparison_ops + sentinel
    read_bytes = query_bytes + key_bytes + weight_bytes + metadata_bytes
    qk_flops = 2 * pairs * heads * head_dim
    shape = (rows, heads, head_dim)
    ledger = {
        "query_rows": rows, "context": max(c for _, c in spans),
        "query_key_pairs": pairs,
        "visible_compressed_keys": key_rows, "heads": heads,
        "kv_heads": kv_heads, "head_dim": head_dim, "topk": topk,
        "compression": compression, "page_size": page_size,
        "sequence_partition": tuple((q, c) for q, c in spans),
        "block_table_columns": columns,
        "key_rows_per_page": page_size,
        "compressed_key_rows": key_rows,
        "compressed_key_rows_by_request": compressed_rows,
        "physical_key_pages": key_pages,
        "logical_token_pages": logical_pages,
        "logical_tokens_per_page": page_size * compression,
        "query_tile_rows": QUERY_TILE_ROWS,
        "cache_owner": cache_owner,
        "l2_capacity_bytes": l2_capacity_bytes,
        "exchange_grant": exchange_grant,
        "mechanism_parameters": {
            "cache_reload": "one visible-key stream per query tile",
            "local_topk_exchange": "one local list per vector owner, capacity topk",
            "local_list_element_bytes": LOCAL_LIST_ELEMENT_BYTES,
            "exchange_storage": ("one per-tile GM slot reused across tiles; "
                                 "caller grant bound once when supplied"),
            "exchange_receive": ("topk entries per row/KV head; aliases the "
                                 "score slot when it fits"),
        },
        "qk_bf16_flops": qk_flops,
        "weighted_head_reduce_ops": weighted_ops,
        "topk_comparison_ops": comparison_ops,
        "sentinel_fill_elements": sentinel,
        "query_bytes": query_bytes, "unique_key_bytes": key_bytes,
        "head_weight_bytes": weight_bytes, "metadata_read_bytes": metadata_bytes,
        "selection_output_bytes": output_bytes,
        "read_bytes_lower_bound": read_bytes,
        "weight_dtype": weight_dtype,
        "traffic_scope": ("unique HBM payload; query-tile cache reload and "
                          "local TopK exchange are plan mechanisms"),
        "selection_scope": "logical weighted-ReLU/head-sum and TopK comparisons",
    }
    implementation = {
        "name": IMPLEMENTATION, "binding_revision": binding_revision,
        "runtime_version": runtime_version,
    }
    view = {
        "query_shape": shape, "query_dtype": query_dtype,
        "cache_dtype": cache_dtype, "cache_head_shape": (kv_heads, head_dim),
        "weight_dtype": weight_dtype, "layout_q": "TND",
        "layout_k": "PA_BBND", "mask_mode": 3, "return_value": 0,
        "compression_ratio": compression, "page_size": page_size,
        "key_page_rows": page_size, "block_table_columns": columns,
        "logical_tokens_per_page": page_size * compression,
        "output_shape": (rows, kv_heads, topk), "output_dtype": "int32",
    }
    return OpSpec(
        op_id=op_id, family="lightning_indexer", phase=phase, stage=stage,
        layer=layer, requests=tuple(requests), deps=tuple(deps), stream=stream,
        semantic_shape=shape, local_shape=shape, padded_shape=shape,
        launch_shape=shape, flops=qk_flops, matrix_flops=(("bf16", qk_flops),),
        hbm_bytes=read_bytes + output_bytes, mte2_bytes=read_bytes,
        mte3_bytes=output_bytes, vector_ops=vector_ops,
        recipe=RECIPE, dtype="bf16", layout="TND/PA_BBND",
        source_boundary="kernel", coverage="bf16_li_mechanism_program",
        metadata=(
            ("native_source", SOURCE), ("implementation", implementation),
            ("lightning_indexer_work", ledger),
            ("native_tensor_view", view),
        ),
        notes=("Native scan and TopK only; metadata generation/fills/packing outside",
               "Query-tile key reload and local TopK exchange are modeled plan phases",
               "HBM keeps the unique payload; reload/exchange are on-chip or workspace traffic"),
    )


def _split_loads(geometry):
    return "mte2_aic" in geometry.resource_units


def _panel_pairs(q_start, q_end, context, compression, key_start, key_end):
    """Causal pairs of query rows [q_start, q_end) against key rows in a panel."""
    total = 0
    for row in range(q_start, q_end):
        visible = (context + row + 1) // compression
        stop = min(key_end, visible)
        if stop > key_start:
            total += stop - key_start
    return total


def _panel_selection_ops(tile, panel, compression, topk, kv_heads):
    """Selection comparisons for one causal panel (Top1 is a max reduce)."""
    if topk != 1:
        return panel["pairs"] * kv_heads * math.log2(topk)
    total = 0
    for row in range(tile["q_start"], tile["q_end"]):
        visible = (tile["context"] + row + 1) // compression
        count = max(0, min(panel["key_end"], visible) - panel["key_start"])
        total += max(0, count - 1)
    return total * kv_heads


def _tile_sentinel(tile, compression, topk, kv_heads):
    q_rows = tile["q_end"] - tile["q_start"]
    return _sentinel_elements(
        ((q_rows, tile["context"] + tile["q_start"]),),
        compression, topk, kv_heads)


def _key_access_plan(spans, *, compression, key_row_bytes, q_row_bytes,
                     q_tile_rows, topk, kv_heads, aiv_units, ub_capacity,
                     l1_capacity, l2_capacity, owner, identity_prefix,
                     gm_capacity=None, exchange_grant=None,
                     score_row_limit=None):
    """Resolve one joint plan for Q/K, scores, lists, exchange and output.

    The plan owns Q tile residency, K panel residency and the causal panel
    order.  It produces logical, GM->L1, L1->L0 and HBM bytes; the executable
    program consumes the same panel boundaries.  L1 capacity is shared by the
    live Q tile and K panels; a full-K retain is selected only when both fit.

    The UB solve includes the exchange receive slot *before* the score panel
    is fixed.  When the receive slot fits inside the score slot it aliases that
    slot after the last panel update; otherwise a dedicated slot is reserved
    and the read is chunked.  GM exchange uses one per-tile grant reused across
    tiles: a tile's write waits for the previous tile's read, and the grant is
    released only after the last read (at ``index_release``).  A caller-held
    grant is bound once instead of allocating a second one.
    """
    line = CACHE_LINE_BYTES
    transfer = CACHE_TRANSFER_BYTES
    l1_capacity = max(0, int(l1_capacity or 0))
    l2_capacity = None if l2_capacity is None else max(0, int(l2_capacity))
    gm_capacity = None if gm_capacity is None else max(0, int(gm_capacity))
    q_tile_rows = max(1, int(q_tile_rows))
    if l1_capacity > 0:
        # Reserve at least half of L1 for K; a smaller Q tile is an
        # analytical sub-tile when the device cannot hold a full M=16 tile.
        q_tile_rows = min(q_tile_rows,
                          max(1, l1_capacity // max(1, 2 * q_row_bytes)))
    q_allocation = q_tile_rows * q_row_bytes if l1_capacity > 0 else 0
    k_capacity = max(0, l1_capacity - q_allocation)
    total_visible = sum(((c + q) // compression) * key_row_bytes
                        for q, c in spans)
    retained = l1_capacity > 0 and total_visible <= k_capacity
    k_allocation = (total_visible if retained
                    else min(total_visible, k_capacity))
    panel_rows = (max(1, k_allocation // key_row_bytes)
                  if k_allocation >= key_row_bytes else 0)
    # Per-AIV destination buffers: one 16-row score panel and one local-list
    # slot.  They are reused across panels/tiles; the score slot is also the
    # exchange staging slot after its panel is consumed.
    max_visible = max((((c + q) // compression) for q, c in spans), default=0)
    list_rows = max(1, min(topk, max_visible))
    list_bytes = q_tile_rows * list_rows * kv_heads * 8
    max_score_rows = 0
    if ub_capacity and list_bytes < ub_capacity:
        max_score_rows = max(1, (ub_capacity - list_bytes) // max(1, q_tile_rows * 4))
        if score_row_limit is not None:
            max_score_rows = min(max_score_rows, max(1, int(score_row_limit)))
        if panel_rows:
            panel_rows = min(panel_rows, max_score_rows)
        else:
            panel_rows = max_score_rows
    elif score_row_limit is not None:
        panel_rows = max(1, min(panel_rows or int(score_row_limit),
                                int(score_row_limit)))
    score_bytes = 0
    l2_retained = (l2_capacity is not None and l2_capacity > 0
                   and total_visible <= l2_capacity)
    l1_state = (CacheState(CacheGeometry(k_allocation, line, transfer, 1))
                if retained and k_allocation > 0 else None)
    l2_state = (CacheState(CacheGeometry(l2_capacity, line, transfer, 1))
                if l2_capacity is not None and l2_capacity > 0 else None)
    tiles = []
    key_gm = key_hbm = key_logical = q_bytes_total = pairs_total = 0
    panels_total = 0
    for request, (q_len, context) in enumerate(spans):
        identity = f"{identity_prefix}|request={request}"
        filled = 0
        for q_start in range(0, q_len, q_tile_rows):
            q_end = min(q_len, q_start + q_tile_rows)
            q_bytes = (q_end - q_start) * q_row_bytes
            visible = (context + q_end) // compression
            if not visible:
                ranges = ()
            elif panel_rows:
                ranges = tuple((start, min(visible, start + panel_rows))
                               for start in range(0, visible, panel_rows))
            else:
                ranges = ((0, visible),)
            panels = []
            for key_start, key_end in ranges:
                rows = key_end - key_start
                logical = rows * key_row_bytes
                pairs = _panel_pairs(q_start, q_end, context, compression,
                                     key_start, key_end)
                if retained:
                    # The retained K prefix is queried as a whole; miss lines
                    # are the newly visible rows for this tile.
                    l1_result = l1_state.access(identity, visible * key_row_bytes)
                    gm = l1_result.miss_lines * line
                else:
                    gm = logical
                if l2_state is not None and l2_retained:
                    # L2 fill is a prefix across panels/tiles; only rows not
                    # already filled cause HBM traffic.
                    first = max(key_start, filled)
                    new_rows = max(0, key_end - first)
                    hbm = new_rows * key_row_bytes
                    filled = max(filled, key_end)
                elif retained:
                    hbm = gm
                else:
                    hbm = logical
                panels.append({
                    "key_start": key_start, "key_end": key_end,
                    "rows": rows, "logical_bytes": logical,
                    "gm_bytes": gm, "hbm_bytes": hbm, "pairs": pairs,
                })
                score_bytes = max(score_bytes, pairs * 4)
                key_gm += gm
                key_hbm += hbm
                key_logical += logical
                pairs_total += pairs
                panels_total += 1
            q_bytes_total += q_bytes
            lists = (min(aiv_units, max(1, (visible + topk - 1) // topk))
                     if visible else 0)
            entries = min(visible, lists * topk) if lists else 0
            exchange_tile = ((q_end - q_start) * entries * 8 * kv_heads
                             if lists > 1 else 0)
            merge_ops = ((q_end - q_start) * entries
                         * max(1.0, math.log2(lists)) * kv_heads
                         if lists > 1 else 0.0)
            panel_merge_ops = ((q_end - q_start) * kv_heads
                               * max(0, len(panels) - 1)
                               if topk == 1 else 0)
            tiles.append({
                "request": request, "q_start": q_start, "q_end": q_end,
                "context": context, "q_bytes": q_bytes,
                "visible_rows": visible, "panels": panels,
                "lists": lists, "entries": entries,
                "exchange_bytes": exchange_tile, "merge_ops": merge_ops,
                "panel_merge_ops": panel_merge_ops,
                "output_bytes": ((q_end - q_start) * kv_heads * topk * 4),
            })
    exchange_total = sum(tile["exchange_bytes"] for tile in tiles)
    exchange_per_tile = max((tile["exchange_bytes"] for tile in tiles), default=0)
    receive_required = (topk * LOCAL_LIST_ELEMENT_BYTES if exchange_total else 0)
    receive_bytes = 0
    receive_alias = "none"
    receive_chunks = 0
    if receive_required:
        if receive_required <= score_bytes:
            # The score slot is dead after the last panel's local update, so
            # the exchange receive slot can reuse it without another buffer.
            receive_alias = "score_slot"
            receive_chunks = 1
        else:
            spare = (max(0, int(ub_capacity) - list_bytes - score_bytes)
                     if ub_capacity else receive_required)
            receive_bytes = min(receive_required, spare)
            if receive_bytes <= 0:
                # The first score panel used the whole residual UB.  Re-solve
                # with a panel small enough to reserve the real receive slot.
                if score_row_limit is None and ub_capacity:
                    spare_rows = ((int(ub_capacity) - list_bytes
                                   - receive_required)
                                  // max(1, q_tile_rows * 4))
                    if spare_rows >= 1:
                        return _key_access_plan(
                            spans, compression=compression,
                            key_row_bytes=key_row_bytes,
                            q_row_bytes=q_row_bytes, q_tile_rows=q_tile_rows,
                            topk=topk, kv_heads=kv_heads, aiv_units=aiv_units,
                            ub_capacity=ub_capacity, l1_capacity=l1_capacity,
                            l2_capacity=l2_capacity, owner=owner,
                            identity_prefix=identity_prefix,
                            gm_capacity=gm_capacity,
                            exchange_grant=exchange_grant,
                            score_row_limit=spare_rows)
                raise ModelingError(
                    "bf16_lightning_indexer_exchange_receive_has_no_ub")
            receive_alias = "dedicated"
            receive_chunks = _ceil_div(receive_required, receive_bytes)
    workspace = (list_bytes + score_bytes
                 + (receive_bytes if receive_alias == "dedicated" else 0))
    if ub_capacity and workspace > ub_capacity:
        raise ModelingError("bf16_lightning_indexer_ub_workspace_exceeds_capacity")
    grant_name = None
    grant_bytes = None
    if exchange_grant is not None:
        if isinstance(exchange_grant, dict):
            grant_name = str(exchange_grant.get("name")
                             or exchange_grant.get("grant") or "")
            grant_bytes = exchange_grant.get("bytes")
        else:
            grant_name = str(exchange_grant)
        if not grant_name:
            raise ModelingError("bf16_lightning_indexer_exchange_grant_invalid")
        if grant_bytes is not None:
            grant_bytes = int(grant_bytes)
            if grant_bytes < exchange_per_tile:
                raise ModelingError(
                    "bf16_lightning_indexer_exchange_grant_too_small")
    if gm_capacity is not None and gm_capacity < exchange_per_tile:
        raise ModelingError("bf16_lightning_indexer_exchange_gm_capacity")
    return {
        "owner": int(owner),
        "aiv_units": int(aiv_units),
        "line_bytes": line, "transfer_bytes": transfer,
        "q_tile_rows": q_tile_rows, "q_row_bytes": q_row_bytes,
        "q_allocation_bytes": q_allocation,
        "k_allocation_bytes": k_allocation,
        "l1_allocation_bytes": q_allocation + k_allocation,
        "l2_allocation_bytes": (min(total_visible, l2_capacity)
                                if l2_capacity is not None and l2_capacity > 0
                                else 0),
        "l1_capacity_bytes": l1_capacity,
        "l2_capacity_bytes": l2_capacity,
        "l1_policy": ("retain_prefix" if retained else
                      "stream_panels" if l1_capacity > 0 else "no_l1"),
        "l2_policy": ("retain_prefix" if l2_retained else
                      "stream_panels" if l2_capacity is not None and l2_capacity > 0
                      else "no_l2"),
        "retained": retained, "l2_retained": l2_retained,
        "panel_rows": panel_rows,
        "score_bytes": score_bytes,
        "list_bytes": list_bytes,
        "exchange_receive_bytes": int(receive_required),
        "exchange_receive_additional_bytes": int(receive_bytes),
        "exchange_receive_alias": receive_alias,
        "exchange_receive_chunks": int(receive_chunks),
        "workspace_bytes": workspace,
        "exchange_total_bytes": int(exchange_total),
        "exchange_per_tile_bytes": int(exchange_per_tile),
        "exchange_gm_bytes": int(exchange_per_tile if grant_name is None else 0),
        "exchange_gm_capacity_bytes": gm_capacity,
        "exchange_gm_fits": (gm_capacity is None
                             or gm_capacity >= exchange_per_tile),
        "exchange_grant": grant_name,
        "exchange_grant_bytes": grant_bytes,
        "exchange_grant_reused": grant_name is not None,
        "exchange_hbm_bytes": 2 * int(exchange_total),
        "full_scores_bytes": pairs_total * 4,
        "full_lists_bytes": sum(q for q, _ in spans) * topk * kv_heads * aiv_units * 8,
        "total_visible_bytes": total_visible,
        "key_logical_bytes": key_logical,
        "key_gm_bytes": key_gm,
        "key_hbm_bytes": key_hbm,
        "q_bytes": q_bytes_total,
        "pairs": pairs_total,
        "panels": panels_total,
        "tiles": tiles,
    }


def _resident_allocations(context, access):
    """Actual Q/K residency allocations, pinned to their physical owner."""
    allocations = []
    owner = int(access["owner"])
    capacity = max(1, int(getattr(context, "core_count", 1) or 1))
    if owner >= capacity:
        raise ModelingError("bf16_lightning_indexer_cache_owner_out_of_range")
    names = getattr(context, "pool_names", {})
    units = getattr(context, "pool_units", {})
    capacities = getattr(context, "pool_capacities", {})
    if "l1" in names:
        l1_units = unit_group(owner, capacity, units.get("l1", capacity))
        if access["q_allocation_bytes"]:
            allocations.append(context.allocate(
                "index_q_l1", "l1", int(access["q_allocation_bytes"]), l1_units))
        if access["k_allocation_bytes"]:
            allocations.append(context.allocate(
                "index_k_l1", "l1", int(access["k_allocation_bytes"]), l1_units))
    if access["l2_allocation_bytes"] and "l2" in names:
        size = int(access["l2_allocation_bytes"])
        if "l2" in capacities:
            size = min(size, int(capacities["l2"]))
        allocations.append(context.allocate("index_k_l2", "l2", size, (0,)))
    if access["exchange_gm_bytes"] and not access["exchange_grant_reused"]:
        gm_pool = ("gm" if "gm" in names else
                   next((name for name in names
                         if "workspace" in str(name)), None))
        if gm_pool is not None:
            allocations.append(context.allocate(
                "index_exchange_gm", gm_pool,
                int(access["exchange_gm_bytes"]), (0,)))
    if access["workspace_bytes"] and "ub" in names:
        aivs = int(access["aiv_units"])
        for sub in range(aivs):
            unit = (owner * aivs + sub,)
            if access["score_bytes"]:
                allocations.append(context.allocate(
                    f"index_score_ub_{sub}", "ub",
                    int(access["score_bytes"]), unit))
            if access["list_bytes"]:
                allocations.append(context.allocate(
                    f"index_lists_ub_{sub}", "ub",
                    int(access["list_bytes"]), unit))
            if access["exchange_receive_additional_bytes"]:
                allocations.append(context.allocate(
                    f"index_receive_ub_{sub}", "ub",
                    int(access["exchange_receive_additional_bytes"]), unit))
    return tuple(allocations)


def _placement(context, owner, roles, *, core_count, aiv_per_aic=1, aiv=False):
    method = getattr(context, "owned", None)
    if method is None:
        return {}
    if not aiv:
        return method(owner, core_count, roles)
    owners = core_count * aiv_per_aic
    merged = {}
    for sub in range(aiv_per_aic):
        part = method(owner * aiv_per_aic + sub, owners, roles)
        for name, units in part.items():
            merged.setdefault(name, [])
            merged[name].extend(units)
    return {name: tuple(units) for name, units in merged.items()}


def _build(op, geometry):
    metadata = dict(op.metadata)
    ledger = metadata.get("lightning_indexer_work")
    if (ledger is None or op.dtype != "bf16"
            or dict(op.matrix_flops).keys() != {"bf16"}
            or op.layout != "TND/PA_BBND" or "native_tensor_view" not in metadata):
        raise ModelingError("bf16_lightning_indexer_native_contract_required")
    spans = tuple(tuple(span) for span in ledger["sequence_partition"])
    compression = int(ledger["compression"])
    topk = int(ledger["topk"])
    heads, width, kv_heads = int(ledger["heads"]), int(ledger["head_dim"]), int(ledger["kv_heads"])
    owner = int(ledger.get("cache_owner", 0))
    core_count = max(1, int(geometry.compute_units or 1))
    if owner >= core_count:
        raise ModelingError("bf16_lightning_indexer_cache_owner_out_of_range")
    vector_units = int(geometry.resource_units.get("vector", 2 * core_count))
    aiv_per_aic = max(1, vector_units // core_count)
    key_row_bytes = int(kv_heads * width * dtype_bytes("bf16"))
    q_row_bytes = int(heads * width * dtype_bytes("bf16"))
    l1_capacity = int(geometry.buffers.get("l1", 0) or 0)
    declared_l2 = ledger.get("l2_capacity_bytes")
    l2_capacity = (min(int(declared_l2), int(geometry.buffers["l2"]))
                   if declared_l2 is not None and "l2" in geometry.buffers
                   else None)
    gm_capacity = (int(geometry.buffers["gm"])
                   if "gm" in geometry.buffers else None)
    access = _key_access_plan(
        spans, compression=compression, key_row_bytes=key_row_bytes,
        q_row_bytes=q_row_bytes, q_tile_rows=QUERY_TILE_ROWS,
        topk=topk, kv_heads=kv_heads, aiv_units=aiv_per_aic,
        ub_capacity=int(geometry.buffers.get("ub", 0) or 0),
        l1_capacity=l1_capacity, l2_capacity=l2_capacity, owner=owner,
        identity_prefix=f"{op.op_id}|layer={op.layer}|stage={op.stage}|phase={op.phase}",
        gm_capacity=gm_capacity, exchange_grant=ledger.get("exchange_grant"))
    exchange_bytes = sum(tile["exchange_bytes"] for tile in access["tiles"])
    merge_ops = sum(tile["merge_ops"] for tile in access["tiles"])
    lists = sum(tile["lists"] * (tile["q_end"] - tile["q_start"])
                for tile in access["tiles"]) * kv_heads
    comparisons = 0.0
    sentinel = 0.0
    for tile in access["tiles"]:
        for panel in tile["panels"]:
            comparisons += _panel_selection_ops(
                tile, panel, compression, topk, kv_heads)
        comparisons += tile["panel_merge_ops"]
        sentinel += _tile_sentinel(tile, compression, topk, kv_heads)
    weight_cast_ops = (int(ledger["query_rows"]) * heads
                       if ledger["weight_dtype"] != "fp32" else 0)
    weighted = float(ledger["weighted_head_reduce_ops"]) + weight_cast_ops
    vector = weighted + comparisons + sentinel + merge_ops
    query = int(access["q_bytes"])
    metadata_bytes = int(ledger["metadata_read_bytes"])
    weight = int(ledger["head_weight_bytes"])
    output = int(ledger["selection_output_bytes"])
    fixpipe = int(access["pairs"]) * heads * 4
    read = read_role(geometry.resource_units)
    split = read == "mte2_aic"
    if split:
        loads = {"mte2_aic": int(access["key_gm_bytes"]) + query + metadata_bytes,
                 "mte2_aiv": weight + exchange_bytes, "mte2": 0}
    else:
        loads = {"mte2": int(access["key_gm_bytes"]) + query + metadata_bytes
                 + weight + exchange_bytes}
    resources = {
        "hbm": int(access["key_hbm_bytes"]) + query + weight + metadata_bytes
        + output + 2 * exchange_bytes,
        "mte3": output + exchange_bytes, "vector": vector,
        "mte1": int(access["key_logical_bytes"]) + query,
        "fixpipe": fixpipe, **loads,
    }
    stage_defs = []
    allocations = bool(access["l1_allocation_bytes"] or access["l2_allocation_bytes"]
                       or access["workspace_bytes"]
                       or (access["exchange_gm_bytes"]
                           and not access["exchange_grant_reused"]))
    reserve = "index_reserve" if allocations else None
    if reserve:
        stage_defs.append((reserve, {}, (), "acquire"))
    previous = reserve
    previous_exchange_read = None
    metadata_left = metadata_bytes
    for tile_index, tile in enumerate(access["tiles"]):
        q_after = (previous,) if previous else ()
        weight_tile = tile["q_end"] - tile["q_start"]
        weight_tile_bytes = weight_tile * heads * (
            4 if ledger["weight_dtype"] == "fp32" else 2)
        metadata_tile = metadata_left if tile_index == 0 else 0
        metadata_left -= metadata_tile
        q_stage = f"query_load_{tile_index}"
        q_work = {
            "hbm": tile["q_bytes"] + weight_tile_bytes + metadata_tile,
            read: tile["q_bytes"] + metadata_tile,
        }
        if split:
            q_work["mte2_aiv"] = weight_tile_bytes
        else:
            q_work["mte2"] = q_work.pop(read) + weight_tile_bytes
        stage_defs.append((q_stage, q_work, q_after, None))
        previous = q_stage
        for panel_index, panel in enumerate(tile["panels"]):
            load_name = f"key_load_{tile_index}_{panel_index}"
            stage_defs.append((load_name, {
                "hbm": panel["hbm_bytes"], read: panel["gm_bytes"],
            }, (previous,), None))
            qk_name = f"qk_weighted_reduce_{tile_index}_{panel_index}"
            qk_work = {
                "vector": panel["pairs"] * (3 * heads - kv_heads)
                + (weight_cast_ops if tile_index == 0 and panel_index == 0 else 0),
                "mte1": panel["logical_bytes"]
                + (tile["q_bytes"] if panel_index == 0 else 0),
                "fixpipe": panel["pairs"] * heads * 4,
                "matrix_bf16": 2 * panel["pairs"] * heads * width,
            }
            stage_defs.append((qk_name, qk_work,
                               (load_name, q_stage), None))
            update_name = f"topk_update_{tile_index}_{panel_index}"
            stage_defs.append((update_name, {
                "vector": _panel_selection_ops(
                    tile, panel, compression, topk, kv_heads),
            }, (qk_name,), None))
            previous = update_name
        exchange_tile = tile["exchange_bytes"]
        output_tile = tile["output_bytes"]
        if exchange_tile:
            write_name = f"exchange_write_{tile_index}"
            write_after = (previous,) if previous else ()
            if previous_exchange_read is not None:
                # One per-tile GM slot is reused; the next write waits for the
                # prior tile's read to complete.
                write_after = (*write_after, previous_exchange_read)
            stage_defs.append((write_name, {
                "hbm": exchange_tile, "mte3": exchange_tile,
            }, write_after, None))
            read_name = f"exchange_read_{tile_index}"
            read_work = {"hbm": exchange_tile}
            if split:
                read_work["mte2_aiv"] = exchange_tile
            else:
                read_work["mte2"] = exchange_tile
            stage_defs.append((read_name, read_work, (write_name,), None))
            previous = read_name
            previous_exchange_read = read_name
        merge_name = f"topk_merge_{tile_index}"
        stage_defs.append((merge_name, {
            "vector": _tile_sentinel(tile, compression, topk, kv_heads)
            + tile["merge_ops"] + tile["panel_merge_ops"],
        }, (previous,), None))
        previous = merge_name
        output_name = f"output_store_{tile_index}"
        stage_defs.append((output_name, {
            "hbm": output_tile, "mte3": output_tile,
        }, (previous,), None))
        previous = output_name
    release = "index_release" if allocations else None
    if release:
        stage_defs.append((release, {}, (previous,), "free"))

    def program(context):
        names = getattr(context, "resource_names", {})
        if names and read not in names:
            raise ModelingError(f"bf16_lightning_indexer_requires_read_port:{read}")
        allocations = _resident_allocations(context, access)
        core = max(1, int(getattr(context, "core_count", 1) or 1))
        aic_roles = ["cube", "mte1", "fixpipe"]
        if read != "mte2_aiv":
            aic_roles.append(read)
        aiv_roles = ["vector", "mte3"]
        if split:
            aiv_roles.append("mte2_aiv")
        aic_place = _placement(context, owner, tuple(aic_roles), core_count=core)
        aiv_place = _placement(context, owner, tuple(aiv_roles),
                               core_count=core, aiv_per_aic=aiv_per_aic, aiv=True)
        generic_mte2_place = ({}
                              if split else
                              _placement(context, owner, ("mte2",),
                                         core_count=core))
        stages = []
        for name, amounts, after, action in stage_defs:
            values = dict(amounts)
            matrix = {"bf16": values.pop("matrix_bf16", 0)}
            work = Work({key: value for key, value in matrix.items() if value},
                        {key: value for key, value in values.items() if value})
            kwargs = {}
            if action == "acquire":
                kwargs["acquire"] = allocations
            elif action == "free":
                kwargs["free"] = allocations
            if name.startswith("key_load"):
                kwargs["placement"] = aic_place
            elif name.startswith("query_load"):
                kwargs["placement"] = ({**aic_place, **aiv_place}
                                       if split else aic_place)
            elif name.startswith("qk_weighted_reduce"):
                kwargs["placement"] = {**aic_place, **aiv_place}
            elif name.startswith("exchange_write"):
                kwargs["placement"] = aiv_place
            elif name.startswith("exchange_read"):
                kwargs["placement"] = (aiv_place if split
                                       else generic_mte2_place)
            elif name.startswith("topk_update"):
                kwargs["placement"] = {**aiv_place, **generic_mte2_place}
            elif name.startswith(("topk_merge", "output_store")):
                kwargs["placement"] = aiv_place
            stages.append(Stage(name, context.bind(work), after=after, **kwargs))
        return KernelProgram(tuple(stages))

    work = Work(dict(op.matrix_flops), resources)
    exchange_grant_name = (
        "none" if not exchange_bytes else
        access["exchange_grant"] or
        ("index_exchange_gm"
         if access["exchange_gm_bytes"] and "gm" in geometry.buffers
         else "external_gm_grant"))
    return OperatorPlan(work, program, {
        "execution_abstraction": "mechanism_phase_program",
        "native_dispatch_reconstructed": False,
        "internal_cache_reload_and_topk_exchange": "modeled",
        "mechanism": {
            "query_tile_rows": int(access["q_tile_rows"]),
            "cache_owner": owner,
            "read_port": read,
            "split_mte2": split,
            "aiv_per_aic": aiv_per_aic,
            "aiv_owner_units": tuple(
                owner * aiv_per_aic + sub for sub in range(aiv_per_aic)),
            "cache_levels": {
                "l1": {"capacity_bytes": l1_capacity,
                       "q_allocation_bytes": int(access["q_allocation_bytes"]),
                       "k_allocation_bytes": int(access["k_allocation_bytes"]),
                       "allocation_bytes": int(access["l1_allocation_bytes"]),
                       "policy": access["l1_policy"], "owner": owner},
                "l2": {"capacity_bytes": access["l2_capacity_bytes"],
                       "declared_capacity_bytes": declared_l2,
                       "allocation_bytes": int(access["l2_allocation_bytes"]),
                       "policy": access["l2_policy"],
                       "scope": ("engine_pool_shared" if l2_capacity is not None
                                 else "declared_policy_no_engine_pool")},
                "gm": {"capacity_bytes": access["exchange_gm_capacity_bytes"],
                       "slot_bytes": int(access["exchange_per_tile_bytes"]),
                       "allocation_bytes": int(access["exchange_gm_bytes"]),
                       "grant": (access["exchange_grant"] or
                                 ("index_exchange_gm"
                                  if access["exchange_gm_bytes"]
                                  else "external_gm_grant")),
                       "policy": "one_per_tile_slot_reused",
                       "scope": ("caller_grant" if access["exchange_grant_reused"]
                                 else "engine_pool_shared"
                                 if access["exchange_gm_bytes"]
                                 and "gm" in geometry.buffers
                                 else "declared_memory_grant")},
            },
            "key_access": {
                "logical_bytes": access["key_logical_bytes"],
                "gm_bytes": access["key_gm_bytes"],
                "hbm_bytes": access["key_hbm_bytes"],
                "q_bytes": query,
                "pairs": access["pairs"],
                "panels": access["panels"],
                "panel_rows": access["panel_rows"],
                "tiles": access["tiles"],
                "capacity_events": ("engine_acquire_free" if allocations
                                    else "none_no_resident_allocation"),
            },
            "key_transport_bytes": int(access["key_logical_bytes"]),
            "key_gm_bytes": int(access["key_gm_bytes"]),
            "key_hbm_bytes": int(access["key_hbm_bytes"]),
            "unique_key_bytes": int(ledger["unique_key_bytes"]),
            "key_hbm_reload_bytes": max(0, int(access["key_hbm_bytes"])
                                        - int(ledger["unique_key_bytes"])),
            "cache_capacity_bytes": l1_capacity,
            "l2_capacity_bytes": access["l2_capacity_bytes"],
            "l1_allocation_bytes": int(access["l1_allocation_bytes"]),
            "q_allocation_bytes": int(access["q_allocation_bytes"]),
            "k_allocation_bytes": int(access["k_allocation_bytes"]),
            "l2_allocation_bytes": int(access["l2_allocation_bytes"]),
            "local_topk_lists": lists,
            "topk_owner_units": tuple(
                owner * aiv_per_aic + sub for sub in range(aiv_per_aic)),
            "local_topk_exchange_kv_heads": kv_heads,
            "local_list_merge_ops": merge_ops,
            "exchange_bytes": exchange_bytes,
            "exchange_hbm_bytes": int(access["exchange_hbm_bytes"]),
            "exchange_storage": {
                "total_bytes": exchange_bytes,
                "per_tile_bytes": int(access["exchange_per_tile_bytes"]),
                "per_aiv_list_bytes": int(access["list_bytes"]),
                "per_aiv_write_bytes": (int(access["exchange_per_tile_bytes"])
                                        // max(1, aiv_per_aic)),
                "receive_bytes": int(access["exchange_receive_bytes"]),
                "receive_additional_bytes": int(
                    access["exchange_receive_additional_bytes"]),
                "receive_alias": access["exchange_receive_alias"],
                "receive_chunks": int(access["exchange_receive_chunks"]),
                "allocations": tuple(
                    f"index_lists_ub_{sub}" for sub in range(aiv_per_aic)),
                "gm_allocation": exchange_grant_name,
                "write_stages": ("exchange_write_<tile>",),
                "read_stage": "exchange_read_<tile>",
                "merge_stage": "topk_merge_<tile>",
                "output_stage": "output_store_<tile>",
                "release_stage": ("index_release" if allocations else "none"),
                "grant": exchange_grant_name,
                "grant_reused": bool(access["exchange_grant_reused"]),
                "capacity_feedback": {
                    "pool": "gm",
                    "capacity_bytes": access["exchange_gm_capacity_bytes"],
                    "slot_bytes": int(access["exchange_per_tile_bytes"]),
                    "fits": bool(access["exchange_gm_fits"]),
                    "allocation_bytes": int(access["exchange_gm_bytes"]),
                    "waits_for_grant": not bool(access["exchange_gm_fits"]),
                    "pool_declared": "gm" in geometry.buffers,
                    "requires_memory_grant": bool(
                        access["exchange_gm_bytes"]
                        and not access["exchange_grant_reused"]),
                },
                "hbm_round_trip_per_tile": 2 * int(
                    access["exchange_per_tile_bytes"]),
                "hbm_round_trip_total": 2 * int(access["exchange_total_bytes"]),
                "reuse": ("one per-tile GM slot; each write waits for the "
                          "previous tile's read; grant released after the last "
                          "read/output at index_release"),
            },
            "workspace": {
                "score_bytes_per_aiv": int(access["score_bytes"]),
                "list_bytes_per_aiv": int(access["list_bytes"]),
                "exchange_receive_bytes_per_aiv": int(
                    access["exchange_receive_bytes"]),
                "exchange_receive_additional_bytes_per_aiv": int(
                    access["exchange_receive_additional_bytes"]),
                "exchange_receive_alias": access["exchange_receive_alias"],
                "exchange_receive_chunks": int(
                    access["exchange_receive_chunks"]),
                "resident_bytes_per_aiv": int(access["workspace_bytes"]),
                "full_scores_if_materialized": int(access["full_scores_bytes"]),
                "full_lists_if_materialized": int(access["full_lists_bytes"]),
                "pool": ("ub" if "ub" in geometry.buffers else "unbound"),
                "reuse": ("score slot reused per panel and as the exchange "
                          "receive when it fits; list slot per 16-row tile"),
            },
            "vector_owner_partitions": aiv_per_aic,
            "source_basis": ("native MIX_AIC M tile plus paired AIV subblocks; "
                             "Q/K panel residency and ordered K panels are "
                             "explicit mechanisms"),
        },
    })
