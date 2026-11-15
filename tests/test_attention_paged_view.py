"""Independently declared paged storage preserves geometry without pricing capacity."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case


ROOT = Path(__file__).resolve().parents[1]


def case(name, **policy):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    raw["runtime"]["attention_policy"] = policy
    return parse_case(raw)


@pytest.mark.parametrize("name,q,heads,kv_heads", [
    ("qwen35-a3-tp4", 2, 4, 1),
    ("qwen36-a3-tp4", 1, 6, 1),
    ("qwen36-a3-tp4", 4, 6, 1),
    ("qwen38-a5", 1, 24, 4),
    ("qwen38-a5", 4, 24, 4),
    ("qwen35-a3-tp4", 8192, 4, 1),
])
def test_paged_views_keep_local_kv_geometry_and_logical_op_work(name, q, heads, kv_heads):
    phase = "prefill" if q == 8192 else "decode"
    kwargs = dict(phase=phase, tokens=q, context=16384, sequences=2)
    original = target_operators(case(name), **kwargs)
    selected = target_operators(case(name, kv_storage="paged"), **kwargs)
    # The frontend's logical operator fields (pairs, Q/O payload, MTE spans)
    # stay storage-independent.  Paged block-table/scale traffic is a
    # recipe-level physical ledger, checked in the next test.
    assert tuple(replace(op, metadata=()) for op in selected) == tuple(
        replace(op, metadata=()) for op in original)
    attention = [op for op in selected if op.family == "attention"]
    assert attention
    for op in attention:
        view = dict(op.metadata)["attention_tensor_view"]
        assert view["query_shape"] == view["output_shape"] == (2 * q, heads, 256)
        assert view["key_tail_shape"] == view["value_tail_shape"] == (128, kv_heads * 256)
        assert view["batch_count"] == 2
        assert view["kv_storage"] == "paged"


def test_page_tail_and_actual_batch_follow_declared_storage_and_fusion():
    base = case("qwen38-a5", kv_storage="paged")
    separate = replace(base, runtime=replace(
        base.runtime, kv_block_size=64,
        attention_policy={"kv_storage": "paged", "fused_batch": False}))
    ops = target_operators(separate, phase="decode", tokens=2, context=2048, sequences=3)
    attention = [op for op in ops if op.family == "attention"]
    for op in attention:
        view = dict(op.metadata)["attention_tensor_view"]
        assert view["query_shape"] == (2, 24, 256)
        assert view["key_tail_shape"] == (64, 1024)
        assert view["batch_count"] == 1
    assert len(attention) == 3 * sum(
        layer.attention == "gqa" for layer in base.model.layer_plans)


def test_unmodeled_context_partition_does_not_claim_an_unsharded_paged_abi():
    base = case("qwen38-a5", kv_storage="paged")
    dies = base.machine.dies if len(base.machine.dies) >= 2 else (
        base.machine.dies[0], replace(base.machine.dies[0], rank=1))
    sharded = replace(
        base,
        machine=replace(base.machine, dies=dies),
        runtime=replace(base.runtime, parallel=replace(base.runtime.parallel, tp=2, cp=2)),
    )
    ops = target_operators(sharded, phase="decode", tokens=1, context=2048)
    attention = [op for op in ops if op.family == "attention"]
    assert attention
    assert all("attention_tensor_view" not in dict(op.metadata) for op in attention)


def test_paged_frontend_view_reaches_recipe_physical_access_not_logical_pairs():
    from mindie_sim.prediction.attention_decode import StagedGQADecode

    paged = target_operators(
        case("qwen38-a5", kv_storage="paged", fused_batch=False),
        phase="decode", tokens=1, context=1028)
    contiguous = target_operators(
        case("qwen38-a5", fused_batch=False),
        phase="decode", tokens=1, context=1028)
    recipe = StagedGQADecode()
    for old, new in zip(
            [op for op in contiguous if op.family == "attention"],
            [op for op in paged if op.family == "attention"]):
        before, after = recipe.work(old), recipe.work(new)
        access = recipe.kv_access(new)
        assert access.paged and access.page_size == 128
        assert access.page_meta_bytes == 4
        # The logical causal pairs and matrix product are storage-independent.
        assert before.matrix == after.matrix
        assert before.resources["vector"] == after.resources["vector"]
        # Horizon 1029 touches nine pages per KV head; only the paged ledger
        # gains the block-table entries.
        assert after.resources["hbm"] == before.resources["hbm"] + 4 * 9 * 4


def test_paged_view_carries_declared_cache_representation_to_the_recipe():
    from dataclasses import replace
    from mindie_sim.prediction.attention_decode import StagedGQADecode

    parsed = case("qwen38-a5", kv_storage="paged", fused_batch=False)
    metadata = dict(parsed.model.metadata)
    metadata["cache_representations"] = {"gqa_kv": {
        "dtype": "int8", "group_size": 64, "scale_dtype": "fp32",
        "suffix_elements": 0, "suffix_dtype": "bf16", "packed_elements": 1}}
    # The model quantizer still declares BF16 KV for this source control; the
    # storage representation is a separate physical declaration.
    parsed = replace(parsed, model=replace(
        parsed.model, metadata=tuple(metadata.items())))
    ops = target_operators(parsed, phase="decode", tokens=1, context=1028)
    attention = [op for op in ops if op.family == "attention"]
    view = dict(attention[0].metadata)["attention_tensor_view"]
    assert view["cache_representations"]["gqa_kv"]["group_size"] == 64
    access = StagedGQADecode().kv_access(attention[0])
    assert access.key_format.dtype == "int8"
    assert access.key_payload_row == 256  # 256 BF16 head width -> INT8 256 B
    assert access.key_scale_row == 16      # 4 groups x FP32 per K row


def test_declared_cache_representation_controls_store_dtype_and_row_bytes():
    from dataclasses import replace

    parsed = case("qwen38-a5", kv_storage="paged", fused_batch=False)
    metadata = dict(parsed.model.metadata)
    metadata["cache_representations"] = {"gqa_kv": {
        "dtype": "int8", "group_size": 64, "scale_dtype": "fp32",
        "suffix_elements": 0, "suffix_dtype": "bf16", "packed_elements": 1}}
    parsed = replace(parsed, model=replace(
        parsed.model, metadata=tuple(metadata.items())))
    ops = target_operators(parsed, phase="decode", tokens=1, context=1028)
    store = next(op for op in ops if op.family == "cache_store")
    # T07 keeps the BF16 source identity; output is 2048 B INT8 + 128 B scale.
    assert store.dtype == "bf16"
    assert store.local_shape == (1, 2048)
    assert store.mte2_bytes == pytest.approx(4096)
    assert store.mte3_bytes == pytest.approx(2176)
    # Cold unresolved source read + packed output write.
    assert store.hbm_bytes == pytest.approx(6272)
    assert store.vector_ops == pytest.approx(4 * 2048 + 32)
    representation = dict(store.metadata)["cache_representation"]
    assert representation["row_bytes"] == 2176
    assert dict(store.metadata)["cache_traffic_basis"] == (
        "legacy_logical_source_plus_stored_bytes")


def test_independent_kv_cache_formats_emit_two_typed_stores():
    from dataclasses import replace
    from mindie_sim.prediction.attention_decode import StagedGQADecode

    parsed = case("qwen38-a5", kv_storage="paged", fused_batch=False)
    metadata = dict(parsed.model.metadata)
    metadata["cache_representations"] = {
        "gqa_kv_key": {"dtype": "bf16", "group_size": 0,
                       "scale_dtype": "none", "suffix_elements": 0,
                       "suffix_dtype": "bf16", "packed_elements": 1},
        "gqa_kv_value": {"dtype": "int8", "group_size": 64,
                         "scale_dtype": "fp32", "suffix_elements": 0,
                         "suffix_dtype": "bf16", "packed_elements": 1}}
    parsed = replace(parsed, model=replace(
        parsed.model, metadata=tuple(metadata.items())))
    ops = target_operators(parsed, phase="decode", tokens=1, context=1028)
    stores = [op for op in ops if op.family == "cache_store"]
    first_layer = stores[0].layer
    stores = [op for op in stores if op.layer == first_layer]
    assert len(stores) == 2
    by_region = {dict(op.metadata).get("cache_representation", dict(
                     op.metadata).get("cache_format"))["region"]: op
                 for op in stores}
    assert by_region["gqa_kv_key"].mte3_bytes == pytest.approx(2048)
    # INT8 1024 B payload + 16*4 scale.
    assert by_region["gqa_kv_value"].mte3_bytes == pytest.approx(1088)
    attention = [op for op in ops if op.family == "attention"][0]
    view = dict(attention.metadata)["attention_tensor_view"]
    assert (view["key_dtype"], view["value_dtype"]) == ("bf16", "int8")
    access = StagedGQADecode().kv_access(attention)
    assert not access.key_needs_conversion and access.value_needs_conversion
    assert access.key_conversion_ops(128) == 0
    assert access.value_conversion_ops(128) == 128 * 256 * 2


def test_shared_suffix_store_row_matches_reader_332_bytes():
    from dataclasses import replace
    from mindie_sim.prediction.attention import _CacheProducer
    from mindie_sim.prediction.cache_representation import store

    parsed = case("qwen38-a5", kv_storage="paged", fused_batch=False)
    metadata = dict(parsed.model.metadata)
    metadata["cache_representations"] = {"gqa_kv": {
        "dtype": "int8", "group_size": 64, "scale_dtype": "fp32",
        "suffix_elements": 64, "suffix_dtype": "bf16",
        "packed_elements": 1}}
    model = replace(parsed.model, metadata=tuple(metadata.items()))
    ops = []
    producer = _CacheProducer(
        ops, prefix="p", stage=0, layer=0, requests=("r",), phase="decode",
        model=model)
    store(producer, "cache", (), 1, 256, "gqa_kv")
    row = ops[-1]
    representation = dict(row.metadata)["cache_representation"]
    assert representation["row_bytes"] == 332
    assert row.mte2_bytes == pytest.approx(512)
    assert row.mte3_bytes == pytest.approx(332)
    assert row.hbm_bytes == pytest.approx(844)


def test_paged_frontend_publishes_ordinary_fia_native_plan():
    ops = target_operators(
        case("qwen35-a3-tp4", kv_storage="paged"),
        phase="decode", tokens=2, context=16384, sequences=2)
    attention = next(op for op in ops if op.family == "attention")
    view = dict(attention.metadata)["attention_tensor_view"]
    plan = view["native_paged_fia"]
    assert plan["kind"] == "paged_fia"
    assert plan["geometry"]["query_heads"] == 4
    assert plan["geometry"]["kv_heads"] == 1
    assert plan["geometry"]["group"] == 4
    assert plan["geometry"]["hq_equals_hkv"] is False
    assert plan["geometry"]["page_size"] == 128
    assert plan["ports"]["query"]["shape"] == (4, 4, 256)
    assert plan["ports"]["key_cache"]["shape"][1] == 128
    assert plan["ports"]["value_cache"]["layout"] == "PA_BBND"
    assert plan["ports"]["output"]["shape"] == (4, 4, 256)
    assert plan["ports"]["lse"] is None
    assert not plan["state"]["writes"]
    assert plan["state"]["lse_materialized"] is False
    assert plan["abi"]["entry"] == "torch_npu.npu_fused_infer_attention_score"
    assert plan["limits"]["d256_specialized_ifa"] is False
    from mindie_sim.prediction.native_binding import bind_native_call
    binding = bind_native_call(attention)
    carried = binding.typed["attrs"]["native_paged_fia"]
    assert carried["kind"] == "paged_fia"
    assert carried["work"]["head_token_pairs"] == 262168
    assert carried["ports"]["lse"] is None
    assert carried["state"]["writes"] == []
