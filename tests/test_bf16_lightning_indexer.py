"""BF16 LI causal work, source selection and inclusive response behavior.

[Role] Check mathematical work and common Engine execution independently.
[Invariants] BF16/FP32 caller contracts never bind the INT8 QLI implementation.
[Timeouts & Locks] Small CPU graphs; no external locks.
[Side-Effect Policy] Read one case fixture; no device or remote execution.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.family_lightning_indexer import (
    IMPLEMENTATION, RECIPE, causal_pairs, make_op,
)
from mindie_sim.prediction.implementations import recipes, select
from mindie_sim.schema.common import ModelingError


def case():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/high-concurrency-20261030/new17/v4flash-a3-tp8-c4.json").read_text())
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
        "lightning_indexer"] = {"implementation": IMPLEMENTATION}
    return raw


@pytest.mark.parametrize("rows,context", [
    (1, 0), (3, 0), (4, 0), (257, 1), (512, 4096), (8192, 122880),
])
def test_compressed_causal_pairs_match_scalar_integer_oracle(rows, context):
    expected = sum((context + i + 1) // 4 for i in range(rows))
    assert causal_pairs(rows, context) == expected
    op = make_op(query_rows=rows, context=context)
    assert dict(op.matrix_flops) == {"bf16": 2 * expected * 4 * 128}
    view = dict(op.metadata)["native_tensor_view"]
    assert view["query_dtype"] == view["cache_dtype"] == "bf16"
    assert view["weight_dtype"] == "fp32"
    assert view["output_shape"] == (rows, 1, 512)
    assert op.source_boundary == "kernel"


def test_pcp_spans_preserve_causal_work_and_boundary_multiplicity():
    q, c = 8192, 122880
    full = causal_pairs(q, c)
    head_tail = causal_pairs(q // 4, c) + causal_pairs(q // 4, c + 3 * q // 4)
    middle = causal_pairs(q // 2, c + q // 4)
    assert full == head_tail + middle == 2 * middle
    a = make_op(query_rows=q // 4, context=c)
    b = make_op(query_rows=q // 4, context=c + 3 * q // 4)
    m = make_op(query_rows=q // 2, context=c + q // 4)
    assert a.flops + b.flops == m.flops
    # Two calls reread their metadata and overlapping K; no global half-cost
    # substitution erases those boundaries or input payloads.
    assert a.mte2_bytes + b.mte2_bytes > m.mte2_bytes


@pytest.mark.parametrize("changes", [
    {"query_dtype": "int8"}, {"cache_dtype": "int8"},
    {"cache_dtype": "fp8_e4m3"}, {"weight_dtype": "int4"},
])
def test_wrong_representation_is_rejected(changes):
    with pytest.raises(ModelingError, match="bf16_lightning_indexer_v2_domain_mismatch"):
        make_op(query_rows=512, context=4096, **changes)


@pytest.mark.parametrize("changes", [
    {"compression": 2}, {"compression": 8}, {"topk": 2048}, {"topk": 128},
    {"topk": 1},
    {"head_dim": 64}, {"head_dim": 256}, {"page_size": 64},
    {"heads": 8, "kv_heads": 2}, {"weight_dtype": "bf16"},
])
def test_unsampled_geometry_uses_the_same_mechanism(changes):
    rows, context = 37, 4099
    op = make_op(query_rows=rows, context=context, **changes)
    ledger = dict(op.metadata)["lightning_indexer_work"]
    compression = changes.get("compression", 4)
    topk = changes.get("topk", 512)
    heads, kv_heads = changes.get("heads", 4), changes.get("kv_heads", 1)
    expected_pairs = sum((context + i + 1) // compression for i in range(rows))
    assert ledger["query_key_pairs"] == expected_pairs
    assert op.flops == 2 * expected_pairs * heads * changes.get("head_dim", 128)
    assert ledger["weighted_head_reduce_ops"] == expected_pairs * (3 * heads - kv_heads)
    if topk == 1:
        visible_rows = sum(1 for i in range(rows)
                           if (context + i + 1) // compression > 0)
        expected_comparisons = (expected_pairs - visible_rows) * kv_heads
    else:
        expected_comparisons = expected_pairs * kv_heads * math.log2(topk)
    assert ledger["topk_comparison_ops"] == pytest.approx(expected_comparisons)
    assert ledger["selection_output_bytes"] == rows * kv_heads * topk * 4
    assert op.hbm_bytes == ledger["read_bytes_lower_bound"] + ledger["selection_output_bytes"]
    assert op.local_shape == (rows, heads, changes.get("head_dim", 128))
    assert dict(op.metadata)["native_tensor_view"]["output_shape"] == (
        rows, kv_heads, topk)


def test_registry_uses_distinct_bf16_recipe_and_requires_source_binding():
    raw = case()
    bindings = raw["runtime"]["kernels"]["lowerings"]
    selected = select(bindings, "lightning_indexer")
    assert selected.name == IMPLEMENTATION
    assert RECIPE in recipes(bindings)
    assert "quant_lightning_indexer_v2" not in recipes(bindings)
    op = make_op(query_rows=512, context=4096)
    unbound = deepcopy(raw)
    del unbound["runtime"]["kernels"]["lowerings"]["lightning_indexer"]
    with pytest.raises(ModelingError, match="source_plan_binding_required"):
        graph_costs(unbound, (op,))


def test_late_inputs_remain_causal_and_programs_are_released():
    op = make_op(query_rows=512, context=4096, deps=("query", "cache", "metadata"))
    result = graph_costs(
        case(), (op,), input_ready_ms=7,
        external_inputs={key: 7 for key in op.deps}, isolated=False)
    assert result["complete_ms"] > 7
    assert all(row["ready_ms"] >= 7 for row in result["timeline"])
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    ledger = dict(op.metadata)["lightning_indexer_work"]
    assert op.hbm_bytes == ledger["read_bytes_lower_bound"] + ledger["selection_output_bytes"]
    assert op.mte3_bytes == 512 * 512 * 4


def test_observed_response_is_inclusive_and_physical_work_can_dominate():
    raw = case()
    op = make_op(query_rows=512, context=4096)
    raw.setdefault("runtime", {}).setdefault("metadata", {})["legacy_diagnostics"] = True
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "independent_test_response", "match": {"recipe": RECIPE},
        "response": {"fixed_us": 10000}, "provenance": {"basis": "test only"},
    }]
    result = graph_costs(raw, (op,), input_ready_ms=7, isolated=False)
    assert result["duration_ms"] == pytest.approx(10)
    assert result["complete_ms"] == pytest.approx(17)
    slow = deepcopy(raw)
    slow.setdefault("machine", {}).setdefault("device", {}).setdefault("rates", {})["hbm"] = 1e6
    slowed = graph_costs(slow, (op,), isolated=False)
    assert slowed["duration_ms"] > 10


def test_ordinary_prediction_does_not_inject_a_service_response_floor():
    raw = case()
    op = make_op(query_rows=512, context=4096)
    result = graph_costs(raw, (op,), input_ready_ms=7, isolated=False)
    assert result["duration_ms"] < 10
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    declared = deepcopy(raw)
    declared["runtime"]["kernels"]["service_regimes"] = [{
        "id": "independent_test_response", "match": {"recipe": RECIPE},
        "response": {"fixed_us": 10000}, "provenance": {"basis": "test only"},
    }]
    honored = graph_costs(declared, (op,), input_ready_ms=7, isolated=False)
    assert honored["duration_ms"] == pytest.approx(10)


def test_mla_frontend_reuses_selected_bf16_li_builder_with_fp32_weights():
    from mindie_sim.prediction.operator_analysis import target_operators
    from mindie_sim.prediction.structure import parse_case

    ops = target_operators(parse_case(case()), phase="decode", tokens=4,
                           context=8192)
    selected = [op for op in ops if op.recipe == RECIPE]
    assert len(selected) == 21
    assert all(op.family == "lightning_indexer" for op in selected)
    view = dict(selected[0].metadata)["native_tensor_view"]
    assert view["query_dtype"] == view["cache_dtype"] == "bf16"
    assert view["weight_dtype"] == "fp32"
    assert view["output_dtype"] == "int32"
    assert view["output_shape"] == (4, 1, 512)
    ledger = dict(selected[0].metadata)["lightning_indexer_work"]
    assert ledger["qk_bf16_flops"] == selected[0].flops
    assert ledger["selection_output_bytes"] == selected[0].mte3_bytes
    # The semantic index_topk placeholder is replaced by the selected builder.
    assert not [op for op in ops if op.recipe == "index_topk"]
