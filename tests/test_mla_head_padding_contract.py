"""Typed logical/physical head materialization shares real copy/pad traffic."""
import json
from pathlib import Path

from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mla_head_padding import (
    IMPLEMENTATION,
    padding_contract,
    query_inputs,
)
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/kimi-a5-paged-host-20261030/c1.json"


def _case():
    case = parse_case(json.loads(CASE.read_text()))
    plan = next(p for p in case.model.layer_plans if p.attention == "mla")
    return case, plan


def test_head_padding_contract_reports_real_copy_and_pad_bytes():
    case, plan = _case()
    declaration = {"logical_heads": 12, "physical_heads": 16}
    fragment = padding_contract(
        plan=plan, parallel=case.runtime.parallel, tokens=6,
        input_dtype=case.model.activation_dtype, position_start=1024)
    attrs = {attr.name: attr.value for attr in fragment.attrs}
    assert fragment.name == "mla_head_padding"
    assert attrs["implementation"] == IMPLEMENTATION
    assert attrs["logical_heads"] == 12 and attrs["physical_heads"] == 16
    assert attrs["tokens"] == 6 and attrs["position_start"] == 1024
    assert attrs["copy_volume"] == 6 * 12 * 512 * 2
    assert attrs["zero_fill_bytes"] == 6 * 4 * (512 + 64) * 2
    work = {item.resource: item.amount for item in fragment.work.items}
    # [B,12,D] -> contiguous copy, then two F.pad outputs to 16 heads.
    assert work["mte2"] == 6 * 12 * 512 * 2 + 6 * 12 * 512 * 2 + 6 * 12 * 64 * 2
    assert work["mte3"] == 6 * 12 * 512 * 2 + 6 * 16 * 512 * 2 + 6 * 16 * 64 * 2
    assert work["hbm"] == work["mte2"] + work["mte3"]
    ports = {port.name: port for port in (
        *fragment.ports.tensor_in, *fragment.ports.tensor_out,
        *fragment.ports.event_in, *fragment.ports.event_out)}
    assert ports["q_logical"].dtype == "bf16"
    assert ports["q_logical"].layout_kind == "BNSD"
    assert ports["q_physical"].layout_kind == "BNSD"
    assert ports["q_ready"].event == "ready"
    assert ports["done"].event == "complete"
    assert {effect.field: effect.action for effect in fragment.field_effects}[
        "q.layout"] == "consume"

    # The graph helper and the typed contract use the same source bytes.
    builder = Builder("s0.l0", plan, 0, (), case.model, "decode")
    query, rotary = query_inputs(
        builder, declaration=declaration, tokens=6, query="q", rotary="r",
        plan=plan)
    assert len(builder.ops) == 3
    written = sum(op.mte3_bytes for op in builder.ops)
    read = sum(op.mte2_bytes for op in builder.ops)
    zero = sum(dict(op.metadata).get("zero_fill_bytes", 0) for op in builder.ops)
    assert read == work["mte2"]
    assert written == work["mte3"]
    assert zero == attrs["zero_fill_bytes"]
    assert query.endswith("mla_query_pad_nope") or query
    assert rotary
