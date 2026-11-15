"""Packed MLA prolog typed contract keeps int8/bf16/fp32 components and work."""
import json
from pathlib import Path

from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mla_prolog_packed import (
    LAYOUT,
    packed_contract,
    store,
    view,
)
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/kimi-a5-paged-host-20261030/c1.json"


def test_packed_prolog_contract_and_store_share_component_and_work():
    case = parse_case(json.loads(CASE.read_text()))
    plan = next(p for p in case.model.layer_plans if p.attention == "mla")
    rows = 6
    declaration = view(plan.kv_lora_rank, plan.qk_rope_head_dim)
    assert declaration["row_bytes"] == 656
    assert declaration["component_offsets"] == {
        "latent": 0, "rope": 512, "scales": 640}
    fragment = packed_contract(plan=plan, rows=rows)
    attrs = {attr.name: attr.value for attr in fragment.attrs}
    assert attrs["layout"] == LAYOUT
    assert attrs["latent_dtype"] == "int8"
    assert attrs["rope_dtype"] == "bf16"
    assert attrs["scale_dtype"] == "fp32"
    assert attrs["row_bytes"] == 656
    assert dict(attrs["component_offsets"]) == declaration["component_offsets"]
    assert attrs["rows"] == rows
    work = {item.resource: item.amount for item in fragment.work.items}
    builder = Builder("s0.l0", plan, 0, (), case.model, "decode")
    store(builder, "packed_kv", ("kv_norm",), rows, plan)
    op = builder.ops[-1]
    assert work["hbm"] == op.hbm_bytes
    assert work["mte2"] == op.mte2_bytes
    assert work["mte3"] == op.mte3_bytes
    assert work["vector"] == op.vector_ops
    ports = {port.name: port for port in (
        *fragment.ports.tensor_out, *fragment.ports.state_out)}
    assert ports["packed_cache"].dtype == "mixed_int8_bf16"
    assert ports["packed_scale"].dtype == "fp32"
    assert ports["kv_cache_state"].representation == "paged_latent_kv"
    effects = {effect.target: effect for effect in fragment.state_effects}
    assert effects["kv_cache"].action == "write"
    assert not fragment.work.by_resource("useful")
