"""Typed family-work bridge: real compiled slices, shapes and state.

[Role]
  Exercise the bridge between a compiled piecewise family and the common
  typed family record.  One real dense GEMM from the compact GLM-5.3 catalog
  and one real window-compressed attention layer provide the slices; synthetic
  GEMM declarations vary shape, dtype and width so every check is an
  independent arithmetic identity rather than a fixture lookup.

[Invariants]
  * Typed operand payloads are the exact derived quantities the family
    compiler charged; a dtype/byte-width disagreement raises.
  * The stage plan is the compiler's serial ledger and lowering it to the
    existing Engine reproduces the declared structural duration.
  * A KV readiness read keeps its state effect and zero work; a store declares
    its cache-row payload and exact bytes.
  * Every family compiler remains listed; a generic fallback is explicit
    physical coverage, never a metadata-only completion claim.

[Timeouts & Locks]
  Pure in-memory arithmetic and bounded Engine advance; no device, network,
  filesystem writes or locks.

[Side-Effect Policy]
  Read-only catalog load and local model copies; no production mutation beyond
  the test-local compiled object.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import math

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, RequestGeometry,
    WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.families import (
    FAMILY_COMPILERS, family_bridge_status, family_work, lower_family_work,
)
from mindie_sim.prediction.piecewise.families import FamilyFragmentProgram
from mindie_sim.prediction.piecewise.families_bridge import (
    generic_family_declaration,
)
from mindie_sim.prediction.piecewise.expressions import as_fraction
from mindie_sim.prediction.subgraphs import FeatureConfig
from mindie_sim.prediction.subgraphs.instantiate import instantiate_model
from mindie_sim.prediction.subgraphs.state import MoeRoutingState, StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature
from mindie_sim.schema.tensors import TensorLayout, TensorValue

SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, split_target=1000,
                              max_splits=4, pipeline_slots=2)
_GEOMETRY = WorkloadGeometry.homogeneous(1, 17, 18)

_CAPS = {
    "q_lora_rank": 128, "kv_lora_rank": 64, "index_heads": 8,
    "index_dim": 32, "index_topk": 64, "expert_latent_size": 256,
}


def _model() -> object:
    """Compact GLM-5.3 with the catalog's declared structural caps."""
    source = load_model("glm-5.3")
    plan = source.layer_plans[0]
    updates = {"index": 0, "native_layer_id": 0}
    for field, cap in _CAPS.items():
        value = int(getattr(plan, field) or 0)
        if value > cap:
            updates[field] = cap
    plan = replace(plan, **updates)
    return replace(
        source, name="glm-5.3-l1", hidden_size=512, intermediate_size=256,
        num_experts=0, top_k=0, moe_intermediate_size=0,
        layers=1, body_layers=1, vocab_size=256, layer_plans=(plan,))


def _instance(model: object) -> object:
    features = FeatureConfig.from_model(
        model, tp=1, dp=1, ep=1, cores=2, token_tile=128, kv_tile=128,
        moe_row_tile=16, split_target=1000, max_splits=4)
    return instantiate_model(model, features)


def _bound(model: object, kind: str, name: str) -> BoundOperator:
    instance = _instance(model)
    for subgraph in instance.layers[0].subgraphs:
        if subgraph.kind != kind:
            continue
        for operator in subgraph.operators:
            if operator.name == name:
                return operator
    raise AssertionError(f"operator_missing:{kind}:{name}")


def _compiled(operator: BoundOperator, kind: str) -> object:
    return OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        operator, kind=kind)


def _gemm_op(k: int, n: int, *, act_format: str = "bf16",
             weight_format: str = "bf16", output_format: str = "",
             act_bytes: float = 2.0) -> BoundOperator:
    weight_width = {"bf16": 2.0, "fp8_e4m3": 1.0, "int8": 1.0,
                    "int4": 0.5}.get(weight_format, 2.0)
    payload = n * k * weight_width
    quant = (("act_bytes", act_bytes), ("act_format", act_format),
             ("weight_format", weight_format),
             ("weight_payload", payload), ("weight_total", payload))
    if output_format:
        quant += (("output_format", output_format),)
    template = OperatorTemplate(name="MatMulV2", family="gemm",
                                role="projection")
    signature = StaticSignature(
        role="MatMulV2", matrix=(("K", k), ("N", n)), quant=quant,
        tiles=(("mt", 16), ("nt", 16), ("cores", 2),
               ("split_target", 1000), ("max_splits", 4)),
        extras=(("batch", 1),))
    return BoundOperator(template, signature)


def _engine(work: object) -> Engine:
    rates = dict(SUPPLY.rates)
    used = {item.resource for stage in work.stages
            for item in stage.work.issued()}
    return Engine([Resource(pipe, rates[pipe]) for pipe in sorted(used)])


def _execute(work: object) -> float:
    engine = _engine(work)
    ready = engine.submit("ready")
    done = lower_family_work(work).start(engine, "family", ready)
    engine.advance(until=(done,))
    return engine.now


# 1. A real dense GEMM slice carries the exact compiled stage ledger.

def test_real_dense_gemm_typed_work_is_the_compiled_ledger():
    operator = _bound(_model(), "mlp", "gate_up")
    compiled = _compiled(operator, "mlp")
    cost = compiled.evaluate(_GEOMETRY)
    work = family_work(compiled, _GEOMETRY)
    theta = operator.theta

    assert work.name == "gate_up" and work.family == "gemm"
    assert work.active and work.projector == (
        "families_matrix:_gemm_typed_declaration")
    operands = {operand.name: operand for operand in work.operands}
    assert set(operands) == {"activation", "weight", "output"}
    k = int(theta.require("K"))
    n = int(theta.require("N"))
    activation = operands["activation"]
    weight = operands["weight"]
    output = operands["output"]
    assert activation.dtype == theta.find("act_format")
    assert weight.dtype == theta.find("weight_format")
    assert output.dtype == theta.find(
        "output_format", theta.find("act_format"))
    assert activation.shape() == (1, 17, k)
    assert weight.shape() == (k, n)
    assert output.shape() == (1, 17, n)
    assert activation.payload_bytes() == cost.get("activation_payload_bytes")
    assert weight.payload_bytes() == cost.get("weight_payload")
    assert output.payload_bytes() == cost.get("output_bytes")

    assert len(work.stages) == len(compiled.program.stages) == 6
    for index, (stage, demands) in enumerate(
            zip(work.stages, compiled.program.stages)):
        assert stage.index == index
        assert stage.depends_on == (() if index == 0
                                    else (f"stage{index - 1}",))
        units = {demand.pipe: demand.unit for demand in demands}
        for item in stage.work.issued():
            assert item.unit == units[item.resource]
    assert work.issued_work() == {pipe: as_fraction(value)
                                  for pipe, value in cost.work}
    assert work.service.elapsed_owner == "stages"
    assert float(work.service.structural_ms) == pytest.approx(
        float(cost.duration_ms), rel=1e-12)
    assert work.service.completion(work.service.structural_ms) == pytest.approx(
        float(cost.duration_ms), rel=1e-12)

    assert work.ports.names == ("activation", "weight", "output")
    assert {port.name for port in work.ports.tensor_in} == {
        "activation", "weight"}
    values = dict(work.tensor_values())
    assert set(values) == {"activation", "weight", "output"}
    assert values["weight"].layout.shape == (k, n)
    assert values["weight"].dtype == weight.dtype
    assert values["weight"].quant == str(
        theta.find("linear_method", "") or "")
    assert values["output"].layout.shape == (1, 17, n)
    payload = work.to_dict()
    assert payload["family"] == "gemm"
    assert payload["operands"][0]["shape"] == [1, 17, k]
    assert payload["service"]["elapsed_owner"] == "stages"
    json.dumps(payload)
    assert {port.name for port in work.ports.tensor_out} == {"output"}
    assert not work.ports.state_in and not work.ports.state_out

    program = lower_family_work(work)
    assert len(program.stages) == len(work.stages)
    for stage, family_stage in zip(program.stages, work.stages):
        assert stage.after == family_stage.depends_on
        assert stage.work_units == {
            item.resource: item.unit
            for item in family_stage.work.issued()}

    with pytest.raises(ModelingError, match="family_work_fragment_requires_single_stage"):
        work.fragment()

    elapsed = _execute(work)
    assert elapsed == pytest.approx(
        float(work.service.structural_ms), rel=1e-9)


def test_real_dense_slice_generalises_over_query_tokens():
    compiled = _compiled(_bound(_model(), "mlp", "gate_up"), "mlp")
    theta = compiled.operator.theta
    k = int(theta.require("K"))
    n = int(theta.require("N"))
    mt = int(theta.find("mt", 128) or 128)
    previous_cube = Fraction(0)
    for tokens in (1, 17, 257):
        geometry = WorkloadGeometry.homogeneous(1, tokens, tokens + 1)
        cost = compiled.evaluate(geometry)
        work = family_work(compiled, geometry)
        operands = {operand.name: operand for operand in work.operands}
        regime = dict(work.geometry)
        assert operands["activation"].shape() == (1, tokens, k)
        assert operands["output"].shape() == (1, tokens, n)
        assert regime["rows"] == tokens
        assert regime["M_TILES"] == -(-tokens // mt)
        assert work.issued_work() == {pipe: as_fraction(value)
                                      for pipe, value in cost.work}
        assert work.pipe_work("cube") >= previous_cube
        previous_cube = work.pipe_work("cube")


# 2. A declared invocation floor completes the body; it is never invented work.

def test_invocation_floor_is_a_completion_constraint_not_work():
    supply = PipeSupply(SUPPLY.rates, invocations=(("cube", 0.25),))
    operator = _gemm_op(64, 32)
    compiled = OperatorPiecewiseCompiler(supply, POLICY).compile_operator(
        operator)
    cost = compiled.evaluate(_GEOMETRY)
    work = family_work(compiled, _GEOMETRY)

    assert work.service.floor_ms == Fraction(1, 4)
    assert work.issued_work() == {pipe: as_fraction(value)
                                  for pipe, value in cost.work}
    assert work.service.completion(work.service.structural_ms) == pytest.approx(
        float(cost.duration_ms), rel=1e-12)
    assert float(cost.duration_ms) >= 0.25
    # The Engine executes only structural work; the floor is applied by the
    # caller through FamilyService, so it cannot be charged twice.
    assert _execute(work) == pytest.approx(
        float(work.service.structural_ms), rel=1e-9)


# 3. Shapes, dtype widths and per-case work follow declared formulas.

def test_dense_work_generalises_over_new_shapes_and_dtypes():
    cases = (
        (96, 80, 37, "bf16", "bf16", "", 2.0, 2.0),
        (96, 80, 37, "fp8_e4m3", "fp8_e4m3", "bf16", 1.0, 1.0),
        (128, 64, 1, "bf16", "int8", "bf16", 2.0, 1.0),
        (160, 96, 129, "fp8_e4m3", "int4", "bf16", 1.0, 0.5),
    )
    for k, n, tokens, act_format, weight_format, out_format, act_bytes, \
            weight_width in cases:
        operator = _gemm_op(
            k, n, act_format=act_format, weight_format=weight_format,
            output_format=out_format, act_bytes=act_bytes)
        compiled = _compiled(operator, "")
        geometry = WorkloadGeometry.homogeneous(1, tokens, tokens + 1)
        cost = compiled.evaluate(geometry)
        work = family_work(compiled, geometry)
        operands = {operand.name: operand for operand in work.operands}
        assert cost.get("flops") == 2 * tokens * n * k
        assert cost.get("rows") == tokens
        assert operands["activation"].shape() == (1, tokens, k)
        assert operands["output"].shape() == (1, tokens, n)
        assert operands["activation"].payload_bytes() == (
            tokens * k * as_fraction(act_bytes))
        assert operands["weight"].payload_bytes() == as_fraction(
            k * n * weight_width)
        out_width = {"bf16": 2.0, "fp8_e4m3": 1.0}.get(
            out_format or act_format, 2.0)
        assert operands["output"].payload_bytes() == (
            tokens * n * as_fraction(out_width))
        assert work.issued_work() == {pipe: as_fraction(value)
                                      for pipe, value in cost.work}

    bf16 = _compiled(_gemm_op(96, 80, act_bytes=2.0), "")
    fp8 = _compiled(_gemm_op(
        96, 80, act_format="fp8_e4m3", weight_format="fp8_e4m3",
        output_format="bf16", act_bytes=1.0), "")
    for tokens in (1, 37, 129):
        geometry = WorkloadGeometry.homogeneous(1, tokens, tokens + 1)
        old = family_work(bf16, geometry)
        new = family_work(fp8, geometry)
        assert old.pipe_work("cube") == new.pipe_work("cube")
        assert new.pipe_work("hbm") < old.pipe_work("hbm")
        assert {operand.name: operand.payload_bytes()
                for operand in new.operands}["weight"] == (
            {operand.name: operand.payload_bytes()
             for operand in old.operands}["weight"] // 2)


def test_typed_width_disagreement_raises_instead_of_silently_choosing():
    operator = _gemm_op(
        64, 32, act_format="fp8_e4m3", output_format="bf16", act_bytes=2.0)
    compiled = _compiled(operator, "")
    cost = compiled.evaluate(_GEOMETRY)
    assert cost.get("activation_payload_bytes") == 2 * 17 * 64
    with pytest.raises(ModelingError,
                       match="family_typed_operand_mismatch:MatMulV2"):
        family_work(compiled, _GEOMETRY)


# 4. A real window-compressed layer carries KV state and dynamic positions.

def _toy_window_model() -> object:
    raw = dict(
        hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=128, name="toy-window", layers=2,
        layer_plans=[
            dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 qk_norm=True),
            dict(index=1, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 qk_norm=True, cache_kind="window_compressed",
                 compress_ratio=4, compressor_kind="weighted_pool",
                 kv_owner_layer_id=0, indexer=True, index_topk=8),
        ])
    return ModelSpec.from_mapping(raw)


def _compiled_in_layer(instance: object, layer: int, kind: str | None,
                       name: str) -> BoundOperator:
    for subgraph in instance.layers[layer].subgraphs:
        if kind is not None and subgraph.kind != kind:
            continue
        for operator in subgraph.operators:
            if operator.name == name:
                return operator
    raise AssertionError(f"operator_missing:{layer}:{kind}:{name}")


def test_kv_read_and_store_keep_state_effects_and_zero_work():
    instance = _instance(_toy_window_model())
    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    read = _compiled(
        _compiled_in_layer(instance, 1, None, "kv_read"), "kv_cache")
    state = StateView().mark_kv(0)
    work = family_work(read, geometry, state)

    assert work.active and len(work.stages) == 1
    assert work.operands == ()
    assert [(effect.action) for effect in work.state_effects] == ["read"]
    assert work.state_effects[0].target.startswith("kv_cache:layer:0")
    assert work.ports.names == (
        f"{work.state_effects[0].target}_state_in",)
    assert work.pipe_work("hbm") == work.pipe_work("mte2") == 0
    assert dict(work.geometry)["kv_positions"] == 16
    assert dict(work.geometry)["valid_rows"] == 4
    with pytest.raises(ModelingError, match="state_port_not_ready:kv_cache"):
        family_work(read, geometry)

    store = _compiled(
        _compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention")
    stored = family_work(store, geometry)
    assert [effect.action for effect in stored.state_effects] == ["write"]
    assert stored.state_effects[0].target == work.state_effects[0].target
    assert stored.ports.names == (
        "kv_cache_row", f"{stored.state_effects[0].target}_state_out")
    operand = stored.operands[0]
    assert operand.shape() == (4, 128)
    assert operand.payload_bytes() == dict(stored.derived)["kv_bytes"]
    assert stored.pipe_work("hbm") == 4 * 128
    kv_value = dict(stored.tensor_values())["kv_cache_row"]
    assert kv_value.layout.shape == (4, 128)
    assert kv_value.dtype == "uint8"
    fragment = stored.fragment()
    assert fragment.work == stored.stages[0].work
    assert fragment.state_effects == stored.state_effects
    # T01 strict projection must not silently drop attrs/state; the complete
    # fragment is retained instead.
    assert not fragment.supports_stage_projection
    lowered = lower_family_work(stored)
    assert lowered.fragments[0].state_effects == fragment.state_effects
    assert lowered.fragments[0].work == fragment.work
    assert lowered.fragments[0].ports == fragment.ports


# 5. Unregistered families stay explicit; every compiler remains listed.

def test_family_status_covers_the_baseline_without_metadata_only_claims():
    status = family_bridge_status()
    assert {row[0] for row in status} == set(FAMILY_COMPILERS)
    projected = {row[0] for row in status if row[1] == "projected"}
    assert {"gemm", "kv_access", "attention", "mla"} <= projected
    assert projected == set(FAMILY_COMPILERS)
    assert not [row for row in status if row[1] == "projected_generic"]
    for family, claim, reason in status:
        assert reason
        assert claim in {"projected", "projected_generic"}
        if claim == "projected":
            assert ":" in reason
        else:
            assert reason == "generic_physical:logical_producer_pending"

    elementwise = _compiled(_bound(_model(), "layernorm", "rmsnorm"), "layernorm")
    work = family_work(elementwise, _GEOMETRY)
    assert work.family == "elementwise"
    assert work.declaration_complete is True
    assert not work.missing_producers
    assert work.stages and work.issued_work()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_moe_fixture", "tests/test_piecewise_operators_matrix_moe.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    generic_op = BoundOperator(
        OperatorTemplate("sinkhorn", "recurrent", "sinkhorn"),
        StaticSignature(role="sinkhorn", tiles=(("mt", 16),),
                        quant=(("act_format", "bf16"),),
                        extras=(("iters", 4), ("mult", 2), ("H", 32))))
    generic_compiled = _compiled(generic_op, "")
    generic_cost = generic_compiled.evaluate(
        WorkloadGeometry.homogeneous(1, 4, 5))
    generic = generic_family_declaration(generic_compiled, generic_cost)
    assert generic.producer_complete is False
    assert generic.missing
    assert generic.operands
    assert all(operand.dtype in {"bf16", "uint8"}
               for operand in generic.operands)
    assert any(operand.dtype == "bf16" for operand in generic.operands)
    recurrent = family_work(generic_compiled,
                            WorkloadGeometry.homogeneous(1, 4, 5))
    assert recurrent.family == "recurrent"
    assert {operand.name for operand in recurrent.operands} == {
        "hidden", "hidden_out"}
    assert recurrent.declaration_complete is True


# 6. Aggregate native results are refused; a real NativeProgram is reused.

def test_native_aggregate_is_refused_as_family_stage_ledger():
    compiled = _compiled(_bound(_model(), "mlp", "gate_up"), "mlp")

    def native_duration(geometry, state):
        # Five-tuple aggregate: duration, dominant, tied, busy, total work.
        # It has no source stage order and must not become one stage.
        return (Fraction(1), "cube", ("cube",), {"cube": Fraction(10)},
                {"cube": Fraction(10)})

    compiled.program = replace(
        compiled.program, program_duration=native_duration)
    with pytest.raises(ModelingError, match="family_stage_ledger_unavailable"):
        family_work(compiled, _GEOMETRY, occurrence=Occurrence(
            kind="kernel", name="aggregate", invocation="aggregate"))


def test_real_native_program_is_reused_and_shares_hbm():
    from mindie_sim.prediction.program import KernelProgram, Stage
    from mindie_sim.prediction.piecewise.compiler import NativeProgram

    native_kernel = KernelProgram((
        Stage("hbm", work={"hbm": 1000.0}),
        Stage("cube", work={"cube": 1000.0}, after=("hbm",)),
    ))
    native = NativeProgram(
        program=native_kernel,
        work=(("hbm", Fraction(1000)), ("cube", Fraction(1000))),
        duration=Fraction(2))

    def native_duration(geometry, state):
        return (Fraction(2), "cube", ("cube",),
                {"hbm": Fraction(1), "cube": Fraction(1)},
                {"hbm": Fraction(1000), "cube": Fraction(1000)},
                native)

    compiled = _compiled(_bound(_model(), "mlp", "gate_up"), "mlp")
    compiled.program = replace(
        compiled.program, program_duration=native_duration)
    work = family_work(compiled, _GEOMETRY)
    assert work.native_program is native
    assert work.native_stages == ()
    assert work.stages == ()
    assert lower_family_work(work) is native

    engine = Engine([Resource("hbm", 1000), Resource("cube", 1000)])
    ready = engine.submit("ready")
    done = native.start(engine, "native", ready)
    engine.advance(until=(done,))
    assert engine.busy_ms["hbm"] == 1.0
    assert engine.busy_ms["cube"] == 1.0
    assert engine.now == 2.0

    engine = Engine([Resource("hbm", 1000), Resource("cube", 1000)])
    ready = engine.submit("ready")
    done = native.start(engine, "native", ready)
    competitor = engine.submit("competitor", work={"hbm": 1000.0},
                               after=(ready,))
    engine.advance(until=(done, competitor))
    assert engine.busy_ms["hbm"] == 2.0
    assert engine.now == 2.0


# 7. The lowered Stage carries the compiler's exact executed caps.

def test_lowered_stage_preserves_compiled_caps():
    operator = _gemm_op(16, 16, act_bytes=2.0, output_format="bf16")
    compiled = _compiled(operator, "")
    geometry = WorkloadGeometry.homogeneous(1, 1, 2)
    work = family_work(compiled, geometry)
    caps = dict(work.caps)
    assert caps.get("cube") == Fraction(500)
    assert work.pipe_work("cube") == 8192
    program = lower_family_work(work)
    assert all(dict(stage.caps).get("cube") == Fraction(500)
               for stage in program.stages)
    engine = _engine(work)
    ready = engine.submit("ready")
    done = program.start(engine, "capped", ready)
    engine.advance(until=(done,))
    assert engine.busy_ms["cube"] == pytest.approx(8192 / 500, rel=1e-12)
    assert engine.now == pytest.approx(
        float(work.service.structural_ms), rel=1e-12)


# 8. State effects execute through T04's conductor; a bare Stage is not enough.

def test_stateful_lowering_publishes_through_fragment_conductor():
    from mindie_sim.prediction.scheduling import FragmentConductor
    from mindie_sim.schema.common import ModelingError

    instance = _instance(_toy_window_model())
    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    store = _compiled(_compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention")
    store.layer = 0
    occurrence = Occurrence(kind="kernel", name="kv", invocation="req-1", rank=0)
    stored = family_work(store, geometry, occurrence=occurrence)
    target = stored.state_effects[0].target
    program = lower_family_work(stored)
    assert hasattr(program, "fragments")
    engine = Engine([Resource("hbm", 1000.0, unit="bytes"),
                     Resource("mte2", 1000.0, unit="bytes"),
                     Resource("mte3", 1000.0, unit="bytes"),
                     Resource("link", 1000.0, unit="bytes")])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    done = program.start(engine, "store", ready, conductor=conductor)
    conductor.advance(until=(done,))
    assert conductor.state.live_versions(target) == (1,)
    assert conductor.state.committed_version(target) is None

    read = _compiled(_compiled_in_layer(instance, 1, None, "kv_read"), "kv_cache")
    read.layer = 0
    read_work = family_work(
        read, geometry, state=StateView().mark_kv(0),
        occurrence=occurrence, state_versions={"kv_cache": 1})
    read_program = lower_family_work(read_work)
    read_end = read_program.start(
        engine, "read", done, conductor=conductor)
    conductor.advance(until=(read_end,))
    assert read_end.time == done.time

    before = _compiled(_compiled_in_layer(instance, 1, None, "kv_read"), "kv_cache")
    before.layer = 0
    early = family_work(
        before, geometry, state=StateView().mark_kv(0),
        occurrence=occurrence, state_versions={"kv_cache": 1})
    fresh = FragmentConductor(Engine([
        Resource("hbm", 1000.0, unit="bytes"),
        Resource("mte2", 1000.0, unit="bytes"),
        Resource("link", 1000.0, unit="bytes")]))
    early_program = lower_family_work(early)
    early_ready = fresh.engine.submit("ready")
    early_program.start(fresh.engine, "read", early_ready, conductor=fresh)
    with pytest.raises(Exception, match="state_"):
        fresh.advance()

    other_layer = _compiled(
        _compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention")
    other_layer.layer = 1
    other = family_work(other_layer, geometry, occurrence=occurrence)
    assert other.state_effects[0].target != target
    slot_a = family_work(
        _compiled(_compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention"),
        geometry, occurrence=occurrence, state_slot="candidate")
    slot_b = family_work(
        _compiled(_compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention"),
        geometry, occurrence=occurrence, state_slot="committed")
    assert slot_a.state_effects[0].target != slot_b.state_effects[0].target


# 9. TensorValue identity is producer/output-bound; inputs stay external.

def test_tensor_values_keep_input_output_and_occurrence_identity():
    operator = _gemm_op(16, 16, act_bytes=2.0, output_format="bf16")
    compiled = _compiled(operator, "")
    geometry = WorkloadGeometry.homogeneous(1, 2, 3)
    first = Occurrence(kind="kernel", name="mm", serial=1, invocation="i")
    second = Occurrence(kind="kernel", name="mm", serial=2, invocation="i")
    one = family_work(compiled, geometry, occurrence=first)
    two = family_work(compiled, geometry, occurrence=second)
    values_one = dict(one.tensor_values())
    values_two = dict(two.tensor_values())
    assert values_one["activation"] != values_one["output"]
    assert values_one["activation"].binding == "external"
    assert values_one["activation"].binding_name.endswith(":activation")
    assert values_one["output"].producer == "kernel:mm:i:serial1"
    assert values_one["output"].output == 0
    assert values_one["activation"] == values_two["activation"]
    assert values_one["output"] != values_two["output"]

    view = TensorLayout((1, 2, 16), (16, 32, 1), storage_shape=(1, 2, 16))
    with_view = dict(one.tensor_values(layouts={"activation": view}))
    assert with_view["activation"].layout == view
    assert with_view["activation"].layout.is_contiguous is False

    upstream = TensorValue(
        "upstream", "bf16", TensorLayout.contiguous((1, 2, 16)))
    reused = dict(one.tensor_values(input_values={"activation": upstream}))
    assert reused["activation"] is upstream


# 10. KV reads use the bound ROWS access range, not valid query rows.

def test_kv_read_uses_bound_rows_and_store_keeps_valid_rows():
    supply = SUPPLY
    policy = POLICY

    def kv_op(name: str) -> BoundOperator:
        template = OperatorTemplate(name=name, family="kv_access", role="consume")
        signature = StaticSignature(
            role=name, matrix=(("rows", 1), ("row_bytes", 128.0)),
            extras=(("owner_layer", 0),))
        return BoundOperator(template, signature)

    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    read = OperatorPiecewiseCompiler(supply, policy).compile_operator(
        kv_op("kv_read"))
    read_work = family_work(read, geometry, state=StateView().mark_kv(0))
    read_operand = read_work.operands[0]
    assert read_operand.shape() == (16, 128)
    assert read_operand.payload_bytes() == 2048
    assert dict(read_work.geometry)["kv_positions"] == 16
    assert dict(read_work.geometry)["valid_rows"] == 4

    store = OperatorPiecewiseCompiler(supply, policy).compile_operator(
        kv_op("kv_store"))
    store_work = family_work(store, geometry)
    store_operand = store_work.operands[0]
    assert store_operand.shape() == (4, 128)
    assert store_operand.payload_bytes() == 512
    assert dict(store_work.geometry)["kv_rows"] == 4


# 11. Two concurrent programs share the same capped resource without losing work.

def test_concurrent_capped_programs_keep_shared_resource_competition():
    base = PipeSupply((
        ("cube", 1000.0), ("vector", 1e15), ("hbm", 1e15),
        ("mte1", 1e15), ("mte2", 1e15), ("mte3", 1e15),
        ("fixpipe", 1e15), ("link", 1e15),
    ))
    operator = _gemm_op(16, 16, act_bytes=2.0, output_format="bf16")
    compiled = OperatorPiecewiseCompiler(base, POLICY).compile_operator(operator)
    work = family_work(compiled, WorkloadGeometry.homogeneous(1, 1, 2))
    assert dict(work.caps)["cube"] == Fraction(500)
    rates = dict(base.rates)
    used = {item.resource for stage in work.stages
            for item in stage.work.issued()}
    engine = Engine([Resource(pipe, rates[pipe]) for pipe in sorted(used)])
    ready = engine.submit("ready")
    first = lower_family_work(work)
    second = lower_family_work(work)
    done_a = first.start(engine, "a", ready)
    done_b = second.start(engine, "b", ready)
    engine.advance(until=(done_a, done_b))
    assert engine.now == pytest.approx(
        float(work.service.structural_ms), rel=1e-12)
    assert engine.serviced["cube"] == pytest.approx(2 * 8192, rel=1e-12)
    assert engine.busy_ms["cube"] == pytest.approx(8192 / 500, rel=1e-12)


# 12. Attention and MLA expose their real q/KV/output typed values.

def test_attention_typed_declaration_uses_bound_q_kv_output_geometry():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_attn_fixture", "tests/test_piecewise_operators_attention_mla.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    compiled = fixture.compile_attention(8, 2, 64)
    operator = compiled.operator("attention")
    geometry = fixture.decode([(1, 16)])
    work = family_work(operator, geometry)
    values = dict(work.tensor_values())
    assert set(values) == {"q", "k_cache", "v_cache", "output"}
    assert values["q"].layout.shape == (8, 64)
    assert values["q"].dtype == "bf16"
    assert values["k_cache"].layout.shape == (17, 2, 64)
    assert values["v_cache"].layout.shape == (17, 2, 64)
    assert values["k_cache"].dtype == "bf16"
    assert values["v_cache"].dtype == "bf16"
    assert values["output"].layout.shape == (8, 64)
    assert work.declaration_complete is True
    derived = dict(work.derived)
    assert values["q"].layout.shape[0] * values["q"].layout.shape[1] * 2 == (
        derived["q_bytes"])
    k_bytes = math.prod(values["k_cache"].layout.shape) * 2
    v_bytes = math.prod(values["v_cache"].layout.shape) * 2
    assert k_bytes == v_bytes == derived["kv_value_bytes"] // 4
    assert k_bytes + v_bytes == derived["kv_value_bytes"] // 2


# 13. Materialization/internal/residency remain part of the family record.

def test_family_record_preserves_materialization_and_residency():
    compiled = _compiled(_bound(_model(), "mlp", "gate_up"), "mlp")
    cost = compiled.evaluate(_GEOMETRY)
    work = family_work(compiled, _GEOMETRY)
    assert dict(work.materialization) == dict(cost.materialization)
    assert dict(work.internal) == dict(cost.internal)
    assert work.residency_bytes == cost.residency_bytes


# 14. Grouped GEMM uses routed rows, per-expert weight and padded output.

def test_grouped_gemm_typed_declaration_uses_routed_geometry():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_grouped_fixture", "tests/test_piecewise_operators_matrix_moe.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    operator = fixture.grouped_op()
    state = StateView(routing=((0, MoeRoutingState((2,) * 8, 2, 8)),))
    compiled = fixture.compile_op(operator)
    work = family_work(
        compiled, WorkloadGeometry.homogeneous(1, 8, 9), state)
    operands = {operand.name: operand for operand in work.operands}
    assert operands["activation"].shape() == (16, 64)
    assert operands["weight"].shape() == (8, 64, 128)
    assert operands["output"].shape() == (128, 128)
    assert work.declaration_complete is True
    assert work.pipe_work("hbm") == sum(
        value for pipe, value in work.issued_work().items() if pipe == "hbm")


# 15. Transport keeps the directed payload and materialized peer traffic.

def test_transport_typed_declaration_keeps_directed_payload():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_transport_fixture", "tests/test_piecewise_operators_matrix_moe.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    operator = fixture.traffic_op(
        name="alltoallv", family="transport", role="dispatch",
        ranks=2, node_ranks=1, mtu=4096, row_bytes=8.0)
    work = family_work(
        fixture.compile_op(operator), WorkloadGeometry.homogeneous(1, 4, 5))
    assert work.family == "transport"
    assert work.declaration_complete is True
    values = dict(work.tensor_values())
    assert "payload" in values and "peer_traffic" in values
    assert values["payload"].dtype == "uint8"


# 16. State gather separates index, selected state and written state rows.

def test_state_gather_typed_declaration_separates_state_and_indices():
    signature = StaticSignature(
        role="gather_ssm_states",
        heads=(("value_heads", 2), ("dk", 4), ("dv", 4)),
        extras=(("state_dtype", "bf16"), ("index_dtype", "int32")))
    operator = BoundOperator(
        OperatorTemplate("gather_ssm_states", "state_gather", "state"),
        signature)
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(operator)
    geometry = WorkloadGeometry(
        (RequestGeometry(4, 8, phase="prefill"),), phase="prefill")
    work = family_work(compiled, geometry)
    operands = {operand.name: operand for operand in work.operands}
    assert operands["indices"].shape() == (1,)
    assert operands["indices"].dtype == "int32"
    assert operands["state_in"].shape() == (1, 2, 4, 4)
    assert operands["state_in"].dtype == "bf16"
    assert operands["state_out"].shape() == (1, 2, 4, 4)
    assert work.declaration_complete is True


# 17. Multi-stage families expose complete per-stage fragments/events.

def test_multistage_family_exposes_stage_fragments_and_events():
    compiled = _compiled(_bound(_model(), "mlp", "gate_up"), "mlp")
    work = family_work(compiled, _GEOMETRY)
    fragments = work.fragments()
    assert len(fragments) == len(work.stages) == 6
    assert fragments[0].ports.tensor_in
    assert not fragments[1].ports.tensor_in
    assert fragments[-1].ports.tensor_out
    for index, fragment in enumerate(fragments):
        assert fragment.occurrence.name.endswith(f":stage{index}")
        assert fragment.ports.event_out[0].name == f"stage{index}_complete"
        if index == 0:
            assert fragment.ports.event_in[0].name == "stage0_ready"
        else:
            assert fragment.ports.event_in[0].name == (
                f"stage{index - 1}_complete")
    assert work.phase == "decode"
    # Stateless multi-stage execution still uses the existing KernelProgram.
    program = lower_family_work(work)
    assert hasattr(program, "stages") and len(program.stages) == 6


# 18. Scatter keeps separate source and destination state widths.

def test_state_scatter_keeps_mixed_source_and_state_widths():
    signature = StaticSignature(
        role="scatter_recurrent_state",
        heads=(("value_heads", 1), ("dk", 64), ("dv", 32)),
        extras=(("state_dtype", "fp32"), ("source_dtype", "fp16"),
                ("index_dtype", "int32"), ("block_size", 128)))
    operator = BoundOperator(
        OperatorTemplate("scatter_recurrent_state", "state_scatter", "state"),
        signature)
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(operator)
    geometry = WorkloadGeometry(
        (RequestGeometry(1, 8, phase="prefill"),), phase="prefill")
    work = family_work(compiled, geometry, state_slot="ssm",
                       state_versions={"recurrent_state": 8})
    operands = {operand.name: operand for operand in work.operands}
    assert operands["state_in"].dtype == "fp16"
    assert operands["state_in"].payload_bytes() == 4096
    assert operands["state_out"].dtype == "fp32"
    assert operands["state_out"].payload_bytes() == 8192
    assert dict(work.derived)["state_scatter_state_read_bytes"] == 4096
    assert dict(work.derived)["state_scatter_index_bytes"] == 4
    assert work.pipe_work("mte2") == 4096 + 4
    assert work.pipe_work("mte3") == 8192
    assert work.pipe_work("hbm") == 12292
    assert len(work.stages) == 1
    assert dict(work.stages[0].exact_work)["hbm"] == 12292
    access = work.state_accesses()[0]
    assert access["direction"] == "write"
    assert access["bytes"] == 8192
    assert access["state_version"] == 8
    assert access["storage"] == "recurrent_state"


# 19. Rope consumes the kernel's per-row R range, not the full head width.

def test_rope_family_uses_per_row_rotary_extent():
    def rope(rope_dim: int) -> BoundOperator:
        signature = StaticSignature(
            role="index_rope",
            matrix=(("elements", 128),),
            quant=(("act_bytes", 2.0), ("act_format", "bf16")),
            tiles=(("mt", 16), ("cores", 2)),
            extras=(("qk_rope_head_dim", rope_dim), ("index_heads", 1),
                    ("index_dim", 128), ("is_neox_style", 1),
                    ("rope_dtype", "bf16")))
        return BoundOperator(
            OperatorTemplate("index_rope", "elementwise", "rope"), signature)

    for rope_dim, read_bytes, write_bytes, math_ops in (
            (64, 256, 128, 192), (32, 128, 64, 96)):
        compiled = _compiled(rope(rope_dim), "")
        geometry = WorkloadGeometry.homogeneous(1, 1, 2)
        work = family_work(compiled, geometry)
        derived = dict(work.derived)
        assert derived["rope_dim"] == rope_dim
        assert derived["logical_elements"] == 128
        assert derived["issued_elements"] == rope_dim
        assert work.pipe_work("mte2") == read_bytes
        assert work.pipe_work("mte3") == write_bytes
        assert derived["rope_math_ops"] == math_ops
        assert work.pipe_work("vector") >= math_ops
        assert work.declaration_complete is True


# 20. Index cache writes expose the K payload and scale cache as two slots.

def test_index_cache_store_splits_k_and_scale_slots():
    signature = StaticSignature(
        role="index_cache_store",
        matrix=(("rows", 1), ("row_bytes", 80.0)),
        extras=(("owner_layer", 0), ("scale_bytes", 16.0),
                ("valid_rows_mode", "explicit"), ("index_dtype", "int32")))
    operator = BoundOperator(
        OperatorTemplate("index_cache_store", "kv_access", "produce"),
        signature)
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(operator)
    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    work = family_work(compiled, geometry)
    operands = {operand.name: operand for operand in work.operands}
    assert set(operands) == {"index_k_cache", "index_scale_cache"}
    assert operands["index_k_cache"].shape() == (1, 64)
    assert operands["index_scale_cache"].shape() == (1, 16)
    assert len({effect.target for effect in work.state_effects}) == 2
    assert len(work.stages) == 2
    assert [dict(stage.exact_work).get("mte3") for stage in work.stages] == [
        Fraction(64), Fraction(16)]
    assert work.pipe_work("mte3") == 80
    assert len(work.fragments()) == 2



# 21. L2Norm consumes the native BD/MBLOCK tile interface.

def test_l2norm_consumes_native_bd_and_mblock():
    signature = StaticSignature(
        role="k_l2norm",
        matrix=(("elements", 64),),
        quant=(("act_bytes", 2.0),),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("ops_per_element", 5), ("BD", 64), ("MBLOCK", 32),
                ("source_kernel", "l2norm_fwd_kernel2")))
    operator = BoundOperator(
        OperatorTemplate("k_l2norm", "elementwise", "norm"), signature)
    compiled = _compiled(operator, "layernorm")
    geometry = WorkloadGeometry.homogeneous(1, 1, 2)
    work = family_work(compiled, geometry)
    derived = dict(work.derived)
    assert derived["BD"] == 64
    assert derived["MBLOCK"] == 32
    assert work.pipe_work("mte2") == 64 * 2
    assert work.pipe_work("mte3") == 64 * 2
    assert derived["logical_elements"] == 64
    assert derived["l2_logical_math"] == 64 * 5
    assert derived["l2_issued_math"] == 32 * 64 * 5


# 22. Draft heads expose real hidden input/output operands.

def test_draft_heads_typed_hidden_operands():
    signature = StaticSignature(
        role="draft_heads",
        matrix=(("K", 32), ("N", 32)),
        tiles=(("mt", 16), ("nt", 16)),
        extras=(("draft_len", 2), ("verify_tokens", 2)))
    operator = BoundOperator(
        OperatorTemplate("draft_heads", "draft", "draft"), signature)
    compiled = _compiled(operator, "mlp")
    work = family_work(compiled, WorkloadGeometry.homogeneous(1, 4, 16))
    operands = {operand.name: operand for operand in work.operands}
    assert operands["hidden"].direction == "in"
    assert operands["draft_hidden"].direction == "out"
    assert work.declaration_complete is True


# 23. Router permutation/finalize expose dense, packed, index and prob values.

def test_router_permute_and_finalize_typed_values():
    extras = (("router_hidden", 64), ("router_dtype_bytes", 2),
              ("router_indices_bytes", 4), ("router_prob_bytes", 2),
              ("router_source_dtype", "bf16"))
    for name, role, expected in (
            ("router_permute", "permute", {"hidden", "indices", "packed"}),
            ("router_finalize", "finalize",
             {"packed", "indices", "probs", "hidden"})):
        signature = StaticSignature(
            role=role, matrix=(("elements", 8),),
            experts=(("E", 8), ("k", 2)),
            tiles=(("mt", 16), ("cores", 1)), extras=extras)
        operator = BoundOperator(
            OperatorTemplate(name, "router", role), signature)
        work = family_work(_compiled(operator, "routed_experts"),
                           WorkloadGeometry.homogeneous(1, 4, 16))
        assert {operand.name for operand in work.operands} == expected
        assert work.declaration_complete is True


# 24. A selected T09 transport plan is retained as a separate wire ledger.

def test_family_work_retains_t09_transport_plan():
    from dataclasses import replace as _replace

    from mindie_sim.prediction.communication import MessageFlow
    from mindie_sim.prediction.topology import Topology
    from mindie_sim.schema.hardware import Hardware, Link
    from mindie_sim.schema.parallel import ParallelSpec

    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend910B1", "num_devices": 2,
        "ranks_per_node": 2,
    })
    topology = Topology(_replace(machine, links=(
        Link("fabric", 60000, (0, 1), 0.0, "fabric", "shared"),)),
        ParallelSpec(tp=2))
    plan = topology.transport_plan([MessageFlow(0, 1, 128)],
                                   chunk_bytes=64)
    work = family_work(
        _compiled(_bound(_model(), "mlp", "gate_up"), "mlp"), _GEOMETRY,
        transport_plan=plan)
    assert work.transport_plan is plan
    assert work.transport_plan_work
    assert sum(amount for _name, amount in work.transport_plan_work) > 0


# 28. RMSNorm keeps activation and gamma as separate values, issued read 96 B.

def test_rmsnorm_activation_and_gamma_are_separate_values():
    signature = StaticSignature(
        role="rmsnorm",
        matrix=(("elements", 16),),
        quant=(("act_bytes", 2.0), ("act_format", "bf16")),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("ops_per_element", 5), ("read_bytes", 96.0),
                ("write_bytes", 32.0), ("gamma_bytes", 64.0),
                ("gamma_format", "fp32")))
    operator = BoundOperator(
        OperatorTemplate("rmsnorm", "elementwise", "norm"), signature)
    work = family_work(_compiled(operator, "layernorm"),
                       WorkloadGeometry.homogeneous(1, 1, 2))
    operands = {operand.name: operand for operand in work.operands}
    assert operands["activation"].dtype == "bf16"
    assert operands["activation"].payload_bytes() == 32
    assert operands["gamma"].dtype == "fp32"
    assert operands["gamma"].payload_bytes() == 64
    assert work.pipe_work("mte2") == 96
    assert work.declaration_complete is True


# 29. Equal-size all-reduce still publishes a post-call output value.

def test_collective_inplace_publishes_distinct_output_value():
    signature = StaticSignature(
        role="allreduce",
        quant=(("wire_bytes", 128.0),),
        tiles=(("mt", 16),),
        extras=(("members", 2), ("topology", "hccs"),
                ("collective", "all_reduce")))
    operator = BoundOperator(
        OperatorTemplate("allreduce", "collective", "allreduce"), signature)
    work = family_work(_compiled(operator, ""),
                       WorkloadGeometry.homogeneous(1, 1, 16))
    values = dict(work.tensor_values())
    assert set(values) == {"payload", "payload_out"}
    assert values["payload"] != values["payload_out"]
    assert values["payload_out"].output == 0
    assert work.declaration_complete is True


# 30. Occurrence fields all participate in output producer identity.

def test_occurrence_request_batch_stream_change_output_identity():
    operator = _gemm_op(16, 16, act_bytes=2.0, output_format="bf16")
    compiled = _compiled(operator, "")
    geometry = WorkloadGeometry.homogeneous(1, 2, 3)
    base = Occurrence(kind="kernel", name="mm", invocation="i",
                      request_ids=("r1",), batch_ids=("b1",), stream="s1")
    other = replace(base, request_ids=("r2",), batch_ids=("b2",),
                    stream="s2")
    one = dict(family_work(compiled, geometry,
                           occurrence=base).tensor_values())
    two = dict(family_work(compiled, geometry,
                           occurrence=other).tensor_values())
    assert one["output"] != two["output"]
    assert one["activation"] == two["activation"]


# 31. Persistent state survives a new invocation on the same request/slot.

def test_same_state_slot_crosses_invocations_with_one_conductor():
    from mindie_sim.prediction.scheduling import FragmentConductor

    instance = _instance(_toy_window_model())
    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    store = _compiled(_compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention")
    store.layer = 0
    read = _compiled(_compiled_in_layer(instance, 1, None, "kv_read"), "kv_cache")
    read.layer = 0
    store_occ = Occurrence(kind="kernel", name="kv", invocation="step-1",
                           request_ids=("req-1",), rank=0)
    read_occ = Occurrence(kind="kernel", name="kv", invocation="step-2",
                          request_ids=("req-1",), rank=0)
    stored = family_work(store, geometry, occurrence=store_occ,
                         state_slot="cache0")
    loaded = family_work(read, geometry, occurrence=read_occ,
                         state=StateView().mark_kv(0), state_slot="cache0",
                         state_versions={"kv_cache": 1})
    assert stored.state_effects[0].target == loaded.state_effects[0].target
    assert "inv:step-1" not in stored.state_effects[0].target
    access = stored.state_accesses()[0]
    assert access["direction"] == "write"
    assert access["storage"] == "kv_cache"
    assert access["span"] == ((0, 4),)
    assert access["bytes"] == 512.0
    assert access["request"] == "req-1"

    engine = Engine([Resource("hbm", 1000.0, unit="bytes"),
                     Resource("mte2", 1000.0, unit="bytes"),
                     Resource("mte3", 1000.0, unit="bytes"),
                     Resource("link", 1000.0, unit="bytes")])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    store_program = lower_family_work(stored)
    store_end = store_program.start(engine, "store", ready,
                                    conductor=conductor)
    conductor.advance(until=(store_end,))
    read_program = lower_family_work(loaded)
    read_end = read_program.start(engine, "read", store_end,
                                  conductor=conductor)
    conductor.advance(until=(read_end,))
    target = loaded.state_effects[0].target
    assert conductor.state.live_versions(target) == (1,)
    assert conductor.state.committed_version(target) is None

    other_request = replace(read_occ, request_ids=("req-2",))
    isolated = family_work(read, geometry, occurrence=other_request,
                            state=StateView().mark_kv(0), state_slot="cache0",
                            state_versions={"kv_cache": 1})
    assert isolated.state_effects[0].target != target


# 32. Compression ratio is an attribute; its write has no fake boolean condition.

def test_compression_ratio_is_attr_not_state_condition():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_compress_fixture", "tests/test_piecewise_operators_attention_mla.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model, instantiated = fixture._a2_model()
    compiled = fixture.compile_instance(
        instantiated.find("kv_compressor", layer=0, role="compress"))
    work = family_work(compiled.operator("compress"),
                       WorkloadGeometry.homogeneous(1, 5, 5),
                       state=StateView())
    attrs = {attr.name: attr.value for attr in work.attrs}
    assert attrs["ratio"] == 4
    assert work.state_effects
    assert work.state_effects[0].action == "write"
    assert work.state_effects[0].condition == ""


# 33. The two public submission entries share conditions/holds/event bindings.

def test_fragment_start_passes_conditions_to_conductor():
    from mindie_sim.schema.operators import StateEffect
    from mindie_sim.prediction.scheduling import FragmentConductor

    instance = _instance(_toy_window_model())
    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    store = _compiled(_compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention")
    store.layer = 0
    occurrence = Occurrence(kind="kernel", name="kv", invocation="step-1",
                            request_ids=("req-1",), rank=0)
    work = family_work(store, geometry, occurrence=occurrence,
                       state_slot="cache0")
    fragment = work.fragments()[0]
    target = work.state_effects[0].target
    conditional = replace(fragment, state_effects=(
        StateEffect(target, "write", condition="emit"),))
    program = FamilyFragmentProgram((conditional,))
    for enabled in (False, True):
        engine = Engine([Resource("hbm", 1000.0, unit="bytes"),
                         Resource("mte2", 1000.0, unit="bytes"),
                         Resource("mte3", 1000.0, unit="bytes"),
                         Resource("link", 1000.0, unit="bytes")])
        conductor = FragmentConductor(engine)
        ready = engine.submit("ready")
        done = program.start(engine, "conditional", ready,
                             conductor=conductor,
                             conditions={"emit": enabled})
        conductor.advance(until=(done,))
        assert conductor.state.live_versions(target) == ((1,) if enabled else ())


# 34. submit_family_work shares the same condition/hold submission path.

def test_submit_family_work_passes_conditions():
    from mindie_sim.prediction.scheduling import FragmentConductor
    from mindie_sim.prediction.piecewise.families import submit_family_work

    instance = _instance(_toy_window_model())
    geometry = WorkloadGeometry.homogeneous(1, 4, 16)
    store = _compiled(_compiled_in_layer(instance, 1, "csa_attention", "kv_store"), "csa_attention")
    store.layer = 0
    occurrence = Occurrence(kind="kernel", name="kv", invocation="step-1",
                            request_ids=("req-1",), rank=0)
    work = family_work(store, geometry, occurrence=occurrence,
                       state_slot="cache0")
    target = work.state_effects[0].target
    conditional_effect = replace(
        work.state_effects[0], condition="emit")
    stage = replace(work.stages[0],
                    state_effects=(conditional_effect,))
    conditional_work = replace(work, stages=(stage,),
                               state_effects=(conditional_effect,))
    engine = Engine([Resource("hbm", 1000.0, unit="bytes"),
                     Resource("mte2", 1000.0, unit="bytes"),
                     Resource("mte3", 1000.0, unit="bytes"),
                     Resource("link", 1000.0, unit="bytes")])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    handle = submit_family_work(conductor, conditional_work,
                                after=(ready,),
                                conditions={"emit": True})
    conductor.advance(until=(handle.event,))
    assert conductor.state.live_versions(target) == (1,)


# 35. L2Norm masks M/N and separates issued reduction lanes from logical math.

def test_l2norm_masks_m_n_and_fp32_destination():
    signature = StaticSignature(
        role="k_l2norm",
        matrix=(("elements", 64),),
        quant=(("act_bytes", 2.0), ("act_format", "bf16")),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("ops_per_element", 5), ("BD", 64), ("MBLOCK", 32),
                ("M", 8), ("N", 64), ("output_dtype", "fp32"),
                ("source_kernel", "l2norm_fwd_kernel2")))
    operator = BoundOperator(
        OperatorTemplate("k_l2norm", "elementwise", "norm"), signature)
    work = family_work(_compiled(operator, "layernorm"),
                       WorkloadGeometry.homogeneous(1, 8, 8))
    derived = dict(work.derived)
    assert work.pipe_work("mte2") == 8 * 64 * 2
    assert work.pipe_work("mte3") == 8 * 64 * 4
    assert derived["l2_logical_math"] == 8 * 64 * 5
    assert derived["l2_issued_math"] == 32 * 64 * 5
    operand = work.operands[0]
    assert operand.shape() == (8, 64)
    assert operand.dtype == "bf16"
    assert operand.payload_bytes() == 8 * 64 * 2
    output = [item for item in work.operands
              if item.direction == "out"][0]
    assert output.dtype == "fp32"
    assert output.payload_bytes() == 8 * 64 * 4


# 36. RoPE keeps qk/cos/sin dtypes separate and the exact per-token access.

def test_rope_keeps_per_value_cos_sin_dtype():
    signature = StaticSignature(
        role="index_rope",
        matrix=(("elements", 128),),
        quant=(("act_bytes", 2.0), ("act_format", "bf16")),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("qk_rope_head_dim", 64), ("index_heads", 1),
                ("rope_dtype", "bf16"), ("cos_dtype", "fp32"),
                ("sin_dtype", "fp32"), ("is_neox_style", 1)))
    operator = BoundOperator(
        OperatorTemplate("index_rope", "elementwise", "rope"), signature)
    work = family_work(_compiled(operator, ""),
                       WorkloadGeometry.homogeneous(1, 1, 2))
    operands = {operand.name: operand for operand in work.operands}
    assert set(operands) == {"qk", "cos", "sin", "output"}
    assert operands["qk"].dtype == "bf16"
    assert operands["qk"].payload_bytes() == 128 * 2
    assert operands["cos"].dtype == "fp32"
    assert operands["cos"].shape() == (1, 32)
    assert operands["cos"].payload_bytes() == 32 * 4
    assert operands["sin"].dtype == "fp32"
    assert operands["output"].dtype == "bf16"
    assert work.pipe_work("mte2") == 384
    assert work.pipe_work("mte3") == 128
    assert dict(work.derived)["rope_math_ops"] == 192


# 37. Norm's three logical outputs are counted by one stage write, not twice.

def test_norm_three_outputs_are_counted_once():
    signature = StaticSignature(
        role="norm",
        matrix=(("elements", 16),),
        quant=(("act_bytes", 2.0), ("act_format", "bf16")),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("ops_per_element", 5), ("read_bytes", 96.0),
                ("write_bytes", 68.0), ("gamma_bytes", 64.0),
                ("gamma_format", "fp32"), ("rstd_bytes", 4.0),
                ("residual_out_bytes", 32.0), ("residual_add", 1)))
    operator = BoundOperator(
        OperatorTemplate("norm", "elementwise", "norm"), signature)
    work = family_work(_compiled(operator, "layernorm"),
                       WorkloadGeometry.homogeneous(1, 1, 2))
    outputs = [operand for operand in work.operands
               if operand.direction == "out"]
    values = {operand.name: operand.payload_bytes() for operand in outputs}
    assert values == {"norm_output": Fraction(32), "residual": Fraction(32),
                      "rstd": Fraction(4)}
    assert sum(values.values()) == 68
    assert work.pipe_work("mte3") == 96
    assert sum(amount for _name, amount in work.materialization) == 68


# 38. RMSNorm uses the evaluated row domain; two rows keep 64+64+64 B.

def test_rmsnorm_two_rows_uses_evaluated_geometry():
    signature = StaticSignature(
        role="rmsnorm", matrix=(("elements", 16),),
        quant=(("act_bytes", 2.0), ("act_format", "bf16")),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("ops_per_element", 5), ("read_bytes", 96.0),
                ("write_bytes", 32.0), ("gamma_bytes", 64.0),
                ("gamma_format", "fp32")))
    operator = BoundOperator(
        OperatorTemplate("rmsnorm", "elementwise", "norm"), signature)
    work = family_work(_compiled(operator, "layernorm"),
                       WorkloadGeometry.homogeneous(1, 2, 2))
    operands = {operand.name: operand for operand in work.operands}
    assert operands["activation"].shape() == (2, 16)
    assert operands["activation"].payload_bytes() == 64
    assert operands["gamma"].shape() == (16,)
    assert operands["gamma"].payload_bytes() == 64
    assert [item.payload_bytes() for item in work.operands
            if item.direction == "out"] == [64]
    assert work.pipe_work("mte2") == 128


# 39. Only an explicit in-place collective aliases the actual upstream storage.

def test_collective_alias_requires_selected_inplace_implementation():
    def collective(inplace: int):
        signature = StaticSignature(
            role="allreduce",
            quant=(("wire_bytes", 128.0),),
            tiles=(("mt", 16),),
            extras=(("members", 2), ("topology", "hccs"),
                    ("collective", "all_reduce"), ("inplace", inplace)))
        return BoundOperator(
            OperatorTemplate("allreduce", "collective", "allreduce"),
            signature)

    upstream = TensorValue(
        "upstream", "uint8", TensorLayout.contiguous((128,)),
        storage="shared-buffer")
    inplace = family_work(_compiled(collective(1), ""),
                          WorkloadGeometry.homogeneous(1, 1, 16))
    values = dict(inplace.tensor_values(input_values={"payload": upstream}))
    assert values["payload_out"].producer != values["payload"].producer
    assert values["payload_out"].storage == "shared-buffer"
    assert values["payload_out"].alias_of == upstream.producer

    distinct = family_work(_compiled(collective(0), ""),
                           WorkloadGeometry.homogeneous(1, 1, 16))
    out = dict(distinct.tensor_values())["payload_out"]
    assert out.alias_of == ""
    assert out.storage != "shared-buffer"


# 40. GQA K/V logical values follow actual KV-head count, not a fixed divisor.

def test_gqa_kv_values_follow_actual_kv_heads():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_gqa_heads", "tests/test_piecewise_operators_attention_mla.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    for kv_heads, expected_each in ((1, 2176), (2, 4352), (4, 8704)):
        compiled = fixture.compile_attention(8, kv_heads, 64)
        work = family_work(compiled.operator("attention"),
                           fixture.decode([(1, 16)]))
        values = dict(work.tensor_values())
        assert values["k_cache"].dtype == "bf16"
        assert values["k_cache"].layout.shape == (17, kv_heads, 64)
        assert values["v_cache"].layout.shape == (17, kv_heads, 64)
        assert math.prod(values["k_cache"].layout.shape) * 2 == expected_each
        assert math.prod(values["v_cache"].layout.shape) * 2 == expected_each


# 41. State accesses are per request and preserve the actual view offset.

def test_state_accesses_split_requests_and_preserve_view_offset():
    from mindie_sim.schema.tensors import TensorLayout

    def store(rows: int):
        signature = StaticSignature(
            role="kv_store", matrix=(("rows", rows), ("row_bytes", 128.0)),
            extras=(("owner_layer", 0), ("rows", rows),
                    ("valid_rows_mode", "explicit")))
        return _compiled(BoundOperator(
            OperatorTemplate("kv_store", "kv_access", "consume"), signature),
            "csa_attention")

    two = family_work(
        store(2), WorkloadGeometry.homogeneous(1, 2, 16),
        occurrence=Occurrence(kind="kernel", name="kv",
                              request_ids=("r1", "r2"), rank=0),
        state_slot="cache0")
    accesses = two.state_accesses()
    assert [(item["request"], item["span"], item["bytes"]) for item in accesses] == [
        ("r1", ((0, 1),), 128.0), ("r2", ((1, 2),), 128.0)]
    assert accesses[0]["producer"] != accesses[1]["producer"]

    layout = TensorLayout((4, 128), (128, 1), offset=12 * 128,
                          storage_shape=(16, 128))
    view = family_work(
        store(4), WorkloadGeometry.homogeneous(1, 4, 16),
        layouts={"kv_cache_row": layout})
    access = view.state_accesses()[0]
    assert access["span"] == ((12, 16),)
    assert access["bytes"] == 512.0


# 42. commit/rollback are decisions, not extra physical state writes.

def test_commit_is_not_projected_as_physical_state_write():
    signature = StaticSignature(
        role="kv_store", matrix=(("rows", 1), ("row_bytes", 128.0)),
        extras=(("owner_layer", 0), ("valid_rows_mode", "explicit")))
    operator = BoundOperator(
        OperatorTemplate("kv_store", "kv_access", "consume"), signature)
    work = family_work(
        _compiled(operator, "csa_attention"), WorkloadGeometry.homogeneous(1, 1, 16),
        state_actions={"kv_cache": "commit"},
        state_versions={"kv_cache": 1})
    assert work.state_effects[0].action == "commit"
    assert work.state_accesses() == ()


# 43. GQA q5/c16 keeps 21 logical positions across both mt settings.

def test_gqa_q5_context16_both_mt_keep_21_positions():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_gqa_q5", "tests/test_piecewise_operators_attention_mla.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    for mt in (16, 32):
        compiled = fixture.compile_attention(8, 2, 64, mt=mt)
        work = family_work(compiled.operator("attention"),
                           fixture.decode([(5, 16)]))
        values = dict(work.tensor_values())
        assert values["k_cache"].layout.shape == (21, 2, 64)
        assert values["v_cache"].layout.shape == (21, 2, 64)
        assert math.prod(values["k_cache"].layout.shape) * 2 == 5376
        assert math.prod(values["v_cache"].layout.shape) * 2 == 5376
        assert work.pipe_work("hbm") > 0


# 44. In-place all-reduce inherits the actual offset/storage, with new output.

def test_inplace_allreduce_inherits_actual_view():
    signature = StaticSignature(
        role="allreduce", quant=(("wire_bytes", 128.0),),
        tiles=(("mt", 16),),
        extras=(("members", 2), ("topology", "hccs"),
                ("collective", "all_reduce"), ("inplace", 1)))
    work = family_work(_compiled(BoundOperator(
        OperatorTemplate("allreduce", "collective", "allreduce"), signature),
        ""), WorkloadGeometry.homogeneous(1, 1, 16))
    upstream = TensorValue(
        "upstream", "uint8", TensorLayout((128,), (1,), offset=128,
                                          storage_shape=(256,)),
        storage="shared-buffer")
    values = dict(work.tensor_values(input_values={"payload": upstream}))
    out = values["payload_out"]
    assert out.producer != upstream.producer
    assert out.storage == "shared-buffer"
    assert out.alias_of == upstream.producer
    assert out.layout.offset == 128
    assert out.layout.storage_shape == (256,)


# 45. RoPE T32/H1/D128/R64 keeps full output and cos/sin values separate.

def test_rope_t32_keeps_full_output_and_angle_bytes():
    signature = StaticSignature(
        role="index_rope", matrix=(("elements", 128),),
        quant=(("act_bytes", 2.0), ("act_format", "bf16")),
        tiles=(("mt", 16), ("cores", 2)),
        extras=(("qk_rope_head_dim", 64), ("index_heads", 1),
                ("rope_dtype", "bf16"), ("cos_dtype", "fp32"),
                ("sin_dtype", "fp32"), ("is_neox_style", 1)))
    work = family_work(_compiled(BoundOperator(
        OperatorTemplate("index_rope", "elementwise", "rope"), signature), ""),
        WorkloadGeometry.homogeneous(1, 32, 32))
    values = {operand.name: operand for operand in work.operands}
    assert values["output"].payload_bytes() == 8192
    assert values["cos"].payload_bytes() == 4096
    assert values["sin"].payload_bytes() == 4096
    assert dict(work.derived)["rope_math_ops"] == 6144
    assert work.pipe_work("mte3") == 4096


# 46. Two q1 index children expose per-request K/scale work and 160 B total.

def test_index_cache_two_requests_per_child_work():
    signature = StaticSignature(
        role="index_cache_store",
        matrix=(("rows", 2), ("row_bytes", 80.0)),
        extras=(("owner_layer", 0), ("scale_bytes", 16.0),
                ("valid_rows_mode", "explicit"), ("index_dtype", "int32")))
    work = family_work(
        _compiled(BoundOperator(
            OperatorTemplate("index_cache_store", "kv_access", "produce"),
            signature), ""),
        WorkloadGeometry((RequestGeometry(1, 0), RequestGeometry(1, 0))),
        occurrence=Occurrence(kind="kernel", name="index",
                              request_ids=("a", "b"), rank=0))
    assert len(work.stages) == 2
    assert dict(work.stages[0].request_work)["mte3"] == (
        ("a", Fraction(64)), ("b", Fraction(64)))
    assert dict(work.stages[1].request_work)["mte3"] == (
        ("a", Fraction(16)), ("b", Fraction(16)))
    assert work.pipe_work("mte3") == 160


# 47. Double ScatterPaCache children expose real read/write totals once.

def test_scatter_pa_children_child_reads_and_totals():
    signature = StaticSignature(
        role="index_cache_store",
        matrix=(("rows", 1024), ("row_bytes", 132.0)),
        extras=(("owner_layer", 0), ("scale_bytes", 4.0),
                ("valid_rows_mode", "explicit"), ("index_dtype", "int32")))
    work = family_work(_compiled(BoundOperator(
        OperatorTemplate("index_cache_store", "kv_access", "produce"),
        signature), ""), WorkloadGeometry.homogeneous(1, 1024, 1024))
    assert dict(work.stages[0].exact_work) == {
        "hbm": Fraction(266240), "mte2": Fraction(135168),
        "mte3": Fraction(131072)}
    assert dict(work.stages[1].exact_work) == {
        "hbm": Fraction(12288), "mte2": Fraction(8192),
        "mte3": Fraction(4096)}
    assert work.pipe_work("mte2") == 143360
    assert work.pipe_work("mte3") == 135168
    assert work.pipe_work("hbm") == 278528


# 48. Stateful KV exposes stage HBM cap 250; actual T04 cap consumption pending.

def test_stateful_kv_preserves_executed_hbm_cap():
    signature = StaticSignature(
        role="kv_store", matrix=(("rows", 1), ("row_bytes", 512.0)),
        extras=(("owner_layer", 0), ("valid_rows_mode", "explicit")))
    work = family_work(
        _compiled(BoundOperator(
            OperatorTemplate("kv_store", "kv_access", "consume"), signature),
            "csa_attention"), WorkloadGeometry.homogeneous(1, 1, 16),
        executed_caps={"hbm": 250})
    assert dict(work.caps)["hbm"] == 250
    assert dict(work.stages[0].caps)["hbm"] == 250
    program = lower_family_work(work)
    assert dict(program.caps)["hbm"] == 250
    assert work.pipe_work("hbm") == 512


# 49. Request rows and reordering preserve each request's KV write ownership.

def test_kv_store_request_rows_and_reordering():
    def store(rows: int):
        signature = StaticSignature(
            role="kv_store", matrix=(("rows", rows), ("row_bytes", 128.0)),
            extras=(("owner_layer", 0), ("rows", rows),
                    ("valid_rows_mode", "explicit")))
        return _compiled(BoundOperator(
            OperatorTemplate("kv_store", "kv_access", "consume"), signature),
            "csa_attention")

    geometry = WorkloadGeometry((RequestGeometry(1, 0),
                                 RequestGeometry(2, 0)))
    occurrence = Occurrence(kind="kernel", name="kv",
                            request_ids=("r1", "r2"), rank=0)
    work = family_work(store(3), geometry, occurrence=occurrence,
                       state_slot="c")
    accesses = work.state_accesses()
    assert [(item["request"], item["span"], item["bytes"])
            for item in accesses] == [
        ("r1", ((0, 1),), 128.0), ("r2", ((1, 3),), 256.0)]
    assert dict(work.stages[0].request_work)["hbm"] == (
        ("r1", Fraction(128)), ("r2", Fraction(256)))

    reversed_geometry = WorkloadGeometry((
        RequestGeometry(2, 0), RequestGeometry(1, 0)))
    reversed_work = family_work(
        store(3), reversed_geometry,
        occurrence=Occurrence(kind="kernel", name="kv",
                              request_ids=("r2", "r1"), rank=0),
        state_slot="c")
    reversed_accesses = reversed_work.state_accesses()
    assert [(item["request"], item["bytes"], item["span"])
            for item in reversed_accesses] == [
        ("r2", 256.0, ((0, 2),)), ("r1", 128.0, ((2, 3),))]


# 50. Scatter version 9 is invisible until the real writer completes.

def test_scatter_version_visible_only_after_writer():
    from mindie_sim.prediction.scheduling import FragmentConductor
    from mindie_sim.schema.operators import StateEffect

    signature = StaticSignature(
        role="scatter_recurrent_state",
        heads=(("value_heads", 1), ("dk", 64), ("dv", 32)),
        extras=(("state_dtype", "fp32"), ("source_dtype", "fp16"),
                ("index_dtype", "int32")))
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        BoundOperator(
            OperatorTemplate("scatter_recurrent_state", "state_scatter",
                             "state"), signature))
    geometry = WorkloadGeometry(
        (RequestGeometry(1, 8, phase="prefill"),), phase="prefill")
    work = family_work(compiled, geometry, state_slot="ssm",
                       state_versions={"recurrent_state": 9})
    target = work.state_effects[0].target
    engine = Engine([Resource("hbm", 1000.0, unit="bytes"),
                     Resource("mte2", 1000.0, unit="bytes"),
                     Resource("mte3", 1000.0, unit="bytes"),
                     Resource("vector", 1000.0, unit="ops")])
    conductor = FragmentConductor(engine, state={target: 8})
    ready = engine.submit("ready")
    program = lower_family_work(work)
    assert conductor.state.live_versions(target) == (8,)
    assert conductor.state.committed_version(target) == 8
    with pytest.raises(Exception, match="state_version_unavailable"):
        conductor.state.check_read(target, 9, at=0.0)
    end = program.start(engine, "scatter", ready, conductor=conductor)
    conductor.advance(until=(end,))
    assert conductor.state.live_versions(target) == (8, 9)
    assert conductor.state.committed_version(target) == 8
    assert conductor.state.check_read(target, 9, at=end.time) == 9


# 51. Stateful 512 B KV completes at 2.048 ms under the selected 250 B/ms cap.

def test_stateful_kv_completes_at_capped_rate():
    from mindie_sim.prediction.scheduling import FragmentConductor

    signature = StaticSignature(
        role="kv_store", matrix=(("rows", 1), ("row_bytes", 512.0)),
        extras=(("owner_layer", 0), ("valid_rows_mode", "explicit")))
    work = family_work(
        _compiled(BoundOperator(
            OperatorTemplate("kv_store", "kv_access", "consume"), signature),
            "csa_attention"), WorkloadGeometry.homogeneous(1, 1, 16),
        executed_caps={"hbm": 250})
    engine = Engine([Resource("hbm", 250.0, unit="bytes"),
                     Resource("mte3", 1000.0, unit="bytes")])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready", release=0.0)
    end = lower_family_work(work).start(
        engine, "kv", ready, conductor=conductor)
    conductor.advance(until=(end,))
    assert end.time == pytest.approx(2.048, rel=1e-12)
    assert dict(work.caps)["hbm"] == 250


# 52. A normal CompiledSubgraph execution consumes the complete family fragment.

def test_compiled_subgraph_consumes_single_stage_family_fragment():
    import json
    from pathlib import Path

    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case

    instance = _instance(_model())
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile(
        instance.find("layernorm", layer=0))
    payload = json.loads(Path(
        "examples/cases/high-concurrency-20261030/"
        "glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json").read_text())
    case = parse_case(payload)
    engine = Engine([
        Resource("hbm", 1e9, unit="bytes"),
        Resource("mte2", 1e9, unit="bytes"),
        Resource("mte3", 1e9, unit="bytes"),
        Resource("vector", 1e9, unit="ops"),
    ])
    execution = Execution(case, CostModel(case), include_submission=False,
                          engine=engine)
    evaluation, submitted = compiled.submit(
        execution, WorkloadGeometry.homogeneous(1, 2, 2),
        ready=execution.engine.submit("ready", release=0.0))
    assert submitted and submitted[0].events
    execution.engine.advance(until=(submitted[-1].events[-1],))
    assert evaluation.total_ms > 0
    assert execution.engine.now > 0


# 53. Ordinary FIA selected plan executes through FamilyWork, not a work tuple.

def test_native_fia_plan_executes_through_family_bridge():
    from mindie_sim.prediction.attention_decode import StagedGQADecode
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.program import RecipeContext
    from mindie_sim.schema.operators import OpSpec

    requests = ((1, 127), (1, 130))
    op = OpSpec(
        op_id="fia", family="attention", phase="decode", stage=0, layer=0,
        requests=("r0", "r1"),
        semantic_shape=(2, 1, 4, 128), local_shape=(2, 1, 4, 128),
        padded_shape=(2, 1, 4, 128), launch_shape=(2, 1, 4, 128),
        metadata=(
            ("gqa_decode", {
                "requests": requests, "query_heads": 4, "kv_heads": 4,
                "head_dim": 128, "query_dtype": "bf16",
                "kv_dtype": "bf16", "causal": True, "window": 0, "cp": 1}),
            ("attention_tensor_view", {
                "kv_storage": "paged", "local_kv_heads": 4,
                "cache_row_width": 1024, "kv_block_size": 128})))
    plan = StagedGQADecode().build(
        op, Geometry(compute_units=1, resource_units={
            "mte2": 1, "mte3": 1, "mte1": 1, "vector": 1}))
    roles = ["hbm", "mte2", "mte3", "mte1", "vector", "cube"]
    context = RecipeContext(
        op=op, work={}, resource_names={role: role for role in roles},
        rates={role: 1000 for role in roles}, core_count=1,
        resource_units={role: 1 for role in roles},
        compute_peaks={"bf16": 1000})

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_a5_fia_fixture", "tests/test_piecewise_operators_attention_mla.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    compiled = fixture.compile_attention(4, 4, 128)
    geometry = fixture.decode([(1, 127), (1, 130)])
    work = family_work(compiled.operator("attention"), geometry,
                       native_plan=plan, native_context=context)
    assert work.native_program is not None
    assert work.stages == ()
    assert dict(work.native_program.work) == {
        "hbm": plan.work.resources["hbm"],
        "mte1": plan.work.resources["mte1"],
        "mte2": plan.work.resources["mte2"],
        "mte3": plan.work.resources["mte3"],
        "vector": plan.work.resources["vector"],
        "cube": sum(plan.work.matrix.values())}
    assert lower_family_work(work) is work.native_program
    operands = {value.name: value for value in work.operands}
    assert set(operands) >= {"q", "k_cache", "v_cache", "output"}
    assert operands["block_table"].dtype == "int32"
    assert operands["block_table"].shape() == (2, 2)
    assert operands["block_table"].payload_bytes() == 16
    assert any(port.name == "block_table"
               for port in work.ports.tensor_in)

    resources = [Resource(role, 1000.0,
                          unit=("ops" if role == "vector"
                                else "flops" if role == "cube" else "bytes"))
                 for role in roles]
    engine = Engine(resources)
    ready = engine.submit("ready", release=0.0)
    end = work.native_program.start(engine, "fia", ready)
    engine.advance(until=(end,))
    assert engine.now == pytest.approx(1066.16, rel=1e-12)
    direct = Engine(resources)
    direct_end = plan.program(context).start(
        direct, "fia", direct.submit("ready", release=0.0))
    direct.advance(until=(direct_end,))
    assert direct.now == pytest.approx(engine.now, rel=1e-12)


def _router_inbound_compiled(received=6, counts=None):
    counts = counts or (6,) + (0,) * 15
    comm = (("mode", "dynamic"), ("policy", ""), ("capacity", 6),
            ("world_size", 2), ("tp", 1), ("top_k", 2),
            ("experts_per_device", 16))
    signature = StaticSignature(
        role="router_inbound", matrix=(("elements", 8),),
        experts=(("E", 16), ("k", 2)),
        tiles=(("mt", 16), ("cores", 1)),
        extras=(("router_received_rows", received),
                ("router_hidden", 64), ("router_payload_bytes", 0.5),
                ("router_scale_cols", 2), ("router_index_bytes", 4),
                ("router_source_dtype", "mxfp4"),
                ("router_expert_token_counts", counts),
                ("router_local_expert_ids", (0,) * int(sum(counts))),
                ("moe_comm", comm)))
    return _compiled(BoundOperator(OperatorTemplate(
        "router_inbound", "router", "router_inbound"), signature), "routed_experts")


# 54. Known received rows evaluate on both adjacent inbound branches.

def test_router_inbound_known_received_evaluates_and_consumes_leaf_plan():
    compiled = _router_inbound_compiled()
    geometry = WorkloadGeometry.homogeneous(1, 6, 16)
    geometry = type(geometry)(geometry.requests, phase="prefill")
    occurrence = Occurrence(kind="kernel", name="inbound", rank=0)
    work = family_work(compiled, geometry, occurrence=occurrence)
    assert work.active
    assert work.stages == ()
    assert work.native_program is not None
    assert work.native_program.program.fragment.occurrence == occurrence
    assert work.native_program.program.service_work
    assert {port.name for port in work.ports.tensor_out} >= {
        "expanded_x", "expanded_row_idx",
        "expert_tokens_count_or_cumsum", "expanded_scale"}
    assert {port.name for port in work.ports.event_out} >= {
        "inbound:rank0.expert_tokens_count_or_cumsum.complete",
        "inbound:rank0.expanded_row_idx.complete"}
    other = family_work(
        compiled, geometry,
        occurrence=Occurrence(kind="kernel", name="inbound", rank=1))
    assert other.native_program.program.fragment.occurrence.rank == 1
    assert any("rank1" in port.name for port in other.ports.event_out)
    assert all("rank1" not in port.name for port in work.ports.event_out)
    # The leaf's count/inverse publication stages are real and precede a
    # consumer that is bound to both named events.
    program = work.native_program.program
    engine = Engine([
        Resource("hbm", 1000.0, unit="bytes"),
        Resource("mte2", 1000.0, unit="bytes"),
        Resource("mte3", 1000.0, unit="bytes"),
        Resource("vector", 1000.0, unit="ops"),
    ])
    ready = engine.submit("ready", release=0.0)
    complete, stages = program.start_events(engine, "inbound", ready)
    consumer = engine.submit(
        "consumer", after=(stages["publish_count"],
                           stages["publish_row_idx"]))
    engine.advance(until=(complete,))
    assert consumer.started >= max(stages["publish_count"].time,
                                   stages["publish_row_idx"].time)
    assert consumer.started < complete.time
    assert engine.now == pytest.approx(complete.time)

    # The same plan through the common conductor publishes named count and
    # inverse events; a Compose consumer binds those ports by name.
    from mindie_sim.prediction.scheduling import FragmentConductor
    from mindie_sim.prediction.piecewise.native_moe_token_permute import (
        execute_staged,
    )
    from mindie_sim.schema.operators import ProgramFragment, WorkSet
    from mindie_sim.schema.subgraphs import EventPortSpec, PortSet

    conducted = Engine([
        Resource("hbm", 1000.0, unit="bytes"),
        Resource("mte2", 1000.0, unit="bytes"),
        Resource("mte3", 1000.0, unit="bytes"),
        Resource("vector", 1000.0, unit="ops"),
    ])
    conductor = FragmentConductor(conducted)
    admitted = conducted.submit("ready", release=0.0)
    staged = execute_staged(
        conductor, program.plan, ready=admitted,
        inputs={"x": admitted, "expert_idx": admitted, "scale": admitted})
    names = {port.name for port in work.ports.event_out
             if "count" in port.name or "row_idx" in port.name}
    assert set(staged["outputs"]) >= names
    consumer_fragment = ProgramFragment(
        name="count_inverse_consumer",
        occurrence=Occurrence("stage", "count_inverse_consumer"),
        ports=PortSet(event_in=tuple(
            EventPortSpec(name, "in", "complete", scope="operator")
            for name in sorted(names))),
        work=WorkSet())
    consumer_handle = conductor.submit(
        consumer_fragment,
        inputs={name: staged["outputs"][name] for name in names})
    conductor.advance(until=(consumer_handle.event,))
    assert consumer_handle.event.started >= max(
        staged["outputs"][name].time for name in names)
    assert len(staged["stage_handles"]) == len(program.plan["stages"])


# 55. Missing received rows stay unknown and do not become M*k.

def test_router_inbound_unknown_received_stays_unpriced():
    compiled = _router_inbound_compiled(received=None,
                                        counts=(6,) + (0,) * 15)
    # A missing ledger with declared counts still has a known route row count;
    # omit both to retain the unknown-row branch.
    compiled = _router_inbound_compiled(received=None, counts=())
    geometry = WorkloadGeometry.homogeneous(1, 6, 16)
    geometry = type(geometry)(geometry.requests, phase="prefill")
    work = family_work(compiled, geometry)
    assert work.active
    assert work.native_program is None
    assert work.issued_work() == {}
    assert int(dict(work.derived)["router_local_tokens"]) == 0
    assert int(dict(work.derived)["router_received_known"]) == 0


# 56. The adjacent unweighted-reorder branch evaluates with known rows.

def test_router_unweighted_known_received_evaluates():
    signature = StaticSignature(
        role="router_unweighted_reorder", matrix=(("elements", 8),),
        experts=(("E", 16), ("k", 2)),
        tiles=(("mt", 16), ("cores", 1)),
        extras=(("router_received_rows", 6), ("router_hidden", 64)))
    compiled = _compiled(BoundOperator(OperatorTemplate(
        "router_unweighted_reorder", "router", "router_unweighted_reorder"),
        signature), "routed_experts")
    geometry = WorkloadGeometry.homogeneous(1, 6, 16)
    geometry = type(geometry)(geometry.requests, phase="prefill")
    work = family_work(compiled, geometry)
    assert work.active
    assert int(dict(work.derived)["router_local_tokens"]) == 6
    assert int(dict(work.derived)["router_received_known"]) == 1
    assert work.issued_work()["mte2"] == 792
    assert work.issued_work()["mte3"] == 768


# 57. Outbound TokenPermute also consumes its selected NATIVE-MOE plan.

def test_router_permute_consumes_leaf_plan():
    signature = StaticSignature(
        role="router_permute", matrix=(("elements", 8),),
        experts=(("E", 16), ("k", 1)),
        tiles=(("mt", 16), ("cores", 1)),
        extras=(("router_hidden", 2), ("k", 1),
                ("router_source_dtype", "bf16")))
    compiled = _compiled(BoundOperator(OperatorTemplate(
        "router_permute", "router", "router_permute"), signature), "routed_experts")
    geometry = WorkloadGeometry.homogeneous(1, 2, 16)
    geometry = type(geometry)(geometry.requests, phase="prefill")
    work = family_work(compiled, geometry)
    assert work.active
    assert work.stages == ()
    assert work.native_program is not None
    assert set(work.native_program.program.service_work) == {
        "mte2", "mte3", "hbm", "vector"}
    assert {port.name for port in work.ports.tensor_out} >= {
        "permuted_tokens", "sorted_indices"}


# 58. Separate histogram consumes plan_histogram with its named output.

def test_router_histogram_consumes_leaf_plan():
    signature = StaticSignature(
        role="router_histogram", matrix=(("elements", 8),),
        experts=(("E", 16), ("k", 1)),
        tiles=(("mt", 16), ("cores", 1)),
        extras=(("router_hidden", 2), ("k", 1),
                ("router_source_dtype", "int32")))
    compiled = _compiled(BoundOperator(OperatorTemplate(
        "router_histogram", "router", "router_histogram"), signature), "moe_router")
    geometry = WorkloadGeometry.homogeneous(1, 2, 16)
    geometry = type(geometry)(geometry.requests, phase="prefill")
    work = family_work(compiled, geometry)
    assert work.active
    assert work.stages == ()
    assert work.native_program is not None
    assert {port.name for port in work.ports.tensor_out} == {"histogram"}
    assert {port.name for port in work.ports.event_out} == {
        "router_histogram.histogram.complete"}
    assert dict(work.native_program.work)["hbm"] == 72
