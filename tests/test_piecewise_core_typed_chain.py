"""Typed/executable chain invariants at the piecewise compile-core boundary.

[Role]
  Lock two compile-core losses that silently changed a prediction:

  1. ``lower_fusion`` derived the A0 Q/K/V byte split from an alias of the
     fusion members' materialisation.  When the QKV projection became an
     independent semantic operator (outside the qk_norm+rotary fusion group),
     ``value`` silently aliased ``rotary_output`` and reported 3x the typed V
     slice.  The declared ``qkv_slices`` contract applies to the producer's
     typed activation, not to an unrelated downstream name.
  2. A family program that computes elapsed time in floats was allowed to
     publish that float through ``OperatorCost.duration_ms``/``times``.  Exact
     serial composition then missed by one ulp.  The compiler now normalises
     the family -> cost boundary to exact ``Fraction`` regardless of whether
     hardware service metadata is present.

[Invariants]
  * Q/K/V bytes equal the declared element x token x activation-width
    contract, for plain and gated A0.
  * A declared QKV contract with no typed producer raises a replayable
    ``fusion_qkv_slices_producer_missing`` gap; it never falls back to an
    unrelated activation name.
  * Compiled cost timing fields, ``SubgraphEvaluation.total_ms`` and serial
    totals are exact rational values.
  * ``analysis.polynomial`` rejects a negative degree with ``ModelingError``,
    not an uncaught ``NameError``.

[Timeouts & Locks]
  Pure CPU arithmetic on repository fixtures; no device, network or weights.

[Side-Effect Policy]
  Read-only test code; constructs immutable records.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from mindie_sim.prediction.piecewise.analysis import polynomial
from mindie_sim.prediction.piecewise.composition import (
    evaluate_serial_split, lower_fusion,
)
from mindie_sim.prediction.piecewise.compiler import OperatorCost
from mindie_sim.prediction.piecewise.expressions import Var
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import ModelPipeline
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError
from tests.test_piecewise_subgraph_model_pipeline import (
    SUPPLY, dense_model, dense_plan, gemma4_model_spec, gemma_config,
    make_model, pipeline,
)


def _a0(pipe):
    return pipe.instantiated.find("gqa_attention", layer=0, role="attention")


def test_a0_fusion_value_slice_comes_from_producer_not_rotary_alias() -> None:
    """The independent QKV projection still types the fused kept V slice."""
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)  # 8 rows
    instance = _a0(pipe)
    costs = pipe._subgraph(instance).evaluate(geometry).operators
    fused = lower_fusion(instance, costs, "qkv_norm_rope", supply=SUPPLY)
    kept = dict(fused.kept)
    rotary = next(cost for cost in costs if cost.name == "rotary")
    rotary_bytes = dict(rotary.materialization)["rotary_output"]

    # q=4*16 elems, k=v=2*16 elems, 8 rows, 2 bytes.  The pre-fix fallback
    # aliased value to the 1536-byte rotary activation.
    q_bytes = 4 * 16 * 2 * 8
    kv_bytes = 2 * 16 * 2 * 8
    assert kept["query"] == q_bytes
    assert kept["key"] == kv_bytes
    assert kept["value"] == kv_bytes
    assert rotary_bytes != kept["value"]
    values = {value.name: value for value in fused.values}
    # The independent QBMM writes V; this fused launch writes only q/k.
    assert dict(fused.fused.materialization) == {
        "query": q_bytes, "key": kv_bytes}
    assert values["value"].logical_bytes == kv_bytes
    assert values["value"].written_bytes == 0
    assert values["query"].written_bytes == q_bytes
    assert values["key"].written_bytes == kv_bytes
    # Surviving V points into the actual projection storage, not offset 0.
    assert values["value"].layout.shape == (8, 2, 16)
    assert values["value"].layout.strides == (128, 16, 1)
    assert values["value"].layout.offset == 96
    serialized = {value["name"]: value for value in fused.to_dict()["values"]}
    assert serialized["value"]["value"]["layout"]["offset"] == 96
    assert serialized["value"]["value"]["layout"]["strides"] == (128, 16, 1)
    assert dict(fused.fused.derived)["kept_bytes"] == q_bytes + 2 * kv_bytes


def test_gated_a0_fusion_types_gate_without_using_the_rotary_alias() -> None:
    pipe = pipeline(make_model(plans=[dense_plan(0, q_output_gate=True)]))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = _a0(pipe)
    costs = pipe._subgraph(instance).evaluate(geometry).operators
    fused = lower_fusion(instance, costs, "qkv_norm_rope", supply=SUPPLY)
    kept = dict(fused.kept)
    q_bytes = 4 * 16 * 2 * 8
    gate_bytes = q_bytes
    kv_bytes = 2 * 16 * 2 * 8
    assert kept["gate"] == gate_bytes
    assert kept["query"] == q_bytes
    assert kept["value"] == kv_bytes
    values = {value.name: value for value in fused.values}
    # qkv_projection writes the gate and V slices; the fused postprocessor
    # writes only q/k.  Their logical values stay visible at full width.
    assert dict(fused.fused.materialization) == {
        "query": q_bytes, "key": kv_bytes}
    assert fused.fused.pipe_work("mte3") == q_bytes + kv_bytes
    for name, size in (("gate", gate_bytes), ("value", kv_bytes)):
        assert values[name].logical_bytes == size
        assert values[name].written_bytes == 0
    assert values["gate"].layout.shape == (8, 4, 16)
    assert values["gate"].layout.strides == (192, 32, 1)
    assert values["gate"].layout.offset == 16
    assert values["value"].layout.shape == (8, 2, 16)
    assert values["value"].layout.strides == (192, 16, 1)
    assert values["value"].layout.offset == 160
    assert dict(fused.fused.derived)["kept_bytes"] == (
        q_bytes + gate_bytes + 2 * kv_bytes)


def test_gemma_k_eq_v_has_full_logical_v_alias_and_zero_extra_write() -> None:
    pipe = pipeline(gemma4_model_spec(gemma_config()))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = pipe.instantiated.find("gqa_attention", layer=1, role="attention")
    costs = pipe._subgraph(instance).evaluate(geometry).operators
    fused = lower_fusion(instance, costs, "qkv_norm_rope", supply=pipe.supply)
    kept = dict(fused.kept)
    values = {value.name: value for value in fused.values}
    assert kept["key"] > 0
    assert kept["value"] == kept["key"]
    # The derived KV cache is not a second copy of the K=V storage.
    assert kept["kv"] == 0
    assert values["value"].role == "alias"
    # V aliases the projection's pre-RoPE K slice, not the fused post-RoPE K.
    assert values["value"].alias_of == "projection_k"
    assert values["value"].logical_bytes == kept["key"]
    assert values["value"].producer == "qkv_projection"
    assert values["value"].output == "qkv_activation"
    assert values["value"].version == "pre_rope"
    assert values["value"].written_bytes == 0
    assert values["projection_k"].role == "logical"
    assert values["projection_k"].producer == "qkv_projection"
    assert values["projection_k"].logical_bytes == kept["key"]
    assert values["projection_k"].version == "pre_rope"
    assert values["value"].layout.shape == (8, 2, 512)
    assert values["value"].layout.strides == (3072, 512, 1)
    assert values["value"].layout.offset == 2048
    assert values["projection_k"].layout.offset == 2048
    assert values["key"].layout.offset == 0
    assert values["key"].layout.strides == (1024, 512, 1)
    assert values["key"].layout.storage_shape == (8, 1024)
    assert values["key"].producer == "fused:qkv_norm_rope"
    assert values["key"].version == "post_rope"
    assert dict(fused.fused.materialization) == {
        "query": kept["query"], "key": kept["key"]}
    assert fused.fused.pipe_work("mte3") == kept["query"] + kept["key"]
    # Q + fused post-RoPE K + projection pre-RoPE K; V alias adds no bytes.
    assert dict(fused.fused.derived)["kept_bytes"] == (
        kept["query"] + 2 * kept["key"])
    serialized = {value["name"]: value for value in fused.to_dict()["values"]}
    assert serialized["value"]["alias_of"] == "projection_k"
    assert serialized["value"]["version"] == "pre_rope"
    assert serialized["key"]["version"] == "post_rope"


def test_missing_qkv_slices_reconstructs_from_selected_producer_theta() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = _a0(pipe)
    costs = pipe._subgraph(instance).evaluate(geometry).operators
    declared = lower_fusion(instance, costs, "qkv_norm_rope", supply=SUPPLY)
    no_contract = replace(
        instance, template=replace(instance.template, contract=()))
    recovered = lower_fusion(no_contract, costs, "qkv_norm_rope", supply=SUPPLY)
    assert dict(recovered.kept) == dict(declared.kept)
    assert [(value.name, value.logical_bytes, value.role, value.alias_of)
            for value in recovered.values] == [
        (value.name, value.logical_bytes, value.role, value.alias_of)
        for value in declared.values]
    recovered_value = next(value for value in recovered.values
                           if value.name == "value")
    assert recovered_value.layout.shape == (8, 2, 16)
    assert recovered_value.layout.strides == (128, 16, 1)
    assert recovered_value.layout.offset == 96
    assert recovered_value.logical_bytes == 512


def test_qkv_producer_is_bound_by_branch_not_largest_same_name_ledger() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = _a0(pipe)
    costs = tuple(pipe._subgraph(instance).evaluate(geometry).operators)
    projection = next(cost for cost in costs if cost.name == "qkv_projection")
    decoy = replace(
        projection, name="decoy",
        materialization=(("qkv_activation", Fraction(999_999)),),
        internal=())
    fused = lower_fusion(instance, (decoy,) + costs, "qkv_norm_rope",
                         supply=SUPPLY)
    assert dict(fused.kept)["value"] == 2 * 16 * 2 * 8
    value = next(value for value in fused.values if value.name == "value")
    assert value.producer == "qkv_projection"
    assert value.logical_bytes == 2 * 16 * 2 * 8


def test_qkv_producer_binds_actual_instance_not_same_name_other_layer() -> None:
    """A same-named projection cost from another layer cannot become input.

    L0 has 8 rows (2048 B projection), L1 has 16 rows (4096 B).  Inserting L1's
    cost between L0's projection and qk_norm must not change L0's typed slices.
    """
    pipe = pipeline(dense_model(layers=2))
    l0 = pipe.instantiated.find("gqa_attention", layer=0, role="attention")
    l1 = pipe.instantiated.find("gqa_attention", layer=1, role="attention")
    costs0 = tuple(pipe._subgraph(l0).evaluate(
        WorkloadGeometry.homogeneous(2, 4, 32)).operators)
    costs1 = tuple(pipe._subgraph(l1).evaluate(
        WorkloadGeometry.homogeneous(2, 8, 32)).operators)
    foreign = next(cost for cost in costs1 if cost.name == "qkv_projection")
    own = next(cost for cost in costs0 if cost.name == "qkv_projection")
    assert dict(own.materialization)["qkv_activation"] == 2048
    assert dict(foreign.materialization)["qkv_activation"] == 4096
    own_index = costs0.index(own)
    poisoned = costs0[:own_index + 1] + (foreign,) + costs0[own_index + 1:]
    for instance in (
            l0,
            replace(l0, template=replace(l0.template, contract=()))):
        fused = lower_fusion(instance, poisoned, "qkv_norm_rope",
                             supply=SUPPLY)
        kept = dict(fused.kept)
        assert kept["query"] == 4 * 16 * 2 * 8
        assert kept["key"] == 2 * 16 * 2 * 8
        assert kept["value"] == 2 * 16 * 2 * 8
        assert dict(fused.fused.materialization) == {
            "query": 4 * 16 * 2 * 8, "key": 2 * 16 * 2 * 8}
        assert fused.fused.pipe_work("mte3") == (4 * 16 + 2 * 16) * 2 * 8
    without_own = tuple(cost for cost in poisoned if cost is not own)
    with pytest.raises(ModelingError, match="fusion_qkv_producer_missing"):
        lower_fusion(l0, without_own, "qkv_norm_rope", supply=SUPPLY)


def test_declared_qkv_contract_without_typed_producer_is_a_replayable_gap() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = _a0(pipe)
    costs = pipe._subgraph(instance).evaluate(geometry).operators
    without_producer = tuple(
        cost for cost in costs if cost.name not in {"qkv_projection"})
    with pytest.raises(ModelingError, match="fusion_qkv_producer_missing"):
        lower_fusion(instance, without_producer, "qkv_norm_rope", supply=SUPPLY)
    no_contract = replace(
        instance, template=replace(instance.template, contract=()))
    with pytest.raises(ModelingError, match="fusion_qkv_producer_missing"):
        lower_fusion(no_contract, without_producer, "qkv_norm_rope", supply=SUPPLY)


def test_native_occurrence_caller_owner_is_rebound_on_cache_hit() -> None:
    """Nested typed/prepare attrs follow the actual instance, not L2."""
    pipe = pipeline(make_model(plans=[
        dense_plan(0, owns_long_kv=True),
        dense_plan(1, owns_long_kv=True),
        dense_plan(2, kv_owner_layer_id=0),
        dense_plan(3, kv_owner_layer_id=1),
    ]))
    instance = pipe.instantiated.find("kv_cache", layer=2, role="kv")
    geometry = WorkloadGeometry.homogeneous(2, 1, 32)
    base = pipe._subgraph(instance).evaluate(
        geometry, StateView().mark_kv(0)).operators[0]
    occurrence = {
        "layer": 2,
        "typed": {"attrs": {"owner_layer": 2, "tokens": 1024}},
        "prepare_args": {"attrs": {"owner_layer": 2, "tokens": 1024}},
    }
    cost = replace(base, native_occurrence=occurrence)
    rebound = ModelPipeline._rebind_cost(cost, 3, owner=1)
    got = rebound.native_occurrence
    assert got["layer"] == 3
    assert got["typed"]["attrs"]["owner_layer"] == 1
    assert got["prepare_args"]["attrs"]["owner_layer"] == 1
    assert got["typed"]["attrs"]["tokens"] == 1024
    assert got["prepare_args"]["attrs"]["tokens"] == 1024


def test_family_float_program_is_normalised_at_the_compile_boundary() -> None:
    """A tuple program's float duration/times cannot leak into serial sums."""
    quant = dict(name="w8a8", weight_format="int8", activation_format="int8",
                 weight_block=(128, 128), scale_bytes=4)
    pipe = pipeline(dense_model(quant=quant))
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    f0 = pipe._subgraph(pipe.instantiated.find("mlp", layer=0, role="dense_ffn"))
    evaluation = f0.evaluate(geometry)
    for cost in evaluation.operators:
        assert isinstance(cost, OperatorCost)
        assert isinstance(cost.duration_ms, Fraction)
        for _pipe, elapsed in cost.times:
            assert isinstance(elapsed, Fraction)
    assert isinstance(evaluation.total_ms, Fraction)

    baseline = evaluate_serial_split(f0, geometry, fusion=("dequant_swiglu_quant",))
    late = evaluate_serial_split(
        f0, geometry, fusion=("dequant_swiglu_quant",), readiness={"gate_up": 3})
    assert isinstance(baseline.total_ms, Fraction)
    assert isinstance(late.total_ms, Fraction)
    assert late.total_ms == baseline.total_ms + late.gap_ms


def test_negative_polynomial_degree_raises_modeling_error() -> None:
    with pytest.raises(ModelingError, match="expression_polynomial_degree_invalid"):
        polynomial(Var("tokens"), "tokens", degree=-1)


def test_plain_supply_still_publishes_exact_rational_cost() -> None:
    pipe = pipeline(dense_model())
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    instance = _a0(pipe)
    cost = pipe._subgraph(instance).evaluate(geometry).operators[0]
    assert isinstance(cost.duration_ms, Fraction)
    assert isinstance(cost.times[0][1], Fraction)


def test_layer_owner_id_is_a_state_port_not_a_template_split() -> None:
    """Two consumers with different KV owners share one structural template."""
    plans = [
        dense_plan(0, owns_long_kv=True),
        dense_plan(1, owns_long_kv=True),
        dense_plan(2, kv_owner_layer_id=0),
        dense_plan(3, kv_owner_layer_id=1),
    ]
    pipe = pipeline(make_model(plans=plans))
    geometry = WorkloadGeometry.homogeneous(2, 1, 32)
    aggregate = pipe.forward(geometry, aggregate=True)
    direct = pipe.forward(geometry, aggregate=False)
    assert aggregate.layers[2].group == aggregate.layers[3].group
    assert aggregate.layers[3].reused is True
    assert direct.layers[3].reused is False
    assert aggregate.evaluation_count == aggregate.template_groups
    # Only static work/time is reused.  Every identity of the reused layer is
    # rebound to the actual instance: subgraph id, external state ports, row
    # layer and native/fused occurrence layer.
    for layer_index in (2, 3):
        reused_layer = aggregate.layers[layer_index]
        direct_layer = direct.layers[layer_index]
        assert reused_layer.duration_ms == direct_layer.duration_ms
        assert [row.cost.name for row in reused_layer.rows] == [
            row.cost.name for row in direct_layer.rows]
        assert [row.cost.work for row in reused_layer.rows] == [
            row.cost.work for row in direct_layer.rows]
        assert len(reused_layer.subgraphs) == len(direct_layer.subgraphs)
        kv_owner = 0 if layer_index == 2 else 1
        for reused, fresh in zip(reused_layer.subgraphs,
                                 direct_layer.subgraphs):
            assert reused.instance_id == fresh.instance_id
            expected_id = f":L{layer_index}:"
            assert expected_id in reused.instance_id
            assert reused.state_in == fresh.state_in
            assert reused.state_out == fresh.state_out
            for subgraph in (reused, fresh):
                if subgraph.kind not in {"gqa_attention", "kv_cache"}:
                    continue
                owners = [port.owner for port in subgraph.state_in
                          if port.name == "kv_cache"]
                assert owners == [f"layer:{kv_owner}"]
                serialized = subgraph.to_dict()
                serialized_owners = [
                    port["owner"] for port in serialized["state_in"]
                    if port["name"] == "kv_cache"]
                assert serialized_owners == [f"layer:{kv_owner}"]
            assert [row.layer for row in reused.rows] == [
                row.layer for row in fresh.rows]
            assert [(row.native_occurrence or {}).get("layer")
                    for row in reused.rows] == [
                (row.native_occurrence or {}).get("layer")
                for row in fresh.rows]
            assert [row.native_call for row in reused.rows] == [
                row.native_call for row in fresh.rows]
            assert [fusion.fused.native_occurrence for fusion
                    in reused.fusion_results] == [
                fusion.fused.native_occurrence for fusion
                in fresh.fusion_results]
            for reused_fusion, fresh_fusion in zip(
                    reused.fusion_results, fresh.fusion_results):
                assert ([member.native_occurrence
                         for member in reused_fusion.semantic]
                        == [member.native_occurrence
                            for member in fresh_fusion.semantic])
                assert [(value.name, value.layout, value.offset,
                         value.layer)
                        for value in reused_fusion.values] == [
                    (value.name, value.layout, value.offset, layer_index)
                    for value in fresh_fusion.values]


def test_legacy_qkv_views_recover_gated_and_gemma_addresses() -> None:
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    for pipe, layer, expected in (
            (pipeline(make_model(plans=[dense_plan(0, q_output_gate=True)])), 0,
             {"gate": (16, (192, 32, 1)), "value": (160, (192, 16, 1))}),
            (pipeline(gemma4_model_spec(gemma_config())), 1,
             {"value": (2048, (3072, 512, 1)),
              "projection_k": (2048, (3072, 512, 1))})):
        instance = pipe.instantiated.find("gqa_attention", layer=layer, role="attention")
        costs = pipe._subgraph(instance).evaluate(geometry).operators
        old = replace(instance, template=replace(instance.template, contract=()))
        fused = lower_fusion(old, costs, "qkv_norm_rope", supply=SUPPLY)
        values = {value.name: value for value in fused.values}
        for name, (offset, strides) in expected.items():
            assert values[name].layout.offset == offset
            assert values[name].layout.strides == strides
            assert values[name].logical_bytes > 0


def test_w8a8_qkv_uses_post_norm_qk_width_and_projection_v_width() -> None:
    quant = dict(name="w8a8", weight_format="int8", activation_format="int8",
                 weight_block=(128, 128), scale_bytes=4)
    pipe = pipeline(dense_model(layers=2, quant=quant))
    instance = pipe.instantiated.find("gqa_attention", layer=1, role="attention")
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    fused = lower_fusion(
        instance, pipe._subgraph(instance).evaluate(geometry).operators,
        "qkv_norm_rope", supply=SUPPLY)
    values = {value.name: value for value in fused.values}
    assert values["query"].logical_bytes == 1024
    assert values["key"].logical_bytes == 512
    assert values["query"].written_bytes == 1024
    assert values["key"].written_bytes == 512
    assert values["value"].logical_bytes == 256
    assert values["value"].written_bytes == 0
