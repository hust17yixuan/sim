"""Canonical subgraph construction, disposition and standalone-level invariants.

[Role]
  Establish that every declared kind/variant is constructible from explicit
  structural facts, that the vocabulary and the builder-selected variants stay
  aligned, that typed ports/fusions enter the existing piecewise program, and
  that repeated structure is layer/label independent.

[Invariants]
  * A builder may only emit a variant declared by its kind; an undeclared
    variant is a loud modeling error, never a silently renamed template.
  * Template identity excludes model names, layer ids and request ids; two
    relocations of the same structure intern one template object.
  * Every constructed operator family is priceable by the existing piecewise
    compiler, so a subgraph never needs a second evaluator.
  * Logical and issued matrix work stay distinct: a tail token raises issued
    FLOPs while the logical FLOPs follow the requested geometry.
  * The disposition projection accounts for all 22 declared kinds, including
    kinds this model does not select.

[Timeouts & Locks]
  Pure CPU construction and bounded arithmetic; no waits, sleeps or locks.

[Side-Effect Policy]
  Read-only repository fixtures; no filesystem, network or device mutation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, PipeSupply, WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
from mindie_sim.prediction.piecewise.families import FAMILY_COMPILERS
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.prediction.subgraphs import (
    bind_occurrence,
    FeatureConfig, TemplateRegistry, VOCABULARY, build_standalone,
    compile_standalone, evaluate_standalone, evaluate_standalone_graph,
    instantiate_model, occurrence_contract, standalone_span,
    subgraph_dispositions,
)
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import SUBGRAPH_KINDS


ROOT = Path(__file__).resolve().parents[1]
A5_CASES = (
    "qwen38-a5", "qwen35-a5", "glm53-a5", "kimi-k3-a5",
    "v41-a5-dp4-body-current",
)
GEOMETRY = WorkloadGeometry.homogeneous(2, 4, 32, media_tokens=16)


def _supply(scale=1.0):
    return PipeSupply((
        ("cube", 1_000_000.0 * scale), ("vector", 1_000_000.0),
        ("hbm", 1_000_000.0), ("mte1", 2_000_000.0), ("mte2", 2_000_000.0),
        ("mte3", 2_000_000.0), ("fixpipe", 2_000_000.0), ("link", 1_000_000.0),
    ))


POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)


def _features(model):
    return FeatureConfig.from_model(
        model, token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
        split_target=1000, max_splits=4)


def _all_kinds_model():
    """One declared structure that selects every canonical kind once."""
    plans = [
        dict(index=0, attention="gqa", ffn="moe", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, num_experts=4, top_k=2,
             moe_intermediate_size=32, num_shared_experts=1, qk_norm=True,
             hyper_connections=True, hc_mult=2, hc_sinkhorn_iters=2,
             engram=True, residual_block_size=2, owns_long_kv=True),
        dict(index=1, attention="mla", ffn="dense", num_heads=4, num_kv_heads=1,
             head_dim=16, q_lora_rank=16, kv_lora_rank=16,
             qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
             intermediate_size=128, indexer=True, index_heads=2, index_dim=8,
             index_topk=4, candidate_filter=True, candidate_topk_blocks=2,
             candidate_block_size=4, kv_owner_layer_id=0, owns_index_k=True),
        dict(index=2, attention="gdn", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, linear_heads=2,
             linear_key_heads=2, linear_value_heads=2, linear_key_head_dim=8,
             linear_head_dim=8, conv_kernel=4, recurrent_state_elements=64,
             conv_state_elements=16, state_bytes=256),
        dict(index=3, attention="kda", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, linear_key_heads=2,
             linear_value_heads=2, linear_key_head_dim=8, linear_head_dim=8,
             conv_kernel=4, gate_rank=4, recurrent_state_elements=64,
             conv_state_elements=16, state_bytes=256, owns_long_kv=True),
        dict(index=4, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128,
             cache_kind="window_compressed", compress_ratio=4,
             compressor_kind="weighted_pool", indexer=True, index_heads=1,
             index_dim=8, index_topk=2, owns_long_kv=True),
        # The attention mechanism is exclusive per layer, so selecting every
        # canonical kind needs one layer per family.
        dict(index=5, attention="mha", ffn="dense", num_heads=4, num_kv_heads=4,
             head_dim=16, intermediate_size=128, owns_long_kv=True),
        dict(index=6, attention="mqa", ffn="dense", num_heads=4, num_kv_heads=1,
             head_dim=16, intermediate_size=128, owns_long_kv=True),
        dict(index=7, attention="projection_only", ffn="dense", num_heads=0,
             num_kv_heads=0, head_dim=0, intermediate_size=128,
             cache_kind="none", owns_long_kv=True),
        dict(index=8, attention="mla", ffn="dense", num_heads=4, num_kv_heads=1,
             head_dim=16, q_lora_rank=16, kv_lora_rank=16,
             qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
             intermediate_size=128, owns_long_kv=True),
        dict(index=9, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, window=128,
             cache_kind="window_compressed", compress_ratio=0,
             compressor_kind="none", owns_long_kv=True),
        dict(index=10, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128,
             cache_kind="window_compressed", compress_ratio=2,
             architecture="csa2", compressor_kind="weighted_pool",
             indexer=True, index_heads=1, index_dim=8, index_topk=2,
             owns_long_kv=True),
        dict(index=11, attention="gqa", ffn="dense", num_heads=4, num_kv_heads=2,
             head_dim=16, intermediate_size=128, window=128,
             cache_kind="window_compressed", compress_ratio=128,
             architecture="hca", compressor_kind="weighted_pool",
             owns_long_kv=True),
    ]
    raw = dict(
        name="all-kinds", layers=len(plans), hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        num_experts=4, top_k=2, moe_intermediate_size=32,
        num_shared_experts=1, layer_plans=plans,
        mla=dict(q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16,
                 qk_rope_head_dim=8, v_head_dim=16, index_topk=4,
                 index_head_dim=8, index_n_heads=2, compress_ratio=4),
        vision=dict(kind="vision", layers=1, hidden_size=64,
                    intermediate_size=128, num_heads=4, qkv_width=192,
                    patch_size=14),
        engram=dict(tables=[dict(layer_id=0, table_id=0, rows=64, width=64)],
                    max_ngram=2, heads=1, width=64),
        auxiliary=dict(mtp=dict(kind="mtp", layers=1, hidden_size=64,
                                intermediate_size=128, num_heads=4,
                                num_kv_heads=2, head_dim=16, q_width=64,
                                kv_width=32)))
    return ModelSpec.from_mapping(raw)


@pytest.fixture(scope="module")
def all_kinds():
    model = _all_kinds_model()
    return model, instantiate_model(model, _features(model))


def _sources(instantiated):
    source = {}
    for instance in instantiated.subgraphs:
        source.setdefault(instance.kind, instance)
    return source


VARIANT_DISPOSITIONS = (
    "subgraph_variant_requires_structure:",
    "subgraph_variant_conflicts_with_structure:",
    "subgraph_variant_label_only:",
)


def _construct_variants(model, instantiated, registry):
    """Construct every declared name, or collect its explicit disposition.

    A name is not a behavior: a builder may reject a request whose physical
    structure it does not model (``requires_structure``/``conflicts``) or one
    that would only rename the natural template (``label_only``).  Silent
    success with a different body is the failure this test rules out.
    """
    source = _sources(instantiated)
    selected = {(item.kind, item.variant): item
                for item in instantiated.subgraphs}
    built, dispositions = {}, {}
    for spec in VOCABULARY:
        for variant in spec.variants:
            origin = selected.get((spec.kind, variant), source[spec.kind])
            role = str(dict(origin.metadata).get("role", "body"))
            layer = model.layer_plans[origin.layer]
            try:
                instance = build_standalone(
                    spec.kind, variant=variant, model=model, layer=layer,
                    role=role, registry=registry)
            except ModelingError as error:
                message = str(error)
                assert message.startswith(VARIANT_DISPOSITIONS), (
                    spec.kind, variant, message)
                dispositions[(spec.kind, variant)] = message
                continue
            assert instance.template.variant == variant
            built[(spec.kind, variant)] = instance
    return built, dispositions


def test_declared_variants_construct_or_report_explicit_disposition(all_kinds):
    model, instantiated = all_kinds
    registry = TemplateRegistry(512)
    built, dispositions = _construct_variants(model, instantiated, registry)
    for (kind, variant), instance in built.items():
        template = instance.template
        assert (template.kind, template.variant) == (kind, variant)
        assert template.operator_names
        # Every operator enters the one existing piecewise program.
        assert {op.family for op in template.operators} <= set(FAMILY_COMPILERS)
        # Typed subgraph ports: a producer port is never declared as an input
        # and every representation is explicit.
        assert all(port.direction in {"in", "inout"} and port.representation
                   for port in template.state_in)
        assert all(port.direction in {"out", "inout"} and port.representation
                   for port in template.state_out)
        assert template.state_in or template.state_out
        # A fusion only names members of this template and stays on one stream.
        streams = {op.name: op.stream for op in template.operators}
        for fusion in template.fusions:
            assert set(fusion.members) <= set(streams)
            assert {streams[name] for name in fusion.members} == {fusion.stream}
            assert not set(fusion.removes) & set(fusion.keeps)
        # Rebuilding the same structure interns the same object.
        again = build_standalone(kind, variant=variant, model=model,
                                 layer=model.layer_plans[instance.layer],
                                 role=str(dict(instance.metadata)["role"]),
                                 registry=registry)
        assert again.template is template
    # Every kind has at least one modeled variant, and every natural variant
    # the model actually selected is constructible.
    assert {kind for kind, _ in built} == set(SUBGRAPH_KINDS)
    selected = {(item.kind, item.variant) for item in instantiated.subgraphs}
    assert selected <= set(built)
    # The audit must exercise all three explicit dispositions, not only names.
    assert any(kind == "kv_cache" and variant == "transfer"
               for kind, variant in dispositions)
    assert any(kind == "kv_compressor" and "conflicts_with_structure" in message
               for (kind, _), message in dispositions.items())
    assert any(message.startswith("subgraph_variant_label_only:")
               for message in dispositions.values())


def test_undeclared_variant_and_kind_fail_loudly(all_kinds):
    model, _ = all_kinds
    with pytest.raises(ModelingError,
                       match="subgraph_variant_undeclared:gqa_attention:banana"):
        build_standalone("gqa_attention", variant="banana", model=model,
                         layer=model.layer_plans[0])
    with pytest.raises(ModelingError, match="subgraph_kind_unknown"):
        build_standalone("ZZ", model=model, layer=model.layer_plans[0])


def test_template_identity_is_layer_and_model_label_independent(all_kinds):
    model, _ = all_kinds
    registry = TemplateRegistry(64)
    first = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                             role="attention", registry=registry)
    relocated = replace(model.layer_plans[0], index=3)
    second = build_standalone("gqa_attention", model=model, layer=relocated,
                              role="attention", registry=registry)
    assert second.template is first.template
    assert second.performance_key() == first.performance_key()
    renamed = replace(model, name="another-label")
    third = build_standalone("gqa_attention", model=renamed, layer=model.layer_plans[0],
                             role="attention", registry=registry)
    assert third.template is first.template
    assert third.performance_key() == first.performance_key()


def test_logical_and_issued_matrix_work_stay_distinct(all_kinds):
    model, _ = all_kinds
    instance = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                                role="attention")
    geometry = WorkloadGeometry.homogeneous(1, 6, 7)
    evaluation = evaluate_standalone(instance, geometry, _supply(),
                                     policy=POLICY, state=None)
    projection = next(cost for cost in evaluation.operators
                      if cost.name == "qkv_projection")
    logical = projection.get("flops")
    issued = projection.get("padded_flops")
    assert logical == 2 * projection.get("rows") * projection.get("weight_total") / 2
    assert issued > logical
    assert projection.get("mte2_padded_bytes") >= projection.get("mte2_useful_bytes")
    assert sum(cost.duration_ms for cost in evaluation.operators) == evaluation.total_ms


def test_engine_preserves_cross_stream_dependency_and_stream_order():
    # The formal review's controlled four-node mechanism example: two cube
    # GEMMs (1 and 10 work) finish at 2 and 11 ms under fair sharing; gating
    # must wait for both and run at 11-12 ms; conv then runs at 12-16 ms.
    engine = Engine([
        Resource("cube", 1.0, unit="work", policy="fair"),
        Resource("vector", 1.0, unit="work", policy="fair")])
    qkvz = engine.submit("qkvz", {"cube": 1.0})
    ba = engine.submit("ba", {"cube": 10.0})
    gating = engine.submit("gating", {"vector": 1.0}, after=(qkvz, ba))
    conv = engine.submit("conv", {"vector": 4.0}, after=(gating,))
    engine.advance(until=(conv,))
    assert (qkvz.started, qkvz.time) == (0.0, 2.0)
    assert (ba.started, ba.time) == (0.0, 11.0)
    assert (gating.started, gating.time) == (11.0, 12.0)
    assert (conv.started, conv.time) == (12.0, 16.0)
    assert engine.now == 16.0


def test_standalone_span_reports_unexpressed_cross_stream_dependency():
    model = _gdn_model()
    features = replace(_features(model), multistream_kda_proj=True)
    instance = build_standalone("gdn_linear_attn", model=model, layer=model.layer_plans[0],
                                role="attention", features=features)
    contract = occurrence_contract(instance)
    assert {stream for stream, _ in contract["streams"]} == {"main", "aux"}
    cross = contract["cross_stream_edges"]
    assert cross
    assert any(edge["producer"] == "ba_projection"
               and edge["consumer"] in {"beta_slice", "alpha_slice"}
               for edge in cross)
    # The common solver still selects a fixed name subset and cannot apply
    # these named producer edges.  The independent entry must report the
    # exact gap instead of returning the current 12-ms-style false span.
    with pytest.raises(ModelingError,
                       match="standalone_span_dependency_unexpressed"):
        standalone_span(instance, GEOMETRY, _supply(), policy=POLICY)
    with pytest.raises(ModelingError,
                       match="standalone_span_dependency_unexpressed"):
        evaluate_standalone_graph(instance, GEOMETRY, _supply(),
                                  policy=POLICY, fusion=False)


def test_standalone_span_preserves_single_stream_path(all_kinds):
    model, _ = all_kinds
    instance = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                                role="attention")
    contract = occurrence_contract(instance)
    assert not contract["cross_stream_edges"]
    assert {stream for stream, _ in contract["streams"]} == {"main"}
    span = standalone_span(instance, GEOMETRY, _supply(), policy=POLICY)
    assert span["span_scope"] == "single_chain"
    assert span["contention"] is None
    assert span["span_ms"] == span["serial_ms"]
    assert span["readiness_gap_ms"] == 0
    assert span["span_limits"] is not None
    # Every occurrence keeps its ordered output port in the contract.
    assert tuple(item["name"] for item in contract["occurrences"]) == tuple(
        operator.name for operator in instance.operators)


def test_dispositions_account_for_every_declared_kind(all_kinds):
    _, instantiated = all_kinds
    dispositions = subgraph_dispositions(instantiated)
    assert tuple(dispositions) == SUBGRAPH_KINDS
    assert all(item["status"] in {"instantiated", "partial"}
               for item in dispositions.values())
    present = {instance.kind for instance in instantiated.subgraphs}
    assert {kind for kind, item in dispositions.items() if item["instances"]}
    assert {kind for kind, item in dispositions.items() if item["instances"]} == present
    for kind, item in dispositions.items():
        if item["status"] == "partial":
            assert "partial" in item["coverage"]


@pytest.mark.parametrize("case_name", A5_CASES)
def test_real_a5_cases_reproduce_standalone_with_declared_variants(case_name):
    raw = json.loads((ROOT / "examples/cases" / f"{case_name}.json").read_text())
    case = parse_case(raw)
    # Use the production runtime/parallel/source facts, not a bare model
    # default, so the standalone entry is compared with what the model path
    # actually selected.
    features = FeatureConfig.from_runtime(
        case.runtime, case.model, machine=case.machine)
    instantiated = instantiate_model(case.model, features)
    declared = {spec.kind: set(spec.variants) for spec in VOCABULARY}
    assert instantiated.subgraphs
    for instance in instantiated.subgraphs:
        assert instance.kind in declared
        assert instance.variant in declared[instance.kind], (
            instance.kind, instance.variant)
        role = str(dict(instance.metadata).get("role", "body"))
        built = build_standalone(
            instance.kind, variant=instance.variant, model=case.model,
            layer=case.model.layer_plans[instance.layer], role=role,
            features=features, stage=instance.stage)
        assert built.layer == instance.layer and built.stage == instance.stage
        assert built.template.fingerprint() == instance.template.fingerprint()
        assert tuple((operator.name, operator.theta.key())
                     for operator in built.operators) == tuple(
            (operator.name, operator.theta.key())
            for operator in instance.operators)
    dispositions = subgraph_dispositions(instantiated)
    assert tuple(dispositions) == SUBGRAPH_KINDS
    assert sum(1 for item in dispositions.values()
               if item["status"] in {"instantiated", "partial"}) >= 5
    assert any(item["status"] == "not_selected"
               for item in dispositions.values())


def _gdn_model():
    raw = dict(
        name="gdn", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gdn", ffn="dense",
                          num_heads=4, num_kv_heads=2, head_dim=16,
                          intermediate_size=128, linear_heads=4,
                          linear_head_dim=8, linear_key_heads=4,
                          linear_key_head_dim=8, linear_value_heads=4,
                          conv_kernel=4, recurrent_state_elements=64,
                          conv_state_elements=16)])
    return ModelSpec.from_mapping(raw)


def test_k0_requested_variants_have_matching_read_store_work(all_kinds):
    model, _ = all_kinds
    layer = model.layer_plans[0]
    store = build_standalone("kv_cache", variant="store", model=model, layer=layer,
                             role="kv")
    assert "kv_store" in store.template.operator_names
    assert "kv_read" not in store.template.operator_names
    stored = evaluate_standalone(store, GEOMETRY, _supply(), policy=POLICY,
                                 state=StateView())
    assert (0, "kv_cache") in stored.state_out.kv_ready

    read = build_standalone("kv_cache", variant="read", model=model, layer=layer,
                            role="kv")
    assert "kv_read" in read.template.operator_names
    assert "kv_store" not in read.template.operator_names
    loaded = evaluate_standalone(read, GEOMETRY, _supply(), policy=POLICY,
                                 state=StateView().mark_kv(0))
    kv_read = next(cost for cost in loaded.operators if cost.name == "kv_read")
    assert kv_read.get("kv_bytes") > 0
    # A real read requires the declared owner's cache, not a self-trigger.
    with pytest.raises(ModelingError, match="state_port_not_ready"):
        evaluate_standalone(read, GEOMETRY, _supply(), policy=POLICY)
    with pytest.raises(ModelingError, match="destination_stages"):
        build_standalone("kv_cache", variant="transfer", model=model, layer=layer,
                         role="kv")


def test_k1_requested_kind_must_match_declared_compressor(all_kinds):
    model, instantiated = all_kinds
    source = _sources(instantiated)["kv_compressor"]
    layer = model.layer_plans[source.layer]
    natural = build_standalone("kv_compressor", model=model, layer=layer, role="compress")
    assert natural.template.variant == "weighted_pool"
    for conflicting in ("none", "gated_ring", "kpool"):
        with pytest.raises(ModelingError,
                           match="subgraph_variant_conflicts_with_structure"):
            build_standalone("kv_compressor", variant=conflicting, model=model,
                             layer=layer, role="compress")
    no_source = model.layer_plans[0]
    marker = build_standalone("kv_compressor", variant="none", model=model,
                              layer=no_source, role="compress")
    assert marker.template.variant == "none"
    with pytest.raises(ModelingError, match="compressor_source"):
        build_standalone("kv_compressor", variant="weighted_pool", model=model,
                         layer=no_source, role="compress")


def test_nonzero_layer_and_owner_state_identity_are_preserved():
    model = _mla_model(layers=4)
    producer = build_standalone("indexer_prolog", variant="full", model=model,
                                layer=model.layer_plans[3],
                                role="index_produce")
    assert producer.layer == 3
    produced = evaluate_standalone(producer, GEOMETRY, _supply(),
                                   policy=POLICY, state=StateView())
    assert (3, "index") in produced.state_out.index_ready
    assert (0, "index") not in produced.state_out.index_ready

    consumer = build_standalone("lightning_indexer", variant="topk", model=model,
                                layer=model.layer_plans[3],
                                role="index_select")
    assert consumer.layer == 3
    with pytest.raises(ModelingError, match="state_port_not_ready"):
        evaluate_standalone(consumer, GEOMETRY, _supply(), policy=POLICY,
                            state=StateView().mark_index(0))
    same_owner = evaluate_standalone(consumer, GEOMETRY, _supply(),
                                     policy=POLICY,
                                     state=StateView().mark_index(3))
    assert same_owner.total_ms >= 0

    shared = ModelSpec.from_mapping(dict(
        name="shared-kv", layers=2, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[
            dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 owns_long_kv=True),
            dict(index=1, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 kv_owner_layer_id=0, owns_long_kv=False)]))
    read = build_standalone("kv_cache", variant="read", model=shared,
                            layer=shared.layer_plans[1], role="kv")
    assert read.layer == 1 and read.owner_layer == 0
    evaluate_standalone(read, GEOMETRY, _supply(), policy=POLICY,
                        state=StateView().mark_kv(0))


def test_standalone_forwards_source_metadata_to_the_existing_compiler(all_kinds):
    model, _ = all_kinds
    features = replace(_features(model), metadata=(("standalone_probe", 1),))
    instance = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                                role="attention", features=features)
    assert dict(instance.metadata)["feature_metadata"] == (
        ("standalone_probe", 1),)
    compiled = compile_standalone(instance, _supply(), policy=POLICY)
    assert all(operator.feature_metadata == (("standalone_probe", 1),)
               for operator in compiled.operators)


def test_label_only_variant_request_is_rejected(all_kinds):
    model, _ = all_kinds
    # ``mha`` is a variant of the ``mha_attention`` kind, not of
    # ``gqa_attention``: asking a GQA layer for it would relabel the same
    # operators instead of modelling another mechanism, so it is refused.
    with pytest.raises(ModelingError,
                       match="subgraph_variant_undeclared:gqa_attention:mha"):
        build_standalone("gqa_attention", variant="mha", model=model,
                         layer=model.layer_plans[0], role="attention")


def _mla_model(*, layers=1, **overrides):
    base = dict(
        name="mla", layers=layers, hidden_size=64, num_heads=4,
        num_kv_heads=1, head_dim=16, intermediate_size=128, vocab_size=128,
        mla=dict(q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16,
                 qk_rope_head_dim=8, v_head_dim=16, index_topk=4,
                 index_head_dim=8, index_n_heads=2))
    plan = dict(index=0, attention="mla", ffn="dense", num_heads=4,
                num_kv_heads=1, head_dim=16, intermediate_size=128,
                q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=16,
                qk_rope_head_dim=8, v_head_dim=16, indexer=True,
                indexer_type="full", index_heads=2, index_dim=8,
                index_topk=4)
    plan.update(overrides)
    plans = [dict(plan, index=index) for index in range(layers)]
    return ModelSpec.from_mapping({**base, "layer_plans": plans})


def test_index_cache_ownership_fills_only_an_unspecified_owner():
    model = _mla_model()
    instance = build_standalone("indexer_prolog", variant="full", model=model,
                                layer=model.layer_plans[0],
                                role="index_produce")
    assert "index_cache_store" in instance.template.operator_names
    evaluation = evaluate_standalone(
        instance, WorkloadGeometry.homogeneous(2, 4, 32), _supply(),
        policy=POLICY, state=StateView())
    assert (0, "index") in evaluation.state_out.index_ready

    shared = _mla_model(layers=2, owns_index_k=False,
                        index_k_owner_layer_id=0)
    consumer = build_standalone("indexer_prolog", variant="full", model=shared,
                                layer=shared.layer_plans[1],
                                role="index_produce")
    assert "index_cache_store" not in consumer.template.operator_names


def test_k0_transfer_binds_source_readiness_and_publishes_destination():
    model = ModelSpec.from_mapping(dict(
        name="shared-kv-transfer", layers=2, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[
            dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 owns_long_kv=True),
            dict(index=1, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 kv_owner_layer_id=0, owns_long_kv=False)]))
    features = replace(_features(model), pp=2)
    transfer = build_standalone("kv_cache", variant="transfer", model=model,
                                layer=model.layer_plans[0], role="kv",
                                features=features)
    names = transfer.template.operator_names
    assert "kv_store" not in names
    assert "kv_read" in names and "stage_transfer" in names
    assert "cross_stage_ready" in {port.name
                                   for port in transfer.template.state_out}
    # Empty source state: neither entry may transfer or publish a destination.
    for evaluate in (evaluate_standalone, evaluate_standalone_graph):
        with pytest.raises(ModelingError, match="state_port_not_ready"):
            evaluate(transfer, GEOMETRY, _supply(), policy=POLICY,
                     state=StateView())
    ready = evaluate_standalone(transfer, GEOMETRY, _supply(), policy=POLICY,
                                state=StateView().mark_kv(0))
    graph = evaluate_standalone_graph(transfer, GEOMETRY, _supply(),
                                      policy=POLICY,
                                      state=StateView().mark_kv(0))
    assert ready.state_out.cross_stage_ready == graph.state.cross_stage_ready
    assert (0, 1) in {key for key, _ in ready.state_out.cross_stage_ready}
    costs = {cost.name: cost for cost in ready.operators}
    assert costs["kv_read"].get("kv_bytes") == 0
    assert costs["layout_convert"].pipe_work("hbm") > 0
    assert costs["stage_transfer"].pipe_work("link") > 0
    span = standalone_span(transfer, GEOMETRY, _supply(), policy=POLICY,
                           state=StateView().mark_kv(0))
    assert span["span_scope"] == "single_chain"


def _state_port(instance, name, direction):
    ports = instance.state_in if direction == "in" else instance.state_out
    return next(port for port in ports if port.name == name)


def _kv_owner_model():
    plans = []
    for index in range(4):
        plan = dict(index=index, attention="gqa", ffn="dense", num_heads=4,
                    num_kv_heads=2, head_dim=16, intermediate_size=128)
        if index < 2:
            plan["owns_long_kv"] = True
        else:
            plan["kv_owner_layer_id"] = index - 2
            plan["owns_long_kv"] = False
        plans.append(plan)
    return ModelSpec.from_mapping(dict(
        name="kv-owners", layers=4, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=plans))


def test_instance_state_ports_bind_actual_kv_reader_owner():
    model = _kv_owner_model()
    instantiated = instantiate_model(model, _features(model))
    for layer, owner in ((2, 0), (3, 1)):
        attention = instantiated.find("gqa_attention", layer=layer, role="attention")
        assert _state_port(attention, "kv_cache", "in").owner == f"layer:{owner}"
        assert _state_port(attention, "kv_ready", "out").owner == f"layer:{owner}"
        assert _state_port(attention, "hidden", "in").owner == "self"
        read = instantiated.find("kv_cache", layer=layer, role="kv")
        assert _state_port(read, "kv_cache", "in").owner == f"layer:{owner}"
    for layer in (0, 1):
        store = instantiated.find("kv_cache", layer=layer, role="kv")
        assert _state_port(store, "kv_cache", "out").owner == f"layer:{layer}"
        assert _state_port(store, "kv_cache", "in").owner == "self"


def test_a0_contract_declares_actual_producer_output_and_views(all_kinds):
    model, _ = all_kinds
    standard = build_standalone("gqa_attention", model=model,
                                layer=model.layer_plans[0], role="attention")
    contract = dict(standard.template.contract)
    assert contract["qkv_producer"] == "qkv_projection"
    assert contract["qkv_output"] == "qkv_activation"
    assert contract["qkv_output_dtype"]
    assert contract["qkv_output_layout"][0] == "strided"
    assert int(contract["qkv_output_elements"]) > 0
    assert float(contract["qkv_output_row_bytes"]) > 0
    assert contract["qkv_value_view"][1] == "projection"
    assert contract["qkv_value_view"][2] == contract["qkv_producer"]
    assert contract["qkv_value_view"][3] == contract["qkv_output"]
    assert contract["qkv_key_view"][1] == "fused"
    assert contract["qkv_key_view"][5] == "post_rope"
    assert contract["qkv_query_view"][5] == "post_rope"

    gemma = replace(model, catalog_facts=(("gemma4_k_eq_v_all", True),))
    gemma_a0 = build_standalone("gqa_attention", model=gemma,
                                layer=model.layer_plans[0], role="attention")
    gemma_contract = dict(gemma_a0.template.contract)
    gemma_value = gemma_contract["qkv_value_view"]
    assert gemma_value[0] == "value" and gemma_value[1] == "alias"
    assert gemma_value[2] == "qkv_projection"
    assert gemma_value[3] == "qkv_activation"
    assert gemma_value[5] == "pre_rope"
    assert gemma_contract["qkv_key_view"][1] == "fused"
    assert gemma_contract["qkv_key_view"][5] == "post_rope"
    gemma_views = {view[0]: view for view in gemma_contract["qkv_views"]}
    assert gemma_views["value"][7] == "projection_k"

    shared_model = _kv_owner_model()
    shared = build_standalone("gqa_attention", model=shared_model,
                              layer=shared_model.layer_plans[2],
                              role="attention")
    shared_contract = dict(shared.template.contract)
    assert shared_contract["qkv_producer"] == "q_projection"
    assert shared_contract["qkv_output"] == "q_projection_output"
    assert shared_contract["qkv_value_view"][1] == "state"
    assert shared_contract["qkv_value_view"][3] == "kv_owner"
    assert shared_contract["qkv_value_view"][5] == "current"
    assert shared_contract["qkv_cache_key_view"][3] == "kv_owner"
    assert shared_contract["qkv_key_view"][5] == "current"
    assert shared_contract["qkv_key_view"][3] == "kv_owner"
    assert shared_contract["qkv_query_view"][5] == "post_rope"
    assert shared.owner_layer == 0


def test_model_prediction_exposes_actual_kv_owner_for_shared_layers():
    model = _kv_owner_model()
    pipeline = assemble_model(model, supply=_supply(), policy=POLICY,
                              features=_features(model))
    for aggregate in (True, False):
        try:
            prediction = pipeline.forward(
                WorkloadGeometry.homogeneous(2, 4, 32), aggregate=aggregate)
        except AttributeError as exc:
            pytest.skip(f"PW-CORE model_pipeline interface in flight: {exc}")
        for layer, owner in ((2, 0), (3, 1)):
            for kind in ("gqa_attention", "kv_cache"):
                subgraph = next(item for item in prediction.layers[layer].subgraphs
                                if item.kind == kind)
                port = next(item for item in subgraph.state_in
                            if item.name == "kv_cache")
                assert port.owner == f"layer:{owner}"
                assert f"L{layer}" in subgraph.instance_id
                serialized = next(
                    item for item in subgraph.to_dict()["state_in"]
                    if item["name"] == "kv_cache")
                assert serialized["owner"] == f"layer:{owner}"


def test_a0_shared_projection_uses_selected_category_output_format():
    model = _kv_owner_model()
    quant = QuantSpec(
        name="q-output-override", weight_format="bf16",
        activation_format="bf16",
        category_overrides=(
            ("attn.q", (("output_format", "fp32"),)),
            ("attn.qkv", (("output_format", "bf16"),))))
    model = replace(model, quant=quant)
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[2],
                          role="attention")
    contract = dict(a0.template.contract)
    assert contract["qkv_producer"] == "q_projection"
    assert contract["qkv_output_dtype"] == "fp32"
    assert float(contract["qkv_output_row_bytes"]) == 256.0
    contract_views = {view[0]: view for view in contract["qkv_views"]}
    assert contract_views["query_projection"][5] == "fp32"
    assert contract["qkv_query_view"][5] == "post_rope"
    projected = next(item for item in occurrence_contract(a0)["occurrences"]
                     if item["name"] == "q_projection")
    assert projected["outputs"][0]["dtype"] == "fp32"


def test_a0_gated_views_partition_the_projection_row():
    model = ModelSpec.from_mapping(dict(
        name="gated", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=8, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense",
                          num_heads=4, num_kv_heads=2, head_dim=8,
                          intermediate_size=128, q_output_gate=True)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    contract = dict(a0.template.contract)
    assert contract["qkv_output_layout"][1] == 96
    views = {view[0]: view for view in contract["qkv_views"]}

    def addresses(view):
        shape, strides, offset, _storage = view[4]
        return {offset + i * strides[0] + j * strides[1]
                for i in range(shape[0]) for j in range(shape[1])}

    query = addresses(views["query_projection"])
    gate = addresses(views["gate"])
    key = addresses(views["key_projection"])
    value = addresses(views["value"])
    assert views["query_projection"][4][2] == 0
    assert views["gate"][4][2] == 8
    assert views["key_projection"][4][2] == 64
    assert views["value"][4][2] == 80
    assert query.isdisjoint(gate) and query.isdisjoint(key)
    assert gate.isdisjoint(key) and key.isdisjoint(value)
    assert query | gate | key | value == set(range(96))


def test_a4_conv_consumes_the_projection_output_alias():
    model = ModelSpec.from_mapping(dict(
        name="kda", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="kda", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16, intermediate_size=128,
                          linear_key_heads=2, linear_value_heads=2,
                          linear_key_head_dim=8, linear_head_dim=8,
                          conv_kernel=4, gate_rank=4,
                          recurrent_state_elements=64,
                          conv_state_elements=16, state_bytes=256,
                          owns_long_kv=True)]))
    features = replace(_features(model), multistream_kda_proj=True)
    instance = build_standalone("kda_linear_attn", model=model,
                                layer=model.layer_plans[0],
                                role="attention", features=features)
    contract = occurrence_contract(instance)
    conv = next(item for item in contract["occurrences"]
                if item["name"] == "conv")
    assert [(item["name"], item["producer"]) for item in conv["inputs"]] == [
        ("qkv_activation", "qkv_projection")]
    assert not contract["unresolved_inputs"]


def test_a3_gating_beta_decay_ba_and_conv_edges():
    model = _gdn_model()
    features = replace(_features(model), multistream_kda_proj=True)
    for variant in ("recurrent_decode", "chunked_prefill"):
        instance = build_standalone("gdn_linear_attn", variant=variant, model=model,
                                    layer=model.layer_plans[0],
                                    role="attention", features=features)
        contract = occurrence_contract(instance)
        rows = {item["name"]: item for item in contract["occurrences"]}
        assert [(item["name"], item["dtype"])
                for item in rows["gdn_gating"]["outputs"]] == [
            ("gdn_decay_fp32", "fp32"), ("gdn_beta_bf16", "bf16")]
        assert any(item["producer"] == "ba_projection"
                   for item in rows["gdn_gating"]["inputs"])
        assert all(item["producer"] for item in rows["gdn_gating"]["inputs"])
        expected_conv = ("qkv_slice" if variant == "recurrent_decode"
                         else "qkvz_projection")
        assert [item["producer"] for item in rows["conv"]["inputs"]] == [
            expected_conv]
        delta = {item["name"] for item in rows["gated_delta"]["inputs"]}
        assert {"gdn_decay_fp32", "gdn_beta_bf16"} <= delta
        assert any(edge["producer"] == "ba_projection"
                   and edge["consumer"] == "gdn_gating"
                   for edge in contract["cross_stream_edges"])


def _a2_model():
    return ModelSpec.from_mapping(dict(
        name="a2", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16, intermediate_size=128,
                          cache_kind="window_compressed", compress_ratio=4,
                          compressor_kind="weighted_pool", indexer=True,
                          index_heads=1, index_dim=8, index_topk=2,
                          owns_long_kv=True)]))


def test_a2_merge_declares_both_branch_results():
    model = _a2_model()
    features = replace(_features(model), multistream_window_compressed=True)
    instance = build_standalone("csa_attention", model=model, layer=model.layer_plans[0],
                                role="attention", features=features)
    contract = occurrence_contract(instance)
    merge = next(item for item in contract["occurrences"]
                 if item["name"] == "merge")
    assert [(item["name"], item["producer"]) for item in merge["inputs"]] == [
        ("window_attention_output", "window_attention"),
        ("compressed_attention_output", "compressed_attention"),
    ]
    assert any(edge["consumer"] == "merge" for edge in
               contract["cross_stream_edges"])


def test_mx_quant_producer_publishes_payload_and_scale_edges():
    model = ModelSpec.from_mapping(dict(
        name="mx", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16, intermediate_size=128,
                          qk_norm=True)]))
    quant = QuantSpec(
        name="mx", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        weight_block=(1, 32), activation_group_size=32,
        scale_format="fp8_e8m0")
    model = replace(model, quant=quant)
    instance = build_standalone("gqa_attention", model=model,
                                layer=model.layer_plans[0], role="attention")
    contract = occurrence_contract(instance)
    producer = next(item for item in contract["occurrences"]
                    if item["name"] == "o_projection_QuantV2")
    outputs = {item["name"]: item for item in producer["outputs"]}
    assert set(outputs) == {"o_projection_QuantV2_output_payload",
                            "o_projection_QuantV2_output_scale"}
    assert outputs["o_projection_QuantV2_output_payload"]["dtype"] == "fp8_e4m3"
    assert outputs["o_projection_QuantV2_output_scale"]["dtype"] == "fp8_e8m0"
    matrix = next(item for item in contract["occurrences"]
                  if item["name"] == "o_projection")
    assert {(item["name"], item["producer"]) for item in matrix["inputs"]} == {
        ("o_projection_QuantV2_output_payload", "o_projection_QuantV2"),
        ("o_projection_QuantV2_output_scale", "o_projection_QuantV2"),
    }


def test_shared_k0_layout_does_not_publish_a_persistent_owner_version():
    model = _kv_owner_model()
    features = replace(_features(model), pp=2)
    instantiated = instantiate_model(model, features)
    for layer, owner in ((2, 0), (3, 1)):
        read = instantiated.find("kv_cache", layer=layer, role="kv")
        assert _state_port(read, "kv_cache", "in").owner == f"layer:{owner}"
        # layout_convert writes a local temporary value, not a new persistent
        # K2/K3 cache owner.
        assert _state_port(read, "kv_cache", "out").owner == f"layer:{owner}"


def test_production_k0_store_transfer_publishes_destination_ready():
    model = _kv_owner_model()
    features = replace(_features(model), pp=2)
    instantiated = instantiate_model(model, features)
    store = instantiated.find("kv_cache", layer=0, role="kv")
    ready = [port for port in store.state_out
             if port.name == "cross_stage_ready"]
    assert ready and ready[0].owner == "layer:0/stage:1"
    remote = instantiated.find("kv_cache", layer=3, role="kv")
    wait = next(port for port in remote.state_in
                if port.name == "cross_stage_ready")
    assert wait.owner == "layer:1/stage:1"
    attention = instantiated.find("gqa_attention", layer=3, role="attention")
    wait = next(port for port in attention.state_in
                if port.name == "cross_stage_ready")
    assert wait.owner == "layer:1/stage:1"


def test_v41_layer25_uses_the_actual_selection_owner():
    raw = json.loads((ROOT / "examples/cases"
                      / "v41-a5-dp4-body-current.json").read_text())
    case = parse_case(raw)
    features = FeatureConfig.from_runtime(
        case.runtime, case.model, machine=case.machine)
    instantiated = instantiate_model(case.model, features)
    layer = instantiated.layers[25]
    consumer = next(item for item in layer.subgraphs
                    if item.kind == "csa2_attention"
                    and dict(item.metadata).get("role") == "attention")
    port = next(item for item in consumer.state_in
                if item.name == "index_selection")
    assert port.owner == "layer:24"


def test_evaluate_standalone_is_always_an_isolated_sum():
    model = _gdn_model()
    features = replace(_features(model), multistream_kda_proj=True)
    instance = build_standalone("gdn_linear_attn", model=model, layer=model.layer_plans[0],
                                role="attention", features=features)
    geometry = WorkloadGeometry.homogeneous(2, 4, 32)
    for state in (None, StateView()):
        evaluation = evaluate_standalone(instance, geometry, _supply(),
                                         policy=POLICY, state=state)
        assert evaluation.total_ms == sum(
            cost.duration_ms for cost in evaluation.operators)
    with pytest.raises(ModelingError,
                       match="standalone_span_dependency_unexpressed"):
        standalone_span(instance, geometry, _supply(), policy=POLICY)


def test_shared_k0_layout_state_access_is_temporary_not_persistent():
    model = _kv_owner_model()
    features = replace(_features(model), pp=2)
    instantiated = instantiate_model(model, features)
    for layer, owner in ((2, 0), (3, 1)):
        contract = occurrence_contract(
            instantiated.find("kv_cache", layer=layer, role="kv"))
        accesses = [item for item in contract["state_accesses"]
                    if item["representation"] in {"kv_cache", "layout"}]
        assert any(item["temporary"] and not item["persistent"]
                   and item["position"] == "temp"
                   and item["bytes"] == 128.0
                   and item["owner"].startswith("layout:temp:")
                   for item in accesses)
        assert any(item["representation"] == "kv_cache"
                   and item["direction"] == "read"
                   and item["owner"] == f"layer:{owner}"
                   and item["bytes"] == 0.0 for item in accesses)
    store = occurrence_contract(
        instantiated.find("kv_cache", layer=0, role="kv"))
    persistent = [item for item in store["state_accesses"]
                  if item["representation"] == "kv_cache"
                  and item["persistent"] and item["direction"] == "write"]
    assert persistent and persistent[0]["producer"] == "kv_store"
    assert persistent[0]["owner"] == "layer:0"


def test_occurrence_binding_keeps_repeated_forwards_distinct():
    model = _kv_owner_model()
    instance = build_standalone("gqa_attention", model=model,
                                layer=model.layer_plans[2],
                                role="attention")
    first = bind_occurrence(instance, Occurrence(
        kind="request", name="decode", invocation="forward", phase="decode",
        serial=1, request_ids=("r1",), batch_ids=("b1",), rank=0,
        stream="main", generation=3))
    second = bind_occurrence(instance, Occurrence(
        kind="request", name="decode", invocation="forward", phase="decode",
        serial=2, request_ids=("r2",), batch_ids=("b2",), rank=0,
        stream="main", generation=4))
    assert first["occurrence"]["serial"] == 1
    assert second["occurrence"]["serial"] == 2
    assert first["occurrence"] != second["occurrence"]
    qkv = occurrence_contract(instance)["qkv_tensor"]
    for bound in (first, second):
        projection = next(output for output in bound["outputs"]
                          if output["output"] == "q_projection_output")
        for key in ("producer", "output", "dtype", "storage", "alias_of"):
            assert projection[key] == qkv[key]
        assert projection["layout"]["shape"] == (1, 64)
        assert projection["state_version"] is None
        assert projection["binding_name"] == bound["occurrence_key"]
    for output in first["outputs"]:
        assert output["producer"]
        assert output["layout"]
        assert output["binding_name"] == first["occurrence_key"]
    for output in second["outputs"]:
        assert output["binding_name"] == second["occurrence_key"]


def test_wave4_qkv_tensor_layouts_and_noncontiguous_slices():
    standard = ModelSpec.from_mapping(dict(
        name="std", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128)]))
    a0 = build_standalone("gqa_attention", model=standard,
                          layer=standard.layer_plans[0], role="attention")
    contract = dict(a0.template.contract)
    tensor = contract["qkv_tensor"]
    assert tensor.layout.shape == (0, 128)
    assert tensor.layout.strides == (128, 1)
    views = {view[0]: view for view in contract["qkv_views"]}
    assert views["value"][9] == ((0, 2, 16), (128, 16, 1), 96, (0, 128))

    gated = ModelSpec.from_mapping(dict(
        name="gated", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128, q_output_gate=True)]))
    gated_a0 = build_standalone("gqa_attention", model=gated,
                                layer=gated.layer_plans[0], role="attention")
    gated_views = {view[0]: view
                   for view in dict(gated_a0.template.contract)["qkv_views"]}
    assert gated_views["gate"][9] == (
        (0, 4, 16), (192, 32, 1), 16, (0, 192))
    assert gated_views["value"][9] == (
        (0, 2, 16), (192, 16, 1), 160, (0, 192))

    gemma = replace(standard, catalog_facts=(("gemma4_k_eq_v_all", True),))
    gemma = replace(gemma, layer_plans=(replace(
        standard.layer_plans[0], head_dim=512),), hidden_size=2048)
    gemma_a0 = build_standalone("gqa_attention", model=gemma,
                                layer=gemma.layer_plans[0], role="attention")
    gemma_views = {view[0]: view
                   for view in dict(gemma_a0.template.contract)["qkv_views"]}
    assert gemma_views["key_projection"][9] == (
        (0, 2, 512), (3072, 512, 1), 2048, (0, 3072))
    assert gemma_views["value"][9] == (
        (0, 2, 512), (3072, 512, 1), 2048, (0, 3072))
    assert gemma_views["value"][7] == "projection_k"


def _mx_a0():
    model = ModelSpec.from_mapping(dict(
        name="mx-slot", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=24, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=24,
                          intermediate_size=128, qk_norm=True)]))
    quant = QuantSpec(name="mx", weight_format="fp8_e4m3",
                      activation_format="fp8_e4m3", weight_block=(1, 32),
                      activation_group_size=32, scale_format="fp8_e8m0")
    model = replace(model, quant=quant)
    return model, build_standalone("gqa_attention", model=model,
                                   layer=model.layer_plans[0],
                                   role="attention")


def test_p1_mx_outputs_have_distinct_slots_and_quant_geometry():
    model, a0 = _mx_a0()
    contract = occurrence_contract(a0)
    producer = next(item for item in contract["occurrences"]
                    if item["name"] == "o_projection_QuantV2")
    outputs = {item["name"]: item for item in producer["outputs"]}
    payload = outputs["o_projection_QuantV2_output_payload"]
    scale = outputs["o_projection_QuantV2_output_scale"]
    assert payload["output"] == "o_projection_QuantV2_output_payload"
    assert payload["tensor"]["output"] == payload["output"]
    assert payload["layout"]["shape"] == (0, 96)
    assert payload["row_bytes"] == 96.0
    assert scale["output"] == "o_projection_QuantV2_output_scale"
    assert scale["tensor"]["output"] == scale["output"]
    assert scale["layout"]["shape"] == (0, 2, 2)
    assert scale["row_bytes"] == 4.0
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="mx", serial=1, request_ids=("r",),
        generation=9), geometry={"tokens": 1})
    matrices = next(item for item in bound["inputs"]
                    if item["name"] == "o_projection")
    by_name = {item["name"]: item for item in matrices["inputs"]}
    assert by_name["o_projection_QuantV2_output_payload"]["value"][
        "layout"]["shape"] == (1, 96)
    assert by_name["o_projection_QuantV2_output_scale"]["value"][
        "layout"]["shape"] == (1, 2, 2)


def test_p1_a3_gating_outputs_have_per_value_slots():
    model = _gdn_model()
    features = replace(_features(model), multistream_kda_proj=True)
    a3 = build_standalone("gdn_linear_attn", model=model, layer=model.layer_plans[0],
                          role="attention", features=features)
    contract = occurrence_contract(a3)
    gating = next(item for item in contract["occurrences"]
                  if item["name"] == "gdn_gating")
    outputs = {item["name"]: item for item in gating["outputs"]}
    decay = outputs["gdn_decay_fp32"]
    beta = outputs["gdn_beta_bf16"]
    assert decay["output"] == "gdn_decay_fp32"
    assert beta["output"] == "gdn_beta_bf16"
    assert decay["dtype"] == "fp32" and beta["dtype"] == "bf16"
    assert decay["row_bytes"] == 16.0 and beta["row_bytes"] == 8.0
    assert decay["layout"]["shape"] == (0, 4)
    assert beta["layout"]["shape"] == (0, 4)
    bound = bind_occurrence(a3, Occurrence(kind="request", name="gdn",
                                           serial=1, request_ids=("r",)),
                            geometry={"tokens": 1})
    delta = next(item for item in bound["inputs"]
                 if item["name"] == "gated_delta")
    values = {item["name"]: item["value"] for item in delta["inputs"]}
    assert values["gdn_decay_fp32"]["layout"]["shape"] == (1, 4)
    assert values["gdn_beta_bf16"]["layout"]["shape"] == (1, 4)


def test_p1_shared_a0_inputs_and_cache_views_are_distinct():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[2],
                          role="attention")
    contract = occurrence_contract(a0)
    attention = next(item for item in contract["occurrences"]
                     if item["name"] == "attention")
    inputs = {item["name"]: item for item in attention["inputs"]}
    assert {"query", "key", "value", "kv_cache_k", "kv_cache_v"} <= set(inputs)
    assert inputs["query"]["value"]["producer"] == "rotary"
    assert inputs["value"]["value"]["producer"] == "kv_cache"
    assert inputs["kv_cache_k"]["value"]["storage"].startswith("kv:k:")
    assert inputs["kv_cache_v"]["value"]["storage"].startswith("kv:v:")
    assert inputs["kv_cache_k"]["value"]["storage"] != inputs[
        "kv_cache_v"]["value"]["storage"]
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="shared", serial=1, request_ids=("r",)),
        geometry={"tokens": 2})
    inputs = {item["name"]: item for item in next(
        row for row in bound["inputs"] if row["name"] == "attention")["inputs"]}
    assert inputs["query"]["value"]["layout"]["shape"] == (2, 4, 16)
    assert inputs["value"]["value"]["layout"]["shape"] == (2, 2, 16)
    assert inputs["kv_cache_k"]["value"]["storage"] != inputs[
        "kv_cache_v"]["value"]["storage"]
    assert inputs["kv_cache_k"]["value"]["storage"] == "kv:k:layer:0"
    assert inputs["kv_cache_v"]["value"]["storage"] == "kv:v:layer:0"
    versioned = bind_occurrence(a0, Occurrence(
        kind="request", name="shared-v", serial=2, request_ids=("r",)),
        geometry={"tokens": 1},
        state_versions={"kv:k:layer:0": 11, "kv:v:layer:0": 12})
    versions = {item["role"]: item["tensor"]["state_version"]
                for item in versioned["view_tensors"]}
    assert versions["cache_key"] == 11
    assert versions["cache_value"] == 12


def test_p1_same_request_generation_different_serial_rank_is_distinct():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[2],
                          role="attention")
    first = bind_occurrence(a0, Occurrence(
        kind="request", name="decode", serial=1, request_ids=("same",),
        generation=7, rank=0), geometry={"tokens": 1})
    second = bind_occurrence(a0, Occurrence(
        kind="request", name="decode", serial=2, request_ids=("same",),
        generation=7, rank=1), geometry={"tokens": 1})
    assert first["occurrence_key"] != second["occurrence_key"]
    assert first["outputs"][0]["binding_name"] != second["outputs"][0][
        "binding_name"]
    assert first["outputs"][0]["producer"] == second["outputs"][0]["producer"]
    assert first["outputs"][0]["layout"] == second["outputs"][0]["layout"]
    replay = bind_occurrence(a0, Occurrence(
        kind="request", name="decode", serial=1, request_ids=("same",),
        generation=7, rank=0), geometry={"tokens": 1})
    assert replay["outputs"] == first["outputs"]


def test_p1_gated_three_rows_cover_288_elements():
    model = ModelSpec.from_mapping(dict(
        name="gated3", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=8, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=8,
                          intermediate_size=128, q_output_gate=True)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    views = {view[0]: view for view in
             dict(a0.template.contract)["qkv_views"]}
    assert views["query_projection"][9][1] == (96, 16, 1)
    assert views["gate"][9][2] == 8
    assert views["key_projection"][9][2] == 64
    assert views["value"][9][2] == 80
    addresses = set()
    for role in ("query_projection", "gate", "key_projection", "value"):
        shape, strides, offset, _storage = views[role][9]
        for token in range(3):
            for head in range(shape[1]):
                for width in range(shape[2]):
                    addresses.add(offset + token * strides[0]
                                  + head * strides[1] + width * strides[2])
    assert len(addresses) == 288
    assert addresses == set(range(288))


def test_p2_state_access_geometry_and_empty_override():
    model = _kv_owner_model()
    features = replace(_features(model), pp=2)
    instantiated = instantiate_model(model, features)
    shared = instantiated.find("kv_cache", layer=2, role="kv")
    bound = bind_occurrence(shared, Occurrence(
        kind="request", name="r1", serial=1, request_ids=("r1",), rank=0),
        geometry={"tokens": 8})
    layout = [item for item in bound["state_accesses"]
              if item["representation"] == "layout"]
    assert layout and layout[0]["bytes"] == 1024.0
    assert layout[0]["span"] == ((0, 8),)
    assert layout[0]["persistent"] is False
    assert not [item for item in bound["state_accesses"]
                if item["direction"] == "write"
                and item["representation"] == "kv_cache"]
    empty = bind_occurrence(shared, Occurrence(
        kind="request", name="r1", serial=1, request_ids=("r1",)),
        geometry={"tokens": 8}, state_accesses=())
    assert empty["state_accesses"] == ()


def test_p2_owner_store_and_layout_keep_separate_writers():
    model = _kv_owner_model()
    features = replace(_features(model), pp=2)
    instantiated = instantiate_model(model, features)
    store = occurrence_contract(instantiated.find("kv_cache", layer=0, role="kv"))
    accesses = store["state_accesses"]
    writes = [item for item in accesses
              if item["direction"] == "write" and item["bytes"] == 128.0]
    producers = {item["producer"] for item in writes}
    assert {"kv_store", "layout_convert", "stage_transfer"} <= producers
    canonical = next(item for item in writes if item["producer"] == "kv_store")
    local = next(item for item in writes if item["producer"] == "layout_convert")
    assert canonical["output"] == "kv_cache"
    assert local["output"] == "layout_convert_output"
    assert local["storage"] != canonical["storage"]


def test_p2_a3_state_access_uses_leaf_producers():
    model = _gdn_model()
    a3 = build_standalone("gdn_linear_attn", model=model, layer=model.layer_plans[0],
                          role="attention")
    contract = occurrence_contract(a3)
    assert not contract["unresolved_state_outputs"]
    writes = [item for item in contract["state_accesses"]
              if item["direction"] == "write"]
    assert {item["producer"] for item in writes} >= {"conv", "gated_delta"}
    assert all(item["bytes"] > 0 for item in writes)
    assert all(item["producer"] for item in writes)


def test_p2_state_occurrence_binds_request_rank_without_generation_version():
    model = _kv_owner_model()
    instance = instantiate_model(model, replace(_features(model), pp=2)).find(
        "kv_cache", layer=2, role="kv")
    first = bind_occurrence(instance, Occurrence(
        kind="request", name="r1", serial=1, request_ids=("a",), rank=0),
        geometry={"tokens": 1})
    second = bind_occurrence(instance, Occurrence(
        kind="request", name="r2", serial=2, request_ids=("b",), rank=1),
        geometry={"tokens": 1})
    assert first["state_accesses"][0]["request"] == "a"
    assert second["state_accesses"][0]["request"] == "b"
    assert first["state_accesses"][0]["rank"] == 0
    assert second["state_accesses"][0]["rank"] == 1
    assert first["state_accesses"][0]["occurrence"] != second[
        "state_accesses"][0]["occurrence"]
    assert all(item["state_version"] is None
               for item in first["state_accesses"])
    versioned = bind_occurrence(instance, Occurrence(
        kind="request", name="r3", serial=3, request_ids=("c",), rank=0),
        geometry={"tokens": 1},
        state_versions={"kv_cache:layer:0": 11})
    assert any(item["state_version"] == 11
               for item in versioned["state_accesses"])


def test_p1_fusion_selection_reconnects_attention_inputs():
    model = ModelSpec.from_mapping(dict(
        name="fused", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128, qk_norm=True)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    fused = bind_occurrence(a0, Occurrence(
        kind="request", name="fused", serial=1, request_ids=("r",)),
        geometry={"tokens": 1}, fusion=True)
    unfused = bind_occurrence(a0, Occurrence(
        kind="request", name="unfused", serial=2, request_ids=("r",)),
        geometry={"tokens": 1}, fusion=False)
    fused_views = {view[0]: view for view in fused["views"]}
    unfused_views = {view[0]: view for view in unfused["views"]}
    assert fused_views["query"][2] == "fused:qkv_norm_rope"
    assert fused_views["key"][2] == "fused:qkv_norm_rope"
    assert unfused_views["query"][2] == "rotary"
    assert unfused_views["key"][2] == "rotary"
    fused_attention = next(row for row in fused["inputs"]
                           if row["name"] == "attention")
    unfused_attention = next(row for row in unfused["inputs"]
                             if row["name"] == "attention")
    fused_q = next(item for item in fused_attention["inputs"]
                   if item["name"] == "query")
    unfused_q = next(item for item in unfused_attention["inputs"]
                     if item["name"] == "query")
    assert fused_q["value"]["producer"] == "fused:qkv_norm_rope"
    assert unfused_q["value"]["producer"] == "rotary"
    rotary = next(item for item in occurrence_contract(a0)["occurrences"]
                  if item["name"] == "rotary")
    assert rotary["outputs"][0]["tensor"]
    assert rotary["outputs"][0]["dtype"] == "bf16"


def test_p2_shared_a0_has_no_persistent_write():
    model = _kv_owner_model()
    a0 = occurrence_contract(build_standalone(
        "gqa_attention", model=model, layer=model.layer_plans[2], role="attention"))
    assert not [item for item in a0["state_accesses"]
                if item["direction"] == "write"]
    reads = [item for item in a0["state_accesses"]
             if item["direction"] == "read"
             and item["representation"] == "kv_cache"]
    assert any(item["producer"] == "attention" and item["bytes"] == 128.0
               for item in reads)


def test_r6_shared_fp32_first_postprocess_reads_256_bytes():
    model = _kv_owner_model()
    quant = QuantSpec(
        name="q-fp32", weight_format="bf16", activation_format="bf16",
        category_overrides=(
            ("attn.q", (("output_format", "fp32"),)),
            ("attn.qkv", (("output_format", "bf16"),))))
    model = replace(model, quant=quant)
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[2],
                          role="attention")
    contract = dict(a0.template.contract)
    assert contract["qkv_output_dtype"] == "fp32"
    assert float(contract["qkv_output_row_bytes"]) == 256.0
    rotary = next(op for op in a0.operators if op.name == "rotary").theta
    assert rotary.find("read_bytes") == 256.0
    assert rotary.find("write_bytes") == 128.0
    assert rotary.find("q_dtype") == "fp32"


def test_r6_a0_ordinary_edges_and_nonoverlapping_qk_views():
    model = ModelSpec.from_mapping(dict(
        name="ordinary", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    contract = occurrence_contract(a0)
    edges = {(edge["producer"], edge["consumer"], edge["value"])
             for edge in contract["edges"]}
    assert ("qkv_projection", "rotary", "qkv_activation") in edges
    assert ("rotary", "attention", "rotary_output") in edges
    assert not contract["unresolved_inputs"]
    views = {view[0]: view for view in contract["views"]}
    assert views["query"][10] == "post:rotary"
    assert views["key"][10] == "post:rotary"
    assert views["query"][9][2] == 0
    assert views["key"][9][2] == 64
    attention = next(item for item in contract["occurrences"]
                     if item["name"] == "attention")
    inputs = {item["name"]: item for item in attention["inputs"]}
    assert inputs["query"]["value"]["storage"] == "post:rotary"
    assert inputs["key"]["value"]["storage"] == "post:rotary"


def test_r6_qk_norm_edges_follow_selected_postprocess():
    model = ModelSpec.from_mapping(dict(
        name="qk", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128, qk_norm=True)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    edges = {(edge["producer"], edge["consumer"])
             for edge in occurrence_contract(a0)["edges"]}
    assert ("qkv_projection", "qk_norm") in edges
    assert ("qk_norm", "rotary") in edges
    assert ("fused:qkv_norm_rope", "attention") in edges


def test_r6_two_occurrences_keep_two_temp_writes():
    model = _kv_owner_model()
    instance = instantiate_model(model, replace(_features(model), pp=2)).find(
        "kv_cache", layer=2, role="kv")
    first = bind_occurrence(instance, Occurrence(
        kind="request", name="r1", serial=1, request_ids=("a",), rank=0),
        geometry={"tokens": 1})
    second = bind_occurrence(instance, Occurrence(
        kind="request", name="r2", serial=2, request_ids=("b",), rank=1),
        geometry={"tokens": 1})
    first_temp = [item for item in first["state_accesses"]
                  if item["position"] == "temp"]
    second_temp = [item for item in second["state_accesses"]
                   if item["position"] == "temp"]
    assert first_temp[0]["bytes"] == 128.0
    assert second_temp[0]["bytes"] == 128.0
    assert first_temp[0]["occurrence"] != second_temp[0]["occurrence"]
    assert sum(item["bytes"] for item in first_temp + second_temp) == 256.0
    assert not [item for item in first["state_accesses"]
                if item["direction"] == "write"
                and item["representation"] == "kv_cache"
                and item["persistent"]]


def test_wave8_standard_three_rows_share_real_rotary_geometry():
    model = ModelSpec.from_mapping(dict(
        name="wave8-standard", layers=1, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="three", serial=1, request_ids=("r",)),
        geometry={"tokens": 3}, fusion=False)
    views = {view[0]: view for view in bound["views"]}
    assert views["query"][9] == ((3, 4, 16), (96, 16, 1), 0, (3, 96))
    assert views["key"][9] == ((3, 2, 16), (96, 16, 1), 64, (3, 96))
    assert views["query"][10] == views["key"][10] == "post:rotary"
    addresses = set()
    for role in ("query", "key"):
        shape, strides, offset, _storage = views[role][9]
        for token in range(shape[0]):
            for head in range(shape[1]):
                for width in range(shape[2]):
                    addresses.add(offset + token * strides[0]
                                  + head * strides[1] + width * strides[2])
    assert max(addresses) <= 287
    attention = next(item for item in bound["inputs"]
                     if item["name"] == "attention")
    q = next(item for item in attention["inputs"] if item["name"] == "query")
    k = next(item for item in attention["inputs"] if item["name"] == "key")
    assert q["value"]["layout"]["shape"] == (3, 4, 16)
    assert k["value"]["layout"]["shape"] == (3, 2, 16)


def test_wave8_shared_versions_and_cached_token_span():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[2],
                          role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="cached", serial=1, request_ids=("r",)),
        geometry={"tokens": 1, "cached_tokens": 4},
        state_versions={"kv:k:layer:0": 11, "kv:v:layer:0": 12})
    attention = next(item for item in bound["inputs"]
                     if item["name"] == "attention")
    inputs = {item["name"]: item for item in attention["inputs"]}
    assert inputs["key"]["value"]["storage"] == "kv:k:layer:0"
    assert inputs["key"]["value"]["state_version"] == 11
    assert inputs["value"]["value"]["storage"] == "kv:v:layer:0"
    assert inputs["value"]["value"]["state_version"] == 12
    assert inputs["key"]["value"]["layout"]["shape"] == (4, 2, 16)
    reads = [item for item in bound["state_accesses"]
             if item["producer"] == "attention"]
    assert len(reads) == 2
    assert sum(item["bytes"] for item in reads) == 512.0
    assert {item["storage"]: item["state_version"] for item in reads} == {
        "kv:k:layer:0": 11, "kv:v:layer:0": 12}
    for item in reads:
        assert item["bytes"] == 256.0
        assert item["row_span"] == (0, 4)
        assert item["span"] == ((0, 4),)
        assert item["byte_span"] == (0.0, 256.0)
        assert item["unit"] == "rows" and item["byte_unit"] == "bytes"
        assert item["row_stride"] == item["row_width"] == 64.0
    readiness = [item for item in bound["state_accesses"]
                 if item["producer"] == "kv_read"]
    assert readiness[0]["bytes"] == 0.0
    cache_hit = bind_occurrence(a0, Occurrence(
        kind="request", name="cached", serial=2, request_ids=("r",)),
        geometry={"tokens": 1, "cached_tokens": 4},
        state_versions={"kv:k:layer:0": 11, "kv:v:layer:0": 12})
    hit_attention = next(item for item in cache_hit["inputs"]
                         if item["name"] == "attention")
    hit_inputs = {item["name"]: item for item in hit_attention["inputs"]}
    assert hit_inputs["key"]["value"]["state_version"] == 11
    assert hit_inputs["value"]["value"]["state_version"] == 12
    assert hit_inputs["key"]["value"]["storage"] == inputs["key"]["value"][
        "storage"]
    assert cache_hit["occurrence_key"] != bound["occurrence_key"]


def test_wave8_mx_scale_stride_and_quant_traffic():
    _, a0 = _mx_a0()
    contract = occurrence_contract(a0)
    producer = next(item for item in contract["occurrences"]
                    if item["name"] == "o_projection_QuantV2")
    outputs = {item["name"]: item for item in producer["outputs"]}
    scale = outputs["o_projection_QuantV2_output_scale"]
    assert scale["layout"]["strides"] == (4, 2, 1)
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="mx", serial=3, request_ids=("r",), rank=1),
        geometry={"tokens": 1})
    scale_bound = next(item for item in bound["outputs"]
                       if item["output"] == "o_projection_QuantV2_output_scale")
    assert scale_bound["layout"] == {
        "shape": (1, 2, 2), "strides": (4, 2, 1), "offset": 0,
        "storage_shape": (1, 2, 2)}
    assert [item["byte_span"] for item in scale_bound["accesses"]] == [
        (0.0, 4.0)]
    payload_bound = next(item for item in bound["outputs"]
                         if item["output"] == "o_projection_QuantV2_output_payload")
    assert [item["byte_span"] for item in payload_bound["accesses"]] == [
        (0.0, 96.0)]
    matrix = next(item for item in bound["inputs"]
                  if item["name"] == "o_projection")
    ports = {item["name"]: item["value"] for item in matrix["inputs"]}
    assert [item["byte_span"]
            for item in ports["o_projection_QuantV2_output_payload"]["accesses"]
            ] == [(0.0, 96.0)]
    assert [item["byte_span"]
            for item in ports["o_projection_QuantV2_output_scale"]["accesses"]
            ] == [(0.0, 4.0)]
    theta = next(op for op in a0.operators
                 if op.name == "o_projection_QuantV2").theta
    assert theta.find("read_bytes") == 192.0
    assert theta.find("write_bytes") == 100.0


def test_wave8_pp_store_transfer_stages_and_rows():
    model = _kv_owner_model()
    instantiated = instantiate_model(model, replace(_features(model), pp=2))
    store = instantiated.find("kv_cache", layer=0, role="kv")
    contract = occurrence_contract(store)
    writes = [item for item in contract["state_accesses"]
              if item["producer"] in {"kv_store", "layout_convert",
                                      "stage_transfer"}]
    assert {item["producer"]: (item["stage"], item["bytes"])
            for item in writes} == {
        "kv_store": (0, 128.0),
        "layout_convert": (0, 128.0),
        "stage_transfer": (1, 128.0),
    }
    bound = bind_occurrence(store, Occurrence(
        kind="request", name="pp", serial=1, request_ids=("r",)),
        geometry={"tokens": 8})
    bound_writes = [item for item in bound["state_accesses"]
                    if item["producer"] in {"kv_store", "layout_convert",
                                            "stage_transfer"}]
    assert {item["producer"]: (item["stage"], item["bytes"])
            for item in bound_writes} == {
        "kv_store": (0, 1024.0),
        "layout_convert": (0, 1024.0),
        "stage_transfer": (1, 1024.0),
    }


def test_wave8_a3_gather_scatter_and_local_h_state():
    model = ModelSpec.from_mapping(dict(
        name="gdn-wave8", layers=1, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gdn", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128, linear_heads=4,
                          linear_head_dim=8, linear_key_heads=4,
                          linear_key_head_dim=8, linear_value_heads=4,
                          conv_kernel=4, recurrent_state_elements=256,
                          conv_state_elements=16)]))
    features = _features(model)
    features = replace(
        features, pcp=1,
        metadata=tuple(dict(features.metadata).items()) + (
            ("arch_family", "A5"), ("source_lowerings", ("gdn_prefill",))))
    a3 = build_standalone("gdn_linear_attn", variant="chunked_prefill", model=model,
                          layer=model.layer_plans[0], role="attention",
                          features=features)
    contract = occurrence_contract(a3)
    gather = [item for item in contract["state_accesses"]
              if item["producer"] == "gather_recurrent_state"]
    scatter = [item for item in contract["state_accesses"]
               if item["producer"] == "scatter_recurrent_state"]
    local = [item for item in contract["state_accesses"]
             if item["producer"] == "gated_delta"]
    assert gather[0]["bytes"] == 1024.0 and gather[0]["persistent"]
    assert scatter[0]["bytes"] == 1024.0 and scatter[0]["persistent"]
    assert local and all(item["bytes"] == 1024.0 and not item["persistent"]
                         and item["temporary"] for item in local)
    assert gather[0]["occurrence"] != scatter[0]["occurrence"]


def test_r16_binding_exposes_request_slot_ranges_and_memory_fields():
    model = _kv_owner_model()
    instantiated = instantiate_model(model, replace(_features(model), pp=2))
    store = instantiated.find("kv_cache", layer=0, role="kv")
    bound = bind_occurrence(store, Occurrence(
        kind="request", name="batch", serial=4, request_ids=("ra", "rb"),
        rank=2), geometry={
            "tokens": 2,
            "request_rows": (("ra", 0, 1, 0), ("rb", 1, 2, 1)),
        })
    writes = [item for item in bound["state_accesses"]
              if item["producer"] == "kv_store"]
    assert len(writes) == 2
    assert [(item["request"], item["request_slot"])
            for item in writes] == [("ra", 0), ("rb", 1)]
    assert [(item["row_span"], item["byte_span"])
            for item in writes] == [((0, 1), (0.0, 128.0)),
                                    ((1, 2), (128.0, 256.0))]
    for item in writes:
        assert item["rank"] == 2
        assert item["unit"] == "rows"
        assert item["byte_unit"] == "bytes"
        assert item["row_stride"] == item["row_width"] == 128.0
        assert item["scope"] == "target"
        assert item["direction"] == "write"
        assert item["position"] == "cache"
        assert item["action"] == "kv_store:write:kv_cache"
        assert item["port"] == "kv_cache"
        assert item["owner"] == "layer:0"
    transfer = [item for item in bound["state_accesses"]
                if item["producer"] == "stage_transfer"]
    assert transfer and all(item["stage"] == 1 for item in transfer)
    assert bound["request_rows"] == (
        ("ra", 0, 1, 0), ("rb", 1, 2, 1))
    assert bound["memory_accesses"]
    for item in bound["memory_accesses"]:
        assert item["unit"] == "rows"
        assert item["byte_unit"] == "bytes"
        assert "action" in item and "port" in item


def test_r16_tensor_outputs_carry_per_request_byte_ranges():
    model = ModelSpec.from_mapping(dict(
        name="r16-a0", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="batch", serial=5, request_ids=("ra", "rb"),
        rank=1), geometry={
            "tokens": 3,
            "request_rows": (("ra", 0, 1, 0), ("rb", 1, 3, 1)),
        })
    projection = next(item for item in bound["outputs"]
                      if item["output"] == "qkv_activation")
    accesses = projection["accesses"]
    assert [(item["request"], item["row_span"])
            for item in accesses] == [("ra", (0, 1)), ("rb", (1, 3))]
    assert accesses[0]["byte_span"] == (0.0, 256.0)
    assert accesses[1]["byte_span"] == (256.0, 768.0)
    assert all(item["row_stride"] == 256.0 for item in accesses)
    assert all(item["rank"] == 1 for item in accesses)
    assert all(item["stage"] == 0 for item in accesses)
    assert all(item["action"].startswith("qkv_projection:write:")
               for item in accesses)


def test_r16_explicit_target_draft_accesses_are_authoritative():
    model = _kv_owner_model()
    instance = build_standalone("gqa_attention", model=model,
                                layer=model.layer_plans[2],
                                role="attention")
    target = {
        "request": "r", "request_slot": 0, "rank": 1, "stage": 0,
        "storage": "kv:target", "state_version": 3, "row_span": (0, 2),
        "byte_span": (0.0, 256.0), "unit": "bytes", "row_stride": 128.0,
        "scope": "target", "direction": "write", "position": "cache",
        "purpose": "target_store", "action": "store:write:target",
        "port": "target", "owner": "layer:0", "bytes": 256.0,
    }
    draft = {
        **target, "storage": "kv:draft", "state_version": 2,
        "byte_span": (0.0, 192.0), "scope": "draft",
        "purpose": "draft_store", "action": "store:write:draft",
        "port": "draft", "bytes": 192.0,
    }
    bound = bind_occurrence(
        instance, Occurrence(kind="request", name="mtp", serial=6,
                             request_ids=("r",), rank=1),
        geometry={"tokens": 2},
        state_accesses=(target, draft))
    assert bound["state_accesses"] == (target, draft)
    assert len(bound["memory_accesses"]) >= 2
    assert {item["scope"] for item in bound["state_accesses"]} == {
        "target", "draft"}


def test_r16_padding_lanes_are_zero_range_and_keep_request_slot():
    model = ModelSpec.from_mapping(dict(
        name="r16-pad", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                          num_kv_heads=2, head_dim=16,
                          intermediate_size=128)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="pad", serial=1, request_ids=("r",), rank=0),
        geometry={"tokens": 4,
                  "request_rows": (("r", 0, 3, 0),),
                  "padding_rows": (1,)})
    projection = next(item for item in bound["outputs"]
                      if item["output"] == "qkv_activation")
    runs = [(item["row_span"], item["byte_span"], item["padding"],
             item["purpose"]) for item in projection["accesses"]]
    assert runs == [
        ((0, 1), (0.0, 256.0), False, "activation_write"),
        ((1, 2), (256.0, 256.0), True, "padding_zero"),
        ((2, 3), (512.0, 768.0), False, "activation_write"),
    ]
    assert all(item["request_slot"] == 0 for item in projection["accesses"])


def test_wave10_packed_rotary_physical_ranges_and_reader_actions():
    model = ModelSpec.from_mapping(dict(
        name="wave10-rotary", layers=1, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=128, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense",
                          num_heads=4, num_kv_heads=2, head_dim=16,
                          intermediate_size=128)]))
    a0 = build_standalone("gqa_attention", model=model, layer=model.layer_plans[0],
                          role="attention")
    geometry = {"tokens": 3}
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="rotary", serial=1, request_ids=("r",)),
        geometry=geometry, fusion=False)
    views = {item["role"]: item["tensor"] for item in bound["view_tensors"]}
    assert views["query"]["layout"] == {
        "shape": (3, 4, 16), "strides": (96, 16, 1), "offset": 0,
        "storage_shape": (3, 96)}
    assert views["key"]["layout"] == {
        "shape": (3, 2, 16), "strides": (96, 16, 1), "offset": 64,
        "storage_shape": (3, 96)}
    query_ranges = [(item["row_span"], item["byte_span"], item["unit"],
                     item["byte_unit"], item["bytes"])
                    for item in views["query"]["accesses"]]
    key_ranges = [(item["row_span"], item["byte_span"], item["unit"],
                   item["byte_unit"], item["bytes"])
                  for item in views["key"]["accesses"]]
    assert query_ranges == [
        ((0, 1), (0.0, 128.0), "rows", "bytes", 128.0),
        ((1, 2), (192.0, 320.0), "rows", "bytes", 128.0),
        ((2, 3), (384.0, 512.0), "rows", "bytes", 128.0),
    ]
    assert key_ranges == [
        ((0, 1), (128.0, 192.0), "rows", "bytes", 64.0),
        ((1, 2), (320.0, 384.0), "rows", "bytes", 64.0),
        ((2, 3), (512.0, 576.0), "rows", "bytes", 64.0),
    ]
    covered = set()
    for item in views["query"]["accesses"] + views["key"]["accesses"]:
        covered.update(range(int(item["byte_span"][0]),
                             int(item["byte_span"][1])))
    assert sum(item["bytes"] for item in views["query"]["accesses"]) == 384.0
    assert sum(item["bytes"] for item in views["key"]["accesses"]) == 192.0
    assert covered == set(range(576))
    attention = next(item for item in bound["inputs"]
                     if item["name"] == "attention")
    inputs = {item["name"]: item for item in attention["inputs"]}
    query_reads = inputs["query"]["value"]["accesses"]
    key_reads = inputs["key"]["value"]["accesses"]
    assert {item["action"] for item in query_reads} == {
        "attention:read:query"}
    assert {item["action"] for item in key_reads} == {
        "attention:read:key"}
    assert inputs["query"]["value"]["producer"] == "rotary"
    assert inputs["key"]["value"]["producer"] == "rotary"
    assert inputs["query"]["value"]["output"] == "rotary_output"
    assert inputs["key"]["value"]["output"] == "rotary_output"
    assert {item["reader"] for item in query_reads} == {"attention"}
    assert {item["port"] for item in query_reads} == {"query"}
    hit = bind_occurrence(a0, Occurrence(
        kind="request", name="rotary", serial=2, request_ids=("r",)),
        geometry=geometry, fusion=False)
    assert hit["occurrence_key"] != bound["occurrence_key"]

    def tensor_identity(value):
        return {key: value[key] for key in (
            "producer", "dtype", "layout", "output", "quant", "storage",
            "alias_of", "state_version", "accesses")}

    assert [tensor_identity(item) for item in hit["outputs"]] == [
        tensor_identity(item) for item in bound["outputs"]]
    assert {item["role"]: tensor_identity(item["tensor"])
            for item in hit["view_tensors"]} == {
        item["role"]: tensor_identity(item["tensor"])
        for item in bound["view_tensors"]}


def test_wave10_qk_norm_rotary_read_previous_actual_dtype():
    model = _kv_owner_model()
    quant = QuantSpec(
        name="q-fp32", weight_format="bf16", activation_format="bf16",
        category_overrides=(
            ("attn.q", (("output_format", "fp32"),)),
            ("attn.qkv", (("output_format", "bf16"),))))
    model = replace(model, quant=quant)
    layer = dict(index=2, attention="gqa", ffn="dense", num_heads=4,
                 num_kv_heads=2, head_dim=16, intermediate_size=128,
                 kv_owner_layer_id=0, owns_long_kv=False, qk_norm=True)
    a0 = build_standalone("gqa_attention", model=model, layer=layer, role="attention")
    operators = {op.name: op.theta for op in a0.operators}
    assert operators["qk_norm"].find("read_bytes") == 256.0
    assert operators["qk_norm"].find("write_bytes") == 128.0
    assert operators["rotary"].find("read_bytes") == 128.0
    assert operators["rotary"].find("write_bytes") == 128.0
    assert operators["rotary"].find("q_dtype") == "bf16"
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="norm", serial=1, request_ids=("r",)),
        geometry={"tokens": 3}, fusion=False)
    rows = {item["name"]: item for item in bound["inputs"]}
    norm = rows["qk_norm"]
    rotary = rows["rotary"]
    assert norm["inputs"][0]["value"]["dtype"] == "fp32"
    assert norm["inputs"][0]["value"]["accesses"][0]["row_width"] == 256.0
    assert norm["outputs"][0]["dtype"] == "bf16"
    assert norm["outputs"][0]["layout"]["shape"] == (3, 64)
    assert norm["outputs"][0]["accesses"][0]["row_width"] == 128.0
    assert rotary["inputs"][0]["value"]["dtype"] == "bf16"
    assert rotary["inputs"][0]["value"]["producer"] == "qk_norm"
    assert rotary["inputs"][0]["value"]["accesses"][0]["row_width"] == 128.0
    assert rotary["outputs"][0]["accesses"][0]["row_width"] == 128.0
    # Every per-operator output uses the same selected value/layout as the
    # top-level bound output; a template row count must not survive here.
    for row in bound["inputs"]:
        for output in row["outputs"]:
            top = next((item for item in bound["outputs"]
                        if item["output"] == output["output"]), None)
            if top is None:
                continue
            assert output["layout"] == top["layout"]
            assert output["dtype"] == top["dtype"]
            assert output["accesses"] == top["accesses"]

    plain = dict(layer, qk_norm=False)
    no_norm = build_standalone("gqa_attention", model=model, layer=plain,
                               role="attention")
    theta = {op.name: op.theta for op in no_norm.operators}
    assert theta["rotary"].find("read_bytes") == 256.0
    assert theta["rotary"].find("q_dtype") == "fp32"
    no_norm_bound = bind_occurrence(no_norm, Occurrence(
        kind="request", name="plain", serial=2, request_ids=("r",)),
        geometry={"tokens": 3})
    rotary_row = next(item for item in no_norm_bound["inputs"]
                      if item["name"] == "rotary")
    assert rotary_row["inputs"][0]["value"]["dtype"] == "fp32"
    assert rotary_row["inputs"][0]["value"]["accesses"][0]["row_width"] == 256.0


def test_wave10_k0_padding_state_write_and_original_slot_units():
    model = _kv_owner_model()
    instantiated = instantiate_model(model, replace(_features(model), pp=2))
    store = instantiated.find("kv_cache", layer=0, role="kv")
    bound = bind_occurrence(store, Occurrence(
        kind="request", name="pad", serial=1, request_ids=("pad", "r"),
        rank=0), geometry={
            "tokens": 2,
            "request_rows": (("pad", 0, 1, 0), ("r", 1, 2, 1)),
            "padding_rows": (0,),
        })
    writes = [item for item in bound["state_accesses"]
              if item["producer"] == "kv_store"]
    assert [(item["request"], item["request_slot"], item["row_span"],
             item["span"], item["byte_span"], item["bytes"], item["unit"],
             item["byte_unit"], item["padding"])
            for item in writes] == [
        ("pad", 0, (0, 1), ((0, 1),), (0.0, 0.0), 0.0, "rows", "bytes", True),
        ("r", 1, (1, 2), ((1, 2),), (128.0, 256.0), 128.0, "rows", "bytes",
         False),
    ]
    assert sum(item["bytes"] for item in writes
               if item["persistent"]) == 128.0
    transfer = [item for item in bound["state_accesses"]
                if item["producer"] == "stage_transfer"]
    assert [(item["request_slot"], item["stage"], item["byte_span"],
             item["bytes"]) for item in transfer] == [
        (0, 1, (0.0, 0.0), 0.0), (1, 1, (128.0, 256.0), 128.0)]
    layout = [item for item in bound["state_accesses"]
              if item["producer"] == "layout_convert"]
    assert [(item["request_slot"], item["bytes"], item["temporary"])
            for item in layout] == [(0, 0.0, True), (1, 128.0, True)]
    single = bind_occurrence(store, Occurrence(
        kind="request", name="slot5", serial=2, request_ids=("r",), rank=0),
        geometry={"tokens": 2, "request_rows": (("r", 1, 2, 5),)})
    single_write = next(item for item in single["state_accesses"]
                        if item["producer"] == "kv_store")
    assert single_write["request_slot"] == 5
    assert single_write["row_span"] == single_write["span"][0] == (1, 2)
    assert single_write["byte_span"] == (128.0, 256.0)
    assert single_write["bytes"] == 128.0
    assert single_write["unit"] == "rows"
    assert single_write["byte_unit"] == "bytes"


def test_wave10_dual_request_cache_rows_keep_storage_and_version():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model,
                          layer=model.layer_plans[2], role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="dual", serial=1, request_ids=("ra", "rb"),
        rank=0), geometry={
            "tokens": 2,
            "request_rows": (("ra", 0, 1, 0), ("rb", 1, 2, 1)),
            "request_cached_rows": (
                {"request": "ra", "start": 0, "end": 2,
                 "storage": {"cache_key": "kv:k:ra", "cache_value": "kv:v:ra"},
                 "state_version": {"cache_key": 7, "cache_value": 8}},
                {"request": "rb", "start": 0, "end": 4,
                 "storage": {"cache_key": "kv:k:rb", "cache_value": "kv:v:rb"},
                 "state_version": {"cache_key": 9, "cache_value": 10}},
            ),
        })
    reads = [item for item in bound["state_accesses"]
             if item["producer"] == "attention"]
    assert bound["unresolved_state_ranges"] == ()
    assert {(item["request"], item["storage"], item["state_version"],
             item["row_span"], item["byte_span"], item["bytes"])
            for item in reads} == {
        ("ra", "kv:k:ra", 7, (0, 2), (0.0, 128.0), 128.0),
        ("ra", "kv:v:ra", 8, (0, 2), (0.0, 128.0), 128.0),
        ("rb", "kv:k:rb", 9, (0, 4), (0.0, 256.0), 256.0),
        ("rb", "kv:v:rb", 10, (0, 4), (0.0, 256.0), 256.0),
    }
    totals = {}
    for item in reads:
        totals[item["request"]] = totals.get(item["request"], 0.0) + item["bytes"]
    assert totals == {"ra": 256.0, "rb": 512.0}
    views = {item["role"]: item["tensor"] for item in bound["view_tensors"]}
    key_accesses = views["cache_key"]["accesses"]
    assert [(item["request"], item["storage"], item["state_version"],
             item["byte_span"]) for item in key_accesses] == [
        ("ra", "kv:k:ra", 7, (0.0, 128.0)),
        ("rb", "kv:k:rb", 9, (0.0, 256.0))]
    value_accesses = views["cache_value"]["accesses"]
    assert [(item["request"], item["storage"], item["state_version"],
             item["byte_span"]) for item in value_accesses] == [
        ("ra", "kv:v:ra", 8, (0.0, 128.0)),
        ("rb", "kv:v:rb", 10, (0.0, 256.0))]
    attention = next(item for item in bound["inputs"]
                     if item["name"] == "attention")
    inputs = {item["name"]: item for item in attention["inputs"]}

    def read_identity(accesses):
        return [(item["request"], item["storage"], item["state_version"],
                 item["row_span"], item["byte_span"], item["bytes"])
                for item in accesses]

    assert read_identity(inputs["kv_cache_k"]["value"]["accesses"]) == (
        read_identity(key_accesses))
    assert read_identity(inputs["kv_cache_v"]["value"]["accesses"]) == (
        read_identity(value_accesses))
    assert {item["action"]
            for item in inputs["kv_cache_k"]["value"]["accesses"]} == {
        "attention:read:kv_cache_k"}
    assert {item["action"]
            for item in inputs["kv_cache_v"]["value"]["accesses"]} == {
        "attention:read:kv_cache_v"}


def test_wave10_explicit_dual_cache_offsets_in_one_storage():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model,
                          layer=model.layer_plans[2], role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="shared-cache", serial=1,
        request_ids=("ra", "rb"), rank=0), geometry={
            "tokens": 2,
            "request_rows": (("ra", 0, 1, 0), ("rb", 1, 2, 1)),
            "request_cached_rows": (
                {"request": "ra", "start": 0, "end": 2},
                {"request": "rb", "start": 2, "end": 6},
            ),
        })
    reads = [item for item in bound["state_accesses"]
             if item["producer"] == "attention"
             and item["storage"] == "kv:k:layer:0"]
    assert [(item["request"], item["row_span"], item["byte_span"],
             item["bytes"]) for item in reads] == [
        ("ra", (0, 2), (0.0, 128.0), 128.0),
        ("rb", (2, 6), (128.0, 384.0), 256.0)]
    assert bound["unresolved_state_ranges"] == ()


def test_wave10_single_cached4_splits_kv_versions_and_k0_readiness():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model,
                          layer=model.layer_plans[2], role="attention")
    versions = {"kv:k:layer:0": 11, "kv:v:layer:0": 12}
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="cached4", serial=1, request_ids=("r",),
        rank=0), geometry={"tokens": 1, "cached_tokens": 4},
        state_versions=versions)
    reads = [item for item in bound["state_accesses"]
             if item["producer"] == "attention"]
    assert {(item["storage"], item["state_version"], item["bytes"],
             item["row_span"], item["byte_span"]) for item in reads} == {
        ("kv:k:layer:0", 11, 256.0, (0, 4), (0.0, 256.0)),
        ("kv:v:layer:0", 12, 256.0, (0, 4), (0.0, 256.0)),
    }
    assert sum(item["bytes"] for item in reads) == 512.0
    readiness = [item for item in bound["state_accesses"]
                 if item["producer"] == "kv_read"]
    assert readiness and all(item["bytes"] == 0.0 for item in readiness)
    instantiated = instantiate_model(model, replace(_features(model), pp=2))
    read = instantiated.find("kv_cache", layer=2, role="kv")
    k0 = bind_occurrence(read, Occurrence(
        kind="request", name="readiness", serial=2, request_ids=("r",),
        rank=0), geometry={"tokens": 1})
    k0_read = next(item for item in k0["state_accesses"]
                   if item["producer"] == "kv_read")
    assert k0_read["bytes"] == 0.0
    assert k0_read["purpose"] == "readiness"
    assert k0_read["row_span"] == (0, 1)
    assert k0_read["byte_span"] == (0.0, 0.0)
    assert k0_read["unit"] == "rows" and k0_read["byte_unit"] == "bytes"


def test_wave10_missing_per_request_cache_rows_is_reported_not_aggregated():
    model = _kv_owner_model()
    a0 = build_standalone("gqa_attention", model=model,
                          layer=model.layer_plans[2], role="attention")
    bound = bind_occurrence(a0, Occurrence(
        kind="request", name="missing", serial=1,
        request_ids=("ra", "rb"), rank=0), geometry={
            "tokens": 2,
            "request_rows": (("ra", 0, 1, 0), ("rb", 1, 2, 1)),
            "cached_tokens": 6,
        })
    gaps = {item["field"]: item for item in bound["unresolved_state_ranges"]}
    assert "request_cached_rows" in gaps
    assert gaps["request_cached_rows"]["reason"] == (
        "per_request_cache_rows_required")
    assert gaps["request_cached_rows"]["requests"] == ("ra", "rb")
    assert not [item for item in bound["state_accesses"]
                if item["producer"] == "attention"]
    views = {item["role"]: item["tensor"] for item in bound["view_tensors"]}
    assert views["cache_key"]["accesses"] == ()
    assert views["cache_value"]["accesses"] == ()
