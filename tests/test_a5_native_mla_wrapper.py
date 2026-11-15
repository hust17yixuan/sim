"""A5 native/graph MLA wrapper: real entry, bucket and mechanism perturbations.

The checks use the public target graph for the real A5 case and a minimal
builder for new geometry. Expected bytes are derived from the request views,
latent/rope widths and page size, not from a shape table or a measured total.
"""
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction import family_paged_mla, family_paged_mla_arch22
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_binding import _typed_payload
from mindie_sim.prediction.paged_mla import build as build_arch35
from mindie_sim.prediction.paged_mla_arch22 import (
    BUFFERS as ARCH22_BUFFERS,
    IMPLEMENTATION as ARCH22,
    build as build_arch22,
)
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.scheduling import FragmentConductor
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_contracts import predicted_contract

ROOT = Path(__file__).resolve().parents[1]
A5_CASE = ROOT / "examples/cases/kimi-a5-paged-host-20261030/c1.json"
A3_CASE = ROOT / (
    "examples/cases/high-concurrency-20261030/kimi-source/"
    "a3_kimi4_tp8_w4a8_c1_source.json")
ARCH22_ORACLE = json.loads(
    (ROOT / "tests/fixtures/paged_mla_arch22_dispatch.json").read_text())["cases"]


def _parsed(path=A5_CASE):
    return parse_case(json.loads(Path(path).read_text()))


def _mla(model):
    return next(plan for plan in model.layer_plans if plan.attention == "mla")


def _views(contexts):
    return tuple(RequestView(str(index), 1, context, False, "decode")
                 for index, context in enumerate(contexts))


def _direct_ops(case, contexts, *, wrapper=family_paged_mla, padded_tokens=None,
                head_contract=None, **options):
    plan = _mla(case.model)
    views = _views(contexts)
    builder = Builder(f"s0.l{plan.index}", plan, 0, views, case.model, "decode")
    heads = plan.num_heads // case.runtime.parallel.tp
    wrapper.lower_attention(
        builder, views=views, plan=plan, heads=heads, runtime=case.runtime,
        query="s0.q", rotary="s0.r", store="s0.kv", head_contract=head_contract,
        padded_tokens=padded_tokens, **options)
    return builder.ops


def _direct(case, contexts, **options):
    return _direct_ops(case, contexts, **options)[-1]


def _expected_work_hbm(*, length, physical_heads, page_size):
    """One dispatch-free split: KV rows, query, page indices and output."""
    return (length * 576 * 2 + physical_heads * 576 * 2
            + math.ceil(length / page_size) * 4 + physical_heads * 512 * 2)


def _geometry(case, op):
    return Geometry.from_devices(case.machine.dies[:case.runtime.parallel.tp])


def test_real_a5_native_graph_entry_keeps_active_bucket_and_page_abi():
    case = _parsed()
    graph = target_graphs(case, scope="body", phase="decode", tokens=1,
                          sequences=1, context=4096)[0]
    op = next(op for op in graph if op.recipe.startswith("ascend_paged_mla"))
    metadata = dict(op.metadata)
    geometry = metadata["paged_mla_geometry"]
    abi = metadata["native_attention_abi"]

    # The graph entry selected one native kernel, while the logical head count
    # and the padded physical head lanes stay separate.
    assert op.recipe == "ascend_paged_mla_decode_arch35"
    assert op.source_boundary == "kernel"
    assert op.local_shape == (1, 16, 1, 512)
    assert op.padded_shape == op.launch_shape == (1, 16, 1, 512)
    assert geometry["contexts"] == (4096,)
    assert geometry["kv_lengths"] == (4097,)
    assert geometry["position_ranges"] == ((4096, 4097),)
    assert geometry["page_size"] == 128
    assert geometry["page_counts"] == (33,)
    assert geometry["active_sequence_count"] == 1
    assert geometry["native_query_capacity"] == 1
    assert geometry["inactive_sequence_kv_length"] == 0
    assert geometry["kv_split_source"] == "source_host_dispatch"
    assert geometry["effective_kv_splits"] is None
    assert geometry["logical_heads"] == 12 and geometry["heads"] == 16

    # The native ABI carries the actual position, page and type values.
    assert abi["call"] == "FusedInferAttentionScore"
    assert abi["branch"] == "arch35_bf16_noquant_mla"
    assert abi["query_layout"] == "BNSD" and abi["output_layout"] == "NBSD"
    assert abi["cache_layout"] == "PA_BSND"
    assert abi["latent_width"] == 512 and abi["rope_width"] == 64
    assert abi["page_table_dtype"] == "int32"
    assert abi["page_entries"] == 33
    assert abi["position_ranges"] == ((4096, 4097),)
    assert abi["quant_scales"] is None
    assert abi["inactive_sequence_kv_length"] == 0
    view = metadata["attention_tensor_view"]
    assert view["kv_split_source"] == "source_host_dispatch"
    assert view["effective_kv_splits"] is None
    assert view["position_ranges"] == ((4096, 4097),)
    contract = predicted_contract(op.to_dict())
    assert contract[:2] == ("attention_paged_mla_qk_rope",
                            "FusedInferAttentionScore")
    assert contract[2][0] == ((1, 16, 1, 512), "bf16", "ND")
    assert contract[2][1] == ((1, 16, 1, 64), "bf16", "ND")
    assert contract[-1] == 1
    planned = build_arch35(op, _geometry(case, op))
    assert planned.metadata["physical_segments"] == 32
    assert planned.metadata["active_aic_owners"] == 32
    assert planned.metadata["host_dispatch"] == "source_split_core_v2"
    assert planned.metadata["requested_kv_splits"] is None
    assert planned.work.resources["hbm"] > 0


def test_independent_native_fragment_carries_typed_ports_state_and_unknowns():
    case = _parsed()
    plan = _mla(case.model)
    views = _views((4096,))
    fragment = family_paged_mla.native_contract(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime, dispatch="source_split_core_v2")
    attrs = {attr.name: attr for attr in fragment.attrs}
    assert fragment.name == "paged_mla_decode"
    assert [ref.key for ref in fragment.semantic_refs] == [
        ("subgraph", "mla_attention", "mla_decode")]
    assert attrs["kv_split_source"].value == "source_host_dispatch"
    assert attrs["effective_kv_splits"].value is None
    assert attrs["contexts"].value == (4096,)
    assert attrs["kv_lengths"].value == (4097,)
    assert attrs["position_ranges"].value == ((4096, 4097),)
    assert attrs["page_size"].value == 128
    assert attrs["quant_scales"].value is None
    assert attrs["unresolved_fields"].value == (
        "native_binary_identity", "actual_split_segments_before_plan",
        "kv_state_version_before_invocation",
        "kv_producer_output_unresolved", "kv_request_rank_unresolved",
        "attention_scale_source_unresolved")
    ports = {port.name: port for port in (
        *fragment.ports.tensor_in, *fragment.ports.tensor_out,
        *fragment.ports.state_in, *fragment.ports.event_in,
        *fragment.ports.event_out)}
    assert ports["q"].dtype == "bf16" and ports["q"].layout_kind == "BNSD"
    assert ports["q_rope"].dtype == "bf16"
    assert ports["kv_cache"].layout_kind == "PA_BSND"
    assert ports["page_table"].dtype == "int32"
    assert ports["attention_output"].layout_kind == "NBSD"
    assert ports["kv_cache_state"].representation == "paged_latent_kv"
    assert {ports[name].event for name in (
        "q_ready", "kv_cache_ready", "page_table_ready")} == {"ready"}
    assert ports["done"].event == "complete"
    assert [(effect.target, effect.action, effect.condition)
            for effect in fragment.state_effects] == [
        ("kv_cache", "read", "active_contexts_only")]
    fields = {effect.field: effect for effect in fragment.field_effects}
    assert fields["q.layout"].action == "consume"
    assert fields["kv_cache.page_size"].source == "runtime.kv_block_size"
    assert fields["scale"].action == "preserve"
    assert not fragment.work.items
    assert "work supplied by selected recipe plan" in fragment.notes[0]


def test_bucket_capacity_is_storage_not_active_kv_or_query_work():
    case = _parsed()
    active = _direct(case, (4096,) * 6)
    bucketed = _direct(case, (4096,) * 6, padded_tokens=8)
    explicit = _direct(case, (4096,) * 6, native_query_capacity=8)
    geometry = _geometry(case, active)
    baseline = build_arch35(active, geometry).work
    for op in (bucketed, explicit):
        assert op.local_shape == (6, 16, 1, 512)
        assert op.padded_shape == op.launch_shape == (8, 16, 1, 512)
        assert dict(op.metadata)["paged_mla_geometry"]["native_query_capacity"] == 8
        assert dict(op.metadata)["paged_mla_geometry"]["active_sequence_count"] == 6
        assert build_arch35(op, geometry).work == baseline
    # The generic caller materializes logical query heads into the physical
    # bucket, but the native kernel's KV work follows only active contexts.
    active_pads = {op.recipe: op for op in _direct_ops(case, (4096,) * 6)
                   if "paged_mla_query_pad" in op.op_id}
    bucket_pads = {op.recipe: op for op in _direct_ops(
        case, (4096,) * 6, padded_tokens=8)
        if "paged_mla_query_pad" in op.op_id}
    assert {op.local_shape[0] for op in active_pads.values()} == {6}
    assert {op.local_shape[0] for op in bucket_pads.values()} == {8}
    for part, width in (("nope", 512), ("rope", 64)):
        before = active_pads["paged_mla_query_pad_" + part]
        after = bucket_pads["paged_mla_query_pad_" + part]
        assert before.mte2_bytes == 6 * 12 * width * 2
        assert after.mte2_bytes == 8 * 12 * width * 2
        assert after.mte3_bytes - before.mte3_bytes == 2 * 16 * width * 2
    with pytest.raises(ModelingError,
                       match="paged_mla_native_query_capacity_below_active_sequences"):
        _direct(case, (4096,) * 6, native_query_capacity=4)
    with pytest.raises(ModelingError,
                       match="paged_mla_padded_tokens_below_active_sequences"):
        _direct(case, (4096,) * 6, padded_tokens=4)


def test_caller_head_contract_owns_physical_padding():
    case = _parsed()
    plan = _mla(case.model)
    views = _views((4096,) * 6)
    builder = Builder(f"s0.l{plan.index}", plan, 0, views, case.model, "decode")
    family_paged_mla.lower_attention(
        builder, views=views, plan=plan, heads=16, runtime=case.runtime,
        query="s0.q", rotary="s0.r", store="s0.kv",
        head_contract={"logical_heads": 12, "physical_heads": 16},
        padded_tokens=8)
    op = builder.ops[-1]
    assert not any("paged_mla_query_pad" in item.op_id for item in builder.ops)
    assert op.padded_shape == op.launch_shape == (8, 16, 1, 512)
    baseline = _direct(case, (4096,) * 6)
    assert build_arch35(op, _geometry(case, op)).work == build_arch35(
        baseline, _geometry(case, baseline)).work


@pytest.mark.parametrize("context,model_heads,page_size", [
    (0, 96, 128), (4096, 96, 128), (8192, 96, 128), (4096, 256, 128),
    (4096, 96, 64),
])
def test_context_head_and_page_perturbations_follow_independent_work(
        context, model_heads, page_size):
    case = _parsed()
    plan = replace(_mla(case.model), num_heads=model_heads)
    runtime = replace(case.runtime, kv_block_size=page_size)
    views = _views((context,))
    builder = Builder("s0.l0", plan, 0, views, case.model, "decode")
    family_paged_mla.lower_attention(
        builder, views=views, plan=plan, heads=model_heads // 8,
        runtime=runtime, query="s0.q", rotary="s0.r", store="s0.kv")
    op = builder.ops[-1]
    physical = 1 << ((model_heads // 8) - 1).bit_length()
    geometry = _geometry(case, op)
    assert dict(op.metadata)["paged_mla_geometry"]["page_size"] == page_size
    expected = _expected_work_hbm(length=context + 1, physical_heads=physical,
                                  page_size=page_size)
    assert build_arch35(op, geometry).work.resources["hbm"] == expected


def test_hardware_supply_perturbation_changes_duration_not_declared_work():
    case = _parsed()
    graph = target_graphs(case, scope="body", phase="decode", tokens=1,
                          sequences=1, context=4096)[0]
    op = next(op for op in graph if op.recipe.startswith("ascend_paged_mla"))
    baseline = operator_costs(case, [op], input_ready_ms=7)["operators"][0]
    assert baseline["duration_ms"] > 0 and baseline["live_tasks"] == 0
    assert all(abs(value) < 1e-7 for value in baseline["remaining_buffers"].values())

    slower_dies = tuple(
        replace(die, rates=tuple((name, value / 2) for name, value in die.rates),
                compute_peaks=tuple((name, value / 2)
                                    for name, value in die.compute_peaks))
        for die in case.machine.dies)
    slower = replace(case, machine=replace(case.machine, dies=slower_dies))
    slowed = operator_costs(slower, [op], input_ready_ms=7)["operators"][0]
    assert slowed["duration_ms"] > baseline["duration_ms"]
    assert slowed["operator"]["hbm_bytes"] == baseline["operator"]["hbm_bytes"]
    assert slowed["operator"]["flops"] == baseline["operator"]["flops"]
    assert slowed["live_tasks"] == 0


def test_t12_typed_payload_sees_split_source_and_position_fields():
    case = _parsed()
    graph = target_graphs(case, scope="body", phase="decode", tokens=1,
                          sequences=1, context=4096)[0]
    op = next(op for op in graph if op.recipe.startswith("ascend_paged_mla"))
    payload = _typed_payload(op, dict(op.metadata))
    assert payload["attrs"]["kv_split_source"] == "source_host_dispatch"
    assert payload["attrs"]["effective_kv_splits"] is None
    assert payload["attrs"]["position_ranges"] == [[4096, 4097]]
    assert payload["attrs"]["quant_scales"] is None
    assert payload["shapes"]["query"] == [1, 16, 1, 512]
    assert payload["shapes"]["query_rope"] == [1, 16, 1, 64]

    a3 = _parsed(A3_CASE)
    a3_op = _arch22_op(a3, None)
    a3_payload = _typed_payload(a3_op, dict(a3_op.metadata))
    assert a3_payload["attrs"]["kv_split_source"] == "source_host_dispatch"
    assert a3_payload["attrs"]["effective_kv_splits"] is None
    assert a3_payload["attrs"]["position_ranges"] == [[1024, 1025]]
    assert a3_payload["shapes"]["query"] == [1, 16, 1, 512]


def test_a3_arch22_native_entry_stays_available_and_separate():
    raw = json.loads(A3_CASE.read_text())
    raw["runtime"]["kernels"].setdefault("lowerings", {})["paged_mla_attention"] = {
        "implementation": ARCH22, "options": {}}
    graph = target_graphs(raw, scope="body", phase="decode", tokens=1,
                          sequences=1, context=4096)[0]
    op = next(op for op in graph if op.recipe == ARCH22)
    metadata = dict(op.metadata)
    geometry = metadata["paged_mla_geometry"]
    abi = metadata["native_attention_abi"]
    assert geometry["contexts"] == (4096,) and geometry["page_size"] == 128
    assert geometry["retained_kv_rows"] == 0
    assert "kv_splits" not in geometry
    assert geometry["kv_split_source"] == "source_host_dispatch"
    assert geometry["effective_kv_splits"] is None
    assert abi["kv_split_source"] == "source_host_dispatch"
    assert abi["effective_kv_splits"] is None
    assert abi["branch"] == "arch22_bf16_noquant_mla"
    assert metadata["attention_tensor_view"]["kv_split_source"] == (
        "source_host_dispatch")
    assert metadata["attention_tensor_view"]["effective_kv_splits"] is None
    assert metadata["implementation_binding"]["host_dispatch"] == (
        "source_common_SplitCore_q1_geometry")
    parsed = parse_case(raw)
    plan = build_arch22(op, _geometry(parsed, op))
    assert plan.work.resources["hbm"] > 0
    result = operator_costs(parsed, [op], input_ready_ms=7)["operators"][0]
    assert result["duration_ms"] > 0
    assert result["live_tasks"] == 0
    assert not any(result["remaining_buffers"].values())


def _arch22_op(case, dispatch, retained_kv_rows=0, *, kv_owner='',
               kv_version=None, **state_options):
    plan = _mla(case.model)
    views = _views((1024,))
    builder = Builder(f"s0.l{plan.index}", plan, 0, views, case.model, "decode")
    family_paged_mla_arch22.lower_attention(
        builder, views=views, plan=plan, heads=plan.num_heads // 8,
        runtime=case.runtime, query="s0.q", rotary="s0.r", store="s0.kv",
        padded_tokens=1, dispatch=dispatch,
        retained_kv_rows=retained_kv_rows,
        kv_owner=kv_owner, kv_version=kv_version, **state_options)
    return builder.ops[-1]


def _arch22_geometry(cores=24):
    units = {role: cores * (2 if role in ("vector", "mte2_aiv", "mte3") else 1)
             for role in ("hbm", "vector", "cube", "mte1", "mte2_aic",
                          "mte2_aiv", "mte3", "fixpipe")}
    units["hbm"] = 1
    units["l2"] = 1
    buffers = {**ARCH22_BUFFERS, "l2": 2 * 512 * 576 * 2}
    units["l2"] = 1
    buffer_units = {pool: (2 * cores if pool == "ub" else cores)
                    for pool in buffers}
    buffer_units["l2"] = 1
    return Geometry(cores, units, buffers, buffer_units)


def _run_arch22(plan, op, geometry, *, ready_ms=7):
    ctx = _arch22_context(plan, op, geometry)
    rates = {role: 1e9 for role in geometry.resource_units}
    rates["cube"] = 1
    engine = Engine(
        [Resource(role, value, units=geometry.resource_units[role])
         for role, value in rates.items()],
        pools=geometry.buffers, pool_units=geometry.buffer_units)
    ready = engine.submit("input", release=ready_ms)
    done = plan.lower(ctx).start(engine, "mla", ready)
    engine.advance(until=(done,))
    for role, amount in ctx.bind(plan.work).items():
        assert engine.serviced[role] == pytest.approx(amount, rel=1e-12)
    assert done.time >= ready_ms
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    return done.time - ready_ms


def _arch22_context(plan, op, geometry):
    rates = {role: 1e9 for role in geometry.resource_units}
    rates["cube"] = 1
    return RecipeContext(
        plan.work.apply(op), {}, {role: role for role in rates}, rates,
        geometry.compute_units, resource_units=geometry.resource_units,
        compute_peaks={"bf16": 1e12},
        pool_names={pool: pool for pool in geometry.buffers},
        pool_capacities=geometry.buffers,
        pool_units=geometry.buffer_units,
        service_modes={
            "mte2_aic_cached": ("mte2_aic", 2e11),
            "l2_read": ("l2", 2e11),
            "l2_write": ("l2", 2e11)})


def test_arch22_implicit_source_dispatch_is_unresolved_not_one_split():
    oracle = next(row for row in ARCH22_ORACLE
                  if row["cores"] == 24 and row["heads"] == 16
                  and tuple(row["contexts"]) == (1024,))
    case = _parsed(A3_CASE)
    implicit = _arch22_op(case, None)
    explicit = _arch22_op(case, oracle)
    for op, source in ((implicit, "source_host_dispatch"),
                       (explicit, "explicit_dispatch")):
        metadata = dict(op.metadata)
        geometry = metadata["paged_mla_geometry"]
        abi = metadata["native_attention_abi"]
        assert geometry["contexts"] == (1024,)
        assert geometry["kv_lengths"] == (1025,)
        assert geometry["page_counts"] == (9,)
        assert geometry["kv_split_source"] == source
        assert geometry["effective_kv_splits"] is None
        assert abi["kv_split_source"] == source
        assert abi["effective_kv_splits"] is None
        assert "kv_splits" not in geometry
        assert op.local_shape == (1, 16, 1, 512)

    fragment = family_paged_mla_arch22.native_contract(
        views=_views((1024,)), plan=_mla(case.model),
        parallel=case.runtime.parallel, runtime=case.runtime)
    fragment_attrs = {attr.name: attr.value for attr in fragment.attrs}
    assert fragment_attrs["kv_split_source"] == "source_host_dispatch"
    assert fragment_attrs["effective_kv_splits"] is None

    geometry = _arch22_geometry()
    implicit_plan = build_arch22(implicit, geometry)
    explicit_plan = build_arch22(explicit, geometry)
    assert implicit_plan.metadata["physical_segments"] == 3
    assert explicit_plan.metadata["physical_segments"] == 3
    assert implicit_plan.metadata["active_aic_owners"] == 3
    assert implicit_plan.metadata["active_fd_aiv_owners"] == 2
    assert implicit_plan.metadata["host_dispatch"] == (
        "source_common_SplitCore_q1_geometry")
    assert explicit_plan.metadata["host_dispatch"] == (
        "declared_common_SplitCore_endpoints")
    # Complete source work and the independently derived tail matrix:
    # 2 full 512-key tiles + one aligned 16-key tail, two 576/512 products.
    assert implicit_plan.work == explicit_plan.work
    expected_matrix = (2 * (2 * 16 * 512 * (576 + 512))
                       + 2 * 16 * 16 * (576 + 512))
    assert dict(implicit_plan.work.matrix) == {"bf16": expected_matrix}
    assert expected_matrix == 36208640
    hbm = implicit_plan.work.resources["hbm"]
    assert hbm > 0 and hbm == explicit_plan.work.resources["hbm"]
    assert set(implicit_plan.work.resources) == {
        "hbm", "vector", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe"}

    body = implicit_plan.lower(
        _arch22_context(implicit_plan, implicit, geometry)).stages[1].program
    assert tuple(stage.name for stage in body.stages) == (
        "reserve", "producers", "merge", "release")
    assert body.stages[2].after == ("producers",)
    assert body.stages[3].after == ("merge",)
    assert len(body.stages[1].program.programs) == 3
    assert len(body.stages[2].program.programs) == 2


def _d3_d1_dispatches():
    d3 = next(row for row in ARCH22_ORACLE
              if row["cores"] == 24 and row["heads"] == 16
              and tuple(row["contexts"]) == (1024,))
    d1 = deepcopy(d3)
    d1["ends"] = [[1, 0, 0]]
    d1["fd"] = []
    d1["fd_ends"] = []
    return d3, d1


def test_graph_and_native_plan_keep_d3_d1_dispatch_topology():
    case = _parsed(A3_CASE)
    plan = _mla(case.model)
    views = _views((1024,))
    geometry = _arch22_geometry()
    d3, d1 = _d3_d1_dispatches()
    plans = {}
    for name, dispatch, segments, owners, merges in (
            ("D3", d3, 3, 3, 2), ("D1", d1, 1, 1, 0)):
        bare = family_paged_mla_arch22.native_plan(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, geometry=geometry, dispatch=dispatch)
        fragment = family_paged_mla_arch22.native_contract(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, dispatch=dispatch)
        fragment_attrs = {attr.name: attr.value for attr in fragment.attrs}
        frozen_dispatch = dict(fragment_attrs["dispatch"])
        assert frozen_dispatch["contexts"] == (1024,)
        assert frozen_dispatch["ends"] == tuple(
            tuple(row) for row in dispatch["ends"])
        assert frozen_dispatch["fd"] == tuple(
            tuple(row) for row in dispatch["fd"])
        assert fragment_attrs["implementation"] == ARCH22
        graph_op = _arch22_op(case, dispatch)
        graph = family_paged_mla_arch22.recipes()[ARCH22].prepare(
            graph_op, geometry)[1]
        assert bare.work == graph.work
        assert bare.metadata == graph.metadata
        assert bare.metadata["physical_segments"] == segments
        assert bare.metadata["active_aic_owners"] == owners
        assert bare.metadata["active_fd_aiv_owners"] == merges
        state = bare.metadata["native_state_access"]
        assert state["target"] == "kv_cache" and state["action"] == "read"
        assert state["owner"] == "self"
        assert state["readiness"] == "producer_complete"
        fragment = family_paged_mla_arch22.native_contract(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, dispatch=dispatch)
        fragment_attrs = {attr.name: attr.value for attr in fragment.attrs}
        assert fragment_attrs["kv_owner"] == "self"
        assert fragment_attrs["kv_version"] is None
        graph_body = graph.lower(
            _arch22_context(graph, graph_op, geometry)).stages[1].program
        bare_body = bare.lower(
            _arch22_context(bare, graph_op, geometry)).stages[1].program
        assert tuple(stage.name for stage in graph_body.stages) == (
            "reserve", "producers", "merge", "release")
        assert tuple(stage.after for stage in graph_body.stages) == (
            (), ("reserve",), ("producers",), ("merge",))
        assert tuple(stage.name for stage in bare_body.stages) == (
            "reserve", "producers", "merge", "release")
        assert tuple(stage.after for stage in bare_body.stages) == (
            (), ("reserve",), ("producers",), ("merge",))
        assert len(graph_body.stages[1].program.programs) == owners
        assert len(graph_body.stages[2].program.programs) == merges
        assert graph_body.stages[0].acquire == bare_body.stages[0].acquire
        assert graph_body.stages[0].free == bare_body.stages[0].free
        assert graph_body.stages[2].after == bare_body.stages[2].after
        assert _run_arch22(graph, graph_op, geometry) > 0
        assert _run_arch22(bare, graph_op, geometry) > 0
        plans[name] = (bare, graph)
    # D3 and D1 keep the same logical matrix work but different issued traffic,
    # owner count and merge topology; neither fragment summary collapses them.
    assert plans["D3"][0].work.matrix == plans["D1"][0].work.matrix == {
        "bf16": 36208640}
    assert plans["D3"][0].work != plans["D1"][0].work
    assert plans["D3"][0].metadata["physical_segments"] == 3
    assert plans["D1"][0].metadata["physical_segments"] == 1
    assert plans["D3"][1].metadata == plans["D3"][0].metadata
    assert plans["D1"][1].metadata == plans["D1"][0].metadata


def test_arch22_native_entry_preserves_retained_kv_bytes_and_allocation():
    case = _parsed(A3_CASE)
    plan = _mla(case.model)
    views = _views((1024,))
    geometry = _arch22_geometry()
    _, d1 = _d3_d1_dispatches()
    cold = family_paged_mla_arch22.native_plan(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime, geometry=geometry, dispatch=d1,
        retained_kv_rows=0)
    warm = family_paged_mla_arch22.native_plan(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime, geometry=geometry, dispatch=d1,
        retained_kv_rows=128)
    graph_op = _arch22_op(case, d1, retained_kv_rows=128)
    graph = family_paged_mla_arch22.recipes()[ARCH22].prepare(
        graph_op, geometry)[1]
    assert warm.work == graph.work
    assert warm.metadata == graph.metadata
    rows = 2 * 128 + 1
    saved = rows * 512 * 2
    assert cold.work.resources["hbm"] - warm.work.resources["hbm"] == saved
    assert cold.work.resources["mte2_aic"] - warm.work.resources["mte2_aic"] == saved
    assert warm.work.resources["mte2_aic_cached"] == saved
    assert warm.work.resources["l2_read"] == saved
    assert warm.work.resources["l2_write"] == rows * 576 * 2
    assert warm.work.matrix == cold.work.matrix
    assert warm.metadata["cache_policy"]["resident_bytes"] == 2 * 128 * 576 * 2
    assert dict(graph_op.metadata)["paged_mla_geometry"]["retained_kv_rows"] == 128

    fragment = family_paged_mla_arch22.native_contract(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime, dispatch=d1, retained_kv_rows=128)
    attrs = {attr.name: attr.value for attr in fragment.attrs}
    assert attrs["retained_kv_rows"] == 128
    assert attrs["retained_kv_layout"] == "source_s2_tile_adjacent_generations"
    assert attrs["retained_kv_capacity_owner"] == "per_active_aic_owner"
    assert attrs["kv_owner"] == "self"
    frozen_dispatch = dict(attrs["dispatch"])
    assert frozen_dispatch["contexts"] == (1024,)
    assert frozen_dispatch["fd"] == ()
    assert frozen_dispatch["heads"] == 16
    cold_body = cold.lower(
        _arch22_context(cold, graph_op, geometry)).stages[1].program
    warm_body = warm.lower(
        _arch22_context(warm, graph_op, geometry)).stages[1].program
    cold_allocations = {item.name for item in cold_body.stages[0].acquire}
    warm_allocations = {item.name for item in warm_body.stages[0].acquire}
    assert "retained_kv" not in cold_allocations
    assert "retained_kv" in warm_allocations
    assert _run_arch22(warm, graph_op, geometry) > 0


def test_owner_binding_and_ready_join_are_actual():
    case = _parsed(A3_CASE)
    plan = _mla(case.model)
    views = _views((1024,))
    geometry = _arch22_geometry()
    _, d1 = _d3_d1_dispatches()
    completions = []
    owner_plans = {}
    for owner, owner_ready in (("layer:0", 7), ("layer:1", 11)):
        engine = Engine(
            [Resource(role, 1e9, units=geometry.resource_units[role])
             for role in geometry.resource_units],
            pools=geometry.buffers, pool_units=geometry.buffer_units)
        conductor = FragmentConductor(engine, state={"kv_cache": 7})
        q_ready = engine.submit("q_ready", release=3)
        page_ready = engine.submit("page_ready", release=2)
        kv_ready = engine.submit("kv_ready", release=owner_ready)
        fragment = family_paged_mla_arch22.native_contract(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, dispatch=d1, kv_owner=owner,
            kv_version=7, kv_producer=f"{owner}.store",
            kv_output="latent_kv", kv_storage="paged_latent_kv",
            kv_span=((0, 1025),), request_rank=0)
        attrs = {attr.name: attr.value for attr in fragment.attrs}
        assert attrs["kv_owner"] == owner
        assert attrs["kv_owner_source"] == "explicit"
        assert attrs["kv_producer"] == f"{owner}.store"
        assert attrs["kv_output"] == "latent_kv"
        assert attrs["kv_storage"] == "paged_latent_kv"
        assert attrs["kv_span"] == ((0, 1025),)
        assert attrs["kv_request_ids"] == ("0",)
        assert attrs["kv_request_rank"] == 0
        state_port = fragment.ports.state_in[0]
        assert state_port.owner == owner
        handle = conductor.submit(fragment, conditions={
            "active_contexts_only": True}, inputs={
            "q": q_ready, "q_rope": q_ready, "kv_cache": kv_ready,
            "page_table": page_ready, "kv_cache_state": kv_ready,
            "q_ready": q_ready, "kv_cache_ready": kv_ready,
            "page_table_ready": page_ready,
        })
        graph_op = _arch22_op(case, d1, kv_owner=owner, kv_version=7,
                              kv_producer=f"{owner}.store")
        graph = family_paged_mla_arch22.recipes()[ARCH22].prepare(
            graph_op, geometry)[1]
        independent = family_paged_mla_arch22.native_plan(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, geometry=geometry, dispatch=d1,
            kv_owner=owner, kv_version=7,
            kv_producer=f"{owner}.store", kv_output="latent_kv",
            kv_storage="paged_latent_kv", kv_span=((0, 1025),),
            request_rank=0)
        assert graph.metadata["native_state_access"]["owner"] == owner
        assert independent.metadata["native_state_access"]["owner"] == owner
        owner_plans[owner] = independent
        assert dict(graph_op.metadata)["native_state_access"]["owner"] == owner
        ctx = _arch22_context(independent, graph_op, geometry)
        plan_ready = engine.submit("plan_ready", after=(q_ready, page_ready, kv_ready))
        done = independent.lower(ctx).start(engine, "mla", plan_ready)
        engine.advance(until=(handle.event,))
        engine.advance(until=(done,))
        assert handle.event.time == owner_ready
        assert done.time >= owner_ready
        assert conductor.state.snapshot()["committed"]["kv_cache"] == 7
        assert not any(effect.action in {"write", "commit"}
                       for effect in fragment.state_effects)
        completions.append((handle.event.time, done.time))
    assert completions[1][0] - completions[0][0] == 4
    assert completions[1][1] - completions[0][1] == pytest.approx(4)
    first, second = owner_plans["layer:0"], owner_plans["layer:1"]
    assert first.work == second.work
    assert first.metadata["native_state_access"]["owner"] == "layer:0"
    assert second.metadata["native_state_access"]["owner"] == "layer:1"
    # Rebinding the second instance must not rewrite the first declaration.
    assert first.metadata["native_state_access"]["version"] == 7
    assert tuple(effect.action for effect in family_paged_mla_arch22.native_contract(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime, dispatch=d1, kv_owner="layer:0",
        kv_version=7).state_effects) == ("read",)


def test_typed_latent_conflict_is_reported_not_rewritten():
    case = _parsed(A3_CASE)
    plan = replace(_mla(case.model), kv_lora_rank=256)
    views = _views((1024,))
    with pytest.raises(ModelingError,
                       match="paged_mla_selected_branch_requires_latent512_rope64:256:64"):
        family_paged_mla_arch22.native_contract(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime)
    with pytest.raises(ModelingError,
                       match="paged_mla_selected_branch_requires_latent512_rope64:256:64"):
        family_paged_mla_arch22.native_plan(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, geometry=_arch22_geometry())
    with pytest.raises(ModelingError,
                       match="paged_mla_selected_bf16_branch_requires_bf16"):
        family_paged_mla_arch22.native_contract(
            views=views, plan=_mla(case.model),
            parallel=case.runtime.parallel, runtime=case.runtime,
            cache_dtype="fp16")


def test_branch_conditions_are_shared_by_all_entry_points():
    case = _parsed(A3_CASE)
    plan = replace(_mla(case.model), window=128)
    views = _views((1024,))
    geometry = _arch22_geometry()
    with pytest.raises(ModelingError,
                       match="paged_mla_requires_q1_bf16_dense_latent512_rope64"):
        family_paged_mla_arch22.validate(
            views=views, plan=plan, parallel=case.runtime.parallel,
            model=case.model, quant=case.model.quant, padded_tokens=1)
    with pytest.raises(ModelingError,
                       match="paged_mla_requires_q1_bf16_dense_latent512_rope64"):
        family_paged_mla_arch22.native_contract(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime)
    with pytest.raises(ModelingError,
                       match="paged_mla_requires_q1_bf16_dense_latent512_rope64"):
        family_paged_mla_arch22.native_plan(
            views=views, plan=plan, parallel=case.runtime.parallel,
            runtime=case.runtime, geometry=geometry)
    builder = Builder("s0.l0", plan, 0, views, case.model, "decode")
    with pytest.raises(ModelingError,
                       match="paged_mla_requires_q1_bf16_dense_latent512_rope64"):
        family_paged_mla_arch22.lower_attention(
            builder, views=views, plan=plan, heads=plan.num_heads // 8,
            runtime=case.runtime, query="q", rotary="r", store="k",
            padded_tokens=1)
    # A legal unseen tail keeps the source-dispatch path available.
    tail_views = _views((1030,))
    tail = family_paged_mla_arch22.native_plan(
        views=tail_views, plan=_mla(case.model),
        parallel=case.runtime.parallel, runtime=case.runtime,
        geometry=geometry)
    assert tail.work.matrix["bf16"] > 0


def test_ragged_positions_pages_and_dispatch_use_real_views():
    case = _parsed()
    plan = _mla(case.model)
    views = _views((1, 127, 128, 4096))
    geometry = family_paged_mla.native_geometry(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime)
    assert geometry["contexts"] == (1, 127, 128, 4096)
    assert geometry["kv_lengths"] == (2, 128, 129, 4097)
    assert geometry["position_ranges"] == ((1, 2), (127, 128), (128, 129),
                                           (4096, 4097))
    assert geometry["page_counts"] == (1, 1, 2, 33)
    assert geometry["page_entries"] == 37
    assert geometry["kv_split_source"] == "declared_kv_splits"
    assert geometry["effective_kv_splits"] == (1, 1, 1, 1)
    split = family_paged_mla.native_geometry(
        views=views, plan=plan, parallel=case.runtime.parallel,
        runtime=case.runtime, kv_splits=4)
    assert split["effective_kv_splits"] == (1, 1, 2, 4)


@pytest.mark.parametrize("factory,match", [
    (lambda: (), "paged_mla_requires_active_decode_sequence"),
    (lambda: _views((-1,)), "paged_mla_requires_nonnegative_integer_context"),
    (lambda: (RequestView("p", 1, 0, False, "decode", padding=True),),
     "paged_mla_rejects_padding_view_as_active_request"),
    (lambda: (RequestView("d", 1, 16, False, "decode",
                          draft_proposal_tokens=2),),
     "paged_mla_does_not_declare_speculative_draft_token_views"),
    (lambda: (RequestView("q", 2, 16, False, "decode"),),
     "paged_mla_requires_single_token_decode_views"),
])
def test_invalid_active_views_raise_specific_requests(factory, match):
    case = _parsed()
    with pytest.raises(ModelingError, match=match):
        family_paged_mla.native_geometry(
            views=factory(), plan=_mla(case.model),
            parallel=case.runtime.parallel, runtime=case.runtime)


def test_wrapper_validate_and_dispatch_errors_are_specific():
    case = _parsed()
    plan = _mla(case.model)
    views = _views((4096,))
    kwargs = dict(views=views, plan=plan, parallel=case.runtime.parallel,
                  model=case.model, quant=case.model.quant, padded_tokens=1)
    family_paged_mla.validate(**kwargs)
    with pytest.raises(ModelingError, match="paged_mla_dispatch_policy_unknown"):
        family_paged_mla.validate(dispatch="another_policy", **kwargs)
    with pytest.raises(ModelingError,
                       match="paged_mla_dispatch_contexts_must_match_active_views"):
        family_paged_mla.validate(
            dispatch={"contexts": (8192,), "heads": 16}, **kwargs)
    with pytest.raises(ModelingError,
                       match="paged_mla_dispatch_heads_must_match_physical_heads"):
        family_paged_mla.validate(
            dispatch={"contexts": (4096,), "heads": 32}, **kwargs)
    with pytest.raises(ModelingError,
                       match="paged_mla_dispatch_heads_must_be_integer"):
        family_paged_mla.validate(
            dispatch={"contexts": (4096,), "heads": "all"}, **kwargs)
    with pytest.raises(ModelingError, match="paged_mla_head_contract_mismatch"):
        _direct(case, (4096,), head_contract={
            "logical_heads": 12, "physical_heads": 32})
    builder = Builder("s0.l0", plan, 0, views, case.model, "decode")
    with pytest.raises(ModelingError,
                       match="paged_mla_caller_heads_must_be_logical_or_physical"):
        family_paged_mla.lower_attention(
            builder, views=views, plan=plan, heads=32, runtime=case.runtime,
            query="q", rotary="r", store="k")
