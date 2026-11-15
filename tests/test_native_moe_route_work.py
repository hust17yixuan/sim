"""Route, layout and common-program work invariants for the native MoE leaves.

[Role] Exercise the existing A5 R64 GLM51 prefill input through the three
  native MoE leaves and through a perturbed route of the same input shape.
  The checks establish that expert counts, stable expert-ascending layout,
  quant metadata and issued work come from the declared occurrence fields,
  and that an unknown route never becomes a balanced assumption.

[Invariants] Counts are integers and conserved; a declared token->expert
  identity list gives the exact ordered layout; counts alone give offsets but
  keep the within-block identity order unresolved; the balanced split exists
  only as an explicit performance control.  Work is the typed descriptor
  traffic and row count, never a target duration or a shape/ABI timing table.
  Missing route facts stay explicit while route-independent work is still
  emitted.

[Timeouts & Locks] CPU-only bounded arithmetic; no device, weights or I/O.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_moe_init_routing_v3 import (
    binding_spec as inbound_spec, dynamic_call as inbound_dynamic,
    plan_inbound, program_fragment as inbound_fragment)
from mindie_sim.prediction.piecewise.native_moe_token_permute import (
    binding_spec as permute_spec, dynamic_call as permute_dynamic,
    execute_staged, native_program, normalize_local_route, normalize_route,
    plan_permute)
from mindie_sim.prediction.piecewise.native_r64_histogram import (
    binding_spec as histogram_spec, dynamic_call as histogram_dynamic,
    leaf_selected, plan_histogram, selected_protocol)
from mindie_sim.prediction.scheduling import (
    FragmentConductor, FragmentContractError)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

R64_IMAGE_SHA = (
    "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e")
R35_IMAGE_SHA = (
    "5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)
COMM = (
    ("mode", "dynamic"), ("policy", "capacity_and_world_size"),
    ("capacity", 64), ("world_size", 16), ("tp", 1), ("top_k", 8),
    ("experts_per_device", 16), ("enable_fused_mc2", 1),
    ("mega_moe_supported", 0), ("enable_prefill_mc2", 0),
    ("max_num_batched_tokens", 8192), ("max_num_reqs", 64),
    ("uniform_decode_query_len", 1),
    ("capture_sizes", (1, 2, 4, 8, 16, 32, 64)),
    ("lora", 0), ("is_draft_model", 0), ("draft_quant_supported", 1),
)
INBOUND_BASE = (
    ("E", 256), ("k", 8), ("router_hidden", 6144),
    ("router_received_rows", 341), ("router_payload_bytes", 0.5),
    ("router_scale_cols", 192), ("router_index_bytes", 4),
    ("router_source_dtype", "fp4_e2m1"), ("router_scale_dtype", "fp8_e8m0"),
    ("routed_weight_format", "mxfp4"),
    ("router_selected", "post_dispatch_received_rows"),
    ("router_source_operation", "MoeInitRoutingV3"),
    ("moe_comm", COMM),
)
PERMUTE_BASE = (
    ("router_hidden", 6144), ("router_dtype_bytes", 2),
    ("router_indices_bytes", 4), ("router_prob_bytes", 2),
    ("router_source_operation", "MoeTokenPermute/MoeInitRoutingV2"),
    ("router_source_dtype", "bf16"), ("router_source_device", "AIV"),
    ("router_selected", "prefill_local_outbound"),
)
HISTOGRAM_BASE = (("E", 256), ("k", 8), ("moe_comm", COMM))
PERMUTE_IDS = ((0, 1), (2, 3), (0, 2), (1, 2))
INBOUND_IDS = (0, 1, 0, 2, 1, 0)
INBOUND_A = (40, 36, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 5)
# The second received-row ledger already used by the existing R64 adapter
# test (669 rows), with an explicit non-uniform distribution over 16 experts.
INBOUND_B = (100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 18, 15, 14, 12, 10)


def _compile(*, kind, name, role, extras, matrix, experts, image=R64_IMAGE_SHA):
    template = OperatorTemplate(name=name, family="router", role=role)
    signature = StaticSignature(
        role=name, matrix=matrix, experts=experts,
        tiles=(("mt", 16), ("cores", 1)), extras=extras)
    operator = BoundOperator(template, signature)
    return OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        operator, kind=kind, feature_metadata=(("image_id", image),))


def _inbound(*, extras=INBOUND_BASE, hidden=6144, scale_cols=192,
             image=R64_IMAGE_SHA, kind="routed_experts"):
    overridden = tuple(
        (key, hidden if key == "router_hidden"
         else scale_cols if key == "router_scale_cols" else value)
        for key, value in extras)
    return _compile(
        kind=kind, name="router_inbound", role="inbound", extras=overridden,
        matrix=(("elements", 8),), experts=(("E", 256), ("k", 8)),
        image=image)


def _permute(*, extras=PERMUTE_BASE, hidden=6144, experts=256, top_k=8):
    overridden = tuple(
        (key, hidden if key == "router_hidden" else value)
        for key, value in extras)
    return _compile(
        kind="moe_router", name="router_permute", role="permute",
        extras=overridden, matrix=(("elements", top_k),),
        experts=(("E", experts), ("k", top_k)))


def _histogram(*, extras=HISTOGRAM_BASE, experts=256, top_k=8,
               image=R64_IMAGE_SHA):
    return _compile(
        kind="moe_router", name="router_histogram", role="histogram", extras=extras,
        matrix=(("elements", top_k),), experts=(("E", experts), ("k", top_k)),
        image=image)


def _bind(compiled, spec_builder):
    spec = spec_builder(compiled, phase="prefill")
    assert spec is not None
    surface = _surface(compiled, spec, source_boundary="kernel")
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None))


def _geometry(tokens=1024, routing=None, phase="prefill"):
    return WorkloadGeometry(
        (RequestGeometry(tokens, 0, phase=phase),), phase=phase,
        routing=tuple(routing or ()))


def _with(extras, **updates):
    values = dict(extras)
    values.update(updates)
    return tuple(sorted(values.items()))


def test_inbound_route_counts_offsets_and_work_follow_the_received_ledger():
    counts_a, counts_b = INBOUND_A, INBOUND_B
    compiled_a = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=counts_a,
        router_received_rows=sum(counts_a)))
    compiled_b = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=counts_b,
        router_received_rows=sum(counts_b)))
    binding_a = _bind(compiled_a, inbound_spec)
    binding_b = _bind(compiled_b, inbound_spec)
    geometry = _geometry()

    first = inbound_dynamic(binding_a, compiled_a, geometry,
                            validate=validate_native_occurrence)
    second = inbound_dynamic(binding_b, compiled_b, geometry,
                             validate=validate_native_occurrence)
    assert first is not None and second is not None
    route_a = first["prepare"]["route"]
    route_b = second["prepare"]["route"]
    assert route_a["basis"] == "declared_expert_token_counts"
    assert tuple(route_a["counts"]) == counts_a
    assert tuple(route_a["expert_offsets"]) == (
        0, 40, 76, 108, 138, 166, 192, 216, 238, 258, 276, 292, 306, 318,
        328, 336, 341)
    assert route_a["active_experts"] == 16
    assert route_a["performance_control"] is False
    assert route_a["token_identities"] == "not_declared"
    assert route_a["assignment_identity"] == "not_declared"
    # Counts give exact blocks but not the identity order inside a block.
    assert route_a["forward_assignment_order"] is None
    assert route_a["inverse_assignment_order"] is None
    assert first["prepare"]["expanded_row_idx"] is None

    assert tuple(route_b["counts"]) == counts_b
    assert route_b["rows"] == sum(counts_b) != route_a["rows"]
    assert first["prepare"]["work"]["mte2_bytes"] == 341 * (3072 + 192 + 4)
    assert first["prepare"]["work"]["mte3_bytes"] == 341 * (3072 + 192 + 4) + 128
    assert first["prepare"]["work"]["vector_ops"] == 341 * 6144 + 16
    assert second["prepare"]["work"]["mte2_bytes"] == (
        sum(counts_b) * (3072 + 192 + 4))
    assert (first["prepare"]["work"]["mte2_bytes"]
            != second["prepare"]["work"]["mte2_bytes"])
    # The full sequence of expanded rows stays unresolved rather than guessed.
    assert first["prepare"]["route"]["within_expert_order"] == (
        "unresolved_no_token_identities")


def test_inbound_declared_ids_produce_the_exact_expert_grouped_layout():
    extras = _with(
        INBOUND_BASE, router_received_rows=len(INBOUND_IDS),
        router_local_expert_ids=INBOUND_IDS)
    compiled = _inbound(extras=extras)
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_token_expert_ids"
    assert tuple(route["counts"]) == (3, 2, 1, 0) + (0,) * 12
    assert tuple(route["expert_offsets"]) == (0, 3, 5, 6) + (6,) * 13
    # Forward order is expert 0 rows 0,2,5; expert 1 rows 1,4; expert 2 row 3.
    # row_idx_type=0 exports the inverse: position of each original row after
    # the expert grouping, i.e. inverse[0..5] = (0,3,1,5,4,2).
    assert tuple(result["prepare"]["expanded_row_idx"]) == (0, 3, 1, 5, 4, 2)
    assert route["token_identities"] == "explicit_assignment"
    assert route["assignment_identity"] == "row_index"
    assert result["prepare"]["work"]["mte2_bytes"] == 6 * (3072 + 192 + 4)


def test_inbound_missing_route_is_explicit_and_never_balanced():
    compiled = _inbound()
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    assert result["prepare"]["route"]["basis"] == "unresolved"
    assert result["prepare"]["route"]["counts"] is None
    assert "expert_token_counts_unresolved" in result["prepare"]["partial_reasons"]
    # Route-independent received-row work is still emitted.
    assert result["prepare"]["work"]["mte2_bytes"] == 341 * (3072 + 192 + 4)


def test_inbound_balanced_split_is_a_named_performance_control():
    extras = _with(INBOUND_BASE, router_route_control="balanced")
    compiled = _inbound(extras=extras)
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "forced_balanced_performance_control"
    assert route["performance_control"] is True
    assert route["token_identities"] == "not_declared"
    assert sum(route["counts"]) == 341
    assert route["counts"] == [22, 22, 22, 22, 22, 21, 21, 21, 21, 21,
                               21, 21, 21, 21, 21, 21]
    assert "forced_balanced_token_identities_not_declared" in route["unresolved"]


def test_inbound_inconsistent_declared_counts_fail_instead_of_guessing():
    extras = _with(
        INBOUND_BASE, router_received_rows=341,
        router_expert_token_counts=(1,) * 16)
    compiled = _inbound(extras=extras)
    binding = _bind(compiled, inbound_spec)
    with pytest.raises(
            ModelingError,
            match="moe_route_local_rows_conflict"):
        inbound_dynamic(binding, compiled, _geometry(),
                        validate=validate_native_occurrence)


def test_inbound_new_shapes_keep_packed_payload_and_scale_padding():
    # ceil(6176/32)=193 logical E8M0 values -> 97 pairs with one pad value.
    compiled = _inbound(hidden=6176, scale_cols=193)
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    descriptors = result["prepare"]["descriptors"]
    assert descriptors["scale"]["shape"] == [341, 97, 2]
    assert descriptors["expanded_scale"]["shape"] == [341, 97, 2]
    quant = result["prepare"]["quant_metadata"]
    assert quant["scale_logical_cols"] == 193
    assert quant["scale_pairs"] == 97
    assert quant["scale_storage_values"] == 194
    assert quant["scale_padding_values"] == 1
    assert quant["payload_row_bytes"] == 3088
    assert quant["payload_row_padding_bits"] == 0
    assert result["prepare"]["work"]["mte2_bytes"] == 341 * (3088 + 194 + 4)

    # An odd H packs the last FP4 value into a padded byte, not a 0.5 byte.
    compiled = _inbound(hidden=6145, scale_cols=193)
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    descriptor = result["prepare"]["descriptors"]["x"]
    assert descriptor["storage_shape"] == [341, 3073]
    assert result["prepare"]["quant_metadata"]["payload_row_padding_bits"] == 4
    assert result["prepare"]["work"]["mte2_bytes"] == 341 * (3073 + 194 + 4)


def test_permute_counts_offsets_and_work_are_input_driven():
    counts_a = [64] * 128 + [0] * 128
    counts_b = [32] * 256
    compiled_a = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=counts_a))
    compiled_b = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=counts_b))
    binding_a = _bind(compiled_a, permute_spec)
    binding_b = _bind(compiled_b, permute_spec)
    geometry = _geometry()
    first = permute_dynamic(binding_a, compiled_a, geometry,
                            validate=validate_native_occurrence)
    second = permute_dynamic(binding_b, compiled_b, geometry,
                             validate=validate_native_occurrence)
    assert first is not None and second is not None
    route_a = first["prepare"]["route"]
    route_b = second["prepare"]["route"]
    assert route_a["basis"] == "declared_expert_token_counts"
    assert tuple(route_a["counts"]) == tuple(counts_a)
    assert tuple(route_b["counts"]) == tuple(counts_b)
    assert route_a["active_experts"] == 128
    assert route_b["active_experts"] == 256
    assert route_a["expert_offsets"][:3] == [0, 64, 128]
    assert route_b["expert_offsets"][:3] == [0, 32, 64]
    # Same M*k bytes: a distribution change does not fabricate extra moves.
    assert first["prepare"]["work"] == second["prepare"]["work"]
    assert first["prepare"]["work"]["mte2_bytes"] == 1024 * 6144 * 2 + 8192 * 4
    assert first["prepare"]["work"]["mte3_bytes"] == 8192 * 6144 * 2 + 8192 * 4
    assert first["prepare"]["work"]["vector_ops"] == 8192 * 3

    large_compiled = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=[64] * 256))
    large_binding = _bind(large_compiled, permute_spec)
    larger = permute_dynamic(large_binding, large_compiled, _geometry(tokens=2048),
                             validate=validate_native_occurrence)
    assert larger is not None
    assert larger["prepare"]["work"]["mte2_bytes"] == 2048 * 6144 * 2 + 16384 * 4
    assert larger["prepare"]["work"]["mte3_bytes"] == 16384 * 6144 * 2 + 16384 * 4


def test_permute_declared_ids_give_the_exact_stable_sorted_indices():
    small = _permute(
        extras=_with(PERMUTE_BASE, router_topk_ids=PERMUTE_IDS),
        hidden=32, experts=8, top_k=2)
    binding = _bind(small, permute_spec)
    result = permute_dynamic(binding, small, _geometry(tokens=4),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_token_expert_ids"
    assert tuple(route["counts"]) == (2, 2, 3, 1, 0, 0, 0, 0)
    assert tuple(route["expert_offsets"]) == (0, 2, 4, 7, 8, 8, 8, 8, 8)
    # Forward order over flattened token*top_k+slot identities is
    # (0,4,1,6,2,5,7,3); the ABI output is its inverse.
    assert tuple(route["forward_assignment_order"]) == (0, 4, 1, 6, 2, 5, 7, 3)
    assert tuple(route["inverse_assignment_order"]) == (0, 2, 4, 7, 1, 5, 3, 6)
    assert tuple(route["assignment_tokens"]) == (0, 2, 0, 3, 1, 2, 3, 1)
    assert tuple(route["assignment_slots"]) == (0, 0, 1, 0, 0, 1, 1, 1)
    assert route["assignment_identity"] == "flattened_token_slot"
    assert tuple(result["prepare"]["sorted_indices"]) == (0, 2, 4, 7, 1, 5, 3, 6)
    assert route["token_identities"] == "explicit_assignment"
    assert result["prepare"]["work"]["mte2_bytes"] == 4 * 32 * 2 + 8 * 4


def test_permute_route_for_another_token_count_stays_unresolved():
    small = _permute(
        extras=_with(PERMUTE_BASE, router_topk_ids=PERMUTE_IDS),
        hidden=32, experts=8, top_k=2)
    binding = _bind(small, permute_spec)
    result = permute_dynamic(binding, small, _geometry(tokens=8),
                             validate=validate_native_occurrence)
    assert result is not None
    assert result["prepare"]["route"]["basis"] == "unresolved"
    assert "router_topk_ids_do_not_match_evaluated_token_geometry" in (
        result["prepare"]["route"]["unresolved"])
    # No prefix of the other token count was consumed.
    assert result["prepare"]["sorted_indices"] is None


def test_permute_balanced_split_is_a_named_performance_control():
    compiled = _permute(extras=_with(
        PERMUTE_BASE, router_route_control="balanced"))
    binding = _bind(compiled, permute_spec)
    result = permute_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "forced_balanced_performance_control"
    assert route["performance_control"] is True
    assert sum(route["counts"]) == 8192
    assert route["counts"] == [32] * 256
    assert route["inverse_assignment_order"] is None


def test_histogram_route_counts_fill_the_int32_output_and_drive_work():
    counts = [64] * 128 + [0] * 128
    compiled = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=counts))
    binding = _bind(compiled, histogram_spec)
    result = histogram_dynamic(binding, compiled, _geometry(),
                               validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_expert_token_counts"
    assert tuple(route["counts"]) == tuple(counts)
    assert result["prepare"]["histogram_values"] == counts
    assert route["expert_offsets"][:3] == [0, 64, 128]
    assert result["prepare"]["work"]["mte2_bytes"] == 1024 * 8 * 4
    assert result["prepare"]["work"]["mte3_bytes"] == 256 * 4
    assert result["prepare"]["work"]["vector_ops"] == 1024 * 8 + 256
    assert result["prepare"]["quant_metadata"]["histogram_dtype"] == "int32"

    large_compiled = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=[128] * 128 + [0] * 128))
    large_binding = _bind(large_compiled, histogram_spec)
    larger = histogram_dynamic(large_binding, large_compiled,
                               _geometry(tokens=2048),
                               validate=validate_native_occurrence)
    assert larger is not None
    assert larger["prepare"]["work"]["mte2_bytes"] == 2048 * 8 * 4
    assert larger["prepare"]["work"]["vector_ops"] == 2048 * 8 + 256


def test_histogram_declared_ids_count_global_experts_exactly():
    small_comm = _with(COMM, capacity=0)
    small = _histogram(
        extras=_with(HISTOGRAM_BASE, router_topk_ids=PERMUTE_IDS,
                     moe_comm=small_comm),
        experts=8, top_k=2)
    binding = _bind(small, histogram_spec)
    result = histogram_dynamic(binding, small, _geometry(tokens=4),
                               validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_token_expert_ids"
    assert tuple(route["counts"]) == (2, 2, 3, 1, 0, 0, 0, 0)
    assert result["prepare"]["histogram_values"] == [2, 2, 3, 1, 0, 0, 0, 0]
    assert result["prepare"]["work"]["vector_ops"] == 8 + 8


def test_histogram_missing_route_keeps_work_but_not_output_values():
    compiled = _histogram()
    binding = _bind(compiled, histogram_spec)
    result = histogram_dynamic(binding, compiled, _geometry(),
                               validate=validate_native_occurrence)
    assert result is not None
    assert result["prepare"]["route"]["basis"] == "unresolved"
    assert result["prepare"]["histogram_values"] is None
    assert "expert_token_counts_unresolved" in result["prepare"]["partial_reasons"]
    assert result["prepare"]["work"]["mte2_bytes"] == 1024 * 8 * 4


def test_histogram_other_protocol_or_image_is_not_a_separate_leaf():
    compiled = _histogram()
    binding = _bind(compiled, histogram_spec)
    assert leaf_selected(binding, _geometry(tokens=32)) is False
    assert histogram_dynamic(binding, compiled, _geometry(tokens=32),
                             validate=validate_native_occurrence) is None
    assert histogram_spec(_histogram(image=R35_IMAGE_SHA),
                          phase="prefill") is None


def test_histogram_inconsistent_counts_fail_instead_of_truncating():
    extras = _with(HISTOGRAM_BASE, router_expert_token_counts=(1,) * 8)
    compiled = _histogram(extras=extras)
    binding = _bind(compiled, histogram_spec)
    with pytest.raises(
            ModelingError,
            match="moe_route_expert_token_counts_scope_mismatch"):
        histogram_dynamic(binding, compiled, _geometry(),
                          validate=validate_native_occurrence)


def test_common_program_slice_is_serviced_by_the_existing_engine():
    def plan_for(counts, label):
        route = normalize_local_route(
            declared={"router_expert_token_counts": counts},
            geometry_counts=None, global_experts=256, local_range=None,
            rows=sum(counts))
        return plan_inbound(
            call_id=label, phase="prefill", hidden=6144, experts=256,
            local_experts=16, received_rows=sum(counts), payload="fp4_e2m1",
            scale_cols=192, scale_pairs=96, route=route)

    plan_a = plan_for(INBOUND_A, "counts-a")
    plan_b = plan_for(INBOUND_B, "counts-b")
    issued = plan_a["work"]["resources"]
    useful = plan_a["work"]["useful_resources"]
    time_a, service_a, serviced_a, _, _, _ = _run_plan(plan_a)
    time_b, service_b, serviced_b, _, _, _ = _run_plan(plan_b)
    assert service_a == issued
    assert serviced_a == issued
    assert service_b == plan_b["work"]["resources"]
    assert serviced_b == plan_b["work"]["resources"]
    assert useful["hbm"] == 2 * 341 * 3072
    lower_a = max(
        issued["mte2"] / 1_000_000.0, issued["mte3"] / 1_000_000.0,
        issued["hbm"] / 1_000_000.0, issued["vector"] / 1_000_000.0)
    assert time_a >= lower_a
    assert time_a != pytest.approx(time_b)
    assert plan_a["fragment"].occurrence.kind == "call"
    assert plan_a["fragment"].occurrence.phase == "prefill"
    assert plan_a["stages"]


def test_permute_and_histogram_stage_programs_lower_without_a_second_timer():
    counts = [64] * 128 + [0] * 128
    permute_route = normalize_route(
        declared={"router_expert_token_counts": counts},
        geometry_counts=None, scope="global", experts=256, rows=8192,
        identity="topk_ids", tokens=1024, top_k=8)
    permute_plan = plan_permute(
        call_id="permute-plan", phase="prefill", tokens=1024, top_k=8,
        hidden=6144, route=permute_route)
    time, service, serviced, _, _, result = _run_plan(permute_plan)
    assert service == permute_plan["work"]["resources"]
    assert serviced == permute_plan["work"]["resources"]
    assert time >= max(serviced[key] / 1_000_000.0
                       for key in serviced)
    assert permute_plan["stages"]
    assert result["stage_completion"] is not None

    histogram_compiled = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=counts))
    histogram_binding = _bind(histogram_compiled, histogram_spec)
    protocol = selected_protocol(histogram_binding, _geometry())
    histogram_route = normalize_route(
        declared={"router_expert_token_counts": counts},
        geometry_counts=None, scope="global", experts=256, rows=8192,
        identity="topk_ids", tokens=1024, top_k=8)
    histogram_plan = plan_histogram(
        call_id="histogram-plan", phase="prefill", tokens=1024, top_k=8,
        experts=256, protocol=protocol, route=histogram_route)
    time, service, serviced, _, _, result = _run_plan(histogram_plan)
    assert service == histogram_plan["work"]["resources"]
    assert serviced == histogram_plan["work"]["resources"]
    assert time >= max(serviced[key] / 1_000_000.0
                       for key in serviced)
    assert histogram_plan["stages"]


def test_production_native_occurrence_exposes_route_work_for_permute_and_histogram():
    """The ordinary ``compiled.evaluate`` path carries the same work record.

    Inbound family evaluation is currently blocked by an external
    ``router_local_tokens`` literal in families_moe_collective; inbound is
    covered through the native adapter without that external family cost.
    """
    permute = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=[64] * 128 + [0] * 128))
    occurrence = permute.evaluate(_geometry()).native_occurrence
    assert occurrence["status"] == "bound"
    assert occurrence["prepare"]["program"]["common_program"] == "moe.F1.router_permute"
    assert occurrence["prepare"]["work"]["mte3_bytes"] == 8192 * 6144 * 2 + 8192 * 4

    histogram = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=[64] * 128 + [0] * 128))
    occurrence = histogram.evaluate(_geometry()).native_occurrence
    assert occurrence["status"] == "bound"
    assert occurrence["prepare"]["program"]["common_program"] == (
        "moe.F1.router_histogram")
    assert occurrence["prepare"]["histogram_values"] == [64] * 128 + [0] * 128


def test_explicit_route_and_balanced_control_are_rejected_together():
    extras = _with(
        INBOUND_BASE, router_received_rows=sum(INBOUND_A),
        router_expert_token_counts=INBOUND_A,
        router_route_control="balanced")
    compiled = _inbound(extras=extras)
    binding = _bind(compiled, inbound_spec)
    with pytest.raises(
            ModelingError,
            match="moe_route_actual_route_and_balanced_control_conflict"):
        inbound_dynamic(binding, compiled, _geometry(),
                        validate=validate_native_occurrence)


def test_route_is_an_occurrence_fact_not_part_of_the_call_identity():
    """The selected API/ABI identity stays fixed across route perturbations."""
    first = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=INBOUND_A,
        router_received_rows=sum(INBOUND_A)))
    other_counts = (22,) * 15 + (11,)
    second = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=other_counts,
        router_received_rows=sum(other_counts)))
    first_binding = _bind(first, inbound_spec)
    second_binding = _bind(second, inbound_spec)
    assert first_binding.call_id == second_binding.call_id
    left = inbound_dynamic(first_binding, first, _geometry(),
                           validate=validate_native_occurrence)
    right = inbound_dynamic(second_binding, second, _geometry(),
                            validate=validate_native_occurrence)
    assert left["prepare"]["route"]["layout_digest"] != (
        right["prepare"]["route"]["layout_digest"])
    # Same R: the route changes but the received-row work does not.
    assert left["prepare"]["work"] == right["prepare"]["work"]

    left_compiled = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=[64] * 128 + [0] * 128))
    right_compiled = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=[32] * 256))
    assert (_bind(left_compiled, permute_spec).call_id
            == _bind(right_compiled, permute_spec).call_id)


def test_outbound_inverse_uses_the_complete_flattened_assignment_identity():
    # Report P1-01 minimal example: one token, two experts (1, 0).  Forward
    # assignment order is (1, 0); the ABI sorted_indices is the inverse (1, 0),
    # never the token-only (0, 0).
    compiled = _permute(
        extras=_with(PERMUTE_BASE, router_topk_ids=((1, 0),)),
        hidden=32, experts=2, top_k=2)
    binding = _bind(compiled, permute_spec)
    result = permute_dynamic(binding, compiled, _geometry(tokens=1),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert tuple(route["forward_assignment_order"]) == (1, 0)
    assert tuple(route["inverse_assignment_order"]) == (1, 0)
    assert tuple(route["assignment_tokens"]) == (0, 0)
    assert tuple(route["assignment_slots"]) == (1, 0)
    assert tuple(result["prepare"]["sorted_indices"]) == (1, 0)
    assert result["prepare"]["assignment"]["identity_space"] == (
        "flattened_token_slot")
    assert tuple(result["prepare"]["assignment"]["tokens"]) == (0, 0)
    assert tuple(result["prepare"]["assignment"]["slots"]) == (1, 0)


def test_inbound_inverse_is_the_row_idx_type_0_gather_output():
    # Report P1-01 minimal example: ids (2,0,1) sort to forward (1,2,0);
    # row_idx_type=0 returns inverse (2,0,1), not the forward order.
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_received_rows=3,
        router_local_expert_ids=(2, 0, 1)))
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert tuple(route["forward_assignment_order"]) == (1, 2, 0)
    assert tuple(route["inverse_assignment_order"]) == (2, 0, 1)
    assert tuple(result["prepare"]["expanded_row_idx"]) == (2, 0, 1)
    assert result["prepare"]["assignment"]["identity_space"] == "row_index"
    assert tuple(result["prepare"]["assignment"]["forward"]) == (1, 2, 0)
    assert tuple(result["prepare"]["assignment"]["inverse"]) == (2, 0, 1)


def test_inbound_ids_and_counts_conflict_is_reported_not_silently_chosen():
    counts = (1, 2) + (0,) * 14
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_received_rows=3,
        router_local_expert_ids=(0, 0, 1),
        router_expert_token_counts=counts))
    binding = _bind(compiled, inbound_spec)
    with pytest.raises(ModelingError, match="moe_route_ids_counts_conflict"):
        inbound_dynamic(binding, compiled, _geometry(),
                        validate=validate_native_occurrence)


def test_geometry_route_and_balanced_control_conflict_is_rejected():
    counts = (4,) + (0,) * 15
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_received_rows=4,
        router_route_control="balanced"))
    binding = _bind(compiled, inbound_spec)
    geometry = _geometry(tokens=4, routing=((0, counts),))
    with pytest.raises(
            ModelingError,
            match="moe_route_actual_route_and_balanced_control_conflict"):
        inbound_dynamic(binding, compiled, geometry,
                        validate=validate_native_occurrence)


def test_stale_topk_identity_keeps_valid_counts_and_offsets():
    # Identity belongs to an older 8-token occurrence; the current occurrence
    # has 4 tokens * 2 slots = 8 rows and a valid count vector.  The identity
    # mapping is unresolved, but counts/offsets remain usable.
    counts = (1, 1, 1, 1, 1, 1, 1, 1)
    stale_ids = tuple((0, 1) for _ in range(8))
    compiled = _permute(
        extras=_with(PERMUTE_BASE, router_topk_ids=stale_ids,
                     router_expert_token_counts=counts),
        hidden=32, experts=8, top_k=2)
    binding = _bind(compiled, permute_spec)
    result = permute_dynamic(binding, compiled, _geometry(tokens=4),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_expert_token_counts"
    assert tuple(route["counts"]) == counts
    assert tuple(route["expert_offsets"]) == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert "router_topk_ids_do_not_match_evaluated_token_geometry" in (
        route["unresolved"])
    assert result["prepare"]["sorted_indices"] is None


def _engine():
    return Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ])


def _run_plan(plan, *, engine=None, conductor=None):
    engine = engine or _engine()
    conductor = conductor or FragmentConductor(engine)
    ready = engine.submit("input_ready")
    inputs = {port.name: engine.submit(f"ready_{port.name}")
              for port in plan["fragment"].ports.tensor_in}
    result = execute_staged(conductor, plan, ready=ready, inputs=inputs)
    conductor.advance()
    serviced = {name: round(float(value), 6)
                for name, value in engine.serviced.items()}
    return (result["completion"].time, serviced, serviced,
            engine, conductor, result)


def test_leaf_plan_is_the_single_source_for_prepare_materialization_and_work():
    counts = INBOUND_A
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=counts,
        router_received_rows=sum(counts)))
    binding = _bind(compiled, inbound_spec)
    route = normalize_local_route(
        declared={"router_expert_token_counts": counts},
        geometry_counts=None, global_experts=256, local_range=None,
        rows=sum(counts))
    plan = plan_inbound(
        call_id=binding.call_id, phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(counts), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route)
    assert plan["materialization"]["input_read_bytes"] == (
        plan["work"]["resources"]["mte2"])
    assert plan["materialization"]["output_write_bytes"] == (
        plan["work"]["resources"]["mte3"])
    assert plan["materialization"]["total_bytes"] == (
        plan["work"]["resources"]["hbm"])
    occurrence = inbound_dynamic(binding, compiled, _geometry(),
                                 validate=validate_native_occurrence)
    assert occurrence is not None
    assert occurrence["prepare"]["work"] == plan["work"]
    assert occurrence["prepare"]["materialization"] == plan["materialization"]
    assert occurrence["prepare"]["program"] == plan["program"]

    permute_route = normalize_route(
        declared={"router_expert_token_counts": [32] * 256},
        geometry_counts=None, scope="global", experts=256, rows=8192,
        identity="topk_ids", tokens=1024, top_k=8)
    permute_plan = plan_permute(
        call_id="permute-plan", phase="prefill", tokens=1024, top_k=8,
        hidden=6144, route=permute_route)
    assert permute_plan["materialization"]["input_read_bytes"] == (
        permute_plan["work"]["resources"]["mte2"])
    assert permute_plan["materialization"]["output_write_bytes"] == (
        permute_plan["work"]["resources"]["mte3"])
    assert permute_plan["materialization"]["total_bytes"] == (
        permute_plan["work"]["resources"]["hbm"])

    histogram_compiled = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=[32] * 256))
    histogram_binding = _bind(histogram_compiled, histogram_spec)
    protocol = selected_protocol(histogram_binding, _geometry())
    histogram_route = normalize_route(
        declared={"router_expert_token_counts": [32] * 256},
        geometry_counts=None, scope="global", experts=256, rows=8192,
        identity="topk_ids", tokens=1024, top_k=8)
    histogram_plan = plan_histogram(
        call_id="histogram-plan", phase="prefill", tokens=1024, top_k=8,
        experts=256, protocol=protocol, route=histogram_route)
    assert histogram_plan["materialization"]["input_read_bytes"] == (
        histogram_plan["work"]["resources"]["mte2"])
    assert histogram_plan["materialization"]["output_write_bytes"] == (
        histogram_plan["work"]["resources"]["mte3"])
    assert histogram_plan["materialization"]["total_bytes"] == (
        histogram_plan["work"]["resources"]["hbm"])


def test_full_fragments_execute_through_fragment_conductor_with_attrs():
    counts = INBOUND_A
    plan = plan_inbound(
        call_id="inbound-run", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(counts), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96,
        route=normalize_local_route(
            declared={"router_expert_token_counts": counts},
            geometry_counts=None, global_experts=256, local_range=None,
            rows=sum(counts)))
    assert plan["fragment"].attrs
    assert plan["stages"]
    time, service, serviced, _, _, result = _run_plan(plan)
    assert service == plan["work"]["resources"]
    assert serviced == plan["work"]["resources"]
    assert result["completion"].time == time
    assert time >= max(
        plan["work"]["resources"][name] / 1_000_000.0
        for name in ("mte2", "mte3", "hbm", "vector"))

    permute_route = normalize_route(
        declared={"router_expert_token_counts": [32] * 256},
        geometry_counts=None, scope="global", experts=256, rows=8192,
        identity="topk_ids", tokens=1024, top_k=8)
    permute_plan = plan_permute(
        call_id="permute-run", phase="prefill", tokens=1024, top_k=8,
        hidden=6144, route=permute_route)
    time, service, serviced, _, _, _ = _run_plan(permute_plan)
    assert service == permute_plan["work"]["resources"]
    assert serviced == permute_plan["work"]["resources"]

    histogram_compiled = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=[32] * 256))
    histogram_binding = _bind(histogram_compiled, histogram_spec)
    protocol = selected_protocol(histogram_binding, _geometry())
    histogram_route = normalize_route(
        declared={"router_expert_token_counts": [32] * 256},
        geometry_counts=None, scope="global", experts=256, rows=8192,
        identity="topk_ids", tokens=1024, top_k=8)
    histogram_plan = plan_histogram(
        call_id="histogram-run", phase="prefill", tokens=1024, top_k=8,
        experts=256, protocol=protocol, route=histogram_route)
    time, service, serviced, _, _, _ = _run_plan(histogram_plan)
    assert service == histogram_plan["work"]["resources"]
    assert serviced == histogram_plan["work"]["resources"]
    assert histogram_plan["fragment"].attrs


def test_two_leaf_stage_programs_share_resources_and_bounded_slots():
    engine = _engine()
    conductor = FragmentConductor(engine)
    route = normalize_local_route(
        declared={"router_expert_token_counts": INBOUND_A},
        geometry_counts=None, global_experts=256, local_range=None,
        rows=sum(INBOUND_A))
    first = plan_inbound(
        call_id="shared-a", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(INBOUND_A), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route)
    second = plan_inbound(
        call_id="shared-b", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(INBOUND_A), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route)
    ready = engine.submit("input_ready")
    result_a = execute_staged(
        conductor, first, ready=ready,
        inputs={port.name: engine.submit(f"a_{port.name}")
                for port in first["fragment"].ports.tensor_in})
    result_b = execute_staged(
        conductor, second, ready=ready,
        inputs={port.name: engine.submit(f"b_{port.name}")
                for port in second["fragment"].ports.tensor_in})
    conductor.advance()
    assert result_a["completion"].time is not None
    assert result_b["completion"].time is not None
    for name in ("mte2", "mte3", "hbm", "vector"):
        assert round(float(engine.serviced[name]), 6) == pytest.approx(
            first["work"]["resources"][name]
            + second["work"]["resources"][name])
    assert not engine.allocations

    # Real leaf stages with a finite slot pool; two calls share one slot.
    pool_engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ], pools={"moe_inbound_block": 40 * (3072 + 192)})
    pool_conductor = FragmentConductor(pool_engine)
    slotted_a = plan_inbound(
        call_id="slot-a", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(INBOUND_A), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route,
        block_slots=40 * (3072 + 192))
    slotted_b = plan_inbound(
        call_id="slot-b", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(INBOUND_A), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route,
        block_slots=40 * (3072 + 192))
    last = [record for record in slotted_a["stages"] if record["tail"]][-1]
    assert last["free"][0].size == 5 * (3072 + 192)
    assert last["free"][0].pool == "moe_inbound_block"
    assert slotted_a["allocation"]["workspace"]["capacity_bytes"] == (
        40 * (3072 + 192))
    pool_ready = pool_engine.submit("pool_ready")
    handle_a = execute_staged(
        pool_conductor, slotted_a, ready=pool_ready,
        inputs={port.name: pool_engine.submit(f"a_{port.name}")
                for port in slotted_a["fragment"].ports.tensor_in})
    handle_b = execute_staged(
        pool_conductor, slotted_b, ready=pool_ready,
        inputs={port.name: pool_engine.submit(f"b_{port.name}")
                for port in slotted_b["fragment"].ports.tensor_in})
    pool_conductor.advance()
    assert handle_a["completion"].time is not None
    assert handle_b["completion"].time is not None
    assert handle_a["grants"] and handle_b["grants"]
    assert handle_a["release_handles"] and handle_b["release_handles"]
    assert pool_engine.peak_allocated["moe_inbound_block"] == 40 * (3072 + 192)
    assert not pool_engine.allocations


def test_leaf_supply_and_family_cost_gap_is_explicit_for_handoff():
    # The leaf supplies one work record; the current production family still
    # computes its own envelope.  These assertions pin the exact gap that T12
    # and PW-FAMILIES must consume/replace, without pretending they already do.
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=INBOUND_A,
        router_received_rows=sum(INBOUND_A)))
    binding = _bind(compiled, inbound_spec)
    route = normalize_local_route(
        declared={"router_expert_token_counts": INBOUND_A},
        geometry_counts=None, global_experts=256, local_range=None,
        rows=sum(INBOUND_A))
    plan = plan_inbound(
        call_id=binding.call_id, phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(INBOUND_A), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route)
    leaf = plan["work"]["resources"]
    count_bytes = 16 * 8
    assert plan["materialization"]["output_write_bytes"] == leaf["mte3"]
    assert plan["program"]["alignment"]["additional_typed_output"][
        "expert_tokens_count_or_cumsum_bytes"] == count_bytes
    # Inbound family cost evaluation is externally blocked by a literal
    # router_local_tokens derived entry; the leaf plan remains the source.
    assert plan["program"]["issued_work"] == leaf

    permute_compiled = _permute(extras=_with(
        PERMUTE_BASE, router_expert_token_counts=[32] * 256))
    permute_binding = _bind(permute_compiled, permute_spec)
    permute_cost = permute_compiled.evaluate(_geometry())
    permute_family = dict(permute_cost.work)
    permute_plan = plan_permute(
        call_id=permute_binding.call_id, phase="prefill", tokens=1024,
        top_k=8, hidden=6144,
        route=normalize_route(
            declared={"router_expert_token_counts": [32] * 256},
            geometry_counts=None, scope="global", experts=256, rows=8192,
            identity="topk_ids", tokens=1024, top_k=8))
    permute_leaf = permute_plan["work"]["resources"]
    assert int(permute_family["mte2"]) == permute_leaf["mte2"]
    assert int(permute_family["mte3"]) == permute_leaf["mte3"]
    assert int(permute_family["vector"]) != permute_leaf["vector"]
    assert permute_plan["program"]["alignment"]["family_vector_basis"] == (
        "indices_bytes")

    histogram_compiled = _histogram(extras=_with(
        HISTOGRAM_BASE, router_expert_token_counts=[32] * 256))
    histogram_cost = histogram_compiled.evaluate(_geometry())
    histogram_family = dict(histogram_cost.work)
    histogram_binding = _bind(histogram_compiled, histogram_spec)
    protocol = selected_protocol(histogram_binding, _geometry())
    histogram_plan = plan_histogram(
        call_id=histogram_binding.call_id, phase="prefill", tokens=1024,
        top_k=8, experts=256, protocol=protocol,
        route=normalize_route(
            declared={"router_expert_token_counts": [32] * 256},
            geometry_counts=None, scope="global", experts=256, rows=8192,
            identity="topk_ids", tokens=1024, top_k=8))
    histogram_leaf = histogram_plan["work"]["resources"]
    assert histogram_leaf["mte3"] == 256 * 4
    assert int(histogram_family["mte3"]) != histogram_leaf["mte3"]
    assert int(histogram_family["vector"]) != histogram_leaf["vector"]
    assert "not used" in histogram_plan["program"]["alignment"][
        "family_envelope_status"]


def test_cpu_permute_unpermute_and_weighted_combine_keep_slots():
    tokens, top_k = 3, 2
    ids = ((2, 0), (1, 2), (0, 1))
    route = normalize_route(
        declared={"router_topk_ids": ids}, geometry_counts=None,
        scope="global", experts=3, rows=tokens * top_k,
        identity="topk_ids", tokens=tokens, top_k=top_k)
    assert route.forward is not None and route.inverse is not None
    values = (10, 11, 20, 21, 30, 31)       # token * 10 + slot
    probabilities = (0.70, 0.30, 0.20, 0.80, 0.60, 0.40)
    grouped = tuple(values[original] for original in route.forward)
    restored = tuple(grouped[route.inverse[original]]
                     for original in range(tokens * top_k))
    assert restored == values
    direct = tuple(
        sum(probabilities[token * top_k + slot] *
            values[token * top_k + slot] for slot in range(top_k))
        for token in range(tokens))
    from_restored = tuple(
        sum(probabilities[token * top_k + slot] *
            restored[token * top_k + slot] for slot in range(top_k))
        for token in range(tokens))
    assert from_restored == pytest.approx(direct)

    inbound_route = normalize_local_route(
        declared={"router_local_expert_ids": (2, 0, 1)},
        geometry_counts=None, global_experts=16, local_range=None, rows=3)
    inbound_values = (100, 200, 300)
    inbound_grouped = tuple(inbound_values[original]
                            for original in inbound_route.forward)
    inbound_restored = tuple(
        inbound_grouped[inbound_route.inverse[original]] for original in range(3))
    assert inbound_restored == inbound_values


def test_stale_inbound_ids_keep_valid_counts_and_offsets():
    counts = (2, 1, 1) + (0,) * 13
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_received_rows=4,
        router_local_expert_ids=(0, 0, 1),
        router_expert_token_counts=counts))
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_expert_token_counts"
    assert tuple(route["counts"]) == counts
    assert tuple(route["expert_offsets"]) == (
        0, 2, 3, 4) + (4,) * 13
    assert "router_local_expert_ids_do_not_match_evaluated_rows" in (
        route["unresolved"])
    assert result["prepare"]["expanded_row_idx"] is None


def test_inbound_missing_received_ledger_stays_symbolic_without_work():
    extras = _with(INBOUND_BASE, router_received_rows=None)
    compiled = _inbound(extras=extras)
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    assert result["actual"]["shapes"]["x"] == [None, 6144]
    assert result["prepare"]["work"] is None
    assert result["prepare"]["program"] is None
    assert result["prepare"]["route"]["basis"] == "unresolved"
    assert "router_received_rows_unresolved" in result["prepare"]["partial_reasons"]
    assert inbound_fragment(binding, compiled, _geometry()) is None


def test_geometry_route_that_disagrees_with_declared_route_is_reported():
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=INBOUND_A,
        router_received_rows=sum(INBOUND_A)))
    binding = _bind(compiled, inbound_spec)
    geometry_counts = (sum(INBOUND_A),) + (0,) * 15
    result = inbound_dynamic(
        binding, compiled,
        _geometry(routing=((0, geometry_counts),)),
        validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert tuple(route["counts"]) == INBOUND_A
    assert "geometry_route_disagrees_with_selected_route" in route["unresolved"]


COMM_NO_LOCAL = tuple((key, value) for key, value in COMM
                      if key != "experts_per_device")


def _inbound_no_local(**updates):
    extras = _with(
        INBOUND_BASE, router_hidden=64, router_scale_cols=2,
        router_received_rows=3, router_local_expert_ids=(0, 0, 1),
        moe_comm=COMM_NO_LOCAL)
    extras = _with(extras, **updates)
    return _inbound(extras=extras, hidden=64, scale_cols=2)


def test_inbound_missing_local_recovers_full_range_from_counts():
    # E256/R3/H64, ids=(0,0,1), no active range, complete local 16 counts.
    counts = (2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    compiled = _inbound_no_local()
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(
        binding, compiled, _geometry(routing=((0, counts),)),
        validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["expert_range"] == 16
    assert route["counts_complete"] is True
    assert tuple(route["counts"]) == counts
    assert result["actual"]["shapes"]["expert_tokens_count_or_cumsum"] == [16]
    assert result["prepare"]["materialization"]["count_write_bytes"] == 128
    assert result["prepare"]["materialization"]["output_write_bytes"] == 242
    assert result["prepare"]["work"]["mte3_bytes"] == 242
    assert any(record["tail"]
               for record in result["prepare"]["program"]["stages"])
    json.dumps(result["prepare"]["program"])
    fragment = inbound_fragment(binding, compiled,
                                _geometry(routing=((0, counts),)))
    assert fragment is not None
    assert fragment.work.by_resource("issued")["mte3"] == 242
    assert fragment.stages
    assert fragment.ports.tensor_out


def test_inbound_missing_local_and_counts_keeps_assignment_not_range():
    compiled = _inbound_no_local()
    binding = _bind(compiled, inbound_spec)
    result = inbound_dynamic(binding, compiled, _geometry(),
                             validate=validate_native_occurrence)
    assert result is not None
    route = result["prepare"]["route"]
    assert route["basis"] == "declared_token_expert_ids"
    assert route["expert_range"] is None
    assert route["counts"] is None
    assert route["counts_complete"] is False
    assert tuple(route["forward_assignment_order"]) == (0, 1, 2)
    assert tuple(route["inverse_assignment_order"]) == (0, 1, 2)
    assert result["actual"]["shapes"]["expert_tokens_count_or_cumsum"] == [None]
    assert result["prepare"]["materialization"]["count_write_bytes"] is None
    assert result["prepare"]["materialization"]["output_write_bytes"] == 114
    assert result["prepare"]["work"]["mte3_bytes"] == 114
    assert result["prepare"]["work"]["count_write_bytes"] is None
    assert "expert_tokens_count_write_unknown" in result["prepare"]["work"][
        "unresolved"]
    assert any(record["tail"]
               for record in result["prepare"]["program"]["stages"])
    fragment = inbound_fragment(binding, compiled, _geometry())
    assert fragment is not None
    assert fragment.work.by_resource("issued")["mte3"] == 114
    assert fragment.stages
    assert fragment.ports.tensor_out


def test_inbound_local_range_above_global_is_rejected():
    comm = _with(COMM, experts_per_device=300)
    compiled = _inbound(extras=_with(
        INBOUND_BASE, moe_comm=comm, router_received_rows=3,
        router_local_expert_ids=(0, 0, 1)))
    binding = _bind(compiled, inbound_spec)
    with pytest.raises(ModelingError, match="moe_route_local_experts_exceed_global"):
        inbound_dynamic(binding, compiled, _geometry(),
                        validate=validate_native_occurrence)


def test_read_processing_write_dependency_precedes_competing_writer():
    counts = (6,) + (0,) * 15
    route = normalize_local_route(
        declared={"router_expert_token_counts": counts,
                  "router_local_expert_ids": (0, 0, 0, 0, 0, 0)},
        geometry_counts=None, global_experts=256, local_range=16, rows=6)
    plan = plan_inbound(
        call_id="causal", phase="prefill", hidden=64, experts=256,
        local_experts=16, received_rows=6, payload="fp4_e2m1",
        scale_cols=2, scale_pairs=1, route=route)
    engine = Engine([
        Resource("mte2", 100.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ], trace_limit=64)
    ready = engine.submit("ready")
    completion = plan["execution_program"].start(engine, "moe.read", ready)
    competitor = engine.submit("competitor", {"mte3": 1_000_000})
    engine.advance(until=(completion, competitor), first=False)
    rows = {row["task"]: row for row in engine.trace}
    index = rows["moe.read:read_index"]
    prefix = rows["moe.read:count_prefix"]
    count_write = rows["moe.read:count_write"]
    inverse = rows["moe.read:write_row_idx"]
    read = rows["moe.read:read_e0"]
    gather = rows["moe.read:gather_e0"]
    write = rows["moe.read:write_e0"]
    other = rows["competitor"]
    assert index["start_ms"] == 0
    assert index["end_ms"] == pytest.approx(0.48)
    assert prefix["start_ms"] >= index["end_ms"]
    assert count_write["start_ms"] >= prefix["end_ms"]
    assert inverse["start_ms"] >= prefix["end_ms"]
    assert read["start_ms"] == 0
    assert read["end_ms"] == pytest.approx(2.28, abs=1e-3)
    assert count_write["start_ms"] < read["end_ms"]
    assert gather["start_ms"] >= read["end_ms"]
    assert write["start_ms"] >= gather["end_ms"]
    assert other["end_ms"] < write["start_ms"]
    assert completion.time == pytest.approx(write["end_ms"])
    assert tuple(plan["fragment"].attrs[[a.name for a in plan["fragment"].attrs].index(
        "expert_token_counts")].value) == counts
    assert tuple(plan["route"].inverse) == (0, 1, 2, 3, 4, 5)
    last = [record for record in plan["stages"] if record["tail"]][-1]
    assert last["work"]["mte3"] == 6 * (32 + 2)


def test_histogram_delayed_values_admit_before_read_and_publish_output():
    ids = ((1,),)
    route = normalize_route(
        declared={"router_topk_ids": ids}, geometry_counts=None,
        scope="global", experts=2, rows=1, identity="topk_ids",
        tokens=1, top_k=1)
    plan = plan_histogram(
        call_id="delayed-hist", phase="prefill", tokens=1, top_k=1,
        experts=2, protocol={"protocol": "alltoall"}, route=route)
    engine = Engine([
        Resource("mte2", 1.0, unit="bytes"),
        Resource("mte3", 1.0, unit="bytes"),
        Resource("hbm", 1.0, unit="bytes"),
        Resource("vector", 1.0, unit="ops"),
    ], trace_limit=128)
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    values = engine.submit("values", latency=100.0)
    minimum = engine.submit("min")
    maximum = engine.submit("max")
    result = execute_staged(
        conductor, plan, ready=ready,
        inputs={"values": values, "min": minimum, "max": maximum})
    conductor.advance()
    output = result["outputs"]["delayed-hist.histogram.complete"]
    rows = {row["task"]: row for row in engine.trace}
    assert next(row for name, row in rows.items()
                if "read_values" in name)["start_ms"] == 100
    assert next(row for name, row in rows.items()
                if "count" in name)["end_ms"] == 105
    assert next(row for name, row in rows.items()
                if "write_bins" in name)["end_ms"] == 113
    assert round(float(engine.serviced["mte2"]), 6) == 4
    assert round(float(engine.serviced["mte3"]), 6) == 8
    assert round(float(engine.serviced["hbm"]), 6) == 12
    assert round(float(engine.serviced["vector"]), 6) == 3
    consumer = engine.submit("consumer", {"vector": 1.0}, after=(output,))
    engine.advance(until=(consumer,))
    assert consumer.started == 113


def test_small_permute_fragment_and_staged_program_charge_once():
    ids = ((1,), (0,))
    route = normalize_route(
        declared={"router_topk_ids": ids}, geometry_counts=None,
        scope="global", experts=2, rows=2, identity="topk_ids",
        tokens=2, top_k=1)
    plan = plan_permute(
        call_id="small-permute", phase="prefill", tokens=2, top_k=1,
        hidden=2, route=route)
    expected = {"mte2": 16, "mte3": 16, "hbm": 32, "vector": 2}
    assert plan["work"]["resources"] == expected
    assert {item.resource: item.amount
            for item in plan["fragment"].work.issued()} == expected
    native = native_program(plan)
    assert native.stages
    json.dumps(native.to_dict())
    assert native.work.by_resource("issued") == expected

    engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ])
    completion = native.start(engine, "native.small", engine.submit("ready"))
    engine.advance(until=(completion,))
    for name, amount in expected.items():
        assert round(float(engine.serviced[name]), 6) == amount

    engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    inputs = {port.name: engine.submit(port.name)
              for port in plan["fragment"].ports.tensor_in}
    result = execute_staged(conductor, plan, ready=ready, inputs=inputs)
    conductor.advance()
    for name, amount in expected.items():
        assert round(float(engine.serviced[name]), 6) == amount
    assert result["outputs"][
        "small-permute.permuted_tokens.complete"].time is not None

    native_engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ])
    native_conductor = FragmentConductor(native_engine)
    native_inputs = {port.name: native_engine.submit(port.name)
                     for port in native.fragment.ports.tensor_in}
    native_result = native.submit_staged(
        native_conductor, ready=native_engine.submit("ready"),
        inputs=native_inputs)
    native_conductor.advance()
    for name, amount in expected.items():
        assert round(float(native_engine.serviced[name]), 6) == amount
    assert native_result["outputs"]


def test_permute_nonuniform_tail_work_and_byte_slot():
    counts = (1, 2, 3)
    route = normalize_route(
        declared={"router_expert_token_counts": counts},
        geometry_counts=None, scope="global", experts=3, rows=6,
        identity="", tokens=6, top_k=1)
    plan = plan_permute(
        call_id="tail-permute", phase="prefill", tokens=6, top_k=1,
        hidden=2, route=route, block_slots=24)
    assert plan["work"]["resources"] == {
        "mte2": 48, "mte3": 48, "hbm": 96, "vector": 6}
    last = [record for record in plan["stages"] if record["tail"]][-1]
    assert last["free"][0].size == 24
    assert last["free"][0].pool == "moe_permute_block"
    assert plan["allocation"]["workspace"]["capacity_bytes"] == 24
    engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ], pools={"moe_permute_block": 24})
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    inputs = {port.name: engine.submit(port.name)
              for port in plan["fragment"].ports.tensor_in}
    result = execute_staged(conductor, plan, ready=ready, inputs=inputs)
    conductor.advance()
    for name, amount in plan["work"]["resources"].items():
        assert round(float(engine.serviced[name]), 6) == amount
    assert engine.peak_allocated["moe_permute_block"] == 24
    assert not engine.allocations
    assert result["grants"]
    assert result["release_handles"]
    assert result["outputs"][
        "tail-permute.permuted_tokens.complete"].time is not None


def test_inbound_count_output_event_precedes_payload_write():
    counts = (6,) + (0,) * 15
    route = normalize_local_route(
        declared={"router_expert_token_counts": counts,
                  "router_local_expert_ids": (0, 0, 0, 0, 0, 0)},
        geometry_counts=None, global_experts=256, local_range=16, rows=6)
    plan = plan_inbound(
        call_id="count-event", phase="prefill", hidden=64, experts=256,
        local_experts=16, received_rows=6, payload="fp4_e2m1",
        scale_cols=2, scale_pairs=1, route=route)
    engine = Engine([
        Resource("mte2", 100.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ], trace_limit=128)
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    inputs = {port.name: engine.submit(port.name)
              for port in plan["fragment"].ports.tensor_in}
    result = execute_staged(conductor, plan, ready=ready, inputs=inputs)
    count_event = result["outputs"][
        "count-event.expert_tokens_count_or_cumsum.complete"]
    row_event = result["outputs"]["count-event.expanded_row_idx.complete"]
    name = "count-event.expert_tokens_count_or_cumsum.complete"
    assert conductor.event(name) is count_event
    from mindie_sim.schema.operators import (
        ProgramFragment, WorkItem, WorkSet)
    from mindie_sim.schema.subgraphs import EventPortSpec, PortSet
    consumer_fragment = ProgramFragment(
        "count_consumer", ports=PortSet(
            event_in=(EventPortSpec(name, "in", "complete"),)),
        work=WorkSet((WorkItem("vector", 1.0, "ops"),)))
    consumer_handle = conductor.submit(consumer_fragment)
    conductor.advance()
    rows = {row["task"]: row for row in engine.trace}
    assert row_event.time == pytest.approx(0.4800640024)
    assert count_event.time == pytest.approx(0.4801680128)
    assert result["completion"].time == pytest.approx(2.2806120128)
    assert count_event.time < next(
        row for name, row in rows.items()
        if "write_e0" in name)["end_ms"]
    assert consumer_handle.event.started == pytest.approx(count_event.time)


def test_execute_staged_rejects_unbound_input_before_work():
    ids = ((1,),)
    route = normalize_route(
        declared={"router_topk_ids": ids}, geometry_counts=None,
        scope="global", experts=2, rows=1, identity="topk_ids",
        tokens=1, top_k=1)
    plan = plan_histogram(
        call_id="unbound-hist", phase="prefill", tokens=1, top_k=1,
        experts=2, protocol={"protocol": "alltoall"}, route=route)
    engine = Engine([
        Resource("mte2", 1.0, unit="bytes"),
        Resource("mte3", 1.0, unit="bytes"),
        Resource("hbm", 1.0, unit="bytes"),
        Resource("vector", 1.0, unit="ops"),
    ])
    conductor = FragmentConductor(engine)
    with pytest.raises(FragmentContractError, match="fragment_input_unbound"):
        execute_staged(conductor, plan, ready=engine.submit("ready"),
                       inputs={})
    assert engine.completed_count == 0


def test_inbound_gather_waits_for_runtime_assignment_not_payload_read():
    ids = (1,) + (0,) * 11
    counts = (11, 1)
    route = normalize_local_route(
        declared={"router_local_expert_ids": ids,
                  "router_expert_token_counts": counts},
        geometry_counts=None, global_experts=256, local_range=2, rows=12)
    plan = plan_inbound(
        call_id="assignment-gather", phase="prefill", hidden=64, experts=256,
        local_experts=2, received_rows=12, payload="fp4_e2m1",
        scale_cols=2, scale_pairs=1, route=route)
    assert route.counts == counts
    assert route.forward == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0)
    assert route.inverse == (11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert plan["work"]["resources"]["mte2"] == 48 + 11 * 34 + 1 * 34
    assert plan["work"]["resources"]["mte3"] == 48 + 11 * 34 + 1 * 34 + 16
    assert plan["work"]["resources"]["hbm"] == (
        48 + 11 * 34 + 1 * 34 + 48 + 11 * 34 + 1 * 34 + 16)
    assert plan["work"]["resources"]["vector"] == 12 * 64 + 2

    engine = Engine([
        Resource("mte2", 1.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ], trace_limit=128)
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    inputs = {port.name: engine.submit(port.name)
              for port in plan["fragment"].ports.tensor_in}
    execute_staged(conductor, plan, ready=ready, inputs=inputs)
    conductor.advance()
    rows = {row["task"]: row for row in engine.trace}

    def task(token):
        return next(row for name, row in rows.items() if token in name)

    read_index = task("read_index")
    read_short = task("read_e1")
    read_long = task("read_e0")
    prefix = task("count_prefix")
    gather_short = task("gather_e1")
    gather_long = task("gather_e0")
    assert read_short["end_ms"] == pytest.approx(102.0, abs=0.5)
    assert read_index["end_ms"] > read_short["end_ms"]
    assert read_long["end_ms"] > read_index["end_ms"]
    assert prefix["start_ms"] >= read_index["end_ms"]
    assert gather_short["start_ms"] >= prefix["end_ms"]
    assert gather_long["start_ms"] >= prefix["end_ms"]
    assert gather_short["start_ms"] > read_short["end_ms"]


def test_histogram_stage_work_uses_conductor_rank_bindings_only():
    ids = ((1,),)
    route = normalize_route(
        declared={"router_topk_ids": ids}, geometry_counts=None,
        scope="global", experts=2, rows=1, identity="topk_ids",
        tokens=1, top_k=1)
    plan = plan_histogram(
        call_id="rank1-hist", phase="prefill", tokens=1, top_k=1,
        experts=2, protocol={"protocol": "alltoall"}, route=route)
    engine = Engine([
        Resource("rank1.mte2", 1.0, unit="bytes"),
        Resource("rank1.mte3", 1.0, unit="bytes"),
        Resource("rank1.hbm", 1.0, unit="bytes"),
        Resource("rank1.vector", 1.0, unit="ops"),
        Resource("rank2.mte2", 1.0, unit="bytes"),
        Resource("rank2.mte3", 1.0, unit="bytes"),
        Resource("rank2.hbm", 1.0, unit="bytes"),
        Resource("rank2.vector", 1.0, unit="ops"),
    ], trace_limit=128)
    conductor = FragmentConductor(engine, resource_bindings={
        "mte2": "rank1.mte2", "mte3": "rank1.mte3",
        "hbm": "rank1.hbm", "vector": "rank1.vector"})
    ready = engine.submit("ready")
    values = engine.submit("values", latency=100.0)
    minimum = engine.submit("min")
    maximum = engine.submit("max")
    result = execute_staged(
        conductor, plan, ready=ready,
        inputs={"values": values, "min": minimum, "max": maximum},
        owner="owner-rank1")
    second_plan = plan_histogram(
        call_id="rank2-hist", phase="prefill", tokens=1, top_k=1,
        experts=2, protocol={"protocol": "alltoall"}, route=route)
    second_conductor = FragmentConductor(engine, resource_bindings={
        "mte2": "rank2.mte2", "mte3": "rank2.mte3",
        "hbm": "rank2.hbm", "vector": "rank2.vector"})
    second_values = engine.submit("values-b", latency=100.0)
    second_result = execute_staged(
        second_conductor, second_plan, ready=engine.submit("ready-b"),
        inputs={"values": second_values, "min": engine.submit("min-b"),
                "max": engine.submit("max-b")}, owner="owner-rank2")
    conductor.advance()
    second_conductor.advance()
    assert round(float(engine.serviced["rank1.mte2"]), 6) == 4
    assert round(float(engine.serviced["rank1.mte3"]), 6) == 8
    assert round(float(engine.serviced["rank1.hbm"]), 6) == 12
    assert round(float(engine.serviced["rank1.vector"]), 6) == 3
    assert result["outputs"]["rank1-hist.histogram.complete"].time == 113
    assert round(float(engine.serviced["rank2.mte2"]), 6) == 4
    assert round(float(engine.serviced["rank2.mte3"]), 6) == 8
    assert round(float(engine.serviced["rank2.hbm"]), 6) == 12
    assert round(float(engine.serviced["rank2.vector"]), 6) == 3
    assert second_result["outputs"]["rank2-hist.histogram.complete"].time == 113
    assert {row["owner"] for row in engine.trace if row["demand"]} == {
        "owner-rank1", "owner-rank2"}


def test_unknown_local_count_has_no_successful_count_event():
    ids = (0, 0, 1)
    route = normalize_local_route(
        declared={"router_local_expert_ids": ids},
        geometry_counts=None, global_experts=256, local_range=None,
        rows=3)
    plan = plan_inbound(
        call_id="unknown-count", phase="prefill", hidden=64, experts=256,
        local_experts=None, received_rows=3, payload="fp4_e2m1",
        scale_cols=2, scale_pairs=1, route=route)
    assert plan["work"]["resources"]["mte3"] == 114
    assert plan["materialization"]["output_write_bytes"] == 114
    assert all("expert_tokens_count_or_cumsum" not in port.name
               for port in plan["fragment"].ports.event_out)
    engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    inputs = {port.name: engine.submit(port.name)
              for port in plan["fragment"].ports.tensor_in}
    result = execute_staged(conductor, plan, ready=ready, inputs=inputs)
    conductor.advance()
    assert all("expert_tokens_count_or_cumsum" not in name
               for name in result["outputs"])
    assert result["outputs"]


def test_two_inbound_occurrences_keep_separate_output_events():
    counts = (6,) + (0,) * 15
    route = normalize_local_route(
        declared={"router_expert_token_counts": counts,
                  "router_local_expert_ids": (0,) * 6},
        geometry_counts=None, global_experts=256, local_range=16, rows=6)
    def make(call_id):
        return plan_inbound(
            call_id=call_id, phase="prefill", hidden=64, experts=256,
            local_experts=16, received_rows=6, payload="fp4_e2m1",
            scale_cols=2, scale_pairs=1, route=route)
    first, second = make("occ-a"), make("occ-b")
    engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ])
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    result_a = execute_staged(
        conductor, first, ready=ready,
        inputs={port.name: engine.submit(f"a_{port.name}")
                for port in first["fragment"].ports.tensor_in})
    result_b = execute_staged(
        conductor, second, ready=ready,
        inputs={port.name: engine.submit(f"b_{port.name}")
                for port in second["fragment"].ports.tensor_in})
    conductor.advance()
    key = "expert_tokens_count_or_cumsum.complete"
    event_a = result_a["outputs"][f"occ-a.{key}"]
    event_b = result_b["outputs"][f"occ-b.{key}"]
    assert event_a is not event_b
    assert conductor.event(f"occ-a.{key}") is event_a
    assert conductor.event(f"occ-b.{key}") is event_b
    assert event_a.time is not None and event_b.time is not None


def test_341_rows_use_capacity_tiles_and_release_last_use():
    counts = INBOUND_A
    route = normalize_local_route(
        declared={"router_expert_token_counts": counts},
        geometry_counts=None, global_experts=256, local_range=None,
        rows=sum(counts))
    base = plan_inbound(
        call_id="tile-base", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(counts), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route)
    tile_bytes = 2 * (3072 + 192)
    plan = plan_inbound(
        call_id="tile-341", phase="prefill", hidden=6144, experts=256,
        local_experts=16, received_rows=sum(counts), payload="fp4_e2m1",
        scale_cols=192, scale_pairs=96, route=route,
        block_slots=tile_bytes)
    assert plan["work"]["resources"] == base["work"]["resources"]
    tail = [record for record in plan["stages"] if record["tail"]][-1]
    assert tail["free"][0].size == 1 * (3072 + 192)
    assert len([record for record in plan["stages"]
                if "read_e15" in record["name"]]) == 3
    engine = Engine([
        Resource("mte2", 1_000_000.0, unit="bytes"),
        Resource("mte3", 1_000_000.0, unit="bytes"),
        Resource("hbm", 1_000_000.0, unit="bytes"),
        Resource("vector", 1_000_000.0, unit="ops"),
    ], pools={"moe_inbound_block": tile_bytes})
    conductor = FragmentConductor(engine)
    ready = engine.submit("ready")
    inputs = {port.name: engine.submit(port.name)
              for port in plan["fragment"].ports.tensor_in}
    result = execute_staged(conductor, plan, ready=ready, inputs=inputs)
    conductor.advance()
    assert engine.peak_allocated["moe_inbound_block"] == tile_bytes
    assert not engine.allocations
    assert result["grants"] and result["release_handles"]
    assert result["outputs"]["tile-341.expanded_x.complete"].time is not None


def test_inbound_family_evaluate_carries_leaf_stage_occurrence():
    counts = INBOUND_A
    compiled = _inbound(extras=_with(
        INBOUND_BASE, router_expert_token_counts=counts,
        router_received_rows=sum(counts)))
    cost = compiled.evaluate(_geometry())
    occurrence = cost.native_occurrence
    assert occurrence is not None
    prepare = occurrence["prepare"]
    assert occurrence["status"] == "bound"
    assert prepare["work"]["resources"] == {
        "mte2": 1114388, "mte3": 1114516,
        "hbm": 2228904, "vector": 2095120}
    assert prepare["materialization"]["output_write_bytes"] == 1114516
    assert prepare["program"]["stage_count"] > 10
    assert any(record["tail"] for record in prepare["program"]["stages"])
    assert prepare["program"]["output_stages"][
        "expert_tokens_count_or_cumsum"] == "publish_count"
