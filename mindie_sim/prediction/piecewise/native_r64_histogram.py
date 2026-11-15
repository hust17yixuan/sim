"""R64 aclnnHistc expert-count histogram leaf with a typed common-program slice.

[Role] Expand the exact R64 ``TokenDispatcherWithAll2AllV._preprocess`` call
  ``torch.histc(topk_ids, bins=self.num_experts, min=0, max=self.num_experts)``
  into the observed ACLNN device contract: one INT32 ``[M, k]`` values tensor,
  two INT32 0-d scalar inputs, and one INT32 ``[E]`` count output.  This is the
  separate ``HistogramV2`` launch after ``MoeGatingTopK``, not the TopK call
  and not the later count all-gather.  A declared global token->expert route
  supplies the exact integer histogram values and offsets; with no route the
  work remains predictable but the output values stay unresolved.

[Invariants] Only the exact R64 image and the source-selected F1
  ``router_histogram`` occurrence are admitted.  ``M`` lives in the
  occurrence; ``E``, ``k``, ``bins=E``, ``min=0``, ``max=E``, the INT32
  producer/output contract and the caller blob stay in the binding.  The
  caller passes ``topk_ids`` directly, so no FP32 cast is claimed.  The ACLNN
  C header and the R64 histogram kernel binary are not retained; the prepare
  record therefore keeps ``complete=False`` and never claims a physically
  complete native execution, a fabricated workspace or a device launch.
  The separate leaf is admitted only when the compiled F1 ``moe_comm``
  selector resolves this geometry to ``alltoall`` (the All2AllV dispatcher);
  MC2, fused MC2, AllGather, other images and other router operators return
  ``None``.  Phase is deliberately not a proxy: a short prefill that selects
  MC2 contains the count in its fused call and is a negative here.

[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.

[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal geometry; no tensor allocation, device access or measurement is
  used to price the call.
"""
from __future__ import annotations

from typing import Any

from mindie_sim.schema.common import ModelingError
from .native_moe_token_permute import (
    MoeRoute, build_fragment, kernel_program, native_program, normalize_route,
    program_record, route_inputs, stage_record, stage_totals, work_record,
)

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
# Same R64 caller blob already used by the outgoing TokenPermute adapter.
R64_SOURCE_SHA = "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
R64_SOURCE_INTERVAL = "5929df13d..f7d558657^"
REVISION = "r64/8d422528"
ENTRY = "torch.histc"
ACLNN_API = "aclnnHistc"
ACLNN_WORKSPACE_API = "aclnnHistcGetWorkspaceSize"
KERNEL_NAME = "HistogramAiCore"
NATIVE_TYPE = "HistogramV2"
PROFILER_NAME = "aclnnHistc_HistogramAiCore_HistogramV2"
PROTOCOL_GATE = "MOE_COMM_SELECTED_ALLTOALL"
SOURCE_LINES = (643, 644)
CALL_LINE = 644
SOURCE_CALL_EXPRESSION = (
    "num_local_tokens_per_expert = torch.histc(topk_ids, "
    "bins=self.num_experts, min=0, max=self.num_experts)")
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    "sha256": R64_SOURCE_SHA,
    "symbol": "TokenDispatcherWithAll2AllV._preprocess",
    "revision": REVISION,
    "state": "task_artifact",
}
# Retained CANN 9.2.0 op library.  It is not the R64 image's own 9.2.1
# libopapi; it proves the ACLNN symbol and the AICore/op-type launch strings.
CANN_OP_LIBRARY = {
    "artifact": ("I-r31/selected/usr/local/Ascend/cann-9.2.0/"
                 "aarch64-linux/lib64/libopapi.so"),
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "version": "cann-9.2.0-retained",
    "symbols": (ACLNN_API, ACLNN_WORKSPACE_API),
    "op_types": ("Histogram", NATIVE_TYPE, "HistogramAiCore"),
    "internal_signature": (
        "l0op::Histogram(aclTensor const*, aclTensor const*, "
        "aclTensor const*, aclTensor const*, long, float, float, "
        "aclOpExecutor*)"),
}
OBSERVED = {
    "case": "glm51_dp16tp1ep16_pf_1k_c1",
    "declared_revision": "r64",
    "source_csv_sha256_prefix": "0223d793",
    "count": 75,
    "profiler_name": PROFILER_NAME,
    "native_type": NATIVE_TYPE,
    "input_shapes": "1024,8;;",
    "input_dtypes": "INT32;INT32;INT32",
    "output_shapes": "256",
    "output_dtypes": "INT32",
}
ROLE = "expert_count_histogram"
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    from . import native_calls
    return native_calls._r64_source_meta(meta)


def _moe_comm_declaration(compiled: Any, operator: Any) -> tuple[Any, str]:
    """The compiled F1 protocol declaration for this semantic histogram.

    The histogram signature itself carries ``moe_comm`` once the builder
    exposes it.  Before that narrow builder change lands, the same F1
    subgraph's ``router_permute`` sibling carries the identical source-selected
    ``MoeCommConfig``; reading it is still a compiled/lowered declaration, not
    a phase proxy.  No declaration means no protocol guess.
    """
    theta = getattr(operator, "theta", None)
    raw = theta.find("moe_comm", None) if theta is not None else None
    if raw is not None:
        return raw, "operator_theta"
    for sibling in getattr(compiled, "siblings", ()) or ():
        sibling_theta = getattr(sibling, "theta", None)
        if sibling_theta is None:
            continue
        raw = sibling_theta.find("moe_comm", None)
        if raw is not None:
            return raw, "sibling_router_permute_theta"
    return None, "unresolved_no_moe_comm_declaration"


def _static_contract(compiled: Any,
                     operator: Any) -> tuple[int, int, Any, str] | None:
    """Prove the semantic histogram's static E/k and protocol declaration."""
    template = getattr(operator, "template", None)
    if template is None or str(getattr(template, "role", "") or "") != "histogram":
        return None
    theta = getattr(operator, "theta", None)
    if theta is None:
        return None
    experts = _int(theta.find("E", 0))
    top_k = _int(theta.find("k", 0))
    elements = theta.find("elements", None)
    if experts < 1 or top_k < 1:
        return None
    # The F1 histogram signature is exactly one row of k assignment ids per
    # token.  A different semantic element count is another operator.
    if elements is None or _int(elements) != top_k:
        return None
    raw_comm, declaration_source = _moe_comm_declaration(compiled, operator)
    if raw_comm is None:
        return None
    return experts, top_k, raw_comm, declaration_source


def selected_protocol(binding: Any, geometry: Any) -> dict | None:
    """The actual source-selected MoE protocol for one evaluated geometry.

    This is the narrow recipe diagnostic owned by the histogram leaf.  It
    consumes the existing product selector; it does not reimplement or tune
    it.  ``separate_histc`` is true only for the All2AllV source dispatcher,
    which launches the separate ``torch.histc`` count.  MC2, fused MC2 and
    AllGather contain the count in their own call.
    """
    attrs = dict(binding.typed.get("attrs") or {})
    raw = attrs.get("moe_comm", None)
    if raw is None:
        return None
    from mindie_sim.prediction.moe_comm_selection import (
        MoeCommConfig, local_tokens)
    config = MoeCommConfig.from_items(raw)
    tokens = int(local_tokens(geometry))
    selected = str(config.select(tokens))
    return {
        "protocol": selected,
        "local_tokens": tokens,
        "capacity": int(config.capacity),
        "world_size": int(config.world_size),
        "gate": PROTOCOL_GATE,
        "separate_histc": selected == "alltoall",
        "declaration_source": str(
            attrs.get("protocol_declaration_source") or "unresolved"),
        "selection_source": "moe_comm_selection.MoeCommConfig.select",
    }


def leaf_selected(binding: Any, geometry: Any) -> bool:
    """True only when this geometry launches the separate histogram count."""
    protocol = selected_protocol(binding, geometry)
    return bool(protocol is not None and protocol["separate_histc"])


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 F1 histogram call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain.  Selection is the exact R64 image plus the compiled F1
    ``moe_comm`` protocol declaration; ``phase`` is accepted for the shared
    API but is deliberately not a proxy for the selected method.  The token
    axis stays an occurrence fact; the static E/k contract and the protocol
    selector stay in the binding.  A missing protocol declaration returns
    ``None`` rather than guessing from prefill/decode.
    """
    if str(getattr(compiled, "kind", "") or "") != "moe_router":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "family", "") or "") != "router"
            or str(getattr(operator, "name", "") or "") != "router_histogram"):
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    contract = _static_contract(compiled, operator)
    if contract is None:
        return None
    experts, top_k, raw_comm, declaration_source = contract
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        acl_api=ACLNN_API, op_type=NATIVE_TYPE,
        profiler_name=PROFILER_NAME,
        kernel_source_status="histogram_kernel_source_not_retained",
        source_status="captured_acl_label_and_retained_cann_op_library",
        acl_library_sha256=CANN_OP_LIBRARY["sha256"])
    typed = {
        "dtype": "int32",
        "layout": "ND",
        "shapes": {
            "values": [top_k],
            "min": [],
            "max": [],
            "histogram": [experts],
        },
        "attrs": {
            "native_kind": NATIVE_TYPE,
            "op_type": NATIVE_TYPE,
            "kernel_name": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "aclnn_workspace_api": ACLNN_WORKSPACE_API,
            "profiler_name": PROFILER_NAME,
            "role": ROLE,
            "protocol_gate": PROTOCOL_GATE,
            "protocol_declaration_source": declaration_source,
            "histogram_leaf_scope": "all_to_allv_only",
            "mc2_count_contract": (
                "contained_in_selected_mc2_count_call; not a separate leaf"),
            "moe_comm": raw_comm,
            "input_contract": ["values", "min", "max"],
            "output_contract": ["histogram"],
            "E": experts,
            "k": top_k,
            "bins": experts,
            "min_value": 0,
            "max_value": experts,
            "values_dtype": "int32",
            "scalar_dtype": "int32",
            "histogram_dtype": "int32",
            "source_argument": "topk_ids",
            "source_bins_expression": "self.num_experts",
            "source_min_expression": "0",
            "source_max_expression": "self.num_experts",
            "source_cast": "none; topk_ids passed directly",
            "source_sha256": SOURCE["sha256"],
            "source_symbol": SOURCE["symbol"],
            "source_lines": list(SOURCE_LINES),
            "source_call_line": CALL_LINE,
            "source_call_expression": SOURCE_CALL_EXPRESSION,
            "source_commit_anchor": R64_SOURCE_COMMIT,
            "source_commit_interval": R64_SOURCE_INTERVAL,
            "source_blob_invariant": 1,
            "acl_interface_status": (
                "captured_acl_label_and_retained_cann_symbols; "
                "c_header_not_retained"),
            "r64_acl_library_status": "r64_libopapi_not_retained",
            "kernel_source_status": "not_retained",
            "binary_selection_status": "uncertain",
            "deployment_image_id": str(
                meta.get("image_id") or meta.get("deployment_image_id")
                or R64_IMAGE_SHA),
            "deployment_source_revision": "r64",
        },
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=(
            "aclnn_c_header_not_retained",
            "r64_libopapi_not_retained",
            "histogram_kernel_source_not_retained",
            "kernel_binary_selection_uncertain",
            "source_commit_not_uniquely_pinned",
            "scalar_slot_order_not_header_verified",
        ))


def _histogram_route(compiled: Any, geometry: Any, attrs: Any, *,
                    experts: int, tokens: int, top_k: int) -> MoeRoute:
    """One normalized global route fact for the separate histogram call."""
    declared = route_inputs(compiled, attrs)
    geometry_source = ""
    geometry_counts = None
    if hasattr(geometry, "routing_for"):
        layer = int(getattr(compiled, "layer", 0) or 0)
        geometry_counts = geometry.routing_for(layer)
        geometry_source = f"geometry.routing[layer={layer}]"
    return normalize_route(
        declared=declared, geometry_counts=geometry_counts, scope="global",
        experts=int(experts), rows=int(tokens) * int(top_k),
        identity="topk_ids", tokens=int(tokens), top_k=int(top_k),
        geometry_source=geometry_source)


def _histogram_fragment(call_id: str, route: MoeRoute, *, tokens: int,
                        top_k: int, experts: int, protocol: dict, work: dict,
                        phase: str):
    attrs = (
        ("native_type", NATIVE_TYPE, "", "captured_native_type"),
        ("acl_api", ACLNN_API, "", "declared_source"),
        ("route_basis", route.basis, "", "occurrence"),
        ("route_layout_digest", route.layout_digest, "", "occurrence"),
        ("assignment_identity", route.to_dict()["assignment_identity"], "",
         "occurrence"),
        ("expert_token_counts", route.counts or (), "assignments", "occurrence"),
        ("expert_token_offsets", route.offsets or (), "assignments", "occurrence"),
        ("active_experts", route.active_experts, "experts", "occurrence"),
        ("values_rows", int(tokens), "rows", "geometry"),
        ("top_k", int(top_k), "experts_per_token", "binding"),
        ("bins", int(experts), "bins", "source_bins_expression"),
        ("min", 0, "expert_id", "source_min_expression"),
        ("max", int(experts), "expert_id", "source_max_expression"),
        ("selected_protocol", str(protocol.get("protocol") or ""), "",
         "moe_comm_selection"),
    )
    return build_fragment(
        name="moe.histogram", call_id=call_id, phase=phase,
        inputs=(("values", "int32", "expert_identity_matrix"),
                ("min", "int32", "scalar_expert_id"),
                ("max", "int32", "scalar_expert_id")),
        outputs=(("histogram", "int32", "per_global_expert_count"),),
        work=work, attrs=attrs,
        notes=("separate_histogram_v2_call", "not_the_topk_or_sort_call",
               "mc2_selected_call_contains_this_count"))


def _histogram_program(route: MoeRoute, work: dict,
                       fragment: Any, *, experts: int,
                       protocol: dict, stage_records: Any = None) -> dict:
    return program_record(
        common_program="moe.F1.router_histogram",
        stage="count_expert_assignments",
        work=work, fragment=fragment, route=route,
        stage_records=stage_records,
        decomposition={
            "api_calls": 1, "kernels": [KERNEL_NAME],
            "members": ["router_histogram"],
            "separate_from": ["MoeGatingTopK", "router_sort"],
            "not_charged": ["top-k scores", "sorted assignment rows"],
        },
        fusion_boundary={
            "owner": "F1:router:router_histogram",
            "charge": "one_per_native_call",
            "selected_protocol": str(protocol.get("protocol") or ""),
            "contained_in_mc2_count": True,
        },
        alignment={
            "family_template": "router_histogram",
            "captured_output_shape": [int(experts)],
            "captured_output_dtype": "int32",
            "native_write_bytes": work["resources"].get("mte3", 0.0),
            "family_envelope_status": (
                "generic router envelope lumps topk/histogram/sort and is "
                "not used for this separate call"),
            "note": ("the retained R64 contract is INT32[E]; the generic "
                     "router envelope is not used"),
        })


def _histogram_materialization(tokens: int, top_k: int, experts: int) -> dict:
    values = int(tokens) * int(top_k) * 4
    histogram = int(experts) * 4
    return {
        "inputs": {"values": values, "min": 0, "max": 0},
        "outputs": {"histogram": histogram},
        "input_read_bytes": values,
        "output_write_bytes": histogram,
        "total_bytes": values + histogram,
    }


def _histogram_stages(tokens: int, top_k: int, experts: int) -> list[dict]:
    rows = int(tokens) * int(top_k)
    return [
        stage_record("read_values", {"mte2": float(rows * 4),
                                     "hbm": float(rows * 4)}, coupled=True),
        stage_record("init_bins", {"vector": float(experts)}),
        stage_record("count", {"vector": float(rows)},
                     after=("read_values", "init_bins")),
        stage_record("write_bins", {"mte3": float(experts * 4),
                                    "hbm": float(experts * 4)}, coupled=True,
                     after=("count",)),
        stage_record("publish_histogram", {}, after=("write_bins",)),
    ]


def plan_histogram(*, call_id: str, phase: str, tokens: int, top_k: int,
                   experts: int, protocol: dict, route: MoeRoute) -> dict:
    """Single leaf supply for one normalized separate-histogram occurrence."""
    tokens, top_k, experts = int(tokens), int(top_k), int(experts)
    rows = tokens * top_k
    if route.rows is not None and int(route.rows) != rows:
        raise ModelingError(
            f"moe_plan_histogram_route_rows_mismatch:{route.rows}!={rows}")
    if route.experts is not None and int(route.experts) != experts:
        raise ModelingError(
            f"moe_plan_histogram_route_experts_mismatch:"
            f"{route.experts}!={experts}")
    actual_shapes = {
        "values": [tokens, top_k], "min": [], "max": [],
        "histogram": [experts],
    }
    values = _descriptor([tokens, top_k], "int32", "ND", "input")
    values.update({
        "producer": "MoeGatingTopK.topk_ids",
        "source_argument": "topk_ids",
        "dtype_evidence": (
            "captured INT32 device input and source passes topk_ids "
            "directly with no cast"),
    })
    min_scalar = _descriptor([], "int32", "ND", "input")
    min_scalar.update({
        "scalar_role": "min", "value": 0, "source_argument": "min",
        "device_representation": "captured 0-d INT32 input",
        "slot_order": "source_argument_order_inferred",
    })
    max_scalar = _descriptor([], "int32", "ND", "input")
    max_scalar.update({
        "scalar_role": "max", "value": experts, "source_argument": "max",
        "device_representation": "captured 0-d INT32 input",
        "slot_order": "source_argument_order_inferred",
    })
    histogram = _descriptor([experts], "int32", "ND", "output")
    histogram.update({
        "consumer": ("dispatch input_splits and "
                     "gather_from_sequence_parallel_region"),
        "count_semantics": "local top-k assignments per expert",
    })
    descriptors = {"values": values, "min": min_scalar, "max": max_scalar,
                   "histogram": histogram}
    arguments = {"values": "values", "bins": experts, "min": 0,
                 "max": experts}
    materialization = _histogram_materialization(tokens, top_k, experts)
    stages = _histogram_stages(tokens, top_k, experts)
    work = work_record(
        issued=stage_totals(stages),
        useful={"hbm": float(materialization["inputs"]["values"])},
        source="native_moe_stage_program")
    fragment = _histogram_fragment(
        call_id, route, tokens=tokens, top_k=top_k, experts=experts,
        protocol=protocol, work=work, phase=phase)
    logical = {
        "dtype": "int32", "layout": "ND", "shapes": actual_shapes,
        "values_dtype": "int32", "histogram_dtype": "int32",
        "scalar_dtype": "int32",
    }
    program = _histogram_program(
        route, work, fragment, experts=experts, protocol=protocol,
        stage_records=stages)
    allocation = {
        "inputs": "external_owner", "outputs": "external_owner",
        "internal": [], "workspace": "not_declared",
        "lifetimes": "owner_retained",
    }
    output_stages = {"histogram": "publish_histogram"}
    unrecognized = (
        "histogram_atomic_contention_not_retained",
    )
    program.update({
        "logical": logical, "materialization": materialization,
        "issued_work": work["resources"], "route": route.to_dict(),
        "allocation": allocation,
        "output_stages": dict(output_stages),
        "unrecognized_work": list(unrecognized),
    })
    return {
        "schema": "mindie-sim/moe-native-leaf-plan/1",
        "call": ENTRY,
        "call_id": str(call_id),
        "phase": str(phase),
        "route": route,
        "logical": logical,
        "descriptors": descriptors,
        "arguments": arguments,
        "materialization": materialization,
        "allocation": allocation,
        "unrecognized_work": unrecognized,
        "stages": tuple(stages),
        "execution_program": kernel_program(stages),
        "output_stages": output_stages,
        "work": work,
        "fragment": fragment,
        "program": program,
    }


def program_fragment(binding: Any, compiled: Any, geometry: Any):
    """Independent typed fragment for the selected separate histogram call."""
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("native_kind") != NATIVE_TYPE
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    protocol = selected_protocol(binding, geometry)
    if protocol is None or not protocol["separate_histc"]:
        return None
    experts = _int(attrs.get("E"))
    top_k = _int(attrs.get("k"))
    tokens = _int(protocol.get("local_tokens"))
    if experts < 1 or top_k < 1 or tokens < 1:
        return None
    route = _histogram_route(compiled, geometry, attrs, experts=experts,
                             tokens=tokens, top_k=top_k)
    plan = plan_histogram(
        call_id=binding.call_id,
        phase=str(getattr(geometry, "phase_name", "") or "prefill"),
        tokens=tokens, top_k=top_k, experts=experts, protocol=protocol,
        route=route)
    if plan.get("fragment") is None or plan.get("execution_program") is None:
        return None
    return native_program(plan)


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R64 histogram."""
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("native_kind") != NATIVE_TYPE
            or attrs.get("acl_api") != ACLNN_API
            or attrs.get("role") != ROLE
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    if (str(getattr(compiled, "kind", "") or "") != "moe_router"
            or str(getattr(getattr(compiled, "operator", None), "name", "")
                   or "") != "router_histogram"):
        return None
    protocol = selected_protocol(binding, geometry)
    if protocol is None or not protocol["separate_histc"]:
        return None
    experts = _int(attrs.get("E"))
    top_k = _int(attrs.get("k"))
    bins = _int(attrs.get("bins"), experts)
    if (experts < 1 or top_k < 1 or bins != experts
            or _int(attrs.get("min_value"), -1) != 0
            or _int(attrs.get("max_value"), -1) != experts):
        return None
    tokens = _int(protocol["local_tokens"])
    if tokens < 1:
        return None
    route = _histogram_route(compiled, geometry, attrs, experts=experts,
                             tokens=tokens, top_k=top_k)
    phase = str(getattr(geometry, "phase_name", "") or "prefill")
    plan = plan_histogram(
        call_id=binding.call_id, phase=phase, tokens=tokens, top_k=top_k,
        experts=experts, protocol=protocol, route=route)
    cache_key = (binding.call_id, tokens, route.basis,
                 route.performance_control, route.source, route.unresolved,
                 route.layout_digest)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = plan["logical"]["shapes"]
    actual_attrs = {
        **attrs,
        "tokens": tokens,
        "bins": bins,
        "min": 0,
        "max": experts,
        "selected_protocol": protocol["protocol"],
        "protocol_gate": PROTOCOL_GATE,
        "separate_histc": 1,
        "protocol_local_tokens": tokens,
        "route_basis": route.basis,
        "expert_token_counts": (list(route.counts)
                                if route.counts is not None else None),
        "expert_token_offsets": (list(route.offsets)
                                 if route.offsets is not None else None),
    }
    actual = {"dtype": "int32", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API,
                   "profiler_name": PROFILER_NAME,
                   "op_type": NATIVE_TYPE,
                   "source_status": "captured_acl_label_and_retained_cann_op_library",
                   "source_status_detail": "histogram_kernel_source_not_retained"})
    prepare = {
        "schema": "mindie-sim/histc-native-prepare/1",
        "call": ENTRY,
        "api": ACLNN_API,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": list(SOURCE_LINES),
        "source_call_line": CALL_LINE,
        "source_call_expression": SOURCE_CALL_EXPRESSION,
        "source_symbol": SOURCE["symbol"],
        "source_sha256": SOURCE["sha256"],
        "kernel": kernel,
        "protocol": protocol,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": plan["arguments"],
        "argument_order": ["values", "bins", "min", "max"],
        "positional_arguments": {"values": "values"},
        "keywords": {"bins": experts, "min": 0, "max": experts},
        "optional_slots": {"out": "not_passed_source_default"},
        "prototype": SOURCE_CALL_EXPRESSION,
        "prototype_status": (
            "exact_source_call_expression; aclnn_c_header_not_retained"),
        "shapes": actual_shapes,
        "descriptors": plan["descriptors"],
        "arrays": plan["descriptors"],
        "logical": plan["logical"],
        "materialization": plan["materialization"],
        "acl_interface": {
            "acl_api": ACLNN_API,
            "acl_workspace_api": ACLNN_WORKSPACE_API,
            "captured_profiler_label": PROFILER_NAME,
            "captured_native_type": NATIVE_TYPE,
            "captured_kernel": KERNEL_NAME,
            "captured_accelerator_core": "MIX_AIV",
            "captured_input_arity": 3,
            "captured_output_arity": 1,
            "retained_op_library": {
                "artifact": CANN_OP_LIBRARY["artifact"],
                "sha256": CANN_OP_LIBRARY["sha256"],
                "version": CANN_OP_LIBRARY["version"],
                "symbols": list(CANN_OP_LIBRARY["symbols"]),
                "op_types": list(CANN_OP_LIBRARY["op_types"]),
                "internal_signature": CANN_OP_LIBRARY["internal_signature"],
            },
            "python_wrapper_source": "retained",
            "aclnn_header": "not_retained",
            "r64_libopapi": "not_retained",
            "status": (
                "captured_label_and_retained_cann_op_library_only; "
                "c_header_not_retained"),
        },
        "device_abi": {
            "input_shapes": str(OBSERVED["input_shapes"]),
            "input_dtypes": str(OBSERVED["input_dtypes"]),
            "output_shapes": str(OBSERVED["output_shapes"]),
            "output_dtypes": str(OBSERVED["output_dtypes"]),
            "scalar_inputs": 2,
            "scalar_representation": "captured 0-d INT32 tensor inputs",
            "scalar_slot_order_status": (
                "min_then_max_inferred_from_source_call_order; "
                "not_header_verified"),
            "workspace_bytes": None,
            "workspace_status": "aclnn_get_workspace_size_not_retained",
        },
        "match": {
            "native_type": NATIVE_TYPE,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "tokens": tokens,
            "k": top_k,
            "E": experts,
            "bins": experts,
            "min": 0,
            "max": experts,
            "values_shape": [tokens, top_k],
            "histogram_shape": [experts],
            "values_dtype": "int32",
            "scalar_dtype": "int32",
            "histogram_dtype": "int32",
            "source_sha256": source.get("sha256"),
            "source_symbol": source.get("symbol"),
            "source_lines": list(SOURCE_LINES),
            "source_call_line": CALL_LINE,
            "protocol_gate": PROTOCOL_GATE,
            "selected_protocol": protocol["protocol"],
            "separate_histc": 1,
            "protocol_declaration_source": protocol["declaration_source"],
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "observed_original": dict(OBSERVED),
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id")
                            or R64_IMAGE_SHA),
            "revision": REVISION,
            "selection": (
                "exact R64 image id + F1 router_histogram + source-selected "
                "MoE protocol all_toallv + TokenDispatcherWithAll2AllV."
                "_preprocess caller blob; no phase proxy"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": R64_SOURCE_COMMIT,
            "source_commit_interval": R64_SOURCE_INTERVAL,
            "source_blob_commits_in_interval": 0,
            "source_commit_status": "caller_blob_invariant_over_interval",
            "protocol_gate": PROTOCOL_GATE,
            "selected_protocol": protocol["protocol"],
            "acl_library_sha256": CANN_OP_LIBRARY["sha256"],
            "acl_library_version": CANN_OP_LIBRARY["version"],
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": ("r64 histogram kernel binary and r64 libopapi are not "
                       "retained; captured ACLNN label and CANN 9.2.0 "
                       "symbols only"),
            "owner_kernel": KERNEL_NAME,
            "op_type": NATIVE_TYPE,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "acl_library_sha256": CANN_OP_LIBRARY["sha256"],
        },
        "route": route.to_dict(),
        "expert_token_counts": (list(route.counts)
                                if route.counts is not None else None),
        "expert_token_offsets": (list(route.offsets)
                                 if route.offsets is not None else None),
        "histogram_values": (list(route.counts)
                             if route.counts is not None else None),
        "assignment": {
            "identity_space": route.identity_space,
            "forward": (list(route.forward)
                        if route.forward is not None else None),
            "inverse": (list(route.inverse)
                        if route.inverse is not None else None),
            "tokens": (list(route.assignment_tokens)
                       if route.assignment_tokens is not None else None),
            "slots": (list(route.assignment_slots)
                      if route.assignment_slots is not None else None),
        },
        "work": plan["work"],
        "issued_work": plan["work"]["resources"],
        "program": plan["program"],
        "quant_metadata": {
            "values_dtype": "int32",
            "values_bytes": 4,
            "scalar_dtype": "int32",
            "scalar_representation": "captured 0-d INT32 tensor",
            "histogram_dtype": "int32",
            "histogram_bytes": 4,
            "histogram_values_semantics": "assignments per global expert",
            "source": "captured INT32 input/output contract",
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "aclnn_c_header_not_retained",
            *(["expert_token_counts_unresolved"]
              if route.counts is None else []),
            "r64_libopapi_not_retained",
            "histogram_kernel_source_not_retained",
            "kernel_binary_selection_uncertain",
            "source_commit_not_uniquely_pinned",
            "scalar_slot_order_not_header_verified",
        ],
        "unresolved": [
            "aclnn_c_header_not_retained",
            *list(route.unresolved),
            "r64_libopapi_not_retained",
            "histogram_kernel_source_not_retained",
            "kernel_binary_selection_uncertain",
            "source_commit_not_uniquely_pinned",
            "scalar_slot_order_not_header_verified",
            "execution_rank_not_selected",
            "workspace_not_prepared",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_complete_native_execution_claim",
            "no_python_wrapper_signature_beyond_source_expression",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": "int32", "layout": "ND",
                         "shapes": actual_shapes, "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
