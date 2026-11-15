"""R64 incoming MoeInitRoutingV3 leaf with a typed common-program slice.

[Role] Expand the real R64 All2All received-token
  ``torch_npu.npu_moe_init_routing_v2`` call from
  ``TokenDispatcherWithAll2AllV._dispatch_postprocess`` into the actual
  FP4/E2M1 payload, INT32 local-expert index, E8M0 MX scale and four output
  descriptors for the evaluated per-layer received-row count, and expose one
  typed `ProgramFragment`/work record for that native call.  Expert counts,
  offsets and the optional stable row layout are occurrence facts; they are
  never replaced by a balanced split unless the caller names that performance
  control.
[Invariants] The source identity is the exact R64 image digest
  ``8d422528...`` and the caller blob ``1638f673...`` whose ``device_op.py``
  is the R64 ``2a251259...`` blob.  The native top-k is the caller's
  ``unsqueeze(-1)`` (one local expert per received row), not the model top-k.
  ``R`` is the declared per-layer received-row ledger, never ``M*k``.
  ``E`` is the global expert count and ``E_local`` is the caller's
  ``[0, self.num_local_experts)`` range.  No tensor is allocated, no pointer
  or ACLNN symbol is fabricated and no device execution is claimed.  FP8
  W8A8-MX dispatch is a different payload feature and stays unbound here.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source
  selection and normal geometry; unknown source/values stay explicit.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from .native_moe_token_permute import (
    MoeRoute, _row_tiles, build_fragment, kernel_program, native_program,
    normalize_local_route, program_record, route_inputs, stage_record,
    stage_totals, work_record,
)

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
REVISION = "r64/8d422528"
ENTRY = "torch_npu.npu_moe_init_routing_v2"
ACLNN_API = "aclnnMoeInitRoutingV3"
OP_TYPE = "MoeInitRoutingV3"
PROFILER_NAME = (
    "aclnnMoeInitRoutingV3_MoeInitRoutingV3_MoeInitRoutingV3")
PAYLOAD = "fp4_e2m1"
SCALE_FORMAT = "fp8_e8m0"
NATIVE_TOP_K = 1

CALLER_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    "sha256": "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5",
    "symbol": "TokenDispatcherWithAll2AllV._dispatch_postprocess",
    "revision": REVISION,
    "state": "task_artifact",
}
ENTRY_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": R64_DEVICE_OP_SHA,
    "symbol": "A5DeviceAdaptor.npu_moe_init_routing",
    "revision": REVISION,
    "state": "task_artifact",
}
SCALE_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py",
    "sha256": "cbf45467019febc009d26d23492a7a4b5a7784a9f047725c58aa8558998bff0e",
    "symbol": "W4A4MXFP4LinearMethod.apply_act_quant",
    "revision": REVISION,
    "state": "task_artifact",
}
SCALE_LAYOUT_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/fused_moe/moe_utils.py",
    "sha256": "2c3dc13f2f7955f0e077df07adf8538929404a8f0af35a0ea9da42d42bcf5b5a",
    "symbol": "maybe_normalize_mxfp_scale_layout",
    "revision": REVISION,
    "state": "task_artifact",
}
KERNEL_SOURCE = {
    "repository": "ops-transformer",
    "path": ("moe/moe_init_routing_v3/op_host/op_api/"
             "aclnn_moe_init_routing_v3.h"),
    "sha256": "b2a04bfe9c59ce839238586d206a3b02657f2975ee58588f2c11c388b1401852",
    "symbol": "aclnnMoeInitRoutingV3GetWorkspaceSize",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "state": "task_artifact",
}
KERNEL_BODY_SOURCE = {
    "repository": "ops-transformer",
    "path": ("moe/moe_init_routing_v3/op_kernel/"
             "moe_init_routing_v3_apt.cpp"),
    "sha256": "ea72941cb47a2d3b27ff28a0d01eacf65192406c9e01418003faf8a665458fbd",
    "symbol": OP_TYPE,
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "state": "task_artifact",
}
TILING_SOURCE = {
    "repository": "ops-transformer",
    "path": ("moe/moe_init_routing_v3/op_host/"
             "moe_init_routing_v3_tiling_arch35.cpp"),
    "sha256": "1507153599d4ccae0cc056dc5b1e4f4849c02cf97893736f3d6913a43b52385d",
    "symbol": "MoeInitRoutingV3TilingArch35::DoOpTiling",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "state": "task_artifact",
}
SOURCE_LINES = (704, 715)
CALL_LINES = (706, 714)
ENTRY_LINES = (1186, 1211)
SCALE_LINES = (446, 450)
SCALE_LAYOUT_LINES = (368, 378)
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _descriptor(shape: list[Any], dtype: str, layout: str, role: str,
                **extra: Any) -> dict:
    row = {"shape": [None if item is None else int(item) for item in shape],
           "dtype": dtype, "layout": layout, "role": role, "pointer": None,
           "allocation": "external"}
    row.update({key: value for key, value in extra.items()
                if value is not None})
    return row


def _int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive(value: Any) -> int | None:
    result = _int(value)
    return result if result is not None and result > 0 else None


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed R64 image and caller blob; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    if source_revision_from_oci(image, image_id) != "r64":
        return {}
    if R64_IMAGE_SHA not in image_id and R64_IMAGE_SHA not in image:
        return {}
    return meta


def _normalize_payload(value: Any) -> str | None:
    key = str(value or "").strip().lower().replace("-", "_")
    if key in {"fp4", "fp4_e2m1", "mxfp4", "mx_fp4", "mxfp4_e2m1"}:
        return PAYLOAD
    return None


def _payload_bytes(dtype: str) -> float | None:
    from mindie_sim.schema.common import dtype_bytes
    try:
        return float(dtype_bytes(dtype))
    except Exception:
        return None


def _moe_comm(theta: Any) -> dict[str, Any]:
    raw = theta.find("moe_comm", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        return dict(raw or ())
    except (TypeError, ValueError):
        return {}


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 incoming MoeInitRoutingV3 call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Decode, non-R64 images, the outgoing
    permute/unpermute rows, the weighted finalize and MC2 dispatch stay
    unbound.  The selected routed-expert payload must be the MXFP4 feature
    observed on the original R64 GLM51 prefill; a mismatched payload byte
    count is not rewritten into a guessed V3 call.
    """
    if str(getattr(compiled, "kind", "") or "") != "routed_experts":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "family", "") or "") != "router"
            or str(getattr(operator, "name", "") or "") != "router_inbound"):
        return None
    if str(phase or "") != "prefill":
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    theta = operator.theta
    operation = theta.find("router_source_operation", None)
    if operation != "MoeInitRoutingV3":
        return None
    selected = str(theta.find("router_selected", "") or "")
    if selected != "post_dispatch_received_rows":
        return None
    payload = _normalize_payload(theta.find("router_source_dtype", ""))
    if payload != PAYLOAD:
        return None
    raw_routed_format = theta.find("routed_weight_format", None)
    if (raw_routed_format is not None
            and _normalize_payload(raw_routed_format) != PAYLOAD):
        return None
    hidden = _positive(theta.find("router_hidden", 0))
    experts = _positive(theta.find("E", 0))
    model_top_k = _positive(theta.find("k", 0))
    scale_cols = _positive(theta.find("router_scale_cols", 0))
    index_bytes = _positive(theta.find("router_index_bytes", 0))
    payload_bytes = theta.find("router_payload_bytes", None)
    if (hidden is None or experts is None or model_top_k is None
            or scale_cols is None or index_bytes != 4):
        return None
    expected_bytes = _payload_bytes(payload)
    if expected_bytes is None or payload_bytes is None:
        return None
    try:
        if abs(float(payload_bytes) - expected_bytes) > 1e-9:
            return None
    except (TypeError, ValueError):
        return None
    # A5 arch35 checks scale dim1 = ceil(H/64) pairs of two E8M0 bytes.
    # The quantizer emits ceil(H/32) E8M0 values and normalizes them to
    # [rows, ceil(H/64), 2]; router_scale_cols is the logical E8M0 count.
    # An odd logical count is padded by one E8M0 value in the last pair, so
    # a new H with ceil(H/32) odd is still a valid typed shape.
    if scale_cols != (hidden + 31) // 32:
        return None
    if str(theta.find("router_scale_dtype", "") or "") != SCALE_FORMAT:
        return None
    raw_received = theta.find("router_received_rows", None)
    received = _positive(raw_received)
    if raw_received is not None and received is None:
        return None
    comm = _moe_comm(theta)
    if str(comm.get("mode") or "") == "always_mc2":
        return None
    if bool(comm.get("enable_prefill_mc2")):
        return None
    local = _positive(comm.get("experts_per_device"))
    scale_pairs = (scale_cols + 1) // 2
    from . import native_calls
    source = native_calls._source(
        CALLER_SOURCE["repository"], CALLER_SOURCE["path"],
        sha256=CALLER_SOURCE["sha256"], symbol=CALLER_SOURCE["symbol"],
        revision=CALLER_SOURCE["revision"], state=CALLER_SOURCE["state"])
    kernel = native_calls._kernel(
        OP_TYPE, "vector", path=KERNEL_BODY_SOURCE["path"],
        sha256=KERNEL_BODY_SOURCE["sha256"], evidence="source_declared",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        source_commit=KERNEL_BODY_SOURCE["revision"],
        kernel_source_status="source_retained")
    typed_shapes: dict[str, list[int]] = {
        "x": [hidden],
        "expert_idx": [NATIVE_TOP_K],
        "scale": [scale_pairs, 2],
        "expanded_x": [hidden],
        "expanded_scale": [scale_pairs, 2],
    }
    if local is not None:
        typed_shapes["expert_tokens_count_or_cumsum"] = [local]
    image_id = str(meta.get("image_id") or meta.get("deployment_image_id")
                   or R64_IMAGE_SHA)
    typed = {
        "dtype": payload,
        "layout": "ND",
        "phase": "prefill",
        "shapes": typed_shapes,
        "attrs": {
            "moe_init_routing_v3": True,
            "native_api": ENTRY,
            "aclnn_api": ACLNN_API,
            "op_type": OP_TYPE,
            "profiler_name": PROFILER_NAME,
            "router_source_operation": "MoeInitRoutingV3",
            "router_selected": selected,
            "router_hidden": hidden,
            "hidden": hidden,
            "E": experts,
            "expert_num": experts,
            "k": model_top_k,
            "model_top_k": model_top_k,
            "native_top_k": NATIVE_TOP_K,
            "router_received_rows": received,
            "received_rows_source": "per-layer router_received_rows ledger",
            "moe_comm_mode": str(comm.get("mode") or ""),
            "moe_comm_policy": str(comm.get("policy") or ""),
            "moe_comm_prefill_mc2": int(bool(comm.get("enable_prefill_mc2"))),
            "expert_num_source": "compiled model/layer num_experts (E)",
            "router_payload_format": payload,
            "router_payload_format_source": (
                "source-selected routed_expert activation"),
            "router_payload_bytes": expected_bytes,
            "payload_family": "mxfp4",
            "x_dtype_runtime": "torch_npu.float4_e2m1fn_x2",
            "x_dtype_runtime_source": (
                "W4A4MXFP4 x_dtype (w4a4_mxfp4.py:439)"),
            "router_scale_format": SCALE_FORMAT,
            "router_scale_cols": scale_cols,
            "router_scale_pairs": scale_pairs,
            "router_index_bytes": index_bytes,
            "quant_mode": -1,
            "drop_pad_mode": 0,
            "expert_capacity": -1,
            "expert_tokens_num_type": 1,
            "expert_tokens_num_flag": True,
            "row_idx_type": 0,
            "active_num": -1,
            "offset": None,
            "caller_source_sha256": CALLER_SOURCE["sha256"],
            "caller_source_lines": list(CALL_LINES),
            "method_source_lines": list(SOURCE_LINES),
            "entry_source_sha256": R64_DEVICE_OP_SHA,
            "entry_source_lines": list(ENTRY_LINES),
            "source_commit": R64_SOURCE_COMMIT,
            "deployment_image_id": image_id,
            "deployment_source_revision": "r64",
            "observed_pf_payload_format": PAYLOAD,
            "scale_producer_sha256": SCALE_SOURCE["sha256"],
            "scale_producer_lines": list(SCALE_LINES),
            "scale_layout_source_sha256": SCALE_LAYOUT_SOURCE["sha256"],
            "scale_layout_lines": list(SCALE_LAYOUT_LINES),
        },
        "rank_width": 2,
    }
    if local is not None:
        typed["attrs"]["local_experts"] = int(local)
        typed["attrs"]["local_experts_source"] = (
            "moe_comm.experts_per_device")
        typed["attrs"]["active_expert_range"] = [0, int(local)]
        typed["attrs"]["active_expert_range_source"] = (
            "caller [0, num_local_experts) via moe_comm.experts_per_device")
    if raw_routed_format is not None:
        typed["attrs"]["routed_weight_format"] = str(raw_routed_format)
    unresolved = [
        "torch_npu_op_plugin_bridge_not_retained",
        "kernel_binary_selection_unresolved",
        "runtime_inputs_not_executed",
        "active_num_caller_default_unobserved",
        "row_idx_type_caller_omitted",
        "mxfp4_scale_layout_from_quantizer",
    ]
    api = native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                            version=REVISION)
    api.update({
        "entry_source_sha256": R64_DEVICE_OP_SHA,
        "entry_source_lines": list(ENTRY_LINES),
        "caller_source_sha256": CALLER_SOURCE["sha256"],
        "caller_source_lines": list(CALL_LINES),
    })
    return native_calls._call_spec(
        api, source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv", unresolved=unresolved)


def _received_rows(attrs: Mapping[str, Any]) -> int | None:
    """The selected call's declared received-row ledger, never a fallback.

    The binding identity already carries the static per-layer value read from
    the source-owned theta.  A missing value stays symbolic so this occurrence
    cannot silently change the identity that was bound.
    """
    return _positive(attrs.get("router_received_rows"))


def _inbound_route(compiled: Any, geometry: Any, attrs: Any, *,
                   experts: int, local: int | None,
                   rows: int | None) -> tuple[MoeRoute, int | None]:
    """One normalized local route fact for this received-row occurrence."""
    declared = route_inputs(compiled, attrs)
    if local is not None and int(local) > int(experts):
        raise ModelingError(
            "moe_route_local_experts_exceed_global:"
            f"{int(local)}!={int(experts)}")
    geometry_source = ""
    geometry_counts = None
    if hasattr(geometry, "routing_for"):
        layer = int(getattr(compiled, "layer", 0) or 0)
        geometry_counts = geometry.routing_for(layer)
        geometry_source = f"geometry.routing[layer={layer}]"
    route = normalize_local_route(
        declared=declared, geometry_counts=geometry_counts,
        global_experts=int(experts), local_range=local, rows=rows,
        geometry_source=geometry_source)
    return route, route.expert_range


def _inbound_fragment(call_id: str, route: MoeRoute, *, rows: int, hidden: int,
                      experts: int, local: int | None, payload: str,
                      scale_cols: int, scale_pairs: int, work: dict):
    attrs = (
        ("native_api", ENTRY, "", "declared_source"),
        ("aclnn_api", ACLNN_API, "", "declared_source"),
        ("route_basis", route.basis, "", "occurrence"),
        ("route_layout_digest", route.layout_digest, "", "occurrence"),
        ("expanded_row_idx_direction", "inverse_assignment", "",
         "row_idx_type_0"),
        ("assignment_identity", route.to_dict()["assignment_identity"], "",
         "occurrence"),
        ("expert_token_counts", route.counts or (), "rows", "occurrence"),
        ("expert_token_offsets", route.offsets or (), "rows", "occurrence"),
        ("active_experts", route.active_experts, "experts", "occurrence"),
        ("received_rows", int(rows), "rows", "router_received_rows_ledger"),
        ("hidden", int(hidden), "elements", "binding"),
        ("payload_format", payload, "", "selected_source_payload"),
        ("payload_row_bytes", (int(hidden) + 1) // 2, "bytes", "packed_2_per_byte"),
        ("scale_logical_cols", int(scale_cols), "values", "ceil_hidden_div_32"),
        ("scale_pairs", int(scale_pairs), "pairs", "ceil_scale_cols_div_2"),
        ("expert_num", int(experts), "experts", "binding"),
        ("local_experts", local, "experts", "active_expert_range"),
    )
    return build_fragment(
        name="moe.init_routing_v3", call_id=call_id, phase="prefill",
        inputs=(("x", payload, "packed_2_per_byte"),
                ("expert_idx", "int32", "local_expert_index"),
                ("scale", SCALE_FORMAT, "scale_pair")),
        outputs=(("expanded_x", payload, "packed_2_per_byte"),
                 ("expanded_row_idx", "int32",
                  "inverse_gather_row_index"),
                 ("expert_tokens_count_or_cumsum", "int64",
                  "per_local_expert_count"),
                 ("expanded_scale", SCALE_FORMAT, "scale_pair")),
        work=work, attrs=attrs,
        event_outputs=tuple(
            port for port in ("expanded_x", "expanded_row_idx",
                              "expert_tokens_count_or_cumsum", "expanded_scale")
            if not (port == "expert_tokens_count_or_cumsum"
                    and local is None)),
        notes=("single_api_call", "no_grouped_expert_compute_charged",
               "no_dispatch_return_bytes_charged"))


def _inbound_program(route: MoeRoute, work: dict | None,
                     fragment: Any, *, rows: int | None, hidden: int,
                     experts: int, local: int | None, scale_cols: int,
                     scale_pairs: int, stage_records: Any = None) -> dict | None:
    if work is None or fragment is None:
        return None
    family_row = float(hidden) * 0.5 + float(scale_cols) + 4.0
    return program_record(
        common_program="moe.F3.router_inbound",
        stage="gather_received_rows_by_local_expert",
        work=work, fragment=fragment, route=route, stage_records=stage_records,
        decomposition={
            "api_calls": 1, "kernels": [OP_TYPE],
            "members": ["router_inbound"],
            "separate_from": ["grouped_expert_matmul", "dispatch_alltoall"],
            "not_charged": ["grouped expert compute", "return transport"],
        },
        fusion_boundary={
            "owner": "F3:router:router_inbound",
            "charge": "one_per_native_call",
            "fused_mc2_contains_dispatch": True,
            "standalone_alltoall_receive_only": True,
        },
        alignment={
            "family_template": "router_inbound",
            "family_row_bytes": family_row,
            "family_read_bytes": float(int(rows) * family_row),
            "family_write_bytes": float(int(rows) * family_row),
            "native_read_bytes": work["resources"].get("mte2", 0.0),
            "native_write_bytes": work["resources"].get("mte3", 0.0),
            "additional_typed_output": {
                "expert_tokens_count_or_cumsum_bytes": float(
                    (int(local) if local is not None else 0) * 8)},
            "vector_basis": "R*H gather elements (same as family template)",
            "note": ("native write includes the INT64 per-local-expert count; "
                     "the shared family row envelope does not"),
        })


def _inbound_materialization(rows: int, hidden: int, local: int | None,
                            scale_pairs: int) -> dict:
    payload_bytes = rows * ((hidden + 1) // 2)
    scale_bytes = rows * scale_pairs * 2
    row_index_bytes = rows * 4
    count_bytes = None if local is None else int(local) * 8
    inputs = payload_bytes + scale_bytes + row_index_bytes
    known_outputs = inputs
    outputs = inputs + (count_bytes or 0)
    return {
        "inputs": {
            "x": payload_bytes,
            "expert_idx": row_index_bytes,
            "scale": scale_bytes,
        },
        "outputs": {
            "expanded_x": payload_bytes,
            "expanded_row_idx": row_index_bytes,
            "expert_tokens_count_or_cumsum": count_bytes,
            "expanded_scale": scale_bytes,
        },
        "input_read_bytes": inputs,
        "output_write_bytes": outputs,
        "known_output_write_bytes": known_outputs,
        "count_write_bytes": count_bytes,
        "complete": count_bytes is not None,
        "total_bytes": inputs + outputs,
    }


def _inbound_stages(route: MoeRoute, *, hidden: int, scale_pairs: int,
                    rows: int, local: int | None,
                    block_slots: int | None) -> list[dict]:
    """Real index/count/inverse and payload-read stages with byte slots."""
    from mindie_sim.schema.operators import AllocationEffect
    payload_row = (int(hidden) + 1) // 2
    scale_row = int(scale_pairs) * 2
    transfer_row = payload_row + scale_row
    index_row = 4
    records: list[dict] = [
        stage_record("read_index", {"mte2": int(rows) * index_row,
                                    "hbm": int(rows) * index_row},
                     coupled=True),
    ]
    if local is not None:
        records.append(stage_record(
            "count_prefix", {"vector": int(local)},
            after=("read_index",)))
        records.append(stage_record(
            "count_write", {"mte3": int(local) * 8,
                            "hbm": int(local) * 8},
            after=("count_prefix",)))
        index_after = "count_prefix"
    else:
        index_after = "read_index"
    records.append(stage_record(
        "write_row_idx", {"mte3": int(rows) * index_row,
                          "hbm": int(rows) * index_row},
        after=(index_after,), coupled=True))
    assignment_after = (("count_prefix",) if local is not None
                        else ("read_index",))

    def grant(label: str, size: int):
        if block_slots is None:
            return (), ()
        if size > int(block_slots):
            raise ModelingError(
                f"moe_plan_inbound_block_exceeds_workspace:{size}!={block_slots}")
        name = f"block_slot_{label}"
        acquire = (AllocationEffect.acquire(name, "moe_inbound_block", size),)
        release = (AllocationEffect.release(name, "moe_inbound_block", size),)
        return acquire, release

    if route.counts is not None and route.counts_complete:
        groups = [(index, value) for index, value in enumerate(route.counts)
                  if value > 0]
    else:
        groups = list(route.observed_counts)
    if not groups and rows > 0:
        groups = [(-1, int(rows))]
    write_names: list[str] = []
    for position, (expert, count) in enumerate(groups):
        base = "all" if expert < 0 else f"e{expert}"
        tiles = _row_tiles(count, block_slots, transfer_row)
        for tile_index, (start, end) in enumerate(tiles):
            tile_count = end - start
            label = (base if len(tiles) == 1
                     else f"{base}t{tile_index}")
            read = f"read_{label}"
            gather = f"gather_{label}"
            write = f"write_{label}"
            block_bytes = tile_count * transfer_row
            acquire, release = grant(label, block_bytes)
            records.append(stage_record(
                read, {"mte2": block_bytes, "hbm": block_bytes},
                acquire=acquire, coupled=True))
            records.append(stage_record(
                gather, {"vector": tile_count * int(hidden)},
                after=(read, *assignment_after)))
            records.append(stage_record(
                write, {"mte3": block_bytes, "hbm": block_bytes},
                after=(gather,), free=release,
                tail=(position == len(groups) - 1
                      and tile_index == len(tiles) - 1), coupled=True))
            write_names.append(write)
    if local is not None:
        records.append(stage_record("publish_count", {},
                                    after=("count_write",)))
    records.append(stage_record("publish_row_idx", {},
                                after=("write_row_idx",)))
    if write_names:
        records.append(stage_record("publish_payload", {},
                                    after=tuple(write_names)))
    return records


def plan_inbound(*, call_id: str, phase: str, hidden: int, experts: int,
                 local_experts: int | None, received_rows: int | None,
                 payload: str, scale_cols: int, scale_pairs: int,
                 route: MoeRoute, block_slots: int | None = None) -> dict:
    """Single leaf supply for one normalized incoming-V3 occurrence.

    All descriptors, logical shapes, materialization and issued work below are
    derived from these normalized parameters and the route object; callers do
    not maintain a second work formula.
    """
    rows = None if received_rows is None else int(received_rows)
    hidden = int(hidden)
    local = None if local_experts is None else int(local_experts)
    if route.rows is not None and rows is not None and int(route.rows) != rows:
        raise ModelingError(
            f"moe_plan_inbound_route_rows_mismatch:{route.rows}!={rows}")
    if (local is not None and route.experts is not None
            and int(route.experts) != local):
        raise ModelingError(
            f"moe_plan_inbound_route_experts_mismatch:{route.experts}!={local}")
    storage_pairs = (hidden + 1) // 2
    shape_r = rows
    actual_shapes: dict[str, list[Any]] = {
        "x": [shape_r, hidden],
        "expert_idx": [shape_r, NATIVE_TOP_K],
        "scale": [shape_r, int(scale_pairs), 2],
        "expanded_x": [shape_r, hidden],
        "expanded_row_idx": [shape_r],
        "expert_tokens_count_or_cumsum": (
            [None] if local is None else [local]),
        "expanded_scale": [shape_r, int(scale_pairs), 2],
    }
    descriptors = {
        "x": _descriptor(
            [shape_r, hidden], payload, "ND", "input",
            storage_bits=4, storage="packed_2_per_byte",
            storage_dtype="uint8", storage_shape=[shape_r, storage_pairs],
            source_role="received_routed_payload"),
        "expert_idx": _descriptor(
            [shape_r, NATIVE_TOP_K], "int32", "ND", "input",
            source_role="local_expert_index_after_dispatch"),
        "scale": _descriptor(
            [shape_r, int(scale_pairs), 2], SCALE_FORMAT, "ND", "input",
            observed_profiler_format="NCL",
            source_role="mxfp4_scale_pairs_from_quantizer"),
        "expanded_x": _descriptor(
            [shape_r, hidden], payload, "ND", "output",
            storage_bits=4, storage="packed_2_per_byte",
            storage_dtype="uint8", storage_shape=[shape_r, storage_pairs]),
        "expanded_row_idx": _descriptor(
            [shape_r], "int32", "ND", "output",
            mapping="gather_inverse_row_index"),
        "expert_tokens_count_or_cumsum": _descriptor(
            ([None] if local is None else [local]), "int64", "ND", "output",
            mode="count"),
        "expanded_scale": _descriptor(
            [shape_r, int(scale_pairs), 2], SCALE_FORMAT, "ND", "output",
            source_role="gather_copied_mxfp4_scale"),
    }
    passed_arguments = {
        "x": "x", "expert_idx": "expert_idx", "scale": "scale",
        "expert_num": int(experts),
        "expert_tokens_num_flag": True,
        "active_expert_range": ([0, local] if local is not None else None),
        "x_dtype": payload,
    }
    omitted_defaults = {
        "active_num": -1, "expert_capacity": -1, "drop_pad_mode": 0,
        "expert_tokens_num_type": 1, "quant_mode": -1, "row_idx_type": 0,
        "offset": None,
    }
    materialization = (_inbound_materialization(
        rows, hidden, local, int(scale_pairs)) if rows is not None else None)
    stages = (_inbound_stages(
        route, hidden=hidden, scale_pairs=int(scale_pairs), rows=rows,
        local=local, block_slots=block_slots) if rows is not None else [])
    totals = stage_totals(stages)
    work = (work_record(
        issued=totals,
        useful={"hbm": float(2 * int(rows) * ((int(hidden) + 1) // 2))},
        source="native_moe_stage_program") if rows is not None else None)
    if work is not None and local is None:
        work["count_write_bytes"] = None
        work["issued_work_complete"] = False
        work["unresolved"] = ["expert_tokens_count_write_unknown"]
    fragment = (_inbound_fragment(
        call_id, route, rows=rows, hidden=hidden, experts=int(experts),
        local=local, payload=payload, scale_cols=int(scale_cols),
        scale_pairs=int(scale_pairs), work=work) if work is not None else None)
    logical = {
        "dtype": payload, "layout": "ND", "shapes": actual_shapes,
        "native_top_k": NATIVE_TOP_K, "model_top_k": None,
    }
    program = _inbound_program(
        route, work, fragment, rows=rows, hidden=hidden, experts=int(experts),
        local=local, scale_cols=int(scale_cols), scale_pairs=int(scale_pairs),
        stage_records=stages)
    allocation = {
        "inputs": "external_owner", "outputs": "external_owner",
        "internal": [], "workspace": "not_declared",
        "lifetimes": "owner_retained",
    }
    if block_slots is not None:
        allocation["internal"] = [{
            "name": "moe_inbound_block", "pool": "moe_inbound_block",
            "capacity_bytes": int(block_slots), "lifetime": "per_block"}]
        allocation["workspace"] = {
            "kind": "row_tile", "pool": "moe_inbound_block",
            "capacity_bytes": int(block_slots), "last_use": "block_write"}
    unrecognized = [
        "sort_algorithm_selection_not_retained",
        "source_row_gather_locality_not_modeled",
        "tile_assignment_boundary_not_retained",
    ]
    if route.counts is None:
        unrecognized.append("expert_tokens_count_values_unknown")
    if local is None:
        unrecognized.append("expert_tokens_count_write_unknown")
    output_stages = {
        "expert_tokens_count_or_cumsum": (
            "publish_count" if local is not None else None),
        "expanded_row_idx": (
            "publish_row_idx" if route.inverse is not None else None),
        "expanded_x": "publish_payload",
        "expanded_scale": "publish_payload",
    }
    if program is not None:
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
        "arguments": {**passed_arguments, **omitted_defaults},
        "passed_arguments": passed_arguments,
        "omitted_defaults": omitted_defaults,
        "rounding": {
            "payload_row_bytes": storage_pairs,
            "payload_row_padding_bits": (storage_pairs * 2 - hidden) * 4,
            "scale_logical_cols": int(scale_cols),
            "scale_pairs": int(scale_pairs),
            "scale_padding_values": int(scale_pairs) * 2 - int(scale_cols),
        },
        "materialization": materialization,
        "allocation": allocation,
        "unrecognized_work": tuple(unrecognized),
        "stages": tuple(stages),
        "execution_program": (kernel_program(stages) if stages else None),
        "output_stages": output_stages,
        "work": work,
        "fragment": fragment,
        "program": program,
    }


def program_fragment(binding: Any, compiled: Any, geometry: Any):
    """Independent typed fragment for the selected incoming V3 occurrence."""
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("moe_init_routing_v3") is not True
            or attrs.get("native_api") != ENTRY
            or attrs.get("aclnn_api") != ACLNN_API
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    hidden = _positive(attrs.get("router_hidden"))
    experts = _positive(attrs.get("E"))
    scale_cols = _positive(attrs.get("router_scale_cols"))
    payload = _normalize_payload(attrs.get("router_payload_format"))
    if (hidden is None or experts is None or scale_cols is None
            or payload != PAYLOAD):
        return None
    scale_pairs = _positive(attrs.get("router_scale_pairs"))
    if scale_pairs is None:
        scale_pairs = (scale_cols + 1) // 2
    local = _positive(attrs.get("local_experts"))
    rows = _received_rows(attrs)
    route, local = _inbound_route(
        compiled, geometry, attrs, experts=experts, local=local, rows=rows)
    if route.rows is not None:
        rows = route.rows
    plan = plan_inbound(
        call_id=binding.call_id, phase="prefill", hidden=hidden,
        experts=experts, local_experts=local, received_rows=rows,
        payload=payload, scale_cols=scale_cols, scale_pairs=scale_pairs,
        route=route)
    if plan.get("fragment") is None or plan.get("execution_program") is None:
        return None
    return native_program(plan)


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected incoming V3 call.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  ``R`` is the caller's received-row ledger;
    a missing value stays symbolic and partial instead of being replaced by
    ``M*k``.  The physical owner is the ``router_inbound`` call itself and is
    charged once; no other subgraph member is charged here.
    """
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("moe_init_routing_v3") is not True
            or attrs.get("native_api") != ENTRY
            or attrs.get("aclnn_api") != ACLNN_API
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    if int(getattr(geometry, "effective_tokens", 0) or 0) < 1:
        return None
    hidden = _positive(attrs.get("router_hidden"))
    experts = _positive(attrs.get("E"))
    model_top_k = _positive(attrs.get("k"))
    scale_cols = _positive(attrs.get("router_scale_cols"))
    scale_pairs = _positive(attrs.get("router_scale_pairs"))
    payload = _normalize_payload(attrs.get("router_payload_format"))
    if (hidden is None or experts is None or model_top_k is None
            or scale_cols is None or payload != PAYLOAD):
        return None
    if scale_pairs is None:
        scale_pairs = (scale_cols + 1) // 2
    local = _positive(attrs.get("local_experts"))
    received = _received_rows(attrs)
    if received is not None and received < 1:
        return None
    route, local = _inbound_route(
        compiled, geometry, attrs, experts=experts, local=local,
        rows=received)
    if route.rows is not None:
        received = route.rows
    plan = plan_inbound(
        call_id=binding.call_id, phase="prefill", hidden=hidden,
        experts=experts, local_experts=local, received_rows=received,
        payload=payload, scale_cols=scale_cols, scale_pairs=scale_pairs,
        route=route)
    cache_key = (binding.call_id, received, hidden, local, payload,
                 route.basis, route.performance_control, route.source,
                 route.unresolved, route.layout_digest)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = plan["logical"]["shapes"]
    actual = {"dtype": payload, "layout": "ND", "shapes": actual_shapes,
              "attrs": {**attrs, "router_received_rows": received,
                        "local_experts": local,
                        "route_basis": route.basis,
                        "expert_token_counts": (
                            list(route.counts)
                            if route.counts is not None else None),
                        "expert_token_offsets": (
                            list(route.offsets)
                            if route.offsets is not None else None)}}
    validated = validate(binding, actual)
    partial_reasons = [
        "runtime_inputs_not_executed",
        "torch_npu_op_plugin_bridge_not_retained",
        "kernel_binary_selection_unresolved",
        "active_num_from_omitted_caller_default",
        "row_idx_type_from_consumer_semantics",
        "expert_tokens_num_type_from_count_consumer",
    ]
    unresolved = [
        "torch_npu_op_plugin_bridge_not_retained",
        "kernel_binary_selection_unresolved",
        "runtime_inputs_not_executed",
        "active_num_caller_default_unobserved",
        "row_idx_type_caller_omitted",
        "mxfp4_scale_layout_from_quantizer",
    ]
    if received is None:
        partial_reasons.append("router_received_rows_unresolved")
        unresolved.append("router_received_rows_unresolved")
    if route.counts is None:
        partial_reasons.append("expert_token_counts_unresolved")
    if local is None:
        partial_reasons.append("local_expert_range_unresolved")
        unresolved.append("local_expert_range_unresolved")
    for reason in route.unresolved:
        if reason not in unresolved:
            unresolved.append(reason)
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    prepare = {
        "schema": "mindie-sim/moe-init-routing-v3-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": list(CALL_LINES),
        "source_method_lines": list(SOURCE_LINES),
        "entry_source": dict(ENTRY_SOURCE),
        "entry_source_lines": list(ENTRY_LINES),
        "scale_producer": dict(SCALE_SOURCE),
        "scale_producer_lines": list(SCALE_LINES),
        "scale_layout_source": dict(SCALE_LAYOUT_SOURCE),
        "scale_layout_lines": list(SCALE_LAYOUT_LINES),
        "kernel": kernel,
        "kernel_source": dict(KERNEL_SOURCE),
        "kernel_body_source": dict(KERNEL_BODY_SOURCE),
        "tiling_source": dict(TILING_SOURCE),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": plan["arguments"],
        "passed_arguments": plan["passed_arguments"],
        "omitted_defaults": plan["omitted_defaults"],
        "argument_order": [
            "x", "expert_idx", "scale", "expert_num", "active_num",
            "expert_capacity", "drop_pad_mode", "expert_tokens_num_type",
            "expert_tokens_num_flag", "quant_mode", "active_expert_range",
            "row_idx_type", "offset",
        ],
        "shapes": actual_shapes,
        "descriptors": plan["descriptors"],
        "arrays": plan["descriptors"],
        "logical": plan["logical"],
        "materialization": plan["materialization"],
        "optional_slots": {
            "scale": "present",
            "offset": "not_passed",
            "expert_tokens_count_or_cumsum": "present",
            "expanded_scale": "present",
        },
        "default_basis": {
            "active_num": ("caller omits; V3 dropless computes min(L,activeNum) "
                           "and -1 means L; observed output rows equal R"),
            "row_idx_type": ("caller omits; expanded_row_idx is consumed as "
                             "npu_moe_token_unpermute sorted_indices (GATHER "
                             "inverse mapping, rowIdxType=0)"),
            "expert_tokens_num_type": ("caller omits; consumer sets "
                                       "group_list_type=1 count and observed "
                                       "output is INT64[E_local]"),
            "drop_pad_mode": "caller omits; observed output R equals input R",
        },
        "provenance": {
            "received_rows": "per-layer router_received_rows ledger",
            "expert_num": "compiled model/layer num_experts (E)",
            "local_experts": "moe_comm.experts_per_device",
            "payload": "source-selected routed_expert activation",
            "scale": "quantizer E8M0 output normalized to [R,ceil(H/64),2]",
        },
        "match": {
            "native_type": OP_TYPE,
            "op_type": OP_TYPE,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "call_id": binding.call_id,
            "received_rows": received,
            "hidden": hidden,
            "native_top_k": NATIVE_TOP_K,
            "model_top_k": model_top_k,
            "expert_num": experts,
            "local_experts": local,
            "payload": payload,
            "payload_bytes": _payload_bytes(payload),
            "x_dtype_runtime": "torch_npu.float4_e2m1fn_x2",
            "scale_format": SCALE_FORMAT,
            "scale_pairs": scale_pairs,
            "quant_mode": -1,
            "expert_tokens_num_type": 1,
            "row_idx_type": 0,
            "source_sha256": source.get("sha256"),
            "source_lines": list(CALL_LINES),
            "entry_source_sha256": R64_DEVICE_OP_SHA,
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "source_commit": R64_SOURCE_COMMIT,
            "observed_pf_payload_format": PAYLOAD,
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "revision": REVISION,
            "selection": ("exact R64 image digest + caller blob whose "
                          "device_op.py is the R64 2a251259 blob"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": R64_SOURCE_COMMIT,
            "entry_source_path": ENTRY_SOURCE["path"],
            "entry_source_symbol": ENTRY_SOURCE["symbol"],
            "entry_source_sha256": R64_DEVICE_OP_SHA,
            "scale_producer_path": SCALE_SOURCE["path"],
            "scale_producer_sha256": SCALE_SOURCE["sha256"],
            "kernel_source_commit": KERNEL_SOURCE["revision"],
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "installed_acl_nn_kernel_selection_not_retained",
            "owner_kernel": OP_TYPE,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "kernel_source_status": "source_retained",
        },
        "route": route.to_dict(),
        "expert_token_counts": (list(route.counts)
                                if route.counts is not None else None),
        "expert_token_offsets": (list(route.offsets)
                                 if route.offsets is not None else None),
        "expanded_row_idx": (list(route.inverse)
                             if route.inverse is not None else None),
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
        "issued_work": (plan["work"]["resources"]
                        if plan["work"] is not None else None),
        "program": plan["program"],
        "quant_metadata": {
            "payload_format": payload,
            "payload_storage_bits": 4,
            "payload_storage": "packed_2_per_byte",
            "payload_row_bytes": plan["rounding"]["payload_row_bytes"],
            "payload_row_padding_bits": plan["rounding"]["payload_row_padding_bits"],
            "scale_format": SCALE_FORMAT,
            "scale_logical_cols": plan["rounding"]["scale_logical_cols"],
            "scale_pairs": plan["rounding"]["scale_pairs"],
            "scale_storage_values": scale_pairs * 2,
            "scale_padding_values": plan["rounding"]["scale_padding_values"],
            "scale_row_bytes": scale_pairs * 2,
            "scale_observed_layout": "NCL",
            "index_dtype": "int32",
            "count_dtype": "int64",
            "source": "selected R64 MXFP4 activation and E8M0 scale pair",
        },
        "complete": False,
        "partial": True,
        "partial_reasons": partial_reasons,
        "unresolved": unresolved,
        "charge": "one_per_native_call",
        "attribution": "none",
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_complete_native_execution",
            "no_fp8_w8a8mx_dispatch_claim",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": payload, "layout": "ND",
                         "shapes": actual_shapes, "attrs": actual["attrs"]},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
