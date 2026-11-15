"""Prepare-only R31 A3 GDN BF16 dense MatMulV3 invocation.

[Role] Expand the selected R31 A3 GDN dense BF16 projection
  (``ba_projection``/``o_projection``) from the same-image
  ``AscendGatedDeltaNetAttention.forward`` caller into the actual
  ``aclnnMatmul``/``MatMulV3`` operand and output descriptors for the
  evaluated token geometry.
[Invariants] Only the exact R31 image digest ``9e869e814846...`` and the
  unquantized BF16 dense source projection roles are admitted.  K/N come from
  the compiled operator's runtime matrix (``theta.matrix``), M from the
  evaluated geometry, H from the role's source projection axis, and the
  physical weight/mat2 storage ``[N,K]`` from the source linear projection
  orientation.  The original sample shapes are not selectors.  W8/INT8/
  quantized/QMM projections return ``None``.  The Python linear wrapper, the
  ACLNN C header/transpose attribute and the MatMulV3 kernel source/core
  selection are not retained and remain explicit ``unresolved``; this adapter
  never claims a complete native execution.  ``aclnnMm`` and
  ``aclnnQuantMatmul*`` are separate APIs and are not aliased.  No tensor is
  allocated, no pointer or symbol is fabricated, no default attribute is
  supplied and no prediction is repriced.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing selection and
  retained same-image caller/method records.
[Independent ABI] The independent-ABI mapper below also accepts a generic CANN-9.2.0
  record without an image; the model-bound spec still requires R31.
"""
from __future__ import annotations

from typing import Any, Mapping

from .native_matrix_abi import (
    ABI_SCHEMA, align, digest, interpretation_reasons,
    result as abi_result, selected as abi_selected, status as abi_status,
    tensor as abi_tensor, slot_view_reasons, with_views)

from .native_source_pins import (  # noqa: E402
    CANN_VERSION as _CANN_VERSION, R31_IMAGE_SHA, R31_IMAGE_PREFIX,
    declared_stack, declared_version, optional_version_gate,
)
R31_REVISION = "r31/9e869e81484"
CANN_VERSION = _CANN_VERSION
SOURCE_PATH = "vllm_ascend/ops/gdn.py"
SOURCE_SHA = "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed"
SOURCE_SYMBOL = "AscendGatedDeltaNetAttention.forward"
ACL_API = "aclnnMatmul"
HOST_OP = "MatMulV3Common"
NATIVE_TYPE = "MatMulV3"
PROFILER_NAME = "aclnnMatmul_MatMulV3Common_MatMulV3"
WORKSPACE_SYMBOL = "aclnnMatmulGetWorkspaceSize"
API_BINARY_PATH = ("usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/"
                   "libopapi.so")
API_BINARY_SHA = ("eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03"
                  "f78c4a5313c456")
CANN_VERSION = "cann-9.2.0-r31"
SCHEMA = "mindie-sim/matmul-bf16-native-prepare/1"
_DYNAMIC_CACHE: dict[tuple, dict] = {}

#: Source-role semantics, not shape selectors.  K/N/H/M are derived at
#: runtime from ``theta.matrix``/``theta.parallel`` and the evaluated geometry.
_SOURCE_ROLES = {
    "ba_projection": {
        "role": "gdn.beta_alpha", "theta_role": "ba",
        "caller": "self.in_proj_ba(hidden_states)",
        "source_lines": [398, 405],
        "source_call_sites": [384, 398, 405],
        "definition_unresolved": "in_proj_ba_definition_not_retained",
        "hidden_basis": "input_width_K",
    },
    "o_projection": {
        "role": "gdn.o", "theta_role": "output",
        "caller": "self.out_proj(core_attn_out)",
        "source_lines": [447],
        "source_call_sites": [447],
        "definition_unresolved": "out_proj_definition_not_retained",
        "hidden_basis": "output_width_N",
    },
}
_UNRESOLVED_STATIC = (
    "linear_method_wrapper_not_retained",
    "aclnn_matmul_c_abi_header_not_retained",
    "transpose_x2_attribute_not_captured",
    "matmul_v3_kernel_source_not_retained",
    "matmul_v3_binary_selection_unresolved",
)


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _r31_metadata(compiled: Any) -> str | None:
    """Only the exact retained R31 image digest admits this source adapter."""
    from mindie_sim.prediction.mx_source_provenance import image_digest
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    digest = image_digest(image_id)
    return digest if digest == R31_IMAGE_SHA else None


def _static_terms(compiled: Any) -> dict[str, Any] | None:
    """The real R31 dense BF16 projection method at runtime K/N, or None.

    K/N are read from the compiled operator's actual matrix, not from a sample
    table.  W8/INT8/quantized representations fail the method checks and stay
    on their existing analytical/other-source path.
    """
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    name = str(getattr(operator, "name", "") or "")
    spec = _SOURCE_ROLES.get(name)
    if spec is None or str(getattr(operator, "family", "") or "") != "gemm":
        return None
    theta = operator.theta
    if str(getattr(theta, "role", "") or "") != spec["theta_role"]:
        return None
    matrix = dict(getattr(theta, "matrix", {}) or ())
    quant = dict(getattr(theta, "quant", {}) or ())
    parallel = dict(getattr(theta, "parallel", {}) or ())
    if str(quant.get("category", "") or "") != spec["role"]:
        return None
    required = ("weight_format", "act_format", "quantized",
                "scale_group", "linear_method", "weight_payload",
                "weight_scale", "weight_zero", "weight_total")
    if any(key not in quant for key in required):
        return None
    # Dense BF16 Linear returns the activation dtype; the generic quant
    # record need not repeat ``output_format``.  An explicit different value
    # is still rejected rather than overwritten.
    output_format = str(quant.get("output_format")
                        or quant.get("act_format") or "").lower()
    if any(str(quant[key]).lower() != "bf16" for key in
           ("weight_format", "act_format")) or output_format != "bf16":
        return None
    if (_int(quant["quantized"]) != 0
            or _int(quant["scale_group"]) != 0
            or str(quant["linear_method"]).lower() != "dense"):
        return None
    k = _int(matrix.get("K"))
    n = _int(matrix.get("N"))
    if k < 1 or n < 1:
        return None
    payload = _int(quant["weight_payload"])
    if payload != n * k * 2:
        return None
    try:
        scale = float(quant["weight_scale"])
        zero = float(quant["weight_zero"])
        total = float(quant["weight_total"])
    except (TypeError, ValueError):
        return None
    if scale != 0.0 or zero != 0.0 or total != float(payload):
        return None
    hidden_basis = spec["hidden_basis"]
    hidden = k if hidden_basis.startswith("input_") else n
    tp_value = parallel.get("tp")
    runtime_tp = _int(tp_value) if tp_value is not None else None
    return {"name": name, "spec": spec, "k": k, "n": n, "hidden": hidden,
            "hidden_basis": hidden_basis, "runtime_tp": runtime_tp}


def _typed(digest: str, terms: dict) -> dict[str, Any]:
    name = terms["name"]
    spec = terms["spec"]
    k, n = terms["k"], terms["n"]
    attrs: dict[str, Any] = {
        "matrix_k": k, "matrix_n": n,
        "hidden": terms["hidden"], "hidden_basis": terms["hidden_basis"],
        "k_basis": "theta.matrix.K", "n_basis": "theta.matrix.N",
        "runtime_tp": terms["runtime_tp"],
        "act_format": "bf16", "weight_format": "bf16",
        "output_format": "bf16",
        "quantized": 0, "scale_group": 0, "linear_method": "dense",
        "weight_payload": n * k * 2,
        "weight_scale": 0.0, "weight_zero": 0.0,
        "bias": False, "bias_operand_present": False,
        "projection": name, "projection_role": spec["role"],
        "caller": spec["caller"],
        "caller_bias_slot": "tuple_return_second_item",
        "caller_definition_status": "not_retained",
        "source_revision": "r31", "source_version": R31_REVISION,
        "deployment_image_id": digest,
        "source_symbol": SOURCE_SYMBOL, "source_sha256": SOURCE_SHA,
        "source_lines": list(spec["source_lines"]),
        "source_call_sites": list(spec["source_call_sites"]),
        "acl_api": ACL_API,
        "workspace_size_symbol": WORKSPACE_SYMBOL,
        "api_namespace": "aclnn",
        "host_op": HOST_OP, "native_type": NATIVE_TYPE,
        "profiler_name": PROFILER_NAME,
        "execution_core_status": "not_retained",
        "api_binary_path": API_BINARY_PATH,
        "api_binary_sha256": API_BINARY_SHA,
        "weight_storage_shape": [n, k],
        "weight_logical_shape": [k, n],
        "weight_orientation": "out_by_in",
        "transpose_basis": "source_projection_weight_out_by_in",
        "weight_layout": "ND", "input_layout": "ND", "output_layout": "ND",
        "contraction_axis": 1,
        "transpose_x2": None,
        "transpose_x2_status": (
            "source projection weight is [out,in]; exact aclnn attribute is "
            "not retained"),
        "kernel_source_status": "kernel_source_not_retained",
        "binary_selection_status": "uncertain",
        "phase": "prefill",
        "api_disambiguation": {
            "selected_api": ACL_API,
            "selected_host_op": HOST_OP,
            "selected_native_type": NATIVE_TYPE,
            "not_selected": ["aclnnMm", "aclnnQuantMatmul",
                             "aclnnQuantMatmulWeightNz"],
            "basis": (
                "actual R31 BF16 MatMulV3 API/kernel evidence with "
                "BF16/BF16 ND operands; the R31 libopapi exports "
                "aclnnMatmul, aclnnMm and aclnnQuantMatmul as separate "
                "symbols"),
        },
    }
    return {
        "dtype": "bf16",
        "layout": "ND",
        "shapes": {"weight": [k, n]},
        "attrs": {key: attrs[key] for key in sorted(attrs)},
        "occurrence_attrs": {"m": "positive_int"},
        "rank_width": 2,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R31 A3 GDN dense BF16 MatMulV3 call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Decode, another image, another source
    role, a quantized/W8 representation or a malformed runtime matrix returns
    ``None`` so ordinary prediction is unchanged.  Runtime K/N are not
    validated against the original sample; the sample shapes are scoped
    verification only.  The returned spec is a real
    ``native_calls._CallSpec``; it is deliberately partial where the Python
    wrapper/C ABI/transpose attribute/kernel source is not retained and
    carries those gaps in ``unresolved``.
    """
    from . import native_calls
    if str(getattr(compiled, "kind", "") or "") != "gdn_linear_attn":
        return None
    if str(phase or "") != "prefill":
        return None
    terms = _static_terms(compiled)
    if terms is None:
        return None
    digest = _r31_metadata(compiled)
    if digest is None:
        return None
    source = native_calls._source(
        "vllm-ascend", SOURCE_PATH, sha256=SOURCE_SHA,
        symbol=SOURCE_SYMBOL, revision=R31_REVISION,
        state="task_artifact")
    kernel = native_calls._kernel(
        NATIVE_TYPE, "cube", evidence="measured",
        acl_api=ACL_API, host_op=HOST_OP,
        profiler_name=PROFILER_NAME,
        workspace_size_symbol=WORKSPACE_SYMBOL,
        api_binary_path=API_BINARY_PATH,
        api_binary_sha256=API_BINARY_SHA,
        source_status="kernel_source_not_retained")
    unresolved = [terms["spec"]["definition_unresolved"],
                  *_UNRESOLVED_STATIC]
    if terms["name"] == "ba_projection":
        unresolved.append("caller_branch_not_selected")
    return native_calls._call_spec(
        native_calls._api("aclnn", ACL_API, aclnn_symbol=ACL_API,
                          version=CANN_VERSION),
        source, _typed(digest, terms), (kernel,), (),
        str(getattr(compiled.operator, "name", "") or ""),
        unresolved=unresolved)


def _descriptor(shape: list[int], dtype: str, layout: str, role: str,
                **extra: Any) -> dict:
    row = {"shape": [int(item) for item in shape], "dtype": dtype,
           "layout": layout, "role": role, "pointer": None,
           "allocation": "external"}
    row.update(extra)
    return row


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected A3 dense BF16 MatMulV3.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  M is the evaluated effective-token axis;
    K/N/H and the physical ``[N,K]`` weight storage are projected from the
    compiled runtime matrix/role, not from the original sample.  The
    projection itself is the one physical owner of the native call; the
    caller's split/copy work is not charged here.  ``prepare.complete`` stays
    false because the exact Python linear wrapper, ACLNN C ABI/transpose
    attribute, execution core and MatMulV3 kernel source are not retained.
    """
    api = getattr(binding, "api", None) or {}
    if api.get("entry") != ACL_API:
        return None
    typed = getattr(binding, "typed", None) or {}
    attrs = dict(typed.get("attrs") or {})
    if (attrs.get("acl_api") != ACL_API
            or attrs.get("native_type") != NATIVE_TYPE
            or attrs.get("source_revision") != "r31"
            or attrs.get("host_op") != HOST_OP):
        return None
    if str(getattr(compiled, "kind", "") or "") != "gdn_linear_attn":
        return None
    terms = _static_terms(compiled)
    digest = _r31_metadata(compiled)
    if terms is None or digest is None:
        return None
    name = terms["name"]
    k, n = terms["k"], terms["n"]
    if (attrs.get("projection") != name
            or attrs.get("deployment_image_id") != digest
            or _int(attrs.get("matrix_k")) != k
            or _int(attrs.get("matrix_n")) != n):
        return None
    source = getattr(binding, "source", None) or {}
    if (source.get("path") != SOURCE_PATH
            or source.get("sha256") != SOURCE_SHA):
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    m = _int(getattr(geometry, "effective_tokens", 0))
    if m < 1:
        return None
    cache_key = (str(getattr(binding, "call_id", "") or ""), name, m)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "input": [m, k], "weight": [k, n], "output": [m, n],
        "weight_storage": [n, k], "mat2": [n, k],
    }
    actual_attrs = {
        **attrs, "m": m, "m_basis": "effective_tokens", "phase": "prefill",
        "input_shape": [m, k], "output_shape": [m, n],
        "weight_storage_shape": [n, k], "mat2_shape": [n, k],
        "valid_tokens": _int(getattr(geometry, "tokens", 0)),
        "selected_tokens": _int(getattr(geometry, "selected_tokens", 0)),
        "physical_tokens": _int(getattr(geometry, "physical_tokens", 0)),
    }
    actual = {"dtype": "bf16", "layout": "ND", "shapes": actual_shapes,
              "attrs": actual_attrs}
    validated = validate(binding, actual)
    weight = _descriptor(
        [n, k], "bf16", "ND", "input",
        aclnn_role="mat2", logical_shape=[k, n], declared_shape=[k, n],
        storage_shape=[n, k], matrix_shape=[k, n],
        device_shape=None, device_layout="ND",
        weight_orientation="out_by_in",
        transpose_basis="source_projection_weight_out_by_in",
        transpose_x2=None,
        transpose_x2_status=(
            "source projection weight is [out,in]; exact aclnn attribute is "
            "not retained"))
    descriptors = {
        "x": _descriptor([m, k], "bf16", "ND", "input",
                         aclnn_role="self"),
        "weight": weight,
        "output": _descriptor([m, n], "bf16", "ND", "output",
                              aclnn_role="out"),
        "bias": {"shape": None, "dtype": None, "layout": None,
                 "role": "input_optional", "aclnn_role": "bias",
                 "present": False, "pointer": None,
                 "allocation": "external"},
    }
    arguments = {"self": "x", "mat2": "weight", "bias": None, "out": "output"}
    argument_order = ["self", "mat2", "bias", "out"]
    spec = terms["spec"]
    kernel = (dict(binding.kernels[0]) if getattr(binding, "kernels", ())
              else {"name": NATIVE_TYPE, "role": "cube",
                    "evidence": "measured"})
    kernel.update({
        "acl_api": ACL_API, "host_op": HOST_OP,
        "profiler_name": PROFILER_NAME,
        "execution_core_status": "not_retained",
        "workspace_size_symbol": WORKSPACE_SYMBOL,
        "api_binary_path": API_BINARY_PATH,
        "api_binary_sha256": API_BINARY_SHA,
        "source_status": "kernel_source_not_retained",
    })
    partial_reasons = [
        spec["definition_unresolved"],
        *_UNRESOLVED_STATIC,
    ]
    if name == "ba_projection":
        partial_reasons.append("caller_branch_not_selected")
    prepare = {
        "schema": SCHEMA,
        "call": ACL_API,
        "call_id": getattr(binding, "call_id", None),
        "signature_hash": typed.get("signature_hash"),
        "api": dict(api),
        "source": dict(source),
        "source_revision": R31_REVISION,
        "source_lines": list(spec["source_lines"]),
        "source_call_sites": list(spec["source_call_sites"]),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": argument_order,
        "argument_order_basis": (
            "captured operand roles (self, mat2, out); the ACLNN C header "
            "and optional transpose_x1/transpose_x2/cubeMathType slots are "
            "not retained and are not fabricated"),
        "optional_slots": {
            "bias": "not_present",
            "transpose_x1": "not_captured",
            "transpose_x2": "not_captured",
            "cube_math_type": "not_captured",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "captured_call": {
            "name": PROFILER_NAME,
            "native_type": NATIVE_TYPE,
            "host_op": HOST_OP,
            "acl_api": ACL_API,
            "execution_core_status": "not_retained",
            "op_state": "dynamic",
            "operands": {"self": [m, k], "mat2": [n, k], "out": [m, n]},
            "input_dtypes": ["bf16", "bf16"],
            "output_dtype": "bf16",
            "input_layouts": ["ND", "ND"],
            "output_layout": "ND",
            "num_input_tensors": 2,
            "bias_operand_present": False,
        },
        "deployment": {
            "image_id": digest,
            "revision": R31_REVISION,
            "selection": (
                "exact R31 image digest + A3 GDN source role + BF16 dense/"
                "unquantized semantics + runtime theta K/N"),
            "source_path": SOURCE_PATH,
            "source_symbol": SOURCE_SYMBOL,
            "source_sha256": SOURCE_SHA,
            "source_lines": list(spec["source_lines"]),
            "api_binary_path": API_BINARY_PATH,
            "api_binary_sha256": API_BINARY_SHA,
            "acl_api": ACL_API, "host_op": HOST_OP,
            "execution_core_status": "not_retained",
            "kernel": NATIVE_TYPE,
        },
        "match": {
            "native_type": NATIVE_TYPE,
            "entry": ACL_API,
            "host_op": HOST_OP,
            "profiler_name": PROFILER_NAME,
            "call_id": getattr(binding, "call_id", None),
            "deployment_image_id": digest,
            "source_revision": R31_REVISION,
            "projection": name,
            "m": m, "k": k, "n": n, "h": terms["hidden"],
            "m_basis": "effective_tokens",
            "k_basis": "theta.matrix.K", "n_basis": "theta.matrix.N",
            "h_basis": terms["hidden_basis"],
            "runtime_tp": terms["runtime_tp"],
            "weight_storage_shape": [n, k],
            "weight_logical_shape": [k, n],
            "weight_orientation": "out_by_in",
            "transpose_basis": "source_projection_weight_out_by_in",
            "transpose_x2": None,
            "transpose_x2_status": (
                "source projection weight is [out,in]; exact aclnn attribute "
                "is not retained"),
            "execution_core_status": "not_retained",
            "input_dtype": "bf16", "weight_dtype": "bf16",
            "output_dtype": "bf16",
            "layout": "ND", "bias": False,
            "source_sha256": SOURCE_SHA,
            "source_lines": list(spec["source_lines"]),
            "kernel_source_status": "kernel_source_not_retained",
            "api_binary_sha256": API_BINARY_SHA,
            "complete": False,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "matmul_v3_kernel_source_and_binary_not_retained",
            "owner_kernel": NATIVE_TYPE,
            "acl_api": ACL_API,
            "host_op": HOST_OP,
            "execution_core_status": "not_retained",
            "api_binary": {
                "path": API_BINARY_PATH,
                "sha256": API_BINARY_SHA,
                "exported_symbols": [ACL_API, WORKSPACE_SYMBOL],
                "observed_name_tokens": [HOST_OP, NATIVE_TYPE],
            },
        },
        "complete": False,
        "partial": True,
        "partial_reasons": list(dict.fromkeys(partial_reasons)),
        "unresolved": list(dict.fromkeys(
            list(getattr(binding, "unresolved", ()) or ())
            + partial_reasons)),
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_complete_native_execution",
            "no_exact_python_linear_wrapper", "no_exact_aclnn_c_abi",
            "no_execution_core_claim", "no_transpose_default",
            "no_sample_geometry_selector",
            "no_kernel_source_claim", "no_mm_or_qmm_alias",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": validated.get("dtype"),
            "layout": validated.get("layout"),
            "shapes": validated.get("shapes"),
            "attrs": validated.get("attrs"),
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Independent ABI leaf: aclnnMatmul (MatMulV3 BF16 dense)
# ---------------------------------------------------------------------------

ABI_ENTRY = ACL_API
ABI_NAMESPACE = "aclnn"
ABI_ATTRS = ()
ABI_OPTIONAL_ATTRS = ("transpose_x1", "transpose_x2",
                      "cube_math_type", "bias")
ABI_UNRESOLVED = tuple(_UNRESOLVED_STATIC)


def abi_contract() -> dict[str, Any]:
    """The independent tensor/attr/version contract for this ABI leaf."""
    return {
        "schema": ABI_SCHEMA,
        "entry": ABI_ENTRY,
        "namespace": ABI_NAMESPACE,
        "aclnn_symbol": ACL_API,
        "version": {"api_prefix": "cann-9.2.0", "source": ["r31"],
                    "image_prefix": R31_IMAGE_SHA[:11]},
        "tensors": [
            {"name": "x1", "role": "input", "dtype": "bf16",
             "rank": 2, "required": True},
            {"name": "x2", "role": "input", "dtype": "bf16",
             "rank": 2, "required": True},
            {"name": "output", "role": "output", "dtype": "bf16",
             "rank": 2, "required": True},
            {"name": "bias", "role": "input", "dtype": "bf16",
             "rank": 1, "required": False},
        ],
        "attrs": ABI_ATTRS,
        "optional_attrs": ABI_OPTIONAL_ATTRS,
        "kernels": [{"name": NATIVE_TYPE, "role": "cube",
                     "host_op": HOST_OP, "profiler_name": PROFILER_NAME}],
        "relation": "one_to_one",
        "source": {"repository": "vllm-ascend", "path": SOURCE_PATH,
                   "sha256": SOURCE_SHA, "symbol": SOURCE_SYMBOL,
                   "revision": R31_REVISION,
                   "api_binary_path": API_BINARY_PATH,
                   "api_binary_sha256": API_BINARY_SHA},
    }


def _abi_version(version: Any) -> tuple[str | None, dict, list[str]]:
    """Declared ABI version, with an *optional* software-stack selector.

    An absent api/source/image is a general shape: the call is admitted with
    an unresolved container identity instead of being rejected by an image
    wall.  An explicit declaration of another CANN release or another
    captured stack (r35/r64) is still a declared mismatch.
    """
    values = declared_version(version)
    api = values.get("api") or values.get("api_version")
    reasons = optional_version_gate(values, stack="r31")
    if reasons:
        return None, values, reasons
    return "r31", {"api": api or CANN_VERSION,
                   "source": values.get("source") or "r31",
                   "image_id": ("sha256:" + R31_IMAGE_SHA
                                if declared_stack(values) == "r31"
                                else values.get("image_id")
                                or "unresolved")}, []


def _abi_orientation(x1: Mapping[str, Any], x2: Mapping[str, Any],
                     output: Mapping[str, Any], attrs: Mapping[str, Any]
                     ) -> tuple[int | None, int | None, int | None,
                                bool | None, bool | None, list[str]]:
    m, n = output["shape"]
    first, second = x1["shape"], x2["shape"]
    declared_x1 = attrs.get("transpose_x1")
    declared_x2 = attrs.get("transpose_x2")
    if declared_x1 is None:
        x1_m = first[0] == m
        x1_k = first[1] == m
        if x1_m == x1_k:
            return None, None, None, None, None, [
                "missing:attrs.transpose_x1"]
        transpose_x1 = x1_k
    else:
        transpose_x1 = bool(declared_x1)
        x1_m = first[1] == m if transpose_x1 else first[0] == m
        if not x1_m:
            return None, None, None, None, None, [
                "mismatch:tensors.x1.shape"]
    k = first[0] if transpose_x1 else first[1]
    if declared_x2 is None:
        x2_k = second[0] == k
        x2_n = second[1] == k
        if x2_k == x2_n:
            return None, None, None, None, None, [
                "missing:attrs.transpose_x2"]
        transpose_x2 = x2_n
    else:
        transpose_x2 = bool(declared_x2)
    if transpose_x2:
        if second[1] != k or second[0] != n:
            return None, None, None, None, None, [
                "mismatch:tensors.x2.shape"]
    elif second[0] != k or second[1] != n:
        return None, None, None, None, None, ["mismatch:tensors.x2.shape"]
    return m, n, k, transpose_x1, transpose_x2, []


def predict_abi(tensors: Mapping[str, Any], *,
                attrs: Mapping[str, Any] | None = None,
                version: Any = None,
                conditions: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
    """Map one independent BF16 MatMulV3 ABI to geometry/work/call.

    The contraction orientation comes from the real tensor shapes and the
    declared transpose attributes; a square/ambiguous operand without an
    attribute is an explicit gap.  No sample shape or target duration is read.
    """
    attrs = dict(attrs or {})
    conditions = dict(conditions or {})
    reasons: list[str] = []
    revision, canonical_version, version_gaps = _abi_version(version)
    reasons.extend(version_gaps)
    unknown_attrs = set(attrs) - set(ABI_ATTRS) - set(ABI_OPTIONAL_ATTRS)
    if unknown_attrs:
        reasons.append("unmodeled:attrs:" + ",".join(sorted(unknown_attrs)))
    if attrs.get("cube_math_type") is not None:
        reasons.append("unmodeled:attrs.cube_math_type")
    if attrs.get("bias") is not None:
        reasons.append("unmodeled:attrs.bias")
    x1, reason = abi_tensor(tensors, "x1", role="input", dtype="bf16",
                            rank=2, dtypes=("bf16", "bfloat16"))
    reasons.extend(reason)
    x2, reason = abi_tensor(tensors, "x2", role="input", dtype="bf16",
                            rank=2, dtypes=("bf16", "bfloat16"))
    reasons.extend(reason)
    output, reason = abi_tensor(tensors, "output", role="output",
                                dtype="bf16", rank=2,
                                dtypes=("bf16", "bfloat16"))
    reasons.extend(reason)
    raw_bias = (tensors or {}).get("bias")
    if (isinstance(raw_bias, Mapping)
            and raw_bias.get("present", True) is not False):
        reasons.append("unmodeled:tensors.bias")
    for name, row in (("x1", x1), ("x2", x2), ("output", output)):
        if row is not None:
            reasons.extend(interpretation_reasons(
                row, name, ("bf16", "bfloat16")))
    m = n = k = None
    transpose_x1 = transpose_x2 = None
    if x1 is not None and x2 is not None and output is not None:
        m, n, k, transpose_x1, transpose_x2, orient_gaps = _abi_orientation(
            x1, x2, output, attrs)
        reasons.extend(orient_gaps)
        if m is not None and n is not None and k is not None:
            reasons.extend(slot_view_reasons(
                x1, "x1",
                logical={"shape": [m, k], "dtype": "bf16", "layout": "ND"},
                storage={"shape": ([k, m] if transpose_x1 else [m, k]),
                         "dtype": "bf16", "layout": "ND"},
                device={"shape": ([k, m] if transpose_x1 else [m, k]),
                        "dtype": "bf16", "layout": "ND"}))
            reasons.extend(slot_view_reasons(
                x2, "x2",
                storage={"shape": ([n, k] if transpose_x2 else [k, n]),
                         "dtype": "bf16", "layout": "ND"},
                device={"shape": ([n, k] if transpose_x2 else [k, n]),
                        "dtype": "bf16", "layout": "ND"}))
            reasons.extend(slot_view_reasons(
                output, "output",
                logical={"shape": [m, n], "dtype": "bf16", "layout": "ND"},
                storage={"shape": [m, n], "dtype": "bf16", "layout": "ND"},
                device={"shape": [m, n], "dtype": "bf16", "layout": "ND"}))
    current = abi_status(reasons)
    invocation = {"tensors": dict(tensors), "attrs": attrs,
                  "version": canonical_version, "conditions": conditions}
    if current != "predicted":
        return abi_result(
            status_value=current, entry=ABI_ENTRY, reasons=reasons,
            invocation=invocation, selected_call=None, geometry=None,
            work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    macs = m * n * k
    issued = align(m, 16) * align(n, 16) * align(k, 16)
    x1_bytes = m * k * 2
    x2_bytes = n * k * 2
    output_bytes = m * n * 2
    total_bytes = x1_bytes + x2_bytes + output_bytes
    x1 = with_views(x1, storage_shape=list(x1["shape"]),
                    storage_dtype="bf16", storage_layout="ND",
                    interpret_dtype="bf16")
    x1["views"]["logical"] = {"shape": [m, k], "dtype": "bf16",
                              "layout": "ND"}
    x2 = with_views(x2, storage_shape=[n, k] if transpose_x2 else [k, n],
                    storage_dtype="bf16", storage_layout="ND",
                    interpret_dtype="bf16")
    x2["views"]["logical"] = {"shape": [k, n], "dtype": "bf16",
                              "layout": "ND"}
    output = with_views(output, storage_shape=[m, n],
                        storage_dtype="bf16", storage_layout="ND",
                        interpret_dtype="bf16")
    geometry = {
        "m": m, "n": n, "k": k,
        "padded_m": align(m, 16), "padded_n": align(n, 16),
        "padded_k": align(k, 16),
        "transpose_x1": bool(transpose_x1),
        "transpose_x2": bool(transpose_x2),
        "weight_storage_shape": [n, k] if transpose_x2 else [k, n],
        "basis": "declared_tensor_shapes_and_transpose_attributes",
    }
    work = {
        "cube": {"macs": macs, "flops": 2 * macs, "issued_macs": issued,
                 "issued_basis": "fractal_16_alignment"},
        "vector": {"ops": 0, "basis": "no_separate_vector_epilogue"},
        "bytes": {"input": x1_bytes, "weight": x2_bytes,
                  "output": output_bytes, "total": total_bytes,
                  "basis": "call_boundary_tensor_footprint"},
        "call_boundary": {
            "tensor_footprint_bytes": total_bytes,
            "issued_traffic": None,
            "issued_traffic_status": "requires_selected_load_store_plan",
        },
    }
    members = tuple(conditions.get("members") or ())
    relation = "many_to_one_fused" if len(members) > 1 else "one_to_one"
    call = abi_selected(
        entry=ABI_ENTRY, namespace=ABI_NAMESPACE, aclnn_symbol=ACL_API,
        api_version=canonical_version["api"], source_revision="r31",
        image_id=canonical_version["image_id"],
        source={"repository": "vllm-ascend", "path": SOURCE_PATH,
                "sha256": SOURCE_SHA, "symbol": SOURCE_SYMBOL,
                "revision": R31_REVISION,
                "api_binary_path": API_BINARY_PATH,
                "api_binary_sha256": API_BINARY_SHA},
        kernels=[{"name": NATIVE_TYPE, "role": "cube",
                  "host_op": HOST_OP, "profiler_name": PROFILER_NAME,
                  "acl_api": ACL_API,
                  "workspace_symbol": WORKSPACE_SYMBOL,
                  "evidence": "measured"}],
        execution_core=None, relation=relation,
        members=members,
        tensors=[x1, x2, output],
        attrs={"transpose_x1": bool(transpose_x1),
               "transpose_x2": bool(transpose_x2),
               "weight_storage_shape": ([n, k] if transpose_x2
                                        else [k, n]),
               "input_dtypes": ["bf16", "bf16", "bf16"],
               "input_slots": ["x1", "x2", "output"],
               "offset": "not_present"},
        regions=["projection"], variant="dense_bf16_matmul")
    call["unresolved"] = list(ABI_UNRESOLVED)
    call["api_binary"] = {"path": API_BINARY_PATH,
                          "sha256": API_BINARY_SHA}
    metadata = {
        "transpose_x1": bool(transpose_x1),
        "transpose_x2": bool(transpose_x2),
        "weight_storage_shape": [n, k] if transpose_x2 else [k, n],
        "weight_logical_shape": [k, n],
        "host_op": HOST_OP,
        "profiler_name": PROFILER_NAME,
        "api_binary_path": API_BINARY_PATH,
        "api_binary_sha256": API_BINARY_SHA,
        "execution_core_status": "not_retained",
        "input_dtypes": ["bf16", "bf16", "bf16"],
        "offset": "not_present",
    }
    accumulator = {
        "dtype": "fp32", "shape": [m, n], "elements": m * n,
        "bytes": m * n * 4, "abi_tensor": False, "on_chip": True,
        "basis": "bf16_matmul_internal_accumulator",
        "residency_status": "requires_selected_matrix_program",
        "source_status": "mechanism_inferred_not_source_declared",
    }
    regions = [{
        "id": "projection", "relation": relation, "rows": m,
        "cols": n, "inner": k, "macs": macs, "outputs": ["output"],
        "members": list(members),
    }]
    return abi_result(
        status_value="predicted", entry=ABI_ENTRY, reasons=[],
        invocation=invocation, selected_call=call, geometry=geometry,
        work=work,
        tensors={"x1": x1, "x2": x2, "output": output},
        scale={}, metadata=metadata, accumulator=accumulator,
        regions=regions)


def operator_plan(record, *, tile=(16, 32, 16), slots=2):
    """T12/T10 consumer entry for the selected real OperatorPlan."""
    from .native_matrix_abi import build_operator_plan
    return build_operator_plan(record, tile=tile, slots=slots)
