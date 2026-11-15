"""R31 W8A8-dynamic ``QuantMatmulWeightNz`` / ``QuantBatchMatmulV3`` adapter.

[Role] Bind the real R31 (CANN 9.2.0 r31) W8A8-dynamic linear projection to
  its deployed native call surface.  The source evidence is the R31
  ``libopapi.so`` export plus the original R31 Qwen3.6 PF4kC1 kernel row
  ``aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3``; the
  per-occurrence row geometry stays in the occurrence.
[Invariants] Only the exact R31 deployment identity plus the W8A8 INT8
  WeightNz family selects this adapter.  The observed source storage scale is
  FP32 (a declared BF16 storage is carried as-is), but the deployed ACLNN row
  consumes one BF16 scale per output channel and one FLOAT scale per
  activation row; those are distinct fields.  The weight
  device shape is the active 32/16 NZ tile shape, never a guessed padded
  logical N.  The R31 Python caller and kernel source are not archived, so
  the identity records them as unresolved instead of borrowing R64/R35.
[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device, lock,
  wait or runtime import.
[Side-Effect Policy] Returns immutable metadata only.  No tensor, pointer,
  allocation or native launch is created.
[Independent ABI] The independent-ABI mapper below also accepts a generic CANN-9.2.0
  record without an R31 image; the model-bound spec still requires R31.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.schema.subgraphs import SOFTMAX_ATTENTION_KINDS
from .native_matrix_abi import (
    ABI_SCHEMA, align, digest, int_value,
    direction_reasons, interpretation_reasons, preparation_record,
    provenance, slot_view_reasons,
    result as abi_result, row_column_scale_epilogue,
    selected as abi_selected, status as abi_status, tensor as abi_tensor,
    view_reasons, with_views)

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e8148"
R31_SOURCE_VERSION = "cann-9.2.0-r31"
R31_WRAPPER_VERSION = "vllm-ascend v0.30.0-a5-cann9.2.0-r31"
_ENTRY = "aclnnQuantMatmulWeightNz"
_NZ_N = 32
_NZ_K = 16
_MISSING_CALLER = (
    "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py")
_LIBOPAPI = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "revision": R31_SOURCE_VERSION,
    "state": "task_artifact",
}
_PACKER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/utils.py",
    "sha256": "69c0b69ef9b80c08b772d191e8e66e8857c5c1ea68e719ddc91b1a197f7ee8c3",
    "symbol": "maybe_trans_nz",
    "revision": R31_REVISION,
    "state": "task_artifact",
    "lines": [394, 406],
    "layout_constant": "ACL_FORMAT_FRACTAL_NZ=29",
}
def _digest_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _r31_image_text(compiled: Any) -> str | None:
    """Return the declared R31 deployment identity, never a model label."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    for key in ("image_id", "oci_image_digest", "deployment_image_id",
                "image", "deployment_image", "oci_image", "source_image"):
        text = _digest_text(meta.get(key))
        if text.startswith(R31_IMAGE_PREFIX):
            return text
    return None


def _r31_dims(compiled: Any) -> tuple[int, int] | None:
    operator = getattr(compiled, "operator", None)
    if operator is None or getattr(operator, "family", "") != "gemm":
        return None
    if getattr(compiled, "kind", "") not in (
            SOFTMAX_ATTENTION_KINDS | {"mlp", "gdn_linear_attn"}):
        return None
    theta = operator.theta
    quant = dict(theta.quant)
    if str(quant.get("linear_method", "")) != "w8a8":
        return None
    if str(quant.get("weight_format", "")) != "int8":
        return None
    if str(quant.get("act_format", "")) != "int8":
        return None
    storage = str(quant.get("weight_scale_format", ""))
    runtime = str(quant.get("runtime_weight_scale_format") or "bf16")
    if storage not in {"fp32", "bf16"} or runtime != "bf16":
        return None
    if int(quant.get("scale_group", 0) or 0) != 1:
        return None
    if not bool(quant.get("quantized", 0)):
        return None
    if bool(theta.find("bias", False)):
        return None
    try:
        k, n = int(dict(theta.matrix)["K"]), int(dict(theta.matrix)["N"])
    except (KeyError, TypeError, ValueError):
        return None
    if k < 1 or n < 1 or n % _NZ_N or k % _NZ_K:
        return None
    if _r31_image_text(compiled) is None:
        return None
    return k, n


def _source_record() -> dict:
    return {
        "repository": _LIBOPAPI["repository"],
        "path": _LIBOPAPI["path"],
        "sha256": _LIBOPAPI["sha256"],
        "symbol": _ENTRY,
        "revision": _LIBOPAPI["revision"],
        "state": _LIBOPAPI["state"],
    }


def _typed(k: int, n: int, image: str, storage: str,
           storage_bytes: float) -> dict:
    tile = [n // _NZ_N, k // _NZ_K, _NZ_K, _NZ_N]
    return {
        "dtype": "int8",
        "layout": "ND",
        "shapes": {
            "x": [k], "weight": [k, n],
            "weight_scale": [n], "output": [n],
        },
        "attrs": {
            "matrix_k": k, "matrix_n": n,
            "quantized": 1, "scale_group": 1,
            "linear_method": "w8a8", "implementation": "W8A8_DYNAMIC",
            "act_format": "int8", "weight_format": "int8",
            "output_format": "bf16", "output_dtype": "bf16",
            "weight_storage_dtype": "int8",
            "weight_logical_shape": [k, n],
            "weight_allocation_shape": None,
            "weight_allocation_status": "python_caller_source_unresolved",
            "weight_pre_nz_layout": "unresolved",
            "weight_device_shape": tile,
            "weight_device_layout": "FRACTAL_NZ",
            "weight_transform": "maybe_trans_nz",
            "weight_packer": _PACKER["symbol"],
            "weight_packer_source_path": _PACKER["path"],
            "weight_packer_source_sha256": _PACKER["sha256"],
            "weight_packer_revision": R31_REVISION,
            "weight_scale_storage_format": storage,
            "weight_scale_storage_bytes": storage_bytes,
            "weight_scale_format": "bf16",
            "weight_scale_bytes": 2.0,
            "weight_scale_runtime_format": "bf16",
            "weight_scale_runtime_bytes": 2.0,
            "weight_scale_allocation_shape": None,
            "weight_scale_allocation_status": "python_caller_source_unresolved",
            "weight_scale_device_shape": [n],
            "pertoken_scale_format": "fp32",
            "pertoken_scale_runtime_bytes": 4.0,
            "n_padding": "active_32",
            "active_n": n, "padded_n": n,
            "n_padding_tail": 0,
            "deployment_image_id": image,
            "deployment_source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "bias": False, "offset": None,
        },
        "rank_width": 2,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real R31 W8A8 WeightNz call spec, or None.

    Selection is the exact R31 deployment identity plus the compiled W8A8
    INT8 GEMM shape.  No model/cell name, R64/R35 source or measured timing
    chooses this branch.  The returned ``_CallSpec`` deliberately keeps the
    missing R31 Python caller/kernel source in ``unresolved``.
    """
    del phase
    dims = _r31_dims(compiled)
    if dims is None:
        return None
    k, n = dims
    image = _r31_image_text(compiled) or R31_IMAGE_SHA
    quant = dict(compiled.operator.theta.quant)
    storage = str(quant.get("weight_scale_format") or "fp32")
    storage_bytes = 4.0 if storage == "fp32" else 2.0
    from .native_calls import _api, _call_spec, _kernel

    return _call_spec(
        _api("aclnn", _ENTRY, aclnn_symbol=_ENTRY,
             version=R31_SOURCE_VERSION),
        _source_record(),
        _typed(k, n, R31_IMAGE_SHA if image.startswith(R31_IMAGE_PREFIX)
               else image, storage, storage_bytes),
        (_kernel("QuantBatchMatmulV3", "mix", evidence="measured"),),
        (),
        getattr(compiled.operator, "name", "gemm"),
        execution_core="mix",
        unresolved=(
            "python_caller_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
            "attributable_source_reference_unresolved",
        ),
    )


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _actual(binding: Any, k: int, n: int, m: int) -> dict:
    attrs = dict(binding.typed.get("attrs") or {})
    actual_attrs = {
        **attrs,
        "m": m, "tokens": m, "k": k, "n": n,
        "weight_scale_device_shape": [n],
        "pertoken_scale_shape": [m],
        "output_shape": [m, n],
        "active_n": n, "padded_n": n, "n_padding_tail": 0,
    }
    return {
        "dtype": binding.typed.get("dtype"),
        "layout": binding.typed.get("layout"),
        "shapes": {
            "x": [m, k], "weight": [k, n],
            "weight_scale": [n], "pertoken_scale": [m],
            "output": [m, n],
        },
        "attrs": actual_attrs,
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one R31 W8A8 WeightNz occurrence for the evaluated geometry.

    ``m`` is the only dynamic tensor axis.  K/N, the NZ tile shape and both
    scale formats stay the static identity.  The envelope mirrors
    ``native_qbmm.dynamic_qbmm`` but never claims a complete caller: the R31
    Python caller and kernel source remain explicit partial reasons.
    """
    api = binding.api or {}
    if api.get("entry") != _ENTRY and api.get("aclnn_symbol") != _ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("implementation") != "W8A8_DYNAMIC":
        return None
    image = str(attrs.get("deployment_image_id") or "")
    if not image.startswith(R31_IMAGE_PREFIX):
        return None
    shapes = dict(binding.typed.get("shapes") or {})
    weight = shapes.get("weight")
    if not weight or len(weight) != 2:
        return None
    k, n = int(weight[0]), int(weight[1])
    if k < 1 or n < 1 or n % _NZ_N or k % _NZ_K:
        return None
    m = int(getattr(geometry, "effective_tokens", 0) or 0)
    if m < 1:
        return None
    operator = getattr(compiled, "operator", None)
    if operator is not None and hasattr(operator, "theta"):
        try:
            matrix = dict(operator.theta.matrix)
            if int(matrix["K"]) != k or int(matrix["N"]) != n:
                return None
        except (KeyError, TypeError, ValueError):
            return None
    actual = _actual(binding, k, n, m)
    validated = validate(binding, actual)
    tile = [n // _NZ_N, k // _NZ_K, _NZ_K, _NZ_N]
    descriptors = {
        "x": _descriptor([m, k], "int8", "ND", "input"),
        "weight": _descriptor([k, n], "int8", "ND", "input"),
        "weight_scale": _descriptor([n], "bf16", "ND", "input"),
        "pertoken_scale": _descriptor([m], "fp32", "ND", "input"),
        "output": _descriptor([m, n], "bf16", "ND", "output"),
        "weight_nz": _descriptor(tile, "int8", "FRACTAL_NZ", "input"),
    }
    descriptors["weight"].update({
        "storage_dtype": "int8", "allocation_shape": None,
        "allocation_status": "python_caller_source_unresolved",
        "logical_shape": [k, n], "device_shape": tile,
        "device_layout": "FRACTAL_NZ",
    })
    descriptors["weight_scale"].update({
        "storage_format": attrs.get("weight_scale_storage_format", "fp32"),
        "storage_bytes": attrs.get("weight_scale_storage_bytes", 4.0),
        "runtime_format": "bf16", "runtime_bytes": 2.0,
        "allocation_shape": None,
        "allocation_status": "python_caller_source_unresolved",
        "device_shape": [n],
    })
    descriptors["pertoken_scale"].update({
        "runtime_format": "fp32", "runtime_bytes": 4.0,
        "producer": "unresolved",
        "producer_status": "python_caller_source_unresolved",
    })
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {
        "name": "QuantBatchMatmulV3", "role": "mix",
        "evidence": "measured",
    }
    kernel.update({
        "observed_task_type": "MIX_AIC",
        "owner_binary_path": _LIBOPAPI["path"],
        "owner_binary_sha256": _LIBOPAPI["sha256"],
        "source_status": "runtime_binary_only",
    })
    prepare = {
        "schema": "mindie-sim/qbmm-native-prepare/1",
        "call": _ENTRY,
        "native_entry": _ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "implementation": "W8A8_DYNAMIC",
        "source": source,
        "source_version": R31_SOURCE_VERSION,
        "wrapper_version": R31_WRAPPER_VERSION,
        "selected_host_variant": None,
        "selected_host_variant_status": "unresolved",
        "missing_source": _MISSING_CALLER,
        "kernel": kernel,
        "source_lines": {
            "caller": None, "kernel": None,
            "packer": list(_PACKER["lines"]),
            "packer_layout_constant": _PACKER["layout_constant"],
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {
            "x": "x", "weight": "weight",
            "weight_scale": "weight_scale",
            "pertoken_scale": "pertoken_scale",
            "bias": None, "offset": None,
            "output_dtype": "bf16", "output": "output",
        },
        "argument_order": ["x", "weight", "weight_scale", "pertoken_scale"],
        "observed_acl_tensor_order": [
            "x", "weight", "weight_scale", "pertoken_scale"],
        "observed_acl_tensor_dtypes": [
            "INT8", "INT8", "DT_BF16", "FLOAT"],
        "observed_acl_tensor_formats": [
            "ND", "FRACTAL_NZ", "ND", "ND"],
        "observed_output": {
            "dtype": "DT_BF16", "format": "ND", "shape": [m, n]},
        "optional_slots": {
            "bias": "not_present", "offset": "not_passed",
            "output_dtype": "bf16",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "weight_nz": tile,
        "weight_storage": {
            "logical_shape": [k, n], "allocation_shape": None,
            "allocation_status": "python_caller_source_unresolved",
            "allocation_dtype": "int8", "device_shape": tile,
            "device_layout": "FRACTAL_NZ",
            "transform": "maybe_trans_nz",
            "pre_nz_layout": "unresolved",
            "packer": dict(_PACKER),
        },
        "scale_storage": {
            "weight_scale": {
                "storage_format": attrs.get(
                    "weight_scale_storage_format", "fp32"),
                "storage_bytes": attrs.get("weight_scale_storage_bytes", 4.0),
                "runtime_format": "bf16", "runtime_bytes": 2.0,
                "allocation_shape": None,
                "allocation_status": "python_caller_source_unresolved",
                "device_shape": [n],
            },
            "pertoken_scale": {
                "runtime_format": "fp32", "runtime_bytes": 4.0,
                "producer": "unresolved",
                "producer_status": "python_caller_source_unresolved",
                "shape": [m],
            },
        },
        "deployment": {
            "image_id": R31_IMAGE_SHA,
            "revision": R31_REVISION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "selection": (
                "R31 image identity + compiled linear_method=w8a8, "
                "int8/int8, scale_group=1, "
                f"{attrs.get('weight_scale_storage_format', 'fp32')} "
                "stored weight scale"),
            "source_path": _LIBOPAPI["path"],
            "source_sha256": _LIBOPAPI["sha256"],
            "packer_path": _PACKER["path"],
            "packer_sha256": _PACKER["sha256"],
            "caller_path": _MISSING_CALLER,
            "caller_sha256": None,
            "caller_status": "unresolved",
            "kernel_source_path": None,
            "kernel_source_sha256": None,
            "kernel_source_status": "runtime_binary_only",
        },
        "match": {
            "native_type": "QuantBatchMatmulV3",
            "entry": _ENTRY,
            "call_id": binding.call_id,
            "deployment_image_id": R31_IMAGE_SHA,
            "source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "implementation": "W8A8_DYNAMIC",
            "k": k, "n": n,
            "active_n": n, "padded_n": n, "n_padding": "active_32",
            "weight_layout": "FRACTAL_NZ",
            "weight_allocation_shape": None,
            "weight_allocation_status": "python_caller_source_unresolved",
            "weight_device_shape": tile,
            "activation_dtype": "int8", "weight_dtype": "int8",
            "weight_scale_storage_format": attrs.get(
                "weight_scale_storage_format", "fp32"),
            "weight_scale_format": "bf16",
            "pertoken_scale_format": "fp32",
            "output_dtype": "bf16",
            "scale_group": 1, "quantized": 1,
            "source_sha256": _LIBOPAPI["sha256"],
            "packer_sha256": _PACKER["sha256"],
            "caller_sha256": None,
            "kernel_sha256": None,
            "complete": False, "partial": True,
        },
        "binary_selection": {
            "status": "runtime_binary_only",
            "reason": "r31_python_caller_and_kernel_source_not_archived",
            "owner_kernel": "QuantBatchMatmulV3",
            "runtime_library": _LIBOPAPI["path"],
            "runtime_library_sha256": _LIBOPAPI["sha256"],
            "acl_symbol": _ENTRY,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "python_caller_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
        ],
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_physical_completion",
        ],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {
                "dtype": validated.get("dtype"),
                "layout": validated.get("layout"),
                "shapes": validated.get("shapes"),
                "attrs": validated.get("attrs"),
            }}


# ---------------------------------------------------------------------------
# Independent ABI leaf: aclnnQuantMatmulWeightNz (QuantBatchMatmulV3 W8A8)
# ---------------------------------------------------------------------------

ABI_ENTRY = _ENTRY
ABI_NAMESPACE = "aclnn"
ABI_ATTRS = ()
ABI_OPTIONAL_ATTRS = ("weight_scale_storage_format", "output_dtype",
                      "scale_group", "weight_layout", "bias",
                      "weight_scale_storage_bytes")
ABI_UNRESOLVED = (
    "python_caller_source_unresolved",
    "kernel_source_runtime_binary_only",
    "acl_binary_selection_trace_only",
    "pertoken_scale_producer_source_unresolved",
)


def abi_contract() -> dict[str, Any]:
    """The independent tensor/attr/version contract for this ABI leaf."""
    return {
        "schema": ABI_SCHEMA,
        "entry": ABI_ENTRY,
        "namespace": ABI_NAMESPACE,
        "aclnn_symbol": _ENTRY,
        "version": {"api_prefix": "cann-9.2.0", "source": ["r31"],
                    "image_prefix": R31_IMAGE_PREFIX},
        "tensors": [
            {"name": "x", "role": "input", "dtype": "int8",
             "rank": 2, "required": True},
            {"name": "weight", "role": "weight", "dtype": "int8",
             "rank": 2, "required": False},
            {"name": "weight_nz", "role": "weight", "dtype": "int8",
             "rank": 4, "required": False, "layout": "FRACTAL_NZ"},
            {"name": "weight_scale", "role": "scale", "dtype": "bf16",
             "rank": 1, "required": True},
            {"name": "pertoken_scale", "role": "scale", "dtype": "fp32",
             "rank": 1, "required": True},
            {"name": "output", "role": "output", "dtype": "bf16",
             "rank": 2, "required": True},
            {"name": "bias", "role": "input", "dtype": "bf16",
             "rank": 1, "required": False},
        ],
        "attrs": ABI_ATTRS,
        "optional_attrs": ABI_OPTIONAL_ATTRS,
        "kernels": [{"name": "QuantBatchMatmulV3", "role": "mix",
                     "evidence": "measured"}],
        "relation": "one_to_one",
        "source": {"repository": "cann", "path": _LIBOPAPI["path"],
                   "sha256": _LIBOPAPI["sha256"], "symbol": _ENTRY,
                   "revision": R31_SOURCE_VERSION,
                   "packer": dict(_PACKER)},
    }


def _abi_version(version: Any) -> tuple[str | None, dict, list[str]]:
    values: dict[str, Any] = {}
    if isinstance(version, str):
        values["api"] = version
    elif isinstance(version, Mapping):
        for key in ("api", "api_version", "aclnn", "cann", "source",
                    "source_revision", "revision", "image_id", "image"):
            if version.get(key) is not None:
                values[key] = str(version[key])
    elif version is not None:
        return None, {}, ["invalid:version"]
    api = values.get("api") or values.get("api_version")
    source = values.get("source") or values.get("source_revision") \
        or values.get("revision")
    image = digest(values.get("image_id") or values.get("image"))
    if api is not None and not api.startswith("cann-9.2.0"):
        return None, values, ["version:api_not_cann_9.2.0"]
    if source is not None and source != "r31":
        return None, values, ["version:source_not_r31"]
    if image and not image.startswith(R31_IMAGE_PREFIX):
        return None, values, ["version:image_not_r31"]
    # A standalone timing-free CSV capture carries no container SHA.  Absent
    # identity remains an unresolved declaration (not a hard gate); an
    # explicitly declared foreign stack above still refuses the r31 lowering.
    return "r31", {"api": api or R31_SOURCE_VERSION, "source": "r31",
                   "image_id": "sha256:" + R31_IMAGE_SHA}, []


def predict_abi(tensors: Mapping[str, Any], *,
                attrs: Mapping[str, Any] | None = None,
                version: Any = None,
                conditions: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
    """One typed WeightNz call: runtime slots, optional ND preparation, epilogue.

    The native call consumes INT8 x, an INT8 FRACTAL_NZ weight, BF16
    weight_scale [N] and FP32 pertoken_scale [M], and writes BF16 [M,N].
    If only ND weight is supplied, a real ``maybe_trans_nz`` producer and
    materialization dependency are declared; a pre-materialized NZ slot is
    not charged again.  The INT32 accumulator is rescaled once after the K
    reduction by the real FP32/BF16 scales.
    """
    attrs = dict(attrs or {})
    conditions = dict(conditions or {})
    reasons: list[str] = []
    revision, canonical_version, version_gaps = _abi_version(version)
    reasons.extend(version_gaps)
    unknown_attrs = set(attrs) - set(ABI_ATTRS) - set(ABI_OPTIONAL_ATTRS)
    if unknown_attrs:
        reasons.append("unmodeled:attrs:" + ",".join(sorted(unknown_attrs)))
    if str(attrs.get("output_dtype", "bf16")).lower() not in {"bf16", "bfloat16"}:
        reasons.append("mismatch:attrs.output_dtype")
    if int_value(attrs.get("scale_group", 1)) != 1:
        reasons.append("unmodeled:attrs.scale_group")
    weight_layout = attrs.get("weight_layout")
    if weight_layout not in (None, "ND", "FRACTAL_NZ"):
        reasons.append("mismatch:attrs.weight_layout")
    if attrs.get("bias") is not None:
        reasons.append("unmodeled:attrs.bias")
    raw_bias = (tensors or {}).get("bias")
    if (isinstance(raw_bias, Mapping)
            and raw_bias.get("present", True) is not False):
        reasons.append("unmodeled:tensors.bias")
    x, reason = abi_tensor(tensors, "x", role="input", dtype="int8",
                           rank=2, dtypes=("int8",))
    reasons.extend(reason)
    logical_weight, reason = abi_tensor(tensors, "weight", role="weight",
                                        dtype="int8", rank=2, optional=True,
                                        dtypes=("int8",))
    reasons.extend(reason)
    nz_weight, reason = abi_tensor(tensors, "weight_nz", role="weight",
                                   dtype="int8", rank=4, optional=True,
                                   layouts=("FRACTAL_NZ",),
                                   dtypes=("int8",))
    reasons.extend(reason)
    raw_ws = (tensors or {}).get("weight_scale")
    raw_ws_dtype = str((raw_ws.get("dtype") if isinstance(raw_ws, Mapping)
                        else None) or "bf16").lower()
    ws_dtype = "fp32" if raw_ws_dtype in {"fp32", "float32"} else "bf16"
    ws_dtypes = (("fp32", "float32") if ws_dtype == "fp32"
                 else ("bf16", "bfloat16"))
    ws_width = 4 if ws_dtype == "fp32" else 2
    weight_scale, reason = abi_tensor(
        tensors, "weight_scale", role="scale", dtype=ws_dtype, rank=1,
        dtypes=("bf16", "bfloat16", "fp32", "float32"))
    reasons.extend(reason)
    pertoken_scale, reason = abi_tensor(
        tensors, "pertoken_scale", role="scale", dtype="fp32", rank=1,
        dtypes=("fp32", "float32"))
    reasons.extend(reason)
    output, reason = abi_tensor(tensors, "output", role="output",
                                dtype="bf16", rank=2,
                                dtypes=("bf16", "bfloat16"))
    reasons.extend(reason)
    if logical_weight is None and nz_weight is None:
        reasons.append("missing:tensors.weight_or_weight_nz")
    if weight_scale is not None:
        reasons.extend(interpretation_reasons(
            weight_scale, "weight_scale", ws_dtypes))
    if pertoken_scale is not None:
        reasons.extend(interpretation_reasons(
            pertoken_scale, "pertoken_scale", ("fp32", "float32")))
    if x is not None:
        reasons.extend(interpretation_reasons(x, "x", ("int8",)))
    if output is not None:
        reasons.extend(interpretation_reasons(
            output, "output", ("bf16", "bfloat16")))
    rows = n = k = None
    if x is not None and output is not None:
        rows, k = x["shape"]
        out_rows, n = output["shape"]
        if out_rows != rows:
            reasons.append("mismatch:tensors.output.rows")
    if nz_weight is not None:
        nz_n, nz_k, tile_k, tile_n = nz_weight["shape"]
        nz_k_total, nz_n_total = nz_k * _NZ_K, nz_n * _NZ_N
        # A FRACTAL_NZ weight stores whole 16x32 tiles, so a K or N tail is
        # padded *up* to the next tile: a 258-wide K is 17 tiles (272 stored).
        # The storage must cover the logical extent with less than one tile of
        # padding; exact divisibility is not a hardware requirement.
        k_short = (k is not None
                   and (nz_k_total < k or nz_k_total - k >= _NZ_K))
        n_short = (n is not None
                   and (nz_n_total < n or nz_n_total - n >= _NZ_N))
        if ([tile_k, tile_n] != [_NZ_K, _NZ_N]
                or k_short or n_short):
            reasons.append("mismatch:tensors.weight_nz.shape")
        k = k if k is not None else nz_k_total
        n = n if n is not None else nz_n_total
        reasons.extend(direction_reasons(
            nz_weight, "weight_nz", ("k_by_n", "in_by_out")))
    logical_direction = None
    if logical_weight is not None:
        first, second = logical_weight["shape"]
        if n is not None and first == n and second != n:
            logical_k, logical_n = second, first
            logical_direction = "n_by_k"
        elif n is not None and second == n:
            logical_k, logical_n = first, second
            logical_direction = "k_by_n"
        else:
            reasons.append("missing:tensors.weight.orientation")
            logical_k = logical_n = None
        if logical_k is not None:
            if k is not None and logical_k != k:
                reasons.append("mismatch:tensors.weight.contraction")
            if n is not None and logical_n != n:
                reasons.append("mismatch:tensors.weight.columns")
            k, n = logical_k, logical_n
            allowed = ("k_by_n", "in_by_out") if logical_direction == "k_by_n" \
                else ("n_by_k", "out_by_in")
            reasons.extend(direction_reasons(logical_weight, "weight", allowed))
    # The NZ tile padding above already proves the storage covers the
    # logical K/N extent; a K or N tail is real work, not a rejection.
    if n is not None and weight_scale is not None:
        if weight_scale["shape"] != [n]:
            reasons.append("mismatch:tensors.weight_scale.shape")
    if rows is not None and pertoken_scale is not None:
        if pertoken_scale["shape"] != [rows]:
            reasons.append("mismatch:tensors.pertoken_scale.shape")
    if rows is not None and n is not None and k is not None:
        nz_tile = [n // _NZ_N, k // _NZ_K, _NZ_K, _NZ_N]
        reasons.extend(view_reasons(x, "x", "storage",
                                    shape=[rows, k], dtype="int8",
                                    layout="ND"))
        reasons.extend(view_reasons(output, "output", "storage",
                                    shape=[rows, n], dtype="bf16",
                                    layout="ND"))
        reasons.extend(view_reasons(pertoken_scale, "pertoken_scale",
                                    "storage", shape=[rows],
                                    dtype="fp32", layout="ND"))
        if weight_scale is not None:
            reasons.extend(view_reasons(weight_scale, "weight_scale",
                                        "storage", shape=[n]))
        nz_logical = ((nz_weight or {}).get("views") or {}).get("logical")
        reasons.extend(slot_view_reasons(
            nz_weight, "weight_nz",
            logical=({"shape": [k, n], "dtype": "int8", "layout": "ND",
                      "direction": "k_by_n"}
                     if str((nz_logical or {}).get("layout") or "").upper()
                     == "ND" else None),
            storage={"shape": nz_tile, "dtype": "int8",
                     "layout": "FRACTAL_NZ"},
            device={"shape": nz_tile, "dtype": "int8",
                    "layout": "FRACTAL_NZ"}))
        reasons.extend(slot_view_reasons(
            x, "x",
            logical={"shape": [rows, k], "dtype": "int8", "layout": "ND"},
            storage={"shape": [rows, k], "dtype": "int8", "layout": "ND"},
            device={"shape": [rows, k], "dtype": "int8", "layout": "ND"}))
        reasons.extend(slot_view_reasons(
            output, "output",
            logical={"shape": [rows, n], "dtype": "bf16", "layout": "ND"},
            storage={"shape": [rows, n], "dtype": "bf16", "layout": "ND"},
            device={"shape": [rows, n], "dtype": "bf16", "layout": "ND"}))
        reasons.extend(slot_view_reasons(
            pertoken_scale, "pertoken_scale",
            logical={"shape": [rows], "dtype": "fp32", "layout": "ND"},
            storage={"shape": [rows], "dtype": "fp32", "layout": "ND"},
            device={"shape": [rows], "dtype": "fp32", "layout": "ND"}))
        reasons.extend(slot_view_reasons(
            weight_scale, "weight_scale",
            logical={"shape": [n], "dtype": ws_dtype, "layout": "ND"},
            storage={"shape": [n], "dtype": ws_dtype, "layout": "ND"},
            device={"shape": [n], "dtype": ws_dtype, "layout": "ND"}))
        if weight_layout == "ND":
            reasons.append("mismatch:attrs.weight_layout")
        if logical_weight is not None:
            expected = ([n, k] if logical_direction == "n_by_k" else [k, n])
            reasons.extend(view_reasons(
                logical_weight, "weight", "storage", shape=expected,
                dtype="int8"))
            reasons.extend(view_reasons(
                logical_weight, "weight", "device", shape=expected,
                dtype="int8"))
    # The call-time weight_scale slot is BF16 or FP32 [N].  A declared storage
    # view on this slot therefore describes runtime storage only; checkpoint /
    # upstream format is a separate attribute record.
    view_storage = (weight_scale or {}).get("views", {}).get("storage") \
        if weight_scale is not None else None
    view_format = str((view_storage or {}).get("dtype") or "").lower() or None
    if view_storage is not None:
        if view_format not in set(ws_dtypes):
            reasons.append("mismatch:tensors.weight_scale.storage_dtype")
        if list(view_storage.get("shape") or ()) != [n]:
            reasons.append("mismatch:tensors.weight_scale.storage_shape")
    view_device = (weight_scale or {}).get("views", {}).get("device") \
        if weight_scale is not None else None
    if view_device is not None:
        if str(view_device.get("dtype") or "").lower() not in set(ws_dtypes):
            reasons.append("mismatch:tensors.weight_scale.device_dtype")
        if list(view_device.get("shape") or ()) != [n]:
            reasons.append("mismatch:tensors.weight_scale.device_shape")
    for name, row, shape, dtype in (
            ("x", x, [rows, k], "int8"),
            ("pertoken_scale", pertoken_scale, [rows], "fp32"),
            ("output", output, [rows, n], "bf16")):
        device = ((row or {}).get("views") or {}).get("device")
        if device is None:
            continue
        if str(device.get("dtype") or "").lower() not in {dtype, "bfloat16"
                                                           if dtype == "bf16"
                                                           else dtype}:
            reasons.append(f"mismatch:tensors.{name}.device_dtype")
        if list(device.get("shape") or ()) != shape:
            reasons.append(f"mismatch:tensors.{name}.device_shape")
    checkpoint_format = attrs.get("weight_scale_storage_format")
    if checkpoint_format is not None:
        checkpoint_format = str(checkpoint_format).lower()
        if checkpoint_format not in {"fp32", "bf16"}:
            reasons.append("unmodeled:attrs.weight_scale_storage_format")
            checkpoint_format = None
    declared_width = int_value(attrs.get("weight_scale_storage_bytes"))
    if declared_width is not None and declared_width not in {2, 4}:
        reasons.append("mismatch:attrs.weight_scale_storage_bytes")
        declared_width = None
    if checkpoint_format is None and declared_width is not None:
        checkpoint_format = "fp32" if declared_width == 4 else "bf16"
    checkpoint_width = (4 if checkpoint_format == "fp32" else 2
                        if checkpoint_format == "bf16" else None)
    if (declared_width is not None and checkpoint_width is not None
            and declared_width != checkpoint_width):
        reasons.append("mismatch:attrs.weight_scale_storage_bytes")
    checkpoint_bytes = (n * checkpoint_width
                        if checkpoint_width is not None else None)
    current = abi_status(reasons)
    invocation = {"tensors": dict(tensors), "attrs": attrs,
                  "version": canonical_version, "conditions": conditions}
    if current != "predicted":
        return abi_result(
            status_value=current, entry=ABI_ENTRY, reasons=reasons,
            invocation=invocation, selected_call=None, geometry=None,
            work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    assert rows is not None and n is not None and k is not None
    assert weight_scale is not None and pertoken_scale is not None
    assert output is not None and x is not None
    preparation = None
    nz_prov = provenance(nz_weight)
    nz_prepared = (nz_prov.get("origin") == "prepared"
                   or nz_prov.get("producer") == _PACKER["symbol"])
    if nz_weight is not None and not nz_prepared:
        weight_slot = with_views(
            nz_weight, storage_shape=nz_tile, storage_dtype="int8",
            storage_layout="FRACTAL_NZ", device_shape=nz_tile,
            device_dtype="int8", device_layout="FRACTAL_NZ",
            interpret_dtype="int8")
        weight_slot["name"] = "weight_nz"
        weight_slot["role"] = "weight"
        weight_slot["views"]["logical"] = {
            "shape": [k, n], "dtype": "int8", "layout": "ND",
            "direction": "k_by_n"}
        weight_slot["provenance"] = {
            "origin": "external", "producer": None,
            "completion": "already_materialized"}
        weight_path = "pre_materialized_nz"
    else:
        input_shape = ([n, k] if logical_direction == "n_by_k" else [k, n])
        if isinstance(nz_prov.get("input"), Mapping):
            input_shape = list(nz_prov["input"].get("shape") or input_shape)
        if nz_prepared and isinstance(nz_prov.get("preparation"), Mapping):
            preparation = dict(nz_prov["preparation"])
            preparation["included_in_call_footprint"] = False
        else:
            preparation = preparation_record(
                kind="maybe_trans_nz", producer=_PACKER["symbol"],
                input_slot="weight", input_shape=input_shape,
                output_slot="weight_nz", output_shape=nz_tile,
                source_path=_PACKER["path"],
                source_sha256=_PACKER["sha256"],
                revision=R31_REVISION)
        if nz_weight is not None:
            weight_slot = with_views(
                nz_weight, storage_shape=nz_tile, storage_dtype="int8",
                storage_layout="FRACTAL_NZ", device_shape=nz_tile,
                device_dtype="int8", device_layout="FRACTAL_NZ",
                interpret_dtype="int8")
            weight_slot["name"] = "weight_nz"
            weight_slot["role"] = "weight"
            weight_slot["views"]["logical"] = {
                "shape": [k, n], "dtype": "int8", "layout": "ND",
                "direction": "k_by_n"}
        else:
            weight_slot = {
                "name": "weight_nz", "role": "weight", "dtype": "int8",
                "layout": "FRACTAL_NZ", "shape": nz_tile,
                "optional": False, "present": True,
                "views": {
                    "logical": {"shape": [k, n], "dtype": "int8",
                                "layout": "ND", "direction": "k_by_n"},
                    "storage": {"shape": nz_tile, "dtype": "int8",
                                "layout": "FRACTAL_NZ", "bytes": n * k},
                    "device": {"shape": nz_tile, "dtype": "int8",
                               "layout": "FRACTAL_NZ", "bytes": n * k},
                    "interpretation": {"dtype": "int8",
                                       "layout": "FRACTAL_NZ"},
                },
            }
        weight_slot["provenance"] = {
            "origin": "prepared", "producer": _PACKER["symbol"],
            "completion": "before_native_call",
            "input": dict(preparation["input"]),
            "output": dict(preparation["output"]),
            "preparation_id": preparation["preparation_id"],
        }
        weight_path = "nd_to_nz_materialization"
    # Normalize the one call-time scale slot in place; selected, tensors and
    # work all read this same object.  Checkpoint storage stays separate.
    weight_scale["views"] = {
        "logical": {"shape": [n], "dtype": ws_dtype, "layout": "ND"},
        "storage": {"shape": [n], "dtype": ws_dtype, "layout": "ND",
                    "bytes": n * ws_width},
        "device": None,
        "interpretation": {"dtype": ws_dtype, "layout": "ND"},
    }
    upstream_storage = (
        {"shape": [n], "dtype": checkpoint_format,
         "format": checkpoint_format, "bytes": checkpoint_bytes,
         "role": "checkpoint_or_upstream"}
        if checkpoint_format is not None else
        {"status": "not_declared", "role": "checkpoint_or_upstream",
         "included_in_call_footprint": False})
    x = with_views(x, storage_shape=[rows, k], storage_dtype="int8",
                   storage_layout="ND", interpret_dtype="int8")
    pertoken_scale = with_views(
        pertoken_scale, storage_shape=[rows], storage_dtype="fp32",
        storage_layout="ND", interpret_dtype="fp32")
    output = with_views(output, storage_shape=[rows, n],
                        storage_dtype="bf16", storage_layout="ND",
                        interpret_dtype="bf16")
    if reasons:
        return abi_result(
            status_value=abi_status(reasons), entry=ABI_ENTRY, reasons=reasons,
            invocation=invocation, selected_call=None, geometry=None,
            work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    epilogue = row_column_scale_epilogue(m=rows, n=n, column_dtype=ws_dtype)
    macs = rows * n * k
    issued = align(rows, 16) * align(n, 32) * align(k, 16)
    weight_scale_bytes = n * ws_width
    x_bytes = rows * k
    weight_bytes = n * k
    pertoken_bytes = rows * 4
    output_bytes = rows * n * 2
    call_footprint_bytes = (x_bytes + weight_bytes
                            + weight_scale_bytes
                            + pertoken_bytes + output_bytes)
    geometry = {
        "m": rows, "n": n, "k": k,
        "padded_m": align(rows, 16), "padded_n": align(n, 32),
        "padded_k": align(k, 16), "weight_nz": nz_tile,
        "weight_path": weight_path,
        "weight_scale_storage_width": checkpoint_width,
        "weight_scale_width": ws_width,
        "epilogue": {"operations_per_output": 4,
                     "row_scale_bytes": rows * 4,
                     "column_scale_bytes": n * ws_width},
        "basis": "declared_tensor_shapes_and_32x16_nz_alignment",
    }
    work = {
        "cube": {"macs": macs, "flops": 2 * macs, "issued_macs": issued,
                 "issued_basis": "16x32x16_nz_padding"},
        "vector": {
            "ops": epilogue["vector_ops"],
            "epilogue_ops": epilogue["vector_ops"],
            "scale_load_elements": rows + n,
            "basis": "int32_accumulator_to_bf16_row_column_scale",
        },
        "epilogue": epilogue,
        "bytes": {
            "input": x_bytes, "weight": weight_bytes,
            "weight_scale": weight_scale_bytes,
            "pertoken_scale": pertoken_bytes,
            "output": output_bytes, "total": call_footprint_bytes,
            "basis": "call_boundary_tensor_footprint",
        },
        "call_boundary": {
            "tensor_footprint_bytes": call_footprint_bytes,
            "issued_traffic": None,
            "issued_traffic_status": "requires_selected_load_store_plan",
        },
        "checkpoint_or_upstream": {
            "status": ("declared" if checkpoint_format is not None
                       else "not_declared"),
            "weight_scale_storage_format": checkpoint_format,
            "weight_scale_storage_bytes": checkpoint_bytes,
            "weight_scale_storage_width": checkpoint_width,
            "included_in_call_footprint": False,
            "conversion_owner": "producer_or_checkpoint_loader",
        },
        "preparation": (
            {**preparation, "included_in_call_footprint": False}
            if preparation is not None else None),
        "scale_load_elements": rows + n,
    }
    members = tuple(conditions.get("members") or ())
    relation = "many_to_one_fused" if len(members) > 1 else "one_to_one"
    variant = ("w8a8_dynamic_weight_nz_pre_materialized"
               if preparation is None else "w8a8_dynamic_weight_nz_from_nd")
    call = abi_selected(
        entry=ABI_ENTRY, namespace=ABI_NAMESPACE, aclnn_symbol=_ENTRY,
        api_version=canonical_version.get("api"), source_revision="r31",
        image_id=canonical_version["image_id"],
        source={"repository": "cann", "path": _LIBOPAPI["path"],
                "sha256": _LIBOPAPI["sha256"], "symbol": _ENTRY,
                "revision": R31_SOURCE_VERSION},
        kernels=[{"name": "QuantBatchMatmulV3", "role": "mix",
                  "evidence": "measured",
                  "runtime_library": _LIBOPAPI["path"],
                  "runtime_library_sha256": _LIBOPAPI["sha256"]}],
        execution_core="mix", relation=relation, members=members,
        tensors=[x, weight_slot, weight_scale,
                 pertoken_scale, output],
        attrs={"weight_scale_format": ws_dtype,
               "output_dtype": "bf16", "scale_group": 1,
               "weight_path": weight_path,
               "epilogue_operations_per_output": 4,
               "input_dtypes": ["int8", "int8", ws_dtype, "fp32", "bf16"],
               "input_slots": ["x", "weight_nz", "weight_scale",
                               "pertoken_scale", "output"],
               "offset": "not_present"},
        regions=["projection"], variant=variant)
    call["unresolved"] = list(ABI_UNRESOLVED)
    metadata = {
        "weight_nz": nz_tile,
        "weight_path": weight_path,
        "weight_device_layout": "FRACTAL_NZ",
        "weight_logical_shape": [k, n],
        "weight_logical_direction": "k_by_n",
        "weight_allocation_shape": None,
        "weight_allocation_status": (
            "python_caller_source_unresolved" if preparation is None
            else "materialized_by_declared_producer"),
        "weight_pre_nz_layout": "unresolved" if preparation is None else "ND",
        "weight_transform": ("maybe_trans_nz" if preparation is not None
                             else "pre_materialized"),
        "weight_packer": dict(_PACKER),
        "weight_scale_storage_shape": [n],
        "weight_scale_storage_format": checkpoint_format,
        "weight_scale_storage_bytes_total": checkpoint_bytes,
        "weight_scale_shape": [n],
        "weight_scale_format": ws_dtype,
        "weight_scale_bytes": weight_scale_bytes,
        "n_padding": "active_32", "padded_n": n,
        "matrix_epilogue": {
            "kind": "row_column_scale",
            "row_scale_bytes": epilogue["row_scale_bytes"],
            "column_scale_bytes": epilogue["column_scale_bytes"],
            "operations_per_output": epilogue["operations_per_output"],
        },
        "matrix_output_contract": "int32_cast_row_scale_column_scale_output_cast",
        "activation_scale_format": "fp32",
        "activation_scale_bytes": rows * 4,
        "activation_scale_shape": [rows],
        "int8_weight_metadata": "scale_only",
        "column_metadata_policy": "declared_scale_only_storage_traffic",
        "input_dtypes": ["int8", "int8", ws_dtype, "fp32", "bf16"],
        "offset": "not_present",
        "footprint_vs_traffic": "weight/scale storage bytes are upstream "
                                "footprint; issued traffic requires the load plan",
    }
    scale = {
        "weight_scale": {
            "shape": [n], "dtype": ws_dtype,
            "bytes": weight_scale_bytes,
            "runtime": {"shape": [n], "dtype": ws_dtype, "format": ws_dtype,
                        "bytes": weight_scale_bytes},
            "storage": upstream_storage,
            "native_call_read_bytes": weight_scale_bytes,
        },
        "pertoken_scale": {
            "shape": [rows], "dtype": "fp32", "bytes": pertoken_bytes,
            "producer": "producer_or_checkpoint_loader",
            "producer_status": "not_part_of_this_call",
        },
        "epilogue": {
            "kind": "row_column_scale",
            "row": dict(epilogue["row_scale"]),
            "column": dict(epilogue["column_scale"]),
            "operations_per_output": epilogue["operations_per_output"],
            "applied_after": "complete_k_reduction",
            "owner": "matrix_epilogue",
        },
    }
    accumulator = {
        "dtype": "int32", "shape": [rows, n], "elements": rows * n,
        "bytes": rows * n * 4, "abi_tensor": False, "on_chip": True,
        "basis": "w8a8_int8_matmul_internal_accumulator",
        "residency_status": "requires_selected_matrix_program",
        "epilogue_consumes": "int32_accumulator",
        "source_status": "mechanism_inferred_not_source_declared",
    }
    regions = [{
        "id": "projection", "relation": relation, "rows": rows,
        "cols": n, "inner": k, "macs": macs,
        "issued_macs": issued, "outputs": ["output"],
        "members": list(members),
    }]
    if logical_weight is not None:
        weight_logical = logical_weight
    else:
        weight_logical = {
            "name": "weight", "role": "weight", "dtype": "int8",
            "shape": [k, n], "layout": "ND", "optional": False,
            "present": True, "direction": "k_by_n",
            "views": {"logical": {"shape": [k, n], "dtype": "int8",
                                  "layout": "ND",
                                  "direction": "k_by_n"},
                      "storage": None, "device": None,
                      "interpretation": {"dtype": "int8", "layout": "ND"}},
        }
    return abi_result(
        status_value="predicted", entry=ABI_ENTRY, reasons=[],
        invocation=invocation, selected_call=call, geometry=geometry,
        work=work,
        tensors={"x": x, "weight": weight_logical,
                 "weight_nz": weight_slot,
                 "weight_scale": weight_scale,
                 "pertoken_scale": pertoken_scale, "output": output},
        scale=scale, metadata=metadata, accumulator=accumulator,
        regions=regions)


def operator_plan(record, *, tile=(16, 32, 16), slots=2):
    """T12/T10 consumer entry for the selected real OperatorPlan."""
    from .native_matrix_abi import build_operator_plan
    return build_operator_plan(record, tile=tile, slots=slots)
