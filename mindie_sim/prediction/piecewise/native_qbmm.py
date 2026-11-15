"""Prepare-only standalone QuantBatchMatmulV3 invocation.

[Role] Expand the selected R64/R35 W8A8-MXFP8 linear wrapper's standalone
  ``torch_npu.npu_quant_matmul`` call into real operand/scale/bias/output
  descriptors for the evaluated token geometry.
[Invariants] Only the source-selected MXFP8 FP8/E8M0 group-32 branch is
  expanded.  Static identity (K/N, dtypes, layout transform, implementation)
  stays in the binding; M, per-token scale rows and output rows live only in
  the occurrence.  No tensor is allocated, no pointer is fabricated and no
  numeric prediction is changed.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing selection and
  source call arguments.
[Independent ABI] The independent-ABI mapper below admits the r64/r35 revisions only when
  the version record names one of them; no revision is guessed from a model.
"""
from __future__ import annotations

from typing import Any, Mapping

from .native_matrix_abi import (
    ABI_SCHEMA, align, ceil_div, digest, direction_reasons,
    interpretation_reasons, int_value,
    result as abi_result, row_column_scale_epilogue,
    preparation_record, provenance,
    selected as abi_selected, status as abi_status, tensor as abi_tensor,
    slot_view_reasons, view_reasons, with_views)

R35_IMAGE_SHA = "5461a1a8253"
_R64_WRAPPER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
    "sha256": "38de2275cf33ced653b8398e98a07f483489dc8e7353b2b3c238a5ff00782f79",
    "symbol": "AscendW8A8MXFP8DynamicLinearMethod.apply",
    "revision": "r64/8d422528",
    "state": "task_artifact",
}
_R35_WRAPPER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
    "sha256": "51ece928e77420a7f6600e2d87c46e9397a5140b6acee324aa75fdb97266d009",
    "symbol": "AscendW8A8MXFP8DynamicLinearMethod.apply",
    "revision": "r35/5461a1a8",
    "state": "task_artifact",
}
_R64_KERNEL = {
    "repository": "cann",
    "path": "op_kernel/quant_batch_matmul_v3/arch35/quant_batch_matmul_v3.cpp",
    "sha256": "13daf996890d70e7b1dc9275ab3a464043860e67d6df730e28742393c023ec88",
    "symbol": "QuantBatchMatmulV3",
    "revision": "cann-9.2.1-r64",
    "state": "task_artifact",
    "evidence": "source_declared",
    "source_status": "kernel_source",
}
_R35_KERNEL = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": "QuantBatchMatmulV3",
    "revision": "cann-9.2.0-r35",
    "state": "task_artifact",
    "evidence": "runtime_binary",
    "source_status": "runtime_binary_only",
}
QBMM_SOURCES = {
    "r64": {
        "wrapper": _R64_WRAPPER, "kernel": _R64_KERNEL,
        "version": "r64/8d422528", "image_sha": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
        "source_lines": {"qmm_call": [143, 153],
                         "scale_allocation": [83, 83],
                         "weight_transform": [160, 230]},
        "selected_host_variant": "apply", "rejected_host_variants": [],
    },
    "r35": {
        "wrapper": _R35_WRAPPER, "kernel": _R35_KERNEL,
        "version": "r35/5461a1a8", "image_sha": R35_IMAGE_SHA,
        "source_lines": {"qmm_call": [150, 160],
                         "scale_allocation": [89, 89],
                         "weight_transform": [223, 305],
                         "alternate_gmm": [167, 221]},
        "selected_host_variant": "apply",
        "rejected_host_variants": ["apply_single_group_gmm:GroupedMatmulV3"],
    },
}
QBMM_SOURCE = QBMM_SOURCES["r64"]["kernel"]
QBMM_WRAPPER_SOURCE = QBMM_SOURCES["r64"]["wrapper"]
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def dynamic_qbmm(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Full prepare-only call for the selected standalone MXFP8 QBMM."""
    api = binding.api or {}
    if api.get("entry") != "torch_npu.npu_quant_matmul":
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("implementation") != "W8A8_MXFP8":
        return None
    selected_source = str(attrs.get("deployment_source_revision") or "")
    source = QBMM_SOURCES.get(selected_source)
    if source is None:
        return None
    shapes = dict(binding.typed.get("shapes") or {})
    if "weight" not in shapes:
        return None
    k, n = (int(item) for item in shapes["weight"])
    m = int(getattr(geometry, "effective_tokens", 0) or 0)
    if m < 1 or k < 1 or n < 1:
        return None
    group = int(attrs.get("group_size", attrs.get("scale_group", 32)) or 32)
    if group < 1:
        return None
    pairs = _ceil_div(_ceil_div(k, group), 2)
    cache_key = (binding.call_id, m)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    act_dtype = str(attrs.get("act_format") or "fp8_e4m3")
    weight_dtype = str(attrs.get("weight_format") or "fp8_e4m3")
    output_dtype = str(attrs.get("output_dtype") or "bf16")
    bias_present = bool(attrs.get("bias"))
    actual_shapes = {
        "x": [m, k], "weight": [k, n],
        "weight_scale": [pairs, n, 2],
        "pertoken_scale": [m, pairs, 2],
        "output": [m, n], "group_sizes": [3],
    }
    if bias_present:
        actual_shapes["bias"] = [n]
    weight_nz = None
    if n % 32 == 0 and k % 16 == 0:
        weight_nz = [n // 32, k // 16, 16, 32]
    descriptors = {
        "x": _descriptor([m, k], act_dtype, "ND", "input"),
        "weight": _descriptor([k, n], weight_dtype, "ND", "input"),
        "weight_scale": _descriptor([pairs, n, 2], "uint8", "ND", "input"),
        "pertoken_scale": _descriptor([m, pairs, 2], "fp8_e8m0", "ND", "input"),
        "output": _descriptor([m, n], output_dtype, "ND", "output"),
        "group_sizes": _descriptor([3], "int64", "ND", "host"),
    }
    descriptors["weight"].update({
        "storage_dtype": weight_dtype, "allocation_shape": [n, k],
        "logical_shape": [k, n], "device_shape": weight_nz,
        "device_layout": "FRACTAL_NZ" if weight_nz else None,
    })
    descriptors["weight_scale"].update({
        "storage_dtype": "uint8", "interpret_dtype": "fp8_e8m0",
        "allocation_shape": [n, (k + group - 1) // group],
        "device_shape": [pairs, n, 2],
    })
    descriptors["pertoken_scale"].update({
        "storage_dtype": "producer_output", "interpret_dtype": "fp8_e8m0",
        "device_shape": [m, pairs, 2],
    })
    if bias_present:
        descriptors["bias"] = _descriptor([n], "fp32", "ND", "input")
    if weight_nz:
        descriptors["weight_nz"] = _descriptor(weight_nz, weight_dtype,
                                                "FRACTAL_NZ", "input")
    producer_inputs = {
        "x_original": _descriptor(
            [m, k], str(attrs.get("prequant_format") or "bf16"), "ND",
            "producer_input"),
    }
    actual_attrs = {
        **attrs, "m": m, "k": k, "n": n,
        "group_sizes": [1, 1, group], "scale_dtype": "fp8_e8m0",
        "pertoken_scale_dtype": "fp8_e8m0",
        "output_dtype": output_dtype,
        "weight_format": weight_dtype, "act_format": act_dtype,
        "linear_method": "mx_fp8", "implementation": "W8A8_MXFP8",
        "weight_transform": "transpose_0_1+maybe_trans_nz",
        "weight_storage_dtype": weight_dtype,
        "weight_allocation_shape": [n, k],
        "weight_logical_shape": [k, n],
        "weight_device_layout": "FRACTAL_NZ" if weight_nz else None,
        "weight_device_shape": weight_nz,
        "weight_scale_storage_dtype": "uint8",
        "weight_scale_interpret_dtype": "fp8_e8m0",
        "weight_scale_allocation_shape": [n, (k + group - 1) // group],
        "weight_scale_device_shape": [pairs, n, 2],
        "bias": bias_present, "offset": None,
    }
    actual = {"dtype": binding.typed.get("dtype"),
              "layout": binding.typed.get("layout"),
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    arguments = {
        "x": "x", "weight": "weight", "weight_scale": "weight_scale",
        "scale_dtype": "fp8_e8m0", "pertoken_scale": "pertoken_scale",
        "pertoken_scale_dtype": "fp8_e8m0",
        "bias": "bias" if bias_present else None,
        "offset": None, "output_dtype": output_dtype,
        "group_sizes": [1, 1, group],
    }
    prepare = {
        "schema": "mindie-sim/qbmm-native-prepare/1",
        "call": "torch_npu.npu_quant_matmul",
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "implementation": "W8A8_MXFP8",
        "source": source["wrapper"],
        "kernel": source["kernel"],
        "source_version": source["version"],
        "source_lines": source["source_lines"],
        "selected_host_variant": source["selected_host_variant"],
        "rejected_host_variants": list(source["rejected_host_variants"]),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "prototype": ("torch_npu.npu_quant_matmul(x1, x2, scale, *, "
                      "offset=None, pertoken_scale=None, bias=None, "
                      "output_dtype=None, x1_dtype=None, x2_dtype=None, "
                      "pertoken_scale_dtype=None, scale_dtype=None, "
                      "group_sizes=None, y_scale=None)"),
        "argument_order": ["x1", "x2", "scale", "offset", "pertoken_scale",
                           "bias", "output_dtype", "x1_dtype", "x2_dtype",
                           "pertoken_scale_dtype", "scale_dtype",
                           "group_sizes", "y_scale"],
        "positional_arguments": {"x1": "x", "x2": "weight",
                                 "scale": "weight_scale"},
        "keywords": {
            "offset": None, "pertoken_scale": "pertoken_scale",
            "bias": "bias" if bias_present else None,
            "output_dtype": output_dtype, "x1_dtype": None, "x2_dtype": None,
            "pertoken_scale_dtype": "fp8_e8m0",
            "scale_dtype": "fp8_e8m0", "group_sizes": [1, 1, group],
            "y_scale": None,
        },
        "optional_slots": {
            "bias": "present" if bias_present else "not_present",
            "offset": "not_passed", "x1_dtype": "not_set",
            "x2_dtype": "not_set", "y_scale": "not_passed",
            "scale_dtype": "fp8_e8m0", "pertoken_scale_dtype": "fp8_e8m0",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "producer_inputs": producer_inputs,
        "weight_nz": weight_nz,
        "weight_storage": {
            "allocation_shape": [n, k], "allocation_dtype": weight_dtype,
            "logical_shape": [k, n], "device_shape": weight_nz,
            "device_layout": "FRACTAL_NZ" if weight_nz else None,
        },
        "scale_storage": {
            "weight_scale": {
                "allocation_shape": [n, (k + group - 1) // group],
                "allocation_dtype": "uint8", "device_shape": [pairs, n, 2],
                "interpret_dtype": "fp8_e8m0",
                "scale_dtype": "fp8_e8m0",
            },
            "pertoken_scale": {
                "device_shape": [m, pairs, 2],
                "interpret_dtype": "fp8_e8m0",
                "storage_dtype": "producer_output",
            },
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "revision": selected_source,
            "selection": "FeatureConfig.metadata.image_id",
            "selected_by": "quant.linear_method=mx_fp8; ordinary linear apply",
            "source_path": source["wrapper"]["path"],
            "source_version": source["version"],
            "source_sha256": source["wrapper"]["sha256"],
            "kernel_sha256": source["kernel"]["sha256"],
        },
        "match": {
            "native_type": "QuantBatchMatmulV3",
            "entry": "torch_npu.npu_quant_matmul",
            "call_id": binding.call_id,
            "deployment_image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "source_revision": selected_source,
            "source_version": source["version"],
            "implementation": "W8A8_MXFP8",
            "selected_host_variant": source["selected_host_variant"],
            "k": k, "n": n, "weight_layout": "FRACTAL_NZ",
            "activation_dtype": act_dtype, "weight_dtype": weight_dtype,
            "scale_dtype": "fp8_e8m0", "output_dtype": output_dtype,
            "group_size": group,
            "source_sha256": source["wrapper"]["sha256"],
            "kernel_sha256": source["kernel"]["sha256"],
            "source_lines": source["source_lines"],
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_nn_binary_name_not_exported_by_python_wrapper",
            "owner_kernel": "QuantBatchMatmulV3",
        },
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration"],
    }
    result = {"actual": validated, "prepare": prepare,
              "prepare_args": {"dtype": validated.get("dtype"),
                               "layout": validated.get("layout"),
                               "shapes": validated.get("shapes"),
                               "attrs": validated.get("attrs")}}
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Independent ABI leaf: torch_npu.npu_quant_matmul (QuantBatchMatmulV3 MXFP8)
# ---------------------------------------------------------------------------

ABI_ENTRY = "torch_npu.npu_quant_matmul"
ABI_NAMESPACE = "torch"
ABI_ATTRS = ()
ABI_OPTIONAL_ATTRS = ("group_size", "output_dtype", "scale_dtype",
                      "pertoken_scale_dtype", "weight_scale_storage_format",
                      "weight_scale_storage_bytes", "scale_group",
                      "weight_layout", "bias")
ABI_NZ_N = 32
ABI_NZ_K = 16
ABI_UNRESOLVED = (
    "aclnn_c_symbol_not_retained",
    "acl_binary_selection_from_host_wrapper_only",
    "mxfp8_internal_accumulator_dtype_not_source_declared",
)
ABI_INT8_UNRESOLVED = (
    "shared_torch_api_source_identity_not_established",
    "weight_nz_layout_source_unresolved",
    "int8_matmul_kernel_binary_selection_unresolved",
)
_E8M0_ALIASES = frozenset({"fp8_e8m0", "e8m0", "ue8m0", "float8_e8m0fnu"})


def abi_contract() -> dict[str, Any]:
    """The independent tensor/attr/version contract for this ABI leaf.

    Two real branches share ``torch_npu.npu_quant_matmul``: the source-pinned
    MXFP8 group-32 wrapper and the shared Torch API INT8 dynamic-quant call.
    Branch selection is by the actual operand dtypes, never by a model label.
    """
    return {
        "schema": ABI_SCHEMA,
        "entry": ABI_ENTRY,
        "namespace": ABI_NAMESPACE,
        "aclnn_symbol": None,
        "version": {"source": ["r64", "r35"], "mxfp8_required": True,
                    "int8_optional": True,
                    "image_prefixes": ["8d422528", "5461a1a8"]},
        "tensors": [
            {"name": "x", "role": "input", "dtype": None,
             "rank": 2, "required": True,
             "views": {"logical": ["M", "K"], "storage": ["M", "K"]}},
            {"name": "weight", "role": "weight", "dtype": None,
             "rank": 2, "required": False,
             "views": {"logical": ["K", "N"], "storage": ["N", "K"]}},
            {"name": "weight_nz", "role": "weight", "dtype": "int8",
             "rank": 4, "required": False, "layout": "FRACTAL_NZ"},
            {"name": "weight_scale", "role": "scale", "dtype": None,
             "rank": None, "required": True},
            {"name": "pertoken_scale", "role": "scale", "dtype": None,
             "rank": None, "required": True},
            {"name": "output", "role": "output", "dtype": "bf16",
             "rank": 2, "required": True},
            {"name": "bias", "role": "input", "dtype": None,
             "rank": 1, "required": False},
        ],
        "variant_slots": {
            "mxfp8_group32": {
                "input_order": ["x", "weight", "weight_scale",
                                "pertoken_scale", "output"],
                "input_dtypes": ["fp8_e4m3", "fp8_e4m3", "uint8",
                                 "fp8_e8m0", "bf16"],
                "input_shapes": ["[M,K]", "[K,N]", "[P,N,2]",
                                 "[M,P,2]", "[M,N]"],
                "bias": "bf16[N]? or fp32[N]?",
                "output_direction": "BF16 [M,N] ND",
            },
            "int8_shared_torch_api_nd": {
                "input_order": ["x", "weight", "weight_scale",
                                "pertoken_scale", "output"],
                "input_dtypes": ["int8", "int8", "bf16", "fp32", "bf16"],
                "input_shapes": ["[M,K]", "[K,N] or [N,K] with direction",
                                 "[N]", "[M]", "[M,N]"],
                "bias": "none",
                "output_direction": "BF16 [M,N] ND",
            },
            "int8_shared_torch_api_nz": {
                "input_order": ["x", "weight_nz", "weight_scale",
                                "pertoken_scale", "output"],
                "input_dtypes": ["int8", "int8", "bf16", "fp32", "bf16"],
                "input_shapes": ["[M,K]", "[N/32,K/16,16,32]", "[N]",
                                 "[M]", "[M,N]"],
                "bias": "none",
                "output_direction": "BF16 [M,N] ND",
            },
        },
        "attrs": ABI_ATTRS,
        "optional_attrs": ABI_OPTIONAL_ATTRS,
        "kernels": [{"name": "QuantBatchMatmulV3", "role": "cube",
                     "variants": ["mxfp8_group32_apply",
                                  "int8_shared_torch_api"]}],
        "relation": "one_to_one",
        "implementations": ["W8A8_MXFP8",
                            "W8A8_INT8_SHARED_TORCH_API"],
        "branches": {
            "mxfp8_group32": {
                "implementation": "W8A8_MXFP8",
                "version_required": True,
                "tensors": ["x:fp8_e4m3", "weight:fp8_e4m3[K,N]",
                            "weight_scale:uint8[P,N,2]",
                            "pertoken_scale:fp8_e8m0[M,P,2]",
                            "output:bf16[M,N]",
                            "bias:bf16[N]? or fp32[N]?"],
                "evidence": "r64/r35 source wrapper + CANN kernel",
            },
            "int8_shared_torch_api": {
                "implementation": "W8A8_INT8_SHARED_TORCH_API",
                "version_required": False,
                "tensors": ["x:int8[M,K]",
                            "weight:int8[K,N] or weight_nz:int8[N/32,K/16,16,32]",
                            "weight_scale:bf16[N]",
                            "pertoken_scale:fp32[M]",
                            "output:bf16[M,N]", "bias:none"],
                "evidence": "shared Torch API slot contract; source "
                            "identity remains unresolved, not R31 WeightNz",
            },
        },
        # T12 binds the source emitted by the selected leaf variant; the
        # two revision records stay separately inspectable here.
        "source": None,
        "sources": {revision: {"wrapper": source["wrapper"],
                               "kernel": source["kernel"],
                               "version": source["version"]}
                    for revision, source in QBMM_SOURCES.items()},
    }


def _abi_version(version: Any, *, required: bool
                 ) -> tuple[str | None, dict, list[str]]:
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
    revision = values.get("source") or values.get("source_revision") \
        or values.get("revision")
    image = digest(values.get("image_id") or values.get("image"))
    if image:
        if image.startswith("8d422528"):
            revision_from_image = "r64"
        elif image.startswith("5461a1a8"):
            revision_from_image = "r35"
        elif not required and image.startswith("9e869e81"):
            revision_from_image = "r31"
        else:
            return None, values, ["version:image_not_r64_or_r35"]
        if revision is not None and revision != revision_from_image:
            return None, values, ["version:source_image_conflict"]
        revision = revision_from_image
    if not required:
        canonical: dict[str, Any] = {}
        if api is not None:
            canonical["api"] = api
        if revision is not None:
            canonical["source"] = revision
        if image:
            canonical["image_id"] = values.get("image_id") or image
        return revision, canonical, []
    if revision is None and api:
        if api.startswith("cann-9.2.1"):
            revision = "r64"
        elif api.startswith("cann-9.2.0"):
            revision = "r35"
    if revision not in QBMM_SOURCES:
        return None, values, ["missing:version.source_or_image"]
    source = QBMM_SOURCES[revision]
    if api and not (api.startswith("cann-9.2.1") and revision == "r64") \
            and not (api.startswith("cann-9.2.0") and revision == "r35"):
        return None, values, ["version:api_source_mismatch"]
    canonical = {"source": revision,
                 "image_id": "sha256:" + source["image_sha"]}
    if api is not None:
        canonical["api"] = api
    return revision, canonical, []


def _raw_tensor_dtype(tensors: Mapping[str, Any], name: str) -> str | None:
    raw = (tensors or {}).get(name)
    if not isinstance(raw, Mapping) or raw.get("present", True) is False:
        return None
    text = str(raw.get("dtype") or "").lower()
    return text or None


def _abi_extents_members(conditions: Mapping[str, Any]
                         ) -> tuple[tuple[str, ...], str]:
    members = tuple(conditions.get("members") or ())
    relation = "many_to_one_fused" if len(members) > 1 else "one_to_one"
    return members, relation


def _scale_attr_reasons(attrs: Mapping[str, Any], *, int8: bool
                        ) -> list[str]:
    reasons: list[str] = []
    if int8:
        if str(attrs.get("scale_dtype", "bf16")).lower() not in {
                "bf16", "bfloat16", "fp32", "float32"}:
            reasons.append("mismatch:attrs.scale_dtype")
        if str(attrs.get("pertoken_scale_dtype", "fp32")).lower() not in {
                "fp32", "float32"}:
            reasons.append("mismatch:attrs.pertoken_scale_dtype")
    else:
        if str(attrs.get("scale_dtype", "fp8_e8m0")).lower() not in _E8M0_ALIASES:
            reasons.append("mismatch:attrs.scale_dtype")
        if str(attrs.get("pertoken_scale_dtype", "fp8_e8m0")).lower() not in _E8M0_ALIASES:
            reasons.append("mismatch:attrs.pertoken_scale_dtype")
    return reasons


def _predict_mxfp8(tensors: Mapping[str, Any], attrs: Mapping[str, Any],
                   version: Any, conditions: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    revision, canonical_version, version_gaps = _abi_version(version, required=False)
    reasons.extend(version_gaps)
    unknown_attrs = set(attrs) - set(ABI_ATTRS) - set(ABI_OPTIONAL_ATTRS)
    if unknown_attrs:
        reasons.append("unmodeled:attrs:" + ",".join(sorted(unknown_attrs)))
    if int_value(attrs.get("group_size", 32)) != 32:
        reasons.append("unmodeled:group_size:" + str(attrs.get("group_size")))
    if str(attrs.get("output_dtype", "bf16")).lower() not in {"bf16", "bfloat16"}:
        reasons.append("mismatch:attrs.output_dtype")
    if int_value(attrs.get("scale_group", 32)) not in {1, 32}:
        reasons.append("mismatch:attrs.scale_group")
    if str(attrs.get("weight_scale_storage_format", "uint8")).lower() not in {
            "uint8", "fp8_e8m0"}:
        reasons.append("mismatch:attrs.weight_scale_storage_format")
    declared_width = int_value(attrs.get("weight_scale_storage_bytes"))
    if declared_width is not None and declared_width != 1:
        reasons.append("mismatch:attrs.weight_scale_storage_bytes")
    if attrs.get("weight_layout") not in (None, "ND", "FRACTAL_NZ"):
        reasons.append("mismatch:attrs.weight_layout")
    reasons.extend(_scale_attr_reasons(attrs, int8=False))
    x, reason = abi_tensor(tensors, "x", role="input", dtype="fp8_e4m3",
                           rank=2, dtypes=("fp8_e4m3", "fp8"))
    reasons.extend(reason)
    weight, reason = abi_tensor(tensors, "weight", role="weight",
                                dtype="fp8_e4m3", rank=2,
                                dtypes=("fp8_e4m3", "fp8"))
    reasons.extend(reason)
    weight_scale, reason = abi_tensor(
        tensors, "weight_scale", role="scale", dtype="uint8", rank=3,
        dtypes=("uint8", "fp8_e8m0"))
    reasons.extend(reason)
    pertoken_scale, reason = abi_tensor(
        tensors, "pertoken_scale", role="scale", dtype="fp8_e8m0", rank=3,
        dtypes=("fp8_e8m0", "uint8"))
    reasons.extend(reason)
    output, reason = abi_tensor(tensors, "output", role="output",
                                dtype="bf16", rank=2,
                                dtypes=("bf16", "bfloat16"))
    reasons.extend(reason)
    bias, reason = abi_tensor(tensors, "bias", role="input", dtype="bf16",
                              rank=1, optional=True,
                              dtypes=("bf16", "bfloat16", "fp32"))
    reasons.extend(reason)
    for name, row, allowed in (
            ("x", x, ("fp8_e4m3", "fp8")),
            ("weight", weight, ("fp8_e4m3", "fp8")),
            ("weight_scale", weight_scale, tuple(_E8M0_ALIASES)),
            ("pertoken_scale", pertoken_scale, tuple(_E8M0_ALIASES)),
            ("output", output, ("bf16", "bfloat16"))):
        if row is not None:
            reasons.extend(interpretation_reasons(row, name, allowed))
    if x is not None and weight is not None and output is not None:
        m, kk = x["shape"]
        k2, nn = weight["shape"]
        # The captured ACLNN slot order declares the weight with its N axis
        # first (``[N, K]``) while the lowered descriptor is ``[K, N]``.  Both
        # spellings describe the same buffer, so the transposed declaration is
        # resolved from the declared activation contraction and output width,
        # never from a captured name or a per-shape table.  The canonical
        # storage view below is ``[N, K]`` for either spelling.
        if (k2 != kk and nn == kk and k2 == int(output["shape"][-1])):
            # Resolve the contraction/width from the declared activation and
            # output extents only.  The captured slot keeps its declared
            # ``[N, K]`` storage spelling, so the ABI echo stays identical to
            # the observation's canonical slots.
            k2, nn = kk, k2
        pairs = ceil_div(ceil_div(kk, 32), 2)
        for name, row, shape in (
                ("weight_scale", weight_scale, [pairs, nn, 2]),
                ("pertoken_scale", pertoken_scale, [m, pairs, 2])):
            device = ((row or {}).get("views") or {}).get("device")
            if device is None:
                continue
            if list(device.get("shape") or ()) != shape:
                reasons.append(f"mismatch:tensors.{name}.device_shape")
            if str(device.get("dtype") or "").lower() not in {
                    "uint8", "fp8_e8m0", "e8m0"}:
                reasons.append(f"mismatch:tensors.{name}.device_dtype")
        if kk % 16 == 0 and nn % 32 == 0:
            reasons.extend(slot_view_reasons(
                weight, "weight",
                logical={"shape": [kk, nn], "dtype": "fp8_e4m3",
                         "layout": "ND", "direction": "k_by_n"},
                storage={"shape": [nn, kk], "dtype": "fp8_e4m3",
                         "layout": "ND"},
                device={"shape": [nn // 32, kk // 16, 16, 32],
                        "dtype": "fp8_e4m3", "layout": "FRACTAL_NZ"}))
            reasons.extend(slot_view_reasons(
                weight_scale, "weight_scale",
                logical={"shape": [pairs, nn, 2], "dtype": "uint8"},
                storage={"shape": [pairs, nn, 2], "dtype": "uint8"},
                device={"shape": [pairs, nn, 2], "dtype": "uint8"}))
            reasons.extend(slot_view_reasons(
                x, "x",
                logical={"shape": [m, kk], "dtype": "fp8_e4m3",
                         "layout": "ND"},
                storage={"shape": [m, kk], "dtype": "fp8_e4m3",
                         "layout": "ND"},
                device={"shape": [m, kk], "dtype": "fp8_e4m3",
                        "layout": "ND"}))
            reasons.extend(slot_view_reasons(
                output, "output",
                logical={"shape": [m, nn], "dtype": "bf16",
                         "layout": "ND"},
                storage={"shape": [m, nn], "dtype": "bf16",
                         "layout": "ND"},
                device={"shape": [m, nn], "dtype": "bf16",
                        "layout": "ND"}))
            reasons.extend(slot_view_reasons(
                pertoken_scale, "pertoken_scale",
                logical={"shape": [m, pairs, 2], "dtype": "fp8_e8m0"},
                storage={"shape": [m, pairs, 2], "dtype": "fp8_e8m0"},
                device={"shape": [m, pairs, 2], "dtype": "fp8_e8m0"}))
    if attrs.get("bias") is True and bias is None:
        reasons.append("missing:tensors.bias")
    if attrs.get("bias") is False and bias is not None:
        reasons.append("mismatch:attrs.bias")
    m = k = n = pairs = None
    if x is not None and weight is not None and output is not None:
        m, k = x["shape"]
        k2, n = weight["shape"]
        if k2 != k:
            reasons.append("mismatch:tensors.weight.contraction")
        if output["shape"] != [m, n]:
            reasons.append("mismatch:tensors.output.shape")
        pairs = ceil_div(ceil_div(k, 32), 2)
        if weight_scale is not None and weight_scale["shape"] != [pairs, n, 2]:
            reasons.append("mismatch:tensors.weight_scale.shape")
        if pertoken_scale is not None and pertoken_scale["shape"] != [m, pairs, 2]:
            reasons.append("mismatch:tensors.pertoken_scale.shape")
        if bias is not None and bias["shape"] != [n]:
            reasons.append("mismatch:tensors.bias.shape")
    current = abi_status(reasons)
    invocation = {"tensors": dict(tensors), "attrs": attrs,
                  "version": canonical_version, "conditions": conditions}
    if current != "predicted":
        return abi_result(
            status_value=current, entry=ABI_ENTRY, reasons=reasons,
            invocation=invocation, selected_call=None, geometry=None,
            work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    weight_nz = ([n // 32, k // 16, 16, 32]
                 if n % 32 == 0 and k % 16 == 0 else None)
    x = with_views(x, storage_shape=[m, k], storage_dtype="fp8_e4m3",
                   storage_layout="ND", interpret_dtype="fp8_e4m3")
    weight = with_views(
        weight, storage_shape=[n, k], storage_dtype="fp8_e4m3",
        storage_layout="ND", interpret_dtype="fp8_e4m3",
        device_shape=weight_nz, device_dtype="fp8_e4m3",
        device_layout="FRACTAL_NZ")
    weight_scale = with_views(
        weight_scale, storage_shape=[n, ceil_div(k, 32)],
        storage_dtype="uint8", storage_layout="ND",
        interpret_dtype="fp8_e8m0", device_shape=[pairs, n, 2])
    pertoken_scale = with_views(
        pertoken_scale, storage_shape=[m, pairs, 2],
        storage_dtype="uint8", storage_layout="ND",
        interpret_dtype="fp8_e8m0", device_shape=[m, pairs, 2])
    output = with_views(output, storage_shape=[m, n],
                        storage_dtype="bf16", storage_layout="ND",
                        interpret_dtype="bf16")
    macs = m * n * k
    issued = align(m, 16) * align(n, 16) * align(k, 32)
    scale_elements = (m + n) * ceil_div(k, 32)
    bias_ops = 0 if bias is None else m * n
    x_bytes = m * k
    weight_bytes = n * k
    weight_scale_bytes = pairs * n * 2
    pertoken_bytes = m * pairs * 2
    output_bytes = m * n * 2
    bias_bytes = 0 if bias is None else n * (4 if bias["dtype"] == "fp32" else 2)
    total_bytes = (x_bytes + weight_bytes + weight_scale_bytes
                   + pertoken_bytes + output_bytes + bias_bytes)
    geometry = {
        "m": m, "n": n, "k": k, "group_size": 32, "pairs": pairs,
        "padded_m": align(m, 16), "padded_n": align(n, 16),
        "padded_k": align(k, 32), "weight_nz": weight_nz,
        "implementation": "W8A8_MXFP8",
        "basis": "declared_tensor_shapes_and_group32_alignment",
    }
    work = {
        "cube": {"macs": macs, "flops": 2 * macs, "issued_macs": issued,
                 "issued_basis": "fractal_16x16_and_group32_padding"},
        "vector": {"ops": bias_ops, "scale_ops": scale_elements,
                   "bias_ops": bias_ops,
                   "basis": "bias_epilogue; mxfp8_scale_application_fused_in_cube"},
        "bytes": {"input": x_bytes, "weight": weight_bytes,
                  "weight_scale": weight_scale_bytes,
                  "pertoken_scale": pertoken_bytes,
                  "output": output_bytes, "bias": bias_bytes,
                  "total": total_bytes,
                  "basis": "call_boundary_tensor_footprint"},
        "call_boundary": {
            "tensor_footprint_bytes": total_bytes,
            "issued_traffic": None,
            "issued_traffic_status": "requires_selected_load_store_plan",
        },
        "scale_elements": scale_elements,
    }
    members, relation = _abi_extents_members(conditions)
    source = QBMM_SOURCES.get(revision) if revision else None
    source_kernel = (source or {}).get("kernel") or {}
    call = abi_selected(
        entry=ABI_ENTRY, namespace=ABI_NAMESPACE, aclnn_symbol=None,
        api_version=canonical_version.get("api"), source_revision=revision,
        image_id=canonical_version.get("image_id"),
        source=(source or {}).get("wrapper"),
        kernels=[{"name": "QuantBatchMatmulV3", "role": "cube",
                  "evidence": ("source_declared" if source
                               else "declared_tensor_abi"),
                  "source_sha256": source_kernel.get("sha256"),
                  "source_status": source_kernel.get("source_status")}],
        execution_core="cube", relation=relation,
        members=members,
        tensors=[x, weight, weight_scale, pertoken_scale, output]
                + ([bias] if bias else []),
        attrs={"group_size": 32, "output_dtype": "bf16",
               "scale_dtype": "fp8_e8m0",
               "pertoken_scale_dtype": "fp8_e8m0",
               "weight_scale_storage_format": "uint8",
               "weight_scale_storage_bytes": 1,
               "scale_group": 32,
               "input_dtypes": ["fp8_e4m3", "fp8_e4m3", "uint8",
                                "fp8_e8m0", "bf16"],
               "input_slots": ["x", "weight", "weight_scale",
                               "pertoken_scale", "output"],
               "offset": "not_present"},
        regions=["linear"],
        variant=((source or {}).get("selected_host_variant")
                 or "mxfp8_group32_declared_abi"))
    call["unresolved"] = list(ABI_UNRESOLVED)
    scale = {
        "weight_scale": {
            "shape": [pairs, n, 2], "dtype": "uint8",
            "interpret_dtype": "fp8_e8m0", "bytes": weight_scale_bytes,
            "allocation_shape": [n, ceil_div(k, 32)],
            "device_shape": [pairs, n, 2],
        },
        "pertoken_scale": {
            "shape": [m, pairs, 2], "dtype": "fp8_e8m0",
            "bytes": pertoken_bytes, "device_shape": [m, pairs, 2],
        },
    }
    metadata = {
        "implementation": "W8A8_MXFP8",
        "group_size": 32, "pairs": pairs, "group_sizes": [1, 1, 32],
        "weight_nz": weight_nz,
        "weight_layout": str(attrs.get("weight_layout") or "FRACTAL_NZ").upper(),
        "weight_allocation_shape": [n, k],
        "weight_logical_shape": [k, n],
        "selected_host_variant": ((source or {}).get("selected_host_variant")
                                  or "mxfp8_group32_declared_abi"),
        "scale_dtype": "fp8_e8m0",
        "pertoken_scale_dtype": "fp8_e8m0",
        "input_dtypes": ["fp8_e4m3", "fp8_e4m3", "uint8",
                         "fp8_e8m0", "bf16"],
        "offset": "not_present",
        "footprint_vs_traffic": "weight/scale bytes are capacity footprint; "
                                "issued traffic requires the load plan",
    }
    accumulator = {
        "dtype": "fp32", "shape": [m, n], "elements": m * n,
        "bytes": m * n * 4, "abi_tensor": False, "on_chip": True,
        "basis": "mxfp8_matmul_internal_accumulator",
        "residency_status": "requires_selected_matrix_program",
        "source_status": "mechanism_inferred_not_source_declared",
    }
    regions = [{
        "id": "linear", "relation": relation, "rows": m, "cols": n,
        "inner": k, "macs": macs, "issued_macs": issued,
        "outputs": ["output"], "members": list(members),
    }]
    return abi_result(
        status_value="predicted", entry=ABI_ENTRY, reasons=[],
        invocation=invocation, selected_call=call, geometry=geometry,
        work=work,
        tensors={"x": x, "weight": weight, "weight_scale": weight_scale,
                 "pertoken_scale": pertoken_scale, "output": output,
                 **({"bias": bias} if bias else {})},
        scale=scale, metadata=metadata, accumulator=accumulator,
        regions=regions)


def _predict_int8(tensors: Mapping[str, Any], attrs: Mapping[str, Any],
                  version: Any, conditions: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    source_revision, canonical_version, version_gaps = _abi_version(
        version, required=False)
    reasons.extend(version_gaps)
    unknown_attrs = set(attrs) - set(ABI_ATTRS) - set(ABI_OPTIONAL_ATTRS)
    if unknown_attrs:
        reasons.append("unmodeled:attrs:" + ",".join(sorted(unknown_attrs)))
    if str(attrs.get("output_dtype", "bf16")).lower() not in {"bf16", "bfloat16"}:
        reasons.append("mismatch:attrs.output_dtype")
    reasons.extend(_scale_attr_reasons(attrs, int8=True))
    if int_value(attrs.get("scale_group", 1)) != 1:
        reasons.append("unmodeled:attrs.scale_group")
    if attrs.get("group_size") is not None:
        reasons.append("unmodeled:attrs.group_size")
    if attrs.get("bias") is not None:
        reasons.append("unmodeled:attrs.bias")
    weight_layout = attrs.get("weight_layout")
    if weight_layout not in (None, "ND", "FRACTAL_NZ"):
        reasons.append("mismatch:attrs.weight_layout")
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
        nz_k_total, nz_n_total = nz_k * ABI_NZ_K, nz_n * ABI_NZ_N
        # A FRACTAL_NZ weight stores whole 16x32 tiles, so a K or N tail is
        # padded *up* to the next tile: a 258-wide K is 17 tiles (272 stored).
        # The storage must cover the logical extent with less than one tile of
        # padding; exact divisibility is not a hardware requirement.
        k_short = (k is not None
                   and (nz_k_total < k or nz_k_total - k >= ABI_NZ_K))
        n_short = (n is not None
                   and (nz_n_total < n or nz_n_total - n >= ABI_NZ_N))
        if ([tile_k, tile_n] != [ABI_NZ_K, ABI_NZ_N]
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
    if n is not None and weight_scale is not None:
        if weight_scale["shape"] != [n]:
            reasons.append("mismatch:tensors.weight_scale.shape")
    if rows is not None and pertoken_scale is not None:
        if pertoken_scale["shape"] != [rows]:
            reasons.append("mismatch:tensors.pertoken_scale.shape")
    # Actual ND path versus pre-materialized NZ path.  Both share the same
    # scale/slot contract; only the selected weight program differs.
    nz_prov = provenance(nz_weight)
    nz_prepared = (nz_prov.get("origin") == "prepared"
                   or nz_prov.get("producer") == "maybe_trans_nz")
    weight_path = "nd"
    if nz_weight is not None:
        weight_path = ("nd_to_nz_materialization" if nz_prepared
                       else "pre_materialized_nz")
        if weight_layout == "ND" and not nz_prepared:
            reasons.append("mismatch:attrs.weight_layout")
    elif weight_layout == "FRACTAL_NZ":
        weight_path = "nd_to_nz_materialization"
    if weight_path == "nd" and logical_direction == "n_by_k":
        # The shared API's ND weight slot is [K,N]; an out-by-in [N,K]
        # input needs the declared transpose/packing producer, not a
        # silently reinterpreted ND slot.
        weight_path = "nd_to_nz_materialization"
    if (weight_path != "nd" and k is not None
            and (k % ABI_NZ_K or (n is not None and n % ABI_NZ_N))):
        reasons.append("mismatch:tensors.weight.nz_tile_divisibility")
    if rows is not None and n is not None and k is not None:
        nz_tile = [ceil_div(n, ABI_NZ_N), ceil_div(k, ABI_NZ_K),
                   ABI_NZ_K, ABI_NZ_N]
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
        if nz_weight is not None:
            reasons.extend(view_reasons(
                nz_weight, "weight_nz", "storage", shape=nz_tile,
                dtype="int8", layout="FRACTAL_NZ"))
            reasons.extend(view_reasons(
                nz_weight, "weight_nz", "device", shape=nz_tile,
                dtype="int8", layout="FRACTAL_NZ"))
        if logical_weight is not None:
            expected = ([n, k] if logical_direction == "n_by_k" else [k, n])
            reasons.extend(view_reasons(
                logical_weight, "weight", "storage", shape=expected,
                dtype="int8"))
            reasons.extend(view_reasons(
                logical_weight, "weight", "device", shape=expected,
                dtype="int8"))
    if rows is not None and n is not None and k is not None:
        nz_tile = [n // ABI_NZ_N, k // ABI_NZ_K, ABI_NZ_K, ABI_NZ_N]
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
    # The call slot is BF16 or FP32 [N].  A storage view here is runtime
    # storage, not checkpoint storage; checkpoint format/width is a separate attr.
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
    nz_tile = [ceil_div(n, ABI_NZ_N), ceil_div(k, ABI_NZ_K),
               ABI_NZ_K, ABI_NZ_N]
    preparation = None
    if weight_path == "nd":
        weight_slot = with_views(
            logical_weight, storage_shape=[k, n], storage_dtype="int8",
            storage_layout="ND", device_shape=[k, n],
            device_dtype="int8", device_layout="ND",
            interpret_dtype="int8")
        weight_slot["name"] = "weight"
        weight_slot["role"] = "weight"
        weight_slot["views"]["logical"] = {
            "shape": [k, n], "dtype": "int8", "layout": "ND",
            "direction": "k_by_n"}
    else:
        input_shape = ([n, k] if logical_direction == "n_by_k"
                       else [k, n])
        if isinstance(nz_prov.get("input"), Mapping):
            input_shape = list(nz_prov["input"].get("shape") or input_shape)
        if nz_weight is not None:
            weight_slot = with_views(
                nz_weight, storage_shape=nz_tile, storage_dtype="int8",
                storage_layout="FRACTAL_NZ", device_shape=nz_tile,
                device_dtype="int8", device_layout="FRACTAL_NZ",
                interpret_dtype="int8")
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
        weight_slot["name"] = "weight_nz"
        weight_slot["role"] = "weight"
        weight_slot["views"]["logical"] = {
            "shape": [k, n], "dtype": "int8", "layout": "ND",
            "direction": "k_by_n"}
        if weight_path == "nd_to_nz_materialization":
            if nz_prepared and isinstance(nz_prov.get("preparation"),
                                          Mapping):
                preparation = dict(nz_prov["preparation"])
                preparation["included_in_call_footprint"] = False
            else:
                preparation = preparation_record(
                    kind=("transpose_0_1+maybe_trans_nz"
                          if logical_direction == "n_by_k"
                          else "maybe_trans_nz"),
                    producer="maybe_trans_nz",
                    input_slot="weight", input_shape=input_shape,
                    output_slot="weight_nz", output_shape=nz_tile)
            weight_slot["provenance"] = {
                "origin": "prepared", "producer": "maybe_trans_nz",
                "completion": "before_native_call",
                "input": dict(preparation["input"]),
                "output": dict(preparation["output"]),
                "preparation_id": preparation["preparation_id"],
            }
        else:
            weight_slot["provenance"] = {
                "origin": "external", "producer": None,
                "completion": "already_materialized"}
    # One normalized call-time scale slot; checkpoint storage is separate.
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
    epilogue = row_column_scale_epilogue(m=rows, n=n, column_dtype=ws_dtype)
    macs = rows * n * k
    issued = align(rows, 16) * align(n, 32) * align(k, 16)
    weight_scale_bytes = n * ws_width
    x_bytes = rows * k
    weight_bytes = n * k
    pertoken_bytes = rows * 4
    output_bytes = rows * n * 2
    total_bytes = (x_bytes + weight_bytes + weight_scale_bytes
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
        "implementation": "W8A8_INT8_SHARED_TORCH_API",
        "basis": "declared_tensor_shapes_and_32x16_nz_alignment",
    }
    members, relation = _abi_extents_members(conditions)
    work = {
        "cube": {"macs": macs, "flops": 2 * macs, "issued_macs": issued,
                 "issued_basis": "16x32x16_nz_padding"},
        "vector": {"ops": epilogue["vector_ops"],
                   "epilogue_ops": epilogue["vector_ops"],
                   "scale_load_elements": rows + n,
                   "basis": "int32_accumulator_to_bf16_row_column_scale"},
        "epilogue": epilogue,
        "bytes": {"input": x_bytes, "weight": weight_bytes,
                  "weight_scale": weight_scale_bytes,
                  "pertoken_scale": pertoken_bytes,
                  "output": output_bytes, "total": total_bytes,
                  "basis": "call_boundary_tensor_footprint"},
        "call_boundary": {
            "tensor_footprint_bytes": total_bytes,
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
    source = {
        "repository": "torch_npu", "path": None, "sha256": None,
        "symbol": ABI_ENTRY, "revision": source_revision,
        "state": "shared_torch_api_source_unresolved",
    }
    variant = {
        "nd": "int8_shared_torch_api_nd",
        "pre_materialized_nz": "int8_shared_torch_api_nz",
        "nd_to_nz_materialization": "int8_shared_torch_api_nz_from_nd",
    }[weight_path]
    call = abi_selected(
        entry=ABI_ENTRY, namespace=ABI_NAMESPACE, aclnn_symbol=None,
        api_version=canonical_version.get("api"),
        source_revision=source_revision,
        image_id=canonical_version.get("image_id"),
        source=source,
        kernels=[{"name": "QuantBatchMatmulV3", "role": "mix",
                  "evidence": "shared_torch_api_abi"}],
        execution_core="mix", relation=relation, members=members,
        tensors=[x, weight_slot, weight_scale,
                 pertoken_scale, output],
        attrs={"weight_scale_format": ws_dtype,
               "output_dtype": "bf16", "scale_group": 1,
               "scale_dtype": ws_dtype,
               "pertoken_scale_dtype": "fp32",
               "weight_path": weight_path,
               "epilogue_operations_per_output": 4,
               "input_dtypes": ["int8", "int8", ws_dtype, "fp32", "bf16"],
               "input_slots": ["x", ("weight" if weight_path == "nd"
                                    else "weight_nz"), "weight_scale",
                               "pertoken_scale", "output"],
               "offset": "not_present"},
        regions=["linear"], variant=variant)
    call["unresolved"] = list(ABI_INT8_UNRESOLVED)
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
    metadata = {
        "implementation": "W8A8_INT8_SHARED_TORCH_API",
        "weight_nz": nz_tile, "weight_path": weight_path,
        "weight_device_layout": (
            "FRACTAL_NZ" if weight_path != "nd" else "ND"),
        "weight_logical_shape": [k, n],
        "weight_logical_direction": "k_by_n",
        "weight_allocation_shape": None,
        "weight_allocation_status": (
            "shared_torch_api_source_unresolved" if preparation is None
            else "materialized_by_declared_producer"),
        "weight_packer": ("shared_torch_api_or_upstream_unresolved"
                          if preparation is None else "maybe_trans_nz"),
        "weight_scale_storage_shape": [n],
        "weight_scale_storage_format": checkpoint_format,
        "weight_scale_storage_bytes_total": checkpoint_bytes,
        "weight_scale_shape": [n],
        "weight_scale_format": ws_dtype,
        "weight_scale_bytes": weight_scale_bytes,
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
                                "footprint; the native call reads BF16 scale "
                                "and issued traffic requires the load plan",
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
        "id": "linear", "relation": relation, "rows": rows, "cols": n,
        "inner": k, "macs": macs, "issued_macs": issued,
        "outputs": ["output"], "members": list(members),
    }]
    result_tensors = {"x": x, "weight_scale": weight_scale,
                      "pertoken_scale": pertoken_scale, "output": output}
    if weight_path == "nd":
        result_tensors["weight"] = weight_slot
    else:
        result_tensors["weight"] = logical_weight or {
            "name": "weight", "role": "weight", "dtype": "int8",
            "shape": [k, n], "layout": "ND", "optional": False,
            "present": True, "direction": "k_by_n",
            "views": {"logical": {"shape": [k, n], "dtype": "int8",
                                  "layout": "ND",
                                  "direction": "k_by_n"},
                      "storage": None, "device": None,
                      "interpretation": {"dtype": "int8", "layout": "ND"}},
        }
        result_tensors["weight_nz"] = weight_slot
    return abi_result(
        status_value="predicted", entry=ABI_ENTRY, reasons=[],
        invocation=invocation, selected_call=call, geometry=geometry,
        work=work, tensors=result_tensors,
        scale=scale, metadata=metadata, accumulator=accumulator,
        regions=regions)


def predict_abi(tensors: Mapping[str, Any], *,
                attrs: Mapping[str, Any] | None = None,
                version: Any = None,
                conditions: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
    """Map either real QBMM branch from independent typed inputs.

    MXFP8 requires the source r64/r35 identity; the shared Torch API INT8
    branch preserves an optional source/image but does not claim an R31
    WeightNz identity.  Neither branch reads a target duration or a shape
    table.
    """
    attrs = dict(attrs or {})
    conditions = dict(conditions or {})
    dtype = _raw_tensor_dtype(tensors, "x")
    if dtype == "int8":
        return _predict_int8(tensors, attrs, version, conditions)
    if dtype in {"fp8_e4m3", "fp8"}:
        return _predict_mxfp8(tensors, attrs, version, conditions)
    reasons = ["missing:tensors.x"] if dtype is None else [
        f"mismatch:tensors.x.dtype:{dtype}"]
    return abi_result(
        status_value=abi_status(reasons), entry=ABI_ENTRY, reasons=reasons,
        invocation={"tensors": dict(tensors), "attrs": attrs,
                    "version": {}, "conditions": conditions},
        selected_call=None, geometry=None, work=None, tensors={},
        scale={}, metadata={}, accumulator=None, regions=[])


def operator_plan(record, *, tile=(16, 32, 16), slots=2):
    """T12/T10 consumer entry for the selected real OperatorPlan."""
    from .native_matrix_abi import build_operator_plan
    return build_operator_plan(record, tile=tile, slots=slots)
