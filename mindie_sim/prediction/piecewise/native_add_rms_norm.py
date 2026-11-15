"""Prepare/execute native AddRmsNorm call for the deployed R31 Qwen3.6 path.

[Role] Bind the real ``aclnnAddRmsNorm``/``AddRmsNorm`` call recorded by the
  original R31 rank capture to the normal N0 fused residual+norm occurrence
  and to the terminal O0 normalization.  Expand only the source-recorded
  tensor roles/geometry; do not re-price the existing fused operator cost.
[Invariants] The image is the exact R31 deployment digest, the kernel is the
  captured ``AddRmsNorm`` on AI_VECTOR_CORE with BF16 x/residual/weight and
  BF16/F32/BF16 outputs.  The local width and SP rows come from the compiled
  theta; captured sample dimensions stay in the evidence block only.  The
  initial plain RmsNorm (no residual sibling) is the disjoint OPRMS2 leaf; it
  and every other image, bias/quant norm, or unsupported geometry return
  ``None``.  The prepare is explicitly partial: the Python caller module and
  ACLNN positional slots were not retained, so no complete native execution is
  claimed.
[Timeouts & Locks] Bounded memo of 32 occurrence descriptors; no waits/locks.
[Side-Effect Policy] Pure metadata; no tensor allocation, device import or I/O.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError

R31_IMAGE_ID = "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_IMAGE_TAG = (
    "swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:"
    "nightly-ci-releases-v0.30.0-a5-cann9.2.0_20261022115200-20261023-r31"
)
CANN_REVISION = "cann-9.2.0-r31"
CANN_LIBOPAPI = {
    "repository": "cann",
    "path": "/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": "AddRmsNorm",
    "revision": CANN_REVISION,
    "state": "task_artifact",
}
CAPTURE = {
    "path": "obs-r4/qwen-real-norm-order.json",
    "sha256": "7d1f93bcd26ed8ffb7a7eb925d6c58a010a6e20ad7527c34ed3e665c5a04b5ff",
    "raw_path": (
        "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
        "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
        "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz"
    ),
    "raw_sha256": "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a",
    "sequence_sha256": "ecddce9e05631a21e623c9654269bee2a2b8ed95f640bc24e19e8f07eb420b29",
    "cell": "smoke_qwen36_single_pf_4k_c1",
    "raw_row_index": 76,
    "raw_row_index_base": "1-based_csv_record",
    "raw_row_index_0based": 75,
    "name": "aclnnAddRmsNorm_AddRmsNorm_AddRmsNorm",
    "type": "AddRmsNorm",
    "accelerator_core": "AI_VECTOR_CORE",
    "block_num": 64,
    "mix_block_num": 0,
    "input_shapes": "2048,5120;2048,5120;5120",
    "input_dtypes": "DT_BF16;DT_BF16;DT_BF16",
    "input_formats": "ND;ND;ND",
    "output_shapes": "2048,5120;2048,1;2048,5120",
    "output_dtypes": "DT_BF16;FLOAT;DT_BF16",
    "output_formats": "ND;ND;ND",
    "captured_rows": 2048,
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}
_MAX_CACHE = 32
_ENTRY = "aclnnAddRmsNorm"
_KERNEL = "AddRmsNorm"


def _feature(compiled: Any) -> dict:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _theta(compiled: Any) -> Any:
    return getattr(getattr(compiled, "operator", None), "theta", None)


def _find(compiled: Any, key: str, default: Any = None) -> Any:
    theta = _theta(compiled)
    if theta is None or not hasattr(theta, "find"):
        return default
    return theta.find(key, default)


def _siblings(compiled: Any) -> set[str]:
    return {str(getattr(item, "name", "")) for item in
            (getattr(compiled, "siblings", ()) or ())}


def _norm_gate(compiled: Any) -> bool:
    """Weighted BF16 RMS norm on R31; no bias, quant, Gemma shift or LayerNorm."""
    theta = _theta(compiled)
    if theta is None or not hasattr(theta, "find"):
        return False
    if str(getattr(getattr(compiled, "operator", None), "family", "")) != "elementwise":
        return False
    if int(_find(compiled, "layernorm", 0) or 0):
        return False
    if float(_find(compiled, "norm_weight_offset", 0.0) or 0.0) != 0.0:
        return False
    if str(_find(compiled, "norm_arch", "") or "") not in {"", "rmsnorm"}:
        return False
    if bool(_find(compiled, "norm_has_bias", False)):
        return False
    if str(_find(compiled, "native_norm_kind", "") or "") in {
            "AddRmsNormBias", "AddRmsNormDynamicQuant",
            "AddRmsNormDynamicMxQuant"}:
        return False
    if bool(int(_find(compiled, "quantized", 0) or 0)):
        return False
    if int(_find(compiled, "quant_passes", 0) or 0):
        return False
    if str(_find(compiled, "output_format", "") or "") not in {"", "bf16"}:
        return False
    hidden = int(_find(compiled, "elements", 0) or 0)
    gamma_bytes = float(_find(compiled, "gamma_bytes", 0.0) or 0.0)
    gamma_format = str(_find(compiled, "gamma_format", "") or "")
    if hidden < 1 or gamma_format != "bf16":
        return False
    if gamma_bytes <= 0 or abs(gamma_bytes - hidden * 2) > 1e-6:
        return False
    sequence_rows = str(_find(compiled, "sequence_rows", "") or "")
    if sequence_rows not in {"", "shard", "selected"}:
        return False
    if int(_find(compiled, "sequence_ranks", 1) or 1) < 1:
        return False
    return True


def _select_kind(compiled: Any) -> str | None:
    kind = str(getattr(compiled, "kind", "") or "")
    name = str(getattr(getattr(compiled, "operator", None), "name", "") or "")
    if kind == "layernorm" and name == "rmsnorm":
        if "residual_add" not in _siblings(compiled):
            return None  # the initial plain RmsNorm is OPRMS, not this call
        return "inner" if _norm_gate(compiled) else None
    if kind == "lm_head_sampler" and name == "final_norm":
        if not _norm_gate(compiled):
            return None
        if int(_find(compiled, "residual_add", 0) or 0) != 1:
            return None
        if str(_find(compiled, "fusion", "") or "") != "add_rmsnorm":
            return None
        if float(_find(compiled, "rstd_bytes", 0.0) or 0.0) <= 0:
            return None
        if float(_find(compiled, "residual_out_bytes", 0.0) or 0.0) <= 0:
            return None
        return "terminal"
    return None


def _r31_source(compiled: Any) -> bool:
    """Whether this compiled operator is the retained R31 AddRmsNorm source.

    An operator that declares *no* container identity is a general shape and
    is admitted on its declared ABI (the same call, kernels and slots); an
    explicitly declared different stack (r35/r64/other) stays unbound.  This
    removes the hardcoded-image wall without inventing an image identity.
    """
    from .native_source_pins import stack_of_metadata
    meta = _feature(compiled)
    stack = stack_of_metadata(meta)
    if stack is None:
        image = str(meta.get("image") or "")
        return not image or "r31" in image or "cann9.2.0" in image
    return stack == "r31"


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real ``_CallSpec`` for R31 N0/O0 AddRmsNorm, else ``None``."""
    if not _r31_source(compiled):
        return None
    kind = _select_kind(compiled)
    if kind is None:
        return None
    from .native_calls import _api, _call_spec, _kernel, _source

    hidden = int(_find(compiled, "elements", 0) or 0)
    if hidden < 1:
        return None
    ranks = int(_find(compiled, "sequence_ranks", 1) or 1)
    sequence_rows = str(_find(compiled, "sequence_rows", "") or "")
    source = _source(
        CANN_LIBOPAPI["repository"], CANN_LIBOPAPI["path"],
        sha256=CANN_LIBOPAPI["sha256"], symbol=CANN_LIBOPAPI["symbol"],
        revision=CANN_LIBOPAPI["revision"], state=CANN_LIBOPAPI["state"])
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "shapes": {
            "x": [hidden], "residual": [hidden], "gamma": [hidden],
            "output": [hidden], "rstd": [1], "residual_out": [hidden],
        },
        "attrs": {
            "hidden": hidden,
            "sequence_rows": sequence_rows,
            "sequence_ranks": ranks,
            "input_layout": "ND",
            "rstd_dtype": "fp32",
            "output_dtype": "bf16",
            "residual_dtype": "bf16",
            "call_name": _ENTRY,
            "kernel_name": _KERNEL,
        },
        "rank_width": 2,
    }
    kernels = (_kernel(
        _KERNEL, "vector", path=CANN_LIBOPAPI["path"],
        sha256=CANN_LIBOPAPI["sha256"], evidence="runtime_binary"),)
    owner = ("fused:add_rmsnorm" if kind == "inner"
             else "O0:elementwise:final_norm")
    members = ("residual_add", "rmsnorm") if kind == "inner" else ()
    return _call_spec(
        _api("aclnn", _ENTRY, aclnn_symbol=_ENTRY, version="cann-9.2.0"),
        source, typed, kernels, members, owner, execution_core="aiv",
        unresolved=[
            "python_caller_module_not_retained",
            "capture_python_stack_absent",
            "kernel_implementation_source_retained_as_runtime_binary",
            "aclnn_positional_slots_not_retained",
        ])


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _rows(compiled: Any, geometry: Any, sequence_rows: str, ranks: int) -> int:
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    if tokens < 1:
        return 0
    if sequence_rows == "selected":
        selected = getattr(geometry, "producing_lm_head_rows", None)
        if not callable(selected):
            return 0
        return int(selected() or 0)
    if sequence_rows == "shard" and ranks > 1:
        if int(getattr(geometry, "operand_tokens", 0) or 0) > 0:
            return tokens
        return int(math.ceil(tokens / ranks))
    return tokens


def _eps(compiled: Any) -> float | None:
    value = _find(compiled, "norm_eps", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R31 AddRmsNorm call."""
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != _ENTRY:
        return None
    source = dict(getattr(binding, "source", {}) or {})
    if (source.get("sha256") != CANN_LIBOPAPI["sha256"]
            or str(source.get("revision") or "") != CANN_REVISION):
        return None
    attrs = dict((getattr(binding, "typed", {}) or {}).get("attrs") or {})
    hidden = int(attrs.get("hidden", 0) or 0)
    if hidden < 1:
        return None
    selected_kind = _select_kind(compiled)
    variant = "terminal_norm" if selected_kind == "terminal" else "fused_attention_norm"
    members = ([] if variant == "terminal_norm"
               else ["residual_add", "rmsnorm"])
    sequence_rows = str(attrs.get("sequence_rows", "") or "")
    ranks = int(attrs.get("sequence_ranks", 1) or 1)
    rows = _rows(compiled, geometry, sequence_rows, ranks)
    if rows < 1:
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    eps = _eps(compiled)
    image = str(_feature(compiled).get("image") or R31_IMAGE_TAG)
    cache_key = (str(getattr(binding, "call_id", "")), rows, tokens, phase,
                 sequence_rows, eps, image)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "x": [rows, hidden],
        "residual": [rows, hidden],
        "gamma": [hidden],
        "output": [rows, hidden],
        "rstd": [rows, 1],
        "residual_out": [rows, hidden],
    }
    actual_attrs = {
        **attrs,
        "rows": rows,
        "tokens": tokens,
        "phase": phase,
        "rstd_bytes": rows * 4,
        "residual_bytes": rows * hidden * 2,
        "norm_output_bytes": rows * hidden * 2,
        "physical_rows": rows,
        "sp_rows": sequence_rows or "all",
        "sp_ranks": ranks,
        "row_basis": sequence_rows or "all",
    }
    if eps is not None:
        actual_attrs["eps"] = eps
        actual_attrs["eps_source"] = "model_config_declared"
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, hidden], "bf16", "ND", "input"),
        "residual": _descriptor([rows, hidden], "bf16", "ND", "input"),
        "gamma": _descriptor([hidden], "bf16", "ND", "weight"),
        "output": _descriptor([rows, hidden], "bf16", "ND", "output"),
        "rstd": _descriptor([rows, 1], "fp32", "ND", "output"),
        "residual_out": _descriptor([rows, hidden], "bf16", "ND", "output"),
    }
    prepare = {
        "schema": "mindie-sim/add-rms-norm-native-prepare/1",
        "call": _ENTRY,
        "call_id": getattr(binding, "call_id", None),
        "signature_hash": (getattr(binding, "typed", {}) or {}).get("signature_hash"),
        "source": source,
        "variant": variant,
        "owner": ("O0:elementwise:final_norm" if variant == "terminal_norm"
                  else "fused:add_rmsnorm"),
        "members": members,
        "kernel": (dict(getattr(binding, "kernels", ())[0])
                   if getattr(binding, "kernels", ()) else
                   {"name": _KERNEL, "role": "vector",
                    "evidence": "runtime_binary"}),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "acl_prepare_only",
            "python_caller_module_not_retained",
            "capture_python_stack_absent",
            "aclnn_positional_slots_not_retained",
        ],
        "shapes": actual_shapes,
        "validated_shapes": validated.get("shapes"),
        "attrs": actual_attrs,
        "descriptors": descriptors,
        "arguments": {
            "input": "x",
            "residual_input": "residual",
            "weight": "gamma",
            "epsilon": eps,
            "norm_output": "output",
            "rstd": "rstd",
            "residual_output": "residual_out",
        },
        "argument_order_status": (
            "tensor_roles_and_dtypes_from_captured_aclnn_call; "
            "positional_slots_not_retained"),
        "outputs": {
            "norm_output": descriptors["output"],
            "rstd": descriptors["rstd"],
            "residual": descriptors["residual_out"],
        },
        "output_count": 3,
        "aliases": {},
        "alias_status": "not_established",
        "capture": dict(CAPTURE),
        "deployment": {
            "image_id": R31_IMAGE_ID,
            "image": image,
            "revision": "r31/9e869e8146",
            "source_sha256": CANN_LIBOPAPI["sha256"],
            "capture_raw_sha256": CAPTURE["raw_sha256"],
            "capture_sequence_sha256": CAPTURE["sequence_sha256"],
        },
        "match": {
            "native_type": _KERNEL,
            "entry": _ENTRY,
            "call_id": getattr(binding, "call_id", None),
            "image_id": R31_IMAGE_ID,
            "source_revision": CANN_REVISION,
            "source_sha256": CANN_LIBOPAPI["sha256"],
            "hidden": hidden,
            "rows": rows,
            "tokens": tokens,
            "row_basis": sequence_rows or "all",
            "dtype": "bf16",
            "gamma_dtype": "bf16",
            "rstd_dtype": "fp32",
            "residual_dtype": "bf16",
            "status": "partial_prepare_not_executed",
        },
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "python_caller_module_not_retained",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": actual["dtype"],
            "layout": actual["layout"],
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= _MAX_CACHE:
        _DYNAMIC_CACHE.pop(next(iter(_DYNAMIC_CACHE)))
    _DYNAMIC_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Shared CPU execution and common resource-stage prediction for native
# Vector/elementwise ABI leaves.
#
# The siblings in this task (RMSNorm, SwiGLU, DynamicQuantV2, Sigmoid,
# in-place copy/zero, Index, AddRmsNormBias, gated norm) import these helpers
# so that every leaf validates against the same prepared descriptors, rounds
# BF16 the same way, and is priced by the same ``CompiledOperator`` elementwise
# / copy family rather than a private timing table.
# ---------------------------------------------------------------------------
EXECUTION_SCHEMA = "mindie-sim/native-vector-execution/1"
PREDICTION_SCHEMA = "mindie-sim/native-vector-prediction/1"
CHAIN_SCHEMA = "mindie-sim/native-vector-chain/1"
A5_DEFAULT_SKU = "Ascend950DT_9582"


class NativeVectorExecutionUnavailable(RuntimeError):
    """The real CPU invocation cannot be formed from the prepared contract."""


def _numpy():
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - numpy is a test extra
        raise NativeVectorExecutionUnavailable("numpy_unavailable") from error
    return np


def _bf16_round(array: Any) -> Any:
    """Round finite FP32 values to exactly representable BF16 values."""
    np = _numpy()
    value = np.ascontiguousarray(array, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _cast_value(array: Any, dtype: str | None) -> Any:
    np = _numpy()
    if dtype in (None, "", "none"):
        return np.asarray(array)
    if dtype == "bf16":
        return _bf16_round(array)
    if dtype == "fp16":
        return np.asarray(array, dtype=np.float16)
    if dtype == "fp32":
        return np.asarray(array, dtype=np.float32)
    if dtype == "int8":
        return np.clip(np.asarray(array), -128, 127).astype(np.int8)
    if dtype == "int32":
        return np.asarray(array, dtype=np.int32)
    if dtype == "int64":
        return np.asarray(array, dtype=np.int64)
    if dtype == "uint8":
        return np.asarray(array, dtype=np.uint8)
    raise NativeVectorExecutionUnavailable(f"dtype_unsupported:{dtype}")


def _prepare_record(prepared: Any) -> dict[str, Any]:
    if not isinstance(prepared, Mapping):
        raise NativeVectorExecutionUnavailable("prepared_record_required")
    record = prepared.get("prepare") if "prepare" in prepared else prepared
    if not isinstance(record, Mapping) or not record.get("descriptors"):
        raise NativeVectorExecutionUnavailable("prepared_descriptors_missing")
    return dict(record)


def _prepared_descriptor(prepared: Any, name: str) -> dict[str, Any]:
    record = _prepare_record(prepared)
    descriptors = record.get("descriptors") or {}
    item = descriptors.get(name)
    if not isinstance(item, Mapping):
        raise NativeVectorExecutionUnavailable(f"descriptor_missing:{name}")
    return dict(item)


def _input_array(prepared: Any, values: Mapping[str, Any], name: str) -> Any:
    np = _numpy()
    if name not in values:
        raise NativeVectorExecutionUnavailable(f"input_missing:{name}")
    item = _prepared_descriptor(prepared, name)
    array = np.asarray(values[name])
    expected = item.get("shape")
    storage = item.get("storage_shape")
    allowed = []
    if expected:
        allowed.append(tuple(int(dim) for dim in expected))
    if storage:
        allowed.append(tuple(int(dim) for dim in storage))
    if allowed and tuple(array.shape) not in allowed:
        raise NativeVectorExecutionUnavailable(
            f"input_shape_mismatch:{name}:{list(array.shape)}!={allowed[0]}")
    return _cast_value(array, str(item.get("dtype") or ""))


def _finish_execution(prepared: Any, outputs: Mapping[str, Any], *,
                      aliases: Mapping[str, str] | None = None,
                      in_place: tuple[str, ...] = (),
                      materialized: Mapping[str, bool] | None = None,
                      out: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record = _prepare_record(prepared)
    descriptors = record.get("descriptors") or {}
    out = out or {}
    materialized = dict(materialized or {})
    result: dict[str, Any] = {}
    cast_cache: dict[int, Any] = {}
    dtypes: dict[str, str] = {}
    layouts: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    value_dtypes: dict[str, str] = {}
    for name, item in descriptors.items():
        if item.get("role") not in {"output", "inout", "inplace_input_output"}:
            continue
        if item.get("present", True) is False:
            continue
        if name not in outputs:
            raise NativeVectorExecutionUnavailable(f"output_missing:{name}")
        raw = outputs[name]
        cache_key = id(raw)
        if cache_key in cast_cache:
            value = cast_cache[cache_key]
        elif (name in in_place
              and _numpy().asarray(raw).dtype == _numpy().float32
              and str(item.get("dtype")) == "bf16"):
            # An in-place BF16-zero write is exactly representable; preserving
            # the caller's array identity is the actual alias contract.
            value = _numpy().asarray(raw)
            cast_cache[cache_key] = value
        else:
            value = _cast_value(raw, item.get("dtype"))
            cast_cache[cache_key] = value
        expected = item.get("shape")
        if expected and list(value.shape) != [int(dim) for dim in expected]:
            raise NativeVectorExecutionUnavailable(
                f"output_shape_mismatch:{name}:{list(value.shape)}!={list(expected)}")
        if name in out:
            if materialized.get(name, True) is False:
                raise NativeVectorExecutionUnavailable(
                    f"out_not_materialized:{name}")
            target = _numpy().asarray(out[name])
            if target.shape != value.shape:
                raise NativeVectorExecutionUnavailable(
                    f"out_shape_mismatch:{name}:{list(target.shape)}!={list(value.shape)}")
            if target.dtype != _numpy().asarray(value).dtype:
                raise NativeVectorExecutionUnavailable(
                    f"out_dtype_mismatch:{name}:{target.dtype}!="
                    f"{_numpy().asarray(value).dtype}")
            _numpy().copyto(target, value)
            value = target
            cast_cache[cache_key] = value
        result[name] = value
        dtypes[name] = str(item.get("dtype") or "")
        layouts[name] = str(item.get("layout") or "ND")
        shapes[name] = [int(dim) for dim in value.shape]
        value_dtypes[name] = str(_numpy().asarray(value).dtype)
        materialized.setdefault(name, True)
    if not result:
        raise NativeVectorExecutionUnavailable("prepared_has_no_outputs")
    return {
        "schema": EXECUTION_SCHEMA,
        "call": record.get("call") or record.get("native_entry"),
        "call_id": record.get("call_id"),
        "native_type": record.get("native_type")
        or (record.get("kernel") or {}).get("name"),
        "stage": "cpu_reference_executed",
        "runtime_executed": True,
        "device_executed": False,
        "outputs": result,
        "output_order": list(result),
        "dtypes": dtypes,
        "value_dtypes": value_dtypes,
        "layouts": layouts,
        "shapes": shapes,
        "materialized": materialized,
        "aliases": dict(aliases or {}),
        "in_place": list(in_place),
        "non_claims": (
            "cpu_reference_not_device_execution",
            "no_runtime_allocation",
            "no_measured_duration",
            "rounding_order_documented_by_leaf",
        ),
    }


def _leaf_supply(hardware: Any = None, supply: Any = None,
                 dtype: str = "bf16") -> tuple[Any, str, Any]:
    """Resolve the declared supply; never substitute another architecture.

    The same row-elementwise program is evaluated on the target die's declared
    rates and invocation floors.  A caller that names no hardware keeps the
    A5 default; a declared A3/A2 supply is used as declared (its own vector
    floor and buffer geometry), not replaced by the A5 profile.
    """
    from .compiler import PipeSupply
    if supply is not None:
        if not isinstance(supply, PipeSupply):
            raise ModelingError("native_vector_supply_invalid")
        arch = str(supply.arch_family or "")
        return supply, arch or "declared", None
    from mindie_sim.schema.hardware import Hardware
    if hardware is None:
        hardware = A5_DEFAULT_SKU
    if isinstance(hardware, str):
        import importlib
        ascend = importlib.import_module("mindie_sim.backends.ascend")
        machine = Hardware.from_mapping(ascend.materialize({"sku": hardware}))
        label = hardware
    elif isinstance(hardware, Hardware):
        machine, label = hardware, str(hardware.arch_family or "")
    else:
        raise ModelingError("native_vector_hardware_invalid")
    arch = str(getattr(machine, "arch_family", "") or "")
    return PipeSupply.from_hardware(machine, dtype=dtype), arch, label


def _leaf_geometry(rows: int, phase: str, *, selected: int | None = None) -> Any:
    from .geometry import RequestGeometry, WorkloadGeometry
    rows = int(rows)
    if rows < 1:
        raise ModelingError("native_vector_rows_invalid")
    if selected is None:
        request = RequestGeometry(rows, 0, phase=phase)
    else:
        selected = int(selected)
        if selected < 1 or selected > rows:
            raise ModelingError("native_vector_selected_rows_invalid")
        request = RequestGeometry(rows, 0, phase=phase,
                                  query_ranges=((0, selected),))
    return WorkloadGeometry((request,), phase=phase, cohort_batch=1,
                            local_batch=1)


def _leaf_cost(prepared: Any, *, family: str, name: str, role: str,
               rows: int, elements: int, ops: int, read_bytes: float,
               write_bytes: float, extras: tuple = (), kind: str = "layernorm",
               phase: str = "prefill", selected: int | None = None,
               hardware: Any = None, supply: Any = None,
               policy: Any = None, act_bytes: float = 2.0) -> tuple[Any, Any, str]:
    from .compiler import CompiledOperator, ImplementationPolicy
    from mindie_sim.schema.subgraphs import (
        BoundOperator, OperatorTemplate, StaticSignature)
    supply_obj, arch, _label = _leaf_supply(hardware, supply)
    policy_obj = policy or ImplementationPolicy(cores=64)
    signature = StaticSignature(
        role, matrix=(("elements", int(elements)),),
        quant=(("act_bytes", float(act_bytes)),),
        extras=(("ops_per_element", int(ops)),
                ("read_bytes", float(read_bytes)),
                ("write_bytes", float(write_bytes)), *tuple(extras)))
    bound = BoundOperator(OperatorTemplate(str(name), str(family), str(role)),
                          signature)
    geometry = _leaf_geometry(rows, phase, selected=selected)
    cost = CompiledOperator(bound, supply_obj, policy_obj,
                            kind=str(kind)).evaluate(geometry)
    return cost, supply_obj, arch


def _prediction_record(cost: Any, *, call: Any, rows: int, family: str,
                       kind: str, program: Mapping[str, Any], arch: str,
                       phase: str, extra: Mapping[str, Any] | None = None,
                       logical_materialization: Mapping[str, Any] | None = None
                       ) -> dict:
    materialization = tuple(cost.materialization)
    derived = dict(cost.derived)
    if logical_materialization is not None:
        materialization = tuple(sorted(
            (str(key), value) for key, value in
            logical_materialization.items()))
        logical_total = sum(int(value) for _, value in materialization)
        derived["kept_bytes"] = logical_total
        derived["logical_effective_bytes"] = logical_total
    return {
        "schema": PREDICTION_SCHEMA,
        "call": call,
        "rows": int(rows),
        "family": str(family),
        "kind": str(kind),
        "phase": str(phase),
        "duration_ms": str(cost.duration_ms),
        "duration_us": float(cost.duration_ms) * 1000.0,
        "dominant_pipe": cost.dominant_pipe,
        "tied_pipes": list(cost.tied_pipes),
        "work": {key: str(value) for key, value in cost.work},
        "times_ms": {key: str(value) for key, value in cost.times},
        "stage_times_ms": [str(value) for value in cost.stage_times],
        "materialization": {key: str(value)
                            for key, value in materialization},
        "stage_work": [
            {pipe: str(value) for pipe, value in (
                stage.items() if hasattr(stage, "items") else stage)}
            for stage in cost.stage_work
        ],
        "native_program_available": bool(
            getattr(cost, "native_program", None) is not None),
        "native_schedule_available": bool(
            getattr(cost, "native_schedule", None) is not None),
        "derived": {key: str(value) for key, value in derived.items()},
        "formula": cost.formula,
        "fidelity": cost.fidelity,
        "program": dict(program),
        "environment": {
            "basis": "common_resource_stage",
            "arch": str(arch),
            "phase": str(phase),
            "rows": int(rows),
        },
        "non_claims": (
            "static_cpu_prediction",
            "no_hardware_dispatch_claim",
            "no_measured_duration_claim",
            "no_target_duration_injected",
            "no_algorithm_fallback",
        ),
        **(dict(extra) if extra else {}),
    }


def _single_leaf_prediction(prepared: Any, *, family: str, name: str,
                            role: str, rows: int, elements: int, ops: int,
                            read_bytes: float, write_bytes: float,
                            extras: tuple = (), kind: str = "layernorm",
                            selected: int | None = None, phase: str = "prefill",
                            hardware: Any = None, supply: Any = None,
                            policy: Any = None,
                            act_bytes: float = 2.0,
                            logical_materialization: Mapping[str, Any] | None = None
                            ) -> dict:
    cost, _supply, arch = _leaf_cost(
        prepared, family=family, name=name, role=role, rows=rows,
        elements=elements, ops=ops, read_bytes=read_bytes,
        write_bytes=write_bytes, extras=extras, kind=kind, phase=phase,
        selected=selected, hardware=hardware, supply=supply, policy=policy,
        act_bytes=act_bytes)
    record = _prepare_record(prepared)
    return _prediction_record(
        cost, call=record.get("call") or record.get("native_entry"),
        rows=rows, family=family, kind=kind, phase=phase,
        logical_materialization=logical_materialization,
        arch=arch, program={
            "common_program": f"{family}.{kind}.{name}",
            "members": [name],
            "family": family,
            "elements": int(elements),
            "ops_per_element": int(ops),
            "read_bytes_per_row": float(read_bytes),
            "write_bytes_per_row": float(write_bytes),
            "extras": list(extras),
        })


# ---- AddRmsNorm: independent prepare/execute/predict and real small chain ----

def prepare_add_rms_norm(*, rows: int, hidden: int, epsilon: float | None,
                         variant: str = "fused_attention_norm",
                         dtype: str = "bf16", layout: str = "ND",
                         phase: str = "prefill",
                         call_id: str | None = None) -> dict[str, Any]:
    """Build the same AddRmsNorm descriptors without a compiled model node."""
    rows, hidden = int(rows), int(hidden)
    if rows < 1 or hidden < 1:
        raise ModelingError("add_rms_norm_geometry_invalid")
    if variant not in {"fused_attention_norm", "terminal_norm"}:
        raise ModelingError(f"add_rms_norm_variant_invalid:{variant}")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ModelingError(f"add_rms_norm_dtype_invalid:{dtype}")
    if epsilon is None:
        raise ModelingError("add_rms_norm_epsilon_required")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ModelingError("add_rms_norm_epsilon_invalid")
    fused = variant == "fused_attention_norm"
    dtype_bytes = 2 if dtype in {"bf16", "fp16"} else 4
    shapes = {"x": [rows, hidden], "residual": [rows, hidden],
              "gamma": [hidden], "output": [rows, hidden],
              "rstd": [rows, 1], "residual_out": [rows, hidden]}
    descriptors = {
        "x": _descriptor([rows, hidden], dtype, layout, "input"),
        "residual": _descriptor([rows, hidden], dtype, layout, "input"),
        "gamma": _descriptor([hidden], dtype, layout, "weight"),
        "output": _descriptor([rows, hidden], dtype, layout, "output"),
        "rstd": _descriptor([rows, 1], "fp32", layout, "output"),
        "residual_out": _descriptor([rows, hidden], dtype, layout, "output"),
    }
    return {
        "schema": "mindie-sim/add-rms-norm-native-prepare/1",
        "call": _ENTRY,
        "call_id": call_id,
        "native_type": _KERNEL,
        "variant": variant,
        "owner": ("fused:add_rmsnorm" if fused
                  else "O0:elementwise:final_norm"),
        "members": ["residual_add", "rmsnorm"] if fused else ["final_norm"],
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "shapes": shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "arguments": {"input": "x", "residual_input": "residual",
                      "weight": "gamma", "epsilon": epsilon,
                      "norm_output": "output", "rstd": "rstd",
                      "residual_output": "residual_out"},
        "attrs": {
            "variant": variant, "hidden": hidden, "rows": rows,
            "phase": phase, "epsilon": epsilon, "norm_eps": epsilon,
            "eps": epsilon, "dtype": dtype, "input_layout": layout,
            "rstd_dtype": "fp32", "output_dtype": dtype,
            "residual_dtype": dtype, "call_name": _ENTRY,
            "kernel_name": _KERNEL, "rstd_bytes": rows * 4,
            "residual_bytes": rows * hidden * dtype_bytes,
            "norm_output_bytes": rows * hidden * dtype_bytes,
            "physical_rows": rows, "row_basis": "explicit",
        },
        "outputs": {"norm_output": descriptors["output"],
                    "rstd": descriptors["rstd"],
                    "residual": descriptors["residual_out"]},
        "output_count": 3,
        "aliases": {},
        "alias_status": (
            "the captured ACLNN contract exposes a separate third residual "
            "output; no input alias is established"),
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "acl_prepare_only",
            "cpu_reference_execution_not_device_execution",
        ],
        "non_claims": [
            "no_device_launch", "no_measured_duration",
            "no_input_output_alias_claim",
        ],
    }


def execute_add_rms_norm(prepared: Any, values: Mapping[str, Any], *,
                         out: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute the real AddRmsNorm value contract on CPU with NumPy.

    ``residual_out = bf16(x + residual)``, ``rstd = 1/sqrt(mean(
    residual_out**2) + epsilon)`` and ``output = bf16(residual_out * rstd *
    gamma)``.  All three external outputs are retained.  The exact kernel
    intermediate cast is not retained, so only the final norm output is
    rounded to the declared payload dtype here.
    """
    np = _numpy()
    record = _prepare_record(prepared)
    attrs = dict(record.get("attrs") or {})
    x = _input_array(record, values, "x")
    residual = _input_array(record, values, "residual")
    gamma = _input_array(record, values, "gamma")
    epsilon = attrs.get("epsilon", attrs.get("norm_eps", attrs.get("eps")))
    if epsilon is None:
        raise NativeVectorExecutionUnavailable(
            "add_rms_norm_epsilon_required")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise NativeVectorExecutionUnavailable(
            "add_rms_norm_epsilon_invalid")
    if (x.ndim != 2 or x.shape != residual.shape
            or gamma.shape != (x.shape[-1],)):
        raise NativeVectorExecutionUnavailable("add_rms_norm_operand_mismatch")
    dtype = str(attrs.get("dtype") or "bf16")
    summed = x.astype(np.float32) + residual.astype(np.float32)
    if dtype == "bf16":
        fused_value = _bf16_round(summed)
    elif dtype == "fp16":
        fused_value = summed.astype(np.float16).astype(np.float32)
    else:
        fused_value = summed.astype(np.float32)
    variance = np.mean(fused_value.astype(np.float32) ** 2, axis=-1,
                       keepdims=True)
    rstd = (1.0 / np.sqrt(variance + epsilon)).astype(np.float32)
    output = fused_value.astype(np.float32) * rstd
    output = output * gamma.astype(np.float32).reshape(1, -1)
    return _finish_execution(
        record, {"output": output, "rstd": rstd,
                 "residual_out": fused_value},
        out=out)


def predict_add_rms_norm(prepared: Any, *, hardware: Any = None,
                         supply: Any = None, policy: Any = None,
                         phase: str | None = None) -> dict[str, Any]:
    """Derive one physical AddRmsNorm program for both caller variants.

    The ACLNN call is the same full three-output kernel whether the caller
    labels it ``terminal_norm`` or ``fused_attention_norm``.  Physical stages
    and transactions are therefore derived once here; the caller variant only
    changes the reported semantic members/fusion, never the work.
    """
    from .compiler import CompiledOperator, ImplementationPolicy
    from .composition import lower_fusion
    from mindie_sim.schema.subgraphs import (
        BoundOperator, FusionContract, OperatorTemplate, StaticSignature,
        SubgraphInstance, SubgraphTemplate)
    record = _prepare_record(prepared)
    attrs = dict(record.get("attrs") or {})
    shapes = record.get("shapes") or {}
    rows = int((shapes.get("x") or [attrs.get("rows", 0)])[0])
    hidden = int((shapes.get("x") or [0, attrs.get("hidden", 0)])[-1])
    if rows < 1 or hidden < 1:
        raise ModelingError("add_rms_norm_geometry_invalid")
    variant = str(record.get("variant") or attrs.get("variant")
                  or "fused_attention_norm")
    if variant not in {"fused_attention_norm", "terminal_norm"}:
        raise ModelingError(f"add_rms_norm_variant_invalid:{variant}")
    terminal = variant == "terminal_norm"
    runtime_phase = str(phase or attrs.get("phase")
                        or ("decode" if rows == 1 else "prefill"))
    supply_obj, arch, _label = _leaf_supply(hardware, supply)
    policy_obj = policy or ImplementationPolicy(cores=64)
    geometry = _leaf_geometry(rows, runtime_phase)
    dtype = str(attrs.get("dtype") or "bf16")
    dtype_bytes = 2 if dtype in {"bf16", "fp16"} else 4

    def theta(role, elements, ops, read, write, extra=()):
        return StaticSignature(
            role, matrix=(("elements", int(elements)),),
            quant=(("act_bytes", float(dtype_bytes)),),
            extras=(("ops_per_element", int(ops)),
                    ("read_bytes", float(read)), ("write_bytes", float(write)),
                    *tuple(extra)))

    residual_signature = theta(
        "residual", hidden, 2, 2 * hidden * dtype_bytes,
        hidden * dtype_bytes)
    norm_signature = theta(
        "norm", hidden, 5, hidden * dtype_bytes, hidden * dtype_bytes,
        (("gamma_bytes", hidden * dtype_bytes),
         ("gamma_storage_bytes", hidden * dtype_bytes),
         ("gamma_format", dtype), ("gamma_storage_format", dtype),
         ("gamma_role", "fused_native_norm"), ("norm_arch", "rmsnorm"),
         ("layernorm", 0), ("norm_weight_offset", 0.0),
         ("norm_has_weight", True), ("norm_has_bias", False)))
    residual_bound = BoundOperator(
        OperatorTemplate("residual_add", "elementwise", "residual"),
        residual_signature)
    norm_bound = BoundOperator(
        OperatorTemplate("rmsnorm", "elementwise", "norm"), norm_signature)
    contract = FusionContract(
        "add_rmsnorm", ("residual_add", "rmsnorm"),
        removes=("residual_add_output", "rmsnorm_output"),
        keeps=("residual", "rstd", "norm_output"), startup=1)
    template = SubgraphTemplate(
        kind="layernorm", variant="native_add_rmsnorm",
        title="Native AddRmsNorm (residual + RMSNorm)",
        operators=(residual_bound.template, norm_bound.template),
        fusions=(contract,))
    instance = SubgraphInstance(
        instance_id=f"native-vector:add_rmsnorm:{rows}x{hidden}",
        template=template, layer=0, stage=0,
        operators=(residual_bound, norm_bound))
    residual_cost = CompiledOperator(
        residual_bound, supply_obj, policy_obj, kind="layernorm").evaluate(geometry)
    norm_cost = CompiledOperator(
        norm_bound, supply_obj, policy_obj, kind="layernorm").evaluate(geometry)
    fused = lower_fusion(instance, (residual_cost, norm_cost),
                         "add_rmsnorm", supply=supply_obj).fused
    fused = _preserve_member_stage_work(
        fused, (residual_cost, norm_cost), supply_obj)
    fused, logical, transactions, rstd_added = _add_rms_norm_write_ledger(
        record, residual_cost, norm_cost, fused, supply=supply_obj)
    caller_members = (["final_norm"] if terminal
                      else ["residual_add", "rmsnorm"])
    caller_program = ("elementwise.O0.add_rmsnorm" if terminal
                      else "elementwise.N0.add_rmsnorm")
    program = {
        "common_program": caller_program,
        "physical_program": "elementwise.N0.add_rmsnorm",
        "caller_program": caller_program,
        "caller_variant": variant,
        "caller_kind": "lm_head_sampler" if terminal else "layernorm",
        "members": caller_members,
        "physical_members": ["residual_add", "rmsnorm"],
        "fusion": (None if terminal else {
            "contract": "add_rmsnorm", "keeps": list(contract.keeps)}),
        "elements": hidden,
        "stage_access": [
            {"inputs": ["x", "residual"], "outputs": ["residual_out"]},
            {"inputs": ["residual_out", "gamma"],
             "outputs": ["output", "rstd"]}],
        "logical_materialization": dict(logical),
        "transactions": dict(transactions),
        "rstd_write_added_bytes": rstd_added,
        "member_costs": [
            {"name": "residual_add",
             "duration_ms": str(residual_cost.duration_ms)},
            {"name": "rmsnorm", "duration_ms": str(norm_cost.duration_ms)},
        ],
    }
    return _prediction_record(
        fused, call=record.get("call"), rows=rows, family="elementwise",
        kind="layernorm", arch=arch, phase=runtime_phase, program=program)


_VALUE_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4,
                      "int8": 1, "int32": 4, "int64": 8}


def _descriptor_value_bytes(descriptor: Mapping[str, Any]) -> int:
    """Effective bytes of one materialized value, from its own descriptor."""
    shape = descriptor.get("view_shape") or descriptor.get("shape")
    dtype = str(descriptor.get("dtype") or "")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise NativeVectorExecutionUnavailable(
            "add_rms_norm_descriptor_shape_missing")
    if dtype not in _VALUE_DTYPE_BYTES:
        raise NativeVectorExecutionUnavailable(
            f"add_rms_norm_descriptor_dtype_unsupported:{dtype}")
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return elements * _VALUE_DTYPE_BYTES[dtype]


def _descriptor_storage_bytes(descriptor: Mapping[str, Any]) -> int:
    shape = descriptor.get("storage_shape") or descriptor.get("shape")
    dtype = str(descriptor.get("dtype") or "")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise NativeVectorExecutionUnavailable(
            "add_rms_norm_descriptor_storage_missing")
    if dtype not in _VALUE_DTYPE_BYTES:
        raise NativeVectorExecutionUnavailable(
            f"add_rms_norm_descriptor_dtype_unsupported:{dtype}")
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return elements * _VALUE_DTYPE_BYTES[dtype]



# ---------------------------------------------------------------------------
# Ordinary norm named access ledger.
#
# RMSNorm/gated RMSNorm publish one named access per real value role before the
# shared elementwise family consumes them.  The family's frozen convention is
# that ``read_bytes`` is a per-row subtotal which already contains the resident
# gamma vector once: the family removes that gamma subtotal, aligns the
# activation row to 32 B and adds gamma back once per call.  Keeping that
# conversion here (instead of two independent guesses in the leaf and family)
# preserves logical materialization, the 32 B row transaction and the storage
# capacity as separate ledgers.
# ---------------------------------------------------------------------------
NORM_ACCESS_SCHEMA = "mindie-sim/native-norm-access-plan/1"
NORM_ACCESS_ALIGNMENT = 32


def _access_align(value: int) -> int:
    value = int(value)
    return ((value + NORM_ACCESS_ALIGNMENT - 1)
            // NORM_ACCESS_ALIGNMENT) * NORM_ACCESS_ALIGNMENT


def _access_bytes(descriptor: Mapping[str, Any] | None,
                  fallback: int) -> tuple[int, int]:
    if not isinstance(descriptor, Mapping):
        return int(fallback), int(fallback)
    logical = _descriptor_value_bytes(descriptor)
    storage = _descriptor_storage_bytes(descriptor)
    return logical, storage


def norm_access_plan(*, rows: int, width: int, dtype_bytes: int,
                     gamma_bytes: int, beta_bytes: int = 0,
                     row_inputs: tuple[tuple[str, str, int], ...] = (),
                     residual_bytes_per_row: int = 0,
                     residual_name: str = "residual",
                     residual_descriptor: str | None = None,
                     output_name: str = "output",
                     output_descriptor: str | None = None,
                     rstd_name: str = "rstd", rstd_bytes: int = 4,
                     gamma_name: str = "weight",
                     gamma_descriptor: str | None = None,
                     beta_name: str = "beta",
                     beta_descriptor: str | None = None,
                     descriptors: Mapping[str, Any] | None = None,
                     family: str = "rmsnorm",
                     norm_before_gate: bool | None = None,
                     source_axis: Any = None,
                     original_shapes: Mapping[str, Any] | None = None,
                     execution_view: Mapping[str, Any] | None = None,
                     ) -> dict[str, Any]:
    """Build one named access ledger for an ordinary row norm.

    ``row_inputs`` entries are ``(name, descriptor, bytes_per_row)``.  Every
    row input is read per row; gamma and beta are resident once per call; the
    actual activation and FP32 rstd are written once per row.  The returned
    ``family`` block is the only conversion to the shared elementwise family.
    """
    rows, width = int(rows), int(width)
    dtype_bytes, gamma_bytes = int(dtype_bytes), int(gamma_bytes)
    if rows < 1 or width < 1 or dtype_bytes < 1:
        raise ModelingError("norm_access_geometry_invalid")
    if gamma_bytes < 0 or beta_bytes < 0 or rstd_bytes < 0:
        raise ModelingError("norm_access_resident_bytes_invalid")
    descriptors = dict(descriptors or {})
    original_shapes = dict(original_shapes or {})
    accesses: list[dict[str, Any]] = []
    read_row_un = 0

    for name, descriptor_name, per_row in row_inputs:
        per_row = int(per_row)
        if per_row < 0:
            raise ModelingError(f"norm_access_row_bytes_invalid:{name}")
        read_row_un += per_row
        descriptor = descriptors.get(descriptor_name)
        logical, storage = _access_bytes(descriptor, per_row * rows)
        accesses.append({
            "schema": NORM_ACCESS_SCHEMA,
            "name": str(name), "descriptor": str(descriptor_name),
            "role": "input", "scope": "row", "access": "read",
            "rows": rows, "bytes_per_row": per_row,
            "logical_bytes": int(per_row) * rows,
            "storage_bytes": int(storage),
            "storage": str((descriptor or {}).get("storage") or
                           descriptor_name or name),
            "materialized": True,
            "alignment_bytes": NORM_ACCESS_ALIGNMENT,
        })

    if residual_bytes_per_row:
        per_row = int(residual_bytes_per_row)
        read_row_un += per_row
        descriptor = descriptors.get(residual_descriptor or residual_name)
        _logical, storage = _access_bytes(descriptor, per_row * rows)
        accesses.append({
            "schema": NORM_ACCESS_SCHEMA,
            "name": "residual", "descriptor": str(
                residual_descriptor or residual_name),
            "role": "input", "scope": "row", "access": "read",
            "rows": rows, "bytes_per_row": per_row,
            "logical_bytes": per_row * rows,
            "storage_bytes": int(storage),
            "storage": str((descriptor or {}).get("storage") or
                           residual_descriptor or residual_name),
            "materialized": True,
            "alignment_bytes": NORM_ACCESS_ALIGNMENT,
        })

    if gamma_bytes:
        descriptor = descriptors.get(gamma_descriptor or gamma_name)
        _logical, storage = _access_bytes(descriptor, gamma_bytes)
        accesses.append({
            "schema": NORM_ACCESS_SCHEMA,
            "name": "gamma", "descriptor": str(
                gamma_descriptor or gamma_name),
            "role": "weight", "scope": "call", "access": "read",
            "rows": 1, "bytes_per_row": 0,
            "logical_bytes": gamma_bytes,
            "storage_bytes": int(storage),
            "storage": str((descriptor or {}).get("storage") or
                           gamma_descriptor or gamma_name),
            "materialized": True,
            "alignment_bytes": 0,
        })
    if beta_bytes:
        descriptor = descriptors.get(beta_descriptor or beta_name)
        _logical, storage = _access_bytes(descriptor, beta_bytes)
        accesses.append({
            "schema": NORM_ACCESS_SCHEMA,
            "name": "beta", "descriptor": str(beta_descriptor or beta_name),
            "role": "weight", "scope": "call", "access": "read",
            "rows": 1, "bytes_per_row": 0,
            "logical_bytes": beta_bytes,
            "storage_bytes": int(storage),
            "storage": str((descriptor or {}).get("storage") or
                           beta_descriptor or beta_name),
            "materialized": True,
            "alignment_bytes": 0,
        })

    output_descriptor = output_descriptor or output_name
    output_per_row = width * dtype_bytes
    output_l, output_s = _access_bytes(
        descriptors.get(output_descriptor), output_per_row * rows)
    accesses.append({
        "schema": NORM_ACCESS_SCHEMA,
        "name": str(output_name), "descriptor": str(output_descriptor),
        "role": "output", "scope": "row", "access": "write",
        "rows": rows, "bytes_per_row": output_per_row,
        "logical_bytes": int(output_l),
        "storage_bytes": int(output_s),
        "storage": str((descriptors.get(output_descriptor) or {}).get(
            "storage") or output_descriptor),
        "materialized": True,
        "alignment_bytes": NORM_ACCESS_ALIGNMENT,
        "own_transaction_bytes": (_access_align(output_per_row) * rows),
    })
    rstd_l, rstd_s = _access_bytes(descriptors.get(rstd_name), rstd_bytes * rows)
    if rstd_bytes:
        accesses.append({
            "schema": NORM_ACCESS_SCHEMA,
            "name": str(rstd_name), "descriptor": str(rstd_name),
            "role": "stats", "scope": "row", "access": "write",
            "rows": rows, "bytes_per_row": rstd_bytes,
            "logical_bytes": int(rstd_l),
            "storage_bytes": int(rstd_s),
            "storage": str((descriptors.get(rstd_name) or {}).get(
                "storage") or rstd_name),
            "materialized": True,
            "alignment_bytes": NORM_ACCESS_ALIGNMENT,
            "own_transaction_bytes": (_access_align(rstd_bytes) * rows),
        })

    write_row_un = output_per_row + int(residual_bytes_per_row) + int(
        rstd_bytes)
    read_row_tx = _access_align(read_row_un)
    write_row_tx = _access_align(write_row_un)
    read_total = rows * read_row_tx + gamma_bytes + beta_bytes
    write_total = rows * write_row_tx
    logical = {
        "norm_output": int(output_l),
        "residual": int(residual_bytes_per_row) * rows,
        "rstd": int(rstd_l) if rstd_bytes else 0,
    }
    family_block = {
        # Family subtotal convention: one per-row activation subtotal plus the
        # resident gamma vector exactly once.  The family subtracts gamma,
        # aligns the row, then adds gamma back once per launch.
        "read_bytes_per_row": read_row_un + gamma_bytes + beta_bytes,
        "write_bytes_per_row": write_row_un,
        # The frozen family subtracts one resident vector subtotal and adds it
        # back once.  Gamma and beta are both resident per call, so the family
        # receives their combined resident bytes while the access ledger keeps
        # them separate.
        "gamma_bytes": gamma_bytes + beta_bytes,
        "beta_bytes": beta_bytes,
        "resident_bytes": gamma_bytes + beta_bytes,
        "rstd_bytes": rstd_bytes,
        "residual_out_bytes": int(residual_bytes_per_row),
        "act_bytes": dtype_bytes,
        "read_row_bytes": rows * read_row_tx,
        "write_row_bytes": write_total,
        "gamma_transaction_bytes": gamma_bytes,
        "mte2_bytes": read_total,
        "mte3_bytes": write_total,
        "hbm_bytes": read_total + write_total,
    }
    return {
        "schema": NORM_ACCESS_SCHEMA,
        "family": str(family),
        "rows": rows, "width": width, "dtype_bytes": dtype_bytes,
        "norm_before_gate": norm_before_gate,
        "source_axis": source_axis,
        "original_shapes": dict(original_shapes),
        "execution_view": dict(execution_view or {}),
        "accesses": accesses,
        "logical_materialization": logical,
        "transactions": {
            "alignment_bytes": NORM_ACCESS_ALIGNMENT,
            "read_row_bytes": read_row_tx,
            "write_row_bytes": write_row_tx,
            "gamma_bytes": gamma_bytes,
            "mte2_bytes": read_total,
            "mte3_bytes": write_total,
            "hbm_bytes": read_total + write_total,
            "basis": ("one 32 B-aligned activation read row plus gamma once; "
                      "one 32 B-aligned output/rstd write row"),
        },
        "family_parameters": family_block,
        "storage": {
            str(item["name"]): int(item["storage_bytes"])
            for item in accesses if item.get("materialized", True)},
        "logical_total_bytes": sum(int(value) for value in logical.values()),
        "non_claims": (
            "static_cpu_work_ledger",
            "logical_value_bytes_are_not_issued_traffic",
            "alignment_is_only_in_transactions",
        ),
    }


def _member_issued_write(cost: Any) -> int:
    """The member's actual MTE3 write transaction (aligned bytes)."""
    values = []
    for stage in cost.stage_work:
        entries = stage.items() if hasattr(stage, "items") else stage
        for pipe, value in entries:
            if pipe == "mte3" and int(value) > 0:
                values.append(int(value))
    if not values:
        # Older callers may not carry stage_work; materialization is the
        # fallback only when it is the actual aligned write ledger.
        values = [int(value) for _name, value in cost.materialization
                  if int(value) > 0]
    if not values:
        raise ModelingError("add_rms_norm_member_write_missing")
    return sum(values)


def _add_stage_bytes(cost: Any, *, amount: int,
                     pipes: tuple[str, ...], supply: Any) -> Any:
    """Add one real access amount to the last physical stage."""
    from dataclasses import replace
    from fractions import Fraction
    if amount <= 0:
        return cost
    if not cost.stage_work:
        raise ModelingError("fused_add_rmsnorm_stage_work_missing")
    stage_work = [dict(stage) for stage in cost.stage_work]
    for pipe in pipes:
        stage_work[-1][pipe] = (stage_work[-1].get(pipe, Fraction(0))
                                + amount)
    rates = {pipe: Fraction(str(supply.effective(pipe)))
             for pipe in {name for stage in stage_work for name in stage}}
    stage_times: list[Fraction] = []
    times: dict[str, Fraction] = {}
    peak: dict[str, Fraction] = {}
    work: dict[str, Fraction] = {}
    for stage in stage_work:
        entries = {pipe: value for pipe, value in stage.items() if value > 0}
        if not entries:
            stage_times.append(Fraction(0))
            continue
        elapsed = {pipe: value / rates[pipe] for pipe, value in entries.items()}
        stage_times.append(max(elapsed.values()))
        for pipe, value in entries.items():
            work[pipe] = work.get(pipe, Fraction(0)) + value
            times[pipe] = times.get(pipe, Fraction(0)) + elapsed[pipe]
            peak[pipe] = max(peak.get(pipe, Fraction(0)), elapsed[pipe])
    derived = dict(cost.derived)
    body = sum(stage_times, Fraction(0))
    duration = max(
        derived.get("service_floor_ms", Fraction(0)),
        derived.get("service_startup_ms", Fraction(0)) + body
        + derived.get("service_dependency_ms", Fraction(0)))
    dominant = max(peak, key=peak.get) if peak else ""
    tied = tuple(pipe for pipe, value in peak.items()
                 if peak and value == max(peak.values()))
    return replace(
        cost, work=tuple(sorted(work.items())),
        times=tuple(sorted(times.items())),
        stage_times=tuple(stage_times), duration_ms=duration,
        dominant_pipe=dominant, tied_pipes=tied,
        stage_work=tuple(stage_work), derived=tuple(sorted(derived.items())))


def _add_stage_write(cost: Any, *, write_bytes: int, supply: Any) -> Any:
    return _add_stage_bytes(
        cost, amount=write_bytes, pipes=("mte3", "hbm"), supply=supply)


def _add_stage_read(cost: Any, *, read_bytes: int, supply: Any) -> Any:
    return _add_stage_bytes(
        cost, amount=read_bytes, pipes=("mte2", "hbm"), supply=supply)


def _preserve_member_stage_work(fused: Any, costs: tuple[Any, ...],
                                supply: Any) -> Any:
    """Keep real member stages when fusion loses some of their access work."""
    from dataclasses import replace
    from fractions import Fraction
    member_stages: list[dict[str, Fraction]] = []
    member_work: dict[str, Fraction] = {}
    member_stage_times: list[Fraction] = []
    member_times: dict[str, Fraction] = {}
    for cost in costs:
        for stage in cost.stage_work:
            entries = stage.items() if hasattr(stage, "items") else stage
            row: dict[str, Fraction] = {}
            for pipe, value in entries:
                amount = Fraction(value)
                if amount > 0:
                    row[str(pipe)] = amount
                    member_work[str(pipe)] = (
                        member_work.get(str(pipe), Fraction(0)) + amount)
            member_stages.append(row)
        member_stage_times.extend(cost.stage_times)
        for pipe, value in cost.times:
            member_times[str(pipe)] = (
                member_times.get(str(pipe), Fraction(0)) + Fraction(value))
    fused_work = {str(pipe): Fraction(value) for pipe, value in fused.work}
    if all(member_work.get(pipe, Fraction(0))
           <= fused_work.get(pipe, Fraction(0)) for pipe in member_work):
        return fused
    derived = dict(fused.derived)
    body = sum(member_stage_times, Fraction(0))
    duration = max(
        derived.get("service_floor_ms", Fraction(0)),
        derived.get("service_startup_ms", Fraction(0)) + body
        + derived.get("service_dependency_ms", Fraction(0)))
    dominant = max(member_times, key=member_times.get) if member_times else ""
    peak = max(member_times.values()) if member_times else Fraction(0)
    tied = tuple(pipe for pipe, value in member_times.items()
                 if value == peak)
    return replace(
        fused, work=tuple(sorted(member_work.items())),
        stage_work=tuple(member_stages),
        stage_times=tuple(member_stage_times),
        times=tuple(sorted(member_times.items())),
        duration_ms=duration, dominant_pipe=dominant, tied_pipes=tied,
        formula="serial_member_stages")


def _add_rms_norm_write_ledger(record: Mapping[str, Any],
                               residual_cost: Any, norm_cost: Any,
                               fused: Any, *, supply: Any,
                               output_name: str = "output",
                               residual_name: str = "residual_out",
                               output_key: str = "norm_output",
                               residual_key: str = "residual") -> tuple:
    """Split logical effective materialization from issued write transactions.

    * logical: each external output's own descriptor shape and dtype.
    * issued: the real member MTE3 writes plus the FP32 rstd write, including
      UB row alignment on the two BF16 outputs.
    * rstd: added once only when the fused stage work does not already carry
      that write.
    """
    from dataclasses import replace
    from fractions import Fraction
    descriptors = record.get("descriptors") or {}
    logical = {
        output_key: _descriptor_value_bytes(descriptors[output_name]),
        residual_key: _descriptor_value_bytes(descriptors[residual_name]),
        "rstd": _descriptor_value_bytes(descriptors["rstd"]),
    }
    logical_total = sum(logical.values())
    residual_issued = _member_issued_write(residual_cost)
    norm_issued = _member_issued_write(norm_cost)
    activation_issued = residual_issued + norm_issued
    rstd_issued = logical["rstd"]
    expected_mte3 = activation_issued + rstd_issued
    current_mte3 = int(dict(fused.work).get("mte3", 0))
    if current_mte3 == expected_mte3:
        rstd_added = 0
    elif current_mte3 == activation_issued:
        rstd_added = rstd_issued
        fused = _add_stage_write(fused, write_bytes=rstd_added,
                                 supply=supply)
    else:
        raise ModelingError(
            "add_rms_norm_write_stage_mismatch:"
            f"{current_mte3}!={activation_issued}+{rstd_issued}")
    issued_total = activation_issued + rstd_issued
    if int(dict(fused.work).get("mte3", 0)) != issued_total:
        raise ModelingError(
            "add_rms_norm_issued_write_mismatch:"
            f"{dict(fused.work).get('mte3')}!={issued_total}")
    issued = {
        (f"{residual_key}_write_bytes"
         if residual_key != "residual" else "residual_write_bytes"):
            residual_issued,
        (f"{output_key}_write_bytes"
         if output_key != "norm_output" else "norm_output_write_bytes"):
            norm_issued,
        "rstd_write_bytes": rstd_issued,
        "total_write_bytes": issued_total,
        "logical_total_bytes": logical_total,
        "alignment_bytes": issued_total - logical_total,
        "mte3_bytes": issued_total,
    }
    fused = replace(fused,
                    materialization=tuple(sorted(logical.items())))
    derived = dict(fused.derived)
    derived["kept_bytes"] = Fraction(logical_total)
    derived["logical_effective_bytes"] = Fraction(logical_total)
    derived["issued_write_bytes"] = Fraction(issued_total)
    derived["write_alignment_bytes"] = Fraction(issued_total - logical_total)
    derived["rstd_write_bytes"] = Fraction(rstd_issued)
    derived["rstd_write_reconciled"] = Fraction(rstd_added)
    fused = replace(fused, derived=tuple(sorted(derived.items())))
    return fused, logical, issued, rstd_added


def add_rms_norm_write_ledger(prepared: Any, *, hardware: Any = None,
                              supply: Any = None, policy: Any = None,
                              phase: str | None = None) -> dict[str, Any]:
    """Read-only write ledger for PW-CORE/T12 consumers.

    ``logical_materialization`` is the effective value capacity from each
    external descriptor; ``issued_transactions`` is the aligned physical MTE3
    write ledger.  Both are exported by the same prediction entry so consumers
    never recompute or conflate them.
    """
    prediction = predict_add_rms_norm(
        prepared, hardware=hardware, supply=supply, policy=policy,
        phase=phase)
    program = dict(prediction.get("program") or {})
    derived = dict(prediction.get("derived") or {})
    return {
        "schema": "mindie-sim/add-rms-norm-write-ledger/1",
        "call": prediction.get("call"),
        "logical_materialization": {
            name: int(value) for name, value in
            prediction["materialization"].items()},
        "logical_total_bytes": int(derived.get("kept_bytes", 0)),
        "issued_transactions": dict(program.get("transactions") or {}),
        "issued_mte3_bytes": int(prediction["work"]["mte3"]),
        "rstd_bytes": int(prediction["materialization"]["rstd"]),
        "rstd_write_added_bytes": int(
            program.get("rstd_write_added_bytes", 0)),
        "physical_program": program.get("physical_program"),
        "carrier_program": program.get("common_program"),
        "caller_variant": program.get("caller_variant"),
        "environment": dict(prediction.get("environment") or {}),
        "non_claims": (
            "static_cpu_prediction",
            "logical_value_capacity_is_not_issued_traffic",
            "alignment_is_only_in_issued_transactions",
            "no_target_duration_injected",
        ),
    }


def _leaf_event(value: Any) -> Any:
    """Resolve a handle/event pair without inventing a synthetic hold."""
    return value.event if hasattr(value, "event") else value


def _leaf_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _leaf_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_leaf_plain(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "name") and value.__class__.__name__ == "Event":
        return value.name
    return value


@dataclass(frozen=True)
class LeafExecutionHandle:
    """Actual handles returned by one leaf submission."""
    event: Any
    handles: tuple[Any, ...] = ()
    output_events: tuple[tuple[str, Any], ...] = ()
    output_values: tuple[tuple[str, Any], ...] = ()
    output_bindings: tuple[tuple[str, Any], ...] = ()
    release_events: tuple[Any, ...] = ()
    storage_bindings: tuple[tuple[str, dict], ...] = ()
    use_events: tuple[tuple[str, Any], ...] = ()
    grants: tuple[tuple[str, dict], ...] = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.event, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.name,
            "outputs": {name: event.name for name, event in self.output_events},
            "output_values": {
                name: value.to_dict() for name, value in self.output_values},
            "output_bindings": {
                name: _leaf_plain(value)
                for name, value in self.output_bindings},
            "storage_bindings": {
                name: _leaf_plain(value)
                for name, value in self.storage_bindings},
            "use_events": {
                name: (event.name if hasattr(event, "name") else str(event))
                for name, event in self.use_events},
            "grants": {
                name: _leaf_plain(value) for name, value in self.grants},
            "releases": [event.name for event in self.release_events],
        }


@dataclass(frozen=True)
class LeafExecutionProgram:
    """One leaf's existing stage/native program plus typed handoff.

    ``program`` is a ``FamilyFragmentProgram`` or a caller-supplied native
    program.  ``start`` binds it to the caller's existing Execution so resource
    names, pool bindings and events are the caller's, never a private Engine.
    """

    name: str
    call: str
    program: Any
    fragments: tuple[Any, ...]
    typed_values: tuple[tuple[str, Any], ...]
    aliases: tuple[tuple[str, str], ...]
    buffers: tuple[Any, ...]
    stage_work: tuple[dict[str, str], ...]
    edges: tuple[tuple[int, ...], ...]
    outputs: tuple[str, ...] = ()
    unmaterialized: tuple[str, ...] = ()
    input_bindings: tuple[tuple[str, str], ...] = ()
    accesses: tuple[dict[str, Any], ...] = ()
    access_plan: Any = None
    abi_handoff: Any = None
    native_program: Any = None
    native_program_available: bool = False
    non_claims: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mindie-sim/leaf-execution-program/1",
            "name": self.name,
            "call": self.call,
            "program_source": ("native_program" if self.native_program is not None
                               else "stage_work"),
            "stage_work": [dict(stage) for stage in self.stage_work],
            "edges": [list(edge) for edge in self.edges],
            "outputs": list(self.outputs),
            "unmaterialized": list(self.unmaterialized),
            "input_bindings": dict(self.input_bindings),
            "accesses": [dict(item) for item in self.accesses],
            "access_plan": _leaf_plain(self.access_plan),
            "abi_handoff": _leaf_plain(self.abi_handoff),
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "typed_values": {
                name: value.to_dict() for name, value in self.typed_values},
            "aliases": dict(self.aliases),
            "buffers": [dict(item) if isinstance(item, Mapping) else str(item)
                        for item in self.buffers],
            "native_program_available": bool(self.native_program_available),
            "non_claims": list(self.non_claims),
        }

    def start(self, execution: Any, name: str | None = None,
              ready: Any = None, *, replica: int = 0, stage: int = 0,
              owner: str = "", inputs: Mapping[str, Any] | None = None,
              holds: Mapping[str, Any] | None = None,
              event_bindings: Mapping[str, Any] | None = None,
              occurrence: Any = None,
              allocations: tuple[Any, ...] | None = None,
              output_grants: Mapping[str, Any] | None = None,
              output_bindings: Mapping[str, Any] | None = None,
              allocation_bindings: Mapping[str, str] | None = None,
              pool_bindings: Mapping[str, str] | None = None) -> Any:
        """Submit one occurrence into the caller's existing Execution.

        The same typed ports, alias roots, descriptors and output bindings are
        converted once into a storage/generation use set.  A declared grant is
        acquired before its first writer; its release waits for the last real
        internal consumer and every externally declared async reader.  A
        borrowed grant is never acquired twice: its real identity is bound and
        its reader registration is completed on the same last-use edge.
        """
        from dataclasses import replace
        from mindie_sim.schema.invocation import Occurrence
        from mindie_sim.schema.operators import AllocationEffect
        name = name or self.name
        engine = execution.engine
        ready = _leaf_event(ready)
        if ready is None:
            ready = engine.submit(f"{name}:ready", owner=owner)
        if self.native_program is not None:
            event = execution.start_native_participant(
                self.native_program, name, ready, owner=owner)
            return LeafExecutionHandle(event=event)
        fragments = self.fragments
        if not fragments:
            raise ModelingError("leaf_execution_program_no_stages")
        typed = dict(self.typed_values)
        aliases = dict(self.aliases)
        unmaterialized = set(self.unmaterialized)

        def root_of(port_name: str) -> str:
            value = typed.get(port_name)
            if value is not None:
                if getattr(value, "storage", ""):
                    return str(value.storage)
                if getattr(value, "alias_of", ""):
                    return str(value.alias_of)
            return str(aliases.get(port_name, port_name))

        written: dict[str, list[int]] = {}
        read: dict[str, list[int]] = {}
        for index, fragment in enumerate(fragments):
            for port in tuple(fragment.ports.tensor_out) + tuple(
                    port for port in fragment.ports.tensor_in
                    if port.direction == "inout"):
                written.setdefault(port.name, []).append(index)
            for port in fragment.ports.tensor_in:
                read.setdefault(port.name, []).append(index)
        roots: dict[str, dict[str, Any]] = {}
        output_root: dict[str, str] = {}
        for port_name, stages in written.items():
            root = root_of(port_name)
            output_root[port_name] = root
            info = roots.setdefault(root, {
                "root": root, "outputs": [], "first_writer": min(stages),
                "writers": {}, "read_stages": set()})
            info["outputs"].append(port_name)
            info["writers"][port_name] = min(stages)
            info["first_writer"] = min(info["first_writer"], min(stages))
        for port_name, stages in read.items():
            root = root_of(port_name)
            if root in roots:
                roots[root]["read_stages"].update(int(item) for item in stages)

        bindings: dict[str, dict[str, Any]] = {}

        def binding_for(output_name: str) -> dict[str, Any]:
            if output_name not in output_root:
                raise ModelingError(
                    f"leaf_execution_output_unknown:{output_name}")
            return bindings.setdefault(output_name, {})

        for output_name, grant in dict(output_grants or {}).items():
            if grant is None:
                continue
            if not isinstance(grant, Mapping):
                raise ModelingError(
                    f"leaf_execution_grant_invalid:{output_name}")
            record = binding_for(output_name)
            record["grant"] = dict(grant)
            for key in ("last_reader", "hold"):
                if grant.get(key) is not None:
                    record.setdefault("readers", []).append(grant[key])
            if isinstance(grant.get("readers"), (list, tuple)):
                record.setdefault("readers", []).extend(grant["readers"])
        for output_name, spec in dict(output_bindings or {}).items():
            record = binding_for(output_name)
            if isinstance(spec, Mapping):
                grant = spec.get("grant")
                if isinstance(grant, Mapping):
                    record["grant"] = {**dict(record.get("grant") or {}),
                                       **dict(grant)}
                elif isinstance(grant, str):
                    record["grant"] = {"name": grant, "borrow": True}
                for key in ("storage", "alias_of", "generation", "value",
                            "binding_name", "materialized"):
                    if key in spec:
                        record[key] = spec[key]
                readers = []
                for key in ("reader", "last_reader", "event", "readers"):
                    if key not in spec:
                        continue
                    value = spec[key]
                    if key == "readers" and isinstance(value, (list, tuple)):
                        readers.extend(value)
                    else:
                        readers.append(value)
                if "hold" in spec:
                    readers.append(spec["hold"])
                if readers:
                    record["readers"] = readers
                if spec.get("borrow") or spec.get("name"):
                    grant_record = record.setdefault("grant", {})
                    if spec.get("borrow"):
                        grant_record["borrow"] = True
                    for key in ("name", "pool", "size", "units", "unit",
                                "lifetime", "owner", "ranges", "event"):
                        if key in spec:
                            grant_record[key] = spec[key]
            else:
                record["readers"] = [spec]

        def resolve_allocation_root(effect: AllocationEffect) -> str:
            explicit = dict(allocation_bindings or {})
            wanted = explicit.get(effect.name)
            if wanted is not None:
                if wanted in roots:
                    return wanted
                if wanted in output_root:
                    return output_root[wanted]
                raise ModelingError(
                    f"leaf_execution_allocation_binding_unknown:{wanted}")
            for root, info in roots.items():
                grant = dict(bindings.get(
                    info["outputs"][0], {}).get("grant") or {})
                if effect.name in {
                        root, info["outputs"][0],
                        str(grant.get("name") or "")}:
                    return root
            # A grant for a cross-stage storage is identified by its actual
            # first writer + later internal reader; this is the AddRmsNorm
            # residual case, not a shape table.
            cross = [
                root for root, info in roots.items()
                if info["read_stages"]
                and max(info["read_stages"]) > info["first_writer"]]
            if len(cross) == 1:
                return cross[0]
            first = min(info["first_writer"] for info in roots.values())
            first_stage = [
                root for root, info in roots.items()
                if info["first_writer"] == first]
            if len(first_stage) == 1:
                return first_stage[0]
            if len(roots) == 1:
                return next(iter(roots))
            raise ModelingError(
                "leaf_execution_allocation_output_ambiguous:"
                + effect.name + ":" + ",".join(sorted(roots)))

        # ``allocations`` is the legacy entry.  Resolve each effect to the same
        # storage/writer as output grants, and consume its paired release into
        # the complete-use hold instead of releasing after the first writer.
        allocation_effects = tuple(allocations or ())
        acquired_effect_names: set[str] = set()
        for effect in allocation_effects:
            if not isinstance(effect, AllocationEffect):
                raise ModelingError("leaf_execution_allocation_effect_invalid")
            if effect.condition:
                raise ModelingError(
                    f"leaf_execution_conditional_allocation_unsupported:"
                    f"{effect.name}")
            if effect.lifetime == "event":
                raise ModelingError(
                    f"leaf_execution_event_allocation_unsupported:{effect.name}")
            root = resolve_allocation_root(effect)
            output_name = roots[root]["outputs"][0]
            record = binding_for(output_name)
            grant = dict(record.get("grant") or {})
            if effect.action == "acquire":
                existing_name = str(grant.get("name") or "")
                if existing_name and existing_name != effect.name:
                    raise ModelingError(
                        f"leaf_execution_grant_name_conflict:"
                        f"{existing_name}!={effect.name}")
                grant.update({
                    "name": effect.name, "pool": effect.pool,
                    "size": effect.size, "units": effect.units,
                    "unit": effect.unit, "lifetime": effect.lifetime})
                record["grant"] = grant
                acquired_effect_names.add(effect.name)
            elif effect.action == "release":
                continue
            else:  # pragma: no cover - AllocationEffect validates actions
                raise ModelingError(
                    f"leaf_execution_allocation_action_unknown:{effect.action}")
        for effect in allocation_effects:
            if effect.action != "release":
                continue
            names = {str((record.get("grant") or {}).get("name") or "")
                     for record in bindings.values()}
            if effect.name not in acquired_effect_names and effect.name not in names:
                raise ModelingError(
                    f"leaf_execution_release_without_acquire:{effect.name}")

        # A caller hold named for a grant/output/effect is an external last
        # use, never a fragment-local release shortcut.  Resolve it after
        # ``allocations`` so legacy effect names bind to the same storage.
        unmatched_holds: dict[str, Any] = {}
        for key, value in dict(holds or {}).items():
            target = None
            for output_name in output_root:
                grant = dict(bindings.get(output_name, {}).get("grant") or {})
                names = {output_name, output_root[output_name],
                         str(grant.get("name") or "")}
                if key in names:
                    target = output_name
                    break
            if target is None:
                unmatched_holds[str(key)] = value
            else:
                bindings[target].setdefault("readers", []).append(value)
        if unmatched_holds:
            raise ModelingError(
                "leaf_execution_hold_output_unknown:"
                + sorted(unmatched_holds)[0])

        # Complete the real use set for every declared/borrowed output.
        memory = getattr(execution, "memory", None)
        if memory is None:
            memory = getattr(engine, "memory", None)
        managed: dict[str, dict[str, Any]] = {}
        seen_grant_names: set[str] = set()
        for output_name, record in bindings.items():
            grant = dict(record.get("grant") or {})
            root = output_root[output_name]
            materialized = bool(record.get("materialized", True))
            if output_name in unmaterialized or root in {
                    output_root[item] for item in unmaterialized
                    if item in output_root}:
                materialized = False
            if not grant and not record.get("readers") and materialized:
                continue
            if not materialized and grant:
                raise ModelingError(
                    f"leaf_execution_unmaterialized_output_grant:"
                    f"{output_name}")
            borrow = bool(grant.get("borrow"))
            grant_name = str(grant.get("name") or output_name)
            live_grant = None
            if hasattr(engine, "memory_grant"):
                live_grant = engine.memory_grant(grant_name)
            if live_grant is None and memory is not None:
                try:
                    live_record = memory.grant_records.get(
                        (replica, grant_name))
                except AttributeError:
                    live_record = None
                if live_record is not None:
                    live_grant = {
                        "name": grant_name,
                        "binding": dict(live_record),
                        "ledger": memory,
                    }
            live_binding = dict((live_grant or {}).get("binding") or {})
            if not borrow:
                pool = str(grant.get("pool") or "")
                size = grant.get("size")
                if not pool or size is None:
                    if grant:
                        raise ModelingError(
                            f"leaf_execution_grant_incomplete:{output_name}")
                    continue
            if grant_name in seen_grant_names and root not in managed:
                raise ModelingError(
                    f"leaf_execution_grant_name_duplicate:{grant_name}")
            seen_grant_names.add(grant_name)
            info = roots[root]
            internal = sorted(set(info["read_stages"])
                              | {slot for slot in info["writers"].values()}
                              | {info["first_writer"]})
            readers = tuple(_leaf_event(item) for item in
                            record.get("readers", ()) if item is not None)
            if root in managed and managed[root]["grant_name"] != grant_name:
                raise ModelingError(
                    f"leaf_execution_storage_multiple_grants:{root}:"
                    f"{managed[root]['grant_name']}!={grant_name}")
            if root in managed:
                continue
            managed[root] = {
                "root": root, "outputs": tuple(info["outputs"]),
                "first_writer": info["first_writer"],
                "internal_stages": tuple(internal),
                "readers": readers, "grant": grant,
                "grant_name": grant_name, "borrow": borrow,
                "storage": str(record.get("storage")
                               or live_binding.get("storage")
                               or grant.get("storage") or root),
                "generation": record.get(
                    "generation", live_binding.get(
                        "generation", grant.get("generation"))),
                "ranges": grant.get("ranges", live_binding.get("ranges")),
                "event": grant.get("event", live_binding.get("event")),
                "live_grant": live_grant,
                "expected_binding": {
                    key: value for key, value in {
                        "owner": grant.get("owner") or owner,
                        "storage": str(record.get("storage")
                                       or live_binding.get("storage")
                                       or grant.get("storage") or root),
                        "generation": record.get(
                            "generation", live_binding.get(
                                "generation", grant.get("generation"))),
                        "ranges": grant.get("ranges",
                                            live_binding.get("ranges")),
                        "event": grant.get("event",
                                           live_binding.get("event")),
                    }.items() if value is not None},
                "alias_of": str(record.get("alias_of") or ""),
                "materialized": materialized,
            }

        # A grant name must identify exactly one storage/generation.
        names: dict[str, str] = {}
        for root, item in managed.items():
            previous = names.get(item["grant_name"])
            if previous is not None and previous != root:
                raise ModelingError(
                    f"leaf_execution_grant_storage_conflict:"
                    f"{item['grant_name']}:{previous}!={root}")
            names[item["grant_name"]] = root

        use_events: dict[str, Any] = {
            root: engine.future(f"{name}:{root}:all_uses_done")
            for root in managed}
        borrow_registrations: list[dict[str, Any]] = []
        memory_bindings: dict[str, dict[str, Any]] = {}
        for root, item in managed.items():
            grant = item["grant"]
            if item["borrow"]:
                registration_event = (item.get("event")
                                      or use_events[root])
                registration = {
                    "grant": item["grant_name"], "storage": item["storage"],
                    "generation": item["generation"],
                    "event": (registration_event.name
                              if hasattr(registration_event, "name")
                              else str(registration_event)),
                    "consumer": (f"{name}:{occurrence.name if occurrence else ''}:"
                                 f"{root}:g{item['generation']}:borrow"),
                    "registered": False,
                }
                if memory is not None:
                    try:
                        registration["registered"] = bool(
                            memory.hold_for_async_consumer(
                                replica, item["grant_name"],
                                registration["consumer"],
                                grant=item["grant_name"],
                                generation=item["generation"],
                                event=registration_event,
                                storage=item["storage"],
                                ranges=item.get("ranges")))
                    except (ValueError, TypeError):
                        registration["registered"] = False
                item["registration"] = registration
                borrow_registrations.append(registration)
                continue
            memory_bindings[item["grant_name"]] = {
                key: value for key, value in {
                    "owner": grant.get("owner") or owner,
                    "storage": item["storage"],
                    "generation": item["generation"],
                    "ranges": grant.get("ranges"),
                    "event": grant.get("event"),
                    "host_bytes": grant.get("host_bytes"),
                }.items() if value is not None}

        fragment_allocations: dict[int, list[Any]] = {}
        grant_holds: dict[str, Any] = {}
        for root, item in managed.items():
            if item["borrow"]:
                continue
            grant = item["grant"]
            effect = AllocationEffect(
                "acquire", item["grant_name"], str(grant["pool"]),
                float(grant["size"]),
                tuple(int(unit) for unit in grant.get("units", (0,))),
                str(grant.get("unit") or "bytes"),
                str(grant.get("lifetime") or "stage"))
            fragment_allocations.setdefault(
                item["first_writer"], []).append(effect)
            grant_holds[item["grant_name"]] = use_events[root]

        fragments = tuple(
            replace(fragment,
                    allocations=tuple(fragment.allocations)
                    + tuple(fragment_allocations.get(index, ())))
            for index, fragment in enumerate(fragments))
        if occurrence is None:
            base = fragments[0].occurrence
            occurrence = Occurrence(
                kind=(base.kind if base is not None else "kernel"),
                name=name,
                invocation=(base.invocation if base is not None else "standalone"),
                phase=(base.phase if base is not None else ""))
        if occurrence is not None:
            fragments = tuple(
                replace(
                    fragment,
                    name=f"{name}:{occurrence.name}:stage{index}",
                    occurrence=replace(
                        occurrence,
                        name=f"{occurrence.name}:stage{index}"))
                for index, fragment in enumerate(fragments))
        conductor = execution.fragment_conductor(
            replica=replica, stage=stage,
            pool_bindings=dict(pool_bindings or {}),
            memory_bindings=memory_bindings,
            memory_owner=owner)
        handles = []
        previous = ready
        supplied = {
            key: _leaf_event(value) for key, value in (inputs or {}).items()}
        produced: dict[str, Any] = {}
        for fragment in fragments:
            bindings_for_fragment: dict[str, Any] = {}
            after = [previous]
            for group in ("tensor_in", "state_in", "event_in"):
                for port in getattr(fragment.ports, group):
                    if port.name in produced:
                        binding = produced[port.name]
                    elif port.name in supplied:
                        binding = supplied[port.name]
                    elif group == "event_in":
                        binding = previous
                    else:
                        binding = ready
                    bindings_for_fragment[port.name] = binding
                    if (group in {"tensor_in", "state_in"}
                            and hasattr(binding, "sequence")
                            and hasattr(binding, "done")):
                        # A caller-supplied tensor Event is real data arrival;
                        # it gates this fragment instead of being a dead
                        # binding that the conductor does not schedule on.
                        after.append(binding)
            fragment_holds = {
                key: value for key, value in grant_holds.items()
                if key in {effect.name for effect in fragment.allocations}}
            output_names = {port.name for port in fragment.ports.event_out}
            fragment_bindings = {
                key: value for key, value in (event_bindings or {}).items()
                if key in output_names}
            handle = conductor.submit(
                fragment, after=tuple(dict.fromkeys(after)),
                inputs=bindings_for_fragment,
                holds=fragment_holds, event_bindings=fragment_bindings,
                owner=owner)
            handles.append(handle)
            previous = handle.event
            for port in tuple(fragment.ports.tensor_out) + tuple(
                    port for port in fragment.ports.tensor_in
                    if port.direction == "inout"):
                produced[port.name] = handle.event

        # Establish the connectable last-use edge before/while submitting, then
        # resolve it from the actual internal consumers and external readers.
        use_join_events: dict[str, Any] = {}
        for root, item in managed.items():
            internal_stages = item["internal_stages"]
            internal_events = tuple(
                handles[index].event for index in internal_stages)
            dependencies = tuple(dict.fromkeys(
                internal_events + tuple(item["readers"])))
            use_done = use_events[root]

            def finish_use(_event, *, _root=root, _done=use_done,
                           _item=item):
                for registration in borrow_registrations:
                    if registration.get("grant") != _item["grant_name"]:
                        continue
                    if not registration.get("registered") or memory is None:
                        continue
                    try:
                        memory.release_async_consumer(
                            replica, _item["grant_name"],
                            registration["consumer"],
                            generation=_item["generation"],
                            event=(_item.get("event") or _done))
                    except (ValueError, TypeError):
                        pass
                    registration["released"] = True
                engine.resolve(_done)

            use_join_events[root] = engine.submit(
                f"{name}:{root}:last_use", after=dependencies,
                on_complete=finish_use, owner=owner)

        releases = tuple(
            event for handle in handles for event in handle.release_events)
        output_events: list[tuple[str, Any]] = []
        output_values: list[tuple[str, Any]] = []
        output_records: list[tuple[str, dict[str, Any]]] = []
        storage_records: dict[str, dict[str, Any]] = {}
        for output_name, value in self.typed_values:
            root = output_root.get(output_name)
            if root is None or root not in roots:
                continue
            info = roots[root]
            writer_stage = info["writers"].get(
                output_name, info["first_writer"])
            writer_event = handles[writer_stage].event
            record = bindings.get(output_name, {})
            managed_item = managed.get(root)
            grant = dict((managed_item or {}).get("grant")
                         or record.get("grant") or {})
            grant_name = str((managed_item or {}).get("grant_name")
                             or grant.get("name") or "")
            materialized = bool((managed_item or {}).get(
                "materialized", record.get("materialized", True)))
            if output_name in unmaterialized:
                materialized = False
            storage = str((managed_item or {}).get("storage")
                          or record.get("storage") or grant.get("storage")
                          or root)
            alias_of = str(record.get("alias_of")
                           or aliases.get(output_name, ""))
            if alias_of == storage:
                alias_of = ""
            readers = tuple((managed_item or {}).get("readers", ()))
            use_event = use_events.get(root)
            use_join = use_join_events.get(root)
            entry = {
                "output": output_name, "storage": storage,
                "alias_of": alias_of, "materialized": materialized,
                "writer_stage": writer_stage,
                "writer_event": writer_event.name,
                "consumers": [
                    handles[index].event.name
                    for index in sorted(info["read_stages"])],
                "external_readers": [
                    event.name for event in readers
                    if hasattr(event, "name")],
                "last_use_event": (use_join.name if use_join is not None
                                   else writer_event.name),
                "use_done_event": (use_event.name if use_event is not None
                                   else writer_event.name),
                "grant": grant,
                "grant_name": grant_name,
                "grant_binding": (
                    ((managed_item or {}).get("live_grant") or {}).get(
                        "binding")
                    or (managed_item or {}).get("expected_binding")),
                "borrow": bool((managed_item or {}).get("borrow", False)),
                "binding_name": str(record.get("binding_name") or grant_name
                                    or output_name),
            }
            output_records.append((output_name, entry))
            output_events.append((output_name, writer_event))
            value_binding_name = str(
                record.get("binding_name") or grant_name or "")
            bound_value = record.get("value")
            if not (hasattr(bound_value, "layout")
                    and hasattr(bound_value, "to_dict")):
                bound_value = value
            value = replace(
                bound_value, producer=writer_event.name,
                storage=(storage if materialized else ""),
                alias_of=alias_of,
                binding=("external" if value_binding_name else value.binding),
                binding_name=(value_binding_name if value_binding_name
                              else value.binding_name))
            output_values.append((output_name, value))
            storage_entry = storage_records.setdefault(root, {
                "root": root, "storage": storage, "alias_of": alias_of,
                "first_writer": info["first_writer"],
                "writer_event": writer_event.name,
                "internal_read_stages": sorted(info["read_stages"]),
                "outputs": [], "grant": grant, "grant_name": grant_name,
                "borrow": bool((managed_item or {}).get("borrow", False)),
                "use_done_event": entry["use_done_event"],
                "materialized": materialized,
            })
            storage_entry["outputs"].append(output_name)

        grants = tuple(
            (root, {
                "grant_name": item["grant_name"],
                "borrow": item["borrow"],
                "pool": item["grant"].get("pool"),
                "size": item["grant"].get("size"),
                "storage": item["storage"],
                "generation": item["generation"],
                "first_writer": item["first_writer"],
                "use_done_event": use_events[root].name,
                "registration": item.get("registration"),
            })
            for root, item in managed.items())
        return LeafExecutionHandle(
            event=previous, handles=tuple(handles),
            output_events=tuple(dict(output_events).items()),
            output_values=tuple(output_values),
            output_bindings=tuple(output_records),
            release_events=releases,
            storage_bindings=tuple(storage_records.items()),
            use_events=tuple(
                (root, use_events[root]) for root in managed),
            grants=grants)


def _port_spec(name: str, descriptor: Mapping[str, Any],
               direction: str | None = None) -> Any:
    from mindie_sim.schema.subgraphs import TensorPortSpec
    role = str(descriptor.get("role") or "")
    if direction is None:
        direction = ("in" if role in {"input", "weight"} else
                     "inout" if role in {"inout", "inplace_input_output"}
                     else "out")
    if role == "weight":
        representation = "weight"
    elif name in {"rstd", "mean"}:
        representation = "stats"
    elif name in {"scale", "offset", "amax"}:
        representation = "quant_scale"
    else:
        representation = "activation"
    layout_kind = str(descriptor.get("layout_kind") or (
        "strided" if str(descriptor.get("layout") or "ND").upper() == "ND"
        else "packed"))
    return TensorPortSpec(
        name, direction, representation,
        dtype=str(descriptor.get("dtype") or ""),
        layout_kind=layout_kind)


def _descriptor_layout(name: str, descriptor: Mapping[str, Any]) -> Any:
    from mindie_sim.schema.tensors import TensorLayout
    shape = descriptor.get("view_shape") or descriptor.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ModelingError(f"leaf_execution_descriptor_shape_missing:{name}")
    shape = tuple(int(dim) for dim in shape)
    strides = descriptor.get("view_stride_elements")
    if not isinstance(strides, (list, tuple)) or not strides:
        layout = TensorLayout.contiguous(shape)
        strides = layout.strides
    else:
        strides = tuple(int(item) for item in strides)
    storage_shape = descriptor.get("storage_shape") or shape
    storage_shape = tuple(int(dim) for dim in storage_shape)
    offset = int(descriptor.get("view_offset_elements", 0) or 0)
    return TensorLayout(shape, strides, offset, storage_shape=storage_shape)


def _tensor_value(name: str, descriptor: Mapping[str, Any], *,
                  producer: str, aliases: Mapping[str, str],
                  storage_root: str, input_bindings: Mapping[str, str]) -> Any:
    from mindie_sim.schema.tensors import TensorValue
    role = str(descriptor.get("role") or "")
    external = role in {"input", "weight", "inout", "inplace_input_output"}
    alias = str(aliases.get(name) or "")
    unmaterialized = (role in {"output", "inout", "inplace_input_output"}
                      and descriptor.get("materialized", True) is False)
    storage = "" if unmaterialized else (
        str(descriptor.get("storage") or storage_root or ""))
    binding = "static"
    binding_name = ""
    if external:
        binding = "external"
        binding_name = str(input_bindings.get(name) or name)
    elif alias and storage:
        binding = "external"
        binding_name = str(input_bindings.get(storage_root) or storage)
    return TensorValue(
        producer=producer, dtype=str(descriptor.get("dtype") or ""),
        layout=_descriptor_layout(name, descriptor), output=name,
        layout_kind=("strided" if str(
            descriptor.get("layout") or "ND").upper() == "ND" else "packed"),
        storage=storage, alias_of=alias, binding=binding,
        binding_name=binding_name)


def _leaf_buffer(key: str, descriptor: Mapping[str, Any], *,
                 aliases: Mapping[str, str], root: str) -> dict[str, Any]:
    role = str(descriptor.get("role") or "")
    value_bytes = _descriptor_value_bytes(descriptor)
    storage_bytes = _descriptor_storage_bytes(descriptor)
    alias = str(aliases.get(key) or "")
    if alias:
        allocation = 0
        size = 0
    elif role in {"input", "weight", "inout", "inplace_input_output"}:
        allocation = 0
        size = storage_bytes
    else:
        allocation = value_bytes
        size = value_bytes
    entry = {
        "name": key, "root": root, "pool": "caller",
        "lifetime": "caller", "bytes": size, "allocation": allocation,
        "alias_of": alias,
        "layout": _descriptor_layout(key, descriptor).to_dict(),
    }
    if role in {"input", "weight", "inout", "inplace_input_output"}:
        entry["read_bytes"] = value_bytes
    if descriptor.get("noncontiguous"):
        entry["noncontiguous"] = True
    return entry


def leaf_execution_program(prepared: Any, prediction: Mapping[str, Any], *,
                           name: str = "", native_program: Any = None,
                           buffers: tuple[Any, ...] = (),
                           input_bindings: Mapping[str, str] | None = None
                           ) -> LeafExecutionProgram:
    """Build the executable fragment/native program for one leaf prediction.

    The work comes from the prediction's existing ``stage_work`` (the
    compiler's issued ledger).  Typed ports/values come from the prepared
    descriptors.  No duration is converted back into work.
    """
    from fractions import Fraction
    from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import ProgramFragment, WorkItem, WorkSet
    from mindie_sim.schema.subgraphs import EventPortSpec, PortSet
    from .families_bridge import FamilyFragmentProgram

    if prediction.get("duration_ms") is None:
        raise ModelingError("leaf_execution_program_requires_prediction")
    record = _prepare_record(prepared)
    all_descriptors = dict(record.get("descriptors") or {})
    descriptors = {
        key: item for key, item in all_descriptors.items()
        if item.get("present", True) is not False
        and item.get("shape")}
    if not descriptors:
        raise ModelingError("leaf_execution_program_descriptors_missing")
    raw_stage_work = list(prediction.get("stage_work") or ())
    if not raw_stage_work and prediction.get("work"):
        raw_stage_work = [dict(prediction["work"])]
    if not raw_stage_work:
        raise ModelingError("leaf_execution_program_stage_work_missing")
    stage_total: dict[str, Fraction] = {}
    work_rows: list[WorkSet] = []
    for index, stage in enumerate(raw_stage_work):
        items = []
        for pipe, amount in dict(stage).items():
            value = Fraction(str(amount))
            if value < 0:
                raise ModelingError(
                    f"leaf_execution_program_negative_work:{pipe}")
            if value == 0:
                continue
            stage_total[pipe] = stage_total.get(pipe, Fraction(0)) + value
            items.append(WorkItem(
                str(pipe), float(value),
                NATIVE_WORK_UNITS.get(str(pipe), "work"), "issued",
                f"{record.get('call')}:{name or 'leaf'}:stage{index}:{pipe}"))
        work_rows.append(WorkSet(tuple(items)))
    declared = {str(pipe): Fraction(str(amount))
                for pipe, amount in dict(prediction.get("work") or {}).items()
                if Fraction(str(amount)) > 0}
    if stage_total != declared:
        raise ModelingError(
            "leaf_execution_program_work_mismatch:"
            f"{sorted(stage_total.items())}!={sorted(declared.items())}")

    occurrence = Occurrence(
        kind="kernel", name=name or str(record.get("call") or "leaf"),
        invocation="standalone",
        phase=str(prediction.get("phase") or record.get("attrs", {}).get(
            "phase") or ""))
    input_bindings = dict(input_bindings or {})
    explicit_aliases: dict[str, str] = {}
    for key, descriptor in descriptors.items():
        alias = descriptor.get("alias_of") or descriptor.get("aliases")
        if isinstance(alias, str) and alias in descriptors:
            explicit_aliases[key] = alias
    for key, value in dict(record.get("aliases") or {}).items():
        if not (key in descriptors and isinstance(value, str)
                and value in descriptors):
            continue
        if key in explicit_aliases:
            continue
        if explicit_aliases.get(value) == key:
            continue
        explicit_aliases[key] = value
    aliases: dict[str, str] = {}
    for key in descriptors:
        seen = {key}
        root = key
        while root in explicit_aliases:
            root = explicit_aliases[root]
            if root in seen:
                raise ModelingError(f"leaf_execution_alias_cycle:{key}")
            seen.add(root)
        if root != key:
            aliases[key] = root
    inputs = [key for key, item in descriptors.items()
              if str(item.get("role") or "") in {
                  "input", "inout", "inplace_input_output"}]
    weights = [key for key, item in descriptors.items()
               if str(item.get("role") or "") == "weight"]
    outputs = [key for key, item in descriptors.items()
               if str(item.get("role") or "") == "output"]
    stage0_outputs = [key for key in outputs
                      if key in {"residual_out", "x"} and key not in inputs]
    access_plan = (prediction.get("program") or {}).get("access_plan")
    explicit_access = (prediction.get("program") or {}).get("stage_access")
    stage_io: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    if (isinstance(access_plan, Mapping)
            and access_plan.get("accesses")
            and len(raw_stage_work) == 1):
        # RMSNorm/gated norm publish the same named accesses their predict
        # used; the single stage's ports are those descriptors, not a second
        # port table inferred from role order.
        read_keys = tuple(dict.fromkeys(
            str(item["descriptor"]) for item in access_plan["accesses"]
            if item.get("access") == "read"))
        write_keys = tuple(dict.fromkeys(
            str(item["descriptor"]) for item in access_plan["accesses"]
            if item.get("access") == "write"))
        unknown = (set(read_keys) | set(write_keys)) - set(descriptors)
        if unknown:
            raise ModelingError(
                f"leaf_execution_stage_access_unknown:{sorted(unknown)[0]}")
        stage_io.append((read_keys, write_keys))
    elif explicit_access and len(explicit_access) == len(raw_stage_work):
        for item in explicit_access:
            stage_inputs = tuple(str(key) for key in item.get("inputs", ()))
            stage_outputs = tuple(str(key) for key in item.get("outputs", ()))
            unknown = (set(stage_inputs) | set(stage_outputs)) - set(descriptors)
            if unknown:
                raise ModelingError(
                    f"leaf_execution_stage_access_unknown:{sorted(unknown)[0]}")
            stage_io.append((stage_inputs, stage_outputs))
    else:
        for index in range(len(raw_stage_work)):
            if len(raw_stage_work) == 1:
                stage_io.append((tuple(dict.fromkeys(inputs + weights)),
                                 tuple(outputs)))
            elif index == 0:
                stage_io.append((tuple(inputs), tuple(stage0_outputs)))
            elif index == len(raw_stage_work) - 1:
                stage_io.append((
                    tuple(dict.fromkeys(weights + stage0_outputs)),
                    tuple(key for key in outputs if key not in stage0_outputs)))
            else:
                stage_io.append(((), ()))
    if (isinstance(access_plan, Mapping)
            and access_plan.get("accesses")
            and len(stage_io) == 1):
        # The named per-value accesses are the same ledger predict consumed;
        # stage_io above only turned them into actual typed port names.
        accesses = tuple(dict(item) for item in access_plan["accesses"])
    else:
        accesses = tuple(
            {"stage": index, "inputs": list(stage_inputs),
             "outputs": list(stage_outputs)}
            for index, (stage_inputs, stage_outputs) in enumerate(stage_io))
    fragments = []
    for index, (stage_inputs, stage_outputs) in enumerate(stage_io):
        tensor_in = tuple(_port_spec(
            key, descriptors[key],
            "inout" if str(descriptors[key].get("role") or "") in {
                "inout", "inplace_input_output"} else "in")
            for key in stage_inputs)
        tensor_out = tuple(_port_spec(
            key, descriptors[key],
            "inout" if str(descriptors[key].get("role") or "") in {
                "inout", "inplace_input_output"} else "out")
            for key in stage_outputs)
        ports = PortSet(
            tensor_in=tensor_in, tensor_out=tensor_out,
            event_in=(EventPortSpec(
                f"{name or 'leaf'}:stage{index}:ready", "in", "ready"),),
            event_out=(EventPortSpec(
                f"{name or 'leaf'}:stage{index}:complete", "out", "complete"),))
        fragments.append(ProgramFragment(
            name=f"{name or record.get('call') or 'leaf'}:stage{index}",
            occurrence=Occurrence(
                kind="kernel",
                name=f"{occurrence.name}:stage{index}",
                invocation=occurrence.invocation, phase=occurrence.phase),
            ports=ports, work=work_rows[index],
            notes=("stage_work_from_operator_cost",
                   "outer_execution_owns_agent_and_resources")))
    edges = tuple(() if index == 0 else (index - 1,)
                  for index in range(len(fragments)))
    if native_program is not None:
        program = native_program
    elif prediction.get("native_program_available"):
        raise ModelingError(
            "leaf_execution_program_native_object_required")
    else:
        program = FamilyFragmentProgram(tuple(fragments), edges)
    stage_producers: dict[str, int] = {}
    for index, (_, stage_outputs) in enumerate(stage_io):
        for key in stage_outputs:
            stage_producers[key] = index
    inout_names = [key for key, item in descriptors.items()
                   if str(item.get("role") or "") in {
                       "inout", "inplace_input_output"}]

    def storage_root(key: str) -> str:
        descriptor = descriptors[key]
        if descriptor.get("storage"):
            return str(descriptor["storage"])
        root = aliases.get(key, key)
        return str(descriptors[root].get("storage") or root)

    typed_values = []
    for key, descriptor in descriptors.items():
        role = str(descriptor.get("role") or "")
        external = role in {"input", "weight", "inout",
                            "inplace_input_output"}
        if descriptor.get("producer"):
            producer = str(descriptor["producer"])
        elif external:
            producer = str(input_bindings.get(key) or "external")
        else:
            producer = (f"{occurrence.name}:stage"
                        f"{stage_producers.get(key, 0)}")
        typed_values.append((key, _tensor_value(
            key, descriptor, producer=producer, aliases=aliases,
            storage_root=storage_root(key), input_bindings=input_bindings)))
    unmaterialized = tuple(
        key for key, item in descriptors.items()
        if item.get("materialized", True) is False)
    declared_buffers = tuple(buffers)
    if not declared_buffers:
        declared_buffers = tuple(
            _leaf_buffer(key, descriptors[key], aliases=aliases,
                         root=storage_root(key))
            for key in descriptors
            if not (str(descriptors[key].get("role") or "") in {
                "output", "inout", "inplace_input_output"}
                and descriptors[key].get("materialized", True) is False))
    return LeafExecutionProgram(
        name=name or str(record.get("call") or "leaf"),
        call=str(record.get("call") or ""),
        program=program, fragments=tuple(fragments),
        typed_values=typed_values,
        aliases=tuple(sorted(aliases.items())),
        buffers=declared_buffers,
        stage_work=tuple(dict(stage) for stage in raw_stage_work),
        edges=edges,
        outputs=tuple(dict.fromkeys(outputs + inout_names)),
        unmaterialized=unmaterialized,
        input_bindings=tuple(sorted(input_bindings.items())),
        accesses=accesses,
        access_plan=access_plan,
        abi_handoff=(prediction.get("program") or {}).get("abi_handoff"),
        native_program=native_program,
        native_program_available=bool(
            prediction.get("native_program_available")),
        non_claims=(
            "outer_execution_owns_engine_and_resources",
            "work_from_operator_cost_stage_work",
            "no_duration_to_work_reconstruction",
            "buffers_are_caller_owned_unless_explicitly_declared",
            "no_internal_buffer_acquire",
        ))


def _typed_tensor(descriptor: Mapping[str, Any], *, producer: str,
                  output: str) -> Any:
    from mindie_sim.schema.tensors import TensorLayout, TensorValue
    shape = descriptor.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise NativeVectorExecutionUnavailable("chain_descriptor_shape_missing")
    dtype = str(descriptor.get("dtype") or "")
    if not dtype:
        raise NativeVectorExecutionUnavailable("chain_descriptor_dtype_missing")
    layout = str(descriptor.get("layout") or "ND").upper()
    layout_kind = str(descriptor.get("layout_kind")
                      or ("strided" if layout == "ND" else "packed"))
    return TensorValue(
        producer=str(producer), dtype=dtype,
        layout=TensorLayout.contiguous([int(dim) for dim in shape]),
        output=str(output), layout_kind=layout_kind,
        quant=str(descriptor.get("quant") or ""))


def _require_chain_endpoint(record: Mapping[str, Any], *, role: str) -> None:
    """Require the actual prepare API and its mandatory typed ports."""
    if role == "producer":
        if str(record.get("call") or "") != _ENTRY:
            raise NativeVectorExecutionUnavailable(
                "chain_producer_api_mismatch:"
                + str(record.get("call") or ""))
        if str(record.get("schema") or "") != (
                "mindie-sim/add-rms-norm-native-prepare/1"):
            raise NativeVectorExecutionUnavailable(
                "chain_producer_schema_mismatch:"
                + str(record.get("schema") or ""))
        required = {"x": "input", "residual": "input", "gamma": "weight",
                    "output": "output", "rstd": "output",
                    "residual_out": "output"}
        prefix = "chain_producer_typed_port_missing"
    elif role == "consumer":
        if str(record.get("call") or "") != "torch_npu.npu_dynamic_quant":
            raise NativeVectorExecutionUnavailable(
                "chain_consumer_api_mismatch:"
                + str(record.get("call") or ""))
        if str(record.get("schema") or "") != (
                "mindie-sim/dynamic-quant-native-prepare/1"):
            raise NativeVectorExecutionUnavailable(
                "chain_consumer_schema_mismatch:"
                + str(record.get("schema") or ""))
        required = {"x": "input", "y": "output", "scale": "output",
                    "offset": "output"}
        prefix = "chain_consumer_typed_port_missing"
    else:
        raise ModelingError(f"chain_endpoint_role_invalid:{role}")
    descriptors = record.get("descriptors") or {}
    for name, expected_role in required.items():
        item = descriptors.get(name)
        if (not isinstance(item, Mapping)
                or item.get("role") != expected_role
                or item.get("present", True) is False):
            raise NativeVectorExecutionUnavailable(f"{prefix}:{name}")


def _typed_link(add_descriptor: Mapping[str, Any],
                quant_descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the real norm-output -> quant-x typed connection."""
    from mindie_sim.schema.subgraphs import TensorPortSpec
    if add_descriptor.get("present", True) is False or add_descriptor.get(
            "role") not in {"output", "inout"}:
        raise NativeVectorExecutionUnavailable(
            "chain_typed_mismatch:producer_role")
    if quant_descriptor.get("present", True) is False or quant_descriptor.get(
            "role") != "input":
        raise NativeVectorExecutionUnavailable(
            "chain_typed_mismatch:consumer_role")
    producer = _typed_tensor(add_descriptor, producer="aclnnAddRmsNorm.output",
                             output="norm_output")
    consumer = _typed_tensor(quant_descriptor, producer="aclnnDynamicQuantV2.x",
                             output="x")
    out_port = TensorPortSpec(
        "norm_output", "out", "activation", dtype=producer.dtype,
        layout_kind=producer.layout_kind)
    in_port = TensorPortSpec(
        "x", "in", "activation", dtype=consumer.dtype,
        layout_kind=consumer.layout_kind)
    mismatches = []
    if producer.dtype != consumer.dtype:
        mismatches.append(f"dtype:{producer.dtype}!={consumer.dtype}")
    if producer.layout.shape != consumer.layout.shape:
        mismatches.append(
            f"shape:{list(producer.layout.shape)}!={list(consumer.layout.shape)}")
    if producer.layout_kind != consumer.layout_kind:
        mismatches.append(
            f"layout_kind:{producer.layout_kind}!={consumer.layout_kind}")
    if producer.quant != consumer.quant:
        mismatches.append(f"quant:{producer.quant}!={consumer.quant}")
    if out_port.representation != in_port.representation:
        mismatches.append("port_representation")
    if mismatches:
        raise NativeVectorExecutionUnavailable(
            "chain_typed_mismatch:" + ";".join(mismatches))
    return {
        "schema": "mindie-sim/native-vector-typed-link/1",
        "producer": producer.to_dict(),
        "consumer": consumer.to_dict(),
        "ports": {"producer": out_port.to_dict(),
                  "consumer": in_port.to_dict()},
        "dtype": producer.dtype,
        "shape": list(producer.layout.shape),
        "layout_kind": producer.layout_kind,
    }


def execute_add_rms_norm_quant_chain(
        values: Mapping[str, Any], *, epsilon: float | None = None,
        is_symmetrical: bool | None = True,
        phase: str = "prefill", hardware: Any = None, supply: Any = None,
        predict: bool = True) -> dict[str, Any]:
    """Run the real AddRmsNorm -> DynamicQuantV2 small chain on CPU.

    The dynamic quantizer consumes the norm output by value; the norm's
    residual and rstd external outputs stay available.  Prediction is the
    serial sum of the two common-stage calls, never a fused multiplier.
    """
    np = _numpy()
    if is_symmetrical is not True:
        raise NativeVectorExecutionUnavailable(
            "chain_requires_symmetric_quant")
    if any(name not in values or values.get(name) is None
           for name in ("x", "residual", "gamma")):
        raise NativeVectorExecutionUnavailable("chain_inputs_missing")
    x = np.asarray(values["x"])
    residual = np.asarray(values["residual"])
    gamma = np.asarray(values["gamma"])
    if x.ndim != 2 or residual.shape != x.shape or gamma.shape != (x.shape[-1],):
        raise NativeVectorExecutionUnavailable("chain_shape_invalid")
    add_prepare = prepare_add_rms_norm(
        rows=int(x.shape[0]), hidden=int(x.shape[1]), epsilon=epsilon,
        variant="fused_attention_norm",
        dtype="bf16", phase=phase)
    add_execution = execute_add_rms_norm(
        add_prepare, {"x": x, "residual": residual, "gamma": gamma})
    from . import native_r31_dynamic_quant_v2 as quant
    quant_prepare = quant.prepare_dynamic_quant_v2(
        rows=int(x.shape[0]), width=int(x.shape[1]),
        is_symmetrical=is_symmetrical, source_dtype="bf16",
        output_dtype="int8", phase=phase)
    add_record = _prepare_record(add_prepare)
    quant_record = _prepare_record(quant_prepare)
    _require_chain_endpoint(add_record, role="producer")
    _require_chain_endpoint(quant_record, role="consumer")
    typed_link = _typed_link(add_record["descriptors"]["output"],
                             quant_record["descriptors"]["x"])
    quant_input = add_execution["outputs"]["output"]
    if tuple(quant_input.shape) != tuple(
            quant_record["descriptors"]["x"]["shape"]):
        raise NativeVectorExecutionUnavailable(
            "chain_value_shape_mismatch")
    quant_execution = quant.execute(quant_prepare, {"x": quant_input})
    result: dict[str, Any] = {
        "schema": CHAIN_SCHEMA,
        "chain": ["aclnnAddRmsNorm", "aclnnDynamicQuantV2"],
        "dependency": "quant_x_reads_add_rms_norm_output",
        "phase": phase,
        "values": {
            "norm_output": add_execution["outputs"]["output"],
            "residual_out": add_execution["outputs"]["residual_out"],
            "rstd": add_execution["outputs"]["rstd"],
            "int8": quant_execution["outputs"]["y"],
            "scale": quant_execution["outputs"]["scale"],
            "offset": quant_execution["outputs"]["offset"],
        },
        "add_rms_norm": add_execution,
        "quant": quant_execution,
        "typed_link": typed_link,
        "aliases": {
            "quant.x": "add_rms_norm.output",
            **{f"quant.{key}": value
               for key, value in quant_execution.get("aliases", {}).items()},
        },
        "non_claims": (
            "cpu_reference_chain_not_device_execution",
            "no_fused_timing_multiplier",
            "no_target_duration_injected",
        ),
    }
    result["prediction"] = predict_add_rms_norm_quant_chain(
        add_prepare, quant_prepare, hardware=hardware, supply=supply,
        phase=phase) if predict else None
    return result


def predict_add_rms_norm_quant_chain(
        add_prepared: Any, quant_prepared: Any, *, hardware: Any = None,
        supply: Any = None, policy: Any = None,
        phase: str | None = None) -> dict[str, Any]:
    """Serial common-stage prediction for AddRmsNorm -> DynamicQuantV2."""
    from . import native_r31_dynamic_quant_v2 as quant
    add_record = _prepare_record(add_prepared)
    quant_record = _prepare_record(quant_prepared)
    _require_chain_endpoint(add_record, role="producer")
    _require_chain_endpoint(quant_record, role="consumer")
    typed_link = _typed_link(add_record["descriptors"]["output"],
                             quant_record["descriptors"]["x"])
    add_prediction = predict_add_rms_norm(
        add_prepared, hardware=hardware, supply=supply, policy=policy,
        phase=phase)
    quant_prediction = quant.predict(
        quant_prepared, hardware=hardware, supply=supply, policy=policy,
        phase=phase)
    if quant_prediction.get("duration_ms") is None:
        return {
            "schema": "mindie-sim/native-vector-chain-prediction/1",
            "status": quant_prediction.get("status", "not_predicted"),
            "chain": ["aclnnAddRmsNorm", "aclnnDynamicQuantV2"],
            "dependency": "serial_value_and_scale",
            "duration_ms": None,
            "duration_us": None,
            "calls": [add_prediction, quant_prediction],
            "typed_link": typed_link,
            "reasons": list(quant_prediction.get("reasons") or ()),
            "non_claims": (
                "static_cpu_prediction",
                "no_measured_duration_claim",
                "no_fused_timing_multiplier",
                "no_target_duration_injected",
                "no_symmetric_substitution",
            ),
        }
    from fractions import Fraction
    total = (Fraction(add_prediction["duration_ms"])
             + Fraction(quant_prediction["duration_ms"]))
    return {
        "schema": "mindie-sim/native-vector-chain-prediction/1",
        "chain": ["aclnnAddRmsNorm", "aclnnDynamicQuantV2"],
        "dependency": "serial_value_and_scale",
        "duration_ms": str(total),
        "duration_us": float(total) * 1000.0,
        "calls": [add_prediction, quant_prediction],
        "typed_link": typed_link,
        "dominant_pipe": max(
            (add_prediction, quant_prediction),
            key=lambda item: Fraction(item["duration_ms"]))["dominant_pipe"],
        "basis": "sum_of_common_resource_stage_calls",
        "non_claims": (
            "static_cpu_prediction",
            "no_measured_duration_claim",
            "no_fused_timing_multiplier",
            "no_target_duration_injected",
        ),
    }


#: Uniform leaf entry used by the other native Vector/elementwise adapters.
execute = execute_add_rms_norm
predict = predict_add_rms_norm


def execution_program(prepared: Any, *, hardware: Any = None,
                      supply: Any = None, policy: Any = None,
                      phase: str | None = None, name: str = "add_rmsnorm",
                      buffers: tuple[Any, ...] = ()) -> Any:
    """Return the executable leaf program for the caller's Execution."""
    from .native_add_rms_norm import leaf_execution_program
    prediction = predict_add_rms_norm(
        prepared, hardware=hardware, supply=supply, policy=policy,
        phase=phase)
    return leaf_execution_program(
        prepared, prediction, name=name, buffers=buffers)


def predict(prepared: Any, *, hardware: Any = None, supply: Any = None,
            policy: Any = None, phase: str | None = None) -> dict[str, Any]:
    """Vector-ABI facade over the fused three-output AddRmsNorm program."""
    return predict_add_rms_norm(prepared, hardware=hardware, supply=supply,
                                policy=policy, phase=phase)
