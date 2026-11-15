"""Prepare/execute R31 standalone DynamicQuantV2 invocation.

[Role] Expand the normal graph's ``*_QuantV2`` producer ahead of each
  ``w8a8_dynamic`` INT8 projection into the actual R31 ACLNN
  ``aclnnDynamicQuantV2`` surface: one BF16 row input, an INT8 payload, and
  the separately materialized FP32 per-row scale and offset outputs.
[Invariants] The R31 image digest is required; the call identity is the
  retained CANN 9.2.0 libopapi host surface, never a model label, profiler
  symbol or kernel-name guess.  M lives only in the occurrence.  The exact
  vllm-ascend W8A8_DYNAMIC Python wrapper is not retained for r31, so the
  caller/shape lowering and ``isSymmetrical`` stay explicit unresolved fields;
  this module never claims a complete native execution.  The offset output is
  retained because the independent profiler contract requires the real
  three-output ABI.  Decode and non-r31 images return ``None`` so ordinary
  prediction stays unchanged.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the selected normal operator
  declaration, the retained R31 host binary and the evaluated geometry.
"""
from __future__ import annotations

from typing import Any
from mindie_sim.schema.subgraphs import SOFTMAX_ATTENTION_KINDS

R31_IMAGE_SHA = "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_IMAGE_PREFIX = "9e869e814846"
R31_REVISION = "r31/9e869e814846"
R31_CANN = "cann-9.2.0-r31"
ENTRY = "torch_npu.npu_dynamic_quant"
ACLNN_API = "aclnnDynamicQuantV2"
KERNEL_NAME = "DynamicQuantV2"
PROFILER_NAME = ("aclnnDynamicQuantV2_DynamicQuantV2_"
                 "DynamicQuantV2")
OP_SURFACE = ("DynamicQuantV2(x, smoothScalesOptional, groupIndexsOptional, "
              "dstType, isSymmetrical, quantMode) -> (y, scale, offset)")
# Retained same-image host side of the ACLNN call.  The device kernel binary
# is not in this bundle; the profiler row proves the measured kernel name and
# the MIX_AIV execution class.
CANNN_LIB = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": ACLNN_API,
    "revision": R31_CANN,
    "state": "task_artifact",
}
# Same-image Python declaration of the torch_npu entry.  This is the API
# declaration only; it is not the model's W8A8_DYNAMIC linear wrapper.
DEVICE_OP = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": "f4392f346816c85f138d7e033a924c6b628ae684cd4fcd7a6d8818518351b5c9",
    "symbol": "BaseDeviceAdaptor.npu_dynamic_quant",
    "revision": R31_REVISION,
    "state": "task_artifact",
    "lines": [316, 327],
    "role": "api_declaration_not_model_caller",
}
# Exact R31 caller/shape lowering is not in the retained source set.  Keep the
# likely site named but not admitted; the status records the proof gap.
PYTHON_CALLER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
    "symbol": "AscendW8A8DynamicLinearMethod.apply",
    "revision": R31_REVISION,
    "status": "not_retained",
    "evidence": ("normal case declares quant_method=w8a8_dynamic and each of "
                 "the 208 int8 projections has one preceding QuantV2 creator"),
}
# Same-family wrapper retained for r35, not admitted as r31 source bytes.
R35_PATTERN = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
    "sha256": "a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910",
    "symbol": "AscendW8A8DynamicLinearMethod.apply",
    "revision": "r35/5461a1a8",
    "status": "same_family_pattern_not_r31_evidence",
}
# Only the structural sites the original r31 capture independently inventories.
_SUPPORTED_SITES = frozenset({
    ("mlp", "dense_mlp", "gate_up_QuantV2"),
    ("mlp", "dense_mlp", "down_QuantV2"),
    *((kind, "attn.qkv", "qkv_projection_QuantV2")
      for kind in SOFTMAX_ATTENTION_KINDS),
    *((kind, "attn.o", "o_projection_QuantV2")
      for kind in SOFTMAX_ATTENTION_KINDS),
    ("gdn_linear_attn", "gdn", "qkvz_projection_QuantV2"),
})
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _r31_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed r31 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    identity = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    digest = identity.split(":", 1)[-1].strip().lower()
    if not digest.startswith(R31_IMAGE_PREFIX):
        return {}
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    if image and "-r31" not in image:
        return {}
    return meta


def _phase_name(compiled: Any, phase: str) -> str:
    return str(phase or getattr(compiled, "phase", "")
               or dict(getattr(compiled, "feature_metadata", ()) or ()).get("phase")
               or "")


def _descriptor(shape: list[int], dtype: str, role: str,
                **extra: Any) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": "ND", "role": role, "pointer": None,
            "allocation": "external", **extra}


def _site(compiled: Any) -> tuple[str, str, str, int] | None:
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if str(getattr(operator, "family", "") or "") != "quantize":
        return None
    theta = operator.theta
    extras = dict(theta.extras)
    kind = str(getattr(compiled, "kind", "") or "")
    category = str(extras.get("category", "") or "")
    name = str(getattr(operator, "name", "") or "")
    if (kind, category, name) not in _SUPPORTED_SITES:
        return None
    if extras.get("quant_op") != "QuantV2":
        return None
    if str(extras.get("out_format", "") or "").lower() != "int8":
        return None
    if str(extras.get("source_format", "") or "").lower() != "bf16":
        return None
    if _int(extras.get("source_bytes"), 0) != 2:
        return None
    if _int(extras.get("row_scale_bytes"), 0) != 4:
        return None
    if _int(extras.get("offset_gm_bytes"), -1) != 0:
        return None
    if _int(extras.get("symmetrical"), 0) != 1:
        return None
    if _int(extras.get("has_smooth"), -1) != 0:
        return None
    if _int(extras.get("quant_passes"), 0) != 2:
        return None
    if str(extras.get("block_axis", "") or "") != "row":
        return None
    width = _int(dict(theta.matrix).get("elements"), 0)
    if width < 1:
        return None
    return kind, category, name, width


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R31 QuantV2 call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Only the prefill QuantV2 guard whose
    normal declaration is an int8 per-token BF16 producer on the witnessed
    r31 image is admitted.  Decode, other images and other quant recipes
    return ``None``.
    """
    if _phase_name(compiled, phase) != "prefill":
        return None
    if not _r31_metadata(compiled):
        return None
    site = _site(compiled)
    if site is None:
        return None
    kind, category, name, width = site
    operator = compiled.operator
    from . import native_calls
    source = native_calls._source(
        CANNN_LIB["repository"], CANNN_LIB["path"],
        sha256=CANNN_LIB["sha256"], symbol=ACLNN_API,
        revision=CANNN_LIB["revision"], state=CANNN_LIB["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="runtime_binary",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        kernel_source_status="kernel_source_not_retained")
    typed = {
        "dtype": "int8",
        "layout": "ND",
        "phase": "prefill",
        "shapes": {"x": [width], "y": [width]},
        "attrs": {
            "acl_api": ACLNN_API,
            "dst_type": "int8",
            "input_arity": 1,
            "is_symmetrical": None,
            "offset_format": "fp32",
            "offset_materialized": True,
            "output_arity": 3,
            "output_format": "int8",
            "outputs": ["y", "scale", "offset"],
            "quant_mode": "pertoken",
            "row_scale_bytes": 4,
            "row_width": width,
            "scale_format": "fp32",
            "source_format": "bf16",
            "source_operation": KERNEL_NAME,
        },
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=R31_REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=(
            "dynamic_quant_python_caller_not_retained",
            "dynamic_quant_kernel_source_not_retained",
            "dynamic_quant_aclnn_c_abi_not_retained",
            "dynamic_quant_binary_selection_unresolved",
            "dynamic_quant_is_symmetrical_not_declared",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R31 QuantV2 creator.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The timing owner is the existing
    ``*_QuantV2`` row itself; the consuming projection is not charged or
    attributed here.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY or api.get("aclnn_symbol") != ACLNN_API:
        return None
    if (binding.source or {}).get("sha256") != CANNN_LIB["sha256"]:
        return None
    meta = _r31_metadata(compiled)
    if not meta:
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("source_operation") != KERNEL_NAME
            or attrs.get("output_format") != "int8"
            or attrs.get("outputs") != ["y", "scale", "offset"]):
        return None
    width = _int(attrs.get("row_width"))
    if width < 1:
        return None
    from mindie_sim.prediction.dynamic_quant_schedule import quant_row_demand
    theta = compiled.operator.theta
    tp = _int(dict(theta.extras).get("tp"), 1)
    demand = quant_row_demand(
        geometry, kind=str(getattr(compiled, "kind", "") or ""), tp=tp)
    rows = int(demand.rows)
    if rows < 1:
        return None
    cache_key = (binding.call_id, rows)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    shapes = {"x": [rows, width], "y": [rows, width],
              "scale": [rows], "offset": [rows]}
    actual = {
        "dtype": "int8",
        "layout": "ND",
        "shapes": shapes,
        "attrs": {**attrs, "tokens": rows, "valid_rows": int(demand.valid),
                  "row_source": str(demand.source)},
    }
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, width], "bf16", "input"),
        "y": _descriptor([rows, width], "int8", "output"),
        "scale": _descriptor([rows], "fp32", "output", per_row=True),
        "offset": _descriptor([rows], "fp32", "output", per_row=True,
                              materialized=True),
    }
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or R31_IMAGE_SHA)
    image = str(meta.get("image") or meta.get("deployment_image") or "")
    projection = str(getattr(compiled.operator, "name", "") or "")
    if projection.endswith("_QuantV2"):
        projection = projection[:-len("_QuantV2")]
    prepare = {
        "schema": "mindie-sim/dynamic-quant-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "native_type": KERNEL_NAME,
        "op_id": binding.op_id,
        "source": dict(binding.source or {}),
        "source_declaration": dict(DEVICE_OP),
        "caller": {**PYTHON_CALLER, "status": "not_retained",
                   "r35_pattern": dict(R35_PATTERN)},
        "kernel": {
            "name": KERNEL_NAME, "role": "vector",
            "source_path": CANNN_LIB["path"],
            "source_sha256": CANNN_LIB["sha256"],
            "evidence": "runtime_binary",
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "kernel_source_status": "kernel_source_not_retained",
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "op_surface": OP_SURFACE,
        "aclnn_c_abi": "not_retained",
        "arguments": {
            "x": "x",
            "smooth_scales": None,
            "group_indexs": None,
            "dst_type": "int8",
            "is_symmetrical": None,
            "quant_mode": "pertoken",
        },
        "argument_order": ["x", "smooth_scales", "group_indexs", "dst_type",
                           "is_symmetrical", "quant_mode"],
        "optional_slots": {"smooth_scales": "not_present",
                           "group_indexs": "not_present"},
        "outputs": ["y", "scale", "offset"],
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "aclnn_symbol": ACLNN_API,
            "call_id": binding.call_id,
            "op_id": binding.op_id,
            "kind": str(getattr(compiled, "kind", "") or ""),
            "projection": projection,
            "tokens": rows,
            "valid_rows": int(demand.valid),
            "row_width": width,
            "source_format": "bf16",
            "output_format": "int8",
            "scale_format": "fp32",
            "offset_format": "fp32",
            "offset_materialized": True,
            "quant_mode": "pertoken",
            "is_symmetrical": None,
            "deployment_image_id": image_id,
            "source_revision": R31_REVISION,
            "source_sha256": CANNN_LIB["sha256"],
            "profiler_name": PROFILER_NAME,
            "kernel": KERNEL_NAME,
            "execution_core": "aiv",
            "python_caller_status": "not_retained",
        },
        "deployment": {
            "image": image,
            "image_id": image_id,
            "revision": R31_REVISION,
            "cann": "9.2.0",
            "host_interface": dict(CANNN_LIB),
            "python_api_declaration": dict(DEVICE_OP),
            "python_caller": {**PYTHON_CALLER, "status": "not_retained",
                              "r35_pattern": dict(R35_PATTERN)},
            "source_selection": (
                "normal quantize declaration quant_op=QuantV2, "
                "out_format=int8, source_format=bf16, row_scale_bytes=4"),
            "shape_lowering": (
                "K from the normal QuantV2 declaration; rows from "
                "dynamic_quant_schedule.quant_row_demand"),
            "attr_selection": (
                "dst_type/quant_mode from captured output contract; "
                "is_symmetrical stays unresolved without the r31 wrapper"),
            "binary_selection": "unresolved",
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "dynamic_quant_python_caller_not_retained",
            "dynamic_quant_is_symmetrical_not_declared",
            "dynamic_quant_kernel_source_not_retained",
            "dynamic_quant_aclnn_c_abi_not_retained",
        ],
        "unresolved": [
            "dynamic_quant_python_caller_not_retained",
            "dynamic_quant_is_symmetrical_not_declared",
            "dynamic_quant_kernel_source_not_retained",
            "dynamic_quant_aclnn_c_abi_not_retained",
            "dynamic_quant_binary_selection_unresolved",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "not_a_complete_native_execution",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": validated.get("dtype"),
            "layout": validated.get("layout"),
            "shapes": dict(validated.get("shapes") or {}),
            "attrs": dict(validated.get("attrs") or {}),
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Independent CPU value contract and common-stage prediction
# ---------------------------------------------------------------------------
EXECUTION_SCHEMA = "mindie-sim/dynamic-quant-v2-execution/1"


def prepare_dynamic_quant_v2(
        *, rows: int, width: int, is_symmetrical: bool | None,
        source_dtype: str = "bf16", output_dtype: str = "int8",
        phase: str = "prefill", call_id: str | None = None) -> dict[str, Any]:
    """Build the real three-output DynamicQuantV2 descriptor without a model.

    ``is_symmetrical=None`` preserves the retained source's unresolved state;
    execution then refuses instead of choosing a mode silently.
    """
    rows, width = int(rows), int(width)
    if rows < 1 or width < 1:
        raise ValueError("dynamic_quant_geometry_invalid")
    if source_dtype not in {"bf16", "fp16"}:
        raise ValueError(f"dynamic_quant_source_dtype_invalid:{source_dtype}")
    if output_dtype != "int8":
        raise ValueError(f"dynamic_quant_output_dtype_invalid:{output_dtype}")
    if is_symmetrical not in (None, True, False):
        raise ValueError("dynamic_quant_is_symmetrical_invalid")
    shapes = {"x": [rows, width], "y": [rows, width],
              "scale": [rows], "offset": [rows]}
    descriptors = {
        "x": _descriptor([rows, width], source_dtype, "input"),
        "y": _descriptor([rows, width], "int8", "output"),
        "scale": _descriptor([rows], "fp32", "output", per_row=True),
        "offset": _descriptor([rows], "fp32", "output", per_row=True,
                              materialized=bool(is_symmetrical is False)),
    }
    attrs = {
        "acl_api": ACLNN_API,
        "dst_type": "int8",
        "input_arity": 1,
        "is_symmetrical": is_symmetrical,
        "offset_format": "fp32",
        "offset_materialized": bool(is_symmetrical is False),
        "output_arity": 3,
        "output_format": "int8",
        "outputs": ["y", "scale", "offset"],
        "quant_mode": "pertoken",
        "row_scale_bytes": 4,
        "row_width": width,
        "rows": rows,
        "scale_format": "fp32",
        "source_format": source_dtype,
        "source_operation": KERNEL_NAME,
        "phase": phase,
        "symmetric_formula": "scale=max(abs(x))/127; offset=0",
        "asymmetric_formula": "not_declared_by_retained_source",
    }
    return {
        "schema": "mindie-sim/dynamic-quant-native-prepare/1",
        "call": ENTRY,
        "call_id": call_id,
        "native_type": KERNEL_NAME,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "shapes": shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "outputs": ["y", "scale", "offset"],
        "output_count": 3,
        "arguments": {"x": "x", "smooth_scales": None,
                      "group_indexs": None, "dst_type": "int8",
                      "is_symmetrical": is_symmetrical,
                      "quant_mode": "pertoken"},
        "attrs": attrs,
        "aliases": {},
        "alias_status": "independent outputs; no input alias",
        "complete": False,
        "partial": True,
        "partial_reasons": ["acl_prepare_only",
                            "cpu_reference_execution_not_device_execution"],
        "non_claims": ["no_device_launch", "no_measured_duration"],
    }


def execute(prepared: Any, values: Any, *,
            out: Any = None) -> dict[str, Any]:
    """Execute symmetric per-token INT8 DynamicQuantV2 on CPU.

    ``scale = max(abs(x), -1) / 127`` and ``y = clip(rint(x / scale),
    -128, 127)`` in FP32, with the payload cast to INT8.  Symmetric mode stores
    no offset, so the ABI's third output slot is returned as an explicit zero
    placeholder marked ``offset_materialized=False``.  Asymmetric mode is not
    implemented because the retained source does not declare its formula.
    """
    from .native_add_rms_norm import (
        NativeVectorExecutionUnavailable, _finish_execution, _input_array,
        _numpy, _prepare_record)
    np = _numpy()
    record = _prepare_record(prepared)
    attrs = dict(record.get("attrs") or {})
    symmetric = attrs.get("is_symmetrical")
    if symmetric is None:
        raise NativeVectorExecutionUnavailable(
            "dynamic_quant_is_symmetrical_required")
    if symmetric is not True:
        raise NativeVectorExecutionUnavailable(
            "dynamic_quant_asymmetric_not_declared")
    out_map = out or {}
    if not bool(attrs.get("offset_materialized", False)):
        if "offset" in out_map:
            # Symmetric mode has no offset write (offset_gm_bytes=0).  Never
            # clear a caller's existing FP32 offset buffer.
            raise NativeVectorExecutionUnavailable(
                "dynamic_quant_symmetric_offset_out_not_writable")
    x = _input_array(record, values, "x").astype(np.float32)
    if x.ndim != 2:
        raise NativeVectorExecutionUnavailable("dynamic_quant_input_rank")
    absmax = np.max(np.abs(x), axis=-1)
    scale = (absmax / 127.0).astype(np.float32)
    safe = scale > 0
    inv = np.zeros_like(scale)
    inv[safe] = 1.0 / scale[safe]
    payload = np.clip(np.rint(x * inv[:, None]), -128.0, 127.0).astype(np.int8)
    offset = np.zeros(int(scale.size), dtype=np.float32)
    return _finish_execution(
        record, {"y": payload, "scale": scale, "offset": offset},
        out=out, materialized={"offset": bool(attrs.get(
            "offset_materialized", False))})


def _prediction_gap(record: dict[str, Any], *, status: str, reason: str,
                    phase: str) -> dict[str, Any]:
    """An explicit unmodeled-mode gap: no duration and no symmetric substitute."""
    attrs = dict(record.get("attrs") or {})
    return {
        "schema": "mindie-sim/native-vector-prediction/1",
        "status": str(status),
        "call": record.get("call") or record.get("native_entry"),
        "rows": int(attrs.get("rows", 0) or 0),
        "family": "quantize",
        "kind": "mlp",
        "phase": str(phase),
        "duration_ms": None,
        "duration_us": None,
        "dominant_pipe": None,
        "tied_pipes": [],
        "work": {},
        "times_ms": {},
        "stage_times_ms": [],
        "materialization": {},
        "derived": {},
        "formula": "",
        "fidelity": "not_predicted",
        "program": {
            "common_program": "quantize.F0.DynamicQuantV2",
            "mode": ("unresolved" if attrs.get("is_symmetrical") is None
                     else "asymmetric" if attrs.get("is_symmetrical") is False
                     else "symmetric"),
            "status": str(status),
            "offset_contract": "external_fp32_offset" if (
                attrs.get("is_symmetrical") is False) else "unresolved",
        },
        "environment": {"basis": "not_evaluated", "arch": "A5",
                        "phase": str(phase)},
        "reasons": [str(reason)],
        "non_claims": (
            "static_cpu_prediction",
            "no_measured_duration_claim",
            "no_target_duration_injected",
            "no_algorithm_fallback",
            "no_symmetric_substitution",
        ),
    }


def predict(prepared: Any, *, hardware: Any = None, supply: Any = None,
            policy: Any = None, phase: str | None = None) -> dict[str, Any]:
    """Price only the declared symmetric DynamicQuantV2 contract.

    ``is_symmetrical=None`` is an insufficient-fields gap; ``False`` is an
    explicit not-implemented gap because the retained source declares an
    external FP32 offset but not its formula.  Neither case is priced as the
    symmetric call.
    """
    from .native_add_rms_norm import _prepare_record, _single_leaf_prediction
    record = _prepare_record(prepared)
    attrs = dict(record.get("attrs") or {})
    shapes = record.get("shapes") or {}
    rows = int((shapes.get("x") or [attrs.get("rows", 0)])[0])
    width = int((shapes.get("x") or [0, attrs.get("row_width", 0)])[-1])
    if rows < 1 or width < 1:
        raise ValueError("dynamic_quant_geometry_invalid")
    runtime_phase = str(phase or attrs.get("phase") or "prefill")
    symmetric = attrs.get("is_symmetrical")
    if symmetric is None:
        return _prediction_gap(
            record, status="insufficient_fields",
            reason="dynamic_quant_is_symmetrical_unresolved",
            phase=runtime_phase)
    if symmetric is not True:
        return _prediction_gap(
            record, status="not_implemented",
            reason="dynamic_quant_asymmetric_offset_mode_not_modeled",
            phase=runtime_phase)
    return _single_leaf_prediction(
        record, family="quantize", name=KERNEL_NAME, role="quant",
        rows=rows, elements=width, ops=3,
        read_bytes=width * 2.0, write_bytes=width + 4.0,
        logical_materialization={
            "y": rows * width, "scale": rows * 4, "offset": 0},
        extras=(("quant_passes", 2), ("out_format", "int8"),
                ("row_scale_bytes", 4), ("scale_format", "fp32"),
                ("symmetrical", 1), ("block_axis", "row"),
                ("quant_op", "QuantV2"), ("source_bytes", 2),
                ("offset_gm_bytes", 0), ("has_smooth", 0)),
        kind="mlp", phase=runtime_phase, hardware=hardware, supply=supply,
        policy=policy)


def execution_program(prepared: Any, *, hardware: Any = None,
                      supply: Any = None, policy: Any = None,
                      phase: str | None = None, name: str = "dynamic_quant_v2",
                      buffers: tuple[Any, ...] = ()) -> Any:
    """Return the executable leaf program; an unmodeled mode stays a gap."""
    from .native_add_rms_norm import leaf_execution_program
    prediction = predict(prepared, hardware=hardware, supply=supply,
                         policy=policy, phase=phase)
    return leaf_execution_program(
        prepared, prediction, name=name, buffers=buffers)
