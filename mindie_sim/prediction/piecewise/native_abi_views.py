"""Timing-free ABI views: inventory, T16 timing-free projection and QBMM ABI map.

[Role] Expose the declared ABI inventory, the timing-free projection that never
  substitutes a target duration, and the 4-slot W8A8 QBMM invocation mapper.
[Invariants] The inventory is a declaration list.  The timing-free view keeps
  work/fidelity separate from any duration.
[Timeouts & Locks] Pure projection; no waits or locks.
[Side-Effect Policy] Read-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import prod
import importlib
import json
import math
from types import SimpleNamespace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_native_contracts import (
    add_rmsnorm_abi_reasons, add_rmsnorm_abi_region,
)
from ..native_binding import (
    NativeAbiRegion, NativeCallBinding, bind_abi_call, join_profiler_rows,
)
from ..runtime.contracts import (
    AllocationBinding, AllocationKey, AllocationSpec, BackingKey, BindingFrame,
    CatalogGeneration, EdgeSpec, OccurrenceKey, PoolKey, PortBinding, PortSpec,
    ProgramCatalog, StageSpec, WorkTerm,
)
from .compiler import ImplementationPolicy, PipeSupply
from .native_add_rms_norm import (
    NativeVectorExecutionUnavailable, prepare_add_rms_norm,
    predict_add_rms_norm,
)
from .native_calls import (
    NativeAbiCallSpec, NativeAbiTensorSpec, a5_native_abi_specs,
    native_abi_inventory, native_abi_spec, validate_native_occurrence,
)
from .native_qbmm import QBMM_SOURCES
from .native_catalog_builder import (
    KernelSupply, StageDraft, catalog_from_drafts, drafts_for,
    drafts_from_work,
)
from .native_occupancy import cacheline_bytes, vector_segments

# Selected cross-module records and helpers (no cycles: see module DAG).
from .native_abi_common import (  # noqa: E402,F401
    ABI_REPLAY_SCHEMA, ABI_REQUEST_SCHEMA, ABI_RESULT_SCHEMA,
    NativeAbiIncomplete, NativeAbiInvocation, NativeAbiMismatch,
    NativeAbiPrediction, NativeAbiTensor, NativeAbiVersionMismatch,
    _ALLOWED_API, _ALLOWED_CONDITIONS, _ALLOWED_REQUEST, _ALLOWED_TENSOR,
    _ALLOWED_VERSION, _DTYPE_ALIASES, _canonical_dtype, _digest_text,
    _execution_machine, _execution_resource_bindings, _full_typed,
    _hardware_execution, _invocation_ports, _invocation_replay,
    _native_cube_rate, _native_pool, _plain, _result, _stage_context, _strict,
)
def native_abi_timing_free(result: NativeAbiPrediction) -> dict[str, Any]:
    """T16 bridge: binding/program/work with no target duration substituted.

    Matrix leaves already produce timing-free work (``duration_ms is None``);
    for timing-capable leaves the measured model remains in ``duration_ms`` but
    is never injected into the work/view.  Callers choose which side to compare.
    """
    prediction = result.prediction or {}
    return {
        "schema": "mindie-sim/native-abi-timing-free/1",
        "status": result.status,
        "binding": result.binding.to_dict() if result.binding is not None else None,
        "program": result.program,
        "work": prediction.get("work"),
        "fidelity": prediction.get("fidelity"),
        "duration_ms": (None if prediction.get("fidelity") == "timing_free_work"
                        else prediction.get("duration_ms")),
        "replay": result.replay,
        "reasons": list(result.reasons),
        "execution": result.execution_receipt,
        "non_claims": ["timing_free_work", "no_target_duration_injected",
                       "no_device_measurement"],
    }


def native_qbmm_w8a8_invocation(
        abi: Mapping[str, Any], *, version: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        conditions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Map the real 4-slot W8A8 QuantBatchMatmul ABI to a T12 invocation.

    Slot order is x, weight (logical or FRACTAL_NZ storage), weight_scale
    (BF16 per-channel), pertoken_scale (FP32 per-token), then output.  The
    actual storage shapes/dtypes/layouts are preserved; no MXFP8 block scale,
    bias, dtype or source revision is invented.  ``version`` is passed through
    unchanged so T02/T16 can supply the real CANN source/image or leave the
    candidate gap explicit.
    """
    inputs = list(abi.get("inputs") or ())
    outputs = list(abi.get("outputs") or ())
    if len(inputs) != 4 or len(outputs) != 1:
        raise ModelingError("native_qbmm_w8a8_slot_count")
    x, weight, weight_scale, pertoken = inputs
    output = outputs[0]

    def slot(row: Mapping[str, Any], name: str, role: str) -> dict[str, Any]:
        present = row.get("present", True) is not False
        return {
            "name": name,
            "shape": list(row.get("shape") or ()) if present else None,
            "dtype": row.get("dtype") if present else None,
            "layout": (row.get("layout") or "ND") if present else "ND",
            "role": role, "present": present,
            "dtype_present": bool(row.get("dtype")) if present else False,
            "layout_present": bool(row.get("layout")) if present else False,
        }

    tensors = {
        "x": slot(x, "x", "input"),
        "weight_nz": slot(weight, "weight_nz", "weight"),
        "weight_scale": slot(weight_scale, "weight_scale", "scale"),
        "pertoken_scale": slot(pertoken, "pertoken_scale", "scale"),
        "output": slot(output, "output", "output"),
    }
    if str(weight.get("layout") or "").upper() != "FRACTAL_NZ":
        tensors["weight"] = tensors.pop("weight_nz")
        tensors["weight"]["name"] = "weight"
    actual_attrs = dict(attrs or {})
    scale_dtype = str(weight_scale.get("dtype") or "").lower()
    if scale_dtype in {"bf16", "bfloat16"}:
        actual_attrs.setdefault("weight_scale_storage_format", "bf16")
    elif scale_dtype in {"fp32", "float32"}:
        actual_attrs.setdefault("weight_scale_storage_format", "fp32")
    return {
        "api": {"namespace": "aclnn",
                "entry": "aclnnQuantMatmulWeightNz",
                "aclnn_symbol": "aclnnQuantMatmulWeightNz"},
        "version": dict(version or {}),
        "tensors": tensors,
        "attrs": actual_attrs,
        "conditions": dict(conditions or {}),
    }

def a5_native_abi_inventory() -> dict[str, Any]:
    """The current A5 independent-ABI catalogue plus its production entry."""
    inventory = native_abi_inventory()
    inventory["prediction_entry"] = (
        "mindie_sim.prediction.piecewise.native_invocation.simulate_aclnn_call")
    inventory["common_program_entry"] = (
        "native_add_rms_norm.predict_add_rms_norm full three-output call")
    inventory["caller_plan_separated"] = True
    from . import native_calls as _catalogue
    add_spec = _catalogue.native_abi_spec("aclnnAddRmsNorm")
    caller_variants = ([variant.to_dict() for variant in add_spec.variants]
                       if add_spec is not None else [])
    inventory["physical_call"] = {
        "leaf": "native_add_rms_norm.predict_add_rms_norm",
        "physical_entry_unique": True,
        "caller_variants": caller_variants,
        "caller_to_leaf_variant": {
            "native_call": "fused_attention_norm",
            "fused_attention_norm": "fused_attention_norm",
            "terminal_norm": "terminal_norm",
        },
        "physical_program_source": (
            "predict_add_rms_norm.program.physical_program"),
        "caller_program_source": (
            "predict_add_rms_norm.program.caller_program"),
        "physical_members_source": (
            "predict_add_rms_norm.program.physical_members"),
        "rstd_charged": True,
        "entries": [
            {"entry": spec.entry, "leaf": spec.leaf,
             "dispatch": spec.dispatch, "implemented": spec.implemented}
            for spec in a5_native_abi_specs() if spec.dispatch
        ],
        "dispatch_modes": sorted({
            spec.dispatch for spec in a5_native_abi_specs() if spec.dispatch
        }),
    }
    return inventory
