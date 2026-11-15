"""Standalone ``DynamicMxQuant`` row quantization.

[Role]
  Map a captured ``DynamicMxQuant`` kernel to the declared row work: one BF16
  activation read, one FP8 payload write and the E8M0 block-scale write.  The
  block geometry is the declared 32-element MX group, so a 64-column pair of
  E8M0 lanes is one scale group of two blocks.

[Invariants]
  * The row count and width come only from the declared ABI shapes.
  * The scale stream is ``rows * ceil(width / 64) * 2`` bytes, derived from
    the group size, not from a shape table.
  * The duration is the selected family program on the declared hardware
    rates and invocation floor.

[Timeouts & Locks]
  Pure bounded arithmetic; no waits or locks.

[Side-Effect Policy]
  Read-only.  No allocation, device import or measurement access.
"""
from __future__ import annotations

from math import ceil, prod
from typing import Any

SCHEMA = "mindie-sim/mx-quant-native-prepare/1"
GROUP = 32
PAIR = 64
OPS_PER_ELEMENT = 5


def _descriptor(shape, dtype, role):
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": "ND", "role": role}


def prepare_mx_quant(*, rows: int | None = None, width: int | None = None,
                     input_shape: Any = None, output_shape: Any = None,
                     scale_shape: Any = None, dtype: str = "bf16",
                     output_dtype: str = "fp8_e4m3",
                     phase: str = "prefill",
                     call_id: str | None = None) -> dict[str, Any]:
    if input_shape is None:
        if rows is None or width is None:
            raise ValueError("mx_quant_geometry_required")
        input_shape = [int(rows), int(width)]
        output_shape = list(input_shape)
    source = [int(item) for item in input_shape]
    target = [int(item) for item in (output_shape or source)]
    if not source or source != target:
        raise ValueError("mx_quant_output_shape_mismatch")
    rows, width = prod(target[:-1]) if len(target) > 1 else 1, target[-1]
    expected_scale = [int(rows), int(ceil(width / PAIR)), 2]
    if scale_shape is not None and [int(x) for x in scale_shape] != expected_scale:
        raise ValueError("mx_quant_scale_shape_mismatch")
    descriptors = {
        "x": _descriptor(source, dtype, "input"),
        "y": _descriptor(target, output_dtype, "output"),
        "scale": _descriptor(expected_scale, "fp8_e8m0", "output"),
    }
    return {
        "schema": SCHEMA,
        "call": "aclnnDynamicMxQuant",
        "call_id": call_id,
        "native_type": "DynamicMxQuant",
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "shapes": {"x": source, "y": target, "scale": expected_scale},
        "descriptors": descriptors,
        "arrays": descriptors,
        "arguments": {"x": "x", "y": "y", "scale": "scale"},
        "attrs": {
            "rows": int(rows), "width": int(width),
            "elements": int(rows) * int(width),
            "dtype": dtype, "output_dtype": output_dtype,
            "group_size": GROUP, "scale_group": PAIR,
            "input_shape": source, "output_shape": target,
            "scale_shape": expected_scale, "phase": phase,
        },
        "outputs": {"y": descriptors["y"], "scale": descriptors["scale"]},
        "aliases": {},
        "alias_status": "separate payload and scale outputs",
        "complete": False,
        "partial": True,
        "partial_reasons": ["acl_prepare_only",
                            "cpu_reference_execution_not_device_execution"],
        "non_claims": ["no_device_launch", "no_measured_duration"],
    }


def predict(prepared: Any, *, hardware: Any = None, supply: Any = None,
            policy: Any = None, phase: str | None = None) -> dict[str, Any]:
    from .native_add_rms_norm import _prepare_record, _single_leaf_prediction
    record = _prepare_record(prepared)
    attrs = dict(record.get("attrs") or {})
    rows = int(attrs.get("rows") or 0)
    width = int(attrs.get("width") or 0)
    if rows < 1 or width < 1:
        raise ValueError("mx_quant_geometry_missing")
    read_bytes = float(width) * 2.0
    scale_bytes = float(ceil(width / PAIR) * 2)
    write_bytes = float(width) + scale_bytes
    result = _single_leaf_prediction(
        record, family="quantize", name="DynamicMxQuant",
        role="quantize", rows=rows, elements=width,
        ops=OPS_PER_ELEMENT, read_bytes=read_bytes,
        write_bytes=write_bytes,
        extras=(("group_size", GROUP), ("scale_group", PAIR),
                ("scale_bytes_per_row", scale_bytes)),
        kind="layernorm", phase=str(phase or attrs.get("phase") or "prefill"),
        hardware=hardware, supply=supply, policy=policy, act_bytes=2.0)
    from .native_clock_physics import (
        aiv_launch_us, declared_block_num, declared_execution_scope,
        rates_from_supply,
    )
    conditions = record.get("request_conditions") or {}
    scope = declared_execution_scope(conditions)
    duration_us = aiv_launch_us(
        kernel_class="quantize", rows=rows, width=width,
        read_bytes=read_bytes * rows, write_bytes=write_bytes * rows,
        execution_scope=scope, dtype_bytes=2,
        block_num=declared_block_num(conditions),
        vector_ops=float(rows * width * OPS_PER_ELEMENT),
        rates=rates_from_supply(supply, hardware))
    result["duration_ms"] = str(duration_us / 1000.0)
    result["duration_us"] = float(duration_us)
    result["formula"] = "declared_aiv_clock_law"
    return result


def execution_program(prepared: Any, *, hardware: Any = None,
                      supply: Any = None, policy: Any = None,
                      phase: str | None = None, name: str = "mx_quant",
                      buffers: tuple[Any, ...] = ()) -> Any:
    from .native_add_rms_norm import leaf_execution_program
    prediction = predict(prepared, hardware=hardware, supply=supply,
                         policy=policy, phase=phase)
    return leaf_execution_program(
        prepared, prediction, name=name, buffers=buffers)
