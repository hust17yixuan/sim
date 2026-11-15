"""Standalone elementwise ACLNN call: Add / Mul / Cast / Gelu / Muls.

[Role]
  Map one captured row-elementwise kernel (two input tensors or one input plus
  a scalar attribute) to the shared family elementwise program.  The physical
  geometry is the declared row count, row width, dtype and the declared number
  of read/write streams; the duration is the selected catalog chain on the
  target die's declared rates and invocation floor.

[Invariants]
  * No shape, request id or measured duration selects a branch.
  * Every byte is a declared tensor stream; a scalar attribute adds no tensor
    read.  The elementwise op count is the declared per-element arithmetic of
    the named primitive and is the same for every shape.
  * The A5/A3/A2 invocation floor comes from ``HardwareCapability`` only.

[Timeouts & Locks]
  Pure bounded arithmetic; no waits or locks.

[Side-Effect Policy]
  Read-only.  No tensor allocation, no device import, no measurement read.
"""
from __future__ import annotations

from math import prod
from typing import Any, Mapping

SCHEMA = "mindie-sim/elementwise-native-prepare/1"

#: Declared per-element arithmetic of each named primitive.  ``Add``/``Mul``
#: are one fused multiply-add style lane op, ``Cast`` is one convert, ``Gelu``
#: is the erf/tanh polynomial, ``Muls`` is one multiply by a scalar.
OPS_PER_ELEMENT = {
    "Add": 3, "Multiply": 3, "Mul": 3, "Muls": 3, "Cast": 2, "Gelu": 11,
    "Silu": 6, "Swish": 6, "AddRmsNorm": 3, "HcPost": 4, "HcPre": 3,
}
DTYPE_BYTES = {"bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
               "fp32": 4, "float32": 4, "int32": 4, "int64": 8,
               "int8": 1, "uint8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}


def _shape_product(shape) -> int:
    return prod(int(item) for item in shape) if shape else 0


def _descriptor(shape, dtype, layout, role):
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout or "ND", "role": role}


def prepare_elementwise(*, name: str = "Add", rows: int | None = None,
                        width: int | None = None,
                        input_shape: Any = None, output_shape: Any = None,
                        dtype: str = "bf16", output_dtype: str | None = None,
                        read_tensors: int = 2, write_tensors: int = 1,
                        broadcast: bool = False,
                        phase: str = "prefill",
                        call_id: str | None = None) -> dict[str, Any]:
    """Build the declared row-elementwise contract without a model node."""
    if input_shape is None:
        if rows is None or width is None:
            raise ValueError("elementwise_geometry_required")
        rows, width = int(rows), int(width)
        input_shape = [rows, width]
        output_shape = [rows, width]
    source_shape = [int(item) for item in input_shape]
    target_shape = [int(item) for item in (output_shape or source_shape)]
    if not source_shape or not target_shape:
        raise ValueError("elementwise_shape_empty")
    if any(item < 1 for item in source_shape + target_shape):
        raise ValueError("elementwise_shape_invalid")
    elements = _shape_product(target_shape)
    width = target_shape[-1]
    rows = elements // width
    output_dtype = str(output_dtype or dtype)
    if dtype not in DTYPE_BYTES or output_dtype not in DTYPE_BYTES:
        raise ValueError(f"elementwise_dtype_invalid:{dtype}/{output_dtype}")
    shapes = {"x1": source_shape, "x2": source_shape,
              "input": source_shape, "output": target_shape}
    descriptors = {
        "x1": _descriptor(source_shape, dtype, "ND", "input"),
        "x2": _descriptor(source_shape, dtype, "ND", "input"),
        "input": _descriptor(source_shape, dtype, "ND", "input"),
        "output": _descriptor(target_shape, output_dtype, "ND", "output"),
    }
    return {
        "schema": SCHEMA,
        "call": "aclnn" + name,
        "call_id": call_id,
        "native_type": name,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "shapes": shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "arguments": {"input": "input", "other": "x2", "output": "output"},
        "attrs": {
            "name": name, "rows": int(rows), "width": int(width),
            "elements": int(elements), "dtype": dtype,
            "output_dtype": output_dtype,
            "dtype_bytes": DTYPE_BYTES[dtype],
            "output_dtype_bytes": DTYPE_BYTES[output_dtype],
            "read_tensors": int(read_tensors),
            "write_tensors": int(write_tensors),
            "broadcast": bool(broadcast),
            "ops_per_element": int(OPS_PER_ELEMENT.get(name, 1)),
            "input_shape": list(source_shape),
            "output_shape": list(target_shape),
            "phase": phase,
        },
        "outputs": {"output": descriptors["output"]},
        "aliases": {},
        "alias_status": "separate output; no input alias",
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
        raise ValueError("elementwise_geometry_missing")
    dtype_bytes = int(attrs.get("dtype_bytes") or 2)
    tensors = int(attrs.get("read_tensors") or 2)
    if attrs.get("broadcast") and rows > 0:
        # One operand is a per-column vector, so its whole stream is read once
        # per block rather than once per row.
        read_bytes = (((tensors - 1) * width * dtype_bytes)
                      + (width * dtype_bytes) / float(rows))
    else:
        read_bytes = float(tensors) * width * dtype_bytes
    write_bytes = (float(attrs.get("write_tensors") or 1) * width
                   * int(attrs.get("output_dtype_bytes") or 2))
    result = _single_leaf_prediction(
        record, family="elementwise", name=str(attrs.get("name") or "Add"),
        role="elementwise", rows=rows, elements=width,
        ops=int(attrs.get("ops_per_element") or 1),
        read_bytes=read_bytes, write_bytes=write_bytes,
        extras=(("read_tensors", int(attrs.get("read_tensors") or 2)),
                ("write_tensors", int(attrs.get("write_tensors") or 1)),
                ("dtype_bytes", int(attrs.get("dtype_bytes") or 2))),
        kind="layernorm", phase=str(phase or attrs.get("phase") or "prefill"),
        hardware=hardware, supply=supply, policy=policy,
        act_bytes=float(attrs.get("dtype_bytes") or 2))
    from .native_clock_physics import (
        aiv_launch_us, declared_block_num, declared_execution_scope,
        rates_from_supply,
    )
    conditions = record.get("request_conditions") or {}
    scope = declared_execution_scope(conditions)
    duration_us = aiv_launch_us(
        kernel_class="elementwise", rows=rows, width=width,
        read_bytes=read_bytes * rows, write_bytes=write_bytes * rows,
        execution_scope=scope, dtype_bytes=dtype_bytes,
        block_num=declared_block_num(conditions),
        vector_ops=float(rows * width * int(attrs.get("ops_per_element") or 3)),
        rates=rates_from_supply(supply, hardware))
    result["duration_ms"] = str(duration_us / 1000.0)
    result["duration_us"] = float(duration_us)
    result["formula"] = "declared_aiv_clock_law"
    return result


def execution_program(prepared: Any, *, hardware: Any = None,
                      supply: Any = None, policy: Any = None,
                      phase: str | None = None, name: str = "elementwise",
                      buffers: tuple[Any, ...] = ()) -> Any:
    from .native_add_rms_norm import leaf_execution_program
    prediction = predict(prepared, hardware=hardware, supply=supply,
                         policy=policy, phase=phase)
    return leaf_execution_program(
        prepared, prediction, name=name, buffers=buffers)
