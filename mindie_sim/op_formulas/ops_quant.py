"""Quantization storage contracts shared by dense and grouped matrix builders.

[Role]
  Own the explicit storage/scale contract for every supported quant mode:
  activation/weight/output byte widths, the compute tier, activation-scale and
  per-channel/group/MX weight-scale bytes, and the unified dequant/requant
  Vector operation count used by both dense and grouped GEMM.

[Invariants]
  * An unknown or empty quant mode raises ``FormulaValidationError``.
  * Group sizes divide K and MX block sizes are fixed at 32.
  * Sub-byte weights round independently per buffer; scale bytes follow the
    declared mode, never a dense fallback.

[Timeouts & Locks]
  Pure arithmetic; no I/O, locks or device work.

[Side-Effect Policy]
  No mutation; returns fresh frozen ``QuantSpec`` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import FormulaValidationError
from .expr_ast import ZERO, ceil_div
from .hardware import dtype_bytes
from .primitives import strict_int
from .work import Algebra


@dataclass(frozen=True)
class QuantSpec:
    act_bytes: float
    weight_bytes: float
    out_bytes: float
    tier: str
    activation_scale_bytes: int
    weight_scale_mode: str
    group_size: int | None
    scale_elem_bytes: int


def _norm(value: str | None) -> str:
    return "".join(char for char in str(value or "").strip().lower()
                   if char.isalnum())


def _quant_spec(quant: str | None, out_dtype: str | None,
                group_size: int | None, k_dim: int) -> QuantSpec:
    key = _norm(quant)
    if not key:
        raise FormulaValidationError("quant mode is required")
    out_bytes = dtype_bytes(out_dtype) if out_dtype else 2.0
    block = strict_int(group_size, "group_size", minimum=1) if group_size is not None else None
    if key in {"w8a8", "w8a8dynamic", "int8", "int8dynamic"}:
        return QuantSpec(1.0, 1.0, out_bytes, "int8", 4, "per_channel", None, 4)
    if key in {"w4a16", "a16w4"}:
        size = block if block is not None else 128
        _require_group(k_dim, size)
        return QuantSpec(2.0, 0.5, out_bytes, "bf16", 0, "group", size, 4)
    if key in {"w4a8", "w4a4"}:
        size = block if block is not None else 128
        _require_group(k_dim, size)
        return QuantSpec(1.0, 0.5, out_bytes, "int8", 4, "group", size, 4)
    if key in {"mxfp8", "mxfloat8"}:
        if block is not None and block != 32:
            raise FormulaValidationError("mxfp8 fixes the OCP MX block at 32")
        return QuantSpec(1.0, 1.0, out_bytes, "fp8", 0, "mx", 32, 1)
    if key in {"mxfp4", "mxfloat4", "fp4", "nvfp4"}:
        if block is not None and block != 32:
            raise FormulaValidationError("mxfp4 fixes the OCP MX block at 32")
        return QuantSpec(1.0, 0.5, out_bytes, "fp4", 0, "mx", 32, 1)
    if key in {"fp8", "fp8e4m3", "fp8e5m2", "fp8blockwise", "hif8"}:
        size = block if block is not None else 128
        _require_group(k_dim, size)
        return QuantSpec(1.0, 1.0, out_bytes, "fp8", 4, "group", size, 4)
    raise FormulaValidationError(f"unknown quant_mode {quant!r}")


def _require_group(k_dim: int, group_size: int) -> None:
    if group_size < 1:
        raise FormulaValidationError("group_size must be positive")
    if k_dim % group_size:
        raise FormulaValidationError(
            f"group_size {group_size} does not divide K={k_dim}")


def _weight_scale_expr(al: Algebra, k_sym: Any, n_sym: Any, spec: QuantSpec,
                       zero_points: bool) -> tuple[Any, int]:
    """Return ``(scale_bytes_expr, numeric_scale_bytes)`` for one weight."""
    if spec.weight_scale_mode == "per_channel":
        expr = n_sym * 4
        return expr, 0
    group = spec.group_size or 1
    expr = ceil_div(k_sym, group) * n_sym * spec.scale_elem_bytes
    if zero_points and spec.weight_bytes < 2.0:
        expr = expr + ceil_div(k_sym, group) * n_sym * 2
    return expr, 0


_DEQUANT_MODES = frozenset({"w4a16", "a16w4", "w4a8", "w4a4", "mxfp4",
                            "mxfloat4", "fp4", "nvfp4", "w4a4dynamic"})
_KNOWN_QUANT_KEYS = frozenset({
    "w8a8", "w8a8dynamic", "int8", "int8dynamic", "w4a16", "a16w4", "w4a8",
    "w4a4", "mxfp8", "mxfloat8", "mxfp4", "mxfloat4", "fp4", "nvfp4", "fp8",
    "fp8e4m3", "fp8e5m2", "fp8blockwise", "hif8",
})


_QUANTIZED_WEIGHT_DTYPES = frozenset({
    "int4", "uint4", "int8", "uint8", "fp4", "mxfp4", "nvfp4", "fp8",
    "fp8e4m3", "fp8e5m2", "mxfp8", "hif8",
})
_FLOAT_ACTIVATION_DTYPES = frozenset({"bf16", "fp16"})
_INT_ACTIVATION_DTYPES = frozenset({"int8", "uint8", "fp8", "fp8e4m3",
                                    "fp8e5m2", "mxfp8", "hif8"})
_QUANT_STORAGE_RULES = {
    "w8a8": (frozenset({"int8", "uint8", "fp8", "fp8e4m3", "fp8e5m2",
                        "mxfp8", "hif8"}), _INT_ACTIVATION_DTYPES),
    "w8a8dynamic": (frozenset({"int8", "uint8", "fp8", "fp8e4m3",
                               "fp8e5m2", "mxfp8", "hif8"}),
                    _INT_ACTIVATION_DTYPES),
    "w4a16": (frozenset({"int4", "uint4"}), _FLOAT_ACTIVATION_DTYPES),
    "w4a8": (frozenset({"int4", "uint4"}), _INT_ACTIVATION_DTYPES),
    "w4a4": (frozenset({"int4", "uint4"}), _INT_ACTIVATION_DTYPES),
    "mxfp4": (frozenset({"fp4", "mxfp4", "nvfp4"}),
              _FLOAT_ACTIVATION_DTYPES | _INT_ACTIVATION_DTYPES),
    "mxfp8": (frozenset({"fp8", "mxfp8", "hif8"}),
              _FLOAT_ACTIVATION_DTYPES | _INT_ACTIVATION_DTYPES),
    "fp8": (frozenset({"fp8", "fp8e4m3", "fp8e5m2", "hif8"}),
            _FLOAT_ACTIVATION_DTYPES | _INT_ACTIVATION_DTYPES),
}


def is_quantized_dtype(dtype: str | None) -> bool:
    """True when the declared storage width is a quantized weight format."""
    return _norm(dtype) in _QUANTIZED_WEIGHT_DTYPES


def _check_quant_storage(quant_key: str | None, act: str,
                         weight: str | None) -> None:
    """Reject a declared storage pair that the quant contract cannot represent."""
    if quant_key is None or weight is None:
        return
    rule = _QUANT_STORAGE_RULES.get(quant_key)
    if rule is None:
        return
    allowed_weights, allowed_acts = rule
    act_key, weight_key = _norm(act), _norm(weight)
    if weight_key in _FLOAT_ACTIVATION_DTYPES or weight_key in {
            "fp32", "int32", "int64"}:
        raise FormulaValidationError(
            f"quant_mode {quant_key!r} requires a quantized weight dtype, "
            f"got {weight!r}")
    if weight_key not in allowed_weights:
        raise FormulaValidationError(
            f"quant_mode {quant_key!r} is incompatible with weight_dtype "
            f"{weight!r}")
    if act_key not in allowed_acts:
        raise FormulaValidationError(
            f"quant_mode {quant_key!r} is incompatible with activation dtype "
            f"{act!r}")


def _inferred_quant(dtype: str, weight_dtype: str | None) -> str | None:
    """Infer the storage contract from declared activation/weight dtypes."""
    act = _norm(dtype)
    weight = _norm(weight_dtype)
    if not weight:
        return None
    if weight in {"int4", "uint4"}:
        return "w4a8" if act in {"int8", "uint8"} else "w4a16"
    if weight in {"fp4", "mxfp4"}:
        return "mxfp4"
    if weight in {"fp8", "fp8e4m3", "fp8e5m2", "mxfp8"}:
        return "mxfp8"
    if weight in {"int8", "uint8"} or act in {"int8", "uint8"}:
        return "w8a8"
    return None


def _quant_vector_ops(spec: QuantSpec, quant_key: str, *, rows: Any, k_sym: Any,
                      n_sym: Any, weight_batches: Any, bias: bool) -> Any:
    """One unified dense/grouped dequantization and scale contract."""
    if quant_key in _DEQUANT_MODES:
        parts = [weight_batches * k_sym * n_sym, rows * n_sym]
    else:
        # Inputs are already quantized (INT8/FP8); only the scale epilogue runs.
        parts = [rows * n_sym]
    if bias:
        parts.append(rows * n_sym)
    total = ZERO
    for part in parts:
        total = total + part
    return total
