"""Shared parameter validation and explicit SRAM buffer recipes.

[Role]
Provide strict shape/dtype/kwarg parsing plus conservative BufferSpec sets for
vector, matmul, and attention flows so every built-in template declares real
SRAM residency instead of leaving liveness unchecked.
[Invariants]
Every shape dimension is validated individually and must be positive (except a
documented zero-launch no-op). Unknown kwargs, bool counts, fractional counts,
NaN, and Inf fail closed. Buffer byte counts are positive integers.
[Timeouts & Locks]
Pure arithmetic; no waits, locks, I/O, or subprocess calls.
[Side-Effect Policy]
Read-only helpers; no filesystem or device effects.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

from .flow_schema import BufferPool, BufferSpec, FlowSchemaError
from .exact_math import layout_capacity
from .hardware_profiles import normalize_dtype


def as_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise FlowSchemaError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def as_float(name: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FlowSchemaError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise FlowSchemaError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return number


_STANDARD_SHAPE_KEYS = frozenset({
    "tokens", "rows", "seq_len", "hidden", "cols", "n", "m", "k", "seq",
    "kv_len", "batch", "heads", "h", "dim", "d", "tile",
    "base_m", "base_n", "base_k",
})


def check_kwargs(kw: dict[str, Any], allowed: Iterable[str]) -> None:
    """Accept the shared shape vocabulary; reject genuinely unknown keys."""
    if "dtype" in kw:
        normalize_dtype(kw["dtype"])
    unknown = set(kw) - set(allowed) - _STANDARD_SHAPE_KEYS
    if unknown:
        raise FlowSchemaError(f"unknown keyword arguments: {sorted(unknown)}")
    for name in ("atomic", "training", "has_mask", "two_pass", "gqa", "reduce_grad"):
        if name in kw and not isinstance(kw[name], bool):
            raise FlowSchemaError(f"{name} must be a bool")


def dtype_arg(kw, default, allowed=None):
    """Validate an optional dtype against this builder's selected storage ABI."""
    value = normalize_dtype(kw.get("dtype", default))
    if value not in (allowed or (default,)):
        raise FlowSchemaError(f"unsupported dtype {value!r}; expected {allowed or (default,)}")
    return value


def pick_int(kw: dict[str, Any], names: tuple[str, ...], default: int, *, minimum: int = 1) -> int:
    for name in names:
        if name in kw:
            return as_int(name, kw[name], minimum=minimum)
    return default


def pick_float(kw: dict[str, Any], names: tuple[str, ...], default: float, *, minimum: float = 0.0) -> float:
    for name in names:
        if name in kw:
            return as_float(name, kw[name], minimum=minimum)
    return default


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def ub_buffers(tile_elements: int, element_bytes: int = 2, *, lanes: int = 1,
               slots: int = 2) -> tuple[BufferSpec, ...]:
    """Two input/output UB stages plus a reduction scratch per active lane."""
    tile = align_up(max(1, min(tile_elements, 4096)) * element_bytes, 32)
    tile = min(tile, 32768)
    specs: list[BufferSpec] = []
    for lane in range(max(1, min(2, lanes))):
        pool = BufferPool.UB_LANE0 if lane == 0 else BufferPool.UB_LANE1
        specs.append(BufferSpec(f"ub_in_{lane}", pool, tile, slots=slots))
        specs.append(BufferSpec(f"ub_out_{lane}", pool, tile, slots=slots))
        specs.append(BufferSpec(f"ub_scratch_{lane}", pool, 4096, slots=1))
    return tuple(specs)


def cube_buffers(base_m: int, base_n: int, base_k: int, element_bytes: int = 2,
                 *, l0c_bytes: int | None = None) -> tuple[BufferSpec, ...]:
    l0c = max(l0c_bytes or 0, layout_capacity(base_m,base_n,32,"L0C"))
    a_bytes = layout_capacity(base_m,base_k,element_bytes*8,"NZ")
    b_bytes = layout_capacity(base_k,base_n,element_bytes*8,"NZ")
    l0a_slots = 2 if a_bytes * 2 <= 65536 else 1
    l0b_slots = 2 if b_bytes * 2 <= 65536 else 1
    return (
        BufferSpec("l1_a", BufferPool.L1, a_bytes, slots=2),
        BufferSpec("l1_b", BufferPool.L1, b_bytes, slots=2),
        BufferSpec("l0a", BufferPool.L0A, a_bytes, slots=l0a_slots),
        BufferSpec("l0b", BufferPool.L0B, b_bytes, slots=l0b_slots),
        BufferSpec("l0c", BufferPool.L0C, l0c, slots=1),
    )


def attention_buffers(block: int, head_dim: int, lanes: int = 2) -> tuple[BufferSpec, ...]:
    block = max(16, min(block, 128))
    q_bytes = align_up(block * head_dim * 2, 32)
    score_bytes = align_up(block * block * 2, 32)
    specs = list(cube_buffers(block, block, head_dim, 2,
                              l0c_bytes=align_up(block * head_dim * 4, 1024)))
    for lane in range(min(2, lanes)):
        pool = BufferPool.UB_LANE0 if lane == 0 else BufferPool.UB_LANE1
        specs.append(BufferSpec(f"attn_q_{lane}", pool, q_bytes, slots=2))
        specs.append(BufferSpec(f"attn_s_{lane}", pool, score_bytes, slots=2))
    return tuple(specs)


def make_flow(name: str, archetype: Any, total_units: float, base_tile_units: float,
              pipe_work: Iterable[Any], *, params: dict[str, Any] | None = None,
              live_bytes_per_unit: float = 0.0, **kwargs: Any) -> Any:
    """Build one OpFlowTemplate with the shared strict schema."""
    from .flow_schema import OpFlowTemplate

    kwargs.setdefault("estimate_kind", "analytic_prior")
    return OpFlowTemplate(
        name=name, archetype=archetype, total_units=float(total_units),
        base_tile_units=float(base_tile_units),
        live_bytes_per_unit=float(live_bytes_per_unit),
        pipe_work=tuple(pipe_work), params=dict(params or {}), **kwargs,
    )


def conditional_prior(flow):
    """Stamp the underlying route before any public builder or alias returns."""
    from .v4_math import classification

    return replace(flow, estimate_kind="conditional_ledger_required", calibrated=False,
                   params=dict(flow.params, **classification(flow.name), canonical_route=flow.name))
