"""Immutable formula-card records and the dual latency-bound contract.

[Role]
  Define the JSON-serializable value objects returned by the formula engine:
  algebraic MAC/FLOP/vector/byte cards, decomposed derating factors, the
  certified floor and the structural reference bound.  Every record is frozen and
  exposes a plain-data ``as_dict`` for CLI/MCP/profiler consumers.

[Invariants]
  * Every numeric work field has a matching ``*_expr`` string rendered from the
    same expression tree; ``card.verify()`` re-parses and re-evaluates all of
    them and raises on any disagreement.
  * ``bindings`` are read-only after construction and always JSON-safe scalars or
    numeric lists (per-expert histograms stay inspectable).
  * ``granule_gm_bytes >= compulsory_gm_bytes >= 0`` and the compulsory
    read/write breakdown is exact.
  * ``LatencyBounds`` enforces ``0 < tau_hw_floor_us <= t_floor_us <= t_ref_us``
    in ``__post_init__``; an invalid triple cannot be constructed.
  * ``measured_us`` is optional; when present it must be finite and >= 0.

[Timeouts & Locks]
  Pure in-memory records; no I/O, locks or sleeps.

[Side-Effect Policy]
  No mutation after construction; ``as_dict`` returns fresh plain containers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .expr_ast import close, evaluate_formula_expr
from .errors import FormulaValidationError

SCHEMA = "mindie-operator-formula/1"


def _freeze(value: Any) -> Any:
    """Recursively freeze mappings to proxies and sequences to tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


@dataclass(frozen=True)
class FormulaCard:
    """Symbolic expression plus its substituted numeric value (same tree)."""

    alg_flops_expr: str
    alg_flops: float
    cube_flops_tiled_expr: str
    cube_flops_tiled: float
    vector_ops_expr: str
    vector_ops: float
    compulsory_gm_bytes_expr: str
    compulsory_gm_bytes: int
    read_bytes_compulsory: int
    write_bytes_compulsory: int
    l2_hot_eligible_bytes: int
    granule_gm_bytes_expr: str
    granule_gm_bytes: int
    cube_macs_expr: str = "0"
    cube_macs: float = 0.0
    compulsory_read_bytes_expr: str = "0"
    compulsory_read_bytes: int = 0
    compulsory_write_bytes_expr: str = "0"
    compulsory_write_bytes: int = 0
    granule_read_bytes_expr: str = "0"
    granule_read_bytes: int = 0
    granule_write_bytes_expr: str = "0"
    granule_write_bytes: int = 0
    comm_bytes_expr: str = "0"
    comm_bytes: int = 0
    comm_ref_bytes_expr: str = "0"
    comm_ref_bytes: int = 0
    bindings: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.compulsory_gm_bytes < 0 or self.granule_gm_bytes < 0:
            raise FormulaValidationError("formula card byte counts must be non-negative")
        if self.granule_gm_bytes < self.compulsory_gm_bytes:
            raise FormulaValidationError(
                "granule_gm_bytes must be >= compulsory_gm_bytes")
        if self.read_bytes_compulsory + self.write_bytes_compulsory != self.compulsory_gm_bytes:
            raise FormulaValidationError(
                "compulsory breakdown must sum to compulsory_gm_bytes")
        if not 0 <= self.l2_hot_eligible_bytes <= self.compulsory_gm_bytes:
            raise FormulaValidationError(
                "l2_hot_eligible_bytes must be within compulsory bytes")
        object.__setattr__(self, "bindings", _freeze(self.bindings))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def expr_pairs(self) -> tuple[tuple[str, str, Any], ...]:
        """Ordered ``(field, rendered_expr, numeric)`` triples."""
        return (
            ("cube_macs", self.cube_macs_expr, self.cube_macs),
            ("alg_flops", self.alg_flops_expr, self.alg_flops),
            ("cube_flops_tiled", self.cube_flops_tiled_expr, self.cube_flops_tiled),
            ("vector_ops", self.vector_ops_expr, self.vector_ops),
            ("compulsory_read_bytes", self.compulsory_read_bytes_expr,
             self.compulsory_read_bytes),
            ("compulsory_write_bytes", self.compulsory_write_bytes_expr,
             self.compulsory_write_bytes),
            ("compulsory_gm_bytes", self.compulsory_gm_bytes_expr,
             self.compulsory_gm_bytes),
            ("granule_read_bytes", self.granule_read_bytes_expr,
             self.granule_read_bytes),
            ("granule_write_bytes", self.granule_write_bytes_expr,
             self.granule_write_bytes),
            ("granule_gm_bytes", self.granule_gm_bytes_expr,
             self.granule_gm_bytes),
            ("comm_bytes", self.comm_bytes_expr, self.comm_bytes),
            ("comm_ref_bytes", self.comm_ref_bytes_expr, self.comm_ref_bytes),
        )

    def verify(self) -> None:
        """Raise unless every rendered expression evaluates to its number."""
        for name, text, number in self.expr_pairs():
            if not isinstance(text, str) or not text.strip():
                raise FormulaValidationError(f"{name}_expr is empty")
            try:
                value = evaluate_formula_expr(text, self.bindings)
            except FormulaValidationError as exc:
                raise FormulaValidationError(
                    f"{name}_expr failed to evaluate: {text!r}: {exc}") from exc
            if not close(value, number):
                raise FormulaValidationError(
                    f"{name}: expr {text!r} -> {value!r} != card value {number!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "alg_flops_expr": self.alg_flops_expr,
            "alg_flops": self.alg_flops,
            "cube_macs_expr": self.cube_macs_expr,
            "cube_macs": self.cube_macs,
            "cube_flops_tiled_expr": self.cube_flops_tiled_expr,
            "cube_flops_tiled": self.cube_flops_tiled,
            "vector_ops_expr": self.vector_ops_expr,
            "vector_ops": self.vector_ops,
            "compulsory_gm_bytes_expr": self.compulsory_gm_bytes_expr,
            "compulsory_gm_bytes": self.compulsory_gm_bytes,
            "compulsory_read_bytes_expr": self.compulsory_read_bytes_expr,
            "compulsory_read_bytes": self.compulsory_read_bytes,
            "compulsory_write_bytes_expr": self.compulsory_write_bytes_expr,
            "compulsory_write_bytes": self.compulsory_write_bytes,
            "read_bytes_compulsory": self.read_bytes_compulsory,
            "write_bytes_compulsory": self.write_bytes_compulsory,
            "l2_hot_eligible_bytes": self.l2_hot_eligible_bytes,
            "granule_gm_bytes_expr": self.granule_gm_bytes_expr,
            "granule_gm_bytes": self.granule_gm_bytes,
            "granule_read_bytes_expr": self.granule_read_bytes_expr,
            "granule_read_bytes": self.granule_read_bytes,
            "granule_write_bytes_expr": self.granule_write_bytes_expr,
            "granule_write_bytes": self.granule_write_bytes,
            "comm_bytes_expr": self.comm_bytes_expr,
            "comm_bytes": self.comm_bytes,
            "comm_ref_bytes_expr": self.comm_ref_bytes_expr,
            "comm_ref_bytes": self.comm_ref_bytes,
            "bindings": _plain(self.bindings),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DeratingBreakdown:
    """Every factor that turns a peak rate into a structural reference."""

    eta_asymp_cube: float
    eta_asymp_vector: float
    eta_asymp_bw: float
    n_tiles: int
    n_cores: int
    n_waves: int
    wave_efficiency: float
    bw_wave_efficiency: float
    coalesce_efficiency: float
    cube_geom_efficiency: float
    vector_geom_efficiency: float
    bw_geom_efficiency: float
    m_tiled: int
    n_tiled: int
    k_tiled: int
    tile_m: int
    tile_n: int
    tile_k: int
    hbm_bytes: int
    l2_bytes: int
    l2_hot_budget_bytes: int
    l2_peak_bps: float
    capacity_reload_factor: float

    def __post_init__(self) -> None:
        for name in ("eta_asymp_cube", "eta_asymp_vector", "eta_asymp_bw",
                     "wave_efficiency", "bw_wave_efficiency", "coalesce_efficiency",
                     "cube_geom_efficiency", "vector_geom_efficiency",
                     "bw_geom_efficiency"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise FormulaValidationError(f"{name} must be in (0, 1], got {value}")

    def as_dict(self) -> dict[str, Any]:
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class LatencyBounds:
    """The certified floor plus the explainable structural reference."""

    tau_hw_floor_us: float
    tau_launch_us: float
    tau_epilogue_us: float
    cube_floor_us: float
    vector_floor_us: float
    gm_floor_us: float
    comm_floor_us: float
    t_floor_us: float
    cube_ref_us: float
    vector_ref_us: float
    bw_ref_us: float
    comm_ref_us: float
    t_ref_us: float
    bottleneck_pipe: str

    def __post_init__(self) -> None:
        if not (self.tau_hw_floor_us > 0):
            raise FormulaValidationError("tau_hw_floor_us must be > 0")
        if not (self.tau_hw_floor_us <= self.t_floor_us + 1e-12):
            raise FormulaValidationError("t_floor_us must be >= tau_hw_floor_us")
        if not (self.t_floor_us <= self.t_ref_us + 1e-12):
            raise FormulaValidationError("t_ref_us must be >= t_floor_us")

    def as_dict(self) -> dict[str, Any]:
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class OperatorFormulaEvaluation:
    """One operator's complete, immutable, JSON-serializable evaluation."""

    operator: str
    variant: str
    family: str
    sku: str
    arch: str
    launch_mode: str
    steady_l2_warm: bool
    formula_card: FormulaCard
    derating: DeratingBreakdown
    latency: LatencyBounds
    assumptions: tuple[str, ...] = ()
    measured_us: float | None = None
    sol_floor_pct: float | None = None
    ref_ratio: float | None = None
    structural_gap_us: float | None = None
    impl_headroom_us: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))
        if self.measured_us is not None:
            number = float(self.measured_us)
            if not math.isfinite(number) or number < 0:
                raise FormulaValidationError("measured_us must be finite and >= 0")
            object.__setattr__(self, "measured_us", number)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "operator": self.operator,
            "variant": self.variant,
            "family": self.family,
            "sku": self.sku,
            "arch": self.arch,
            "launch_mode": self.launch_mode,
            "steady_l2_warm": self.steady_l2_warm,
            "formula_card": self.formula_card.as_dict(),
            "derating": self.derating.as_dict(),
            "latency": self.latency.as_dict(),
            "assumptions": list(self.assumptions),
            "measured_us": self.measured_us,
            "sol_floor_pct": self.sol_floor_pct,
            "ref_ratio": self.ref_ratio,
            "structural_gap_us": self.structural_gap_us,
            "impl_headroom_us": self.impl_headroom_us,
        }
        return _plain(payload)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True,
                          indent=indent, allow_nan=False)

    @property
    def t_floor_us(self) -> float:
        return self.latency.t_floor_us

    @property
    def t_ref_us(self) -> float:
        return self.latency.t_ref_us

    @property
    def bottleneck_pipe(self) -> str:
        return self.latency.bottleneck_pipe

    @property
    def bindings(self) -> Mapping[str, Any]:
        return self.formula_card.bindings


__all__ = [
    "SCHEMA", "DeratingBreakdown", "FormulaCard", "LatencyBounds",
    "OperatorFormulaEvaluation",
]
