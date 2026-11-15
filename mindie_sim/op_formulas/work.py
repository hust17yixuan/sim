"""Intermediate per-operator work record built from typed expressions.

[Role]
  Decouple operator builders (which know the algebra and the ABI) from the
  latency engine (which knows SKU peaks, launch classes and efficiency
  decomposition).  Every work/byte quantity is an ``Expr``: its numeric value is
  *derived* by evaluating that exact tree over ``bindings``, so there is no
  second hand-written number to contradict the rendered formula.

[Invariants]
  * ``cube_macs``, ``cube_flops_tiled``, ``vector_ops`` and every byte quantity
    are expression trees; no independent numeric field exists for them.
  * A ``structure="vector_only"`` (or ``comm_only``) record has exactly zero
    Cube MACs and zero tiled Cube FLOPs; constructing anything else raises.
  * ``granule_*`` is never smaller than ``compulsory_*`` for a built record; a
    builder that violates it fails loudly rather than being clamped.
  * ``l2_hot_bytes`` never exceeds compulsory bytes; the cap is an explicit
    ``min(...)`` in the tree, not a silent post-hoc number.
  * ``bindings`` are immutable, JSON-safe scalars or numeric lists.

[Timeouts & Locks]
  Pure in-memory record; evaluation is linear in expression size.

[Side-Effect Policy]
  No mutation of shared inputs; derived values are computed on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from .errors import FormulaError
from .expr_ast import (
    Const, Expr, ListSym, Sym, ZERO, as_expr, evaluate_formula_expr, min_expr,
)
from .primitives import strict_finite_float, strict_int, strict_pos_int


def _binding_value(value: Any, name: str) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise FormulaError(f"binding {name!r} must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise FormulaError(f"binding {name!r} list must be numeric")
            out.append(item)
        return tuple(out)
    raise FormulaError(f"binding {name!r} is not JSON-safe: {type(value).__name__}")


def freeze_bindings(bindings: Mapping[str, Any]) -> Mapping[str, Any]:
    clean = {str(key): _binding_value(value, str(key))
             for key, value in bindings.items()}
    return MappingProxyType(clean)


@dataclass(frozen=True)
class OpWork:
    """One operator's typed work algebra, before SKU-specific pricing."""

    family: str
    variant: str = "default"
    cube_macs: Expr = ZERO
    cube_flops_tiled: Expr = ZERO
    vector_ops: Expr = ZERO
    read_bytes: Expr = ZERO
    write_bytes: Expr = ZERO
    granule_read_bytes: Expr = ZERO
    granule_write_bytes: Expr = ZERO
    comm_bytes: Expr = ZERO
    comm_ref_bytes: Expr = ZERO
    l2_hot_bytes: Expr = ZERO
    n_tiles: int = 1
    n_cores: int = 0
    coalesce_efficiency: float = 1.0
    m_tiled: int = 0
    n_tiled: int = 0
    k_tiled: int = 0
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    comm_bw_bps: float = 0.0
    comm_alpha_us: float = 0.0
    comm_measured_fit: tuple[float, float] | None = None
    capacity_reload_factor: float = 1.0
    tau_epilogue_us: float = 0.0
    extra_host_us: float = 0.0
    cube_tier: str = "bf16"
    structure: str = "compute"
    alg_override: Expr | None = None
    vector_ref_ops: Expr | None = None
    notes: tuple[str, ...] = ()
    bindings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            raise FormulaError("family must be non-empty")
        for name in ("cube_macs", "cube_flops_tiled", "vector_ops", "read_bytes",
                     "write_bytes", "granule_read_bytes", "granule_write_bytes",
                     "comm_bytes", "comm_ref_bytes", "l2_hot_bytes"):
            if not isinstance(getattr(self, name), Expr):
                raise FormulaError(f"{name} must be an Expr, not a bare number")
        for name in ("alg_override", "vector_ref_ops"):
            item = getattr(self, name)
            if item is not None and not isinstance(item, Expr):
                raise FormulaError(f"{name} must be an Expr or None")
        strict_pos_int(self.n_tiles, "n_tiles")
        strict_int(self.n_cores, "n_cores", minimum=0)
        strict_finite_float(self.coalesce_efficiency, "coalesce_efficiency",
                            minimum=0.0, maximum=1.0, exclusive_min=True)
        strict_finite_float(self.capacity_reload_factor, "capacity_reload_factor",
                            minimum=1.0)
        strict_finite_float(self.tau_epilogue_us, "tau_epilogue_us", minimum=0.0)
        strict_finite_float(self.extra_host_us, "extra_host_us", minimum=0.0)
        if self.comm_bw_bps < 0:
            raise FormulaError("comm_bw_bps must be >= 0")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        object.__setattr__(self, "bindings", freeze_bindings(self.bindings))
        if self.structure in {"vector_only", "comm_only", "sampler"}:
            macs = self.cube_macs.value(self.bindings)
            tiled = self.cube_flops_tiled.value(self.bindings)
            if macs != 0 or tiled != 0:
                raise FormulaError(
                    f"{self.family}: {self.structure} operator must have zero "
                    f"Cube work, got macs={macs} tiled={tiled}")

    # ------------------------------------------------------------------ #
    # Derived algebra (single source of truth)
    # ------------------------------------------------------------------ #
    @property
    def alg_flops(self) -> Expr:
        if self.alg_override is not None:
            return self.alg_override
        if isinstance(self.cube_macs, Const) and self.cube_macs.raw == 0:
            return ZERO
        return 2 * self.cube_macs

    @property
    def reference_vector_ops(self) -> Expr:
        return self.vector_ops if self.vector_ref_ops is None else self.vector_ref_ops

    @property
    def compulsory_bytes(self) -> Expr:
        return self.read_bytes + self.write_bytes

    @property
    def granule_bytes(self) -> Expr:
        return self.granule_read_bytes + self.granule_write_bytes

    def capped(self) -> "OpWork":
        """Cap L2-hot bytes at compulsory bytes with an explicit ``min`` node."""
        if not self.l2_hot_bytes.syms():
            hot = int(self.l2_hot_bytes.value({}))
            total = int(self.compulsory_bytes.value(self.bindings))
            if hot <= total:
                return self
        return replace(self, l2_hot_bytes=min_expr(self.l2_hot_bytes,
                                                   self.compulsory_bytes))

    def values(self) -> dict[str, Any]:
        """Evaluate every work/byte quantity from its own AST."""
        bindings = self.bindings
        alg = self.alg_flops.value(bindings)
        return {
            "cube_macs": self.cube_macs.value(bindings),
            "alg_flops": alg,
            "cube_flops_tiled": self.cube_flops_tiled.value(bindings),
            "vector_ops": self.vector_ops.value(bindings),
            "vector_ref_ops": self.reference_vector_ops.value(bindings),
            "read_bytes": self.read_bytes.value(bindings),
            "write_bytes": self.write_bytes.value(bindings),
            "compulsory_gm_bytes": self.compulsory_bytes.value(bindings),
            "granule_read_bytes": self.granule_read_bytes.value(bindings),
            "granule_write_bytes": self.granule_write_bytes.value(bindings),
            "granule_gm_bytes": self.granule_bytes.value(bindings),
            "comm_bytes": self.comm_bytes.value(bindings),
            "comm_ref_bytes": self.comm_ref_bytes.value(bindings),
            "l2_hot_bytes": self.l2_hot_bytes.value(bindings),
        }

    def expr_fields(self) -> tuple[tuple[str, Expr], ...]:
        """Ordered ``(field, tree)`` pairs verified by the card builder."""
        return (
            ("cube_macs", self.cube_macs),
            ("alg_flops", self.alg_flops),
            ("cube_flops_tiled", self.cube_flops_tiled),
            ("vector_ops", self.vector_ops),
            ("vector_ref_ops", self.reference_vector_ops),
            ("compulsory_read_bytes", self.read_bytes),
            ("compulsory_write_bytes", self.write_bytes),
            ("compulsory_gm_bytes", self.compulsory_bytes),
            ("granule_read_bytes", self.granule_read_bytes),
            ("granule_write_bytes", self.granule_write_bytes),
            ("granule_gm_bytes", self.granule_bytes),
            ("comm_bytes", self.comm_bytes),
            ("comm_ref_bytes", self.comm_ref_bytes),
        )

    def self_check(self) -> None:
        """Verify render -> parse -> evaluate agrees for every quantity."""
        values = self.values()
        for name, tree in self.expr_fields():
            rendered = tree.render()
            reparsed = evaluate_formula_expr(rendered, self.bindings)
            direct = tree.value(self.bindings)
            if abs(float(reparsed) - float(direct)) > 1e-9 * max(
                    1.0, abs(float(direct))):
                raise FormulaError(
                    f"{self.family}: {name} render/eval mismatch "
                    f"{rendered!r}: {reparsed} != {direct}")
            if name in values and abs(float(values[name]) - float(direct)) > 1e-9 * max(
                    1.0, abs(float(direct))):
                raise FormulaError(f"{self.family}: {name} value mismatch")
        if int(values["granule_gm_bytes"]) < int(values["compulsory_gm_bytes"]):
            raise FormulaError(
                f"{self.family}: granule bytes {values['granule_gm_bytes']} < "
                f"compulsory {values['compulsory_gm_bytes']}")
        if int(values["l2_hot_bytes"]) > int(values["compulsory_gm_bytes"]):
            raise FormulaError(f"{self.family}: l2_hot exceeds compulsory bytes")

    def prefixed(self, prefix: str) -> "OpWork":
        """Return a copy with every symbol and binding qualified by ``prefix``."""
        mapping = {name: f"{prefix}{name}" for name in self.all_syms()}
        bindings = {mapping.get(name, name): value
                    for name, value in self.bindings.items()}
        return replace(
            self,
            cube_macs=self.cube_macs.rename(mapping),
            cube_flops_tiled=self.cube_flops_tiled.rename(mapping),
            vector_ops=self.vector_ops.rename(mapping),
            vector_ref_ops=(None if self.vector_ref_ops is None
                            else self.vector_ref_ops.rename(mapping)),
            alg_override=(None if self.alg_override is None
                          else self.alg_override.rename(mapping)),
            read_bytes=self.read_bytes.rename(mapping),
            write_bytes=self.write_bytes.rename(mapping),
            granule_read_bytes=self.granule_read_bytes.rename(mapping),
            granule_write_bytes=self.granule_write_bytes.rename(mapping),
            comm_bytes=self.comm_bytes.rename(mapping),
            comm_ref_bytes=self.comm_ref_bytes.rename(mapping),
            l2_hot_bytes=self.l2_hot_bytes.rename(mapping),
            bindings=bindings,
        )

    def all_syms(self) -> set[str]:
        out: set[str] = set()
        for _name, tree in self.expr_fields():
            out |= tree.syms()
        return out

    def with_binding(self, key: str, value: Any) -> "OpWork":
        bindings = dict(self.bindings)
        bindings[key] = value
        return replace(self, bindings=bindings)


class Algebra:
    """Collect symbol bindings while composing the exact work expressions."""

    def __init__(self) -> None:
        self.bindings: dict[str, Any] = {}

    def sym(self, name: str, value: Any) -> Expr:
        self.bindings[name] = int(value)
        return Sym(name)

    def real(self, name: str, value: Any) -> Expr:
        self.bindings[name] = float(value)
        return Sym(name)

    def list(self, name: str, values: Any) -> Expr:
        self.bindings[name] = list(values)
        return ListSym(name)

    def merged(self) -> dict[str, Any]:
        return dict(self.bindings)


class ByteAccount:
    """Accumulate exact compulsory and per-buffer aligned read/write bytes."""

    def __init__(self, granule: int) -> None:
        self.granule = int(granule)
        self.read: Expr = ZERO
        self.write: Expr = ZERO
        self.granule_read: Expr = ZERO
        self.granule_write: Expr = ZERO

    def _aligned(self, elements: Expr | int, elem_bytes: float,
                 rows: Expr | int | None,
                 width: Expr | int | None) -> Expr:
        from .primitives import contiguous_aligned, rows_aligned
        if rows is not None and width is not None:
            return rows_aligned(rows, width, elem_bytes, self.granule)
        return contiguous_aligned(elements, elem_bytes, self.granule)

    def read_buffer(self, elements: Expr | int, elem_bytes: float, *,
                    rows: Expr | int | None = None,
                    width: Expr | int | None = None) -> Expr:
        from .primitives import bytes_of
        compulsory = bytes_of(elements, elem_bytes)
        self.read = self.read + compulsory
        self.granule_read = self.granule_read + self._aligned(
            elements, elem_bytes, rows, width)
        return compulsory

    def write_buffer(self, elements: Expr | int, elem_bytes: float, *,
                     rows: Expr | int | None = None,
                     width: Expr | int | None = None) -> Expr:
        from .primitives import bytes_of
        compulsory = bytes_of(elements, elem_bytes)
        self.write = self.write + compulsory
        self.granule_write = self.granule_write + self._aligned(
            elements, elem_bytes, rows, width)
        return compulsory

    def add_read_raw(self, compulsory: Expr, aligned: Expr) -> None:
        self.read = self.read + compulsory
        self.granule_read = self.granule_read + aligned

    def add_write_raw(self, compulsory: Expr, aligned: Expr) -> None:
        self.write = self.write + compulsory
        self.granule_write = self.granule_write + aligned


def add_expr(*parts: Expr) -> Expr:
    total = ZERO
    for part in parts:
        total = total + as_expr(part)
    return total


__all__ = ["Algebra", "ByteAccount", "OpWork", "add_expr", "freeze_bindings"]
