"""Public explainable per-operator compute/memory formula engine.

[Role]
  Publish the stable API for turning an operator ABI plus a SKU into an
  immutable ``OperatorFormulaEvaluation``.  This package owns the algebra
  (`formula_card`), the certified floor and the structural reference; profiler
  annotation and MCP receipts live in ``mindie_profiler`` and consume this
  package.

[Invariants]
  * The public objects are frozen dataclasses with plain-data ``as_dict`` /
    ``to_json`` output.
  * Every evaluation satisfies ``0 < tau_hw <= t_floor <= t_ref``.
  * Unsupported operators or missing dimensions raise ``FormulaError``; no
    placeholder evaluation is produced.
  * The catalogue covers at least 18 canonical operator families.

[Timeouts & Locks]
  Importing performs no I/O, device access or global mutation.

[Side-Effect Policy]
  Pure declarations and pure arithmetic.
"""

from __future__ import annotations

from .engine import diagnose, evaluate_operator, evaluate_with_diagnostics
from .errors import FormulaError, FormulaValidationError
from .expr_ast import evaluate_formula_expr
from .hardware import (
    A2, A3, A5, A5_FABRIC_FITS, A5_MC2_FITS, CALIBRATION_PROVENANCE,
    DEFAULT_SKU, HardwareProfile, LAUNCH_MODES, SKU_ALIASES, SKUS,
    cube_dtype, cube_tier, dtype_bytes, resolve_sku,
)
from .model import (
    SCHEMA, DeratingBreakdown, FormulaCard, LatencyBounds,
    OperatorFormulaEvaluation,
)
from .registry import canonical_name, supported_operators
from .work import OpWork

evaluate = evaluate_operator
evaluate_op_formula = evaluate_operator


def formula_catalog() -> tuple[dict[str, str], ...]:
    """Return a stable catalogue of canonical operators and their families."""
    from .registry import _CATALOG  # local import keeps the public surface small

    return tuple(
        {"operator": name, "family": str(spec[0]), "kind": str(spec[1])}
        for name, spec in sorted(_CATALOG.items())
    )


__all__ = [
    "A2", "A3", "A5", "A5_FABRIC_FITS", "A5_MC2_FITS",
    "CALIBRATION_PROVENANCE", "DEFAULT_SKU", "DeratingBreakdown", "FormulaCard",
    "FormulaError", "FormulaValidationError", "HardwareProfile",
    "LAUNCH_MODES", "LatencyBounds",
    "OpWork", "OperatorFormulaEvaluation", "SCHEMA", "SKU_ALIASES", "SKUS",
    "canonical_name", "cube_dtype", "cube_tier", "diagnose", "dtype_bytes",
    "evaluate", "evaluate_formula_expr",
    "evaluate_op_formula", "evaluate_operator", "evaluate_with_diagnostics",
    "formula_catalog", "resolve_sku", "supported_operators",
]
