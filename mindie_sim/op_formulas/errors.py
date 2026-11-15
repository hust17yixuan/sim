"""Stable error types shared by the formula engine and the profiler bridge.

[Role]
  Define ``FormulaError`` (any invalid evaluation input) and its structured
  subclass ``FormulaValidationError`` (an ABI/parameter rejection that the
  annotator reports as ``status="invalid_abi"``).  Keeping these in a leaf
  module prevents the expression algebra and the numeric primitives from
  importing each other.

[Invariants]
  * Both derive from ``ValueError`` so existing callers that catch ``ValueError``
    keep working.
  * Messages are actionable and contain no memory addresses or timings.

[Timeouts & Locks]
  Pure declarations; no I/O.

[Side-Effect Policy]
  No side effects.
"""

from __future__ import annotations


class FormulaError(ValueError):
    """Invalid formula input with an actionable, stable message."""


class FormulaValidationError(FormulaError):
    """Structured ABI/parameter rejection (``status="invalid_abi"``)."""


__all__ = ["FormulaError", "FormulaValidationError"]
