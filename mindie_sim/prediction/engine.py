"""Unique public Engine/Event/Resource surface.

[Role] Re-export the one runtime.engine implementation and its typed records.
[Invariants] No duplicate implementation, no old-executor fallback and no
  alternate state root lives here.
[Timeouts & Locks] None.
[Side-Effect Policy] Import only.
"""
from __future__ import annotations

from .runtime.engine import (
    Allocation, Engine, Event, ExecutionError, Resource, unit_rates,
)
from .runtime.contracts import *  # noqa: F401,F403
from .runtime import __all__ as _runtime_all

__all__ = [
    "Allocation", "Engine", "Event", "ExecutionError", "Resource", "unit_rates",
    *_runtime_all,
]
