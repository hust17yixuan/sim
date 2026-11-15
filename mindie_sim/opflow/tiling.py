"""Stateless retile versus stateful multi-pass spill planning.

[Role]
Split a requested tile into legal SRAM-sized pieces. Row/tile-independent
operators retile with zero GM spill; only algorithms that explicitly require
cross-pass workspace pay a real spill write/read.
[Invariants]
Every dimension is finite and positive; `requested_tile_units <= 0` is rejected
instead of being treated as a default. A single indivisible unit that does not
fit the pool fails closed. Spill bytes exist only for algorithm_pass_workspace.
[Timeouts & Locks]
Bounded integer/float arithmetic only; no waits, locks, I/O, or device access.
[Side-Effect Policy]
Pure functions; no filesystem or device effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class TilingError(ValueError):
    """An indivisible tile cannot fit the selected SRAM pool."""


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TilingError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise TilingError(f"{name} must be finite and positive, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class TilingPlan:
    requested_tile_units: float
    tile_units: float
    tile_count: int
    passes: int
    slots: int
    live_bytes_per_tile: int
    capacity_bytes: int
    spill_read_bytes: int
    spill_write_bytes: int
    pool: str
    feasible: bool = True
    retiled: bool = False
    tile_units_exact: tuple[int, ...] = ()
    tail_units: int = 0

    def __post_init__(self) -> None:
        for name in ("feasible", "retiled"):
            if not isinstance(getattr(self, name), bool):
                raise TilingError(f"{name} must be bool")

    @property
    def last_tile_units(self) -> int:
        return self.tail_units or int(self.tile_units)


def derive_tiling(total_units: float, base_tile_units: float,
                  live_bytes_per_unit: float, capacity_bytes: int, *,
                  slots: int = 2, pool: str = "UB_LANE0",
                  requested_tile_units: float | None = None,
                  spill_policy: str = "independent_retile") -> TilingPlan:
    """Return a legal tile plan; spill only for algorithm-owned workspace."""
    total = _positive("total_units", total_units)
    base = _positive("base_tile_units", base_tile_units)
    live = float(live_bytes_per_unit)
    if not math.isfinite(live) or live < 0:
        raise TilingError("live_bytes_per_unit must be finite and non-negative")
    if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
        raise TilingError("capacity_bytes must be a positive integer")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise TilingError("slots must be an integer >= 1")
    if spill_policy not in ("independent_retile", "algorithm_pass_workspace"):
        raise TilingError(f"unknown spill_policy: {spill_policy!r}")
    if requested_tile_units is not None:
        if (isinstance(requested_tile_units, bool)
                or not isinstance(requested_tile_units, (int, float))
                or not math.isfinite(float(requested_tile_units))
                or float(requested_tile_units) != int(requested_tile_units)):
            raise TilingError("requested_tile_units must be a finite positive integer")
    requested = base if requested_tile_units is None else _positive(
        "requested_tile_units", requested_tile_units
    )
    requested = min(requested, total)
    if live == 0:
        max_units = total
    else:
        max_units = float(math.floor(capacity_bytes / (live * slots)))
        if max_units < 1:
            raise TilingError(
                f"indivisible tile needs {math.ceil(live * slots)} B but {pool} "
                f"holds {capacity_bytes} B; fail closed instead of overflowing SRAM"
            )
    tile_units = min(requested, max_units, total)
    if tile_units < 1:
        raise TilingError(f"no legal tile fits {pool}")
    tile_count = int(math.ceil(total / tile_units))
    tail = int(min(tile_units, total - (tile_count - 1) * tile_units))
    passes = 1
    spill = 0
    if spill_policy == "algorithm_pass_workspace":
        passes = int(math.ceil(requested / max_units))
        used = min(total, tile_count * tile_units)
        spill = int(math.ceil((passes - 1) * used * live))
    return TilingPlan(
        requested_tile_units=requested, tile_units=tile_units,
        tile_count=tile_count, passes=passes, slots=slots,
        live_bytes_per_tile=int(math.ceil(tile_units * live * slots)),
        capacity_bytes=capacity_bytes, spill_read_bytes=spill,
        spill_write_bytes=spill, pool=pool,
        retiled=requested > tile_units, tail_units=tail,
    )


__all__ = ["TilingError", "TilingPlan", "derive_tiling"]
