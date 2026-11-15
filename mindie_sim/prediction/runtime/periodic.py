"""Read-only periodic candidate construction and exact summary algebra.

[Role] Fold settled counter deltas into exact period summaries for the Engine
  and its CycleDetector. Engine remains the only validator/committer.
[Invariants] Additive quantities add exactly; peaks take max, never multiply;
  the fold never reads or mutates a clock.
[Timeouts & Locks] Constant work per snapshot.
[Side-Effect Policy] Pure; returns frozen values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True, kw_only=True)
class IntervalSummary:
    dt: float = 0.0
    completed: int = 0
    sequences: int = 0
    allocation_events: int = 0
    service: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}))
    busy_ms: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}))
    peak_allocated: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}))

    def plus(self, other: "IntervalSummary") -> "IntervalSummary":
        """Compose adjacent intervals; peaks are maxima, never sums."""
        service = dict(self.service)
        for key, value in other.service.items():
            service[key] = service.get(key, 0.0) + value
        busy = dict(self.busy_ms)
        for key, value in other.busy_ms.items():
            busy[key] = busy.get(key, 0.0) + value
        peaks = dict(self.peak_allocated)
        for key, value in other.peak_allocated.items():
            peaks[key] = max(peaks.get(key, 0.0), value)
        return IntervalSummary(
            dt=self.dt + other.dt,
            completed=self.completed + other.completed,
            sequences=self.sequences + other.sequences,
            allocation_events=self.allocation_events + other.allocation_events,
            service=MappingProxyType(service), busy_ms=MappingProxyType(busy),
            peak_allocated=MappingProxyType(peaks))

    def scaled(self, periods: int) -> "IntervalSummary":
        """Repeat an exactly homogeneous interval; peaks stay peaks."""
        if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
            raise ValueError("period_count_invalid")
        return IntervalSummary(
            dt=self.dt * periods, completed=self.completed * periods,
            sequences=self.sequences * periods,
            allocation_events=self.allocation_events * periods,
            service=MappingProxyType(
                {key: value * periods for key, value in self.service.items()}),
            busy_ms=MappingProxyType(
                {key: value * periods for key, value in self.busy_ms.items()}),
            peak_allocated=self.peak_allocated)


def summarise(state) -> IntervalSummary:
    """Read one bounded exact snapshot of the live causal root."""
    return IntervalSummary(
        dt=state.now,
        completed=state.completed_count,
        sequences=state._sequence,
        allocation_events=state.allocation_event_count,
        service=MappingProxyType(dict(state.serviced)),
        busy_ms=MappingProxyType(dict(state.busy_ms)),
        peak_allocated=MappingProxyType(dict(state.peak_allocated)))


def delta(previous: IntervalSummary, current: IntervalSummary) -> IntervalSummary:
    """Exact nonnegative difference of two monotone snapshots."""
    def diff(left: Mapping[str, float], right: Mapping[str, float]):
        keys = set(left) | set(right)
        return MappingProxyType({
            key: max(0.0, float(right.get(key, 0.0)) - float(left.get(key, 0.0)))
            for key in keys})
    return IntervalSummary(
        dt=max(0.0, current.dt - previous.dt),
        completed=max(0, current.completed - previous.completed),
        sequences=max(0, current.sequences - previous.sequences),
        allocation_events=max(0, current.allocation_events - previous.allocation_events),
        service=diff(previous.service, current.service),
        busy_ms=diff(previous.busy_ms, current.busy_ms),
        peak_allocated=current.peak_allocated)
