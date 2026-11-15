"""Host-release and explicit-gap evidence for ordered trace replay.

[Role]
  Derive one per-occurrence host release delay from dependency/stream readiness
  and explicit captured gaps, and price only idle time that no explicit gap
  already represents.  It also owns the observed window, signed terminal-tail
  evidence and launch-cost helpers; ``diff_trace_replay`` consumes them without
  re-adding a captured delay to a start floor that already contains it.

[Invariants]
  * ``release_delays`` measures the delay from the op's *earliest* ready time
    and never from its own already-pinned start; replaying the same trace can
    therefore not create a second release.
  * Explicit gaps are evidence, not extra time: the delay is the larger of the
    implied pre-op idle and the explicit gap duration, never their sum.
  * ``gap_evidence`` counts a gap only where it intersects device-idle time
    after subtracting busy intervals; an excluded gap end is never charged
    again as a host bubble.
  * Explicit captured terminal coverage, including a zero-duration gap,
    replaces the predicted tail in either direction; a missing captured total
    is reported and never silently certified as reconciled.

[Timeouts & Locks]
  Pure in-memory interval arithmetic proportional to ops + gaps.

[Side-Effect Policy]
  No I/O, device access or simulation imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from ._diff_support import snapshot_call

EPS = 1e-9


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _union(intervals, window=None):
    low, high = window if window else (None, None)
    clipped = []
    for item in intervals:
        if not item:
            continue
        start, end = item
        if low is not None:
            start = max(start, low)
        if high is not None:
            end = min(end, high)
        if end > start:
            clipped.append((start, end))
    merged = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1] + EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _total(intervals) -> float:
    return math.fsum(end - start for start, end in intervals)


def _complement(busy, window):
    low, high = window
    gaps, cursor = [], low
    for start, end in busy:
        if start > cursor + EPS:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if high > cursor + EPS:
        gaps.append((cursor, high))
    return gaps


def _intersect(left, right):
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            out.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return out


def interval(start: float, end: float) -> tuple[float, float] | None:
    """Half-open interval or ``None`` for non-finite/empty spans."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return (start, end)


def union(intervals, window=None) -> list[tuple[float, float]]:
    """Merge overlapping half-open intervals, optionally clipped to a window."""
    return _union(intervals, window=window)


def total(intervals) -> float:
    """Exact (compensated) length of already-disjoint intervals."""
    return _total(intervals)


def complement(busy, window) -> list[tuple[float, float]]:
    """Idle intervals inside ``window`` that no busy interval covers."""
    return _complement(busy, window)


def intersect(left, right) -> list[tuple[float, float]]:
    """Intersection of two sorted disjoint interval lists."""
    return _intersect(left, right)


def normalize_gaps(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize explicit gap evidence into ``{stream,start,duration,kind}``."""
    raw = packet.get("gaps") or packet.get("stream_gaps") or ()
    gaps = []
    for item in raw if isinstance(raw, Sequence) else ():
        if isinstance(item, Mapping):
            length = (_num(item["duration_ms"]) if "duration_ms" in item
                      else _num(item["gap_ms"]) if "gap_ms" in item
                      else _num(item.get("idle_us")) / 1000.0)
            gaps.append({"stream": int(_num(item.get("stream", 0))),
                         "start": _num(item.get("start_ms")), "duration": length,
                         "kind": item.get("kind")})
        else:
            gaps.append({"stream": 0, "start": 0.0, "duration": _num(item),
                         "kind": None})
    return [gap for gap in gaps if gap["duration"] > 0]


def window_of(packet: Mapping[str, Any], ops: Sequence[Mapping[str, Any]]
              ) -> tuple[float, float] | None:
    """Explicit window/total, else the extent of BOTH timed ops and gaps.

    An operator-only envelope would clip an explicit terminal gap and then
    certify a truncated step as complete.  Gaps therefore extend the inferred
    extent even when ``ops`` end earlier.
    """
    explicit = packet.get("window_ms")
    if (isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes))
            and len(explicit) == 2):
        return (_num(explicit[0]), _num(explicit[1]))
    total_ms = packet.get("total_ms", packet.get("step_ms"))
    if total_ms is not None:
        return (0.0, max(0.0, _num(total_ms)))
    lows = [_num(op["start"]) for op in ops]
    highs = [_num(op["start"]) + max(0.0, _num(op["duration"])) for op in ops]
    for gap in _normalize_gaps_snapshot(packet):
        lows.append(gap["start"])
        highs.append(gap["start"] + gap["duration"])
    if not lows:
        return None
    return (min(lows), max(highs))


def captured_total_explicit(packet: Mapping[str, Any]) -> bool:
    """Whether the captured packet states its own observed step total."""
    explicit = packet.get("window_ms")
    if (isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes))
            and len(explicit) == 2):
        return True
    return (packet.get("total_ms") is not None
            or packet.get("step_ms") is not None)


def terminal_gap_evidence(packet: Mapping[str, Any],
                          window: tuple[float, float],
                          op_end: float) -> tuple[bool, float]:
    """Explicit captured terminal coverage after ``op_end``, including zero.

    A zero-duration terminal gap is positive evidence that the captured trace
    ends at ``start_ms``: it is explicitly reported rather than missing.
    """
    low, high = window
    end = min(max(_num(op_end), low), high)
    found = False
    intervals = []
    raw = packet.get("gaps") or packet.get("stream_gaps") or ()
    for item in raw if isinstance(raw, Sequence) else ():
        if isinstance(item, Mapping):
            start = _num(item.get("start_ms"))
            length = (_num(item["duration_ms"]) if "duration_ms" in item
                      else _num(item["gap_ms"]) if "gap_ms" in item
                      else _num(item.get("idle_us")) / 1000.0)
        else:
            start, length = 0.0, _num(item)
        stop = start + max(0.0, length)
        if start > high + EPS:
            continue
        if stop > end + EPS or (stop <= start + EPS and start >= end - EPS):
            found = True
            clip_start = max(low, start, end)
            clip_stop = min(high, stop)
            if clip_stop > clip_start:
                intervals.append((clip_start, clip_stop))
    return found, _total(_union(intervals, window=window))


def launch_cost(packet: Mapping[str, Any], launch_overhead_us: float | None
                ) -> tuple[float, float, int]:
    """Return ``(per-launch_us, modeled_launch_ms, launch_count)``."""
    launches = packet.get("launch_count")
    count = int(_num(launches)) if launches is not None else 0
    overhead = 0.0
    if launch_overhead_us is not None or launches is not None:
        overhead = max(0.0, _num(launch_overhead_us if launch_overhead_us is not None
                                else packet.get("modeled_launch_overhead_us")))
    return overhead, (overhead / 1000.0) * count, count


def release_delays(captured: Sequence[Mapping[str, Any]],
                   gaps: Sequence[Mapping[str, Any]], origin: float,
                   deps: Sequence[Sequence[int]]
                   ) -> list[float]:
    """Return one host-release delay per captured occurrence.

    ``deps[index]`` is the resolved predecessor list for that occurrence.  The
    delay is measured from the later of every dependency end and the previous
    occurrence on the same ``(rank, stream)``; an explicit gap at the op start
    can raise that implied idle but is never added on top of it.
    """
    ends: dict[int, float] = {}
    previous: dict[tuple[int, int], float] = {}
    releases: list[float] = []
    gap_rows = []
    for gap in gaps:
        gstart = _num(gap.get("start"))
        gend = gstart + max(0.0, _num(gap.get("duration")))
        gap_rows.append((gap.get("stream"), gap.get("rank"), gstart, gend))
    for index, op in enumerate(captured):
        ready = _num(origin)
        for dep in deps[index]:
            ready = max(ready, ends.get(dep, ready))
        stream_key = (int(op.get("rank", 0)), int(op["stream"]))
        ready = max(ready, previous.get(stream_key, ready))
        start_ms = _num(op.get("start"))
        idle = max(0.0, start_ms - ready)
        explicit_intervals = []
        for stream, rank, gstart, gend in gap_rows:
            if stream is not None and int(stream) != int(op["stream"]):
                continue
            if (rank is not None
                    and int(rank) != int(op.get("rank", 0))):
                continue
            if gend <= start_ms + 1e-6 and gstart >= ready - 1e-6:
                explicit_intervals.append([gstart, gend])
        explicit = _total(_union(explicit_intervals))
        releases.append(max(idle, explicit))
        ends[index] = start_ms + max(0.0, _num(op.get("duration")))
        previous[stream_key] = ends[index]
    return releases


def tail_gap(gaps: Sequence[Mapping[str, Any]],
             window: tuple[float, float], op_end: float,
             implicit: bool = True) -> float:
    """Union of the implicit terminal interval with explicit terminal gaps.

    ``total_ms - max_op_end_ms`` is the baseline predicted tail.  A captured
    tail can add explicit gap evidence but the same milliseconds are never
    counted twice: both sources are merged as half-open intervals.  With
    ``implicit=False`` only recorded terminal gaps are priced, so an
    unreported captured tail stays an unexplained residual.
    """
    low, high = window
    end = min(max(_num(op_end), low), high)
    intervals = [[end, high]] if implicit and high > end else []
    for gap in gaps:
        gstart = _num(gap.get("start"))
        start = max(low, gstart, end)
        stop = min(high, gstart + max(0.0, _num(gap.get("duration"))))
        if stop > start:
            intervals.append([start, stop])
    return _total(_union(intervals, window=window))


def gap_evidence(gaps: Sequence[Mapping[str, Any]],
                 window: tuple[float, float],
                 replay_starts: Sequence[float],
                 durations: Sequence[float],
                 exclude_ends: set[float] | None = None
                 ) -> tuple[float, float, int]:
    """Idle time that an explicit captured gap actually explains."""
    busy = _union([[_num(replay_starts[index]),
                    _num(replay_starts[index]) + max(0.0, _num(durations[index]))]
                   for index in range(len(replay_starts))], window=window)
    idle = max(0.0, (window[1] - window[0]) - _total(busy))
    if exclude_ends:
        gaps = [gap for gap in gaps if round(
            _num(gap.get("start")) + _num(gap.get("duration")), 9)
            not in exclude_ends]
    if not gaps:
        return 0.0, idle, 0
    gap_union = _union([[_num(gap.get("start")),
                         _num(gap.get("start")) + max(0.0, _num(gap.get("duration")))]
                        for gap in gaps], window=window)
    return (_total(_intersect(gap_union, _complement(busy, window))), idle,
            len(gaps))


def op_uid(run_id: str, op: Mapping[str, Any]) -> str:
    """Canonical immutable occurrence selector digest."""
    payload = [run_id, op["rank"], op["step"], op["stream"], op["raw_op_index"],
               str(op.get("abi_digest") or "")]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Internal calls consume detached inputs; public aliases validate ingress.
_union_snapshot = union
union = snapshot_call(union, 'intervals', 'window')
_total_snapshot = total
total = snapshot_call(total, 'intervals')
_complement_snapshot = complement
complement = snapshot_call(complement, 'busy', 'window')
_intersect_snapshot = intersect
intersect = snapshot_call(intersect, 'left', 'right')
_normalize_gaps_snapshot = normalize_gaps
normalize_gaps = snapshot_call(normalize_gaps, 'packet')
_window_of_snapshot = window_of
window_of = snapshot_call(window_of, 'packet', 'ops')
_captured_total_explicit_snapshot = captured_total_explicit
captured_total_explicit = snapshot_call(captured_total_explicit, 'packet')
_terminal_gap_evidence_snapshot = terminal_gap_evidence
terminal_gap_evidence = snapshot_call(terminal_gap_evidence, 'packet', 'window')
_launch_cost_snapshot = launch_cost
launch_cost = snapshot_call(launch_cost, 'packet')
_release_delays_snapshot = release_delays
release_delays = snapshot_call(release_delays, 'captured', 'gaps', 'deps')
_tail_gap_snapshot = tail_gap
tail_gap = snapshot_call(tail_gap, 'gaps', 'window')
_gap_evidence_snapshot = gap_evidence
gap_evidence = snapshot_call(gap_evidence, 'gaps', 'window', 'replay_starts',
                            'durations', 'exclude_ends', set_fields=('exclude_ends',))
_op_uid_snapshot = op_uid
op_uid = snapshot_call(op_uid, 'run_id', 'op')


__all__ = [
    "captured_total_explicit", "complement", "gap_evidence", "intersect",
    "interval", "launch_cost", "normalize_gaps", "op_uid", "release_delays",
    "tail_gap", "terminal_gap_evidence", "total", "union", "window_of",
]
