"""Inclusive physical-allocation lifetime accounting.

[Role]
Sweep per-owner, per-lane pools while deduplicating explicitly aliased views.
[Invariants]
An allocation key is (pool, owner, lane, alloc_id). Anonymous views are distinct.
Slots multiply aligned bytes, inclusive endpoints overlap, and separate owners
have separate capacities. An alias uses its largest backing extent throughout
the union of all view lifetimes; contradictory slot counts fail closed.
[Timeouts & Locks]
O(B log B) in declared buffers; no waits or locks.
[Side-Effect Policy]
Pure functions; caller records are never mutated.
"""

from .exact_math import MathDomainError, align, integer


def allocation_peaks(records, alignments, max_stage=None):
    rows = tuple(records)
    end_stage = max((r.get("end", 0) for r in rows), default=0)
    if max_stage is not None:
        end_stage = integer("max_stage", max_stage)
    allocations = {}
    for index, row in enumerate(rows):
        pool, owner, lane = row["pool"], row.get("owner", 0), row.get("lane")
        integer("owner", owner)
        if lane is not None:
            integer("lane", lane)
            if lane not in (0, 1):
                raise MathDomainError("lane must be 0, 1, or None")
        slots = integer("slots", row.get("slots", 1), 1)
        size = align(integer("bytes", row["bytes"]), alignments[pool]) * slots
        start = integer("start", row.get("start", 0))
        end = integer("end", row.get("end", 0))
        resident = row.get("resident", False)
        if not isinstance(resident, bool) or start > end or end > end_stage:
            raise MathDomainError("invalid lifetime")
        if resident:
            start, end = 0, end_stage
        ident = row.get("alloc_id")
        if ident is not None and (not isinstance(ident, str) or not ident):
            raise MathDomainError("alloc_id must be a nonempty string or None")
        key = (pool, owner, lane, ("id", ident) if ident is not None else ("anonymous", index))
        prior = allocations.setdefault(key, {"size": size, "slots": slots,
                                            "intervals": [], "resident": False})
        if prior["slots"] != slots:
            raise MathDomainError("aliased allocation has contradictory slot counts")
        prior["size"] = max(prior["size"], size)
        prior["resident"] |= resident
        prior["intervals"].append((start, end))
    events, totals = {}, {}
    for key, alloc in allocations.items():
        group = key[:3]
        totals[group] = totals.get(group, 0) + alloc["size"]
        merged = []
        for start, end in sorted(alloc["intervals"]):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        delta = events.setdefault(group, {})
        for start, end in merged:
            delta[start] = delta.get(start, 0) + alloc["size"]
            delta[end + 1] = delta.get(end + 1, 0) - alloc["size"]
    peaks = {}
    for group, delta in events.items():
        active = peak = 0
        for stage in sorted(delta):
            active += delta[stage]  # all simultaneous starts/releases together
            peak = max(peak, active)
        if active != 0:
            raise MathDomainError("unbalanced lifetime sweep")
        peaks[group] = peak
    return {"peaks": peaks, "totals": totals, "allocations": len(allocations),
            "residents": sum(a["resident"] for a in allocations.values())}
