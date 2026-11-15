"""Finite deterministic scheduling oracles with explicit credit scope.

[Role]
Evaluate stage chains, whole-item slot recurrence, waves, and independent queues.
[Invariants]
First-item latency pays every dependency. A cycle-mean bound is not a finite
makespan. Barrier waves can be slower on more cores with shared contention.
All durations remain Fraction values through summation.
[Timeouts & Locks]
Finite chain expansion is bounded by one million stage nodes; no I/O or locks.
[Side-Effect Policy]
Pure functions with no device or filesystem effects.
"""

from fractions import Fraction

from .exact_math import MathDomainError, integer, rational


def pipeline_chain(durations, items, slots=None, *, recurrence_latency=0,
                   recurrence_distance=1):
    """Serial engines, whole-item Q credits held through final completion.

    slots=None means independent stage buffers with enough credits for the
    ideal pipeline. A finite Q implements the explicit F[i-Q,last] edge.
    Feedback constrains starts at the named item distance, not dispatch waves.
    """
    ts = tuple(rational(t) for t in durations)
    integer("items", items)
    integer("distance", recurrence_distance, 1)
    if not ts or min(ts) < 0:
        raise MathDomainError("nonempty nonnegative stage durations required")
    if slots is not None:
        integer("slots", slots, 1)
    latency = rational(recurrence_latency)
    if latency < 0:
        raise MathDomainError("negative feedback latency")
    first = sum(ts, Fraction())
    resource = max(ts)
    cycle = first / slots if slots else Fraction()
    tau = max(resource, cycle, latency / recurrence_distance)
    if items == 0:
        finish = Fraction()
    elif slots == 1 and not latency:
        finish = items * first
    elif (slots is None or slots * resource >= first) and not latency:
        finish = first + (items - 1) * resource
    else:
        if items * len(ts) > 1_000_000:
            raise MathDomainError("finite schedule expansion exceeds node budget")
        previous = [Fraction()] * len(ts)
        finishes, starts = [], []
        for i in range(items):
            slot_ready = finishes[i - slots] if slots is not None and i >= slots else 0
            feedback = starts[i - recurrence_distance] + latency if i >= recurrence_distance else 0
            current = []
            for j, t in enumerate(ts):
                start = max(current[j - 1] if j else max(slot_ready, feedback), previous[j])
                current.append(start + t)
                if j == 0:
                    starts.append(start)
            previous = current
            finishes.append(current[-1])
        finish = finishes[-1]
    return {"first_item": first if items else Fraction(), "resource_bound": resource,
            "cycle_ratio": cycle, "tau_lower_bound": tau, "exact_makespan": finish,
            "affine_first_plus_N_minus_one_tau": first + max(0, items - 1) * tau if items else Fraction()}


def pipeline_tail(full, tail, full_items, tail_items=1, slots=None, *,
                  recurrence_latency=0, recurrence_distance=1):
    """Exact finite chain with a full prefix and actual shorter final items.

    The homogeneous prefix is solved by pipeline_chain. Its final engine
    completions and required credit/feedback history are recovered by prefix
    queries, then only the heterogeneous suffix is expanded. This preserves
    the fast path even for millions of identical full tiles.
    """
    full, tail = tuple(map(rational, full)), tuple(map(rational, tail))
    integer("full_items", full_items)
    integer("tail_items", tail_items)
    if not tail or len(full) != len(tail) or min(full + tail) < 0:
        raise MathDomainError("tail must match nonnegative full stages")
    kwargs = dict(recurrence_latency=recurrence_latency, recurrence_distance=recurrence_distance)
    if full_items == 0 and tail_items > 0:
        return pipeline_chain(tail, tail_items, slots, **kwargs)
    result = pipeline_chain(full, full_items + tail_items, slots, **kwargs)
    if full == tail or tail_items == 0:
        return result
    if slots == 1 and not recurrence_latency:
        return dict(result, exact_makespan=full_items*sum(full)+tail_items*sum(tail))
    if slots is not None or recurrence_latency:
        if (full_items + tail_items) * len(full) > 1_000_000:
            raise MathDomainError("finite schedule expansion exceeds node budget")
        # Credits/feedback require full-chain completion history; bounded DAG.
        return pipeline_items([full] * full_items + [tail] * tail_items, slots, **kwargs)
    previous = [pipeline_chain(full[:j+1], full_items)["exact_makespan"] for j in range(len(full))]
    for _ in range(tail_items):
        current = []
        for j, t in enumerate(tail):
            current.append(max(current[j-1] if j else 0, previous[j]) + t)
        previous = current
    result["exact_makespan"] = previous[-1]
    return result


def pipeline_items(items, slots=None, *, recurrence_latency=0, recurrence_distance=1):
    """Longest path over chain, engine, Q-credit and distance feedback edges."""
    if not items or len(items) * len(items[0]) > 1_000_000:
        raise MathDomainError("nonempty finite schedule within node budget required")
    reference = pipeline_chain(items[0], 1, slots, recurrence_latency=recurrence_latency,
                               recurrence_distance=recurrence_distance)
    previous = [Fraction()] * len(items[0])
    finishes, starts = [], []
    for i, durations in enumerate(items):
        ts = tuple(map(rational, durations))
        if len(ts) != len(previous) or min(ts) < 0:
            raise MathDomainError("inconsistent or negative stage durations")
        ready = finishes[i-slots] if slots and i >= slots else 0
        feedback = starts[i-recurrence_distance] + rational(recurrence_latency) if i >= recurrence_distance else 0
        current = []
        for j, t in enumerate(ts):
            start = max(previous[j], current[j-1] if j else max(ready, feedback))
            current.append(start + t)
            if j == 0:
                starts.append(start)
        previous = current
        finishes.append(current[-1])
    return dict(reference, exact_makespan=finishes[-1])


def cycle_bound(latencies, distances):
    if not latencies or len(latencies) != len(distances):
        raise MathDomainError("a cycle needs paired edges")
    ls = tuple(rational(x) for x in latencies)
    for d in distances:
        integer("distance", d)
    if min(ls) < 0 or sum(distances) == 0:
        raise MathDomainError("zero-distance or negative-latency cycle")
    return sum(ls) / sum(distances)


def wave_makespan(durations, cores, shared_time_per_tile=0):
    integer("cores", cores, 1)
    ts = tuple(rational(t) for t in durations)
    shared = rational(shared_time_per_tile)
    if any(t < 0 for t in ts) or shared < 0:
        raise MathDomainError("negative duration")
    return sum((max(max(ts[i:i + cores]), shared * len(ts[i:i + cores]))
                for i in range(0, len(ts), cores)), Fraction())


def independent_makespan(durations, cores):
    """Greedy earliest-free queue, fixed order, independent fixed-duration jobs."""
    integer("cores", cores, 1)
    queues = [Fraction()] * cores
    for value in durations:
        t = rational(value)
        if t < 0:
            raise MathDomainError("negative duration")
        i = min(range(cores), key=queues.__getitem__)
        queues[i] += t
    return max(queues)
