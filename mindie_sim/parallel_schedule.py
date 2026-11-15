"""Two-stream dependency schedules and PP stage imbalance.

[Role]
  Price a non-serial MoE overlap from an explicit ``(compute_stream,
  comm_stream)`` dependency schedule, and turn a declared PP layer partition
  into per-stage durations plus the pipeline fill/drain bubble.

[Invariants]
  * Within one microbatch the chain is ``Attn_l -> Dispatch_l -> Expert_l ->
    Combine_l -> Attn_{l+1}``: layer ``l+1`` cannot dispatch before layer
    ``l``'s Combine and its own pre-MoE attention finish.  Overlap only exists
    between *different* microbatches on disjoint resources.
  * The schedule is deterministic and conserves work: every dispatched byte is
    combined exactly once, and the makespan is never below either resource's
    total work.  Serial tasks carry real start/finish timestamps, so the trace
    is executable rather than a zero-filled placeholder.
  * PP latency comes from an exact ``(microbatch, stage)`` dependency trace
    ``finish[m][s] = max(finish[m-1][s], finish[m][s-1] + p2p) + d_s``.  With
    one microbatch the bubble beyond ``sum(d_s)`` is therefore zero (plus the
    point-to-point hops), never an extra ``(PP-1)*max_stage`` fill.
  * A hop reserves every physical resource it traverses (ports and shared
    switch/cut identities); different logical PP boundaries that share one
    switch serialize on that identity.

[Timeouts & Locks] O(layers) bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Pure functions returning fresh immutable tuples.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from mindie_sim.schema.common import ModelingError

SERIAL_MODES = frozenset({"serial"})
NONSERIAL_MODES = frozenset({"dual_microbatch", "dbo", "1f1b", "shared_expert"})
OVERLAP_MODES = SERIAL_MODES | NONSERIAL_MODES


def validate_overlap_mode(mode: str) -> str:
    """Return the normalized mode or refuse an undeclared overlap claim."""
    value = str(mode).strip().lower()
    if value not in OVERLAP_MODES:
        raise ModelingError(f"parallel_explorer_overlap_mode_invalid:{value}")
    return value


def _schedule_tasks(
    tasks: Sequence[tuple[str, str, float, tuple[str, ...]]],
) -> tuple[tuple[tuple[str, float, float], ...], float]:
    """Deterministic list schedule of an explicit ``(name, resource, dur, deps)`` DAG."""
    end: dict[str, float] = {}
    resource_free: dict[str, float] = {}
    pending = list(tasks)
    result: list[tuple[str, float, float]] = []
    while pending:
        ready = [task for task in pending
                 if all(dep in end for dep in task[3])]
        if not ready:
            raise ModelingError("parallel_explorer_two_stream_deadlock")
        task = min(ready, key=lambda item: (
            max((end[dep] for dep in item[3]), default=0.0),
            resource_free.get(item[1], 0.0), item[0]))
        name, resource, duration, deps = task
        start = max(resource_free.get(resource, 0.0),
                    max((end[dep] for dep in deps), default=0.0))
        finish = start + duration
        end[name] = finish
        resource_free[resource] = finish
        result.append((name, start, finish))
        pending.remove(task)
    return tuple(result), max(end.values(), default=0.0)


def _causal_chain_tasks(microbatches: int, layers: int, attn: float,
                        compute: float, comm: float, *,
                        serial: bool) -> list[tuple[str, str, float, tuple[str, ...]]]:
    """One causal ``Attn -> Dispatch -> Expert -> Combine`` chain per microbatch.

    The communication cost is the Dispatch+Combine pair per layer, split
    evenly.  Within a microbatch every later layer waits for the previous
    Combine (and its own attention), so ``mb0.dispatch:1`` can never start
    before ``mb0.combine:0`` finishes.
    """
    dispatch = comm / 2.0
    combine = comm / 2.0
    tasks: list[tuple[str, str, float, tuple[str, ...]]] = []
    for mb in range(microbatches):
        previous = ""
        for index in range(layers):
            attn_name = f"mb{mb}.attn:{index}"
            dispatch_name = f"mb{mb}.dispatch:{index}"
            expert_name = f"mb{mb}.expert:{index}"
            combine_name = f"mb{mb}.combine:{index}"
            tasks.append((attn_name, "serial" if serial else "compute", attn,
                          (previous,) if previous else ()))
            tasks.append((dispatch_name, "serial" if serial else "comm",
                          dispatch, (attn_name,)))
            tasks.append((expert_name, "serial" if serial else "compute",
                          compute, (dispatch_name,)))
            tasks.append((combine_name, "serial" if serial else "comm",
                          combine, (expert_name,)))
            previous = combine_name
    return tasks


def two_stream_schedule(
    moe_layers: int,
    compute_per_layer_ms: float,
    comm_per_layer_ms: float,
    mode: str,
    *,
    attn_per_layer_ms: float = 0.0,
) -> dict[str, object]:
    """Critical path of one MoE stack over two explicit resources.

    The compute stream owns attention and the expert FFN; the communication
    stream owns the Dispatch/Combine pair.  Within one microbatch the layer
    chain is strictly causal.  In a declared dual-microbatch mode two
    independent halves are pipelined on the two resources; the makespan is the
    list-schedule trace of that DAG, never a scalar subtraction.
    """
    layers = max(0, int(moe_layers))
    mode = validate_overlap_mode(mode)
    attn = max(0.0, float(attn_per_layer_ms))
    compute = max(0.0, float(compute_per_layer_ms)) * layers
    comm = max(0.0, float(comm_per_layer_ms)) * layers
    attn_total = attn * layers
    if layers == 0 or compute <= 0.0 and comm <= 0.0 and attn_total <= 0.0:
        return {"mode": mode, "compute_ms": compute, "comm_ms": comm,
                "makespan_ms": max(compute, comm), "exposed_ms": 0.0,
                "tasks": ()}
    if mode in SERIAL_MODES:
        tasks = _causal_chain_tasks(1, layers, attn,
                                    compute / layers, comm / layers, serial=True)
        result, makespan = _schedule_tasks(tasks)
        exposed = comm
        return {"mode": mode, "compute_ms": compute, "comm_ms": comm,
                "makespan_ms": makespan, "exposed_ms": exposed,
                "tasks": result}
    # Two microbatches split each layer's work in half; the total work over
    # both microbatches still equals the aggregate compute/comm.
    tasks = _causal_chain_tasks(2, layers, attn / 2.0,
                                compute / (2.0 * layers),
                                comm / (2.0 * layers), serial=False)
    result, makespan = _schedule_tasks(tasks)
    exposed = max(0.0, makespan - compute - attn_total)
    return {"mode": mode, "compute_ms": compute, "comm_ms": comm,
            "makespan_ms": makespan, "exposed_ms": exposed, "tasks": result}


def stage_durations(
    layer_partition: Sequence[int],
    per_layer_ms: Sequence[float],
) -> tuple[float, ...]:
    """Sum real per-layer durations over a declared PP partition."""
    counts = tuple(int(value) for value in layer_partition)
    if any(value < 0 for value in counts):
        raise ModelingError(f"parallel_explorer_stage_partition_invalid:{counts}")
    costs = tuple(float(value) for value in per_layer_ms)
    if sum(counts) != len(costs):
        raise ModelingError(
            f"parallel_explorer_stage_partition_mismatch:{sum(counts)}!={len(costs)}")
    result, cursor = [], 0
    for count in counts:
        result.append(float(sum(costs[cursor:cursor + count])))
        cursor += count
    return tuple(result)


class ScheduleResult(dict):
    """Schedule trace that also compares numerically to its makespan."""

    def __float__(self) -> float:
        return float(self.get("makespan_ms", 0.0))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return float(self) == float(other)
        return super().__eq__(other)

    __hash__ = None


def _hop_resource_cell(cell: Any) -> tuple[tuple[str, float], ...]:
    """Normalize one hop's ``(resource_id, service_ms)`` requirements."""
    if cell is None:
        return ()
    items = cell.items() if isinstance(cell, Mapping) else cell
    merged: dict[str, float] = {}
    for resource, service in items:
        key = str(resource)
        merged[key] = max(merged.get(key, 0.0), max(0.0, float(service)))
    return tuple(sorted(merged.items()))


def pp_microbatch_schedule_ms(
    microbatch_stages: Sequence[Sequence[float]],
    p2p_hops: Sequence[Sequence[float]] | None = None,
    hop_resources: Sequence[Sequence[Any]] | None = None,
) -> dict[str, object]:
    """Exact ``(microbatch, stage)`` trace on every physical hop resource.

    ``d[m][s]`` is the *actual* work of microbatch ``m`` on stage ``s``; the
    hop ``p2p[m][s]`` is the physical activation transfer from stage ``s`` to
    ``s + 1`` and ``hop_resources[m][s]`` names every physical resource that
    transfer traverses (source/destination ports and shared switch/cut IDs),
    with that resource's own service occupancy.

    A hop starts only after its dependency finishes *and* every traversed
    resource is free; each resource free time advances by its own service, so
    two different logical PP boundaries that share one physical switch are
    serialized instead of overlapping:

        ``hop_start = max(finish[m][s], max(free[r] for r in resources))``
        ``free[r] = hop_start + service[r]``
        ``finish[m][s+1] = max(finish[m-1][s+1], hop_start + p2p[m][s]) + d``

    A hop with no declared resource falls back to its logical boundary, which
    keeps the historical same-boundary serialization.  ``bubble_ms`` is only
    phantom idle time: it excludes every real stage and hop service, so one
    microbatch (or any PP=1 plan) has exactly zero bubble.
    """
    rows = tuple(tuple(max(0.0, float(value)) for value in row)
                 for row in microbatch_stages)
    micro = len(rows)
    if micro < 1:
        raise ModelingError("parallel_explorer_microbatches_invalid:0")
    stages = len(rows[0])
    if any(len(row) != stages for row in rows):
        raise ModelingError("parallel_explorer_microbatch_stage_width")
    if stages < 1:
        return {"makespan_ms": 0.0, "bubble_ms": 0.0, "p2p_ms": 0.0,
                "serial_ms": 0.0, "trace": ()}
    if p2p_hops is None:
        hops = tuple((0.0,) * stages for _ in range(micro))
    else:
        hops = tuple(tuple(max(0.0, float(value)) for value in row)
                     for row in p2p_hops)
        if len(hops) != micro or any(len(row) != stages for row in hops):
            raise ModelingError("parallel_explorer_p2p_shape")
    if hop_resources is None:
        requirements = tuple(tuple(() for _ in range(stages))
                             for _ in range(micro))
    else:
        if len(hop_resources) != micro or any(
                len(row) != stages for row in hop_resources):
            raise ModelingError("parallel_explorer_hop_resource_shape")
        requirements = tuple(
            tuple(_hop_resource_cell(cell) for cell in row)
            for row in hop_resources)
    finish: list[list[float]] = []
    trace: list[tuple[int, int, float, float]] = []
    hop_trace: list[tuple[int, int, float, float]] = []
    hop_free: dict[str, float] = {}
    for mb in range(micro):
        row: list[float] = []
        for stage in range(stages):
            ready = 0.0
            if stage > 0:
                required = requirements[mb][stage - 1]
                if not required:
                    required = ((f"hop:{stage - 1}", hops[mb][stage - 1]),)
                hop_start = max([row[stage - 1]]
                                + [hop_free.get(key, 0.0) for key, _ in required])
                hop_end = hop_start + hops[mb][stage - 1]
                for key, service in required:
                    hop_free[key] = hop_start + service
                hop_trace.append((mb, stage - 1, hop_start, hop_end))
                ready = max(ready, hop_end)
            if mb > 0:
                ready = max(ready, finish[mb - 1][stage])
            done = ready + rows[mb][stage]
            row.append(done)
            trace.append((mb, stage, ready, done))
        finish.append(row)
    serial = float(sum(sum(row) for row in rows))
    p2p = float(sum(sum(row) for row in hops))
    makespan = finish[-1][-1]
    return ScheduleResult(makespan_ms=makespan,
                          bubble_ms=max(0.0, makespan - serial - p2p),
                          p2p_ms=p2p, serial_ms=serial,
                          trace=tuple(trace), hop_trace=tuple(hop_trace),
                          hop_resources=requirements)


def pp_pipeline_ms(stage_ms: Sequence[float], microbatches: int,
                   p2p_ms: float = 0.0) -> dict[str, object]:
    """Exact ``(microbatch, stage)`` dependency trace.

    ``finish[m][s] = max(finish[m-1][s], finish[m][s-1] + p2p) + d_s``.  With a
    single microbatch the makespan is ``sum(d_s) + (PP-1)*p2p`` and the bubble
    beyond ``sum(d_s)`` is exactly the point-to-point hops, never an extra
    ``(PP-1)*max_stage`` fill.  The returned ``bubble_ms`` is the makespan in
    excess of the serialized stage service.
    """
    micro = int(microbatches)
    if micro < 1:
        raise ModelingError(f"parallel_explorer_microbatches_invalid:{micro}")
    stages = tuple(max(0.0, float(value)) for value in stage_ms)
    hop = max(0.0, float(p2p_ms))
    if not stages:
        return {"makespan_ms": 0.0, "bubble_ms": 0.0, "max_stage_ms": 0.0,
                "serial_ms": 0.0, "trace": ()}
    serial = float(sum(stages))
    finish: list[list[float]] = []
    trace: list[tuple[int, int, float, float]] = []
    for mb in range(micro):
        row: list[float] = []
        for stage, duration in enumerate(stages):
            ready = 0.0
            if stage > 0:
                ready = max(ready, row[stage - 1] + hop)
            if mb > 0:
                ready = max(ready, finish[mb - 1][stage])
            start = ready
            done = start + duration
            row.append(done)
            trace.append((mb, stage, start, done))
        finish.append(row)
    makespan = finish[-1][-1]
    return {"makespan_ms": makespan,
            "bubble_ms": max(0.0, makespan - serial),
            "max_stage_ms": max(stages), "serial_ms": serial,
            "trace": tuple(trace)}


def pp_bubble_ms(stage_ms: Sequence[float], microbatches: int) -> tuple[float, float]:
    """Return ``(max_stage_ms, (PP-1)*max_stage/microbatches)``.

    This is the declared pipeline-imbalance *indicator* retained for reporting
    compatibility; latency is priced from :func:`pp_pipeline_ms` instead.
    """
    micro = int(microbatches)
    if micro < 1:
        raise ModelingError(f"parallel_explorer_microbatches_invalid:{micro}")
    stages = tuple(max(0.0, float(value)) for value in stage_ms)
    if not stages:
        return 0.0, 0.0
    maximum = max(stages)
    return maximum, (len(stages) - 1) * maximum / micro


def layer_partition(layers: int, pp: int, declared: Sequence[int] | None = None
                    ) -> tuple[int, ...]:
    """Normalize a complete PP split, preserving a valid declared one."""
    count = int(pp)
    total = int(layers)
    if count < 1 or total < count:
        raise ModelingError(f"parallel_explorer_pp_invalid:{count}>{total}")
    if declared is not None:
        values = tuple(int(value) for value in declared)
        if len(values) == count and sum(values) == total and all(values):
            return values
    return tuple(total // count + (1 if index < total % count else 0)
                 for index in range(count))


def moe_layer_mask(plans: Sequence[object], layers: int) -> tuple[bool, ...]:
    """Per-physical-layer MoE mask from the model's actual layer plans."""
    mask = [False] * max(0, int(layers))
    for plan in plans:
        index = int(getattr(plan, "index", -1))
        if 0 <= index < len(mask):
            mask[index] = str(getattr(plan, "ffn", "")) == "moe"
    return tuple(mask)


__all__ = ["NONSERIAL_MODES", "OVERLAP_MODES", "SERIAL_MODES", "ScheduleResult",
           "layer_partition",
           "moe_layer_mask", "pp_bubble_ms", "pp_microbatch_schedule_ms",
           "pp_pipeline_ms", "stage_durations",
           "two_stream_schedule", "validate_overlap_mode"]
