"""Serial, fork/join, fusion and state composition of compiled operators.

[Role]
  Compose guarded operator cost functions into subgraph costs.  A single-stream
  post-fusion kernel chain is an exact sum when no external wait is declared.
  Overlapping branches run through the shared ``Engine`` so HBM/Cube/Vector
  contention changes the makespan; taking the maximum of isolated scalars is
  deliberately not used.  Native fusion lowers semantic groups to physical
  kernels and removes only declared internal materialisations.

[Invariants]
  * ``serial.total_ms == sum(operator durations) + declared readiness gaps``.
    A late input adds its real positive gap, never a fixed overlap discount.
  * Fork/join contention is executed on the one Engine: shared pipes are not
    duplicated and disjoint pipes may overlap.
  * Fusion keeps every declared external output; only contract-listed
    internal materialisation bytes and repeated startup are removed.
  * Pipeline recurrence keeps finite fill/drain and slot limits; one tile is
    ``load+compute(+store)`` rather than a steady-state ``max``.
  * State effects are applied in order; a consumer before its producer raises.

[Timeouts & Locks]
  Pure finite arithmetic and bounded Engine events.  No waits or locks.

[Side-Effect Policy]
  Returns immutable results; Engine calls are caller-owned virtual execution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from collections.abc import Mapping as _AbcMapping
from typing import Any, Callable, Iterable, Mapping, Sequence

from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.runtime.contracts import (
    BindingFrame, CatalogGeneration, EdgeSpec, PortSpec, ProgramCatalog,
    StageSpec, WorkTerm,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import (
    OpSpec, ProgramFragment, WorkItem, WorkSet,
)
from mindie_sim.schema.subgraphs import (
    ALL_ATTENTION_KINDS, LATENT_ATTENTION_KINDS, STANDARD_ATTENTION_KINDS,
    SubgraphInstance,
)
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .compiler import ( CompiledOperator, CompiledSubgraph, NativeProgram, NativeTask,
    OperatorCost, PipeSupply, RequestGeometry, SubgraphEvaluation,
    WorkloadGeometry, _PIPE_UNITS, _apply_operator_state, )
from .sfa_prolog import visible_adjustment as _sfa_visible_adjustment
from .expressions import Expr, as_fraction
from .guards import PipeDemand

@dataclass(frozen=True)
class SerialResult:
    costs: tuple[OperatorCost, ...]; total_ms: Fraction; state: Any
    gaps_ms: tuple[Fraction, ...] = (); fused: tuple[str, ...] = ()
    readiness: tuple[tuple[str, Fraction], ...] = ()
    fusion_results: tuple["FusionResult", ...] = ()
    contention: "ContentionEnvelope | None" = None
    unfused_costs: tuple[OperatorCost, ...] = ()
    states: tuple[Any, ...] = ()
    submission: tuple["SubmittedOperator", ...] = ()
    @property
    def gap_ms(self) -> Fraction: return sum(self.gaps_ms, Fraction(0))
    def by_name(self, name: str) -> OperatorCost:
        for cost in self.costs:
            if cost.name == name: return cost
        raise ModelingError(f"serial_cost_missing:{name}")
    def to_dict(self) -> dict[str, Any]:
        return { "total_ms": str(self.total_ms), "gap_ms": str(self.gap_ms), "fused": list(self.fused),
            "fusion_results": [item.to_dict() for item in self.fusion_results], "operators": [cost.to_dict() for cost in self.costs],
            "submitted": len(self.submission), }

@dataclass(frozen=True)
class FusionValue:
    """One logical/physical value at a fusion boundary.

    The identity/repr fields come from the existing T01 ``TensorValue``
    (producer, output, layout, storage, alias_of).  This record adds only the
    fusion-boundary byte/role/version facts: ``logical_bytes`` is the visible
    value size (a K=V alias still has a full V), and ``written_bytes`` is what
    *this* fused launch physically writes.  ``role`` is ``logical``,
    ``alias`` (same storage as ``alias_of``) or ``aggregate`` (a derived
    cache/row view such as ``kv``; never summed with its members).
    """
    name: str
    value: TensorValue
    logical_bytes: Fraction
    role: str = "logical"
    written_bytes: Fraction = Fraction(0)
    view_of: str = ""
    version: str = ""
    layer: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ModelingError("fusion_value_name_invalid")
        if not isinstance(self.value, TensorValue):
            raise ModelingError(f"fusion_value_tensor_invalid:{self.name}")
        if self.role not in {"logical", "alias", "aggregate"}:
            raise ModelingError(f"fusion_value_role_invalid:{self.name}:{self.role}")
        for field in ("logical_bytes", "written_bytes"):
            value = as_fraction(getattr(self, field))
            if value < 0:
                raise ModelingError(f"fusion_value_bytes_invalid:{self.name}:{field}")
            object.__setattr__(self, field, value)
        if self.role == "alias" and not self.value.alias_of:
            raise ModelingError(f"fusion_value_alias_target_missing:{self.name}")
        if self.role == "aggregate" and not self.view_of:
            raise ModelingError(f"fusion_value_aggregate_view_missing:{self.name}")
        if self.role != "logical" and self.written_bytes:
            raise ModelingError(f"fusion_value_nonlogical_written:{self.name}")

    @property
    def producer(self) -> str: return self.value.producer
    @property
    def alias_of(self) -> str: return self.value.alias_of
    @property
    def storage(self) -> str: return self.value.storage
    @property
    def output(self) -> str: return str(self.value.output)
    @property
    def dtype(self) -> str: return self.value.dtype
    @property
    def layout(self) -> TensorLayout: return self.value.layout
    @property
    def offset(self) -> int: return self.value.layout.offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": self.value.to_dict(),
            "logical_bytes": str(self.logical_bytes),
            "role": self.role, "producer": self.producer,
            "written_bytes": str(self.written_bytes),
            "alias_of": self.alias_of, "view_of": self.view_of,
            "storage": self.storage, "output": self.output,
            "version": self.version, "layer": self.layer,
        }

@dataclass(frozen=True)
class FusionResult:
    name: str; semantic: tuple[OperatorCost, ...]; fused: OperatorCost
    removed: tuple[tuple[str, Fraction], ...]; kept: tuple[tuple[str, Fraction], ...]
    startup_count: int = 1; isolated_ms: Fraction = Fraction(0)
    #: Explicit logical/physical disposition of each kept value.  Empty for
    #: fusions without a declared typed slice contract.
    values: tuple[FusionValue, ...] = ()
    @property
    def removed_bytes(self) -> Fraction: return sum((value for _, value in self.removed), Fraction(0))
    def to_dict(self) -> dict[str, Any]:
        return { "name": self.name, "members": [cost.name for cost in self.semantic], "fused": self.fused.to_dict(),
            "removed": {key: str(value) for key, value in self.removed},
            "kept": {key: str(value) for key, value in self.kept}, "startup_count": self.startup_count,
            "isolated_ms": str(self.isolated_ms),
            "values": [value.to_dict() for value in self.values], }

@dataclass(frozen=True)
class ForkJoinResult:
    prefix_ms: Fraction; join_ms: Fraction; isolated_max_ms: Fraction
    contention_ms: Fraction; engine_ms: Fraction
    branches: tuple[tuple[str, Fraction], ...]; resources: tuple[tuple[str, Fraction], ...] = ()
    @property
    def total_ms(self) -> Fraction: return self.prefix_ms + self.contention_ms + self.join_ms
    @property
    def rejected_isolated_max(self) -> bool: return self.contention_ms != self.isolated_max_ms
    def to_dict(self) -> dict[str, Any]:
        return { "prefix_ms": str(self.prefix_ms), "join_ms": str(self.join_ms),
            "isolated_max_ms": str(self.isolated_max_ms), "contention_ms": str(self.contention_ms),
            "engine_ms": str(self.engine_ms), "total_ms": str(self.total_ms),
            "branches": {name: str(value) for name, value in self.branches},
            "resources": {name: str(value) for name, value in self.resources}, "rejected_isolated_max": self.rejected_isolated_max, }

@dataclass(frozen=True)
class ContentionEnvelope:
    """Shared-capacity envelope for concurrently ready branches.

    ``isolated_max_ms`` is reported only as a rejected counterfactual.  The
    contract is the one-Engine makespan over the union of resource work plus
    the per-resource lower bound ``sum(work_p)/rate_p``.  Two HBM-only streams
    therefore add their bytes; disjoint Cube/Link streams may overlap.
    """
    makespan_ms: Fraction; isolated_max_ms: Fraction; isolated_sum_ms: Fraction
    resource_work: tuple[tuple[str, Fraction], ...]; resource_bounds: tuple[tuple[str, Fraction], ...]
    shared_pipes: tuple[str, ...]; disjoint_pipes: tuple[str, ...]; branches: tuple[tuple[str, Fraction], ...]
    # Engine completion of each chain, not the isolated counterfactual in branches.
    completions: tuple[tuple[str, Fraction], ...] = ()
    @property
    def additivity_gap_ms(self) -> Fraction: return self.makespan_ms - self.isolated_sum_ms
    @property
    def lower_bound_ms(self) -> Fraction: return max((value for _, value in self.resource_bounds), default=Fraction(0))
    @property
    def rejected_isolated_max(self) -> bool: return self.makespan_ms != self.isolated_max_ms
    def to_dict(self) -> dict[str, Any]:
        return { "makespan_ms": str(self.makespan_ms), "isolated_max_ms": str(self.isolated_max_ms),
            "isolated_sum_ms": str(self.isolated_sum_ms), "lower_bound_ms": str(self.lower_bound_ms),
            "additivity_gap_ms": str(self.additivity_gap_ms), "resource_work": {key: str(value) for key, value in self.resource_work},
            "resource_bounds": {key: str(value) for key, value in self.resource_bounds},
            "shared_pipes": list(self.shared_pipes), "disjoint_pipes": list(self.disjoint_pipes),
            "branches": {name: str(value) for name, value in self.branches}, "rejected_isolated_max": self.rejected_isolated_max, }

@dataclass(frozen=True)
class ClosedForm:
    formula: str; certified: bool = True; reason: str = ""
    def to_dict(self) -> dict[str, Any]:
        return {"formula": self.formula, "certified": self.certified, "reason": self.reason}

@dataclass(frozen=True)
class EngineResult:
    makespan_ms: Fraction; branches: tuple[tuple[str, Fraction], ...]
    resources: tuple[tuple[str, Fraction], ...]
    summary: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class OperatorSegment:
    """One kernel on a stream: resource stages plus its own latency events.

    ``floor`` starts when the kernel is released and overlaps startup, body
    and dependency.  ``startup`` delays the body.  ``dependency`` follows the
    body.  The next kernel on the same stream is released at
    ``max(floor, startup + body + dependency)``.  Caps, when set, are executed
    rates below the supply and apply only to this kernel's resource stages.
    """

    stages: tuple[tuple[tuple[str, Fraction], ...], ...]
    floor: Fraction = Fraction(0)
    startup: Fraction = Fraction(0)
    dependency: Fraction = Fraction(0)
    caps: tuple[tuple[str, Fraction], ...] = ()
    native_stages: tuple[tuple[NativeTask, ...], ...] = ()
    native_program: NativeProgram | None = None


def operator_segment(cost: OperatorCost, supply: PipeSupply | None = None) -> OperatorSegment:
    """Lift one compiled operator into a causal engine segment.

    Native stage work is copied unchanged.  A cap is recorded only when the
    compiler's elapsed rate for that pipe is below the live supply, which is
    how a declared slower execution dtype stretches the body without adding
    FLOPs or bytes.
    """
    fields = dict(cost.derived)
    if cost.stage_work:
        stages = tuple(
            tuple((str(pipe), as_fraction(amount)) for pipe, amount in stage)
            for stage in cost.stage_work)
    elif cost.work:
        stages = (tuple((str(pipe), as_fraction(amount)) for pipe, amount in cost.work),)
    else:
        stages = ()
    caps: list[tuple[str, Fraction]] = []
    if supply is not None:
        times = dict(cost.times)
        for pipe, amount in cost.work:
            elapsed = as_fraction(times.get(pipe, 0))
            native = as_fraction(amount)
            if native <= 0 or elapsed <= 0:
                continue
            executed = native / elapsed
            base = as_fraction(supply.effective(str(pipe)))
            if executed < base:
                caps.append((str(pipe), executed))
    native_program = cost.native_program
    if native_program is not None:
        # The original executable program owns its own owner placement,
        # reserves, dependencies and completion.  Do not add a fabricated
        # floor/startup or replay an aggregate.
        return OperatorSegment(
            stages=(), floor=Fraction(0), startup=Fraction(0),
            dependency=Fraction(0), native_program=native_program)
    schedule = cost.native_schedule
    if schedule is not None and schedule.stages:
        # The native owner program is the source work.  Its analytical
        # stage_work can collapse register work to zero or sum a different
        # aggregate (the conv 756 us replay); never replay that substitute.
        return OperatorSegment(
            stages=(),
            floor=as_fraction(fields.get("service_floor_ms", 0)),
            startup=schedule.startup,
            dependency=schedule.dependency,
            native_stages=schedule.stages,
        )
    return OperatorSegment(
        stages=stages,
        floor=as_fraction(fields.get("service_floor_ms", 0)),
        startup=as_fraction(fields.get("service_startup_ms", 0)),
        dependency=as_fraction(fields.get("service_dependency_ms", 0)),
        caps=tuple(caps),
    )


def _resource_units(pipe: str) -> str:
    return _PIPE_UNITS.get(pipe, "work")

def run_engine(branches: Mapping[str, Sequence[Sequence[PipeDemand]]], supply: PipeSupply, env: Mapping[str, Any]) -> EngineResult:
    """Execute symbolic branch stage chains on one shared-resource Engine."""
    from .expressions import evaluate
    bound = {name: tuple(tuple((demand.pipe, evaluate(demand.work, env)) for demand in stage) for stage in stages)
             for name, stages in branches.items()}
    return run_engine_work(bound, supply)

def _inferred_floor(stages: Sequence[Sequence[tuple[str, Fraction]]], supply: PipeSupply) -> Fraction:
    """Infer a declared invocation floor for a direct branch caller.

    Stage work carries no latency.  When a caller supplies only stage work, the
    highest applicable declared minimum from the stage's resource classes is
    the nonshared floor.  Explicit ``latencies`` always replace this.
    """
    inv = dict(supply.invocations)
    pipes = {pipe for stage in stages for pipe, amount in stage if amount}
    classes = [name for name in ("cube", "vector") if name in pipes and name in inv]
    return max((as_fraction(inv[name]) for name in classes), default=Fraction(0))


def _stage_work_map(stage: Sequence[tuple[str, Fraction]]) -> dict[str, float]:
    work: dict[str, float] = {}
    for pipe, amount in stage:
        if amount > 0:
            work[pipe] = work.get(pipe, 0.0) + float(amount)
    return work


def _schedule_segments(engine: Engine, name: str, segments: Sequence[OperatorSegment]) -> Any:
    """Schedule one stream of kernels.  Each kernel's latency is its own event.

    A later kernel is not released until the earlier kernel completes, so a
    floor cannot hide inside the previous kernel's body.  Two streams that
    share this engine still contend for the same pipes.
    """
    previous = None
    for seg_index, segment in enumerate(segments):
        release = (previous,) if previous is not None else ()
        if segment.native_program is not None:
            ready = engine.submit(f"{name}:{seg_index}:native_ready",
                                  after=release)
            previous = segment.native_program.start(
                engine, f"{name}:{seg_index}:native", ready)
            continue
        caps = dict(segment.caps)
        floor_event = None
        if segment.floor > 0:
            floor_event = engine.submit(
                f"{name}:{seg_index}:floor", after=release, latency=float(segment.floor))
        ready = previous
        if segment.startup > 0:
            ready = engine.submit(
                f"{name}:{seg_index}:startup", after=release, latency=float(segment.startup))
        body = ready
        for stage_index, stage in enumerate(segment.stages):
            work = _stage_work_map(stage)
            stage_caps = {pipe: float(caps[pipe]) for pipe in work if pipe in caps}
            body = engine.submit(
                f"{name}:{seg_index}:{stage_index}", work=work,
                after=(body,) if body is not None else (),
                caps=stage_caps or None)
        for stage_index, tasks in enumerate(segment.native_stages):
            task_events = []
            for task_index, task in enumerate(tasks):
                task_caps = ({task.resource: float(task.cap)}
                             if task.cap is not None else None)
                task_events.append(engine.submit(
                    f"{name}:{seg_index}:native:{stage_index}:{task_index}",
                    work={task.resource: float(task.work)},
                    after=(body,) if body is not None else (),
                    caps=task_caps))
            if task_events:
                body = engine.submit(
                    f"{name}:{seg_index}:native:{stage_index}:join",
                    after=tuple(task_events))
        if segment.dependency > 0:
            body = engine.submit(
                f"{name}:{seg_index}:dependency",
                after=(body,) if body is not None else release,
                latency=float(segment.dependency))
        waited = tuple(event for event in (body, floor_event) if event is not None)
        if not waited:
            previous = engine.submit(f"{name}:{seg_index}:empty", after=release)
        elif len(waited) == 1:
            previous = waited[0]
        else:
            previous = engine.submit(f"{name}:{seg_index}:complete", after=waited)
    if previous is None:
        previous = engine.submit(f"{name}:empty", work={})
    return previous


def _validate_stage_work(branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]]) -> None:
    for stages in branches.values():
        for stage in stages:
            for pipe, amount in stage:
                val = float(amount)
                if not math.isfinite(val) or val < 0:
                    raise ModelingError(f"engine_work_negative_or_nonfinite:{pipe}:{amount}")


def _validate_segments(chains: Mapping[str, Sequence[OperatorSegment]]) -> None:
    for name, segments in chains.items():
        for segment in segments:
            for value, label in (
                    (segment.floor, "floor"), (segment.startup, "startup"),
                    (segment.dependency, "dependency")):
                if value < 0 or not math.isfinite(float(value)):
                    raise ModelingError(f"engine_latency_invalid:{name}:{label}:{value}")
            _validate_stage_work({name: segment.stages})
            for stage in segment.native_stages:
                for task in stage:
                    if (not task.resource or not math.isfinite(float(task.work))
                            or task.work < 0):
                        raise ModelingError(
                            f"engine_native_task_invalid:{name}:{task.resource}")
                    if (task.cap is not None
                            and (not math.isfinite(float(task.cap))
                                 or task.cap <= 0)):
                        raise ModelingError(
                            f"engine_native_cap_invalid:{name}:{task.resource}")


def run_engine_work(branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]], supply: PipeSupply,
                    latencies: Mapping[str, Fraction] | None = None, *,
                    chains: Mapping[str, Sequence[OperatorSegment]] | None = None) -> EngineResult:
    """Engine execution from bound stage chains or causal operator segments.

    ``latencies`` are per-branch completion floors anchored at branch release.
    That form is one event for the whole stage list: two concurrent copies of
    one kernel finish at ``max(body, floor)``.  ``chains`` is the multi-kernel
    form.  Each segment carries its own floor, startup and dependency, and the
    next segment on that stream waits for the previous completion.
    """
    if chains is not None:
        _validate_segments(chains)
        has_program = any(segment.native_program is not None
                          for segments in chains.values() for segment in segments)
        if has_program:
            # The original program may use any declared native rate/pool.
            pipes = sorted({str(name) for name, _rate in supply.rates} | {
                pipe for segments in chains.values() for segment in segments
                for stage in segment.stages for pipe, amount in stage if amount} | {
                task.resource for segments in chains.values()
                for segment in segments for stage in segment.native_stages
                for task in stage if task.work})
        else:
            pipes = sorted({
                pipe for segments in chains.values() for segment in segments
                for stage in segment.stages for pipe, amount in stage if amount} | {
                task.resource for segments in chains.values()
                for segment in segments for stage in segment.native_stages
                for task in stage if task.work})
        units = dict(supply.resource_units)
        resources = tuple(Resource(pipe, supply.effective(pipe),
                                   _resource_units(pipe), "fair",
                                   int(units.get(pipe, 1)))
                          for pipe in pipes)
        pools = dict(supply.buffers)
        pool_units = dict(supply.pool_units)
        engine = Engine(resources, pools=pools, pool_units=pool_units)
        events = {name: _schedule_segments(engine, name, segments) for name, segments in chains.items()}
        engine.advance()
        makespan = Fraction(str(engine.now))
        completions = tuple(sorted(
            (name, Fraction(str(event.time if event.time is not None else engine.now)))
            for name, event in events.items()))
        summary = tuple(sorted(engine.summary().items()))
        return EngineResult(
            makespan, completions,
            tuple(sorted((key, Fraction(str(value))) for key, value in engine.busy_ms.items())),
            summary)
    _validate_stage_work(branches)
    latency_by_branch = (
        {str(name): _inferred_floor(stages, supply) for name, stages in branches.items()}
        if latencies is None else
        {str(name): as_fraction(value) for name, value in latencies.items()}
    )
    for name, value in latency_by_branch.items():
        if value < 0 or not math.isfinite(float(value)):
            raise ModelingError(f"engine_latency_invalid:{name}:{value}")
    pipes = sorted({pipe for stages in branches.values() for stage in stages for pipe, amount in stage if amount})
    resources = tuple(Resource(pipe, supply.effective(pipe), _resource_units(pipe), "fair") for pipe in pipes)
    engine = Engine(resources)
    events: dict[str, Any] = {}
    for name, stages in branches.items():
        previous = None
        for index, stage in enumerate(stages):
            event = engine.submit(
                f"{name}:{index}", work=_stage_work_map(stage),
                after=(previous,) if previous is not None else ())
            previous = event
        if previous is None:
            previous = engine.submit(f"{name}:empty", work={})
        floor = latency_by_branch.get(str(name), Fraction(0))
        if floor > 0:
            # One kernel, or a caller that already folded a single latency:
            # the floor starts at branch release and overlaps the body.
            floor_event = engine.submit(f"{name}:floor", after=(), latency=float(floor))
            previous = engine.submit(f"{name}:complete", after=(previous, floor_event))
        events[name] = previous
    engine.advance()
    makespan = Fraction(str(engine.now))
    completions = tuple(sorted( (name, Fraction(str(event.time if event.time is not None else engine.now)))
        for name, event in events.items()))
    summary = tuple(sorted(engine.summary().items()))
    return EngineResult(makespan, completions, tuple(sorted((key, Fraction(str(value))) for key, value in engine.busy_ms.items())), summary)

def _isolated_segment_ms(segment: OperatorSegment, supply: PipeSupply) -> Fraction:
    """No-contention completion of one kernel, including its own latency."""
    if segment.native_program is not None:
        return max(segment.floor,
                   segment.startup + segment.native_program.duration
                   + segment.dependency)
    caps = dict(segment.caps)
    body = Fraction(0)
    for stage in segment.stages:
        coalesced: dict[str, Fraction] = {}
        for pipe, amount in stage:
            if amount > 0:
                coalesced[pipe] = coalesced.get(pipe, Fraction(0)) + amount
        stage_time = Fraction(0)
        for pipe, amount in coalesced.items():
            rate = as_fraction(supply.effective(pipe))
            cap = caps.get(pipe)
            if cap is not None:
                rate = min(rate, cap)
            if rate > 0:
                stage_time = max(stage_time, amount / rate)
        body += stage_time
    for tasks in segment.native_stages:
        stage_time = Fraction(0)
        for task in tasks:
            rate = as_fraction(supply.effective(task.resource))
            if task.cap is not None:
                rate = min(rate, as_fraction(task.cap))
            if rate > 0:
                stage_time = max(stage_time, as_fraction(task.work) / rate)
        body += stage_time
    return max(segment.floor, segment.startup + body + segment.dependency)


def contention_envelope( branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]], supply: PipeSupply,
                         latencies: Mapping[str, Fraction] | None = None, *,
                         chains: Mapping[str, Sequence[OperatorSegment]] | None = None, ) -> ContentionEnvelope:
    """Compose concurrently ready branches through one shared-capacity engine.

    ``chains`` schedules each kernel's floor, startup and dependency at that
    kernel's release.  A single ``latencies`` value remains one branch-level
    floor anchored at branch release, which is the one-kernel form.
    """
    if chains is not None:
        result = run_engine_work({}, supply, chains=chains)
        branch_totals: dict[str, Fraction] = {}
        work: dict[str, Fraction] = {}
        users: dict[str, set[str]] = {}
        for name, segments in chains.items():
            branch_totals[name] = sum((_isolated_segment_ms(segment, supply) for segment in segments), Fraction(0))
            for segment in segments:
                for stage in segment.stages:
                    for pipe, amount in stage:
                        if amount > 0:
                            work[pipe] = work.get(pipe, Fraction(0)) + amount
                            users.setdefault(pipe, set()).add(name)
                for stage in segment.native_stages:
                    for task in stage:
                        if task.work > 0:
                            work[task.resource] = work.get(
                                task.resource, Fraction(0)) + task.work
                            users.setdefault(task.resource, set()).add(name)
                if segment.native_program is not None:
                    rates = dict(supply.rates)
                    modes = dict(supply.service_modes)
                    for pipe, amount in segment.native_program.work:
                        if amount <= 0:
                            continue
                        if pipe in rates:
                            resource = pipe
                        elif pipe in modes and modes[pipe][0] in rates:
                            resource = modes[pipe][0]
                        else:
                            continue
                        work[resource] = work.get(resource, Fraction(0)) + amount
                        users.setdefault(resource, set()).add(name)
        bounds = tuple(sorted((pipe, amount / supply.effective(pipe)) for pipe, amount in work.items()))
        shared = tuple(sorted(pipe for pipe, owners in users.items() if len(owners) > 1))
        disjoint = tuple(sorted(pipe for pipe, owners in users.items() if len(owners) == 1))
        isolated_max = max(branch_totals.values(), default=Fraction(0))
        return ContentionEnvelope( result.makespan_ms, isolated_max, sum(branch_totals.values(), Fraction(0)),
            tuple(sorted(work.items())), bounds, shared, disjoint, tuple(sorted(branch_totals.items())),
            result.branches)
    result = run_engine_work(branches, supply, latencies)
    latency_by_branch = (
        {str(name): _inferred_floor(stages, supply) for name, stages in branches.items()}
        if latencies is None else
        {str(name): as_fraction(value) for name, value in latencies.items()}
    )
    branch_totals = {}
    work = {}
    users = {}
    for name, stages in branches.items():
        total = Fraction(0)
        for stage in stages:
            coalesced: dict[str, Fraction] = {}
            for pipe, amount in stage:
                if amount > 0:
                    coalesced[pipe] = coalesced.get(pipe, Fraction(0)) + amount
            stage_time = max((amount / as_fraction(supply.effective(pipe)) for pipe, amount in coalesced.items()), default=Fraction(0))
            total += stage_time
            for pipe, amount in coalesced.items():
                work[pipe] = work.get(pipe, Fraction(0)) + amount
                users.setdefault(pipe, set()).add(name)
        # A single branch floor overlaps the body; it does not follow it.
        branch_totals[name] = max(total, latency_by_branch.get(str(name), Fraction(0)))
    bounds = tuple(sorted((pipe, amount / supply.effective(pipe)) for pipe, amount in work.items()))
    shared = tuple(sorted(pipe for pipe, owners in users.items() if len(owners) > 1))
    disjoint = tuple(sorted(pipe for pipe, owners in users.items() if len(owners) == 1))
    isolated_max = max(branch_totals.values(), default=Fraction(0))
    return ContentionEnvelope( result.makespan_ms, isolated_max, sum(branch_totals.values(), Fraction(0)),
        tuple(sorted(work.items())), bounds, shared, disjoint, tuple(sorted(branch_totals.items())),
        result.branches)

def pipeline_makespan(stage_ms: Sequence[Fraction | float], tiles: int, slots: int) -> Fraction:
    """Finite two-level pipeline recurrence with explicit buffer slots and O(stages + slots) memory."""
    durations = tuple(as_fraction(value) for value in stage_ms)
    if not durations:
        raise ModelingError("pipeline_stages_empty")
    if tiles < 1:
        raise ModelingError("pipeline_tiles_invalid")
    if slots < 1:
        raise ModelingError("pipeline_slots_invalid")
    if any(value < 0 for value in durations):
        raise ModelingError("pipeline_stage_time_invalid")
    slots = min(slots, tiles)
    prev_end = [Fraction(0) for _ in durations]
    slot_free = [Fraction(0) for _ in range(slots)]
    for tile in range(tiles):
        slot = tile % slots
        start = slot_free[slot] if slots < tiles else Fraction(0)
        for index, duration in enumerate(durations):
            if tile:
                start = max(start, prev_end[index])
            prev_end[index] = start + duration
            start = prev_end[index]
        slot_free[slot] = prev_end[-1]
    return max(prev_end)

def _subtract_stage(stages: list[dict[str, Fraction]], pipe: str, amount: Fraction, *, reverse: bool = False) -> Fraction:
    """Subtract up to ``amount`` of one pipe, returning the amount removed."""
    if amount <= 0:
        return Fraction(0)
    remaining = amount
    order = range(len(stages) - 1, -1, -1) if reverse else range(len(stages))
    for index in order:
        value = stages[index].get(pipe, Fraction(0))
        if value <= 0:
            continue
        take = min(value, remaining)
        stages[index][pipe] = value - take
        remaining -= take
        if remaining <= 0:
            break
    return amount - remaining

_FUSION_ALIASES = { "residual": ("residual_add_output", "residual_output", "residual"),
    "norm_output": ("rmsnorm_output", "norm_output"),
    "kv": ("kv_store_output", "kv_cache", "kv_norm_output", "rotary_output"), "kv_cache": ("kv_store_output", "kv_cache", "kv_norm_output"),
    "query": ("q_activation", "q_direct_output", "qk_norm_output", "q_projection_output", "q_lora_activation"),
    "key": ("qk_norm_output", "rotary_output", "kv_norm_output", "kv_cache"),
    "value": ("v_expand_output", "mla_attention_output", "attention_output", "rotary_output"),
    "down_input": ("activation_intermediate", "ffn_activation"),
    "mla_attention_output": ("mla_attention_output", "attention_output"),
    "attention_output": ("attention_output", "mla_attention_output"), }

def _narrow_keep_bytes(name, instance, members):
    """Size rstd and scales from their real shape, not a full activation."""
    owners = {"rstd": {"rmsnorm"}, "scale": {"rmsnorm", "activation_quant"},
              "activation_scale": {"activation", "shared_activation"}}[name]
    derived = {cost.name: dict(cost.derived) for cost in members}
    for operator in instance.operators:
        if operator.name not in owners:
            continue
        cols = int(operator.theta.find("elements", 0) or 0)
        total = as_fraction(derived.get(operator.name, {}).get("elements", 0))
        if cols <= 0 or total <= 0:
            continue
        rows = total / cols
        return rows * (4 if name == "rstd" else math.ceil(cols / 32))
    return Fraction(0)

def _fusion_rates(members: Sequence[OperatorCost]) -> dict[str, Fraction]:
    """Recover exact supply from evaluated work/time counters; never treat FLOPs/bytes as time."""
    rates: dict[str, Fraction] = {}
    for cost in members:
        work, times = dict(cost.work), dict(cost.times)
        for pipe in set(work) | set(times):
            amount, elapsed = as_fraction(work.get(pipe, 0)), as_fraction(times.get(pipe, 0))
            if amount < 0 or elapsed < 0 or (amount == 0) != (elapsed == 0):
                raise ModelingError("fusion_supply_required")
            if amount == 0:
                continue
            rate = amount / elapsed
            if pipe in rates and rates[pipe] != rate:
                raise ModelingError("fusion_supply_required")
            rates[pipe] = rate
    if not rates:
        raise ModelingError("fusion_supply_required")
    return rates

@dataclass(frozen=True)
class QkvBinding:
    """One selected QKV projection input and its typed output/view binding.

    This is the single producer -> output -> occurrence record used by both
    the first evaluation and a cache hit.  Static work stays on the costs;
    this record only says which actual instance operator produced the tensor,
    how many rows it wrote, and which slice/version each logical value names.
    """
    producer: str
    output: str
    dtype: str
    layout: str
    row_bytes: Fraction
    elements: Fraction
    version: str
    produced: Fraction
    tokens: Fraction
    q_elems: Fraction
    gate_elems: Fraction
    k_elems: Fraction
    v_elems: Fraction
    total_elems: Fraction
    act_bytes: Fraction
    post_bytes: Fraction
    q_heads: int
    kv_heads: int
    width: int
    rows: int
    query_view: tuple[str, ...]
    key_view: tuple[str, ...]
    value_view: tuple[str, ...]
    gate_view: tuple[str, ...]
    kv_view: tuple[str, ...]
    source: str
    views: tuple[tuple[str, dict[str, Any]], ...] = ()
    layer: int = 0


def _qkv_number(value: Any) -> Fraction:
    """Read a numeric contract item; serialized tuples keep strings."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ModelingError(f"fusion_qkv_numeric_invalid:{value!r}")
    try:
        return Fraction(str(value))
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"fusion_qkv_numeric_invalid:{value!r}") from exc


def _qkv_view(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return tuple(part.strip().strip("'\"") for part in text.split(",") if part.strip())
    return tuple(str(value) for value in raw)


def _qkv_declaration(instance: SubgraphInstance) -> dict[str, Any] | None:
    """Read and validate the declared ``qkv_slices`` record, if present."""
    raw = dict(instance.template.contract).get("qkv_slices")
    if not raw:
        return None
    try:
        q_elems, gate_elems, k_elems, v_elems, total_elems, act_bytes = (
            Fraction(str(value)) for value in str(raw).split(","))
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"fusion_qkv_slices_invalid:{raw}") from exc
    if (min(q_elems, total_elems, act_bytes) <= 0
            or min(gate_elems, k_elems, v_elems) < 0
            or total_elems != q_elems + gate_elems + k_elems + v_elems):
        raise ModelingError(f"fusion_qkv_slices_invalid:{raw}")
    return {
        "q_elems": q_elems, "gate_elems": gate_elems,
        "k_elems": k_elems, "v_elems": v_elems,
        "total_elems": total_elems, "act_bytes": act_bytes,
        "source": "declared_qkv_slices", "raw": str(raw),
    }


def _qkv_declaration_from_producer(instance: SubgraphInstance,
                                   producer: str,
                                   projection: Any | None = None) -> dict[str, Any] | None:
    """Reconstruct the Q/K/V split from the actual instance's typed theta.

    A public subgraph type may omit ``qkv_slices``.  The projection's ``N``,
    the attention head counts and ``qk_norm`` element width, and the declared
    gate/K=V flags are the real typed producer description.  The projection is
    the actual bound operator of this instance, never a same-named cost from
    another layer.
    """
    try:
        projection = (projection if projection is not None
                      else instance.operator(producer))
    except ModelingError:
        return None
    attention = next((operator for operator in instance.operators
                      if operator.family == "attention"
                      or operator.name == "attention"), None)
    norm = next((operator for operator in instance.operators
                 if operator.name in {"qk_norm", "rotary"}), None)
    if attention is None or norm is None:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_unsupported:{producer}")
    heads = dict(attention.theta.heads)
    q_heads = int(heads.get("q", 0) or 0)
    kv_heads = int(heads.get("kv", 0) or 0)
    if q_heads < 1 or kv_heads < 0:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_heads:{producer}")
    shared = int(attention.theta.find("shared_consumer", 0) or 0)
    gate = int(attention.theta.find("q_output_gate", 0) or 0)
    k_eq_v = int(projection.theta.find("k_eq_v", 0) or 0)
    norm_elems = as_fraction(norm.theta.find("elements", 0))
    denominator = q_heads + (0 if shared else kv_heads)
    if norm_elems <= 0 or denominator < 1:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_width:{producer}")
    width = norm_elems / denominator
    if width.denominator != 1 or width <= 0:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_width:{producer}:{width}")
    q_elems = q_heads * width
    k_elems = 0 if shared else kv_heads * width
    v_elems = 0 if shared or k_eq_v else k_elems
    gate_elems = q_elems if gate else 0
    total_elems = q_elems + gate_elems + k_elems + v_elems
    act_bytes = as_fraction(projection.theta.find("act_bytes", 0))
    if act_bytes <= 0:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_act_bytes:{producer}")
    n_width = as_fraction(projection.theta.find("N", 0))
    expected_n = (q_elems * (2 if gate else 1)
                  + (1 if k_eq_v else 2) * k_elems)
    if n_width != expected_n:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_mismatch:"
            f"{producer}:{n_width}!={expected_n}")
    return {
        "q_elems": as_fraction(q_elems), "gate_elems": as_fraction(gate_elems),
        "k_elems": as_fraction(k_elems), "v_elems": as_fraction(v_elems),
        "total_elems": as_fraction(total_elems), "act_bytes": act_bytes,
        "source": "producer_theta", "producer": producer,
        "shared_consumer": bool(shared), "k_eq_v": bool(k_eq_v),
    }


def _qkv_views(instance: SubgraphInstance, rows: int) -> dict[str, dict[str, Any]]:
    """Parse SUBGRAPHS' per-value view layout for this instance.

    The builder contract carries the actual producer/output/slice layout; PW-CORE
    must not replace it with a contiguous reconstruction.  A leading dynamic
    dimension 0 is bound to the current row count.
    """
    raw = dict(instance.template.contract).get("qkv_views") or ()
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) < 10:
            continue
        name = str(item[0])
        layout = item[9] if item[9] else item[4]
        if not isinstance(layout, (tuple, list)) or len(layout) != 4:
            continue
        shape, strides, offset, storage_shape = layout
        shape = [int(value) for value in shape]
        strides = [int(value) for value in strides]
        storage_shape = [int(value) for value in storage_shape]
        if shape and shape[0] == 0:
            shape[0] = rows
        if storage_shape and storage_shape[0] == 0:
            storage_shape[0] = rows
        result[name] = {
            "kind": str(item[1]), "producer": str(item[2]),
            "output": str(item[3]), "shape": tuple(shape),
            "strides": tuple(strides), "offset": int(offset),
            "storage_shape": tuple(storage_shape), "dtype": str(item[5]),
            "version": str(item[6]), "alias_of": str(item[7]),
            "storage": str(item[8]),
        }
    return result


def _synth_qkv_views(*, rows: int, q_heads: int, kv_heads: int,
                     width: int, total: int, q_elems: int, gate_elems: int,
                     k_elems: int, v_elems: int, gate: bool, k_eq_v: bool,
                     producer: str, output: str, dtype: str,
                     version: str) -> dict[str, dict[str, Any]]:
    """Recover legal old qkv declarations from the selected projection.

    The formula is the same typed slice relation SUBGRAPHS emits for a full
    ``qkv_views`` record: Q starts at the projection row base, optional gate is
    interleaved at one head width, K follows Q+gate and V follows K (or aliases
    K for Gemma).  It is a general layout formula, not a per-model answer table.
    """
    def view(name: str, offset: int, heads: int, stride_width: int,
             kind: str = "projection", alias: str = "",
             stride_row: int | None = None,
             storage_width: int | None = None) -> dict[str, Any]:
        row = int(total if stride_row is None else stride_row)
        storage = int(total if storage_width is None else storage_width)
        return {
            "kind": kind, "producer": producer, "output": output,
            "shape": (rows, heads, width),
            "strides": (row, stride_width, 1),
            "offset": int(offset),
            "storage_shape": (rows, storage),
            "dtype": dtype, "version": version,
            "alias_of": alias, "storage": output,
        }
    if k_eq_v:
        result = {
            "query": view("query", 0, q_heads, width,
                           stride_row=q_elems,
                           storage_width=q_elems),
            "key": view("key", 0, kv_heads, width,
                        stride_row=k_elems, storage_width=k_elems),
        }
    else:
        result = {
            "query": view("query", 0, q_heads, width),
            "key": view("key", q_elems + gate_elems, kv_heads, width),
        }
    if gate:
        result["gate"] = view("gate", width, q_heads, 2 * width)
    v_offset = q_elems + gate_elems + k_elems
    if k_eq_v:
        result["key_projection"] = view(
            "key_projection", q_elems + gate_elems, kv_heads, width)
        result["value"] = view(
            "value", q_elems + gate_elems, kv_heads, width,
            kind="alias", alias="key_projection")
    else:
        result["value"] = view("value", v_offset, kv_heads, width)
    return result


def _qkv_producer_operator(instance: SubgraphInstance, raw: Mapping[str, Any]) -> Any:
    declared = str(raw.get("qkv_producer") or "")
    if declared:
        try:
            operator = instance.operator(declared)
        except ModelingError as exc:
            raise ModelingError(
                f"fusion_qkv_producer_missing:{instance.instance_id}:"
                f"{declared}:not_in_instance") from exc
        if operator.family != "gemm":
            raise ModelingError(
                f"fusion_qkv_producer_family_mismatch:{instance.instance_id}:"
                f"{declared}:{operator.family}")
        return operator
    candidates = [operator for operator in instance.operators
                  if operator.name in {"qkv_projection", "q_projection"}]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ModelingError(
            f"fusion_qkv_producer_missing:{instance.instance_id}:"
            "qkv_projection:no_projection_operator")
    raise ModelingError(
        f"fusion_qkv_producer_ambiguous:{instance.instance_id}:"
        f"{sorted(operator.name for operator in candidates)}")


def _qkv_cost_layer(cost: OperatorCost) -> int | None:
    occurrence = getattr(cost, "native_occurrence", None)
    if isinstance(occurrence, _AbcMapping) and occurrence.get("layer") is not None:
        try:
            return int(occurrence["layer"])
        except (TypeError, ValueError):
            return None
    return None


def _qkv_member_tokens(members: Sequence[OperatorCost],
                       declaration: Mapping[str, Any]) -> Fraction | None:
    per_row = ((as_fraction(declaration["q_elems"])
                + as_fraction(declaration["k_elems"]))
               * as_fraction(declaration["act_bytes"]))
    if per_row <= 0:
        return None
    for cost in members:
        derived = dict(cost.derived)
        observed_row = as_fraction(derived.get("ub_write_row_bytes", 0))
        if observed_row <= 0:
            rows = as_fraction(derived.get("rows", 0))
            written = as_fraction(derived.get("write_bytes", 0))
            if rows > 0 and written > 0:
                observed_row = written / rows
        for name, value in (*cost.materialization, *cost.internal):
            if name in {"qk_norm_output", "rotary_output"}:
                amount = as_fraction(value)
                if amount > 0:
                    if observed_row > 0 and amount % observed_row == 0:
                        return amount / observed_row
                    if amount % per_row == 0:
                        return amount / per_row
    return None


def _qkv_producer_cost(instance: SubgraphInstance, costs: Sequence[OperatorCost],
                       producer: str, output: str, row_bytes: Fraction,
                       expected_tokens: Fraction | None,
                       expected_dtype: str = "", expected_layout: str = ""
                       ) -> tuple[OperatorCost, Fraction, Fraction]:
    candidates: list[tuple[OperatorCost, Fraction, Fraction]] = []
    rejected: list[str] = []
    for cost in costs:
        if cost.name != producer:
            continue
        layer = _qkv_cost_layer(cost)
        if layer is not None and layer != instance.layer:
            rejected.append(f"{cost.name}@layer:{layer}")
            continue
        occurrence = getattr(cost, "native_occurrence", None) or {}
        typed = dict(occurrence.get("typed") or {}) if isinstance(occurrence, _AbcMapping) else {}
        attrs = dict(typed.get("attrs") or {}) if isinstance(typed, _AbcMapping) else {}
        # Only an explicitly declared output dtype/layout is authoritative;
        # the top-level typed dtype can be a quantized input/weight payload.
        dtype = str(attrs.get("output_dtype") or attrs.get("out_dtype") or "")
        layout = str(attrs.get("output_layout") or attrs.get("out_layout") or "")
        if expected_dtype and dtype and dtype != expected_dtype:
            rejected.append(f"{cost.name}@layer:{layer}:dtype:{dtype}")
            continue
        if expected_layout and layout and layout != expected_layout:
            rejected.append(f"{cost.name}@layer:{layer}:layout:{layout}")
            continue
        values = dict(cost.materialization)
        value = as_fraction(values.get(output, cost.internal and dict(cost.internal).get(output, 0)))
        if value <= 0:
            rejected.append(f"{cost.name}@layer:{layer}:no:{output}")
            continue
        if row_bytes <= 0 or value % row_bytes != 0:
            rejected.append(f"{cost.name}@layer:{layer}:row_bytes:{value}")
            continue
        tokens = value / row_bytes
        if expected_tokens is not None and tokens != expected_tokens:
            rejected.append(
                f"{cost.name}@layer:{layer}:tokens:{tokens}!={expected_tokens}")
            continue
        candidates.append((cost, value, tokens))
    if len(candidates) == 1:
        return candidates[0]
    detail = ",".join(rejected) or "no_positive_producer_output"
    if not candidates:
        raise ModelingError(
            f"fusion_qkv_producer_missing:{instance.instance_id}:"
            f"{producer}:{output}:{detail}")
    raise ModelingError(
        f"fusion_qkv_producer_ambiguous:{instance.instance_id}:"
        f"{producer}:{output}:{detail}")


def _qkv_post_bytes(members: Sequence[OperatorCost], declaration: Mapping[str, Any],
                    default: Fraction) -> Fraction:
    """Post-norm Q/K element width when it differs from projection width."""
    per_row = (as_fraction(declaration["q_elems"])
               + as_fraction(declaration["k_elems"]))
    if per_row <= 0:
        return default
    for cost in members:
        derived = dict(cost.derived)
        row = as_fraction(derived.get("ub_write_row_bytes", 0))
        if row <= 0:
            rows = as_fraction(derived.get("rows", 0))
            written = as_fraction(derived.get("write_bytes", 0))
            if rows > 0 and written > 0:
                row = written / rows
        if row <= 0:
            continue
        value = row / per_row
        if value > 0 and value.denominator == 1:
            return value
    return default


def _qkv_binding(instance: SubgraphInstance, costs: Sequence[OperatorCost],
                 contract: Any, members: Sequence[OperatorCost]) -> QkvBinding:
    raw = dict(instance.template.contract)
    projection = _qkv_producer_operator(instance, raw)
    producer = projection.name
    declaration = _qkv_declaration(instance)
    source = "declared_qkv_slices"
    if declaration is None:
        declaration = _qkv_declaration_from_producer(
            instance, producer, projection=projection)
        source = "producer_theta"
    if declaration is None:
        raise ModelingError(
            f"fusion_qkv_slices_reconstruction_unsupported:{producer}")
    total = as_fraction(declaration["total_elems"])
    act_bytes = as_fraction(declaration["act_bytes"])
    output = str(raw.get("qkv_output")
                 or ("q_projection_output" if producer == "q_projection"
                     else "qkv_activation"))
    elements = _qkv_number(raw.get("qkv_output_elements", total))
    if elements != total:
        raise ModelingError(
            f"fusion_qkv_output_mismatch:{instance.instance_id}:"
            f"elements:{elements}!={total}")
    row_bytes = _qkv_number(raw.get("qkv_output_row_bytes", total * act_bytes))
    if row_bytes != total * act_bytes:
        raise ModelingError(
            f"fusion_qkv_output_mismatch:{instance.instance_id}:"
            f"row_bytes:{row_bytes}!={total * act_bytes}")
    expected_tokens = _qkv_member_tokens(members, declaration)
    dtype = str(raw.get("qkv_output_dtype") or "bf16")
    layout_raw = raw.get("qkv_output_layout") or "ND"
    layout = (str(layout_raw[0]) if isinstance(layout_raw, (tuple, list))
              else str(layout_raw))
    _cost, produced, tokens = _qkv_producer_cost(
        instance, costs, producer, output, row_bytes, expected_tokens,
        expected_dtype=dtype, expected_layout=layout)
    if tokens <= 0:
        raise ModelingError(
            f"fusion_qkv_producer_missing:{instance.instance_id}:"
            f"{producer}:{output}:zero_rows")
    attention = next((operator for operator in instance.operators
                      if operator.family == "attention"
                      or operator.name == "attention"), None)
    gate = bool(int(attention.theta.find("q_output_gate", 0) or 0)) if attention else False
    shared = bool(int(attention.theta.find("shared_consumer", 0) or 0)) if attention else False
    k_eq_v = bool(int(projection.theta.find("k_eq_v", 0) or 0))
    fused_name = "fused:qkv_norm_rope"
    query_view = _qkv_view(raw.get("qkv_query_view")) or (
        "query", "fused", fused_name, "query", "q_slice", "post_rope")
    key_view = _qkv_view(raw.get("qkv_key_view")) or (
        "key", "fused", fused_name, "key", "", "post_rope")
    if shared:
        value_view = _qkv_view(raw.get("qkv_value_view")) or (
            "value", "state", "kv_cache", "kv_owner", "", "pre_rope")
    elif k_eq_v:
        value_view = _qkv_view(raw.get("qkv_value_view")) or (
            "value", "alias", producer, output, "k_slice", "pre_rope")
    else:
        value_view = _qkv_view(raw.get("qkv_value_view")) or (
            "value", "projection", producer, output, "v_slice", "pre_rope")
    gate_view = (_qkv_view(raw.get("qkv_gate_view"))
                 or ("gate", "projection", producer, output,
                     "gate_slice", "pre_rope")) if gate else ()
    kv_view = _qkv_view(raw.get("qkv_kv_view")) or (
        "kv", "aggregate", "key", "value")
    heads = dict(attention.theta.heads) if attention is not None else {}
    q_heads = int(heads.get("q", 0) or 0)
    kv_heads = int(heads.get("kv", 0) or 0)
    if q_heads < 1:
        raise ModelingError(
            f"fusion_qkv_shape_heads:{instance.instance_id}:{q_heads}")
    if tokens.denominator != 1:
        raise ModelingError(
            f"fusion_qkv_rows_nonintegral:{instance.instance_id}:{tokens}")
    rows = int(tokens)
    width = as_fraction(declaration["q_elems"]) / q_heads
    if width.denominator != 1 or width <= 0:
        raise ModelingError(
            f"fusion_qkv_shape_width:{instance.instance_id}:{width}")
    width = int(width)
    views = _qkv_views(instance, rows)
    if not views:
        views = _synth_qkv_views(
            rows=rows, q_heads=q_heads, kv_heads=kv_heads, width=width,
            total=int(total), q_elems=int(as_fraction(declaration["q_elems"])),
            gate_elems=int(as_fraction(declaration["gate_elems"])),
            k_elems=int(as_fraction(declaration["k_elems"])),
            v_elems=int(as_fraction(declaration["v_elems"])),
            gate=gate, k_eq_v=k_eq_v, producer=producer, output=output,
            dtype=dtype, version=str(raw.get("qkv_version") or "pre_rope"))
    return QkvBinding(
        producer=producer, output=output,
        dtype=dtype, layout=layout,
        row_bytes=row_bytes, elements=elements,
        version=str(raw.get("qkv_version") or "pre_rope"),
        produced=produced, tokens=tokens,
        q_elems=as_fraction(declaration["q_elems"]),
        gate_elems=as_fraction(declaration["gate_elems"]),
        k_elems=as_fraction(declaration["k_elems"]),
        v_elems=as_fraction(declaration["v_elems"]),
        total_elems=total, act_bytes=act_bytes,
        post_bytes=_qkv_post_bytes(members, declaration, act_bytes),
        q_heads=q_heads, kv_heads=kv_heads, width=width, rows=rows,
        query_view=query_view, key_view=key_view,
        value_view=value_view, gate_view=gate_view, kv_view=kv_view,
        source=source,
        views=tuple(sorted(views.items())), layer=int(instance.layer))


def _qkv_slices(binding: QkvBinding | None, contract_name: str):
    """Typed logical Q/K/V/gate and this launch's physical write split."""
    if contract_name != "qkv_norm_rope" or binding is None:
        return None
    scale = binding.tokens * binding.act_bytes
    post_scale = binding.tokens * binding.post_bytes
    query_bytes = binding.q_elems * post_scale
    gate_bytes = binding.gate_elems * scale
    key_bytes = binding.k_elems * post_scale
    stored_value = binding.v_elems * scale
    alias_value = (binding.value_view[1] == "alias"
                   if len(binding.value_view) > 1 else False)
    value_bytes = key_bytes if alias_value else stored_value
    kv_bytes = 0 if alias_value else key_bytes + value_bytes
    return {
        "tokens": binding.tokens,
        "query": query_bytes, "gate": gate_bytes, "key": key_bytes,
        "value": value_bytes, "stored_value": stored_value,
        "value_aliases_key": alias_value, "kv": kv_bytes,
        "qk": (binding.q_elems + binding.k_elems) * post_scale,
        "qk_input": (binding.q_elems + binding.k_elems) * scale,
        "produced": binding.produced, "producer": binding.producer,
        "output": binding.output, "dtype": binding.dtype,
        "layout": binding.layout, "version": binding.version,
        "declaration_source": binding.source,
        "binding": binding,
    }


def _qkv_fusion_values(binding: QkvBinding, slices: Mapping[str, Any]
                       ) -> tuple[tuple[FusionValue, ...], dict[str, Fraction]]:
    """Project the actual bound TensorValues onto the fusion ledger.

    SUBGRAPHS owns the per-value producer/output/slice layout.  PW-CORE keeps
    that ``TensorValue`` layout (offset/strides/storage_shape), including on a
    cache hit; it never reconstructs a surviving slice as a contiguous buffer.
    """
    def split(view: tuple[str, ...], default_producer: str,
              default_output: str, default_version: str
              ) -> tuple[str, str, str, str]:
        kind = view[1] if len(view) > 1 else "logical"
        producer = view[2] if len(view) > 2 and view[2] else default_producer
        output = view[3] if len(view) > 3 and view[3] else default_output
        version = view[5] if len(view) > 5 and view[5] else default_version
        return kind, producer, output, version

    views = dict(binding.views)

    def bound(name: str, fallback_producer: str, fallback_output: str,
              fallback_shape: tuple[int, ...], *,
              fallback_alias: str = "", fallback_version: str = ""
              ) -> tuple[TensorValue, str, str, str]:
        view = views.get(name)
        if view is None:
            producer, output = fallback_producer, fallback_output
            version, alias = fallback_version, fallback_alias
            layout = TensorLayout.contiguous(fallback_shape)
            dtype = binding.dtype
            storage = fallback_output
        else:
            producer = str(view["producer"] or fallback_producer)
            output = str(view["output"] or fallback_output)
            version = str(view["version"] or fallback_version)
            alias = str(view["alias_of"] or fallback_alias)
            dtype = str(view["dtype"] or binding.dtype)
            shape = tuple(int(value) for value in view["shape"])
            strides = tuple(int(value) for value in view["strides"])
            storage_shape = tuple(int(value) for value in view["storage_shape"])
            if not shape:
                shape = fallback_shape
            if not storage_shape:
                storage_shape = shape
            layout = TensorLayout(shape, strides, int(view["offset"]),
                                  storage_shape)
            storage = str(view["storage"] or output)
        value = TensorValue(
            producer=producer, dtype=dtype, layout=layout, output=output,
            layout_kind=(binding.layout.lower() or "strided"),
            storage=storage, alias_of=alias)
        return value, producer, output, version

    entries: list[FusionValue] = []
    extra: dict[str, Fraction] = {}
    q_kind, q_producer, q_output, q_version = split(
        binding.query_view, binding.producer, binding.output, "post_rope")
    q_value, q_producer, q_output, q_version = bound(
        "query", q_producer, q_output,
        (binding.rows, binding.q_heads, binding.width),
        fallback_version=q_version)
    entries.append(FusionValue(
        "query", q_value, slices["query"], "logical", slices["query"],
        version=q_version, layer=binding.layer))
    k_kind, k_producer, k_output, k_version = split(
        binding.key_view, binding.producer, binding.output, "post_rope")
    k_value, k_producer, k_output, k_version = bound(
        "key", k_producer, k_output,
        (binding.rows, binding.kv_heads, binding.width),
        fallback_version=k_version)
    entries.append(FusionValue(
        "key", k_value, slices["key"], "logical", slices["key"],
        version=k_version, layer=binding.layer))
    v_kind, v_producer, v_output, v_version = split(
        binding.value_view, binding.producer, binding.output, "pre_rope")
    if slices["value"] > 0:
        v_value, v_producer, v_output, v_version = bound(
            "value", v_producer, v_output,
            (binding.rows, binding.kv_heads, binding.width),
            fallback_alias=("key_projection"
                            if slices["value_aliases_key"] else ""),
            fallback_version=v_version)
        if v_kind == "alias" or slices["value_aliases_key"]:
            v_value = replace(v_value, alias_of="projection_k")
            kp_value, kp_producer, kp_output, kp_version = bound(
                "key_projection", v_producer, v_output,
                (binding.rows, binding.kv_heads, binding.width),
                fallback_version=v_version)
            entries.append(FusionValue(
                "projection_k", kp_value, slices["key"], "logical", 0,
                version=kp_version, layer=binding.layer))
            extra["projection_k"] = slices["key"]
            entries.append(FusionValue(
                "value", v_value, slices["value"], "alias", 0,
                version=v_version, layer=binding.layer))
        else:
            entries.append(FusionValue(
                "value", v_value, slices["value"], "logical", 0,
                version=v_version, layer=binding.layer))
    if slices["gate"] > 0:
        g_kind, g_producer, g_output, g_version = split(
            binding.gate_view, binding.producer, binding.output, "pre_rope")
        g_value, g_producer, g_output, g_version = bound(
            "gate", g_producer, g_output,
            (binding.rows, binding.q_heads, binding.width),
            fallback_version=g_version)
        entries.append(FusionValue(
            "gate", g_value, slices["gate"], "logical", 0,
            version=g_version, layer=binding.layer))
    if slices["value_aliases_key"]:
        kv_view_of = "projection_k"
    else:
        kv_view_of = "key,value"
    kv_value, _kp, _ko, _kv = bound(
        "kv", "kv_cache", "kv",
        (binding.rows, max(0, binding.kv_heads), binding.width),
        fallback_version=binding.version)
    entries.append(FusionValue(
        "kv", kv_value, slices["kv"], "aggregate", 0,
        view_of=kv_view_of, version=binding.version, layer=binding.layer))
    return tuple(entries), extra


def lower_fusion(instance: SubgraphInstance, costs: Sequence[OperatorCost],
                 contract_name: str, *, supply: PipeSupply | None = None) -> FusionResult:
    """Lower one declared native-fusion group to a physical kernel."""
    contract = next((item for item in instance.template.fusions
                     if item.name == contract_name), None)
    if contract is None:
        raise ModelingError(f"fusion_contract_missing:{instance.instance_id}:{contract_name}")
    by_name = {cost.name: cost for cost in costs}
    missing = [name for name in contract.members if name not in by_name]
    if missing:
        raise ModelingError(f"fusion_members_missing:{contract_name}:{sorted(missing)}")
    members = [by_name[name] for name in contract.members]
    rates = _fusion_rates(members) if supply is None else None
    streams = {operator.stream for operator in instance.operators if operator.name in contract.members}
    if len(streams) > 1:
        raise ModelingError(f"fusion_crosses_stream:{contract_name}:{sorted(streams)}")
    if contract.stream not in streams and streams:
        raise ModelingError(f"fusion_stream_mismatch:{contract_name}")

    outputs: dict[str, Fraction] = {}
    producer: dict[str, int] = {}
    for index, cost in enumerate(members):
        for name, value in (*cost.materialization, *cost.internal):
            outputs[name] = outputs.get(name, Fraction(0)) + value
            if value > 0:
                producer.setdefault(name, index)
    # One actual-instance producer -> output -> occurrence binding is shared
    # by first evaluation and cache hits.  It is never selected by name order
    # or the largest same-named ledger from another layer.
    qkv_binding = (_qkv_binding(instance, costs, contract, members)
                   if contract_name == "qkv_norm_rope" else None)
    slices = _qkv_slices(qkv_binding, contract_name)

    kept_names: dict[str, str | None] = {}
    external: set[str] = set()
    for name in contract.keeps:
        if slices is not None and name in {"query", "key", "value", "gate", "kv"}:
            kept_names[name] = None
            continue
        candidates = _FUSION_ALIASES.get(name, (name,))
        chosen = next((candidate for candidate in candidates if outputs.get(candidate, Fraction(0)) > 0), None)
        kept_names[name] = chosen
        if chosen is not None:
            external.add(chosen)
        if name in outputs:
            external.add(name)
    for name, _ in (*members[-1].materialization, *members[-1].internal):
        external.add(name)
    removed_names = [name for name in contract.removes if name in outputs]
    write_removed = {name for name in removed_names if name not in external}

    kept: dict[str, Fraction] = {}
    for cost in members:
        for name, value in cost.materialization:
            if name not in write_removed:
                kept[name] = kept.get(name, Fraction(0)) + value
    for name in contract.keeps:
        if name in kept or (slices is not None and name in {"query", "key", "value", "gate", "kv"}):
            continue
        if name in {"rstd", "scale", "activation_scale"}:
            kept[name] = _narrow_keep_bytes(name, instance, members)
            continue
        chosen = kept_names[name]
        kept[name] = outputs.get(chosen, Fraction(0)) if chosen is not None else Fraction(0)
    if slices is not None:
        for stale in ("rotary_output", "qk_norm_output", "qkv_activation"):
            kept.pop(stale, None)
        kept["query"] = slices["query"]
        kept["key"] = slices["key"]
        # The logical V is always the full value.  Gemma K=V aliases the
        # projection's pre-RoPE K slice (a separate logical value from the
        # fused post-RoPE K) and has zero extra physical V write.
        if slices["value"] > 0:
            kept["value"] = slices["value"]
        if slices["value_aliases_key"]:
            kept["projection_k"] = slices["key"]
        if slices["gate"] > 0:
            kept["gate"] = slices["gate"]
        kept["kv"] = slices["kv"]

    stages_by_member: list[list[dict[str, Fraction]]] = []
    removed: dict[str, Fraction] = {}
    kept_startup = False
    has_cube_startup = any(
        len(cost.stage_work) > 1 and len(cost.stage_work[-1]) == 1 and "cube" in dict(cost.stage_work[-1])
        for cost in members
    )
    for cost in members:
        base = ([dict(stage) for stage in cost.stage_work] if cost.stage_work else [dict(cost.work)])
        if contract.startup in {"single", 1} and len(base) > 1 and len(base[-1]) == 1:
            if kept_startup or (has_cube_startup and "cube" not in base[-1]):
                base = base[:-1]
            else:
                kept_startup = True
        stages_by_member.append(base)
    for name in removed_names:
        owner = producer.get(name)
        if owner is None:
            continue
        produced = sum((value for key, value in (*members[owner].materialization, *members[owner].internal) if key == name), Fraction(0))
        if produced <= 0:
            continue
        removed_here = Fraction(0)
        if slices is not None and name in {"qkv_activation", "q_projection_output"}:
            # Only the projection's intermediate Q/K write disappears; the
            # gate and V slices stay stored, and norm/rotary republish Q/K.
            # HBM is paired with the MTE bytes actually removed so a narrower
            # consumer cannot eat the producer's weight traffic or its own
            # surviving store.
            wrote = _subtract_stage(stages_by_member[owner], "mte3", slices["qk_input"], reverse=True)
            _subtract_stage(stages_by_member[owner], "hbm", wrote, reverse=True)
            removed_here += wrote
            if owner < len(members) - 1:
                read = _subtract_stage(stages_by_member[owner + 1], "mte2", produced)
                _subtract_stage(stages_by_member[owner + 1], "hbm", read)
                removed_here += read
            removed[name] = removed.get(name, Fraction(0)) + removed_here
            continue
        if name in write_removed:
            wrote = _subtract_stage(stages_by_member[owner], "mte3", produced, reverse=True)
            _subtract_stage(stages_by_member[owner], "hbm", wrote, reverse=True)
            removed_here += wrote
        if owner < len(members) - 1:
            read = _subtract_stage(stages_by_member[owner + 1], "mte2", produced)
            _subtract_stage(stages_by_member[owner + 1], "hbm", read)
            removed_here += read
        if removed_here > 0:
            removed[name] = removed.get(name, Fraction(0)) + removed_here

    adjustment = _sfa_visible_adjustment(instance, outputs) if contract_name == "mla_prolog" else None
    if adjustment is not None:
        # The seven members stay the internal GEMM, RMS, RoPE and store.
        # q_norm was forwarded into q_up above. The unabsorbed q_up write has
        # no fusion consumer, so only that write leaves. The cache capacity
        # is not a kept output; the packed-slot delta is store traffic.
        q_act = outputs.get("q_activation", Fraction(0))
        q_owner = producer.get("q_activation")
        if q_act > 0 and q_owner is not None:
            wrote = _subtract_stage(stages_by_member[q_owner], "mte3", q_act, reverse=True)
            _subtract_stage(stages_by_member[q_owner], "hbm", wrote, reverse=True)
            if wrote > 0:
                removed["q_activation"] = removed.get("q_activation", Fraction(0)) + wrote
        kept = {name: adjustment["kept_" + name] for name in
                ("ql_nope", "q_pe", "q_c", "q_c_scale", "cache_index")}
        store = stages_by_member[-1]
        delta = adjustment["store_delta"]
        if delta > 0:
            store.append({"mte3": delta, "hbm": delta})
        elif delta < 0:
            wrote = _subtract_stage(store, "mte3", -delta, reverse=True)
            _subtract_stage(store, "hbm", wrote, reverse=True)
        if adjustment["index_read"] > 0:
            store.append({"mte2": adjustment["index_read"], "hbm": adjustment["index_read"]})
        if adjustment["visible_write"] > 0:
            store.append({"mte3": adjustment["visible_write"], "hbm": adjustment["visible_write"]})

    native_prolog = dict(getattr(instance.template, "contract", ()) or ()).get("mla_prolog_v3")
    stage_rate_override: dict[int, dict[str, Fraction]] = {}
    bf16_peak = None
    if supply is not None:
        bf16_peak = dict(getattr(supply, "compute_peaks", ()) or ()).get("bf16")
        bf16_peak = Fraction(str(bf16_peak)) if bf16_peak else None
    if contract_name == "mla_prolog" and native_prolog:
        # One genuine fourth product: Qup -> W_UK (BF16), absorbed into the
        # visible ql_nope.  It is not a fifth generic GEMM member and its
        # output is not re-counted when visible_adjustment already pays it.
        parts = dict(item.split("=", 1) for item in str(native_prolog).split(":") if "=" in item)
        wuk = str(parts.get("W_UK", "64,192,512"))
        try:
            bh, kk, nn = (int(v) for v in wuk.split(","))
        except ValueError as exc:
            raise ModelingError("mla_prolog_v3_wuk_shape_invalid") from exc
        q_up_cost = next((cost for cost in members if cost.name == "q_up"), None)
        q_up_op = next((op for op in instance.operators if op.name == "q_up"), None)
        if q_up_cost is not None and q_up_op is not None:
            q_width = int(q_up_op.theta.find("N", 0) or 0)
            q_act = Fraction(outputs.get("q_activation", 0))
            tokens = (q_act // (q_width * 2)) if q_width > 0 else Fraction(0)
            if tokens > 0:
                flops = Fraction(2) * tokens * bh * kk * nn
                weight = Fraction(bh * kk * nn * 2)
                act = Fraction(tokens * bh * kk * 2)
                out = Fraction(tokens * bh * nn * 2)
                # Public work remains the true BF16 FLOPs.  Only this stage's
                # cube service uses the declared BF16 peak; never scale the
                # physical numerator to imitate a slower rate.
                stage: dict[str, Fraction] = {"cube": flops,
                                              "mte2": weight + act,
                                              "hbm": weight + act}
                if adjustment is None:
                    stage["mte3"] = out
                    stage["hbm"] += out
                if bf16_peak is not None:
                    stage_rate_override[id(stage)] = {"cube": bf16_peak}
                q_index = next((i for i, cost in enumerate(members)
                                if cost.name == "q_up"), None)
                if q_index is not None:
                    stages_by_member.insert(q_index + 1, [stage])
    flat: list[tuple[tuple[str, Fraction], ...]] = []
    flat_rates: list[dict[str, Fraction]] = []
    work: dict[str, Fraction] = {}
    for stages in stages_by_member:
        for stage in stages:
            entries = tuple(sorted((pipe, value) for pipe, value in stage.items() if value > 0))
            flat.append(entries)
            flat_rates.append(stage_rate_override.get(id(stage), {}))
            for pipe, value in entries:
                work[pipe] = work.get(pipe, Fraction(0)) + value
    def rate(pipe: str) -> Fraction:
        if supply is not None:
            return Fraction(str(supply.effective(pipe)))
        return rates[pipe]

    stage_times: list[Fraction] = []
    times: dict[str, Fraction] = {}
    peak: dict[str, Fraction] = {}
    for index, entries in enumerate(flat):
        overrides = flat_rates[index]
        stage_time = max((value / overrides.get(pipe, rate(pipe))
                          for pipe, value in entries), default=Fraction(0))
        stage_times.append(stage_time)
        for pipe, value in entries:
            elapsed = value / overrides.get(pipe, rate(pipe))
            times[pipe] = times.get(pipe, Fraction(0)) + elapsed
            peak[pipe] = max(peak.get(pipe, Fraction(0)), elapsed)
    body = sum(stage_times, Fraction(0))
    # One fused launch keeps the strictest declared floor.  Startup and
    # propagation stay latency: they are not summed across members and they
    # do not re-enter the work ledger.
    floor = startup = dependency = Fraction(0)
    for cost in members:
        fields = dict(cost.derived)
        floor = max(floor, as_fraction(fields.get("service_floor_ms", 0)))
        startup = max(startup, as_fraction(fields.get("service_startup_ms", 0)))
        dependency = max(dependency, as_fraction(fields.get("service_dependency_ms", 0)))
    duration = max(floor, startup + body + dependency)
    if contract_name == "add_rmsnorm":
        # Source AddRmsNorm physically writes norm, residual and F32 rstd
        # only; the member-local names are consumed inside the launch.
        kept = {name: value for name, value in kept.items()
                if name in {"norm_output", "residual", "rstd"}}
    removed_total = sum(removed.values(), Fraction(0))
    fusion_values: tuple[FusionValue, ...] = ()
    if slices is not None:
        fusion_values, _extra = _qkv_fusion_values(
            slices["binding"], slices)
    if fusion_values:
        # Only the fused launch's own stores are physical materialisation.
        # The independent QBMM's V/gate writes and Gemma's alias V are not
        # re-charged here; they are still present in the logical ledger.
        material = tuple(sorted(
            (entry.name, entry.written_bytes) for entry in fusion_values
            if entry.written_bytes > 0))
        ledger_names = {entry.name for entry in fusion_values}
        kept_total = sum(
            (entry.logical_bytes for entry in fusion_values
             if entry.role != "aggregate" and not entry.alias_of),
            Fraction(0))
        kept_total += sum((value for name, value in kept.items()
                           if name not in ledger_names), Fraction(0))
        kept_outputs = len([entry for entry in fusion_values
                            if entry.role != "aggregate"])
        kept_outputs += len([name for name in kept
                             if name not in ledger_names])
    else:
        material = tuple(sorted(kept.items()))
        kept_total = sum(kept.values(), Fraction(0))
        kept_outputs = len(kept)
    dominant = max(peak, key=peak.get) if peak else ""
    tied = tuple(pipe for pipe, value in peak.items() if peak and value == max(peak.values()))
    derived = { "fused": Fraction(1), "physical_kernels": Fraction(1),
        "kept_outputs": Fraction(max(1, kept_outputs)),
        "removed_bytes": removed_total, "kept_bytes": kept_total,
        "member_stages": Fraction(len(flat)), }
    if floor > 0 or startup > 0 or dependency > 0:
        derived["service_floor_ms"] = floor
        derived["service_startup_ms"] = startup
        derived["service_dependency_ms"] = dependency
    fused = OperatorCost( name=f"fused:{contract_name}", family="fused", kind=instance.kind, segment_index=-1, regime=(), guard_cells=(),
        work=tuple(sorted(work.items())), times=tuple(sorted(times.items())), stage_times=tuple(stage_times), duration_ms=duration,
        stage_work=tuple(flat), dominant_pipe=dominant, tied_pipes=tied, materialization=material, internal=(),
        residency_bytes=sum((cost.residency_bytes for cost in members), Fraction(0)),
        derived=tuple(sorted(derived.items())), formula=f"fused({contract_name})", fidelity="fused_physical_kernel", )
    from .native_calls import attach_fusion_native
    fused = attach_fusion_native(fused, instance, contract, tuple(members))
    return FusionResult(contract_name, tuple(members), fused, tuple(sorted(removed.items())), tuple(sorted(kept.items())), contract.startup,
                        sum((as_fraction(cost.duration_ms) for cost in members), Fraction(0)),
                        fusion_values)

@dataclass(frozen=True)
class SubmittedOperator:
    """Events submitted to a caller-owned Execution for one compiled cost."""
    name: str
    events: tuple[Any, ...]
    native: bool = False
    work: tuple[tuple[str, float], ...] = ()
    entering_state: Any = None


def _execution_resource_bindings(execution: Any, replica: int, stage: int
                                 ) -> dict[str, str]:
    """Map piecewise pipe names onto this Execution stage's physical names."""
    suffix = f"@r{replica}:s{stage}"
    bindings: dict[str, str] = {}
    aliases = {"matrix": "cube", "memory": "hbm"}
    engine = getattr(execution, "engine", execution)
    for name in engine.resources:
        if not name.endswith(suffix):
            continue
        base = name.split("@", 1)[0]
        target = aliases.get(base, base)
        if target in {"cube", "hbm", "vector", "mte1", "mte2", "mte3",
                      "fixpipe", "host", "link"}:
            bindings[target] = name
    if "link" not in bindings and "link:d2d" in engine.resources:
        bindings["link"] = "link:d2d"
    return bindings


def submit_cost(cost: OperatorCost, execution: Any, *, name: str,
                ready: Any = None, replica: int = 0, stage: int = 0,
                supply: PipeSupply | None = None,
                resource_bindings: Mapping[str, str] | None = None,
                unit_conversions: Mapping[tuple[str, str], float] | None = None,
                operator: Any | None = None, geometry: Any | None = None,
                state: Any | None = None, memory: Any | None = None,
                native_kwargs: Mapping[str, Any] | None = None,
                conditions: Mapping[str, bool] | None = None,
                holds: Mapping[str, Any] | None = None,
                event_bindings: Mapping[str, Any] | None = None,
                ) -> SubmittedOperator:
    """Submit one evaluated cost to the caller's Execution.

    A native owner program starts through the existing
    ``NativeProgram.start``; an analytical multi-stage cost becomes one
    ``ProgramFragment`` per actual stage in the original order.  No duration is
    converted back into pipe work and no private Engine is created here.
    """
    if not hasattr(execution, "engine"):
        raise ModelingError("submit_cost_requires_execution")
    if ready is None:
        ready = execution.engine.submit(f"{name}:ready", work={})
    bindings = dict(resource_bindings
                    if resource_bindings is not None else
                    _execution_resource_bindings(execution, replica, stage))
    supplied = supply if supply is not None else getattr(execution, "supply", None)
    conversions = dict(unit_conversions or {})
    if "cube" in bindings and supplied is not None:
        conversions.setdefault(("flops", "peak_ms"),
                               float(supplied.effective("cube")))
    native = cost.native_program
    if native is not None:
        # A selected native program is authoritative: never replace it with a
        # re-evaluated family fragment/work-only substitute.
        if isinstance(ready, (tuple, list)):
            ready = execution.engine.submit(
                f"{name}:native_ready_join", after=tuple(ready))
        event = native.start(execution.engine, name, ready,
                             **dict(native_kwargs or {}))
        return SubmittedOperator(
            name, (event,), native=True,
            work=tuple((str(pipe), float(as_fraction(amount)))
                       for pipe, amount in cost.work if as_fraction(amount) > 0),
            entering_state=state)

    # Otherwise consume the family's actual per-stage fragments (ports,
    # events, effects and allocations).  Multi-stage programs are never
    # rebuilt from aggregate work.
    if operator is not None and geometry is not None:
        from .families_bridge import family_work
        try:
            family = family_work(operator, geometry, state)
        except ModelingError:
            family = None
        if family is not None:
            fragments = family.fragments()
            conductor = execution.fragment_conductor(
                replica=replica, stage=stage,
                resource_bindings=bindings or None,
                unit_conversions=conversions or None,
            )
            current = ready
            events: list[Any] = []
            submitted_work: list[tuple[str, float]] = []
            for index, fragment in enumerate(fragments):
                fragment = replace(
                    fragment, name=f"{name}:{fragment.name}")
                inputs = {
                    port.name: current
                    for group in ("tensor_in", "state_in", "event_in")
                    for port in getattr(fragment.ports, group)
                    if current is not None
                }
                if memory is not None and fragment.allocations:
                    if not memory.attach_allocation_effects(
                            fragment.allocations, replica=replica, stage=stage):
                        raise ModelingError("submission_allocation_rejected")
                after = (tuple(current) if isinstance(current, (tuple, list))
                         else ((current,) if current else ()))
                handle = conductor.submit(
                    fragment, after=after, inputs=inputs,
                    conditions=conditions, holds=holds,
                    event_bindings=event_bindings)
                events.append(handle.event)
                current = handle.event
                submitted_work.extend(
                    (str(item.resource), float(item.amount))
                    for item in fragment.work.items if item.amount > 0)
            return SubmittedOperator(name, tuple(events), native=False,
                                     work=tuple(submitted_work),
                                     entering_state=state)
    stages = (tuple(cost.stage_work) if cost.stage_work
              else (tuple(cost.work),))
    conductor = execution.fragment_conductor(
        replica=replica, stage=stage,
        resource_bindings=bindings or None,
        unit_conversions=conversions or None,
    )
    events = []
    submitted_work: list[tuple[str, float]] = []
    current = ready
    for index, entries in enumerate(stages):
        items = tuple(
            WorkItem(str(pipe), float(as_fraction(amount)),
                     _PIPE_UNITS.get(str(pipe), "work"))
            for pipe, amount in entries if as_fraction(amount) > 0)
        if not items:
            continue
        fragment = ProgramFragment(
            name=f"{name}:s{index}", work=WorkSet(items),
            notes=("piecewise.compiler", "actual_instance_cost"))
        after = (tuple(current) if isinstance(current, (tuple, list))
                 else ((current,) if current else ()))
        handle = conductor.submit(fragment, after=after)
        events.append(handle.event)
        current = handle.event
        submitted_work.extend((str(pipe), float(as_fraction(amount)))
                              for pipe, amount in entries
                              if as_fraction(amount) > 0)
    return SubmittedOperator(name, tuple(events), native=False,
                             work=tuple(submitted_work),
                             entering_state=state)


def _catalog_resource_bindings(execution: Any, replica: int,
                               stage: int) -> dict[str, str]:
    """Map catalog role names to this Execution's physical resources."""
    engine = getattr(execution, "engine", execution)
    suffix = f"@r{replica}:s{stage}"
    bindings: dict[str, str] = {}
    for name in engine.resources:
        if not name.endswith(suffix):
            continue
        base = name.split("@", 1)[0]
        if base == "matrix":
            bindings["matrix"] = name
            bindings.setdefault("cube", name)
        elif base in {"mte1", "mte3", "fixpipe", "vector", "host"}:
            bindings[base] = name
        elif base == "mte2_aic":
            bindings.setdefault("mte2", name)
        elif base == "mte2_aiv":
            bindings["mte2"] = name
        elif base == "memory":
            bindings["hbm"] = name
        elif base == "hbm":
            bindings["hbm"] = name
    if "link" not in bindings and "link:d2d" in engine.resources:
        bindings["link"] = "link:d2d"
    return bindings


def _declared_stage_effects(compiled: Any, cost: Any, geometry: Any,
                            state: Any) -> tuple[Any, ...]:
    """Resolve one evaluated operator's declared typed ``StateEffect``s.

    The single source of truth is the family declaration that
    ``submit_serial_result``/``submit_cost`` already consume: the operator's
    bound family program.  No target/action string is inferred here from an
    operator name, model label or path.  A cost whose operator cannot be bound
    to a family declaration simply owns no declared effect, exactly as the
    serial path leaves it to the fragment/conductor.
    """
    if compiled is None:
        return ()
    operator = None
    try:
        operator = compiled.operator(cost.name)
    except (ModelingError, AttributeError, TypeError):
        operator = None
    if operator is None or geometry is None:
        return ()
    from .families_bridge import family_work
    try:
        family = family_work(operator, geometry, state)
    except ModelingError:
        return ()
    if family is None:
        return ()
    declared: list[Any] = []
    for stage_index, _stage in enumerate(family.stages):
        for effect in family.stages[stage_index].state_effects:
            if effect not in declared:
                declared.append(effect)
    return tuple(declared)


def submit_evaluation_catalog(
        evaluation: SubgraphEvaluation, execution: Any, *, name: str,
        ready: Any = None, replica: int = 0, stage: int = 0,
        supply: PipeSupply | None = None, compiled: Any | None = None,
        geometry: Any | None = None, state: Any | None = None,
        states: Sequence[Any] | None = None) -> tuple[SubmittedOperator, ...]:
    """Submit one evaluated subgraph as one immutable ProgramCatalog.

    Every evaluated operator stage becomes an explicit StageSpec carrying its
    already-computed physical work.  The single Submission is created on the
    caller's Execution; the returned SubmittedOperator rows alias its actual
    terminal event and preserve their own work/entering state.  No isolated
    per-operator Engine and no duration-to-work conversion is performed.

    ``compiled``+``geometry`` bind the same declared typed ``StateEffect``s the
    serial/fragment route uses onto each operator's terminal stage and the
    catalog state schema, so one catalog submission commits one real state
    write instead of dropping it.
    """
    costs = tuple(evaluation.operators)
    if not costs:
        raise ModelingError("submit_evaluation_catalog_requires_costs")
    conversion = 1.0
    if supply is not None:
        try:
            peak = float(supply.effective("cube"))
        except (AttributeError, TypeError, ValueError):
            peak = 0.0
        if peak > 0:
            conversion = 1.0 / peak
    specs = []
    local = 0
    cost_locals: list[tuple[int, ...]] = []
    for cost in costs:
        row_ids: list[int] = []
        stages = tuple(cost.stage_work) or (tuple(cost.work),)
        for work in stages:
            entries = work.items() if hasattr(work, "items") else work
            terms = []
            for pipe, amount in entries:
                value = float(as_fraction(amount))
                if value <= 0:
                    continue
                rate = conversion if str(pipe) in {"cube", "matrix"} else 1.0
                terms.append(WorkTerm(
                    physical_resource_role=str(pipe), issued=value,
                    service_conversion=rate))
            if not terms:
                continue
            after = (EdgeSpec(source_node=local - 1),) if local else ()
            specs.append(StageSpec(
                local_id=local, label=f"{name}:{cost.name}:s{local}",
                class_id=0, relative_after=after, work=tuple(terms)))
            row_ids.append(local)
            local += 1
        cost_locals.append(tuple(row_ids))
    if not specs:
        raise ModelingError("submit_evaluation_catalog_no_positive_work")
    schema: list[Any] = []
    for index, cost in enumerate(costs):
        row_ids = cost_locals[index]
        entering = (states[index] if states is not None and index < len(states)
                    else state)
        declared = _declared_stage_effects(compiled, cost, geometry, entering)
        if not row_ids or not declared:
            continue
        terminal = row_ids[-1]
        existing = tuple(getattr(specs[terminal], "state_effects", ()) or ())
        merged = tuple(existing) + tuple(
            effect for effect in declared if effect not in existing)
        specs[terminal] = replace(specs[terminal], state_effects=merged)
        for effect in merged:
            if effect not in schema:
                schema.append(effect)
    output = f"{name}:done"
    specs[-1] = replace(
        specs[-1], output_ports=(output,))
    catalog = ProgramCatalog(
        catalog_id=f"subgraph:{name}:{local}",
        implementation_id=str(getattr(evaluation, "kind", "") or "subgraph"),
        generation=CatalogGeneration(
            implementation_generation=str(getattr(evaluation, "kind", "")),
            hardware_fingerprint=str(getattr(supply, "arch_family", ""))),
        stages=tuple(specs), ports=(PortSpec(
            name=output, kind="event", direction="out", access="signal",
            producer_node=local - 1),),
        state_schema=tuple(schema),
        selected_call_count=len(costs), semantic_sources=tuple(
            cost.name for cost in costs))
    engine = getattr(execution, "engine", execution)
    submitter = getattr(execution, "submit_catalog", None) or getattr(
        engine, "submit_catalog", None)
    if submitter is None:
        raise ModelingError("submit_evaluation_catalog_requires_submit_catalog")
    occurrence = engine.new_occurrence(
        replica=int(replica), rank=0, stage=int(stage), ordinal=0)
    raw = tuple(ready) if isinstance(ready, (tuple, list)) else (ready,)
    upstream = tuple(item.event if hasattr(item, "event") else item
                     for item in raw if item is not None)
    if not upstream:
        upstream = (engine.submit(f"{name}:ready"),)
    frame = BindingFrame(
        occurrence=occurrence,
        geometry={"tokens": int(getattr(
            getattr(evaluation, "geometry", None), "tokens", 0) or 0)},
        resources=_catalog_resource_bindings(execution, replica, stage),
        values={})
    submission = submitter(catalog, frame, name=name, after=upstream)
    def actual_events(ids: tuple[int, ...]):
        events = []
        for local_id in ids:
            try:
                events.append(engine.event_for(occurrence, local_id))
            except Exception:  # node not materialized means no event to alias
                continue
        return tuple(events) or (submission.done,)

    rows = tuple(SubmittedOperator(
        cost.name, actual_events(cost_locals[index]),
        native=bool(cost.native_program),
        work=tuple((str(pipe), float(as_fraction(amount)))
                   for pipe, amount in cost.work
                   if as_fraction(amount) > 0),
        entering_state=None) for index, cost in enumerate(costs))
    return rows


def submit_serial_result(result: SerialResult, execution: Any, *,
                         replica: int = 0, stage: int = 0,
                         supply: PipeSupply | None = None,
                         ready: Any = None, compiled: Any | None = None,
                         geometry: Any | None = None,
                         state: Any | None = None,
                         readiness: Mapping[str, Any] | None = None,
                         memory: Any | None = None,
                         native_kwargs: Mapping[str, Any] | None = None,
                         states: Sequence[Any] | None = None,
                         conditions: Mapping[str, bool] | None = None,
                         holds: Mapping[str, Any] | None = None,
                         event_bindings: Mapping[str, Any] | None = None
                         ) -> tuple[SubmittedOperator, ...]:
    """Submit one serial ``SerialResult`` in its actual cost order.

    Readiness is a real dependency, not a report delay: a named cost gets a
    wait event at its absolute ready time before submission.  When a compiled
    subgraph is supplied, its bound operator is used to preserve typed
    ports/state effects.
    """
    current = ready
    readiness = dict(readiness or {})
    submitted: list[SubmittedOperator] = []
    for index, cost in enumerate(result.costs):
        operator = None
        if compiled is not None:
            try:
                operator = compiled.operator(cost.name)
            except ModelingError:
                operator = None
        if cost.name in readiness:
            # Absolute readiness is a separate edge; join it with the current
            # predecessor instead of adding the wait after that predecessor.
            wait = max(0.0, float(readiness[cost.name])
                       - float(execution.engine.now))
            wait_event = execution.engine.submit(
                f"{cost.name}#{index}:ready_at", latency=wait, after=())
            if current is not None:
                current = execution.engine.submit(
                    f"{cost.name}#{index}:ready_join",
                    after=(current, wait_event))
            else:
                current = wait_event
        item = submit_cost(
            cost, execution, name=f"{cost.name}#{index}",
            ready=current, replica=replica, stage=stage, supply=supply,
            operator=operator, geometry=geometry,
            state=(states[index] if states is not None and index < len(states)
                   else state),
            memory=memory)
        submitted.append(item)
        if item.events:
            current = item.events[-1]
    return tuple(submitted)


def _declared_effects(operator: CompiledOperator) -> tuple[tuple[str, str], ...]:
    raw = operator.operator.theta.find("state_effects", None)
    if not raw:
        return ()
    if isinstance(raw, (tuple, list)):
        return tuple((str(item[0]), str(item[1])) for item in raw
                     if isinstance(item, (tuple, list)) and len(item) == 2)
    result = []
    for item in str(raw).split(","):
        parts = item.strip().split(":", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            result.append((parts[0], parts[1]))
    return tuple(result)

def _check_requirements(state: Any, operator: CompiledOperator,
                        geometry: WorkloadGeometry | None = None) -> None:
    theta = operator.operator.theta
    kv_owner = theta.find("require_kv", None)
    if kv_owner is not None:
        state.require_kv(int(kv_owner))
    index_owner = theta.find("require_index", None)
    if index_owner is not None:
        state.require_index(int(index_owner))
    selection_owner = theta.find("require_selection", None)
    if selection_owner is not None:
        state.require_selection(int(selection_owner))
    ratio = theta.find("require_compression", None)
    if ratio is not None and geometry is not None and int(ratio) > 0:
        owner = int(theta.find("compression_owner", operator.layer) or 0)
        # Only a committed span that crossed a ratio boundary must have a
        # published row.  A rejected speculative tail can leave consumed below
        # the next boundary without invalidating earlier cache rows.
        record = state.compression_state(owner, int(ratio))
        if record.consumed >= int(ratio) and record.rows < 1:
            raise ModelingError(
                f"state_port_not_ready:compressed_kv:layer={owner}:rows=0")

def _apply_declared_effects(state: Any, operator: CompiledOperator, instance: SubgraphInstance) -> Any:
    for effect, value in _declared_effects(operator):
        if effect == "index" and value == "index_store":
            owner = operator.operator.theta.find("owner_layer", instance.layer)
            state = state.mark_index(int(owner) if int(owner) >= 0 else instance.layer)
        elif effect == "index" and value == "index_read":
            state = state.require_index(instance.layer)
            state = state
        elif effect == "kv" and value == "kv_store":
            state = state.mark_kv(instance.layer)
        elif effect == "kv" and value == "kv_read":
            state.require_kv(instance.layer)
        elif effect == "selection" and value in {"index_select", "select"}:
            owner = operator.operator.theta.find("owner_layer", instance.layer)
            state = state.mark_selection(int(owner) if int(owner) >= 0 else instance.layer)
        elif effect == "cross_stage" and value == "transfer":
            owner = int(operator.operator.theta.find("owner_layer", instance.layer) or 0)
            for stage in operator.operator.theta.find("destination_stages", ()) or ():
                state = state.mark_cross_stage(owner, int(stage))
    return state

def advance_state(compiled: CompiledSubgraph, state: Any, geometry: WorkloadGeometry) -> Any:
    """Apply one subgraph's declared write effects without re-pricing it."""
    current = state if state is not None else _empty_state()
    for operator in compiled.operators:
        _check_requirements(current, operator, geometry)
        current = _apply_operator_state(current, operator, None, compiled.instance, geometry)
        current = _apply_declared_effects(current, operator, compiled.instance)
    return current

def evaluate_serial_split( compiled: CompiledSubgraph, geometry: WorkloadGeometry, state: Any = None, *,
    geometry_for: Callable[[CompiledOperator], WorkloadGeometry] | None = None, fusion: Sequence[str] = (),
    readiness: Mapping[str, Fraction | float] | None = None, skip: Sequence[str] = (), ) -> SerialResult:
    """Evaluate one subgraph with per-operator geometry and ordered state."""
    current = state if state is not None else _empty_state()
    costs: list[OperatorCost] = []
    states_by_name: dict[str, Any] = {}
    skipped = frozenset(skip)
    for operator in compiled.operators:
        if operator.name in skipped:
            continue
        local = geometry_for(operator) if geometry_for is not None else geometry
        _check_requirements(current, operator, local)
        states_by_name[operator.name] = current
        cost = operator.evaluate(local, current)
        costs.append(cost)
        current = _apply_operator_state(current, operator, cost, compiled.instance, local)
        current = _apply_declared_effects(current, operator, compiled.instance)
    unfused_costs = tuple(costs)
    fused_names: list[str] = []
    fusion_results: list[FusionResult] = []
    waits = {key: as_fraction(value) for key, value in (readiness or {}).items()}
    fused_ready: dict[str, Fraction] = {}
    for name in fusion:
        result = lower_fusion(compiled.instance, costs, name, supply=compiled.supply)
        fusion_results.append(result)
        members = {cost.name for cost in result.semantic}
        member_ready = max((waits.get(cost.name, Fraction(0)) for cost in result.semantic), default=Fraction(0))
        fused_ready[result.fused.name] = max(member_ready, waits.get(result.fused.name, Fraction(0)))
        replaced: list[OperatorCost] = []
        for cost in costs:
            if cost.name == result.semantic[0].name:
                replaced.append(result.fused)
            elif cost.name not in members:
                replaced.append(cost)
        costs = replaced
        fused_names.append(name)
    gaps, completion, envelope = [], Fraction(0), None
    streams = {op.name: op.operator.stream for op in compiled.operators}
    if not waits and "aux" in streams.values():
        fork_ops = {"qkvz_projection", "qkv_projection", "ba_projection", "gate_projection", "conv", "prepare",
                    "window_attention", "compressed_attention"}
        # A fused kernel replaces its members in place and keeps their stream.
        # Dropping it onto the serial tail would undo the overlap and could
        # make the fused schedule slower than the unfused one.
        fork_names = set(fork_ops)
        for result in fusion_results:
            member_streams = {streams.get(cost.name, "main") for cost in result.semantic}
            streams[result.fused.name] = next(iter(member_streams)) if len(member_streams) == 1 else "main"
            if any(cost.name in fork_ops for cost in result.semantic):
                fork_names.add(result.fused.name)
        chains: dict[str, list[OperatorSegment]] = {"main": [], "aux": []}
        for cost in costs:
            if cost.name in fork_names:
                branch = "aux" if streams.get(cost.name) == "aux" else "main"
                # Each kernel keeps its own floor, startup and dependency.
                # Summing those durations into one t=0 floor lets a later
                # floor overlap the previous kernel's body.
                chains[branch].append(operator_segment(cost, compiled.supply))
            else:
                completion += cost.duration_ms
            gaps.append(Fraction(0))
        live = {name: segments for name, segments in chains.items() if segments}
        envelope = contention_envelope({}, compiled.supply, chains=live)
        completion += envelope.makespan_ms
    else:
        for cost in costs:
            ready = max(waits.get(cost.name, Fraction(0)), fused_ready.get(cost.name, Fraction(0)))
            gaps.append(max(Fraction(0), ready - completion))
            completion = max(completion, ready) + cost.duration_ms
    effective = {**waits, **fused_ready}
    fused_state = {
        result.fused.name: states_by_name.get(result.semantic[0].name, current)
        for result in fusion_results
    }
    states = tuple(
        fused_state.get(cost.name, states_by_name.get(cost.name, current))
        for cost in costs)
    return SerialResult(tuple(costs), completion, current, tuple(gaps),
                        tuple(fused_names), tuple(sorted(effective.items())),
                        tuple(fusion_results), envelope, unfused_costs,
                        states)

def evaluate_subgraphs_padded( compiled: Iterable[CompiledSubgraph], active: WorkloadGeometry, padded: WorkloadGeometry | None = None, *,
    active_kinds: Iterable[str] = tuple(sorted(
        ALL_ATTENTION_KINDS
        | {"kv_cache", "kv_compressor", "indexer_prolog", "lightning_indexer",
           "moe_router", "moe_token_dispatcher", "routed_experts", "spec_decode"})),
    fusion: bool = True, state: Any = None, ) -> tuple[SerialResult, ...]:
    """Sequential subgraph composition with capture-padding separation.

    Attention/MoE/transport families keep their real active rows by default;
    matrix/elementwise/copy families receive the padded bucket geometry when
    the caller supplied one.  Attention family operators inside A0/A1 are also
    active even though the containing macro exists in the padded set.
    """
    active_set = frozenset(active_kinds)
    current = state if state is not None else _empty_state()
    results = []
    for compiled in compiled:
        kind = compiled.instance.kind

        def geometry_for(operator, kind=kind):
            if padded is None:
                return active
            if kind in active_set:
                return active
            if (kind in (STANDARD_ATTENTION_KINDS | LATENT_ATTENTION_KINDS)
                    and operator.family in {"attention", "mla"}):
                return active
            return padded

        names = tuple(item.name for item in compiled.instance.template.fusions) if fusion else ()
        result = evaluate_serial_split(compiled, active, current, geometry_for=geometry_for, fusion=names)
        current = result.state
        results.append(result)
    return tuple(results)

class SubgraphCostFunction:
    """Analytical facade over one compiled subgraph instance."""
    def __init__(self, compiled: CompiledSubgraph):
        if not isinstance(compiled, CompiledSubgraph):
            raise ModelingError("cost_function_requires_compiled_subgraph")
        self.compiled = compiled
        self.instance = compiled.instance
        self.supply = compiled.supply
        self.policy = compiled.policy
    def evaluate_serial(self, geometry: WorkloadGeometry, state: Any = None, *, fusion: Sequence[str] = (),
                        readiness: Mapping[str, Fraction | float] | None = None ) -> SerialResult:
        state = state or _empty_state()
        base = self.compiled.evaluate(geometry, state)
        costs = list(base.operators)
        fused_names: list[str] = []
        fusion_results: list[FusionResult] = []
        readiness = {key: as_fraction(value) for key, value in (readiness or {}).items()}
        fused_ready: dict[str, Fraction] = {}
        for name in fusion:
            result = lower_fusion(self.instance, costs, name, supply=self.supply)
            fusion_results.append(result)
            members = {cost.name for cost in result.semantic}
            member_ready = max((readiness.get(cost.name, Fraction(0)) for cost in result.semantic), default=Fraction(0))
            fused_ready[result.fused.name] = max( member_ready, readiness.get(result.fused.name, Fraction(0)))
            replaced: list[OperatorCost] = []
            for cost in costs:
                if cost.name == result.semantic[0].name:
                    replaced.append(result.fused)
                elif cost.name not in members:
                    replaced.append(cost)
            costs = replaced
            fused_names.append(name)
        gaps: list[Fraction] = []
        completion = Fraction(0)
        ordered = []
        for cost in costs:
            ready = max(readiness.get(cost.name, Fraction(0)), fused_ready.get(cost.name, Fraction(0)))
            gap = max(Fraction(0), ready - completion)
            start = max(completion, ready)
            completion = start + cost.duration_ms
            gaps.append(gap)
            ordered.append(cost)
        effective = dict(readiness)
        effective.update(fused_ready)
        return SerialResult(tuple(ordered), completion, base.state_out, tuple(gaps), tuple(fused_names),
                            tuple(sorted(effective.items())), tuple(fusion_results))
    def evaluate_isolated(self, geometry: WorkloadGeometry, state: Any = None) -> SerialResult:
        return self.evaluate_serial(geometry, state)
    def evaluate_fork_join(self, geometry: WorkloadGeometry, branches: Mapping[str, Sequence[OperatorCost]],
                           *, prefix_ms: Fraction | float = 0, join_ms: Fraction | float = 0, state: Any = None) -> ForkJoinResult:
        state = state or _empty_state()
        costs = {name: tuple(values) for name, values in branches.items()}
        stage_work = {}
        isolated = {}
        for name, operators in costs.items():
            stages: list[list[tuple[str, Fraction]]] = []
            for cost in operators:
                if cost.stage_work:
                    stages.extend([[(pipe, value) for pipe, value in stage] for stage in cost.stage_work])
                else:
                    stages.append([(pipe, value) for pipe, value in cost.work])
            stage_work[name] = stages
            isolated[name] = sum((cost.duration_ms for cost in operators), Fraction(0))
        engine_result = run_engine_work(stage_work, self.supply)
        prefix, join = as_fraction(prefix_ms), as_fraction(join_ms)
        return ForkJoinResult( prefix, join, max(isolated.values()) if isolated else Fraction(0),
            engine_result.makespan_ms, engine_result.makespan_ms, engine_result.branches, engine_result.resources)
    def evaluate_fork_join_stages( self, branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]],
        *, prefix_ms: Fraction | float = 0, join_ms: Fraction | float = 0, ) -> ForkJoinResult:
        engine_result = run_engine_work(branches, self.supply)
        isolated = {}
        for name, stages in branches.items():
            isolated[name] = sum( (max((value / self.supply.effective(pipe) for pipe, value in stage),
                     default=Fraction(0)) for stage in stages), Fraction(0))
        return ForkJoinResult( as_fraction(prefix_ms), as_fraction(join_ms),
            max(isolated.values()) if isolated else Fraction(0), engine_result.makespan_ms, engine_result.makespan_ms,
            engine_result.branches, engine_result.resources)
    def explain(self, geometry: WorkloadGeometry, state: Any = None):
        from .diagnostics import explain
        return explain(self, geometry, state)
    def closed_form(self) -> ClosedForm:
        stages = [stage for operator in self.compiled.operators for stage in operator.program.stages]
        if any(len(operator.program.stages) > 1 for operator in self.compiled.operators):
            return ClosedForm("", False, "multi_stage_operator_requires_engine")
        if any(operator.axes for operator in self.compiled.operators):
            return ClosedForm( "sum_m max_p W_{m,p}(X,Z)/R_p ; guards lazy", True, "piecewise_guarded_serial_chain")
        return ClosedForm("sum_m max_p W_{m,p}/R_p", True, "single_stage_serial_chain")
    def to_iteration_work(self, geometry: WorkloadGeometry, state: Any = None) -> tuple[OpSpec, ...]:
        """Lower the same evaluated work to the existing iteration contract."""
        state = state or _empty_state()
        evaluation = self.compiled.evaluate(geometry, state)
        records = []
        for cost in evaluation.operators:
            mte2 = cost.pipe_work("mte2")
            mte3 = cost.pipe_work("mte3")
            vector = cost.pipe_work("vector")
            flops = cost.get("flops") or cost.get("logical_flops") or cost.pipe_work("cube")
            hbm = cost.pipe_work("hbm") or (mte2 + mte3)
            records.append(OpSpec( op_id=f"{self.instance.instance_id}:{cost.name}",
                family=cost.family, phase=geometry.phase_name, stage=self.instance.stage, layer=self.instance.layer,
                requests=tuple(f"r{index}" for index in range(geometry.batch)), semantic_shape=(geometry.effective_tokens,),
                local_shape=(geometry.tokens,), padded_shape=(geometry.effective_tokens,),
                launch_shape=(geometry.effective_tokens,), flops=float(flops), hbm_bytes=float(hbm),
                mte2_bytes=float(mte2), mte3_bytes=float(mte3),
                vector_ops=float(vector), recipe=cost.formula[:200] or cost.family, dtype="bf16", layout="contiguous",
                metadata=(("piecewise_segment", cost.segment_index), ("piecewise_regime", cost.regime),
                          ("dominant_pipe", cost.dominant_pipe)), ))
        return tuple(records)
    def to_engine_work(self, geometry: WorkloadGeometry, state: Any = None) -> dict[str, list[list[tuple[str, Fraction]]]]:
        state = state or _empty_state()
        evaluation = self.compiled.evaluate(geometry, state)
        stages: list[list[tuple[str, Fraction]]] = []
        for cost in evaluation.operators:
            if cost.stage_work:
                stages.extend([[ (pipe, value) for pipe, value in stage] for stage in cost.stage_work])
            elif cost.work:
                stages.append([(pipe, value) for pipe, value in cost.work])
        return {self.instance.instance_id: stages}

def p2p_transfer_cost(layer: int, tokens: int, width: int, act_bytes: float, supply: PipeSupply) -> OperatorCost:
    """Hidden-state point-to-point at an actual pipeline stage cut.

    This is a directed copy, not a collective: no ring expansion is applied
    and the operation is not classified as AllReduce.
    """
    payload = Fraction(tokens * width * float(act_bytes))
    stage_work = ((("link", payload), ("mte2", payload)), (("hbm", 2 * payload), ("mte3", payload)))
    rates = {pipe: Fraction(str(supply.effective(pipe))) for pipe in ("link", "mte2", "mte3", "hbm")}
    times = {"link": payload / rates["link"], "mte2": payload / rates["mte2"], "mte3": payload / rates["mte3"],
             "hbm": 2 * payload / rates["hbm"]}
    stage_times = tuple(max(times[pipe] for pipe, _ in stage) for stage in stage_work)
    work: dict[str, Fraction] = {}
    for stage in stage_work:
        for pipe, value in stage:
            work[pipe] = work.get(pipe, Fraction(0)) + value
    duration = sum(stage_times, Fraction(0))
    peak = max(times.values())
    return OperatorCost( name="stage_transfer", family="p2p", kind="PP", segment_index=-1,
        regime=(), guard_cells=(), work=tuple(sorted(work.items())),
        times=tuple(sorted(times.items())), stage_times=stage_times, duration_ms=duration, stage_work=stage_work,
        dominant_pipe=max(times, key=times.get), tied_pipes=tuple(pipe for pipe, value in times.items() if value == peak),
        materialization=(), internal=(), residency_bytes=Fraction(0),
        derived=(("payload_bytes", payload), ("p2p", Fraction(1)), ("destination_stage_delta", Fraction(1))),
        formula="pipeline_send;p2p", fidelity="isolated_device_time")

def compose_sequence(subgraphs: Iterable[CompiledSubgraph], geometry: WorkloadGeometry, state: Any = None,
                     *, readiness: Mapping[str, Fraction | float] | None = None,
                     execution: Any | None = None, replica: int = 0,
                     stage: int = 0, ready: Any = None,
                     memory: Any | None = None) -> SerialResult:
    """Compose subgraphs in model order, carrying one explicit state view.

    With ``execution`` the actual evaluated costs are submitted to that
    caller-owned context after state progression; durations remain the
    independent observation, not a substitute for submission.
    """
    state, costs, gaps, completion = state or _empty_state(), [], [], Fraction(0)
    readiness = {key: as_fraction(value) for key, value in (readiness or {}).items()}
    subgraphs = tuple(subgraphs)
    supply = next((item.supply for item in subgraphs), None)
    submitted_all: list[SubmittedOperator] = []
    terminal = ready
    for subgraph in subgraphs:
        evaluation = subgraph.evaluate(geometry, state)
        state = evaluation.state_out
        for cost in evaluation.operators:
            ready_at = readiness.get(cost.name, Fraction(0))
            gaps.append(max(Fraction(0), ready_at - completion))
            completion = max(completion, ready_at) + cost.duration_ms
            costs.append(cost)
        if execution is not None:
            submitted_all.extend(submit_serial_result(
                SerialResult(evaluation.operators, evaluation.total_ms,
                             evaluation.state_out),
                execution, replica=replica, stage=stage,
                supply=supply, ready=terminal, compiled=subgraph,
                geometry=geometry, state=evaluation.state_out,
                readiness=readiness, memory=memory,
                states=evaluation.states))
            if submitted_all and submitted_all[-1].events:
                terminal = submitted_all[-1].events[-1]
    result = SerialResult(tuple(costs), completion, state, tuple(gaps),
                          submission=tuple(submitted_all))
    return result
def _empty_state():
    from mindie_sim.prediction.subgraphs.state import StateView
    return StateView()
