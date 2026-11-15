"""Unit-bearing work, semantic effects and iteration records.

[Role]
  Define immutable shape lineage, work records, semantic/state/field effects
  and the common program-fragment contract used by workload lowering and
  downstream model/framework/native adapters.

[Invariants]
  * Every op keeps semantic, local, padded and launch shapes distinct.
  * GEMM work is 2*M*N*K; vector work is elements*ops_per_element.
  * No output token is added by a work constructor.
  * Every WorkItem carries a unit; useful work is never charged as issued
    service.  Ports, semantic references, state effects and allocations are
    explicit immutable records, not inferred defaults.

[Timeouts & Locks]
  Pure arithmetic.  No I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from .common import ModelingError
from .invocation import Occurrence
from .subgraphs import PortSet
from .workload import IterationPlan


#: What a semantic reference points at.  References explain structure and
#: matching; they never decide whether an occurrence executes.
SEMANTIC_REF_KINDS = frozenset({
    "model", "subgraph", "operator", "layer", "request", "batch",
    "implementation", "runtime",
})

#: State actions keep produce/consume/commit/rollback distinct.
STATE_EFFECT_ACTIONS = frozenset({"read", "write", "commit", "rollback"})

#: Every transformation states what happens to a field it consumes.
FIELD_ACTIONS = frozenset({"preserve", "rewrite", "consume"})

#: Useful work is what the user asked for; issued work is what the selected
#: implementation really executes.  Only issued work consumes resource service.
WORK_KINDS = frozenset({"useful", "issued"})


def _frozen_attr(value: Any, reason: str) -> Any:
    """Freeze an attribute value without silently losing mapping entries."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _frozen_attr(item, reason))
                            for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_attr(item, reason) for item in value)
    raise ModelingError(f"{reason}:unsupported_value_type:{type(value).__name__}")


@dataclass(frozen=True)
class SemanticRef:
    """A nullable/multi-valued model/subgraph/operator relationship."""

    kind: str
    name: str
    role: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SEMANTIC_REF_KINDS:
            raise ModelingError(f"semantic_ref_kind_unknown:{self.kind}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError(f"semantic_ref_name_invalid:{self.kind}")
        if not isinstance(self.role, str):
            raise ModelingError(f"semantic_ref_role_invalid:{self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "role": self.role}

    @property
    def key(self) -> tuple[str, str, str]:
        return self.kind, self.name, self.role


@dataclass(frozen=True)
class SemanticAttr:
    """One named semantic attribute with its unit and provenance."""

    name: str
    value: Any
    unit: str = ""
    source: str = "declared"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("semantic_attr_name_invalid")
        if not isinstance(self.unit, str) or not isinstance(self.source, str):
            raise ModelingError(f"semantic_attr_metadata_invalid:{self.name}")
        object.__setattr__(self, "value", _frozen_attr(
            self.value, f"semantic_attr_value_invalid:{self.name}"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit,
                "source": self.source}


@dataclass(frozen=True)
class StateEffect:
    """A declared state read/write/commit/rollback, not a device operation."""

    target: str
    action: str
    version: int | None = None
    condition: str = ""
    arbiter: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ModelingError(f"state_effect_target_invalid:{self.target}")
        if self.action not in STATE_EFFECT_ACTIONS:
            raise ModelingError(f"state_effect_action_unknown:{self.action}")
        if self.version is not None and (
                not isinstance(self.version, int) or isinstance(self.version, bool)
                or self.version < 0):
            raise ModelingError(f"state_effect_version_invalid:{self.target}")
        if self.action in {"commit", "rollback"} and self.version is None:
            raise ModelingError(f"state_effect_commit_version_required:{self.target}")
        if not isinstance(self.condition, str) or not isinstance(self.arbiter, str):
            raise ModelingError(f"state_effect_policy_invalid:{self.target}")

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "action": self.action,
                "version": self.version, "condition": self.condition,
                "arbiter": self.arbiter}


@dataclass(frozen=True)
class FieldDisposition:
    """How a transform handles one required input field.

    ``preserve`` keeps the same value/attribute; ``rewrite`` produces a new
    value with a conversion/materialisation; ``consume`` folds the field into
    the selected algorithm, work expression or constraint.  Rewrite/consume
    keep a locatable source relation.
    """

    field: str
    action: str
    source: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ModelingError("field_disposition_field_invalid")
        if self.action not in FIELD_ACTIONS:
            raise ModelingError(f"field_disposition_action_unknown:{self.action}")
        if not isinstance(self.source, str) or not isinstance(self.detail, str):
            raise ModelingError(f"field_disposition_metadata_invalid:{self.field}")
        if self.action in {"rewrite", "consume"} and not self.source:
            raise ModelingError(
                f"field_disposition_source_required:{self.action}:{self.field}")

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "action": self.action,
                "source": self.source, "detail": self.detail}


@dataclass(frozen=True)
class WorkItem:
    """One resource demand with an explicit unit and work class."""

    resource: str
    amount: float
    unit: str
    kind: str = "issued"
    source: str = "derived"

    def __post_init__(self) -> None:
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ModelingError("work_item_resource_invalid")
        if isinstance(self.amount, bool):
            raise ModelingError(f"work_item_amount_invalid:{self.resource}")
        try:
            amount = float(self.amount)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelingError(f"work_item_amount_invalid:{self.resource}") from exc
        if not math.isfinite(amount) or amount < 0:
            raise ModelingError(f"work_item_amount_invalid:{self.resource}")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ModelingError(f"work_item_unit_invalid:{self.resource}")
        if self.kind not in WORK_KINDS:
            raise ModelingError(f"work_item_kind_unknown:{self.kind}")
        if not isinstance(self.source, str):
            raise ModelingError(f"work_item_source_invalid:{self.resource}")
        object.__setattr__(self, "amount", amount)

    def to_dict(self) -> dict[str, Any]:
        return {"resource": self.resource, "amount": self.amount,
                "unit": self.unit, "kind": self.kind, "source": self.source}


@dataclass(frozen=True)
class WorkSet:
    """A finite set of useful and issued work.

    At most one item of each ``(resource, kind)`` pair is allowed.  A resource
    keeps one native unit across useful/issued declarations.  ``service_work``
    returns only issued work, so declaring desirable work never silently adds
    device service.
    """

    items: tuple[WorkItem, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.items)
        seen: set[tuple[str, str]] = set()
        units: dict[str, str] = {}
        for item in items:
            if not isinstance(item, WorkItem):
                raise ModelingError("work_set_item_invalid")
            key = item.resource, item.kind
            if key in seen:
                raise ModelingError(f"work_set_duplicate:{item.resource}:{item.kind}")
            seen.add(key)
            known = units.setdefault(item.resource, item.unit)
            if known != item.unit:
                raise ModelingError(f"work_set_unit_conflict:{item.resource}")
        object.__setattr__(self, "items", items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def by_resource(self, kind: str = "issued") -> dict[str, float]:
        return {item.resource: item.amount for item in self.items
                if item.kind == kind}

    def issued(self) -> tuple[WorkItem, ...]:
        return tuple(item for item in self.items if item.kind == "issued")

    def useful(self) -> tuple[WorkItem, ...]:
        return tuple(item for item in self.items if item.kind == "useful")

    def service_work(self) -> dict[str, float]:
        """Resource demand in Engine work units; useful work is not charged."""
        return self.by_resource("issued")

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items],
                "service_work": self.service_work(),
                "useful_work": self.by_resource("useful")}


@dataclass(frozen=True)
class ResourcePolicy:
    """Declared sharing/priority policy for one physical resource.

    Maps to the existing ``engine.Resource(policy=...)``: ``fair`` is
    work-conserving max-min sharing, ``fifo`` admits one task at a time in
    deterministic submission order.  ``priority`` is admission order only;
    the existing Engine never preempts a running task, so ``preemptive=True``
    is rejected rather than silently downgraded.
    """

    sharing: str = "fair"
    priority: int = 0
    preemptive: bool = False

    def __post_init__(self) -> None:
        if self.sharing not in {"fair", "fifo"}:
            raise ModelingError(f"resource_sharing_unknown:{self.sharing}")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) \
                or self.priority < 0:
            raise ModelingError("resource_priority_invalid")
        if not isinstance(self.preemptive, bool):
            raise ModelingError("resource_preemptive_invalid")
        if self.preemptive:
            raise ModelingError("resource_preemption_unsupported")

    def engine_kwargs(self) -> dict[str, object]:
        return {"policy": self.sharing}

    def to_dict(self) -> dict[str, object]:
        return {"sharing": self.sharing, "priority": self.priority,
                "preemptive": self.preemptive}


@dataclass(frozen=True)
class AllocationEffect:
    """One declared acquire/release of a pooled buffer or non-preemptive slot.

    A slot is a pool with ``size=1`` and unit capacity; a stage that acquires
    it keeps it until that stage completes (including declared latency), so
    waiting does not invent extra work.  ``lifetime`` names the boundary that
    owns the release.  ``prediction.program.stage_from_fragment`` converts the
    record to the existing Engine ``Allocation``.
    """

    action: str
    name: str
    pool: str
    size: float
    units: tuple[int, ...] = (0,)
    unit: str = "bytes"
    lifetime: str = "stage"
    condition: str = ""

    def __post_init__(self) -> None:
        if self.action not in {"acquire", "release"}:
            raise ModelingError(f"allocation_effect_action_unknown:{self.action}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("allocation_effect_name_invalid")
        if not isinstance(self.pool, str) or not self.pool.strip():
            raise ModelingError(f"allocation_effect_pool_invalid:{self.name}")
        if isinstance(self.size, bool):
            raise ModelingError(f"allocation_effect_size_invalid:{self.name}")
        try:
            size = float(self.size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelingError(f"allocation_effect_size_invalid:{self.name}") from exc
        if not math.isfinite(size) or size < 0:
            raise ModelingError(f"allocation_effect_size_invalid:{self.name}")
        units = tuple(self.units)
        if (not units or len(set(units)) != len(units)
                or any(not isinstance(unit, int) or isinstance(unit, bool)
                       or unit < 0 for unit in units)):
            raise ModelingError(f"allocation_effect_units_invalid:{self.name}")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ModelingError(f"allocation_effect_unit_invalid:{self.name}")
        if self.lifetime not in {"stage", "slot", "program", "event"}:
            raise ModelingError(f"allocation_effect_lifetime_unknown:{self.lifetime}")
        if not isinstance(self.condition, str):
            raise ModelingError(f"allocation_effect_condition_invalid:{self.name}")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "units", units)

    @classmethod
    def acquire(cls, name: str, pool: str, size: float,
                units: tuple[int, ...] = (0,), *, unit: str = "bytes",
                lifetime: str = "stage",
                condition: str = "") -> "AllocationEffect":
        return cls("acquire", name, pool, size, tuple(units), unit, lifetime,
                   condition)

    @classmethod
    def release(cls, name: str, pool: str, size: float,
                units: tuple[int, ...] = (0,), *, unit: str = "bytes",
                lifetime: str = "stage",
                condition: str = "") -> "AllocationEffect":
        return cls("release", name, pool, size, tuple(units), unit, lifetime,
                   condition)

    def to_dict(self) -> dict[str, object]:
        return {"action": self.action, "name": self.name, "pool": self.pool,
                "size": self.size, "units": list(self.units), "unit": self.unit,
                "lifetime": self.lifetime, "condition": self.condition}


@dataclass(frozen=True)
class ProgramFragment:
    """Common declarative fragment consumed by model/framework/native adapters.

    It is a contract, not an executor: typed tensor/state/event ports,
    occurrence identity, nullable semantic references, attrs, unit-bearing
    work, state/field effects and allocation effects.  The existing
    ``prediction.program.stage_from_fragment`` lowers its finite work and
    stage-scoped allocations onto ``Stage``; execution stays on the existing
    ``KernelProgram``/``Engine`` path.
    """

    name: str
    occurrence: Occurrence | None = None
    semantic_refs: tuple[SemanticRef, ...] | None = None
    attrs: tuple[SemanticAttr, ...] = ()
    ports: PortSet = field(default_factory=PortSet)
    work: WorkSet = field(default_factory=WorkSet)
    policies: tuple[tuple[str, ResourcePolicy], ...] = ()
    state_effects: tuple[StateEffect, ...] = ()
    field_effects: tuple[FieldDisposition, ...] = ()
    allocations: tuple[AllocationEffect, ...] = ()
    #: All issued resources in this fragment progress as one coupled flow.
    coupled: bool = False
    repeats: int = 1
    priority: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("program_fragment_name_invalid")
        if self.occurrence is not None and not isinstance(self.occurrence, Occurrence):
            raise ModelingError("program_fragment_occurrence_invalid")
        refs = () if self.semantic_refs is None else tuple(self.semantic_refs)
        if not all(isinstance(ref, SemanticRef) for ref in refs):
            raise ModelingError("program_fragment_semantic_refs_invalid")
        object.__setattr__(self, "semantic_refs", refs)
        attrs = tuple(self.attrs)
        if not all(isinstance(attr, SemanticAttr) for attr in attrs):
            raise ModelingError("program_fragment_attrs_invalid")
        if len({attr.name for attr in attrs}) != len(attrs):
            raise ModelingError("program_fragment_attr_duplicate")
        object.__setattr__(self, "attrs", attrs)
        ports = self.ports
        if not isinstance(ports, PortSet):
            raise ModelingError("program_fragment_ports_invalid")
        if not isinstance(self.work, WorkSet):
            raise ModelingError("program_fragment_work_invalid")
        policies = self.policies
        if isinstance(policies, Mapping):
            policies = tuple(policies.items())
        policies = tuple((str(resource), policy) for resource, policy in policies)
        seen: set[str] = set()
        for resource, policy in policies:
            if resource in seen or not isinstance(policy, ResourcePolicy):
                raise ModelingError(f"program_fragment_policy_invalid:{resource}")
            if resource not in self.work.by_resource("issued"):
                raise ModelingError(
                    f"program_fragment_policy_resource_unused:{resource}")
            seen.add(resource)
        object.__setattr__(self, "policies", policies)
        effects = tuple(self.state_effects)
        if not all(isinstance(effect, StateEffect) for effect in effects):
            raise ModelingError("program_fragment_state_effects_invalid")
        commits: dict[str, StateEffect] = {}
        for effect in effects:
            if effect.action not in {"commit", "rollback"}:
                continue
            previous = commits.get(effect.target)
            if (previous is not None and previous.version != effect.version
                    and not (previous.arbiter and effect.arbiter)):
                raise ModelingError(
                    f"state_effect_conflict_requires_arbiter:{effect.target}")
            commits[effect.target] = effect
        object.__setattr__(self, "state_effects", effects)
        fields = tuple(self.field_effects)
        if not all(isinstance(effect, FieldDisposition) for effect in fields):
            raise ModelingError("program_fragment_field_effects_invalid")
        if len({effect.field for effect in fields}) != len(fields):
            raise ModelingError("program_fragment_field_effect_duplicate")
        object.__setattr__(self, "field_effects", fields)
        allocations = tuple(self.allocations)
        if not all(isinstance(effect, AllocationEffect) for effect in allocations):
            raise ModelingError("program_fragment_allocations_invalid")
        names = [effect.name for effect in allocations if effect.action == "acquire"]
        releases = [effect.name for effect in allocations if effect.action == "release"]
        if len(set(names)) != len(names) or len(set(releases)) != len(releases):
            raise ModelingError("program_fragment_allocation_duplicate")
        object.__setattr__(self, "allocations", allocations)
        if not isinstance(self.coupled, bool):
            raise ModelingError("program_fragment_coupled_invalid")
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) \
                or self.repeats < 1:
            raise ModelingError("program_fragment_repeats_invalid")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) \
                or self.priority < 0:
            raise ModelingError("program_fragment_priority_invalid")
        notes = tuple(self.notes)
        if not all(isinstance(note, str) for note in notes):
            raise ModelingError("program_fragment_notes_invalid")
        object.__setattr__(self, "notes", notes)

    @property
    def service_work(self) -> dict[str, float]:
        """Issued work in Engine units; useful-only declarations are free."""
        return self.work.service_work()

    @property
    def stage_projection_blockers(self) -> tuple[str, ...]:
        """Semantics a single static ``Stage`` cannot execute or represent.

        ``prediction.program.stage_from_fragment`` must raise when this tuple
        is non-empty.  A caller that needs repeats, conditional/program-scoped
        allocations, state/field effects, resource policy, semantic attrs or
        event-port binding consumes the complete ``ProgramFragment`` (also
        retained on ``Stage.fragment``) instead of a lossy projection.
        ``priority`` is not a blocker: ``Stage.prepare`` forwards it to the
        existing Engine's ready heap.
        """
        blockers: list[str] = []
        if self.repeats != 1:
            blockers.append("repeats")
        if self.attrs:
            blockers.append("attrs")
        if self.policies:
            blockers.append("policies")
        if self.state_effects:
            blockers.append("state_effects")
        if self.field_effects:
            blockers.append("field_effects")
        # KernelProgram.start() has one generic ready event and no port
        # resolver.  Any declared event input names a specific event/readiness
        # generation, so it cannot be replaced by that generic ready.  A single
        # completion output can be represented by the returned Stage event;
        # commit/release/grant/record/wait or multiple completion outputs
        # require the T04 port binder.
        if self.ports.event_in:
            blockers.append("event_in")
        out_events = self.ports.event_out
        if out_events and (len(out_events) != 1
                           or out_events[0].event != "complete"
                           or out_events[0].direction not in {"out", "inout"}):
            blockers.append("event_out")
        for effect in self.allocations:
            if effect.lifetime not in {"stage", "slot"}:
                blockers.append("allocation_lifetime")
            if effect.condition:
                blockers.append("conditional_allocation")
        return tuple(dict.fromkeys(blockers))

    @property
    def supports_stage_projection(self) -> bool:
        """True only when one static Stage can carry every declared semantic."""
        return not self.stage_projection_blockers

    def policy(self, resource: str) -> ResourcePolicy:
        for name, policy in self.policies:
            if name == resource:
                return policy
        return ResourcePolicy()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "occurrence": self.occurrence.to_dict() if self.occurrence else None,
            "semantic_refs": [ref.to_dict() for ref in self.semantic_refs],
            "attrs": [attr.to_dict() for attr in self.attrs],
            "ports": self.ports.to_dict(), "work": self.work.to_dict(),
            "policies": {resource: policy.to_dict()
                         for resource, policy in self.policies},
            "state_effects": [effect.to_dict() for effect in self.state_effects],
            "field_effects": [effect.to_dict() for effect in self.field_effects],
            "allocations": [effect.to_dict() for effect in self.allocations],
            "coupled": self.coupled,
            "repeats": self.repeats, "priority": self.priority,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class OpSpec:
    op_id: str
    family: str
    phase: str
    stage: int
    layer: int
    requests: tuple[str, ...]
    semantic_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    launch_shape: tuple[int, ...]
    flops: float = 0.0
    hbm_bytes: float = 0.0
    mte2_bytes: float = 0.0
    mte3_bytes: float = 0.0
    vector_ops: float = 0.0
    link_bytes: float = 0.0
    link_scope: str = "d2d"
    deps: tuple[str, ...] = ()
    recipe: str = ""
    coverage: str = "decomposed"
    metadata: tuple[tuple[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    dtype: str = ""
    layout: str = "contiguous"
    #: Declared return quantities beyond the primary output (e.g. the
    #: ``softmax_max``/``softmax_sum`` partials an LSE merge consumes).  An op
    #: that omits this declares no mergeable partial; a merge consumer must
    #: name the quantities it reads or it is not a valid composition.
    returns: tuple[str, ...] = ()
    #: Declared device stream for this op (``main``/``aux``), when the lowering
    #: can state it from the runtime (V4/V4.1 attention path).  Empty means the
    #: default main stream is used, including its ordering edges.  Streams
    #: are ordering relations, never capacity-1 resources.
    stream: str = ""
    #: Fused matrix work uses separate precision rates on one physical resource.
    #: Empty retains the single-dtype work declaration.
    matrix_flops: tuple[tuple[str, float], ...] = ()
    #: A source call expands into kernels before pricing and host submission.
    #: Both source boundaries require an exact planner; empty is semantic work.
    source_boundary: Literal["", "kernel", "call"] = ""

    @property
    def plan_required(self) -> bool:
        return bool(self.source_boundary)

    def to_dict(self, *, include_work: bool = True) -> dict[str, Any]:
        payload = {
            "op_id": self.op_id, "family": self.family, "phase": self.phase,
            "stage": self.stage, "layer": self.layer,
            "requests": list(self.requests),
            "semantic_shape": list(self.semantic_shape),
            "local_shape": list(self.local_shape),
            "padded_shape": list(self.padded_shape),
            "launch_shape": list(self.launch_shape),
            "recipe": self.recipe, "deps": list(self.deps),
            "dtype": self.dtype, "layout": self.layout,
            "coverage": self.coverage, "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.returns:
            payload["returns"] = list(self.returns)
        if self.plan_required:
            payload["source_boundary"] = self.source_boundary
        if self.stream:
            payload["stream"] = self.stream
        if include_work:
            payload.update({
                "flops": self.flops, "hbm_bytes": self.hbm_bytes,
                "mte2_bytes": self.mte2_bytes, "mte3_bytes": self.mte3_bytes,
                "vector_ops": self.vector_ops, "link_bytes": self.link_bytes,
                "link_scope": self.link_scope,
            })
            if self.matrix_flops:
                payload["matrix_flops"] = dict(self.matrix_flops)
        return payload


@dataclass(frozen=True)
class RoutingStats:
    routing_tokens: float
    local_experts: int
    slowest_rank_tokens: float
    local_padded_tokens: float
    capacity_per_expert: float
    dropped_tokens: float
    distribution: str
    local_routing_tokens: float = 0.0
    global_scale: float = 1.0
    local_expert_counts: tuple[int, ...] = ()
    #: Logical (unpadded) rows per local expert.  ``local_expert_counts``
    #: retains the declared aligned implementation view;
    #: the served grouped shape uses this field so the op's M is the real
    #: routed row count, not the graph-bucket or per-expert padding cap.
    local_expert_logical_counts: tuple[int, ...] = ()
    #: Expected number of this rank's local experts that receive at least one
    #: routed token in the iteration.  A decode step with one token and top-k=6
    #: does not read all 96 local expert weight matrices; a grouped GEMM that
    #: skips the zero-token experts streams only the active ones.  Derived from
    #: the declared routing distribution and the EP group's routed-token volume,
    #: never from a measured latency.
    active_local_experts: float = 0.0
    #: Binomial inclusion moments: expected active experts and expected
    #: aligned rows.  These are statistical work identities, not integer shapes.
    expected_active_experts: float = 0.0
    expected_padded_rows: float | None = None
    expected_padded_rows_status: str = "unknown"
    #: Global expert ids of the rank whose counts are reported, in local order.
    local_expert_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_tokens": self.routing_tokens,
            "local_routing_tokens": self.local_routing_tokens,
            "global_scale": self.global_scale,
            "local_expert_counts": list(self.local_expert_counts),
            **({"local_expert_logical_counts": list(self.local_expert_logical_counts)}
               if self.local_expert_logical_counts else {}),
            **({"active_local_experts": self.active_local_experts}
               if self.active_local_experts > 0.0 else {}),
            "local_experts": self.local_experts,
            **({"local_expert_ids": list(self.local_expert_ids)}
               if self.local_expert_ids else {}),
            "slowest_rank_tokens": self.slowest_rank_tokens,
            "local_padded_tokens": self.local_padded_tokens,
            "capacity_per_expert": self.capacity_per_expert,
            "dropped_tokens": self.dropped_tokens,
            "distribution": self.distribution,
        }


@dataclass(frozen=True)
class IterationWork:
    ops: tuple[OpSpec, ...]
    plan: IterationPlan
    stage_count: int
    valid_tokens: int
    padded_tokens: int
    graph_bucket: int | None
    mode: str
    recipe: str
    routing: RoutingStats | None = None
    recipe_guards: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    #: Replay scope for this iteration: ``none``, ``full`` or ``piecewise``.
    #: PIECEWISE captures the non-attention subgraph and leaves attention/split
    #: ops eager; it is never a whole-iteration FULL claim.
    replay_scope: str = ""
    #: Declared splitting_ops for PIECEWISE; ``None`` uses the target default
    #: attention boundary.
    splitting_ops: tuple[str, ...] | None = None
    #: Rank-local token count when the execution plan splits the batch (PCP).
    #: ``valid_tokens`` stays the client-visible global count so throughput and
    #: padding accounting are not rewritten by a rank-local view.
    local_tokens: int = 0
    #: Declared plan/work record (strategy, degrees, pairs, local rows).  Kept
    #: separate from the prediction plan so an execution trace can be audited
    #: without re-deriving the split.
    cp_work: tuple[tuple[str, Any], ...] = ()
    #: Comm buffers/collectives this iteration's plan requires.
    cp_comm: tuple[tuple[str, Any], ...] = ()
    implementation_selections: tuple[dict[str, Any], ...] = ()

    def to_dict(self, *, include_ops: bool = True) -> dict[str, Any]:
        payload = {
            "iteration": self.plan.iteration,
            "mode": self.mode,
            "recipe": self.recipe,
            "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens,
            "padding_tokens": self.padded_tokens - self.valid_tokens,
            "graph_bucket": self.graph_bucket,
            "replay_scope": self.replay_scope,
            "stage_count": self.stage_count,
            "routing": self.routing.to_dict() if self.routing is not None else None,
            "recipe_guards": [dict(item) for item in self.recipe_guards],
            "notes": list(self.notes),
            "requests": [view.to_dict() for view in self.plan.requests],
            "implementation_selections": list(self.implementation_selections),
        }
        if self.local_tokens:
            payload["local_tokens"] = int(self.local_tokens)
        if self.cp_work:
            payload["cp_work"] = dict(self.cp_work)
        if self.cp_comm:
            payload["cp_comm"] = [dict(self.cp_comm)]
        if include_ops:
            payload["ops"] = [op.to_dict() for op in self.ops]
        return payload


__all__ = [
    "FIELD_ACTIONS", "SEMANTIC_REF_KINDS", "STATE_EFFECT_ACTIONS",
    "WORK_KINDS", "AllocationEffect", "FieldDisposition", "IterationWork",
    "OpSpec", "ProgramFragment", "ResourcePolicy", "RoutingStats",
    "SemanticAttr", "SemanticRef", "StateEffect", "WorkItem", "WorkSet",
]
