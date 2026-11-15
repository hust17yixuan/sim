"""Typed bridge from one compiled piecewise family to common family work.

[Role] Project a family compiler's already evaluated program into one typed
  family record: per-value dtype/layout/dynamic shape, declared semantic
  attributes, bound geometry, state effects/ports, and the exact issued work
  of every serial stage.  Amounts are read from ``OperatorCost``; this module
  never re-derives a formula and never turns a measured duration into work.

[Invariants]
  * A registered family supplies a logical typed declaration.  An
    unregistered family receives the physical generic declaration and is
    marked incomplete; registration alone never asserts that the remaining
    logical producer is migrated.
  * Every operand's declared shape x payload width must equal the exact value
    of an existing derived quantity (``bytes_key``).  A declared byte width
    that disagrees with its dtype raises instead of silently choosing one.
  * The stage plan is the family compiler's serial stage ledger.  If the
    evaluated cost collapsed stages into a native aggregate, the bridge
    raises ``family_stage_ledger_unavailable`` rather than presenting that
    aggregate as the family's dependencies.
  * State reads/writes are mapped to explicit common effects and ports.  A
    missing producer state is propagated from the family binder unchanged.
  * Dynamic coordinates keep both their symbol and bound value; no shape or
    duration lookup table is created.

[Timeouts & Locks] Pure bounded arithmetic over one already-evaluated
  operator.  No I/O, waits, device or locks.

[Side-Effect Policy] Returns immutable records; the only external call is the
  supplied compiled operator's existing ``evaluate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import (
    AllocationEffect, ProgramFragment, SemanticAttr, StateEffect, WorkItem,
    WorkSet,
)
from mindie_sim.schema.subgraphs import (
    EventPortSpec, PortSet, StatePortSpec, TensorPortSpec,
)
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .expressions import as_fraction

__all__ = [
    "FamilyDeclaration", "FamilyFragmentProgram", "FamilyService",
    "FamilyStage", "FamilyWork", "TypedOperand", "family_bridge_status",
    "family_work", "lower_family_work", "register_family_operands",
    "submit_family_work",
]

_OPERAND_DIRECTIONS = frozenset({"in", "out", "inout"})
_STORAGE_SCOPES = frozenset({"device", "host", "ub", "l1", "l0c"})

# Bound coordinates that are geometry (rows/tiles/positions), not work or
# service.  They are exported beside the regime so a consumer sees both the
# symbolic operand dimension and its evaluated value.
_GEOMETRY_KEYS = (
    "rows", "tokens", "kv_positions", "kv_rows", "valid_rows",
    "eligible_query_rows", "moe_rows", "index_rows", "query_tiles", "kv_tiles",
)


def _ceil_product(values: Iterable[int]) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return product


def _fraction(value: Any, reason: str) -> Fraction:
    try:
        result = as_fraction(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"{reason}:{value}") from exc
    if result < 0:
        raise ModelingError(f"{reason}:{value}")
    return result


def _milliseconds(value: Any) -> Fraction:
    """Keep a duration's decimal declaration, not its binary float expansion."""
    if isinstance(value, float):
        return Fraction(str(value))
    return as_fraction(value)


def _occurrence_identity(occurrence: Occurrence | None,
                         work: "FamilyWork") -> str:
    if occurrence is None:
        return f"{work.family}:{work.name}:standalone"
    parts = [occurrence.kind, occurrence.name]
    if occurrence.phase:
        parts.append(occurrence.phase)
    if occurrence.invocation:
        parts.append(occurrence.invocation)
    parts.append(f"serial{occurrence.serial}")
    if occurrence.request_ids:
        parts.append("req:" + "+".join(occurrence.request_ids))
    if occurrence.batch_ids:
        parts.append("batch:" + "+".join(occurrence.batch_ids))
    if occurrence.stream:
        parts.append(f"stream:{occurrence.stream}")
    if occurrence.generation is not None:
        parts.append(f"gen{occurrence.generation}")
    if occurrence.rank is not None:
        parts.append(f"rank{occurrence.rank}")
    return ":".join(parts)


def _state_scope(compiled: Any, occurrence: Occurrence | None,
                 slot: str = "", override: str = "",
                 request: str | None = None) -> str:
    if override:
        return str(override)
    parts = [f"layer:{int(getattr(compiled, 'layer', 0) or 0)}"]
    if slot:
        parts.append(f"slot:{slot}")
    if occurrence is not None:
        # Persistent state identity is the request/entity and physical owner.
        # Batch and invocation are execution identities unless the producer
        # supplies an explicit state_scope.  With multiple requests each
        # effect is emitted per request by family_work.
        if request is not None:
            parts.append(f"req:{request}")
        elif occurrence.request_ids:
            parts.append("req:" + "+".join(occurrence.request_ids))
        if occurrence.rank is not None:
            parts.append(f"rank:{occurrence.rank}")
    return ":".join(parts)


def _state_target(base: str, action: str, compiled: Any,
                  theta: Any, occurrence: Occurrence | None,
                  slot: str = "", scope_override: str = "",
                  request: str | None = None) -> str:
    layer = int(getattr(compiled, "layer", 0) or 0)
    if action == "read":
        owner = theta.find("owner_layer", None)
        if owner is not None and int(owner) >= 0:
            layer = int(owner)
    scope = _state_scope(compiled, occurrence, slot, scope_override, request)
    # The read target's layer is the producer's layer; replace the self layer
    # part while keeping the occurrence/request/rank scope.
    parts = scope.split(":")
    if parts and parts[0] == "layer":
        parts[1] = str(layer)
    return f"{base}:{':'.join(parts)}"


@dataclass(frozen=True)
class TypedOperand:
    """One declared value of a family, with its own representation.

    ``dims`` keeps the symbolic form (``"rows"``, ``"K"``) while ``values``
    holds the bound dynamic coordinates for this evaluation.  ``bytes_key``
    names the existing derived quantity that the full payload must equal;
    the bridge checks that identity instead of trusting either side alone.
    """

    name: str
    direction: str
    representation: str
    dtype: str
    layout: str
    dims: tuple[str | int, ...]
    values: tuple[tuple[str, int], ...] = ()
    bytes_per_element: Fraction = Fraction(0)
    bytes_key: str = ""
    quant: str = ""
    state_version: int | None = None
    storage: str = "device"
    alias_of: str = ""
    source: str = ""
    layout_spec: TensorLayout | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("typed_operand_name_invalid")
        if self.direction not in _OPERAND_DIRECTIONS:
            raise ModelingError(f"typed_operand_direction_invalid:{self.name}")
        for field in ("representation", "dtype", "layout"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ModelingError(f"typed_operand_{field}_invalid:{self.name}")
        dims = tuple(self.dims)
        if not dims:
            raise ModelingError(f"typed_operand_dims_invalid:{self.name}")
        for dim in dims:
            if isinstance(dim, bool) or not isinstance(dim, (str, int)):
                raise ModelingError(f"typed_operand_dims_invalid:{self.name}")
            if isinstance(dim, int) and dim < 0:
                raise ModelingError(f"typed_operand_dims_invalid:{self.name}")
            if isinstance(dim, str) and not dim.strip():
                raise ModelingError(f"typed_operand_dims_invalid:{self.name}")
        values = tuple(self.values)
        for key, value in values:
            if (not isinstance(key, str) or not key.strip()
                    or not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ModelingError(f"typed_operand_value_invalid:{self.name}")
        if len({key for key, _ in values}) != len(values):
            raise ModelingError(f"typed_operand_value_duplicate:{self.name}")
        bound = {key for key, _ in values}
        for dim in dims:
            if isinstance(dim, str) and dim not in bound:
                raise ModelingError(
                    f"typed_operand_dimension_unbound:{self.name}:{dim}")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "values", values)
        bpe = _fraction(self.bytes_per_element,
                        f"typed_operand_width_invalid:{self.name}")
        object.__setattr__(self, "bytes_per_element", bpe)
        if not isinstance(self.bytes_key, str):
            raise ModelingError(f"typed_operand_bytes_key_invalid:{self.name}")
        if not isinstance(self.quant, str):
            raise ModelingError(f"typed_operand_quant_invalid:{self.name}")
        if self.state_version is not None and (
                not isinstance(self.state_version, int)
                or isinstance(self.state_version, bool)
                or self.state_version < 0):
            raise ModelingError(
                f"typed_operand_state_version_invalid:{self.name}")
        if self.storage not in _STORAGE_SCOPES:
            raise ModelingError(f"typed_operand_storage_invalid:{self.name}")
        if self.layout_spec is not None:
            if not isinstance(self.layout_spec, TensorLayout):
                raise ModelingError(f"typed_operand_layout_invalid:{self.name}")
            if tuple(self.layout_spec.shape) != tuple(self.shape()):
                raise ModelingError(
                    f"typed_operand_layout_shape_mismatch:{self.name}:"
                    f"{self.layout_spec.shape}!={self.shape()}")

    def shape(self) -> tuple[int, ...]:
        values = dict(self.values)
        return tuple(value if isinstance(value, int) else values[value]
                     for value in self.dims)

    def payload_bytes(self) -> Fraction:
        return Fraction(_ceil_product(self.shape())) * self.bytes_per_element

    def port(self) -> TensorPortSpec:
        return TensorPortSpec(
            self.name, self.direction, self.representation,
            dtype=self.dtype, layout_kind=self.layout, scope="operator",
            owner="self", readiness="producer_complete",
            optional=bool(self.alias_of),
        )

    def tensor_value(
        self, producer: str, *, output: int = 0,
        binding: str = "static", binding_name: str = "",
        layout: TensorLayout | None = None,
        state_version: int | None = None,
        storage: str | None = None,
        alias_of: str | None = None,
    ) -> TensorValue:
        """Materialise the common typed value for this bound operand.

        The symbolic dimensions are already bound for this occurrence.  The
        caller supplies the real producer/output identity and binding: an
        input reuses its upstream value or an explicit external binding, an
        output names this occurrence and its own output index.  ``layout``
        preserves a declared view; the contiguous fallback is only used when
        the family declaration has no better layout and is marked as such by
        the caller's note.
        """
        if not isinstance(producer, str) or not producer.strip():
            raise ModelingError(f"typed_operand_producer_invalid:{self.name}")
        if output < 0:
            raise ModelingError(f"typed_operand_output_invalid:{self.name}")
        actual = layout if layout is not None else self.layout_spec
        if actual is None:
            actual = TensorLayout.contiguous(self.shape())
        if tuple(actual.shape) != tuple(self.shape()):
            raise ModelingError(
                f"typed_operand_layout_shape_mismatch:{self.name}:"
                f"{actual.shape}!={self.shape()}")
        version = self.state_version if state_version is None else state_version
        return TensorValue(
            producer, self.dtype, actual, output=output,
            quant=self.quant, layout_kind=self.layout,
            storage=(self.storage if storage is None else storage),
            alias_of=(self.alias_of if alias_of is None else alias_of),
            state_version=version, binding=binding,
            binding_name=binding_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "representation": self.representation, "dtype": self.dtype,
            "layout": self.layout, "dims": list(self.dims),
            "values": dict(self.values), "shape": list(self.shape()),
            "bytes_per_element": str(self.bytes_per_element),
            "payload_bytes": str(self.payload_bytes()),
            "bytes_key": self.bytes_key, "quant": self.quant,
            "state_version": self.state_version, "storage": self.storage,
            "alias_of": self.alias_of, "source": self.source,
            "layout_spec": (self.layout_spec.to_dict()
                            if self.layout_spec is not None else None),
            "layout_certified": self.layout_spec is not None,
        }


@dataclass(frozen=True)
class FamilyDeclaration:
    """The family-owned typed part of one operator, before stage projection."""

    operands: tuple[TypedOperand, ...] = ()
    attrs: tuple[SemanticAttr, ...] = ()
    notes: tuple[str, ...] = ()
    #: ``(stage_index, effect)`` bindings for a state effect that occurs on a
    #: specific serial stage.  Empty means "the single stage owns it"; a
    #: multi-stage family must declare this explicitly.
    stage_state_effects: tuple[tuple[int, StateEffect], ...] = ()
    stage_ports: tuple[tuple[int, PortSet], ...] = ()
    stage_allocations: tuple[tuple[int, tuple[AllocationEffect, ...]], ...] = ()
    event_ports: tuple[EventPortSpec, ...] = ()
    allocations: tuple[AllocationEffect, ...] = ()
    phase: str = ""
    #: False when the declaration has only physical byte output ranges and
    #: still lacks a producer-owned logical shape/input mapping.
    producer_complete: bool = True
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operands = tuple(self.operands)
        attrs = tuple(self.attrs)
        if not all(isinstance(item, TypedOperand) for item in operands):
            raise ModelingError("family_declaration_operand_invalid")
        if len({item.name for item in operands}) != len(operands):
            raise ModelingError("family_declaration_operand_duplicate")
        if not all(isinstance(item, SemanticAttr) for item in attrs):
            raise ModelingError("family_declaration_attr_invalid")
        if len({item.name for item in attrs}) != len(attrs):
            raise ModelingError("family_declaration_attr_duplicate")
        notes = tuple(str(item) for item in self.notes)
        bound_effects = tuple(self.stage_state_effects)
        for stage_index, effect in bound_effects:
            if (not isinstance(stage_index, int) or isinstance(stage_index, bool)
                    or stage_index < 0
                    or not isinstance(effect, StateEffect)):
                raise ModelingError("family_declaration_stage_effect_invalid")
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "attrs", attrs)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "stage_state_effects", bound_effects)
        stage_ports = tuple(self.stage_ports)
        for stage_index, ports in stage_ports:
            if (not isinstance(stage_index, int) or stage_index < 0
                    or not isinstance(ports, PortSet)):
                raise ModelingError("family_declaration_stage_ports_invalid")
        stage_allocations = tuple(self.stage_allocations)
        for stage_index, effects in stage_allocations:
            if (not isinstance(stage_index, int) or stage_index < 0
                    or not all(isinstance(effect, AllocationEffect)
                               for effect in effects)):
                raise ModelingError(
                    "family_declaration_stage_allocations_invalid")
        object.__setattr__(self, "stage_ports", stage_ports)
        object.__setattr__(self, "stage_allocations", stage_allocations)
        object.__setattr__(self, "event_ports", tuple(self.event_ports))
        object.__setattr__(self, "allocations", tuple(self.allocations))
        if not all(isinstance(effect, AllocationEffect)
                   for effect in self.allocations):
            raise ModelingError("family_declaration_allocations_invalid")
        if not isinstance(self.phase, str):
            raise ModelingError("family_declaration_phase_invalid")
        if not isinstance(self.producer_complete, bool):
            raise ModelingError("family_declaration_producer_flag_invalid")
        missing = tuple(str(item) for item in self.missing)
        object.__setattr__(self, "missing", missing)


@dataclass(frozen=True)
class FamilyStage:
    """One serial stage of the family program and its issued work.

    ``work`` is the common float-bearing ``WorkSet`` used by the Engine.
    ``exact_work`` keeps the compiler's reported rational amounts so
    attribution and conservation do not inherit float rounding.
    """

    name: str
    index: int
    depends_on: tuple[str, ...]
    work: WorkSet
    exact_work: tuple[tuple[str, Fraction], ...] = ()
    request_work: tuple[tuple[str, tuple[tuple[str, Fraction], ...]], ...] = ()
    caps: tuple[tuple[str, Fraction], ...] = ()
    state_effects: tuple[StateEffect, ...] = ()
    ports: PortSet | None = None
    event_in: tuple[EventPortSpec, ...] = ()
    event_out: tuple[EventPortSpec, ...] = ()
    allocations: tuple[AllocationEffect, ...] = ()
    phase: str = ""
    occurrence: Occurrence | None = None
    elapsed_ms: Fraction | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelingError("family_stage_name_invalid")
        if not isinstance(self.index, int) or isinstance(self.index, bool) \
                or self.index < 0:
            raise ModelingError(f"family_stage_index_invalid:{self.name}")
        depends_on = tuple(self.depends_on)
        if not all(isinstance(item, str) and item for item in depends_on):
            raise ModelingError(f"family_stage_dependency_invalid:{self.name}")
        if not isinstance(self.work, WorkSet):
            raise ModelingError(f"family_stage_work_invalid:{self.name}")
        exact = tuple((str(pipe), _fraction(value,
                       f"family_stage_work_invalid:{self.name}"))
                      for pipe, value in self.exact_work)
        if len({pipe for pipe, _ in exact}) != len(exact):
            raise ModelingError(f"family_stage_work_duplicate:{self.name}")
        object.__setattr__(self, "exact_work", exact)
        caps = tuple((str(pipe), _fraction(value,
                       f"family_stage_cap_invalid:{self.name}"))
                     for pipe, value in self.caps)
        if len({pipe for pipe, _ in caps}) != len(caps):
            raise ModelingError(f"family_stage_cap_duplicate:{self.name}")
        object.__setattr__(self, "caps", caps)
        effects = tuple(self.state_effects)
        if not all(isinstance(effect, StateEffect) for effect in effects):
            raise ModelingError(f"family_stage_state_effect_invalid:{self.name}")
        object.__setattr__(self, "state_effects", effects)
        request_work = tuple(
            (str(pipe), tuple((str(req), as_fraction(amount))
                              for req, amount in shares))
            for pipe, shares in self.request_work)
        object.__setattr__(self, "request_work", request_work)
        if self.ports is not None and not isinstance(self.ports, PortSet):
            raise ModelingError(f"family_stage_ports_invalid:{self.name}")
        object.__setattr__(self, "event_in", tuple(self.event_in))
        object.__setattr__(self, "event_out", tuple(self.event_out))
        object.__setattr__(self, "allocations", tuple(self.allocations))
        if self.occurrence is not None and not isinstance(self.occurrence, Occurrence):
            raise ModelingError(f"family_stage_occurrence_invalid:{self.name}")
        if self.elapsed_ms is not None:
            elapsed = _fraction(self.elapsed_ms,
                                f"family_stage_elapsed_invalid:{self.name}")
            object.__setattr__(self, "elapsed_ms", elapsed)
        object.__setattr__(self, "depends_on", depends_on)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "index": self.index,
            "depends_on": list(self.depends_on),
            "work": [item.to_dict() for item in self.work.issued()],
            "exact_work": {pipe: str(value)
                           for pipe, value in self.exact_work},
            "caps": {pipe: str(value) for pipe, value in self.caps},
            "state_effects": [effect.to_dict()
                              for effect in self.state_effects],
            "ports": self.ports.to_dict() if self.ports is not None else None,
            "event_in": [port.to_dict() for port in self.event_in],
            "event_out": [port.to_dict() for port in self.event_out],
            "allocations": [effect.to_dict()
                            for effect in self.allocations],
            "phase": self.phase,
            "occurrence": (self.occurrence.to_dict()
                           if self.occurrence is not None else None),
            "elapsed_ms": (None if self.elapsed_ms is None
                           else str(self.elapsed_ms)),
            "source": self.source,
        }


@dataclass(frozen=True)
class FamilyService:
    """Declared completion constraint around the structural stage body.

    ``structural_ms`` is the exact sum of the family's own stage times when
    the stage ledger owned elapsed; it is ``None`` when a native program owns
    duration.  ``completion_ms`` is the compiler's final result for this
    geometry and may include a family minimum beyond the three terms.
    """

    floor_ms: Fraction
    startup_ms: Fraction
    dependency_ms: Fraction
    structural_ms: Fraction | None
    completion_ms: Fraction
    elapsed_owner: str

    def completion(self, body_ms: Any) -> Fraction:
        body = _fraction(body_ms, "family_service_body_invalid")
        return max(self.floor_ms, self.startup_ms + body + self.dependency_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "floor_ms": str(self.floor_ms),
            "startup_ms": str(self.startup_ms),
            "dependency_ms": str(self.dependency_ms),
            "structural_ms": (None if self.structural_ms is None
                              else str(self.structural_ms)),
            "completion_ms": str(self.completion_ms),
            "elapsed_owner": self.elapsed_owner,
        }


@dataclass(frozen=True)
class FamilyWork:
    """Common typed record for one evaluated family operator."""

    name: str
    family: str
    kind: str
    variant: str
    active: bool
    operands: tuple[TypedOperand, ...]
    attrs: tuple[SemanticAttr, ...]
    ports: PortSet
    state_effects: tuple[StateEffect, ...]
    stages: tuple[FamilyStage, ...]
    reported_work: tuple[tuple[str, Fraction], ...]
    caps: tuple[tuple[str, Fraction], ...]
    materialization: tuple[tuple[str, Fraction], ...]
    internal: tuple[tuple[str, Fraction], ...]
    residency_bytes: Fraction
    geometry: tuple[tuple[str, Fraction], ...]
    derived: tuple[tuple[str, Fraction], ...]
    service: FamilyService
    formula: str
    fidelity: str
    notes: tuple[str, ...]
    projector: str
    phase: str = ""
    layer: int = 0
    allocations: tuple[AllocationEffect, ...] = ()
    event_ports: tuple[EventPortSpec, ...] = ()
    occurrence: Occurrence | None = None
    native_program: Any = None
    native_stages: tuple[tuple[Any, ...], ...] = ()
    declaration_complete: bool = True
    missing_producers: tuple[str, ...] = ()
    input_values: tuple[tuple[str, TensorValue], ...] = ()
    layouts: tuple[tuple[str, TensorLayout], ...] = ()
    request_rows: tuple[tuple[str, int], ...] = ()
    transport_plan: Any = None
    transport_plan_work: tuple[tuple[str, Fraction], ...] = ()

    def stage_work(self) -> tuple[dict[str, Fraction], ...]:
        """Exact issued work per stage; a projection of the compiler ledger."""
        return tuple(dict(stage.exact_work) for stage in self.stages)

    def issued_work(self) -> dict[str, Fraction]:
        """Exact aggregate issued work, for attribution/conservation only.

        This is not a second service authority: timing and dependencies stay
        on ``stages``/``service``; the Engine consumes the float ``WorkSet``.
        """
        return dict(self.reported_work)

    def pipe_work(self, pipe: str) -> Fraction:
        return self.issued_work().get(pipe, Fraction(0))

    def tensor_values(
        self, *,
        occurrence: Occurrence | None = None,
        input_values: Mapping[str, TensorValue] | None = None,
        layouts: Mapping[str, TensorLayout] | None = None,
    ) -> tuple[tuple[str, TensorValue], ...]:
        """Bind each operand to its real producer/value identity.

        Inputs either reuse a supplied upstream ``TensorValue`` or become
        explicit external bindings.  Outputs use this occurrence's identity
        and a distinct output index; they are never the same logical value as
        an input that merely has the same shape and dtype.
        """
        occurrence = occurrence or self.occurrence
        occurrence_name = _occurrence_identity(occurrence, self)
        provided = dict(input_values or self.input_values)
        layout_map = dict(layouts or self.layouts)
        values: list[tuple[str, TensorValue]] = []
        output_index = 0
        for operand in self.operands:
            if operand.direction in {"in", "inout"} and operand.name in provided:
                upstream = provided[operand.name]
                if not isinstance(upstream, TensorValue):
                    raise ModelingError(
                        f"family_input_value_invalid:{operand.name}")
                if upstream.dtype != operand.dtype or tuple(
                        upstream.layout.shape) != tuple(operand.shape()):
                    raise ModelingError(
                        f"family_input_value_mismatch:{operand.name}:"
                        f"{upstream.dtype}/{upstream.layout.shape}")
                values.append((operand.name, upstream))
                continue
            if operand.direction == "in":
                values.append((operand.name, operand.tensor_value(
                    "external", output=0, binding="external",
                    binding_name=f"{self.family}:{self.name}:{operand.name}",
                    layout=layout_map.get(operand.name))))
                continue
            upstream = None
            if operand.alias_of:
                upstream = provided.get(operand.alias_of)
                if upstream is None:
                    upstream = next((item for name, item in values
                                     if name == operand.alias_of), None)
            value = operand.tensor_value(
                occurrence_name, output=output_index, binding="static",
                layout=(upstream.layout if upstream is not None
                        else layout_map.get(operand.name)),
                storage=(upstream.storage if upstream is not None else None),
                alias_of=(upstream.producer if upstream is not None else None))
            output_index += 1
            values.append((operand.name, value))
        return tuple(values)

    def fragment(self) -> ProgramFragment:
        """Complete T01 fragment for a single-stage family, if exact.

        A multi-stage family has no exact per-stage operand access map here,
        so it must not be flattened into one fragment.  The returned fragment
        carries attrs/state effects and is therefore intentionally not
        ``stage_projection_blockers``-free; T04 consumes the complete fragment.
        """
        if len(self.stages) != 1:
            raise ModelingError(
                f"family_work_fragment_requires_single_stage:"
                f"{self.family}:{self.name}:{len(self.stages)}")
        occurrence = self.occurrence or Occurrence(
            kind="kernel", name=f"{self.family}:{self.name}",
            invocation="standalone", phase=self.kind)
        stage = self.stages[0]
        return ProgramFragment(
            name=f"{self.family}:{self.name}",
            occurrence=stage.occurrence or occurrence,
            attrs=self.attrs, ports=stage.ports or self.ports,
            work=stage.work, state_effects=stage.state_effects,
            allocations=stage.allocations, notes=self.notes,
        )

    def state_accesses(self) -> tuple[dict[str, Any], ...]:
        """Project existing bindings into SERVICE access records.

        Physical read/write accesses only.  commit/rollback remain decisions
        and are not counted as another write.  Requests, spans, versions and
        source producers come from this projection's occurrence/effects and
        the caller-provided value/layout bindings.
        """
        occurrence = self.occurrence or Occurrence(
            kind="kernel", name=f"{self.family}:{self.name}",
            invocation="standalone", phase=self.kind)
        request_ids = tuple(occurrence.request_ids)
        inputs = dict(self.input_values)
        layouts = dict(self.layouts)
        result: list[dict[str, Any]] = []
        for stage in self.stages:
            for effect in stage.state_effects:
                if effect.action in {"commit", "rollback"}:
                    continue
                operand = None
                for candidate in self.operands:
                    if candidate.name in effect.target:
                        operand = candidate
                        break
                if operand is None:
                    preferred = ("out" if effect.action != "read" else "in")
                    for candidate in self.operands:
                        if candidate.direction == preferred and (
                                "cache" in candidate.name
                                or "state" in candidate.name):
                            operand = candidate
                            break
                    if operand is None:
                        for candidate in self.operands:
                            if candidate.direction in {"in", "out"} and (
                                    "cache" in candidate.name
                                    or "state" in candidate.name):
                                operand = candidate
                                break
                request = ""
                marker = effect.target.find("req:")
                if marker >= 0:
                    tail = effect.target[marker + 4:]
                    request = tail.split(":", 1)[0]
                if not request and request_ids:
                    request = request_ids[0]
                request_index = (request_ids.index(request)
                                 if request in request_ids else 0)
                layout = (layouts.get(operand.name) if operand is not None
                          else None)
                if layout is None and operand is not None:
                    layout = operand.layout_spec
                span: tuple[tuple[int, int], ...] = ()
                amount = Fraction(0)
                storage = effect.target.split(":", 1)[0]
                if operand is not None:
                    shape = operand.shape()
                    total = operand.payload_bytes()
                    row_start = 0
                    rows = int(shape[0]) if shape else 0
                    if (layout is not None and len(shape) >= 2
                            and layout.strides):
                        stride0 = int(layout.strides[0] or 1)
                        row_start = int(layout.offset) // stride0
                        rows = int(layout.shape[0])
                    request_row_map = dict(self.request_rows)
                    if (len(request_ids) > 1
                            and sum(request_row_map.get(req, 0)
                                    for req in request_ids) == rows):
                        before = sum(request_row_map.get(req, 0)
                                     for req in request_ids[:request_index])
                        per_rows = request_row_map.get(request, 0)
                        row_start += before
                        rows = per_rows
                        total = total * Fraction(per_rows, sum(
                            request_row_map.get(req, 1)
                            for req in request_ids))
                    elif len(request_ids) > 1 and rows % len(request_ids) == 0:
                        per_rows = rows // len(request_ids)
                        row_start += request_index * per_rows
                        rows = per_rows
                        total = total / len(request_ids)
                    if rows > 0:
                        span = ((row_start, row_start + rows),)
                    amount = total
                upstream = inputs.get(operand.name) if operand is not None else None
                access_occ = (replace(occurrence, request_ids=(request,))
                              if request and request in request_ids
                              else occurrence)
                producer = (_occurrence_identity(access_occ, self)
                            if effect.action != "read"
                            else (upstream.producer if upstream is not None
                                  else "external"))
                if upstream is not None and upstream.storage:
                    storage = upstream.storage
                result.append({
                    "occurrence": _occurrence_identity(access_occ, self),
                    "producer": producer,
                    "request": request,
                    "stage": stage.index, "layer": self.layer,
                    "direction": ("read" if effect.action == "read"
                                  else "write"),
                    "position": ("cache" if "cache" in effect.target
                                 or "kv" in effect.target else "slot"),
                    "output": effect.target,
                    "storage": storage,
                    "state_version": effect.version,
                    "span": span, "rank": occurrence.rank,
                    "scope": ("draft" if "draft" in effect.target
                              else "target"),
                    "bytes": float(amount),
                    "source": (upstream.producer if effect.action == "read"
                               and upstream is not None else effect.target),
                    "reader": "", "grant": "", "decision": "", "last_use": "",
                })
        return tuple(result)

    def fragments(self) -> tuple[ProgramFragment, ...]:
        """Complete per-stage fragments with ports/events/allocations.

        The stage port projection is conservative: the first stage owns the
        operator inputs, the last owns the outputs, and state/effect ports
        stay with the stage that declares them.  Family producers can supply
        exact per-stage ports through ``stage_ports``.
        """
        occurrence = self.occurrence or Occurrence(
            kind="kernel", name=f"{self.family}:{self.name}",
            invocation="standalone", phase=self.kind)
        result: list[ProgramFragment] = []
        for index, stage in enumerate(self.stages):
            stage_occ = stage.occurrence or replace(
                occurrence, name=f"{occurrence.name}:stage{index}")
            result.append(ProgramFragment(
                name=f"{self.family}:{self.name}:stage{index}",
                occurrence=stage_occ, attrs=self.attrs,
                ports=stage.ports or PortSet(), work=stage.work,
                state_effects=stage.state_effects,
                allocations=stage.allocations, notes=self.notes,
            ))
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "family": self.family, "kind": self.kind,
            "variant": self.variant, "active": self.active,
            "operands": [operand.to_dict() for operand in self.operands],
            "attrs": [attr.to_dict() for attr in self.attrs],
            "ports": self.ports.to_dict(),
            "state_effects": [effect.to_dict()
                              for effect in self.state_effects],
            "stages": [stage.to_dict() for stage in self.stages],
            "reported_work": {pipe: str(value)
                              for pipe, value in self.reported_work},
            "caps": {pipe: str(value) for pipe, value in self.caps},
            "materialization": {key: str(value)
                                for key, value in self.materialization},
            "internal": {key: str(value)
                         for key, value in self.internal},
            "residency_bytes": str(self.residency_bytes),
            "geometry": {key: str(value)
                         for key, value in self.geometry},
            "derived": {key: str(value)
                        for key, value in self.derived},
            "service": self.service.to_dict(),
            "formula": self.formula, "fidelity": self.fidelity,
            "notes": list(self.notes), "projector": self.projector,
            "phase": self.phase,
            "allocations": [effect.to_dict()
                            for effect in self.allocations],
            "event_ports": [port.to_dict() for port in self.event_ports],
            "transport_plan": self.transport_plan is not None,
            "transport_plan_work": {resource: str(amount)
                                    for resource, amount in self.transport_plan_work},
            "declaration_complete": self.declaration_complete,
            "missing_producers": list(self.missing_producers),
            "occurrence": (self.occurrence.to_dict()
                           if self.occurrence is not None else None),
            "native": (self.native_program is not None
                       or bool(self.native_stages)),
        }


@dataclass(frozen=True)
class FamilyProjection:
    """One bound instance handed to every family projector."""

    geometry: Any = field(repr=False, compare=False)
    occurrence: Occurrence | None = None
    input_values: tuple[tuple[str, TensorValue], ...] = ()
    layouts: tuple[tuple[str, TensorLayout], ...] = ()
    phase: str = ""
    cost: Any = field(default=None, repr=False, compare=False)
    compiled: Any = field(default=None, repr=False, compare=False)
    selected_outputs: tuple[str, ...] = ()
    native_plan: Any = field(default=None, repr=False, compare=False)
    native_context: Any = field(default=None, repr=False, compare=False)

    def input_value(self, name: str) -> TensorValue | None:
        return dict(self.input_values).get(name)

    def layout(self, name: str) -> TensorLayout | None:
        return dict(self.layouts).get(name)


_Projector = Callable[[Any, Any, FamilyProjection | None], FamilyDeclaration]
_PROJECTORS: dict[str, tuple[_Projector, str]] = {}


def register_family_operands(family: str, projector: _Projector, *,
                             source: str = "") -> None:
    """Register one family's typed operands/attrs declaration.

    The projector receives ``(compiled_operator, evaluated_cost)`` and must
    return a :class:`FamilyDeclaration`.  Registration covers only the typed
    projection; the numerical family compiler remains the work authority.
    """
    if not isinstance(family, str) or not family.strip():
        raise ModelingError("family_typed_projector_family_invalid")
    if not callable(projector):
        raise ModelingError(f"family_typed_projector_invalid:{family}")
    if family in _PROJECTORS:
        raise ModelingError(f"family_typed_projector_duplicate:{family}")
    label = source or (
        f"{getattr(projector, '__module__', '')}:"
        f"{getattr(projector, '__name__', '')}")
    _PROJECTORS[str(family)] = (projector, label)


def family_bridge_status() -> tuple[tuple[str, str, str], ...]:
    """Every registered family compiler, marked projected or unsupported.

    Unsupported rows name the missing interface instead of implying that the
    family's numeric migration is complete.
    """
    from .families import FAMILY_COMPILERS

    rows = []
    for family in sorted(FAMILY_COMPILERS):
        entry = _PROJECTORS.get(family)
        rows.append((
            family,
            "projected" if entry is not None else "projected_generic",
            entry[1] if entry is not None
            else "generic_physical:logical_producer_pending",
        ))
    return tuple(rows)


def _derived_map(cost: Any) -> dict[str, Fraction]:
    return {str(name): as_fraction(value) for name, value in cost.derived}


def _declared_dtype(compiled: Any, cost: Any) -> str:
    theta = compiled.operator.theta
    annotations = dict(getattr(compiled.program, "annotations", ()) or ())
    for key in ("act_format", "activation_format", "dtype", "input_dtype",
                "output_format", "out_format", "kv_dtype", "weight_format"):
        value = theta.find(key, None)
        if isinstance(value, str) and value:
            return value
        value = annotations.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _generic_attr(name: str, value: Any, source: str) -> SemanticAttr | None:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float, str)):
        return SemanticAttr(name, value, "", source)
    return None


def generic_family_declaration(compiled: Any, cost: Any, context: Any = None) -> FamilyDeclaration:
    """Physical byte/element producer for a family without a logical projector.

    Outputs are the exact materialization/internal ranges the family compiler
    already reports.  The dtype comes from declared theta/annotations; when a
    payload width divides the byte count, the declaration preserves that
    dtype and element count, otherwise it is an honest byte-addressed uint8
    range.  Logical input shapes and per-value scale/zero tensors remain the
    family owner's producer contract and are named in ``missing``.
    """
    from .families_matrix import payload_width

    theta = compiled.operator.theta
    dtype = _declared_dtype(compiled, cost)
    try:
        width = Fraction(str(payload_width(dtype))) if dtype else Fraction(1)
    except (TypeError, ValueError):
        width = Fraction(1)
    if width <= 0:
        width = Fraction(1)

    operands: list[TypedOperand] = []
    seen: set[str] = set()
    for name, amount in tuple(getattr(cost, "materialization", ())) + tuple(
            getattr(cost, "internal", ())):
        label = str(name)
        if label in seen:
            continue
        seen.add(label)
        value = as_fraction(amount)
        if dtype and value % width == 0:
            elements = int(value / width)
            dims: tuple[str | int, ...] = ("elements",)
            values = (("elements", elements),)
            element_width = width
            element_dtype = dtype
        else:
            dims = ("bytes",)
            values = (("bytes", int(value)),)
            element_width = Fraction(1)
            element_dtype = "uint8"
        shape_tuple = (values[0][1],)
        operands.append(TypedOperand(
            label, "out", label, element_dtype, "strided", dims,
            values=values, bytes_per_element=element_width,
            bytes_key=label, storage="device",
            source="family:materialization",
            layout_spec=TensorLayout.contiguous(shape_tuple),
        ))
    if not operands:
        for key in ("output_bytes", "score_output_bytes", "selection_bytes",
                    "kv_bytes", "wire_bytes", "read_bytes"):
            value = _derived_map(cost).get(key)
            if value is None or value <= 0:
                continue
            if dtype and value % width == 0:
                elements = int(value / width)
                operands.append(TypedOperand(
                    "output", "out", "output", dtype, "strided",
                    ("elements",), values=(("elements", elements),),
                    bytes_per_element=width, bytes_key=key,
                    source=f"cost.derived:{key}",
                    layout_spec=TensorLayout.contiguous((elements,))))
            else:
                operands.append(TypedOperand(
                    "output", "out", "output", "uint8", "strided",
                    ("bytes",), values=(("bytes", int(value)),),
                    bytes_per_element=Fraction(1), bytes_key=key,
                    source=f"cost.derived:{key}",
                    layout_spec=TensorLayout.contiguous((int(value),))))
            break

    derived = _derived_map(cost)
    for key in ("read_bytes", "input_bytes", "wire_bytes", "bytes_per_token"):
        value = derived.get(key)
        if value is None or value <= 0:
            continue
        if dtype and value % width == 0:
            elements = int(value / width)
            operands.append(TypedOperand(
                "input", "in", "input", dtype, "strided",
                ("elements",), values=(("elements", elements),),
                bytes_per_element=width, bytes_key=key,
                source=f"cost.derived:{key}",
                layout_spec=TensorLayout.contiguous((elements,))))
        else:
            operands.append(TypedOperand(
                "input", "in", "input", "uint8", "strided",
                ("bytes",), values=(("bytes", int(value)),),
                bytes_per_element=Fraction(1), bytes_key=key,
                source=f"cost.derived:{key}",
                layout_spec=TensorLayout.contiguous((int(value),))))
        break

    attrs: list[SemanticAttr] = []
    annotations = dict(getattr(compiled.program, "annotations", ()) or ())
    for key, source in (("family", "program.annotations"),
                        ("mode", "program.annotations"),
                        ("collective", "program.annotations"),
                        ("topology", "program.annotations")):
        value = annotations.get(key)
        attr = _generic_attr(key, value, source)
        if attr is not None:
            attrs.append(attr)
    for group, values in (("matrix", theta.matrix), ("heads", theta.heads),
                          ("experts", theta.experts), ("quant", theta.quant),
                          ("parallel", theta.parallel), ("tiles", theta.tiles)):
        for key, value in values:
            attr = _generic_attr(str(key), value, f"theta.{group}")
            if attr is not None:
                attrs.append(attr)
    if dtype:
        attrs.append(SemanticAttr("dtype", dtype, "", "theta/annotations"))
    unique: dict[str, SemanticAttr] = {}
    for attr in attrs:
        unique.setdefault(attr.name, attr)

    missing = [
        "logical_input_producer",
        "logical_output_shape",
        "scale_zero_tensors",
    ]
    notes = [
        "generic physical producer: outputs are exact materialization/internal "
        "ranges; logical input/shape and scale/zero producers remain with the "
        "family owner",
    ]
    return FamilyDeclaration(
        tuple(operands), tuple(unique.values()), tuple(notes),
        producer_complete=False, missing=tuple(missing))


def _validate_declaration(declaration: FamilyDeclaration, cost: Any) -> None:
    derived = _derived_map(cost)
    for name, value in getattr(cost, "materialization", ()):
        derived.setdefault(str(name), as_fraction(value))
    for name, value in getattr(cost, "internal", ()):
        derived.setdefault(str(name), as_fraction(value))
    from .families_matrix import payload_width

    problems: list[str] = []
    for operand in declaration.operands:
        try:
            dtype_bytes(operand.dtype)
        except ModelingError:
            problems.append(f"unsupported_dtype:{operand.name}:{operand.dtype}")
            continue
        declared_width = as_fraction(payload_width(operand.dtype))
        if operand.bytes_per_element != declared_width:
            problems.append(
                f"dtype_width_mismatch:{operand.name}:"
                f"{operand.bytes_per_element}!={declared_width}:"
                f"{operand.dtype}")
        if not operand.bytes_key:
            # A per-value producer may supply an operand whose bytes are not
            # one aggregate cost key (e.g. RMSNorm activation + gamma).  The
            # shape/dtype/width are still checked; the caller-owned bytes are
            # not inferred from an unrelated aggregate.
            continue
        declared = derived.get(operand.bytes_key)
        if declared is None:
            problems.append(
                f"bytes_key_missing:{operand.name}:{operand.bytes_key}")
            continue
        expected = operand.payload_bytes()
        if expected != declared:
            problems.append(
                f"payload_mismatch:{operand.name}:"
                f"{expected}!={declared}:{operand.bytes_key}")
    if problems:
        raise ModelingError(
            "family_typed_operand_mismatch:"
            + f"{getattr(cost, 'name', '?')}:" + ",".join(problems))


def _state_effect(action: str, value: str) -> tuple[str, str, str]:
    """Map a family state effect to a common target/action/condition."""
    effect = str(action)
    detail = str(value or "")
    lowered = detail.lower()
    if effect == "kv":
        base = "kv_cache"
    elif effect == "index":
        base = "index_cache"
    elif effect == "residual":
        base = "residual"
    elif effect == "compress":
        base = "compressed_kv"
    elif effect in {"index_k_cache", "index_scale_cache"}:
        base = effect
    elif effect == "recurrent_state":
        base = "recurrent_state"
        if "gather" in lowered or "read" in lowered:
            return base, "read", ""
        return base, "write", ""
    else:
        raise ModelingError(f"family_state_effect_unknown:{effect}:{detail}")
    if "rollback" in lowered:
        return base, "rollback", detail
    if "commit" in lowered:
        return base, "commit", detail
    if "read" in lowered:
        return base, "read", ""
    # ``ratio=4`` is an algorithm attribute, not a runtime boolean condition.
    return base, "write", ""


def _state_ports(effects: tuple[StateEffect, ...], theta: Any
                 ) -> tuple[StatePortSpec, ...]:
    owner = "self"
    raw_owner = theta.find("owner_layer", None)
    if raw_owner is not None:
        layer = int(raw_owner)
        owner = f"layer:{layer}" if layer >= 0 else "self"
    ports: list[StatePortSpec] = []
    seen: set[str] = set()
    for effect in effects:
        if effect.action == "read":
            direction, suffix = "in", "state_in"
        else:
            direction, suffix = "out", "state_out"
        name = f"{effect.target}_{suffix}"
        if name in seen:
            continue
        seen.add(name)
        ports.append(StatePortSpec(
            name, direction, effect.target,
            scope=("stage" if effect.target.startswith("compressed_kv")
                   else "layer"),
            owner=owner if direction == "in" else "self",
            readiness="producer_complete",
        ))
    return tuple(ports)


def _project_ports(declaration: FamilyDeclaration,
                   effects: tuple[StateEffect, ...], theta: Any) -> PortSet:
    tensor_in = tuple(item.port() for item in declaration.operands
                      if item.direction in {"in", "inout"})
    tensor_out = tuple(item.port() for item in declaration.operands
                       if item.direction == "out")
    state = _state_ports(effects, theta)
    return PortSet(
        tensor_in=tensor_in, tensor_out=tensor_out,
        state_in=tuple(port for port in state if port.direction == "in"),
        state_out=tuple(port for port in state if port.direction == "out"),
    )


def _stage_port_set(
    index: int, count: int, declaration: FamilyDeclaration,
    effects: tuple[StateEffect, ...], theta: Any, operator_ports: PortSet,
    phase: str, occurrence: Occurrence | None,
) -> PortSet:
    explicit = {stage: ports for stage, ports in declaration.stage_ports}
    if index in explicit:
        base = explicit[index]
    elif count == 1:
        base = operator_ports
    else:
        state = _state_ports(effects, theta)
        base = PortSet(
            tensor_in=(operator_ports.tensor_in if index == 0 else ()),
            tensor_out=(operator_ports.tensor_out if index == count - 1 else ()),
            state_in=tuple(port for port in state if port.direction == "in"),
            state_out=tuple(port for port in state if port.direction == "out"),
        )
    event_in = (EventPortSpec(
        f"stage{index}_ready", "in", "ready", scope="stage"),)
    event_out = (EventPortSpec(
        f"stage{index}_complete", "out", "complete", scope="stage"),)
    if index > 0:
        event_in = (EventPortSpec(
            f"stage{index - 1}_complete", "in", "complete", scope="stage"),)
    return PortSet(
        tensor_in=base.tensor_in, tensor_out=base.tensor_out,
        state_in=base.state_in, state_out=base.state_out,
        event_in=event_in, event_out=event_out,
    )


def _project_stages(
    compiled: Any, cost: Any, *,
    caps: tuple[tuple[str, Fraction], ...] = (),
    stage_state_effects: tuple[tuple[int, StateEffect], ...] = (),
    declaration: FamilyDeclaration | None = None,
    operator_ports: PortSet | None = None,
    phase: str = "", occurrence: Occurrence | None = None,
    request_rows: tuple[tuple[str, int], ...] = (),
) -> tuple[FamilyStage, ...]:
    program = compiled.program
    family = str(getattr(compiled, "family", ""))
    name = str(getattr(compiled, "name", ""))
    if len(cost.stage_work) != len(program.stages):
        raise ModelingError(
            f"family_stage_ledger_unavailable:{family}:{name}:"
            f"compiled_stages={len(program.stages)}:"
            f"reported_stages={len(cost.stage_work)}")
    stages: list[FamilyStage] = []
    previous: tuple[str, ...] = ()
    elapsed = tuple(_milliseconds(item) for item in cost.stage_times)
    for index, (demands, reported) in enumerate(
            zip(program.stages, cost.stage_work)):
        units = {str(demand.pipe): str(demand.unit) for demand in demands}
        items: list[WorkItem] = []
        for pipe, amount in reported:
            unit = units.get(str(pipe))
            if unit is None:
                raise ModelingError(
                    f"family_stage_pipe_undeclared:{family}:{name}:"
                    f"stage={index}:pipe={pipe}")
            items.append(WorkItem(
                str(pipe), float(as_fraction(amount)), unit, "issued",
                f"{family}:{name}:stage{index}:{pipe}"))
        request_work: list[tuple[str, tuple[tuple[str, Fraction], ...]]] = []
        if (len(request_rows) > 1
                and sum(rows for _req, rows in request_rows) > 0):
            denominator = sum(rows for _req, rows in request_rows)
            for pipe, amount in reported:
                total = as_fraction(amount)
                request_work.append((str(pipe), tuple(
                    (request, total * Fraction(rows, denominator))
                    for request, rows in request_rows)))
        stage_name = f"stage{index}"
        effects = tuple(effect for stage_index, effect in stage_state_effects
                        if stage_index == index)
        ports = None
        allocations: tuple[AllocationEffect, ...] = ()
        if declaration is not None and operator_ports is not None:
            ports = _stage_port_set(
                index, len(program.stages), declaration, effects,
                compiled.operator.theta, operator_ports, phase, occurrence)
            alloc_map = {stage: tuple(values)
                         for stage, values in declaration.stage_allocations}
            allocations = alloc_map.get(index, ())
        stage_occ = None
        if occurrence is not None:
            stage_occ = replace(
                occurrence, name=f"{occurrence.name}:stage{index}")
        stages.append(FamilyStage(
            stage_name, index, previous, WorkSet(tuple(items)),
            exact_work=tuple((str(pipe), as_fraction(amount))
                             for pipe, amount in reported),
            request_work=tuple(request_work),
            caps=caps, state_effects=effects, ports=ports,
            event_in=(ports.event_in if ports is not None else ()),
            event_out=(ports.event_out if ports is not None else ()),
            allocations=allocations, phase=phase, occurrence=stage_occ,
            elapsed_ms=(elapsed[index] if len(elapsed) == len(program.stages)
                        else None),
            source=f"{family}:{name}"))
        previous = (stage_name,)
    reported_total: dict[str, Fraction] = {}
    for row in cost.stage_work:
        for pipe, amount in row:
            reported_total[str(pipe)] = (
                reported_total.get(str(pipe), Fraction(0))
                + as_fraction(amount))
    compiled_total = {str(pipe): as_fraction(amount)
                      for pipe, amount in cost.work}
    if reported_total != compiled_total:
        raise ModelingError(
            f"family_work_conservation_lost:{family}:{name}:"
            f"stages={sorted(reported_total.items())}:"
            f"cost={sorted(compiled_total.items())}")
    return tuple(stages)


def _project_native_stages(
    compiled: Any, native_stages: Iterable[Iterable[Any]], *,
    declaration: FamilyDeclaration | None = None,
    operator_ports: PortSet | None = None, phase: str = "",
    occurrence: Occurrence | None = None,
) -> tuple[tuple[FamilyStage, ...], tuple[tuple[str, Fraction], ...]]:
    """Project the native owner schedule as serial task groups with real caps."""
    family = str(getattr(compiled, "family", ""))
    name = str(getattr(compiled, "name", ""))
    groups = tuple(tuple(group) for group in native_stages)
    stages: list[FamilyStage] = []
    aggregate_caps: dict[str, Fraction] = {}
    previous: tuple[str, ...] = ()
    for index, tasks in enumerate(groups):
        items: list[WorkItem] = []
        exact: list[tuple[str, Fraction]] = []
        caps: dict[str, Fraction] = {}
        for task in tasks:
            resource = str(getattr(task, "resource", ""))
            amount = as_fraction(getattr(task, "work", 0))
            cap = getattr(task, "cap", None)
            unit = str(getattr(task, "unit", "") or "work")
            items.append(WorkItem(resource, float(amount), unit, "issued",
                                  f"{family}:{name}:native{index}:{resource}"))
            exact.append((resource, amount))
            if cap is not None:
                caps[resource] = as_fraction(cap)
                aggregate_caps[resource] = as_fraction(cap)
        stage_name = f"stage{index}"
        ports = None
        if declaration is not None and operator_ports is not None:
            ports = _stage_port_set(
                index, len(groups), declaration, (), compiled.operator.theta,
                operator_ports, phase, occurrence)
        stage_occ = (replace(occurrence, name=f"{occurrence.name}:stage{index}")
                     if occurrence is not None else None)
        stages.append(FamilyStage(
            stage_name, index, previous, WorkSet(tuple(items)),
            exact_work=tuple(exact), caps=tuple(sorted(caps.items())),
            ports=ports, event_in=(ports.event_in if ports else ()),
            event_out=(ports.event_out if ports else ()),
            phase=phase, occurrence=stage_occ,
            source=f"{family}:{name}:native_schedule"))
        previous = (stage_name,)
    return tuple(stages), tuple(sorted(aggregate_caps.items()))


def _geometry(regime: Iterable[tuple[str, Any]],
              operands: Iterable[TypedOperand],
              derived: Mapping[str, Fraction]) -> tuple[tuple[str, Fraction], ...]:
    values: dict[str, Fraction] = {}
    for key, value in regime:
        values[str(key)] = as_fraction(value)
    for operand in operands:
        for key, value in operand.values:
            values[key] = Fraction(value)
    for key in _GEOMETRY_KEYS:
        if key in derived and key not in values:
            values[key] = derived[key]
    return tuple(sorted(values.items()))


def _call_projector(projector: _Projector, compiled: Any, cost: Any,
                    context: FamilyProjection) -> FamilyDeclaration:
    try:
        return projector(compiled, cost, context)
    except TypeError as exc:
        # A newly registered projector may not yet accept the optional bound
        # context; do not misclassify its own TypeErrors as an arity issue.
        if ("positional argument" not in str(exc)
                and "takes 2" not in str(exc)):
            raise
        return projector(compiled, cost, None)


def _selected_moe_program(compiled: Any, geometry: Any,
                          occurrence: Occurrence | None) -> Any:
    """Return the NATIVE-MOE leaf program selected by this occurrence.

    The family entry does not reconstruct the leaf work from its historical
    envelope: the inbound leaf supplies the same stage program, fragment,
    named outputs and occurrence identity that its independent entry uses.
    """
    kind = str(getattr(compiled, "kind", "") or "")
    operator = getattr(compiled, "operator", None)
    name = str(getattr(operator, "name", "") or "")
    if kind != "routed_experts" and not (kind == "moe_router" and name == "router_histogram"):
        return None
    if name not in {"router_inbound", "router_permute", "router_histogram"}:
        return None
    if str(getattr(geometry, "phase_name", "") or "") not in {"prefill"}:
        return None
    from .native_moe_token_permute import normalize_local_route, route_inputs
    from .native_moe_token_permute import native_program as moe_native_program
    theta = operator.theta
    extras = dict(getattr(theta, "extras", ()) or ())
    experts = int(theta.find("E", extras.get("E", 0)) or 0)
    hidden = int(theta.find("router_hidden", 0) or 0)
    scale_cols = int(theta.find("router_scale_cols", 0) or 0)
    raw_received = theta.find("router_received_rows", None)
    rows = None if raw_received is None else max(0, int(raw_received))
    if experts <= 0 or hidden <= 0:
        return None
    if name == "router_inbound" and scale_cols <= 0:
        return None
    comm_raw = extras.get("moe_comm", theta.find("moe_comm", None))
    try:
        comm = dict(comm_raw or ())
    except (TypeError, ValueError):
        comm = {}
    local = comm.get("experts_per_device")
    local = int(local) if local is not None else None
    declared = route_inputs(compiled, extras)
    geometry_counts = (geometry.routing_for(int(getattr(compiled, "layer", 0) or 0))
                       if hasattr(geometry, "routing_for") else None)
    route = normalize_local_route(
        declared=declared, geometry_counts=geometry_counts,
        global_experts=experts, local_range=local, rows=rows,
        geometry_source="family occurrence")
    phase = str(getattr(geometry, "phase_name", "") or "prefill")
    call_id = str(getattr(occurrence, "name", "") or name)
    if occurrence is not None and getattr(occurrence, "rank", None) is not None:
        call_id = f"{call_id}:rank{occurrence.rank}"
    if name == "router_inbound":
        from .native_moe_init_routing_v3 import plan_inbound
        payload = str(theta.find("router_source_dtype", "mxfp4") or "mxfp4")
        scale_pairs = (scale_cols + 1) // 2
        plan = plan_inbound(
            call_id=call_id, phase=phase, hidden=hidden, experts=experts,
            local_experts=local, received_rows=rows, payload=payload,
            scale_cols=scale_cols, scale_pairs=scale_pairs, route=route,
            block_slots=(int(theta.find("moe_inbound_block_bytes", 0) or 0)
                         or None))
    else:
        from .native_moe_token_permute import plan_permute
        tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
        top_k = int(theta.find("k", 1) or 1)
        route_rows = tokens * top_k
        if route.rows is not None and route.rows != route_rows:
            return None
        route = normalize_local_route(
            declared=declared, geometry_counts=geometry_counts,
            global_experts=experts, local_range=local, rows=route_rows,
            geometry_source="family occurrence")
        if name == "router_histogram":
            from .native_r64_histogram import plan_histogram
            protocol = {"separate_histc": True, "local_tokens": tokens,
                        "protocol": "family_selected_histc",
                        "declaration_source": "family_work"}
            plan = plan_histogram(
                call_id=call_id, phase=phase, tokens=tokens, top_k=top_k,
                experts=experts, protocol=protocol, route=route)
        elif name == "router_permute":
            plan = plan_permute(
                call_id=call_id, phase=phase, tokens=tokens, top_k=top_k,
                hidden=hidden, route=route,
                block_slots=(int(theta.find("moe_permute_block_bytes", 0) or 0)
                             or None))
        else:
            return None
    if plan.get("fragment") is None or plan.get("execution_program") is None:
        return None
    program = moe_native_program(plan)
    if occurrence is not None and program.fragment is not None:
        program = replace(program, fragment=replace(
            program.fragment, occurrence=occurrence))
    object.__setattr__(program, "plan", plan)
    return program


def family_work(
    compiled: Any, geometry: Any, state: Any = None, *,
    readiness: bool = True,
    occurrence: Occurrence | None = None,
    state_versions: Mapping[str, int] | None = None,
    state_actions: Mapping[str, str] | None = None,
    state_slot: str = "",
    state_scope: str = "",
    input_values: Mapping[str, TensorValue] | None = None,
    layouts: Mapping[str, TensorLayout] | None = None,
    transport_plan: Any = None,
    executed_caps: Mapping[str, Fraction] | None = None,
    native_plan: Any = None,
    native_context: Any = None,
) -> FamilyWork:
    """Project one compiled family operator into common typed family work.

    The compiled operator's own ``evaluate`` produces the exact stage ledger;
    this function only shapes it into typed operands, attrs, ports, state and
    dependencies.  The actual occurrence, state versions/actions and upstream
    typed values are caller-owned execution facts, never inferred by the
    bridge.
    """
    family = str(getattr(compiled, "family", ""))
    name = str(getattr(compiled, "name", ""))
    if native_plan is None:
        native_plan = _selected_moe_program(compiled, geometry, occurrence)
    entry = _PROJECTORS.get(family)
    cost = compiled.evaluate(geometry, state, readiness=readiness)
    occurrence = occurrence or Occurrence(
        kind="kernel", name=f"{family}:{name}",
        invocation="standalone", phase=str(getattr(compiled, "kind", "") or ""))
    phase = str(getattr(geometry, "phase_name", "") or occurrence.phase or "")
    projection = FamilyProjection(
        geometry=geometry, occurrence=occurrence,
        input_values=tuple((str(key), value)
                           for key, value in dict(input_values or {}).items()),
        layouts=tuple((str(key), value)
                      for key, value in dict(layouts or {}).items()),
        phase=phase, cost=cost, compiled=compiled,
        native_plan=native_plan, native_context=native_context)
    if entry is None:
        projector = generic_family_declaration
        label = "families_bridge:generic_family_declaration"
    else:
        projector, label = entry
    declaration = _call_projector(projector, compiled, cost, projection)
    if not isinstance(declaration, FamilyDeclaration):
        raise ModelingError(
            f"family_typed_declaration_invalid:{family}:{name}")
    if native_plan is not None:
        cache = dict(getattr(native_plan, "metadata", {}) or {}).get(
            "kv_cache_access") or {}
        if isinstance(cache, Mapping) and cache.get("storage") == "paged":
            page_size = int(cache.get("page_size") or 128)
            requests = tuple(getattr(geometry, "requests", ()) or ())
            pages = max((max(1, -(-int(request.kv_len) // page_size))
                         for request in requests), default=0)
            if requests and pages > 0:
                shape = (len(requests), pages)
                block = TypedOperand(
                    "block_table", "in", "page_table", "int32", "strided",
                    shape, bytes_per_element=Fraction(4), bytes_key="",
                    source="selected plan kv_cache_access",
                    layout_spec=TensorLayout.contiguous(shape))
                declaration = replace(
                    declaration,
                    operands=tuple(declaration.operands) + (block,),
                    attrs=tuple(declaration.attrs) + (
                        SemanticAttr("page_size", page_size, "",
                                     "selected plan kv_cache_access"),))
    _validate_declaration(declaration, cost)

    from .composition import operator_segment
    segment = (None if native_plan is not None else
               operator_segment(cost, getattr(compiled, "supply", None)))
    if segment is not None and segment.native_stages:
        schedule = getattr(cost, "native_schedule", None)
        # The public compiler also presents a duration/busy/total-work
        # aggregate as one NativeSchedule.  It is not an ordered execution
        # plan and must not become a family stage.  Only a source schedule
        # (or the original NativeProgram below) crosses this boundary.
        if schedule is None or str(getattr(
                schedule, "fidelity", "partial_aggregate")) == (
                    "partial_aggregate"):
            raise ModelingError(
                f"family_stage_ledger_unavailable:{family}:{name}:"
                "aggregate_native_schedule")
    transport_plan_work: tuple[tuple[str, Fraction], ...] = ()
    if transport_plan is not None:
        try:
            resource_work = transport_plan.resource_work()
        except (AttributeError, TypeError) as exc:
            raise ModelingError(
                f"family_transport_plan_invalid:{family}:{name}") from exc
        wire = dict(resource_work.get("wire") or {})
        transport_plan_work = tuple(sorted(
            (str(resource), as_fraction(amount))
            for resource, amount in wire.items() if as_fraction(amount) > 0))
    theta = compiled.operator.theta
    phase = str(declaration.phase or phase)
    version_map = dict(state_versions or {})
    action_map = dict(state_actions or {})
    request_rows: tuple[tuple[str, int], ...] = ()
    requests = tuple(getattr(geometry, "requests", ()) or ())
    if occurrence.request_ids and len(occurrence.request_ids) == len(requests):
        request_rows = tuple(
            (str(request_id), max(0, int(getattr(request, "q_len", 0) or 0)))
            for request_id, request in zip(occurrence.request_ids, requests))

    effects: list[StateEffect] = []
    for action, value in compiled.program.state_effects:
        base, mapped_action, condition = _state_effect(action, value)
        mapped_action = str(action_map.get(base, mapped_action))
        if mapped_action not in {"read", "write", "commit", "rollback"}:
            raise ModelingError(
                f"family_state_action_unknown:{family}:{name}:{mapped_action}")
        requests = (tuple(occurrence.request_ids)
                    if not state_scope and len(occurrence.request_ids) > 1
                    else (None,))
        for request in requests:
            target = _state_target(base, mapped_action, compiled, theta,
                                   occurrence, str(state_slot or ""),
                                   str(state_scope or ""), request)
            version = version_map.get(base, version_map.get(target))
            if mapped_action in {"commit", "rollback"} and version is None:
                raise ModelingError(
                    f"family_state_version_required:{family}:{name}:"
                    f"{target}:{mapped_action}")
            effects.append(StateEffect(
                target, mapped_action,
                version=(int(version) if version is not None else None),
                condition=condition))
    declared_effects = tuple(effect
                             for _stage, effect in declaration.stage_state_effects)
    for effect in declared_effects:
        if effect not in effects:
            effects.append(effect)
    effects_tuple = tuple(effects)
    ports = _project_ports(declaration, effects_tuple, theta)
    notes = list(declaration.notes)
    active = cost.fidelity != "inactive"

    stage_effects: tuple[tuple[int, StateEffect], ...] = ()
    if declaration.stage_state_effects:
        stage_effects = declaration.stage_state_effects
        if any(stage_index >= len(compiled.program.stages)
               for stage_index, _effect in stage_effects):
            raise ModelingError(
                f"family_state_stage_out_of_range:{family}:{name}")
    elif effects_tuple:
        if len(compiled.program.stages) == 1:
            stage_effects = tuple((0, effect) for effect in effects_tuple)
        elif family == "kv_access" and name == "index_cache_store":
            # K child owns stage 0, scale child stage 1; each request's effect
            # stays with the child that actually produces it.
            if len(compiled.program.stages) != 2:
                raise ModelingError(
                    f"family_state_stage_binding_required:{family}:{name}:"
                    f"stages={len(compiled.program.stages)}")
            stage_effects = tuple(
                (0 if effect.target.startswith("index_k_cache") else 1,
                 effect) for effect in effects_tuple)
        elif (segment is not None
              and (segment.native_program is not None
                   or segment.native_stages)):
            # The native program owns its own state publication order; do not
            # invent a stage split for effects the family did not declare.
            notes.append(
                "native program owns state publication order; operator-level "
                "effects are retained without a synthetic stage split")
        else:
            raise ModelingError(
                f"family_state_stage_binding_required:{family}:{name}:"
                f"stages={len(compiled.program.stages)}")

    caps: tuple[tuple[str, Fraction], ...] = tuple(
        (str(pipe), as_fraction(value))
        for pipe, value in (segment.caps if segment is not None else ()))
    if executed_caps:
        merged = dict(caps)
        for pipe, value in executed_caps.items():
            amount = as_fraction(value)
            if amount <= 0:
                raise ModelingError(
                    f"family_executed_cap_invalid:{family}:{name}:{pipe}")
            merged[str(pipe)] = amount
        caps = tuple(sorted(merged.items()))
    stages: tuple[FamilyStage, ...] = ()
    native_stages: tuple[tuple[Any, ...], ...] = ()
    native_program = None
    native_ports = None
    if native_plan is not None:
        from .compiler import NativeProgram
        if (hasattr(native_plan, "fragment")
                and hasattr(native_plan, "service_work")):
            # NATIVE-MOE leaf program: already a KernelProgram with its typed
            # fragment, named ports/output events and original occurrence.
            native_work = tuple(sorted(
                (str(pipe), as_fraction(amount))
                for pipe, amount in dict(native_plan.service_work).items()
                if as_fraction(amount) > 0))
            native_program = NativeProgram(native_plan, native_work,
                                           Fraction(0))
            native_ports = getattr(native_plan, "ports", None)
        else:
            if native_context is None:
                raise ModelingError(
                    f"family_native_plan_context_required:{family}:{name}")
            planned_work = getattr(native_plan, "work", None)
            resources = dict(getattr(planned_work, "resources", {}) or {})
            matrix = dict(getattr(planned_work, "matrix", {}) or {})
            cube = sum((as_fraction(value) for value in matrix.values()),
                       Fraction(0))
            if cube:
                resources["cube"] = cube
            native_work = tuple(sorted(
                (str(pipe), as_fraction(amount))
                for pipe, amount in resources.items()
                if as_fraction(amount) > 0))
            native_program = NativeProgram(
                native_plan.program(native_context), native_work,
                Fraction(0))
    elif segment is not None:
        native_program = segment.native_program
    if native_ports is not None:
        ports = native_ports
    if active and native_program is not None:
        notes.append(
            "native program owns execution; no analytical stage ledger is "
            "substituted")
    elif active and segment is not None and segment.native_stages:
        native_stages = tuple(tuple(group) for group in segment.native_stages)
        stages, caps = _project_native_stages(
            compiled, native_stages, declaration=declaration,
            operator_ports=ports, phase=phase, occurrence=occurrence)
        notes.append(
            "native task schedule retained as serial task groups with executed "
            "caps; no aggregate is re-expanded as family work")
    elif active:
        stages = _project_stages(
            compiled, cost, caps=caps, stage_state_effects=stage_effects,
            declaration=declaration, operator_ports=ports, phase=phase,
            occurrence=occurrence, request_rows=request_rows)
    else:
        notes.append("inactive for this geometry: no stage work was charged")

    if (len(stages) > 1 and effects_tuple and not stage_effects
            and not declaration.stage_state_effects):
        raise ModelingError(
            f"family_state_stage_binding_required:{family}:{name}:"
            f"stages={len(stages)}")

    floor = _milliseconds(cost.get("service_floor_ms", 0))
    startup = _milliseconds(cost.get("service_startup_ms", 0))
    dependency = _milliseconds(cost.get("service_dependency_ms", 0))
    if native_stages and segment is not None:
        floor = _milliseconds(segment.floor)
        startup = _milliseconds(segment.startup)
        dependency = _milliseconds(segment.dependency)
    stage_times = tuple(_milliseconds(item) for item in cost.stage_times)
    structural = (sum(stage_times, Fraction(0)) if stage_times else None)
    completion = _milliseconds(cost.duration_ms)
    if not active:
        owner = "inactive"
        structural = None
    elif native_program is not None:
        owner = "native-program"
        structural = None
    elif native_stages:
        owner = "native-stages"
        structural = None
    elif stages and stage_times:
        owner = "stages"
        expected = max(float(floor), float(startup) + float(structural)
                       + float(dependency))
        if abs(expected - float(completion)) > max(
                1.0, abs(float(completion))) * 1e-12:
            notes.append(
                "completion differs from stage body after a declared "
                "family minimum; completion_ms is the compiler's result")
    elif compiled.program.program_duration is not None:
        owner = "native-program"
        notes.append(
            "family program_duration owns elapsed; stage work remains a "
            "structural ledger and is not a second roofline")
    else:
        owner = "stages"
    if not stages:
        structural = None
    event_ports = tuple(declaration.event_ports) + tuple(
        port for stage in stages for port in (*stage.event_in, *stage.event_out))
    if native_ports is not None:
        event_ports += tuple(native_ports.event_out)
    if any(operand.layout_spec is None for operand in declaration.operands):
        notes.append(
            "at least one operand has no producer layout declaration; the "
            "standalone TensorValue uses a declared contiguous default and "
            "must not be read as a non-contiguous view certification")

    return FamilyWork(
        name=name, family=family,
        kind=str(getattr(compiled, "kind", "") or ""),
        variant=str(getattr(compiled, "variant", "") or ""),
        active=active,
        operands=declaration.operands, attrs=declaration.attrs,
        ports=ports, state_effects=effects_tuple, stages=stages,
        reported_work=(native_work if native_plan is not None else
                       tuple((str(pipe), as_fraction(amount))
                             for pipe, amount in cost.work)),
        caps=caps,
        materialization=tuple((str(key), as_fraction(value))
                              for key, value in cost.materialization),
        internal=tuple((str(key), as_fraction(value))
                       for key, value in cost.internal),
        residency_bytes=as_fraction(cost.residency_bytes),
        geometry=_geometry(cost.regime, declaration.operands,
                           _derived_map(cost)),
        derived=tuple((str(key), as_fraction(value))
                      for key, value in cost.derived),
        service=FamilyService(
            floor, startup, dependency, structural,
            completion, owner),
        formula=str(cost.formula), fidelity=str(cost.fidelity),
        notes=tuple(notes), projector=label, phase=phase,
        layer=int(getattr(compiled, "layer", 0) or 0),
        allocations=declaration.allocations,
        event_ports=event_ports,
        declaration_complete=declaration.producer_complete,
        missing_producers=declaration.missing,
        occurrence=occurrence, native_program=native_program,
        native_stages=native_stages,
        input_values=tuple((str(name), value)
                           for name, value in dict(input_values or {}).items()),
        layouts=tuple((str(name), value)
                      for name, value in dict(layouts or {}).items()),
        request_rows=request_rows,
        transport_plan=transport_plan,
        transport_plan_work=transport_plan_work,
    )

def _readiness_inputs(fragment: ProgramFragment, ready: Any
                      ) -> dict[str, Any]:
    """Bind declared tensor/state consumer ports to the actual ready event.

    T04's conductor checks every declared consumer port; a family fragment
    from a standalone operator has no external value registry here, so the
    predecessor completion is the truthful readiness source.  This also keeps
    T04 from treating a state port as an untyped optional tensor port.
    """
    if ready is None:
        return {}
    names: dict[str, Any] = {}
    for group in ("tensor_in", "state_in", "event_in"):
        for port in getattr(fragment.ports, group):
            names[port.name] = ready
    return names


def _submit_fragment_chain(
    conductor: Any, fragments: tuple[ProgramFragment, ...], *,
    after: Iterable[Any] = (), inputs: Mapping[str, Any] | None = None,
    conditions: Mapping[str, bool] | None = None,
    holds: Mapping[str, Any] | None = None,
    event_bindings: Mapping[str, Any] | None = None,
    owner: str = "",
) -> Any:
    """One submission implementation for both public family entries."""
    predecessors = tuple(after)
    if not predecessors:
        raise ModelingError("family_fragment_chain_requires_ready_event")
    base_inputs = dict(inputs or {})
    handle = None
    for fragment in fragments:
        ready_event = predecessors[-1]
        bound_inputs = dict(base_inputs)
        for name, value in _readiness_inputs(fragment, ready_event).items():
            bound_inputs.setdefault(name, value)
        handle = conductor.submit(
            fragment, after=predecessors, inputs=bound_inputs or None,
            conditions=conditions, holds=holds,
            event_bindings=event_bindings, owner=owner)
        predecessors = (handle.event,)
    return handle


@dataclass(frozen=True)
class FamilyFragmentProgram:
    """Submit one or more complete fragments through T04's conductor.

    A stateful family cannot be represented by a bare ``Stage``: the
    conductor is what checks read versions and publishes writes at the
    fragment completion boundary.  Each fragment keeps its own occurrence and
    state effects; edges are the family's serial stage dependencies.
    """

    fragments: tuple[ProgramFragment, ...]
    edges: tuple[tuple[int, ...], ...] = ()
    caps: tuple[tuple[str, Fraction], ...] = ()

    def __post_init__(self) -> None:
        fragments = tuple(self.fragments)
        if not fragments:
            raise ModelingError("family_fragment_program_empty")
        if not all(isinstance(fragment, ProgramFragment) for fragment in fragments):
            raise ModelingError("family_fragment_program_fragment_invalid")
        edges = tuple(tuple(edge) for edge in self.edges)
        if edges and len(edges) != len(fragments):
            raise ModelingError("family_fragment_program_edges_invalid")
        for index, row in enumerate(edges):
            if any(not isinstance(item, int) or item < 0 or item >= index
                   for item in row):
                raise ModelingError(
                    f"family_fragment_program_edge_invalid:{index}")
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "edges", edges or tuple(
            (() if index == 0 else (index - 1,))
            for index in range(len(fragments))))

    def start(self, engine: Any, name: str, ready: Any, *,
              conductor: Any = None, state: Mapping[str, int] | None = None,
              owner: str = "", inputs: Mapping[str, Any] | None = None,
              conditions: Mapping[str, bool] | None = None,
              holds: Mapping[str, Any] | None = None,
              event_bindings: Mapping[str, Any] | None = None) -> Any:
        from mindie_sim.prediction.scheduling import FragmentConductor

        if conductor is None:
            conductor = FragmentConductor(engine, state=state or {})
        handle = _submit_fragment_chain(
            conductor, self.fragments, after=(ready,), inputs=inputs,
            conditions=conditions, holds=holds,
            event_bindings=event_bindings, owner=owner)
        return handle.event


def _execution_fragments(work: FamilyWork) -> tuple[ProgramFragment, ...]:
    fragments = work.fragments()
    if not fragments:
        raise ModelingError(
            f"family_fragments_empty:{work.family}:{work.name}")
    return fragments

def submit_family_work(
    conductor: Any, work: FamilyWork, *,
    after: Iterable[Any] = (), inputs: Mapping[str, Any] | None = None,
    conditions: Mapping[str, bool] | None = None,
    holds: Mapping[str, Any] | None = None,
    event_bindings: Mapping[str, Any] | None = None,
    owner: str = "",
) -> Any:
    """Submit a family's complete fragment chain through T04's conductor.

    The conductor owns the shared Engine and the state ledger.  This function
    does not create a second execution state or auto-commit writes.
    """
    if not isinstance(work, FamilyWork):
        raise ModelingError("submit_family_work_requires_family_work")
    if work.native_program is not None or work.native_stages:
        raise ModelingError(
            f"submit_family_work_native_plan_requires_native_owner:"
            f"{work.family}:{work.name}")
    fragments = _execution_fragments(work)
    return _submit_fragment_chain(
        conductor, fragments, after=after, inputs=inputs,
        conditions=conditions, holds=holds,
        event_bindings=event_bindings, owner=owner)


def lower_family_work(work: FamilyWork) -> Any:
    """Lower family work onto the existing execution contracts.

    Stateless multi-stage families use ``KernelProgram``/``Stage`` with the
    compiler's exact caps.  Stateful families return a
    ``FamilyFragmentProgram`` so T04's ``FragmentConductor`` performs read
    validation and write publication; attaching a fragment to a bare Stage
    is never treated as an executed state effect.  Native owner programs are
    returned as the original executable program.
    """
    from mindie_sim.prediction.program import KernelProgram, Stage

    if not isinstance(work, FamilyWork):
        raise ModelingError("lower_family_work_requires_family_work")
    if work.native_program is not None:
        return work.native_program
    has_state = bool(work.state_effects) or any(
        stage.state_effects for stage in work.stages)
    if has_state:
        fragments = _execution_fragments(work)
        # Preserve the executed caps beside the fragment chain.  T04's current
        # conductor has no cap parameter; the consumer can read these instead
        # of silently losing the family's cap when it selects the state path.
        return FamilyFragmentProgram(fragments, caps=work.caps)
    stages = []
    names: set[str] = set()
    fragment = work.fragment() if len(work.stages) == 1 else None
    caps = dict(work.caps)
    for stage in work.stages:
        if stage.name in names:
            raise ModelingError(f"family_stage_name_duplicate:{stage.name}")
        names.add(stage.name)
        stages.append(Stage(
            stage.name, dict(stage.work.service_work()),
            after=tuple(stage.depends_on),
            caps={pipe: float(value) for pipe, value in stage.caps} or
                 {pipe: float(value) for pipe, value in caps.items()},
            work_units={item.resource: item.unit
                        for item in stage.work.issued()},
            fragment=fragment,
        ))
    return KernelProgram(tuple(stages))
