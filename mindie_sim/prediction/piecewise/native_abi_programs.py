"""Selected native execution programs submitted into a caller Engine.

[Role] Wrap the selected leaf/program (matrix, paged FIA, GDN, RoPE, leaf) so
  one caller Execution can start it, hold its ports and expose its work.
[Invariants] A program never reprices the selected plan and never owns the
  Engine, allocations or events; the caller does.  Preparation binds the
  caller stage context once and is idempotent.  No cached runtime identity is
  stored on the prediction result.
[Timeouts & Locks] No waits or locks; submission is the caller Engine's job.
[Side-Effect Policy] Pure wrappers over the selected plan; no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import prod
import importlib
import json
import math
from types import SimpleNamespace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_native_contracts import (
    add_rmsnorm_abi_reasons, add_rmsnorm_abi_region,
)
from ..native_binding import (
    NativeAbiRegion, NativeCallBinding, bind_abi_call, join_profiler_rows,
)
from ..runtime.contracts import (
    AllocationBinding, AllocationKey, AllocationSpec, BackingKey, BindingFrame,
    CatalogGeneration, EdgeSpec, OccurrenceKey, PoolKey, PortBinding, PortSpec,
    ProgramCatalog, StageSpec, WorkTerm,
)
from .compiler import ImplementationPolicy, PipeSupply
from .native_add_rms_norm import (
    NativeVectorExecutionUnavailable, prepare_add_rms_norm,
    predict_add_rms_norm,
)
from .native_calls import (
    NativeAbiCallSpec, NativeAbiTensorSpec, a5_native_abi_specs,
    native_abi_inventory, native_abi_spec, validate_native_occurrence,
)
from .native_qbmm import QBMM_SOURCES
from .native_catalog_builder import (
    KernelSupply, StageDraft, catalog_from_drafts, drafts_for,
    drafts_from_work,
)
from .native_occupancy import cacheline_bytes, vector_segments

# Selected cross-module records and helpers (no cycles: see module DAG).
from .native_abi_common import (  # noqa: E402,F401
    ABI_REPLAY_SCHEMA, ABI_REQUEST_SCHEMA, ABI_RESULT_SCHEMA,
    NativeAbiIncomplete, NativeAbiInvocation, NativeAbiMismatch,
    NativeAbiPrediction, NativeAbiTensor, NativeAbiVersionMismatch,
    _ALLOWED_API, _ALLOWED_CONDITIONS, _ALLOWED_REQUEST, _ALLOWED_TENSOR,
    _ALLOWED_VERSION, _DTYPE_ALIASES, _canonical_dtype, _digest_text,
    _execution_machine, _execution_resource_bindings, _full_typed,
    _hardware_execution, _invocation_ports, _invocation_replay,
    _native_cube_rate, _native_pool, _plain, _result, _stage_context, _strict,
)
def _candidate_kernels(kernels):
    """Declared kernel names without claiming an established binary identity."""
    rows = []
    for kernel in kernels:
        row = {key: value for key, value in dict(kernel).items()
               if key not in {"source_path", "source_sha256"}}
        row["evidence"] = "candidate_declaration"
        row["binary_identity"] = "candidate_not_established"
        rows.append(row)
    return tuple(rows)

def _source_establishment(spec: NativeAbiCallSpec,
                          inv: NativeAbiInvocation) -> dict:
    """Only a matching deployment image establishes the declared binary."""
    declared = _digest_text(spec.image_id)
    observed = _digest_text(inv.version.get("image_id"))
    established = bool(declared and observed and observed == declared)
    return {
        "declared_image_id": spec.image_id,
        "observed_image_id": inv.version.get("image_id"),
        "declared_source_revision": spec.source_revision,
        "observed_source_revision": inv.version.get("source"),
        "established": established,
        "basis": ("deployment_image_identity" if established
                  else "api_version_only"),
        "candidate_source": dict(spec.source) if spec.source else None,
        "binding_source": (dict(spec.source)
                           if established and spec.source else None),
    }

def _caller_program_projection(*, variant: Any, inv: NativeAbiInvocation,
                               leaf_program: dict,
                               physical_prediction: dict,
                               physical_prepared: dict,
                               region: dict, rstd_bytes: int,
                               leaf_variant: str) -> dict:
    """One public caller projection shared by every result surface.

    ``physical_members`` are the actual executed steps from the VEC leaf.  The
    caller plan uses only the effective variant selected by T12.  A bare native
    request has no caller program/kind and no semantic members; it still keeps
    the true physical members.
    """
    bare = variant.name == "native_call"
    caller_members = list(variant.members)
    layer = inv.conditions.get("layer")
    physical_program = (leaf_program.get("physical_program")
                        or leaf_program.get("common_program"))
    caller_program = (None if bare
                      else (leaf_program.get("caller_program")
                            or leaf_program.get("common_program")))
    physical_members = list(leaf_program.get("physical_members") or ())
    transactions = dict(leaf_program.get("transactions") or ())
    descriptors = physical_prepared.get("descriptors") or {}
    prepared_descriptors = {
        name: {"shape": list(item.get("shape") or ()),
               "dtype": item.get("dtype"),
               "layout": item.get("layout"),
               "role": item.get("role")}
        for name, item in descriptors.items()
    }
    cross_view = {
        "basis": "same_physical_call",
        "caller_variant": variant.name,
        "caller_members": caller_members,
        "physical_program": physical_program,
        "physical_members": physical_members,
        "leaf_caller_program": caller_program,
        "prepared_schema": physical_prepared.get("schema"),
        "prepared_variant": (None if bare else leaf_variant),
        "prepared_descriptors": prepared_descriptors,
        "occurrence_shapes": {
            name: list(shape)
            for name, shape in region["occurrence_shapes"].items()
        },
        "rstd_required_bytes": rstd_bytes,
        "rstd_materialized_bytes": physical_prediction[
            "materialization"].get("rstd"),
        "work": dict(physical_prediction["work"]),
        "materialization": dict(physical_prediction["materialization"]),
        "consistent": True,
    }
    return {
        "physical": {
            "shared_leaf": "native_add_rms_norm.predict_add_rms_norm",
            "physical_program": physical_program,
            "common_program": physical_program,
            "caller_program": caller_program,
            "leaf_prepared_variant": (None if bare else leaf_variant),
            "physical_members": physical_members,
            "transactions": transactions,
            "physical_kind": physical_prediction.get("kind"),
            "family": physical_prediction.get("family"),
            "formula": physical_prediction["formula"],
            "work": dict(physical_prediction["work"]),
            "materialization": dict(physical_prediction["materialization"]),
        },
        "caller_plan": {
            "variant": variant.name,
            "owner": variant.owner,
            "members": caller_members,
            "layer": layer,
            "source_boundary": variant.source_boundary,
        },
        "cross_view": cross_view,
    }


def _binding_from_leaf_dict(row: Any) -> Any:
    if not isinstance(row, Mapping):
        return None
    members = row.get("members") if isinstance(row.get("members"), Mapping) else {}
    return NativeCallBinding(
        call_id=str(row.get("call_id") or ""),
        op_id=str(row.get("op_id") or ""),
        family=str(row.get("family") or "native"),
        recipe=str(row.get("recipe") or "native"),
        source_boundary=str(row.get("source_boundary") or "call"),
        api=dict(row.get("api") or {}),
        source=(dict(row["source"]) if isinstance(row.get("source"), Mapping)
                else row.get("source")),
        typed=dict(row.get("typed") or {}),
        occurrence=dict(row.get("occurrence") or {}),
        observed_aliases=tuple(dict(item) for item in row.get("observed_aliases") or ()),
        kernels=tuple(dict(item) for item in row.get("kernels") or ()),
        member_op_ids=tuple(str(item) for item in members.get("op_ids") or ()),
        execution_core=row.get("execution_core"),
        timing_owner=dict(row.get("timing_owner") or {}),
        evidence=dict(row.get("evidence") or {}),
        unresolved=tuple(row.get("unresolved") or ()),
        physical=row.get("physical"),
        non_claims=tuple(row.get("non_claims") or ()))


def _generic_program(*, spec: NativeAbiCallSpec, leaf_program: Any,
                     prediction: dict | None, binding: Any,
                     inv: NativeAbiInvocation) -> dict:
    """One public projection for a leaf-owned program; no field is re-priced."""
    program = dict(leaf_program or {})
    members = list(binding.member_op_ids) if binding is not None else []
    physical_members = list(program.get("members") or members)
    caller_plan = {
        "variant": str(inv.conditions.get("variant") or "native_call"),
        "owner": (binding.timing_owner or {}).get("op_id") if binding else None,
        "members": members,
        "layer": inv.conditions.get("layer"),
        "source_boundary": binding.source_boundary if binding else "call",
    }
    physical = dict(program)
    physical["physical_members"] = physical_members
    physical["work"] = (prediction or {}).get("work")
    physical["materialization"] = (prediction or {}).get("materialization")
    cross_view = {
        "basis": "leaf_owned_program",
        "caller_variant": caller_plan["variant"],
        "caller_members": members,
        "physical_members": physical_members,
        "work": (prediction or {}).get("work"),
        "materialization": (prediction or {}).get("materialization"),
        "request_conditions": dict(inv.conditions),
    }
    return {"physical": physical, "caller_plan": caller_plan,
            "cross_view": cross_view}

@dataclass(frozen=True)
class NativeExecutionHandle:
    """Live events/ports for one submitted native program."""

    event: Any
    start_event: Any = None
    events: tuple = ()
    stage_events: tuple = ()
    output_events: tuple = ()
    release_events: tuple = ()
    typed_values: tuple = ()
    work: Any = None
    selection: Any = None
    #: The actual Submission created by the shared Engine, if still live.
    submission: Any = None

    @property
    def done(self) -> bool:
        return bool(self.event is not None and self.event.done)

    def to_dict(self) -> dict[str, Any]:
        def event_view(event):
            return {
                "name": str(getattr(event, "name", "")),
                "done": bool(getattr(event, "done", False)),
                "time": (None if getattr(event, "time", None) is None
                         else float(event.time)),
            }
        return {
            "schema": "mindie-sim/native-execution-handle/1",
            "event": event_view(self.event),
            "events": [event_view(item) for item in self.events],
            "stage_events": [event_view(item) for item in self.stage_events],
            "output_events": {
                str(name): event_view(event)
                for name, event in self.output_events},
            "release_events": [event_view(item) for item in self.release_events],
            "work": _plain(self.work),
        }

class NativeExecutionProgram:
    """One selected native program, prepared once and submitted by the caller."""

    kind = "native"

    def __init__(self, *, name: str, call: str, ports=(), stages=(),
                 work: Any = None, selection: Any = None,
                 metadata: Mapping | None = None):
        self.name = str(name)
        self.call = str(call)
        self.ports = tuple(ports)
        self.stages = tuple(stages)
        self.work = work
        self.selection = selection
        self.metadata = dict(metadata or {})
        self._program = None
        self._context = None
        self._prepared_execution = None
        self._catalog_prepared = False

    @property
    def prepared(self) -> bool:
        # The selected program is materialized either as its own plan program
        # (``prepare`` + ``start``) or as the compiled native catalog the
        # recorded submission actually runs.  Both are one submittable
        # artifact for this occurrence, never a second plan.
        return self._program is not None or self._catalog_prepared

    def mark_catalog_prepared(self) -> None:
        """Record that the selected catalog is the prepared artifact."""
        self._catalog_prepared = True

    def prepare(self, execution: Any, replica: int = 0,
                stage: int = 0) -> "NativeExecutionProgram":
        return self

    def _start_program(self, execution: Any, name: str, ready: Any,
                       owner: str):
        return execution.start_native_participant(
            self._program, name, ready, owner=owner)

    def start(self, execution: Any, name: str | None = None,
              ready: Any = None, *, replica: int = 0, stage: int = 0,
              owner: str = "", inputs: Mapping[str, Any] | None = None,
              holds: Mapping[str, Any] | None = None,
              event_bindings: Mapping[str, Any] | None = None,
              occurrence: Any = None,
              allocations: tuple[Any, ...] | None = None,
              output_grants: Mapping[str, Any] | None = None,
              output_bindings: Mapping[str, Any] | None = None,
              pool_bindings: Mapping[str, str] | None = None) -> Any:
        self.prepare(execution, replica=replica, stage=stage)
        engine = execution.engine
        name = name or self.name
        if hasattr(ready, "event"):
            ready = ready.event
        if ready is None:
            ready = engine.submit(f"{name}:ready", owner=owner)
        start_ready = ready
        grants = []
        if output_grants:
            from ..engine import Allocation
            for output_name, grant in dict(output_grants).items():
                if grant is None or grant.get("borrow"):
                    continue
                if not isinstance(grant, Mapping):
                    raise ModelingError(
                        f"native_execution_grant_invalid:{output_name}")
                pool = str(grant.get("pool") or "")
                size = grant.get("size")
                if not pool or size is None:
                    raise ModelingError(
                        f"native_execution_grant_incomplete:{output_name}")
                allocation = Allocation(
                    str(grant.get("name") or f"{output_name}_grant"),
                    pool, float(size))
                acquire = engine.submit(
                    f"{name}:grant:{output_name}", after=(start_ready,),
                    acquire=(allocation,), owner=owner)
                grants.append((allocation, acquire, grant))
                start_ready = acquire
        event = self._start_program(execution, name, start_ready, owner)
        releases = []
        for allocation, _acquire, grant in grants:
            after = [event]
            hold = grant.get("last_reader", grant.get("hold"))
            if hold is not None:
                after.append(hold.event if hasattr(hold, "event") else hold)
            releases.append(engine.submit(
                f"{name}:release:{allocation.name}", after=tuple(after),
                free=(allocation,), owner=owner))
        return NativeExecutionHandle(
            event=event, start_event=ready, events=(event,),
            release_events=tuple(releases), typed_values=self.ports,
            work=self.work)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mindie-sim/native-execution-program/1",
            "kind": self.kind,
            "name": self.name,
            "call": self.call,
            "program_available": self._program is not None,
            "ports": [dict(item) if isinstance(item, Mapping) else item
                      for item in self.ports],
            "stages": [dict(item) if isinstance(item, Mapping) else str(item)
                       for item in self.stages],
            "work": _plain(self.work),
            "selection": _plain(self.selection),
            "metadata": _plain(self.metadata),
            "non_claims": ["selected_plan_not_repriced",
                           "caller_execution_owns_engine_and_memory"],
        }


class MatrixExecutionProgram(NativeExecutionProgram):
    """The selected ``OperatorPlan.program`` for a matrix ABI record."""

    kind = "matrix"

    def __init__(self, *, plan: Any, leaf: Mapping[str, Any],
                 inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                 selection: Any = None):
        descriptor = dict(plan.metadata.get("operator_plan") or {})
        super().__init__(
            name="native_matrix", call=spec.entry,
            ports=_invocation_ports(inv),
            stages=tuple(descriptor.get("kernels") or ()),
            work={"matrix": dict(plan.work.matrix),
                  "resources": dict(plan.work.resources)},
            selection=selection,
            metadata={"entry": spec.entry,
                      "variant": descriptor.get("variant"),
                      "preparation": descriptor.get("preparation"),
                      "preparation_owner": descriptor.get("preparation_owner")})
        self.plan = plan
        self.leaf = dict(leaf)

    def prepare(self, execution: Any, replica: int = 0,
                stage: int = 0) -> "MatrixExecutionProgram":
        if self._program is not None:
            return self
        from .native_matrix_abi import _execution_inputs
        op, _recipe = _execution_inputs(self.leaf)
        context = _stage_context(execution, op, self.plan.work,
                                 replica=replica, stage=stage)
        self._program = self.plan.program(context)
        self._context = context
        return self


class PagedFIAExecutionProgram(NativeExecutionProgram):
    """The ordinary paged-FIA ``StagedGQADecode`` program for one op."""

    kind = "paged_fia"

    def __init__(self, *, op: Any, recipe: Any, plan: Any,
                 native_plan: Mapping[str, Any],
                 inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                 selection: Any = None):
        super().__init__(
            name="native_paged_fia", call=spec.entry,
            ports=tuple(
                {"name": name, **dict(value)}
                for name, value in (native_plan.get("ports") or {}).items()
                if value),
            stages=("q_load", "key_tiles", "final_store"),
            work={"native": dict(native_plan.get("work") or {}),
                  "resources": dict(plan.work.resources),
                  "matrix": dict(plan.work.matrix)},
            selection=selection,
            metadata={"variant": (native_plan.get("abi") or {}).get("entry"),
                      "strategy": native_plan.get("strategy")})
        self.op = op
        self.recipe = recipe
        self.plan = plan
        self.native_plan = dict(native_plan)

    def prepare(self, execution: Any, replica: int = 0,
                stage: int = 0) -> "PagedFIAExecutionProgram":
        if self._program is not None:
            return self
        context = _stage_context(execution, self.op, self.plan.work,
                                 replica=replica, stage=stage)
        self._program = self.plan.program(context)
        self._context = context
        return self


class GDNExecutionProgram(NativeExecutionProgram):
    """The selected F-STATE GDN ``call`` submitted to the caller Execution."""

    kind = "gdn"

    def __init__(self, *, call: Any, program: Mapping[str, Any],
                 inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                 selection: Any = None):
        super().__init__(
            name="native_gdn", call=spec.entry,
            ports=_invocation_ports(inv),
            stages=tuple(program.get("members") or ()),
            work={"native": dict(program.get("work") or {}),
                  "state_ports": tuple(program.get("state_ports") or ())},
            selection=selection,
            metadata={"partition": program.get("partition"),
                      "state_ports": tuple(program.get("state_ports") or ()),
                      "binding_relation": program.get("binding_relation")})
        self.call = call
        self.program = dict(program)

    def start(self, execution: Any, name: str | None = None,
              ready: Any = None, *, replica: int = 0, stage: int = 0,
              owner: str = "", inputs: Mapping[str, Any] | None = None,
              holds: Mapping[str, Any] | None = None,
              event_bindings: Mapping[str, Any] | None = None,
              occurrence: Any = None,
              allocations: tuple[Any, ...] | None = None,
              output_grants: Mapping[str, Any] | None = None,
              output_bindings: Mapping[str, Any] | None = None,
              pool_bindings: Mapping[str, str] | None = None) -> Any:
        from mindie_sim.schema.operators import IterationWork
        from mindie_sim.schema.workload import IterationPlan
        engine = execution.engine
        name = name or self.name
        if hasattr(ready, "event"):
            ready = ready.event
        if ready is None:
            ready = engine.submit(f"{name}:ready", owner=owner)
        work = IterationWork(
            self.call.ops, IterationPlan(()), 1, 0, 0, None, "graph",
            "declared_graph", replay_scope="none")
        terminal = engine.future(f"{name}:complete")

        def begin(_event):
            saved = execution.recipes
            execution.recipes = self.call.recipes
            try:
                step = execution.add(work, replica, engine.now)
            finally:
                execution.recipes = saved
            engine.submit(f"{name}:forward", after=(step.completion,),
                          on_complete=lambda _done: engine.resolve(terminal),
                          owner=owner)

        engine.submit(f"{name}:begin", after=(ready,), on_complete=begin,
                      owner=owner)
        return NativeExecutionHandle(
            event=terminal, start_event=ready, events=(terminal,),
            typed_values=self.ports, work=self.work)


class RoPEExecutionProgram(NativeExecutionProgram):
    """The selected family RoPE cost submitted through ``submit_cost``."""

    kind = "rope"

    def __init__(self, *, cost: Any, compiled: Any, geometry: Any,
                 supply: Any, inv: NativeAbiInvocation,
                 spec: NativeAbiCallSpec, selection: Any = None,
                 program: Mapping[str, Any] | None = None):
        super().__init__(
            name="native_rope", call=spec.entry,
            ports=_invocation_ports(inv),
            stages=tuple(getattr(cost, "stage_work", ()) or ()),
            work={"cost": dict(cost.work)},
            selection=selection,
            metadata={"dominant_pipe": cost.dominant_pipe,
                      "program": dict(program or {})})
        self.cost = cost
        self.compiled = compiled
        self.geometry = geometry
        self.supply = supply
        self.program = dict(program or {})

    def start(self, execution: Any, name: str | None = None,
              ready: Any = None, *, replica: int = 0, stage: int = 0,
              owner: str = "", inputs: Mapping[str, Any] | None = None,
              holds: Mapping[str, Any] | None = None,
              event_bindings: Mapping[str, Any] | None = None,
              occurrence: Any = None,
              allocations: tuple[Any, ...] | None = None,
              output_grants: Mapping[str, Any] | None = None,
              output_bindings: Mapping[str, Any] | None = None,
              pool_bindings: Mapping[str, str] | None = None) -> Any:
        from .composition import submit_cost
        engine = execution.engine
        name = name or self.name
        if hasattr(ready, "event"):
            ready = ready.event
        if ready is None:
            ready = engine.submit(f"{name}:ready", owner=owner)
        submitted = submit_cost(
            self.cost, execution, name=name, ready=ready,
            supply=self.supply, operator=self.compiled,
            geometry=self.geometry,
            resource_bindings=_execution_resource_bindings(
                execution, replica, stage),
            replica=replica, stage=stage)
        events = tuple(submitted.events)
        terminal = events[-1] if events else ready
        return NativeExecutionHandle(
            event=terminal, start_event=ready, events=events,
            typed_values=self.ports, work=self.work)


def _selection_program_options(selection: Any) -> dict[str, Any]:
    """Read the selected program's actual options; never infer a new plan."""
    if not isinstance(selection, Mapping):
        return {}
    delivery = selection.get("delivery")
    program = selection.get("program")
    if not isinstance(program, Mapping):
        program = (delivery or {}).get("program") if isinstance(
            delivery, Mapping) else None
    if not isinstance(program, Mapping):
        return {}
    options = program.get("options")
    result = dict(options) if isinstance(options, Mapping) else {}
    for key in ("tile", "slots", "key_tile", "query_tile", "kv_splits",
                "target_cores", "kv_mapping", "template_execution"):
        if key in program and key not in result:
            result[key] = program[key]
    return result


def _attach_selection_view(result: NativeAbiPrediction,
                           selection: Mapping[str, Any]) -> NativeAbiPrediction:
    """Attach the same selected delivery to the already-selected program.

    The actual program was chosen before preparation from this selection; this
    function only makes the same fact visible to the binding/JSON consumers.
    """
    selected = dict(selection)
    program = dict(result.program or {})
    program["selection"] = selected
    physical = dict(program.get("physical") or {})
    physical["selection"] = selected
    physical["selection_program"] = selected.get("program")
    physical["input_handles"] = selected.get("input_handles")
    physical["output_handles"] = selected.get("output_handles")
    delivery = selected.get("delivery")
    if isinstance(delivery, Mapping):
        physical["selection_public_roots"] = list(
            delivery.get("public_roots") or ())
    program["physical"] = physical
    occurrence = dict(result.occurrence or {})
    occurrence["selection"] = selected
    prediction = dict(result.prediction or {})
    prediction["program"] = program
    prediction["selection"] = selected
    replay = dict(result.replay or {})
    replay["selection"] = selected
    binding = result.binding
    if binding is not None:
        binding = replace(
            binding, typed={**binding.typed, "selection": selected},
            occurrence={**binding.occurrence, "selection": selected},
            physical={**(binding.physical or {}), "selection": selected})
    return replace(
        result, binding=binding, prediction=prediction, program=program,
        occurrence=occurrence, replay=replay)
