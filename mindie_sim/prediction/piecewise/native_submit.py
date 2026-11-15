"""Public simulate/submit entry points for one declared native call.

[Role] Normalize + dispatch one invocation, compile the selected catalog and
  submit it once into a caller Execution (sync or async).
[Invariants] Exactly one catalog is submitted per call; a non-predicted status
  never fabricates a duration or an execution.  Submission requires a caller
  Engine/Execution and never creates one.
[Timeouts & Locks] One bounded Engine submission; no locks.
[Side-Effect Policy] Submits only to the caller-supplied Execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
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
from .native_abi_common import (  # noqa: E402
    ABI_REPLAY_SCHEMA, ABI_REQUEST_SCHEMA, ABI_RESULT_SCHEMA,
    NativeAbiIncomplete, NativeAbiInvocation, NativeAbiMismatch,
    NativeAbiPrediction, NativeAbiTensor, NativeAbiVersionMismatch,
    _ALLOWED_API, _ALLOWED_CONDITIONS, _ALLOWED_REQUEST, _ALLOWED_TENSOR,
    _ALLOWED_VERSION, _DTYPE_ALIASES, _canonical_dtype, _digest_text,
    _execution_machine, _execution_resource_bindings, _full_typed,
    _hardware_execution, _invocation_ports, _invocation_replay,
    _native_cube_rate, _native_pool, _plain, _result, _stage_context, _strict,
)

from .native_abi_programs import (  # noqa: E402
    NativeExecutionHandle, _attach_selection_view, _caller_program_projection,
    _candidate_kernels, _source_establishment,
)

from .native_catalog import (  # noqa: E402
    native_binding_for_submission,
)

from .native_receipt import (  # noqa: E402
    _attach_execution_result, _selected_native_catalog,
    _with_selected_catalog,
)

from .native_abi_normalize import (  # noqa: E402
    _missing_fields, _replay_for, _request_entry, _unknown_replay,
    normalize_native_abi,
)
from .native_environment import (  # noqa: E402
    _environment, _implementation_policy, _select_variant,
)
from .native_leaf_dispatch import (  # noqa: E402
    _adapt_prediction, _binding_typed, _dispatch_leaf_abi, _occurrence,
)

def _simulate_aclnn_call(invocation: Any, *, hardware: Any = None,
                         supply: PipeSupply | None = None,
                         policy: ImplementationPolicy | None = None,
                         cores: int | None = None, mt: int | None = None,
                         phase: str | None = None,
                         profiler_rows: Any = None,
                         strict: bool = False,
                         selection: Mapping[str, Any] | None = None,
                         execution: Any = None,
                         replica: int = 0, stage: int = 0,
                         prepare_execution: bool = False
                         ) -> NativeAbiPrediction:
    """Predict one declared A5 native call from its independent typed input.

    ``invocation`` is API + tensor descriptors + attrs + version (+ optional
    plan conditions).  The call is normalized once, mapped to the declared
    common program and evaluated on CPU.  Missing fields return a replayable
    minimal input; an unknown/unsupported call returns ``no_match`` or
    ``not_implemented`` with no algorithm/envelope fallback.  ``strict=True``
    raises instead of returning a non-predicted status.
    """
    entry = _request_entry(invocation)
    # Resolve through the module attribute: the ordered-slot installer patches
    # ``native_calls.native_abi_spec`` after this module's imports bind, and a
    # cached function reference would miss the captured ACLNN aliases.
    from . import native_calls as _catalogue
    spec = _catalogue.native_abi_spec(entry) if entry else None
    if spec is None:
        return _strict(_result(
            "no_match", replay=_unknown_replay(invocation),
            reasons=("native_abi_spec_not_declared",),
            environment={"basis": "not_evaluated"}), strict)
    try:
        inv = normalize_native_abi(invocation, spec)
    except NativeAbiIncomplete as gap:
        missing = _missing_fields(invocation, spec) or gap.missing
        return _strict(_result(
            "insufficient_fields", spec=spec,
            replay=_replay_for(spec, invocation, missing,
                               "insufficient_fields"),
            reasons=(gap.reason, *missing),
            environment={"basis": "not_evaluated"}), strict)
    except NativeAbiVersionMismatch as mismatch:
        return _strict(_result(
            "version_mismatch", spec=spec,
            replay=_replay_for(spec, invocation, (), "version_mismatch"),
            reasons=(mismatch.reason,),
            environment={"basis": "not_evaluated"}), strict)
    except NativeAbiMismatch as mismatch:
        return _strict(_result(
            "no_match", spec=spec,
            replay=_replay_for(spec, invocation, (), "request_mismatch"),
            reasons=(mismatch.reason,),
            environment={"basis": "not_evaluated"}), strict)
    except ModelingError as error:
        return _strict(_result(
            "no_match", spec=spec,
            replay=_replay_for(spec, invocation, (), "request_malformed"),
            reasons=(error.reason,),
            environment={"basis": "not_evaluated"}), strict)
    if not spec.implemented:
        return _strict(_result(
            "not_implemented", spec=spec, invocation=inv.to_dict(),
            replay=_replay_for(spec, invocation, (), "not_implemented"),
            reasons=spec.gap or ("native_abi_program_mapper_missing",),
            environment={"basis": "not_evaluated"}), strict)
    if spec.dispatch:
        return _dispatch_leaf_abi(
            inv, spec, invocation, hardware=hardware, supply=supply,
            policy=policy, cores=cores, mt=mt, phase=phase,
            profiler_rows=profiler_rows, strict=strict,
            selection=selection, execution=execution, replica=replica,
            stage=stage, prepare_execution=prepare_execution)
    try:
        environment, supply_obj, arch = _environment(hardware, supply)
    except ModelingError as error:
        return _strict(_result(
            "environment_unavailable", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec), reasons=(error.reason,),
            environment={"basis": "not_evaluated"}), strict)
    if spec.arch and arch and spec.arch != arch:
        return _strict(_result(
            "hardware_mismatch", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"native_abi_arch_mismatch:{arch}!={spec.arch}",),
            environment=environment), strict)
    if spec.entry != "aclnnAddRmsNorm":
        return _strict(_result(
            "not_implemented", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=("native_abi_program_mapper_missing",),
            environment=environment), strict)
    sequence_rows = str(inv.conditions.get("sequence_rows") or "")
    sequence_ranks = int(inv.conditions.get("sequence_ranks") or 1)
    tensors = inv.tensor_map()
    reasons = add_rmsnorm_abi_reasons(
        tensors, inv.attrs, sequence_rows=sequence_rows,
        sequence_ranks=sequence_ranks)
    region = add_rmsnorm_abi_region(
        tensors, inv.attrs, sequence_rows=sequence_rows,
        sequence_ranks=sequence_ranks)
    if region is None:
        return _strict(_result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=("add_rmsnorm_abi_not_supported", *reasons),
            environment=environment), strict)
    try:
        variant = _select_variant(spec, inv.conditions)
    except NativeAbiMismatch as mismatch:
        return _strict(_result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec), reasons=(mismatch.reason,),
            environment=environment), strict)
    runtime_phase = str(
        phase or inv.conditions.get("phase")
        or ("decode" if int(region["rows"]) == 1 else "prefill"))
    default_cores = cores if cores is not None else (
        64 if arch == "A5" else None)
    try:
        policy_obj = policy if policy is not None else _implementation_policy(
            arch, default_cores, mt)[0]
    except ModelingError as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(getattr(error, "reason", str(error)),),
            environment=environment), strict)

    source_state = _source_establishment(spec, inv)
    api = {
        "namespace": inv.api["namespace"], "entry": inv.api["entry"],
        "aclnn_symbol": inv.api.get("aclnn_symbol"),
        "version": inv.version.get("api") or spec.api_version,
    }
    typed = _binding_typed(region)
    binding_kernels = (spec.kernels if source_state["established"]
                       else _candidate_kernels(spec.kernels))
    binding_unresolved = list(spec.unresolved)
    if not source_state["established"]:
        binding_unresolved.extend([
            "deployment_image_identity_not_established",
            "kernel_binary_identity_not_established",
        ])
    try:
        binding = bind_abi_call(
            NativeAbiRegion(
                op_id=variant.op_id, family=region["family"],
                recipe=region["recipe"],
                source_boundary=variant.source_boundary,
                dtype=region["dtype"], layout=region["layout"],
                metadata=tuple(sorted(typed["attrs"].items()))),
            api=api, source=source_state["binding_source"], typed=typed,
            kernels=binding_kernels, members=variant.members,
            owner=variant.owner, execution_core=spec.execution_core,
            unresolved=binding_unresolved,
            empty_members=(variant.name == "native_call"))
    except ModelingError as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"binding_failed:{getattr(error, 'reason', error)}",),
            environment=environment), strict)
    if binding is None:
        return _strict(_result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=("native_abi_binding_unavailable",),
            environment=environment), strict)

    # The VEC leaf owns AddRmsNorm physical prepare/predict.  T12 passes the
    # caller's real leaf variant and consumes that result unchanged; it does
    # not impose a fixed terminal_norm work program of its own.
    leaf_variant = ("terminal_norm" if variant.name == "terminal_norm"
                    else "fused_attention_norm")
    try:
        physical_prepared = prepare_add_rms_norm(
            rows=int(region["rows"]), hidden=int(region["hidden"]),
            epsilon=region["epsilon"], variant=leaf_variant,
            dtype=region["dtype"], layout=region["layout"],
            phase=runtime_phase, call_id=binding.call_id)
        physical_prediction = predict_add_rms_norm(
            physical_prepared, supply=supply_obj, policy=policy_obj,
            phase=runtime_phase)
    except (ModelingError, NativeVectorExecutionUnavailable) as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"native_vector_prepare_failed:"
                     f"{getattr(error, 'reason', error)}",),
            environment=environment), strict)

    descriptors = physical_prepared.get("descriptors") or {}
    for name, shape in region["occurrence_shapes"].items():
        descriptor = descriptors.get(name)
        expected_dtype = "fp32" if name == "rstd" else region["dtype"]
        if (not isinstance(descriptor, Mapping)
                or list(descriptor.get("shape") or ()) != list(shape)
                or str(descriptor.get("dtype") or "") != expected_dtype
                or str(descriptor.get("layout") or "") != region["layout"]):
            return _strict(_result(
                "prediction_failed", spec=spec, invocation=inv.to_dict(),
                replay=_invocation_replay(inv, spec),
                reasons=(f"native_vector_prepare_contract_mismatch:{name}",),
                environment=environment), strict)
    rstd_bytes = int(region["rows"]) * 4
    if str(physical_prediction.get("materialization", {}).get("rstd")) != str(
            rstd_bytes):
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=("native_vector_rstd_materialization_missing",),
            environment=environment), strict)

    environment = {
        **environment, "phase": runtime_phase,
        "rows": int(region["rows"]), "hidden": int(region["hidden"]),
        "vector_cores": int(policy_obj.cores),
        "policy": {"cores": int(policy_obj.cores), "mt": int(policy_obj.mt),
                   "pipeline_slots": int(policy_obj.pipeline_slots)},
        "pipe_supply": supply_obj.to_dict(),
    }
    leaf_program = dict(physical_prediction.get("program") or {})
    public_program = _caller_program_projection(
        variant=variant, inv=inv, leaf_program=leaf_program,
        physical_prediction=physical_prediction,
        physical_prepared=physical_prepared, region=region,
        rstd_bytes=rstd_bytes, leaf_variant=leaf_variant)
    prediction = _adapt_prediction(
        physical_prediction, environment, public_program)
    try:
        occurrence = _occurrence(
            binding, region, inv, variant, runtime_phase, prediction)
        profiler = (join_profiler_rows(profiler_rows, [binding])
                    if profiler_rows is not None else None)
    except ModelingError as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"occurrence_failed:{getattr(error, 'reason', error)}",),
            environment=environment), strict)
    return _strict(_result(
        "predicted", spec=spec, invocation=inv.to_dict(), binding=binding,
        prediction=prediction, program=public_program,
        occurrence=occurrence, replay=_invocation_replay(inv, spec),
        environment=environment, profiler_join=profiler,
        binding_bound=binding.bound,
        source_establishment=source_state), strict)






def simulate_aclnn_call(invocation: Any, *, hardware: Any = None,
                        supply: PipeSupply | None = None,
                        policy: ImplementationPolicy | None = None,
                        cores: int | None = None, mt: int | None = None,
                        phase: str | None = None,
                        profiler_rows: Any = None,
                        strict: bool = False,
                        selection: Mapping[str, Any] | None = None,
                        execution: Any = None, blocking: bool = True,
                        execute: bool | None = None,
                        replica: int = 0, stage: int = 0,
                        name: str | None = None,
                        output_grants: Mapping[str, Any] | None = None,
                        allocations: tuple[Any, ...] | None = None,
                        pool_bindings: Mapping[str, str] | None = None
                        ) -> NativeAbiPrediction:
    """Independent ABI entry: select, prepare, then submit the same program.

    ``selection`` is consumed before preparation when it names concrete plan
    options (tile/slots for matrix, key_tile/slots for paged FIA); it is not a
    post-hoc projection of an independently simulated call.  A caller
    ``execution`` is the one object used by ``submit_native_call``.  With no
    explicit execution, ``execute=True`` creates the bounded hardware-only
    Execution used for a blocking independent run; asynchronous submission
    always requires the caller's Execution.
    """
    if execute is None:
        execute = execution is not None
    if execute and execution is None and not blocking:
        raise ModelingError(
            "native_async_submission_requires_caller_execution")
    result = _simulate_aclnn_call(
        invocation, hardware=hardware, supply=supply, policy=policy,
        cores=cores, mt=mt, phase=phase, profiler_rows=profiler_rows,
        strict=strict, selection=selection, execution=execution,
        replica=replica, stage=stage, prepare_execution=bool(execute))
    result = _with_selected_catalog(result)
    if selection is not None:
        result = _attach_selection_view(result, selection)
    caller_execution = execution is not None
    target = execution
    if execute and target is None:
        target = _hardware_execution(hardware, supply=supply)
    if execute and target is not None and (
            result.execution_catalog is not None
            or result.execution_program is not None):
        receipt = submit_native_call(
            result, target, blocking=blocking, name=name,
            output_grants=output_grants, allocations=allocations,
            pool_bindings=pool_bindings, replica=replica, stage=stage,
            _use_catalog=not caller_execution)
        result = _attach_execution_result(result, receipt)
    return result


def _coerce_work_amount(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Fraction)):
        amount = float(value)
        return amount if math.isfinite(amount) else None
    if isinstance(value, str):
        try:
            amount = float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            try:
                amount = float(value)
            except ValueError:
                return None
        return amount if math.isfinite(amount) else None
    return None


def _normalize_handle_work(raw_work: Any) -> tuple[tuple[str, float], ...]:
    """Normalize selected-program work into ordered (resource, amount) pairs."""
    if isinstance(raw_work, Mapping):
        resources = raw_work.get("resources")
        if isinstance(resources, Mapping):
            matrix = raw_work.get("matrix")
            matrix_items = matrix.items() if isinstance(matrix, Mapping) else ()
            return (
                tuple(
                    (str(key), amt)
                    for key, val in resources.items()
                    if (amt := _coerce_work_amount(val)) is not None
                )
                + tuple(
                    (f"matrix:{key}", amt)
                    for key, val in matrix_items
                    if (amt := _coerce_work_amount(val)) is not None
                )
            )
        cost = raw_work.get("cost")
        if isinstance(cost, Mapping):
            return tuple(
                (str(key), amt)
                for key, val in cost.items()
                if (amt := _coerce_work_amount(val)) is not None
            )
        native = raw_work.get("native")
        if isinstance(native, Mapping):
            return tuple(
                (str(key), amt)
                for key, val in native.items()
                if (amt := _coerce_work_amount(val)) is not None
            )
        return tuple(
            (str(key), amt)
            for key, val in raw_work.items()
            if (amt := _coerce_work_amount(val)) is not None
        )
    if isinstance(raw_work, (tuple, list)):
        pairs: list[tuple[str, float]] = []
        for entry in raw_work:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                amt = _coerce_work_amount(entry[1])
                if amt is not None:
                    pairs.append((str(entry[0]), amt))
        return tuple(pairs)
    return ()


def submit_native_call(result: NativeAbiPrediction, execution: Any, *,
                       blocking: bool = True, name: str | None = None,
                       ready: Any = None, occurrence: Any = None,
                       inputs: Mapping[str, Any] | None = None,
                       holds: Mapping[str, Any] | None = None,
                       event_bindings: Mapping[str, Any] | None = None,
                       replica: int = 0, stage: int = 0,
                       owner: str = "",
                       output_grants: Mapping[str, Any] | None = None,
                       allocations: tuple[Any, ...] | None = None,
                       output_bindings: Mapping[str, Any] | None = None,
                       pool_bindings: Mapping[str, str] | None = None,
                       _use_catalog: bool = False
                       ) -> dict[str, Any]:
    """Submit the one selected catalog/program into the caller's Execution."""
    engine = getattr(execution, "engine", execution)
    submit_catalog = getattr(execution, "submit_catalog", None)
    if submit_catalog is None:
        submit_catalog = getattr(engine, "submit_catalog", None)
    if execution is None or submit_catalog is None:
        raise ModelingError("native_submit_requires_engine_execution")
    selected_program = result.execution_program
    if (selected_program is not None
            and getattr(selected_program, "kind", "") in {"matrix", "paged_fia", "gdn", "rope"}
            and (not _use_catalog or result.execution_catalog is None)):
        catalog = result.execution_catalog
        call_name = name or (str(catalog.catalog_id)
                             if catalog is not None else selected_program.name)
        if occurrence is None:
            occurrence = engine.new_occurrence(
                replica=int(replica), rank=0, stage=int(stage), ordinal=0)
        elif not isinstance(occurrence, OccurrenceKey):
            raise ModelingError(
                "native_submit_occurrence_must_be_occurrence_key")
        raw_handle = selected_program.start(
            execution, name=call_name, ready=ready, replica=replica,
            stage=stage, owner=owner, inputs=inputs, holds=holds,
            event_bindings=event_bindings, occurrence=occurrence,
            allocations=allocations, output_grants=output_grants,
            output_bindings=output_bindings, pool_bindings=pool_bindings)
        terminal = raw_handle.event
        if blocking and terminal is not None:
            engine.advance(until=(terminal,), first=False)
        duration_ms = None
        if (blocking and terminal is not None
                and raw_handle.start_event is not None
                and terminal.time is not None
                and raw_handle.start_event.time is not None):
            duration_ms = float(terminal.time - raw_handle.start_event.time)
        selected_id = (str(catalog.catalog_id)
                       if catalog is not None else str(selected_program.name))
        if catalog is not None:
            ports_map = {
                spec.name: PortBinding(
                    ready=(terminal if spec.direction == "out"
                           else raw_handle.start_event),
                    access=spec.access,
                    output=(spec.name if spec.direction == "out" else ""))
                for spec in catalog.ports
            }
        else:
            ports_map = {
                str(p.get("name", idx)): PortBinding(
                    ready=(terminal if p.get("role") == "output"
                           else raw_handle.start_event),
                    access=("write" if p.get("role") == "output" else "read"),
                    output=(str(p.get("name", ""))
                            if p.get("role") == "output" else ""))
                for idx, p in enumerate(selected_program.ports)
                if isinstance(p, Mapping)
            }
        submission = SimpleNamespace(
            selected_program_id=selected_id, occurrence=occurrence,
            start=raw_handle.start_event, done=terminal,
            ports=ports_map, allocations={})
        handle = replace(
            raw_handle,
            submission=submission,
            work=_normalize_handle_work(raw_handle.work),
        )
        return {
            "schema": "mindie-sim/native-execution-receipt/1",
            "call_id": (result.binding.call_id
                        if result.binding is not None else None),
            "status": result.status,
            "blocking": bool(blocking),
            "program_available": True,
            "catalog": catalog,
            "catalog_id": (catalog.catalog_id if catalog is not None else None),
            "handle": handle,
            "submission": submission,
            "selected_program_id": selected_id,
            "allocations": {},
            "output_allocations": {},
            "start_event": raw_handle.start_event,
            "terminal_event": terminal,
            "duration_ms": duration_ms,
            "execution": execution,
            "snapshot": {
                "selected_program_id": selected_id,
                "occurrence": {
                    "replica": occurrence.replica, "rank": occurrence.rank,
                    "stage": occurrence.stage, "ordinal": occurrence.ordinal,
                },
            },
            "non_claims": ["cpu_engine_terminal",
                           "no_observed_duration_injected",
                           "selected_plan_submitted_once"],
        }
    events = tuple(dict.fromkeys(
        str(key) for key in tuple(holds or {}) + tuple(event_bindings or {})))
    catalog = _selected_native_catalog(
        result, output_grants=output_grants, extra_events=events)
    call_name = name or str(catalog.catalog_id)
    start_ready = ready
    if start_ready is None:
        start_ready = engine.submit(f"{call_name}:ready", owner=owner)
    if occurrence is None:
        occurrence = engine.new_occurrence(
            replica=int(replica), rank=0, stage=int(stage), ordinal=0)
    elif not isinstance(occurrence, OccurrenceKey):
        raise ModelingError("native_submit_occurrence_must_be_occurrence_key")
    frame = native_binding_for_submission(
        catalog, result, execution, name=call_name, ready=start_ready,
        occurrence=occurrence, inputs=inputs, holds=holds,
        event_bindings=event_bindings, replica=replica, stage=stage,
        output_grants=output_grants, allocations=allocations,
        pool_bindings=pool_bindings)
    upstream = start_ready.event if hasattr(start_ready, "event") else start_ready
    after = (upstream,) + tuple(
        frame.values[spec.name].ready for spec in catalog.ports
        if spec.direction == "in" and spec.kind == "event"
        and spec.name in frame.values)
    submission = submit_catalog(
        catalog, frame, name=call_name, after=after, owner=owner)
    selected_program = result.execution_program
    if selected_program is not None and hasattr(
            selected_program, "mark_catalog_prepared"):
        selected_program.mark_catalog_prepared()
    if blocking:
        engine.advance(until=(submission.done,))
    terminal = submission.done
    duration_ms = None
    if blocking and terminal.time is not None and submission.start.time is not None:
        duration_ms = float(terminal.time - submission.start.time)
    handle = NativeExecutionHandle(
        event=terminal, start_event=submission.start,
        events=(submission.start, terminal),
        output_events=tuple((str(port), binding.ready)
                            for port, binding in submission.ports.items()
                            if binding.output or str(port) in {
                                item.name for item in catalog.ports
                                if item.direction == "out"}),
        typed_values=tuple({"name": spec.name, "role": spec.access,
                            "dtype": None, "layout": None, "shape": [],
                            "present": True}
                           for spec in catalog.ports
                           if spec.direction == "out"),
        work=tuple((term.physical_resource_role, float(term.issued))
                   for stage in catalog.stages for term in stage.work),
        submission=submission)
    return {
        "schema": "mindie-sim/native-execution-receipt/1",
        "call_id": (result.binding.call_id
                    if result.binding is not None else None),
        "status": result.status,
        "blocking": bool(blocking),
        "program_available": True,
        "catalog": catalog,
        "catalog_id": catalog.catalog_id,
        "handle": handle,
        "submission": submission,
        "selected_program_id": submission.selected_program_id,
        "allocations": {str(key): str(value) for key, value in
                        dict(submission.allocations).items()},
        "output_allocations": {
            str(spec.external_port): submission.allocations.get(str(spec.symbol))
            for spec in catalog.allocation_symbols
            if spec.external_port is not None},
        "start_event": submission.start,
        "terminal_event": terminal,
        "duration_ms": duration_ms,
        "execution": execution,
        "snapshot": engine.snapshot_submission(submission),
        "non_claims": ["cpu_engine_terminal", "no_observed_duration_injected",
                       "selected_plan_submitted_once"],
    }
