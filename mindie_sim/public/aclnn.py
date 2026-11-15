"""Public ACLNN call entry and receipt projection for T12 native programs.

[Role]
  Expose one selected T12 native program through ``simulate_aclnn_call`` and
  project its single submission receipt into ``actual_execution``.

[Invariants]
  * ``execute=True`` submits the exact program selected during preparation
    through ``submit_native_call`` once; ``execute=False`` stays timing-free.
  * ``actual_execution`` projects the per-submission receipt and handle work
    pairs, never a second price model or a global engine-service delta.

[Timeouts & Locks]
  Delegates bounded CPU Engine preparation/advancement to ``native_invocation``.

[Side-Effect Policy]
  No I/O, device context or registry mutation; caller ``Execution`` advances
  only when ``execute=True``.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .errors import CapabilityGap, ExtensionError


def simulate_aclnn_call(invocation: Any, *, hardware: Any = None,
                        supply: Any = None, policy: Any = None,
                        cores: int | None = None, mt: int | None = None,
                        phase: str | None = None, profiler_rows: Any = None,
                        strict: bool = False, execution: Any = None,
                        input_ready_ms: float = 0.0, replica: int = 0,
                        stage: int = 0, wait: bool = True,
                        execute: bool = True, name: str | None = None,
                        selection: Mapping[str, Any] | None = None,
                        occurrence: Any = None, inputs: Any = None,
                        holds: Any = None, event_bindings: Any = None,
                        output_grants: Any = None, allocations: Any = None,
                        output_bindings: Any = None,
                        pool_bindings: Any = None) -> dict[str, Any]:
    """Expose one T12 selected native program through the public SDK.

    ``execute=True`` submits the exact program selected during preparation
    through the common ``submit_native_call`` path; without a caller
    ``Execution`` a bounded hardware-only Execution is created once.  Only an
    explicit ``execute=False`` is preparation/timing-free.  The returned
    ``actual_execution`` is a projection of that submission receipt, never a
    second price model or a global engine-service delta.
    """
    from mindie_sim.prediction.piecewise.native_invocation import (
        _attach_execution_result, _attach_selection_view, _hardware_execution,
        _simulate_aclnn_call, _with_selected_catalog, submit_native_call,
    )
    from .api import _environment, _gap_kind, _machine_reference

    if (not isinstance(input_ready_ms, (int, float))
            or isinstance(input_ready_ms, bool)
            or not math.isfinite(float(input_ready_ms))
            or float(input_ready_ms) < 0):
        raise ExtensionError("aclnn_input_ready_invalid")
    input_ready_ms = float(input_ready_ms)
    environment = _environment(hardware)
    if environment is not None:
        hardware = environment.machine
    if execute and not wait and execution is None:
        raise ExtensionError(
            "aclnn_async_submission_requires_caller_execution")
    result = _simulate_aclnn_call(
        invocation, hardware=hardware, supply=supply, policy=policy,
        cores=cores, mt=mt, phase=phase, profiler_rows=profiler_rows,
        strict=strict, selection=selection, execution=execution,
        replica=replica, stage=stage, prepare_execution=bool(execute))
    result = _with_selected_catalog(result)
    if selection is not None:
        result = _attach_selection_view(result, selection)
    actual = None
    if execute:
        selected_catalog = getattr(result, "execution_catalog", None)
        if result.status == "predicted" and selected_catalog is not None:
            target = execution
            if target is None:
                target = _hardware_execution(hardware, supply=supply)
            ready = None
            if input_ready_ms > 0:
                call_id = str(getattr(result.binding, "call_id", "") or "native")
                ready = target.engine.submit(
                    "aclnn:ready:" + call_id[:16], release=input_ready_ms)
            receipt = submit_native_call(
                result, target, blocking=wait, name=name, ready=ready,
                occurrence=occurrence, inputs=inputs, holds=holds,
                event_bindings=event_bindings, replica=replica, stage=stage,
                output_grants=output_grants, allocations=allocations,
                output_bindings=output_bindings,
                pool_bindings=pool_bindings)
            result = _attach_execution_result(result, receipt)
            actual = _aclnn_execution_projection(
                result, target, wait=wait, replica=replica, stage=stage)
        else:
            actual = {
                "schema": "mindie-sim.aclnn-execution/1",
                "status": "not_submitted",
                "reasons": list(result.reasons) or [
                    "native_execution_program_unavailable"],
            }
    payload = result.to_dict()
    prediction = result.prediction or {}
    payload["duration_ms"] = prediction.get("duration_ms")
    payload["work"] = prediction.get("work")
    payload["sdk"] = {
        "schema": "mindie-sim.aclnn-call/1",
        "entry": "simulate_aclnn_call",
        "physical_plan": "T12",
        "binding_source": "native_invocation",
        "repriced": False,
        "declared_native_entry": "simulate_native_call",
        "execution_requested": bool(execute),
    }
    if result.status != "predicted":
        reasons = list(result.reasons) or ["native_program_unavailable"]
        gap = CapabilityGap(
            capability="native", entry="simulate_aclnn_call",
            kind=_gap_kind(reasons[0]), code=reasons[0].split(":", 1)[0],
            message="; ".join(str(item) for item in reasons),
            affected_results=("duration_ms", "work", "actual_execution"),
            minimal_replay_input={
                "entry": "simulate_aclnn_call",
                "invocation": (invocation if isinstance(invocation, Mapping)
                               and len(repr(invocation)) < 4096 else
                               {"type": type(invocation).__name__}),
                "hardware": (_machine_reference(environment)
                             or (hardware if isinstance(hardware, str) else None)),
            },
            fallback="not_attempted")
        payload["sdk"]["gap"] = gap.to_dict()
    if actual is not None:
        payload["actual_execution"] = actual
    return payload


def _aclnn_execution_projection(result: Any, execution: Any, *,
                                wait: bool, replica: int = 0,
                                stage: int = 0) -> dict[str, Any]:
    """Project the actual T12 submission receipt, never a global delta."""
    receipt = dict(result.execution_receipt or {})
    if not receipt:
        return {
            "schema": "mindie-sim.aclnn-execution/1",
            "status": "not_submitted",
            "reasons": ["native_execution_receipt_unavailable"],
        }
    binding = result.binding
    prediction = dict(result.prediction or {})
    program = dict(result.program or {})
    exec_program_obj = getattr(result, "execution_program", None)
    exec_program = (exec_program_obj.to_dict()
                    if exec_program_obj is not None
                    and hasattr(exec_program_obj, "to_dict")
                    else {})
    handle = result.execution_handle
    start = dict(receipt.get("start") or {})
    terminal = dict(receipt.get("terminal") or {})
    events: list[str] = []
    for event in getattr(handle, "events", ()) or ():
        name = getattr(event, "name", None)
        if name is not None:
            events.append(str(name))
    duration = receipt.get("duration_ms")
    stage_work = program.get("stage_work")
    if stage_work is None:
        stage_work = exec_program.get("stage_work")
    if stage_work is None:
        stage_work = prediction.get("stage_work")
    program_source = (
        program.get("program_source")
        or exec_program.get("program_source")
        or exec_program.get("kind")
        or getattr(exec_program_obj, "kind", None)
        or ("stage_work" if stage_work is not None else None)
    )
    return {
        "schema": "mindie-sim.aclnn-execution/1",
        "status": "completed" if wait else "submitted",
        "binding_call_id": (getattr(binding, "call_id", None)
                            if binding is not None else None),
        "program_source": program_source,
        "plan_source": "t12_selected_program",
        "stage_work": stage_work,
        "input_ready_ms": start.get("time"),
        "replica": int(replica),
        "stage": int(stage),
        "ready_ms": start.get("time"),
        "terminal_ms": terminal.get("time"),
        "duration_ms": duration,
        "charge": {
            "physical_id": ("aclnn:"
                            + str(getattr(binding, "call_id", ""))[:16]),
            "ready_ms": start.get("time"),
            "complete_ms": terminal.get("time"),
            "duration_ms": duration,
            "work": getattr(handle, "work", None),
            "service_accounting": "submission_receipt",
        },
        "handle": handle,
        "events": events,
        "execution": (execution.summary() if wait
                      and hasattr(execution, "summary") else None),
        "receipt": receipt,
    }


__all__ = ["simulate_aclnn_call", "_aclnn_execution_projection"]
