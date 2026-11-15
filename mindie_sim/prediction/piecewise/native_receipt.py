"""Execution receipts and duration attachment for a submitted native call.

[Role] Project the live handle/submission into a serializable receipt and
  attach the caller-Engine terminal duration to the prediction.
[Invariants] A receipt reports the modeled Engine terminal, never an observed
  target duration.  Timing-free work stays available beside it.
[Timeouts & Locks] Reads the caller handle only; no waits.
[Side-Effect Policy] Pure projection; no submission, no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
from math import prod
import importlib
import json
import math
from types import SimpleNamespace
from fractions import Fraction
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
from .native_catalog import (  # noqa: E402
    catalog_modeled_duration_ms, native_catalog_for_result,
)


def _receipt_json_value(value: Any) -> Any:
    """Convert live runtime/submission objects into plain JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, PortBinding):
        return {
            "ready": _receipt_json_value(value.ready),
            "access": str(value.access),
            "output": str(value.output),
            "producer": _receipt_json_value(value.producer),
            "allocation": (None if value.allocation is None
                           else str(value.allocation)),
            "backing": (None if value.backing is None
                        else str(value.backing)),
        }
    if type(value).__name__ == "Event" or (
            hasattr(value, "name") and hasattr(value, "done")
            and hasattr(value, "time") and not isinstance(value, Mapping)):
        raw_time = getattr(value, "time", None)
        return {
            "name": str(getattr(value, "name", "")),
            "done": bool(getattr(value, "done", False)),
            "time": None if raw_time is None else float(raw_time),
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _receipt_json_value(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _receipt_json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _receipt_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_receipt_json_value(item) for item in value]
    return str(value)


def _receipt_view(receipt: Mapping[str, Any]) -> dict[str, Any]:
    execution = receipt.get("execution")
    summary = execution.engine.summary() if execution is not None else {}
    handle = receipt.get("handle")
    submission = receipt.get("submission")
    snapshot = receipt.get("snapshot")
    if snapshot is None and submission is not None and execution is not None:
        if type(submission).__name__ == "Submission":
            snapshot = execution.engine.snapshot_submission(submission)
        else:
            occ = getattr(submission, "occurrence", None)
            snapshot = {
                "selected_program_id": str(
                    getattr(submission, "selected_program_id", "")),
                "occurrence": _receipt_json_value(occ),
            }
    snapshot_json = (_receipt_json_value(snapshot)
                     if isinstance(snapshot, Mapping) else None)
    ports = {}
    if submission is not None:
        for port_name, binding in dict(submission.ports).items():
            ports[str(port_name)] = _receipt_json_value(binding)
    terminal = receipt.get("terminal_event")
    return {
        "schema": "mindie-sim/native-execution-receipt/1",
        "call_id": receipt.get("call_id"),
        "status": receipt.get("status"),
        "blocking": bool(receipt.get("blocking")),
        "duration_ms": receipt.get("duration_ms"),
        "selected_program_id": (str(getattr(submission, "selected_program_id", ""))
                                if submission is not None
                                else receipt.get("selected_program_id")),
        "allocations": ({str(key): str(value) for key, value in
                         dict(submission.allocations).items()}
                        if submission is not None else {}),
        "output_allocations": _receipt_json_value(
            receipt.get("output_allocations") or {}),
        "occurrence": (snapshot_json.get("occurrence")
                       if isinstance(snapshot_json, Mapping) else
                       _receipt_json_value(receipt.get("occurrence"))),
        "ports": ports,
        "snapshot": snapshot_json,
        "start": _receipt_json_value(receipt.get("start_event")),
        "terminal": _receipt_json_value(terminal),
        "handle": (handle.to_dict()
                   if handle is not None and hasattr(handle, "to_dict")
                   else None),
        "serviced": {str(key): float(value) for key, value in
                     dict(summary.get("serviced") or {}).items()},
        "peak_allocated": _receipt_json_value(
            dict(summary.get("peak_allocated") or {})),
        "live_allocated": _receipt_json_value(
            dict(summary.get("live_allocated") or {})),
        "completed_tasks": int(summary.get("completed_tasks") or 0),
        "non_claims": ["cpu_engine_terminal",
                       "no_observed_duration_injected"],
    }


def native_execution_receipt(result: NativeAbiPrediction,
                            execution: Any = None) -> dict[str, Any]:
    """Fresh receipt for a previously submitted (possibly async) program."""
    handle = result.execution_handle
    if handle is None:
        return dict(result.execution_receipt or {})
    submission = getattr(handle, "submission", None)
    receipt = {
        "call_id": (result.binding.call_id
                    if result.binding is not None else None),
        "status": result.status, "blocking": False,
        "handle": handle,
        "submission": submission,
        "selected_program_id": (None if submission is None else
                                str(submission.selected_program_id)),
        "start_event": (None if submission is None else submission.start),
        "terminal_event": (handle.event if submission is None
                           else submission.done),
        "execution": execution, "duration_ms": None,
    }
    if handle.start_event is not None and handle.event is not None:
        if (handle.start_event.time is not None
                and handle.event.time is not None):
            receipt["duration_ms"] = float(
                handle.event.time - handle.start_event.time)
    return _receipt_view(receipt)


def _attach_execution_result(result: NativeAbiPrediction,
                             receipt: Mapping[str, Any]
                             ) -> NativeAbiPrediction:
    view = _receipt_view(receipt)
    prediction = dict(result.prediction or {})
    if receipt.get("duration_ms") is not None:
        prediction["execution_duration_ms"] = receipt["duration_ms"]
        kind = getattr(result.execution_program, "kind", "")
        # A selected catalog's actual terminal is a modeled CPU-Engine
        # boundary, not an observed target duration.  Every executable
        # family, including matrix, exposes it once the caller asked for a
        # real submission; the timing-free work remains available above.
        if kind in {"paged_fia", "gdn", "rope", "matrix"} or (
                prediction.get("fidelity") != "timing_free_work"):
            prediction["duration_ms"] = receipt["duration_ms"]
            prediction["duration_basis"] = "caller_execution_terminal"
            if kind in {"paged_fia", "gdn", "rope", "matrix"}:
                prediction["fidelity"] = "caller_execution_terminal"
    prediction["execution"] = view
    program = dict(result.program or {})
    program["execution"] = view
    physical = dict(program.get("physical") or {})
    physical["execution"] = view
    program["physical"] = physical
    prediction["program"] = program
    occurrence = dict(result.occurrence or {})
    occurrence["execution"] = view
    priced = dict(occurrence.get("priced") or {})
    priced["execution"] = view
    occurrence["priced"] = priced
    binding = result.binding
    if binding is not None:
        binding = replace(
            binding, physical={**(binding.physical or {}),
                               "execution": view})
    return replace(
        result, binding=binding, prediction=prediction, program=program,
        occurrence=occurrence, execution_handle=receipt.get("handle"),
        execution_receipt=view)


def _selected_native_catalog(
        result: NativeAbiPrediction, *,
        output_grants: Mapping[str, Any] | None = None,
        extra_events: tuple[str, ...] = ()) -> ProgramCatalog:
    """Return the immutable selected catalog, compiling it only if absent."""
    if (result.execution_catalog is not None and not output_grants
            and not extra_events):
        return result.execution_catalog
    return native_catalog_for_result(
        result, extra_events=extra_events, output_grants=output_grants)


def _with_selected_catalog(result: NativeAbiPrediction) -> NativeAbiPrediction:
    if result.status != "predicted" or result.execution_catalog is not None:
        return result
    prediction_view = result.prediction or {}
    if (result.execution_program is None
            and prediction_view.get("duration_ms") is not None):
        # A family leaf that already carries its physical duration owns the
        # timing; there is no selected executable program to compile, so a
        # second catalog must not re-price the same call.  The row-norm
        # classes are the exception: their declared launch floors/prologues
        # are added by the catalog chain itself.
        try:
            from .native_catalog import _kernel_class as _classify
            _class = _classify(result)
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            _class = None
        if _class not in {"row_norm", "fused_row_norm"}:
            return result
    try:
        catalog = native_catalog_for_result(result)
    except ModelingError:
        return result
    prediction = dict(result.prediction or {})
    kernel_class = None
    try:
        from .native_catalog import _kernel_class as _classify
        kernel_class = _classify(result)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        kernel_class = None
    if (kernel_class in {"row_norm", "fused_row_norm"}
            and prediction.get("fidelity") != "timing_free_work"
            and prediction.get("duration_ms") is not None):
        # The selected catalog *is* the native model: its per-stage chain is
        # the duration every actual submission will reproduce.  Keeping a
        # different family duration beside it would price the same call twice.
        modeled = catalog_modeled_duration_ms(result)
        if modeled is not None:
            # The canonical projection stores the exact rational form, the same
            # convention the declared family cost uses.
            prediction["duration_ms"] = str(Fraction(modeled))
            prediction["duration_basis"] = "native_catalog_chain"
    return replace(result, prediction=prediction, execution_catalog=catalog)
