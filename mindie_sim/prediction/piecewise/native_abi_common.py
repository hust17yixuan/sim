"""Shared immutable records, errors and hardware carriers for the native ABI.

[Role] Hold the normalized-invocation records, the prediction record, the
  declared-error hierarchy and the hardware/receipt helpers every native ABI
  module shares.  Keeping them in one place removes the historical 4k-line
  monolith without duplicating the record schemas.
[Invariants] Records are immutable value objects.  No module here reads a
  measurement, a profiler row or a sealed observation; the hardware carrier is
  built from the declared profile only.
[Timeouts & Locks] Pure construction; no waits or locks.
[Side-Effect Policy] No I/O, no device work, no global mutation.
"""

from __future__ import annotations

ABI_REQUEST_SCHEMA = "mindie-sim/native-abi-request/1"
ABI_REPLAY_SCHEMA = "mindie-sim/native-abi-replay/1"
ABI_RESULT_SCHEMA = "mindie-sim/native-abi-prediction/1"

_DTYPE_ALIASES = {
    "bfloat16": "bf16", "float32": "fp32", "float16": "fp16", "float64": "fp64",
    "int8": "int8", "int32": "int32", "int64": "int64", "bool": "bool",
    "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2", "uint8": "uint8",
    "uint64": "uint64", "dt_uint64": "uint64", "uint32": "uint32",
    "dt_uint32": "uint32", "uint1": "uint1", "dt_uint1": "uint1",
    "fp8_e8m0": "fp8_e8m0", "dt_float8_e8m0": "fp8_e8m0",
    "dt_float8_e4m3fn": "fp8_e4m3", "int4": "int4", "dt_int4": "int4",
    "uint4": "uint4", "dt_uint4": "uint4", "fp4_e2m1": "fp4_e2m1",
    "dt_float4_e2m1": "fp4_e2m1", "fp4": "fp4_e2m1", "mxfp4": "fp4_e2m1",
    "e2m1": "fp4_e2m1",
}
_ALLOWED_REQUEST = {"schema", "api", "version", "tensors", "attrs", "conditions",
                    "missing", "reason", "replayable", "abi_hash",
                    "declared_entries"}
_ALLOWED_API = {"namespace", "entry", "symbol", "aclnn_symbol", "version",
                "api_version", "candidates"}
_ALLOWED_VERSION = {"api", "api_version", "aclnn", "cann", "source",
                    "source_revision", "revision", "image_id", "image",
                    # Software facts a timing-free capture may declare next to
                    # the CANN version.  They identify the runtime, never a
                    # duration or a schedule.
                    "torch", "torch_npu", "python", "driver", "firmware",
                    "vllm", "container_image"}
_ALLOWED_TENSOR = {"name", "shape", "dtype", "layout", "role",
                   "optional", "present"}
_ALLOWED_CONDITIONS = {"variant", "owner", "members", "layer",
                       "sequence_rows", "sequence_ranks", "phase"}


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

def _plain(value: Any) -> Any:
    """JSON-safe projection for live execution receipts/programs."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)

class NativeAbiIncomplete(ModelingError):
    """A complete replayable invocation is missing required typed fields."""

    def __init__(self, reason: str, missing=()) -> None:
        super().__init__(reason)
        self.missing = tuple(dict.fromkeys(str(item) for item in missing))


class NativeAbiMismatch(ModelingError):
    """The request names a different API/ABI/version than the declared call."""


class NativeAbiVersionMismatch(NativeAbiMismatch):
    """The API/source/image version is outside the declared native condition."""


@dataclass(frozen=True)
class NativeAbiTensor:
    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    optional: bool = False
    present: bool = True
    fields: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        row = {key: value for key, value in self.fields}
        row.update({
            "name": self.name, "role": self.role,
            "shape": list(self.shape) if self.present else None,
            "dtype": self.dtype, "layout": self.layout,
            "optional": self.optional, "present": self.present})
        return row


@dataclass(frozen=True)
class NativeAbiInvocation:
    """One normalized actual native invocation, before any program pricing."""

    api: dict
    version: dict
    tensors: tuple[NativeAbiTensor, ...]
    attrs: dict
    conditions: dict
    abi_hash: str

    def tensor_map(self) -> dict[str, NativeAbiTensor]:
        return {tensor.name: tensor for tensor in self.tensors}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mindie-sim/native-abi-invocation/1",
            "api": dict(self.api), "version": dict(self.version),
            "tensors": {tensor.name: tensor.to_dict() for tensor in self.tensors},
            "attrs": dict(self.attrs), "conditions": dict(self.conditions),
            "abi_hash": self.abi_hash,
        }


@dataclass(frozen=True)
class NativeAbiPrediction:
    """Result of the independent native call: predicted or an explicit gap."""

    status: str
    spec: dict | None
    invocation: dict | None
    binding: Any
    prediction: dict | None
    program: dict | None
    occurrence: dict | None
    replay: dict
    reasons: tuple[str, ...]
    environment: dict
    profiler_join: dict | None = None
    binding_bound: bool = False
    source_establishment: dict | None = None
    prepared: Any = None
    execution_program: Any = None
    #: Immutable selected PlanCatalog compiled before any runtime exists.
    execution_catalog: ProgramCatalog | None = None
    operator_plan: Any = None
    #: Live caller-Execution handle for a non-blocking submission.  This is
    #: process-local and deliberately not serialized.
    execution_handle: Any = None
    #: Serializable receipt (event names/times, resource delta, work).
    execution_receipt: dict | None = None
    non_claims: tuple[str, ...] = (
        "static_cpu_prediction", "no_hardware_dispatch_claim",
        "no_measured_duration_claim", "no_target_duration_injected",
        "no_algorithm_fallback")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ABI_RESULT_SCHEMA,
            "status": self.status,
            "spec": self.spec,
            "invocation": self.invocation,
            "binding": self.binding.to_dict() if self.binding is not None else None,
            "prediction": self.prediction,
            "program": self.program,
            "occurrence": self.occurrence,
            "profiler_join": self.profiler_join,
            "replay_minimal": self.replay,
            "reasons": list(self.reasons),
            "environment": self.environment,
            "native_call_succeeded": self.status == "predicted",
            "binding_bound": (self.binding.bound
                              if self.binding is not None
                              else self.binding_bound),
            "source_establishment": self.source_establishment,
            "execution_program_available": self.execution_program is not None,
            "execution_catalog_available": self.execution_catalog is not None,
            "execution_catalog_id": (self.execution_catalog.catalog_id
                                     if self.execution_catalog is not None
                                     else None),
            "operator_plan_available": self.operator_plan is not None,
            "execution_program": (self.execution_program.to_dict()
                                  if self.execution_program is not None
                                  and hasattr(self.execution_program, "to_dict")
                                  else None),
            "execution_handle_available": self.execution_handle is not None,
            "execution_receipt": self.execution_receipt,
            "fallback_performed": False,
            "non_claims": list(self.non_claims),
        }

def _canonical_dtype(value: Any) -> str:
    text = str(value or "").strip().lower()
    return _DTYPE_ALIASES.get(text, text)


def _digest_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text

def _invocation_ports(inv: NativeAbiInvocation) -> tuple:
    return tuple({
        "name": tensor.name, "role": tensor.role,
        "dtype": tensor.dtype, "layout": tensor.layout,
        "shape": list(tensor.shape), "present": bool(tensor.present),
    } for tensor in inv.tensors if tensor.present)


def _execution_machine(hardware: Any):
    """Resolve the actual hardware carrier without creating a model/case."""
    from mindie_sim.schema.hardware import Hardware
    if isinstance(hardware, Hardware):
        return hardware
    sku = hardware if isinstance(hardware, str) else "Ascend950DT_9582"
    import importlib
    ascend = importlib.import_module("mindie_sim.backends.ascend")
    return Hardware.from_mapping(ascend.materialize({"sku": sku}))


def _hardware_execution(hardware: Any, *, supply: Any = None):
    """Build the bounded hardware-only Execution used by an independent call."""
    from types import SimpleNamespace
    from mindie_sim.schema.case import Software
    from ..cost import CostModel
    from ..execution import Execution
    machine = _execution_machine(hardware)
    context = SimpleNamespace(machine=machine, runtime=Software())
    return Execution(context, CostModel(context), include_submission=False)


def _execution_resource_bindings(execution: Any, replica: int,
                                 stage: int) -> dict[str, str]:
    """Map declared pipe roles to this Execution stage's physical names.

    Every ``<pipe>@r<replica>:s<stage>`` resource binds its own pipe role, so
    a declared stage term for ``mte1``/``fixpipe``/``vector``/``pipe_s_issue``
    resolves without a per-pipe table.  ``memory`` also answers ``hbm`` and
    either MTE2 carrier answers the combined ``mte2`` role.
    """
    suffix = f"@r{replica}:s{stage}"
    bindings: dict[str, str] = {}
    for key in execution.engine.resources:
        if not key.endswith(suffix):
            continue
        base = key.split("@", 1)[0]
        bindings[base] = key
        if base in {"mte2_aic", "mte2_aiv", "mte2"}:
            bindings["mte2"] = key
        elif base == "memory":
            bindings["hbm"] = key
    return bindings


def _native_cube_rate(result: NativeAbiPrediction) -> float | None:
    environment = dict(result.environment or {})
    supply = dict(environment.get("pipe_supply") or {})
    rates = supply.get("rates")
    if rates is None:
        rates = environment.get("rates")
    if isinstance(rates, Mapping):
        rates = rates.items()
    for name, rate in rates or ():
        if str(name) == "cube":
            value = float(rate)
            return value if value > 0 else None
    return None

def _native_pool(grant: Mapping[str, Any], rank: int) -> PoolKey:
    return PoolKey(
        physical_domain=str(grant.get("domain") or "device"), rank=rank,
        tier=str(grant.get("tier") or "hbm"),
        name=str(grant.get("pool")), unit=0, quantity_unit="bytes")

def _stage_context(execution: Any, op: Any, plan_work: Any, *,
                   replica: int = 0, stage: int = 0):
    """Bind the selected plan's work to the caller's actual stage context."""
    key = (replica, stage)
    base = execution.stage_contexts.get(key)
    if base is None:
        base = execution.stage_contexts[(replica, 0)]
    demand = base.bind(plan_work)
    return replace(base, op=op, work=demand, native_work={},
                   setup_ms=0, minimum_ms=0)

def _result(status: str, *, spec: NativeAbiCallSpec | None = None,
            invocation: dict | None = None, binding: Any = None,
            prediction: dict | None = None, program: dict | None = None,
            occurrence: dict | None = None, replay: dict | None = None,
            reasons=(), environment: dict | None = None,
            profiler_join: dict | None = None,
            binding_bound: bool | None = None,
            source_establishment: dict | None = None,
            prepared: Any = None, execution_program: Any = None,
            operator_plan: Any = None) -> NativeAbiPrediction:
    return NativeAbiPrediction(
        status=status,
        spec=spec.to_dict() if spec is not None else None,
        invocation=invocation, binding=binding, prediction=prediction,
        program=program, occurrence=occurrence, replay=replay or {},
        reasons=tuple(dict.fromkeys(str(item) for item in reasons)),
        environment=environment or {},
        profiler_join=profiler_join,
        binding_bound=(binding.bound if binding is not None
                       else bool(binding_bound)),
        source_establishment=source_establishment,
        prepared=prepared, execution_program=execution_program,
        operator_plan=operator_plan)


def _strict(result: NativeAbiPrediction, strict: bool) -> NativeAbiPrediction:
    if strict and result.status != "predicted":
        raise ModelingError(
            "native_abi_unavailable:" + result.status + ":"
            + ";".join(result.reasons or (result.replay.get("reason", ""),)))
    return result


def _full_typed(inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                geometry: dict, *, execution_view: list[int] | None = None,
                occurrence_attrs: Mapping[str, Any] | None = None,
                extra: Mapping[str, Any] | None = None) -> dict:
    """Complete typed ABI used by BIND1 and every JSON/view projection."""
    present = [tensor for tensor in inv.tensors if tensor.present]
    tensors = {tensor.name: tensor.to_dict() for tensor in present}
    shapes = {tensor.name: list(tensor.shape) for tensor in present}
    dtypes = {tensor.name: tensor.dtype for tensor in present}
    layouts = {tensor.name: tensor.layout for tensor in present}
    presence = {tensor.name: bool(tensor.present) for tensor in inv.tensors}
    primary = next((tensor for tensor in present
                    if tensor.role in {"input", "inout", "weight"}), None)
    if primary is None and present:
        primary = present[0]
    payload = {
        "entry": spec.entry,
        "dtype": primary.dtype if primary is not None else None,
        "layout": primary.layout if primary is not None else None,
        "shapes": shapes,
        "attrs": dict(inv.attrs),
        "rank_width": max((len(shape) for shape in shapes.values()),
                          default=0),
        "tensors": tensors,
        "dtypes": dtypes,
        "layouts": layouts,
        "presence": presence,
        "execution_view": list(execution_view or geometry.get(
            "execution_shape") or ()),
        "source_shapes": dict(geometry.get("source_shapes") or {}),
    }
    if occurrence_attrs:
        payload["occurrence_attrs"] = dict(occurrence_attrs)
    if extra:
        payload.update({key: value for key, value in extra.items()
                        if value is not None})
    logical_value_bytes = {}
    from mindie_sim.schema.common import dtype_bytes as _dtype_bytes
    for name, shape in shapes.items():
        dtype = dtypes.get(name)
        try:
            logical_value_bytes[name] = int(
                prod(shape)) * int(_dtype_bytes(str(dtype)))
        except (KeyError, TypeError, ValueError):
            continue
    payload["logical_value_bytes"] = logical_value_bytes
    payload["signature_hash"] = sha256(json.dumps(
        payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return payload


def _invocation_replay(inv: NativeAbiInvocation,
                       spec: NativeAbiCallSpec) -> dict:
    """Replay one complete normalized invocation without rewriting it."""
    return {
        "schema": ABI_REPLAY_SCHEMA,
        "api": dict(inv.api),
        "version": dict(inv.version),
        "tensors": {tensor.name: tensor.to_dict()
                    for tensor in inv.tensors},
        "attrs": dict(inv.attrs),
        "conditions": dict(inv.conditions),
        "missing": [],
        "replayable": True,
        "reason": "complete_invocation",
        "abi_hash": inv.abi_hash,
    }
