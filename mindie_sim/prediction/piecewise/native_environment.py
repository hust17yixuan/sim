"""Declared prediction environment, implementation policy and variant pick.

[Role] Resolve the target die's pipe supply, core count, dtype and phase, and
  select the single declared implementation variant for those conditions.
[Invariants] The environment is the declared hardware/profile only; a missing
  die raises instead of substituting another SKU.
[Timeouts & Locks] Pure resolution; no waits or locks.
[Side-Effect Policy] Read-only.
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
def _environment(hardware: Any, supply: Any) -> tuple[dict, PipeSupply, str | None]:
    from mindie_sim.schema.hardware import Hardware
    if supply is not None:
        if not isinstance(supply, PipeSupply):
            raise ModelingError("native_abi_supply_invalid")
        arch = supply.arch_family or None
        return ({"basis": "explicit_pipe_supply", "arch": arch,
                 "dtype": "bf16", "rates": supply.to_dict()},
                supply, arch)
    if hardware is None:
        hardware, basis = "Ascend950DT_9582", "default_a5_sku"
    elif isinstance(hardware, str):
        basis = "declared_sku"
    elif isinstance(hardware, Hardware):
        basis = "hardware_object"
    else:
        raise ModelingError("native_abi_hardware_invalid")
    if isinstance(hardware, str):
        import importlib
        ascend = importlib.import_module("mindie_sim.backends.ascend")
        machine = Hardware.from_mapping(
            ascend.materialize({"sku": hardware}))
        sku = hardware
    else:
        machine, sku = hardware, None
    supply_obj = PipeSupply.from_hardware(machine, dtype="bf16")
    arch = machine.arch_family or None
    return ({
        "basis": basis, "hardware": sku or arch, "arch": arch,
        "dtype": "bf16",
        "compute_units": (int(machine.dies[0].compute_units)
                          if machine.dies else None),
    }, supply_obj, arch)


def _implementation_policy(arch: str | None, cores: int | None,
                           mt: int | None) -> tuple[ImplementationPolicy, int]:
    base = ImplementationPolicy()
    cores = int(base.cores if cores is None else cores)
    if cores < 1:
        raise ModelingError("native_abi_cores_invalid")
    mt = int(base.mt if mt is None else mt)
    if mt < 1:
        raise ModelingError("native_abi_mt_invalid")
    return ImplementationPolicy(cores=cores, mt=mt), cores


def _select_variant(spec: NativeAbiCallSpec, conditions: dict) -> Any:
    name = str(conditions.get("variant") or "")
    for variant in spec.variants:
        if variant.name == name:
            return variant
    raise NativeAbiMismatch(f"native_abi_variant_not_declared:{name}")
