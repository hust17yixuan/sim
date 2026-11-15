"""Cube (MatMulV3 / QuantBatchMatmulV3) leaf prediction from declared shapes.

[Role] Price one declared matrix leaf: whole Cube tiles, K-wave padding, the
  L0C/Fixpipe store and the declared operand bytes, mapped onto the shared
  operator plan when one is selected.
[Invariants] No measured duration, no per-shape table, no fallback algorithm.
  A K tail is a whole extra K tile; a tile selection changes issued work.
[Timeouts & Locks] Pure arithmetic; no waits or locks.
[Side-Effect Policy] Read-only; returns a prediction record.
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
from .native_abi_programs import (  # noqa: E402
    MatrixExecutionProgram, _binding_from_leaf_dict, _generic_program, _selection_program_options, _source_establishment,
)
from .native_leaf_dispatch import (  # noqa: E402
    _invocation_replay,
)
from .native_vector_abi import (  # noqa: E402
    _full_typed,
)
def _matrix_leaf_result(leaf: dict, *, spec: NativeAbiCallSpec,
                        inv: NativeAbiInvocation,
                        environment: dict,
                        selection: Mapping[str, Any] | None = None
                        ) -> NativeAbiPrediction:
    status = str(leaf.get("status") or "no_match")
    if status != "predicted":
        return _result(
            status, spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=tuple(leaf.get("reasons") or ()),
            environment=environment)
    selected = dict(leaf.get("selected_call") or {})
    work = leaf.get("work")
    tensors = dict(leaf.get("tensors") or {})
    selected_source = (dict(selected["source"])
                       if isinstance(selected.get("source"), Mapping)
                       else None)
    request_source = (inv.version.get("source")
                      or inv.version.get("source_revision"))
    request_image = inv.version.get("image_id") or inv.version.get("image")
    source = ((selected_source if selected_source
               and selected_source.get("sha256")
               else (dict(spec.source) if spec.source else None))
              if (request_source or request_image) else None)
    geometry = leaf.get("geometry") or {}
    execution = []
    if isinstance(geometry, Mapping) and isinstance(
            geometry.get("m"), int) and isinstance(geometry.get("n"), int):
        execution = [int(geometry["m"]), int(geometry["n"])]
    typed = _full_typed(
        inv, spec, {"execution_shape": execution,
                    "source_shapes": {
                        name: list(tensor.shape)
                        for name, tensor in inv.tensor_map().items()
                        if tensor.present}},
        execution_view=execution,
        extra={"variant": selected.get("variant"),
               "source": selected_source or source})
    region = NativeAbiRegion(
        op_id="native:" + str(spec.entry), family="native",
        recipe="matrix", source_boundary="call",
        dtype=typed.get("dtype"), layout=typed.get("layout"),
        metadata=(("leaf", spec.leaf), ("execution_view", tuple(execution))))
    binding = bind_abi_call(
        region,
        api={"namespace": str(selected.get("namespace") or spec.namespace),
             "entry": str(selected.get("entry") or spec.entry),
             "aclnn_symbol": selected.get("aclnn_symbol") or spec.aclnn_symbol,
             "version": selected.get("api_version") or spec.api_version},
        source=source, typed=typed,
        kernels=tuple({
            **dict(item),
            "evidence": item.get("evidence") or (
                "source_declared" if isinstance(source, Mapping)
                and source.get("sha256") else "unresolved"),
        } for item in spec.kernels), members=(),
        owner="native:" + str(spec.entry),
        execution_core=spec.execution_core,
        unresolved=tuple(leaf.get("reasons") or spec.unresolved),
        empty_members=True)
    if binding is None:
        return _result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=("native_abi_binding_unavailable",),
            environment=environment)
    if work is not None:
        binding = replace(binding, physical=work)
    operator_plan = None
    execution_program = None
    options = _selection_program_options(selection)
    tile = tuple(int(item) for item in (
        options.get("tile") or (16, 32, 16)))
    slots = int(options.get("slots", 2))
    try:
        module = importlib.import_module(
            "mindie_sim.prediction.piecewise." + spec.leaf)
        operator_plan = module.operator_plan(leaf, tile=tile, slots=slots)
        if operator_plan is not None:
            execution_program = MatrixExecutionProgram(
                plan=operator_plan, leaf=leaf, inv=inv, spec=spec,
                selection=selection)
    except Exception:  # noqa: BLE001 - plan remains an explicit gap
        operator_plan = None
        execution_program = None
    prediction = {
        "basis": "native_leaf_timing_free_work",
        "duration_ms": None, "duration_us": None,
        "dominant_pipe": None, "tied_pipes": [],
        "work": work, "times_ms": {}, "stage_times_ms": [],
        "materialization": {"tensors": tensors},
        "derived": {"geometry": leaf.get("geometry")},
        "formula": None, "fidelity": "timing_free_work",
        "environment": environment,
        "non_claims": ["timing_free_work", "no_measured_duration",
                       "no_target_duration_injected"],
    }
    program = _generic_program(
        spec=spec, leaf_program={
            "mode": "matrix_abi", "entry": spec.entry,
            "geometry": leaf.get("geometry"),
            "regions": leaf.get("regions"),
            "tensors": tensors, "scale": leaf.get("scale"),
            "metadata": leaf.get("metadata"),
            "accumulator": leaf.get("accumulator"),
            "selected_call": selected,
            "work": work,
        }, prediction=prediction, binding=binding, inv=inv)
    prediction["program"] = program
    occurrence = {
        "schema": "mindie-sim/piecewise-native-occurrence/1",
        "status": "predicted" if status == "predicted" else status,
        "call_id": binding.call_id, "op_id": binding.op_id,
        "family": binding.family, "recipe": binding.recipe,
        "source_boundary": binding.source_boundary,
        "phase": str(inv.conditions.get("phase") or ""),
        "kind": "native", "mode": "native_call",
        "layer": inv.conditions.get("layer"),
        "semantic_refs": [], "members": [],
        "request_conditions": dict(inv.conditions),
        "repeat": 1, "index": 0, "rank": None, "charge": "call",
        "call_owner": binding.op_id, "timing_owner": dict(binding.timing_owner),
        "typed": selected, "priced": {"duration_ms": None,
                                      "dominant_pipe": None,
                                      "work": work,
                                      "charged": "once_at_timing_owner"},
        "signature_matches": True, "unresolved": list(binding.unresolved),
    }
    return _result(status, spec=spec, invocation=inv.to_dict(),
                   binding=binding, prediction=prediction, program=program,
                   occurrence=occurrence, operator_plan=operator_plan,
                   execution_program=execution_program,
                   replay={"schema": ABI_REPLAY_SCHEMA,
                           "api": {"namespace": spec.namespace,
                                   "entry": spec.entry,
                                   "aclnn_symbol": spec.aclnn_symbol},
                           "version": dict(inv.version),
                           "tensors": {name: item.to_dict()
                                       for name, item in inv.tensor_map().items()},
                           "attrs": dict(inv.attrs),
                           "conditions": dict(inv.conditions),
                           "missing": [], "replayable": True},
                   reasons=tuple(leaf.get("reasons") or ()),
                   environment=environment, binding_bound=binding.bound,
                   source_establishment=_source_establishment(spec, inv))


def _leaf_simulate_result(leaf: dict, *, spec: NativeAbiCallSpec,
                          inv: NativeAbiInvocation,
                          environment: dict) -> NativeAbiPrediction:
    status = str(leaf.get("status") or "no_match")
    binding = _binding_from_leaf_dict(leaf.get("binding"))
    prediction = dict(leaf.get("prediction") or {})
    leaf_program = dict(leaf.get("program") or {})
    program = _generic_program(
        spec=spec, leaf_program=leaf_program, prediction=prediction,
        binding=binding, inv=inv)
    prediction["program"] = program
    occurrence = dict(leaf.get("occurrence") or {})
    if occurrence:
        occurrence["layer"] = inv.conditions.get("layer")
        occurrence["request_conditions"] = dict(inv.conditions)
        occurrence["priced"] = {
            **(occurrence.get("priced") or {}),
            "duration_ms": prediction.get("duration_ms"),
            "work": prediction.get("work")}
    return _result(
        status, spec=spec, invocation=leaf.get("invocation") or inv.to_dict(),
        binding=binding, prediction=prediction, program=program,
        occurrence=occurrence, replay=leaf.get("replay_minimal") or {},
        reasons=tuple(leaf.get("reasons") or ()),
        environment=environment or leaf.get("environment") or {},
        profiler_join=leaf.get("profiler_join"),
        binding_bound=binding.bound if binding is not None else False,
        source_establishment=_source_establishment(spec, inv))
