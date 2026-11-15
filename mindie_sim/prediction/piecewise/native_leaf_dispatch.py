"""Dispatch one normalized native call to its declared leaf mapper.

[Role] Route matrix / paged-FIA / vector / deferred leaves to their mappers
  and build the typed occurrence view the caller sees.
[Invariants] An unknown or unimplemented leaf stays an explicit gap; no
  algorithm or envelope fallback.  The occurrence view is derived from the
  declared binding only.
[Timeouts & Locks] Pure dispatch; no waits or locks.
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
from .native_abi_programs import (  # noqa: E402
    GDNExecutionProgram, RoPEExecutionProgram,
)
from .native_environment import (  # noqa: E402
    _environment,
)
from .native_matrix_abi_predict import (  # noqa: E402
    _leaf_simulate_result, _matrix_leaf_result,
)
from .native_paged_fia_abi import (  # noqa: E402
    _paged_fia_result,
)
from .native_vector_abi import (  # noqa: E402
    _dispatch_vector_leaf,
)
def _deferred_leaf_execution(inv: NativeAbiInvocation,
                            spec: NativeAbiCallSpec, *, hardware: Any,
                            supply: Any, policy: Any, phase: str | None,
                            selection: Mapping[str, Any] | None = None):
    """Prepare GDN/RoPE's original program without advancing the caller."""
    from . import native_gdn as gdn
    if spec.leaf == "native_gdn":
        environment, supply_obj, arch, machine = gdn._leaf_environment(
            hardware, supply)
        policy_obj = gdn._leaf_policy(policy, arch, None, None)
        plan = gdn._gdn_abi_program(
            inv, supply_obj, policy_obj, str(phase or ""), machine,
            None, None)
        if plan is None:
            return None
        return GDNExecutionProgram(
            call=plan["call"], program=plan["program"], inv=inv, spec=spec,
            selection=selection)
    if spec.leaf == "native_r64_triton_rope_siso":
        from . import native_r64_triton_rope_siso as rope
        environment, supply_obj, arch, machine = gdn._leaf_environment(
            hardware, supply)
        policy_obj = gdn._leaf_policy(policy, arch, None, None)
        plan = rope._abi_program(
            inv, supply_obj, policy_obj, str(phase or ""), None, None, None)
        if plan is None:
            return None
        return RoPEExecutionProgram(
            cost=plan["cost"], compiled=plan["compiled"],
            geometry=plan["geometry"], supply=supply_obj, inv=inv,
            spec=spec, selection=selection, program=plan.get("program"))
    return None


def _dispatch_leaf_abi(inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                       request: Any, *, hardware: Any, supply: Any,
                       policy: Any, cores: int | None, mt: int | None,
                       phase: str | None, profiler_rows: Any,
                       strict: bool,
                       selection: Mapping[str, Any] | None = None,
                       execution: Any = None, replica: int = 0,
                       stage: int = 0,
                       prepare_execution: bool = False
                       ) -> NativeAbiPrediction:
    if spec.dispatch == "paged_fia_build":
        environment = {"basis": "ordinary_paged_fia_native_plan",
                       "leaf": spec.leaf, "phase": phase}
        try:
            base_environment, supply_obj, _arch = _environment(
                hardware, supply)
            environment = {
                **base_environment, **environment,
                "pipe_supply": {"rates": list(supply_obj.rates)},
            }
        except ModelingError:
            pass
        return _paged_fia_result(
            inv, spec, environment, hardware=hardware,
            execution=execution, replica=replica, stage=stage,
            selection=selection, strict=strict)
    module = importlib.import_module(
        "mindie_sim.prediction.piecewise." + spec.leaf)
    environment = {"basis": "native_leaf_timing_free" if spec.dispatch in {
                       "matrix_predict_abi", "generic_matrix_abi"}
                   else "native_leaf_independent",
                   "leaf": spec.leaf, "phase": phase}
    if spec.dispatch in {"matrix_predict_abi", "generic_matrix_abi"}:
        # The physical catalog converts the declared Cube work through the
        # target SKU's actual matrix rate.  Without this supply, timing-free
        # work would be reported with no physical service basis.
        try:
            base_environment, supply_obj, _arch = _environment(
                hardware, supply)
            environment = {
                **base_environment, **environment,
                "pipe_supply": {"rates": list(supply_obj.rates)},
            }
        except ModelingError:
            pass
    try:
        if spec.dispatch == "vector_prepare_predict":
            result = _dispatch_vector_leaf(
                inv, spec, hardware=hardware, supply=supply,
                policy=policy, cores=cores, mt=mt, phase=phase,
                strict=False)
        elif spec.dispatch == "generic_matrix_abi":
            from .native_binding import predict_bf16_matmul
            leaf = predict_bf16_matmul(
                {tensor.name: tensor.to_dict() for tensor in inv.tensors
                 if tensor.present},
                attrs=dict(inv.attrs), version=dict(inv.version),
                conditions=dict(inv.conditions))
            result = _matrix_leaf_result(
                leaf, spec=spec, inv=inv, environment=environment,
                selection=selection)
        elif spec.dispatch == "matrix_predict_abi":
            normalized_tensors = {
                tensor.name: tensor.to_dict() for tensor in inv.tensors
                if tensor.present
            }
            # The shared torch QBMM entry has one weight position.  A real
            # FRACTAL_NZ weight in that position is the leaf's ``weight_nz``
            # branch, selected by the captured rank/layout, never by a shape
            # table or an operator label.
            weight = normalized_tensors.get("weight")
            if (isinstance(weight, Mapping)
                    and str(weight.get("layout") or "").upper() == "FRACTAL_NZ"
                    and len(weight.get("shape") or ()) == 4):
                normalized_tensors["weight_nz"] = normalized_tensors.pop(
                    "weight")
            leaf = module.predict_abi(
                normalized_tensors, attrs=dict(inv.attrs),
                version=dict(inv.version),
                conditions=dict(inv.conditions))
            result = _matrix_leaf_result(
                leaf, spec=spec, inv=inv, environment=environment,
                selection=selection)
        elif spec.dispatch == "leaf_simulate_independent":
            leaf = module.simulate_independent(
                request, hardware=hardware, supply=supply, policy=policy,
                cores=cores, mt=mt, phase=phase, profiler_rows=profiler_rows)
            result = _leaf_simulate_result(
                leaf, spec=spec, inv=inv, environment=environment)
            if prepare_execution and spec.leaf in {
                    "native_gdn", "native_r64_triton_rope_siso"}:
                program = _deferred_leaf_execution(
                    inv, spec, hardware=hardware, supply=supply,
                    policy=policy, phase=phase, selection=selection)
                if program is not None:
                    result = replace(result, execution_program=program)
        else:
            raise ModelingError(
                f"native_abi_dispatch_unknown:{spec.dispatch}")
    except (ModelingError, ValueError) as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"native_leaf_dispatch_failed:"
                     f"{getattr(error, 'reason', error)}",),
            environment=environment), strict)
    return _strict(result, strict)


def _adapt_prediction(record: dict, environment: dict,
                      program: dict) -> dict:
    """Project the VEC leaf's physical prediction without re-pricing it."""
    return {
        "basis": "shared_native_add_rms_norm_physical_call",
        "call": record.get("call"),
        "duration_ms": record["duration_ms"],
        "duration_us": record["duration_us"],
        "dominant_pipe": record["dominant_pipe"],
        "tied_pipes": list(record.get("tied_pipes") or ()),
        "work": dict(record["work"]),
        "times_ms": dict(record["times_ms"]),
        "stage_times_ms": list(record["stage_times_ms"]),
        "stage_work": [
            {str(pipe): str(value) for pipe, value in
             (stage.items() if hasattr(stage, "items") else stage)}
            for stage in record.get("stage_work") or ()],
        "materialization": dict(record["materialization"]),
        "derived": dict(record["derived"]),
        "formula": record["formula"],
        "fidelity": record["fidelity"],
        "program": program,
        "environment": environment,
        "non_claims": [
            "static_cpu_prediction", "no_measured_duration_claim",
            "no_target_duration_injected",
        ],
    }


def _binding_typed(region: dict) -> dict:
    return {
        "dtype": region["dtype"],
        "layout": region["layout"],
        "shapes": {name: list(shape)
                   for name, shape in region["binding_shapes"].items()},
        "attrs": dict(region["binding_attrs"]),
        "rank_width": 2,
    }


def _occurrence_typed(region: dict, inv: NativeAbiInvocation,
                      phase: str) -> dict:
    rows = int(region["rows"])
    hidden = int(region["hidden"])
    sequence_rows = str(inv.conditions.get("sequence_rows") or "")
    sequence_ranks = int(inv.conditions.get("sequence_ranks") or 1)
    attrs = {
        **region["binding_attrs"],
        "rows": rows, "tokens": rows, "phase": phase,
        "rstd_bytes": rows * 4,
        "residual_bytes": rows * hidden * 2,
        "norm_output_bytes": rows * hidden * 2,
        "physical_rows": rows,
        "sp_rows": sequence_rows or "all",
        "sp_ranks": sequence_ranks,
        "row_basis": sequence_rows or "all",
    }
    if region.get("epsilon") is not None:
        attrs["eps"] = region["epsilon"]
        attrs["eps_source"] = "native_abi_declared"
    return {
        "dtype": region["dtype"], "layout": region["layout"],
        "shapes": {name: list(shape)
                   for name, shape in region["occurrence_shapes"].items()},
        "attrs": attrs,
    }


def _occurrence(binding: Any, region: dict, inv: NativeAbiInvocation,
                variant: Any, phase: str, prediction: dict) -> dict:
    typed = validate_native_occurrence(
        binding, _occurrence_typed(region, inv, phase))
    unresolved = list(binding.unresolved)
    if "execution_rank_not_selected" not in unresolved:
        unresolved.append("execution_rank_not_selected")
    bare = variant.name == "native_call"
    return {
        "schema": "mindie-sim/piecewise-native-occurrence/1",
        "status": "bound" if binding.bound else "candidate",
        "call_id": binding.call_id,
        "op_id": binding.op_id,
        "family": binding.family,
        "recipe": binding.recipe,
        "source_boundary": binding.source_boundary,
        "phase": phase,
        "kind": ("native" if bare
                 else (variant.op_id.split(":", 1)[0]
                       if variant.op_id else "")),
        "mode": variant.name,
        "layer": inv.conditions.get("layer"),
        "semantic_refs": [],
        "members": list(variant.members),
        "repeat": 1,
        "index": -1,
        "rank": None,
        "charge": "call",
        "call_owner": variant.owner,
        "timing_owner": dict(binding.timing_owner),
        "typed": typed,
        "priced": {
            "duration_ms": prediction["duration_ms"],
            "dominant_pipe": prediction["dominant_pipe"],
            "work": dict(prediction["work"]),
            "charged": "once_at_timing_owner",
        },
        "signature_matches": True,
        "unresolved": unresolved,
    }

