"""Vector (row elementwise / norm) ABI geometry, preparation and dispatch.

[Role] Validate the declared row geometry, dtype and same-shape contract of a
  vector leaf, prepare the CPU leaf contract and price the fused or standalone
  kernel through the shared family program.
[Invariants] Shapes and dtypes come from the declaration, never from a model
  name.  No tensor is allocated and no device runtime is imported.
[Timeouts & Locks] Pure arithmetic; no waits or locks.
[Side-Effect Policy] Read-only; the leaf contract is returned, not executed.
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
    _generic_program, _source_establishment,
)
from .native_environment import (  # noqa: E402
    _environment, _implementation_policy,
)
from .native_leaf_dispatch import (  # noqa: E402
    _invocation_replay,
)
def _vector_logical(inv: NativeAbiInvocation, name: str
                    ) -> tuple[Any, list[int]]:
    tensor = inv.tensor_map().get(name)
    if tensor is None or not tensor.present:
        raise NativeAbiMismatch(f"native_vector_tensor_missing:{name}")
    fields = dict(tensor.fields)
    views = fields.get("views") if isinstance(fields.get("views"), Mapping) else {}
    logical = (views.get("logical") if isinstance(views, Mapping) else None)
    shape = (logical.get("shape") if isinstance(logical, Mapping) else None)
    return tensor, list(shape or tensor.shape)


def _vec_shape(inv: NativeAbiInvocation, name: str) -> list[int] | None:
    tensor = inv.tensor_map().get(name)
    return list(tensor.shape) if tensor is not None and tensor.present else None


def _vec_dtype(inv: NativeAbiInvocation, name: str, default: str = "bf16") -> str:
    tensor = inv.tensor_map().get(name)
    return str(tensor.dtype if tensor is not None and tensor.dtype
               else default)


def _vector_same(name: str, actual: list[int], expected: list[int]) -> None:
    if list(actual) != list(expected):
        raise NativeAbiMismatch(
            f"native_vector_shape_mismatch:{name}:"
            f"{actual}!={expected}")


def _vector_dtype_same(name: str, actual: str, expected: str) -> None:
    if str(actual) != str(expected):
        raise NativeAbiMismatch(
            f"native_vector_dtype_mismatch:{name}:{actual}!={expected}")


def _vector_abi_geometry(inv: NativeAbiInvocation,
                         spec: NativeAbiCallSpec) -> dict:
    """Validate the full typed value contract and derive one execution view."""
    leaf = spec.leaf
    tensors = {tensor.name: tensor for tensor in inv.tensors}
    logical = {name: _vector_logical(inv, name)[1]
               for name, tensor in tensors.items() if tensor.present}
    dtypes = {name: str(tensor.dtype)
              for name, tensor in tensors.items() if tensor.present}
    layouts = {name: str(tensor.layout)
               for name, tensor in tensors.items() if tensor.present}
    source_shapes = {name: list(tensor.shape)
                     for name, tensor in tensors.items() if tensor.present}

    def require(name: str) -> list[int]:
        if name not in logical:
            raise NativeAbiMismatch(f"native_vector_tensor_missing:{name}")
        return logical[name]

    def require_dtype(name: str, expected: str) -> str:
        actual = dtypes.get(name, "")
        _vector_dtype_same(name, actual, expected)
        return actual

    def flatten(shape: list[int]) -> tuple[int, int]:
        if not shape:
            raise NativeAbiMismatch("native_vector_shape_empty")
        rows = prod(shape[:-1]) if len(shape) > 1 else 1
        return int(rows), int(shape[-1])

    value_map = None
    if leaf == "native_r31_sigmoid":
        source = require("input")
        output = require("output")
        _vector_same("output", output, source)
        require_dtype("output", dtypes["input"])
        rows, width = flatten(source)
        execution = [rows, width]
    elif leaf == "native_swiglu":
        source = require("input")
        output = require("output")
        rank = len(source)
        dim = int(inv.attrs.get("dim", -1))
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise NativeAbiMismatch("native_vector_swiglu_dim_invalid")
        if source[dim] % 2:
            raise NativeAbiMismatch("native_vector_swiglu_axis_not_even")
        expected = list(source)
        expected[dim] //= 2
        _vector_same("output", output, expected)
        require_dtype("output", dtypes["input"])
        other = prod([size for index, size in enumerate(output)
                      if index != dim]) or 1
        if dim == 0:
            rows, width = int(output[0]), int(other)
            execution_layout = "split_axis0_concat_last"
        else:
            rows, width = int(other), int(output[dim])
            execution_layout = "split_axis_as_last"
        value_map_layout = execution_layout
        # The selected leaf performs the half split on its last axis.  Move
        # the API's split axis to that execution axis without changing values.
        execution = [int(rows), int(2 * width)]
        value_map = {
            "split_axis": dim,
            "execution_layout": value_map_layout,
            "source_shape": list(source),
            "output_shape": list(output),
            "execution_input_shape": execution,
            "execution_output_shape": [int(rows), width],
            "gate": "source[..., :source.shape[dim]//2]",
            "up": "source[..., source.shape[dim]//2:]",
            "output": "silu(gate) * up",
        }
    elif leaf == "native_rms_norm":
        source = require("x")
        output = require("output")
        if prod(output) != prod(source):
            raise NativeAbiMismatch("native_vector_rmsnorm_output_shape")
        if require("weight") != [source[-1]]:
            raise NativeAbiMismatch("native_vector_rmsnorm_weight_shape")
        rows, width = flatten(source)
        rstd = logical.get("rstd")
        if rstd is not None:
            if rstd not in ([rows, 1], list(source[:-1]) + [1]):
                raise NativeAbiMismatch("native_vector_rmsnorm_rstd_shape")
            require_dtype("rstd", "fp32")
        require_dtype("output", dtypes["x"])
        execution = [rows, width]
        value_map = {
            "source_shape": list(source),
            "execution_input_shape": [int(rows), int(width)],
            "execution_output_shape": [int(rows), int(width)],
            "original_rstd_shape": list(rstd or [int(rows), 1]),
            "execution_rstd_shape": [int(rows), 1],
        }
    elif leaf == "native_r31_dynamic_quant_v2":
        source = require("x")
        output = require("y")
        _vector_same("y", output, source)
        rows, width = flatten(source)
        for name in ("scale", "offset"):
            if not logical.get(name):
                # A symmetric quantizer does not materialize the zero point.
                continue
            shape = require(name)
            if shape not in ([rows], [rows, 1]):
                raise NativeAbiMismatch(
                    f"native_vector_dynamic_quant_scale_shape:{name}")
            require_dtype(name, "fp32")
        require_dtype("y", "int8")
        execution = [rows, width]
    elif leaf == "native_elementwise":
        names = [name for name in ("x1", "x2", "input")
                 if logical.get(name)]
        if not names:
            raise NativeAbiMismatch("native_elementwise_input_missing")
        source = require(names[0])
        output = require("output")
        if prod(source) != prod(output):
            raise NativeAbiMismatch("native_elementwise_element_count")
        for name in names[1:]:
            other = require(name)
            if other != source and other != [source[-1]]:
                raise NativeAbiMismatch(
                    f"native_elementwise_shape_mismatch:{name}")
        rows = prod(output[:-1]) if len(output) > 1 else 1
        width = int(output[-1])
        execution = [int(rows), int(width)]
    elif leaf == "native_mx_quant":
        source = require("x")
        output = require("y")
        _vector_same("y", output, source)
        scale = require("scale")
        rows, width = flatten(source)
        expected = [int(rows), int(math.ceil(width / 64)), 2]
        if scale != expected:
            raise NativeAbiMismatch("native_mx_quant_scale_shape")
        require_dtype("scale", "fp8_e8m0")
        execution = [int(rows), int(width)]
    elif leaf == "native_add_rms_norm":
        source = require("x")
        _vector_same("residual", require("residual"), source)
        _vector_same("output", require("output"), source)
        if require("gamma") != [source[-1]]:
            raise NativeAbiMismatch("native_vector_add_rms_gamma_shape")
        rows, width = flatten(source)
        _vector_same("rstd", require("rstd"), [rows, 1])
        require_dtype("rstd", "fp32")
        execution = [rows, width]
    elif leaf == "native_gated_layer_norm_npu":
        source = require("x")
        output = require("output")
        gate = require("gate")
        _vector_same("output", output, source)
        _vector_same("gate", gate, source)
        if require("weight") != [source[-1]]:
            raise NativeAbiMismatch("native_vector_gated_weight_shape")
        require_dtype("output", dtypes["x"])
        require_dtype("gate", dtypes["x"])
        rows, width = flatten(source)
        head_dim = int(inv.attrs.get("head_dim", width))
        if head_dim < 1 or width % head_dim:
            raise NativeAbiMismatch("native_vector_gated_head_dim_mismatch")
        execution = [rows, width]
    elif leaf == "native_r31_aclnn_index":
        hidden = require("hidden_states")
        indices = require("logits_indices")
        output = require("sample_hidden_states")
        if len(hidden) != 2 or len(indices) != 1:
            raise NativeAbiMismatch("native_vector_index_rank_mismatch")
        selected, width = int(indices[0]), int(hidden[-1])
        rows = selected
        _vector_same("sample_hidden_states", output, [selected, width])
        require_dtype("logits_indices", "int64")
        require_dtype("sample_hidden_states", dtypes["hidden_states"])
        execution = [selected, width]
    elif leaf == "native_r31_inplace_copy":
        source = require("source")
        output = require("output")
        if len(source) != 3 or len(output) != 3:
            raise NativeAbiMismatch("native_vector_inplace_copy_rank_mismatch")
        tokens, source_heads, head_dim = (int(source[0]), int(source[1]),
                                          int(source[2]))
        if (int(output[0]) != tokens or int(output[2]) != head_dim
                or int(output[1]) > source_heads):
            raise NativeAbiMismatch("native_vector_inplace_copy_shape_mismatch")
        require_dtype("output", dtypes["source"])
        rows = tokens
        width = int(output[1]) * int(output[2])
        execution = list(output)
    elif leaf == "native_r31_inplace_zero":
        source = require("self")
        output = require("output")
        _vector_same("output", output, source)
        require_dtype("output", dtypes["self"])
        rows = int(source[0])
        width = int(source[1]) * int(source[2])
        execution = list(output)
    elif leaf == "native_r64_add_rms_norm_bias":
        source = require("x1")
        x2 = require("x2")
        output = require("y")
        _vector_same("x2", x2, source)
        _vector_same("y", output, source)
        if require("gamma") != [source[-1]]:
            raise NativeAbiMismatch("native_vector_add_rms_bias_gamma_shape")
        rows, width = flatten(source)
        _vector_same("rstd", require("rstd"), [rows, 1])
        if "beta" in logical and logical["beta"] != [width]:
            raise NativeAbiMismatch("native_vector_add_rms_bias_beta_shape")
        require_dtype("x2", dtypes["x1"])
        require_dtype("y", dtypes["x1"])
        if "beta" in dtypes:
            require_dtype("beta", dtypes["x1"])
        require_dtype("rstd", "fp32")
        execution = [rows, width]
    elif leaf == "native_a5_generic":
        # The captured op_summary row declares ordered slots, not semantic
        # names.  The widest row-major present tensor gives the AIV execution
        # view; every present slot keeps its own declared stream.
        present = [name for name, tensor in tensors.items() if tensor.present]
        if not present:
            raise NativeAbiMismatch("native_vector_generic_tensors_missing")
        widest = max(present, key=lambda name: (
            len(source_shapes[name]), prod(source_shapes[name]),
            source_shapes[name][-1] if source_shapes[name] else 0))
        rows, width = flatten(logical[widest])
        execution = [rows, width]
    else:
        raise NativeAbiMismatch(f"native_vector_leaf_adapter_missing:{leaf}")
    return {
        "leaf": leaf,
        "rows": rows,
        "width": width,
        "execution_shape": execution,
        "source_shapes": source_shapes,
        "dtypes": dtypes,
        "layouts": layouts,
        "presence": {name: bool(tensors[name].present) for name in tensors},
        "attrs": dict(inv.attrs),
        "value_map": value_map,
        "source_axis": (value_map or {}).get("split_axis"),
    }


def _vector_prepare(inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                    geometry: dict, *, typed: dict, call_id: str,
                    phase: str | None) -> Any:
    module = importlib.import_module(
        "mindie_sim.prediction.piecewise." + spec.leaf)
    attrs = dict(inv.attrs)
    runtime_phase = str(phase or inv.conditions.get("phase") or "prefill")
    leaf = spec.leaf
    rows = int(geometry["rows"])
    width = int(geometry["width"])
    if leaf == "native_rms_norm":
        prepared = module.prepare_rms_norm(
            rows=rows, hidden=width,
            epsilon=float(attrs.get("epsilon", 1e-6)),
            weight_mode=str(attrs.get("weight_mode") or "shifted"),
            dtype=_vec_dtype(inv, "x"), phase=runtime_phase, call_id=call_id)
    elif leaf == "native_swiglu":
        prepared = module.prepare_swiglu(
            rows=rows, elements=width, dtype=_vec_dtype(inv, "input"),
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_r31_sigmoid":
        prepared = module.prepare_sigmoid(
            rows=rows, width=width, dtype=_vec_dtype(inv, "input"),
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_elementwise":
        names = [name for name in ("x1", "x2", "input")
                 if name in geometry["source_shapes"]]
        first = names[0]
        name = (spec.entry[5:] if str(spec.entry).startswith("aclnn")
                else str(spec.entry).split(".")[-1])
        source_shape = geometry["source_shapes"][first]
        broadcast = any(
            len(geometry["source_shapes"][other]) == 1
            and len(source_shape) > 1 for other in names[1:])
        prepared = module.prepare_elementwise(
            name=name, input_shape=source_shape,
            output_shape=geometry["source_shapes"]["output"],
            dtype=_vec_dtype(inv, first),
            output_dtype=_vec_dtype(inv, "output"),
            read_tensors=len(names), write_tensors=1, broadcast=broadcast,
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_mx_quant":
        source_shape = geometry["source_shapes"]["x"]
        prepared = module.prepare_mx_quant(
            input_shape=source_shape,
            output_shape=geometry["source_shapes"]["y"],
            scale_shape=geometry["source_shapes"]["scale"],
            dtype=_vec_dtype(inv, "x"),
            output_dtype=_vec_dtype(inv, "y", "fp8_e4m3"),
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_add_rms_norm":
        prepared = module.prepare_add_rms_norm(
            rows=int(rows), hidden=int(width),
            epsilon=float(attrs.get("epsilon", 1e-6)),
            variant=str(attrs.get("variant")
                        or inv.conditions.get("variant")
                        or "fused_attention_norm"),
            dtype=_vec_dtype(inv, "x"), phase=runtime_phase,
            call_id=call_id)
    elif leaf == "native_r31_dynamic_quant_v2":
        symmetrical = attrs.get("is_symmetrical")
        if symmetrical is None and "is_symmetrical" not in attrs:
            symmetrical = not bool(
                geometry.get("presence", {}).get("offset"))
        prepared = module.prepare_dynamic_quant_v2(
            rows=rows, width=width,
            is_symmetrical=symmetrical,
            source_dtype=_vec_dtype(inv, "x"),
            output_dtype=_vec_dtype(inv, "y", "int8"),
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_gated_layer_norm_npu":
        prepared = module.prepare_gated_norm(
            rows=rows,
            head_dim=int(attrs.get("head_dim", width)),
            epsilon=float(attrs.get("epsilon", 1e-6)),
            dtype=_vec_dtype(inv, "x"),
            norm_before_gate=attrs.get("norm_before_gate"),
            gate_present=True, phase=runtime_phase, call_id=call_id)
    elif leaf == "native_r31_aclnn_index":
        hidden = geometry["source_shapes"]["hidden_states"]
        selected = geometry["source_shapes"]["logits_indices"]
        prepared = module.prepare_index(
            producing_rows=int(hidden[0]), selected_rows=int(selected[0]),
            hidden=int(hidden[-1]), dtype=_vec_dtype(inv, "hidden_states"),
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_r31_inplace_copy":
        source = geometry["source_shapes"]["source"]
        output = geometry["source_shapes"]["output"]
        prepared = module.prepare_inplace_copy(
            tokens=int(source[0]), source_heads=int(source[1]),
            kept_heads=int(output[1]), head_dim=int(source[-1]),
            slice_offset_heads=int(attrs.get("slice_offset_heads", 0)),
            dtype=_vec_dtype(inv, "source"), phase=runtime_phase,
            call_id=call_id)
    elif leaf == "native_r31_inplace_zero":
        source = geometry["source_shapes"]["self"]
        prepared = module.prepare_inplace_zero(
            tokens=int(source[0]), kept_heads=int(source[1]),
            head_dim=int(source[-1]), dtype=_vec_dtype(inv, "self"),
            phase=runtime_phase, call_id=call_id)
    elif leaf == "native_r64_add_rms_norm_bias":
        prepared = module.prepare_add_rms_norm_bias(
            rows=rows, hidden=width,
            epsilon=float(attrs.get("epsilon", 1e-6)),
            beta_present=("beta" in geometry["presence"]
                          and geometry["presence"]["beta"]),
            dtype=_vec_dtype(inv, "x1"),
            scope=str(attrs.get("scope") or "n0"), phase=runtime_phase,
            call_id=call_id)
    elif leaf == "native_a5_generic":
        kernel = str((spec.kernels[0] or {}).get(
            "name") if spec.kernels else "") or str(spec.entry)
        kind = "elementwise" if kernel in {
            "Slice", "ScatterPaKvCache"} else "vector"
        prepared = module.prepare_generic(
            tensors=typed["tensors"], kernel=kernel, kind=kind,
            phase=runtime_phase, call_id=call_id,
            conditions=dict(inv.conditions))
    else:
        raise NativeAbiMismatch(
            f"native_vector_leaf_adapter_missing:{leaf}")
    prepared["request_tensors"] = typed["tensors"]
    prepared["request_shapes"] = typed["shapes"]
    prepared["request_attrs"] = typed["attrs"]
    prepared["request_conditions"] = dict(inv.conditions)
    prepared["source_shapes"] = geometry["source_shapes"]
    prepared["execution_view"] = geometry["execution_shape"]
    prepared["value_map"] = geometry.get("value_map")
    prepared["source_axis"] = geometry.get("source_axis")
    prepared["typed"] = typed
    return prepared


def _vector_leaf_result(record: dict, *, spec: NativeAbiCallSpec,
                        inv: NativeAbiInvocation, environment: dict,
                        binding: Any, typed: dict,
                        geometry: dict) -> NativeAbiPrediction:
    module = importlib.import_module(
        "mindie_sim.prediction.piecewise." + spec.leaf)
    predicted = module.predict(
        record, hardware=environment.get("_hardware"),
        supply=environment.get("_supply"), policy=environment.get("_policy"),
        phase=environment.get("phase"))
    prediction = dict(predicted)
    prediction["source_shapes"] = dict(geometry["source_shapes"])
    prediction["execution_view"] = list(geometry["execution_shape"])
    prediction["request_tensors"] = dict(typed["tensors"])
    prediction["request_conditions"] = dict(inv.conditions)
    prediction["value_map"] = geometry.get("value_map")
    prediction["source_axis"] = geometry.get("source_axis")
    prediction["logical_value_bytes"] = dict(typed.get(
        "logical_value_bytes") or {})
    if prediction.get("work") is not None:
        binding = replace(binding, physical=prediction.get("work"))
    program = _generic_program(
        spec=spec, leaf_program=prediction.get("program"),
        prediction=prediction, binding=binding, inv=inv)
    program["physical"]["request_tensors"] = dict(typed["tensors"])
    program["physical"]["source_shapes"] = dict(geometry["source_shapes"])
    program["physical"]["execution_view"] = list(geometry["execution_shape"])
    program["physical"]["request_conditions"] = dict(inv.conditions)
    program["physical"]["value_map"] = geometry.get("value_map")
    program["physical"]["source_axis"] = geometry.get("source_axis")
    program["physical"]["logical_value_bytes"] = dict(typed.get(
        "logical_value_bytes") or {})
    program["cross_view"]["request_conditions"] = dict(inv.conditions)
    program["cross_view"]["logical_value_bytes"] = dict(typed.get(
        "logical_value_bytes") or {})
    program["cross_view"]["value_map"] = geometry.get("value_map")
    prediction["program"] = program
    try:
        execution_program = module.execution_program(
            record, hardware=environment.get("_hardware"),
            supply=environment.get("_supply"),
            policy=environment.get("_policy"),
            phase=environment.get("phase"), name=spec.leaf)
    except Exception:  # noqa: BLE001 - execution handoff remains explicit
        execution_program = None
    members = list(binding.member_op_ids)
    status = str(prediction.get("status") or "predicted")
    layer = inv.conditions.get("layer")
    if status != "predicted":
        occurrence = {
            "schema": "mindie-sim/piecewise-native-occurrence/1",
            "status": status, "call_id": binding.call_id,
            "op_id": binding.op_id, "family": binding.family,
            "recipe": binding.recipe,
            "source_boundary": binding.source_boundary,
            "phase": str(environment.get("phase") or ""), "kind": "native",
            "mode": "native_call", "layer": layer, "semantic_refs": [],
            "members": members, "repeat": 1, "index": 0, "rank": None,
            "charge": "call", "call_owner": binding.op_id,
            "timing_owner": dict(binding.timing_owner), "typed": typed,
            "request_conditions": dict(inv.conditions),
            "priced": {"duration_ms": None, "dominant_pipe": None,
                       "work": None, "charged": "not_priced"},
            "signature_matches": True, "unresolved": list(binding.unresolved),
        }
        return _result(
            status, spec=spec, invocation=inv.to_dict(), binding=binding,
            prediction=prediction, program=program, occurrence=occurrence,
            replay=_invocation_replay(inv, spec),
            reasons=tuple(prediction.get("reasons") or ()),
            environment={key: value for key, value in environment.items()
                         if not key.startswith("_")},
            binding_bound=False,
            source_establishment=_source_establishment(spec, inv),
            prepared=record, execution_program=execution_program)
    occurrence = {
        "schema": "mindie-sim/piecewise-native-occurrence/1",
        "status": "predicted", "call_id": binding.call_id,
        "op_id": binding.op_id, "family": binding.family,
        "recipe": binding.recipe, "source_boundary": binding.source_boundary,
        "phase": str(environment.get("phase") or ""), "kind": "native",
        "mode": "native_call", "layer": layer, "semantic_refs": [],
        "members": members, "repeat": 1, "index": 0, "rank": None,
        "charge": "call", "call_owner": binding.op_id,
        "timing_owner": dict(binding.timing_owner), "typed": typed,
        "request_conditions": dict(inv.conditions),
        "priced": {"duration_ms": prediction.get("duration_ms"),
                   "dominant_pipe": prediction.get("dominant_pipe"),
                   "work": prediction.get("work"),
                   "charged": "once_at_timing_owner"},
        "signature_matches": True, "unresolved": list(binding.unresolved),
    }
    return _result(
        "predicted", spec=spec, invocation=inv.to_dict(), binding=binding,
        prediction=prediction, program=program, occurrence=occurrence,
        replay=_invocation_replay(inv, spec), reasons=(),
        environment={key: value for key, value in environment.items()
                     if not key.startswith("_")},
        binding_bound=binding.bound,
        source_establishment=_source_establishment(spec, inv),
        prepared=record, execution_program=execution_program)


def _dispatch_vector_leaf(inv: NativeAbiInvocation,
                          spec: NativeAbiCallSpec, *, hardware: Any,
                          supply: Any, policy: Any, cores: int | None,
                          mt: int | None, phase: str | None,
                          strict: bool) -> NativeAbiPrediction:
    try:
        geometry = _vector_abi_geometry(inv, spec)
        typed = _full_typed(inv, spec, geometry)
        caller_members = tuple(inv.conditions.get("members") or ())
        caller_owner = inv.conditions.get("owner")
        source_state = _source_establishment(spec, inv)
        region = NativeAbiRegion(
            op_id="native:" + str(spec.entry), family="native",
            recipe="vector", source_boundary="call",
            dtype=typed.get("dtype"), layout=typed.get("layout"),
            metadata=(("leaf", spec.leaf),
                      ("execution_view", tuple(
                          geometry["execution_shape"]))))
        binding = bind_abi_call(
            region, api={"namespace": inv.api["namespace"],
                         "entry": inv.api["entry"],
                         "aclnn_symbol": inv.api.get("aclnn_symbol"),
                         "version": inv.version.get("api") or spec.api_version},
            source=source_state["binding_source"], typed=typed,
            kernels=spec.kernels, members=caller_members,
            owner=caller_owner or ("native:" + str(spec.entry)),
            execution_core=spec.execution_core,
            unresolved=spec.unresolved,
            empty_members=not caller_members)
        if binding is None:
            return _result(
                "no_match", spec=spec, invocation=inv.to_dict(),
                replay=_invocation_replay(inv, spec),
                reasons=("native_abi_binding_unavailable",),
                environment={"basis": "native_leaf_prepare_predict"})
        if policy is not None:
            policy_obj = policy
        elif cores is not None or mt is not None:
            policy_obj = _implementation_policy(None, cores, mt)[0]
        else:
            policy_obj = None
        environment = {"basis": "native_leaf_prepare_predict",
                       "leaf": spec.leaf, "phase": phase,
                       "policy": asdict(policy_obj) if policy_obj else None,
                       "_hardware": hardware, "_supply": supply,
                       "_policy": policy_obj}
        # Carry the declared SKU/arch into the catalog so the selected stage
        # chain is priced on the same die the leaf was evaluated on (A3/A2
        # must not fall back to the A5 default inside the catalog builder).
        try:
            base_env, resolved_supply, resolved_arch = _environment(
                hardware, supply)
            environment.update({
                "hardware": base_env.get("hardware"),
                "arch": resolved_arch,
                "rates": base_env.get("rates")})
            if environment.get("_supply") is None:
                environment["_supply"] = resolved_supply
        except ModelingError:
            pass
        prepared = _vector_prepare(
            inv, spec, geometry, typed=typed, call_id=binding.call_id,
            phase=phase)
        return _strict(_vector_leaf_result(
            prepared, spec=spec, inv=inv, environment=environment,
            binding=binding, typed=typed, geometry=geometry), strict)
    except NativeAbiMismatch as error:
        return _strict(_result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(getattr(error, "reason", str(error)),),
            environment={"basis": "native_leaf_prepare_predict"}),
            strict)
    except (ModelingError, ValueError) as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"native_vector_leaf_failed:"
                     f"{getattr(error, 'reason', error)}",),
            environment={"basis": "native_leaf_prepare_predict"}), strict)
