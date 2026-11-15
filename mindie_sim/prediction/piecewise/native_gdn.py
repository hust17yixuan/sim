"""Prepare-only GDN invocation plus the shared state-leaf independent ABI path.

[Role] Expand one selected ``aclnnChunkGatedDeltaRuleFwd`` occurrence from the
  actual request geometry into the wrapper's named arguments and tensor
  descriptors, using :mod:`mindie_sim.prediction.call_gdn` as the only source
  of shapes, dtypes and varlen arrays.  The same module hosts the small
  independent-native-input harness shared by the attention/index/state leaves:
  API+tensor+attr+version are normalized once, mapped into the existing family
  program (``CompiledOperator``) and priced on CPU.  It is a leaf-level slice;
  the common ABI catalogue and public entry remain owned by T12.
[Invariants] The static binding identity stays untouched; token/batch lanes,
  ``cu_seqlens`` and ``chunk_indices`` exist only in the dynamic occurrence.
  No tensor is allocated, no runtime is imported and no duration is changed.
  Unsupported ragged geometry returns ``None`` so the caller can mark the
  occurrence unresolved instead of inventing an ABI.  An unknown or
  mismatched native ABI/attr/version returns an explicit non-predicted status;
  it never falls back to another algorithm, envelope or measured duration.
  State effects are declared as ordered slots with their read/write direction
  and tensor role, not folded into a duration.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure descriptors derived from the existing ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError


def _delta_theta(compiled: Any) -> Any:
    delta = next((operator for operator in getattr(compiled, "siblings", ())
                  if operator.name == "gated_delta"), None)
    return delta.theta if delta is not None else compiled.operator.theta


def gdn_dims(compiled: Any) -> tuple[int, int, int, int, int]:
    theta = _delta_theta(compiled)
    heads, extras = dict(theta.heads), dict(theta.extras)
    keys = int(heads.get("q", extras.get("key_heads", 1)) or 1)
    values = int(heads.get("kv", extras.get("value_heads", keys)) or keys)
    dk = int(heads.get("dn", extras.get("dk", 1)) or 1)
    dv = int(heads.get("dv", extras.get("dv", dk)) or dk)
    chunk = int(extras.get("chunk", 64) or 64)
    return keys, values, dk, dv, chunk


_DYNAMIC_CACHE: dict[tuple, dict] = {}


def dynamic_gdn(binding: Any, compiled: Any, geometry: Any, *,
                validate: Any) -> dict | None:
    """Full prepare-only invocation from the existing GDNCall device program."""
    if (binding.api or {}).get("aclnn_symbol") != "aclnnChunkGatedDeltaRuleFwd":
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    requests = tuple(getattr(geometry, "requests", ()) or ())
    if not requests:
        return None
    lengths = tuple(int(request.q_len) for request in requests)
    if not lengths or min(lengths) < 1:
        return None
    batch, tokens = len(lengths), sum(lengths)
    keys, values, dk, dv, chunk = gdn_dims(compiled)
    cache_key = (binding.call_id, batch, lengths)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selected = dict(binding.typed.get("attrs") or {})
    try:
        from mindie_sim.prediction import call_gdn
        call = call_gdn.make_call("scan_prefill", {
            "batch": batch, "tokens": tokens, "lengths": list(lengths),
            "v_heads": values,
            "k_heads": keys, "k_dim": dk, "v_dim": dv,
            "chunk": int(selected.get("chunk_size", chunk)),
            "chunk_size": int(selected.get("chunk_size", chunk)),
            "phase": "prefill",
            "scale": float(selected.get("scale", dk ** -0.5)),
            "a_log": None, "dt_bias": None, "input_dtype": "bf16",
            "gate_dtype": "fp32", "beta_dtype": "bf16", "state_dtype": "fp32",
            "layout": str(binding.typed.get("layout") or "BSND"),
            "use_exp2": bool(selected.get("use_exp2", True)),
            "use_qk_l2norm_in_kernel": bool(
                selected.get("use_qk_l2norm_in_kernel", True)),
            "allow_neg_eigval": bool(selected.get("allow_neg_eigval", False)),
            "disable_recompute": bool(selected.get("disable_recompute", True)),
            "state_v_first": bool(selected.get("state_v_first", True)),
            "output_final_state": bool(selected.get("output_final_state", True)),
            "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": bool(
                selected.get("use_beta_sigmoid_in_kernel", False)),
            "return_intermediate_states": False, "initial_state": True,
        })
    except (ImportError, ModelingError, TypeError, ValueError):
        return None
    program = call.device_program
    g_cumsum_materialization = next(
        (item for item in (program.get("materializations") or ())
         if item.get("source_output") == "g_cumsum"
         or item.get("recipe") == "g_cumsum_export"), None)
    source_layout = dict((g_cumsum_materialization or {}).get("source_layout") or {})
    target_layout = dict((g_cumsum_materialization or {}).get("target_layout") or {})
    third_transpose = None
    if source_layout and target_layout:
        third_transpose = {
            "component": "Transpose", "tensor": "g_cumsum",
            "source_shape": list(source_layout.get("shape") or ()),
            "source_strides": list(source_layout.get("strides") or ()),
            "source_storage_shape": list(source_layout.get("storage_shape") or ()),
            "target_shape": list(target_layout.get("shape") or ()),
            "target_strides": list(target_layout.get("strides") or ()),
            "target_storage_shape": list(target_layout.get("storage_shape") or ()),
            "evidence": "GDNCall.device_program.materializations:g_cumsum_export",
        }
    arguments = dict(program.get("arguments") or {})
    # Keep the prepare JSON projection stable for existing consumers while
    # F-STATE's call ledger may carry tuples internally.
    for key in ("cu_seqlens", "lengths"):
        if key in arguments and isinstance(arguments[key], tuple):
            arguments[key] = list(arguments[key])
    partition = {
        "physical_batch": 1,
        "sequence_count": batch,
        "lengths": list(lengths),
        "cu_seqlens": list(arguments.get("cu_seqlens") or ()),
        "chunks": int(arguments.get("chunks") or 0),
        "chunk_head_tasks": int(arguments.get("chunks") or 0) * values,
        "state_rows": batch,
        "partition_owner": "F-STATE.call_gdn",
    }
    chunk_meta = dict(arguments.get("chunk_indices") or {})
    chunk_values = list(chunk_meta.get("values") or ())
    if not chunk_values:
        return None
    arguments["chunk_indices"] = chunk_values
    arguments["chunk_indices_shape"] = [len(chunk_values)]
    inputs = dict(program.get("inputs") or {})
    outputs = dict(program.get("outputs") or {})
    exports = {name: value for name, value in (program.get("exports") or {}).items()
               if isinstance(value, dict)}
    shapes, descriptors = {}, {}
    for name, descriptor in {**inputs, **outputs, **exports}.items():
        shape = list(descriptor.get("shape") or ())
        if not shape:
            return None
        shapes[name] = shape
        descriptors[name] = {**descriptor, "role": (
            "input" if name in inputs else "output")}
    if "initial_state" in inputs:
        shapes["state"] = list(inputs["initial_state"]["shape"])
    shapes["cu_seqlens"] = [len(arguments.get("cu_seqlens") or ())]
    shapes["chunk_indices"] = [len(chunk_values)]
    attrs = dict(binding.typed.get("attrs") or {})
    attrs.update({
        "batch": batch, "tokens": tokens, "lengths": list(lengths),
        "cu_seqlens": list(arguments.get("cu_seqlens") or ()),
        "chunk_indices": chunk_values, "chunk_size": int(arguments["chunk_size"]),
        "layout": str(arguments["layout"]), "scale": float(arguments["scale"]),
        "chunks": int(arguments.get("chunks") or 0),
    })
    actual = {"dtype": binding.typed.get("dtype"),
              "layout": binding.typed.get("layout"),
              "shapes": shapes, "attrs": attrs}
    validated = validate(binding, actual)
    named = {
        "q": "q", "k": "k", "v": "v", "g": "g", "beta": "beta",
        "initial_state": "initial_state", "a_log": None, "dt_bias": None,
        "cu_seqlens": list(arguments.get("cu_seqlens") or ()),
        "chunk_indices": chunk_values,
        "layout": str(arguments["layout"]), "scale": float(arguments["scale"]),
        "chunk_size": int(arguments["chunk_size"]),
        "use_exp2": bool(attrs.get("use_exp2")),
        "use_qk_l2norm_in_kernel": bool(attrs.get("use_qk_l2norm_in_kernel")),
        "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": bool(attrs.get("use_beta_sigmoid_in_kernel")),
        "allow_neg_eigval": bool(attrs.get("allow_neg_eigval")),
        "disable_recompute": bool(attrs.get("disable_recompute")),
        "output_final_state": bool(attrs.get("output_final_state")),
        "return_intermediate_states": False,
        "state_v_first": bool(attrs.get("state_v_first")),
    }
    output_slots = {
        "o": {"acl_slot": "oOut", "status": "required"},
        "final_state": {"acl_slot": "finalStateOutOptional",
                        "status": "enabled" if attrs.get("output_final_state")
                        else "disabled"},
        "g_cumsum": {"acl_slot": "gCumsumOutOptional",
                     "status": "enabled" if attrs.get("disable_recompute")
                     else "disabled"},
        "A": {"acl_slot": "aOutOptional",
              "status": "enabled" if attrs.get("disable_recompute") else "disabled"},
        "beta_eff": {"acl_slot": "betaEffOutOptional",
                     "status": "enabled"
                     if attrs.get("use_beta_sigmoid_in_kernel") else "disabled",
                     "reason": "use_beta_sigmoid_in_kernel=false"},
        "h": {"acl_slot": "hOutOptional", "status": "disabled",
              "reason": "return_intermediate_states=false"},
        "q_hat": {"acl_slot": "qHatOutOptional",
                  "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                  else "disabled"},
        "k_hat": {"acl_slot": "kHatOutOptional",
                  "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                  else "disabled"},
        "q_rstd": {"acl_slot": "qRstdOutOptional",
                   "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                   else "disabled"},
        "k_rstd": {"acl_slot": "kRstdOutOptional",
                   "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                   else "disabled"},
    }
    enabled_outputs = {name: descriptors[name] for name, slot in output_slots.items()
                       if slot["status"] != "disabled" and name in descriptors}
    disabled_outputs = {name: dict(slot) for name, slot in output_slots.items()
                        if slot["status"] == "disabled"}
    prepare = {
        "schema": "mindie-sim/gdn-native-prepare/1",
        "call": program.get("call"), "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "wrapper": "fla_npu.ops.ascendc._stable.npu_chunk_gated_delta_rule_fwd",
        "wrapper_lines": [1472, 1516],
        "wrapper_source": program.get("wrapper_source"),
        "acl_signature": "aclnnChunkGatedDeltaRuleFwd",
        "acl_signature_source": {
            "archive": (program.get("source") or {}).get("archive"),
            "member": "ops/ascendc/_aclnn_ctypes.py",
            "sha256": "2aae5a0d7131f78f26a285f2b2cce2cbae3e2001fb20cf7e8fd373daf9e4cd8a",
            "lines": [150, 180],
        },
        "static_shapes": dict(binding.typed.get("shapes") or {}),
        "shapes": dict(validated.get("shapes") or {}),
        "arguments": arguments, "named_arguments": named,
        "chunk_indices_meta": chunk_meta,
        "arrays": descriptors, "descriptors": descriptors,
        "outputs": enabled_outputs, "output_slots": output_slots,
        "disabled_outputs": disabled_outputs,
        "physical_presentations_per_call": 7,
        "g_cumsum_materialization": g_cumsum_materialization,
        "physical_presentations": [
            {"component": "Cast", "role": "copy", "count": 1},
            {"component": "Transpose", "role": "copy", "count": 3,
             "third_transpose": third_transpose},
            {"component": "Prepare32", "role": "vector", "count": 1},
            {"component": "H24", "role": "cube", "count": 1},
            {"component": "O32", "role": "cube", "count": 1},
        ],
        "host_arrays": {
            "cu_seqlens": {"dtype": "int64",
                           "shape": [len(arguments.get("cu_seqlens") or ())]},
            "chunk_indices": {"dtype": "int64", "shape": [len(chunk_values)]},
        },
        "source": binding.source,
        "device_kernels": list(program.get("device_kernels") or ()),
        "allocation": "not_performed",
        "boundary": "prepare_only_host_descriptors",
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration"],
    }
    input_state = dict(inputs.get("initial_state") or {})
    final_state = dict(outputs.get("final_state") or {})
    chunk_order = STATE_EFFECT_ORDER["chunk_compute"]
    output_final = bool(selected.get("output_final_state", True))
    prepare["state_effects"] = [
        state_effect(
            order=chunk_order, direction="read", slot="gdn_chunk_state",
            tensor="initial_state", rows=batch,
            dtype=str(input_state.get("dtype") or ""),
            role="recurrent_initial_state_read",
            shape=list(input_state.get("shape") or ()),
            source="call_program.inputs.initial_state",
            visibility="caller_owned_until_call"),
    ]
    prepare["disabled_state_effects"] = []
    if output_final and final_state:
        prepare["state_effects"].append(state_effect(
            order=chunk_order, direction="write", slot="gdn_chunk_state",
            tensor="final_state", rows=batch,
            dtype=str(final_state.get("dtype") or ""),
            role="recurrent_final_state_write",
            shape=list(final_state.get("shape") or ()),
            requires_attr="output_final_state",
            requires_value=True,
            source="call_program.outputs.final_state",
            visibility="call_completion_then_cache_scatter"))
    else:
        prepare["disabled_state_effects"].append({
            "tensor": "final_state", "slot": "gdn_chunk_state",
            "direction": "write", "reason": "output_final_state_false",
            "condition": "output_final_state",
        })
    prepare["state_slot"] = "gdn_chunk_state"
    prepare["state_effect_sequence"] = {
        "slot": STATE_SLOT_SSM,
        "ordered": [
            {"order": STATE_EFFECT_ORDER["gather_read"],
             "owner": "gather_ssm_states", "direction": "read"},
            {"order": chunk_order, "owner": "chunk_gated_delta_rule_fwd",
             "direction": "read_write"},
            {"order": STATE_EFFECT_ORDER["scatter_write"],
             "owner": "scatter_ssm_states", "direction": "write"},
        ],
        "note": ("the numeric order is the source call order; it is never "
                 "added as elapsed time"),
    }
    prepare["implementation"] = {
        "wrapper": prepare["wrapper"],
        "wrapper_lines": list(prepare["wrapper_lines"]),
        "acl_signature": prepare["acl_signature"],
        "acl_signature_source": dict(prepare["acl_signature_source"]),
        "source_revision": str((binding.source or {}).get("revision") or ""),
        "source_sha256": str((binding.source or {}).get("sha256") or ""),
        "signature_hash": binding.typed.get("signature_hash"),
        "state": "selected_runtime_implementation",
    }
    prepare["binding_relation"] = {
        "semantic_region": "A3:recurrent:gated_delta",
        "relation": "one_to_many",
        "members": ["prepare", "gated_delta"],
        "sibling_calls": ["gather_ssm_states", "scatter_ssm_states"],
        "charge": "call",
        "members_are_not_separate_timing_owners": True,
    }
    prepare["partition"] = partition
    result = {"actual": validated, "prepare": prepare,
              "prepare_args": {"dtype": validated.get("dtype"),
                               "layout": validated.get("layout"),
                               "shapes": validated.get("shapes"),
                               "attrs": validated.get("attrs")}}
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result




# ---------------------------------------------------------------------------
# Shared independent native-input path for the attention/index/state leaves.
#
# The normal model path already binds a ``compiled`` operator and evaluates the
# family mechanism.  This slice lets the same leaf be exercised from a real
# standalone native input (API + tensor descriptors + attrs + explicit version)
# without a model case, weights or device initialization.  Each leaf supplies a
# ``NativeAbiCallSpec`` and a mapper to its existing family program; the
# harness owns normalization, explicit-status returns, binding identity and the
# state-effect view.  T12 owns the common ABI catalogue and public entry.
# ---------------------------------------------------------------------------

LEAF_ABI_SCHEMA = "mindie-sim/leaf-native-abi-prediction/2"
LEAF_REPLAY_SCHEMA = "mindie-sim/native-abi-replay/2"
# Ordered effect vocabulary shared by the GDN prefill state leaves.  The
# numbers are the real order of the source call sites (gather -> chunk ->
# scatter), not a timing weight; they are never added as milliseconds.
STATE_SLOT_SSM = "ssm_state_cache"
STATE_EFFECT_ORDER = {"gather_read": 1, "chunk_compute": 2, "scatter_write": 3}

LEAF_NON_CLAIMS = (
    "static_cpu_prediction",
    "no_hardware_dispatch_claim",
    "no_measured_duration_claim",
    "no_target_duration_injected",
    "no_algorithm_fallback",
)

_MISSING = object()

_LEAF_DTYPES = {
    "bfloat16": "bf16", "float32": "fp32", "float16": "fp16",
    "int8": "int8", "int32": "int32", "int64": "int64",
    "bool": "bool", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2",
    "uint8": "uint8",
}


class LeafAbiUnsupported(ModelingError):
    """The normalized input is a different/unsupported native invocation."""


class LeafAbiConstraint(LeafAbiUnsupported):
    """The input violates a known implementation limit (not a missing model)."""


def _leaf_canonical_dtype(value: Any) -> str:
    text = str(value or "").strip().lower()
    return _LEAF_DTYPES.get(text, text)


def _jsonable(value: Any) -> Any:
    """Copy a request for replay without dropping rejected fields."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _leaf_entry(request: Any) -> str | None:
    """Read one API entry from a string/mapping request without guessing."""
    if isinstance(request, str):
        return request
    if not isinstance(request, dict):
        return None
    api = request.get("api")
    if isinstance(api, str):
        return api
    if isinstance(api, dict):
        for key in ("entry", "symbol", "aclnn_symbol"):
            value = api.get(key)
            if isinstance(value, str) and value:
                return value
    return None


# ---------------------------------------------------------------------------
# Normalized descriptor consumers.  A mapper must read dtype/shape/layout from
# the normalized tensor records (the representation fact) and cross-check any
# duplicate attr instead of rebuilding the value from the attr alone.
# ---------------------------------------------------------------------------

def leaf_tensor(inv: Any, name: str) -> Any:
    tensor = inv.tensor_map().get(name)
    if tensor is None or not tensor.present:
        raise LeafAbiUnsupported(f"leaf_tensor_missing:{name}")
    return tensor


def leaf_shape(inv: Any, name: str) -> list[int]:
    return list(leaf_tensor(inv, name).shape)


def leaf_dtype(inv: Any, name: str) -> str:
    return str(leaf_tensor(inv, name).dtype)


def leaf_attr(inv: Any, name: str, default: Any = _MISSING) -> Any:
    if name in inv.attrs:
        return inv.attrs[name]
    if default is _MISSING:
        raise LeafAbiUnsupported(f"leaf_attr_missing:{name}")
    return default


def leaf_int(inv: Any, name: str, default: Any = _MISSING) -> int:
    value = leaf_attr(inv, name, default)
    if isinstance(value, bool):
        raise LeafAbiUnsupported(f"leaf_attr_bool_not_int:{name}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise LeafAbiUnsupported(f"leaf_attr_not_int:{name}") from None
    return number


def leaf_require_tensor_dtype(inv: Any, tensor_name: str, expected: str,
                              *, attr: str | None = None) -> str:
    actual = leaf_dtype(inv, tensor_name)
    if actual != expected:
        raise LeafAbiUnsupported(
            "leaf_tensor_attr_dtype_conflict:"
            f"{tensor_name}:{actual}!={expected}"
            + (f"@{attr}" if attr else ""))
    return actual


def leaf_require_attr_matches_tensor_dtype(inv: Any, attr_name: str,
                                           tensor_names: tuple[str, ...]) -> str:
    expected = _leaf_canonical_dtype(leaf_attr(inv, attr_name))
    for name in tensor_names:
        leaf_require_tensor_dtype(inv, name, expected, attr=attr_name)
    return expected


def leaf_require_alias(inv: Any, output_name: str, source_name: str) -> None:
    output, source = leaf_tensor(inv, output_name), leaf_tensor(inv, source_name)
    if tuple(output.shape) != tuple(source.shape):
        raise LeafAbiUnsupported(
            f"leaf_alias_shape_conflict:{output_name}!={source_name}")
    if output.dtype != source.dtype:
        raise LeafAbiUnsupported(
            f"leaf_alias_dtype_conflict:{output_name}:"
            f"{output.dtype}!={source.dtype}")


def leaf_launch_attr(inv: Any, name: str, expected: Any, *,
                    source: str = "") -> Any:
    value = leaf_attr(inv, name, expected)
    if value != expected:
        raise LeafAbiUnsupported(
            f"leaf_fixed_launch_attr_mismatch:{name}:{value}!={expected}"
            + (f"@{source}" if source else ""))
    return value


def leaf_stride(inv: Any, name: str, minimum: int) -> int | None:
    value = leaf_attr(inv, name, None)
    if value is None:
        return None
    if isinstance(value, bool):
        raise LeafAbiUnsupported(f"leaf_stride_not_int:{name}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise LeafAbiUnsupported(f"leaf_stride_not_int:{name}") from None
    if number < minimum:
        raise LeafAbiUnsupported(f"leaf_stride_too_small:{name}:{number}")
    return number


def _leaf_environment(hardware: Any, supply: Any) -> tuple[dict, Any, str | None, Any]:
    """Build the same CPU supply the ordinary family program consumes."""
    from mindie_sim.prediction.piecewise.compiler import PipeSupply
    import importlib
    from mindie_sim.schema.hardware import Hardware
    if supply is not None:
        if not isinstance(supply, PipeSupply):
            raise ModelingError("leaf_abi_supply_invalid")
        arch = supply.arch_family or None
        machine = None
        basis = "explicit_pipe_supply"
        if hardware is not None:
            if isinstance(hardware, Hardware):
                machine, basis = hardware, "hardware_plus_explicit_pipe_supply"
            elif isinstance(hardware, str):
                ascend = importlib.import_module("mindie_sim.backends.ascend")
                machine = Hardware.from_mapping(
                    ascend.materialize({"sku": hardware}))
                basis = "declared_sku_plus_explicit_pipe_supply"
            else:
                raise ModelingError("leaf_abi_hardware_invalid")
        return ({"basis": basis, "arch": arch,
                 "dtype": "bf16", "rates": supply.to_dict(),
                 "hardware_preserved": machine is not None},
                supply, arch, machine)
    if hardware is None:
        hardware, basis = "Ascend950DT_9582", "default_a5_sku"
    elif isinstance(hardware, str):
        basis = "declared_sku"
    elif isinstance(hardware, Hardware):
        basis = "hardware_object"
    else:
        raise ModelingError("leaf_abi_hardware_invalid")
    if isinstance(hardware, str):
        ascend = importlib.import_module("mindie_sim.backends.ascend")
        machine = Hardware.from_mapping(ascend.materialize({"sku": hardware}))
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
    }, supply_obj, arch, machine)


def _leaf_policy(policy: Any, arch: str | None, cores: Any, mt: Any):
    from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy
    base = ImplementationPolicy()
    cores = int(base.cores if cores is None else cores)
    mt = int(base.mt if mt is None else mt)
    if cores < 1:
        raise ModelingError("leaf_abi_cores_invalid")
    if mt < 1:
        raise ModelingError("leaf_abi_mt_invalid")
    if policy is not None:
        return policy
    if arch == "A5" and cores == int(base.cores):
        cores = 64
    return ImplementationPolicy(cores=cores, mt=mt)


def _leaf_declared_template(spec: Any, request: Any) -> dict:
    raw = request if isinstance(request, dict) else {}
    provided = raw.get("tensors") if isinstance(raw.get("tensors"), dict) else {}
    tensors = {}
    for tensor in spec.tensors:
        item = provided.get(tensor.name)
        item = item if isinstance(item, dict) else {}
        shape = item.get("shape")
        tensors[tensor.name] = {
            "role": item.get("role") or tensor.role,
            "dtype": item.get("dtype") or tensor.dtype,
            "shape": (list(shape) if isinstance(shape, (list, tuple)) and shape
                      else None),
            "layout": str(item.get("layout") or "ND"),
            "optional": bool(tensor.optional),
            "present": bool(item.get("present", True))
            if isinstance(item, dict) and item else False,
        }
    raw_attrs = raw.get("attrs") if isinstance(raw.get("attrs"), dict) else {}
    attrs = {name: raw_attrs.get(name)
             for name in (*spec.attrs, *spec.optional_attrs)
             if name in raw_attrs}
    conditions = (dict(raw.get("conditions"))
                  if isinstance(raw.get("conditions"), dict) else {})
    version = {"api": spec.api_version, "source": spec.source_revision,
               "image_id": spec.image_id}
    raw_version = raw.get("version")
    if isinstance(raw_version, dict):
        for key in ("api", "api_version", "source", "revision", "image_id"):
            if raw_version.get(key) is not None:
                target = ("api" if key in {"api", "api_version"}
                          else "source" if key in {"source", "revision"}
                          else "image_id")
                version[target] = raw_version[key]
    return {
        "schema": LEAF_REPLAY_SCHEMA,
        "api": {"namespace": spec.namespace, "entry": spec.entry,
                "aclnn_symbol": spec.aclnn_symbol},
        "version": {key: value for key, value in version.items()
                    if value is not None},
        "tensors": tensors, "attrs": attrs, "conditions": conditions,
    }


def _leaf_replay(spec: Any, request: Any, missing=(), *,
                 reason: str = "insufficient_fields",
                 replayable: bool | None = None,
                 conflicts=()) -> dict:
    """Replay record that preserves the rejected request verbatim.

    The declared template is a fill-in aid; it is not a second prediction
    request.  ``original_request`` keeps unknown/conflicting attrs and tensors
    so a replay cannot silently become a different successful call.
    """
    template = _leaf_declared_template(spec, request)
    raw = request if isinstance(request, dict) else {
        "api": request if isinstance(request, str) else None}
    raw_attrs = raw.get("attrs") if isinstance(raw.get("attrs"), dict) else {}
    raw_tensors = (raw.get("tensors")
                   if isinstance(raw.get("tensors"), dict) else {})
    declared_attrs = set(spec.attrs) | set(spec.optional_attrs)
    declared_tensors = {tensor.name for tensor in spec.tensors}
    rejected_attrs = sorted(set(raw_attrs) - declared_attrs)
    rejected_tensors = sorted(set(raw_tensors) - declared_tensors)
    if replayable is None:
        replayable = not (
            missing or conflicts or rejected_attrs or rejected_tensors)
    return {
        **template,
        "original_request": _jsonable(request),
        "declared_template": template,
        "missing": list(dict.fromkeys(str(item) for item in missing)),
        "rejected": {
            "attrs": rejected_attrs, "tensors": rejected_tensors,
            "conflicts": [str(item) for item in conflicts],
        },
        "replayable": bool(replayable),
        "reason": reason,
        "semantics": ("declared_template_is_not_a_prediction_request; "
                      "original_request_preserves_rejected_fields"),
    }


def _leaf_missing(spec: Any, request: Any) -> tuple[str, ...]:
    """Every typed field needed for a replay, not only the first gap."""
    raw = request if isinstance(request, dict) else {}
    provided = raw.get("tensors") if isinstance(raw.get("tensors"), dict) else {}
    missing: list[str] = []
    for tensor in spec.tensors:
        item = provided.get(tensor.name)
        if not isinstance(item, dict) or item.get("present", True) is False:
            if not tensor.optional:
                missing.append(f"tensors.{tensor.name}")
            continue
        shape = item.get("shape")
        if (not isinstance(shape, (list, tuple)) or not shape
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in shape)):
            missing.append(f"tensors.{tensor.name}.shape")
        if not _leaf_canonical_dtype(item.get("dtype") or tensor.dtype):
            missing.append(f"tensors.{tensor.name}.dtype")
    attrs = raw.get("attrs") if isinstance(raw.get("attrs"), dict) else {}
    for name in spec.attrs:
        if attrs.get(name) is None:
            missing.append(f"attrs.{name}")
    version = raw.get("version")
    version = version if isinstance(version, dict) else {}
    if spec.api_version and not (version.get("api") or version.get("api_version")):
        missing.append("version.api")
    if spec.source_revision and not (version.get("source")
                                     or version.get("revision")):
        missing.append("version.source")
    if spec.image_id and version.get("image_id") is None:
        missing.append("version.image_id")
    return tuple(dict.fromkeys(missing))


def _leaf_prediction(cost: Any, environment: dict, state_effects: Any,
                     *, occurrences: Any = (), parts: Any = ()) -> dict:
    payload = {
        "basis": "common_program_structural_work",
        "duration_ms": str(cost.duration_ms),
        "duration_us": float(cost.duration_ms) * 1000.0,
        "dominant_pipe": cost.dominant_pipe,
        "tied_pipes": list(cost.tied_pipes),
        "work": {key: str(value) for key, value in cost.work},
        "times_ms": {key: str(value) for key, value in cost.times},
        "stage_times_ms": [str(value) for value in cost.stage_times],
        "stage_work": [{pipe: str(value) for pipe, value in stage}
                       for stage in cost.stage_work],
        "materialization": {key: str(value)
                            for key, value in cost.materialization},
        "derived": {key: str(value) for key, value in cost.derived},
        "formula": cost.formula,
        "fidelity": cost.fidelity,
        "state_effects": list(state_effects),
        "environment": environment,
        "non_claims": list(LEAF_NON_CLAIMS),
    }
    payload["cost_record"] = {
        "name": cost.name,
        "duration_ms": str(cost.duration_ms),
        "dominant_pipe": cost.dominant_pipe,
        "tied_pipes": list(cost.tied_pipes),
        "work": {key: str(value) for key, value in cost.work},
        "times_ms": {key: str(value) for key, value in cost.times},
        "stage_times_ms": [str(value) for value in cost.stage_times],
        "stage_work": [{pipe: str(value) for pipe, value in stage}
                       for stage in cost.stage_work],
        "materialization": {key: str(value)
                            for key, value in cost.materialization},
        "derived": {key: str(value) for key, value in cost.derived},
        "formula": cost.formula,
    }
    if occurrences:
        payload["occurrences"] = list(occurrences)
    if parts:
        payload["parts"] = list(parts)
    return payload


LEAF_PREPARED_SCHEMA = "mindie-sim/leaf-prepared-program/1"
LEAF_EXECUTION_SCHEMA = "mindie-sim/leaf-native-execution/1"


def _leaf_event(value: Any) -> Any:
    """Actual Event carried by a handle/value, never a name-only projection."""
    return getattr(value, "event", value)


def _leaf_event_view(event: Any) -> dict:
    return {
        "name": str(getattr(event, "name", "")),
        "done": bool(getattr(event, "done", False)),
        "time": (None if getattr(event, "time", None) is None
                 else float(event.time)),
    }


def _leaf_work_map(pairs: Any) -> dict:
    """Sum one occurrence's resource work without dropping repeated pipes."""
    from fractions import Fraction
    result = {}
    for key, value in pairs or ():
        name = str(key)
        result[name] = result.get(name, Fraction(0)) + Fraction(str(value))
    return result


def _leaf_stage_pairs(stage: Any) -> tuple:
    """Normalize one declared/reported stage into ordered (pipe, work) pairs."""
    from fractions import Fraction
    if isinstance(stage, Mapping):
        if "resource" in stage:
            return ((str(stage.get("resource")),
                     Fraction(str(stage.get("serviced", 0)))),)
        return tuple((str(key), Fraction(str(value)))
                     for key, value in stage.items())
    if (isinstance(stage, (list, tuple)) and stage
            and isinstance(stage[0], Mapping) and "resource" in stage[0]):
        return tuple((str(item.get("resource")),
                      Fraction(str(item.get("serviced", 0))))
                     for item in stage)
    return tuple((str(key), Fraction(str(value))) for key, value in stage)


def _leaf_resource_bindings(execution: Any, replica: int = 0,
                            stage: int = 0) -> dict[str, str]:
    """Map declared pipes onto this caller stage's physical resources."""
    suffix = f"@r{replica}:s{stage}"
    bindings: dict[str, str] = {}
    for key in execution.engine.resources:
        if not key.endswith(suffix):
            continue
        base = key.split("@", 1)[0]
        if base in {"mte1", "mte3", "fixpipe", "vector", "host"}:
            bindings[base] = key
        elif base == "mte2_aiv":
            bindings["mte2"] = key
        elif base == "mte2_aic":
            bindings.setdefault("mte2", key)
        elif base == "memory":
            bindings["hbm"] = key
        elif base == "matrix":
            bindings["cube"] = key
            bindings["matrix"] = key
    return bindings


def _leaf_machine_with_supply(machine: Any, supply: Any) -> Any:
    """The actual machine view whose physical rates are this supply.

    ``Execution`` derives its resource rates from the machine carrier.  An
    explicit ``PipeSupply`` therefore has to be carried into that carrier;
    otherwise a low HBM supply would price the cost but execute at the
    hardware default.  Only declared supply pipes are substituted.
    """
    if machine is None or supply is None:
        return machine
    from dataclasses import replace
    rates = dict(getattr(supply, "rates", ()) or ())

    def target(role: str) -> str:
        return "hbm" if role == "memory" else role

    dies = []
    for die in machine.dies:
        adjusted = []
        for role, value in die.rates:
            supplied = rates.get(target(str(role)))
            adjusted.append((role, (float(supplied) * 1000.0
                                    if supplied is not None else value)))
        dies.append(replace(die, rates=tuple(adjusted)))
    return replace(machine, dies=tuple(dies))


def leaf_execution(hardware: Any = None, supply: Any = None, *,
                   trace_limit: Any = 0) -> tuple:
    """Create the single outermost Execution for an independent leaf call.

    The context carries hardware and runtime only: no model or workload is
    fabricated.  The same explicit supply that priced the program also sets
    the physical rates used by the caller's engine.
    """
    from types import SimpleNamespace
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.schema.case import Software
    environment, supply_obj, _arch, machine = _leaf_environment(
        hardware, supply)
    if machine is None:
        _env, _supply, _arch, machine = _leaf_environment(None, None)
    context = SimpleNamespace(
        machine=_leaf_machine_with_supply(machine, supply_obj),
        runtime=Software())
    execution = Execution(
        context, CostModel(context), trace_limit=int(trace_limit or 0),
        include_submission=False)
    return execution, environment, supply_obj


def _leaf_engine_state(execution: Any) -> dict:
    from fractions import Fraction
    engine = execution.engine
    return {
        "now": Fraction(str(engine.now)),
        "serviced": {str(key): Fraction(str(value))
                     for key, value in engine.serviced.items()},
        "busy_ms": {str(key): Fraction(str(value))
                    for key, value in engine.busy_ms.items()},
        "completed": int(engine.completed_count),
        "trace": int(len(engine.trace)),
        "trace_dropped": int(engine.trace_dropped),
    }


@dataclass(frozen=True)
class LeafExecutionHandle:
    """Actual writer/terminal/state handles returned by one submission.

    ``event`` is the occurrence terminal.  ``events`` retain every submitted
    stage/forward event.  ``output_events``/``writers`` name the values that
    can be connected downstream, ``state_events`` carry the ordered
    read/write event, and ``release_events`` are the grant/last-use releases.
    No advancement happens here; a standalone caller waits its own terminal
    outside, an embedded caller keeps the pending events.
    """

    event: Any = None
    events: tuple = ()
    start_event: Any = None
    output_events: tuple = ()
    state_events: tuple = ()
    release_events: tuple = ()
    work: tuple = ()
    stage_work: tuple = ()
    part_work: tuple = ()
    part_spans: tuple = ()
    typed_values: tuple = ()
    state_values: tuple = ()
    writers: tuple = ()
    reads: tuple = ()
    grants: tuple = ()
    owner: str = ""
    placement: tuple = (0, 0)
    execution: Any = None
    before: Any = None

    @property
    def done(self) -> bool:
        return bool(getattr(self.event, "done", False))

    @property
    def duration_ms(self) -> Any:
        from fractions import Fraction
        end = getattr(self.event, "time", None)
        start = getattr(self.start_event, "time", None)
        if end is None or start is None:
            return None
        return Fraction(str(end)) - Fraction(str(start))

    def reconcile(self, before: Any = None) -> dict:
        """Whole-engine delta from this submission; never occurrence work."""
        from fractions import Fraction
        snapshot = self.before if before is None else before
        if self.execution is None or snapshot is None:
            raise ModelingError("leaf_execution_reconciliation_unavailable")
        after = _leaf_engine_state(self.execution)
        serviced, busy = {}, {}
        for name, value in after["serviced"].items():
            delta = value - snapshot["serviced"].get(name, Fraction(0))
            if delta:
                serviced[str(name)] = str(delta)
        for name, value in after["busy_ms"].items():
            delta = value - snapshot["busy_ms"].get(name, Fraction(0))
            if delta:
                busy[str(name)] = str(delta)
        return {
            "basis": "whole_engine_serviced_delta",
            "serviced": serviced, "busy_ms": busy,
            "completed_tasks": int(after["completed"])
            - int(snapshot["completed"]),
            "used_for_occurrence_work": False,
        }

    def to_dict(self) -> dict:
        return {
            "schema": LEAF_EXECUTION_SCHEMA,
            "terminal": _leaf_event_view(self.event),
            "start": _leaf_event_view(self.start_event),
            "events": [_leaf_event_view(event) for event in self.events],
            "outputs": {name: _leaf_event_view(event)
                        for name, event in self.output_events},
            "state_events": [{
                **{key: value for key, value in dict(item).items()
                   if key != "event"},
                "event": _leaf_event_view(item.get("event")),
            } for item in self.state_events],
            "writers": [{
                **{key: value for key, value in dict(item).items()
                   if key != "event"},
                "event": _leaf_event_view(item.get("event")),
            } for item in self.writers],
            "reads": [{
                **{key: value for key, value in dict(item).items()
                   if key != "event"},
                "event": _leaf_event_view(item.get("event")),
            } for item in self.reads],
            "grants": [dict(item) for item in self.grants],
            "releases": [_leaf_event_view(event)
                         for event in self.release_events],
            "typed_values": _jsonable(self.typed_values),
            "state_values": _jsonable(self.state_values),
            "work": {str(key): str(value)
                     for key, value in _leaf_work_map(self.work).items()},
            "stage_work": [{str(key): str(value) for key, value in stage}
                          for stage in self.stage_work],
            "owner": self.owner,
            "placement": list(self.placement),
            "pending": not self.done,
            "non_claims": [
                "events_are_engine_terminal_not_device_measurement",
                "no_internal_advance",
            ],
        }


@dataclass
class LeafExecutionProgram:
    """One selected leaf program, prepared once and submitted by the caller.

    ``call`` retains GDN's original ``call.ops``/``call.recipes``.  Family
    leaves retain the same ``cost``/``compiled``/``geometry`` objects that
    were evaluated during prepare.  ``start`` binds them to an existing
    Execution; it never builds a second plan or a private engine.
    """

    name: str
    call: Any
    parts: tuple
    supply: Any
    policy: Any
    machine: Any
    ports: tuple = ()
    state_ports: tuple = ()
    state_value: Any = None
    program: Mapping[str, Any] = field(default_factory=dict)
    replica: int = 0
    stage: int = 0
    owner: str = ""
    prepared: bool = False

    def to_dict(self) -> dict:
        parts = []
        for part in self.parts:
            cost = part.get("cost")
            parts.append({
                "name": str(part.get("name") or ""),
                "kind": str(part.get("kind") or ""),
                "program_source": ("family" if part.get("compiled")
                                   is not None else "cost"),
                "work": ({key: str(value) for key, value in cost.work}
                         if cost is not None else {}),
                "stage_work": ([{key: str(value) for key, value in stage}
                                for stage in cost.stage_work]
                               if cost is not None else []),
                "state_effects": [dict(item)
                                  for item in part.get("effects") or ()],
            })
        return {
            "schema": LEAF_PREPARED_SCHEMA,
            "name": self.name,
            "call": (None if self.call is None else {
                "entry": getattr(getattr(self.call, "ledger", {}), "get",
                                 lambda *_: None)("call"),
                "operators": len(self.call.ops),
                "recipes": sorted(dict(getattr(self.call, "recipes",
                                               {}).recipes)),
            }),
            "kind": ("call" if self.call is not None else "parts"),
            "parts": parts,
            "typed_ports": [dict(item) for item in self.ports],
            "state_ports": [dict(item) for item in self.state_ports],
            "state_value": _jsonable(self.state_value),
            "recipes": (None if self.call is None else sorted(
                dict(getattr(self.call, "recipes", {}).recipes))),
            "owner": self.owner,
            "placement": [int(self.replica), int(self.stage)],
            "prepared": bool(self.prepared),
            "start_signature": ("start(execution, *, name=None, ready=None, "
                                "replica=None, stage=None, owner=None, "
                                "holds=None, state=None, grants=None, "
                                "output_grants=None, event_bindings=None)"),
            "non_claims": ["selected_plan_not_repriced",
                           "caller_execution_owns_engine"],
        }

    def _placement(self, execution: Any, replica: Any, stage: Any) -> tuple:
        replica = int(self.replica if replica is None else replica)
        stage = int(self.stage if stage is None else stage)
        if (replica, stage) not in execution.stage_contexts:
            raise ModelingError(
                f"leaf_execution_placement_unavailable:{replica}:{stage}")
        return replica, stage

    def _grant_events(self, execution: Any, name: str, ready: Any, owner: str,
                      grants: Any) -> tuple:
        from mindie_sim.prediction.engine import Allocation
        engine = execution.engine
        records, releases = [], []
        current = ready
        for output_name, grant in dict(grants or {}).items():
            if grant is None or grant.get("borrow"):
                continue
            if not isinstance(grant, Mapping):
                raise ModelingError(
                    f"leaf_execution_grant_invalid:{output_name}")
            pool = str(grant.get("pool") or "")
            size = grant.get("size")
            if not pool or size is None:
                raise ModelingError(
                    f"leaf_execution_grant_incomplete:{output_name}")
            allocation = Allocation(
                str(grant.get("name") or f"{output_name}_grant"),
                pool, float(size))
            acquire = engine.submit(
                f"{name}:grant:{output_name}", after=(current,),
                acquire=(allocation,), owner=owner)
            records.append((allocation, acquire, grant))
            current = acquire
        if current is not ready:
            ready = current
        return ready, records

    def start(self, execution: Any, *, name: str | None = None,
              ready: Any = None, replica: Any = None, stage: Any = None,
              owner: str | None = None, holds: Any = None,
              state: Any = None, grants: Any = None,
              output_grants: Any = None,
              event_bindings: Any = None,
              allocations: Any = None) -> LeafExecutionHandle:
        """Submit this prepared occurrence into the caller's Execution."""
        if not hasattr(execution, "engine"):
            raise ModelingError("leaf_execution_requires_caller_execution")
        replica, stage = self._placement(execution, replica, stage)
        owner = self.owner if owner is None else str(owner)
        name = str(name or self.name)
        start = _leaf_event(ready)
        if start is None:
            start = execution.engine.submit(f"{name}:ready", owner=owner)
        grant_ready, grants = self._grant_events(
            execution, name, start, owner,
            output_grants if output_grants is not None else grants)
        hold_events = tuple(
            _leaf_event(value) for value in dict(holds or {}).values())
        if self.call is not None:
            handle = self._start_call(
                execution, name, grant_ready, owner=owner, holds=hold_events,
                state=state, replica=replica, stage=stage)
        else:
            handle = self._start_parts(
                execution, name, grant_ready, owner=owner, holds=holds,
                state=state, replica=replica, stage=stage,
                event_bindings=event_bindings)
        releases = []
        terminal = handle.event
        for allocation, _acquire, grant in grants:
            after = [terminal]
            hold = grant.get("last_reader", grant.get("hold"))
            if hold is not None:
                after.append(_leaf_event(hold))
            releases.append(execution.engine.submit(
                f"{name}:release:{allocation.name}", after=tuple(after),
                free=(allocation,), owner=owner))
        self.prepared = True
        return replace(
            handle, grants=tuple(
                {"name": allocation.name, "pool": allocation.pool,
                 "size": str(allocation.size),
                 "acquire": _leaf_event_view(acquire),
                 "last_reader": (None if grant.get("last_reader") is None
                                 else str(getattr(grant.get("last_reader"),
                                                  "name",
                                                  grant.get("last_reader"))))}
                for allocation, acquire, grant in grants),
            release_events=tuple(releases), owner=owner,
            placement=(replica, stage), execution=execution)

    def _start_parts(self, execution: Any, name: str, ready: Any, *,
                     owner: str, holds: Any, state: Any, replica: int,
                     stage: int, event_bindings: Any) -> LeafExecutionHandle:
        from mindie_sim.prediction.piecewise.composition import submit_cost
        engine = execution.engine
        bindings = _leaf_resource_bindings(execution, replica, stage)
        current = ready
        events, work, part_work, part_spans = [], [], [], []
        output_events, state_events, writers, reads = [], [], [], []
        hold_map = dict(holds or {})
        if hold_map:
            join_ready = engine.submit(
                f"{name}:holds_ready", after=(ready, *(
                    _leaf_event(value) for value in hold_map.values())),
                owner=owner)
        else:
            join_ready = ready
        current = join_ready
        for index, part in enumerate(self.parts):
            cost = part["cost"]
            part_start = current
            # A state read needs a caller-owned committed version; an
            # independent leaf request names the port but carries no ledger.
            # Submit the same cost's physical stages in that case and keep the
            # typed state port in the program/occurrence record.
            has_state = bool(part.get("effects"))
            submitted = submit_cost(
                cost, execution, name=f"{name}:p{index}", ready=current,
                supply=self.supply, resource_bindings=bindings,
                operator=(None if has_state else part.get("compiled")),
                geometry=(None if has_state else part.get("geometry")),
                state=state, holds=(None if has_state else hold_map),
                event_bindings=event_bindings, replica=replica, stage=stage)
            part_events = tuple(submitted.events)
            events.extend(part_events)
            work.extend(submitted.work)
            part_work.append((str(part.get("name") or f"p{index}"),
                              tuple(submitted.work)))
            if part_events:
                current = part_events[-1]
                start_time = (None if getattr(part_start, "time", None) is None
                              else float(part_start.time))
                end_time = (None if getattr(current, "time", None) is None
                            else float(current.time))
                part_spans.append((
                    str(part.get("name") or f"p{index}"),
                    _leaf_event_view(part_events[0]),
                    _leaf_event_view(current), start_time, end_time,
                    (None if start_time is None or end_time is None
                     else end_time - start_time)))
            for effect in part.get("effects") or ():
                direction = str(effect.get("direction") or "")
                record = {
                    "tensor": str(effect.get("tensor") or ""),
                    "direction": direction,
                    "order": effect.get("order"),
                    "slot": effect.get("slot"),
                    "event": (current if direction in {"write", "read_write"}
                              else ready),
                }
                state_events.append(record)
                if direction in {"write", "read_write"}:
                    writers.append({
                        "tensor": record["tensor"],
                        "event": record["event"],
                        "producer": str(effect.get("role") or "state_write"),
                        "writer": "terminal",
                    })
                if direction in {"read", "read_write"}:
                    reads.append({
                        "tensor": record["tensor"],
                        "event": ready,
                        "reader": str(effect.get("role") or "state_read"),
                    })
        terminal = current if current is not ready else ready
        if not events:
            terminal = engine.submit(f"{name}:empty_terminal", after=(ready,),
                                     owner=owner)
            events = [terminal]
        stages = []
        for part in self.parts:
            stages.extend(tuple(stage) for stage in
                          getattr(part["cost"], "stage_work", ()))
        stage_work = tuple(stages)
        return LeafExecutionHandle(
            event=terminal, events=tuple(events), start_event=ready,
            output_events=tuple(output_events),
            state_events=tuple(state_events),
            work=tuple(work), stage_work=stage_work,
            part_work=tuple(part_work), part_spans=tuple(part_spans),
            typed_values=tuple(self.ports),
            state_values=(tuple(self.state_value.items())
                          if isinstance(self.state_value, Mapping)
                          else tuple(self.state_value or ())),
            writers=tuple(writers), reads=tuple(reads))

    def _start_call(self, execution: Any, name: str, ready: Any, *,
                    owner: str, holds: tuple, state: Any, replica: int,
                    stage: int) -> LeafExecutionHandle:
        from mindie_sim.schema.operators import IterationWork
        from mindie_sim.schema.workload import IterationPlan
        engine = execution.engine
        terminal = engine.future(f"{name}:complete")
        part = self.parts[0]
        plan = IterationWork(
            self.call.ops, IterationPlan(()), 1, 0, 0, None, "graph",
            "declared_graph", replay_scope="none")

        def begin(_event: Any) -> None:
            saved = execution.recipes
            execution.recipes = self.call.recipes
            try:
                step = execution.add(plan, replica, engine.now,
                                     state_accesses=state)
            finally:
                execution.recipes = saved
            engine.submit(
                f"{name}:forward", after=(step.completion,),
                on_complete=lambda _done: engine.resolve(terminal),
                owner=owner)

        after = (ready, *holds)
        engine.submit(f"{name}:begin", after=tuple(after), on_complete=begin,
                      owner=owner)
        state_events, writers, reads = [], [], []
        for effect in part.get("effects") or ():
            direction = str(effect.get("direction") or "")
            event = terminal if direction in {"write", "read_write"} else ready
            record = {
                "tensor": str(effect.get("tensor") or ""),
                "direction": direction, "order": effect.get("order"),
                "slot": effect.get("slot"), "event": event,
            }
            state_events.append(record)
            if direction in {"write", "read_write"}:
                writers.append({
                    "tensor": record["tensor"], "event": event,
                    "producer": str(effect.get("role") or "state_write"),
                    "writer": "terminal",
                })
            if direction in {"read", "read_write"}:
                reads.append({
                    "tensor": record["tensor"], "event": ready,
                    "reader": str(effect.get("role") or "state_read"),
                })
        ledger = (self.call.ledger if isinstance(self.call.ledger, Mapping)
                  else {})
        output_events = tuple(
            (str(name), terminal)
            for name in dict(ledger.get("outputs") or {}))
        return LeafExecutionHandle(
            event=terminal, events=(terminal,), start_event=ready,
            output_events=output_events,
            work=tuple(part.get("work") or ()),
            stage_work=tuple(part.get("stage_work") or ()),
            part_work=(("call", tuple(part.get("work") or ())),),
            typed_values=tuple(self.ports),
            state_values=(tuple(self.state_value.items())
                          if isinstance(self.state_value, Mapping)
                          else tuple(self.state_value or ())),
            state_events=tuple(state_events),
            writers=tuple(writers), reads=tuple(reads))


def _leaf_execution_report(handle: LeafExecutionHandle, execution: Any,
                           before: dict, *, own_execution: bool,
                           program_duration: Any, trace_limit: Any,
                           trace_rows: Any = (),
                           program_completed_tasks: Any = None) -> dict:
    """Separate occurrence work from whole-engine reconciliation."""
    from fractions import Fraction
    after = _leaf_engine_state(execution)
    engine = execution.engine
    serviced = {}
    busy = {}
    for name, value in after["serviced"].items():
        delta = value - before["serviced"].get(name, Fraction(0))
        if delta:
            serviced[str(name)] = delta
    for name, value in after["busy_ms"].items():
        delta = value - before["busy_ms"].get(name, Fraction(0))
        if delta:
            busy[str(name)] = delta
    completed_delta = int(after["completed"]) - int(before["completed"])
    retained_delta = max(0, int(after["trace"]) - int(before["trace"]))
    dropped_delta = max(0, int(after["trace_dropped"])
                        - int(before["trace_dropped"]))
    duration = handle.duration_ms
    rows = tuple(trace_rows)
    if not own_execution and retained_delta:
        rows = rows[-retained_delta:]
    resources = [{
        "resource": str(name), "unit": resource.unit,
        "capacity_per_ms": resource.rate,
        "busy_ms": str(busy.get(str(name), 0)),
        "work": str(serviced.get(str(name), 0)),
    } for name, resource in engine.resources.items()]
    summary = engine.summary()
    event_done = handle.done
    return {
        "schema": "mindie-sim/leaf-execution-report/1",
        "status": "completed" if (own_execution and event_done) else "pending",
        "own_execution": bool(own_execution),
        "advanced": bool(own_execution and event_done),
        "owner": handle.owner,
        "placement": list(handle.placement),
        "terminal": _leaf_event_view(handle.event),
        "start": _leaf_event_view(handle.start_event),
        "duration_ms": (str(duration) if duration is not None else None),
        "program_duration_ms": str(program_duration),
        "work": {str(key): str(value)
                 for key, value in _leaf_work_map(handle.work).items()},
        "stage_work": [{str(key): str(value) for key, value in stage}
                       for stage in handle.stage_work],
        "part_work": [{"part": part,
                       "work": {str(key): str(value)
                                for key, value in _leaf_work_map(work).items()}}
                      for part, work in handle.part_work],
        "part_spans": [{"part": item[0], "start": item[1], "terminal": item[2],
                        "duration_ms": (None if item[5] is None
                                        else str(item[5]))}
                       for item in handle.part_spans],
        "completed_tasks": (completed_delta if program_completed_tasks is None
                            else int(program_completed_tasks)),
        "trace": {
            "limit": int(getattr(engine.trace, "maxlen", 0) or 0),
            "retained": retained_delta,
            "dropped": dropped_delta,
            "events": [{key: (str(value) if isinstance(value, Fraction)
                              else value)
                        for key, value in dict(row).items()}
                       for row in rows[-128:]],
        },
        "memory": {
            "peak_allocated": dict(summary["peak_allocated"]),
            "live_allocated": dict(summary["live_allocated"]),
            "peak_allocated_per_unit": dict(
                summary["peak_allocated_per_unit"]),
        },
        "resources": resources,
        "engine_reconciliation": {
            "basis": "whole_engine_serviced_delta",
            "serviced": {key: str(value) for key, value in serviced.items()},
            "busy_ms": {key: str(value) for key, value in busy.items()},
            "completed_tasks": completed_delta,
            "used_for_occurrence_work": False,
        },
        "occurrence_work_basis": "submitted_fragment_work",
        "non_claims": [
            "engine_delta_is_whole_machine_reconciliation_only",
            "trace_limit_changes_diagnostics_only",
        ],
    }


def simulate_leaf(request: Any, spec: Any, mapper: Any, *,
                  hardware: Any = None, supply: Any = None,
                  policy: Any = None, cores: Any = None, mt: Any = None,
                  phase: str | None = None, profiler_rows: Any = None,
                  require_explicit_version: bool = True,
                  require_explicit_variant: bool = False,
                  trace_limit: Any = None,
                  execution: Any = None,
                  ready: Any = None, ready_ms: Any = None,
                  holds: Any = None, state: Any = None,
                  grants: Any = None, output_grants: Any = None,
                  event_bindings: Any = None, owner: str = "",
                  replica: int = 0, stage: int = 0,
                  blocking: bool | None = None,
                  prepare_only: bool = False,
                  strict: bool = False) -> dict:
    """Predict one leaf native call from its independent typed input.

    ``mapper(invocation, supply, policy, phase, machine, trace_limit,
    execution)`` returns a plan with
    ``cost``/``region``/``typed``/``occurrence``/``effects`` for one call, or
    ``parts`` for a request that expands into several child kernels.  A
    missing field returns a replayable template that keeps the original
    request; unknown ABI/attr/version or a known implementation constraint
    returns an explicit status and never selects another algorithm or duration
    envelope.
    """
    from mindie_sim.prediction.piecewise.native_invocation import (
        NativeAbiIncomplete, NativeAbiMismatch, NativeAbiVersionMismatch,
        normalize_native_abi)
    from mindie_sim.prediction.native_binding import (
        bind_abi_call, join_profiler_rows)
    from mindie_sim.schema.common import ModelingError as _Error
    from mindie_sim.prediction.piecewise.native_calls import (
        validate_native_occurrence)

    def result(status: str, *, invocation=None, binding=None,
               bindings=(), prediction=None, program=None, occurrence=None,
               occurrences=(), replay=None, reasons=(), environment=None,
               spec_dict=None, profiler_join=None, execution_program=None,
               submission=None, execution_report=None,
               execution_context=None) -> dict:
        payload = {
            "schema": LEAF_ABI_SCHEMA, "status": status,
            "spec": spec_dict if spec_dict is not None else spec.to_dict(),
            "invocation": invocation,
            "binding": binding.to_dict() if binding is not None else None,
            "bindings": [item.to_dict() for item in bindings],
            "prediction": prediction, "program": program,
            "occurrence": occurrence,
            "occurrences": list(occurrences),
            "profiler_join": profiler_join,
            "replay_minimal": replay or {},
            "reasons": list(dict.fromkeys(str(item) for item in reasons)),
            "environment": environment or {"basis": "not_evaluated"},
            "execution": execution_report,
            "submission": (None if submission is None else submission.to_dict()),
            "execution_handle": submission,
            "execution_program": execution_program,
            "execution_context": execution_context,
            "native_call_succeeded": status == "predicted",
            "fallback_performed": False,
            "non_claims": list(LEAF_NON_CLAIMS),
        }
        if strict and status != "predicted":
            raise ModelingError(
                "leaf_abi_unavailable:" + status + ":"
                + ";".join(payload["reasons"]))
        return payload

    entry = _leaf_entry(request)
    if entry not in {spec.entry, spec.aclnn_symbol}:
        return result("no_match", replay=_leaf_replay(
            spec, request, ("api.entry",), reason="native_abi_api_entry_mismatch",
            replayable=False),
            reasons=(f"leaf_abi_entry_not_declared:{entry}",))
    try:
        inv = normalize_native_abi(request, spec)
    except NativeAbiIncomplete as gap:
        missing = _leaf_missing(spec, request) or gap.missing
        return result("insufficient_fields", replay=_leaf_replay(
            spec, request, missing, replayable=False),
            reasons=(gap.reason, *missing))
    except NativeAbiVersionMismatch as mismatch:
        return result("version_mismatch", replay=_leaf_replay(
            spec, request, (), reason="version_mismatch", replayable=False,
            conflicts=(mismatch.reason,)),
            reasons=(mismatch.reason,))
    except NativeAbiMismatch as mismatch:
        return result("no_match", replay=_leaf_replay(
            spec, request, (), reason="request_mismatch", replayable=False,
            conflicts=(mismatch.reason,)),
            reasons=(mismatch.reason,))
    except _Error as error:
        return result("no_match", replay=_leaf_replay(
            spec, request, (), reason="request_malformed", replayable=False,
            conflicts=(getattr(error, "reason", str(error)),)),
            reasons=(getattr(error, "reason", str(error)),))
    if require_explicit_version:
        missing = tuple(name for name, key in (
            ("version.api", "api"), ("version.source", "source"),
            ("version.image_id", "image_id")) if not inv.version.get(key))
        if missing:
            return result("insufficient_fields", invocation=inv.to_dict(),
                          replay=_leaf_replay(spec, request, missing,
                                              replayable=False),
                          reasons=("leaf_abi_explicit_version_required", *missing))
    if require_explicit_variant:
        raw_conditions = (request.get("conditions")
                          if isinstance(request, dict) else None) or {}
        if not (raw_conditions.get("variant") or raw_conditions.get("owner")):
            return result("insufficient_fields", invocation=inv.to_dict(),
                          replay=_leaf_replay(
                              spec, request, ("conditions.variant",),
                              reason="variant_required", replayable=False),
                          reasons=("leaf_abi_variant_required",))
    if not spec.implemented:
        return result("not_implemented", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="not_implemented",
                                          replayable=False),
                      reasons=spec.gap or ("leaf_abi_program_mapper_missing",))
    try:
        environment, supply_obj, arch, machine = _leaf_environment(
            hardware, supply)
        policy_obj = _leaf_policy(policy, arch, cores, mt)
    except _Error as error:
        return result("environment_unavailable", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="environment_unavailable",
                                          replayable=False),
                      reasons=(getattr(error, "reason", str(error)),))
    if spec.arch and arch and spec.arch != arch:
        return result("hardware_mismatch", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="hardware_mismatch",
                                          replayable=False),
                      reasons=(f"leaf_abi_arch_mismatch:{arch}!={spec.arch}",),
                      environment=environment)
    variant = next((item for item in spec.variants
                    if item.name == str(inv.conditions.get("variant") or "")),
                   None)
    if (variant is None and not require_explicit_variant
            and len(spec.variants) == 1):
        # A standalone capture retains no caller variant; a single declared
        # physical variant is the only possible mapping.
        variant = spec.variants[0]
    if variant is None:
        return result("no_match", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="variant_not_declared",
                                          replayable=False),
                      reasons=("leaf_abi_variant_not_declared",),
                      environment=environment)
    runtime_phase = str(phase or inv.conditions.get("phase") or "")
    try:
        plan = mapper(inv, supply_obj, policy_obj, runtime_phase, machine,
                      trace_limit, execution)
    except LeafAbiConstraint as limit:
        return result("constraint_violation", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason=limit.reason,
                                          replayable=False,
                                          conflicts=(limit.reason,)),
                      reasons=(limit.reason,), environment=environment)
    except LeafAbiUnsupported as gap:
        return result("no_match", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="unsupported_geometry",
                                          replayable=False,
                                          conflicts=(gap.reason,)),
                      reasons=(gap.reason,), environment=environment)
    except _Error as error:
        return result("prediction_failed", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="prediction_failed",
                                          replayable=False),
                      reasons=(getattr(error, "reason", str(error)),),
                      environment=environment)
    if plan is None:
        return result("no_match", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="unsupported_geometry",
                                          replayable=False),
                      reasons=("leaf_abi_geometry_not_supported",),
                      environment=environment)
    api = {
        "namespace": inv.api["namespace"], "entry": inv.api["entry"],
        "aclnn_symbol": inv.api.get("aclnn_symbol"),
        "version": inv.version.get("api") or spec.api_version,
    }
    raw_parts = plan.get("parts")
    if raw_parts is None:
        raw_parts = ({
            "cost": plan["cost"], "region": plan["region"],
            "typed": plan["typed"], "occurrence": plan.get("occurrence"),
            "effects": plan.get("effects") or (),
            "unresolved": plan.get("unresolved") or (),
            "compiled": plan.get("compiled"),
            "geometry": plan.get("geometry"),
            "mode": variant.name, "op_id": variant.op_id,
            "kind": plan.get("kind") or variant.op_id.split(":", 1)[0],
            "phase": plan.get("phase") or runtime_phase,
            "charge": "call", "source_boundary": variant.source_boundary,
            "owner": variant.owner, "members": variant.members,
        },)
    bindings, occurrences, effects, part_views = [], [], [], []
    for index, part in enumerate(raw_parts):
        unresolved = tuple(dict.fromkeys(
            (*spec.unresolved, *tuple(part.get("unresolved") or ()),
             "execution_rank_not_selected")))
        try:
            binding = bind_abi_call(
                part["region"], api=api, source=spec.source,
                typed=part["typed"],
                kernels=part.get("kernels") or spec.kernels,
                members=part.get("members", variant.members),
                owner=part.get("owner", variant.owner),
                execution_core=spec.execution_core, unresolved=unresolved)
        except _Error as error:
            return result("prediction_failed", invocation=inv.to_dict(),
                          replay=_leaf_replay(spec, request, (),
                                              reason="binding_failed",
                                              replayable=False),
                          reasons=(f"binding_failed:{getattr(error, 'reason', error)}",),
                          environment=environment)
        if binding is None:
            return result("no_match", invocation=inv.to_dict(),
                          replay=_leaf_replay(spec, request, (),
                                              reason="binding_unavailable",
                                              replayable=False),
                          reasons=("leaf_abi_binding_unavailable",),
                          environment=environment)
        part_occurrence = {
            "schema": "mindie-sim/piecewise-native-occurrence/1",
            "status": "bound", "call_id": binding.call_id,
            "op_id": part.get("op_id", variant.op_id),
            "family": part["region"].family, "recipe": part["region"].recipe,
            "source_boundary": part.get("source_boundary",
                                        variant.source_boundary),
            "phase": part.get("phase") or runtime_phase,
            "kind": part.get("kind", variant.op_id.split(":", 1)[0]),
            "mode": part.get("mode", variant.name),
            "charge": part.get("charge", "call"),
            "call_owner": part.get("owner", variant.owner),
            "typed": part.get("occurrence") or {},
            "priced": {
                "duration_ms": str(part["cost"].duration_ms),
                "dominant_pipe": part["cost"].dominant_pipe,
                "work": {key: str(value)
                         for key, value in part["cost"].work},
                "charged": "once_at_timing_owner",
            },
            "state_effects": list(part.get("effects") or ()),
            "unresolved": list(unresolved),
        }
        try:
            part_occurrence["typed"] = validate_native_occurrence(
                binding, dict(part_occurrence["typed"]))
        except _Error as error:
            return result("prediction_failed", invocation=inv.to_dict(),
                          replay=_leaf_replay(spec, request, (),
                                              reason="occurrence_failed",
                                              replayable=False),
                          reasons=(f"occurrence_failed:{getattr(error, 'reason', error)}",),
                          environment=environment)
        bindings.append(binding)
        occurrences.append(part_occurrence)
        effects.extend(part.get("effects") or ())
        part_views.append({
            "index": index, "call_id": binding.call_id,
            "mode": part_occurrence["mode"], "charge": part_occurrence["charge"],
            "duration_ms": str(part["cost"].duration_ms),
            "work": {key: str(value) for key, value in part["cost"].work},
            "state_effects": list(part.get("effects") or ()),
        })
    cost = plan.get("cost")
    if cost is None and raw_parts:
        # Multiple child kernels: project the actual per-child stages instead
        # of extracting work from two OperatorCosts into one fake program.
        from types import SimpleNamespace
        from fractions import Fraction
        work: dict[str, Fraction] = {}
        times: dict[str, Fraction] = {}
        stage_work: list[tuple] = []
        stage_times: list[Fraction] = []
        materialization: dict[str, Fraction] = {}
        internal: dict[str, Fraction] = {}
        for part in raw_parts:
            part_cost = part["cost"]
            for key, value in part_cost.work:
                work[key] = work.get(key, Fraction(0)) + value
            for key, value in part_cost.times:
                times[key] = times.get(key, Fraction(0)) + value
            stage_work.extend(tuple(stage) for stage in part_cost.stage_work)
            stage_times.extend(part_cost.stage_times)
            for key, value in part_cost.materialization:
                materialization[key] = (
                    materialization.get(key, Fraction(0)) + value)
            for key, value in part_cost.internal:
                internal[key] = internal.get(key, Fraction(0)) + value
        for key, value in (plan.get("aggregate_materialization") or ()):
            materialization[str(key)] = Fraction(str(value))
        duration = sum((part["cost"].duration_ms for part in raw_parts),
                       Fraction(0))
        peak = max(times.values()) if times else Fraction(0)
        tied = tuple(key for key, value in times.items() if value == peak)
        dominant = tied[0] if tied else ""
        cost = SimpleNamespace(
            name=plan.get("aggregate_name", "leaf.children"),
            duration_ms=duration, dominant_pipe=dominant, tied_pipes=tied,
            work=tuple(sorted(work.items())),
            times=tuple(sorted(times.items())),
            stage_times=tuple(stage_times),
            stage_work=tuple(stage_work),
            materialization=tuple(sorted(materialization.items())),
            internal=tuple(sorted(internal.items())),
            residency_bytes=max((part["cost"].residency_bytes
                                 for part in raw_parts), default=Fraction(0)),
            derived=tuple((str(key), Fraction(str(value)))
                          for key, value in (plan.get("aggregate_derived") or ())),
            formula=" ; ".join(part["cost"].formula for part in raw_parts),
            fidelity="isolated_device_time")
    if cost is None:
        return result("prediction_failed", invocation=inv.to_dict(),
                      replay=_leaf_replay(spec, request, (),
                                          reason="aggregate_cost_missing",
                                          replayable=False),
                      reasons=("leaf_abi_aggregate_cost_missing",),
                      environment=environment)
    runtime_phase = str(plan.get("phase") or runtime_phase
                        or (raw_parts[0].get("phase") if raw_parts else ""))
    environment = {
        **environment, "phase": runtime_phase,
        **dict(plan.get("environment") or {}),
        "vector_cores": int(policy_obj.cores),
        "policy": {"cores": int(policy_obj.cores), "mt": int(policy_obj.mt),
                   "pipeline_slots": int(policy_obj.pipeline_slots)},
        "pipe_supply": supply_obj.to_dict(),
    }
    profiler = (join_profiler_rows(profiler_rows, bindings)
                if profiler_rows is not None else None)
    final_status = str(plan.get("status") or "predicted")
    final_reasons = tuple(plan.get("reasons") or ())
    prediction = _leaf_prediction(
        cost, environment, effects, occurrences=part_views,
        parts=plan.get("program_parts") or ())
    # A leaf may price its elapsed launch from a declared clock law while the
    # family compiler keeps the declared work ledger (transpose-batch GEMM):
    # the physical time replaces only the elapsed duration.
    elapsed = plan.get("duration_ms")
    if elapsed is not None:
        from fractions import Fraction as _Fraction
        value = _Fraction(str(elapsed))
        prediction["duration_ms"] = str(value)
        prediction["duration_us"] = float(value) * 1000.0
        if isinstance(prediction.get("cost_record"), dict):
            prediction["cost_record"]["duration_ms"] = str(value)
    from fractions import Fraction
    call = plan.get("call")
    pricing = dict(plan.get("pricing") or {})
    program_name = "native:" + str(inv.api["entry"])
    program_parts, part_work = [], []
    for index, part in enumerate(raw_parts):
        if call is not None and index == 0:
            work = tuple((str(key), Fraction(str(value)))
                         for key, value in (pricing.get("work") or {}).items())
            merged = tuple(
                pair for stage in (pricing.get("stage_work") or ())
                for pair in _leaf_stage_pairs(stage))
            stage_work = (merged,) if merged else ()
        else:
            work = tuple(part["cost"].work)
            stage_work = tuple(tuple(stage)
                               for stage in part["cost"].stage_work)
        part_work.append(work)
        program_parts.append({
            "name": str(part.get("mode") or f"p{index}"),
            "kind": str(part.get("kind") or ""),
            "cost": part["cost"], "compiled": part.get("compiled"),
            "geometry": part.get("geometry"),
            "effects": tuple(part.get("effects") or ()),
            "occurrence": dict(part.get("occurrence") or {}),
            "work": work, "stage_work": stage_work,
        })
    execution_program = LeafExecutionProgram(
        name=program_name, call=call, parts=tuple(program_parts),
        supply=supply_obj, policy=policy_obj, machine=machine,
        ports=tuple({
            "name": tensor.name, "role": tensor.role,
            "dtype": tensor.dtype, "layout": tensor.layout,
            "shape": list(tensor.shape), "present": bool(tensor.present),
        } for tensor in inv.tensors if tensor.present),
        state_ports=tuple(plan.get("state_ports") or ()),
        state_value=(plan.get("program") or {}).get("state_value"),
        program=plan.get("program") or {}, replica=replica, stage=stage,
        owner=owner)
    submission = None
    execution_report = None
    execution_context = None
    if final_status == "predicted" and not prepare_only:
        own_execution = execution is None
        if own_execution:
            execution_context, _env, _supply = leaf_execution(
                hardware, supply, trace_limit=trace_limit)
        else:
            execution_context = execution
        start_ready = ready
        if start_ready is None and ready_ms is not None:
            start_ready = execution_context.engine.submit(
                f"{program_name}:input_ready", {},
                release=float(ready_ms), owner=owner)
        before = _leaf_engine_state(execution_context)
        try:
            submission = execution_program.start(
                execution_context, name=program_name, ready=start_ready,
                replica=int(replica), stage=int(stage), owner=owner,
                holds=holds, state=state,
                grants=grants, output_grants=output_grants,
                event_bindings=event_bindings)
            submission = replace(submission, before=before)
        except _Error as error:
            return result("prediction_failed", invocation=inv.to_dict(),
                          replay=_leaf_replay(
                              spec, request, (),
                              reason="submission_failed", replayable=False),
                          reasons=(f"submission_failed:{getattr(error, 'reason', error)}",),
                          environment=environment)
        if own_execution and blocking is not False:
            execution_context.engine.advance(
                until=(submission.event,), first=False)
        trace_rows = tuple(execution_context.engine.trace)
        execution_report = _leaf_execution_report(
            submission, execution_context, before,
            own_execution=own_execution, program_duration=cost.duration_ms,
            trace_limit=trace_limit, trace_rows=trace_rows,
            program_completed_tasks=(
                pricing.get("completed_tasks")
                if call is not None else None))
        for index, occurrence in enumerate(occurrences):
            submitted = (submission.part_work[index][1]
                         if index < len(submission.part_work)
                         else part_work[index])
            occurrence["priced"]["work"] = {
                str(key): str(value) for key, value in submitted}
            occurrence["priced"]["attribution"] = "submitted_fragment_work"
            span = (submission.part_spans[index]
                    if index < len(submission.part_spans) else None)
            if span is not None and span[5] is not None:
                occurrence["priced"]["duration_ms"] = str(
                    Fraction(str(span[5])))
                occurrence["priced"]["start"] = span[1]
                occurrence["priced"]["terminal"] = span[2]
            occurrence["submission"] = {
                "terminal": _leaf_event_view(submission.event),
                "part": (submission.part_spans[index][0]
                         if index < len(submission.part_spans) else None),
            }
        prediction["occurrence_work"] = {
            str(key): str(value)
            for key, value in _leaf_work_map(submission.work).items()}
        prediction["execution_status"] = execution_report["status"]
        program = dict(plan.get("program") or {})
        program["execution"] = execution_report
        program["submission"] = submission.to_dict()
        program["typed_ports"] = [dict(item) for item in execution_program.ports]
        serialized_handle = submission.to_dict()
        program["state_handles"] = {
            "ports": [dict(item) for item in execution_program.state_ports],
            "events": serialized_handle["state_events"],
            "writers": serialized_handle["writers"],
            "reads": serialized_handle["reads"],
            "terminal": serialized_handle["terminal"],
            "values": _jsonable(execution_program.state_value),
        }
        program["prepared_program"] = execution_program.to_dict()
        if call is not None:
            program["recipes"] = sorted(
                dict(getattr(call.recipes, "recipes", {}) or {}).keys())
            call_report = dict(program.get("call_report") or {})
            call_report.update({
                "own_execution": bool(own_execution),
                "duration_ms": execution_report["duration_ms"],
                "completed_tasks": execution_report["completed_tasks"],
                "execution": {
                    "work": {str(key): str(value)
                             for key, value in submission.work},
                    "recipes": dict(pricing.get("recipes") or {}),
                    "completed_tasks": execution_report["completed_tasks"],
                    "serviced": execution_report[
                        "engine_reconciliation"]["serviced"],
                    "status": execution_report["status"],
                },
                "resources": execution_report["resources"],
                "memory": execution_report["memory"],
                "stage_work": execution_report["stage_work"],
                "stage_work_basis": execution_report[
                    "occurrence_work_basis"],
                "trace": {
                    "limit": int(pricing.get("trace_limit", 0) or 0),
                    "retained": int(
                        pricing.get("retained_trace_events", 0) or 0),
                    "dropped": int(pricing.get("trace_dropped", 0) or 0),
                    "events": list(pricing.get("trace") or ())[:128],
                },
            })
            program["call_report"] = call_report
        plan["program"] = program
    if final_status != "predicted":
        # A partial declaration is not a complete native call: keep its known
        # work visible for diagnosis but remove it from the charged/scoring
        # set and do not advertise a successful call.
        for occurrence in occurrences:
            occurrence["charge"] = "unresolved"
            occurrence["priced"]["charged"] = "excluded_partial"
        prediction["scoring"] = "excluded_partial"
        prediction["complete"] = False
        prediction["partial_reasons"] = list(final_reasons)
        plan["program"]["complete"] = False
    return result(
        final_status, invocation=inv.to_dict(),
        binding=bindings[0] if bindings else None,
        bindings=tuple(bindings),
        prediction=prediction,
        program=plan["program"],
        occurrence=occurrences[0] if occurrences else None,
        occurrences=tuple(occurrences),
        replay=_leaf_replay(
            spec, request, (),
            reason=("partial_invocation" if final_status == "partial"
                    else "complete_invocation"),
            replayable=final_status == "predicted"),
        reasons=final_reasons,
        environment=environment, spec_dict=spec.to_dict(),
        profiler_join=profiler, execution_program=execution_program,
        submission=submission, execution_report=execution_report,
        execution_context=execution_context)


def state_effect(*, order: int, direction: str, slot: str, tensor: str,
                 rows: int, dtype: str, role: str,
                 occurrence: str = "call", **extra: Any) -> dict:
    """One ordered state read/write effect, never a duration."""
    if direction not in {"read", "write", "read_write"}:
        raise ModelingError(f"leaf_state_effect_direction_invalid:{direction}")
    if set(extra) & {"duration", "duration_ms", "time_ms", "ms", "seconds"}:
        raise ModelingError("leaf_state_effect_must_not_carry_duration")
    record = {"order": int(order), "direction": direction, "slot": str(slot),
              "tensor": str(tensor), "rows": int(rows), "dtype": str(dtype),
              "role": str(role), "occurrence": str(occurrence)}
    record.update({key: value for key, value in extra.items()
                   if value is not None})
    return record


def leaf_abi_specs() -> tuple:
    """All leaf ABI declarations, for T12's single catalogue/dispatcher.

    This only advertises declarations owned by the attention/index/state leaf
    modules; T12 remains the single owner of the public dispatcher and
    inventory.  Import is lazy so the catalogue has no import cycle.
    """
    from . import (
        native_gather_ssm_states, native_l2norm_fwd_kernel2,
        native_r64_aclnn_cat, native_r64_layer_norm_with_impl_mode,
        native_r64_muls_add, native_r64_scatter_pa_cache,
        native_r64_transpose_batch_matmul, native_r64_triton_rope_siso,
        native_scatter_ssm_states, native_split_qkv_rmsnorm_mrope)
    specs = [
        native_gather_ssm_states.abi_spec(),
        native_scatter_ssm_states.abi_spec(),
        native_r64_scatter_pa_cache.abi_spec(),
        native_l2norm_fwd_kernel2.abi_spec(),
        native_r64_triton_rope_siso.abi_spec(),
        native_r64_aclnn_cat.abi_spec(),
        native_r64_muls_add.abi_spec(),
        native_r64_layer_norm_with_impl_mode.abi_spec(),
        native_r64_transpose_batch_matmul.abi_spec(),
        native_split_qkv_rmsnorm_mrope.abi_spec(),
        gdn_abi_spec(),
    ]
    return tuple(specs)


# ---------------------------------------------------------------------------
# Independent GDN chunk ABI (packed/dense, lengths or qsl)
#
# The parent call is one ACLNN call with prepare/gated_delta members.  This
# entry consumes the real q/k/v/g/beta/state tensors, normalizes dense/packed
# physical axes against the same sequence partition, and executes the selected
# F-STATE call once on the ordinary Engine.  Complete service is reduced from
# Engine.serviced; Engine.trace is diagnostic only.
# ---------------------------------------------------------------------------

GDN_ABI_ENTRY = "aclnnChunkGatedDeltaRuleFwd"
GDN_ABI_SOURCE_REVISION = "c3d00a196905"


def gdn_abi_spec() -> Any:
    """The declared independent ABI of the GDN chunk prefill call."""
    from .native_calls import (
        NativeAbiCallSpec, NativeAbiTensorSpec, NativeAbiVariant)
    from mindie_sim.prediction import call_gdn
    from .native_scatter_ssm_states import QWEN36_IMAGE_SHA
    source = {
        "repository": "vllm-ascend",
        "path": call_gdn.PREFILL_SOURCE["archive"],
        "sha256": call_gdn.PREFILL_SOURCE["sha256"],
        "symbol": "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd",
        "revision": GDN_ABI_SOURCE_REVISION,
        "state": "task_artifact",
    }
    kernels = (
        {"name": "Cast", "kind": "copy", "evidence": "measured"},
        {"name": "Transpose", "kind": "copy", "evidence": "measured",
         "count": 3},
        {"name": "Prepare32", "kind": "vector", "evidence": "measured"},
        {"name": "H24", "kind": "cube", "evidence": "measured"},
        {"name": "O32", "kind": "cube", "evidence": "measured"},
    )
    return NativeAbiCallSpec(
        entry=GDN_ABI_ENTRY, namespace="aclnn",
        aclnn_symbol=GDN_ABI_ENTRY,
        api_version=GDN_ABI_SOURCE_REVISION, source=source,
        source_revision=GDN_ABI_SOURCE_REVISION, image_id=QWEN36_IMAGE_SHA,
        kernels=kernels,
        tensors=(
            NativeAbiTensorSpec("q", "input", "bf16",
                                "batch,tokens,key_heads,dk"),
            NativeAbiTensorSpec("k", "input", "bf16",
                                "batch,tokens,key_heads,dk"),
            NativeAbiTensorSpec("v", "input", "bf16",
                                "batch,tokens,value_heads,dv"),
            NativeAbiTensorSpec("g", "input", "fp32",
                                "batch,tokens,value_heads"),
            NativeAbiTensorSpec("beta", "input", "bf16",
                                "batch,tokens,value_heads"),
            NativeAbiTensorSpec("initial_state", "inout", "fp32",
                                "batch,value_heads,dv,dk"),
            NativeAbiTensorSpec("o", "output", "bf16",
                                "batch,tokens,value_heads,dv"),
            NativeAbiTensorSpec("final_state", "output", "fp32",
                                "batch,value_heads,dv,dk"),
        ),
        attrs=("chunk_size", "scale", "use_qk_l2norm_in_kernel", "use_exp2",
               "disable_recompute", "state_v_first", "output_final_state",
               "use_gate_in_kernel", "use_beta_sigmoid_in_kernel",
               "allow_neg_eigval", "return_intermediate_states"),
        optional_attrs=("lengths", "cu_seqlens"),
        variants=(NativeAbiVariant(
            "chunked_prefill", "A3:recurrent:gated_delta",
            "A3:recurrent:gated_delta", ("prepare", "gated_delta"),
            "call", "one_to_many"),),
        execution_core="mix", arch="A5", implemented=True,
        unresolved=("python_caller_module_not_retained",
                    "conditional_output_requires_f_state_call_gdn"))


def _materialization_from_ledger(ledger: Any) -> tuple:
    from fractions import Fraction
    rows = tuple(ledger.get("materializations") or ())
    read = sum(Fraction(str(row.get("read_bytes", 0))) for row in rows
               if isinstance(row, dict))
    write = sum(Fraction(str(row.get("write_bytes", 0))) for row in rows
                if isinstance(row, dict))
    return (("read_bytes", read), ("write_bytes", write),
            ("records", Fraction(len(rows))))


def _call_execution_cost(call: Any, machine: Any, *,
                         supply: Any = None, policy: Any = None,
                         trace_limit: Any = None,
                         execution: Any = None) -> tuple[Any, dict]:
    """Price the selected call with the same explicit supply it will use.

    This prepare-time evaluation builds the declared work/stage record once.
    It is not the occurrence evidence of a caller submission: the actual
    caller Execution starts the same ``call.ops``/``call.recipes`` through
    :class:`LeafExecutionProgram`.  The context is hardware+runtime only; no
    ModelSpec or WorkloadSpec is fabricated.
    """
    from dataclasses import dataclass
    from fractions import Fraction
    from types import SimpleNamespace
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.schema.case import Software
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan

    limit = 0 if trace_limit is None else int(trace_limit)
    if limit < 0:
        raise LeafAbiUnsupported("gdn_abi_trace_limit_invalid")
    # Host ready/schedule events share the bounded trace with program tasks.
    # Reserve their exact count so the requested program window stays intact.
    control_tasks = 1 + len(call.ops)
    capacity = 0 if limit == 0 else limit + control_tasks
    created = execution is None
    if created:
        context = SimpleNamespace(
            machine=_leaf_machine_with_supply(machine, supply),
            runtime=Software())
        cost_model = CostModel(context)
        execution = Execution(
            context, cost_model, recipes=call.recipes, trace_limit=capacity,
            include_submission=False)
    engine = execution.engine
    before_serviced = {key: float(value) for key, value in engine.serviced.items()}
    before_busy = {key: float(value) for key, value in engine.busy_ms.items()}
    before_work = dict(execution.work)
    before_completed = engine.completed_count
    before_dropped = engine.trace_dropped
    before_recipes = dict(execution.recipes_used)
    before_retained = len(engine.trace)
    release = 0.0 if created else engine.now
    work = IterationWork(
        call.ops, IterationPlan(()), 1, 0, 0, None, "graph",
        "declared_graph", replay_scope="none")
    step = execution.add(work, 0, release)
    engine.advance(until=(step.completion,), first=False)
    summary = engine.summary()
    completed_delta = int(summary["completed_tasks"]) - int(before_completed)
    retained_delta = max(0, int(summary["retained_trace_events"]) - before_retained)
    dropped_delta = int(summary["trace_dropped"]) - int(before_dropped)
    serviced = {str(key): Fraction(str(value)) - Fraction(str(before_serviced.get(key, 0.0)))
                for key, value in summary["serviced"].items()}
    busy = {str(key): Fraction(str(value)) - Fraction(str(before_busy.get(key, 0.0)))
            for key, value in summary["resource_busy_ms"].items()}
    resources = []
    for name, resource in execution.engine.resources.items():
        resources.append({
            "resource": str(name), "work": str(serviced.get(name, 0)),
            "unit": resource.unit, "capacity_per_ms": resource.rate,
            "busy_ms": str(busy.get(name, 0)),
        })
    recipe_counts = {
        key: int(value) - int(before_recipes.get(key, 0))
        for key, value in execution.recipes_used.items()
        if int(value) - int(before_recipes.get(key, 0))
    }
    ledger = call.ledger or {}
    arguments = dict(ledger.get("arguments") or {})
    state = dict(ledger.get("state") or {})
    outputs = dict(ledger.get("outputs") or {})
    lengths = tuple(arguments.get("lengths") or ())
    chunks = int(arguments.get("chunks") or 0)
    derived = (
        ("sequence_count", len(lengths)),
        ("physical_tokens", sum(lengths)),
        ("chunks", chunks),
        ("chunk_head_tasks", chunks * int(
            arguments.get("heads") or arguments.get("v_heads") or 0)),
        ("state_rows", len(lengths)),
        ("expanded_operators", len(call.ops)),
        ("completed_tasks", completed_delta),
        ("state_reads", 1 if state.get("initial") else 0),
        ("state_writes", 1 if state.get("final") else 0),
        ("output_values", len(outputs)),
        ("trace_retained", retained_delta),
        ("trace_dropped", dropped_delta),
    )
    dense_stage = tuple(sorted(
        (name, value) for name, value in serviced.items() if value))
    peak = max(busy.values(), default=Fraction(0))
    tied = tuple(key for key, value in busy.items() if value == peak)
    dominant = tied[0] if tied else ""

    @dataclass(frozen=True)
    class _CallExecutionCost:
        name: str
        duration_ms: Any
        dominant_pipe: str
        tied_pipes: tuple
        work: tuple
        times: tuple
        stage_times: tuple
        materialization: tuple
        internal: tuple
        residency_bytes: Any
        derived: tuple
        formula: str
        fidelity: str
        stage_work: tuple

    duration = Fraction(str(step.completion.time)) - Fraction(str(release))
    semantic = tuple(sorted(
        (str(key), Fraction(str(value)) - Fraction(str(before_work.get(key, 0))))
        for key, value in dict(execution.work).items()))
    formula = (
        "F-STATE selected call executed once: "
        + ", ".join(f"{name}x{count}" for name, count in sorted(
            recipe_counts.items()))
        + f"; operators={len(call.ops)}; completed_tasks="
        + str(completed_delta))
    cost = _CallExecutionCost(
        name="ChunkGatedDeltaRuleFwd", duration_ms=duration,
        dominant_pipe=dominant, tied_pipes=tied,
        work=semantic, times=tuple(sorted(busy.items())),
        stage_times=(duration,), materialization=_materialization_from_ledger(
            ledger), internal=(), residency_bytes=Fraction(0),
        derived=derived, formula=formula,
        fidelity="isolated_device_time",
        stage_work=(dense_stage,) if dense_stage else ())
    trace = []
    program_rows = [
        row for row in tuple(engine.trace)
        if not str(row.get("task", "")).endswith((":schedule", ":ready"))
    ]
    program_completed = max(0, completed_delta - control_tasks)
    trace_rows = (program_rows[-limit:] if limit else ())
    for row in trace_rows:
        trace.append({key: (str(value) if isinstance(value, Fraction)
                            else value)
                      for key, value in dict(row).items()})
    retained = len(trace_rows)
    return cost, {
        "basis": "prepare_price_isolated_submission",
        "own_execution": created,
        "requested_trace_limit": limit,
        "trace_limit": limit,
        "trace": trace,
        "trace_dropped": max(0, program_completed - retained),
        "retained_trace_events": retained,
        "completed_tasks": program_completed,
        "control_tasks_excluded": control_tasks,
        "resources": resources,
        "memory": {
            "peak_allocated": dict(summary["peak_allocated"]),
            "live_allocated": dict(summary["live_allocated"]),
            "peak_allocated_per_unit": dict(
                summary["peak_allocated_per_unit"]),
        },
        "serviced": {key: str(value) for key, value in serviced.items()},
        "busy_ms": {key: str(value) for key, value in busy.items()},
        "work": {key: str(value) for key, value in semantic},
        "recipes": recipe_counts,
        "duration_ms": str(duration),
        "stage_work": [{"resource": key, "serviced": str(value)}
                       for key, value in dense_stage],
    }


def _execute_compiled_cost(cost: Any, compiled: Any, geometry: Any,
                           supply: Any, machine: Any, *,
                           execution: Any = None, trace_limit: Any = None,
                           name: str = "native_leaf") -> tuple[Any, dict]:
    """Submit one already-built family program through the shared Engine.

    Standalone execution is created only here, at the caller's outermost
    boundary.  An embedded caller passes its existing ``Execution``; no private
    engine is created.  The returned cost's duration/times are the actual event
    window, while work and stage_work remain the selected program's ledger.
    """
    from dataclasses import replace
    from fractions import Fraction
    from types import SimpleNamespace
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.composition import submit_cost
    from mindie_sim.schema.case import Software

    limit = 0 if trace_limit is None else int(trace_limit)
    if limit < 0:
        raise LeafAbiUnsupported("leaf_abi_trace_limit_invalid")
    created = execution is None
    if created:
        if machine is None:
            raise LeafAbiUnsupported("leaf_abi_execution_hardware_required")
        context = SimpleNamespace(machine=machine, runtime=Software())
        execution = Execution(
            context, CostModel(context), trace_limit=limit,
            include_submission=False)
    engine = execution.engine
    before_serviced = {key: float(value) for key, value in engine.serviced.items()}
    before_busy = {key: float(value) for key, value in engine.busy_ms.items()}
    before_completed = engine.completed_count
    before_dropped = engine.trace_dropped
    before_retained = len(engine.trace)
    release = 0.0 if created else engine.now
    bindings = {}
    suffix = "@r0:s0"
    for key, value in engine.resources.items():
        if not key.endswith(suffix):
            continue
        base = key.split("@", 1)[0]
        if base in {"mte1", "mte3", "fixpipe", "vector", "host"}:
            bindings[base] = key
        elif base == "mte2_aiv":
            bindings["mte2"] = key
        elif base == "mte2_aic":
            bindings.setdefault("mte2", key)
        elif base == "memory":
            bindings["hbm"] = key
        elif base == "matrix":
            bindings["matrix"] = key
    submitted = submit_cost(
        cost, execution, name=name, supply=supply, operator=compiled,
        geometry=geometry, resource_bindings=bindings or None)
    events = tuple(submitted.events)
    engine.advance(until=events, first=False)
    summary = engine.summary()
    completed_delta = int(summary["completed_tasks"]) - int(before_completed)
    retained_delta = max(0, int(summary["retained_trace_events"]) - before_retained)
    dropped_delta = int(summary["trace_dropped"]) - int(before_dropped)
    serviced = {str(key): Fraction(str(value)) - Fraction(str(before_serviced.get(key, 0.0)))
                for key, value in summary["serviced"].items()}
    busy = {str(key): Fraction(str(value)) - Fraction(str(before_busy.get(key, 0.0)))
            for key, value in summary["resource_busy_ms"].items()}
    end = max((event.time for event in events if event.time is not None),
              default=release)
    duration = Fraction(str(end)) - Fraction(str(release))
    peak = max(busy.values(), default=Fraction(0))
    tied = tuple(key for key, value in busy.items() if value == peak)
    actual = replace(
        cost, duration_ms=duration,
        times=tuple(sorted(busy.items())),
        stage_times=(duration,), dominant_pipe=(tied[0] if tied else ""),
        tied_pipes=tied)
    resources = [{
        "resource": str(name), "work": str(serviced.get(name, 0)),
        "unit": resource.unit, "capacity_per_ms": resource.rate,
        "busy_ms": str(busy.get(name, 0)),
    } for name, resource in engine.resources.items()]
    trace_rows = tuple(engine.trace)
    if not created and retained_delta:
        trace_rows = trace_rows[-retained_delta:]
    report = {
        "basis": "complete_engine_serviced",
        "own_execution": created,
        "trace_limit": int(engine.trace.maxlen or 0),
        "trace": [{key: (str(value) if isinstance(value, Fraction) else value)
                   for key, value in dict(row).items()} for row in trace_rows],
        "trace_dropped": dropped_delta,
        "retained_trace_events": retained_delta,
        "completed_tasks": completed_delta,
        "resources": resources,
        "memory": {
            "peak_allocated": dict(summary["peak_allocated"]),
            "live_allocated": dict(summary["live_allocated"]),
            "peak_allocated_per_unit": dict(
                summary["peak_allocated_per_unit"]),
        },
        "serviced": {key: str(value) for key, value in serviced.items()},
        "busy_ms": {key: str(value) for key, value in busy.items()},
        "work": {key: str(value) for key, value in cost.work},
        "duration_ms": str(duration),
        "stage_work": [{"resource": key, "serviced": str(value)}
                       for key, value in serviced.items() if value],
    }
    return actual, report


def _normalize_gdn_partition(inv: Any) -> dict:
    """One normalized partition for dense/packed/qsl/lengths inputs."""
    from mindie_sim.prediction.piecewise.native_gdn import (
        LeafAbiUnsupported, leaf_attr)
    raw_lengths = leaf_attr(inv, "lengths", None)
    raw_qsl = leaf_attr(inv, "cu_seqlens", None)
    if raw_lengths is None and raw_qsl is None:
        raise LeafAbiUnsupported("gdn_abi_sequence_partition_required")
    if raw_qsl is not None:
        if (isinstance(raw_qsl, (str, bytes))
                or not isinstance(raw_qsl, (list, tuple))
                or len(raw_qsl) < 2):
            raise LeafAbiUnsupported("gdn_abi_cu_seqlens_invalid")
        if any(isinstance(item, bool) or not isinstance(item, int)
               for item in raw_qsl):
            raise LeafAbiUnsupported("gdn_abi_cu_seqlens_not_int")
        qsl = tuple(raw_qsl)
        if qsl[0] != 0 or any(left > right
                              for left, right in zip(qsl, qsl[1:])):
            raise LeafAbiUnsupported("gdn_abi_cu_seqlens_not_monotonic")
    else:
        qsl = None
    if raw_lengths is not None:
        if (isinstance(raw_lengths, (str, bytes))
                or not isinstance(raw_lengths, (list, tuple))
                or not raw_lengths):
            raise LeafAbiUnsupported("gdn_abi_lengths_invalid")
        lengths = tuple(raw_lengths)
        if any(isinstance(item, bool) or not isinstance(item, int)
               or item < 1 for item in lengths):
            raise LeafAbiUnsupported("gdn_abi_lengths_not_positive_int")
        derived_qsl = [0]
        for length in lengths:
            derived_qsl.append(derived_qsl[-1] + length)
        if qsl is not None and qsl != tuple(derived_qsl):
            raise LeafAbiUnsupported("gdn_abi_lengths_cu_seqlens_mismatch")
        qsl = tuple(derived_qsl)
    else:
        lengths = tuple(right - left for left, right in zip(qsl, qsl[1:]))
        if any(length < 1 for length in lengths):
            raise LeafAbiUnsupported("gdn_abi_qsl_empty_sequence")
    return {
        "lengths": lengths, "cu_seqlens": qsl,
        "source": ("lengths" if raw_lengths is not None
                   else "cu_seqlens"),
        "sequence_count": len(lengths), "total_tokens": sum(lengths),
    }


def _gdn_physical_view(q_shape: list, partition: dict) -> dict:
    """Resolve dense [B,T,...] versus packed [1,sum(T),...] explicitly."""
    from mindie_sim.prediction.piecewise.native_gdn import LeafAbiUnsupported
    batch, axis_tokens = q_shape[0], q_shape[1]
    total = partition["total_tokens"]
    sequences = partition["sequence_count"]
    lengths = partition["lengths"]
    if batch == 1:
        if axis_tokens != total:
            raise LeafAbiUnsupported(
                "gdn_abi_packed_axis_must_equal_partition_total")
        return {
            "layout": "packed", "physical_batch": 1,
            "sequence_axis_tokens": total, "packed_tokens": total,
            "view_transform": "identity",
        }
    if batch == sequences and axis_tokens * batch == total and all(
            length == axis_tokens for length in lengths):
        return {
            "layout": "dense", "physical_batch": batch,
            "sequence_axis_tokens": axis_tokens, "packed_tokens": total,
            "view_transform": "explicit_contiguous_BxT_to_1xBT",
        }
    raise LeafAbiUnsupported(
        "gdn_abi_dense_physical_batch_and_T_do_not_match_partition")


def _gdn_abi_program(inv: Any, supply: Any, policy: Any,
                     phase: str, machine: Any = None,
                     trace_limit: Any = None,
                     execution: Any = None) -> dict[str, Any] | None:
    from mindie_sim.prediction.native_binding import NativeAbiRegion
    from mindie_sim.prediction.piecewise.native_gdn import (
        STATE_EFFECT_ORDER, STATE_SLOT_SSM, LeafAbiUnsupported,
        leaf_attr, leaf_dtype, leaf_int, leaf_shape, state_effect)
    if phase not in {"", "prefill"}:
        raise LeafAbiUnsupported(f"gdn_abi_phase_not_prefill:{phase}")
    if machine is None:
        raise LeafAbiUnsupported("gdn_abi_hardware_carrier_required")
    partition = _normalize_gdn_partition(inv)
    lengths = partition["lengths"]
    qsl = list(partition["cu_seqlens"])
    sequence_count = partition["sequence_count"]
    total_tokens = partition["total_tokens"]
    q_shape = leaf_shape(inv, "q")
    k_shape = leaf_shape(inv, "k")
    v_shape = leaf_shape(inv, "v")
    g_shape = leaf_shape(inv, "g")
    beta_shape = leaf_shape(inv, "beta")
    state_shape = leaf_shape(inv, "initial_state")
    o_shape = leaf_shape(inv, "o")
    final_shape = leaf_shape(inv, "final_state")
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise LeafAbiUnsupported("gdn_abi_qkv_rank_invalid")
    physical = _gdn_physical_view(q_shape, partition)
    physical_batch = physical["physical_batch"]
    axis_tokens = physical["sequence_axis_tokens"]
    key_heads, dk = q_shape[2], q_shape[3]
    if k_shape != q_shape:
        raise LeafAbiUnsupported("gdn_abi_k_shape_mismatch")
    value_heads, dv = v_shape[2], v_shape[3]
    if v_shape[:2] != [physical_batch, axis_tokens]:
        raise LeafAbiUnsupported("gdn_abi_v_batch_tokens_mismatch")
    if g_shape != [physical_batch, axis_tokens, value_heads] or beta_shape != g_shape:
        raise LeafAbiUnsupported("gdn_abi_gate_shape_mismatch")
    if state_shape != [sequence_count, value_heads, dv, dk]:
        raise LeafAbiUnsupported("gdn_abi_state_rows_must_equal_sequence_count")
    if o_shape != [physical_batch, axis_tokens, value_heads, dv]:
        raise LeafAbiUnsupported("gdn_abi_output_shape_mismatch")
    if final_shape != state_shape:
        raise LeafAbiUnsupported("gdn_abi_final_state_shape_mismatch")
    for name, dtype in (("q", "bf16"), ("k", "bf16"), ("v", "bf16"),
                        ("g", "fp32"), ("beta", "bf16"),
                        ("initial_state", "fp32"), ("o", "bf16"),
                        ("final_state", "fp32")):
        if leaf_dtype(inv, name) != dtype:
            raise LeafAbiUnsupported(f"gdn_abi_{name}_dtype_mismatch")
    chunk = leaf_int(inv, "chunk_size")
    scale = float(leaf_attr(inv, "scale"))
    if chunk < 1:
        raise LeafAbiUnsupported("gdn_abi_length_invalid")
    if (chunk, dk, dv) != (64, 128, 128):
        raise LeafAbiUnsupported(
            f"gdn_abi_selected_call_requires_chunk64_k128_v128:"
            f"{chunk},{dk},{dv}")
    booleans = {}
    for name in ("use_qk_l2norm_in_kernel", "use_exp2", "disable_recompute",
                 "state_v_first", "output_final_state", "use_gate_in_kernel",
                 "use_beta_sigmoid_in_kernel", "allow_neg_eigval",
                 "return_intermediate_states"):
        value = leaf_attr(inv, name)
        if not isinstance(value, bool):
            raise LeafAbiUnsupported(f"gdn_abi_attr_not_bool:{name}")
        booleans[name] = value
    if not booleans["output_final_state"]:
        # call_gdn.make_call("scan_prefill") currently fixes this parameter to
        # true.  The leaf must not fabricate the conditional final-state output.
        raise LeafAbiUnsupported(
            "gdn_conditional_output_requires_f_state_call_gdn:"
            "make_call.scan_prefill.output_final_state_is_true")
    from mindie_sim.prediction import call_gdn
    params = {
        "batch": sequence_count, "tokens": total_tokens,
        "lengths": list(lengths), "cu_seqlens": qsl,
        "v_heads": value_heads, "k_heads": key_heads,
        "k_dim": dk, "v_dim": dv, "chunk": chunk, "chunk_size": chunk,
        "phase": "prefill", "scale": scale, "a_log": None, "dt_bias": None,
        "input_dtype": "bf16", "gate_dtype": "fp32", "beta_dtype": "bf16",
        "state_dtype": "fp32", "layout": "BSND", **booleans,
    }
    try:
        call = call_gdn.make_call("scan_prefill", params)
    except ModelingError as error:
        raise LeafAbiUnsupported(getattr(error, "reason", str(error))) from None
    cost, execution_report = _call_execution_cost(
        call, machine, supply=supply, policy=policy,
        trace_limit=trace_limit, execution=None)
    ledger = call.ledger or {}
    arguments = dict(ledger.get("arguments") or {})
    state = dict(ledger.get("state") or {})
    outputs = dict(ledger.get("outputs") or {})
    device = call.device_program
    effects = []
    if state.get("initial") is not None:
        effects.append(state_effect(
            order=STATE_EFFECT_ORDER["chunk_compute"], direction="read",
            slot="gdn_chunk_state", tensor="initial_state",
            rows=sequence_count, dtype="fp32",
            role="recurrent_initial_state_read", shape=state_shape,
            source="fsm_call.ledger.state.initial"))
    if booleans["output_final_state"] and "final_state" in outputs:
        effects.append(state_effect(
            order=STATE_EFFECT_ORDER["chunk_compute"], direction="write",
            slot="gdn_chunk_state", tensor="final_state",
            rows=sequence_count, dtype="fp32",
            role="recurrent_final_state_write", shape=final_shape,
            requires_attr="output_final_state",
            requires_value=True,
            source="fsm_call.ledger.outputs.final_state"))
    chunks = int(arguments.get("chunks") or 0)
    qsl = list(arguments.get("cu_seqlens") or ())
    # Concrete state ports for the gather -> local initial -> H/final ->
    # scatter chain.  They carry sequence, slot, version, producer and the
    # last-use hook; execution owners consume these records instead of
    # re-deriving a second partition.
    state_contract = dict(state.get("contract") or {})
    state_records = list(state.get("effects") or ())
    active_slots = tuple(state_contract.get("active_slots") or ())
    targets = tuple(state_contract.get("targets") or ())
    queries = tuple(state_contract.get("queries") or ())
    read_versions = tuple(state_contract.get("read_versions") or ())
    publish_versions = tuple(state_contract.get("publish_versions") or ())
    local_versions = tuple(state_contract.get("local_commit_versions") or ())
    ports = []
    for index in range(sequence_count):
        target = targets[index] if index < len(targets) else f"seq{index}"
        writes = [item for item in state_records
                  if item.get("target") == target
                  and item.get("action") == "write"]
        reads = [item for item in state_records
                 if item.get("target") == target
                 and item.get("action") == "read"]
        ports.append({
            "sequence": index,
            "request_slot": (active_slots[index]
                             if index < len(active_slots) else index),
            "target": target,
            "queries": (queries[index] if index < len(queries) else None),
            "read_version": (read_versions[index]
                             if index < len(read_versions) else None),
            "write_version": (writes[-1]["version"] if writes else None),
            "publish_version": (publish_versions[index]
                                if index < len(publish_versions) else None),
            "local_commit_version": (local_versions[index]
                                     if index < len(local_versions) else None),
            "producer": "chunk_gated_delta_rule_fwd",
            "consumer": "scatter_ssm_states",
            "last_use": "call_return_then_scatter",
            "last_reader": "scatter_ssm_states",
            "terminal": bool(state_contract.get("accepted_checkpoint")
                            == "final"),
            "storage": state_contract.get("state_layout"),
            "state_dtype": state_contract.get("state_dtype"),
            "span": state_contract.get("query_ranges"),
            "commit_policy": state_contract.get("commit_policy"),
        })
    typed = {
        "dtype": "bf16", "layout": "BSND", "phase": "prefill",
        "shapes": {"q": [key_heads, dk], "k": [key_heads, dk],
                   "v": [value_heads, dv], "g": [value_heads],
                   "beta": [value_heads],
                   "state": [value_heads, dv, dk]},
        "attrs": {
            "chunk_size": chunk, "scale": scale,
            "use_qk_l2norm_in_kernel": booleans["use_qk_l2norm_in_kernel"],
            "use_exp2": booleans["use_exp2"],
            "disable_recompute": booleans["disable_recompute"],
            "state_v_first": booleans["state_v_first"],
            "output_final_state": booleans["output_final_state"],
        },
        "occurrence_attrs": {"batch": "positive_int",
                             "tokens": "positive_int",
                             "lengths": "int_list"},
        "rank_width": 3,
    }
    occurrence = {
        "dtype": "bf16", "layout": "BSND", "phase": "prefill",
        "shapes": {"q": q_shape, "k": k_shape, "v": v_shape,
                   "g": g_shape, "beta": beta_shape,
                   "state": state_shape, "o": o_shape,
                   "final_state": final_shape},
        "attrs": {**typed["attrs"], **booleans,
                  "batch": sequence_count, "physical_batch": physical_batch,
                  "tokens": total_tokens,
                  "physical_axis_tokens": axis_tokens,
                  "physical_layout": physical["layout"],
                  "view_transform": physical["view_transform"],
                  "lengths": [int(item) for item in lengths],
                  "cu_seqlens": qsl, "chunks": chunks,
                  "chunk_head_tasks": chunks * value_heads},
    }
    return {
        "cost": cost, "call": call, "pricing": execution_report,
        "phase": "prefill", "kind": "gdn_linear_attn",
        "environment": {
            "sequence_count": sequence_count,
            "physical_batch": physical_batch, "tokens": total_tokens,
            "physical_axis_tokens": axis_tokens,
            "physical_layout": physical["layout"],
            "view_transform": physical["view_transform"],
            "lengths": list(lengths), "cu_seqlens": qsl, "chunks": chunks,
            "chunk_head_tasks": chunks * value_heads,
            "state_rows": sequence_count,
            "kernels": list(device.get("device_kernels") or ()),
        },
        "region": NativeAbiRegion(
            op_id="A3:recurrent:gated_delta", family="recurrent",
            recipe="gated_delta", source_boundary="call",
            dtype="bf16", layout="BSND",
            metadata=tuple(sorted({
                "chunk_size": chunk, "key_heads": key_heads,
                "value_heads": value_heads, "dk": dk, "dv": dv,
                "layout": "BSND",
            }.items()))),
        "typed": typed,
        "occurrence": occurrence,
        "effects": tuple(effects),
        "unresolved": ("python_caller_module_not_retained",
                       "conditional_output_requires_f_state_call_gdn"),
        "program": {
            "mode": "chunked_prefill", "family": "recurrent",
            "common_program": "F-STATE.selected_scan_prefill_call",
            "members": ["prepare", "gated_delta"], "phase": "prefill",
            "state_slot": "gdn_chunk_state",
            "state_effects": list(effects),
            "state_contract": state.get("contract"),
            "state_records": list(state.get("effects") or ()),
            "state_ports": ports,
            "state_value": {
                "initial": state.get("initial"),
                "final": state.get("final"),
                "use_initial_state": state.get("use_initial_state"),
                "output_final_state": state.get("output_final_state"),
            },
            "state_effect_sequence": {
                "slot": STATE_SLOT_SSM,
                "ordered": [
                    {"order": STATE_EFFECT_ORDER["gather_read"],
                     "owner": "gather_ssm_states", "direction": "read"},
                    {"order": STATE_EFFECT_ORDER["chunk_compute"],
                     "owner": "chunk_gated_delta_rule_fwd",
                     "direction": "read_write"},
                    {"order": STATE_EFFECT_ORDER["scatter_write"],
                     "owner": "scatter_ssm_states", "direction": "write"},
                ],
            },
            "case_basis": "hardware_runtime_context_only; no model/workload",
            "execution_basis": "complete_engine_serviced",
            "implementation_policy": {
                "cores": int(getattr(policy, "cores", 0) or 0),
                "mt": int(getattr(policy, "mt", 0) or 0),
            },
            "call_report": {
                "boundary": "device_work_of_complete_call",
                "own_execution": execution_report["own_execution"],
                "duration_ms": execution_report["duration_ms"],
                "operator_occurrences": len(call.ops),
                "execution": {
                    "work": execution_report["work"],
                    "recipes": execution_report["recipes"],
                    "completed_tasks": execution_report["completed_tasks"],
                    "serviced": execution_report["serviced"],
                },
                "resources": execution_report["resources"],
                "memory": execution_report["memory"],
                "stage_work": execution_report["stage_work"],
                "stage_work_basis": "complete_engine_serviced",
                "trace": {
                    "limit": execution_report["trace_limit"],
                    "retained": execution_report["retained_trace_events"],
                    "dropped": execution_report["trace_dropped"],
                    "events": execution_report["trace"][:128],
                },
            },
            "call_program": {
                "arguments": arguments,
                "inputs": dict(device.get("inputs") or {}),
                "outputs": dict(device.get("outputs") or {}),
                "device_kernels": list(device.get("device_kernels") or ()),
                "materializations": list(device.get("materializations") or ()),
                "effects": list(device.get("effects") or ()),
            },
            "native_kernel_counts": {
                "Cast": 1 if "beta_fp32" in execution_report["recipes"] else 0,
                "Transpose": sum(
                    1 for name in ("q_head_first", "k_head_first",
                                   "v_head_first")
                    if name in execution_report["recipes"]),
                "Prepare32": sum(
                    1 for name in execution_report["recipes"]
                    if "gdn_prepare" in name),
                "H24": 1 if "chunk_fwd_h" in execution_report["recipes"] else 0,
                "O32": sum(
                    1 for name in execution_report["recipes"]
                    if "gdn_output" in name),
            },
            "partition": {
                "sequence_count": sequence_count,
                "physical_batch": physical_batch,
                "physical_axis_tokens": axis_tokens,
                "physical_layout": physical["layout"],
                "view_transform": physical["view_transform"],
                "lengths": list(lengths),
                "cu_seqlens": qsl,
                "chunks": chunks,
                "chunk_head_tasks": chunks * value_heads,
                "state_rows": sequence_count,
                "partition_owner": "F-STATE.call_gdn",
            },
            "implementation": {
                "wrapper": "fla_npu.ops.ascendc._stable."
                           "npu_chunk_gated_delta_rule_fwd",
                "acl_signature": GDN_ABI_ENTRY,
                "source_revision": GDN_ABI_SOURCE_REVISION,
                "source_sha256": call_gdn.PREFILL_SOURCE["sha256"],
            },
            "binding_relation": {
                "semantic_region": "A3:recurrent:gated_delta",
                "relation": "one_to_many",
                "members": ["prepare", "gated_delta"],
                "sibling_calls": ["gather_ssm_states", "scatter_ssm_states"],
                "charge": "call",
            },
            "formula": cost.formula,
            "work": {key: str(value) for key, value in cost.work},
            "derived": {key: str(value) for key, value in cost.derived},
        },
    }


def simulate_independent(request: Any, **kwargs: Any) -> dict:
    """Predict the GDN chunk call from its independent native input."""
    return simulate_leaf(request, gdn_abi_spec(), _gdn_abi_program, **kwargs)


def simulate_declared_abi(request: Any, **kwargs: Any) -> dict:
    """Dispatch one declared leaf ABI entry to its leaf entry.

    T12 owns the public catalogue and result schema; this helper is the
    leaf-side map so that catalogue can reuse the exact same implementations
    instead of rebuilding a second mapper.  The parent ScatterPaCache API is
    dispatched to its explicit gap entry, not to the child kernels.
    """
    from . import (
        native_gather_ssm_states, native_l2norm_fwd_kernel2,
        native_r64_aclnn_cat, native_r64_layer_norm_with_impl_mode,
        native_r64_muls_add, native_r64_scatter_pa_cache,
        native_r64_transpose_batch_matmul, native_r64_triton_rope_siso,
        native_scatter_ssm_states, native_split_qkv_rmsnorm_mrope)
    entry = _leaf_entry(request)
    mapping = {
        native_gather_ssm_states.KERNEL_NAME:
            native_gather_ssm_states.simulate_independent,
        native_scatter_ssm_states.ENTRY:
            native_scatter_ssm_states.simulate_independent,
        native_r64_scatter_pa_cache.CHILD_ENTRY:
            native_r64_scatter_pa_cache.simulate_independent,
        native_r64_scatter_pa_cache.PARENT_ENTRY:
            native_r64_scatter_pa_cache.simulate_parent,
        native_l2norm_fwd_kernel2.ENTRY:
            native_l2norm_fwd_kernel2.simulate_independent,
        native_r64_triton_rope_siso.ENTRY:
            native_r64_triton_rope_siso.simulate_independent,
        native_r64_aclnn_cat.ENTRY:
            native_r64_aclnn_cat.simulate_independent,
        native_r64_muls_add.ENTRY:
            native_r64_muls_add.simulate_independent,
        native_r64_layer_norm_with_impl_mode.ENTRY:
            native_r64_layer_norm_with_impl_mode.simulate_independent,
        native_r64_transpose_batch_matmul.ENTRY:
            native_r64_transpose_batch_matmul.simulate_independent,
        native_split_qkv_rmsnorm_mrope.ENTRY:
            native_split_qkv_rmsnorm_mrope.simulate_independent,
        GDN_ABI_ENTRY: simulate_independent,
    }
    handler = mapping.get(entry or "")
    if handler is None:
        raise ModelingError(f"leaf_abi_entry_not_declared:{entry}")
    return handler(request, **kwargs)
