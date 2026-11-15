"""Project declared operator records into typed per-value structure facts.

[Role] Preserve existing typed outputs, source/target views, transforms, scale
  associations, semantic attrs and state facts before falling back to the
  compact comparison tuple.
[Invariants] A comparison tuple adds only fields it declares. A lone native
  name is a correspondence, not a selection. Missing layout/shape/ABI becomes a
  named gap; measured shapes never gate recognition.
[Timeouts & Locks] Bounded metadata arithmetic; no I/O, locks or device access.
[Side-Effect Policy] Pure records; no profiler, capture or timing dependency.
"""
import math
import re

from mindie_sim.schema.common import dtype_bytes as _dtype_bytes


# --- Typed value projection -------------------------------------------------
# A native contract is a compact comparison tuple.  Consumers that bind an
# actual call also need one record per value with its own representation.  The
# projection preserves existing typed records/views/state; it never invents a
# tensor that neither the contract nor the declared metadata contains.

_TENSOR_LAYOUTS = frozenset({
    "ND", "NCL", "NZ", "NDCHW", "NHWC", "TND", "BNSD", "NBSD", "NDHWC",
})
_QUANT_DTYPES = frozenset({
    "int8", "int4", "int2", "fp8_e4m3", "fp8_e5m2", "mxfp4", "fp4", "fp4_e2m1",
})


def _shape(value):
    if (isinstance(value, (list, tuple))
            and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value)):
        return [int(dim) for dim in value]
    return None


def _is_tensor(value):
    """A declared ``(shape, dtype, layout)`` triple, never a matrix tuple."""
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and _shape(value[0]) is not None
            and (value[1] is None or isinstance(value[1], str))
            and (value[2] is None
                 or (isinstance(value[2], str) and value[2].upper() in _TENSOR_LAYOUTS)))


def _is_tensor_seq(value):
    return (isinstance(value, (list, tuple)) and bool(value)
            and all(_is_tensor(item) for item in value))


def _nbytes(dtype, shape):
    if not dtype or shape is None:
        return None
    try:
        return float(math.prod(shape) * _dtype_bytes(dtype))
    except (KeyError, ValueError, TypeError):
        return None


def _value(name, role, *, producer="", output=None, dtype=None, shape=None,
           storage_shape=None, strides=None, offset=0, layout_kind=None,
           quant="", attrs=None, materialized=None, contract="", logical_bytes=None,
           storage_bytes=None, write_bytes=None, declared_bytes=None,
           storage="", alias_of="", state_version=None, binding=None,
           binding_name=None, storage_unknown=False):
    """One per-value record with logical/storage representation and provenance."""
    logical = _shape(shape) if shape is not None else None
    physical = (None if storage_unknown else
                _shape(storage_shape) if storage_shape is not None else logical)
    if physical is None and not storage_unknown:
        physical = logical
    record_attrs = {"source": attrs.get("source", "")} if attrs else {"source": ""}
    if attrs:
        record_attrs.update(attrs)
    if logical_bytes is None:
        logical_bytes = _nbytes(dtype, logical)
    if storage_bytes is None and not storage_unknown:
        storage_bytes = _nbytes(dtype, physical)
    return {
        "name": name,
        "output": output if output is not None else name,
        "role": role,
        "producer": producer,
        "shape": logical,
        "logical_shape": logical,
        "storage_shape": physical,
        "strides": _shape(strides) if strides is not None else None,
        "offset": int(offset or 0),
        "dtype": dtype,
        "layout": layout_kind,
        "layout_kind": layout_kind,
        "quant": quant if quant else (dtype if dtype in _QUANT_DTYPES else ""),
        "attrs": record_attrs,
        "materialized": materialized,
        "contract": contract,
        "bytes": logical_bytes,
        "storage_bytes": storage_bytes,
        "write_bytes": write_bytes,
        "declared_bytes": declared_bytes,
        "storage": storage,
        "alias_of": alias_of,
        "state_version": state_version,
        "binding": binding,
        "binding_name": binding_name,
    }


def _tensor_record(name, role, spec, path, *, attrs=None, producer="",
                   output=None, layout_kind=None):
    shape, dtype, layout = spec
    record_attrs = {"source": path}
    if attrs:
        record_attrs.update(attrs)
    return _value(
        name, role, producer=producer, output=output, dtype=dtype, shape=shape,
        storage_shape=shape, layout_kind=layout if layout_kind is None else layout_kind,
        attrs=record_attrs, contract="native_tensor_contract")



def _named_tensors(prefix, role, specs, path):
    return [_tensor_record(f"{prefix}{index}", role, spec, f"{path}[{index}]")
            for index, spec in enumerate(specs)
            if not (spec[1] is None and spec[2] is None)]


def _quantization(operator):
    meta = operator.get("metadata", {})
    fields = {}
    for key in ("quantization", "quant_category", "activation_format", "weight_format",
                "output_format", "input_format", "compute_format",
                "activation_scale_format", "weight_scale_format"):
        if meta.get(key) not in (None, ""):
            fields[key] = meta[key]
    if not fields:
        return None
    activation = fields.get("activation_format") or fields.get("input_format")
    weight = fields.get("weight_format")
    output = fields.get("output_format")
    scheme = None
    if activation and weight:
        if activation == weight == "int8":
            scheme = "w8a8"
        elif activation == "int8" and weight == "int4":
            scheme = "w4a8"
        elif weight == "mxfp4":
            scheme = "mxfp4"
        elif activation.startswith("fp8") and weight.startswith("fp8"):
            scheme = "fp8"
        elif activation == weight == "bf16":
            scheme = "bf16"
        else:
            scheme = f"a_{activation}_w_{weight}"
    elif output in _QUANT_DTYPES:
        # A standalone quantize/cast value, not a matrix weight scheme.
        scheme = f"dynamic_{output}"
    return {"scheme": scheme, **fields}


def _matrix_values(contract):
    kind, geometry = contract[0], contract[1]
    if kind == "matrix":
        m, n, k = geometry
        activation, weight, output = contract[2], contract[3], contract[4]
        scale_shape, scale_format = contract[5], contract[6]
        inputs = [
            _tensor_record("activation", "input", ([m, k], activation, "ND"),
                           "contract[2]", attrs={"matrix_geometry": [m, n, k],
                                                 "shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("weight", "input", ([n, k], weight, "ND"),
                           "contract[3]", attrs={"matrix_geometry": [m, n, k],
                                                 "shape_basis": "matrix_m_n_k_convention"}),
        ]
        if scale_shape:
            inputs.append(_tensor_record(
                "activation_scale", "input", (scale_shape, scale_format, "ND"),
                "contract[5]", attrs={"not_a_payload_value": True}))
        outputs = [_tensor_record("output", "output", ([m, n], output, "ND"),
                                  "contract[4]", attrs={"matrix_geometry": [m, n, k]})]
        return inputs, outputs, [], {"matrix_geometry": [m, n, k]}
    if kind == "matrix_bias":
        m, n, k = geometry
        activation, weight, output = contract[2], contract[3], contract[4]
        bias_shape, bias_dtype = contract[5], contract[6]
        inputs = [
            _tensor_record("activation", "input", ([m, k], activation, "ND"),
                           "contract[2]", attrs={"shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("weight", "input", ([n, k], weight, "ND"),
                           "contract[3]", attrs={"shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("bias", "input", (bias_shape, bias_dtype, "ND"), "contract[5]"),
        ]
        outputs = [_tensor_record("output", "output", ([m, n], output, "ND"), "contract[4]")]
        return inputs, outputs, [], {"matrix_geometry": [m, n, k]}
    if kind == "grouped_matrix":
        bank, m, n, k = geometry
        activation, weight, output = contract[2], contract[3], contract[4]
        scale_shape, scale_format = contract[5], contract[6]
        inputs = [
            _tensor_record("activation", "input", ([m, k], activation, "ND"),
                           "contract[2]", attrs={"shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("weight", "input", ([bank, n, k], weight, "ND"),
                           "contract[3]", attrs={"expert_bank": bank,
                                                 "shape_basis": "grouped_matrix_m_n_k_convention"}),
        ]
        if scale_shape:
            inputs.append(_tensor_record(
                "activation_scale", "input", (scale_shape, scale_format, "ND"),
                "contract[5]", attrs={"not_a_payload_value": True}))
        outputs = [_tensor_record("output", "output", ([m, n], output, "ND"), "contract[4]")]
        return inputs, outputs, [], {"expert_bank": bank, "matrix_geometry": [m, n, k]}
    return None


def _common_values(contract):
    """Project the contract shapes that share an unambiguous tensor order."""
    kind = contract[0]
    if kind == "native_tensor_abi":
        if len(contract) == 4 and _is_tensor_seq(contract[2]) and _is_tensor_seq(contract[3]):
            return (_named_tensors("input", "input", contract[2], "contract[2]"),
                    _named_tensors("output", "output", contract[3], "contract[3]"), [], {})
        return None
    if kind == "layernorm_active":
        # x, weight, bias, output; the native boundary is one row-wise call.
        if len(contract) == 6 and all(_is_tensor(item) for item in contract[2:]):
            inputs = [_tensor_record("x", "input", contract[2], "contract[2]"),
                      _tensor_record("weight", "input", contract[3], "contract[3]"),
                      _tensor_record("bias", "input", contract[4], "contract[4]")]
            outputs = [_tensor_record("output", "output", contract[5], "contract[5]")]
            return inputs, outputs, [], {}
        return None
    if kind == "attention_active_qkv":
        if (len(contract) == 4 and len(contract[2]) == 3
                and len(contract[3]) == 1
                and _is_tensor_seq(contract[2]) and _is_tensor_seq(contract[3])):
            inputs = [_tensor_record(name, "input", spec, f"contract[2][{index}]")
                      for index, (name, spec) in enumerate(zip(("query", "key", "value"),
                                                               contract[2]))]
            outputs = [_tensor_record("output", "output", contract[3][0], "contract[3][0]")]
            return inputs, outputs, [], {}
        return None
    if kind in {"sparse_attention_active", "quant_lightning_indexer_active",
                "lightning_indexer_active"}:
        # (query/weights..., cache heads..., primary output); the cache may be
        # repeated for key and value, so each entry keeps its own index.
        if len(contract) == 5 and _is_tensor_seq(contract[2]) \
                and _is_tensor_seq(contract[3]) and _is_tensor(contract[4]):
            inputs = (_named_tensors("input", "input", contract[2], "contract[2]")
                      + _named_tensors("cache", "input", contract[3], "contract[3]"))
            outputs = [_tensor_record("output", "output", contract[4], "contract[4]")]
            return inputs, outputs, [], {}
        return None
    if kind == "attention_paged_active_qkv":
        if (len(contract) == 6 and _is_tensor(contract[2])
                and _is_tensor_seq(contract[3]) and _is_tensor(contract[4])):
            inputs = [_tensor_record("query", "input", contract[2], "contract[2]")]
            inputs += _named_tensors("cache", "input", contract[3], "contract[3]")
            outputs = [_tensor_record("output", "output", contract[4], "contract[4]")]
            return inputs, outputs, [], {"batch_count": contract[5]}
        return None
    if kind == "attention_paged_mla_qk_rope":
        if (len(contract) == 6 and _is_tensor_seq(contract[2])
                and _is_tensor_seq(contract[3]) and _is_tensor(contract[4])):
            inputs = (_named_tensors("query", "input", contract[2], "contract[2]")
                      + _named_tensors("cache", "input", contract[3], "contract[3]"))
            outputs = [_tensor_record("output", "output", contract[4], "contract[4]")]
            return inputs, outputs, [], {"batch_count": contract[5]}
        return None
    if kind == "mla_prolog_active":
        if (len(contract) == 6 and _is_tensor(contract[2])
                and _is_tensor_seq(contract[3]) and _is_tensor_seq(contract[4])
                and _is_tensor_seq(contract[5])):
            inputs = ([_tensor_record("input", "input", contract[2], "contract[2]")]
                      + _named_tensors("parameter", "input", contract[3], "contract[3]")
                      + _named_tensors("cache", "input", contract[4], "contract[4]"))
            outputs = _named_tensors("output", "output", contract[5], "contract[5]")
            return inputs, outputs, [], {}
        return None
    if kind == "dynamic_quant_active":
        if len(contract) == 5 and all(_is_tensor(item) for item in contract[2:]):
            return ([_tensor_record("x", "input", contract[2], "contract[2]")],
                    [_tensor_record("output", "output", contract[3], "contract[3]"),
                     _tensor_record("scale", "output", contract[4], "contract[4]",
                                    attrs={"not_a_payload_value": True})], [], {})
        return None
    if kind == "add_rmsnorm_dynamic_quant_active":
        if (len(contract) == 5 and _is_tensor_seq(contract[2])
                and _is_tensor_seq(contract[3])):
            names = ("x", "residual", "gamma", "beta")
            inputs = [_tensor_record(names[index], "input", spec, f"contract[2][{index}]",
                                     attrs={"not_a_payload_value": names[index] == "beta"})
                      for index, spec in enumerate(contract[2])
                      if not (spec[1] is None and spec[2] is None)]
            outputs = _named_tensors("output", "output", contract[3], "contract[3]")
            return inputs, outputs, [], {}
        return None
    if kind == "rmsnorm_mxquant_active":
        if len(contract) == 6 and all(_is_tensor(item) for item in contract[2:]):
            return ([_tensor_record("x", "input", contract[2], "contract[2]"),
                     _tensor_record("gamma", "input", contract[3], "contract[3]")],
                    [_tensor_record("output", "output", contract[4], "contract[4]"),
                     _tensor_record("scale", "output", contract[5], "contract[5]",
                                    attrs={"not_a_payload_value": True})], [], {})
        return None
    if kind in {"swiglu", "relu"}:
        if len(contract) == 3:
            spec = (contract[1], contract[2], "ND")
            return ([_tensor_record("input", "input", spec, "contract[1:3]")],
                    [_tensor_record("output", "output", spec, "contract[1:3]")], [], {})
        return None
    if kind == "slice_copy":
        if len(contract) == 5:
            source = (contract[1], contract[3], "ND")
            target = (contract[2], contract[4], "ND")
            return ([_tensor_record("input", "input", source, "contract[1,3]")],
                    [_tensor_record("output", "output", target, "contract[2,4]")], [], {})
        return None
    if kind == "transpose_batchmatmul_active":
        if len(contract) == 6:
            batch, rows, n, k = contract[2]
            activation = ([batch, rows, k], contract[3], "ND")
            weight = ([batch, n, k], contract[4], "ND")
            output = ([batch, rows, n], contract[5], "ND")
            return ([_tensor_record("input", "input", activation, "contract[2:5]",
                                    attrs={"shape_basis": "batch_rows_n_k_convention"}),
                     _tensor_record("weight", "input", weight, "contract[2:5]",
                                    attrs={"shape_basis": "batch_rows_n_k_convention"})],
                    [_tensor_record("output", "output", output, "contract[2:5]")], [], {})
        return None
    if kind == "causal_conv1d_active":
        if len(contract) == 7 and all(_is_tensor(item) for item in contract[2:]):
            inputs = [
                _tensor_record("x", "input", contract[2], "contract[2]"),
                _tensor_record("weight", "input", contract[3], "contract[3]"),
                _tensor_record("history", "input", contract[5], "contract[5]",
                               attrs={"state": True}),
            ]
            if not (contract[4][1] is None and contract[4][2] is None):
                inputs.append(_tensor_record("bias", "input", contract[4], "contract[4]"))
            outputs = [_tensor_record("output", "output", contract[6], "contract[6]")]
            return inputs, outputs, [], {}
        return None
    if kind == "kv_cache_store_active_elements":
        if len(contract) == 4 and len(contract[2]) == 2:
            values = []
            for name, (elements, dtype) in zip(("key", "value"), contract[2]):
                values.append(_value(
                    name, "input", dtype=dtype, shape=None,
                    attrs={"source": "contract[2]", "elements": elements,
                           "shape_basis": "active_element_count_only"},
                    contract="active_element_count"))
            return values, [], [], {"slot_shape": list(contract[3])}
        return None
    if kind in {"rmsnorm", "add_rmsnorm"}:
        if len(contract) == 4:
            spec = (contract[1], contract[2], "ND")
            inputs = [_tensor_record("x", "input", spec, "contract[1:3]")]
            if contract[3]:
                inputs.append(_tensor_record("weight", "input",
                                             ((contract[1][-1],), contract[2], "ND"),
                                             "contract[1,3]"))
            outputs = [_tensor_record("output", "output", spec, "contract[1:3]")]
            return inputs, outputs, [], {}
        return None
    return None


def _scan_tensors(value, path):
    """Collect declared tensor triples with their structural path."""
    found = []
    if _is_tensor(value):
        if not (value[1] is None and value[2] is None):
            found.append((path, value))
        return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_scan_tensors(item, f"{path}[{index}]"))
    return found


def _declared_metadata_values(operator):
    """Explicit operator shapes, used only when they name their own value."""
    meta, dtype = operator.get("metadata", {}), operator.get("dtype")
    layout = _layout_hint(operator)
    dtypes = {
        "input": meta.get("input_dtype") or meta.get("activation_format") or dtype,
        "weight": meta.get("weight_dtype") or meta.get("weight_format") or dtype,
        "output": meta.get("output_dtype") or meta.get("output_format") or dtype,
    }
    inputs, outputs = [], []
    for name, key, role in (("input", "input_shape", "input"),
                            ("weight", "weight_shape", "input"),
                            ("output", "output_shape", "output")):
        shape = _shape(meta.get(key))
        if shape is None:
            continue
        value = _value(
            name, role, producer=operator.get("op_id", "") if role == "output" else "",
            output=name, dtype=dtypes[name], shape=shape, storage_shape=shape,
            layout_kind=layout,
            attrs={"source": f"operator.metadata.{key}",
                   "layout_source": "operator.layout" if layout else "not_declared"},
            contract="declared_operator_value")
        (outputs if role == "output" else inputs).append(value)
    bias_shape = _shape(meta.get("bias_shape"))
    if bias_shape is not None and meta.get("bias_dtype"):
        inputs.append(_value(
            "bias", "input", output="bias", dtype=meta["bias_dtype"],
            shape=bias_shape, storage_shape=bias_shape, layout_kind=layout,
            attrs={"source": "operator.metadata.bias_shape",
                   "layout_source": "operator.layout" if layout else "not_declared"},
            contract="declared_operator_value"))
    return inputs, outputs


_EXPLICIT_LAYOUT_KINDS = frozenset({
    "native_tensor_abi", "layernorm_active", "attention_active_qkv",
    "sparse_attention_active", "quant_lightning_indexer_active",
    "lightning_indexer_active", "attention_paged_active_qkv",
    "attention_paged_mla_qk_rope", "mla_prolog_active", "dynamic_quant_active",
    "add_rmsnorm_dynamic_quant_active", "rmsnorm_mxquant_active",
})
_SEMANTIC_ATTR_KEYS = (
    "norm_eps", "epsilon", "norm_has_weight", "norm_has_bias",
    "norm_weight_offset", "norm_gamma_runtime_format", "norm_gamma_storage_format",
    "reduction_width", "activation_kind", "activation_beta",
    "activation_linear_beta", "activation_compute_dtype", "activation_group_size",
    "causal", "window", "scale", "state_dtype", "conv_state_dtype",
    "conv_kernel", "state_layout", "slots", "commit_policy",
)


def _layout_hint(operator):
    return "strided" if operator.get("layout") in {"contiguous", "strided"} else None


def _input_ports(metadata):
    ports = []
    for record in metadata.get("fused_origin_scopes", ()):
        if record.get("kind") == "input_ports":
            for port in record.get("ports", ()):
                if port not in ports:
                    ports.append(port)
    return ports


def _metadata_boundary(metadata):
    for record in reversed(metadata.get("fused_origin_scopes", ())):
        if record.get("kind") in {"transform_boundary", "native_fusion_boundary",
                                  "activation_materialization"}:
            boundary = record.get("boundary")
            if isinstance(boundary, dict):
                return boundary
    boundary = metadata.get("transform_boundary")
    return boundary if isinstance(boundary, dict) else None


def _typed_record_value(record, default_role, *, name=None, op_id=""):
    raw_layout = record.get("layout") if isinstance(record.get("layout"), dict) else {}
    logical = record.get("logical_shape") or record.get("shape") or raw_layout.get("shape")
    storage = (record.get("storage_shape") or raw_layout.get("storage_shape")
               or logical)
    strides = record.get("strides") or raw_layout.get("strides")
    offset = record.get("offset", raw_layout.get("offset", 0))
    output = record.get("output") or name or default_role
    attrs = {
        "source": record.get("source") or f"{op_id}:{output}",
        "layout_source": "typed_record" if raw_layout else "not_declared",
        "declared_contract": record.get("contract", ""),
    }
    for key in ("consumer", "consumer_field", "value_id", "semantic_op",
                "semantic_producer", "semantic_output", "access_shape",
                "read_elements", "repeats", "read_bytes", "member", "port"):
        if record.get(key) is not None:
            attrs[key] = record[key]
    if record.get("hbm_materialized") is not None:
        attrs["hbm_materialized"] = record["hbm_materialized"]
    layout_kind = record.get("layout_kind")
    if not (isinstance(layout_kind, str)
            and (layout_kind.upper() in _TENSOR_LAYOUTS or layout_kind == "strided")):
        layout_kind = "strided" if raw_layout else None
    return _value(
        name or output, record.get("role") or default_role,
        producer=record.get("producer") or op_id, output=output,
        dtype=record.get("dtype"), shape=logical, storage_shape=storage,
        strides=strides, offset=offset, layout_kind=layout_kind,
        quant=record.get("quant", ""), attrs=attrs,
        materialized=record.get("materialized"),
        contract=record.get("contract", ""), logical_bytes=record.get("bytes"),
        storage_bytes=record.get("storage_bytes"), write_bytes=record.get("write_bytes"),
        declared_bytes=record.get("declared_bytes"), storage=record.get("storage", ""),
        alias_of=record.get("alias_of", ""), state_version=record.get("state_version"),
        binding=record.get("binding"), binding_name=record.get("binding_name"))


def _audit_inputs(operator, boundary, typed):
    if boundary.get("inputs"):
        return list(boundary["inputs"]), []
    if typed.get("operands"):
        return list(typed["operands"]), []
    candidates = list(boundary.get("typed_inputs")
                      or typed.get("inputs") or ())
    deps = set(operator.get("deps") or ())
    actual = [record for record in candidates
              if record.get("producer") in deps
              and record.get("role") in {"activation", "input"}]
    if not actual:
        actual = [record for record in candidates
                  if record.get("producer")
                  and record.get("role") in {"activation", "input"}]
    declarations = [record for record in candidates if record not in actual]
    return actual, declarations


def _audit_projection(operator, boundary):
    op_id = operator.get("op_id", "")
    typed = boundary.get("typed") if isinstance(boundary.get("typed"), dict) else {}
    raw_inputs, declaration_inputs = _audit_inputs(operator, boundary, typed)
    raw_outputs = boundary.get("typed_outputs") or typed.get("external_outputs") or ()
    inputs = [_typed_record_value(record, "input", name=f"input{index}", op_id=op_id)
              for index, record in enumerate(raw_inputs)]
    for value, record in zip(inputs, raw_inputs):
        if record.get("consumer") and record.get("consumer") != op_id:
            value["attrs"]["declared_consumer"] = record["consumer"]
            value["attrs"]["consumer"] = op_id
            value["attrs"]["consumer_basis"] = "producer_dependency"
    outputs = [_typed_record_value(record, "output", name=record.get("output"),
                                   op_id=op_id)
               for record in raw_outputs]
    declared = []
    seen = {(value["producer"], value["output"]) for value in (*inputs, *outputs)}
    for record in declaration_inputs:
        value = _typed_record_value(record, record.get("role") or "parameter",
                                    name=record.get("output") or record.get("field"),
                                    op_id=op_id)
        value["attrs"]["declaration_only"] = True
        if value["output"] in {item["output"] for item in outputs} and (
                value["producer"] != op_id):
            value["attrs"]["conflict_with_output"] = True
        key = (value["producer"], value["output"])
        if key not in seen:
            declared.append(value)
            seen.add(key)
    for record in typed.get("internal", ()):
        key = (record.get("producer"), record.get("output"))
        if key in seen:
            continue
        value = _typed_record_value(record, "internal", name=record.get("output"),
                                    op_id=op_id)
        value["attrs"]["internal"] = True
        declared.append(value)
        seen.add(key)
    for record in (typed.get("inputs") or ()):
        value = _typed_record_value(record, record.get("role") or "parameter",
                                    name=record.get("field"), op_id=op_id)
        value["attrs"]["declaration_only"] = True
        declared.append(value)
    roles = dict(boundary.get("typed_roles") or {})
    roles.update(typed.get("roles") or {})
    for role, records in roles.items():
        if role in {"input", "output"}:
            continue
        for record in records:
            field = record.get("field") or role
            key = (op_id, field)
            if key in seen:
                continue
            declared.append(_value(
                f"{role}:{field}", role, producer=op_id, output=field,
                dtype=record.get("dtype"), shape=record.get("shape"),
                layout_kind=record.get("layout_kind") or "strided",
                attrs={"source": f"{op_id}:typed_roles.{role}.{field}",
                       "declared_contract": "typed_role"},
                contract="typed_role"))
            seen.add(key)
    external_outputs = (boundary.get("external_outputs")
                        or typed.get("external_ports")
                        or [value["output"] for value in outputs])
    return {
        "inputs": inputs, "outputs": outputs, "declared": declared,
        "semantic_attrs": list(boundary.get("semantic_attrs") or ()),
        "external_outputs": list(external_outputs),
        "external_ports": list(typed.get("external_ports") or ()),
        "external_consumers": list(boundary.get("external_consumers") or ()),
        "field_dispositions": list(boundary.get("field_dispositions") or ()),
        "selection": boundary.get("selection"),
        "basis": "typed_records",
    }


def _layout_value(name, role, *, producer, output, dtype, layout, attrs,
                  materialized=True, contract="declared_view"):
    layout = layout if isinstance(layout, dict) else {}
    return _value(
        name, role, producer=producer, output=output, dtype=dtype,
        shape=layout.get("shape"), storage_shape=layout.get("storage_shape"),
        strides=layout.get("strides"), offset=layout.get("offset", 0),
        layout_kind="strided", attrs=attrs, materialized=materialized,
        contract=contract)


def _view_values(operator, context=None):
    meta = operator.get("metadata", {})
    source = meta.get("source_layout")
    target = meta.get("target_layout")
    if not source and not target:
        return None
    op_id = operator.get("op_id", "")
    deps = operator.get("deps", ())
    native = meta.get("native_source") if isinstance(meta.get("native_source"), dict) else {}
    common = {
        "materialization_reason": meta.get("materialization_reason"),
        "source_output": meta.get("source_output"),
        "native_source": {key: native.get(key) for key in
                          ("repository", "revision", "path", "sha256", "symbol",
                           "source_state") if native.get(key)},
    }
    inputs, outputs, gaps = [], [], []
    producer, producer_output, producer_basis = "", meta.get("source_output") or "source", "unresolved"
    if source and context and len(deps) == 1:
        resolved = _producer_inputs(operator, context)
        if len(resolved) == 1 and resolved[0]["producer"]:
            producer = resolved[0]["producer"]
            producer_output = resolved[0]["output"] or producer_output
            producer_basis = "single_declared_dependency"
    if source:
        inputs.append(_layout_value(
            "source_view", "input", producer=producer,
            output=producer_output,
            dtype=meta.get("input_format") or operator.get("dtype"),
            layout=source,
            attrs={**common, "source": "operator.metadata.source_layout",
                   "layout_source": "declared_source_layout",
                   "producer_basis": producer_basis,
                   "producer_unresolved": not bool(producer),
                   "view": "source"}))
        if not producer:
            gaps.append({"code": "source_view_producer_unresolved",
                         "detail": {"op_id": op_id, "deps": list(deps),
                                    "source_output": meta.get("source_output")}})
    else:
        gaps.append({"code": "view_source_layout_missing",
                     "detail": {"op_id": op_id}})
    if target:
        outputs.append(_layout_value(
            "target_view", "output", producer=op_id, output="output",
            dtype=meta.get("output_format") or operator.get("dtype"),
            layout=target,
            attrs={**common, "source": "operator.metadata.target_layout",
                   "layout_source": "declared_target_layout",
                   "view": "target"}))
    else:
        gaps.append({"code": "view_target_layout_missing",
                     "detail": {"op_id": op_id}})
    return {"inputs": inputs, "outputs": outputs, "declared": [],
            "basis": "declared_view", "gaps": gaps}


def _producer_inputs(operator, context):
    context = context or {}
    declared_op = context.get("op")
    outputs_by_id = context.get("outputs_by_id") or {}
    if declared_op is None or not outputs_by_id:
        return []
    from mindie_sim.prediction.implementations import producer_input_values

    try:
        records = producer_input_values(declared_op, outputs_by_id=outputs_by_id)
    except (KeyError, ValueError, TypeError):
        return []
    return [_typed_record_value(record, "input", name=record.get("role") or "input",
                                op_id=operator.get("op_id", ""))
            for record in records]


def _swiglu_values(operator, contract, context):
    meta = operator.get("metadata", {})
    op_id = operator.get("op_id", "")
    shape, dtype = _shape(contract[1]), contract[2]
    output = _value("output", "output", producer=op_id, dtype=dtype, shape=shape,
                    storage_shape=shape, layout_kind=_layout_hint(operator),
                    attrs={"source": "contract[1:3]", "shape_basis": "swiglu_output"},
                    contract="swiglu_semantic_output")
    inputs = _producer_inputs(operator, context)
    gaps = []
    if not inputs:
        rows = shape[0] if shape else None
        width = shape[-1] if shape else None
        read = meta.get("read_bytes_override", meta.get("read_bytes"))
        dtype_bytes = _nbytes(dtype, [1])
        input_width = None
        if rows and read and dtype_bytes:
            unit = read / (rows * dtype_bytes)
            if float(unit).is_integer():
                input_width = int(unit)
        basis = "declared_read_bytes"
        if input_width is None and width and meta.get("activation_kind") == "silu":
            input_width = 2 * width
            basis = "swiglu_gate_up_width_2x"
        if rows and input_width:
            inputs = [_value(
                "input", "input", producer="",
                output="output", dtype=meta.get("input_format") or operator.get("dtype"),
                shape=[rows, input_width], storage_shape=[rows, input_width],
                layout_kind=_layout_hint(operator),
                attrs={"source": "swiglu_declared_geometry", "shape_basis": basis},
                contract="declared_operand_value")]
        else:
            gaps.append({"code": "swiglu_input_unresolved",
                         "detail": {"rows": rows, "output_width": width,
                                    "read_bytes": read}})
    if inputs and not inputs[0].get("producer"):
        gaps.append({"code": "swiglu_input_producer_unresolved",
                     "detail": {"deps": list(operator.get("deps") or ())}})
    return {"inputs": inputs, "outputs": [output], "declared": [],
            "basis": "contract+producer" if inputs else "contract",
            "gaps": gaps}


def _source_layout_producer(operator, layout, context):
    metadata_by_id = (context or {}).get("metadata_by_id") or {}
    target = tuple(layout.get("storage_shape") or ())
    target_elements = math.prod(target) if target else 0
    matches = []
    for dep in operator.get("deps") or ():
        meta = metadata_by_id.get(dep) or {}
        for candidate in meta.get("source_layouts") or ():
            storage = tuple(candidate.get("storage_shape") or ())
            if (target_elements and storage
                    and math.prod(storage) == target_elements):
                if dep not in matches:
                    matches.append(dep)
                break
    return matches[0] if len(matches) == 1 else ""


def _source_view_value(name, role, *, producer, output, dtype, layout,
                      contract_shape, source_index, producer_basis="",
                      output_unresolved=False, root_version=None):
    layout = layout if isinstance(layout, dict) else {}
    shape = _shape(layout.get("shape"))
    storage = _shape(layout.get("storage_shape"))
    attrs = {
        "source": f"operator.metadata.source_layouts[{source_index}]",
        "shape_basis": "declared_source_layout",
        "source_layout_index": source_index,
        "contract_shape": list(contract_shape) if contract_shape else None,
        "shared_storage": True,
        "separate_materialization": False,
        "producer_basis": producer_basis,
        "producer_unresolved": not bool(producer),
        "producer_output_unresolved": output_unresolved,
        "root_storage_shape": list(storage) if storage else None,
        "root_storage_bytes": _nbytes(dtype, storage),
        "root_producer": producer,
        "root_output": output,
        "root_state_version": root_version,
    }
    return _value(
        name, role, producer=producer, output=output, dtype=dtype,
        shape=shape, storage_shape=storage, strides=layout.get("strides"),
        offset=layout.get("offset", 0), layout_kind="strided", attrs=attrs,
        materialized=True, contract="declared_source_view", write_bytes=0.0)


def _recurrent_values(operator, contract, context=None):
    meta = operator.get("metadata", {})
    op_id = operator.get("op_id", "")
    tokens, key_heads, heads, dk, dv = contract[1]
    state_dtype = contract[3]
    dtype = operator.get("dtype") or meta.get("input_format")
    deps = operator.get("deps") or ()
    layouts = list(meta.get("source_layouts") or ())
    q_shape = [tokens, key_heads, dk]
    k_shape = [tokens, key_heads, dk]
    v_shape = [tokens, heads, dv]
    operand_unresolved = []

    def operand(name, role, shape, layout=None, source_index=0):
        if layout is not None:
            producer = _source_layout_producer(operator, layout, context)
            root_version = None
            for root in ((context or {}).get("outputs_by_id", {}).get(producer) or ()):
                if root.get("output") == "output":
                    root_version = root.get("state_version")
                    break
            return _source_view_value(
                name, role, producer=producer, output=("output" if producer else ""),
                dtype=dtype, layout=layout, contract_shape=shape,
                source_index=source_index,
                producer_basis=("shared_source_layout" if producer else
                                "unresolved"),
                output_unresolved=not bool(producer),
                root_version=root_version)
        producer = ""
        if context and (context.get("metadata_by_id") or {}):
            operand_unresolved.append(name)
        return _value(
            name, role, producer=producer, output="output", dtype=dtype,
            shape=shape, layout_kind=_layout_hint(operator),
            attrs={"source": "contract[1]",
                   "shape_basis": "recurrent_declared_geometry",
                   "producer_unresolved": True},
            contract="declared_operand_value")

    q = operand("query", "input", q_shape,
                layouts[0] if len(layouts) >= 1 else None, 0)
    k = operand("key", "input", k_shape,
                layouts[1] if len(layouts) >= 2 else None, 1)
    v = operand("value", "input", v_shape,
                layouts[2] if len(layouts) >= 3 else None, 2)
    state = meta.get("gdn_state_contract") or (
        (meta.get("program_execution") or {}).get("state_contract")
        if isinstance(meta.get("program_execution"), dict) else None) or {}
    layout = state.get("state_layout")
    active_count = len(state.get("active_slots") or ()) or state.get("slots") or 1
    capacity = state.get("state_capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        capacity = None
    active_shape = None
    if layout == "V,K":
        active_shape = [active_count, heads, dv, dk]
    elif layout == "K,V":
        active_shape = [active_count, heads, dk, dv]
    capacity_shape = None
    if active_shape is not None and capacity is not None:
        capacity_shape = [capacity, *active_shape[1:]]
    state_attrs = {
        "source": "operator.metadata.gdn_state_contract",
        "shape_basis": "declared_state_layout",
        "state_layout": layout, "active_slots": state.get("active_slots"),
        "state_capacity": capacity,
        "state_capacity_lower_bound": state.get("state_capacity_lower_bound"),
        "targets": state.get("targets"),
        "slot_indices": state.get("slot_indices"),
        "slot_source": state.get("slot_source"),
        "publish_point": state.get("publish_point"),
        "read_versions": state.get("read_versions"),
        "capacity_basis": ("declared_state_capacity" if capacity is not None
                           else "capacity_not_declared"),
    }
    output = _value("output", "output", producer=op_id, output="output", dtype=dtype,
                    shape=v_shape, layout_kind=_layout_hint(operator),
                    attrs={"source": "contract[1]",
                           "shape_basis": "recurrent_declared_geometry"},
                    contract="recurrent_semantic_output")
    state_value = _value(
        "state", "input", producer="", output="state", dtype=state_dtype,
        shape=active_shape, storage_shape=capacity_shape,
        storage_unknown=capacity_shape is None,
        layout_kind=layout or None, attrs=state_attrs,
        contract="declared_state_value")
    state_out = _value(
        "state_out", "output", producer=op_id, output="state", dtype=state_dtype,
        shape=active_shape, storage_shape=capacity_shape,
        storage_unknown=capacity_shape is None,
        layout_kind=layout or None,
        attrs={**state_attrs, "commit_versions": state.get("commit_versions")},
        contract="declared_state_value")
    gaps = []
    if active_shape is None:
        gaps.append({"code": "state_shape_not_declared",
                     "detail": {"state_layout": layout,
                                "active_slots": state.get("active_slots"),
                                "heads": heads, "dk": dk, "dv": dv}})
    if operand_unresolved:
        gaps.append({"code": "operand_producer_unresolved",
                     "detail": {"operands": operand_unresolved,
                                "effect": "no unique source_layout producer"}})
    if any(not value["producer"] for value in (q, k, v)):
        gaps.append({"code": "source_view_producer_unresolved",
                     "detail": {"operands": [value["name"] for value in (q, k, v)
                                             if not value["producer"]]}})
    if capacity_shape is None:
        gaps.append({"code": "state_capacity_not_declared",
                     "detail": {"state_layout": layout,
                                "active_shape": active_shape,
                                "active_slots": state.get("active_slots"),
                                "state_capacity": state.get("state_capacity"),
                                "effect": "storage_shape/storage_bytes unknown; active bytes retained"}})
    return {"inputs": [q, k, v, state_value], "outputs": [output, state_out],
            "declared": [], "basis": "contract+state", "gaps": gaps}


def _scale_details(meta):
    details = {}
    if meta.get("activation_scale_format"):
        details["activation"] = {
            "format": meta.get("activation_scale_format"),
            "shape": list(meta.get("activation_scale_shape") or ()) or None,
            "group_size": meta.get("activation_group_size"),
            "bytes": meta.get("activation_scale_bytes"),
            "in_mte2": meta.get("activation_scale_in_mte2"),
            "producer": meta.get("activation_producer"),
            "source": "operator.metadata.activation_scale_format",
        }
    if meta.get("weight_scale_format"):
        details["weight"] = {
            "format": meta.get("weight_scale_format"),
            "storage_format": meta.get("weight_scale_storage_format"),
            "bytes": meta.get("weight_scale_bytes"),
            "storage_bytes": meta.get("weight_scale_storage_bytes"),
            "group_size": meta.get("weight_scale_group_size"),
            "shape": list(meta.get("weight_scale_shape") or ()) or None,
            "source": "operator.metadata.weight_scale_format",
        }
    if meta.get("output_scale_format"):
        details["output"] = {
            "format": meta.get("output_scale_format"),
            "shape": list(meta.get("output_scale_shape") or ()) or None,
            "bytes": meta.get("output_scale_bytes"),
            "source": "operator.metadata.output_scale_format",
        }
    return details


def _scale_values(operator, values, quantization):
    meta = operator.get("metadata", {})
    details = _scale_details(meta)
    if not details:
        return
    op_id = operator.get("op_id", "")
    returns = set(operator.get("returns") or ())
    output_names = {value["output"] for value in values["outputs"]}
    scale_output = bool({"activation_scales", "activation_scale"} &
                        (returns | output_names))
    inputs_by_name = {value["name"]: value for value in values["inputs"]}
    inputs_by_output = {value["output"]: value for value in values["inputs"]}
    outputs_by_output = {value["output"]: value for value in values["outputs"]}
    for role, detail in details.items():
        name = f"{role}_scale"
        attrs = {
            "source": detail["source"], "association": role,
            "format": detail.get("format"),
            "storage_format": detail.get("storage_format") or detail.get("format"),
            "group_size": detail.get("group_size"),
            "bytes": detail.get("bytes"),
            "storage_bytes": detail.get("storage_bytes"),
            "shape_basis": ("declared_scale_shape" if detail.get("shape")
                            else "not_declared"),
        }
        if role == "activation":
            attrs.update(in_mte2=detail.get("in_mte2"),
                         producer=detail.get("producer"))
        if role == "activation" and scale_output:
            value = (next((item for item in values["outputs"]
                           if item["output"] in {"activation_scales", "activation_scale"}
                           and item["producer"] == op_id), None)
                     or outputs_by_output.get("activation_scales")
                     or outputs_by_output.get("activation_scale"))
            if value is None:
                value = _value("activation_scales", "scale", producer=op_id,
                               output="activation_scales",
                               dtype=detail.get("format"), shape=detail.get("shape"),
                               storage_shape=detail.get("shape"),
                               layout_kind=_layout_hint(operator), attrs=attrs,
                               contract="declared_scale_output")
                values["outputs"].append(value)
                outputs_by_output[value["output"]] = value
            value["dtype"] = detail.get("format") or value["dtype"]
            value["shape"] = value["shape"] or detail.get("shape")
            value["logical_shape"] = value["shape"]
            value["storage_shape"] = value["storage_shape"] or value["shape"]
            attrs.update(payload="payload", payload_producer=op_id)
            value["attrs"].update(attrs)
        else:
            value = inputs_by_name.get(name) or inputs_by_output.get(
                "activation_scales" if role == "activation" else name)
            if role == "activation":
                payload = (inputs_by_name.get("activation")
                           or inputs_by_output.get("payload"))
                if payload is not None:
                    attrs.update(payload_producer=payload.get("producer"),
                                 payload_output=payload.get("output"))
            if value is None:
                value = _value(name, "scale" if role != "output" else "auxiliary",
                               producer=op_id, output=f"{role}_scales",
                               dtype=detail.get("format"), shape=detail.get("shape"),
                               storage_shape=detail.get("shape"),
                               layout_kind=_layout_hint(operator),
                               attrs=attrs,
                               contract="declared_scale_association")
                target = values["inputs"] if role == "activation" else values["declared"]
                target.append(value)
            else:
                value["dtype"] = detail.get("format") or value["dtype"]
                value["shape"] = value["shape"] or detail.get("shape")
                value["logical_shape"] = value["shape"]
                value["storage_shape"] = value["storage_shape"] or value["shape"]
                value["attrs"].update(attrs)
            inputs_by_name[value["name"]] = value
            inputs_by_output[value["output"]] = value
    quantization["scales"] = {
        role: {key: detail.get(key) for key in
               ("format", "storage_format", "shape", "group_size", "bytes",
                "storage_bytes", "in_mte2", "producer") if detail.get(key) is not None}
        for role, detail in details.items()}


def _semantic_attrs(operator, boundary_attrs):
    meta = operator.get("metadata", {})
    attrs = list(boundary_attrs or ())
    seen = {(item.get("name"), item.get("value")) for item in attrs
            if isinstance(item, dict)}
    for key in _SEMANTIC_ATTR_KEYS:
        if key not in meta:
            continue
        item = {"name": key, "value": meta[key], "unit": "",
                "source": f"{operator.get('op_id', '')}.metadata"}
        if (key, meta[key]) not in seen:
            attrs.append(item)
            seen.add((key, meta[key]))
    return attrs


def _state_facts(operator):
    meta = operator.get("metadata", {})
    execution = meta.get("program_execution")
    execution = execution if isinstance(execution, dict) else {}
    contract = meta.get("gdn_state_contract") or execution.get("state_contract")
    effects = meta.get("gdn_state_effects") or execution.get("state_effects")
    return contract, effects


def _postprocess_compact_layout(values, operator, contract):
    if contract is None or contract[0] in _EXPLICIT_LAYOUT_KINDS:
        return
    hint = _layout_hint(operator)
    for value in (*values["inputs"], *values["outputs"], *values["declared"]):
        if value["layout_kind"] == "ND" and str(value["attrs"].get("source", "")).startswith("contract"):
            value["layout_kind"] = hint
            value["layout"] = hint
            value["attrs"]["layout_basis"] = "compact_contract_does_not_declare_layout"


def _merge_values(values, extra):
    if extra is None:
        return
    for key in ("inputs", "outputs", "declared"):
        existing = {(value["name"], value["producer"], value["output"]): value
                    for value in values[key]}
        for value in extra.get(key, ()):
            current = existing.get((value["name"], value["producer"], value["output"]))
            if current is None:
                values[key].append(value)
                existing[(value["name"], value["producer"], value["output"])] = value
            else:
                for field, item in value.items():
                    if current.get(field) in (None, "", []) and item not in (None, "", []):
                        current[field] = item
                current["attrs"].update({name: item for name, item in value["attrs"].items()
                                         if item is not None})


def _contract_gap(operator, contract, declared):
    meta = operator.get("metadata", {})
    family, recipe = operator.get("family"), operator.get("recipe")
    if contract is None:
        if operator.get("source_boundary") == "call":
            return {"code": "source_call_boundary",
                    "detail": {"recipe": recipe,
                               "effect": "native tensors belong to expanded children"}}
        if meta.get("program_kind") == "matmul":
            return {"code": "matrix_tensor_abi_unavailable",
                    "detail": {"recipe": recipe,
                               "matrix_groups": meta.get("matrix_groups"),
                               "activation_format": meta.get("activation_format"),
                               "weight_format": meta.get("weight_format"),
                               "output_format": meta.get("output_format"),
                               "bias": bool(meta.get("bias")),
                               "activation_scale_shape": meta.get("activation_scale_shape")}}
        if family == "attention":
            return {"code": "attention_tensor_abi_unavailable",
                    "detail": {"recipe": recipe,
                               "attention_tensor_view": meta.get("attention_tensor_view")}}
        return {"code": "native_tensor_abi_unavailable",
                "detail": {"family": family, "recipe": recipe}}
    kind = contract[0]
    if not declared:
        return {"code": "contract_value_layout_unmodeled",
                "detail": {"contract_kind": kind,
                           "effect": "contract tuple retained; no per-value projection"}}
    return None


def _tensor_copy_values(operator, contract, context=None):
    """Public values for a copy/concatenate without a target view.

    A concatenate's source_layouts describe slices of one root allocation.  The
    public output is that root (shape from its declared semantic size when it
    agrees with the root storage); consumers keep their own view shapes.
    """
    meta = operator.get("metadata", {})
    op_id = operator.get("op_id", "")
    layouts = [item for item in (meta.get("source_layouts") or ())
               if isinstance(item, dict)]
    if not layouts:
        return None
    dtype = meta.get("output_format") or operator.get("dtype")
    storage = _shape(layouts[0].get("storage_shape"))
    storage_elements = math.prod(storage) if storage else 0
    semantic = _shape(operator.get("semantic_shape") or operator.get("local_shape"))
    shape = semantic if semantic and (not storage_elements
                                      or math.prod(semantic) == storage_elements) else storage
    if shape is None:
        return None
    output = _value(
        "output", "output", producer=op_id, output="output", dtype=dtype,
        shape=shape, storage_shape=storage, layout_kind="strided",
        attrs={"source": "operator.metadata.source_layouts",
               "shape_basis": "declared_semantic_shape" if semantic and
               math.prod(shape) == math.prod(semantic) else "root_storage",
               "materialization_reason": meta.get("materialization_reason"),
               "root_storage_shape": list(storage) if storage else None,
               "root_elements": storage_elements,
               "source_views": [{"shape": list(_shape(item.get("shape")) or ()),
                                 "offset": item.get("offset", 0),
                                 "strides": list(item.get("strides") or ())}
                                for item in layouts]},
        contract="declared_copy_output")
    gaps = []
    if len(layouts) > 1:
        gaps.append({"code": "copy_source_views_preserved",
                     "detail": {"op_id": op_id, "views": len(layouts),
                                "root_storage_shape": list(storage) if storage else None}})
    return {"inputs": [], "outputs": [output], "declared": [], "gaps": gaps,
            "basis": "source_layouts"}


def _attention_values(operator):
    meta = operator.get("metadata", {})
    gqa = meta.get("gqa_decode") if isinstance(meta.get("gqa_decode"), dict) else {}
    execution = meta.get("program_execution")
    kv = (execution or {}).get("kv_cache_access") if isinstance(execution, dict) else None
    op_id = operator.get("op_id", "")
    deps = operator.get("deps") or ()
    query_dtype = gqa.get("query_dtype") or operator.get("dtype")
    kv_dtype = gqa.get("kv_dtype") or query_dtype
    query_attrs = {
        "source": "operator.metadata.gqa_decode",
        "shape_basis": "semantic_geometry_only",
        "query_tokens": gqa.get("uniform_query_tokens"),
        "query_heads": gqa.get("query_heads"),
        "kv_heads": gqa.get("kv_heads"),
        "head_dim": gqa.get("head_dim"),
        "causal": gqa.get("causal"),
        "window": gqa.get("window"),
        "cp": gqa.get("cp"),
    }
    cache_attrs = {
        "source": "operator.metadata.program_execution.kv_cache_access",
        "shape_basis": "native_cache_storage_not_declared",
        "kv_heads": gqa.get("kv_heads"),
        "head_dim": gqa.get("head_dim"),
        "context_tokens": gqa.get("uniform_context_tokens"),
        "storage": (kv or {}).get("storage"),
        "page_size": (kv or {}).get("page_size"),
        "key_format": (kv or {}).get("key_format"),
        "value_format": (kv or {}).get("value_format"),
        "key_payload_row_bytes": (kv or {}).get("key_payload_row_bytes"),
        "value_payload_row_bytes": (kv or {}).get("value_payload_row_bytes"),
        "payload_scope": (kv or {}).get("payload_scope"),
    }
    inputs = [
        _value("query", "input", output="query", dtype=query_dtype, shape=None,
               attrs={**query_attrs, "producer_unresolved": True},
               contract="declared_semantic_geometry"),
        _value("key_cache", "input", output="key_cache", dtype=kv_dtype,
               shape=None, attrs={**cache_attrs, "external": True},
               contract="declared_cache_access"),
        _value("value_cache", "input", output="value_cache", dtype=kv_dtype,
               shape=None, attrs={**cache_attrs, "external": True},
               contract="declared_cache_access"),
    ]
    output_shape = None
    if (gqa.get("uniform_query_tokens") is not None
            and gqa.get("query_heads") is not None
            and gqa.get("head_dim") is not None):
        output_shape = [gqa["uniform_query_tokens"], gqa["query_heads"],
                        gqa["head_dim"]]
    outputs = [_value("output", "output", producer=op_id, output="output",
                      dtype=query_dtype, shape=output_shape,
                      attrs={"source": "operator.metadata.gqa_decode",
                             "shape_basis": "semantic_geometry_only",
                             "query_tokens": gqa.get("uniform_query_tokens"),
                             "query_heads": gqa.get("query_heads"),
                             "head_dim": gqa.get("head_dim")},
                      contract="declared_semantic_geometry")]
    attrs = [
        {"name": "query_tokens", "value": gqa.get("uniform_query_tokens"),
         "unit": "tokens", "source": "operator.metadata.gqa_decode"},
        {"name": "context_tokens", "value": gqa.get("uniform_context_tokens"),
         "unit": "tokens", "source": "operator.metadata.gqa_decode"},
        {"name": "query_heads", "value": gqa.get("query_heads"),
         "unit": "heads", "source": "operator.metadata.gqa_decode"},
        {"name": "kv_heads", "value": gqa.get("kv_heads"),
         "unit": "heads", "source": "operator.metadata.gqa_decode"},
        {"name": "head_dim", "value": gqa.get("head_dim"),
         "unit": "elements", "source": "operator.metadata.gqa_decode"},
        {"name": "query_dtype", "value": query_dtype, "unit": "",
         "source": "operator.metadata.gqa_decode"},
        {"name": "kv_dtype", "value": kv_dtype, "unit": "",
         "source": "operator.metadata.gqa_decode"},
        {"name": "causal", "value": gqa.get("causal"), "unit": "",
         "source": "operator.metadata.gqa_decode"},
        {"name": "window", "value": gqa.get("window"), "unit": "tokens",
         "source": "operator.metadata.gqa_decode"},
        {"name": "cp", "value": gqa.get("cp"), "unit": "",
         "source": "operator.metadata.gqa_decode"},
    ]
    gaps = [{"code": "attention_native_slots_unresolved",
             "detail": {"recipe": operator.get("recipe"),
                        "missing": ["query_layout", "key_cache_layout",
                                    "value_cache_layout", "mask", "scale",
                                    "block_table", "sequence_lengths",
                                    "output_layout", "cache_capacity"],
                        "known": ["query_tokens", "context_tokens", "query_heads",
                                  "kv_heads", "head_dim", "query_dtype", "kv_dtype",
                                  "causal", "window", "kv_cache_access"]}}]
    return {"inputs": inputs, "outputs": outputs, "declared": [],
            "basis": "semantic_geometry", "gaps": gaps,
            "semantic_attrs": attrs, "kv_cache_access": kv}


def _apply_input_ports(operator, values, context, gaps):
    ports = _input_ports(operator.get("metadata", {}))
    if not ports:
        return
    outputs_by_id = (context or {}).get("outputs_by_id") or {}
    op_id = operator.get("op_id", "")
    used = set()
    for port in ports:
        producer = port.get("op")
        name = port.get("output")
        if not producer or not name:
            continue
        candidates = [value for value in values["inputs"]
                      if id(value) not in used]
        target = None
        if name in {"payload", "activation", "output"}:
            target = next((value for value in candidates
                           if value["name"] in {"activation", "input", "source_view"}),
                          None)
        elif name in {"activation_scales", "activation_scale"}:
            target = next((value for value in candidates
                           if value["name"] in {"activation_scale", "activation_scales"}
                           or value["role"] == "scale"), None)
        if target is None:
            target = next((value for value in candidates
                           if value["name"] == name or value["output"] == name), None)
        if target is None:
            continue
        used.add(id(target))
        produced = list(outputs_by_id.get(producer) or ())
        source = next((value for value in produced
                       if value.get("output") == name), None)
        target["producer"] = producer
        target["output"] = name
        target["attrs"]["input_port"] = dict(port)
        target["attrs"]["input_port_source"] = "operator.metadata.input_ports"
        if source is None:
            target["attrs"]["producer_output_unresolved"] = True
            gaps.append({"code": "input_port_producer_unresolved",
                         "detail": {"op_id": op_id, "producer": producer,
                                    "output": name}})
            continue
        for key in ("dtype", "shape", "logical_shape", "storage_shape", "strides",
                    "offset", "layout", "layout_kind", "quant", "bytes",
                    "storage_bytes", "write_bytes", "declared_bytes", "storage",
                    "alias_of", "state_version", "binding", "binding_name"):
            if key in source and source[key] is not None:
                target[key] = source[key]
        target["attrs"]["producer_output"] = name
        target["attrs"]["producer_contract"] = source.get("contract")


def _fill_from_produced(existing, produced, *, replace=False, contract=None):
    keep_attrs = dict(existing.get("attrs") or {})
    ordinal = keep_attrs.get("abi_ordinal")
    matrix = contract is not None and str(contract[0]).startswith(
        ("matrix", "grouped_matrix"))
    for key in ("dtype", "shape", "logical_shape", "storage_shape", "strides",
                "offset", "layout", "layout_kind", "quant", "bytes",
                "storage_bytes", "write_bytes", "declared_bytes", "storage",
                "alias_of", "state_version", "binding", "binding_name",
                "materialized", "producer", "output"):
        if produced.get(key) is None:
            continue
        if matrix and key in {"layout", "layout_kind"} and not replace:
            continue
        if replace or existing.get(key) in (None, "", []):
            existing[key] = produced[key]
    if not replace and (existing.get("shape") != produced.get("shape")):
        keep_attrs["produced_shape_conflict"] = {
            "existing": existing.get("shape"), "produced": produced.get("shape")}
    if replace:
        existing["name"] = produced.get("output") or produced.get("name")
    keep_attrs["identity_source"] = "produced_values"
    if ordinal is not None:
        keep_attrs["abi_ordinal"] = ordinal
    existing["attrs"] = {**keep_attrs, **{
        key: value for key, value in produced.get("attrs", {}).items()
        if value is not None}}


def _merge_produced_outputs(operator, contract, values, produced):
    if not produced:
        return
    outputs = values["outputs"]
    op_id = operator.get("op_id", "")
    for index, record in enumerate(produced):
        actual = _typed_record_value(record, "output",
                                     name=record.get("output") or "output",
                                     op_id=op_id)
        match = next((value for value in outputs
                      if value.get("output") == actual["output"]
                      or value.get("name") == actual["name"]), None)
        if match is not None:
            _fill_from_produced(match, actual, contract=contract)
            continue
        generic = next((value for value in outputs
                        if re.fullmatch(r"output\d+", str(value.get("name")))),
                       None)
        if generic is not None:
            generic["attrs"]["abi_ordinal"] = outputs.index(generic)
            _fill_from_produced(generic, actual, replace=True, contract=contract)
            continue
        outputs.append(actual)
    # A native norm's auxiliary value is independent even when the producer
    # helper exposes only the primary output.  Keep the ABI ordinal mapping.
    if contract is not None and contract[0] == "native_tensor_abi":
        native = str(contract[1])
        if "RmsNorm" in native or "LayerNorm" in native:
            for ordinal, value in enumerate(outputs):
                if value.get("name") in {"output0", "output1"}:
                    value["attrs"]["abi_ordinal"] = ordinal
                    if ordinal == 0 and value["dtype"] not in {None, "fp32"}:
                        value["name"] = value["output"] = "norm"
                    elif ordinal == 1 and value["dtype"] == "fp32":
                        value["name"] = value["output"] = "rstd"


def typed_values(operator, contract, *, context=None):
    """Project one operator's declared values and representation facts.

    Existing typed records (transform/native-fusion audits), declared source
    views and producer outputs take precedence over the compact comparison
    tuple. The tuple fills only fields it actually declares. Recognition is
    metadata-only: no measurement table, target duration or hardware rate is
    consulted, and an unmeasured shape is accepted.
    """
    meta = operator.get("metadata", {})
    context = context or {}
    op_id = operator.get("op_id", "")
    gaps = []
    quantization = _quantization(operator)
    state_contract, state_effects = _state_facts(operator)
    semantic_attrs, external_outputs, external_consumers = [], [], []
    external_ports = []
    field_dispositions, selection = [], None
    attention_data = None
    boundary = _metadata_boundary(meta)

    if boundary is not None:
        audit = _audit_projection(operator, boundary)
        values = {
            "inputs": audit["inputs"], "outputs": audit["outputs"],
            "declared": audit["declared"], "basis": "typed_records",
        }
        semantic_attrs = audit["semantic_attrs"]
        external_outputs = audit["external_outputs"]
        external_ports = audit["external_ports"]
        external_consumers = audit["external_consumers"]
        field_dispositions = audit["field_dispositions"]
        selection = audit["selection"]
        if contract is not None and (not values["inputs"] or not values["outputs"]):
            projected = _matrix_values(contract) or _common_values(contract)
            if projected is not None:
                extra = {"inputs": projected[0], "outputs": projected[1],
                         "declared": projected[2]}
                _merge_values(values, extra)
                if projected[0] or projected[1]:
                    values["basis"] = "typed_records+contract"
    else:
        attention = (_attention_values(operator)
                     if operator.get("family") == "attention"
                     and meta.get("gqa_decode") and contract is None else None)
        view = _view_values(operator, context) if meta.get("source_layout") or meta.get(
            "target_layout") else None
        if attention is not None:
            attention_data = attention
            values = attention
            gaps.extend(attention["gaps"])
            semantic_attrs = attention["semantic_attrs"]
            external_outputs = ["output"]
        elif view is not None:
            values = view
            gaps.extend(view["gaps"])
        elif operator.get("family") == "swiglu" and contract is not None:
            values = _swiglu_values(operator, contract, context)
            gaps.extend(values["gaps"])
        elif operator.get("family") == "recurrent_vector" and contract is not None:
            values = _recurrent_values(operator, contract, context)
            gaps.extend(values["gaps"])
        elif operator.get("family") == "tensor_copy":
            copied = _tensor_copy_values(operator, contract, context)
            if copied is not None:
                values = copied
                gaps.extend(copied["gaps"])
            else:
                inputs, outputs = _declared_metadata_values(operator)
                values = {"inputs": inputs, "outputs": outputs, "declared": [],
                          "basis": "operator_metadata" if inputs or outputs
                                   else "unavailable"}
        else:
            projected = (_matrix_values(contract) or _common_values(contract)
                         if contract is not None else None)
            if projected is not None:
                inputs, outputs, declared, _extra = projected
                values = {"inputs": inputs, "outputs": outputs,
                          "declared": declared, "basis": "contract"}
            elif contract is not None:
                scanned = _scan_tensors(contract, "contract")
                declared = []
                for index, (path, spec) in enumerate(scanned):
                    declared.append(_value(
                        f"declared{index}", "declared", dtype=spec[1],
                        shape=spec[0], storage_shape=spec[0],
                        layout_kind=spec[2], attrs={"source": path},
                        contract="native_tensor_contract"))
                values = {"inputs": [], "outputs": [], "declared": declared,
                          "basis": "contract_scan"}
                if declared:
                    gaps.append({"code": "tensor_roles_not_distinguished",
                                 "detail": {"contract_kind": contract[0],
                                            "value_count": len(declared)}})
            else:
                inputs, outputs = _declared_metadata_values(operator)
                values = {"inputs": inputs, "outputs": outputs, "declared": [],
                          "basis": "operator_metadata" if inputs or outputs
                                   else "unavailable"}

    if contract is not None:
        _postprocess_compact_layout(values, operator, contract)

    produced = (context or {}).get("produced") or ()
    if produced and operator.get("family") not in {
            "recurrent_vector", "tensor_copy"}:
        _merge_produced_outputs(operator, contract, values, produced)

    deps = operator.get("deps") or ()
    for value in values["outputs"]:
        if not value["producer"]:
            value["producer"] = op_id
        if value["output"] is None:
            value["output"] = value["name"]

    _apply_input_ports(operator, values, context, gaps)
    native_norm = (contract is not None and contract[0] == "native_tensor_abi"
                   and ("RmsNorm" in str(contract[1])
                        or "LayerNorm" in str(contract[1])))
    for index, value in enumerate(values["inputs"]):
        if (value["producer"] or value["role"] not in {"input", "activation"}
                or value["name"] in {"weight", "bias", "state", "key_cache",
                                     "value_cache"}
                or (native_norm and index > 0)):
            if native_norm and index > 0:
                value["attrs"]["parameter"] = True
            continue
        value["attrs"]["producer_unresolved"] = True
        gaps.append({"code": "operand_producer_unresolved",
                     "detail": {"value": value["name"],
                                "deps": list(operator.get("deps") or ()),
                                "effect": "no actual input port; not inferred by position"}})
    if contract is not None and str(contract[0]).startswith(
            ("matrix", "grouped_matrix")):
        for value in values["inputs"]:
            if value["name"] == "weight" and not value["producer"]:
                value["attrs"]["weight_source_unresolved"] = True
                gaps.append({"code": "weight_source_unresolved",
                             "detail": {"matrix_kind": contract[0],
                                        "weight_shape": value.get("shape"),
                                        "effect": "declared weight retained; producer not named"}})
    _scale_values(operator, values, quantization)
    semantic_attrs = _semantic_attrs(operator, semantic_attrs)
    if not external_outputs:
        external_outputs = [
            value["output"] for value in values["outputs"]
            if value["materialized"] is not False
            and value["role"] in {"output", "activation", "residual", "auxiliary"}]

    gap = _contract_gap(operator, contract,
                        values["inputs"] or values["outputs"] or values["declared"])
    if gap is not None:
        gaps.append(gap)
    for value in (*values["inputs"], *values["outputs"], *values["declared"]):
        missing = [key for key in ("dtype", "layout") if value.get(key) is None]
        if missing:
            gaps.append({"code": "value_representation_partial",
                         "detail": {"value": value["name"], "missing": missing}})
        if value.get("shape") is None:
            gaps.append({"code": "value_shape_not_declared",
                         "detail": {"value": value["name"],
                                    "shape_basis": value["attrs"].get("shape_basis")}})
    unique = []
    for gap in gaps:
        if gap not in unique:
            unique.append(gap)
    return {
        "basis": values["basis"],
        "contract_kind": contract[0] if contract is not None else None,
        "kv_cache_access": (attention_data or {}).get("kv_cache_access"),
        "inputs": values["inputs"],
        "outputs": values["outputs"],
        "declared": values["declared"],
        "quantization": quantization,
        "semantic_attrs": semantic_attrs,
        "state_contract": state_contract,
        "state_effects": state_effects,
        "external_outputs": external_outputs,
        "external_ports": external_ports,
        "external_consumers": external_consumers,
        "field_dispositions": field_dispositions,
        "input_ports": _input_ports(meta),
        "selection": selection,
        "gaps": unique,
    }
