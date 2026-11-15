"""MindIE-SD (mindiesd) diffusion/DiT custom-operator lowering.

[Role] Model the explicit MindIE-SD custom kernels (``rainfusion``,
  ``ada_bsa``, ``eagle_bsa``, ``eagle_qbsa``, ``adalayernorm``,
  ``norm_rope_concat``, ``laurel``, ``quant_four_over_six_a5``, ``eagle_ffn``)
  and the effective denoising step count of DiTCache/TeaCache/TaylorSeer.
[Invariants] Only an exact vendor operator or an explicitly named
  implementation is lowered; a shared family (``rmsnorm``/``gemm``/
  ``elementwise``) is never silently mapped.  Sparse attention requires an
  explicit sparsity ratio, block size and logical FLOPs.  ``eagle_qbsa`` reads
  a declared cache dtype; its bandwidth factor is derived, not a 0.5 constant.
  Quantized output uses the declared input dtype and MX32 padding.  The
  analytical summary does not invent hardware rates: without declared rates it
  returns work and a ``requires_declared_hardware_rates`` basis rather than a
  private A2 cost model.
[Timeouts & Locks] Bounded local arithmetic; no subprocesses, sleeps or locks.
[Side-Effect Policy] Returns immutable ``OpSpec`` and summary dicts; no I/O or
  device access.
"""
from __future__ import annotations

from dataclasses import replace
import math
from math import prod
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage

MINDIE_SD_VENDOR = "mindie_sd"

_ALL_SOCS = ("Ascend910B", "Ascend910_93", "Ascend950")
_ATTENTION_DTYPES = ("bf16", "fp16")
_ATTN_FAMILIES = ("rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa", "indexed_attention", "gqa_attention")

MINDIE_SD_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "rainfusion": {
        "native_kind": "RainFusionBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_rainfusion"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "ada_bsa": {
        "native_kind": "AdaBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_ada_bsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "eagle_bsa": {
        "native_kind": "EagleBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_bsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "eagle_qbsa": {
        "native_kind": "EagleQuantBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_qbsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size", "kv_dtype"),
        "sparse": True, "quantized_cache": True,
    },
    "adalayernorm": {
        "native_kind": "AdaLayerNormFused", "bound": "memory",
        "supported_socs": _ALL_SOCS,
        "families": ("adalayernorm", "mindie_sd_adalayernorm", "rmsnorm", "add_rmsnorm"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("gamma_beta_count",),
    },
    "norm_rope_concat": {
        "native_kind": "RmsNormRopeConcatFused", "bound": "memory",
        "supported_socs": _ALL_SOCS,
        "families": ("norm_rope_concat", "mindie_sd_norm_rope_concat", "rmsnorm", "elementwise"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
    },
    "laurel": {
        "native_kind": "LaurelLowRankResidual", "bound": "cube",
        "supported_socs": _ALL_SOCS,
        "families": ("laurel", "mindie_sd_laurel", "gemm", "elementwise"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("low_rank",),
    },
    "quant_four_over_six_a5": {
        "native_kind": "QuantFourOverSixA5", "bound": "vector",
        "supported_socs": ("Ascend950",),
        "families": ("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5",
                     "quantize", "elementwise"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
        "a5_only": True,
    },
    "eagle_ffn": {
        "native_kind": "CatlassMatmulAddFfn", "bound": "cube",
        "supported_socs": _ALL_SOCS,
        "families": ("eagle_ffn", "mindie_sd_eagle_ffn", "gemm", "ffn"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
    },
}

CAPABILITIES = frozenset({
    "explicit_caller_implementation",
    "mindie_sd",
    "A2",
    "A3",
    "A5",
    "block_sparse_attention",
    "declared_sparsity",
    "declared_work_preserved",
    "ordinary_resource_envelope",
    "a5_mxfp4_quantization",
})

# Exact vendor semantic names.  Shared families remain explicit-only.
_EXACT_SEMANTIC_ALIASES = {
    "rainfusion": "rainfusion",
    "ada_bsa": "ada_bsa",
    "eagle_bsa": "eagle_bsa",
    "eagle_qbsa": "eagle_qbsa",
    "adalayernorm": "adalayernorm",
    "norm_rope_concat": "norm_rope_concat",
    "laurel": "laurel",
    "quant_four_over_six_a5": "quant_four_over_six_a5",
    "eagle_ffn": "eagle_ffn",
}

def capabilities() -> dict[str, Any]:
    """Machine-readable registry of the declarations behind this wrapper."""
    from .implementations import IMPLEMENTATIONS

    implementations = tuple(
        name for name, item in IMPLEMENTATIONS.items() if item.module == __name__)
    selector_operators = {
        operator for name, item in IMPLEMENTATIONS.items()
        if item.module == __name__ for operator in item.operators}
    return {
        "vendor": MINDIE_SD_VENDOR,
        "capabilities": tuple(sorted(CAPABILITIES)),
        "implementations": implementations,
        "semantic_aliases": dict(_EXACT_SEMANTIC_ALIASES),
        "operators": {
            name: {
                "native_kind": spec["native_kind"],
                "bound": spec["bound"],
                "families": tuple(spec["families"]),
                "selector_families": tuple(
                    family for family in spec["families"]
                    if family in selector_operators),
                "direct_only_families": tuple(
                    family for family in spec["families"]
                    if family not in selector_operators),
                "supported_socs": tuple(spec["supported_socs"]),
                "input_dtypes": tuple(spec["input_dtypes"]),
                "attributes": tuple(spec["attributes"]),
                "sparse": bool(spec.get("sparse")),
                "quantized_cache": bool(spec.get("quantized_cache")),
            }
            for name, spec in MINDIE_SD_CUSTOM_OPS.items()
        },
    }

def recipes() -> dict[str, Any]:
    """Register one typed plan per composite MindIE-SD occurrence."""
    return {f"mindie_sd_{name}": PlannedRecipe(_build_contract_plan)
            for name in MINDIE_SD_CUSTOM_OPS}

def _normalize_op_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    clean = str(raw).strip().lower().removeprefix("mindie_sd_")
    return clean if clean in MINDIE_SD_CUSTOM_OPS else str(raw).strip()

def _resolve_op_name(op: OpSpec, explicit_op_name: str | None) -> str:
    norm = _normalize_op_name(explicit_op_name)
    if norm is not None:
        if norm not in MINDIE_SD_CUSTOM_OPS:
            raise ModelingError(f"mindie_sd_custom_op_unknown:{explicit_op_name}")
        return norm
    for candidate in (op.recipe, op.family):
        if candidate in _EXACT_SEMANTIC_ALIASES:
            return _EXACT_SEMANTIC_ALIASES[candidate]
        normalized = _normalize_op_name(candidate)
        if normalized in MINDIE_SD_CUSTOM_OPS:
            return normalized
    raise ModelingError(
        f"mindie_sd_custom_op_requires_explicit_name:{op.family}:{op.recipe}")

def _shape(op: OpSpec) -> tuple[int, ...]:
    shape = tuple(int(value) for value in op.local_shape)
    if not shape or any(value < 0 for value in shape):
        raise ModelingError("mindie_sd_custom_op_requires_nonnegative_local_shape")
    return shape

def _tokens_width(op: OpSpec) -> tuple[int, int]:
    shape = _shape(op)
    return shape[0], prod(shape[1:]) if len(shape) > 1 else 1

def _input_dtype(op: OpSpec, meta: dict[str, Any]) -> str:
    dtype = op.dtype or meta.get("input_format") or meta.get("compute_format")
    if not dtype:
        raise ModelingError("mindie_sd_custom_op_requires_declared_input_dtype")
    return str(dtype)

def _ratio(meta: dict[str, Any], option, name: str) -> float:
    value = option if option is not None else meta.get(name)
    if value is None:
        raise ModelingError(f"mindie_sd_custom_op_requires_declared_{name}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"mindie_sd_custom_op_invalid_{name}") from exc
    return value

def _positive_int(meta: dict[str, Any], option, name: str) -> int:
    value = option if option is not None else meta.get(name)
    if type(value) is not int or value <= 0:
        raise ModelingError(f"mindie_sd_custom_op_requires_declared_{name}")
    return value

def _as_shape(value, name):
    if (isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list))
            or not value or any(type(item) is not int or item < 0 for item in value)):
        raise ModelingError(f"mindie_sd_{name}_shape_invalid")
    return tuple(value)

def _bytes(shape, dtype):
    return float(prod(shape) * dtype_bytes(dtype))

def _role(name, shape, dtype, *, materialized=True, alias=None, kind="value"):
    return {
        "name": name, "shape": tuple(shape), "dtype": str(dtype),
        "bytes": _bytes(shape, dtype), "materialized": bool(materialized),
        "alias": alias, "kind": kind,
    }

def _contract(reads, writes, compute, *, attributes, basis, limitations=(), values=None):
    mte2 = sum(item["bytes"] for item in reads)
    mte3 = sum(item["bytes"] for item in writes
                if item.get("materialized", True))
    vector = sum(float(item.get("vector_ops", 0)) for item in compute)
    matrix = {}
    for item in compute:
        for dtype, flops in item.get("matrix_flops", ()):
            matrix[dtype] = matrix.get(dtype, 0) + float(flops)
    bound = dict(values or {})
    for item in reads:
        bound[item["name"]] = {
            "role": "read", "shape": item.get("shape"), "dtype": item.get("dtype"),
            "bytes": item.get("bytes"), "kind": item.get("kind"),
            "producer": item.get("producer"),
        }
    for item in writes:
        bound[item["name"]] = {
            "role": "write", "shape": item.get("shape"), "dtype": item.get("dtype"),
            "bytes": item.get("bytes"), "kind": item.get("kind"),
            "source": item.get("source"),
            "materialized": item.get("materialized", True),
            "alias": item.get("alias"),
        }
    for item in compute:
        for output in item.get("outputs", (item["name"],)):
            bound.setdefault(output, {"role": "internal", "producer": item["name"]})
    public_outputs = tuple(item["name"] for item in writes
                           if item.get("materialized", True)
                           and item.get("kind") in {"value", "legacy"})
    return {
        "reads": tuple(reads), "writes": tuple(writes),
        "compute": tuple(compute), "values": bound,
        "public_outputs": public_outputs,
        "attributes": dict(attributes),
        "mte2_bytes": mte2, "mte3_bytes": mte3, "hbm_bytes": mte2 + mte3,
        "vector_ops": vector,
        "matrix_flops": tuple(sorted(matrix.items())),
        "logical_flops": sum(matrix.values()),
        "basis": basis, "limitations": tuple(limitations),
    }


def _read(name, shape, dtype, *, kind="value", producer=None):
    item = _role(name, shape, dtype, kind=kind)
    item["producer"] = producer
    return item


def _contract_work(contract):
    reads = sum(item["bytes"] for item in contract["reads"])
    writes = sum(item["bytes"] for item in contract["writes"]
                 if item.get("materialized", True))
    matrix = {}
    for item in contract["compute"]:
        for dtype, flops in item.get("matrix_flops", ()):
            matrix[dtype] = matrix.get(dtype, 0) + float(flops)
    return Work(matrix, {"hbm": reads + writes, "mte2": reads,
                         "mte3": writes,
                         "vector": float(contract["vector_ops"])})


def _build_contract_plan(op: OpSpec, geometry):
    contract = dict(dict(op.metadata).get("mindie_sd_contract") or {})
    if not contract:
        contract = _contract(
            [{"name": "declared_read", "bytes": float(op.mte2_bytes),
              "shape": tuple(), "dtype": op.dtype, "materialized": True,
              "alias": None, "kind": "declared", "producer": None}],
            [{"name": "declared_write", "bytes": float(op.mte3_bytes),
              "shape": tuple(), "dtype": op.dtype, "materialized": True,
              "alias": None, "kind": "declared", "source": "declared_compute"}],
            [{"name": "declared_compute", "kind": "declared",
              "vector_ops": float(op.vector_ops),
              "matrix_flops": op.matrix_flops or ((op.dtype, op.flops),),
              "inputs": (), "outputs": ("declared_compute",)}],
            attributes={}, basis="declared_op_fields")
    total = _contract_work(contract)
    node_specs = {}
    role_node = {}
    write_names = {item["name"] for item in contract["writes"]
                   if item.get("materialized", True)}
    compute_names = {item["name"] for item in contract["compute"]}
    for item in contract["reads"]:
        name = "read_" + item["name"]
        if name in node_specs:
            raise ModelingError("mindie_sd_contract_duplicate_stage:" + name)
        node_specs[name] = {"kind": "read", "item": item}
        role_node[item["name"]] = name
    for item in contract["compute"]:
        name = "compute_" + item["name"]
        if name in node_specs:
            raise ModelingError("mindie_sd_contract_duplicate_stage:" + name)
        node_specs[name] = {"kind": "compute", "item": item}
        for output in item.get("outputs", (name,)):
            if output in role_node:
                raise ModelingError("mindie_sd_contract_duplicate_value:" + output)
            role_node[output] = name
    for item in contract["writes"]:
        if not item.get("materialized", True):
            continue
        name = "write_" + item["name"]
        if name in node_specs:
            raise ModelingError("mindie_sd_contract_duplicate_stage:" + name)
        node_specs[name] = {"kind": "write", "item": item}

    def resolve(value):
        if value in role_node:
            return role_node[value]
        if value in write_names:
            return "write_" + value
        if value in compute_names:
            return "compute_" + value
        raise ModelingError("mindie_sd_contract_missing_producer:" + value)

    def dependencies(node):
        spec = node_specs[node]
        item = spec["item"]
        if spec["kind"] == "read":
            producer = item.get("producer")
            return () if producer is None else (resolve(producer),)
        if spec["kind"] == "compute":
            return tuple(resolve(value) for value in item.get("inputs", ()))
        source = item.get("source")
        if source is None:
            raise ModelingError(
                "mindie_sd_contract_write_missing_source:" + item["name"])
        return (resolve(source),)

    order, visited, visiting = [], set(), set()

    def visit(node):
        if node in visited:
            return
        if node in visiting:
            raise ModelingError("mindie_sd_contract_cycle:" + node)
        visiting.add(node)
        for dep in dependencies(node):
            visit(dep)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    for node in node_specs:
        visit(node)
    workspace_writes = {
        item["name"]: item for item in contract["writes"]
        if item.get("kind") == "workspace" and item.get("materialized", True)}
    workspace_consumers = {}
    for item in contract["reads"]:
        producer = item.get("producer")
        if producer in workspace_writes:
            workspace_consumers["read_" + item["name"]] = producer
    stages = []
    for node in order:
        spec = node_specs[node]
        item = spec["item"]
        if spec["kind"] == "read":
            work = Work(resources={"hbm": item["bytes"], "mte2": item["bytes"]})
        elif spec["kind"] == "compute":
            work = Work(dict(item.get("matrix_flops", ())),
                        {"vector": float(item.get("vector_ops", 0))})
        else:
            work = Work(resources={"hbm": item["bytes"], "mte3": item["bytes"]})
        acquire_name = None
        free_name = None
        if spec["kind"] == "write" and item["name"] in workspace_writes:
            acquire_name = item["name"]
        if spec["kind"] == "read" and node in workspace_consumers:
            free_name = workspace_consumers[node]
        stages.append((node, work, dependencies(node), acquire_name, free_name))

    workspace_pool_name = str(dict(op.metadata).get("workspace_pool", "gm"))
    workspace_grant = dict(op.metadata).get("workspace_grant")

    def program(context):
        grant = dict(getattr(context.op, "metadata", ()) or {}).get(
            "workspace_grant", workspace_grant)
        allocations = {}
        pool_name = workspace_pool_name
        if pool_name not in context.pool_names:
            pool_name = next((name for name in context.pool_names
                              if "workspace" in name or name in {"gm", "hbm"}),
                             pool_name)
        for write_name, item in workspace_writes.items():
            if grant is not None:
                continue
            if pool_name in context.pool_names:
                allocations[write_name] = (context.allocate(
                    "mindie_sd_workspace:" + write_name, pool_name,
                    item["bytes"], (0,)),)
        return KernelProgram(tuple(
            Stage(name, context.bind(work), after=after,
                  acquire=allocations.get(acquire_name, ()),
                  free=allocations.get(free_name, ()))
            for name, work, after, acquire_name, free_name in stages))

    return OperatorPlan(total, program, metadata={
        "mindie_sd_contract": contract,
        "shared_plan": "per_value_read_compute_write_dag",
        "workspace_values": tuple(workspace_writes),
        "workspace_bytes": int(sum(item["bytes"] for item in workspace_writes.values())),
        "workspace_allocation": ("gm" if workspace_writes else "none"),
        "workspace_allocation_pool": workspace_pool_name,
        "workspace_requires_memory_grant": bool(workspace_writes),
        "workspace_grant_reused": workspace_grant is not None,
    })


def _norm_rope_contract(op, meta, dtype, returns):
    q_shape = _as_shape(meta.get("query_shape", meta.get("q_shape", _shape(op))),
                        "query")
    k_shape = _as_shape(meta.get("key_shape", q_shape), "key")
    v_shape = _as_shape(meta.get("value_shape", q_shape), "value")
    if any(len(shape) != 4 for shape in (q_shape, k_shape, v_shape)):
        raise ModelingError("mindie_sd_norm_rope_requires_BSND_shapes")
    b, sq, h, d = q_shape
    sk, sv = k_shape[1], v_shape[1]
    enc_q = meta.get("encoder_query_shape")
    enc_k = meta.get("encoder_key_shape")
    enc_v = meta.get("encoder_value_shape")
    enc_q = _as_shape(enc_q, "encoder_query") if enc_q is not None else None
    enc_k = _as_shape(enc_k, "encoder_key") if enc_k is not None else None
    enc_v = _as_shape(enc_v, "encoder_value") if enc_v is not None else None
    seq = enc_q[1] if enc_q else 0
    sek = enc_k[1] if enc_k else 0
    sev = enc_v[1] if enc_v else 0
    norm_type = int(meta.get("norm_type", 0))
    norm_added_type = int(meta.get("norm_added_type", 0))
    rope_type = int(meta.get("rope_type", 0))
    concat_order = int(meta.get("concat_order", 0))
    training = bool(meta.get("is_training", False))
    if norm_type not in (0, 1, 2, 3, 4) or norm_added_type not in (0, 1, 2, 3, 4):
        raise ModelingError("mindie_sd_norm_rope_invalid_norm_type")
    if rope_type not in (0, 1, 2) or concat_order not in (0, 1):
        raise ModelingError("mindie_sd_norm_rope_invalid_rope_or_concat")
    materialized = tuple(meta.get("materialized_outputs", ("query", "key", "value")))
    if set(materialized) - {"query", "key", "value"}:
        raise ModelingError("mindie_sd_norm_rope_output_roles_unknown")
    aliases = dict(meta.get("output_aliases") or {})
    reads, writes, compute = [], [], []
    reads.extend((_read("query", q_shape, dtype), _read("key", k_shape, dtype),
                  _read("value", v_shape, dtype)))
    if enc_q:
        reads.append(_read("encoder_query", enc_q, dtype))
    if enc_k:
        reads.append(_read("encoder_key", enc_k, dtype))
    if enc_v:
        reads.append(_read("encoder_value", enc_v, dtype))
    q_src, k_src, v_src = "query", "key", "value"
    enc_q_src = "encoder_query" if enc_q else None
    enc_k_src = "encoder_key" if enc_k else None
    norm_ops = {0: 0.0, 1: 6.0, 2: 7.0, 3: 4.0, 4: 5.0}
    stats_sources = {}
    if norm_type:
        q_norm_inputs, k_norm_inputs = ["query"], ["key"]
        if norm_type in (2, 4):
            reads.append(_read("norm_query_weight", (d,), dtype))
            reads.append(_read("norm_key_weight", (d,), dtype))
            q_norm_inputs.append("norm_query_weight")
            k_norm_inputs.append("norm_key_weight")
            if norm_type == 2:
                reads.append(_read("norm_query_bias", (d,), dtype))
                reads.append(_read("norm_key_bias", (d,), dtype))
                q_norm_inputs.append("norm_query_bias")
                k_norm_inputs.append("norm_key_bias")
        compute.append({"name": "query_norm", "kind": "vector",
                        "inputs": tuple(q_norm_inputs),
                        "outputs": ("query_normed", "norm_query_mean", "norm_query_rstd"),
                        "vector_ops": float(b * sq * h * d * norm_ops[norm_type])})
        compute.append({"name": "key_norm", "kind": "vector",
                        "inputs": tuple(k_norm_inputs),
                        "outputs": ("key_normed", "norm_key_mean", "norm_key_rstd"),
                        "vector_ops": float(b * sk * h * d * norm_ops[norm_type])})
        q_src, k_src = "query_normed", "key_normed"
        stats_sources.update({"norm_query": "query_norm", "norm_key": "key_norm"})
    if enc_q and norm_added_type:
        enc_q_inputs = ["encoder_query"]
        if norm_added_type in (2, 4):
            reads.append(_read("norm_added_query_weight", (d,), dtype))
            enc_q_inputs.append("norm_added_query_weight")
            if norm_added_type == 2:
                reads.append(_read("norm_added_query_bias", (d,), dtype))
                enc_q_inputs.append("norm_added_query_bias")
        compute.append({"name": "encoder_query_norm", "kind": "vector",
                        "inputs": tuple(enc_q_inputs),
                        "outputs": ("encoder_query_normed",
                                    "norm_added_query_mean", "norm_added_query_rstd"),
                        "vector_ops": float(prod(enc_q) * norm_ops[norm_added_type])})
        enc_q_src = "encoder_query_normed"
        stats_sources["norm_added_query"] = "encoder_query_norm"
    if enc_k and norm_added_type:
        enc_k_inputs = ["encoder_key"]
        if norm_added_type in (2, 4):
            reads.append(_read("norm_added_key_weight", (d,), dtype))
            enc_k_inputs.append("norm_added_key_weight")
            if norm_added_type == 2:
                reads.append(_read("norm_added_key_bias", (d,), dtype))
                enc_k_inputs.append("norm_added_key_bias")
        compute.append({"name": "encoder_key_norm", "kind": "vector",
                        "inputs": tuple(enc_k_inputs),
                        "outputs": ("encoder_key_normed",
                                    "norm_added_key_mean", "norm_added_key_rstd"),
                        "vector_ops": float(prod(enc_k) * norm_ops[norm_added_type])})
        enc_k_src = "encoder_key_normed"
        stats_sources["norm_added_key"] = "encoder_key_norm"
    if rope_type:
        rope_len = int(meta.get("rope_len", max(sq, sk)))
        rope_shape = _as_shape(meta.get("rope_sin_shape", (rope_len, d // 2)),
                               "rope_sin")
        rope_dtype = str(meta.get("rope_dtype", "fp32"))
        reads.append(_read("rope_sin", rope_shape, rope_dtype))
        reads.append(_read("rope_cos", rope_shape, rope_dtype))
        for role, source, seq_len in (("query", q_src, sq), ("key", k_src, sk)):
            compute.append({
                "name": f"{role}_rope", "kind": "vector",
                "inputs": (source, "rope_sin", "rope_cos"),
                "outputs": (f"{role}_roped",),
                "vector_ops": float(b * h * min(rope_len, seq_len) * d * 2),
            })
        q_src, k_src = "query_roped", "key_roped"
    q_out = (b, h, sq + seq, d)
    k_out = (b, h, sk + sek, d)
    v_out = (b, h, sv + sev, d)
    output_shapes = {"query": q_out, "key": k_out, "value": v_out}
    for role_name in ("query", "key", "value"):
        shape = output_shapes[role_name]
        alias = aliases.get(role_name)
        material = role_name in materialized and alias is None
        if not material:
            writes.append(_role(role_name + "_output", shape, dtype,
                                materialized=False, alias=alias))
            continue
        if role_name == "query":
            inputs = [q_src] + ([enc_q_src] if enc_q_src else [])
        elif role_name == "key":
            inputs = [k_src] + ([enc_k_src] if enc_k_src else [])
        else:
            inputs = [v_src] + (["encoder_value"] if enc_v else [])
        compute_name = role_name + "_concat_transpose"
        compute.append({"name": compute_name, "kind": "layout",
                        "inputs": tuple(inputs),
                        "outputs": (role_name + "_output_value",),
                        "vector_ops": float(prod(shape))})
        writes.append({**_role(role_name + "_output", shape, dtype),
                       "source": compute_name})
    if training and norm_type in (1, 2):
        for name, length in (("norm_query", sq), ("norm_key", sk)):
            for suffix in ("mean", "rstd"):
                writes.append({**_role(f"{name}_{suffix}", (b, length, h, 1), "fp32"),
                               "source": stats_sources[name]})
    if training and norm_added_type in (1, 2):
        for name, shape in (("norm_added_query", enc_q), ("norm_added_key", enc_k)):
            if shape:
                for suffix in ("mean", "rstd"):
                    writes.append({**_role(f"{name}_{suffix}", (b, shape[1], h, 1),
                                           "fp32"),
                                   "source": stats_sources[name]})
    return _contract(reads, writes, compute, attributes={
        "norm_type": norm_type, "norm_added_type": norm_added_type,
        "rope_type": rope_type, "concat_order": concat_order,
        "is_training": training, "materialized_outputs": materialized,
        "output_aliases": aliases,
    }, basis="per_value_BSND_norm_rope_concat_dag",
        limitations=("encoder_stats_only_when_layer_norm_and_training",))


def _eagle_ffn_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_eagle_ffn_requires_matrix_x")
    m = prod(x_shape[:-1])
    k = x_shape[-1]
    w1 = meta.get("weight1_shape")
    w2 = meta.get("weight2_shape")
    if w1 is None or w2 is None:
        if op.flops > 0 and op.mte2_bytes > 0:
            reads = [{"name": "declared_operands", "shape": tuple(),
                      "dtype": dtype, "bytes": float(op.mte2_bytes),
                      "materialized": True, "alias": None,
                      "kind": "declared", "producer": None}]
            writes = [{"name": "y", "shape": x_shape, "dtype": dtype,
                       "bytes": float(op.mte3_bytes), "materialized": True,
                       "alias": None, "kind": "declared",
                       "source": "declared_ffn"}]
            compute = [{"name": "declared_ffn", "kind": "matrix",
                        "matrix_flops": ((dtype, float(op.flops)),),
                        "vector_ops": float(op.vector_ops),
                        "inputs": ("declared_operands",),
                        "outputs": ("declared_ffn_out",)}]
            return _contract(reads, writes, compute, attributes={},
                             basis="declared_ffn_flops_legacy",
                             limitations=("weight_shapes_not_declared",))
        raise ModelingError("mindie_sd_eagle_ffn_requires_weight1_and_weight2_shapes")
    w1 = _as_shape(w1, "weight1")
    w2 = _as_shape(w2, "weight2")
    if len(w1) != 2 or len(w2) != 2:
        raise ModelingError("mindie_sd_eagle_ffn_weights_must_be_matrix")
    if w1[1] == k:
        h_total, w1_layout = w1[0], "linear_HxK"
    elif w1[0] == k:
        h_total, w1_layout = w1[1], "canonical_KxH"
    else:
        raise ModelingError("mindie_sd_eagle_ffn_weight1_K_mismatch")
    activation = str(meta.get("activation", "silu")).lower()
    h_act = h_total
    if activation == "swiglu":
        if h_total % 2:
            raise ModelingError("mindie_sd_eagle_ffn_swiglu_requires_even_H")
        h_act = h_total // 2
    if w2[1] == h_act:
        n, w2_layout = w2[0], "linear_NxH"
    elif w2[0] == h_act:
        n, w2_layout = w2[1], "canonical_HxN"
    else:
        raise ModelingError("mindie_sd_eagle_ffn_weight2_H_mismatch")
    bias_dtype = str(meta.get("bias_dtype", dtype))
    bias1_shape = meta.get("bias1_shape")
    bias2_shape = meta.get("bias2_shape")
    if bias1_shape is not None:
        bias1_shape = _as_shape(bias1_shape, "bias1")
        if prod(bias1_shape) != h_total:
            raise ModelingError("mindie_sd_eagle_ffn_bias1_width_mismatch")
    if bias2_shape is not None:
        bias2_shape = _as_shape(bias2_shape, "bias2")
        if prod(bias2_shape) != n:
            raise ModelingError("mindie_sd_eagle_ffn_bias2_width_mismatch")
    out_shape = tuple(x_shape[:-1]) + (n,)
    hidden_bytes = float(m * h_act * dtype_bytes(dtype))
    reads = [_read("x", x_shape, dtype), _read("weight1", w1, dtype),
             _read("weight2", w2, dtype)]
    linear1_inputs = ["x", "weight1"]
    linear2_inputs = ["hidden", "weight2"]
    if bias1_shape is not None:
        reads.append(_read("bias1", bias1_shape, bias_dtype))
        linear1_inputs.append("bias1")
    if bias2_shape is not None:
        reads.append(_read("bias2", bias2_shape, bias_dtype))
        linear2_inputs.append("bias2")
    writes = [
        {"name": "hidden_store", "shape": (m, h_act), "dtype": dtype,
         "bytes": hidden_bytes, "materialized": True, "alias": None,
         "kind": "workspace", "source": "activation"},
        {"name": "y", "shape": out_shape, "dtype": dtype,
         "bytes": _bytes(out_shape, dtype), "materialized": True,
         "alias": None, "kind": "value", "source": "linear2"},
    ]
    reads.append({"name": "hidden", "shape": (m, h_act), "dtype": dtype,
                  "bytes": hidden_bytes, "materialized": True, "alias": None,
                  "kind": "workspace", "producer": "hidden_store"})
    activation_ops = 6.0 if activation == "swiglu" else 5.0
    compute = [
        {"name": "linear1", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * m * h_total * k),),
         "inputs": tuple(linear1_inputs), "outputs": ("linear1_out",)},
        {"name": "activation", "kind": "vector",
         "vector_ops": float(m * h_act * activation_ops),
         "inputs": ("linear1_out",), "outputs": ("activation_out",)},
        {"name": "linear2", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * m * n * h_act),),
         "inputs": tuple(linear2_inputs), "outputs": ("linear2_out",)},
    ]
    return _contract(reads, writes, compute, attributes={
        "activation": activation, "weight1_shape": w1, "weight2_shape": w2,
        "weight1_layout": w1_layout, "weight2_layout": w2_layout,
        "bias_dtype": bias_dtype, "bias1_shape": bias1_shape,
        "bias2_shape": bias2_shape, "hidden_shape": (m, h_act),
        "hidden_dtype": dtype, "hidden_barrier": "hidden_store",
    }, basis="two_linear_activation_hidden_gm_materialization",
        limitations=("hidden_store_is_a_real_gm_value",))


def _sparse_shape(meta, op, name, default):
    raw = meta.get(f"{name}_shape", default)
    shape = _as_shape(raw, name)
    if len(shape) == 4:
        layout = str(meta.get("layout", meta.get(f"{name}_layout", "BSND"))).upper()
        if layout == "BNSD":
            b, h, s, d = shape
            canonical = (b, s, h, d)
        else:
            b, s, h, d = shape
            canonical = (b, s, h, d)
            layout = "BSND"
        lengths = tuple(int(v) for v in meta.get(
            f"{name}_lengths", (s,) * b))
        if len(lengths) != b or any(v < 0 or v > s for v in lengths):
            raise ModelingError(f"mindie_sd_sparse_{name}_lengths_invalid")
        return canonical, lengths, layout
    if len(shape) == 3:
        total, h, d = shape
        lengths = tuple(int(v) for v in meta.get(
            f"{name}_lengths", meta.get("sequence_lengths", (total,))))
        if not lengths or any(v < 0 for v in lengths) or sum(lengths) > total:
            raise ModelingError(f"mindie_sd_sparse_{name}_TND_lengths_invalid")
        return shape, lengths, "TND"
    raise ModelingError(f"mindie_sd_sparse_{name}_requires_BNSD_or_TND")


def _sparse_contract(op, meta, dtype, returns, resolved, options):
    ratio = _ratio(meta, options.get("sparsity_ratio"), "sparsity_ratio")
    if not (0.0 <= ratio < 1.0):
        raise ModelingError(f"mindie_sd_invalid_sparsity_ratio:{ratio}")
    block = _positive_int(meta, options.get("block_size"), "block_size")
    q = meta.get("query_shape")
    k = meta.get("key_shape")
    v = meta.get("value_shape")
    if q is None or k is None or v is None:
        if op.flops <= 0:
            raise ModelingError("mindie_sd_sparse_requires_declared_logical_flops")
        q_legacy = _shape(op)
        active = float(op.flops) * (1.0 - ratio)
        reads = [{"name": "declared_operand", "shape": q_legacy, "dtype": dtype,
                  "bytes": float(op.mte2_bytes), "materialized": True,
                  "alias": None, "kind": "legacy", "producer": None}]
        writes = [{"name": "attention_out", "shape": q_legacy, "dtype": dtype,
                   "bytes": float(op.mte3_bytes), "materialized": True,
                   "alias": None, "kind": "legacy", "source": "declared_sparse"}]
        compute = [{"name": "declared_sparse", "kind": "matrix",
                    "matrix_flops": ((dtype, active),),
                    "vector_ops": float(op.vector_ops),
                    "inputs": ("declared_operand",),
                    "outputs": ("attention_out",)}]
        return _contract(reads, writes, compute, attributes={
            "sparsity_ratio": ratio, "block_size": block,
            "kv_dtype": options.get("kv_dtype") or meta.get("kv_dtype"),
        }, basis="declared_single_shape_legacy",
            limitations=("qkv_roles_not_declared", "aggregate_kv_not_inferred"))
    q_shape, q_lengths, q_layout = _sparse_shape(meta, op, "query", q)
    k_shape, k_lengths, k_layout = _sparse_shape(meta, op, "key", k)
    v_shape, v_lengths, v_layout = _sparse_shape(meta, op, "value", v)
    if not (q_layout == k_layout == v_layout):
        raise ModelingError("mindie_sd_sparse_layouts_must_match")
    if len(q_lengths) != len(k_lengths) or len(q_lengths) != len(v_lengths):
        raise ModelingError("mindie_sd_sparse_batch_lengths_mismatch")
    b = len(q_lengths)
    if len(q_shape) == 4:
        _, _, hq, dq = q_shape
        _, _, hkv, dk = k_shape
        _, _, hv, dv = v_shape
    else:
        _, hq, dq = q_shape
        _, hkv, dk = k_shape
        _, hv, dv = v_shape
    if hkv != hv or dq != dk or hq % hkv:
        raise ModelingError("mindie_sd_sparse_qkv_head_geometry_mismatch")
    q_dtype = str(meta.get("q_dtype", meta.get("query_dtype", dtype)))
    k_dtype = str(meta.get("k_dtype", meta.get("kv_dtype", dtype)))
    v_dtype = str(meta.get("v_dtype", meta.get("kv_dtype", dtype)))
    q_bytes = sum(q_lengths) * hq * dq * dtype_bytes(q_dtype)
    per_batch_pairs = []
    for b_index, (ql, kl) in enumerate(zip(q_lengths, k_lengths)):
        per_batch_pairs.append((b_index, (ql + block - 1) // block,
                                (kl + block - 1) // block))
    total_pairs = sum(qb * kb for _, qb, kb in per_batch_pairs)
    mask_values = meta.get("block_mask")
    flat = None
    head_flags = None
    if mask_values is not None:
        if (isinstance(mask_values, (str, bytes))
                or not isinstance(mask_values, (tuple, list))):
            raise ModelingError("mindie_sd_sparse_block_mask_invalid")
        if (len(mask_values) == total_pairs
                and all(isinstance(value, (int, bool)) for value in mask_values)):
            flat = tuple(1 if value else 0 for value in mask_values)
        elif (all(isinstance(value, (int, bool)) for value in mask_values)
              and per_batch_pairs
              and all(qb * kb == len(mask_values) for _, qb, kb in per_batch_pairs)):
            flat = tuple(1 if value else 0 for value in mask_values) * b
        else:
            head_rows = []
            for batch_index, qb, kb in per_batch_pairs:
                if batch_index >= len(mask_values):
                    raise ModelingError("mindie_sd_sparse_block_mask_batch_missing")
                row = mask_values[batch_index]
                if isinstance(row, (str, bytes)):
                    raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
                if len(row) == hq:
                    heads = []
                    for head in row:
                        if isinstance(head, (str, bytes)):
                            raise ModelingError(
                                "mindie_sd_sparse_block_mask_shape_mismatch")
                        if len(head) == qb * kb:
                            heads.append(tuple(1 if value else 0 for value in head))
                        elif (len(head) == qb
                              and all(not isinstance(item, (str, bytes))
                                      and len(item) == kb for item in head)):
                            heads.append(tuple(
                                1 if head[qi][ki] else 0
                                for qi in range(qb) for ki in range(kb)))
                        else:
                            raise ModelingError(
                                "mindie_sd_sparse_block_mask_shape_mismatch")
                    head_rows.append(tuple(heads))
                elif len(row) == qb * kb:
                    flags = tuple(1 if value else 0 for value in row)
                    head_rows.append(tuple(flags for _ in range(hq)))
                else:
                    raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
            head_flags = tuple(head_rows)
            total_head_pairs = sum(len(row[0]) for row in head_flags) * hq
            if total_head_pairs != sum(qb * kb for _, qb, kb in per_batch_pairs) * hq:
                raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
    qk_flops = pv_flops = 0.0
    active_rows = active_pairs = 0
    kv_blocks = set()
    if head_flags is None:
        cursor = 0
        for batch_index, qb, kb in per_batch_pairs:
            ql, kl = q_lengths[batch_index], k_lengths[batch_index]
            selected = []
            if flat is None:
                selected_count = max(0, min(qb * kb, int(round((1.0 - ratio) * qb * kb))))
                for index in range(selected_count):
                    qi, ki = divmod(index, kb)
                    selected.append((qi, ki))
            else:
                for qi in range(qb):
                    for ki in range(kb):
                        if flat[cursor]:
                            selected.append((qi, ki))
                        cursor += 1
            for qi, ki in selected:
                qrows = min(block, ql - qi * block)
                rows = min(block, kl - ki * block)
                active_pairs += 1
                active_rows += qrows * rows
                qk_flops += 2 * qrows * rows * hq * dq
                pv_flops += 2 * qrows * rows * hq * dv
                kv_blocks.add((batch_index, ki))
    else:
        for batch_index, (qb, kb) in enumerate(
                (qb, kb) for _, qb, kb in per_batch_pairs):
            ql, kl = q_lengths[batch_index], k_lengths[batch_index]
            for head in range(hq):
                flags = head_flags[batch_index][head]
                for qi in range(qb):
                    for ki in range(kb):
                        if not flags[qi * kb + ki]:
                            continue
                        qrows = min(block, ql - qi * block)
                        rows = min(block, kl - ki * block)
                        active_pairs += 1
                        active_rows += qrows * rows
                        qk_flops += 2 * qrows * rows * dq
                        pv_flops += 2 * qrows * rows * dv
                        kv_blocks.add((batch_index, ki))
    kv_load_bytes = 0.0
    for batch_index, ki in kv_blocks:
        rows = min(block, k_lengths[batch_index] - ki * block)
        kv_load_bytes += rows * hkv * (dk + dv) * dtype_bytes(k_dtype)
    if head_flags is not None:
        mask_bytes = float(meta.get(
            "mask_bytes",
            sum(qb * kb for _, qb, kb in per_batch_pairs) * hq
            * dtype_bytes(str(meta.get("mask_dtype", "int8")))))
    else:
        mask_bytes = float(meta.get("mask_bytes", total_pairs * 4))
    reads = [_read("query", q_shape, q_dtype),
             {"name": "key", "shape": k_shape, "dtype": k_dtype,
              "bytes": 0.0, "materialized": True, "alias": None,
              "kind": "paged_kv_view", "producer": None},
             {"name": "value", "shape": v_shape, "dtype": v_dtype,
              "bytes": 0.0, "materialized": True, "alias": None,
              "kind": "paged_kv_view", "producer": None},
             {"name": "kv_load", "shape": (active_pairs,), "dtype": k_dtype,
              "bytes": kv_load_bytes, "materialized": True, "alias": None,
              "kind": "block_load", "producer": None},
             {"name": "block_mask", "shape": (total_pairs,), "dtype": "int32",
              "bytes": mask_bytes, "materialized": True, "alias": None,
              "kind": "mask", "producer": None}]
    for name, shape_key, dt_key in (
            ("q_scale", "q_scale_shape", "q_scale_dtype"),
            ("k_scale", "k_scale_shape", "k_scale_dtype"),
            ("v_scale", "v_scale_shape", "v_scale_dtype")):
        shape = meta.get(shape_key)
        if shape is not None:
            reads.append(_read(name, _as_shape(shape, name),
                               str(meta.get(dt_key, "fp32"))))
    if q_layout == "BNSD":
        out_shape = (b, hq, max(q_lengths), dv)
    elif q_layout == "TND":
        out_shape = (sum(q_lengths), hq, dv)
    else:
        out_shape = (b, max(q_lengths), hq, dv)
    out_dtype = str(meta.get("output_dtype", dtype))
    writes = [{**_role("attention_out", out_shape, out_dtype), "source": "pv"}]
    lse_shape = meta.get("lse_shape")
    if lse_shape is None and bool(meta.get("return_softmax_lse", False)):
        lse_shape = ((b, hq, max(q_lengths), 1) if q_layout == "BNSD"
                     else (sum(q_lengths), hq))
    if lse_shape is not None:
        lse_shape = _as_shape(lse_shape, "lse")
        writes.append({**_role("softmax_lse", lse_shape, "fp32"),
                       "source": "softmax"})
    scale_inputs = tuple(item["name"] for item in reads
                         if item["name"] in {"q_scale", "k_scale", "v_scale"})
    compute = [
        {"name": "qk", "kind": "matrix",
         "matrix_flops": ((q_dtype, qk_flops),),
         "inputs": ("query", "kv_load", "block_mask") + scale_inputs,
         "outputs": ("qk_scores",)},
        {"name": "softmax", "kind": "vector",
         "vector_ops": float(active_rows * hq * 2),
         "inputs": ("qk_scores", "block_mask"),
         "outputs": ("probs", "softmax_lse")},
        {"name": "pv", "kind": "matrix",
         "matrix_flops": ((v_dtype, pv_flops),),
         "inputs": ("probs", "kv_load") + scale_inputs,
         "outputs": ("attention_out",)},
    ]
    return _contract(reads, writes, compute, attributes={
        "sparsity_ratio": ratio, "block_size": block,
        "active_block_pairs": active_pairs, "block_pairs": total_pairs,
        "active_rows": active_rows, "layout": q_layout, "batch": b,
        "q_lengths": q_lengths, "k_lengths": k_lengths, "v_lengths": v_lengths,
        "kv_dtype": k_dtype, "q_dtype": q_dtype, "k_dtype": k_dtype,
        "v_dtype": v_dtype, "output_dtype": out_dtype,
        "mask_bytes": mask_bytes, "active_kv_blocks": len(kv_blocks),
        "mask_layout": ("B,H,Qblock,KVblock" if head_flags is not None
                        else "global"),
        "kv_load_bytes": kv_load_bytes,
    }, basis="BNSD_TND_batch_heads_actual_active_blocks",
        limitations=("kv_block_reuse_modelled_per_batch_block",))


def _adalayernorm_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_adalayernorm_requires_at_least_2d_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    scale_shape = _as_shape(meta.get("scale_shape", (width,)), "scale")
    shift_shape = _as_shape(meta.get("shift_shape", (width,)), "shift")
    batch = x_shape[0] if len(x_shape) >= 2 else 1
    def broadcast_ok(shape):
        return (prod(shape) in {width, rows * width}
                or (len(shape) == 2 and shape[0] == batch and shape[1] == width))
    if not broadcast_ok(scale_shape) or not broadcast_ok(shift_shape):
        raise ModelingError("mindie_sd_adalayernorm_modulation_width_mismatch")
    reads = [_read("x", x_shape, dtype), _read("scale", scale_shape, dtype),
             _read("shift", shift_shape, dtype)]
    weight_shape = meta.get("weight_shape")
    bias_shape = meta.get("bias_shape")
    if weight_shape is not None:
        reads.append(_read("weight", _as_shape(weight_shape, "weight"), dtype))
    if bias_shape is not None:
        reads.append(_read("bias", _as_shape(bias_shape, "bias"), dtype))
    writes = [{**_role("y", x_shape, dtype), "source": "norm"}]
    compute = [{"name": "norm", "kind": "vector",
                "vector_ops": float(rows * width * 5),
                "inputs": tuple(item["name"] for item in reads),
                "outputs": ("y_value",)}]
    if bool(meta.get("return_stats", False)):
        mean_shape = tuple(x_shape[:-1]) + (1,)
        for name in ("mean", "rstd"):
            writes.append({**_role(name, mean_shape, dtype), "source": "norm"})
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "scale_shape": scale_shape, "shift_shape": shift_shape,
        "weight_shape": weight_shape, "bias_shape": bias_shape,
    }, basis="three_dimensional_adaln_modulation_and_optional_affine")


def _laurel_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_laurel_requires_matrix_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    rank = _positive_int(meta, None, "low_rank")
    wa_shape = meta.get("weight_a_shape", (width, rank))
    wb_shape = meta.get("weight_b_shape", (rank, width))
    wa_shape = _as_shape(wa_shape, "weight_a")
    wb_shape = _as_shape(wb_shape, "weight_b")
    if len(wa_shape) != 2 or len(wb_shape) != 2 or wa_shape[-1] != wb_shape[0]:
        raise ModelingError("mindie_sd_laurel_weight_geometry_mismatch")
    reads = [_read("x", x_shape, dtype), _read("weight_a", wa_shape, dtype),
             _read("weight_b", wb_shape, dtype)]
    writes = [{**_role("y", x_shape, dtype), "source": "residual"}]
    compute = [
        {"name": "low_rank_a", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * rows * wa_shape[-1] * width),),
         "inputs": ("x", "weight_a"), "outputs": ("low_rank_hidden",)},
        {"name": "low_rank_b", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * rows * wb_shape[-1] * wb_shape[0]),),
         "inputs": ("low_rank_hidden", "weight_b"), "outputs": ("low_rank_out",)},
        {"name": "residual", "kind": "vector",
         "vector_ops": float(rows * width),
         "inputs": ("x", "low_rank_out"), "outputs": ("y_value",)},
    ]
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "low_rank": rank,
        "weight_a_shape": wa_shape, "weight_b_shape": wb_shape,
    }, basis="two_low_rank_matrix_products_plus_residual")


def _quant_four_over_six_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_quant_four_over_six_requires_matrix_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    groups = (width + 31) // 32
    padded_groups = (groups + 1) // 2 * 2
    payload_width = (width + 1) // 2
    reads = [_read("x", x_shape, dtype)]
    if meta.get("scale_shape") is not None:
        reads.append(_read("scale", _as_shape(meta["scale_shape"], "scale"), "fp32"))
    payload_shape = tuple(x_shape[:-1]) + (payload_width,)
    scale_shape = tuple(x_shape[:-1]) + (padded_groups // 2, 2)
    writes = [
        {**_role("mxfp4_payload", payload_shape, "uint8"), "source": "quantize"},
        {**_role("mxfp4_scale", scale_shape, "fp8_e8m0"), "source": "quantize"},
    ]
    compute = [{"name": "quantize", "kind": "vector",
                "vector_ops": float(rows * padded_groups * 32 * 4),
                "inputs": tuple(item["name"] for item in reads),
                "outputs": ("mxfp4_payload", "mxfp4_scale")}]
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "logical_width": width,
        "num_scale_groups": groups, "padded_scale_groups": padded_groups,
        "payload_width": payload_width,
        "input_dtype": dtype, "output_payload_dtype": "fp4_e2m1",
        "payload_storage_dtype": "uint8",
        "payload_bytes": prod(payload_shape), "scale_bytes": prod(scale_shape),
        "output_scale_format": "fp8_e8m0", "scale_group_size": 32,
        "scale_padding": padded_groups - groups,
    }, basis="a5_fp4_32_group_payload_and_even_padded_e8m0_scale")


def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    sparsity_ratio: float | None = None,
    low_rank: int | None = None,
    block_size: int | None = None,
    kv_dtype: str | None = None,
) -> OpSpec:
    """Lower one exact semantic occurrence into a MindIE-SD custom operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = MINDIE_SD_CUSTOM_OPS[resolved]
    if op.family not in spec["families"] and op.recipe not in spec["families"]:
        raise ModelingError(f"mindie_sd_custom_op_family_mismatch:{resolved}:{op.family}")
    if soc_family is not None and soc_family not in spec["supported_socs"]:
        raise ModelingError(f"mindie_sd_custom_op_soc_unsupported:{resolved}:{soc_family}")

    meta = dict(op.metadata)
    dtype = _input_dtype(op, meta)
    if dtype not in spec["input_dtypes"]:
        raise ModelingError(
            f"mindie_sd_custom_op_input_dtype_unsupported:{resolved}:{dtype}")

    flops = float(op.flops)
    vec_ops = float(op.vector_ops)
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    returns = tuple(op.returns)
    shape = _shape(op)
    rows, width = _tokens_width(op)
    elem_bytes = dtype_bytes(dtype)
    contract = None
    matrix_flops = op.matrix_flops
    options = {"sparsity_ratio": sparsity_ratio, "block_size": block_size,
               "low_rank": low_rank, "kv_dtype": kv_dtype}

    if spec.get("sparse"):
        contract = _sparse_contract(op, meta, dtype, returns, resolved, options)
        if contract.get("attributes", {}).get("kv_dtype") is None and spec.get("quantized_cache"):
            raise ModelingError("mindie_sd_eagle_qbsa_requires_declared_cache_dtype")
    elif resolved == "norm_rope_concat":
        contract = _norm_rope_contract(op, meta, dtype, returns)
    elif resolved == "adalayernorm":
        contract = _adalayernorm_contract(op, meta, dtype, returns)
    elif resolved == "laurel":
        if low_rank is not None:
            meta["low_rank"] = int(low_rank)
        contract = _laurel_contract(op, meta, dtype, returns)
    elif resolved == "quant_four_over_six_a5":
        contract = _quant_four_over_six_contract(op, meta, dtype, returns)
    elif resolved == "eagle_ffn":
        contract = _eagle_ffn_contract(op, meta, dtype, returns)
    if contract is not None:
        read_bytes = float(contract["mte2_bytes"])
        write_bytes = float(contract["mte3_bytes"])
        vec_ops = float(contract["vector_ops"])
        flops = float(contract["logical_flops"])
        matrix_flops = tuple(contract["matrix_flops"])
        meta["mindie_sd_contract"] = contract
        meta.update(contract["attributes"])
        if resolved == "norm_rope_concat":
            extra = (
                "query_output", "key_output", "value_output",
                "norm_query_mean", "norm_query_rstd", "norm_key_mean",
                "norm_key_rstd", "norm_added_query_mean",
                "norm_added_query_rstd", "norm_added_key_mean",
                "norm_added_key_rstd",
            )
        elif resolved == "eagle_ffn":
            extra = ("y",)
        elif spec.get("sparse"):
            extra = ("attention_out",)
            if "softmax_lse" in {item["name"] for item in contract["writes"]}:
                extra = (*extra, "softmax_lse")
        elif resolved == "adalayernorm":
            extra = ("y",) + tuple(item["name"] for item in contract["writes"]
                                    if item["name"] in {"mean", "rstd"})
        elif resolved == "laurel":
            extra = ("y",)
            meta["laurel_rank"] = int(contract["attributes"]["low_rank"])
        elif resolved == "quant_four_over_six_a5":
            extra = ("mxfp4_payload", "mxfp4_scale")
        else:
            extra = ()
        returns = tuple(dict.fromkeys((*returns, *extra)))
    meta.update({
        "mindie_sd_custom_op": f"mindie_sd_{resolved}",
        "native_vendor": MINDIE_SD_VENDOR,
        "native_kind": spec["native_kind"],
        "bound": spec["bound"],
        "native_implementation": {
            "framework": "mindie-sd",
            "vendor": MINDIE_SD_VENDOR,
            "native_kinds": (spec["native_kind"],),
            "op_name": f"mindie_sd_{resolved}",
            "selection": "explicit_caller_implementation",
            "soc_family": soc_family,
        },
    })
    if contract is not None:
        public = next((item for item in contract["writes"]
                       if item.get("materialized", True)
                       and item.get("kind") in {"value", "legacy"}
                       and item["name"] not in {"mean", "rstd"}),
                      next((item for item in contract["writes"]
                            if item.get("materialized", True)), None))
        output_shape = tuple(public["shape"]) if public else tuple(op.local_shape)
    else:
        output_shape = tuple(op.local_shape)
    return replace(
        op,
        recipe=f"mindie_sd_{resolved}",
        flops=flops,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        matrix_flops=matrix_flops,
        local_shape=output_shape,
        padded_shape=output_shape,
        launch_shape=output_shape,
        metadata=tuple(meta.items()),
    )

def lower_mindie_sd_custom_op(
    op_spec: OpSpec,
    hardware: Mapping[str, Any] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Lower one operator and return its work plus a disclosed Roofline view.

    No hardware rate is defaulted.  Without declared ``cube_tflops``,
    ``vector_tflops`` and ``hbm_bandwidth_gbps`` the result carries work and a
    basis string instead of an invented duration.
    """
    lowered = lower_op(op_spec, **options)
    meta = dict(lowered.metadata)
    result: dict[str, Any] = {
        "op_name": meta["mindie_sd_custom_op"],
        "native_kind": meta["native_kind"],
        "family": lowered.family,
        "bound": meta["bound"],
        "cube_flops": lowered.flops,
        "vector_flops": lowered.vector_ops,
        "memory_bytes": lowered.hbm_bytes,
        "sparsity_ratio": float(meta.get("sparsity_ratio", 0.0)),
        **({"laurel_rank": meta["laurel_rank"]} if "laurel_rank" in meta else {}),
        "hardware": dict(hardware or {}),
        "estimate_basis": "requires_declared_hardware_rates",
        "estimated_us": None,
    }
    hw = dict(hardware or {})
    required = ("cube_tflops", "vector_tflops", "hbm_bandwidth_gbps")
    if hw:
        if any(key not in hw for key in required):
            result["estimate_basis"] = "declared_hardware_rates_incomplete"
            result["lowered_op"] = lowered
            return result
        cube_tflops = float(hw["cube_tflops"])
        vec_tflops = float(hw["vector_tflops"])
        hbm_gbps = float(hw["hbm_bandwidth_gbps"])
        if min(cube_tflops, vec_tflops, hbm_gbps) <= 0:
            raise ModelingError("mindie_sd_hardware_rates_must_be_positive")
        cube_us = lowered.flops / (cube_tflops * 1e6)
        vec_us = lowered.vector_ops / (vec_tflops * 1e6)
        mem_us = lowered.hbm_bytes / (hbm_gbps * 1e3)
        result["estimated_us"] = max(cube_us, vec_us, mem_us)
        result["estimate_basis"] = "declared_hardware_roofline"
    result["lowered_op"] = lowered
    return result

def compute_mindie_sd_effective_denoising_steps(
    infer_steps: int,
    warmup_steps: int = 2,
    cache_interval: int = 2,
    cache_hit_ratio: float | None = None,
) -> dict[str, float | int]:
    """Model DiTCache/TeaCache/TaylorSeer denoising step skipping."""
    total = int(infer_steps)
    warmup = max(0, min(int(warmup_steps), total))
    interval = max(1, int(cache_interval))
    if total <= 0:
        raise ModelingError(f"mindie_sd_invalid_infer_steps:{infer_steps}")
    remaining = max(0, total - warmup)
    if cache_hit_ratio is not None:
        hit = max(0.0, min(1.0, float(cache_hit_ratio)))
        cached = int(round(remaining * hit))
        computed = max(1, total - cached)
        cached = total - computed
    else:
        computed_after_warmup = math.ceil(remaining / interval) if remaining > 0 else 0
        computed = max(1, warmup + computed_after_warmup)
        cached = max(0, total - computed)
    ratio = float(computed) / float(total)
    return {
        "infer_steps": total,
        "warmup_steps": warmup,
        "cache_interval": interval,
        "computed_steps": computed,
        "cached_steps": cached,
        "cache_hit_ratio": float(cached) / float(total),
        "effective_step_ratio": ratio,
        "speedup_factor": float(total) / float(computed),
    }

__all__ = [
    "CAPABILITIES",
    "MINDIE_SD_CUSTOM_OPS",
    "MINDIE_SD_VENDOR",
    "capabilities",
    "compute_mindie_sd_effective_denoising_steps",
    "lower_mindie_sd_custom_op",
    "lower_op",
    "recipes",
]
