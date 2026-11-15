"""Explicit xLLM ``custom_xllm_math`` lowering over the common resource envelope.

[Role] Bind an explicitly selected xLLM custom operator to its declared
  semantic occurrence.  Six source operators cover MFusion/MLA, sparse
  attention LSE, fused grouped gate/up+SwiGLU+quant, GDN decode, OneRec beam
  selection and A5 paged KV-cache writes.
[Invariants] Selection is exact: a generic family is never silently rewritten
  into a custom operator.  The input dtype is read from the occurrence, never
  defaulted.  Existing semantic work is preserved; a representation-specific
  output (LSE, quantized payload, paged write) is added once.  The fused
  grouped operator requires an occurrence/region that already declares the
  fused semantics; applying it to an ordinary gate/up matrix is an error.
  A2/A3/A5 SoC support is declared per source operator and enforced when the
  caller declares one.
[Timeouts & Locks] Bounded local arithmetic; no subprocesses, sleeps or locks.
[Side-Effect Policy] Returns immutable ``OpSpec`` records; no I/O or device.
"""
from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import Any

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import Allocation, KernelProgram, Stage

XLLM_VENDOR = "custom_xllm_math"
FUSED_GROUPED_RECIPE = "moe_grouped_matmul_swiglu_quant"
FP8_GROUPED_OWNER_RECIPE = "ascend_mxfp8_grouped_swiglu_quant"
_ATTENTION_DTYPES = ("bf16", "fp16")
_QUANT_OUTPUT_DTYPES = ("int8", "fp8_e4m3", "fp8_e5m2")

XLLM_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "smla_va": {
        "native_kind": "SmlaVa",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/smla_va",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("smla_va", "mla_attention", "paged_mla_attention", "attention"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("op_name", "soc_family"),
        "atomic_occurrence": True,
    },
    "sparse_flash_attention_lse": {
        "native_kind": "SparseFlashAttentionLse",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/sparse_flash_attention_lse",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("sparse_flash_attention_lse", "indexed_attention", "mla_attention",
                     "attention"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("op_name", "soc_family", "return_lse"),
        "atomic_occurrence": True,
    },
    "moe_grouped_matmul_swiglu_quant": {
        "native_kind": "MoeGroupedMatmulSwigluQuant",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/moe_grouped_matmul_swiglu_quant",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("moe_grouped_matmul_swiglu_quant", "grouped_gemm"),
        "input_dtypes": ("int8",),
        "attributes": ("op_name", "soc_family", "weight_scale_format",
                       "output_format"),
        "atomic_occurrence": False,
        "requires_declared_fusion": True,
        "quantization": "int8_per_channel_float_scale",
        "fp8_owner_recipe": FP8_GROUPED_OWNER_RECIPE,
    },
    "mega_gdn_decode": {
        "native_kind": "MegaGdnDecode",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/mega_gdn_decode",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("mega_gdn_decode", "gdn_decode", "recurrent_vector",
                     "recurrent_scan"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("op_name", "soc_family"),
        "atomic_occurrence": True,
    },
    "onerec_final_beam_select": {
        "native_kind": "OnerecFinalBeamSelect",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/onerec_final_beam_select",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("onerec_final_beam_select", "beam_select", "elementwise"),
        "input_dtypes": ("bf16", "fp16", "fp32"),
        "attributes": ("op_name", "soc_family", "beam_width"),
        "atomic_occurrence": True,
    },
    "reshape_and_cache_a5": {
        "native_kind": "ReshapeAndCacheA5",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/reshape_and_cache_a5",
        "supported_socs": ("Ascend950",),
        "families": ("reshape_and_cache_a5", "kv_cache_write", "cache_store"),
        "input_dtypes": ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"),
        "attributes": ("op_name", "soc_family", "kv_page_size"),
        "atomic_occurrence": True,
        "a5_only": True,
    },
}

CAPABILITIES = frozenset({
    "explicit_caller_implementation",
    "custom_xllm_math",
    "A2",
    "A3",
    "A5",
    "atomic_occurrence_lowering",
    "dtype_checked",
    "declared_work_preserved",
    "fused_grouped_gemm_requires_declared_region",
    "int8_grouped_gemm",
    "fp8_mx32_owned_by_f_matrix",
})

# Exact vendor semantic names may resolve without an option.  Shared families
# (elementwise, grouped_gemm, attention, cache_store) do not: those require an
# explicit implementation name so a generic graph node is never reinterpreted.
_EXACT_SEMANTIC_ALIASES = {
    "smla_va": "smla_va",
    "mla_attention": "smla_va",
    "paged_mla_attention": "smla_va",
    "sparse_flash_attention_lse": "sparse_flash_attention_lse",
    "indexed_attention": "sparse_flash_attention_lse",
    "moe_grouped_matmul_swiglu_quant": "moe_grouped_matmul_swiglu_quant",
    "mega_gdn_decode": "mega_gdn_decode",
    "gdn_decode": "mega_gdn_decode",
    "onerec_final_beam_select": "onerec_final_beam_select",
    "beam_select": "onerec_final_beam_select",
    "reshape_and_cache_a5": "reshape_and_cache_a5",
    "kv_cache_write": "reshape_and_cache_a5",
}

def capabilities() -> dict[str, Any]:
    """Machine-readable registry of the declarations behind this wrapper."""
    from .implementations import IMPLEMENTATIONS

    implementations = tuple(
        name for name, item in IMPLEMENTATIONS.items()
        if item.module == __name__)
    selector_operators = {
        operator for name, item in IMPLEMENTATIONS.items()
        if item.module == __name__ for operator in item.operators}
    return {
        "vendor": XLLM_VENDOR,
        "capabilities": tuple(sorted(CAPABILITIES)),
        "implementations": implementations,
        "semantic_aliases": dict(_EXACT_SEMANTIC_ALIASES),
        "operators": {
            name: {
                "native_kind": spec["native_kind"],
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
                "atomic_occurrence": bool(spec["atomic_occurrence"]),
                "requires_declared_fusion": bool(
                    spec.get("requires_declared_fusion")),
                "quantization": spec.get("quantization"),
                "fp8_owner_recipe": spec.get("fp8_owner_recipe"),
            }
            for name, spec in XLLM_CUSTOM_OPS.items()
        },
    }

def recipes() -> dict[str, Any]:
    """Register the complete fused grouped call; other vendors use envelopes."""
    return {
        FUSED_GROUPED_RECIPE: PlannedRecipe(_build_fused_grouped),
        "mega_gdn_decode": PlannedRecipe(_build_mega_gdn_decode),
        "reshape_and_cache_a5_plan": PlannedRecipe(_build_cache_a5),
    }

def _declared_dtype(op: OpSpec, meta: dict[str, Any]) -> str:
    dtype = op.dtype or meta.get("input_format") or meta.get("compute_format")
    if not dtype:
        raise ModelingError("xllm_custom_op_requires_declared_input_dtype")
    return str(dtype)

def _shape(op: OpSpec) -> tuple[int, ...]:
    shape = tuple(int(value) for value in op.local_shape)
    if not shape or any(value < 0 for value in shape):
        raise ModelingError("xllm_custom_op_requires_nonnegative_local_shape")
    return shape

def _tokens_width(op: OpSpec) -> tuple[int, int]:
    shape = _shape(op)
    return shape[0], prod(shape[1:]) if len(shape) > 1 else 1

def _attention_heads(op: OpSpec, meta: dict[str, Any]) -> int:
    shape = _shape(op)
    if len(shape) >= 5:
        heads = shape[2]
    elif len(shape) == 4:
        heads = shape[2]
    elif len(shape) == 3:
        heads = shape[1]
    else:
        heads = int(meta.get("heads", 1) or 1)
    if heads < 1:
        raise ModelingError("xllm_attention_requires_positive_heads")
    return heads

def _cache_row_bytes(op: OpSpec, meta: dict[str, Any], dtype: str) -> tuple[int, str]:
    """Typed paged-cache row size.

    ``cache_store`` declares a byte row; the vendor semantic shape declares
    elements.  When the source view is present, K/V shapes and their own dtypes
    are authoritative and mixed K/V dtypes are not collapsed.
    """
    view = dict(meta.get("cache_tensor_view") or {})
    if view:
        total = 0
        for shape_key, dtype_key in (("key_shape", "key_dtype"),
                                     ("value_shape", "value_dtype")):
            shape = view.get(shape_key)
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                total += prod(int(value) for value in shape[1:]) * dtype_bytes(
                    view.get(dtype_key) or dtype)
        if total > 0:
            return total, "cache_tensor_view_key_value_bytes"
    shape = _shape(op)
    if op.family == "cache_store" or meta.get("cache_kind"):
        return shape[1], "cache_store_declared_row_bytes"
    return prod(shape[1:]) * dtype_bytes(dtype), "vendor_shape_elements_times_dtype"

def _matmul_dims(op: OpSpec, meta: dict[str, Any]) -> tuple[int, int, int]:
    shape = _shape(op)
    if len(shape) == 3:
        rows, output_width, input_width = shape
    elif len(shape) == 2:
        rows, output_width = shape
        input_width = meta.get("input_width", meta.get("k"))
        if type(input_width) is not int or input_width < 1:
            raise ModelingError("xllm_grouped_matmul_requires_declared_K")
    else:
        raise ModelingError("xllm_grouped_matmul_requires_MNK_geometry")
    groups = tuple(meta.get("matrix_group_rows") or ())
    if groups and (any(type(value) is not int or value < 0 for value in groups)
                   or sum(groups) != rows):
        raise ModelingError("xllm_grouped_matmul_group_rows_disagree_with_M")
    if rows < 1 or output_width < 2 or input_width < 1 or output_width % 2:
        raise ModelingError("xllm_grouped_matmul_requires_positive_even_gate_up_width")
    return rows, output_width, input_width

def _resolve_op_name(op: OpSpec, explicit_op_name: str | None) -> str:
    if explicit_op_name is not None:
        clean = str(explicit_op_name).strip()
        if clean not in XLLM_CUSTOM_OPS:
            raise ModelingError(f"xllm_custom_op_unknown:{clean}")
        return clean
    # Exact vendor semantic name only.  A shared family such as
    # ``elementwise`` or ``grouped_gemm`` is not a declaration that this
    # vendor operator runs; it must be named by the caller.
    for candidate in (op.recipe, op.family):
        if candidate in _EXACT_SEMANTIC_ALIASES:
            return _EXACT_SEMANTIC_ALIASES[candidate]
    raise ModelingError(f"xllm_custom_op_requires_explicit_name:{op.family}:{op.recipe}")

def _check_family(op: OpSpec, spec: dict[str, Any], resolved: str) -> None:
    if op.family in spec["families"] or op.recipe in spec["families"]:
        return
    raise ModelingError(f"xllm_custom_op_family_mismatch:{resolved}:{op.family}")

def _fused_declared(op: OpSpec, meta: dict[str, Any], resolved: str) -> bool:
    scopes = meta.get("fused_origin_scopes") or ()
    return op.recipe == resolved or (
        isinstance(scopes, (tuple, list)) and len(scopes) >= 3)

def _fused_quant_contract(op: OpSpec, meta: dict[str, Any], dtype: str):
    """Resolve the source INT8 grouped contract or the legal FP8 owner."""
    declared = str(
        meta.get("activation_format") or meta.get("input_format")
        or meta.get("x_format") or dtype).lower()
    output_format = str(meta.get("output_format", "int8")).lower()
    if (dtype in {"fp8_e4m3", "fp8_e5m2"} or declared in {"fp8", "fp8_e4m3", "fp8_e5m2"}
            or output_format in {"fp8", "fp8_e4m3", "fp8_e5m2"}
            or meta.get("activation_scale_group_size") == 32
            or meta.get("scale_group_size") == 32):
        raise ModelingError(
            "xllm_fused_grouped_matmul_fp8_requires_owner_recipe:"
            + FP8_GROUPED_OWNER_RECIPE)
    if dtype != "int8" or declared != "int8":
        raise ModelingError(
            f"xllm_fused_grouped_matmul_requires_int8_activation:{dtype}:{declared}")
    weight_format = str(meta.get("weight_format") or meta.get("weight_dtype", "int8")).lower()
    if weight_format != "int8":
        raise ModelingError(
            f"xllm_fused_grouped_matmul_requires_int8_weight:{weight_format}")
    if output_format != "int8":
        raise ModelingError(
            f"xllm_fused_grouped_matmul_requires_int8_output:{output_format}")
    weight_scale_format = str(
        meta.get("weight_scale_format") or meta.get("weight_scale_dtype", "fp32")).lower()
    if weight_scale_format not in {"fp32", "float32", "bf16", "bfloat16", "fp16", "float16"}:
        raise ModelingError(
            "xllm_fused_grouped_matmul_unsupported_weight_scale:"
            + weight_scale_format)
    x_scale_format = str(
        meta.get("x_scale_format") or meta.get("activation_scale_format", "fp32")).lower()
    if x_scale_format not in {"fp32", "float32"}:
        raise ModelingError(
            "xllm_fused_grouped_matmul_requires_fp32_activation_scale:"
            + x_scale_format)
    y_scale_format = str(
        meta.get("output_scale_format") or meta.get("y_scale_format", "fp32")).lower()
    if y_scale_format not in {"fp32", "float32"}:
        raise ModelingError(
            "xllm_fused_grouped_matmul_requires_fp32_output_scale:"
            + y_scale_format)
    weight_scale_bytes = dtype_bytes(weight_scale_format)
    return output_format, weight_scale_format, weight_scale_bytes

def _fused_config(op: OpSpec, meta: dict[str, Any]) -> dict[str, Any]:
    rows, gate_up_width, input_width = _matmul_dims(op, meta)
    groups = tuple(meta.get("matrix_group_rows") or ())
    if not groups:
        raise ModelingError("xllm_fused_grouped_matmul_requires_matrix_group_rows")
    if any(type(value) is not int or value < 0 for value in groups) or sum(groups) != rows:
        raise ModelingError("xllm_fused_grouped_matmul_group_rows_disagree_with_M")
    output_width = gate_up_width // 2
    output_format, weight_scale_format, weight_scale_bytes = _fused_quant_contract(
        op, meta, str(op.dtype))
    active_groups = sum(1 for value in groups if value > 0)
    activation_bytes = rows * input_width
    weight_bytes = active_groups * gate_up_width * input_width
    weight_scale_bytes_total = active_groups * gate_up_width * weight_scale_bytes
    x_scale_bytes = rows * 4
    group_list_bytes = len(groups) * 8
    output_bytes = rows * output_width
    output_scale_bytes = rows * 4
    workspace_bytes = rows * gate_up_width * 4
    limit_rows = meta.get("workspace_limit_rows")
    if limit_rows is None:
        limit_rows = (64 * 1024 * 1024 // 2 // 4) // gate_up_width
    if type(limit_rows) is not int or limit_rows < 1:
        raise ModelingError("xllm_fused_grouped_invalid_workspace_limit_rows")
    split_workspace = rows > 2 * limit_rows
    workspace_live_bytes = min(rows, 2 * limit_rows) * gate_up_width * 4
    read_bytes = (activation_bytes + weight_bytes + weight_scale_bytes_total
                  + x_scale_bytes + group_list_bytes)
    write_bytes = output_bytes + output_scale_bytes
    return {
        "rows": rows, "gate_up_width": gate_up_width,
        "output_width": output_width, "k": input_width,
        "group_rows": groups, "group_count": len(groups),
        "active_groups": active_groups,
        "matrix_flops": 2 * rows * gate_up_width * input_width,
        "activation_bytes": activation_bytes,
        "weight_bytes": weight_bytes,
        "weight_scale_bytes": weight_scale_bytes_total,
        "weight_scale_format": weight_scale_format,
        "x_scale_bytes": x_scale_bytes,
        "group_list_bytes": group_list_bytes,
        "output_bytes": output_bytes,
        "output_scale_bytes": output_scale_bytes,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "workspace_bytes": workspace_bytes,
        "workspace_live_bytes": workspace_live_bytes,
        "workspace_limit_rows": limit_rows,
        "workspace_pool": str(meta.get("workspace_pool", "gm")),
        "workspace_grant": meta.get("workspace_grant"),
        "split_workspace": split_workspace,
        "output_format": output_format,
        "quantization": "int8_per_channel_float_scale",
    }

def _fused_epilogue_work(config: dict[str, Any], geometry) -> tuple[Work, dict[str, Any]]:
    """Reuse the arch35 dequant/SwiGLU/quant capacity model without its HBM."""
    aivs = int(geometry.resource_units.get("vector", geometry.compute_units))
    ub = int(geometry.buffers.get("ub", 0))
    if "reg_alu_lanes" in geometry.resource_units and ub > 0 and aivs > 0:
        from .grouped_dequant_arch35 import (
            Config, assignments, capacity, inventories)
        try:
            cfg = Config(tuple(config["group_rows"]), config["output_width"],
                         config["rows"], aivs, ub)
        except (ModelingError, ValueError) as exc:
            raise ModelingError("xllm_fused_epilogue_geometry_unsupported") from exc
        work = Work()
        schedules = tuple(assignments(cfg, owner) for owner in range(cfg.blocks))
        for schedule in schedules:
            for _, rows in schedule:
                full, tail = divmod(rows, cfg.tile_rows)
                for count, repeat in ((cfg.tile_rows, full), (tail, bool(tail))):
                    if not repeat:
                        continue
                    for inventory in inventories(count, cfg.width):
                        work += capacity(inventory).scaled(repeat)
        return work, {"basis": "arch35_dequant_swiglu_quant_capacity",
                      "aiv": cfg.blocks, "tile_rows": cfg.tile_rows}
    # A2/A3 profiles have no A5 register-lane service.  The source epilogue is
    # still one dequant, SwiGLU, reduce and quant pass per output element; keep
    # it on the common vector pipe rather than inventing register rates.
    elements = config["rows"] * config["output_width"]
    return Work(resources={"vector": float(2 * elements + 5 * elements
                                           + 4 * elements + 2 * config["rows"])}), {
        "basis": "common_vector_equivalent_without_a5_register_lanes",
        "elements": elements}

def _build_cache_a5(op: OpSpec, geometry):
    meta = dict(op.metadata)
    index_bytes = float(meta.get("cache_slot_index_read_bytes", 0))
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    ready = bool(meta.get("cache_update_ready", False))
    total = Work(resources={"hbm": read_bytes + write_bytes,
                            "mte2": index_bytes + (read_bytes - index_bytes),
                            "mte3": write_bytes})
    def program(context):
        stages = [Stage("slot_external_ready", {}),
                  Stage("slot_index_read", context.bind(Work(resources={
                      "hbm": index_bytes, "mte2": index_bytes})),
                        after=("slot_external_ready",))]
        if write_bytes > 0:
            stages.append(Stage("masked_cache_write", context.bind(Work(
                resources={"hbm": write_bytes, "mte3": write_bytes})),
                after=("slot_index_read",)))
            stages.append(Stage("cache_publish", {}, after=("masked_cache_write",)))
        return KernelProgram(tuple(stages))
    return OperatorPlan(total, program, metadata={
        "cache_slot_mapping_known": ready,
        "cache_index_read_bytes": int(index_bytes),
        "cache_write_bytes": int(write_bytes),
        "cache_conditional_write_upper_bound": meta.get(
            "cache_conditional_write_upper_bound"),
        "cache_execution": "slot_index_read_then_masked_write_and_publish",
    })


def _build_fused_grouped(op: OpSpec, geometry):
    config = dict(dict(op.metadata).get("xllm_fused_config") or {})
    if not config:
        raise ModelingError("xllm_fused_grouped_requires_prepared_config")
    rows = int(config["rows"])
    limit = int(config["workspace_limit_rows"])
    split = bool(config["split_workspace"])
    window_rows = ([min(limit, rows - start) for start in range(0, rows, limit)]
                   if split else [rows])
    read_work = Work(resources={
        "hbm": config["read_bytes"], "mte2": config["read_bytes"]})
    write_work = Work(resources={
        "hbm": config["write_bytes"], "mte3": config["write_bytes"]})
    epilogue, basis = _fused_epilogue_work(config, geometry)
    total = Work()
    windows = []
    for index, count in enumerate(window_rows):
        share = float(count) / float(rows)
        read = read_work.scaled(share)
        matrix = Work(matrix={"int8": 2.0 * count * config["gate_up_width"]
                              * config["k"]},
                      resources={"hbm": float(count * config["gate_up_width"] * 4),
                                 "mte3": float(count * config["gate_up_width"] * 4)})
        space = Work(resources={"hbm": float(count * config["gate_up_width"] * 4),
                                "mte2": float(count * config["gate_up_width"] * 4)})
        epi = epilogue.scaled(share)
        write = write_work.scaled(share)
        total += read + matrix + space + epi + write
        windows.append((index, read, matrix, space, epi, write))

    def program(context):
        pool_name = str(config.get("workspace_pool", "gm"))
        if pool_name not in context.pool_names:
            pool_name = next((name for name in context.pool_names
                              if "workspace" in name or name in {"gm", "hbm"}),
                             pool_name)
        pool = context.pool_names.get(pool_name)
        allocations = {}
        if config.get("workspace_grant") is not None:
            pool = None
        if pool is not None and split:
            size = float(min(limit, rows) * config["gate_up_width"] * 4)
            allocations = {
                slot: (context.allocate(f"xllm_ws_{slot}", pool_name, size, (0,)),)
                for slot in (0, 1)}
        elif pool is not None:
            size = float(rows * config["gate_up_width"] * 4)
            allocations = {0: (context.allocate("xllm_ws_0", pool_name, size, (0,)),)}
        stages = []
        for index, read, matrix, space, epi, write in windows:
            slot = index % 2
            alloc = allocations.get(slot, ())
            stages.extend((
                Stage(f"read_{index}", context.bind(read)),
                Stage(f"matrix_{index}", context.bind(matrix),
                      after=((f"read_{index}",)
                             + ((f"workspace_read_{index - 2}",)
                                if index >= 2 else ())),
                      acquire=alloc),
                Stage(f"workspace_read_{index}", context.bind(space),
                      after=(f"matrix_{index}",), free=alloc),
                Stage(f"dequant_swiglu_quant_{index}", context.bind(epi),
                      after=(f"workspace_read_{index}",)),
                Stage(f"write_{index}", context.bind(write),
                      after=(f"dequant_swiglu_quant_{index}",)),
            ))
        return KernelProgram(tuple(stages))

    return OperatorPlan(total, program, metadata={
        "xllm_grouped_plan": config,
        "epilogue_work_basis": basis,
        "intermediate_hbm_bytes": int(config["workspace_bytes"]),
        "workspace_live_bytes": int(config["workspace_live_bytes"]),
        "workspace_window_rows": tuple(window_rows),
        "workspace_windows": len(window_rows),
        "workspace_requires_memory_grant": True,
        "workspace_grant_reused": config.get("workspace_grant") is not None,
        "workspace_allocation_pool": str(config.get("workspace_pool", "gm")),
        "split_workspace": split,
        "source": XLLM_VENDOR + ":moe_grouped_matmul_swiglu_quant",
    })


def _mega_gdn_config(op: OpSpec, meta: dict[str, Any]) -> dict[str, Any]:
    head_dim = int(meta.get("head_dim", 128))
    if head_dim != 128:
        raise ModelingError("xllm_mega_gdn_requires_source_head_dim_128")
    qkv_shape = meta.get("qkv_shape")
    z_shape = meta.get("z_shape")
    if qkv_shape is None or z_shape is None:
        raise ModelingError("xllm_mega_gdn_requires_qkv_and_z_shapes")
    qkv_shape = tuple(int(value) for value in qkv_shape)
    z_shape = tuple(int(value) for value in z_shape)
    if len(qkv_shape) != 2 or len(z_shape) != 3:
        raise ModelingError("xllm_mega_gdn_requires_2d_qkv_and_3d_BVH_Z")
    batch, conv_dim = qkv_shape
    if z_shape[0] != batch or z_shape[2] != head_dim or z_shape[1] < 1:
        raise ModelingError("xllm_mega_gdn_z_shape_disagrees_batch_or_heads")
    v_heads = z_shape[1]
    k_heads = meta.get("num_k_heads")
    if k_heads is None:
        remaining = conv_dim // head_dim - v_heads
        if remaining < 2 or remaining % 2:
            raise ModelingError("xllm_mega_gdn_cannot_resolve_key_heads")
        k_heads = remaining // 2
    k_heads, v_heads = int(k_heads), int(v_heads)
    if (k_heads < 1 or v_heads % k_heads
            or conv_dim != (2 * k_heads + v_heads) * head_dim):
        raise ModelingError("xllm_mega_gdn_convolution_geometry_mismatch")
    lengths = tuple(int(value) for value in meta.get("sequence_tokens", (1,) * batch))
    if len(lengths) != batch or any(value < 1 for value in lengths):
        raise ModelingError("xllm_mega_gdn_requires_one_positive_length_per_batch")
    conv_state_shape = _shape_meta(meta, "conv_state_shape", (batch, 3, conv_dim))
    ssm_state_shape = _shape_meta(meta, "ssm_state_shape",
                                  (batch, v_heads, head_dim, head_dim))
    if (len(conv_state_shape) < 2 or len(ssm_state_shape) != 4
            or ssm_state_shape[1:] != (v_heads, head_dim, head_dim)):
        raise ModelingError("xllm_mega_gdn_state_geometry_mismatch")
    b_shape = _shape_meta(meta, "b_shape", (batch, v_heads))
    a_shape = _shape_meta(meta, "a_shape", (batch, v_heads))
    a_log_shape = _shape_meta(meta, "a_log_shape", (v_heads,))
    dt_bias_shape = _shape_meta(meta, "dt_bias_shape", (v_heads,))
    norm_weight_shape = _shape_meta(meta, "norm_weight_shape", (head_dim,))
    read_indices = tuple(meta.get("read_state_indices",
                                  meta.get("state_indices", (0,) * batch)))
    write_indices = tuple(meta.get("write_state_indices", read_indices))
    if (len(read_indices) != batch or len(write_indices) != batch
            or any(type(index) is not int or index < -1
                   for index in (*read_indices, *write_indices))):
        raise ModelingError("xllm_mega_gdn_state_indices_invalid")
    if prod(b_shape) != batch * v_heads or prod(a_shape) != batch * v_heads:
        raise ModelingError("xllm_mega_gdn_gate_shapes_disagree_with_batch_heads")
    if prod(a_log_shape) != v_heads or prod(dt_bias_shape) != v_heads:
        raise ModelingError("xllm_mega_gdn_a_log_dt_bias_shapes_disagree_with_heads")
    if prod(norm_weight_shape) != head_dim:
        raise ModelingError("xllm_mega_gdn_norm_weight_shape_mismatch")
    state_layout = str(meta.get("state_layout",
                                "V,K" if meta.get("fla_ssm_state_layout", True)
                                else "K,V"))
    read_versions = meta.get("read_versions")
    write_versions = meta.get("write_versions")
    for name, values in (("read_versions", read_versions),
                         ("write_versions", write_versions)):
        if values is not None and (len(values) != batch
                                   or any(type(value) is not int for value in values)):
            raise ModelingError(f"xllm_mega_gdn_{name}_invalid")
    return {
        "batch": batch, "qkv_shape": qkv_shape, "z_shape": z_shape,
        "conv_dim": conv_dim, "k_heads": k_heads, "v_heads": v_heads,
        "head_dim": head_dim, "lengths": lengths,
        "conv_state_shape": conv_state_shape,
        "ssm_state_shape": ssm_state_shape,
        "conv_weight_shape": _shape_meta(meta, "conv_weight_shape",
                                         (4, conv_dim)),
        "b_shape": b_shape, "a_shape": a_shape,
        "a_log_shape": a_log_shape, "dt_bias_shape": dt_bias_shape,
        "norm_weight_shape": norm_weight_shape,
        "read_state_indices": read_indices,
        "write_state_indices": write_indices,
        "read_versions": tuple(read_versions) if read_versions is not None else None,
        "write_versions": tuple(write_versions) if write_versions is not None else None,
        "state_layout": state_layout,
        "state_capacity": meta.get("state_capacity"),
    }


def _shape_meta(meta, key, default):
    value = meta.get(key, default)
    shape = tuple(int(item) for item in value)
    if not shape or any(item < 0 for item in shape):
        raise ModelingError("xllm_mega_gdn_invalid_shape:" + key)
    return shape


def _build_mega_gdn_decode(op: OpSpec, geometry):
    config = dict(dict(op.metadata).get("xllm_mega_gdn_config") or {})
    if not config:
        raise ModelingError("xllm_mega_gdn_requires_prepared_config")
    batch, dim = config["batch"], config["head_dim"]
    k_heads, v_heads, conv_dim = (config["k_heads"], config["v_heads"],
                                  config["conv_dim"])
    bf16, f32 = dtype_bytes("bf16"), dtype_bytes("fp32")
    z_bytes = batch * v_heads * dim * bf16
    out_bytes = z_bytes
    conv_read = (batch * conv_dim * bf16 + 4 * conv_dim * bf16
                 + 3 * batch * conv_dim * bf16)
    conv_write = batch * conv_dim * bf16 + batch * 3 * conv_dim * bf16
    conv_work = Work(resources={"hbm": conv_read + conv_write,
                                "mte2": conv_read, "mte3": conv_write,
                                "vector": float(batch * conv_dim * 9)})
    gate_read = (batch * conv_dim * bf16 + dim * bf16
                 + prod(config["b_shape"]) * bf16
                 + prod(config["a_shape"]) * bf16
                 + prod(config["a_log_shape"]) * f32
                 + prod(config["dt_bias_shape"]) * f32
                 + 2 * batch * 4)
    qk_norm_work = Work(resources={
        "hbm": gate_read, "mte2": gate_read,
        "vector": float(batch * (2 * k_heads + v_heads) * dim * 5
                        + batch * (v_heads + v_heads) * 2)})
    from .gdn_recurrent import build as build_recurrent
    recurrent_op = OpSpec(
        op_id=op.op_id + ".recurrent",
        family="recurrent_vector", phase="decode", stage=op.stage,
        layer=op.layer, requests=op.requests,
        semantic_shape=(batch, v_heads, dim, dim),
        local_shape=(batch, v_heads, dim, dim),
        padded_shape=(batch, v_heads, dim, dim),
        launch_shape=(batch, v_heads, dim, dim),
        dtype="bf16", recipe="recurrent_scan",
        metadata=tuple({
            "sequence_tokens": config["lengths"],
            "heads": v_heads, "key_heads": k_heads,
            "dk": dim, "dv": dim, "state_dtype": "fp32",
            "state_read_indices": config["read_state_indices"],
            "state_write_indices": config["write_state_indices"],
            "state_read_versions": config["read_versions"],
            "state_write_versions": config["write_versions"],
            "state_layout": config["state_layout"],
            "state_capacity": config["state_capacity"],
        }.items()),
    )
    recurrent = build_recurrent(recurrent_op, geometry)
    attention_write = batch * v_heads * dim * bf16
    # The fused source writes the final readout only once.  F-STATE's own
    # program owns the state/readout write; the post-readout vector pass below
    # consumes Z/norm on chip and does not add a second GM write.
    readout_work = Work(resources={
        "hbm": z_bytes + dim * bf16,
        "mte2": z_bytes + dim * bf16,
        "vector": float(batch * v_heads * dim * 7)})
    total = conv_work + qk_norm_work + recurrent.work + readout_work

    def program(context):
        return KernelProgram((
            Stage("conv", context.bind(conv_work)),
            Stage("qk_norm_gate", context.bind(qk_norm_work), after=("conv",)),
            Stage("recurrent_state", after=("qk_norm_gate",),
                  program=recurrent.program(context)),
            Stage("readout_rmsnorm_zgate", context.bind(readout_work),
                  after=("recurrent_state",)),
        ))

    effects = []
    for index, (read_slot, write_slot) in enumerate(zip(
            config["read_state_indices"], config["write_state_indices"])):
        read_version = (config["read_versions"][index]
                        if config["read_versions"] is not None else None)
        write_version = (config["write_versions"][index]
                         if config["write_versions"] is not None else None)
        for target, slot, version, action in (
                ("conv_state", read_slot, read_version, "read"),
                ("conv_state", write_slot, write_version, "write"),
                ("ssm_state", read_slot, read_version, "read"),
                ("ssm_state", write_slot, write_version, "write")):
            if slot is None or slot < 0:
                continue
            span = (3 * conv_dim * bf16 if target == "conv_state"
                    else v_heads * dim * dim * 4)
            effects.append({
                "target": target, "slot": int(slot), "version": version,
                "action": action, "occurrence": index,
                "owner": op.op_id, "storage": target,
                "request": (op.requests[0] if op.requests else None),
                "stage": op.stage,
                "span_bytes": span,
                "span": (int(slot) * span, (int(slot) + 1) * span),
                "ready_event": f"{op.op_id}:{target}:{slot}:ready",
                "commit_event": (f"{op.op_id}:{target}:{slot}:commit"
                                 if action == "write" else None),
            })
    return OperatorPlan(total, program, metadata={
        "xllm_mega_gdn_config": config,
        "state_mechanism": "gdn_recurrent.build_program",
        "state_program": "gdn_recurrent.program",
        "gdn_state_effects": tuple(effects),
        "state_versions": {"read": config["read_versions"],
                           "write": config["write_versions"]},
        "state_traffic_bytes": {
            "conv_read": 3 * batch * conv_dim * bf16,
            "conv_write": 3 * batch * conv_dim * bf16,
            "ssm_read": batch * v_heads * dim * dim * 4,
            "ssm_write": batch * v_heads * dim * dim * 4,
        },
        "readout_order": ("conv", "qk_norm_gate", "recurrent_state",
                          "readout_rmsnorm_zgate"),
        "two_states": ("conv_state_out", "ssm_state_out"),
        "state_indices": {
            "read": config["read_state_indices"],
            "write": config["write_state_indices"],
            "layout": config["state_layout"],
            "read_versions": config["read_versions"],
            "write_versions": config["write_versions"],
        },
        "readout_onchip": True,
        "final_output_write_bytes": attention_write,
    })


def fused_region_contract(members, outputs, selection, options=None):
    """Consume a declared three-member gate/up+SwiGLU+quant region.

    T11 owns the region replacement.  This function supplies the one typed
    contract that both the region path and direct complete-call lowering use;
    it does not rewrite a graph or claim ownership of member occurrences.
    """
    members = tuple(members)
    options = dict(options or {})
    if set(options) - {"op_name", "soc_family"}:
        raise ModelingError("xllm_fused_region_has_unexpected_options")
    matrices = [op for op in members
                if op.family in {"grouped_gemm"} or op.recipe in {
                    "grouped_gate_up", "balanced_grouped_gate_up"}]
    activations = [op for op in members if op.family == "swiglu"]
    quantizers = [op for op in members if op.family == "quantize"]
    if (len(members) != 3 or len(matrices) != 1 or len(activations) != 1
            or len(quantizers) != 1):
        raise ModelingError("xllm_fused_region_requires_gate_up_swiglu_quant")
    matrix, activation, quant = matrices[0], activations[0], quantizers[0]
    if (activation.deps != (matrix.op_id,) or quant.deps != (activation.op_id,)
            or set(outputs) != {quant.op_id}
            or any(op.plan_required or "implementation_binding" in dict(op.metadata)
                   for op in members)):
        raise ModelingError(
            "xllm_fused_region_requires_ordinary_exclusive_member_chain")
    meta = dict(matrix.metadata)
    config = _fused_config(matrix, meta)
    return {
        "config": config,
        "members": tuple(op.op_id for op in members),
        "external_outputs": tuple(outputs),
        "implementation": selection.name,
        "native_kind": XLLM_CUSTOM_OPS[FUSED_GROUPED_RECIPE]["native_kind"],
        "source": {
            "repository": "xllm",
            "op": FUSED_GROUPED_RECIPE,
            "path": XLLM_CUSTOM_OPS[FUSED_GROUPED_RECIPE]["source_path"],
            "scope": "declared_three_member_region; replacement_owned_by_T11",
        },
    }

def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    return_lse: bool = True,
    beam_width: int | None = None,
    kv_page_size: int = 128,
) -> OpSpec:
    """Lower one complete semantic occurrence into an xLLM custom operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = XLLM_CUSTOM_OPS[resolved]
    _check_family(op, spec, resolved)
    if soc_family is not None and soc_family not in spec["supported_socs"]:
        raise ModelingError(f"xllm_custom_op_soc_unsupported:{resolved}:{soc_family}")
    if type(kv_page_size) is not int or kv_page_size <= 0:
        raise ModelingError(f"xllm_custom_op_invalid_page_size:{kv_page_size}")

    meta = dict(op.metadata)
    dtype = _declared_dtype(op, meta)
    if resolved == FUSED_GROUPED_RECIPE and dtype != "int8":
        _fused_quant_contract(op, meta, dtype)
    if dtype not in spec["input_dtypes"]:
        raise ModelingError(f"xllm_custom_op_input_dtype_unsupported:{resolved}:{dtype}")
    result_dtype = dtype
    local_shape = tuple(op.local_shape)
    if type(return_lse) is not bool:
        raise ModelingError("xllm_sparse_flash_attention_requires_boolean_return_lse")

    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    vec_ops = float(op.vector_ops)
    returns = tuple(op.returns)

    if resolved in {"smla_va", "sparse_flash_attention_lse"} and op.family == "attention":
        recipe = str(op.recipe).lower()
        if not any(token in recipe for token in ("latent", "mla", "sparse", "smla")):
            raise ModelingError(
                f"xllm_{resolved}_requires_latent_sparse_or_mla_attention:{op.recipe}")

    if resolved == "smla_va":
        shape = _shape(op)
        q_shape = tuple(meta.get("q_shape", shape))
        if len(q_shape) == 4:
            batch, seq, heads, dim = q_shape
            tokens = batch * seq
        elif len(q_shape) == 3:
            tokens, heads, dim = q_shape
            batch, seq = 1, tokens
        else:
            tokens, heads = shape[0], _attention_heads(op, meta)
            batch, seq, dim = 1, tokens, shape[-1]
        dv = int(meta.get("value_head_dim", dim))
        read_bytes = tokens * heads * (dim + dv) * 2
        for key in ("ori_kv_shape", "cmp_kv_shape"):
            if meta.get(key) is not None:
                read_bytes += prod(int(v) for v in meta[key]) * 2
        read_bytes += int(meta.get("sparse_index_count", 0)) * 4
        write_bytes = tokens * heads * dv * 2
        meta["xllm_attention_geometry"] = {
            "query_tokens": tokens, "heads": heads, "batch": batch,
            "sequence": seq, "value_head_dim": dv,
            "mode": "variable_absorption"}
        meta["xllm_ports"] = {
            "inputs": ("q", "ori_kv", "cmp_kv", "ori_sparse_indices",
                       "cmp_sparse_indices", "block_tables", "seqlens"),
            "outputs": ("attn_out", "softmax_lse"),
        }
        if "attn_out" not in returns:
            returns = (*returns, "attn_out")
        if bool(meta.get("return_softmax_lse", False)):
            lse = tokens * heads * 4
            write_bytes += lse
            read_bytes = read_bytes
            if "softmax_lse" not in returns:
                returns = (*returns, "softmax_lse")
    elif resolved == "sparse_flash_attention_lse":
        shape = _shape(op)
        heads = _attention_heads(op, meta)
        q_shape = tuple(meta.get("q_shape", shape))
        if len(q_shape) == 4:
            batch, seq = q_shape[0], q_shape[1]
            tokens = batch * seq
        elif len(q_shape) == 3:
            tokens = q_shape[0]
        else:
            tokens = shape[0]
        stats = float(2 * tokens * heads * 4)
        write_bytes += stats
        read_bytes = float(tokens * heads * shape[-1] * 2
                           + int(meta.get("sparse_index_count", 0)) * 4)
        for key in ("key_shape", "value_shape"):
            if meta.get(key) is not None:
                read_bytes += prod(int(v) for v in meta[key]) * 2
        meta["xllm_ports"] = {
            "inputs": ("query", "key", "value", "sparse_indices",
                       "block_table", "actual_seq_lengths"),
            "outputs": ("attention_out", "softmax_max", "softmax_sum",
                        "softmax_lse"),
        }
        for item in ("softmax_max", "softmax_sum"):
            if item not in returns:
                returns = (*returns, item)
        if return_lse and "softmax_lse" not in returns:
            returns = (*returns, "softmax_lse")
        meta["xllm_attention_geometry"] = {
            "query_tokens": tokens, "heads": heads, "return_lse": return_lse,
            "softmax_stats_bytes": int(stats),
            "lse_representation": "softmax_max_plus_sum",
        }
    elif resolved == "moe_grouped_matmul_swiglu_quant":
        if not _fused_declared(op, meta, resolved):
            raise ModelingError(
                "xllm_fused_grouped_matmul_requires_declared_fusion_region")
        config = _fused_config(op, meta)
        rows, output_width = config["rows"], config["output_width"]
        read_bytes = float(config["read_bytes"] + config["workspace_bytes"])
        write_bytes = float(config["write_bytes"] + config["workspace_bytes"])
        vec_ops = 0.0
        local_shape = (rows, output_width)
        meta["xllm_swiglu_quant_fused"] = True
        meta["xllm_fused_config"] = config
        meta["xllm_grouped_geometry"] = config
        # The complete fused occurrence owns its group ledger in the config;
        # the generic grouped-matrix row adapter must not reinterpret the
        # now-output-shaped occurrence as an MxNxK matrix.
        meta.pop("matrix_group_rows", None)
        meta["xllm_fused_output_shape"] = (rows, output_width)
        meta["xllm_fused_scale_shape"] = (rows,)
        returns = ("y", "y_scale")
        result_dtype = "int8"
    elif resolved == "mega_gdn_decode":
        if dtype != "bf16":
            raise ModelingError(
                f"xllm_mega_gdn_requires_bf16_inputs:{dtype}")
        if (op.family in {"recurrent_vector", "recurrent_scan"}
                and not _fused_declared(op, meta, resolved)):
            raise ModelingError(
                "xllm_mega_gdn_decode_requires_declared_fusion_region")
        config = _mega_gdn_config(op, meta)
        meta["xllm_mega_gdn_config"] = config
        meta["xllm_gdn_geometry"] = {
            "batch": config["batch"], "k_heads": config["k_heads"],
            "v_heads": config["v_heads"], "head_dim": config["head_dim"],
            "conv_dim": config["conv_dim"],
            "state_dtype": str(meta.get("state_dtype", "fp32")),
            "conv_state_shape": config["conv_state_shape"],
            "ssm_state_shape": config["ssm_state_shape"],
        }
        local_shape = config["z_shape"]
        returns = ("convOut", "convStateOut", "ssmStateOut", "out")
        conv_dim = config["conv_dim"]
        batch, dim = config["batch"], config["head_dim"]
        bf16, f32 = dtype_bytes("bf16"), dtype_bytes("fp32")
        conv_read = (batch * conv_dim * bf16 + 4 * conv_dim * bf16
                     + 3 * batch * conv_dim * bf16)
        conv_write = batch * conv_dim * bf16 + batch * 3 * conv_dim * bf16
        norm_read = batch * config["v_heads"] * dim * bf16 + dim * bf16
        read_bytes = float(conv_read + norm_read)
        write_bytes = float(conv_write)
        vec_ops = float(batch * conv_dim * 9
                        + batch * (2 * config["k_heads"] + config["v_heads"]) * dim * 5
                        + batch * config["v_heads"] * dim * 5)
        meta["xllm_mega_gdn_declared_work"] = {
            "conv_read_bytes": int(conv_read),
            "conv_write_bytes": int(conv_write),
            "norm_read_bytes": int(norm_read),
            "basis": "source_conv_norm_stage_work_before_recurrent_plan",
        }
    elif resolved == "onerec_final_beam_select":
        rows, width = _tokens_width(op)
        k_beam = int(beam_width if beam_width is not None else
                     meta.get("active_beam", meta.get("beam_width", 0)))
        if k_beam <= 0:
            raise ModelingError(f"xllm_onerec_requires_positive_beam_width:{k_beam}")
        # Source outputs: three result-width scalars (INT32 ids/index, FP32
        # log-probs) plus an INT32 sequence tensor.  The candidate top-k is an
        # input width, not the final result width.
        result_width = int(meta.get("result_width",
                                    meta.get("num_return_sequences", k_beam)))
        topk = int(meta.get("topk", meta.get("candidate_topk", width)))
        batch = int(meta.get("batch", meta.get("request_batch",
                                               max(1, len(op.requests)))))
        current_step = meta.get("current_step")
        sequence_capacity = int(meta.get("sequence_capacity", 0))
        if current_step is not None:
            current_step = int(current_step)
            if sequence_capacity < 1:
                sequence_capacity = current_step + 1
            if current_step < 0 or sequence_capacity < current_step + 1:
                raise ModelingError("xllm_onerec_sequence_capacity_too_small")
            sequence_shape = (batch, result_width, current_step + 1)
        else:
            sequence_shape = meta.get("sequence_shape")
        sequence_bytes = (prod(int(v) for v in sequence_shape) * 4
                          if sequence_shape is not None else 0)
        scalar_bytes = float(3 * result_width * 4)
        beam_write = scalar_bytes + sequence_bytes
        beam_read = float(rows * topk * 4 + 2 * rows * k_beam * 4
                          + rows * 4 + sequence_bytes)
        declared_write = write_bytes
        # OneRec owns its public outputs; inherited shape bytes are not an
        # additional old-output read/write.
        write_bytes = beam_write
        read_bytes = beam_read
        vec_ops = max(vec_ops, float(rows * k_beam * 4))
        for item in ("out_token_ids", "out_token_index", "out_log_probs",
                     "out_sequence"):
            if item not in returns:
                returns = (*returns, item)
        meta["beam_width"] = k_beam
        meta["xllm_ports"] = {
            "inputs": ("log_probs", "top_tokens", "top_probs", "sequence"),
            "outputs": ("out_token_ids", "out_token_index", "out_log_probs",
                        "out_sequence"),
        }
        meta["xllm_beam_selection"] = {
            "rows": rows, "input_width": width, "beam_width": k_beam,
            "selected_write_bytes": int(beam_write),
            "selected_read_bytes": int(beam_read),
            "active_beam": k_beam, "result_width": result_width, "topk": topk,
            "batch": batch, "sequence_capacity": sequence_capacity,
            "current_step": current_step,
            "sequence_shape": sequence_shape,
            "declared_semantic_write_bytes": int(declared_write)}
    elif resolved == "reshape_and_cache_a5":
        tokens = _shape(op)[0]
        cache_view = dict(meta.get("cache_tensor_view") or {})
        cache_dtype = str(
            meta.get("kv_dtype") or meta.get("cache_dtype")
            or cache_view.get("key_dtype") or cache_view.get("value_dtype") or dtype)
        if cache_dtype not in spec["input_dtypes"]:
            raise ModelingError(
                f"xllm_cache_write_dtype_unsupported:{cache_dtype}")
        row_bytes, row_basis = _cache_row_bytes(op, meta, cache_dtype)
        slot_mapping = meta.get("slot_mapping")
        total_slots = meta.get("cache_capacity_slots", meta.get("total_slots"))
        capacity_inferred = False
        mapping_known = slot_mapping is not None
        if slot_mapping is None:
            valid_tokens = None
            invalid_tokens = None
            if type(total_slots) is not int or total_slots < 1:
                total_slots = tokens
            # The external INT32 index tensor is real even when its values are
            # not yet bound.  It is read now; the K/V write remains conditional.
            slot_index_bytes = tokens * 4
        else:
            if (isinstance(slot_mapping, (str, bytes))
                    or not isinstance(slot_mapping, (tuple, list))
                    or len(slot_mapping) != tokens
                    or any(type(slot) is not int for slot in slot_mapping)):
                raise ModelingError("xllm_cache_slot_mapping_requires_one_integer_per_token")
            if type(total_slots) is not int or total_slots < 1:
                total_slots = max(1, max(slot_mapping, default=-1) + 1)
                capacity_inferred = True
            # Source skips slot < 0 and slot >= total_slots; it does not reject
            # the call.  Those rows are not written.
            valid_tokens = sum(1 for slot in slot_mapping
                               if 0 <= slot < total_slots)
            invalid_tokens = tokens - valid_tokens
            slot_index_bytes = tokens * 4
        # Page capacity is the actual allocated slot count, not a page-aligned
        # guess; written K/V rows remain the valid token count.
        declared_write = write_bytes
        page_capacity_bytes = float(total_slots * row_bytes)
        if valid_tokens is None:
            # Binding happens later.  No ordinary all-valid K/V write is issued;
            # only the index read and readiness are known now.
            write_bytes = 0.0
            read_bytes = float(slot_index_bytes)
            write_basis = "deferred_slot_binding"
        else:
            write_bytes = float(valid_tokens * row_bytes)
            read_bytes = float(valid_tokens * row_bytes + slot_index_bytes)
            write_basis = "valid_token_rows_not_page_capacity"
        meta["kv_page_size"] = int(kv_page_size)
        # Page mapping span is recorded, but it does not size the issued write.
        meta["aligned_page_rows"] = int(align_up(tokens, kv_page_size))
        meta["cache_write_dtype"] = cache_dtype
        meta["cache_row_bytes"] = int(row_bytes)
        meta["cache_row_basis"] = row_basis
        meta["cache_capacity_slots"] = int(total_slots)
        meta["cache_page_capacity_bytes"] = page_capacity_bytes
        meta["cache_valid_write_rows"] = (None if valid_tokens is None
                                          else int(valid_tokens))
        meta["cache_invalid_slot_rows"] = (None if invalid_tokens is None
                                           else int(invalid_tokens))
        meta["cache_slot_mapping_known"] = mapping_known
        meta["cache_capacity_inferred_from_slots"] = capacity_inferred
        meta["cache_conditional_write_upper_bound"] = (
            float(tokens * row_bytes) if valid_tokens is None else None)
        meta["cache_update_ready"] = bool(valid_tokens is not None)
        meta["cache_slot_index_read_bytes"] = int(slot_index_bytes)
        meta["declared_semantic_write_bytes"] = int(declared_write)
        meta["cache_work_basis"] = write_basis
        if "xtensor_kv_page" not in returns:
            returns = (*returns, "xtensor_kv_page")
        for item in ("key_cache", "value_cache"):
            if item not in returns:
                returns = (*returns, item)

    meta.update({
        "xllm_custom_op": resolved,
        "native_vendor": XLLM_VENDOR,
        "native_kind": spec["native_kind"],
        "native_source": {
            "repository": "xllm",
            "vendor": XLLM_VENDOR,
            "path": spec["source_path"],
            "scope": "independent_source_semantics; custom_xllm_math_lowering",
        },
        "native_implementation": {
            "framework": "xllm",
            "vendor": XLLM_VENDOR,
            "native_kinds": (spec["native_kind"],),
            "op_name": resolved,
            "selection": "explicit_caller_implementation",
            "soc_family": soc_family,
            "atomic_occurrence": bool(spec["atomic_occurrence"]),
        },
    })

    if resolved == FUSED_GROUPED_RECIPE:
        matrix_flops = (("int8", float(dict(meta["xllm_fused_config"])["matrix_flops"])),)
        flops = 0.0
        semantic_shape = local_shape
        padded_shape = local_shape
        launch_shape = local_shape
    elif resolved == "mega_gdn_decode":
        matrix_flops = ()
        flops = 0.0
        semantic_shape = local_shape
        padded_shape = local_shape
        launch_shape = local_shape
    else:
        matrix_flops = op.matrix_flops
        flops = op.flops
        semantic_shape = op.semantic_shape
        padded_shape = op.padded_shape
        launch_shape = op.launch_shape
    return replace(
        op,
        recipe=resolved,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        dtype=result_dtype,
        flops=flops,
        matrix_flops=matrix_flops,
        semantic_shape=semantic_shape,
        local_shape=local_shape,
        padded_shape=padded_shape,
        launch_shape=launch_shape,
        metadata=tuple(meta.items()),
    )

__all__ = ["CAPABILITIES", "FUSED_GROUPED_RECIPE", "XLLM_CUSTOM_OPS",
           "XLLM_VENDOR", "capabilities", "fused_region_contract", "lower_op",
           "recipes"]
