"""Compile one selected native program into an immutable ProgramCatalog.

[Role] Read the already-normalized ABI/leaf prediction and emit the physical
  launch catalog: declared prologue/drain, per-owner load/compute/store at the
  active owners' provisioned rates, Cube waves, Fixpipe drain and paged-KV
  walk.  Also binds that catalog to one fresh caller occurrence.
[Invariants] The catalog is pure and occurrence-free; one result rebinds to
  every caller occurrence.  Declared bytes/ops are preserved exactly; only the
  launch physics around them is added.  No Engine, Event or allocation
  identity enters the catalog.
[Timeouts & Locks] Pure bounded arithmetic; no waits or locks.
[Side-Effect Policy] Returns immutable records; no I/O or global mutation.
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
_NATIVE_KERNEL_CLASS = {
    "aclnnAddRmsNorm": "fused_row_norm",
    "torch_npu.npu_rms_norm": "row_norm",
    "aclnnAdd": "elementwise",
    "aclnnMul": "elementwise",
    "aclnnMuls": "elementwise",
    "aclnnGelu": "elementwise",
    "aclnnSilu": "elementwise",
    "aclnnSwish": "elementwise",
    "aclnnCast": "elementwise",
    "Add": "elementwise",
    "Mul": "elementwise",
    "Muls": "elementwise",
    "Gelu": "elementwise",
    "Swish": "elementwise",
    "Silu": "elementwise",
    "Cast": "elementwise",
    "torch.add": "elementwise",
    "torch.ops.vllm.muls_add": "elementwise",
    "aclnnDynamicMxQuant": "quantize",
    "aclnnDynamicQuant": "quantize",
    "aclnnDynamicQuantV2": "quantize",
    "DynamicMxQuant": "quantize",
    "DynamicQuant": "quantize",
    "DynamicQuantV2": "quantize",
    "torch_npu.npu_swiglu": "vector",
    "aclnnSwiGlu": "vector",
    "SwiGlu": "vector",
}

#: Declared elementwise stream counts per fused/standalone row kernel.  The
#: fused call reads ``x1, x2`` and writes ``y, residual_out`` (four whole-width
#: streams); the standalone norm reads one and writes one; a plain add reads
#: two and writes one.
_ROW_STREAMS = {
    "fused_row_norm": (2, 2),
    "row_norm": (1, 1),
    "elementwise": (2, 1),
    "quantize": (1, 1),
    "vector": (1, 1),
}



def _declared_pipe_totals(result: NativeAbiPrediction) -> dict[str, float]:
    """Sum the leaf's declared physical work per pipe, exactly as declared."""
    totals: dict[str, float] = {}
    for stage in (result.prediction or {}).get("stage_work") or ():
        entries = stage.items() if hasattr(stage, "items") else stage
        for pipe, value in entries:
            if isinstance(value, bool):
                continue
            try:
                amount = float(value)
            except (TypeError, ValueError):
                continue
            if amount > 0:
                totals[str(pipe)] = totals.get(str(pipe), 0.0) + amount
    return totals


def _kernel_class(result: NativeAbiPrediction) -> str:
    program = result.execution_program
    kind = str(getattr(program, "kind", "") or "")
    if kind == "matrix":
        descriptor = dict((getattr(getattr(program, "plan", None),
                                   "metadata", {}) or {}).get(
                                       "operator_plan") or {})
        digest = str(descriptor.get("variant") or "").lower()
        quantized = any(token in digest for token in
                        ("int8", "quant", "w8a8", "mxfp", "fp8"))
        return "matrix_quant" if quantized else "matrix_bf16"
    if kind == "paged_fia":
        return "paged_fia"
    entry = str((result.spec or {}).get("entry") or "")
    return _NATIVE_KERNEL_CLASS.get(entry, "vector")


def _row_orientation(result: NativeAbiPrediction, kernel_class: str
                     ) -> dict[str, Any] | None:
    """Declared row geometry and streams of a row elementwise invocation."""
    tensors = (result.invocation or {}).get("tensors") or {}
    shape: list[int] | None = None
    dtype = "bf16"
    for name, row in tensors.items():
        if not isinstance(row, Mapping) or row.get("present") is False:
            continue
        if str(row.get("role") or "") == "output":
            continue
        candidate = [int(item) for item in (row.get("shape") or ())]
        if len(candidate) >= 2 and min(candidate) > 0:
            shape = candidate
            dtype = str(row.get("dtype") or "bf16")
            break
    if shape is None:
        for row in tensors.values():
            if not isinstance(row, Mapping) or row.get("present") is False:
                continue
            if str(row.get("role") or "") != "output":
                continue
            candidate = [int(item) for item in (row.get("shape") or ())]
            if len(candidate) >= 2 and min(candidate) > 0:
                shape = candidate
                dtype = str(row.get("dtype") or "bf16")
                break
    if shape is None:
        for row in tensors.values():
            if not isinstance(row, Mapping) or row.get("present") is False:
                continue
            if str(row.get("role") or "") == "output":
                continue
            candidate = [int(item) for item in (row.get("shape") or ())]
            if len(candidate) == 1 and candidate[0] > 0:
                shape = [1, candidate[0]]
                dtype = str(row.get("dtype") or "bf16")
                break
    if shape is None:
        return None
    rows = int(shape[0])
    width = 1
    for item in shape[1:]:
        width *= int(item)
    program_meta = dict((result.prediction or {}).get("program") or {})
    if kernel_class != "fused_row_norm":
        prog_elems = program_meta.get("elements")
        if (isinstance(prog_elems, (int, float)) and int(prog_elems) > 0
                and (kernel_class == "row_norm" or rows == 1)):
            width = int(prog_elems)
    dtype_bytes = {"fp32": 4, "float32": 4, "int8": 1, "fp8_e4m3": 1,
                   "fp8_e5m2": 1, "fp4": 1, "mxfp4": 1}.get(dtype, 2)
    # The resident norm vector: a declared rank-1 weight/gamma tensor.  Its
    # storage bytes are the broadcast every active block preloads into UB.
    gamma_bytes = 0
    for row in tensors.values():
        if not isinstance(row, Mapping) or row.get("present") is False:
            continue
        role = str(row.get("role") or "")
        if role not in {"weight", "gamma"} and not str(
                row.get("name") or "").endswith("gamma"):
            continue
        vector = [int(item) for item in (row.get("shape") or ())]
        if len(vector) != 1 or vector[0] < 1:
            continue
        element = {"fp32": 4, "float32": 4, "int8": 1, "fp8_e4m3": 1,
                   "fp8_e5m2": 1, "fp4": 1,
                   "mxfp4": 1}.get(str(row.get("dtype") or dtype), 2)
        gamma_bytes = vector[0] * element
        break
    totals = _declared_pipe_totals(result)
    reads, writes = _ROW_STREAMS.get(kernel_class, (2, 1))
    declared_totals = dict(totals)
    row_bytes = max(1, width * dtype_bytes)
    read_bytes = totals.get("mte2")
    write_bytes = totals.get("mte3")
    if (kernel_class != "fused_row_norm"
            and (read_bytes is None or write_bytes is None)
            and program_meta.get("read_bytes_per_row") is not None):
        read_bytes = float(program_meta["read_bytes_per_row"]) * float(rows)
        write_bytes = (
            float(program_meta.get("write_bytes_per_row") or 0.0)
            * float(rows))
    else:
        if read_bytes is None:
            read_bytes = totals.get("hbm")
        if write_bytes is None:
            write_bytes = totals.get("hbm")
    if read_bytes is None or write_bytes is None:
        # No declared traffic: fall back to the declared stream counts.
        read_bytes = float(rows * reads * row_bytes)
        write_bytes = float(rows * writes * row_bytes)
    vector_ops = totals.get("vector")
    if vector_ops is None:
        vector_ops = float(rows * width * 2.0)
    _env = result.environment or {}
    _execution = _env.get("execution") or {}
    arch = str(_execution.get("arch") or _env.get("arch") or "")
    declared_conditions = dict((result.invocation or {}).get("conditions") or {})
    return {
        "rows": rows, "width": width, "dtype_bytes": dtype_bytes,
        "read_bytes": float(read_bytes), "write_bytes": float(write_bytes),
        "vector_ops": float(vector_ops), "gamma_bytes": int(gamma_bytes),
        "columns_split": kernel_class == "elementwise",
        "declared_totals": declared_totals,
        "arch": arch,
        "physics": "clock",
        "execution_scope": str(
            declared_conditions.get("execution_scope") or "serving"),
        "block_num": declared_conditions.get("block_num"),
    }


def _matrix_orientation(result: NativeAbiPrediction) -> dict[str, Any] | None:
    program = result.execution_program
    plan = getattr(program, "plan", None)
    descriptor = dict((getattr(plan, "metadata", {}) or {}).get(
        "operator_plan") or {})
    leaf = getattr(program, "leaf", None) or {}
    geometry = dict(leaf.get("geometry") or {})
    tile = descriptor.get("tile") or geometry.get("tile")
    if not isinstance(tile, (list, tuple)) or len(tile) != 3:
        return None
    try:
        m = int(geometry["m"])
        n = int(geometry["n"])
        k = int(geometry["k"])
    except (KeyError, TypeError, ValueError):
        return None
    totals = _declared_pipe_totals(result)
    work = dict(getattr(program, "work", {}) or {})
    resources = work.get("resources")
    resources = dict(resources) if isinstance(resources, Mapping) else {}
    bytes_ = dict((leaf.get("work") or {}).get("bytes") or {})
    read_bytes = resources.get("mte2", totals.get("mte2"))
    write_bytes = resources.get("mte3", bytes_.get("output"))
    if read_bytes is None:
        read_bytes = bytes_.get("total")
    if write_bytes is None:
        write_bytes = m * n * 2
    hbm_bytes = totals.get("hbm")
    if hbm_bytes is None:
        hbm_bytes = float(read_bytes or 0.0) + float(write_bytes or 0.0)
    vector_ops = totals.get("vector")
    if vector_ops is None:
        # A quantized matrix declares its epilogue as a named work record
        # (row/column scale + cast) instead of a flat pipe total.
        declared_vector = (result.prediction or {}).get("work", {}).get("vector")
        if isinstance(declared_vector, Mapping):
            for name in ("ops", "vector_ops", "epilogue_ops"):
                if declared_vector.get(name):
                    vector_ops = float(declared_vector[name])
                    break
    metadata = dict(leaf.get("metadata") or {})
    tensors = (result.invocation or {}).get("tensors") or {}
    in_tensors = [
        row for row in tensors.values()
        if isinstance(row, Mapping) and row.get("present") is not False
        and str(row.get("role") or "") != "output"
        and len(row.get("shape") or ()) >= 2
    ]
    out_tensors = [
        row for row in tensors.values()
        if isinstance(row, Mapping) and row.get("present") is not False
        and str(row.get("role") or "") == "output"
        and len(row.get("shape") or ()) >= 2
    ]
    in_dt = str(metadata.get("input_dtype")
                or (in_tensors[0].get("dtype") if in_tensors else "")
                or "").lower()
    w_row = in_tensors[1] if len(in_tensors) >= 2 else (
        in_tensors[0] if in_tensors else {})
    declared_attrs = dict((result.invocation or {}).get("attrs") or {})
    weight_dtype = str(metadata.get("weight_dtype")
                       or w_row.get("dtype") or in_dt or "").lower()
    weight_layout = str(metadata.get("weight_layout")
                        or declared_attrs.get("weight_layout")
                        or w_row.get("layout") or "ND").upper()
    if len(w_row.get("shape") or ()) >= 4:
        weight_layout = "FRACTAL_NZ"
    dt_map = {"fp32": 4, "float32": 4, "int8": 1, "uint8": 1,
              "fp8_e4m3": 1, "fp8_e5m2": 1, "mxfp8": 1, "fp8": 1,
              "dt_float8_e4m3fn": 1, "dt_int8": 1}
    dtype_bytes = int(metadata.get("dtype_bytes")
                      or dt_map.get(in_dt)
                      or dt_map.get(weight_dtype) or 2)
    out_dt = str(metadata.get("output_dtype")
                 or (out_tensors[0].get("dtype") if out_tensors else "")
                 or "bf16").lower()
    out_dtype_bytes = int(dt_map.get(out_dt, 2))
    if out_dtype_bytes != 2 and not resources.get("mte3") and not bytes_.get("output"):
        write_bytes = m * n * out_dtype_bytes
    _env = result.environment or {}
    _execution = _env.get("execution") or {}
    arch = str(_execution.get("arch") or _env.get("arch") or "")
    declared_conditions = dict((result.invocation or {}).get("conditions") or {})
    return {"m": m, "n": n, "k": k, "tile": tuple(int(item) for item in tile),
            "read_bytes": float(read_bytes or 0.0),
            "write_bytes": float(write_bytes or 0.0),
            "hbm_bytes": float(hbm_bytes or 0.0),
            "vector_ops": float(vector_ops or 0.0),
            "dtype_bytes": dtype_bytes,
            "out_dtype_bytes": out_dtype_bytes,
            "weight_dtype": weight_dtype,
            "weight_layout": weight_layout,
            "arch": arch,
            "physics": "clock",
            "block_num": declared_conditions.get("block_num"),
            "execution_scope": str(
                declared_conditions.get("execution_scope") or "serving")}


def _paged_orientation(result: NativeAbiPrediction) -> dict[str, Any] | None:
    program = result.execution_program
    native_plan = dict(getattr(program, "native_plan", {}) or {})
    geometry = dict(native_plan.get("geometry") or {})
    paging = dict(native_plan.get("captured_paging") or {})
    work = dict(native_plan.get("work") or {})
    if not geometry:
        return None
    lengths = [int(item) for item in paging.get("kv_lengths") or ()]
    visible = max(lengths) if lengths else int(
        geometry.get("kv_tokens") or 0)
    visible_scope = "declared_kv_lengths" if lengths else "declared_kv_tokens"
    if visible <= 0:
        # A captured kernel row retains the page table but not the runtime
        # sequence lengths.  The declared page-table capacity (pages x page
        # size) is the only physical extent the ABI carries, so the launch is
        # priced at that declared visible-KV bound instead of at zero.
        pages = int(geometry.get("max_pages") or 0)
        page = int(geometry.get("page_size") or 128)
        if pages > 0:
            visible = pages * page
            visible_scope = "block_table_capacity_bound"
        else:
            visible = 1
            visible_scope = "unresolved_visible_kv"
    # The selected plan's own resource ledger is the declared pipe work of
    # this launch: the paged streams, the visible KV extent, the block table,
    # the softmax vector and the output store.
    resources = dict((getattr(program, "work", {}) or {}).get("resources") or {})
    read_bytes = (float(resources.get("mte2_aic") or 0.0)
                  + float(resources.get("mte2_aiv") or 0.0))
    write_bytes = float(resources.get("mte3") or 0.0)
    hbm_bytes = float(resources.get("hbm") or 0.0)
    if not read_bytes:
        read_bytes = float(work.get("kv_valid_bytes") or 0.0)
    if not hbm_bytes:
        hbm_bytes = read_bytes + write_bytes
    return {
        "batch": int(geometry.get("batch") or 1),
        "kv_heads": int(geometry.get("kv_heads") or 1),
        "visible_kv": visible,
        "visible_kv_scope": visible_scope,
        "page_size": int(geometry.get("page_size") or 128),
        "head_dim": int(geometry.get("head_dim") or 128),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "hbm_bytes": hbm_bytes,
        "vector_ops": float(resources.get("vector") or 0.0),
        "head_token_pairs": int(work.get("head_token_pairs") or 0),
        "dtype_bytes": 2,
        "physics": "clock",
    }


def _kernel_supply(result: NativeAbiPrediction) -> KernelSupply:
    """Declared supply of the target die, with any caller rate overrides."""
    environment = dict(result.environment or {})
    hardware = environment.get("hardware")
    if not hasattr(hardware, "dies"):
        sku = hardware if isinstance(hardware, str) and hardware.startswith(
            "Ascend") else "Ascend950DT_9582"
        hardware = _execution_machine(sku)
    try:
        base = PipeSupply.from_hardware(hardware)
    except ModelingError:
        return KernelSupply(rates={}, invocations={}, units={})
    supply = KernelSupply.from_supply(base)
    declared = environment.get("pipe_supply")
    if isinstance(declared, Mapping):
        rates = {}
        for name, value in declared.items():
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if rate > 0:
                rates[str(name)] = rate
        if rates:
            supply = replace(supply, rates=rates)
    return supply


def _native_drafts(result: NativeAbiPrediction) -> tuple[list[StageDraft], str]:
    """Declared physical stages of the selected native program."""
    supply = _kernel_supply(result)
    kernel_class = _kernel_class(result)
    orientation: dict[str, Any] | None = None
    if kernel_class in {"matrix_bf16", "matrix_quant"}:
        orientation = _matrix_orientation(result)
    elif kernel_class == "paged_fia":
        orientation = _paged_orientation(result)
    elif kernel_class in {"fused_row_norm", "row_norm", "elementwise",
                          "quantize", "vector"}:
        orientation = _row_orientation(result, kernel_class)
    if orientation is not None:
        drafts = drafts_for(kernel_class, orientation, supply)
        if drafts:
            return drafts, kernel_class
    work = _declared_stage_work(result)
    return drafts_from_work(work, kernel_class=kernel_class, supply=supply), \
        kernel_class


def _declared_stage_work(
        result: NativeAbiPrediction
) -> tuple[tuple[tuple[str, float, float], ...], ...]:
    """Fallback declared stage chain (bytes/ops at the declared supply)."""
    prediction = result.prediction or {}
    stages: list[tuple[tuple[str, float, float], ...]] = []

    def _amount(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None
        return amount if amount > 0 else None

    for stage in prediction.get("stage_work") or ():
        entries = stage.items() if hasattr(stage, "items") else stage
        row = tuple((str(pipe), amount, 1.0)
                    for pipe, value in entries
                    if (amount := _amount(value)) is not None)
        if row:
            stages.append(row)
    if not stages and result.execution_program is not None:
        work = dict(getattr(result.execution_program, "work", {}) or {})
        row: list[tuple[str, float, float]] = []
        resources = work.get("resources")
        if isinstance(resources, Mapping):
            row.extend((str(pipe), amount, 1.0)
                       for pipe, value in resources.items()
                       if (amount := _amount(value)) is not None)
        matrix = work.get("matrix")
        if isinstance(matrix, Mapping):
            flops = sum(amount for value in matrix.values()
                        if (amount := _amount(value)) is not None)
            cube = _native_cube_rate(result)
            if flops > 0 and cube:
                row.append(("matrix", flops, 1.0 / cube))
        if row:
            stages.append(tuple(row))
    if not stages:
        work = prediction.get("work") or {}
        row = tuple((str(pipe), amount, 1.0)
                    for pipe, value in work.items()
                    if (amount := _amount(value)) is not None)
        if row:
            stages.append(tuple(row))
    return tuple(stages)


def _native_output_names(result: NativeAbiPrediction) -> tuple[str, ...]:
    tensors = (result.invocation or {}).get("tensors") or {}
    return tuple(str(name) for name, row in tensors.items()
                 if str(row.get("role") or "") == "output")


def _native_input_names(result: NativeAbiPrediction) -> tuple[str, ...]:
    tensors = (result.invocation or {}).get("tensors") or {}
    return tuple(str(name) for name, row in tensors.items()
                 if str(row.get("role") or "") != "output"
                 and row.get("present", True) is not False)


def native_catalog_for_result(
        result: NativeAbiPrediction, *,
        extra_events: tuple[str, ...] = (),
        output_grants: Mapping[str, Any] | None = None) -> ProgramCatalog:
    """Compile the selected physical plan into one immutable catalog.

    Pure: reads the already-normalized ABI/leaf prediction only.  No Engine,
    Event, allocation grant or call-site state enters the catalog, so the same
    result can be rebound for every actual caller occurrence.

    The stage chain is the real launch: declared kernel prologue/drain, the
    per-owner load/compute/store copies at the active owners' provisioned
    rates, and the Cube/Fixpipe or paged-KV body.  Declared bytes and ops are
    preserved exactly; only the launch physics around them is added here.
    """
    drafts, kernel_class = _native_drafts(result)
    if not drafts:
        raise ModelingError("native_catalog_stage_work_unavailable")
    inputs = _native_input_names(result)
    outputs = _native_output_names(result)
    events = tuple(dict.fromkeys(str(name) for name in extra_events
                                 if str(name) not in inputs))
    allocation_specs = []
    for index, (name, grant) in enumerate(dict(output_grants or {}).items()):
        if not isinstance(grant, Mapping) or grant.get("borrow"):
            continue
        pool = str(grant.get("pool") or "")
        size = grant.get("size")
        if not pool or size is None:
            raise ModelingError(
                f"native_catalog_grant_incomplete:{name}")
        allocation_specs.append(AllocationSpec(
            symbol=index + 1, label=str(name), pool_role=pool,
            amount=float(size), scope_kind="output", units=(0,),
            external_port=str(name)))
    environment = dict(result.environment or {})
    implementation = str((result.binding.call_id if result.binding is not None
                          else "") or (result.invocation or {}).get("abi_hash")
                         or "native")
    occurrence = result.occurrence or {}
    return catalog_from_drafts(
        drafts, implementation=implementation, inputs=inputs,
        outputs=outputs, events=events, allocation_specs=allocation_specs,
        environment=environment, occurrence=occurrence,
        semantic_sources=(
            str(item) for item in
            (tuple((result.program or {}).get("physical", {}).get(
                "physical_members") or ())
             or tuple(getattr(result.binding, "member_op_ids", ()) or ())
             or tuple(occurrence.get("members") or ()))),
        guard=(("entry", str((result.spec or {}).get("entry") or "")),
               ("typed", str((result.binding.typed if result.binding is not None
                              else {}) or "")),
               ("stage_count", len(drafts)),
               ("kernel_class", kernel_class)))

def native_binding_for_submission(
        catalog: ProgramCatalog, result: NativeAbiPrediction, execution: Any,
        *, name: str, ready: Any, occurrence: OccurrenceKey | None = None,
        inputs: Mapping[str, Any] | None = None,
        holds: Mapping[str, Any] | None = None,
        event_bindings: Mapping[str, Any] | None = None,
        replica: int = 0, stage: int = 0,
        output_grants: Mapping[str, Any] | None = None,
        allocations: tuple[Any, ...] | None = None,
        pool_bindings: Mapping[str, Any] | None = None) -> BindingFrame:
    """Bind the immutable catalog to one fresh caller occurrence.

    Inputs/holds are resolved to their actual readiness Events here.  No
    Event/Execution/allocation identity is cached on the prediction result.
    """
    engine = getattr(execution, "engine", execution)
    if occurrence is None:
        occurrence = engine.new_occurrence(replica=int(replica), rank=0,
                                           stage=int(stage), ordinal=0)
    if ready is None:
        ready = engine.submit(f"{name}:ready", owner="")
    start = ready.event if hasattr(ready, "event") else ready
    values: dict[str, PortBinding] = {}
    declared = {spec.name: spec for spec in catalog.ports}
    for port_name, spec in declared.items():
        if spec.direction != "in":
            continue
        supplied = None
        if inputs and port_name in inputs:
            supplied = inputs[port_name]
        elif holds and port_name in holds:
            supplied = holds[port_name]
        elif event_bindings and port_name in event_bindings:
            supplied = event_bindings[port_name]
        if supplied is None:
            if spec.optional:
                continue
            values[port_name] = PortBinding(ready=start, access=spec.access)
            continue
        if isinstance(supplied, PortBinding):
            values[port_name] = supplied
        elif hasattr(supplied, "done") and hasattr(supplied, "name"):
            values[port_name] = PortBinding(ready=supplied)
        elif hasattr(supplied, "event"):
            values[port_name] = PortBinding(ready=supplied.event)
        else:
            values[port_name] = PortBinding(ready=start, value=supplied)
    symbols = {spec.symbol: spec for spec in catalog.allocation_symbols}
    allocation_rows: dict[int, AllocationBinding] = {}
    for symbol, spec in symbols.items():
        grant = dict(output_grants or {}).get(spec.label) or {}
        if not grant or grant.get("borrow"):
            continue
        key = AllocationKey(
            occurrence=occurrence, scope_kind=spec.scope_kind,
            scope_ordinal=symbol - 1, symbol=symbol,
            generation=int(grant.get("generation") or 0))
        backing = grant.get("backing")
        if isinstance(backing, Mapping):
            backing = BackingKey(
                physical_domain=str(backing.get("physical_domain") or "device"),
                rank=int(backing.get("rank") or 0),
                tier=str(backing.get("tier") or "hbm"),
                storage_id=str(backing.get("storage_id") or spec.label),
                generation=int(backing.get("generation") or 0))
        elif not isinstance(backing, BackingKey):
            backing = None
        allocation_rows[symbol] = AllocationBinding(
            key=key, pool=_native_pool(grant, 0),
            amount=float(grant.get("size")),
            backing=backing, owner=str(spec.label),
            storage=str(grant.get("storage") or ""),
            access="write", acquire=True)
    for item in allocations or ():
        pool = getattr(item, "pool", None)
        amount = getattr(item, "amount", getattr(item, "size", None))
        if pool is None or amount is None:
            continue
        label = str(getattr(item, "name", ""))
        symbol = len(allocation_rows) + 1
        allocation_rows[symbol] = AllocationBinding(
            key=AllocationKey(occurrence=occurrence, scope_kind="invocation",
                              scope_ordinal=symbol - 1, symbol=symbol),
            pool=PoolKey(physical_domain="device", rank=0, tier="hbm",
                         name=str(pool)),
            amount=float(amount), owner=label, access="readwrite",
            acquire=True)
    geometry = {
        "rows": int((result.prediction or {}).get("environment", {}).get(
            "rows") or 0),
        "dtype": str((result.invocation or {}).get("attrs", {}).get(
            "dtype") or ""),
        "layout": str((result.invocation or {}).get("attrs", {}).get(
            "layout") or ""),
        "phase": str((result.prediction or {}).get("environment", {}).get(
            "phase") or ""),
    }
    return BindingFrame(
        occurrence=occurrence, geometry=geometry, values=values,
        resources=_execution_resource_bindings(execution, replica, stage),
        allocations=allocation_rows,
        pool_bindings={spec.label: _native_pool(
            dict(output_grants or {}).get(spec.label) or
            {"pool": spec.pool_role}, 0)
            for spec in catalog.allocation_symbols
            if (output_grants or {}).get(spec.label)
            and not (output_grants or {}).get(spec.label, {}).get("borrow")},
    )


def catalog_modeled_duration_ms(result: "NativeAbiPrediction") -> float | None:
    """Analytic duration of the selected catalog chain (same arithmetic as the Engine).

    Each stage runs after the previous one releases, so the call is the sum of
    the per-stage maxima.  ``host`` terms are elapsed time at unit rate; every
    other term is ``issued * conversion / declared_rate``.  Returns ``None``
    when no catalog can be compiled.
    """
    try:
        drafts, _kernel_class = _native_drafts(result)
    except ModelingError:
        return None
    if not drafts:
        return None
    supply = _kernel_supply(result)
    total = 0.0
    for draft in drafts:
        stage = float(draft.latency_ms)
        for pipe, issued, conversion in draft.work:
            declared = supply.rate(pipe)
            rate = float(declared) if declared and declared > 0 else 1.0
            stage = max(stage, float(issued) * float(conversion) / rate)
        total += stage
    return total
