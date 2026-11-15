"""Paged FlashInfer attention geometry, page tail and dispatch.

[Role] Map the ordered FIA slots to BNSD query / BNBD paged KV, keep the
  captured page table and visible KV extent (including a partial last page),
  and price the S1 kernel body.
[Invariants] The visible extent is never rounded to a page.  A page tail is a
  real whole page of DMA traffic.  No measured duration enters the mapping.
[Timeouts & Locks] Pure arithmetic; no waits or locks.
[Side-Effect Policy] Read-only; returns the paged plan and prediction.
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
    PagedFIAExecutionProgram, _candidate_kernels, _generic_program, _selection_program_options, _source_establishment,
)
from .native_leaf_dispatch import (  # noqa: E402
    _invocation_replay,
)
from .native_vector_abi import (  # noqa: E402
    _full_typed,
)
def _paged_fia_output_tokens(out_shape, query_shape, query_heads, query_dim,
                             total_tokens):
    """Resolve the output geometry of a captured paged FIA row.

    Four accepted declarations describe the same buffer:
      * TND ``[tokens, heads, dim]`` with the identical query shape,
      * packed ``[tokens, heads * dim]``,
      * BNSD ``[batch, heads, q, dim]``,
      * BSND ``[batch, q, heads, dim]`` (the head axis is the one that carries
        the declared head count, so both 4-D spellings are resolved from the
        declared geometry instead of from a captured name).
    """
    out_shape = [int(item) for item in out_shape]
    query_shape = [int(item) for item in query_shape]
    if len(out_shape) == len(query_shape) == 4:
        # BNSD ``[batch, heads, q, D_v]``, BSND ``[batch, q, heads, D_v]`` and
        # the head-major ``[heads, batch, q, D_v]`` spelling all keep the
        # batch/query product; the head axis is the one that carries the
        # declared query-head count.  The value head width may differ from the
        # query/key width, so it is not used as an identity.
        for head_axis, token_axis in ((1, 2), (2, 1), (0, 1), (0, 2)):
            if out_shape[head_axis] != int(query_heads):
                continue
            others = [axis for axis in range(4)
                      if axis not in {head_axis, 3}]
            product = 1
            for axis in others:
                product *= out_shape[axis]
            if product == total_tokens:
                return total_tokens
        return None
    if len(out_shape) == len(query_shape) == 3:
        if out_shape != query_shape:
            return None
        return out_shape[0]
    if len(out_shape) == 2 and len(query_shape) == 3:
        if (out_shape[0] == query_shape[0]
                and out_shape[1] == int(query_heads) * int(query_dim)):
            return out_shape[0]
        return None
    if len(out_shape) == 1 and len(query_shape) == 3:
        return out_shape[0] if out_shape[0] == query_shape[0] else None
    return None


def _paged_fia_geometry(inv: NativeAbiInvocation,
                        spec: NativeAbiCallSpec) -> dict:
    """Derive the ordinary paged-FIA op/ports from one normalized invocation.

    The captured ACLNN slot order names ``key``/``value`` and
    ``actual_seq_lengths_kv``; the vLLM-style alias set (``key_cache``/
    ``value_cache``/``sequence_lengths``) is normalized before this point.
    Query may be TND ``[tokens, heads, dim]`` or BNSD
    ``[batch, heads, q, dim]``; the paged KV block may be page-major
    ``[pages, block_size, kv_heads, dim]`` or head-major
    ``[pages, kv_heads, block_size, dim]``.  Both are resolved from the
    declared ``block_size`` and head counts, never from an op name.
    """
    from mindie_sim.schema.operators import OpSpec
    tensors = inv.tensor_map()
    required = ("query", "key", "value", "block_table",
                "actual_seq_lengths_kv", "output")
    missing = [name for name in required
               if name not in tensors or not tensors[name].present]
    if missing:
        raise NativeAbiIncomplete(
            "native_paged_fia_required_tensors",
            tuple("tensors." + name for name in missing))
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    block = tensors["block_table"]
    lengths = tensors["actual_seq_lengths_kv"]
    output = tensors["output"]
    if len(query.shape) == 4:
        batch_q, query_heads, query_length, query_dim = query.shape
        query_layout = "BNSD"
        total_tokens = int(batch_q) * int(query_length)
    elif len(query.shape) == 3:
        total_tokens, query_heads, query_dim = query.shape
        batch_q = None
        query_length = None
        query_layout = "TND"
    else:
        raise NativeAbiMismatch("native_paged_fia_query_rank_must_be_3_or_4")
    out_tokens = _paged_fia_output_tokens(
        output.shape, query.shape, int(query_heads), int(query_dim),
        int(total_tokens))
    if out_tokens is None:
        raise NativeAbiMismatch("native_paged_fia_output_shape_mismatch")
    if len(key.shape) != 4 or len(value.shape) != 4:
        raise NativeAbiMismatch("native_paged_fia_cache_rank_must_be_4")
    if len(block.shape) != 2 or len(lengths.shape) != 1:
        raise NativeAbiMismatch("native_paged_fia_page_table_rank_invalid")
    if list(key.shape) != list(value.shape):
        raise NativeAbiMismatch("native_paged_fia_key_value_shape_mismatch")
    page_size = int(inv.attrs.get("block_size")
                    or inv.attrs.get("page_size") or 0)
    declared_heads = int(inv.attrs.get("num_kv_heads")
                         or inv.attrs.get("num_key_value_heads")
                         or inv.attrs.get("num_kv_head") or 0)
    if declared_heads < 0 or page_size < 0:
        raise NativeAbiMismatch("native_paged_fia_block_size_mismatch")
    if page_size and declared_heads:
        if (int(key.shape[1]) == page_size
                and int(key.shape[2]) == declared_heads):
            paged_layout = "PA_BBND"
        elif (int(key.shape[1]) == declared_heads
                and int(key.shape[2]) == page_size):
            paged_layout = "BNBD"
        else:
            raise NativeAbiMismatch("native_paged_fia_block_size_mismatch")
        kv_heads = declared_heads
    elif int(key.shape[1]) in {page_size} and page_size:
        paged_layout = "PA_BBND"
        kv_heads = int(key.shape[2])
    elif declared_heads and int(key.shape[1]) == declared_heads:
        paged_layout = "BNBD"
        kv_heads = declared_heads
        page_size = int(key.shape[2])
    elif key.layout.upper() == "PA_BBND":
        paged_layout = "PA_BBND"
        page_size = int(key.shape[1])
        kv_heads = int(key.shape[2])
    else:
        # The only remaining declaration is the captured ACLNN block layout
        # ``[num_blocks, kv_heads, block_size, head_dim]``.
        paged_layout = "BNBD"
        kv_heads = int(key.shape[1])
        page_size = int(key.shape[2])
    if page_size < 1 or kv_heads < 1:
        raise NativeAbiMismatch("native_paged_fia_block_size_mismatch")
    # The vLLM-style ``key_cache`` declaration is a page-major PA_BBND view;
    # an ND format on that view is a real mismatch.  The captured ACLNN block
    # layout is head-major BNBD and legitimately carries ND, so it is not
    # rejected here.
    if (key.layout.upper() == "ND" and paged_layout == "PA_BBND"):
        raise NativeAbiMismatch(
            "native_paged_fia_requires_pa_bbnd_key_value")
    head_dim = int(key.shape[3])
    if int(query_dim) != head_dim:
        raise NativeAbiMismatch("native_paged_fia_head_dim_mismatch")
    if int(query_heads) != int(query.shape[1]):
        raise NativeAbiMismatch("native_paged_fia_head_count_mismatch")
    declared_query_heads = int(inv.attrs.get("num_heads") or query_heads)
    if declared_query_heads != int(query_heads):
        raise NativeAbiMismatch("native_paged_fia_head_count_mismatch")
    if int(key.shape[1]) != kv_heads and paged_layout == "BNBD":
        raise NativeAbiMismatch("native_paged_fia_head_count_mismatch")
    if head_dim == 256:
        raise NativeAbiMismatch("native_paged_fia_d256_specialized_ifa")
    if int(query_heads) % kv_heads:
        raise NativeAbiMismatch(
            "native_paged_fia_query_heads_not_divisible_by_kv_heads")
    batch = int(block.shape[0])
    if int(lengths.shape[0]) != batch:
        raise NativeAbiMismatch("native_paged_fia_sequence_lengths_batch")
    if query_layout == "BNSD" and int(batch_q) != batch:
        raise NativeAbiMismatch("native_paged_fia_query_batch_mismatch")
    real_dtype = str(key.dtype)
    if real_dtype != str(value.dtype):
        raise NativeAbiMismatch("native_paged_fia_key_value_dtype_mismatch")
    if str(query.dtype) != str(output.dtype):
        raise NativeAbiMismatch("native_paged_fia_output_dtype_mismatch")

    def attr_sequence(name):
        value = inv.attrs.get(name)
        if value is None:
            value = inv.conditions.get(name)
        if value is None:
            return None
        if isinstance(value, (str, bytes)) or not isinstance(
                value, (list, tuple)):
            raise NativeAbiMismatch(
                f"native_paged_fia_{name}_must_be_sequence")
        return tuple(int(item) for item in value)

    if query_layout == "BNSD":
        query_lengths = (int(query_length),) * batch
    else:
        query_lengths = attr_sequence("query_lengths")
        if query_lengths is None:
            if total_tokens == batch:
                query_lengths = (1,) * batch
            elif batch and total_tokens % batch == 0:
                query_lengths = (total_tokens // batch,) * batch
            else:
                raise NativeAbiMismatch(
                    "native_paged_fia_query_lengths_unresolved")
    kv_lengths = attr_sequence("kv_lengths")
    if kv_lengths is None:
        fields = dict(lengths.fields)
        for name in ("values", "value", "data", "sequence_lengths",
                     "actual_seq_lengths_kv"):
            value = fields.get(name)
            if isinstance(value, (list, tuple)):
                kv_lengths = tuple(int(item) for item in value)
                break
    if kv_lengths is None:
        kv_lengths = attr_sequence("actual_seq_lengths_kv")
    if kv_lengths is None:
        # The caller may declare the runtime context length it actually ran
        # (``kv_len``/``context_length``/``seq_len``).  That is a timing-blind
        # workload declaration and it is preferred over the allocated table
        # capacity, which is only an upper bound.
        declared_kv = None
        for name in ("kv_len", "context_length", "seq_len"):
            value = inv.conditions.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and int(value) > 0:
                declared_kv = int(value)
                break
        if declared_kv is not None:
            kv_lengths = (declared_kv,) * batch
    if kv_lengths is None:
        # A timing-free kernel row retains the block table's declared shape but
        # not the host-side length values.  The table width is the allocated
        # per-sequence page count, so the physical page walk is bounded by the
        # declared cache geometry (width x page size), not fabricated.
        block_shape = [int(item) for item in getattr(block, "shape", ()) or ()]
        if len(block_shape) >= 2 and block_shape[1] > 0:
            kv_lengths = (int(block_shape[1]) * int(page_size),) * batch
    if kv_lengths is None:
        raise NativeAbiMismatch(
            "native_paged_fia_actual_kv_lengths_required")
    if len(query_lengths) != batch or len(kv_lengths) != batch:
        raise NativeAbiMismatch("native_paged_fia_lengths_batch")
    if sum(query_lengths) != total_tokens:
        raise NativeAbiMismatch("native_paged_fia_query_lengths_mismatch")
    requests = []
    for query_len, kv_len in zip(query_lengths, kv_lengths):
        if query_len < 1 or kv_len < query_len:
            raise NativeAbiMismatch("native_paged_fia_length_invalid")
        requests.append((query_len, kv_len - query_len))
    page_counts = tuple((kv + page_size - 1) // page_size
                        for kv in kv_lengths)
    if int(block.shape[1]) < max(page_counts, default=0):
        raise NativeAbiMismatch(
            "native_paged_fia_block_table_capacity_too_small")
    phase = str(inv.conditions.get("phase")
                or inv.attrs.get("phase") or "decode")
    if phase not in {"decode", "draft"}:
        raise NativeAbiMismatch(
            f"native_paged_fia_phase_not_ordinary_decode:{phase}")
    gqa = {
        "requests": requests,
        "query_heads": int(query_heads), "kv_heads": kv_heads,
        "head_dim": head_dim, "query_dtype": str(query.dtype),
        "kv_dtype": real_dtype, "causal": bool(
            inv.attrs.get("causal", 1)), "window": 0, "cp": 1,
        "page_size": page_size, "paged_layout": paged_layout,
        "page_counts": list(page_counts),
    }
    view = {
        "kind": "paged_gqa", "logical_layout": query_layout,
        "kv_storage": "paged", "paged_layout": paged_layout,
        "query_shape": list(query.shape),
        "output_shape": list(output.shape),
        "key_tail_shape": [page_size, kv_heads * head_dim],
        "value_tail_shape": [page_size, kv_heads * head_dim],
        "query_dtype": str(query.dtype),
        "output_dtype": str(output.dtype),
        "key_dtype": real_dtype, "value_dtype": real_dtype,
        "batch_count": batch, "kv_block_size": page_size,
        "local_kv_heads": kv_heads, "head_dim": head_dim,
        "cache_row_width": 2 * kv_heads * head_dim,
        "cache_head_width": head_dim,
        "cache_source_residency": "unresolved",
        "page_counts": list(page_counts),
    }
    shape = (total_tokens, max(kv for _q, kv in requests), int(query_heads),
             head_dim)
    op = OpSpec(
        "gqa_decode", "attention", phase, 0,
        int(inv.conditions.get("layer") or 0),
        tuple(f"r{index}" for index in range(batch)),
        shape, shape, shape, shape, recipe="staged_gqa_decode",
        dtype=str(query.dtype),
        metadata=(("gqa_decode", gqa),
                  ("attention_tensor_view", view)))
    return {"op": op, "native_view": view, "gqa": gqa,
            "query_lengths": query_lengths, "kv_lengths": kv_lengths}


#: Declared query-rope width of the absorbed-MLA KV payload.  The absorbed
#: cache stores ``query_nope`` (``head_dim``) and ``query_rope`` per token, so
#: the active KV bytes per token are ``head_dim + MLA_ROPE_DIM`` elements.
MLA_ROPE_DIM = 64


def _paged_fia_options(selection: Mapping[str, Any] | None) -> dict:
    options = _selection_program_options(selection)
    result = {}
    for key in ("key_tile", "query_tile", "kv_splits", "slots",
                "target_cores", "kv_mapping"):
        if key in options and options[key] is not None:
            result[key] = options[key]
    return result


def _paged_fia_result(inv: NativeAbiInvocation, spec: NativeAbiCallSpec,
                      environment: dict, *, hardware: Any,
                      execution: Any = None, replica: int = 0,
                      stage: int = 0,
                      selection: Mapping[str, Any] | None = None,
                      strict: bool = False) -> NativeAbiPrediction:
    from mindie_sim.prediction.attention_decode import StagedGQADecode
    from mindie_sim.prediction.mechanism import Geometry
    try:
        descriptor = _paged_fia_geometry(inv, spec)
        recipe = StagedGQADecode(**_paged_fia_options(selection))
        if execution is not None:
            geometry = execution.stage_geometry.get((replica, stage))
            if geometry is None:
                geometry = execution.stage_geometry[(replica, 0)]
        else:
            machine = _execution_machine(hardware)
            geometry = Geometry.from_devices(tuple(machine.dies[:1]))
        plan = recipe.build(descriptor["op"], geometry)
        native_plan = recipe.native_plan(descriptor["op"])
        if native_plan is None:
            raise NativeAbiMismatch(
                "native_paged_fia_native_plan_unavailable")
        # Keep the captured paging declaration on the selected plan so the
        # public projection shows the real page tail (layout and per-request
        # extents), not only the page-rounded cache capacity.  It is a
        # separate capture view: the recipe's own geometry keys are unchanged.
        native_plan = dict(native_plan)
        native_plan["captured_paging"] = {
            "paged_layout": descriptor["gqa"]["paged_layout"],
            "requests": [list(item) for item in descriptor["gqa"]["requests"]],
            "kv_lengths": list(descriptor["kv_lengths"]),
            "query_lengths": list(descriptor["query_lengths"]),
            "page_counts": list(descriptor["gqa"]["page_counts"]),
        }
    except NativeAbiMismatch as error:
        return _strict(_result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(getattr(error, "reason", str(error)),),
            environment=environment), strict)
    except (ModelingError, ValueError) as error:
        return _strict(_result(
            "prediction_failed", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=(f"native_paged_fia_plan_failed:"
                     f"{getattr(error, 'reason', error)}",),
            environment=environment), strict)

    typed = _full_typed(
        inv, spec, {
            "execution_shape": [sum(descriptor["query_lengths"]),
                                descriptor["gqa"]["query_heads"]
                                * descriptor["gqa"]["head_dim"]],
            "source_shapes": {
                name: list(tensor.shape)
                for name, tensor in inv.tensor_map().items()
                if tensor.present}},
        execution_view=[sum(descriptor["query_lengths"]),
                        descriptor["gqa"]["query_heads"]
                        * descriptor["gqa"]["head_dim"]],
        extra={"variant": "ordinary_paged_fia",
               "source": None})
    region = NativeAbiRegion(
        op_id="native:" + str(spec.entry), family="native",
        recipe="paged_fia", source_boundary="call",
        dtype=descriptor["gqa"]["query_dtype"], layout="TND",
        metadata=(("leaf", spec.leaf),
                  ("native_plan", native_plan["schema"])))
    source_state = _source_establishment(spec, inv)
    binding = bind_abi_call(
        region,
        api={"namespace": inv.api["namespace"], "entry": inv.api["entry"],
             "aclnn_symbol": inv.api.get("aclnn_symbol"),
             "version": inv.version.get("api") or spec.api_version},
        source=source_state["binding_source"], typed=typed,
        kernels=_candidate_kernels(spec.kernels), members=(),
        owner="native:" + str(spec.entry),
        execution_core=spec.execution_core,
        unresolved=tuple(dict.fromkeys(
            (*spec.unresolved,
             "deployment_image_identity_not_established",
             "binary_identity_not_established"))),
        empty_members=True)
    if binding is None:
        return _strict(_result(
            "no_match", spec=spec, invocation=inv.to_dict(),
            replay=_invocation_replay(inv, spec),
            reasons=("native_abi_binding_unavailable",),
            environment=environment), strict)
    work = dict(native_plan["work"])
    # Absorbed-MLA decode (one declared KV head of width ``nope + rope``) has
    # its own declared paged-clock law: the visible window is a straight line
    # in ``batch * kv_len``.  A plain GQA paged tail keeps the analytic recipe
    # duration, which already prices its own walk/Cube stages.
    absorbed = (int(descriptor["gqa"]["kv_heads"]) == 1
                and int(descriptor["gqa"]["head_dim"]) >= 256)
    elapsed = None
    if absorbed:
        from .native_clock_die_laws import mla_paged_times
        from .native_clock_physics import (
            dtype_bytes as _dtype_bytes, rates_from_supply)
        kv_max = max(descriptor["kv_lengths"])
        batch_count = int(descriptor["gqa"]["requests"].__len__())
        kv_elem_bytes = float(_dtype_bytes(descriptor["gqa"]["kv_dtype"]))
        active_bytes = (float(batch_count) * float(kv_max)
                        * float(descriptor["gqa"]["head_dim"] + MLA_ROPE_DIM)
                        * kv_elem_bytes)
        key_tensor = inv.tensor_map().get("key")
        if (key_tensor is not None and key_tensor.present
                and len(key_tensor.shape) == 4 and min(key_tensor.shape) > 0):
            # Cold HBM traffic from a paged KV pool cannot exceed the physical
            # pool capacity (shared prefix pages hit in L2/page cache).
            pool_tokens = (float(key_tensor.shape[0])
                           * float(key_tensor.shape[2]))
            pool_bytes = (pool_tokens
                          * float(descriptor["gqa"]["head_dim"] + MLA_ROPE_DIM)
                          * kv_elem_bytes)
            active_bytes = min(active_bytes, pool_bytes)
        elapsed = mla_paged_times(
            batch=batch_count, kv_len=kv_max, active_bytes=active_bytes,
            rates=rates_from_supply(environment.get("pipe_supply")))
    prediction = {
        "basis": "ordinary_paged_fia_native_plan",
        "duration_ms": (str(elapsed.total_ms()) if elapsed is not None
                        else None),
        "duration_us": (float(elapsed.total_ms()) * 1000.0
                        if elapsed is not None else None),
        "dominant_pipe": ("hbm" if elapsed is not None else None),
        "tied_pipes": (["hbm"] if elapsed is not None else []),
        "work": work,
        "times_ms": ({pipe: str(value) for pipe, value in elapsed.body}
                     if elapsed is not None else {}),
        "stage_times_ms": ([str(value) for _pipe, value in elapsed.body]
                           if elapsed is not None else []),
        "materialization": {
            "tensors": {name: {"shape": list(tensor.shape),
                               "dtype": tensor.dtype,
                               "layout": tensor.layout}
                        for name, tensor in inv.tensor_map().items()
                        if tensor.present}},
        "derived": {"native_paged_fia": native_plan},
        "formula": ("native_clock_physics.mla_paged_times"
                    if elapsed is not None else None),
        "fidelity": ("declared_clock_physics" if elapsed is not None
                     else "timing_free_work"),
        "environment": environment,
        "non_claims": ["ordinary_paged_fia", "no_lse",
                       "read_only_kv", "no_target_duration_injected"],
    }
    program = _generic_program(
        spec=spec, leaf_program={
            "mode": "paged_fia",
            "entry": spec.entry,
            "native_paged_fia": native_plan,
            "geometry": native_plan["geometry"],
            "ports": native_plan["ports"],
            "strategy": native_plan["strategy"],
            "state": native_plan["state"],
            "work": work,
            "builder": "StagedGQADecode.build",
        }, prediction=prediction, binding=binding, inv=inv)
    program["physical"]["native_paged_fia"] = native_plan
    program["physical"]["typed_ports"] = native_plan["ports"]
    prediction["program"] = program
    occurrence = {
        "schema": "mindie-sim/piecewise-native-occurrence/1",
        "status": "predicted", "call_id": binding.call_id,
        "op_id": binding.op_id, "family": binding.family,
        "recipe": binding.recipe, "source_boundary": binding.source_boundary,
        "phase": str(descriptor["op"].phase), "kind": "native",
        "mode": "native_call", "layer": inv.conditions.get("layer"),
        "semantic_refs": [], "members": [],
        "request_conditions": dict(inv.conditions),
        "repeat": 1, "index": 0, "rank": None, "charge": "call",
        "call_owner": binding.op_id,
        "timing_owner": dict(binding.timing_owner),
        "typed": {"native_paged_fia": native_plan,
                  "ports": native_plan["ports"]},
        "priced": {"duration_ms": None, "dominant_pipe": None,
                   "work": work, "charged": "once_at_timing_owner"},
        "signature_matches": True, "unresolved": list(binding.unresolved),
    }
    # The absorbed-MLA launch is priced by the declared paged clock law, so it
    # carries no second engine plan that would re-price the same launch from a
    # different stage chain.  An ordinary GQA paged tail keeps its executable
    # recipe program.
    execution_program = None if absorbed else PagedFIAExecutionProgram(
        op=descriptor["op"], recipe=recipe, plan=plan,
        native_plan=native_plan, inv=inv, spec=spec, selection=selection)
    return _strict(_result(
        "predicted", spec=spec, invocation=inv.to_dict(), binding=binding,
        prediction=prediction, program=program, occurrence=occurrence,
        operator_plan=plan, execution_program=execution_program,
        replay=_invocation_replay(inv, spec), reasons=(),
        environment=environment, binding_bound=binding.bound,
        source_establishment=source_state), strict)
