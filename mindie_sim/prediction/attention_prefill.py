"""Staged analytic GQA prefill on the common finite resource engine.

[Role]
  Lower the declared ``metadata['gqa_prefill']`` geometry into one task per
  ``(request, query_head, query_token_tile)``.  A task keeps its query tile and
  online softmax state resident, streams bounded K/V key tiles for that query
  head, and writes its output once.  Tasks are cycled over the actual physical
  owners.

[Invariants]
  * Q, K/V and output traffic follow real request lengths.  A task reloads K/V
    per query head; the flattened decode GQA reuse mapping is not applied.
  * QK and PV each execute the full ``query_rows * actual_kv_rows`` rectangle,
    including the causal-masked diagonal.  Logical valid pairs are reported but
    never substituted for that physical execution.
  * Online max/sum/numerator persist across key tiles.  Scores are ready before
    PV, and the previous accumulator is complete before the next update.
  * On-chip buffers use advertised pools and are released by their owner; they
    are never charged to HBM.  Finished tasks, programs and buffers are released.

[Timeouts & Locks]
  Planning is pure arithmetic.  Tasks are generated lazily by owner stride and
  key tiles by bounded pipeline window; no all-pairs or all-task list is kept.

[Side-Effect Policy]
  No device runtimes, measurements, files or external I/O.  Programs submit only
  to the caller's virtual ``Engine``.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Mapping

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import (
    Geometry, OperatorPlan, PlannedRecipe, Work,
)
from mindie_sim.prediction.ownership import read_role, unit_group
from .attention_kv_access import KVCacheAccess, kv_access
from mindie_sim.prediction.attention_prefill_issue import (
    source_c2_path, source_compatible_alternatives, source_issue_kind,
    source_issue_operand,
)
from mindie_sim.prediction.attention_prefill_nofd import _PrefillProgram, owner_s2_plan
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, RecipeContext, Stage
from mindie_sim.prediction.refill_steady import compact_periodic_pipeline
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

IMPLEMENTATION = "staged_gqa_prefill"
RECIPE_ID = IMPLEMENTATION

# Mechanism references pinned at ops-transformer HEAD.  They support the
# declared head-local policy; they are not evidence of a captured dispatch.
SOURCE_REFERENCES = (
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
     "748-864,1095-1178",
     "Q heads and outer query blocks are independent tiling axes"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/fused_infer_attention_score/op_host/arch22/"
     "fused_infer_attention_score_tiling_v3.cpp", "1403-1408",
     "arch22 RouteToFia rejects TND without rope for headDim 256"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h",
     "220-253,373-399",
     "FAInfer also uses one query head per task when query length exceeds 128"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
     "6415-6443",
     "BF16 CVDiff starts Souter at 64 for aligned head size >=200 and at 128 "
     "for head size >=32; the query-tile panel is selected by head dimension"),
    ("cann/ops-transformer", "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
     "attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp",
     "161-167,2886-2894",
     "PP_BLOCK_BUFFER_SIZE=128*128 bounds the resident query/key panel; "
     "mUbd=floor(PP_BLOCK_BUFFER_SIZE/embeddingSizeAligned/128)*128"),
)

_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
    "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2",
    "int8": "int8", "uint8": "uint8", "int4": "int4",
    "fp4": "fp4_e2m1", "fp4_e2m1": "fp4_e2m1",
}
_FP32_BYTES = 4
_PREFILL_PHASES = {"prefill", "draft_prefill"}
# Source CVDiff Souter start (prompt_flash_attention_tiling.cpp 6415-6443).
# The query panel is chosen by the aligned head dimension, not by a request
# length or a measured duration.  Explicit recipe options still override it.
_SOURCE_QUERY_TILE_LARGE_HEAD = 64      # alignedHeadSize >= 200
_SOURCE_QUERY_TILE_DEFAULT = 128        # 32 <= alignedHeadSize < 200
_SOURCE_LARGE_HEAD_THRESHOLD = 200


def source_softmax_outer(head_dim: int) -> int:
    """Arch35 ``softmaxSOuterFactor`` for the selected V4 FIA branch.

    ``FusedInferAttentionScoreTilingImpl::AdjustSinnerAndSouter`` starts at
    ``SOUTER_64`` and sets ``softmaxSOuterFactor=SOUTER_32`` only when
    ``vHeadDim > 128`` (and the prefill S1 is not 1).  This is the vector
    chunk, not the cube S1 basic block: ``sOuterSize = SOuterFactor*CV_RATIO``
    is 128 for the default V4 branch.
    """
    dim = _positive_int(head_dim, "head_dim")
    return 32 if dim > 128 else 64


def arch35_query_tile(head_dim: int) -> int:
    """Arch35 V4 cube S1 basic block (``SOuterFactor*CV_RATIO``).

    ``AdjustSinnerAndSouter`` defaults SOuter to 64; ``CV_RATIO`` is 2, so
    one cube task owns 128 query rows.  D256 non-quant GQA is excluded from
    the registered templates and uses this V4 branch; D128 uses the same
    128-row cube block in the fallback branch.
    """
    _positive_int(head_dim, "head_dim")
    return 128


def source_query_tile(head_dim: int) -> int:
    """CVDiff Souter start for the declared head dimension (source tiling)."""
    dim = _positive_int(head_dim, "head_dim")
    return (_SOURCE_QUERY_TILE_LARGE_HEAD
            if dim >= _SOURCE_LARGE_HEAD_THRESHOLD
            else _SOURCE_QUERY_TILE_DEFAULT)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"attention_prefill_{name}_must_be_positive_int")
    return value


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"attention_prefill_{name}_must_be_nonnegative_int")
    return value


def _dtype(value, name: str) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"attention_prefill_{name}_must_be_dtype")
    try:
        return _DTYPE_ALIASES[value.strip().lower()]
    except KeyError:
        raise ModelingError(f"attention_prefill_unsupported_{name}:{value}") from None


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def logical_causal_pairs(rows: int, context: int, first_token: int = 0) -> int:
    """Valid causal pairs for contiguous query rows starting at ``first_token``."""
    return rows * (context + first_token + 1) + rows * (rows - 1) // 2


# Unselected native path.  GQA's resident set is PRELOAD_N=3.  Simplified
# runInfo[4] is four task descriptors, not four K/V panels; that cursor runs
# only when issue_kind is explicit.  Depth 3 here is not a 589824 B refusal.
NATIVE_S2_PRELOAD = 3


def _explicit_reconstruct(metadata: Mapping):
    """Top-level flag only.  Absence does not select Simplified or GQA."""
    if "reconstruct" not in metadata:
        return None
    value = metadata["reconstruct"]
    if value is not True and value is not False:
        raise ModelingError("attention_prefill_reconstruct_must_be_bool_or_absent")
    return value


@dataclass(frozen=True)
class StagedGQAPrefill:
    """Explicit prefill tiling policy.

    ``query_tile`` bounds query rows per task; ``None`` selects the native
    CVDiff start from the declared head dimension.  ``key_tile`` bounds the
    actual K/V rows in one key-tile step.  ``slots`` is the finite K/V pipeline
    window.  Explicit integers remain available for counterfactual schedules.
    """

    query_tile: int | None = None
    key_tile: int = 128
    # Generic K/V double buffer for non-native callers.  Native FIA with no
    # explicit reconstruct flag keeps preload 3.  That is not a template pick.
    slots: int = 2
    # SOFTMAX_BUFFER_NUM=3 is scratch capacity for softmax Exp/BMM2, not
    # concurrent tile depth.
    softmax_slots: int = 3

    def __post_init__(self):
        if self.query_tile is not None:
            _positive_int(self.query_tile, "query_tile")
        _positive_int(self.key_tile, "key_tile")
        _positive_int(self.slots, "slots")
        _positive_int(self.softmax_slots, "softmax_slots")

    @classmethod
    def arch35_noquant_candidate(cls, **overrides):
        """Explicit counterfactual with the source S2 preload already used natively."""
        options = {"slots": 3, "query_tile": 128, "key_tile": 128}
        options.update(overrides)
        return cls(**options)

    def spec(self, op: OpSpec) -> "_Spec":
        return _spec(op, self)

    def work(self, op: OpSpec, *, include_mte1: bool = False,
             read: str = "mte2", vread: str | None = None,
             write: str | None = "mte3", operand: str | None = "mte1") -> Work:
        """Physical work, including the executed causal rectangle."""
        return _total_work(self.spec(op), include_mte1=include_mte1,
                           read=read, vread=vread, write=write, operand=operand)

    def report(self, op: OpSpec, *, resource_roles=None,
               cores: int = 32) -> dict:
        """Geometry ledger separating logical pairs from executed rectangles.

        ``cores`` enables the source S2 ownership/FD ledger.  FD bytes are a
        workspace diagnostic until the split-range execution is bound; they
        are not silently added to the cyclic task work.
        """
        spec = self.spec(op)
        report = _report(spec)
        if spec.native_fia:
            gqa = dict(op.metadata).get("gqa_prefill")
            fields = gqa if isinstance(gqa, Mapping) else {}
            report["source_compatible"] = source_compatible_alternatives(
                spec.head_dim,
                tuple(query for query, _context in spec.requests),
                query_dtype=spec.query_dtype,
                kv_dtype=spec.kv_dtype,
                out_dtype=fields.get("out_dtype", spec.query_dtype),
                layout=fields.get("input_layout", "TND"),
                quant=fields.get("quant", "NoQuant"),
                mask=fields.get("mask", True),
                rope=fields.get("rope", False),
                kv_paged=fields.get("kv_paged", False),
                kv_prefix=fields.get("kv_prefix", False),
            )
        plan = owner_s2_plan(spec, cores)
        rows = min(spec.query_tile, max((q for q, _ in spec.requests), default=0))
        partial = plan["fd_splits"] * rows * spec.head_dim * _FP32_BYTES
        lse = plan["fd_splits"] * rows * 2 * _FP32_BYTES
        report.update({
            "owner_s2_loads": plan["loads"],
            "owner_s2_total": plan["total"],
            "fd_splits": plan["fd_splits"],
            "fd_fragments": plan["fd_fragments"],
            "fd_partial_write_bytes": partial,
            "fd_partial_read_bytes": partial,
            "fd_lse_bytes": lse,
            "fd_reduce_ops": plan["fd_splits"] * rows * (spec.head_dim + 2),
            "fd_service_basis": "source_owner_split_diagnostic",
        })
        if resource_roles is not None:
            roles = set(resource_roles)
            report["unmodeled_transfer_roles"] = [
                role for role, present in (
                    ("matrix_input_copy", bool(roles & {"mte2", "mte2_aic"})),
                    ("matrix_operand_copy", "mte1" in roles),
                    ("output_store", "mte3" in roles),
                ) if not present]
        return report

    def owner_plan(self, op: OpSpec, cores: int = 32) -> dict:
        """Source S2-tile ownership and FD split count for this geometry."""
        return owner_s2_plan(self.spec(op), cores)

    def build(self, op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
        spec = self.spec(op)
        plan = None
        cores = 32
        if geometry is None:
            read, vread, write, operand = "mte2", "mte2", "mte3", "mte1"
            mte1 = False
        else:
            read = read_role(geometry.resource_units)
            vread = read_role(geometry.resource_units, aiv=0)
            write = "mte3" if "mte3" in geometry.resource_units else None
            operand = "mte1" if "mte1" in geometry.resource_units else None
            mte1 = operand is not None
            cores = max(1, int(geometry.compute_units))
            if spec.kv_cache.needs_conversion and operand is None:
                raise ModelingError(
                    "attention_prefill_conversion_requires_mte1_operand_copy")
            if spec.kv_cache.needs_conversion and write is None:
                raise ModelingError(
                    "attention_prefill_conversion_requires_mte3_result_copy")
        if spec.native_fia and geometry is not None:
            from .attention_prefill_nofd import owner_s2_plan
            plan = owner_s2_plan(spec, cores)
        work = _total_work(
            spec, include_mte1=mte1, read=read, vread=vread,
            write=write, operand=operand, plan=plan)
        metadata = {"kv_cache_access": spec.kv_cache.report()}
        if plan is not None:
            metadata["prefill_plan"] = {
                "source": "owner_s2_plan",
                "cores": cores,
                "total": plan["total"],
                "loads": tuple(plan["loads"]),
                "fd_splits": plan["fd_splits"],
                "fd_fragments": 2 * plan["fd_fragments"],
                "split_tasks": plan["fd_fragments"],
                "issue_kind": spec.issue_kind,
            }
        return OperatorPlan(
            work, lambda context: _PrefillProgram(context, self, spec),
            metadata=metadata)

    def lower(self, context: RecipeContext) -> KernelProgram:
        spec = self.spec(context.op)
        return KernelProgram(
            (
                Stage("setup", latency=context.setup_ms),
                Stage("body", after=("setup",),
                      program=_PrefillProgram(context, self, spec)),
            ),
            minimum_ms=context.minimum_ms,
        )


@dataclass(frozen=True)
class _Task:
    request: int
    query_head: int
    kv_head: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    horizon: int
    key_tiles: int
    key_start: int = 0
    heads: int = 1

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True)
class _Spec:
    requests: tuple[tuple[int, int], ...]
    query_heads: int
    kv_heads: int
    head_dim: int
    query_dtype: str
    kv_dtype: str
    query_tile: int
    key_tile: int
    slots: int
    s2_preload: int
    softmax_slots: int
    softmax_outer: int
    native_fia: bool
    reconstruct: bool | None
    issue_kind: str | None
    request_ends: tuple[int, ...]
    request_tile_counts: tuple[int, ...]
    kv_cache: KVCacheAccess = KVCacheAccess()

    @property
    def group(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def query_bytes(self) -> int:
        return int(dtype_bytes(self.query_dtype))

    @property
    def kv_bytes(self) -> int:
        return int(dtype_bytes(self.kv_dtype))

    @property
    def task_count(self) -> int:
        return self.request_ends[-1] if self.request_ends else 0


def _spec(op: OpSpec, recipe: StagedGQAPrefill) -> _Spec:
    if getattr(op, "phase", None) not in _PREFILL_PHASES:
        raise ModelingError("attention_prefill_only_supports_prefill_phases")
    metadata = dict(getattr(op, "metadata", ()))
    raw = metadata.get("gqa_prefill")
    if not isinstance(raw, Mapping):
        raise ModelingError("attention_prefill_gqa_prefill_metadata_required")
    raw_requests = raw.get("requests")
    if not isinstance(raw_requests, (list, tuple)):
        raise ModelingError("attention_prefill_requests_must_be_sequence")
    requests = []
    for item in raw_requests:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ModelingError("attention_prefill_request_must_be_query_context_pair")
        requests.append((_nonnegative_int(item[0], "query_len"),
                         _nonnegative_int(item[1], "context")))
    requests = tuple(requests)
    heads = _positive_int(raw.get("query_heads"), "query_heads")
    kv_heads = _positive_int(raw.get("kv_heads"), "kv_heads")
    dim = _positive_int(raw.get("head_dim"), "head_dim")
    if heads % kv_heads:
        raise ModelingError("attention_prefill_query_heads_not_divisible_by_kv_heads")
    query_dtype = _dtype(raw.get("query_dtype"), "query_dtype")
    kv_dtype = _dtype(raw.get("kv_dtype"), "kv_dtype")
    if raw.get("causal") is not True:
        raise ModelingError("attention_prefill_causal_true_required")
    if raw.get("window", 0) != 0:
        raise ModelingError("attention_prefill_windowed_prefill_not_selected")
    if raw.get("cp", 1) != 1:
        raise ModelingError("attention_prefill_context_parallel_not_selected")
    # The ordinary FIA descriptor binds the actual selected primitive
    # (FusedInferAttentionScore).  Generic GQA prefill callers retain the
    # CVDiff query-tile rule; the selected A5 FIA V4 branch uses the arch35
    # cube S1 basic block (SOuter64*CV_RATIO).
    native_fia = any(
        str(key) == "native_operator" and str(value) == "FusedInferAttentionScore"
        for key, value in metadata.items())
    query_tile = (recipe.query_tile if recipe.query_tile is not None
                  else (arch35_query_tile(dim) if native_fia
                        else source_query_tile(dim)))

    ends, tile_counts, total = [], [], 0
    for query, _context in requests:
        tiles = _ceil_div(query, query_tile)
        tile_counts.append(tiles)
        total += heads * tiles
        ends.append(total)
    softmax_outer = (source_softmax_outer(dim) if native_fia else 8)
    # Preload 3 stays on the unselected path.  issue_kind is set only from an
    # explicit reconstruct flag and is never inferred from the head dimension.
    s2_preload = NATIVE_S2_PRELOAD if native_fia else recipe.slots
    reconstruct = _explicit_reconstruct(metadata)
    issue_kind = source_issue_kind(reconstruct, dim) if native_fia else None
    return _Spec(requests, heads, kv_heads, dim, query_dtype, kv_dtype,
                 query_tile, recipe.key_tile, recipe.slots, s2_preload,
                 recipe.softmax_slots, softmax_outer, native_fia,
                 reconstruct, issue_kind, tuple(ends), tuple(tile_counts),
                 kv_access(op, query_dtype=query_dtype, kv_dtype=kv_dtype,
                           head_dim=dim))


def _task_at(spec: _Spec, index: int) -> _Task:
    """Resolve one global cyclic task index without materializing task rows."""
    if index < 0 or index >= spec.task_count:
        raise IndexError("attention_prefill_task_index_out_of_range")
    request = bisect_right(spec.request_ends, index)
    base = spec.request_ends[request - 1] if request else 0
    local = index - base
    tiles = spec.request_tile_counts[request]
    query_head, tile = divmod(local, tiles)
    query, context = spec.requests[request]
    token_start = tile * spec.query_tile
    token_end = min(query, token_start + spec.query_tile)
    horizon = context + token_end
    return _Task(request, query_head, query_head // spec.group,
                 token_start, token_end, token_start, token_end, horizon,
                 _ceil_div(horizon, spec.key_tile), 0, 1)


def _iter_tasks(spec: _Spec):
    if spec.native_fia:
        for _index, task in _native_tasks(spec):
            yield task
        return
    for index in range(spec.task_count):
        yield _task_at(spec, index)


def _native_tasks(spec: _Spec):
    """Native TND S1G M tiles: time x G flattened into 128-row tiles.

    A tile covers ``128/G`` query tokens and all G query heads that share one
    KV head.  K/V is loaded once for the tile, not once per query head.
    """
    index = 0
    width = spec.group
    for request, (query, context) in enumerate(spec.requests):
        total = query * width
        for kv_head in range(spec.kv_heads):
            for m in range(_ceil_div(total, 128)):
                row_start = m * 128
                row_end = min(total, row_start + 128)
                token_start = row_start // width
                token_end = _ceil_div(row_end, width)
                horizon = context + token_end
                yield index, _Task(
                    request, kv_head * width, kv_head, row_start, row_end,
                    token_start, token_end, horizon,
                    _ceil_div(horizon, spec.key_tile), 0, width)
                index += 1


def _task_at_for(spec: _Spec, index: int) -> _Task:
    if spec.native_fia:
        for position, (_idx, task) in enumerate(_native_tasks(spec)):
            if position == index:
                return task
        raise IndexError("attention_prefill_native_task_index_out_of_range")
    return _task_at(spec, index)


def _time_major_tasks(spec: _Spec):
    """S1G flattening order: outer query tile, inner query head.

    The registered TND GQA tiler walks ``[sequence, group]`` time-major, not
    the head-major order used by the cyclic executor.  Owner costs are S2
    tiles in this order.
    """
    if spec.native_fia:
        for index, task in _native_tasks(spec):
            yield index, task
        return
    for request, (query, _context) in enumerate(spec.requests):
        base = spec.request_ends[request - 1] if request else 0
        tiles = spec.request_tile_counts[request]
        for tile in range(tiles):
            for head in range(spec.query_heads):
                index = base + head * tiles + tile
                yield index, _task_at(spec, index)


def _task_pairs(task: _Task, context: int) -> tuple[int, int]:
    """Return (logical valid pairs, physical full-rectangle pairs)."""
    heads = max(1, task.heads)
    rows = task.rows // heads
    logical = heads * logical_causal_pairs(rows, context, task.token_start)
    return logical, task.rows * task.horizon


def _task_work(spec: _Spec, task: _Task, *, include_mte1: bool, partial: bool,
               read: str, vread: str, write: str | None,
                operand: str | None):
    cache = spec.kv_cache
    _query, context = spec.requests[task.request]
    rows, dim = task.rows, spec.head_dim
    q_bytes = rows * dim * spec.query_bytes
    if partial:
        out_bytes = rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES)
        vector_out = rows * (dim + 2)
    else:
        out_bytes = rows * dim * spec.query_bytes
        vector_out = rows * (dim + 1)
    load_start = task.key_start * spec.key_tile
    load_end = min((task.key_start + task.key_tiles) * spec.key_tile,
                   task.horizon)
    kv_rows = max(0, load_end - load_start)
    key_payload = kv_rows * cache.key_payload_row
    value_payload = kv_rows * cache.value_payload_row
    page_meta = page_scale = 0
    if cache.paged:
        pages = cache.pages(load_start, load_end)
        page_meta = pages * cache.page_meta_bytes
        if cache.scale_scope == "page":
            page_scale = pages * cache.page_scale_bytes
    row_scale = (cache.row_scale_bytes(kv_rows)
                 if cache.scale_scope == "row" else 0)
    producer = (q_bytes + page_meta
                + (0 if cache.key_needs_conversion else key_payload)
                + (0 if cache.value_needs_conversion else value_payload))
    vector_reads = (page_scale + row_scale
                    + (key_payload if cache.key_needs_conversion else 0)
                    + (value_payload if cache.value_needs_conversion else 0))
    mte3_extra = 0
    if cache.key_needs_conversion:
        mte3_extra += kv_rows * cache.key_operand_row
    if cache.value_needs_conversion:
        mte3_extra += kv_rows * cache.value_operand_row
    mte1 = 0
    if include_mte1:
        mte1 += q_bytes * task.key_tiles
        mte1 += (kv_rows * cache.key_operand_row
                 if cache.key_needs_conversion else key_payload)
        mte1 += (kv_rows * cache.value_operand_row
                 if cache.value_needs_conversion else value_payload)
    vector = vector_out + cache.conversion_ops(kv_rows)
    flops = 0
    for tile in range(task.key_tiles):
        valid = min(spec.key_tile, task.horizon
                    - (task.key_start + tile) * spec.key_tile)
        rect = rows * valid
        if include_mte1:
            mte1 += rect * spec.query_bytes
        flops += 4 * dim * rect
        vector += 7 * rect + rows * (3 * dim + 10)
    return {
        "producer": producer, "vector_reads": vector_reads,
        "writes": out_bytes, "mte3_extra": mte3_extra,
        "vector": vector, "mte1": mte1, "flops": flops,
    }


def _total_work(spec: _Spec, *, include_mte1: bool, read: str = "mte2",
                vread: str | None = None, write: str | None = "mte3",
                operand: str | None = "mte1", plan=None) -> Work:
    cache = spec.kv_cache
    vread = vread or read
    producer = vector_reads = writes = mte3_extra = vector = mte1 = 0
    flops = 0
    if plan is None:
        for task in _iter_tasks(spec):
            part = _task_work(
                spec, task, include_mte1=include_mte1, partial=False,
                read=read, vread=vread, write=write, operand=operand)
            producer += part["producer"]
            vector_reads += part["vector_reads"]
            writes += part["writes"]
            mte3_extra += part["mte3_extra"]
            vector += part["vector"]
            mte1 += part["mte1"]
            flops += part["flops"]
    else:
        task_owners = {}
        for core, ranges in enumerate(plan["ranges"]):
            for ti, key_start, key_end in ranges:
                task_owners.setdefault(ti, []).append(core)
                base = _task_at_for(spec, ti)
                item = replace(base, key_start=key_start,
                               key_tiles=key_end - key_start)
                part = _task_work(
                    spec, item, include_mte1=include_mte1,
                    partial=key_start > 0 or key_end < base.key_tiles,
                    read=read, vread=vread, write=write, operand=operand)
                producer += part["producer"]
                vector_reads += part["vector_reads"]
                writes += part["writes"]
                mte3_extra += part["mte3_extra"]
                vector += part["vector"]
                mte1 += part["mte1"]
                flops += part["flops"]
        for ti, owners in sorted(task_owners.items()):
            if len(owners) < 2:
                continue
            base = _task_at_for(spec, ti)
            rows, dim = base.rows, spec.head_dim
            fragments = len(owners)
            frag_rows = _ceil_div(rows, 2)
            for fragment in range(2):
                this_rows = min(frag_rows, rows - fragment * frag_rows)
                if this_rows <= 0:
                    continue
                part = fragments * this_rows * (dim * _FP32_BYTES
                                                + 2 * _FP32_BYTES)
                out = this_rows * dim * spec.query_bytes
                vector_reads += part
                writes += out
                vector += fragments * this_rows * (dim + 2)
    resources = {"hbm": producer + vector_reads + writes, "vector": vector}
    if read:
        resources[read] = resources.get(read, 0) + producer
    if vread:
        resources[vread] = resources.get(vread, 0) + vector_reads
    if write and (writes or mte3_extra):
        resources[write] = (resources.get(write, 0) + writes + mte3_extra)
    if include_mte1 and operand:
        resources[operand] = mte1
    return Work(matrix={spec.query_dtype: flops}, resources=resources)

def _report(spec: _Spec) -> dict:
    tasks = logical = physical = kv_read = q_read = out_write = 0
    kv_payload = kv_scale = page_table = 0
    q_l1_l0a = 0
    key_loads = 0
    cache = spec.kv_cache
    for task in _iter_tasks(spec):
        _query, context = spec.requests[task.request]
        tasks += 1
        task_logical, task_physical = _task_pairs(task, context)
        logical += task_logical
        physical += task_physical
        key_loads += task.key_tiles
        task_q = task.rows * spec.head_dim * spec.query_bytes
        q_read += task_q
        q_l1_l0a += task_q * task.key_tiles
        out_write += task.rows * spec.head_dim * spec.query_bytes
        payload = cache.payload_bytes(task.horizon)
        kv_payload += payload
        if cache.scale_scope == "row":
            kv_scale += cache.row_scale_bytes(task.horizon)
        if cache.paged:
            page_start = task.key_start * spec.key_tile
            page_end = min((task.key_start + task.key_tiles) * spec.key_tile,
                           task.horizon)
            pages = cache.pages(page_start, page_end)
            page_table += pages * cache.page_meta_bytes
            if cache.scale_scope == "page":
                kv_scale += pages * cache.page_scale_bytes
        kv_read += payload
    if spec.native_fia:
        # M tiles cut [time, group] and tails can split a head row; logical
        # causal pairs are still per original query head and KV group.
        logical = sum(spec.query_heads * logical_causal_pairs(query, context)
                      for query, context in spec.requests)
    return {
        "implementation": IMPLEMENTATION,
        "tasks": tasks,
        "key_tile_loads": key_loads,
        "logical_pairs": logical,
        "executed_rectangle_pairs": physical,
        "causal_masked_pairs": physical - logical,
        "q_bytes": q_read,
        "q_l1_l0a_bytes": q_l1_l0a,
        "kv_bytes": kv_payload + kv_scale + page_table,
        "kv_payload_bytes": kv_payload,
        "kv_scale_bytes": kv_scale,
        "page_table_bytes": page_table,
        "output_bytes": out_write,
        "query_tile": spec.query_tile,
        "key_tile": spec.key_tile,
        "slots": spec.slots,
        "s2_preload": spec.s2_preload,
        "reconstruct": spec.reconstruct,
        "issue_kind": spec.issue_kind,
        "source_state": "pinned_source_mechanism_not_captured_dispatch",
    }


def build(op: OpSpec, geometry: Geometry | None = None) -> OperatorPlan:
    """Independent verification binding for the default recipe options."""
    return StagedGQAPrefill().build(op, geometry)


# ---------------------------------------------------------------------------
# Placement, buffers and executable programs.
# ---------------------------------------------------------------------------


def _placement(context: RecipeContext, core: int, *roles: str) -> dict:
    present = tuple(role for role in roles if role in context.resource_names)
    return context.owned(core, context.core_count, present) if present else {}


def _read_role(context: RecipeContext, *, vector: bool = False) -> str:
    return read_role(context.resource_names, aiv=0 if vector else None)


def _write_role(context: RecipeContext) -> str | None:
    return "mte3" if "mte3" in context.resource_names else None


def _operand_role(context: RecipeContext) -> str | None:
    return "mte1" if "mte1" in context.resource_names else None


def _allocation(context: RecipeContext, core: int, pool: str, name: str,
                size: float) -> Allocation | None:
    if pool not in context.pool_names or size <= 0:
        return None
    capacity = context.pool_capacities.get(pool)
    if capacity is not None and size > capacity:
        raise ModelingError(
            f"attention_prefill_buffer_exceeds_pool:{pool}:{name}:{size}>{capacity}")
    units = unit_group(core, context.core_count,
                       context.pool_units.get(pool, 1))
    return Allocation(name, context.pool_names[pool], float(size), units)


def _allocations(items) -> tuple[Allocation, ...]:
    return tuple(item for item in items if item is not None)


def _task_program(context: RecipeContext, recipe: StagedGQAPrefill,
                  spec: _Spec, task: _Task, core: int,
                  *, partial: bool = False) -> KernelProgram:
    dim, rows = spec.head_dim, task.rows
    q_bytes = rows * dim * spec.query_bytes
    read = _read_role(context)
    write = _write_role(context)
    operand = _operand_role(context)
    cache = spec.kv_cache
    if cache.needs_conversion and not operand:
        raise ModelingError(
            "attention_prefill_conversion_requires_mte1_operand_copy")
    copy_in = (read,)
    vector_copy_in = (_read_role(context, vector=True),)
    copy_out = (write,) if write else ()
    operand_roles = (operand,) if operand else ()
    cube, vector = ("cube",), ("vector",)

    q_l1 = _allocation(context, core, "l1", "q_l1", q_bytes)
    # Arch35 streams Q from L1 through the two shared 32KiB L0A microtile
    # buffers; full Q is not resident in L0A.
    q_l0a = _allocation(context, core, "l0a", "q_l0a", min(q_bytes, 32 * 1024))
    ub_state = _allocation(context, core, "ub", "online_state",
                           rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES))
    persistent = _allocations((q_l1, q_l0a, ub_state))
    # Q remains resident while PV consumes a probability operand.  A larger
    # declared tile cannot make progress if these two mandatory buffers do not
    # fit together, even when each allocation individually fits.
    outer = min(rows, spec.softmax_outer)
    softmax_window = min(recipe.softmax_slots, task.key_tiles)
    l0a_required = 64 * 1024  # two shared 32 KiB arch35 L0A microtile buffers
    preload = spec.s2_preload
    ub_required = rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES) \
        + softmax_window * outer * min(recipe.key_tile, task.horizon) * _FP32_BYTES
    max_rows = min(recipe.key_tile, task.horizon)
    if cache.needs_conversion:
        # conversion needs packed input + converted output + scales in UB,
        # and the converted operand in L1 before the MTE1 L1->L0B copy.
        slots = max(1, min(recipe.slots, spec.s2_preload, task.key_tiles))
        per_slot = max_rows * (
            cache.key_payload_row + cache.value_payload_row
            + cache.key_scale_row + cache.value_scale_row)
        ub_conv = max(
            max_rows * (cache.key_payload_row + cache.key_operand_row
                        + cache.key_scale_row),
            max_rows * (cache.value_payload_row + cache.value_operand_row
                        + cache.value_scale_row))
        ub_required = (rows * (dim * _FP32_BYTES + 2 * _FP32_BYTES)
                       + softmax_window * outer * max_rows * _FP32_BYTES
                       + slots * per_slot + ub_conv)
        operand_row = max(cache.key_operand_row, cache.value_operand_row)
        l1_required = q_bytes + slots * max_rows * operand_row
        l0b_required = slots * max_rows * operand_row
    else:
        l1_required = q_bytes + max_rows * (
            cache.key_payload_row + cache.value_payload_row)
        l0b_required = 0
    required_pools = [("l0a", l0a_required), ("ub", ub_required),
                      ("l1", l1_required)]
    if cache.needs_conversion:
        required_pools.append(("l0b", l0b_required))
    for pool, required in required_pools:
        capacity = context.pool_capacities.get(pool)
        if pool in context.pool_names and capacity is not None and required > capacity:
            raise ModelingError(
                f"attention_prefill_resident_working_set_exceeds_pool:{pool}:{required}>{capacity}")
    q_work = {"hbm": q_bytes, read: q_bytes}
    stages = [
        Stage("q_load", context.bind(Work(resources=q_work)),
              acquire=persistent,
              placement={**_placement(context, core, *copy_in),
                         **_placement(context, core, *operand_roles)}),
    ]
    page_meta = page_scale = 0
    page_scale_ub = None
    if cache.paged:
        page_start = task.key_start * recipe.key_tile
        page_end = min((task.key_start + task.key_tiles) * recipe.key_tile,
                       task.horizon)
        pages = cache.pages(page_start, page_end)
        page_meta = pages * cache.page_meta_bytes
        if cache.scale_scope == "page":
            page_scale = pages * cache.page_scale_bytes
    if page_meta or page_scale:
        page_work = {"hbm": page_meta + page_scale}
        if page_meta:
            page_work[read] = page_work.get(read, 0) + page_meta
        if page_scale:
            vector_read = _read_role(context, vector=True)
            page_work[vector_read] = page_work.get(vector_read, 0) + page_scale
            page_scale_ub = _allocation(
                context, core, "ub", "page_scale_ub", page_scale)
        stages.append(Stage(
            "page_access", context.bind(Work(resources=page_work)),
            acquire=_allocations((page_scale_ub,)), after=("q_load",),
            placement={**_placement(context, core, *copy_in),
                       **_placement(context, core, *vector_copy_in)}))
    key_ready = "page_access" if (page_meta or page_scale) else "q_load"
    max_tile = min(recipe.key_tile, max(1, task.horizon))

    def tile(index: int):
        start = (task.key_start + index) * recipe.key_tile
        valid = min(recipe.key_tile, task.horizon - start)
        return tile_template(valid, index % preload)

    # All full key panels of an owner use the same work and buffer slots.
    # Only the final partial panel differs. Reuse bound stage descriptions;
    # PipelineProgram still creates independent events and lifetimes per panel.
    @lru_cache(maxsize=preload + 1)
    def tile_template(valid: int, slot: int):
        rect = rows * valid
        key_needs = cache.key_needs_conversion
        value_needs = cache.value_needs_conversion
        key_payload = valid * cache.key_payload_row
        value_payload = valid * cache.value_payload_row
        key_operand = valid * cache.key_operand_row
        value_operand = valid * cache.value_operand_row
        key_scale = (valid * cache.key_scale_row
                     if cache.scale_scope == "row" else 0)
        value_scale = (valid * cache.value_scale_row
                       if cache.scale_scope == "row" else 0)
        suffix = f"s{slot}"
        k_l1 = (_allocation(context, core, "l1", f"k_l1_{suffix}", key_payload)
                if not key_needs else None)
        k_l0b = (_allocation(
            context, core, "l0b", f"k_l0b_{suffix}",
            min(key_payload, 32 * 1024)) if not key_needs else None)
        # CVDiff softmax/PV cuts the probability panel to
        # softmaxSOuterFactor rows; it does not materialize the full Souter
        # row set in L0A/UB at once.
        scores = _allocation(context, core, "l0c", f"scores_{suffix}",
                             outer * max_tile * _FP32_BYTES)
        probs = _allocation(context, core, "ub", f"probs_{suffix}",
                            outer * max_tile * _FP32_BYTES)
        p_l0a = _allocation(context, core, "l0a", f"p_l0a_{suffix}",
                           min(outer * valid * spec.query_bytes, 32 * 1024))
        v_l1 = (_allocation(context, core, "l1", f"v_l1_{suffix}", value_payload)
                if not value_needs else None)
        v_l0b = (_allocation(
            context, core, "l0b", f"v_l0b_{suffix}",
            min(value_payload, 32 * 1024)) if not value_needs else None)
        pv_acc = _allocation(context, core, "l0c", f"pv_acc_{suffix}",
                             rows * dim * _FP32_BYTES)
        score_acquire = _allocations((scores,))
        prob_acquire = _allocations((probs,))
        pv_acquire = _allocations((pv_acc,))
        score_release = _allocations((scores,))
        prob_release = _allocations((probs,))

        qk = Work(matrix={spec.query_dtype: 2 * dim * rect},
                  resources=({"mte1": q_bytes} if operand else {}))
        softmax = Work(resources={"vector": 7 * rect + rows})
        pv = Work(matrix={spec.query_dtype: 2 * dim * rect})
        online = Work(resources={"vector": rows * (3 * dim + 9)})
        dequant = cache.dequant_ops_per_element
        emitted = []

        # K: packed payload -> UB, vector conversion -> UB, MTE3 -> L1,
        # MTE1 -> L0B, QK.  Non-converting sides keep the direct path.
        key_needs = cache.key_needs_conversion
        value_needs = cache.value_needs_conversion
        if key_needs:
            k_payload_ub = _allocation(
                context, core, "ub", f"k_payload_{suffix}", key_payload)
            k_conv_ub = _allocation(
                context, core, "ub", f"k_conv_{suffix}", key_operand)
            k_post_l1 = _allocation(
                context, core, "l1", f"k_post_{suffix}", key_operand)
            k_l0b = _allocation(
                context, core, "l0b", f"k_l0b_{suffix}", key_operand)
            k_scale_ub = (_allocation(
                context, core, "ub", f"k_scale_{suffix}", key_scale)
                if key_scale else None)
            emitted.append(Stage(
                "k_load", context.bind(Work(resources={
                    "hbm": key_payload,
                    _read_role(context, vector=True): key_payload})),
                acquire=_allocations((k_payload_ub,)),
                placement=_placement(context, core, *vector_copy_in)))
            after = ("k_load",)
            if key_scale:
                emitted.append(Stage(
                    "k_scale_load", context.bind(Work(resources={
                        "hbm": key_scale,
                        _read_role(context, vector=True): key_scale})),
                    after=("k_load",), acquire=_allocations((k_scale_ub,)),
                    placement=_placement(context, core, *vector_copy_in)))
                after = ("k_scale_load",)
            emitted.append(Stage(
                "dequant_k", context.bind(Work(resources={
                    "vector": cache.key_conversion_ops(valid)})),
                after=after, acquire=_allocations((k_conv_ub,)),
                free=_allocations((k_payload_ub, k_scale_ub)),
                placement=_placement(context, core, *vector)))
            emitted.append(Stage(
                "conv_store_k", context.bind(Work(resources={write: key_operand})),
                after=("dequant_k",), acquire=_allocations((k_post_l1,)),
                free=_allocations((k_conv_ub,)),
                placement=_placement(context, core, *copy_out)))
            emitted.append(Stage(
                "operand_k", context.bind(Work(resources={operand: key_operand})),
                after=("conv_store_k",), acquire=_allocations((k_l0b,)),
                free=_allocations((k_post_l1,)),
                placement=_placement(context, core, *operand_roles)))
            k_ready = "operand_k"
            k_free = _allocations((k_l0b,))
        else:
            k_load = {"hbm": key_payload, read: key_payload}
            if operand:
                k_load["mte1"] = key_payload
            emitted.append(Stage(
                "k_load", context.bind(Work(resources=k_load)),
                acquire=_allocations((k_l1, k_l0b)),
                placement={**_placement(context, core, *copy_in),
                           **_placement(context, core, *operand_roles)}))
            k_ready = "k_load"
            k_free = _allocations((k_l1, k_l0b))
        emitted.append(Stage(
            "qk", context.bind(qk), after=(k_ready,),
            acquire=score_acquire, free=k_free,
            placement={**_placement(context, core, *cube),
                       **_placement(context, core, *operand_roles)}))
        emitted.append(Stage(
            "softmax", context.bind(softmax), after=("qk",),
            acquire=prob_acquire, free=score_release,
            placement=_placement(context, core, *vector)))

        # V mirrors K; PV consumes the materialized V operand and probability.
        if value_needs:
            v_payload_ub = _allocation(
                context, core, "ub", f"v_payload_{suffix}", value_payload)
            v_conv_ub = _allocation(
                context, core, "ub", f"v_conv_{suffix}", value_operand)
            v_post_l1 = _allocation(
                context, core, "l1", f"v_post_{suffix}", value_operand)
            v_l0b = _allocation(
                context, core, "l0b", f"v_l0b_{suffix}", value_operand)
            v_scale_ub = (_allocation(
                context, core, "ub", f"v_scale_{suffix}", value_scale)
                if value_scale else None)
            emitted.append(Stage(
                "v_load", context.bind(Work(resources={
                    "hbm": value_payload,
                    _read_role(context, vector=True): value_payload})),
                after=("softmax",), acquire=_allocations((v_payload_ub,)),
                placement=_placement(context, core, *vector_copy_in)))
            after = ("v_load",)
            if value_scale:
                emitted.append(Stage(
                    "v_scale_load", context.bind(Work(resources={
                        "hbm": value_scale,
                        _read_role(context, vector=True): value_scale})),
                    after=("v_load",), acquire=_allocations((v_scale_ub,)),
                    placement=_placement(context, core, *vector_copy_in)))
                after = ("v_scale_load",)
            emitted.append(Stage(
                "dequant_v", context.bind(Work(resources={
                    "vector": cache.value_conversion_ops(valid)})),
                after=after, acquire=_allocations((v_conv_ub,)),
                free=_allocations((v_payload_ub, v_scale_ub)),
                placement=_placement(context, core, *vector)))
            emitted.append(Stage(
                "conv_store_v", context.bind(Work(resources={write: value_operand})),
                after=("dequant_v",), acquire=_allocations((v_post_l1,)),
                free=_allocations((v_conv_ub,)),
                placement=_placement(context, core, *copy_out)))
            emitted.append(Stage(
                "operand_v", context.bind(Work(resources={operand: value_operand})),
                after=("conv_store_v",), acquire=_allocations((v_l0b,)),
                free=_allocations((v_post_l1,)),
                placement=_placement(context, core, *operand_roles)))
            v_ready = "operand_v"
            v_free = _allocations((v_l0b,))
        else:
            v_load = {"hbm": value_payload, read: value_payload}
            if operand:
                v_load["mte1"] = value_payload
            emitted.append(Stage(
                "v_load", context.bind(Work(resources=v_load)),
                after=("softmax",), acquire=_allocations((v_l1, v_l0b)),
                placement={**_placement(context, core, *copy_in),
                           **_placement(context, core, *operand_roles)}))
            v_ready = "v_load"
            v_free = _allocations((v_l1, v_l0b))
        emitted.extend((
            Stage("probability_operand",
                  context.bind(Work(resources=(
                      {"mte1": rect * spec.query_bytes} if operand else {}))),
                  after=("softmax",), acquire=_allocations((p_l0a,)),
                  placement=_placement(context, core, *operand_roles)),
            Stage("pv", context.bind(pv),
                  after=(v_ready, "probability_operand"),
                  acquire=pv_acquire,
                  free=_allocations((v_l0b, p_l0a)) + tuple(pv_acquire)
                  + (_allocations((v_l1,)) if not value_needs else ()),
                  placement=_placement(context, core, *cube)),
            Stage("online_update", context.bind(online), after=("pv",),
                  free=prob_release, previous=("online_update",),
                  placement=_placement(context, core, *vector)),
        ))
        return tuple(emitted)

    use_compact = len(set(spec.requests)) <= 1
    # Simplified init is the first three taskIds and the drain is three extra
    # gS1 steps on the last BN.  GQA withholds C2 until PRELOAD_N-1.  Three
    # covers both without selecting a kernel.
    fill = NATIVE_S2_PRELOAD if spec.native_fia else 1
    stages.append(Stage(
        "key_tiles", after=(key_ready,),
        program=(
            compact_periodic_pipeline(
                context, task.key_tiles, preload, tile,
                head=fill, tail=fill, resident=persistent)
            if use_compact
            else PipelineProgram(task.key_tiles, preload, tile)
        ),
    ))
    if page_scale_ub is not None:
        stages.append(Stage("page_scale_release", after=("key_tiles",),
                            free=_allocations((page_scale_ub,))))
    out_bytes = rows * dim * spec.query_bytes
    if partial:
        # A split S2 owner publishes an FP32 accumulator plus LSE partial;
        # the FD merge below consumes both and writes the final BF16 output.
        part = rows * dim * _FP32_BYTES
        lse = rows * 2 * _FP32_BYTES
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": part + lse,
                **(({write: part + lse}) if write else {}),
                "vector": rows * (dim + 2),
            })),
            free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    else:
        stages.append(Stage(
            "final_store", after=("key_tiles",),
            work=context.bind(Work(resources={
                "hbm": out_bytes,
                **(({write: out_bytes}) if write else {}),
                "vector": rows * (dim + 1),
            })),
            free=persistent,
            placement={**_placement(context, core, *copy_out),
                       **_placement(context, core, *vector)},
        ))
    return KernelProgram(tuple(stages))


def planned_prefill_recipe(**options) -> PlannedRecipe:
    """Preserve all prefill recipe options for the T10 factory."""
    return PlannedRecipe(StagedGQAPrefill(**options).build)


__all__ = [
    "IMPLEMENTATION",
    "RECIPE_ID",
    "SOURCE_REFERENCES",
    "StagedGQAPrefill",
    "build",
    "logical_causal_pairs",
    "planned_prefill_recipe",
]
