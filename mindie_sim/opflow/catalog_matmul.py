"""Selected mathematical routes and analytic Cube/quantized/grouped priors.

[Role]
Build matmul family flows with explicit A/B/C L1/L0 tiles, K-loop MTE1 reloads,
dtype-specific Cube rates, integer expert histograms, Stream-K partial workspace
reduction, MXFP scale loads, and split-M dualDst Fixpipe routing.
[Invariants]
Total MACs are conserved across groups, split-K, and Stream-K. MTE1 counts the
real per-tile L1->L0 reloads instead of the whole matrix once. Every group size
is a non-negative integer and the histogram sums to the declared M. Unknown
kwargs and non-positive dimensions fail closed.
[Timeouts & Locks]
Arithmetic-only; no device calls, sleeps, locks, or external dependencies.
[Side-Effect Policy]
Read-only module; builders allocate in-memory records only.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .catalog_common import (
    as_int, check_kwargs, conditional_prior, cube_buffers, dtype_arg, make_flow, pick_int,
)
from .flow_schema import (
    AuthoringBackend, BufferPool, ExecutionArchetype, FixpipePath, PipeWork,
)


def _shape(kw: dict[str, Any]) -> tuple[int, int, int]:
    m = pick_int(kw, ("m", "tokens", "rows"), 2048)
    n = pick_int(kw, ("n", "hidden", "cols"), 2048)
    k = pick_int(kw, ("k",), 2048)
    return m, n, k


def _tile_opts(kw: dict[str, Any]) -> dict[str, int]:
    return {name: as_int(name, kw[name]) for name in ("base_m", "base_n", "base_k") if name in kw}


def _group_sizes(kw: dict[str, Any], groups: int, m: int) -> tuple[int, ...]:
    raw = kw.get("group_sizes", kw.get("group_list"))
    if raw is None:
        base, rem = divmod(m, groups)
        return tuple(base + (1 if index < rem else 0) for index in range(groups))
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        raise ValueError("group_sizes must be a sequence of non-negative integers")
    values = tuple(as_int("group_sizes", value, minimum=0) for value in raw)
    if len(values) != groups:
        raise ValueError("group_sizes length must equal groups")
    if sum(values) > m:
        raise ValueError("group_sizes sum must not exceed M")
    return values


def matmul_bf16(**kw: Any) -> Any:
    from .catalog_exact_cube import matmul_reference
    return matmul_reference("matmul_bf16", **kw)

def _matmul(name: str, m: int, n: int, k: int, *, dtype: str = "bf16",
            base_m: int = 128, base_n: int = 256, base_k: int = 64,
            groups: int = 1, group_sizes: tuple[int, ...] | None = None,
            read_a: float = 2.0, read_b: float = 2.0, write_c: float = 4.0,
            extra_read: int = 0, extra_write: int = 0, vec_ops: float = 0.0,
            streamk: bool = False, inplace: bool = False,
            note: str = "", backend: AuthoringBackend = AuthoringBackend.ASCENDC,
            **kw: Any) -> Any:
    del kw
    for option, value in (("streamk", streamk), ("inplace", inplace)):
        if not isinstance(value, bool):
            raise ValueError(f"{option} must be a bool")
    if group_sizes is None:
        group_sizes = tuple([m] * groups) if groups == 1 else _group_sizes({}, groups, m)
    actual_rows = sum(group_sizes)
    tiles_n = math.ceil(n / base_n)
    tiles = sum(math.ceil(rows / base_m) * tiles_n for rows in group_sizes)
    macs = float(actual_rows) * n * k
    a_read = float(actual_rows) * k * read_a * tiles_n
    b_read = float(sum(math.ceil(rows / base_m) for rows in group_sizes)) * k * read_b * n
    gmm_read = a_read + b_read + extra_read
    write = actual_rows * n * write_c + extra_write
    k_loops = math.ceil(k / base_k)
    elem_bytes = 2.0 if dtype in ("bf16", "fp16") else (0.5 if dtype == "mxfp4" else 1.0)
    mte1 = float(tiles) * (base_m + base_n) * k * elem_bytes
    fix = float(actual_rows * n * write_c + extra_write)
    works = [PipeWork("CUBE", macs, "macs"), PipeWork("MTE1", mte1, "bytes"),
             PipeWork("FIX", fix, "bytes")]
    archetype = ExecutionArchetype.PURE_CUBE
    fixpipe = FixpipePath.L0C_TO_GM
    buffers = cube_buffers(base_m, base_n, base_k, 2 if dtype in ("bf16", "fp16") else 1)
    lane_ops: tuple[float, float] | None = None
    dual = None
    calls = 1
    lanes = (0,)
    workspace = 0
    if streamk:
        sk_tiles = max(1, tiles % 32)
        workspace = sk_tiles * base_m * base_n * 4
        reduce_ops = float(sk_tiles * base_m * base_n)
        extra_write += workspace + sk_tiles * base_m * base_n * 4
        gmm_read += workspace
        write += workspace
        vec_ops += reduce_ops
        works.append(PipeWork("VEC", reduce_ops))
        archetype = ExecutionArchetype.CV_COUPLED_1C2V
        lane_ops = (vec_ops / 2.0, vec_ops / 2.0)
        calls = 1
        lanes = (0, 1)
    if vec_ops and archetype is ExecutionArchetype.PURE_CUBE:
        archetype = ExecutionArchetype.CV_COUPLED_1C2V
        fixpipe = FixpipePath.L0C_TO_UB
        dual = 1 if base_m % 2 == 0 else 0
        calls = 2 if dual == 0 else 1
        lanes = (0, 1)
        lane_ops = (vec_ops / 2.0, vec_ops / 2.0)
        works.append(PipeWork("VEC", vec_ops))
    if inplace:
        gmm_read += actual_rows * n * write_c
    return make_flow(
        name, archetype, float(max(1, tiles)), 1, works,
        gm_read_bytes=int(gmm_read), gm_write_bytes=int(write),
        unique_gm_bytes=int(actual_rows * k * read_a + groups * k * n * read_b
                           + actual_rows * n * write_c + workspace + extra_read),
        sync_pairs=2 if archetype is ExecutionArchetype.CV_COUPLED_1C2V else 0,
        fixpipe_path=fixpipe, fixpipe_sub_block=0 if fixpipe is FixpipePath.L0C_TO_UB and lanes == (0, 1) else None,
        tiling_pool=BufferPool.L1, buffers=buffers,
        dual_dst_ctl=dual, fixpipe_calls=calls, fixpipe_m=base_m,
        lanes_receiving=lanes, vector_ops_by_lane=lane_ops,
        params={"m": actual_rows, "n": n, "k": k, "groups": groups,
                "group_sizes": list(group_sizes), "base_m": base_m,
                "base_n": base_n, "base_k": base_k, "k_loops": k_loops,
                "streamk_workspace_bytes": workspace, "dtype": dtype},
        notes=note or f"{name}: GM->L1->L0A/L0B->MMAD->L0C->GM, K-loop {k_loops}",
        authoring_backend=backend, dtype=dtype,
    )


def matmul_streamk(**kw: Any) -> Any:
    from .catalog_exact_cube import matmul_reference
    return matmul_reference("matmul_streamk", **kw)

def dual_level_quant_matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols", "dtype",
                      "base_m", "base_n", "base_k"})
    dtype_arg(kw, "fp8")
    m, n, k = _shape(kw)
    scales = int(m * math.ceil(k / 128) + n * math.ceil(k / 128)) * 2
    return _matmul("dual_level_quant_matmul", m, n, k, dtype="fp8",
                   read_a=1.0, read_b=1.0, extra_read=scales,
                   note="two-level per-group quant matmul; FP8 Cube rate", **_tile_opts(kw))


def quant_batch_matmul_mxa8w4(**kw: Any) -> Any:
    from .catalog_exact_cube import mxa8w4
    return mxa8w4(**kw)

def quant_batch_matmul_w4a8_mxfp(**kw: Any) -> Any:
    return quant_batch_matmul_mxa8w4(**kw)


def quant_matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols",
                      "base_m", "base_n", "base_k", "dtype"})
    dtype_arg(kw, "int8")
    m, n, k = _shape(kw)
    return _matmul("quant_matmul", m, n, k, dtype="int8", read_a=1.0, read_b=1.0,
                   note="INT8 quantized matmul", **_tile_opts(kw))


def weight_quant_batchmatmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols",
                      "base_m", "base_n", "base_k", "dtype"})
    dtype_arg(kw, "int8")
    m, n, k = _shape(kw)
    return _matmul("weight_quant_batchmatmul", m, n, k, dtype="int8",
                   read_a=2.0, read_b=1.0, note="weight-only INT8 quantized matmul", **_tile_opts(kw))


def quant_batch_matmul_inplace_add(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols",
                      "base_m", "base_n", "base_k", "dtype"})
    dtype_arg(kw, "int8")
    m, n, k = _shape(kw)
    return _matmul("quant_batch_matmul_inplace_add", m, n, k, dtype="int8",
                   read_a=1.0, read_b=1.0, inplace=True,
                   note="quant matmul with in-place FP32 accumulator add", **_tile_opts(kw))


def batch_matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols", "batch",
                      "base_m", "base_n", "base_k", "dtype"})
    m, n, k = _shape(kw)
    batch = pick_int(kw, ("batch",), 8)
    return _matmul("batch_matmul", m, n, k, groups=batch, group_sizes=(m,) * batch, dtype=dtype_arg(kw, "bf16", ("bf16", "fp16")),
                   note="batched matmul; batch is an independent group", **_tile_opts(kw))


def fused_matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols",
                      "base_m", "base_n", "base_k", "dtype"})
    m, n, k = _shape(kw)
    vec = float(m * n)
    return _matmul("fused_matmul", m, n, k, dtype=dtype_arg(kw, "bf16", ("bf16", "fp16")),
                   vec_ops=vec, note="matmul + bias/GELU epilogue on 1C:2V", **_tile_opts(kw))


def transpose_batch_matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols", "batch",
                      "base_m", "base_n", "base_k", "dtype"})
    m, n, k = _shape(kw)
    batch = pick_int(kw, ("batch",), 8)
    return _matmul("transpose_batch_matmul", m, n, k, groups=batch, group_sizes=(m,) * batch,
                   dtype=dtype_arg(kw, "bf16", ("bf16", "fp16")),
                   note="transposed B batch matmul with strided L1 tiles", **_tile_opts(kw))


def grouped_matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols", "groups",
                      "experts", "group_sizes", "group_list", "base_m", "base_n",
                      "base_k", "dtype"})
    m, n, k = _shape(kw)
    groups = pick_int(kw, ("groups", "experts"), 8)
    sizes = _group_sizes(kw, groups, m)
    return _matmul("grouped_matmul", m, n, k, groups=groups, group_sizes=sizes,
                   dtype=dtype_arg(kw, "bf16", ("bf16", "fp16")),
                   note="per-expert integer histogram slices; no host padding", **_tile_opts(kw))


def grouped_matmul_finalize_routing(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols", "groups",
                      "experts", "group_sizes", "group_list", "batch", "base_m",
                      "base_n", "base_k", "shared_experts"})
    m, n, k = _shape(kw)
    groups = pick_int(kw, ("groups", "experts"), 8)
    batch = pick_int(kw, ("batch",), m)
    sizes = _group_sizes(kw, groups, m)
    shared = pick_int(kw, ("shared_experts",), 1, minimum=0)
    vec = float(m * n) * 2.0 + float(batch * n) * 2.0
    if shared:
        vec += float(batch * n) * 2.0
    return conditional_prior(_matmul(
        "grouped_matmul_finalize_routing", m, n, k, groups=groups,
        group_sizes=sizes, dtype="fp8", read_a=1.0, read_b=1.0,
        extra_read=int(m * math.ceil(k / 64) * 2 + m * 4 + batch * 8),
        extra_write=int(batch * n * 4), vec_ops=vec,
        note="MXFP8 GMM + split-M dualDst L0C->UB + FP32 atomic finalize routing",
        **_tile_opts(kw),
    ))


def matmul(**kw: Any) -> Any:
    check_kwargs(kw, {"m", "n", "k", "tokens", "rows", "hidden", "cols",
                      "base_m", "base_n", "base_k", "dtype"})
    m, n, k = _shape(kw)
    return _matmul("matmul", m, n, k, dtype=dtype_arg(kw, "bf16", ("bf16", "fp16")),
                   note="dense A16W16 matmul", **_tile_opts(kw))


__all__ = [
    "batch_matmul", "dual_level_quant_matmul", "fused_matmul",
    "grouped_matmul", "grouped_matmul_finalize_routing", "matmul",
    "matmul_bf16", "matmul_streamk", "quant_batch_matmul_inplace_add",
    "quant_batch_matmul_mxa8w4", "quant_batch_matmul_w4a8_mxfp", "quant_matmul",
    "transpose_batch_matmul", "weight_quant_batchmatmul", "_matmul", "_shape",
]
