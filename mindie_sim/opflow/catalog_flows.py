"""Selected mathematical routes and analytic vector/recurrent priors.

[Role]
Build the vector/RMSNorm/reduction/recurrent-KDA/Engram/mHC/Vision families as
immutable flows whose traffic, MAC/op counts, recurrence scope, and explicit
SRAM buffers carry explicit selected-route or analytic-prior scope.
[Invariants]
Shapes are validated dimension by dimension; tokens=0 only for the documented
zero-launch RMSNorm-family no-op. Vector work is global and conserved. Stateful
KDA flows are PURE_VECTOR with FP32 D x D state and sequential token steps.
Every builder rejects unknown kwargs and non-positive dimensions.
[Timeouts & Locks]
Arithmetic-only; no device calls, sleeps, locks, or external dependencies.
[Side-Effect Policy]
Read-only module; builders allocate in-memory records only.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable

from .catalog_common import (
    align_up, as_int, check_kwargs, conditional_prior, cube_buffers, make_flow, pick_float,
    pick_int, ub_buffers,
)
from .flow_schema import (
    AuthoringBackend, BufferPool, BufferSpec, ExecutionArchetype,
    LoopCarriedRecurrence, PipeWork,
)

Builder = Callable[..., Any]
_FP16, _BF16, _FP32 = 2, 2, 4


def _zero(name: str, params: dict[str, Any]) -> Any:
    return make_flow(
        name, ExecutionArchetype.PURE_VECTOR, 0, 1, (), params=params,
        no_launch=True, notes="zero_launch_noop",
    )


def _vector(name: str, elements: int, *, ops_per_element: float,
            read_bytes: int, write_bytes: int, unique_bytes: int,
            live_bytes: float = 6.0, tile: int = 4096,
            extra: tuple[PipeWork, ...] = (), sync: int = 0,
            buffers: tuple[BufferSpec, ...] | None = None,
            params: dict[str, Any] | None = None, notes: str = "",
            backend: AuthoringBackend = AuthoringBackend.ASCENDC,
            recurrence: LoopCarriedRecurrence | None = None,
            recurrence_steps: int = 1, lanes: int = 1,
            dtype: str = "bf16", vector_mode: str = "simd") -> Any:
    elements = max(1, int(elements))
    works = (PipeWork("VEC", float(elements) * ops_per_element),) + extra
    return make_flow(
        name, ExecutionArchetype.PURE_VECTOR, elements,
        min(max(1, tile), elements), works,
        gm_read_bytes=int(read_bytes), gm_write_bytes=int(write_bytes),
        unique_gm_bytes=int(unique_bytes), sync_pairs=sync,
        tiling_pool=BufferPool.UB_LANE0, live_bytes_per_unit=live_bytes,
        buffers=buffers if buffers is not None else ub_buffers(min(4096, elements), 2, lanes=lanes),
        params=params or {}, notes=notes, authoring_backend=backend,
        recurrence=recurrence, recurrence_steps=recurrence_steps, dtype=dtype,
        vector_mode=vector_mode,
    )


def add_rms_norm(**kw: Any) -> Any:
    from .catalog_exact import add_rms
    return add_rms(**kw)

def rms_norm(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "seq_len", "hidden", "cols", "n", "tile"})
    tokens = pick_int(kw, ("tokens", "rows", "seq_len"), 32)
    hidden = pick_int(kw, ("hidden", "cols", "n"), 1536)
    tile = pick_int(kw, ("tile",), 8192)
    elements = tokens * hidden
    read = elements * _BF16 + hidden * _BF16
    write = elements * _BF16
    return _vector(
        "rms_norm", elements, ops_per_element=5.0, read_bytes=read, write_bytes=write,
        unique_bytes=read + write, live_bytes=6.0, tile=tile,
        buffers=ub_buffers(min(tile, 4096), 2, lanes=2),
        params={"tokens": tokens, "hidden": hidden}, notes="row RMSNorm, independent rows",
    )


def layernorm_welford(**kw: Any) -> Any:
    from .catalog_exact import welford
    return welford(**kw)

def block_attn_res_update_rms_norm(**kw: Any) -> Any:
    from .catalog_exact import attnres
    return attnres(**kw)

def engram_gate(**kw: Any) -> Any:
    from .catalog_exact import engram
    return engram(**kw)

def mhc_hcpost(**kw: Any) -> Any:
    from .catalog_exact import hcpost
    return hcpost(**kw)

def mhc_hcpre_cv_splitk(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "hc_mult", "split_k",
                      "split", "sinkhorn_iters", "base_m", "base_n", "base_k"})
    tokens = pick_int(kw, ("tokens", "rows"), 16)
    hidden = pick_int(kw, ("hidden", "cols"), 4096)
    hc = pick_int(kw, ("hc_mult",), 4)
    split = pick_int(kw, ("split_k", "split"), 4)
    iters = pick_int(kw, ("sinkhorn_iters",), 16, minimum=1)
    in_dim = hc * hidden
    out_dim = (hc + 2) * hc
    tasks = tokens * split
    total_macs = float(tokens) * in_dim * out_dim
    per_task = total_macs / tasks
    workspace = tokens * (out_dim + 1) * _FP32 * split
    read = tokens * in_dim * _BF16 + in_dim * out_dim * _BF16 + workspace
    write = tokens * out_dim * _FP32
    vec = tokens * (2*hc*hidden + 3 + out_dim + 12*hc + (6+4*iters)*hc*hc
                    + (2*hc-1)*hidden + out_dim*(split-1))
    base_m = pick_int(kw, ("base_m",), 64)
    base_n = pick_int(kw, ("base_n",), 64)
    base_k = pick_int(kw, ("base_k",), 64)
    return conditional_prior(make_flow(
        "mhc_hcpre_cv_splitk", ExecutionArchetype.CV_COUPLED_1C2V, tasks, 1,
        (PipeWork("CUBE", total_macs, "macs"), PipeWork("VEC", vec),
         PipeWork("FIX", float(tokens * out_dim * _FP32), "bytes")),
        gm_read_bytes=read, gm_write_bytes=write,
        unique_gm_bytes=tokens * in_dim * _BF16 + in_dim * out_dim * _BF16 + write,
        sync_pairs=2, fixpipe_path=__import__("mindie_sim.opflow.flow_schema", fromlist=["FixpipePath"]).FixpipePath.L0C_TO_UB,
        fixpipe_sub_block=0, tiling_pool=BufferPool.L1,
        buffers=cube_buffers(base_m, base_n, base_k),
        dual_dst_ctl=0, fixpipe_calls=2, lanes_receiving=(0, 1),
        vector_ops_by_lane=(vec / 2.0, vec / 2.0),
        params={"tokens": tokens, "hidden": hidden, "hc_mult": hc, "split_k": split,
                "in_dim": in_dim, "out_dim": out_dim, "total_macs": total_macs,
                "workspace_bytes": workspace},
        notes="HcPre [T,hc_mult*D]x[hc_mult*D,(hc_mult+2)*hc_mult]; MACs conserved",
        authoring_backend=AuthoringBackend.CANNBOT_DSL,
        dtype="bf16",
    ))


def swiglu_dynamic_quant(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile"})
    tokens = pick_int(kw, ("tokens", "rows"), 32)
    hidden = pick_int(kw, ("hidden", "cols"), 4096)
    elements = tokens * hidden
    read = elements * _BF16 * 2
    write = elements * 1 + elements // 128 * _FP32
    return _vector(
        "swiglu_dynamic_quant", elements, ops_per_element=12.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=6.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
    )


def dynamic_mx_quant_sr(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile"})
    tokens = pick_int(kw, ("tokens", "rows"), 32)
    hidden = pick_int(kw, ("hidden", "cols"), 4096)
    elements = tokens * hidden
    read = elements * _BF16
    write = elements // 2 + elements // 32
    return _vector(
        "dynamic_mx_quant_sr", elements, ops_per_element=14.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=5.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
    )


def softmax_long_r(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile"})
    rows = pick_int(kw, ("tokens", "rows"), 32)
    cols = pick_int(kw, ("hidden", "cols"), 8192)
    two_pass = cols * _FP32 * 2 > 253952
    read = rows * cols * _FP32 * (2 if two_pass else 1)
    write = rows * cols * _FP32
    return _vector(
        "softmax_long_r", rows * cols, ops_per_element=10.0, read_bytes=read,
        write_bytes=write, unique_bytes=rows * cols * _FP32 + write, live_bytes=8.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(2048, 4, lanes=2),
        params={"tokens": rows, "hidden": cols, "two_pass": two_pass},
        notes="independent rows; online max/sum within each row",
        backend=AuthoringBackend.CANNBOT_DSL,
    )


def index_gather(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile"})
    rows = pick_int(kw, ("tokens", "rows"), 256)
    hidden = pick_int(kw, ("hidden", "cols"), 512)
    elements = rows * hidden
    read = elements * _BF16 + rows * 8
    write = elements * _BF16
    return _vector(
        "index_gather", elements, ops_per_element=4.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=4.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
        extra=(PipeWork("SIMT", float(rows * hidden) * 0.05),),
        params={"tokens": rows, "hidden": hidden}, notes="SIMT scalar index lowering",
    )


def gather_v2(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile"})
    return replace(index_gather(**kw), name="gather_v2")


def scatter_elements(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile", "atomic"})
    rows = pick_int(kw, ("tokens", "rows"), 256)
    hidden = pick_int(kw, ("hidden", "cols"), 512)
    elements = rows * hidden
    read = elements * _BF16 + rows * 8
    write = elements * _BF16
    atomic = kw.get("atomic", True)
    if not isinstance(atomic, bool):
        raise ValueError("atomic must be a bool")
    return _vector(
        "scatter_elements", elements, ops_per_element=6.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=4.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
        extra=(PipeWork("SIMT", float(rows * hidden) * (0.2 if atomic else 0.05)),),
        params={"tokens": rows, "hidden": hidden, "atomic": atomic},
        notes="duplicate-index serialization; atomic path is non-deterministic",
    )


def top_k_mrgsort(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "k", "tile"})
    rows = pick_int(kw, ("tokens", "rows"), 32)
    cols = pick_int(kw, ("hidden", "cols"), 4096)
    k = pick_int(kw, ("k",), 128)
    if k > cols:
        raise ValueError("k must not exceed the row length")
    elements = rows * cols
    merges = rows * cols * max(1.0, math.log2(max(2, k)))
    return _vector(
        "top_k_mrgsort", elements, ops_per_element=6.0,
        read_bytes=elements * _FP32, write_bytes=rows * k * 8,
        unique_bytes=elements * _FP32 + rows * k * 8, live_bytes=8.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(2048, 4, lanes=2),
        extra=(PipeWork("SIMT", merges * 0.4),),
        params={"tokens": rows, "hidden": cols, "k": k},
        notes="two-level merge sort; temporary buffers stay in UB",
    )


def rejection_random_sampling(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile", "candidates"})
    rows = pick_int(kw, ("tokens", "rows"), 256)
    hidden = pick_int(kw, ("hidden", "cols"), 512)
    elements = rows * hidden
    read = elements * _BF16
    return _vector(
        "rejection_random_sampling", elements, ops_per_element=6.0,
        read_bytes=read, write_bytes=rows * 4, unique_bytes=read + rows * 4,
        live_bytes=4.0, tile=pick_int(kw, ("tile",), 4096),
        buffers=ub_buffers(4096, 2, lanes=2),
        extra=(PipeWork("SIMT", float(elements) * 0.5),),
        params={"tokens": rows, "hidden": hidden},
        notes="exponential-race rejection sampling; SIMT scalar RNG path",
    )


_THREADS = (32, 64, 128, 256, 512, 1024)


def _threads(value: Any) -> int:
    threads = as_int("threads", value)
    if threads not in _THREADS:
        raise ValueError(f"threads must be one of {_THREADS}")
    return threads


def add(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "elements", "tile", "threads"})
    elements = pick_int(kw, ("elements", "tokens", "rows"), 4096)
    if "hidden" in kw or "cols" in kw:
        elements = pick_int(kw, ("tokens", "rows"), 64) * pick_int(kw, ("hidden", "cols"), 64)
    read = elements * _FP32 * 2
    write = elements * _FP32
    return _vector(
        "add", elements, ops_per_element=1.0, read_bytes=read, write_bytes=write,
        unique_bytes=read + write, live_bytes=8.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 4, lanes=2),
        extra=(PipeWork("SIMT", float(elements)),),
        params={"elements": elements, "threads": _threads(kw.get("threads", 1024))},
        notes="GM->UB->SIMT add->GM with 32 B tiling and 2-slot ring buffers",
        vector_mode="simt",
    )


def attn_epilogue(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "heads", "dim", "hidden", "cols", "tile"})
    rows = pick_int(kw, ("tokens", "rows"), 32)
    heads = pick_int(kw, ("heads",), 32)
    dim = pick_int(kw, ("dim", "d"), 128)
    elements = rows * heads * dim
    read = elements * _BF16 * 2
    write = elements * _BF16
    return _vector(
        "attn_epilogue", elements, ops_per_element=6.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=6.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
        params={"tokens": rows, "heads": heads, "dim": dim},
        notes="attention epilogue scale/residual/cast",
        backend=AuthoringBackend.CANNBOT_DSL,
    )


def flash_kda_cp_preprocess(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "heads", "dim", "hidden", "cols", "tile", "seq"})
    rows = pick_int(kw, ("tokens", "rows", "seq"), 32)
    heads = pick_int(kw, ("heads",), 8)
    dim = pick_int(kw, ("dim", "d"), 128)
    elements = rows * heads * dim
    read = elements * _BF16 * 5 + rows * heads * _FP32
    write = elements * _BF16 * 2
    return _vector(
        "flash_kda_cp_preprocess", elements, ops_per_element=8.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=6.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
        params={"tokens": rows, "heads": heads, "dim": dim},
        notes="q/k/v/g/beta scan preprocessing for context parallelism",
        backend=AuthoringBackend.CANNBOT_DSL,
    )


def gdn2_prepare(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "tile"})
    rows = pick_int(kw, ("tokens", "rows"), 32)
    hidden = pick_int(kw, ("hidden", "cols"), 4096)
    elements = rows * hidden
    read = elements * _BF16 * 2
    write = elements * _BF16
    return _vector(
        "gdn2_prepare", elements, ops_per_element=6.0, read_bytes=read,
        write_bytes=write, unique_bytes=read + write, live_bytes=5.0,
        tile=pick_int(kw, ("tile",), 4096), buffers=ub_buffers(4096, 2, lanes=2),
        params={"tokens": rows, "hidden": hidden},
        notes="GDN2 state/layout preparation, AIV only",
    )


def _state_buffers() -> tuple[BufferSpec, ...]:
    return (
        BufferSpec("kda_state_fp32", BufferPool.UB_LANE0, 128 * 128 * 4, slots=2),
        BufferSpec("kda_vec", BufferPool.UB_LANE0, 4096, slots=2),
        BufferSpec("kda_out", BufferPool.UB_LANE0, 4096, slots=2),
    )


def fused_recurrent_kda_replayssm(**kw: Any) -> Any:
    from .catalog_exact import replay
    return replay(**kw)

def commit_recurrent_kda(**kw: Any) -> Any:
    from .catalog_exact import commit
    return commit(**kw)

__all__ = [
    "add", "add_rms_norm", "attn_epilogue", "block_attn_res_update_rms_norm",
    "commit_recurrent_kda", "dynamic_mx_quant_sr", "engram_gate",
    "flash_kda_cp_preprocess", "fused_recurrent_kda_replayssm", "gather_v2",
    "gdn2_prepare", "index_gather", "layernorm_welford", "mhc_hcpost",
    "mhc_hcpre_cv_splitk", "rejection_random_sampling", "rms_norm",
    "scatter_elements", "softmax_long_r", "swiglu_dynamic_quant",
    "top_k_mrgsort", "_vector", "_zero",
]
