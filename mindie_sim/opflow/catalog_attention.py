"""Analytic-prior attention, KDA, MLA, indexer, and MoonEP flow builders.

[Role]
Encode query-block attention, chunked Kimi Delta Attention, ReplaySSM,
DeepSeek V4.1 mega_mla / mega_full_index, and the MoonEP five-op communication
suite with explicit stage work, lane partition, and recurrence scope.
[Invariants]
Vector work is conserved globally and split explicitly across the two AIV lanes.
Cube and human-visible traffic use dtype-specific rates. KDA/ReplaySSM state is
FP32 D x D in UB and AIV-only. MoonEP expansion modes produce distinct resource
contention rather than a shared local-Vector placeholder.
[Timeouts & Locks]
Arithmetic-only; no device calls, sleeps, locks, or external dependencies.
[Side-Effect Policy]
Read-only module; builders allocate in-memory records only.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .catalog_common import (
    as_int, attention_buffers, check_kwargs, conditional_prior, cube_buffers, dtype_arg, make_flow, pick_int,
)
from .flow_schema import (
    AuthoringBackend, BufferPool, BufferSpec, ExecutionArchetype,
    FixpipePath, LoopCarriedRecurrence, PipeWork,
)


def _attn(name: str, batch: int, heads: int, seq: int, dim: int, *,
          kv_len: int | None = None, block_q: int = 128, block_kv: int = 128,
          dtype: str = "bf16", sparse_topk: int | None = None,
          extra_read: int = 0, extra_write: int = 0, quant: bool = False,
          note: str = "", backend: AuthoringBackend = AuthoringBackend.CANNBOT_DSL) -> Any:
    if not isinstance(quant, bool):
        raise ValueError("quant must be a bool")
    kv = kv_len or seq
    q_blocks = math.ceil(seq / block_q)
    items = batch * heads * q_blocks
    active_kv = min(kv, sparse_topk) if sparse_topk else kv
    macs = 2.0 * items * block_q * active_kv * dim
    vec = float(items) * block_q * active_kv * 8.0 + float(items) * dim * 4.0
    q_bytes = batch * heads * seq * dim * (1 if quant else 2)
    kv_bytes = batch * heads * active_kv * dim * (1 if quant else 2)
    out_bytes = batch * heads * seq * dim * 2
    read = int(q_bytes + 2 * kv_bytes * q_blocks + extra_read)
    write = int(out_bytes + extra_write)
    unique = int(q_bytes + 2 * kv_bytes + out_bytes)
    fix = float(items * block_q * dim * 2)
    return make_flow(
        name, ExecutionArchetype.CV_COUPLED_1C2V, float(items), 1,
        (PipeWork("CUBE", macs, "macs"), PipeWork("VEC", vec),
         PipeWork("FIX", fix, "bytes")),
        gm_read_bytes=read, gm_write_bytes=write, unique_gm_bytes=unique,
        sync_pairs=2, fixpipe_path=FixpipePath.L0C_TO_UB, fixpipe_sub_block=0,
        tiling_pool=BufferPool.UB_LANE0,
        buffers=attention_buffers(block_q, dim),
        dual_dst_ctl=0, fixpipe_calls=2, fixpipe_m=block_q, fixpipe_n=block_q,
        lanes_receiving=(0, 1), vector_ops_by_lane=(vec / 2.0, vec / 2.0),
        params={"batch": batch, "heads": heads, "seq": seq, "dim": dim,
                "kv_len": kv, "block_q": block_q, "block_kv": block_kv,
                "q_blocks": q_blocks, "sparse_topk": sparse_topk},
        notes=note or "query-block attention with L0C->UB dual-lane handoff",
        authoring_backend=backend, dtype=dtype,
    )


def flash_attn(**kw: Any) -> Any:
    check_kwargs(kw, {"batch", "tokens", "rows", "heads", "seq", "kv_len", "dim",
                      "hidden", "cols", "block_q", "block_kv", "dtype"})
    batch = pick_int(kw, ("batch", "tokens", "rows"), 8)
    heads = pick_int(kw, ("heads",), 32)
    seq = pick_int(kw, ("seq",), 2048)
    dim = pick_int(kw, ("dim", "d"), 128)
    return _attn("flash_attn", batch, heads, seq, dim,
                 kv_len=pick_int(kw, ("kv_len",), seq),
                 block_q=pick_int(kw, ("block_q",), 128),
                 block_kv=pick_int(kw, ("block_kv",), 128),
                 dtype=dtype_arg(kw, "bf16", ("bf16", "fp16")))


def flash_attn_score(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "batch", "heads", "h", "dim", "d",
                      "seq", "seq_len", "kv_len", "block_q", "block_kv", "dtype"})
    dtype_arg(kw, "bf16")
    batch = pick_int(kw, ("batch", "tokens", "rows"), 8)
    heads = pick_int(kw, ("heads", "h"), 32)
    seq = pick_int(kw, ("seq", "seq_len", "kv_len"), 2048)
    dim = pick_int(kw, ("dim", "d"), 128)
    return _attn("flash_attn_score", batch, heads, seq, dim,
                 block_q=pick_int(kw, ("block_q",), 128),
                 block_kv=pick_int(kw, ("block_kv",), 128))


def flash_attn_fp8_fullquant(**kw: Any) -> Any:
    check_kwargs(kw, {"batch", "tokens", "rows", "heads", "seq", "kv_len", "dim",
                      "block_q", "block_kv"})
    batch = pick_int(kw, ("batch", "tokens", "rows"), 8)
    heads = pick_int(kw, ("heads",), 32)
    seq = pick_int(kw, ("seq",), 2048)
    dim = pick_int(kw, ("dim", "d"), 128)
    scale_bytes = batch * heads * seq * 4
    return _attn("flash_attn_fp8_fullquant", batch, heads, seq, dim,
                 kv_len=seq, quant=True, dtype="fp8", extra_read=scale_bytes,
                 extra_write=scale_bytes, note="FP8 full-quantized FA with per-row scales")


def sparse_flash_mla(**kw: Any) -> Any:
    check_kwargs(kw, {"batch", "tokens", "rows", "heads", "seq", "kv_len", "dim",
                      "topk", "block_q", "block_kv", "dtype"})
    dtype_arg(kw, "bf16")
    batch = pick_int(kw, ("batch", "tokens", "rows"), 8)
    heads = pick_int(kw, ("heads",), 32)
    seq = pick_int(kw, ("seq",), 2048)
    dim = pick_int(kw, ("dim", "d"), 128)
    topk = pick_int(kw, ("topk",), 2048)
    return _attn("sparse_flash_mla", batch, heads, seq, dim, kv_len=seq,
                 sparse_topk=topk, extra_read=batch * topk * 4,
                 note="sparse MLA top-k index gather; page/shared-KV layout")


def flash_kda(**kw: Any) -> Any:
    check_kwargs(kw, {"batch", "tokens", "rows", "heads", "seq", "dim", "chunk_size",
                      "training", "gqa"})
    batch = pick_int(kw, ("batch", "tokens", "rows"), 1)
    heads = pick_int(kw, ("heads",), 2)
    seq = pick_int(kw, ("seq",), 65)
    dim = pick_int(kw, ("dim",), 128)
    chunk = pick_int(kw, ("chunk_size",), 64)
    training = kw.get("training", False)
    if not isinstance(training, bool):
        raise ValueError("training must be bool")
    chunks = math.ceil(seq / chunk)
    items = batch * heads
    per_chunk = (2.0 * chunk * chunk * dim + chunk ** 3 / 6.0
                 + 2.0 * chunk * dim * dim)
    macs = float(items * chunks) * per_chunk
    vec = float(batch * heads * seq * dim * 8.0)
    fix = float(batch * heads * seq * dim * 2)
    metadata_bytes = batch * heads * chunks * 16
    read = batch * heads * seq * dim * 6 * 2 + heads * dim * 4 + metadata_bytes
    write = batch * heads * seq * dim * (2 + (2 * chunk * 4 * 3 if training else 0))
    return conditional_prior(make_flow(
        "flash_kda", ExecutionArchetype.CV_COUPLED_1C2V, float(items), 1,
        (PipeWork("CUBE", macs, "macs"), PipeWork("VEC", vec),
         PipeWork("FIX", fix, "bytes")),
        gm_read_bytes=read, gm_write_bytes=write, unique_gm_bytes=read + write,
        sync_pairs=2, fixpipe_path=FixpipePath.L0C_TO_UB, fixpipe_sub_block=0,
        tiling_pool=BufferPool.UB_LANE0,
        buffers=attention_buffers(chunk, dim),
        dual_dst_ctl=0, fixpipe_calls=2, fixpipe_m=chunk, fixpipe_n=chunk,
        lanes_receiving=(0, 1), vector_ops_by_lane=(vec / 2.0, vec / 2.0),
        params={"batch": batch, "heads": heads, "seq": seq, "dim": dim,
                "chunk_size": chunk, "chunks": chunks, "training": training,
                "metadata": "flash_kda_metadata", "metadata_bytes": metadata_bytes},
        notes="chunk=64 causal QK + triangular solve + FP32 inter-chunk state",
        authoring_backend=AuthoringBackend.CANNBOT_DSL, dtype="bf16",
        recurrence_steps=chunks,
        recurrence=LoopCarriedRecurrence(1.0, distance_items=1, state_name="kda_chunk_state"),
    ))


def flash_kda_fwd(**kw: Any) -> Any:
    allowed = {"batch", "tokens", "rows", "heads", "seq", "dim", "chunk_size",
               "training", "gqa"}
    check_kwargs(kw, allowed)
    return replace(flash_kda(**kw), name="flash_kda_fwd")


def fused_recurrent_kda_snapshot(**kw: Any) -> Any:
    check_kwargs(kw, {"batch", "blocks", "heads", "dim", "accepted_tokens"})
    batch = pick_int(kw, ("batch", "blocks"), 2)
    heads = pick_int(kw, ("heads",), 6)
    dim = pick_int(kw, ("dim",), 128)
    items = batch * heads
    read = items * dim * dim * 4 + batch * heads * dim * 4
    write = items * dim * dim * 4
    return make_flow(
        "fused_recurrent_kda_snapshot", ExecutionArchetype.PURE_VECTOR, items, 1,
        (PipeWork("VEC", float(items * dim * dim)),),
        gm_read_bytes=read, gm_write_bytes=write, unique_gm_bytes=read + write,
        tiling_pool=BufferPool.UB_LANE0, live_bytes_per_unit=dim * dim * 4,
        buffers=(BufferSpec("kda_snapshot_state", BufferPool.UB_LANE0,
                            dim * dim * 4, slots=2),),
        params={"batch": batch, "heads": heads, "dim": dim},
        notes="checkpoint snapshot write from read-only FP32 state",
        authoring_backend=AuthoringBackend.CANNBOT_DSL,
    )


def mega_recurrent_kda(**kw: Any) -> Any:
    allowed = {"batch", "tokens", "rows", "heads", "seq", "dim", "chunk_size",
               "training", "gqa"}
    check_kwargs(kw, allowed)
    return replace(flash_kda(**kw), name="mega_recurrent_kda")


def mega_mla(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "heads", "hidden", "q_rank", "head_dim",
                      "kw", "kc", "kv_records", "pages", "block_q"})
    tokens = pick_int(kw, ("tokens", "rows"), 16)
    heads = pick_int(kw, ("heads",), 16)
    hidden = pick_int(kw, ("hidden",), 5120)
    q_rank = pick_int(kw, ("q_rank",), 1280)
    head_dim = pick_int(kw, ("head_dim",), 512)
    kw_records = pick_int(kw, ("kw",), 512)
    kc_records = pick_int(kw, ("kc",), 512)
    records = pick_int(kw, ("kv_records",), kw_records + kc_records)
    macs = (float(tokens) * hidden * q_rank
            + float(tokens) * hidden * head_dim
            + float(tokens) * q_rank * heads * head_dim
            + 2.0 * tokens * records * heads * head_dim)
    vec = (float(tokens) * heads * head_dim * 8.0 + float(tokens) * hidden * 4.0
           + float(tokens) * (kw_records + kc_records) * 2.0)
    read = (tokens * hidden * 2
            + (q_rank * hidden + head_dim * hidden + heads * head_dim * q_rank)
            + tokens * (kw_records * 544 + kc_records * 320 + records * 4))
    write = tokens * heads * head_dim * 2 + tokens * 544
    return conditional_prior(make_flow(
        "mega_mla", ExecutionArchetype.CV_COUPLED_1C2V, float(tokens), 1,
        (PipeWork("CUBE", macs, "macs"), PipeWork("VEC", vec),
         PipeWork("FIX", float(tokens * heads * head_dim * 2), "bytes")),
        gm_read_bytes=read, gm_write_bytes=write,
        unique_gm_bytes=tokens * hidden * 2 + heads * head_dim * q_rank + write,
        sync_pairs=4, fixpipe_path=FixpipePath.L0C_TO_UB, fixpipe_sub_block=0,
        tiling_pool=BufferPool.UB_LANE0, buffers=attention_buffers(64, head_dim),
        dual_dst_ctl=0, fixpipe_calls=2, fixpipe_m=64, fixpipe_n=64,
        lanes_receiving=(0, 1), vector_ops_by_lane=(vec / 2.0, vec / 2.0),
        params={"tokens": tokens, "heads": heads, "hidden": hidden,
                "q_rank": q_rank, "head_dim": head_dim, "kw": kw_records,
                "kc": kc_records, "window_record_bytes": 544,
                "compressed_record_bytes": 320},
        notes="Q/KV projection + RMS/RoPE + window 544 B + compressed 320 B dual cache",
        authoring_backend=AuthoringBackend.CANNBOT_DSL, dtype="fp8",
        recurrence_steps=math.ceil(records / 128),
    ))


def mega_full_index(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "q_rank", "latent", "heads",
                      "head_dim", "kv_len", "topk", "candidates", "page"})
    tokens = pick_int(kw, ("tokens", "rows"), 8)
    hidden = pick_int(kw, ("hidden",), 5120)
    q_rank = pick_int(kw, ("q_rank",), 1280)
    latent = pick_int(kw, ("latent",), 512)
    heads = pick_int(kw, ("heads",), 32)
    head_dim = pick_int(kw, ("head_dim",), 128)
    kv_len = pick_int(kw, ("kv_len",), 1024)
    topk = pick_int(kw, ("topk",), 512)
    candidates = pick_int(kw, ("candidates",), 2048)
    page = pick_int(kw, ("page",), 128)
    pages = math.ceil(kv_len / page)
    macs = (float(tokens) * hidden * q_rank + float(tokens) * hidden * latent
            + float(tokens) * hidden + float(tokens) * heads * kv_len * head_dim)
    vec = (float(tokens) * hidden * 4.0 + float(tokens) * heads * kv_len * 2.0
           + float(tokens) * candidates * 4.0 + float(tokens) * topk * 2.0)
    read = (tokens * hidden * 2 + q_rank * hidden + latent * hidden + hidden
            + tokens * kv_len * latent // 2 + tokens * kv_len * 4)
    write = tokens * topk * 4 + tokens * candidates * 4 + tokens * latent // 2
    return conditional_prior(make_flow(
        "mega_full_index", ExecutionArchetype.CV_COUPLED_1C2V, float(tokens), 1,
        (PipeWork("CUBE", macs, "macs"), PipeWork("VEC", vec),
         PipeWork("FIX", float(tokens * latent // 2), "bytes")),
        gm_read_bytes=read, gm_write_bytes=write,
        unique_gm_bytes=tokens * hidden * 2 + tokens * topk * 4 + write,
        sync_pairs=4, fixpipe_path=FixpipePath.L0C_TO_UB, fixpipe_sub_block=0,
        tiling_pool=BufferPool.UB_LANE0, buffers=attention_buffers(64, head_dim),
        dual_dst_ctl=0, fixpipe_calls=2, fixpipe_m=64, fixpipe_n=64,
        lanes_receiving=(0, 1), vector_ops_by_lane=(vec / 2.0, vec / 2.0),
        params={"tokens": tokens, "hidden": hidden, "q_rank": q_rank,
                "latent": latent, "heads": heads, "head_dim": head_dim,
                "kv_len": kv_len, "topk": topk, "candidates": candidates,
                "pages": pages, "mxfp4_cache": True},
        notes="Indexer prologue QW/K + QLI Top-512 + MXFP4 paged cache",
        authoring_backend=AuthoringBackend.CANNBOT_DSL, dtype="mxfp4",
        recurrence_steps=pages,
    ))


def moonep(**kw: Any) -> Any:
    check_kwargs(kw, {"tokens", "rows", "hidden", "cols", "experts", "topk",
                      "ranks", "hccl_expansion_mode", "mode", "reduce_grad"})
    tokens = pick_int(kw, ("tokens", "rows"), 256)
    hidden = pick_int(kw, ("hidden", "cols"), 2048)
    experts = pick_int(kw, ("experts",), 256)
    topk = pick_int(kw, ("topk",), 8)
    ranks = pick_int(kw, ("ranks",), 64)
    mode = str(kw.get("hccl_expansion_mode", kw.get("mode", "AIV"))).upper()
    if mode not in ("AIV", "CCU_SCHED", "AICPU"):
        raise ValueError("hccl_expansion_mode must be AIV, CCU_SCHED, or AICPU")
    dispatched = tokens * topk
    payload = dispatched * hidden * 2
    prefetch = experts * hidden * 2
    local_ops = float(tokens * hidden) * 2.0 + float(dispatched) * 4.0
    comm_ops = float(payload) / 4.0
    if mode == "AIV":
        vec = local_ops + comm_ops
        extra = ()
        sync = 3
    elif mode == "CCU_SCHED":
        vec = local_ops + comm_ops * 0.1
        extra = (PipeWork("MTE3", float(payload), "bytes"),)
        sync = 1
    else:
        vec = local_ops + comm_ops * 0.5
        extra = (PipeWork("MTE3", float(payload), "bytes"), PipeWork("SIMT", float(tokens) * 8.0))
        sync = 6
    read = int(payload / max(1, ranks) + prefetch + tokens * hidden * 2)
    write = int(payload + tokens * hidden * 2)
    return conditional_prior(make_flow(
        "moonep", ExecutionArchetype.COMMUNICATION_HYBRID, float(tokens), 1,
        (PipeWork("VEC", vec),) + extra,
        gm_read_bytes=read, gm_write_bytes=write,
        unique_gm_bytes=tokens * hidden * 2 * 2 + prefetch,
        sync_pairs=sync, tiling_pool=BufferPool.UB_LANE0, live_bytes_per_unit=8.0,
        buffers=(BufferSpec("moonep_stage", BufferPool.UB_LANE0, 32768, slots=2),),
        params={"tokens": tokens, "hidden": hidden, "experts": experts,
                "topk": topk, "ranks": ranks, "hccl_expansion_mode": mode,
                "ops_suite": ["plan", "prefetch_weight", "reduce_grad",
                              "dispatch", "combine"]},
        notes=f"MoonEP five-op suite with {mode} communication expansion",
        authoring_backend=AuthoringBackend.CANNBOT_DSL,
        vector_mode="simt" if mode == "AICPU" else "simd",
    ))


__all__ = [
    "flash_attn", "flash_attn_fp8_fullquant", "flash_attn_score", "flash_kda",
    "flash_kda_fwd", "fused_recurrent_kda_snapshot", "mega_full_index",
    "mega_mla", "mega_recurrent_kda", "moonep", "sparse_flash_mla", "_attn",
]
