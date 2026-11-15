"""Formula builders for prefill/decode attention, MLA prolog and DSA sparse ops.
[Role]
  Encode the per-sequence attention contract: paired query/KV lengths, exact
  causal visibility geometry (decode sees the full cache, square prefill the
  inclusive triangle), independent ``head_dim_qk``/``head_dim_v``, explicit mask
  ABI traffic, GQA/MQA unique-KV-head accounting, and true Cube-tile counts.
  Also cover MLA prolog, LightningIndexer selection and SparseFlashMLA.

[Invariants]
  * Compulsory pairs are ``sum_i attended_pairs(q_i, kv_i)``; batch totals are
    never cross-multiplied.
  * Causal decode ``q=1, kv=S`` counts all ``S`` keys; square causal prefill
    counts ``S*(S+1)//2``; chunked prefill ``q*P + q*(q+1)//2``.
  * ``active_causal_tiles`` counts exact non-empty ``(tile_q, tile_kv)`` blocks
    and is bound so algorithmic and tiled work stay consistent.
  * Mask bytes are charged only when a mask tensor is declared/bound.
  * LightningIndexer and SparseFlashMLA are distinct operators; sparse
    compulsory KV is bounded by unique visible cache tokens, and every byte
    count is non-negative and per-buffer aligned.

[Timeouts & Locks]
  Pure integer arithmetic over declared sequence/head counts; no tensor I/O.

[Side-Effect Policy]
  No mutation, no device access, no I/O.
"""

from __future__ import annotations

from typing import Any, Sequence

from .errors import FormulaValidationError
from .expr_ast import Const, Expr, ZERO, align_up, log2_ceil
from .hardware import HardwareProfile, cube_tier, dtype_bytes
from .primitives import (
    active_causal_tiles, align_up as align_up_int, attended_pairs, ceil_div,
    coalesce_efficiency,
    maybe_l2_hot, normalize_causal_offset, row_bytes, row_bytes_expr,
    strict_bool, strict_int, tensor_bytes, wave_efficiency,
)
from .work import Algebra, ByteAccount, OpWork


def _dims(q_len: int, kv_len: int, q_lens: Sequence[int] | None,
          kv_lens: Sequence[int] | None, batch: int
          ) -> tuple[list[int], list[int]]:
    if q_lens is None:
        q_values = [strict_int(q_len, "q_len", minimum=1)] * batch
    else:
        q_values = [strict_int(item, "q_lens entry", minimum=0) for item in q_lens]
        if len(q_values) != batch:
            raise FormulaValidationError(
                f"q_lens must have {batch} entries, got {len(q_values)}")
    if kv_lens is None:
        kv_values = [strict_int(kv_len, "kv_len", minimum=1)] * batch
    else:
        kv_values = [strict_int(item, "kv_lens entry", minimum=0) for item in kv_lens]
        if len(kv_values) != batch:
            raise FormulaValidationError(
                f"kv_lens must have {batch} entries, got {len(kv_values)}")
    if len(q_values) != len(kv_values):
        raise FormulaValidationError("q_lens and kv_lens must have equal length")
    if sum(q_values) <= 0:
        raise FormulaValidationError("zero active query tokens (T_active == 0)")
    if sum(kv_values) <= 0:
        raise FormulaValidationError("zero visible KV tokens")
    for index, (q_i, kv_i) in enumerate(zip(q_values, kv_values)):
        if q_i > q_len:
            raise FormulaValidationError(
                f"sequence {index}: q_len {q_i} exceeds static storage {q_len}")
        if kv_i > kv_len:
            raise FormulaValidationError(
                f"sequence {index}: kv_len {kv_i} exceeds static cache {kv_len}")
    return q_values, kv_values


def _visible_rows(q_len: int, kv_len: int, causal: bool,
                  offset: int) -> int:
    """KV rows a sequence can actually access under the declared causality."""
    if q_len <= 0 or kv_len <= 0:
        return 0
    if not causal:
        return kv_len
    return min(kv_len, max(0, offset + q_len))


def _visible_scan(q_len: int, kv_len: int, causal: bool, offset: int) -> int:
    """Exact number of visible query-key pairs, including an empty sequence."""
    if q_len <= 0 or kv_len <= 0:
        return 0
    if not causal:
        return q_len * kv_len
    return sum(min(kv_len, max(0, offset + position + 1))
               for position in range(q_len))


def _hot_bytes(hardware: HardwareProfile, steady: bool,
               sizes: Sequence[int]) -> int:
    if not steady:
        return 0
    budget = hardware.l2_hot_budget()
    hot = 0
    for size in sizes:
        grant = maybe_l2_hot(True, int(size), budget)
        hot += grant
        budget -= grant
    return hot


def attention_work(*, q_len: int = 1, kv_len: int = 1, hq: int = 32,
                   hkv: int = 1, head_dim: int = 128,
                   head_dim_v: int | None = None, batch: int = 1,
                   causal: bool = False, phase: str = "decode",
                   layout: str = "TND", dtype: str = "bf16",
                   kv_dtype: str | None = None,
                   q_lens: Sequence[int] | None = None,
                   kv_lens: Sequence[int] | None = None,
                   causal_offset: int | None = None,
                   has_attn_mask: bool | None = None,
                   attn_mask_shape: Sequence[int] | None = None,
                   mask_dtype: str = "uint8",
                   storage_q: int | None = None, storage_kv: int | None = None,
                   paged: bool = False, block_size: int = 128,
                   paged_blocks: int = 0, steady_l2_warm: bool = False,
                   hardware: HardwareProfile,
                   family: str = "flash_attention",
                   padding_skipped: bool = False) -> OpWork:
    """Build one prefill/decode/chunked/paged attention work record."""
    q_static = strict_int(q_len, "q_len", minimum=1)
    kv_static = strict_int(kv_len, "kv_len", minimum=1)
    hq_i = strict_int(hq, "hq", minimum=1)
    hkv_i = strict_int(hkv, "hkv", minimum=1)
    dk_i = strict_int(head_dim, "head_dim", minimum=1)
    dv_i = dk_i if head_dim_v is None else strict_int(head_dim_v, "head_dim_v",
                                                      minimum=1)
    b_i = strict_int(batch, "batch", minimum=1)
    causal_b = strict_bool(causal, "causal")
    declared_mask = (None if has_attn_mask is None
                     else strict_bool(has_attn_mask, "has_attn_mask"))
    if attn_mask_shape is not None and declared_mask is False:
        raise FormulaValidationError(
            "attn_mask_shape and has_attn_mask=False are contradictory metadata")
    mask_b = declared_mask is True or attn_mask_shape is not None
    if hq_i % hkv_i:
        raise FormulaValidationError("hq must be a multiple of hkv")
    if storage_q is not None and strict_int(storage_q, "storage_q", minimum=1) < q_static:
        raise FormulaValidationError("storage_q below declared q_len")
    if storage_kv is not None and strict_int(storage_kv, "storage_kv", minimum=1) < kv_static:
        raise FormulaValidationError("storage_kv below declared kv_len")
    q_values, kv_values = _dims(q_static, kv_static, q_lens, kv_lens, b_i)
    # Each sequence owns its cache-append offset: sequence order is irrelevant.
    offsets = [normalize_causal_offset(q_i, kv_i, causal_offset)
               for q_i, kv_i in zip(q_values, kv_values)]
    pair_total = sum(attended_pairs(q_i, kv_i, causal=causal_b,
                                    causal_offset=off_i)
                     for q_i, kv_i, off_i in zip(q_values, kv_values, offsets))
    offset = max(offsets) if offsets else 0
    tile_q, tile_kv = 16, 128
    if causal_b:
        active_tiles = sum(active_causal_tiles(q_i, kv_i, tile_q, tile_kv, off_i)
                           for q_i, kv_i, off_i in zip(q_values, kv_values, offsets))
        tiled_pairs = active_tiles * tile_q * tile_kv
    else:
        active_tiles = 0
        tiled_pairs = sum(align_up_int(q_i, tile_q) * align_up_int(kv_i, tile_kv)
                          for q_i, kv_i in zip(q_values, kv_values))
    total_q, total_kv = sum(q_values), sum(kv_values)
    max_kv = max(kv_values)
    visible_kv = sum(_visible_rows(q_i, kv_i, causal_b, off_i)
                     for q_i, kv_i, off_i in zip(q_values, kv_values, offsets))
    visible_pairs = sum(_visible_scan(q_i, kv_i, causal_b, off_i)
                        for q_i, kv_i, off_i in zip(q_values, kv_values, offsets))
    q_bpe = dtype_bytes(dtype)
    kv_bpe = dtype_bytes(kv_dtype or dtype)
    out_bpe = q_bpe
    mask_bpe = dtype_bytes(mask_dtype) if mask_b else 1.0

    al = Algebra()
    total_q_s = al.sym("total_q", total_q)
    total_kv_s = al.sym("total_kv", total_kv)
    hq_s = al.sym("hq", hq_i)
    hkv_s = al.sym("hkv", hkv_i)
    dk_s = al.sym("head_dim_qk", dk_i)
    dv_s = al.sym("head_dim_v", dv_i)
    batch_s = al.sym("batch", b_i)
    pairs_s = al.sym("attended_pairs", pair_total)
    tiled_pairs_s = al.sym("tiled_pairs", tiled_pairs)
    tiles_s = al.sym("active_causal_tiles", active_tiles)
    q_bytes_s = al.real("q_bytes", q_bpe)
    kv_bytes_s = al.real("kv_bytes_per_elem", kv_bpe)
    out_bytes_s = al.real("out_bytes", out_bpe)
    mask_s = al.sym("mask_bytes", 0)
    block_s = al.sym("block_bytes", 0)

    if mask_b:
        if attn_mask_shape is not None:
            mask_elems = 1
            for dim in attn_mask_shape:
                mask_elems *= strict_int(dim, "attn_mask_shape dim", minimum=1)
        else:
            mask_elems = total_q * max_kv
        mask_value = int(tensor_bytes((mask_elems,), mask_bpe))
        mask_s = al.sym("mask_bytes", mask_value)
        mask_elems_s = al.sym("mask_elems", mask_elems)
        mask_bpe_s = al.real("mask_bytes_per_elem", mask_bpe)
    else:
        mask_elems_s = ZERO
        mask_bpe_s = ZERO
    block_value = 0
    if paged:
        blocks = (strict_int(paged_blocks, "paged_blocks", minimum=0)
                  if paged_blocks else (b_i * ceil_div(max_kv, block_size)))
        block_value = blocks * 4
        block_s = al.sym("block_bytes", block_value)

    granule = hardware.granule_for(layout)
    q_elems = total_q_s * hq_s * dk_s
    out_elems = total_q_s * hq_s * dv_s
    visible_s = al.sym("visible_kv_rows", visible_kv)
    k_elems = visible_s * hkv_s * dk_s
    v_elems = visible_s * hkv_s * dv_s
    account = ByteAccount(granule)
    q_width = row_bytes_expr(dk_s, q_bpe)
    kv_width_k = row_bytes_expr(dk_s, kv_bpe)
    kv_width_v = row_bytes_expr(dv_s, kv_bpe)
    account.read_buffer(q_elems * q_bytes_s, 1.0,
                        rows=total_q_s * hq_s, width=q_width)
    account.read_buffer(k_elems * kv_bytes_s, 1.0,
                        rows=total_kv_s * hkv_s, width=kv_width_k)
    account.read_buffer(v_elems * kv_bytes_s, 1.0,
                        rows=total_kv_s * hkv_s, width=kv_width_v)
    if mask_b:
        account.read_buffer(mask_elems_s * mask_bpe_s, 1.0)
    if block_value:
        account.read_buffer(block_s, 1.0)
    account.write_buffer(out_elems * out_bytes_s, 1.0,
                         rows=total_q_s * hq_s,
                         width=row_bytes_expr(dv_s, out_bpe))

    kv_bytes = int(tensor_bytes((visible_kv, hkv_i, dk_i), kv_bpe)
                   + tensor_bytes((visible_kv, hkv_i, dv_i), kv_bpe))
    hot = _hot_bytes(hardware, steady_l2_warm,
                     (int(total_q * hq_i * dk_i * q_bpe),
                      int(total_q * hq_i * dv_i * out_bpe),
                      kv_bytes, block_value))
    reload_factor = 1.0
    if phase == "prefill" and kv_bytes > hardware.l2_hot_budget():
        reload_factor = float(min(4, ceil_div(kv_bytes, hardware.l2_hot_budget())))

    if phase == "prefill":
        n_tiles = max(1, b_i * hq_i * ceil_div(max(q_values), 128))
    elif hkv_i > 1:
        n_tiles = max(1, b_i * hkv_i)
    else:
        n_tiles = max(1, min(hardware.cores,
                             b_i * hq_i * ceil_div(max(q_values), 16)))
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    coalesce = coalesce_efficiency(row_bytes(dk_i, q_bpe), granule)
    macs = hq_s * pairs_s * (dk_s + dv_s)
    tiled = 2 * hq_s * tiled_pairs_s * (dk_s + dv_s)
    vector = 5 * hq_s * pairs_s + hq_s * total_q_s * dv_s
    bindings = al.merged()
    bindings.update({
        "S_q": total_q, "S_kv": total_kv, "q_len": q_values[0],
        "kv_len": kv_values[0], "total_q_lens": list(q_values),
        "total_kv_lens": list(kv_values), "causal": causal_b,
        "visible_kv_rows": visible_kv, "visible_pairs": visible_pairs,
        "causal_offset": offset, "causal_offsets": list(offsets),
        "has_attn_mask": mask_b,
        "kv_bytes": kv_bytes, "n_tiles": n_tiles, "n_waves": waves,
        "n_cores": hardware.cores, "wave_efficiency": wave_eff,
        "kv_reload_factor": reload_factor, "paged": bool(paged),
        "layout": layout, "dtype": str(dtype),
        "kv_dtype": str(kv_dtype or dtype), "tile_q": tile_q,
        "tile_kv": tile_kv,
    })
    notes = [
        f"per-sequence pairs: sum_i q_i*kv_i (B={b_i}, total_q={total_q}, "
        f"total_kv={total_kv})",
        "KV bytes charge hkv unique heads (GQA/MQA broadcast)",
        ("causal geometry: exact visible keys per query position"
         if causal_b else "full (non-causal) attention"),
    ]
    if padding_skipped:
        notes.append("padded rows skipped: valid actual_seq_lens reduced active tokens")
    if paged:
        notes.append(f"paged KV: block_size={block_size}, accessed blocks only")
    if mask_b:
        notes.append("explicit attention mask tensor charged")
    return OpWork(
        family=family,
        variant=f"{phase}_{layout}_{'causal' if causal_b else 'full'}",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot),
        n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=coalesce,
        m_tiled=align_up_int(total_q * hq_i, 16), n_tiled=align_up_int(max_kv, 16),
        k_tiled=align_up_int(dk_i, 16), tile_m=16, tile_n=tile_kv, tile_k=tile_q,
        capacity_reload_factor=reload_factor, cube_tier=cube_tier(dtype),
        notes=tuple(notes), bindings=bindings,
    )


def mla_prolog_work(*, tokens: int, hidden: int, num_heads: int,
                    head_dim: int, q_lora_rank: int, kv_lora_rank: int,
                    head_dim_v: int | None = None, rope_dim: int = 64,
                    dtype: str = "bf16", fused: bool = True,
                    steady_l2_warm: bool = False,
                    hardware: HardwareProfile) -> OpWork:
    """Fused MatMul + RMSNorm + RoPE + KV-cache scatter (MLAPO)."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(hidden, "hidden", minimum=1)
    hq_i = strict_int(num_heads, "num_heads", minimum=1)
    dk_i = strict_int(head_dim, "head_dim", minimum=1)
    dv_i = dk_i if head_dim_v is None else strict_int(head_dim_v, "head_dim_v",
                                                      minimum=1)
    ql_i = strict_int(q_lora_rank, "q_lora_rank", minimum=1)
    kvl_i = strict_int(kv_lora_rank, "kv_lora_rank", minimum=1)
    rope_i = strict_int(rope_dim, "rope_dim", minimum=0)
    bpe_i = dtype_bytes(dtype)
    fused_b = strict_bool(fused, "fused")
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    H = al.sym("hidden", h_i)
    HQ = al.sym("heads", hq_i)
    DK = al.sym("head_dim_qk", dk_i)
    DV = al.sym("head_dim_v", dv_i)
    QL = al.sym("q_lora", ql_i)
    KVL = al.sym("kv_lora", kvl_i)
    ROPE = al.sym("rope_dim", rope_i)
    BPE = al.real("dtype_bytes", bpe_i)
    account = ByteAccount(granule)
    # Four projections: W_dq [H,QL], W_uq_qr [QL,Hq*(D+R)],
    # per-head W_uk [Hq,D,KVL] and W_dkv_kr [H,KVL+R].
    account.read_buffer(T * H * BPE, 1.0, rows=T, width=row_bytes_expr(H, bpe_i))
    account.read_buffer(H * QL * BPE, 1.0, rows=H, width=row_bytes_expr(QL, bpe_i))
    account.read_buffer(QL * HQ * (DK + ROPE) * BPE, 1.0, rows=QL,
                        width=row_bytes_expr(HQ * (DK + ROPE), bpe_i))
    account.read_buffer(HQ * DK * KVL * BPE, 1.0, rows=HQ * DK,
                        width=row_bytes_expr(KVL, bpe_i))
    account.read_buffer(H * (KVL + ROPE) * BPE, 1.0, rows=H,
                        width=row_bytes_expr(KVL + ROPE, bpe_i))
    # Both RMSNorm gammas, RoPE sin/cos and the INT64 cache slot indices.
    account.read_buffer(QL * BPE, 1.0)
    account.read_buffer(KVL * BPE, 1.0)
    account.read_buffer(T * ROPE * 4, 1.0)
    account.read_buffer(T * 8, 1.0)
    if not fused_b:
        account.read_buffer(T * QL * BPE, 1.0, rows=T,
                            width=row_bytes_expr(QL, bpe_i))
    # q_nope/uk-projected compressed query, q_rope, and the T updated cache rows.
    account.write_buffer(T * HQ * KVL * BPE, 1.0, rows=T * HQ,
                         width=row_bytes_expr(KVL, bpe_i))
    account.write_buffer(T * HQ * ROPE * BPE, 1.0, rows=T * HQ,
                         width=row_bytes_expr(ROPE, bpe_i))
    account.write_buffer(T * KVL * BPE, 1.0, rows=T,
                         width=row_bytes_expr(KVL, bpe_i))
    account.write_buffer(T * ROPE * BPE, 1.0, rows=T,
                         width=row_bytes_expr(ROPE, bpe_i))
    if not fused_b:
        account.write_buffer(T * QL * BPE, 1.0, rows=T,
                             width=row_bytes_expr(QL, bpe_i))
    macs = (T * H * QL + T * QL * HQ * (DK + ROPE)
            + T * HQ * DK * KVL + T * H * (KVL + ROPE))
    tiled = 2 * (align_up(t_i, 16) * align_up(ql_i, 16) * align_up(h_i, 16)
                 + align_up(t_i, 16) * align_up(hq_i * (dk_i + rope_i), 16)
                 * align_up(ql_i, 16)
                 + align_up(t_i * hq_i, 16) * align_up(kvl_i, 16)
                 * align_up(dk_i, 16)
                 + align_up(t_i, 16) * align_up(kvl_i + rope_i, 16)
                 * align_up(h_i, 16))
    vector = 5 * T * QL + 5 * T * KVL + 3 * T * HQ * ROPE + 3 * T * ROPE
    n_tiles = max(1, t_i * hq_i)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    hot = _hot_bytes(hardware, steady_l2_warm,
                     (t_i * h_i * bpe_i, t_i * hq_i * kvl_i * bpe_i,
                      (ql_i + kvl_i) * bpe_i))
    return OpWork(
        family="mla_prolog", variant="fused" if fused_b else "unfused",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=n_tiles,
        n_cores=hardware.cores, coalesce_efficiency=1.0,
        m_tiled=align_up_int(t_i, 16), n_tiled=align_up_int(hq_i * (dk_i + rope_i), 16),
        k_tiled=align_up_int(h_i, 16), tile_m=16, tile_n=128, tile_k=128,
        cube_tier=cube_tier(dtype),
        notes=("fused buffer elision: no q_lora GM round trip",) if fused_b else
              ("unfused: q_lora intermediate is written and re-read",),
        bindings={**al.merged(), "n_tiles": n_tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


def _lengths(tokens: int, kv_len: int,
             q_lens: Sequence[int] | None, kv_lens: Sequence[int] | None
             ) -> tuple[list[int], list[int]]:
    if q_lens is None and kv_lens is None:
        return [1] * tokens, [strict_int(kv_len, "kv_len", minimum=1)] * tokens
    if q_lens is None or kv_lens is None:
        raise FormulaValidationError(
            "q_lens and kv_lens must be supplied together for a packed batch")
    q_values = [strict_int(item, "q_lens entry", minimum=0) for item in q_lens]
    kv_values = [strict_int(item, "kv_lens entry", minimum=1) for item in kv_lens]
    if len(q_values) != len(kv_values):
        raise FormulaValidationError("q_lens and kv_lens must have equal length")
    if sum(q_values) != tokens:
        raise FormulaValidationError(
            f"sum(q_lens)={sum(q_values)} must equal tokens={tokens}")
    if any(kv_i > int(kv_len) for kv_i in kv_values):
        raise FormulaValidationError(
            f"a kv_lens entry exceeds the static cache {kv_len}")
    return q_values, kv_values


def lightning_indexer_work(*, tokens: int, index_heads: int, index_dim: int,
                           kv_len: int, topk: int = 2048,
                           kv_lens: Sequence[int] | None = None,
                           q_lens: Sequence[int] | None = None,
                           output_index_heads: int | None = None,
                           causal: bool = False,
                           causal_offset: int | None = None,
                           dtype: str = "bf16",
                           kv_dtype: str | None = None,
                           steady_l2_warm: bool = False,
                           hardware: HardwareProfile) -> OpWork:
    """LightningIndexer: compressed index scoring + top-k index output."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(index_heads, "index_heads", minimum=1)
    idx_i = strict_int(index_dim, "index_dim", minimum=1)
    kv_i_static = strict_int(kv_len, "kv_len", minimum=1)
    k_i = strict_int(topk, "topk", minimum=1)
    out_h = h_i if output_index_heads is None else strict_int(
        output_index_heads, "output_index_heads", minimum=1)
    q_values, kv_values = _lengths(t_i, kv_i_static, q_lens, kv_lens)
    causal_b = strict_bool(causal, "causal")
    offsets = [normalize_causal_offset(q_i, kv_i, causal_offset)
               for q_i, kv_i in zip(q_values, kv_values)]
    scans = [_visible_scan(q_i, kv_i, causal_b, off_i)
             for q_i, kv_i, off_i in zip(q_values, kv_values, offsets)]
    pairs = sum(scans)
    visible_rows = [_visible_rows(q_i, kv_i, causal_b, off_i)
                    for q_i, kv_i, off_i in zip(q_values, kv_values, offsets)]
    tile_q, tile_kv = 16, 128
    tiled_tiles = 0
    for q_i, kv_i, off_i in zip(q_values, kv_values, offsets):
        if causal_b:
            tiled_tiles += active_causal_tiles(q_i, kv_i, tile_q, tile_kv, off_i)
        else:
            tiled_tiles += (ceil_div(q_i, tile_q) * ceil_div(kv_i, tile_kv))
    # Per-sequence tiled Cube work: independent sequences add, never merge.
    tiled_value = 2 * h_i * tiled_tiles * tile_q * tile_kv * idx_i
    eff_topk = [min(k_i, kv_value) for kv_value in kv_values]
    # Only causally visible positions are scored/compared; inaccessible cache
    # capacity is neither a compulsory read nor a selection-scan position.
    selection_ops = pairs
    select_scan = sum(
        scan * int(log2_ceil(max(2, min(k_i, kv_i))).value({}))
        for scan, kv_i in zip(scans, kv_values))
    bpe_i = dtype_bytes(dtype)
    kv_bpe = dtype_bytes(kv_dtype or dtype)
    index_bytes = sum(q * out_h * top * 4
                      for q, top in zip(q_values, eff_topk))
    granule = hardware.granule_for("ND")
    al = Algebra()
    HQ = al.sym("index_heads", h_i)
    IDX = al.sym("index_dim", idx_i)
    Q = al.sym("S_q", sum(q_values))
    KV = al.sym("S_kv", sum(visible_rows))
    PAIRS = al.sym("attended_pairs", pairs)
    OUT_H = al.sym("output_index_heads", out_h)
    TOPK = al.sym("eff_topk", eff_topk[0])
    SEL = al.sym("selection_ops", selection_ops)
    SCAN = al.sym("select_scan", select_scan)
    IDXB = al.sym("index_bytes", index_bytes)
    BPE = al.real("dtype_bytes", bpe_i)
    KVB = al.real("kv_bytes_per_elem", kv_bpe)
    account = ByteAccount(granule)
    account.read_buffer(Q * HQ * IDX * BPE, 1.0, rows=Q * HQ,
                        width=row_bytes_expr(IDX, bpe_i))
    account.read_buffer(KV * HQ * IDX * KVB, 1.0, rows=KV * HQ,
                        width=row_bytes_expr(IDX, kv_bpe))
    account.write_buffer(IDXB, 1.0)
    macs = PAIRS * HQ * IDX
    tiled = Const(tiled_value)
    vector = SEL + SCAN
    n_tiles = max(1, t_i * h_i * ceil_div(max(kv_values), 128))
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family="lightning_indexer", variant="topk_selection",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(idx_i, bpe_i), granule),
        m_tiled=align_up_int(sum(q_values), 16),
        n_tiled=align_up_int(max(kv_values), 16), k_tiled=align_up_int(idx_i, 16),
        tile_m=16, tile_n=128, tile_k=128, cube_tier=cube_tier(dtype),
        notes=(f"topk={k_i} clamped to visible kv per sequence",
               "per-sequence cube tiles are summed, never merged across sequences",
               "output is the int32 top-k index tensor, not attention output"),
        bindings={**al.merged(), "topk": k_i, "tokens": t_i,
                  "kv_len": kv_i_static, "active_causal_tiles": tiled_tiles,
                  "visible_kv_rows": sum(visible_rows),
                  "visible_pairs": pairs, "causal": causal_b,
                  "tile_q": tile_q, "tile_kv": tile_kv,
                  "causal_offsets": list(offsets),
                  "n_tiles": n_tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


def sparse_mla_work(*, tokens: int, num_heads: int, head_dim: int,
                    kv_len: int, topk: int = 2048,
                    head_dim_v: int | None = None,
                    kv_lens: Sequence[int] | None = None,
                    q_lens: Sequence[int] | None = None,
                    causal: bool = False, causal_offset: int | None = None,
                    dtype: str = "bf16", kv_dtype: str | None = None,
                    has_attn_mask: bool = False,
                    unique_selected_kv: int | Sequence[int] | None = None,
                    hardware: HardwareProfile,
                    steady_l2_warm: bool = False) -> OpWork:
    """SparseFlashMLA: sparse attention over ``topk`` selected KV entries."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    hq_i = strict_int(num_heads, "num_heads", minimum=1)
    dk_i = strict_int(head_dim, "head_dim", minimum=1)
    dv_i = dk_i if head_dim_v is None else strict_int(head_dim_v, "head_dim_v",
                                                      minimum=1)
    kv_i_static = strict_int(kv_len, "kv_len", minimum=1)
    k_i = strict_int(topk, "topk", minimum=1)
    causal_b = strict_bool(causal, "causal")
    q_values, kv_values = _lengths(t_i, kv_i_static, q_lens, kv_lens)
    offsets = [normalize_causal_offset(q_i, kv_i, causal_offset)
               for q_i, kv_i in zip(q_values, kv_values)]
    sparse_pairs = 0
    unknown_floor = 0
    expected_union = 0
    union_floors: list[int] = []
    union_ceilings: list[int] = []
    for q_i, kv_i, off_i in zip(q_values, kv_values, offsets):
        scans: list[int] = []
        for position in range(q_i):
            visible = (min(kv_i, max(0, off_i + position + 1)) if causal_b
                       else kv_i)
            scans.append(visible)
            sparse_pairs += min(k_i, visible)
        if not scans or kv_i <= 0:
            floor_i = ceiling_i = 0
        else:
            # Every query may reuse the same keys: the minimum union is the
            # largest single-query selection, and the maximum union is bounded
            # by the reachable prefix, never by the whole cache.
            floor_i = min(k_i, scans[-1])
            ceiling_i = min(scans[-1], sum(min(k_i, value) for value in scans))
        union_floors.append(floor_i)
        union_ceilings.append(ceiling_i)
        unknown_floor += floor_i
        expected_union += ceiling_i
    if unique_selected_kv is not None:
        if isinstance(unique_selected_kv, bool) or not isinstance(
                unique_selected_kv, (int, Sequence)) or isinstance(
                    unique_selected_kv, (str, bytes)):
            raise FormulaValidationError(
                "unique_selected_kv must be an integer or a sequence of "
                "integers")
        if isinstance(unique_selected_kv, int):
            declared = [strict_int(unique_selected_kv,
                                   "unique_selected_kv", minimum=0)
                        ] * len(q_values)
        else:
            declared = [strict_int(item, "unique_selected_kv entry", minimum=0)
                        for item in unique_selected_kv]
            if len(declared) != len(q_values):
                raise FormulaValidationError(
                    f"unique_selected_kv must have {len(q_values)} entries")
        for index, (value, floor_i, ceiling_i) in enumerate(
                zip(declared, union_floors, union_ceilings)):
            if not floor_i <= value <= ceiling_i:
                raise FormulaValidationError(
                    f"unique_selected_kv[{index}]={value} outside "
                    f"[{floor_i}, {ceiling_i}]")
        unique_kv = sum(declared)
    else:
        unique_kv = unknown_floor
    bpe_i = dtype_bytes(dtype)
    kv_bpe = dtype_bytes(kv_dtype or dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    HQ = al.sym("hq", hq_i)
    DK = al.sym("head_dim_qk", dk_i)
    DV = al.sym("head_dim_v", dv_i)
    Q = al.sym("total_q", sum(q_values))
    SPARSE = al.sym("sparse_pairs", sparse_pairs)
    UNIQUE = al.sym("unique_kv_tokens", unique_kv)
    IDXE = al.sym("index_elems", sum(q * hq_i * k_i for q in q_values))
    BPE = al.real("dtype_bytes", bpe_i)
    KVB = al.real("kv_bytes_per_elem", kv_bpe)
    account = ByteAccount(granule)
    kv_row = dk_i + dv_i
    account.read_buffer(Q * HQ * DK * BPE, 1.0, rows=Q * HQ,
                        width=row_bytes_expr(DK, bpe_i))
    account.read_buffer(UNIQUE * (DK + DV) * KVB, 1.0, rows=UNIQUE,
                        width=row_bytes_expr(DK + DV, kv_bpe))
    account.read_buffer(IDXE * 4, 1.0, rows=Q * HQ, width=k_i * 4)
    account.write_buffer(Q * HQ * DV * BPE, 1.0, rows=Q * HQ,
                         width=row_bytes_expr(DV, bpe_i))
    kv_row_bytes = row_bytes(kv_row, kv_bpe)
    gather = align_up(SPARSE * HQ * kv_row_bytes, granule)
    gather_value = align_up_int(sparse_pairs * hq_i * kv_row_bytes, granule)
    account.granule_read = account.granule_read + gather
    macs = HQ * SPARSE * (DK + DV)
    tiled = 2 * HQ * SPARSE * (DK + DV)
    vector = 5 * HQ * SPARSE
    n_tiles = max(1, sum(q_values) * hq_i * ceil_div(k_i, 128))
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family="sparse_flash_mla", variant="dsa_topk",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=1.0, m_tiled=align_up_int(sum(q_values) * hq_i, 16),
        n_tiled=align_up_int(k_i, 16), k_tiled=align_up_int(kv_row, 16),
        tile_m=16, tile_n=128, tile_k=128, cube_tier=cube_tier(dtype),
        notes=("compulsory unique KV is min(kv_i, topk) per sequence",
               "expected union min(kv_i, q_i*topk) stays in the reference gather",
               "granule traffic charges repeated gather transactions",
               "sparse attention only: indexer selection is a separate operator"),
        bindings={**al.merged(), "tokens": t_i, "kv_len": kv_i_static,
                  "topk": k_i, "gather_bytes": gather_value,
                  "expected_unique_kv_tokens": expected_union,
                  "causal_offsets": list(offsets),
                  "n_tiles": n_tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


__all__ = [
    "attention_work", "lightning_indexer_work", "mla_prolog_work",
    "sparse_mla_work",
]
