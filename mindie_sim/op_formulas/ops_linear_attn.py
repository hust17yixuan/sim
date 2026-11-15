"""Formula builders for chunk/recurrent linear attention and RecoverSSM.

[Role]
  Own the matrix-state physics of linear attention: a full recurrent state is
  ``H*Dk*Dv*state_bytes`` per sequence (never ``rank``-sized), and the recover
  path carries a separate compact record ``H*(Dv+2*Dk)*4`` per token.  Supports
  independent ``Dk``/``Dv``, explicit batch ownership, speculative slots and
  accepted-token commit semantics.

[Invariants]
  * A rank-one update never shrinks the state tensor: ``state_bytes_per_seq =
    H*Dk*Dv*state_dtype_bytes``.
  * ``KdaRecoverSsmVerify`` reads B initial full states, steps S slots and writes
    B*S compact records + verify outputs; it writes **no** full state matrix.
  * ``KdaRecoverSsmCommit`` reads B states + B*m compact records and writes B
    committed states (plus declared boundary-crossing pages).
  * Recurrent decode reads ``B*H*Dk*Dv*4``; speculative per-slot final states are
    written only when ``write_per_slot_states`` is true and ``S > 1``.
  * Chunk attention charges ``N_seq`` initial/final states for varlen batches.
  * CausalConv1d is Vector-only: zero Cube MACs and zero tiled Cube FLOPs.

[Timeouts & Locks]
  Pure arithmetic over declared state geometry; no tensor allocation.

[Side-Effect Policy]
  No mutation, no device access, no I/O.
"""

from __future__ import annotations

from typing import Any, Sequence

from .errors import FormulaValidationError
from .expr_ast import ZERO, Const, align_up
from .hardware import HardwareProfile, dtype_bytes
from .primitives import (
    align_up as align_up_int, ceil_div, coalesce_efficiency, decode_sequence_lengths,
    maybe_l2_hot, row_bytes, row_bytes_expr, strict_bool, strict_int,
    strict_state_dtype, tensor_bytes, wave_efficiency,
)
from .work import Algebra, ByteAccount, OpWork


def _state_bytes(h: int, dk: int, dv: int, state_bpe: float) -> int:
    return int(tensor_bytes((h, dk, dv), state_bpe))


def chunk_linear_attention_work(*, tokens: int, num_heads: int,
                                head_dim_k: int = 128,
                                head_dim_v: int | None = None,
                                batch: int | None = None,
                                actual_seq_lens: Sequence[int] | None = None,
                                chunk_size: int = 64, dtype: str = "bf16",
                                state_dtype: str = "fp32",
                                beta_elems: int | None = None,
                                beta_dtype: str | None = None,
                                gate_elems: int | None = None,
                                gate_dtype: str | None = None,
                                out_dtype: str | None = None,
                                prefix_replay: bool = False,
                                hardware: HardwareProfile,
                                family: str = "chunk_kda") -> OpWork:
    """ChunkKda / ChunkGatedDeltaRule prefill linear-attention work."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(num_heads, "num_heads", minimum=1)
    dk_i = strict_int(head_dim_k, "head_dim_k", minimum=1)
    dv_i = dk_i if head_dim_v is None else strict_int(head_dim_v, "head_dim_v",
                                                      minimum=1)
    lens = (None if actual_seq_lens is None else
            decode_sequence_lengths(actual_seq_lens, encoding="lengths",
                                    total_tokens=t_i,
                                    label="chunk actual_seq_lens"))
    if batch is not None:
        declared = strict_int(batch, "batch", minimum=1)
        if lens is not None and declared != len(lens):
            raise FormulaValidationError(
                f"batch={declared} != {len(lens)} supplied sequences")
        n_seq = declared
    elif lens is not None:
        n_seq = len(lens)
    else:
        n_seq = 1
    c_i = strict_int(chunk_size, "chunk_size", minimum=1)
    bpe_i = dtype_bytes(dtype)
    out_bpe = dtype_bytes(out_dtype or dtype)
    beta_bpe = dtype_bytes(beta_dtype or dtype)
    gate_bpe = dtype_bytes(gate_dtype or dtype)
    beta_n = (t_i * h_i * dk_i if beta_elems is None
              else strict_int(beta_elems, "beta_elems", minimum=0))
    gate_n = (t_i * h_i * dk_i if gate_elems is None
              else strict_int(gate_elems, "gate_elems", minimum=0))
    state_key = strict_state_dtype(state_dtype)
    sbpe = dtype_bytes(state_key)
    state = _state_bytes(h_i, dk_i, dv_i, sbpe)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    H = al.sym("heads", h_i)
    DK = al.sym("head_dim_k", dk_i)
    DV = al.sym("head_dim_v", dv_i)
    NSEQ = al.sym("sequences", n_seq)
    C = al.sym("chunk_size", c_i)
    BPE = al.real("dtype_bytes", bpe_i)
    SBPE = al.real("state_bytes", sbpe)
    account = ByteAccount(granule)
    account.read_buffer(T * H * DK * BPE, 1.0, rows=T * H, width=row_bytes_expr(DK, bpe_i))
    account.read_buffer(T * H * DK * BPE, 1.0, rows=T * H, width=row_bytes_expr(DK, bpe_i))
    account.read_buffer(T * H * DV * BPE, 1.0, rows=T * H, width=row_bytes_expr(DV, bpe_i))
    # beta/gate are charged from their declared shapes and precision: scalar
    # per head ([T,H]) is not a silently expanded [T,H,Dk] tensor.
    account.read_buffer(beta_n * al.real("beta_elem_bytes", beta_bpe), 1.0)
    account.read_buffer(gate_n * al.real("gate_elem_bytes", gate_bpe), 1.0)
    account.read_buffer(NSEQ * H * DK * DV * SBPE, 1.0,
                        rows=NSEQ * H, width=DK * DV * SBPE)
    if prefix_replay:
        account.read_buffer(NSEQ * H * DK * DV * SBPE, 1.0,
                            rows=NSEQ * H, width=DK * DV * SBPE)
    account.write_buffer(T * H * DV * al.real("out_bytes", out_bpe), 1.0,
                         rows=T * H, width=row_bytes_expr(DV, out_bpe))
    account.write_buffer(NSEQ * H * DK * DV * SBPE, 1.0,
                         rows=NSEQ * H, width=DK * DV * SBPE)
    macs = T * H * (C * DK + DK * DV)
    tiled = 2 * (align_up(T * H, 16) * align_up(DK, 16) * align_up(C, 16)
                 + align_up(T * H, 16) * align_up(DK, 16) * align_up(DV, 16))
    vector = 6 * T * H * DK
    n_tiles = max(1, t_i * h_i * ceil_div(c_i, 16))
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family=family, variant="prefix_replay" if prefix_replay else "prefill",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=1.0, m_tiled=align_up_int(t_i * h_i, 16),
        n_tiled=align_up_int(dk_i, 16), k_tiled=align_up_int(dv_i, 16),
        tile_m=16, tile_n=128, tile_k=128, cube_tier="bf16",
        notes=("linear attention: full matrix state is compulsory",
               "prefix replay re-reads the carried state"),
        bindings={**al.merged(), "state_bytes_per_seq": state,
                  "beta_bytes": int(beta_n * beta_bpe),
                  "gate_bytes": int(gate_n * gate_bpe),
                  "qkv_bytes": int(tensor_bytes((t_i, h_i, 2 * dk_i + dv_i), bpe_i)),
                  "n_tiles": n_tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


def recurrent_linear_attention_work(*, tokens: int, num_heads: int,
                                    head_dim_k: int = 128,
                                    head_dim_v: int | None = None,
                                    batch: int | None = None,
                                    dtype: str = "bf16",
                                    out_dtype: str | None = None,
                                    beta_elems: int | None = None,
                                    beta_dtype: str | None = None,
                                    gate_elems: int | None = None,
                                    gate_dtype: str | None = None,
                                    state_dtype: str = "fp32",
                                    spec_slots: int = 1,
                                    write_per_slot_states: bool = False,
                                    hardware: HardwareProfile,
                                    family: str = "recurrent_kda") -> OpWork:
    """RecurrentKda / FusedRecurrentGdn decode and (K+1)-slot spec verify."""
    b_i = (strict_int(tokens, "tokens", minimum=1) if batch is None
           else strict_int(batch, "batch", minimum=1))
    h_i = strict_int(num_heads, "num_heads", minimum=1)
    dk_i = strict_int(head_dim_k, "head_dim_k", minimum=1)
    dv_i = dk_i if head_dim_v is None else strict_int(head_dim_v, "head_dim_v",
                                                      minimum=1)
    slots = strict_int(spec_slots, "spec_slots", minimum=1)
    per_slot = strict_bool(write_per_slot_states, "write_per_slot_states")
    bpe_i = dtype_bytes(dtype)
    out_bpe = dtype_bytes(out_dtype or dtype)
    beta_n = (b_i * slots * h_i if beta_elems is None
              else strict_int(beta_elems, "beta_elems", minimum=0))
    gate_n = (b_i * slots * h_i * dk_i if gate_elems is None
              else strict_int(gate_elems, "gate_elems", minimum=0))
    beta_bpe = dtype_bytes(beta_dtype or dtype)
    gate_bpe = dtype_bytes(gate_dtype or dtype)
    state_key = strict_state_dtype(state_dtype)
    sbpe = dtype_bytes(state_key)
    granule = hardware.granule_for("ND")
    al = Algebra()
    B = al.sym("batch", b_i)
    S = al.sym("spec_slots", slots)
    H = al.sym("heads", h_i)
    DK = al.sym("head_dim_k", dk_i)
    DV = al.sym("head_dim_v", dv_i)
    BPE = al.real("dtype_bytes", bpe_i)
    SBPE = al.real("state_bytes", sbpe)
    account = ByteAccount(granule)
    account.read_buffer(B * S * H * DK * BPE, 1.0, rows=B * S * H,
                        width=row_bytes_expr(DK, bpe_i))
    account.read_buffer(B * S * H * DK * BPE, 1.0, rows=B * S * H,
                        width=row_bytes_expr(DK, bpe_i))
    account.read_buffer(B * S * H * DV * BPE, 1.0, rows=B * S * H,
                        width=row_bytes_expr(DV, bpe_i))
    account.read_buffer(gate_n * al.real("gate_elem_bytes", gate_bpe), 1.0)
    account.read_buffer(beta_n * al.real("beta_elem_bytes", beta_bpe), 1.0)
    account.read_buffer(B * H * DK * DV * SBPE, 1.0, rows=B * H,
                        width=DK * DV * SBPE)
    account.write_buffer(B * S * H * DV * al.real("out_bytes", out_bpe), 1.0,
                         rows=B * S * H, width=row_bytes_expr(DV, out_bpe))
    slot_writes = slots > 1 and per_slot
    state_outs = b_i * (slots if slot_writes else 1)
    account.write_buffer(state_outs * H * DK * DV * SBPE, 1.0,
                         rows=state_outs * H, width=DK * DV * SBPE)
    # Gated Delta Network recurrence on the FP32 [Dv, Dk] state, per
    # (token, head):
    #   key contraction      Dv*Dk mul + Dv*(Dk-1) add  = 2*Dk*Dv
    #   delta = b*(v-a*vh)   3*Dv + Dk
    #   S' = a*S + outer(d,k) Dk*Dv + 2*Dk*Dv           = 3*Dk*Dv
    #   o = S' @ q           2*Dk*Dv
    # Total >= 7*Dk*Dv + 3*Dv + Dk.  Every term is a Vector-lane scalar op.
    per_step = 7 * DK * DV + 3 * DV + DK
    vector = B * S * H * per_step
    macs = ZERO
    tiled = ZERO
    n_tiles = max(1, b_i * slots * h_i)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    notes = (("spec_slots>1: per-slot states are written for every slot",)
             if slot_writes else
             ("decode: one committed state per sequence",))
    notes = notes + (
        "FP32 matrix-vector recurrence executes on Vector, not BF16 Cube",
        "per (token,head): 7*Dk*Dv + 3*Dv + Dk scalar operations")
    return OpWork(
        family=family, variant="spec_verify" if slots > 1 else "decode",
        cube_macs=macs, cube_flops_tiled=tiled, vector_ops=vector,
        alg_override=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=1.0, m_tiled=align_up_int(b_i * slots * h_i, 16),
        n_tiled=align_up_int(dk_i, 16), k_tiled=align_up_int(dv_i, 16),
        tile_m=16, tile_n=128, tile_k=128, cube_tier="bf16", notes=notes,
        structure="vector_only",
        bindings={**al.merged(), "write_per_slot_states": per_slot,
                  "state_outs": state_outs,
                  "state_bytes_per_seq": _state_bytes(h_i, dk_i, dv_i, sbpe),
                  "recurrence_ops_per_token_head": int(7 * dk_i * dv_i
                                                      + 3 * dv_i + dk_i),
                  "n_tiles": n_tiles, "n_waves": waves,
                  "wave_efficiency": wave_eff},
    )


def recover_ssm_work(*, tokens: int, num_heads: int, head_dim_k: int = 128,
                     head_dim_v: int | None = None, spec_slots: int = 1,
                     accepted_tokens: int | None = None, phase: str = "verify",
                     dtype: str = "bf16", state_dtype: str = "fp32",
                     align_boundary_crossings: int = 0,
                     hardware: HardwareProfile) -> OpWork:
    """KdaRecoverSsmVerify / Commit: full matrix state + compact records."""
    b_i = strict_int(tokens, "tokens", minimum=1)
    h_i = strict_int(num_heads, "num_heads", minimum=1)
    dk_i = strict_int(head_dim_k, "head_dim_k", minimum=1)
    dv_i = dk_i if head_dim_v is None else strict_int(head_dim_v, "head_dim_v",
                                                      minimum=1)
    slots = strict_int(spec_slots, "spec_slots", minimum=1)
    if phase not in {"verify", "commit"}:
        raise FormulaValidationError("recover_ssm phase must be 'verify' or 'commit'")
    bpe_i = dtype_bytes(dtype)
    sbpe = dtype_bytes(strict_state_dtype(state_dtype))
    if phase == "commit":
        m_i = slots if accepted_tokens is None else strict_int(
            accepted_tokens, "accepted_tokens", minimum=1)
        if m_i > slots:
            raise FormulaValidationError(
                f"accepted_tokens={m_i} exceeds spec_slots={slots}")
    else:
        m_i = slots
    state_per_seq = _state_bytes(h_i, dk_i, dv_i, sbpe)
    # Compact record: v_new (Dv) + k (Dk) + g (Dk), all FP32.
    compact_per_token = int(tensor_bytes((h_i, dv_i + 2 * dk_i), 4))
    boundary = strict_int(align_boundary_crossings, "align_boundary_crossings",
                          minimum=0)
    granule = hardware.granule_for("ND")
    al = Algebra()
    B = al.sym("batch", b_i)
    H = al.sym("heads", h_i)
    DK = al.sym("head_dim_k", dk_i)
    DV = al.sym("head_dim_v", dv_i)
    S = al.sym("spec_slots", slots)
    M = al.sym("accepted_tokens", m_i)
    SBPE = al.real("state_bytes", sbpe)
    BPE = al.real("dtype_bytes", bpe_i)
    account = ByteAccount(granule)
    account.read_buffer(B * H * DK * DV * SBPE, 1.0, rows=B * H,
                        width=DK * DV * SBPE)
    if phase == "verify":
        # q, k, v, gate for every verify slot of every sequence.
        account.read_buffer(B * S * H * (2 * DK + DV) * BPE, 1.0,
                            rows=B * S * H, width=row_bytes_expr(2 * DK + DV, bpe_i))
        account.write_buffer(B * S * H * (DV + 2 * DK) * 4, 1.0,
                             rows=B * S * H, width=row_bytes_expr(DV + 2 * DK, 4.0))
        account.write_buffer(B * S * H * DV * BPE, 1.0, rows=B * S * H,
                             width=row_bytes_expr(DV, bpe_i))
        macs = B * S * H * DK * DV
        vector = 4 * B * S * H * DK * DV
        read_note = "B full states + verify inputs"
        write_note = "B*S compact records + verify outputs (no full state write)"
        variant = "verify"
    else:
        account.read_buffer(B * M * H * (DV + 2 * DK) * 4, 1.0,
                            rows=B * M * H, width=row_bytes_expr(DV + 2 * DK, 4.0))
        account.write_buffer(B * H * DK * DV * SBPE, 1.0, rows=B * H,
                             width=DK * DV * SBPE)
        if boundary:
            account.write_buffer(B * boundary * H * DK * DV * SBPE, 1.0,
                                 rows=B * boundary * H, width=DK * DV * SBPE)
        macs = B * M * H * DK * DV
        vector = 2 * B * M * H * DK * DV
        read_note = "B full states + B*m accepted compact records"
        write_note = "B committed full states (+ boundary pages)"
        variant = "commit"
    n_tiles = max(1, h_i)
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    return OpWork(
        family="kda_recover_ssm", variant=variant,
        cube_macs=macs,
        cube_flops_tiled=2 * align_up(B * M * H, 16) * align_up(DK, 16)
        * align_up(DV, 16),
        vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=ZERO, n_tiles=n_tiles, n_cores=hardware.cores,
        coalesce_efficiency=1.0, m_tiled=align_up_int(b_i * m_i, 16),
        n_tiled=align_up_int(dk_i, 16), k_tiled=align_up_int(dv_i, 16),
        tile_m=16, tile_n=128, tile_k=128, cube_tier="bf16",
        notes=(read_note, write_note,
               "matrix state is H*Dk*Dv*4; compact record is H*(Dv+2*Dk)*4"),
        bindings={**al.merged(), "phase": phase,
                  "state_bytes_per_seq": state_per_seq,
                  "compact_record_bytes": compact_per_token,
                  "boundary_crossings": boundary, "n_tiles": n_tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff},
    )


def causal_conv1d_work(*, tokens: int, channels: int, width: int = 4,
                       run_mode: int = 0, batch: int | None = None,
                       actual_seq_lens: Sequence[int] | None = None,
                       dtype: str = "bf16",
                       steady_l2_warm: bool = False,
                       hardware: HardwareProfile) -> OpWork:
    """CausalConv1d prefill (run_mode=0) and decode ring-state update (1)."""
    t_i = strict_int(tokens, "tokens", minimum=1)
    c_i = strict_int(channels, "channels", minimum=1)
    w_i = strict_int(width, "width", minimum=1)
    mode = strict_int(run_mode, "run_mode", minimum=0)
    if mode not in {0, 1}:
        raise FormulaValidationError("run_mode must be 0 (prefill) or 1 (decode)")
    lens = (None if actual_seq_lens is None else
            decode_sequence_lengths(actual_seq_lens, encoding="lengths",
                                    total_tokens=t_i,
                                    label="conv actual_seq_lens"))
    if batch is not None:
        n_seq = strict_int(batch, "batch", minimum=1)
        if lens is not None and n_seq != len(lens):
            raise FormulaValidationError(
                f"batch={n_seq} != {len(lens)} supplied sequences")
    elif lens is not None:
        n_seq = len(lens)
    elif mode == 1:
        n_seq = t_i
    else:
        n_seq = 1
    bpe_i = dtype_bytes(dtype)
    granule = hardware.granule_for("ND")
    al = Algebra()
    T = al.sym("tokens", t_i)
    C = al.sym("channels", c_i)
    W = al.sym("width", w_i)
    NSEQ = al.sym("sequences", n_seq)
    BPE = al.real("dtype_bytes", bpe_i)
    account = ByteAccount(granule)
    account.read_buffer(T * C * BPE, 1.0, rows=T, width=row_bytes_expr(C, bpe_i))
    account.read_buffer(W * C * BPE, 1.0, rows=W, width=row_bytes_expr(C, bpe_i))
    if mode == 1:
        # Every sequence owns a private ring state; weights are shared.
        account.read_buffer(NSEQ * (W - 1) * C * BPE, 1.0, rows=NSEQ * (W - 1),
                            width=row_bytes_expr(C, bpe_i))
        account.write_buffer(NSEQ * (W - 1) * C * BPE, 1.0,
                             rows=NSEQ * (W - 1),
                             width=row_bytes_expr(C, bpe_i))
    account.write_buffer(T * C * BPE, 1.0, rows=T, width=row_bytes_expr(C, bpe_i))
    vector = T * C * W
    n_tiles = max(1, ceil_div(t_i * c_i, 8192))
    waves, wave_eff = wave_efficiency(n_tiles, hardware.cores)
    weight_bytes = int(tensor_bytes((w_i, c_i), bpe_i))
    ring_bytes = int(tensor_bytes((n_seq, w_i - 1, c_i), bpe_i)) if mode == 1 else 0
    hot = maybe_l2_hot(steady_l2_warm, weight_bytes + ring_bytes,
                       hardware.l2_hot_budget())
    return OpWork(
        family="causal_conv1d", variant="decode_ring" if mode == 1 else "prefill",
        cube_macs=ZERO, cube_flops_tiled=ZERO, vector_ops=vector,
        read_bytes=account.read, write_bytes=account.write,
        granule_read_bytes=account.granule_read,
        granule_write_bytes=account.granule_write,
        l2_hot_bytes=Const(hot), n_tiles=n_tiles,
        n_cores=hardware.cores,
        coalesce_efficiency=coalesce_efficiency(row_bytes(c_i, bpe_i), granule),
        m_tiled=align_up_int(t_i, 16), n_tiled=align_up_int(c_i, 16),
        k_tiled=align_up_int(w_i, 16), tile_m=1, tile_n=128, tile_k=1,
        structure="vector_only",
        notes=("Vector-only depthwise kernel: Cube is idle",
               "run_mode=1 updates one private ring state per sequence") if mode == 1 else
              ("Vector-only depthwise kernel: Cube is idle",
               "run_mode=0 computes the full causal convolution"),
        bindings={**al.merged(), "run_mode": mode, "sequences": n_seq,
                  "n_tiles": n_tiles,
                  "n_waves": waves, "wave_efficiency": wave_eff,
                  "weight_bytes": weight_bytes, "ring_bytes": ring_bytes},
    )


__all__ = [
    "causal_conv1d_work", "chunk_linear_attention_work", "recover_ssm_work",
    "recurrent_linear_attention_work",
]
