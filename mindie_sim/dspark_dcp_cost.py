"""Non-causal DSpark draft-block cost under DCP/PCP with TP head ownership.

[Role]
  Price the non-causal draft block's index preparation, DCP query gather,
  latent attention and output/LSE merge, using the same tensors as
  ``mla_cp.py`` and ``dspark_swa_indices.py``.

[Invariants]
  * The gather moves query NoPE + RoPE fragments, not draft KV; PCP-replicated
    query heads reduce the exchange to distinct TP heads.
  * The merge exchanges the packed gathered-head tensor, then PCP gathers
    that full resulting tensor (latent output + packed FP32 LSE); both phases
    come from the same CP shape helper as ``gathered_heads``.
  * One aligned index plane plus one per-row length vector (int32 or int64),
    bounded by ``query block + SWA window`` and scaled by batch and layers.
  * Latency sums gather, attention and combine along the causal dependency
    chain; it never hides one behind ``max``.
  * The combinatorial extra-pair identity ``B(B-1)/2`` holds on the unbounded
    causal path; with an SWA window causal and non-causal pairs coincide and
    ``extra_pairs`` is exactly zero.
  * All dimensions are strict integers; NaN, booleans and non-positive rates
    reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    ceil_div, dtype_elem_bytes, positive_rate, strict_bool, strict_dtype,
    strict_int, strict_number,
)
from mindie_sim.schema.common import ModelingError


@typed_inputs
@dataclass(frozen=True)
class DcpDraftBlockCost:
    """Non-causal draft block over DCP-local KV with PCP/TP head ownership."""

    draft_block_size: int
    context_tokens: int
    dcp: int
    pcp: int
    tp: int
    num_q_heads: int
    q_nope_dim: int
    q_rope_dim: int
    kv_heads: int
    head_dim: int
    latent_width: int
    num_draft_layers: int
    batch: int
    swa_window: int
    kv_dtype: str = "bf16"
    q_dtype: str = "bf16"
    out_dtype: str = "bf16"
    lse_dtype: str = "fp32"
    index_width_alignment: int = 128
    index_entry_bytes: int = 4
    index_length_bytes: int = 4
    index_length_dtype: str = "int32"
    physical_index_path: bool = False
    index_entries_per_us: float = 1000.0
    collective_bytes_per_s: float = 1.0e12
    hbm_bytes_per_s: float = 1.0e12
    compute_flops_per_s: float = 4.0e14
    kernel_fixed_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("draft_block_size", "dcp", "pcp", "tp", "num_q_heads",
                     "q_nope_dim", "q_rope_dim", "kv_heads", "head_dim",
                     "latent_width", "num_draft_layers", "batch",
                     "swa_window", "index_width_alignment",
                     "index_entry_bytes", "index_length_bytes"):
            minimum = 0 if name in {"q_rope_dim", "swa_window"} else 1
            strict_int(name, getattr(self, name), minimum=minimum,
                       reason="dspark")
        strict_int("context_tokens", self.context_tokens, minimum=0,
                   reason="dspark")
        for name in ("kv_dtype", "q_dtype", "out_dtype", "lse_dtype",
                     "index_length_dtype"):
            strict_dtype(name, getattr(self, name), reason="dspark")
        if self.out_dtype not in {"bf16", "fp16", "fp32", "bfloat16", "float16", "float32"} \
                or self.lse_dtype not in {"fp32", "float32"}:
            raise ModelingError("dspark_packed_output_lse_dtype_unsupported")
        strict_bool("physical_index_path", self.physical_index_path,
                    reason="dspark")
        if int(dtype_elem_bytes("index_length_dtype",
                                self.index_length_dtype,
                                reason="dspark")) != self.index_length_bytes:
            raise ModelingError("dspark_index_length_dtype_mismatch")
        if self.physical_index_path and self.index_length_dtype != "int64":
            raise ModelingError("dspark_physical_index_requires_int64_lengths")
        # Query heads are TP-partitioned before any collective decision.  A
        # floored partition is not a shard, so reject it before arithmetic.
        if self.num_q_heads % self.tp:
            raise ModelingError(
                "dspark_query_heads_not_divisible_by_tp:"
                f"{self.num_q_heads}%{self.tp}")
        if self.pcp > 1 and self.dcp < self.pcp:
            raise ModelingError("dspark_dcp_smaller_than_pcp")
        for name in ("index_entries_per_us", "collective_bytes_per_s",
                     "hbm_bytes_per_s", "compute_flops_per_s"):
            positive_rate(name, getattr(self, name), reason="dspark")
        if strict_number("kernel_fixed_ms", self.kernel_fixed_ms,
                         reason="dspark") is None:
            raise ModelingError("dspark_kernel_fixed_invalid")
        if self.index_width_alignment & (self.index_width_alignment - 1):
            raise ModelingError("dspark_index_alignment_not_power_of_two")

    @property
    def local_q_heads(self) -> int:
        return self.num_q_heads // self.tp

    @property
    def cp_span(self) -> int:
        """Distinct TP sources gathered by one PCP/DCP exchange.

        ``_dcp_all_gather_fragments`` returns the local tensor when
        ``dcp_size == pcp_size`` and otherwise gathers the full distinct TP
        head set over PCP (``dcp // pcp`` on the canonical ``tp * pcp``
        topology); a plain DCP gather spans ``dcp`` ranks on the TP1 path.
        One helper owns every derived head/collective shape.
        """
        if self.dcp <= 1:
            return 0
        if self.pcp > 1:
            if self.dcp == self.pcp:
                return 0
            # ``common_cp.py`` gathers the TP group, i.e. distinct TP heads
            # per source.  On the canonical ``dcp = tp * pcp`` topology this
            # equals ``dcp // pcp``; using the ratio alone would shrink an
            # explicitly declared TP world.
            return self.tp
        return self.dcp

    @property
    def gather_peers(self) -> int:
        """Peers whose query fragments this rank must gather."""
        if self.dcp <= 1:
            return 0
        if self.pcp > 1:
            return 0 if self.dcp == self.pcp else max(0, self.tp - 1)
        return self.dcp - 1

    @property
    def query_len(self) -> int:
        return self.draft_block_size

    @property
    def sequence_len(self) -> int:
        return self.context_tokens + self.query_len

    @property
    def local_context_tokens(self) -> int:
        return ceil_div(self.context_tokens, self.dcp)

    @property
    def start_pos(self) -> int:
        prefix = self.sequence_len - self.query_len
        return max(0, prefix - self.swa_window)

    @property
    def visible_tokens(self) -> int:
        return min(self.query_len + self.swa_window, self.sequence_len)

    @property
    def index_width(self) -> int:
        return (ceil_div(self.visible_tokens, self.index_width_alignment)
                * self.index_width_alignment)

    @property
    def rows(self) -> int:
        return self.batch * self.num_draft_layers * self.query_len

    @property
    def index_bytes(self) -> int:
        per_row = (self.index_width * self.index_entry_bytes
                   + self.index_length_bytes)
        return self.rows * per_row

    @property
    def query_elements_per_row(self) -> int:
        return (self.q_nope_dim + self.q_rope_dim) * self.local_q_heads

    @property
    def gather_bytes(self) -> float:
        return (float(self.gather_peers * self.batch * self.num_draft_layers
                      * self.query_len * self.query_elements_per_row)
                * dtype_elem_bytes("q_dtype", self.q_dtype, reason="dspark"))

    @property
    def gathered_heads(self) -> int:
        """Heads present after the gather, derived by the CP contract.

        ``dcp == pcp`` leaves a PCP-replicated query local; a wider PCP span
        gathers ``dcp // pcp`` distinct TP sources per local head; the plain
        DCP path gathers ``dcp`` query shards.
        """
        if self.cp_span <= 0:
            return self.local_q_heads
        return self.local_q_heads * self.cp_span

    @property
    def local_merge_payload_bytes(self) -> float:
        """Logical local-head view; not the tensor sent to all-to-all."""
        elements = float(self.batch * self.num_draft_layers * self.query_len
                         * self.local_q_heads)
        output = elements * self.latent_width * dtype_elem_bytes(
            "out_dtype", self.out_dtype, reason="dspark")
        lse = elements * dtype_elem_bytes(
            "lse_dtype", self.lse_dtype, reason="dspark")
        return output + lse

    @property
    def merge_payload_bytes(self) -> float:
        """Logical output + FP32 LSE before packing across gathered heads."""
        elements = float(self.batch * self.num_draft_layers * self.query_len
                         * self.gathered_heads)
        output = elements * self.latent_width * dtype_elem_bytes(
            "out_dtype", self.out_dtype, reason="dspark")
        lse = elements * dtype_elem_bytes(
            "lse_dtype", self.lse_dtype, reason="dspark")
        return output + lse

    @property
    def lse_pack_dim(self) -> int:
        """Source ``_lse_pack_dim``: four 16-bit elements or one FP32."""
        return 1 if self.out_dtype in {"fp32", "float32"} else 4

    @property
    def packed_merge_shape(self) -> tuple[int, int, int, int]:
        scatter = max(1, self.cp_span)
        return (scatter, self.gathered_heads // scatter, self.rows,
                self.latent_width + self.lse_pack_dim)

    @property
    def packed_merge_bytes(self) -> float:
        return (self.rows * self.gathered_heads
                * (self.latent_width + self.lse_pack_dim)
                * dtype_elem_bytes("out_dtype", self.out_dtype, reason="dspark"))

    @property
    def combine_tp_bytes(self) -> float:
        """TP head exchange share, applied to the pre-exchange tensor."""
        if self.dcp <= 1 or self.pcp <= 1 or self.dcp <= self.pcp:
            return 0.0
        return (self.tp - 1) / self.tp * self.packed_merge_bytes

    @property
    def combine_pcp_bytes(self) -> float:
        """PCP all-gather of the *resulting* contribution tensor."""
        if self.dcp <= 1 or self.pcp <= 1:
            return 0.0
        return (self.pcp - 1) * self.packed_merge_bytes

    @property
    def combine_bytes(self) -> float:
        """One-direction remote bytes of the packed contribution tensor.

        With PCP the source composes TP all_to_all_single and PCP AllGather
        (``common_cp.py:222-244``); a single ``(dcp-1)/dcp`` factor does not
        represent that composition, so the two shares are exposed separately.
        """
        if self.dcp <= 1:
            return 0.0
        if self.pcp > 1:
            return self.combine_tp_bytes + self.combine_pcp_bytes
        return (self.dcp - 1) / self.dcp * self.packed_merge_bytes

    @property
    def local_visible_tokens(self) -> int:
        """Window-bounded, DCP-sharded visible keys per query row."""
        if self.swa_window <= 0:
            return self.local_context_tokens + self.query_len
        return ceil_div(self.visible_tokens, self.dcp)

    @property
    def causal_pairs(self) -> int:
        block = self.query_len
        if self.swa_window > 0:
            return block * self.local_visible_tokens
        return block * self.local_context_tokens + block * (block + 1) // 2

    @property
    def noncausal_pairs(self) -> int:
        block = self.query_len
        if self.swa_window > 0:
            # Every draft query sees the whole visible window, including the
            # current draft block; the bound is per DCP shard.
            return block * self.local_visible_tokens
        return block * self.local_context_tokens + block * block

    @property
    def extra_pairs(self) -> int:
        return self.noncausal_pairs - self.causal_pairs

    @property
    def attention_flops(self) -> float:
        pairs = self.noncausal_pairs * self.batch * self.num_draft_layers
        dims = (self.q_nope_dim + self.q_rope_dim + self.latent_width)
        # Every rank prices the heads it actually attends after the gather.
        return float(4 * pairs * self.gathered_heads * dims)

    @property
    def kv_row_elements(self) -> int:
        """One MLA KV row: latent NoPE width plus its RoPE fragment."""
        return self.latent_width + self.q_rope_dim

    @property
    def attention_ms(self) -> float:
        flops = self.attention_flops / self.compute_flops_per_s * 1000.0
        kv_bytes = (float(self.noncausal_pairs * self.batch
                          * self.num_draft_layers * self.kv_row_elements)
                    * dtype_elem_bytes("kv_dtype", self.kv_dtype,
                                       reason="dspark"))
        return max(flops, kv_bytes / self.hbm_bytes_per_s * 1000.0)

    @property
    def gather_ms(self) -> float:
        return self.gather_bytes / self.collective_bytes_per_s * 1000.0

    @property
    def combine_ms(self) -> float:
        return self.combine_bytes / self.collective_bytes_per_s * 1000.0

    @property
    def index_prep_ms(self) -> float:
        return self.rows * self.index_width / self.index_entries_per_us / 1000.0

    def to_dict(self) -> dict[str, Any]:
        attn = self.attention_ms
        total = (self.kernel_fixed_ms + self.index_prep_ms + self.gather_ms
                 + attn + self.combine_ms)
        return {"local_context_tokens": self.local_context_tokens,
                "visible_tokens": self.visible_tokens,
                "local_visible_tokens": self.local_visible_tokens,
                "index_width": self.index_width, "rows": self.rows,
                "start_pos": self.start_pos,
                "local_q_heads": self.local_q_heads,
                "cp_span": self.cp_span,
                "gather_peers": self.gather_peers,
                "gather_bytes": self.gather_bytes,
                "combine_bytes": self.combine_bytes,
                "combine_tp_bytes": self.combine_tp_bytes,
                "combine_pcp_bytes": self.combine_pcp_bytes,
                "local_merge_payload_bytes": self.local_merge_payload_bytes,
                "merge_payload_bytes": self.merge_payload_bytes,
                "packed_merge_bytes": self.packed_merge_bytes,
                "packed_merge_shape": list(self.packed_merge_shape),
                "pcp_gather_shape": [self.packed_merge_shape[0] * self.pcp,
                                     *self.packed_merge_shape[1:]],
                "merge_path": "packed_all_to_all_then_pcp_all_gather",
                "gathered_heads": self.gathered_heads,
                "index_length_dtype": self.index_length_dtype,
                "index_entries": self.rows * self.index_width,
                "index_bytes": self.index_bytes,
                "causal_pairs": self.causal_pairs,
                "noncausal_pairs": self.noncausal_pairs,
                "extra_pairs": self.extra_pairs,
                "attention_flops": self.attention_flops,
                "gather_ms": self.gather_ms,
                "attention_ms": attn,
                "combine_ms": self.combine_ms,
                "index_prep_ms": self.index_prep_ms,
                "total_ms": total,
                "dependency_chain": ["gather", "attention", "combine"]}


@typed_inputs
def non_causal_block_drafter_cost(**kwargs: Any) -> dict[str, Any]:
    """Cost the non-causal block mask, index prep, DCP gather/attn/combine."""
    return DcpDraftBlockCost(**kwargs).to_dict()


__all__ = ["DcpDraftBlockCost", "non_causal_block_drafter_cost"]
