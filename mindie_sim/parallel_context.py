"""Dual-phase PCP+DCP plans and real mrv2 sparse PCP builders.

[Role]
  Resolve the two independent phase fields (``prefill_cp_plan`` in
  ``none|pcp|dsa_cp`` and ``decode_cp_plan`` in ``none|dcp``), validate the
  resolved plan against the model's actual attention families, and register
  concrete mrv2 builders for DSA, SWA, CSA, CSA2 and HCA.  A builder declares
  query ownership, retained K/V rows, sparse-index ownership, boundary halos,
  communication dependencies and a conserved valid-pair count.

[Invariants]
  * A plan name is not a capability proof.  ``dsa_cp`` requires a real sparse
    indexer and declared top-k; mrv2 PCP requires at least one registered
    sparse family builder.  Supplied ``attention_families`` labels cannot
    replace the model's actual families.
  * For request query set ``Q`` and admissible keys ``J(q)``, each PCP rank
    owns a disjoint query chunk and ``sum_r A_r = A``.  Dense FLOPs divided by
    PCP is not a substitute for sparse top-k/window/compression accounting.
  * ``pcp_index_k_gather`` precedes index consumers and ``_pcp_kv_gather``
    precedes attention over the current chunk.  Their payloads are separate;
    uniform ownership receives ``S*(PCP-1)/PCP`` bytes.
  * DCP accounts max, sum and numerator reductions; it never allgathers the
    full KV cache.  Empty ranks contribute zero without ``exp(-inf-(-inf))``.
  * A PCP-prefill to DCP-decode transition declares ownership, bytes, epoch
    and barrier; replica-lane activation also pays KV/index/state remap.

[Timeouts & Locks] Pure arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Returns fresh immutable records; no global mutation
  except the explicit builder registry owned by this module.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

from mindie_sim.catalog.context_parallel import (pcp_query_tokens,
                                                pcp_rank_chunk_ranges,
                                                validate_cp_plan)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.parallel_bridge import (PhaseBridge, bridge_remap_bytes,
                                        phase_bridge)
from mindie_sim.schema.parallel import ParallelSpec

SPARSE_FAMILIES = ("dsa", "swa", "csa", "csa2", "hca")
PRE_REFILL_PLANS = ("none", "pcp", "dsa_cp")
DECODE_PLANS = ("none", "dcp")
ABI = "mrv2_kv_index_pair_v1"


@dataclass(frozen=True)
class PcpBuilder:
    family: str
    runtime_revision: str
    phase: str
    plan: str
    abi: str
    query_owner: str
    kv_owner: str
    index_owner: str
    comm_ops: tuple[str, ...]
    halo_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


_BUILDERS: dict[tuple[str, str, str, str], PcpBuilder] = {}


def _register(family: str, phase: str, plan: str, query: str, kv: str,
              index: str, ops: Sequence[str], halo: int = 0) -> None:
    _BUILDERS[(family, "mrv2", phase, plan)] = PcpBuilder(
        family, "mrv2", phase, plan, ABI, query, kv, index, tuple(ops), int(halo))


for _family in SPARSE_FAMILIES:
    _register(_family, "prefill", "pcp", "query_dual_chunk_swap",
              "gathered_current_chunk" if _family != "swa" else "window_ring",
              "whole_block_interleave", ("pcp_index_k_gather", "_pcp_kv_gather"), 0)
    _register(_family, "decode", "dcp", "decode_query_replicated",
              "kv_shard_by_dcp", "local_index_shard",
              ("dcp_kv_shard", "dcp_lse_merge"), 0)


def register_pcp_builder(builder: PcpBuilder) -> None:
    """Install an explicit builder (versioned, never a SKU-name fallback)."""
    if builder.family not in SPARSE_FAMILIES and builder.plan != "pcp":
        raise ModelingError(f"pcp_builder_family_unsupported:{builder.family}")
    _BUILDERS[(builder.family, builder.runtime_revision, builder.phase,
               builder.plan)] = builder


def registered_builders() -> tuple[PcpBuilder, ...]:
    return tuple(_BUILDERS[key] for key in sorted(_BUILDERS))


def family_of_layer(plan: Any) -> str:
    """One layer's actual family, from its own architecture/compression fields."""
    architecture = str(getattr(plan, "architecture", "")).lower()
    attention = str(getattr(plan, "attention", "gqa")).lower()
    ratio = int(getattr(plan, "compress_ratio", 0) or 0)
    compressed = ratio > 0 or str(
        getattr(plan, "cache_kind", "")).lower() == "window_compressed"
    if "hca" in architecture or getattr(plan, "hyper_connections", False):
        return "hca"
    if "dsa" in architecture:
        return "dsa"
    if "csa2" in architecture or (compressed and ratio == 2):
        return "csa2"
    if "csa" in architecture or compressed:
        return "csa"
    if int(getattr(plan, "window", 0) or 0) > 0:
        return "swa"
    if bool(getattr(plan, "indexer", False)) or \
            str(getattr(plan, "indexer_type", "none")).lower() not in {"none", ""}:
        return "dsa"
    return attention


def model_families(model: Any) -> tuple[str, ...]:
    """Actual per-layer attention families, never caller-supplied labels."""
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    if not plans:
        return (str(getattr(model, "attention", "gqa")).lower(),)
    return tuple(sorted({family_of_layer(plan) for plan in plans}))


def _dtype_bytes(model: Any, field_name: str, default: str = "bf16") -> float:
    return float(dtype_bytes(str(getattr(model, field_name, default) or default)))


def _kv_shape(model: Any) -> tuple[int, int, int]:
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    heads = max((int(getattr(plan, "num_kv_heads", 0) or 0) for plan in plans),
                default=int(getattr(model, "num_kv_heads", 1) or 1))
    head_dim = max((int(getattr(plan, "head_dim", 0) or 0) for plan in plans),
                   default=int(getattr(model, "head_dim", 128) or 128))
    layers = len(plans) or int(getattr(model, "layers", 1) or 1)
    return max(1, heads), max(1, head_dim), max(1, layers)


def _index_shape(model: Any) -> tuple[int, int]:
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    dims = [int(getattr(plan, "index_dim", 0) or 0) for plan in plans]
    topk = [int(getattr(plan, "index_topk", 0) or 0) for plan in plans]
    return max(dims or [0]), max(topk or [0])


def _query_ranges(q_len: int, pcp: int) -> tuple[tuple[int, int], ...]:
    chunks = pcp_query_tokens(q_len, pcp)
    result, start = [], 0
    for chunk in chunks:
        result.append((start, start + int(chunk)))
        start += int(chunk)
    if start != q_len:
        raise ModelingError(f"pcp_query_chunks_not_conserved:{start}!={q_len}")
    return tuple(result)


def _valid_keys(family: str, position: int, window: int, top_k: int,
                compress_ratio: int, causal: bool, context_len: int) -> int:
    full_keys = max(0, int(context_len))
    causal_keys = min(full_keys, position + 1) if causal else full_keys
    if family == "swa":
        return min(causal_keys, window) if window > 0 else causal_keys
    if family in {"csa", "csa2"}:
        ratio = max(2, compress_ratio or (2 if family == "csa2" else 4))
        compressed = 1 + position // ratio
        return min(top_k, compressed) if top_k > 0 else compressed
    if family == "dsa":
        keys = causal_keys
        if window > 0:
            keys = min(keys, window)
        return min(top_k, keys) if top_k > 0 else keys
    return causal_keys


def pcp_attention_pairs(*, isl: int, q_tokens: int, pcp: int, family: str,
                        index_topk: int = 0, sliding_window: int = 0,
                        compress_ratio: int = 0,
                        load_balance: str = "DualChunkSwap",
                        causal: bool = True) -> dict[str, Any]:
    """Family-specific admissible pairs owned by every PCP lane.

    Dense ``GQA``/``MLA`` with ``PCP=1`` reproduce the exact causal triangle
    ``q*(q+1)/2``.  Sparse index/window/compression families use
    :func:`valid_pairs`, so ``index_topk=1`` and ``index_topk=512`` no longer
    share one dense triangular pair count.
    """
    context = max(1, int(isl))
    query = min(context, max(0, int(q_tokens)))
    lanes = max(1, int(pcp))
    name = str(family or "gqa").strip().lower()
    top_k = max(0, int(index_topk))
    window = max(0, int(sliding_window))
    ratio = max(0, int(compress_ratio))
    policy = str(load_balance or "DualChunkSwap")
    if lanes == 1 or policy.lower() not in {"dualchunkswap", "dual_chunk_swap", "dcs"}:
        bounds = [round(index * query / lanes) for index in range(lanes + 1)]
        ranges = [((bounds[index], bounds[index + 1]),) if bounds[index + 1] > bounds[index]
                  else () for index in range(lanes)]
        offset = context - query
        per_lane = tuple(
            sum(_valid_keys(name, offset + index, window, top_k, ratio,
                            causal, context)
                for start, end in lane_ranges for index in range(start, end))
            for lane_ranges in ranges)
    else:
        per_lane = tuple(
            valid_pairs(name, query, context, lanes, rank, window=window,
                        top_k=top_k, compress_ratio=ratio, causal=causal)
            for rank in range(lanes))
    return {"family": name, "pcp": lanes, "load_balance": policy,
            "per_lane_pairs": per_lane, "total_pairs": int(sum(per_lane)),
            "critical_pairs": int(max(per_lane, default=0)),
            "q_tokens": query, "context_tokens": context,
            "index_topk": top_k, "sliding_window": window,
            "compress_ratio": ratio}


def _expected_pairs(family: str, q_len: int, context: int, *, window: int = 0,
                    top_k: int = 0, compress_ratio: int = 0, causal: bool = True,
                    query_offset: int | None = None) -> int:
    """Ground truth pair count independent of any per-rank decomposition."""
    base = max(0, int(context) - int(q_len)) if query_offset is None else int(query_offset)
    total = 0
    for index in range(max(0, int(q_len))):
        total += _valid_keys(family, base + index, int(window), int(top_k),
                             int(compress_ratio), bool(causal), int(context))
    return total


def valid_pairs(family: str, q_len: int, context: int, pcp: int, rank: int, *,
                window: int = 0, top_k: int = 0, compress_ratio: int = 0,
                causal: bool = True, query_offset: int | None = None) -> int:
    """Exact admissible pair count for one PCP rank's catalog-owned chunks.

    DualChunkSwap owns two non-contiguous ranges per rank (a head chunk and a
    mirrored tail chunk).  Re-deriving a single contiguous ``[start, end)``
    range silently changes the ownership map and produced (10, 26) where the
    catalog's (18, 18) split is the conserved answer.
    """
    ranges = pcp_rank_chunk_ranges(q_len, pcp, rank)
    if not ranges:
        return 0
    base = int(context) - int(q_len) if query_offset is None else int(query_offset)
    total = 0
    for start, end in ranges:
        for index in range(int(start), int(end)):
            total += _valid_keys(family, base + index, int(window), int(top_k),
                                 int(compress_ratio), bool(causal), int(context))
    return total


def pair_conservation(model: Any, parallel: ParallelSpec, q_len: int, context: int,
                      families: Sequence[str] | None = None, *,
                      causal: bool = True) -> dict[str, Any]:
    """Per-family total and per-PCP-rank counts against an independent truth."""
    names = tuple(families or model_families(model))
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    result = {}
    for family in names:
        window = max((int(getattr(plan, "window", 0) or 0) for plan in plans
                      if str(getattr(plan, "architecture", "")).find(family) >= 0),
                     default=max((int(getattr(plan, "window", 0) or 0) for plan in plans),
                                 default=0))
        top_k = max((int(getattr(plan, "index_topk", 0) or 0) for plan in plans),
                    default=0)
        ratio = max((int(getattr(plan, "compress_ratio", 0) or 0) for plan in plans),
                    default=0)
        per_rank = tuple(valid_pairs(family, q_len, context, parallel.pcp, rank,
                                     window=window, top_k=top_k,
                                     compress_ratio=ratio, causal=causal)
                         for rank in range(parallel.pcp))
        total = sum(per_rank)
        expected = _expected_pairs(family, q_len, context, window=window,
                                   top_k=top_k, compress_ratio=ratio,
                                   causal=causal)
        result[family] = {"total_pairs": total, "expected_pairs": expected,
                          "per_rank_pairs": per_rank,
                          "conserved": total == expected}
    return result

def pcp_kv_gather_bytes(chunk_tokens: int, pcp: int, kv_heads: int, head_dim: int,
                        element_bytes: float, *, rank: int | None = None,
                        owners: Sequence[float] | None = None) -> dict[str, float]:
    """Current-chunk K/V gather; older replicated KV is not re-sent."""
    if pcp < 1:
        raise ModelingError(f"pcp_kv_gather_pcp_invalid:{pcp}")
    tokens = max(0, int(chunk_tokens))
    total = float(tokens * 2 * max(1, int(kv_heads)) * max(1, int(head_dim))
                  * float(element_bytes))
    if owners is None:
        received = total * (pcp - 1) / pcp
        profile = tuple(total / pcp for _ in range(pcp))
    else:
        profile = tuple(float(value) for value in owners)
        if len(profile) != pcp or abs(sum(profile) - total) > max(1.0, total * 1e-9):
            raise ModelingError("pcp_kv_gather_owner_payload_not_conserved")
        local = profile[rank or 0]
        received = total - local
    return {"total_bytes": total, "received_bytes": received,
            "sent_bytes": max(0.0, received), "owner_profile": profile}


def pcp_index_k_gather_bytes(context_tokens: int, pcp: int, index_dim: int,
                             index_bytes: float, *, top_k: int = 0,
                             owners: Sequence[float] | None = None) -> dict[str, float]:
    """Sparse index-K gather payload, separate from the K/V payload."""
    if pcp < 1:
        raise ModelingError(f"pcp_index_gather_pcp_invalid:{pcp}")
    rows = max(0, int(context_tokens))
    if top_k > 0:
        rows = min(rows, int(top_k) * max(1, pcp))
    total = float(rows * max(1, int(index_dim)) * float(index_bytes))
    if owners is None:
        received = total * (pcp - 1) / pcp
        profile = tuple(total / pcp for _ in range(pcp))
    else:
        profile = tuple(float(value) for value in owners)
        if len(profile) != pcp or abs(sum(profile) - total) > max(1.0, total * 1e-9):
            raise ModelingError("pcp_index_gather_owner_payload_not_conserved")
        received = total - profile[0]
    return {"total_bytes": total, "received_bytes": received, "owner_profile": profile}


def dcp_merge_payloads(tp: int, dcp: int, query_batch: int, query_heads: int,
                       value_dim: int, element_bytes: float) -> dict[str, float]:
    """DCP max/sum/numerator reduction over ``[B_q, H_q, D_v + 1]``.

    ``query_batch`` is the admitted decode query rows, never the KV length;
    ``query_heads`` is the query-head count, never the KV-head count.
    """
    if dcp < 1 or dcp > tp or tp % dcp != 0:
        raise ModelingError(f"dcp_reduction_geometry_invalid:{dcp}!|{tp}")
    batch = max(0, int(query_batch))
    heads = max(1, int(query_heads))
    dim = max(1, int(value_dim))
    elem = max(1.0, float(element_bytes))
    output_elems = float(batch * heads * (dim + 1))
    per_lse = float(batch * heads)
    max_bytes = per_lse * elem
    sum_bytes = per_lse * elem
    numerator_bytes = output_elems * elem
    lse_merge_bytes = output_elems * elem
    return {"max_bytes": max_bytes, "sum_bytes": sum_bytes,
            "numerator_bytes": numerator_bytes,
            "lse_merge_bytes": lse_merge_bytes,
            "output_elements": output_elems,
            "total_bytes": max_bytes + sum_bytes + numerator_bytes,
            "element_bytes": elem,
            "geometry": "query_batch_x_query_heads_x_(value_dim+1)"}


@dataclass(frozen=True)
class ContextParallelPlan:
    prefill_plan: str
    decode_plan: str
    pcp: int
    dcp: int
    mrv2_pcp: bool
    attention_families: tuple[str, ...]
    prefill_collectives: tuple[str, ...]
    decode_collectives: tuple[str, ...]
    compatible: bool
    reason: str | None = None
    builders: tuple[dict[str, Any], ...] = ()
    pair_conservation: tuple[tuple[str, dict[str, Any]], ...] = ()
    index_gather: tuple[tuple[str, float], ...] = ()
    kv_gather: tuple[tuple[str, float], ...] = ()
    dcp_reductions: tuple[tuple[str, float], ...] = ()
    bridge: PhaseBridge | None = None
    lane_policy: str = "park_extra_lanes"
    remap_bytes: float = 0.0
    remap_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"prefill_plan": self.prefill_plan, "decode_plan": self.decode_plan,
                "pcp": self.pcp, "dcp": self.dcp, "mrv2_pcp": self.mrv2_pcp,
                "attention_families": list(self.attention_families),
                "prefill_collectives": list(self.prefill_collectives),
                "decode_collectives": list(self.decode_collectives),
                "compatible": self.compatible, "reason": self.reason,
                "builders": [dict(row) for row in self.builders],
                "pair_conservation": {name: dict(row)
                                      for name, row in self.pair_conservation},
                "index_gather": dict(self.index_gather),
                "kv_gather": dict(self.kv_gather),
                "dcp_reductions": dict(self.dcp_reductions),
                "bridge": self.bridge.to_dict() if self.bridge else None,
                "lane_policy": self.lane_policy,
                "remap_bytes": self.remap_bytes, "remap_ms": self.remap_ms}


def plan_context_parallel(model: Any, parallel: ParallelSpec, *,
                          prefill_plan: str | None = None,
                          decode_plan: str | None = None,
                          mrv2_pcp: bool = False,
                          attention_families: Sequence[str] | None = None,
                          isl: int = 4096, context_tokens: int | None = None,
                          window: int | None = None, top_k: int | None = None,
                          compress_ratio: int | None = None,
                          lane_bw_gbps: float = 100.0,
                          lane_kv_bytes: float = 0.0,
                          lane_state_bytes: float = 0.0,
                          query_batch: int | None = None) -> ContextParallelPlan:
    """Validate the resolved dual-phase plan and its actual builders."""
    actual = model_families(model)
    declared = (tuple(sorted({str(item).strip().lower() for item in attention_families}))
                if attention_families is not None else actual)
    prefill = str(prefill_plan or ("pcp" if parallel.pcp > 1 else
                                   "dsa_cp" if parallel.enable_dsa_cp else "none")).strip().lower()
    decode = str(decode_plan or ("dcp" if parallel.cp > 1 else "none")).strip().lower()
    reason = None
    if prefill not in PRE_REFILL_PLANS:
        reason = f"prefill_plan_unknown:{prefill}"
    elif decode not in DECODE_PLANS:
        reason = f"decode_plan_unknown:{decode}"
    elif prefill == "pcp" and parallel.pcp <= 1:
        reason = "pcp_plan_requires_pcp"
    elif prefill == "none" and parallel.pcp > 1:
        reason = "pcp_requires_prefill_pcp_plan"
    elif prefill == "dsa_cp":
        if decode == "dcp":
            reason = "dsa_cp_excludes_dcp"
        elif parallel.pcp > 1 or parallel.cp > 1:
            reason = "dsa_cp_excludes_pcp_and_dcp"
        elif not any(bool(getattr(plan, "indexer", False))
                     for plan in tuple(getattr(model, "layer_plans", ()) or ())):
            reason = "dsa_cp_requires_sparse_indexer"
    elif decode == "dcp" and parallel.cp <= 1:
        reason = "dcp_plan_requires_cp"
    elif decode == "none" and parallel.cp > 1:
        reason = "dcp_requires_decode_dcp_plan"
    if reason is None and set(declared) != set(actual):
        reason = f"attention_family_mismatch:{'+'.join(declared)}!={'+'.join(actual)}"
    builders = []
    if reason is None and mrv2_pcp:
        if prefill != "pcp" or parallel.pcp <= 1:
            reason = "mrv2_pcp_requires_pcpp_plan"
        elif not set(actual) & set(SPARSE_FAMILIES):
            reason = "mrv2_pcp_requires_sparse_model"
        else:
            for family in actual:
                if family in SPARSE_FAMILIES:
                    builder = _BUILDERS.get((family, "mrv2", "prefill", "pcp"))
                    if builder is None:
                        reason = f"pcp_builder_missing:{family}"
                        break
                    builders.append(builder.to_dict())
    resolved = parallel
    if reason is None:
        try:
            if prefill == "dsa_cp":
                resolved = replace(parallel, pcp=1, cp=1, dcp=1,
                                   cp_plan="dsa_cp", enable_dsa_cp=True)
            elif prefill == "pcp":
                resolved = replace(parallel, pcp=max(2, parallel.pcp),
                                   cp_plan="pcp")
            elif decode == "dcp":
                resolved = replace(parallel, cp=max(2, parallel.cp),
                                   dcp=max(2, parallel.cp), cp_plan="dcp")
            else:
                resolved = replace(parallel, cp=1, dcp=1, pcp=1, cp_plan="none",
                                   enable_dsa_cp=False)
            validate_cp_plan(model, resolved)
        except (ModelingError, ValueError) as exc:
            reason = str(exc)
    prefill_ops: list[str] = []
    decode_ops: list[str] = []
    conservation, index_gather, kv_gather, reductions = {}, {}, {}, {}
    bridge = None
    remap_bytes = remap_ms = 0.0
    if reason is None:
        if prefill == "dsa_cp":
            prefill_ops = ["dsa_cp_local_query_topk", "dsa_cp_output_all_gather"]
        elif prefill == "pcp":
            prefill_ops = ["pcp_query_split", "pcp_kv_gather"]
            if mrv2_pcp:
                prefill_ops = ["pcp_index_k_gather", "_pcp_kv_gather", *prefill_ops]
        if decode == "dcp":
            decode_ops = ["dcp_kv_shard", "dcp_max_reduce", "dcp_sum_reduce",
                          "dcp_numerator_reduce", "dcp_lse_merge"]
        if parallel.pcp > 1:
            context = int(context_tokens or getattr(model, "max_model_len", isl) or isl)
            conservation = pair_conservation(model, parallel, int(isl), context, actual)
            heads, head_dim, _layers = _kv_shape(model)
            index_dim, index_topk = _index_shape(model)
            chunk = max(pcp_query_tokens(int(isl), parallel.pcp))
            kv = pcp_kv_gather_bytes(chunk, parallel.pcp, heads, head_dim,
                                     _dtype_bytes(model, "kv_dtype"))
            index = pcp_index_k_gather_bytes(
                context, parallel.pcp, index_dim or 1,
                float(dtype_bytes("bf16")), top_k=index_topk)
            kv_gather = {"_pcp_kv_gather": kv["received_bytes"]}
            index_gather = {"pcp_index_k_gather": index["received_bytes"]}
        if parallel.cp > 1:
            _heads, head_dim, _layers = _kv_shape(model)
            plans = tuple(getattr(model, "layer_plans", ()) or ())
            query_heads = max((int(getattr(plan, "num_heads", 0) or 0)
                               for plan in plans), default=1)
            query_heads = max(1, query_heads // max(1, parallel.tp))
            merge = dcp_merge_payloads(
                parallel.tp, parallel.cp, max(1, int(query_batch or 1)),
                query_heads, head_dim, _dtype_bytes(model, "kv_dtype"))
            reductions = {key: float(value) for key, value in merge.items()
                          if isinstance(value, (int, float))}
        if parallel.pcp > 1 and parallel.cp > 1:
            bridged = phase_bridge(
                parallel, parallel, kv_bytes=float(lane_kv_bytes),
                index_bytes=float(sum(index_gather.values())),
                state_bytes=float(lane_state_bytes), epoch=0,
                lane_bw_gbps=lane_bw_gbps)
            if parallel.pcp_lane_policy == "replica_lanes":
                remap = bridge_remap_bytes(bridged)
                remap_bytes = float(remap["total_bytes"])
                bridge = bridged
            else:
                # Parked lanes carry no remap payload, but every source lane is
                # still recorded so the ownership map is complete.
                bridge = replace(bridged, destination_lanes=())
                remap_bytes = 0.0
            if remap_bytes > 0 and lane_bw_gbps <= 0:
                raise ModelingError("lane_remap_bandwidth_invalid")
            if remap_bytes > 0:
                remap_ms = remap_bytes / (float(lane_bw_gbps) * 1e9) * 1000.0
    return ContextParallelPlan(
        prefill, decode, int(parallel.pcp), int(parallel.cp), bool(mrv2_pcp),
        declared, tuple(prefill_ops), tuple(decode_ops), reason is None, reason,
        tuple(builders), tuple(sorted(conservation.items())),
        tuple(sorted(index_gather.items())), tuple(sorted(kv_gather.items())),
        tuple(sorted(reductions.items())), bridge, parallel.pcp_lane_policy,
        float(remap_bytes), float(remap_ms))


__all__ = ["ABI", "DECODE_PLANS", "PRE_REFILL_PLANS", "SPARSE_FAMILIES",
           "ContextParallelPlan", "PcpBuilder", "PhaseBridge",
           "bridge_remap_bytes", "dcp_merge_payloads", "model_families",
           "pair_conservation", "pcp_attention_pairs", "pcp_index_k_gather_bytes",
           "pcp_kv_gather_bytes",
           "phase_bridge", "plan_context_parallel", "register_pcp_builder",
           "registered_builders", "valid_pairs"]
