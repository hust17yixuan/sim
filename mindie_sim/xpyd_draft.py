"""Actual-payload draft-step roofline and physical route qualification.

[Role]
  Price one speculative draft step from declared draft geometry and qualify its
  actual TP/EP/PCP/DCP/PP communication on the physical endpoint map.

[Invariants]
  * ``endpoint_ids`` is a complete local-rank -> physical-rank map validated
    before any group is priced; group membership uses ``endpoint_ids[rank]``.
  * Every active axis is route-qualified and priced at its actual payload
    (TP/ETP/EP dispatch+combine/EDP/PCP/DCP plus every PP hop); a declared
    scalar draft duration never bypasses that evidence or that service.
  * Incomplete draft geometry returns a typed ``missing_draft_geometry`` record
    before any default dimension is substituted.
  * Cut and per-endpoint port demands are emitted for one step at ``batch``.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable record; caller Case untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.catalog.types import DraftSpec
from mindie_sim.parallel_cuts import (collective_cut_demands,
                                      collective_port_demands, merge_demands,
                                      peer_cut_demands, peer_port_demands,
                                      switch_bound_from_demands)
from mindie_sim.parallel_decode_cost import validate_endpoint_map
from mindie_sim.parallel_placement import build_placement
from mindie_sim.parallel_stage_ops import stage_groups, stage_layer_slices, axis_layer_indices
from mindie_sim.parallel_comm_ledger import CommLedgerBuilder, reduce_route_evidence
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.xpyd_transfer import _int


@dataclass(frozen=True)
class DraftStepCost:
    """Priced draft step plus actual-payload route evidence and cut demand."""

    step_ms: float
    route_qualified: bool
    route_evidence: str
    comm_demands: dict[str, float] = field(default_factory=dict)
    groups: tuple[tuple[int, ...], ...] = ()
    payload_bytes: float = 0.0
    layers: int = 0
    port_demands: dict[str, float] = field(default_factory=dict)
    mandatory_ms: float = 0.0

    def execution_duration(self, declared_ms: float) -> tuple[float, str]:
        """A scalar can replace the roofline, but cannot erase physical service."""
        if declared_ms > 0.0:
            return max(float(declared_ms), self.mandatory_ms), "declared_layer_cost"
        return self.step_ms, "priced_draft_spec"

    def to_dict(self) -> dict[str, Any]:
        return {"step_ms": self.step_ms,
                "route_qualified": self.route_qualified,
                "route_evidence": self.route_evidence,
                "comm_demands": dict(self.comm_demands),
                "port_demands": dict(self.port_demands),
                "groups": [list(group) for group in self.groups],
                "payload_bytes": self.payload_bytes,
                "layers": self.layers, "mandatory_ms": self.mandatory_ms}


def draft_geometry_error(draft: Any) -> str | None:
    """First invalid/missing field, checked on raw values before any coercion."""
    for name in ("layers", "hidden_size", "intermediate_size", "num_heads",
                 "num_kv_heads", "head_dim", "q_width", "kv_width"):
        value = getattr(draft, name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return name
    expert_fields = ("num_experts", "top_k", "expert_intermediate_size",
                     "num_shared_experts")
    expert_dims = {name: getattr(draft, name, 0) for name in expert_fields}
    for name, value in expert_dims.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return name
    # All-zero expert dimensions denote a dense draft.  Any MoE declaration
    # requires all three routed dimensions, with a realizable routing fan-out.
    if any(expert_dims.values()):
        for name in expert_fields[:3]:
            if expert_dims[name] <= 0:
                return name
        if expert_dims["top_k"] > expert_dims["num_experts"]:
            return "top_k"
    if str(getattr(draft, "attention_kind", "gqa") or "gqa").lower() == "mla":
        for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                     "v_head_dim"):
            value = getattr(draft, name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return name
    return None


def draft_geometry_reason(draft: Any) -> str | None:
    """Stable typed reason shared by direct pricing and pre-memory admission."""
    name = draft_geometry_error(draft)
    if name is None:
        return None
    value = getattr(draft, name, 0)
    missing = type(value) is int and value == 0
    return f"{'missing' if missing else 'invalid'}_draft_geometry:{name}"


def raw_shared_geometry_reason(case: Any) -> str | None:
    """Guard declared shared-target dimensions before parsing or placement.

    Mapping inputs may omit dimensions that the schema derives; explicitly
    supplied invalid values must never reach coercion or divisibility checks.
    """
    def get(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    model, runtime = get(case, "model"), get(case, "runtime", {})
    meta = dict(get(runtime, "metadata", {}) or {})
    if (get(model, "auxiliary") is not None
            or not any(meta.get(k) for k in ("mtp_shared_weights", "draft_shared_weights", "eagle_shared_weights"))
            or not (meta.get("speculative") or meta.get("mtp_acceptance_length"))):
        return None
    aliases = (("hidden_size", "d_model", "hidden_dim"),
               ("intermediate_size", "ffn_dim"),
               ("num_heads", "num_attention_heads", "n_heads"),
               ("num_kv_heads", "num_key_value_heads"), ("head_dim", "d_head"))
    for names in aliases:
        name = next((n for n in names if n in model), None) if isinstance(model, Mapping) else names[0]
        if name is None:
            continue
        value = get(model, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return "missing_draft_geometry"
    return None


def shared_target_draft(model: Any) -> DraftSpec | None:
    """Derive one shared target layer; weight sharing never removes execution.

    Dimensions remain raw until validation, so absent, boolean and fractional
    geometry cannot become a qualified communicator through integer coercion.
    """
    names = ("hidden_size", "intermediate_size", "num_heads", "num_kv_heads",
             "head_dim")
    dims = {name: getattr(model, name, None) for name in names}
    if any(type(value) is not int or value <= 0 for value in dims.values()):
        return None
    moe = str(getattr(model, "ffn", "dense")) == "moe"
    draft = DraftSpec(
        kind="mtp", layers=1, **dims,
        q_width=dims["num_heads"] * dims["head_dim"],
        kv_width=dims["num_kv_heads"] * dims["head_dim"],
        vocab_size=getattr(model, "vocab_size", 0),
        attention_kind=getattr(model, "attention", "gqa"),
        activation_dtype=getattr(model, "activation_dtype", "bf16"),
        num_experts=getattr(model, "num_experts", 0) if moe else 0,
        top_k=getattr(model, "top_k", 0) if moe else 0,
        expert_intermediate_size=getattr(model, "moe_intermediate_size", 0) if moe else 0,
        num_shared_experts=getattr(model, "num_shared_experts", 0) if moe else 0,
        kv_lora_rank=getattr(model, "kv_lora_rank", 0),
        qk_nope_head_dim=getattr(model, "qk_nope_head_dim", 0),
        qk_rope_head_dim=getattr(model, "qk_rope_head_dim", 0),
        v_head_dim=getattr(model, "v_head_dim", 0),
        target_shared_embedding=True, target_shared_lm_head=True)
    return draft if draft_geometry_error(draft) is None else None


def resolve_draft_geometry(candidate):
    """Resolve and validate the same geometry before memory and execution."""
    auxiliary = getattr(candidate.model, "auxiliary", None)
    draft = (getattr(auxiliary, "mtp", None) or getattr(auxiliary, "dspark", None)
             if auxiliary is not None else None)
    meta = dict(candidate.runtime.metadata)
    shared = any(meta.get(name) for name in ("mtp_shared_weights", "draft_shared_weights", "eagle_shared_weights"))
    if draft is None and shared and (meta.get("speculative") or meta.get("mtp_acceptance_length")):
        draft = shared_target_draft(candidate.model)
        if draft is None:
            return None, "missing_draft_geometry"
    return draft, draft_geometry_reason(draft) if draft is not None else None


def priced_draft_step_ms(candidate: CaseSpec, draft: Any, *, batch: int = 1,
                         context_tokens: int = 0, fabric: Any = None,
                         endpoints: Sequence[int] | None = None,
                         dtype: str | None = None,
                         topology: str | None = None,
                         groups: Sequence[Sequence[int]] | None = None,
                         ) -> DraftStepCost | None:
    """Batch/context-aware roofline price of one draft step.

    The FFN alone is ``batch * 6 * layers * H * I / TP / cube_peak``; attention
    projections, vocabulary projection, draft KV reads and activations are
    added, then the draft TP AllReduce is charged on every TP group.  A
    batch-independent price undercharged large admitted batches by more than
    an order of magnitude.
    """
    if draft is None:
        return None
    geometry_reason = draft_geometry_reason(draft)
    if geometry_reason is not None:
        # Typed fail-closed result: never substitute a default dimension and
        # never price a partially declared draft.
        return DraftStepCost(0.0, False, geometry_reason)
    layers = int(getattr(draft, "layers", 0) or 0)
    hidden = int(getattr(draft, "hidden_size", 0) or 0)
    intermediate = int(getattr(draft, "intermediate_size", 0) or 0)
    rows = _int("draft_batch", batch)
    context = _int("draft_context", context_tokens, minimum=0)
    p = candidate.runtime.parallel
    tp = max(1, int(p.tp))
    world = int(p.world_size)
    tree = fabric or HierarchicalFabricTree.from_hardware(candidate.machine)
    # The endpoint map is a complete local-rank -> physical-rank mapping and is
    # validated before any timing or route probe, never coerced per group.
    endpoint_ids = validate_endpoint_map(endpoints, world, tree=tree)
    meta = dict(candidate.runtime.metadata)
    from mindie_sim.memory_ledger_types import _draft_bytes
    from mindie_sim.prediction.capacity import Memory
    # A complete geometry must produce its real weight ledger.  Swallowing a
    # Memory failure here silently substituted zero draft weights.
    mem = Memory(candidate)
    draft_bytes, _detail = _draft_bytes(candidate, mem)
    die = candidate.machine.dies[0]
    hbm_bw = float(dict(die.rates)["hbm"])
    cube = float(die.peak(candidate.model.compute_dtype or "bf16"))
    vector = float(dict(die.rates)["vector"])
    element = float(dtype_bytes(getattr(draft, "activation_dtype", "bf16")))
    shared = bool(meta.get("mtp_shared_weights")
                  or meta.get("draft_shared_weights")
                  or meta.get("eagle_shared_weights"))
    weight_bytes = 0.0 if shared else max(0.0, float(sum(draft_bytes)))
    q_width = int(getattr(draft, "q_width", 0))
    qkv = float(4 * hidden * q_width)
    ffn = float(6 * hidden * intermediate)
    routed = 0.0
    experts = int(getattr(draft, "num_experts", 0) or 0)
    top_k = int(getattr(draft, "top_k", 0))
    expert_inter = int(getattr(draft, "expert_intermediate_size", 0) or 0)
    if experts > 0 and top_k > 0 and expert_inter > 0:
        routed = float(top_k * 6 * hidden * expert_inter)
    vocab = int(getattr(draft, "vocab_size", 0) or 0)
    projection = (float(2 * hidden * vocab)
                  if vocab > 0 and not bool(getattr(draft, "target_shared_lm_head", False))
                  else 0.0)
    flops = float(rows) * float(layers) * (qkv + ffn + routed + projection) / float(tp)
    compute_ms = flops / cube * 1000.0 if cube > 0 else 0.0
    kv_heads = max(1, int(getattr(draft, "num_kv_heads", 1) or 1))
    head_dim = int(getattr(draft, "head_dim", 0))
    draft_kv_per_token = float(2 * kv_heads * head_dim * element * layers) / float(tp)
    hbm_bytes = (weight_bytes
                 + float(rows) * float(context) * draft_kv_per_token
                 + float(rows) * float(hidden) * element * 4.0 * float(layers)
                 / float(tp))
    hbm_ms = hbm_bytes / hbm_bw * 1000.0 if hbm_bw > 0 else 0.0
    vector_ms = (float(rows) * float(hidden) * 8.0 * float(layers) / float(tp)
                 / vector * 1000.0 if vector > 0 else 0.0)
    collective_ms = 0.0
    hop_service_us = 0.0
    route_qualified = True
    route_evidence = "declared"
    payload = float(rows * hidden * element)
    plan = build_placement(candidate, p).to_dict()
    if groups is not None:
        plan = dict(plan)
        plan["tp_groups"] = tuple(tuple(int(rank) for rank in group)
                                  for group in groups)
    _, stage_layers, masks = stage_layer_slices(candidate, draft=draft)
    ledger = CommLedgerBuilder(tree, "draft")
    physical_groups = []
    flags = []
    for stage, indices in enumerate(stage_layers):
        stage_ledger = CommLedgerBuilder(tree, "draft")
        stage_service_us = 0.0
        for axis, kind in (("tp", "all_reduce"), ("etp", "all_reduce"),
                           ("ep", "all_to_all_v"), ("edp", "all_reduce"),
                           ("pcp", "all_gather"), ("dcp", "all_reduce")):
            count = len(axis_layer_indices(indices, masks, axis))
            axis_groups = tuple(tuple(endpoint_ids[rank] for rank in group)
                                for group in stage_groups(plan, stage, axis) if len(group) > 1)
            if not count or not axis_groups:
                continue
            axis_payload = payload * (top_k if axis == "ep" else 1)
            calls = 2 if axis == "ep" else 1
            results = [tree.collective(kind, axis_payload, len(ids), endpoints=ids,
                                       dtype=dtype, topology=topology) for ids in axis_groups]
            stage_service_us += max(float(r.total_us) for r in results) * calls * count
            flags.extend((r.route_qualified, r.evidence) for r in results)
            members = getattr(tree, "member_ranks", None)
            if members and any(not set(ids) <= set(members) for ids in axis_groups):
                flags.append((False, "disconnected_participants"))
            for _call in range(calls):
                stage_ledger.collective(axis, kind, axis_payload, axis_groups, count=count)
            if axis == "tp":
                physical_groups.extend(axis_groups)
        emitted = stage_ledger.build()
        ledger.merge(emitted)
        collective_ms += max(stage_service_us,
                             switch_bound_from_demands(tree, emitted.cut_demands())) / 1000.0
    route_qualified, route_evidence = reduce_route_evidence(*flags)
    comm_demands = ledger.build().cut_demands()
    port_demands = ledger.build().port_demands()
    # A draft step also crosses every adjacent PP stage boundary.  Those hops
    # are qualified, counted on their physical route and added to the service
    # duration; aggregate bytes are not evidence that the hop is free.
    if int(p.pp) > 1:
        rank_map = plan.get("rank_map") or ()
        blocks: list[list[int]] = [[] for _ in range(max(0, int(p.pp)))]
        if rank_map:
            for rank, coordinate in enumerate(rank_map):
                stage = int(coordinate[0])
                if rank < world and 0 <= stage < len(blocks):
                    blocks[stage].append(int(endpoint_ids[rank]))
        else:
            width = max(1, world // max(1, int(p.pp)))
            for stage in range(len(blocks)):
                lo = stage * width
                hi = world if stage == len(blocks) - 1 else (stage + 1) * width
                blocks[stage].extend(int(value) for value in endpoint_ids[lo:hi])
        hop_payload = payload / float(max(1, tp))
        for stage in range(len(blocks) - 1):
            hop_ids = tuple(blocks[stage]) + tuple(blocks[stage + 1])
            comm_demands = merge_demands(
                comm_demands, peer_cut_demands(tree, hop_ids, hop_payload))
            port_demands = merge_demands(
                port_demands, peer_port_demands(tree, hop_ids, hop_payload))
            if hasattr(tree, "peer_transfer"):
                hop = tree.peer_transfer(hop_payload, hop_ids, dtype=dtype,
                                         topology=topology)
                hop_service_us += float(hop.total_us)
                if not hop.route_qualified:
                    route_qualified = False
                    route_evidence = str(hop.evidence)
    collective_ms += hop_service_us / 1000.0
    step_ms = (max(weight_bytes / hbm_bw * 1000.0 if hbm_bw > 0 else 0.0,
                   compute_ms, hbm_ms, vector_ms) + collective_ms)
    return DraftStepCost(float(step_ms), bool(route_qualified),
                         str(route_evidence), comm_demands,
                         tuple(physical_groups), payload, int(layers),
                         port_demands, mandatory_ms=float(collective_ms))

__all__ = ["DraftStepCost", "draft_geometry_error", "draft_geometry_reason",
           "priced_draft_step_ms"]
