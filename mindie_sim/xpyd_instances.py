"""xPyD instance enumeration, fabric gating and per-config facts.

[Role]
  Validate one P/G parallel candidate, gate every assigned physical device's
  fabric connectivity and HBM ledger, and price its prefill/decode service.

[Invariants]
  * A local instance plan maps onto concrete device IDs via ``rank_offset``;
    every active communicator group on every axis is route-qualified on its
    physical devices, and service timing reuses those same endpoints.
  * A P-to-D handoff requires a connected fabric path spanning the source and
    destination endpoints, and its transfer rate is bounded by the slowest
    declared endpoint/switch cut on that path.
  * Prefill throughput scales with independent DP replicas, never with the
    decode PCP/DCP lane policy.
  * EPLB EP-slot destinations are mapped onto the full DP*TP*PP*PCP world; a
    fractional slot vector is rejected before any ``int`` coercion.
  * Every speculative draft branch is charged: an explicit per-layer cost or a
    priced ``DraftSpec`` roofline, never a free non-shared draft.
  * Decode cut demand is one operation-derived ledger: ``1 + K`` target
    verification rows and ``K`` draft steps per request per cycle, scaled by
    the exact renewal cycle count, never a one-token reconstruction.
  * A declared draft duration never bypasses actual-payload route/anchor
    qualification on any active TP/ETP/EP/PP/PCP/DCP draft axis; missing
    auxiliary draft geometry fails closed instead of probing one byte.
  * Prefill and decode pool cut demands are the emitted step ledger normalized
    per request, never a generic axis reconstruction.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh records; caller Cases untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.memory_breakdown import memory_ledger
from mindie_sim.memory_eplb import resolve_eplb_epoch_config
from mindie_sim.moe_skew_eplb import replicated_expert_bytes, speculative_accept_tokens
from mindie_sim.parallel_cuts import merge_demands
from mindie_sim.parallel_ep_knee import (DecodeCostModel, check_route_qualified,
                                         execution_context, iter_axis_groups,
                                         prefill_step_ms, validate_slot_vector)
from mindie_sim.parallel_placement import validate_placement
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_draft import (DraftStepCost, resolve_draft_geometry,
                                    priced_draft_step_ms, raw_shared_geometry_reason)
from mindie_sim.xpyd_transfer import (_int, _nic_bw_gbps, _positive,
                                      completed_requests_per_s,
                                      expected_decode_cycles)


def _route_qualified(tree: Any, plan: Mapping[str, Any],
                     rank_offset: int = 0, *, dtype: str | None = None,
                     topology: str | None = None) -> tuple[bool, str]:
    """Gate every physical participant and every active axis, not one group.

    ``rank_offset`` maps a local instance plan onto its concrete device IDs.
    A device outside every declared link member set is disconnected even when
    its own instance has width one.  Each axis is probed with the collective it
    actually issues (AllReduce for TP/ETP/EDP/DCP, AllGather for PCP, AllToAllV
    for EP), never a blanket AllToAllV.
    """
    table = plan.get("rank_table") or ()
    local_ranks = {int(rank) for group in table for rank in group}
    if not local_ranks:
        local_ranks = set(range(int(plan.get("world", 1))))
    physical = {int(rank_offset) + rank for rank in local_ranks}
    members = getattr(tree, "member_ranks", None)
    if members and not physical <= set(members):
        missing = tuple(sorted(physical - set(members)))
        return False, f"disconnected_participants:{missing}"
    try:
        return check_route_qualified(tree, plan, int(rank_offset),
                                     dtype=dtype, topology=topology)
    except ModelingError as exc:
        return False, f"no_declared_route:{exc}"


def p_to_d_path(tree: Any, p_devices: Sequence[int], d_devices: Sequence[int], *,
                dtype: str | None = None, topology: str | None = None,
                probe_bytes: float = 1.0) -> tuple[bool, str, float | None]:
    """Connectivity and rate bound of the actual P-to-D KV handoff path.

    A P shard and its target D shard must lie on one connected route; a pool
    whose P and D dies sit in disjoint link components is not qualified even
    when each instance is individually well-formed.  The returned GB/s bound is
    the slowest declared endpoint/switch service on the traversed path.
    """
    endpoints = tuple(sorted({int(value) for value in p_devices}
                             | {int(value) for value in d_devices}))
    if len(endpoints) < 2:
        return True, "single_rank", None
    try:
        route = tree.collective("all_to_all_v", float(probe_bytes),
                                len(endpoints), endpoints=endpoints,
                                dtype=dtype, topology=topology)
    except ModelingError as exc:
        return False, f"no_declared_p_to_d_route:{exc}", 0.0
    if not route.route_qualified:
        return False, route.evidence, 0.0
    rates: list[float] = []
    for tier in getattr(tree, "tiers", ()):
        for rank, rate in getattr(tier, "endpoint_rates", ()):
            if int(rank) in endpoints:
                rates.append(float(rate))
    # A shared switch cut is a physical resource shared by *every* logical
    # instance on the path: it must never be collapsed into a per-instance
    # dedicated port rate and then multiplied by n_prefill/n_decode.
    bw_gbps = (min(rates) / 1e6) if rates else None
    return True, route.evidence, bw_gbps


def p_to_d_shared_cuts(tree: Any, endpoints: Sequence[int]) -> dict[str, float]:
    """Physical shared cut capacities (bytes/s) traversed by a pool flow.

    Resource identity is preserved by name so concurrent P/D instances share
    one ceiling.  Both a declared switch capacity and a ``shared``/``shared_bus``
    tier bandwidth are physical cuts; endpoint port rates remain dedicated.
    """
    members = {int(value) for value in endpoints}
    cuts: dict[str, float] = {}
    for tier in getattr(tree, "tiers", ()):
        links = getattr(tier, "link_members", ())
        if links and members and not any(members <= set(group) for group in links):
            # The tier is not on this pool's physical path.
            if not (members & {int(rank) for group in links for rank in group}):
                continue
        switch = getattr(tier, "switch_capacity_bytes_per_ms", None)
        if switch is not None:
            cuts[f"switch:{tier.name}"] = float(switch) * 1e3
        if str(getattr(tier, "semantics", "shared")) in {"shared", "mesh"}:
            cuts[f"tier:{tier.name}"] = float(tier.bw_gbps) * 1e9
    return cuts


def _comm_ledger_demands(ledger: Any, plan: Mapping[str, Any], *,
                         phase: str) -> dict[str, float]:
    """Consume the step's executed ledger; never reconstruct active traffic.

    A step implementation that omits the ledger fails closed. An emitted
    empty ledger is valid for single-rank work or stages with no layers that
    execute their available expert communicators.
    """
    if ledger is None or not hasattr(ledger, "cut_demands"):
        raise ModelingError(f"xpyd_comm_ledger_missing:{phase}")
    # A typed empty execution is legitimate when the only multi-rank groups
    # are expert axes but every owned layer is dense (independent DP work).
    return {str(key): float(value)
            for key, value in ledger.cut_demands().items()}


def _normalize_prefill_demands(ledger: Any, plan: Mapping[str, Any],
                               step: Mapping[str, Any]
                               ) -> tuple[dict[str, float], dict[str, float]]:
    """Per-request prefill demand from the emitted batch/chunk ledger."""
    q_tokens = max(0.0, float(step.get("q_tokens", 0) or 0.0))
    tokens = float(step.get("tokens_global", 0.0) or 0.0)
    if q_tokens <= 0.0:
        return {}, {}
    if tokens <= 0.0:
        raise ModelingError("xpyd_prefill_ledger_tokens_missing")
    raw = _comm_ledger_demands(ledger, plan, phase="prefill")
    scale = q_tokens / tokens
    ports = getattr(ledger, "port_demands", None)
    port_raw = ports() if callable(ports) else {}
    return ({key: value * scale for key, value in raw.items()},
            {key: float(value) * scale for key, value in port_raw.items()})


def _normalize_decode_demands(ledger: Any, plan: Mapping[str, Any], *,
                              batch: int, cycles: float
                              ) -> tuple[dict[str, float], dict[str, float]]:
    """Per-request decode demand from the emitted DP*B/verification ledger."""
    requests = max(1, int(plan["dp"])) * max(1, int(batch))
    raw = _comm_ledger_demands(ledger, plan, phase="decode")
    ports = getattr(ledger, "port_demands", None)
    port_raw = ports() if callable(ports) else {}
    factor = float(cycles) / float(requests)
    return ({key: value * factor for key, value in raw.items()},
            {key: float(value) * factor for key, value in port_raw.items()})


def pool_config_facts(case: CaseSpec, parallel: ParallelSpec, *, phase: str, isl: int,
                  osl: int, prefix_hit_rate: float, tpot_slo_ms: float,
                  hccl_buffsize_mb: float | None, routing_skew: float,
                  eplb_redundant_experts: int, enable_eplb: bool,
                  tree: Any, rho: float,
                  eplb_redundant_slots_per_rank: Sequence[int] | None = None,
                  allow_unqualified: bool = False,
                  staging_bytes: float | None = None,
                  rank_offset: int = 0,
                  transfer_plan: Any = None,
                  hccl_buffsize_bytes: int | None = None,
                  comm_arena_qualified: bool = False,
                  comm_arena_domains: Mapping[str, float] | None = None,
                  strict_memory_qualification: bool = False) -> dict[str, Any]:
    reason = raw_shared_geometry_reason(case)
    if reason is not None:
        return {"parallel": parallel, "feasible": False, "reason": reason,
                "world": parallel.world_size, "mu_requests_per_s": 0.0,
                "route_qualified": False, "fabric_route_qualified": False,
                "route_evidence": reason, "draft_comm_demands": {}}
    if hasattr(case, "model") and hasattr(case, "runtime"):
        _, pre_geometry_reason = resolve_draft_geometry(case)
        if pre_geometry_reason is not None:
            return {"parallel": parallel, "feasible": False, "reason": pre_geometry_reason,
                    "world": parallel.world_size, "mu_requests_per_s": 0.0,
                    "route_qualified": False, "fabric_route_qualified": False,
                    "route_evidence": pre_geometry_reason, "draft_comm_demands": {}}
    rank_offset = _int("rank_offset", rank_offset, minimum=0)
    plan_staging = 0.0
    if transfer_plan is not None:
        peak = getattr(transfer_plan, "phase_peak_bytes", None)
        if callable(peak):
            plan_staging = max(0.0, float(peak(phase)))
        else:
            raise ModelingError(
                f"xpyd_transfer_plan_invalid:{type(transfer_plan).__name__}")
    staging_reserve = max(float(staging_bytes or 0.0), plan_staging)
    plan_record = (transfer_plan.to_dict()
                   if hasattr(transfer_plan, "to_dict") and transfer_plan
                   is not None else {})
    plan_fields = {name: float(plan_record[name])
                   for name in ("source_retention_bytes",
                                "destination_retention_bytes",
                                "handoff_staging_bytes",
                                "eplb_rebalance_peak_bytes",
                                "phase_peak_bytes")
                   if plan_record.get(name) is not None}
    if transfer_plan is not None:
        plan_fields["phase_peak_bytes"] = float(plan_staging)
    plan_page_owners = tuple(int(rank) for rank in
                             getattr(transfer_plan, "page_owner_ranks", ()) or ())
    candidate, plan, reason = validate_placement(case, parallel)
    if candidate is None or plan is None:
        return {"parallel": parallel, "feasible": False, "reason": reason,
                "world": parallel.world_size, "mu_requests_per_s": 0.0,
                "route_qualified": False}
    redundant = _int("eplb_redundant_experts", eplb_redundant_experts, minimum=0)
    evidence = execution_context(candidate)
    route_ok, route_evidence = _route_qualified(
        tree, plan, rank_offset, **evidence)
    if not route_ok and not allow_unqualified:
        return {"parallel": parallel, "plan": plan, "feasible": False,
                "reason": "fabric_route_unqualified", "world": plan["world"],
                "mu_requests_per_s": 0.0, "route_qualified": False,
                "route_evidence": route_evidence}
    context = isl + osl if phase == "decode" else isl
    meta = dict(candidate.runtime.metadata)
    draft_spec_early, geometry_reason = resolve_draft_geometry(candidate)
    if geometry_reason is not None:
        return {"parallel": parallel, "plan": plan, "feasible": False,
                "reason": geometry_reason, "world": plan["world"], "mu_requests_per_s": 0.0,
                "route_qualified": False, "fabric_route_qualified": False,
                "route_evidence": geometry_reason, "draft_comm_demands": {}}
    model = DecodeCostModel.from_case(candidate, context_tokens=context)
    max_batch = int(candidate.runtime.max_num_seqs)
    supplied = validate_slot_vector(eplb_redundant_slots_per_rank)
    endpoints = tuple(range(int(rank_offset), int(rank_offset) + int(plan["world"])))
    world = int(plan["world"])
    ep = max(1, int(parallel.ep))
    ep_coords = tuple(int(row[5]) for row in plan["rank_map"])
    epoch_config = resolve_eplb_epoch_config(
        enable_eplb=enable_eplb, redundant_experts=redundant,
        slot_vector=supplied, ep=ep, world=world,
        num_experts=int(candidate.model.num_experts or 0),
        ep_coords=ep_coords,
        top_k=max(1, int(candidate.model.top_k or 1)), routing_skew=routing_skew)
    eplb_active = epoch_config.active
    redundant = epoch_config.redundant

    def epoch_for(rows: int) -> tuple[tuple[int, ...], tuple[float, ...],
                                      Any, float] | None:
        """One placement epoch for execution, resident copies and staging."""
        from mindie_sim.moe_skew_eplb import place_eplb
        tokens_ep = max(1, int(plan["dp"])) * int(rows)
        placement = place_eplb(tokens_ep, model.top_k, ep, routing_skew,
                               num_experts=model.num_experts,
                               redundant_experts=redundant, per_token_ms=1.0)
        executed = tuple(int(value) for value in
                         placement.redundant_slots_per_rank)
        if len(executed) != ep:
            raise ModelingError("xpyd_eplb_slots_length")
        mapped = tuple(executed[ep_coords[rank]] for rank in range(world))
        slots = (epoch_config.slots_rank if epoch_config.slots_rank is not None
                 else mapped)
        # Unconditional: a caller-supplied vector is an execution claim, not a
        # ledger-only override.  A mismatch rejects this batch.
        if tuple(int(value) for value in slots) != mapped:
            return None
        replica = replicated_expert_bytes(
            candidate, redundant, redundant_slots_per_rank=slots)
        staging = tuple(float(value) for value in replica["per_rank_bytes"])
        placement = place_eplb(
            tokens_ep, model.top_k, ep, routing_skew,
            num_experts=model.num_experts, redundant_experts=redundant,
            per_token_ms=1.0,
            extra_weight_bytes_per_rank=max(staging, default=0.0),
            rebalance_bw_gbps=_nic_bw_gbps(candidate))
        executed = tuple(int(value) for value in
                         placement.redundant_slots_per_rank)
        mapped = tuple(executed[ep_coords[rank]] for rank in range(world))
        if tuple(int(value) for value in slots) != mapped:
            return None
        return (tuple(int(value) for value in slots), staging, placement,
                float(placement.rebalance_ms))

    ledger_cache: dict[tuple, Any] = {}

    def ledger_for(slots_b: Sequence[int] | None,
                   staging_b: Sequence[float] | None):
        key = ((tuple(slots_b) if slots_b is not None else None),
               (tuple(staging_b) if staging_b is not None else None))
        cached = ledger_cache.get(key)
        if cached is None:
            cached = memory_ledger(
                candidate, sequence_tokens=context,
                resolved_draft=draft_spec_early,
                hccl_buffsize_mb=hccl_buffsize_mb,
                enable_eplb=eplb_active, eplb_redundant_experts=redundant,
                eplb_redundant_slots_per_rank=slots_b,
                eplb_rebalance_staging_bytes_per_rank=staging_b,
                kv_transfer_staging_bytes=staging_reserve,
                kv_transfer_rank_bytes=tuple(max(float(staging_bytes or 0), value) for value in
                    transfer_plan.phase_rank_peak_bytes(phase)) if transfer_plan else (),
                kv_transfer_fields=plan_fields,
                kv_transfer_page_owner_ranks=plan_page_owners,
                hccl_buffsize_bytes=hccl_buffsize_bytes,
                comm_arena_qualified=comm_arena_qualified,
                comm_arena_domains=comm_arena_domains,
                strict_memory_qualification=strict_memory_qualification)
            ledger_cache[key] = cached
        return cached

    def route_failure(step_evidence: str | None, **extra: Any):
        row = {"parallel": parallel, "plan": plan, "feasible": False,
               "reason": "fabric_route_unqualified", "world": plan["world"],
               "mu_requests_per_s": 0.0, "route_qualified": False,
               "fabric_route_qualified": False,
               "route_evidence": step_evidence or route_evidence}
        row.update(extra)
        return row

    if phase == "prefill":
        q_total = max(0, int(math.ceil(isl * (1.0 - prefix_hit_rate))))
        token_budget = max(1, int(candidate.runtime.max_num_batched_tokens))
        b_max = max_batch if q_total <= 0 else max(1, min(max_batch, token_budget // q_total))
        chosen = None
        last_memory = None
        last_reason = "prefill_memory_infeasible"
        for batch in range(b_max, 0, -1):
            epoch = epoch_for(batch) if eplb_active else None
            if eplb_active and epoch is None:
                last_reason = "eplb_epoch_mismatch"
                continue
            slots_b = epoch[0] if epoch else None
            staging_b = epoch[1] if epoch else None
            memory = ledger_for(slots_b, staging_b).at(batch, isl)
            last_memory = memory
            if not memory.oom_feasible:
                continue
            step = prefill_step_ms(candidate, isl=isl, prefix_hit_rate=prefix_hit_rate,
                                   batch=batch, fabric=tree, endpoints=endpoints,
                                   dtype=evidence["dtype"],
                                   topology=evidence["topology"])
            if not step.get("route_qualified", True) and not allow_unqualified:
                return route_failure(step.get("route_evidence"))
            chosen = (batch, memory, step)
            break
        if chosen is None:
            if (last_reason == "prefill_memory_infeasible"
                    and last_memory is not None
                    and last_memory.status == "memory_unqualified"):
                last_reason = f"{last_memory.status}:{last_memory.reason}"
            return {"parallel": parallel, "plan": plan, "feasible": False,
                    "reason": last_reason, "world": plan["world"],
                    "mu_requests_per_s": 0.0, "route_qualified": route_ok,
                    "route_evidence": route_evidence, "memory": last_memory,
                    "kv_transfer_staging_bytes": staging_reserve,
                    "bridge_lifetime_plan": (
                        transfer_plan.to_dict()
                        if hasattr(transfer_plan, "to_dict") else None)}
        batch, memory, step = chosen
        # Prefill throughput is independent of the decode PCP/DCP lane policy:
        # PCP lanes cooperate on one request, so only independent DP replicas
        # multiply the service rate.
        replicas = int(plan["dp"])
        mu = step["mu_requests_per_s"] * replicas
        try:
            shared_cut_demands, port_demands = _normalize_prefill_demands(
                step.get("comm_ledger"), plan, step)
        except ModelingError as exc:
            return {"parallel": parallel, "plan": plan, "feasible": False,
                    "reason": str(exc), "world": plan["world"],
                    "mu_requests_per_s": 0.0, "route_qualified": False}
        return {"parallel": parallel, "plan": plan, "feasible": True, "world": plan["world"],
                "memory": memory, "step_ms": step["step_ms"], "batch": batch,
                "mu_requests_per_s": mu,
                "output_tokens_per_s": mu * osl,
                "q_tokens": step["q_tokens"], "kv_tokens": isl,
                "prefill_steps": step["steps"], "prefill_compute_ms": step["compute_ms"],
                "prefill_collective_ms": step["collective_ms"],
                "prefill_moe_comm_ms": step.get("moe_comm_ms", 0.0),
                "active_replicas": replicas,
                "kv_transfer_staging_bytes": staging_reserve,
                "bridge_lifetime_plan": (
                    transfer_plan.to_dict()
                    if hasattr(transfer_plan, "to_dict") else None),
                "kv_transfer_fields": dict(plan_fields),
                "kv_transfer_page_owner_ranks": list(plan_page_owners),
                "comm_persistent_qualified": bool(
                    getattr(memory, "comm_persistent_qualified", False)),
                "comm_persistent_basis": str(
                    getattr(memory, "comm_persistent_basis", "unknown")),
                "shared_cut_demands": shared_cut_demands,
                "port_demands": port_demands,
                "route_qualified": bool(route_ok and step.get("route_qualified", True)),
                "fabric_route_qualified": bool(step.get("route_qualified", True)),
                "route_evidence": (step.get("route_evidence") if
                                   not step.get("route_qualified", True)
                                   else route_evidence)}
    def _meta_count(name: str, default: int, minimum: int = 0) -> int:
        value = meta.get(name, default)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ModelingError(f"xpyd_{name}_invalid:{value}")
        return value

    draft = _meta_count("mtp_acceptance_length", 0)
    speculative = bool(meta.get("speculative")) or draft > 0
    bonus_token = bool(meta.get("bonus_token", True))
    shared_draft = bool(meta.get("mtp_shared_weights")
                        or meta.get("draft_shared_weights")
                        or meta.get("eagle_shared_weights"))
    layer_cost = _positive(
        "mtp_draft_layer_cost_ms",
        float(meta.get("mtp_draft_layer_cost_ms", 0.0) or 0.0), strict=False)
    declared_layer_cost = layer_cost if layer_cost > 0.0 else float(
        meta.get("mtp_draft_cost_ms", 0.0) or 0.0)
    accept_supplied = "mtp_accept_rate" in meta
    accept_rate = _positive("mtp_accept_rate",
                            float(meta.get("mtp_accept_rate", 0.75)), strict=False)
    if accept_rate > 1.0:
        raise ModelingError(f"xpyd_mtp_accept_rate_invalid:{accept_rate}")
    draft_spec = None
    if speculative:
        draft = max(1, draft)
        draft_ledger = memory_ledger(
            candidate, sequence_tokens=context, hccl_buffsize_mb=hccl_buffsize_mb,
            enable_eplb=False, resolved_draft=draft_spec_early)
        if float(draft_ledger.draft_weight_bytes) <= 0.0 and not (
                shared_draft and draft_spec_early is not None):
            return {"parallel": parallel, "plan": plan, "feasible": False,
                    "reason": "speculative_requires_draft_weight_evidence",
                    "world": plan["world"], "mu_requests_per_s": 0.0,
                    "route_qualified": route_ok,
                    "route_evidence": route_evidence}
        max_tokens = float(draft + (1 if bonus_token else 0))
        stats_k = draft if speculative else 0
        stats_accept = accept_rate if (speculative and accept_supplied) else None
        if "mtp_useful_tokens_per_step" in meta:
            useful = _positive("useful_tokens_per_step",
                               float(meta["mtp_useful_tokens_per_step"]))
            if useful < 1.0 - 1e-12 or useful > max_tokens + 1e-12:
                return {"parallel": parallel, "plan": plan, "feasible": False,
                        "reason": "speculative_yield_exceeds_declared_heads",
                        "world": plan["world"], "mu_requests_per_s": 0.0,
                        "route_qualified": route_ok,
                        "route_evidence": route_evidence}
            if speculative and accept_supplied:
                mean = speculative_accept_tokens(
                    accept_rate, draft, bonus_token=bonus_token)
                if abs(mean - useful) > 1e-9:
                    raise ModelingError(
                        f"xpyd_useful_tokens_conflict:{mean}!={useful}")
            # Scalar-only yield: renewal uses the exact two-point law with this
            # mean while the known K still bounds its support.
        else:
            useful = speculative_accept_tokens(accept_rate, draft,
                                               bonus_token=bonus_token)
            stats_accept = accept_rate
        draft_spec = draft_spec_early
    else:
        # A non-speculative plan never inherits a speculative token yield.
        useful = 1.0
        stats_accept = None
    draft_cost = _positive("mtp_draft_cost_ms", float(meta.get("mtp_draft_cost_ms", 0.0)),
                           strict=False)
    rollback = _positive("mtp_rollback_ms", float(meta.get("mtp_rollback_ms", 0.0)),
                         strict=False)
    host = _positive("mtp_host_ms", float(meta.get("mtp_host_ms", 0.0)), strict=False)
    chosen = None
    last_memory = None
    last_reason = "decode_memory_infeasible"
    memory_feasible_seen = False
    draft_step_used = 0.0
    draft_source_used = ""
    draft_demands: dict[str, float] = {}
    draft_ports: dict[str, float] = {}
    for batch in range(max_batch, 0, -1):
        epoch = epoch_for(batch) if eplb_active else None
        if eplb_active and epoch is None:
            last_reason = "eplb_epoch_mismatch"
            continue
        slots_b = epoch[0] if epoch else None
        staging_b = epoch[1] if epoch else None
        placement = epoch[2] if epoch else None
        rebalance = epoch[3] if epoch else 0.0
        verification_rows = batch * (1 + draft) if speculative else batch
        memory = ledger_for(slots_b, staging_b).at(verification_rows, context)
        last_memory = memory
        if not memory.oom_feasible:
            continue
        memory_feasible_seen = True
        common = dict(context=context, fabric=tree, routing_skew=routing_skew,
                      eplb_redundant_experts=redundant,
                      enable_eplb=eplb_active,
                      active_replicas=int(plan["active_replicas"]),
                      extra_expert_bytes_per_rank=0.0,
                      eplb_rebalance_ms=rebalance,
                      endpoints=endpoints,
                      dtype=evidence["dtype"], topology=evidence["topology"],
                      axis_groups=plan,
                      eplb_placement=placement)
        step = model.step(batch, per_replica_batch=batch, **common)
        step_ok = bool(step.route_qualified)
        step_evidence = step.route_evidence
        if speculative:
            verification = model.step(verification_rows, per_replica_batch=verification_rows,
                                      **common)
            step_ok = step_ok and bool(verification.route_qualified)
            if not verification.route_qualified:
                step_evidence = verification.route_evidence
            draft_step = layer_cost
            source = "declared_layer_cost" if layer_cost > 0.0 else ""
            parallel_draft = max(getattr(candidate.runtime.parallel, axis) or 1
                                 for axis in ("tp", "etp", "ep", "pp", "pcp", "dcp", "expert_dp"))
            # Duration source never decides whether actual draft payloads are
            # qualified: a complete draft geometry is always priced on the
            # assigned physical endpoints, even when a scalar duration wins.
            priced = None
            if draft_spec is not None:
                priced = priced_draft_step_ms(
                    candidate, draft_spec, batch=batch, context_tokens=context,
                    fabric=tree, endpoints=endpoints, dtype=evidence["dtype"],
                    topology=evidence["topology"], groups=plan.get("tp_groups"))
            if priced is not None:
                draft_step, source = priced.execution_duration(layer_cost)
                if not priced.route_qualified:
                    step_ok = False
                    step_evidence = priced.route_evidence
                draft_demands = dict(priced.comm_demands)
                draft_ports = dict(priced.port_demands)
            elif draft_spec is not None:
                # A draft model exists but its geometry cannot produce actual
                # payload bytes: never substitute a one-byte connectivity probe.
                step_ok = False
                step_evidence = "missing_draft_geometry"
            elif parallel_draft > 1 or shared_draft:
                # An auxiliary slot without a usable draft geometry is also an
                # unmeasured draft communicator, not evidence of connectivity.
                step_ok = False
                step_evidence = "missing_draft_geometry"
            if draft_step <= 0.0:
                last_reason = "speculative_draft_execution_unpriced"
                continue
            cycle = (verification.step_ms + draft * draft_step
                     + draft_cost + (1.0 - accept_rate) * rollback + host)
        else:
            cycle = step.step_ms
            draft_step, source = 0.0, ""
        if not step_ok and not allow_unqualified:
            return route_failure(
                step_evidence, draft_comm_demands=dict(draft_demands),
                draft_execution_source=source or None)
        if cycle <= tpot_slo_ms:
            traffic_ledger = (verification.comm_ledger if speculative
                              else step.comm_ledger)
            chosen = (batch, memory, cycle, traffic_ledger)
            draft_step_used, draft_source_used = draft_step, source
            break
    if chosen is None:
        reason = (last_reason if last_reason != "decode_memory_infeasible"
                  else "decode_tpot_sla_or_memory_exceeded" if memory_feasible_seen
                  else "decode_memory_infeasible")
        if (reason == "decode_memory_infeasible" and last_memory is not None
                and last_memory.status == "memory_unqualified"):
            reason = f"{last_memory.status}:{last_memory.reason}"
        return {"parallel": parallel, "plan": plan, "feasible": False,
                "reason": reason, "world": plan["world"],
                "mu_requests_per_s": 0.0, "route_qualified": route_ok,
                "route_evidence": route_evidence, "memory": last_memory,
                "kv_transfer_staging_bytes": staging_reserve,
                "bridge_lifetime_plan": (
                    transfer_plan.to_dict()
                    if hasattr(transfer_plan, "to_dict") else None)}
    batch, memory, cycle, traffic_ledger = chosen
    cycles = expected_decode_cycles(
        osl, stats_k if speculative else 0, stats_accept, useful,
        bonus_token=bonus_token)
    # One operation-derived traffic ledger: the verification step executes
    # ``1 + K`` target rows per request and ``K`` draft steps per request.
    # Normalize the emitted DP*B step bytes by the requests one step completes,
    # then scale by the exact renewal cycle count.
    try:
        shared_cut_demands, port_demands = _normalize_decode_demands(
            traffic_ledger, plan, batch=batch, cycles=cycles)
    except ModelingError as exc:
        return {"parallel": parallel, "plan": plan, "feasible": False,
                "reason": str(exc), "world": plan["world"],
                "mu_requests_per_s": 0.0, "route_qualified": False}
    if speculative and (draft_demands or draft_ports):
        draft_rounds = float(draft) * float(cycles)
        scale = draft_rounds / float(batch)
        shared_cut_demands = merge_demands(
            shared_cut_demands,
            {key: value * scale for key, value in draft_demands.items()})
        port_demands = merge_demands(
            port_demands,
            {key: value * scale for key, value in draft_ports.items()})
    mu = (completed_requests_per_s(
        batch, useful, osl, cycle,
        speculative_k=stats_k if speculative else 0,
        accept_rate=stats_accept, bonus_token=bonus_token)
        * int(plan["active_replicas"]))
    return {"parallel": parallel, "plan": plan, "feasible": True, "world": plan["world"],
            "memory": memory, "step_ms": cycle, "batch": batch,
            "b_kv": max_batch, "b_sla": batch,
            "useful_tokens_per_step": useful, "speculative": speculative,
            "draft_len": draft if speculative else 0,
            "bonus_token": bonus_token,
            "draft_step_ms": draft_step_used,
            "draft_execution_source": draft_source_used or None,
            "shared_draft": shared_draft,
            "device_ids": endpoints,
            "rank_offset": int(rank_offset),
            "itl_basis": "analytic_cycle_mean",
            "modeled_cycle_ms": cycle,
            "mu_requests_per_s": mu, "output_tokens_per_s": mu * osl,
            "tpot_ms": cycle, "route_qualified": bool(route_ok and step_ok),
            "decode_cycles": cycles,
            "committed_tokens_per_cycle": useful,
            "shared_cut_bytes_per_output_token": {
                key: value / float(osl)
                for key, value in shared_cut_demands.items()},
            "verification_rows_per_cycle": int(1 + draft) if speculative else 1,
            "draft_steps_per_cycle": int(draft) if speculative else 0,
            "kv_transfer_staging_bytes": staging_reserve,
            "bridge_lifetime_plan": (
                transfer_plan.to_dict()
                if hasattr(transfer_plan, "to_dict") else None),
            "kv_transfer_fields": dict(plan_fields),
            "kv_transfer_page_owner_ranks": list(plan_page_owners),
            "comm_persistent_qualified": bool(
                getattr(memory, "comm_persistent_qualified", False)),
            "comm_persistent_basis": str(
                getattr(memory, "comm_persistent_basis", "unknown")),
            "shared_cut_demands": shared_cut_demands,
            "port_demands": port_demands,
            "draft_comm_demands": dict(draft_demands),
            "fabric_route_qualified": bool(step_ok),
            "route_evidence": (step_evidence if not step_ok else route_evidence),
            "burst_itl_ms": cycle}

def _default_configs(case: CaseSpec, phase: str, total: int) -> list[ParallelSpec]:
    model, base = case.model, case.runtime.parallel
    moe = int(model.num_experts or 0) > 0
    worlds = [world for world in (1, 2, 4, 8, 16, 32, 64) if world <= total]
    result = []
    for world in worlds:
        ep = (world if (phase == "decode" and moe
                        and model.num_experts % world == 0) else 1)
        for tp in ({world, 1} if ep > 1 else {world}):
            if model.num_heads % tp != 0:
                continue
            dp = world // tp
            if dp < 1:
                continue
            if ep > 1 and ep > dp * tp:
                continue
            result.append(ParallelSpec(dp=dp, tp=tp, ep=ep, pp=1, cp=1, pcp=1))
    return result


@dataclass(frozen=True)
class _LanePlan:
    """Minimal lane view used only for remap byte accounting."""

    raw: Mapping[str, Any]

    @property
    def pcp_lane_policy(self) -> str:
        return str(self.raw.get("pcp_lane_policy", "park_extra_lanes"))

    @property
    def parked_lanes(self) -> int:
        return int(self.raw.get("parked_lanes", 0))

    @property
    def pcp(self) -> int:
        return int(self.raw.get("pcp", 1))

    @property
    def active_lanes(self) -> int:
        return int(self.raw.get("active_lanes", 1))



__all__ = ["DraftStepCost", "_LanePlan", "_default_configs", "_route_qualified",
           "p_to_d_path", "pool_config_facts", "priced_draft_step_ms"]
