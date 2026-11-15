"""Per-rank EPLB replica-weight accounting for the HBM ledger.

[Role] Turn an exact integer destination vector into per-rank, per-stage expert
  weight replicas plus an optional rebalance staging copy.
[Invariants] A rank with N destination slots is charged N weight copies; the
  total is never ``ceil(redundant/EP)`` and staging is not persistent weight.
  An unpinned vector is a conservative envelope over legal routing working
  sets, never one arbitrary epoch; a pinned vector is bound exactly.
[Timeouts & Locks] Bounded by MoE layers; no I/O or locks.
[Side-Effect Policy] Fresh tuples; caller Case untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.parallel_placement import expert_owner_slot as _owner_slot
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError

_ALIGNMENT = 512


@dataclass(frozen=True)
class EplbEpochConfig:
    """One validated EPLB enablement/count/slot declaration.

    ``slots_ep`` is the canonical ``EP``-length destination vector; ``slots_rank``
    lifts it onto every physical rank through the placement's EP coordinate.
    Every public entry point resolves this record before branching, so a supplied
    vector can never be ignored merely because the scalar redundant count was
    left at its default.
    """

    enable_eplb: bool
    redundant: int
    slots_ep: tuple[int, ...] | None
    slots_rank: tuple[int, ...] | None
    supplied: bool
    active: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"enable_eplb": self.enable_eplb, "redundant": self.redundant,
                "slots_ep": None if self.slots_ep is None else list(self.slots_ep),
                "slots_rank": (None if self.slots_rank is None
                               else list(self.slots_rank)),
                "supplied": self.supplied, "active": self.active,
                "source": self.source}


def resolve_eplb_epoch_config(
    *,
    enable_eplb: bool | None,
    redundant_experts: int | None,
    slot_vector: Sequence[int] | None,
    ep: int,
    world: int,
    num_experts: int,
    ep_coords: Sequence[int],
    top_k: int = 1,
    routing_skew: float = 1.0,
) -> EplbEpochConfig:
    """Canonical EPLB enablement/count/slot resolution for every API.

    Rules (fail closed, never silently ignore a supplied vector):

    * ``redundant_experts`` is a strict integer when supplied.
    * A supplied vector is strict-integer, non-negative, and either exactly
      ``EP`` long or a consistent lift of an ``EP``-length vector onto the
      world.
    * Positive slots with an explicitly disabled EPLB conflict.
    * A positive explicit count must equal the slot total; a zero/unspecified
      count infers ``R = sum(slots)`` from the supplied vector.
    * Every slot obeys ``0 <= slots[r] <= min(R, E - owner_count[r])``.
    """
    if enable_eplb is not None and not isinstance(enable_eplb, bool):
        raise ModelingError(f"eplb_enable_invalid:{enable_eplb}")
    if isinstance(ep, bool) or not isinstance(ep, int) or ep < 1:
        raise ModelingError(f"eplb_ep_invalid:{ep}")
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        raise ModelingError(f"eplb_world_invalid:{world}")
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) \
            or num_experts < 0:
        raise ModelingError(f"eplb_num_experts_invalid:{num_experts}")
    top = _resolve_int("eplb_top_k", top_k, 1)
    redundant: int | None
    if redundant_experts is None:
        redundant = None
    else:
        redundant = _resolve_int("eplb_redundant_experts", redundant_experts, 0)
    coordinates = tuple(int(value) for value in ep_coords)
    if len(coordinates) != world:
        raise ModelingError(f"eplb_ep_coords_invalid:{len(coordinates)}!={world}")
    if any(value < 0 or value >= ep for value in coordinates):
        raise ModelingError(f"eplb_ep_coords_invalid:{coordinates}")

    supplied = slot_vector is not None
    slots_ep: tuple[int, ...] | None = None
    slots_rank: tuple[int, ...] | None = None
    if supplied:
        raw = tuple(_resolve_int("eplb_slots", value, 0) for value in slot_vector)
        if len(raw) == ep:
            slots_ep = raw
        elif len(raw) == world:
            # Legacy world-length lift: it is accepted only when it is exactly
            # the EP vector repeated through the authoritative EP coordinate.
            canonical: list[int | None] = [None] * ep
            for rank, value in enumerate(raw):
                slot = coordinates[rank]
                if canonical[slot] is None:
                    canonical[slot] = value
                elif canonical[slot] != value:
                    raise ModelingError(
                        f"eplb_slots_not_ep_consistent:{rank}:{value}")
            slots_ep = tuple(0 if value is None else value for value in canonical)
        else:
            raise ModelingError(f"eplb_slots_length:{len(raw)}!={ep}")
        slots_rank = tuple(slots_ep[coordinates[rank]] for rank in range(world))

    positive = bool(slots_ep is not None and any(value > 0 for value in slots_ep))
    enabled = enable_eplb
    if positive and enabled is False:
        raise ModelingError("eplb_slots_conflict_with_disabled_eplb")

    if redundant is None or redundant == 0:
        effective = int(sum(slots_ep)) if slots_ep is not None else 0
        if effective > 0:
            enabled = True if enabled is None else enabled
            source = "supplied_slots"
        else:
            source = "scalar" if redundant is not None else "unset"
    else:
        effective = redundant
        if slots_ep is not None and sum(slots_ep) != effective:
            raise ModelingError(
                f"eplb_slots_undercount:{sum(slots_ep)}!={effective}")
        if enabled is None:
            enabled = True
        source = "scalar" if slots_ep is None else "supplied_slots"
    if effective < 0:
        raise ModelingError(f"eplb_redundant_experts_invalid:{effective}")
    if effective > 0 and enabled is None:
        enabled = True
    if enabled is None:
        enabled = False
    active = bool(enabled and effective > 0)

    if effective > 0 and slots_ep is not None:
        if num_experts <= 0:
            raise ModelingError("eplb_slots_require_experts")
        # The canonical contiguous owner map is defined for uneven ``E % EP``.
        owner_counts = [
            sum(1 for expert in range(num_experts)
                if _owner_slot(expert, num_experts, ep) == slot)
            for slot in range(ep)]
        for slot, value in enumerate(slots_ep):
            bound = min(effective, max(0, num_experts - owner_counts[slot]))
            if value > bound:
                raise ModelingError(
                    f"eplb_slots_exceed_bound:{slot}:{value}>{bound}")
    return EplbEpochConfig(bool(enabled), int(effective), slots_ep, slots_rank,
                           supplied, active, source)


def _one_expert_bytes(case: CaseSpec, plan, quant: QuantSpec) -> float:
    p = case.runtime.parallel
    total = 0.0
    for term in layer_weight_terms(plan, case.model):
        if term.category != "routed_expert" or int(term.count) <= 0:
            continue
        local_count = max(1, math.ceil(int(term.count) / max(1, int(p.ep))))
        total += float(term_bytes(term, quant, p, case.machine.ranks_per_node)) / local_count
    return total


def _eplb_extra_bytes(
    case: CaseSpec, mem: Any, redundant: int, *,
    slots_per_rank: Sequence[int] | None = None,
    staging_bytes_per_rank: Sequence[float] | None = None,
    rank_stage: Sequence[int] | None = None,
) -> tuple[list[float], list[list[float]], dict[str, float]]:
    """Exact per-rank replica weights, never ``ceil(redundant / EP)``.

    ``place_eplb`` owns the integer destination vector.  Every rank is charged
    for the replicas it actually owns; the rebalance staging copy is a full
    per-rank transient allocation (never divided by PP), attributed only to
    that rank's own stage.
    """
    from mindie_sim.parallel_placement import rank_parts

    p = case.runtime.parallel
    pp, world = int(p.pp), int(p.world_size)
    extra = [0.0] * pp
    per_rank = [[0.0] * pp for _ in range(world)]
    if redundant <= 0 and not staging_bytes_per_rank:
        return extra, per_rank, {}
    ep = max(1, int(p.ep))
    if slots_per_rank is None:
        from mindie_sim.moe_skew_eplb import conservative_eplb_slots
        # Unpinned admission: a per-rank envelope over legal working sets, not
        # one synthetic epoch that can leave another rank's copy uncharged.
        owners = conservative_eplb_slots(
            redundant, ep, int(case.model.num_experts or 0),
            top_k=max(1, int(case.model.top_k or 1)))
        slots = tuple(owners[rank_parts(rank, p)[5]] for rank in range(world))
    else:
        slots = tuple(_resolve_int("eplb_slots", value, 0)
                      for value in slots_per_rank)
        if len(slots) != world:
            raise ModelingError(f"memory_breakdown_eplb_slots_invalid:{slots}")
        if redundant > 0 and sum(slots) < redundant:
            raise ModelingError(f"memory_breakdown_eplb_slots_undercount:{sum(slots)}<{redundant}")
    staging = tuple(_non_negative("eplb_staging", value)
                    for value in (staging_bytes_per_rank or ()))
    if staging and len(staging) != world:
        raise ModelingError("memory_breakdown_eplb_staging_invalid")
    experts = int(case.model.num_experts or 0)
    if redundant > 0 and experts > 0:
        owner_counts = [
            sum(1 for expert in range(experts)
                if _owner_slot(expert, experts, ep) == slot)
            for slot in range(ep)]
        for rank, value in enumerate(slots):
            slot = rank_parts(rank, p)[5]
            non_local = experts - owner_counts[min(ep - 1, max(0, slot))]
            bound = min(redundant, non_local)
            if value > bound:
                raise ModelingError(
                    f"memory_breakdown_eplb_slots_exceed_bound:{value}>{bound}")
    stages_of = (tuple(int(value) for value in rank_stage)
                 if rank_stage is not None else
                 tuple(int(rank_parts(rank, p)[0]) for rank in range(world)))
    if len(stages_of) != world:
        raise ModelingError("memory_breakdown_eplb_rank_stage_invalid")
    quant = case.model.quant or QuantSpec("bf16")
    layer_bytes: list[float] = [0.0] * pp
    for plan in case.model.layer_plans:
        if plan.ffn != "moe" or not getattr(plan, "num_experts", 0):
            continue
        layer_bytes[mem.layer_stage[plan.index]] += _one_expert_bytes(case, plan, quant)
    for rank in range(world):
        stage = max(0, min(pp - 1, stages_of[rank]))
        value = slots[rank] * layer_bytes[stage]
        if staging:
            value += staging[rank]
        per_rank[rank][stage] = value
        extra[stage] = max(extra[stage], value)
    return extra, per_rank, {
        "redundant_experts": float(redundant),
        "max_replicas_per_rank": float(max(slots, default=0)),
        "replica_slots_sum": float(sum(slots)),
        "rebalance_staging_bytes_max": float(max(staging, default=0.0)),
    }



def _cache_detail(case: CaseSpec, mem: Any, tokens: int, sequences: int):
    stages = [dict() for _ in range(case.runtime.parallel.pp)]
    for plan in case.model.layer_plans:
        components = layer_cache_components(
            case.model, plan, case.runtime.parallel, tokens, capacity_padded=True,
            interleave=case.runtime.kv_block_size)
        stage = mem.layer_stage[plan.index]
        for name, value in components.items():
            stages[stage][name] = stages[stage].get(name, 0.0) + value * sequences
    return stages




def _non_negative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"memory_breakdown_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ModelingError(f"memory_breakdown_{name}_invalid:{value}")
    return number


def _positive(name: str, value: Any) -> float:
    number = _non_negative(name, value)
    if number <= 0:
        raise ModelingError(f"memory_breakdown_{name}_invalid:{value}")
    return number


def _resolve_int(name: str, value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"memory_breakdown_{name}_invalid:{value}")
    return value


def _align(value: float, alignment: int = _ALIGNMENT) -> float:
    return math.ceil(max(0.0, float(value)) / alignment) * alignment

