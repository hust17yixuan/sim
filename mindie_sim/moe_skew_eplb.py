"""MoE routing skew, integer EPLB placement and MTP batch reversal.

[Role]
  Turn a declared routing distribution into the discrete token load of the
  slowest EP rank, prove an integer min-max hot-expert replica placement with
  its epoch rebalance cost, and locate every batch crossing where a complete
  speculative cycle stops beating autoregressive TPOT.

[Invariants]
  * ``B=0`` has zero assignments, zero expert work and ``eta_skew=None``; a
    declared collective may still carry its startup separately.
  * ``routing_skew=1`` is exactly the finite multinomial case with rank
    variance ``n*p*(1-p)*(n+a)/(1+a)`` (``a`` is the popularity concentration),
    not the older ``(1-1/R)/n`` reading.  Results are integer ceilings bounded
    by ``B*top_k``.
  * The Cantelli tail bound is a separately named, risk-qualified diagnostic;
    ``max_rank_tokens`` is a declared expected-maximum estimate and is never
    labelled an exact P99.
  * Captured per-rank counts are accepted only if they are non-negative
    integers summing to ``B*top_k``.  They are never multiplied by peer count.
  * EPLB assigns each integer token demand ``x_(e,j)`` to an explicit replica
    rank.  A token is indivisible: no scalar shave can split one token across
    eight ranks.  Replica weight bytes and ``c/I`` rebalance cost are charged;
    negative net benefit is a valid result.
  * MTP compares the complete cycle (verification plus draft, rollback and
    host work) with ``g*T_base`` and returns every crossing, equality included.

[Timeouts & Locks] O(B*top_k log ranks + experts) bounded arithmetic; no I/O.
[Side-Effect Policy] Pure immutable records; caller Cases are never mutated.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import heapq
import math
from typing import Any

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.parallel_placement import expert_owner_slot
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError


def _int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError(f"moe_skew_{name}_invalid:{value}")
    return value


def _number(name: str, value: Any, minimum: float = 0.0, *, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"moe_skew_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ModelingError(f"moe_skew_{name}_invalid:{value}")
    return number


def _divisor(name: str, value: Any) -> float:
    return _number(name, value, 0.0, strict=True)


def order_statistic_z(ranks: int) -> float:
    """Expected maximum of ``ranks`` independent standard normal variables."""
    count = _int("ranks", ranks)
    if count < 2:
        return 0.0
    n = float(count)
    log_n = math.log(n)
    root = math.sqrt(2.0 * log_n)
    return max(0.0, root - (math.log(log_n) + math.log(4.0 * math.pi)) / (2.0 * root))


def _concentration(ranks: int, routing_skew: float) -> float:
    """Dirichlet concentration matching the declared popularity variance."""
    skew = _number("routing_skew", routing_skew, 1.0)
    extra = max(0.0, skew * skew - 1.0)
    if extra <= 0.0:
        return float("inf")
    return max(1e-9, (ranks - 1.0) / extra - 1.0)


def _rank_variance(assignments: float, ranks: int, concentration: float) -> float:
    p = 1.0 / float(ranks)
    if math.isinf(concentration):
        overdispersion = 1.0
    else:
        overdispersion = (assignments + concentration) / (1.0 + concentration)
    return max(0.0, assignments * p * (1.0 - p) * overdispersion)


def rank_load_cv(batch_tokens: int, top_k: int, ep: int, routing_skew: float,
                 num_experts: int) -> float:
    """CV of one rank's routed-token load under the declared concentration."""
    batch = _int("batch_tokens", batch_tokens, minimum=0)
    top = _int("top_k", top_k, minimum=0)
    ranks = _int("ep", ep)
    _int("num_experts", num_experts)
    assignments = float(batch * top)
    if assignments <= 0:
        return 0.0
    variance = _rank_variance(assignments, ranks, _concentration(ranks, routing_skew))
    mean = assignments / ranks
    return math.sqrt(variance) / mean if mean > 0 else 0.0


def eta_skew(rank_counts: Sequence[int]) -> float | None:
    """``max_r n_r / mean``; ``None`` for an empty assignment set."""
    if not rank_counts:
        raise ModelingError("moe_skew_rank_counts_empty")
    values = []
    for value in rank_counts:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"moe_skew_rank_count_invalid:{value}")
        values.append(value)
    total = sum(values)
    if total == 0:
        return None
    return max(values) / (total / len(values))


def max_rank_tokens(batch_tokens: int, top_k: int, ep: int, routing_skew: float = 1.0,
                    *, num_experts: int | None = None,
                    captured_skew_ratio: float | None = None,
                    captured_rank_counts: Sequence[int] | None = None,
                    risk_delta: float = 0.05,
                    bound: str = "cantelli") -> float:
    """Discrete straggler rank load (``CombineV3`` barrier height).

    The ranking default is the declared Cantelli upper risk quantile, not a
    bare Gaussian mean.  ``bound='expected_max'`` remains available for
    diagnostics and is labelled as an expected-value estimate.
    """
    batch = _int("batch_tokens", batch_tokens, minimum=0)
    top = _int("top_k", top_k, minimum=0)
    ranks = _int("ep", ep)
    experts = _int("num_experts", num_experts if num_experts is not None else max(ranks * 8, 128))
    if top > experts:
        raise ModelingError(f"moe_skew_top_k_exceeds_experts:{top}>{experts}")
    assignments = float(batch * top)
    counts = None
    if captured_rank_counts is not None:
        counts = tuple(captured_rank_counts)
        if not counts:
            raise ModelingError("moe_skew_captured_rank_counts_empty")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in counts):
            raise ModelingError(f"moe_skew_captured_rank_counts_invalid:{counts}")
        if len(counts) != ranks:
            raise ModelingError(f"moe_skew_captured_rank_counts_invalid:{len(counts)}!={ranks}")
        if sum(counts) != int(assignments):
            raise ModelingError(
                f"moe_skew_captured_rank_counts_not_conserved:{sum(counts)}!={int(assignments)}")
        # One token selects a given expert at most once.  Owner rank ``r`` can
        # therefore receive at most ``B * min(top_k, owner_population[r])``
        # assignments, using the exact contiguous owner population (never the
        # uniform ``ceil(E/EP)`` approximation).
        owner_population = [
            sum(1 for expert in range(experts)
                if expert_owner_slot(expert, experts, ranks) == slot)
            for slot in range(ranks)]
        for rank, value in enumerate(counts):
            cap = batch * min(top, owner_population[rank])
            if value > cap:
                raise ModelingError(
                    f"moe_skew_captured_rank_counts_infeasible:"
                    f"{rank}:{value}>{cap}")
    if assignments <= 0:
        return 0.0
    if counts is not None:
        return float(max(counts))
    if top >= experts and captured_skew_ratio is None:
        # Every token selects *every* expert.  The integer load of EP slot ``r``
        # is therefore ``B * owner_count[r]``; with an uneven ``E % EP`` the
        # maximum owner population is ``ceil(E/EP)``, never the fractional
        # ``B*E/EP``.  ``max_rank_tokens(1, 3, 2, 1.0, num_experts=3) == 2``.
        owner_counts = [
            sum(1 for expert in range(experts)
                if expert_owner_slot(expert, experts, ranks) == slot)
            for slot in range(ranks)]
        return float(batch * max(owner_counts))
    mean = assignments / ranks
    if captured_skew_ratio is not None:
        ratio = _number("captured_skew_ratio", captured_skew_ratio, 1.0)
        return min(assignments, float(math.ceil(mean * max(1.0, ratio))))
    concentration = _concentration(ranks, routing_skew)
    variance = _rank_variance(assignments, ranks, concentration)
    if bound == "expected_max":
        value = mean + order_statistic_z(ranks) * math.sqrt(variance)
    elif bound == "cantelli":
        risk = _number("risk_delta", risk_delta, 0.0, strict=True)
        if risk >= 1.0:
            raise ModelingError(f"moe_skew_risk_delta_invalid:{risk}")
        value = mean + math.sqrt(max(0.0, variance * (ranks / risk - 1.0)))
    else:
        raise ModelingError(f"moe_skew_bound_invalid:{bound}")
    return min(assignments, float(math.ceil(max(0.0, value) - 1e-12)))


def expert_demand(batch_tokens: int, top_k: int, num_experts: int,
                  routing_skew: float = 1.0) -> tuple[int, ...]:
    """Integer per-expert demand summing to ``B*top_k`` without replacement.

    One token selects at most one copy of an expert, hence
    ``max(demand) <= batch_tokens`` for every routing skew.  The previous
    gravity model could return ``(3, 1, 0, 0)`` for one token and ``top_k=4``.
    """
    batch = _int("batch_tokens", batch_tokens, minimum=0)
    top = _int("top_k", top_k, minimum=0)
    experts = _int("num_experts", num_experts)
    if top > experts:
        raise ModelingError(f"moe_skew_top_k_exceeds_experts:{top}>{experts}")
    assignments = int(batch * top)
    if assignments == 0:
        return (0,) * experts
    skew = _number("routing_skew", routing_skew, 1.0)
    weights = [1.0 + (skew - 1.0) * math.exp(-index / max(1.0, experts / 8.0))
               for index in range(experts)]
    total = sum(weights)
    raw = [assignments * weight / total for weight in weights]
    demand = [min(batch, int(math.floor(value))) for value in raw]
    remainder = assignments - sum(demand)
    if remainder < 0:
        raise ModelingError("moe_skew_expert_demand_not_conserved")
    while remainder > 0:
        order = sorted((index for index in range(experts)
                        if demand[index] < batch),
                       key=lambda index: (-(raw[index] - demand[index]), index))
        if not order:
            raise ModelingError("moe_skew_expert_demand_capacity_exhausted")
        for index in order:
            if remainder <= 0:
                break
            demand[index] += 1
            remainder -= 1
    if sum(demand) != assignments or any(value > batch for value in demand):
        raise ModelingError("moe_skew_expert_demand_not_conserved")
    return tuple(demand)


def canonical_eplb_slots(redundant_experts: int, ep: int, num_experts: int, *,
                         top_k: int = 1, routing_skew: float = 1.0) -> tuple[int, ...]:
    """Exact destination vector of the canonical ``batch_tokens == EP`` epoch.

    This is one concrete placement epoch, not a bound over other legal routing
    working sets.  ``place_eplb(8, 1, 8, 100, num_experts=16,
    redundant_experts=5)`` returns ``(0,0,0,1,1,1,1,1)`` while this canonical
    epoch returns ``(0,0,0,0,2,1,1,1)``; rank 3 is therefore charged zero
    replicas by the exact epoch.  Callers that need an HBM admission bound must
    use :func:`conservative_eplb_slots` or pin the active vector explicitly.
    """
    count = _int("ep", ep)
    redundant = _int("redundant_experts", redundant_experts, minimum=0)
    experts = _int("num_experts", num_experts)
    top = _int("top_k", top_k, minimum=0)
    if redundant <= 0 or count < 2:
        return (0,) * count
    placement = place_eplb(
        count, max(1, top), count, _number("routing_skew", routing_skew, 1.0),
        num_experts=experts, redundant_experts=redundant)
    slots = tuple(int(value) for value in placement.redundant_slots_per_rank)
    return slots if len(slots) == count else (0,) * count


def conservative_eplb_slots(redundant_experts: int, ep: int, num_experts: int, *,
                            top_k: int = 1, routing_skew: float = 1.0,
                            ) -> tuple[int, ...]:
    """Proven tight unpinned per-rank destination envelope for HBM admission.

    ``place_eplb`` never places expert ``e`` on its base owner
    ``expert_owner_slot(e, E, EP)`` and never places one expert twice on the
    same rank.  Hence across *every* legal ``(batch_tokens, routing_skew)``
    epoch EP slot ``r`` can hold at most

        ``min(R, E - base_owner_count[r])``

    replicas: at most all ``R`` redundant experts, and only experts not already
    base-owned by ``r``.  The bound is attained (rank 7 receives all five
    replicas in ``place_eplb(13, 1, 8, 1.0, num_experts=16,
    redundant_experts=5)``), so it is tight and needs no batch sampling.
    Callers that know the active routing working set pass the exact
    ``place_eplb`` vector instead.
    """
    count = _int("ep", ep)
    redundant = _int("redundant_experts", redundant_experts, minimum=0)
    experts = _int("num_experts", num_experts)
    # Kept for signature compatibility and boundary validation only: the bound
    # is routing- and batch-independent.
    _number("routing_skew", routing_skew, 1.0)
    _int("top_k", top_k, minimum=0)
    if redundant <= 0 or count < 2:
        return (0,) * count
    owner_counts = [
        sum(1 for expert in range(experts)
            if expert_owner_slot(expert, experts, count) == slot)
        for slot in range(count)]
    return tuple(min(redundant, experts - owners) for owners in owner_counts)


def replicated_expert_bytes(case: CaseSpec, redundant_experts: int, *,
                            redundant_slots_per_rank: Sequence[int] | None = None,
                            rebalance_staging_bytes_per_rank: Sequence[float] | None = None,
                            ) -> dict[str, Any]:
    """Exact per-rank HBM bytes of the EPLB destination vector.

    ``redundant_slots_per_rank`` comes from ``place_eplb``; a rank with two
    destination replicas is charged two weight copies even when
    ``ceil(redundant_experts / EP)`` is one.  Without an explicit vector the
    canonical min-max placement is used, never the rounded ``ceil`` bound.
    """
    parsed = case
    redundant = _int("redundant_experts", redundant_experts, minimum=0)
    model, p = parsed.model, parsed.runtime.parallel
    quant = model.quant or QuantSpec("bf16")
    world = int(p.world_size)
    if redundant_slots_per_rank is None:
        from mindie_sim.parallel_placement import rank_parts
        owners = conservative_eplb_slots(
            redundant, int(p.ep), int(model.num_experts or 0),
            top_k=max(1, int(model.top_k or 1)))
        slots = tuple(owners[rank_parts(rank, p)[5]] for rank in range(world))
    else:
        slots = tuple(_int("redundant_slots_per_rank", value, minimum=0)
                      for value in redundant_slots_per_rank)
        if len(slots) != world:
            raise ModelingError("moe_skew_redundant_slots_length")
        if redundant <= 0 and any(slots):
            redundant = int(sum(slots))
        if redundant > 0 and sum(slots) < redundant:
            raise ModelingError("moe_skew_redundant_slots_undercount")
    experts = int(model.num_experts or 0)
    if redundant > 0 and experts > 0:
        from mindie_sim.parallel_placement import rank_parts
        ep = max(1, int(p.ep))
        owner_counts = [
            sum(1 for expert in range(experts)
                if expert_owner_slot(expert, experts, ep) == slot)
            for slot in range(ep)]
        for rank, value in enumerate(slots):
            slot = rank_parts(rank, p)[5]
            non_local = experts - owner_counts[min(ep - 1, max(0, slot))]
            bound = min(redundant, non_local)
            if value > bound:
                raise ModelingError(
                    f"moe_skew_redundant_slots_exceed_bound:{value}>{bound}")
    staging = tuple(_number("rebalance_staging", value)
                    for value in (rebalance_staging_bytes_per_rank or ()))
    if staging and len(staging) != world:
        raise ModelingError("moe_skew_rebalance_staging_length")
    partition = parsed.runtime.pp_layer_partition
    if len(partition) != p.pp or sum(partition) != model.layers:
        partition = tuple(model.layers // p.pp + (i < model.layers % p.pp)
                          for i in range(p.pp))
    stage_of = tuple(stage for stage, count in enumerate(partition) for _ in range(count))
    layer_bytes = [0.0] * p.pp
    for plan in model.layer_plans:
        if plan.ffn != "moe" or not getattr(plan, "num_experts", 0):
            continue
        one_expert = 0.0
        for term in layer_weight_terms(plan, model):
            if term.category != "routed_expert" or int(term.count) <= 0:
                continue
            local_count = max(1, math.ceil(int(term.count) / max(1, int(p.ep))))
            one_expert += float(term_bytes(term, quant, p,
                                           parsed.machine.ranks_per_node)) / local_count
        layer_bytes[stage_of[plan.index]] += one_expert
    from mindie_sim.parallel_placement import rank_parts
    rank_stage = tuple(int(rank_parts(rank, p)[0]) for rank in range(world))
    per_rank_stage = [[0.0] * p.pp for _ in range(world)]
    for rank in range(world):
        stage = max(0, min(p.pp - 1, rank_stage[rank]))
        # A rank holds its own stage's replica weights and its full per-rank
        # rebalance staging buffer, never 1/PP of that buffer.
        per_rank_stage[rank][stage] = (slots[rank] * layer_bytes[stage]
                                       + (staging[rank] if staging else 0.0))
    per_stage = [max((per_rank_stage[rank][stage] for rank in range(world)
                      if rank_stage[rank] == stage), default=0.0)
                 for stage in range(p.pp)]
    per_rank = [max(row) for row in per_rank_stage]
    return {"per_rank_replicas": max(slots, default=0),
            "slots_per_rank": slots,
            "per_stage_bytes": tuple(per_stage),
            "per_rank_bytes": tuple(per_rank),
            "bytes_per_rank": float(max(per_rank, default=0.0)),
            "one_expert_bytes_per_layer": (sum(layer_bytes) / max(1, len(model.layer_plans))),
            "total_replica_layers": float(sum(slots))}


@dataclass(frozen=True)
class EplbPlacement:
    assignments: tuple[tuple[tuple[int, int], ...], ...]
    rank_tokens: tuple[int, ...]
    rank_service_ms: tuple[float, ...]
    max_rank_service_ms: float
    baseline_max_rank_service_ms: float
    replica_ranks: tuple[tuple[int, ...], ...]
    feasible: bool
    reason: str | None
    extra_weight_bytes_per_rank: float
    rebalance_ms: float
    net_benefit_ms: float
    redundant_slots_per_rank: tuple[int, ...] = ()
    rebalance_staging_bytes_per_rank: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [[[expert, rank, tokens] for expert, rank, tokens
                             in row] for row in self.assignments],
            "rank_tokens": list(self.rank_tokens),
            "rank_service_ms": list(self.rank_service_ms),
            "max_rank_service_ms": self.max_rank_service_ms,
            "baseline_max_rank_service_ms": self.baseline_max_rank_service_ms,
            "replica_ranks": [list(row) for row in self.replica_ranks],
            "feasible": self.feasible, "reason": self.reason,
            "extra_weight_bytes_per_rank": self.extra_weight_bytes_per_rank,
            "rebalance_ms": self.rebalance_ms,
            "net_benefit_ms": self.net_benefit_ms,
            "redundant_slots_per_rank": list(self.redundant_slots_per_rank),
            "rebalance_staging_bytes_per_rank": self.rebalance_staging_bytes_per_rank,
        }


class _Dinic:
    """Small integer max-flow used to prove an exact min-max token placement."""

    def __init__(self, nodes: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(nodes)]

    def add(self, source: int, target: int, capacity: int) -> int:
        forward = [target, capacity, len(self.graph[target])]
        backward = [source, 0, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(backward)
        return len(self.graph[source]) - 1

    def flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = [source]
            for node in queue:
                for target, capacity, _ in self.graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            cursor = [0] * len(self.graph)

            def push(node: int, limit: int) -> int:
                if node == sink:
                    return limit
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, capacity, reverse = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        sent = push(target, min(limit, capacity))
                        if sent > 0:
                            edge[1] -= sent
                            self.graph[target][reverse][1] += sent
                            return sent
                    cursor[node] += 1
                return 0

            while True:
                sent = push(source, 1 << 60)
                if sent <= 0:
                    break
                total += sent


def _flow_placement(demand: Sequence[int], replicas: Sequence[Sequence[int]],
                    capacity: Sequence[int] | None, limit: int, ranks: int):
    experts = len(demand)
    source, sink = experts + ranks, experts + ranks + 1
    graph = _Dinic(sink + 1)
    for expert, count in enumerate(demand):
        if count <= 0:
            continue
        graph.add(source, expert, int(count))
        for rank in dict.fromkeys(int(item) for item in replicas[expert]):
            graph.add(expert, experts + rank, int(count))
    for rank in range(ranks):
        cap = int(capacity[rank]) if capacity is not None else limit
        graph.add(experts + rank, sink, max(0, min(cap, limit)))
    value = graph.flow(source, sink)
    if value != sum(int(item) for item in demand):
        return None
    loads = [0] * ranks
    assignment: list[list[tuple[int, int, int]]] = [[] for _ in range(ranks)]
    for expert, count in enumerate(demand):
        if count <= 0:
            continue
        for edge in graph.graph[expert]:
            target, remaining, _ = edge
            if target >= experts and target < source:
                sent = int(count) - int(remaining)
                if sent > 0:
                    rank = target - experts
                    loads[rank] += sent
                    assignment[rank].append((expert, rank, sent))
    return tuple(loads), tuple(tuple(row) for row in assignment)


def _min_max_placement(demand: Sequence[int], replicas: Sequence[Sequence[int]],
                       capacity: Sequence[int] | None, ranks: int):
    if not demand:
        return None
    upper = max(sum(int(item) for item in demand), 1)
    lower = 0
    best = None
    while lower <= upper:
        middle = (lower + upper) // 2
        result = _flow_placement(demand, replicas, capacity, middle, ranks)
        if result is not None:
            best = result
            upper = middle - 1
        else:
            lower = middle + 1
    return best


def _place_once(demand: Sequence[int], ep: int, replicas: Sequence[Sequence[int]],
                capacity: Sequence[int] | None, per_token_ms: float, fixed_ms: float):
    placed = _min_max_placement(demand, replicas, capacity, ep)
    if placed is None:
        return None
    loads, assignment = placed
    service = tuple(fixed_ms + per_token_ms * value for value in loads)
    return loads, service, assignment


def place_eplb(batch_tokens: int, top_k: int, ep: int, routing_skew: float, *,
               num_experts: int, redundant_experts: int,
               per_token_ms: float = 1.0, fixed_ms: float = 0.0,
               rank_token_capacity: Sequence[int] | None = None,
               extra_weight_bytes_per_rank: float = 0.0,
               rebalance_bw_gbps: float | None = None,
               rebalance_interval_steps: int = 1024,
               replica_efficiency: float = 1.0) -> EplbPlacement:
    """Exact integer min-max assignment for a greedy replica set."""
    batch = _int("batch_tokens", batch_tokens, minimum=0)
    top = _int("top_k", top_k, minimum=0)
    ranks = _int("ep", ep)
    experts = _int("num_experts", num_experts)
    redundant = _int("redundant_experts", redundant_experts, minimum=0)
    interval = _int("rebalance_interval_steps", rebalance_interval_steps)
    per_token = _number("per_token_ms", per_token_ms, 0.0, strict=True)
    fixed = _number("fixed_ms", fixed_ms)
    efficiency = _number("replica_efficiency", replica_efficiency)
    if efficiency > 1.0:
        raise ModelingError(f"moe_skew_replica_efficiency_invalid:{efficiency}")
    extra = _number("extra_weight_bytes_per_rank", extra_weight_bytes_per_rank)
    capacity = None if rank_token_capacity is None else tuple(
        _int("rank_token_capacity", value, minimum=0) for value in rank_token_capacity)
    if capacity is not None and len(capacity) != ranks:
        raise ModelingError("moe_skew_rank_capacity_length")
    demand = expert_demand(batch, top, experts, routing_skew)
    base_replicas = tuple((expert_owner_slot(expert, experts, ranks),)
                          for expert in range(experts))
    slots = [0] * ranks
    baseline = _place_once(demand, ranks, base_replicas, capacity, per_token, fixed)
    if baseline is None:
        return EplbPlacement((), (0,) * ranks, (fixed,) * ranks, float("inf"),
                             float("inf"), tuple(() for _ in range(experts)), False,
                             "baseline_capacity_infeasible", extra, 0.0, 0.0,
                             tuple(slots), extra)
    replicas = [list(row) for row in base_replicas]
    current = baseline
    hot = sorted(range(experts), key=lambda expert: (-demand[expert], expert))
    hot = [expert for expert in hot if demand[expert] > 0][:max(2, redundant * 3)]
    added, capacity_legal = 0, False
    for _ in range(redundant):
        best = None
        for expert in hot:
            candidates = [rank for rank in range(ranks) if rank not in replicas[expert]]
            candidates.sort(key=lambda rank: (current[0][rank], rank))
            for candidate in candidates[:4]:
                trial = [list(row) for row in replicas]
                trial[expert].append(candidate)
                placed = _place_once(demand, ranks, trial, capacity, per_token, fixed)
                if placed is None:
                    continue
                capacity_legal = True
                key = (max(placed[1]), placed[1][candidate], expert, candidate)
                if best is None or key < best[0]:
                    best = (key, expert, candidate, trial, placed)
        if best is None:
            break
        _key, expert, candidate, trial, placed = best
        replicas = trial
        current = placed
        added += 1
    loads, service, assignment = current
    for expert, row in enumerate(replicas):
        base_rank = expert_owner_slot(expert, experts, ranks)
        for rank in row:
            if int(rank) != base_rank:
                slots[int(rank)] += 1
    rebalance_ms = 0.0
    if extra > 0.0:
        if rebalance_bw_gbps is None:
            raise ModelingError("moe_skew_rebalance_bw_required_with_extra_bytes")
        bandwidth = _divisor("rebalance_bw_gbps", rebalance_bw_gbps)
        rebalance_ms = extra / (bandwidth * 1e9) * 1000.0 / interval
    shave = max(0.0, max(baseline[1]) - max(service))
    net = efficiency * shave - rebalance_ms
    if redundant > 0 and added == 0 and any(demand) and not capacity_legal:
        return EplbPlacement(assignment, loads, service, max(service), max(baseline[1]),
                             tuple(tuple(row) for row in replicas), False,
                             "no_feasible_receiving_rank", extra, rebalance_ms, net,
                             tuple(slots), extra)
    return EplbPlacement(assignment, loads, service, max(service), max(baseline[1]),
                         tuple(tuple(row) for row in replicas), True, None,
                         extra, rebalance_ms, net, tuple(slots), extra)


@dataclass(frozen=True)
class EplbEffect:
    baseline_peak_tokens: float
    peak_tokens: float
    shaved_tokens: float
    hot_expert_tokens: float
    extra_weight_bytes_per_rank: float
    amortized_rebalance_ms: float
    redundant_experts: int
    replica_efficiency: float
    feasible: bool = True
    placement: EplbPlacement | None = None
    net_benefit_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_peak_tokens": self.baseline_peak_tokens,
            "peak_tokens": self.peak_tokens, "shaved_tokens": self.shaved_tokens,
            "hot_expert_tokens": self.hot_expert_tokens,
            "extra_weight_bytes_per_rank": self.extra_weight_bytes_per_rank,
            "amortized_rebalance_ms": self.amortized_rebalance_ms,
            "redundant_experts": self.redundant_experts,
            "replica_efficiency": self.replica_efficiency,
            "feasible": self.feasible,
            "net_benefit_ms": self.net_benefit_ms,
            "placement": self.placement.to_dict() if self.placement else None,
        }


def eplb_effect(batch_tokens: int, top_k: int, ep: int, routing_skew: float,
                *, num_experts: int, redundant_experts: int,
                extra_weight_bytes_per_rank: float = 0.0,
                rebalance_bw_gbps: float | None = None,
                rebalance_interval_steps: int = 1024,
                replica_efficiency: float = 1.0,
                per_token_ms: float = 1.0) -> EplbEffect:
    """Replay an integer replica placement; never a free scalar shave."""
    placement = place_eplb(
        batch_tokens, top_k, ep, routing_skew, num_experts=num_experts,
        redundant_experts=redundant_experts, per_token_ms=per_token_ms,
        extra_weight_bytes_per_rank=extra_weight_bytes_per_rank,
        rebalance_bw_gbps=rebalance_bw_gbps,
        rebalance_interval_steps=rebalance_interval_steps,
        replica_efficiency=replica_efficiency)
    return EplbEffect(
        baseline_peak_tokens=placement.baseline_max_rank_service_ms,
        peak_tokens=placement.max_rank_service_ms,
        shaved_tokens=max(0.0, placement.baseline_max_rank_service_ms
                          - placement.max_rank_service_ms),
        hot_expert_tokens=float(max(expert_demand(batch_tokens, top_k, num_experts,
                                                  routing_skew), default=0)),
        extra_weight_bytes_per_rank=placement.extra_weight_bytes_per_rank,
        amortized_rebalance_ms=placement.rebalance_ms,
        redundant_experts=_int("redundant_experts", redundant_experts, minimum=0),
        replica_efficiency=_number("replica_efficiency", replica_efficiency),
        feasible=placement.feasible, placement=placement,
        net_benefit_ms=placement.net_benefit_ms)



from mindie_sim.moe_mtp import (MtpReversal, mtp_batch_reversal,
                               speculative_accept_tokens)  # noqa: E402

__all__ = [
    "EplbEffect", "EplbPlacement", "MtpReversal", "canonical_eplb_slots",
    "conservative_eplb_slots", "eta_skew", "eplb_effect",
    "expert_demand", "max_rank_tokens", "mtp_batch_reversal", "order_statistic_z",
    "place_eplb", "rank_load_cv", "replicated_expert_bytes",
    "speculative_accept_tokens",
]
