"""Complete speculative/MTP cycle accounting.

[Role] Price verification, draft, rollback and host work; return every batch
  crossing including equality; respect ``bonus_token`` and full-rank memory
  evidence.
[Invariants] A non-speculative plan has yield 1.0.  Without the bonus token
  ``K=1, alpha=1`` commits one token, never two.  Memory qualification requires
  explicit rank bytes and usable capacity, not an auxiliary-only check; a
  per-batch callback qualifies every batch in the grid on the limiting rank
  instead of one scalar for the whole sweep.  ``burst_itl_ms`` is an analytic
  cycle mean, not a committed-token timestamp trace.
[Timeouts & Locks] O(grid) bounded arithmetic; no I/O or device context.
[Side-Effect Policy] Pure immutable records.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.schema.common import ModelingError


def _int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError(f"moe_skew_{name}_invalid:{value}")
    return value


def _number(name: str, value: Any, minimum: float = 0.0, *,
            strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"moe_skew_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ModelingError(f"moe_skew_{name}_invalid:{value}")
    return number


def _cumulative_pmf(draft: int, probabilities: Sequence[float], *,
                    bonus_token: bool) -> tuple[float, ...]:
    """Accepted-token law from cumulative per-draft acceptance probabilities.

    ``probabilities[i]`` is the probability that drafts ``0..i`` are all
    accepted (a non-increasing sequence starting at ``<= 1``).  With the bonus
    token the outcomes are ``1..K+1``; without it the final draft position is
    capped into ``K`` committed tokens.  This is exactly the geometric law of
    :func:`speculative_step_pmf` when ``probabilities[i] == alpha ** (i+1)``.
    """
    if isinstance(probabilities, (str, bytes)) or not isinstance(
            probabilities, Sequence):
        raise ModelingError(
            f"moe_skew_accept_probability_invalid:{probabilities}")
    if len(probabilities) != draft:
        raise ModelingError("moe_skew_accept_probability_length")
    previous = 1.0
    result: list[float] = []
    for index, value in enumerate(probabilities):
        probability = _number("accept_probability", value, 0.0)
        if probability > 1.0 or probability > previous + 1e-12:
            raise ModelingError(f"moe_skew_accept_probability_order:{index}")
        result.append(max(0.0, previous - probability))
        previous = probability
    if bonus_token:
        result.append(previous)
    elif result:
        result[-1] += previous
    total = float(sum(result))
    if abs(total - 1.0) > 1e-9:
        raise ModelingError(f"moe_skew_accept_probability_not_conserved:{total}")
    return tuple(float(value) for value in result)


def speculative_step_pmf(k: int, alpha: float, *,
                         bonus_token: bool = True,
                         accept_probabilities: Sequence[float] | None = None,
                         ) -> tuple[float, ...]:
    """One canonical discrete accepted-token law for every consumer.

    Index ``j-1`` is ``P(Y = j)``.  With the bonus token the outcomes are
    ``1..K+1``; without it a step is capped at ``max(1, K)`` committed tokens.
    An explicit cumulative-probability vector is normalized through the same
    law, never through a second ``1 + sum`` formula.  Both branches sum to
    exactly one and are the only source of the yield ``g = E[Y]`` and of the
    finite-OSL renewal DP.
    """
    draft = _int("draft_len", k, minimum=0)
    rate = _number("accept_rate", alpha, minimum=0.0)
    if rate > 1.0:
        raise ModelingError(f"moe_skew_accept_rate_invalid:{rate}")
    if draft == 0:
        if accept_probabilities is not None and len(tuple(accept_probabilities)):
            raise ModelingError("moe_skew_accept_probability_length")
        return (1.0,)
    if accept_probabilities is not None:
        return _cumulative_pmf(draft, accept_probabilities,
                               bonus_token=bonus_token)
    if bonus_token:
        probabilities = [rate ** (index - 1) * (1.0 - rate)
                         for index in range(1, draft + 1)]
        probabilities.append(rate ** draft)
    elif draft == 1:
        probabilities = [1.0]
    else:
        probabilities = [rate ** (index - 1) * (1.0 - rate)
                         for index in range(1, draft)]
        probabilities.append(rate ** (draft - 1))
    return tuple(float(value) for value in probabilities)


def speculative_accept_tokens(accept_rate: float, draft_len: int, *,
                              bonus_token: bool = True,
                              accept_probabilities: Sequence[float] | None = None
                              ) -> float:
    """Expected committed tokens per verification cycle from the one PMF."""
    pmf = speculative_step_pmf(draft_len, accept_rate,
                               bonus_token=bonus_token,
                               accept_probabilities=accept_probabilities)
    return float(sum((index + 1) * probability
                     for index, probability in enumerate(pmf)))


@dataclass(frozen=True)
class MtpReversal:
    draft_len: int
    accept_rate: float
    accepted_tokens_per_step: float
    b_reversal: int | None
    tpot_base_ms: float | None
    tpot_mtp_ms: float | None
    speedup_at_one: float
    curve: tuple[tuple[int, float, float], ...]
    reason: str
    crossings: tuple[int, ...] = ()
    memory_qualified: bool = False
    draft_rollback_bytes_per_rank: float = 0.0
    first_crossing_curve: tuple[tuple[int, float, float], ...] = ()
    burst_itl_ms: float | None = None
    amortized_useful_token_ms: float | None = None
    itl_basis: str = "analytic_cycle_mean"
    modeled_cycle_ms: float | None = None
    memory_by_batch: tuple[tuple[int, float, float, bool], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_len": self.draft_len, "accept_rate": self.accept_rate,
            "accepted_tokens_per_step": self.accepted_tokens_per_step,
            "b_reversal": self.b_reversal,
            "tpot_base_ms": self.tpot_base_ms, "tpot_mtp_ms": self.tpot_mtp_ms,
            "speedup_at_one": self.speedup_at_one,
            "curve": [{"b": b, "tpot_base_ms": base, "tpot_mtp_ms": mtp}
                      for b, base, mtp in self.curve],
            "crossings": list(self.crossings),
            "reason": self.reason,
            "memory_qualified": self.memory_qualified,
            "draft_rollback_bytes_per_rank": self.draft_rollback_bytes_per_rank,
            "first_crossing_curve": [
                {"b": b, "tpot_base_ms": base, "tpot_mtp_ms": mtp}
                for b, base, mtp in self.first_crossing_curve],
            "burst_itl_ms": self.burst_itl_ms,
            "amortized_useful_token_ms": self.amortized_useful_token_ms,
            "itl_basis": self.itl_basis,
            "modeled_cycle_ms": self.modeled_cycle_ms,
            "memory_by_batch": [{"b": b, "required_bytes": required,
                                 "capacity_bytes": capacity, "qualified": ok}
                                for b, required, capacity, ok in self.memory_by_batch],
        }


def mtp_batch_reversal(step_ms: Callable[[int], float], *, draft_len: int = 3,
                       accept_rate: float = 0.75, max_batch: int = 4096,
                       cycle_ms: Callable[[int, int], float] | None = None,
                       draft_cost_ms: float = 0.0,
                       draft_layer_cost_ms: float = 0.0,
                       rollback_ms: float = 0.0, host_ms: float = 0.0,
                       bonus_token: bool = True,
                       accept_probabilities: Sequence[float] | None = None,
                       batch_grid: Sequence[int] | None = None,
                       draft_rollback_bytes_per_rank: float = 0.0,
                       memory_budget_bytes: float | None = None,
                       rank_hbm_bytes: float | None = None,
                       usable_capacity_bytes: float | None = None,
                       memory_at_batch: Callable[[int], tuple[float, float]] | None = None,
                       rank_hbm_bytes_at: Callable[[int], float] | None = None) -> MtpReversal:
    """Every batch where a complete MTP cycle is no longer cheaper per token."""
    if not callable(step_ms):
        raise ModelingError("moe_skew_step_model_must_be_callable")
    draft = _int("draft_len", draft_len, minimum=0)
    limit = _int("max_batch", max_batch, minimum=1)
    rate = _number("accept_rate", accept_rate, 0.0)
    if rate > 1.0:
        raise ModelingError(f"moe_skew_accept_rate_invalid:{rate}")
    accepted = speculative_accept_tokens(
        rate, draft, bonus_token=bonus_token,
        accept_probabilities=accept_probabilities)
    if draft == 0:
        return MtpReversal(0, rate, 1.0, None, None, None, 1.0, (),
                           "draft_len_zero_has_no_speculative_step", (),
                           True, 0.0)
    draft_cost = _number("draft_cost_ms", draft_cost_ms)
    layer_cost = _number("draft_layer_cost_ms", draft_layer_cost_ms)
    rollback = _number("rollback_ms", rollback_ms)
    host = _number("host_ms", host_ms)
    rollback_bytes = _number("draft_rollback_bytes_per_rank",
                             draft_rollback_bytes_per_rank)
    qualified = False
    per_batch_memory: dict[int, tuple[float, float, bool]] = {}
    if memory_at_batch is not None or rank_hbm_bytes_at is not None:
        if memory_at_batch is not None and rank_hbm_bytes_at is not None:
            raise ModelingError("moe_skew_mtp_memory_evidence_duplicate")
        qualified = True
    elif rank_hbm_bytes is not None or usable_capacity_bytes is not None:
        if rank_hbm_bytes is None or usable_capacity_bytes is None:
            raise ModelingError("moe_skew_mtp_memory_evidence_incomplete")
        rank_bytes = _number("rank_hbm_bytes", rank_hbm_bytes)
        capacity = _number("usable_capacity_bytes", usable_capacity_bytes, 0.0,
                           strict=True)
        qualified = rank_bytes + rollback_bytes <= capacity
    elif memory_budget_bytes is not None:
        # An auxiliary-only budget can disqualify, never qualify a full rank.
        budget = _number("memory_budget_bytes", memory_budget_bytes, 0.0, strict=True)
        qualified = bool(rollback_bytes <= budget and rank_hbm_bytes is not None)
    if memory_at_batch is not None and not callable(memory_at_batch):
        raise ModelingError("moe_skew_mtp_memory_model_must_be_callable")
    if rank_hbm_bytes_at is not None and not callable(rank_hbm_bytes_at):
        raise ModelingError("moe_skew_mtp_memory_model_must_be_callable")
    grid = tuple(sorted({_int("batch_grid", value) for value in batch_grid})) \
        if batch_grid is not None else tuple(range(1, limit + 1))
    if grid and grid[-1] > limit:
        raise ModelingError(f"moe_skew_batch_grid_exceeds_max:{grid[-1]}")
    if rank_hbm_bytes_at is not None:
        # The scalar-requirement callback must still obey the caller's finite
        # full-rank capacity.  An absent/non-finite capacity can never qualify.
        finite_capacity = False
        capacity_value = 0.0
        if usable_capacity_bytes is not None:
            try:
                candidate_capacity = float(usable_capacity_bytes)
            except (TypeError, ValueError):
                candidate_capacity = float("nan")
            finite_capacity = (math.isfinite(candidate_capacity)
                               and candidate_capacity > 0.0)
            capacity_value = (candidate_capacity if finite_capacity else 0.0)
        for batch in grid:
            required = _number("rank_hbm_bytes", rank_hbm_bytes_at(int(batch)))
            ok = (finite_capacity
                  and required + rollback_bytes <= capacity_value)
            per_batch_memory[batch] = (required, capacity_value, ok)
        qualified = bool(finite_capacity and all(
            ok for _required, _capacity, ok in per_batch_memory.values()))
    cache: dict[int, float] = {}

    def cost(tokens: int) -> float:
        if tokens not in cache:
            value = step_ms(int(tokens))
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ModelingError(f"moe_skew_step_model_invalid:{tokens}:{value}")
            cache[tokens] = float(value)
        return cache[tokens]

    def base(batch: int) -> float:
        return cost(batch)

    def cycle(batch: int) -> float:
        if cycle_ms is not None:
            value = cycle_ms(int(batch), draft)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ModelingError(f"moe_skew_cycle_model_invalid:{batch}:{value}")
            return float(value)
        return (cost(batch * (1 + draft)) + draft_cost
                + draft * layer_cost + rollback + host)

    def tpot(batch: int) -> tuple[float, float]:
        return base(batch), cycle(batch) / accepted

    base_one, mtp_one = tpot(1)
    if memory_at_batch is not None:
        for batch in grid:
            pair = memory_at_batch(int(batch))
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ModelingError(f"moe_skew_mtp_memory_pair_invalid:{pair}")
            required = _number("rank_hbm_bytes", pair[0])
            capacity = _number("usable_capacity_bytes", pair[1], 0.0,
                               strict=True)
            ok = required + rollback_bytes <= capacity
            per_batch_memory[batch] = (required, capacity, ok)
        qualified = all(ok for _required, _capacity, ok in
                        per_batch_memory.values())
    memory_ok = {batch: value[2] for batch, value in per_batch_memory.items()}
    crossings = tuple(batch for batch in grid
                      if memory_ok.get(batch, True)
                      and cycle(batch) >= accepted * base(batch) - 1e-12)
    curve_points = set(grid) if len(grid) <= 256 else {
        batch for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, limit)
        if batch in grid}
    for batch in crossings:
        curve_points.update({batch, max(1, batch - 1), min(limit, batch + 1)})
    curve = tuple((batch, *tpot(batch)) for batch in sorted(curve_points) if batch <= limit)
    memory_rows = tuple((batch, required, capacity, ok) for batch,
                        (required, capacity, ok) in sorted(per_batch_memory.items()))
    if not crossings:
        return MtpReversal(draft, rate, accepted, None, None, None,
                           base_one / mtp_one if mtp_one > 0 else 0.0, curve,
                           "no_crossing_within_declared_grid", (), qualified,
                           rollback_bytes, memory_by_batch=memory_rows)
    first = crossings[0]
    base_c, mtp_c = tpot(first)
    first_curve = tuple((batch, *tpot(batch)) for batch in
                        sorted({first, max(1, first - 1), min(limit, first + 1)}))
    modeled = cycle(first)
    return MtpReversal(draft, rate, accepted, first, base_c, mtp_c,
                       base_one / mtp_one if mtp_one > 0 else 0.0, curve,
                       "mtp_cycle_not_below_autoregressive_tpot", crossings,
                       qualified, rollback_bytes, first_curve,
                       burst_itl_ms=modeled, amortized_useful_token_ms=mtp_c,
                       itl_basis="analytic_cycle_mean",
                       modeled_cycle_ms=modeled, memory_by_batch=memory_rows)



__all__ = ["MtpReversal", "mtp_batch_reversal", "speculative_accept_tokens",
           "speculative_step_pmf"]
