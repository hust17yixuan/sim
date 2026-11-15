"""Acceptance distributions, independent of execution cost.

[Role] Convert prefix marginals or conditional probabilities to accepted counts.
[Invariants] Commit = accepted prefix + one verified bonus token. No step-time
  constants or output-rate formulas participate in this declaration.
[Timeouts & Locks] O(proposal length), no I/O.
[Side-Effect Policy] Immutable records.
"""
from dataclasses import dataclass, fields
import math

from .common import ModelingError


@dataclass(frozen=True)
class SpeculativeSpec:
    draft_len: int
    acceptance: tuple[float, ...] = ()
    acceptance_mode: str = "marginal"
    accepted_distribution: tuple[float, ...] | None = None
    source: str = ""
    label: str = ""

    def __post_init__(self):
        if self.draft_len < 0 or self.acceptance_mode not in {"marginal", "conditional"}:
            raise ModelingError("speculative_proposal_or_acceptance_mode_invalid")
        if self.acceptance and self.accepted_distribution is not None:
            raise ModelingError("speculative_acceptance_ambiguous")
        if self.accepted_distribution is not None:
            values = self.accepted_distribution
            if len(values) != self.draft_len + 1 or abs(sum(values) - 1) > 1e-8:
                raise ModelingError("speculative_distribution_must_cover_all_accepted_counts")
        else:
            values = self.acceptance
            if len(values) != self.draft_len:
                raise ModelingError("speculative_acceptance_requires_each_proposal_position")
            if self.acceptance_mode == "marginal" and any(a < b for a, b in zip(values, values[1:])):
                raise ModelingError("prefix_acceptance_must_be_nonincreasing")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ModelingError("acceptance_probability_invalid")

    @classmethod
    def from_mapping(cls, raw):
        names = {f.name for f in fields(cls)}
        # ``num_speculative_tokens`` is the upstream vLLM spelling of the
        # declared draft length.  A repeated declaration must agree.
        aliases = {"num_speculative_tokens"}
        unknown = raw.keys() - names - aliases
        if unknown:
            raise ModelingError("speculative_unknown_fields:" + ",".join(sorted(unknown)))
        data = dict(raw)
        if "num_speculative_tokens" in data:
            tokens = data.pop("num_speculative_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise ModelingError("speculative_num_speculative_tokens_invalid")
            if "draft_len" in data and int(data["draft_len"]) != tokens:
                raise ModelingError(
                    f"speculative_draft_len_conflict:{data['draft_len']}!={tokens}")
            data["draft_len"] = tokens
        for key in ("acceptance", "accepted_distribution"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def acceptance_marginals(self):
        if self.accepted_distribution is not None:
            return tuple(math.fsum(self.accepted_distribution[index + 1:]) for index in range(self.draft_len))
        if self.acceptance_mode == "marginal":
            return self.acceptance
        result, survival = [], 1
        for probability in self.acceptance:
            survival *= probability
            result.append(survival)
        return tuple(result)

    def acceptance_distribution(self):
        if self.accepted_distribution is not None:
            return self.accepted_distribution
        survival = (1.0, *self.acceptance_marginals(), 0.0)
        return tuple(a - b for a, b in zip(survival, survival[1:]))

    def expected_committed_tokens(self):
        return 1 + math.fsum(self.acceptance_marginals())
