"""Invocation timing and execution-occurrence identity declarations.

[Role] Parse declared startup/completion constraints and bind an occurrence to
  its real invocation/batch/request/rank scope.
[Invariants] Nonnegative milliseconds; provenance is retained, never a proof
  of universality. Empty declarations introduce no inferred device overhead.
  An occurrence is an identity, not a duration and not a semantic owner.
[Timeouts & Locks] Pure finite metadata; no I/O or locks.
[Side-Effect Policy] Copies caller-owned provenance.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from numbers import Real
from typing import Any


#: Occurrences name what actually executes.  Model/subgraph membership lives
#: in nullable semantic references and never replaces this identity.
OCCURRENCE_KINDS = frozenset({
    "call", "kernel", "phase", "stage", "task", "replay", "request", "batch",
})


@dataclass(frozen=True)
class Occurrence:
    """The real execution identity of one call/kernel/phase/stage.

    ``serial``/``phase``/``generation`` distinguish repeated executions.
    ``request_ids``/``batch_ids``/``rank``/``stream`` bind the occurrence to
    the scope it actually serves.  An empty scope is valid for a standalone
    operator/subgraph fixture.
    """

    kind: str
    name: str
    invocation: str = ""
    phase: str = ""
    serial: int = 0
    request_ids: tuple[str, ...] = ()
    batch_ids: tuple[str, ...] = ()
    rank: int | None = None
    stream: str = ""
    generation: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in OCCURRENCE_KINDS:
            raise ValueError(f"occurrence_kind_unknown:{self.kind}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("occurrence_name_invalid")
        for name in ("invocation", "phase", "stream"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"occurrence_{name}_requires_string")
        if not isinstance(self.serial, int) or isinstance(self.serial, bool) \
                or self.serial < 0:
            raise ValueError("occurrence_serial_requires_nonnegative_int")
        if isinstance(self.request_ids, str) or isinstance(self.batch_ids, str):
            raise TypeError("occurrence_scope_ids_require_sequence")
        object.__setattr__(self, "request_ids", tuple(self.request_ids))
        object.__setattr__(self, "batch_ids", tuple(self.batch_ids))
        for name in ("request_ids", "batch_ids"):
            values = getattr(self, name)
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError(f"occurrence_{name}_invalid")
            if len(set(values)) != len(values):
                raise ValueError(f"occurrence_{name}_duplicate")
        if self.rank is not None and (
                not isinstance(self.rank, int) or isinstance(self.rank, bool)
                or self.rank < 0):
            raise ValueError("occurrence_rank_requires_nonnegative_int")
        if self.generation is not None and (
                not isinstance(self.generation, int) or isinstance(self.generation, bool)
                or self.generation < 0):
            raise ValueError("occurrence_generation_requires_nonnegative_int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name,
            "invocation": self.invocation, "phase": self.phase,
            "serial": self.serial, "request_ids": list(self.request_ids),
            "batch_ids": list(self.batch_ids), "rank": self.rank,
            "stream": self.stream, "generation": self.generation,
        }

    @property
    def key(self) -> tuple:
        return (self.kind, self.name, self.invocation, self.phase, self.serial,
                self.request_ids, self.batch_ids, self.rank, self.stream,
                self.generation)


class _FrozenProvenance(dict):
    """Immutable-by-convention provenance dict that caches its canonical JSON signature."""
    __slots__ = ("_sig",)


def _declared_ms(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invocation_{name}_requires_finite_nonnegative_number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invocation_{name}_requires_finite_nonnegative_number")
    return value


@dataclass(frozen=True)
class FloorConstraint:
    """One independently provenanced startup/completion floor contribution.

    ``_FloorJoinProgram``/`InvocationProgram` keep every merged source separate
    so physical admission can accept one constraint while the Engine still
    honors the maximum over all of them.
    """

    startup_ms: float = 0.0
    minimum_ms: float = 0.0
    provenance: Mapping = field(default_factory=dict)
    source_kind: str = "invocation"

    def __post_init__(self):
        object.__setattr__(self, "startup_ms", _declared_ms("startup_ms", self.startup_ms))
        object.__setattr__(self, "minimum_ms", _declared_ms("minimum_ms", self.minimum_ms))
        if not isinstance(self.provenance, Mapping):
            raise TypeError("floor_constraint_provenance_requires_mapping")
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise ValueError("floor_constraint_requires_source_kind")
        if isinstance(self.provenance, _FrozenProvenance):
            object.__setattr__(self, "provenance", self.provenance)
        else:
            frozen = _FrozenProvenance(deepcopy(dict(self.provenance)))
            frozen._sig = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
            object.__setattr__(self, "provenance", frozen)

    @property
    def _cache_sig(self) -> tuple[float, float, str, str]:
        return (self.startup_ms, self.minimum_ms,
                getattr(self.provenance, "_sig", ""), self.source_kind)

    def to_dict(self) -> dict:
        return {
            "startup_ms": self.startup_ms,
            "minimum_ms": self.minimum_ms,
            "provenance": deepcopy(dict(self.provenance)),
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True)
class InvocationTiming:
    startup_ms: float = 0.0
    minimum_ms: float = 0.0
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "startup_ms", _declared_ms("startup_ms", self.startup_ms))
        object.__setattr__(self, "minimum_ms", _declared_ms("minimum_ms", self.minimum_ms))
        if not isinstance(self.provenance, Mapping):
            raise TypeError("invocation_provenance_requires_mapping")
        if isinstance(self.provenance, _FrozenProvenance):
            object.__setattr__(self, "provenance", self.provenance)
        else:
            fp = _FrozenProvenance(deepcopy(dict(self.provenance)))
            fp._sig = json.dumps(fp, sort_keys=True, separators=(",", ":"))
            object.__setattr__(self, "provenance", fp)

    @property
    def _cache_sig(self) -> tuple[float, float, str]:
        return (self.startup_ms, self.minimum_ms, getattr(self.provenance, "_sig", ""))

    @classmethod
    def from_mapping(cls, data: Mapping) -> "InvocationTiming":
        if not isinstance(data, Mapping):
            raise TypeError("invocation_timing_requires_mapping")
        unknown = data.keys() - {"startup_ms", "minimum_ms", "provenance"}
        if unknown:
            raise ValueError("invocation_timing_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "startup_ms": self.startup_ms,
            "minimum_ms": self.minimum_ms,
            "provenance": deepcopy(dict(self.provenance)),
        }


__all__ = ["FloorConstraint", "InvocationTiming", "OCCURRENCE_KINDS", "Occurrence"]


