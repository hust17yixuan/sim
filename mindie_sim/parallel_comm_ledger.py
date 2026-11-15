"""Operation-derived communication ledger shared by execution and admission.

[Role]
  Record every collective and peer transfer a prefill or decode step actually
  executes: role, collective kind, per-call payload, physical rank groups and
  the resolved switch/tier cut bytes that operation places on each shared
  resource.  Pool admission consumes this same record instead of reconstructing
  a generic axis model.

[Invariants]
  * One operation names one executed call site.  ``count`` repeats an identical
    call (for example one collective per layer); synchronous groups are kept in
    one operation and their wire bytes aggregate on every traversed cut.
  * ``dedupe`` is used only where the execution path issues the same call once
    per synchronous step, never to hide a sequential dispatch/combine pair.
  * Cut bytes are resolved at emission through the same fabric tree that prices
    the step, so a later tree cannot silently re-price a recorded operation.
  * Every operation also records the per-endpoint ingress/egress bytes each
    participant actually moves; admission consumes those distinct port
    identities, never an unweighted pool NIC aggregate.
  * ``cut_demands`` is linear in payload and count: the ledger conserves byte
    sums by resource identity and never divides by a logical rank count.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh immutable records; caller trees untouched.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.parallel_cuts import (collective_cut_demands,
                                      collective_port_demands, merge_demands,
                                      peer_cut_demands, peer_port_demands)


def _groups(values: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(rank) for rank in group) for group in values)


def reduce_route_evidence(*rows: tuple[bool, str]) -> tuple[bool, str]:
    """Join the evidence lattice: trivial < declared < measured < failure.

    Any false qualification dominates every positive observation.  Among
    failures, disconnected > unmeasured > other failures; a lexical tie-break
    makes the join associative, commutative and independent of execution order.
    Known negative evidence cannot become positive through a caller's flag.
    """
    def key(row: tuple[bool, str]) -> tuple[int, int, str]:
        ok, evidence = row
        negative = ("unmeasured" in evidence or "disconnected" in evidence)
        if not ok or negative:
            return (1, 2 if "disconnected" in evidence else
                    1 if "unmeasured" in evidence else 0, evidence)
        positive = (0 if evidence in {"trivial", "single_rank", "single_stage"}
                    else 2 if evidence == "measured_profile" else 1)
        return (0, positive, evidence)

    worst = max(rows or ((True, "declared"),), key=key)
    return key(worst)[0] == 0, str(worst[1])


@dataclass(frozen=True)
class CommOperation:
    """One executed communication call site on its physical rank groups."""

    role: str
    kind: str
    payload_bytes: float
    groups: tuple[tuple[int, ...], ...] = ()
    count: int = 1
    cut_demands: tuple[tuple[str, float], ...] = ()
    port_demands: tuple[tuple[str, float], ...] = ()

    @property
    def total_payload_bytes(self) -> float:
        return max(0.0, float(self.payload_bytes)) * max(0, int(self.count))

    def demands(self) -> dict[str, float]:
        return {str(key): float(value) * max(0, int(self.count))
                for key, value in self.cut_demands}

    def ports(self) -> dict[str, float]:
        return {str(key): float(value) * max(0, int(self.count))
                for key, value in self.port_demands}

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "kind": self.kind,
                "payload_bytes": float(self.payload_bytes),
                "count": int(self.count),
                "groups": [list(group) for group in self.groups],
                "cut_demands": dict(self.cut_demands),
                "port_demands": dict(self.port_demands)}


@dataclass(frozen=True)
class CommLedger:
    """Complete executed communication record for one phase step."""

    phase: str
    operations: tuple[CommOperation, ...] = ()
    tokens: float = 0.0
    requests: float = 0.0

    def cut_demands(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for operation in self.operations:
            result = merge_demands(result, operation.demands())
        return result

    def port_demands(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for operation in self.operations:
            result = merge_demands(result, operation.ports())
        return result

    def role_payloads(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for operation in self.operations:
            result[operation.role] = (result.get(operation.role, 0.0)
                                     + operation.total_payload_bytes)
        return result

    def merged(self, other: "CommLedger") -> "CommLedger":
        if not isinstance(other, CommLedger):
            raise TypeError("parallel_comm_ledger_merge_type")
        return CommLedger(self.phase,
                          tuple(self.operations) + tuple(other.operations),
                          float(self.tokens) + float(other.tokens),
                          float(self.requests) + float(other.requests))

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase,
                "operations": [item.to_dict() for item in self.operations],
                "tokens": float(self.tokens), "requests": float(self.requests),
                "cut_demands": self.cut_demands(),
                "port_demands": self.port_demands(),
                "role_payloads": self.role_payloads()}


class CommLedgerBuilder:
    """Emit operations while pricing a step, resolving cut bytes immediately."""

    def __init__(self, tree: Any, phase: str):
        self.tree = tree
        self.phase = str(phase)
        self.operations: list[CommOperation] = []
        self._seen: set[tuple[Any, str, float]] = set()

    def collective(self, role: str, kind: str, payload: float,
                   groups: Sequence[Sequence[int]], *, count: int = 1,
                   dedupe: bool = False) -> None:
        value = max(0.0, float(payload))
        repeat = max(0, int(count))
        ids = tuple(group for group in _groups(groups) if len(group) > 1)
        if value <= 0.0 or repeat <= 0 or not ids:
            return
        key = (ids, str(kind), value)
        if dedupe:
            if key in self._seen:
                return
            self._seen.add(key)
        demands = tuple(sorted(collective_cut_demands(
            self.tree, ids, str(kind), value).items()))
        ports = tuple(sorted(collective_port_demands(
            self.tree, ids, str(kind), value).items()))
        self.operations.append(CommOperation(
            str(role), str(kind), value, ids, repeat, demands, ports))

    def peer(self, role: str, payload: float,
             endpoints: Sequence[int], *, count: int = 1) -> None:
        value = max(0.0, float(payload))
        repeat = max(0, int(count))
        ids = tuple(int(rank) for rank in endpoints)
        if value <= 0.0 or repeat <= 0 or len(ids) < 2:
            return
        demands = tuple(sorted(peer_cut_demands(
            self.tree, ids, value).items()))
        ports = tuple(sorted(peer_port_demands(
            self.tree, ids, value).items()))
        self.operations.append(CommOperation(
            str(role), "peer", value, (ids,), repeat, demands, ports))

    def merge(self, ledger: CommLedger | None) -> None:
        if ledger is not None:
            self.operations.extend(ledger.operations)

    def build(self, *, tokens: float = 0.0,
              requests: float = 0.0) -> CommLedger:
        return CommLedger(self.phase, tuple(self.operations),
                          max(0.0, float(tokens)), max(0.0, float(requests)))


def empty_ledger(phase: str) -> CommLedger:
    """Typed empty ledger for a genuine zero-work step."""
    return CommLedger(str(phase))


def merge_ledgers(phase: str, ledgers: Sequence[CommLedger]) -> CommLedger:
    """Concatenate per-microbatch ledgers into one step record."""
    result = CommLedger(str(phase))
    for ledger in ledgers:
        result = result.merged(ledger)
    return result


__all__ = ["CommLedger", "CommLedgerBuilder", "CommOperation", "empty_ledger",
           "merge_ledgers", "reduce_route_evidence"]
