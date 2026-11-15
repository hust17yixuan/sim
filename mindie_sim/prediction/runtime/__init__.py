"""Unique causal runtime: frozen contracts, one state root, one Engine.

[Role] Public import surface for every T04 producer/consumer.
[Invariants] Engine/Event are resolved lazily from the single runtime.engine
  implementation; no second engine or compatibility shim is exported.
[Timeouts & Locks] None.
[Side-Effect Policy] Import only.
"""
from __future__ import annotations

from .contracts import (
    AllocationBinding, AllocationKey, BackingKey, BindingFrame, CapacityDomainState,
    CapTerm, CatalogGeneration, EdgeSpec, EventKey, FiniteProgramSpec, GrantResult,
    HostProgramSpec, OccurrenceKey, PlacementSpec, PoolKey, PortBinding, PortSpec,
    ProgramCatalog, ReaderKey, ReleaseResult, ReleaseSpec, Submission,
    TranslationCertificate, VersionBinding, WorkTerm,
)

_ENGINE_NAMES = frozenset({
    "Allocation", "Engine", "Event", "ExecutionError", "Resource",
    "PreparedStage",
})


def __getattr__(name):
    if name in _ENGINE_NAMES:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(name)


__all__ = [
    "AllocationBinding", "AllocationKey", "BackingKey", "BindingFrame",
    "CapacityDomainState", "CapTerm", "CatalogGeneration", "EdgeSpec",
    "EventKey", "FiniteProgramSpec", "GrantResult", "HostProgramSpec",
    "OccurrenceKey", "PlacementSpec", "PoolKey", "PortBinding", "PortSpec",
    "ProgramCatalog", "ReaderKey", "ReleaseResult", "ReleaseSpec",
    "Submission", "TranslationCertificate", "VersionBinding", "WorkTerm",
    "Allocation", "Engine", "Event", "ExecutionError", "Resource",
    "PreparedStage",
]
