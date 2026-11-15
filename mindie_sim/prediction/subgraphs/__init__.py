"""Canonical subgraph vocabulary, instantiation and state contracts.

[Role] Expose the typed template registry and deterministic instantiation API.
[Invariants] Importing performs no model build, device probe or measurement.
[Timeouts & Locks] No import-time waits, I/O or locks.
[Side-Effect Policy] Registers pure builders only; callers own caches/state.
"""
from .instantiate import (
    FeatureConfig, InstantiatedModel, LayerInstances, LayerSlot,
    bind_occurrence, build_standalone, compile_standalone, evaluate_standalone,
    evaluate_standalone_graph, instantiate, instantiate_layer,
    instantiate_model, occurrence_contract, standalone_span,
    subgraph_dispositions,
)
from .registry import (
    COMBINATION_RULES, KIND_BUILDERS, VOCABULARY, TemplateRegistry, kind_spec,
    validate_combination, vocabulary,
)
from .state import (
    AcceptanceState, CompressionState, MoeRoutingState, ResidualSlotState,
    SharedKvContract, StateTransition, StateView,
)

__all__ = [
    "AcceptanceState", "COMBINATION_RULES", "CompressionState", "FeatureConfig",
    "InstantiatedModel", "KIND_BUILDERS", "LayerInstances", "LayerSlot",
    "MoeRoutingState", "ResidualSlotState", "SharedKvContract",
    "StateTransition", "StateView", "TemplateRegistry", "VOCABULARY",
    "bind_occurrence", "build_standalone", "compile_standalone", "evaluate_standalone",
    "evaluate_standalone_graph",
    "instantiate", "instantiate_layer", "instantiate_model", "kind_spec",
    "occurrence_contract", "standalone_span",
    "subgraph_dispositions", "validate_combination", "vocabulary",
]
