"""Public API for the universal Ascend operator-flow engine.

[Role]
Re-export hardware, schema, transfer, tiling, recurrence, pipeline, Triton
lowering, network-group catalog, and bounded MCP APIs.
[Invariants]
Imports are side-effect free and stay within the standard library. No module
import touches a device, the network, or the filesystem.
[Timeouts & Locks]
Import-only; no locks, waits, or timeout paths.
[Side-Effect Policy]
Read-only import surface.
"""

from __future__ import annotations

from .flow_schema import (
    AuthoringBackend, BufferPool, BufferSpec, ExecutionArchetype, ExecutionMode,
    FixpipeConfig, FixpipeError, FixpipePath, FixpipeSelection, FlowSchemaError,
    LoopCarriedRecurrence, OpFlowTemplate, PipeWork, buffer_capacity,
    check_flow_fixpipe, select_fixpipe,
)
from .sku_registry import (
    LegacyFrozenAlias, PublicSkuEntry, SkuRegistryError, SkuResolution,
    ambiguous_sku_tokens, forbidden_sku_inferences,
    list_legacy_frozen_aliases, list_public_skus, resolve_public_sku,
)
from .hardware_profiles import (
    HardwareProfile, MemCopyEdge, MemCopyError, MemoryGeometry, PeakTFLOPS,
    ProfileError, RateModel, SpecStatus, catalog as hardware_catalog,
    check_mem_copy_edge, find_profile, get_profile, is_legal_mem_copy,
    legal_mem_copy_edges, list_profiles, mem_copy_edges, resolve_profile,
    supports_mem_copy, validate_mem_copy,
)
from .network_groups import (
    DS41_OPERATORS, K3_OPERATORS, NETWORK_GROUPS, build_network_flow,
    list_network_operator_group, network_membership,
)
from .paradigm_catalog import (
    CATALOG, STRUCTURAL_ONLY, UncalibratedOperatorError, build_flow,
    catalog_info, evaluate_operator, list_operators,
)
from .pipeline_evaluator import (
    BubbleLedger, ExecutionModeError, FlowEvaluation, LAUNCH_FLOORS,
    LAUNCH_FLOORS_US, TilingError, TilingPlan, derive_tiling,
    effective_gm_bandwidth, evaluate_flow, launch_floor_ordering,
    mode_launch_floor_us,
)
from .transfer_accounting import (
    BufferPlan, GmStrides, ReadPolicy, TransferAccounting, TransferError,
    TransferKind, TransferRequest, account_transfer, align_up,
    check_pool_capacity, plan_buffer_pool,
)
from .triton_lowering import (
    MFUReport, TritonFeatures, TritonLoweringError, TritonLoweringResult,
    analyze_triton_lowering, classify_cache_regime, evaluate_mfu_gates,
)
from .triton_rules import (
    GATHER_PATHS, MFU_GATES, TRITON_RULES, TritonRule, rule_by_number,
    rules_by_tag,
)

__all__ = [
    "AuthoringBackend", "BufferPlan", "BufferPool", "BufferSpec", "BubbleLedger",
    "CATALOG", "DS41_OPERATORS", "ExecutionArchetype", "ExecutionMode",
    "ExecutionModeError", "FixpipeConfig", "FixpipeError", "FixpipePath",
    "FixpipeSelection", "FlowEvaluation", "FlowSchemaError", "GATHER_PATHS",
    "GmStrides", "HardwareProfile", "K3_OPERATORS", "LAUNCH_FLOORS",
    "LAUNCH_FLOORS_US", "LoopCarriedRecurrence", "MFU_GATES", "MFUReport",
    "MemCopyEdge", "MemCopyError", "MemoryGeometry", "NETWORK_GROUPS",
    "OpFlowTemplate", "PeakTFLOPS", "PipeWork", "ProfileError", "RateModel",
    "ReadPolicy", "STRUCTURAL_ONLY", "SpecStatus", "TRITON_RULES", "TilingError",
    "TilingPlan", "UncalibratedOperatorError",
    "TransferAccounting", "TransferError", "TransferKind", "TransferRequest",
    "TritonFeatures", "TritonLoweringError", "TritonLoweringResult", "TritonRule",
    "account_transfer", "align_up", "analyze_triton_lowering", "buffer_capacity",
    "build_flow", "build_network_flow", "catalog_info", "check_flow_fixpipe",
    "check_pool_capacity", "check_mem_copy_edge", "classify_cache_regime", "derive_tiling",
    "effective_gm_bandwidth", "evaluate_flow", "evaluate_mfu_gates",
    "evaluate_operator", "find_profile", "get_profile", "hardware_catalog",
    "launch_floor_ordering", "legal_mem_copy_edges", "list_network_operator_group",
    "list_operators", "list_profiles", "mem_copy_edges", "mode_launch_floor_us",
    "network_membership", "plan_buffer_pool", "resolve_profile", "rule_by_number",
    "rules_by_tag",
    "LegacyFrozenAlias", "PublicSkuEntry", "SkuRegistryError", "SkuResolution",
    "ambiguous_sku_tokens", "forbidden_sku_inferences",
    "list_legacy_frozen_aliases", "list_public_skus", "resolve_public_sku",
    "select_fixpipe", "supports_mem_copy", "is_legal_mem_copy", "validate_mem_copy",
]
