"""CANNBot-DSL K3 and DS41 network-group membership.

[Role]
Mirror `cannbot-dsl/net/k3/network.json` and `net/ds41/network.json` exactly
and tag every catalog flow with `k3`/`ds41`/`common` membership.
[Invariants]
The K3 and DS41 tuples are the ten shipped names, in order. Membership is
symmetric and total. Generic construction delegates to the one catalog builder
registry so there is no second placeholder formula path.
[Timeouts & Locks]
Pure constants and arithmetic; no I/O, locks, or waits.
[Side-Effect Policy]
Read-only; no filesystem or device effect.
"""

from __future__ import annotations

from typing import Any

from .flow_schema import OpFlowTemplate

K3_OPERATORS: tuple[str, ...] = (
    "block_attn_res_update_rms_norm",
    "flash_kda",
    "flash_kda_fwd",
    "flash_kda_cp_preprocess",
    "flash_kda_cp_main",
    "fused_recurrent_kda_snapshot",
    "fused_recurrent_kda_replayssm",
    "mega_recurrent_kda",
    "commit_recurrent_kda",
    "quant_mla_prolog",
)

DS41_OPERATORS: tuple[str, ...] = (
    "mixed_quant_sparse_flash_mla",
    "quant_lightning_indexer",
    "quant_sparse_lightning_indexer",
    "indexer_prologue_qw",
    "indexer_prologue_k",
    "engram_gate",
    "attn_epilogue",
    "mega_mla",
    "mega_full_index",
    "mega_compressor",
)

NETWORK_GROUPS: dict[str, tuple[str, ...]] = {"k3": K3_OPERATORS, "ds41": DS41_OPERATORS}


def network_membership(name: str) -> tuple[str, ...]:
    groups = [group for group, names in NETWORK_GROUPS.items() if name in names]
    return tuple(groups) if groups else ("common",)


def list_network_operator_group(network: str) -> tuple[str, ...]:
    key = str(network).strip().lower()
    if key not in NETWORK_GROUPS:
        raise KeyError(f"unknown network group: {network!r}")
    return NETWORK_GROUPS[key]


def is_network_operator(name: str) -> bool:
    return network_membership(name) != ("common",)


def build_network_flow(name: str, **kw: Any) -> OpFlowTemplate:
    """Build a K3/DS41 operator through the single catalog builder registry."""
    if not is_network_operator(name):
        raise KeyError(f"not a K3/DS41 network operator: {name!r}")
    from .paradigm_catalog import build_flow

    return build_flow(name, **kw)


__all__ = [
    "DS41_OPERATORS", "K3_OPERATORS", "NETWORK_GROUPS", "build_network_flow",
    "is_network_operator", "list_network_operator_group", "network_membership",
]
