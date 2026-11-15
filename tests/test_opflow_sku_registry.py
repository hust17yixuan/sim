"""Regression tests for the 94-entry unredacted public SKU registry and legacy-frozen aliases.

[Role]
Verify that all 94 public Ascend SKUs resolve without desensitization while
legacy A2/A3/A5 calibration tokens remain preserved as legacy_frozen aliases.
[Invariants]
Exact 94 public SKUs, 23 legacy aliases with preserve_legacy policy, strict
rejection of bare ambiguous family tokens, and lazy `mindie_sim.opflow` access.
[Timeouts & Locks]
In-memory registry lookups only; no waits or locks.
[Side-Effect Policy]
Read-only assertions with zero filesystem writes.
"""

from __future__ import annotations

import pytest

import mindie_sim
from mindie_sim.opflow import (
    SkuRegistryError,
    ambiguous_sku_tokens,
    forbidden_sku_inferences,
    list_legacy_frozen_aliases,
    list_public_skus,
    resolve_public_sku,
)


def test_public_sku_registry_has_94_unredacted_entries() -> None:
    skus = list_public_skus()
    assert len(skus) == 94
    ids = {entry.canonical_id for entry in skus}
    assert len(ids) == 94
    assert {"Ascend310B1", "Ascend910B4", "Ascend950DT", "Ascend950PR"} <= ids
    assert all(entry.publication_status == "public_unredacted" for entry in skus)
    assert len(forbidden_sku_inferences()) == 8


def test_legacy_frozen_a2_a3_a5_aliases_preserved() -> None:
    aliases = list_legacy_frozen_aliases()
    assert len(aliases) == 23
    for token in ("a2", "800i-a2", "a3", "atlas-800i-a3", "a5", "atlas-950"):
        resolved = resolve_public_sku(token)
        assert resolved.legacy_frozen is True
        assert resolved.calibration_key_policy == "preserve_legacy"
    with pytest.raises(SkuRegistryError):
        resolve_public_sku("a2", allow_legacy=False)


def test_ambiguous_family_tokens_fail_closed() -> None:
    assert len(ambiguous_sku_tokens()) == 4
    for bad in ("Ascend910_93", "Ascend950", "Ascend310B", "Ascend310P"):
        with pytest.raises(SkuRegistryError, match="ambiguous_sku_family_token"):
            resolve_public_sku(bad)
    assert mindie_sim.opflow.resolve_public_sku("Ascend 910B4").canonical_id == "Ascend910B4"
