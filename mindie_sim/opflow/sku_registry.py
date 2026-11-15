"""Zero-desensitization 94-entry public Ascend SKU registry and legacy-frozen alias resolver.

[Role]
Load the unredacted 94-SKU public Ascend hardware registry and resolve canonical
public identifiers alongside legacy-frozen A2/A3/A5 calibration aliases.
[Invariants]
All 94 public SKUs remain unredacted; legacy A2/A3/A5 calibration keys carry
`legacy_frozen=True` and preserve `calibration_key_policy == "preserve_legacy"`;
ambiguous family tokens are rejected in strict mode instead of guessed.
[Timeouts & Locks]
Bounded local package JSON read cached in memory; no locks, waits, or device I/O.
[Side-Effect Policy]
Read-only module cache returning immutable records.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).with_name("public_sku_registry.json")


class SkuRegistryError(ValueError):
    """Raised when an unknown or ambiguous SKU token is queried in strict mode."""


@dataclass(frozen=True)
class PublicSkuEntry:
    """One unredacted public SKU entry from the 94-SKU hardware registry."""

    canonical_id: str
    identifier_kind: str
    public_identifiers: tuple[str, ...]
    a_family: str | None
    npu_arch: str | None
    publication_status: str
    spec_status: str
    facts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LegacyFrozenAlias:
    """Legacy A2/A3/A5 alias preserving calibrated runtime keys."""

    token: str
    runtime_key: str
    public_candidate: str | None
    identity_equivalence: str
    topology_preset: str | None
    calibration_key_policy: str
    legacy_frozen: bool


@dataclass(frozen=True)
class SkuResolution:
    """Resolved public SKU record with optional legacy-frozen calibration binding."""

    query: str
    canonical_id: str
    runtime_key: str
    legacy_frozen: bool
    calibration_key_policy: str
    topology_preset: str | None
    sku: PublicSkuEntry | None


def _norm_token(token: str) -> str:
    if not isinstance(token, str) or not token.strip():
        raise SkuRegistryError("sku_token_empty")
    return " ".join(token.strip().split()).lower()


@lru_cache(maxsize=1)
def _load_tables() -> tuple[
    str,
    tuple[PublicSkuEntry, ...],
    dict[str, PublicSkuEntry],
    dict[str, LegacyFrozenAlias],
    frozenset[str],
    tuple[str, ...],
]:
    raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    skus: list[PublicSkuEntry] = []
    by_token: dict[str, PublicSkuEntry] = {}
    for item in raw["skus"]:
        entry = PublicSkuEntry(
            canonical_id=str(item["canonical_id"]),
            identifier_kind=str(item["identifier_kind"]),
            public_identifiers=tuple(str(x) for x in item["public_identifiers"]),
            a_family=item.get("a_family"),
            npu_arch=item.get("npu_arch"),
            publication_status=str(item["publication_status"]),
            spec_status=str(item["spec_status"]),
            facts=tuple(dict(f) for f in item.get("facts", ())),
        )
        skus.append(entry)
        by_token[_norm_token(entry.canonical_id)] = entry
        for ident in entry.public_identifiers:
            by_token[_norm_token(ident)] = entry
    aliases: dict[str, LegacyFrozenAlias] = {}
    for item in raw["legacy_aliases"]:
        alias = LegacyFrozenAlias(
            token=str(item["token"]),
            runtime_key=str(item["runtime_key"]),
            public_candidate=item.get("public_candidate"),
            identity_equivalence=str(item["identity_equivalence"]),
            topology_preset=item.get("topology_preset"),
            calibration_key_policy=str(item["calibration_key_policy"]),
            legacy_frozen=bool(item.get("legacy_frozen", False)),
        )
        aliases[_norm_token(alias.token)] = alias
    ambiguous = frozenset(_norm_token(t) for t in raw["ambiguous_family_tokens"])
    forbidden = tuple(str(x) for x in raw["forbidden_automatic_inferences"])
    return str(raw["revision"]), tuple(skus), by_token, aliases, ambiguous, forbidden


def list_public_skus() -> tuple[PublicSkuEntry, ...]:
    """Return all 94 unredacted public Ascend SKU entries."""
    return _load_tables()[1]


def list_legacy_frozen_aliases() -> tuple[LegacyFrozenAlias, ...]:
    """Return all legacy-frozen calibration aliases (`a2`, `a3`, `a5`, etc.)."""
    return tuple(_load_tables()[3].values())


def ambiguous_sku_tokens() -> frozenset[str]:
    """Return normalized family tokens that require explicit disambiguation."""
    return _load_tables()[4]


def forbidden_sku_inferences() -> tuple[str, ...]:
    """Return forbidden automatic SKU inference rules."""
    return _load_tables()[5]


def resolve_public_sku(token: str, *, allow_legacy: bool = True) -> SkuResolution:
    """Resolve a public SKU identifier or a legacy-frozen A2/A3/A5 alias."""
    norm = _norm_token(token)
    _rev, _skus, by_token, aliases, ambiguous, _forbidden = _load_tables()
    if norm in ambiguous:
        raise SkuRegistryError(f"ambiguous_sku_family_token:{token}")
    if norm in by_token:
        sku = by_token[norm]
        return SkuResolution(
            query=token,
            canonical_id=sku.canonical_id,
            runtime_key=sku.canonical_id,
            legacy_frozen=False,
            calibration_key_policy="canonical_public",
            topology_preset=None,
            sku=sku,
        )
    alias_key = norm
    if alias_key not in aliases and alias_key.startswith(("atlas-", "atlas ")):
        alias_key = alias_key[6:].strip()
    if allow_legacy and alias_key in aliases:
        alias = aliases[alias_key]
        cand = _norm_token(alias.public_candidate) if alias.public_candidate else ""
        sku = by_token.get(cand) if cand else None
        canonical = sku.canonical_id if sku is not None else (alias.public_candidate or alias.runtime_key)
        return SkuResolution(
            query=token,
            canonical_id=canonical,
            runtime_key=alias.runtime_key,
            legacy_frozen=alias.legacy_frozen,
            calibration_key_policy=alias.calibration_key_policy,
            topology_preset=alias.topology_preset,
            sku=sku,
        )
    raise SkuRegistryError(f"unknown_sku_token:{token}")
