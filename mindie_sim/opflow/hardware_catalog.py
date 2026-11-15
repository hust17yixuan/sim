"""Canonical SKU catalog, factory, and public resolution API.

[Role]
Hold the explicit public/legacy SKU rows and build immutable HardwareProfile
instances with strict numeric overrides and identifier-only status handling.
[Invariants]
No SKU is inferred from a family-name suffix. Identifier-only rows are marked
and require caller-supplied core/clock values before prediction. Every override
is type-checked; bool/NaN/Inf/fractional values fail closed.
[Timeouts & Locks]
Pure in-memory constants and arithmetic; no locks, waits, I/O, or subprocesses.
[Side-Effect Policy]
Read-only module; constructing a profile has no filesystem or device effect.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from .hardware_profiles import (
    KIB, MIB, HardwareProfile, MemoryGeometry, ProfileError, RateModel,
    SpecStatus, UB_2201_SIMD_BYTES, UB_3510_SIMD_BYTES, _as_float, _as_int,
)


@dataclass(frozen=True, slots=True)
class _SkuRow:
    family: str
    sku: str
    aic: int
    freq: int
    hbm: float
    l2: int
    status: str
    basis: str
    arch_code: int = 3510
    l0c: int = 256 * KIB
    missing: tuple[str, ...] = ()
    ref_freq: int = 1650


def _row(family, sku, aic, freq, hbm, l2, status, basis, arch_code=3510,
         l0c=256 * KIB, missing=(), ref_freq=1650) -> _SkuRow:
    return _SkuRow(family, sku, aic, freq, hbm, l2, status, basis,
                   arch_code, l0c, missing, ref_freq)


_ROWS: tuple[_SkuRow, ...] = (
    _row("Ascend910B", "Ascend910B1", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.LEGACY.value, "legacy_sim_profiles", 2201, 128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910B2", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.LEGACY.value, "legacy_sim_profiles", 2201, 128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910B2C", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.LEGACY.value, "legacy_sim_profiles", 2201, 128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910B3", 20, 1800, 1.6e12, 192 * MIB,
         SpecStatus.LEGACY.value, "legacy_sim_profiles", 2201, 128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910B4", 20, 1800, 0.8e12, 128 * MIB,
         SpecStatus.LEGACY.value, "legacy_sim_profiles", 2201, 128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_9362", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "legacy_sim_profiles_24c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_9372", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "legacy_sim_profiles_24c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_9381", 20, 1800, 1.6e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "legacy_sim_profiles_20c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_9382", 20, 1800, 1.6e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "legacy_sim_profiles_20c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_9391", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "legacy_sim_profiles_24c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_9392", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "legacy_sim_profiles_24c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend910B", "Ascend910_93", 24, 1800, 1.8e12, 192 * MIB,
         SpecStatus.FAMILY_ALIAS_LEGACY.value, "family_alias_24c", 2201,
         128 * KIB, ref_freq=1800),
    _row("Ascend950PR", "Ascend950PR_957b", 28, 1650, 1.4e12, 112 * MIB,
         SpecStatus.COMPLETE_PUBLIC_SPEC.value, "pr_json_957b"),
    _row("Ascend950PR", "Ascend950PR_9589", 32, 1650, 1.6e12, 128 * MIB,
         SpecStatus.COMPLETE_PUBLIC_SPEC.value, "pr_json_9589"),
    _row("Ascend950PR", "Ascend950PR_9579", 28, 1650, 1.6e12, 128 * MIB,
         SpecStatus.IDENTIFIER_ONLY.value, "public_compiler_table_28c",
         missing=("core_count",)),
    _row("Ascend950DT", "Ascend950DT_28c", 28, 1650, 4.0e12, 128 * MIB,
         SpecStatus.WHITEPAPER_SPEC_VARIANT.value, "whitepaper_dt_28c"),
    _row("Ascend950DT", "Ascend950DT_32c", 32, 1650, 4.0e12, 128 * MIB,
         SpecStatus.WHITEPAPER_SPEC_VARIANT.value, "whitepaper_dt_32c"),
    _row("Ascend950DT", "Ascend950DT_36c", 36, 1650, 4.0e12, 128 * MIB,
         SpecStatus.WHITEPAPER_SPEC_VARIANT.value, "whitepaper_dt_36c"),
    _row("Ascend950DT", "Ascend950DT_9582", 32, 1650, 4.0e12, 128 * MIB,
         SpecStatus.IDENTIFIER_ONLY.value, "public_compiler_table_32c",
         missing=("core_count",)),
    _row("Ascend950DT", "Ascend950DT_95a", 32, 1650, 4.0e12, 128 * MIB,
         SpecStatus.IDENTIFIER_ONLY.value, "identifier_only",
         missing=("core_count", "frequency_mhz")),
)

_ALIASES = {
    "a2": "Ascend910B4", "a3": "Ascend910_9392", "910b": "Ascend910B2",
    "ascend910b": "Ascend910B2", "ascend910b2": "Ascend910B2",
    "910bv22024": "Ascend910B2", "v220": "Ascend910B2",
    "ascend950pr": "Ascend950PR_9589", "950pr": "Ascend950PR_9589",
    "ascend950dt": "Ascend950DT_32c", "950dt": "Ascend950DT_32c",
    "ascend91093": "Ascend910_93", "ascend9109362": "Ascend910_9362",
}


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_BY_KEY = {_norm_key(row.sku): row for row in _ROWS}
_ALIAS_BY_KEY = {_norm_key(key): value for key, value in _ALIASES.items()}
_OVERRIDE_FIELDS = {
    "aic_cores", "aiv_cores", "freq_mhz", "hbm_bw_gbps", "l2_bw_gbps",
    "l2_size_bytes", "ub_bytes_per_lane", "rw_contention_alpha", "cores",
}


def _rates_for(arch_code: int, freq: int, hbm: float, l2: float, alpha: float) -> RateModel:
    if arch_code == 2201:
        return RateModel(
            vector_ops_s=142 * freq * 1e6, trans_ops_s=36 * freq * 1e6,
            simt_ops_s=53 * freq * 1e6, cube_macs_s=2048.0 * freq * 1e6,
            fix_bytes_s=128.0 * freq * 1e6, mte1_bytes_s=256.0 * freq * 1e6,
            hbm_bytes_s=hbm, l2_bytes_s=l2, rw_contention_alpha=alpha,
            mte2_aic_bytes_s=48.0 * freq * 1e6, mte2_aiv_bytes_s=24.0 * freq * 1e6,
            mte3_bytes_s=80.0 * freq * 1e6,
            per_owner_port_bw=100e9 * (freq / 1800.0),
        )
    return RateModel(
        vector_ops_s=512 * freq * 1e6, trans_ops_s=128 * freq * 1e6,
        simt_ops_s=96 * freq * 1e6, cube_macs_s=4096.0 * freq * 1e6,
        fix_bytes_s=128.0 * freq * 1e6, mte1_bytes_s=256.0 * freq * 1e6,
        hbm_bytes_s=hbm, l2_bytes_s=l2, rw_contention_alpha=alpha,
        mte2_aic_bytes_s=80.0 * freq * 1e6, mte2_aiv_bytes_s=40.0 * freq * 1e6,
        mte3_bytes_s=100.0 * freq * 1e6, per_owner_port_bw=100e9,
    )


def _make_profile(row: _SkuRow, overrides: Mapping[str, Any]) -> HardwareProfile:
    aic = _as_int("aic_cores", overrides.get("aic_cores", row.aic))
    if aic <= 0:
        raise ProfileError("aic_cores must be positive")
    aiv = 2 * aic
    if "aiv_cores" in overrides and _as_int("aiv_cores", overrides["aiv_cores"]) != aiv:
        raise ProfileError("AIV must equal 2 x AIC for DAV_3510/V220 1C:2V")
    freq = _as_int("freq_mhz", overrides.get("freq_mhz", row.freq))
    if freq <= 0:
        raise ProfileError("freq_mhz must be positive")
    hbm = _as_float("hbm_bw_gbps", overrides.get("hbm_bw_gbps", row.hbm / 1e9)) * 1e9
    l2_size = _as_int("l2_size_bytes", overrides.get("l2_size_bytes", row.l2))
    l2_bw = _as_float("l2_bw_gbps", overrides.get("l2_bw_gbps", 0.0)) * 1e9
    if l2_size <= 0 or hbm <= 0:
        raise ProfileError("HBM bandwidth and L2 size must be positive")
    alpha = _as_float("rw_contention_alpha", overrides.get("rw_contention_alpha", 0.30))
    if not 0.0 <= alpha <= 1.0:
        raise ProfileError("rw_contention_alpha must be in [0, 1]")
    if "l2_bw_gbps" in overrides and l2_bw <= 0:
        raise ProfileError("explicit L2 bandwidth must be positive")
    if l2_bw <= 0:
        l2_bw = (aiv * 100e9) if row.arch_code == 3510 else 4.8e12
    ub = _as_int("ub_bytes_per_lane", overrides.get(
        "ub_bytes_per_lane",
        UB_2201_SIMD_BYTES if row.arch_code == 2201 else UB_3510_SIMD_BYTES,
    ))
    if ub <= 0:
        raise ProfileError("ub_bytes_per_lane must be positive")
    memory = MemoryGeometry(
        ub_per_lane=ub, l1=512 * KIB, l0a=64 * KIB, l0b=64 * KIB,
        l0c=row.l0c, l2=l2_size,
    )
    rates = _rates_for(row.arch_code, freq, hbm, l2_bw, alpha)
    missing = row.missing
    ready = row.status != SpecStatus.IDENTIFIER_ONLY.value
    if not ready:
        remaining = set(missing)
        if "core_count" in remaining and "aic_cores" in overrides:
            remaining.discard("core_count")
        if "frequency_mhz" in remaining and "freq_mhz" in overrides:
            remaining.discard("frequency_mhz")
        ready = not remaining
        missing = tuple(sorted(remaining))
    return HardwareProfile(
        family=row.family, sku=row.sku,
        arch="DAV_3510" if row.arch_code == 3510 else "V220",
        arch_code=row.arch_code, aic_cores=aic, aiv_cores=aiv, freq_mhz=freq,
        memory=memory, rates=rates,
        fixpipe_to_ub=row.arch_code == 3510 and ub >= UB_3510_SIMD_BYTES,
        spec_status=row.status, prediction_ready=ready,
        missing_parameters=missing, reference_freq_mhz=row.ref_freq,
        geometry_basis=row.basis, overridden_fields=frozenset(overrides),
    )


def _select_row(name: str, sku: str | None, overrides: Mapping[str, Any]) -> _SkuRow:
    token = sku if sku is not None else name
    key = _norm_key(token)
    if key in _BY_KEY:
        return _BY_KEY[key]
    if _norm_key(name) == "ascend950dt" and "cores" in overrides:
        cores = _as_int("cores", overrides["cores"])
        if cores not in (28, 32, 36):
            raise ProfileError("Ascend950DT cores must be 28, 32, or 36")
        return _BY_KEY[_norm_key(f"Ascend950DT_{cores}c")]
    if key in _ALIAS_BY_KEY:
        return _BY_KEY[_norm_key(_ALIAS_BY_KEY[key])]
    raise ProfileError(
        f"unknown hardware SKU or family: name={name!r} sku={sku!r}. "
        "Use a canonical public identifier such as 'Ascend950PR_9589'."
    )


def get_profile(name: str, sku: str | None = None, *, strict: bool = False,
                **overrides: Any) -> HardwareProfile:
    """Resolve one concrete profile from a canonical or normalized identifier."""
    if not isinstance(strict, bool):
        raise ProfileError("strict must be bool")
    unknown = set(overrides) - _OVERRIDE_FIELDS
    if unknown:
        raise ProfileError(f"unsupported profile overrides: {sorted(unknown)}")
    if "cores" in overrides and "aic_cores" in overrides:
        raise ProfileError("pass either cores or aic_cores, not both")
    if "cores" in overrides and (sku is not None or _norm_key(name) != "ascend950dt"):
        raise ProfileError("cores is valid only for the Ascend950DT family")
    row = _select_row(name, sku, overrides)
    profile = _make_profile(row, overrides)
    profile.require_prediction_ready() if strict else None
    return profile


def resolve_profile(name: str, sku: str | None = None, **overrides: Any) -> HardwareProfile:
    """Strict resolver used by prediction entrypoints."""
    return get_profile(name, sku, strict=True, **overrides)


def catalog() -> tuple[HardwareProfile, ...]:
    return tuple(get_profile(row.sku) for row in _ROWS)


def list_profiles() -> list[dict[str, Any]]:
    return [entry.to_dict() for entry in catalog()]


def find_profile(sku: str) -> HardwareProfile:
    key = _norm_key(sku)
    if key in _BY_KEY:
        return get_profile(_BY_KEY[key].sku)
    if key in _ALIAS_BY_KEY:
        return get_profile(_ALIAS_BY_KEY[key])
    raise ProfileError(f"unknown SKU: {sku!r}")
