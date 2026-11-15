"""Canonical Ascend hardware profiles with explicit SKU provenance.

[Role]
Resolve public canonical SKU identifiers (case-preserving and normalized) to
immutable DAV_2201 / DAV_3510 geometry, dtype-specific Cube/Vector rates, and
whitepaper-rounding tables kept separate from frequency-derived nominal peaks.
[Invariants]
DAV_2201 L0C is 131072 B; 192 KiB is UB only. Every 3510 profile keeps L2 at
AIV_count * 100 GB/s. Frequencies scale every pipe rate and every derived peak
by the same ratio. Identifier-only SKUs are never presented as complete; they
require explicit core/clock overrides before prediction. All numeric inputs
reject bool, NaN, Inf, and non-integral values.
[Timeouts & Locks]
Pure in-memory computation; no locks, I/O, subprocesses, or device access.
[Side-Effect Policy]
Read-only module. Constructing a profile has no filesystem or device effect.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

GIB = 1024**3
MIB = 1024**2
KIB = 1024

UB_3510_SIMD_BYTES = 253952
UB_3510_SIMT_BYTES = 221184
UB_3510_SIMT_CARVE_BYTES = 32768
UB_2201_SIMD_BYTES = 196608
SSBUF_3510_MIX_BYTES = 3072
SSBUF_3510_PURE_BYTES = 1024


class ProfileError(ValueError):
    """Raised when a profile cannot be selected or is incomplete."""


class MemCopyError(ProfileError):
    """A mem_copy direction is illegal on the selected NPU architecture."""


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be an integer, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ProfileError(f"{name} must be a finite integer, got {value!r}")
    return int(number)


def _as_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ProfileError(f"{name} must be finite, got {value!r}")
    return number


class SpecStatus(str, Enum):
    COMPLETE_PUBLIC_SPEC = "complete_public_spec"
    WHITEPAPER_SPEC_VARIANT = "whitepaper_spec_variant"
    IDENTIFIER_ONLY = "identifier_only_requires_overrides"
    FAMILY_ALIAS_LEGACY = "family_alias_legacy_compatibility"
    LEGACY = "legacy_compatibility"


class MemCopyEdge(str, Enum):
    GM_TO_UB = "GM->UB"
    UB_TO_GM = "UB->GM"
    GM_TO_L1 = "GM->L1"
    L1_TO_GM = "L1->GM"
    L1_TO_L0A = "L1->L0A"
    L1_TO_L0B = "L1->L0B"
    GM_TO_L0A = "GM->L0A"
    GM_TO_L0B = "GM->L0B"
    L0C_TO_GM = "L0C->GM"
    L0C_TO_L1 = "L0C->L1"
    L0C_TO_UB = "L0C->UB"
    UB_TO_L1 = "UB->L1"
    L1_TO_UB = "L1->UB"
    L1_TO_BIAS = "L1->BIAS"


@dataclass(frozen=True, slots=True)
class MemoryGeometry:
    """Per-core software-visible SRAM capacities plus shared L2."""

    ub_per_lane: int
    l1: int
    l0a: int
    l0b: int
    l0c: int
    l2: int
    ub_align: int = 32
    l1_align: int = 512
    l0c_align: int = 1024

    def capacity(self, pool: str) -> int:
        key = str(pool).upper()
        if key in ("UB_LANE0", "UB_LANE1", "UB"):
            return self.ub_per_lane
        if key == "L1":
            return self.l1
        if key == "L0A":
            return self.l0a
        if key == "L0B":
            return self.l0b
        if key == "L0C":
            return self.l0c
        if key == "BIAS":
            return 4096
        raise KeyError(f"unknown SRAM pool: {pool}")


@dataclass(frozen=True, slots=True)
class PeakTFLOPS:
    total: float
    cube: float
    vector: float

    @property
    def cube_gap_pct(self) -> float:
        return 100.0 * (self.total - self.cube) / self.cube if self.cube else 0.0


# Cube MACs/cycle and Vector element-ops/cycle per AIC/AIV lane.
_CUBE_MACS: Mapping[str, float] = {
    "bf16": 4096, "fp16": 4096, "fp32": 0, "int8": 8192,
    "fp8": 8192, "fp8_e4m3": 8192, "fp8_e5m2": 8192, "hif8": 8192,
    "mxfp8": 8192, "mxfp4": 16384, "tf32": 2048, "hf32": 2048,
}
_VEC_3510: Mapping[str, float] = {
    "bf16": 512, "fp16": 512, "fp32": 256, "int8": 512, "int16": 256,
    "int32": 128, "int64": 64, "fp8": 512, "fp8_e4m3": 512,
    "fp8_e5m2": 512, "hif8": 512, "mxfp8": 512, "mxfp4": 512,
    "tf32": 256, "hf32": 256,
}
_VEC_2201: Mapping[str, float] = {
    "bf16": 142, "fp16": 142, "fp32": 71, "int8": 142, "int16": 71,
    "int32": 35, "int64": 17, "fp8": 142, "fp8_e4m3": 142,
    "fp8_e5m2": 142, "hif8": 142, "mxfp8": 142, "mxfp4": 142,
    "tf32": 71, "hf32": 71,
}
_DTYPE_ALIASES = {
    "BFLOAT16": "bf16", "BF16": "bf16", "FLOAT16": "fp16", "HALF": "fp16",
    "FLOAT32": "fp32", "FLOAT": "fp32", "FP8E4M3": "fp8_e4m3",
    "FP8E4M3FN": "fp8_e4m3", "FP8E5M2": "fp8_e5m2", "E4M3": "fp8_e4m3",
    "E5M2": "fp8_e5m2", "FP4": "mxfp4", "MXFP4": "mxfp4", "E2M1": "mxfp4",
    "INT8": "int8", "INT16": "int16", "INT32": "int32", "INT64": "int64",
    "HIFLOAT8": "hif8", "FP8": "fp8", "MXFP8": "mxfp8",
}
_VALID_DTYPES = frozenset(_CUBE_MACS) | {"fp32", "int16", "int32", "int64"}


def normalize_dtype(dtype: str) -> str:
    key = str(dtype).strip().lower()
    key = _DTYPE_ALIASES.get(key.upper(), key)
    if key not in _VALID_DTYPES:
        raise ProfileError(f"unsupported dtype: {dtype!r}")
    return key


# Whitepaper-rounded peaks at the reference clock (three independent columns).
_WHITEPAPER_CUBE: Mapping[tuple[str, int], Mapping[str, float]] = {
    ("Ascend950PR", 28): {"bf16": 378, "fp16": 378, "fp8": 756, "hif8": 756,
                          "mxfp8": 756, "mxfp4": 1513, "tf32": 189},
    ("Ascend950PR", 32): {"bf16": 432, "fp16": 432, "fp8": 865, "hif8": 865,
                          "mxfp8": 865, "mxfp4": 1730, "tf32": 216},
    ("Ascend950DT", 28): {"bf16": 378, "fp16": 378, "fp8": 756, "hif8": 756,
                          "mxfp8": 756, "mxfp4": 1513, "tf32": 189},
    ("Ascend950DT", 32): {"bf16": 432, "fp16": 432, "fp8": 865, "hif8": 865,
                          "mxfp8": 865, "mxfp4": 1730, "tf32": 216},
    ("Ascend950DT", 36): {"bf16": 486, "fp16": 486, "fp8": 973, "hif8": 973,
                          "mxfp8": 973, "mxfp4": 1946, "tf32": 243},
}
_WHITEPAPER_VECTOR: Mapping[tuple[str, int], Mapping[str, float]] = {
    ("Ascend950PR", 28): {"bf16": 47, "fp16": 47, "fp32": 23},
    ("Ascend950PR", 32): {"bf16": 54, "fp16": 54, "fp32": 27},
    ("Ascend950DT", 28): {"bf16": 47, "fp16": 47, "fp32": 23},
    ("Ascend950DT", 32): {"bf16": 54, "fp16": 54, "fp32": 27},
    ("Ascend950DT", 36): {"bf16": 60, "fp16": 60, "fp32": 30},
}
_WHITEPAPER_TOTAL: Mapping[tuple[str, int], Mapping[str, float]] = {
    ("Ascend950PR", 28): {"bf16": 425, "fp16": 425, "fp8": 804, "hif8": 804,
                          "mxfp8": 804, "mxfp4": 1561, "tf32": 212},
    ("Ascend950PR", 32): {"bf16": 486, "fp16": 486, "fp8": 919, "hif8": 919,
                          "mxfp8": 919, "mxfp4": 1784, "tf32": 243},
    ("Ascend950DT", 28): {"bf16": 425, "fp16": 425, "fp8": 804, "hif8": 804,
                          "mxfp8": 804, "mxfp4": 1561, "tf32": 212},
    ("Ascend950DT", 32): {"bf16": 486, "fp16": 486, "fp8": 919, "hif8": 919,
                          "mxfp8": 919, "mxfp4": 1784, "tf32": 243},
    ("Ascend950DT", 36): {"bf16": 547, "fp16": 547, "fp8": 1034, "hif8": 1034,
                          "mxfp8": 1034, "mxfp4": 2007, "tf32": 273},
}
_WHITEPAPER_2201: Mapping[str, Mapping[str, float]] = {
    "bf16": {"cube": 177.4, "vector": 12.3, "total": 189.7},
    "fp16": {"cube": 177.4, "vector": 12.3, "total": 189.7},
    "fp32": {"cube": 44.3, "vector": 12.3, "total": 56.6},
    "int8": {"cube": 354.8, "vector": 12.3, "total": 367.1},
}


@dataclass(frozen=True, slots=True)
class RateModel:
    """Per-owner service rates; chip totals are derived from active cores."""

    vector_ops_s: float
    trans_ops_s: float
    simt_ops_s: float
    cube_macs_s: float
    fix_bytes_s: float
    mte1_bytes_s: float
    hbm_bytes_s: float
    l2_bytes_s: float
    rw_contention_alpha: float = 0.30
    mte2_aic_bytes_s: float = 100e9
    mte2_aiv_bytes_s: float = 40e9
    mte3_bytes_s: float = 100e9
    per_owner_port_bw: float = 100e9


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """One concrete SKU with explicit specification status."""

    family: str
    sku: str
    arch: str
    arch_code: int
    aic_cores: int
    aiv_cores: int
    freq_mhz: int
    memory: MemoryGeometry
    rates: RateModel
    fixpipe_to_ub: bool
    spec_status: str = SpecStatus.COMPLETE_PUBLIC_SPEC.value
    prediction_ready: bool = True
    missing_parameters: tuple[str, ...] = ()
    reference_freq_mhz: int = 0
    geometry_basis: str = "sku_catalog"
    overridden_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("fixpipe_to_ub", "prediction_ready"):
            if not isinstance(getattr(self, name), bool):
                raise ProfileError(f"{name} must be bool")

    @property
    def npu_arch(self) -> str:
        return self.arch

    @property
    def cube_vector_ratio(self) -> str:
        return f"1:{self.aiv_cores // self.aic_cores}"

    @property
    def chip_vector_ops_s(self) -> float:
        return self.rates.vector_ops_s * self.aiv_cores

    @property
    def chip_cube_macs_s(self) -> float:
        return self.rates.cube_macs_s * self.aic_cores

    @property
    def l2_size_bytes(self) -> int:
        return self.memory.l2

    @property
    def hbm_bw_gbps(self) -> float:
        return self.rates.hbm_bytes_s / 1e9

    @property
    def l2_bw_gbps(self) -> float:
        return self.rates.l2_bytes_s / 1e9

    @property
    def ub_bytes_per_lane(self) -> int:
        return self.memory.ub_per_lane

    @property
    def is_complete(self) -> bool:
        return self.prediction_ready and not self.missing_parameters

    def dtype_rates(self, dtype: str = "bf16") -> tuple[float, float]:
        """Return (Cube MACs/cycle, Vector ops/cycle) for one dtype."""
        key = normalize_dtype(dtype)
        if self.arch_code == 2201 and key in ("fp8", "fp8_e4m3", "fp8_e5m2", "hif8", "mxfp8", "mxfp4"):
            raise ProfileError(f"2201 has no supported {key} arithmetic route")
        table = _VEC_3510 if self.arch_code == 3510 else _VEC_2201
        cube = float(_CUBE_MACS.get(key, 0))
        if self.arch_code == 2201 and cube:
            cube *= 0.5  # V220 BF16 Cube is half the 3510 per-cycle width
        return cube, float(table[key])

    def cube_macs_per_second(self, dtype: str = "bf16") -> float:
        macs, _ = self.dtype_rates(dtype)
        return macs * self.freq_mhz * 1e6

    def vector_ops_per_second(self, dtype: str = "bf16") -> float:
        _, ops = self.dtype_rates(dtype)
        return ops * self.freq_mhz * 1e6

    def rate_for(self, pipe: str, dtype: str = "bf16") -> float:
        key = str(pipe).upper()
        if key == "VEC":
            return self.vector_ops_per_second(dtype)
        if key == "VEC_TRANS":
            base = self.vector_ops_per_second("bf16")
            return self.rates.trans_ops_s * (self.vector_ops_per_second(dtype) / base)
        if key == "SIMT":
            base = self.vector_ops_per_second("bf16")
            return self.rates.simt_ops_s * (self.vector_ops_per_second(dtype) / base)
        if key == "CUBE":
            rate = self.cube_macs_per_second(dtype)
            if rate <= 0:
                raise ProfileError(f"Cube does not support dtype {dtype!r}")
            return rate
        if key == "FIX":
            return self.rates.fix_bytes_s
        if key == "MTE1":
            return self.rates.mte1_bytes_s
        if key == "MTE2_AIC":
            return self.rates.mte2_aic_bytes_s
        if key == "MTE2_AIV":
            return self.rates.mte2_aiv_bytes_s
        if key == "MTE3":
            return self.rates.mte3_bytes_s
        raise KeyError(f"unknown pipe rate: {pipe}")

    def active_units(self, archetype: str) -> int:
        key = str(archetype).upper()
        if key == "PURE_VECTOR":
            return self.aiv_cores
        if key == "PURE_CUBE":
            return self.aic_cores
        if key == "CV_COUPLED_1C2V":
            return self.aic_cores
        if key == "COMMUNICATION_HYBRID":
            return self.aiv_cores
        raise KeyError(f"unknown archetype: {archetype}")

    def supports_fixpipe_l0c_to_ub(self) -> bool:
        return self.arch_code == 3510 and self.fixpipe_to_ub

    def derived_nominal_peaks(self, dtype: str = "bf16", freq_mhz: int | None = None) -> PeakTFLOPS:
        """Exact formula: macs/cycle * cores * clock * 2 and vector ops * clocks."""
        freq = self.freq_mhz if freq_mhz is None else _as_int("freq_mhz", freq_mhz)
        macs, ops = self.dtype_rates(dtype)
        cube = self.aic_cores * macs * freq * 1e6 * 2 / 1e12
        vector = self.aiv_cores * ops * freq * 1e6 / 1e12
        return PeakTFLOPS(cube + vector, cube, vector)

    def whitepaper_rounded_peaks(self, dtype: str = "bf16") -> PeakTFLOPS:
        """Static reference table at the documented reference clock."""
        key = normalize_dtype(dtype)
        if self.arch_code == 2201:
            row = _WHITEPAPER_2201.get(key)
            if row is None:
                raise ProfileError(f"no whitepaper peak for 2201 dtype {dtype!r}")
            scale = self.aic_cores / 24.0
            return PeakTFLOPS(row["total"] * scale, row["cube"] * scale, row["vector"] * scale)
        family = "Ascend950PR" if self.family == "Ascend950PR" else "Ascend950DT"
        key_name = "fp8" if key in ("fp8_e4m3", "fp8_e5m2") else key
        if key_name == "int8":
            key_name = "fp8"
        if key_name == "hf32":
            key_name = "tf32"
        if key_name not in ("bf16", "fp16", "fp8", "hif8", "mxfp8", "mxfp4", "tf32", "fp32"):
            raise ProfileError(f"no whitepaper peak for {family} dtype {dtype!r}")
        table_key = (family, self.aic_cores)
        if table_key not in _WHITEPAPER_CUBE:
            raise ProfileError(f"no whitepaper row for {family} {self.aic_cores}c")
        if key_name == "fp32":
            vector = _WHITEPAPER_VECTOR[table_key].get("fp32", 0.0)
            return PeakTFLOPS(vector, 0.0, vector)
        cube = _WHITEPAPER_CUBE[table_key][key_name]
        vector = _WHITEPAPER_VECTOR[table_key]["fp32" if key_name == "tf32" else "bf16"]
        total = _WHITEPAPER_TOTAL[table_key][key_name]
        return PeakTFLOPS(total, cube, vector)

    def peaks(self, dtype: str = "bf16") -> PeakTFLOPS:
        return self.derived_nominal_peaks(dtype)

    def cube_peak_tflops(self, dtype: str = "bf16") -> float:
        return self.derived_nominal_peaks(dtype).cube

    def vector_peak_tflops(self, dtype: str = "bf16") -> float:
        return self.derived_nominal_peaks(dtype).vector

    def advertised_peak_tflops(self, dtype: str = "bf16") -> float:
        """Rounded marketing total (independent Cube+Vector column)."""
        return self.whitepaper_rounded_peaks(dtype).total

    def ub_usable_bytes(self, vf_mode: str = "simd") -> int:
        mode = str(vf_mode).strip().lower()
        if mode not in ("simd", "simt"):
            raise ProfileError(f"unknown vf_mode: {vf_mode!r}")
        if self.arch_code == 2201:
            if mode == "simt":
                raise ProfileError(
                    "vf(mode='simt') is unavailable on NPU ARCH 2201/V220; "
                    "fall back to SIMD or select DAV_3510"
                )
            return min(self.memory.ub_per_lane, UB_2201_SIMD_BYTES)
        if self.arch_code != 3510:
            raise ProfileError(f"unknown NPU arch code: {self.arch_code}")
        physical = min(self.memory.ub_per_lane, UB_3510_SIMD_BYTES)
        return physical - UB_3510_SIMT_CARVE_BYTES if mode == "simt" else physical

    def ssbuf_bytes(self, is_mix: bool = True) -> int:
        if not isinstance(is_mix, bool):
            raise ProfileError("is_mix must be bool")
        if self.arch_code != 3510:
            return 0
        return SSBUF_3510_MIX_BYTES if is_mix else SSBUF_3510_PURE_BYTES

    def require_prediction_ready(self) -> None:
        if self.prediction_ready:
            return
        missing = ", ".join(self.missing_parameters) or "core_count, frequency_mhz"
        raise ProfileError(
            f"SKU {self.sku!r} is {self.spec_status}; missing parameters: {missing}. "
            "Supply them explicitly via get_profile(..., aic_cores=..., freq_mhz=...)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "sku": self.sku, "arch": self.arch,
            "arch_code": self.arch_code, "aic": self.aic_cores, "aiv": self.aiv_cores,
            "freq_mhz": self.freq_mhz, "ub": self.memory.ub_per_lane,
            "l1": self.memory.l1, "l0a": self.memory.l0a, "l0b": self.memory.l0b,
            "l0c": self.memory.l0c, "l2": self.memory.l2,
            "hbm_gbps": round(self.hbm_bw_gbps, 3), "l2_gbps": round(self.l2_bw_gbps, 3),
            "fixpipe_ub": self.fixpipe_to_ub, "spec_status": self.spec_status,
            "prediction_ready": self.prediction_ready,
            "missing_parameters": list(self.missing_parameters),
            "reference_freq_mhz": self.reference_freq_mhz,
            "geometry_basis": self.geometry_basis,
        }


_EDGES_3510 = frozenset({
    "GM->UB", "UB->GM", "GM->L1", "L1->L0A", "L1->L0B", "L1->BIAS",
    "L0C->GM", "L0C->L1", "L0C->UB", "UB->L1", "L1->UB",
})
_EDGES_2201 = frozenset({
    "GM->UB", "UB->GM", "GM->L1", "L1->GM", "L1->L0A", "L1->L0B",
    "GM->L0A", "GM->L0B", "L0C->GM", "L0C->L1", "L1->BIAS",
})
_LOC_ALIASES = {
    "UB_LANE0": "UB", "UB_LANE1": "UB", "UNIFIED_BUFFER": "UB",
    "GLOBAL": "GM", "GLOBAL_MEMORY": "GM",
}


def _normalize_loc(loc: str) -> str:
    key = str(loc).strip().upper()
    return _LOC_ALIASES.get(key, key)


def legal_mem_copy_edges(profile: HardwareProfile) -> tuple[str, ...]:
    if profile.arch_code == 3510:
        return tuple(sorted(_EDGES_3510))
    if profile.arch_code == 2201:
        return tuple(sorted(_EDGES_2201))
    raise ProfileError(f"unknown NPU arch code: {profile.arch_code}")


def supports_mem_copy(src: str, dst: str, profile: HardwareProfile) -> bool:
    return f"{_normalize_loc(src)}->{_normalize_loc(dst)}" in legal_mem_copy_edges(profile)


def validate_mem_copy(src: str, dst: str, profile: HardwareProfile) -> str:
    edge = f"{_normalize_loc(src)}->{_normalize_loc(dst)}"
    if edge not in legal_mem_copy_edges(profile):
        raise MemCopyError(
            f"mem_copy {edge} is illegal on {profile.arch}/{profile.family}; "
            "3510 removes L1->GM and GM->L0A/L0B, while 2201 lacks UB<->L1 and L0C->UB"
        )
    return edge


# Public resolution API lives in the catalog module to keep this file focused.
from .hardware_catalog import (  # noqa: E402  (definitions above avoid a cycle)
    catalog, find_profile, get_profile, list_profiles, resolve_profile,
)

check_mem_copy_edge = validate_mem_copy
is_legal_mem_copy = supports_mem_copy
mem_copy_edges = legal_mem_copy_edges

__all__ = [
    "HardwareProfile", "MemCopyEdge", "MemCopyError", "MemoryGeometry",
    "PeakTFLOPS", "ProfileError", "RateModel", "SpecStatus",
    "SSBUF_3510_MIX_BYTES", "SSBUF_3510_PURE_BYTES", "UB_2201_SIMD_BYTES",
    "UB_3510_SIMD_BYTES", "UB_3510_SIMT_BYTES", "catalog", "check_mem_copy_edge",
    "find_profile", "get_profile", "is_legal_mem_copy", "legal_mem_copy_edges",
    "list_profiles", "mem_copy_edges", "normalize_dtype", "resolve_profile",
    "supports_mem_copy", "validate_mem_copy",
]
