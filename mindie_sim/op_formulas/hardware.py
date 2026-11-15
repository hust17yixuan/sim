"""Ascend SKU peaks, strict dtype/precision tables and qualified A5 fits.

[Role]
  Own the single hardware truth table: peak HBM/Cube/Vector rates in their
  native *operations per second* units, L2 capacity/bandwidth, core counts,
  launch classes, declared UB/HCCS/RoCE transport peaks, and the measured A5
  collective/MC2 fits.  Also own strict dtype normalization and the storage ->
  Cube-precision mapping used by every GEMM-family builder.

[Invariants]
  * ``vector_ops_per_s`` is the declared lane-operation rate from the SKU
    profile (A5 ``rate_ops_per_s = 6.7584e12``); it is never divided by a byte
    width and never conflated with ``hbm_bps``.
  * Every rate is stored with an explicit unit suffix in its name; a Cube peak is
    selected by the real compute precision and a missing tier raises instead of
    silently falling back to BF16.
  * ``dtype_bytes`` maps every declared alias to an exact storage width and
    raises ``FormulaValidationError`` for any unknown dtype (no 2-byte FP8, no
    silent BF16 default).
  * A measured fabric/MC2 fit is returned only for A5 + UB transport; A2/A3 and
    RoCE callers always fall back to the analytic alpha-beta model.
  * Constants are immutable mappings; no caller can mutate shared calibration.

[Timeouts & Locks]
  Pure import-time constants and pure lookups; no I/O, locks or device access.

[Side-Effect Policy]
  Read-only declarations.  No file, network or device side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from .errors import FormulaValidationError

LAUNCH_MODES = ("superkernel_subtask", "aclgraph", "eager", "python_eager")


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _norm(value: str | None) -> str:
    return "".join(char for char in str(value or "").strip().lower()
                   if char.isalnum())


# canonical -> bytes per element
_DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0, "hf32": 4.0, "tf32": 4.0, "bf16": 2.0, "fp16": 2.0,
    "int8": 1.0, "uint8": 1.0, "bool": 1.0,
    "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "hif8": 1.0, "fp8": 1.0,
    "int32": 4.0, "uint32": 4.0, "int64": 8.0, "uint64": 8.0,
    "int4": 0.5, "uint4": 0.5, "fp4": 0.5, "mxfp4": 0.5,
    "mxfp8": 1.0, "w4a16": 0.5, "w4a8": 0.5, "w8a8": 1.0,
}

_DTYPE_ALIASES: dict[str, str] = {}


def _alias(canonical: str, *names: str) -> None:
    for name in names:
        _DTYPE_ALIASES[_norm(name)] = canonical


_alias("fp32", "fp32", "float32", "float", "dt_float")
_alias("hf32", "hf32", "tf32", "dt_hf32")
_alias("bf16", "bf16", "bfloat16", "dt_bf16", "brainfloat16")
_alias("fp16", "fp16", "float16", "half", "dt_float16")
_alias("int8", "int8", "i8", "dt_int8", "s8")
_alias("uint8", "uint8", "u8", "dt_uint8")
_alias("bool", "bool", "dt_bool", "boolean")
_alias("fp8_e4m3", "fp8_e4m3", "fp8_e4m3fn", "float8_e4m3fn", "e4m3",
       "fp8e4m3", "hif8")
_alias("fp8_e5m2", "fp8_e5m2", "float8_e5m2", "e5m2", "fp8e5m2")
_alias("fp8", "fp8", "float8", "dt_fp8")
_alias("int32", "int32", "i32", "dt_int32")
_alias("uint32", "uint32", "u32", "dt_uint32")
_alias("int64", "int64", "i64", "dt_int64")
_alias("uint64", "uint64", "u64", "dt_uint64")
_alias("int4", "int4", "i4", "dt_int4")
_alias("uint4", "uint4", "u4")
_alias("fp4", "fp4", "float4", "nvfp4")
_alias("mxfp4", "mxfp4", "mxfloat4")
_alias("mxfp8", "mxfp8", "mxfloat8")
_alias("w4a16", "w4a16")
_alias("w4a8", "w4a8")
_alias("w8a8", "w8a8")


def dtype_bytes(dtype: str | None) -> float:
    """Bytes per element for a declared dtype; unknown names are rejected."""
    if not isinstance(dtype, str) or not dtype.strip():
        raise FormulaValidationError(
            f"dtype must be a declared string, got {dtype!r}")
    canonical = _DTYPE_ALIASES.get(_norm(dtype))
    if canonical is None:
        raise FormulaValidationError(
            f"unknown dtype {dtype!r}; declare a storage type explicitly")
    return _DTYPE_BYTES[canonical]


def canonical_dtype(dtype: str | None) -> str:
    if not isinstance(dtype, str) or not dtype.strip():
        raise FormulaValidationError("dtype must be a declared string")
    canonical = _DTYPE_ALIASES.get(_norm(dtype))
    if canonical is None:
        raise FormulaValidationError(f"unknown dtype {dtype!r}")
    return canonical


_QUANT_TIER: dict[str, str] = {
    "w8a8": "int8", "int8": "int8", "int8dynamic": "int8", "w8a16": "bf16",
    "w4a16": "bf16", "a16w4": "bf16", "w4a8": "int8", "w4a4": "int8",
    "fp8": "fp8", "fp8e4m3": "fp8", "fp8e5m2": "fp8", "mxfp8": "fp8",
    "fp8blockwise": "fp8", "hif8": "fp8", "fp4": "fp4", "mxfp4": "fp4",
    "nvfp4": "fp4",
}

_DTYPE_TIER: dict[str, str] = {
    "fp32": "fp32", "hf32": "hf32", "bf16": "bf16", "fp16": "fp16",
    "int8": "int8", "uint8": "int8", "fp8": "fp8", "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8", "mxfp8": "fp8", "fp4": "fp4", "mxfp4": "fp4",
    "int4": "fp4", "uint4": "fp4",
}

DTYPE_BITS: Mapping[str, float] = _freeze(
    {name: value * 8.0 for name, value in _DTYPE_BYTES.items()})


def cube_tier(dtype: str | None = None, quant: str | None = None) -> str:
    """Map storage/quant spelling to the physical Cube MAC precision tier."""
    if quant is not None:
        tier = _QUANT_TIER.get(_norm(quant))
        if tier is None:
            raise FormulaValidationError(f"unknown quant_mode {quant!r}")
        return tier
    canonical = canonical_dtype(dtype)
    tier = _DTYPE_TIER.get(canonical)
    if tier is None:
        raise FormulaValidationError(
            f"dtype {dtype!r} has no Cube MAC precision tier")
    return tier


def cube_dtype(dtype: str | None, quant: str | None = None) -> str:
    """Backwards-compatible alias for ``cube_tier``."""
    return cube_tier(dtype, quant)


@dataclass(frozen=True)
class HardwareProfile:
    """One immutable SKU specification used by every formula card."""

    sku: str
    arch: str
    cores: int
    hbm_bps: float
    vector_ops_per_s: float
    cube_ops_per_s: Mapping[str, float]
    l2_bps: float
    l2_budget_bytes: int
    ub_bytes: int
    l1_bytes: int
    ub_intra_bps: float
    ub_superpod_bps: float
    roce_bps: float
    hccs_bps: float
    launch_floor_us: Mapping[str, float]
    host_dispatch_us: Mapping[str, float]
    granule_bytes: Mapping[str, int]
    default_granule: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cube_ops_per_s", _freeze(self.cube_ops_per_s))
        object.__setattr__(self, "launch_floor_us", _freeze(self.launch_floor_us))
        object.__setattr__(self, "host_dispatch_us", _freeze(self.host_dispatch_us))
        object.__setattr__(self, "granule_bytes", _freeze(self.granule_bytes))
        if min(self.hbm_bps, self.vector_ops_per_s, self.l2_bps,
               self.ub_intra_bps, self.ub_superpod_bps, self.roce_bps,
               self.hccs_bps) <= 0:
            raise ValueError(f"{self.sku}: all rates must be positive")
        for tier, rate in self.cube_ops_per_s.items():
            if float(rate) <= 0:
                raise ValueError(f"{self.sku}: cube tier {tier} must be positive")
        floors = [self.launch_floor_us[mode] for mode in LAUNCH_MODES]
        if min(floors) <= 0 or not (floors[0] < floors[1] < floors[2]):
            raise ValueError(f"{self.sku}: launch floors must be positive and ordered")

    def cube_peak(self, tier: str) -> float:
        key = str(tier).strip().lower()
        if key not in self.cube_ops_per_s:
            raise FormulaValidationError(
                f"{self.sku}: no declared Cube peak for tier {tier!r}")
        return float(self.cube_ops_per_s[key])

    def launch_floor(self, launch_mode: str) -> float:
        if launch_mode not in self.launch_floor_us:
            raise FormulaValidationError(
                f"{self.sku}: unknown launch mode {launch_mode!r}")
        return float(self.launch_floor_us[launch_mode])

    def host_dispatch(self, family: str, launch_mode: str) -> float:
        if launch_mode != "python_eager":
            return 0.0
        return float(self.host_dispatch_us.get(family, 20.0))

    def granule_for(self, layout: str | None) -> int:
        key = str(layout or "ND").strip().upper()
        return int(self.granule_bytes.get(key, self.default_granule))

    def link_bps(self, world_size: int, transport: str | None = None) -> float:
        """Declared per-rank link peak for an explicit transport."""
        kind = str(transport or ("ub" if self.arch == "A5" else "hccs")).lower()
        if kind == "ub":
            return float(self.ub_superpod_bps if int(world_size) > 8
                         else self.ub_intra_bps)
        if kind in {"hccs", "hccs_ub"}:
            return float(self.hccs_bps)
        if kind == "roce":
            return float(self.roce_bps)
        raise FormulaValidationError(f"unknown transport {transport!r}")

    def transport_label(self, world_size: int, transport: str | None = None) -> str:
        kind = str(transport or ("ub" if self.arch == "A5" else "hccs")).lower()
        if kind == "ub":
            return ("a5_ub_urma_superpod" if int(world_size) > 8
                    else "a5_ub_urma_intra_node")
        if kind in {"hccs", "hccs_ub"}:
            return "hccs_intra_node"
        if kind == "roce":
            return "roce_inter_node"
        raise FormulaValidationError(f"unknown transport {transport!r}")

    def l2_hot_budget(self) -> int:
        return int(self.l2_budget_bytes)


_COMMON_HOST = {
    "matmul": 18.5, "grouped_matmul": 20.0, "quant": 19.5, "rms_norm": 21.5,
    "swiglu": 20.0, "rope": 20.5, "attention": 46.0, "mla": 46.0,
    "linear_attention": 24.0, "conv": 21.0, "moe_routing": 20.0,
    "moe_distribute": 30.0, "collective": 20.0, "embedding": 20.0,
    "sampler": 21.0, "composite": 18.5,
}

_GRANULES = {
    "ND": 128, "TND": 128, "BSND": 128, "BNSD": 128, "BSH": 128, "SBH": 128, "NCHW": 128,
    "FRACTAL_NZ": 512, "NZ": 512, "FRACTAL_Z": 512, "SCALAR": 32,
}

A5 = HardwareProfile(
    sku="Ascend950DT_9581", arch="A5", cores=32,
    hbm_bps=4.0e12, vector_ops_per_s=6.7584e12,
    cube_ops_per_s={"bf16": 432.5376e12, "fp16": 432.5376e12,
                    "int8": 865.0752e12, "fp8": 865.0752e12,
                    "fp4": 1730.1504e12,
                    "fp32": 27.0336e12, "hf32": 216.2688e12},
    l2_bps=8.0e12, l2_budget_bytes=96 * 1024 * 1024,
    ub_bytes=253952, l1_bytes=524288,
    ub_intra_bps=420.0e9, ub_superpod_bps=512.0e9,
    roce_bps=50.0e9, hccs_bps=420.0e9,
    launch_floor_us={"superkernel_subtask": 0.45, "aclgraph": 1.10,
                     "eager": 4.0, "python_eager": 4.0},
    host_dispatch_us=_COMMON_HOST,
    granule_bytes=_GRANULES, default_granule=128,
)

A3 = HardwareProfile(
    sku="Ascend910_9362", arch="A3", cores=20,
    hbm_bps=1.2e12, vector_ops_per_s=2.112e12,
    cube_ops_per_s={"bf16": 270.336e12, "fp16": 270.336e12,
                    "int8": 540.672e12, "fp8": 540.672e12,
                    "fp4": 1081.344e12,
                    "fp32": 16.896e12, "hf32": 135.168e12},
    l2_bps=3.2e12, l2_budget_bytes=64 * 1024 * 1024,
    ub_bytes=196608, l1_bytes=524288,
    ub_intra_bps=196.0e9, ub_superpod_bps=196.0e9,
    roce_bps=50.0e9, hccs_bps=196.0e9,
    launch_floor_us={"superkernel_subtask": 0.50, "aclgraph": 1.50,
                     "eager": 5.0, "python_eager": 5.0},
    host_dispatch_us=_COMMON_HOST,
    granule_bytes=_GRANULES, default_granule=128,
)

A2 = HardwareProfile(
    sku="Ascend910B2C", arch="A2", cores=24,
    hbm_bps=1.8e12, vector_ops_per_s=2.7648e12,
    cube_ops_per_s={"bf16": 353.8944e12, "fp16": 353.8944e12,
                    "int8": 707.7888e12, "fp8": 707.7888e12,
                    "fp4": 1415.5776e12,
                    "fp32": 22.1184e12, "hf32": 176.9472e12},
    l2_bps=3.6e12, l2_budget_bytes=64 * 1024 * 1024,
    ub_bytes=196608, l1_bytes=524288,
    ub_intra_bps=168.0e9, ub_superpod_bps=168.0e9,
    roce_bps=25.0e9, hccs_bps=168.0e9,
    launch_floor_us={"superkernel_subtask": 0.55, "aclgraph": 1.60,
                     "eager": 5.0, "python_eager": 5.0},
    host_dispatch_us=_COMMON_HOST,
    granule_bytes=_GRANULES, default_granule=128,
)

A5_9582 = replace(A5, sku="Ascend950DT_9582")
A5_PR = replace(A5, sku="Ascend950PR_9579", hbm_bps=1.6e12)
A3_9391 = replace(A3, sku="Ascend910_9391", cores=24, hbm_bps=1.6e12,
                  vector_ops_per_s=2.8416e12,
                  cube_ops_per_s={**dict(A3.cube_ops_per_s),
                                  "bf16": 363.7248e12, "fp16": 363.7248e12,
                                  "int8": 727.4496e12, "fp8": 727.4496e12,
                                  "fp4": 1454.8992e12,
                                  "fp32": 22.1184e12, "hf32": 181.8624e12})
A3_9381 = replace(A3, sku="Ascend910_9381", hbm_bps=1.6e12,
                  vector_ops_per_s=2.304e12,
                  cube_ops_per_s={**dict(A3.cube_ops_per_s),
                                  "bf16": 294.912e12, "fp16": 294.912e12,
                                  "int8": 589.824e12, "fp8": 589.824e12,
                                  "fp4": 1179.648e12,
                                  "fp32": 18.432e12, "hf32": 147.456e12})
A2_B1 = replace(A2, sku="Ascend910B1",
                cube_ops_per_s={**dict(A2.cube_ops_per_s),
                                "bf16": 363.7248e12, "fp16": 363.7248e12,
                                "int8": 727.4496e12, "fp8": 727.4496e12,
                                "fp4": 1454.8992e12,
                                "fp32": 22.1184e12, "hf32": 181.8624e12})

SKUS: Mapping[str, HardwareProfile] = _freeze({
    "Ascend950DT_9581": A5, "Ascend950DT_9582": A5_9582,
    "Ascend950PR_9579": A5_PR, "Ascend910_9362": A3,
    "Ascend910_9391": A3_9391, "Ascend910_9381": A3_9381,
    "Ascend910B2C": A2, "Ascend910B1": A2_B1, "Ascend910B2": A2,
})

SKU_ALIASES: Mapping[str, str] = _freeze({
    "a5": "Ascend950DT_9581", "950": "Ascend950DT_9581",
    "950dt": "Ascend950DT_9581", "ascend950dt": "Ascend950DT_9581",
    "a5poddt": "Ascend950DT_9581", "a5serverdt": "Ascend950DT_9581",
    "a3": "Ascend910_9362", "910c": "Ascend910_9362",
    "ascend9109362": "Ascend910_9362", "ascend9109391": "Ascend910_9391",
    "ascend9109381": "Ascend910_9381",
    "a2": "Ascend910B2C", "910b": "Ascend910B2C",
    "ascend910b2c": "Ascend910B2C", "ascend910b1": "Ascend910B1",
    "a5podpr": "Ascend950PR_9579", "a5serverpr": "Ascend950PR_9579",
    "950pr": "Ascend950PR_9579", "ascend950pr": "Ascend950PR_9579",
    "950dt9582": "Ascend950DT_9582", "ascend950dt9582": "Ascend950DT_9582",
})

DEFAULT_SKU = "Ascend950DT_9581"


def resolve_sku(sku: str | None = None) -> HardwareProfile:
    text = str(sku or DEFAULT_SKU).strip()
    if text in SKUS:
        return SKUS[text]
    canonical = SKU_ALIASES.get(_norm(text))
    if canonical and canonical in SKUS:
        return SKUS[canonical]
    raise FormulaValidationError(
        f"unknown SKU {sku!r}; try one of {sorted(SKUS)} or a5/a3/a2")


# Measured A5 fabric fits: (op, world_size, tier) -> (alpha_us, beta_ns_per_byte).
A5_FABRIC_FITS: Mapping[tuple[str, int, str], tuple[float, float]] = _freeze({
    ("all_reduce", 2, "small"): (49.96, 0.0307), ("all_reduce", 2, "large"): (74.7, 0.0213),
    ("all_gather", 2, "small"): (48.01, 0.0115), ("all_gather", 2, "large"): (63.39, 0.0039),
    ("reduce_scatter", 2, "small"): (52.73, 0.0105), ("reduce_scatter", 2, "large"): (69.81, 0.0041),
    ("all_to_all_single", 2, "small"): (47.46, 0.0119), ("all_to_all_single", 2, "large"): (60.53, 0.0056),
    ("all_reduce", 4, "small"): (62.14, 0.0196), ("all_reduce", 4, "large"): (93.77, 0.0116),
    ("all_gather", 4, "small"): (69.77, 0.0554), ("all_gather", 4, "large"): (78.72, 0.0078),
    ("reduce_scatter", 4, "small"): (65.31, 0.0114), ("reduce_scatter", 4, "large"): (248.49, 0.0116),
    ("all_to_all_single", 4, "large"): (117.12, 0.0033),
    ("all_reduce", 8, "small"): (71.29, 0.0169), ("all_reduce", 8, "large"): (125.6, 0.0083),
    ("all_gather", 8, "small"): (66.9, 0.0), ("all_gather", 8, "large"): (84.58, 0.0141),
    ("reduce_scatter", 8, "small"): (70.49, 0.0), ("reduce_scatter", 8, "large"): (96.03, 0.0243),
    ("all_to_all_single", 8, "small"): (65.38, 0.0), ("all_to_all_single", 8, "large"): (69.18, 0.0019),
    ("all_reduce", 16, "small"): (42.63, 0.0367), ("all_reduce", 16, "large"): (75.79, 0.0074),
    ("all_gather", 16, "small"): (74.95, 0.0), ("all_gather", 16, "large"): (135.97, 0.04),
    ("reduce_scatter", 16, "small"): (42.14, 0.038), ("reduce_scatter", 16, "large"): (151.52, 0.0485),
    ("all_to_all_single", 16, "small"): (43.24, 0.0453), ("all_to_all_single", 16, "large"): (96.32, 0.0016),
})

# Measured A5 MC2 decode fits: ep -> stage -> (alpha_us, us_per_expanded_token).
A5_MC2_FITS: Mapping[int, Mapping[str, tuple[float, float]]] = _freeze({
    2: _freeze({"dispatch": (286.2, 0.1076), "combine": (248.2, 0.1265)}),
    4: _freeze({"dispatch": (528.5, 0.0337), "combine": (527.6, 0.0)}),
    8: _freeze({"dispatch": (1221.5, 0.0139), "combine": (1298.4, 0.0)}),
    16: _freeze({"dispatch": (1135.6, 0.0), "combine": (962.9, 0.0)}),
})

MC2_FIT_DOMAIN = _freeze({"max_tokens_per_rank": 64, "skew": 1.0,
                          "arch": "A5", "transport": "ub",
                          "hidden": 7168, "experts_total": 64, "topk": 8,
                          "dtype": "bf16", "quant": "none"})

# Declared per-step collective latency (us) by transport label.  Ring/H-D/mesh
# schedules consume this so the advertised step count enters t_ref.
COLLECTIVE_STEP_ALPHA_US: Mapping[str, float] = _freeze({
    "ub": 0.18, "hccs": 0.35, "roce": 0.55, "auto": 0.35,
})


def collective_step_alpha_us(transport: str | None) -> float:
    """Per-step synchronization latency for a declared transport label."""
    text = str(transport or "").strip().lower()
    for key, value in COLLECTIVE_STEP_ALPHA_US.items():
        if key != "auto" and key in text:
            return float(value)
    return float(COLLECTIVE_STEP_ALPHA_US["auto"])

CALIBRATION_PROVENANCE: Mapping[str, str] = _freeze({
    "a5_peaks": "mindie_sim/backends/profiles/ascend.json:Ascend950DT_9581",
    "a5_fabric": "live_a5_captures/data/fabric_a5_16npu.json (fits[]); "
                 "A5 UB/URMA transport only",
    "a5_mc2": "live_a5_captures/data/mc2_megamoe_ep_sweep.json "
              "(decode T<=64, balanced skew=1.0 only)",
    "a5_launch": "A5 device kernels: rms_norm/matmul/FIA eager p50 asymptotes",
    "a5_l2": "L2 model: 2x HBM peak, 96 MiB hot budget (declared engineering envelope)",
    "a5_transport": "UB intra-node 420 GB/s, UB superpod 512 GB/s per-rank, "
                    "RoCE 50 GB/s unidirectional (declared; not interchangeable)",
})


def _fit_allowed(hardware: HardwareProfile, transport: str | None) -> bool:
    kind = str(transport or ("ub" if hardware.arch == "A5" else "hccs")).lower()
    return hardware.arch == "A5" and kind == "ub"


def fabric_fit(op: str, world_size: int, size_bytes: int,
               hardware: HardwareProfile | None = None,
               transport: str | None = None,
               threshold_bytes: int = 1024 * 1024) -> tuple[float, float] | None:
    """Return the measured A5 UB fit or ``None`` outside its declared domain."""
    if hardware is None or not _fit_allowed(hardware, transport):
        return None
    tier = "large" if int(size_bytes) >= int(threshold_bytes) else "small"
    return A5_FABRIC_FITS.get((op, int(world_size), tier))


def _mc2_geometry_ok(hidden: int | None, experts_total: int | None,
                     topk: int | None, dtype: str | None,
                     quant: str | None) -> bool:
    checks = (
        (hidden, int(MC2_FIT_DOMAIN["hidden"])),
        (experts_total, int(MC2_FIT_DOMAIN["experts_total"])),
        (topk, int(MC2_FIT_DOMAIN["topk"])),
    )
    for value, expected in checks:
        if value is not None and int(value) != expected:
            return False
    if dtype is not None and canonical_dtype(str(dtype)) != MC2_FIT_DOMAIN["dtype"]:
        return False
    if quant not in (None, "") and str(quant).strip().lower() != "none":
        return False
    return True


def mc2_fit(ep: int, stage: str, hardware: HardwareProfile | None = None,
            transport: str | None = None,
            tokens_per_rank: int | None = None,
            skew: float = 1.0,
            hidden: int | None = None, experts_total: int | None = None,
            topk: int | None = None, dtype: str | None = None,
            quant: str | None = None) -> tuple[float, float] | None:
    """Return the measured A5 MC2 decode fit inside its calibrated domain."""
    if hardware is None or not _fit_allowed(hardware, transport):
        return None
    if not _mc2_geometry_ok(hidden, experts_total, topk, dtype, quant):
        return None
    if tokens_per_rank is not None and (
            int(tokens_per_rank) > int(MC2_FIT_DOMAIN["max_tokens_per_rank"])):
        return None
    if float(skew) != float(MC2_FIT_DOMAIN["skew"]):
        return None
    row = A5_MC2_FITS.get(int(ep))
    return None if row is None else row.get(stage)


def mc2_fit_domain(tokens_per_rank: int, skew: float,
                   hardware: HardwareProfile, transport: str | None,
                   hidden: int | None = None,
                   experts_total: int | None = None,
                   topk: int | None = None, dtype: str | None = None,
                   quant: str | None = None) -> str:
    if not _fit_allowed(hardware, transport):
        return "outside_a5_ub_transport"
    if not _mc2_geometry_ok(hidden, experts_total, topk, dtype, quant):
        return "outside_captured_workload_geometry"
    if int(tokens_per_rank) > int(MC2_FIT_DOMAIN["max_tokens_per_rank"]):
        return "outside_balanced_decode_fit"
    if float(skew) != 1.0:
        return "outside_balanced_decode_fit"
    return "in_calibrated_ref_domain"


__all__ = [
    "A2", "A3", "A5", "A5_FABRIC_FITS", "A5_MC2_FITS", "CALIBRATION_PROVENANCE",
    "COLLECTIVE_STEP_ALPHA_US", "DEFAULT_SKU", "DTYPE_BITS", "HardwareProfile",
    "LAUNCH_MODES", "MC2_FIT_DOMAIN", "SKU_ALIASES", "SKUS", "canonical_dtype",
    "cube_dtype", "cube_tier", "collective_step_alpha_us", "dtype_bytes",
    "fabric_fit", "mc2_fit", "mc2_fit_domain", "resolve_sku",
]
