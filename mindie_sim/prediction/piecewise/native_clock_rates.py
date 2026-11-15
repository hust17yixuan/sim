"""Declared per-SKU sustained supply rates for the native clock physics laws.

[Role] Resolve one Ascend die's declared sustained hardware supply
  (``NativeRates``: HBM, MTE2-AIC, MTE2-AIV, MTE3, MTE1, vector, Fixpipe,
  Cube peak per dtype, scalar issue clock, L2/UB capacities, and class launch
  floors) from ``PipeSupply`` / ``KernelSupply`` or a materialized SKU name.
[Invariants]
  * Every rate is derived from the selected SKU's declared hardware profile;
    rates are never substituted across SKUs when a hardware supply is given.
  * The scalar issue rate scales with the die clock derived from the declared
    vector bytes/cycle width (64 B/cycle on A5, 32 B/cycle on A2/A3) and AIV
    owner count.
  * A2/A3 class launch baselines use the declared extrapolated profile floors
    when ``measured_floors`` is false.
[Timeouts & Locks] Pure bounded arithmetic with bounded ``lru_cache``; no locks.
[Side-Effect Policy] Read-only pure functions; no I/O, global mutation or
  device access.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

UB_BYTES = 253952
L2_BYTES = 128 * 1024 * 1024

HBM_RATE = 4.0e12
MTE2_RATE = 4.224e12
MTE2_AIV_RATE = 4.224e12
MTE3_RATE = 13.5168e12
VECTOR_RATE = 6.7584e12
FIXPIPE_RATE = 6.7584e12
SCALAR_ISSUE_RATE = 105.6e9
CUBE_BF16 = 400.0e12
CUBE_FP32 = 14.5e12
CUBE_INT8 = 800.0e12
CUBE_FP8 = 800.0e12
CUBE_SUSTAINED = 0.925

MTE2_EFFICIENCY = {"bf16": 0.891, "fp32": 0.352, "int8": 0.815, "mxfp8": 1.045}

DISPATCH_MS = 0.00026
LAUNCH_FLOOR_MS = {
    "matrix_bf16": 0.001578,
    "matrix_mxfp8": 0.003020,
    "matrix_int8": 0.001900,
    "transpose_batch": 0.002220,
    "row_norm": 0.001312,
    "fused_row_norm": 0.001312,
    "elementwise": 0.001312,
    "quantize": 0.001312,
    "swiglu": 0.001312,
    "vector": 0.001312,
}
EXTRAPOLATED_FLOOR_MS = {"vector": 0.0025, "cube": 0.00265, "matrix_int8": 0.00225}

_DTYPE_BYTES = {
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "fp32": 4,
    "float32": 4, "int8": 1, "uint8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "fp8": 1, "mxfp8": 1, "hf32": 2, "int32": 4, "int64": 8,
    "int4": 1, "fp4_e2m1": 1, "uint4": 1, "e2m1": 1,
}
_DTYPE_PEAK_KEYS = {
    "bf16": ("bf16",), "fp16": ("fp16",), "fp32": ("fp32",),
    "int8": ("int8",), "mxfp8": ("fp8_e4m3", "fp8_e5m2", "int8"),
    "fp8": ("fp8_e4m3", "fp8_e5m2"),
}
_VECTOR_BYTES_PER_CYCLE = {"A2": 32.0, "A3": 32.0, "A5": 64.0}
_REFERENCE_CLOCK_HZ = 1.65e9


@dataclass(frozen=True)
class NativeRates:
    """One die's declared sustained supply, in service units per second."""

    arch: str = "A5"
    aic: int = 32
    aiv: int = 64
    ub_bytes: int = UB_BYTES
    l2_bytes: int = L2_BYTES
    hbm: float = HBM_RATE
    mte2_aic: float = MTE2_RATE
    mte2_aiv: float = MTE2_AIV_RATE
    mte3: float = MTE3_RATE
    mte1: float = MTE2_RATE
    vector: float = VECTOR_RATE
    fixpipe: float = FIXPIPE_RATE
    scalar: float = SCALAR_ISSUE_RATE
    cube: tuple[tuple[str, float], ...] = (
        ("bf16", CUBE_BF16), ("fp16", CUBE_BF16), ("fp32", CUBE_FP32),
        ("int8", CUBE_INT8), ("mxfp8", CUBE_FP8))
    launch_ms: tuple[tuple[str, float], ...] = tuple(sorted(LAUNCH_FLOOR_MS.items()))
    measured_floors: bool = True
    clock_hz: float = _REFERENCE_CLOCK_HZ

    def cube_rate(self, kind: str) -> float:
        table = dict(self.cube)
        for key in _DTYPE_PEAK_KEYS.get(kind, (kind,)):
            if key in table:
                return float(table[key])
        return float(table.get("bf16") or CUBE_BF16)

    def launch(self, kernel_class: str) -> float:
        """Declared launch floor of one kernel class (or matrix dtype kind)."""
        table = dict(self.launch_ms)
        if kernel_class in table:
            return float(table[kernel_class])
        if kernel_class in {"int8", "w8a8"}:
            return float(table.get("matrix_int8") or LAUNCH_FLOOR_MS["matrix_int8"])
        if kernel_class in {"mxfp8", "fp8", "fp8_e4m3", "fp8_e5m2"}:
            return float(table.get("matrix_mxfp8") or LAUNCH_FLOOR_MS["matrix_mxfp8"])
        if kernel_class.startswith("matrix"):
            return float(table.get("matrix_bf16") or LAUNCH_FLOOR_MS["matrix_bf16"])
        return float(table.get("vector") or DISPATCH_MS)

    def mte2_stream(self, kind: str) -> float:
        """Sustained die copy-path rate of one panel stream (bytes/s)."""
        share = MTE2_EFFICIENCY.get(kind, MTE2_EFFICIENCY["bf16"])
        aic_burst = (0.85 * float(self.mte2_aiv) if self.arch == "A2"
                     else (0.66 * float(self.mte2_aiv) if self.arch == "A3"
                           else float(self.mte2_aic)))
        return max(1.0, max(aic_burst, float(self.hbm) * 1.056) * share)

    def mte2_local(self) -> float:
        return max(1.0, float(self.mte2_aic))

    def scalar_scale(self) -> float:
        return max(0.2, min(5.0, SCALAR_ISSUE_RATE / max(1.0, float(self.scalar))))


A5_RATES = NativeRates()


def _field(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return obj.get(name) if (value is None and isinstance(obj, Mapping)) else value


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): item for key, item in value}
    except TypeError:
        return {}


def _per_second(mapping: Mapping[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in mapping.items():
        try:
            out[str(key)] = float(value) * 1000.0
        except (TypeError, ValueError):
            continue
    return out


@lru_cache(maxsize=64)
def _hardware_rates(sku: str) -> NativeRates | None:
    try:
        import importlib
        from mindie_sim.schema.hardware import Hardware
        ascend = importlib.import_module("mindie_sim.backends.ascend")
        from mindie_sim.prediction.piecewise.compiler import PipeSupply
        machine = Hardware.from_mapping(ascend.materialize({"sku": sku}))
        supply = PipeSupply.from_hardware(machine, dtype="bf16")
    except Exception:  # noqa: BLE001 - unknown SKU keeps the declared bound
        return None
    return _rates_from_pipe_supply(supply, arch=supply.arch_family or None, sku=sku)


def declared_clock_hz(arch: str, vector_rate: float, aiv_units: int) -> float:
    """Declared die clock derived from the die's AIV vector rate and owner count."""
    width = _VECTOR_BYTES_PER_CYCLE.get(str(arch).upper(), 64.0)
    if vector_rate <= 0 or aiv_units <= 0 or width <= 0:
        return _REFERENCE_CLOCK_HZ
    clock = float(vector_rate) / (float(aiv_units) * float(width))
    return clock if 0.2e9 <= clock <= 5.0e9 else _REFERENCE_CLOCK_HZ


def _extrapolated_floor(kernel_class: str) -> float:
    """Declared A2/A3 class baseline of one kernel class (ms)."""
    if kernel_class == "matrix_int8":
        return EXTRAPOLATED_FLOOR_MS["matrix_int8"]
    if kernel_class.startswith("matrix") or kernel_class == "transpose_batch":
        return EXTRAPOLATED_FLOOR_MS["cube"]
    return EXTRAPOLATED_FLOOR_MS["vector"]


def _class_floor(kernel_class: str, invocations: Mapping[str, Any]) -> float | None:
    primary = {"matrix_bf16": "cube", "matrix_quant": "w8a8_nd_bf16",
               "row_norm": "vector", "fused_row_norm": "vector",
               "elementwise": "vector", "quantize": "vector",
               "vector": "vector"}.get(kernel_class)
    if not primary:
        return None
    try:
        value = float(invocations.get(primary) or 0.0)
    except (TypeError, ValueError):
        return None
    if value <= 0 and primary == "w8a8_nd_bf16":
        value = float(invocations.get("cube") or 0.0)
    return value or None


def _rates_from_pipe_supply(supply: Any, *, arch: str | None, sku: str | None = None) -> NativeRates:
    del sku
    rates = _per_second(_as_mapping(_field(supply, "rates")))
    units = _as_mapping(_field(supply, "resource_units")) or _as_mapping(_field(supply, "units"))
    peaks = _as_mapping(_field(supply, "compute_peaks")) or _as_mapping(_field(supply, "peaks"))
    invocations = _as_mapping(_field(supply, "invocations"))
    buffers = _as_mapping(_field(supply, "buffers"))
    arch = str(arch or _field(supply, "arch") or _field(supply, "arch_family") or "A5")
    aic = int(units.get("mte2_aic") or units.get("compute_units") or _field(supply, "compute_units") or 32)
    aiv = int(units.get("vector") or units.get("mte2_aiv") or 64)
    ub = int(buffers.get("ub") or invocations.get("ub_bytes") or UB_BYTES)
    l2 = int(buffers.get("l2") or _field(supply, "l2_bytes") or L2_BYTES)
    cube = []
    for kind, keys in _DTYPE_PEAK_KEYS.items():
        peak = 0.0
        for key in keys:
            try:
                peak = max(peak, float(peaks.get(key) or 0.0) * 1000.0)
            except (TypeError, ValueError):
                continue
        cube.append((kind, peak * CUBE_SUSTAINED if peak else 0.0))
    measured = arch == "A5"
    vector_rate = float(rates.get("vector") or VECTOR_RATE)
    clock_hz = declared_clock_hz(arch, vector_rate, aiv)
    fp32_scale = (clock_hz / _REFERENCE_CLOCK_HZ) * (float(max(1, aic)) / 32.0)
    cube = [(kind, value or (A5_RATES.cube_rate("fp32") * fp32_scale
                             if kind == "fp32" and not measured else A5_RATES.cube_rate(kind)))
            for kind, value in cube]
    launch = {kc: float((_class_floor(kc, invocations) if measured else None)
                        or (A5_RATES.launch(kc) if measured
                            else _extrapolated_floor(kc) * (
                                SCALAR_ISSUE_RATE / max(1.0, float(
                                    rates.get("pipe_s_issue") or (
                                        SCALAR_ISSUE_RATE * (clock_hz / _REFERENCE_CLOCK_HZ))))
                                if kc.startswith("matrix") else 1.0)))
              for kc in LAUNCH_FLOOR_MS}
    scalar = float(rates.get("pipe_s_issue") or 0.0) or (SCALAR_ISSUE_RATE * (clock_hz / _REFERENCE_CLOCK_HZ))
    return NativeRates(
        arch=arch, aic=max(1, aic), aiv=max(1, aiv), ub_bytes=max(1, ub), l2_bytes=max(1, l2),
        hbm=float(rates.get("hbm") or rates.get("memory") or HBM_RATE),
        mte2_aic=float(rates.get("mte2_aic") or rates.get("mte2") or MTE2_RATE),
        mte2_aiv=float(rates.get("mte2_aiv") or rates.get("mte2") or MTE2_AIV_RATE),
        mte3=float(rates.get("mte3") or MTE3_RATE),
        mte1=float(rates.get("mte1") or rates.get("mte2_aic") or MTE2_RATE),
        vector=vector_rate, fixpipe=float(rates.get("fixpipe") or FIXPIPE_RATE),
        scalar=max(1.0, scalar), cube=tuple(cube),
        launch_ms=tuple(sorted(launch.items())), measured_floors=measured, clock_hz=clock_hz)


def rates_from_supply(supply: Any = None, hardware: Any = None,
                      *, default: NativeRates | None = None) -> NativeRates:
    """Resolve the declared die supply of one leaf/catalog evaluation."""
    if supply is not None:
        arch = _field(supply, "arch") or _field(supply, "arch_family")
        if _field(supply, "rates") is not None:
            return _rates_from_pipe_supply(supply, arch=arch)
    if isinstance(hardware, str) and hardware:
        resolved = _hardware_rates(hardware)
        if resolved is not None:
            return resolved
    elif hardware is not None and not isinstance(hardware, str):
        try:
            from mindie_sim.prediction.piecewise.compiler import PipeSupply
            pipe = PipeSupply.from_hardware(hardware, dtype="bf16")
            return _rates_from_pipe_supply(pipe, arch=pipe.arch_family or None)
        except Exception:  # noqa: BLE001 - keep the declared A5 reference
            pass
    return default or A5_RATES


def dtype_bytes(dtype: str | None, default: int = 2) -> int:
    return int(_DTYPE_BYTES.get(str(dtype or "").lower(), default))
