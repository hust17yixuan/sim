"""Triton-Ascend lowering, CV-serialism, gather-path, and MFU evaluator.

[Role]
Quantify Triton-Ascend lowering hazards on the shared OpFlow substrate: CV
scope serialization, the three random-gather-to-dot paths, scalar/i64 and
MLA-only BLOCK_QO hazards, chained fp32 accumulator spill, and the five-gate
MFU validity contract with cache-vs-HBM regime tagging.
[Invariants]
No hardware call is made. Serialism is a derived ratio, never clamped: the
overlap denominator can never fall below max(aic, aiv). FP32 spill penalties
stay floating-point microseconds. Cache residency is decided from the unique
working set only. A zero-FLOPs reference fails the Model gate; unknown feature
keys and non-finite values fail closed.
[Timeouts & Locks]
Pure in-memory arithmetic; no sleeps, locks, I/O, or subprocesses.
[Side-Effect Policy]
No side effects; results are immutable records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .hardware_profiles import HardwareProfile
from .exact_math import rational
from fractions import Fraction
from .triton_rules import GATHER_PATHS, MFU_GATES

MIB = 1024**2
_LAUNCH_US = 1.2
_FIELD_NAMES = frozenset({
    "has_cv_pingpong", "pipe_stages", "gm_intermediate_fallback",
    "chained_dot_fp32_acc_spill", "chained_buffer_bytes",
    "random_gather_to_dot_mode", "gather_bytes", "per_core_working_set_bytes",
    "has_i64_mask_or_noncontig_reduce", "mla_block_qo", "is_mla_kernel",
    "flops", "input_bytes", "max_input_bytes", "max_runtime_ms", "impl_runs",
    "model_ref_valid", "launch_us", "evidence",
})


class TritonLoweringError(ValueError):
    """Invalid Triton lowering feature declaration."""


def _finite(name: str, value: object, *, allow_zero: bool = True) -> float:
    if not isinstance(allow_zero, bool):
        raise TritonLoweringError("allow_zero must be bool")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TritonLoweringError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise TritonLoweringError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return number


def _int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TritonLoweringError(f"{name} must be an integer, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number != int(number) or int(number) < minimum:
        raise TritonLoweringError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(number)


@dataclass(frozen=True, slots=True)
class TritonFeatures:
    has_cv_pingpong: bool = True
    pipe_stages: int = 2
    gm_intermediate_fallback: bool = False
    chained_dot_fp32_acc_spill: bool = False
    chained_buffer_bytes: int = 0
    random_gather_to_dot_mode: str = "none"
    gather_bytes: int = 0
    per_core_working_set_bytes: int = 0
    has_i64_mask_or_noncontig_reduce: bool = False
    mla_block_qo: int = 1
    flops: float = 0.0
    input_bytes: int = 0
    max_input_bytes: int = 10_000_000_000
    max_runtime_ms: float = 4000.0
    impl_runs: bool = True
    model_ref_valid: bool = True
    launch_us: float = _LAUNCH_US
    evidence: str = "caller_assumed"
    is_mla_kernel: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipe_stages", _int("pipe_stages", self.pipe_stages, minimum=1))
        if self.random_gather_to_dot_mode not in GATHER_PATHS:
            raise TritonLoweringError(f"random_gather_to_dot_mode must be one of {GATHER_PATHS}")
        object.__setattr__(self, "mla_block_qo", _int("mla_block_qo", self.mla_block_qo, minimum=1))
        for name in ("gather_bytes", "per_core_working_set_bytes", "input_bytes",
                     "chained_buffer_bytes", "max_input_bytes"):
            object.__setattr__(self, name, _int(name, getattr(self, name)))
        for name in ("flops", "max_runtime_ms", "launch_us"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.max_runtime_ms <= 0:
            raise TritonLoweringError("max_runtime_ms must be positive")
        for name in ("has_cv_pingpong", "gm_intermediate_fallback",
                     "chained_dot_fp32_acc_spill", "has_i64_mask_or_noncontig_reduce",
                     "is_mla_kernel", "impl_runs", "model_ref_valid"):
            if not isinstance(getattr(self, name), bool):
                raise TritonLoweringError(f"{name} must be bool")
        if self.evidence not in ("caller_assumed", "empirically_measured", "reference_doc"):
            raise TritonLoweringError(f"unknown evidence grade: {self.evidence!r}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | "TritonFeatures" | None) -> "TritonFeatures":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise TritonLoweringError("triton features must be a mapping")
        unknown = set(payload) - _FIELD_NAMES
        if unknown:
            raise TritonLoweringError(f"unknown triton feature keys: {sorted(unknown)}")
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise TritonLoweringError(f"invalid triton feature mapping: {exc}") from exc


@dataclass(frozen=True, slots=True)
class MFUReport:
    effective_bw_gbps: float
    hbm_peak_bw_gbps: float
    achieved_tflops: float
    mfu: Fraction | None
    gates: Mapping[str, bool]
    gates_passed: int
    valid: bool
    cache_resident: bool
    regime: str
    nonmonotonic_scale_tag: str
    notes: tuple[str, ...] = ()
    gate_status: Mapping[str, str] = field(default_factory=dict)
    fully_empirical: bool = False

    def __post_init__(self) -> None:
        for name in ("valid", "cache_resident", "fully_empirical"):
            if not isinstance(getattr(self, name), bool):
                raise TritonLoweringError(f"{name} must be bool")
        if not isinstance(self.gates, Mapping) or any(not isinstance(v, bool) for v in self.gates.values()):
            raise TritonLoweringError("gates must map names to bool values")

    def as_dict(self) -> dict[str, Any]:
        return {
            "bw": round(self.effective_bw_gbps, 4), "hbm": round(self.hbm_peak_bw_gbps, 4),
            "tflops": round(self.achieved_tflops, 6), "mfu": None if self.mfu is None else float(round(self.mfu, 6)),
            "gates": dict(sorted(self.gates.items())), "valid": self.valid,
            "regime": self.regime, "empirical": self.fully_empirical,
        }


@dataclass(frozen=True, slots=True)
class TritonLoweringResult:
    aic_us: float
    aiv_us: float
    aiv_degraded_us: float
    cv_wall_us: float
    t_overlapped_us: float
    serialism_ratio: float
    cv_serialism: bool
    serial_penalty_us: float
    gather_path: str
    gather_us: float
    workspace_round_trip_bytes: int
    gather_l2_resident: bool
    vector_degradation_factor: float
    scalar_hazard: bool
    mla_block_qo: int
    nonaffine_store_hazard: bool
    block_qo_penalty_us: float
    fp32_acc_spill: bool
    spill_penalty_us: float
    total_penalty_us: float
    effective_service_us: float
    mfu: MFUReport
    warnings: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("cv_serialism", "gather_l2_resident", "scalar_hazard",
                     "nonaffine_store_hazard", "fp32_acc_spill"):
            if not isinstance(getattr(self, name), bool):
                raise TritonLoweringError(f"{name} must be bool")

    @property
    def t_aic_us(self) -> float:
        return self.aic_us

    @property
    def t_aiv_us(self) -> float:
        return self.aiv_degraded_us

    @property
    def observed_overlap_ratio(self) -> float:
        return self.extra["observed_overlap_ratio"]

    @property
    def modeled_overlap_ratio(self) -> float:
        return (self.aic_us + self.aiv_degraded_us) / max(self.t_overlapped_us, 1e-12)

    @property
    def potential_overlap_gain(self) -> float:
        return (self.aic_us + self.aiv_degraded_us) / max(self.aic_us, self.aiv_degraded_us, 1e-12)

    def as_dict(self) -> dict[str, Any]:
        return {
            "serialism": round(self.serialism_ratio, 4),
            "observed_overlap": round(self.observed_overlap_ratio, 4),
            "potential_gain": round(self.potential_overlap_gain, 4),
            "cv_serial": self.cv_serialism, "gather": self.gather_path,
            "gather_us": round(self.gather_us, 8),
            "workspace_bytes": self.workspace_round_trip_bytes,
            "scalar_factor": round(self.vector_degradation_factor, 3),
            "block_qo": self.mla_block_qo, "nonaffine": self.nonaffine_store_hazard,
            "fp32_spill": self.fp32_acc_spill,
            "penalty_us": round(self.total_penalty_us, 8),
            "service_us": round(self.effective_service_us, 8),
            "mfu": self.mfu.as_dict(),
        }


def _bytes_per_second(profile: HardwareProfile, working_set: int) -> float:
    l2 = float(profile.memory.l2)
    low = float(profile.rates.l2_bytes_s)
    high = float(profile.rates.hbm_bytes_s)
    if working_set <= l2:
        return low
    if working_set >= 2.0 * l2:
        return high
    return max(high, low + (high - low) * (working_set - l2) / l2)


def classify_cache_regime(working_set_bytes: int, l2_bytes: int) -> str:
    if isinstance(working_set_bytes, bool) or not isinstance(working_set_bytes, int):
        raise TritonLoweringError("working_set_bytes must be an integer")
    if working_set_bytes < 0 or l2_bytes <= 0:
        raise TritonLoweringError("working set must be non-negative and L2 positive")
    return "cache_resident_small_shape" if working_set_bytes <= l2_bytes else "hbm_resident_scaled_shape"


def evaluate_mfu_gates(profile: HardwareProfile, *, wall_us: float,
                       features: TritonFeatures, working_set_bytes: int,
                       total_bytes: int | None = None, dtype: str = "bf16") -> MFUReport:
    """Apply the five MFU gates on one consistent whole-op byte/flop scope."""
    if isinstance(wall_us, bool) or not isinstance(wall_us, (int, float)):
        raise TritonLoweringError("wall_us must be numeric")
    wall_us = _finite("wall_us", wall_us, allow_zero=False)
    if isinstance(working_set_bytes, bool) or not isinstance(working_set_bytes, int) or working_set_bytes < 0:
        raise TritonLoweringError("working_set_bytes must be a non-negative integer")
    moved = features.input_bytes if total_bytes is None else _int("total_bytes", total_bytes)
    effective_bw = moved / (wall_us * 1e-6) / 1e9
    achieved = features.flops / (wall_us * 1e-6) / 1e12
    peak = profile.cube_peak_tflops(dtype)
    raw_mfu = (rational(features.flops)*10**6 / rational(wall_us) / (rational(peak)*10**12)
               if features.flops > 0 and peak > 0 else None)
    mfu = raw_mfu if raw_mfu is not None and raw_mfu <= 1 else None
    l2 = profile.memory.l2
    regime = classify_cache_regime(working_set_bytes, l2)
    cache_resident = regime == "cache_resident_small_shape"
    limit = profile.l2_bw_gbps if cache_resident else profile.hbm_bw_gbps
    gates = {
        "input_budget_ok": features.input_bytes <= features.max_input_bytes,
        "model_ref_valid": features.model_ref_valid and mfu is not None,
        "implementation_runs": features.impl_runs,
        "single_run_time_budget_ok": wall_us <= features.max_runtime_ms * 1000.0,
        "effective_bw_within_hbm": effective_bw <= limit,
    }
    status = {
        "input_budget_ok": "computed",
        "model_ref_valid": "empirically_measured" if features.evidence == "empirically_measured" else "caller_assumed",
        "implementation_runs": "empirically_measured" if features.evidence == "empirically_measured" else "caller_assumed",
        "single_run_time_budget_ok": "computed",
        "effective_bw_within_hbm": "computed",
    }
    notes: list[str] = []
    if raw_mfu is not None and raw_mfu > 1:
        notes.append("inconsistent FLOP/time scope: implied Cube MFU exceeds one")
    if features.flops <= 0:
        notes.append("reference Model must declare positive FLOPs")
    if not gates["input_budget_ok"]:
        notes.append("input_bytes exceeds the configured budget")
    if not gates["effective_bw_within_hbm"]:
        notes.append(f"effective bandwidth exceeds the {regime} service limit ({limit:.1f} GB/s)")
    if regime == "hbm_resident_scaled_shape":
        notes.append("scaled shape is HBM-resident; small-shape cache MFU is not comparable")
    return MFUReport(
        effective_bw_gbps=effective_bw, hbm_peak_bw_gbps=profile.hbm_bw_gbps,
        achieved_tflops=achieved, mfu=mfu, gates=gates,
        gates_passed=sum(1 for name in MFU_GATES if gates[name]),
        valid=all(gates[name] for name in MFU_GATES),
        cache_resident=cache_resident, regime=regime,
        nonmonotonic_scale_tag=regime, notes=tuple(notes), gate_status=status,
        fully_empirical=all(value == "empirically_measured" for value in status.values()),
    )


def analyze_triton_lowering(profile: HardwareProfile, *, aic_us: float, aiv_us: float,
                            gm_us: float = 0.0,
                            features: TritonFeatures | Mapping[str, Any] | None = None,
                            working_set_bytes: int = 0,
                            per_core_working_set_bytes: int | None = None,
                            total_bytes: int | None = None, items: int = 1,
                            wall_us_hint: float | None = None, dtype: str = "bf16") -> TritonLoweringResult:
    """Compute deterministic lowering penalties on one consistent scope."""
    spec = TritonFeatures.from_mapping(features)
    aic_us = _finite("aic_us", aic_us)
    aiv_us = _finite("aiv_us", aiv_us)
    gm_us = _finite("gm_us", gm_us)
    items = _int("items", items, minimum=1)
    warnings: list[str] = []
    factor = 3.3 if spec.has_i64_mask_or_noncontig_reduce else 1.0
    if factor > 1.0:
        warnings.append("i64_mask_and_noncontig_reduce_scalar_lowering")
    aiv_degraded = aiv_us * factor
    scalar_penalty = aiv_degraded - aiv_us
    cv_serial = (aic_us > 0 and aiv_degraded > 0) and (
        not spec.has_cv_pingpong or spec.gm_intermediate_fallback or spec.pipe_stages < 2
    )
    if aic_us > 0 and aiv_degraded > 0:
        peak = max(aic_us, aiv_degraded)
        if cv_serial:
            t_overlapped = aic_us + aiv_degraded
            cv_wall = aic_us + aiv_degraded
            warnings.append("cv_scope_serialized_no_pingpong_or_gm_fallback")
        else:
            t_overlapped = peak + 0.05 * min(aic_us, aiv_degraded)
            cv_wall = t_overlapped
        serialism = (aic_us + aiv_degraded) / max(t_overlapped, 1e-12)
    else:
        t_overlapped = max(aic_us, aiv_degraded)
        cv_wall = t_overlapped
        serialism = 1.0
    serial_penalty = cv_wall - max(aic_us, aiv_degraded)

    unique = max(int(working_set_bytes), int(per_core_working_set_bytes or 0))
    read_bw = _bytes_per_second(profile, unique)
    gather_l2 = unique <= profile.memory.l2
    gather_bytes = spec.gather_bytes
    gather_us = 0.0
    workspace_round_trip = 0
    if spec.random_gather_to_dot_mode == "gather_out_to_ub":
        gather_us = gather_bytes / read_bw * 1e6 + 0.35
        warnings.append("random_gather_gather_out_to_ub")
    elif spec.random_gather_to_dot_mode == "in_kernel_l2_stage":
        per_core = per_core_working_set_bytes if per_core_working_set_bytes is not None else unique
        stage_bw = profile.rates.l2_bytes_s if per_core <= profile.memory.l2 else profile.rates.hbm_bytes_s
        gather_us = gather_bytes / stage_bw * 1e6 + 0.20
        if per_core > profile.memory.l2:
            warnings.append("in_kernel_stage_exceeds_l2_falls_back_to_hbm")
        warnings.append("random_gather_in_kernel_l2_stage")
    elif spec.random_gather_to_dot_mode == "device_gather_workspace":
        workspace_round_trip = 2 * gather_bytes
        bw = profile.rates.hbm_bytes_s if unique > profile.memory.l2 else profile.rates.l2_bytes_s
        gather_us = workspace_round_trip / bw * 1e6 + spec.launch_us
        warnings.append("random_gather_two_kernel_workspace_round_trip")
    elif spec.random_gather_to_dot_mode != "none":
        raise TritonLoweringError(f"unknown gather path: {spec.random_gather_to_dot_mode!r}")

    spill_penalty = 0.0
    if spec.chained_dot_fp32_acc_spill:
        chain_bytes = spec.chained_buffer_bytes or max(gather_bytes, unique // 2)
        bw = profile.rates.hbm_bytes_s if unique > profile.memory.l2 else profile.rates.l2_bytes_s
        spill_penalty = 2 * chain_bytes / bw * 1e6
        warnings.append("chained_dot_fp32_accumulator_spill")

    nonaffine = spec.is_mla_kernel and spec.mla_block_qo != 1
    block_qo_penalty = 0.0
    if nonaffine:
        block_qo_penalty = 0.6 * max(aiv_degraded, gm_us) + 0.5
        warnings.append("mla_block_qo_nonaffine_store_hazard")
    elif spec.mla_block_qo != 1:
        warnings.append("block_qo_ignored_outside_mla_scope")

    total_penalty = serial_penalty + scalar_penalty + gather_us + spill_penalty + block_qo_penalty
    service = max(cv_wall, gm_us) + gather_us + spill_penalty + block_qo_penalty
    if wall_us_hint is not None:
        wall = _finite("wall_us_hint", wall_us_hint, allow_zero=False)
    else:
        # Only per-item compute repeats; gather/spill/workspace and BLOCK_QO
        # are call-level costs and must not be charged once per item.
        call_level = gather_us + spill_penalty + block_qo_penalty
        per_item = max(cv_wall, gm_us)
        wall = call_level + per_item * items
    moved = _int("total_bytes", total_bytes) if total_bytes is not None else (gather_bytes or spec.input_bytes)
    mfu = evaluate_mfu_gates(
        profile, wall_us=max(wall, 1e-6),
        features=replace(spec, input_bytes=spec.input_bytes or moved),
        working_set_bytes=unique, total_bytes=moved, dtype=dtype,
    )
    return TritonLoweringResult(
        aic_us=aic_us, aiv_us=aiv_us, aiv_degraded_us=aiv_degraded,
        cv_wall_us=cv_wall, t_overlapped_us=t_overlapped,
        serialism_ratio=serialism, cv_serialism=cv_serial,
        serial_penalty_us=serial_penalty, gather_path=spec.random_gather_to_dot_mode,
        gather_us=gather_us, workspace_round_trip_bytes=workspace_round_trip,
        gather_l2_resident=gather_l2, vector_degradation_factor=factor,
        scalar_hazard=factor > 1.0, mla_block_qo=spec.mla_block_qo,
        nonaffine_store_hazard=nonaffine, block_qo_penalty_us=block_qo_penalty,
        fp32_acc_spill=spec.chained_dot_fp32_acc_spill,
        spill_penalty_us=spill_penalty, total_penalty_us=total_penalty,
        effective_service_us=service, mfu=mfu,
        warnings=tuple(sorted(set(warnings))),
        extra={
            "serialism_threshold": 1.5, "serialism_exceeded": serialism > 1.5,
            "observed_overlap_ratio": (aic_us + aiv_us) * items / max(wall, 1e-12),
            "modeled_overlap_ratio": (aic_us + aiv_degraded) / max(t_overlapped, 1e-12),
            "wall_us_hint": wall_us_hint,
            "potential_overlap_gain": (aic_us + aiv_degraded) / max(aic_us, aiv_degraded, 1e-12),
            "gather_paths": GATHER_PATHS, "mfu_gates": MFU_GATES,
        },
    )


__all__ = [
    "MFUReport", "TritonFeatures", "TritonLoweringError", "TritonLoweringResult",
    "analyze_triton_lowering", "classify_cache_regime", "evaluate_mfu_gates",
]
