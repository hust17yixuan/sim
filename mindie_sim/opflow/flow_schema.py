"""Declarative multi-pipe operator-flow schema and Fixpipe selection.

[Role]
Define immutable operator-flow records shared by the catalog, transfer
accounting, pipeline evaluator, and MCP wrappers, including per-item stage
dependencies, 1C:2V lane ownership, dualDstCtl, and recurrence-step scope.
[Invariants]
Every count/size is finite and type-checked; bool is never accepted as int.
Vector work is conserved globally and partitioned explicitly across lanes.
L0C->UB requires DAV_3510 plus AIC issue ownership and a valid dualDstCtl.
Unknown dtype/backend/archetype/mode values fail closed with typed errors.
[Timeouts & Locks]
Pure validation only; no locks, waits, I/O, or timeout paths.
[Side-Effect Policy]
No side effects. Functions return records or raise typed schema errors.
"""

from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .hardware_profiles import HardwareProfile, normalize_dtype


class FlowSchemaError(ValueError):
    """Invalid flow declaration."""


class FixpipeError(FlowSchemaError):
    """Fixpipe path is unavailable or misconfigured."""


class ExecutionArchetype(str, Enum):
    PURE_VECTOR = "PURE_VECTOR"
    PURE_CUBE = "PURE_CUBE"
    CV_COUPLED_1C2V = "CV_COUPLED_1C2V"
    COMMUNICATION_HYBRID = "COMMUNICATION_HYBRID"


class ExecutionMode(str, Enum):
    EAGER = "eager"
    ACLGRAPH_PIECEWISE = "aclgraph_piecewise"
    ACLGRAPH_FULL = "aclgraph_full"
    ACLGRAPH_SUPERKERNEL = "aclgraph_superkernel"


class AuthoringBackend(str, Enum):
    ASCENDC = "ascendc"
    CANNBOT_DSL = "cannbot_dsl"
    TRITON_ASCEND = "triton_ascend"


class VectorMode(str, Enum):
    SIMD = "simd"
    SIMT = "simt"


def require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FlowSchemaError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def require_positive_float(name: str, value: Any, *, allow_zero: bool = True) -> float:
    if not isinstance(allow_zero, bool):
        raise FlowSchemaError("allow_zero must be bool")
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise FlowSchemaError(f"{name} must be a finite number, got {value!r}")
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (value == 0 and not allow_zero):
        raise FlowSchemaError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return value


@dataclass(frozen=True, slots=True)
class LoopCarriedRecurrence:
    """Stateful recurrence bound: tau_steady >= latency / distance_items."""

    latency_us: float = 0.0
    cycles: float = 0.0
    distance_items: int = 1
    state_name: str = ""
    step_latencies_us: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_us", require_positive_float("recurrence latency", self.latency_us))
        object.__setattr__(self, "cycles", require_positive_float("recurrence cycles", self.cycles))
        distance = require_int("recurrence_distance_items", self.distance_items, minimum=1)
        object.__setattr__(self, "distance_items", distance)
        if self.latency_us <= 0 and self.cycles <= 0 and not self.step_latencies_us:
            raise FlowSchemaError("recurrence needs latency_us, cycles, or step_latencies_us")
        steps = tuple(require_positive_float("step latency", value) for value in self.step_latencies_us)
        if any(value <= 0 for value in steps):
            raise FlowSchemaError("step latencies must be positive")
        object.__setattr__(self, "step_latencies_us", steps)

    def latency_us_at(self, freq_mhz: int) -> float:
        cycle_us = self.cycles / float(freq_mhz)
        return max(float(self.latency_us), cycle_us)

    @property
    def bound_us(self) -> float:
        """max_cycle(sum_latency / sum_distance) with distance >= 1."""
        if self.step_latencies_us:
            return sum(self.step_latencies_us) / self.distance_items
        return self.latency_us / self.distance_items


class FixpipePath(str, Enum):
    L0C_TO_GM = "L0C_TO_GM"
    L0C_TO_L1 = "L0C_TO_L1"
    L0C_TO_UB = "L0C_TO_UB"


class BufferPool(str, Enum):
    UB_LANE0 = "UB_LANE0"
    UB_LANE1 = "UB_LANE1"
    L1 = "L1"
    L0A = "L0A"
    L0B = "L0B"
    L0C = "L0C"
    GM = "GM"
    SSBUF = "SSBUF"
    BIAS = "BIAS"


PIPE_NAMES = (
    "MTE2_AIC", "MTE1", "CUBE", "FIX", "MTE2_AIV", "VEC",
    "VEC_TRANS", "SIMT", "MTE3",
)
_STAGE_CHAINS = {
    ExecutionArchetype.PURE_CUBE: (("MTE2_AIC", "MTE1", "CUBE", "FIX"),),
    ExecutionArchetype.PURE_VECTOR: (("MTE2_AIV", "VEC", "MTE3"),),
    ExecutionArchetype.CV_COUPLED_1C2V: (
        ("MTE2_AIC", "MTE1", "CUBE", "FIX"),
        ("MTE2_AIV", "VEC", "MTE3"),
    ),
    ExecutionArchetype.COMMUNICATION_HYBRID: (("MTE2_AIV", "VEC", "MTE3"),),
}


@dataclass(frozen=True, slots=True)
class PipeWork:
    """Total work assigned to one overlappable pipeline over the whole op."""

    pipe: str
    total_work: float
    work_kind: str = "ops"

    def __post_init__(self) -> None:
        if self.pipe not in PIPE_NAMES:
            raise FlowSchemaError(f"unknown pipe: {self.pipe!r}")
        object.__setattr__(self, "total_work", require_positive_float("pipe work", self.total_work))
        if self.work_kind not in ("ops", "macs", "bytes"):
            raise FlowSchemaError(f"unknown work_kind: {self.work_kind!r}")


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """Per-tile buffer residency and optional sequential-stage slot sharing."""

    name: str
    pool: BufferPool
    bytes_per_tile: int
    slots: int = 1
    resident: bool = False
    first_stage: int = 0
    last_stage: int = 0
    owner: int = 0
    alloc_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise FlowSchemaError("buffer name is required")
        try:
            object.__setattr__(self, "pool", self.pool if isinstance(self.pool, BufferPool) else BufferPool(str(self.pool)))
        except ValueError as exc:
            raise FlowSchemaError(f"unknown buffer pool: {self.pool!r}") from exc
        object.__setattr__(self, "bytes_per_tile", require_int("buffer bytes", self.bytes_per_tile))
        object.__setattr__(self, "slots", require_int("buffer slots", self.slots, minimum=1))
        object.__setattr__(self, "first_stage", require_int("first_stage", self.first_stage))
        object.__setattr__(self, "last_stage", require_int("last_stage", self.last_stage))
        object.__setattr__(self, "owner", require_int("owner", self.owner))
        if self.alloc_id is not None and (not isinstance(self.alloc_id, str) or not self.alloc_id):
            raise FlowSchemaError("alloc_id must be a nonempty string or None")
        if self.first_stage > self.last_stage:
            raise FlowSchemaError("invalid buffer stage lifetime")
        if not isinstance(self.resident, bool):
            raise FlowSchemaError("resident must be bool")

    @property
    def lane(self) -> int | None:
        if self.pool is BufferPool.UB_LANE0:
            return 0
        if self.pool is BufferPool.UB_LANE1:
            return 1
        return None


@dataclass(frozen=True, slots=True)
class OwnerWork:
    """Identical independent owners, each with its own recurrent Vector work."""

    vector_ops: int
    read_bytes: int
    write_bytes: int
    steps: int
    multiplicity: int = 1

    def __post_init__(self):
        for name in ("vector_ops", "read_bytes", "write_bytes", "steps", "multiplicity"):
            require_int(name, getattr(self, name), minimum=1 if name == "multiplicity" else 0)
        if self.steps == 0 and self.vector_ops:
            raise FlowSchemaError("zero-step owner cannot execute Vector work")


@dataclass(frozen=True, slots=True)
class OpFlowTemplate:
    """Concrete parameterized flow, ready for tiling and timing."""

    name: str
    archetype: ExecutionArchetype
    total_units: float
    base_tile_units: float
    live_bytes_per_unit: float
    pipe_work: tuple[PipeWork, ...]
    gm_read_bytes: int = 0
    gm_write_bytes: int = 0
    unique_gm_bytes: int = 0
    sync_pairs: int = 0
    fixpipe_path: FixpipePath = FixpipePath.L0C_TO_GM
    fixpipe_sub_block: int | None = None
    tiling_pool: BufferPool = BufferPool.UB_LANE0
    buffers: tuple[BufferSpec, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""
    authoring_backend: AuthoringBackend = AuthoringBackend.ASCENDC
    recurrence: LoopCarriedRecurrence | None = None
    triton_features: Mapping[str, Any] = field(default_factory=dict)
    recurrence_latency_us: float = 0.0
    recurrence_cycles: float = 0.0
    recurrence_distance_items: int = 1
    recurrence_steps: int = 1
    dtype: str = "bf16"
    vector_mode: str = VectorMode.SIMD.value
    stage_order: tuple[tuple[str, ...], ...] = ()
    dual_dst_ctl: int | None = None
    fixpipe_calls: int = 1
    fixpipe_m: int = 0
    fixpipe_n: int = 0
    quant_mode: str = "NoQuant"
    relu_enabled: bool = False
    lanes_receiving: tuple[int, ...] = (0,)
    vector_ops_by_lane: tuple[float, float] | None = None
    estimate_kind: str = "analytic_prior"
    calibrated: bool = False
    no_launch: bool = False
    spill_policy: str = "independent_retile"
    spill_workspace_bytes: int = 0
    weight_bytes_read_once: int = 0
    compute_owner: str = "AIC"
    pipeline_slots: int | None = None
    schedule_policy: str = "independent_service_bound"
    fixpipe_layout: str = "ND"
    fixpipe_column_major: bool = False
    owner_work: tuple[OwnerWork, ...] = ()

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "archetype", self.archetype if isinstance(self.archetype, ExecutionArchetype) else ExecutionArchetype(str(self.archetype)))
            object.__setattr__(self, "fixpipe_path", self.fixpipe_path if isinstance(self.fixpipe_path, FixpipePath) else FixpipePath(str(self.fixpipe_path)))
            object.__setattr__(self, "tiling_pool", self.tiling_pool if isinstance(self.tiling_pool, BufferPool) else BufferPool(str(self.tiling_pool)))
        except ValueError as exc:
            raise FlowSchemaError(f"invalid enum value: {exc}") from exc
        for flag in ("calibrated", "no_launch", "relu_enabled"):
            if not isinstance(getattr(self, flag), bool):
                raise FlowSchemaError(f"{flag} must be bool")
        if self.calibrated and self.estimate_kind in ("structural_schema_only", "conditional_ledger_required", "analytic_prior"):
            raise FlowSchemaError("an unverified prior cannot claim calibration")
        if not str(self.name).strip():
            raise FlowSchemaError("flow name is required")
        if isinstance(self.total_units, bool):
            raise FlowSchemaError("total_units must not be bool")
        object.__setattr__(self, "total_units", require_positive_float("total_units", self.total_units))
        if self.total_units <= 0 and not self.no_launch:
            raise FlowSchemaError("total_units must be positive unless no_launch")
        object.__setattr__(self, "base_tile_units", require_positive_float("base_tile_units", self.base_tile_units, allow_zero=False))
        object.__setattr__(self, "live_bytes_per_unit", require_positive_float("live_bytes_per_unit", self.live_bytes_per_unit))
        for name in ("gm_read_bytes", "gm_write_bytes", "unique_gm_bytes", "sync_pairs",
                     "spill_workspace_bytes", "weight_bytes_read_once", "fixpipe_calls"):
            object.__setattr__(self, name, require_int(name, getattr(self, name)))
        if self.sync_pairs < 0:
            raise FlowSchemaError("sync_pairs must be non-negative")
        object.__setattr__(self, "recurrence_steps", require_int("recurrence_steps", self.recurrence_steps, minimum=1))
        object.__setattr__(self, "recurrence_latency_us",
                           require_positive_float("recurrence_latency_us", self.recurrence_latency_us))
        object.__setattr__(self, "recurrence_cycles",
                           require_positive_float("recurrence_cycles", self.recurrence_cycles))
        object.__setattr__(self, "recurrence_distance_items",
                           require_int("recurrence_distance_items", self.recurrence_distance_items, minimum=1))
        try:
            object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        except ValueError as exc:
            raise FlowSchemaError(str(exc)) from exc
        if self.vector_mode not in (VectorMode.SIMD.value, VectorMode.SIMT.value):
            raise FlowSchemaError(f"unknown vector_mode: {self.vector_mode!r}")
        if not isinstance(self.authoring_backend, AuthoringBackend):
            try:
                object.__setattr__(self, "authoring_backend", AuthoringBackend(str(self.authoring_backend)))
            except ValueError as exc:
                raise FlowSchemaError(f"unknown authoring_backend: {self.authoring_backend!r}") from exc
        pipes = {item.pipe for item in self.pipe_work}
        if self.owner_work:
            if self.archetype is not ExecutionArchetype.PURE_VECTOR or pipes - {"VEC"}:
                raise FlowSchemaError("owner ledger currently requires a pure Vector flow")
            if not all(isinstance(x, OwnerWork) for x in self.owner_work):
                raise FlowSchemaError("owner_work requires OwnerWork records")
            for name, expected in (("read_bytes", self.gm_read_bytes), ("write_bytes", self.gm_write_bytes),
                                   ("vector_ops", sum(x.total_work for x in self.pipe_work))):
                if sum(getattr(x, name) * x.multiplicity for x in self.owner_work) != expected:
                    raise FlowSchemaError(f"owner ledger does not conserve {name}")
            if sum(x.multiplicity for x in self.owner_work) != self.total_units or self.base_tile_units != 1:
                raise FlowSchemaError("owner ledger requires one tile per independent owner")
        if self.archetype is ExecutionArchetype.PURE_VECTOR and pipes & {"CUBE", "FIX", "MTE1"}:
            raise FlowSchemaError("PURE_VECTOR may not declare Cube/Fixpipe work")
        if self.archetype is ExecutionArchetype.PURE_CUBE and pipes & {"VEC", "VEC_TRANS", "SIMT"}:
            raise FlowSchemaError("PURE_CUBE may not declare Vector work")
        if self.archetype is ExecutionArchetype.CV_COUPLED_1C2V:
            if "CUBE" not in pipes or not (pipes & {"VEC", "VEC_TRANS", "SIMT"}):
                raise FlowSchemaError("CV_COUPLED_1C2V needs both CUBE and Vector work")
        if isinstance(self.fixpipe_sub_block, bool) or self.fixpipe_sub_block not in (None, 0, 1):
            raise FlowSchemaError("fixpipe_sub_block must be None, 0, or 1")
        if isinstance(self.dual_dst_ctl, bool) or self.dual_dst_ctl not in (None, 0, 1, 2):
            raise FlowSchemaError("dual_dst_ctl must be None, 0, 1, or 2")
        if self.pipeline_slots is not None:
            require_int("pipeline_slots", self.pipeline_slots, minimum=1)
        if self.schedule_policy not in ("independent_service_bound", "barrier_waves"):
            raise FlowSchemaError("unknown schedule policy")
        if self.fixpipe_layout not in ("ND", "NZ", "ZN") or not isinstance(self.fixpipe_column_major, bool):
            raise FlowSchemaError("invalid Fixpipe layout")
        for name in ("fixpipe_m", "fixpipe_n"):
            require_int(name, getattr(self, name))
        if self.fixpipe_calls < 1:
            raise FlowSchemaError("fixpipe_calls must be >= 1")
        lanes = tuple(sorted(set(self.lanes_receiving)))
        if not lanes or any(type(lane) is not int or lane not in (0, 1) for lane in lanes):
            raise FlowSchemaError("lanes_receiving must be a non-empty subset of (0, 1)")
        object.__setattr__(self, "lanes_receiving", lanes)
        if self.quant_mode not in ("NoQuant", "F322F16", "F322BF16"):
            raise FlowSchemaError(f"unknown Fixpipe quant mode: {self.quant_mode!r}")
        if self.estimate_kind not in ("source_faithful", "structural_schema_only", "conditional_ledger_required", "reference_algorithm", "analytic_prior"):
            raise FlowSchemaError(f"unknown estimate_kind: {self.estimate_kind!r}")
        if self.spill_policy not in ("independent_retile", "algorithm_pass_workspace"):
            raise FlowSchemaError(f"unknown spill_policy: {self.spill_policy!r}")
        chains = tuple(tuple(chain) for chain in self.stage_order) or _STAGE_CHAINS[self.archetype]
        if "SIMT" in pipes:
            if self.archetype is ExecutionArchetype.CV_COUPLED_1C2V:
                chains = (chains[0], ("MTE2_AIV", "VEC", "SIMT", "MTE3")
                          if any(pipe in pipes for pipe in ("VEC", "VEC_TRANS"))
                          else ("MTE2_AIV", "SIMT", "MTE3"))
            elif any(pipe in pipes for pipe in ("VEC", "VEC_TRANS")):
                chains = (("MTE2_AIV", "VEC", "SIMT", "MTE3"),)
            else:
                chains = (("MTE2_AIV", "SIMT", "MTE3"),)
        for chain in chains:
            if not chain or any(pipe not in PIPE_NAMES for pipe in chain):
                raise FlowSchemaError(f"invalid stage chain: {chain!r}")
        object.__setattr__(self, "stage_order", chains)
        if self.vector_ops_by_lane is not None:
            if len(self.vector_ops_by_lane) != 2:
                raise FlowSchemaError("vector_ops_by_lane must have two entries")
            values = tuple(require_positive_float("lane vector work", value) for value in self.vector_ops_by_lane)
            total = sum(work.total_work for work in self.pipe_work if work.pipe in ("VEC", "VEC_TRANS"))
            if sum(values) != total:
                raise FlowSchemaError("vector work must be conserved across lanes")
            if self.fixpipe_path is FixpipePath.L0C_TO_UB:
                if any(work > 0 and lane not in lanes for lane, work in enumerate(values)):
                    raise FlowSchemaError("Vector work assigned to a non-receiving FIX lane")
            object.__setattr__(self, "vector_ops_by_lane", values)
        if self.recurrence is None and (self.recurrence_latency_us > 0 or self.recurrence_cycles > 0):
            object.__setattr__(self, "recurrence", LoopCarriedRecurrence(
                latency_us=float(self.recurrence_latency_us),
                cycles=float(self.recurrence_cycles),
                distance_items=int(self.recurrence_distance_items),
            ))
        if self.recurrence is not None and not isinstance(self.recurrence, LoopCarriedRecurrence):
            raise FlowSchemaError("recurrence must be a LoopCarriedRecurrence")
        if self.authoring_backend is AuthoringBackend.TRITON_ASCEND:
            mode = str(self.triton_features.get("random_gather_to_dot_mode", "none"))
            if mode not in ("none", "gather_out_to_ub", "in_kernel_l2_stage", "device_gather_workspace"):
                raise FlowSchemaError(f"unknown random gather lowering path: {mode!r}")


@dataclass(frozen=True, slots=True)
class FixpipeConfig:
    path: FixpipePath
    sub_block_id: int | None = None
    is_to_ub: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.is_to_ub, bool):
            raise FixpipeError("is_to_ub must be bool")
        try:
            object.__setattr__(self, "path", self.path if isinstance(self.path, FixpipePath) else FixpipePath(str(self.path)))
        except ValueError as exc:
            raise FixpipeError(f"unknown Fixpipe path: {self.path!r}") from exc
        if isinstance(self.sub_block_id, bool) or self.sub_block_id not in (None, 0, 1):
            raise FixpipeError("subBlockId must be 0, 1, or None")
        if self.is_to_ub != (self.path is FixpipePath.L0C_TO_UB):
            raise FixpipeError("is_to_ub must agree with L0C_TO_UB")


@dataclass(frozen=True, slots=True)
class FixpipeSelection:
    path: FixpipePath
    lane: int | None
    is_to_ub: bool
    allowed: bool
    reason: str

    def __post_init__(self) -> None:
        for name in ("is_to_ub", "allowed"):
            if not isinstance(getattr(self, name), bool):
                raise FixpipeError(f"{name} must be bool")


def select_fixpipe(profile: HardwareProfile, path: FixpipePath | str,
                   sub_block_id: int | None = None) -> FixpipeSelection:
    try:
        chosen = path if isinstance(path, FixpipePath) else FixpipePath(str(path))
    except ValueError as exc:
        raise FixpipeError(f"unknown Fixpipe path: {path!r}") from exc
    lane = sub_block_id
    if isinstance(lane, bool) or lane not in (None, 0, 1):
        raise FixpipeError("subBlockId must be 0, 1, or None")
    if chosen is FixpipePath.L0C_TO_UB:
        if lane is None:
            lane = 0
        if not profile.supports_fixpipe_l0c_to_ub():
            raise FixpipeError(
                f"{profile.family}/{profile.arch} has no L0C->UB Fixpipe; "
                "isToUB is silently dropped on V220, so this flow fails closed"
            )
        return FixpipeSelection(chosen, lane, True, True, "DAV_3510 L0C->UB lane selected")
    if chosen is FixpipePath.L0C_TO_L1:
        return FixpipeSelection(chosen, None, False, True, "L0C->L1 available")
    return FixpipeSelection(chosen, None, False, True, "L0C->GM available")


def check_flow_fixpipe(flow: OpFlowTemplate, profile: HardwareProfile) -> FixpipeSelection:
    """Validate Fixpipe for every archetype, including PURE_CUBE."""
    if flow.fixpipe_path is not FixpipePath.L0C_TO_UB:
        return select_fixpipe(profile, flow.fixpipe_path, None)
    if flow.archetype is ExecutionArchetype.PURE_VECTOR:
        raise FixpipeError("PURE_VECTOR has no Cube/Fixpipe path")
    selection = select_fixpipe(profile, flow.fixpipe_path, flow.fixpipe_sub_block)
    if flow.compute_owner != "AIC":
        raise FixpipeError("L0C->UB must be issued by an AIC owner")
    if flow.dual_dst_ctl in (None, 0) and len(flow.lanes_receiving) == 1:
        if flow.lanes_receiving[0] != selection.lane:
            raise FixpipeError("single FIX destination must equal the selected receiving lane")
    if flow.dual_dst_ctl in (1, 2):
        if flow.lanes_receiving != (0, 1):
            raise FixpipeError("split-M/N requires both receiving lanes")
        if flow.fixpipe_layout != "ND" or flow.fixpipe_column_major:
            raise FixpipeError("split-M/N requires row-major ND output")
    if flow.dual_dst_ctl == 1:
        if flow.fixpipe_m % 2 or flow.fixpipe_m <= 0:
            raise FixpipeError("dualDstCtl=1 split-M requires a positive even M")
        if flow.quant_mode != "NoQuant" or flow.relu_enabled:
            raise FixpipeError("dualDstCtl=1 requires NoQuant and no ReLU")
    elif flow.dual_dst_ctl == 2:
        if flow.fixpipe_n % 32 or flow.fixpipe_n <= 0:
            raise FixpipeError("dualDstCtl=2 split-N requires N % 32 == 0")
        if flow.quant_mode != "NoQuant" or flow.relu_enabled:
            raise FixpipeError("dualDstCtl=2 requires NoQuant and no ReLU")
    elif flow.dual_dst_ctl == 0:
        if len(flow.lanes_receiving) == 2 and flow.fixpipe_calls < 2:
            raise FixpipeError("dual-lane dualDstCtl=0 requires two explicit Fixpipe calls")
    elif flow.dual_dst_ctl is None:
        if len(flow.lanes_receiving) == 2:
            raise FixpipeError("dual-lane L0C->UB requires an explicit dualDstCtl")
    return selection


def buffer_capacity(profile: HardwareProfile, pool: BufferPool | str, *,
                    archetype: ExecutionArchetype | str = ExecutionArchetype.PURE_VECTOR) -> int:
    try:
        chosen = pool if isinstance(pool, BufferPool) else BufferPool(str(pool))
    except ValueError as exc:
        raise FlowSchemaError(f"unknown buffer pool: {pool!r}") from exc
    if chosen is BufferPool.GM:
        raise FlowSchemaError("GM is not a finite SRAM pool")
    if chosen is BufferPool.SSBUF:
        return profile.ssbuf_bytes(ExecutionArchetype(archetype) is ExecutionArchetype.CV_COUPLED_1C2V)
    return profile.memory.capacity(chosen.value)


_FLOW_FIELDS = frozenset(OpFlowTemplate.__dataclass_fields__)


def flow_from_mapping(name: str, payload: Mapping[str, Any]) -> OpFlowTemplate:
    """Rehydrate a flow record from JSON-friendly values, rejecting extras."""
    unknown = set(payload) - _FLOW_FIELDS - {"archetype", "pipe_work"}
    if unknown:
        raise FlowSchemaError(f"unknown flow fields: {sorted(unknown)}")
    try:
        values = dict(payload)
        values["name"] = name
        values["pipe_work"] = tuple(PipeWork(**item) for item in payload.get("pipe_work", ()))
        values["buffers"] = tuple(BufferSpec(**item) for item in payload.get("buffers", ()))
        values["owner_work"] = tuple(OwnerWork(**item) for item in payload.get("owner_work", ()))
        if payload.get("recurrence") is not None:
            values["recurrence"] = LoopCarriedRecurrence(**payload["recurrence"])
        return OpFlowTemplate(**values)
    except TimeoutError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise FlowSchemaError(f"cannot build flow {name!r}: {exc}") from exc


__all__ = [
    "AuthoringBackend", "BufferPool", "BufferSpec", "ExecutionArchetype",
    "ExecutionMode", "LoopCarriedRecurrence", "FixpipeConfig", "FixpipeError",
    "FixpipePath", "FixpipeSelection", "FlowSchemaError", "OpFlowTemplate",
    "PIPE_NAMES", "PipeWork", "OwnerWork", "VectorMode", "buffer_capacity",
    "check_flow_fixpipe", "flow_from_mapping", "require_int",
    "require_positive_float", "select_fixpipe",
]
