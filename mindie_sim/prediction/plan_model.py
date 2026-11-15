"""Physical plan records and declared hardware capabilities.

[Role] Define ``HardwareProfile`` (declared per-rank rates/capacities),
  ``PlanNode`` (one physical task with UB reservations and phase work) and
  ``EngineEvaluation`` (observed Engine result).
[Invariants] Missing capability fields are distinguishable from declared false;
  UB reservations are per-owner; phase work preserves every declared resource.
[Timeouts & Locks] Finite record validation; no I/O, clocks or locks.
[Side-Effect Policy] Pure immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .fusion_feasibility import CubeResources

CAPABILITY_NAMES = ("aclgraph_capture", "aclgraph_prefetch", "fixpipe_epilogue",
                    "cube_prologue", "weight_prefetch", "ub_to_l1", "l0c_to_ub",
                    "fixpipe")


@dataclass(frozen=True)
class HardwareProfile:
    """Declared per-rank capability; no model or vendor name selects policy."""
    hbm_bytes_per_ms: float
    vector_ops_per_ms: float
    cube_flops_per_ms: float
    onchip_bytes_per_ms: float
    ub_bytes_per_aiv: int
    prefetch_capacity_bytes: int
    max_streams: int = 2
    launch_ms: float = 0.002
    device_min_ms: float = 0.003
    stream_switch_ms: float = 0.0
    tensor_parallel: int = 1
    capabilities: tuple[tuple[str, bool], ...] = ()
    ub_units: int = 1
    prefetch_units: int = 1
    usable_ub_bytes: int | None = None
    cube_capacities: CubeResources | None = None
    event_record_ms: float = 0.0
    event_wait_ms: float = 0.0
    mte2_bytes_per_ms: float = 0.0
    mte3_bytes_per_ms: float = 0.0
    prefetch_priority: int = 10
    provenance: str = "declared"

    def __post_init__(self):
        for name in ("hbm_bytes_per_ms", "vector_ops_per_ms", "cube_flops_per_ms",
                     "onchip_bytes_per_ms"):
            value = getattr(self, name)
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"hardware_rate_invalid:{name}")
        if self.ub_bytes_per_aiv < 1 or self.prefetch_capacity_bytes < 0:
            raise ValueError("hardware_capacity_invalid")
        if self.usable_ub_bytes is not None:
            if not isinstance(self.usable_ub_bytes, int) \
                    or isinstance(self.usable_ub_bytes, bool) \
                    or self.usable_ub_bytes <= 0:
                raise ValueError("hardware_usable_ub_invalid")
        if self.max_streams < 1 or self.max_streams > 3:
            raise ValueError("hardware_max_streams_invalid")
        if (self.launch_ms < 0 or self.device_min_ms < 0
                or self.stream_switch_ms < 0 or self.tensor_parallel < 1
                or self.ub_units < 1 or self.prefetch_units < 1):
            raise ValueError("hardware_timing_invalid")
        if self.event_record_ms < 0 or self.event_wait_ms < 0 \
                or self.mte2_bytes_per_ms < 0 or self.mte3_bytes_per_ms < 0:
            raise ValueError("hardware_event_cost_invalid")
        if self.prefetch_priority < 0:
            raise ValueError("hardware_prefetch_priority_invalid")

    @property
    def usable_ub(self) -> int:
        return int(self.usable_ub_bytes if self.usable_ub_bytes is not None
                   else self.ub_bytes_per_aiv)

    def capability(self, name: str) -> bool:
        return bool(dict(self.capabilities).get(name, False))

    def capability_declared(self, name: str) -> bool:
        return name in dict(self.capabilities)

    def require_capability(self, name: str, *, allow_false: bool = False) -> None:
        declared = dict(self.capabilities)
        if name not in declared and not allow_false:
            raise ValueError(f"hardware_capability_undeclared:{name}")

    @property
    def record_ms(self) -> float:
        return self.event_record_ms or self.stream_switch_ms / 2.0

    @property
    def wait_ms(self) -> float:
        return self.event_wait_ms or self.stream_switch_ms / 2.0

    @classmethod
    def from_case(cls, case, **overrides) -> "HardwareProfile":
        """Derive declared capacity/rates from the parsed Case machine."""
        machine = getattr(case, "machine", None)
        if machine is None or not getattr(machine, "dies", None):
            raise ValueError("hardware_case_machine_missing")
        die = machine.dies[0]
        rates = dict(die.rates)
        buffers = dict(die.buffers)
        runtime = getattr(case, "runtime", None)
        parallel = getattr(runtime, "parallel", None)
        capture = str(getattr(runtime, "graph_capture_mode", "none"))
        paths = {path.name for path in getattr(die.compute_arch, "paths", ())}
        capabilities = {
            "aclgraph_capture": capture in {"full", "piecewise"},
            "aclgraph_prefetch": False,
            # A Vector->Cube prologue needs a real UB->L1 handoff; ``gm->l1``
            # alone is operand loading, not a prologue path.  A Cube->Vector
            # epilogue likewise needs L0C->UB (or the declared fixpipe path),
            # never an inferred GM round trip.
            "ub_to_l1": "ub->l1" in paths,
            "l0c_to_ub": "l0c->ub" in paths,
            "fixpipe": "l0c->ub" in paths,
            "fixpipe_epilogue": "l0c->ub" in paths,
            "cube_prologue": "ub->l1" in paths,
            "weight_prefetch": "gm->l1" in paths and "l1->l0a" in paths,
        }
        cube = CubeResources(l1=int(buffers.get("l1", 0)),
                             l0a=int(buffers.get("l0a", 0)),
                             l0b=int(buffers.get("l0b", 0)),
                             l0c=int(buffers.get("l0c", 0)))
        fields = dict(
            hbm_bytes_per_ms=float(rates["hbm"]) / 1000.0,
            vector_ops_per_ms=float(rates["vector"]) / 1000.0,
            cube_flops_per_ms=float(getattr(die, "cube_flops_per_s", 0.0)) / 1000.0,
            onchip_bytes_per_ms=float(rates.get("mte1", rates["hbm"])) / 1000.0,
            ub_bytes_per_aiv=int(buffers.get("ub", 0)),
            prefetch_capacity_bytes=int(buffers.get("l1", 0))
            * int(die.pool_units("l1") if "l1" in buffers else 1),
            prefetch_units=int(die.pool_units("l1") if "l1" in buffers else 1),
            max_streams=2,
            launch_ms=float(getattr(die, "launch_us", 0.0)) / 1000.0,
            device_min_ms=0.0,
            tensor_parallel=int(getattr(parallel, "tp", 1) or 1),
            capabilities=tuple(sorted(capabilities.items())),
            ub_units=int(die.pool_units("ub")) if "ub" in buffers else 1,
            usable_ub_bytes=int(buffers.get("ub", 0)),
            cube_capacities=cube,
            mte2_bytes_per_ms=float(rates.get("mte2_aiv", rates["hbm"])) / 1000.0,
            mte3_bytes_per_ms=float(rates.get("mte3", rates["hbm"])) / 1000.0,
            provenance="case_machine",
        )
        fields.update(overrides)
        return cls(**fields)


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    work: tuple[tuple[str, float], ...]
    deps: tuple[str, ...] = ()
    stream: str = "main"
    kind: str = "kernel"
    args: tuple[str, ...] = ()
    recipe_id: str = ""
    hbm_read: float = 0.0
    hbm_write: float = 0.0
    prefetch_for: str = ""
    caps: tuple[tuple[str, float], ...] = ()
    priority: int = 0
    retained: tuple[str, ...] = ()
    collective: bool = False
    state_effect: str = ""
    capture_boundary: str = ""
    phase: str = "decode"
    ub_allocations: tuple[tuple[str, str, int, tuple[int, ...]], ...] = ()
    prefetch_alloc: tuple[str, str, int, tuple[int, ...]] | None = None
    free_alloc: tuple[str, str, int, tuple[int, ...]] | None = None
    numerical_policy: str = "exact"
    group_buffers: tuple[tuple[str, str, int, tuple[int, ...]], ...] = ()
    capture_dynamic: bool = False
    capture_ineligible: bool = False
    pointer_stable: bool = True
    phase_work: tuple[tuple[str, dict[str, float]], ...] = ()
    #: Buffers that belong to one active phase (e.g. Cube L1/L0 operands);
    #: acquired when that phase starts, released with the kernel.
    phase_acquisitions: tuple[tuple[str, tuple[tuple[str, str, int, tuple[int, ...]], ...]], ...] = ()

    def work_dict(self) -> dict:
        return dict(self.work)

    @property
    def launches(self) -> int:
        return 1 if self.kind == "kernel" else 0

    @property
    def event_deps(self) -> tuple[str, ...]:
        value = self.work_dict()
        return tuple(f"event:{resource}" for resource in value
                     if resource in {"event_record", "event_wait"})


@dataclass(frozen=True)
class EngineEvaluation:
    status: str
    duration_ms: float
    launches: int
    hbm_bytes: float
    cube_flops: float
    vector_ops: float
    started: tuple[tuple[str, float], ...] = ()
    completed: tuple[tuple[str, float], ...] = ()
    reason: str = ""
    prefetch_completed: tuple[tuple[str, float], ...] = ()
    capture_ms: float | None = None
    total_work: tuple[tuple[str, float], ...] = ()

    def start(self, node_id: str) -> float:
        return dict(self.started).get(node_id, math.inf)

    def complete(self, node_id: str) -> float:
        return dict(self.completed).get(node_id, math.inf)

    def prefetched(self, node_id: str) -> float:
        return dict(self.prefetch_completed).get(node_id, 0.0)


