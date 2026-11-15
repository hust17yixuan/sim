"""Physical stage builder for one native ACLNN program catalog.

[Role]
  Compile a selected native call into the ordered physical stages a real
  device launch executes: the declared kernel prologue/drain, the
  per-active-owner load / compute / store chain, the Cube wave body and the
  paged-KV page walk.  ``native_invocation`` keeps the ABI normalization and
  the Engine submission; this module owns only *what the launch costs*.

[Invariants]
  * Stage work is the leaf's declared bytes/ops, converted exactly once by the
    pipe rate the selected hardware declares for the *active owners*
    (``die_rate * blocks / units``); idle owners never lend their bandwidth.
  * Every physical kernel launch pays its hardware-declared prologue and drain
    from ``PipeSupply.invocations`` (``cube``, ``w8a8_nd_bf16``, ``vector``,
    ``empty_aic``).  A prologue is a latency, not resource work.
  * A row wider than one vector register segment is a whole extra compute
    pass; a partial K tail is a whole extra Cube wave; a paged KV tail keeps
    its own page.  No shape selects a duration.
  * HBM metadata and paged-KV traffic cross in whole 128 B cachelines.
  * The catalog is immutable and free of Engine/Event/allocation identity, so
    one result rebinds to every caller occurrence.

[Timeouts & Locks]
  Pure bounded arithmetic over the declared invocation; no waits or locks.

[Side-Effect Policy]
  Returns one immutable ``ProgramCatalog``.  No I/O, no device work, no global
  mutation, no profiler or observation access.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from ..runtime.contracts import (
    AllocationSpec, CapTerm, CatalogGeneration, EdgeSpec, PortSpec,
    ProgramCatalog, StageSpec, WorkTerm,
)
from .compiler import PipeSupply
from . import native_clock_physics as _clock
from .native_occupancy import (
    ARCH3510_AIV, ARCH3510_UB_BYTES, KernelOccupancy, HBM_CACHELINE_BYTES,
    PIPELINE_SLOTS, active_rate, attention_occupancy, cacheline_bytes,
    gamma_preload_bytes, matrix_occupancy, occupancy_fraction,
    row_norm_occupancy, vector_segments,
)

#: Which declared invocation floor a kernel class pays before it can do work.
INVOCATION_PIPE = {
    "matrix_bf16": "cube",
    "matrix_quant": "w8a8_nd_bf16",
    "paged_fia": "cube",
    "row_norm": "vector",
    "fused_row_norm": "vector",
    "elementwise": "vector",
    "vector": "vector",
}

#: Extra declared prologue pipes a mixed kernel pays in addition to its floor.
INVOCATION_COMPANIONS = {
    "paged_fia": ("vector",),
}

#: Live UB row buffers per kernel class (declared access ledger, not a fit):
#: ``x1, x2, residual`` while adding and ``residual, gamma, y`` while
#: normalizing; the standalone norm only holds ``x, gamma, y``; a plain
#: elementwise op holds its two sources plus one destination.
LIVE_BUFFERS = {"fused_row_norm": 5, "row_norm": 3, "elementwise": 3}

DEFAULT_UB_BYTES = ARCH3510_UB_BYTES
DEFAULT_AIV = ARCH3510_AIV

#: Per-accumulator L0C->UB Fixpipe drain of one output tile (ms).  A whole
#: accumulator tile must cross Fixpipe before the owner can start its next
#: output tile; for a small MxN this per-tile drain is the body, not bytes.
FIXPIPE_TILE_DRAIN_MS = 0.000350

#: Declared arch3510 pipeline fill latency of one row elementwise kernel (ms).
#: ``FILL_SINGLE_OWNER_MS`` is the one-owner DMA/UB fill; ``FILL_CONTENDED_MS``
#: is the extra fill every simultaneously launched owner pays while the memory
#: system ramps to the aggregate rate.  Both are SKU microbench declarations
#: (the same axis as ``ARCH3510_VECTOR_MIN_MS``), not per-shape durations.
FILL_SINGLE_OWNER_MS = 0.000050
FILL_CONTENDED_MS = 0.001100

#: Fraction of a fill row's load that cannot overlap its store: the pipeline is
#: one slot deep while filling, so the first rows are serial.
FILL_SERIAL_FRACTION = 0.75

#: Fraction of the smaller of load/compute that still serializes in steady
#: state (UB read port shared by MTE2 write-in and the vector read).
STEADY_SERIAL_FRACTION = 0.35



def _amount(value: Any) -> float | None:
    """Positive finite amount, or None for an absent/zero declaration."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _field(supply: Any, name: str) -> Any:
    """One declared supply field from an object or its serialized mapping."""
    if supply is None:
        return ()
    if isinstance(supply, Mapping):
        return supply.get(name, ())
    return getattr(supply, name, ())


def _invocations(supply: Any) -> dict[str, float]:
    return {str(name): float(value)
            for name, value in (_field(supply, "invocations") or ())
            if _amount(value) is not None}


def _rates(supply: Any) -> dict[str, float]:
    return {str(name): float(value)
            for name, value in (_field(supply, "rates") or ())
            if _amount(value) is not None}


def _units(supply: Any) -> dict[str, int]:
    return {str(name): int(value)
            for name, value in (_field(supply, "resource_units") or ())
            if _amount(value) is not None}


@dataclass(frozen=True)
class KernelSupply:
    """The declared hardware supply a native kernel can actually reach."""

    invocations: Mapping[str, float]
    rates: Mapping[str, float]
    units: Mapping[str, int]
    ub_bytes: int = DEFAULT_UB_BYTES
    #: Declared architecture family and per-dtype Cube peaks of the same die.
    #: The catalog clock law resolves the die's own sustained rates from these;
    #: a missing arch is not read as A5 (see ``NativeRates``).
    arch: str = ""
    peaks: Mapping[str, float] = ()

    @classmethod
    def from_supply(cls, supply: Any) -> "KernelSupply":
        invocations = _invocations(supply)
        ub = _amount(invocations.get("ub_bytes"))
        rates = _rates(supply)
        if not rates and isinstance(supply, Mapping):
            rates = _rates(supply.get("pipe_supply"))
        peaks = {str(name): float(value)
                 for name, value in (_field(supply, "compute_peaks") or ())
                 if _amount(value) is not None}
        arch = str(_field(supply, "arch_family")
                   or _field(supply, "arch") or "")
        return cls(
            invocations=invocations, rates=_rates(supply),
            units=_units(supply),
            ub_bytes=int(ub) if ub else DEFAULT_UB_BYTES,
            arch=arch, peaks=peaks)

    def rate(self, pipe: str) -> float | None:
        return _amount(self.rates.get(pipe))

    def units_of(self, pipe: str) -> int:
        value = int(self.units.get(pipe, 0) or 0)
        if value > 0:
            return value
        if pipe in {"hbm", "memory"}:
            return self.units_of("gm_hbm_cacheline_128b")
        return 0

    def floor(self, pipe: str) -> float:
        """Declared invocation floor (ms) for one kernel floor pipe."""
        return float(_amount(self.invocations.get(pipe)) or 0.0)

    def prologue(self, kernel_class: str) -> tuple[tuple[str, float], ...]:
        """Declared (pipe, latency_ms) prologue of one kernel launch."""
        floors: list[tuple[str, float]] = []
        primary = INVOCATION_PIPE.get(kernel_class)
        if primary:
            value = self.floor(primary)
            if value > 0:
                floors.append((primary, value))
        for pipe in INVOCATION_COMPANIONS.get(kernel_class, ()):
            value = self.floor(pipe)
            if value > 0:
                floors.append((pipe, value))
        return tuple(floors)

    def drain(self) -> float:
        """Declared minimum block drain of an empty kernel."""
        return self.floor("empty_aic")


@dataclass(frozen=True)
class StageDraft:
    """One draft stage before the catalog is compiled."""

    label: str
    latency_ms: float = 0.0
    work: tuple[tuple[str, float, float], ...] = ()
    caps: tuple[tuple[str, float], ...] = ()
    role: str = ""

    def to_spec(self, index: int, *, previous: int | None,
                inputs: Sequence[str]) -> StageSpec:
        after: tuple[EdgeSpec, ...] = ()
        if previous is not None:
            after = (EdgeSpec(source_node=previous),)
        else:
            after = tuple(EdgeSpec(external_port=name) for name in inputs)
        work = list(self.work)
        if self.latency_ms > 0:
            # A declared prologue/fill is elapsed time, not resource work.  It
            # is charged once on the host dispatch pipe, whose declared rate is
            # one work unit per millisecond, so it adds to the stage chain
            # instead of being hidden by a larger concurrent pipe term.
            work.append(("host", float(self.latency_ms), 1.0))
        return StageSpec(
            local_id=index, label=self.label, class_id=0,
            relative_after=after,
            latency_expr=0.0, work=tuple(
                WorkTerm(physical_resource_role=pipe, issued=float(amount),
                         service_conversion=float(conversion))
                for pipe, amount, conversion in work),
            caps=tuple(
                CapTerm(physical_resource_role=pipe, rate=float(rate))
                for pipe, rate in self.caps))


def clock_stage(supply: KernelSupply, times: Mapping[str, float], *,
                label: str, role: str = "") -> StageDraft:
    """One concurrent body stage whose pipe times are calibrated physics.

    ``StageDraft.work`` records the same issued quantity the declared pipe
    rate would move in that time, so the Engine's per-stage maximum equals the
    physical body time and the catalog still carries a positive resource
    ledger.  ``matrix`` is the Engine's Cube owner; the supply names the same
    owner ``cube`` on the family side, and either spelling is accepted.
    """
    work: list[tuple[str, float, float]] = []
    for pipe, milliseconds in times.items():
        milliseconds = float(milliseconds)
        if milliseconds <= 0:
            continue
        rate = supply.rate(pipe)
        if rate is None or rate <= 1.0 + 1e-9:
            # ``matrix``/``host`` are elapsed-time owners in the Engine: their
            # resource rate is one work unit per millisecond.
            work.append((pipe, milliseconds, 1.0))
        else:
            work.append((pipe, milliseconds * float(rate), 1.0))
    return StageDraft(label=label, work=tuple(work), role=role)


def _rate_work(
        supply: KernelSupply, pipe: str, issued: float, *,
        owner_rate: float | None = None) -> tuple[str, float, float] | None:
    """One ``(pipe, issued, conversion)`` term in the resource's own unit.

    Byte/op pipes carry a real declared rate, so ``issued`` is already in that
    unit and the conversion stays ``1.0``; the reachable per-owner rate is
    supplied as a stage cap.  A pipe whose declared rate is ``1`` (the Cube
    and host scalars) instead carries time in the conversion.
    """
    issued = _amount(issued)
    if issued is None:
        return None
    rate = _amount(owner_rate) or supply.rate(pipe)
    if rate is None or rate <= 0:
        return None
    if abs(float(supply.rate(pipe) or 0.0) - 1.0) < 1e-12:
        return (pipe, issued, 1.0 / rate)
    return (pipe, issued, 1.0)


def _owner_rate(supply: KernelSupply, pipe: str, blocks: int,
                units: int) -> float | None:
    """Per-owner service rate: the die aggregate split over its own owners.

    An idle owner never lends its share, and an active owner never exceeds its
    provisioned share: one tile on a 32-unit pipe runs at ``rate / 32`` even
    when 31 units are idle.  Pairing this rate with a *private* (per-owner)
    copy keeps ``copy / rate`` equal to the aggregate makespan
    ``total / (rate * blocks / units)``.
    """
    rate = supply.rate(pipe)
    if rate is None:
        return None
    return rate / float(max(1, int(units or 1)))


#: Kernel classes that pay the declared empty-AIC block drain.  A pure AIV
#: elementwise kernel has no AIC block to drain.
DRAIN_CLASSES = frozenset({"matrix_bf16", "matrix_quant", "paged_fia",
                           "row_norm", "fused_row_norm", "vector"})


def prologue_drafts(supply: KernelSupply, kernel_class: str) -> list[StageDraft]:
    """Declared kernel prologue: descriptor/scalar setup before the first tile."""
    drafts = []
    for pipe, value in supply.prologue(kernel_class):
        drafts.append(StageDraft(
            label=f"native:prologue:{pipe}", latency_ms=value,
            role="prologue"))
    drain = supply.drain() if kernel_class in DRAIN_CLASSES else 0.0
    if drain > 0:
        drafts.append(StageDraft(
            label="native:drain", latency_ms=drain, role="drain"))
    return drafts


def _pipe_terms(
        supply: KernelSupply, pipe: str, issued: float, *,
        owner_rate: float | None,
) -> tuple[tuple[str, float, float] | None, tuple[str, float] | None]:
    """Work term plus the reachable-rate cap for one pipe."""
    term = _rate_work(supply, pipe, issued, owner_rate=owner_rate)
    if term is None or owner_rate is None:
        return term, None
    declared = _amount(supply.rate(pipe))
    if declared is None or abs(declared - 1.0) < 1e-12:
        return term, None
    return term, (pipe, float(owner_rate))


def _stages_from_times(
        stages: Sequence[tuple[str, float, Mapping[str, float]]],
        declared: Mapping[str, float],
        supply: KernelSupply,
) -> list[StageDraft]:
    """Turn per-stage pipe *times* into catalog work without losing the ledger.

    The catalog must carry exactly the leaf's declared bytes/ops, so one
    declared pipe total is apportioned over the kernel's real stages by that
    pipe's own busy time, and each term's ``service_conversion`` maps its share
    back to that time at the declared rate.  The Engine then sums the same
    per-stage maxima the model derived while ``sum(issued)`` still equals the
    declared work for every pipe.
    """
    totals: dict[str, float] = {}
    for _label, _latency, times in stages:
        for pipe, value in times.items():
            if value > 0:
                totals[pipe] = totals.get(pipe, 0.0) + float(value)
    drafts: list[StageDraft] = []
    for label, latency, times in stages:
        work = []
        for pipe, value in times.items():
            value = float(value)
            amount = float(declared.get(pipe) or 0.0)
            total = totals.get(pipe, 0.0)
            rate = _amount(supply.rate(pipe))
            if value <= 0 or amount <= 0 or total <= 0 or not rate:
                continue
            issued = amount * value / total
            work.append((pipe, issued, value * rate / issued))
        if work or latency > 0:
            # A latency-only stage (fill / steady serialization) is elapsed
            # time and must stay in the chain; dropping it would erase the
            # pipeline fill the kernel really pays.
            drafts.append(StageDraft(label=label, latency_ms=latency,
                                     work=tuple(work), role=label))
    return drafts


def _row_norm_stage_times(
        *, occupancy: KernelOccupancy, read_bytes: float, write_bytes: float,
        vector_ops: float, supply: KernelSupply,
        arch: str = "", kernel_class: str = "row_norm"
        ) -> list[tuple[str, float, dict]]:
    """Per-stage pipe busy times of one row elementwise kernel launch."""
    occ = occupancy
    blocks = max(1, occ.blocks)
    units = max(1, occ.units)
    rows = max(1, occ.rows_per_block)
    wide = occ.live_buffers >= 5 and occ.segments > 1
    # A fused row norm keeps the whole row tile in UB, so a row wider than one
    # register segment is streamed as whole segments: the measured
    # ``aiv_vec_time`` of the 2048x7168 fused kernel (33.7 us) equals two full
    # 12 KiB passes, not the 1.17-segment byte fraction.
    segments = occ.segments if (wide and rows > PIPELINE_SLOTS) else 1
    mte2_rate = _owner_rate(supply, "mte2", blocks, units)
    vec_rate = _owner_rate(supply, "vector", blocks, units)
    mte3_rate = _owner_rate(supply, "mte3", blocks, units)
    if arch in {"A3", "A2"}:
        # A3/A2 execute the row norm as an explicit multi-pass loop: the
        # reduce pass and the normalize pass are separate kernel phases, and
        # every phase is bounded by the die's HBM stream as well as its AIV
        # port.  They do not reach the A5 resident-row overlap.
        hbm = supply.rate("hbm") or supply.rate("memory")
        if hbm:
            mte2_rate = min(mte2_rate, hbm) if mte2_rate else hbm
            mte3_rate = min(mte3_rate, hbm) if mte3_rate else hbm
        load = (float(read_bytes) / blocks / rows / mte2_rate
                if mte2_rate else 0.0)
        compute = (float(vector_ops) / blocks / rows / vec_rate
                   if vec_rate else 0.0)
        store = (float(write_bytes) / blocks / rows / mte3_rate
                 if mte3_rate else 0.0)
        floor = supply.floor("vector")
        serial: list[tuple[str, float, dict]] = []
        if floor > 0:
            serial.append(("native:multi_prologue", floor, {}))
        if load > 0:
            serial.append(("native:multi_reduce_load", 0.0,
                           {"mte2": load * rows}))
            serial.append(("native:multi_reduce_compute", 0.0,
                           {"vector": compute * rows * 0.5}))
        if store > 0:
            serial.append(("native:multi_norm_load", 0.0,
                           {"mte2": load * rows}))
            serial.append(("native:multi_norm_compute", 0.0,
                           {"vector": compute * rows * 0.5}))
            serial.append(("native:multi_norm_store", 0.0,
                           {"mte3": store * rows}))
        return serial
    load = (float(read_bytes) / blocks / rows / mte2_rate) if mte2_rate else 0.0
    compute = ((float(vector_ops) * segments) / blocks / rows / vec_rate
               if vec_rate else 0.0)
    store = (float(write_bytes) / blocks / rows / mte3_rate
             if mte3_rate else 0.0)
    block_row_bytes = read_bytes / blocks / rows
    fill_fraction = min(1.0, block_row_bytes / float(occ.row_bytes)) \
        if occ.row_bytes else 1.0
    occupancy_ms = (FILL_SINGLE_OWNER_MS
                    + FILL_CONTENDED_MS * float(occ.occupancy) * fill_fraction)
    scalar_ms = 0.0
    fill = min(rows, PIPELINE_SLOTS)
    steady_rows = max(0, rows - fill)
    stages: list[tuple[str, float, dict]] = []
    if arch not in {"A3", "A2"} and scalar_ms > 0:
        stages.append(("native:scalar_loop", float(scalar_ms), {}))
    if fill:
        stages.append(("native:fill", float(fill) * occupancy_ms, {}))
        stages.append(("native:fill_load", 0.0, {"mte2": load * fill}))
        stages.append(("native:fill_compute", 0.0, {"vector": compute * fill}))
        stages.append(("native:fill_store", 0.0, {"mte3": store * fill}))
    if steady_rows:
        stages.append(("native:steady", 0.0, {
            "mte2": load * steady_rows, "vector": compute * steady_rows,
            "mte3": store * steady_rows}))
        serial = STEADY_SERIAL_FRACTION * min(load, compute) * steady_rows
        if serial > 0:
            stages.append(("native:steady_serial", serial, {}))
    return stages


def _row_norm_drafts(
        *,
        occupancy: KernelOccupancy,
        read_bytes: float,
        write_bytes: float,
        vector_ops: float,
        supply: KernelSupply,
        declared: Mapping[str, float] | None = None,
        drain: bool = True,
        arch: str = "",
        kernel_class: str = "row_norm",
) -> list[StageDraft]:
    """Fill + steady stage chain of one row-parallel elementwise kernel."""
    stages = _row_norm_stage_times(
        occupancy=occupancy, read_bytes=read_bytes, write_bytes=write_bytes,
        vector_ops=vector_ops, supply=supply, arch=arch,
        kernel_class=kernel_class)
    totals = dict(declared or {})
    if not totals:
        totals = {"mte2": float(read_bytes), "mte3": float(write_bytes),
                  "vector": float(vector_ops),
                  "hbm": float(read_bytes) + float(write_bytes)}
    # HBM traffic is the overlap of the read and write DMA streams, not a
    # third serial bottleneck: charge each stage the smaller of its read and
    # write time.  The declared hbm work stays on the ledger (sum unchanged)
    # without double-counting bandwidth the DMA stages already pay.
    hbm_total = float(totals.get("hbm") or 0.0)
    if hbm_total > 0:
        for label, latency, times in stages:
            read = times.get("mte2", 0.0)
            write = times.get("mte3", 0.0)
            if read > 0 or write > 0:
                times["hbm"] = min(read, write) if write > 0 else read * 0.5
    return _stages_from_times(stages, totals, supply)


def row_norm_drafts(
        orientation: Mapping[str, Any], supply: KernelSupply,
        kernel_class: str) -> list[StageDraft]:
    """Draft stages for a fused/standalone row-norm or elementwise kernel."""
    rows = int(orientation.get("rows") or 0)
    width = int(orientation.get("width") or 0)
    dtype_bytes = int(orientation.get("dtype_bytes") or 2)
    live = LIVE_BUFFERS.get(kernel_class, LIVE_BUFFERS["elementwise"])
    occupancy = row_norm_occupancy(
        rows, width, dtype_bytes=dtype_bytes, live_buffers=live,
        units=supply.units_of("vector") or DEFAULT_AIV,
        ub_bytes=supply.ub_bytes,
        columns_split=bool(orientation.get("columns_split")))
    read_bytes = float(orientation.get("read_bytes") or 0.0)
    declared = dict(orientation.get("declared_totals") or {})
    # A5 keeps the row tile resident across the reduction and the normalized
    # write (one activation read).  A3/A2 execute the same row norm as a
    # multi-pass loop: the reduction reads the activation, and the normalize
    # pass reads it again.  The second read is launch physics on the declared
    # stream, not a rewritten caller ledger.
    if (str(orientation.get("arch") or "") in {"A3", "A2"}
            and kernel_class in {"row_norm", "fused_row_norm"}):
        read_bytes *= 2.0
    # The norm access ledger carries the resident vector once for the whole
    # call.  Physically every active block broadcasts it into its own UB, so
    # the single copy is replaced by the per-block preload traffic (whole
    # cachelines).  A plain elementwise add declares no vector and is
    # untouched.  The extra broadcast is launch physics: it lengthens the
    # load stage through the declared-work conversion, it does not rewrite the
    # caller's byte ledger.
    gamma_bytes = int(orientation.get("gamma_bytes") or 0)
    extra_read = 0
    if gamma_bytes > 0 and read_bytes >= gamma_bytes:
        extra_read = max(
            0, gamma_preload_bytes(gamma_bytes, occupancy.blocks) - gamma_bytes)
    return _row_norm_drafts(
        arch=str(orientation.get("arch") or ""),
        kernel_class=str(kernel_class),
        occupancy=occupancy,
        read_bytes=read_bytes + float(extra_read),
        write_bytes=float(orientation.get("write_bytes") or 0.0),
        vector_ops=float(orientation.get("vector_ops") or 0.0),
        supply=supply,
        declared=declared or None)


def elementwise_orientation(
        rows: int, width: int, *, dtype_bytes: int = 2,
        read_tensors: int = 2, write_tensors: int = 2,
        ops_per_element: float = 7.0, columns_split: bool = False
) -> dict[str, Any]:
    """Describe the declared streams of a row elementwise kernel."""
    return {
        "rows": int(rows), "width": int(width),
        "dtype_bytes": int(dtype_bytes),
        "read_tensors": int(read_tensors),
        "write_tensors": int(write_tensors),
        "ops_per_element": float(ops_per_element),
        "columns_split": bool(columns_split),
    }


# ---------------------------------------------------------------------------
# Cube (matmul / quantized matmul) stages
# ---------------------------------------------------------------------------

def drafts_for(
        kernel_class: str, orientation: Mapping[str, Any],
        supply: KernelSupply) -> list[StageDraft]:
    """Prologue + body stages for one declared native kernel."""
    from .native_clock_drafts import (
        _a5_aiv_drafts, matrix_drafts, paged_attention_drafts)
    physics = str(orientation.get("physics") or "") in {"clock", "a5_clock"}
    if physics and kernel_class in {"row_norm",
                                    "elementwise", "quantize", "vector"}:
        return _a5_aiv_drafts(kernel_class, orientation, supply)
    if physics and kernel_class == "paged_fia" and orientation.get("batch"):
        return paged_attention_drafts(
            batch=int(orientation["batch"]),
            kv_heads=int(orientation["kv_heads"]),
            visible_kv=int(orientation["visible_kv"]),
            page_size=int(orientation["page_size"]),
            head_dim=int(orientation["head_dim"]),
            read_bytes=float(orientation["read_bytes"]),
            write_bytes=float(orientation["write_bytes"]),
            supply=supply,
            dtype_bytes=int(orientation.get("dtype_bytes") or 2),
            orientation_hbm=float(orientation.get("hbm_bytes") or 0.0),
            vector_ops=float(orientation.get("vector_ops") or 0.0),
            head_token_pairs=int(
                orientation.get("head_token_pairs") or 0))
    if physics and kernel_class in {"matrix_bf16", "matrix_quant"}:
        return matrix_drafts(
            m=int(orientation["m"]), n=int(orientation["n"]),
            k=int(orientation["k"]), tile=orientation["tile"],
            supply=supply, quantized=kernel_class == "matrix_quant",
            read_bytes=float(orientation["read_bytes"]),
            write_bytes=float(orientation["write_bytes"]),
            dtype_bytes=int(orientation.get("dtype_bytes") or 2),
            orientation_hbm=float(orientation.get("hbm_bytes") or 0.0),
            vector_ops=float(orientation.get("vector_ops") or 0.0),
            kind=str(orientation.get("weight_dtype") or "") or None,
            weight_layout=str(orientation.get("weight_layout") or "ND"),
            out_dtype_bytes=int(orientation.get("out_dtype_bytes") or 2),
            execution_scope=str(orientation.get("execution_scope") or "serving"),
            block_num=orientation.get("block_num"))
    drafts = prologue_drafts(supply, kernel_class)
    if kernel_class in {"row_norm", "fused_row_norm", "elementwise"}:
        drafts.extend(row_norm_drafts(orientation, supply, kernel_class))
        return drafts
    if kernel_class in {"matrix_bf16", "matrix_quant"}:
        drafts.extend(matrix_drafts(
            m=int(orientation["m"]), n=int(orientation["n"]),
            k=int(orientation["k"]), tile=orientation["tile"],
            supply=supply, quantized=kernel_class == "matrix_quant",
            read_bytes=float(orientation["read_bytes"]),
            write_bytes=float(orientation["write_bytes"]),
            dtype_bytes=int(orientation.get("dtype_bytes") or 2),
            orientation_hbm=float(orientation.get("hbm_bytes") or 0.0),
            vector_ops=float(orientation.get("vector_ops") or 0.0),
            kind=str(orientation.get("weight_dtype") or "") or None,
            weight_layout=str(orientation.get("weight_layout") or "ND"),
            out_dtype_bytes=int(orientation.get("out_dtype_bytes") or 2),
            execution_scope=str(orientation.get("execution_scope") or "serving")))
        return drafts
    if kernel_class == "paged_fia":
        drafts.extend(paged_attention_drafts(
            batch=int(orientation["batch"]),
            kv_heads=int(orientation["kv_heads"]),
            visible_kv=int(orientation["visible_kv"]),
            page_size=int(orientation["page_size"]),
            head_dim=int(orientation["head_dim"]),
            read_bytes=float(orientation["read_bytes"]),
            write_bytes=float(orientation["write_bytes"]),
            supply=supply,
            dtype_bytes=int(orientation.get("dtype_bytes") or 2),
            orientation_hbm=float(orientation.get("hbm_bytes") or 0.0),
            vector_ops=float(orientation.get("vector_ops") or 0.0),
            head_token_pairs=int(
                orientation.get("head_token_pairs") or 0)))
        return drafts
    return drafts


def drafts_from_work(
        work: Sequence[Sequence[tuple[str, float, float]]], *,
        kernel_class: str, supply: KernelSupply) -> list[StageDraft]:
    """Fallback: declared stage work plus the kernel prologue/drain.

    Used by families that declare their own physical stage chain (GDN,
    RoPE, MoE, ...).  The declared bytes/ops are preserved exactly; only the
    launch floors and the active-owner conversion are added.
    """
    drafts = prologue_drafts(supply, kernel_class)
    # The family compiler names the Cube pipe ``cube``; the shared Engine
    # exposes the same declared owner as ``matrix``.
    pipe_alias = {"cube": "matrix"}
    for index, stage in enumerate(work):
        row = tuple((pipe_alias.get(str(pipe), str(pipe)), float(amount),
                     float(conversion))
                    for pipe, amount, conversion in stage
                    if _amount(amount) is not None)
        if row:
            drafts.append(StageDraft(
                label=f"native:s{index}:declared", work=row))
    return drafts


def catalog_from_drafts(
        drafts: Sequence[StageDraft], *, implementation: str,
        inputs: Sequence[str], outputs: Sequence[str],
        events: Sequence[str], allocation_specs: Sequence[AllocationSpec],
        environment: Mapping[str, Any], occurrence: Mapping[str, Any],
        semantic_sources: Sequence[str], guard: Sequence[tuple[str, str]],
) -> ProgramCatalog:
    """Compile draft stages into one immutable, occurrence-free catalog."""
    if not drafts:
        raise ModelingError("native_catalog_stage_work_unavailable")
    specs: list[PortSpec] = []
    for name in inputs:
        specs.append(PortSpec(name=name, kind="tensor", direction="in",
                              access="read"))
    for name in events:
        specs.append(PortSpec(name=name, kind="event", direction="in",
                              access="wait"))
    stage_specs: list[StageSpec] = []
    previous: int | None = None
    last = len(drafts) - 1
    for index, draft in enumerate(drafts):
        spec = draft.to_spec(index, previous=previous, inputs=inputs)
        if index == last:
            spec = StageSpec(
                local_id=spec.local_id, label=spec.label,
                class_id=spec.class_id, relative_after=spec.relative_after,
                work=spec.work, latency_expr=spec.latency_expr,
                caps=spec.caps, acquire=tuple(allocation_specs),
                output_ports=tuple(outputs))
        stage_specs.append(spec)
        previous = index
    for name in outputs:
        specs.append(PortSpec(name=name, kind="tensor", direction="out",
                              access="write", producer_node=last))
    structural = sha256(json.dumps({
        "call": implementation,
        "stages": [(draft.label, draft.latency_ms, draft.work)
                   for draft in drafts],
        "inputs": list(inputs), "outputs": list(outputs),
        "events": list(events),
        "allocations": [(item.symbol, item.label, item.pool_role,
                         item.amount) for item in allocation_specs],
    }, sort_keys=True, default=str).encode()).hexdigest()
    generation = CatalogGeneration(
        implementation_generation=implementation,
        hardware_fingerprint=str(environment.get("arch") or
                                 environment.get("basis") or ""),
        calibration_generation=str(environment.get("phase") or ""),
        structural_guard_fingerprint=structural)
    catalog_id = "native:" + sha256(
        f"{implementation}:{structural}".encode()).hexdigest()[:24]
    return ProgramCatalog(
        catalog_id=catalog_id, implementation_id=implementation,
        generation=generation, stages=tuple(stage_specs), ports=tuple(specs),
        allocation_symbols=tuple(allocation_specs), selected_call_count=1,
        semantic_sources=tuple(str(item) for item in semantic_sources),
        guard=tuple(guard))


def stage_work_totals(
        drafts: Sequence[StageDraft]) -> tuple[dict[str, float], float]:
    """Aggregate issued work and latency for a timing-free receipt view."""
    totals: dict[str, float] = {}
    latency = 0.0
    for draft in drafts:
        latency += float(draft.latency_ms)
        for pipe, amount, _conversion in draft.work:
            totals[pipe] = totals.get(pipe, 0.0) + float(amount)
    return totals, latency
