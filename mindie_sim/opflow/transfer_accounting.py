"""Exact three-quantity transfer accounting for operator data movement.

[Role]
Separate logical shape bytes, actual source-read/destination-write bytes,
physical buffer capacity, and address span for ND/NZ, Load3D, strided-GM,
packed sub-byte dtypes, and ordinary moves. Every move is architecture-gated.
[Invariants]
Per-reference Load3D counts every valid non-pad reference; full_source_once
preloads the full input if dispatch is nonempty. Slots multiply aligned
capacity per slot. NZ block alignment is 16 x (256 // element_bits). The
requested primary pool selects the matching source/destination capacity.
[Timeouts & Locks]
Load3D uses logarithmic signed floor sums; no pixel/window iteration or waits.
[Side-Effect Policy]
Pure functions over immutable requests; no I/O or device access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from .flow_schema import BufferPool, BufferSpec, FlowSchemaError, ExecutionArchetype, buffer_capacity
from .hardware_profiles import HardwareProfile, validate_mem_copy
from .lifetime_math import allocation_peaks
from .exact_math import MathDomainError, integer, layout_capacity, load3d_counts, nddma_counts

MAX_DIM = 1_000_000


class TransferError(ValueError):
    """Invalid transfer geometry."""


class TransferKind(str, Enum):
    COPY = "copy"
    GM_STRIDED = "gm_strided"
    NDDMA = "nddma"
    DN2NZ = "dn2nz"
    NZ2DN = "nz2dn"
    LOAD3D = "load3d"
    FIXPIPE = "fixpipe"


class ReadPolicy(str, Enum):
    FULL_SOURCE_ONCE = "full_source_once"
    PER_REFERENCE = "per_reference"


def _as_int(name: str, value: object, *, minimum: int = 0) -> int:
    try:
        return integer(name, value, minimum)
    except TimeoutError:
        raise
    except MathDomainError as exc:
        raise TransferError(str(exc)) from exc


def align_up(value: int, alignment: int) -> int:
    value = _as_int("value", value)
    alignment = _as_int("alignment", alignment, minimum=1)
    return ((value + alignment - 1) // alignment) * alignment


def _pool_align(pool: BufferPool | str, profile: HardwareProfile) -> int:
    try:
        chosen = pool if isinstance(pool, BufferPool) else BufferPool(str(pool))
    except ValueError as exc:
        raise TransferError(f"unknown pool: {pool!r}") from exc
    if chosen is BufferPool.BIAS:
        return 64
    if chosen is BufferPool.GM:
        return 32
    if chosen is BufferPool.SSBUF:
        return 32
    if chosen in (BufferPool.UB_LANE0, BufferPool.UB_LANE1):
        return profile.memory.ub_align
    if chosen is BufferPool.L0C:
        return profile.memory.l0c_align
    return profile.memory.l1_align


def _require_shape(rows: int, cols: int) -> tuple[int, int]:
    rows = _as_int("rows", rows, minimum=1)
    cols = _as_int("cols", cols, minimum=1)
    if rows > MAX_DIM or cols > MAX_DIM:
        raise TransferError(f"matrix shape exceeds model budget: {rows}x{cols}")
    return rows, cols


@dataclass(frozen=True, slots=True)
class GmStrides:
    rows: int
    row_bytes: int
    stride_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _as_int("strides.rows", self.rows, minimum=1))
        object.__setattr__(self, "row_bytes", _as_int("strides.row_bytes", self.row_bytes, minimum=1))
        object.__setattr__(self, "stride_bytes", _as_int("strides.stride_bytes", self.stride_bytes, minimum=1))
        if self.stride_bytes < self.row_bytes:
            raise TransferError("GM row stride must not overlap row bytes")
        if self.rows * self.stride_bytes > 1 << 48:
            raise TransferError("strided GM span exceeds model budget")

    @property
    def effective_bytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def address_span_bytes(self) -> int:
        return (self.rows - 1) * self.stride_bytes + self.row_bytes

    @property
    def capacity_bytes(self) -> int:
        return self.address_span_bytes

    @property
    def full_pitch_reserve(self) -> int:
        return self.rows * self.stride_bytes


@dataclass(frozen=True, slots=True)
class TransferRequest:
    kind: TransferKind | str
    rows: int = 1
    cols: int = 1
    element_bytes: int = 2
    src_pool: BufferPool | str = BufferPool.GM
    dst_pool: BufferPool | str = BufferPool.UB_LANE0
    read_policy: ReadPolicy | str = ReadPolicy.FULL_SOURCE_ONCE
    gm_strides: GmStrides | Mapping[str, int] | None = None
    source_hw: tuple[int, int] | None = None
    channels: int = 1
    kernel: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    pad_before: tuple[int, int] = (0, 0)
    pad_after: tuple[int, int] = (0, 0)
    slots: int = 1
    resident: bool = False
    notes: str = ""
    bits_per_element: int = 0
    layout: str = "ND"
    fractal_layout: str = "NZ"
    packing: str = "continuous"
    capacity_policy: str = "minimum_span"
    nd_shape: tuple[int, ...] = ()
    stride_bits: tuple[int, ...] = ()
    base_bits: int = 0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", self.kind if isinstance(self.kind, TransferKind) else TransferKind(str(self.kind)))
        except ValueError as exc:
            raise TransferError(f"unknown transfer kind: {self.kind!r}") from exc
        try:
            object.__setattr__(self, "read_policy", self.read_policy if isinstance(self.read_policy, ReadPolicy) else ReadPolicy(str(self.read_policy)))
        except ValueError as exc:
            raise TransferError(f"unknown read policy: {self.read_policy!r}") from exc
        try:
            object.__setattr__(self, "src_pool", self.src_pool if isinstance(self.src_pool, BufferPool) else BufferPool(str(self.src_pool)))
            object.__setattr__(self, "dst_pool", self.dst_pool if isinstance(self.dst_pool, BufferPool) else BufferPool(str(self.dst_pool)))
        except ValueError as exc:
            raise TransferError(f"unknown pool: {exc}") from exc
        object.__setattr__(self, "rows", _as_int("rows", self.rows))
        object.__setattr__(self, "cols", _as_int("cols", self.cols))
        object.__setattr__(self, "element_bytes", _as_int("element_bytes", self.element_bytes))
        object.__setattr__(self, "bits_per_element", _as_int("bits_per_element", self.bits_per_element))
        if self.bits_per_element and not 1 <= self.bits_per_element <= 64:
            raise TransferError("bits_per_element must be in [1, 64]")
        if not self.bits_per_element and self.element_bytes not in (1, 2, 4, 8):
            raise TransferError("element_bytes must be 1, 2, 4, or 8 (or set bits_per_element)")
        object.__setattr__(self, "slots", _as_int("slots", self.slots, minimum=1))
        object.__setattr__(self, "channels", _as_int("channels", self.channels, minimum=1))
        if self.layout not in ("ND", "NZ", "ZN") or self.fractal_layout not in ("NZ", "ZN", "L0C", "MXA8W4"):
            raise TransferError("unsupported layout")
        if self.packing not in ("continuous", "row_restart"):
            raise TransferError("unknown packing policy")
        if self.capacity_policy not in ("minimum_span", "full_pitch"):
            raise TransferError("unknown allocation policy")
        if not isinstance(self.resident, bool):
            raise TransferError("resident must be bool")
        for name, dims in (("kernel", self.kernel), ("stride", self.stride), ("dilation", self.dilation)):
            if not isinstance(dims, (tuple, list)) or len(dims) != 2:
                raise TransferError(f"{name} must be a 2-tuple")
            values = tuple(_as_int(f"{name}[{i}]", value, minimum=1) for i, value in enumerate(dims))
            object.__setattr__(self, name, values)
        for name, dims in (("pad_before", self.pad_before), ("pad_after", self.pad_after)):
            if not isinstance(dims, (tuple, list)) or len(dims) != 2:
                raise TransferError(f"{name} must be a 2-tuple")
            object.__setattr__(self, name, tuple(_as_int(f"{name}[{i}]", value) for i, value in enumerate(dims)))

    @property
    def effective_elem_bytes(self) -> float:
        return self.bits_per_element / 8.0 if self.bits_per_element else float(self.element_bytes)

    def element_bytes_for(self, elements: int) -> int:
        if self.bits_per_element:
            return (elements * self.bits_per_element + 7) // 8
        return elements * self.element_bytes


@dataclass(frozen=True, slots=True)
class TransferAccounting:
    kind: TransferKind
    logical_input_bytes: int
    logical_output_bytes: int
    actual_read_bytes: int
    actual_write_bytes: int
    physical_capacity_bytes: int
    source_capacity_bytes: int
    destination_capacity_bytes: int
    physical_capacity_by_pool: Mapping[str, int]
    logical_bytes: int
    slots: int
    resident: bool
    notes: str = ""
    address_span_bytes: int = 0
    raw_destination_capacity_bytes: int = 0
    raw_source_capacity_bytes: int = 0
    output_hw: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resident, bool):
            raise TransferError("resident must be bool")

    @property
    def actual_rw_bytes(self) -> int:
        return self.actual_read_bytes + self.actual_write_bytes

    @property
    def capacity_bytes(self) -> int:
        return self.physical_capacity_bytes


def _coerce_strides(value: GmStrides | Mapping[str, int] | None) -> GmStrides | None:
    if value is None or isinstance(value, GmStrides):
        return value
    if not isinstance(value, Mapping):
        raise TransferError("gm_strides must be GmStrides or a mapping")
    unknown = set(value) - {"rows", "row_bytes", "stride_bytes"}
    if unknown:
        raise TransferError(f"unknown gm_strides fields: {sorted(unknown)}")
    try:
        return GmStrides(rows=value["rows"], row_bytes=value["row_bytes"],
                         stride_bytes=value["stride_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TransferError("gm_strides requires integer rows/row_bytes/stride_bytes") from exc


def account_transfer(request: TransferRequest, profile: HardwareProfile,
                     *, pool: BufferPool | str | None = None) -> TransferAccounting:
    """Compute the exact logical/actual/physical triple for one move."""
    if not isinstance(request, TransferRequest):
        raise TransferError("request must be a TransferRequest")
    kind = request.kind
    rows, cols = _require_shape(request.rows, request.cols)
    validate_mem_copy(request.src_pool.value, request.dst_pool.value, profile)
    elem = request.element_bytes
    bits = request.bits_per_element or elem * 8
    logical = (rows * request.element_bytes_for(cols) if request.packing == "row_restart"
               else request.element_bytes_for(rows * cols))
    src_align = _pool_align(request.src_pool, profile)
    dst_align = _pool_align(request.dst_pool, profile)
    try:
        capacity_pool = request.dst_pool if pool is None else (pool if isinstance(pool, BufferPool) else BufferPool(str(pool)))
    except ValueError as exc:
        raise TransferError(f"unknown primary pool: {pool!r}") from exc
    if capacity_pool not in (request.src_pool, request.dst_pool):
        raise TransferError(
            f"primary pool {capacity_pool.value} must be the source or destination pool"
        )
    if capacity_pool is BufferPool.GM:
        cap_align = 32
    else:
        cap_align = _pool_align(capacity_pool, profile)
    slots = request.slots
    strides = _coerce_strides(request.gm_strides)

    logical_in = logical
    logical_out = logical
    actual_read = logical
    actual_write = logical
    source_capacity = logical
    destination_capacity = logical
    address_span = logical
    output_hw = None
    note = request.notes

    if kind is TransferKind.NDDMA:
        if profile.arch_code != 3510:
            raise TransferError("NDDMA requires 3510")
        try:
            counts = nddma_counts(request.nd_shape, request.stride_bits, bits, request.base_bits)
        except TimeoutError:
            raise
        except MathDomainError as exc:
            raise TransferError(str(exc)) from exc
        logical_in = logical_out = (counts["logical_bits"] + 7) // 8
        actual_read = actual_write = counts["actual_transfer_bytes"]
        address_span = source_capacity = counts["address_span_bytes"]
        destination_capacity = actual_write
        if request.capacity_policy == "full_pitch":
            outer = max(range(len(request.stride_bits)), key=request.stride_bits.__getitem__)
            pitch = request.stride_bits[outer]
            if len(request.nd_shape) != 2 or pitch <= 0 or pitch % 8:
                raise TransferError("full-pitch reserve requires a 2D byte-pitch layout")
            source_capacity = (request.nd_shape[outer] * pitch // 8
                               if counts["logical_bits"] else 0)
            if source_capacity < address_span:
                raise TransferError("full-pitch allocation does not cover addressed span")
    elif kind is TransferKind.GM_STRIDED:
        if strides is None:
            raise TransferError("gm_strided requires gm_strides")
        if strides.rows != rows or strides.row_bytes != request.element_bytes_for(cols):
            raise TransferError(
                "gm_strides must match the declared logical extent "
                f"({rows}x{cols}x{elem} = {logical} B, got "
                f"{strides.rows}x{strides.row_bytes} = {strides.effective_bytes} B)"
            )
        actual_read = strides.effective_bytes
        actual_write = strides.effective_bytes
        source_capacity = (strides.full_pitch_reserve if request.capacity_policy == "full_pitch"
                           else strides.capacity_bytes)
        destination_capacity = align_up(actual_write, dst_align)
        address_span = strides.address_span_bytes
        logical_in = logical
        note = note or "strided source address span is physical; destination is contiguous"
    elif kind is TransferKind.DN2NZ:
        try:
            nz_capacity = layout_capacity(rows, cols, bits, request.fractal_layout)
        except TimeoutError:
            raise
        except MathDomainError as exc:
            raise TransferError(str(exc)) from exc
        destination_capacity = nz_capacity
        actual_write = logical
        if strides is not None:
            if strides.rows != rows or strides.row_bytes != request.element_bytes_for(cols):
                raise TransferError("DN strides must match the declared ND extent")
            actual_read = strides.effective_bytes
            source_capacity = (strides.full_pitch_reserve if request.capacity_policy == "full_pitch"
                           else strides.capacity_bytes)
        address_span = nz_capacity
        note = note or "NZ block tail padding is physical only"
    elif kind is TransferKind.NZ2DN:
        try:
            nz_capacity = layout_capacity(rows, cols, bits, request.fractal_layout)
        except TimeoutError:
            raise
        except MathDomainError as exc:
            raise TransferError(str(exc)) from exc
        actual_read = nz_capacity
        source_capacity = nz_capacity
        destination_capacity = align_up(logical, dst_align)
        address_span = source_capacity
        if strides is not None:
            if request.dst_pool is not BufferPool.GM:
                raise TransferError("NZ2DN GM strides require a GM destination")
            if strides.rows != rows or strides.row_bytes != request.element_bytes_for(cols):
                raise TransferError("NZ2DN strides must match the ND destination extent")
            destination_capacity = (strides.full_pitch_reserve if request.capacity_policy == "full_pitch"
                                    else strides.capacity_bytes)
            address_span = strides.address_span_bytes
        note = note or "NZ source read includes block-tail padding"
    elif kind is TransferKind.LOAD3D:
        if not isinstance(request.source_hw, (tuple, list)) or len(request.source_hw) != 2:
            raise TransferError("load3d requires source_hw=(height,width)")
        try:
            counts = load3d_counts(
                H=request.source_hw[0], W=request.source_hw[1], C=request.channels,
                kh=request.kernel[0], kw=request.kernel[1], sh=request.stride[0],
                sw=request.stride[1], dh=request.dilation[0], dw=request.dilation[1],
                pt=request.pad_before[0], pl=request.pad_before[1],
                pb=request.pad_after[0], pr=request.pad_after[1], bits=bits,
                layout=request.layout)
        except TimeoutError:
            raise
        except MathDomainError as exc:
            raise TransferError(str(exc)) from exc
        logical_in = counts["logical_input_bytes"]
        logical_out = counts["unfolded_logical_output_bytes"]
        source_capacity = logical_in
        destination_capacity = counts["dest_physical_capacity_bytes"]
        key = ("actual_source_read_bytes_once" if request.read_policy is ReadPolicy.FULL_SOURCE_ONCE
               else "actual_source_read_bytes_per_ref")
        actual_read = counts[key]
        actual_write = logical_out
        address_span = destination_capacity
        output_hw = (counts["output_h"], counts["output_w"])
        note = note or "full input preload or exact valid-reference payload; zero-output suppresses dispatch"
    elif kind is TransferKind.FIXPIPE:
        actual_read = logical
        destination_capacity = align_up(logical, dst_align)
        address_span = destination_capacity
    elif kind is TransferKind.COPY:
        pass
    else:  # pragma: no cover
        raise TransferError(f"unsupported transfer kind: {kind.value}")

    per_slot = {
        request.src_pool.value: source_capacity,
        request.dst_pool.value: destination_capacity,
    }
    if capacity_pool is request.src_pool:
        per_slot[capacity_pool.value] = max(
            per_slot.get(capacity_pool.value, 0), source_capacity
        )
    elif capacity_pool is request.dst_pool:
        per_slot[capacity_pool.value] = max(
            per_slot.get(capacity_pool.value, 0), destination_capacity
        )
    by_pool: dict[str, int] = {}
    for name, size in per_slot.items():
        by_pool[name] = max(by_pool.get(name, 0), align_up(size, _pool_align(name, profile)) * slots)
    primary = by_pool[capacity_pool.value]
    return TransferAccounting(
        kind=kind, logical_input_bytes=logical_in, logical_output_bytes=logical_out,
        actual_read_bytes=actual_read, actual_write_bytes=actual_write,
        physical_capacity_bytes=primary,
        source_capacity_bytes=by_pool[request.src_pool.value],
        destination_capacity_bytes=by_pool[request.dst_pool.value],
        physical_capacity_by_pool=dict(sorted(by_pool.items())),
        logical_bytes=max(logical_in, logical_out), slots=slots,
        resident=request.resident, address_span_bytes=address_span, notes=note,
        raw_destination_capacity_bytes=destination_capacity,
        raw_source_capacity_bytes=source_capacity, output_hw=output_hw,
    )


@dataclass(frozen=True, slots=True)
class BufferPlan:
    pool: BufferPool
    peak_bytes: int
    logical_bytes: int
    reuse_savings_bytes: int
    allocation_count: int
    residents: int
    peak_by_owner: Mapping[int, int] = field(default_factory=dict)


def _buffer_alignment(profile: HardwareProfile, pool: BufferPool) -> int:
    if pool is BufferPool.BIAS:
        return 64
    if pool is BufferPool.SSBUF:
        return 32
    if pool is BufferPool.L0C:
        return profile.memory.l0c_align
    if pool in (BufferPool.UB_LANE0, BufferPool.UB_LANE1):
        return profile.memory.ub_align
    return profile.memory.l1_align


def plan_buffer_pool(buffers: Iterable[BufferSpec], pool: BufferPool | str,
                     profile: HardwareProfile) -> BufferPlan:
    """Peak-capacity plan with global lifetime overlap and slot alignment."""
    try:
        chosen = pool if isinstance(pool, BufferPool) else BufferPool(str(pool))
    except ValueError as exc:
        raise TransferError(f"unknown pool: {pool!r}") from exc
    specs = tuple(buffers)
    for spec in specs:
        if not isinstance(spec, BufferSpec):
            raise TransferError("buffers must contain BufferSpec records")
    selected = tuple(spec for spec in specs if spec.pool is chosen)
    records = [dict(pool=chosen.value, owner=spec.owner, lane=spec.lane,
                    alloc_id=spec.alloc_id, bytes=spec.bytes_per_tile, slots=spec.slots,
                    resident=spec.resident, start=spec.first_stage, end=spec.last_stage)
               for spec in selected]
    ledger = allocation_peaks(records, {chosen.value: _buffer_alignment(profile, chosen)})
    peaks = {owner: size for (_, owner, _), size in ledger["peaks"].items()}
    logical = max(ledger["totals"].values(), default=0)
    peak = max(peaks.values(), default=0)
    return BufferPlan(pool=chosen, peak_bytes=peak, logical_bytes=logical,
                      reuse_savings_bytes=logical - peak,
                      allocation_count=ledger["allocations"], residents=ledger["residents"],
                      peak_by_owner=peaks)


def check_pool_capacity(plan: BufferPlan, profile: HardwareProfile, *,
                        archetype: ExecutionArchetype | str = ExecutionArchetype.PURE_VECTOR) -> int:
    """Raise if a planned pool exceeds SRAM; return remaining bytes."""
    if plan.pool is BufferPool.GM:
        raise FlowSchemaError("GM has no finite capacity check")
    if plan.pool is BufferPool.SSBUF:
        capacity = buffer_capacity(profile, plan.pool, archetype=archetype)
    elif plan.pool in (BufferPool.UB_LANE0, BufferPool.UB_LANE1):
        capacity = profile.ub_usable_bytes("simd")
    else:
        capacity = profile.memory.capacity(plan.pool.value)
    if plan.peak_bytes > capacity:
        raise TransferError(f"{plan.pool.value} plan {plan.peak_bytes} B exceeds {capacity} B")
    return capacity - plan.peak_bytes


__all__ = [
    "BufferPlan", "GmStrides", "ReadPolicy", "TransferAccounting", "TransferError",
    "TransferKind", "TransferRequest", "account_transfer", "align_up",
    "check_pool_capacity", "plan_buffer_pool",
]
