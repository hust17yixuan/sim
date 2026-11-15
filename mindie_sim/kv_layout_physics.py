"""Physical KV-cache layout: typed Fractal-NZ planes, MTE2 reads, PD pages.

[Role]
  Model the three physical layout decisions that move Ascend KV capacity and
  bandwidth: typed ``PA_NZ`` NoPE/RoPE/scale planes with a dtype-derived
  innermost tile, the MTE2 cacheline/DMA behaviour across physical pages, and
  the component-major MLA page that PD transfer moves as one raw slot.

[Invariants]
  * The Fractal-NZ innermost tile is 32 bytes: 16 BF16/FP16 elements, 32
    INT8/FP8 elements.  ``512`` NoPE + ``64`` RoPE BF16 elements are already
    16-element aligned, so a BF16 MLA cache invents exactly ``0 B`` padding.
  * A compute-path ``view`` requires ``head_size % C0 == 0``; the transfer
    reformat may ceil-pad explicitly (``pad_to_tile=True``) and then reports
    the padding apart from the logical payload.
  * c8 scales are a per-token/per-head plane with their own dtype; a scale
    plane on a non-one-byte payload is a typed rejection.
  * ``enable_kv_nz`` requires non-sparse MLA on a PD decode consumer; a P
    producer, non-MLA model, sparse-SFA cache or non-PD deployment is a typed
    rejection, never a silently ignored switch.
  * NZ writes always use the two-pass gather/scatter path in this backend
    (``synchronize`` + reusable K/V staging); a fused ``2 * payload`` NZ
    transpose is a capability rejection, not a default.
  * Component-major MLA PD is one descriptor per page per layer; the legacy
    split is two.  Payload bytes are conserved apart from declared padding.
  * All arithmetic is integer-exact; NaN, booleans, negative dimensions and
    non-integral counts are rejected before multiplication.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    ceil_div, dtype_elem_bytes, optional_text, strict_bool, strict_dtype,
    strict_int,
)
from mindie_sim.schema.common import ModelingError

#: The Ascend Fractal-NZ innermost tile is one 32-byte cache unit.
FRACTAL_NZ_C0_BYTES = 32
#: Default BF16 element count of that tile (kept for callers that name it).
KV_NZ_TILE = FRACTAL_NZ_C0_BYTES // 2
NZ_MLAPO_SUPPORTED_BY_DEFAULT = False


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="kv_layout")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="kv_layout")


@typed_inputs
def fractal_c0_elements(dtype: str) -> int:
    """Elements in one 32-byte NZ tile for a payload dtype."""
    elem = dtype_elem_bytes("dtype", dtype, reason="kv_nz")
    if FRACTAL_NZ_C0_BYTES % elem:
        raise ModelingError(f"kv_nz_c0_not_integral_for_dtype:{dtype}")
    return int(FRACTAL_NZ_C0_BYTES // elem)


@typed_inputs
@dataclass(frozen=True)
class KvNzPreconditions:
    """Declared topology inputs of ``enable_kv_nz``."""

    enable_kv_nz: bool
    is_mla: bool
    is_sparse_sfa: bool = False
    kv_connector: str | None = None
    kv_connector_module_path: str | None = None
    is_kv_consumer: bool = False
    is_kv_producer: bool = False
    pd_disaggregated: bool = False
    kv_role: str | None = None
    mlapo_eligible: bool = False
    nz_supports_mlapo: bool = NZ_MLAPO_SUPPORTED_BY_DEFAULT

    def __post_init__(self) -> None:
        for name in ("enable_kv_nz", "is_mla", "is_sparse_sfa",
                     "is_kv_consumer", "is_kv_producer", "pd_disaggregated",
                     "mlapo_eligible", "nz_supports_mlapo"):
            _bool(name, getattr(self, name))
        optional_text("kv_connector", self.kv_connector, reason="kv_layout")
        optional_text("kv_connector_module_path", self.kv_connector_module_path,
                      reason="kv_layout")
        optional_text("kv_role", self.kv_role, reason="kv_layout")
        if self.kv_role is not None:
            if self.kv_role not in {"kv_consumer", "kv_producer"}:
                raise ModelingError(f"kv_layout_kv_role_invalid:{self.kv_role}")
            if self.kv_role == "kv_consumer" and not self.is_kv_consumer:
                raise ModelingError("kv_layout_kv_role_consumer_mismatch")
            if self.kv_role == "kv_producer" and not self.is_kv_producer:
                raise ModelingError("kv_layout_kv_role_producer_mismatch")
        if self.is_kv_consumer and self.is_kv_producer:
            raise ModelingError("kv_layout_pd_role_ambiguous")
        if self.is_kv_consumer or self.is_kv_producer:
            if not self.pd_disaggregated:
                raise ModelingError("kv_layout_pd_role_without_pd")


@typed_inputs
@dataclass(frozen=True)
class KvNzValidation:
    """Diagnostic codes and effective fusion state of one NZ declaration."""

    codes: tuple[str, ...]
    mlapo_enabled: bool
    mlapo_disabled_by_nz: bool
    pd_decode_node: bool
    mlapo_eligible: bool

    @property
    def ok(self) -> bool:
        return not self.codes

    def raise_if_invalid(self) -> "KvNzValidation":
        if self.codes:
            raise ModelingError("kv_nz_invalid:" + ",".join(self.codes))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "codes": list(self.codes),
                "mlapo_enabled": self.mlapo_enabled,
                "mlapo_disabled_by_nz": self.mlapo_disabled_by_nz,
                "pd_decode_node": self.pd_decode_node,
                "mlapo_eligible": self.mlapo_eligible}


@typed_inputs
def validate_kv_nz(config: KvNzPreconditions, *,
                   strict: bool = False) -> KvNzValidation:
    """Apply the vllm-ascend ``enable_kv_nz`` preconditions in source order."""
    if not isinstance(config, KvNzPreconditions):
        raise ModelingError("kv_layout_kv_nz_config_invalid")
    _bool("strict", strict)
    codes: list[str] = []
    enabled = config.enable_kv_nz
    pd_decode = bool(config.pd_disaggregated and config.is_kv_consumer)
    if enabled:
        if not config.is_mla:
            codes.append("kv_nz_requires_mla")
        if config.is_sparse_sfa:
            codes.append("kv_nz_rejects_sparse_sfa")
        if not config.pd_disaggregated:
            codes.append("kv_nz_requires_pd_deployment")
        elif not config.is_kv_consumer:
            codes.append("kv_nz_rejected_on_prefill_node")
    disabled_by_nz = bool(enabled and config.mlapo_eligible
                          and not config.nz_supports_mlapo)
    result = KvNzValidation(
        tuple(codes),
        mlapo_enabled=bool(config.mlapo_eligible and not disabled_by_nz),
        mlapo_disabled_by_nz=disabled_by_nz, pd_decode_node=pd_decode,
        mlapo_eligible=config.mlapo_eligible)
    if strict:
        result.raise_if_invalid()
    return result


@typed_inputs
@dataclass(frozen=True)
class KvNzPlane:
    """One typed NoPE/RoPE/scale plane with its exact 5-D physical shape."""

    name: str
    head_size: int
    dtype: str
    num_blocks: int
    num_kv_heads: int
    block_size: int
    scale_dim: int = 0
    scale_dtype: str = "fp32"
    pad_to_tile: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in {"nope", "rope"}:
            raise ModelingError(f"kv_nz_plane_name_invalid:{self.name}")
        for name in ("head_size", "num_blocks", "num_kv_heads", "block_size"):
            _int(name, getattr(self, name), minimum=1)
        _int("scale_dim", self.scale_dim, minimum=0)
        strict_dtype("dtype", self.dtype, reason="kv_nz")
        strict_dtype("scale_dtype", self.scale_dtype, reason="kv_nz")
        _bool("pad_to_tile", self.pad_to_tile)
        if self.scale_dim:
            if dtype_elem_bytes("dtype", self.dtype, reason="kv_nz") != 1.0:
                raise ModelingError("kv_nz_scale_plane_requires_one_byte_payload")
        if self.requires_padding and not self.pad_to_tile:
            raise ModelingError(
                "kv_nz_plane_view_incompatible:"
                f"{self.name}:{self.head_size}%{self.c0_elements}")

    @property
    def element_bytes(self) -> float:
        return dtype_elem_bytes("dtype", self.dtype, reason="kv_nz")

    @property
    def c0_elements(self) -> int:
        return fractal_c0_elements(self.dtype)

    @property
    def tiles_per_head(self) -> int:
        return ceil_div(self.head_size, self.c0_elements)

    @property
    def padded_head_size(self) -> int:
        return self.tiles_per_head * self.c0_elements

    @property
    def requires_padding(self) -> bool:
        return self.head_size % self.c0_elements != 0

    @property
    def view_compatible(self) -> bool:
        return not self.requires_padding

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (self.num_blocks, self.num_kv_heads, self.tiles_per_head,
                self.block_size, self.c0_elements)

    @property
    def logical_bytes(self) -> float:
        return (float(self.num_blocks * self.num_kv_heads * self.block_size
                      * self.head_size) * self.element_bytes)

    @property
    def physical_bytes(self) -> float:
        return (float(self.num_blocks * self.num_kv_heads
                      * self.padded_head_size * self.block_size)
                * self.element_bytes)

    @property
    def padding_bytes(self) -> float:
        return self.physical_bytes - self.logical_bytes

    @property
    def scale_bytes(self) -> float:
        return (float(self.num_blocks * self.block_size * self.num_kv_heads
                      * self.scale_dim)
                * dtype_elem_bytes("scale_dtype", self.scale_dtype,
                                   reason="kv_nz"))

    @property
    def total_bytes(self) -> float:
        return self.physical_bytes + self.scale_bytes

    @property
    def bytes_per_block(self) -> float:
        return self.total_bytes / self.num_blocks

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape),
                "dtype": self.dtype, "element_bytes": self.element_bytes,
                "c0_elements": self.c0_elements,
                "c0_bytes": FRACTAL_NZ_C0_BYTES,
                "tiles_per_head": self.tiles_per_head,
                "padded_head_size": self.padded_head_size,
                "view_compatible": self.view_compatible,
                "logical_bytes": self.logical_bytes,
                "physical_bytes": self.physical_bytes,
                "padding_bytes": self.padding_bytes,
                "scale_bytes": self.scale_bytes,
                "total_bytes": self.total_bytes}


@typed_inputs
@dataclass(frozen=True)
class KvNzLayout:
    """A complete MLA page: typed NoPE plane, optional RoPE, optional scales."""

    num_blocks: int
    num_kv_heads: int
    block_size: int
    nope_head_size: int
    nope_dtype: str = "bf16"
    rope_head_size: int = 0
    rope_dtype: str = "bf16"
    scale_dim: int = 0
    scale_dtype: str = "fp32"
    pad_to_tile: bool = False

    def __post_init__(self) -> None:
        for name in ("num_blocks", "num_kv_heads", "block_size",
                     "nope_head_size"):
            _int(name, getattr(self, name), minimum=1)
        _int("rope_head_size", self.rope_head_size, minimum=0)
        _int("scale_dim", self.scale_dim, minimum=0)
        _bool("pad_to_tile", self.pad_to_tile)
        strict_dtype("nope_dtype", self.nope_dtype, reason="kv_nz")
        strict_dtype("rope_dtype", self.rope_dtype, reason="kv_nz")
        strict_dtype("scale_dtype", self.scale_dtype, reason="kv_nz")
        _ = self.planes

    @property
    def nope(self) -> KvNzPlane:
        return KvNzPlane("nope", self.nope_head_size, self.nope_dtype,
                         self.num_blocks, self.num_kv_heads, self.block_size,
                         scale_dim=self.scale_dim,
                         scale_dtype=self.scale_dtype,
                         pad_to_tile=self.pad_to_tile)

    @property
    def rope(self) -> KvNzPlane | None:
        if not self.rope_head_size:
            return None
        return KvNzPlane("rope", self.rope_head_size, self.rope_dtype,
                         self.num_blocks, self.num_kv_heads, self.block_size,
                         pad_to_tile=self.pad_to_tile)

    @property
    def planes(self) -> tuple[KvNzPlane, ...]:
        rope = self.rope
        return (self.nope,) if rope is None else (self.nope, rope)

    @property
    def logical_bytes(self) -> float:
        return sum(plane.logical_bytes for plane in self.planes)

    @property
    def physical_bytes(self) -> float:
        return sum(plane.physical_bytes for plane in self.planes)

    @property
    def padding_bytes(self) -> float:
        return self.physical_bytes - self.logical_bytes

    @property
    def scale_bytes(self) -> float:
        return self.nope.scale_bytes

    @property
    def total_bytes(self) -> float:
        return self.physical_bytes + self.scale_bytes

    @property
    def bytes_per_block(self) -> float:
        return self.total_bytes / self.num_blocks

    @property
    def view_compatible(self) -> bool:
        return all(plane.view_compatible for plane in self.planes)

    @property
    def padding_fraction(self) -> float:
        return self.padding_bytes / self.physical_bytes if self.physical_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.nope.shape),
                "c0_elements": self.nope.c0_elements,
                "tiles_per_head": self.nope.tiles_per_head,
                "padded_head_size": self.nope.padded_head_size,
                "logical_bytes": self.logical_bytes,
                "physical_bytes": self.physical_bytes,
                "padding_bytes": self.padding_bytes,
                "scale_bytes": self.scale_bytes,
                "total_bytes": self.total_bytes,
                "bytes_per_block": self.bytes_per_block,
                "padding_fraction": self.padding_fraction,
                "view_compatible": self.view_compatible,
                "planes": [plane.to_dict() for plane in self.planes],
                "scale_plane": (None if not self.scale_dim else {
                    "shape": [self.num_blocks, self.block_size,
                              self.num_kv_heads, self.scale_dim],
                    "dtype": self.scale_dtype,
                    "element_bytes": dtype_elem_bytes(
                        "scale_dtype", self.scale_dtype, reason="kv_nz"),
                    "bytes": self.scale_bytes})}



_READ_EXPORTS = frozenset({"DEFAULT_CACHELINE_BYTES", "Mte2ReadModel",
                           "mte2_read_model"})
_TRANSFER_EXPORTS = frozenset({"BACKEND_SUPPORTS_FUSED_NZ", "NzReformatCost",
                               "REFORMAT_PATHS", "nd_to_nz_reformat_cost"})
_PD_EXPORTS = frozenset({"COMPONENT_MAJOR_CONNECTORS", "PdPageDescriptors",
                         "component_major_pd_comparison",
                         "pd_page_descriptors",
                         "supports_component_major_mla_pd"})


def __getattr__(name: str) -> Any:
    """Lazily re-export the companion read/transfer/PD records."""
    if name in _READ_EXPORTS:
        from mindie_sim import kv_nz_read_model
        return getattr(kv_nz_read_model, name)
    if name in _TRANSFER_EXPORTS:
        from mindie_sim import kv_nz_transfer
        return getattr(kv_nz_transfer, name)
    if name in _PD_EXPORTS:
        from mindie_sim import kv_pd_descriptors
        return getattr(kv_pd_descriptors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BACKEND_SUPPORTS_FUSED_NZ", "COMPONENT_MAJOR_CONNECTORS",
           "DEFAULT_CACHELINE_BYTES", "FRACTAL_NZ_C0_BYTES", "KV_NZ_TILE",
           "KvNzLayout", "KvNzPlane", "KvNzPreconditions", "KvNzValidation",
           "Mte2ReadModel", "NzReformatCost", "PdPageDescriptors",
           "component_major_pd_comparison", "fractal_c0_elements",
           "mte2_read_model", "nd_to_nz_reformat_cost",
           "pd_page_descriptors", "supports_component_major_mla_pd",
           "validate_kv_nz"]
