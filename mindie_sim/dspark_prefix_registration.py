"""Immutable registration evidence for DSpark draft companion pages.

[Role]
  Hold the accepted ``DSparkPrefixPage`` records, their region extents and the
  cache/group coverage inventory produced by ``dspark_prefix_page_bytes``.
  ``DSparkPrefixPlan`` revalidates every scalar claim against this frozen
  record, so a caller cannot forge ``num_blocks``, ``page_bytes`` or relax the
  registered ``require_hash_equals_block`` invariant.

[Invariants]
  * Every page is a validated ``DSparkPrefixPage``; its size and stride are
    exactly ``kernel_bytes * scale`` / ``kernel_stride_bytes * scale``.
  * Shape determines block coverage, scale, stride and physical capacity;
    replacement scalar claims never establish their own validity.
  * The retained loaded-page inventory must match every registered page.
  * Page/stride totals, object count and block-size equality are recomputed
    here, never trusted from the caller's mapping.
  * The records are frozen; later mutation is impossible.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import strict_bool, strict_int
from mindie_sim.schema.common import ModelingError


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="dspark")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="dspark")


@typed_inputs
@dataclass(frozen=True)
class DSparkRegisteredPage:
    """One accepted page plus its registration-time region extent."""

    page: Any
    scale: int
    size: int
    stride: int
    region_size: int

    def __post_init__(self) -> None:
        from mindie_sim.dspark_prefix_cache import DSparkPrefixPage
        if not isinstance(self.page, DSparkPrefixPage):
            raise ModelingError("dspark_prefix_page_invalid")
        for name in ("scale", "size", "stride", "region_size"):
            _int(f"registered_{name}", getattr(self, name), minimum=1)
        if self.size != self.page.kernel_bytes * self.scale:
            raise ModelingError("dspark_prefix_registered_page_size_mismatch")
        if self.stride != self.page.kernel_stride_bytes * self.scale:
            raise ModelingError("dspark_prefix_registered_page_stride_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {**self.page.to_dict(), "scale": self.scale, "size": self.size,
                "stride": self.stride, "region_size": self.region_size}

    def validate_geometry(self, num_blocks: int, block_size: int) -> None:
        """Re-derive the registration from the retained physical tensor."""
        _int("num_blocks", num_blocks, minimum=1)
        _int("block_size", block_size, minimum=1)
        page = self.page
        if page.shape0 % num_blocks:
            raise ModelingError("dspark_prefix_shape0_not_divisible_by_blocks")
        scale = page.shape0 // num_blocks
        if scale < 1 or scale * page.shape1 != block_size:
            raise ModelingError("dspark_prefix_kernel_pages_do_not_cover_block")
        if page.kernel_bytes < page.shape1 * page.row_bytes:
            raise ModelingError("dspark_prefix_page_capacity_too_small")
        if page.kernel_stride_bytes < page.kernel_bytes:
            raise ModelingError("dspark_prefix_kernel_stride_smaller_than_page")
        if scale > 1 and page.kernel_stride_bytes != page.kernel_bytes:
            raise ModelingError("dspark_prefix_scaled_pages_not_adjacent")
        if (self.scale, self.size, self.stride) != (
                scale, page.kernel_bytes * scale, page.kernel_stride_bytes * scale):
            raise ModelingError("dspark_prefix_registered_physical_geometry_mismatch")
        physical_extent = (page.shape0 - 1) * page.kernel_stride_bytes + page.kernel_bytes
        if self.region_size != physical_extent:
            raise ModelingError("dspark_prefix_registration_physical_region_extent")


@typed_inputs
@dataclass(frozen=True)
class DSparkPrefixRegistration:
    """Immutable validation evidence carried from registration into a plan."""

    pages: tuple[DSparkRegisteredPage, ...]
    num_blocks: int
    block_size: int
    hash_block_size: int
    page_bytes: int
    stride_bytes: int
    objects_per_key: int
    groups: tuple[int, ...]
    block_sizes: tuple[int, ...]
    draft_groups: tuple[int, ...]
    require_hash_equals_block: bool
    loaded_pages: tuple[Any, ...]

    def __post_init__(self) -> None:
        from mindie_sim.dspark_prefix_cache import DSparkPrefixPage
        if isinstance(self.pages, (str, bytes)) or not isinstance(
                self.pages, tuple) or not self.pages:
            raise ModelingError("dspark_prefix_registration_pages_invalid")
        for record in self.pages:
            if not isinstance(record, DSparkRegisteredPage):
                raise ModelingError("dspark_prefix_page_invalid")
        if not isinstance(self.loaded_pages, tuple) or not self.loaded_pages \
                or not all(isinstance(page, DSparkPrefixPage) for page in self.loaded_pages) \
                or tuple(record.page for record in self.pages) != self.loaded_pages:
            raise ModelingError("dspark_prefix_registration_loaded_cache_inventory")
        for name in ("num_blocks", "block_size", "hash_block_size",
                     "page_bytes", "stride_bytes", "objects_per_key"):
            _int(f"registered_{name}", getattr(self, name), minimum=1)
        _bool("registered_require_hash_equals_block",
              self.require_hash_equals_block)
        if self.block_size % self.hash_block_size:
            raise ModelingError("dspark_prefix_block_hash_alignment_invalid")
        for record in self.pages:
            record.validate_geometry(self.num_blocks, self.block_size)
        for name, value, minimum in (("groups", self.groups, 0),
                                     ("block_sizes", self.block_sizes, 1),
                                     ("draft_groups", self.draft_groups, 0)):
            if not isinstance(value, tuple) or not value or not all(
                    type(item) is int and item >= minimum for item in value):
                raise ModelingError(
                    f"dspark_prefix_registration_{name}_invalid")
        if len(set(self.groups)) != len(self.groups) or \
                list(self.groups) != sorted(self.groups):
            raise ModelingError(
                "dspark_prefix_registration_groups_not_canonical")
        page_groups = {record.page.group_id for record in self.pages}
        if page_groups != set(self.groups):
            raise ModelingError("dspark_prefix_registration_group_coverage")
        if self.draft_groups != self.groups:
            raise ModelingError("dspark_prefix_registration_draft_coverage")
        if min(self.groups) < 0 or max(self.groups) >= len(self.block_sizes):
            raise ModelingError("dspark_prefix_registration_group_range")
        if any(self.block_sizes[group] != self.block_size
               for group in self.groups):
            raise ModelingError("dspark_prefix_registration_block_size")
        if any(record.region_size != (self.num_blocks - 1) * record.stride
               + record.size for record in self.pages):
            raise ModelingError("dspark_prefix_registration_region_extent")
        if self.require_hash_equals_block and \
                self.block_size != self.hash_block_size:
            raise ModelingError(
                "dspark_prefix_hash_block_must_equal_logical_block")
        if sum(record.size for record in self.pages) != self.page_bytes:
            raise ModelingError("dspark_prefix_registration_page_bytes")
        if sum(record.stride for record in self.pages) != self.stride_bytes:
            raise ModelingError("dspark_prefix_registration_stride_bytes")
        if len(self.pages) != self.objects_per_key:
            raise ModelingError("dspark_prefix_registration_object_count")

    @property
    def coverage(self) -> dict[str, Any]:
        return {"groups": list(self.groups),
                "loaded_pages": [page.to_dict() for page in self.loaded_pages],
                "objects_per_key": self.objects_per_key,
                "pages": [record.to_dict() for record in self.pages],
                "draft_groups": list(self.draft_groups),
                "block_sizes": list(self.block_sizes)}

    def claims(self) -> dict[str, Any]:
        """Every scalar a caller could otherwise forge."""
        return {"num_blocks": self.num_blocks, "block_size": self.block_size,
                "hash_block_size": self.hash_block_size,
                "page_bytes": self.page_bytes,
                "stride_bytes": self.stride_bytes,
                "objects_per_key": self.objects_per_key,
                "draft_groups": tuple(self.draft_groups),
                "block_sizes": tuple(self.block_sizes),
                "require_hash_equals_block":
                    self.require_hash_equals_block}

    def to_dict(self) -> dict[str, Any]:
        return {**self.claims(), "draft_groups": list(self.draft_groups),
                "block_sizes": list(self.block_sizes),
                "coverage": self.coverage}


__all__ = ["DSparkPrefixRegistration", "DSparkRegisteredPage"]
