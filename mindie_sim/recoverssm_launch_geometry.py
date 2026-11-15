"""RecoverSSM commit/verify kernel launch and program-grid geometry.

[Role]
  Reproduce the Ascend Triton commit kernel's tile selection and program grid
  from ``ops/triton/kda/recoverssm.py`` so Stage-2 latency separates fixed
  kernel launches from program-grid execution instead of multiplying launch
  overhead by a capacity ratio.

[Invariants]
  * ``block_k = next_pow2(key_dim)`` and the normal value tile starts at the
    source's 64 columns (doubled on A5 for the non-grouped kernel), then is
    capped by the power-of-two row budget of the unified buffer.
  * The boundary kernel uses six matrices per head (times the grouped head
    tile) and derives its own power-of-two value tile; on a 192 KiB UB with
    ``key_dim = 128`` the non-A5 tiles are exactly 64 and 32.
  * Program counts are ``ceil(value_dim / block_v) * head_groups * layers *
    batch``.  Fixed launches are plan=1, conv=1, commit=1 and boundary=1 only
    in align mode; the boundary kernel is launched on every align step and
    fast-exits non-crossing requests.
  * Tile geometry never changes a fixed launch count; a zero per-program cost
    makes the program-grid term zero, so a raw capacity ratio cannot pose as
    a measured latency ratio.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    ceil_div, next_power_of_two, strict_bool, strict_dtype, strict_int, strict_number,
)
from mindie_sim.schema.common import ModelingError

#: Mirrors the source constants.
RECOVERSSM_COMMIT_HEAD_TILE = 2
RECOVERSSM_COMMIT_UB_MATRICES = 2
RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES = 6
RECOVERSSM_COMMIT_UB_HEADROOM_ROWS = 2
RECOVERSSM_VALUE_TILE = 64
RECOVERSSM_VALUE_TILE_LARGE = 128
RECOVERSSM_LARGE_TILE_MIN_PROGRAMS = 96
RECOVERSSM_CONV_TILE = 4096
RECOVERSSM_CONV_STRIDED_TILE = 256
DEFAULT_UB_BYTES = 192 * 1024


def _tile_rows(ub_bytes: int, matrices: int, block_k: int,
               head_tile: int = 1) -> int:
    per_row = matrices * block_k * 4 * head_tile
    return max(1, ub_bytes // per_row - RECOVERSSM_COMMIT_UB_HEADROOM_ROWS)


def _power_tile(limit: int, cap: int) -> int:
    """Largest power of two at most both ``limit`` and ``cap``."""
    if limit < 1:
        return 1
    return min(cap, 1 << (limit.bit_length() - 1))


@typed_inputs
@dataclass(frozen=True)
class CommitLaunchGeometry:
    """One layer-group commit call's tiles, launches and program counts."""

    ub_bytes: int
    key_dim: int
    value_dim: int
    num_heads: int
    batch: int
    layers: int
    align_mode: bool
    is_a5: bool = False
    recurrent_dtype: str = "fp32"
    program_dispatch_us: float = 0.0
    conv_dim: int = 0
    conv_dim_stride: int = 1

    def __post_init__(self) -> None:
        strict_int("ub_bytes", self.ub_bytes, minimum=1)
        for name in ("key_dim", "value_dim", "num_heads", "batch", "layers"):
            strict_int(name, getattr(self, name), minimum=1)
        strict_bool("align_mode", self.align_mode)
        strict_bool("is_a5", self.is_a5)
        strict_dtype("recurrent_dtype", self.recurrent_dtype, reason="recoverssm")
        strict_int("conv_dim", self.conv_dim, minimum=0)
        strict_int("conv_dim_stride", self.conv_dim_stride, minimum=1)
        number = strict_number("program_dispatch_us", self.program_dispatch_us,
                               reason="recoverssm")
        if number is None:
            raise ModelingError("recoverssm_program_dispatch_us_invalid")

    @property
    def grouped_commit(self) -> bool:
        return (self.is_a5 and self.recurrent_dtype == "fp32"
                and self.key_dim == self.value_dim == 128
                and self.num_heads in (32, 64))

    @property
    def block_k(self) -> int:
        return next_power_of_two(self.key_dim)

    @property
    def head_groups(self) -> int:
        if self.grouped_commit:
            return self.num_heads // RECOVERSSM_COMMIT_HEAD_TILE
        return self.num_heads

    @property
    def commit_block_v(self) -> int:
        tile = RECOVERSSM_VALUE_TILE
        if self.grouped_commit:
            return min(next_power_of_two(self.value_dim), tile)
        if self.is_a5:
            tile *= 2
        rows = _tile_rows(self.ub_bytes, RECOVERSSM_COMMIT_UB_MATRICES,
                          self.block_k)
        tile = min(tile, _power_tile(rows, RECOVERSSM_VALUE_TILE_LARGE))
        return min(next_power_of_two(self.value_dim), tile)

    @property
    def boundary_block_v(self) -> int:
        head_tile = (RECOVERSSM_COMMIT_HEAD_TILE if self.grouped_commit else 1)
        rows = _tile_rows(self.ub_bytes,
                          RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES,
                          self.block_k, head_tile)
        tile = min(RECOVERSSM_VALUE_TILE, _power_tile(rows, RECOVERSSM_VALUE_TILE))
        return min(next_power_of_two(self.value_dim), tile)

    @property
    def ub_capacity_rows(self) -> tuple[int, int]:
        """Raw row capacity bounds (not the actual value tile)."""
        grouped = RECOVERSSM_COMMIT_HEAD_TILE if self.grouped_commit else 1
        normal = _tile_rows(self.ub_bytes, RECOVERSSM_COMMIT_UB_MATRICES,
                            self.block_k)
        boundary = _tile_rows(self.ub_bytes,
                              RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES,
                              self.block_k, grouped)
        return normal, boundary

    @property
    def commit_programs(self) -> int:
        return (ceil_div(self.value_dim, self.commit_block_v)
                * self.head_groups * self.layers * self.batch)

    @property
    def boundary_programs(self) -> int:
        if not self.align_mode:
            return 0
        return (ceil_div(self.value_dim, self.boundary_block_v)
                * self.head_groups * self.layers * self.batch)

    @property
    def plan_programs(self) -> int:
        return self.batch

    @property
    def conv_tile(self) -> int:
        """Source ``_conv_compaction_block_d``: 256 on non-A5 strided channels."""
        if self.is_a5 or self.conv_dim_stride == 1:
            return RECOVERSSM_CONV_TILE
        return RECOVERSSM_CONV_STRIDED_TILE

    @property
    def conv_programs(self) -> int:
        conv_dim = self.conv_dim or self.key_dim
        conv_tile = min(self.conv_tile, max(1, conv_dim))
        return ceil_div(conv_dim, conv_tile) * self.batch * self.layers

    @property
    def program_count(self) -> int:
        return (self.plan_programs + self.conv_programs + self.commit_programs
                + self.boundary_programs)

    @property
    def kernel_launches(self) -> int:
        """plan + conv + commit, plus the always-launched align boundary."""
        return 3 + (1 if self.align_mode else 0)

    @property
    def program_grid_ms(self) -> float:
        return self.program_count * self.program_dispatch_us / 1000.0

    @property
    def program_grid_ratio(self) -> float:
        normal = self.commit_programs
        boundary = self.boundary_programs or normal
        return boundary / normal if normal else 1.0

    @property
    def boundary_tile_slowdown(self) -> float:
        """Program-count ratio only; never a measured latency ratio."""
        return self.program_grid_ratio

    def to_dict(self) -> dict[str, Any]:
        normal_rows, boundary_rows = self.ub_capacity_rows
        return {"block_k": self.block_k, "head_groups": self.head_groups,
                "commit_block_v": self.commit_block_v,
                "boundary_block_v": self.boundary_block_v,
                "ub_capacity_rows": [normal_rows, boundary_rows],
                "commit_programs": self.commit_programs,
                "boundary_programs": self.boundary_programs,
                "conv_tile": self.conv_tile,
                "conv_programs": self.conv_programs,
                "program_count": self.program_count,
                "kernel_launches": self.kernel_launches,
                "program_grid_ms": self.program_grid_ms,
                "program_grid_ratio": self.program_grid_ratio,
                "grouped_commit": self.grouped_commit,
                "timing_model": "serial_estimate"}


@typed_inputs
def commit_launch_geometry(*, ub_bytes: int = DEFAULT_UB_BYTES,
                           key_dim: int, value_dim: int, num_heads: int,
                           batch: int, layers: int, align_mode: bool,
                           is_a5: bool = False,
                           recurrent_dtype: str = "fp32",
                           program_dispatch_us: float = 0.0,
                           conv_dim: int = 0,
                           conv_dim_stride: int = 1) -> CommitLaunchGeometry:
    """Construct the validated Stage-2 launch geometry."""
    return CommitLaunchGeometry(
        ub_bytes=ub_bytes, key_dim=key_dim, value_dim=value_dim,
        num_heads=num_heads, batch=batch, layers=layers,
        align_mode=align_mode, is_a5=is_a5,
        recurrent_dtype=recurrent_dtype,
        program_dispatch_us=program_dispatch_us, conv_dim=conv_dim,
        conv_dim_stride=conv_dim_stride)


__all__ = ["CommitLaunchGeometry", "DEFAULT_UB_BYTES",
           "RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES",
           "RECOVERSSM_COMMIT_HEAD_TILE", "RECOVERSSM_COMMIT_UB_HEADROOM_ROWS",
           "RECOVERSSM_COMMIT_UB_MATRICES", "RECOVERSSM_CONV_STRIDED_TILE",
           "RECOVERSSM_CONV_TILE",
           "RECOVERSSM_VALUE_TILE", "RECOVERSSM_VALUE_TILE_LARGE",
           "commit_launch_geometry"]
