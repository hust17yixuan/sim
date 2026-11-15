"""ND -> NZ transfer-end reformat traffic, staging and capability gating.

[Role]
  Charge the xPyD decode-end transposition that turns ND pages into
  Fractal-NZ pages: the two-pass gather/scatter HBM traffic, the one-time
  synchronize barrier, and the reusable K/V staging high-water mark.

[Invariants]
  * This backend's NZ path is always the two-pass gather/scatter path; a
    fused ``2 * payload`` NZ transpose is rejected as an unsupported
    capability (the fused custom op concatenates GQA/MHA caches only).
  * Staging is allocated once for the whole page set, never per layer, and is
    de-duplicated against an explicitly configured staging buffer.
  * ``nz_page_bytes >= nd_payload + scale_bytes``: negative padding is
    impossible.
  * HBM and launch rates are finite and strictly positive.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.kv_layout_physics import KvNzLayout
from mindie_sim.nr_s7_validate import (
    positive_rate, strict_bool, strict_int, strict_number,
)
from mindie_sim.schema.common import ModelingError


def _int(name, value, *, minimum=0):
    from mindie_sim.nr_s7_validate import strict_int
    return strict_int(name, value, minimum=minimum, reason="kv_layout")

REFORMAT_PATHS = frozenset({"two_pass", "fused"})
BACKEND_SUPPORTS_FUSED_NZ = False


@typed_inputs
@dataclass(frozen=True)
class NzReformatCost:
    """ND -> NZ reformat at the xPyD transfer end (D node)."""

    pages: int
    layers: int
    nd_page_bytes: int
    hbm_bytes_per_s: float
    path: str
    kernel_fixed_ms: float
    synchronize_ms: float
    nz_page_bytes: float | None = None
    nd_scale_bytes: int = 0
    configured_staging_bytes: int = 0

    def __post_init__(self) -> None:
        _int("pages", self.pages, minimum=0)
        _int("layers", self.layers, minimum=0)
        _int("nd_page_bytes", self.nd_page_bytes, minimum=0)
        _int("nd_scale_bytes", self.nd_scale_bytes, minimum=0)
        _int("configured_staging_bytes", self.configured_staging_bytes,
             minimum=0)
        positive_rate("hbm_bytes_per_s", self.hbm_bytes_per_s,
                      reason="kv_layout")
        if not isinstance(self.path, str) or self.path not in REFORMAT_PATHS:
            raise ModelingError(f"kv_layout_reformat_path_invalid:{self.path}")
        strict_number("kernel_fixed_ms", self.kernel_fixed_ms,
                      reason="kv_layout")
        strict_number("synchronize_ms", self.synchronize_ms,
                      reason="kv_layout")
        if self.nz_page_bytes is not None:
            extent = strict_number("nz_page_bytes", self.nz_page_bytes,
                                   reason="kv_layout")
            assert extent is not None
            if extent < float(self.nd_page_bytes + self.nd_scale_bytes):
                raise ModelingError(
                    "kv_layout_reformat_extent_too_small:"
                    f"{extent}<{self.nd_page_bytes + self.nd_scale_bytes}")
        if self.path == "fused" and not BACKEND_SUPPORTS_FUSED_NZ:
            raise ModelingError("kv_layout_fused_nz_reformat_unsupported")

    @property
    def payload_bytes(self) -> float:
        return float(self.pages * self.layers
                     * (self.nd_page_bytes + self.nd_scale_bytes))

    @property
    def staging_bytes(self) -> float:
        """Reusable K/V staging: one page-set payload, not per layer."""
        return float(self.pages * (self.nd_page_bytes + self.nd_scale_bytes))

    @property
    def staging_reconciled_bytes(self) -> float:
        return max(0.0, self.staging_bytes - self.configured_staging_bytes)

    @property
    def hbm_traffic_bytes(self) -> float:
        return self.payload_bytes * (2.0 if self.path == "fused" else 4.0)

    @property
    def kernel_launches(self) -> int:
        return self.layers * (1 if self.path == "fused" else 2) + 1

    @property
    def hbm_ms(self) -> float:
        return self.hbm_traffic_bytes / self.hbm_bytes_per_s * 1000.0

    @property
    def fixed_ms(self) -> float:
        return self.kernel_launches * self.kernel_fixed_ms + self.synchronize_ms

    @property
    def total_ms(self) -> float:
        return self.hbm_ms + self.fixed_ms

    @property
    def nz_padding_bytes(self) -> float | None:
        if self.nz_page_bytes is None:
            return None
        return (self.nz_page_bytes - self.nd_page_bytes
                - self.nd_scale_bytes) * self.pages * self.layers

    def to_dict(self) -> dict[str, Any]:
        return {"pages": self.pages, "layers": self.layers,
                "path": self.path, "payload_bytes": self.payload_bytes,
                "hbm_traffic_bytes": self.hbm_traffic_bytes,
                "kernel_launches": self.kernel_launches,
                "hbm_ms": self.hbm_ms, "fixed_ms": self.fixed_ms,
                "total_ms": self.total_ms,
                "staging_bytes": self.staging_bytes,
                "staging_reconciled_bytes": self.staging_reconciled_bytes,
                "nz_page_bytes": self.nz_page_bytes,
                "nz_padding_bytes": self.nz_padding_bytes}


@typed_inputs
def nd_to_nz_reformat_cost(*, pages: int, layers: int, nd_page_bytes: int,
                           hbm_bytes_per_s: float,
                           fused_op: bool = False,
                           kernel_fixed_ms: float = 0.0,
                           synchronize_ms: float = 0.0,
                           nz_layout: KvNzLayout | None = None,
                           nd_scale_bytes: int = 0,
                           configured_staging_bytes: int = 0) -> NzReformatCost:
    """Charge the transfer-end transposition that turns ND pages into NZ pages."""
    _int("pages", pages, minimum=0)
    _int("layers", layers, minimum=0)
    _int("nd_page_bytes", nd_page_bytes, minimum=0)
    # A raw integer truthiness is not a capability declaration.
    strict_bool("fused_op", fused_op, reason="kv_layout")
    from mindie_sim.kv_layout_physics import KvNzLayout
    physical: float | None = None
    if nz_layout is not None:
        if not isinstance(nz_layout, KvNzLayout):
            raise ModelingError("kv_layout_reformat_nz_layout_invalid")
        if pages > nz_layout.num_blocks:
            raise ModelingError(
                f"kv_layout_reformat_pages_exceed_blocks:{pages}>{nz_layout.num_blocks}")
        physical = nz_layout.bytes_per_block
    return NzReformatCost(
        pages=pages, layers=layers, nd_page_bytes=nd_page_bytes,
        hbm_bytes_per_s=hbm_bytes_per_s,
        path="two_pass" if not fused_op else "fused",
        kernel_fixed_ms=kernel_fixed_ms, synchronize_ms=synchronize_ms,
        nz_page_bytes=physical, nd_scale_bytes=nd_scale_bytes,
        configured_staging_bytes=configured_staging_bytes)



__all__ = ["BACKEND_SUPPORTS_FUSED_NZ", "NzReformatCost", "REFORMAT_PATHS", "nd_to_nz_reformat_cost"]
