"""Component-major versus split MLA PD page descriptor accounting.

[Role]
  Price ``[all NoPE][all RoPE][padding]`` as one raw RDMA descriptor per page
  per layer against the legacy two-descriptor split, and decide which layout a
  declared PD connector can carry.

[Invariants]
  * Payload is conserved apart from declared component-major padding.
  * A custom connector module path always falls back to the split layout.
  * Non-PD deployments are unrestricted; descriptor setup and doorbell costs
    are finite and non-negative.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (optional_text, strict_bool, strict_int,
                                       strict_number)
from mindie_sim.schema.common import ModelingError


def _int(name, value, *, minimum=0):
    from mindie_sim.nr_s7_validate import strict_int
    return strict_int(name, value, minimum=minimum, reason="kv_layout")


def _bool(name, value):
    from mindie_sim.nr_s7_validate import strict_bool
    return strict_bool(name, value, reason="kv_layout")

COMPONENT_MAJOR_CONNECTORS = frozenset({"MooncakeConnectorV2",
                                        "MooncakePullConnector"})


@typed_inputs
@dataclass(frozen=True)
class PdPageDescriptors:
    """Per-page NDMA descriptor accounting for one PD page layout."""

    pages: int
    layers: int
    nope_bytes: int
    rope_bytes: int
    padding_bytes: int
    component_major: bool
    scale_bytes: int = 0
    per_descriptor_setup_us: float = 0.0
    doorbell_us: float = 0.0

    def __post_init__(self) -> None:
        _int("pages", self.pages, minimum=0)
        _int("layers", self.layers, minimum=0)
        _int("nope_bytes", self.nope_bytes, minimum=0)
        _int("rope_bytes", self.rope_bytes, minimum=0)
        _int("padding_bytes", self.padding_bytes, minimum=0)
        _int("scale_bytes", self.scale_bytes, minimum=0)
        _bool("component_major", self.component_major)
        strict_number("per_descriptor_setup_us", self.per_descriptor_setup_us,
                      reason="kv_layout")
        strict_number("doorbell_us", self.doorbell_us, reason="kv_layout")
        if self.nope_bytes + self.rope_bytes <= 0:
            raise ModelingError("kv_layout_pd_page_empty")
        if self.component_major and self.padding_bytes and self.rope_bytes == 0:
            raise ModelingError("kv_layout_pd_padding_without_rope")

    @property
    def component_count(self) -> int:
        return 1 if self.component_major or self.rope_bytes == 0 else 2

    @property
    def descriptors(self) -> int:
        return self.pages * self.layers * self.component_count

    @property
    def payload_bytes(self) -> int:
        # The FP32 scale plane is part of the typed c8 page in every layout.
        payload = self.nope_bytes + self.rope_bytes + self.scale_bytes
        if self.component_major:
            payload += self.padding_bytes
        return self.pages * self.layers * payload

    @property
    def startup_ms(self) -> float:
        return (self.descriptors * (self.per_descriptor_setup_us
                                    + self.doorbell_us) / 1000.0)

    def to_dict(self) -> dict[str, Any]:
        return {"descriptors": self.descriptors,
                "component_count": self.component_count,
                "payload_bytes": self.payload_bytes,
                "scale_bytes": self.scale_bytes,
                "startup_ms": self.startup_ms,
                "component_major": self.component_major}


@typed_inputs
def pd_page_descriptors(*, pages: int, layers: int, nope_bytes: int,
                        rope_bytes: int, padding_bytes: int = 0,
                        component_major: bool, scale_bytes: int = 0,
                        per_descriptor_setup_us: float = 0.0,
                        doorbell_us: float = 0.0) -> PdPageDescriptors:
    """One raw descriptor for component-major, two for the legacy split."""
    return PdPageDescriptors(
        pages=pages, layers=layers, nope_bytes=nope_bytes,
        rope_bytes=rope_bytes, padding_bytes=padding_bytes,
        component_major=component_major, scale_bytes=scale_bytes,
        per_descriptor_setup_us=per_descriptor_setup_us,
        doorbell_us=doorbell_us)


@typed_inputs
def supports_component_major_mla_pd(*, kv_connector: str | None,
                                    kv_connector_module_path: str | None,
                                    pd_disaggregated: bool) -> bool:
    """Whether PD transfer can carry ``[all NoPE][all RoPE][padding]`` raw."""
    _bool("pd_disaggregated", pd_disaggregated)
    optional_text("kv_connector", kv_connector, reason="kv_layout")
    optional_text("kv_connector_module_path", kv_connector_module_path,
                  reason="kv_layout")
    if not pd_disaggregated:
        return True
    if kv_connector_module_path is not None:
        return False
    return kv_connector in COMPONENT_MAJOR_CONNECTORS


@typed_inputs
def component_major_pd_comparison(*, pages: int, layers: int, nope_bytes: int,
                                  rope_bytes: int, padding_bytes: int = 0,
                                  scale_bytes: int = 0,
                                  per_descriptor_setup_us: float = 0.0,
                                  doorbell_us: float = 0.0) -> dict[str, Any]:
    """Price both layouts; payload is conserved against the physical page.

    The physical page is ``NoPE + RoPE + every typed plane`` (the FP32 scale
    plane included).  Comparing two equally scale-stripped totals would
    conserve a payload that does not exist, so the check is anchored on the
    full page, not merely on the other layout.
    """
    common = dict(pages=pages, layers=layers, nope_bytes=nope_bytes,
                  rope_bytes=rope_bytes, padding_bytes=padding_bytes,
                  scale_bytes=scale_bytes,
                  per_descriptor_setup_us=per_descriptor_setup_us,
                  doorbell_us=doorbell_us)
    major = pd_page_descriptors(component_major=True, **common)
    split = pd_page_descriptors(component_major=False, **common)
    physical_page_bytes = nope_bytes + rope_bytes + scale_bytes
    physical_payload_bytes = physical_page_bytes * pages * layers
    required_major = physical_payload_bytes + pages * layers * padding_bytes
    return {"component_major": major.to_dict(), "split": split.to_dict(),
            "descriptor_reduction": (split.descriptors / major.descriptors
                                     if major.descriptors else 1.0),
            "startup_saving_ms": split.startup_ms - major.startup_ms,
            "payload_delta_bytes": major.payload_bytes - split.payload_bytes,
            "physical_page_bytes": physical_page_bytes,
            "physical_payload_bytes": physical_payload_bytes,
            "payload_conserved": (
                split.payload_bytes == physical_payload_bytes
                and major.payload_bytes == required_major)}



__all__ = ["COMPONENT_MAJOR_CONNECTORS", "PdPageDescriptors", "component_major_pd_comparison", "pd_page_descriptors", "supports_component_major_mla_pd"]
