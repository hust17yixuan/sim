"""Counterfactual hardware scaling separated from fabric price arithmetic.

[Role] Scale one declared physical supply (HBM, cube, vector, links, on-chip
  buffers) and project a Case onto another SKU without mutating the caller.
[Invariants] Every declared affected resource scales together; integer
  capacities floor after scaling; L1/UB buffer capacity is real and drives the
  tiling/prefetch eligibility of the step model.
[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Returns fresh immutable records; caller objects untouched.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device, Hardware, Link


def _positive(name: str, value: Any, *, minimum: float = 0.0,
              strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"fabric_{name}_invalid:{value}")
    number = float(value)
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        raise ModelingError(f"fabric_{name}_invalid:{value}")
    return number

def _floor(value: float) -> int:
    return max(0, int(math.floor(float(value))))

def scale_hardware(target: Hardware | Any, *, hbm_bw: float = 1.0,
                   cube_tflops: float = 1.0, vector_ops: float = 1.0,
                   link_bw: float = 1.0, launch_us: float = 1.0,
                   hbm_bytes: float = 1.0, l1_bytes: float = 1.0,
                   ub_bytes: float = 1.0) -> Hardware | Any:
    """Scale one declared hardware supply at a time and return a new record."""
    factors = {"hbm_bw": hbm_bw, "cube_tflops": cube_tflops, "vector_ops": vector_ops,
               "link_bw": link_bw, "launch_us": launch_us, "hbm_bytes": hbm_bytes,
               "l1_bytes": l1_bytes, "ub_bytes": ub_bytes}
    for name, value in factors.items():
        _positive(name, value, strict=(name not in {"l1_bytes", "ub_bytes"}))
    if all(math.isclose(float(value), 1.0) for value in factors.values()):
        return target
    machine = target.machine if hasattr(target, "machine") else target
    if not isinstance(machine, Hardware):
        raise ModelingError(f"scale_hardware_requires_hardware_or_case:{type(target).__name__}")

    def device(item: Device) -> Device:
        peaks = tuple((name, float(value) * cube_tflops) for name, value in item.compute_peaks)

        def scaled_rate(name: str, value: float) -> float:
            if name in {"hbm", "memory"} or name.startswith(("hbm", "gm_hbm")):
                return value * hbm_bw
            if name == "vector" or name.startswith(("vector", "reg_", "vreg_", "ub_", "aiv")):
                return value * vector_ops
            return value

        buffers = tuple((name, _floor(value * (l1_bytes if name == "l1" else
                                                 ub_bytes if name == "ub" else 1.0)))
                        for name, value in item.buffers)
        return replace(item, memory_bytes=max(1, _floor(item.memory_bytes * hbm_bytes)),
                       compute_peaks=tuple(sorted(peaks)),
                       rates=tuple(sorted((name, float(scaled_rate(name, value)))
                                          for name, value in item.rates)),
                       buffers=tuple(sorted(buffers)),
                       launch_us=float(item.launch_us) * launch_us)

    def link(item: Link) -> Link:
        return replace(
            item,
            capacity_bytes_per_ms=_floor(item.capacity_bytes_per_ms * link_bw) or 1,
            switch_capacity_bytes_per_ms=(_floor(item.switch_capacity_bytes_per_ms * link_bw)
                                          if item.switch_capacity_bytes_per_ms else None),
            peak_capacity_bytes_per_ms=(_floor(item.peak_capacity_bytes_per_ms * link_bw)
                                        if item.peak_capacity_bytes_per_ms else None),
            endpoint_rates=tuple((rank, _floor(value * link_bw) or 1)
                                 for rank, value in item.endpoint_rates))

    options = tuple(sorted((key, (float(value) * link_bw
                                  if "bw" in key and key.endswith("_gbps") else value))
                           for key, value in machine.topology_options))
    scaled = replace(machine, dies=tuple(device(item) for item in machine.dies),
                     links=tuple(link(item) for item in machine.links),
                     topology_options=options)
    return replace(target, machine=scaled) if hasattr(target, "machine") else scaled

def project_case(case: Any, sku: str, *, num_devices: int | None = None,
                 ranks_per_node: int | None = None,
                 topology_preset: str | None = None) -> Any:
    """Project one Case onto another SKU without touching model or workload."""
    if not hasattr(case, "machine"):
        raise ModelingError("project_case_requires_case")
    from mindie_sim.compose import resolve_hardware_sku
    from mindie_sim.backends.ascend import materialize as ascend_materialize
    canonical = resolve_hardware_sku(str(sku))
    count = int(num_devices or max(len(case.machine.dies), case.runtime.parallel.world_size))
    if case.runtime.parallel.world_size > count:
        raise ModelingError(f"project_case_world_exceeds_target:"
                            f"{case.runtime.parallel.world_size}>{count}")
    request: dict[str, Any] = {"backend": "ascend", "sku": canonical,
                               "num_devices": count,
                               "ranks_per_node": int(ranks_per_node
                                                     or min(count, case.machine.ranks_per_node))}
    if topology_preset:
        request["topology_preset"] = str(topology_preset)
    return replace(case, machine=Hardware.from_mapping(ascend_materialize(request)))


__all__ = ["project_case", "scale_hardware"]
