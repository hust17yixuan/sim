"""Multi-tier outbound-ratio and A5 EP MC2 analytical models.

[Role]
  Materialize declared Ethernet/UB tier domains as physical links, expose an
  explicit directed EP all-to-all transport plan, and keep the older
  group-level effective-bandwidth budget for comparison.

[Invariants]
  * Tier links are non-overlapping physical groups of the declared width; an
    EP pair uses the narrowest domain that contains both. A tier that does not
    enlarge the domain for the selected EP is not duplicated.
  * A shared-domain link, a per-port injection plane and a per-pair direction
    are distinct resources. The transport plan reports aggregate binding work
    separately from the full Engine span.
  * The legacy ``analyze_ep_mc2_communication`` result is an explicitly
    labelled group-level analytic budget, not a topology service span; its
    startup defaults are caller-visible placeholders, not T06-identified
    parameters. Missing chunk/credit/startup inputs stay unknown.
  * Precision sizes are explicit: MXFP8 block scaling is 33/32 bytes per value,
    FP8/INT8 are one byte and BF16/FP16 are two bytes.
  * No serving floor, fitted residual or measurement-presence check is applied.

[Timeouts & Locks] Pure bounded arithmetic and bounded EP expansion.
[Side-Effect Policy] No I/O, device context or mutable global state.
"""
from __future__ import annotations

from dataclasses import replace
import math
from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.parallel import ParallelSpec
from .communication import MessageFlow
from .topology import Topology


EP_TIER_PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "a5_pcie_16p": (
        {"name": "intra_4p", "max_ranks": 4, "bw_gbps": 288.0},
        {"name": "intra_8p", "max_ranks": 8, "bw_gbps": 128.0},
        {"name": "intra_16p", "max_ranks": 16, "bw_gbps": 116.0},
        {"name": "inter_node", "max_ranks": 8192, "bw_gbps": 50.0},
    ),
    "a5_ub_pod": (
        {"name": "superpod", "max_ranks": 8192, "bw_gbps": 800.0},
    ),
    "a3_superpod": (
        {"name": "intra_node", "max_ranks": 16, "bw_gbps": 392.0},
        {"name": "superpod", "max_ranks": 384, "bw_gbps": 392.0},
        {"name": "inter_pod", "max_ranks": 8192, "bw_gbps": 50.0},
    ),
}

_BYTES_PER_ELEMENT = {
    "mxfp8": 33.0 / 32.0,
    "mx_fp8": 33.0 / 32.0,
    "mxfp8_e4m3": 33.0 / 32.0,
    "fp8": 1.0,
    "int8": 1.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
}


def _dtype_bytes(dtype: Any, label: str) -> float:
    key = str(dtype).strip().lower().replace("-", "_")
    if key not in _BYTES_PER_ELEMENT:
        raise ModelingError(f"ep_mc2_unsupported_{label}_dtype:{dtype}")
    return _BYTES_PER_ELEMENT[key]


def _non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ModelingError(f"{label}_invalid:{value}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ModelingError(f"{label}_invalid:{value}") from None
    if not math.isfinite(result) or result < 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return result


def _positive(value: Any, label: str) -> float:
    result = _non_negative(value, label)
    if result <= 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return result


def _tiers_for(preset: str, tiers: Sequence[Mapping[str, Any]] | None
               ) -> tuple[dict[str, Any], ...]:
    if tiers is None:
        if preset not in EP_TIER_PRESETS:
            raise ModelingError(f"ep_tier_preset_unknown:{preset}")
        return EP_TIER_PRESETS[preset]
    if isinstance(tiers, (str, bytes)) or not isinstance(tiers, Sequence) or not tiers:
        raise ModelingError("ep_tier_sequence_invalid")
    normalized = []
    previous = 0
    for index, tier in enumerate(tiers):
        if not isinstance(tier, Mapping):
            raise ModelingError(f"ep_tier_record_invalid:{index}")
        unknown = set(tier) - {"name", "max_ranks", "bw_gbps", "latency_us"}
        if unknown:
            raise ModelingError(f"ep_tier_field_unknown:{sorted(unknown)[0]}")
        max_ranks = tier.get("max_ranks")
        if (not isinstance(max_ranks, int) or isinstance(max_ranks, bool)
                or max_ranks <= previous):
            raise ModelingError(f"ep_tier_max_ranks_invalid:{index}")
        bw_gbps = _positive(tier.get("bw_gbps"), f"ep_tier_bw_gbps_invalid:{index}")
        latency = _non_negative(tier.get("latency_us", 0.0),
                                f"ep_tier_latency_us_invalid:{index}")
        name = str(tier.get("name", f"tier_{index}"))
        normalized.append({"name": name, "max_ranks": max_ranks,
                           "bw_gbps": bw_gbps, "latency_us": latency})
        previous = max_ranks
    return tuple(normalized)


def compute_speculative_accept_step(accept_rate: float, draft_tokens: int) -> float:
    """Expected accepted tokens per verification step.

    ``A5_EP建模.xlsx`` evaluates ``(1 - a^(d+1)) / (1 - a)`` for
    ``a=0.75`` and ``d=3`` to exactly ``2.734375``.
    """
    rate = _non_negative(accept_rate, "speculative_accept_rate")
    if rate > 1.0:
        raise ModelingError("speculative_accept_rate_invalid")
    if (not isinstance(draft_tokens, int) or isinstance(draft_tokens, bool)
            or draft_tokens < 0):
        raise ModelingError("speculative_draft_tokens_invalid")
    if math.isclose(rate, 1.0):
        return float(draft_tokens + 1)
    return (1.0 - rate ** (draft_tokens + 1)) / (1.0 - rate)


def compute_ep_effective_bandwidth(
    ep: int,
    tiers: Sequence[Mapping[str, Any]] | None = None,
    *,
    bw_utilization: float = 1.0,
    preset: str = "a5_pcie_16p",
) -> dict[str, Any]:
    """Return the bottleneck effective all-to-all bandwidth for ``ep`` ranks."""
    if not isinstance(ep, int) or isinstance(ep, bool) or ep < 2:
        raise ModelingError(f"ep_effective_bandwidth_requires_ep_at_least_two:{ep}")
    utilization = _positive(bw_utilization, "ep_bw_utilization_invalid")
    if utilization > 1.0:
        raise ModelingError("ep_bw_utilization_invalid")
    normalized = _tiers_for(str(preset), tiers)
    active: list[dict[str, Any]] = []
    first = normalized[0]
    if first["max_ranks"] > 1:
        outbound = (min(ep, first["max_ranks"]) - 1) / ep
        raw = first["bw_gbps"] / outbound
        active.append({
            "index": 0, "name": first["name"],
            "max_ranks": first["max_ranks"], "bw_gbps": first["bw_gbps"],
            "outbound_ratio": outbound, "raw_effective_bw_gbps": raw,
            "effective_bw_gbps": raw * utilization,
        })
    for index in range(1, len(normalized)):
        previous = normalized[index - 1]
        if ep <= previous["max_ranks"]:
            break
        tier = normalized[index]
        outbound = (ep - previous["max_ranks"]) / ep
        raw = tier["bw_gbps"] / outbound
        active.append({
            "index": index, "name": tier["name"],
            "max_ranks": tier["max_ranks"], "bw_gbps": tier["bw_gbps"],
            "outbound_ratio": outbound, "raw_effective_bw_gbps": raw,
            "effective_bw_gbps": raw * utilization,
        })
    if not active and len(normalized) == 1:
        # A single declared domain still serves EPs above its nominal maximum;
        # the source keeps the same ``(ep - 1) / ep`` outbound fraction.
        outbound = (ep - 1) / ep
        raw = first["bw_gbps"] / outbound
        active.append({
            "index": 0, "name": first["name"],
            "max_ranks": first["max_ranks"], "bw_gbps": first["bw_gbps"],
            "outbound_ratio": outbound, "raw_effective_bw_gbps": raw,
            "effective_bw_gbps": raw * utilization,
        })
    if not active:
        raise ModelingError(f"ep_effective_bandwidth_no_active_tier:{ep}")
    bottleneck = min(active, key=lambda row: row["raw_effective_bw_gbps"])
    for row in active:
        row["bottleneck"] = row is bottleneck
    raw_effective = float(bottleneck["raw_effective_bw_gbps"])
    return {
        "ep": ep,
        "preset": str(preset),
        "bw_utilization": utilization,
        "outbound_ratio": float(bottleneck["outbound_ratio"]),
        "raw_effective_bw_gbps": raw_effective,
        "effective_bw_gbps": raw_effective * utilization,
        "tier_breakdown": active,
        "tiers": active,
    }


def ep_tier_links(
    ep: int,
    tiers: Sequence[Mapping[str, Any]] | None = None,
    *,
    preset: str = "a5_pcie_16p",
    semantics: str = "per_port",
) -> tuple[Link, ...]:
    """Materialize nested EP tier domains as physical link records.

    Each tier contributes non-overlapping groups of its own width.  A pair
    inside one 4P/8P/... group uses that group's declared rate; a pair that
    only shares the larger domain uses the larger (usually slower) tier.  The
    link semantics is explicit because a per-rank injection port and a shared
    forwarding plane are different resources with different contention.
    """
    if not isinstance(ep, int) or isinstance(ep, bool) or ep < 2:
        raise ModelingError(f"ep_tier_topology_requires_ep_at_least_two:{ep}")
    if semantics not in {"shared", "per_port", "per_pair", "mesh"}:
        raise ModelingError(f"ep_tier_semantics_unknown:{semantics}")
    normalized = _tiers_for(str(preset), tiers)
    links = []
    previous_width = 1
    for tier in normalized:
        width = min(ep, int(tier["max_ranks"]))
        # A tier that does not enlarge the physical domain for this EP is a
        # duplicate of the previous (narrower) tier.  Keep the narrower one.
        if width <= previous_width:
            continue
        previous_width = width
        if width < 2:
            continue
        for first in range(0, ep, width):
            members = tuple(range(first, min(first + width, ep)))
            if len(members) < 2:
                continue
            links.append(Link(
                name=f"ep_tier:{tier['name']}:{first}",
                capacity_bytes_per_ms=float(tier["bw_gbps"]) * 1e6,
                members=members,
                latency_us=float(tier.get("latency_us", 0.0)),
                channel=f"ep_{tier['name']}",
                semantics=semantics,
            ))
    if not links:
        raise ModelingError(f"ep_tier_topology_empty:{ep}")
    return tuple(links)


def _tier_site_width(ep: int, tiers, preset: str) -> int:
    """Declared on-node width, not merely the largest named EP tier.

    ``a3_superpod`` has a 16-rank node followed by a 384-rank superpod; the
    superpod link must still connect different nodes, so its site width is 16,
    not 384.  A named superpod stops the on-node expansion; otherwise the last
    tier before the scale-out marker is the node width.
    """
    previous = 1
    for tier in _tiers_for(str(preset), tiers):
        width = min(ep, int(tier["max_ranks"]))
        text = str(tier["name"]).lower().replace("-", "_")
        if any(marker in text for marker in ("inter_node", "inter_pod")):
            return max(1, previous)
        if "superpod" in text:
            return max(1, previous) if previous > 1 else max(1, width)
        if width > previous:
            previous = width
    return max(1, previous)


def ep_tier_machine(
    ep: int,
    tiers: Sequence[Mapping[str, Any]] | None = None,
    *,
    preset: str = "a5_pcie_16p",
    semantics: str = "per_port",
    sku: str = "Ascend950DT_9581",
    ranks_per_node: int | None = None,
) -> Hardware:
    """Declared A5 device records plus the selected EP tier links."""
    links = ep_tier_links(ep, tiers, preset=preset, semantics=semantics)
    site = (int(ranks_per_node) if ranks_per_node is not None
            else _tier_site_width(ep, tiers, preset))
    if site < 1:
        raise ModelingError(f"ep_tier_ranks_per_node_invalid:{site}")
    base = Hardware.from_mapping({
        "backend": "ascend", "sku": sku,
        "num_devices": ep, "ranks_per_node": min(ep, site),
    })
    return replace(base, links=links)


def ep_alltoall_transport(
    ep: int,
    per_source_bytes: float,
    *,
    tiers: Sequence[Mapping[str, Any]] | None = None,
    preset: str = "a5_pcie_16p",
    semantics: str = "per_port",
    chunk_bytes: float | None = None,
    receive_capacity_bytes=None,
    hop_startup_us: float = 0.0,
    ranks_per_node: int | None = None,
    max_ranks: int = 128,
):
    """Build an explicit EP all-to-all transport over actual tier domains.

    ``per_source_bytes`` is the total payload one source sends to all peers;
    the uniform peer share is ``per_source_bytes / (ep - 1)``.  The returned
    plan separates the aggregate shared-link lower bound from the full
    chunk/credit span and never treats one group's duration as a link rate.
    """
    if not isinstance(ep, int) or isinstance(ep, bool) or ep < 2:
        raise ModelingError(f"ep_transport_requires_ep_at_least_two:{ep}")
    if ep > int(max_ranks):
        raise ModelingError(
            "ep_transport_topology_requires_domain_aggregation:"
            f"{ep}>{int(max_ranks)}")
    total = _non_negative(per_source_bytes, "ep_transport_per_source_bytes")
    machine = ep_tier_machine(
        ep, tiers, preset=preset, semantics=semantics,
        ranks_per_node=ranks_per_node)
    topology = Topology(machine, ParallelSpec(tp=ep))
    share = total / (ep - 1)
    flows = tuple(MessageFlow(source, destination, share)
                  for source in range(ep)
                  for destination in range(ep) if source != destination)
    plan = topology.transport_plan(
        flows, chunk_bytes=chunk_bytes,
        receive_capacity_bytes=receive_capacity_bytes,
        hop_startup_us=hop_startup_us)
    return {
        "ep": ep,
        "preset": str(preset),
        "sharing": semantics,
        "per_source_bytes": total,
        "peer_share_bytes": share,
        "topology": topology,
        "plan": plan,
        "links": tuple(link.to_dict() for link in machine.links),
        "lower_bound": plan.lower_bound(),
        "stage_count": plan.stage_count,
        "unknowns": dict(plan.unknowns),
        "coverage": "explicit_directed_ep_domains_not_group_effective_bw",
    }


def analyze_ep_mc2_communication(
    ep_values: Sequence[int] = (2, 4, 8, 16, 32, 64, 128, 256),
    batch_size: int = 32,
    hidden_size: int = 7168,
    top_k: int = 8,
    num_moe_layers: int = 61,
    dispatch_dtype: str = "mxfp8",
    combine_dtype: str = "bf16",
    eplb_peak_to_avg: float = 1.2,
    accept_rate: float = 0.75,
    draft_tokens: int = 3,
    n_hops: int = 1,
    per_hop_latency_us: float = 1.5,
    switch_base_latency_us: float = 3.0,
    software_launch_latency_us: float = 7.0,
    bw_utilizations: Sequence[float] = (0.55, 0.65, 0.75, 0.85),
    preset: str = "a5_ub_pod",
    tiers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze MC2 dispatch/combine communication over EP and bandwidth tiers."""
    for label, value in (("batch_size", batch_size), ("hidden_size", hidden_size),
                         ("top_k", top_k), ("num_moe_layers", num_moe_layers)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelingError(f"ep_mc2_{label}_invalid:{value}")
    if (not isinstance(n_hops, int) or isinstance(n_hops, bool) or n_hops < 0):
        raise ModelingError(f"ep_mc2_n_hops_invalid:{n_hops}")
    per_hop = _non_negative(per_hop_latency_us, "ep_mc2_per_hop_latency_us_invalid")
    switch = _non_negative(switch_base_latency_us, "ep_mc2_switch_base_latency_us_invalid")
    software = _non_negative(software_launch_latency_us,
                             "ep_mc2_software_launch_latency_us_invalid")
    peak_to_avg = _positive(eplb_peak_to_avg, "ep_mc2_eplb_peak_to_avg_invalid")
    dispatch_bytes = _dtype_bytes(dispatch_dtype, "dispatch")
    combine_bytes = _dtype_bytes(combine_dtype, "combine")
    if isinstance(ep_values, (str, bytes)) or not isinstance(ep_values, Sequence):
        raise ModelingError("ep_mc2_ep_values_invalid")
    for ep in ep_values:
        if not isinstance(ep, int) or isinstance(ep, bool) or ep < 2:
            raise ModelingError(f"ep_mc2_ep_value_invalid:{ep}")
    if (isinstance(bw_utilizations, (str, bytes))
            or not isinstance(bw_utilizations, Sequence)):
        raise ModelingError("ep_mc2_bw_utilizations_invalid")
    utilizations = []
    for utilization in bw_utilizations:
        value = _positive(utilization, "ep_mc2_bw_utilization_invalid")
        if value > 1.0:
            raise ModelingError("ep_mc2_bw_utilization_invalid")
        utilizations.append(value)
    if draft_tokens:
        accept_step = compute_speculative_accept_step(accept_rate, draft_tokens)
    else:
        accept_step = 1.0
    startup_us = n_hops * per_hop + switch + software
    dispatch_per_layer = (batch_size * hidden_size * top_k
                          * dispatch_bytes * peak_to_avg)
    combine_per_layer = (batch_size * hidden_size * top_k
                         * combine_bytes * peak_to_avg)
    rows = []
    for ep in ep_values:
        for utilization in utilizations:
            tier = compute_ep_effective_bandwidth(
                ep, tiers, bw_utilization=utilization, preset=preset)
            effective = tier["effective_bw_gbps"]
            dispatch_us = startup_us + (dispatch_per_layer / (effective * 1e9)) * 1e6
            combine_us = startup_us + (combine_per_layer / (effective * 1e9)) * 1e6
            total_comm_ms = (dispatch_us + combine_us) * num_moe_layers / 1000.0
            wire_service_ms = (
                (dispatch_per_layer + combine_per_layer)
                / (effective * 1e9) * 1e6 * num_moe_layers / 1000.0)
            rows.append({
                "ep": ep,
                "bw_utilization": utilization,
                "outbound_ratio": tier["outbound_ratio"],
                "raw_effective_bw_gbps": tier["raw_effective_bw_gbps"],
                "effective_bw_gbps": effective,
                "dispatch_us": dispatch_us,
                "combine_us": combine_us,
                "total_comm_ms": total_comm_ms,
                # This excludes the declared startup term; it is still a
                # group-level budget because one effective bandwidth replaces
                # the actual route/sharing domains.
                "wire_service_lower_bound_ms": wire_service_ms,
                "tpot_comm_ms": total_comm_ms / accept_step,
            })
    return {
        "preset": str(preset),
        "accept_step": accept_step,
        "startup_latency_us": startup_us,
        "startup_parameters": {
            "per_hop_latency_us": per_hop,
            "switch_base_latency_us": switch,
            "software_launch_latency_us": software,
            "source": "caller_visible_analytic_inputs_not_T06_identified",
            "omitted_arguments_use_function_defaults": True,
            "boundary": "one additive startup term per dispatch/combine call",
        },
        "dispatch_bytes_per_layer": dispatch_per_layer,
        "combine_bytes_per_layer": combine_per_layer,
        "num_moe_layers": num_moe_layers,
        "ep_values": list(ep_values),
        "bw_utilizations": utilizations,
        "rows": rows,
        "results": rows,
        "ep_results": rows,
        "coverage": "group_level_analytic_budget_not_a_topology_service_span",
        "topology_example": "use ep_alltoall_transport() for explicit domains",
        "unknowns": {
            "protocol_startup_us": (
                "analytic defaults are caller-visible placeholders, not a "
                "T06 identified per-resource service"),
            "receiver_credit_bytes": "not declared by this group-level budget",
            "per_hop_route": "collapsed into one effective group bandwidth",
        },
    }
