"""Physical bridge owners, replica delivery and transfer lifetimes.

[Role]
  Join unique source shards to every required destination copy, preserving
  absolute block ownership and charging one header per physical page message.
[Invariants]
  PCP producers are selected after the priced KV gather by absolute block ID;
  DCP membership is block % DCP. DP weights describe request routing only.
  Source head replicas are canonicalized; destination head replicas are kept.
  Stage/rank lifetimes use transferred extents, never resident-cache ratios.
[Timeouts & Locks] Bounded pure arithmetic; no locks or external calls.
[Side-Effect Policy] New records only; inputs are not mutated.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mindie_sim.parallel_placement import normalize_partition
from mindie_sim.schema.common import ModelingError


def _rank_slices(case: Any, item: Mapping[str, Any], layer: int,
                 pcp_lane: int, dcp_lane: int | None, component: str, *, canonical=True):
    """Unique producers or complete consumers, with request routing over DP.

    PP owns its actual layer slice; PCP/DCP owns the bridge's page slice.
    GQA heads are integral TP shards (choose one producer for replicated heads).
    MLA/index replicas have one source leader; recurrent state is TP-sharded.
    These are byte intervals, so different source/destination TP layouts are
    joined by intersection, never by an all-ranks equal-share assumption.
    """
    plan = item["plan"]
    partition = normalize_partition(case, int(plan["pp"]))
    stage, end = 0, partition[0]
    while layer >= end:
        stage += 1
        end += partition[stage]
    ids = tuple(item["device_ids"])
    cp_positions = {rank: lane for group in plan.get("dcp_groups", ())
                    for lane, rank in enumerate(group)}
    owners = [(ids[rank], row) for rank, row in enumerate(plan["rank_map"])
              if row[0] == stage and row[2] == pcp_lane
              and (dcp_lane is None or cp_positions.get(rank, 0) == dcp_lane)]
    if not owners:
        raise ModelingError("bridge_physical_owner_missing")
    layer_plan = case.model.layer_plans[layer]
    heads = (max(1, int(layer_plan.num_kv_heads))
             if component == "kv_bytes" and layer_plan.attention in {"gqa", "projection_only"} else 1)
    replicas = sorted({row[1] for _rank, row in owners})
    result = []
    for replica in replicas:
        members = [(rank, row[4]) for rank, row in sorted(owners, key=lambda pair: pair[1][4])
                   if row[1] == replica]
        seen = set()
        for position, (rank, tp) in enumerate(members):
            if component == "state_bytes" and layer_plan.attention in {"kda", "gdn"}:
                lo, hi = tp / int(plan["tp"]), (tp + 1) / int(plan["tp"])
                intervals = ((lo, hi),)
            elif component == "kv_bytes" and layer_plan.attention in {"gqa", "projection_only"}:
                width = math.ceil(heads / int(plan["tp"]))
                start = (tp * width if heads >= int(plan["tp"])
                         else tp * heads // int(plan["tp"]))
                owned = set(range(start, min(start + width, heads)))
                selected = owned - seen if canonical else owned
                # Each TP shard is contiguous, including after removal of a
                # replicated prefix. Join one interval, not every head pair.
                intervals = ((min(selected) / heads, (max(selected) + 1) / heads),) if selected else ()
                seen.update(owned)
            else:
                intervals = ((0.0, 1.0),) if not canonical or position == 0 else ()
            for lo, hi in intervals:
                result.append((int(rank), lo, hi, 1.0 / len(replicas)))
        if component == "kv_bytes" and heads > 1 and len(seen) != heads:
            raise ModelingError("bridge_kv_head_ownership_incomplete")
    return result


def _join_slices(edges, left, right, payload: float, *, same_replica=False):
    for src, lo, hi, alpha in left:
        for dst, start, end, beta in right:
            overlap = max(0.0, min(hi, end) - max(lo, start))
            if overlap <= 0.0:
                continue
            # Remap keeps the request on the same DP replica.  Source/dest
            # lists have matching DP order; the caller joins one replica at a time.
            amount = payload * overlap * (alpha if same_replica else alpha * beta)
            edges[(src, dst)] = edges.get((src, dst), 0.0) + amount


def _context_lane(plan, page, component, spec):
    if component == "state_bytes":
        return None
    if component == "index_bytes" and spec.cp_plan != "dsa_cp":
        return None
    return page % max(1, int(plan["dcp"]))


def physical_bridge_edges(case, bridge, transfer, left, right, *, remap=False):
    """Reconcile unique data, destination copies and physical page messages.

    A PCP gather has materialized complete KV pages on the prefill lanes.
    Canonical page producers use absolute block % PCP, independently of cache
    filtering or caller list order. A DCP lane remains block % DCP. Each
    destination head/index replica consumes an interval, even when another
    destination has consumed that same interval already.
    """
    data_edges, headers, source_retained, destination_retained = {}, {}, {}, {}
    payload = bridge.kv_bytes + bridge.index_bytes + bridge.state_bytes
    unique = transfer["shard_bytes_aggregate"] + transfer["fixed_state_transfer_bytes"]
    if not math.isclose(payload, unique, rel_tol=1e-12, abs_tol=1e-6):
        raise ModelingError("bridge_transfer_extent_not_conserved")
    pages = tuple(sorted(page for bucket in bridge.source_pages for page in bucket))
    for component, expected in (("kv_bytes", bridge.kv_bytes), ("index_bytes", bridge.index_bytes),
                                ("state_bytes", bridge.state_bytes)):
        if not math.isclose(sum(row[component] for row in transfer["layer_extents"]), expected,
                            rel_tol=1e-12, abs_tol=1e-6):
            raise ModelingError("bridge_layer_extent_not_conserved")
    for extent in transfer["layer_extents"]:
        page_rows = extent.get("page_extents", ())
        if page_rows:
            if sorted(row["page"] for row in page_rows) != list(pages):
                raise ModelingError("bridge_page_extent_incomplete")
            for component in ("kv_bytes", "index_bytes", "state_bytes", "metadata_bytes"):
                if not math.isclose(sum(row[component] for row in page_rows), extent[component],
                                    rel_tol=1e-12, abs_tol=1e-6):
                    raise ModelingError("bridge_page_extent_not_conserved")
    # PCP replica coordinates never come from the bridge's DCP page buckets.
    targets = (tuple(range(bridge.destination_spec.active_pcp_lanes)) if remap else (0,))
    page_lookup = {row["layer"]: {p["page"]: p for p in row.get("page_extents", ())}
                   for row in transfer["layer_extents"]}
    slices = {}
    joins = {}
    expected_data = 0.0

    def owners(item, layer, lane, cp, component, canonical):
        key = (id(item), layer, lane, cp, component, canonical)
        if key not in slices:
            slices[key] = _rank_slices(case, item, layer, lane, cp, component,
                                       canonical=canonical)
        return slices[key]

    dp_of = {rank: row[1] for rank, row in zip(right["device_ids"], right["plan"]["rank_map"])}

    def join(a, b, value):
        key = (tuple(a), tuple(b))
        if key in joins:
            return {edge: fraction * value for edge, fraction in joins[key].items()}
        edges = {}
        if remap:
            for replica in range(int(right["plan"]["dp"])):
                _join_slices(edges, [v for v in a if dp_of[v[0]] == replica],
                             [v for v in b if dp_of[v[0]] == replica], 1.0,
                             same_replica=True)
        else:
            _join_slices(edges, a, b, 1.0)
        joins[key] = edges
        return {edge: fraction * value for edge, fraction in edges.items()}

    for page in pages:
        for extent in transfer["layer_extents"]:
            layer = int(extent["layer"])
            page_extent = page_lookup[layer].get(page)
            for target in targets:
                messages = {}
                for component in ("kv_bytes", "index_bytes", "state_bytes"):
                    value = (float(page_extent[component]) if page_extent is not None
                             else float(extent[component]) / len(pages))
                    if value <= 0:
                        continue
                    cp = _context_lane(right["plan"], page, component, bridge.destination_spec)
                    a = (owners(right, layer, 0, cp, component, True) if remap else
                         owners(left, layer, page % int(left["plan"]["pcp"]),
                                _context_lane(left["plan"], page, component, bridge.source_spec), component, True))
                    b = owners(right, layer, target, cp, component, False)
                    if not math.isclose(sum((hi - lo) * share for _, lo, hi, share in a), 1.0):
                        raise ModelingError("bridge_source_extent_not_conserved")
                    moved = join(a, b, value)
                    required = value * math.fsum(hi - lo for _, lo, hi, beta in b) / int(right["plan"]["dp"])
                    if not math.isclose(sum(moved.values()), required, rel_tol=1e-12, abs_tol=1e-6):
                        raise ModelingError("bridge_destination_copy_incomplete")
                    expected_data += required
                    for edge, amount in moved.items():
                        data_edges[edge] = data_edges.get(edge, 0.0) + amount
                    # Each physical source/destination pair owns one header,
                    # irrespective of how many KV/index/state intervals it carries.
                    share = 1.0 / int(right["plan"]["dp"])
                    if not remap:
                        share /= int(left["plan"]["dp"])
                    messages.update((edge, share) for edge in moved)
                    if not remap:
                        for rank, lo, hi, _share in a:
                            source_retained[rank] = source_retained.get(rank, 0.0) + value * (hi - lo)
                    for rank, lo, hi, _share in b:
                        destination_retained[rank] = destination_retained.get(rank, 0.0) + value * (hi - lo)
                if not remap:
                    if not messages and float(extent["metadata_bytes"]) > 0:
                        cp = page % int(right["plan"]["dcp"])
                        a = owners(left, layer, page % int(left["plan"]["pcp"]),
                                   page % int(left["plan"]["dcp"]), "metadata_bytes", True)
                        b = owners(right, layer, target, cp, "metadata_bytes", True)
                        messages = join(a, b, 1.0)
                    for edge, share in messages.items():
                        headers[edge] = headers.get(edge, 0.0) + float(extent["metadata_bytes"]) / len(pages) * share
    if remap:
        for target in targets:
            a = owners(right, 0, 0, 0, "metadata_bytes", True)
            b = owners(right, 0, target, 0, "metadata_bytes", True)
            for edge, value in join(a, b, bridge.metadata_bytes / len(targets)).items():
                headers[edge] = headers.get(edge, 0.0) + value
    edges = dict(data_edges)
    for edge, value in headers.items():
        edges[edge] = edges.get(edge, 0.0) + value
    if not math.isclose(sum(data_edges.values()), expected_data, rel_tol=1e-12, abs_tol=1e-6):
        raise ModelingError("bridge_physical_data_not_conserved")
    return {"edges": edges, "data_edges": data_edges,
            "unique_logical_bytes": float(unique), "destination_data_bytes": expected_data,
            "destination_replica_bytes": expected_data - unique,
            "header_bytes": math.fsum(headers.values()),
            "total_bytes": math.fsum(edges.values()),
            "source_retained": source_retained, "destination_retained": destination_retained}


def rank_lifetime_peaks(case, bridge, transfer):
    """Actual per-rank retention plus the declared largest-edge staging bound.

    DP routing probabilities are removed for memory: a routed request needs
    its complete allocation on the rank it selects. Wire envelopes are not
    retained KV; the separate 4096-B control allocation is live in each buffer.
    """
    from mindie_sim.parallel_placement import build_placement, with_world
    def instance(spec, start):
        return {"plan": build_placement(with_world(case, spec), spec).to_dict(),
                "device_ids": tuple(range(start, start + spec.world_size))}
    p, d = bridge.source_spec, bridge.destination_spec
    left, right = instance(p, 0), instance(d, p.world_size)
    row = physical_bridge_edges(case, bridge, transfer, left, right)
    remap = (physical_bridge_edges(case, bridge, transfer, right, right, remap=True)
             if bridge.ownership_mode == "replicated" else None)
    outgoing, incoming = {}, {}
    for (src, dst), amount in row["data_edges"].items():
        conditional = amount * p.dp * d.dp
        outgoing[src] = max(outgoing.get(src, 0.0), conditional)
        incoming[dst] = max(incoming.get(dst, 0.0), conditional)
    if remap:
        for (src, dst), amount in remap["data_edges"].items():
            if src != dst:
                incoming[dst] = max(incoming.get(dst, 0.0), amount * d.dp)
                incoming[src] = max(incoming.get(src, 0.0), amount * d.dp)
    retained = remap["destination_retained"] if remap else row["destination_retained"]
    control = 8192.0 if bridge.metadata_bytes > 0 else 0.0
    prefill = tuple(row["source_retained"].get(r, 0.0) + outgoing.get(r, 0.0) + control
                    for r in left["device_ids"])
    decode = tuple(retained.get(r, 0.0) + incoming.get(r, 0.0) + control
                   for r in right["device_ids"])
    return {"prefill": prefill, "decode": decode,
            "owners": tuple(r - p.world_size for r, amount in retained.items() if amount > 0),
            "source_retention": max(row["source_retained"].values(), default=0.0) + control / 2,
            "destination_retention": max(retained.values(), default=0.0) + control / 2,
            "handoff_staging": max((*outgoing.values(), *incoming.values()), default=0.0) + control / 2}
