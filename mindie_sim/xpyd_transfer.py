"""xPyD transfer-byte ledger and simultaneous-cut contention pricing.

[Role]
  Produce the page/shard-granular KV+index+state transfer ledger for one
  request handoff and price the simultaneous P-HBM, P-NIC, RoCE, D-NIC,
  D-HBM and staging cuts with a strict ``<1`` utilization margin.

[Invariants]
  * Token-paged KV+index bytes are charged once over the *actual* missing page
    interval ``[cached // page, (prompt-1) // page]``; fixed per-sequence state
    is charged once, and ``transfer_total_bytes`` is exactly conserved.
    Compression completion uses absolute token endpoints before capacity padding.
  * ``prefix_hash`` is a content identity derived only from supplied prefix
    tokens/digests; model geometry is never a prefix.  A cache epoch/digest
    mismatch is a miss, and an unverified partial cache fails closed.
  * A full destination cache needs content identity and complete coverage; an
    empty or incomplete page plan rejects instead of synthesizing ownership.
  * A scalar accepted-token yield is legal only inside the declared ``(K,
    bonus_token)`` support and is renewed through the one canonical law.
  * A cut at (or within 1e-9 of) saturation is never admitted and has an
    infinite M/D/1 queue; the KV wire cut obeys ``utilization_cap``.
  * Routed endpoint-port demands have their own capacities; the limiting port,
    not an unweighted NIC aggregate, sets admission and M/D/1 service.
  * A verification cycle can complete at most the requests present in it, so
    ``mu`` never exceeds ``batch / step``.  ``mu`` is requests/s, not tokens/s.

[Timeouts & Locks] Pure bounded arithmetic over a finite flow list.
[Side-Effect Policy] Fresh immutable outputs; caller Cases untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import math
from typing import Any

from mindie_sim.catalog.cache import (index_row_bytes, layer_cache_components,
                                      new_completed_rows)
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.parallel_bridge import validate_page_plan
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.xpyd_flow import Flow, progressive_max_min


DEFAULT_RHO = 0.9
_FIXED_COMPONENTS = {"recurrent_state", "conv_state", "compressor_state",
                     "index_compressor_state", "index_pool_tail"}


from mindie_sim.nr_s7_validate import _validate_nested, strict_int, strict_number, typed_inputs


def _positive(name: str, value: Any, *, strict: bool = True) -> float:
    number = strict_number(name, value, reason="xpyd")
    assert number is not None
    if strict and number <= 0.0:
        raise ModelingError(f"xpyd_{name}_invalid:{value}")
    return number


def _int(name: str, value: Any, minimum: int = 1) -> int:
    return strict_int(name, value, minimum=minimum, reason="xpyd")


def _optional_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return strict_int(name, value, minimum=0, reason="xpyd")


def _fraction(name: str, value: Any) -> float:
    number = _positive(name, value, strict=False)
    if number > 1.0:
        raise ModelingError(f"xpyd_{name}_invalid:{value}")
    return number


def _cache_extents(components: Mapping[str, float]) -> tuple[float, float, float]:
    """Separate paged KV, index and fixed state without counting any twice."""
    classes = (("gqa_kv", "mla_kv", "window", "compressed_kv", "draft_kv"),
               ("index_k", "index_pool_tail", "index_compressor_state"),
               ("recurrent_state", "conv_state", "compressor_state"))
    return tuple(math.fsum(float(components.get(key, 0.0)) for key in names)
                 for names in classes)


def _completed_capacity(plan, name, origin, end, block):
    """Complete absolute groups first, then pad only this interval's new rows.

    Subtracting padded endpoint capacities can hide new rows in a previously
    allocated partial block. Cumulative capacity relative to the row at origin
    instead partitions the transferred allocation across contiguous pages.
    """
    ratio = (max(1, plan.compress_ratio, plan.index_kpool)
             if name == "index_k" else plan.compress_ratio)
    rows = new_completed_rows(origin, max(0, end - origin), ratio)
    capacity = max(1, block // max(1, ratio)) if name == "compressed_kv" else 1
    return ((rows + capacity - 1) // capacity) * capacity


def _interval_components(model, plan, parallel, origin, end, block):
    """Keep window/block contracts while retaining absolute completion phase."""
    components = layer_cache_components(
        model, plan, parallel, end - origin, capacity_padded=True, interleave=block)
    for name in ("compressed_kv", "index_k"):
        if name in components:
            width = (index_row_bytes(model, plan.index_dim) if name == "index_k"
                     else cache_format(model, name).row_bytes(plan.cache_row_elements))
            components[name] = _completed_capacity(plan, name, origin, end, block) * width
    return components


def _page_extents(plan, components, pages, page_size, block, prompt, page_meta):
    """Place transferred windows at their absolute tail and fixed state once.

    KV uses allocated block rows; index uses actual completed rows. Compressed
    KV pads new absolute completed groups, not fictitious future tokens.
    Windows occupy trailing allocated pages; fixed endpoints travel once.
    """
    rows = {page: dict(page=page, kv_bytes=0.0, index_bytes=0.0,
                      state_bytes=0.0, metadata_bytes=page_meta) for page in pages}
    end = math.ceil(prompt / page_size) * page_size
    for name, value in components.items():
        if value <= 0:
            continue
        kv, index, state = _cache_extents({name: value})
        key = "kv_bytes" if kv else "index_bytes" if index else "state_bytes"
        if not (kv or index or state):
            continue
        window = int(plan.window or 0) if name in {"gqa_kv", "mla_kv", "window"} else 0
        start = max(pages[0] * page_size, end - math.ceil(window / block) * block) if window else pages[0] * page_size
        if name in _FIXED_COMPONENTS:
            weights = [int(page == pages[-1]) for page in pages]
        elif name in {"index_k", "compressed_kv"}:
            def completed(position):
                return _completed_capacity(plan, name, start, min(prompt, position), block)
            weights = [completed((page + 1) * page_size) - completed(page * page_size) for page in pages]
        else:
            weights = [max(0, min(end, (page + 1) * page_size) - max(start, page * page_size)) for page in pages]
        total = sum(weights)
        for page, weight in zip(pages, weights):
            rows[page][key] += float(value) * weight / total
    return tuple(rows.values())


@typed_inputs
def logical_transfer_bytes(case: CaseSpec | Mapping[str, Any], *,
                           prompt_tokens: int, output_tokens: int = 0,
                           destination_cached_tokens: int = 0,
                           block_size: int | None = None,
                           page_size: int | None = None,
                           shard_ranks: int | None = None,
                           metadata_bytes_per_page: float = 64.0,
                           epoch: int | None = 0,
                           prefix_tokens: Sequence[int] | None = None,
                           prefix_digest: str | None = None,
                           source_prefix_hash: str | None = None,
                           destination_prefix_hash: str | None = None,
                           source_epoch: int | None = None,
                           destination_epoch: int | None = None,
                           source_page_ids: Sequence[int] | None = None,
                           destination_page_ids: Sequence[int] | None = None,
                           allow_unverified_partial_cache: bool = False,
                           ) -> dict[str, Any]:
    """Unique logical KV + index + recurrent bytes for one request handoff."""
    parsed = parse_case(case)
    _validate_nested(parsed, "case")
    model = parsed.model
    prompt = _int("prompt_tokens", prompt_tokens)
    output = _int("output_tokens", output_tokens, minimum=0)
    cached = _int("destination_cached_tokens", destination_cached_tokens, minimum=0)
    block = _int("block_size", block_size if block_size is not None else parsed.runtime.kv_block_size)
    page = _int("page_size", page_size if page_size is not None else block)
    shards = _int("shard_ranks", shard_ranks if shard_ranks is not None
                  else max(1, int(parsed.runtime.parallel.tp)))
    page_meta = _positive("metadata_bytes_per_page", metadata_bytes_per_page,
                          strict=False)
    cached_request = min(cached, prompt)
    layer_count = max(1, len(model.layer_plans))
    # -- Prefix/epoch identity: content evidence only, never model geometry ----
    supplied_hash = str(prefix_digest or "")
    if prefix_tokens is not None:
        for token in prefix_tokens:
            _int("prefix_token", token, minimum=0)
        content = ",".join(str(int(token)) for token in prefix_tokens)
        supplied_hash = hashlib.sha256(
            f"prefix|{content}|{block}|{page}".encode()).hexdigest()
    if source_prefix_hash or destination_prefix_hash:
        source_hash = str(source_prefix_hash or supplied_hash or "")
        destination_hash = str(destination_prefix_hash or supplied_hash or "")
        identity_verified = bool(source_hash) and source_hash == destination_hash
        prefix_hash = source_hash or None
        identity_reason = (None if identity_verified
                           else "prefix_content_hash_mismatch")
    else:
        identity_verified = bool(supplied_hash)
        prefix_hash = supplied_hash or None
        identity_reason = (None if identity_verified
                           else "prefix_content_identity_missing")
    epoch = _optional_int("epoch", epoch)
    source_epoch = _optional_int("source_cache_epoch", source_epoch)
    destination_epoch = _optional_int("destination_cache_epoch",
                                      destination_epoch)
    epoch_matched = True
    if source_epoch is not None and destination_epoch is not None:
        epoch_matched = source_epoch == destination_epoch
        if not epoch_matched:
            cached_request = 0
            identity_verified = False
            identity_reason = "cache_epoch_mismatch"
    if cached_request > 0 and identity_reason == "prefix_content_hash_mismatch":
        cached_request = 0
    partial_unverified = bool(0 < cached_request < prompt and not identity_verified)
    if partial_unverified and not allow_unverified_partial_cache:
        # Partial cache reuse requires explicit content identity.
        cached_request = 0
    unique_pages = int(math.ceil(prompt / page)) if prompt else 0
    # Empty evidence means zero owned pages; only ``None`` selects the default plan.
    source_pages, destination_pages = validate_page_plan(
        source_page_ids, destination_page_ids, required_pages=unique_pages)
    if source_page_ids is not None and cached_request < prompt:
        needed = set(range(cached_request // page, (prompt - 1) // page + 1))
        lacking = tuple(sorted(needed - set(source_pages)))
        if lacking:
            raise ModelingError(
                f"bridge_page_plan_source_incomplete:{lacking}")
    if destination_page_ids is not None and cached_request > 0:
        prefix = set(range(cached_request // page))
        lacking = tuple(sorted(prefix - set(destination_pages)))
        if lacking:
            raise ModelingError(
                f"bridge_page_plan_prefix_missing:{lacking}")
    destination_complete = bool(
        destination_page_ids is None
        or set(destination_pages) == set(range(unique_pages)))
    missing = max(0, prompt - cached_request)
    unique_tokens = int(math.ceil(missing / block) * block) if missing else 0
    logical = replace(parsed.runtime.parallel, tp=1, dp=1, ep=1, pp=1,
                      cp=1, dcp=1, pcp=1, cp_plan="none", enable_dsa_cp=False)
    paged = {"gqa_kv": 0.0, "mla_kv": 0.0, "window": 0.0, "compressed_kv": 0.0,
             "draft_kv": 0.0, "index_k": 0.0, "index_pool_tail": 0.0,
             "recurrent_state": 0.0, "conv_state": 0.0,
             "compressor_state": 0.0, "index_compressor_state": 0.0}
    endpoint_components = []
    for plan in model.layer_plans:
        endpoint = layer_cache_components(
            model, plan, logical, prompt, capacity_padded=True, interleave=block) if missing else {}
        endpoint_components.append({k: v for k, v in endpoint.items() if k in _FIXED_COMPONENTS})
        components = _interval_components(
            model, plan, logical, cached_request, prompt, block)
        components.update(endpoint_components[-1])
        for name, value in components.items():
            if name in paged:
                paged[name] += float(value)
    kv = (paged["gqa_kv"] + paged["mla_kv"] + paged["window"]
          + paged["compressed_kv"] + paged["draft_kv"])
    index = (paged["index_k"] + paged["index_pool_tail"]
             + paged["index_compressor_state"])
    # Recurrent/SSM state adds one fixed per-sequence copy across the prompt span.
    recurrent = math.fsum(_cache_extents(comp)[2] for comp in endpoint_components)
    metadata = unique_pages * page_meta * layer_count
    total = kv + index + recurrent + metadata
    if missing:
        # Include any partial boundary page covering [cached, prompt).
        start_page = cached_request // page
        end_page = (prompt - 1) // page
        page_ids = tuple(range(start_page, end_page + 1))
        transfer_pages = len(page_ids)
        # Page-granularity transfer moves the full span covered by ``page_ids``.
        layer_extents = []
        for layer, plan in enumerate(model.layer_plans):
            components = _interval_components(
                model, plan, logical, start_page * page, prompt, block)
            components.update(endpoint_components[layer])
            pages = _page_extents(plan, components, page_ids, page, block, prompt, page_meta)
            layer_extents.append({
                "layer": layer,
                **{key: sum(row[key] for row in pages)
                   for key in ("kv_bytes", "index_bytes", "state_bytes")},
                "metadata_bytes": transfer_pages * page_meta,
                "page_extents": pages})
        transfer_kv = math.fsum(row["kv_bytes"] for row in layer_extents)
        transfer_index = math.fsum(row["index_bytes"] for row in layer_extents)
        paged_bytes = transfer_kv + transfer_index
        shard_bytes_value = paged_bytes / float(shards)
        shard_layout = tuple(
            (shard, start_page, transfer_pages, shard_bytes_value)
            for shard in range(shards))
        shard_bytes = (shard_bytes_value,) * shards
        state_shards = (recurrent / float(shards),) * shards
        transfer_aggregate = shard_bytes_value * shards
        transfer_metadata = transfer_pages * page_meta * layer_count
        fixed_state = recurrent
    else:
        transfer_pages = 0
        page_ids = ()
        transfer_kv = transfer_index = 0.0
        layer_extents = []
        shard_layout = ()
        shard_bytes = (0.0,) * shards
        state_shards = (0.0,) * shards
        transfer_aggregate = 0.0
        transfer_metadata = 0.0
        fixed_state = 0.0
    # One unique conserved ledger: paged shards + fixed state shards (summed
    # once) + metadata.  ``transfer_total_bytes`` is exactly the conserved sum.
    transfer_total = (sum(shard_bytes) + sum(state_shards) + transfer_metadata)
    if abs(transfer_total - (transfer_aggregate + fixed_state
                             + transfer_metadata)) > 1e-6 * max(1.0, transfer_total):
        raise ModelingError("xpyd_transfer_total_not_conserved")
    # ``prefix_hash`` is the cache key; ``prefix_content_hash`` tracks supplied content.
    declared_key = hashlib.sha256(
        f"declared_cache_key|{prompt}|{cached_request}|{block}|{page}|"
        f"{int(epoch or 0)}".encode()).hexdigest()
    zero_transfer_qualified = bool(missing == 0 and cached_request > 0
                                   and identity_verified
                                   and destination_complete)
    return {"prompt_tokens": prompt, "output_tokens": output,
            "destination_cached_tokens": cached_request,
            "requested_cached_tokens": cached,
            "unique_tokens": unique_tokens,
            "unique_pages": unique_pages, "block_size": block,
            "paged_kv_bytes": kv, "index_k_bytes": index,
            "recurrent_state_bytes": recurrent, "metadata_bytes": metadata,
            "total_bytes": total, "detached_output_tokens": output,
            "missing_tokens": missing, "page_size": page,
            "shard_ranks": shards, "missing_tokens_per_shard": missing,
            "transfer_pages": transfer_pages,
            "transfer_bytes": (shard_bytes[0] if shard_bytes else 0.0),
            "shard_bytes": shard_bytes,
            "state_shard_bytes": state_shards,
            "shard_bytes_aggregate": transfer_aggregate,
            "transferred_kv_bytes": transfer_kv,
            "transferred_index_bytes": transfer_index,
            "layer_extents": tuple(layer_extents),
            "fixed_state_transfer_bytes": fixed_state,
            "transfer_metadata_bytes": transfer_metadata,
            "transfer_total_bytes": transfer_total,
            "shard_layout": shard_layout, "page_ids": page_ids,
            "prefix_hash": declared_key, "prefix_content_hash": prefix_hash,
            "epoch": int(epoch or 0),
            "source_epoch": source_epoch,
            "destination_epoch": destination_epoch,
            "source_page_ids": source_pages,
            "destination_page_ids": destination_pages,
            "destination_page_coverage_complete": destination_complete,
            "epoch_matched": bool(epoch_matched),
            "prefix_identity_verified": bool(identity_verified),
            "prefix_identity_reason": identity_reason,
            "partial_cache_rejected": bool(partial_unverified
                                           and not allow_unverified_partial_cache),
            "zero_transfer_qualified": zero_transfer_qualified,
            "transfer_ms_at_zero_bw": 0.0 if missing == 0 else None,
            "layout_independent": True}


@typed_inputs
def expected_decode_cycles(osl: int, speculative_k: int = 0,
                           accept_rate: float | None = None,
                           useful_tokens_per_step: float | None = None, *,
                           bonus_token: bool = True) -> float:
    """Exact finite-OSL renewal from the one canonical accepted-token PMF.

    When ``(speculative_k, accept_rate)`` are declared, the PMF in
    :func:`mindie_sim.moe_mtp.speculative_step_pmf` drives both the yield
    ``g = E[Y]`` and this DP.  An explicit scalar override is validated
    against that PMF when both are supplied; with only the override it uses the
    exact two-point integer-step law with mean ``g``.
    """
    from mindie_sim.moe_mtp import speculative_step_pmf

    output = _int("osl", osl)
    draft = _int("speculative_k", speculative_k, minimum=0)
    override = None
    if useful_tokens_per_step is not None:
        override = _positive("useful_tokens_per_step", useful_tokens_per_step,
                             strict=False)
        if not math.isfinite(override) or override < 1.0:
            raise ModelingError(
                f"xpyd_useful_tokens_per_step_invalid:{useful_tokens_per_step}")
    alpha = None if accept_rate is None else _fraction("accept_rate", accept_rate)
    if draft >= 1 and alpha is not None:
        pmf = speculative_step_pmf(draft, alpha, bonus_token=bonus_token)
        distribution = [(index + 1, probability)
                        for index, probability in enumerate(pmf)
                        if probability > 0.0]
        if override is not None:
            mean = sum(tokens * probability
                       for tokens, probability in distribution)
            if abs(mean - override) > 1e-9:
                raise ModelingError(
                    f"xpyd_useful_tokens_conflict:{mean}!={override}")
    else:
        # Declared scalar yield must lie within [1, K + bonus_token].
        support_high = (float(draft + (1 if bonus_token else 0))
                        if draft >= 1 else 1.0)
        yield_per_cycle = (1.0 if override is None else override)
        if yield_per_cycle < 1.0 - 1e-12 or yield_per_cycle > support_high + 1e-12:
            raise ModelingError(
                f"xpyd_useful_tokens_support:{yield_per_cycle}!>{support_high}")
        if abs(yield_per_cycle - round(yield_per_cycle)) <= 1e-9:
            step = max(1, int(round(yield_per_cycle)))
            return float(math.ceil(output / step))
        low = max(1, int(math.floor(yield_per_cycle)))
        high = max(low, int(math.ceil(yield_per_cycle)))
        weight = yield_per_cycle - float(low)
        if high == low:
            distribution = [(low, 1.0)]
        else:
            distribution = [(low, 1.0 - weight), (high, weight)]
    cycles = [0.0] * (output + 1)
    for remaining in range(1, output + 1):
        cycles[remaining] = 1.0 + sum(
            probability * cycles[max(0, remaining - tokens)]
            for tokens, probability in distribution)
    return cycles[output]


@typed_inputs
def completed_requests_per_s(decode_batch: int, useful_tokens_per_step: float,
                             osl: int, step_ms: float, *,
                             speculative_k: int = 0,
                             accept_rate: float | None = None,
                             bonus_token: bool = True) -> float:
    """Decode completion rate in requests/s from finite-OSL renewal cycles.

    A request of ``OSL`` tokens needs exact renewal cycles; tokens accepted
    past the request's remaining output belong after termination.  At
    ``OSL=1`` this is exactly ``B/T``; ``completed_requests_per_s(8, 3.0, 4,
    1.0) == 4000.0``.
    """
    batch = _int("decode_batch", decode_batch)
    useful = _positive("useful_tokens_per_step", useful_tokens_per_step)
    output = _int("osl", osl)
    step = _positive("decode_step_ms", step_ms, strict=False)
    _int("speculative_k", speculative_k, minimum=0)
    if accept_rate is not None:
        _fraction("accept_rate", accept_rate)
    if step <= 0.0:
        return 0.0
    cycles = expected_decode_cycles(
        output, speculative_k, accept_rate, useful, bonus_token=bonus_token)
    return batch / (cycles * step / 1000.0)



@typed_inputs
def kv_transfer_contention(
    kv_bytes_per_request: float, *, n_prefill: int, n_decode: int,
    nic_bw_gbps: float, hbm_bw_bytes_per_s: float, p_world: int, d_world: int,
    decode_step_ms: float, osl: int, requests_per_s: float,
    roce_bw_gbps: float | None = None, roce_oversubscription: float = 1.0,
    staging_bytes: float | None = None, transfer_latency_ms: float = 0.1,
    p_hbm_fraction: float = 1.0, d_hbm_fraction: float = 1.0,
    utilization_cap: float = DEFAULT_RHO,
    shared_cut_bytes_per_s: float | None = None,
    port_demands: Mapping[str, float] | None = None,
    port_capacities: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Simultaneous cut proof for one declared KV handoff flow class."""
    kv = _positive("kv_bytes_per_request", kv_bytes_per_request, strict=False)
    np_ = _int("n_prefill", n_prefill)
    nd = _int("n_decode", n_decode)
    nic = _positive("nic_bw_gbps", nic_bw_gbps)
    hbm = _positive("hbm_bw_bytes_per_s", hbm_bw_bytes_per_s)
    pw = _int("p_world", p_world)
    dw = _int("d_world", d_world)
    step = _positive("decode_step_ms", decode_step_ms, strict=False)
    out_tokens = _int("osl", osl, minimum=0)
    rps = _positive("requests_per_s", requests_per_s, strict=False)
    oversub = _positive("roce_oversubscription", roce_oversubscription, strict=False)
    if oversub < 1.0:
        raise ModelingError(f"xpyd_roce_oversubscription_invalid:{oversub}")
    latency = _positive("transfer_latency_ms", transfer_latency_ms, strict=False)
    p_frac = _fraction("p_hbm_fraction", p_hbm_fraction)
    d_frac = _fraction("d_hbm_fraction", d_hbm_fraction)
    rho_cap = _positive("utilization_cap", utilization_cap)
    if rho_cap >= 1.0:
        raise ModelingError(f"xpyd_utilization_cap_not_stable:{rho_cap}")
    roce_bw = None if roce_bw_gbps is None else _positive("roce_bw_gbps", roce_bw_gbps, strict=False)
    staging_size = None if staging_bytes is None else _positive("staging_bytes", staging_bytes, strict=False)
    shared_cut = None if shared_cut_bytes_per_s is None else _positive("shared_cut_bytes_per_s", shared_cut_bytes_per_s)
    port_terms = {str(key): _positive("port_demand", value, strict=False) for key, value in dict(port_demands or {}).items()}
    port_caps = {str(key): _positive("port_capacity", value, strict=False) for key, value in dict(port_capacities or {}).items()}
    if kv == 0.0:
        return _zero_transfer_handoff(step)
    source_nic, destination_nic = np_ * nic * 1e9, nd * nic * 1e9
    p_hbm, d_hbm = np_ * pw * hbm * p_frac, nd * dw * hbm * d_frac
    roce = roce_bw * 1e9 / oversub if roce_bw is not None else min(source_nic, destination_nic)
    staging = staging_size / (latency / 1000.0) if staging_size is not None and latency > 0 else float("inf")
    resources = {"p_hbm": p_hbm, "p_nic": source_nic, "roce": roce, "d_nic": destination_nic, "d_hbm": d_hbm}
    if shared_cut is not None:
        resources["shared_cut"] = shared_cut
    if math.isfinite(staging):
        resources["staging"] = staging
    used_ports = {}
    for key, demand in port_terms.items():
        capacity = port_caps.get(key)
        if demand <= 0.0 or capacity is None or capacity <= 0.0:
            continue
        resources[key] = capacity
        used_ports[key] = demand
    coefficients = tuple([(name, kv) for name in resources
                          if name not in used_ports]
                         + [(key, demand)
                            for key, demand in used_ports.items()])
    allocation = progressive_max_min(
        resources, (Flow("kv_transfer", coefficients),))
    capacity_rps = float(allocation["rates"]["kv_transfer"])
    p_hbm_rps = p_hbm / kv
    d_hbm_rps = d_hbm / kv
    wire_rps = capacity_rps
    p_fraction_used = (rps * kv / p_hbm) if p_hbm > 0 else float("inf")
    d_fraction_used = (rps * kv / d_hbm) if d_hbm > 0 else float("inf")
    # Strict stability: a cut at (or within 1e-9 of) saturation has no finite
    # queue.  The previous ``> capacity`` test admitted exact NIC saturation.
    unstable_margin = 1.0 - 1e-9
    # The bottleneck cut's own service time defines the M/D/1 waiting time.
    # Using the NIC service for a RoCE/staging/shared cut underestimate was off
    # by the ratio of the two capacities (1000x in the RoCE witness).
    limiting_service_ms = (1000.0 / capacity_rps
                           if kv > 0.0 and capacity_rps > 0.0 else 0.0)
    wire_service_ms = limiting_service_ms
    rho_kv = (rps / capacity_rps) if capacity_rps > 0 else float("inf")
    if rho_kv >= unstable_margin:
        kv_queue_ms = float("inf")
        kv_wire_infeasible = True
    else:
        kv_queue_ms = rho_kv * limiting_service_ms / (2.0 * (1.0 - rho_kv))
        kv_wire_infeasible = rho_kv > rho_cap + 1e-12
    overloaded = (d_fraction_used >= unstable_margin
                  or p_fraction_used >= unstable_margin
                  or rps >= capacity_rps * unstable_margin
                  or kv_wire_infeasible)
    transfer_ms = max(kv / (nic * 1e9) * 1000.0 * max(1.0, np_ / nd),
                      (kv / p_hbm * 1000.0 if p_hbm > 0 else float("inf"))
                      + (kv / d_hbm * 1000.0 if d_hbm > 0 else float("inf")),
                      kv / roce * 1000.0 if roce > 0 else 0.0,
                      limiting_service_ms)
    if not math.isfinite(transfer_ms):
        transfer_ms = float("inf")
    jitter = (d_fraction_used / (1.0 - d_fraction_used)
              if d_fraction_used < unstable_margin else float("inf"))
    return {
        "wire_ms": wire_service_ms,
        "p_hbm_ms": kv / p_hbm * 1000.0 if p_hbm > 0 else float("inf"),
        "d_hbm_ms": kv / d_hbm * 1000.0 if d_hbm > 0 else float("inf"),
        "transfer_ms": transfer_ms, "concurrency": max(1.0, np_ / nd),
        "jitter_ratio": jitter if math.isfinite(jitter) else float("inf"),
        "tpot_jitter_ms": (None if overloaded else
                           step / (1.0 - d_fraction_used)
                           + (transfer_ms + latency) / max(1, out_tokens)),
        "wire_requests_per_s": wire_rps,
        "p_hbm_requests_per_s": p_hbm_rps,
        "d_hbm_requests_per_s": d_hbm_rps,
        "incoming_bytes_per_s": rps * kv,
        "p_source_fraction": p_fraction_used,
        "d_write_fraction": d_fraction_used,
        "overloaded": bool(overloaded), "feasible": not overloaded,
        "reason": ("kv_wire_utilization_cap_exceeded" if kv_wire_infeasible
                   else "d_hbm_or_p_hbm_overloaded" if overloaded else None),
        "handoff_ms": latency,
        "queue_ms": kv_queue_ms,
        "kv_wire_utilization": rho_kv,
        "kv_wire_service_ms": wire_service_ms,
        "limiting_cut_service_ms": limiting_service_ms,
        "limiting_capacity_rps": capacity_rps,
        "shared_cut_bytes_per_s": shared_cut,
        "kv_wire_requests_per_s": capacity_rps,
        "kv_wire_cap": rho_cap,
        "resource_cuts": dict(allocation["resources"]),
        "saturated_resources": list(allocation["saturated"]),
        "port_demands": dict(used_ports),
        "deadlock": bool(allocation["deadlock"]),
    }


def _zero_transfer_handoff(decode_step_ms: float) -> dict[str, Any]:
    """Full destination cache: no pages move, no wire or HBM cut is consumed."""
    return {"wire_ms": 0.0, "p_hbm_ms": 0.0, "d_hbm_ms": 0.0,
            "transfer_ms": 0.0, "concurrency": 1.0, "jitter_ratio": 0.0,
            "tpot_jitter_ms": float(decode_step_ms),
            "wire_requests_per_s": float("inf"),
            "p_hbm_requests_per_s": float("inf"),
            "d_hbm_requests_per_s": float("inf"),
            "incoming_bytes_per_s": 0.0, "p_source_fraction": 0.0,
            "d_write_fraction": 0.0, "overloaded": False, "feasible": True,
            "reason": None, "handoff_ms": 0.0, "queue_ms": 0.0,
            "kv_wire_utilization": 0.0, "kv_wire_service_ms": 0.0,
            "kv_wire_requests_per_s": float("inf"),
            "resource_cuts": {}, "saturated_resources": [], "deadlock": False}

def _nic_bw_gbps(case: CaseSpec) -> float:
    for channel in ("d2d_inter_node", "d2d_intra_node", "d2d_intra_8p", "d2d_intra_package"):
        values = [link.capacity_bytes_per_ms / 1e6 for link in case.machine.links
                  if link.channel == channel]
        if values:
            return float(min(values))
    return max(link.capacity_bytes_per_ms / 1e6 for link in case.machine.links)

@typed_inputs
def nz_transfer_end_reformat(
    *, pages: int, layers: int, nd_page_bytes: int, hbm_bytes_per_s: float,
    fused_op: bool = False, kernel_fixed_ms: float = 0.0, synchronize_ms: float = 0.0,
    nz_layout: Any = None, nd_scale_bytes: int = 0, configured_staging_bytes: int = 0,
) -> dict[str, Any]:
    """ND -> NZ transposition charged at the xPyD transfer end (D node)."""
    from mindie_sim.kv_layout_physics import nd_to_nz_reformat_cost
    return nd_to_nz_reformat_cost(
        pages=pages, layers=layers, nd_page_bytes=nd_page_bytes, hbm_bytes_per_s=hbm_bytes_per_s,
        fused_op=fused_op, kernel_fixed_ms=kernel_fixed_ms, synchronize_ms=synchronize_ms,
        nz_layout=nz_layout, nd_scale_bytes=nd_scale_bytes, configured_staging_bytes=configured_staging_bytes,
    ).to_dict()


@typed_inputs
def component_major_pd_transfer(
    *, pages: int, layers: int, nope_bytes: int, rope_bytes: int, padding_bytes: int = 0,
    scale_bytes: int = 0, per_descriptor_setup_us: float = 0.0, doorbell_us: float = 0.0,
    kv_connector: str | None = None, kv_connector_module_path: str | None = None,
    pd_disaggregated: bool = True,
) -> dict[str, Any]:
    """Descriptor/payload ledger selected by the declared PD connector (NoPE + RoPE + scale)."""
    from mindie_sim.kv_layout_physics import component_major_pd_comparison, supports_component_major_mla_pd
    plane_args = {
        k: strict_int(k, v, minimum=0, reason="xpyd")
        for k, v in (
            ("pages", pages), ("layers", layers), ("nope_bytes", nope_bytes),
            ("rope_bytes", rope_bytes), ("padding_bytes", padding_bytes), ("scale_bytes", scale_bytes),
        )
    }
    supported = supports_component_major_mla_pd(
        kv_connector=kv_connector, kv_connector_module_path=kv_connector_module_path, pd_disaggregated=pd_disaggregated,
    )
    result = component_major_pd_comparison(**plane_args, per_descriptor_setup_us=per_descriptor_setup_us, doorbell_us=doorbell_us)
    result["component_major_supported"] = supported
    result["selected"] = "component_major" if supported else "split"
    return result


@typed_inputs
def dspark_prefix_transfer_bytes(
    *, prompt_tokens: int, prefix_hit_tokens: int, block_size: int, draft_page_bytes: int,
    aux_layer_ids: Sequence[int], hidden_size: int, dtype: str, tp_size: int = 1,
    hash_block_size: int | None = None, block_hash_count: int | None = None,
    metadata_bytes_per_object: int = 0, exact: bool = True,
) -> dict[str, Any]:
    """Draft-only companion transfer under the shared strict restore contract."""
    from mindie_sim.dspark_prefix_cache import aux_feature_bytes
    from mindie_sim.nr_s7_validate import RestoreExtent
    for name, value, minimum in (
        ("prompt_tokens", prompt_tokens, 0), ("prefix_hit_tokens", prefix_hit_tokens, 0),
        ("block_size", block_size, 1), ("draft_page_bytes", draft_page_bytes, 0),
        ("hidden_size", hidden_size, 1), ("tp_size", tp_size, 1),
        ("metadata_bytes_per_object", metadata_bytes_per_object, 0),
    ):
        if isinstance(value, bool) or type(value) is not int or value < minimum:
            raise ModelingError(f"xpyd_dspark_{name}_invalid:{value!r}")
    hash_size = block_size if hash_block_size is None else hash_block_size
    hashes = prompt_tokens // block_size if block_hash_count is None else block_hash_count
    aux = tuple(aux_layer_ids)
    if not aux or not all(type(value) is int and value >= 0 for value in aux):
        raise ModelingError("xpyd_dspark_aux_layer_ids_invalid")
    blocks = RestoreExtent(
        prefix_hit_tokens=prefix_hit_tokens, prompt_tokens=prompt_tokens,
        block_size=block_size, hash_block_size=hash_size, block_hash_count=hashes, exact=exact,
    ).validated_blocks()
    per_rank = float(blocks * (draft_page_bytes + metadata_bytes_per_object))
    return {
        "full_blocks": blocks, "stored_tokens": blocks * block_size,
        "draft_kv_bytes": float(blocks * draft_page_bytes),
        "metadata_bytes": float(blocks * metadata_bytes_per_object),
        "per_rank_transfer_bytes": per_rank, "aggregate_transfer_bytes": per_rank * tp_size,
        "aux_scratch_bytes": aux_feature_bytes(layer_count=len(aux), tokens=blocks * block_size, hidden_size=hidden_size, dtype=dtype),
        "aux_scratch_persistent": False, "transfer_scope": "per_rank; aggregate multiplies by tp_size",
    }
