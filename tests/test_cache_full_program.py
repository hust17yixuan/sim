"""Complete cache data-path: typed ranges, event-time state and Engine program.

The tests execute real Engine work. They do not read weights or start devices.
Build order is intentionally different from ready order in the ordering cases.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim.prediction.cache_execution import (
    CacheAccessOperation, CacheAccessRequest, CacheExecution,
)
from mindie_sim.prediction.cache_path import (
    CacheAccessGeometry, CacheAccessProgram, CacheGeometry, CachePolicy,
    CacheState, cache_work_stages,
)
from mindie_sim.prediction.cache_geometry import (
    Span, byte_spans_of_line_spans, full_line_spans, intersect_spans,
    lines_of_byte_spans, normalize_spans, range_count, subtract_spans, span,
    take_first_spans, union_spans,
)
from mindie_sim.prediction.cache_representation import CacheLowering, cache_identity
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.workload import RequestView

ROOT = Path(__file__).resolve().parents[1]
PACKED_MLA_CASE = (ROOT / "examples/cases/longctx-decode-20261030/"
                   "v41-a5-tp1-dp8-ep8.json")


def geometry(capacity=256):
    return CacheGeometry(capacity_bytes=capacity, line_bytes=128,
                         transfer_bytes=32)


def engine(pool_bytes=1 << 20, mte2_rate=2000.0):
    return Engine([Resource("hbm", 1000.0, unit="bytes"),
                   Resource("mte2", mte2_rate), Resource("mte3", 2000.0),
                   Resource("vector", 1000.0)],
                  pools={"ub": pool_bytes})


def packed_builder():
    case = parse_case(json.loads(PACKED_MLA_CASE.read_text()))
    plan = case.model.layer_plans[0]
    return (Builder("l0", plan, 0,
                    (RequestView("r0", 1, 100, False, "decode"),),
                    case.model, "decode"), case, plan)


def test_typed_geometry_produces_compact_transactions_not_per_byte():
    contiguous = CacheAccessGeometry(rows=1, row_bytes=300)
    summary = contiguous.summary(128)
    assert summary.line_count == 3 and summary.logical_bytes == 300
    assert summary.partial_first is False and summary.partial_last is True

    strided = CacheAccessGeometry(rows=4, row_bytes=64, stride_bytes=128)
    summary = strided.summary(128)
    assert summary.line_count == 4 and summary.repeated is True

    paged = CacheAccessGeometry(rows=3, row_bytes=100, stride_bytes=256,
                                page_bytes=1024)
    summary = paged.summary(128)
    assert summary.line_count == 3 and summary.page_count == 1

    rmw = CacheAccessGeometry(offset_bytes=20, rows=2, row_bytes=100,
                              stride_bytes=128,
                              partial_line="read_modify_write")
    summary = rmw.summary(128, write=True)
    assert summary.line_count == 2 and summary.rmw_read_lines == 2
    masked = CacheAccessGeometry(offset_bytes=20, rows=2, row_bytes=100,
                                 stride_bytes=128, partial_line="masked")
    assert masked.summary(128, write=True).rmw_read_lines == 0


def test_event_time_state_overrides_build_order_and_competitor():
    store = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(write_policy="write_back"))
    load = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=False)

    # Same build order [store, load]; load ready first misses before store.
    state = CacheState(geometry(128))
    e = engine()
    r_store = e.submit("r_store", release=10)
    r_load = e.submit("r_load", release=0)
    events = CacheExecution(state).access(
        load, e, "load", r_load), CacheExecution(state).access(
        store, e, "store", r_store)
    e.advance(until=events, first=False)
    assert dict(e.serviced)["hbm"] == 128.0  # load fill; store retains dirty
    assert state.resident("A") == (1, 1)

    # Same build order; store ready first makes the later load a hit.
    state = CacheState(geometry(128))
    e = engine()
    r_store = e.submit("r_store", release=0)
    r_load = e.submit("r_load", release=10)
    events = CacheExecution(state).access(
        store, e, "store", r_store), CacheExecution(state).access(
        load, e, "load", r_load)
    e.advance(until=events, first=False)
    assert dict(e.serviced)["hbm"] == 0.0
    assert state.resident("A") == (1, 1)


def test_shared_hbm_competitor_conserves_work_without_changing_state_order():
    state = CacheState(geometry(256))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=False)
    e = engine()
    competitor = e.submit("external_hbm", {"hbm": 128.0},
                          release=0, latency=0.0)
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "read", ready)
    e.advance(until=(done,), first=False)
    assert dict(e.serviced)["hbm"] == 256.0  # competitor + miss fill
    assert competitor.done and state.resident("A") == (1, 0)


def test_event_execution_binds_staging_and_rejects_small_pool():
    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=False)
    with pytest.raises(ModelingError):
        CacheExecution(state, pool=None).access(
            request, engine(), "no_pool", engine().submit("r"))
    small = engine(pool_bytes=64)
    with pytest.raises(ExecutionError):
        CacheExecution(state).access(
            request, small, "small", small.submit("r"))
    enough = engine(pool_bytes=128)
    ready = enough.submit("ready", release=0)
    done = CacheExecution(state).access(request, enough, "normal", ready)
    enough.advance(until=(done,), first=False)
    assert dict(enough.peak_allocated)["ub"] == 128.0
    assert dict(enough.allocated)["ub"] == 0


def test_fill_publication_and_counters_happen_at_completion_events():
    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=False)
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "read", ready)
    e.advance(horizon=0.1)
    # Reserved but not yet filled; no hit and no counter.
    assert state.resident_bytes == 0
    assert state.snapshot()["reserved_bytes"] == 128
    assert state.counters["accesses"] == 0
    e.advance(horizon=0.15)
    # Fill published at the fill event; deliver/finish is still in flight.
    assert state.resident_bytes == 128
    assert state.snapshot()["reserved_bytes"] == 0
    assert state.counters["accesses"] == 0
    e.advance(until=(done,), first=False)
    assert state.counters["accesses"] == 1
    assert state.resident("A") == (1, 0)


def test_reservation_cancel_restores_existing_dirty_prefix():
    state = CacheState(geometry(256))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    reservation = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(write_policy="write_back"))
    assert state.resident("A") == (0, 0)
    reservation.cancel()
    assert state.resident("A") == (1, 1)
    assert state.resident_bytes == 128 and state.dirty_bytes == 128


def test_reserved_capacity_competition_is_explicit_and_recoverable():
    state = CacheState(geometry(128))
    first = state.prepare_access("A", CacheAccessGeometry(rows=1, row_bytes=128))
    with pytest.raises(ModelingError):
        state.prepare_access("B", CacheAccessGeometry(rows=1, row_bytes=128))
    first.publish()
    first.finish()
    second = state.prepare_access("B", CacheAccessGeometry(rows=1, row_bytes=128))
    # Capacity is full; B's reservation evicts clean A (no writeback).
    second.publish()
    second.finish()
    assert state.resident("B") == (1, 0)
    assert state.resident("A") == (0, 0)


def test_program_and_execution_cold_rmw_and_external_source_event():
    geometry_rmw = CacheAccessGeometry(
        rows=1, row_bytes=64, partial_line="read_modify_write")
    state = CacheState(geometry(128))
    reservation = state.prepare_access(
        "A", geometry_rmw, write=True,
        policy=CachePolicy(write_policy="write_through"))
    result = reservation.result
    reservation.cancel()
    assert result.hbm_read_bytes == 128 and result.hbm_write_bytes == 128
    e = engine()
    program = CacheAccessProgram(result, onchip_resource="mte2", pool="ub")
    ready = e.submit("ready", release=0)
    e.advance(until=(program.start(e, "program", ready),), first=False)
    assert dict(e.serviced)["hbm"] == 256.0
    assert dict(e.serviced)["mte2"] == 64.0

    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", geometry_rmw, write=True,
        policy=CachePolicy(write_policy="write_through"))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "execution", ready)
    e.advance(until=(done,), first=False)
    assert dict(e.serviced)["hbm"] == 256.0
    assert dict(e.serviced)["mte3"] == 64.0

    load_state = CacheState(geometry(128))
    load_reservation = load_state.prepare_access(
        "L", CacheAccessGeometry(rows=1, row_bytes=64))
    load_result = load_reservation.result
    load_reservation.cancel()
    e = engine(mte2_rate=1000.0)
    program = CacheAccessProgram(load_result, onchip_resource="mte2",
                                 pool="ub", source_required=True)
    with pytest.raises(ModelingError):
        program.start(e, "missing", e.submit("ready", release=0))
    source = e.submit("source", release=10)
    ready = e.submit("ready2", release=0)
    done = program.start(e, "load", ready, source=source)
    e.advance(until=(done,), first=False)
    assert done.started >= 10.0
    assert done.time == pytest.approx(10.192)
    assert dict(e.serviced)["hbm"] == 128.0

    e = engine(mte2_rate=1000.0)
    execution = CacheExecution(load_state)
    with pytest.raises(ModelingError):
        execution.access(
            CacheAccessRequest("L2", CacheAccessGeometry(rows=1, row_bytes=64)),
            e, "missing", e.submit("ready", release=0),
            source_required=True)
    source = e.submit("source2", release=10)
    ready = e.submit("ready3", release=0)
    done = execution.access(
        CacheAccessRequest("L2", CacheAccessGeometry(rows=1, row_bytes=64)),
        e, "load2", ready, source=source, source_required=True)
    e.advance(until=(done,), first=False)
    assert done.started >= 10.0
    assert done.time == pytest.approx(10.192)


def test_state_access_projection_matches_existing_service_fields():
    state = CacheState(geometry(4096))
    request = CacheAccessRequest(
        "index@r0", CacheAccessGeometry(rows=2, row_bytes=32, stride_bytes=1025))
    access = CacheExecution(state).state_access(
        request, occurrence="indexer_prolog:index_pool_tail_store",
        storage="index_pool_tail", request_id="r0", layer=0, rank=0,
        state_version=3)
    assert access["span"] == [[0, 1], [8, 1]]
    assert access["bytes"] == 64
    assert access["direction"] == "read" and access["position"] == "cache"
    assert access["occurrence"] == "indexer_prolog:index_pool_tail_store"


def test_masked_byte_validity_and_full_line_rmw():
    state = CacheState(geometry(128))
    masked = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=32), write=True,
        policy=CachePolicy(write_policy="write_back"))
    assert masked.result.hbm_write_bytes == 0
    masked.publish()
    masked.finish()
    assert state.dirty_bytes == 32
    partial = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=64, rows=1, row_bytes=32))
    assert partial.result.hbm_read_bytes == 128
    partial.publish()
    partial.finish()
    assert state.dirty_bytes == 32  # the modified [0,32) is retained
    full = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    assert full.result.hbm_read_bytes == 0 and full.result.hit_lines == 1
    full.publish()
    full.finish()
    rmw = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=64), write=True,
        policy=CachePolicy(write_policy="write_through"))
    assert rmw.result.hbm_read_bytes == 0
    assert rmw.result.hbm_write_bytes == 128


def test_65537_repeated_stride_range_and_hole():
    state = CacheState(CacheGeometry(capacity_bytes=8388736,
                                     line_bytes=128, transfer_bytes=32))
    large = CacheAccessGeometry(rows=65537, row_bytes=64, stride_bytes=256)
    summary = large.summary(128, write=False, transfer_bytes=32)
    assert summary.line_count == 65537
    assert summary.line_segments == (Span(0, 1, 2, 65537),)
    assert summary.ranges_complete is False
    assert summary.coverage_bytes == 4194368
    assert summary.gm_bytes == 4194368
    access = state.prepare_access("A", large)
    assert access.result.hbm_read_bytes == 8388736
    assert access.result.fill_bytes == 8388736
    access.publish()
    access.finish()
    hole = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=64,
                                 stride_bytes=256))
    assert hole.result.hbm_read_bytes == 128
    hole.publish()
    hole.finish()


def test_zero_capacity_rmw_and_sparse_gm():
    state = CacheState(CacheGeometry(capacity_bytes=0, line_bytes=128,
                                     transfer_bytes=32))
    overlap = CacheAccessGeometry(
        rows=2, row_bytes=64, stride_bytes=32,
        partial_line="read_modify_write")
    access = state.prepare_access(
        "A", overlap, write=True,
        policy=CachePolicy(write_policy="write_through"))
    assert access.result.hbm_read_bytes == 128
    assert access.result.hbm_write_bytes == 128
    assert access.result.gm_bytes == 128
    assert access.result.onchip_bytes == 128

    sparse = CacheAccessGeometry(rows=2, row_bytes=1, stride_bytes=128)
    access = state.prepare_access("S", sparse)
    assert access.result.logical_bytes == 2
    assert access.result.gm_bytes == 64
    assert access.result.onchip_bytes == 64
    assert access.result.hbm_read_bytes == 256


def test_large_strided_and_overlapping_rmw_ranges():
    state = CacheState(CacheGeometry(capacity_bytes=1 << 20,
                                     line_bytes=128, transfer_bytes=32))
    large = CacheAccessGeometry(
        rows=4097, row_bytes=64, stride_bytes=256,
        partial_line="read_modify_write")
    summary = large.summary(128, write=True, transfer_bytes=32)
    assert summary.line_count == 4097
    assert summary.line_ranges[0] == (0, 1)
    assert summary.line_ranges[-1] == (8192, 1)
    access = state.prepare_access(
        "L", large, write=True,
        policy=CachePolicy(write_policy="write_through"))
    assert access.result.hbm_read_bytes == 524416
    assert access.result.hbm_write_bytes == 524416

    state = CacheState(CacheGeometry(capacity_bytes=1 << 20,
                                     line_bytes=128, transfer_bytes=32))
    overlap = CacheAccessGeometry(
        rows=2, row_bytes=64, stride_bytes=32,
        partial_line="read_modify_write")
    assert overlap.summary(128, write=True).coverage_bytes == 96
    access = state.prepare_access(
        "O", overlap, write=True,
        policy=CachePolicy(write_policy="write_through"))
    assert access.result.hbm_read_bytes == 128
    assert access.result.hbm_write_bytes == 128


def test_sync_access_uses_same_range_resolver_and_sparse_gm():
    state = CacheState(geometry(256))
    high = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=128))
    assert high.result.hbm_read_bytes == 128
    high.publish()
    high.finish()
    low = state.access("A", 128)
    assert low.hbm_read_bytes == 128
    assert state.resident_bytes == 256

    state = CacheState(geometry(256))
    sparse = CacheAccessGeometry(rows=2, row_bytes=1, stride_bytes=128)
    access = state.prepare_access("S", sparse)
    assert access.result.logical_bytes == 2
    assert access.result.gm_bytes == 64
    assert access.result.hbm_read_bytes == 256


def test_clean_bypass_write_cancel_restores_removed_identity():
    state = CacheState(geometry(128))
    state.access("A", 128)
    reservation = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(bypass=True))
    assert state.resident("A") == (0, 0)
    reservation.cancel()
    assert state.resident("A") == (1, 0)
    assert state.resident_bytes == 128 and state.dirty_bytes == 0


def test_partial_victim_cancel_restores_capacity_and_dirty():
    state = CacheState(geometry(256))
    state.access("A", 256, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    reservation = state.prepare_access(
        "B", CacheAccessGeometry(rows=1, row_bytes=128))
    assert state.resident("A") == (1, 1)
    reservation.cancel()
    assert state.resident_bytes == 256
    assert state.resident("A") == (2, 2)
    assert state.dirty_bytes == 256
    assert state.flush().bytes == 256


def test_inflight_write_holds_capacity_until_terminal():
    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(write_policy="write_through"))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "A", ready)
    e.advance(horizon=0.07)
    assert state.resident("A") == (1, 0) and not done.done
    with pytest.raises(ModelingError, match="cache_capacity_inflight_held"):
        state.prepare_access(
            "B", CacheAccessGeometry(rows=1, row_bytes=128),
            write=True, policy=CachePolicy(write_policy="write_back"))
    e.advance(until=(done,), first=False)
    assert done.time == 0.192
    later = state.prepare_access(
        "B", CacheAccessGeometry(rows=1, row_bytes=128),
        write=True, policy=CachePolicy(write_policy="write_back"))
    later.cancel()


def test_failed_ready_preparation_releases_staging_and_keeps_blocker():
    state = CacheState(geometry(128))
    blocker = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    request = CacheAccessRequest(
        "B", CacheAccessGeometry(rows=1, row_bytes=128))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "B", ready)
    with pytest.raises(ModelingError):
        e.advance(until=(done,), first=False)
    e.advance(horizon=1.0, first=False)
    assert dict(e.allocated)["ub"] == 0.0
    assert state.resident("A") == (0, 0)
    blocker.cancel()
    assert state.resident_bytes == 0 and state.dirty_bytes == 0


def test_builder_real_op_id_ports_and_internal_binding_without_generic_ready():
    builder, case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    identity = cache_identity("window", layer=0, request="r0", block=0)
    store_op = lowering.store(
        "kv_store", (), 5, plan.cache_row_elements, "window",
        identity=identity, source_residency="on_chip")
    load_op = lowering.load(
        "kv_read", (store_op,), 5, plan.cache_row_elements, "window",
        identity=identity)
    assert store_op.endswith("l0.kv_store")
    assert load_op.endswith("l0.kv_read")
    e = engine()
    events = lowering.execute(e)
    e.advance(until=events, first=False)
    store_event, load_event = events
    assert load_event.started >= store_event.time
    assert dict(e.serviced)["hbm"] == 2816.0
    assert dict(e.serviced)["mte2"] == 7840.0
    assert dict(e.serviced)["mte3"] == 2720.0
    assert dict(e.serviced)["vector"] == 10320.0
    request = lowering.requests[0][1]
    assert request.producer == store_op and request.output
    projected = CacheExecution(state).state_access(
        request, occurrence=store_op, storage="kv_store")
    assert projected["producer"] == store_op
    assert projected["output"] == request.output
    assert projected["byte_span"] == [[0, 2720]]


def test_actual_offset_ranges_stride_holes_and_rmw_work():
    state = CacheState(geometry(256))
    first = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=0, rows=1, row_bytes=128))
    assert first.result.hbm_read_bytes == 128
    first.publish()
    first.finish()
    second = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=128))
    assert second.result.hbm_read_bytes == 128 and second.result.hit_lines == 0
    second.publish()
    second.finish()
    assert state.resident_bytes == 256
    assert state.resident_ranges("A")[0] == (Span(0, 2, 2, 1),)

    state = CacheState(geometry(4096))
    stride = CacheAccessGeometry(rows=2, row_bytes=32, stride_bytes=1025)
    summary = stride.summary(128)
    assert summary.line_count == 2
    assert summary.line_ranges == ((0, 1), (8, 1))
    access = state.prepare_access("S", stride)
    assert access.result.hbm_read_bytes == 256
    access.publish()
    access.finish()
    assert state.resident_ranges("S")[0] == (Span(0, 1, 1, 1), Span(8, 1, 1, 1))

    for protocol, read_bytes in (("masked", 0), ("read_modify_write", 512)):
        state = CacheState(geometry(4096))
        geometry_rmw = CacheAccessGeometry(
            rows=4, row_bytes=64, stride_bytes=128, partial_line=protocol)
        access = state.prepare_access(
            "R", geometry_rmw, write=True,
            policy=CachePolicy(write_policy="write_through"))
        assert access.result.hbm_read_bytes == read_bytes
        assert access.result.hbm_write_bytes == 512
        access.publish()
        access.finish()


def test_partial_overlap_read_and_dirty_update_use_actual_ranges():
    state = CacheState(geometry(256))
    fill = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=0, rows=1, row_bytes=256))
    fill.publish()
    fill.finish()
    update = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=128),
        write=True, policy=CachePolicy(write_policy="write_back"))
    assert update.result.hit_lines == 1 and update.result.hbm_write_bytes == 0
    update.publish()
    update.finish()
    assert state.resident_ranges("A")[1] == (Span(1, 1, 1, 1),)
    flush = state.flush()
    e = engine()
    ready = e.submit("ready", release=0)
    e.advance(until=(flush.start(e, "flush", ready),), first=False)
    assert flush.bytes == 128 and state.dirty_bytes == 0
    low = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=0, rows=1, row_bytes=128))
    assert low.result.hbm_read_bytes == 0 and low.result.hit_lines == 1
    low.publish()
    low.finish()
    high = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=128))
    assert high.result.hbm_read_bytes == 0 and high.result.hit_lines == 1
    high.publish()
    high.finish()


def test_prebound_flush_resolves_after_write_done():
    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=256), write=True,
        policy=CachePolicy(write_policy="write_back"))
    e = engine()
    ready = e.submit("ready", release=0)
    execution = CacheExecution(state)
    write_done = execution.access(request, e, "write", ready)
    # Bind flush before any advance; it must snapshot at write_done, not now.
    flush = execution.flush(e, "flush", write_done)
    e.advance(until=(flush,), first=False)
    assert dict(e.serviced)["hbm"] == 256.0
    assert state.dirty_bytes == 0
    assert state.counters["hbm_write_bytes"] == 256


def test_write_through_bypass_rejects_while_external_write_in_flight():
    state = CacheState(geometry(256))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(write_policy="write_through"))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "write", ready)
    e.advance(horizon=0.07)  # onchip stage done, HBM write still pending
    with pytest.raises(ModelingError):
        state.prepare_access(
            "A", CacheAccessGeometry(rows=1, row_bytes=128),
            policy=CachePolicy(bypass=True))
    with pytest.raises(ModelingError):
        state.prepare_access(
            "A", CacheAccessGeometry(rows=1, row_bytes=128))
    e.advance(until=(done,), first=False)
    bypass = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128),
        policy=CachePolicy(bypass=True))
    assert bypass.result.hbm_read_bytes == 128
    bypass.publish()
    bypass.finish()


def test_cancel_restores_evicted_dirty_victim_and_later_flush():
    state = CacheState(geometry(128))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    reservation = state.prepare_access(
        "B", CacheAccessGeometry(rows=1, row_bytes=128))
    assert reservation.victims and state.resident("A") == (0, 0)
    reservation.cancel()
    assert state.resident("A") == (1, 1)
    assert state.dirty_bytes == 128 and state.resident("B") == (0, 0)
    assert state.flush().bytes == 128


def test_evicted_dirty_victim_is_held_until_writeback_terminal():
    state = CacheState(geometry(128))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    reservation = state.prepare_access(
        "B", CacheAccessGeometry(rows=1, row_bytes=128))
    with pytest.raises(ModelingError):
        state.prepare_access("A", CacheAccessGeometry(rows=1, row_bytes=128))
    # Target B may not occupy A's single physical slot before A's drain.
    with pytest.raises(ModelingError,
                       match="cache_reservation_drain_incomplete"):
        reservation.publish()
    reservation.mark_drained()
    reservation.publish()
    reservation.finish()
    assert state.resident("A") == (0, 0)
    assert state.flush().bytes == 0


def test_builder_source_conversion_and_dependency_work():
    for residency, source_hbm in (("on_chip", 0), ("unresolved", 5120)):
        builder, case, plan = packed_builder()
        state = CacheState(geometry(4096))
        lowering = CacheLowering(builder, state)
        identity = cache_identity("window", layer=0, request="r0", block=0)
        lowering.store("kv_store", ("rotary",), 5, plan.cache_row_elements,
                       "window", identity=identity,
                       source_residency=residency)
        lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements,
                      "window", identity=identity)
        op = builder.ops[0]
        assert op.mte2_bytes == 5120 and op.vector_ops == 10320
        e = engine()
        external = e.submit("external", release=0)
        r_store = e.submit("r_store", release=0)
        r_load = e.submit("r_load", release=0)
        events = lowering.execute(
            e, ready=external,
            ready_by_name={"rotary": external, "kv_store": r_store,
                           "kv_read": r_load})
        e.advance(until=events, first=False)
        assert dict(e.serviced)["hbm"] == source_hbm + 2816
        assert dict(e.serviced)["mte2"] == 5120 + 2720
        assert dict(e.serviced)["mte3"] == 2720
        assert dict(e.serviced)["vector"] == 10320


def test_two_dependent_packed_writes_plus_competitor_is_5760():
    builder, case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    identity = cache_identity("window", layer=0, request="r0", block=0)
    lowering.store("kv_store", ("rotary",), 5, plan.cache_row_elements,
                   "window", identity=identity, source_residency="on_chip")
    lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements,
                  "window", identity=identity)
    e = engine()
    external = e.submit("external", release=0)
    competitor = e.submit("competitor", {"hbm": 128.0}, release=0)
    first_store = e.submit("first_store", release=0)
    first_load = e.submit("first_load", release=10)
    first = lowering.execute(
        e, ready=external,
        ready_by_name={"rotary": external, "kv_store": first_store,
                       "kv_read": first_load})
    second_store = e.submit("second_store", release=20)
    second_load = e.submit("second_load", release=30)
    second = lowering.execute(
        e, ready=external,
        ready_by_name={"rotary": external, "kv_store": second_store,
                       "kv_read": second_load})
    e.advance(until=tuple(first) + tuple(second), first=False)
    assert dict(e.serviced)["hbm"] == 5760.0
    assert dict(e.serviced)["mte2"] == 15680.0
    assert dict(e.serviced)["mte3"] == 5440.0
    assert dict(e.serviced)["vector"] == 20640.0
    assert state.resident_bytes == 2816


def test_inflight_overlap_and_flush_are_explicitly_rejected():
    state = CacheState(geometry(128))
    first = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    with pytest.raises(ModelingError):
        state.prepare_access("A", CacheAccessGeometry(rows=1, row_bytes=128))
    with pytest.raises(ModelingError):
        state.flush()
    with pytest.raises(ModelingError):
        state.invalidate()
    with pytest.raises(ModelingError):
        state.reset()
    with pytest.raises(ModelingError):
        state.clone()
    first.publish()
    first.finish()
    assert state.resident("A") == (1, 0)


def test_partial_line_masked_and_rmw_supply_differ_only_by_declared_reads():
    for policy, read_bytes in (("masked", 0), ("read_modify_write", 256)):
        state = CacheState(geometry(512))
        request = CacheAccessRequest(
            "A", CacheAccessGeometry(offset_bytes=20, rows=2, row_bytes=100,
                                     stride_bytes=128, partial_line=policy),
            write=True, policy=CachePolicy(write_policy="write_through"))
        e = engine()
        ready = e.submit("ready", release=0)
        done = CacheExecution(state).access(request, e, "write", ready)
        e.advance(until=(done,), first=False)
        assert dict(e.serviced).get("hbm", 0.0) == 256.0 + read_bytes


def test_outstanding_transactions_are_bounded_and_conserved():
    state = CacheState(geometry(4096))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=2720), write=False,
        outstanding=4, response_ms=0.001)
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "read", ready)
    e.advance(until=(done,), first=False)
    assert dict(e.serviced)["hbm"] == 2816.0
    assert state.resident("A") == (22, 0)


def test_builder_packed_mla_program_runs_two_iterations_and_ready_reversal():
    builder, case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    identity = cache_identity("window", layer=0, request="r0", block=0)
    lowering.store("kv_store", ("rotary",), 5, plan.cache_row_elements,
                   "window", identity=identity, source_residency="on_chip")
    lowering.load("kv_read", ("kv_store",), 5, plan.cache_row_elements,
                  "window", identity=identity)
    written, read = builder.ops
    metadata = dict(written.metadata)
    assert written.hbm_bytes == read.hbm_bytes == 2816.0
    assert metadata["cache_static_basis"] == "cold_declared_request"
    assert metadata["cache_dynamic_state_resolved_at_execution"] is True
    assert metadata["cache_program_entry"] == "CacheLowering.execute"
    assert metadata["cache_transaction_summary"]["line_count"] == 22

    def run_iteration(*, store_first):
        e = engine()
        r_store = e.submit("r_store", release=0 if store_first else 10)
        r_load = e.submit("r_load", release=0 if not store_first else 10)
        external = e.submit("external", release=0)
        events = lowering.execute(
            e, ready=external,
            ready_by_name={"rotary": external, "kv_store": r_store,
                           "kv_read": r_load})
        e.advance(until=events, first=False)
        return dict(e.serviced)

    first = run_iteration(store_first=True)
    second = run_iteration(store_first=False)
    assert first["hbm"] == 2816.0   # first store materializes, load hits
    assert second["hbm"] == 2816.0  # second load hits, store writes again
    assert second["mte2"] == 7840.0   # source 5120 + load cache copy 2720
    assert second["mte3"] == 2720.0   # store cache copy
    assert second["vector"] == 10320.0
    assert state.resident_bytes == 2816


def test_builder_write_back_program_flushes_across_two_iterations():
    builder, case, plan = packed_builder()
    state = CacheState(geometry(128))
    lowering = CacheLowering(builder, state)
    lowering.store("kv_store", ("rotary",), 4, 64, "mla_kv",
                   identity="A", write_policy="write_back",
                   source_residency="on_chip")
    static = builder.ops[-1]
    assert static.hbm_bytes == 512.0  # static cold upper bound
    for iteration in range(2):
        e = engine()
        ready = e.submit("ready", release=0)
        events = lowering.execute(e, ready=ready,
                                  ready_by_name={"rotary": ready})
        e.advance(until=events, first=False)
        direct = dict(e.serviced)["hbm"]
        flush_ready = e.submit("flush_ready", release=10)
        flush = lowering.flush(e, f"flush{iteration}", flush_ready)
        e.advance(until=(flush,), first=False)
        total = dict(e.serviced)["hbm"]
        assert direct == 384.0 and total == 512.0
        assert state.dirty_bytes == 0
        assert dict(e.peak_allocated)["ub"] == 128.0
        assert dict(e.allocated)["ub"] == 0


def test_r7_1000001_strided_access_keeps_true_holes_and_prefix_residency():
    state = CacheState(CacheGeometry(capacity_bytes=256, line_bytes=128,
                                     transfer_bytes=32))
    geometry = CacheAccessGeometry(rows=1000001, row_bytes=64,
                                   stride_bytes=256)
    summary = geometry.summary(128, transfer_bytes=32)
    assert summary.line_segments == (Span(0, 1, 2, 1000001),)
    assert summary.line_count == 1000001
    assert summary.coverage_bytes == 64000064
    assert summary.ranges_complete is False
    access = state.prepare_access("A", geometry)
    assert access.result.hbm_read_bytes == 128000128
    assert access.result.fill_bytes == 256
    assert access.result.miss_lines == 1000001
    access.publish()
    access.finish()
    assert state.resident("A") == (2, 0)
    assert state.resident_ranges("A")[0] == (Span(0, 1, 2, 2),)
    hole = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=64))
    assert hole.result.hbm_read_bytes == 128
    hole.publish()
    hole.finish()


def test_r7_same_span_arithmetic_covers_writeback_dirty_and_rmw():
    geometry = CacheAccessGeometry(
        rows=1000001, row_bytes=64, stride_bytes=256,
        partial_line="read_modify_write")
    state = CacheState(CacheGeometry(capacity_bytes=256, line_bytes=128,
                                     transfer_bytes=32))
    access = state.prepare_access(
        "A", geometry, write=True,
        policy=CachePolicy(write_policy="write_back"))
    assert access.result.hbm_read_bytes == 128000128
    assert access.result.direct_write_bytes == 999999 * 128
    assert access.result.writeback_bytes == 0
    access.publish()
    access.finish()
    assert state.resident("A") == (2, 2)
    assert state.dirty_bytes == 128
    plan = state.flush()
    assert plan.bytes == 256
    engine = Engine([Resource("hbm", 1000.0, unit="bytes")])
    ready = engine.submit("ready", release=0)
    engine.advance(until=(plan.start(engine, "flush", ready),), first=False)
    assert dict(engine.serviced)["hbm"] == 256.0
    assert state.counters["direct_write_bytes"] == 999999 * 128
    assert state.counters["hbm_write_bytes"] == 1000001 * 128


def test_r7_134217728_contiguous_summary_is_segment_bounded():
    summary = CacheAccessGeometry(rows=1, row_bytes=134217728).summary(128)
    assert summary.byte_segments == (Span(0, 134217728, 134217728, 1),)
    assert summary.line_segments == (Span(0, 1048576, 1048576, 1),)
    assert summary.line_count == 1048576
    assert summary.byte_ranges == ((0, 134217728),)
    assert summary.line_ranges == ((0, 1048576),)
    assert summary.ranges_complete is True


def test_r7_sparse_staging_is_one_demand_for_resolver_and_operation():
    geometry = CacheGeometry(capacity_bytes=0, line_bytes=128,
                             transfer_bytes=32)
    access_geometry = CacheAccessGeometry(rows=2, row_bytes=1,
                                          stride_bytes=128)
    state = CacheState(geometry)
    reservation = state.prepare_access("S", access_geometry)
    assert reservation.result.gm_bytes == 64
    assert reservation.result.onchip_bytes == 64
    assert reservation.result.staging_bytes == 64
    assert reservation.result.hbm_read_bytes == 256

    request = CacheAccessRequest("S", access_geometry)
    operation = CacheAccessOperation(CacheState(geometry), request)
    assert operation.staging_bytes() == 64
    engine = globals()["engine"](pool_bytes=64)
    ready = engine.submit("ready", release=0)
    done = CacheExecution(CacheState(geometry)).access(
        request, engine, "sparse", ready)
    engine.advance(until=(done,), first=False)
    assert dict(engine.serviced)["hbm"] == 256.0
    assert dict(engine.peak_allocated)["ub"] == 64.0
    assert dict(engine.allocated)["ub"] == 0.0

    program_engine = globals()["engine"](pool_bytes=64)
    program_ready = program_engine.submit("ready", release=0)
    program = CacheAccessProgram(reservation.result, pool="ub").start(
        program_engine, "program", program_ready)
    program_engine.advance(until=(program,), first=False)
    assert dict(program_engine.serviced)["hbm"] == 256.0
    assert dict(program_engine.peak_allocated)["ub"] == 64.0
    assert dict(program_engine.allocated)["ub"] == 0.0
    reservation.cancel()


def test_r7_bypass_write_holds_identity_until_external_terminal():
    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(bypass=True, write_policy="write_through"))
    engine = globals()["engine"]()
    ready = engine.submit("ready", release=0)
    done = CacheExecution(state).access(request, engine, "A", ready)
    engine.advance(horizon=0.07)
    assert not done.done
    assert state.inflight_identities == 1
    with pytest.raises(ModelingError, match="cache_identity_access_in_flight"):
        state.prepare_access("A", CacheAccessGeometry(rows=1, row_bytes=128))
    engine.advance(until=(done,), first=False)
    assert done.time == 0.192
    assert state.inflight_identities == 0
    read_request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    read_ready = engine.submit("read_ready", release=0)
    read_done = CacheExecution(state).access(
        read_request, engine, "read_after", read_ready)
    engine.advance(until=(read_done,), first=False)
    assert read_done.done and state.resident("A") == (1, 0)
    assert dict(engine.serviced)["hbm"] == 256.0


def test_r7_zero_byte_holder_never_releases_the_nonempty_holder():
    state = CacheState(geometry(256))
    zero = state.prepare_access("A", CacheAccessGeometry(rows=0, row_bytes=0))
    assert zero.held_identities == ()
    nonempty = state.prepare_access("A", CacheAccessGeometry(rows=1,
                                                             row_bytes=128))
    snapshot = state.snapshot()
    assert snapshot["reserved_bytes"] == 128
    assert snapshot["inflight_identities"] == 1
    zero.publish()
    zero.finish()
    snapshot = state.snapshot()
    assert snapshot["reserved_bytes"] == 128
    assert snapshot["inflight_identities"] == 1
    assert state.resident("A") == (0, 0)
    nonempty.publish()
    nonempty.finish()
    assert state.snapshot()["reserved_bytes"] == 0
    assert state.inflight_identities == 0
    assert state.resident("A") == (1, 0)


def test_r7_named_external_port_is_required_and_binds_the_real_source_event():
    builder, case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    lowering.load("kv_read", ("src",), 1, 64, "mla_kv", identity="A")
    engine = globals()["engine"]()
    generic = engine.submit("generic_ready", release=0)
    with pytest.raises(ModelingError, match="cache_lowering_port_unbound:src"):
        lowering.execute(engine, ready=generic)
    source = engine.submit("src_event", release=10)
    events = lowering.execute(engine, ready=generic,
                              ready_by_name={"src": source})
    engine.advance(until=events, first=False)
    assert events[0].started == 10.0
    assert events[0].time == 10.192
    assert dict(engine.serviced)["hbm"] == 128.0
    binding = lowering.execution.binding("kv_read")
    assert binding.started == 10.0 and binding.completed == 10.192


def test_r7_direct_store_executes_source_conversion_and_typed_view():
    builder, case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    identity = cache_identity("window", layer=0, request="r0", block=0)
    store_op = lowering.store(
        "kv_store", (), 5, plan.cache_row_elements, "window",
        identity=identity, source_residency="on_chip")
    request = lowering.requests[0][1]
    assert request.view.producer == store_op
    assert request.view.output == store_op + ".output"
    assert request.view.layout == "packed_row"
    assert request.view.payload_bytes + request.view.scale_bytes == 2720

    engine = globals()["engine"]()
    ready = engine.submit("ready", release=0)
    execution = CacheExecution(state)
    event = execution.access(request, engine, "store", ready)
    engine.advance(until=(event,), first=False)
    assert dict(engine.serviced)["hbm"] == 2816.0
    assert dict(engine.serviced)["mte2"] == 5120.0
    assert dict(engine.serviced)["mte3"] == 2720.0
    assert dict(engine.serviced)["vector"] == 10320.0
    binding = execution.binding("store")
    assert binding.producer == store_op
    assert binding.output == store_op + ".output"
    assert binding.storage == "window"
    assert binding.owner == identity
    assert binding.completed == event.time
    assert binding.state_version == state.version
    assert binding.staging["size"] == 128
    assert binding.grant is None
    projected = execution.state_access(request, occurrence="store",
                                       storage="window")
    assert projected["producer"] == store_op
    assert projected["output"] == request.view.output
    assert projected["storage"] == "window" and projected["owner"] == identity
    assert projected["dtype"] == request.view.dtype
    assert projected["layout"] == "packed_row"
    assert projected["completed"] == event.time
    assert projected["payload_bytes"] == request.view.payload_bytes
    assert projected["scale_bytes"] == request.view.scale_bytes
    metadata = dict(builder.ops[-1].metadata)
    assert metadata["cache_typed_view"]["producer"] == store_op
    assert metadata["cache_typed_view"]["output"] == request.view.output
    assert metadata["cache_access_program"]["producer"] == store_op

    # The pre-resolved static adapter consumes the same stage list as the
    # event-time executor, including source and conversion.
    resolved = state.prepare_access(
        request.identity, request.geometry, write=True, policy=request.policy)
    program_engine = globals()["engine"]()
    program_ready = program_engine.submit("ready", release=0)
    program = CacheAccessProgram(
        resolved.result, pool="ub", source_bytes=request.source_bytes,
        convert_ops=request.convert_ops,
        source_residency=request.source_residency,
        write_resource="mte3").start(program_engine, "program", program_ready)
    program_engine.advance(until=(program,), first=False)
    assert dict(program_engine.serviced) == {
        "hbm": 2816.0, "mte2": 5120.0, "mte3": 2720.0, "vector": 10320.0}
    resolved.cancel()


def test_r7_same_request_rebinds_per_occurrence_and_keeps_its_version():
    state = CacheState(geometry(256))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    engine = globals()["engine"]()
    execution = CacheExecution(state)
    first_ready = engine.submit("first_ready", release=0)
    first = execution.access(request, engine, "first", first_ready)
    engine.advance(until=(first,), first=False)
    second_ready = engine.submit("second_ready", release=10)
    second = execution.access(request, engine, "second", second_ready)
    engine.advance(until=(second,), first=False)
    assert first.time < second.time
    assert state.resident("A") == (1, 0)
    first_view = execution.state_access(request, occurrence="first",
                                        storage="A")
    second_view = execution.state_access(request, occurrence="second",
                                         storage="A")
    assert first_view["completed"] == first.time
    assert second_view["completed"] == second.time
    assert first_view["started"] == 0.0
    assert second_view["started"] == 10.0
    assert second_view["state_version"] > first_view["state_version"]
    assert first_view["span"] == [[0, 1]]
    assert second_view["span"] == [[0, 1]]


def test_r8_same_stride_full_line_union_has_no_false_rmw_or_expansion():
    for count in (3, 65537):
        state = CacheState(CacheGeometry(
            capacity_bytes=count * 128, line_bytes=128, transfer_bytes=32))
        low = CacheAccessGeometry(rows=count, row_bytes=64, stride_bytes=256)
        summary = low.summary(128, transfer_bytes=32)
        assert summary.line_segments == (Span(0, 1, 2, count),)
        if count == 3:
            # Direct byte/line set check before the large-set control.
            assert summary.byte_segments == (Span(0, 64, 256, 3),)
            assert summary.line_ranges == ((0, 1), (2, 1), (4, 1))
        first = state.prepare_access("A", low)
        assert first.result.hbm_read_bytes == count * 128
        first.publish()
        first.finish()
        assert state.resident_ranges("A")[0] == (Span(0, 1, 2, count),)
        high = CacheAccessGeometry(
            offset_bytes=64, rows=count, row_bytes=64, stride_bytes=256,
            partial_line="read_modify_write")
        update = state.prepare_access(
            "A", high, write=True,
            policy=CachePolicy(write_policy="write_back"))
        assert update.result.hbm_read_bytes == 0
        assert update.result.hbm_write_bytes == 0
        assert update.result.hit_lines == count
        update.publish()
        update.finish()
        assert state.dirty_bytes == count * 64
        plan = state.flush()
        assert plan.bytes == count * 128
        plan.cancel()


def test_r8_masked_low_then_high_half_union_and_missing_byte_rmw():
    count = 65537
    state = CacheState(CacheGeometry(
        capacity_bytes=count * 128, line_bytes=128, transfer_bytes=32))
    low = CacheAccessGeometry(
        rows=count, row_bytes=64, stride_bytes=256,
        partial_line="masked")
    first = state.prepare_access(
        "A", low, write=True, policy=CachePolicy(write_policy="write_back"))
    assert first.result.hbm_read_bytes == 0
    first.publish()
    first.finish()
    assert state.dirty_bytes == count * 64
    high = CacheAccessGeometry(
        offset_bytes=64, rows=count, row_bytes=64, stride_bytes=256,
        partial_line="read_modify_write")
    update = state.prepare_access(
        "A", high, write=True, policy=CachePolicy(write_policy="write_back"))
    assert update.result.hbm_read_bytes == 0
    assert update.result.hbm_write_bytes == 0
    update.publish()
    update.finish()
    assert state.dirty_bytes == count * 128
    plan = state.flush()
    assert plan.bytes == count * 128
    plan.cancel()

    # One byte short of the high half leaves a real per-line hole.
    hole_state = CacheState(CacheGeometry(
        capacity_bytes=count * 128, line_bytes=128, transfer_bytes=32))
    hole_low = hole_state.prepare_access(
        "A", low, write=True, policy=CachePolicy(write_policy="write_back"))
    hole_low.publish()
    hole_low.finish()
    short = CacheAccessGeometry(
        offset_bytes=64, rows=count, row_bytes=63, stride_bytes=256,
        partial_line="read_modify_write")
    missing = hole_state.prepare_access(
        "A", short, write=True, policy=CachePolicy(write_policy="write_back"))
    assert missing.result.hbm_read_bytes == count * 128
    missing.cancel()
    odd = hole_state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=3 * 128, rows=1, row_bytes=64))
    assert odd.result.hbm_read_bytes == 128
    odd.cancel()


def test_r8_128m_contiguous_rmw_is_compact_and_page_count_is_touched():
    summary = CacheAccessGeometry(
        rows=1, row_bytes=134217728,
        partial_line="read_modify_write").summary(128, write=True)
    assert summary.line_segments == (Span(0, 1048576, 1048576, 1),)
    assert summary.line_ranges == ((0, 1048576),)
    assert summary.rmw_read_lines == 0
    assert full_line_spans(summary.byte_segments, 128) == (
        Span(0, 1048576, 1048576, 1),)
    pages = CacheAccessGeometry(
        rows=2, row_bytes=1, stride_bytes=8192,
        page_bytes=4096).summary(128)
    assert pages.page_count == 2


def test_r8_bypass_rollback_holds_capacity_until_cancel_or_publish():
    state = CacheState(geometry(128))
    state.access("A", 128)
    reservation = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(bypass=True))
    assert state.resident("A") == (0, 0)
    snapshot = state.snapshot()
    assert snapshot["resident_bytes"] == 0
    assert snapshot["reserved_bytes"] == 128
    with pytest.raises(ModelingError,
                       match="cache_capacity_reserved_exhausted"):
        state.prepare_access("B", CacheAccessGeometry(rows=1, row_bytes=128))
    reservation.cancel()
    assert state.resident("A") == (1, 0)
    assert state.resident_bytes == 128
    assert state.snapshot()["reserved_bytes"] == 0
    later = state.prepare_access("B", CacheAccessGeometry(rows=1, row_bytes=128))
    later.publish()
    later.finish()
    assert state.resident("B") == (1, 0)
    assert state.resident("A") == (0, 0)
    assert state.resident_bytes == 128

    state = CacheState(geometry(128))
    state.access("A", 128)
    published = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128), write=True,
        policy=CachePolicy(bypass=True))
    published.publish()
    published.finish()
    assert state.resident("A") == (0, 0)
    assert state.resident_bytes == 0
    assert state.snapshot()["reserved_bytes"] == 0


def test_r8_failed_ready_abort_never_completes_a_successor():
    state = CacheState(geometry(128))
    blocker = state.prepare_access(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    request = CacheAccessRequest(
        "B", CacheAccessGeometry(rows=1, row_bytes=128))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "B", ready)
    completed = []
    successor = e.submit("successor", after=(done,),
                         on_complete=lambda _event: completed.append(1))
    assert not successor.done
    with pytest.raises(ModelingError):
        e.advance(until=(done,), first=False)
    e.advance(horizon=1.0, first=False)
    assert completed == []
    assert not done.done
    assert done.sequence not in e._promises
    assert dict(e.allocated)["ub"] == 0.0
    assert state.snapshot()["reserved_bytes"] == 128
    blocker.cancel()
    retry_ready = e.submit("retry_ready", release=0)
    retry = CacheExecution(state).access(request, e, "B_retry", retry_ready)
    e.advance(until=(retry,), first=False)
    assert retry.done and state.resident("B") == (1, 0)
    assert completed == []


def test_r8_cancelled_eviction_restores_the_original_lru_position():
    state = CacheState(geometry(384))
    for identity in ("A", "B", "C"):
        state.access(identity, 128)
    assert tuple(state._resident) == ("A", "B", "C")
    reservation = state.prepare_access(
        "D", CacheAccessGeometry(rows=1, row_bytes=128))
    assert tuple(state._resident) == ("B", "C")
    reservation.cancel()
    assert tuple(state._resident) == ("A", "B", "C")
    assert state.resident_bytes == 384


def _packed_store_lowering(residency, *, outstanding=0, response_ms=0.0):
    builder, _case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    identity = cache_identity("window", layer=0, request="r0", block=0)
    # The case declares FP8/group32 for the window cache row. Using the plain
    # mla_kv region here would select a different 1024 B row and silently turn
    # the packed control into a different physical case.
    lowering.store(
        "kv_store", (), 5, plan.cache_row_elements, "window",
        identity=identity, source_residency=residency,
        outstanding=outstanding, response_ms=response_ms)
    return lowering, state, lowering.requests[0][1]


def test_r8_stage_dag_orders_source_conversion_and_cache_effects():
    for residency, labels, finish in (
            ("on_chip",
             ("source_onchip", "convert", "stage", "writeback", "retire"),
             17.056),
            ("unresolved",
             ("source_hbm", "source_onchip", "convert", "stage",
              "writeback", "retire"),
             22.176)):
        lowering, state, request = _packed_store_lowering(residency)
        assert request.source_bytes == 5120
        reservation = state.prepare_access(
            request.identity, request.geometry, write=True,
            policy=request.policy)
        stages = cache_work_stages(
            reservation.result, source_bytes=request.source_bytes,
            source_residency=residency, convert_ops=request.convert_ops)
        assert tuple(stage.label for stage in stages) == labels
        by_label = {stage.label: stage for stage in stages}
        if residency == "on_chip":
            assert by_label["source_onchip"].after == ("ready",)
        else:
            assert by_label["source_hbm"].after == ("ready",)
            assert by_label["source_onchip"].after == ("source_hbm",)
        assert by_label["convert"].after == ("source_onchip",)
        assert by_label["stage"].after == ("convert",)
        assert by_label["writeback"].after == ("stage",)
        assert by_label["retire"].after == ("writeback",)
        reservation.cancel()

        op = CacheAccessOperation(
            CacheState(geometry(4096)), request, write_resource="mte3")
        e = engine()
        ready = e.submit("ready", release=0)
        done = op.start(e, "packed", ready)
        e.advance(until=(done,), first=False)
        assert done.time == pytest.approx(finish)
        assert op.stage_events["source_onchip"].started == pytest.approx(
            0.0 if residency == "on_chip" else 5.12)
        assert op.stage_events["convert"].started == pytest.approx(
            2.56 if residency == "on_chip" else 7.68)
        assert op.stage_events["retire"].time == pytest.approx(finish)

        program_state = CacheState(geometry(4096))
        program_reservation = program_state.prepare_access(
            request.identity, request.geometry, write=True,
            policy=request.policy)
        program = CacheAccessProgram(
            program_reservation.result, pool="ub",
            source_bytes=request.source_bytes,
            convert_ops=request.convert_ops,
            source_residency=residency, write_resource="mte3")
        program_engine = engine()
        program_ready = program_engine.submit("ready", release=0)
        program_done = program.start(program_engine, "packed_program",
                                     program_ready)
        program_engine.advance(until=(program_done,), first=False)
        assert program_done.time == pytest.approx(finish)
        assert dict(program_engine.serviced) == dict(e.serviced)
        program_reservation.cancel()


def test_r8_both_adapters_share_declared_credit_and_response_work():
    lowering, state, request = _packed_store_lowering(
        "unresolved", outstanding=4, response_ms=0.001)
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "credit", ready)
    e.advance(until=(done,), first=False)

    program_state = CacheState(geometry(4096))
    reservation = program_state.prepare_access(
        request.identity, request.geometry, write=True,
        policy=request.policy)
    program = CacheAccessProgram(
        reservation.result, pool="ub", source_bytes=request.source_bytes,
        convert_ops=request.convert_ops,
        source_residency=request.source_residency,
        write_resource="mte3", outstanding=request.outstanding,
        response_ms=request.response_ms)
    program_engine = engine()
    program_ready = program_engine.submit("ready", release=0)
    program_done = program.start(program_engine, "credit_program",
                                 program_ready)
    program_engine.advance(until=(program_done,), first=False)
    assert dict(program_engine.serviced) == dict(e.serviced)
    assert program_done.time == pytest.approx(done.time)
    reservation.cancel()


def test_r8_store_source_bytes_come_from_activation_not_output_row():
    builder, case, plan = packed_builder()
    metadata = dict(case.model.metadata)
    representations = dict(metadata.get("cache_representations", {}))
    representations["window"] = {"dtype": "fp4_e2m1"}
    metadata["cache_representations"] = representations
    model = replace(case.model, metadata=tuple(metadata.items()))
    builder = Builder("l0", plan, 0,
                      (RequestView("r0", 1, 100, False, "decode"),),
                      model, "decode")
    lowering = CacheLowering(builder, CacheState(geometry(4096)))
    lowering.store("kv_store", (), 5, 64, "window", identity="W",
                   source_residency="unresolved")
    request = lowering.requests[0][1]
    op = builder.ops[-1]
    expected = 5 * 64 * dtype_bytes(model.activation_dtype)
    assert expected != int(op.mte2_bytes)
    assert request.source_bytes == expected


def test_r8_missing_later_port_submits_nothing():
    builder, _case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    lowering.store("kv_store", (), 5, plan.cache_row_elements, "mla_kv",
                   identity="A", source_residency="on_chip")
    lowering.load("kv_read", ("missing",), 5, plan.cache_row_elements,
                  "mla_kv", identity="A")
    e = engine()
    generic = e.submit("generic", release=10)
    snapshot = state.snapshot()
    version = state.version
    with pytest.raises(ModelingError,
                       match="cache_lowering_port_unbound:missing"):
        lowering.execute(e, ready=generic)
    assert e._sequence == generic.sequence
    assert e.completed_count == 0
    assert all(value == 0.0 for value in e.serviced.values())
    assert state.version == version
    assert state.snapshot() == snapshot
    assert not lowering._executions


def test_r8_declared_output_alias_is_not_an_actual_port():
    builder, _case, plan = packed_builder()
    lowering = CacheLowering(builder, CacheState(geometry(4096)))
    store_op = lowering.store(
        "kv_store", (), 5, plan.cache_row_elements, "mla_kv",
        identity="A", source_residency="on_chip")
    lowering.load("kv_read", (store_op + ".output",), 5,
                  plan.cache_row_elements, "mla_kv", identity="A")
    e = engine()
    with pytest.raises(ModelingError,
                       match="cache_lowering_port_unregistered"):
        lowering.execute(e)
    assert e._sequence == 0


def test_r8_dependency_order_is_preflighted_before_submission():
    builder, _case, plan = packed_builder()
    lowering = CacheLowering(builder, CacheState(geometry(4096)))
    lowering.load("later", ("earlier",), 5, plan.cache_row_elements,
                  "mla_kv", identity="A")
    lowering.load("earlier", (), 5, plan.cache_row_elements,
                  "mla_kv", identity="A")
    e = engine()
    with pytest.raises(ModelingError,
                       match="cache_lowering_dependency_order:earlier"):
        lowering.execute(e)
    assert e._sequence == 0


def test_r8_generic_and_named_ready_join_instead_of_dropping_boundary():
    builder, _case, plan = packed_builder()
    lowering = CacheLowering(builder, CacheState(geometry(4096)))
    lowering.load("kv_read", (), 1, 64, "mla_kv", identity="A")
    # The selected 64-element mla_kv row is one ordinary 128 B line. With the
    # standard 2000 B/ms MTE2 rate the copy is .064 ms: .128 + .064 = .192.
    e = engine()
    generic = e.submit("generic", release=10)
    named = e.submit("kv_read", release=0)
    events = lowering.execute(e, ready=generic,
                              ready_by_name={"kv_read": named})
    e.advance(until=events, first=False)
    assert events[0].started == 10.0
    assert events[0].time == pytest.approx(10.192)


def test_r8_direct_access_requires_declared_source_port_before_submission():
    state = CacheState(geometry(256))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=64),
        source_port="src")
    e = engine(mte2_rate=1000.0)
    ready = e.submit("ready", release=0)
    execution = CacheExecution(state)
    with pytest.raises(ModelingError,
                       match="cache_access_source_port_unbound:src"):
        execution.access(request, e, "declared", ready)
    assert e._sequence == ready.sequence
    source = e.submit("src", release=5)
    done = execution.access(request, e, "declared", ready, source=source)
    e.advance(until=(done,), first=False)
    assert done.started == 5.0
    assert done.time == pytest.approx(5.192)
    assert execution.binding("declared").source is source


def test_r8_runtime_receipts_keep_identity_and_reject_ghost_or_mismatch():
    state = CacheState(geometry(256))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    e = engine()
    execution = CacheExecution(state)
    first_ready = e.submit("first_ready", release=0)
    first_event = execution.access(request, e, "first", first_ready)
    e.advance(until=(first_event,), first=False)
    first = execution.binding("first")
    key = first.occurrence
    assert first.launch_name == "first"

    second_ready = e.submit("second_ready", release=10)
    second_event = execution.access(request, e, "second", second_ready)
    e.advance(until=(second_event,), first=False)
    second = execution.binding("second")
    assert second.occurrence != key

    third_ready = e.submit("third_ready", release=20)
    third_event = execution.access(request, e, "first", third_ready)
    e.advance(until=(third_event,), first=False)
    old = execution.binding(key)
    assert old.event is first_event and old.completed == first.completed
    assert old.output == first.output
    assert old.state_version == first.state_version
    assert execution.binding(second.occurrence).event is second_event

    with pytest.raises(ModelingError,
                       match="cache_access_occurrence_unknown:ghost"):
        execution.state_access(request, occurrence="ghost", storage="A")
    with pytest.raises(ModelingError,
                       match="cache_access_binding_mismatch:second"):
        execution.state_access(request, occurrence="second", storage="A",
                               binding=old)

    projected = execution.state_access(request, occurrence=key, storage="A")
    assert projected["runtime_bound"] is True
    assert projected["output_actual"] is False
    assert projected["grant"] is None
    assert projected["grant_connected"] is False
    assert projected["staging"]["size"] == 128
    assert "T08_grant_not_connected" in projected["gaps"]
    assert projected["evidence"] == "runtime"

    source = e.submit("real_source", release=0)
    sourced_ready = e.submit("sourced_ready", release=0)
    sourced_event = execution.access(
        request, e, "sourced", sourced_ready, source=source)
    e.advance(until=(sourced_event,), first=False)
    sourced = execution.state_access(
        request, occurrence="sourced", storage="A")
    assert sourced["source"] is source
    assert sourced["source_actual"] is True


def _oracle_bytes(spans):
    result = set()
    for item in normalize_spans(spans):
        for index in range(item.count):
            start = item.start + index * item.stride
            result.update(range(start, start + item.width))
    return result


def _oracle_ranges(values):
    ordered = sorted(values)
    result = []
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
            end += 1
        result.append(span(ordered[index], end - index + 1))
        index = end + 1
    return normalize_spans(result)


def test_r10_overlapping_span_oracle_covers_line_rmw_and_prefix_sets():
    """Independent finite byte-set oracle for the compact span algebra."""
    cases = (
        (),
        (Span(0, 1, 1, 1),),
        (Span(0, 3, 8, 3),),
        (Span(2, 1, 8, 3),),
        (Span(0, 2, 5, 3), Span(3, 2, 5, 3)),
        (Span(0, 3, 5, 4), Span(1, 4, 4, 4)),
        (Span(0, 2, 10, 2), Span(5, 1, 3, 3)),
        (Span(0, 1, 1, 1), Span(3, 1, 1, 1), Span(5, 1, 1, 1)),
        (Span(0, 2, 3, 3), Span(1, 3, 3, 3)),
        (Span(0, 64, 256, 3), Span(128, 1, 1, 1)),
        (Span(0, 4, 7, 4), Span(2, 2, 11, 3)),
    )
    for left in cases:
        for right in cases:
            left_bytes = _oracle_bytes(left)
            right_bytes = _oracle_bytes(right)
            whole = left_bytes | right_bytes
            assert _oracle_bytes(union_spans(left, right)) == whole
            assert _oracle_bytes(subtract_spans(left, right)) == (
                left_bytes - right_bytes)
            assert _oracle_bytes(intersect_spans(left, right)) == (
                left_bytes & right_bytes)
            for limit in (0, 1, 3, 7, len(whole)):
                assert _oracle_bytes(take_first_spans(left + right, limit)) == (
                    set(sorted(whole)[:limit]))
            for line in (1, 2, 4, 8, 16, 32):
                touched = {byte // line for byte in whole}
                assert _oracle_bytes(lines_of_byte_spans(
                    left + right, line)) == touched
                full = {number for number in touched
                        if set(range(number * line, (number + 1) * line))
                        <= whole}
                assert _oracle_bytes(full_line_spans(left + right, line)) \
                    == full

    # The same oracle checks RMW holes: every requested line not completely
    # supplied by request bytes plus resident valid bytes needs a read.
    for offset, rows, row_bytes, stride in (
            (0, 3, 64, 256), (20, 2, 100, 128), (5, 4, 32, 64),
            (3, 5, 48, 128), (127, 2, 7, 256)):
        access = CacheAccessGeometry(
            offset_bytes=offset, rows=rows, row_bytes=row_bytes,
            stride_bytes=stride, partial_line="read_modify_write")
        summary = access.summary(16, write=True)
        request_lines = _oracle_bytes(summary.line_segments)
        full = _oracle_bytes(full_line_spans(summary.byte_segments, 16))
        assert summary.rmw_read_lines == len(request_lines - full)


def test_r10_interleaved_short_span_exposes_tail_line_holes_and_high_hits():
    state = CacheState(CacheGeometry(capacity_bytes=1024, line_bytes=128,
                                     transfer_bytes=32))
    first = state.prepare_access(
        "A", CacheAccessGeometry(offset_bytes=128, rows=1, row_bytes=1),
        write=True, policy=CachePolicy(write_policy="write_back"))
    first.publish()
    first.finish()
    shape = CacheAccessGeometry(
        offset_bytes=0, rows=3, row_bytes=64, stride_bytes=256,
        partial_line="read_modify_write")
    summary = shape.summary(128, write=True)
    assert summary.line_segments == (Span(0, 1, 2, 3),)
    assert full_line_spans(summary.byte_segments, 128) == ()
    assert summary.rmw_read_lines == 3
    second = state.prepare_access(
        "A", shape, write=True, policy=CachePolicy(write_policy="write_back"))
    assert second.result.hbm_read_bytes == 384
    second.publish()
    second.finish()
    for offset in (64, 320, 576):
        high = state.prepare_access(
            "A", CacheAccessGeometry(offset_bytes=offset, rows=1,
                                     row_bytes=64))
        assert high.result.hbm_read_bytes == 0
        assert high.result.hit_lines == 1
        high.publish()
        high.finish()


def test_r10_async_intersection_is_bounded_by_actual_row_pairs():
    left = Span(0, 100000007, 200000011, 2)
    right = Span(1, 100000007, 300000013, 2)
    got = intersect_spans((left,), (right,))
    pairs = []
    for left_index in range(left.count):
        left_start = left.start + left_index * left.stride
        for right_index in range(right.count):
            right_start = right.start + right_index * right.stride
            lo = max(left_start, right_start)
            hi = min(left_start + left.width, right_start + right.width)
            if hi > lo:
                pairs.append(span(lo, hi - lo))
    assert got == normalize_spans(pairs)
    assert got == (Span(1, 100000006, 100000006, 1),
                   Span(300000014, 4, 4, 1))


def test_r10_failed_issued_stage_releases_staging_and_reservation(
        monkeypatch):
    state = CacheState(geometry(128))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheExecution(state).access(request, e, "aborted", ready)
    completed = []
    successor = e.submit("successor", after=(done,),
                         on_complete=lambda _event: completed.append(1))
    real_submit = e.submit
    injected = {"done": False}

    def fail_publish(name, *args, **kwargs):
        if name.endswith(":publish") and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("injected_stage_submission_failure")
        return real_submit(name, *args, **kwargs)

    monkeypatch.setattr(e, "submit", fail_publish)
    with pytest.raises(RuntimeError,
                       match="injected_stage_submission_failure"):
        e.advance(until=(done,), first=False)
    # The fill was already issued. Drive its real terminal and the explicit
    # failure closure; no successful successor may complete.
    e.advance(horizon=1.0, first=False)
    assert completed == []
    assert not done.done and not successor.done
    assert done.sequence not in e._promises
    assert dict(e.allocated)["ub"] == 0.0
    assert state.snapshot()["reserved_bytes"] == 0
    assert state.inflight_identities == 0
    assert state.resident("A") == (0, 0)
    # The same state/Engine can retry after the explicit failure closure.
    monkeypatch.setattr(e, "submit", real_submit)
    retry_ready = e.submit("retry_ready", release=0)
    retry = CacheExecution(state).access(request, e, "retry", retry_ready)
    e.advance(until=(retry,), first=False)
    assert retry.done and state.resident("A") == (1, 0)


def test_r10_dirty_victim_drains_before_target_reuses_the_slot():
    state = CacheState(geometry(128))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    request = CacheAccessRequest(
        "B", CacheAccessGeometry(rows=1, row_bytes=128))
    e = engine()
    ready = e.submit("ready", release=0)
    operation = CacheAccessOperation(state, request)
    done = operation.start(e, "B", ready)
    e.advance(horizon=0.2)
    drain = operation.stage_events["drain"]
    fill = operation.stage_events["fill"]
    assert drain.time <= fill.started
    assert state.snapshot()["resident_bytes"] <= 128
    assert state.snapshot()["reserved_bytes"] == 128
    assert state.resident("A") == (0, 0)
    e.advance(until=(done,), first=False)
    assert state.resident("B") == (1, 0)
    assert state.resident("A") == (0, 0)
    assert state.resident_bytes == 128
    assert dict(e.serviced)["hbm"] == 256.0
    assert state.counters["hbm_write_bytes"] == 128


def test_r10_later_resource_preflight_leaves_engine_and_state_untouched():
    builder, _case, plan = packed_builder()
    state = CacheState(geometry(4096))
    lowering = CacheLowering(builder, state)
    lowering.load("first", (), 1, 64, "mla_kv", identity="A")
    lowering.store("second", (), 5, plan.cache_row_elements, "window",
                   identity="B", source_residency="on_chip")
    e = Engine([Resource("hbm", 1000.0, unit="bytes"),
                Resource("mte2", 2000.0), Resource("mte3", 2000.0)],
               pools={"ub": 1 << 20})
    snapshot = state.snapshot()
    version = state.version
    with pytest.raises(ModelingError,
                       match="vector_resource_missing"):
        lowering.execute(e)
    assert e._sequence == 0
    assert e.completed_count == 0
    assert state.snapshot() == snapshot
    assert state.version == version


def test_r10_two_hit_reservations_restore_stable_lru_order():
    state = CacheState(geometry(384))
    for identity in ("A", "B", "C"):
        state.access(identity, 128)
    first = state.prepare_access("A", CacheAccessGeometry(
        rows=1, row_bytes=128))
    second = state.prepare_access("B", CacheAccessGeometry(
        rows=1, row_bytes=128))
    assert tuple(state._resident) == ("C",)
    first.cancel()
    assert tuple(state._resident) == ("A", "C")
    second.cancel()
    assert tuple(state._resident) == ("A", "B", "C")
    victim = state.prepare_access("D", CacheAccessGeometry(
        rows=1, row_bytes=128))
    assert state.resident("A") == (0, 0)
    victim.cancel()
    assert tuple(state._resident) == ("A", "B", "C")


def test_r10_occurrences_cross_engines_and_sources_cannot_be_fabricated():
    state = CacheState(geometry(256))
    request = CacheAccessRequest(
        "A", CacheAccessGeometry(rows=1, row_bytes=128))
    first_engine = engine()
    first_execution = CacheExecution(state)
    first_ready = first_engine.submit("first_ready", release=0)
    first_event = first_execution.access(
        request, first_engine, "first", first_ready)
    first_engine.advance(until=(first_event,), first=False)
    first = first_execution.binding("first")
    other_source = first_engine.submit("other_source", release=0)
    with pytest.raises(ModelingError,
                       match="cache_access_source_not_recorded"):
        first_execution.state_access(request, occurrence=first.occurrence,
                                     storage="A", source=other_source)

    second_state = CacheState(geometry(256))
    second_engine = engine()
    second_execution = CacheExecution(second_state)
    second_ready = second_engine.submit("first_ready", release=0)
    second_event = second_execution.access(
        request, second_engine, "first", second_ready)
    second_engine.advance(until=(second_event,), first=False)
    second = second_execution.binding("first")
    assert second.occurrence != first.occurrence
    assert second_execution.binding(first.occurrence) is None
    assert first.event is first_event
    assert first.completed == first_event.time
