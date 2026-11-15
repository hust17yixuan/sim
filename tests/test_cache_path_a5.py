"""A5 cache path: logical/GM/on-chip/HBM separation and stateful reuse.

Fixtures use ordinary arithmetic and one explicit packed-MLA case plus one A3
plain-cache case. They do not read weights, start devices or inject an observed
duration. The counterexamples target the review's writeback conservation,
identity, geometry-export and direct-path contracts.
"""
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.prediction.cache_path import (
    AccessSummary, CacheAccessProgram, CacheGeometry, CachePolicy, CacheState,
)
from mindie_sim.prediction.cache_representation import cache_identity, load, store
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView

ROOT = Path(__file__).resolve().parents[1]
PACKED_MLA_CASE = (ROOT / "examples/cases/longctx-decode-20261030/"
                   "v41-a5-tp1-dp8-ep8.json")
PLAIN_A3_CASE = ROOT / "examples/cases/v4f-a3-w8a8-tp8.json"


def geometry(capacity_bytes=256, *, line_bytes=128, transfer_bytes=32, staging_lines=1):
    return CacheGeometry(capacity_bytes=capacity_bytes, line_bytes=line_bytes,
                         transfer_bytes=transfer_bytes, staging_lines=staging_lines)


def parsed_builder(path=PACKED_MLA_CASE):
    case = parse_case(json.loads(Path(path).read_text()))
    plan = case.model.layer_plans[0]
    views = (RequestView("r0", 1, 100, False, "decode"),)
    return Builder("l0", plan, 0, views, case.model, "decode"), case, plan


def engine(hbm_rate=1000.0, onchip_rate=2000.0, pool_bytes=1 << 20):
    return Engine([Resource("hbm", hbm_rate), Resource("mte2", onchip_rate)],
                  pools={"ub": pool_bytes})


def execute_access_and_writeback(result, state, *, hbm_rate=1000.0):
    """Execute one access program and the state's executable flush on one Engine."""
    e = engine(hbm_rate=hbm_rate)
    ready = e.submit("ready", release=0)
    access = CacheAccessProgram(result, pool="ub").start(e, "access", ready)
    plan = state.flush()
    flush = plan.start(e, "flush", access)
    e.advance(until=(flush,))
    return e, plan


def execute_writeback(state, *, hbm_rate=1000.0):
    e = engine(hbm_rate=hbm_rate)
    ready = e.submit("ready", release=0)
    plan = state.flush()
    done = plan.start(e, "flush", ready)
    e.advance(until=(done,))
    return e, plan


def test_cold_then_warm_read_separates_gm_from_hbm():
    state = CacheState(geometry())
    cold = state.access("kv:layer0:request0", 128)
    warm = state.access("kv:layer0:request0", 128)
    assert (cold.logical_bytes, cold.gm_bytes, cold.onchip_bytes) == (128, 128, 128)
    assert (cold.cache_read_bytes, cold.hbm_read_bytes) == (0, 128)
    # The second GM access is real on-chip work even though it is a cache hit.
    assert (warm.logical_bytes, warm.gm_bytes, warm.onchip_bytes) == (128, 128, 128)
    assert warm.cache_read_bytes == 128 and warm.hbm_read_bytes == 0
    assert warm.hit_lines == 1 and warm.miss_lines == 0
    assert state.resident_bytes == 128
    assert state.dirty_bytes == 0


def test_capacity_change_keeps_semantic_work_but_changes_physical_traffic():
    sequence = [("kv:layer0:request0", 128), ("kv:layer0:request0", 128)]
    cold = CacheState(geometry(0))
    cold_reads = [cold.access(identity, logical).hbm_read_bytes
                  for identity, logical in sequence]
    cached = CacheState(geometry(256))
    cached_reads = [cached.access(identity, logical).hbm_read_bytes
                    for identity, logical in sequence]
    assert cold_reads == [128, 128]
    assert cached_reads == [128, 0]
    assert cold.counters["gm_bytes"] == cached.counters["gm_bytes"] == 256
    assert cold.counters["onchip_bytes"] == cached.counters["onchip_bytes"] == 256


def test_bypass_never_fills_but_keeps_service_staging_and_ready():
    state = CacheState(geometry(256))
    first = state.access("kv:layer0:request0", 128,
                         policy=CachePolicy(bypass=True, ready_ms=0.002))
    second = state.access("kv:layer0:request0", 128,
                          policy=CachePolicy(bypass=True))
    assert first.hbm_read_bytes == second.hbm_read_bytes == 128
    assert first.onchip_bytes == second.onchip_bytes == 128
    assert first.staging_bytes == 128 and first.ready_ms == 0.002
    assert first.bypassed is True and first.cold_path is False
    assert state.resident_bytes == 0
    # The direct path did not populate a cache another access could hit.
    normal = state.access("kv:layer0:request0", 128)
    assert normal.hbm_read_bytes == 128


def test_dirty_eviction_writes_back_once_and_clean_eviction_does_not():
    state = CacheState(geometry(256))
    state.access("A", 256, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    assert state.dirty_bytes == 256 and state.resident_bytes == 256
    evicted = state.access("B", 256, write=True,
                           policy=CachePolicy(write_policy="write_back"))
    assert evicted.evicted_bytes == 256 and evicted.writeback_bytes == 256
    assert evicted.hbm_write_bytes == 256
    assert evicted.hbm_write_bytes == (
        evicted.direct_write_bytes + evicted.writeback_bytes)
    e, plan = execute_access_and_writeback(evicted, state)
    assert plan.bytes == 256
    assert dict(e.serviced)["hbm"] == 512.0  # evicted A + dirty B flush
    assert state.counters["hbm_write_bytes"] == 512
    assert state.dirty_bytes == 0 and state.flush().bytes == 0
    # A write-through line is clean: its eviction adds no second HBM write.
    state = CacheState(geometry(256))
    state.access("A", 256, write=True)
    through = state.access("B", 256, write=True)
    assert through.evicted_bytes == 256 and through.writeback_bytes == 0
    assert through.hbm_write_bytes == 256


def test_write_back_over_capacity_materializes_unretained_lines_now():
    state = CacheState(geometry(128))
    result = state.access("A", 256, write=True,
                          policy=CachePolicy(write_policy="write_back"))
    # The second 128B line cannot be retained; it must reach HBM now.
    assert result.direct_write_bytes == 128
    assert result.writeback_bytes == 0
    assert result.hbm_write_bytes == 128
    assert result.hbm_write_bytes == (
        result.direct_write_bytes + result.writeback_bytes)
    assert state.resident("A") == (1, 1)
    assert state.dirty_bytes == 128
    e, plan = execute_access_and_writeback(result, state)
    assert plan.bytes == 128
    assert dict(e.serviced)["hbm"] == 256.0  # direct 128 + dirty-prefix 128
    assert result.hbm_write_bytes + plan.bytes == 256  # whole object conserved
    assert state.dirty_bytes == 0 and state.flush().bytes == 0


def test_over_capacity_write_program_submits_direct_lines():
    state = CacheState(geometry(128))
    result = state.access("A", 256, write=True,
                          policy=CachePolicy(write_policy="write_back"))
    e, plan = execute_access_and_writeback(result, state)
    assert plan.bytes == 128
    assert dict(e.serviced) == {"hbm": 256.0, "mte2": 256.0}
    assert state.dirty_bytes == 0


def test_partial_write_back_update_flushes_only_its_dirty_prefix():
    state = CacheState(geometry(256))
    state.access("A", 256)  # two clean retained lines
    assert state.resident("A") == (2, 0)
    partial = state.access("A", 128, write=True,
                           policy=CachePolicy(write_policy="write_back"))
    assert partial.direct_write_bytes == 0
    assert state.resident("A") == (2, 1)
    assert state.dirty_bytes == 128
    e, plan = execute_writeback(state)
    assert plan.bytes == 128 and dict(e.serviced)["hbm"] == 128.0
    assert state.dirty_bytes == 0  # not both retained lines
    # A full update dirties both retained lines.
    state.access("A", 256, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    assert state.resident("A") == (2, 2)
    e, plan = execute_writeback(state)
    assert plan.bytes == 256 and dict(e.serviced)["hbm"] == 256.0


def test_flush_completion_boundary_blocks_overlapping_transitions():
    state = CacheState(geometry(128))
    result = state.access("A", 256, write=True,
                          policy=CachePolicy(write_policy="write_back"))
    e = engine()
    ready = e.submit("ready", release=0)
    access = CacheAccessProgram(result, pool="ub").start(e, "access", ready)
    plan = state.flush()
    assert plan.bytes == 128 and state.dirty_bytes == 128
    # Dirty data is not cleared by creating the plan.
    for action in (
            lambda: state.access("B", 128),
            lambda: state.flush(),
            lambda: state.invalidate(),
            lambda: state.reset(),
            lambda: state.clone()):
        with pytest.raises(ModelingError):
            action()
    done = plan.start(e, "flush", access)
    assert state.dirty_bytes == 128  # completion boundary, not submission
    e.advance(until=(done,))
    assert state.dirty_bytes == 0
    assert dict(e.serviced)["hbm"] == 256.0
    assert state.counters["hbm_write_bytes"] == 256
    second = state.flush()
    assert second.bytes == 0
    e2 = engine()
    ready2 = e2.submit("ready", release=0)
    e2.advance(until=(second.start(e2, "flush2", ready2),))
    assert e2.serviced.get("hbm", 0.0) == 0.0


def test_clone_of_dirty_state_is_rejected_to_avoid_double_writeback():
    state = CacheState(geometry(128))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    with pytest.raises(ModelingError):
        state.clone()
    e, plan = execute_writeback(state)
    assert plan.bytes == 128
    assert state.clone().dirty_bytes == 0


def test_flush_cancel_releases_state_without_clearing_dirty():
    state = CacheState(geometry(128))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    plan = state.flush()
    assert state.dirty_bytes == 128
    plan.cancel()
    # Cancelling does not pretend to materialize the dirty line.
    assert state.dirty_bytes == 128
    with pytest.raises(ModelingError):
        plan.start(engine(), "cancelled", engine().submit("r"))
    # The state is usable again and the dirty line can still be flushed.
    e, plan = execute_writeback(state)
    assert plan.bytes == 128 and state.dirty_bytes == 0


def test_invalidate_rejects_dirty_and_removes_only_clean_lines():
    state = CacheState(geometry(256))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    with pytest.raises(ModelingError):
        state.invalidate()
    e, plan = execute_writeback(state)
    assert plan.bytes == 128 and dict(e.serviced)["hbm"] == 128.0
    assert state.invalidate() == 1
    assert state.resident("A") == (0, 0) and state.resident_bytes == 0
    assert state.invalidate() == 0


def test_bypass_same_identity_rejects_dirty_and_invalidates_clean_write():
    state = CacheState(geometry(256))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    with pytest.raises(ModelingError):
        state.access("A", 128, policy=CachePolicy(bypass=True))
    with pytest.raises(ModelingError):
        state.access("A", 128, write=True, policy=CachePolicy(bypass=True))
    assert state.resident("A") == (1, 1) and state.dirty_bytes == 128
    e, plan = execute_writeback(state)
    assert plan.bytes == 128
    # A clean bypass read may keep the valid cached copy.
    bypass_read = state.access("A", 128, policy=CachePolicy(bypass=True))
    assert bypass_read.hbm_read_bytes == 128
    assert state.resident("A") == (1, 0)
    # A bypass write overwrites HBM, so the clean resident copy is dropped.
    bypass_write = state.access("A", 128, write=True,
                                policy=CachePolicy(bypass=True))
    assert bypass_write.hbm_write_bytes == 128
    assert state.resident("A") == (0, 0) and state.resident_bytes == 0
    miss = state.access("A", 128)
    assert miss.hbm_read_bytes == 128


def test_normal_load_reports_staging_gap_and_binds_pool():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    load(builder, "kv_read", ("kv_store",), 1, 64, "mla_kv",
         state=state, identity="A")
    metadata = dict(builder.ops[-1].metadata)
    assert metadata["cache_access"]["staging_bytes"] == 128
    assert "cache_execution_gap" in metadata
    result = state.access("B", 128)
    with pytest.raises(ModelingError):
        CacheAccessProgram(result, pool=None).start(
            engine(), "no_pool", engine().submit("r"))
    small = engine(pool_bytes=64)
    ready = small.submit("ready", release=0)
    with pytest.raises(ExecutionError):
        CacheAccessProgram(result, pool="ub").start(small, "small", ready)
    enough = engine(pool_bytes=128)
    ready = enough.submit("ready", release=0)
    done = CacheAccessProgram(result, pool="ub").start(enough, "normal", ready)
    enough.advance(until=(done,))
    assert dict(enough.peak_allocated)["ub"] == 128.0
    assert dict(enough.allocated)["ub"] == 0


def test_read_eviction_carries_hbm_read_and_writeback_into_program():
    state = CacheState(geometry(128))
    written = state.access("A", 128, write=True,
                           policy=CachePolicy(write_policy="write_back"))
    read = state.access("B", 128)
    assert (read.hbm_read_bytes, read.hbm_write_bytes) == (128, 128)
    assert read.direct_write_bytes == 0 and read.writeback_bytes == 128
    e = engine()
    ready = e.submit("ready", release=0)
    first = CacheAccessProgram(written, pool="ub").start(e, "write", ready)
    second = CacheAccessProgram(read, pool="ub").start(e, "read", first)
    e.advance(until=(second,))
    # The read program must submit the evicted dirty line, not only its fill.
    assert dict(e.serviced) == {"hbm": 256.0, "mte2": 256.0}
    assert dict(e.allocated)["ub"] == 0


def test_object_larger_than_capacity_streams_and_retains_only_its_prefix():
    state = CacheState(geometry(128))
    first = state.access("stream", 300)
    second = state.access("stream", 300)
    assert first.hbm_read_bytes == 384 and first.fill_bytes == 128
    assert second.hit_lines == 1 and second.miss_lines == 2
    assert second.hbm_read_bytes == 256
    assert state.resident_bytes == 128


def test_engine_program_prices_real_service_and_independent_rate_change():
    state = CacheState(geometry(512))
    result = state.access("kv:layer0:request0", 300)
    observed = []
    for rate in (1000.0, 2000.0):
        e = engine(hbm_rate=rate)
        ready = e.submit("ready", release=0)
        done = CacheAccessProgram(result, pool="ub").start(e, "access", ready)
        e.advance(until=(done,))
        observed.append((done.time - ready.time, dict(e.serviced),
                         dict(e.peak_allocated), dict(e.allocated)))
    assert observed[0][1] == observed[1][1]
    assert observed[0][1]["hbm"] == 384.0
    assert observed[0][1]["mte2"] == 320.0
    assert observed[0][2]["ub"] == 128.0
    # 384 B / rate + 320 B / 2000 B/ms; only the HBM term responds.
    assert observed[0][0] == pytest.approx(0.544)
    assert observed[1][0] == pytest.approx(0.352)
    assert observed[0][3]["ub"] == 0  # staging is released on completion


def test_direct_program_binds_positive_ready_and_staging():
    state = CacheState(geometry(256))
    result = state.access("stream", 128,
                          policy=CachePolicy(bypass=True, ready_ms=0.002))
    e = engine()
    ready = e.submit("ready", release=0)
    done = CacheAccessProgram(result, pool="ub").start(e, "direct", ready)
    e.advance(until=(done,))
    # 128 B HBM + 128 B on-chip copy + 2 us ready.
    assert done.time - ready.time == pytest.approx(0.194)
    assert dict(e.peak_allocated)["ub"] == 128.0
    assert dict(e.allocated)["ub"] == 0


def test_direct_program_requires_pool_and_rejects_insufficient_buffer():
    state = CacheState(geometry(256))
    result = state.access("stream", 128, policy=CachePolicy(bypass=True))
    with pytest.raises(ModelingError):
        CacheAccessProgram(result, pool=None).start(
            engine(), "no_pool", engine().submit("r"))
    small = engine(pool_bytes=64)
    ready = small.submit("ready", release=0)
    with pytest.raises(ExecutionError):
        CacheAccessProgram(result, pool="ub").start(small, "small", ready)


def test_hit_program_does_not_require_an_unused_hbm_resource():
    state = CacheState(geometry(512))
    state.access("kv:layer0:request0", 128)
    hit = state.access("kv:layer0:request0", 128)
    e = Engine([Resource("mte2", 1000.0)], pools={"ub": 1024})
    ready = e.submit("ready", release=0)
    done = CacheAccessProgram(hit, hbm_resource="absent", pool="ub").start(
        e, "hit", ready)
    e.advance(until=(done,))
    assert dict(e.serviced) == {"mte2": 128.0}


def test_second_read_binds_after_first_completion_without_second_hbm_fill():
    state = CacheState(geometry(512))
    first = state.access("kv:layer0:request0", 300)
    second = state.access("kv:layer0:request0", 300)
    e = engine()
    ready = e.submit("ready", release=0)
    a = CacheAccessProgram(first, pool="ub").start(e, "first", ready)
    b = CacheAccessProgram(second, pool="ub").start(e, "second", a)
    e.advance(until=(b,))
    assert dict(e.serviced) == {"hbm": 384.0, "mte2": 640.0}
    assert b.time >= a.time
    assert first.hbm_bytes == 384 and second.hbm_bytes == 0


def test_store_binds_declared_onchip_source_and_line_rounded_output():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    rows = 5
    identity = cache_identity("window", layer=0, request="r0", block=0)
    store(builder, "kv_store", ("rotary",), rows, plan.cache_row_elements,
          "window", state=state, identity=identity, source_residency="on_chip")
    op = builder.ops[-1]
    fmt = cache_format(case.model, "window")
    stored = rows * fmt.row_bytes(plan.cache_row_elements)
    source = rows * plan.cache_row_elements * 2
    assert op.mte2_bytes == source  # declared on-chip source read
    assert op.mte3_bytes == stored
    assert op.hbm_bytes == pytest.approx(128 * ((stored + 127) // 128))
    assert op.hbm_bytes > stored  # line padding is explicit physical traffic
    access = dict(op.metadata)["cache_access"]
    ledger = dict(op.metadata)["cache_hbm_write_ledger"]
    assert access["hbm_write_bytes"] == op.hbm_bytes
    assert access["gm_bytes"] == access["onchip_bytes"] == stored
    assert access["bypassed"] is False and access["hbm_read_bytes"] == 0
    assert ledger["hbm_write_bytes"] == (
        ledger["direct_write_bytes"] + ledger["writeback_bytes"])
    assert dict(op.metadata)["cache_source_residency"] == "on_chip"
    assert dict(op.metadata)["cache_representation"]["source_read_scope"] == "on_chip_not_HBM"
    assert dict(op.metadata)["cache_access"]["staging_bytes"] == 128
    assert "cache_execution_gap" in dict(op.metadata)


def test_unresolved_source_read_remains_external_hbm():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    store(builder, "kv_store", ("rotary",), 5, plan.cache_row_elements,
          "window", state=state, identity="A")
    op = builder.ops[-1]
    fmt = cache_format(case.model, "window")
    source = 5 * plan.cache_row_elements * 2
    line_rounded = 128 * ((5 * fmt.row_bytes(plan.cache_row_elements) + 127) // 128)
    metadata = dict(op.metadata)
    assert metadata["cache_source_residency"] == "unresolved"
    assert metadata["cache_representation"]["source_read_scope"] == (
        "unresolved_source_read_counted_in_HBM")
    assert op.hbm_bytes == source + line_rounded
    assert op.mte2_bytes == source
    assert op.mte3_bytes == 5 * fmt.row_bytes(plan.cache_row_elements)


def test_plain_cache_store_keeps_row_abi_and_declared_source_scope():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    store(builder, "plain_store", ("rotary",), 3, 64, "mla_kv", state=state,
          identity="plain", source_residency="on_chip")
    op = builder.ops[-1]
    assert op.local_shape == (3, 128)
    assert op.vector_ops == 3 * 128
    assert op.dtype == "bf16"
    assert op.hbm_bytes == 384  # 3 rows of 128B, not source+output
    assert op.mte2_bytes == op.mte3_bytes == 384
    assert "cache_execution_gap" in dict(op.metadata)


def test_undeclared_store_preserves_legacy_A3_and_A5_byte_semantics():
    # A3 DeepSeek V4 has window_compressed with a plain BF16 cache row.
    builder, case, plan = parsed_builder(PLAIN_A3_CASE)
    store(builder, "kv_store", ("rotary",), 2, plan.cache_row_elements, "window")
    op = builder.ops[-1]
    assert dict(op.metadata)["cache_resolution"] == "legacy_no_cache_declaration"
    assert op.local_shape == (2, 1024)
    assert op.padded_shape == (8, 1024)
    assert op.hbm_bytes == 2 * (2 * 1024)
    assert op.mte2_bytes == op.mte3_bytes == 2 * 1024
    assert op.vector_ops == 2 * 1024
    # A5 packed legacy grouped path keeps source + stored bytes, no line level.
    builder, case, plan = parsed_builder()
    store(builder, "kv_store", ("rotary",), 2, plan.cache_row_elements, "window")
    op = builder.ops[-1]
    fmt = cache_format(case.model, "window")
    stored = 2 * fmt.row_bytes(plan.cache_row_elements)
    source = 2 * plan.cache_row_elements * 2
    assert dict(op.metadata)["cache_resolution"] == "legacy_no_cache_declaration"
    assert op.hbm_bytes == source + stored
    assert op.mte2_bytes == source and op.mte3_bytes == stored


def test_store_then_load_reuses_only_an_explicit_identity():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    identity = cache_identity("window", layer=0, request="r0", block=0)
    store(builder, "kv_store", ("rotary",), 5, plan.cache_row_elements,
          "window", state=state, identity=identity, source_residency="on_chip")
    load(builder, "kv_read", ("kv_store",), 5, plan.cache_row_elements,
         "window", state=state, identity=identity)
    written, read = builder.ops
    assert read.hbm_bytes == 0
    assert read.mte2_bytes == written.mte3_bytes
    assert dict(read.metadata)["cache_access"]["hit_lines"] == dict(
        written.metadata)["cache_access"]["hbm_write_lines"]


def test_identity_axes_do_not_alias_requests_or_blocks():
    first = cache_identity("window", layer=0, request="r0", block=0)
    second = cache_identity("window", layer=0, request="r0", block=1)
    comma = cache_identity("window", layer=0, request="a,b")
    separate = cache_identity("window", layer=0, request="a")
    assert first != second
    assert comma != separate
    assert cache_identity("window", layer=1, request="r0") != cache_identity(
        "window", layer=0, request="r0")
    with pytest.raises(ModelingError):
        cache_identity("window")
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    store(builder, "kv_store", ("rotary",), 1, 64, "mla_kv", state=state,
          identity=first, source_residency="on_chip")
    load(builder, "kv_read", ("kv_store",), 1, 64, "mla_kv", state=state,
         identity=second)
    assert builder.ops[-1].hbm_bytes == 128  # different block: cold miss
    load(builder, "kv_read_again", ("kv_store",), 1, 64, "mla_kv", state=state,
         identity=first)
    assert builder.ops[-1].hbm_bytes == 0  # same block: retained hit


def test_no_identity_takes_cold_path_and_never_fills_shared_state():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    store(builder, "kv_store", ("rotary",), 1, 64, "mla_kv", state=state,
          source_residency="on_chip")
    load(builder, "kv_read", ("kv_store",), 1, 64, "mla_kv", state=state)
    written, read = builder.ops
    for op in (written, read):
        metadata = dict(op.metadata)
        assert metadata["cache_access"]["cold_path"] is True
        assert metadata["cache_cold_path"] == "unidentified_storage"
    assert read.hbm_bytes == 128 and state.resident_bytes == 0


def test_geometry_export_comes_from_the_state_and_conflicts_are_rejected():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    store(builder, "kv_store", ("rotary",), 1, 64, "mla_kv", state=state,
          identity="A", source_residency="on_chip")
    metadata = dict(builder.ops[-1].metadata)
    assert metadata["cache_geometry"]["capacity_bytes"] == 4096
    assert metadata["cache_geometry"]["line_bytes"] == 128
    with pytest.raises(ModelingError):
        store(builder, "kv_bad", ("rotary",), 1, 64, "mla_kv", state=state,
              geometry=geometry(128), identity="A")
    with pytest.raises(ModelingError):
        load(builder, "kv_no_geometry", ("kv_store",), 1, 64, "mla_kv")
    with pytest.raises(ModelingError):
        store(builder, "kv_no_geometry", ("rotary",), 1, 64, "mla_kv",
              source_residency="on_chip")
    with pytest.raises(ModelingError):
        store(builder, "kv_identity_only", ("rotary",), 1, 64, "mla_kv",
              identity="A")


def test_builder_attached_state_controls_call_sites_without_signature_changes():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(4096))
    builder.cache_state = state
    identity = "kv:layer0:block0"
    store(builder, "kv_store", ("rotary",), 5, plan.cache_row_elements,
          "window", identity=identity, source_residency="on_chip")
    load(builder, "kv_read", ("kv_store",), 5, plan.cache_row_elements,
         "window", identity=identity)
    written, read = builder.ops
    assert dict(written.metadata)["cache_resolution"] == "builder_cache_state"
    assert read.hbm_bytes == 0 and state.resident_bytes > 0


def test_declared_store_without_state_is_cold_but_explicitly_declared():
    builder, case, plan = parsed_builder()
    store(builder, "kv_store", ("rotary",), 5, plan.cache_row_elements, "window",
          geometry=geometry(4096), identity="A", source_residency="on_chip")
    metadata = dict(builder.ops[-1].metadata)
    assert metadata["cache_resolution"] == "ephemeral_cold"
    assert metadata["cache_access"]["cold_path"] is False
    assert builder.ops[-1].hbm_bytes == 2816
    # A direct path without a geometry cannot be priced or executed here.
    with pytest.raises(ModelingError):
        store(builder, "kv_direct", ("rotary",), 5, plan.cache_row_elements,
              "window", bypass=True, ready_ms=0.001)


def test_direct_store_reports_the_graph_envelope_gap():
    builder, case, plan = parsed_builder()
    store(builder, "kv_store", ("rotary",), 1, 64, "mla_kv",
          geometry=geometry(4096), identity="A", source_residency="on_chip",
          bypass=True, ready_ms=0.001)
    metadata = dict(builder.ops[-1].metadata)
    assert "cache_execution_gap" in metadata
    assert metadata["cache_access"]["bypassed"] is True


def test_write_back_store_materializes_unretained_lines_and_flushes_prefix():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(128))
    store(builder, "kv_store", ("rotary",), 4, 64, "mla_kv", state=state,
          identity="A", write_policy="write_back", source_residency="on_chip")
    op = builder.ops[-1]
    metadata = dict(op.metadata)
    assert op.hbm_bytes == 384  # 3 of 4 lines cannot be retained
    assert metadata["cache_hbm_write_ledger"] == {
        "direct_write_bytes": 384, "writeback_bytes": 0,
        "hbm_write_bytes": 384}
    assert state.resident("A") == (1, 1)
    assert state.dirty_bytes == 128
    e, plan = execute_writeback(state)
    assert plan.bytes == 128 and dict(e.serviced)["hbm"] == 128.0
    assert op.hbm_bytes + plan.bytes == 512  # whole written object conserved


def test_load_op_carries_dirty_eviction_writeback():
    builder, case, plan = parsed_builder()
    state = CacheState(geometry(128))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    load(builder, "kv_read", ("kv_store",), 1, 64, "mla_kv", state=state,
         identity="B")
    op = builder.ops[-1]
    metadata = dict(op.metadata)
    assert op.hbm_bytes == 256
    assert op.mte2_bytes == 128
    assert metadata["cache_access"]["hbm_read_bytes"] == 128
    assert metadata["cache_access"]["hbm_write_bytes"] == 128
    assert metadata["cache_hbm_write_ledger"]["writeback_bytes"] == 128


def test_zero_byte_access_does_not_flush_or_dirty_resident_state():
    state = CacheState(geometry(4096))
    state.access("A", 128, write=True,
                 policy=CachePolicy(write_policy="write_back"))
    before = (state.resident_bytes, state.dirty_bytes)
    result = state.access("A", 0, write=True,
                          policy=CachePolicy(write_policy="write_through"))
    assert result.hbm_bytes == 0 and result.logical_bytes == 0
    assert (state.resident_bytes, state.dirty_bytes) == before


def test_new_geometries_predict_without_observations():
    for width in (1, 37, 513, 1025):
        for rows in (1, 3):
            fmt = CacheFormat("int8", group_size=128, scale_dtype="fp32")
            stored = fmt.stored_bytes(width, rows)
            state = CacheState(geometry(max(128, 4 * 128)))
            result = state.access(f"unseen:{width}:{rows}", stored)
            expected = 128 * ((stored + 127) // 128)
            assert result.hbm_read_bytes == expected
            assert result.gm_bytes == 32 * ((stored + 31) // 32)
            assert result.logical_bytes == stored


def test_cache_format_rounds_subbyte_rows_once_and_reports_transactions():
    assert CacheFormat("fp4_e2m1").row_bytes(3) == 2
    assert CacheFormat("fp4_e2m1").row_bytes(5) == 3
    assert CacheFormat("uint8", packed_elements=2).row_bytes(5) == 3
    with pytest.raises(ModelingError):
        CacheFormat("fp4_e2m1", packed_elements=2)
    packed = CacheFormat("fp4_e2m1", group_size=16, scale_dtype="bf16")
    assert packed.row_bytes(576) == 288 + 72
    assert packed.stored_bytes(576, 2) == 720
    assert packed.transaction_count(576, 2, 128) == 6
    assert packed.transaction_bytes(576, 2, 128) == 768


def test_summary_aggregates_work_without_one_object_per_line():
    state = CacheState(geometry(2048))
    summary = AccessSummary()
    for index in range(64):
        summary.add(state.access(f"request{index}", 512))
    assert summary.accesses == 64
    assert summary.logical_bytes == 64 * 512
    assert summary.resident_peak_bytes == 2048
    assert state.resident_identities <= 16  # 2048 / 128 retained lines
    assert summary.onchip_bytes == summary.gm_bytes == summary.logical_bytes
    assert summary.direct_write_bytes == 0
    # Residency is an access-side requirement, not an allocation.
    assert "resident_bytes" not in state.snapshot()["counters"]
