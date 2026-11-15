"""Low-level ABI, context and ownership regressions (engine-arena/2).

[Role] Exercise the raw ``msim_ctx_*`` surface: probe identity, independent
  contexts, 64-bit sequences, capacity faults and link-slice ownership.
[Invariants] Every test uses the shared isolated cache and asserts a kernel
  status cell; no scheduler arithmetic is reimplemented in Python.
[Timeouts & Locks] No threads, sleeps or unbounded loops.
[Side-Effect Policy] Writes only its own temporary arena buffers.
"""
from __future__ import annotations

import ctypes

import pytest

from mindie_sim.prediction.runtime import accel_bridge, accel_build
from mindie_sim.prediction.runtime.accel_tables import (
    ABI_VERSION, Arena, E_ABI, E_FULL, E_RANGE, E_VALUE, KernelFault,
    PROBE_GOLDEN, bits,
)


def _library():
    status = accel_build.build_accel()
    assert status["ok"], status.get("detail") or status.get("reason")
    return accel_bridge.load_library(status)


def _raw_submit(arena, rows, links):
    nrows, nlinks = len(rows), len(links)
    raw_rows = (ctypes.c_uint64 * (nrows * 12))()
    for i, row in enumerate(rows):
        for cell, value in enumerate(row):
            raw_rows[i * 12 + cell] = value
    raw_links = (ctypes.c_uint64 * (nlinks * 6))()
    for i, link in enumerate(links):
        for cell, value in enumerate(link):
            raw_links[i * 6 + cell] = value
    slots = (ctypes.c_int32 * nrows)()
    rc = arena.library.msim_ctx_submit(
        arena.ctx, raw_rows, nrows, 12, raw_links, nlinks, 6, None, 0,
        None, None, 0, slots, nrows, arena.st, 8)
    return rc, [int(x) for x in arena.st]


def test_abi_probe_golden_and_version():
    library = _library()
    assert int(library.msim_abi_version()) == ABI_VERSION
    probe = (ctypes.c_uint64 * 16)()
    assert int(library.msim_abi_probe(probe, 16)) == 0
    assert tuple(int(x) for x in probe) == PROBE_GOLDEN
    assert int(library.msim_abi_probe(probe, 4)) != 0

def test_concurrent_kernel_contexts_are_independent():
    library = _library()
    arenas = [Arena(library) for _ in range(2)]
    for arena, capacity in zip(arenas, (256.0, 512.0)):
        arena.configure([capacity], [1], [0])
        rows = [(index + 1, 0, 0, 1, 0, 0.0, 0.0, 0.0,
                 [(0, 1.0, float("inf"), ())]) for index in range(128)]
        assert len(arena.submit(rows)) == 128
    results = []
    for arena, capacity in zip(arenas, (256.0, 512.0)):
        now, boundary, steps, ncomp, nstart = arena.run(
            0.0, float("inf"), float("inf"), 100000)
        busy, serv = arena.counters()
        results.append((ncomp, busy[0], serv[0], capacity))
        arena.close()
    for ncomp, busy, serv, capacity in results:
        assert ncomp == 128
        assert abs(serv - 128.0) < 1e-12
        assert abs(busy - 128.0 / capacity) < 1e-12

def test_kernel_preserves_64_bit_sequences():
    library = _library()
    for big in (2 ** 31 + 100, 2 ** 63 + 7, 2 ** 64 - 1):
        arena = Arena(library)
        arena.configure([100.0], [1], [0])
        arena.submit([(big, 0, 0, 2 ** 64 - 2, 0, 0.0, 0.0, 0.0,
                       [(0, 1.0, float("inf"), ())])])
        arena.run(0.0, float("inf"), float("inf"), 100)
        assert arena.completions(1)[0][0] == big
        arena.close()

def test_unrepresentable_priority_is_rejected_not_wrapped():
    """R4: a priority above INT64_MAX is a range error, never a negative one."""
    library = _library()
    arena = Arena(library)
    arena.configure([10.0], [1], [0])
    with pytest.raises(KernelFault) as info:
        arena.submit([(1, 0, 2 ** 63, 1, 0, 0.0, 0.0, 0.0,
                       [(0, 1.0, float("inf"), ())])])
    assert info.value.code == 1
    with pytest.raises(KernelFault) as cohort:
        arena.submit([(1, 0, 0, 2 ** 64 + 1, 0, 0.0, 0.0, 0.0,
                       [(0, 1.0, float("inf"), ())])])
    assert cohort.value.code == 1
    with pytest.raises(KernelFault) as sequence:
        arena.submit([(2 ** 64, 0, 0, 1, 0, 0.0, 0.0, 0.0,
                       [(0, 1.0, float("inf"), ())])])
    assert sequence.value.code == 1
    arena.close()

def test_kernel_fault_is_fatal_and_not_replayed():
    library = _library()
    arena = Arena(library)
    arena.configure([10.0], [1], [0])
    arena.submit([(1, 0, 0, 1, 0, 0.0, 0.0, 0.0,
                   [(0, 1.0, float("inf"), ())])])
    arena.hdr[0] = ABI_VERSION + 1
    arena.hdr[1] = 0
    arena.hdr[2] = 10
    arena.io[0] = 0.0
    arena.io[1] = float("inf")
    arena.io[3] = float("inf")
    rc = library.msim_ctx_run(
        arena.ctx, arena.hdr, 16, arena.st, 8, arena.io, 8,
        arena.comp, 3 * 4096, arena.svc, 3 * 32768, arena.started, 2 * 4096)
    assert rc != 0 and int(arena.st[1]) == E_ABI
    assert float(arena.io[0]) == 0.0
    arena.close()

def test_overlapping_link_ownership_is_rejected_before_mutation():
    library = _library()
    arena = Arena(library)
    arena.configure([10.0], [1], [0])
    rows = [(index + 1, 0, 0, 1, 0, 0, bits(0.0), bits(0.0), bits(0.0),
             1, 0, 0) for index in range(65)]
    links = [(0, 0, bits(1.0), bits(float("inf")), 0, 0)]
    rc, status = _raw_submit(arena, rows, links)
    assert rc != 0 and status[1] == E_RANGE
    # The rejected preflight mutated no task/link cell: a valid submit still
    # runs and completes on the same context.
    arena.submit([(1, 0, 0, 1, 0, 0.0, 0.0, 0.0,
                   [(0, 1.0, float("inf"), ())])])
    now, _boundary, _steps, ncomp, _nstart = arena.run(
        0.0, float("inf"), float("inf"), 100)
    assert ncomp == 1 and now == 0.1
    arena.close()

def test_malformed_abi_inputs_are_rejected_before_mutation():
    library = _library()
    arena = Arena(library)
    arena.configure([10.0], [1], [0])
    arena.submit([(1, 0, 0, 1, 0, 0.0, 0.0, 0.0,
                   [(0, 1.0, float("inf"), ())])])
    arena.hdr[0] = ABI_VERSION
    arena.hdr[1] = 0
    arena.hdr[2] = 10
    arena.io[0] = 0.0
    arena.io[1] = float("inf")
    arena.io[3] = float("inf")
    # A truthful ABI header with no completion capacity is a buffer fault,
    # established independently of the version/field checks below.
    rc = library.msim_ctx_run(
        arena.ctx, arena.hdr, 16, arena.st, 8, arena.io, 8,
        arena.comp, 0, arena.svc, 3 * 32768, arena.started, 2 * 4096)
    assert rc != 0 and int(arena.st[1]) == E_FULL
    arena.hdr[0] = ABI_VERSION + 1
    rc = library.msim_ctx_run(
        arena.ctx, arena.hdr, 16, arena.st, 8, arena.io, 8,
        arena.comp, 3 * 4096, arena.svc, 3 * 32768, arena.started, 2 * 4096)
    assert rc != 0 and int(arena.st[1]) == E_ABI
    arena.hdr[0] = ABI_VERSION
    arena.io[0] = float("nan")
    rc = library.msim_ctx_run(
        arena.ctx, arena.hdr, 16, arena.st, 8, arena.io, 8,
        arena.comp, 3 * 4096, arena.svc, 3 * 32768, arena.started, 2 * 4096)
    assert rc != 0 and int(arena.st[1]) == E_VALUE
    now, _boundary, _steps, ncomp, _nstart = arena.run(
        0.0, float("inf"), float("inf"), 100)
    assert ncomp == 1 and now == 0.1
    arena.close()
