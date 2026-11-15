"""Physical overlap bounds for a UB-resident row-normalization pipeline.

[Role] Exercise row continuation on real declared A2/A3/A5 supplies.
[Invariants] One-row launch/fill/drain is unchanged; overlap cannot consume
  more time than preceding pipe service or exist without two resident slots.
[Timeouts & Locks] Bounded in-process arithmetic; no waits or locks.
[Side-Effect Policy] No device work, capture reads or filesystem writes.
"""
from dataclasses import replace

import pytest

from mindie_sim.prediction.piecewise.native_clock_physics import aiv_times
from mindie_sim.prediction.piecewise.native_clock_rates import rates_from_supply


def _times(rates, *, batch, width=4096, ops_per_element=7, dtype_bytes=2):
    blocks = min(8, rates.aiv)
    rows = blocks * batch
    return aiv_times(
        kernel_class="row_norm", rows=rows, width=width,
        read_bytes=(rows + blocks) * width * dtype_bytes,
        write_bytes=rows * (width * dtype_bytes + 4),
        blocks=blocks, units=rates.aiv,
        vector_ops=rows * width * ops_per_element,
        dtype_bytes=dtype_bytes, rates=rates)


@pytest.mark.parametrize("sku", ("Ascend910B4", "Ascend910_9392", "Ascend950DT_9582"))
@pytest.mark.parametrize("dtype_bytes", (2, 4))
def test_row_pipeline_needs_two_resident_slots_and_preserves_single_row(sku, dtype_bytes):
    rates = rates_from_supply(hardware=sku)
    one_slot = replace(rates, ub_bytes=3 * 4096 * dtype_bytes)
    assert _times(rates, batch=1, dtype_bytes=dtype_bytes) == _times(
        one_slot, batch=1, dtype_bytes=dtype_bytes)
    savings = []
    for batch in (2, 3, 8):
        serial = _times(one_slot, batch=batch, dtype_bytes=dtype_bytes)
        pipelined = _times(rates, batch=batch, dtype_bytes=dtype_bytes)
        assert pipelined.total_ms() <= serial.total_ms()
        savings.append(serial.total_ms() - pipelined.total_ms())
        # Pipe work, start-up and final store are conserved, rather than
        # divided by the batch size to get a desired elapsed duration.
        assert pipelined.body == serial.body
        assert pipelined.tail == serial.tail
    # A load-bound launch may already hide all scalar work. Such a case must
    # remain equal, while the shorter rows expose the continuation overlap.
    assert any(saving > 0 for saving in savings)


@pytest.mark.parametrize("batch", (2, 4, 16))
def test_row_continuation_overlap_is_bounded_by_preceding_pipe_service(batch):
    rates = rates_from_supply(hardware="Ascend950DT_9582")
    one_slot = replace(rates, ub_bytes=3 * 4096 * 2)
    empty_work = dict(kernel_class="row_norm", rows=8 * batch, width=4096,
                      read_bytes=0, write_bytes=0, vector_ops=0,
                      blocks=8, units=rates.aiv)
    assert aiv_times(**empty_work, rates=rates) == aiv_times(
        **empty_work, rates=one_slot)
    previous_saving = 0.0
    for ops in (1, 3, 7):
        serial = _times(one_slot, batch=batch, ops_per_element=ops)
        pipelined = _times(rates, batch=batch, ops_per_element=ops)
        saving = serial.total_ms() - pipelined.total_ms()
        # No extra overlap can exceed the earlier rows' entire body window.
        available_ms = max(dict(serial.body).values()) * (batch - 1) / batch
        assert 0 < saving <= available_ms + 1e-15
        assert saving >= previous_saving - 1e-15
        previous_saving = saving


def test_faster_vector_does_not_lose_overlap_in_a_dma_bound_row_pipeline():
    rates = rates_from_supply(hardware="Ascend950DT_9582")
    previous = float("inf")
    for speedup in (1, 2, 4, 8):
        result = _times(replace(rates, vector=rates.vector * speedup), batch=2)
        assert result.total_ms() <= previous
        previous = result.total_ms()
