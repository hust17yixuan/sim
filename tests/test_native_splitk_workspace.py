"""Independent conservation and supply invariants for split-K workspace work.

[Role] Check FP32 partials, AIV reduction services and the AIC write path.
[Invariants] No observations or timing targets; exact work and monotone supply.
[Timeouts & Locks] Bounded CPU arithmetic, no locks or waits.
[Side-Effect Policy] No filesystem, device or external access.
"""
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.piecewise import native_clock_physics as clock
from mindie_sim.prediction.piecewise.native_catalog_builder import (
    KernelSupply, catalog_from_drafts,
)
from mindie_sim.prediction.piecewise.native_clock_drafts import (
    _a5_aiv_drafts, _validated_catalog_times, matrix_drafts,
)
from mindie_sim.prediction.piecewise.native_clock_rates import NativeRates, rates_from_supply
from mindie_sim.prediction.piecewise.native_cube_physics import (
    FIXPIPE_TILE_MS, KernelTimes, aic_body_traffic, matrix_times,
    split_k_for, split_k_workspace,
)
from mindie_sim.prediction.runtime.contracts import BindingFrame
from mindie_sim.schema.common import ModelingError


def workspace(*, rates=None, **overrides):
    args = dict(m=37, n=257, split=3, owners=8, out_dtype_bytes=2)
    args.update(overrides)
    return split_k_workspace(**args, rates=rates or NativeRates())


def services(work, rates=None):
    # This test declares an artificial instruction/flag schedule explicitly.
    # Its numbers are not defaults or claims about a captured native kernel.
    return dict(work.services_ms(
        rates or NativeRates(), scalar_instructions_per_block=17,
        sync_cycles=11))


def matrix(*, m=32, n=256, k=4096, split=4, rates=None, width=2):
    return matrix_times(
        m=m, n=n, k=k, read_bytes=2 * (m + n) * k,
        write_bytes=width * m * n, weight_bytes=2 * n * k,
        act_bytes=2 * m * k, out_bytes=width * m * n,
        owners=32, units=32, kind="bf16", rates=rates or NativeRates(),
        split_k=split, out_dtype_bytes=width)


@pytest.mark.parametrize("split", [2, 3, 5, 16])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_workspace_conserves_partial_sums_adds_and_output(split, width):
    work = workspace(split=split, out_dtype_bytes=width)
    elements = 37 * 257
    assert work.partial_bytes == split * elements * 4
    assert work.output_bytes == elements * max(2, width)
    assert work.add_elements == (split - 1) * elements
    assert work.cast_elements == (0 if width == 4 else elements)
    assert work.aiv_blocks == 16
    assert work.spill_bytes == 0


def test_vector_supply_uses_fp32_lane_bytes_not_element_count():
    r = NativeRates()
    work = workspace()
    t = services(work, r)
    lane_bytes = 4 * (3 * 37 * 257)
    assert t["vector"] == pytest.approx(lane_bytes / (r.vector / 4) * 1000)
    assert t["mte2_aiv"] == pytest.approx(work.partial_bytes / (r.mte2_aiv / 4) * 1000)
    assert t["mte3"] == pytest.approx(work.output_bytes / (r.mte3 / 4) * 1000)


@pytest.mark.parametrize("axis,values", [
    ("m", [1, 7, 16, 31, 32, 33, 96, 127, 128, 129, 512]),
    ("n", [15, 16, 127, 128, 255, 256, 257, 1535, 1536, 1537, 3335, 3336, 3337, 4097]),
    ("split", [1, 2, 3, 4, 5, 8, 16]),
])
def test_workspace_work_and_service_are_nondecreasing(axis, values):
    previous = None
    for value in values:
        work = workspace(**{axis: value})
        current = (work.partial_bytes, work.output_bytes, work.add_elements,
                   work.cast_elements, work.chunks_per_block,
                   *services(work).values())
        if previous is not None:
            assert all(a <= b for a, b in zip(previous, current))
        previous = current


@pytest.mark.parametrize("field", ["mte2_aiv", "vector", "mte3", "hbm", "scalar", "clock_hz"])
def test_faster_supply_never_increases_service(field):
    base = replace(NativeRates(), l2_bytes=0, hbm=1e9)
    work = workspace(rates=base)
    before = services(work, base)
    faster = replace(base, **{field: getattr(base, field) * 2})
    after = services(work, faster)
    assert all(after[key] <= value for key, value in before.items())
    pipe = {"hbm": "mte2_aiv", "clock_hz": "sync"}.get(field, field)
    if field != "mte2_aiv":
        assert after[pipe] < before[pipe]
    else:
        resident = workspace()
        assert services(resident, faster)[pipe] < services(resident, base)[pipe]


def test_resident_workspace_does_not_consume_hbm_read_service():
    r = NativeRates()
    work = workspace(rates=r)
    assert services(work, r) == services(work, replace(r, hbm=1))


def test_l2_retention_boundary_is_continuous():
    base = workspace()
    amounts = [0, 1, 127, 128, 129, base.partial_bytes]
    for spill in amounts:
        r = replace(NativeRates(), l2_bytes=base.partial_bytes - spill)
        assert workspace(rates=r).spill_bytes == spill


def test_ub_chunks_preserve_work_and_fit_live_buffers():
    small = workspace(rates=replace(NativeRates(), ub_bytes=4096))
    large = workspace(rates=replace(NativeRates(), ub_bytes=8192))
    assert small.chunk_elements * 10 <= 4096
    assert (small.chunk_elements + 32) * 10 > 4096
    assert small.chunks_per_block >= large.chunks_per_block
    assert small.partial_bytes == large.partial_bytes
    assert services(small) == services(large)


def test_reshape_and_former_shape_boxes_do_not_change_workspace_law():
    # Equal element counts have equal work; no dependence on a panel cutoff.
    assert workspace(m=3, n=4096) == workspace(m=24, n=512)
    for center in (1536, 3336, 4096):
        left, mid, right = [services(workspace(n=n)) for n in range(center - 1, center + 2)]
        for pipe in ("mte2_aiv", "vector", "mte3"):
            assert right[pipe] - mid[pipe] == pytest.approx(mid[pipe] - left[pipe])


def test_more_participating_aivs_do_not_borrow_idle_private_bandwidth():
    r = NativeRates()
    few, many = workspace(owners=4), workspace(owners=8)
    for pipe in ("mte2_aiv", "vector", "mte3"):
        assert services(few, r)[pipe] == pytest.approx(2 * services(many, r)[pipe])


def test_split_fixpipe_writes_fp32_partials_at_participating_supply():
    r = NativeRates()
    actual = dict(matrix(rates=r).body)["fixpipe"]
    # Two 128-column spatial panels, four K slices => eight AICs.
    expected_payload = (4 * 32 * 256 * 4) / (r.fixpipe * 8 / 32) * 1000
    expected_tile_issue = (2 * 16 * 4) * FIXPIPE_TILE_MS
    assert actual == pytest.approx(expected_payload + expected_tile_issue)


def test_fixpipe_service_is_independent_of_k_and_monotone_in_output_work():
    values = [dict(matrix(k=k).body)["fixpipe"] for k in (2048, 4096, 8192)]
    assert values == [values[0]] * len(values)
    for axis in ("m", "n"):
        # Fixed tile grid and split: growing output cannot reduce its service.
        values = [dict(matrix(**{axis: size}).body)["fixpipe"] for size in range(17, 32)]
        assert values == sorted(values)


def test_one_slice_has_no_workspace_or_reduction_work():
    work = workspace(split=1)
    assert work.partial_bytes == work.output_bytes == work.add_elements == 0
    assert work.chunks_per_block == work.aiv_blocks == 0
    times = dict(work.services_ms(NativeRates(), scalar_instructions_per_block=0, sync_cycles=0))
    assert set(times.values()) == {0.0}


def test_invalid_geometry_and_undeclared_capacity_are_rejected():
    with pytest.raises(ValueError):
        workspace(split=0)
    with pytest.raises(ValueError):
        workspace(rates=replace(NativeRates(), ub_bytes=1))
    with pytest.raises(ValueError):
        workspace().services_ms(NativeRates(), scalar_instructions_per_block=-1, sync_cycles=0)


def traffic(*, m=37, n=257, k=131, split=3, width=2, rates=None):
    work = workspace(m=m, n=n, split=split, out_dtype_bytes=width, rates=rates)
    # An explicitly declared schedule: three weight reads and five activation
    # reads at L1, two weight reads at the GM/L2 boundary, one HBM input read.
    weights, activations = 2 * n * k, 2 * m * k
    body = aic_body_traffic(
        operand_bytes=3 * weights + 5 * activations,
        first_touch_read_bytes=2 * weights + activations,
        hbm_read_bytes=weights + activations,
        out_bytes=width * m * n, workspace=work)
    return body, work


@pytest.mark.parametrize("split", [1, 2, 3, 5, 16])
@pytest.mark.parametrize("width", [2, 4])
def test_aic_stores_conserve_each_boundary_without_reduction_output(split, width):
    m, n, k = 3, 7, 11
    body, work = traffic(m=m, n=n, k=k, split=split, width=width)
    # Independently enumerate output slots, including every producer slice.
    stores = sum((4 if split > 1 else width)
                 for _slice in range(split) for _row in range(m) for _col in range(n))
    weights, activations = 2 * n * k, 2 * m * k
    assert body.store_bytes == stores
    assert body.operand_bytes == 3 * weights + 5 * activations
    assert body.panel_bytes - body.operand_bytes == stores
    assert body.first_touch_bytes - (2 * weights + activations) == stores
    assert body.hbm_bytes - (weights + activations) == (width * m * n if split == 1 else 0)
    if split > 1:
        assert body.store_bytes == work.partial_bytes
        assert work.output_bytes == width * m * n


@pytest.mark.parametrize("capacity", [0, 1, 4095, 4096, 4097, 8192])
def test_aic_writeback_and_reduction_reload_count_each_spill_once(capacity):
    m, n, split = 16, 32, 4
    r = replace(NativeRates(), l2_bytes=capacity)
    body, work = traffic(m=m, n=n, k=13, split=split, rates=r)
    retained = min(capacity, 4 * split * m * n)
    spilled = 4 * split * m * n - retained
    assert body.spill_bytes == work.spill_bytes == spilled
    assert body.hbm_bytes == 2 * (m + n) * 13 + spilled
    # The final output is stored only by reduction; evicted partials cross
    # HBM twice (AIC write then reduction read), resident partials never do.
    total_hbm = body.hbm_bytes + work.spill_bytes + work.output_bytes
    assert total_hbm == 2 * (m + n) * 13 + 2 * spilled + 2 * m * n


@pytest.mark.parametrize("axis,values", [
    ("m", [1, 2, 4, 8, 16, 31, 32, 33, 96, 127, 128, 129, 384]),
    ("n", [1, 15, 16, 17, 48, 127, 128, 129, 1535, 1536, 1537, 3336]),
    ("k", [1, 15, 16, 17, 511, 512, 513, 2047, 2048, 4096, 7168]),
    ("split", [1, 2, 3, 4, 5, 8, 16]),
])
def test_aic_conserved_work_is_monotone_with_fixed_reuse_and_capacity(axis, values):
    previous = None
    for value in values:
        body, _ = traffic(**{axis: value}, rates=replace(NativeRates(), l2_bytes=0))
        current = (body.operand_bytes, body.panel_bytes, body.first_touch_bytes,
                   body.hbm_bytes, body.store_bytes, body.spill_bytes)
        if previous is not None:
            assert all(left <= right for left, right in zip(previous, current))
        previous = current


@pytest.mark.parametrize("axis,centers", [
    ("m", [8, 16, 32, 96, 128, 384]),
    ("n", [48, 128, 256, 512, 1536, 3072, 3336]),
    ("k", [512, 1024, 2048, 4096, 5120, 7168]),
])
def test_aic_byte_law_has_no_shape_box_jumps(axis, centers):
    for center in centers:
        bodies = [traffic(**{axis: value})[0] for value in range(center - 1, center + 2)]
        for field in ("operand_bytes", "panel_bytes", "first_touch_bytes", "store_bytes"):
            left, mid, right = [getattr(body, field) for body in bodies]
            assert right - mid == pytest.approx(mid - left)


def test_aic_output_work_is_independent_of_k_and_read_reuse():
    small, work = traffic(k=31)
    large, _ = traffic(k=8192)
    assert small.store_bytes == large.store_bytes == work.partial_bytes
    assert large.panel_bytes - small.panel_bytes == large.operand_bytes - small.operand_bytes
    extra_reads = aic_body_traffic(
        operand_bytes=small.operand_bytes * 7, first_touch_read_bytes=0,
        hbm_read_bytes=0, out_bytes=work.output_bytes, workspace=work)
    assert extra_reads.panel_bytes - extra_reads.operand_bytes == work.partial_bytes
    assert extra_reads.first_touch_bytes == work.partial_bytes


def test_reduction_envelope_meets_all_independent_capacity_bounds():
    r = NativeRates()
    work = workspace(rates=r)
    pipe, duration = work.bottleneck_ms(r)
    share = work.aiv_blocks / r.aiv
    demands = {
        "mte2_aiv": (work.partial_bytes, r.mte2_aiv * share),
        "vector": (4 * (work.add_elements + work.cast_elements), r.vector * share),
        "mte3": (work.output_bytes, r.mte3 * share),
        "hbm": (work.spill_bytes + work.output_bytes, r.hbm),
    }
    for amount, rate in demands.values():
        assert amount <= rate * duration / 1000 * (1 + 1e-12)
    amount, rate = demands[pipe]
    assert amount == pytest.approx(rate * duration / 1000)


@pytest.mark.parametrize("field", ["mte2_aiv", "vector", "mte3", "hbm"])
def test_reduction_envelope_scales_with_its_limiting_rate(field):
    r = replace(NativeRates(), **{field: 1e6})
    work = workspace(rates=r)
    pipe, before = work.bottleneck_ms(r)
    after_pipe, after = work.bottleneck_ms(replace(r, **{field: 2e6}))
    assert pipe == after_pipe == field
    assert after == pytest.approx(before / 2)


def test_resident_reduction_hbm_service_contains_only_final_output():
    r = replace(NativeRates(), hbm=1e6)
    resident = workspace(rates=r)
    assert resident.bottleneck_ms(r) == ("hbm", resident.output_bytes / r.hbm * 1000)
    spilled = workspace(rates=replace(r, l2_bytes=0))
    assert spilled.bottleneck_ms(r) == (
        "hbm", (spilled.partial_bytes + spilled.output_bytes) / r.hbm * 1000)


@pytest.mark.parametrize("axis,values", [
    ("m", range(17, 33)), ("n", range(129, 145)),
    ("k", range(4080, 4113)), ("split", range(1, 17)),
])
def test_reduction_data_work_is_monotone_for_fixed_participating_owners(axis, values):
    times = [traffic(**{axis: value})[1].bottleneck_ms(NativeRates())[1] for value in values]
    assert times == sorted(times)
    if axis == "k":
        assert times == [times[0]] * len(times)


@pytest.mark.parametrize("axis,values", [
    ("m", range(17, 33)), ("n", range(129, 145)), ("k", range(4080, 4113)),
])
def test_integrated_matrix_is_monotone_inside_a_fixed_tile_schedule(axis, values):
    # Changing split or tile ownership can lower elapsed time via parallelism;
    # global latency monotonicity across such dispatch changes is not asserted.
    times = [matrix(**{axis: value}).total_ms() for value in values]
    assert times == sorted(times)


@pytest.mark.parametrize("field", [
    "mte2_aic", "mte2_aiv", "mte1", "fixpipe", "hbm", "vector", "mte3", "scalar", "clock_hz",
])
def test_integrated_matrix_cannot_slow_down_with_faster_supply(field):
    r = NativeRates()
    faster = replace(r, **{field: 2 * getattr(r, field)})
    assert matrix(rates=faster).total_ms() <= matrix(rates=r).total_ms()


def test_mte1_operand_path_is_independent_of_partial_count_and_output_precision():
    # Make input transfer, rather than inherited scalar issue, limit MTE1.
    r = replace(NativeRates(), mte1=1e9)
    times = [dict(matrix(split=split, width=width, rates=r).body)["mte1"]
             for split in (2, 4, 8, 16) for width in (2, 4)]
    assert times == [times[0]] * len(times)


def catalog_supply(vector_rate=6.7584e9):
    return KernelSupply(
        invocations={}, arch="A5",
        rates={"mte2_aic": 4.224e9, "mte2_aiv": 4.224e9,
               "hbm": 4e9, "mte3": 13.5168e9, "mte1": 13.5168e9,
               "fixpipe": 6.7584e9, "vector": vector_rate,
               "pipe_s_issue": 105.6e6},
        units={"mte2_aic": 32, "mte2_aiv": 64, "vector": 64, "fixpipe": 32})


def catalog_elapsed(drafts, supply):
    catalog = catalog_from_drafts(
        drafts, implementation="serial-tail-test", inputs=(), outputs=(),
        events=(), allocation_specs=(), environment={}, occurrence={},
        semantic_sources=(), guard=())
    roles = {term.physical_resource_role
             for stage in catalog.stages for term in stage.work}
    engine = Engine([Resource(role, supply.rate(role) or 1.0) for role in sorted(roles)])
    submission = engine.submit_catalog(
        catalog, BindingFrame(occurrence=engine.new_occurrence()), name="matrix")
    engine.advance(until=(submission.done,))
    assert submission.done.done
    return submission.done.time


def int8_drafts(supply):
    return matrix_drafts(
        m=32, n=256, k=4096, tile=(16, 32, 16), supply=supply,
        quantized=True, read_bytes=(32 + 256) * 4096, write_bytes=2 * 32 * 256,
        dtype_bytes=1, weight_layout="FRACTAL_NZ", block_num=32)


@pytest.mark.parametrize("vector_rate", (1e6, 6.7584e9))
def test_catalog_executes_split_reduction_and_int8_epilogue_serially(vector_rate):
    supply = catalog_supply(vector_rate)
    rates = rates_from_supply(supply)
    m, n, k = 32, 256, 4096
    split = split_k_for(m=m, n=n, k=k, owners=32, kind="int8",
                        weight_layout="FRACTAL_NZ")
    times = matrix_times(
        m=m, n=n, k=k, read_bytes=(m + n) * k, write_bytes=2 * m * n,
        weight_bytes=n * k, act_bytes=m * k, out_bytes=2 * m * n,
        owners=32, units=32, kind="int8", rates=rates, split_k=split,
        weight_layout="FRACTAL_NZ", tile=(16, 32, 16))
    expected_pipe = "vector" if vector_rate == 1e6 else "mte2_aiv"
    assert split > 1 and [pipe for pipe, _ in times.tail] == [expected_pipe, "vector"]
    drafts = int8_drafts(supply)
    assert [(stage.label, stage.role) for stage in drafts[2:]] == [
        ("native:splitk_reduce", "reduce"), ("native:cube_epilogue:1", "epilogue")]
    expected_tail = sum(duration for _, duration in times.tail)
    assert catalog_elapsed(drafts[2:], supply) == pytest.approx(expected_tail, rel=1e-12)
    assert catalog_elapsed(drafts, supply) == pytest.approx(times.total_ms(), rel=1e-12)
    work = split_k_workspace(m=m, n=n, split=split, owners=32,
                             out_dtype_bytes=2, rates=rates)
    vector_bound = 1000 * 4 * (work.add_elements + work.cast_elements) / rates.vector
    assert catalog_elapsed(drafts[2:3], supply) >= vector_bound


@pytest.mark.parametrize("builder", ("matrix", "aiv"))
@pytest.mark.parametrize("tail", (
    (("vector", 0.002), ("vector", 0.003)),
    (("mte3", 0.002), ("vector", 0.003), ("fixpipe", 0.005)),
    (("mte3", 0.0), ("vector", 0.003), ("vector", 0.0), ("mte3", 0.002)),
    (),
))
def test_catalog_lowering_preserves_every_positive_tail_stage(monkeypatch, builder, tail):
    supply = catalog_supply()
    times = KernelTimes(0.001, (("mte2", 0.003), ("vector", 0.004)), tail, split_k=4)
    if builder == "matrix":
        monkeypatch.setattr(clock, "matrix_times", lambda **_: times)
        drafts = int8_drafts(supply)
    else:
        monkeypatch.setattr(clock, "aiv_times", lambda **_: times)
        drafts = _a5_aiv_drafts("row_norm", {"rows": 8, "width": 32}, supply)
    positive = [(pipe, duration) for pipe, duration in tail if duration > 0]
    assert len(drafts[2:]) == len(positive)
    assert len({stage.label for stage in drafts[2:]}) == len(positive)
    for stage, (pipe, duration) in zip(drafts[2:], positive):
        assert [term[0] for term in stage.work] == [pipe]
        assert catalog_elapsed([stage], supply) == pytest.approx(duration, rel=1e-12)
    assert catalog_elapsed(drafts, supply) == pytest.approx(times.total_ms(), rel=1e-12)


@pytest.mark.parametrize("capacity", (1, 32768, 131071, 131072, 131073, 2000000))
@pytest.mark.parametrize("hbm", (0.25, 1e6, 1e9, 4e12))
@pytest.mark.parametrize("kind,k,width", (
    ("bf16", 2048, 2), ("int8", 4096, 1), ("mxfp8", 4096, 1), ("fp32", 2048, 4),
))
def test_isolated_split_output_honors_declared_l2_and_hbm(capacity, hbm, kind, k, width):
    m, n = 32, 256
    rates = replace(NativeRates(), l2_bytes=capacity, hbm=hbm)
    split = split_k_for(m=m, n=n, k=k, owners=32, kind=kind,
                        weight_layout="FRACTAL_NZ")
    assert split > 1
    work = split_k_workspace(m=m, n=n, split=split, owners=32,
                             out_dtype_bytes=2, rates=rates)
    times = matrix_times(
        m=m, n=n, k=k, read_bytes=width * (m + n) * k,
        write_bytes=2 * m * n, weight_bytes=width * n * k,
        act_bytes=width * m * k, out_bytes=2 * m * n,
        owners=32, units=32, kind=kind, rates=rates, split_k=split,
        weight_layout="FRACTAL_NZ", execution_scope="isolated")
    # Even free operand reads cannot remove the partial write, reload or output.
    spill = max(0, 4 * split * m * n - capacity)
    output = 2 * m * n
    assert work.spill_bytes == spill and work.output_bytes == output
    assert dict(times.body)["hbm"] >= 1000 * spill / hbm
    assert times.total_ms() >= 1000 * (2 * spill + output) / hbm


def test_isolated_resident_partials_do_not_make_oversized_operands_l2_hot():
    m, n, k, split = 32, 256, 2048, 4
    rates = replace(NativeRates(), l2_bytes=4 * split * m * n, hbm=1e9)
    times = matrix_times(
        m=m, n=n, k=k, read_bytes=2 * (m + n) * k,
        write_bytes=2 * m * n, weight_bytes=2 * n * k,
        act_bytes=2 * m * k, out_bytes=2 * m * n,
        owners=32, units=32, kind="bf16", rates=rates, split_k=split,
        execution_scope="isolated")
    assert dict(times.body)["hbm"] >= 1000 * 2 * (m + n) * k / rates.hbm


def duration_declaration(component, milliseconds):
    fields = dict(prologue_ms=0.001, body=(("mte2", 0.003), ("vector", 0.004)),
                  tail=(("vector", 0.002), ("fixpipe", 0.001), ("vector", 0.003)),
                  split_k=4)
    # A later valid entry must not hide an invalid service in max() or dict().
    fields[component] = (milliseconds if component == "prologue_ms" else
                         (("vector", 0.002), ("vector", milliseconds), ("vector", 0.003)))
    return fields


def injected_drafts(monkeypatch, builder, times, boundary="producer"):
    if boundary == "shared":
        monkeypatch.setattr(clock, "catalog_body_share", lambda *_: times)
    else:
        producer = "matrix_times" if builder == "matrix" else "aiv_times"
        monkeypatch.setattr(clock, producer, lambda **_: times)
    supply = catalog_supply()
    return (int8_drafts(supply) if builder == "matrix" else
            _a5_aiv_drafts("row_norm", {"rows": 8, "width": 32}, supply))


@pytest.mark.parametrize("component", ("prologue_ms", "body", "tail"))
@pytest.mark.parametrize("milliseconds", (-0.001, float("nan"), float("inf"), -float("inf")))
def test_kernel_times_rejects_invalid_elapsed_services(component, milliseconds):
    with pytest.raises(ModelingError, match="^native_kernel_invalid_duration$"):
        KernelTimes(**duration_declaration(component, milliseconds))


@pytest.mark.parametrize("representation", ("unchecked", "duck"))
@pytest.mark.parametrize("component", ("prologue_ms", "body", "tail"))
@pytest.mark.parametrize("milliseconds", (-0.001, float("nan"), float("inf"), -float("inf")))
def test_kernel_total_revalidates_elapsed_services(representation, component, milliseconds):
    fields = duration_declaration(component, milliseconds)
    if representation == "unchecked":
        times = object.__new__(KernelTimes)
        for name, value in fields.items():
            object.__setattr__(times, name, value)
    else:
        times = SimpleNamespace(**fields, __post_init__=lambda: None)
    with pytest.raises(ModelingError, match="^native_kernel_invalid_duration$"):
        KernelTimes.total_ms(times)


@pytest.mark.parametrize("builder", ("matrix", "aiv"))
@pytest.mark.parametrize("boundary", ("producer", "shared"))
@pytest.mark.parametrize("component", ("prologue_ms", "body", "tail"))
@pytest.mark.parametrize("milliseconds", (-0.001, float("nan"), float("inf"), -float("inf")))
def test_catalog_rejects_invalid_duck_typed_elapsed_services(
        monkeypatch, builder, boundary, component, milliseconds):
    times = SimpleNamespace(**duration_declaration(component, milliseconds),
                            __post_init__=lambda: None)
    with pytest.raises(ModelingError, match="^native_kernel_invalid_duration$"):
        injected_drafts(monkeypatch, builder, times, boundary)


@pytest.mark.parametrize("builder", ("matrix", "aiv"))
@pytest.mark.parametrize("duck_typed", (False, True))
@pytest.mark.parametrize("zero", (0.0, -0.0))
@pytest.mark.parametrize("positive", (False, True))
def test_catalog_and_total_preserve_signed_zero_services(
        monkeypatch, builder, duck_typed, zero, positive):
    fields = dict(prologue_ms=zero,
                  body=(("mte2", zero), ("vector", 0.004 if positive else zero)),
                  tail=(("vector", 0.002 if positive else zero), ("fixpipe", zero),
                        ("vector", 0.003 if positive else zero)))
    times = SimpleNamespace(**fields, split_k=4) if duck_typed else KernelTimes(**fields, split_k=4)
    expected = 0.009 if positive else 0.0
    assert KernelTimes.total_ms(times) == pytest.approx(expected, rel=1e-14, abs=0)
    drafts = injected_drafts(monkeypatch, builder, times)
    assert len(drafts) == (4 if positive else 2)
    assert catalog_elapsed(drafts, catalog_supply()) == pytest.approx(expected, rel=1e-14, abs=0)


INVALID_DURATION_VALUES = [
    *[pytest.param(Decimal(value), id="decimal-" + value)
      for value in ("0", "-0", "0.002", "-1e-1000", "NaN", "sNaN", "Infinity")],
    pytest.param(True, id="bool-true"), pytest.param(False, id="bool-false"),
    pytest.param(Fraction(-1, 10**400), id="fraction-negative-underflow"),
    pytest.param(Fraction(10**400), id="fraction-overflow"),
    pytest.param(10**400, id="integer-overflow"),
    pytest.param("0.002", id="string"), pytest.param(None, id="none"),
    pytest.param(complex(0.002), id="complex"), pytest.param(object(), id="object"),
    pytest.param(np.float64("nan"), id="numpy-nan"),
    pytest.param(np.float32("inf"), id="numpy-inf"),
    pytest.param(np.bool_(False), id="numpy-bool"),
]
VALID_DURATION_VALUES = [
    pytest.param(0.0, id="float-zero"), pytest.param(-0.0, id="float-signed-zero"),
    pytest.param(0.002, id="float-positive"),
    pytest.param(0, id="integer-zero"), pytest.param(1, id="integer-positive"),
    pytest.param(Fraction(0), id="fraction-zero"),
    pytest.param(Fraction(1, 500), id="fraction-positive"),
    pytest.param(Fraction(1, 10**400), id="fraction-underflow"),
    *[pytest.param(kind(0.002), id=kind.__name__)
      for kind in (np.float16, np.float32, np.float64, np.longdouble)],
]


def unchecked_duration(fields, representation):
    if representation == "duck":
        return SimpleNamespace(**fields, __post_init__=lambda: None)
    times = object.__new__(KernelTimes)
    for name, value in fields.items():
        object.__setattr__(times, name, value)
    return times


@pytest.mark.parametrize("milliseconds", INVALID_DURATION_VALUES)
@pytest.mark.parametrize("component", ("prologue_ms", "body", "tail"))
def test_duration_domain_rejects_unsupported_values_everywhere(
        monkeypatch, component, milliseconds):
    fields = duration_declaration(component, milliseconds)
    if component == "body":
        fields[component] = tuple(("mte2", value) for _, value in fields[component])
    error = "^native_kernel_invalid_duration$"
    with pytest.raises(ModelingError, match=error):
        KernelTimes(**fields)
    for representation in ("unchecked", "duck"):
        times = unchecked_duration(fields, representation)
        with pytest.raises(ModelingError, match=error):
            KernelTimes.__post_init__(times)
        with pytest.raises(ModelingError, match=error):
            KernelTimes.total_ms(times)
        for shared in (4.0, 8.0):
            rates = replace(NativeRates(), mte2_aic=4.0, mte2_aiv=shared)
            with pytest.raises(ModelingError, match=error):
                clock.catalog_body_share(times, rates)
            with pytest.raises(ModelingError, match=error):
                _validated_catalog_times(times, rates)
            for builder in ("matrix", "aiv"):
                for boundary in ("producer", "shared"):
                    with monkeypatch.context() as patch:
                        patch.setattr(clock, "rates_from_supply", lambda _: rates)
                        with pytest.raises(ModelingError, match=error):
                            injected_drafts(patch, builder, times, boundary)


@pytest.mark.parametrize("milliseconds", VALID_DURATION_VALUES)
@pytest.mark.parametrize("component", ("prologue_ms", "body", "tail"))
def test_real_duration_domain_preserves_float_arithmetic_and_execution(
        monkeypatch, component, milliseconds):
    fields = dict(prologue_ms=0.001, body=(("mte2", 0.004), ("vector", 0.001)),
                  tail=(("vector", 0.003), ("vector", 0.002)), split_k=4)
    fields[component] = (milliseconds if component == "prologue_ms" else
                         ((("mte2", milliseconds),) if component == "body" else
                          (("vector", milliseconds), ("vector", 0.002))))
    prologue = float(fields["prologue_ms"])
    body = tuple((pipe, float(value)) for pipe, value in fields["body"])
    tail = tuple((pipe, float(value)) for pipe, value in fields["tail"])
    expected = prologue + max(value for _, value in body) + sum(value for _, value in tail)
    instances = [KernelTimes(**fields)]
    instances += [unchecked_duration(fields, kind) for kind in ("unchecked", "duck")]
    for times in instances:
        assert struct.pack("!d", KernelTimes.total_ms(times)) == struct.pack("!d", expected)
        for shared in (4.0, 8.0):
            rates = replace(NativeRates(), mte2_aic=4.0, mte2_aiv=shared)
            shared_body = tuple((pipe, value * (shared / 4.0) if pipe == "mte2" else value)
                                for pipe, value in body)
            shared_expected = (prologue + max(value for _, value in shared_body)
                               + sum(value for _, value in tail))
            for converted in (clock.catalog_body_share(times, rates),
                              _validated_catalog_times(times, rates)):
                assert struct.pack("!d", KernelTimes.total_ms(converted)) == struct.pack("!d", shared_expected)
            for builder in ("matrix", "aiv"):
                for boundary in ("producer", "shared"):
                    with monkeypatch.context() as patch:
                        patch.setattr(clock, "rates_from_supply", lambda _: rates)
                        drafts = injected_drafts(patch, builder, times, boundary)
                        float_times = KernelTimes(prologue, body, tail, split_k=4)
                        reference = injected_drafts(patch, builder, float_times, boundary)
                    assert drafts == reference
                    assert all(type(stage.latency_ms) is float for stage in drafts)
                    assert all(type(amount) is float for stage in drafts
                               for _, amount, _ in stage.work)
                    elapsed = shared_expected if boundary == "producer" else expected
                    assert catalog_elapsed(drafts, catalog_supply()) == pytest.approx(
                        elapsed, rel=1e-13, abs=0)
