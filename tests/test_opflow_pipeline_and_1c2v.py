"""Tier-0 tests for opflow hardware, tiling, transfer, and catalog behavior.

[Role]
Verify DAV_3510 vs V220 Fixpipe gates, UB spill, contention/L2 knees, exact
three-quantity transfer accounting, and the parameterized catalog timing laws.
[Invariants]
Every test is deterministic, sub-second, and uses no device or network.
[Timeouts & Locks]
In-memory only; no sleeps, locks, or I/O.
[Side-Effect Policy]
Read-only; no files are created.
"""

from __future__ import annotations

import pytest

from mindie_sim.opflow import (
    BufferPool,
    FixpipeError,
    FixpipePath,
    FlowSchemaError,
    GmStrides,
    ReadPolicy,
    TilingError,
    TransferKind,
    TransferRequest,
    account_transfer,
    catalog_info,
    derive_tiling,
    effective_gm_bandwidth,
    evaluate_flow,
    evaluate_operator,
    get_profile,
    launch_floor_ordering,
    list_operators,
    select_fixpipe,
    validate_mem_copy,
)
from mindie_sim.opflow.paradigm_catalog import build_flow
from mindie_sim.opflow.pipeline_evaluator import LAUNCH_FLOORS_US

P950 = get_profile("Ascend950PR", sku="ascend950pr_9589")
P950DT = get_profile("Ascend950DT", sku="Ascend950DT_36c")
P910 = get_profile("Ascend910B")


def test_fixpipe_l0c_to_ub_3510_only() -> None:
    selection = select_fixpipe(P950, FixpipePath.L0C_TO_UB, 1)
    assert selection.allowed and selection.lane == 1 and selection.is_to_ub
    assert select_fixpipe(P950DT, FixpipePath.L0C_TO_UB, 0).lane == 0
    with pytest.raises(FixpipeError):
        select_fixpipe(P910, FixpipePath.L0C_TO_UB, 0)
    with pytest.raises(FixpipeError):
        evaluate_flow(build_flow("flash_attn_score"), P910)


def test_ub_spill_and_indivisible_fail_closed() -> None:
    plan = derive_tiling(1000.0, 1000.0, 300.0, 253952, slots=2,
                         spill_policy="algorithm_pass_workspace")
    assert plan.passes >= 2 and plan.tile_count >= 2
    assert plan.spill_read_bytes > 0 and plan.spill_write_bytes > 0
    assert plan.live_bytes_per_tile <= 253952
    retiled = derive_tiling(1000.0, 1000.0, 300.0, 253952, slots=2)
    assert retiled.passes == 1 and retiled.spill_read_bytes == 0 and retiled.retiled
    with pytest.raises(TilingError):
        derive_tiling(10.0, 10.0, 300000.0, 253952, slots=2)


def test_contention_alpha_and_l2_knee() -> None:
    base = build_flow("add_rms_norm")
    no_alpha = get_profile("Ascend950PR", sku="ascend950pr_9589", rw_contention_alpha=0.0)
    full_alpha = get_profile("Ascend950PR", sku="ascend950pr_9589", rw_contention_alpha=0.6)
    t0 = evaluate_flow(base, no_alpha).t_est_us
    t1 = evaluate_flow(base, full_alpha).t_est_us
    assert t1 >= t0
    l2 = P950.memory.l2
    assert effective_gm_bandwidth(P950, l2 // 2) == pytest.approx(P950.rates.l2_bytes_s)
    assert effective_gm_bandwidth(P950, 3 * l2) == pytest.approx(P950.rates.hbm_bytes_s)


def test_three_quantity_transfer_accounting() -> None:
    dn2nz = account_transfer(
        TransferRequest(kind=TransferKind.DN2NZ, rows=17, cols=33, element_bytes=2),
        P950,
    )
    assert dn2nz.logical_bytes == 1122
    assert dn2nz.actual_write_bytes == 1122
    assert dn2nz.destination_capacity_bytes >= 32 * 48 * 2
    full = account_transfer(
        TransferRequest(
            kind=TransferKind.LOAD3D, rows=1, cols=1, element_bytes=2,
            source_hw=(8, 8), read_policy=ReadPolicy.FULL_SOURCE_ONCE,
        ),
        P950,
    )
    per_ref = account_transfer(
        TransferRequest(
            kind=TransferKind.LOAD3D, rows=1, cols=1, element_bytes=2,
            source_hw=(8, 8), kernel=(3, 3), read_policy=ReadPolicy.PER_REFERENCE,
        ),
        P950,
    )
    assert full.actual_read_bytes == 8 * 8 * 2
    assert per_ref.actual_read_bytes > full.actual_read_bytes
    strides = GmStrides(rows=4, row_bytes=64, stride_bytes=128)
    strided = account_transfer(
        TransferRequest(kind=TransferKind.GM_STRIDED, rows=4, cols=32, element_bytes=2, gm_strides=strides),
        P950,
        pool=BufferPool.GM,
    )
    assert strided.logical_bytes == 4 * 32 * 2
    assert strided.actual_read_bytes == 4 * 64
    assert strided.physical_capacity_bytes == 3 * 128 + 64
    with pytest.raises(Exception):
        validate_mem_copy("GM", "L0A", P950)
    assert validate_mem_copy("GM", "L0A", P910) == "GM->L0A"


def test_catalog_count_and_all_templates_evaluate() -> None:
    names = list_operators()
    assert len(names) >= 24
    assert len({entry["name"] for entry in catalog_info()}) == len(names)
    for name in names:
        flow = build_flow(name)
        result = evaluate_flow(flow, P950, allow_structural_prior=True)
        assert result.t_est_us >= result.launch_floor_us
        assert result.tiling.tile_count >= 1 and result.waves >= 1
        assert result.t_est_us < 1e7


def test_catalog_shape_and_core_monotonicity() -> None:
    small = evaluate_operator("add_rms_norm", P950, tokens=16)
    large = evaluate_operator("add_rms_norm", P950, tokens=64)
    assert large.t_est_us >= small.t_est_us
    f28 = evaluate_flow(build_flow("matmul_bf16"), get_profile("Ascend950DT", cores=28))
    f36 = evaluate_flow(build_flow("matmul_bf16"), get_profile("Ascend950DT", cores=36))
    assert f36.t_est_us <= f28.t_est_us
    assert f36.t_est_us < f28.t_est_us


def test_block_attn_measured_regime_and_launch_order() -> None:
    block = evaluate_operator("block_attn_res_update_rms_norm", P950)
    assert build_flow("block_attn_res_update_rms_norm").params["hidden"] == 7168
    assert 2.0 <= block.t_est_us <= 20.0
    assert "tokens=0" in build_flow("block_attn_res_update_rms_norm").notes
    zero = evaluate_flow(build_flow("block_attn_res_update_rms_norm", tokens=0), P950)
    assert zero.t_est_us == 0.0 and not zero.launched
    order = launch_floor_ordering(P950, build_flow("rms_norm"))
    assert order["eager"] > order["aclgraph_full"] > order["aclgraph_superkernel"]
    for mode, floor in LAUNCH_FLOORS_US.items():
        assert evaluate_flow(build_flow("rms_norm"), P950, mode).t_est_us >= floor - 1e-9
    with pytest.raises(FlowSchemaError):
        evaluate_flow(build_flow("rms_norm"), P950, active_units=0)
