"""Tier-0 tests for unified AscendC/DSL/Triton absorption and recurrence.

[Role]
Verify authoring backends, Cube-only vs total TFLOPS, SIMD/SIMT and SSBUF
gates, 2201->3510 mem_copy edges, the tau >= L/d recurrence law, monotonic
active-unit scaling, K3/DS41 groups, the 33 Triton rules, and MFU five gates.
[Invariants]
Deterministic pure-Python checks; no device, file, or network access.
[Timeouts & Locks]
In-memory only; no sleeps, locks, or I/O.
[Side-Effect Policy]
Read-only; no files are created.
"""

from __future__ import annotations

import pytest

from mindie_sim.opflow import (
    AuthoringBackend,
    BufferPool,
    CATALOG,
    DS41_OPERATORS,
    ExecutionArchetype,
    ExecutionMode,
    FixpipePath,
    K3_OPERATORS,
    LoopCarriedRecurrence,
    OpFlowTemplate,
    PipeWork,
    TRITON_RULES,
    TritonFeatures,
    TritonLoweringError,
    analyze_triton_lowering,
    build_flow,
    build_network_flow,
    classify_cache_regime,
    evaluate_flow,
    evaluate_mfu_gates,
    get_profile,
    legal_mem_copy_edges,
    list_network_operator_group,
    network_membership,
    rule_by_number,
    supports_mem_copy,
    validate_mem_copy,
)
from mindie_sim.opflow.pipeline_evaluator import _wave_services

P950 = get_profile("Ascend950PR", sku="ascend950pr_9589")
P910 = get_profile("Ascend910B")


def _synthetic(
    name: str,
    *,
    pip: str = "VEC",
    work: float = 1e8,
    units: float = 640,
    tile: float = 1,
    recurrence: LoopCarriedRecurrence | None = None,
    backend: AuthoringBackend = AuthoringBackend.ASCENDC,
    features: dict | None = None,
) -> OpFlowTemplate:
    archetype = ExecutionArchetype.PURE_VECTOR
    work_kind = "ops"
    if pip == "CUBE":
        archetype = ExecutionArchetype.PURE_CUBE
        work_kind = "macs"
    return OpFlowTemplate(
        name=name,
        archetype=archetype,
        total_units=units,
        base_tile_units=tile,
        live_bytes_per_unit=0.0,
        pipe_work=(PipeWork(pip, work, work_kind),),
        gm_read_bytes=1024,
        gm_write_bytes=1024,
        unique_gm_bytes=2048,
        tiling_pool=BufferPool.UB_LANE0 if archetype is ExecutionArchetype.PURE_VECTOR else BufferPool.L1,
        authoring_backend=backend,
        recurrence=recurrence,
        triton_features=features or {},
    )


def test_three_authoring_backends_present() -> None:
    assert build_flow("rms_norm").authoring_backend is AuthoringBackend.ASCENDC
    assert build_flow("block_attn_res_update_rms_norm").authoring_backend is AuthoringBackend.CANNBOT_DSL
    triton = _synthetic("triton", backend=AuthoringBackend.TRITON_ASCEND)
    result = evaluate_flow(triton, P950)
    assert result.authoring_backend == "triton_ascend"
    assert "triton" in result.extra
    assert evaluate_flow(build_flow("rms_norm"), P950).authoring_backend == "ascendc"


def test_cube_only_vs_advertised_peaks() -> None:
    # Derived peaks are frequency-consistent exact formulas; the whitepaper
    # table is a separately rounded reference column.
    assert P950.whitepaper_rounded_peaks("bf16").cube == pytest.approx(432.0)
    assert P950.cube_peak_tflops("bf16") == pytest.approx(432.5376)
    assert P950.cube_peak_tflops("bf16") == pytest.approx(2 * P950.chip_cube_macs_s / 1e12)
    assert P950.whitepaper_rounded_peaks("bf16").total == pytest.approx(486.0)
    assert P950.peaks("bf16").cube_gap_pct == pytest.approx(12.5, rel=1e-3)
    dt = get_profile("Ascend950DT", cores=36)
    assert dt.whitepaper_rounded_peaks("bf16").cube == pytest.approx(486.0)
    assert dt.whitepaper_rounded_peaks("bf16").total == pytest.approx(547.0)
    assert dt.vector_peak_tflops("bf16") == pytest.approx(60.8256)
    assert get_profile("Ascend950PR", freq_mhz=825).cube_peak_tflops("bf16") == pytest.approx(
        P950.cube_peak_tflops("bf16") / 2)
    assert P950.cube_peak_tflops("bf16") < P950.advertised_peak_tflops("bf16")
    with pytest.raises(Exception):
        P950.cube_peak_tflops("not-a-dtype")


def test_ub_simt_ssbuf_and_memcopy_arch_gates() -> None:
    assert P950.ub_usable_bytes("simd") == 253952
    assert P950.ub_usable_bytes("simt") == 221184
    assert P950.ssbuf_bytes(True) == 3072 and P950.ssbuf_bytes(False) == 1024
    assert P910.ub_usable_bytes("simd") == 196608
    assert P910.ssbuf_bytes(True) == 0
    with pytest.raises(Exception):
        P910.ub_usable_bytes("simt")
    assert supports_mem_copy("UB", "L1", P950)
    assert supports_mem_copy("L0C", "UB", P950)
    assert not supports_mem_copy("L1", "GM", P950)
    assert not supports_mem_copy("GM", "L0A", P950)
    assert supports_mem_copy("L1", "GM", P910)
    assert supports_mem_copy("GM", "L0B", P910)
    assert not supports_mem_copy("UB", "L1", P910)
    assert validate_mem_copy("L0C", "UB", P950) == "L0C->UB"
    assert "L1->GM" in legal_mem_copy_edges(P910)
    assert "L1->GM" not in legal_mem_copy_edges(P950)
    with pytest.raises(Exception):
        evaluate_flow(build_flow("add"), P910)


def test_recurrence_law_tau_and_total() -> None:
    rec = LoopCarriedRecurrence(latency_us=50.0, distance_items=2, state_name="s")
    flow = _synthetic("rec", work=1e6, units=64, tile=1, recurrence=rec)
    result = evaluate_flow(flow, P950, active_units=8)
    assert result.recurrence_interval_us == pytest.approx(25.0)
    assert result.tau_steady_us >= result.recurrence_interval_us
    assert result.tau_steady_us >= result.max_pipe_load_us
    assert result.tau_steady_us >= max(result.pipe_us.values())
    no_rec = _synthetic("norec", work=1e6, units=64, tile=1)
    base = evaluate_flow(no_rec, P950, active_units=8)
    assert result.t_est_us > base.t_est_us
    # Distance-two feedback permits alternating short gaps and 50us groups.
    # A cycle mean is not an exact finite affine completion law.
    groups, tail = divmod(result.extra["advance_steps"] - 1, 2)
    expected_unit = result.t_first_item_us + groups * 50 + tail * result.max_pipe_load_us
    assert result.t_unit_us == pytest.approx(expected_unit)
    assert result.tau_steady_us >= result.recurrence_interval_us
    cycles = LoopCarriedRecurrence(cycles=1650 * 10, distance_items=1)
    cf = _synthetic("cyc", work=1.0, units=8, tile=1, recurrence=cycles)
    cr = evaluate_flow(cf, P950)
    assert cr.recurrence_interval_us == pytest.approx(10.0)
    assert cr.tau_steady_us >= 10.0


def test_independent_service_bound_active_units_and_cap() -> None:
    flow = build_flow("matmul_bf16")
    times = [evaluate_flow(flow, P950, active_units=a).t_est_us for a in (1, 2, 4, 8, 16, 32)]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(times, times[1:]))
    assert times[-1] < times[0]
    assert evaluate_flow(flow, P950, active_units=64).t_est_us == pytest.approx(times[-1])
    assert evaluate_flow(flow, P950, active_units=999).t_est_us == pytest.approx(times[-1])


def test_divisor_wave_counts_at_fixed_shared_bandwidth() -> None:
    kwargs = dict(
        tile_count=64, compute_tile_us=0.01, read_per_tile=65536.0,
        write_per_tile=1024.0, bandwidth=6.4e12, per_lane_cap=100e9, alpha=0.3,
    )
    previous = float("inf")
    for active in (1, 2, 4, 8, 16, 32, 64):
        services, _, _ = _wave_services(active_units=active, **kwargs)
        current = sum(services)
        assert current <= previous + 1e-9
        previous = current


def test_network_groups_cover_all_members() -> None:
    assert len(K3_OPERATORS) == 10 and len(DS41_OPERATORS) == 10
    assert list_network_operator_group("k3") == K3_OPERATORS
    assert list_network_operator_group("DS41") == DS41_OPERATORS
    with pytest.raises(KeyError):
        list_network_operator_group("nope")
    for name in K3_OPERATORS + DS41_OPERATORS:
        assert name in CATALOG
        assert "k3" in CATALOG[name].networks or "ds41" in CATALOG[name].networks
        flow = build_network_flow(name, allow_structural_prior=True)
        assert flow.name == name
        assert evaluate_flow(flow, P950, allow_structural_prior=True).t_est_us > 0
    assert network_membership("add_rms_norm") == ("common",)
    assert "mega_mla" in list_network_operator_group("ds41")


def test_triton_33_rules_and_hazards() -> None:
    assert len(TRITON_RULES) == 33
    assert [rule.number for rule in TRITON_RULES] == list(range(1, 34))
    assert len({rule.slug for rule in TRITON_RULES}) == 33
    assert rule_by_number(1).slug == "constexpr_parameters"
    assert rule_by_number(33).is_terminal and rule_by_number(31).is_terminal
    tags = {tag for rule in TRITON_RULES for tag in rule.hazard_tags}
    assert {"random_gather_to_dot", "cv_serialism", "mla_block_qo"} <= tags
    with pytest.raises(KeyError):
        rule_by_number(34)


def test_triton_serialism_gather_scalar_and_mfu() -> None:
    serial = analyze_triton_lowering(
        P950, aic_us=5.0, aiv_us=2.0, gm_us=0.1,
        features=TritonFeatures(has_cv_pingpong=False), wall_us_hint=100.0,
    )
    assert serial.cv_serialism and serial.serialism_ratio == pytest.approx(1.0)
    assert serial.t_overlapped_us >= 7.0
    assert serial.potential_overlap_gain == pytest.approx(1.4)
    ping = analyze_triton_lowering(
        P950, aic_us=5.0, aiv_us=2.0, gm_us=0.1,
        features=TritonFeatures(has_cv_pingpong=True), wall_us_hint=100.0,
    )
    assert not ping.cv_serialism
    assert ping.serialism_ratio == pytest.approx(7.0 / 5.1)
    assert ping.potential_overlap_gain == pytest.approx(serial.potential_overlap_gain)
    none = analyze_triton_lowering(P950, aic_us=0.0, aiv_us=1.0, wall_us_hint=10.0)
    ub = analyze_triton_lowering(
        P950, aic_us=0.0, aiv_us=1.0, wall_us_hint=10.0,
        features=TritonFeatures(random_gather_to_dot_mode="gather_out_to_ub", gather_bytes=1_000_000),
    )
    dev = analyze_triton_lowering(
        P950, aic_us=0.0, aiv_us=1.0, wall_us_hint=10.0,
        features=TritonFeatures(random_gather_to_dot_mode="device_gather_workspace", gather_bytes=1_000_000),
    )
    assert none.gather_us == 0.0 and ub.gather_us > 0.0
    assert dev.gather_us > ub.gather_us and dev.workspace_round_trip_bytes == 2_000_000
    hbm_dev = analyze_triton_lowering(
        P950, aic_us=0.0, aiv_us=1.0, wall_us_hint=10.0,
        features=TritonFeatures(random_gather_to_dot_mode="device_gather_workspace", gather_bytes=1_000_000),
        per_core_working_set_bytes=P950.memory.l2 * 2,
    )
    assert hbm_dev.gather_us > dev.gather_us
    degraded = analyze_triton_lowering(
        P950, aic_us=1.0, aiv_us=1.0, wall_us_hint=10.0,
        features=TritonFeatures(has_i64_mask_or_noncontig_reduce=True, mla_block_qo=2,
                                is_mla_kernel=True, chained_dot_fp32_acc_spill=True,
                                chained_buffer_bytes=1_000_000),
    )
    assert degraded.vector_degradation_factor == pytest.approx(3.3)
    assert degraded.scalar_hazard and degraded.nonaffine_store_hazard
    assert degraded.block_qo_penalty_us > 0 and degraded.fp32_acc_spill
    bad = evaluate_mfu_gates(
        P950, wall_us=1.0, features=TritonFeatures(input_bytes=10_000_000, flops=1e9),
        working_set_bytes=1024, total_bytes=10_000_000,
    )
    assert not bad.valid and not bad.gates["effective_bw_within_hbm"]
    assert classify_cache_regime(1024, P950.memory.l2) == "cache_resident_small_shape"
    assert classify_cache_regime(P950.memory.l2 * 2, P950.memory.l2) == "hbm_resident_scaled_shape"
    with pytest.raises(TritonLoweringError):
        TritonFeatures(random_gather_to_dot_mode="magic")


def test_triton_profile_integration_penalizes_cv() -> None:
    flow = OpFlowTemplate(
        name="triton-cv", archetype=ExecutionArchetype.CV_COUPLED_1C2V,
        total_units=64, base_tile_units=1, live_bytes_per_unit=0.0,
        pipe_work=(PipeWork("CUBE", 5e9, "macs"), PipeWork("VEC", 2e9, "ops")),
        gm_read_bytes=1_000_000, gm_write_bytes=1_000_000, unique_gm_bytes=2_000_000,
        fixpipe_path=FixpipePath.L0C_TO_UB, fixpipe_sub_block=0,
        authoring_backend=AuthoringBackend.TRITON_ASCEND,
        triton_features={"has_cv_pingpong": False, "random_gather_to_dot_mode": "device_gather_workspace",
                         "gather_bytes": 1_000_000, "mla_block_qo": 2, "is_mla_kernel": True},
    )
    result = evaluate_flow(flow, P950)
    triton = result.extra["triton"]
    assert triton["cv_serial"] and triton["serialism"] == pytest.approx(1.0)
    assert triton["potential_gain"] > 1.0
    assert triton["workspace_bytes"] == 2_000_000
    assert triton["nonaffine"] and triton["penalty_us"] > 0
    assert "mfu" in result.extra
