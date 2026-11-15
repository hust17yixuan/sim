"""Regression contracts for the F01-F26 / v3 counterexample closure.

[Role]
Pin the corrected hardware, transfer, pipeline, catalog, Triton, ACLGraph,
distiller provenance, normalization, and compact-MCP contracts so the Round-1
findings cannot regress silently.
[Invariants]
Every check is deterministic and in-memory. Live-corpus checks skip only if the
read-only pinned repositories are absent; this workspace has them present.
[Timeouts & Locks]
No sleeps, locks, subprocesses, or writes outside pytest's tmp_path.
[Side-Effect Policy]
Read-only except an optional tmp_path JSONL write.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mindie_knowledge")

from mindie_knowledge import cannbot_aclgraph_distiller as dist
from mindie_knowledge.knowledge_unit import KnowledgeUnit, KnowledgeUnitError
pytest.importorskip("mindie_profiler")
from mindie_profiler.aclgraph_mcp import call_tool as acl_call
from mindie_profiler.aclgraph_multistream import (
    ACLGraphModelError, ACLGraphWrapperModel, ParallelRegion, StreamInterval,
    UpdatableGraphModel, classify_capture_error, diagnose_region,
    effective_launch_floor_us,
)
from mindie_sim.opflow import (
    BufferPool, BufferSpec, ExecutionArchetype, FixpipePath, FlowSchemaError,
    GmStrides, LoopCarriedRecurrence, OpFlowTemplate, PipeWork, ProfileError,
    TritonFeatures, TritonLoweringError, UncalibratedOperatorError,
    account_transfer, analyze_triton_lowering, build_flow, derive_tiling,
    evaluate_flow, evaluate_mfu_gates, evaluate_operator, get_profile,
    plan_buffer_pool, validate_mem_copy,
)
from mindie_sim.opflow.opflow_mcp import call_tool as op_call
from mindie_sim.opflow.token_budget import envelope_size
from mindie_sim.opflow.transfer_accounting import (
    ReadPolicy, TransferError, TransferRequest,
)

P = get_profile("Ascend950PR")
P2 = get_profile("Ascend910B")
REF_ROOT = Path("/Users/maoxx241/code/refs")
VLLM_ROOT = Path("/Users/maoxx241/code/vllm-ascend")


def test_canonical_sku_geometry_and_l0c() -> None:
    assert P2.memory.l0c == 128 * 1024
    assert P2.memory.ub_per_lane == 192 * 1024
    assert get_profile("Ascend950PR_957b").memory.l0c == 256 * 1024
    assert get_profile("Ascend950PR_9589").l2_bw_gbps == 6400.0
    assert get_profile("Ascend950PR_957b").l2_bw_gbps == 5600.0
    assert get_profile("A2").sku == "Ascend910B4"
    assert get_profile("A3").sku == "Ascend910_9392"
    for sku in ("Ascend910B1", "Ascend910B2", "Ascend910B2C", "Ascend910B3",
                "Ascend910B4", "Ascend910_93", "Ascend910_9362", "Ascend910_9372",
                "Ascend910_9381", "Ascend910_9382", "Ascend910_9391", "Ascend910_9392"):
        assert get_profile(sku).memory.l0c == 128 * 1024
    for sku in ("Ascend950PR_957b", "Ascend950PR_9589"):
        assert get_profile(sku).arch_code == 3510


def test_identifier_only_sku_requires_overrides() -> None:
    incomplete = get_profile("Ascend950PR", sku="Ascend950PR_9579")
    assert incomplete.spec_status == "identifier_only_requires_overrides"
    assert incomplete.prediction_ready is False
    with pytest.raises(ProfileError):
        evaluate_flow(build_flow("rms_norm"), incomplete)
    completed = get_profile("Ascend950PR", sku="Ascend950PR_9579",
                            aic_cores=32, freq_mhz=1650)
    assert completed.prediction_ready is True
    dt = get_profile("Ascend950DT", cores=36)
    assert dt.aic_cores == 36 and dt.hbm_bw_gbps == 4000.0
    with pytest.raises(ProfileError):
        get_profile("Ascend950DT", sku="Ascend950DT_9799")
    with pytest.raises(ProfileError):
        get_profile("Ascend950DT", cores=30)


def test_frequency_and_dtype_consistency() -> None:
    half = get_profile("Ascend950PR", freq_mhz=825)
    assert half.cube_peak_tflops("bf16") == pytest.approx(P.cube_peak_tflops("bf16") / 2)
    assert half.rates.cube_macs_s == pytest.approx(P.rates.cube_macs_s / 2)
    assert half.rates.vector_ops_s == pytest.approx(P.rates.vector_ops_s / 2)
    assert P.cube_peak_tflops("bf16") == pytest.approx(2 * P.chip_cube_macs_s / 1e12)
    assert P.cube_peak_tflops("mxfp4") > P.cube_peak_tflops("bf16")
    assert P.cube_peak_tflops("tf32") < P.cube_peak_tflops("bf16")
    assert P.whitepaper_rounded_peaks("bf16").cube == pytest.approx(432.0)
    assert P.vector_peak_tflops("fp32") < P.vector_peak_tflops("bf16")
    with pytest.raises(ProfileError):
        P.cube_peak_tflops("not-a-dtype")
    with pytest.raises(ProfileError):
        get_profile("Ascend950PR", aic_cores=True)
    with pytest.raises(ProfileError):
        get_profile("Ascend950PR", aic_cores=28.9)
    with pytest.raises(ProfileError):
        get_profile("Ascend950PR", hbm_bw_gbps=float("nan"))


def test_architectural_memcopy_gates_at_account_entry() -> None:
    for src, dst, profile in (
        ("GM", "L0A", P), ("L1", "GM", P), ("UB_LANE0", "L1", P2),
        ("L0C", "UB_LANE0", P2),
    ):
        with pytest.raises(Exception):
            account_transfer(TransferRequest("copy", src_pool=BufferPool(src),
                                             dst_pool=BufferPool(dst)), profile)
    assert validate_mem_copy("GM", "L0A", P2) == "GM->L0A"
    with pytest.raises(TransferError):
        account_transfer(TransferRequest("copy", src_pool="BOGUS"), P)


def test_load3d_source_examples_and_padding() -> None:
    full = account_transfer(TransferRequest(
        "load3d", source_hw=(2, 2), kernel=(2, 2), pad_after=(1, 1),
        read_policy=ReadPolicy.FULL_SOURCE_ONCE, src_pool=BufferPool.L1,
        dst_pool=BufferPool.L0A), P)
    per_ref = account_transfer(TransferRequest(
        "load3d", source_hw=(2, 2), kernel=(2, 2), pad_after=(1, 1),
        read_policy=ReadPolicy.PER_REFERENCE, src_pool=BufferPool.L1,
        dst_pool=BufferPool.L0A), P)
    assert (full.actual_read_bytes, full.logical_output_bytes, full.actual_write_bytes) == (8, 32, 32)
    assert (per_ref.actual_read_bytes, per_ref.logical_output_bytes) == (18, 32)
    shifted = account_transfer(TransferRequest(
        "load3d", source_hw=(3, 3), kernel=(3, 3), stride=(2, 2),
        pad_before=(1, 1), read_policy="per_reference",
        src_pool=BufferPool.L1, dst_pool=BufferPool.L0A), P)
    assert (shifted.actual_read_bytes, shifted.logical_output_bytes) == (8, 18)


def test_slots_alignment_resident_and_strides() -> None:
    two = account_transfer(TransferRequest("copy", rows=128, cols=256, slots=2), P)
    assert two.physical_capacity_bytes == 131072
    assert two.physical_capacity_by_pool["UB_LANE0"] == 131072
    plan = plan_buffer_pool((BufferSpec("x", BufferPool.L1, 1, slots=2),), BufferPool.L1, P)
    assert plan.peak_bytes == 1024
    resident = BufferSpec("resident", BufferPool.UB_LANE0, 200000, resident=True)
    later = BufferSpec("later", BufferPool.UB_LANE0, 100000, first_stage=10, last_stage=10)
    assert plan_buffer_pool((resident, later), BufferPool.UB_LANE0, P).peak_bytes == 300000
    assert plan_buffer_pool((later, resident), BufferPool.UB_LANE0, P).peak_bytes == 300000
    with pytest.raises(TransferError):
        account_transfer(TransferRequest("gm_strided", rows=3, cols=16,
                                         gm_strides=GmStrides(2, 8, 64)), P)
    ok = account_transfer(TransferRequest("gm_strided", rows=4, cols=32,
                                          gm_strides=GmStrides(4, 64, 128)), P, pool=BufferPool.GM)
    assert ok.actual_read_bytes == 256 and ok.address_span_bytes == 448


def test_nz_block_alignment_is_dtype_aware() -> None:
    fp32 = account_transfer(TransferRequest("dn2nz", rows=17, cols=17,
                                            element_bytes=4, dst_pool=BufferPool.L1), P)
    assert fp32.destination_capacity_bytes == 3072
    bf16 = account_transfer(TransferRequest("dn2nz", rows=17, cols=33,
                                            element_bytes=2, dst_pool=BufferPool.L1), P)
    assert bf16.destination_capacity_bytes == 3072
    nz = account_transfer(TransferRequest("nz2dn", rows=17, cols=17,
                                          src_pool=BufferPool.L0C,
                                          dst_pool=BufferPool.GM), P, pool=BufferPool.L0C)
    assert nz.physical_capacity_bytes == nz.source_capacity_bytes == 2048
    sub = account_transfer(TransferRequest("copy", rows=3, cols=3,
                                           bits_per_element=4), P)
    assert sub.logical_input_bytes == 5


def test_pipeline_first_item_and_single_buffer_serialization() -> None:
    flow = OpFlowTemplate(
        name="chain", archetype=ExecutionArchetype.PURE_CUBE, total_units=1,
        base_tile_units=1, live_bytes_per_unit=0,
        pipe_work=(PipeWork("MTE1", P.rates.mte1_bytes_s * 1e-4, "bytes"),
                   PipeWork("CUBE", P.rates.cube_macs_s * 1e-4, "macs"),
                   PipeWork("FIX", P.rates.fix_bytes_s * 1e-4, "bytes")))
    single = evaluate_flow(flow, P, double_buffer=False)
    assert single.t_first_item_us >= 300.0 and single.t_est_us >= 300.0
    two = replace(flow, total_units=2, base_tile_units=1)
    serial_two = evaluate_flow(two, P, active_units=1, double_buffer=False).t_est_us
    double_two = evaluate_flow(two, P, active_units=1, double_buffer=True).t_est_us
    assert serial_two >= 300.0
    assert double_two < serial_two


def test_two_lane_credit_requires_sync_and_ssbuf_capacity() -> None:
    flow = build_flow("mhc_hcpre_cv_splitk")
    synced = evaluate_flow(flow, P, allow_structural_prior=True)
    assert synced.extra["vector_lanes"] == 2
    unsynced = evaluate_flow(replace(flow, sync_pairs=0), P, allow_structural_prior=True)
    assert unsynced.extra["vector_lanes"] == 1
    assert "1c2v_without_explicit_sync" in unsynced.warnings
    over = replace(build_flow("add"), buffers=(BufferSpec("s", BufferPool.SSBUF, 4096),))
    with pytest.raises(Exception):
        evaluate_flow(over, P)
    within = replace(build_flow("add"), buffers=(BufferSpec("s", BufferPool.SSBUF, 1024),))
    assert evaluate_flow(within, P).extra["sram_peak_bytes_by_pool"]["SSBUF"] == 1024


def test_tau_includes_all_pipes_and_recurrence_steps() -> None:
    flow = OpFlowTemplate(
        name="rec", archetype=ExecutionArchetype.PURE_VECTOR, total_units=64,
        base_tile_units=1, live_bytes_per_unit=0,
        pipe_work=(PipeWork("VEC", 1e6), PipeWork("MTE3", 1e6, "bytes")),
        gm_read_bytes=1024, gm_write_bytes=1024, unique_gm_bytes=2048,
        recurrence=LoopCarriedRecurrence(latency_us=50.0, distance_items=2),
        recurrence_steps=4)
    result = evaluate_flow(flow, P, active_units=8)
    assert result.tau_steady_us >= result.recurrence_interval_us == pytest.approx(25.0)
    assert result.tau_steady_us >= result.max_pipe_load_us
    assert result.extra["advance_steps"] == result.waves * 4
    with pytest.raises(FlowSchemaError):
        LoopCarriedRecurrence(latency_us=1.0, distance_items=True)
    with pytest.raises(FlowSchemaError):
        LoopCarriedRecurrence(latency_us=1.0, distance_items=0)


def test_wave_tail_retile_and_unique_working_set() -> None:
    a = evaluate_flow(build_flow("add_rms_norm", tokens=128, hidden=4096, tile=32768), P)
    b = evaluate_flow(build_flow("add_rms_norm", tokens=128, hidden=4096, tile=65536), P)
    assert a.tiling.spill_read_bytes == b.tiling.spill_read_bytes == 0
    assert a.tiling.passes == b.tiling.passes == 1
    assert evaluate_flow(build_flow("add_rms_norm", tokens=1, hidden=8193), P).t_est_us >= \
        evaluate_flow(build_flow("add_rms_norm", tokens=1, hidden=8192), P).t_est_us
    reread = OpFlowTemplate(
        name="reread", archetype=ExecutionArchetype.PURE_VECTOR, total_units=64,
        base_tile_units=1, live_bytes_per_unit=0,
        pipe_work=(PipeWork("VEC", 1e6),), gm_read_bytes=1024 ** 3,
        gm_write_bytes=0, unique_gm_bytes=1024 ** 2)
    result = evaluate_flow(reread, P)
    assert result.working_set_bytes == 1024 ** 2 and result.l2_resident
    with pytest.raises(Exception):
        derive_tiling(100.0, 100.0, 1.0, 253952, requested_tile_units=0)


def test_sram_liveness_consumes_flow_buffers() -> None:
    huge = build_flow("matmul_bf16")
    with pytest.raises(Exception):
        evaluate_flow(replace(huge, buffers=(
            BufferSpec("l1_big", BufferPool.L1, 10 ** 9),
            BufferSpec("ub_big", BufferPool.UB_LANE1, 10 ** 9))), P)
    simt = OpFlowTemplate(
        name="simt", archetype=ExecutionArchetype.PURE_VECTOR, total_units=1000,
        base_tile_units=1000, live_bytes_per_unit=1,
        pipe_work=(PipeWork("SIMT", 1e6),), vector_mode="simt",
        buffers=(BufferSpec("u", BufferPool.UB_LANE0, 230000),))
    with pytest.raises(Exception):
        evaluate_flow(simt, P)
    with pytest.raises(Exception):
        evaluate_flow(replace(simt, vector_mode="simd"), P2)


def test_selected_independent_service_bound_core_scaling() -> None:
    for name in ("add_rms_norm", "matmul_bf16", "flash_attn_score"):
        flow = build_flow(name)
        limit = P.active_units(flow.archetype.value)
        previous = float("inf")
        for active in range(1, limit + 1):
            value = evaluate_flow(flow, P, active_units=active).t_est_us
            assert value <= previous + 1e-9, (name, active, value, previous)
            previous = value
    base = build_flow("add_rms_norm")
    values = [evaluate_flow(base, get_profile("Ascend950PR", aic_cores=n)).t_est_us
              for n in (28, 32, 36)]
    assert values == sorted(values, reverse=True)


def test_source_exact_catalog_contracts() -> None:
    block = build_flow("block_attn_res_update_rms_norm")
    assert block.params["hidden"] == 7168
    assert block.recurrence is None
    assert block.unique_gm_bytes == 10 * 128 * 7168 + 8 * 128 + 6 * 7168 + 6 * 128 * 7168
    noop = evaluate_flow(build_flow("block_attn_res_update_rms_norm", tokens=0), P)
    assert noop.t_est_us == 0.0 and noop.launched is False
    engram = build_flow("engram_gate", hidden=8512)
    assert engram.params["hc_mult"] == 4 and engram.params["passes"] == 2
    assert build_flow("engram_gate", hidden=8448).params["passes"] == 1
    assert build_flow("fused_recurrent_kda_replayssm").archetype is ExecutionArchetype.PURE_VECTOR
    macs = lambda split: sum(w.total_work for w in build_flow(
        "mhc_hcpre_cv_splitk", split_k=split).pipe_work if w.pipe == "CUBE")
    assert macs(1) == macs(8)
    assert build_flow("mhc_hcpost").params["hc_mult"] == 4
    assert build_flow("mega_mla").params["window_record_bytes"] == 544
    assert build_flow("mega_mla").params["compressed_record_bytes"] == 320
    assert build_flow("mega_full_index").params["topk"] == 512
    modes = [evaluate_flow(build_flow("moonep", hccl_expansion_mode=mode), P, allow_structural_prior=True).t_est_us
             for mode in ("AIV", "CCU_SCHED", "AICPU")]
    assert len(set(round(value, 6) for value in modes)) >= 2


def test_uncalibrated_catalog_entries_fail_closed() -> None:
    with pytest.raises(UncalibratedOperatorError):
        evaluate_operator("attn_prologue", P)
    with pytest.raises(UncalibratedOperatorError):
        build_flow("attn_prologue", allow_structural_prior=False)
    prior = build_flow("attn_prologue")
    assert prior.estimate_kind == "structural_schema_only" and prior.calibrated is False
    with pytest.raises(UncalibratedOperatorError):
        evaluate_flow(prior, P)
    result = evaluate_flow(prior, P, allow_structural_prior=True)
    assert "structural_schema_only_uncalibrated_prior" in result.warnings
    allowed = evaluate_operator("attn_prologue", P, allow_structural_prior=True)
    assert allowed.calibrated is False


def test_triton_serialism_spill_gather_and_mfu() -> None:
    serial = analyze_triton_lowering(P, aic_us=100, aiv_us=1,
                                     features={"has_cv_pingpong": False})
    assert serial.t_overlapped_us >= 100 and serial.serialism_ratio == pytest.approx(1.0)
    assert serial.potential_overlap_gain == pytest.approx(1.01)
    ping = analyze_triton_lowering(P, aic_us=100, aiv_us=100,
                                   features={"has_cv_pingpong": True})
    assert ping.t_overlapped_us >= 100 and ping.serialism_ratio > 1.0
    spill = analyze_triton_lowering(P, aic_us=10, aiv_us=10,
                                    features={"chained_dot_fp32_acc_spill": True,
                                              "chained_buffer_bytes": 65536})
    assert spill.spill_penalty_us == pytest.approx(0.02048)
    l2_gather = analyze_triton_lowering(P, aic_us=1, aiv_us=1,
                                        features={"random_gather_to_dot_mode": "in_kernel_l2_stage",
                                                  "gather_bytes": 64 * 1024 ** 2},
                                        working_set_bytes=1024,
                                        per_core_working_set_bytes=1024)
    hbm = analyze_triton_lowering(P, aic_us=1, aiv_us=1,
                                  features={"random_gather_to_dot_mode": "in_kernel_l2_stage",
                                            "gather_bytes": 64 * 1024 ** 2},
                                  working_set_bytes=2 * P.memory.l2,
                                  per_core_working_set_bytes=2 * P.memory.l2)
    assert hbm.gather_us > l2_gather.gather_us
    zero = evaluate_mfu_gates(P, wall_us=1, features=TritonFeatures(flops=0),
                              working_set_bytes=1, total_bytes=1)
    assert not zero.valid and zero.valid is False
    over = evaluate_mfu_gates(P, wall_us=1,
                              features=TritonFeatures(flops=1, input_bytes=20_000_000_000),
                              working_set_bytes=1, total_bytes=1)
    assert not over.gates["input_budget_ok"]
    with pytest.raises(TritonLoweringError):
        TritonFeatures.from_mapping({"bogus_feature": 1})
    with pytest.raises(TritonLoweringError):
        TritonFeatures(gather_bytes=float("nan"))
    no_mla = analyze_triton_lowering(P, aic_us=1, aiv_us=1,
                                     features={"mla_block_qo": 4})
    assert no_mla.nonaffine_store_hazard is False


def test_aclgraph_draft_evidence_and_mode_order() -> None:
    assert ACLGraphWrapperModel(use_eagle=True).replay_sync_required is True
    assert ACLGraphWrapperModel(use_eagle=True, is_draft_model=True).replay_sync_required is False
    assert UpdatableGraphModel(use_eagle=True).needs_sync is True
    assert UpdatableGraphModel(use_eagle=True, is_draft_model=True).needs_sync is False
    a = StreamInterval("vector_a", 0, 0, 10, aiv_blocks=40, io_bytes=1024)
    b = StreamInterval("vector_b", 1, 20, 30, aiv_blocks=0, io_bytes=1024)
    diag = diagnose_region(ParallelRegion("no_events", a, b, event_pairs=0))
    assert diag.cause != "premature_event_record_wait"
    vectors = diagnose_region(ParallelRegion("two_vectors", a, b, event_pairs=0))
    assert vectors.cause != "hccl_aiv_expansion_steals_vector_cores"
    hccl = diagnose_region(ParallelRegion(
        "hccl", StreamInterval("AllReduce", 0, 0, 10, aiv_blocks=40, io_bytes=4096),
        StreamInterval("dispatch", 1, 0, 10, aiv_blocks=30, io_bytes=4096),
        aiv_cores=64))
    assert hccl.code == "A3" and "hccl_aiv" in hccl.cause
    assert hccl.evidence_grade in ("supported_hypothesis", "observed", "confirmed")
    dangling = diagnose_region(ParallelRegion(
        "dangling", a, StreamInterval("b", 1, 0, 10, waits_on_capture_root=False),
        capture_root_stream_id=0))
    assert dangling.code == "A5"
    assert effective_launch_floor_us("aclgraph_superkernel") == 0.32
    assert effective_launch_floor_us("superkernel") == 0.32
    with pytest.raises(ACLGraphModelError):
        effective_launch_floor_us("bad_backend", superkernel=True)
    with pytest.raises(ACLGraphModelError):
        effective_launch_floor_us("piecewise", superkernel=True)
    assert classify_capture_error("error 207008 stream resources are insufficient") == "stream_resource_207008"
    assert classify_capture_error("Alloc sq cq fail") == "old_hdk_alloc_sq_cq"


def test_distiller_provenance_and_normalization() -> None:
    units = dist.distill_static()
    assert len(units) == 110
    assert dist.stack_counts(units) == {
        "ascendc": 3, "cannbot_dsl": 59, "framework_aclgraph": 7, "triton_ascend": 41,
    }
    for unit in units:
        assert len(unit.source_commit) == 40
        assert len(unit.provenance["content_sha256"]) == 64
        assert len(unit.body.encode("utf-8")) <= unit.token_count
        assert unit.source_path and not unit.source_path.startswith(f"{unit.source_repo}/")
    legacy = dist.normalize_text("SuperKernel 仅支持 ge_graph 模式，不支持 aclgraph。")
    assert legacy.preserved_legacy and legacy.text == "SuperKernel 仅支持 ge_graph 模式，不支持 aclgraph。"
    assert legacy.classification == "torchair_legacy_only"
    assert "仅支持 aclgraph" not in legacy.text
    kernel = dist.normalize_text("Use 248 KiB UB for a SIMD kernel.")
    assert kernel.applicability == "operator_kernel_agnostic"
    adapted = dist.normalize_text(
        'exe_mode: "ge_graph"; SuperKernel does not support aclgraph; Notify_Wait')
    assert adapted.preserved_legacy and adapted.text == adapted.source_text
    with pytest.raises(KnowledgeUnitError):
        KnowledgeUnit("unit-xyz", "t", "L1", "axiom", "ascendc",
                      "operator_kernel_agnostic", "long body", token_count=1,
                      provenance={"source_repo": "r", "source_path": "a.md",
                                  "source_commit": "0" * 40, "content_sha256": "0" * 64})
    unit = KnowledgeUnit("unit-xyz", "t", "L1", "axiom", "ascendc",
                         "operator_kernel_agnostic", "汉字",
                         provenance={"source_repo": "r", "source_path": "a.md",
                                     "source_commit": "0" * 40, "content_sha256": "0" * 64})
    assert unit.token_count == len("汉字".encode("utf-8"))


def test_live_provenance_when_corpus_present(tmp_path: Path) -> None:
    if not (REF_ROOT / "cannbot-dsl/catalog/operators.json").exists():
        pytest.skip("live reference corpus unavailable")
    live = dist.scan_live(REF_ROOT, VLLM_ROOT)
    assert live.ok and live.live_verified and not live.degraded_offline
    units = dist.distill_units(live)
    assert len(units) == 110
    repos = {"cannbot-skills": REF_ROOT / "cannbot-skills",
             "cannbot-dsl": REF_ROOT / "cannbot-dsl",
             "cannbot-knowledge": REF_ROOT / "cannbot-knowledge",
             "vllm-ascend": VLLM_ROOT}
    for unit in units:
        root = repos[unit.source_repo]
        assert (root / unit.source_path).exists(), (unit.unit_id, unit.source_path)
        assert unit.source_commit == live.source_commits[unit.source_repo]
    target = dist.write_units(units, tmp_path)
    assert target.name == "units.jsonl" and len(target.read_text().splitlines()) == 110


def test_nr_s4_graded_unit_schema_key_compatibility() -> None:
    reference = Path(
        "/Volumes/data/Codex/2026-11-12/kan-x/work/next-release/nr_s4_knowledge_units/"
        "corpus/knowledge-operators/_common/"
        "unit-ascend-aclgraph-bucket-padding-and-varlen-axioms.md"
    )
    assert reference.exists()
    front = reference.read_text(encoding="utf-8").split("---", 2)[1]
    keys = {line.split(":", 1)[0] for line in front.splitlines()
            if re.match(r"^[a-z_]+:", line)}
    unit = dist.distill_static()[0].to_dict()
    assert keys <= set(unit)
    assert {"source_locator", "source_digest"} <= set(unit["provenance"])
    assert unit["schema"] == "mindie-graded-unit/1" and unit["revision"] >= 1


def test_timeout_is_not_swallowed(monkeypatch) -> None:
    def timeout(**kwargs):
        raise TimeoutError("audit injected source timeout")
    monkeypatch.setattr(dist, "scan_live", timeout)
    with pytest.raises(TimeoutError):
        dist.distill_units()
    monkeypatch.setattr(dist, "scan_live", timeout)
    with pytest.raises(TimeoutError):
        dist.distill_units(allow_offline=True)


def test_compact_mcp_envelopes_keep_identity() -> None:
    responses = [
        op_call("opflow_catalog", {}),
        op_call("opflow_hardware", {}),
        op_call("opflow_evaluate", {"op": "block_attn_res_update_rms_norm"}),
        op_call("opflow_evaluate", {"op": "fused_norm_rope_attn_rope_cast_fwd"}),
        op_call("opflow_transfer", {"kind": "dn2nz", "rows": 128, "cols": 256}),
        op_call("opflow_triton", {}),
        acl_call("aclgraph_diagnose", {"region": {
            "name": "r", "branch_a": {"op": "AllReduce", "stream_id": 0, "start_us": 0, "end_us": 10},
            "branch_b": {"op": "dispatch", "stream_id": 1, "start_us": 0, "end_us": 10}}}),
        acl_call("aclgraph_launch_floor", {"mode": "bad_backend", "superkernel": True}),
    ]
    for response in responses:
        assert envelope_size(response)["utf8_bytes"] <= 250
        assert envelope_size(response)["token_estimate"] <= 250
        assert response.get("err") != "EBUDGET"
    hardware = op_call("opflow_hardware", {})
    assert hardware["sku"] == "Ascend950PR_9589" and hardware["l0c"] == 262144
    evaluated = op_call("opflow_evaluate", {"op": "block_attn_res_update_rms_norm"})
    assert evaluated["op"] == "block_attn_res_update_rms_norm"
    assert evaluated["mode"] == "aclgraph_full"
    assert evaluated["warn"] == []
