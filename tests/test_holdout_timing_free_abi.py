"""Timing-free ABI holdout binding for H1/H2/H3 on the real Engine.

[Role] Feed the sealed-input package's timing-free descriptors (MatMulV3 with a
  K tail, 4-slot INT8 QBMM with absent optional bias, paged FIA with an
  ``actual_seq_lengths_kv`` page tail) through the generic ACLNN slot binder and
  prove every request becomes a modeled physical submission with a real
  execution receipt.
[Invariants] The only holdout file read is the ``inputs-only`` timing-free
  descriptor package; no observation, result, handoff or evidence file is
  consulted.  Predictions are derived from declared geometry, the target SKU's
  physical rates and the selected Cube/MTE2/Fixpipe program, never from a
  per-shape table.  The same geometry with two explicit tile selections must
  produce distinct physical stage work and distinct engine terminals.
[Timeouts & Locks] CPU-only virtual engine; bounded one-shot submissions.
[Side-Effect Policy] Read-only except for the explicitly requested executions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Importing the ordered-slot binder installs the captured ACLNN positional
# schemas (MatMulV3 / 4-slot QBMM / paged FIA) into the ABI catalogue.
from mindie_sim.prediction.piecewise import native_binding
from mindie_sim.prediction.piecewise.native_invocation import simulate_aclnn_call

SKU = "Ascend950DT_9582"
_INPUT_NAME = "timing-free-prediction-inputs.json"


def _inputs_path() -> Path:
    candidates = []
    declared = os.environ.get("A5_HOLDOUT_INPUTS")
    if declared:
        candidates.append(Path(declared))
    candidates.extend((
        Path("/root/agy-w1-b") / "holdout-inputs.json",
        Path("/Volumes/data/Codex/2026-11-12/kan-x/outputs/"
             "a5-implementation/holdout/inputs-only") / _INPUT_NAME,
        Path(__file__).resolve().parents[1] / "holdout-inputs.json",
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError(
        "timing-free holdout inputs not found; set A5_HOLDOUT_INPUTS to the "
        "inputs-only descriptor JSON")


@pytest.fixture(scope="module")
def inputs():
    return json.loads(_inputs_path().read_text())


@pytest.fixture(scope="module")
def predictions(inputs):
    pytest.importorskip("mindie_profiler")
    from mindie_profiler.sim_bridge import predict_native_requests
    return predict_native_requests(
        inputs, hardware=SKU, execute=True, blocking=True)


def test_all_three_requests_are_modeled_with_positive_physical_duration(
        predictions):
    rows = {row["request_id"]: row for row in predictions["predictions"]}
    assert set(rows) == {
        "H1-bf16-matmul-k-tail", "H2-int8-qbmm-k-tail",
        "H3-paged-fia-page-tail"}
    assert predictions["all_modeled"] is True
    assert all(row["status"] == "modeled" for row in rows.values())
    assert predictions["counts"]["requests"] == 3
    assert predictions["counts"]["submitted"] == 3
    assert predictions["counts"].get("status_unmodeled", 0) == 0
    for request_id, row in rows.items():
        assert row["status"] == "modeled", (request_id, row.get("reason"))
        duration = row["duration_ms"]
        assert duration is not None and duration > 0.0, request_id
        assert row["eligibility"]["cpu_mechanism"] is True
        assert row["eligibility"]["selected_plan"] is True
        actual = row["actual_execution"]
        assert actual is not None, request_id
        assert actual["status"] in {"completed", "predicted"}, request_id
        assert actual["call_id"]
        assert actual["selected_program_id"]
        assert actual["terminal"]["done"] is True
        assert actual["terminal"]["time"] is not None
        assert actual["start"]["time"] is not None
        assert actual["terminal"]["time"] >= actual["start"]["time"]
        assert actual["duration_ms"] == pytest.approx(duration)
        assert row["t12"]["status"] == "predicted"
        assert row["t12"]["binding"] is not None
        assert row["t12"]["binding"]["call_id"]


def test_k_tail_matmul_prices_whole_cube_waves(predictions):
    row = {item["request_id"]: item
           for item in predictions["predictions"]}["H1-bf16-matmul-k-tail"]
    program = row["t12"]["program"]
    physical = program["physical"]
    geometry = physical["geometry"]
    work = physical["work"]
    assert geometry["k"] == 257
    assert geometry["k"] % geometry["tile"][2] != 0
    expected_waves = (
        (geometry["m"] + geometry["tile"][0] - 1) // geometry["tile"][0]
        * ((geometry["n"] + geometry["tile"][1] - 1) // geometry["tile"][1])
        * ((geometry["k"] + geometry["tile"][2] - 1) // geometry["tile"][2]))
    assert geometry["cube_waves"] == expected_waves
    logical_macs = geometry["m"] * geometry["n"] * geometry["k"]
    assert work["cube"]["issued_macs"] == (
        expected_waves * geometry["tile"][0] * geometry["tile"][1]
        * geometry["tile"][2])
    assert work["cube"]["issued_macs"] > logical_macs
    assert geometry["transpose_x1"] is False
    assert geometry["transpose_x2"] is False
    execution = simulate_aclnn_call(
        row["t12"]["invocation"], hardware=SKU, execute=True)
    assert execution.status == "predicted", execution.reasons
    catalog = execution.execution_catalog
    assert catalog is not None
    # The submitted catalog is the multi-stage Cube/MTE/Fixpipe program, not one
    # aggregate row: HBM/MTE2 -> MTE1 -> Cube matrix -> Fixpipe drain.
    roles = {term.physical_resource_role
             for stage in catalog.stages for term in stage.work}
    assert {"hbm", "mte2", "mte1", "matrix", "fixpipe"} <= roles


def test_four_slot_qbmm_has_no_bias_or_nz_slot_but_keeps_real_operands(
        predictions):
    row = {item["request_id"]: item
           for item in predictions["predictions"]}["H2-int8-qbmm-k-tail"]
    invocation = row["t12"]["invocation"]
    tensors = invocation["tensors"]
    assert set(tensors) == {
        "x", "weight", "weight_scale", "pertoken_scale", "bias", "output"}
    # The optional bias position is retained but explicitly absent; it is not
    # filled with a guessed descriptor and not charged.
    assert tensors["bias"]["present"] is False
    assert "weight_nz" not in tensors
    assert tensors["x"]["dtype"] == "int8"
    assert tensors["weight"]["dtype"] == "int8"
    assert tensors["weight_scale"]["dtype"] == "bf16"
    assert tensors["pertoken_scale"]["dtype"] == "fp32"
    assert tensors["output"]["dtype"] == "bf16"
    geometry = row["t12"]["program"]["physical"]["geometry"]
    assert geometry["k"] == 258
    assert geometry["k"] % 8 != 0
    assert row["t12"]["program"]["physical"]["work"]["bytes"]["total"] > 0


def test_paged_fia_page_tail_uses_actual_lengths_and_block_layout(
        predictions):
    row = {item["request_id"]: item
           for item in predictions["predictions"]}["H3-paged-fia-page-tail"]
    invocation = row["t12"]["invocation"]
    tensors = invocation["tensors"]
    assert tensors["query"]["shape"] == [2, 8, 1, 128]
    assert tensors["key"]["shape"] == [5, 8, 128, 128]
    assert tensors["block_table"]["shape"] == [2, 3]
    assert tensors["actual_seq_lengths_kv"]["shape"] == [2]
    plan = row["t12"]["program"]["physical"]["native_paged_fia"]
    geometry = plan["geometry"]
    paging = plan["captured_paging"]
    assert geometry["page_size"] == 128
    assert paging["paged_layout"] == "BNBD"
    assert paging["requests"] == [[1, 128], [1, 256]]
    assert paging["page_counts"] == [2, 3]
    assert paging["kv_lengths"] == [129, 257]
    assert geometry["kv_heads"] == geometry["query_heads"] == 8
    # The 257-token request ends one row into its third page: the plan keeps
    # that page tail and does not round the visible KV extent to page size.
    assert plan["work"]["kv_valid_bytes"] > 0
    assert plan["work"]["pages"] == 5
    assert row["actual_execution"]["status"] in {
        "completed", "predicted"}


def test_explicit_tile_selection_changes_stage_work_and_engine_terminal(
        predictions):
    row = {item["request_id"]: item
           for item in predictions["predictions"]}["H1-bf16-matmul-k-tail"]
    invocation = row["t12"]["invocation"]
    wide = simulate_aclnn_call(
        invocation, hardware=SKU, execute=True,
        selection={"program": {"options": {"tile": [16, 32, 16],
                                           "slots": 2}}})
    tall = simulate_aclnn_call(
        invocation, hardware=SKU, execute=True,
        selection={"program": {"options": {"tile": [16, 16, 16],
                                           "slots": 2}}})
    assert wide.status == "predicted", wide.reasons
    assert tall.status == "predicted", tall.reasons
    assert wide.execution_catalog is not None
    assert tall.execution_catalog is not None
    assert wide.execution_catalog.stages != tall.execution_catalog.stages
    wide_work = sum(
        term.issued for stage in wide.execution_catalog.stages
        for term in stage.work if term.physical_resource_role == "matrix")
    tall_work = sum(
        term.issued for stage in tall.execution_catalog.stages
        for term in stage.work if term.physical_resource_role == "matrix")
    assert wide_work != tall_work
    wide_ms = wide.prediction["duration_ms"]
    tall_ms = tall.prediction["duration_ms"]
    assert wide_ms > 0 and tall_ms > 0
    assert wide_ms != tall_ms
    wide_terminal = wide.execution_receipt["terminal"]["time"]
    tall_terminal = tall.execution_receipt["terminal"]["time"]
    assert wide_terminal != tall_terminal


def test_binder_resolves_operator_and_captured_symbol_aliases():
    assert native_binding.install_catalogue()["installed"] is True
    from mindie_sim.prediction.piecewise.native_calls import native_abi_spec
    for alias in ("MatMulV3", "aclnnMatmul_MatMulV3Common_MatMulV3",
                  "QuantBatchMatmulV3", "FusedInferAttentionScore"):
        spec = native_abi_spec(alias)
        assert spec is not None, alias
    assert native_abi_spec("torch.matmul").dispatch == "generic_matrix_abi"
    assert native_abi_spec(
        "torch_npu.npu_fused_infer_attention_score").dispatch == (
            "paged_fia_build")
