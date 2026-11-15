"""T12 production execution: selected native programs on one caller Execution.

[Role] Run the selected matrix, ordinary paged-FIA, GDN and RoPE programs on
  the caller's real Execution, including a non-blocking embedding.
[Invariants] Stage/work/ports come from the same plan that was selected from
  the normalized invocation; no second plan, whole-call task or target
  duration is created.  Blocking advances only to this terminal.
[Side-Effect Policy] CPU Engine only; no device, weights or I/O.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from mindie_sim.prediction.piecewise.native_invocation import (
    _hardware_execution, native_execution_receipt, simulate_aclnn_call,
    submit_native_call,
)
from test_native_abi_catalog_dispatch import _matrix_request
from test_native_attnstate_abi_leaves import _gdn_request, _rope_request
from test_native_matrix_abi_leaves import R31

SKU = "Ascend950DT_9582"


def _execution():
    execution = _hardware_execution(SKU)
    execution.engine.extend(pools={"scratch": 64})
    return execution


def _matrix_bf16_request():
    return _matrix_request(
        "aclnnMatmul", "aclnn",
        {"x1": {"shape": [4, 65], "dtype": "bf16", "role": "input"},
         "x2": {"shape": [65, 32], "dtype": "bf16", "role": "input"},
         "output": {"shape": [4, 32], "dtype": "bf16",
                    "role": "output"}},
        {}, R31)


def _matrix_int8_request():
    return _matrix_request(
        "torch_npu.npu_quant_matmul", "torch",
        {"x": {"shape": [3, 66], "dtype": "int8", "role": "input"},
         "weight": {"shape": [66, 32], "dtype": "int8", "role": "weight"},
         "weight_scale": {"shape": [32], "dtype": "bf16",
                          "role": "scale"},
         "pertoken_scale": {"shape": [3], "dtype": "fp32",
                            "role": "scale"},
         "output": {"shape": [3, 32], "dtype": "bf16",
                    "role": "output"}},
        {"weight_scale_storage_format": "bf16"}, R31)


def _fia_request(*, key_layout="PA_BBND", **attrs):
    return {
        "api": {"namespace": "torch",
                "entry": "torch_npu.npu_fused_infer_attention_score"},
        "version": {"api": "cann-9.2.0"},
        "tensors": {
            "query": {"shape": [2, 4, 128], "dtype": "bf16",
                      "layout": "TND", "role": "input"},
            "key_cache": {"shape": [3, 128, 4, 128], "dtype": "bf16",
                          "layout": key_layout, "role": "input"},
            "value_cache": {"shape": [3, 128, 4, 128], "dtype": "bf16",
                            "layout": key_layout, "role": "input"},
            "block_table": {"shape": [2, 2], "dtype": "int32",
                            "layout": "row_major", "role": "input"},
            "sequence_lengths": {"shape": [2], "dtype": "int32",
                                 "layout": "row_major", "role": "input",
                                 "values": [127, 130]},
            "output": {"shape": [2, 4, 128], "dtype": "bf16",
                       "layout": "TND", "role": "output"},
        },
        "attrs": {"block_size": 128, "num_heads": 4,
                  "num_kv_heads": 4, "causal": 1, **attrs},
        "conditions": {"phase": "decode"},
    }


def _selection(key_tile):
    return {
        "selector": "test",
        "selected": {"implementation": "staged_gqa_decode"},
        "program": {"recipes": ["staged_gqa_decode"],
                    "options": {"key_tile": key_tile, "slots": 2}},
    }


def test_matrix_bf16_control_runs_load_compute_store_on_caller_execution():
    execution = _execution()
    result = simulate_aclnn_call(
        _matrix_bf16_request(), hardware=SKU, execution=execution,
        blocking=True)
    assert result.status == "predicted", result.reasons
    program = result.execution_program
    assert program.kind == "matrix"
    assert program.plan is result.operator_plan
    assert program.prepared is True
    assert [item["stage"] for item in program.stages] == [
        "load", "compute", "store"]
    assert program.plan.work.matrix == {"bf16": 16640.0}
    assert program.plan.work.resources == {
        "hbm": 4936.0, "mte2": 4680.0, "mte3": 256.0}
    assert result.execution_handle.done
    assert result.execution_receipt["duration_ms"] > 0
    serviced = result.execution_receipt["serviced"]
    assert serviced["memory@r0:s0"] == 4936.0
    assert serviced["mte3@r0:s0"] == 256.0
    assert (serviced.get("mte2_aic@r0:s0", 0.0)
            + serviced.get("mte2_aiv@r0:s0", 0.0)) == 4680.0
    assert result.prediction["work"]["cube"]["macs"] == 8320
    assert execution.engine.peak_allocated["l1@r0:s0"] > 0
    assert execution.engine.peak_allocated["l0c@r0:s0"] > 0
    assert result.program["physical"]["selected_call"]["entry"] == (
        "aclnnMatmul")
    assert [item["shape"] for item in program.ports] == [
        [4, 65], [65, 32], [4, 32]]
    assert result.program["execution"]["terminal"]["done"] is True


def test_matrix_bf16_control_async_embedding_does_not_advance_implicitly():
    execution = _execution()
    pending = simulate_aclnn_call(
        _matrix_bf16_request(), hardware=SKU, execution=execution,
        blocking=False)
    assert pending.status == "predicted"
    assert not pending.execution_handle.done
    assert execution.engine.completed_count == 0
    assert execution.engine.serviced["memory@r0:s0"] == 0.0
    execution.engine.advance(until=(pending.execution_handle.event,))
    receipt = native_execution_receipt(pending, execution)
    assert pending.execution_handle.done
    assert receipt["duration_ms"] > 0
    assert receipt["terminal"]["done"] is True


def test_submit_native_call_executes_the_prepared_matrix_program():
    prepared = simulate_aclnn_call(_matrix_bf16_request(), hardware=SKU)
    assert prepared.status == "predicted"
    assert prepared.execution_program is not None
    assert prepared.execution_program.prepared is False
    execution = _execution()
    receipt = submit_native_call(prepared, execution, blocking=True)
    assert receipt["handle"].done
    assert receipt["duration_ms"] > 0
    assert prepared.execution_program.prepared is True
    assert execution.engine.completed_count > 0


def test_matrix_int8_nd_runs_epilogue_and_store_stages():
    execution = _execution()
    result = simulate_aclnn_call(
        _matrix_int8_request(), hardware=SKU, execution=execution)
    assert result.status == "predicted", result.reasons
    stages = [item["stage"] for item in result.execution_program.stages]
    assert stages[-3:] == ["scales", "epilogue", "store"]
    assert result.prediction["work"]["bytes"]["total"] == 2578
    assert result.prediction["work"]["cube"]["macs"] == 6336
    serviced = result.execution_receipt["serviced"]
    assert serviced["memory@r0:s0"] == 2578.0
    assert serviced["mte3@r0:s0"] == 192.0
    assert serviced["vector@r0:s0"] == 384.0
    assert (serviced.get("mte2_aic@r0:s0", 0.0)
            + serviced.get("mte2_aiv@r0:s0", 0.0)) == 2386.0


def test_ordinary_paged_fia_control_runs_the_selected_plan_and_ports():
    execution = _execution()
    result = simulate_aclnn_call(
        _fia_request(), hardware=SKU, execution=execution)
    assert result.status == "predicted", result.reasons
    native = result.program["physical"]["native_paged_fia"]
    assert native["geometry"] == {
        "batch": 2, "query_heads": 4, "kv_heads": 4, "head_dim": 128,
        "group": 1, "hq_equals_hkv": True, "page_size": 128,
        "max_pages": 2, "page_counts": (1, 2)}
    assert native["work"]["head_token_pairs"] == 1028
    assert native["work"]["qk_mac"] == 131584
    assert native["work"]["pv_mac"] == 131584
    assert native["work"]["q_bytes"] == 2048
    assert native["work"]["out_bytes"] == 2048
    assert native["work"]["kv_valid_bytes"] == 526336
    assert native["work"]["lse_bytes"] == 0
    assert native["work"]["state_writes"] == 0
    assert native["ports"]["key_cache"]["layout"] == "PA_BBND"
    assert native["ports"]["block_table"]["shape"] == (2, 2)
    assert native["state"]["read_only_kv"] is True
    assert native["limits"]["d256_specialized_ifa"] is False
    assert result.execution_program.kind == "paged_fia"
    assert result.execution_program.plan is result.operator_plan
    assert result.execution_program.prepared is True
    serviced = result.execution_receipt["serviced"]
    assert serviced["memory@r0:s0"] == pytest.approx(530480.0)
    assert serviced["mte3@r0:s0"] == pytest.approx(2048.0)
    assert serviced["vector@r0:s0"] == pytest.approx(12956.0)
    assert execution.engine.peak_allocated["l1@r0:s0"] > 0
    assert execution.engine.peak_allocated["ub@r0:s0"] > 0
    assert result.execution_receipt["duration_ms"] > 0


def test_paged_fia_async_embedding_and_grant_last_use_are_real_events():
    execution = _execution()
    reader = execution.engine.future("reader_done")
    pending = simulate_aclnn_call(
        _fia_request(), hardware=SKU, execution=execution, blocking=False,
        output_grants={"output": {
            "pool": "scratch", "size": 32, "name": "fia_out_grant",
            "last_reader": reader}})
    handle = pending.execution_handle
    assert not handle.done
    assert execution.engine.allocated["scratch"] == 0.0
    execution.engine.advance(until=(handle.event,))
    assert handle.done
    assert execution.engine.allocated["scratch"] == 32.0
    assert not any(event.done for event in handle.release_events)
    execution.engine.resolve(reader)
    execution.engine.advance(until=tuple(handle.release_events),
                             first=False)
    assert execution.engine.allocated["scratch"] == 0.0
    assert all(event.done for event in handle.release_events)


def test_gdn_and_rope_use_the_callers_execution_asynchronously():
    gdn_execution = _execution()
    gdn = simulate_aclnn_call(
        _gdn_request(), hardware=SKU, execution=gdn_execution,
        blocking=False)
    assert gdn.status == "predicted", gdn.reasons
    assert gdn.execution_program.kind == "gdn"
    assert not gdn.execution_handle.done
    gdn_execution.engine.advance(until=(gdn.execution_handle.event,),
                                 first=False)
    gdn_receipt = native_execution_receipt(gdn, gdn_execution)
    assert gdn_receipt["completed_tasks"] > 0
    assert gdn_receipt["duration_ms"] > 0
    assert gdn.execution_program.to_dict()["metadata"]["state_ports"]
    assert gdn.execution_program.to_dict()["stages"] == [
        "prepare", "gated_delta"]
    state_ports = gdn.execution_program.to_dict()["metadata"]["state_ports"]
    assert state_ports
    assert all(port["last_use"] and port["last_reader"]
               and "terminal" in port for port in state_ports)
    assert gdn.execution_program.to_dict()["metadata"][
        "binding_relation"]["charge"] == "call"

    rope_execution = _execution()
    rope = simulate_aclnn_call(
        _rope_request(), hardware=SKU, execution=rope_execution,
        blocking=False)
    assert rope.status == "predicted", rope.reasons
    assert rope.execution_program.kind == "rope"
    assert not rope.execution_handle.done
    rope_execution.engine.advance(until=(rope.execution_handle.event,),
                                  first=False)
    rope_receipt = native_execution_receipt(rope, rope_execution)
    assert rope_receipt["duration_ms"] > 0
    assert rope_receipt["serviced"]["memory@r0:s0"] > 0
    assert rope_receipt["serviced"]["mte3@r0:s0"] > 0
    assert [item["name"] for item in rope.execution_program.ports] == [
        "qk", "cos", "sin", "output"]
    assert rope.execution_program.program["output_alias"] == {
        "output": "qk", "in_place": True}
    assert rope.execution_program.program["access"][
        "effective_read_bytes"] > 0


def test_explicit_selection_changes_the_submitted_fia_program_and_terminal():
    timings = {}
    for key_tile in (128, 64):
        execution = _execution()
        result = simulate_aclnn_call(
            _fia_request(), hardware=SKU, execution=execution,
            selection=_selection(key_tile))
        assert result.status == "predicted", result.reasons
        native = result.program["physical"]["native_paged_fia"]
        assert native["strategy"]["key_tile"] == key_tile
        assert result.program["selection"] == _selection(key_tile)
        assert result.execution_program.selection == _selection(key_tile)
        timings[key_tile] = result.execution_receipt["duration_ms"]
    assert timings[128] != timings[64]


def test_matrix_output_grant_holds_until_last_reader_then_releases():
    execution = _execution()
    reader = execution.engine.future("matrix_reader_done")
    pending = simulate_aclnn_call(
        _matrix_bf16_request(), hardware=SKU, execution=execution,
        blocking=False,
        output_grants={"output": {
            "pool": "scratch", "size": 32, "name": "matrix_out_grant",
            "last_reader": reader}})
    handle = pending.execution_handle
    assert not handle.done
    execution.engine.advance(until=(handle.event,))
    assert handle.done
    assert execution.engine.allocated["scratch"] == 32.0
    assert not any(event.done for event in handle.release_events)
    execution.engine.resolve(reader)
    execution.engine.advance(until=tuple(handle.release_events),
                             first=False)
    assert execution.engine.allocated["scratch"] == 0.0
    assert all(event.done for event in handle.release_events)


def test_hardware_only_execution_runs_without_a_model_or_caller():
    for request in (_matrix_bf16_request(), _fia_request()):
        result = simulate_aclnn_call(request, hardware=SKU, execute=True)
        assert result.status == "predicted", result.reasons
        assert result.execution_handle.done
        assert result.execution_receipt["duration_ms"] > 0
        assert result.execution_receipt["terminal"]["done"] is True


def test_matrix_selection_changes_the_actual_tile_program_and_terminal():
    timings = {}
    for tile in ([16, 32, 16], [16, 16, 16]):
        execution = _execution()
        selection = {"program": {"options": {"tile": tile, "slots": 2}}}
        result = simulate_aclnn_call(
            _matrix_bf16_request(), hardware=SKU, execution=execution,
            selection=selection)
        assert result.status == "predicted", result.reasons
        descriptor = result.execution_program.plan.metadata["operator_plan"]
        assert descriptor["tile"] == tile
        timings[tuple(tile)] = result.execution_receipt["duration_ms"]
    assert timings[(16, 32, 16)] != timings[(16, 16, 16)]


def test_paged_fia_non_pa_layout_is_no_match_with_no_work():
    execution = _execution()
    result = simulate_aclnn_call(
        _fia_request(key_layout="ND"), hardware=SKU, execution=execution)
    assert result.status == "no_match"
    assert result.execution_program is None
    assert result.prediction is None
    assert execution.engine.completed_count == 0
    assert sum(execution.engine.serviced.values()) == 0
