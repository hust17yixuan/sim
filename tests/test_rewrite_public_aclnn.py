"""S-owned public SDK receipt test for the generic T12 native path.

[Role] Prove that ``public.simulate_aclnn_call`` executes the exact selected
  T12 program through ``submit_native_call`` for every executable leaf family,
  without the former AddRmsNorm-only branch and without charging a global
  engine-service delta.
[Invariants] The returned ``actual_execution`` is the real submission receipt:
  its start/terminal events bracket the handle, per-call work comes from the
  handle, and an asynchronous call leaves a real pending event.  Preparation
  only (``execute=False``) performs no submission.
[Side-Effect Policy] Remote CPU-only virtual engine; no devices or I/O.
"""
from __future__ import annotations

import pytest

from mindie_sim.public.api import simulate_aclnn_call
from mindie_sim.prediction.piecewise.native_invocation import _hardware_execution

from test_native_abi_execution_a5 import (
    SKU,
    _fia_request,
    _matrix_bf16_request,
    _matrix_int8_request,
)
from test_native_attnstate_abi_leaves import _gdn_request, _rope_request


def _executable_family_requests():
    return (
        ("matrix_bf16", _matrix_bf16_request()),
        ("matrix_int8", _matrix_int8_request()),
        ("paged_fia", _fia_request()),
        ("gdn", _gdn_request()),
        ("rope", _rope_request()),
    )


@pytest.mark.parametrize("label,invocation", _executable_family_requests())
def test_public_aclnn_executes_each_selected_family_generically(label, invocation):
    payload = simulate_aclnn_call(invocation, hardware=SKU, execute=True, wait=True)
    assert payload["status"] == "predicted"
    assert payload["sdk"]["repriced"] is False
    actual = payload["actual_execution"]
    assert actual["status"] == "completed"
    assert actual["binding_call_id"]
    assert actual["ready_ms"] is not None
    assert actual["terminal_ms"] >= actual["ready_ms"]
    assert actual["duration_ms"] == pytest.approx(
        actual["terminal_ms"] - actual["ready_ms"])
    # Per-call work is the submitted handle's work, not a global delta.
    assert "serviced" not in actual["charge"]
    assert actual["charge"]["work"] == actual["handle"].work
    assert actual["charge"]["service_accounting"] == "submission_receipt"
    assert actual["events"]


def test_public_aclnn_prepare_only_does_not_submit():
    payload = simulate_aclnn_call(
        _matrix_bf16_request(), hardware=SKU, execute=False)
    assert payload["status"] == "predicted"
    assert payload["execution_program_available"] is True
    assert "actual_execution" not in payload
    assert payload["execution_handle_available"] is False


def test_public_aclnn_uses_callers_execution_for_blocking_and_async():
    execution = _hardware_execution(SKU)
    blocking = simulate_aclnn_call(
        _matrix_bf16_request(), hardware=SKU, execution=execution,
        execute=True, wait=True)
    actual = blocking["actual_execution"]
    assert actual["execution"]["advance_calls"] >= 1
    assert execution.engine.now == pytest.approx(actual["terminal_ms"])
    assert actual["handle"].done

    pending = simulate_aclnn_call(
        _rope_request(), hardware=SKU, execution=execution,
        execute=True, wait=False)
    pending_actual = pending["actual_execution"]
    assert pending_actual["status"] == "submitted"
    assert pending_actual["duration_ms"] is None
    assert pending_actual["handle"].done is False
    assert pending_actual["handle"].event.time is None
    execution.engine.advance(until=(pending_actual["handle"].event,))
    assert pending_actual["handle"].event.done is True


def test_public_aclnn_input_ready_is_the_submission_start_boundary():
    payload = simulate_aclnn_call(
        _matrix_bf16_request(), hardware=SKU, execute=True, wait=True,
        input_ready_ms=5.0)
    actual = payload["actual_execution"]
    assert actual["ready_ms"] == pytest.approx(5.0)
    assert actual["terminal_ms"] >= 5.0
    assert actual["duration_ms"] == pytest.approx(
        actual["terminal_ms"] - 5.0)


def test_public_aclnn_rejects_negative_input_ready():
    from mindie_sim.public.errors import ExtensionError
    with pytest.raises(ExtensionError):
        simulate_aclnn_call(
            _matrix_bf16_request(), hardware=SKU, input_ready_ms=-1.0)


def test_public_aclnn_forwards_explicit_selection_to_the_actual_program():
    for tile in ([16, 32, 16], [16, 16, 16]):
        selection = {"program": {"options": {"tile": tile, "slots": 2}}}
        payload = simulate_aclnn_call(
            _matrix_bf16_request(), hardware=SKU, execute=True, wait=True,
            selection=selection)
        assert payload["program"]["selection"] == selection
        assert payload["actual_execution"]["status"] == "completed"
        assert payload["actual_execution"]["handle"].done is True
        assert payload["actual_execution"]["charge"]["work"]


def test_public_aclnn_preserves_nm_semantic_provenance():
    from test_native_abi_invocation_a5 import _request as _add_request

    invocation = _add_request(variant="fused_attention_norm")
    invocation["conditions"] = {
        "variant": "fused_attention_norm",
        "members": ["residual_add", "rmsnorm"],
    }
    payload = simulate_aclnn_call(
        invocation, hardware=SKU, execute=False)
    assert payload["status"] == "predicted"
    assert payload["occurrence"]["members"] == ["residual_add", "rmsnorm"]
    assert payload["program"]["physical"]["physical_members"] == [
        "residual_add", "rmsnorm"]
    assert payload["binding"]["members"]["op_ids"] == [
        "residual_add", "rmsnorm"]
    assert payload["sdk"]["execution_requested"] is False


def test_public_aclnn_async_requires_the_caller_execution():
    from mindie_sim.public.errors import ExtensionError
    with pytest.raises(ExtensionError):
        simulate_aclnn_call(
            _matrix_bf16_request(), hardware=SKU, execute=True, wait=False)
