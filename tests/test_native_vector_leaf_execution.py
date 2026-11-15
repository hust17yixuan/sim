"""Real CPU value contracts and common-stage prediction for native Vector leaves.

[Role] Prove that the independent native Vector/elementwise ABI adapters accept
  real caller tensors, preserve declared output dtype/layout/alias/order, and
  price the same call through the shared piecewise elementwise/copy families.
  AddRmsNorm -> DynamicQuantV2 is exercised as one real value chain.
[Invariants] The reference formulas here are independent of the product code;
  no tensor table is keyed by shape or captured duration, and no device is
  initialized.
"""
from __future__ import annotations

import copy as copy_module
from fractions import Fraction

import numpy as np
import pytest

from mindie_sim.prediction.piecewise import (
    native_add_rms_norm as add,
    native_gated_layer_norm_npu as gated,
    native_r31_aclnn_index as index,
    native_r31_dynamic_quant_v2 as quant,
    native_r31_inplace_copy as copy,
    native_r31_inplace_zero as zero,
    native_r31_sigmoid as sigmoid,
    native_rms_norm as rms,
    native_r64_add_rms_norm_bias as bias,
    native_swiglu as swiglu,
)
from mindie_sim.prediction.piecewise.native_add_rms_norm import (
    NativeVectorExecutionUnavailable,
)


def _bf16(array):
    value = np.ascontiguousarray(array, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _rstd(value, epsilon):
    return (1.0 / np.sqrt(np.mean(value.astype(np.float32) ** 2,
                                   axis=-1, keepdims=True)
                          + epsilon)).astype(np.float32)


def _normal(value):
    return Fraction(str(value))


def test_add_rms_norm_executes_all_three_external_outputs():
    rng = np.random.default_rng(7)
    x = _bf16(rng.uniform(-2, 2, (3, 5)))
    residual = _bf16(rng.uniform(-2, 2, (3, 5)))
    gamma = _bf16(rng.uniform(0.5, 2.0, 5))
    epsilon = 1e-5
    prepared = add.prepare_add_rms_norm(rows=3, hidden=5, epsilon=epsilon)
    result = add.execute(prepared, {"x": x, "residual": residual,
                                    "gamma": gamma})
    fused = _bf16(x + residual)
    rstd = _rstd(fused, epsilon)
    expected = _bf16(fused * rstd * gamma.reshape(1, -1))
    np.testing.assert_array_equal(result["outputs"]["residual_out"], fused)
    np.testing.assert_array_equal(result["outputs"]["rstd"], rstd)
    np.testing.assert_array_equal(result["outputs"]["output"], expected)
    assert result["output_order"] == ["output", "rstd", "residual_out"]
    assert result["dtypes"] == {"output": "bf16", "rstd": "fp32",
                                "residual_out": "bf16"}
    assert result["layouts"] == {"output": "ND", "rstd": "ND",
                                 "residual_out": "ND"}
    assert result["aliases"] == {}
    assert result["outputs"]["output"].dtype == np.float32  # BF16 payload
    assert result["device_executed"] is False


def test_add_rms_norm_quant_chain_matches_independent_formula():
    rng = np.random.default_rng(11)
    x = _bf16(rng.uniform(-1, 1, (4, 7)))
    residual = _bf16(rng.uniform(-1, 1, (4, 7)))
    gamma = _bf16(rng.uniform(0.5, 1.5, 7))
    chain = add.execute_add_rms_norm_quant_chain(
        {"x": x, "residual": residual, "gamma": gamma},
        epsilon=1e-6, is_symmetrical=True)
    norm = _bf16(_bf16(x + residual) * _rstd(_bf16(x + residual), 1e-6)
                 * gamma.reshape(1, -1))
    absmax = np.max(np.abs(norm), axis=-1)
    scale = (absmax / 127.0).astype(np.float32)
    inv = np.zeros_like(scale)
    inv[scale > 0] = 1.0 / scale[scale > 0]
    expected_y = np.clip(np.rint(norm * inv[:, None]), -128, 127).astype(np.int8)
    np.testing.assert_array_equal(chain["values"]["norm_output"], norm)
    np.testing.assert_array_equal(chain["values"]["scale"], scale)
    np.testing.assert_array_equal(chain["values"]["int8"], expected_y)
    assert chain["values"]["int8"].dtype == np.int8
    assert chain["aliases"]["quant.x"] == "add_rms_norm.output"
    assert chain["quant"]["materialized"]["offset"] is False
    assert chain["typed_link"]["dtype"] == "bf16"
    assert chain["typed_link"]["shape"] == [4, 7]
    prediction = chain["prediction"]
    add_call = add.predict(add.prepare_add_rms_norm(
        rows=4, hidden=7, epsilon=1e-6))
    quant_call = quant.predict(quant.prepare_dynamic_quant_v2(
        rows=4, width=7, is_symmetrical=True))
    expected_total = (_normal(add_call["duration_ms"])
                      + _normal(quant_call["duration_ms"]))
    assert _normal(prediction["duration_ms"]) == expected_total
    assert prediction["basis"] == "sum_of_common_resource_stage_calls"


def test_dynamic_quant_requires_declared_mode_and_rejects_asymmetric():
    values = {"x": np.ones((2, 4), np.float32)}
    unknown = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=None)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="is_symmetrical_required"):
        quant.execute(unknown, values)
    asymmetric = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=False)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="asymmetric_not_declared"):
        quant.execute(asymmetric, {"x": np.ones((2, 4), np.float32)})


def test_dynamic_quant_zero_row_and_unseen_width():
    prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=9, is_symmetrical=True)
    x = np.zeros((2, 9), np.float32)
    x[0, :3] = np.array([-127.0, 63.5, 12.0], np.float32)
    result = quant.execute(prepared, {"x": x})
    np.testing.assert_array_equal(result["outputs"]["scale"],
                                  np.array([1.0, 0.0], np.float32))
    np.testing.assert_array_equal(result["outputs"]["y"][1],
                                  np.zeros(9, np.int8))
    assert result["outputs"]["y"][0].tolist()[:3] == [-127, 64, 12]
    prediction = quant.predict(prepared)
    assert prediction["program"]["elements"] == 9
    assert "QUANT_ROWS" in prediction["formula"]
    assert _normal(prediction["duration_ms"]) > 0


def test_rms_norm_shifted_and_raw_weight_modes():
    x = _bf16(np.array([[1.0, -2.0, 3.0, 0.5]], np.float32))
    epsilon = 1e-6
    raw = rms.prepare_rms_norm(rows=1, hidden=4, epsilon=epsilon,
                               weight_mode="raw")
    raw_weight = np.array([0.5, 1.0, -0.5, 2.0], np.float32)
    result = rms.execute(raw, {"x": x, "weight": raw_weight})
    rstd = _rstd(x, epsilon)
    expected = _bf16(x * rstd * (raw_weight + 1.0).reshape(1, -1))
    np.testing.assert_array_equal(result["outputs"]["output"], expected)
    shifted = rms.prepare_rms_norm(rows=1, hidden=4, epsilon=epsilon)
    result = rms.execute(shifted, {"x": x,
                                   "weight": (raw_weight + 1.0)})
    np.testing.assert_array_equal(result["outputs"]["output"], expected)


def test_swiglu_and_sigmoid_values_and_out_alias():
    x = _bf16(np.array([[0.0, 1.0, 2.0, -1.0]], np.float32))
    swiglu_prepared = swiglu.prepare_swiglu(rows=1, elements=2)
    result = swiglu.execute(swiglu_prepared, {"input": x})
    gate, up = x[:, :2], x[:, 2:]
    expected = _bf16(gate * (1.0 / (1.0 + np.exp(-gate))) * up)
    np.testing.assert_array_equal(result["outputs"]["output"], expected)

    sigmoid_prepared = sigmoid.prepare_sigmoid(rows=1, width=4)
    destination = np.empty((1, 4), np.float32)
    result = sigmoid.execute(sigmoid_prepared, {"input": x}, out={
        "output": destination})
    np.testing.assert_array_equal(result["outputs"]["output"], destination)
    assert result["outputs"]["output"] is destination
    np.testing.assert_array_equal(
        destination, _bf16(1.0 / (1.0 + np.exp(-x))))
    assert result["aliases"] == {"output": "out"}


def test_inplace_zero_preserves_caller_storage_and_copy_materializes_slice():
    storage = np.ones((2, 2, 3), np.float32)
    prepared = zero.prepare_inplace_zero(tokens=2, kept_heads=2, head_dim=3)
    result = zero.execute(prepared, {"self": storage})
    assert result["outputs"]["self"] is storage
    assert result["outputs"]["output"] is storage
    np.testing.assert_array_equal(storage, np.zeros((2, 2, 3), np.float32))
    assert result["aliases"] == {"self": "input", "output": "self"}

    source = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    prepared = copy.prepare_inplace_copy(
        tokens=2, source_heads=4, kept_heads=2, head_dim=3,
        slice_offset_heads=1)
    destination = np.empty((2, 2, 3), np.float32)
    result = copy.execute(prepared, {"source": source}, out={
        "output": destination})
    assert result["outputs"]["output"] is destination
    np.testing.assert_array_equal(destination, source[:, 1:3, :])


def test_index_gather_and_selected_row_prediction():
    hidden = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    prepared = index.prepare_index(
        producing_rows=5, selected_rows=2, hidden=4)
    result = index.execute(prepared, {
        "hidden_states": hidden,
        "logits_indices": np.array([4, 1], np.int64)})
    np.testing.assert_array_equal(result["outputs"]["sample_hidden_states"],
                                  hidden[[4, 1]])
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="out_of_range"):
        index.execute(prepared, {
            "hidden_states": hidden,
            "logits_indices": np.array([5, 1], np.int64)})
    prediction = index.predict(prepared)
    assert prediction["environment"]["rows"] == 5
    assert "T_SELECTED" in prediction["formula"]


def test_add_rms_norm_bias_golden_rounding_and_scope():
    x1 = _bf16(np.array([[1.0, 2.0, 3.0, 4.0]], np.float32))
    x2 = _bf16(np.array([[0.5, 0.5, 0.5, 0.5]], np.float32))
    gamma = np.array([1.0, 2.0, 0.5, -1.0], np.float32)
    beta = np.array([0.25, -0.5, 1.0, 0.0], np.float32)
    prepared = bias.prepare_add_rms_norm_bias(
        rows=1, hidden=4, epsilon=1e-6, beta_present=True)
    result = bias.execute(prepared, {"x1": x1, "x2": x2, "gamma": gamma,
                                     "beta": beta})
    fused = _bf16(x1 + x2)
    rstd = _rstd(fused, 1e-6)
    mid = _bf16(fused * rstd)
    expected = _bf16(mid * gamma.reshape(1, -1) + beta.reshape(1, -1))
    np.testing.assert_array_equal(result["outputs"]["y"], expected)
    np.testing.assert_array_equal(result["outputs"]["x"], fused)
    np.testing.assert_array_equal(result["outputs"]["rstd"], rstd)
    n0 = bias.predict(prepared)
    o0 = bias.predict(bias.prepare_add_rms_norm_bias(
        rows=1, hidden=4, epsilon=1e-6, beta_present=True,
        scope="o0_final"))
    assert n0["program"]["common_program"].endswith("add_rmsnorm_bias")
    assert o0["program"]["members"] == ["final_norm"]
    assert o0["program"]["beta_present"] is True


def test_gated_norm_both_orders_and_no_silent_default():
    x = _bf16(np.array([[1.0, 2.0, 3.0]], np.float32))
    gate = _bf16(np.array([[0.5, -1.0, 2.0]], np.float32))
    weight = np.ones(3, np.float32)
    activated = gate * (1.0 / (1.0 + np.exp(-gate)))
    for norm_before_gate in (True, False):
        prepared = gated.prepare_gated_norm(
            rows=1, head_dim=3, norm_before_gate=norm_before_gate)
        result = gated.execute(prepared, {"x": x, "weight": weight,
                                          "gate": gate})
        if norm_before_gate:
            rstd = _rstd(x, gated.EPSILON)
            expected = _bf16(x * rstd * activated)
        else:
            gated_x = x * activated
            rstd = _rstd(gated_x, gated.EPSILON)
            expected = _bf16(gated_x * rstd)
        np.testing.assert_array_equal(result["outputs"]["output"], expected)
        np.testing.assert_array_equal(result["outputs"]["rstd"],
                                      rstd.reshape(-1))
    unknown = gated.prepare_gated_norm(
        rows=1, head_dim=3, norm_before_gate=None)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="norm_before_gate_required"):
        gated.execute(unknown, {"x": x, "weight": weight, "gate": gate})


@pytest.mark.parametrize("rows,width", [(1, 3), (7, 17), (33, 129)])
def test_unseen_shapes_use_the_normal_formula(rows, width):
    x = np.linspace(-1.0, 1.0, rows * width, dtype=np.float32).reshape(rows, width)
    gamma = np.ones(width, np.float32)
    add_prepared = add.prepare_add_rms_norm(
        rows=rows, hidden=width, epsilon=1e-6)
    add_result = add.execute(add_prepared, {
        "x": x, "residual": np.zeros_like(x), "gamma": gamma})
    assert add_result["outputs"]["output"].shape == (rows, width)
    add_prediction = add.predict(add_prepared)
    assert _normal(add_prediction["duration_ms"]) > 0
    assert "ROWS" not in add_prediction["formula"]
    assert "elements" in add_prediction["program"]

    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=rows, width=width, is_symmetrical=True)
    quant_result = quant.execute(quant_prepared, {
        "x": add_result["outputs"]["output"]})
    assert quant_result["outputs"]["y"].shape == (rows, width)
    quant_prediction = quant.predict(quant_prepared)
    assert _normal(quant_prediction["duration_ms"]) > 0
    assert quant_prediction["program"]["elements"] == width
    assert all(item["environment"]["basis"] == "common_resource_stage"
               for item in (add_prediction, quant_prediction))
    assert all("no_target_duration_injected" in item["non_claims"]
               for item in (add_prediction, quant_prediction))


def test_every_leaf_prediction_declares_common_stage_and_no_fallback():
    prepared = [
        (add, add.prepare_add_rms_norm(rows=4, hidden=8, epsilon=1e-6)),
        (rms, rms.prepare_rms_norm(rows=4, hidden=8, epsilon=1e-6)),
        (swiglu, swiglu.prepare_swiglu(rows=4, elements=8)),
        (sigmoid, sigmoid.prepare_sigmoid(rows=4, width=8)),
        (quant, quant.prepare_dynamic_quant_v2(
            rows=4, width=8, is_symmetrical=True)),
        (zero, zero.prepare_inplace_zero(
            tokens=4, kept_heads=2, head_dim=4)),
        (copy, copy.prepare_inplace_copy(
            tokens=4, source_heads=3, kept_heads=2, head_dim=4,
            slice_offset_heads=1)),
        (index, index.prepare_index(
            producing_rows=4, selected_rows=2, hidden=8)),
        (bias, bias.prepare_add_rms_norm_bias(
            rows=4, hidden=8, epsilon=1e-6, beta_present=False)),
        (gated, gated.prepare_gated_norm(
            rows=4, head_dim=8, norm_before_gate=True)),
    ]
    for module, item in prepared:
        prediction = module.predict(item)
        assert prediction["environment"]["basis"] == "common_resource_stage"
        assert prediction["environment"]["arch"] == "A5"
        assert _normal(prediction["duration_ms"]) > 0
        assert prediction["formula"]
        assert "no_algorithm_fallback" in prediction["non_claims"]
        assert "no_target_duration_injected" in prediction["non_claims"]


@pytest.mark.parametrize("dtype,value_dtype", [
    ("bf16", np.float32), ("fp16", np.float16), ("fp32", np.float32)])
def test_add_rms_norm_preserves_declared_payload_dtype(dtype, value_dtype):
    x = np.array([[1.0, -2.0, 3.0]], np.float32)
    residual = np.array([[0.5, 0.5, 0.5]], np.float32)
    gamma = np.array([1.0, 0.5, -1.0], np.float32)
    prepared = add.prepare_add_rms_norm(
        rows=1, hidden=3, epsilon=1e-6, dtype=dtype)
    result = add.execute(prepared, {"x": x, "residual": residual,
                                    "gamma": gamma})
    fused = x + residual
    if dtype == "bf16":
        fused = _bf16(fused)
    elif dtype == "fp16":
        fused = fused.astype(np.float16)
    rstd = (1.0 / np.sqrt(np.mean(fused.astype(np.float32) ** 2,
                                  axis=-1, keepdims=True)
                          + 1e-6)).astype(np.float32)
    expected = fused.astype(np.float32) * rstd * gamma.reshape(1, -1)
    expected = (_bf16(expected) if dtype == "bf16"
                else expected.astype(value_dtype))
    assert result["dtypes"]["output"] == dtype
    assert result["outputs"]["output"].dtype == value_dtype
    np.testing.assert_array_equal(result["outputs"]["output"], expected)
    assert add.predict(prepared)["environment"]["basis"] == (
        "common_resource_stage")


@pytest.mark.parametrize("dtype,value_dtype", [
    ("fp16", np.float16), ("fp32", np.float32)])
def test_add_rms_norm_bias_preserves_declared_payload_dtype(dtype, value_dtype):
    x1 = np.array([[1.0, 2.0, 3.0]], np.float32)
    x2 = np.array([[0.5, 0.5, 0.5]], np.float32)
    gamma = np.array([1.0, 0.5, -1.0], np.float32)
    prepared = bias.prepare_add_rms_norm_bias(
        rows=1, hidden=3, epsilon=1e-6, dtype=dtype)
    result = bias.execute(prepared, {"x1": x1, "x2": x2, "gamma": gamma})
    fused = x1 + x2
    if dtype == "fp16":
        fused = fused.astype(np.float16)
    rstd = (1.0 / np.sqrt(np.mean(fused.astype(np.float32) ** 2,
                                  axis=-1, keepdims=True)
                          + 1e-6)).astype(np.float32)
    mid = fused.astype(np.float32) * rstd
    if dtype == "fp16":
        mid = mid.astype(np.float16)
    expected = (mid.astype(np.float32) * gamma.reshape(1, -1)).astype(
        value_dtype)
    assert result["dtypes"]["y"] == dtype
    assert result["dtypes"]["rstd"] == "fp32"
    assert result["outputs"]["y"].dtype == value_dtype
    np.testing.assert_array_equal(result["outputs"]["y"], expected)
    np.testing.assert_array_equal(result["outputs"]["x"],
                                  fused.astype(value_dtype))


def test_add_rms_norm_n0_charges_the_real_fp32_rstd_write():
    # 2048x5120 BF16: residual 20971520 + norm 20971520 + FP32 rstd 8192.
    prepared = add.prepare_add_rms_norm(rows=2048, hidden=5120, epsilon=1e-6)
    prediction = add.predict(prepared)
    assert prediction["materialization"]["rstd"] == "8192"
    assert prediction["work"]["mte3"] == "41951232"
    assert prediction["work"]["hbm"] == "83904512"
    assert prediction["work"]["mte2"] == "41953280"
    assert prediction["work"]["vector"] == "73564160"
    assert prediction["derived"]["kept_bytes"] == "41951232"
    assert prediction["derived"]["issued_write_bytes"] == "41951232"
    assert prediction["program"]["transactions"]["total_write_bytes"] == (
        41951232)
    assert prediction["program"]["transactions"]["rstd_write_bytes"] == 8192
    assert _normal(prediction["duration_ms"]) > 0


def test_dynamic_quant_prediction_matches_the_real_mode_contract():
    for mode, status, reason in (
            (None, "insufficient_fields",
             "dynamic_quant_is_symmetrical_unresolved"),
            (False, "not_implemented",
             "dynamic_quant_asymmetric_offset_mode_not_modeled")):
        prepared = quant.prepare_dynamic_quant_v2(
            rows=2, width=4, is_symmetrical=mode)
        prediction = quant.predict(prepared)
        assert prediction["status"] == status
        assert prediction["duration_ms"] is None
        assert prediction["duration_us"] is None
        assert prediction["work"] == {}
        assert reason in prediction["reasons"]
        with pytest.raises(NativeVectorExecutionUnavailable):
            quant.execute(prepared, {"x": np.ones((2, 4), np.float32)})

    symmetric = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    prediction = quant.predict(symmetric)
    assert prediction["duration_ms"] is not None
    assert "dynamic_quant_is_symmetrical" not in " ".join(
        prediction.get("reasons", ()))
    assert symmetric["attrs"]["is_symmetrical"] is True


def test_chain_rejects_incompatible_typed_ports_and_reports_mode_gap():
    add_prepared = add.prepare_add_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=3, width=7, is_symmetrical=True)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_typed_mismatch:shape"):
        add.predict_add_rms_norm_quant_chain(add_prepared, quant_prepared)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_shape_invalid"):
        add.execute_add_rms_norm_quant_chain(
            {"x": np.ones((2, 4), np.float32),
             "residual": np.ones((3, 7), np.float32),
             "gamma": np.ones(4, np.float32)}, epsilon=1e-6)

    fp16_quant = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True, source_dtype="fp16")
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_typed_mismatch:dtype"):
        add.predict_add_rms_norm_quant_chain(add_prepared, fp16_quant)

    asymmetric = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=False)
    gap = add.predict_add_rms_norm_quant_chain(add_prepared, asymmetric)
    assert gap["status"] == "not_implemented"
    assert gap["duration_ms"] is None
    assert gap["typed_link"]["dtype"] == "bf16"
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_requires_symmetric_quant"):
        add.execute_add_rms_norm_quant_chain(
            {"x": np.ones((2, 4), np.float32),
             "residual": np.ones((2, 4), np.float32),
             "gamma": np.ones(4, np.float32)},
            epsilon=1e-6, is_symmetrical=False)


def test_inplace_zero_rejects_read_only_or_copied_storage():
    prepared = zero.prepare_inplace_zero(tokens=2, kept_heads=2, head_dim=2)
    readonly = np.ones((2, 2, 2), np.float32)
    readonly.setflags(write=False)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="inplace_zero_self_read_only"):
        zero.execute(prepared, {"self": readonly})
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="inplace_zero_requires_writable_ndarray"):
        zero.execute(prepared, {"self": [[[1.0, 1.0], [1.0, 1.0]],
                                         [[1.0, 1.0], [1.0, 1.0]]]})
    writable = np.ones((2, 2, 2), np.float32)
    other = np.empty_like(writable)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="out_must_alias_self"):
        zero.execute(prepared, {"self": writable}, out={"output": other})


def test_index_rejects_fractional_or_float_indices_without_truncation():
    prepared = index.prepare_index(
        producing_rows=5, selected_rows=2, hidden=3)
    hidden = np.arange(15, dtype=np.float32).reshape(5, 3)
    for bad in (np.array([1.9, 1.0]), np.array([1.0, 2.0])):
        with pytest.raises(NativeVectorExecutionUnavailable,
                           match="index_dtype_must_be_integer"):
            index.execute(prepared, {"hidden_states": hidden,
                                     "logits_indices": bad})
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="index_value_exceeds_int64"):
        index.execute(prepared, {
            "hidden_states": hidden,
            "logits_indices": np.array([2 ** 63, 0], np.uint64)})
    result = index.execute(prepared, {
        "hidden_states": hidden,
        "logits_indices": np.array([1, 4], np.int32)})
    np.testing.assert_array_equal(result["outputs"]["sample_hidden_states"],
                                  hidden[[1, 4]])


def test_add_rms_norm_caller_variants_share_one_physical_program():
    fused = add.predict(add.prepare_add_rms_norm(
        rows=2048, hidden=5120, epsilon=1e-6,
        variant="fused_attention_norm"))
    terminal = add.predict(add.prepare_add_rms_norm(
        rows=2048, hidden=5120, epsilon=1e-6, variant="terminal_norm"))
    assert fused["work"] == terminal["work"] == {
        "hbm": "83904512", "mte2": "41953280",
        "mte3": "41951232", "vector": "73564160",
    }
    assert fused["duration_ms"] == terminal["duration_ms"]
    assert fused["materialization"] == terminal["materialization"] == {
        "norm_output": "20971520", "residual": "20971520",
        "rstd": "8192",
    }
    assert fused["program"]["members"] == ["residual_add", "rmsnorm"]
    assert fused["program"]["fusion"]["contract"] == "add_rmsnorm"
    assert terminal["program"]["members"] == ["final_norm"]
    assert terminal["program"]["fusion"] is None
    assert terminal["program"]["physical_members"] == [
        "residual_add", "rmsnorm"]
    assert fused["program"]["transactions"] == terminal["program"][
        "transactions"] == {
        "residual_write_bytes": 20971520,
        "norm_output_write_bytes": 20971520,
        "rstd_write_bytes": 8192,
        "total_write_bytes": 41951232,
        "logical_total_bytes": 41951232,
        "alignment_bytes": 0,
        "mte3_bytes": 41951232,
    }


def test_chain_rejects_wrong_prepare_api_and_missing_typed_ports():
    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    add_prepared = add.prepare_add_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    rms_prepared = rms.prepare_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_producer_api_mismatch"):
        add.predict_add_rms_norm_quant_chain(rms_prepared, quant_prepared)

    index_prepared = index.prepare_index(
        producing_rows=2, selected_rows=1, hidden=4)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_consumer_api_mismatch"):
        add.predict_add_rms_norm_quant_chain(add_prepared, index_prepared)

    missing_residual = copy_module.deepcopy(add_prepared)
    missing_residual["descriptors"].pop("residual")
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_producer_typed_port_missing:residual"):
        add.predict_add_rms_norm_quant_chain(
            missing_residual, quant_prepared)

    missing_offset = copy_module.deepcopy(quant_prepared)
    missing_offset["descriptors"].pop("offset")
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="chain_consumer_typed_port_missing:offset"):
        add.predict_add_rms_norm_quant_chain(add_prepared, missing_offset)


def test_symmetric_quant_rejects_offset_out_before_any_write():
    prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    y_out = np.full((2, 4), -5, dtype=np.int8)
    scale_out = np.array([11.0, 12.0], np.float32)
    offset_out = np.array([7.0, 9.0], np.float32)
    with pytest.raises(NativeVectorExecutionUnavailable,
                       match="symmetric_offset_out_not_writable"):
        quant.execute(prepared, {"x": np.ones((2, 4), np.float32)},
                      out={"y": y_out, "scale": scale_out,
                           "offset": offset_out})
    np.testing.assert_array_equal(y_out, np.full((2, 4), -5, np.int8))
    np.testing.assert_array_equal(scale_out, np.array([11.0, 12.0],
                                                      np.float32))
    np.testing.assert_array_equal(offset_out, np.array([7.0, 9.0],
                                                       np.float32))
    prediction = quant.predict(prepared)
    assert ("offset_gm_bytes", 0) in prediction["program"]["extras"]
    assert prepared["attrs"]["offset_materialized"] is False


def _abi_request(rows=2, hidden=4, variant=None):
    conditions = {"sequence_rows": "", "sequence_ranks": 1}
    if variant is not None:
        conditions["variant"] = variant
    return {
        "api": {"namespace": "aclnn", "entry": "aclnnAddRmsNorm",
                "aclnn_symbol": "aclnnAddRmsNorm"},
        "version": {"api": "cann-9.2.0"},
        "tensors": {
            "x": {"shape": [rows, hidden], "dtype": "bf16", "layout": "ND",
                  "role": "input"},
            "residual": {"shape": [rows, hidden], "dtype": "bf16",
                         "layout": "ND", "role": "input"},
            "gamma": {"shape": [hidden], "dtype": "bf16", "layout": "ND",
                      "role": "weight"},
            "output": {"shape": [rows, hidden], "dtype": "bf16",
                       "layout": "ND", "role": "output"},
            "rstd": {"shape": [rows, 1], "dtype": "fp32", "layout": "ND",
                     "role": "output"},
            "residual_out": {"shape": [rows, hidden], "dtype": "bf16",
                             "layout": "ND", "role": "output"},
        },
        "attrs": {"epsilon": 1e-6},
        "conditions": conditions,
    }


def test_add_rms_norm_2x4_separates_logical_values_and_issued_writes():
    prepared = add.prepare_add_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    prediction = add.predict(prepared)
    assert prediction["materialization"] == {
        "norm_output": "16", "residual": "16", "rstd": "8"}
    assert prediction["derived"]["kept_bytes"] == "40"
    assert prediction["derived"]["logical_effective_bytes"] == "40"
    assert prediction["derived"]["issued_write_bytes"] == "136"
    assert prediction["derived"]["write_alignment_bytes"] == "96"
    assert prediction["work"]["mte3"] == "136"
    assert prediction["program"]["logical_materialization"] == {
        "norm_output": 16, "residual": 16, "rstd": 8}
    assert prediction["program"]["transactions"] == {
        "residual_write_bytes": 64,
        "norm_output_write_bytes": 64,
        "rstd_write_bytes": 8,
        "total_write_bytes": 136,
        "logical_total_bytes": 40,
        "alignment_bytes": 96,
        "mte3_bytes": 136,
    }
    ledger = add.add_rms_norm_write_ledger(prepared)
    assert ledger["logical_materialization"] == {
        "norm_output": 16, "residual": 16, "rstd": 8}
    assert ledger["logical_total_bytes"] == 40
    assert ledger["issued_transactions"]["total_write_bytes"] == 136
    assert ledger["issued_mte3_bytes"] == 136
    assert add.predict(prepared)["materialization"] == prediction[
        "materialization"]


def test_add_rms_norm_2x4_chain_and_t12_public_records_use_same_ledger():
    values = {"x": np.ones((2, 4), np.float32),
              "residual": np.ones((2, 4), np.float32),
              "gamma": np.ones(4, np.float32)}
    chain = add.execute_add_rms_norm_quant_chain(
        values, epsilon=1e-6, is_symmetrical=True)
    first = chain["prediction"]["calls"][0]
    assert first["materialization"] == {
        "norm_output": "16", "residual": "16", "rstd": "8"}
    assert first["work"]["mte3"] == "136"
    assert first["program"]["transactions"]["total_write_bytes"] == 136
    standalone = chain["prediction"]["calls"][0]
    prepared = add.prepare_add_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    direct = add.predict(prepared)
    assert standalone["materialization"] == direct["materialization"]
    assert standalone["program"]["transactions"] == direct["program"][
        "transactions"]
    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    chain_prediction = add.predict_add_rms_norm_quant_chain(
        prepared, quant_prepared)
    chain_first = chain_prediction["calls"][0]
    assert chain_first["materialization"] == direct["materialization"]
    assert chain_first["program"]["transactions"] == direct["program"][
        "transactions"]
    assert chain_first["work"]["mte3"] == "136"

    from mindie_sim.prediction.piecewise.native_invocation import (
        simulate_aclnn_call,
    )
    for variant in (None, "fused_attention_norm", "terminal_norm"):
        result = simulate_aclnn_call(
            _abi_request(variant=variant), hardware="Ascend950DT_9582")
        assert result.status == "predicted", result.reasons
        assert result.prediction["materialization"] == {
            "norm_output": "16", "residual": "16", "rstd": "8"}
        assert result.prediction["work"]["mte3"] == "136"
        assert result.program["physical"]["materialization"] == {
            "norm_output": "16", "residual": "16", "rstd": "8"}
        assert result.program["cross_view"]["work"]["mte3"] == "136"


def _execution(*, memory=1_000_000.0, vector=1_000_000.0,
               mte2=1_000_000.0, mte3=1_000_000.0):
    from test_mechanism_contract import hardware_case
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    raw = hardware_case()
    raw["machine"]["device"]["rates"].update({
        "memory": float(memory), "vector": float(vector),
        "mte2": float(mte2), "mte3": float(mte3)})
    parsed = parse_case(raw)
    return Execution(parsed, CostModel(parsed), include_submission=False)


def _leaf_programs():
    return (
        ("add", add, add.prepare_add_rms_norm(
            rows=2, hidden=4, epsilon=1e-6)),
        ("quant", quant, quant.prepare_dynamic_quant_v2(
            rows=2, width=4, is_symmetrical=True)),
        ("rms", rms, rms.prepare_rms_norm(
            rows=2, hidden=4, epsilon=1e-6)),
        ("swiglu", swiglu, swiglu.prepare_swiglu(rows=2, elements=4)),
        ("sigmoid", sigmoid, sigmoid.prepare_sigmoid(rows=2, width=4)),
        ("copy", copy, copy.prepare_inplace_copy(
            tokens=2, source_heads=3, kept_heads=2, head_dim=2,
            slice_offset_heads=0)),
        ("zero", zero, zero.prepare_inplace_zero(
            tokens=2, kept_heads=2, head_dim=2)),
        ("index", index, index.prepare_index(
            producing_rows=2, selected_rows=1, hidden=4)),
        ("bias", bias, bias.prepare_add_rms_norm_bias(
            rows=2, hidden=4, epsilon=1e-6, beta_present=False)),
        ("gated", gated, gated.prepare_gated_norm(
            rows=2, head_dim=4, norm_before_gate=True)),
    )


def test_every_leaf_execution_program_submits_to_the_caller_execution():
    for label, module, prepared in _leaf_programs():
        execution = _execution()
        prediction = module.predict(prepared)
        handoff = module.execution_program(prepared)
        assert handoff.native_program_available is False
        assert handoff.to_dict()["program_source"] == "stage_work"
        assert [dict(stage) for stage in handoff.stage_work] == [
            dict(stage) for stage in prediction["stage_work"]]
        program_work = {}
        for fragment in handoff.fragments:
            for item in fragment.work.issued():
                program_work[item.resource] = (
                    program_work.get(item.resource, 0.0) + item.amount)
        for pipe, amount in prediction["work"].items():
            if float(amount) == 0:
                continue
            assert program_work[pipe] == pytest.approx(float(amount))
        ready = execution.engine.submit(f"{label}:ready")
        event = handoff.start(execution, name=label, ready=ready)
        execution.engine.advance(until=(event.event,))
        assert event.done
        names = execution.stage_names[(0, 0)]
        for pipe, amount in prediction["work"].items():
            if float(amount) == 0:
                continue
            assert execution.engine.serviced[names[pipe]] == pytest.approx(
                float(amount))
        assert any(value > 0 for value in execution.engine.busy_ms.values())


def test_same_context_add_quant_chain_shares_hbm_and_releases_last_reader():
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        AllocationEffect, ProgramFragment, WorkItem, WorkSet,
    )
    from mindie_sim.schema.subgraphs import PortSet

    def run(*, contested):
        execution = _execution(memory=100.0)
        execution.engine.extend(pools={"reader_pool": 16})
        ready = execution.engine.submit("ready")
        competitor = None
        if contested:
            fragment = ProgramFragment(
                "hbm_competitor",
                occurrence=Occurrence("kernel", "hbm_competitor"),
                ports=PortSet(),
                work=WorkSet((WorkItem("hbm", 1000.0, "bytes"),)))
            competitor = execution.fragment_conductor(
                replica=0, stage=0).submit(
                    fragment, after=(ready,), owner="competitor").event
        add_prepared = add.prepare_add_rms_norm(
            rows=2, hidden=4, epsilon=1e-6)
        quant_prepared = quant.prepare_dynamic_quant_v2(
            rows=2, width=4, is_symmetrical=True)
        add_program = add.execution_program(add_prepared)
        quant_program = quant.execution_program(quant_prepared)
        add_event = add_program.start(execution, name="add", ready=ready)
        quant_event = quant_program.start(
            execution, name="quant", ready=add_event,
            inputs={"x": add_event})
        reader = ProgramFragment(
            "last_reader",
            occurrence=Occurrence("kernel", "last_reader"),
            ports=PortSet(), work=WorkSet(()),
            allocations=(
                AllocationEffect("acquire", "buf", "reader_pool", 16),
                AllocationEffect("release", "buf", "reader_pool", 16)))
        reader_event = execution.fragment_conductor(
            replica=0, stage=0,
            pool_bindings={"reader_pool": "reader_pool"}).submit(
                reader, after=(quant_event.event,), owner="last_reader").event
        targets = tuple(item.event if hasattr(item, "event") else item
                        for item in
                        (competitor, add_event, quant_event, reader_event)
                        if item is not None)
        execution.engine.advance(until=targets, first=False)
        return (execution, competitor, add_event, quant_event, reader_event,
                add_program, quant_program)

    (control, _c, control_add, control_quant, _r,
     control_add_program, control_quant_program) = run(contested=False)
    (contended, competitor, add_event, quant_event, reader_event,
     add_program, quant_program) = run(contested=True)
    assert competitor.done and reader_event.done
    assert {name for name, _ in quant_event.output_events} == {
        "y", "scale", "offset"}
    assert dict(add_event.output_events)["output"].done
    assert add_event.time > control_add.time
    assert quant_event.time > control_quant.time
    assert reader_event.time == quant_event.time
    assert contended.engine.allocated["reader_pool"] == 0
    assert contended.engine.peak_allocated["reader_pool"] == 16
    names = contended.stage_names[(0, 0)]
    for pipe in ("mte2", "mte3", "vector"):
        assert contended.engine.serviced[names[pipe]] == pytest.approx(
            control.engine.serviced[names[pipe]])
    assert (contended.engine.serviced[names["hbm"]]
            - control.engine.serviced[names["hbm"]]) == pytest.approx(1000.0)
    def program_work(program):
        totals = {}
        for fragment in program.fragments:
            for item in fragment.work.issued():
                totals[item.resource] = totals.get(item.resource, 0.0) + item.amount
        return totals

    expected = {}
    for program in (add_program, quant_program):
        for pipe, amount in program_work(program).items():
            expected[pipe] = expected.get(pipe, 0.0) + amount
    for pipe, amount in expected.items():
        assert control.engine.serviced[names[pipe]] == pytest.approx(amount)
    chain = add.predict_add_rms_norm_quant_chain(
        add.prepare_add_rms_norm(rows=2, hidden=4, epsilon=1e-6),
        quant.prepare_dynamic_quant_v2(
            rows=2, width=4, is_symmetrical=True))
    for name, program in (("add", add_program), ("quant", quant_program)):
        call = chain["calls"][0 if name == "add" else 1]
        assert [dict(stage) for stage in program.stage_work] == [
            dict(stage) for stage in call["stage_work"]]
    assert control_add_program.stage_work == add_program.stage_work
    assert control_quant_program.stage_work == quant_program.stage_work


def test_leaf_execution_programs_carry_typed_ports_dependencies_and_aliases():
    add_prepared = add.prepare_add_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    add_program = add.execution_program(add_prepared)
    assert len(add_program.fragments) == 2
    assert add_program.edges == ((), (0,))
    first_ports = add_program.fragments[0].ports
    second_ports = add_program.fragments[1].ports
    assert {port.name for port in first_ports.tensor_in} == {"x", "residual"}
    assert {port.name for port in first_ports.tensor_out} == {"residual_out"}
    assert {port.name for port in second_ports.tensor_in} == {
        "residual_out", "gamma"}
    assert {port.name: port.direction for port in second_ports.tensor_in}[
        "residual_out"] == "in"
    assert {port.name for port in second_ports.tensor_out} == {
        "output", "rstd"}
    assert add_program.accesses[0] == {
        "stage": 0, "inputs": ["x", "residual"],
        "outputs": ["residual_out"]}
    assert add_program.accesses[1] == {
        "stage": 1, "inputs": ["residual_out", "gamma"],
        "outputs": ["output", "rstd"]}
    assert add_program.to_dict()["program_source"] == "stage_work"
    assert add_program.native_program_available is False
    add_buffer_names = {item["name"] for item in add_program.buffers}
    assert {"x", "residual", "gamma", "output", "rstd",
            "residual_out"} <= add_buffer_names
    assert all(item["pool"] == "caller" for item in add_program.buffers)

    zero_prepared = zero.prepare_inplace_zero(
        tokens=2, kept_heads=2, head_dim=2)
    zero_program = zero.execution_program(zero_prepared)
    assert dict(zero_program.aliases)["output"] == "self"
    zero_values = dict(zero_program.to_dict()["typed_values"])
    assert zero_values["output"]["alias_of"] == "self"
    assert zero_values["self"]["binding"] == "external"
    assert {item["name"] for item in zero_program.to_dict()["buffers"]} == {
        "self", "output"}

    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    quant_program = quant.execution_program(quant_prepared)
    ports = quant_program.fragments[0].ports
    assert {port.name for port in ports.tensor_in} == {"x"}
    assert {port.name for port in ports.tensor_out} == {
        "y", "scale", "offset"}


def test_inplace_copy_program_preserves_actual_view_offset_and_storage():
    source = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    for offset, expected in (
            (0, [0, 1, 2, 3, 6, 7, 8, 9]),
            (1, [2, 3, 4, 5, 8, 9, 10, 11])):
        prepared = copy.prepare_inplace_copy(
            tokens=2, source_heads=3, kept_heads=2, head_dim=2,
            slice_offset_heads=offset)
        program = copy.execution_program(prepared)
        values = dict(program.typed_values)
        assert values["source"].producer == "A3:gemm:qkvz_projection"
        layout = values["source"].layout
        assert layout.shape == (2, 2, 2)
        assert layout.strides == (6, 2, 1)
        assert layout.offset == 2 * offset
        assert layout.storage_shape == (2, 3, 2)
        source_buffer = next(
            item for item in program.buffers if item["name"] == "source")
        assert source_buffer["bytes"] == 24
        assert source_buffer["read_bytes"] == 16
        assert source_buffer["allocation"] == 0
        output_buffer = next(
            item for item in program.buffers if item["name"] == "output")
        assert output_buffer["allocation"] == 16
        destination = np.empty((2, 2, 2), np.float32)
        result = copy.execute(prepared, {"source": source}, out={
            "output": destination})
        np.testing.assert_array_equal(result["outputs"]["output"], source.reshape(-1)[
            expected].reshape(2, 2, 2))
        assert program.to_dict()["typed_values"]["source"]["layout"][
            "offset"] == 2 * offset


def test_inplace_zero_program_keeps_one_storage_and_no_new_allocation():
    prepared = zero.prepare_inplace_zero(
        tokens=2, kept_heads=2, head_dim=2)
    program = zero.execution_program(prepared)
    assert dict(program.aliases) == {"output": "self"}
    values = dict(program.typed_values)
    assert values["self"].storage == "core_attn_out"
    assert values["output"].storage == "core_attn_out"
    assert values["output"].alias_of == "self"
    assert values["self"].alias_of == ""
    buffers = {item["name"]: item for item in program.buffers}
    assert buffers["self"]["root"] == "core_attn_out"
    assert buffers["self"]["bytes"] == 16
    assert buffers["self"]["allocation"] == 0
    assert buffers["output"]["root"] == "core_attn_out"
    assert buffers["output"]["bytes"] == 0
    assert buffers["output"]["allocation"] == 0
    assert sum(item["allocation"] for item in program.buffers) == 0
    execution = _execution()
    handle = program.start(execution, name="zero")
    execution.engine.advance(until=(handle.event,))
    outputs = dict(handle.output_events)
    assert outputs["self"] is outputs["output"]


def test_quant_program_keeps_logical_offset_slot_unmaterialized():
    prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    program = quant.execution_program(prepared)
    assert program.unmaterialized == ("offset",)
    values = dict(program.typed_values)
    assert values["offset"].storage == ""
    assert "offset" not in {item["name"] for item in program.buffers}
    buffers = {item["name"]: item for item in program.buffers}
    assert buffers["y"]["bytes"] == 8
    assert buffers["scale"]["bytes"] == 8
    assert sum(item["allocation"] for item in program.buffers) == 16


def test_add_rms_norm_bias_n0_ledger_closes_rstd_and_scopes():
    for beta_present in (False, True):
        prepared = bias.prepare_add_rms_norm_bias(
            rows=2, hidden=4, epsilon=1e-6,
            beta_present=beta_present, scope="n0")
        prediction = bias.predict(prepared)
        assert prediction["work"] == ({
            "hbm": "216", "mte2": "80", "mte3": "136",
            "vector": "8192"} if beta_present else {
            "hbm": "208", "mte2": "72", "mte3": "136",
            "vector": "8192"})
        assert [dict(stage) for stage in prediction["stage_work"]] == [
            {"hbm": "128", "mte2": "64", "mte3": "64", "vector": "4096"},
            ({"hbm": "88", "mte2": "16", "mte3": "72", "vector": "4096"}
             if beta_present else
             {"hbm": "80", "mte2": "8", "mte3": "72", "vector": "4096"}),
        ]
        assert prediction["materialization"] == {
            "y": "16", "x": "16", "rstd": "8"}
        assert prediction["derived"]["kept_bytes"] == "40"
        transactions = prediction["program"]["transactions"]
        assert transactions["y_write_bytes"] == 64
        assert transactions["x_write_bytes"] == 64
        assert transactions["rstd_write_bytes"] == 8
        assert transactions["total_write_bytes"] == 136
        assert prediction["program"]["rstd_write_added_bytes"] == 8
        program = bias.execution_program(prepared)
        assert len(program.fragments) == 2
        assert {port.name for port in program.fragments[1].ports.tensor_out} == {
            "y", "rstd"}
        assert program.accesses[0]["outputs"] == ["x"]
        assert program.accesses[1]["inputs"] == (
            ["gamma", "beta", "x"] if beta_present else ["gamma", "x"])

    for scope in ("n0", "o0_final"):
        same = bias.predict(bias.prepare_add_rms_norm_bias(
            rows=2, hidden=4, epsilon=1e-6, beta_present=False,
            scope=scope))
        assert same["work"] == {
            "hbm": "208", "mte2": "72", "mte3": "136",
            "vector": "8192"}
        assert same["materialization"] == {
            "y": "16", "x": "16", "rstd": "8"}
        assert len(same["stage_work"]) == 2
        assert same["program"]["caller_scope"] == scope
        assert same["program"]["physical_members"] == [
            "residual_add", "rmsnorm"]
        assert same["program"]["caller_members"] == (
            ["final_norm"] if scope == "o0_final"
            else ["residual_add", "rmsnorm"])
        same_program = bias.execution_program(bias.prepare_add_rms_norm_bias(
            rows=2, hidden=4, epsilon=1e-6, beta_present=False,
            scope=scope))
        assert len(same_program.fragments) == 2
        assert same_program.accesses[0]["outputs"] == ["x"]
        assert same_program.accesses[1]["outputs"] == ["y", "rstd"]
        with_beta = bias.predict(bias.prepare_add_rms_norm_bias(
            rows=2, hidden=4, epsilon=1e-6, beta_present=True,
            scope=scope))
        assert with_beta["work"]["mte3"] == same["work"]["mte3"] == "136"
        assert with_beta["work"]["hbm"] == "216"
        assert with_beta["work"]["mte2"] == "80"
        assert with_beta["stage_work"][1]["hbm"] == "88"
        assert with_beta["stage_work"][1]["mte2"] == "16"
        assert with_beta["materialization"] == same["materialization"]
        assert with_beta["program"]["transactions"][
            "rstd_write_bytes"] == 8

    for scope in ("n0", "o0_final"):
        ragged = bias.prepare_add_rms_norm_bias(
            rows=3, hidden=7, epsilon=1e-6, beta_present=False,
            scope=scope)
        ragged_prediction = bias.predict(ragged)
        assert ragged_prediction["work"]["mte3"] == "204"
        assert ragged_prediction["work"]["hbm"] == "314"
        assert ragged_prediction["materialization"] == {
            "y": "42", "x": "42", "rstd": "12"}
        assert len(ragged_prediction["stage_work"]) == 2
        assert ragged_prediction["program"]["transactions"][
            "rstd_write_bytes"] == 12


def test_async_reader_holds_real_norm_grant_until_release():
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        AllocationEffect, ProgramFragment, WorkItem, WorkSet,
    )
    from mindie_sim.schema.subgraphs import PortSet, TensorPortSpec

    execution = _execution(memory=100.0)
    execution.engine.extend(pools={"norm_pool": 16})
    ready = execution.engine.submit("ready")
    hold = execution.engine.future("async_reader_done")
    add_prepared = add.prepare_add_rms_norm(
        rows=2, hidden=4, epsilon=1e-6)
    add_program = add.execution_program(add_prepared)
    add_handle = add_program.start(
        execution, name="add", ready=ready,
        allocations=(AllocationEffect(
            "acquire", "norm_grant", "norm_pool", 16),),
        holds={"norm_grant": hold},
        pool_bindings={"norm_pool": "norm_pool"})
    assert add_handle.output_events
    assert add_handle.release_events

    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True)
    quant_program = quant.execution_program(quant_prepared)
    quant_handle = quant_program.start(
        execution, name="quant", ready=add_handle,
        inputs={"x": add_handle})
    execution.engine.advance(until=(quant_handle.event,), first=False)
    assert execution.engine.allocated["norm_pool"] == 16
    assert not any(event.done for event in add_handle.release_events)

    reader = ProgramFragment(
        "async_reader",
        occurrence=Occurrence("kernel", "async_reader"),
        ports=PortSet(tensor_in=(TensorPortSpec(
            "norm_output", "in", "activation", dtype="bf16"),)),
        work=WorkSet(()))
    reader_conductor = execution.fragment_conductor(
        replica=0, stage=0,
        pool_bindings={"norm_pool": "norm_pool"})
    reader_handle = reader_conductor.submit(
        reader, after=(quant_handle.event,),
        inputs={"norm_output": add_handle}, owner="async_reader")
    execution.engine.advance(until=(reader_handle.event,), first=False)
    assert execution.engine.allocated["norm_pool"] == 16
    blocked_request = ProgramFragment(
        "blocked_request",
        occurrence=Occurrence("kernel", "blocked_request"),
        ports=PortSet(), work=WorkSet(()),
        allocations=(
            AllocationEffect("acquire", "blocked_buf", "norm_pool", 16),
            AllocationEffect("release", "blocked_buf", "norm_pool", 16)))
    blocked = reader_conductor.submit(
        blocked_request, after=(reader_handle.event,), owner="blocked")
    assert not blocked.event.done
    assert execution.engine.allocated["norm_pool"] == 16
    execution.engine.resolve(hold)
    execution.engine.advance(
        until=tuple(add_handle.release_events) + (blocked.event,),
        first=False)
    assert all(event.done for event in add_handle.release_events)
    assert blocked.event.done
    assert execution.engine.allocated["norm_pool"] == 0

    next_request = ProgramFragment(
        "next_request",
        occurrence=Occurrence("kernel", "next_request"),
        ports=PortSet(),
        work=WorkSet((WorkItem("hbm", 0.0, "bytes"),)),
        allocations=(
            AllocationEffect("acquire", "next_buf", "norm_pool", 16),
            AllocationEffect("release", "next_buf", "norm_pool", 16)))
    next_handle = reader_conductor.submit(
        next_request, after=tuple(add_handle.release_events),
        owner="next_request")
    execution.engine.advance(until=(next_handle.event,), first=False)
    assert next_handle.event.done
    assert execution.engine.allocated["norm_pool"] == 0


def test_one_add_rms_norm_template_binds_two_occurrences():
    from mindie_sim.schema.invocation import Occurrence
    execution = _execution()
    program = add.execution_program(add.prepare_add_rms_norm(
        rows=2, hidden=4, epsilon=1e-6))
    first = program.start(
        execution, name="call_a", occurrence=Occurrence(
            "kernel", "call_a", request_ids=("r1",)))
    second = program.start(
        execution, name="call_b", occurrence=Occurrence(
            "kernel", "call_b", request_ids=("r2",)))
    execution.engine.advance(
        until=(first.event, second.event), first=False)
    assert first.event.name != second.event.name
    assert dict(first.output_events)["output"].name != dict(
        second.output_events)["output"].name
    assert first.handles[0].fragment.occurrence.request_ids == ("r1",)
    assert second.handles[0].fragment.occurrence.request_ids == ("r2",)


def test_rms_and_gated_stats_share_predict_program_and_stage_totals():
    for module, prepared in (
            (rms, rms.prepare_rms_norm(
                rows=2, hidden=4, epsilon=1e-6)),
            (gated, gated.prepare_gated_norm(
                rows=2, head_dim=4, norm_before_gate=True))):
        prediction = module.predict(prepared)
        program = module.execution_program(prepared)
        assert prediction["materialization"]["norm_output"] == "16"
        assert prediction["materialization"]["residual"] == "0"
        assert prediction["materialization"]["rstd"] == "8"
        assert prediction["derived"]["kept_bytes"] == "24"
        assert prediction["work"]["mte3"] == "64"
        buffers = {item["name"]: item for item in program.buffers}
        assert buffers["output"]["allocation"] == 16
        assert buffers["rstd"]["allocation"] == 8
        totals = {}
        for fragment in program.fragments:
            for item in fragment.work.issued():
                totals[item.resource] = totals.get(item.resource, 0.0) + item.amount
        for pipe, amount in prediction["work"].items():
            assert totals[pipe] == pytest.approx(float(amount))


def test_first_output_writer_waits_for_occupied_grant_and_holds_it():
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        AllocationEffect, ProgramFragment, WorkItem, WorkSet,
    )
    from mindie_sim.schema.subgraphs import PortSet, TensorPortSpec

    execution = _execution(memory=100.0)
    execution.engine.extend(pools={"norm_pool": 16})
    ready = execution.engine.submit("ready")
    holder_hold = execution.engine.future("holder_release")
    holder = ProgramFragment(
        "holder", occurrence=Occurrence("kernel", "holder"),
        ports=PortSet(), work=WorkSet(()),
        allocations=(AllocationEffect(
            "acquire", "holder", "norm_pool", 16),))
    holder_conductor = execution.fragment_conductor(
        replica=0, stage=0,
        pool_bindings={"norm_pool": "norm_pool"})
    holder_handle = holder_conductor.submit(
        holder, after=(ready,), holds={"holder": holder_hold},
        owner="holder")
    execution.engine.advance(until=(holder_handle.event,), first=False)
    assert execution.engine.allocated["norm_pool"] == 16

    reader_done = execution.engine.future("residual_reader_done")
    add_program = add.execution_program(add.prepare_add_rms_norm(
        rows=2, hidden=4, epsilon=1e-6))
    add_handle = add_program.start(
        execution, name="add", ready=ready,
        output_grants={"residual_out": {
            "pool": "norm_pool", "size": 16,
            "name": "residual_grant", "last_reader": reader_done}},
        pool_bindings={"norm_pool": "norm_pool"})
    stage0 = add_handle.handles[0].event
    stage1 = add_handle.handles[1].event
    assert not stage0.done
    assert not dict(add_handle.output_events)["residual_out"].done
    assert dict(add_handle.output_values)["residual_out"].producer == (
        dict(add_handle.output_events)["residual_out"].name)
    assert execution.engine.serviced["memory@r0:s0"] == 0
    assert execution.engine.serviced["memory@r0:s0"] == 0
    assert execution.engine.serviced["mte2@r0:s0"] == 0
    assert execution.engine.serviced["mte3@r0:s0"] == 0
    assert execution.engine.serviced["vector@r0:s0"] == 0

    execution.engine.resolve(holder_hold)
    execution.engine.advance(
        until=tuple(holder_handle.release_events), first=False)
    execution.engine.advance(until=(stage0,), first=False)
    assert stage0.done
    assert execution.engine.allocated["norm_pool"] == 16
    execution.engine.advance(until=(stage1,), first=False)
    assert stage1.done
    assert execution.engine.allocated["norm_pool"] == 16

    reader = ProgramFragment(
        "residual_reader",
        occurrence=Occurrence("kernel", "residual_reader"),
        ports=PortSet(tensor_in=(TensorPortSpec(
            "residual_out", "in", "activation", dtype="bf16"),)),
        work=WorkSet((
            WorkItem("hbm", 8.0, "bytes"),
            WorkItem("vector", 8.0, "ops"))))
    reader_handle = holder_conductor.submit(
        reader, after=(stage1,),
        inputs={"residual_out": add_handle}, owner="residual_reader")
    execution.engine.advance(until=(reader_handle.event,), first=False)
    assert reader_handle.event.done
    assert execution.engine.allocated["norm_pool"] == 16
    assert not any(event.done for event in add_handle.release_events)
    execution.engine.resolve(reader_done)
    execution.engine.advance(
        until=tuple(add_handle.release_events), first=False)
    assert execution.engine.allocated["norm_pool"] == 0


def test_caller_owned_zero_uses_two_generations_without_allocation():
    from mindie_sim.schema.invocation import Occurrence
    execution = _execution()
    program = zero.execution_program(zero.prepare_inplace_zero(
        tokens=2, kept_heads=2, head_dim=2))
    first = program.start(
        execution, name="zero_a",
        occurrence=Occurrence("kernel", "zero", generation=0),
        output_grants={"output": {"borrow": True}})
    second = program.start(
        execution, name="zero_b",
        occurrence=Occurrence("kernel", "zero", generation=1),
        output_grants={"output": {"borrow": True}})
    execution.engine.advance(
        until=(first.event, second.event), first=False)
    assert first.event.name != second.event.name
    assert first.handles[0].fragment.occurrence.generation == 0
    assert second.handles[0].fragment.occurrence.generation == 1
    assert first.handles[0].fragment.allocations == ()
    assert second.handles[0].fragment.allocations == ()
    assert sum(execution.engine.allocated.values()) == 0


def test_swiglu_selected_axis_preserves_original_output_mapping():
    source = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    prepared = swiglu.prepare_swiglu(
        input_shape=[4, 32], output_shape=[2, 32], dim=0)
    prediction = swiglu.predict(prepared)
    assert prediction["work"] == {
        "hbm": "384", "mte2": "256", "mte3": "128",
        "vector": "4096"}
    assert prediction["materialization"] == {"swiglu_output": "128"}
    view = prepared["attrs"]["execution_view"]
    assert view["axis"] == 0
    assert view["gate"] == {"axis": 0, "start": 0, "stop": 2}
    assert view["up"] == {"axis": 0, "start": 2, "stop": 4}
    program = swiglu.execution_program(prepared)
    values = dict(program.typed_values)
    assert values["input"].layout.shape == (4, 32)
    assert values["output"].layout.shape == (2, 32)
    buffers = {item["name"]: item for item in program.buffers}
    assert buffers["input"]["bytes"] == 256
    assert buffers["output"]["bytes"] == 128
    result = swiglu.execute(prepared, {"input": source})
    expected = _bf16(np.stack((
        source[0] / (1.0 + np.exp(-source[0])) * source[2],
        source[1] / (1.0 + np.exp(-source[1])) * source[3])))
    np.testing.assert_allclose(result["outputs"]["output"], expected, rtol=0,
                               atol=0)

    last = swiglu.prepare_swiglu(
        input_shape=[4, 32], output_shape=[4, 16], dim=-1)
    last_prediction = swiglu.predict(last)
    assert last_prediction["work"]["mte2"] == "256"
    assert last_prediction["work"]["mte3"] == "128"
    total = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    last_result = swiglu.execute(last, {"input": total})
    expected_last = _bf16(
        total[:, :16] /
        (1.0 + np.exp(-total[:, :16])) * total[:, 16:])
    np.testing.assert_allclose(last_result["outputs"]["output"],
                               expected_last, rtol=0, atol=0)

    with pytest.raises(ValueError, match="swiglu_output_shape_mismatch"):
        swiglu.prepare_swiglu(
            input_shape=[4, 32], output_shape=[4, 16], dim=0)


def test_rms_norm_keeps_original_rstd_leading_shape_and_flat_mapping():
    source = np.arange(4 * 3 * 16, dtype=np.float32).reshape(4, 3, 16)
    weight = np.ones(16, np.float32)
    prepared = rms.prepare_rms_norm(
        input_shape=[4, 3, 16], output_shape=[4, 3, 16],
        rstd_shape=[4, 3, 1], epsilon=1e-6)
    prediction = rms.predict(prepared)
    assert prediction["materialization"] == {
        "norm_output": "384", "residual": "0", "rstd": "48"}
    view = prepared["attrs"]["execution_view"]
    assert view["input"]["shape"] == [12, 16]
    assert view["rstd"]["shape"] == [12, 1]
    assert view["original_rstd_shape"] == [4, 3, 1]
    assert view["rstd_flat_index"] == 12
    program = rms.execution_program(prepared)
    values = dict(program.typed_values)
    assert values["rstd"].layout.shape == (4, 3, 1)
    buffers = {item["name"]: item for item in program.buffers}
    assert buffers["x"]["bytes"] == 384
    assert buffers["weight"]["bytes"] == 32
    assert buffers["output"]["bytes"] == 384
    assert buffers["rstd"]["bytes"] == 48

    original = rms.execute(prepared, {"x": source, "weight": weight})
    flat = rms.execute(prepared, {
        "x": source.reshape(12, 16), "weight": weight})
    np.testing.assert_allclose(original["outputs"]["output"],
                               flat["outputs"]["output"], rtol=0, atol=0)
    np.testing.assert_allclose(original["outputs"]["rstd"],
                               flat["outputs"]["rstd"], rtol=0, atol=0)
    assert original["outputs"]["output"].shape == (4, 3, 16)
    assert original["outputs"]["rstd"].shape == (4, 3, 1)

    with pytest.raises(ValueError, match="rms_norm_rstd_shape_mismatch"):
        rms.prepare_rms_norm(
            input_shape=[4, 3, 16], rstd_shape=[4, 3], epsilon=1e-6)
    last = rms.prepare_rms_norm(rows=2, hidden=4, epsilon=1e-6)
    assert last["attrs"]["rstd_shape"] == [2, 1]
    assert last["attrs"]["execution_rstd_shape"] == [2, 1]


def test_swiglu_dim0_and_high_rank_rms_execute_same_leaf_program():
    execution = _execution()
    names = execution.stage_names[(0, 0)]
    swiglu_prepared = swiglu.prepare_swiglu(
        input_shape=[4, 32], output_shape=[2, 32], dim=0)
    swiglu_program = swiglu.execution_program(swiglu_prepared)
    swiglu_before = dict(execution.engine.serviced)
    swiglu_handle = swiglu_program.start(
        execution, name="swiglu_dim0",
        inputs={"input": execution.engine.submit("swiglu_input")})
    execution.engine.advance(until=(swiglu_handle.event,))
    swiglu_work = swiglu.predict(swiglu_prepared)["work"]
    for pipe, amount in swiglu_work.items():
        assert (execution.engine.serviced[names[pipe]]
                - swiglu_before[names[pipe]]) == pytest.approx(
                    float(amount))
    assert swiglu_handle.event.done
    assert dict(swiglu_handle.output_events)["output"].done
    assert dict(swiglu_handle.output_values)["output"].producer == (
        dict(swiglu_handle.output_events)["output"].name)
    assert swiglu_program.fragments[0].ports.tensor_in[0].name == "input"
    assert swiglu_program.buffers[0]["bytes"] == 256
    assert swiglu_program.buffers[1]["bytes"] == 128

    rms_prepared = rms.prepare_rms_norm(
        input_shape=[4, 3, 16], rstd_shape=[4, 3, 1], epsilon=1e-6)
    rms_program = rms.execution_program(rms_prepared)
    rms_before = dict(execution.engine.serviced)
    rms_handle = rms_program.start(
        execution, name="rms_high_rank",
        inputs={
            "x": execution.engine.submit("rms_x"),
            "weight": execution.engine.submit("rms_weight")})
    execution.engine.advance(until=(rms_handle.event,))
    rms_work = rms.predict(rms_prepared)["work"]
    for pipe, amount in rms_work.items():
        assert (execution.engine.serviced[names[pipe]]
                - rms_before[names[pipe]]) == pytest.approx(
                    float(amount))
    assert rms_handle.event.done
    assert set(dict(rms_handle.output_events)) == {"output", "rstd"}
    rms_values = dict(rms_handle.output_values)
    assert rms_values["rstd"].layout.shape == (4, 3, 1)
    assert rms_program.buffers[0]["bytes"] == 384
    assert rms_program.buffers[1]["bytes"] == 32
    assert rms_program.buffers[2]["bytes"] == 384
    assert rms_program.buffers[3]["bytes"] == 48


def _norm_public_request(entry, namespace, tensors, attrs):
    return {
        "api": {"namespace": namespace, "entry": entry},
        "version": {
            "api": "cann-9.2.0", "source": "r31",
            "image": ("sha256:9e869e8148467296f56d1001b0dac4fea"
                      "893c841ac7d0a03d4e824a233c2ffea")},
        "tensors": tensors,
        "attrs": dict(attrs),
    }


def test_rms_norm_named_accesses_reach_predict_program_and_public_dispatch():
    from mindie_sim.prediction.piecewise.native_invocation import (
        simulate_aclnn_call,
    )

    prepared = rms.prepare_rms_norm(
        input_shape=[4, 3, 16], rstd_shape=[4, 3, 1], epsilon=1e-6)
    plan = prepared["access_plan"]
    by_name = {item["name"]: item for item in plan["accesses"]}
    assert set(by_name) == {"x", "gamma", "output", "rstd"}
    assert by_name["x"]["scope"] == "row"
    assert by_name["x"]["logical_bytes"] == 384
    assert by_name["x"]["storage_bytes"] == 384
    assert by_name["gamma"]["scope"] == "call"
    assert by_name["gamma"]["logical_bytes"] == 32
    assert by_name["output"]["bytes_per_row"] == 32
    assert by_name["output"]["logical_bytes"] == 384
    assert by_name["rstd"]["bytes_per_row"] == 4
    assert by_name["rstd"]["logical_bytes"] == 48
    assert plan["logical_materialization"] == {
        "norm_output": 384, "residual": 0, "rstd": 48}
    assert plan["transactions"]["read_row_bytes"] == 32
    assert plan["transactions"]["write_row_bytes"] == 64
    assert (plan["transactions"]["mte2_bytes"],
            plan["transactions"]["mte3_bytes"],
            plan["transactions"]["hbm_bytes"]) == (416, 768, 1184)

    prediction = rms.predict(prepared)
    assert prediction["work"]["mte2"] == "416"
    assert prediction["work"]["mte3"] == "768"
    assert prediction["work"]["hbm"] == "1184"
    assert prediction["materialization"] == {
        "norm_output": "384", "residual": "0", "rstd": "48"}
    assert prediction["program"]["accesses"] == plan["accesses"]
    assert prediction["program"]["access_transactions"] == plan[
        "transactions"]

    program = rms.execution_program(prepared)
    assert [item["name"] for item in program.accesses] == [
        "x", "gamma", "output", "rstd"]
    assert [item.name for item in
            program.fragments[0].ports.tensor_in] == ["x", "weight"]
    assert [item.name for item in
            program.fragments[0].ports.tensor_out] == ["output", "rstd"]
    assert program.access_plan == plan
    assert program.abi_handoff["source_axis"]["input_rank"] == 3
    assert program.abi_handoff["original_descriptors"]["rstd"]["shape"] == [
        4, 3, 1]

    execution = _execution()
    names = execution.stage_names[(0, 0)]
    before = dict(execution.engine.serviced)
    handle = program.start(execution, name="rms_access")
    execution.engine.advance(until=(handle.event,))
    assert ((execution.engine.serviced[names["mte2"]] - before[names["mte2"]]),
            (execution.engine.serviced[names["mte3"]] - before[names["mte3"]]),
            (execution.engine.serviced[names["hbm"]] - before[names["hbm"]])) \
        == pytest.approx((416, 768, 1184))
    for name, event in handle.output_events:
        assert event.done
        assert dict(handle.output_values)[name].producer == event.name
    assert set(dict(handle.output_bindings)) == {"output", "rstd"}

    result = simulate_aclnn_call(
        _norm_public_request(
            "torch_npu.npu_rms_norm", "torch",
            {"x": {"shape": [4, 3, 16], "dtype": "bf16"},
             "weight": {"shape": [16], "dtype": "bf16"},
             "output": {"shape": [4, 3, 16], "dtype": "bf16"},
             "rstd": {"shape": [4, 3, 1], "dtype": "fp32"}},
            {"epsilon": 1e-6}),
        hardware="Ascend950DT_9582")
    assert result.status == "predicted", result.reasons
    physical = result.program["physical"]
    assert physical["work"]["mte2"] == "416"
    assert physical["work"]["mte3"] == "768"
    assert physical["work"]["hbm"] == "1184"
    assert physical["access_plan"]["accesses"] == plan["accesses"]
    assert physical["access_plan"]["original_shapes"] == {
        "input": [4, 3, 16], "output": [4, 3, 16],
        "rstd": [4, 3, 1], "weight": [16]}
    handoff = physical["abi_handoff"]
    assert handoff["source_axis"]["leading_shape"] == [4, 3]
    assert handoff["descriptors"]["input"]["shape"] == [4, 3, 16]
    assert handoff["execution_descriptors"]["rstd"]["shape"] == [12, 1]
    assert handoff["start_entry"] == "LeafExecutionProgram.start"
    assert result.execution_program.access_plan["accesses"] == plan["accesses"]


def test_gated_norm_named_accesses_two_orders_plain_and_ragged_width():
    for order in (True, False):
        prepared = gated.prepare_gated_norm(
            rows=2, head_dim=32, norm_before_gate=order)
        plan = prepared["access_plan"]
        by_name = {item["name"]: item for item in plan["accesses"]}
        assert by_name["x"]["logical_bytes"] == 128
        assert by_name["gate"]["logical_bytes"] == 128
        assert by_name["gamma"]["scope"] == "call"
        assert by_name["gamma"]["logical_bytes"] == 64
        assert by_name["output"]["logical_bytes"] == 128
        assert by_name["rstd"]["logical_bytes"] == 8
        assert plan["logical_materialization"] == {
            "norm_output": 128, "residual": 0, "rstd": 8}
        prediction = gated.predict(prepared)
        assert prediction["work"]["mte2"] == "320"
        assert prediction["work"]["mte3"] == "192"
        assert prediction["work"]["hbm"] == "512"
        assert prediction["materialization"] == {
            "norm_output": "128", "residual": "0", "rstd": "8"}
        assert prediction["program"]["accesses"] == plan["accesses"]
        program = gated.execution_program(prepared)
        assert [item["name"] for item in program.accesses] == [
            "x", "gate", "gamma", "output", "rstd"]
        assert [item.name for item in
                program.fragments[0].ports.tensor_in] == [
                    "x", "gate", "weight"]
        execution = _execution()
        names = execution.stage_names[(0, 0)]
        before = dict(execution.engine.serviced)
        handle = program.start(execution, name=f"gated_{order}")
        execution.engine.advance(until=(handle.event,))
        assert ((execution.engine.serviced[names["mte2"]]
                 - before[names["mte2"]]),
                (execution.engine.serviced[names["mte3"]]
                 - before[names["mte3"]])) == pytest.approx((320, 192))

    from mindie_sim.prediction.piecewise.native_invocation import (
        simulate_aclnn_call,
    )
    public = simulate_aclnn_call(
        _norm_public_request(
            gated.ENTRY, "triton",
            {"x": {"shape": [2, 32], "dtype": "bf16"},
             "weight": {"shape": [32], "dtype": "bf16"},
             "gate": {"shape": [2, 32], "dtype": "bf16"},
             "output": {"shape": [2, 32], "dtype": "bf16"}},
            {"head_dim": 32, "epsilon": 1e-6, "dtype": "bf16",
             "norm_before_gate": True, "gate_present": True}),
        hardware="Ascend950DT_9582")
    assert public.status == "predicted", public.reasons
    assert public.program["physical"]["work"]["mte2"] == "320"
    assert public.program["physical"]["work"]["mte3"] == "192"
    assert public.program["physical"]["work"]["hbm"] == "512"
    assert public.program["physical"]["access_plan"]["accesses"] == \
        gated.prepare_gated_norm(
            rows=2, head_dim=32, norm_before_gate=True)[
                "access_plan"]["accesses"]
    assert public.execution_program.access_plan["accesses"] == \
        public.program["physical"]["access_plan"]["accesses"]

    # Plain two-dimensional geometry and a non-nice width use the same formula,
    # not a captured shape table.
    for rows, width, expected in (
            (3, 8, {"mte2": 112, "mte3": 96, "hbm": 208}),
            (2, 7, {"mte2": 78, "mte3": 64, "hbm": 142})):
        prepared = gated.prepare_gated_norm(
            rows=rows, head_dim=width, norm_before_gate=True)
        plan = prepared["access_plan"]
        prediction = gated.predict(prepared)
        assert {key: int(prediction["work"][key]) for key in expected} == \
            expected
        assert plan["transactions"]["mte2_bytes"] == expected["mte2"]
        assert plan["transactions"]["mte3_bytes"] == expected["mte3"]
        assert prediction["materialization"] == {
            "norm_output": str(rows * width * 2),
            "residual": "0", "rstd": str(rows * 4)}


def test_first_output_writer_default_last_use_keeps_grant_until_stage1_read():
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        AllocationEffect, ProgramFragment, WorkSet,
    )
    from mindie_sim.schema.subgraphs import PortSet

    execution = _execution(memory=100.0)
    execution.engine.extend(pools={"norm_pool": 16})
    ready = execution.engine.submit("ready")
    holder_hold = execution.engine.future("holder_release")
    conductor = execution.fragment_conductor(
        replica=0, stage=0, pool_bindings={"norm_pool": "norm_pool"})
    holder = ProgramFragment(
        "holder", occurrence=Occurrence("kernel", "holder"),
        ports=PortSet(), work=WorkSet(()),
        allocations=(AllocationEffect(
            "acquire", "holder", "norm_pool", 16),))
    holder_handle = conductor.submit(
        holder, after=(ready,), holds={"holder": holder_hold},
        owner="holder")
    execution.engine.advance(until=(holder_handle.event,), first=False)
    assert execution.engine.allocated["norm_pool"] == 16

    gamma_ready = execution.engine.future("gamma_ready")
    program = add.execution_program(add.prepare_add_rms_norm(
        rows=2, hidden=4, epsilon=1e-6))
    handle = program.start(
        execution, name="add_first_writer", ready=ready,
        output_grants={"residual_out": {
            "pool": "norm_pool", "size": 16, "name": "residual_grant"}},
        inputs={"gamma": gamma_ready},
        pool_bindings={"norm_pool": "norm_pool"})
    stage0, stage1 = handle.handles[0].event, handle.handles[1].event
    assert [effect.name for effect in
            handle.handles[0].fragment.allocations] == ["residual_grant"]
    assert handle.handles[1].fragment.allocations == ()
    assert execution.engine.serviced["mte2@r0:s0"] == 0
    assert execution.engine.serviced["mte3@r0:s0"] == 0
    assert execution.engine.serviced["vector@r0:s0"] == 0

    execution.engine.resolve(holder_hold)
    execution.engine.advance(
        until=tuple(holder_handle.release_events), first=False)
    execution.engine.advance(until=(stage0,), first=False)
    assert stage0.done
    assert execution.engine.allocated["norm_pool"] == 16
    assert not stage1.done

    competitor = ProgramFragment(
        "competitor", occurrence=Occurrence("kernel", "competitor"),
        ports=PortSet(), work=WorkSet(()),
        allocations=(
            AllocationEffect("acquire", "competitor", "norm_pool", 16),
            AllocationEffect("release", "competitor", "norm_pool", 16)))
    competitor_handle = conductor.submit(
        competitor, after=(ready,), owner="competitor")
    assert not competitor_handle.event.done
    assert execution.engine.allocated["norm_pool"] == 16

    execution.engine.resolve(gamma_ready)
    execution.engine.advance(until=(stage1,), first=False)
    assert stage1.done
    execution.engine.advance(
        until=tuple(handle.release_events), first=False)
    assert execution.engine.allocated["norm_pool"] == 0
    execution.engine.advance(until=(competitor_handle.event,), first=False)
    assert competitor_handle.event.done
    assert execution.engine.allocated["norm_pool"] == 0


def test_external_reader_joins_internal_last_use_before_grant_release():
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        ProgramFragment, WorkItem, WorkSet,
    )
    from mindie_sim.schema.subgraphs import PortSet, TensorPortSpec

    execution = _execution(memory=100.0)
    execution.engine.extend(pools={"norm_pool": 16})
    reader_done = execution.engine.future("external_reader_done")
    program = add.execution_program(add.prepare_add_rms_norm(
        rows=2, hidden=4, epsilon=1e-6))
    handle = program.start(
        execution, name="add_external_reader",
        output_grants={"residual_out": {
            "pool": "norm_pool", "size": 16, "name": "residual_grant"}},
        output_bindings={"residual_out": {
            "storage": "residual_buffer", "readers": (reader_done,)}},
        pool_bindings={"norm_pool": "norm_pool"})
    assert dict(handle.storage_bindings)["residual_out"][
        "storage"] == "residual_buffer"
    binding = dict(handle.output_bindings)["residual_out"]
    assert binding["writer_event"] == dict(
        handle.output_events)["residual_out"].name
    assert binding["external_readers"] == ["external_reader_done"]
    assert binding["grant"]["name"] == "residual_grant"

    execution.engine.advance(until=(handle.event,), first=False)
    assert handle.event.done
    assert execution.engine.allocated["norm_pool"] == 16
    assert not any(event.done for event in handle.release_events)

    conductor = execution.fragment_conductor(
        replica=0, stage=0, pool_bindings={"norm_pool": "norm_pool"})
    reader = ProgramFragment(
        "external_reader", occurrence=Occurrence("kernel", "external_reader"),
        ports=PortSet(tensor_in=(TensorPortSpec(
            "residual_out", "in", "activation", dtype="bf16"),)),
        work=WorkSet((WorkItem("hbm", 8.0, "bytes"),)))
    reader_handle = conductor.submit(
        reader, after=(handle.event,),
        inputs={"residual_out": handle}, owner="external_reader")
    execution.engine.advance(until=(reader_handle.event,), first=False)
    assert reader_handle.event.done
    assert execution.engine.allocated["norm_pool"] == 16
    assert not any(event.done for event in handle.release_events)

    execution.engine.resolve(reader_done)
    execution.engine.advance(
        until=tuple(handle.release_events), first=False)
    assert execution.engine.allocated["norm_pool"] == 0


def test_external_rstd_reader_also_joins_the_last_use_release():
    execution = _execution()
    execution.engine.extend(pools={"stat_pool": 48})
    reader_done = execution.engine.future("rstd_reader_done")
    program = rms.execution_program(rms.prepare_rms_norm(
        input_shape=[4, 3, 16], rstd_shape=[4, 3, 1], epsilon=1e-6))
    handle = program.start(
        execution, name="rms_rstd_reader",
        output_grants={"rstd": {
            "pool": "stat_pool", "size": 48, "name": "rstd_grant"}},
        output_bindings={"rstd": {"readers": (reader_done,)}},
        pool_bindings={"stat_pool": "stat_pool"})
    execution.engine.advance(until=(handle.event,), first=False)
    assert execution.engine.allocated["stat_pool"] == 48
    assert not any(event.done for event in handle.release_events)
    assert dict(handle.output_bindings)["rstd"]["external_readers"] == [
        "rstd_reader_done"]
    execution.engine.resolve(reader_done)
    execution.engine.advance(
        until=tuple(handle.release_events), first=False)
    assert execution.engine.allocated["stat_pool"] == 0


def test_borrowed_grant_registers_reader_without_second_acquire():
    from mindie_sim.prediction.capacity import Memory, MemoryPolicy

    execution = _execution()
    memory = Memory(execution.case, MemoryPolicy(derive_workspace=False))
    assert execution.bind_memory(memory)
    assert memory.grant_batch(0, [
        {"name": "zero_buf_a", "bytes": 16, "storage": "zero_buf_a",
         "ranges": (0, 16), "generation": 0},
        {"name": "zero_buf_b", "bytes": 16, "storage": "zero_buf_b",
         "ranges": (0, 16), "generation": 1},
    ])
    assert memory.occupied_bytes(0, 0) == 32
    reader_a = execution.engine.future("zero_reader_a")
    reader_b = execution.engine.future("zero_reader_b")
    program = zero.execution_program(zero.prepare_inplace_zero(
        tokens=2, kept_heads=2, head_dim=2))
    first = program.start(
        execution, name="zero_a",
        output_bindings={"output": {
            "borrow": True, "name": "zero_buf_a",
            "readers": (reader_a,)}})
    second = program.start(
        execution, name="zero_b",
        output_bindings={"output": {
            "borrow": True, "name": "zero_buf_b",
            "storage": "zero_buf_b", "generation": 1,
            "readers": (reader_b,)}})
    assert first.handles[0].fragment.allocations == ()
    assert second.handles[0].fragment.allocations == ()
    assert sum(execution.engine.allocated.values()) == 0
    first_binding = dict(first.output_bindings)["output"]
    second_binding = dict(second.output_bindings)["output"]
    assert first_binding["grant_name"] == "zero_buf_a"
    assert first_binding["storage"] == "zero_buf_a"
    assert first_binding["grant_binding"]["storage"] == "zero_buf_a"
    assert second_binding["grant_name"] == "zero_buf_b"
    assert first_binding["storage"] != second_binding["storage"]
    assert first_binding["use_done_event"] != second_binding[
        "use_done_event"]
    readers = memory.grant_readers
    assert (0, "zero_buf_a") in readers
    assert (0, "zero_buf_b") in readers
    assert readers[(0, "zero_buf_a")]
    assert readers[(0, "zero_buf_b")]

    execution.engine.advance(
        until=(first.event, second.event), first=False)
    assert first.event.done and second.event.done
    assert (0, "zero_buf_a") in memory.grant_readers
    assert (0, "zero_buf_b") in memory.grant_readers
    execution.engine.resolve(reader_a)
    execution.engine.resolve(reader_b)
    execution.engine.advance(
        until=(dict(first.use_events)["core_attn_out"],
               dict(second.use_events)["core_attn_out"]), first=False)
    assert (0, "zero_buf_a") not in memory.grant_readers
    assert (0, "zero_buf_b") not in memory.grant_readers
    assert {row["kind"] for row in memory.release_events} == {
        "grant_reader"}
    assert sum(execution.engine.allocated.values()) == 0


def test_quant_async_reader_keeps_its_output_grant_until_completion():
    from mindie_sim.schema.invocation import Occurrence
    from mindie_sim.schema.operators import (
        ProgramFragment, WorkItem, WorkSet,
    )
    from mindie_sim.schema.subgraphs import PortSet, TensorPortSpec

    execution = _execution()
    execution.engine.extend(pools={"quant_pool": 8})
    reader_done = execution.engine.future("quant_reader_done")
    program = quant.execution_program(quant.prepare_dynamic_quant_v2(
        rows=2, width=4, is_symmetrical=True))
    handle = program.start(
        execution, name="quant_reader",
        output_grants={"y": {
            "pool": "quant_pool", "size": 8, "name": "y_grant"}},
        output_bindings={"y": {"readers": (reader_done,)}},
        pool_bindings={"quant_pool": "quant_pool"})
    execution.engine.advance(until=(handle.event,), first=False)
    assert execution.engine.allocated["quant_pool"] == 8
    assert not any(event.done for event in handle.release_events)
    conductor = execution.fragment_conductor(
        replica=0, stage=0, pool_bindings={"quant_pool": "quant_pool"})
    reader = ProgramFragment(
        "quant_reader", occurrence=Occurrence("kernel", "quant_reader"),
        ports=PortSet(tensor_in=(TensorPortSpec(
            "y", "in", "activation", dtype="int8"),)),
        work=WorkSet((WorkItem("hbm", 8.0, "bytes"),)))
    reader_handle = conductor.submit(
        reader, after=(handle.event,), inputs={"y": handle},
        owner="quant_reader")
    execution.engine.advance(until=(reader_handle.event,), first=False)
    assert execution.engine.allocated["quant_pool"] == 8
    execution.engine.resolve(reader_done)
    execution.engine.advance(
        until=tuple(handle.release_events), first=False)
    assert execution.engine.allocated["quant_pool"] == 0


def test_every_leaf_output_value_and_writer_come_from_the_same_binding():
    for label, module, prepared in _leaf_programs():
        execution = _execution()
        program = module.execution_program(prepared)
        handle = program.start(execution, name=label)
        execution.engine.advance(until=(handle.event,))
        records = dict(handle.output_bindings)
        values = dict(handle.output_values)
        assert records
        assert set(values) == set(records)
        for name, record in records.items():
            assert record["writer_event"]
            assert record["writer_event"] == dict(
                handle.output_events)[name].name
            if record["materialized"]:
                assert record["storage"]
                assert values[name].storage == record["storage"]
            assert values[name].producer == record["writer_event"]


def test_every_leaf_grant_is_acquired_at_its_first_writer_and_released_after_use():
    for label, module, prepared in _leaf_programs():
        execution = _execution()
        execution.engine.extend(pools={"leaf_pool": 1 << 20})
        program = module.execution_program(prepared)
        record = prepared.get("prepare") if "prepare" in prepared else prepared
        descriptors = dict(record.get("descriptors") or {})
        unmaterialized = set(program.unmaterialized)
        buffers = {item["name"]: item for item in program.buffers
                   if isinstance(item, dict)}
        grants = {}
        for output, descriptor in descriptors.items():
            if (descriptor.get("role") != "output"
                    or descriptor.get("materialized", True) is False
                    or output in unmaterialized):
                continue
            size = max(1, int(buffers.get(output, {}).get("bytes", 1)))
            grants[output] = {
                "pool": "leaf_pool", "size": size,
                "name": f"{label}_{output}_grant"}
        assert grants, label
        handle = program.start(
            execution, name=f"{label}_grant",
            output_grants=grants,
            pool_bindings={"leaf_pool": "leaf_pool"})
        records = dict(handle.output_bindings)
        for index, fragment in enumerate(handle.handles):
            for effect in fragment.fragment.allocations:
                assert effect.action == "acquire"
        for output, record in records.items():
            grant_name = record["grant_name"]
            if output not in grants:
                continue
            assert grant_name == grants[output]["name"]
            writer_stage = record["writer_stage"]
            assert any(effect.name == grant_name for effect in
                       handle.handles[writer_stage].fragment.allocations)
            assert all(
                effect.name != grant_name
                for index, item in enumerate(handle.handles)
                if index != writer_stage
                for effect in item.fragment.allocations)
        execution.engine.advance(until=(handle.event,), first=False)
        execution.engine.advance(
            until=tuple(handle.release_events), first=False)
        assert all(event.done for event in handle.release_events)
        assert sum(execution.engine.allocated.values()) == 0


def test_norm_access_plan_keeps_gamma_and_beta_resident_once():
    plan = add.norm_access_plan(
        rows=2, width=16, dtype_bytes=2, gamma_bytes=32, beta_bytes=8,
        row_inputs=(("x", "x", 32),),
        output_name="output", rstd_name="rstd", rstd_bytes=4,
        gamma_name="weight", beta_name="beta", family="rmsnorm")
    by_name = {item["name"]: item for item in plan["accesses"]}
    assert by_name["x"]["scope"] == "row"
    assert by_name["gamma"]["scope"] == "call"
    assert by_name["beta"]["scope"] == "call"
    assert by_name["gamma"]["logical_bytes"] == 32
    assert by_name["beta"]["logical_bytes"] == 8
    assert plan["family_parameters"]["gamma_bytes"] == 40
    assert plan["family_parameters"]["beta_bytes"] == 8
    assert plan["family_parameters"]["read_bytes_per_row"] == 72
    assert plan["transactions"]["read_row_bytes"] == 32
    assert plan["transactions"]["mte2_bytes"] == 2 * 32 + 40
    assert plan["transactions"]["mte3_bytes"] == 2 * 64
    assert plan["logical_materialization"] == {
        "norm_output": 64, "residual": 0, "rstd": 8}
