"""Public T12 catalogue/normalizer/dispatcher for every declared leaf.

[Role] Send one representative independent input through the public
  ``simulate_aclnn_call`` for each matrix and attention/state leaf, and compare
  the public binding/program/work with the leaf's own independent result.
[Invariants] The normalizer preserves group_list.values, non-ND layouts and
  logical/storage/device fields; matrix work stays timing-free; the leaf's
  selected call and physical work are not re-priced by T12.
[Side-Effect Policy] Pure CPU records; no device, weights or I/O.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mindie_sim.prediction.piecewise import (
    native_grouped_matmul_swiglu_quant_v2 as gmm1,
    native_grouped_matmul_v4 as gmm2,
    native_qbmm as qbmm,
    native_r31_matmul_bf16 as matmul,
    native_r31_quantmatmul_weight_nz as qmmnz,
)
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.native_binding import (
    bind_native_call, bind_selected_native_call, declare_native_call,
)
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.native_binding import (
    bind_native_call, bind_selected_native_call, declare_native_call,
)
from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.piecewise.native_invocation import (
    a5_native_abi_inventory,
    native_abi_timing_free,
    native_qbmm_w8a8_invocation,
    simulate_aclnn_call,
    submit_native_call,
)
from mindie_sim.prediction.piecewise.native_gdn import (
    gdn_abi_spec,
    simulate_independent as simulate_gdn,
)
from mindie_sim.prediction.piecewise.native_gather_ssm_states import (
    simulate_independent as simulate_gather,
)
from mindie_sim.prediction.piecewise.native_l2norm_fwd_kernel2 import (
    simulate_independent as simulate_l2norm,
)
from mindie_sim.prediction.piecewise.native_r64_aclnn_cat import (
    simulate_independent as simulate_cat,
)
from mindie_sim.prediction.piecewise.native_r64_layer_norm_with_impl_mode import (
    simulate_independent as simulate_layer_norm,
)
from mindie_sim.prediction.piecewise.native_r64_muls_add import (
    simulate_independent as simulate_muls,
)
from mindie_sim.prediction.piecewise.native_r64_scatter_pa_cache import (
    simulate_independent as simulate_scatter_pa,
)
from mindie_sim.prediction.piecewise.native_r64_transpose_batch_matmul import (
    simulate_independent as simulate_tbmm,
)
from mindie_sim.prediction.piecewise.native_r64_triton_rope_siso import (
    simulate_independent as simulate_rope,
)
from mindie_sim.prediction.piecewise.native_scatter_ssm_states import (
    simulate_independent as simulate_scatter,
)
from mindie_sim.prediction.piecewise.native_split_qkv_rmsnorm_mrope import (
    ENTRY as SPLIT_ENTRY, R31_IMAGE_SHA as SPLIT_IMAGE,
    SOURCE as SPLIT_SOURCE,
    simulate_independent as simulate_split,
)

from test_native_attnstate_abi_leaves import (
    _cat_request, _gather_request, _gdn_request, _layernorm_request,
    _l2norm_request, _muls_request, _rope_request, _scatter_pa_request,
    _scatter_request, _tbmm_request,
)
from test_native_matrix_abi_leaves import (
    _grouped_tensors, _qbmm_tensors, R31, R64,
)

SKU = "Ascend950DT_9582"


def _matrix_request(entry, namespace, tensors, attrs, version):
    return {
        "api": {"namespace": namespace, "entry": entry},
        "version": version, "tensors": tensors, "attrs": attrs,
    }


def _split_request():
    return {
        "api": {"namespace": "cann_kernel", "entry": SPLIT_ENTRY},
        "version": {"api": "r31-original-capture",
                    "source": SPLIT_SOURCE["revision"],
                    "image_id": SPLIT_IMAGE},
        "tensors": {
            "input": {"shape": [4, 48], "dtype": "bf16"},
            "q_weight": {"shape": [8], "dtype": "bf16"},
            "k_weight": {"shape": [8], "dtype": "bf16"},
            "cos_sin_cache": {"shape": [3, 4, 8], "dtype": "bf16"},
            "positions": {"shape": [3, 4], "dtype": "int64"},
            "q": {"shape": [4, 32], "dtype": "bf16"},
        },
        "attrs": {
            "q_hidden_size": 32, "kv_hidden_size": 8, "head_dim": 8,
            "q_size": 32, "kv_size": 8, "gate_size": 0, "rope_dim": 8,
            "eps": 1e-6, "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    }


MATRIX_CASES = (
    ("qbmm", _matrix_request(
        "torch_npu.npu_quant_matmul", "torch", _qbmm_tensors(4),
        {"group_size": 32}, R64), qbmm.predict_abi),
    ("grouped_v4", _matrix_request(
        "torch_npu.npu_grouped_matmul", "torch",
        _grouped_tensors(5, (2, 0, 3) + (0,) * 13),
        {"group_list_type": 1}, R64), gmm2.predict_abi),
    ("grouped_swiglu", _matrix_request(
        "torch_npu.npu_grouped_matmul_swiglu_quant_v2", "torch",
        _grouped_tensors(
            32, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
            n=8192, k=6144, swiglu=True),
        {"group_list_type": 0}, R64), gmm1.predict_abi),
    ("matmul", _matrix_request(
        "aclnnMatmul", "aclnn",
        {"x1": {"shape": [257, 5120], "dtype": "bf16", "role": "input"},
         "x2": {"shape": [17408, 5120], "dtype": "bf16", "role": "input"},
         "output": {"shape": [257, 17408], "dtype": "bf16",
                    "role": "output"}},
        {"transpose_x1": False, "transpose_x2": True}, R31),
     matmul.predict_abi),
    ("weight_nz", _matrix_request(
        "aclnnQuantMatmulWeightNz", "aclnn",
        {"x": {"shape": [17, 5120], "dtype": "int8", "role": "input"},
         "weight": {"shape": [17408, 5120], "dtype": "int8",
                    "role": "weight"},
         "weight_scale": {"shape": [17408], "dtype": "bf16",
                          "role": "scale"},
         "pertoken_scale": {"shape": [17], "dtype": "fp32",
                            "role": "scale"},
         "output": {"shape": [17, 17408], "dtype": "bf16",
                    "role": "output"}},
        {"weight_scale_storage_format": "fp32"}, R31),
     qmmnz.predict_abi),
)


@pytest.mark.parametrize("name,invocation,predict_abi", MATRIX_CASES)
def test_matrix_leaf_public_dispatch_keeps_selected_work(name, invocation,
                                                         predict_abi):
    direct = predict_abi(
        invocation["tensors"], attrs=invocation["attrs"],
        version=invocation["version"], conditions={})
    assert direct["status"] == "predicted", direct.get("gap")
    result = simulate_aclnn_call(invocation, hardware=SKU)
    assert result.status == "predicted", result.reasons
    assert result.prediction["fidelity"] == "timing_free_work"
    assert result.prediction["duration_ms"] is None
    assert result.prediction["work"] == direct["work"]
    assert result.binding is not None
    assert result.binding.op_id.startswith("native:")
    assert result.program is result.prediction["program"]
    assert result.program["physical"]["work"] == direct["work"]
    assert result.program["physical"]["geometry"] == direct["geometry"]
    assert result.program["physical"]["scale"] == direct["scale"]
    assert result.program["physical"]["metadata"] == direct["metadata"]
    assert result.program["physical"]["accumulator"] == direct["accumulator"]
    assert result.program["physical"]["regions"] == direct["regions"]
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert json.loads(json.dumps(payload))["prediction"]["program"] == (
        json.loads(json.dumps(result.program)))


ATTN_CASES = (
    ("gather", _gather_request(), simulate_gather),
    ("scatter", _scatter_request(), simulate_scatter),
    ("scatter_pa", _scatter_pa_request(), simulate_scatter_pa),
    ("l2norm", _l2norm_request(), simulate_l2norm),
    ("rope", _rope_request(), simulate_rope),
    ("cat", _cat_request(), simulate_cat),
    ("muls", _muls_request(), simulate_muls),
    ("layer_norm", _layernorm_request(), simulate_layer_norm),
    ("tbmm", _tbmm_request(), simulate_tbmm),
    ("gdn", _gdn_request(), simulate_gdn),
    ("split_qkv", _split_request(), simulate_split),
)


@pytest.mark.parametrize("name,invocation,simulate", ATTN_CASES)
def test_attention_leaf_public_dispatch_keeps_leaf_program(name, invocation,
                                                            simulate):
    direct = simulate(invocation)
    assert direct["status"] in {"predicted", "partial"}, direct.get("reasons")
    result = simulate_aclnn_call(invocation, hardware=SKU)
    assert result.status == direct["status"], result.reasons
    if direct["status"] == "predicted":
        assert result.prediction["work"] == direct["prediction"]["work"]
        assert result.prediction["duration_ms"] == direct["prediction"][
            "duration_ms"]
    direct_binding = direct["binding"]
    assert result.binding is not None
    assert result.binding.member_op_ids == tuple(
        direct_binding["members"]["op_ids"])
    assert result.program is result.prediction["program"]
    assert result.program["physical"]["members"] == direct["program"].get(
        "members")
    assert result.program["physical"]["work"] == direct["prediction"]["work"]


def test_catalog_advertises_every_declared_leaf_and_dispatch():
    inventory = a5_native_abi_inventory()
    assert inventory["totals"]["declared_entries"] == 27
    assert inventory["totals"]["implemented_entries"] == 27
    entries = {row["entry"]: row for row in inventory["entries"]}
    assert entries["torch_npu.npu_fused_infer_attention_score"][
        "dispatch"] == "paged_fia_build"
    for name, request, _ in MATRIX_CASES:
        entry = request["api"]["entry"]
        assert entry in entries
        assert entries[entry]["implemented"] is True
        assert entries[entry]["dispatch"] == "matrix_predict_abi"
    for name, request, _ in ATTN_CASES:
        entry = request["api"]["entry"]
        assert entry in entries
        assert entries[entry]["implemented"] is True
        assert entries[entry]["dispatch"] == "leaf_simulate_independent"
    assert entries["torch_npu.npu_rms_norm"]["implemented"] is True
    assert entries["torch_npu.npu_swiglu"]["implemented"] is True
    assert entries["torch_npu.npu_rms_norm"]["dispatch"] == (
        "vector_prepare_predict")
    assert entries["torch_npu.npu_swiglu"]["dispatch"] == (
        "vector_prepare_predict")


def test_group_list_values_and_weight_nz_layout_survive_normalization():
    grouped = _matrix_request(
        "torch_npu.npu_grouped_matmul", "torch",
        _grouped_tensors(5, (2, 0, 3) + (0,) * 13),
        {"group_list_type": 1}, R64)
    result = simulate_aclnn_call(grouped, hardware=SKU)
    assert result.status == "predicted"
    invocation = result.invocation
    group = invocation["tensors"]["group_list"]
    assert group["values"] == [2, 0, 3] + [0] * 13
    assert group["dtype"] == "int64"
    assert result.program["physical"]["metadata"]["group_list"]["values"] == (
        [2, 0, 3] + [0] * 13)

    nz_tensors = {
        "x": {"shape": [17, 5120], "dtype": "int8", "role": "input"},
        "weight_nz": {"shape": [544, 320, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [17408], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [17], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [17, 17408], "dtype": "bf16",
                   "role": "output"},
    }
    nz = simulate_aclnn_call(_matrix_request(
        "aclnnQuantMatmulWeightNz", "aclnn", nz_tensors,
        {"weight_scale_storage_format": "fp32"}, R31), hardware=SKU)
    assert nz.status == "predicted"
    assert nz.invocation["tensors"]["weight_nz"]["layout"] == "FRACTAL_NZ"
    assert nz.program["physical"]["tensors"]["weight_nz"]["views"][
        "storage"]["layout"] == "FRACTAL_NZ"


R31_VERSION = {
    "api": "cann-9.2.0", "source": "r31",
    "image": "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea",
}


def _vector_request(entry, namespace, tensors, attrs=()):
    return {
        "api": {"namespace": namespace, "entry": entry},
        "version": R31_VERSION, "tensors": tensors, "attrs": dict(attrs),
    }


VECTOR_CASES = (
    ("rms_norm", _vector_request(
        "torch_npu.npu_rms_norm", "torch",
        {"x": {"shape": [4, 16], "dtype": "bf16"},
         "weight": {"shape": [16], "dtype": "bf16"},
         "output": {"shape": [4, 16], "dtype": "bf16"},
         "rstd": {"shape": [4, 1], "dtype": "fp32"}},
        {"epsilon": 1e-6}), "rmsnorm"),
    ("swiglu", _vector_request(
        "torch_npu.npu_swiglu", "torch",
        {"input": {"shape": [4, 32], "dtype": "bf16"},
         "output": {"shape": [4, 16], "dtype": "bf16"}}), "swiglu"),
    ("sigmoid", _vector_request(
        "torch.sigmoid", "torch",
        {"input": {"shape": [4, 16], "dtype": "bf16"},
         "output": {"shape": [4, 16], "dtype": "bf16"}}), "sigmoid"),
    ("dynamic_quant", _vector_request(
        "torch_npu.npu_dynamic_quant", "torch",
        {"x": {"shape": [4, 16], "dtype": "bf16"},
         "y": {"shape": [4, 16], "dtype": "int8"},
         "scale": {"shape": [4], "dtype": "fp32"},
         "offset": {"shape": [4], "dtype": "fp32"}},
        {"is_symmetrical": True}), "DynamicQuantV2"),
    ("gated_norm", _vector_request(
        "vllm_ascend.ops.triton.layernorm_gated.layer_norm_fwd_npu",
        "triton",
        {"x": {"shape": [4, 16], "dtype": "bf16"},
         "weight": {"shape": [16], "dtype": "bf16"},
         "gate": {"shape": [4, 16], "dtype": "bf16"},
         "output": {"shape": [4, 16], "dtype": "bf16"}},
        {"head_dim": 16, "epsilon": 1e-6, "dtype": "bf16",
         "norm_before_gate": True, "gate_present": True}), "gated_norm"),
    ("index", _vector_request(
        "aclnnIndex", "aclnn",
        {"hidden_states": {"shape": [8, 16], "dtype": "bf16"},
         "logits_indices": {"shape": [3], "dtype": "int64"},
         "sample_hidden_states": {"shape": [3, 16], "dtype": "bf16"}}),
     "aclnn_index"),
    ("inplace_copy", _vector_request(
        "aclnnInplaceCopy", "aclnn",
        {"source": {"shape": [2, 4, 8], "dtype": "bf16"},
         "output": {"shape": [2, 2, 8], "dtype": "bf16"}},
        {"slice_offset_heads": 0, "dtype": "bf16"}), "inplace_copy"),
    ("inplace_zero", _vector_request(
        "aclnnInplaceZero", "aclnn",
        {"self": {"shape": [2, 2, 8], "dtype": "bf16"},
         "output": {"shape": [2, 2, 8], "dtype": "bf16"}},
        {"dtype": "bf16"}), "inplace_zero"),
    ("add_rms_norm_bias", _vector_request(
        "torch.ops._C_ascend.npu_add_rms_norm_bias", "torch",
        {"x1": {"shape": [4, 16], "dtype": "bf16"},
         "x2": {"shape": [4, 16], "dtype": "bf16"},
         "gamma": {"shape": [16], "dtype": "bf16"},
         "y": {"shape": [4, 16], "dtype": "bf16"},
         "rstd": {"shape": [4, 1], "dtype": "fp32"}},
        {"epsilon": 1e-6}), ("residual_add", "rmsnorm")),
)


@pytest.mark.parametrize("name,invocation,member", VECTOR_CASES)
def test_vector_leaf_public_dispatch_keeps_leaf_work(name, invocation,
                                                      member):
    result = simulate_aclnn_call(invocation, hardware=SKU)
    assert result.status == "predicted", result.reasons
    assert result.prediction["work"]
    assert result.binding is not None
    assert result.binding.member_op_ids == ()
    expected_physical = ((member,) if isinstance(member, str)
                         else tuple(member))
    assert result.program["physical"]["physical_members"] == list(
        expected_physical)
    assert result.program is result.prediction["program"]
    assert result.program["physical"]["work"] == result.prediction["work"]
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert json.loads(json.dumps(payload))["program"] == json.loads(
        json.dumps(result.program))


def test_dynamic_quant_keeps_asymmetric_and_unresolved_status():
    asymmetric = dict(VECTOR_CASES[3][1])
    asymmetric["attrs"] = {"is_symmetrical": False}
    result = simulate_aclnn_call(asymmetric, hardware=SKU)
    assert result.status == "not_implemented"
    assert result.prediction["status"] == "not_implemented"
    assert "dynamic_quant_asymmetric_offset_mode_not_modeled" in (
        result.reasons)

    unresolved = dict(VECTOR_CASES[3][1])
    unresolved["attrs"] = {"is_symmetrical": None}
    result = simulate_aclnn_call(unresolved, hardware=SKU)
    assert result.status == "insufficient_fields"
    assert result.prediction["status"] == "insufficient_fields"
    assert "dynamic_quant_is_symmetrical_unresolved" in result.reasons


def test_gdn_public_json_keeps_multi_kernel_call_expansion():
    request = _gdn_request()
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted", result.reasons
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert payload["binding"]["members"]["op_ids"] == [
        "prepare", "gated_delta"]
    assert len(payload["binding"]["kernels"]) == 5
    assert payload["program"]["physical"]["members"] == [
        "prepare", "gated_delta"]
    assert payload["program"]["physical"]["physical_members"] == [
        "prepare", "gated_delta"]
    assert payload["occurrence"]["priced"]["work"] == (
        payload["prediction"]["work"])


def test_matrix_timing_free_bridge_has_no_target_duration():
    from mindie_sim.prediction.piecewise.native_invocation import (
        native_abi_timing_free,
    )
    name, invocation, predict_abi = MATRIX_CASES[1]
    result = simulate_aclnn_call(invocation, hardware=SKU)
    assert result.status == "predicted"
    view = native_abi_timing_free(result)
    assert view["duration_ms"] is None
    assert view["work"] == result.prediction["work"]
    assert view["binding"] == result.binding.to_dict()
    assert view["program"] == result.program
    assert view["fidelity"] == "timing_free_work"


def _sigmoid_request(shape=(4, 16), out_shape=None, dtype="bf16",
                     out_dtype=None, conditions=None):
    return {
        "api": {"namespace": "torch", "entry": "torch.sigmoid"},
        "version": R31_VERSION,
        "tensors": {
            "input": {"shape": list(shape), "dtype": dtype},
            "output": {"shape": list(out_shape or shape),
                       "dtype": out_dtype or dtype},
        },
        "attrs": {},
        "conditions": dict(conditions or {}),
    }


def test_vector_abi_shape_and_dtype_contradictions_are_no_match():
    mismatch = simulate_aclnn_call(
        _sigmoid_request(out_shape=(4, 32)), hardware=SKU)
    assert mismatch.status == "no_match"
    assert "native_vector_shape_mismatch:output" in " ".join(
        mismatch.reasons)
    assert mismatch.prediction is None
    assert mismatch.replay["tensors"]["output"]["shape"] == [4, 32]
    assert mismatch.replay["tensors"]["input"]["shape"] == [4, 16]
    assert simulate_aclnn_call(
        mismatch.replay, hardware=SKU).status == "no_match"

    dtype = simulate_aclnn_call(
        _sigmoid_request(out_dtype="fp32"), hardware=SKU)
    assert dtype.status == "no_match"
    assert "native_vector_dtype_mismatch:output" in " ".join(dtype.reasons)
    assert dtype.prediction is None
    assert dtype.replay["tensors"]["output"]["dtype"] == "fp32"


def test_sigmoid_high_order_shape_uses_formula_and_preserves_source_abi():
    request = _sigmoid_request(shape=(4, 3, 16))
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted", result.reasons
    assert result.binding.member_op_ids == ()
    assert result.program["physical"]["physical_members"] == ["sigmoid"]
    assert result.program["physical"]["source_shapes"] == {
        "input": [4, 3, 16], "output": [4, 3, 16]}
    assert result.program["physical"]["execution_view"] == [12, 16]
    assert result.prediction["source_shapes"] == {
        "input": [4, 3, 16], "output": [4, 3, 16]}
    assert result.prediction["execution_view"] == [12, 16]
    assert result.binding.typed["shapes"]["input"] == [4, 3, 16]
    assert result.binding.typed["execution_view"] == [12, 16]
    assert result.binding.typed["tensors"]["output"]["shape"] == [4, 3, 16]
    assert result.program["physical"]["request_tensors"]["input"]["shape"] == [
        4, 3, 16]
    assert result.program["physical"]["work"]["mte2"] == "384"
    assert result.program["physical"]["work"]["mte3"] == "384"
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert json.loads(json.dumps(payload))["program"] == json.loads(
        json.dumps(result.program))


def test_sigmoid_explicit_cores_mt_matches_policy_and_identity():
    request = _sigmoid_request()
    explicit = simulate_aclnn_call(
        request, hardware=SKU, cores=1, mt=1)
    policy = simulate_aclnn_call(
        request, hardware=SKU,
        policy=ImplementationPolicy(cores=1, mt=1))
    assert explicit.status == policy.status == "predicted"
    assert explicit.prediction == policy.prediction
    assert explicit.program == policy.program
    assert explicit.binding.call_id == policy.binding.call_id
    assert explicit.environment["policy"]["cores"] == 1
    assert explicit.environment["policy"]["mt"] == 1
    assert policy.environment["policy"]["cores"] == 1
    assert policy.environment["policy"]["mt"] == 1


def test_vector_typed_identity_distinguishes_dtype_and_bare_caller():
    bf16 = simulate_aclnn_call(
        _sigmoid_request(dtype="bf16", out_dtype="bf16"), hardware=SKU)
    fp32 = simulate_aclnn_call(
        _sigmoid_request(dtype="fp32", out_dtype="fp32"), hardware=SKU)
    assert bf16.status == fp32.status == "predicted"
    assert bf16.binding.call_id != fp32.binding.call_id
    assert bf16.binding.typed["signature_hash"] != fp32.binding.typed[
        "signature_hash"]
    assert bf16.binding.member_op_ids == ()
    assert bf16.program["caller_plan"]["members"] == []
    assert bf16.program["physical"]["physical_members"] == ["sigmoid"]
    assert bf16.occurrence["members"] == []


def test_vector_unknown_or_wrong_caller_is_no_match_and_replay_rejects():
    for conditions in (
            {"variant": "no_such_variant"},
            {"owner": "lm_head_sampler:elementwise:final_norm"},
            {"members": ["lm_head_sampler:elementwise:final_norm"]}):
        request = _sigmoid_request(conditions=conditions)
        result = simulate_aclnn_call(request, hardware=SKU)
        assert result.status == "no_match", (conditions, result.reasons)
        assert result.prediction is None
        assert result.replay["conditions"] == conditions
        assert simulate_aclnn_call(
            result.replay, hardware=SKU).status == "no_match"


def test_vector_layer_flows_to_occurrence_plan_and_timing_free_view():
    request = _sigmoid_request(conditions={"layer": 7})
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted", result.reasons
    assert result.occurrence["layer"] == 7
    assert result.program["caller_plan"]["layer"] == 7
    assert result.program["cross_view"]["request_conditions"]["layer"] == 7
    payload = result.to_dict()
    assert payload["occurrence"]["layer"] == 7
    assert payload["program"]["caller_plan"]["layer"] == 7
    view = native_abi_timing_free(result)
    assert view["program"]["caller_plan"]["layer"] == 7
    assert view["binding"] == result.binding.to_dict()


def _real_t18_qbmm_abi():
    """The four captured T18 QBMM inputs plus the BF16 output slot."""
    return {
        "inputs": [
            {"shape": [4, 8704], "dtype": "int8", "layout": "ND",
             "present": True, "quality": "tensor"},
            {"shape": [160, 544, 16, 32], "dtype": "int8",
             "layout": "FRACTAL_NZ", "present": True,
             "quality": "tensor"},
            {"shape": [5120], "dtype": "bf16", "layout": "ND",
             "present": True, "quality": "tensor"},
            {"shape": [4], "dtype": "fp32", "layout": "ND",
             "present": True, "quality": "tensor"},
        ],
        "outputs": [
            {"shape": [4, 5120], "dtype": "bf16", "layout": "ND",
             "present": True, "quality": "tensor"},
        ],
    }


def test_t18_qbmm_four_slot_real_scale_geometry():
    abi = _real_t18_qbmm_abi()
    # T02/T16 have not supplied a CANN source/image for this capture.  The API
    # version is explicit; the binding must stay candidate, not claim R31.
    invocation = native_qbmm_w8a8_invocation(
        abi, version={"api": "cann-9.2.0"},
        conditions={"phase": "decode"})
    assert set(invocation["tensors"]) == {
        "x", "weight_nz", "weight_scale", "pertoken_scale", "output"}
    result = simulate_aclnn_call(
        invocation, hardware="Ascend950DT_9582", phase="decode")
    assert result.status == "predicted", result.reasons
    assert result.binding_bound is False
    assert result.binding.source is None
    assert result.source_establishment["established"] is False
    work = result.prediction["work"]["bytes"]
    assert work["total"] == 44_650_512
    assert work["weight_scale"] == 10_240
    assert work["pertoken_scale"] == 16
    assert work["output"] == 40_960
    epilogue = result.prediction["work"].get("epilogue", {})
    assert epilogue["column_scale_bytes"] == 10_240
    assert epilogue["row_scale_bytes"] == 16
    typed = result.binding.typed
    assert typed["shapes"]["weight_nz"] == [160, 544, 16, 32]
    assert typed["tensors"]["weight_nz"]["layout"] == "FRACTAL_NZ"
    assert typed["tensors"]["weight_scale"]["dtype"] == "bf16"
    assert typed["tensors"]["weight_scale"]["shape"] == [5120]
    assert typed["tensors"]["pertoken_scale"]["dtype"] == "fp32"
    assert typed["tensors"]["pertoken_scale"]["shape"] == [4]
    assert typed["tensors"]["output"]["shape"] == [4, 5120]
    assert result.program["physical"]["work"]["bytes"]["total"] == 44_650_512
    assert result.program["physical"]["selected_call"]["entry"] == (
        "aclnnQuantMatmulWeightNz")
    assert result.operator_plan is not None
    plan_meta = dict(result.operator_plan.metadata).get("operator_plan") or {}
    assert plan_meta["kernels"] == [
        {"stage": "load", "owner": "mte2"},
        {"stage": "compute", "owner": "cube"},
        {"stage": "scales", "owner": "mte2"},
        {"stage": "epilogue", "owner": "vector"},
        {"stage": "store", "owner": "mte3"},
    ]
    assert plan_meta["preparation"] is None
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert payload["binding"]["call_id"] == result.binding.call_id
    assert result.binding.call_id


def test_swiglu_dim0_uses_api_axis_and_rejects_wrong_output():
    request = {
        "api": {"namespace": "torch", "entry": "torch_npu.npu_swiglu"},
        "version": R31_VERSION,
        "tensors": {
            "input": {"shape": [4, 32], "dtype": "bf16"},
            "output": {"shape": [2, 32], "dtype": "bf16"},
        },
        "attrs": {"dim": 0},
        "conditions": {},
    }
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted", result.reasons
    assert result.binding.member_op_ids == ()
    assert result.program["physical"]["value_map"]["split_axis"] == 0
    assert result.program["physical"]["value_map"]["execution_input_shape"] == [
        2, 64]
    assert result.program["physical"]["value_map"]["output_shape"] == [2, 32]
    assert result.program["physical"]["execution_view"] == [2, 64]
    assert result.prediction["logical_value_bytes"] == {
        "input": 256, "output": 128}
    assert result.prediction["work"]["mte2"] == "256"
    assert result.prediction["work"]["mte3"] == "128"
    assert result.binding.typed["source_shapes"]["input"] == [4, 32]
    assert result.binding.typed["source_shapes"]["output"] == [2, 32]
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert json.loads(json.dumps(payload))["prediction"]["program"] == (
        json.loads(json.dumps(result.program)))
    view = native_abi_timing_free(result)
    assert view["program"]["physical"]["value_map"] == result.program[
        "physical"]["value_map"]

    wrong = dict(request)
    wrong["tensors"] = {
        "input": {"shape": [4, 32], "dtype": "bf16"},
        "output": {"shape": [4, 16], "dtype": "bf16"},
    }
    failed = simulate_aclnn_call(wrong, hardware=SKU)
    assert failed.status == "no_match"
    assert "native_vector_shape_mismatch:output" in " ".join(failed.reasons)
    assert failed.prediction is None
    assert failed.replay["tensors"]["output"]["shape"] == [4, 16]
    assert simulate_aclnn_call(failed.replay, hardware=SKU).status == "no_match"


def test_rmsnorm_high_order_preserves_original_rstd_and_execution_view():
    request = {
        "api": {"namespace": "torch", "entry": "torch_npu.npu_rms_norm"},
        "version": R31_VERSION,
        "tensors": {
            "x": {"shape": [4, 3, 16], "dtype": "bf16"},
            "weight": {"shape": [16], "dtype": "bf16"},
            "output": {"shape": [4, 3, 16], "dtype": "bf16"},
            "rstd": {"shape": [4, 3, 1], "dtype": "fp32"},
        },
        "attrs": {"epsilon": 1e-6},
        "conditions": {},
    }
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted", result.reasons
    assert result.binding.member_op_ids == ()
    assert result.binding.typed["shapes"] == {
        "x": [4, 3, 16], "weight": [16],
        "output": [4, 3, 16], "rstd": [4, 3, 1]}
    assert result.program["physical"]["execution_view"] == [12, 16]
    assert result.program["physical"]["value_map"]["original_rstd_shape"] == [
        4, 3, 1]
    assert result.program["physical"]["value_map"]["execution_rstd_shape"] == [
        12, 1]
    assert result.prediction["logical_value_bytes"] == {
        "x": 384, "weight": 32, "output": 384, "rstd": 48}
    payload = result.to_dict()
    assert payload["program"] == payload["prediction"]["program"]
    assert json.loads(json.dumps(payload))["program"] == json.loads(
        json.dumps(result.program))
    assert payload["replay_minimal"]["tensors"]["rstd"]["shape"] == [
        4, 3, 1]
    view = native_abi_timing_free(result)
    assert view["program"]["physical"]["value_map"] == result.program[
        "physical"]["value_map"]


def test_t11_selection_delivery_binds_same_plan_and_handles():
    op = OpSpec(
        "s0.l0.down", "matmul", "decode", 0, 0, ("analysis0:0",),
        (4, 5120, 8704), (4, 5120, 8704), (16, 5120, 8704),
        (4, 5120, 8704), dtype="int8", layout="FRACTAL_NZ",
        recipe="down_projection",
        metadata=(("implementation", "selected_down"),),
    )
    declared = declare_native_call(
        op,
        api={"namespace": "torch", "entry": "torch_npu.npu_quant_matmul"},
        source={"repository": "cann", "path": "opapi.so",
                "sha256": "a" * 64, "symbol": "aclnnQuantMatmulWeightNz",
                "revision": "r31", "state": "task_artifact"},
        kernels=[{"name": "QuantBatchMatmulV3", "role": "mix",
                  "evidence": "source_declared"}],
    )
    plan = OperatorPlan(
        Work(matrix={"int8": 1_000}, resources={"hbm": 44_650_512}),
        lambda context: None)
    delivery = {
        "selector": "q_a",
        "selected": {"implementation": "selected_down"},
        "program": {"kind": "declared_software_plan", "recipes": ["down"]},
        "producer_outputs": [
            {"producer": "s0.l0.down", "output": "out", "dtype": "bf16",
             "storage_shape": [4, 5120], "materialized": True,
             "state_version": "v1", "binding": "b0"}],
        "executions": [
            {"producer": "s0.l0.down", "family": "matmul",
             "recipe": "down_projection", "deps": ["s0.prefix"],
             "stream": "main", "graph_segment": "eager"}],
        "public_roots": [["s0.l0.down", "out"]],
        "fragment": {"id": "frag-0"},
    }
    selected_op = replace(
        declared, metadata=declared.metadata + ((
            "implementation_selections",
            ({"selector": "q_a", "selected": delivery["selected"],
              "plan": delivery["program"], "delivery": delivery},)),))
    automatic = bind_native_call(selected_op, plan=plan)
    assert automatic.typed["selection"]["selected"]["implementation"] == (
        "selected_down")

    direct = bind_native_call(declared, plan=plan)
    bound = bind_selected_native_call(
        declared, plan=plan, delivery=delivery,
        input_handles=("x", "weight_nz", "weight_scale", "pertoken_scale"),
        output_handles=("out",))
    assert bound is not None
    assert bound.call_id == direct.call_id
    assert bound.typed["selection"]["selected"]["implementation"] == (
        "selected_down")
    assert bound.typed["selection"]["producer_outputs"][0]["producer"] == (
        "s0.l0.down")
    assert bound.typed["selection"]["public_roots"] == (
        ["s0.l0.down", "out"],)
    assert bound.physical["selection_program"] == delivery["program"]
    assert bound.physical["input_handles"] == [
        "x", "weight_nz", "weight_scale", "pertoken_scale"]
    assert bound.physical["output_handles"] == ["out"]
    assert bound.physical["matrix"] == {"int8": 1_000}
    assert bound.physical["resources"] == {"hbm": 44_650_512}
    assert bound.occurrence["selection"] == bound.typed["selection"]


def test_t11_selection_projects_into_independent_abi_result():
    selection = {
        "selector": "q_a", "selected": {"implementation": "selected_down"},
        "program": {"recipes": ["down"]},
        "producer_outputs": [{"producer": "s0.l0.down", "output": "out"}],
        "public_roots": [["s0.l0.down", "out"]],
    }
    request = _sigmoid_request()
    plain = simulate_aclnn_call(request, hardware=SKU)
    result = simulate_aclnn_call(request, hardware=SKU, selection=selection)
    assert result.binding.call_id == plain.binding.call_id
    assert result.program["selection"] == selection
    assert result.prediction["selection"] == selection
    assert result.occurrence["selection"] == selection
    assert result.replay["selection"] == selection
    assert result.binding.typed["selection"] == selection
    assert result.binding.physical["selection"] == selection
    assert result.program is result.prediction["program"]
    payload = result.to_dict()
    assert payload["program"]["selection"] == selection
    assert payload["occurrence"]["selection"] == selection


def test_vector_execution_program_submits_to_caller_execution_sync_and_async():
    from test_native_vector_leaf_execution import _execution
    result = simulate_aclnn_call(_sigmoid_request(), hardware=SKU)
    assert result.status == "predicted"
    assert result.execution_program is not None
    execution = _execution()
    receipt = submit_native_call(
        result, execution, name="sigmoid", blocking=True)
    assert receipt["duration_ms"] is not None
    assert receipt["duration_ms"] >= 0
    assert receipt["handle"].done
    assert receipt["call_id"] == result.binding.call_id

    async_execution = _execution()
    pending = submit_native_call(
        result, async_execution, name="sigmoid", blocking=False)
    assert not pending["handle"].done
    async_execution.engine.advance(until=(pending["terminal_event"],))
    assert pending["handle"].done


def test_matrix_version_conflict_and_wrapper_commit_are_no_match():
    conflict = _matrix_request(
        "aclnnMatmul", "aclnn",
        {"x1": {"shape": [257, 5120], "dtype": "bf16", "role": "input"},
         "x2": {"shape": [17408, 5120], "dtype": "bf16", "role": "input"},
         "output": {"shape": [257, 17408], "dtype": "bf16",
                    "role": "output"}},
        {"transpose_x1": False, "transpose_x2": True}, R31)
    conflict["api"]["version"] = "cann-8.0.0"
    failed = simulate_aclnn_call(conflict, hardware=SKU)
    assert failed.status in {"no_match", "version_mismatch"}
    assert failed.binding is None
    assert failed.prediction is None
    assert failed.replay["api"].get("version") == "cann-8.0.0"

    wrapper = _matrix_request(
        "aclnnQuantMatmulWeightNz", "aclnn",
        {"x": {"shape": [4, 8704], "dtype": "int8", "role": "input"},
         "weight_nz": {"shape": [160, 544, 16, 32], "dtype": "int8",
                       "role": "weight", "layout": "FRACTAL_NZ"},
         "weight_scale": {"shape": [5120], "dtype": "bf16",
                          "role": "scale"},
         "pertoken_scale": {"shape": [4], "dtype": "fp32",
                            "role": "scale"},
         "output": {"shape": [4, 5120], "dtype": "bf16",
                    "role": "output"}},
        {"weight_scale_storage_format": "bf16"},
        {"api": "cann-9.2.0", "source": "6e3299ab492ddbadb4c8e3a00b88c982c9a28131"})
    wrapper_failed = simulate_aclnn_call(wrapper, hardware=SKU)
    assert wrapper_failed.status == "no_match"
    assert wrapper_failed.binding is None
    assert wrapper_failed.prediction is None
    assert "version:source_not_r31" in wrapper_failed.reasons


def test_known_matrix_mechanism_without_capture_certificate_still_predicts():
    request = _matrix_request(
        "aclnnMatmul", "aclnn",
        {"x1": {"shape": [257, 5120], "dtype": "bf16", "role": "input"},
         "x2": {"shape": [17408, 5120], "dtype": "bf16", "role": "input"},
         "output": {"shape": [257, 17408], "dtype": "bf16",
                    "role": "output"}},
        {"transpose_x1": False, "transpose_x2": True}, R31)
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted"
    assert result.binding is not None
    assert result.prediction["fidelity"] == "timing_free_work"
    assert result.prediction["work"]


def test_t18_qbmm_api_version_and_top_version_are_equivalent():
    abi = _real_t18_qbmm_abi()
    top = native_qbmm_w8a8_invocation(
        abi, version={"api": "cann-9.2.0"}, conditions={"phase": "decode"})
    api_version = native_qbmm_w8a8_invocation(
        abi, version={}, conditions={"phase": "decode"})
    api_version["api"]["version"] = "cann-9.2.0"
    first = simulate_aclnn_call(top, hardware="Ascend950DT_9582", phase="decode")
    second = simulate_aclnn_call(
        api_version, hardware="Ascend950DT_9582", phase="decode")
    assert first.status == second.status == "predicted"
    assert first.binding.call_id == second.binding.call_id
    assert first.prediction["work"]["bytes"]["total"] == 44_650_512
    assert second.prediction["work"]["bytes"]["total"] == 44_650_512
    assert first.binding_bound == second.binding_bound is False
