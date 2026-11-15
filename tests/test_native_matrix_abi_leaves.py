"""Independent ABI contracts for the five native matrix leaves.

[Role] Prove that a real tensor/attribute/version record alone selects one
  matrix call and yields formula work/geometry, without a compiled model, a
  device, a weight load, a sample-shape table or a target duration.
[Invariants] Work scales with the declared geometry; grouped work is the sum
  of the declared group rows; scale, metadata and accumulator stay separate
  fields; unsupported/missing ABI fields return an explicit gap and no
  fallback prediction or alternative algorithm.
[Timeouts & Locks] Pure CPU arithmetic over small dicts.
[Side-Effect Policy] Read-only tests; no allocation, device context or I/O.
"""
from __future__ import annotations

import json

from mindie_sim.prediction.piecewise import (
    native_grouped_matmul_swiglu_quant_v2 as gmm1,
    native_grouped_matmul_v4 as gmm2,
    native_qbmm as qbmm,
    native_r31_matmul_bf16 as matmul,
    native_r31_quantmatmul_weight_nz as qmmnz,
)
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise.native_invocation import simulate_aclnn_call
from mindie_sim.prediction.piecewise.native_matrix_abi import (
    build_operator_plan, execute_operator_plan, operator_plan_descriptor)

LEAVES = (gmm2, gmm1, qbmm, matmul, qmmnz)
R31 = {"api": "cann-9.2.0", "source": "r31",
       "image": "sha256:" + matmul.R31_IMAGE_SHA}
R64 = {"api": "cann-9.2.1", "source": "r64",
       "image": "sha256:" + gmm2.R64_IMAGE_SHA}


def _assert_predicted(result):
    assert result["status"] == "predicted"
    assert result["native_call_ready"] is True
    assert result["fallback_performed"] is False
    assert result["gap"] == []
    assert result["selected_call"]["entry"] == result["entry"]
    assert result["selected_call"]["signature_hash"]
    assert "duration_ms" not in result
    assert "duration_us" not in result["work"]


def _assert_gap(result, prefix):
    assert result["status"] in {"insufficient_fields", "no_match",
                                "unmodeled_abi"}
    assert result["selected_call"] is None
    assert result["work"] is None
    assert result["geometry"] is None
    assert result["fallback_performed"] is False
    assert any(reason.startswith(prefix) for reason in result["gap"])


def _qbmm_tensors(m: int, k: int = 5120, n: int = 8192,
                  bias: bool = False):
    pairs = ((k + 31) // 32 + 1) // 2
    tensors = {
        "x": {"shape": [m, k], "dtype": "fp8_e4m3", "role": "input"},
        "weight": {"shape": [k, n], "dtype": "fp8_e4m3", "role": "weight"},
        "weight_scale": {"shape": [pairs, n, 2], "dtype": "uint8",
                         "role": "scale"},
        "pertoken_scale": {"shape": [m, pairs, 2], "dtype": "fp8_e8m0",
                           "role": "scale"},
        "output": {"shape": [m, n], "dtype": "bf16", "role": "output"},
    }
    if bias:
        tensors["bias"] = {"shape": [n], "dtype": "bf16", "role": "input"}
    return tensors


def _grouped_tensors(rows, counts, experts=16, n=6144, k=2048,
                     swiglu=False):
    pairs = ((k + 31) // 32 + 1) // 2
    if swiglu:
        width = n // 2
        out_pairs = ((width + 31) // 32 + 1) // 2
        return {
            "x": {"shape": [rows, k], "dtype": "fp4_e2m1", "role": "input"},
            "x_scale": {"shape": [rows, pairs, 2], "dtype": "fp8_e8m0",
                        "role": "scale"},
            "group_list": {"shape": [experts], "dtype": "int64",
                           "role": "metadata", "values": list(counts)},
            "weight": {"shape": [experts, n, k], "dtype": "fp4_e2m1",
                       "role": "weight"},
            "weight_scale": {"shape": [experts, n, pairs, 2],
                             "dtype": "fp8_e8m0", "role": "scale"},
            "output": {"shape": [rows, width], "dtype": "fp4_e2m1",
                       "role": "output"},
            "output_scale": {"shape": [rows, out_pairs, 2],
                             "dtype": "fp8_e8m0", "role": "scale"},
        }
    return {
        "x": {"shape": [rows, k], "dtype": "fp4_e2m1", "role": "input"},
        "weight": {"shape": [experts, n, k], "dtype": "fp4_e2m1",
                   "role": "weight"},
        "scale": {"shape": [experts, n, pairs, 2], "dtype": "fp8_e8m0",
                  "role": "scale"},
        "per_token_scale": {"shape": [rows, pairs, 2],
                            "dtype": "fp8_e8m0", "role": "scale"},
        "group_list": {"shape": [experts], "dtype": "int64",
                       "role": "metadata", "values": list(counts)},
        "output": {"shape": [rows, n], "dtype": "bf16", "role": "output"},
    }


def test_contracts_are_declared_without_a_shape_table():
    entries = {leaf.abi_contract()["entry"] for leaf in LEAVES}
    assert entries == {
        "torch_npu.npu_grouped_matmul",
        "torch_npu.npu_grouped_matmul_swiglu_quant_v2",
        "torch_npu.npu_quant_matmul",
        "aclnnMatmul",
        "aclnnQuantMatmulWeightNz",
    }
    for leaf in LEAVES:
        contract = leaf.abi_contract()
        assert contract["schema"] == "mindie-sim/native-matrix-abi/1"
        assert contract["tensors"]
        assert "duration" not in str(contract).lower()
        assert contract["relation"]


def test_qbmm_m_is_a_formula_and_scale_is_separate_from_accumulator():
    k, n = 5120, 8192
    signatures = set()
    for m in (1, 2, 17, 257):
        result = qbmm.predict_abi(_qbmm_tensors(m),
                                  attrs={"group_size": 32}, version=R64)
        _assert_predicted(result)
        assert result["work"]["cube"]["macs"] == m * n * k
        assert result["work"]["cube"]["flops"] == 2 * m * n * k
        assert result["geometry"]["m"] == m
        assert result["geometry"]["k"] == k
        assert result["geometry"]["n"] == n
        assert result["scale"]["weight_scale"]["dtype"] == "uint8"
        assert result["scale"]["pertoken_scale"]["dtype"] == "fp8_e8m0"
        assert result["accumulator"]["dtype"] == "fp32"
        assert result["accumulator"]["abi_tensor"] is False
        assert result["accumulator"]["bytes"] == m * n * 4
        assert result["metadata"]["group_size"] == 32
        total = result["work"]["bytes"]
        assert total["total"] == sum(
            total[name] for name in (
                "input", "weight", "weight_scale", "pertoken_scale",
                "output", "bias"))
        assert result["tensors"]["weight"]["views"]["storage"]["shape"] == [
            n, k]
        assert result["tensors"]["weight"]["views"]["device"]["layout"] == \
            "FRACTAL_NZ"
        signatures.add(result["selected_call"]["signature_hash"])
    assert len(signatures) == 1
    bias = qbmm.predict_abi(_qbmm_tensors(17, bias=True),
                            attrs={"group_size": 32}, version=R64)
    assert bias["work"]["bytes"]["bias"] == n * 2
    assert bias["work"]["vector"]["bias_ops"] == 17 * n
    assert bias["selected_call"]["signature_hash"] not in signatures


def test_qbmm_explicit_gap_has_no_algorithm_or_envelope_fallback():
    missing = _qbmm_tensors(17)
    del missing["pertoken_scale"]
    gap = qbmm.predict_abi(missing, version=R64)
    _assert_gap(gap, "missing:tensors.pertoken_scale")
    assert gap["reasons"]
    assert gap["native_call_ready"] is False

    unknown = qbmm.predict_abi(
        _qbmm_tensors(17), version={"image": "sha256:" + "f" * 64})
    _assert_gap(unknown, "version:")
    assert unknown["selected_call"] is None


def test_grouped_v4_work_is_the_declared_group_sum_and_regions_match():
    experts, n, k = 16, 6144, 2048
    counts = (2, 0, 3) + (0,) * (experts - 3)
    first = gmm2.predict_abi(
        _grouped_tensors(5, counts), attrs={"group_list_type": 1},
        version=R64)
    _assert_predicted(first)
    assert first["work"]["cube"]["macs"] == 5 * n * k
    assert first["geometry"]["group_rows"] == list(counts)
    assert len(first["regions"]) == experts
    assert sum(region["macs"] for region in first["regions"]) == \
        first["work"]["cube"]["macs"]
    assert [region["rows"] for region in first["regions"][:3]] == [2, 0, 3]

    # Same total rows, different distribution: logical work is equal while
    # the per-expert regions are preserved.
    second = gmm2.predict_abi(
        _grouped_tensors(5, (1, 1, 3) + (0,) * (experts - 3)),
        attrs={"group_list_type": 1}, version=R64)
    assert second["work"]["cube"]["macs"] == first["work"]["cube"]["macs"]
    assert [region["rows"] for region in second["regions"][:3]] == [1, 1, 3]
    assert second["selected_call"]["signature_hash"] == \
        first["selected_call"]["signature_hash"]

    cumulative = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)
    third = gmm2.predict_abi(
        _grouped_tensors(5, cumulative), attrs={"group_list_type": 0},
        version=R64)
    assert third["status"] == "predicted"
    assert third["geometry"]["group_rows"] == list(counts)
    assert third["metadata"]["group_list"]["values"] == cumulative
    assert third["metadata"]["group_list"]["counts"] == list(counts)
    assert third["metadata"]["group_list"]["canonical_cumulative"] == cumulative


def test_grouped_v4_symbolic_or_inconsistent_group_list_is_a_gap():
    no_values = _grouped_tensors(5, (2, 0, 3) + (0,) * 13)
    del no_values["group_list"]["values"]
    _assert_gap(gmm2.predict_abi(no_values, attrs={"group_list_type": 1},
                                 version=R64),
                "missing:tensors.group_list.values")

    wrong = _grouped_tensors(5, (1, 2) + (0,) * 14)
    _assert_gap(gmm2.predict_abi(wrong, attrs={"group_list_type": 1},
                                 version=R64),
                "mismatch:tensors.group_list.sum")


def test_grouped_swiglu_preserves_two_outputs_and_activation_work():
    rows, experts, k, n = 32, 16, 6144, 8192
    width = n // 2
    counts = (2,) * experts
    cumulative = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)
    result = gmm1.predict_abi(
        _grouped_tensors(rows, cumulative, experts, n, k, swiglu=True),
        attrs={"group_list_type": 0}, version=R64)
    assert result["status"] == "predicted"
    assert result["work"]["cube"]["macs"] == rows * n * k
    assert result["work"]["vector"]["ops"] == 3 * rows * width
    assert result["work"]["vector"]["silu_ops"] == rows * width
    assert result["metadata"]["outputs"] == ["out", "out_scale"]
    assert result["metadata"]["output_arity"] == 2
    assert result["geometry"]["group_list_type_consumed"] == 0
    assert result["geometry"]["cumulative_group_list"][-1] == rows

    # A caller type-1 list is preserved and normalized, not rejected.
    counts_input = {**result["invocation"]["tensors"],
                    "group_list": {"shape": [experts], "dtype": "int64",
                                   "role": "metadata", "values": list(counts)}}
    from_counts = gmm1.predict_abi(counts_input,
                                   attrs={"group_list_type": 1},
                                   version=R64)
    assert from_counts["status"] == "predicted"
    assert from_counts["geometry"]["cumulative_group_list"] == cumulative
    assert from_counts["selected_call"]["signature_hash"] == \
        result["selected_call"]["signature_hash"]


def test_bf16_matmul_orientation_and_square_ambiguity():
    m, k, n = 257, 5120, 17408
    request = {
        "x1": {"shape": [m, k], "dtype": "bf16", "role": "input"},
        "x2": {"shape": [n, k], "dtype": "bf16", "role": "input"},
        "output": {"shape": [m, n], "dtype": "bf16", "role": "output"},
    }
    result = matmul.predict_abi(
        request, attrs={"transpose_x1": False, "transpose_x2": True},
        version=R31)
    _assert_predicted(result)
    assert result["work"]["cube"]["macs"] == m * n * k
    assert result["geometry"]["weight_storage_shape"] == [n, k]
    assert result["tensors"]["x2"]["views"]["logical"]["shape"] == [k, n]
    assert result["tensors"]["x2"]["views"]["storage"]["shape"] == [n, k]
    assert result["accumulator"]["dtype"] == "fp32"

    # Same ABI without transpose attrs: unique shapes infer the orientation.
    inferred = matmul.predict_abi(request, version=R31)
    assert inferred["status"] == "predicted"
    assert inferred["geometry"]["transpose_x2"] is True

    # A square operand pair cannot choose an orientation without attributes.
    square = {
        "x1": {"shape": [8, 8], "dtype": "bf16"},
        "x2": {"shape": [8, 8], "dtype": "bf16"},
        "output": {"shape": [8, 8], "dtype": "bf16"},
    }
    _assert_gap(matmul.predict_abi(square, version=R31),
                "missing:attrs.transpose_x1")


def test_weight_nz_keeps_logical_nz_scale_and_accumulator_distinct():
    m, k, n = 17, 5120, 17408
    logical = {
        "x": {"shape": [m, k], "dtype": "int8", "role": "input"},
        "weight": {"shape": [n, k], "dtype": "int8", "role": "weight"},
        "weight_scale": {"shape": [n], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [m], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [m, n], "dtype": "bf16", "role": "output"},
    }
    result = qmmnz.predict_abi(
        logical, attrs={"weight_scale_storage_format": "fp32"}, version=R31)
    _assert_predicted(result)
    assert result["work"]["cube"]["macs"] == m * n * k
    assert result["geometry"]["weight_nz"] == [n // 32, k // 16, 16, 32]
    assert result["metadata"]["weight_allocation_shape"] is None
    assert result["scale"]["weight_scale"]["storage"]["format"] == "fp32"
    assert result["scale"]["weight_scale"]["storage"]["bytes"] == n * 4
    assert result["scale"]["weight_scale"]["runtime"]["format"] == "bf16"
    assert result["scale"]["weight_scale"]["runtime"]["bytes"] == n * 2
    assert result["scale"]["pertoken_scale"]["dtype"] == "fp32"
    assert result["accumulator"]["dtype"] == "int32"
    assert result["accumulator"]["abi_tensor"] is False

    nz = dict(logical)
    del nz["weight"]
    nz["weight_nz"] = {"shape": [n // 32, k // 16, 16, 32],
                       "dtype": "int8", "role": "weight",
                       "layout": "FRACTAL_NZ"}
    from_nz = qmmnz.predict_abi(
        nz, attrs={"weight_scale_storage_format": "fp32"}, version=R31)
    assert from_nz["status"] == "predicted"
    assert from_nz["work"]["cube"]["macs"] == result["work"]["cube"]["macs"]
    assert result["work"]["preparation"] is not None
    assert result["work"]["preparation"]["producer"] == "maybe_trans_nz"
    assert from_nz["work"]["preparation"] is None
    assert from_nz["selected_call"]["signature_hash"] != \
        result["selected_call"]["signature_hash"]


def test_declared_matches_model_catalogue_where_the_entry_exists():
    for name, leaf in (("aclnnMatmul", matmul),
                       ("aclnnQuantMatmulWeightNz", qmmnz)):
        declared = native_calls.native_abi_spec(name)
        assert declared is not None
        request = {
            "x1": {"shape": [4, 5120], "dtype": "bf16", "role": "input"},
            "x2": {"shape": [8192, 5120], "dtype": "bf16",
                   "role": "input"},
            "output": {"shape": [4, 8192], "dtype": "bf16",
                       "role": "output"},
        } if name == "aclnnMatmul" else {
            "x": {"shape": [4, 5120], "dtype": "int8", "role": "input"},
            "weight": {"shape": [8192, 5120], "dtype": "int8",
                       "role": "weight"},
            "weight_scale": {"shape": [8192], "dtype": "bf16",
                            "role": "scale"},
            "pertoken_scale": {"shape": [4], "dtype": "fp32",
                               "role": "scale"},
            "output": {"shape": [4, 8192], "dtype": "bf16",
                       "role": "output"},
        }
        result = leaf.predict_abi(request, version=R31)
        assert result["status"] == "predicted"
        assert result["selected_call"]["entry"] == declared.entry
        assert result["selected_call"]["source"]["sha256"] == \
            declared.source["sha256"]


def test_unmodeled_abi_variants_are_explicit_gaps_not_substitutions():
    matmul_request = {
        "x1": {"shape": [4, 5120], "dtype": "bf16", "role": "input"},
        "x2": {"shape": [8192, 5120], "dtype": "bf16", "role": "input"},
        "output": {"shape": [4, 8192], "dtype": "bf16", "role": "output"},
    }
    cube_math = matmul.predict_abi(
        matmul_request,
        attrs={"transpose_x1": False, "transpose_x2": True,
               "cube_math_type": 1}, version=R31)
    _assert_gap(cube_math, "unmodeled:attrs.cube_math_type")

    matmul_bias = matmul.predict_abi(
        {**matmul_request,
         "bias": {"shape": [8192], "dtype": "bf16", "role": "input"}},
        attrs={"transpose_x1": False, "transpose_x2": True}, version=R31)
    _assert_gap(matmul_bias, "unmodeled:tensors.bias")

    grouped_bias = gmm2.predict_abi(
        {**_grouped_tensors(5, (5,) + (0,) * 15),
         "bias": {"shape": [16, 6144], "dtype": "bf16", "role": "input"}},
        attrs={"group_list_type": 1}, version=R64)
    _assert_gap(grouped_bias, "unmodeled:tensors.bias")

    swiglu_limit = gmm1.predict_abi(
        _grouped_tensors(32, tuple(range(2, 34, 2)), swiglu=True),
        attrs={"group_list_type": 0, "swiglu_limit": 1.0}, version=R64)
    _assert_gap(swiglu_limit, "unmodeled:attrs.swiglu_limit")

    qmm_bias = qmmnz.predict_abi(
        {**{"x": {"shape": [2, 5120], "dtype": "int8", "role": "input"},
            "weight": {"shape": [8192, 5120], "dtype": "int8",
                       "role": "weight"},
            "weight_scale": {"shape": [8192], "dtype": "bf16",
                             "role": "scale"},
            "pertoken_scale": {"shape": [2], "dtype": "fp32",
                               "role": "scale"},
            "output": {"shape": [2, 8192], "dtype": "bf16",
                       "role": "output"}},
         "bias": {"shape": [8192], "dtype": "bf16", "role": "input"}},
        version=R31)
    _assert_gap(qmm_bias, "unmodeled:tensors.bias")

    qbmm_unknown = qbmm.predict_abi(
        _qbmm_tensors(2), attrs={"group_size": 32, "custom_axis": 1},
        version=R64)
    _assert_gap(qbmm_unknown, "unmodeled:attrs:custom_axis")


def test_region_members_can_describe_a_fused_one_to_one_call():
    qbmm_request = _qbmm_tensors(4)
    single = qbmm.predict_abi(qbmm_request, version=R64)
    fused = qbmm.predict_abi(
        qbmm_request, version=R64,
        conditions={"members": ["linear", "residual"]})
    assert single["selected_call"]["relation"] == "one_to_one"
    assert fused["selected_call"]["relation"] == "many_to_one_fused"
    assert fused["selected_call"]["members"] == ["linear", "residual"]
    assert fused["regions"][0]["members"] == ["linear", "residual"]
    assert single["selected_call"]["signature_hash"] != \
        fused["selected_call"]["signature_hash"]


def _small_grouped(rows, counts, experts=2, n=64, k=64):
    return _grouped_tensors(rows, counts, experts=experts, n=n, k=k,
                            swiglu=False)


def _small_swiglu(rows, counts, experts=2, n=64, k=64):
    return _grouped_tensors(rows, counts, experts=experts, n=n, k=k,
                            swiglu=True)


def test_grouped_issued_work_is_the_sum_of_region_tails():
    # [1,1] has two independent 16-row tail blocks; [2,0] has one.  The
    # aggregate row count alone cannot produce the same issued work.
    one_one = gmm1.predict_abi(
        _small_swiglu(2, [1, 1]), attrs={"group_list_type": 1},
        version=R64)
    two_zero = gmm1.predict_abi(
        _small_swiglu(2, [2, 0]), attrs={"group_list_type": 1},
        version=R64)
    for result in (one_one, two_zero):
        assert result["status"] == "predicted"
        assert result["work"]["cube"]["issued_macs"] == sum(
            region["issued_macs"] for region in result["regions"])
    assert one_one["work"]["cube"]["issued_macs"] == 2 * 16 * 64 * 64
    assert two_zero["work"]["cube"]["issued_macs"] == 16 * 64 * 64
    assert one_one["work"]["cube"]["issued_macs"] == 131072
    assert two_zero["work"]["cube"]["issued_macs"] == 65536

    # The delivered 16x2-row SwiGLU example conserves work too.
    counts = (2,) * 16
    result = gmm1.predict_abi(
        _grouped_tensors(32, counts, swiglu=True),
        attrs={"group_list_type": 1}, version=R64)
    assert result["work"]["cube"]["issued_macs"] == sum(
        region["issued_macs"] for region in result["regions"])
    assert result["work"]["cube"]["issued_macs"] == 16 * (
        16 * 6144 * 2048)

    # GMM V4 follows the same region-sum rule.
    for counts, rows in (([1, 1], 2), ([2, 0], 2)):
        result = gmm2.predict_abi(
            _small_grouped(rows, counts), attrs={"group_list_type": 1},
            version=R64)
        assert result["work"]["cube"]["issued_macs"] == sum(
            region["issued_macs"] for region in result["regions"])


def test_grouped_valid_received_capacity_are_distinct():
    extents = {"valid_rows": 16, "received_rows": 32, "capacity_rows": 64}
    for leaf, request in (
            (gmm2, _small_grouped(32, [16, 0])),
            (gmm1, _small_swiglu(32, [16, 0]))):
        result = leaf.predict_abi(
            request, attrs={"group_list_type": 1}, version=R64,
            conditions=extents)
        assert result["status"] == "predicted"
        assert result["geometry"]["valid_rows"] == 16
        assert result["geometry"]["received_rows"] == 32
        assert result["geometry"]["interface_rows"] == 32
        assert result["geometry"]["capacity_rows"] == 64
        assert result["geometry"]["row_extent_explicit"] is True
        assert result["geometry"]["padding_rows"] == 16
        assert result["geometry"]["capacity_slack"] == 32
        assert result["metadata"]["rows"]["valid_rows"] == 16
        assert result["work"]["cube"]["issued_macs"] == sum(
            region["issued_macs"] for region in result["regions"])
        assert result["work"]["cube"]["issued_macs"] == 16 * 64 * 64
        # The interface/scale rows are the received rows, not the valid rows.
        assert result["geometry"]["m"] == 16
        if leaf is gmm2:
            assert result["tensors"]["per_token_scale"]["shape"][0] == 32
        else:
            assert result["tensors"]["x_scale"]["shape"][0] == 32

    bad = (
        {"valid_rows": 40, "received_rows": 32, "capacity_rows": 64},
        {"valid_rows": 16, "received_rows": 31, "capacity_rows": 64},
        {"valid_rows": 16, "received_rows": 32, "capacity_rows": 31},
    )
    for conditions in bad:
        for leaf, request in (
                (gmm2, _small_grouped(32, [16, 0])),
                (gmm1, _small_swiglu(32, [16, 0]))):
            result = leaf.predict_abi(
                request, attrs={"group_list_type": 1}, version=R64,
                conditions=conditions)
            assert result["status"] == "no_match"
            assert result["work"] is None


def test_weight_nz_boundary_is_660_bytes_and_checkpoint_is_upstream():
    request = {
        "x": {"shape": [1, 16], "dtype": "int8", "role": "input"},
        "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [32], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [1, 32], "dtype": "bf16", "role": "output"},
    }
    native = qmmnz.predict_abi(request, version=R31)
    _assert_predicted(native)
    assert native["work"]["bytes"]["input"] == 16
    assert native["work"]["bytes"]["weight"] == 512
    assert native["work"]["bytes"]["weight_scale"] == 64
    assert native["work"]["bytes"]["pertoken_scale"] == 4
    assert native["work"]["bytes"]["output"] == 64
    assert native["work"]["bytes"]["total"] == 660
    assert native["work"]["vector"]["ops"] == 4 * 1 * 32
    assert "quant_ops" not in native["work"]["vector"]
    assert native["work"]["epilogue"]["operations_per_output"] == 4
    assert native["work"]["epilogue"]["row_scale_bytes"] == 4
    assert native["work"]["epilogue"]["column_scale_bytes"] == 64
    assert native["work"]["call_boundary"]["issued_traffic"] is None
    assert native["scale"]["weight_scale"]["native_call_read_bytes"] == 64
    assert native["tensors"]["weight_nz"]["views"]["storage"]["layout"] == \
        "FRACTAL_NZ"

    checkpoint = qmmnz.predict_abi(
        request, attrs={"weight_scale_storage_format": "fp32",
                        "weight_scale_storage_bytes": 4},
        version=R31)
    _assert_predicted(checkpoint)
    assert checkpoint["work"]["bytes"]["total"] == 660
    assert checkpoint["work"]["vector"]["ops"] == 4 * 1 * 32
    assert checkpoint["scale"]["weight_scale"]["storage"]["bytes"] == 128
    assert checkpoint["work"]["checkpoint_or_upstream"][
        "included_in_call_footprint"] is False
    assert checkpoint["selected_call"]["signature_hash"] == \
        native["selected_call"]["signature_hash"]

    conflict = qmmnz.predict_abi(
        request, attrs={"weight_scale_storage_format": "fp32",
                        "weight_scale_storage_bytes": 2},
        version=R31)
    _assert_gap(conflict, "mismatch:attrs.weight_scale_storage_bytes")


def test_qbmm_scale_interpretation_attributes_are_consumed():
    valid = qbmm.predict_abi(
        _qbmm_tensors(17), attrs={"scale_dtype": "fp8_e8m0",
                                  "pertoken_scale_dtype": "e8m0"},
        version=R64)
    _assert_predicted(valid)
    assert valid["selected_call"]["attrs"]["scale_dtype"] == "fp8_e8m0"
    assert valid["selected_call"]["attrs"]["pertoken_scale_dtype"] == \
        "fp8_e8m0"
    for attrs in ({"scale_dtype": "fp32"},
                  {"pertoken_scale_dtype": "fp32"}):
        gap = qbmm.predict_abi(_qbmm_tensors(17), attrs=attrs, version=R64)
        _assert_gap(gap, "mismatch:attrs.")


def test_grouped_typed_scale_and_operand_attributes_are_consumed():
    valid4 = gmm2.predict_abi(
        _small_grouped(2, [2, 0]),
        attrs={"group_list_type": 1, "per_token_scale_dtype": "e8m0"},
        version=R64)
    assert valid4["status"] == "predicted"
    assert valid4["selected_call"]["attrs"]["per_token_scale_dtype"] == \
        "fp8_e8m0"
    bad4 = gmm2.predict_abi(
        _small_grouped(2, [2, 0]),
        attrs={"group_list_type": 1, "per_token_scale_dtype": "fp32"},
        version=R64)
    _assert_gap(bad4, "mismatch:attrs.per_token_scale_dtype")

    valid1 = gmm1.predict_abi(
        _small_swiglu(2, [2, 0]),
        attrs={"group_list_type": 1, "weight_scale_dtype": "e8m0",
               "x_scale_dtype": "fp8_e8m0"},
        version=R64)
    assert valid1["status"] == "predicted"
    for attrs in ({"weight_scale_dtype": "fp32"},
                  {"x_scale_dtype": "fp32"},
                  {"output_scale_dtype": "fp32"}):
        gap = gmm1.predict_abi(
            _small_swiglu(2, [2, 0]),
            attrs={"group_list_type": 1, **attrs}, version=R64)
        _assert_gap(gap, "mismatch:attrs.")


def test_swiglu_preserves_prepare_packed_storage_view():
    request = _small_swiglu(2, [2, 0])
    result = gmm1.predict_abi(request, attrs={"group_list_type": 1},
                              version=R64)
    _assert_predicted(result)
    weight = result["tensors"]["weight"]["views"]
    assert weight["logical"]["shape"] == [2, 64, 64]
    assert weight["storage"]["shape"] == [2, 32, 64]
    assert weight["storage"]["layout"] == "ND_TRANSPOSED"
    assert weight["interpretation"]["dtype"] == "fp4_e2m1"
    scale = result["tensors"]["weight_scale"]["views"]
    assert scale["logical"]["shape"] == [2, 64, 1, 2]
    assert scale["storage"]["shape"] == [2, 1, 64, 2]
    assert scale["storage"]["layout"] == "ND_TRANSPOSED"

    wrong = {**request,
             "weight": {**request["weight"],
                        "storage_shape": [2, 64, 32],
                        "storage_dtype": "uint8",
                        "storage_layout": "ND_transposed",
                        "interpret_dtype": "fp4_e2m1"}}
    _assert_gap(gmm1.predict_abi(wrong, attrs={"group_list_type": 1},
                                 version=R64),
                "mismatch:tensors.weight.storage_shape")


def test_qbmm_source_only_and_r35_replay_are_stable():
    for version in ({"source": "r64"}, {"source": "r35"},
                    {"image": "sha256:" + gmm2.R64_IMAGE_SHA}):
        result = qbmm.predict_abi(_qbmm_tensors(17), version=version)
        _assert_predicted(result)
        replay = qbmm.predict_abi(
            result["invocation"]["tensors"],
            attrs=result["invocation"]["attrs"],
            version=result["invocation"]["version"],
            conditions=result["invocation"]["conditions"])
        assert replay["status"] == "predicted"
        assert replay["selected_call"]["signature_hash"] == \
            result["selected_call"]["signature_hash"]


def test_all_leaves_replay_their_normalized_invocation():
    requests = {
        "matmul": (matmul, {
            "x1": {"shape": [4, 5120], "dtype": "bf16"},
            "x2": {"shape": [8192, 5120], "dtype": "bf16"},
            "output": {"shape": [4, 8192], "dtype": "bf16"}},
            {"transpose_x1": False, "transpose_x2": True},
            {"source": "r31"}),
        "qmmnz": (qmmnz, {
            "x": {"shape": [4, 5120], "dtype": "int8"},
            "weight": {"shape": [8192, 5120], "dtype": "int8"},
            "weight_scale": {"shape": [8192], "dtype": "bf16"},
            "pertoken_scale": {"shape": [4], "dtype": "fp32"},
            "output": {"shape": [4, 8192], "dtype": "bf16"}},
            None, {"source": "r31"}),
        "qbmm": (qbmm, _qbmm_tensors(4), None, {"source": "r64"}),
        "gmm4": (gmm2, _small_grouped(2, [2, 0]),
                 {"group_list_type": 1}, {"source": "r64"}),
        "gmm1": (gmm1, _small_swiglu(2, [2, 0]),
                 {"group_list_type": 1}, {"source": "r64"}),
    }
    for name, (leaf, tensors, attrs, version) in requests.items():
        result = leaf.predict_abi(tensors, attrs=attrs, version=version)
        _assert_predicted(result)
        replay = leaf.predict_abi(
            result["invocation"]["tensors"], attrs=result["invocation"]["attrs"],
            version=result["invocation"]["version"],
            conditions=result["invocation"]["conditions"])
        assert replay["status"] == "predicted", name
        assert replay["selected_call"]["signature_hash"] == \
            result["selected_call"]["signature_hash"], name


def test_shared_torch_api_int8_qbmm_keeps_bf16_scale_and_source_identity():
    # Synthetic M=120/K=N=5120 example retained only for the BF16 10,240 B
    # scale check; the real T02 row 120 replay is the next test.
    m, k, n = 120, 5120, 5120
    request = {
        "x": {"shape": [m, k], "dtype": "int8", "role": "input"},
        "weight": {"shape": [k, n], "dtype": "int8", "role": "weight"},
        "weight_scale": {"shape": [n], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [m], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [m, n], "dtype": "bf16", "role": "output"},
    }
    result = qbmm.predict_abi(request, version={"api": "cann-9.2.1"})
    _assert_predicted(result)
    assert result["selected_call"]["variant"] == "int8_shared_torch_api_nd"
    assert result["selected_call"]["source_revision"] is None
    assert result["selected_call"]["source"]["repository"] == "torch_npu"
    assert result["scale"]["weight_scale"]["runtime"]["bytes"] == n * 2
    assert result["scale"]["weight_scale"]["runtime"]["bytes"] == 10240
    assert result["work"]["bytes"]["weight_scale"] == 10240
    assert result["work"]["vector"]["ops"] == 4 * m * n
    assert result["work"]["epilogue"]["operations_per_output"] == 4
    assert result["work"]["epilogue"]["row_scale_bytes"] == m * 4
    assert result["work"]["epilogue"]["column_scale_bytes"] == n * 2
    assert result["accumulator"]["dtype"] == "int32"
    assert result["tensors"]["weight"]["views"]["storage"]["layout"] == "ND"

    nz_request = {
        "x": request["x"],
        "weight_nz": {"shape": [n // 32, k // 16, 16, 32],
                      "dtype": "int8", "role": "weight",
                      "layout": "FRACTAL_NZ"},
        "weight_scale": request["weight_scale"],
        "pertoken_scale": request["pertoken_scale"],
        "output": request["output"],
    }
    from_nz = qbmm.predict_abi(nz_request, version={"api": "cann-9.2.1"})
    _assert_predicted(from_nz)
    assert from_nz["work"]["cube"]["macs"] == result["work"]["cube"]["macs"]
    assert from_nz["selected_call"]["variant"] == "int8_shared_torch_api_nz"
    assert from_nz["selected_call"]["signature_hash"] != \
        result["selected_call"]["signature_hash"]

    checkpoint = qbmm.predict_abi(
        request,
        attrs={"weight_scale_storage_format": "fp32",
               "weight_scale_storage_bytes": 4},
        version={"api": "cann-9.2.1"})
    _assert_predicted(checkpoint)
    assert checkpoint["work"]["bytes"]["total"] == \
        result["work"]["bytes"]["total"]
    assert checkpoint["scale"]["weight_scale"]["storage"]["bytes"] == n * 4
    assert checkpoint["selected_call"]["signature_hash"] == \
        result["selected_call"]["signature_hash"]


def test_explicit_logical_view_is_preserved_and_validated():
    request = {
        "x": {"shape": [1, 16], "dtype": "int8", "role": "input"},
        "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ",
                      "logical_shape": [16, 32],
                      "logical_layout": "ND",
                      "storage_shape": [1, 1, 16, 32],
                      "storage_dtype": "int8",
                      "storage_layout": "FRACTAL_NZ",
                      "device_shape": [1, 1, 16, 32],
                      "device_dtype": "int8",
                      "device_layout": "FRACTAL_NZ",
                      "interpret_dtype": "int8"},
        "weight_scale": {"shape": [32], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [1, 32], "dtype": "bf16", "role": "output"},
    }
    result = qmmnz.predict_abi(request, version=R31)
    _assert_predicted(result)
    weight = result["tensors"]["weight_nz"]["views"]
    assert weight["logical"]["shape"] == [16, 32]
    assert weight["logical"]["dtype"] == "int8"
    assert weight["logical"]["layout"] == "ND"
    assert weight["logical"]["direction"] == "k_by_n"
    assert weight["storage"]["layout"] == "FRACTAL_NZ"
    assert weight["device"]["bytes"] == 16 * 32

    wrong = {**request, "weight_nz": {**request["weight_nz"],
                                      "logical_shape": [16, 31]}}
    _assert_gap(qmmnz.predict_abi(wrong, version=R31),
                "mismatch:tensors.weight_nz.logical_shape")


def test_real_t02_row120_qbmm_cpu_replay():
    # T02 frozen raw row 120 (kernel_details.csv line 120).  This is the real
    # captured slot geometry, not the synthetic M=120/K=N=5120 example.
    request = {
        "x": {"shape": [4, 8704], "dtype": "int8", "role": "input",
              "storage_shape": [4, 8704], "storage_dtype": "int8",
              "storage_layout": "ND", "device_shape": [4, 8704],
              "device_dtype": "int8", "device_layout": "ND"},
        "weight_nz": {"shape": [160, 544, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ",
                      "logical_shape": [8704, 5120],
                      "logical_layout": "ND",
                      "logical_direction": "k_by_n",
                      "storage_shape": [160, 544, 16, 32],
                      "storage_dtype": "int8",
                      "storage_layout": "FRACTAL_NZ",
                      "storage_bytes": 8704 * 5120,
                      "device_shape": [160, 544, 16, 32],
                      "device_dtype": "int8",
                      "device_layout": "FRACTAL_NZ",
                      "interpret_dtype": "int8"},
        "weight_scale": {"shape": [5120], "dtype": "bf16", "role": "scale",
                         "storage_shape": [5120],
                         "storage_dtype": "bf16",
                         "storage_layout": "ND",
                         "storage_bytes": 10240,
                         "interpret_dtype": "bf16"},
        "pertoken_scale": {"shape": [4], "dtype": "fp32", "role": "scale",
                           "storage_shape": [4], "storage_dtype": "fp32",
                           "storage_layout": "ND", "storage_bytes": 16,
                           "interpret_dtype": "fp32"},
        "output": {"shape": [4, 5120], "dtype": "bf16", "role": "output",
                   "storage_shape": [4, 5120], "storage_dtype": "bf16",
                   "storage_layout": "ND", "storage_bytes": 40960,
                   "interpret_dtype": "bf16"},
    }
    result = qbmm.predict_abi(request, version=None)
    _assert_predicted(result)
    assert result["work"]["bytes"]["total"] == 44_650_512
    assert result["work"]["bytes"]["input"] == 4 * 8704
    assert result["work"]["bytes"]["weight"] == 5120 * 8704
    assert result["work"]["bytes"]["weight_scale"] == 10_240
    assert result["work"]["bytes"]["pertoken_scale"] == 16
    assert result["work"]["bytes"]["output"] == 4 * 5120 * 2
    assert result["scale"]["weight_scale"]["runtime"]["bytes"] == 10_240
    assert result["scale"]["weight_scale"]["storage"]["status"] == \
        "not_declared"
    row_desc = operator_plan_descriptor(result)
    assert row_desc["m"] == 4 and row_desc["n"] == 5120 and \
        row_desc["k"] == 8704
    assert row_desc["matrix_builder"]["scales"]["weight_scale"][
        "runtime"]["bytes"] == 10_240
    assert row_desc["preparation"] is None
    assert result["work"]["epilogue"]["row_scale_bytes"] == 4 * 4
    assert result["work"]["epilogue"]["column_scale_bytes"] == 5_120 * 2
    assert result["work"]["epilogue"]["operations_per_output"] == 4
    assert result["work"]["vector"]["ops"] == 4 * 4 * 5120
    assert result["selected_call"]["variant"] == "int8_shared_torch_api_nz"
    assert result["selected_call"]["source"]["repository"] == "torch_npu"
    assert result["selected_call"]["source_revision"] is None
    assert result["selected_call"]["source"]["path"] is None
    assert result["geometry"]["weight_path"] == "pre_materialized_nz"
    assert result["work"]["preparation"] is None
    weight = result["tensors"]["weight_nz"]["views"]
    assert weight["logical"]["shape"] == [8704, 5120]
    assert weight["storage"]["layout"] == "FRACTAL_NZ"
    assert weight["device"]["layout"] == "FRACTAL_NZ"


def test_flat_and_nested_views_are_equivalent_and_conflicts_reject():
    flat = {
        "x": {"shape": [1, 16], "dtype": "int8", "role": "input"},
        "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [32], "dtype": "bf16", "role": "scale",
                         "storage_shape": [32], "storage_dtype": "bf16",
                         "storage_layout": "ND", "storage_bytes": 64,
                         "interpret_dtype": "bf16"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [1, 32], "dtype": "bf16", "role": "output"},
    }
    nested = {**flat,
              "weight_scale": {"shape": [32], "dtype": "bf16",
                               "role": "scale",
                               "views": {"logical": {"shape": [32],
                                                     "dtype": "bf16",
                                                     "layout": "ND"},
                                         "storage": {"shape": [32],
                                                     "dtype": "bf16",
                                                     "layout": "ND",
                                                     "bytes": 64},
                                         "interpretation": {
                                             "dtype": "bf16",
                                             "layout": "ND"}}}}
    a = qmmnz.predict_abi(flat, version=R31)
    b = qmmnz.predict_abi(nested, version=R31)
    _assert_predicted(a)
    _assert_predicted(b)
    assert a["work"]["bytes"]["total"] == 660
    assert b["work"]["bytes"]["total"] == 660
    assert a["selected_call"]["signature_hash"] == \
        b["selected_call"]["signature_hash"]

    conflict = {**nested,
                "weight_scale": {**nested["weight_scale"],
                                 "storage_dtype": "fp32"}}
    _assert_gap(qmmnz.predict_abi(conflict, version=R31),
                "mismatch:views.storage")

    wrong_direction = {**flat,
                       "weight_nz": {**flat["weight_nz"],
                                     "logical_direction": "n_by_k"}}
    _assert_gap(qmmnz.predict_abi(wrong_direction, version=R31),
                "mismatch:tensors.weight_nz.direction")

    mx = _qbmm_tensors(2)
    mx["weight_scale"] = {**mx["weight_scale"],
                          "interpret_dtype": "fp32"}
    _assert_gap(qbmm.predict_abi(mx, version=R64),
                "mismatch:tensors.weight_scale.interpret_dtype")

    wrong_flat = {**_small_swiglu(2, [2, 0]),
                  "weight": {**_small_swiglu(2, [2, 0])["weight"],
                             "storage_shape": [2, 64, 32],
                             "storage_dtype": "uint8",
                             "storage_layout": "ND_transposed",
                             "interpret_dtype": "fp4_e2m1"}}
    wrong_nested = {**_small_swiglu(2, [2, 0]),
                    "weight": {**_small_swiglu(2, [2, 0])["weight"],
                               "views": {"storage": {
                                   "shape": [2, 64, 32],
                                   "dtype": "uint8",
                                   "layout": "ND_TRANSPOSED"},
                                   "interpretation": {
                                       "dtype": "fp4_e2m1",
                                       "layout": "ND"}}}}
    flat_result = gmm1.predict_abi(wrong_flat,
                                   attrs={"group_list_type": 1},
                                   version=R64)
    nested_result = gmm1.predict_abi(wrong_nested,
                                     attrs={"group_list_type": 1},
                                     version=R64)
    assert flat_result["status"] == nested_result["status"] == "no_match"
    assert "mismatch:tensors.weight.storage_shape" in flat_result["gap"]
    assert "mismatch:tensors.weight.storage_shape" in nested_result["gap"]


def test_int8_epilogue_math_and_two_k_panels():
    # All-one operands: accumulator=K, output = K * row_scale * column_scale.
    base = {
        "x": {"shape": [1, 16], "dtype": "int8", "role": "input"},
        "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [32], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [1, 32], "dtype": "bf16", "role": "output"},
    }
    one_panel = qmmnz.predict_abi(base, version=R31)
    _assert_predicted(one_panel)
    epilogue = one_panel["work"]["epilogue"]
    assert 16 * 2 * 3 == 96
    assert epilogue["row_scale_bytes"] == 4
    assert epilogue["column_scale_bytes"] == 64
    assert epilogue["operations_per_output"] == 4
    assert one_panel["work"]["vector"]["ops"] == 4 * 1 * 32

    two_panel = {**base,
                 "x": {"shape": [1, 32], "dtype": "int8"},
                 "weight_nz": {"shape": [1, 2, 16, 32], "dtype": "int8",
                               "layout": "FRACTAL_NZ"},
                 "output": {"shape": [1, 32], "dtype": "bf16"}}
    two = qmmnz.predict_abi(two_panel, version=R31)
    _assert_predicted(two)
    assert two["work"]["cube"]["macs"] == 2 * one_panel["work"]["cube"]["macs"]
    assert two["work"]["epilogue"]["operations_per_output"] == 4
    assert two["work"]["vector"]["ops"] == 4 * 1 * 32

    i8 = qbmm.predict_abi(base, version=None)
    _assert_predicted(i8)
    assert i8["work"]["epilogue"]["row_scale_bytes"] == 4
    assert i8["work"]["epilogue"]["column_scale_bytes"] == 64
    assert i8["work"]["vector"]["ops"] == 4 * 1 * 32


def test_weight_nz_nd_requires_declared_packing_and_nz_does_not():
    nd = {
        "x": {"shape": [1, 32], "dtype": "int8", "role": "input"},
        "weight": {"shape": [32, 64], "dtype": "int8", "role": "weight",
                   "weight_layout": "ND"},
        "weight_scale": {"shape": [64], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [1, 64], "dtype": "bf16", "role": "output"},
    }
    from_nd = qmmnz.predict_abi(nd, version=R31)
    _assert_predicted(from_nd)
    assert from_nd["selected_call"]["variant"] == \
        "w8a8_dynamic_weight_nz_from_nd"
    assert from_nd["work"]["preparation"]["producer"] == "maybe_trans_nz"
    prep = from_nd["work"]["preparation"]
    assert prep["read_bytes"] == 32 * 64
    assert prep["write_bytes"] == 32 * 64
    assert prep["traffic_bytes"] == 2 * 32 * 64
    assert from_nd["tensors"]["weight_nz"]["provenance"]["origin"] == \
        "prepared"
    assert from_nd["tensors"]["weight_nz"]["provenance"]["output"]["slot"] == \
        "weight_nz"
    replay = qmmnz.predict_abi(
        json.loads(json.dumps(from_nd["tensors"])), version=R31)
    _assert_predicted(replay)
    assert replay["selected_call"]["variant"] == \
        "w8a8_dynamic_weight_nz_from_nd"
    assert replay["work"]["preparation"]["preparation_id"] == \
        prep["preparation_id"]
    assert replay["selected_call"]["signature_hash"] == \
        from_nd["selected_call"]["signature_hash"]
    assert from_nd["work"]["call_boundary"]["tensor_footprint_bytes"] == (
        32 + 32 * 64 + 64 * 2 + 4 + 64 * 2)

    pre = {
        "x": nd["x"],
        "weight_nz": {"shape": [2, 2, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ"},
        "weight_scale": nd["weight_scale"],
        "pertoken_scale": nd["pertoken_scale"],
        "output": nd["output"],
    }
    from_nz = qmmnz.predict_abi(pre, version=R31)
    _assert_predicted(from_nz)
    assert from_nz["work"]["preparation"] is None
    assert from_nz["selected_call"]["variant"] == \
        "w8a8_dynamic_weight_nz_pre_materialized"
    assert from_nz["work"]["call_boundary"]["tensor_footprint_bytes"] == \
        from_nd["work"]["call_boundary"]["tensor_footprint_bytes"]
    assert from_nz["selected_call"]["signature_hash"] != \
        from_nd["selected_call"]["signature_hash"]


def test_qbmm_int8_nd_and_pre_materialized_nz_are_distinct_plans():
    nd = {
        "x": {"shape": [1, 32], "dtype": "int8", "role": "input"},
        "weight": {"shape": [32, 64], "dtype": "int8", "role": "weight",
                   "storage_shape": [32, 64],
                   "storage_dtype": "int8", "storage_layout": "ND",
                   "device_shape": [32, 64],
                   "device_dtype": "int8", "device_layout": "ND"},
        "weight_scale": {"shape": [64], "dtype": "bf16", "role": "scale"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32", "role": "scale"},
        "output": {"shape": [1, 64], "dtype": "bf16", "role": "output"},
    }
    plan_nd = qbmm.predict_abi(nd, attrs={"weight_layout": "ND"},
                               version=None)
    _assert_predicted(plan_nd)
    assert plan_nd["selected_call"]["variant"] == "int8_shared_torch_api_nd"
    assert "weight_nz" not in plan_nd["tensors"]
    assert plan_nd["work"]["preparation"] is None
    assert plan_nd["tensors"]["weight"]["views"]["storage"]["layout"] == "ND"

    pre = {
        "x": nd["x"],
        "weight_nz": {"shape": [2, 2, 16, 32], "dtype": "int8",
                      "role": "weight", "layout": "FRACTAL_NZ"},
        "weight_scale": nd["weight_scale"],
        "pertoken_scale": nd["pertoken_scale"],
        "output": nd["output"],
    }
    plan_nz = qbmm.predict_abi(pre, version=None)
    _assert_predicted(plan_nz)
    assert plan_nz["selected_call"]["variant"] == "int8_shared_torch_api_nz"
    assert plan_nz["work"]["preparation"] is None
    assert plan_nz["selected_call"]["signature_hash"] != \
        plan_nd["selected_call"]["signature_hash"]

    # ND input with explicit NZ requirement keeps the real pack producer.
    required = qbmm.predict_abi(nd, attrs={"weight_layout": "FRACTAL_NZ"},
                                version=None)
    _assert_predicted(required)
    assert required["selected_call"]["variant"] == \
        "int8_shared_torch_api_nz_from_nd"
    assert required["work"]["preparation"]["producer"] == "maybe_trans_nz"
    prep = required["work"]["preparation"]
    assert prep["read_bytes"] == 32 * 64
    assert prep["write_bytes"] == 32 * 64
    assert prep["traffic_bytes"] == 2 * 32 * 64


def test_serialized_typed_views_replay_the_same_native_call():
    cases = {}
    cases["matmul"] = (matmul, {
        "x1": {"shape": [4, 5120], "dtype": "bf16"},
        "x2": {"shape": [8192, 5120], "dtype": "bf16"},
        "output": {"shape": [4, 8192], "dtype": "bf16"}},
        {"transpose_x1": False, "transpose_x2": True},
        {"source": "r31"})
    cases["qmmnz"] = (qmmnz, {
        "x": {"shape": [1, 16], "dtype": "int8"},
        "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                      "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [32], "dtype": "bf16"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32"},
        "output": {"shape": [1, 32], "dtype": "bf16"}},
        None, {"source": "r31"})
    cases["qbmm_nd"] = (qbmm, {
        "x": {"shape": [1, 32], "dtype": "int8"},
        "weight": {"shape": [32, 64], "dtype": "int8"},
        "weight_scale": {"shape": [64], "dtype": "bf16"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32"},
        "output": {"shape": [1, 64], "dtype": "bf16"}},
        {"weight_layout": "ND"}, None)
    cases["qbmm_nz"] = (qbmm, {
        "x": {"shape": [1, 32], "dtype": "int8"},
        "weight_nz": {"shape": [2, 2, 16, 32], "dtype": "int8",
                      "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [64], "dtype": "bf16"},
        "pertoken_scale": {"shape": [1], "dtype": "fp32"},
        "output": {"shape": [1, 64], "dtype": "bf16"}},
        None, None)
    cases["gmm4"] = (gmm2, _small_grouped(2, [2, 0]),
                     {"group_list_type": 1}, {"source": "r64"})
    cases["gmm1"] = (gmm1, _small_swiglu(2, [2, 0]),
                     {"group_list_type": 1}, {"source": "r64"})
    for name, (leaf, tensors, attrs, version) in cases.items():
        first = leaf.predict_abi(tensors, attrs=attrs, version=version)
        _assert_predicted(first)
        payload = json.loads(json.dumps(first["tensors"]))
        replay = leaf.predict_abi(
            payload, attrs=first["invocation"]["attrs"],
            version=first["invocation"]["version"],
            conditions=first["invocation"]["conditions"])
        assert replay["status"] == "predicted", name
        assert replay["selected_call"]["variant"] == \
            first["selected_call"]["variant"], name
        assert replay["selected_call"]["signature_hash"] == \
            first["selected_call"]["signature_hash"], name
        assert replay["work"]["bytes"]["total"] == \
            first["work"]["bytes"]["total"], name


def test_t12_public_entry_dispatches_all_matrix_leaves():
    requests = {}
    requests["aclnnMatmul"] = {
        "api": {"entry": "aclnnMatmul"},
        "version": {"source": "r31"},
        "attrs": {"transpose_x1": False, "transpose_x2": True},
        "tensors": {
            "x1": {"shape": [4, 5120], "dtype": "bf16"},
            "x2": {"shape": [8192, 5120], "dtype": "bf16"},
            "output": {"shape": [4, 8192], "dtype": "bf16"}},
    }
    requests["aclnnQuantMatmulWeightNz"] = {
        "api": {"entry": "aclnnQuantMatmulWeightNz"},
        "version": {"source": "r31"},
        "tensors": {
            "x": {"shape": [1, 16], "dtype": "int8"},
            "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                          "layout": "FRACTAL_NZ"},
            "weight_scale": {"shape": [32], "dtype": "bf16"},
            "pertoken_scale": {"shape": [1], "dtype": "fp32"},
            "output": {"shape": [1, 32], "dtype": "bf16"}},
    }
    requests["torch_npu.npu_quant_matmul"] = {
        "api": {"entry": "torch_npu.npu_quant_matmul"},
        "version": {"source": "r64"},
        "tensors": _qbmm_tensors(17),
    }
    requests["torch_npu.npu_grouped_matmul"] = {
        "api": {"entry": "torch_npu.npu_grouped_matmul"},
        "version": {"source": "r64"},
        "attrs": {"group_list_type": 1},
        "tensors": _small_grouped(2, [2, 0]),
    }
    requests["torch_npu.npu_grouped_matmul_swiglu_quant_v2"] = {
        "api": {"entry":
                "torch_npu.npu_grouped_matmul_swiglu_quant_v2"},
        "version": {"source": "r64"},
        "attrs": {"group_list_type": 1},
        "tensors": _small_swiglu(2, [2, 0]),
    }
    expected = {
        "aclnnMatmul": 4 * 8192 * 5120,
        "aclnnQuantMatmulWeightNz": 1 * 32 * 16,
        "torch_npu.npu_quant_matmul": 17 * 8192 * 5120,
        "torch_npu.npu_grouped_matmul": 2 * 64 * 64,
        "torch_npu.npu_grouped_matmul_swiglu_quant_v2": 2 * 64 * 64,
    }
    for entry, request in requests.items():
        result = simulate_aclnn_call(request)
        assert result.status == "predicted", (entry, result.reasons)
        assert result.prediction["work"]["cube"]["macs"] == expected[entry]
        assert result.binding.typed["entry"] == entry
        assert result.occurrence["priced"]["charged"] == \
            "once_at_timing_owner"
    qmm = simulate_aclnn_call(requests["aclnnQuantMatmulWeightNz"])
    assert qmm.prediction["work"]["bytes"]["total"] == 660
    assert qmm.prediction["work"]["epilogue"]["column_scale_bytes"] == 64
    assert qmm.prediction["materialization"]["tensors"]["weight_nz"][
        "views"]["device"]["layout"] == "FRACTAL_NZ"

    # Real T02 row 120 through the same public entry.
    row120 = dict(requests["torch_npu.npu_quant_matmul"])
    row120["tensors"] = {
        "x": {"shape": [4, 8704], "dtype": "int8"},
        "weight_nz": {"shape": [160, 544, 16, 32], "dtype": "int8",
                      "layout": "FRACTAL_NZ"},
        "weight_scale": {"shape": [5120], "dtype": "bf16"},
        "pertoken_scale": {"shape": [4], "dtype": "fp32"},
        "output": {"shape": [4, 5120], "dtype": "bf16"},
    }
    row = simulate_aclnn_call(row120)
    assert row.status == "predicted", row.reasons
    assert row.prediction["work"]["bytes"]["total"] == 44_650_512
    assert row.prediction["work"]["epilogue"]["column_scale_bytes"] == 10_240
    assert row.binding.typed["variant"] == "int8_shared_torch_api_nz"
    assert row.binding.typed["source"]["repository"] == "torch_npu"


def test_t12_serializes_and_replays_a_mixed_view_invocation():
    request = {
        "api": {"entry": "aclnnQuantMatmulWeightNz"},
        "version": {"source": "r31"},
        "tensors": {
            "x": {"shape": [1, 16], "dtype": "int8"},
            "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                          "layout": "FRACTAL_NZ",
                          "logical_shape": [16, 32],
                          "logical_layout": "ND",
                          "logical_direction": "k_by_n",
                          "storage_shape": [1, 1, 16, 32],
                          "storage_dtype": "int8",
                          "storage_layout": "FRACTAL_NZ",
                          "storage_bytes": 512,
                          "device_shape": [1, 1, 16, 32],
                          "device_dtype": "int8",
                          "device_layout": "FRACTAL_NZ"},
            "weight_scale": {
                "shape": [32], "dtype": "bf16",
                "views": {
                    "storage": {"shape": [32], "dtype": "bf16",
                                "layout": "ND", "bytes": 64},
                    "interpretation": {"dtype": "bf16", "layout": "ND"},
                }},
            "pertoken_scale": {"shape": [1], "dtype": "fp32"},
            "output": {"shape": [1, 32], "dtype": "bf16"},
        },
    }
    first = simulate_aclnn_call(request)
    assert first.status == "predicted", first.reasons
    assert first.prediction["work"]["bytes"]["total"] == 660
    replay = {
        "api": {"entry": "aclnnQuantMatmulWeightNz"},
        "version": first.replay["version"],
        "tensors": first.replay["tensors"],
        "attrs": first.replay["attrs"],
        "conditions": first.replay["conditions"],
    }
    second = simulate_aclnn_call(replay)
    assert second.status == "predicted", second.reasons
    assert second.prediction["work"]["bytes"]["total"] == 660
    assert second.occurrence["call_id"] == first.occurrence["call_id"]
    assert second.replay["tensors"]["weight_nz"]["device_layout"] == \
        "FRACTAL_NZ"
    assert second.replay["tensors"]["weight_scale"]["views"]["storage"][
        "dtype"] == "bf16"


def test_int8_checkpoint_replay_flat_nested_for_both_leaves():
    def qmmnz_request(nested=False):
        scale = {"shape": [32], "dtype": "bf16", "role": "scale"}
        if nested:
            scale["views"] = {"storage": {"shape": [32], "dtype": "bf16",
                                          "layout": "ND", "bytes": 64},
                              "interpretation": {"dtype": "bf16",
                                                 "layout": "ND"}}
        else:
            scale.update({"storage_shape": [32], "storage_dtype": "bf16",
                          "storage_layout": "ND", "storage_bytes": 64,
                          "interpret_dtype": "bf16"})
        return {"x": {"shape": [1, 16], "dtype": "int8"},
                "weight_nz": {"shape": [1, 1, 16, 32], "dtype": "int8",
                              "layout": "FRACTAL_NZ"},
                "weight_scale": scale,
                "pertoken_scale": {"shape": [1], "dtype": "fp32"},
                "output": {"shape": [1, 32], "dtype": "bf16"}}

    attrs = {"weight_scale_storage_format": "fp32",
             "weight_scale_storage_bytes": 4}
    for leaf, version in ((qmmnz, R31), (qbmm, None)):
        first = leaf.predict_abi(qmmnz_request(False), attrs=attrs,
                                 version=version)
        second = leaf.predict_abi(qmmnz_request(True), attrs=attrs,
                                  version=version)
        replay = leaf.predict_abi(
            json.loads(json.dumps(first["tensors"])), attrs=attrs,
            version=first["invocation"]["version"])
        for result in (first, second, replay):
            assert result["status"] == "predicted"
            assert result["work"]["bytes"]["total"] == 660
            assert result["work"]["bytes"]["weight_scale"] == 64
            assert result["scale"]["weight_scale"]["runtime"]["bytes"] == 64
            assert result["scale"]["weight_scale"]["storage"]["bytes"] == 128
            assert result["work"]["checkpoint_or_upstream"][
                "weight_scale_storage_bytes"] == 128
            assert result["work"]["checkpoint_or_upstream"][
                "included_in_call_footprint"] is False
            assert result["work"]["epilogue"]["vector_ops"] == 128
        assert first["selected_call"]["signature_hash"] == \
            second["selected_call"]["signature_hash"] == \
            replay["selected_call"]["signature_hash"]


def test_mxfp8_device_view_positive_and_conflict():
    request = _qbmm_tensors(2, k=128, n=128)
    request["weight"] = {
        **request["weight"],
        "device_shape": [4, 8, 16, 32],
        "device_dtype": "fp8_e4m3",
        "device_layout": "FRACTAL_NZ",
        "device_bytes": 16_384,
    }
    request["weight_scale"] = {
        **request["weight_scale"],
        "device_shape": [2, 128, 2],
        "device_dtype": "uint8",
        "device_layout": "ND",
        "device_bytes": 512,
    }
    legal = qbmm.predict_abi(request, version=R64)
    _assert_predicted(legal)
    assert legal["work"]["bytes"]["weight"] == 16_384
    assert legal["work"]["bytes"]["weight_scale"] == 512
    assert legal["work"]["bytes"]["total"] == 17_672
    assert legal["tensors"]["weight"]["views"]["device"]["dtype"] == \
        "fp8_e4m3"
    assert legal["tensors"]["weight_scale"]["views"]["device"]["dtype"] == \
        "uint8"

    conflict = {**request,
                "weight": {**request["weight"],
                           "device_dtype": "fp32",
                           "device_bytes": 65_536}}
    gap = qbmm.predict_abi(conflict, version=R64)
    assert gap["status"] == "no_match"
    assert "mismatch:tensors.weight.device_dtype" in gap["gap"]
    assert gap["selected_call"] is None
    assert gap["work"] is None
    assert gap["geometry"] is None

    scale_conflict = {**request,
                      "weight_scale": {**request["weight_scale"],
                                       "device_dtype": "fp32",
                                       "device_bytes": 2048}}
    scale_gap = qbmm.predict_abi(scale_conflict, version=R64)
    _assert_gap(scale_gap, "mismatch:tensors.weight_scale.device_dtype")


def test_int8_nd_weight_device_conflicts_with_or_without_int8_storage():
    def request(device_dtype, with_storage):
        weight = {"shape": [32, 64], "dtype": "int8",
                  "device_shape": [32, 64], "device_dtype": device_dtype,
                  "device_layout": "ND"}
        if with_storage:
            weight.update({"storage_shape": [32, 64],
                           "storage_dtype": "int8",
                           "storage_layout": "ND"})
        return {"x": {"shape": [1, 32], "dtype": "int8"},
                "weight": weight,
                "weight_scale": {"shape": [64], "dtype": "bf16"},
                "pertoken_scale": {"shape": [1], "dtype": "fp32"},
                "output": {"shape": [1, 64], "dtype": "bf16"}}

    for with_storage in (False, True):
        gap = qbmm.predict_abi(request("fp32", with_storage),
                               attrs={"weight_layout": "ND"}, version=None)
        assert gap["status"] == "no_match"
        assert "mismatch:tensors.weight.device_dtype" in gap["gap"]
        assert gap["selected_call"] is None
        assert gap["work"] is None
        assert gap["geometry"] is None
        legal = qbmm.predict_abi(request("int8", with_storage),
                                 attrs={"weight_layout": "ND"}, version=None)
        _assert_predicted(legal)
        assert legal["work"]["bytes"]["total"] == 2340
        assert legal["work"]["preparation"] is None
        assert legal["selected_call"]["variant"] == \
            "int8_shared_torch_api_nd"


def test_t12_qmmnz_nd_and_qbmm_nd_nz_boundaries():
    qnd = {
        "api": {"entry": "aclnnQuantMatmulWeightNz"},
        "version": {"source": "r31"},
        "tensors": {
            "x": {"shape": [1, 32], "dtype": "int8"},
            "weight": {"shape": [32, 64], "dtype": "int8"},
            "weight_scale": {"shape": [64], "dtype": "bf16"},
            "pertoken_scale": {"shape": [1], "dtype": "fp32"},
            "output": {"shape": [1, 64], "dtype": "bf16"},
        },
    }
    qnd_result = simulate_aclnn_call(qnd)
    assert qnd_result.status == "predicted"
    assert qnd_result.binding.typed["variant"] == \
        "w8a8_dynamic_weight_nz_from_nd"
    assert qnd_result.prediction["work"]["bytes"]["total"] == 2340
    prep = qnd_result.prediction["work"]["preparation"]
    assert prep["read_bytes"] == 2048
    assert prep["write_bytes"] == 2048
    assert prep["traffic_bytes"] == 4096
    assert prep["included_in_call_footprint"] is False
    assert qnd_result.prediction["materialization"]["tensors"]["weight_nz"][
        "provenance"]["origin"] == "prepared"
    qreplay = simulate_aclnn_call({
        "api": {"entry": "aclnnQuantMatmulWeightNz"},
        "version": qnd_result.replay["version"],
        "tensors": json.loads(json.dumps(qnd_result.replay["tensors"])),
        "attrs": qnd_result.replay["attrs"]})
    assert qreplay.status == "predicted"
    assert qreplay.binding.typed["variant"] == \
        "w8a8_dynamic_weight_nz_from_nd"
    assert qreplay.prediction["work"]["bytes"]["total"] == 2340
    assert qreplay.prediction["work"]["preparation"]["traffic_bytes"] == 4096
    assert qreplay.occurrence["call_id"] == qnd_result.occurrence["call_id"]

    bnd = {
        "api": {"entry": "torch_npu.npu_quant_matmul"},
        "tensors": {
            "x": {"shape": [1, 32], "dtype": "int8"},
            "weight": {"shape": [32, 64], "dtype": "int8"},
            "weight_scale": {"shape": [64], "dtype": "bf16"},
            "pertoken_scale": {"shape": [1], "dtype": "fp32"},
            "output": {"shape": [1, 64], "dtype": "bf16"},
        },
        "attrs": {"weight_layout": "ND"},
    }
    nd = simulate_aclnn_call(bnd)
    assert nd.status == "predicted"
    assert nd.binding.typed["variant"] == "int8_shared_torch_api_nd"
    assert nd.prediction["work"]["bytes"]["total"] == 2340
    assert nd.prediction["work"]["preparation"] is None

    bnz = {"api": {"entry": "torch_npu.npu_quant_matmul"},
           "tensors": {"x": {"shape": [1, 32], "dtype": "int8"},
                       "weight_nz": {"shape": [2, 2, 16, 32],
                                     "dtype": "int8",
                                     "layout": "FRACTAL_NZ"},
                       "weight_scale": {"shape": [64], "dtype": "bf16"},
                       "pertoken_scale": {"shape": [1], "dtype": "fp32"},
                       "output": {"shape": [1, 64], "dtype": "bf16"}}}
    nz = simulate_aclnn_call(bnz)
    assert nz.status == "predicted"
    assert nz.binding.typed["variant"] == "int8_shared_torch_api_nz"
    assert nz.prediction["work"]["bytes"]["total"] == 2340
    assert nz.prediction["work"]["preparation"] is None
    assert nd.occurrence["call_id"] != nz.occurrence["call_id"]
    req_nz = {"api": {"entry": "torch_npu.npu_quant_matmul"},
              "tensors": bnd["tensors"],
              "attrs": {"weight_layout": "FRACTAL_NZ"}}
    direct_nz = simulate_aclnn_call(req_nz)
    assert direct_nz.status == "predicted"
    assert direct_nz.binding.typed["variant"] == \
        "int8_shared_torch_api_nz_from_nd"
    assert direct_nz.prediction["work"]["bytes"]["total"] == 2340
    assert direct_nz.prediction["work"]["preparation"]["traffic_bytes"] == 4096
    replay_nz = simulate_aclnn_call({
        "api": {"entry": "torch_npu.npu_quant_matmul"},
        "version": direct_nz.replay["version"],
        "tensors": json.loads(json.dumps(direct_nz.replay["tensors"])),
        "attrs": direct_nz.replay["attrs"]})
    assert replay_nz.status == "predicted"
    assert replay_nz.binding.typed["variant"] == \
        "int8_shared_torch_api_nz_from_nd"
    assert replay_nz.prediction["work"]["preparation"][
        "preparation_id"] == direct_nz.prediction["work"]["preparation"][
            "preparation_id"]
    assert replay_nz.occurrence["call_id"] == direct_nz.occurrence["call_id"]


def test_operator_plan_executes_bf16_and_int8_controls():
    bf = matmul.predict_abi(
        {"x1": {"shape": [4, 65], "dtype": "bf16"},
         "x2": {"shape": [32, 65], "dtype": "bf16"},
         "output": {"shape": [4, 32], "dtype": "bf16"}},
        attrs={"transpose_x2": True}, version={"source": "r31"})
    _assert_predicted(bf)
    assert bf["work"]["cube"]["macs"] == 8320
    assert bf["work"]["bytes"]["input"] == 520
    assert bf["work"]["bytes"]["weight"] == 4160
    assert bf["work"]["bytes"]["output"] == 256
    bf_plan = matmul.operator_plan(bf, tile=(16, 32, 16))
    bf_builder = bf_plan.metadata["operator_plan"]["matrix_builder"]
    assert bf_builder["m"] == 4 and bf_builder["n"] == 32 and \
        bf_builder["k"] == 65
    assert bf_builder["transpose"] == {"x1": False, "x2": True}
    assert bf_builder["runtime_views"]["x2"]["storage"]["shape"] == [32, 65]
    assert bf_builder["input_dtypes"] == ["bf16", "bf16", "bf16"]
    assert bf_plan.metadata["operator_plan"]["issued"]["program_macs"] == 8_320
    assert bf_plan.metadata["operator_plan"]["issued"][
        "declared_padded_macs"] == 40_960
    assert bf_plan.metadata["operator_plan"]["kernels"] == [
        {"stage": "load", "owner": "mte2"},
        {"stage": "compute", "owner": "cube"},
        {"stage": "store", "owner": "mte3"},
    ]
    bf_exec = execute_operator_plan(bf, tile=(16, 32, 16))
    assert bf_exec["serviced"] == {
        "cube": 16_640.0, "vector": 0.0, "hbm": 4_936.0,
        "mte1": 0.0, "mte2": 4_680.0, "mte3": 256.0,
    }
    assert bf_exec["output_handles"][0]["shape"] == [4, 32]
    assert bf_exec["preparation"] is None
    assert type(bf_exec["plan"].lower(bf_exec["context"])).__name__ == \
        "NativeProgram"

    i8 = qbmm.predict_abi(
        {"x": {"shape": [3, 66], "dtype": "int8"},
         "weight": {"shape": [66, 32], "dtype": "int8"},
         "weight_scale": {"shape": [32], "dtype": "bf16"},
         "pertoken_scale": {"shape": [3], "dtype": "fp32"},
         "output": {"shape": [3, 32], "dtype": "bf16"}},
        attrs={"weight_layout": "ND"}, version=None)
    _assert_predicted(i8)
    assert i8["work"]["cube"]["macs"] == 6336
    assert i8["work"]["bytes"] == {
        "input": 198, "weight": 2112, "weight_scale": 64,
        "pertoken_scale": 12, "output": 192, "total": 2578,
        "basis": "call_boundary_tensor_footprint",
    }
    assert i8["work"]["vector"]["ops"] == 384
    i8_plan = qbmm.operator_plan(i8, tile=(16, 32, 16))
    i8_builder = i8_plan.metadata["operator_plan"]["matrix_builder"]
    assert i8_builder["quant"]["quantized"] is True
    assert i8_builder["runtime_views"]["weight"]["shape"] == [66, 32]
    assert i8_builder["scales"]["weight_scale"]["runtime"]["bytes"] == 64
    assert i8_builder["scales"]["weight_scale"]["runtime"]["dtype"] == "bf16"
    assert i8_plan.metadata["operator_plan"]["issued"]["program_macs"] == 6_336
    assert i8_plan.metadata["operator_plan"]["issued"][
        "declared_padded_macs"] == 40_960
    assert i8_plan.metadata["operator_plan"]["kernels"] == [
        {"stage": "load", "owner": "mte2"},
        {"stage": "compute", "owner": "cube"},
        {"stage": "scales", "owner": "mte2"},
        {"stage": "epilogue", "owner": "vector"},
        {"stage": "store", "owner": "mte3"},
    ]
    i8_exec = execute_operator_plan(i8, tile=(16, 32, 16))
    assert i8_exec["serviced"] == {
        "cube": 12_672.0, "vector": 384.0, "hbm": 2_578.0,
        "mte1": 0.0, "mte2": 2_386.0, "mte3": 192.0,
    }
    assert i8_exec["preparation"] is None


def test_operator_plan_carries_preparation_outside_call_footprint():
    req = {"x": {"shape": [1, 32], "dtype": "int8"},
           "weight": {"shape": [32, 64], "dtype": "int8"},
           "weight_scale": {"shape": [64], "dtype": "bf16"},
           "pertoken_scale": {"shape": [1], "dtype": "fp32"},
           "output": {"shape": [1, 64], "dtype": "bf16"}}
    from_nd = qmmnz.predict_abi(req, version=R31)
    _assert_predicted(from_nd)
    plan = build_operator_plan(from_nd, tile=(16, 32, 16))
    desc = plan.metadata["operator_plan"]
    assert desc["preparation"]["read_bytes"] == 2048
    assert desc["preparation"]["write_bytes"] == 2048
    assert desc["preparation"]["traffic_bytes"] == 4096
    assert desc["preparation"]["included_in_call_footprint"] is False
    assert desc["preparation_owner"] == "producer_or_upstream"
    executed = execute_operator_plan(from_nd, tile=(16, 32, 16))
    assert executed["footprint_bytes"] == 2340
    assert executed["serviced"] == {
        "cube": 2 * 32 * 64, "vector": 4 * 1 * 64,
        "hbm": 2340.0, "mte1": 0.0, "mte2": 2248.0, "mte3": 128.0,
    }

    external = qbmm.predict_abi(
        {"x": req["x"],
         "weight_nz": {"shape": [2, 2, 16, 32], "dtype": "int8",
                       "layout": "FRACTAL_NZ"},
         "weight_scale": req["weight_scale"],
         "pertoken_scale": req["pertoken_scale"],
         "output": req["output"]}, version=None)
    _assert_predicted(external)
    assert operator_plan_descriptor(external)["preparation"] is None
