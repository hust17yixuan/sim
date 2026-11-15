"""Focused R31 A3 GDN dense BF16 MatMulV3 adapter checks.

[Role] Prove the real R31 A3 ``ba_projection``/``o_projection`` BF16 dense
  rows select one ``aclnnMatmul``/``MatMulV3`` call and prepare the exact
  ``self``/``mat2``/``out`` tensors from the runtime K/N matrix.
[Invariants] Only the R31 image and BF16 dense projection roles are admitted;
  K/N/H are static, M is the only dynamic axis, and the physical mat2 storage
  is [N,K].  The missing wrapper/C ABI/kernel source stays explicit.
[Side-Effect Policy] Read-only CPU metadata; no device, weights or I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _CallSpec, _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r31_matmul_bf16 import (
    ACL_API, HOST_OP, NATIVE_TYPE, R31_IMAGE_SHA, binding_spec, dynamic_call)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_model)

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"
R31 = (("image_id", "sha256:" + R31_IMAGE_SHA),)
R64 = (("image_id", "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),)
DOMAIN = AnalysisDomain(max_tokens=10 ** 9, max_moe_rows=10 ** 9,
                        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
                        max_index_rows=10 ** 12, max_work=10 ** 15,
                        max_q=10 ** 9, max_s_kv=10 ** 9, max_seq=10 ** 9,
                        max_batch=10 ** 6)


def _case():
    case = parse_case(json.loads(QWEN.read_text()))
    features = FeatureConfig.from_model(
        case.model, tp=2, dp=4, ep=1, pp=1, cores=64, token_tile=16,
        kv_tile=16, moe_row_tile=4, split_target=1000, max_splits=4)
    return case, instantiate_model(case.model, features)


def _compiler(case):
    return OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="bf16"),
        ImplementationPolicy(cores=64), domain=DOMAIN)


def _geometry(tokens):
    return WorkloadGeometry(
        (RequestGeometry(tokens, tokens, phase="prefill"),),
        phase="prefill", cohort_batch=1, local_batch=1)


def _binding(compiled, spec):
    surface = _surface(compiled, spec, source_boundary="kernel")
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved))


def _subgraph(inst, kind, name):
    return next(sub for layer in inst.layers for sub in layer.subgraphs
                if sub.kind == kind
                and any(operator.name == name for operator in sub.operators))


@pytest.fixture(scope="module")
def qwen():
    return _case()


def test_r31_bf16_projections_select_matmulv3_and_keep_static_k_n(qwen):
    case, inst = qwen
    compiler = _compiler(case)
    expected = {
        "ba_projection": (5120, 48, 5120),
        "o_projection": (3072, 5120, 5120),
    }
    call_ids = {}
    for name, (k, n, hidden) in expected.items():
        compiled = compiler.compile(
            _subgraph(inst, "gdn_linear_attn", name), feature_metadata=R31)
        operator = compiled.operator(name)
        spec = binding_spec(operator, phase="prefill")
        assert isinstance(spec, _CallSpec)
        assert spec.api["entry"] == ACL_API
        assert spec.api["aclnn_symbol"] == ACL_API
        assert spec.source["sha256"] == (
            "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed")
        assert spec.source["revision"] == "r31/9e869e81484"
        assert spec.kernels[0]["name"] == NATIVE_TYPE
        assert spec.kernels[0]["host_op"] == HOST_OP
        assert spec.kernels[0]["evidence"] == "measured"
        assert spec.execution_core is None
        assert spec.typed["attrs"]["execution_core_status"] == "not_retained"
        attrs = spec.typed["attrs"]
        assert attrs["matrix_k"] == k
        assert attrs["matrix_n"] == n
        assert attrs["hidden"] == hidden
        assert attrs["weight_storage_shape"] == [n, k]
        assert attrs["weight_logical_shape"] == [k, n]
        assert attrs["weight_orientation"] == "out_by_in"
        assert attrs["transpose_x2"] is None
        assert attrs["deployment_image_id"] == R31_IMAGE_SHA
        assert "aclnn_matmul_c_abi_header_not_retained" in spec.unresolved
        binding = _binding(operator, spec)
        env = dynamic_call(binding, operator, _geometry(4096),
                           validate=validate_native_occurrence)
        assert env is not None
        actual = env["actual"]["shapes"]
        assert actual["input"] == [4096, k]
        assert actual["weight"] == [k, n]
        assert actual["weight_storage"] == [n, k]
        assert actual["output"] == [4096, n]
        prepare = env["prepare"]
        assert prepare["complete"] is False
        assert prepare["partial"] is True
        assert prepare["runtime_executed"] is False
        assert prepare["descriptors"]["weight"]["shape"] == [n, k]
        assert prepare["descriptors"]["weight"]["logical_shape"] == [k, n]
        assert prepare["descriptors"]["output"]["shape"] == [4096, n]
        assert prepare["captured_call"]["operands"] == {
            "self": [4096, k], "mat2": [n, k], "out": [4096, n]}
        assert prepare["match"]["k"] == k
        assert prepare["match"]["n"] == n
        assert prepare["match"]["h"] == hidden
        assert prepare["match"]["complete"] is False
        call_ids[name] = binding.call_id
    assert len(set(call_ids.values())) == 2


def test_r31_bf16_static_identity_survives_independent_m(qwen):
    case, inst = qwen
    compiled = _compiler(case).compile(
        _subgraph(inst, "gdn_linear_attn", "o_projection"), feature_metadata=R31)
    operator = compiled.operator("o_projection")
    spec = binding_spec(operator, phase="prefill")
    binding = _binding(operator, spec)
    seen = {}
    for tokens in (4096, 2048, 2, 1):
        env = dynamic_call(binding, operator, _geometry(tokens),
                           validate=validate_native_occurrence)
        assert env is not None
        shapes = env["actual"]["shapes"]
        assert shapes["input"] == [tokens, 3072]
        assert shapes["weight"] == [3072, 5120]
        assert shapes["output"] == [tokens, 5120]
        assert env["prepare"]["descriptors"]["weight"]["shape"] == [5120, 3072]
        assert env["prepare"]["match"]["k"] == 3072
        assert env["prepare"]["match"]["n"] == 5120
        seen[tokens] = binding.call_id
    assert len(set(seen.values())) == 1


def test_r31_bf16_rejects_quantized_rows_other_images_and_decode(qwen):
    case, inst = qwen
    compiler = _compiler(case)
    quantized = compiler.compile(
        _subgraph(inst, "gdn_linear_attn", "qkvz_projection"), feature_metadata=R31)
    assert binding_spec(quantized.operator("qkvz_projection"),
                        phase="prefill") is None
    wrong_image = compiler.compile(
        _subgraph(inst, "gdn_linear_attn", "ba_projection"), feature_metadata=R64)
    assert binding_spec(wrong_image.operator("ba_projection"),
                        phase="prefill") is None
    compiled = compiler.compile(
        _subgraph(inst, "gdn_linear_attn", "ba_projection"), feature_metadata=R31)
    assert binding_spec(compiled.operator("ba_projection"),
                        phase="decode") is None
