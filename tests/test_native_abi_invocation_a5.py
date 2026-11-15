"""Independent A5 ACLNN input slice: ABI -> common program -> CPU prediction.

[Role] Prove the reverse native entry: one real AddRmsNorm ACLNN invocation is
  normalized from API/tensor/attr/version metadata, bound to the same immutable
  call identity the model path emits, and priced by the shared piecewise
  program.  Also prove declared gaps return a replayable minimal input instead
  of a fabricated prediction or an algorithm/envelope fallback.
[Invariants] No tensor is allocated, no device runtime is imported, no model
  case is required, and no duration is copied from a target observation.
  Identity is source+kernels+typed ABI; region members are many-to-one/one-to-
  many relations, never forced 1:1.
[Timeouts & Locks] CPU-only bounded arithmetic; no waits or locks.
[Side-Effect Policy] Read-only test code.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.native_binding import (
    bind_native_call, coverage_from_records, declare_native_call,
    join_profiler_rows,
)
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, OperatorPiecewiseCompiler,
)
from mindie_sim.prediction.piecewise import native_add_rms_norm as adapter
from mindie_sim.prediction.piecewise import native_calls, native_invocation
from mindie_sim.prediction.piecewise.native_invocation import (
    a5_native_abi_inventory, simulate_aclnn_call,
)
from mindie_sim.schema.common import ModelingError

SKU = "Ascend950DT_9582"
R31_IMAGE = (
    "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")


def _request(*, hidden=5120, rows=2048, dtype="bf16", rstd_dtype="fp32",
             epsilon=1e-6, version=None, variant=None,
             sequence_rows="shard", sequence_ranks=2):
    request = {
        "api": {"namespace": "aclnn", "entry": "aclnnAddRmsNorm",
                "aclnn_symbol": "aclnnAddRmsNorm"},
        "version": ({"api": "cann-9.2.0", "image_id": R31_IMAGE}
                    if version is None else version),
        "tensors": {
            "x": {"shape": [rows, hidden], "dtype": dtype, "layout": "ND",
                  "role": "input"},
            "residual": {"shape": [rows, hidden], "dtype": dtype,
                         "layout": "ND", "role": "input"},
            "gamma": {"shape": [hidden], "dtype": dtype, "layout": "ND",
                      "role": "weight"},
            "output": {"shape": [rows, hidden], "dtype": dtype,
                       "layout": "ND", "role": "output"},
            "rstd": {"shape": [rows, 1], "dtype": rstd_dtype, "layout": "ND",
                     "role": "output"},
            "residual_out": {"shape": [rows, hidden], "dtype": dtype,
                             "layout": "ND", "role": "output"},
        },
        "attrs": {"epsilon": epsilon},
        "conditions": {"sequence_rows": sequence_rows,
                       "sequence_ranks": sequence_ranks},
    }
    if version is not None and isinstance(version, dict):
        request["version"] = dict(version)
    if variant is not None:
        request["conditions"]["variant"] = variant
    return request


class _Theta:
    def __init__(self, values):
        self.values = dict(values)

    def find(self, key, default=None):
        return self.values.get(key, default)


def _compiled(*, kind="layernorm", name="rmsnorm", siblings=("residual_add", "rmsnorm"),
              image_id=adapter.R31_IMAGE_ID, image=adapter.R31_IMAGE_TAG,
              **theta):
    values = {
        "elements": 5120,
        "gamma_bytes": 10240.0,
        "gamma_format": "bf16",
        "sequence_rows": "shard",
        "sequence_ranks": 2,
        "layernorm": 0,
        "norm_weight_offset": 0.0,
        "norm_arch": "rmsnorm",
        "norm_eps": 1e-6,
        "rstd_bytes": 4.0 if kind == "lm_head_sampler" else None,
        "residual_out_bytes": 10240.0 if kind == "lm_head_sampler" else None,
        "residual_add": 1 if kind == "lm_head_sampler" else None,
        "fusion": "add_rmsnorm" if kind == "lm_head_sampler" else None,
    }
    values.update(theta)
    operator = SimpleNamespace(
        family="elementwise", name=name, role="norm",
        template=SimpleNamespace(role="norm"), theta=_Theta(values))
    return SimpleNamespace(
        kind=kind, layer=0, operator=operator,
        siblings=tuple(SimpleNamespace(name=item) for item in siblings),
        feature_metadata=(("image_id", image_id), ("image", image)))


def _model_binding(compiled):
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    boundary = "call" if len(spec.members) > 1 else "kernel"
    owner_kind = {"layernorm": "N0", "lm_head_sampler": "O0"}.get(
        compiled.kind, compiled.kind)
    if compiled.kind != owner_kind:
        compiled = SimpleNamespace(**{**vars(compiled), "kind": owner_kind})
    surface = native_calls._surface(compiled, spec, source_boundary=boundary)
    binding = bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None))
    if spec.owner != compiled.operator.name:
        binding = replace(binding, timing_owner={
            "op_id": spec.owner, "scope": "call"})
    return spec, binding


def test_independent_add_rms_norm_predicts_from_typed_abi():
    result = simulate_aclnn_call(
        _request(variant="fused_attention_norm"), hardware=SKU)
    assert result.status == "predicted"
    assert result.reasons == ()
    assert result.binding_bound is True
    assert result.source_establishment["established"] is True
    prediction = result.prediction
    assert Fraction(prediction["duration_ms"]) > 0
    assert prediction["fidelity"] == "fused_physical_kernel"
    assert prediction["work"] == {
        "hbm": "83904512", "mte2": "41953280",
        "mte3": "41951232", "vector": "73564160",
    }
    assert prediction["materialization"]["rstd"] == "8192"
    assert prediction["environment"]["arch"] == "A5"
    assert prediction["environment"]["hidden"] == 5120
    assert result.program["physical"]["common_program"] == (
        "elementwise.N0.add_rmsnorm")
    assert result.program["physical"]["shared_leaf"] == (
        "native_add_rms_norm.predict_add_rms_norm")
    assert result.program["caller_plan"] == {
        "variant": "fused_attention_norm", "owner": "fused:add_rmsnorm",
        "members": ["residual_add", "rmsnorm"], "layer": None,
        "source_boundary": "call",
    }
    assert result.program["cross_view"]["consistent"] is True

    binding = result.binding
    assert binding is not None
    assert binding.bound is True
    assert binding.api["entry"] == "aclnnAddRmsNorm"
    assert binding.api["aclnn_symbol"] == "aclnnAddRmsNorm"
    assert binding.source["revision"] == "cann-9.2.0-r31"
    assert binding.source["sha256"] == adapter.CANN_LIBOPAPI["sha256"]
    assert binding.typed["shapes"]["x"] == [5120]
    assert binding.typed["attrs"]["sequence_rows"] == "shard"
    assert binding.member_op_ids == ("residual_add", "rmsnorm")
    assert binding.timing_owner == {"op_id": "fused:add_rmsnorm",
                                    "scope": "call"}
    assert binding.source_boundary == "call"
    assert binding.op_id == "N0:elementwise:rmsnorm"
    assert binding.evidence["kernels"] == "partial"
    assert "aclnn_positional_slots_not_retained" in binding.unresolved

    occurrence = result.occurrence
    assert occurrence["charge"] == "call"
    assert occurrence["call_id"] == binding.call_id
    assert occurrence["timing_owner"] == binding.timing_owner
    assert occurrence["kind"] == "N0"
    assert occurrence["layer"] is None
    assert occurrence["members"] == ["residual_add", "rmsnorm"]
    assert occurrence["semantic_refs"] == []
    assert occurrence["typed"]["shapes"]["x"] == [2048, 5120]
    assert occurrence["typed"]["shapes"]["rstd"] == [2048, 1]
    assert occurrence["typed"]["attrs"]["eps"] == 1e-6
    assert occurrence["signature_matches"] is True
    json.dumps(result.to_dict(), sort_keys=True)


def test_native_and_model_bindings_share_identity():
    for kind, name, variant in (
            ("layernorm", "rmsnorm", "fused_attention_norm"),
            ("lm_head_sampler", "final_norm", "terminal_norm")):
        spec, model = _model_binding(_compiled(kind=kind, name=name))
        result = simulate_aclnn_call(
            _request(variant=variant), hardware=SKU)
        assert result.status == "predicted"
        assert result.binding.call_id == model.call_id
        assert result.binding.typed == model.typed
        assert result.binding.api["version"] == spec.api["version"]
        assert result.binding.source == spec.source
        assert result.binding.kernels == spec.kernels
        assert result.binding.timing_owner["op_id"] == spec.owner
        assert result.binding.op_id == model.op_id
        assert result.binding.source_boundary == model.source_boundary


def test_callers_consume_leaf_owned_work_and_members_are_replayable():
    cases = (
        ("native_call", "fused_attention_norm"),
        ("fused_attention_norm", "fused_attention_norm"),
        ("terminal_norm", "terminal_norm"),
    )
    for caller, leaf_variant in cases:
        request = (_request() if caller == "native_call"
                   else _request(variant=caller))
        result = simulate_aclnn_call(request, hardware=SKU)
        assert result.status == "predicted"
        prepared = adapter.prepare_add_rms_norm(
            rows=2048, hidden=5120, epsilon=1e-6,
            variant=leaf_variant, phase="prefill")
        physical = adapter.predict_add_rms_norm(
            prepared, hardware=SKU, phase="prefill")
        assert result.program["physical"]["leaf_prepared_variant"] == (
            None if caller == "native_call" else leaf_variant)
        assert result.program["cross_view"]["prepared_variant"] == (
            None if caller == "native_call" else leaf_variant)
        assert result.program["physical"]["physical_members"] == [
            "residual_add", "rmsnorm"]
        assert result.program["physical"]["shared_leaf"] == (
            "native_add_rms_norm.predict_add_rms_norm")
        # The declared leaf body is preserved exactly; the native catalog adds
        # only the kernel launch physics (declared prologue + per-owner fill),
        # so its terminal is the leaf body plus those floors, never a reprice.
        assert result.prediction["work"] == physical["work"]
        assert result.prediction["times_ms"] == physical["times_ms"]
        assert result.prediction["duration_basis"] == "native_catalog_chain"
        assert float(Fraction(result.prediction["duration_ms"])) > 0.0
        # The leaf body above is the CPU-side ledger; the native catalog adds
        # the declared launch floors, so the two are not required to be equal.
        assert float(Fraction(physical["duration_ms"])) > 0.0
        assert result.prediction["materialization"] == physical[
            "materialization"]
        assert result.prediction["materialization"]["rstd"] == "8192"
        members = list(result.binding.to_dict()["members"]["op_ids"])
        assert members == list(result.program["caller_plan"]["members"])
        assert members == list(result.occurrence["members"])
        replay_request = _request()
        replay_request["conditions"] = {
            "variant": caller, "members": members}
        replay = simulate_aclnn_call(replay_request, hardware=SKU)
        assert replay.status == "predicted"
        assert list(replay.binding.member_op_ids) == members


def test_terminal_norm_caller_plan_uses_one_effective_member():
    result = simulate_aclnn_call(
        _request(variant="terminal_norm"), hardware=SKU)
    assert result.status == "predicted"
    member = "O0:elementwise:final_norm"
    assert result.binding.member_op_ids == (member,)
    assert result.program["caller_plan"]["members"] == [member]
    assert result.occurrence["members"] == [member]
    assert result.binding.source_boundary == "kernel"
    assert result.occurrence["kind"] == "O0"
    assert result.occurrence["layer"] is None
    assert result.occurrence["typed"]["attrs"]["rows"] == 2048
    assert result.prediction["materialization"]["rstd"] == "8192"
    assert result.program["caller_plan"]["variant"] == "terminal_norm"
    assert result.program["physical"]["leaf_prepared_variant"] == (
        "terminal_norm")
    assert result.program["cross_view"]["prepared_variant"] == "terminal_norm"
    assert result.program["physical"]["common_program"] == (
        "elementwise.N0.add_rmsnorm")
    assert result.program["physical"]["caller_program"] == (
        "elementwise.O0.add_rmsnorm")
    assert result.program["physical"]["work"] == {
        "hbm": "83904512", "mte2": "41953280",
        "mte3": "41951232", "vector": "73564160",
    }
    replay_request = _request()
    replay_request["conditions"] = {
        "variant": "terminal_norm",
        "members": list(result.binding.to_dict()["members"]["op_ids"]),
    }
    replay = simulate_aclnn_call(replay_request, hardware=SKU)
    assert replay.status == "predicted"
    assert replay.binding.member_op_ids == (member,)
    assert replay.occurrence["members"] == [member]


def test_cross_view_typed_outputs_and_work_consistency():
    cases = (
        ("native_call", "fused_attention_norm"),
        ("fused_attention_norm", "fused_attention_norm"),
        ("terminal_norm", "terminal_norm"),
    )
    for caller, leaf_variant in cases:
        request = (_request() if caller == "native_call"
                   else _request(variant=caller))
        result = simulate_aclnn_call(request, hardware=SKU)
        prepared = adapter.prepare_add_rms_norm(
            rows=2048, hidden=5120, epsilon=1e-6,
            variant=leaf_variant, phase="prefill")
        physical = adapter.predict_add_rms_norm(
            prepared, hardware=SKU, phase="prefill")
        assert result.program["physical"]["leaf_prepared_variant"] == (
            None if caller == "native_call" else leaf_variant)
        assert result.program["cross_view"]["prepared_variant"] == (
            None if caller == "native_call" else leaf_variant)
        # The declared leaf body is preserved exactly; the native catalog adds
        # only the kernel launch physics (declared prologue + per-owner fill),
        # so its terminal is the leaf body plus those floors, never a reprice.
        assert result.prediction["work"] == physical["work"]
        assert result.prediction["times_ms"] == physical["times_ms"]
        assert result.prediction["duration_basis"] == "native_catalog_chain"
        assert float(Fraction(result.prediction["duration_ms"])) > 0.0
        # The leaf body above is the CPU-side ledger; the native catalog adds
        # the declared launch floors, so the two are not required to be equal.
        assert float(Fraction(physical["duration_ms"])) > 0.0
        assert result.prediction["materialization"] == physical[
            "materialization"]
        for name, shape in result.occurrence["typed"]["shapes"].items():
            descriptor = prepared["descriptors"][name]
            assert list(descriptor["shape"]) == list(shape)
            expected = "fp32" if name == "rstd" else "bf16"
            assert descriptor["dtype"] == expected
            assert descriptor["layout"] == "ND"
        assert prepared["attrs"]["rstd_bytes"] == 2048 * 4
        assert result.program["cross_view"]["rstd_required_bytes"] == 8192
        assert result.program["cross_view"]["rstd_materialized_bytes"] == (
            "8192")
        assert result.program["cross_view"]["consistent"] is True


def test_chain_first_add_uses_the_same_leaf_work():
    from mindie_sim.prediction.piecewise import (
        native_r31_dynamic_quant_v2 as quant,
    )
    add_prepared = adapter.prepare_add_rms_norm(
        rows=2048, hidden=5120, epsilon=1e-6,
        variant="fused_attention_norm", phase="prefill")
    quant_prepared = quant.prepare_dynamic_quant_v2(
        rows=2048, width=5120, is_symmetrical=True, phase="prefill")
    chain = adapter.predict_add_rms_norm_quant_chain(
        add_prepared, quant_prepared, hardware=SKU, phase="prefill")
    chain_add = chain["calls"][0]
    for request in (_request(), _request(variant="fused_attention_norm")):
        result = simulate_aclnn_call(request, hardware=SKU)
        # The chain prices the same declared leaf work; the independent native
        # call adds the declared launch floors on top of that same body.
        assert result.prediction["work"] == chain_add["work"]
        assert result.prediction["duration_basis"] == "native_catalog_chain"
        assert float(Fraction(result.prediction["duration_ms"])) > 0.0
        assert float(Fraction(chain_add["duration_ms"])) > 0.0
        assert chain_add["materialization"] == result.prediction[
            "materialization"]


def test_new_shapes_and_selected_rows_generalize():
    result = simulate_aclnn_call(
        _request(hidden=4096, rows=7, epsilon=1e-5,
                 sequence_rows="selected", sequence_ranks=4),
        hardware=SKU)
    assert result.status == "predicted"
    assert result.binding.typed["shapes"]["x"] == [4096]
    assert result.occurrence["typed"]["shapes"]["x"] == [7, 4096]
    assert result.occurrence["typed"]["shapes"]["gamma"] == [4096]
    assert result.occurrence["typed"]["shapes"]["rstd"] == [7, 1]
    assert result.occurrence["typed"]["attrs"]["physical_rows"] == 7
    assert result.occurrence["typed"]["attrs"]["row_basis"] == "selected"
    assert Fraction(result.prediction["duration_ms"]) > 0
    assert result.prediction["materialization"]["rstd"] == "28"

    decode = simulate_aclnn_call(
        _request(hidden=4096, rows=1, sequence_rows="", sequence_ranks=1),
        hardware=SKU)
    assert decode.status == "predicted"
    assert decode.prediction["environment"]["phase"] == "decode"
    assert decode.occurrence["typed"]["shapes"]["x"] == [1, 4096]
    assert decode.prediction["materialization"]["rstd"] == "4"


def test_gap_returns_a_replayable_minimal_input():
    request = _request()
    request["tensors"].pop("gamma")
    request["attrs"] = {}
    gap = simulate_aclnn_call(request, hardware=SKU)
    assert gap.status == "insufficient_fields"
    assert gap.prediction is None
    assert gap.binding is None
    assert "tensors.gamma" in gap.replay["missing"]
    assert "attrs.epsilon" in gap.replay["missing"]
    assert gap.replay["replayable"] is False
    assert "gamma" not in gap.replay["tensors"]
    assert gap.to_dict()["fallback_performed"] is False

    replay = copy.deepcopy(gap.replay)
    replay["tensors"]["gamma"] = {
        "shape": [5120], "dtype": "bf16", "role": "weight"}
    replay["attrs"]["epsilon"] = 1e-6
    predicted = simulate_aclnn_call(replay, hardware=SKU)
    assert predicted.status == "predicted"
    assert predicted.binding.call_id == simulate_aclnn_call(
        _request(), hardware=SKU).binding.call_id


def test_version_mismatch_and_unknown_entry_never_fall_back():
    mismatch = simulate_aclnn_call(
        _request(version={"api": "cann-9.0.0"}), hardware=SKU)
    assert mismatch.status == "version_mismatch"
    assert mismatch.prediction is None
    assert mismatch.binding is None
    assert mismatch.to_dict()["fallback_performed"] is False
    with pytest.raises(ModelingError):
        simulate_aclnn_call(
            _request(version={"api": "cann-9.0.0"}), hardware=SKU,
            strict=True)

    unknown = simulate_aclnn_call({"api": {"entry": "aclnnNotDeclared"}})
    assert unknown.status == "no_match"
    assert unknown.prediction is None
    assert "aclnnAddRmsNorm" in unknown.replay["declared_entries"]

    wrong_dtype = simulate_aclnn_call(_request(dtype="fp16"), hardware=SKU)
    assert wrong_dtype.status == "no_match"
    assert wrong_dtype.prediction is None
    assert "native_abi_tensor_dtype_mismatch" in " ".join(
        wrong_dtype.reasons)


def test_declared_catalogue_preserves_fused_and_one_to_many_relations():
    inventory = a5_native_abi_inventory()
    assert "aclnnAddRmsNorm" in inventory["implemented"]
    assert inventory["totals"]["implemented_entries"] == 27
    assert inventory["totals"]["declared_entries"] == 27
    entries = {entry["entry"]: entry for entry in inventory["entries"]}
    fia = entries["torch_npu.npu_fused_infer_attention_score"]
    assert fia["dispatch"] == "paged_fia_build"
    assert fia["variants"][0]["name"] == "ordinary_paged_fia"
    add = entries["aclnnAddRmsNorm"]
    assert [variant["relation"] for variant in add["variants"]] == [
        "native_call", "many_to_one_fused", "one_to_one_fused"]
    assert add["variants"][0]["members"] == []
    assert add["variants"][2]["members"] == ["O0:elementwise:final_norm"]
    assert add["kernels"][0]["name"] == "AddRmsNorm"
    physical_call = inventory["physical_call"]
    assert "prepared_variant" not in physical_call
    assert physical_call["physical_entry_unique"] is True
    assert [variant["name"] for variant in physical_call[
        "caller_variants"]] == [
            "native_call", "fused_attention_norm", "terminal_norm"]
    assert physical_call["caller_to_leaf_variant"] == {
        "native_call": "fused_attention_norm",
        "fused_attention_norm": "fused_attention_norm",
        "terminal_norm": "terminal_norm",
    }
    gdn = entries["aclnnChunkGatedDeltaRuleFwd"]
    assert gdn["implemented"] is True
    assert gdn["variants"][0]["relation"] == "one_to_many"
    assert [kernel["name"] for kernel in gdn["kernels"]] == [
        "Cast", "Transpose", "Prepare32", "H24", "O32"]

    request = {
        "api": {"namespace": "aclnn",
                "entry": "aclnnChunkGatedDeltaRuleFwd"},
        "version": {},
        "tensors": {
            name: {"shape": [1, 2], "dtype": "bf16", "layout": "ND"}
            for name in ("q", "k", "v", "g", "beta", "state")
        },
        "attrs": {"chunk_size": 64, "scale": 0.125},
    }
    gap = simulate_aclnn_call(request)
    assert gap.status in {"insufficient_fields", "no_match"}
    assert gap.prediction is None
    assert gap.replay["replayable"] is False
    assert gap.replay["missing"]
    assert len(gap.spec["kernels"]) == 5
    assert gap.spec["variants"][0]["members"] == ["prepare", "gated_delta"]


def test_profiler_and_model_consumers_reuse_the_same_binding():
    result = simulate_aclnn_call(_request(), hardware=SKU)
    binding = result.binding
    joined = join_profiler_rows(
        [{"name": "aclnnAddRmsNorm"}, {"name": "AddRmsNorm"},
         {"name": "UnrelatedKernel"}], [binding])
    assert joined["joined"] == 2
    assert joined["unmatched"] == 1
    assert [row["basis"] for row in joined["records"]] == [
        "declared_entry", "declared_child_kernel"]
    assert all(row["call_ids"] == [binding.call_id]
               for row in joined["records"])
    coverage = coverage_from_records([binding.to_dict()], emitted=1)
    assert coverage["totals"]["bound"] == 1
    assert coverage["totals"]["unresolved"] == 0
    assert binding.to_dict()["schema"] == "mindie-sim/native-call-binding/1"


def test_inconsistent_gamma_is_an_explicit_no_match():
    request = _request()
    request["tensors"]["gamma"]["shape"] = [4095]
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "no_match"
    assert result.prediction is None
    assert result.program is None
    assert "add_rmsnorm_abi_not_supported" in result.reasons
    assert "tensor_shape_mismatch:gamma" in result.reasons


def test_non_a5_environment_is_rejected_without_substitution():
    result = simulate_aclnn_call(_request(), hardware="Ascend910B2")
    assert result.status == "hardware_mismatch"
    assert result.prediction is None
    assert "A5" in " ".join(result.reasons)


class _AttentionTheta:
    matrix = {}
    heads = {"q": 6, "kv": 1, "dk": 256, "dv": 256}
    quant = {}
    extras = {"causal": 1, "window": 0}

    def find(self, key, default=None):
        return self.extras.get(key, default)


def test_ordinary_attention_keeps_selected_fia_call_without_r35_source():
    """F2: no retained R35 image must not erase the selected FIA call."""
    spec = native_calls._attention_spec(
        _AttentionTheta(), {"family": "attention"}, kind="gqa_attention",
        name="attention", phase="decode", feature_metadata=())
    assert spec is not None
    assert spec.api["entry"] == "torch_npu.npu_fused_infer_attention_score"
    assert spec.typed["shapes"]["q"] == [6, 256]
    assert spec.source is None
    assert spec.kernels[0]["evidence"] == "unresolved"
    assert "deployment_image_identity_not_established" in spec.unresolved
    assert "host_caller_source_not_retained" in spec.unresolved

    retained = native_calls._attention_spec(
        _AttentionTheta(), {"family": "attention"}, kind="gqa_attention",
        name="attention", phase="decode",
        feature_metadata=(("image_id", "sha256:" + native_calls.R35_IMAGE_SHA),))
    assert retained is not None
    assert retained.source["revision"] == "a5-cann-r35-retained-source"
    assert retained.typed["attrs"]["input_layout"] == "TND"



def test_production_a0_attention_keeps_selected_fia_call():
    """Minimal production reproduction of QA0 F2 on the real A0 path.

    `examples/cases/qwen3.8-27b.json` currently declares an `fp8_e4m3` peak
    on an A3 machine without an A3 native fp8 mode, so `parse_case` raises
    before this path.  Only that out-of-scope case/profile peak is removed in
    memory; the model instantiate/compile/evaluate production path is real.
    """
    from pathlib import Path
    from mindie_sim.prediction.piecewise.compiler import (
        ImplementationPolicy, PipeSupply,
    )
    from mindie_sim.prediction.piecewise.geometry import (
        RequestGeometry, WorkloadGeometry,
    )
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.prediction.subgraphs.instantiate import (
        FeatureConfig, instantiate_model,
    )

    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/qwen3.8-27b.json").read_text())
    raw["machine"]["compute_peaks"].pop("fp8_e4m3", None)
    case = parse_case(raw)
    features = FeatureConfig.from_model(
        case.model, tp=4, dp=1, ep=1, pp=1, cores=64)
    instantiated = instantiate_model(case.model, features)
    instance = next(
        item for layer in instantiated.layers for item in layer.subgraphs
        if item.kind == "gqa_attention")
    domain = AnalysisDomain(
        max_tokens=10 ** 9, max_moe_rows=10 ** 9,
        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
        max_index_rows=10 ** 12, max_work=10 ** 15, max_q=10 ** 9,
        max_s_kv=10 ** 9, max_seq=10 ** 9, max_batch=10 ** 6)
    compiler = OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="bf16"),
        ImplementationPolicy(cores=64), domain=domain)
    geometry = WorkloadGeometry(
        (RequestGeometry(1, 1024, phase="decode"),), phase="decode",
        cohort_batch=1, local_batch=1)
    cost = next(
        cost for cost in compiler.compile(instance).evaluate(geometry).operators
        if cost.name == "attention")
    assert cost.native_call["api"]["entry"] == (
        "torch_npu.npu_fused_infer_attention_score")
    assert cost.native_occurrence["typed"]["shapes"]["q"][0] == 1
    assert (cost.native_occurrence["typed"]["shapes"]["q"][-2:]
            == cost.native_call["typed"]["shapes"]["q"])
    assert "deployment_image_identity_not_established" in (
        cost.native_occurrence["unresolved"])


def test_bare_native_call_has_no_model_attribution():
    result = simulate_aclnn_call(_request(), hardware=SKU)
    assert result.status == "predicted"
    assert result.binding.op_id == "native:aclnnAddRmsNorm"
    assert result.binding.timing_owner == {
        "op_id": "native:aclnnAddRmsNorm", "scope": "call"}
    assert result.binding.member_op_ids == ()
    assert result.occurrence["kind"] == "native"
    assert result.occurrence["mode"] == "native_call"
    assert result.occurrence["layer"] is None
    assert result.occurrence["members"] == []
    assert result.occurrence["semantic_refs"] == []
    assert result.program["caller_plan"]["variant"] == "native_call"
    assert result.program["caller_plan"]["members"] == []
    assert result.program["physical"]["caller_program"] is None
    assert result.program["physical"]["leaf_prepared_variant"] is None
    assert result.program["physical"]["physical_members"] == [
        "residual_add", "rmsnorm"]
    assert "caller_variant" not in result.program["physical"]
    assert "caller_kind" not in result.program["physical"]
    assert result.binding.bound is True
    assert result.binding_bound is True
    replay_request = _request()
    replay_request["conditions"] = {
        "variant": "native_call",
        "members": list(result.binding.to_dict()["members"]["op_ids"]),
    }
    replay = simulate_aclnn_call(replay_request, hardware=SKU)
    assert replay.status == "predicted"
    assert replay.binding.member_op_ids == ()
    assert replay.occurrence["members"] == []


def test_explicit_layer_is_preserved_but_not_invented():
    bare = simulate_aclnn_call(_request(), hardware=SKU)
    assert bare.occurrence["layer"] is None
    explicit = simulate_aclnn_call(
        _request(variant="fused_attention_norm"), hardware=SKU)
    assert explicit.occurrence["layer"] is None
    request = _request(variant="fused_attention_norm")
    request["conditions"]["layer"] = 3
    layered = simulate_aclnn_call(request, hardware=SKU)
    assert layered.status == "predicted"
    assert layered.occurrence["layer"] == 3
    assert layered.program["caller_plan"]["layer"] == 3


def test_owner_and_members_conflicts_are_rejected():
    cases = (
        ({"variant": "fused_attention_norm",
          "owner": "O0:elementwise:final_norm"},
         "owner_conflicts_variant"),
        ({"variant": "fused_attention_norm", "members": ["final_norm"]},
         "members_conflict"),
        ({"owner": "fused:add_rmsnorm", "members": ["residual_add"]},
         "members_conflict"),
        ({"members": ["not_a_member"]}, "members_not_declared"),
        ({"variant": "no_such_variant"}, "variant_not_declared"),
    )
    for conditions, reason in cases:
        request = _request()
        request["conditions"].update(conditions)
        result = simulate_aclnn_call(request, hardware=SKU)
        assert result.status == "no_match"
        assert reason in " ".join(result.reasons)
        assert result.prediction is None
        assert result.binding is None


def test_no_image_does_not_claim_r31_binary():
    for version in (
            {"api": "cann-9.2.0"},
            {"api": "cann-9.2.0", "source": "cann-9.2.0-r31"}):
        result = simulate_aclnn_call(
            _request(version=version), hardware=SKU)
        assert result.status == "predicted"
        assert result.binding_bound is False
        assert result.binding.bound is False
        assert result.binding.source is None
        assert result.binding.evidence["source"] == "unresolved"
        assert result.binding.kernels[0].get("source_sha256") is None
        assert result.binding.kernels[0].get("source_path") is None
        assert result.binding.kernels[0]["evidence"] == (
            "candidate_declaration")
        assert result.source_establishment["established"] is False
        assert result.source_establishment["candidate_source"]["sha256"] == (
            adapter.CANN_LIBOPAPI["sha256"])
        assert adapter.CANN_LIBOPAPI["sha256"] not in json.dumps(
            result.binding.to_dict(), sort_keys=True)
        assert result.occurrence["status"] == "candidate"
        assert "deployment_image_identity_not_established" in (
            result.binding.unresolved)

    bound = simulate_aclnn_call(_request(), hardware=SKU)
    assert bound.binding_bound is True
    assert bound.binding.bound is True
    assert bound.binding.source["sha256"] == adapter.CANN_LIBOPAPI["sha256"]
    assert bound.binding.kernels[0]["source_sha256"] == (
        adapter.CANN_LIBOPAPI["sha256"])


def test_members_only_selects_the_matching_caller_plan():
    request = _request()
    request["conditions"] = {"members": ["residual_add", "rmsnorm"]}
    result = simulate_aclnn_call(request, hardware=SKU)
    assert result.status == "predicted"
    assert result.program["caller_plan"]["variant"] == "fused_attention_norm"
    assert result.program["caller_plan"]["members"] == [
        "residual_add", "rmsnorm"]
    bare = _request()
    bare["conditions"] = {"members": []}
    native = simulate_aclnn_call(bare, hardware=SKU)
    assert native.status == "predicted"
    assert native.program["caller_plan"]["variant"] == "native_call"
    assert native.program["caller_plan"]["members"] == []
    o0 = _request()
    o0["conditions"] = {
        "members": ["O0:elementwise:final_norm"]}
    terminal = simulate_aclnn_call(o0, hardware=SKU)
    assert terminal.status == "predicted"
    assert terminal.program["caller_plan"]["variant"] == "terminal_norm"
    assert terminal.program["caller_plan"]["members"] == [
        "O0:elementwise:final_norm"]


def test_public_json_projection_is_consistent_for_three_callers():
    cases = (
        ("native_call", [], "native", None, None),
        ("fused_attention_norm", ["residual_add", "rmsnorm"], "N0",
         "elementwise.N0.add_rmsnorm", "fused_attention_norm"),
        ("terminal_norm", ["O0:elementwise:final_norm"], "O0",
         "elementwise.O0.add_rmsnorm", "terminal_norm"),
    )
    for (caller, expected_members, expected_kind, expected_caller_program,
         expected_leaf_variant) in cases:
        request = (_request() if caller == "native_call"
                   else _request(variant=caller))
        result = simulate_aclnn_call(request, hardware=SKU)
        assert result.status == "predicted"
        assert result.program is result.prediction["program"]
        payload = result.to_dict()
        assert payload["program"] == payload["prediction"]["program"]
        program = payload["prediction"]["program"]
        assert set(program) == {"physical", "caller_plan", "cross_view"}
        assert program["caller_plan"] == payload["program"]["caller_plan"]
        assert program["caller_plan"]["variant"] == caller
        assert program["caller_plan"]["members"] == expected_members
        assert program["physical"]["physical_program"] == (
            "elementwise.N0.add_rmsnorm")
        assert program["physical"]["caller_program"] == expected_caller_program
        assert program["physical"]["physical_members"] == [
            "residual_add", "rmsnorm"]
        assert program["physical"]["leaf_prepared_variant"] == (
            expected_leaf_variant)
        assert "caller_variant" not in program["physical"]
        assert "caller_kind" not in program["physical"]
        assert program["cross_view"]["caller_variant"] == caller
        assert program["cross_view"]["caller_members"] == expected_members
        assert program["cross_view"]["physical_members"] == [
            "residual_add", "rmsnorm"]
        assert program["cross_view"]["prepared_variant"] == (
            expected_leaf_variant)
        assert payload["binding"]["members"]["op_ids"] == expected_members
        assert payload["occurrence"]["members"] == expected_members
        assert payload["occurrence"]["kind"] == expected_kind
        assert payload["occurrence"]["semantic_refs"] == []
        assert list(payload["invocation"]["conditions"]["members"]) == (
            expected_members)
        assert payload["prediction"]["work"] == program["physical"]["work"]
        assert payload["prediction"]["duration_ms"] == str(
            Fraction(payload["prediction"]["duration_ms"]))
        assert json.loads(json.dumps(payload))["prediction"]["program"] == (
            program)
        replay_request = _request()
        replay_request["conditions"] = {
            "variant": caller, "members": expected_members}
        replay = simulate_aclnn_call(replay_request, hardware=SKU)
        assert replay.status == "predicted"
        assert list(replay.binding.member_op_ids) == expected_members


def test_missing_epsilon_replay_does_not_add_source_or_image():
    request = _request(version={"api": "cann-9.2.0"})
    del request["attrs"]["epsilon"]
    failed = simulate_aclnn_call(request)
    assert failed.status == "insufficient_fields"
    replay = failed.replay
    assert replay["version"] == {"api": "cann-9.2.0"}
    assert "image_id" not in replay["version"]
    assert "source" not in replay["version"]
    assert replay["conditions"] == {
        "sequence_rows": "shard", "sequence_ranks": 2}
    assert "attrs.epsilon" in replay["missing"]
    assert failed.to_dict()["replay_minimal"] == replay

    replay["attrs"]["epsilon"] = 1e-6
    completed = simulate_aclnn_call(replay, hardware=SKU)
    assert completed.status == "predicted", completed.reasons
    assert completed.binding_bound is False
    assert completed.binding.source is None
    assert completed.binding.kernels


def test_wrong_api_version_replay_keeps_the_wrong_version():
    request = _request(version={"api": "cann-9.0.0"})
    failed = simulate_aclnn_call(request, hardware=SKU)
    assert failed.status == "version_mismatch"
    replay = failed.replay
    assert replay["version"] == {"api": "cann-9.0.0"}
    assert "image_id" not in replay["version"]
    assert failed.to_dict()["replay_minimal"] == replay

    again = simulate_aclnn_call(replay, hardware=SKU)
    assert again.status == "version_mismatch"
    assert again.prediction is None


def test_members_only_o0_replay_does_not_invent_native_variant():
    request = _request()
    del request["attrs"]["epsilon"]
    request["conditions"] = {"members": ["O0:elementwise:final_norm"]}
    failed = simulate_aclnn_call(request, hardware=SKU)
    assert failed.status == "insufficient_fields"
    replay = failed.replay
    assert replay["conditions"] == {
        "members": ["O0:elementwise:final_norm"]}
    assert "variant" not in replay["conditions"]
    assert failed.to_dict()["replay_minimal"] == replay

    replay["attrs"]["epsilon"] = 1e-6
    completed = simulate_aclnn_call(replay, hardware=SKU)
    assert completed.status == "predicted", completed.reasons
    assert completed.binding.member_op_ids == (
        "O0:elementwise:final_norm",)
    assert completed.program["caller_plan"]["variant"] == "terminal_norm"
