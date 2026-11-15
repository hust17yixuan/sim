"""Regression tests for draft sparse/window KV inheritance and generalized RecoverSSM.

[Role] Verify PR #33 draft LayerPlan sparse/window/compression/mHC inheritance at
  128K decode and family-agnostic RecoverSSM lowering across all linear-attention
  recurrences (kda, gdn, mamba, gla, deltanet, lightning, retnet).
[Invariants] Draft attention KV reads stay bounded by topk/window at 128K context;
  RecoverSSM cuts speculative Verify (q > 1) state writes from q * O(Dk * Dv) to
  q * O(Dk + Dv) + 1 * O(Dk * Dv) while preserving q == 1 and recover_ssm=False.
[Timeouts & Locks] Tier 0 (<5s), pure in-memory catalog and graph evaluation.
[Side-Effect Policy] Isolates HOME and MINDIE_* env roots; writes no host state.
"""
from __future__ import annotations

from dataclasses import replace
import pytest

from mindie_sim import (
    Case, Hardware, LayerPlan, ModelSpec, ParallelSpec,
    RequestSpec, Software, WorkloadSpec, predict, resolve_hardware_sku,
)
from mindie_sim.backends.ascend import materialize as materialize_ascend
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.linear_state_physics import (
    LINEAR_STATE_KINDS, LinearLayer, compare_linear_state_memory,
)
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.recover_ssm_work import record_head_bytes
from mindie_sim.prediction.speculation import draft_layer_plans
from mindie_sim.recoverssm_config import RecoverSsmConfig, validate_recoverssm
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.workload import IterationPlan, RequestView


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(tmp_path / "diag"))


def _hardware(sku: str = "Ascend950DT_9582", count: int = 8) -> Hardware:
    resolved = resolve_hardware_sku(sku)
    return Hardware.from_mapping(materialize_ascend({"sku": resolved, "num_devices": count}))


def _software(
    tp: int = 8,
    *,
    recover_ssm: bool = False,
    metadata: tuple[tuple[str, object], ...] = (),
    kernels: dict | None = None,
) -> Software:
    return Software(
        parallel=ParallelSpec(tp=tp),
        recover_ssm=recover_ssm,
        metadata=metadata,
        kernels=kernels or {},
    )


def _draft_attn_kv_hbm(ops, prefix: str = "draft") -> float:
    attn_families = {"mla_decode_score", "mla_decode_value", "attention"}
    return sum(
        float(op.hbm_bytes)
        for op in ops
        if op.op_id.startswith(prefix) and op.family in attn_families
    )


def test_glm53_flash_and_glm53_draft_sparse_mla_and_mhc_inheritance():
    flash = load_model("glm-5.3-flash")
    plans = draft_layer_plans(flash)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.attention == "mla" and plan.architecture == "mla_dsa"
    assert plan.kv_cache_elements == 512 and plan.owns_long_kv is True
    assert plan.indexer is True and plan.index_topk == 2048 and plan.index_kpool == 4
    assert plan.owns_index_k is False and plan.owns_topk is True
    assert plan.indexer_type == "shared"
    assert plan.hyper_connections is True and plan.hc_mult == 4
    assert plan.hc_sinkhorn_iters == 20

    mach = _hardware(count=4)
    sw = _software(tp=4, metadata=(("draft_mode", "mtp"),))
    fe = Frontend()
    ops_8k = fe(
        IterationPlan((RequestView("r0", 6, 8192, True, "decode", draft_proposal_tokens=5),)),
        flash, sw, mach,
    ).ops
    ops_128k = fe(
        IterationPlan((RequestView("r0", 6, 131072, True, "decode", draft_proposal_tokens=5),)),
        flash, sw, mach,
    ).ops
    kv_8k = _draft_attn_kv_hbm(ops_8k)
    kv_128k = _draft_attn_kv_hbm(ops_128k)
    assert kv_8k > 0
    assert kv_128k == pytest.approx(kv_8k)
    assert not any(
        op.op_id.startswith("draft") and "index_k_cache" in op.op_id
        for op in ops_128k
    )
    assert any(
        op.op_id.startswith("draft") and op.family in {"indexer", "topk"}
        for op in ops_128k
    )

    glm53 = load_model("glm-5.3")
    p53 = draft_layer_plans(glm53)[0]
    assert p53.indexer is True and p53.index_topk == 2048
    assert p53.owns_index_k is False and p53.owns_topk is True
    assert p53.kv_cache_elements == 576 and p53.owns_long_kv is True
    assert p53.hyper_connections is False and p53.hc_mult == 0


def test_deepseek_v41_flash_dspark_swa_only_zero_128k_inflation():
    model = load_model("deepseek-v4.1-flash")
    plans = draft_layer_plans(model)
    assert len(plans) == 3
    for plan in plans:
        assert plan.architecture == "swa"
        assert plan.cache_kind == "window_compressed"
        assert plan.window == 128
        assert plan.owns_long_kv is False
        assert plan.compress_ratio == 0
        assert plan.indexer is False
        assert plan.kv_cache_elements == 512

    mach = _hardware(count=4)
    sw = _software(tp=4, metadata=(("draft_mode", "mtp"),))
    fe = Frontend()
    ops_8k = fe(
        IterationPlan((RequestView("r0", 4, 8192, True, "decode", draft_proposal_tokens=3),)),
        model, sw, mach,
    ).ops
    ops_128k = fe(
        IterationPlan((RequestView("r0", 4, 131072, True, "decode", draft_proposal_tokens=3),)),
        model, sw, mach,
    ).ops
    kv_8k = _draft_attn_kv_hbm(ops_8k)
    kv_128k = _draft_attn_kv_hbm(ops_128k)
    assert kv_8k > 0
    assert kv_128k == pytest.approx(kv_8k)
    dense_128k_floor = 3 * 131072 * 576 * 2
    assert kv_128k < dense_128k_floor / 200.0


@pytest.mark.parametrize(
    "model_id",
    ("deepseek-v4-pro-0813", "deepseek-v4-10t", "deepseek-v4-flash-0731"),
)
def test_deepseek_v4_swa_and_mixed_compressed_draft_layers(model_id: str):
    model = load_model(model_id)
    plans = draft_layer_plans(model)
    assert len(plans) == 3
    for swa in plans:
        assert swa.architecture == "swa"
        assert swa.cache_kind == "window_compressed"
        assert swa.window == 128
        assert swa.owns_long_kv is False
        assert swa.compress_ratio == 0
        assert swa.indexer is False

    mach = _hardware(count=8)
    sw = _software(tp=8, metadata=(("draft_mode", "mtp"),))
    fe = Frontend()
    ops_8k = fe(
        IterationPlan((RequestView("r0", 4, 8192, True, "decode", draft_proposal_tokens=3),)),
        model, sw, mach,
    ).ops
    ops_128k = fe(
        IterationPlan((RequestView("r0", 4, 131072, True, "decode", draft_proposal_tokens=3),)),
        model, sw, mach,
    ).ops
    assert _draft_attn_kv_hbm(ops_128k) == pytest.approx(_draft_attn_kv_hbm(ops_8k))

    # Also verify mixed explicit draft_compress_ratios=(4, 0, 128) inheritance
    mtp = model.auxiliary.mtp
    mixed_mtp = replace(mtp, draft_compress_ratios=(4, 0, 128))
    mixed_model = replace(model, auxiliary=replace(model.auxiliary, mtp=mixed_mtp))
    p0, p1, p2 = draft_layer_plans(mixed_model)
    assert p0.architecture == "csa" and p0.compress_ratio == 4 and p0.compressor_kind == "weighted_pool"
    assert p0.owns_long_kv is True and p0.indexer is True and p0.index_topk > 0
    assert p1.architecture == "swa" and p1.compress_ratio == 0 and p1.owns_long_kv is False and not p1.indexer
    assert p2.architecture == "hca" and p2.compress_ratio == 128 and p2.compressor_kind == "weighted_pool"
    assert p2.owns_long_kv is True and p2.indexer is False

    mixed_128k = fe(
        IterationPlan((RequestView("r0", 4, 131072, True, "decode", draft_proposal_tokens=3),)),
        mixed_model, sw, mach,
    ).ops
    dense_128k_uncompressed = 3 * 131072 * p0.kv_cache_elements * 2
    assert 0 < _draft_attn_kv_hbm(mixed_128k) < dense_128k_uncompressed / 20.0


def test_qwen38_flash_next_and_kimi_k3_draft_boundaries():
    qwen = load_model("qwen3.8-flash-next")
    plans = draft_layer_plans(qwen)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.attention == "gqa" and plan.architecture == "gated_gqa"
    assert plan.owns_long_kv is True
    assert plan.indexer is True and plan.index_topk == 2048 and plan.index_kpool == 4
    assert plan.kv_cache_elements == 2 * qwen.auxiliary.mtp.num_kv_heads * qwen.auxiliary.mtp.head_dim
    assert plan.hyper_connections is True and plan.hc_mult == 4

    mach = _hardware(count=4)
    sw = _software(tp=4, metadata=(("draft_mode", "mtp"),))
    fe = Frontend()
    ops_8k = fe(
        IterationPlan((RequestView("r0", 3, 8192, True, "decode", draft_proposal_tokens=2),)),
        qwen, sw, mach,
    ).ops
    ops_128k = fe(
        IterationPlan((RequestView("r0", 3, 131072, True, "decode", draft_proposal_tokens=2),)),
        qwen, sw, mach,
    ).ops
    assert _draft_attn_kv_hbm(ops_128k) == pytest.approx(_draft_attn_kv_hbm(ops_8k))

    kimi = load_model("kimi-k3")
    kimi_draft_plans = draft_layer_plans(kimi, kind="dspark")
    assert len(kimi_draft_plans) > 0
    assert all(not layer.indexer and layer.owns_long_kv for layer in kimi_draft_plans)


def test_generalized_recover_ssm_physics_and_config_across_all_linear_kinds():
    expected = frozenset({"kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet"})
    assert LINEAR_STATE_KINDS == expected

    for kind in sorted(expected):
        layer = LinearLayer(
            name=f"layer_{kind}", kind=kind, index=0,
            num_heads=16, key_dim=128, value_dim=128, conv_width=6144, recurrent_dtype="fp32",
        )
        cfg = RecoverSsmConfig(
            use_replayssm=True, speculative_k=8,
            enable_prefix_caching=False, mamba_cache_mode="none",
            tp=2, model_type=f"generic_{kind}",
        )
        report = compare_linear_state_memory(
            cfg, (layer,), sequence_tokens=4096, active_sequences=4,
            allow_generic_model=True,
        )
        assert report["applicable"] is True
        assert report["saved_bytes"] > 0
        assert report["saved_full_state_slots"] == 8
        assert report["ratio"] > 3.0

    for bad_kind in ("mla", "gqa", "unknown"):
        with pytest.raises(ModelingError, match="linear_state_layer_kind_invalid"):
            LinearLayer(
                name="bad", kind=bad_kind, index=0,
                num_heads=8, key_dim=128, value_dim=128, conv_width=3072,
            )

    qwen_cfg = RecoverSsmConfig(
        use_replayssm=True, speculative_k=8,
        enable_prefix_caching=False, mamba_cache_mode="none",
        tp=4, model_type="qwen3_next", generic_backend=True,
    )
    qwen_val = validate_recoverssm(qwen_cfg, strict=False)
    assert qwen_val.ok is True and qwen_val.model_is_glm is False
    assert qwen_cfg.glm_kda_backend is False

    with pytest.raises(ModelingError, match="linear_state_spec_tokens_exceed_kda_limit"):
        RecoverSsmConfig(
            use_replayssm=True, speculative_k=7,
            enable_prefix_caching=False, mamba_cache_mode="none",
            model_type="glm5_next",
        )


def test_software_recover_ssm_schema_validation_and_roundtrip():
    assert Software().recover_ssm is False
    sw = Software.from_mapping({"recover_ssm": True, "tp": 2})
    assert sw.recover_ssm is True and sw.parallel.tp == 2
    roundtripped = Software.from_mapping(sw.to_dict())
    assert roundtripped.recover_ssm is True and roundtripped.parallel.tp == 2
    for bad in (1, "true", None):
        with pytest.raises(ModelingError, match="runtime_recover_ssm_invalid"):
            Software.from_mapping({"recover_ssm": bad})


def _recurrent_write_bytes(ops) -> float:
    total = 0.0
    for op in ops:
        if op.op_id.startswith("draft"):
            continue
        meta = dict(op.metadata)
        if op.family in {"recurrent_scan", "recurrent_vector"} or meta.get("recover_ssm_stage") == "commit":
            if op.plan_required:
                total += sum(
                    float(acc["bytes"])
                    for acc in meta.get("state_accesses", ())
                    if acc.get("direction") == "write"
                ) or float(op.mte3_bytes)
            else:
                total += float(op.mte3_bytes)
    return total


def test_glm53_flash_kda_and_qwen38_gdn_recover_ssm_verify_hbm_savings():
    glm = load_model("glm-5.3-flash")
    mach_a5 = _hardware(count=4)
    fe = Frontend()
    spec_view = RequestView("r0", 6, 4096, True, "decode", draft_proposal_tokens=5)

    ops_base = fe(
        IterationPlan((spec_view,)), glm,
        _software(tp=4, recover_ssm=False, metadata=(("arch_family", "a5"),)), mach_a5,
    ).ops
    ops_rec = fe(
        IterationPlan((spec_view,)), glm,
        _software(tp=4, recover_ssm=True, metadata=(("arch_family", "a5"),)), mach_a5,
    ).ops
    assert any(dict(op.metadata).get("recover_ssm_stage") == "commit" for op in ops_rec)
    assert _recurrent_write_bytes(ops_rec) < 0.35 * _recurrent_write_bytes(ops_base)

    q1_view = RequestView("r0", 1, 4096, True, "decode")
    q1_base = fe(IterationPlan((q1_view,)), glm, _software(tp=4, recover_ssm=False), mach_a5).ops
    q1_rec = fe(IterationPlan((q1_view,)), glm, _software(tp=4, recover_ssm=True), mach_a5).ops
    assert _recurrent_write_bytes(q1_rec) == pytest.approx(_recurrent_write_bytes(q1_base))

    qwen = load_model("qwen3.8-flash-next")
    gdn_view = RequestView("r0", 5, 4096, True, "decode", draft_proposal_tokens=4)
    for impl in ("analytical", "arch35_fp32"):
        kernels = {"lowerings": {"gdn_decode": {
            "implementation": "vllm_ascend_gdn_decode",
            "options": {"recurrent_implementation": impl},
        }}}
        g_base = fe(
            IterationPlan((gdn_view,)), qwen,
            _software(tp=4, recover_ssm=False, kernels=kernels), mach_a5,
        ).ops
        g_rec = fe(
            IterationPlan((gdn_view,)), qwen,
            _software(tp=4, recover_ssm=True, kernels=kernels), mach_a5,
        ).ops
        assert any(dict(op.metadata).get("recover_ssm_stage") == "commit" for op in g_rec)
        if impl == "analytical":
            assert _recurrent_write_bytes(g_rec) < 0.40 * _recurrent_write_bytes(g_base)
        else:
            rec_ops = [op for op in g_rec if op.recipe.endswith("_recover_ssm")]
            assert len(rec_ops) > 0


@pytest.mark.parametrize("kind", ("kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet"))
def test_all_seven_linear_attention_families_end_to_end_recover_ssm(kind: str):
    plan = LayerPlan(
        index=0, attention=kind, ffn="dense", intermediate_size=1024,
        linear_heads=8, linear_key_heads=8, linear_value_heads=8,
        linear_head_dim=128, linear_key_head_dim=128,
        conv_kernel=4, conv_state_dtype="bf16", recurrent_state_dtype="fp32",
    )
    draft = DraftSpec(
        kind="mtp", layers=1, hidden_size=512, intermediate_size=1024,
        num_heads=8, num_kv_heads=8, head_dim=64, q_width=512, kv_width=512,
        vocab_size=4096, attention_kind="gqa", kv_cache_elements=1024,
        target_shared_embedding=True, target_shared_lm_head=True,
    )
    gqa_ref = LayerPlan(
        index=1, attention="gqa", ffn="dense",
        num_heads=8, num_kv_heads=8, head_dim=64, intermediate_size=1024,
        kv_cache_elements=1024, indexer=True, index_topk=256, index_heads=4, index_dim=64,
        owns_index_k=True, owns_topk=True,
    )
    model = ModelSpec(
        name=f"synthetic-{kind}", layers=2, hidden_size=512,
        num_heads=8, head_dim=64, num_kv_heads=8,
        intermediate_size=1024, vocab_size=4096,
        attention="gqa", ffn="dense",
        layer_plans=(plan, gqa_ref), auxiliary=AuxiliarySpec(mtp=draft),
    )
    mach = _hardware(count=1)
    fe = Frontend()
    view = RequestView("req0", 1, 512, True, "decode", draft_proposal_tokens=4, draft_advanced_tokens=3)
    ops_off = fe(IterationPlan((view,)), model, _software(tp=1, recover_ssm=False), mach).ops
    ops_on = fe(IterationPlan((view,)), model, _software(tp=1, recover_ssm=True), mach).ops

    state_bytes = 8 * 128 * 128 * dtype_bytes("fp32")
    rec_bytes = 8 * record_head_bytes(128, 128, "bf16")
    expected_saving = (5 * state_bytes) - (5 * rec_bytes + state_bytes)
    assert _recurrent_write_bytes(ops_off) - _recurrent_write_bytes(ops_on) == pytest.approx(expected_saving)

    workload = WorkloadSpec(requests=(RequestSpec("req0", 0.0, 16, 6),))
    base_case = Case(
        workload=workload, model=model,
        runtime=_software(
            tp=1, recover_ssm=False,
            metadata=(("draft_mode", "mtp"), ("draft_proposal_tokens", 3), ("draft_accept_tokens", 2)),
        ),
        machine=mach,
    )
    rec_case = replace(base_case, runtime=replace(base_case.runtime, recover_ssm=True))
    res_off = predict(base_case)
    res_on = predict(rec_case)
    assert res_off["requests"][0]["generated_tokens"] == 6
    assert res_on["requests"][0]["generated_tokens"] == 6
    assert res_on["execution"]["live_tasks"] == 0
