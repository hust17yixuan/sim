"""mrv2 PCP + PCP/DCP coexistence and vLLM configuration aliases.

[Role]
  Establish three behaviors end to end: (1) ``pcp > 1`` and ``cp > 1`` can be
  declared together, with PCP owning the prefill query split/gather and DCP
  owning the decode KV shard/LSE merge; (2) PCP works on sparse latent (DSA)
  *and* window-compressed (SWA/CSA/CSA2/HCA) mrv2 layers, including the
  gathered full-sequence index-K prefix; (3) upstream vLLM / vLLM-Ascend
  configuration spellings map onto the canonical simulation mechanics and a
  conflicting duplicate declaration is refused.

[Invariants]
  * A prefill geometry never pays a decode-only collective and vice versa;
    the dual plan's prefill cost equals the PCP-only prefill cost.
  * The indexer policy is declared on the selector operator, never inferred
    from topology by a consumer.
  * ``dsa_cp`` stays a query-token SP plan: it excludes DCP and PCP.
  * An alias never silently wins over an explicit canonical field.

[Timeouts & Locks]
  Pure CPU construction and bounded arithmetic; no waits, I/O or locks.

[Side-Effect Policy]
  Read-only repository fixtures; no filesystem, network or device mutation.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.context_parallel import (
    CP_PLAN_PCP_DCP, cp_layer_scope, verify_pcp_model,
)
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise.compiler import ImplementationPolicy, PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.subgraphs import (
    COMPRESSED_ATTENTION_KINDS, LATENT_ATTENTION_KINDS,
    WINDOW_SPARSE_ATTENTION_KINDS,
)

SUPPLY = PipeSupply((("cube", 1_000_000.0), ("vector", 1_000_000.0),
                     ("hbm", 1_000_000.0), ("mte1", 2_000_000.0),
                     ("mte2", 2_000_000.0), ("mte3", 2_000_000.0),
                     ("fixpipe", 2_000_000.0), ("link", 1_000_000.0)))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=2, moe_row_tile=4,
                              split_target=1000, max_splits=4)
PREFILL = WorkloadGeometry((RequestGeometry(16, 0, phase="prefill"),),
                           phase="prefill")
DECODE = WorkloadGeometry((RequestGeometry(1, 16, phase="decode"),),
                          phase="decode")


def _features(model, **values):
    base = dict(token_tile=16, kv_tile=16, cores=2, moe_row_tile=4,
                split_target=1000, max_splits=4)
    base.update(values)
    return FeatureConfig.from_model(model, **base)


def _pipeline(model, **values):
    features = _features(model, **values)
    return assemble_model(model, supply=SUPPLY, policy=POLICY,
                          features=features)


def _find_attention(pipe, layer=0, role=None):
    """The declared attention macro of one layer, whatever its family."""
    kinds = LATENT_ATTENTION_KINDS | WINDOW_SPARSE_ATTENTION_KINDS
    for instance in pipe.instantiated.layers[layer].subgraphs:
        if instance.kind not in kinds:
            continue
        if role is not None and str(dict(instance.metadata).get("role", "")) != role:
            continue
        return instance
    raise AssertionError(f"no attention macro at layer {layer}")


def _rows(prediction, name):
    return [row for row in prediction.rows if row.cost.name == name]


def _positive(prediction, name):
    return [row for row in _rows(prediction, name) if row.duration_ms > 0]


def _compressed_model(cache_kind, ratio, architecture, *, indexer):
    """One declared window/compressed layer with a real compressor source."""
    plan = dict(index=0, attention="mla", ffn="dense", num_heads=4,
                num_kv_heads=1, head_dim=16, q_lora_rank=16, kv_lora_rank=16,
                qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8,
                intermediate_size=64, window=128, cache_kind=cache_kind,
                compress_ratio=ratio, architecture=architecture,
                compressor_kind="weighted_pool" if ratio > 0 else "none",
                owns_long_kv=True,
                indexer=indexer, index_heads=2, index_dim=8, index_topk=8,
                owns_index_k=indexer, index_k_owner_layer_id=0 if indexer else -1,
                owns_topk=indexer, topk_owner_layer_id=0 if indexer else -1,
                candidate_filter=architecture == "csa2")
    raw = dict(name=f"mrv2-{architecture}", layers=1, hidden_size=64,
               num_heads=4, num_kv_heads=1, head_dim=16,
               intermediate_size=64, vocab_size=128, layer_plans=[plan],
               mla=dict(q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8,
                        qk_rope_head_dim=8, v_head_dim=8, index_topk=8,
                        index_head_dim=8, index_n_heads=2, compress_ratio=ratio))
    return ModelSpec.from_mapping(raw)


def _dsa_model():
    return load_model("glm-5.3")


# ---------------------------------------------------------------------------
# (a) Simultaneous PCP + DCP

def test_dual_plan_resolves_phase_strategies_without_an_axis_conflict():
    dual = ParallelSpec(pcp=2, cp=2, tp=2)
    assert dual.cp_plan == CP_PLAN_PCP_DCP
    assert (dual.prefill_cp_plan, dual.decode_cp_plan) == ("pcp", "dcp")
    # PCP is a rank axis, DCP reuses the TP group.
    assert dual.world_size == dual.dp * dual.pp * dual.tp * dual.pcp
    explicit = ParallelSpec(pcp=2, cp=2, tp=2, cp_plan="auto")
    assert explicit.cp_plan == CP_PLAN_PCP_DCP
    orthogonal = ParallelSpec.from_mapping({
        "pcp": 2, "cp": 2, "tp": 2,
        "prefill_cp_plan": "pcp", "decode_cp_plan": "dcp"})
    assert orthogonal.cp_plan == CP_PLAN_PCP_DCP
    assert (orthogonal.prefill_cp_plan, orthogonal.decode_cp_plan) == ("pcp", "dcp")
    with pytest.raises(ModelingError, match="cp_plan_conflicts_with_phase_plans"):
        ParallelSpec.from_mapping({"pcp": 2, "cp": 2, "tp": 2, "cp_plan": "pcp",
                                   "prefill_cp_plan": "pcp",
                                   "decode_cp_plan": "dcp"})
    # One degree alone keeps its historical plan.
    assert ParallelSpec(pcp=2, cp=1, tp=2).cp_plan == "pcp"
    assert ParallelSpec(pcp=1, cp=2, tp=2).cp_plan == "dcp"


def test_prefill_runs_pcp_while_decode_runs_dcp_lse_merge():
    model = _dsa_model()
    dual = _pipeline(model, tp=2, cp=2, pcp=2, cp_plan=CP_PLAN_PCP_DCP)
    pcp_only = _pipeline(model, tp=2, cp=2, pcp=2, cp_plan="pcp")

    prefill = dual.forward(PREFILL, fusion=False)
    assert _positive(prefill, "pcp_kv_gather")
    assert _positive(prefill, "pcp_index_k_gather")
    assert not _positive(prefill, "dcp_allgather")
    assert not _positive(prefill, "dcp_lse_merge")

    decode = dual.forward(DECODE, fusion=False)
    assert _positive(decode, "dcp_allgather")
    assert _positive(decode, "dcp_lse_merge")
    assert not _positive(decode, "pcp_kv_gather")
    assert not _positive(decode, "pcp_index_k_gather")

    # PCP owns prefill in both plans, so the dual declaration must not change
    # the prefill body or its gather volume.
    pcp_prefill = pcp_only.forward(PREFILL, fusion=False)
    assert prefill.total_ms == pcp_prefill.total_ms
    gather = _positive(prefill, "pcp_kv_gather")[0]
    assert gather.duration_ms == _positive(pcp_prefill, "pcp_kv_gather")[0].duration_ms


def test_cp_layer_scope_declares_both_phases():
    model = _dsa_model()
    dual = ParallelSpec(pcp=2, cp=2, tp=2)
    scope = cp_layer_scope(model, dual.cp_plan)
    assert scope["prefill_cp_plan"] == "pcp"
    assert scope["decode_cp_plan"] == "dcp"
    assert scope["decode_merge"] == "npu_attention_update"
    assert scope["pcp_kv_gather"] == "full_index_k_and_active_kv_rows"
    assert scope["context_addressed"]


# ---------------------------------------------------------------------------
# (b) PCP on DSA and on mrv2 compressed/window families

def _pcp_model_matrix():
    return (
        ("dsa-glm53", _dsa_model()),
        ("csa-v4f", _compressed_model("window_compressed", 4, "csa", indexer=True)),
        ("csa2-v41", _compressed_model("window_compressed", 2, "csa2", indexer=True)),
        ("hca-v4", _compressed_model("window_compressed", 128, "hca", indexer=False)),
        ("swa-v41", _compressed_model("window_compressed", 0, "swa", indexer=False)),
    )


@pytest.mark.parametrize("label,model", _pcp_model_matrix())
@pytest.mark.parametrize("pcp", (2, 4))
def test_pcp_gathers_index_k_and_active_kv_rows(label, model, pcp):
    pipe = _pipeline(model, tp=1, cp=1, pcp=pcp, cp_plan="pcp")
    kinds = {instance.kind for instance in pipe.instantiated.subgraphs}
    sparse = kinds & (LATENT_ATTENTION_KINDS | COMPRESSED_ATTENTION_KINDS)
    compressed = kinds & WINDOW_SPARSE_ATTENTION_KINDS
    assert sparse or compressed, (label, kinds)

    prefill = pipe.forward(PREFILL, fusion=False)
    assert _positive(prefill, "pcp_kv_gather"), label
    has_indexer = "indexer_prolog" in kinds
    has_index_k_gather = False
    if has_indexer:
        index_k = pipe.instantiated.find("indexer_prolog", layer=0)
        has_index_k_gather = "pcp_index_k_gather" in {
            operator.name for operator in index_k.operators}
    core = next(instance for instance in pipe.instantiated.subgraphs
                if instance.kind in (LATENT_ATTENTION_KINDS
                                     | WINDOW_SPARSE_ATTENTION_KINDS))
    if has_index_k_gather:
        # The gathered plane is (pcp-1)/pcp of the full cache rows.
        gather_op = next(operator for operator in index_k.operators
                         if operator.name == "pcp_index_k_gather")
        theta = dict(gather_op.theta.extras)
        assert theta["gather_scope"] == "full_sequence_index_k"
        assert theta["indexer_policy"] == "pcp_local_query_full_index_k_gather"
        assert _positive(prefill, "pcp_index_k_gather"), label
        selector = pipe.instantiated.find("lightning_indexer", layer=0)
        extras = dict(selector.operators[0].theta.extras)
        assert extras["pcp"] == pcp
        assert extras["index_query_scope"] == "local_zigzag_chunk"
        assert extras["index_key_scope"] == "gathered_full_prefix"
    assert [row for row in _positive(prefill, "pcp_kv_gather")], label
    assert core.kind in (LATENT_ATTENTION_KINDS
                         | WINDOW_SPARSE_ATTENTION_KINDS)
    # Decode is replicated under a PCP-only plan.
    decode = pipe.forward(DECODE, fusion=False)
    assert not _positive(decode, "pcp_kv_gather")
    assert not _positive(decode, "pcp_index_k_gather")


def test_pcp_accepts_window_compressed_families_but_rejects_unknown():
    for name in ("csa", "csa2", "hca", "swa"):
        model = _compressed_model("window_compressed", 4 if name == "csa" else
                                  2 if name == "csa2" else 128 if name == "hca" else 0,
                                  name, indexer=name in {"csa", "csa2"})
        verify_pcp_model(model, ParallelSpec(pcp=2, cp=1, tp=1))
    unknown = ModelSpec.from_mapping(dict(
        name="unknown", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=64, vocab_size=128,
        layer_plans=[dict(index=0, attention="unmodeled", ffn="dense")]))
    with pytest.raises(ModelingError,
                       match="pcp_requires_declared_query_shard_builder"):
        verify_pcp_model(unknown, ParallelSpec(pcp=2, cp=1, tp=1))


# ---------------------------------------------------------------------------
# (c) dsa_cp regression parity

def test_dsa_cp_remains_query_token_sp_and_excludes_dcp_pcp():
    model = _dsa_model()
    pipe = _pipeline(model, tp=2, cp=1, pcp=1, cp_plan="dsa_cp")
    instance = _find_attention(pipe)
    names = {operator.name for operator in instance.operators}
    assert "dsa_cp_output_all_gather" in names
    assert not names & {"pcp_kv_gather", "pcp_index_k_gather",
                        "dcp_allgather", "dcp_lse_merge"}
    prefill = pipe.forward(PREFILL, fusion=False)
    assert not _positive(prefill, "pcp_kv_gather")
    assert not _positive(prefill, "dcp_lse_merge")
    with pytest.raises(ModelingError, match="dsa_cp_query_token_sp_excludes_dcp"):
        ParallelSpec(cp=2, cp_plan="dsa_cp", tp=2)
    with pytest.raises(ModelingError, match="dsa_cp_query_token_sp_excludes_pcp"):
        ParallelSpec(pcp=2, cp_plan="dsa_cp", tp=2)
    with pytest.raises(ModelingError, match="enable_dsa_cp_conflicts_with_plan"):
        ParallelSpec(cp=2, pcp=2, cp_plan=CP_PLAN_PCP_DCP, tp=2,
                     enable_dsa_cp=True)


# ---------------------------------------------------------------------------
# (d) 4-layer vLLM configuration aliases

def test_parallel_axis_aliases_map_one_to_one():
    spec = ParallelSpec.from_mapping({
        "tensor_parallel_size": 8, "data_parallel_size": 2,
        "pipeline_parallel_size": 1, "decode_context_parallel_size": 2,
        "prefill_context_parallel_size": 2,
    })
    assert (spec.tp, spec.dp, spec.pp, spec.cp, spec.pcp) == (8, 2, 1, 2, 2)
    assert spec.cp_plan == CP_PLAN_PCP_DCP
    full_ep = ParallelSpec.from_mapping({
        "tensor_parallel_size": 4, "enable_expert_parallel": True})
    assert full_ep.ep == 4
    declared = ParallelSpec.from_mapping({
        "tensor_parallel_size": 4, "expert_parallel_size": 2,
        "enable_expert_parallel": True})
    assert declared.ep == 2
    with pytest.raises(ModelingError, match="tp_conflict"):
        ParallelSpec.from_mapping({"tp": 2, "tensor_parallel_size": 4})
    with pytest.raises(ModelingError, match="cp_conflict"):
        ParallelSpec.from_mapping({"cp": 2, "decode_context_parallel_size": 4})
    with pytest.raises(ModelingError, match="expert_parallel_conflict"):
        ParallelSpec.from_mapping({"ep": 2, "enable_expert_parallel": False})
    with pytest.raises(ModelingError, match="expert_parallel_conflict"):
        ParallelSpec.from_mapping({"ep": 1, "enable_expert_parallel": True})


def test_compilation_dispatcher_and_block_size_aliases():
    software = Software.from_mapping({
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 8],
            "splitting_ops": ["attn", "mlp"],
        },
        "all2all_backend": "deepep_high_throughput",
        "block_size": 64,
    })
    assert software.graph_capture_mode == "full"
    assert software.graph_capture_keys == (1, 2, 8)
    assert software.splitting_ops == ("attn", "mlp")
    assert dict(software.metadata)["captured_execution_mode"] == "FULL_DECODE_ONLY"
    assert (software.moe_recipe, software.deepep_high_throughput) == ("alltoall", True)
    assert software.kv_block_size == 64
    piecewise = Software.from_mapping({"compilation_config": {
        "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [4, 2]}})
    assert piecewise.graph_capture_mode == "piecewise"
    assert piecewise.piecewise_capture_keys == (2, 4)
    assert piecewise.graph_capture_keys == ()
    mc2 = Software.from_mapping({"all2all_backend": "mc2"})
    assert mc2.moe_recipe == "mc2"
    with pytest.raises(ModelingError, match="runtime_cudagraph_mode_unsupported"):
        Software.from_mapping({"compilation_config": {"cudagraph_mode": "SOMETIMES"}})
    with pytest.raises(ModelingError, match="graph_capture_mode"):
        Software.from_mapping({"graph_capture_mode": "piecewise",
                               "compilation_config": {"cudagraph_mode": "NONE"}})
    with pytest.raises(ModelingError, match="all2all_backend_conflict"):
        Software.from_mapping({"moe_recipe": "alltoall", "all2all_backend": "mc2"})
    with pytest.raises(ModelingError, match="block_size_conflict"):
        Software.from_mapping({"kv_block_size": 128, "block_size": 64})
    with pytest.raises(ModelingError, match="all2all_backend_unsupported"):
        Software.from_mapping({"all2all_backend": "deepep_sometimes"})


def test_speculative_and_layer_plan_aliases():
    spec = SpeculativeSpec.from_mapping(
        {"num_speculative_tokens": 3, "acceptance": (1.0, 0.8, 0.5)})
    assert spec.draft_len == 3
    assert SpeculativeSpec.from_mapping(
        {"draft_len": 3, "num_speculative_tokens": 3,
         "acceptance": (1.0, 0.8, 0.5)}).draft_len == 3
    with pytest.raises(ModelingError, match="draft_len_conflict"):
        SpeculativeSpec.from_mapping({"draft_len": 2, "num_speculative_tokens": 3})
    assert LayerPlan(index=0, indexer_type="per_layer").indexer_type == "full"
    assert LayerPlan(index=0, indexer_type="c4").indexer_type == "c4"
    assert LayerPlan(index=0, indexer_type=None).indexer_type == "none"
    with pytest.raises(ModelingError, match="indexer_type_unknown"):
        LayerPlan(index=0, indexer_type="sometimes")
    # A normalized alias is a declared value: it reaches the slot/builder
    # vocabulary unchanged.
    plan = replace(LayerPlan(index=0, attention="mla", indexer=True,
                             num_heads=4, num_kv_heads=1, head_dim=16,
                             q_lora_rank=16, kv_lora_rank=16,
                             qk_nope_head_dim=8, qk_rope_head_dim=8,
                             v_head_dim=8, owns_index_k=True,
                             index_k_owner_layer_id=0, owns_topk=True,
                             topk_owner_layer_id=0, index_topk=8,
                             index_heads=2, index_dim=8),
                   indexer_type="per_layer")
    assert plan.indexer_type == "full"
