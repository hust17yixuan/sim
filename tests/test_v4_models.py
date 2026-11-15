"""Pinned v4 operator equations, classifications, and typed rejection tests.

[Role]
Pin all sixteen declared operator count vectors and their allocation records,
check public route integration, and prevent synthetic ledgers from becoming
physical certifications or catalog entries from being matched by array index.
[Invariants]
Expected JSON comes from the readonly pack; the test never computes its own
expected values through production formulas. Source coverage is not calibration.
[Timeouts & Locks]
Bounded local reads and deterministic arithmetic; no waits or locks.
[Side-Effect Policy]
Read fixtures and live provenance only; no writes.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("mindie_knowledge")

from mindie_knowledge.distiller_sources import scan_live, static_ingestion
from mindie_knowledge.cannbot_aclgraph_distiller import distill_units
pytest.importorskip("mindie_profiler")
from mindie_profiler.aclgraph_multistream import ParallelRegion, StreamInterval, diagnose_region
from mindie_sim.opflow import build_flow, catalog_info, evaluate_operator, get_profile
from mindie_sim.opflow.exact_math import MathDomainError, floor_sum, integer, memory_seconds, semantic_vector_costs
from mindie_sim.opflow.v4_math import classification, evaluate_ast, evaluate_math, model

FIXTURE = json.loads((Path(__file__).parent / "fixtures/v4_exact_vectors.json").read_text())


@pytest.mark.parametrize("vector",FIXTURE["exact_test_tables"]["semantic_cost_vectors"])
def test_semantic_scalar_work_vectors(vector):
    assert semantic_vector_costs(vector["D"]) == vector


def test_ebudget_identity_fails_closed_at_all_limits():
    from mindie_sim.opflow.token_budget import bounded_envelope, envelope_size
    for limit in (32,48,250):
        result = bounded_envelope({"ok":1,"op":"identity"*1000},limit=limit,required=("ok","op"))
        assert result["ok"] == 0 and result["err"] == "EBUDGET"
        assert envelope_size(result)["utf8_bytes"] <= limit


@pytest.mark.parametrize("name,vector",[(name,v) for name,vs in FIXTURE["operators"].items() for v in vs],
                         ids=[name for name,vs in FIXTURE["operators"].items() for v in vs])
def test_all_operator_equations_and_allocations(name,vector):
    result = evaluate_math(name,vector["bindings"])
    assert result["counts"] == vector["expected"]
    assert result["buffers"] == vector["buffer_expected"]
    assert result["confidence"] == "guarded_mathematical_counts"
    assert result["full_physical_certification"] is False
    if result["coverage"] == "conditional_ledger_required":
        with pytest.raises(MathDomainError):
            evaluate_math(name,vector["bindings"],physical=True)
        with pytest.raises(MathDomainError):
            evaluate_math(name,vector["bindings"],physical=True,ledger={"evidence":"synthetic_fixture"})


def test_v4_counts_not_r1_index_mapping():
    assert model()["source_sha256"] == FIXTURE["source_sha256"]
    catalog = model()["catalog_53"]
    assert len(catalog) == len({row["id"] for row in catalog}) == 53
    assert Counter(x["classification"] for x in catalog) == {"full_sample_physics":5,"structural_schema_only":48}
    assert Counter(x["coverage"] for x in model()["operators"]) == {
        "closed_selected_route":5,"exact_reference_plan":5,"conditional_ledger_required":6}
    live = scan_live()
    assert {x["id"]:x["source"] for x in catalog} == dict(live.operator_sources)
    assert static_ingestion().operator_sources == live.operator_sources
    assert static_ingestion().operator_tests == live.operator_tests
    info = {x["name"]:x for x in catalog_info()}
    for row in reversed(catalog):
        assert info[row["id"]]["catalog_classification"] == row["classification"]
        assert info[row["id"]]["derivation_coverage"] == row["derivation_coverage"]
        assert info[row["id"]]["estimate_kind"] != "source_faithful"
    assert classification("mhc_hcpre_cv_splitk")["catalog_classification"] == "outside_dsl_catalog_53"
    units = distill_units(live)
    cards = [u for u in units if "operator" in u.conditions]
    assert len(cards) == 53
    assert Counter(u.conditions["catalog_classification"] for u in cards) == {
        "full_sample_physics":5,"structural_schema_only":48}
    for u in units:
        assert u.source_commit == live.source_commits[u.source_repo]
        assert u.provenance["content_sha256"] == live.content_hashes[f"{u.source_repo}/{u.source_path}"]
    assert len(distill_units(live,include_aclgraph=False)) == 103


@pytest.mark.parametrize("bad",[True,1.0,-1,"1",float("nan"),float("inf"),None])
def test_typed_count_domain_rejections(bad):
    with pytest.raises(MathDomainError):
        integer("count",bad)
    bindings = dict(FIXTURE["operators"]["block_attn_res_update_rms_norm"][0]["bindings"],T=bad)
    with pytest.raises(MathDomainError):
        evaluate_math("block_attn_res_update_rms_norm",bindings)


def test_missing_symbols_zero_denominators_and_abi_guards():
    with pytest.raises(MathDomainError):
        evaluate_ast({"op":"symbol","name":"missing"},{})
    with pytest.raises(MathDomainError):
        floor_sum(10,0,1,1)
    with pytest.raises(MathDomainError):
        memory_seconds(1,1,1,1,2)
    with pytest.raises(ValueError):
        build_flow("block_attn_res_update_rms_norm",hidden=7169)
    with pytest.raises(ValueError):
        build_flow("fused_recurrent_kda_replayssm",dim=129)
    with pytest.raises(ValueError):
        build_flow("quant_batch_matmul_mxa8w4",k=33)
    with pytest.raises(ValueError):
        evaluate_operator("flash_kda",get_profile("Ascend950PR"))
    result = evaluate_operator("flash_kda",get_profile("Ascend950PR"),allow_structural_prior=True)
    assert result.estimate_kind == "conditional_ledger_required" and result.calibrated is False


def test_public_selected_routes_match_v4_work_and_service():
    cases = [
        ("block_attn_res_update_rms_norm",dict(tokens=3,active_owners=2)),
        ("fused_recurrent_kda_replayssm",dict(batch=2,heads=12,seq=5)),
        ("commit_recurrent_kda",dict(batch=3,heads=2,accepted_tokens=[2,3,0])),
        ("matmul_bf16",dict(m=17,n=19,k=33,base_m=16,base_n=16,base_k=16)),
        ("matmul_streamk",dict(m=33,n=35,k=64,base_m=16,base_n=16,base_k=16,
                             partitions=2,tail_tiles=1,tail_valid_elems=3,tail_write_elems=16)),
        ("quant_batch_matmul_mxa8w4",dict(m=65,n=70,k=96)),
    ]
    for name,kw in cases:
        f = build_flow(name,**kw)
        c = evaluate_math(name,f.params["count_bindings"])["counts"]
        assert f.gm_read_bytes == c["actual_read_bytes"]
        assert f.gm_write_bytes == c["actual_write_bytes"]
        assert f.unique_gm_bytes == c["logical_read_bytes"]+c["logical_write_bytes"]
        if f.archetype.value == "PURE_VECTOR":
            assert sum(w.total_work for w in f.pipe_work if w.pipe=="VEC") == c["ops_vec"]
    commit = build_flow("commit_recurrent_kda",batch=3,heads=2,accepted_tokens=[2,3,0])
    assert commit.gm_write_bytes == 262144
    mx = build_flow("quant_batch_matmul_mxa8w4",m=65,n=70,k=96)
    assert mx.dtype == "mxfp8"
    assert mx.params["exact_counts"]["bit_ops"] == 294912
    assert (mx.gm_read_bytes,mx.gm_write_bytes) == (96256,32768)
    assert not any(b.pool.value=="SSBUF" for b in mx.buffers)


def test_profiler_requires_declared_read_write_and_unique_scopes():
    a = StreamInterval("a",0,0,10,io_bytes=256,read_bytes=256,write_bytes=0,unique_bytes=64)
    b = StreamInterval("b",1,0,10,read_bytes=0,write_bytes=0,unique_bytes=0)
    d = diagnose_region(ParallelRegion("cache",a,b,event_pairs=0))
    assert d.details["unique_bytes"] == 64
    assert d.details["read_service_bytes"] == 256
    assert d.details["write_service_bytes"] == 0
    legacy = diagnose_region(ParallelRegion("unknown",StreamInterval("a",0,0,10,io_bytes=256),b,event_pairs=0))
    assert legacy.details["unique_bytes"] is None
    assert legacy.details["memory_evidence"] == "io_total_upper_bound"
