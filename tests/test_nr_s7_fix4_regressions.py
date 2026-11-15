"""Permanent R4 source-equation, tensor-extent and ledger counterexamples.

[Role]
  Close R4 counterexamples and FIX8 WorkloadSpec construction/consumer gaps.
  Expected values come from physical page facts and source equations.

[Invariants]
  * Both GLM classes reconcile once on every PP stage, including signed release.
  * Branch record loads and accumulator work precede PMF expectation.
  * Packed collective extents and retained page shapes determine byte counts.
  * Invalid typed inputs reject before coercion; timeout exceptions propagate.

[Timeouts & Locks]
  Bounded CPU tests and AST/file reads; no NPU or network access.

[Side-Effect Policy]
  No persistent writes; pytest uses the caller's external basetemp.
"""
from __future__ import annotations

import ast
import dataclasses as dc
from collections import UserList
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from nr_s7_fix2_support import (
    _case, _config, _dcp, _ds41, _geometry, _glm, _keys, _kimi, _layer, _spec)
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.dspark_dcp_cost import DcpDraftBlockCost
from mindie_sim.dspark_prefix_cache import DSparkPrefixPage, dspark_prefix_page_bytes
from mindie_sim.dspark_prefix_cache import dspark_prefix_plan
from mindie_sim.hybrid_kv_state_options import HybridKvStateOptions, model_is_sparse_sfa
from mindie_sim.hybrid_kv_state_pool import BaselinePricing, baseline_pricing, stage_deltas
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state as E
from mindie_sim.kv_layout_physics import KvNzPlane
from mindie_sim.linear_state_physics import (
    accepted_token_pmf, compare_linear_state_memory, linear_state_memory)
from mindie_sim.linear_state_replay import verify_commit_cost
from mindie_sim.recoverssm_config import validate_recoverssm
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import WorkloadSpec

CASE = lru_cache(None)(_case)


def test_f01_standard_generic_k8_comparison_isolates_unavailable_backend():
    cfg = _config(use_replayssm=False, speculative_k=8, model_type="generic_linear")
    args = dict(sequence_tokens=2048)
    assert linear_state_memory(cfg, [_layer()], **args).total_bytes == 5_326_848
    comparison = compare_linear_state_memory(cfg, [_layer()], **args)
    assert comparison["standard"]["total_bytes"] == 5_326_848
    assert comparison["recoverssm"] is None and comparison["applicable"] is False
    assert comparison["reason"] == "recoverssm_inapplicable"
    assert comparison["codes"] == ["recoverssm_requires_glm5_next_model"]
    with pytest.raises(ModelingError, match="spec_tokens_exceed_kda_limit"):
        _config(speculative_k=8)


def test_f01_public_kimi_standard_k8_survives_counterfactual():
    report = E(CASE("kimi-k3"), _kimi(num_spec=8, dspark_active=False))
    assert report["errors"] == [] and report["status"] != "rejected"
    assert report["linear_state"]["step_cost"]["mode"] == "standard"
    assert report["linear_state"]["comparison"]["applicable"] is False


@pytest.mark.parametrize("text,hf,wrapper,expected", [
    (None, NS(index_topk=0), {}, False),
    (NS(), NS(index_topk=8), {}, False),
    (NS(index_topk=8), NS(), {"index_kpool": 1}, True),
    (NS(index_topk=0), NS(), {}, True),
    (NS(index_topk=8, index_kpool=1), NS(), {}, False),
    (NS(index_topk=8), NS(index_kpool=None), {}, False),
    (NS(index_topk=8), NS(compress_ratios=[]), {}, False),
    (NS(index_topk=8), NS(), {"compress_ratios": []}, True),
])
def test_f03_source_object_locations_and_attribute_presence(text, hf, wrapper, expected):
    assert model_is_sparse_sfa(NS(hf_text_config=text, hf_config=hf, **wrapper)) is expected


@pytest.mark.parametrize("value", [1, 0, None])
def test_f03_normalized_catalog_preserves_explicit_kpool_presence(value):
    model = CASE("deepseek-v3.2").model
    assert model_is_sparse_sfa(model) is True
    facts = {**dict(model.catalog_facts), "index_kpool": value}
    assert model_is_sparse_sfa(dc.replace(model, catalog_facts=tuple(facts.items()))) is False


def _kv_stages(report, ledger="baseline"):
    return next(c["per_stage_bytes"] for c in report["memory"][ledger]["components"]
                if c["name"] == "kv_and_state_pool_bytes")


def test_f04_small_class_replaces_only_the_bytes_actually_priced():
    case = CASE()
    report = E(case, _glm())
    pool = report["layout"]["mixed_pool"]
    components = {}
    for plan in case.model.layer_plans:
        for name, value in layer_cache_components(
                case.model, plan, case.runtime.parallel, 2048, capacity_padded=True).items():
            components[name] = components.get(name, 0) + 8 * value
    assert components["index_k"] == 11_534_336
    assert components["index_pool_tail"] == 360_448
    assert pool["small_class_baseline_bytes"] == 11_894_784
    assert pool["small_class_bytes"] == 160 * 11 * 131_072 == 230_686_720
    assert pool["main_class_bytes"] == 160 * 12 * 608_256 == 1_167_851_520
    assert pool["small_class_bytes"] - pool["small_class_baseline_bytes"] == 218_791_936
    assert _kv_stages(report)[0] == 344_064_000
    assert report["memory"]["net_delta_per_stage"] == [1_054_474_240]
    assert _kv_stages(report, "adjusted") == [1_398_538_240]
    assert pool["physical_pool_bytes"] == 1_398_538_240
    assert pool["missing_total_bytes"] == 0


@pytest.mark.parametrize("partition", [(1, 44), (22, 23)])
def test_f04_pp_stage_projection_reconciles_both_classes(partition):
    original = CASE(pp=2)
    case = dc.replace(original, runtime=dc.replace(original.runtime, pp_layer_partition=partition))
    report = E(case, _glm(use_replayssm=False))
    pool = report["layout"]["mixed_pool"]
    assert not report["errors"] and pool["blocks"] == 256
    offset = 0
    for stage, size in enumerate(partition):
        indexed = sum(p.indexer for p in case.model.layer_plans[offset:offset + size])
        record = pool["stages"][stage]
        main = 256 * record["main_slot_count"] * 567_296
        small = 256 * indexed * 131_072
        assert record["small_slot_count"] == indexed
        assert record["main_class_bytes"] == main
        assert record["small_class_bytes"] == small
        assert _kv_stages(report, "adjusted")[stage] == main + small
        offset += size
    assert pool["missing_total_bytes"] == 0
    if partition == (1, 44):
        assert _kv_stages(report, "adjusted") == [145_227_776, 1_966_604_288]
    rejected = E(case, _glm())
    assert "recoverssm_requires_pp1" in rejected["errors"]
    assert rejected["linear_state"]["stage_state_bytes"] == [0.0, 0.0]


def test_f04_declared_combined_pool_is_replaced_once_even_for_signed_release():
    groups = ({"layers": tuple(range(45)), "tokens_per_page": 128,
               "bytes_per_group_page": 2_000_000_000, "fixed_pages": 1},)
    report = E(_case(memory={"cache_page_groups": groups}), _glm())
    pool = report["layout"]["mixed_pool"]
    assert pool["baseline_replaced_per_stage"] == _kv_stages(report)
    assert pool["delta_per_stage"][0] < 0
    assert _kv_stages(report, "adjusted") == [1_398_538_240]
    assert pool["missing_total_bytes"] == 0


@pytest.mark.parametrize("ids,expected", [((40,), 81_920.0), ((41,), 0.0), ((0, 41), 0.0)])
def test_f06_rejected_aux_ids_zero_all_published_scratch(ids, expected):
    report = E(CASE("deepseek-v4.1-flash"), _ds41(dspark_aux_layer_ids=ids))
    assert report["memory"]["aux_scratch_bytes"] == expected
    assert report["memory"]["aux_scratch_per_stage"] == [expected]
    assert report["dspark"]["prefix"]["aux_scratch_bytes"] == expected
    if 41 in ids:
        assert "dspark_aux_layer_out_of_range:41" in report["errors"]
        assert report["status"] == "rejected"


@pytest.mark.parametrize("pcp,dcp,heads,tp_bytes,pcp_bytes", [
    (1, 2, 16, 0, 0), (2, 2, 8, 0, 206_400), (2, 16, 64, 1_444_800, 1_651_200)])
def test_f08_source_packed_tensor_extent(pcp, dcp, heads, tp_bytes, pcp_bytes):
    cost = DcpDraftBlockCost(**_dcp(pcp=pcp, dcp=dcp))
    packed = 25 * heads * (512 + 4) * 2
    assert cost.gathered_heads == heads and cost.packed_merge_bytes == packed
    assert cost.combine_tp_bytes == tp_bytes and cost.combine_pcp_bytes == pcp_bytes
    scatter = (8 if dcp > pcp else 1) if pcp > 1 else dcp
    assert cost.packed_merge_shape == (scatter, heads // scatter, 25, 516)
    assert cost.combine_bytes == packed * ((scatter - 1) / scatter + pcp - 1)
    assert cost.extra_pairs == 0


@pytest.mark.parametrize("dtype,dim,element", [("bf16", 4, 2), ("fp16", 4, 2), ("fp32", 1, 4)])
def test_f08_packing_dtype_and_pcp_full_tensor_multiplier(dtype, dim, element):
    cost = DcpDraftBlockCost(**_dcp(pcp=4, dcp=32, out_dtype=dtype))
    assert cost.packed_merge_bytes == 25 * 64 * (512 + dim) * element
    assert cost.combine_pcp_bytes == 3 * cost.packed_merge_bytes


def test_f09_correlated_scalar_forgery_cannot_invent_a_physical_page():
    geometry = _geometry()
    reg = geometry["registration"]
    with pytest.raises(ModelingError, match="shape0_not_divisible"):
        forged = dc.replace(reg, num_blocks=2,
                            pages=(dc.replace(reg.pages[0], region_size=256),))
        geometry.update(forged.claims())
        geometry["registration"] = forged
        plan = dspark_prefix_plan(keys=_keys(), page_geometry=geometry, aux_layer_ids=(1,),
            target_hidden_size=8, target_dtype="bf16", tokens=256,
            prompt_tokens=256, block_hash_count=2)
        plan.validate_block_table({0: [0, 1]}, [0, 1])


def _scaled_registration():
    return dspark_prefix_page_bytes(
        pages=(DSparkPrefixPage("draft", 0, 4, 64, 64, 2, 64, 2),),
        num_blocks=2, block_size=128, hash_block_size=128,
        group_ids=(0,), block_sizes=(128,), draft_group_ids=(0,))["registration"]


def test_f09_scaled_shape_logical_coverage_stride_and_capacity_are_derived():
    reg = _scaled_registration()
    record = reg.pages[0]
    assert (record.scale, record.size, record.stride, record.region_size) == (2, 256, 256, 512)
    assert record.region_size == (record.page.shape0 - 1) * record.page.kernel_stride_bytes + record.page.kernel_bytes
    with pytest.raises(ModelingError, match="cover_block"):
        dc.replace(reg, num_blocks=4, page_bytes=128, stride_bytes=128,
                   pages=(dc.replace(record, scale=1, size=128, stride=128),))
    for changes, reason in (({"stride0_elements": 128}, "not_adjacent"),
                            ({"stride0_elements": 32}, "stride_smaller"),
                            ({"row_bytes": 4}, "capacity_too_small")):
        page = dc.replace(record.page, **changes)
        changed = dc.replace(record, page=page, stride=page.kernel_stride_bytes * 2)
        with pytest.raises(ModelingError, match=reason):
            dc.replace(reg, pages=(changed,), loaded_pages=(page,), stride_bytes=changed.stride)


def test_f09_loaded_cache_inventory_cannot_drop_or_replace_pages():
    reg = _scaled_registration()
    with pytest.raises(ModelingError, match="loaded_cache_inventory"):
        dc.replace(reg, pages=(dc.replace(reg.pages[0], page=dc.replace(
            reg.pages[0].page, name="different_loaded_cache")),))
    with pytest.raises(ModelingError, match="loaded_cache_inventory"):
        dc.replace(reg, pages=reg.pages * 2, page_bytes=512, stride_bytes=512, objects_per_key=2)
    with pytest.raises(ModelingError, match="physical_region_extent"):
        dc.replace(reg, pages=(dc.replace(reg.pages[0], region_size=1024),))


def _commit(start, accepted=None, alpha=0.0, **kwargs):
    return verify_commit_cost(_config(mamba_cache_mode="align", enable_prefix_caching=True),
        [_layer()], batch=2, sequence_tokens=start, accepted_tokens=accepted,
        accept_rate=alpha, hbm_bytes_per_s=1e12, **kwargs)


def test_f11_boundary_records_and_matrix_work_match_source_equations():
    inside, cross = _commit(120, 5), _commit(126, 5)
    # Each value program loads 128 BF16 keys, 128 BF16 gates, and BV FP32 corrections.
    assert inside["record_read_bytes_per_token"] == 8 * 2 * (128 * 4 + 64 * 4) == 12_288
    assert cross["boundary_record_read_bytes_per_token"] == 8 * 4 * (128 * 4 + 32 * 4) == 20_480
    assert cross["stage2_record_read_bytes"] == 2 * 5 * 20_480
    assert cross["hbm_bytes"] == 4_685_824
    assert cross["total_ms"] == 0.004685824
    assert cross["hbm_bytes"] - cross["total_ms"] * 1e12 / 1000 == 0
    matrix = 2 * 131_072 * (5 + 1) * 2
    assert inside["stage2_matrix_flops"] == matrix
    assert cross["boundary_matrix_flops"] == matrix == 3_145_728
    assert cross["stage2_matrix_flops"] == 2 * matrix
    assert inside["stage2_vector_flops"] == 2 * 8 * 2 * 5 * (8 * 128 - 1)
    assert cross["stage2_vector_flops"] == 2 * 8 * 4 * 5 * (10 * 128 - 1)
    assert cross["stage2_vector_special_ops"] == {
        "vector_exp_ops": 64 * 641, "vector_sigmoid_ops": 64 * 640,
        "vector_rsqrt_ops": 64 * 5}


def test_f11_branch_physics_precedes_exact_pmf_expectation():
    for start in (123, 124, 125, 126, 127, 128):
        for alpha in (0.0, 0.2, 0.8, 1.0):
            cost = _commit(start, alpha=alpha)
            expected_bytes = expected_matrix = expected_boundary = 0.0
            for m, probability in accepted_token_pmf(4, alpha):
                crossing = start // 128 != (start + m) // 128
                record = 20_480 if crossing else 12_288
                expected_bytes += probability * 2 * m * record
                expected_matrix += probability * 2 * 2 * 131_072 * (m + 1) * (1 + crossing)
                expected_boundary += probability * crossing
            assert cost["stage2_record_read_bytes"] == pytest.approx(expected_bytes)
            assert cost["stage2_matrix_flops"] == pytest.approx(expected_matrix)
            assert cost["boundary_probability"] == pytest.approx(expected_boundary)
            assert cost["total_ms"] == pytest.approx(cost["hbm_bytes"] / 1e9)


@pytest.mark.parametrize("sku", ["Ascend950DT_9581", "Ascend950PR_9579"])
def test_f11_public_a5_and_pr_launch_geometry_survives(sku):
    report = E(CASE(sku=sku, topology="a5_pod"), _glm(batch=1))
    cost = report["linear_state"]["step_cost"]
    assert report["linear_state"]["commit_hardware"]["ub_bytes"] == 253_952
    assert (cost["commit_block_v"], cost["boundary_block_v"]) == (128, 64)
    assert cost["launch_geometry"]["commit_programs"] == 272
    assert cost["stage2_program_count"] == 307


def test_f11_strided_conv_launch_survives():
    report = E(CASE(), _glm(batch=1, conv_dim_stride=2))
    geometry = report["linear_state"]["step_cost"]["launch_geometry"]
    assert geometry["conv_tile"] == 256 and geometry["conv_programs"] == 408


@pytest.mark.parametrize("bad", [[], {}, None, 1, False])
def test_f14_modes_reject_before_hash_lookup(bad):
    for construct in (lambda: HybridKvStateOptions(mamba_cache_mode=bad),
                      lambda: _config(mamba_cache_mode=bad),
                      lambda: _layer(kind=bad), lambda: _spec("s", kind=bad),
                      lambda: _spec("s", scope=bad),
                      lambda: KvNzPlane(bad, 128, "bf16", 1, 1, 128)):
        with pytest.raises(ModelingError):
            construct()


@pytest.mark.parametrize("bad", [None, [], (), "1", {}, (0, 0)])
def test_f14_both_baseline_stage_vectors_are_typed_and_exact_length(bad):
    for args in ((bad, (0,)), ((0,), bad)):
        with pytest.raises(ModelingError):
            BaselinePricing(1, *args, 0, 0, 0)


@pytest.mark.parametrize("bad", ["false", 0, 1, [], None])
def test_f14_boolean_controls_never_use_truthiness(bad):
    with pytest.raises(ModelingError):
        stage_deltas(modeled=(0,), replaced=(1,), conservative=bad)
    with pytest.raises(ModelingError):
        baseline_pricing(CASE(), tokens=128, active_sequences=1, include_draft=bad)
    with pytest.raises(ModelingError):
        validate_recoverssm(_config(), strict=bad)


@pytest.mark.parametrize("bad", [None, (), [], "1", {}, 0])
def test_f14_delta_stage_containers_reject_before_length_or_iteration(bad):
    for kwargs in ({"modeled": bad, "replaced": (0,)},
                   {"modeled": (0,), "replaced": bad}):
        with pytest.raises(ModelingError):
            stage_deltas(**kwargs)
    assert stage_deltas(modeled=(0,), replaced=(1,), conservative=False) == (-1.0,)


@pytest.mark.parametrize("bad", [True, Fraction(1, 2), float("nan"), float("inf"), "1024", "١٠٢٤"])
def test_f14_public_staging_rejects_raw_invalid_values(bad):
    case = CASE()
    broken = dc.replace(case, runtime=dc.replace(case.runtime, memory={"kv_transfer_staging_bytes": bad}))
    with pytest.raises(ModelingError):
        E(broken, _glm(use_replayssm=False))


def test_g01_all_scoped_broad_handlers_reraise_timeout_first():
    from nr_s7_fix4_verify import audit_scope
    audit = audit_scope()
    assert audit["timeout_violations"] == []
    assert audit["hygiene_violations"] == []


@pytest.mark.parametrize("function", ["_gates", "_dspark"])
def test_g01_generator_timeout_injection_propagates(function):
    # Execute the original try node with a timeout-raising constructor/grouping
    # double. This reaches each exact handler without expensive unrelated setup.
    path = Path(__file__).with_name("nr_s7_fix3_verify.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    guarded = next(n for n in ast.walk(node) if isinstance(n, ast.Try))
    def timeout(*args, **kwargs):
        raise TimeoutError("injected R4 timeout")
    env = {"RecoverSsmConfig": timeout, "kimi_k3_dspark_grouping": timeout,
           "noncausal": (), "OUT": {"gates": {}, "grouping": {}}}
    with pytest.raises(TimeoutError, match="injected R4 timeout"):
        exec(compile(ast.Module(body=[guarded], type_ignores=[]), str(path), "exec"), env)


_FIX8_BAD_REQUESTS = ((), [], {}, set(), "", b"", None, (123,), [None])
_FIX8_BAD_SYNTHETIC = (1, "yes", 0, None, (), [])


@pytest.mark.parametrize("bad", _FIX8_BAD_REQUESTS)
def test_fix8_workload_construction_and_replace_reject_invalid_requests(bad):
    workload = CASE().workload
    with pytest.raises(ModelingError, match=r"^workload\.requests_invalid$"):
        WorkloadSpec(requests=bad)
    with pytest.raises(ModelingError, match=r"^workload\.requests_invalid$"):
        dc.replace(workload, requests=bad)


@pytest.mark.parametrize("container", (tuple, list, UserList))
@pytest.mark.parametrize("synthetic", (False, True))
def test_fix8_workload_valid_sequences_normalize_without_aliasing(container, synthetic):
    base = CASE().workload
    req = base.requests[0]
    for expected in ((req,), (req, dc.replace(req, request_id=req.request_id + "_other"))):
        supplied = container(expected)
        workloads = (WorkloadSpec(requests=supplied, synthetic=synthetic),
                     dc.replace(base, requests=supplied, synthetic=synthetic))
        if isinstance(supplied, (list, UserList)):
            supplied.clear()
        for workload in workloads:
            assert type(workload.requests) is tuple and workload.requests == expected
            assert workload.synthetic is synthetic
            assert WorkloadSpec.from_mapping({"requests": [r.to_dict() for r in expected],
                                               "synthetic": synthetic}).requests == expected


@pytest.mark.parametrize("bad", _FIX8_BAD_SYNTHETIC)
def test_fix8_workload_synthetic_requires_bool(bad):
    workload = CASE().workload
    with pytest.raises(ModelingError, match=r"^workload\.synthetic_invalid$"):
        WorkloadSpec(requests=workload.requests, synthetic=bad)
    with pytest.raises(ModelingError, match=r"^workload\.synthetic_invalid$"):
        dc.replace(workload, synthetic=bad)


def test_fix8_workload_duplicate_ids_reject_at_all_construction_paths():
    workload = CASE().workload
    req = workload.requests[0]
    requests = (req, dc.replace(req, request_id=req.request_id + "_other"),
                dc.replace(req, arrival_ms=req.arrival_ms + 1))
    for call in (lambda: WorkloadSpec(requests=requests),
                 lambda: dc.replace(workload, requests=requests),
                 lambda: WorkloadSpec.from_mapping({"requests": [r.to_dict() for r in requests]})):
        with pytest.raises(ModelingError, match=r"^workload\.request_id_duplicate$"):
            call()


@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
@pytest.mark.parametrize("bad", _FIX8_BAD_REQUESTS)
def test_fix8_workload_consumers_reject_bypass_injected_requests(capture, bad):
    from mindie_sim.memory_breakdown import build_memory_ledger, memory_breakdown
    case = CASE()
    broken_ws = object.__new__(WorkloadSpec)
    for field in dc.fields(case.workload):
        value = bad if field.name == "requests" else getattr(case.workload, field.name)
        object.__setattr__(broken_ws, field.name, value)
    broken = dc.replace(case, workload=broken_ws,
                        runtime=dc.replace(case.runtime, graph_capture_mode=capture))
    for call in (lambda: E(broken), lambda: E(broken, _glm(sequence_tokens=128)),
                 lambda: build_memory_ledger(broken), lambda: memory_breakdown(broken)):
        with pytest.raises(ModelingError, match="requests_invalid"):
            call()


def test_fix8_workload_rejects_text_bytes_iterators_and_later_invalid_items():
    req = CASE().workload.requests[0]
    for bad in (bytearray(), bytearray(b"r"), "r", b"r", (req, None), iter((req,))):
        test_fix8_workload_construction_and_replace_reject_invalid_requests(bad)


@pytest.mark.parametrize("source", ("case", "machine", "hardware"))
def test_fix8b_parse_model_sentinel_remains_strict_at_workload_boundaries(source):
    from mindie_sim.memory_breakdown import build_memory_ledger, memory_breakdown
    from mindie_sim.prediction.structure import parse_model
    base = CASE()
    declaration = base if source == "case" else {
        source: base.machine.to_dict(), "model": {"catalog_key": "glm-5.3-flash"},
        "runtime": base.runtime.to_dict()}
    case = parse_model(declaration)
    assert case.workload.requests == () and case.workload.metadata == ()
    assert case.workload.synthetic is False
    with pytest.raises(dc.FrozenInstanceError):
        case.workload.synthetic = True
    with pytest.raises(ModelingError, match=r"^workload\.requests_invalid$"):
        dc.replace(case.workload, synthetic=True)
    for call in (lambda: E(case), lambda: E(case, _glm(sequence_tokens=None)),
                 lambda: build_memory_ledger(case), lambda: memory_breakdown(case)):
        with pytest.raises(ModelingError, match="requests_invalid"):
            call()
    assert base.workload.requests


def fix8_workload_witnesses():
    """Re-execute permanent constructor and bypass-consumer regression cases."""
    for bad in _FIX8_BAD_REQUESTS:
        test_fix8_workload_construction_and_replace_reject_invalid_requests(bad)
        for capture in ("none", "full", "piecewise"):
            test_fix8_workload_consumers_reject_bypass_injected_requests(capture, bad)
    for container in (tuple, list, UserList):
        for synthetic in (False, True):
            test_fix8_workload_valid_sequences_normalize_without_aliasing(container, synthetic)
    for bad in _FIX8_BAD_SYNTHETIC:
        test_fix8_workload_synthetic_requires_bool(bad)
    test_fix8_workload_duplicate_ids_reject_at_all_construction_paths()
    test_fix8_workload_rejects_text_bytes_iterators_and_later_invalid_items()
    for source in ("case", "machine", "hardware"):
        test_fix8b_parse_model_sentinel_remains_strict_at_workload_boundaries(source)
    return {"direct_rejections": 9, "replace_rejections": 9, "consumer_rejections": 108,
            "r8_subset_consumer_rejections": 72, "synthetic_rejections": 12,
            "duplicate_rejections": 3, "extra_request_rejections": 12,
            "valid_sequence_controls": 24, "pure_model_parses": 3,
            "sentinel_replace_rejections": 3, "sentinel_consumer_rejections": 12,
            "sentinel_immutability_rejections": 3, "failures": 0}


def _fix10_derived(record, **changes):
    cls = type("Derived" + type(record).__name__, (type(record),), {})
    value = cls(**{field.name: getattr(record, field.name) for field in dc.fields(record)})
    for name, item in changes.items():
        object.__setattr__(value, name, item)
    return value


def fix10_adjacent_witnesses():
    from mindie_sim.catalog.types import DraftSpec
    from mindie_sim.dspark_kv_physics import DsparkSwaFootprint, KimiK3DsparkGrouping
    from mindie_sim.kv_nz_read_model import Mte2ReadModel
    from mindie_sim.memory_breakdown import build_memory_ledger, memory_breakdown
    from mindie_sim.nr_s7_validate import _record_hints, _validate_nested
    from mindie_sim.parallel_decode_cost import DecodeCostModel
    from mindie_sim.recoverssm_config import GlmKdaExecution
    from mindie_sim.schema.model import SloSpec
    from test_nr_s7_fix2_regressions_r2 import fix9_forge
    case = CASE()
    decode = DecodeCostModel.from_case(case)
    memory = memory_breakdown(case)
    records = [
        (case.model, [{"hidden_size": 0}, {"norm_eps": 0}, {"state_bytes_per_request": -1},
                      {"expert_routing": (-1.0,)}, {"attention": "invalid"}]),
        (case.model.mla, [{"kv_lora_rank": 0}]),
        (case.model.layer_plans[0], [{"head_dim": -1}, {"state_bytes": -1.0}]),
        (DraftSpec("mtp", 1, 128, 256, 1, 1, 128, 128, 128), [{"layers": 0}]),
        (case.runtime, [{"max_num_seqs": 0}, {"mtp_acceptance_length": -1},
            {"mtp_acceptance_by_output_tokens": ((0, 1.0),)}, {"graph_buckets": (-1,)},
            {"gpu_memory_utilization": 2.0}, {"graph_capture_mode": "invalid"}]),
        (case.runtime.parallel, [{"tp": 0}]),
        (case.machine, [{"dies": ()}]),
        (case.machine.dies[0], [{"memory_bytes": 0}, {"compute_units": 0}, {"rank": -1},
            {"rates": (("hbm", 0.0), ("vector", 1.0))}, {"compute_peaks": (("fp32", 0.0),)},
            {"buffers": (("ub", -1),)}, {"resource_units": (("cube", 0),)}]),
        (SloSpec(), [{"ttft_ms": 0}, {"tpot_ms": -1.0}]),
        (decode, [{"hidden": 0}, {"moe_layers": decode.layers + 1}, {"launch_ms": -1.0}]),
        (decode.step(1), [{"batch_tokens": 0}, {"straggler_ratio": 0.5},
            {"scaling_mode": "invalid"}, {"overlap_mode": "invalid"}, {"pp_stage_ms": (-1.0,)}]),
        (GlmKdaExecution("kda", "bf16", 128, 128, 4, 4, "none", 128, -5.0, 128, ()),
            [{"key_dim": 0}, {"cache_mode": "invalid"}, {"activation_dtype": "invalid"}]),
        (validate_recoverssm(_config()), [{"expected_cache_mode": "invalid"}]),
        (KimiK3DsparkGrouping((), None, 0, 0, 0), [{"mamba_layers": -1}, {"mode": "invalid"}]),
        (DsparkSwaFootprint(128, 128, 0, 128, 1, 1.0), [{"row_bytes_per_token": 0}]),
        (Mte2ReadModel(*([1] * 12)), [{"head_bytes": 0}]),
        (build_memory_ledger(case), [{"pp": 0}, {"stage_capacity": [0.0]}, {"aux_rank": ((-1.0,),)}]),
        (memory, [{"stage_peak_bytes": ()}, {"per_rank_bytes": ((0, -1, 0, 0.0, 0.0),)}]),
    ]
    rejections = 0
    for record, mutations in records:
        derived = _fix10_derived(record)
        _validate_nested(derived, "valid_subclass")
        for changes in mutations:
            with pytest.raises(ModelingError):
                _validate_nested(fix9_forge(derived, **changes), "invalid_subclass")
            rejections += 1
    # The inherited diagnostic/metadata exceptions and Case workload ordering
    # must survive too, so valid extensible payloads do not become false rejects.
    for record, changes in ((memory.components[0], {"detail": (("note", ["ok"]),)}),
                            (memory, {"detail": (("note", "ok"),)}),
                            (case.model, {"metadata": {"note": ["ok"]}}),
                            (case.runtime, {"metadata": {"note": ["ok"]}})):
        _validate_nested(_fix10_derived(record, **changes), "valid_metadata")
    derived_case = _fix10_derived(case)
    assert _record_hints(type(derived_case))[0][0] == "workload"
    _validate_nested(derived_case, "valid_case")
    with pytest.raises(ModelingError, match="workload.*requests_invalid"):
        _validate_nested(fix9_forge(derived_case, workload=WorkloadSpec._pure_model_sentinel(),
                                    model=fix9_forge(case.model, hidden_size=0)), "case")
    return {"record_families": 20, "domain_rejections": rejections + 1, "metadata_controls": 4}


def test_fix10_adjacent_subclass_domains():
    fix10_adjacent_witnesses()


@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
def test_fix10_valid_subclasses_and_unhashable_consumers(capture):
    from mindie_sim.schema.model import RequestSpec
    from test_nr_s7_fix2_regressions_r2 import (
        _Fix10Request, _Fix10Workload, fix9_consumer_calls, fix9_forge, fix10_reject_before_body)
    class HashableId(str):
        pass
    class UnhashableId(str):
        __hash__ = None
    base = CASE()
    case = _fix10_derived(base, runtime=dc.replace(base.runtime, graph_capture_mode=capture))
    for rcls in (RequestSpec, _Fix10Request):
        for wcls in (WorkloadSpec, _Fix10Workload):
            request = rcls(HashableId("valid"), 0, 128, 1)
            workload = wcls((request,))
            for call in fix9_consumer_calls(dc.replace(case, workload=workload)).values():
                assert call() is not None
            bad = fix9_forge(request, request_id=UnhashableId("invalid"))
            for call in (lambda: wcls((bad,)), lambda: dc.replace(workload, requests=(bad,)),
                         lambda: wcls((request, request)),
                         lambda: dc.replace(workload, requests=(request, request))):
                with pytest.raises(ModelingError):
                    call()
            broken = dc.replace(case, workload=fix9_forge(workload, requests=(bad,)))
            fix10_reject_before_body(fix9_consumer_calls(broken).values())


def fix10_witnesses():
    from test_nr_s7_fix2_regressions_r2 import (
        test_fix10_subclass_consumer_boundaries, test_fix10_unhashable_string_id_at_request_boundaries)
    test_fix10_unhashable_string_id_at_request_boundaries()
    adjacent = fix10_adjacent_witnesses()
    for capture in ("none", "full", "piecewise"):
        test_fix10_subclass_consumer_boundaries(capture)
        test_fix10_valid_subclasses_and_unhashable_consumers(capture)
    return {"subclass_consumer_rejections": 480, "consumer_body_entries": 0,
            "unhashable_request_rejections": 10, "unhashable_consumer_rejections": 120,
            "valid_subclass_consumer_controls": 120, "adjacent": adjacent,
            "capture_modes": ["none", "full", "piecewise"], "consumer_forms": 10}


def fix10_source_evidence(artifact_dir):
    import json
    load = lambda name: json.loads((artifact_dir / name).read_text())
    subclass = load("r10_subclass_results.json")
    rows = subclass["request_rows"] + subclass["workload_rows"]
    assert len(rows) == 480 and all(row["outcome"] == "ModelingError"
        and not row["body_entries"] and row["nested_validator"]["outcome"] == "ModelingError" for row in rows)
    fresh = load("r10_fresh_results.json")
    assert fresh["checks"] == 11013 and not fresh["failures"] and not fresh["consumer_body_entries"]
    unhashable = fresh["unhashable_string_id"]
    uid_rows = [row for row in unhashable.values() if "outcome" in row]
    uid_rows += [row for mode in ("none", "full", "piecewise") for row in unhashable[mode].values()]
    assert len(uid_rows) == 37 and all(row["outcome"] == "ModelingError" for row in uid_rows)
    assert fresh["unhashable_rejections"] == 37 and fresh["unhashable_consumer_calls"] == 30
    bpe = load("r10_bpe_results.json")
    assert bpe["accepted"] + bpe["mandatory_budget_rejections"] == 3624
    assert all(row["bytes"] <= 250 and max(row["tokens"].values()) <= 250
               and max(row["summary_tokens"].values()) <= 250 for row in bpe["rows"])
    return {"subclass_calls": len(rows), "ordinary_rejections": fresh["checks"],
            "ordinary_consumer_calls": sum(n for key, n in fresh["counts"].items() if "consumers" in key),
            "unhashable_rejections": len(uid_rows),
            "bpe": {k: bpe[k] for k in ("accepted", "mandatory_budget_rejections", "maximum_tokens", "maximum_bytes")}}


def fix10_report(data):
    evidence = data["fix10"]
    return f"""**R10-P2-01 and R10-P3-02 closure**

Domain dispatch follows registered base-class identity through inheritance for
RequestSpec, WorkloadSpec and all other special record families in nr_s7_validate.
Case subclasses retain workload-first traversal. Empty base and subclass sentinel
workloads reject before consumer bodies; parse_model retains its sentinel factory.
The request contract checks hash(request_id) and translates TypeError to ModelingError
at construction, replacement, mapping and both shared/nested bypass validators.
Hashable string subclasses remain accepted, including in RequestSpec subclasses.

Permanent witnesses: `{evidence['permanent']}`.
The copied R10 subclass probe rejects all {evidence['subclass_calls']} calls with
zero invalid accepts and zero consumer implementation entries. The fresh probe
rejects {evidence['ordinary_rejections']} ordinary calls plus
{evidence['unhashable_rejections']} unhashable-ID calls. The ordinary matrix has
{evidence['ordinary_consumer_calls']} profiled consumers with zero body entries;
the unhashable consumer calls are also profiled before entry.
The copied probes strengthen defect-reproduction assertions to rejection assertions.
The dual-BPE sweep emits {evidence['bpe']['accepted']} envelopes and rejects
{evidence['bpe']['mandatory_budget_rejections']} insufficient budgets; both real
tokenizers remain within the 250-token cap (maxima {evidence['bpe']['maximum_tokens']}).
"""
