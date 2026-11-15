"""NR-S7 FIX2 regression witnesses (grouping, DCP, prefix, KDA writes).

[Role]
  Encode the Round-2 counterexamples for F07..F11: source-strict Kimi grouping,
  DCP output-head/window/alignment physics, joint identity and prompt extent,
  immutable prefix plans, committed conv traffic, per-layer verify launches and
  the non-A5 strided conv tile.

[Invariants]
  Each assertion is an exact source-derived identity or typed rejection; no
  test asserts a value copied from the implementation under test.

[Timeouts & Locks]
  Pure in-memory arithmetic; no device, subprocess or network access.

[Side-Effect Policy]
  Builds immutable records; creates nothing outside pytest's temporary root.
"""
from __future__ import annotations

import dataclasses as dc
import json

import pytest

from nr_s7_fix2_support import (
    _base_specs, _case, _config, _dcp, _draft, _ds41, _geometry, _glm,
    _identity, _kda_layers, _keys, _kimi, _layer, _spec, _target)

from mindie_sim.dspark_dcp_cost import DcpDraftBlockCost
from mindie_sim.dspark_kv_physics import kimi_k3_dspark_grouping
from mindie_sim.dspark_prefix_cache import dspark_prefix_plan
from mindie_sim.hybrid_kv_state_report import evaluate_hybrid_kv_state
from mindie_sim.linear_state_replay import verify_commit_cost
from mindie_sim.recoverssm_launch_geometry import commit_launch_geometry
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec, WorkloadSpec, SloSpec
from mindie_sim.hybrid_kv_state_options import HybridKvStateOptions
from mindie_sim.memory_breakdown import (
    analyze_memory_breakdown, build_memory_ledger, memory_breakdown, memory_ledger)


# ------------------------------------------- F07 source-strict grouping


def test_f07_arbitrary_names_are_not_a_source_grouping():
    specs = [_spec("t0", "full_attention"), _spec("d0", "full_attention",
                                                  scope="draft"),
             _spec("m0")]
    grouping = kimi_k3_dspark_grouping(specs)
    assert grouping.ok is False
    assert grouping.mode == "analytical_fallback_split"
    assert grouping.source_compatible is False
    assert grouping.fallback_reason == "unrecognized_target_layer_prefix"
    with pytest.raises(ModelingError, match="unrecognized_target_layer_prefix"):
        kimi_k3_dspark_grouping(specs, strict=True)


def test_f07_source_compatible_prefixes_merge_one_table():
    grouping = kimi_k3_dspark_grouping(_base_specs(), strict=True)
    assert grouping.ok and grouping.mode == "source_uniform_grouping"
    assert grouping.scheduler_block_tables == 1
    # One mixed attention table plus one balanced recurrent group of three.
    assert grouping.group_sizes == (3, 3)


def test_f07_sliding_attention_cannot_join_the_uniform_group():
    specs = [_spec(_target(0), "sliding_attention"),
             _spec(_target(1), "full_attention"),
             _spec(_draft(0), "full_attention", scope="draft"),
             _spec(_target(10))]
    grouping = kimi_k3_dspark_grouping(specs)
    assert grouping.fallback_reason == "attention_spec_is_not_full_attention"


def test_f07_global_page_size_must_match_across_all_specs():
    specs = _base_specs()
    specs[2] = _spec(_draft(0), "full_attention", scope="draft",
                     page=200_000)
    grouping = kimi_k3_dspark_grouping(specs)
    assert grouping.fallback_reason == "page_size_mismatch"


def test_f07_mamba_specs_must_be_fully_resolved():
    specs = _base_specs()
    specs[-1] = _spec(_target(12), shapes=(), dtypes=())
    grouping = kimi_k3_dspark_grouping(specs)
    assert grouping.fallback_reason == "mamba_spec_not_fully_resolved"


def test_f07_attention_head_mismatch_blocks_the_merge():
    specs = _base_specs()
    specs[2] = _spec(_draft(0), "full_attention", scope="draft", heads=32)
    grouping = kimi_k3_dspark_grouping(specs)
    assert grouping.fallback_reason == "attention_spec_mismatch"


# ----------------------------------------------- F08 DCP physics


def test_f08_combine_multiplies_output_and_lse_by_gathered_heads():
    cost = DcpDraftBlockCost(**_dcp())
    assert cost.gathered_heads == 16
    assert cost.combine_bytes == 206_400
    assert cost.combine_bytes == pytest.approx(
        0.5 * 25 * 16 * (512 + 4) * 2)


def test_f08_pcp_output_merge_composes_tp_scatter_and_pcp_gather():
    cost = DcpDraftBlockCost(**_dcp(pcp=2, dcp=4, tp=8))
    assert cost.gather_peers == 7
    # ``common_cp.py`` gathers the distinct TP head set: 8 local heads times
    # the 8 TP sources, independently of the declared DCP ratio.
    payload = 25 * (8 * 8) * (512 * 2 + 4)
    packed = 25 * 64 * (512 + 4) * 2
    assert cost.gathered_heads == 64
    assert cost.merge_payload_bytes == payload
    assert cost.combine_tp_bytes == pytest.approx(7 / 8 * packed)
    assert cost.combine_pcp_bytes == pytest.approx(packed)
    assert cost.combine_bytes == pytest.approx(
        cost.combine_tp_bytes + cost.combine_pcp_bytes)
    equal = DcpDraftBlockCost(**_dcp(pcp=2, dcp=2))
    assert equal.gathered_heads == 8
    assert equal.combine_tp_bytes == 0.0
    assert equal.combine_bytes == equal.packed_merge_bytes == 206_400


def test_f08_equal_pcp_size_has_zero_query_gather():
    cost = DcpDraftBlockCost(**_dcp(pcp=2, dcp=2))
    assert cost.gather_peers == 0
    assert cost.gather_bytes == 0.0
    assert cost.combine_bytes > 0.0


def test_f08_index_alignment_is_128_with_int64_physical_lengths():
    single = DcpDraftBlockCost(**_dcp())
    assert single.index_width == 256
    assert single.index_bytes == 25_700
    ds41 = DcpDraftBlockCost(**_dcp(num_draft_layers=3))
    assert ds41.rows == 15
    assert ds41.index_bytes == 15_420
    physical = DcpDraftBlockCost(**_dcp(
        physical_index_path=True, index_length_dtype="int64",
        index_length_bytes=8))
    assert physical.index_bytes == 25 * (256 * 4 + 8)
    with pytest.raises(ModelingError, match="physical_index_requires_int64"):
        DcpDraftBlockCost(**_dcp(physical_index_path=True))


def test_f08_swa_attention_work_is_window_bounded():
    short = DcpDraftBlockCost(**_dcp(context_tokens=4096))
    long = DcpDraftBlockCost(**_dcp(context_tokens=8192))
    assert short.visible_tokens == long.visible_tokens == 133
    assert short.attention_flops == long.attention_flops
    assert short.attention_ms == long.attention_ms
    assert short.local_visible_tokens == 67
    assert short.kv_row_elements == 576


# ------------------------------------- F09 identity, extent, immutability


def test_f09_joint_identity_distinguishes_all_five_kimi_variants():
    case = _case("kimi-k3")
    auxiliary = case.model.auxiliary
    draft = auxiliary.dspark
    variants = [
        case.model,
        dc.replace(case.model, auxiliary=dc.replace(
            auxiliary, dspark=dc.replace(draft, activation_dtype="fp32"))),
        dc.replace(case.model, auxiliary=dc.replace(
            auxiliary, dspark=dc.replace(
                draft, markov_rank=512,
                confidence_input_width=draft.hidden_size + 512))),
        dc.replace(case.model, norm_eps=case.model.norm_eps * 2),
        dc.replace(case.model, layer_plans=tuple(
            dc.replace(plan, recurrent_state_dtype="bf16")
            if plan.attention == "kda" else plan
            for plan in case.model.layer_plans)),
    ]
    digests = []
    for model in variants:
        report = evaluate_hybrid_kv_state(
            dc.replace(case, model=model), _kimi(dspark_active=True))
        digests.append(report["dspark"]["prefix"]["identity"])
    assert len(set(digests)) == 5


def test_f09_explicit_available_hashes_bound_the_prompt_extent():
    report = evaluate_hybrid_kv_state(
        _case("kimi-k3", max_model_len=8192, prompt=256),
        _kimi(dspark_active=True, sequence_tokens=256,
              prefix_hit_tokens=128, prefix_available_hashes=1))
    assert report["status"] == "rejected"
    assert any("missing_block_hashes:2>1" in code
               for code in report["errors"])


def test_f09_restore_beyond_the_actual_prompt_extent_is_rejected():
    report = evaluate_hybrid_kv_state(
        _case("deepseek-v4.1-flash", max_model_len=8192, prompt=128),
        _ds41(dspark_active=True, prefix_hit_tokens=4096,
              sequence_tokens=4096, batch=1))
    assert report["status"] == "rejected"
    assert any("hit_exceeds_prompt:4096>128" in code
               for code in report["errors"])
    assert report["transfer"]["dspark_prefix_per_rank_bytes"] == 0.0


def test_f09_plan_is_immutable_and_validates_the_destination_table():
    plan = dspark_prefix_plan(
        keys=_keys(), page_geometry=_geometry(), aux_layer_ids=(1,),
        target_hidden_size=128, target_dtype="bf16", tokens=128,
        prompt_tokens=128, block_hash_count=1)
    assert plan.num_blocks == 1 and plan.draft_groups == (0,)
    assert plan.require_hash_equals_block is True
    plan.validate_block_table({0: [0]}, [0])
    for table in ({}, {0: []}, {0: [1]}, {0: [True]}):
        with pytest.raises(ModelingError):
            plan.validate_block_table(table, [0])
    with pytest.raises(dc.FrozenInstanceError):
        plan.num_blocks = 2


def test_f09_mutated_geometry_and_boolean_aux_are_rejected():
    with pytest.raises(ModelingError, match="dspark_aux_layers_invalid"):
        dspark_prefix_plan(
            keys=_keys(), page_geometry=_geometry(), aux_layer_ids=(True,),
            target_hidden_size=128, target_dtype="bf16", tokens=128,
            prompt_tokens=128, block_hash_count=1)
    geometry = {**_geometry(), "hash_block_size": 64}
    with pytest.raises(ModelingError, match="geometry_mutated:hash_block_size"):
        dspark_prefix_plan(
            keys=_keys(), page_geometry=geometry, aux_layer_ids=(1,),
            target_hidden_size=128, target_dtype="bf16", tokens=256,
            prompt_tokens=256, block_hash_count=2)
    with pytest.raises(ModelingError, match="exceeds_registered_blocks"):
        dspark_prefix_plan(
            keys=_keys(), page_geometry=_geometry(), aux_layer_ids=(1,),
            target_hidden_size=128, target_dtype="bf16", tokens=256,
            prompt_tokens=256, block_hash_count=2)


# ---------------------------------------------------- F11 KDA write physics


def test_f11_committed_conv_and_boundary_checkpoint_bytes():
    align = _config(mamba_cache_mode="align", enable_prefix_caching=True)
    cost = verify_commit_cost(
        align, [_layer()], batch=2, sequence_tokens=2047,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02, accepted_tokens=3)
    assert cost["committed_conv_bytes"] == 2 * 18_432
    assert cost["boundary_read_bytes"] == 2 * 18_432
    assert cost["boundary_write_bytes"] == 2 * 542_720
    assert cost["boundary_extra_bytes"] == 1_122_304
    assert cost["conv_state_bytes"] == 2 * 43_008


def test_f11_stage1_verify_launches_scale_with_kda_layers():
    layers = tuple(_layer(name=f"layer{index}", index=index)
                   for index in range(34))
    align = _config(mamba_cache_mode="align", enable_prefix_caching=True)
    cost = verify_commit_cost(
        align, layers, batch=2, sequence_tokens=2048,
        hbm_bytes_per_s=1.2e12, kernel_launch_ms=0.02)
    assert cost["stage1_timing"]["launch_ms"] == pytest.approx(0.68)
    assert cost["kernel_launches"] == 34 + 4
    assert cost["stage2_timing"]["launch_ms"] == pytest.approx(4 * 0.02)


def test_f11_strided_conv_uses_the_256_channel_tile_on_non_a5():
    strided = commit_launch_geometry(
        key_dim=128, value_dim=128, num_heads=8, batch=1, layers=1,
        align_mode=False, is_a5=False, conv_dim=3072, conv_dim_stride=2)
    assert strided.conv_tile == 256
    assert strided.conv_programs == 12
    contiguous = commit_launch_geometry(
        key_dim=128, value_dim=128, num_heads=8, batch=1, layers=1,
        align_mode=False, is_a5=False, conv_dim=3072, conv_dim_stride=1)
    assert contiguous.conv_tile == 4096
    assert contiguous.conv_programs == 1
    a5 = commit_launch_geometry(
        key_dim=128, value_dim=128, num_heads=8, batch=1, layers=1,
        align_mode=False, is_a5=True, conv_dim=3072, conv_dim_stride=2)
    assert a5.conv_tile == 4096 and a5.conv_programs == 1


# F14 / R9-P2-01: constructor, replacement and independent consumer defenses.


def fix9_bad_request_fields():
    integers = (-1, True, False, 1.0, 1.5, "1", None, [], {},
                float("nan"), float("inf"), -float("inf"), 10**400, -10**400)
    groups = {name: integers for name in ("prompt_tokens", "output_tokens",
        "prefix_tokens", "priority", "text_prompt_tokens", "media_tokens", "chain_id")}
    for name in ("prompt_tokens", "output_tokens"):
        groups[name] += (0,)
    groups.update(request_id=([], {}, None, 1, True, "", " \t\n", b"ok"),
                  media_encoded=(0, 1, "true", None, [], {}))
    for name in ("arrival_ms", "gap_ms"):
        groups[name] = (-1, -0.1, True, False, "0", None, [], {},
                        float("nan"), float("inf"), -float("inf"), 10**400, -10**400)
    return [(name, bad) for name, values in groups.items() for bad in values] + [
        ("prefix_tokens", 129), ("media_tokens", 129),
        ("text_prompt_tokens", 127), ("text_prompt_tokens", 129)]


def fix9_forge(record, **changes):
    """Deliberately skip constructors so consumer checks are tested separately."""
    forged = object.__new__(type(record))
    for field in dc.fields(record):
        object.__setattr__(forged, field.name, changes.get(field.name, getattr(record, field.name)))
    return forged


def fix9_consumer_calls(case):
    return {
        "hybrid_default": lambda: evaluate_hybrid_kv_state(case),
        "hybrid_explicit": lambda: evaluate_hybrid_kv_state(
            case, HybridKvStateOptions(sequence_tokens=128)),
        "ledger_default": lambda: build_memory_ledger(case),
        "ledger_explicit": lambda: build_memory_ledger(case, sequence_tokens=128),
        "memory_default": lambda: memory_breakdown(case),
        "memory_explicit": lambda: memory_breakdown(case, sequence_tokens=128, sequences=1),
        "ledger_alias_default": lambda: memory_ledger(case),
        "ledger_alias_explicit": lambda: memory_ledger(case, sequence_tokens=128),
        "analysis_default": lambda: analyze_memory_breakdown(case),
        "analysis_explicit": lambda: analyze_memory_breakdown(case, sequence_tokens=128, sequences=1),
    }


@pytest.mark.parametrize("field", tuple(RequestSpec.__dataclass_fields__))
def test_fix9_request_constructor_replace_mapping_and_workload(field):
    request = RequestSpec("ok", 0, 128, 1)
    workload = WorkloadSpec((request,))
    for name, bad in fix9_bad_request_fields():
        if name != field:
            continue
        changes = {name: bad}
        with pytest.raises(ModelingError):
            RequestSpec(**{**dc.asdict(request), **changes})
        with pytest.raises(ModelingError):
            dc.replace(request, **changes)
        with pytest.raises(ModelingError):
            RequestSpec.from_mapping({**request.to_dict(), **changes}, 0)
        forged = fix9_forge(request, **changes)
        with pytest.raises(ModelingError):
            WorkloadSpec((forged,))
        with pytest.raises(ModelingError):
            dc.replace(workload, requests=(forged,))


@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
def test_fix9_bypass_request_rejected_by_every_consumer(capture):
    case = _case()
    case = dc.replace(case, runtime=dc.replace(case.runtime, graph_capture_mode=capture))
    request = RequestSpec("bad", 0, 128, 1)
    for name, bad in fix9_bad_request_fields():
        forged = fix9_forge(request, **{name: bad})
        # Later elements must be checked even though token defaults use the first.
        workload = fix9_forge(case.workload, requests=(*case.workload.requests, forged))
        broken = dc.replace(case, workload=workload)
        for api, call in fix9_consumer_calls(broken).items():
            with pytest.raises(ModelingError) as error:
                call()
            assert error.value.reason, (name, api)


def fix9_bad_metadata():
    return (None, 1, True, {}, set(), "ab", b"ab", bytearray(b"ab"),
            (None,), (1,), ((),), (("key",),), (("a", "b", "c"),),
            ("ab",), (b"ab",), ({"a": "b"},), ((1, "b"),),
            (("a", None),), (("a", []),), (("ok", "v"), ("bad", False)))


def test_fix9_metadata_constructor_replace_and_immutable_snapshot():
    from collections import UserList, deque
    request = RequestSpec("ok", 0, 128, 1)
    workload = WorkloadSpec((request,))
    for bad in fix9_bad_metadata():
        with pytest.raises(ModelingError):
            WorkloadSpec((request,), metadata=bad)
        with pytest.raises(ModelingError):
            dc.replace(workload, metadata=bad)
    for container in (list, tuple, UserList, deque):
        pair = ["source", "fixture"]
        supplied = container([pair])
        normalized = dc.replace(workload, metadata=supplied)
        pair[1] = "changed"
        assert normalized.metadata == (("source", "fixture"),)
        assert type(normalized.metadata) is tuple and type(normalized.metadata[0]) is tuple


@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
def test_fix9_workload_domains_before_consumer_body(capture):
    import sys
    case = _case()
    case = dc.replace(case, runtime=dc.replace(case.runtime, graph_capture_mode=capture))
    malformed = [fix9_forge(case.workload, metadata=bad) for bad in fix9_bad_metadata()]
    malformed += [fix9_forge(case.workload, requests=case.workload.requests * 2),
                  fix9_forge(case.workload, synthetic=1), WorkloadSpec._pure_model_sentinel(),
                  fix9_forge(case.workload, requests=(fix9_forge(
                      case.workload.requests[0], output_tokens=-1),))]
    bodies = {fn.__wrapped__.__code__ for fn in (evaluate_hybrid_kv_state,
        build_memory_ledger, memory_breakdown, memory_ledger, analyze_memory_breakdown)}
    entered = []
    def profile(frame, event, arg):
        if event == "call" and frame.f_code in bodies:
            entered.append(frame.f_code.co_name)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        for workload in malformed:
            for call in fix9_consumer_calls(dc.replace(case, workload=workload)).values():
                with pytest.raises(ModelingError):
                    call()
    finally:
        sys.setprofile(previous)
    assert entered == []


def test_fix9_valid_request_shorthand_media_boundaries_and_roundtrip():
    class RequestId(str):
        pass
    requests = [RequestSpec("ok", 0, 128, 1),
        RequestSpec("media", 1, 128, 1, prefix_tokens=128, text_prompt_tokens=96,
                    media_tokens=32, priority=1, chain_id=1, gap_ms=2, media_encoded=True),
        RequestSpec("media_only", 0, 128, 1, media_tokens=128),
        RequestSpec(RequestId("string_subclass"), 0, 128, 1)]
    for request in requests:
        assert type(request.arrival_ms) is type(request.gap_ms) is float
        assert dc.replace(request) == request
        roundtrip = RequestSpec.from_mapping(request.to_dict(), 0)
        assert roundtrip.to_dict() == request.to_dict()
        WorkloadSpec([request], metadata=[["source", "fixture"]])
    with pytest.raises(ModelingError):
        RequestSpec("bad", 0, 128, 1, text_prompt_tokens=100, media_tokens=32)
    events = []
    def pairs():
        events.append("consumed")
        yield ("source", "fixture")
    stream = pairs()
    with pytest.raises(ModelingError):
        WorkloadSpec((requests[0],), metadata=stream)
    assert events == []
    stream.close()


@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
def test_fix9_positive_consumer_controls(capture):
    case = _case()
    case = dc.replace(case, workload=WorkloadSpec([RequestSpec("ok", 0, 128, 1)]),
                      runtime=dc.replace(case.runtime, graph_capture_mode=capture))
    for call in fix9_consumer_calls(case).values():
        assert call() is not None


def test_fix9_adjacent_schema_domains_at_consumers():
    case = _case()
    # Other schema records retain their public constructors; nested consumer
    # checks must reject malformed records produced by normal replacement too.
    records = [("model", case.model, {"norm_eps": 0}),
        ("model", case.model, {"state_bytes_per_request": -1}),
        ("model", case.model, {"hidden_size": 0}),
        ("model", case.model, {"num_heads": 10**400}),
        ("model", case.model, {"hidden_size": "128"}),
        ("model", case.model, {"mla": dc.replace(case.model.mla, kv_lora_rank=0)}),
        ("model", case.model, {"layer_plans": (dc.replace(case.model.layer_plans[0], head_dim=-1),)}),
        ("model", case.model, {"layer_plans": (dc.replace(case.model.layer_plans[0], state_bytes=-1),)}),
        ("runtime", case.runtime, {"max_num_seqs": 0}),
        ("runtime", case.runtime, {"max_model_len": 10**400}),
        ("runtime", case.runtime, {"gpu_memory_utilization": float("inf")}),
        ("runtime", case.runtime, {"mtp_acceptance_length": -1}),
        ("runtime", case.runtime, {"parallel": None}),
        ("slo", SloSpec(), {"ttft_ms": 0}), ("slo", SloSpec(), {"tpot_ms": -1}),
        ("slo", SloSpec(), {"itl_ms": True}), ("slo", SloSpec(), {"e2e_ms": 10**400})]
    for slot, record, changes in records:
        broken = dc.replace(case, **{slot: dc.replace(record, **changes)})
        for call in fix9_consumer_calls(broken).values():
            with pytest.raises(ModelingError):
                call()
    for field in ("tp", "pp", "dp", "ep", "cp", "dcp", "pcp"):
        for bad in (0, -1, True, "1"):
            with pytest.raises(ModelingError):
                dc.replace(case.runtime.parallel, **{field: bad})
    huge = dc.replace(case.runtime.parallel, tp=10**400)
    broken = dc.replace(case, runtime=dc.replace(case.runtime, parallel=huge))
    for call in fix9_consumer_calls(broken).values():
        with pytest.raises(ModelingError):
            call()


def test_fix9_giant_values_cannot_break_error_formatting():
    request = RequestSpec("ok", 0, 128, 1)
    case = _case()
    giant = 10**5000  # Beyond Python's default decimal-formatting limit too.
    for name in RequestSpec.__dataclass_fields__:
        for bad in (giant, [giant], {"bad": giant}):
            changes = {name: bad}
            with pytest.raises(ModelingError):
                RequestSpec(**{**dc.asdict(request), **changes})
            with pytest.raises(ModelingError):
                dc.replace(request, **changes)
            with pytest.raises(ModelingError):
                RequestSpec.from_mapping({**request.to_dict(), **changes}, 0)
            forged = fix9_forge(request, **changes)
            with pytest.raises(ModelingError):
                WorkloadSpec((forged,))
            broken = dc.replace(case, workload=fix9_forge(case.workload, requests=(forged,)))
            for call in fix9_consumer_calls(broken).values():
                with pytest.raises(ModelingError):
                    call()


def fix9_witnesses():
    for field in RequestSpec.__dataclass_fields__:
        test_fix9_request_constructor_replace_mapping_and_workload(field)
    test_fix9_metadata_constructor_replace_and_immutable_snapshot()
    test_fix9_valid_request_shorthand_media_boundaries_and_roundtrip()
    test_fix9_adjacent_schema_domains_at_consumers()
    test_fix9_giant_values_cannot_break_error_formatting()
    for capture in ("none", "full", "piecewise"):
        test_fix9_bypass_request_rejected_by_every_consumer(capture)
        test_fix9_workload_domains_before_consumer_body(capture)
        test_fix9_positive_consumer_controls(capture)
    mutations = len(fix9_bad_request_fields())
    metadata = len(fix9_bad_metadata())
    return {"request_mutations": mutations, "request_constructor_rejections": mutations,
        "request_replace_rejections": mutations, "request_mapping_rejections": mutations,
        "workload_bypass_constructor_replace_rejections": 2 * mutations,
        "request_consumer_rejections": mutations * 3 * 10,
        "metadata_constructor_replace_rejections": metadata * 2,
        "workload_consumer_rejections_before_body": (metadata + 4) * 3 * 10,
        "consumer_body_entries": 0, "valid_consumer_controls": 30,
        "capture_modes": ["none", "full", "piecewise"], "consumer_forms": 10}


def fix9_report(data):
    checks = data["fix9"]
    source = data["source_probes"]["r9_malformed"]["summary"]
    fresh = data["source_probes"]["r9_fresh"]["summary"]
    return f"""**R9-P2-01 request fields and workload metadata**

RequestSpec construction and dataclasses.replace require a nonblank string ID,
positive strict-int prompt/output counts, nonnegative strict-int prefix/priority/
text/media/chain counts, finite nonnegative real arrival/gap times stored as floats,
and a bool media_encoded flag. Counts beyond finite binary64 reject before arithmetic.
Prefix/media cannot exceed the prompt; a positive text count requires text+media=prompt.
The direct text=0/media=0 shorthand remains valid. Explicit mapping fields use the same
request contract. WorkloadSpec validates every request before hashing IDs and snapshots
metadata sequences into immutable (str, str) pairs. Malformed pairs reject.

The shared consumer validator rechecks these invariants on bypass-injected records
before default/explicit sequence-token selection. The six main forms plus both forms
of memory_ledger and analyze_memory_breakdown run in none/full/piecewise capture modes.
Permanent evidence: `{checks}`. Profiling confirms zero consumer implementation entries
for invalid metadata, duplicate IDs, non-bool synthetic, sentinel and negative output.
Adjacent ModelSpec/MlaSpec/LayerPlan, Software, ParallelSpec, Case and SloSpec records
retain nested type/domain validation; optional zero geometry and signed gate bounds
remain valid. Additional regressions exercise malformed normal replacements at consumers.

The adapted independent R9 malformed-request probe executes {source['request_mutations']}
request mutations and {source['metadata_mutations']} metadata mutations:
**{source['rejections']}/{source['rejections']} ModelingError rejections
({source['rejection_percent']:.0f}%), {source['failures']} failures**, with
{source['valid_consumer_controls']} accepted consumer controls. Its original defect
assertions were replaced with rejection assertions; constructor bypass is deliberate
and used only to test the independent workload/consumer defense.

**Retained R8/FIX8B workload and pure-model boundary**

Empty/non-sequence requests and non-RequestSpec elements reject; valid public sequences
become tuples, synthetic requires bool, and duplicate request IDs reject.
Permanent counts: `{data['fix8']['workload']}`. The copied R8 probe retains 12/12
constructor/replacement and 72/72 bypass-consumer rejections.
The fresh R9 workload probe retains {fresh['construct']} constructor,
{fresh['replace_valid']} valid-workload replacement and {fresh['replace_sentinel']}
sentinel replacement rejections; {fresh['bypass_consumers']} malformed-container
consumer and {fresh['sentinel_consumers']} sentinel consumer rejections; zero failures.
Pure-model parsing still uses the immutable empty sentinel, whose ordinary replacement
and workload consumption reject; replacing it with valid requests creates a normal workload.
"""


class _Fix10Request(RequestSpec):
    pass


class _Fix10Workload(WorkloadSpec):
    pass


def fix10_reject_before_body(calls):
    import sys
    bodies = {fn.__wrapped__.__code__ for fn in (evaluate_hybrid_kv_state,
        build_memory_ledger, memory_breakdown, memory_ledger, analyze_memory_breakdown)}
    entered = []
    def profile(frame, event, arg):
        if event == "call" and frame.f_code in bodies:
            entered.append(frame.f_code.co_name)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        for call in calls:
            with pytest.raises(ModelingError):
                call()
    finally:
        sys.setprofile(previous)
    assert entered == []


@pytest.mark.parametrize("capture", ("none", "full", "piecewise"))
def test_fix10_subclass_consumer_boundaries(capture):
    from mindie_sim.nr_s7_validate import _validate_nested
    base = _case()
    case = dc.replace(base, runtime=dc.replace(base.runtime, graph_capture_mode=capture))
    malformed = []
    for rcls in (RequestSpec, _Fix10Request):
        for wcls in (WorkloadSpec, _Fix10Workload):
            for field, bad in (("output_tokens", -1), ("arrival_ms", -1.0), ("prefix_tokens", -1)):
                request = rcls("bad", 0, 128, 1)
                workload = wcls((request,))
                object.__setattr__(request, field, bad)
                malformed.append(workload)
    request = RequestSpec("duplicate", 0, 128, 1)
    for wcls in (WorkloadSpec, _Fix10Workload):
        malformed.extend((fix9_forge(wcls((request,)), requests=(request, request)),
                          wcls._pure_model_sentinel()))
    calls = []
    for workload in malformed:
        with pytest.raises(ModelingError):
            _validate_nested(workload, "workload")
        calls.extend(fix9_consumer_calls(dc.replace(case, workload=workload)).values())
    assert len(calls) == 160
    fix10_reject_before_body(calls)


def test_fix10_unhashable_string_id_at_request_boundaries():
    from mindie_sim.nr_s7_validate import _validate_nested, _validate_request
    class UnhashableId(str):
        __hash__ = None
    class HashableId(str):
        pass
    for cls in (RequestSpec, _Fix10Request):
        request = cls("valid", 0, 128, 1)
        uid = UnhashableId("bad")
        forged = fix9_forge(request, request_id=uid)
        for call in (lambda: cls(uid, 0, 128, 1),
                     lambda: dc.replace(request, request_id=uid),
                     lambda: cls.from_mapping({**request.to_dict(), "request_id": uid}, 0),
                     lambda: _validate_request(forged),
                     lambda: _validate_nested(forged, "request")):
            with pytest.raises(ModelingError, match="request_id_invalid:unhashable"):
                call()
        hid = HashableId("valid")
        controls = (cls(hid, 0, 128, 1), dc.replace(request, request_id=hid),
                    cls.from_mapping({**request.to_dict(), "request_id": hid}, 0))
        for valid in controls:
            assert type(valid.request_id) is HashableId
            _validate_request(valid)
            _validate_nested(valid, "request")
        with pytest.raises(ModelingError):
            cls("bad", -1, 128, 1)
        with pytest.raises(ModelingError):
            dc.replace(request, arrival_ms=-1)
