"""Permanent FIX4–FIX8 regressions for R3–R7 closure contracts.

[Role]
Exercise exported builders, sentence polarity, strict booleans, event causality
and all tail-only scheduling metadata through their public boundaries.
[Invariants]
Conditional aliases keep canonical provenance. Expected schedules use an
independent recurrence; neither event existence nor truthiness proves consent.
[Timeouts & Locks]
Small deterministic loops; no waits, locks, network or subprocesses.
[Side-Effect Policy]
Read-only tests; test artifacts belong in caller-selected external scratch.
"""
import pytest as _pytest; _pytest.importorskip("mindie_profiler")

from dataclasses import fields
from fractions import Fraction
import pytest
from mindie_knowledge.distiller_text import normalize_text
from mindie_profiler.aclgraph_mcp import aclgraph_diagnose, aclgraph_launch_floor
from mindie_profiler.aclgraph_multistream import diagnose_region
from mindie_sim.opflow import build_flow, evaluate_flow, get_profile
from mindie_sim.opflow.catalog_attention import _attn
from mindie_sim.opflow.catalog_matmul import _matmul
from mindie_sim.opflow.opflow_mcp import call_tool, opflow_evaluate, opflow_triton
from mindie_sim.opflow.paradigm_catalog import CATALOG, STRUCTURAL_ONLY, evaluate_operator
from mindie_sim.opflow.schedule_math import pipeline_tail
from mindie_sim.opflow.triton_lowering import TritonFeatures
P = get_profile("Ascend950PR")
BAD_BOOLS = ("false", "true", None, 0, 1, [], {})
CONDITIONAL = dict.fromkeys(("flash_kda", "mega_mla", "mega_full_index", "moonep",
                            "mhc_hcpre_cv_splitk", "grouped_matmul_finalize_routing"))
CONDITIONAL = {name: name for name in CONDITIONAL} | {
    "flash_kda_fwd": "flash_kda", "mega_recurrent_kda": "flash_kda"}

@pytest.mark.parametrize("name", tuple(CONDITIONAL))
def test_conditional_raw_builders_and_all_prediction_boundaries(name):
    raw = CATALOG[name].builder()
    assert raw.estimate_kind == "conditional_ledger_required"
    assert raw.params["canonical_route"] == CONDITIONAL[name]
    assert raw.params["derivation_coverage"] == raw.estimate_kind
    assert raw.params["runtime_confidence"] == "analytic_prior_not_source_certification"
    assert not raw.calibrated
    for options in ({}, {"allow_structural_prior": False},
                    *({"allow_structural_prior": bad} for bad in BAD_BOOLS)):
        with pytest.raises(ValueError):
            evaluate_flow(raw, P, **options)
        with pytest.raises(ValueError):
            evaluate_operator(name, P, **options)
        assert opflow_evaluate(name, **options)["ok"] == 0
        assert opflow_triton(name, **options)["ok"] == 0
    for result in (evaluate_flow(raw, P, allow_structural_prior=True),
                   evaluate_operator(name, P, allow_structural_prior=True)):
        assert result.estimate_kind == "conditional_ledger_required"
        assert not result.calibrated
    for endpoint in (opflow_evaluate, opflow_triton):
        result = endpoint(name, allow_structural_prior=True)
        assert result["ok"] == 1 and result["cal"] == 0
        assert result["est"] == "conditional_ledger_required"

@pytest.mark.parametrize("name", tuple(CATALOG))
def test_every_catalog_builder_preserves_gate_when_wrapped(name):
    raw = CATALOG[name].builder(name) if name in STRUCTURAL_ONLY else CATALOG[name].builder()
    wrapped = build_flow(name)
    expected = ("conditional_ledger_required" if name in CONDITIONAL else
                "structural_schema_only" if name in STRUCTURAL_ONLY else
                "reference_algorithm" if name in ("matmul_bf16", "matmul_streamk") else "analytic_prior")
    assert raw.estimate_kind == wrapped.estimate_kind == expected
    assert raw.pipe_work == wrapped.pipe_work and raw.buffers == wrapped.buffers
    if name in CONDITIONAL:
        for key in ("canonical_route", "derivation_coverage", "runtime_confidence"):
            assert wrapped.params[key] == raw.params[key]
    if expected in ("analytic_prior", "reference_algorithm"):
        assert not evaluate_flow(raw, P).calibrated
    else:
        with pytest.raises(ValueError):
            evaluate_flow(raw, P)

@pytest.mark.parametrize("alias", ["flash_kda_fwd", "mega_recurrent_kda"])
def test_conditional_alias_parameters_retain_canonical_coverage(alias):
    for training in (False, True):
        options = dict(batch=2, heads=3, seq=129, training=training)
        canonical = CATALOG["flash_kda"].builder(**options)
        for flow in (CATALOG[alias].builder(**options), build_flow(alias, **options)):
            assert flow.name == alias and flow.params == canonical.params
            assert flow.pipe_work == canonical.pipe_work and flow.buffers == canonical.buffers
            assert flow.estimate_kind == canonical.estimate_kind

ENGLISH_POLARITY = (
    "only", "exclusive", "exclusively", "sole", "solely", "limited to", "restricted to", "specific to",
    "but not", "except", "without", "not", "n't", "never", "no", "neither", "nor", "unsupported", "unavailable",
    "incompatible", "cannot", "can't", "doesn't", "does not", "is not", "aren't", "are not", "fails", "forbidden", "prohibited")
CHINESE_POLARITY = (
    "仅", "只", "唯", "独", "专有", "专属", "限于", "局限于", "不支持", "不可", "不能", "无法",
    "未支持", "不兼容", "禁用", "除外", "而非", "而不是", "但", "却")

@pytest.mark.parametrize("backend", ["ge_graph", "TorchAir", "GE 图", "aclgraph",
                                     "torch.npu.npu_fusion_attention"])
def test_sentence_polarity_is_independent_of_word_order(backend):
    for token in ENGLISH_POLARITY + CHINESE_POLARITY:
        for text in (f"{backend} supports SuperKernel, {token} selected mode.",
                     f"{token} selected mode: SuperKernel uses {backend}.",
                     f"SuperKernel {backend} {token} selected mode。"):
            result = normalize_text(text)
            assert result.text == result.source_text == text
            assert result.preserved_legacy and not result.applied
            assert result.classification == "torchair_legacy_only"

@pytest.mark.parametrize("text", [
    "SuperKernel is supported in ge_graph but not in aclgraph.",
    "ge_graph is the only supported backend for SuperKernel.",
    "SuperKernel 在 ge_graph 中可用，aclgraph 下不支持。",
    "SuperKernel is exclusive to ge_graph.",
    "SuperKernel isn’t supported in aclgraph; ge_graph is available.",
    "TorchAir’s backend can’t execute the selected graph.",
    "torch.npu.npu_fusion_attention is supported only in legacy graph mode.",
    "The graph is unavailable in torchair.scope.npu_stream_switch.",
    "Use ge_graph, but migrate the supported cases to aclgraph.",
])
def test_restrictive_legacy_sentences_remain_verbatim(text):
    result = normalize_text(text)
    assert result.text == text and result.preserved_legacy and not result.applied

@pytest.mark.parametrize("text,expected", [
    ('exe_mode: "ge_graph"', 'exe_mode: "aclgraph"'),
    ("SuperKernel is supported in ge_graph.", "SuperKernel is supported in aclgraph."),
    ("ge_graph supports SuperKernel.", "aclgraph supports SuperKernel."),
    ("SuperKernel 在 ge_graph 中可用。", "SuperKernel 在 aclgraph 中可用。"),
    ("Use torchair.scope.npu_stream_switch.", "Use vllm_ascend.utils.npu_stream_switch."),
    ("No allocation is needed. Use ge_graph.", "No allocation is needed. Use aclgraph."),
    ("Only one buffer is needed。使用 ge_graph。", "Only one buffer is needed。使用 aclgraph。"),
    ("Use ge_graph for the soleus and notional examples.",
     "Use aclgraph for the soleus and notional examples."),
])
def test_positive_normalization_and_unrelated_sentence_scope(text, expected):
    result = normalize_text(text)
    assert result.text == expected and result.source_text == text
    assert result.applied and not result.preserved_legacy

@pytest.mark.parametrize("name,option", [
    ("scatter_elements", "atomic"), ("flash_kda", "training"),
    ("flash_kda_fwd", "training"), ("mega_recurrent_kda", "training"),
    ("engram_gate", "has_mask"), ("layernorm_welford", "two_pass"),
    ("flash_kda", "gqa"), ("flash_kda_fwd", "gqa"),
    ("mega_recurrent_kda", "gqa"), ("moonep", "reduce_grad"),
])
def test_every_declared_builder_boolean_rejects_nonbooleans(name, option):
    for bad in BAD_BOOLS:
        with pytest.raises(ValueError, match=option + " must be"):
            CATALOG[name].builder(**{option: bad})
        with pytest.raises(ValueError, match=option + " must be"):
            build_flow(name, **{option: bad})
        result = opflow_evaluate(name, **{option: bad}, allow_structural_prior=True)
        assert result["ok"] == 0 and result["err"] == "EEVAL"
    for flag in (False, True):
        options = {option: flag, **({"unmasked": 32} if option == "has_mask" else {})}
        assert not CATALOG[name].builder(**options).calibrated

def test_scatter_boolean_default_and_exact_work():
    default = build_flow("scatter_elements")
    assert default == build_flow("scatter_elements", atomic=True)
    for flag, expected in ((False, 6553.6), (True, 26214.4)):
        flow = build_flow("scatter_elements", atomic=flag)
        assert flow.params["atomic"] is flag
        assert next(w.total_work for w in flow.pipe_work if w.pipe == "SIMT") == expected
        result = opflow_evaluate("scatter_elements", atomic=flag)
        assert result["ok"] == 1

@pytest.mark.parametrize("builder,args,option", [
    (_attn, ("attention", 1, 1, 16, 16), "quant"),
    (_matmul, ("matmul", 16, 16, 16), "streamk"),
    (_matmul, ("matmul", 16, 16, 16), "inplace"),
])
def test_exported_builder_helpers_require_boolean_options(builder, args, option):
    for bad in BAD_BOOLS:
        with pytest.raises(ValueError, match=option + " must be a bool"):
            builder(*args, **{option: bad})
    for flag in (False, True):
        assert not builder(*args, **{option: flag}).calibrated

def test_triton_and_launch_boolean_boundaries_reject_truthiness():
    for field in fields(TritonFeatures):
        if isinstance(field.default, bool):
            for bad in BAD_BOOLS:
                with pytest.raises(ValueError):
                    TritonFeatures(**{field.name: bad})
                assert opflow_triton(**{field.name: bad})["ok"] == 0
    for bad in BAD_BOOLS:
        assert aclgraph_launch_floor(superkernel=bad)["ok"] == 0

def event_graph(release, *, observed=False):
    return dict(name="event", event_causality_observed=observed, nodes=[
        dict(id="a", stream_id=0, start_us=0, end_us=10),
        dict(id="b", stream_id=1, start_us=release, end_us=release+10)],
        events=[dict(id="e", record="a", wait="b")])

@pytest.mark.parametrize("release", [11.5, 20, 100])
@pytest.mark.parametrize("observed", [False, True])
def test_nonbinding_event_does_not_override_release_evidence(release, observed):
    graph = event_graph(release, observed=observed)
    actual = diagnose_region(graph)
    base = diagnose_region(dict(graph, events=[]))
    assert actual.wall_us == base.wall_us == release+10
    assert actual.details["starts_us"] == base.details["starts_us"]
    assert actual.details["event_us"] == 0
    assert actual.details["cross_stream_event_delays_us"] == {}
    assert actual.cause == "insufficient_trace_evidence_for_event_attribution"
    assert actual.evidence_grade == "observed"
    envelope = aclgraph_diagnose(graph)
    assert envelope["ok"] == 1 and envelope["grade"] == "observed"

@pytest.mark.parametrize("observed", [False, True])
def test_binding_event_requires_causal_evidence_for_confirmation(observed):
    result = diagnose_region(event_graph(0, observed=observed))
    assert result.wall_us == 21.5 and result.details["event_us"] == 1.5
    assert result.details["cross_stream_event_delays_us"] == {"b": 11.5}
    assert result.cause == "premature_event_record_wait"
    assert result.evidence_grade == ("confirmed" if observed else "supported_hypothesis")

def test_event_binding_off_critical_path_and_independent_dependencies():
    graph = event_graph(0, observed=True)
    graph["nodes"].append(dict(id="long", stream_id=2, start_us=100, end_us=200))
    result = diagnose_region(graph)
    assert result.details["event_us"] == 0  # long release-bound branch hides wall delta
    assert result.details["cross_stream_event_delays_us"] == {"b": 11.5}
    assert result.evidence_grade == "confirmed"
    graph["dependencies"] = [["a", "b"]]
    result = diagnose_region(graph)
    assert result.details["cross_stream_event_delays_us"] == {"b": 1.5}

def test_same_stream_event_cannot_prove_cross_stream_wait():
    graph = dict(name="local", event_causality_observed=True, nodes=[
        dict(id="a", stream_id=0, duration_us=10),
        dict(id="b", stream_id=0, duration_us=10),
        dict(id="c", stream_id=1, start_us=30, end_us=40)],
        events=[dict(id="e", record="a", wait="b")])
    result = diagnose_region(graph)
    assert result.details["cross_stream_event_delays_us"] == {}
    assert result.cause == "insufficient_trace_evidence_for_event_attribution"

def expanded_tail(stages, count, slots, latency, distance):
    ends, starts = {}, {}
    for i in range(count):
        for j, duration in enumerate(stages):
            ready = [Fraction()]
            if i: ready.append(ends[i-1, j])
            if j: ready.append(ends[i, j-1])
            if not j:
                if slots and i >= slots: ready.append(ends[i-slots, len(stages)-1])
                if i >= distance: ready.append(starts[i-distance]+latency)
                starts[i] = max(ready)
            ends[i, j] = max(ready)+duration
    return ends[count-1, len(stages)-1]

@pytest.mark.parametrize("slots", [None, 1, 2, 4])
@pytest.mark.parametrize("count", [1, 2, 7])
@pytest.mark.parametrize("latency,distance", [(0, 1), (7, 2)])
def test_all_tail_only_metadata_uses_present_workload(slots, count, latency, distance):
    for stages in ((Fraction(1),)*3, (Fraction(1, 2), Fraction(2, 3), Fraction(3, 4))):
        first, resource = sum(stages), max(stages)
        cycle = first/slots if slots else Fraction()
        tau = max(resource, cycle, Fraction(latency, distance))
        expected = dict(first_item=first, resource_bound=resource, cycle_ratio=cycle,
                        tau_lower_bound=tau,
                        affine_first_plus_N_minus_one_tau=first+(count-1)*tau,
                        exact_makespan=expanded_tail(stages, count, slots, latency, distance))
        result = pipeline_tail((100, 200, 300), stages, 0, count, slots,
                               recurrence_latency=latency, recurrence_distance=distance)
        assert result == expected
        assert all(isinstance(value, Fraction) for value in result.values())

def test_large_tail_only_fast_paths_and_absent_template_validation():
    count = 10**12
    for slots, expected in ((1, 3*count), (None, count+2)):
        result = pipeline_tail((100,)*3, (1,)*3, 0, count, slots)
        assert result["exact_makespan"] == result["affine_first_plus_N_minus_one_tau"] == expected
        assert result["first_item"] == 3 and result["resource_bound"] == 1
    for full, tail in (((-1,), (1,)), ((1, 2), (1,))):
        with pytest.raises(ValueError):
            pipeline_tail(full, tail, 0, 1)

# FIX5: cover every declared boolean, including directly constructed results.
import ast
import importlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from mindie_knowledge import cannbot_aclgraph_distiller as distiller
from mindie_knowledge import distiller_sources as sources
from mindie_profiler import aclgraph_multistream as multistream
from mindie_sim.opflow import flow_schema, pipeline_evaluator, triton_lowering
from mindie_sim.opflow import account_transfer, TransferRequest
from mindie_sim.opflow.schedule_eval import _gm_bytes, _sram_plans
from mindie_sim.opflow.v4_math import evaluate_math
R4_BAD_BOOLS = (*BAD_BOOLS, 0.0, 1.0, (), set(), object())
R4_POLARITY = (
    "avoid", "avoids", "avoided", "avoiding", "non-functional", "non-support",
    "non-compatible", "non-api_2", "unless", "instead of", "instead\t of", "rather than",
    "do not", "does not", "did not", "cannot", "can't", "can’t", "don't", "don’t",
    "doesn't", "doesn’t", "shouldn't", "shouldn’t", "mustn't", "mustn’t",
    "disallow", "disallows", "disallowed", "disallowing", "deprecated",
    "避免", "不要", "切勿", "请勿", "勿", "禁止", "严禁", "不受支持", "不适合",
    "不宜", "不得", "不应", "除非", "代替", "替代")

@pytest.mark.parametrize("mention", ["ge_graph", "TorchAir", "GE 图", "aclgraph",
    "torch.npu.npu_fusion_attention", "npu_create_tagged_stream", "Notify_Wait", "limit_core_num"])
def test_r4_all_polarity_families_and_normalized_symbols(mention):
    for token in R4_POLARITY:
        for text in (f"{token} {mention} for this selected mode.",
                     f"This mode: {mention}, {token} the alternative.",
                     f"Neutral text; {mention} {token} this mode。"):
            result = normalize_text(text)
            assert result.text == result.source_text == text
            assert result.preserved_legacy and not result.applied
            assert result.classification == "torchair_legacy_only"

@pytest.mark.parametrize("text", [
    "Avoid ge_graph for SuperKernel.", "SuperKernel with ge_graph should be avoided.",
    "The ge_graph path is non-functional.", "The ge_graph backend has non-support for this feature.",
    "Use ge_graph unless the graph is dynamic.", "Unless capture is disabled, avoid ge_graph.",
    "Use aclgraph instead of ge_graph.", "Use aclgraph rather than ge_graph.",
    "避免使用 ge_graph。", "不要使用 ge_graph。", "勿使用 ge_graph。", "禁止使用 ge_graph。",
    "SuperKernel 在 ge_graph 下不受支持。", "ge_graph 不适合此图。", "除非使用 ge_graph 才能运行。",
    "使用 aclgraph 代替 ge_graph。", "应使用 aclgraph 而非 ge_graph。",
    "Run SuperKernel without ge_graph.", "No ge_graph execution is permitted.",
    "SuperKernel is supported in ge_graph but not in aclgraph.", "Never use ge_graph for SuperKernel.",
    "SuperKernel is unsupported in ge_graph.", "ge_graph cannot execute this SuperKernel.",
    "Do not use ge_graph.", "Don't use ge_graph.",
    "ge_graph is the only supported backend for SuperKernel.", "Use every backend except ge_graph.",
    "Neutral statement; avoid ge_graph for SuperKernel.", "中性陈述；避免使用 ge_graph。",
    "Avoid torchair.scope.npu_stream_switch for this graph.",
    "Use npu_stream_switch instead of npu_create_tagged_stream.",
])
def test_r4_natural_prohibitions_and_contrasts(text):
    result = normalize_text(text)
    assert result.text == result.source_text == text
    assert result.preserved_legacy and not result.applied

@pytest.mark.parametrize("text,expected", [
    ("Use ge_graph.", "Use aclgraph."),
    ("No allocation is needed. Use ge_graph.", "No allocation is needed. Use aclgraph."),
    ("Never allocate; use ge_graph.", "Never allocate; use aclgraph."),
    ("Use ge_graph for notional and soleus tests.", "Use aclgraph for notional and soleus tests."),
    ("使用 ge_graph。", "使用 aclgraph。"),
    ("Use torchair.scope.npu_stream_switch.", "Use vllm_ascend.utils.npu_stream_switch."),
])
def test_r4_neutral_scope_controls(text, expected):
    result = normalize_text(text)
    assert result.text == expected and result.source_text == text and not result.preserved_legacy

def _bool_parameters(fn):
    return [p.name for p in inspect.signature(fn).parameters.values() if "bool" in str(p.annotation) or isinstance(p.default, bool)]

@pytest.fixture(scope="module")
def r4_samples():
    flow = build_flow("matmul")
    evaluation = evaluate_flow(flow, P)
    region = multistream.ParallelRegion.from_mapping(event_graph(0))
    lowering = triton_lowering.analyze_triton_lowering(P, aic_us=2, aiv_us=1)
    request = TransferRequest("copy")
    records = [sources.static_ingestion(), normalize_text("Use ge_graph."), region, region.dag,
        region.dag.nodes[0].interval, multistream.check_superkernel_eligibility(True),
        multistream.ACLGraphWrapperModel(), multistream.BreakableACLGraphWrapperModel(),
        multistream.UpdatableGraphModel(), flow.buffers[0], replace(flow, estimate_kind="source_faithful"),
        flow_schema.FixpipeConfig("L0C_TO_GM"), flow_schema.select_fixpipe(P, "L0C_TO_GM"),
        P, CATALOG["matmul"], evaluation, evaluation.tiling, request, account_transfer(request, P),
        TritonFeatures(), lowering.mfu, lowering]
    return dict(flow=flow, evaluation=evaluation, records=records)

def _r4_calls(samples):
    flow, tiling = samples["flow"], samples["evaluation"].tiling
    offline = samples["records"][0]
    fixture = json.loads((Path(__file__).parent / "fixtures/v4_exact_vectors.json").read_text())
    bindings = fixture["operators"]["mhc_hcpost"][0]["bindings"]
    return [(P.ssbuf_bytes, (), {}), (get_profile, ("Ascend950PR",), {}),
        (pipeline_evaluator.mode_launch_floor_us, ("aclgraph_full",), {}),
        (pipeline_evaluator.launch_floor_ordering, (P, flow), {}),
        (flow_schema.require_positive_float, ("x", 1), {}), (build_flow, ("matmul",), {}), (evaluate_flow, (flow, P), {}),
        (sources.scan_live, (), {}), (sources.static_ingestion, (), {}),
        (distiller.distill_live, (), {}), (distiller.distill_units, (offline,), {}),
        (_attn, ("attention", 1, 1, 16, 16), {}), (_matmul, ("matmul", 16, 16, 16), {}),
        (_gm_bytes, (flow, 1, 1), {}), (_sram_plans, (flow, P, tiling), {}),
        (triton_lowering._finite, ("x", 1), {}), (evaluate_math, ("mhc_hcpost", bindings), {}),
        (multistream.check_superkernel_eligibility, (), {}),
        (multistream.effective_launch_floor_us, ("full",), {}), (aclgraph_launch_floor, (), {}),
        (importlib.import_module("mindie_profiler.aclgraph_mcp").aclgraph_superkernel, (), {}),
        (opflow_evaluate, ("matmul",), {}), (opflow_triton, ("matmul",), {})]

@pytest.mark.parametrize("bad", R4_BAD_BOOLS)
def test_r4_every_declared_boolean_parameter(r4_samples, bad):
    for fn, args, defaults in _r4_calls(r4_samples):
        for flag in _bool_parameters(fn):
            options = dict(defaults, **{flag: bad})
            # Required booleans other than the one under test must be valid.
            if fn is multistream.check_superkernel_eligibility:
                options.setdefault("static_decode", True)
            if fn.__module__.endswith("_mcp"):
                result = fn(*args, **options)
                assert result["ok"] == 0 and result["err"] == "EARGS", (fn, flag, result)
            else:
                with pytest.raises((ValueError, sources.DistillError), match=flag):
                    fn(*args, **options)

@pytest.mark.parametrize("bad", R4_BAD_BOOLS)
def test_r4_every_dataclass_boolean_field(r4_samples, bad):
    for record in r4_samples["records"]:
        for f in fields(record):
            if f.type is bool or f.type == "bool":
                with pytest.raises((ValueError, sources.DistillError)):
                    replace(record, **{f.name: bad})
            elif "bool" in str(f.type):
                assert f.name == "gates"
                with pytest.raises(ValueError, match="gates"):
                    replace(record, gates={"input_budget_ok": bad})

def test_r4_boolean_ast_inventory_has_no_untested_public_declarations(r4_samples):
    sim_root = Path(__import__("mindie_sim").__file__).resolve().parents[1]
    prof_root = Path(__import__("mindie_profiler").__file__).resolve().parents[1]
    know_root = Path(__import__("mindie_knowledge").__file__).resolve().parents[1]
    parameters, dataclasses = set(), set()
    runtime = (
        [(sim_root, p) for p in sorted((sim_root / "mindie_sim" / "opflow").glob("*.py"))]
        + [(prof_root, prof_root / "mindie_profiler" / n) for n in ('aclgraph_dag.py', 'aclgraph_mcp.py', 'aclgraph_multistream.py', 'aclgraph_schema.py', 'memory_evidence.py')]
        + [(know_root, know_root / "mindie_knowledge" / n) for n in ('cannbot_aclgraph_distiller.py', 'distiller_clause.py', 'distiller_lexicon.py', 'distiller_sources.py', 'distiller_text.py', 'knowledge_unit.py')]
    )
    assert len(runtime) == 35
    for pkg_root, path in runtime:
        parts = path.relative_to(pkg_root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for node in ast.parse(path.read_text()).body:
            members = node.body if isinstance(node, ast.ClassDef) else [node]
            owner = node.name + "." if isinstance(node, ast.ClassDef) else ""
            for item in members:
                if isinstance(item, ast.AnnAssign) and "bool" in ast.unparse(item.annotation):
                    dataclasses.add((module, owner[:-1], item.target.id))
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = item.args.posonlyargs + item.args.args
                defaults = [None] * (len(args) - len(item.args.defaults)) + item.args.defaults
                for arg, default in zip(args + item.args.kwonlyargs, defaults + item.args.kw_defaults):
                    if (arg.annotation and "bool" in ast.unparse(arg.annotation)) or (
                            isinstance(default, ast.Constant) and isinstance(default.value, bool)):
                        parameters.add((module, owner + item.name, arg.arg))
    exercised = {(fn.__module__, fn.__qualname__, flag)
                 for fn, _, _ in _r4_calls(r4_samples) for flag in _bool_parameters(fn)}
    assert parameters == exercised
    exercised_fields = {(type(r).__module__, type(r).__name__, f.name)
                        for r in r4_samples["records"] for f in fields(r)
                        if f.type is bool or "bool" in str(f.type)}
    assert dataclasses == exercised_fields

def test_r4_actual_boolean_controls_and_omitted_permission(r4_samples, tmp_path):
    for record in r4_samples["records"]:
        for f in fields(record):
            if f.type is bool or f.type == "bool":
                for flag in (False, True):
                    options = {f.name: flag}
                    if isinstance(record, flow_schema.FixpipeConfig):
                        options["path"] = "L0C_TO_UB" if flag else "L0C_TO_GM"
                    assert getattr(replace(record, **options), f.name) is flag
    assert P.ssbuf_bytes() == P.ssbuf_bytes(True) == 3072
    assert P.ssbuf_bytes(False) == 1024
    for flag in (False, True):
        assert get_profile("Ascend950PR", strict=flag) == P
        assert flow_schema.require_positive_float("x", 1, allow_zero=flag) == 1
        assert sources.static_ingestion(flag).degraded_offline is flag
        assert len(distiller.distill_units(r4_samples["records"][0], include_aclgraph=flag)) == (110 if flag else 103)
        assert pipeline_evaluator.mode_launch_floor_us("eager", static_decode=flag,
                                                      superkernel_eligible=flag) == 5
    for name, kind in (("flash_kda", "conditional_ledger_required"),
                       ("attn_prologue", "structural_schema_only"), ("matmul", "analytic_prior")):
        assert build_flow(name).estimate_kind == build_flow(name, allow_structural_prior=True).estimate_kind == kind
        for bad in R4_BAD_BOOLS:
            with pytest.raises(ValueError, match="allow_structural_prior"):
                build_flow(name, allow_structural_prior=bad)
    with pytest.raises(sources.DistillError):
        sources.scan_live(tmp_path, tmp_path, allow_offline=False)
    assert sources.scan_live(tmp_path, tmp_path, allow_offline=True).degraded_offline is True
    assert len(distiller.distill_live(tmp_path, tmp_path, allow_offline=True)) == 110
    for arch in ("Ascend950PR", "Ascend910B"):
        for bad in R4_BAD_BOOLS:
            with pytest.raises(ValueError):
                get_profile(arch).ssbuf_bytes(bad)

def test_r4_ingestion_rejects_invalid_flags_before_io_or_branching(monkeypatch):
    def touched(*args, **kwargs):
        raise AssertionError("invalid boolean reached I/O or delegated work")

    monkeypatch.setattr(sources, "Path", touched)
    monkeypatch.setattr(distiller, "scan_live", touched)
    monkeypatch.setattr(distiller, "_base_units", touched)
    for bad in R4_BAD_BOOLS:
        for fn, options in ((sources.scan_live, {"allow_offline": bad}),
                            (sources.static_ingestion, {"degraded_offline": bad}),
                            (distiller.distill_live, {"allow_offline": bad}),
                            (distiller.distill_units, {"allow_offline": bad}),
                            (distiller.distill_units, {"include_aclgraph": bad})):
            with pytest.raises(sources.DistillError, match="must be bool"):
                fn(**options)

@pytest.mark.parametrize("wrap", ["\n", "\r\n"])
def test_r5_soft_wraps_preserve_complete_restrictive_input(wrap):
    texts = ["Avoid\nge_graph for SuperKernel.", "Use ge_graph\nunless the graph is dynamic.",
        "避免\n使用 ge_graph。", "不要使用\nge_graph。", "ge_graph\n不受支持。",
        "使用 aclgraph 代替\nge_graph。", "Avoid legacy APIs, e.g.\nNotify_Wait.",
        "Use ge_graph. Neutral; avoid\nNotify_Wait in this scope. Use limit_core_num."]
    for phrase in ("instead of", "rather than", "do not", "does not", "did not",
                   "limited to", "restricted to", "specific to"):
        texts.append("Use the selected API " + phrase.replace(" ", "\n") + " ge_graph.")
    for text in texts:
        text = text.replace("\n", wrap)
        result = normalize_text(text)
        assert result.text == result.source_text == text
        assert result.preserved_legacy and not result.applied

@pytest.mark.parametrize("abbreviation", ["e.g.", "i.e.", "vs.", "cf.", "etc.", "fig.",
                                         "no.", "al.", "approx.", "min.", "max."])
def test_r5_abbreviations_keep_restriction_and_api_in_one_clause(abbreviation):
    for abbr in (abbreviation, abbreviation.upper()):
        neutral = f"Avoid tensor.{abbr} Use ge_graph."
        assert normalize_text(neutral).text == neutral.replace("ge_graph", "aclgraph")
        for mention in ("ge_graph", "torchair.scope.npu_stream_switch", "Notify_Wait",
                        "torch.npu.npu_fusion_attention", "npu_create_tagged_stream"):
            label = abbr + (" 2:" if abbreviation == "no." else "")
            text = f"Use ge_graph. Neutral; avoid legacy APIs, {label} {mention}. Use limit_core_num."
            result = normalize_text(text)
            assert result.text == result.source_text == text
            assert result.preserved_legacy and not result.applied

@pytest.mark.parametrize("boundary", [".\n", ";\n", "\n\n", "\r\n \t\r\n",
    "\n- ", "\n* ", "\n+ ", "\n• ", "\n1. ", "\r\n2) "])
def test_r5_paragraph_list_and_sentence_boundaries_remain_independent(boundary):
    text = f"Avoid allocation{boundary}Use ge_graph."
    result = normalize_text(text)
    assert result.text == text.replace("ge_graph", "aclgraph")
    assert result.source_text == text and not result.preserved_legacy

@pytest.mark.parametrize("mention,target,rule", [
    ("ge_graph", "aclgraph", "N7"),
    ("torchair.scope.npu_stream_switch", "vllm_ascend.utils.npu_stream_switch", "N5"),
    ("Notify_Wait", "EventRecord/EventWait (ACLGraph capture order is explicit)", "N4"),
    ("npu_create_tagged_stream", "npu_stream_switch + RoutedMoEMilestones", "N3"),
    ("limit_core_num", "ACLGraph static block dims / HCCL CCU_SCHED", "N6")])
@pytest.mark.parametrize("case", [str.lower, str.upper, str.title])
@pytest.mark.parametrize("wrap", [" ", "\n", "\r\n"])
@pytest.mark.parametrize("round_id", [6, 7])
def test_r6_r7_abbreviation_and_sentence_scope(mention, target, rule, case, wrap, round_id):
    etc, no = case("etc."), case("no")
    boundaries = ("\n\n", "\r\n \t\r\n", "\n- ", "\n* ", "\n+ ", "\n• ", "\n1) ", "\r\n2. ")
    if round_id == 6:
        samples = [(template.format(a=case(abbr), w=wrap, m=mention), preserve)
            for abbr in ("sec.", "ref.", "secs.", "refs.", "figs.", "nos.", "ch.", "eq.", "no.")
            for template, preserve in (("Do not use the backend in {a}{w}2: {m}.", True),
                ("Avoid the legacy API documented in {a}{w}2: {m}.", True),
                ("Use configuration {a}{w}2 with {m}.", False), ("Avoid tensor.{a}{w}Use {m}.", False))]
        samples += [(f"Do not use legacy APIs, {etc}{wrap}{mention}.", True),
            (f"Avoid {mention}, {etc}, in capture.", True), (f"Use {mention}, {etc}{wrap}不受支持。", True),
            (f"Use {mention}, {etc}{wrap}and buffers, {etc}{wrap}不受支持。", True),
            (f"Avoid tensor.{etc}{wrap}Use {mention}.", False), (f"{no} {mention} capture is supported.", True),
            (f"Support for {mention}: {no}.", True), (f"Support for {mention}: {no}.{wrap}Use buffers.", True),
            (f"Use {mention}. Support for buffers: {no}.", False)]
        samples += [(f"Avoid allocations, locks, {etc}{wrap}{s} {mention}。", False)
                    for s in ("Use", "Enable", "Prefer", "Apply", "In", "The", "This", "When", "If", "使用")]
        samples += [(f"Avoid allocations, {etc}{b}{mention}.", False)
                    for b in ("\n\n", "\r\n\r\n", "\n- ", "; ", "。")]
    else:
        counts = ("1 buffer is allocated", "2 buffers are allocated", "12 streams run", "2048 events remain")
        samples = [(f"Support for {mention}{d} {no}.{wrap}{count}.", True) for count in counts
                   for d in (":", "：", "?", "？", "=", "—", "--", ";", "；", ",", "，", " is", " was", " are", " were")]
        samples += [(f"{mention} {no}.{b}{count}.", True) for count in counts for b in (*boundaries, wrap)]
        samples += [(f"Support for {mention}: {no}.{b}2 buffers are allocated.", True) for b in boundaries]
        samples += [(f"Is {mention} supported? {no}.{wrap}1 buffer is allocated.", True),
            (f"{no}.{wrap}2 uses {mention}.", False), (f"{no}.{wrap}2 buffers are allocated for {mention}.", False),
            (f"Use {mention} with configuration {no}.{wrap}2.", False),
            (f"Configuration {no}.{wrap}2 is supported with {mention}.", False),
            (f"Avoid the API in {no}.{wrap}2 with {mention} in use.", True)]
        connectors = ("例如 比如 诸如 如 包括 包含 含 尤其是 尤其 特别是 特别 即 以及 及 和 与 或 或者 还有 乃至 甚至".split()
            + "such as|including|especially|particularly|specifically|namely|for example|for instance|e.g.|i.e.|and|or|nor|as well as".split("|"))
        samples += [(f"Do not use legacy APIs, {etc}{wrap}{case(c).replace(' ', wrap)} {mention}。", True)
                    for c in connectors]
        samples += [(f"Do not use legacy APIs, {etc}{wrap}{c}{mention}。", True) for c in ("如", "如：", "包括", "及")]
        samples += [(f"Do not use legacy APIs, {etc}{b}例如 {mention}。", False) for b in (*boundaries, "; ", "。")]
        samples += [(f"Avoid allocations, locks, {etc}{wrap}{s} {mention}.", False) for s in
            ("CPU kernels use", "GPU kernels use", "API selection uses", "NPU streams use", "HCCL collectives use",
             "A backend uses", "I/O threads use", "DMA transfers use", "BF16 kernels use", "USE", "使用", "如果启用则使用")]
        samples += [(f"Use {mention}, {etc}{wrap}{predicate}", True) for predicate in
                    ("不受支持。", "is unsupported.", "IS unsupported.", "ARE unavailable.", "Unsupported.", "FORBIDDEN.", "Cannot run.")]
        samples += [(f"Do not use legacy APIs, {etc}{wrap}{m}.", True) for m in
                    (mention, "aclgraph", "GE 图", "TorchAir", "torch.npu.npu_fusion_attention")]
        samples += [(f"Avoid tensor.{a}{wrap}Use {mention}.", False) for a in (etc, no + ".", case("sec."))]
    for text, preserve in samples:
        if preserve:
            text = "Use ge_graph. " + text + " Use limit_core_num.\t"
        expected = text if preserve else text.replace(mention, target)
        result = normalize_text(text)
        assert result.text.encode() == expected.encode(), (text, result)
        assert result.source_text.encode() == text.encode()
        assert result.preserved_legacy is preserve and result.applied == (() if preserve else (rule,))
        assert result.classification == ("torchair_legacy_only" if preserve else
                                         "aclgraph_adapted_from_torchair")

@pytest.mark.parametrize("tokens", [0, 1])
@pytest.mark.parametrize("flag", [f.name for f in fields(TritonFeatures)
                                if isinstance(f.default, bool)])
def test_r5_triton_flags_validate_before_noop_in_both_mapping_locations(tokens, flag):
    op = "block_attn_res_update_rms_norm"
    flow = build_flow(op, tokens=tokens)
    for bad in R4_BAD_BOOLS:
        features = {flag: bad}
        stored = replace(flow, triton_features=features)
        for candidate, caller in ((flow, features), (stored, None), (stored, {flag: True})):
            with pytest.raises(triton_lowering.TritonLoweringError, match=flag):
                evaluate_flow(candidate, P, triton_features=caller)
        with pytest.raises(triton_lowering.TritonLoweringError, match=flag):
            evaluate_operator(op, P, tokens=tokens, triton_features=features)
        options = dict(op=op, tokens=tokens, triton_features=features)
        for result in (opflow_evaluate(**options), call_tool("opflow_evaluate", options)):
            assert result["ok"] == 0 and result["err"] == "EEVAL"
        assert opflow_triton(op, **features)["err"] == "ETRITON"
    for value in (False, True):
        expected = evaluate_flow(flow, P, triton_features={flag: value})
        stored = replace(flow, triton_features={flag: not value})
        assert evaluate_flow(stored, P, triton_features={flag: value}) == expected
        assert evaluate_flow(replace(flow, triton_features={flag: value}), P) == expected
        assert expected.launched is (tokens > 0)
        result = opflow_evaluate(op, tokens=tokens, triton_features={flag: value})
        assert result["ok"] == 1
        if tokens == 0:
            assert result["us"] == 0.0 and result["warn"] == ["NOOP"]

@pytest.mark.parametrize("tokens", [0, 1])
def test_r5_triton_mapping_types_fields_and_empty_controls(tokens):
    class HostileTruth:
        def __bool__(self):
            raise AssertionError("mapping validation must precede truth testing")
        __eq__ = __bool__

    flow = build_flow("block_attn_res_update_rms_norm", tokens=tokens)
    for bad in (False, 0, [], (), "", "false", object(), HostileTruth(), TritonFeatures(),
                {"unknown_flag": False}, {"pipe_stages": 0}, {"flops": float("nan")}):
        for candidate, caller in ((flow, bad), (replace(flow, triton_features=bad), None)):
            with pytest.raises(triton_lowering.TritonLoweringError):
                evaluate_flow(candidate, P, triton_features=caller)
    with pytest.raises(triton_lowering.TritonLoweringError):
        evaluate_flow(replace(flow, triton_features=None), P)
    assert evaluate_flow(flow, P) == evaluate_flow(flow, P, triton_features={})
    for features in (None, {}):
        result = opflow_evaluate(flow.name, tokens=tokens, triton_features=features)
        assert result["ok"] == 1
        if tokens == 0:
            assert result["us"] == 0.0 and result["warn"] == ["NOOP"]
    assert opflow_evaluate(flow.name, allow_structural_prior=None)["err"] == "EARGS"
    assert call_tool("opflow_evaluate", [])["err"] == "EARGS"
