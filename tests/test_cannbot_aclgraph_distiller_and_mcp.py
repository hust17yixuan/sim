"""Tier-0 tests for live 3-stack distillation and bounded MCP envelopes.

[Role]
Verify live ingestion of 53 CANNBot-DSL operators, K3/DS41 groups, 33 Triton
rules, fixpipe evidence, the TorchAir->ACLGraph normalization engine, zero
SKILL.md emission, and <=250-token envelopes across all MCP calls.
[Invariants]
Live tests skip cleanly when the read-only reference corpus is absent; all
offline tests are deterministic. No test writes outside pytest's tmp_path.
[Timeouts & Locks]
Bounded reads of known files only; no sleeps or locks.
[Side-Effect Policy]
Read-only except tmp_path JSONL output.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest; pytest.importorskip("mindie_knowledge")

from mindie_knowledge.cannbot_aclgraph_distiller import (
    NORMALIZATION_RULES,
    STATIC_OPERATORS,
    distill_static,
    distill_units,
    normalize_text,
    scan_live,
    stack_counts,
    static_ingestion,
    write_units,
)
from mindie_knowledge.knowledge_unit import validate_units
from mindie_sim.opflow.opflow_mcp import TOOL_SPECS as OPFLOW_SPECS, call_tool as opflow_call
from mindie_sim.opflow.token_budget import envelope_size
pytest.importorskip("mindie_profiler")
from mindie_profiler.aclgraph_mcp import TOOL_SPECS as ACL_SPECS, call_tool as acl_call

REF_ROOT = Path("/Users/maoxx241/code/refs")
VLLM_ROOT = Path("/Users/maoxx241/code/vllm-ascend")


def test_offline_distillation_counts_and_stacks() -> None:
    units = distill_static()
    assert len(units) == 110
    counts = stack_counts(units)
    assert counts["ascendc"] >= 3
    assert counts["cannbot_dsl"] > 53
    assert counts["triton_ascend"] > 33
    assert counts["framework_aclgraph"] >= 6
    operators = [unit for unit in units if unit.unit_id.startswith("unit-cannbot-dsl-op-")]
    rules = [unit for unit in units if unit.unit_id.startswith("unit-triton-ascend-rule-")]
    gates = [unit for unit in units if unit.unit_id.startswith("unit-triton-ascend-mfu-gate-")]
    assert len(operators) == 53 and len(rules) == 33 and len(gates) == 5
    assert len({unit.unit_id for unit in units}) == len(units)
    assert validate_units(units) == units
    first = units[0]
    assert first.id == first.unit_id and first.tokens == first.token_count
    assert first.source_repo and first.source_commit and first.source_path
    assert "adaptation_notes" in first.provenance


def test_live_ingestion_and_normalization() -> None:
    if not (REF_ROOT / "cannbot-dsl/catalog/operators.json").exists():
        pytest.skip("live cannbot reference corpus unavailable")
    live = scan_live(REF_ROOT, VLLM_ROOT)
    assert len(live.operators) == 53
    assert len(live.k3) == 10 and len(live.ds41) == 10
    assert len(live.triton_rules) == 33
    assert live.has_fixpipe_doc
    assert live.hardware_constants["ub_simd"] == 253952
    assert live.hardware_constants["ub_simt"] == 221184
    assert live.hardware_constants["ssbuf_mix"] == 3072
    assert "ACLGraphWrapper" in live.aclgraph_symbols
    assert "super_kernel_optimize" in live.aclgraph_symbols
    assert "207008" in live.aclgraph_symbols
    assert len(live.knowledge_files) >= 4
    units = distill_units(live)
    assert len(units) == 110
    cards = [unit for unit in units if unit.unit_id.startswith("unit-cannbot-dsl-op-")]
    assert len(cards) == 53
    assert all(any(name in unit.body for unit in cards) for name in live.operators)


def test_normalization_engine_rewrites_torchair_claims() -> None:
    assert len(NORMALIZATION_RULES) == 7
    text = (
        'exe_mode: "ge_graph"; SuperKernel does not support aclgraph; '
        "npu_create_tagged_stream; Notify_Wait; torchair.scope.npu_stream_switch; limit_core_num"
    )
    result = normalize_text(text)
    assert result.preserved_legacy and result.text == text
    result = normalize_text(text.replace("SuperKernel does not support aclgraph; ", ""))
    assert set(result.applied) == {"N1", "N3", "N4", "N5", "N6"}
    assert result.applicability == "aclgraph_adapted_from_torchair"
    assert "ge_graph" not in result.text
    assert "aclgraph" in result.text
    assert result.audit and all("rule" in row for row in result.audit)
    native = normalize_text("ACLGraphWrapper replays a captured NPUGraph")
    assert native.applied == () and native.applicability == "aclgraph_native"
    assert len(STATIC_OPERATORS) == 53


def test_write_units_emits_jsonl_and_zero_skill(tmp_path: Path) -> None:
    target = write_units(distill_static(), tmp_path)
    assert target.name == "units.jsonl"
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 110
    assert not (tmp_path / "SKILL.md").exists()
    assert not any(path.name == "SKILL.md" for path in tmp_path.rglob("SKILL.md"))


def test_mcp_token_budgets_across_all_wrappers() -> None:
    assert len(OPFLOW_SPECS) == 6 and len(ACL_SPECS) == 4
    opflow_calls = [
        ("opflow_catalog", {}),
        ("opflow_hardware", {"name": "Ascend950PR", "sku": "ascend950pr_9589"}),
        ("opflow_hardware", {"name": "Ascend910B"}),
        ("opflow_evaluate", {"op": "block_attn_res_update_rms_norm"}),
        ("opflow_evaluate", {"op": "x" * 5000, "name": "y" * 5000}),
        ("opflow_network_group", {"network": "k3"}),
        ("opflow_network_group", {"network": "z" * 5000}),
        ("opflow_triton", {"op": "flash_attn_score", "has_cv_pingpong": False}),
        ("opflow_transfer", {"kind": "dn2nz", "rows": 128, "cols": 256}),
        ("opflow_transfer", {"kind": "magic", "rows": 1, "cols": 1}),
        ("missing", {}),
    ]
    for name, arguments in opflow_calls:
        result = opflow_call(name, arguments)
        size = envelope_size(result)
        assert size["utf8_bytes"] <= 250, (name, size)
        assert size["token_estimate"] <= 250
        assert result.get("err") != "EBUDGET"
    assert opflow_call("opflow_catalog", {})["count"] >= 24 + 15
    assert opflow_call("opflow_network_group", {"network": "ds41"})["count"] == 10
    acl_calls = [
        ("aclgraph_diagnose", {"region": {"name": "n" * 5000, "branch_a": {"op": "a", "stream_id": 0, "start_us": 0, "end_us": 1}, "branch_b": {"op": "b", "stream_id": 0, "start_us": 0, "end_us": 1}}}),
        ("aclgraph_sq_budget", {"captured_batch_sizes": 8, "subgraphs_per_step": 32, "active_streams_per_graph": 16}),
        ("aclgraph_superkernel", {"static_decode": True}),
        ("aclgraph_launch_floor", {"mode": "full"}),
        ("missing", {}),
    ]
    for name, arguments in acl_calls:
        result = acl_call(name, arguments)
        size = envelope_size(result)
        assert size["utf8_bytes"] <= 250, (name, size)
        assert result.get("err") != "EBUDGET"


def test_static_ingestion_matches_live_cardinality() -> None:
    live = static_ingestion()
    assert live.ok and len(live.operators) == 53
    counts = Counter(unit.authoring_stack for unit in distill_static())
    assert sum(counts.values()) == 110


# FIX9: literal output oracles for semantic polarity and URI preservation.
from mindie_knowledge.distiller_text import normalize_texts

R8_APIS = (
    ("ge_graph", "aclgraph", "N7"),
    ("torchair.scope.npu_stream_switch", "vllm_ascend.utils.npu_stream_switch", "N5"),
    ("Notify_Wait", "EventRecord/EventWait (ACLGraph capture order is explicit)", "N4"),
    ("npu_create_tagged_stream", "npu_stream_switch + RoutedMoEMilestones", "N3"),
    ("limit_core_num", "ACLGraph static block dims / HCCL CCU_SCHED", "N6"),
)


def _r8_result(source, expected, applied=(), preserve=False):
    result = normalize_text(source)
    assert result.text.encode() == expected.encode(), (source, result)
    assert result.source_text.encode() == source.encode()
    assert result.preserved_legacy is preserve and result.applied == applied
    if preserve or applied:
        classification = "torchair_legacy_only" if preserve else "aclgraph_adapted_from_torchair"
        assert result.classification == result.applicability == classification
    assert normalize_texts((source,)) == (result,)
    return result


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.title, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r8_number_labels_use_immediate_context(api, target, rule, case, gap):
    no = case("no.") + gap + "2"
    for prefix in ("For capture, ", "Mode: ", "捕获时，", "注意："):
        for suffix in (f" ({api}).", f": {api}.", f" with {api} in use."):
            text = f"Use ge_graph. {prefix}avoid configuration {no}{suffix} Use limit_core_num.\t"
            _r8_result(text, text, preserve=True)
        for phrase in (f"use {api} with configuration {no}.",
                       f"use configuration {no} with {api}.",
                       f"configuration {no} is supported with {api}."):
            text = prefix + phrase
            _r8_result(text, text.replace(api, target), (rule,))
    for delimiter in (":", "?", "；", "=", "—", ",", " is"):
        for boundary in (gap, "\n\n", "\r\n \t\r\n"):
            text = f"Support for {api}{delimiter} {case('no.')}{boundary}2 buffers are allocated."
            _r8_result(text, text, preserve=True)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.title, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r8_etc_predicates_and_independent_subjects(api, target, rule, case, gap):
    etc = case("etc.")
    for tail in ("never work in capture.", "only work in legacy mode.",
                 "no longer supported.", "still unsupported.", "always fail in capture.",
                 "exclusively run in legacy mode.", "also cannot run."):
        for capitalize in (str.lower, str.capitalize, str.upper):
            text = f"Use ge_graph. Legacy APIs such as {api}, {etc}{gap}"
            text += capitalize(tail).replace(" ", gap) + " Use limit_core_num."
            _r8_result(text, text, preserve=True)
    for subject in ("CPU kernels use", "GPU kernels use", "I/O threads use", "The runtime uses",
                    "2D kernels use", "3D convolutions use",
                    "64-bit pointers use", "16x16 tiles use", "8-bit kernels can use"):
        text = f"Avoid allocations, locks, {etc}{gap}{subject} {api}."
        _r8_result(text, text.replace(api, target), (rule,))
        # A pending API subject does not swallow a new sentence with its own subject.
        text = f"Legacy APIs such as {api}, {etc}{gap}{subject} {api}."
        _r8_result(text, text.replace(api, target), (rule,))
    text = f"Legacy APIs such as {api}, {etc}{gap}Only CPU kernels need buffers."
    _r8_result(text, text.replace(api, target), (rule,))
    text = f"Avoid allocations, {etc}{gap}Only CPU kernels use {api}."
    _r8_result(text, text, preserve=True)
    for tail in (api + ".", "2: " + api + ".", case("no.") + gap + "2: " + api + "."):
        text = f"Do not use legacy APIs, {etc}{gap}{tail}"
        _r8_result(text, text, preserve=True)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.title, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
@pytest.mark.parametrize("emphasis", ["*", "**", "***", "_", "__", "___"])
def test_r8_markdown_negative_answers(api, target, rule, case, gap, emphasis):
    for inner in ("", " ", gap):
        for answer in (emphasis + inner + case("no.") + inner + emphasis,
                       emphasis + inner + case("no") + inner + emphasis + "."):
            for delimiter in ("?", "？", ";", ":"):
                text = f"Use ge_graph. Is {api} supported{delimiter}{gap}{answer}"
                text += gap + "2 buffers are allocated. Use limit_core_num.\t"
                _r8_result(text, text, preserve=True)
            for boundary in ("\n\n", "\r\n \t\r\n", "\n- ", "\n1. "):
                text = f"Is {api} supported?{boundary}{answer} 2 buffers are allocated."
                _r8_result(text, text.replace(api, target), (rule,))


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("ticks", ["`", "``", "```", "`````"])
def test_r8_negative_reply_code_spans_are_literals(api, target, rule, ticks):
    for answer in ("No.", "**No.**", "__NO.__", "***no***.", "_No_."):
        for delimiter in ("?", ":", ";"):
            text = f"Is {api} supported{delimiter} {ticks}{answer}{ticks} 2 buffers are allocated."
            _r8_result(text, text.replace(api, target), (rule,))
        for gap in (" ", "\n", "\r\n"):
            text = f"{ticks}Is {api} supported?{gap}{answer}{ticks}"
            _r8_result(text, text.replace(api, target), (rule,))
        text = f"Is {ticks}{api}{ticks} supported? {answer} 2 buffers are allocated."
        _r8_result(text, text, preserve=True)
    text = f"Avoid {ticks}{api}{ticks}."
    _r8_result(text, text, preserve=True)
    if len(ticks) > 1:
        text = f"Is {api} supported? {ticks}literal ` **No.**{ticks}"
        _r8_result(text, text.replace(api, target), (rule,))
    # Blank paragraphs end inline parsing, even with matching delimiter runs later.
    text = f"{ticks}literal\n\nIs {api} supported? **No.** 2 buffers. {ticks}"
    _r8_result(text, text, preserve=True)


R8_URI_TEMPLATES = (
    "http://{a}.example.test/{a}?backend={a}#{a}",
    "https://[2001:db8::1]:8443/{a}?backend={a}#torchair",
    "https://user:pass@example.test:443/docs/({a})?q=({a})#{a}",
    "mailto:alice@example.test?subject={a}&body={a}",
    "custom+v1.-://server:9000/{a}?root=/{a}#{a}",
    "urn:example:{a}:version1", "x:{a}", "file:///docs/{a}",
    "data:text/plain,{a}", "git+ssh://user@host/{a}",
    "https://example.test/only/no?unsupported={a}#not",
)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("template", R8_URI_TEMPLATES)
@pytest.mark.parametrize("wrapper", ["See {u}.", "See ({u}).", "[guide]({u})",
                                     "[guide](<{u}>)", "<{u}>"])
def test_r8_uris_are_opaque_to_rules_and_polarity(api, target, rule, template, wrapper):
    uri = template.format(a=api)
    link = wrapper.format(u=uri)
    _r8_result(link, link)
    text = f"Use {api}. {link} Use {api}."
    expected = f"Use {target}. {link} Use {target}."
    result = _r8_result(text, expected, (rule,))
    assert result.audit[0]["count"] == "2"
    text = f"Use {api} with {link}"
    _r8_result(text, f"Use {target} with {link}", (rule,))
    text = f"Avoid {api}. {link}"
    _r8_result(text, text, preserve=True)


def test_r8_uri_boundaries_preserve_rule_order_and_audit_counts():
    uri = "https://[::1]:8443/(ge_graph)?x=Notify_Wait#torchair.scope.npu_stream_switch"
    source = f'(ge_graph) [ge_graph]({uri})<{uri}> ge_graph'
    expected = f'(aclgraph) [aclgraph]({uri})<{uri}> aclgraph'
    result = _r8_result(source, expected, ("N7",))
    assert result.audit[0]["count"] == "3"
    # The N1 expression remains prose; encoded URI data stays opaque to N1..N7.
    uri = "urn:example:exe_mode%3A%22ge_graph%22:Notify_Wait:limit_core_num"
    source = f'exe_mode: "ge_graph"; {uri}; Notify_Wait; limit_core_num; ge_graph'
    expected = (f'exe_mode: "aclgraph"; {uri}; EventRecord/EventWait '
                '(ACLGraph capture order is explicit); ACLGraph static block dims / HCCL CCU_SCHED; aclgraph')
    result = _r8_result(source, expected, ("N1", "N4", "N6", "N7"))
    assert [r["count"] for r in result.audit] == ["1"] * 4
    uri = "https://example.test/SuperKernel%20does%20not%20support%20aclgraph"
    _r8_result(f"Use ge_graph ({uri}).", f"Use aclgraph ({uri}).", ("N7",))
    text = f"SuperKernel does not support aclgraph. See {uri}."
    _r8_result(text, text, preserve=True)
    # Sentence-final punctuation stays outside URI polarity masking.
    text = "Avoid allocations https://example.test/docs. Use ge_graph."
    _r8_result(text, text.replace("ge_graph", "aclgraph"), ("N7",))
    for uri in ("https://example.test/docs)ge_graph?x=Notify_Wait",
                "urn:example:ge_graph)torchair.scope.npu_stream_switch",
                "https://example.test/a!$&'()*+,;=ge_graph#Notify_Wait"):
        for link in (uri, "<" + uri + ">"):
            _r8_result(f"Use ge_graph. {link}", f"Use aclgraph. {link}", ("N7",))


# FIX10: compositional grammar regressions, including vocabulary beyond R9.
@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.title, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r9_numbered_nouns_own_relative_and_label_continuations(api, target, rule, case, gap):
    for noun in ("configuration", "config", "option", "item", "case", "mode", "rule", "step",
                 "section", "issue", "entry", "setting", "parameter", "variant", "choice",
                 "reference", "revision"):
        for delimiter in (" ", ": ", " is "):
            label = noun + delimiter + case("no.") + gap + "17"
            for relative, positive in ((f"where {api} is enabled", "where buffers are allocated"),
                                       (f"when {api} is enabled", "when buffers are allocated"),
                                       (f"whose kernels use {api}", "whose buffers are allocated"),
                                       (f"which uses {api}", "which allocates buffers"),
                                       (f"that uses {api}", "that allocates buffers"),
                                       (f"in which {api} is enabled", "in which buffers are allocated")):
                text = f"Use ge_graph. For capture, avoid {label} {relative}. Use limit_core_num.\t"
                _r8_result(text, text, preserve=True)
                text = f"Mode: use {api} with {label} {positive}."
                _r8_result(text, text.replace(api, target), (rule,))
            for tail in (f" ({api}).", f": {api}.", f" with {api} in use."):
                text = f"Avoid {label}{tail}"
                _r8_result(text, text, preserve=True)
    for boundary in ("\n\n", "\n2. ", "\n- "):
        text = f"Avoid configuration {case('no.')}{boundary}17 where {api} is enabled."
        _r8_result(text, text.replace(api, target), (rule,))
    for label in (f"Is {api} supported?", f"Support for {api}:"):
        text = label + gap + case("no.") + gap + "17 buffers are allocated."
        _r8_result(text, text, preserve=True)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r9_pending_predicates_have_modifiers_but_no_subject(api, target, rule, case, gap):
    for tail in ("never really work", "only ever execute", "no longer fully supported",
                 "still completely unsupported", "always immediately fail",
                 "only partially allowed", "never reliably vectorize",
                 "rarely asynchronously dispatch", "seldom successfully enqueue",
                 "not yet entirely compiled", "only strictly serialized",
                 "still currently unavailable", "still generally prohibited",
                 "still unexpectedly rejected", "still totally broken",
                 "no longer remotely portable", "only very occasionally specialize",
                 "only deliberately apply buffers", "never readily comply with capture constraints"):
        text = f"Use ge_graph. Legacy APIs such as {api}, etc.{gap}"
        text += case(tail).replace(" ", gap) + ". Use limit_core_num.\t"
        _r8_result(text, text, preserve=True)
    # Known subjects remain independent even after leading restrictive adverbs.
    for tail in ("only CPU kernels fail", "only the dispatcher stalls", "only they fail",
                 "only 4K tensors fail", "only supported kernels fail",
                 "only 32-bit fused kernel operators fail"):
        text = f"Legacy APIs such as {api}, etc.{gap}{case(tail).replace(' ', gap)}."
        _r8_result(text, text.replace(api, target), (rule,))
    text = f"Legacy APIs such as {api}, etc.{gap}{case('always fully execute')}."
    _r8_result(text, text.replace(api, target), (rule,))


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.title, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r9_technical_subjects_do_not_depend_on_verb_whitelists(api, target, rule, case, gap):
    for subject in ("2D kernels", "3D convolutions", "64-bit pointers", "16x16 tiles",
                    "4K tensors", "FP16 buffers", "INT8 streams", "32-bit operators",
                    "8-bit ops", "2 layers", "FP32 matrices", "8x8 inputs", "4 outputs",
                    "BF16 graphs", "1D modes", "INT4 paths", "2D executions",
                    "3 launches", "32x8 fused kernels", "12-bit kernel operators"):
        for predicate in ("execute using", "remain valid with", "rely on", "adopt",
                          "enable", "call", "invoke", "interoperate with",
                          "dispatch through", "can independently specialize for", "comply with"):
            clause = case(f"{subject} {predicate}").replace(" ", gap)
            text = f"Avoid allocations, locks, etc.{gap}{clause} {api}."
            _r8_result(text, text.replace(api, target), (rule,))
    for tail in (api, "17: " + api, "2D kernels, " + api, "FP16 buffers with " + api):
        text = f"Avoid legacy APIs, etc.{gap}{tail}."
        _r8_result(text, text, preserve=True)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("ticks", ["`", "````", "`````", "```````"])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r9_code_masks_polarity_and_internal_sentence_marks(api, target, rule, ticks, gap):
    for payload in ("No", "NO", "no", "*No*", "**No**", "_No_", "Avoid", "unsupported",
                    "never allowed", "only", "不支持"):
        for period in ("", "."):
            for delimiter in (":", "=", " is"):
                text = f"Support for {api}{delimiter}{gap}{ticks}{payload}{period}{ticks}."
                _r8_result(text, text.replace(api, target), (rule,))
    text = f"{ticks}Avoid{gap}{api}. Never supported.{ticks}"
    _r8_result(text, text.replace(api, target), (rule,))
    text = f"Avoid {ticks}{api}. No.{ticks}."
    _r8_result(text, text, preserve=True)
    for punctuation in ("?", "!", ".", ";"):
        text = f"Avoid the literal {ticks}No{punctuation}{ticks}: {api}."
        _r8_result(text, text, preserve=True)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("tail", ["?", "!", ".", ";", ",", ":"])
@pytest.mark.parametrize("wrapper", ["<{u}>", "[guide]({u})", "[nested [guide]](<{u}>)",
                                     "[nested [guide]]({u})", "[soft\n wrap]({u})",
                                     "[escaped \\] label]({u})", "[even \\\\[nested]]({u})"])
def test_r9_explicit_uri_endpoints_keep_internal_punctuation(api, target, rule, tail, wrapper):
    for template in R8_URI_TEMPLATES:
        uri = template.format(a=api) + tail
        link = wrapper.format(u=uri)
        text = f"Use ge_graph. Avoid the API documented at {link}: {api}. Use limit_core_num."
        _r8_result(text, text, preserve=True)
        text = f"Use {api} with {link}."
        result = _r8_result(text, f"Use {target} with {link}.", (rule,))
        assert [r["count"] for r in result.audit] == ["1"]
        for outside in (".", "?", "!", ";"):
            text = f"Avoid allocations {link}{outside} Use {api}."
            _r8_result(text, text.replace(f"Use {api}", f"Use {target}"), (rule,))
    text = f"Avoid allocations https://example.test/docs?backend={api}. Use {api}."
    _r8_result(text, text.replace(f"Use {api}", f"Use {target}"), (rule,))


# FIX11: pending predicates, object qualifiers and adverb-led subject contrasts.
from itertools import product


def _r10_results(rows, api, target, rule):
    singles = []
    for source, preserve in rows:
        expected = source if preserve else source.replace(api, target)
        result = _r8_result(source, expected, () if preserve else (rule,), preserve)
        assert [(a["rule"], a["count"]) for a in result.audit] == [
            ("LEGACY", "1") if preserve else (rule, str(source.count(api)))]
        singles.append(result)
    assert normalize_texts(source for source, _ in rows) == tuple(singles)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r10_bare_predicates_cannot_be_technical_np_modifiers(api, target, rule, case, gap):
    rows = []
    for (verb, head), specifier, ending in product(
        (("run", "kernels"), ("support", "kernels"), ("use", "buffers"),
         ("call", "operators"), ("enable", "streams"), ("compile", "graphs")),
        ("", "2D ", "FP16 "),
        ("quickly.", "directly, and retain capture.", "safely;", "only.",
         "compiled for capture.", "used in decode.", "captured by runtime."),
    ):
        tail = case(f"never {verb} {specifier}{head} {ending}").replace(" ", gap)
        text = f"Use ge_graph. Legacy APIs such as {api}, etc.{gap}{tail} Use limit_core_num."
        rows.append((text, True))
    for modifier in ("only", "never", "always", "directly"):
        tail = case(f"{modifier} support kernels compiled for capture.").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}", modifier in ("only", "never")))
    _r10_results(rows, api, target, rule)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r10_ly_predicates_keep_determined_and_qualified_objects(api, target, rule, case, gap):
    rows = []
    for (verb, head), determiner, specifier, ending in product(
        (("apply", "buffers"), ("multiply", "tensors"), ("supply", "inputs"), ("imply", "graphs")),
        ("the", "these", "our", "their"), ("", "2D ", "BF16 "),
        ("", " quickly", " compiled for capture", " only"),
    ):
        predicate = f"{verb} {determiner} {specifier}{head}"
        tail = case("never " + predicate + ending).replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}.", True))
        clause = case("2D kernels " + predicate + " via").replace(" ", gap)
        suffix = case(ending).replace(" ", gap)
        rows.append((f"Avoid allocations, locks, etc.{gap}{clause} {api}{suffix}.", ending == " only"))
    for predicate in ("rely on the 2D buffers", "comply with these FP16 operators"):
        tail = case("never " + predicate + " quickly").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}.", True))
    _r10_results(rows, api, target, rule)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r10_productive_adverbs_retain_independent_subjects(api, target, rule, case, gap):
    rows = []
    for modifier, subject, (predicate, ending) in product(
        ("typically", "normally", "explicitly", "only"),
        ("kernels", "2D kernels", "the custom kernels", "these 2D operators", "our FP16 kernels",
         "NPU kernels", "I/O threads", "32-bit fused kernel operators", "compiled kernels"),
        (("run", " quickly"), ("apply these 2D buffers", " quickly"),
         ("supply their FP16 inputs", ""), ("multiply our tensors compiled for capture", "")),
    ):
        tail = case(f"{modifier} {subject} {predicate}").replace(" ", gap)
        suffix = case(ending).replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}{suffix}.", False))
        rows.append((f"Avoid allocations, locks, etc.{gap}{tail} via {api}{suffix}.", modifier == "only"))
    _r10_results(rows, api, target, rule)


# FIX12: conjunctions are not verb complements; finite past is not passive.
@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r11_coordinated_ly_predicates_and_terminal_adverbs(api, target, rule, case, gap):
    rows = []
    for verb, adv, connector in product(
        ("apply", "reapply", "resupply", "premultiply"), ("quickly", "rapidly", "normally"),
        ("and", "or", "but", "nor", "yet", "while", "because", "if", "unless", "as"),
    ):
        ending = (f"{connector} retain capture" if connector in ("and", "or", "but", "nor", "yet")
                  else f"{connector} the streams run")
        tail = case(f"never {verb} the buffers {adv} {ending}.").replace(" ", gap)
        rows.append((f"Use ge_graph. Legacy APIs such as {api}, etc.{gap}{tail} Use limit_core_num.", True))
        tail = case(f"only the kernels {verb} the buffers {adv} {ending}.").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}", False))
        if connector in ("and", "or", "yet"):
            tail = case(f"2D kernels {verb} the buffers {adv} {ending} via").replace(" ", gap)
            rows.append((f"Avoid allocations, locks, etc.{gap}{tail} {api}.", False))
    for predicate, connector in product(
        ("apply", "reapply quickly", "resupply", "premultiply", "rely on buffers",
         "comply with streams", "reply to operators"), ("and", "or", "yet"),
    ):
        for prefix, ending in (("never", "retain capture"), ("always", "never allocate tensors")):
            tail = case(f"{prefix} {predicate} {connector} {ending}.").replace(" ", gap)
            rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}", True))
        tail = case(f"2D kernels {predicate} {connector} retain capture via").replace(" ", gap)
        rows.append((f"Avoid allocations, locks, etc.{gap}{tail} {api}.", False))
    for connector in ("and", "or"):
        rows.append((f"Legacy APIs such as {api}, etc. Never rely on {api} {connector} allocate buffers.", True))
    _r10_results(rows, api, target, rule)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r11_sentence_adverbs_keep_finite_past_and_present_clauses(api, target, rule, case, gap):
    rows = []
    for adverb, predicate in product(
        ("normally", "typically", "usually", "locally", "automatically", "manually",
         "actively", "cautiously", "independently", "constantly", "repeatedly", "seemingly",
         "reliably", "visibly", "outwardly", "unwisely", "quickly", "explicitly", "simply"),
        ("applied these buffers", "multiplied 2D tensors", "supplied capture streams",
         "reapplied our buffers", "resupplied the inputs", "premultiplied matrices",
         "relied on these buffers", "complied with these streams", "replied to these operators",
         "apply these buffers", "reapply the buffers", "have supplied inputs", "are applying buffers"),
    ):
        tail = case(f"{adverb} the 2D kernels {predicate}").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail} only.", False))
        rows.append((f"Avoid allocations, locks, etc.{gap}{tail} via {api}.", False))
    for predicate in ("applies these buffers", "reapplies inputs", "resupplies streams",
                      "relies on buffers", "replies to operators", "complies with streams",
                      "applied buffers", "relied on streams", "applying buffers", "relying on streams"):
        tail = case(predicate).replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail} only.", True))
        subject = "the kernel is " if "lying" in predicate else "the kernel "
        clause = case("quickly " + subject + predicate).replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{clause} only.", False))
    _r10_results(rows, api, target, rule)


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r11_passive_and_active_object_qualifiers(api, target, rule, case, gap):
    rows = []
    for verb, qualifier in product(
        ("apply", "reapply", "resupply", "premultiply"),
        ("allocated via capture", "pinned on NPU", "pinned to streams", "compiled in capture",
         "supplied by runtime", "reapplied for capture", "chosen", "frozen safely",
         "using streams", "applying inputs", "supplying streams", "relying on streams"),
    ):
        tail = case(f"never {verb} the buffers {qualifier}.").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}", True))
        tail = case(f"2D kernels {verb} the buffers {qualifier} via").replace(" ", gap)
        rows.append((f"Avoid allocations, locks, etc.{gap}{tail} {api}.", False))
    for prep in ("on", "with", "to", "into", "in", "for", "from", "by", "at", "upon", "via",
                 "through", "over", "under", "across", "within", "without"):
        tail = case(f"never apply {prep} the buffers and retain capture.").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{gap}{tail}", True))
    _r10_results(rows, api, target, rule)


# FIX13: all five R12 phenomena compose without transferring NP ownership.
@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r12_modifiers_qualifiers_and_coordination_compose(api, target, rule, case, gap):
    rows = []
    def add(tail, preserve, prefix="Legacy APIs such as {a}, etc."):
        rows.append((prefix.format(a=api) + gap + case(tail).replace(" ", gap), preserve))
    for sentence, adv, focus, qualifier in product(
        ("perhaps", "maybe", "otherwise", "meanwhile", "however", "indeed"),
        ("simply", "amply", "deeply", "sharply", "triply"),
        ("only", "directly", "merely"), ("now supplied by runtime", "already applying inputs"),
    ):
        obj = f"{focus} the buffers {qualifier} {adv} yet retain capture."
        add(f"{sentence} never apply {obj}", True)
        add(f"{sentence} the kernels applied {obj}", False)
        add(f"{sentence} always supply {obj.replace('retain capture', 'never allocate tensors')}", True)
    for verb, ending in product(
        ("apply", "supply", "reply", "multiply", "rely", "comply", "imply", "fly", "ally",
         "bully", "rally", "tally", "dally", "sally", "ply", "reapply", "misapply", "overapply",
         "underapply", "resupply", "oversupply", "undersupply", "premultiply", "overfly",
         "outfly", "noncomply"), (".", " and retain capture.", " yet retain capture."),
    ):
        add(f"never {verb} simply{ending}", True)
        add(f"only the kernels {verb} amply{ending}", False)
    for adv, connector in product(("steeply", "cheaply", "crisply"),
                                 ("and", "or", "but", "nor", "yet", ", and", ", yet")):
        add(f"never apply the buffers {adv} {connector} retain capture.", True)
    for modifier, predicate in product(("quickly", "briefly", "simply"),
        ("applied only these buffers", "reply only to the operators", "replied directly to operators",
         "worked", "worked just in capture", "have replied merely to the operators")):
        add(f"{modifier} the kernels {predicate} only.", False)
    for qualifier in ("not yet initialized in capture", "currently frozen", "previously chosen",
                      "still applying inputs", "quickly supplied by runtime"):
        add(f"never apply the buffers {qualifier} yet retain capture.", True)
    _r10_results(rows, api, target, rule)


# FIX14: lexical verbs establish object ownership before suffix analysis.
@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper, str.title])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r13_clause_ownership_composes_with_recipient_passives(api, target, rule, case, gap):
    rows = []
    def add(tail, preserve, prefix="Legacy APIs such as {a}, etc."):
        rows.append((prefix.format(a=api) + gap + case(tail).replace(" ", gap), preserve))
    for participle, modifier, focus, adjunct in product(
        ("given", "assigned", "allocated", "offered", "sent", "taught", "told", "lent"),
        ("already", "very recently"), ("only", "just", "merely"),
        ("", " by the scheduler", " in capture mode"),
    ):
        obj = f"the operators {modifier} {participle} {focus} these buffers{adjunct}"
        add(f"indeed, never, ever supply {obj} so they remain idle.", True)
        add(f"but fortunately, perhaps, the kernels supply {obj} so they remain idle.", False)
        add(f"for the kernels were {modifier} {participle} {focus} these buffers{adjunct}.", False)
    for coordinator, predicate in product(
        ("and", "or", "nor", "but", "for", "yet", "so"),
        ("applied only these buffers", "assigned only these buffers to the stream",
         "assigned only these buffers in capture", "allocated only these buffers in capture mode",
         "have given only these buffers to the stream", "were given only these buffers"),
    ):
        add(f"{coordinator} fortunately, perhaps, the kernels {predicate}.", False)
    for lead, degree, verb in product(
        ("for the kernels in capture mode,", "for the kernels assigned buffers by the scheduler,",
         "for the kernels allocated buffers in capture mode,", "in capture mode, indeed,",
         "fortunately, perhaps,", "never, ever,"),
        ("so directly", "so fast", "so rarely"), ("apply", "resupply", "premultiply"),
    ):
        add(f"{lead} never {degree} {verb} the buffers so they remain idle.", True)
        add(f"{lead} the kernels {degree} {verb} the buffers only.", False)
    for participle in ("taken", "chosen", "written", "driven", "hidden", "broken",
                       "spoken", "stolen", "frozen", "proven", "shown", "known", "seen",
                       "done"):
        add(f"indeed, never apply the buffers already {participle} in capture.", True)
        add(f"perhaps, the kernels have {participle} only these buffers.", False)
    for auxiliary in ("do", "can", "should"):
        add(f"nor {auxiliary} the kernels apply buffers only.", False)
        text = "Avoid allocations, locks, etc." + gap
        text += case(f"nor {auxiliary} the kernels apply buffers via").replace(" ", gap)
        rows.append((text + " " + api + ".", True))
    for predicate in ("have been idle", "have gone offline"):
        add(f"perhaps, the kernels {predicate} only in capture.", False)
    for head in ("for legacy buffers via", "for the kernels in capture mode via",
                 "the kernels given buffers via", "and the kernels given buffers via"):
        rows.append(("Avoid allocations, locks, etc." + gap +
                     case(head).replace(" ", gap) + " " + api + ".", True))
    for boundary in ("; ", "! ", "\n\n", "\r\n \r\n", "\n- ", "\n2. "):
        tail = case("indeed, never supply kernels assigned only these buffers.").replace(" ", gap)
        rows.append((f"Legacy APIs such as {api}, etc.{boundary}{tail}", False))
        text = "Avoid allocations, locks, etc." + boundary
        text += case("fortunately, perhaps, 2D kernels supply the operators assigned just buffers so they remain idle via").replace(" ", gap)
        rows.append((text + " " + api + ".", False))
    _r10_results(rows, api, target, rule)
    # Literal masks cannot contribute polarity or consume a clause boundary.
    for ticks, lead, preserve in product(("`", "````"),
                                       ("indeed, never", "but perhaps, the kernels"), (True, False)):
        uri = f"https://[::1]:8443/{api}?q=only#never"
        body = f"Legacy APIs such as {ticks}{{a}}{ticks}, etc. "
        body += case(f"{lead} supply the operators given just buffers so they remain idle.").replace(" ", gap)
        if not preserve:
            body = ticks + "Never, ever supply {a} so they remain idle!" + ticks
        expected_preservation = preserve and "never" in lead
        source = body.format(a=api) + f" [guide]({uri}) <{uri}>"
        expected = source if expected_preservation else body.format(a=target) + f" [guide]({uri}) <{uri}>"
        result = _r8_result(source, expected, () if expected_preservation else (rule,), expected_preservation)
        assert result.text.count(uri) == 2
        assert [(a["rule"], a["count"]) for a in result.audit] == [
            ("LEGACY", "1") if expected_preservation else (rule, "1")]


# Compositional constituent boundaries generalize across unseen vocabulary and case.
@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize(
    "case",
    [
        str.lower,
        str.capitalize,
        str.upper,
        lambda s: "".join(c.lower() if i % 2 == 0 else c.upper() for i, c in enumerate(s)),
    ],
)
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r14_compositional_constituents_and_case_invariance(api, target, rule, case, gap):
    rows = []

    def add(tail, preserve, prefix="Legacy APIs such as {a}, etc."):
        rows.append((prefix.format(a=api) + gap + case(tail).replace(" ", gap), preserve))

    for part, agent in product(
        ("assigned", "loaned", "leased", "rented", "delegated", "initialized"),
        ("", " by the dispatcher", " during replay"),
    ):
        theme = "" if part == "initialized" else " just these buffers"
        add(f"for the kernels {part}{theme}{agent}, never call aclmdlRICaptureBegin.", True)
        add(f"for the kernels {part}{theme}{agent}, the streams assign only local buffers.", False)
    add("for the kernels entrusted with just these buffers by the runtime, never call aclmdlRICaptureBegin.", True)
    for noun, qual in (
        ("threads", "woken by the scheduler"),
        ("kernels", "forgiven by the scheduler"),
        ("workflows", "undertaken during setup"),
        ("streams", "withdrawn from capture"),
        ("launches", "foreseen by the compiler"),
        ("operators", "spoken of during setup"),
        ("kernels", "proven correct"),
        ("kernels", "in capture mode forgiven by the scheduler"),
        ("kernels", "chosen for replay entrusted with these buffers by the runtime"),
    ):
        rows.append(("Avoid allocations, locks, etc." + gap + case(f"the {noun} {qual} via").replace(" ", gap) + f" {api}.", True))
        rows.append(("Avoid allocations, locks, etc." + gap + case(f"2D {noun} {qual} interoperate with").replace(" ", gap) + f" {api}.", False))
        add(f"the {noun} {qual} never dispatch asynchronously.", False)
    for degree, verb in product(
        ("so very cautiously", "so remarkably very carefully", "so extraordinarily fast", "so fairly early"),
        ("apply", "resupply", "premultiply"),
    ):
        add(f"never {degree} {verb} the buffers so they remain idle.", True)
        add(f"the kernels {degree} {verb} the buffers only.", False)
    for coord, chain in product(
        ("but", "for", "yet", "so", "and", "or", "nor"),
        ("surprisingly,", "quite frankly,", "in capture mode, admittedly,"),
    ):
        add(f"{coord}, {chain} the kernels leased only these buffers to the allocator.", False)
    for aux in ("do", "should", "would", "must", "could"):
        add(f"nor, indeed, {aux} the kernels dispatch only these buffers.", False)
        rows.append(("Avoid allocations, locks, etc." + gap + case(f"nor, indeed, {aux} the kernels dispatch through").replace(" ", gap) + f" {api}.", True))
    _r10_results(rows, api, target, rule)
    for ticks, (clause, preserve) in product(
        ("`", "````"),
        (
            ("for the kernels loaned just these buffers, never call aclmdlRICaptureBegin.", True),
            ("yet, quite frankly, the kernels leased only these buffers to the allocator.", False),
        ),
    ):
        uri = f"https://[::1]:8443/{api}?mode=only#never"
        body = f"Legacy APIs such as {ticks}{{a}}{ticks}, etc." + gap + case(clause).replace(" ", gap)
        source = body.format(a=api) + f" [guide]({uri}) <{uri}>"
        expected = source if preserve else body.format(a=target) + f" [guide]({uri}) <{uri}>"
        res = _r8_result(source, expected, () if preserve else (rule,), preserve)
        assert res.text.count(uri) == 2


@pytest.mark.parametrize("api,target,rule", R8_APIS)
@pytest.mark.parametrize("case", [str.lower, str.capitalize, str.upper, lambda s: "".join(c.lower() if i % 2 == 0 else c.upper() for i, c in enumerate(s))])
@pytest.mark.parametrize("gap", [" ", "\n", "\r\n"])
def test_r15_compound_nominals_and_case_invariance(api, target, rule, case, gap):
    lead_l, lead_a, uri = f"Legacy APIs such as {api}, etc.{gap}", f"Avoid allocations, locks, etc.{gap}", f"https://[::1]:8443/({api})?q=only#never"
    rows = [(lead_l + case(t).replace(" ", gap), p) for t, p in (
        ("for the completion queues drained by the host firmware, never call aclmdlRICaptureBegin.", True), ("descriptors never advance.", False),
        ("for the doorbell registers drained by the device driver, the controller retires only local pages.", False), ("semaphores chosen for replay never advance.", False),
        ("the descriptors in the ring overwritten by the host firmware never advance.", False), ("the descriptors in our routing chosen by the device driver never advance.", False),
        ("the descriptors so really slowly retire only these pages.", False), ("never so really quite efficiently invalidate the pages so they remain resident.", True),
        ("but, as documented, the descriptors retire only these pages.", False), ("nor, once armed, does the host runtime retire only local pages.", False),
    )] + [(lead_a + case(h).replace(" ", gap) + f" {api}.", p) for h, p in (
        ("the descriptors invalidated by the host firmware via", True), ("the controllers polling the completion queues via", True), ("prefer", False),
        ("the completion queues invalidated by the device driver retire pages via", False), ("the descriptors in the ring advance via", False), ("nor, as expected, do the descriptors retire pages via", True))]
    _r10_results(rows, api, target, rule)
    assert _r8_result(f"Legacy APIs such as `{api}`, etc.{gap}" + case("but, as documented, the descriptors retire only these pages.").replace(" ", gap) + f" [u]({uri})", f"Legacy APIs such as `{target}`, etc.{gap}" + case("but, as documented, the descriptors retire only these pages.").replace(" ", gap) + f" [u]({uri})", (rule,), False).text.count(uri) == 1
