"""Round-8 regressions for `astra_review_r8.md` (P1-01/P1-15/P2-19/P2-23).

[Role] Turn every Round-8 counterexample and its structural variant into a
  public launcher, credential-inventory, request-evidence, coordinate and
  bounded-work assertion.  The module also enforces the report-wide source
  span invariant on every document fixture it builds.
[Invariants] YAML availability is frozen by the process's initial import
  environment; no test mutates ``sys.path``.  A completed report must have no
  invented coordinates, a conflicting timestamp must fail closed for TTFT and
  TPOT together, and credential discovery must be complete before admission.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] No file, network or device writes; fixtures are in-memory.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim.feature_reports import (
    Context, SourceIdentity, canonical_json, launch_markdown,
)
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_documents import DocumentInputError, parse_document
from mindie_sim.launch_taint import prescan_document_secrets
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.sol import compare_sol
from mindie_sim.prediction.sol_work import Peak, project_program
import mindie_sim.launch_documents as launch_documents
import mindie_sim.launch_yaml_taint as yaml_taint

try:  # Availability is fixed at collection time for the whole process.
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
VERSION = "vllm-ascend@0.30.0"
CANARY = "R8_CANARY"


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION, **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _canary_free(report):
    assert CANARY not in canonical_json(report)
    assert CANARY not in launch_markdown(report)
    json.loads(canonical_json(report))


def _assert_spans_in_bounds(text, report):
    lines = text.splitlines() or [""]
    for item in _settings(report):
        span = item.span
        assert 1 <= span.line_start <= span.line_end <= len(lines), item.name
        assert 1 <= span.column_start <= len(lines[span.line_start - 1]) + 1, item.name
        assert 1 <= span.column_end <= len(lines[span.line_end - 1]) + 1, item.name


def _plan():
    return project_program(
        KernelProgram((Stage("a", {"hbm": 1}),)),
        (Peak("hbm", "bytes", 1.0, "declared", "fixture"),))


def _metrics(trace, *, phase="decode", commits=2.8):
    plan = _plan()
    context = Context("c8", plan.plan_id, "model_step", phase, 1, 2, 128,
                      "boundary", "complete", (SourceIdentity("r8"),))
    report = compare_sol(
        plan, 2.0, context=context, request_metrics_enabled=True,
        commits=commits, prefill_predicted_ms=2.0, prefill_sol_ms=1.0,
        request_trace=trace)
    return report.request_metrics


YAML_MULTI = (
    "containers:\n"
    "  - name: a\n"
    "    env: {\"API_\\u004bEY\": " + CANARY + "}\n"
    "    command: [vllm, serve, /m]\n"
    "  - name: b\n"
    "    command: [vllm, serve, /m, --served-model-name, " + CANARY + "]\n"
)
YAML_METADATA = (
    "metadata: {\"API_\\u004bEY\": " + CANARY + "}\n"
    "command: [vllm, serve, /m, --served-model-name, " + CANARY + "]\n"
)
YAML_CREDENTIAL = (
    "credential: {value: " + CANARY + "}\n"
    "command: [vllm, serve, /m, --served-model-name, " + CANARY + "]\n"
)


# ---------------------------------------------------------------- P1-01
def test_p1_01_escaped_flow_key_taints_unselected_container():
    assert CANARY in prescan_document_secrets(YAML_MULTI, "yaml")
    report = _parse(YAML_MULTI, format="yaml", container="b")
    if not HAS_YAML:
        assert report.status == "failed"
        assert any(item.code == "yaml_parser_unavailable"
                   for item in report.diagnostics)
        _canary_free(report)
        return
    assert report.status == "completed"
    assert not report.diagnostics
    served = _setting(report, "--served-model-name")
    assert served.redacted is True
    assert served.value == "[REDACTED]"
    _canary_free(report)


def test_p1_01_metadata_and_nested_credential_leaves_are_tainted():
    for label, text in (("metadata", YAML_METADATA),
                        ("credential", YAML_CREDENTIAL)):
        assert CANARY in prescan_document_secrets(text, "yaml"), label
        report = _parse(text, format="yaml")
        _canary_free(report)
        if not HAS_YAML:
            assert report.status == "failed", label
            assert any(item.code == "yaml_parser_unavailable"
                       for item in report.diagnostics), label
            continue
        assert report.status == "completed", label
        assert _setting(report, "--served-model-name").redacted is True, label


def test_p1_01_container_not_found_redacts_canary_and_message():
    text = ("containers:\n"
            "  - name: a\n"
            "    env: {\"API_\\u004bEY\": " + CANARY + "}\n"
            "    command: [vllm, serve, /m]\n")
    report = _parse(text, format="yaml", container=CANARY)
    _canary_free(report)
    if not HAS_YAML:
        assert report.status == "failed"
        return
    assert report.status == "failed"
    diagnostic = next(item for item in report.diagnostics
                      if item.code == "container_not_found")
    assert "container:[REDACTED]" in diagnostic.message
    assert CANARY not in diagnostic.message


def test_p1_01_malformed_json_escapes_fail_structurally_without_nameerror():
    documents = [
        '{"notes":"\\q","command":["vllm","serve","/m"]}',
        '{"notes":"\\uZZZZ","command":["vllm","serve","/m"]}',
        '{"notes":"\\u12',
        '{"notes":"\\',
    ]
    for text in documents:
        assert isinstance(prescan_document_secrets(text, "json"), tuple)
        report = _parse(text, format="json")
        assert report.status == "failed", text
        assert any(item.code == "invalid_json" for item in report.diagnostics), text
        json.loads(canonical_json(report))


def test_p1_01_malformed_yaml_keeps_secret_without_leaking():
    text = ("env: {\"API_\\u004bEY\": " + CANARY + "\n"
            "command: [vllm, serve, /m\n")
    assert CANARY in prescan_document_secrets(text, "yaml")
    report = _parse(text, format="yaml")
    _canary_free(report)
    assert report.status == "failed"


# ---------------------------------------------------------------- P2-19
class _AuditedLines(list):
    suffix_slices = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = len(self) if key.stop is None else key.stop
            if stop - start > 4:
                type(self).suffix_slices += 1
        return super().__getitem__(key)


class _AuditedText(str):
    splitlines_calls = 0
    char_reads = 0
    max_slice = 0

    def splitlines(self, *args, **kwargs):
        type(self).splitlines_calls += 1
        return _AuditedLines(super().splitlines(*args, **kwargs))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = len(self) if key.stop is None else key.stop
            type(self).max_slice = max(type(self).max_slice, stop - start)
        else:
            type(self).char_reads += 1
        return super().__getitem__(key)


def test_p2_19_yaml_scan_is_linear_with_zero_suffix_list_slices(monkeypatch):
    monkeypatch.setattr(yaml_taint, "_event_taint",
                        lambda text, found: None)
    for count in (1000, 2000, 4000):
        _AuditedText.splitlines_calls = 0
        _AuditedText.char_reads = 0
        _AuditedText.max_slice = 0
        _AuditedLines.suffix_slices = 0
        payload = ("notes:\n"
                   + "".join(f"  API_KEY_{index}:\n    - same\n"
                             for index in range(count))
                   + "command: [vllm, serve, /m]\n")
        found = set()
        yaml_taint.yaml_document_taint(_AuditedText(payload), found)
        assert found == {"same"}, count
        assert _AuditedText.splitlines_calls == 0, count
        assert _AuditedLines.suffix_slices == 0, count
        assert _AuditedText.char_reads <= 8 * len(payload), count
        assert _AuditedText.max_slice <= 64, count


def test_p2_19_document_prescan_runs_once(monkeypatch):
    calls = []
    original = launch_documents.prescan_document_secrets

    def counted(text, format="shell"):
        calls.append((format, len(text)))
        return original(text, format)

    monkeypatch.setattr(launch_documents, "prescan_document_secrets", counted)
    json_payload = json.dumps({
        "notes": ["abcdefghij"] * 4000,
        "command": ["vllm", "serve", "/m"],
    })
    document = parse_document(json_payload, format="json")
    assert document.argv == ("vllm", "serve", "/m")
    assert len(calls) == 1

    calls.clear()
    yaml_payload = ("notes:\n"
                    + "".join(f"  API_KEY_{index}:\n    - same\n"
                              for index in range(2000))
                    + "command: [vllm, serve, /m]\n")
    if HAS_YAML:
        document = parse_document(yaml_payload, format="yaml")
        assert document.argv == ("vllm", "serve", "/m")
    else:
        with pytest.raises(DocumentInputError) as caught:
            parse_document(yaml_payload, format="yaml")
        assert caught.value.code == "yaml_parser_unavailable"
    assert len(calls) == 1


# ---------------------------------------------------------------- P1-15
def test_p1_15_conflicting_commit_evidence_fails_ttft_and_tpot():
    trace = {
        "request_id": "rA", "arrival_ms": 0, "first_commit_ms": 10,
        "decode_steps": 1,
        "commit_events": [
            {"request_id": "rA", "time_ms": 10, "commit_ms": 100},
            {"request_id": "rA", "time_ms": 20, "commit_ms": 5},
        ],
    }
    ttft, tpot = _metrics(trace)
    assert ttft.status == "unavailable"
    assert ttft.reason == "conflicting_event_timestamps"
    assert tpot.status == "unavailable"

    other = {
        "request_id": "rA", "arrival_ms": 0, "first_commit_ms": 10,
        "decode_steps": 1,
        "commit_events": [
            {"request_id": "rA", "time_ms": 100},
            {"request_id": "rA", "time_ms": 50},
        ],
        "commit_times": [10, 20],
    }
    ttft, tpot = _metrics(other)
    assert ttft.status == "unavailable"
    assert ttft.reason == "conflicting_event_timestamps"
    assert tpot.status == "unavailable"


def test_p1_15_null_alias_and_full_range_alias_agreement():
    null_alias = {
        "request_id": "rA", "arrival_ms": 0, "first_commit_ms": 10,
        "decode_steps": 1,
        "commit_events": [
            {"request_id": "rA", "time_ms": 10, "commit_ms": None},
            {"request_id": "rA", "time_ms": 20},
        ],
    }
    ttft, tpot = _metrics(null_alias)
    assert ttft.status == "unavailable"
    assert ttft.reason == "conflicting_event_timestamps"
    assert tpot.status == "unavailable"

    tolerance_chain = {
        "request_id": "rA", "arrival_ms": 0, "first_commit_ms": 0,
        "decode_steps": 1,
        "commit_events": [
            {"request_id": "rA", "time_ms": 0, "commit_ms": 0.9e-9,
             "time": 1.8e-9, "at_ms": 2.7e-9, "timestamp_ms": 3.6e-9},
            {"request_id": "rA", "time_ms": 10},
        ],
    }
    ttft, tpot = _metrics(tolerance_chain)
    assert ttft.status == "unavailable"
    assert ttft.reason == "conflicting_event_timestamps"
    assert tpot.status == "unavailable"


def test_p1_15_gate_runs_when_tpot_is_not_requested():
    trace = {
        "request_id": "rA", "arrival_ms": 0, "first_commit_ms": 10,
        "decode_steps": 1,
        "commit_events": [
            {"request_id": "rA", "time_ms": 100},
            {"request_id": "rA", "time_ms": 50},
        ],
    }
    for phase in ("prefill", "decode"):
        ttft, tpot = _metrics(trace, phase=phase, commits=None)
        assert ttft.status == "unavailable", phase
        assert ttft.reason == "conflicting_event_timestamps", phase
        assert tpot.status == "unavailable", phase

    agreeing = {
        "request_id": "rA", "arrival_ms": 0, "first_commit_ms": 10,
        "decode_steps": 1,
        "commit_events": [
            {"request_id": "rA", "time_ms": 10},
            {"request_id": "rA", "time_ms": 20},
        ],
    }
    ttft, tpot = _metrics(agreeing)
    assert ttft.status == "available"
    assert tpot.status == "available"
    assert tpot.predicted_ms == pytest.approx(10.0 / 2.8)


# ---------------------------------------------------------------- P2-23
def test_p2_23_multiline_yaml_json_key_has_exact_end_interval():
    text = ("command: \"vllm serve /m --compilation-config "
            "'{\\\"cudagraph_\\\n  mode\\\":\\\"FULL\\\"}'\"\n")
    report = _parse(text, format="yaml")
    if not HAS_YAML:
        assert report.status == "failed"
        assert any(item.code == "yaml_parser_unavailable"
                   for item in report.diagnostics)
        return
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (1, 49)
    assert (span.line_end, span.column_end) == (2, 9)
    lines = text.splitlines()
    assert len(lines[0]) == 60
    assert 1 <= span.column_end <= len(lines[span.line_end - 1]) + 1


def test_p2_23_shell_in_json_key_has_exact_end_interval():
    quoted = json.dumps({"command": (
        r'vllm serve /m --compilation-config '
        r'"{\"cudagraph_mode\":\"FULL\"}"')}, indent=2)
    report = _parse(quoted, format="json")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (2, 56)
    assert (span.line_end, span.column_end) == (2, 75)
    line = quoted.splitlines()[1]
    assert line[span.column_start - 1] == '"'
    assert line[span.column_end - 2] == '"'


def test_p2_23_root_json_key_never_binds_to_shadowed_nested_key():
    inner = '{"splitting_ops":[],"nested":{"cudagraph_mode":"FULL"},"cudagraph_mode":"NONE"}'
    text = ('{"command":["vllm","serve","/m","--compilation-config",'
            + json.dumps(inner) + ']}')
    report = _parse(text, format="json")
    assert report.status == "partial"
    span = _setting(report, "cudagraph_mode").span
    assert span.line_start == 1
    assert span.column_start == text.rindex("cudagraph_mode")
    assert span.column_start != text.index("cudagraph_mode")
    assert text[span.column_end - 1] == ":"
    assert "NONE" in text[span.column_end:span.column_end + 12]


def test_p2_23_docker_env_script_provenance_is_in_bounds():
    script = ("vllm serve /m --compilation-config "
              "'{\"cudagraph_mode\":\"FULL\"}'")
    documents = [
        json.dumps({
            "env": {"SCRIPT": script},
            "command": 'docker run -e SCRIPT image sh -c "$SCRIPT"',
        }, indent=2),
        json.dumps({
            "env": {"SCRIPT": script},
            "command": ["docker", "run", "-e", "SCRIPT", "image",
                        "sh", "-c", '"$SCRIPT"'],
        }, indent=2),
        json.dumps({"command": (
            'export SCRIPT="' + script.replace('"', '\\"')
            + '"\ndocker run -e SCRIPT image sh -c "$SCRIPT"')}, indent=2),
    ]
    for document in documents:
        report = _parse(document, format="json")
        assert report.status == "completed"
        assert not report.diagnostics
        leaf = _setting(report, "cudagraph_mode").span
        lines = document.splitlines()
        assert 1 <= leaf.line_start <= leaf.line_end <= len(lines)
        assert 1 <= leaf.column_start <= len(lines[leaf.line_start - 1]) + 1
        assert 1 <= leaf.column_end <= len(lines[leaf.line_end - 1]) + 1
        assert "SCRIPT" in lines[leaf.line_start - 1]
        _assert_spans_in_bounds(document, report)

    external_documents = [
        ('docker run -e SCRIPT image sh -c "$SCRIPT"', "shell"),
        (json.dumps({"command": 'docker run -e SCRIPT image sh -c "$SCRIPT"'}),
         "json"),
        (json.dumps({"command": ["docker", "run", "-e", "SCRIPT", "image",
                                 "sh", "-c", '"$SCRIPT"']}), "json"),
    ]
    for text, format in external_documents:
        external = _parse(text, format=format, env={"SCRIPT": script})
        assert external.status == "failed", format
        assert any(item.code == "unmapped_shell_expansion_provenance"
                   for item in external.diagnostics), format


def test_p2_23_every_report_span_stays_inside_its_source_text():
    fixtures = [
        json.dumps({"notes": "decoy", "command": (
            "export VLLM_ALL2ALL_BACKEND=mc2\r\n"
            "vllm serve /m --compilation-config "
            "'{\"cudagraph_mode\":\"FULL\"}'")}, indent=2),
        json.dumps({"command": [
            "docker", "run", "image", "sh", "-c",
            "vllm serve /m --compilation-config "
            "'{\"cudagraph_mode\":\"FULL\"}'"]}, indent=2),
        json.dumps({"env": {"SCRIPT": (
            "vllm serve /m --compilation-config "
            "'{\"cudagraph_mode\":\"FULL\"}'")},
            "command": 'docker run -e SCRIPT image sh -c "$SCRIPT"'},
            indent=2),
        json.dumps({"command": [
            "vllm", "serve", "/m", "--compilation-config",
            '{"cudagraph_mode":"FULL"}']}, indent=2),
    ]
    if HAS_YAML:
        fixtures.extend([
            ('command: "vllm serve /m\n  --compilation-config '
             '\'{\\"cudagraph_mode\\":\\"FULL\\"}\'"\n', "yaml"),
            ('command: |2\n    vllm serve /m --compilation-config '
             '\'{"cudagraph_mode":"FULL"}\'\n', "yaml"),
        ])
    for fixture in fixtures:
        if isinstance(fixture, tuple):
            text, format = fixture
        else:
            text, format = fixture, "json"
        report = _parse(text, format=format)
        _assert_spans_in_bounds(text, report)
