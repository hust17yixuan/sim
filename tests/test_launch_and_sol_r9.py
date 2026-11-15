"""Round-9 regressions for `astra_review_r9.md` (P1-01/P2-19/P2-23).

[Role] Turn every Round-9 counterexample into a public launcher,
  credential-inventory, linear-work and exact-interval assertion, preserving
  the Round-8 literals at the same time.
[Invariants] Credential discovery is document-wide across normalized env
  scalars and embedded shell/argv declarations; raw YAML folding matches the
  loader; whitespace runs are consumed once; a half-open interval closes at
  the last key character's end; Docker expansion carries the exact variable's
  provenance and fails closed when that provenance is absent.
[Timeouts & Locks] Entire module stays under ten seconds; no locks.
[Side-Effect Policy] No file, network or device writes; fixtures are in-memory.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_coords import Lines
from mindie_sim.launch_taint import prescan_document_secrets
from mindie_sim.launch_yaml_coords import yaml_scalar_coords
import mindie_sim.launch_docker as launch_docker
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
CANARY = "R9_CANARY"
NUMERIC = 8675309123
SCRIPT = ('vllm serve /m --compilation-config '
          "'{\"cudagraph_mode\":\"FULL\"}'")


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION, **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _leaks(report, needle):
    return needle in canonical_json(report) or needle in launch_markdown(report)


def _yaml_unavailable(report):
    return any(item.code == "yaml_parser_unavailable"
               for item in report.diagnostics)


# ---------------------------------------------------------------- P1-01
@pytest.mark.parametrize("format", ["json", "yaml"])
def test_p1_01_numeric_env_scalar_is_a_document_credential(format):
    if format == "json":
        text = json.dumps({"containers": [
            {"name": "a", "env": {"API_KEY": NUMERIC},
             "command": ["vllm", "serve", "/m"]},
            {"name": "b", "command": ["vllm", "serve", "/m",
                                      "--served-model-name", str(NUMERIC)]},
        ]})
    else:
        text = ("containers:\n"
                "  - name: a\n"
                "    env: {API_KEY: " + str(NUMERIC) + "}\n"
                "    command: [vllm, serve, /m]\n"
                "  - name: b\n"
                "    command: [vllm, serve, /m, --served-model-name, \""
                + str(NUMERIC) + "\"]\n")
    assert str(NUMERIC) in prescan_document_secrets(text, format)
    report = _parse(text, format=format, container="b")
    assert not _leaks(report, str(NUMERIC))
    if format == "yaml" and not HAS_YAML:
        assert report.status == "failed" and _yaml_unavailable(report)
        return
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").redacted is True


@pytest.mark.parametrize("format", ["json", "yaml"])
def test_p1_01_unselected_command_string_credentials_are_discovered(format):
    if format == "json":
        text = json.dumps({"containers": [
            {"name": "a", "command": "API_KEY=" + CANARY + " vllm serve /m"},
            {"name": "b", "command": ["vllm", "serve", "/m",
                                      "--served-model-name", CANARY]},
        ]})
    else:
        text = ("containers:\n"
                "  - name: a\n"
                "    command: API_KEY=" + CANARY + " vllm serve /m\n"
                "  - name: b\n"
                "    command: [vllm, serve, /m, --served-model-name, "
                + CANARY + "]\n")
    assert CANARY in prescan_document_secrets(text, format)
    report = _parse(text, format=format, container="b")
    assert not _leaks(report, CANARY)
    if format == "yaml" and not HAS_YAML:
        assert report.status == "failed" and _yaml_unavailable(report)
        return
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").redacted is True


@pytest.mark.parametrize("format", ["json", "yaml"])
def test_p1_01_container_not_found_redacts_embedded_command_secret(format):
    if format == "json":
        text = json.dumps({"containers": [
            {"name": "a", "command": "API_KEY=" + CANARY + " vllm serve /m"},
        ]})
    else:
        text = ("containers:\n"
                "  - name: a\n"
                "    command: API_KEY=" + CANARY + " vllm serve /m\n")
    assert CANARY in prescan_document_secrets(text, format)
    report = _parse(text, format=format, container=CANARY)
    assert report.status == "failed"
    assert not _leaks(report, CANARY)
    if HAS_YAML or format == "json":
        diagnostic = next(item for item in report.diagnostics
                          if item.code == "container_not_found")
        assert "container:[REDACTED]" in diagnostic.message


def test_p1_01_argv_secret_flag_is_discovered_before_selection():
    text = json.dumps({"containers": [
        {"name": "a", "command": ["vllm", "serve", "/m",
                                  "--api-key", CANARY]},
        {"name": "b", "command": ["vllm", "serve", "/m",
                                  "--served-model-name", CANARY]},
    ]})
    assert CANARY in prescan_document_secrets(text, "json")
    report = _parse(text, format="json", container="b")
    assert report.status == "completed"
    assert not _leaks(report, CANARY)
    assert _setting(report, "--served-model-name").redacted is True


def test_p1_01_raw_yaml_multiline_scalars_match_loader_values():
    cases = {
        "single_quoted": ("credential: 'R9_\n  CANARY'\n", "R9_ CANARY"),
        "plain_flow": ("credential: {value: R9_\n  CANARY}\n", "R9_ CANARY"),
        "folded_block": ("credential: >-\n  R9_\n\n  CANARY\n",
                         "R9_\nCANARY"),
    }
    for label, (text, expected) in cases.items():
        assert expected in prescan_document_secrets(text, "yaml"), label


def test_p1_01_raw_yaml_block_chomping_matches_loader():
    cases = {
        "literal_clip": ("credential: |\n  R9_\n  CANARY\n",
                         "R9_\nCANARY\n"),
        "literal_strip": ("credential: |-\n  R9_\n  CANARY\n",
                          "R9_\nCANARY"),
        "literal_keep": ("credential: |+\n  R9_\n  CANARY\n\n\n",
                         "R9_\nCANARY\n\n\n"),
        "folded_clip": ("credential: >\n  R9_\n\n  CANARY\n",
                        "R9_\nCANARY\n"),
        "folded_keep": ("credential: >+\n  R9_\n\n  CANARY\n\n",
                        "R9_\nCANARY\n\n"),
    }
    for label, (text, expected) in cases.items():
        assert expected in prescan_document_secrets(text, "yaml"), label


def test_p1_01_blocked_multiline_source_identity_is_redacted():
    text = "credential: 'R9_\n  CANARY'\ncommand: [vllm, serve, /m]\n"
    report = _parse(text, format="yaml", source="R9_ CANARY")
    assert not _leaks(report, "R9_ CANARY")
    if HAS_YAML:
        assert report.status == "completed"
    else:
        assert report.status == "failed" and _yaml_unavailable(report)


# ---------------------------------------------------------------- P2-19
class _Audited(str):
    reads = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(super().__getitem__(key))
        type(self).reads += 1
        return super().__getitem__(key)


class _Mark:
    def __init__(self, index):
        self.index = index


class _Node:
    def __init__(self, start, end, value, style):
        self.start_mark, self.end_mark = _Mark(start), _Mark(end)
        self.value, self.style = value, style


def test_p2_19_raw_prescan_character_reads_are_linear(monkeypatch):
    monkeypatch.setattr(yaml_taint, "_event_taint",
                        lambda text, found: None)
    counts = {}
    for size in (256, 512, 1024, 2048, 4096):
        value = "a" + " " * size + "b"
        text = 'credential: "' + value + '"\ncommand: [vllm, serve, /m]\n'
        _Audited.reads = 0
        found = prescan_document_secrets(_Audited(text), "yaml")
        counts[size] = _Audited.reads
        assert value in found, size
        assert _Audited.reads <= 8 * len(text), size
    assert counts[2048] / counts[256] < 9.0


def test_p2_19_yaml_coordinate_reads_are_linear():
    counts = {}
    for size in (256, 512, 1024, 2048, 4096):
        value = "a" + " " * size + "b"
        text = 'credential: "' + value + '"\ncommand: [vllm, serve, /m]\n'
        start = text.index('"')
        node = _Node(start, start + len(value) + 2, value, '"')
        _Audited.reads = 0
        coords = yaml_scalar_coords(_Audited(text), node, value, Lines(text))
        counts[size] = _Audited.reads
        assert len(coords) == len(value) + 1, size
        assert _Audited.reads <= 8 * len(text), size
    assert counts[2048] / counts[256] < 9.0


def test_p2_19_flow_whitespace_run_is_consumed_once():
    value = "a" + " " * 4096 + "b"
    text = 'credential: "' + value + '"\ncommand: [vllm, serve, /m]\n'
    start = text.index('"')
    node = _Node(start, start + len(value) + 2, value, '"')
    _Audited.reads = 0
    coords = yaml_scalar_coords(_Audited(text), node, value, Lines(text))
    # The whole run is emitted in one pass: O(len(text)), not O(n^2).
    assert len(coords) == len(value) + 1
    assert _Audited.reads <= 8 * len(text)


# ---------------------------------------------------------------- P2-23
def test_p2_23_discarded_line_continuation_closes_at_last_key_char():
    text = ("command: \"vllm serve /m --compilation-config "
            "'{\\\"cudagraph_mode\\\"\\\n  :\\\"FULL\\\"}'\"\n")
    report = _parse(text, format="yaml")
    if not HAS_YAML:
        assert report.status == "failed" and _yaml_unavailable(report)
        return
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (1, 49)
    assert (span.line_end, span.column_end) == (1, 66)


def test_p2_23_adjacent_shell_quote_delimiters_close_at_last_key_char():
    text = json.dumps({"command": (
        "vllm serve /m --compilation-config "
        "'{\"cudagraph_mode\"'':\"FULL\"}'")}, indent=2)
    report = _parse(text, format="json")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (2, 53)
    assert (span.line_end, span.column_end) == (2, 70)


def test_p2_23_retained_r8_key_intervals_still_match():
    yaml_text = ("command: \"vllm serve /m --compilation-config "
                 "'{\\\"cudagraph_\\\n  mode\\\":\\\"FULL\\\"}'\"\n")
    report = _parse(yaml_text, format="yaml")
    if HAS_YAML:
        assert report.status == "completed"
        span = _setting(report, "cudagraph_mode").span
        assert (span.line_start, span.column_start) == (1, 49)
        assert (span.line_end, span.column_end) == (2, 9)
    else:
        assert report.status == "failed" and _yaml_unavailable(report)

    quoted = json.dumps({"command": (
        r'vllm serve /m --compilation-config '
        r'"{\"cudagraph_mode\":\"FULL\"}"')}, indent=2)
    report = _parse(quoted, format="json")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (2, 56)
    assert (span.line_end, span.column_end) == (2, 75)


def test_p2_23_docker_uses_the_expanded_variables_own_provenance():
    text = json.dumps({
        "env": {"DECOY": SCRIPT, "SCRIPT": SCRIPT},
        "command": 'docker run -e DECOY -e SCRIPT image sh -c "$SCRIPT"',
    }, indent=2)
    report = _parse(text, format="json")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (4, 54)
    assert (span.line_end, span.column_end) == (4, 71)
    assert "SCRIPT" in text.splitlines()[span.line_start - 1]


def test_p2_23_shell_expansion_uses_the_outer_occurrence_map():
    other = SCRIPT.replace("FULL", "NONE")
    inline = other.replace('"', '\\"')
    text = json.dumps({
        "env": {"SCRIPT": SCRIPT},
        "command": ('docker run -e SCRIPT="' + inline
                    + '" image sh -c "$SCRIPT"'),
    }, indent=2)
    report = _parse(text, format="json")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (3, 54)
    assert (span.line_end, span.column_end) == (3, 71)


def test_p2_23_docker_crlf_env_value_keeps_its_map():
    text = json.dumps({
        "env": {"SCRIPT": SCRIPT + "\r\n"},
        "command": 'docker run -e SCRIPT image sh -c "$SCRIPT"',
    }, indent=2)
    report = _parse(text, format="json")
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert (span.line_start, span.column_start) == (3, 54)
    assert (span.line_end, span.column_end) == (3, 71)


def test_p2_23_external_expansion_without_a_map_fails_closed():
    text = json.dumps({
        "env": {"DECOY": SCRIPT},
        "command": 'docker run -e DECOY -e SCRIPT image sh -c "$SCRIPT"',
    }, indent=2)
    report = _parse(text, format="json", env={"SCRIPT": SCRIPT})
    assert report.status == "failed"
    assert any(item.code == "unmapped_shell_expansion_provenance"
               for item in report.diagnostics)


def test_p2_23_external_value_overrides_document_provenance():
    other = ('vllm serve /m --compilation-config '
             "'{\"cudagraph_mode\":\"NONE\"}'")
    text = json.dumps({
        "env": {"DECOY": SCRIPT, "SCRIPT": SCRIPT},
        "command": 'docker run -e DECOY -e SCRIPT image sh -c "$SCRIPT"',
    }, indent=2)
    report = _parse(text, format="json", env={"SCRIPT": other})
    assert report.status == "failed"
    assert any(item.code == "unmapped_shell_expansion_provenance"
               for item in report.diagnostics)


def test_p2_23_inline_docker_literal_replaces_external_provenance():
    text = json.dumps({"command": [
        "docker", "run", "-e", "SCRIPT=" + SCRIPT, "image",
        "sh", "-c", '"$SCRIPT"']}, indent=2)
    report = _parse(text, format="json", env={"SCRIPT": "not-the-document"})
    assert report.status == "completed"
    span = _setting(report, "cudagraph_mode").span
    assert "SCRIPT=" in text.splitlines()[span.line_start - 1]


def test_p2_23_value_equality_fallback_is_deleted():
    assert not hasattr(launch_docker, "_expanded_script_coords")


def test_p2_23_report_span_end_uses_last_key_character():
    text = json.dumps({"command": (
        "vllm serve /m --compilation-config "
        "'{\"cudagraph_mode\"'':\"FULL\"}'")}, indent=2)
    span = _setting(_parse(text, format="json"), "cudagraph_mode").span
    line = text.splitlines()[span.line_start - 1]
    assert line[span.column_end - 2] == '"'
    assert line[span.column_end - 1] in "':"
