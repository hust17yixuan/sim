"""Round-11 regressions for `astra_review_r11.md` (P1-01, P2-19, P2-23).

[Role] Turn every Round-11 counterexample into a public launcher,
  credential-inventory, coordinate and bounded-work assertion in all three
  import modes (default, PyYAML 6.0.3 and blocked PyYAML), without changing
  parse admission or mutating ``sys.path``.
[Invariants] Block mapping and sequence scalars compose with their sibling
  command/args field for every configured pair; a block scalar advances exactly
  once past its last line; folded more-indented whitespace and every YAML break
  (CR, CRLF, LF, NEL, LS, PS) match ``SafeLoader``; strip/clip/keep use the
  final content line's break; static ``$VAR`` discovery resolves the container's
  literal env (inline assignments included) while an unsupported operator still
  fails the selected command; a failure-tolerant taint scan keeps words on both
  sides of that operator; YAML 1.1 booleans contribute the adapter's lowercase
  spelling; composition runs only for a dirty command/args frame; and the YAML
  line index uses loader break syntax with an enforced in-bounds leaf check.
[Timeouts & Locks] Every case is in-memory and bounded; the module stays under
  a few seconds per mode and takes no locks.
[Side-Effect Policy] No file, network or device writes; fixtures are in-memory.
"""
from __future__ import annotations

import json

import pytest

from mindie_sim import launch_documents as documents
from mindie_sim import launch_spans as spans
from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_coords import Lines
from mindie_sim.launch_taint import _shell_taint, prescan_document_secrets
from mindie_sim.launch_yaml_raw import raw_document_taint
from mindie_sim.launch_yaml_taint import _event_taint

try:  # The availability is fixed at collection time for this process.
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
VERSION = "vllm-ascend@0.30.0"
SECRET = "R11_SECRET_4c7e9d"
PAIRS = (("command", "args"), ("entrypoint", "cmd"),
         ("command", "cmd"), ("entrypoint", "args"))
REPRESENTATIONS = ("literal_command", "folded_command", "block_arg_scalar")
BREAKS = (("lf", "\n"), ("crlf", "\r\n"), ("cr", "\r"),
          ("nel", "\x85"), ("ls", "\u2028"), ("ps", "\u2029"))
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


def _unavailable(report):
    return any(item.code == "yaml_parser_unavailable"
               for item in report.diagnostics)


def _no_coordinate_failure(report):
    return not any(item.code == "source_coordinates_unavailable"
                   for item in report.diagnostics)


def _containers_text(pair, representation, fmt):
    left, right = pair
    if representation == "literal_command":
        first = f"  {left}: |-\n    vllm serve /m --api-key\n"
    elif representation == "folded_command":
        first = f"  {left}: >-\n    vllm serve /m --api-key\n"
    else:
        first = (f"  {left}: [vllm, serve, /m, --api-key]\n"
                 f"  {right}:\n  - |-\n    {SECRET}\n")
    if representation != "block_arg_scalar":
        first += f"  {right}: [{SECRET}]\n"
    if fmt == "json":
        document = {"containers": []}
        command = ["vllm", "serve", "/m", "--served-model-name", SECRET]
        if representation == "block_arg_scalar":
            document["containers"] = [
                {"name": "a", left: ["vllm", "serve", "/m", "--api-key"],
                 right: [SECRET]},
                {"name": "b", "command": command},
            ]
        else:
            value = ("vllm serve /m --api-key" if representation == "literal_command"
                     else "vllm serve /m --api-key")
            document["containers"] = [
                {"name": "a", left: value, right: [SECRET]},
                {"name": "b", "command": command},
            ]
        return json.dumps(document)
    return ("containers:\n- name: a\n" + first
            + "- name: b\n  command: [vllm, serve, /m, "
              f"--served-model-name, {SECRET}]\n")


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_p1_01a_block_scalars_compose_with_sibling_fields(
        pair, representation, fmt):
    text = _containers_text(pair, representation, fmt)
    assert SECRET in prescan_document_secrets(text, fmt)
    report = _parse(text, format=fmt, container="b", source=SECRET)
    if fmt == "yaml" and not HAS_YAML:
        assert report.status == "failed" and _unavailable(report)
    else:
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)
    missing = _parse(text, format=fmt, container=SECRET)
    assert not _leaks(missing, SECRET)


def test_p1_01a_block_cursor_advances_to_the_next_line():
    text = ("notes: |-\n  ordinary\ncredential: " + SECRET
            + "\ncommand: [vllm, serve, /m]\n")
    assert SECRET in prescan_document_secrets(text, "yaml")
    report = _parse(text, format="yaml", source=SECRET)
    if not HAS_YAML:
        assert _unavailable(report)
    assert not _leaks(report, SECRET)


def test_p1_01a_quoted_flag_in_a_block_composed_command_is_discovered():
    for fmt in ("json", "yaml"):
        command = 'vllm serve /m --api"-key"'
        if fmt == "json":
            text = json.dumps({"containers": [
                {"name": "a", "command": command, "args": [SECRET]},
                {"name": "b", "command": ["vllm", "serve", "/m",
                                          "--served-model-name", SECRET]},
            ]})
        else:
            text = ("containers:\n- name: a\n  command: "
                    + json.dumps(command) + "\n  args: [" + SECRET + "]\n"
                    "- name: b\n  command: [vllm, serve, /m, "
                      f"--served-model-name, {SECRET}]\n")
        assert SECRET in prescan_document_secrets(text, fmt)
        report = _parse(text, format=fmt, container="b")
        if fmt == "yaml" and not HAS_YAML:
            assert report.status == "failed"
        else:
            assert report.status == "completed"
        assert not _leaks(report, SECRET)


BLOCK_CASES = {
    "folded_more_indented_whitespace": (
        "credential: >-\n  R11_A\n      \n  R11_B\n"
        "command: [vllm, serve, /m]\n", "R11_A\n    \nR11_B"),
    "literal_more_indented_whitespace": (
        "credential: |\n  R11_A\n    \n  R11_B\n"
        "command: [vllm, serve, /m]\n", "R11_A\n  \nR11_B\n"),
    "folded_blank_line": (
        "credential: >\n  R11_A\n\n  R11_B\n"
        "command: [vllm, serve, /m]\n", "R11_A\nR11_B\n"),
    "clip_keeps_final_ls_content_break": (
        "credential: |\u2028  R11_A\u2028\u2028\n"
        "command: [vllm, serve, /m]\n", "R11_A\u2028"),
    "clip_keeps_final_ps_content_break": (
        "credential: |\u2029  R11_A\u2029\u2029\n"
        "command: [vllm, serve, /m]\n", "R11_A\u2029"),
    "strip_ls_content_breaks": (
        "credential: |-\u2028  R11_A\u2028  \u2028  R11_B\u2028"
        "command: [vllm, serve, /m]\n", "R11_A\u2028\u2028R11_B"),
    "keep_whitespace_only_implicit_indent": (
        "credential: |+\n  \ncommand: [vllm, serve, /m]\n", "\n"),
    "crlf_literal_normalizes": (
        "credential: |-\r\n  R11_A\r\n  R11_B\r\n"
        "command: [vllm, serve, /m]\n", "R11_A\nR11_B"),
    "nel_clip_normalizes": (
        "credential: |\x85  R11_A\x85command: [vllm, serve, /m]\n",
        "R11_A\n"),
    "cr_clip_normalizes": (
        "credential: |\r  R11_A\rcommand: [vllm, serve, /m]\n", "R11_A\n"),
    "plain_ls_keeps_identity": (
        "credential: R11_A\u2028  R11_B\ncommand: [vllm, serve, /m]\n",
        "R11_A\u2028R11_B"),
    "plain_ps_keeps_identity": (
        "credential: R11_A\u2029  R11_B\ncommand: [vllm, serve, /m]\n",
        "R11_A\u2029R11_B"),
    "plain_nel_folds": (
        "credential: R11_A\x85  R11_B\ncommand: [vllm, serve, /m]\n",
        "R11_A R11_B"),
    "plain_cr_folds": (
        "credential: R11_A\r  R11_B\ncommand: [vllm, serve, /m]\n",
        "R11_A R11_B"),
    "plain_crlf_folds": (
        "credential: R11_A\r\n  R11_B\ncommand: [vllm, serve, /m]\n",
        "R11_A R11_B"),
}


@pytest.mark.parametrize("label", tuple(BLOCK_CASES))
def test_p1_01b_block_and_plain_values_match_the_loader(label):
    text, expected = BLOCK_CASES[label]
    if HAS_YAML:
        assert str(_yaml.safe_load(text)["credential"]) == expected, label
    found = prescan_document_secrets(text, "yaml")
    assert expected in found, (label, found)
    if not label.startswith("plain_"):
        raw = set()
        raw_document_taint(text, raw, _shell_taint)
        assert expected in raw, (label, raw)
        if HAS_YAML:
            event = set()
            _event_taint(text, event, _shell_taint)
            assert expected in event, (label, event)
    report = _parse(text, format="yaml", source=expected)
    if HAS_YAML:
        assert report.status == "completed"
        assert _no_coordinate_failure(report)
    else:
        assert report.status == "failed" and _unavailable(report)
    if expected.strip():
        # A whitespace-only needle cannot be meaningful in Markdown output,
        # which necessarily contains line breaks of its own.
        assert not _leaks(report, expected)


def test_p1_01b_raw_block_coordinates_are_exact_and_in_bounds():
    if not HAS_YAML:
        return
    from mindie_sim.launch_yaml_coords import yaml_scalar_coords

    for label, (text, expected) in BLOCK_CASES.items():
        if label.startswith("plain_"):
            continue
        root = _yaml.compose(text)
        node = next(value for key, value in root.value
                    if key.value == "credential")
        lines = Lines(text)
        lines.configure(True)
        coords = yaml_scalar_coords(text, node, node.value, lines)
        assert len(coords) == len(expected) + 1, label
        for entry in coords:
            assert lines.bounds(entry[0], entry[1], entry[3], entry[4]), label


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_p1_01c_static_env_expansion_is_discovered(fmt):
    first = {"name": "a", "env": {"VALUE": SECRET},
             "command": 'vllm serve /m --api-key "$VALUE"'}
    second = {"name": "b",
              "command": ["vllm", "serve", "/m", "--served-model-name", SECRET]}
    if fmt == "json":
        text = json.dumps({"containers": [first, second]})
    else:
        text = ("containers:\n- name: a\n  env: {VALUE: " + SECRET + "}\n"
                "  command: 'vllm serve /m --api-key \"$VALUE\"'\n"
                "- name: b\n  command: [vllm, serve, /m, "
                f"--served-model-name, {SECRET}]\n")
    found = prescan_document_secrets(text, fmt)
    assert SECRET in found and "$VALUE" in found
    report = _parse(text, format=fmt, container="b")
    if fmt == "yaml" and not HAS_YAML:
        assert report.status == "failed"
    else:
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)
    missing = _parse(text, format=fmt, container=SECRET)
    assert not _leaks(missing, SECRET)


@pytest.mark.parametrize("fmt", ["json", "yaml"])
@pytest.mark.parametrize("layout", ["block", "flow"])
def test_p1_01c_env_declared_after_the_command_still_resolves(fmt, layout):
    second = {"name": "b",
              "command": ["vllm", "serve", "/m", "--served-model-name", SECRET]}
    first = {"name": "a", "command": 'vllm serve /m --api-key "$VALUE"',
             "env": {"VALUE": SECRET}}
    if fmt == "json":
        text = json.dumps({"containers": [first, second]})
    elif layout == "flow":
        text = ("containers:\n- name: a\n"
                "  command: 'vllm serve /m --api-key \"$VALUE\"'\n"
                "  env: {VALUE: " + SECRET + "}\n"
                "- name: b\n  command: [vllm, serve, /m, "
                f"--served-model-name, {SECRET}]\n")
    else:
        text = ("containers:\n- name: a\n"
                "  command: 'vllm serve /m --api-key \"$VALUE\"'\n"
                "  env:\n    VALUE: " + SECRET + "\n"
                "- name: b\n  command: [vllm, serve, /m, "
                f"--served-model-name, {SECRET}]\n")
    assert SECRET in prescan_document_secrets(text, fmt)
    report = _parse(text, format=fmt, container="b")
    if fmt == "yaml" and not HAS_YAML:
        assert report.status == "failed"
    else:
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)


def test_p1_01c_inline_assignment_expands_in_global_discovery():
    command = 'VALUE=' + SECRET + ' vllm serve /m --api-key "$VALUE"'
    text = json.dumps({"containers": [
        {"name": "a", "command": command},
        {"name": "b", "command": ["vllm", "serve", "/m",
                                  "--served-model-name", SECRET]},
    ]})
    assert SECRET in prescan_document_secrets(text, "json")
    report = _parse(text, format="json", container="b")
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)


SUFFIXES = ("| true", "&& true", "; true", "> /tmp/out", "$(true)", "`true`")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_p1_01c_prefix_tolerant_discovery_after_unsupported_operator(suffix):
    command = 'vllm serve /m --api"-key" ' + SECRET + " " + suffix
    text = json.dumps({"containers": [
        {"name": "a", "command": command},
        {"name": "b", "command": ["vllm", "serve", "/m",
                                  "--served-model-name", SECRET]},
    ]})
    assert SECRET in prescan_document_secrets(text, "json")
    report = _parse(text, format="json", container="b")
    assert report.status == "completed"
    assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)


def test_p1_01c_selected_command_still_rejects_the_operator():
    text = json.dumps({"containers": [
        {"name": "a", "command": "vllm serve /m | true"},
        {"name": "b", "command": ["vllm", "serve", "/m"]},
    ]})
    report = _parse(text, format="json", container="a")
    assert report.status == "failed"
    assert any(item.code == "unsupported_shell_syntax"
               for item in report.diagnostics)


@pytest.mark.parametrize("raw,expected", [
    ("yes", "true"), ("TRUE", "true"), ("On", "true"),
    ("off", "false"), ("False", "false"), ("NO", "false"),
])
def test_p1_01c_yaml_boolean_normalization(raw, expected):
    text = ("containers:\n- name: a\n  env: {API_KEY: " + raw + "}\n"
            "  command: [vllm, serve, /m]\n"
            "- name: b\n  command: [vllm, serve, /m, --served-model-name, \""
            + expected + "\"]\n")
    found = prescan_document_secrets(text, "yaml")
    assert raw in found and expected in found
    report = _parse(text, format="yaml", container="b")
    if not HAS_YAML:
        assert report.status == "failed"
        return
    assert report.status == "completed"
    setting = _setting(report, "--served-model-name")
    assert setting.redacted is True
    assert setting.value == "[REDACTED]"


def _composition_document(n):
    return json.dumps({
        "command": ["vllm", "serve", "/m", "--api-key"],
        "args": ["C" * n],
        **{f"note{index}": "x" for index in range(n)},
    })


def _surfaces(mode, text):
    calls = []

    def counted(surface, found):
        calls.append(surface)

    if mode == "event":
        _event_taint(text, set(), counted)
    else:
        raw_document_taint(text, set(), counted)
    return calls


def _joined(surface):
    return " ".join(surface) if isinstance(surface, tuple) else str(surface)


@pytest.mark.parametrize("mode", ["event", "raw_flow"])
def test_p2_19_composition_never_rescans_unchanged_fields(mode):
    if mode == "event" and not HAS_YAML:
        return
    counts = {}
    for n in (64, 128, 256, 512):
        text = _composition_document(n)
        calls = _surfaces(mode, text)
        composed = [surface for surface in calls
                    if "vllm serve /m --api-key" in _joined(surface)
                    and "C" * 8 in _joined(surface)]
        # FIX25 removes the unsafe joined-shell view; only the argv view runs.
        assert len(composed) == 1 and isinstance(composed[0], tuple), (mode, n, composed)
        counts[n] = len(calls)
        assert counts[n] < 8 * n + 40, (mode, n, counts[n])
    assert counts[512] - counts[256] <= 2 * (512 - 256) * 4


def test_p2_19_real_scanner_work_is_linear(monkeypatch):
    import mindie_sim.launch_taint as taint

    original = taint.scan
    totals = {}
    try:
        for n in (64, 128, 256, 512):
            text = _composition_document(n)
            total = {"chars": 0}

            def counted(value, *args, **kwargs):
                total["chars"] += len(value)
                return original(value, *args, **kwargs)

            monkeypatch.setattr(taint, "scan", counted)
            found = taint.prescan_document_secrets(text, "yaml")
            assert "C" * n in found
            totals[n] = total["chars"]
    finally:
        monkeypatch.setattr(taint, "scan", original)
    assert totals[64] <= 60 * 64 + 3000
    assert totals[512] <= 10 * totals[64]


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_p2_19_one_document_index_and_one_prescan(monkeypatch, fmt):
    indices, prescans = [], []
    original_prescan = documents.prescan_document_secrets
    original_lines = spans.Lines.__init__

    def counting_lines(self, text, yaml_breaks=False):
        indices.append(len(text))
        original_lines(self, text, yaml_breaks)

    def counting_prescan(text, format="shell"):
        prescans.append((format, len(text)))
        return original_prescan(text, format)

    monkeypatch.setattr(spans.Lines, "__init__", counting_lines)
    monkeypatch.setattr(documents, "prescan_document_secrets", counting_prescan)
    text = (json.dumps({"notes": ["abcdefghij"] * 800,
                        "command": ["vllm", "serve", "/m"]})
            if fmt == "json" else
            "notes:\n" + "- abcdefghij\n" * 800
            + "command: [vllm, serve, /m]\n")
    try:
        document = documents.parse_document(text, format=fmt)
    except documents.DocumentInputError as exc:
        assert fmt == "yaml" and not HAS_YAML and exc.code == (
            "yaml_parser_unavailable")
    else:
        assert document.argv == ("vllm", "serve", "/m")
    assert len(prescans) == 1
    if fmt == "yaml" and not HAS_YAML:
        assert indices == []
    else:
        assert len(indices) == 1


@pytest.mark.parametrize("label,break_char", BREAKS)
def test_p2_23_yaml_break_coordinates_are_exact(label, break_char):
    text = "notes: harmless" + break_char + "command: " + json.dumps(SCRIPT) + "\n"
    report = _parse(text, format="yaml")
    if not HAS_YAML:
        assert report.status == "failed" and _unavailable(report)
        return
    assert report.status == "completed"
    item = _setting(report, "cudagraph_mode")
    assert (item.span.line_start, item.span.column_start) == (2, 49), label
    assert (item.span.line_end, item.span.column_end) == (2, 66), label
    assert not _leaks(report, SCRIPT)


def test_p2_23_lines_index_models_yaml_breaks_and_bounds():
    text = "notes: harmless\u2028command: x\n"
    default = Lines(text)
    assert default.position(16) == (1, 17)
    assert default.line_len(1) == len(text) - 1
    yaml_lines = Lines(text)
    yaml_lines.configure(True)
    assert yaml_lines.position(16) == (2, 1)
    assert yaml_lines.line_len(1) == 15
    assert yaml_lines.line_len(2) == 10
    assert yaml_lines.bounds(2, 1, 2, 11)
    assert not yaml_lines.bounds(1, 17, 1, 17)
    assert not yaml_lines.bounds(0, 1, 1, 1)
    assert yaml_lines.bounds(3, 1, 3, 1)
    assert not yaml_lines.bounds(4, 1, 4, 1)
    crlf = Lines("a\r\nb\r")
    crlf.configure(True)
    assert crlf.position(3) == (2, 1)
    assert crlf.position(5) == (3, 1)
    assert not crlf.bounds(1, 3, 1, 3)


def test_p2_23_out_of_bounds_leaf_fails_closed():
    if not HAS_YAML:
        return
    text = "notes: harmless\u2029command: " + json.dumps(SCRIPT) + "\n"
    root = spans.parse_yaml(text)
    leaf = root.member("command")
    assert leaf is not None and leaf.char_coords
    lines = Lines(text)
    lines.configure(True)
    for entry in leaf.char_coords:
        assert lines.bounds(entry[0], entry[1], entry[3], entry[4])
    broken = Lines(text)
    assert broken.position(leaf.char_coords[0][2]) != (leaf.line, leaf.column)
