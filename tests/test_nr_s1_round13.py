"""Round-13 regressions for `astra_review_r13.md` (P1-01, P1-02, P2-01, P2-02).

[Role] Turn every Round-13 counterexample into a public launcher,
  credential-inventory and bounded-work assertion in all three import modes
  (default, PyYAML 6.0.3 and blocked PyYAML), without changing parse admission
  or mutating ``sys.path``.
[Invariants] A command-local assignment/export prefix scopes Docker forwarding
  and the inner script; an exact ``$SCRIPT`` reference resolves only from the
  explicit forwarded environment.  Assignment overlays never copy the base
  map, a raw block env installs each scalar pre-normalized, and sibling block
  mappings keep separate env/field frames.
[Timeouts & Locks] Every case is in-memory and linear; the module stays under
  bounded execution per mode and takes no locks.
[Side-Effect Policy] No file, network or device writes; fixtures are in-memory.
"""
from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path

import pytest

from mindie_sim.feature_reports import canonical_json, launch_markdown
from mindie_sim.launch_bindings import parse_launch
from mindie_sim.launch_env import EnvMap, _EnvOverlay
from mindie_sim.launch_taint import (
    _decoded_shell_words, _shell_taint, _tolerant_words,
    prescan_document_secrets,
)
from mindie_sim.launch_yaml_raw import raw_document_taint
from mindie_sim.launch_yaml_taint import _event_taint

try:  # Availability is fixed at collection time for this process.
    import yaml as _yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

WORKLOAD = {"num_requests": 2, "prompt_tokens": 128, "output_tokens": 8,
            "prefix_tokens": 0}
BASE = {"machine": "a3", "model": "qwen3-30b-a3b", "workload": WORKLOAD}
VERSION = "vllm-ascend@0.30.0"
SECRET = "R13_SECRET_71c4d9"
SCRIPT = 'vllm serve /m --api"-key" "$VALUE"'
REFLECTOR = {"name": "b", "command": ["vllm", "serve", "/m",
                                     "--served-model-name", SECRET]}
PREFIXES = ("VALUE=" + SECRET + " ", "env VALUE=" + SECRET + " ",
            "export VALUE=" + SECRET + "\n")
WRAPPERS = ("sh -c ", "bash -c ",
            "docker run -e VALUE image sh -c ",
            "docker run --env=VALUE image /bin/bash -c ")


def _parse(text, **kwargs):
    return parse_launch(text, **{**BASE, "framework_version": VERSION,
                                 **kwargs})


def _settings(report):
    return (*report.applied, *report.neutral_operational,
            *report.unmodeled_or_unsupported)


def _setting(report, name):
    return next(item for item in _settings(report) if item.name == name)


def _leaks(report, needle):
    return needle in canonical_json(report) or needle in launch_markdown(report)


def _json_document(command, env=None):
    first = {"name": "a"}
    if env is not None:
        first["env"] = env
    first["command"] = command
    return json.dumps({"containers": [first, REFLECTOR]})


def _block_document(command, env=None):
    lines = ["containers:", "- name: a"]
    if env is not None:
        lines += ["  env:", "    VALUE: " + env]
    lines += ["  command: " + json.dumps(command)]
    lines += ["- name: b", "  command: [vllm, serve, /m, "
              "--served-model-name, " + SECRET + "]"]
    return "\n".join(lines) + "\n"


def _assert_reflector_safe(text, fmt):
    report = _parse(text, format=fmt, container="b")
    if fmt == "yaml" and not HAS_YAML:
        assert report.status == "failed"
        assert any(item.code == "yaml_parser_unavailable"
                   for item in report.diagnostics)
    else:
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").redacted is True
    assert not _leaks(report, SECRET)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("layout", ["json", "block"])
def test_r13_assignment_prefix_reaches_supported_wrapper(prefix, wrapper, layout):
    command = prefix + wrapper + shlex.quote(SCRIPT)
    text = (_json_document(command) if layout == "json"
            else _block_document(command))
    fmt = "json" if layout == "json" else "yaml"
    assert SECRET in prescan_document_secrets(text, fmt)
    if fmt == "yaml":
        found = set()
        raw_document_taint(text, found, _shell_taint)
        assert SECRET in found
    report = _parse(text, format=fmt, container="a")
    if fmt == "json" or HAS_YAML:
        # Admission accepts the Docker envelope with an assignment prefix;
        # a prefixed bare shell wrapper remains an unsupported command.
        assert report.status == ("completed" if wrapper.startswith("docker")
                                 else "failed")
        assert not _leaks(report, SECRET)
    _assert_reflector_safe(text, fmt)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_r13_local_assignment_overrides_existing_frame_value(wrapper):
    command = "VALUE=" + SECRET + " " + wrapper + shlex.quote(SCRIPT)
    text = _json_document(command, env={"VALUE": "OLD_VALUE"})
    found = prescan_document_secrets(text, "json")
    assert SECRET in found
    assert "OLD_VALUE" not in found
    _assert_reflector_safe(text, "json")


@pytest.mark.parametrize("command", [
    "export VALUE\ndocker run -e VALUE image sh -c " + shlex.quote(SCRIPT),
    "export A B; docker run -e VALUE image sh -c " + shlex.quote(SCRIPT),
    "export VALUE\nsh -c " + shlex.quote(SCRIPT),
])
def test_r13_bare_export_prefix_keeps_the_frame_value(command):
    text = _json_document(command, env={"VALUE": SECRET})
    assert SECRET in prescan_document_secrets(text, "json")
    report = _parse(text, format="json", container="a")
    if "docker run" in command and ";" not in command:
        assert report.status == "completed"
    else:
        # Admission still rejects a prefixed bare shell wrapper and the
        # unsupported operator; discovery must see the export scope anyway.
        assert report.status == "failed"
        assert any(item.code == "unsupported_shell_syntax"
                   for item in report.diagnostics)
    assert not _leaks(report, SECRET)
    _assert_reflector_safe(text, "json")


@pytest.mark.parametrize("prefix", ["", "VALUE=" + SECRET + " "])
def test_r13_unforwarded_variable_does_not_enter_inner_script(prefix):
    command = prefix + "docker run -e OTHER image sh -c " + shlex.quote(SCRIPT)
    text = _json_document(command)
    found = prescan_document_secrets(text, "json")
    assert SECRET not in found
    assert "$VALUE" in found


@pytest.mark.parametrize("command", [
    ["docker", "run", "-e", "VALUE", "-e", "SCRIPT", "image", "sh", "-c",
     "$SCRIPT"],
    "docker run -e VALUE -e SCRIPT image sh -c '$SCRIPT'",
    'docker run -e VALUE -e SCRIPT image sh -c "$SCRIPT"',
])
def test_r13_exact_script_reference_resolves_before_inner_scan(command):
    text = _json_document(command, env={"VALUE": SECRET, "SCRIPT": SCRIPT})
    assert SECRET in prescan_document_secrets(text, "json")
    assert _parse(text, format="json", container="a").status == "completed"
    _assert_reflector_safe(text, "json")


def test_r13_unforwarded_exact_script_reference_is_not_resolved():
    command = ["docker", "run", "-e", "VALUE=OUTER", "image", "sh", "-c",
               "$SCRIPT"]
    text = _json_document(command, env={"SCRIPT": SCRIPT})
    found = prescan_document_secrets(text, "json")
    assert SECRET not in found
    report = _parse(text, format="json", container="a")
    assert report.status == "failed"
    assert not _leaks(report, SECRET)


class _CountingMapping(Mapping):
    """Base view that records every lookup and enumeration."""

    def __init__(self, data):
        self.data = dict(data)
        self.lookups = 0
        self.iterations = 0

    def __getitem__(self, name):
        self.lookups += 1
        return self.data[name]

    def __iter__(self):
        self.iterations += 1
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("tolerant", [False, True])
def test_r13_assignment_overlay_never_enumerates_the_base(tolerant):
    probe = _CountingMapping({f"V{index}": ("x", False) for index in range(512)})
    overlay = _EnvOverlay(probe)
    if tolerant:
        words = _tolerant_words('LOCAL=x "$V511" | true', overlay)
    else:
        words = _decoded_shell_words('LOCAL=x "$V511"', overlay)
    assert "x" in words
    assert probe.iterations == 0
    assert probe.lookups <= 2


def test_r13_decoder_has_no_base_copy_statement():
    import mindie_sim.launch_taint as taint

    source = Path(taint.__file__).read_text(encoding="utf-8")
    assert "local = dict(mapping)" not in source
    assert "local = dict(base)" not in source


@pytest.mark.parametrize("dollar", [False, True])
def test_r13_raw_block_env_normalization_is_linear(monkeypatch, dollar):
    import mindie_sim.launch_yaml_raw as raw

    original = raw.normalize_env
    work = {}
    for n in (64, 128, 256, 512, 1024):
        calls = []

        def counted(env):
            if env and not isinstance(env, EnvMap):
                calls.append(len(env))
            return original(env)

        monkeypatch.setattr(raw, "normalize_env", counted)
        value = '"$UNKNOWN"' if dollar else "x"
        lines = ["env:"] + [f"  V{index}: {value}" for index in range(n)]
        lines.append("command: [vllm, serve, /m]")
        lines += [f"note{index}: x" for index in range(n)]
        raw.raw_document_taint("\n".join(lines) + "\n", set(), _shell_taint)
        work[n] = (len(calls), sum(calls))
    for n, (calls, entries) in work.items():
        assert entries <= n + 8, (n, calls, entries)
        assert calls <= n + 2, (n, calls, entries)
    for small, large in ((64, 128), (128, 256), (256, 512), (512, 1024)):
        assert work[large][1] <= 2.2 * work[small][1] + 8


def _sibling_document(reverse, services=False):
    one = ["env:", "  VALUE: " + SECRET, "notes: harmless"]
    two = ['command: vllm serve /m --api"-key" "$VALUE"']
    blocks = [("a" if services else "one", one),
              ("b" if services else "two", two)]
    if reverse:
        blocks.reverse()
    lines = ["command: [vllm, serve, /m, --served-model-name, " + SECRET + "]",
             "services:" if services else "metadata:"]
    for name, body in blocks:
        lines.append("  " + name + ":")
        lines += ["    " + line for line in body]
    return "\n".join(lines) + "\n"


def _json_sibling(reverse, services=False):
    one = {"env": {"VALUE": SECRET}, "notes": "harmless"}
    two = {"command": 'vllm serve /m --api"-key" "$VALUE"'}
    blocks = [("a" if services else "one", one), ("b" if services else "two", two)]
    if reverse:
        blocks.reverse()
    body = dict(blocks)
    holder = {"services": body} if services else {"metadata": body}
    holder["command"] = ["vllm", "serve", "/m", "--served-model-name", SECRET]
    return json.dumps(holder)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("services", [False, True])
def test_r13_block_sibling_mappings_keep_separate_frames(reverse, services):
    block = _sibling_document(reverse, services)
    found = set()
    raw_document_taint(block, found, _shell_taint)
    assert SECRET not in found
    assert SECRET not in prescan_document_secrets(block, "yaml")
    flow = _json_sibling(reverse, services)
    assert SECRET not in prescan_document_secrets(flow, "json")
    assert SECRET not in prescan_document_secrets(flow, "yaml")
    if HAS_YAML:
        report = _parse(block, format="yaml")
        assert report.status == "completed"
        assert _setting(report, "--served-model-name").value == SECRET
        # JSON and raw flow preserve the public value too: nothing declares
        # it as a credential, so the block report must not invent one.
        assert _setting(report, "--served-model-name").redacted is False
        assert _leaks(report, SECRET)


@pytest.mark.parametrize("pending", [False, True])
def test_r13_block_late_env_inside_one_frame_still_resolves(pending):
    lines = ["metadata:", "  one:"]
    if pending:
        lines += ["    command:", "      - vllm", "      - serve", "      - /m",
                  '      - "--api-key"', '      - "$VALUE"']
    else:
        lines += ['    command: vllm serve /m --api"-key" "$VALUE"']
    lines += ["    env:", "      VALUE: " + SECRET]
    text = "\n".join(lines) + "\n"
    assert SECRET in prescan_document_secrets(text, "yaml")
    found = set()
    raw_document_taint(text, found, _shell_taint)
    assert SECRET in found
    if HAS_YAML:
        events = set()
        _event_taint(text, events, _shell_taint)
        assert SECRET in events


JSON_SECRET = "operand_secret_93cb7e"


def _operand_document(fields, layout):
    first = dict(name="a", **fields)
    sibling = dict(name="b", command=["vllm", "serve", "/m",
                   "--served-model-name", "pre", "--revision", JSON_SECRET])
    if layout in ("json", "flow"):
        return json.dumps(dict(containers=[first, sibling]))
    lines = ["containers:", "- name: a"]
    for key, value in fields.items():
        if layout == "block" and isinstance(value, str):
            lines += ["  " + key + ": |-"]
            lines += ["    " + line for line in value.split("\n")]
        else:
            lines += ["  " + key + ": " + json.dumps(value)]
    return "\n".join(lines + ["- name: b", "  command: " +
                            json.dumps(sibling["command"])]) + "\n"


def _assert_operand_document(fields, layout, sensitive=True):
    from mindie_sim.feature_reports import validate_launch_report, to_dict
    from mindie_sim.schema.case import Case
    text = _operand_document(fields, layout)
    fmt = "json" if layout == "json" else "yaml"
    inventories = [set(prescan_document_secrets(text, fmt))]
    if fmt == "yaml":
        for scan in (raw_document_taint, _event_taint):
            found = set()
            scan(text, found, _shell_taint)
            if scan is raw_document_taint or HAS_YAML:
                inventories.append(found)
    for found in inventories:
        assert (JSON_SECRET in found) == sensitive
        assert "pre" not in found
    for selection in ("b", "missing"):
        report = _parse(text, format=fmt, container=selection, source=JSON_SECRET)
        assert report.source.name == ("[REDACTED]" if sensitive else JSON_SECRET)
        if sensitive:
            assert not _leaks(report, JSON_SECRET)
        if selection == "b" and (fmt == "json" or HAS_YAML):
            assert report.status == "completed"
            Case.from_mapping(report.case)
            validate_launch_report(report)
            validate_launch_report(to_dict(report))
            benign = _setting(report, "--served-model-name")
            assert benign.value == "pre" and not benign.redacted
            credential = _setting(report, "--revision")
            assert credential.value == ("[REDACTED]" if sensitive else JSON_SECRET)
            assert credential.redacted is sensitive
        else:
            assert report.status == "failed"
            expected = "yaml_parser_unavailable" if fmt == "yaml" and not HAS_YAML else "container_not_found"
            assert [d.code for d in report.diagnostics] == [expected]


@pytest.mark.parametrize("variant", ["plain", "escaped", "adjacent", "array", "benign"])
@pytest.mark.parametrize("form", ["scalar", "argv", "split", "mixed", "docker"])
@pytest.mark.parametrize("flag,attached", [("--additional-config", False),
                         ("--additional-config", True), ("-c", False), ("-c", True)])
@pytest.mark.parametrize("layout", ["json", "flow", "mapping", "block"])
def test_complete_json_operands_are_discovered_globally(variant, form, flag, attached, layout):
    body = json.dumps({"api_key": JSON_SECRET, "model": "pre"})
    if variant == "escaped":
        body = body.replace("api_key", r"api\u005fkey")
    elif variant == "array":
        body = " [" + body + "]"
    elif variant == "benign":
        body = '{"model":"pre"}'
    quoted = shlex.quote(body)
    if variant == "adjacent":
        offset = body.index("key")
        quoted = shlex.quote(body[:offset]) + '"key"' + shlex.quote(body[offset + 3:])
    atoms = [flag + "=" + body] if attached else [flag, body]
    command = "vllm serve /m " + flag + ("=" if attached else " ") + quoted
    fields = {"command": command}
    if form == "argv":
        fields = {"command": ["vllm", "serve", "/m", *atoms]}
    elif form == "split":
        fields = {"command": ["vllm", "serve", "/m", *atoms[:-1]], "args": atoms[-1:]}
    elif form == "mixed":
        fields = {"command": "vllm serve /m" + (" " + flag if not attached else ""),
                  "args": atoms if attached else [body]}
    elif form == "docker":
        fields = {"command": ["docker", "run", "image", "sh", "-c", command]}
    _assert_operand_document(fields, layout, variant != "benign")


@pytest.mark.parametrize("form", ["inline", "env_scalar", "env_argv", "env_only", "env_files"])
@pytest.mark.parametrize("layout", ["json", "flow", "mapping", "block"])
def test_json_assignment_and_environment_values_are_global(form, layout):
    body = ' {"api\\u005fkey":"' + JSON_SECRET + '","model":"pre"}'
    fields = {"command": 'vllm serve /m --additional-config "$CFG"', "env": {"CFG": body}}
    if form == "inline":
        fields = {"command": "CFG=" + shlex.quote(body) + " " + fields["command"]}
    elif form == "env_argv":
        fields["command"] = ["vllm", "serve", "/m", "-c=$CFG"]
    elif form == "env_only":
        fields["command"] = ["vllm", "serve", "/m"]
    elif form == "env_files":
        fields = {"command": ["vllm", "serve", "/m"], "env_files": {"config.env": {"CFG": body}}}
    _assert_operand_document(fields, layout)


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("flag", ["--additional-config", "-c"])
def test_static_json_argv_expansion_preserves_benign_contributors(attached, flag):
    body = '{"api\\u005fkey":"' + JSON_SECRET + '","model":"pre"}'
    words = (flag + "=$CFG",) if attached else (flag, "$CFG")
    found = set()
    _shell_taint(words, found, {"CFG": body})
    assert JSON_SECRET in found and body not in found and "pre" not in found
    found = set()
    _shell_taint(words, found, {"CFG": '{"model":"pre"}'})
    assert not found


@pytest.mark.parametrize("surface", ["env", "env_files", "argv"])
def test_explicit_json_values_are_scanned_before_rejection(surface):
    body = '{"api\\u005fkey":"' + JSON_SECRET + '"}'
    extra = {surface: {"CFG": body} if surface == "env" else {"values.env": {"CFG": body}}}
    text, fmt = "invalid $MISSING", "shell"
    if surface == "argv":
        text, fmt, extra = ["unsupported", "-c", body], "argv", {}
    report = _parse(text, format=fmt, source=JSON_SECRET, **extra)
    assert report.status != "completed"
    assert report.source.name == "[REDACTED]" and not _leaks(report, JSON_SECRET)


@pytest.mark.parametrize("shape", ["redirect_flag", "redirect_value", "redirect_inline",
                         "redirect_assignment", "option", "statement", "split_operand"])
def test_json_discovery_keeps_option_and_statement_boundaries(shape):
    body = '{"api\\u005fkey":"' + JSON_SECRET + '"}'
    quoted = shlex.quote(body)
    surfaces = {
        "redirect_flag": "vllm serve /m > --additional-config " + quoted,
        "redirect_value": "vllm serve /m --additional-config > " + quoted + " --model pre",
        "redirect_inline": "vllm serve /m > --additional-config=" + quoted,
        "redirect_assignment": "vllm serve /m > CFG=" + quoted,
        "option": ("--additional-config", "--other=" + "prefix" + body),
        "statement": "vllm serve /m --additional-config; " + quoted,
        "split_operand": ("--additional-config", '{"api\\u005f', 'key":"' + JSON_SECRET + '"}'),
    }
    found = set()
    _shell_taint(surfaces[shape], found)
    assert JSON_SECRET not in found and "pre" not in found


@pytest.mark.parametrize("form", ["argv", "env"])
@pytest.mark.parametrize("layout", ["json", "mapping"])
def test_deep_json_operand_discovery_survives_loader_limits(form, layout):
    body = ('{"api_key":"' + JSON_SECRET + '","data":' +
            "[" * 1100 + "0" + "]" * 1100 + "}")
    fields = {"command": ["vllm", "serve", "/m", "--additional-config", body]}
    if form == "env":
        fields = {"command": ["vllm", "serve", "/m"], "env": {"CFG": body}}
    _assert_operand_document(fields, layout)
