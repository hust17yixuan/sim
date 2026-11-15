"""Strict padding-schema acceptance across the Python, MCP and CLI routes.

[Role]
  Reject malformed padding maps before any arithmetic, CAS publication or
  caller-output access, while preserving legitimate zero and positive counts.
[Invariants]
  Operator names are non-empty strings, operator specs are mappings and every
  supplied ``bucket_tokens`` is a finite non-negative non-boolean integer.
[Timeouts & Locks]
  CLI subprocesses have a bounded deadline; no blocking lock is introduced.
[Side-Effect Policy]
  Inputs, artifacts and caller files stay inside pytest temporary directories.
"""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mindie_profiler")
from mindie_profiler.events import ProfilerUsageError
from mindie_sim import diff_trace as diff

PACKET = {"total_ms": 1, "ops": [{"name": "Matmul", "duration_ms": 1}]}
SENTINEL = b"PADDING CALLER\x00\xff\n"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAD_BUCKET = [
    ("empty_list", []),
    ("empty_object", {}),
    ("nonempty_list", [4]),
    ("false", False),
    ("none", None),
    ("negative", -1),
    ("text", "4"),
]
INVALID = [("non_object_spec", {"Matmul": []})] + [
    (f"bucket_{name}", {"Matmul": {"bucket_tokens": value}})
    for name, value in BAD_BUCKET]
POSITIVE = [
    ("zero", {"Matmul": {"bucket_tokens": 0}}, 0),
    ("four", {"Matmul": {"bucket_tokens": 4}}, 4),
    ("omitted", {"Matmul": {}}, 0),
]


def strict_json(text):
    def reject(value):
        raise AssertionError("nonstandard JSON constant: " + value)

    payload = json.loads(text, parse_constant=reject)
    json.dumps(payload, allow_nan=False)
    return payload


def cli_inputs(tmp_path, padding):
    predicted = tmp_path / "predicted.json"
    captured = tmp_path / "captured.json"
    pad = tmp_path / "padding.json"
    predicted.write_text(json.dumps(PACKET), encoding="utf-8")
    captured.write_text(json.dumps(PACKET), encoding="utf-8")
    pad.write_text(json.dumps(padding), encoding="utf-8")
    return ["--predicted", str(predicted), "--captured", str(captured),
            "--padding", str(pad)]


def cli_env():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT.parent / "profiler")]),
                MINDIE_REPRO_SHORT_ROOT="cas")


def invoke(route, args, tmp_path):
    if route == "package":
        run = subprocess.run(
            [sys.executable, "-B", "-m", "mindie_sim", "diff-trace", *args],
            cwd=tmp_path, env=cli_env(), capture_output=True, text=True, timeout=60)
        return run.returncode, run.stdout, run.stderr
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = diff.main(args)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def destination_state(path):
    stat = path.stat()
    return path.read_bytes(), stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns


@pytest.mark.parametrize("label,padding", INVALID, ids=[row[0] for row in INVALID])
def test_raw_padding_messages(label, padding):
    message = ("padding operator specs must be objects" if label == "non_object_spec"
               else "bucket_tokens must be a non-negative integer")
    with pytest.raises(ValueError, match=message):
        diff._attribute_diff_snapshot(PACKET, PACKET, padding=copy.deepcopy(padding))


@pytest.mark.parametrize("padding", [
    {"": {"bucket_tokens": 0}},
    {1: {"bucket_tokens": 0}},
    {"Matmul": None},
    {"Matmul": {"bucket_tokens": 0}, "Other": "not-an-object"},
])
def test_operator_names_and_specs_are_never_dropped(padding):
    with pytest.raises(ValueError, match="padding operator"):
        diff._attribute_diff_snapshot(PACKET, PACKET, padding=padding)


@pytest.mark.parametrize("label,padding", INVALID, ids=[row[0] for row in INVALID])
@pytest.mark.parametrize("route", ["api", "mcp"])
def test_invalid_padding_rejected_before_publication(
        label, padding, route, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    with patch.object(diff, "_diff_root",
                      side_effect=AssertionError("invalid padding reached CAS")):
        if route == "api":
            with pytest.raises((ValueError, ProfilerUsageError)):
                diff.attribute_diff(PACKET, PACKET, padding=copy.deepcopy(padding))
        else:
            envelope = diff.call_mcp_tool(
                {"predicted": PACKET, "captured": PACKET,
                 "padding": copy.deepcopy(padding)})
            assert envelope["status"] == "invalid_request"
            assert not any(envelope[key] for key in
                           ("reconciled", "persisted", "retrievable"))
            assert envelope.get("artifact_id") is None
            assert envelope.get("artifact_path") is None
            strict_json(json.dumps(envelope, allow_nan=False))
    assert not (tmp_path / "cas").exists()


@pytest.mark.parametrize("label,padding", INVALID, ids=[row[0] for row in INVALID])
@pytest.mark.parametrize("mode", ["stdout", "absent", "sentinel"])
@pytest.mark.parametrize("route", ["helper", "package"])
def test_invalid_padding_cli_preserves_caller_output(
        label, padding, mode, route, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = cli_inputs(tmp_path, padding)
    destination = tmp_path / "output.json"
    if mode == "sentinel":
        destination.write_bytes(SENTINEL)
    if mode != "stdout":
        args += ["--output", str(destination)]
    before = destination_state(destination) if mode == "sentinel" else None
    if route == "helper":
        with patch.object(diff, "_diff_root",
                          side_effect=AssertionError("CAS accessed")):
            code, out, err = invoke(route, args, tmp_path)
    else:
        code, out, err = invoke(route, args, tmp_path)
    assert code == 2 and not out and "Traceback" not in err
    if route == "package":
        assert strict_json(err)["status"] == "failed"
    else:
        assert "diff-trace: invalid input" in err
    if before is not None:
        assert destination_state(destination) == before
    else:
        assert not destination.exists()
    assert not (tmp_path / "cas").exists()


@pytest.mark.parametrize("op_name", ["", None, 3])
def test_effective_rows_rejects_bad_operator_names(op_name):
    with pytest.raises(ValueError, match="op_name must be a non-empty string"):
        diff._effective_operator_rows_snapshot(op_name, 0, None)
    with pytest.raises(ProfilerUsageError):
        diff.effective_operator_rows(op_name, 0, None)


@pytest.mark.parametrize("label,bucket", BAD_BUCKET, ids=[row[0] for row in BAD_BUCKET])
def test_effective_rows_rejects_bad_buckets(label, bucket):
    with pytest.raises(ValueError,
                       match="bucket_tokens must be a non-negative integer"):
        diff._effective_operator_rows_snapshot("Matmul", bucket, None)
    with pytest.raises(ProfilerUsageError):
        diff.effective_operator_rows("Matmul", bucket, None)


def test_effective_rows_accepts_representable_buckets():
    for value, expected in ((0, 0), (4, 4), (4.0, 4)):
        row = diff.effective_operator_rows("Matmul", value, None)
        assert row["bucket_tokens"] == expected and row["valid"] is True


@pytest.mark.parametrize("label,padding,bucket", POSITIVE,
                         ids=[row[0] for row in POSITIVE])
@pytest.mark.parametrize("route", ["api", "mcp", "helper", "package"])
def test_valid_padding_accepted_on_every_route(
        label, padding, bucket, route, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    if route == "api":
        result = diff.attribute_diff(PACKET, PACKET, padding=copy.deepcopy(padding))
        assert result["status"] == "reconciled"
    elif route == "mcp":
        envelope = diff.call_mcp_tool(
            {"predicted": PACKET, "captured": PACKET,
             "padding": copy.deepcopy(padding)})
        assert envelope["reconciled"] and envelope["persisted"]
        assert envelope["retrievable"] and envelope["padding_valid"]
        data = Path(envelope["artifact_path"]).read_bytes()
        assert envelope["artifact_id"] == "sha256:" + hashlib.sha256(data).hexdigest()
        result = strict_json(data)
    else:
        destination = tmp_path / "output.json"
        args = cli_inputs(tmp_path, padding)
        code, out, err = invoke(route, args + ["--output", str(destination)], tmp_path)
        assert code == 0 and not out and not err
        result = strict_json(destination.read_bytes())
    assert result["padding"]["valid"] is True
    assert len(result["padding"]["charges"]) == 1
    charge = result["padding"]["charges"][0]
    assert charge["op_name"] == "Matmul" and charge["bucket_tokens"] == bucket
    assert charge["valid"] is True and result["issues"] == []
