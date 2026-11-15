"""Finite diff results and CLI validation before caller-file publication.

[Role]
  Exercise Python, MCP, helper CLI and real package subprocess boundaries.
[Invariants]
  Finite representable launch costs succeed; invalid results publish nothing.
  Usage errors exit two and preserve existing or absent caller output files.
[Timeouts & Locks]
  Subprocesses have a deadline; exact TimeoutError instances pass through.
[Side-Effect Policy]
  All artifacts and subprocess working directories belong to pytest tmp_path.
"""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("mindie_profiler")
from mindie_profiler.events import ProfilerUsageError
from mindie_sim import __main__ as cli
from mindie_sim import diff_trace as diff
from mindie_sim import diff_trace_replay as engine


SENTINEL = b"CALLER CONTENT\x00\xff\n"
INVALID = (
    "predicted_array", "captured_array", "missing_window", "padding_array",
    "padding_bucket", "rows_object", "depth_70", "depth_1200", "malformed",
    "invalid_utf8", "surrogate", "large_decimal", "nan_leaf", "clock_overflow",
    "launch_overflow", "sum_overflow", "interval_overflow",
)
OVERFLOWS = ("clock_overflow", "launch_overflow", "sum_overflow", "interval_overflow")


def packets(kind="valid"):
    predicted = {"total_ms": 1, "ops": [{"name": "Matmul", "duration_ms": 1}]}
    captured = {"total_ms": 2, "ops": [{"name": "Matmul", "duration_ms": 2}]}
    padding = None
    if kind == "predicted_array":
        predicted = []
    elif kind == "captured_array":
        captured = []
    elif kind == "missing_window":
        predicted = {}
    elif kind == "padding_array":
        padding = []
    elif kind == "padding_bucket":
        padding = {"Matmul": {"bucket_tokens": {"invalid": 1}}}
    elif kind == "rows_object":
        captured["ops"] = {"invalid": 1}
    elif kind == "depth_70":
        nested = 0
        for _ in range(70):
            nested = [nested]
        captured["alignment"] = {"evidence": nested}
    elif kind == "nan_leaf":
        captured["alignment"] = {"evidence": float("nan")}
    elif kind == "surrogate":
        captured["alignment"] = {"evidence": "\ud800"}
    elif kind == "clock_overflow":
        captured.update(aic_mhz=1e308, nominal_aic_mhz=1e-308)
    elif kind in ("launch_overflow", "launch_finite"):
        captured.update(modeled_launch_overhead_us=1e308,
                        launch_count=1e308 if kind == "launch_overflow" else 2)
    elif kind == "sum_overflow":
        predicted["ops"] = [{"name": "Matmul", "duration_ms": 1e308,
                             "stream": stream} for stream in (0, 1)]
        predicted["total_ms"] = 1e308
    elif kind == "interval_overflow":
        captured["window_ms"] = [-1e308, 1e308]
        captured.pop("total_ms")
    return predicted, captured, padding


def input_files(tmp_path, kind):
    predicted, captured, padding = packets(kind)
    p, c = tmp_path / "predicted.json", tmp_path / "captured.json"
    p.write_text(json.dumps(predicted), encoding="utf-8")
    c.write_text(json.dumps(captured), encoding="utf-8")
    if kind == "depth_1200":
        c.write_text('{"total_ms":2,"alignment":' + '[' * 1200 + '0'
                     + ']' * 1200 + '}', encoding="utf-8")
    elif kind == "malformed":
        c.write_text('{"total_ms":', encoding="utf-8")
    elif kind == "invalid_utf8":
        c.write_bytes(b"\xff")
    elif kind == "large_decimal":
        c.write_text('{"total_ms":' + '9' * 5000 + '}', encoding="utf-8")
    args = ["--predicted", str(p), "--captured", str(c)]
    if padding is not None:
        path = tmp_path / "padding.json"
        path.write_text(json.dumps(padding), encoding="utf-8")
        args += ["--padding", str(path)]
    return args


def strict_payload(text):
    def reject(value):
        raise AssertionError("nonstandard JSON constant: " + value)
    payload = json.loads(text, parse_constant=reject)
    json.dumps(payload, allow_nan=False)
    return payload


def invoke(route, args, tmp_path, capsys):
    if route == "package":
        run = subprocess.run(
            [sys.executable, "-B", "-m", "mindie_sim", "diff-trace", *args],
            cwd=tmp_path, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")])),
            capture_output=True, text=True, timeout=30)
        return run.returncode, run.stdout, run.stderr
    try:
        code = diff.main(args)
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("kind", INVALID)
@pytest.mark.parametrize("mode", ["stdout", "absent", "sentinel"])
@pytest.mark.parametrize("route", ["helper", "package"])
def test_invalid_cli_preserves_output(kind, mode, route, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    args = input_files(tmp_path, kind)
    destination = tmp_path / "output.json"
    if mode == "sentinel":
        destination.write_bytes(SENTINEL)
    if mode != "stdout":
        args += ["--output", str(destination)]
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    code, out, err = invoke(route, args, tmp_path, capsys)
    assert code == 2 and not out and "Traceback" not in err
    if route == "package":
        assert strict_payload(err)["status"] == "failed"
    else:
        assert "diff-trace: invalid input" in err
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("kind", OVERFLOWS)
@pytest.mark.parametrize("route", ["python", "replay", "mcp"])
def test_nonfinite_computation_rejected_before_publication(kind, route, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    predicted, captured, _ = packets(kind)
    before = copy.deepcopy([predicted, captured])
    def no_publication():
        pytest.fail("rejected diff reached the CAS root")
    monkeypatch.setattr(diff, "_diff_root", no_publication)
    if route == "mcp":
        result = diff.call_mcp_tool({"predicted": predicted, "captured": captured})
        assert result["status"] == "invalid_request"
        assert not any(result[key] for key in ("persisted", "retrievable", "reconciled"))
        assert result.get("artifact_id") is None and result.get("artifact_path") is None
        json.dumps(result, allow_nan=False)
    else:
        module = diff if route == "python" else engine
        with pytest.raises((ValueError, ProfilerUsageError)):
            module.attribute_diff(predicted, captured)
    assert [predicted, captured] == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("route", ["python", "replay", "mcp", "helper", "package"])
def test_representable_launch_cost_is_finite(count, route, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    predicted, captured, _ = packets("launch_finite")
    captured["launch_count"] = count
    if route in ("python", "replay"):
        module = diff if route == "python" else engine
        result = module.attribute_diff(predicted, captured)
    elif route == "mcp":
        envelope = diff.call_mcp_tool({"predicted": predicted, "captured": captured})
        assert envelope["persisted"] and envelope["retrievable"] and envelope["reconciled"]
        result = strict_payload(Path(envelope["artifact_path"]).read_bytes())
    else:
        args = input_files(tmp_path, "launch_finite")
        (tmp_path / "captured.json").write_text(json.dumps(captured), encoding="utf-8")
        destination = tmp_path / "output.json"
        destination.write_bytes(SENTINEL)
        code, out, err = invoke(route, args + ["--output", str(destination)], tmp_path, capsys)
        assert code == 0 and not out and not err
        result = strict_payload(destination.read_bytes())
    assert result["status"] == "reconciled"
    actual = result["host_dispatch"]["modeled_launch_ms"]
    assert math.isfinite(actual) and actual == (1e308 / 1000.0) * count
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), {1, 2}, "\ud800"])
@pytest.mark.parametrize("route", ["helper", "package", "mcp", "persist"])
def test_serialization_fails_before_open(bad, route, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    args = input_files(tmp_path, "valid")
    destination = tmp_path / "output.json"
    destination.write_bytes(SENTINEL)
    payload = {"status": "reconciled", "unexpected": bad}
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    if route == "helper":
        monkeypatch.setattr(diff, "attribute_diff", lambda *a, **kw: payload)
        code, out, err = invoke(route, args + ["--output", str(destination)], tmp_path, capsys)
        assert code == 2 and not out and "invalid input" in err
    elif route == "package":
        monkeypatch.setattr(cli, "run", lambda args: payload)
        assert cli.main(["diff-trace", *args, "--output", str(destination)]) == 2
        streams = capsys.readouterr()
        assert not streams.out and strict_payload(streams.err)["status"] == "failed"
    elif route == "mcp":
        monkeypatch.setattr(diff, "_attribute_diff_snapshot", lambda *a, **kw: payload)
        predicted, captured, _ = packets()
        result = diff.call_mcp_tool({"predicted": predicted, "captured": captured})
        assert result["status"] == "invalid_request" and not result["persisted"]
    else:
        with pytest.raises((ValueError, TypeError)):
            diff._persist_diff(payload)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


class Deadline(TimeoutError):
    """A caller-owned timeout subclass."""


@pytest.mark.parametrize("error_type", [TimeoutError, Deadline])
@pytest.mark.parametrize("route", ["python", "replay", "mcp", "helper", "package"])
def test_exact_timeouts_preserved(route, error_type, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", "cas")
    args = input_files(tmp_path, "valid")
    destination = tmp_path / "output.json"
    destination.write_bytes(SENTINEL)
    sentinel = error_type("caller deadline")
    def fail(*args, **kwargs):
        raise sentinel
    monkeypatch.setattr(engine, "_compute_diff", fail)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    predicted, captured, _ = packets()
    with pytest.raises(error_type) as caught:
        if route == "mcp":
            diff.call_mcp_tool({"predicted": predicted, "captured": captured})
        elif route == "python":
            diff.attribute_diff(predicted, captured)
        elif route == "replay":
            engine.attribute_diff(predicted, captured)
        else:
            command = args + ["--output", str(destination)]
            (cli.main(["diff-trace", *command]) if route == "package"
             else diff.main(command))
    assert caught.value is sentinel
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_package_retains_file_size_budget(tmp_path, capsys):
    args = input_files(tmp_path, "valid")
    with (tmp_path / "captured.json").open("wb") as stream:
        stream.truncate(50 * 1024**2 + 1)
    output = tmp_path / "output.json"
    output.write_bytes(SENTINEL)
    assert cli.main(["diff-trace", *args, "--output", str(output)]) == 2
    assert "metadata_input_exceeds_50_MiB" in strict_payload(capsys.readouterr().err)["error"]
    assert output.read_bytes() == SENTINEL
