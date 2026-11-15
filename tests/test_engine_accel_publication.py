"""Publication, cache-precedence and build-memo regressions (engine-arena/2).

[Role] Keep the immutable-generation publication rules separate from the
  accelerator parity fixtures while still exercising the real setuptools path.
[Invariants] Every test uses the shared isolated cache and never touches the
  user's cache; a concurrent builder publishes distinct immutable inodes.
[Timeouts & Locks] Subprocess calls carry explicit timeouts; the publication
  race uses a bounded threading barrier.
[Side-Effect Policy] Writes only temporary artifacts under the isolated cache.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from mindie_sim.prediction.runtime import accel_bridge, accel_build
from mindie_sim.prediction.runtime.accel_tables import PROBE_GOLDEN


def test_cache_root_precedence(monkeypatch, tmp_path):
    for name in ("MINDIE_CACHE_HOME", "MINDIE_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache-home"))
    assert accel_build.cache_root() == (
        tmp_path / "cache-home" / "mindie" / "sim" / "engine")
    monkeypatch.delenv("MINDIE_CACHE_HOME")
    monkeypatch.setenv("MINDIE_DATA_HOME", str(tmp_path / "data"))
    assert accel_build.cache_root() == (
        tmp_path / "data" / "cache" / "mindie" / "sim" / "engine")
    monkeypatch.delenv("MINDIE_DATA_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert accel_build.cache_root() == (
        tmp_path / "xdg" / "mindie" / "sim" / "engine")


def test_force_rebuild_publishes_a_new_immutable_path():
    first = accel_build.build_accel(force=True)
    assert first["ok"], first.get("detail")
    second = accel_build.build_accel(force=True)
    assert second["ok"], second.get("detail")
    assert Path(first["cache_path"]).is_file()
    assert Path(second["cache_path"]).is_file()
    assert second["cache_path"] != first["cache_path"]


def test_build_accel_cli_reports_non_zero_on_failure(monkeypatch, capsys):
    from mindie_sim.__main__ import main

    monkeypatch.setenv("MINDIE_SIM_CC", "/nonexistent/cc")
    accel_build.reset_build_memo()
    code = main(["build-accel"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert code != 0


def test_check_only_never_claims_a_compiled_library(monkeypatch):
    monkeypatch.setenv("MINDIE_SIM_CC", "/nonexistent/cc")
    accel_build.reset_build_memo()
    status = accel_build.build_accel(check_only=True)
    assert status["ok"] is False
    assert status["library_ready"] is False
    assert status["toolchain_ready"] is False


def test_concurrent_publication_never_reuses_a_generation(monkeypatch):
    original = accel_build._validate
    barrier = threading.Barrier(2)

    def synchronized(path):
        result = original(path)
        barrier.wait(timeout=120)
        return result

    monkeypatch.setattr(accel_build, "_validate", synchronized)
    accel_build.reset_build_memo()
    results = {}

    def build(index):
        results[index] = accel_build.build_accel(force=True)

    threads = [threading.Thread(target=build, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    assert len(results) == 2
    assert all(status["ok"] for status in results.values())
    paths = [Path(results[index]["cache_path"]) for index in range(2)]
    assert paths[0] != paths[1]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert paths[0].stat().st_ino != paths[1].stat().st_ino


def test_build_memo_rechecks_the_current_artifact_and_identity(monkeypatch):
    accel_build.reset_build_memo()
    first = accel_build.build_accel()
    assert first["ok"] and Path(first["cache_path"]).is_file()
    monkeypatch.setattr(accel_build, "_compiler_identity",
                        lambda compiler, *, persist=False: "replaced-toolchain")
    second = accel_build.build_accel()
    assert second["ok"]
    assert second["cache_path"] != first["cache_path"]
    Path(second["cache_path"]).unlink()
    third = accel_build.build_accel()
    assert third["ok"] and Path(third["cache_path"]).is_file()
    assert third["cache_hit"] is False


def test_golden_probe_rejects_zero_cells(monkeypatch):
    class Probe:
        returncode = 0
        stderr = ""

        def __init__(self, cells):
            self.stdout = json.dumps({"v": 2, "rc": 0, "cells": cells})

    monkeypatch.setattr(accel_build.subprocess, "run",
                        lambda *_args, **_kwargs: Probe([0] * 16))
    ok, reason = accel_build._validate(Path("ignored"))
    assert not ok and reason == "abi_probe_mismatch"
    monkeypatch.setattr(
        accel_build.subprocess, "run",
        lambda *_args, **_kwargs: Probe([int(value) for value in PROBE_GOLDEN]))
    ok, reason = accel_build._validate(Path("ignored"))
    assert ok and reason == ""


def test_check_only_reports_python_fallback_without_a_library(monkeypatch, tmp_path):
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache"))
    accel_build.reset_build_memo()
    status = accel_build.build_accel(check_only=True)
    assert status["backend"] == "python_fallback"
    assert status["library_ready"] is False
    built = accel_build.build_accel()
    assert built["ok"]
    again = accel_build.build_accel(check_only=True)
    assert again["backend"] == "c_native"
    assert again["library_ready"] is True
    accel_build.reset_build_memo()
