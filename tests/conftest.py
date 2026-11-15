"""Shared isolated-runtime and isolated-cache fixtures for MindIE-Sim tests.

[Role] Keep user HOME/config/data/diagnostics/cache directories and accelerator
  build memos untouched across every test module.
[Invariants] `isolated_cache` provides one session-local `MINDIE_CACHE_HOME` and
  cleans it on teardown; `isolated_runtime` isolates per-test runtime paths.
[Timeouts & Locks] No threads, sleeps, or external I/O beyond temporary dirs.
[Side-Effect Policy] Writes only temporary directories created by pytest/tempfile.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindie_sim.observations.kernels import PROFILER_API_REQUIREMENT  # noqa: E402
from mindie_sim.prediction.runtime import accel_build  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def isolated_cache():
    base = tempfile.mkdtemp(prefix="nr-s3-accel-")
    names = (
        "MINDIE_CACHE_HOME",
        "MINDIE_DATA_HOME",
        "XDG_CACHE_HOME",
        "MINDIE_SIM_ACCEL",
        "MINDIE_SIM_ENGINE_BACKEND",
        "MINDIE_SIM_CC",
    )
    old = {name: os.environ.get(name) for name in names}
    os.environ["MINDIE_CACHE_HOME"] = base
    for name in names[1:]:
        os.environ.pop(name, None)
    accel_build.reset_build_memo()
    try:
        yield Path(base)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        accel_build.reset_build_memo()
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(root / "home"))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(root / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(root / "diagnostics"))


@pytest.fixture
def profiler_api():
    for module in ("mindie_profiler.normalized", "mindie_profiler.tensor_contracts"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in {"mindie_profiler", module}:
                raise
            pytest.skip(PROFILER_API_REQUIREMENT)
