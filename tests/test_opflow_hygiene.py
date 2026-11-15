"""Workspace hygiene checks for operator-flow Python sources and repository cleanliness.

[Role]
Enforce the <800-line limit, Unix LF, the four required docstring headers in
lines 1-30, TimeoutError-before-OSError ordering, no Harness SKILL.md, and zero
leftover Python/test caches.
[Invariants]
The scan is read-only and covers operator-flow modules, companion tests, and
repository-wide line-ending and cache cleanliness.
[Timeouts & Locks]
Small bounded reads only; no waits or locks.
[Side-Effect Policy]
Read-only; no files are created or removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ("[Role]", "[Invariants]", "[Timeouts & Locks]", "[Side-Effect Policy]")
CACHE_NAMES = {".ruff_cache", ".pytest_cache", "__pycache__"}
OPFLOW_TESTS = (
    'tests/test_aclgraph_multistream_and_superkernel.py',
    'tests/test_cannbot_aclgraph_distiller_and_mcp.py',
    'tests/test_cannbot_aclgraph_np_agreement.py',
    'tests/test_fix1_contracts.py',
    'tests/test_opflow_hygiene.py',
    'tests/test_opflow_pipeline_and_1c2v.py',
    'tests/test_opflow_sku_registry.py',
    'tests/test_r2_closure_contracts.py',
    'tests/test_r3_closure_contracts.py',
    'tests/test_unified_3stack_and_recurrence.py',
    'tests/test_v4_models.py',
    'tests/test_v4_schedules.py',
    'tests/test_v4_transfers.py',
)


def _python_files() -> list[Path]:
    files = sorted((ROOT / "mindie_sim" / "opflow").rglob("*.py"))
    for rel in OPFLOW_TESTS:
        candidate = ROOT / rel
        if candidate.exists():
            files.append(candidate)
    return files


def test_python_line_limit_and_lf() -> None:
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            raw_p = path.read_bytes()
            if b"\x00" not in raw_p[:8192] and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".gz", ".whl", ".so", ".sqlite", ".db", ".o"}:
                assert b"\r" not in raw_p, f"CR byte in {path}"
    for path in _python_files():
        raw = path.read_bytes()
        assert b"\r" not in raw, f"CR byte in {path}"
        lines = raw.count(b"\n") + int(bool(raw) and not raw.endswith(b"\n"))
        assert lines < 800, f"{path} has {lines} lines"


def test_required_headers_within_first_30_lines() -> None:
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        doc = ast.get_docstring(tree, clean=False)
        assert doc is not None, f"missing module docstring: {path}"
        doc_node = tree.body[0]
        head = "\n".join(source.splitlines()[doc_node.lineno - 1 : min(30, doc_node.end_lineno)])
        for marker in MARKERS:
            assert marker in doc and marker in head, (
                f"{path} missing module-docstring {marker} in lines 1-30"
            )


def test_timeout_before_broad_handlers() -> None:
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Try):
                continue
            guarded = False
            for handler in node.handlers:
                names = (
                    {x.id for x in ast.walk(handler.type) if isinstance(x, ast.Name)}
                    if handler.type
                    else {"bare"}
                )
                if names == {"TimeoutError"}:
                    assert len(handler.body) == 1 and isinstance(handler.body[0], ast.Raise)
                    assert handler.body[0].exc is None
                    guarded = True
                elif names & {
                    "Exception",
                    "BaseException",
                    "OSError",
                    "IOError",
                    "EnvironmentError",
                    "bare",
                }:
                    assert guarded, f"unguarded broad handler: {path}:{handler.lineno}"


def test_zero_harness_skills_and_caches() -> None:
    assert not any(path.name == "SKILL.md" for path in ROOT.rglob("SKILL.md"))
    for directory in ROOT.rglob("*"):
        if ".git" in directory.parts:
            continue
        assert directory.name not in CACHE_NAMES, f"cache left behind: {directory}"
        assert directory.suffix != ".pyc" and not directory.name.endswith(".egg-info")
