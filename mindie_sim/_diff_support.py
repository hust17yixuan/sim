"""Optional mindie_profiler ingress helpers with standalone wheel-install fallbacks.

[Role]
Import `ProfilerUsageError`, `snapshot`, `snapshot_call`, and `load_cli_json`
from `mindie_profiler` when co-installed, or provide equivalent bounded pure
Python fallbacks when `mindie_sim` runs from a standalone wheel.
[Invariants]
Co-installed workspaces bind the exact `mindie_profiler` classes and helpers;
standalone wheel environments never fail package import and enforce the same
64-level nesting, 1M-member budget, and TimeoutError pass-through rules.
[Timeouts & Locks]
Pure in-memory JSON and container validation; `TimeoutError` always propagates.
[Side-Effect Policy]
Read-only input copying and bounded UTF-8 file decoding; mutates no caller state.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import json
import math
from pathlib import Path

try:
    from mindie_profiler.events import ProfilerUsageError
    from mindie_profiler.repro_snapshot import snapshot, snapshot_call
    from mindie_profiler.repro_values import load_cli_json
except ModuleNotFoundError:

    class ProfilerUsageError(RuntimeError):
        """Standalone fallback usage error when mindie_profiler is absent."""

    def _literal(item):
        if item is None or isinstance(item, (bool, str)):
            return item
        if type(item) is int:
            if item.bit_length() > 4096:
                raise ProfilerUsageError("integer exceeds bounded limit")
            return item
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            val = float(item)
            if not math.isfinite(val):
                raise ProfilerUsageError("non-finite numeric value")
            return int(item) if isinstance(item, int) else val
        raise ProfilerUsageError("unsupported scalar leaf")

    def snapshot(value, *, _sets=()):
        budget, memo, active = [0], {}, set()

        def copy(item, depth):
            is_set = id(item) in _sets and isinstance(item, (set, frozenset))
            if not (isinstance(item, (Mapping, list, tuple)) or is_set):
                return _literal(item), 0
            if depth >= 64:
                raise ProfilerUsageError("input container nesting exceeds 64 levels")
            ident = id(item)
            if ident in active:
                raise ProfilerUsageError("input containers must not contain cycles")
            if ident in memo:
                _owner, res, cost, height = memo[ident]
                if depth + height > 64 or budget[0] + cost > 1_000_000:
                    raise ProfilerUsageError("expanded container budget exceeded")
                budget[0] += cost
                return res, height
            active.add(ident)
            begin, height = budget[0], 1
            mapping = isinstance(item, Mapping)
            res = {} if mapping else []
            items_iter = item.items() if mapping else ((None, x) for x in item)
            for k, member in items_iter:
                budget[0] += 1
                if budget[0] > 1_000_000:
                    raise ProfilerUsageError("expanded container budget exceeded")
                if mapping and type(k) is not str:
                    raise ProfilerUsageError("mapping keys must be exact str")
                child, ch = copy(member, depth + 1)
                height = max(height, ch + 1)
                if mapping:
                    res[k] = child
                else:
                    res.append(child)
            active.remove(ident)
            memo[ident] = (item, res, budget[0] - begin, height)
            return res, height

        try:
            return copy(value, 0)[0]
        except ProfilerUsageError:
            raise
        except TimeoutError:
            raise
        except Exception as exc:
            raise ProfilerUsageError("invalid input container protocol") from exc

    def snapshot_call(function, *fields, set_fields=()):
        code = function.__code__
        pos_names = code.co_varnames[: code.co_argcount]

        @wraps(function)
        def public(*args, **kwargs):
            bound = dict(zip(pos_names, args))
            bound.update(kwargs)
            selected = {name: bound[name] for name in fields if name in bound}
            set_ids = {id(bound[name]) for name in set_fields if name in bound}
            snapped = snapshot(selected, _sets=set_ids)
            new_args = [snapped.get(n, v) for n, v in zip(pos_names, args)]
            if len(args) > len(pos_names):
                new_args.extend(args[len(pos_names) :])
            new_kw = {k: snapped.get(k, v) for k, v in kwargs.items()}
            try:
                return function(*new_args, **new_kw)
            except TimeoutError:
                raise
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
                raise ProfilerUsageError("invalid container input") from exc

        return public

    def load_cli_json(path):
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ProfilerUsageError("invalid UTF-8 JSON input") from exc
        if len(raw) > 50 * 1024**2:
            raise ProfilerUsageError("JSON input exceeds budget")

        def bad_const(token):
            raise ProfilerUsageError(f"non-finite JSON constant: {token}")

        try:
            parsed = json.loads(raw, parse_constant=bad_const)
        except (ValueError, RecursionError) as exc:
            raise ProfilerUsageError(f"invalid JSON input: {exc}") from exc
        return snapshot(parsed)
