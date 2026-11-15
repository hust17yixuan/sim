"""Private exact-period detection for the existing finite pipeline Engine.

[Role] Detect a genuinely repeated *settled* Engine structure and return a
  precise, bounded skip description.  The consumer (T04's Engine adapter) owns
  every mutation.  This module imports no Engine, scheduling or program code.
[Invariants] A cycle requires (a) exact structural identity of the supplied
  normalized state, (b) an identical active cohort/phase class, (c) two
  consecutive equal measured periods (each aggregated from real intervals),
  and (d) the same per-run issue, retire and sequence strides.  Phase remaining is only an upper bound on the
  number of cycles; it is never part of the structural key.  Floating
  rounding is tolerated only at a few ulps for measured time/service values;
  structural state values are compared deeply and exactly.  A cheap structural
  fingerprint may only fast-reject; a match always uses exact identity or
  full equality.  Mutable/opaque external state keeps the full validation path.
[Timeouts & Locks] Pure CPU, bounded observation history, no threads, sleeps,
  I/O, retries or engine access.  Unsafe or inconsistent input is rejected
  with ``PeriodicContractError`` rather than silently becoming a skip.
[Side-Effect Policy] ``CycleDetector.observe`` mutates only this detector's
  private observation history.  Returned records are frozen.  The caller must
  explicitly apply a ``CycleSkip`` and remains responsible for real event,
  allocation, peak, service, busy and sequence accounting.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
import math
import sys
import time
from types import MappingProxyType
from typing import Any
import weakref

__all__ = [
    "CycleCorrespondence",
    "CycleDetector",
    "CycleSkip",
    "CanonicalState",
    "Cursor",
    "IntervalDelta",
    "PeriodicContractError",
    "RunCorrespondence",
    "SettledFrontier",
    "StateInterner",
    "TemplateRef",
    "WindowTile",
    "metrics",
    "reset_metrics",
    "reset_state_caches",
    "structural_difference",
    "structural_equal",
]


class PeriodicContractError(ValueError):
    """The caller supplied an unsafe or incomplete periodic contract."""


# Only floating measurement rounding may be tolerated.  Structural values never
# use this tolerance; they go through ``structural_equal`` below.
_ULP_SLOP = 8
_MAX_STATE_DEPTH = 64
# A canonical tree adds a tagged tuple around every input container, so its
# depth is bounded by roughly twice the validated input depth.
_MAX_CANONICAL_DEPTH = 2 * _MAX_STATE_DEPTH + 8

# Bounded process-local structural template cache.  Old generations are evicted;
# the cache never grows with observation history.  Exact template equality is
# required even on a fingerprint hit.
_DEFAULT_INTERNER_LIMIT = 64
# Conservative local shares of the host cache envelope.  A host (T04) may pass
# the actual remaining byte budget explicitly; the helper never assumes that a
# whole 24/48 MiB host scope is available to one cache.
_DEFAULT_INTERNER_BYTES = 8 * 1024 * 1024
_DEFAULT_HISTORY_BYTES = 8 * 1024 * 1024
_TEMPLATE_REPR_LIMIT = 256
_TEMPLATE_OVERHEAD = 256
_FRAME_OVERHEAD = 128
_CURSOR_OVERHEAD = 96
_MAPPING_ENTRY_OVERHEAD = 64
_INTERVAL_OVERHEAD = 128
_OBSERVATION_OVERHEAD = 64
_STATE_METRICS = {
    "canonical_frames": 0,
    "raw_frames": 0,
    "template_validations": 0,
    "template_nodes": 0,
    "template_builds": 0,
    "template_build_nodes": 0,
    "template_hits": 0,
    "template_hit_nodes": 0,
    "template_probes": 0,
    "dynamic_validations": 0,
    "dynamic_nodes": 0,
    "raw_nodes": 0,
}
_STATE_VERSION = 0


def reset_metrics() -> None:
    """Reset process-local structural validation counters (diagnostics only)."""
    for key in _STATE_METRICS:
        _STATE_METRICS[key] = 0


def metrics() -> dict:
    """Return a copy of the structural validation counters."""
    return dict(_STATE_METRICS)


def _next_state_version() -> int:
    global _STATE_VERSION
    _STATE_VERSION += 1
    return _STATE_VERSION


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    return value


def _require_float(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    allow_infinite: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    number = float(value)
    if math.isnan(number) or (math.isinf(number) and not allow_infinite):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    if positive and number <= 0:
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    return number


def _require_run_id(value: Any, *, name: str = "run_id") -> Hashable:
    # The frozen interface uses ``complete.sequence`` as the run id.  Keeping
    # this to concrete scalar keys avoids an accidental opaque/unhashable id.
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    return value


def _check_structural(
    value: Any,
    *,
    path: str = "normalized_state",
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    """Reject opaque state so identity cannot silently degrade to a digest."""
    if counter is not None:
        counter[0] += 1
    if depth > _MAX_STATE_DEPTH:
        raise PeriodicContractError(f"periodic_state_too_deep:{path}")
    if value is None or isinstance(value, (bool, str, bytes)):
        return
    if _is_int(value):
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise PeriodicContractError(f"periodic_state_nan:{path}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _check_structural(item, path=f"{path}[{index}]", depth=depth + 1, counter=counter)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_structural(key, path=f"{path}.key", depth=depth + 1, counter=counter)
            _check_structural(item, path=f"{path}[{key!r}]", depth=depth + 1, counter=counter)
        return
    if isinstance(value, (set, frozenset)):
        for item in value:
            _check_structural(item, path=f"{path}{{item}}", depth=depth + 1, counter=counter)
        return
    raise PeriodicContractError(f"periodic_state_opaque:{path}:{type(value).__name__}")


def _numeric_close(first: float, second: float) -> bool:
    """Equality for measured floats, allowing only a few representation ulps."""
    if first == second:
        return True
    if not (math.isfinite(first) and math.isfinite(second)):
        return False
    scale = max(abs(first), abs(second))
    return abs(first - second) <= _ULP_SLOP * math.ulp(scale)


def _structural_equal(first: Any, second: Any) -> bool:
    """Exact typed structural equality.

    Python ``==`` is deliberately not used as a shortcut: ``True``, ``1`` and
    ``1.0`` are distinct typed values, and the same mutable container may be
    observed at two different times.  Containers are compared recursively by
    exact type/content; sets and mappings are matched without relying on
    Python hash/equality merging.
    """
    if type(first) is not type(second):
        return False
    kind = type(first)
    if first is None:
        return True
    if kind is bool or kind is int or kind is str or kind is bytes:
        return first == second
    if kind is float:
        if math.isnan(first) or math.isnan(second):
            return False
        return first == second
    if isinstance(first, type):
        return first is second
    if kind is tuple or kind is list:
        return len(first) == len(second) and all(
            _structural_equal(left, right)
            for left, right in zip(first, second))
    if isinstance(first, Mapping):
        if len(first) != len(second):
            return False
        unmatched = list(second.items())
        for key, value in first.items():
            for index, (other_key, other_value) in enumerate(unmatched):
                if (_structural_equal(key, other_key)
                        and _structural_equal(value, other_value)):
                    del unmatched[index]
                    break
            else:
                return False
        return True
    if isinstance(first, (set, frozenset)):
        if len(first) != len(second):
            return False
        unmatched = list(second)
        for item in first:
            for index, other in enumerate(unmatched):
                if _structural_equal(item, other):
                    del unmatched[index]
                    break
            else:
                return False
        return True
    return False


def structural_equal(first: Any, second: Any) -> bool:
    """Public, exact structural comparison used by the detector and tests."""
    _check_structural(first)
    _check_structural(second)
    return _structural_equal(first, second)


def _first_difference(first: Any, second: Any, path: str = "state") -> str | None:
    """Return the first unequal structural component for adapter diagnostics."""
    if first is second:
        return None
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not (isinstance(first, Mapping) and isinstance(second, Mapping)):
            return f"{path}:mapping_type"
        if first.keys() != second.keys():
            left = sorted(map(repr, first.keys() - second.keys()))
            right = sorted(map(repr, second.keys() - first.keys()))
            return f"{path}.keys:left_only={left[:4]}:right_only={right[:4]}"
        for key in first:
            found = _first_difference(first[key], second[key], f"{path}[{key!r}]")
            if found is not None:
                return found
        return None
    if isinstance(first, (set, frozenset)) or isinstance(second, (set, frozenset)):
        if not (isinstance(first, (set, frozenset)) and isinstance(second, (set, frozenset))):
            return f"{path}:set_type"
        if first != second:
            return f"{path}:set_value"
        return None
    if type(first) is not type(second):
        return f"{path}:type:{type(first).__name__}!={type(second).__name__}"
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            return f"{path}.length:{len(first)}!={len(second)}"
        for index, (left, right) in enumerate(zip(first, second)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if first != second:
        return f"{path}:{first!r}!={second!r}"
    return None


def structural_difference(first: Any, second: Any) -> str | None:
    """Public first-difference diagnostic; values are already validated."""
    _check_structural(first)
    _check_structural(second)
    return _first_difference(first, second)


def _bounded_repr(value: Any) -> str:
    """Bound diagnostic text; display text is never used as identity."""
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - builtin state only
        return "<unrepresentable>"
    if len(text) > _TEMPLATE_REPR_LIMIT:
        return text[:_TEMPLATE_REPR_LIMIT] + "..."
    return text


def _measure_retained(value: Any, *, seen: set[int] | None = None) -> int:
    """Approximate bytes owned by a validated value graph.

    Shared sub-objects inside one observation are counted once.  The result is
    a retention lower bound used for cache admission, not an RSS measurement.
    """
    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    try:
        size = sys.getsizeof(value)
    except TypeError:  # pragma: no cover - builtin state only
        size = 0
    kind = type(value)
    if kind is tuple or kind is list:
        for item in value:
            size += _measure_retained(item, seen=seen)
    elif kind is set or kind is frozenset:
        for item in value:
            size += _measure_retained(item, seen=seen)
    elif isinstance(value, Mapping):
        # ``MappingProxyType`` only reports its own small wrapper size, so the
        # payload/table share is charged per entry here.  Keys and values are
        # measured once per shared identity.
        size += _MAPPING_ENTRY_OVERHEAD * len(value)
        for key, item in value.items():
            size += _measure_retained(key, seen=seen)
            size += _measure_retained(item, seen=seen)
    return size


def _template_cost(template: tuple, template_repr: str) -> int:
    """Admission cost for one cached template (tree + index/reference bookkeeping)."""
    return _measure_retained(template) + len(template_repr) + _TEMPLATE_OVERHEAD


def _check_canonical_snapshot(
    value: Any, *, path: str = "template", depth: int = 0,
) -> None:
    """Reject anything that is not an internally tagged immutable type tree.

    ``from_validated`` is a public factory for an already validated snapshot.
    Rechecking its tags/container identities here keeps an external caller from
    bypassing the interner's type-exact boundary by supplying a raw builtin tree
    in the snapshot slot.
    """
    if depth > _MAX_CANONICAL_DEPTH:
        raise PeriodicContractError(f"periodic_snapshot_too_deep:{path}")
    if type(value) is not tuple or not value:
        raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
    tag = value[0]
    if type(tag) is not str:
        raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
    if tag == "none":
        if len(value) != 1:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        return
    if tag in {"bool", "int", "float", "str", "bytes"}:
        if len(value) != 2:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        payload = value[1]
        expected = {
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "bytes": bytes,
        }[tag]
        if type(payload) is not expected:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        if tag == "float" and math.isnan(payload):
            raise PeriodicContractError(f"periodic_snapshot_nan:{path}")
        return
    if tag in {"tuple", "list"}:
        if len(value) != 2 or type(value[1]) is not tuple:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        for index, item in enumerate(value[1]):
            _check_canonical_snapshot(
                item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if tag == "dict" or tag == "mapping":
        if tag == "dict":
            if len(value) != 2 or type(value[1]) is not tuple:
                raise PeriodicContractError(
                    f"periodic_template_snapshot_invalid:{path}")
            items = value[1]
        else:
            # The real emitted class is retained for exact type identity.  Its
            # metaclass may legitimately be ``ABCMeta`` (e.g. UserDict), so the
            # check is "class object and a concrete Mapping subclass", never
            # ``type(cls) is type``.  A forged tag cannot smuggle in ``dict``
            # or an unrelated/abstract class.
            cls = value[1] if len(value) == 3 else None
            if (
                len(value) != 3
                or not isinstance(cls, type)
                or cls is dict
                or not issubclass(cls, Mapping)
                or getattr(cls, "__abstractmethods__", ())
                or type(value[2]) is not tuple
            ):
                raise PeriodicContractError(
                    f"periodic_template_snapshot_invalid:{path}")
            items = value[2]
        for index, pair in enumerate(items):
            if type(pair) is not tuple or len(pair) != 2:
                raise PeriodicContractError(
                    f"periodic_template_snapshot_invalid:{path}[{index}]")
            _check_canonical_snapshot(
                pair[0], path=f"{path}[{index}].key", depth=depth + 1)
            _check_canonical_snapshot(
                pair[1], path=f"{path}[{index}].value", depth=depth + 1)
        return
    if tag in {"set", "frozenset"}:
        if len(value) != 2 or type(value[1]) is not tuple:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        for index, item in enumerate(value[1]):
            _check_canonical_snapshot(
                item, path=f"{path}{{item}}[{index}]", depth=depth + 1)
        return
    raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")


def _freeze_snapshot(
    value: Any, *, path: str = "snapshot", depth: int = 0,
    counter: list[int] | None = None, sizes: list[int] | None = None,
) -> tuple:
    """Deep-freeze a validated value into an immutable typed canonical form.

    Every container carries an explicit type tag; bool/int/float are separate
    tags.  The returned tree contains only tuples and immutable scalars, so a
    later caller mutation cannot rewrite history and canonical ``==`` cannot
    merge Python-equal but differently typed values.

    ``sizes`` optionally accumulates the real ``sys.getsizeof`` total of the
    produced tree.  The tree is being built anyway, so charging each new node
    there costs one C call per node and lets a historical frame report its
    retained bytes without a second recursive pass over the same nodes.
    """
    if depth > _MAX_STATE_DEPTH:
        raise PeriodicContractError(f"periodic_snapshot_too_deep:{path}")
    if counter is not None:
        counter[0] += 1
    kind = type(value)
    if value is None:
        result = ("none",)
    elif kind is bool:
        result = ("bool", value)
    elif kind is int:
        result = ("int", value)
    elif kind is float:
        if math.isnan(value):
            raise PeriodicContractError(f"periodic_snapshot_nan:{path}")
        result = ("float", value)
    elif kind is str:
        result = ("str", value)
    elif kind is bytes:
        result = ("bytes", value)
    elif kind is tuple:
        result = ("tuple", tuple(
            _freeze_snapshot(item, path=f"{path}[{index}]", depth=depth + 1,
                             counter=counter, sizes=sizes)
            for index, item in enumerate(value)))
    elif kind is list:
        result = ("list", tuple(
            _freeze_snapshot(item, path=f"{path}[{index}]", depth=depth + 1,
                             counter=counter, sizes=sizes)
            for index, item in enumerate(value)))
    elif isinstance(value, Mapping):
        items = [
            (_freeze_snapshot(key, path=f"{path}.key", depth=depth + 1,
                              counter=counter, sizes=sizes),
             _freeze_snapshot(item, path=f"{path}[{key!r}]", depth=depth + 1,
                              counter=counter, sizes=sizes))
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: repr(pair[0]))
        if kind is dict:
            result = ("dict", tuple(items))
        else:
            # Keep the real mapping class (identity, not a display name) so two
            # same-repr mapping types can never share a template key.
            result = ("mapping", kind, tuple(items))
    elif kind is set:
        result = ("set", tuple(sorted(
            (_freeze_snapshot(item, path=f"{path}{{item}}", depth=depth + 1,
                              counter=counter, sizes=sizes)
             for item in value),
            key=repr)))
    elif kind is frozenset:
        result = ("frozenset", tuple(sorted(
            (_freeze_snapshot(item, path=f"{path}{{item}}", depth=depth + 1,
                              counter=counter, sizes=sizes)
             for item in value),
            key=repr)))
    else:
        raise PeriodicContractError(
            f"periodic_snapshot_opaque:{path}:{kind.__name__}")
    if sizes is not None:
        sizes[0] += sys.getsizeof(result)
    return result


def _check_engine_dynamic(value: Any, counter: list[int] | None = None) -> None:
    """Fast restricted validator for the known Engine dynamic residue.

    This validates every changed remaining/waiting/release/start field and the
    small runs/allocations/pools/releases tail.  It is intentionally strict:
    only the immutable scalar/container shapes emitted by the T04 adapter pass.
    Any other shape keeps the generic recursive validator.
    """
    if type(value) is not tuple or len(value) != 5:
        raise PeriodicContractError("periodic_engine_dynamic_shape_invalid")
    task_dynamic, runs, allocations, pools, releases = value
    for part in (task_dynamic, runs, allocations, pools, releases):
        if type(part) is not tuple:
            raise PeriodicContractError("periodic_engine_dynamic_part_invalid")
    if counter is not None:
        counter[0] += 5
    for entry in task_dynamic:
        if type(entry) is not tuple or len(entry) != 4:
            raise PeriodicContractError("periodic_engine_task_dynamic_invalid")
        remaining, waiting, release, start = entry
        if type(remaining) is not tuple:
            raise PeriodicContractError("periodic_engine_remaining_invalid")
        for pair in remaining:
            if type(pair) is not tuple or len(pair) != 2:
                raise PeriodicContractError("periodic_engine_remaining_pair_invalid")
            key, amount = pair
            if type(key) is not str:
                raise PeriodicContractError("periodic_engine_remaining_key_invalid")
            if type(amount) is not int and type(amount) is not float:
                raise PeriodicContractError("periodic_engine_remaining_value_invalid")
            if type(amount) is float and not math.isfinite(amount):
                raise PeriodicContractError("periodic_engine_remaining_value_invalid")
            if counter is not None:
                counter[0] += 2
        if type(waiting) is not int or waiting < 0:
            raise PeriodicContractError("periodic_engine_waiting_invalid")
        for offset in (release, start):
            if offset is None:
                continue
            if type(offset) is not int and type(offset) is not float:
                raise PeriodicContractError("periodic_engine_offset_invalid")
            if type(offset) is float and not math.isfinite(offset):
                raise PeriodicContractError("periodic_engine_offset_invalid")
        if counter is not None:
            counter[0] += 4
    for part in (runs, allocations, pools, releases):
        _check_structural(part, path="engine_dynamic_tail", counter=counter)


def _freeze_engine_dynamic(
    value: Any, counter: list[int] | None = None,
    sizes: list[int] | None = None,
) -> tuple:
    """Validate and freeze the known engine dynamic residue in one pass.

    ``sizes`` accumulates the real retained size of the frozen tree as it is
    produced.  A historical engine frame therefore knows its byte cost without
    a second recursive ``sys.getsizeof`` walk over the same nodes.
    """
    if type(value) is not tuple or len(value) != 5:
        raise PeriodicContractError("periodic_engine_dynamic_shape_invalid")
    task_dynamic, runs, allocations, pools, releases = value
    for part in (task_dynamic, runs, allocations, pools, releases):
        if type(part) is not tuple:
            raise PeriodicContractError("periodic_engine_dynamic_part_invalid")
    if counter is not None:
        counter[0] += 5
    def charged(node):
        # Charge the node we just built, in the same pass.  This is the real
        # sys.getsizeof total of the retained frozen tree, not an estimate.
        if sizes is not None:
            sizes[0] += sys.getsizeof(node)
        return node

    frozen_tasks = []
    for entry in task_dynamic:
        if type(entry) is not tuple or len(entry) != 4:
            raise PeriodicContractError("periodic_engine_task_dynamic_invalid")
        remaining, waiting, release, start = entry
        if type(remaining) is not tuple:
            raise PeriodicContractError("periodic_engine_remaining_invalid")
        frozen_remaining = []
        for pair in remaining:
            if type(pair) is not tuple or len(pair) != 2:
                raise PeriodicContractError("periodic_engine_remaining_pair_invalid")
            key, amount = pair
            if type(key) is not str:
                raise PeriodicContractError("periodic_engine_remaining_key_invalid")
            if type(amount) is not int and type(amount) is not float:
                raise PeriodicContractError("periodic_engine_remaining_value_invalid")
            if type(amount) is float and not math.isfinite(amount):
                raise PeriodicContractError("periodic_engine_remaining_value_invalid")
            amount_tag = "float" if type(amount) is float else "int"
            frozen_remaining.append(charged(
                ("tuple", (charged(("str", key)),
                           charged((amount_tag, amount))))))
            if counter is not None:
                counter[0] += 2
        if type(waiting) is not int or waiting < 0:
            raise PeriodicContractError("periodic_engine_waiting_invalid")
        frozen_offsets = []
        for offset in (release, start):
            if offset is None:
                frozen_offsets.append(charged(("none",)))
                continue
            if type(offset) is not int and type(offset) is not float:
                raise PeriodicContractError("periodic_engine_offset_invalid")
            if type(offset) is float and not math.isfinite(offset):
                raise PeriodicContractError("periodic_engine_offset_invalid")
            offset_tag = "float" if type(offset) is float else "int"
            frozen_offsets.append(charged((offset_tag, offset)))
        frozen_tasks.append(charged(
            ("tuple", (
                ("tuple", charged(tuple(frozen_remaining))),
                charged(("int", waiting)),
                frozen_offsets[0],
                frozen_offsets[1],
            ))))
        if counter is not None:
            counter[0] += 4
    frozen_tail = tuple(
        _freeze_snapshot(part, path="engine_dynamic_tail", counter=counter,
                         sizes=sizes)
        for part in (runs, allocations, pools, releases))
    # Charge exactly the reference-only containers this function creates on
    # the retained path: the task tuple, its tag wrapper, the 1-tuple head, the
    # concatenated body and the outer result.  Leaves were charged above.
    tasks_tuple = charged(tuple(frozen_tasks))
    tasks_wrapper = charged(("tuple", tasks_tuple))
    body = charged((tasks_wrapper,) + frozen_tail)
    result = charged(("tuple", body))
    return result


def _engine_scalar_key(value: Any, counter: list[int], *, path: str) -> tuple:
    """Type-exact scalar key used by the restricted Engine static schema."""
    counter[0] += 1
    kind = type(value)
    if value is None:
        return ("none",)
    if kind is bool:
        return ("bool", value)
    if kind is int:
        return ("int", value)
    if kind is float:
        if not math.isfinite(value):
            raise PeriodicContractError(
                f"periodic_engine_static_invalid:{path}")
        return ("float", value)
    if kind is str:
        return ("str", value)
    if kind is bytes:
        return ("bytes", value)
    raise PeriodicContractError(f"periodic_engine_static_invalid:{path}")


def _engine_caps_key(value: Any, counter: list[int]) -> tuple:
    if type(value) is not tuple:
        raise PeriodicContractError("periodic_engine_static_caps_invalid")
    result = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise PeriodicContractError("periodic_engine_static_caps_invalid")
        name, amount = pair
        if type(name) is not str:
            raise PeriodicContractError("periodic_engine_static_caps_invalid")
        if type(amount) is int:
            amount_key = ("int", amount)
        elif type(amount) is float:
            if not math.isfinite(amount):
                raise PeriodicContractError(
                    "periodic_engine_static_caps_invalid")
            amount_key = ("float", amount)
        else:
            raise PeriodicContractError("periodic_engine_static_caps_invalid")
        result.append((("str", name), amount_key))
        counter[0] += 2
    return tuple(result)


def _engine_tuple_key(value: Any, counter: list[int]) -> tuple:
    """Exact key for small tuple identity/placement fields of the static schema."""
    if type(value) is not tuple:
        raise PeriodicContractError("periodic_engine_static_key_invalid")
    result = []
    for item in value:
        if type(item) is tuple:
            result.append(_engine_tuple_key(item, counter))
        elif type(item) is int:
            result.append(("int", item))
            counter[0] += 1
        elif type(item) is str:
            result.append(("str", item))
            counter[0] += 1
        else:
            raise PeriodicContractError("periodic_engine_static_key_invalid")
    counter[0] += 1
    return tuple(result)


def _engine_static_key(static: Any, counter: list[int]) -> tuple:
    """Build the reusable exact lookup key for the restricted Engine static part.

    It validates the known schema directly and tags every scalar with its real
    builtin type, so equal-looking ``bool``/``int``/``float`` values can never
    share a key.  A hit therefore needs no canonical-tree rebuild: the cached
    fixed tree is returned as-is.
    """
    if type(static) is not tuple:
        raise PeriodicContractError("periodic_engine_static_shape_invalid")
    result = []
    for task in static:
        if type(task) is not tuple or len(task) != 6:
            raise PeriodicContractError("periodic_engine_static_task_invalid")
        key, priority, caps, placement, coupled, owner = task
        if type(priority) is not int or type(coupled) is not bool:
            raise PeriodicContractError("periodic_engine_static_task_invalid")
        result.append((
            _engine_tuple_key(key, counter),
            priority,
            _engine_caps_key(caps, counter),
            _engine_tuple_key(placement, counter),
            coupled,
            _engine_scalar_key(owner, counter, path="owner"),
        ))
        counter[0] += 4
    return tuple(result)


@dataclass(frozen=True)
class TemplateRef:
    """Immutable validated invariant snapshot with a generation-scoped id.

    ``template`` is a deep-frozen typed canonical tree (lists become tagged
    immutable tuples, dicts/sets/mapping types are tagged and ordered
    deterministically).  Identity and matching use this tree, never display
    text.  ``template_repr`` is a bounded diagnostic preview retained for
    compatibility; it is deliberately not a comparison key.
    A direct constructor validates/freezes a caller builtin tree;
    ``from_validated`` accepts only an already canonical internal tree.
    """

    template_id: int
    generation: Hashable
    fingerprint: int
    template: tuple
    template_repr: str = ""
    nodes: int = 0
    retained_bytes: int = 0

    def __post_init__(self) -> None:
        _require_int("template_id", self.template_id, minimum=1)
        if not isinstance(self.generation, (int, str)) or isinstance(
                self.generation, bool):
            raise PeriodicContractError(
                f"periodic_template_generation_invalid:{self.generation!r}")
        if not _is_int(self.fingerprint):
            raise PeriodicContractError(
                f"periodic_template_fingerprint_invalid:{self.fingerprint!r}")
        if not isinstance(self.template, tuple):
            raise PeriodicContractError("periodic_template_must_be_tuple")
        if not isinstance(self.template_repr, str):
            raise PeriodicContractError("periodic_template_repr_invalid")
        counter = [0]
        _check_structural(self.template, path="template", counter=counter)
        representation = (
            self.template_repr[:_TEMPLATE_REPR_LIMIT]
            if self.template_repr else "")
        snapshot = _freeze_snapshot(self.template, path="template")
        object.__setattr__(self, "template", snapshot)
        object.__setattr__(self, "template_repr", representation)
        object.__setattr__(self, "nodes", counter[0])
        object.__setattr__(
            self, "retained_bytes", _template_cost(snapshot, representation))
        _STATE_METRICS["template_validations"] += 1
        _STATE_METRICS["template_nodes"] += counter[0]

    @classmethod
    def from_validated(
        cls,
        template_id: int,
        generation: Hashable,
        fingerprint: int,
        template: tuple,
        template_repr: str,
        nodes: int,
        retained_bytes: int | None = None,
    ) -> "TemplateRef":
        """Build a ref from an internal, already canonical snapshot only."""
        _require_int("template_id", template_id, minimum=1)
        if not isinstance(generation, (int, str)) or isinstance(generation, bool):
            raise PeriodicContractError(
                f"periodic_template_generation_invalid:{generation!r}")
        if not _is_int(fingerprint):
            raise PeriodicContractError(
                f"periodic_template_fingerprint_invalid:{fingerprint!r}")
        if not isinstance(template, tuple):
            raise PeriodicContractError("periodic_template_must_be_tuple")
        if not isinstance(template_repr, str):
            raise PeriodicContractError("periodic_template_repr_invalid")
        _require_int("template_nodes", nodes, minimum=0)
        _check_canonical_snapshot(template, path="template")
        template_repr = template_repr[:_TEMPLATE_REPR_LIMIT]
        # Cost is always derived from the actual retained tree.  A caller may
        # omit it, but an explicitly supplied value must match exactly: the
        # history cache only accepts internally computed costs.
        computed_bytes = _template_cost(template, template_repr)
        if retained_bytes is not None:
            _require_int("template_retained_bytes", retained_bytes, minimum=0)
            if retained_bytes != computed_bytes:
                raise PeriodicContractError(
                    "periodic_template_retained_bytes_mismatch")
        retained_bytes = computed_bytes
        obj = object.__new__(cls)
        object.__setattr__(obj, "template_id", template_id)
        object.__setattr__(obj, "generation", generation)
        object.__setattr__(obj, "fingerprint", fingerprint)
        object.__setattr__(obj, "template", template)
        object.__setattr__(obj, "template_repr", template_repr)
        object.__setattr__(obj, "nodes", nodes)
        object.__setattr__(obj, "retained_bytes", retained_bytes)
        _STATE_METRICS["template_validations"] += 1
        _STATE_METRICS["template_nodes"] += nodes
        return obj


class StateInterner:
    """Bounded exact-structure interner for immutable invariant fragments.

    ``intern`` is the only path that validates a new template.  A hit is an
    exact full-canonical-tree match in the fingerprint bucket; a display repr
    or a fingerprint alone is never accepted.  Entries are LRU-evicted at
    ``limit`` and by their measured retained bytes, and all entries from an
    older generation are dropped when the generation changes.  An item larger
    than the remaining byte budget is returned as a live ref but not cached.
    The cache therefore cannot grow with observation history or bypass the
    caller's byte envelope.
    """

    def __init__(
        self,
        *,
        limit: int = _DEFAULT_INTERNER_LIMIT,
        retained_bytes_limit: int = _DEFAULT_INTERNER_BYTES,
    ) -> None:
        self.limit = _require_int("interner_limit", limit, minimum=1)
        self.retained_bytes_limit = _require_int(
            "interner_retained_bytes_limit", retained_bytes_limit, minimum=1)
        self._buckets: dict[int, list[TemplateRef]] = {}
        self._engine_buckets: dict[int, list[tuple[tuple, TemplateRef]]] = {}
        self._refs: dict[int, TemplateRef] = {}
        self._order: OrderedDict[int, None] = OrderedDict()
        self._next_id = 1
        self._generation: Hashable | None = None
        self._retained_bytes = 0
        self._live: weakref.WeakSet = weakref.WeakSet()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.uncached = 0
        self.oversized = 0
        self.rejected = 0

    @property
    def templates(self) -> int:
        return len(self._refs)

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    def _touch(self, ref: TemplateRef) -> None:
        self._order[ref.template_id] = None
        self._order.move_to_end(ref.template_id)

    def _drop(self, ref: TemplateRef) -> None:
        if self._refs.pop(ref.template_id, None) is None:
            return
        self._order.pop(ref.template_id, None)
        self._retained_bytes = max(
            0, self._retained_bytes - ref.retained_bytes)
        bucket = self._buckets.get(ref.fingerprint)
        if bucket is not None:
            self._buckets[ref.fingerprint] = [item for item in bucket if item.template_id != ref.template_id]
            if not self._buckets[ref.fingerprint]:
                del self._buckets[ref.fingerprint]
        engine_bucket = self._engine_buckets.get(ref.fingerprint)
        if engine_bucket is not None:
            self._engine_buckets[ref.fingerprint] = [
                entry for entry in engine_bucket
                if entry[1].template_id != ref.template_id]
            if not self._engine_buckets[ref.fingerprint]:
                del self._engine_buckets[ref.fingerprint]
        self.evictions += 1

    def _evict_oldest(self) -> None:
        while self._order and (
            len(self._refs) > self.limit
            or self._retained_bytes > self.retained_bytes_limit
        ):
            template_id, _ = self._order.popitem(last=False)
            ref = self._refs.get(template_id)
            if ref is not None:
                self._drop(ref)

    def evict_generation(self, generation: Hashable) -> None:
        for ref in tuple(self._refs.values()):
            if ref.generation != generation:
                self._drop(ref)

    def clear(self) -> None:
        self._buckets.clear()
        self._engine_buckets.clear()
        self._refs.clear()
        self._order.clear()
        self._next_id = 1
        self._generation = None
        self._retained_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.uncached = self.oversized = self.rejected = 0

    def intern(
        self,
        template: tuple,
        *,
        generation: Hashable = 0,
        fingerprint: int | None = None,
    ) -> TemplateRef:
        if not isinstance(template, tuple):
            raise PeriodicContractError("periodic_intern_template_must_be_tuple")
        if not isinstance(generation, (int, str)) or isinstance(generation, bool):
            raise PeriodicContractError(f"periodic_intern_generation_invalid:{generation!r}")
        if self._generation != generation:
            self.evict_generation(generation)
            self._generation = generation
        # Type-exact canonicalization happens before any hit is possible.  The
        # display repr and a caller-supplied fingerprint are bucket selectors
        # only; a hit always checks the full canonical tree.
        counter = [0]
        snapshot = _freeze_snapshot(template, path="template", counter=counter)
        if fingerprint is None:
            fingerprint = hash(snapshot)
        if not _is_int(fingerprint):
            raise PeriodicContractError(
                f"periodic_intern_fingerprint_invalid:{fingerprint!r}")
        probes = 0
        for ref in self._buckets.get(fingerprint, ()):
            probes += 1
            if ref.generation == generation and ref.template == snapshot:
                _STATE_METRICS["template_hits"] += 1
                _STATE_METRICS["template_hit_nodes"] += counter[0]
                _STATE_METRICS["template_probes"] += probes
                self.hits += 1
                self._touch(ref)
                return ref
        _STATE_METRICS["template_probes"] += probes
        self.misses += 1
        _STATE_METRICS["template_builds"] += 1
        _STATE_METRICS["template_build_nodes"] += counter[0]
        ref = TemplateRef.from_validated(
            self._next_id, generation, fingerprint, snapshot,
            "", counter[0])
        self._next_id += 1
        try:
            self._live.add(ref)
        except TypeError:  # pragma: no cover - refs are weakref-able
            pass
        if ref.retained_bytes > self.retained_bytes_limit:
            # A single item cannot be retained without exceeding the caller's
            # remaining byte budget.  It is still a valid live ref; it is just
            # not admitted to the LRU/index.
            self.uncached += 1
            self.oversized += 1
            self.rejected += 1
            return ref
        self._refs[ref.template_id] = ref
        self._buckets.setdefault(ref.fingerprint, []).append(ref)
        self._touch(ref)
        self._retained_bytes += ref.retained_bytes
        self._evict_oldest()
        return ref

    def intern_engine_static(
        self,
        static: tuple,
        *,
        generation: Hashable = 0,
    ) -> TemplateRef:
        """Intern the known Engine static schema without rebuilding the tree.

        The schema is validated directly into an exact reusable key.  A hit
        returns the already fixed canonical tree and only the key/probe cost is
        charged; only a miss freezes the tree.
        """
        if not isinstance(static, tuple):
            raise PeriodicContractError("periodic_intern_template_must_be_tuple")
        if not isinstance(generation, (int, str)) or isinstance(generation, bool):
            raise PeriodicContractError(
                f"periodic_intern_generation_invalid:{generation!r}")
        if self._generation != generation:
            self.evict_generation(generation)
            self._generation = generation
        counter = [0]
        key = _engine_static_key(static, counter)
        fingerprint = hash(key)
        probes = 0
        for entry_key, ref in self._engine_buckets.get(fingerprint, ()):
            probes += 1
            if ref.generation == generation and entry_key == key:
                _STATE_METRICS["template_hits"] += 1
                _STATE_METRICS["template_hit_nodes"] += counter[0]
                _STATE_METRICS["template_probes"] += probes
                self.hits += 1
                self._touch(ref)
                return ref
        _STATE_METRICS["template_probes"] += probes
        self.misses += 1
        build_counter = [0]
        snapshot = _freeze_snapshot(
            static, path="template", counter=build_counter)
        _STATE_METRICS["template_builds"] += 1
        _STATE_METRICS["template_build_nodes"] += build_counter[0]
        ref = TemplateRef.from_validated(
            self._next_id, generation, fingerprint, snapshot,
            "", build_counter[0])
        self._next_id += 1
        try:
            self._live.add(ref)
        except TypeError:  # pragma: no cover - refs are weakref-able
            pass
        if ref.retained_bytes > self.retained_bytes_limit:
            self.uncached += 1
            self.oversized += 1
            self.rejected += 1
            return ref
        self._refs[ref.template_id] = ref
        self._engine_buckets.setdefault(
            ref.fingerprint, []).append((key, ref))
        self._touch(ref)
        self._retained_bytes += ref.retained_bytes
        self._evict_oldest()
        return ref

    def stats(self) -> dict:
        live = [ref for ref in self._live if ref is not None]
        return {
            "templates": self.templates,
            "limit": self.limit,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "retained_bytes": self._retained_bytes,
            "retained_bytes_limit": self.retained_bytes_limit,
            "uncached": self.uncached,
            "oversized": self.oversized,
            "rejected": self.rejected,
            "live_refs": len(live),
            "live_bytes": sum(ref.retained_bytes for ref in live),
        }


@dataclass(frozen=True)
class CanonicalState:
    """Validated invariant template plus only the changed dynamic residue.

    ``template`` is already validated and interned; ``dynamic`` is validated
    here on every construction (the changed restricted fields).  A dynamic
    fingerprint is a fast-reject only; exact ``dynamic`` equality remains the
    match condition.

    ``retained_bytes`` is the real ``sys.getsizeof`` total of the frozen
    dynamic tree.  It is measured while that tree is built, so a history frame
    can charge its own evidence in O(1) instead of re-walking the whole frozen
    graph.  It stays 0 for the generic validated path, whose cost is measured
    by ``_measure_retained`` at admission time.
    """

    template: TemplateRef
    dynamic: tuple
    version: int = 0
    origin: str = "trusted"
    dynamic_fingerprint: int | None = None
    nodes: int = 0
    retained_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.template, TemplateRef):
            raise PeriodicContractError("periodic_canonical_template_invalid")
        if not isinstance(self.dynamic, tuple):
            raise PeriodicContractError("periodic_canonical_dynamic_must_be_tuple")
        counter = [0]
        sizes = [0] if self.origin == "engine" else None
        if self.origin == "engine":
            snapshot = _freeze_engine_dynamic(self.dynamic, counter, sizes)
        else:
            _check_structural(self.dynamic, path="dynamic", counter=counter)
            # Own the observed value: a later caller mutation of a shared
            # list/dict must not rewrite this historical frontier.
            snapshot = _freeze_snapshot(
                self.dynamic, path="dynamic", counter=counter)
        object.__setattr__(self, "dynamic", snapshot)
        object.__setattr__(self, "dynamic_fingerprint", hash(snapshot))
        object.__setattr__(self, "nodes", counter[0])
        if sizes is not None:
            object.__setattr__(self, "retained_bytes", int(sizes[0]))
        _require_int("canonical_version", self.version)
        if self.origin not in {"engine", "trusted", "external"}:
            raise PeriodicContractError(f"periodic_canonical_origin_invalid:{self.origin!r}")
        _STATE_METRICS["dynamic_validations"] += 1
        _STATE_METRICS["dynamic_nodes"] += counter[0]


def split_engine_state(state: tuple) -> tuple[tuple, tuple] | None:
    """Split the current Engine structural 5-tuple into static/dynamic parts.

    The known form is the T04 adapter's tile-index-independent state:
    ``(tasks, runs, allocations, pools, releases)`` with each task
    ``(key, remaining, waiting, release, start, priority, caps, placement,
    coupled, owner)``.  The static fields are interned; only remaining/waiting/
    release/start plus the small runs/allocations/pools/releases part is
    validated on every frame.  Any other shape, or static data that is not
    hashable/immutable, returns ``None`` and keeps the full validation path.
    """
    if type(state) is not tuple or len(state) != 5:
        return None
    tasks, runs, allocations, pools, releases = state
    if not all(type(part) is tuple for part in (tasks, runs, allocations, pools, releases)):
        return None
    static_parts = []
    dynamic_parts = []
    for task in tasks:
        if type(task) is not tuple or len(task) != 10:
            return None
        (key, remaining, waiting, release, start, priority, caps, placement, coupled, owner) = task
        static_parts.append((key, priority, caps, placement, coupled, owner))
        dynamic_parts.append((remaining, waiting, release, start))
    static = tuple(static_parts)
    try:
        hash(static)
    except TypeError:
        return None
    dynamic = (tuple(dynamic_parts), runs, allocations, pools, releases)
    return static, dynamic


_DEFAULT_INTERNER = StateInterner(
    limit=_DEFAULT_INTERNER_LIMIT,
    retained_bytes_limit=_DEFAULT_INTERNER_BYTES,
)


def reset_state_caches() -> None:
    """Reset the process-local template interner and structural counters."""
    _DEFAULT_INTERNER.clear()
    reset_metrics()


def _canonical_equal(first: CanonicalState, second: CanonicalState) -> bool:
    """Exact canonical comparison with a fast fingerprint rejection."""
    if first.origin == "external" or second.origin == "external":
        # External canonical values keep the full exact comparison path; only
        # the trusted engine/trusted origins may use the template identity.
        return _structural_equal((first.template.template, first.dynamic), (second.template.template, second.dynamic))
    if (
        first.template.fingerprint != second.template.fingerprint
        or first.template.generation != second.template.generation
    ):
        return False
    if (
        first.dynamic_fingerprint is not None
        and second.dynamic_fingerprint is not None
        and first.dynamic_fingerprint != second.dynamic_fingerprint
    ):
        return False
    if first.template is not second.template and first.template.template != second.template.template:
        return False
    return first.dynamic == second.dynamic


def _state_equal(first: SettledFrontier, second: SettledFrontier) -> bool:
    left = first._canonical
    right = second._canonical
    if left is not None and right is not None:
        return _canonical_equal(left, right)
    if left is None and right is None:
        # History evidence is the frozen view captured at observation, never
        # the public mutable input (which may have been rewritten since).
        left_raw = first._raw_snapshot
        right_raw = second._raw_snapshot
        if left_raw is None or right_raw is None:
            return False
        return _structural_equal(left_raw, right_raw)
    return False


def _state_difference(first: SettledFrontier, second: SettledFrontier) -> str | None:
    left = first._canonical
    right = second._canonical
    if left is not None and right is not None:
        if left.origin == "external" or right.origin == "external":
            return _first_difference(
                (left.template.template, left.dynamic),
                (right.template.template, right.dynamic),
                path="canonical.external",
            )
        if left.template.fingerprint != right.template.fingerprint:
            return "canonical.template_fingerprint"
        if left.template is not right.template and left.template.template != right.template.template:
            return _first_difference(left.template.template, right.template.template, path="canonical.template")
        if left.dynamic != right.dynamic:
            return _first_difference(left.dynamic, right.dynamic, path="canonical.dynamic")
        return None
    if left is None and right is None:
        left_raw = first._raw_snapshot
        right_raw = second._raw_snapshot
        if left_raw is None or right_raw is None:
            return "raw.missing_snapshot"
        return _first_difference(left_raw, right_raw, path="raw")
    return "canonical.mixed_state_form"


def _freeze_map(name: str, value: Mapping, *, value_check) -> Mapping:
    if not isinstance(value, Mapping):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    frozen = {}
    for key, item in value.items():
        frozen[key] = value_check(key, item)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class Cursor:
    """One active run's progress and its current structural phase class.

    ``slots`` must describe the live window in a tile-index-independent form
    (for example the sorted live slot offsets).  ``next_issue``/``retired`` are
    absolute counters and are deliberately excluded from the structural key.
    """

    run_id: Hashable
    phase_ordinal: int
    phase_end_exclusive: int
    next_issue: int
    retired: int
    slots: tuple | int

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_int("phase_ordinal", self.phase_ordinal)
        _require_int("phase_end_exclusive", self.phase_end_exclusive)
        _require_int("next_issue", self.next_issue)
        _require_int("retired", self.retired)
        if isinstance(self.slots, bool) or not isinstance(self.slots, (int, tuple)):
            raise PeriodicContractError("periodic_cursor_slots_must_be_int_or_tuple")
        if isinstance(self.slots, int):
            _require_int("cursor_slots", self.slots, minimum=1)
        else:
            _check_structural(self.slots, path=f"cursor[{self.run_id}].slots")
            object.__setattr__(
                self, "slots",
                _freeze_snapshot(
                    self.slots, path=f"cursor[{self.run_id}].slots"))
        if self.next_issue < self.retired:
            raise PeriodicContractError(f"periodic_cursor_retired_ahead:{self.run_id}")
        # ``phase_end_exclusive`` may legitimately be behind ``next_issue``
        # for one observation at a phase transition.  The bound calculation
        # then yields zero cycles and the next movement stays detailed.


@dataclass(frozen=True)
class SettledFrontier:
    """A fully settled Engine landmark plus the complete structural state.

    ``normalized_state`` is the adapter's exact, tile-index-independent view of
    the *whole* active Engine cohort: all live remaining maps, latency, edges,
    FIFO order, ready/release/completable order, allocation identities with
    pool capacities/units/scope, and any other field that can affect the next
    interval.  It is a tuple of ordinary nested containers only; callables,
    Engine objects and precomputed digests are rejected.
    """

    now: float
    epoch: Hashable
    normalized_state: tuple | CanonicalState
    cursors: tuple[Cursor, ...]
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", _require_float("now", self.now, allow_infinite=False))
        if not isinstance(self.epoch, (int, str)) or isinstance(self.epoch, bool):
            raise PeriodicContractError(f"periodic_epoch_invalid:{self.epoch!r}")
        state = self.normalized_state
        if isinstance(state, CanonicalState):
            # Trusted/interned canonical input: the template was validated by
            # StateInterner/TemplateRef; only the dynamic residue is checked.
            if state.template.generation != self.epoch:
                raise PeriodicContractError("periodic_canonical_generation_mismatch")
            object.__setattr__(self, "_canonical", state)
            object.__setattr__(self, "_raw_snapshot", None)
            object.__setattr__(self, "_state_form", "canonical")
            _STATE_METRICS["canonical_frames"] += 1
        elif isinstance(state, tuple):
            split = split_engine_state(state)
            canonical = None
            if split is not None:
                static, dynamic = split
                try:
                    template = _DEFAULT_INTERNER.intern_engine_static(
                        static, generation=self.epoch)
                    canonical = CanonicalState(
                        template, dynamic,
                        version=_next_state_version(), origin="engine")
                except PeriodicContractError:
                    # The restricted fast path accepted only known exact
                    # builtin scalar/container types.  Anything else keeps the
                    # generic validated path (which rejects rather than
                    # storing an opaque descendant).
                    canonical = None
            if canonical is not None:
                object.__setattr__(self, "_canonical", canonical)
                object.__setattr__(self, "_raw_snapshot", None)
                object.__setattr__(self, "_state_form", "engine")
                _STATE_METRICS["canonical_frames"] += 1
            else:
                counter = [0]
                _check_structural(state, counter=counter)
                object.__setattr__(self, "_canonical", None)
                object.__setattr__(
                    self, "_raw_snapshot",
                    _freeze_snapshot(state, path="normalized_state"))
                object.__setattr__(self, "_state_form", "raw")
                _STATE_METRICS["raw_frames"] += 1
                _STATE_METRICS["raw_nodes"] += counter[0]
        else:
            raise PeriodicContractError("periodic_normalized_state_must_be_tuple_or_canonical")
        if not isinstance(self.cursors, tuple):
            raise PeriodicContractError("periodic_cursors_must_be_tuple")
        seen: set[Hashable] = set()
        for cursor in self.cursors:
            if not isinstance(cursor, Cursor):
                raise PeriodicContractError("periodic_cursor_type_invalid")
            if cursor.run_id in seen:
                raise PeriodicContractError(f"periodic_duplicate_run_id:{cursor.run_id}")
            seen.add(cursor.run_id)
        _require_int("sequence", self.sequence)

    @property
    def state_form(self) -> str:
        return self._state_form

    @property
    def canonical_state(self) -> CanonicalState | None:
        return self._canonical


@dataclass(frozen=True)
class IntervalDelta:
    """Actual measured progress between two settled frontiers.

    ``issued_retired_by_run`` maps every active ``run_id`` to
    ``(issued_tiles, retired_tiles)`` in this interval.  ``service_by_resource``
    and ``busy_by_resource`` are real resource-unit / millisecond amounts.
    ``per_unit_peaks`` maps a caller-defined pool/unit identity to the peak
    occupancy observed in the interval.  Nothing here is fabricated by the
    detector: the adapter must measure it from the Engine.
    """

    dt: float
    issued_retired_by_run: Mapping
    sequence_count: int
    service_by_resource: Mapping
    busy_by_resource: Mapping
    allocation_event_count: int
    per_unit_peaks: Mapping
    logical_completions: int
    crossed_external_or_opaque: bool
    allocation_grants: int = 0
    allocation_last_uses: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _require_float("interval_dt", self.dt))
        if self.dt < 0:
            raise PeriodicContractError(f"periodic_interval_dt_invalid:{self.dt!r}")

        def issued_check(key, value):
            _require_run_id(key, name="interval_run_id")
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise PeriodicContractError(f"periodic_interval_issued_invalid:{key!r}")
            return (
                _require_int("issued", value[0]),
                _require_int("retired", value[1]),
            )

        object.__setattr__(
            self,
            "issued_retired_by_run",
            _freeze_map(
                "issued_retired_by_run",
                self.issued_retired_by_run,
                value_check=issued_check,
            ),
        )

        def amount_check(key, value):
            if not isinstance(key, str) or not key:
                raise PeriodicContractError(f"periodic_interval_resource_invalid:{key!r}")
            return _require_float("interval_amount", value)

        object.__setattr__(
            self,
            "service_by_resource",
            _freeze_map(
                "service_by_resource",
                self.service_by_resource,
                value_check=amount_check,
            ),
        )
        object.__setattr__(
            self,
            "busy_by_resource",
            _freeze_map("busy_by_resource", self.busy_by_resource, value_check=amount_check),
        )

        def peak_check(key, value):
            _check_structural(key, path="interval_peak_key")
            return _require_float("interval_peak", value)

        object.__setattr__(
            self,
            "per_unit_peaks",
            _freeze_map("per_unit_peaks", self.per_unit_peaks, value_check=peak_check),
        )
        _require_int("sequence_count", self.sequence_count)
        _require_int("allocation_event_count", self.allocation_event_count)
        _require_int("logical_completions", self.logical_completions)
        _require_int("allocation_grants", self.allocation_grants)
        _require_int("allocation_last_uses", self.allocation_last_uses)
        if not isinstance(self.crossed_external_or_opaque, bool):
            raise PeriodicContractError("periodic_interval_crossed_flag_must_be_bool")


@dataclass(frozen=True)
class RunCorrespondence:
    """Exact counter translation for one run in the skipped cohort."""

    run_id: Hashable
    phase_ordinal: int
    phase_end_exclusive: int
    next_issue_before: int
    next_issue_after: int
    retired_before: int
    retired_after: int
    tile_offset: int
    finished_offset: int

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        for name in (
            "phase_ordinal",
            "phase_end_exclusive",
            "next_issue_before",
            "next_issue_after",
            "retired_before",
            "retired_after",
            "tile_offset",
            "finished_offset",
        ):
            _require_int(name, getattr(self, name))
        if self.next_issue_after != self.next_issue_before + self.tile_offset:
            raise PeriodicContractError(f"periodic_correspondence_issue_mismatch:{self.run_id}")
        if self.retired_after != self.retired_before + self.tile_offset:
            raise PeriodicContractError(f"periodic_correspondence_retire_mismatch:{self.run_id}")


@dataclass(frozen=True)
class WindowTile:
    """One live-window tile whose absolute identity must translate on skip."""

    run_id: Hashable
    old_tile: int
    new_tile: int

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_int("old_tile", self.old_tile)
        _require_int("new_tile", self.new_tile)


@dataclass(frozen=True)
class CycleCorrespondence:
    """How live identity maps from the observed frontier to the skipped one.

    Event sequences of already-submitted live tasks are *not* rewritten: they
    keep their object identity and heap positions.  ``event_sequence_offset``
    is the amount by which the Engine's future ``_sequence`` must advance and
    the offset at which the skipped cycle's never-created events would have
    appeared.  ``tile_offset_by_run`` applies to tile-scoped allocation names
    and to ``_periodic_tasks`` metadata; persistent allocation names do not
    move.
    """

    runs: tuple[RunCorrespondence, ...]
    window: tuple[WindowTile, ...]
    event_sequence_offset: int
    tile_offset_by_run: Mapping

    def __post_init__(self) -> None:
        if not isinstance(self.runs, tuple):
            raise PeriodicContractError("periodic_correspondence_runs_invalid")
        if not isinstance(self.window, tuple):
            raise PeriodicContractError("periodic_correspondence_window_invalid")
        seen: set[Hashable] = set()
        for run in self.runs:
            if not isinstance(run, RunCorrespondence):
                raise PeriodicContractError("periodic_correspondence_run_invalid")
            if run.run_id in seen:
                raise PeriodicContractError(f"periodic_correspondence_duplicate_run:{run.run_id}")
            seen.add(run.run_id)
        for tile in self.window:
            if not isinstance(tile, WindowTile) or tile.run_id not in seen:
                raise PeriodicContractError("periodic_correspondence_window_invalid")
            run = next(item for item in self.runs if item.run_id == tile.run_id)
            if tile.new_tile - tile.old_tile != run.tile_offset:
                raise PeriodicContractError(f"periodic_correspondence_window_offset:{tile.run_id}")
        _require_int("event_sequence_offset", self.event_sequence_offset)

        def offset_check(key, value):
            _require_run_id(key, name="correspondence_run_id")
            return _require_int("tile_offset", value)

        object.__setattr__(
            self,
            "tile_offset_by_run",
            _freeze_map("tile_offset_by_run", self.tile_offset_by_run, value_check=offset_check),
        )
        for run in self.runs:
            if self.tile_offset_by_run[run.run_id] != run.tile_offset:
                raise PeriodicContractError(f"periodic_correspondence_offset_mismatch:{run.run_id}")

    def run(self, run_id: Hashable) -> RunCorrespondence:
        for item in self.runs:
            if item.run_id == run_id:
                return item
        raise KeyError(run_id)


@dataclass(frozen=True)
class CycleSkip:
    """A bounded, exact skip description; the adapter applies it explicitly.

    ``delta`` is the measured delta for **one** period.  Use
    :meth:`total_delta` (or multiply by ``cycles``) for accounting the whole
    skip.  ``tile_strides_by_run`` and ``seq_stride`` are also per period.
    """

    cycles: int
    period: float
    tile_strides_by_run: Mapping
    seq_stride: int
    delta: IntervalDelta
    correspondence: CycleCorrespondence

    def __post_init__(self) -> None:
        _require_int("cycles", self.cycles, minimum=1)
        object.__setattr__(self, "period", _require_float("period", self.period, positive=True))

        def stride_check(key, value):
            _require_run_id(key, name="skip_run_id")
            return _require_int("tile_stride", value)

        object.__setattr__(
            self,
            "tile_strides_by_run",
            _freeze_map(
                "tile_strides_by_run",
                self.tile_strides_by_run,
                value_check=stride_check,
            ),
        )
        _require_int("seq_stride", self.seq_stride, minimum=1)
        if not isinstance(self.delta, IntervalDelta):
            raise PeriodicContractError("periodic_skip_delta_invalid")
        if not isinstance(self.correspondence, CycleCorrespondence):
            raise PeriodicContractError("periodic_skip_correspondence_invalid")
        if not _numeric_close(self.delta.dt, self.period):
            raise PeriodicContractError("periodic_skip_delta_period_mismatch")
        if set(self.tile_strides_by_run) != {run.run_id for run in self.correspondence.runs}:
            raise PeriodicContractError("periodic_skip_correspondence_cohort")
        if not any(self.tile_strides_by_run.values()):
            raise PeriodicContractError("periodic_skip_without_tile_progress")
        if self.correspondence.event_sequence_offset != (self.cycles * self.seq_stride):
            raise PeriodicContractError("periodic_skip_sequence_offset_mismatch")
        for run in self.correspondence.runs:
            stride = self.tile_strides_by_run[run.run_id]
            if run.tile_offset != self.cycles * stride:
                raise PeriodicContractError(f"periodic_skip_tile_offset_mismatch:{run.run_id}")

    def total_delta(self) -> IntervalDelta:
        """Measured per-period delta multiplied by the exact skip count."""
        scale = self.cycles
        issued = {
            run_id: (issued * scale, retired * scale)
            for run_id, (issued, retired) in self.delta.issued_retired_by_run.items()
        }
        return IntervalDelta(
            dt=self.period * scale,
            issued_retired_by_run=issued,
            sequence_count=self.seq_stride * scale,
            service_by_resource={name: amount * scale for name, amount in self.delta.service_by_resource.items()},
            busy_by_resource={name: amount * scale for name, amount in self.delta.busy_by_resource.items()},
            allocation_event_count=self.delta.allocation_event_count * scale,
            per_unit_peaks=dict(self.delta.per_unit_peaks),
            logical_completions=self.delta.logical_completions * scale,
            crossed_external_or_opaque=False,
            allocation_grants=self.delta.allocation_grants * scale,
            allocation_last_uses=self.delta.allocation_last_uses * scale,
        )


@dataclass(frozen=True)
class _HistoryFrame:
    """Detached history evidence: never retains the public mutable input."""

    now: float
    epoch: Hashable
    cursors: tuple
    sequence: int
    _canonical: CanonicalState | None
    _raw_snapshot: tuple | None
    _state_form: str
    retained_bytes: int


def _history_frame(
    frame: SettledFrontier, *, seen: set[int] | None = None,
) -> _HistoryFrame:
    if seen is None:
        seen = set()
    canonical = frame._canonical
    if canonical is not None:
        # The template is shared with the interner and accounted once there;
        # history pays for the per-frontier dynamic snapshot, identifiers and
        # cursors.  An engine-origin snapshot already charged its own nodes
        # while it was frozen, so no second recursive walk is needed.
        if canonical.origin == "engine":
            owned = canonical.retained_bytes
        else:
            owned = _measure_retained(canonical.dynamic, seen=seen)
    else:
        owned = _measure_retained(frame._raw_snapshot, seen=seen)
    owned += _measure_retained(frame.epoch, seen=seen)
    for cursor in frame.cursors:
        owned += _CURSOR_OVERHEAD
        owned += _measure_retained(cursor.run_id, seen=seen)
        owned += _measure_retained(cursor.slots, seen=seen)
    owned += _FRAME_OVERHEAD
    return _HistoryFrame(
        now=frame.now,
        epoch=frame.epoch,
        cursors=frame.cursors,
        sequence=frame.sequence,
        _canonical=canonical,
        _raw_snapshot=frame._raw_snapshot,
        _state_form=frame._state_form,
        retained_bytes=owned,
    )


def _interval_retained_bytes(
    interval: IntervalDelta | None, *, seen: set[int] | None = None,
) -> int:
    """Charge the whole frozen interval, including every mapping key/value."""
    if interval is None:
        return 0
    if seen is None:
        seen = set()
    owned = _INTERVAL_OVERHEAD
    for name in (
        "issued_retired_by_run",
        "service_by_resource",
        "busy_by_resource",
        "per_unit_peaks",
    ):
        owned += _measure_retained(getattr(interval, name), seen=seen)
    return owned


@dataclass(frozen=True)
class _Observation:
    frame: _HistoryFrame
    interval: IntervalDelta | None
    retained_bytes: int


@dataclass(frozen=True)
class _CycleEvidence:
    period: float
    tile_strides: Mapping
    retired_strides: Mapping
    seq_stride: int
    delta: IntervalDelta


class CycleDetector:
    """Pure detector for the frozen ``CycleDetector.observe`` interface.

    One tile retirement normally spans several settled micro-steps (load,
    compute, store, retire, allocation grant/release), so the full structural
    state repeats at a *lag*, not necessarily on adjacent observations.  The
    detector keeps a bounded rolling history, compares the current frontier
    with older frontiers, and aggregates every real interval across the whole
    candidate period.  It then requires the same lag/period/strides/delta to
    occur for ``evidence_periods`` consecutive cycles.  Phase remaining is
    used only to bound the skip; it is never a structural key.
    """

    def __init__(
        self,
        *,
        evidence_periods: int = 2,
        max_lag: int = 24,
        history_limit: int | None = None,
        history_bytes_limit: int = _DEFAULT_HISTORY_BYTES,
    ) -> None:
        self.evidence_periods = _require_int("evidence_periods", evidence_periods, minimum=1)
        self.max_lag = _require_int("max_lag", max_lag, minimum=1)
        minimum_history = (self.evidence_periods + 1) * self.max_lag
        if history_limit is None:
            history_limit = minimum_history
        self.history_limit = _require_int("history_limit", history_limit, minimum=minimum_history)
        self.history_bytes_limit = _require_int(
            "history_bytes_limit", history_bytes_limit, minimum=1)
        self._observations: list[_Observation] = []
        self._history_bytes = 0
        self._history_templates: dict[int, list] = {}
        self._last_rejection = "initial"
        self._last_difference: str | None = None
        self._last_skip: CycleSkip | None = None
        self.observations_seen = 0
        self.skips_returned = 0
        self._stats = {
            "observations": 0,
            "skips": 0,
            "skipped_cycles": 0,
            "candidate_lags": 0,
            "fast_state_compares": 0,
            "exact_state_compares": 0,
            "self_s": 0.0,
            "rejections": {},
        }

    @property
    def last_rejection(self) -> str:
        """Why the most recent observation did not return a skip."""
        return self._last_rejection

    @property
    def last_difference(self) -> str | None:
        """First unequal structural component of the latest adjacent frontier."""
        return self._last_difference

    @property
    def last_skip(self) -> CycleSkip | None:
        return self._last_skip

    @property
    def observations(self) -> int:
        return len(self._observations)

    def _set_rejection(self, reason: str) -> None:
        self._last_rejection = reason
        counts = self._stats["rejections"]
        counts[reason] = counts.get(reason, 0) + 1

    def stats(self) -> dict:
        """Return bounded detector counters and structural-cache evidence."""
        result = dict(self._stats)
        result["rejections"] = dict(self._stats["rejections"])
        result["observations_in_history"] = len(self._observations)
        result["history_bytes"] = self._history_bytes
        result["history_bytes_limit"] = self.history_bytes_limit
        result["history_templates"] = len(self._history_templates)
        result["last_rejection"] = self._last_rejection
        result["interner"] = _DEFAULT_INTERNER.stats()
        result["state_metrics"] = metrics()
        result["helper_self_s"] = result["self_s"]
        result["frames"] = result["observations"]
        result["candidate_count"] = result["candidate_lags"]
        result["rejection_reasons"] = dict(self._stats["rejections"])
        return result

    def reset_stats(self) -> None:
        for key in (
            "observations",
            "skips",
            "skipped_cycles",
            "candidate_lags",
            "fast_state_compares",
            "exact_state_compares",
        ):
            self._stats[key] = 0
        self._stats["self_s"] = 0.0
        self._stats["rejections"] = {}

    def reset(self) -> None:
        """Forget the private history; the next observe becomes a baseline."""
        self._clear_history()
        self._last_rejection = "reset"

    def _clear_history(self) -> None:
        self._observations.clear()
        self._history_bytes = 0
        self._history_templates.clear()

    def _retain_template(self, frame: _HistoryFrame) -> None:
        canonical = frame._canonical
        if canonical is None:
            return
        template = canonical.template
        key = id(template)
        entry = self._history_templates.get(key)
        if entry is None:
            self._history_templates[key] = [template, 1, template.retained_bytes]
            self._history_bytes += template.retained_bytes
        else:
            entry[1] += 1

    def _release_template(self, frame: _HistoryFrame) -> None:
        canonical = frame._canonical
        if canonical is None:
            return
        template = canonical.template
        key = id(template)
        entry = self._history_templates.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            self._history_bytes -= entry[2]
            del self._history_templates[key]

    def _store_observation(
        self, frame: SettledFrontier, interval: IntervalDelta | None,
    ) -> bool:
        seen: set[int] = set()
        history = _history_frame(frame, seen=seen)
        owned = (
            history.retained_bytes
            + _interval_retained_bytes(interval, seen=seen)
            + _OBSERVATION_OVERHEAD
        )
        self._observations.append(_Observation(history, interval, owned))
        self._history_bytes += owned
        self._retain_template(history)
        return self._trim()

    def _trim(self) -> bool:
        """Trim by item count and retained bytes; ``False`` means no baseline."""
        while self._observations and (
            len(self._observations) > self.history_limit
            or self._history_bytes > self.history_bytes_limit
        ):
            if len(self._observations) <= 1:
                # The current exact representation alone does not fit the
                # caller's remaining budget.  Drop the baseline rather than
                # retaining an unaccounted tree or approximating it.
                self._clear_history()
                return False
            removed = self._observations.pop(0)
            self._history_bytes -= removed.retained_bytes
            self._release_template(removed.frame)
        return True

    def observe(
        self,
        frame: SettledFrontier,
        interval: IntervalDelta | None,
        *,
        stop_at: float,
    ) -> CycleSkip | None:
        """Frozen call boundary; records self time and delegates."""
        started = time.perf_counter()
        self._stats["observations"] += 1
        try:
            return self._observe(frame, interval, stop_at=stop_at)
        finally:
            self._stats["self_s"] += time.perf_counter() - started

    def _observe(
        self,
        frame: SettledFrontier,
        interval: IntervalDelta | None,
        *,
        stop_at: float,
    ) -> CycleSkip | None:
        """Observe one settled landmark; return a skip only when proven.

        ``stop_at`` is the adapter's actual safe boundary (horizon, pending
        external release or another true external event).  It is not a target
        the detector may invent; ``math.inf`` is accepted when the only bound
        is phase remaining.  A ``None`` interval is allowed for the initial
        landmark or a caller that cannot measure a zero-time step, but no
        candidate period may contain one.
        """
        if not isinstance(frame, SettledFrontier):
            raise PeriodicContractError("periodic_observe_requires_frontier")
        if interval is not None and not isinstance(interval, IntervalDelta):
            raise PeriodicContractError("periodic_observe_interval_invalid")
        stop = _require_float("stop_at", stop_at, allow_infinite=True)
        self.observations_seen += 1
        self._last_skip = None

        reset_reason = None
        if self._observations and self._observations[-1].frame.epoch != frame.epoch:
            self._clear_history()
            reset_reason = "epoch_changed"
        elif self._observations and not self._same_phase_class(self._observations[-1].frame, frame):
            # A phase/wave boundary is a real boundary.  Drop old candidates
            # immediately instead of holding complete frontiers from a prior
            # phase class in the bounded cache.
            self._clear_history()
            reset_reason = "phase_changed"
        self._last_difference = None
        if self._observations and interval is not None:
            self._last_difference = _state_difference(self._observations[-1].frame, frame)
        if interval is not None and interval.crossed_external_or_opaque:
            self._clear_history()
            self._store_observation(frame, interval)
            self._set_rejection("external_or_opaque")
            return None
        if not self._store_observation(frame, interval):
            self._last_difference = None
            self._set_rejection("history_budget")
            return None

        evidence = self._find_evidence()
        if evidence is None:
            if reset_reason is not None:
                self._set_rejection(reset_reason)
            elif len(self._observations) < self.evidence_periods + 1:
                self._set_rejection("baseline")
            else:
                self._set_rejection("no_repeated_span")
            return None

        cycles = self._bounded_cycles(frame, evidence, stop)
        if cycles < 1:
            self._set_rejection("boundary")
            return None

        skip = self._build_skip(frame, evidence, cycles)
        self.reset()
        self._last_skip = skip
        self._set_rejection("skip")
        self.skips_returned += 1
        self._stats["skips"] += 1
        self._stats["skipped_cycles"] += cycles
        return skip

    def _find_evidence(self) -> _CycleEvidence | None:
        count = len(self._observations)
        max_lag = min(self.max_lag, (count - 1) // self.evidence_periods)
        for lag in range(1, max_lag + 1):
            indices = [count - 1 - step * lag for step in range(self.evidence_periods, -1, -1)]
            if indices[0] < 0:
                continue
            frames = [self._observations[index].frame for index in indices]
            if not all(self._same_class(frames[0], other) for other in frames[1:]):
                continue
            spans = [self._aggregate(left, right) for left, right in zip(indices, indices[1:])]
            if any(span is None for span in spans):
                continue
            first = spans[0]
            if not all(self._evidence_equal(first, span) for span in spans[1:]):
                continue
            self._stats["candidate_lags"] += 1
            return spans[-1]
        return None

    @staticmethod
    def _same_phase_class(
        first: SettledFrontier,
        second: SettledFrontier,
    ) -> bool:
        left = {item.run_id: item for item in first.cursors}
        right = {item.run_id: item for item in second.cursors}
        if left.keys() != right.keys():
            return False
        return all(
            left[run_id].phase_ordinal == right[run_id].phase_ordinal
            and left[run_id].phase_end_exclusive == right[run_id].phase_end_exclusive
            for run_id in left
        )

    def _same_class(self, first: SettledFrontier, second: SettledFrontier) -> bool:
        if first.epoch != second.epoch:
            return False
        left_canonical = first._canonical
        right_canonical = second._canonical
        if (
            left_canonical is not None
            and right_canonical is not None
            and left_canonical.origin != "external"
            and right_canonical.origin != "external"
            and left_canonical.template is right_canonical.template
        ):
            self._stats["fast_state_compares"] += 1
        else:
            self._stats["exact_state_compares"] += 1
        if not _state_equal(first, second):
            return False
        left = {item.run_id: item for item in first.cursors}
        right = {item.run_id: item for item in second.cursors}
        if left.keys() != right.keys():
            return False
        for run_id, cursor in left.items():
            other = right[run_id]
            if cursor.phase_ordinal != other.phase_ordinal or cursor.phase_end_exclusive != other.phase_end_exclusive:
                return False
            if not _structural_equal(cursor.slots, other.slots):
                return False
        return True

    def _aggregate(
        self,
        start_index: int,
        end_index: int,
    ) -> _CycleEvidence | None:
        start = self._observations[start_index].frame
        end = self._observations[end_index].frame
        parts = [self._observations[index].interval for index in range(start_index + 1, end_index + 1)]
        if any(part is None for part in parts):
            return None
        if any(part.crossed_external_or_opaque for part in parts):
            return None
        period = end.now - start.now
        if not math.isfinite(period) or period <= 0:
            return None
        measured_dt = math.fsum(part.dt for part in parts)
        if not _numeric_close(measured_dt, period):
            return None

        start_cursors = {item.run_id: item for item in start.cursors}
        end_cursors = {item.run_id: item for item in end.cursors}
        if start_cursors.keys() != end_cursors.keys():
            return None
        issued: dict[Hashable, int] = {run_id: 0 for run_id in start_cursors}
        retired: dict[Hashable, int] = {run_id: 0 for run_id in start_cursors}
        for part in parts:
            if set(part.issued_retired_by_run) != set(issued):
                return None
            for run_id, (
                part_issued,
                part_retired,
            ) in part.issued_retired_by_run.items():
                issued[run_id] += part_issued
                retired[run_id] += part_retired
        for run_id, cursor in end_cursors.items():
            before = start_cursors[run_id]
            stride = cursor.next_issue - before.next_issue
            retire_stride = cursor.retired - before.retired
            if stride < 0 or retire_stride < 0:
                return None
            if (issued[run_id], retired[run_id]) != (stride, retire_stride):
                return None

        seq_stride = end.sequence - start.sequence
        sequence_count = sum(part.sequence_count for part in parts)
        if seq_stride < 0 or sequence_count != seq_stride:
            return None
        if not any(issued.values()) or seq_stride == 0:
            return None

        service: dict[str, list[float]] = {}
        busy: dict[str, list[float]] = {}
        for part in parts:
            for name, amount in part.service_by_resource.items():
                service.setdefault(name, []).append(amount)
            for name, amount in part.busy_by_resource.items():
                busy.setdefault(name, []).append(amount)
        peaks: dict[Any, float] = {}
        for part in parts:
            for key, value in part.per_unit_peaks.items():
                peaks[key] = max(peaks.get(key, 0.0), value)
        aggregate_delta = IntervalDelta(
            dt=period,
            issued_retired_by_run={run_id: (issued[run_id], retired[run_id]) for run_id in issued},
            sequence_count=seq_stride,
            service_by_resource={name: math.fsum(values) for name, values in service.items()},
            busy_by_resource={name: math.fsum(values) for name, values in busy.items()},
            allocation_event_count=sum(part.allocation_event_count for part in parts),
            per_unit_peaks=peaks,
            logical_completions=sum(part.logical_completions for part in parts),
            crossed_external_or_opaque=False,
            allocation_grants=sum(part.allocation_grants for part in parts),
            allocation_last_uses=sum(part.allocation_last_uses for part in parts),
        )
        return _CycleEvidence(
            period=period,
            tile_strides=dict(issued),
            retired_strides=dict(retired),
            seq_stride=seq_stride,
            delta=aggregate_delta,
        )

    @staticmethod
    def _evidence_equal(
        first: _CycleEvidence,
        second: _CycleEvidence,
    ) -> bool:
        if not _numeric_close(first.period, second.period):
            return False
        if dict(first.tile_strides) != dict(second.tile_strides):
            return False
        if dict(first.retired_strides) != dict(second.retired_strides):
            return False
        if first.seq_stride != second.seq_stride:
            return False
        left, right = first.delta, second.delta
        if not _numeric_close(left.dt, right.dt):
            return False
        if dict(left.issued_retired_by_run) != dict(right.issued_retired_by_run):
            return False
        if left.sequence_count != right.sequence_count:
            return False
        if left.allocation_event_count != right.allocation_event_count:
            return False
        if left.logical_completions != right.logical_completions:
            return False
        if left.allocation_grants != right.allocation_grants:
            return False
        if left.allocation_last_uses != right.allocation_last_uses:
            return False
        if left.crossed_external_or_opaque or right.crossed_external_or_opaque:
            return False
        if set(left.service_by_resource) != set(right.service_by_resource):
            return False
        if set(left.busy_by_resource) != set(right.busy_by_resource):
            return False
        if set(left.per_unit_peaks) != set(right.per_unit_peaks):
            return False
        for name in left.service_by_resource:
            if not _numeric_close(left.service_by_resource[name], right.service_by_resource[name]):
                return False
        for name in left.busy_by_resource:
            if not _numeric_close(left.busy_by_resource[name], right.busy_by_resource[name]):
                return False
        for key in left.per_unit_peaks:
            if not _numeric_close(left.per_unit_peaks[key], right.per_unit_peaks[key]):
                return False
        return True

    @staticmethod
    def _bounded_cycles(
        frame: SettledFrontier,
        evidence: _CycleEvidence,
        stop_at: float,
    ) -> int:
        if stop_at < frame.now:
            return 0
        if math.isinf(stop_at):
            cycles = math.inf
        else:
            # The report requires ``now + n*period < stop_at``: an exact
            # boundary is left for the original advance so it can process the
            # tie/release/completion in real order.  Floor first, then remove
            # an endpoint that floating addition placed at/after the boundary.
            quotient = (stop_at - frame.now) / evidence.period
            cycles = math.floor(quotient) if math.isfinite(quotient) else 0
            if cycles > 0 and frame.now + cycles * evidence.period >= stop_at:
                cycles -= 1
        for cursor in frame.cursors:
            stride = evidence.tile_strides[cursor.run_id]
            if stride <= 0:
                continue
            remaining = cursor.phase_end_exclusive - cursor.next_issue
            if remaining < 0:
                return 0
            cycles = min(cycles, remaining // stride)
        if not math.isfinite(cycles):
            return 0
        return int(cycles)

    @staticmethod
    def _build_skip(
        frame: SettledFrontier,
        evidence: _CycleEvidence,
        cycles: int,
    ) -> CycleSkip:
        runs: list[RunCorrespondence] = []
        window: list[WindowTile] = []
        offsets: dict[Hashable, int] = {}
        for cursor in sorted(frame.cursors, key=lambda item: item.run_id):
            stride = evidence.tile_strides[cursor.run_id]
            offset = cycles * stride
            offsets[cursor.run_id] = offset
            runs.append(
                RunCorrespondence(
                    run_id=cursor.run_id,
                    phase_ordinal=cursor.phase_ordinal,
                    phase_end_exclusive=cursor.phase_end_exclusive,
                    next_issue_before=cursor.next_issue,
                    next_issue_after=cursor.next_issue + offset,
                    retired_before=cursor.retired,
                    retired_after=cursor.retired + offset,
                    tile_offset=offset,
                    finished_offset=cycles * evidence.retired_strides[cursor.run_id],
                )
            )
            for old_tile in range(cursor.retired, cursor.next_issue):
                window.append(
                    WindowTile(
                        run_id=cursor.run_id,
                        old_tile=old_tile,
                        new_tile=old_tile + offset,
                    )
                )
        correspondence = CycleCorrespondence(
            runs=tuple(runs),
            window=tuple(window),
            event_sequence_offset=cycles * evidence.seq_stride,
            tile_offset_by_run=offsets,
        )
        return CycleSkip(
            cycles=cycles,
            period=evidence.period,
            tile_strides_by_run=dict(evidence.tile_strides),
            seq_stride=evidence.seq_stride,
            delta=evidence.delta,
            correspondence=correspondence,
        )
