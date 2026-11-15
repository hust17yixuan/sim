"""Overflow-safe padding-cost arithmetic for large finite inputs.

[Role]
  Pin the public and raw cost helpers to fraction-first scaling, finite
  results, exact large-value rounding and strict JSON output.
[Invariants]
  A finite mathematical product never becomes an intermediate overflow;
  genuinely unrepresentable setup/per-row outputs raise ``OverflowError``.
[Timeouts & Locks]
  Pure in-memory arithmetic; no subprocesses or blocking locks.
[Side-Effect Policy]
  No I/O, publication or caller-container mutation.
"""

import builtins
import json
import math

import pytest

pytest.importorskip("mindie_profiler")

from mindie_sim import diff_trace as diff

CASES = [
    ("full", "Matmul", 4, None, None, 4, 1.0),
    ("grouped", "aclnnGroupedMatmulV4", 64, None,
     {"group_list": [16, 32, 48, 48]}, 48, 0.75),
    ("masked", "aclnnFlashAttentionVarLenScore", 48, 33, None, 33, 0.6875),
]
COSTS = (0.0, 1.0, 1e306, 1e307, 7.5e307, 1e308)
ROUTES = (("public", diff.pad_aware_operator_cost),
          ("raw", diff._pad_aware_operator_cost_snapshot))


@pytest.mark.parametrize("cost", COSTS)
@pytest.mark.parametrize(
    "label,op_name,bucket,logical,bound,charged,fraction", CASES,
    ids=[case[0] for case in CASES])
@pytest.mark.parametrize("route,function", ROUTES, ids=[row[0] for row in ROUTES])
def test_fraction_first_large_finite_cost(
        route, function, label, op_name, bucket, logical, bound, charged,
        fraction, cost):
    result = function(op_name, cost, bucket, logical, bound_args=bound)
    estimate = cost * fraction
    expected = estimate if label == "grouped" else cost
    assert result["valid"] is True
    assert result["charged_tokens"] == charged
    assert result["row_scaled_estimate_ms"] == estimate
    assert result["duration_ms"] == expected
    assert math.isfinite(result["duration_ms"])
    assert math.isfinite(result["row_scaled_estimate_ms"])
    text = json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert json.loads(text)["duration_ms"] == result["duration_ms"]


def test_large_values_never_reach_round(monkeypatch):
    real_round = builtins.round

    def guarded(value, ndigits=None):
        if ndigits == 9 and abs(value) >= 1e290:
            raise AssertionError("round(..., 9) called on a large finite cost")
        return real_round(value, ndigits) if ndigits is not None else real_round(value)

    monkeypatch.setattr(builtins, "round", guarded)
    for cost in (7.5e307, 1e308):
        result = diff.pad_aware_operator_cost(
            "aclnnGroupedMatmulV4", cost, 64, None,
            bound_args={"group_list": [16, 32, 48, 48]})
        assert result["duration_ms"] == cost * 0.75
        assert result["row_scaled_estimate_ms"] == cost * 0.75


def test_large_cost_round_probe_is_not_relied_upon():
    # Builds differ on whether round(1e308, 9) overflows; the helper must not
    # depend on either outcome.
    for value in (7.5e307, 1e308):
        try:
            rounded = round(value, 9)
        except OverflowError:
            continue
        assert rounded == value or not math.isfinite(rounded)


@pytest.mark.parametrize("route,function", ROUTES, ids=[row[0] for row in ROUTES])
def test_unrepresentable_setup_plus_per_row_raises(route, function):
    with pytest.raises(OverflowError):
        function("Matmul", 1.0, 4, None, setup_ms=1e308, per_row_ms=1e308)


def test_large_finite_setup_plus_per_row_serializes():
    result = diff.pad_aware_operator_cost(
        "Matmul", 1e308, 4, None, setup_ms=1e307, per_row_ms=1e307)
    assert result["duration_ms"] == 5e307
    assert result["row_scaled_estimate_ms"] == 1e308
    assert json.loads(json.dumps(result, allow_nan=False))["duration_ms"] == 5e307
