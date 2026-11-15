"""Guarded v4 mathematical records and name-based catalog classification.

[Role]
Evaluate the pinned closed AST with exact integers/Fractions and expose its
coverage without claiming a measured or complete physical source schedule.
[Invariants]
No eval, imports, attributes, missing-symbol defaults, or implicit ledgers.
Native int excludes bool. Conditional physical counters require a caller ledger;
reference algorithms and structural catalog entries never become certifications.
The 53 DSL entries and 67 runtime templates are joined only by explicit names.
[Timeouts & Locks]
Bounded AST depth and node count; read one packaged JSON, no waits or locks.
[Side-Effect Policy]
Read-only packaged data; no source repository or device changes.
"""

import json
import operator
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .exact_math import (MathDomainError, align, ceildiv, floor_sum, integer,
                         lattice_count, layout_capacity, out_axis, valid_axis)


@lru_cache(maxsize=1)
def model():
    return json.loads(Path(__file__).with_name("v4_math_model.json").read_text())


def _sumprod(a, b):
    if len(a) != len(b):
        raise MathDomainError("sumprod length mismatch")
    return sum(x * y for x, y in zip(a, b))


def _divide(a, b):
    value = Fraction(a, b)
    return value.numerator if value.denominator == 1 else value


_BINARY = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
           "div": _divide, "floordiv": operator.floordiv,
           "mod": operator.mod, "pow": operator.pow, "eq": operator.eq,
           "ne": operator.ne, "lt": operator.lt, "le": operator.le,
           "gt": operator.gt, "ge": operator.ge}
_CALLS = {"ceildiv": ceildiv, "align": align, "min": min, "max": max,
          "abs": abs, "sum": sum, "len": len, "sumprod": _sumprod,
          "floor_sum": floor_sum, "lattice_count": lattice_count,
          "out_axis": out_axis, "valid_axis": valid_axis,
          "nz_capacity": layout_capacity}


def evaluate_ast(node, bindings):
    budget = [10000]

    def visit(n, depth=0):
        budget[0] -= 1
        if depth > 100 or budget[0] < 0 or not isinstance(n, dict):
            raise MathDomainError("invalid or oversized math AST")
        op = n.get("op")
        if op == "const":
            value = n["value"]
            if type(value) is not int:
                raise MathDomainError("AST constants must be integers")
            return value
        if op == "symbol":
            if n["name"] not in bindings:
                raise MathDomainError(f"missing symbol: {n['name']}")
            return bindings[n["name"]]
        args = n.get("args", [])
        if op == "if" and len(args) == 3:
            return visit(args[1] if visit(args[0], depth + 1) else args[2], depth + 1)
        if op in ("and", "or"):
            return (all if op == "and" else any)(visit(a, depth + 1) for a in args)
        values = [visit(a, depth + 1) for a in args]
        if op == "neg" and len(values) == 1:
            return -values[0]
        if op in _BINARY and len(values) == 2:
            if op == "pow" and (type(values[1]) is not int or not 0 <= values[1] <= 16):
                raise MathDomainError("unsupported exponent")
            if op in ("div", "floordiv", "mod") and values[1] == 0:
                raise MathDomainError("zero denominator")
            return _BINARY[op](*values)
        if op == "call" and n.get("name") in _CALLS:
            return _CALLS[n["name"]](*values)
        raise MathDomainError(f"unsupported AST node: {op}")

    return visit(node)


def operator_record(name):
    alias = {"quant_batch_matmul_mxa8w4": "quant_batch_matmul_w4a8_mxfp"}
    chosen = alias.get(name, name)
    for op in model()["operators"]:
        if op["id"] == chosen:
            return op
    raise MathDomainError(f"no v4 derivation for {name}")


def classification(name):
    entry = next((x for x in model()["catalog_53"] if x["id"] == name), None)
    try:
        coverage = entry["derivation_coverage"] if entry else operator_record(name)["coverage"]
    except TimeoutError:
        raise
    except MathDomainError:
        coverage = "not_derived_in_this_pack"
    return {"catalog_classification": entry["classification"] if entry else "outside_dsl_catalog_53",
            "derivation_coverage": coverage,
            "runtime_confidence": "analytic_prior_not_source_certification"}


def evaluate_math(name, bindings, *, physical: bool = False, ledger=None):
    """Return a guarded count vector; this API does not infer latency or ABI.

    Physical conditional evaluation requires a separately supplied source event
    ledger. A synthetic fixture is accepted only for symbolic evaluation and is
    never promoted to source evidence. All textual ABI guards remain visible.
    """
    if not isinstance(physical, bool):
        raise MathDomainError("physical must be bool")
    op = operator_record(name)
    values = dict(bindings)
    if physical and op["physical_evaluation_requires_ledger"]:
        if not isinstance(ledger, dict) or ledger.get("evidence") != "source_events":
            raise MathDomainError("physical evaluation requires an explicit source event ledger")
        if not ledger.get("source_locator") or not isinstance(ledger.get("bindings"), dict):
            raise MathDomainError("ledger needs source locator and event bindings")
        values.update(ledger["bindings"])
    for symbol, value in values.items():
        if isinstance(value, (list, tuple)):
            for element in value:
                integer(symbol, element)
        else:
            integer(symbol, value)
    for guard in op["checked_domain_guards"]:
        if not evaluate_ast(guard["ast"], values):
            raise MathDomainError(f"domain guard failed: {guard['expression']}")
    if name == "block_attn_res_update_rms_norm" and values["U"] > min(values["T"], 64):
        raise MathDomainError("at most 64 active AttnRes owners")
    if name == "fused_recurrent_kda_replayssm" and values["v"] < max(values["r"], 64):
        raise MathDomainError("ReplaySSM ABI requires v >= max(r, 64)")
    counts = {key: evaluate_ast(eq["ast"], values) for key, eq in op["equations"].items()}
    buffers = []
    for buf in op["sram_buffers"]:
        raw = evaluate_ast(buf["bytes"]["ast"], values)
        slots = buf.get("slots", 1)
        slots = evaluate_ast(slots["ast"], values) if isinstance(slots, dict) else slots
        buffers.append({"name": buf["name"], "raw_bytes": raw,
                        "aligned_bytes_per_slot": align(raw, model()["pool_alignment_bytes"][buf["pool"]]),
                        "slots": slots})
    return {"counts": counts, "buffers": buffers, "coverage": op["coverage"],
            "confidence": "caller_ledger_unverified" if physical else "guarded_mathematical_counts",
            "abi_guards": tuple(op["domain_guards"]), "full_physical_certification": False}
