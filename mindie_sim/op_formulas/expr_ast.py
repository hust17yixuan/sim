"""Typed work algebra: one AST per quantity, rendered and evaluated exactly.

[Role]
  Own the single source of truth for every symbolic work/byte quantity.  An
  ``Expr`` node is rendered into the human-readable ``*_expr`` string and
  evaluated over the same ``bindings`` mapping, so a card can never publish a
  formula whose arithmetic disagrees with its number.  ``evaluate_formula_expr``
  is the public, evaluator-free verifier used by the engine invariant and tests.

[Invariants]
  * ``evaluate_formula_expr`` accepts only literals, names, ``+ - * / //`` and
    the whitelisted integer helpers; no attribute, subscript, lambda, call to a
    builtin outside the whitelist, or ``eval`` can execute.
  * ``ceil_div``/``align_up``/``log2_ceil`` are exact integer functions; ``/``
    is the only float-producing operator and is used solely for constants.
  * ``render`` is deterministic and re-parses to the same value for the same
    bindings; parenthesisation is precedence-driven, never ad hoc.
  * ``rename`` rewrites symbols and list-symbol names together so a composite
    chain keeps every component binding fully qualified.
  * Unknown names or list symbols raise ``FormulaError`` instead of returning a
    silent zero.

[Timeouts & Locks]
  Pure in-memory trees; every operation is linear in node count.

[Side-Effect Policy]
  Immutable nodes; no mutation of bindings or global state.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import FormulaError


def _num(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormulaError(f"{name} must be a real number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise FormulaError(f"{name} must be finite, got {value!r}")
    return value


def ceil_div_int(numerator: int, denominator: int) -> int:
    if isinstance(numerator, float) and numerator.is_integer():
        numerator = int(numerator)
    if isinstance(denominator, float) and denominator.is_integer():
        denominator = int(denominator)
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise FormulaError("ceil_div requires integers")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise FormulaError("ceil_div requires integers")
    if denominator <= 0:
        raise FormulaError("ceil_div denominator must be > 0")
    return -(-numerator // denominator)


def _align_up_int(value: int, granule: int) -> int:
    return ceil_div_int(value, granule) * granule


def log2_ceil_int(value: int) -> int:
    """Exact ``ceil(log2(value))``; ``log2_ceil(1) == 0`` and ``(8) == 3``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormulaError("log2_ceil requires an integer")
    if value < 1:
        raise FormulaError("log2_ceil requires value >= 1")
    return (value - 1).bit_length()


def _seq(value: Any, name: str) -> Sequence[int | float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FormulaError(f"{name} must be bound to a numeric sequence")
    out: list[int | float] = []
    for item in value:
        out.append(_num(item, name))
    return out


def count_nonzero_list(values: Iterable[Any]) -> int:
    return sum(1 for item in values if item != 0)


# --------------------------------------------------------------------------- #
# Node types
# --------------------------------------------------------------------------- #
class Expr:
    """Base class with operator sugar and the two authoritative operations."""

    precedence = 3

    def value(self, bindings: Mapping[str, Any]) -> int | float:  # pragma: no cover
        raise NotImplementedError

    def render(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def rename(self, mapping: Mapping[str, str]) -> "Expr":  # pragma: no cover
        raise NotImplementedError

    def syms(self) -> set[str]:
        return set()

    def __add__(self, other: Any) -> "Expr":
        return BinOp("+", self, as_expr(other))

    def __radd__(self, other: Any) -> "Expr":
        return BinOp("+", as_expr(other), self)

    def __sub__(self, other: Any) -> "Expr":
        return BinOp("-", self, as_expr(other))

    def __rsub__(self, other: Any) -> "Expr":
        return BinOp("-", as_expr(other), self)

    def __mul__(self, other: Any) -> "Expr":
        return BinOp("*", self, as_expr(other))

    def __rmul__(self, other: Any) -> "Expr":
        return BinOp("*", as_expr(other), self)

    def __truediv__(self, other: Any) -> "Expr":
        return BinOp("/", self, as_expr(other))

    def __rtruediv__(self, other: Any) -> "Expr":
        return BinOp("/", as_expr(other), self)

    def __floordiv__(self, other: Any) -> "Expr":
        return BinOp("//", self, as_expr(other))

    def __rfloordiv__(self, other: Any) -> "Expr":
        return BinOp("//", as_expr(other), self)

    def __neg__(self) -> "Expr":
        return Neg(self)

    def at(self, bindings: Mapping[str, Any]) -> int | float:
        return self.value(bindings)


@dataclass(frozen=True)
class Const(Expr):
    raw: int | float

    def __post_init__(self) -> None:
        _num(self.raw, "constant")

    def value(self, bindings: Mapping[str, Any]) -> int | float:
        return self.raw

    def render(self) -> str:
        if isinstance(self.raw, bool):  # pragma: no cover - _num rejects
            raise FormulaError("bool constant is not a work quantity")
        if isinstance(self.raw, int):
            return str(self.raw)
        return repr(float(self.raw))

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        return self


@dataclass(frozen=True)
class Sym(Expr):
    name: str

    def __post_init__(self) -> None:
        if not str(self.name).isidentifier():
            raise FormulaError(f"symbol {self.name!r} is not a valid identifier")

    def value(self, bindings: Mapping[str, Any]) -> int | float:
        if self.name not in bindings:
            raise FormulaError(f"expression symbol {self.name!r} is unbound")
        return _num(bindings[self.name], self.name)

    def render(self) -> str:
        return self.name

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        return Sym(mapping.get(self.name, self.name))

    def syms(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class ListSym(Expr):
    """A bound per-expert histogram; never rendered as bare arithmetic."""

    name: str

    def __post_init__(self) -> None:
        if not str(self.name).isidentifier():
            raise FormulaError(f"list symbol {self.name!r} is not an identifier")

    def value(self, bindings: Mapping[str, Any]) -> Sequence[int | float]:
        if self.name not in bindings:
            raise FormulaError(f"list symbol {self.name!r} is unbound")
        return _seq(bindings[self.name], self.name)

    def render(self) -> str:  # pragma: no cover - only used inside calls
        return self.name

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        return ListSym(mapping.get(self.name, self.name))

    def syms(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Neg(Expr):
    operand: Expr
    precedence = 2

    def value(self, bindings: Mapping[str, Any]) -> int | float:
        return -self.operand.value(bindings)

    def render(self) -> str:
        inner = self.operand.render()
        # ``-a + b`` parses as ``(-a) + b``, not ``-(a + b)``; and
        # ``-a // b`` rounds differently from ``-(a // b)``.  Parenthesise every
        # binary operand so unary minus is semantics-preserving.
        if isinstance(self.operand, BinOp) or (
                self.operand.precedence < self.precedence):
            inner = f"({inner})"
        return f"-{inner}"

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        return Neg(self.operand.rename(mapping))

    def syms(self) -> set[str]:
        return self.operand.syms()


_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "//": 2}


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in _PREC:
            raise FormulaError(f"unsupported operator {self.op!r}")

    @property
    def precedence(self) -> int:  # type: ignore[override]
        return _PREC[self.op]

    def value(self, bindings: Mapping[str, Any]) -> int | float:
        left = self.left.value(bindings)
        right = self.right.value(bindings)
        if self.op == "+":
            return left + right
        if self.op == "-":
            return left - right
        if self.op == "*":
            return left * right
        if self.op == "/":
            if right == 0:
                raise FormulaError("division by zero in formula expression")
            return left / right
        if right == 0:
            raise FormulaError("floor division by zero in formula expression")
        return int(left) // int(right)

    def render(self) -> str:
        left = self.left.render()
        right = self.right.render()
        if self.left.precedence < self.precedence:
            left = f"({left})"
        # Float arithmetic is not associative: ``1e16 + (-1e16 + 1)`` is 0,
        # while ``1e16 + -1e16 + 1`` is 1.  Always preserve equal-precedence
        # right-child grouping; higher-precedence right children stay bare.
        if self.right.precedence <= self.precedence:
            right = f"({right})"
        return f"{left} {self.op} {right}"

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        return BinOp(self.op, self.left.rename(mapping), self.right.rename(mapping))

    def syms(self) -> set[str]:
        return self.left.syms() | self.right.syms()


_FUNCS: dict[str, Callable[..., Any]] = {
    "ceil_div": ceil_div_int,
    "align_up": _align_up_int,
    "log2_ceil": log2_ceil_int,
    "min": min,
    "max": max,
}


@dataclass(frozen=True)
class Call(Expr):
    func: str
    args: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if self.func not in {"ceil_div", "align_up", "log2_ceil", "min", "max",
                             "sum_vec", "sum_align_up", "count_nonzero"}:
            raise FormulaError(f"unsupported formula function {self.func!r}")
        expected = {"ceil_div": 2, "align_up": 2, "log2_ceil": 1,
                    "sum_vec": 1, "sum_align_up": 2, "count_nonzero": 1}
        if self.func in expected and len(self.args) != expected[self.func]:
            raise FormulaError(f"{self.func} expects {expected[self.func]} args")
        if self.func in {"min", "max"} and len(self.args) < 1:
            raise FormulaError(f"{self.func} expects at least one arg")

    def value(self, bindings: Mapping[str, Any]) -> int | float:
        if self.func == "sum_vec":
            return int(sum(_seq(self.args[0].value(bindings), "sum_vec")))
        if self.func == "count_nonzero":
            return int(count_nonzero_list(_seq(self.args[0].value(bindings),
                                               "count_nonzero")))
        if self.func == "sum_align_up":
            values = _seq(self.args[0].value(bindings), "sum_align_up")
            granule = self.args[1].value(bindings)
            return int(sum(_align_up_int(int(item), int(granule)) for item in values))
        args = [arg.value(bindings) for arg in self.args]
        if self.func in {"min", "max"} and len(args) == 1:
            return args[0]
        return _FUNCS[self.func](*args)

    def render(self) -> str:
        return f"{self.func}({', '.join(arg.render() for arg in self.args)})"

    def rename(self, mapping: Mapping[str, str]) -> "Expr":
        return Call(self.func, tuple(arg.rename(mapping) for arg in self.args))

    def syms(self) -> set[str]:
        out: set[str] = set()
        for arg in self.args:
            out |= arg.syms()
        return out


def as_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    return Const(_num(value, "literal"))


# --------------------------------------------------------------------------- #
# Public constructors
# --------------------------------------------------------------------------- #
def sym(name: str) -> Sym:
    return Sym(name)


def lst(name: str) -> ListSym:
    return ListSym(name)


ZERO = Const(0)


def ceil_div(a: Any, b: Any) -> Expr:
    return Call("ceil_div", (as_expr(a), as_expr(b)))


def align_up(a: Any, granule: Any) -> Expr:
    return Call("align_up", (as_expr(a), as_expr(granule)))


def log2_ceil(a: Any) -> Expr:
    return Call("log2_ceil", (as_expr(a),))


def min_expr(*args: Any) -> Expr:
    return Call("min", tuple(as_expr(arg) for arg in args))


def max_expr(*args: Any) -> Expr:
    return Call("max", tuple(as_expr(arg) for arg in args))


def sum_vec(name: str | ListSym) -> Expr:
    return Call("sum_vec", (name if isinstance(name, Expr) else lst(name),))


def sum_align_up(name: str | ListSym, granule: Any) -> Expr:
    base = name if isinstance(name, Expr) else lst(name)
    return Call("sum_align_up", (base, as_expr(granule)))


def count_nonzero(name: str | ListSym) -> Expr:
    base = name if isinstance(name, Expr) else lst(name)
    return Call("count_nonzero", (base,))


# --------------------------------------------------------------------------- #
# Safe verification
# --------------------------------------------------------------------------- #
_BIN_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "//"}


def _eval_node(node: ast.AST, bindings: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError("formula literal must be int or float")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise FormulaError(f"unbound symbol {node.id!r} in formula")
        return bindings[node.id]
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise FormulaError("unsupported binary operator in formula")
        left = _eval_node(node.left, bindings)
        right = _eval_node(node.right, bindings)
        if isinstance(left, bool) or isinstance(right, bool):
            raise FormulaError("bool is not a work quantity")
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if right == 0:
                raise FormulaError("division by zero in formula")
            return left / right
        return int(left) // int(right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, bindings)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_eval_node(node.operand, bindings)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise FormulaError("formula calls must be whitelisted positional functions")
        name = node.func.id
        args = [_eval_node(arg, bindings) for arg in node.args]
        if name == "sum_vec":
            return int(sum(_seq(args[0], "sum_vec")))
        if name == "count_nonzero":
            return int(count_nonzero_list(_seq(args[0], "count_nonzero")))
        if name == "sum_align_up":
            return int(sum(_align_up_int(int(item), int(args[1]))
                           for item in _seq(args[0], "sum_align_up")))
        func = _FUNCS.get(name)
        if func is None:
            raise FormulaError(f"unknown formula function {name!r}")
        if name in {"min", "max"} and len(args) == 1:
            return args[0]
        return func(*args)
    raise FormulaError(f"unsupported formula syntax {type(node).__name__}")


def evaluate_formula_expr(expr_str: str, bindings: Mapping[str, Any]) -> int | float:
    """Evaluate one rendered formula against ``bindings`` with no ``eval``."""
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise FormulaError("formula expression must be a non-empty string")
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise FormulaError(f"formula is not valid Python syntax: {exc}") from exc
    return _eval_node(tree.body, bindings)


def close(left: Any, right: Any, *, rel: float = 1e-9) -> bool:
    """Exact integer equality or relative float agreement for the card check."""
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= rel * scale


__all__ = [
    "BinOp", "Call", "Const", "Expr", "ListSym", "Neg", "Sym", "ZERO",
    "align_up", "as_expr", "ceil_div", "ceil_div_int", "close", "count_nonzero",
    "count_nonzero_list", "evaluate_formula_expr", "log2_ceil", "log2_ceil_int",
    "lst", "max_expr", "min_expr", "sym", "sum_align_up", "sum_vec",
]
