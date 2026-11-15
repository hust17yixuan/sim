"""Strict scalar validation, integer geometry and per-buffer byte algebra.

[Role]
  Own the pre-coercion validation layer and the deterministic integer helpers
  shared by every operator builder: dimension checks that reject bools,
  fractional floats, strings, NaN and infinities; Cube fractal alignment; wave
  quantization; burst coalescing; and exact per-buffer compulsory/aligned byte
  expressions built on ``expr_ast``.

[Invariants]
  * ``strict_int`` accepts only ``int`` (never ``bool``, ``float`` or ``str``);
    every registry/engine/annotator parameter passes through it before any
    arithmetic, so ``int(1.75)`` and ``int(True)`` are unrepresentable.
  * ``strict_finite_float`` rejects ``bool`` and every non-finite value;
    ``strict_tol`` enforces ``0.0 <= tol < 1.0``.
  * ``wave_efficiency`` is in (0, 1]; ``align_up`` is exact; ``granule >=
    compulsory`` holds for every transfer helper.
  * A byte expression is built from the same AST that is rendered, never from a
    second hand-written string.

[Timeouts & Locks]
  Pure arithmetic; no I/O, loops over tensor data or global state.

[Side-Effect Policy]
  No mutation of inputs; returns fresh values and immutable expressions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from .errors import FormulaError, FormulaValidationError
from .expr_ast import (
    Expr, align_up as _align_up_ast, as_expr, ceil_div as _ceil_div_ast,
)
from .hardware import dtype_bytes


def strict_int(value: Any, name: str, *, minimum: int | None = None,
               maximum: int | None = None) -> int:
    """Accept exactly ``int``; never truncate a float or coerce a string."""
    if isinstance(value, bool):
        raise FormulaValidationError(f"{name} must be an integer, got a bool")
    if not isinstance(value, int):
        raise FormulaValidationError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}")
    if minimum is not None and value < minimum:
        raise FormulaValidationError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise FormulaValidationError(f"{name} must be <= {maximum}, got {value}")
    return int(value)


def strict_nonneg_int(value: Any, name: str) -> int:
    return strict_int(value, name, minimum=0)


def strict_pos_int(value: Any, name: str) -> int:
    return strict_int(value, name, minimum=1)


def strict_bool(value: Any, name: str) -> bool:
    """Accept exactly ``bool``; truthy strings/ints are rejected."""
    if type(value) is not bool:
        raise FormulaValidationError(
            f"{name} must be a bool, got {type(value).__name__}")
    return value


def strict_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Accept only a real mapping (never a list, tuple or scalar)."""
    if not isinstance(value, Mapping):
        raise FormulaValidationError(
            f"{name} must be a mapping, got {type(value).__name__}")
    return value


def strict_finite_float(value: Any, name: str, *,
                        minimum: float | None = None,
                        maximum: float | None = None,
                        exclusive_min: bool = False) -> float:
    if isinstance(value, bool):
        raise FormulaValidationError(f"{name} must be numeric, got a bool")
    if not isinstance(value, (int, float)):
        raise FormulaValidationError(
            f"{name} must be numeric, got {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormulaValidationError(
            f"{name} cannot be represented as a finite float") from exc
    if not math.isfinite(number):
        raise FormulaValidationError(f"{name} must be finite")
    if minimum is not None:
        if exclusive_min and number <= minimum:
            raise FormulaValidationError(f"{name} must be > {minimum}, got {number}")
        if not exclusive_min and number < minimum:
            raise FormulaValidationError(f"{name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise FormulaValidationError(f"{name} must be <= {maximum}, got {number}")
    return number


def strict_nonneg_float(value: Any, name: str) -> float:
    return strict_finite_float(value, name, minimum=0.0)


def strict_tol(value: Any, name: str = "tol") -> float:
    return strict_finite_float(value, name, minimum=0.0, maximum=0.999999999,
                               exclusive_min=False)


def strict_int_list(value: Any, name: str, *, length: int | None = None
                    ) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FormulaValidationError(f"{name} must be a sequence of integers")
    out = tuple(strict_int(item, f"{name}[{index}]")
                for index, item in enumerate(value))
    if length is not None and len(out) != length:
        raise FormulaValidationError(
            f"{name} must have length {length}, got {len(out)}")
    return out


# Compatibility aliases: the strict behaviour is the only behaviour.
def as_int(value: Any, name: str, *, minimum: int = 0) -> int:
    return strict_int(value, name, minimum=minimum)


def as_float(value: Any, name: str, *, positive: bool = False) -> float:
    return strict_finite_float(value, name, minimum=0.0,
                               exclusive_min=positive)


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise FormulaError("denominator must be > 0")
    return -(-int(numerator) // int(denominator))


def align_up(value: int, granule: int) -> int:
    granule = int(granule)
    if granule < 1:
        raise FormulaError("granule must be >= 1")
    return ceil_div(int(value), granule) * granule


def wave_efficiency(n_tiles: int, n_cores: int) -> tuple[int, float]:
    """Return ``(n_waves, n_tiles / (n_waves * n_cores))`` in (0, 1]."""
    tiles = int(n_tiles)
    cores = int(n_cores)
    if tiles < 1 or cores < 1:
        raise FormulaError("n_tiles and n_cores must be >= 1")
    waves = ceil_div(tiles, cores)
    return waves, min(1.0, tiles / float(waves * cores))


def coalesce_efficiency(contiguous_row_bytes: float, burst_bytes: int) -> float:
    row = float(contiguous_row_bytes)
    burst = float(burst_bytes)
    if burst <= 0:
        raise FormulaError("burst_bytes must be > 0")
    if row <= 0:
        return 1.0
    return max(1e-6, min(1.0, row / burst))


def choose_block(m: int, n: int, k: int) -> tuple[int, int, int]:
    tile_m = 128 if int(m) > 16 else 16
    tile_n = 256 if int(n) > 2048 else 128
    tile_k = 128 if int(k) > 256 else max(16, align_up(int(k), 16))
    return tile_m, tile_n, tile_k


def tile_geometry(m: int, n: int, k: int, tile_m: int | None = None,
                  tile_n: int | None = None, tile_k: int | None = None
                  ) -> dict[str, int]:
    m_i, n_i, k_i = int(m), int(n), int(k)
    if tile_m is None or tile_n is None or tile_k is None:
        default_m, default_n, default_k = choose_block(m_i, n_i, k_i)
        tile_m = default_m if tile_m is None else tile_m
        tile_n = default_n if tile_n is None else tile_n
        tile_k = default_k if tile_k is None else tile_k
    tile_m = align_up(int(tile_m), 16)
    tile_n = align_up(int(tile_n), 16)
    tile_k = align_up(int(tile_k), 16)
    m_tiled, n_tiled, k_tiled = align_up(m_i, 16), align_up(n_i, 16), align_up(k_i, 16)
    return {
        "m": m_i, "n": n_i, "k": k_i,
        "m_tiled": m_tiled, "n_tiled": n_tiled, "k_tiled": k_tiled,
        "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
        "m_blocks": ceil_div(m_tiled, tile_m),
        "n_blocks": ceil_div(n_tiled, tile_n),
        "k_blocks": ceil_div(k_tiled, tile_k),
        "n_tiles": max(1, ceil_div(m_tiled, tile_m) * ceil_div(n_tiled, tile_n)),
    }


def tensor_bytes(shape: Iterable[int], bpe: float) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(math.ceil(total * float(bpe)))


def row_bytes(last_dim: int, bpe: float) -> int:
    return int(math.ceil(int(last_dim) * float(bpe)))


def granule_for_rows(rows: int, row_width_bytes: float, granule: int) -> int:
    return int(rows) * align_up(int(math.ceil(max(0.0, float(row_width_bytes)))),
                                int(granule))


def l2_split(compulsory_bytes: int, l2_hot_bytes: int) -> tuple[int, int]:
    total = int(compulsory_bytes)
    hot = min(total, int(l2_hot_bytes))
    return total - hot, hot


def maybe_l2_hot(steady_l2_warm: bool, tensor_size: int, budget: int) -> int:
    return int(tensor_size) if steady_l2_warm and int(tensor_size) <= int(budget) else 0


def us_from_work(work: float, rate_per_s: float) -> float:
    if rate_per_s <= 0:
        raise FormulaError("rate must be > 0")
    return max(0.0, float(work)) / float(rate_per_s) * 1e6


def bpe(dtype: str | None, default: float | None = None) -> float:
    if dtype is None:
        if default is None:
            raise FormulaValidationError("dtype is required (no silent default)")
        return float(default)
    return dtype_bytes(dtype)


# --------------------------------------------------------------------------- #
# Exact byte expressions
# --------------------------------------------------------------------------- #
def bytes_of(elements: Expr | int, elem_bytes: float) -> Expr:
    """``ceil(elements * elem_bytes)`` as an exact integer expression."""
    fraction = Fraction(float(elem_bytes)).limit_denominator(16)
    num, den = fraction.numerator, fraction.denominator
    expr = as_expr(elements)
    if den == 1:
        return expr if num == 1 else expr * num
    return _ceil_div_ast(expr * num, den)


def row_bytes_expr(width: Expr | int, elem_bytes: float) -> Expr:
    return bytes_of(width, elem_bytes)


def rows_aligned(rows: Expr | int, width: Expr | int, elem_bytes: float,
                 granule: int) -> Expr:
    """Aligned traffic for ``rows`` contiguous rows of ``width`` elements."""
    width_bytes = row_bytes_expr(width, elem_bytes)
    return as_expr(rows) * _align_up_ast(width_bytes, int(granule))


def contiguous_aligned(elements: Expr | int, elem_bytes: float,
                       granule: int) -> Expr:
    """Aligned traffic for one contiguous (possibly 1-D) buffer."""
    return _align_up_ast(bytes_of(elements, elem_bytes), int(granule))


# --------------------------------------------------------------------------- #
# Shared sequence-length / state-dtype contracts (direct and native ABI paths)
# --------------------------------------------------------------------------- #
SEQUENCE_LENGTH_ALIASES: tuple[tuple[str, ...], ...] = (
    ("actual_seq_lengths", "actual_seq_lens", "actual_seq_lens_q",
     "actual_seq_lengths_q", "actual_seq_qlen", "seq_lens_q", "seq_lens"),
    ("actual_seq_lengths_kv", "actual_seq_lens_kv", "actual_seq_kvlen",
     "seq_lens_kv"),
)

_SEQ_ENCODINGS = {
    "per_seq": "lengths", "per_sequence": "lengths", "lengths": "lengths",
    "length": "lengths", "1": "lengths",
    "cumsum": "cumsum", "cumulative": "cumsum", "prefix": "cumsum", "0": "cumsum",
}

_STATE_DTYPES = frozenset({"fp32", "float32", "bf16", "bfloat16", "fp16",
                           "float16"})


def strict_state_dtype(value: Any, name: str = "state_dtype") -> str:
    """Accept only declared floating-point state storage types."""
    if not isinstance(value, str) or not value.strip():
        raise FormulaValidationError(f"{name} must be a floating dtype name")
    text = value.strip().lower()
    if text not in _STATE_DTYPES:
        raise FormulaValidationError(
            f"{name} must be a floating dtype, got {value!r}")
    return text


def sequence_encoding(value: Any, name: str = "actual_seq_lengths_type") -> str:
    """Normalize a declared encoding spelling to ``lengths`` or ``cumsum``."""
    if value is None or value == "":
        return "lengths"
    text = str(value).strip().lower()
    encoding = _SEQ_ENCODINGS.get(text)
    if encoding is None:
        raise FormulaValidationError(f"{name} is unknown: {value!r}")
    return encoding


def decode_sequence_lengths(values: Any, *, encoding: Any = "lengths",
                            total_tokens: int | None = None,
                            expected_count: int | None = None,
                            label: str = "sequence_lengths") -> tuple[int, ...]:
    """Decode per-sequence lengths or cumulative offsets into exact lengths.

    ``[2, 5]`` and ``[3, 7]`` in cumulative form decode to ``[2, 3]`` and
    ``[3, 4]``; a leading-zero vector of length ``batch + 1`` is accepted.
    Negative steps, non-monotone cumulative vectors and a terminal offset beyond
    ``total_tokens`` are rejected.  The same helper serves the direct formula
    registry and the native ABI bridge so both routes share one contract.
    """
    numbers = strict_int_list(values, label)
    if not numbers:
        raise FormulaValidationError(f"{label} must not be empty")
    mode = sequence_encoding(encoding)
    if any(item < 0 for item in numbers):
        raise FormulaValidationError(f"{label} must be non-negative")
    if mode == "cumsum":
        body = numbers
        if expected_count is not None and len(numbers) == expected_count + 1:
            if numbers[0] != 0:
                raise FormulaValidationError(
                    f"{label} cumulative vector is missing the leading zero")
            body = numbers[1:]
        elif len(numbers) > 1 and numbers[0] == 0 and (
                expected_count is None or len(numbers) - 1 == expected_count):
            body = numbers[1:]
        lengths = tuple(body[0] if index == 0 else body[index] - body[index - 1]
                        for index in range(len(body)))
        if any(step < 0 for step in lengths):
            raise FormulaValidationError(f"{label} cumulative steps decrease")
        terminal = body[-1] if body else 0
        if total_tokens is not None and terminal > int(total_tokens):
            raise FormulaValidationError(
                f"{label} terminal sum {terminal} exceeds {total_tokens}")
    else:
        lengths = numbers
        total = sum(numbers)
        if total_tokens is not None and total > int(total_tokens):
            raise FormulaValidationError(
                f"{label} total {total} exceeds {total_tokens}")
    if expected_count is not None and len(lengths) != int(expected_count):
        raise FormulaValidationError(
            f"{label} must have {expected_count} entries, got {len(lengths)}")
    if sum(lengths) <= 0:
        raise FormulaValidationError(f"{label} has zero active tokens")
    return lengths


def normalize_causal_offset(q_len: int, kv_len: int,
                           causal_offset: int | None = None) -> int:
    """Right-aligned cache-append offset: ``max(0, kv - q)`` unless explicit."""
    q_i, kv_i = int(q_len), int(kv_len)
    if causal_offset is not None:
        offset = strict_int(causal_offset, "causal_offset", minimum=0)
        return offset
    return max(0, kv_i - q_i)


def attended_pairs(q_len: int, kv_len: int, *, causal: bool,
                   causal_offset: int | None = None) -> int:
    """Exact number of visible ``(query, key)`` pairs for one sequence.

    Decode at the end of a full cache (``q=1, kv=4096``) sees all 4096 keys;
    square causal prefill sees the inclusive triangle ``S*(S+1)/2``; chunked
    prefill ``q < kv`` sees ``q*P + q*(q+1)/2``.  The closed form is exactly
    ``sum_t min(kv, offset + t + 1)``.
    """
    q_i, kv_i = int(q_len), int(kv_len)
    if q_i <= 0 or kv_i <= 0:
        return 0
    if not causal:
        return q_i * kv_i
    offset = normalize_causal_offset(q_i, kv_i, causal_offset)
    full_at = kv_i - offset - 1
    partial = min(q_i, max(0, full_at))
    partial_sum = partial * offset + partial * (partial + 1) // 2
    return partial_sum + (q_i - partial) * kv_i


def active_causal_tiles(q_len: int, kv_len: int, tile_q: int, tile_kv: int,
                        causal_offset: int | None = None) -> int:
    """Count exact active ``(tile_q, tile_kv)`` blocks under causal masking."""
    q_i, kv_i = int(q_len), int(kv_len)
    if q_i <= 0 or kv_i <= 0:
        return 0
    if tile_q < 1 or tile_kv < 1:
        raise FormulaError("tiles must be >= 1")
    offset = normalize_causal_offset(q_i, kv_i, causal_offset)
    blocks = 0
    for start in range(0, q_i, int(tile_q)):
        end = min(q_i - 1, start + int(tile_q) - 1)
        limit = offset + end + 1
        if limit <= 0:
            continue
        blocks += min(ceil_div(kv_i, int(tile_kv)), ceil_div(limit, int(tile_kv)))
    return blocks


__all__ = [
    "FormulaError", "FormulaValidationError", "SEQUENCE_LENGTH_ALIASES",
    "align_up", "as_float", "as_int", "bpe", "bytes_of", "ceil_div",
    "choose_block", "coalesce_efficiency", "contiguous_aligned",
    "decode_sequence_lengths", "granule_for_rows", "l2_split", "maybe_l2_hot",
    "row_bytes", "row_bytes_expr", "rows_aligned", "sequence_encoding",
    "strict_bool", "strict_finite_float", "strict_int", "strict_int_list",
    "strict_mapping", "strict_nonneg_float", "strict_nonneg_int",
    "strict_pos_int", "strict_state_dtype", "strict_tol", "tensor_bytes",
    "active_causal_tiles", "attended_pairs", "normalize_causal_offset",
    "tile_geometry", "us_from_work", "wave_efficiency",
]
