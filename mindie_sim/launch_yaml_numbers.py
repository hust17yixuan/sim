"""YAML 1.1 plain-scalar numeric/boolean resolution for both prescan modes.

[Role]
  Reproduce PyYAML 6.0.3 `SafeLoader` integer/float/boolean resolution without
  importing the loader, so the raw fallback and the enabled event walk
  contribute the same loader-normalized spelling for a plain scalar.  The
  launch adapter renders a YAML 1.1 boolean as the lowercase ``true`` or
  ``false`` string, so that spelling is the credential value an env binding
  actually receives.

[Invariants]
  * Binary, octal (`0[0-7_]+`), decimal, hex and sexagesimal integers match
    PyYAML's resolver exactly; `0o` forms are not YAML 1.1 integers.
  * Floats require the YAML 1.1 exponent sign and sexagesimal dot rules, so
    `1.5e3` stays a string while `1.5e+3` is `1500.0`.
  * A resolved spelling is returned only when it is non-empty and finite.
  * Only PyYAML's 18 implicit boolean spellings resolve; `y`/`n` stay
    strings.
[Timeouts & Locks]
  Linear in the scalar spelling; no locks or I/O.
[Side-Effect Policy]
  Pure functions over strings; no environment lookup or global mutation.
"""
from __future__ import annotations

import math
import re

_INTEGER = re.compile(
    r"[-+]?(?:0b[0-1_]+|0[0-7_]+|(?:0|[1-9][0-9_]*)|0x[0-9a-fA-F_]+"
    r"|[1-9][0-9_]*(?::[0-5]?[0-9])+)\Z")
_FLOAT = re.compile(
    r"(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?"
    r"|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?"
    r"|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))\Z")


def _sexagesimal(body, scale):
    total, base = scale(0), 1
    for part in reversed(body.split(":")):
        digit = scale(part)
        # A zero digit contributes nothing even when the base exceeds float.
        if scale is not float or digit != 0.0:
            total += digit * base
        base *= 60
    return total


def numeric_spelling(value):
    """The loader-normalized spelling of one plain YAML 1.1 numeric scalar."""
    text = str(value).strip()
    if not text or any(char in text for char in " \t"):
        return None
    try:
        if _INTEGER.fullmatch(text):
            cleaned = text.replace("_", "")
            sign = -1 if cleaned.startswith("-") else 1
            body = cleaned.lstrip("+-")
            if body == "0":
                number = 0
            elif body.startswith("0b"):
                number = int(body[2:], 2)
            elif body.startswith("0x"):
                number = int(body[2:], 16)
            elif body.startswith("0"):
                number = int(body, 8)
            elif ":" in body:
                number = _sexagesimal(body, int)
            else:
                number = int(body)
            return str(sign * number)
        if _FLOAT.fullmatch(text):
            cleaned = text.replace("_", "").lower()
            sign = -1 if cleaned.startswith("-") else 1
            body = cleaned.lstrip("+-")
            if body == ".inf":
                return "-inf" if sign < 0 else "inf"
            if body == ".nan":
                return "nan"
            if ":" in body:
                number = sign * _sexagesimal(body, float)
            else:
                number = sign * float(body)
            return str(number) if math.isfinite(number) else None
    except TimeoutError:
        raise
    except (TypeError, ValueError, OverflowError):
        return None
    return None


_BOOLEAN = frozenset({
    "yes", "Yes", "YES", "no", "No", "NO",
    "true", "True", "TRUE", "false", "False", "FALSE",
    "on", "On", "ON", "off", "Off", "OFF",
})
_TRUE = frozenset({"yes", "Yes", "YES", "true", "True", "TRUE",
                   "on", "On", "ON"})


def bool_spelling(value):
    """The adapter-normalized ``true``/``false`` spelling, or ``None``."""
    text = str(value)
    if text not in _BOOLEAN:
        return None
    return "true" if text in _TRUE else "false"


__all__ = ["bool_spelling", "numeric_spelling"]
