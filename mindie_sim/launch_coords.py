"""Exact decoded-character source coordinates for launch documents.

[Role]
  Map every decoded character of a JSON/YAML scalar or POSIX shell word back
  to its one-based ``(line, column)`` and byte offset in the outer document.
  ``launch_spans`` composes these maps through the structured parser;
  ``launch_documents`` and ``launch_actions`` pass them through Docker and
  shell unwrapping so a leaf span names the exact source character.

[Invariants]
  * One ``Lines`` index is built per outer document and shared by every scalar;
    no mapper re-scans or re-indexes the document per token.  YAML documents
    configure CRLF/CR/LF/NEL/LS/PS breaks to agree with ``yaml.Reader`` marks;
    JSON, shell and CLI text keep LF-only numbering.
  * ``Lines.bounds`` rejects an interval outside the physical line count or one
    column past that line's content, so a wrong index fails closed.
  * Every returned map has ``len(value) + 1`` entries of
    ``(start_line, start_col, start_offset, end_line, end_col, end_offset)``.
    The end is the half-open source position after the character, so a
    multi-character escape or multiline sequence closes at its real endpoint.
  * A decoder reproduces the parser's decoded value exactly or returns ``()``.
  * JSON escapes start at the escaped character (``\\"`` -> the quote); YAML
    flow folding, block indentation/chomping and shell quoting follow the real
    loaders/tokenizers, including CRLF normalization in lockstep.
  * Shell unquoting keeps one coordinate per emitted character; a substitution
    or an unsupported form leaves the map empty instead of misaligned.
[Timeouts & Locks]
  Linear in scalar/document characters with a shared index; no locks or I/O.
[Side-Effect Policy]
  Pure functions over strings; no mutation, subprocess or environment lookup.
"""
from __future__ import annotations

import bisect
import json

_JSON_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                 "n": "\n", "r": "\r", "t": "\t"}


class Lines:
    """One shared line index for an outer document.

    ``configure(yaml_breaks=True)`` selects YAML's break syntax: ``\\r\\n`` is
    one break, and ``\\n``, a lone ``\\r``, U+0085, U+2028 and U+2029 each start
    a new line exactly as ``yaml.Reader`` counts them.  JSON, shell and CLI text
    keep the default LF-only index, where a carriage return is content.

    The index is built on first use so exactly one ``Lines(text)`` construction
    serves each document while the YAML syntax can still be selected by the
    structured parser before any position is queried.
    """

    __slots__ = ("text", "_yaml", "_starts", "_ends")

    def __init__(self, text, yaml_breaks=False):
        self.text = text
        self._yaml = bool(yaml_breaks)
        self._starts = None
        self._ends = None

    def configure(self, yaml_breaks=True):
        """Select the break syntax before the index is first used."""
        self._yaml = bool(yaml_breaks)
        self._starts = self._ends = None

    def _build(self):
        if self._starts is not None:
            return
        text, size = self.text, len(self.text)
        starts, ends, pos = [0], [], 0
        while pos < size:
            char = text[pos]
            if self._yaml:
                # Mirrors ``launch_yaml_coords.yaml_break`` width/decoding: CRLF
                # is one break whose CR is not a line start of its own.
                if char == "\r":
                    ends.append(pos)
                    if pos + 1 < size and text[pos + 1] == "\n":
                        starts.append(pos + 2)
                        pos += 2
                    else:
                        starts.append(pos + 1)
                        pos += 1
                    continue
                if char in "\n\x85\u2028\u2029":
                    ends.append(pos)
                    starts.append(pos + 1)
                    pos += 1
                    continue
            elif char == "\n":
                ends.append(pos)
                starts.append(pos + 1)
                pos += 1
                continue
            pos += 1
        ends.append(size)
        self._starts, self._ends = starts, ends

    @property
    def starts(self):
        self._build()
        return self._starts

    def position(self, offset):
        starts = self.starts
        row = bisect.bisect_right(starts, offset) - 1
        return row + 1, offset - starts[row] + 1

    def line_len(self, line):
        """Content characters on one one-based line, excluding its break."""
        self._build()
        if not 1 <= line <= len(self._starts):
            return None
        return self._ends[line - 1] - self._starts[line - 1]

    def bounds(self, start_line, start_col, end_line, end_col):
        """True when one half-open interval fits the physical lines."""
        self._build()
        count = len(self._starts)
        if not (1 <= start_line <= end_line <= count):
            return False
        start_len = self._ends[start_line - 1] - self._starts[start_line - 1]
        end_len = self._ends[end_line - 1] - self._starts[end_line - 1]
        return (1 <= start_col <= start_len + 1
                and 1 <= end_col <= end_len + 1)


def line_starts(text):
    """A plain newline index that does not construct/patch ``Lines``."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def at(starts, line, column):
    row = min(max(1, int(line)), len(starts))
    return starts[row - 1] + max(0, int(column) - 1)


def normalize_newlines(text, coords=()):
    """Normalize CRLF/CR to LF, filtering coordinates one-for-one."""
    if "\r" not in text:
        return text, tuple(coords)
    chars, mapped, index = [], [], 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                chars.append("\n")
                if coords:
                    cr, lf = coords[index], coords[index + 1]
                    mapped.append((cr[0], cr[1], cr[2], lf[3], lf[4], lf[5]))
                index += 2
            else:
                chars.append("\n")
                mapped.append(coords[index] if coords else ())
                index += 1
        else:
            chars.append(char)
            mapped.append(coords[index] if coords else ())
            index += 1
    if not coords:
        return "".join(chars), ()
    mapped.append(coords[-1])
    return "".join(chars), tuple(mapped)


def unquote_word(raw, coords=(), escaped_char=False):
    """Decode one POSIX shell word; return text and aligned coordinates.

    A backslash escape consumes two source characters but emits one decoded
    character.  Its map starts at the backslash and closes at the *escaped
    character's* end, so the full consumed source interval survives downstream
    leaf/assignment composition.  ``escaped_char=True`` keeps the legacy
    outer-scalar spelling for callers that already decoded a JSON/YAML string.
    """
    if not raw:
        return "", tuple(coords[:1]) if coords else ()
    if not coords:
        return raw, ()
    out, mapped, index = [], [], 0

    def emit(char, position):
        out.append(char)
        mapped.append(coords[position])

    def emit_escape(char, backslash, escaped):
        start, end = coords[backslash], coords[escaped]
        out.append(char)
        if escaped_char:
            # The raw text was itself decoded from an outer JSON/YAML scalar:
            # its payload character already names that outer source spelling.
            mapped.append(end)
        else:
            # A plain shell source starts the consumed interval at the
            # backslash and closes it at the escaped character's real end.
            mapped.append((start[0], start[1], start[2],
                           end[3], end[4], end[5]))

    while index < len(raw):
        char = raw[index]
        if char == "'":
            index += 1
            while index < len(raw) and raw[index] != "'":
                emit(raw[index], index)
                index += 1
            if index >= len(raw):
                return raw, ()
            index += 1
        elif char == '"':
            index += 1
            while index < len(raw) and raw[index] != '"':
                current = raw[index]
                if current == "\\" and index + 1 < len(raw):
                    nxt = raw[index + 1]
                    if nxt == "\n":
                        index += 2
                        continue
                    if nxt in '$`"\\':
                        emit_escape(nxt, index, index + 1)
                    else:
                        emit("\\", index)
                        emit(nxt, index + 1)
                    index += 2
                    continue
                emit(current, index)
                index += 1
            if index >= len(raw):
                return raw, ()
            index += 1
        elif char == "\\" and index + 1 < len(raw):
            if raw[index + 1] == "\n":
                index += 2
                continue
            emit_escape(raw[index + 1], index, index + 1)
            index += 2
        else:
            emit(char, index)
            index += 1
    mapped.append(coords[len(raw)])
    return "".join(out), tuple(mapped)


def raw_word_coords(raw, line, column, offset):
    """One exact source interval per raw shell-word character, plus the end."""
    coords = []
    for char in raw:
        start_line, start_column, start_offset = line, column, offset
        if char == "\n":
            line, column = line + 1, 1
        else:
            column += 1
        offset += 1
        coords.append((start_line, start_column, start_offset,
                       line, column, offset))
    coords.append((line, column, offset, line, column, offset))
    return tuple(coords)


def shell_word_coords(raw, text, line, column, offset, escaped_char=False):
    """Decoded shell-word coordinates aligned with ``text``, or ``()``."""
    decoded, mapped = unquote_word(
        raw, raw_word_coords(raw, line, column, offset), escaped_char)
    return mapped if decoded == text else ()


def json_char_coords(text, start, end, lines):
    """Decoded-character -> exact outer source interval for one JSON token."""
    pos, stop = start + 1, end - 1
    line, column = lines.position(pos)
    coords = []
    while pos < stop:
        char = text[pos]
        if char == "\\" and pos + 1 < stop:
            nxt = text[pos + 1]
            if nxt == "u" and pos + 6 <= stop:
                try:
                    code = int(text[pos + 2:pos + 6], 16)
                except ValueError:
                    return ()
                if (0xD800 <= code <= 0xDBFF and pos + 12 <= stop
                        and text[pos + 6:pos + 8] == "\\u"):
                    try:
                        low = int(text[pos + 8:pos + 12], 16)
                    except ValueError:
                        return ()
                    if 0xDC00 <= low <= 0xDFFF:
                        coords.append((line, column, pos,
                                       line, column + 12, pos + 12))
                        pos += 12
                        column += 12
                        continue
                coords.append((line, column, pos, line, column + 6, pos + 6))
                pos += 6
                column += 6
                continue
            if nxt not in _JSON_ESCAPES:
                return ()
            # Start names the escaped character; the end closes the whole
            # two-character source escape.
            coords.append((line, column + 1, pos + 1,
                           line, column + 2, pos + 2))
            pos += 2
            column += 2
            continue
        start_line, start_column = line, column
        if char == "\n":
            line, column = line + 1, 1
        else:
            column += 1
        coords.append((start_line, start_column, pos, line, column, pos + 1))
        pos += 1
    try:
        size = len(json.loads(text[start:end]))
    except ValueError:
        return ()
    # The final entry is the half-open boundary after the closing quote.
    coords.append((line, column + 1, pos + 1, line, column + 1, pos + 1))
    return tuple(coords) if len(coords) == size + 1 else ()


from .launch_yaml_coords import (
    yaml_block_coords, yaml_double_value, yaml_lines, yaml_scalar_coords,
    yaml_single_value,
)


__all__ = [
    "Lines", "at", "json_char_coords", "line_starts", "normalize_newlines",
    "raw_word_coords", "shell_word_coords", "unquote_word", "yaml_block_coords", "yaml_double_value", "yaml_lines",
    "yaml_scalar_coords", "yaml_single_value",
]
