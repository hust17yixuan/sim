"""Raw YAML block-scalar cursor scanning and exact loader decoding.

[Role]
  Consume one literal (`|`) or folded (`>`) block scalar, or one multiline
  block-context plain scalar, from a raw YAML document without a YAML library.
  Return the exact PyYAML 6.0.3 ``SafeLoader`` value/whitespace run and the
  cursor offset, so ``launch_yaml_raw`` can include decoded block command and
  argument values in credential discovery and sibling-field composition.

[Invariants]
  * The value is produced by the same ``block_stream``/``block_value`` pair
    that maps coordinates, so a raw credential string and its enabled-mode
    coordinate decoder can never disagree about folding or chomping.
  * A more-indented whitespace line is content: only a line at or below the
    content indent is an empty line.  CRLF/CR/LF/NEL normalize to LF while
    U+2028/U+2029 keep their identity; clip keeps the final content line's
    break, strip removes it, and keep retains every trailing break.
  * The returned offset is the start of the first non-block line, so a caller
    never advances a second time over the line that follows a block.
  * Scanning is linear: each physical line is read once and a line below the
    content indent ends the block without being consumed.

[Timeouts & Locks]
  Linear in the block's characters; no locks, I/O, devices or env lookup.
[Side-Effect Policy]
  Pure functions over text; no mutation, subprocess or global state.
"""
from __future__ import annotations

from .launch_yaml_coords import (
    block_layout, block_stream, block_value, is_break, yaml_break,
)


def line_end(text, index):
    """Offset of the first YAML break at or after ``index``."""
    size = len(text)
    while index < size and not is_break(text[index]):
        index += 1
    return index


def indent_at(text, index):
    """``(leading_space_width, content_offset)`` for the line at ``index``."""
    start = index
    while index < len(text) and text[index] in " \t":
        index += 1
    return index - start, index


def next_line(text, index):
    """Advance past one YAML break without re-reading earlier text."""
    size = len(text)
    cursor = index
    while cursor < size and not is_break(text[cursor]):
        cursor += 1
    if cursor >= size:
        return size
    consumed, _break = yaml_break(text, cursor, size)
    return cursor + consumed


def block_scalar(text, index, parent, limit=None):
    """``(value, end, content_lines)`` for the block scalar at ``index``.

    ``parent`` is the indentation of the line that owns the block; the loader
    content indent is at least ``parent + 1`` (or the explicit header digit).
    """
    stop = len(text) if limit is None else min(len(text), limit)
    chunks, tail, trailing, chomp = block_stream(text, index, stop, parent)
    value, _chosen = block_value(chunks, tail, trailing, chomp)
    indent, _folded, _chomp, rows, _leading, end = block_layout(
        text, index, stop, parent)
    content_lines = [row["line"][indent:] for row in rows]
    return value, end, content_lines


def plain_run_end(text, index):
    """PyYAML's plain-scalar run end inside a block context."""
    size, cursor = len(text), index
    while cursor < size:
        char = text[cursor]
        if char in " \t" or is_break(char):
            break
        if char == ":" and (cursor + 1 >= size
                            or text[cursor + 1] in " \t"
                            or is_break(text[cursor + 1])):
            break
        if char == "#" and cursor == index:
            break
        cursor += 1
    return cursor


def plain_line_span(text, index):
    """One plain-scalar continuation line: ``(content, break_offset)``.

    The whole physical line participates in the fold, not just its first
    token.  A whitespace-prefixed ``#`` comment and a ``: `` mapping indicator
    still terminate the plain scalar, matching PyYAML's scanner.
    """
    stop = line_end(text, index)
    line = text[index:stop]
    cut = line.find(" #")
    cut = len(line) if cut < 0 else cut
    probe = 0
    while True:
        colon = line.find(":", probe)
        if colon < 0:
            break
        if colon + 1 >= len(line) or line[colon + 1] in " \t":
            cut = min(cut, colon)
            break
        probe = colon + 1
    return line[:cut].rstrip(" \t"), stop


def plain_spaces(text, index):
    """PyYAML ``scan_plain_spaces``: ``(chunk, next, column)``."""
    size, cursor = len(text), index
    while cursor < size and text[cursor] == " ":
        cursor += 1
    whitespaces = text[index:cursor]
    column = len(whitespaces)
    if cursor >= size or not is_break(text[cursor]):
        return whitespaces, cursor, column
    consumed, first = yaml_break(text, cursor, size)
    cursor += consumed
    column = 0
    breaks = []
    while cursor < size and (text[cursor] == " " or is_break(text[cursor])):
        if text[cursor] == " ":
            cursor += 1
            column += 1
            continue
        consumed, decoded = yaml_break(text, cursor, size)
        breaks.append(decoded)
        cursor += consumed
        column = 0
    if first != "\n":
        chunk = first + "".join(breaks)
    elif breaks:
        chunk = "".join(breaks)
    else:
        chunk = " "
    return chunk, cursor, column


__all__ = ["block_scalar", "indent_at", "line_end", "next_line",
           "plain_line_span", "plain_run_end", "plain_spaces"]
