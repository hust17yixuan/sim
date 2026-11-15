"""Exact YAML scalar coordinate and escape decoding for launch documents.

[Role]
  Decode PyYAML scalar styles (single/double quoted, literal/folded block and
  plain) into one source coordinate per decoded character, and expose the same
  escape decoding to the raw credential pre-scan.  `launch_coords` composes
  these maps with the shared document line index.

[Invariants]
  * A decoder reproduces the loader's value exactly or returns ``()``; no
    position is guessed.  Explicit block indentation, folding of blank and
    more-indented lines, escaped line continuations and chomping all follow
    PyYAML's own semantics.
  * YAML line breaks are consumed by `yaml_break`: CRLF, CR, LF and U+0085
    normalize to one LF; U+2028/U+2029 stay themselves, exactly as
    ``Scanner.scan_line_break`` does.
  * Flow folding maps one LF break to a space and every following empty line to
    one newline; a non-LF break is emitted itself.  A backslash-escaped first
    break disappears, while each immediately following blank line emits one
    newline.
  * Returned maps have ``len(value) + 1`` entries.
[Timeouts & Locks]
  Linear in the scalar source characters with maximal whitespace/break runs;
  no locks, I/O or devices.
[Side-Effect Policy]
  Pure functions over strings; no subprocess or environment lookup.
"""
from __future__ import annotations

_YAML_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                 "n": "\n", "r": "\r", "t": "\t", "0": "\0", "a": "\a",
                 "v": "\v", "e": "\x1b", " ": " ", "_": "\xa0", "N": "\x85",
                 "L": "\u2028", "P": "\u2029"}
_BREAKS = frozenset("\r\n\x85\u2028\u2029")
_SPACES = " \t"


def is_break(char):
    """True for every character PyYAML treats as a YAML line break."""
    return char in _BREAKS


def yaml_break(text, index, stop=None):
    """``(consumed, decoded)`` for one YAML line break at ``index``."""
    limit = len(text) if stop is None else stop
    if index >= limit:
        return 0, ""
    char = text[index]
    if char == "\r":
        if index + 1 < limit and text[index + 1] == "\n":
            return 2, "\n"
        return 1, "\n"
    if char == "\n" or char == "\x85":
        return 1, "\n"
    if char == "\u2028" or char == "\u2029":
        return 1, char
    return 0, ""


def yaml_lines(text, start, end):
    """``(line_start, content_end, break_start, break_end, break_char)`` rows."""
    segments, pos = [], start
    while pos < end:
        cursor = pos
        while cursor < end and text[cursor] not in _BREAKS:
            cursor += 1
        if cursor >= end:
            segments.append((pos, end, None, None, ""))
            break
        consumed, decoded = yaml_break(text, cursor, end)
        segments.append((pos, cursor, cursor, cursor + consumed, decoded))
        pos = cursor + consumed
    return segments


def _fold_chars(breaks):
    """PyYAML flow folding for one maximal break run."""
    first = breaks[0]
    if first[2] != "\n":
        out = [(first[2], first[0], first[1])]
        out.extend((item[2], item[0], item[1]) for item in breaks[1:])
        return out
    if len(breaks) == 1:
        return [(" ", first[0], first[1])]
    return [(item[2], item[0], item[1]) for item in breaks[1:]]


def _header_shape(rest):
    """``(indent_digit, chomping)`` from a block scalar header tail."""
    body = rest.split("#", 1)[0].strip()
    digit, chomp = 0, ""
    for char in body:
        if char in "-+":
            chomp = char
        elif char.isdigit():
            digit = int(char)
        elif char:
            return 0, ""
    return digit, chomp


def _block_row(text, position, size):
    """One physical line with its leading spaces, content and break."""
    stop = position
    while stop < size and not is_break(text[stop]):
        stop += 1
    line = text[position:stop]
    width = len(line) - len(line.lstrip(" "))
    if stop < size:
        consumed, brk = yaml_break(text, stop, size)
    else:
        consumed, brk = 0, ""
    return {"start": position, "end": stop, "line": line, "width": width,
            "rest": line[width:], "brk": brk,
            "brk_start": stop if consumed else None,
            "brk_end": stop + consumed if consumed else None}


def block_layout(text, start, limit, parent):
    """PyYAML-exact block scalar layout after the ``|``/``>`` header.

    Returns ``(indent, folded, chomp, rows, leading, end)``.  ``indent`` is the
    loader content indentation, ``rows`` are only the physical lines the
    loader consumes, ``leading`` counts rows consumed by the implicit
    indentation probe and ``end`` is the offset of the first non-block line.
    """
    size = min(len(text), limit)
    header_end = start
    while header_end < size and not is_break(text[header_end]):
        header_end += 1
    digit, chomp = _header_shape(text[start + 1:header_end])
    folded = text[start:start + 1] == ">"
    consumed, _brk = yaml_break(text, header_end, size)
    position = header_end + consumed
    min_indent = parent + 1
    rows, leading_width, leading_count = [], 0, 0
    end = position
    if digit:
        indent = min_indent + digit - 1
    else:
        first = None
        while position < size:
            row = _block_row(text, position, size)
            if row["rest"]:
                first = row
                leading_width = max(leading_width, row["width"])
                break
            rows.append(row)
            leading_count += 1
            leading_width = max(leading_width, row["width"])
            end = row["brk_end"] if row["brk_end"] is not None else size
            if row["brk_end"] is None:
                position = size
                break
            position = row["brk_end"]
        indent = max(min_indent, leading_width)
        if first is not None:
            if first["width"] >= indent:
                rows.append(first)
                leading_count += 1
                if first["brk_end"] is None:
                    position = size
                else:
                    position = first["brk_end"]
                end = position
            else:
                end = first["start"]
    while position < size:
        row = _block_row(text, position, size)
        if row["rest"] and row["width"] < indent:
            break
        rows.append(row)
        if row["brk_end"] is None:
            position = size
        else:
            position = row["brk_end"]
        end = position
    return indent, folded, chomp, rows, leading_count, end


def _block_content(row, indent, index, leading_count, digit):
    """``(is_content, body)`` for one row under PyYAML's block rules.

    A whitespace-only line at or below the content indent is consumed as an
    empty line by ``scan_block_scalar_breaks``; only spaces strictly beyond
    the indent make it a more-indented content line.
    """
    if row["rest"] == "":
        if not digit and index < leading_count:
            return False, ""
        if row["width"] <= indent:
            return False, ""
        return True, row["line"][indent:]
    return True, row["line"][indent:]


def block_stream(text, start, limit, parent):
    """PyYAML's pre-chomp block chunks with exact source intervals.

    Returns ``(chunks, tail, trailing, chomp)``: ``chunks`` is every emitted
    character in loader order, ``tail`` is the final content line's break and
    ``trailing`` the breaks of blank lines after it.  Each emission is
    ``(decoded_char, source_start, source_end)``.
    """
    indent, folded, chomp, rows, leading, _end = block_layout(
        text, start, limit, parent)
    digit = _header_shape(text[start + 1:_line_stop(text, start, limit)])[0]
    marked, content_count = [], 0
    for index, row in enumerate(rows):
        content, body = _block_content(row, indent, index, leading, digit)
        if content:
            content_count += 1
        marked.append((content, row, body))
    chunks, pending = [], []
    index = 0
    while index < len(marked) and not marked[index][0]:
        row = marked[index][1]
        if row["brk"]:
            pending.append((row["brk"], row["brk_start"], row["brk_end"]))
        index += 1
    tail = None
    while index < len(marked):
        _content, row, body = marked[index]
        chunks.extend(pending)
        pending = []
        base = row["start"] + indent
        chunks.extend((char, base + step, base + step + 1)
                      for step, char in enumerate(body))
        following = index + 1
        while following < len(marked) and not marked[following][0]:
            blank = marked[following][1]
            if blank["brk"]:
                pending.append(
                    (blank["brk"], blank["brk_start"], blank["brk_end"]))
            following += 1
        if following < len(marked):
            nxt = marked[following][2]
            if (folded and row["brk"] == "\n"
                    and body[:1] not in (" ", "\t")
                    and nxt[:1] not in (" ", "\t")):
                if not pending:
                    chunks.append((" ", row["brk_start"], row["brk_end"]))
            elif row["brk"]:
                chunks.append((row["brk"], row["brk_start"], row["brk_end"]))
        else:
            tail = ((row["brk"], row["brk_start"], row["brk_end"])
                    if row["brk"] else None)
        index = following
    if content_count:
        trailing = pending
    else:
        trailing, chunks, tail = pending, [], None
    return chunks, tail, trailing, chomp


def block_value(chunks, tail, trailing, chomp):
    """Apply YAML strip/clip/keep chomping to the exact chunk stream."""
    chosen = list(chunks)
    if chomp != "-" and tail is not None:
        chosen.append(tail)
    if chomp == "+":
        chosen.extend(trailing)
    return "".join(item[0] for item in chosen), chosen


def _line_stop(text, start, limit):
    stop, size = start, min(len(text), limit)
    while stop < size and not is_break(text[stop]):
        stop += 1
    return stop


def yaml_block_coords(text, start, end, value, folded, lines, parent_indent=None):
    """Literal/folded block scalar coordinates with exact chomping."""
    if folded != (text[start:start + 1] == ">"):
        return ()
    header_line = lines.position(start)[0]
    header_start = lines.starts[header_line - 1]
    prefix = text[header_start:start]
    parent = (len(prefix) - len(prefix.lstrip(" ")) if parent_indent is None
              else parent_indent)
    chunks, tail, trailing, chomp = block_stream(text, start, end, parent)
    generated, chosen = block_value(chunks, tail, trailing, chomp)
    if generated != value:
        return ()
    coords = []
    for _char, source_start, source_end in chosen:
        line, column = lines.position(source_start)
        end_line, end_column = lines.position(source_end)
        coords.append((line, column, source_start,
                       end_line, end_column, source_end))
    end_line, end_column = lines.position(end)
    coords.append((end_line, end_column, end, end_line, end_column, end))
    return tuple(coords)


def _folds(text, index, stop):
    """Consume one YAML flow whitespace run; return (breaks, next_index).

    The maximal run is scanned exactly once.  ``breaks is None`` means the run
    held only horizontal whitespace and the caller must emit every character.
    Each break is ``(start, end, decoded)``.
    """
    cursor = index
    while cursor < stop and text[cursor] in _SPACES:
        cursor += 1
    if cursor >= stop or text[cursor] not in _BREAKS:
        return None, cursor
    breaks = []
    while True:
        consumed, decoded = yaml_break(text, cursor, stop)
        if not consumed:
            break
        breaks.append((cursor, cursor + consumed, decoded))
        cursor += consumed
        while cursor < stop and text[cursor] in _SPACES:
            cursor += 1
    return breaks, cursor


def _flow_coords(text, start, stop, style, value, lines):
    chars, coords = [], []

    def emit(char, offset, end=None):
        end = offset + 1 if end is None else end
        line, column = lines.position(offset)
        end_line, end_column = lines.position(end)
        chars.append(char)
        coords.append((line, column, offset,
                       end_line, end_column, end))

    index = start
    while index < stop:
        char = text[index]
        if style == "'" and char == "'" and index + 1 < stop and text[index + 1] == "'":
            emit("'", index + 1, index + 2)
            index += 2
            continue
        if style == '"' and char == "\\":
            cursor = index + 1
            while cursor < stop and text[cursor] in _SPACES:
                cursor += 1
            consumed, _decoded = yaml_break(text, cursor, stop)
            if consumed:
                # The escaped first break disappears; every immediately
                # following blank line emits one newline.
                cursor += consumed
                while cursor < stop and text[cursor] in _SPACES:
                    cursor += 1
                while True:
                    consumed, decoded = yaml_break(text, cursor, stop)
                    if not consumed:
                        break
                    emit(decoded, cursor, cursor + consumed)
                    cursor += consumed
                    while cursor < stop and text[cursor] in _SPACES:
                        cursor += 1
                index = cursor
                continue
            if index + 1 >= stop:
                return ()
            nxt = text[index + 1]
            width = {"x": 4, "u": 6, "U": 10}.get(nxt)
            if width and index + width <= stop:
                try:
                    code = int(text[index + 2:index + width], 16)
                except ValueError:
                    return ()
                emit(chr(code), index, index + width)
                index += width
                continue
            if nxt not in _YAML_ESCAPES:
                return ()
            emit(_YAML_ESCAPES[nxt], index + 1, index + 2)
            index += 2
            continue
        if char in _SPACES or char in _BREAKS:
            breaks, cursor = _folds(text, index, stop)
            if breaks is None:
                # A horizontal-only run is emitted in full and consumed once;
                # never re-scan the remaining run from index + 1.
                for offset in range(index, cursor):
                    emit(text[offset], offset)
            else:
                for item, first, last in _fold_chars(breaks):
                    emit(item, first, last)
            index = cursor
            continue
        emit(char, index, index + 1)
        index += 1
    if len(chars) != len(value) or "".join(chars) != value:
        return ()
    line, column = lines.position(stop)
    coords.append((line, column, stop, line, column, stop))
    return tuple(coords)


def yaml_scalar_coords(text, item, value, lines, parent_indent=None):
    """Exact decoded coordinates for one PyYAML string scalar."""
    start, end = item.start_mark.index, item.end_mark.index
    style = getattr(item, "style", None)
    if style == "|":
        return yaml_block_coords(text, start, end, value, False, lines,
                                parent_indent)
    if style == ">":
        return yaml_block_coords(text, start, end, value, True, lines,
                                parent_indent)
    if style == "'":
        return _flow_coords(text, start + 1, end - 1, "'", value, lines)
    if style == '"':
        return _flow_coords(text, start + 1, end - 1, '"', value, lines)
    return _flow_coords(text, start, end, None, value, lines)


def yaml_double_value(raw):
    """Decode one YAML double-quoted string token for credential prescanning.

    Handles flow folding too, so a multiline credential value under an
    inherited secret scope yields the loader's joined string rather than the
    physical source spelling.
    """
    chars, index, size = [], 0, len(raw)
    while index < size:
        char = raw[index]
        if char == "\\":
            cursor = index + 1
            while cursor < size and raw[cursor] in _SPACES:
                cursor += 1
            consumed, _decoded = yaml_break(raw, cursor, size)
            if consumed:
                # An escaped line break and its indentation disappear; each
                # following blank line contributes one newline.
                cursor += consumed
                while cursor < size and raw[cursor] in _SPACES:
                    cursor += 1
                while True:
                    consumed, decoded = yaml_break(raw, cursor, size)
                    if not consumed:
                        break
                    chars.append(decoded)
                    cursor += consumed
                    while cursor < size and raw[cursor] in _SPACES:
                        cursor += 1
                index = cursor
                continue
            if index + 1 >= size:
                return ""
            nxt = raw[index + 1]
            width = {"x": 4, "u": 6, "U": 10}.get(nxt)
            if width and index + width <= size:
                try:
                    chars.append(chr(int(raw[index + 2:index + width], 16)))
                except ValueError:
                    return ""
                index += width
                continue
            if nxt not in _YAML_ESCAPES:
                return ""
            chars.append(_YAML_ESCAPES[nxt])
            index += 2
            continue
        if char in _SPACES or char in _BREAKS:
            breaks, cursor = _folds(raw, index, size)
            if breaks is None:
                # Consume the whole horizontal run at once; emitting one
                # character here would rescan the remaining n - 1 spaces.
                for offset in range(index, cursor):
                    chars.append(raw[offset])
            else:
                chars.extend(item for item, _first, _last in _fold_chars(breaks))
            index = cursor
            continue
        chars.append(char)
        index += 1
    return "".join(chars)


def yaml_single_value(raw):
    """Decode one YAML single-quoted scalar, including flow folding."""
    out, index, size = [], 0, len(raw)
    while index < size:
        char = raw[index]
        if char == "'" and index + 1 < size and raw[index + 1] == "'":
            out.append("'")
            index += 2
            continue
        if char in _SPACES or char in _BREAKS:
            breaks, cursor = _folds(raw, index, size)
            if breaks is None:
                for offset in range(index, cursor):
                    out.append(raw[offset])
            else:
                out.extend(item for item, _first, _last in _fold_chars(breaks))
            index = cursor
            continue
        out.append(char)
        index += 1
    return "".join(out)

__all__ = [
    "is_break", "yaml_block_coords", "yaml_break", "yaml_double_value",
    "yaml_lines", "yaml_scalar_coords", "yaml_single_value",
]
