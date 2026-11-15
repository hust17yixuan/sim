"""Structural source spans for JSON and optional-YAML launch documents.

[Role]
  Parse one structured launch document into a read-only node tree that keeps
  the exact one-based ``(line, column)`` and character ``(start, end)`` range
  of every object, key, array and scalar, plus a per-decoded-character
  ``char_coords`` map back to the outer document.  `launch_documents` selects a
  container by the decoded payload index, then reads that node's own
  ``command``/``args``/``env`` children, so attribution never depends on
  textual occurrence order or a global cursor.

[Invariants]
  * Scalars are decoded with ``json.loads`` on their own source slice; the
    parser itself never evaluates a shell word or a YAML tag.
  * Object member order is irrelevant; a selected node's own members are the
    only source of its field spans (a decoy elsewhere cannot win).
  * One shared ``Lines`` index is built per document; every scalar mapper
    receives it, so coordinate work is linear in document characters.
  * Every JSON/YAML string scalar carries ``char_coords`` covering decoded
    indices ``0..len(value)``; a decoder that cannot reproduce the loader's
    value raises `SpanUnavailable` instead of reporting a guessed position.
  * Shell words decoded out of a structured scalar keep unquoted
    ``char_coords`` aligned with their text; substitution leaves it empty.
  * Raw-document credential pre-scanning is conservative: over-redaction is
    safe, echoing a credential is not, and it never changes parse admission.
  * Nodes are immutable and bounded by the document text; no state is shared.

[Timeouts & Locks]
  Linear in document characters/nodes; no locks or I/O.
[Side-Effect Policy]
  No subprocess, environment lookup, file read or mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

from .launch_coords import (
    Lines, at, json_char_coords, line_starts, normalize_newlines, unquote_word,
    yaml_scalar_coords,
)
from .launch_parser import SourceSpan, Word
from .launch_taint import (
    STRUCTURAL_KEYS, duplicate_key_label, prescan_cli_secret_values,
    prescan_document_secrets, prescan_payload_secret_values, taintable_literal,
)

_Lines = Lines
#: Credential-name sensitivity is decided by the name, never the character
#: class: this compatibility predicate accepts digits, hex and short strings.
_plausible_secret = taintable_literal


class SpanUnavailable(ValueError):
    """The optional structured parser cannot produce trustworthy spans."""


@dataclass(frozen=True)
class SpanNode:
    """One immutable JSON/YAML node with its exact source range."""
    kind: str
    start: int
    end: int
    line: int
    column: int
    end_line: int
    end_column: int
    value: object = None
    items: tuple["SpanNode", ...] = ()
    members: tuple[tuple["SpanNode", "SpanNode"], ...] = ()
    char_coords: tuple[tuple[int, ...], ...] = ()

    def member(self, name):
        for key, value in self.members:
            if str(key.value) == str(name):
                return value
        return None

    def key_node(self, name):
        for key, _value in self.members:
            if str(key.value) == str(name):
                return key
        return None


def span_of(node, source):
    return SourceSpan(source, node.line, node.column, node.end_line,
                      node.end_column)


class _JsonParser:
    def __init__(self, text):
        self.text, self.index, self.lines = text, 0, Lines(text)

    def parse(self):
        node = self._value()
        self._space()
        if self.index != len(self.text):
            raise SpanUnavailable("json_trailing_content")
        return node

    def _space(self):
        while self.index < len(self.text) and self.text[self.index] in " \t\r\n":
            self.index += 1

    def _node(self, kind, start, value=None, items=(), members=(), char_coords=()):
        end = self.index
        line, column = self.lines.position(start)
        end_line, end_column = self.lines.position(end)
        if not self.lines.bounds(line, column, end_line, end_column):
            raise SpanUnavailable("json_node_out_of_bounds")
        return SpanNode(kind, start, end, line, column, end_line, end_column,
                        value, tuple(items), tuple(members), tuple(char_coords))

    def _value(self):
        self._space()
        if self.index >= len(self.text):
            raise SpanUnavailable("json_value_required")
        char = self.text[self.index]
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        if char == '"':
            return self._string()
        return self._literal()

    def _object(self):
        start = self.index
        self.index += 1
        members = []
        self._space()
        if self.text.startswith("}", self.index):
            self.index += 1
            return self._node("object", start, members=members)
        while True:
            self._space()
            if self.index >= len(self.text) or self.text[self.index] != '"':
                raise SpanUnavailable("json_object_key_required")
            key = self._string()
            self._space()
            if self.index >= len(self.text) or self.text[self.index] != ":":
                raise SpanUnavailable("json_colon_required")
            self.index += 1
            members.append((key, self._value()))
            self._space()
            if self.index >= len(self.text):
                raise SpanUnavailable("json_object_unterminated")
            if self.text[self.index] == "}":
                self.index += 1
                return self._node("object", start, members=members)
            if self.text[self.index] != ",":
                raise SpanUnavailable("json_comma_required")
            self.index += 1

    def _array(self):
        start = self.index
        self.index += 1
        items = []
        self._space()
        if self.text.startswith("]", self.index):
            self.index += 1
            return self._node("array", start, items=items)
        while True:
            items.append(self._value())
            self._space()
            if self.index >= len(self.text):
                raise SpanUnavailable("json_array_unterminated")
            if self.text[self.index] == "]":
                self.index += 1
                return self._node("array", start, items=items)
            if self.text[self.index] != ",":
                raise SpanUnavailable("json_comma_required")
            self.index += 1

    def _string(self):
        start = self.index
        self.index += 1
        while self.index < len(self.text):
            char = self.text[self.index]
            if char == "\\":
                self.index += 2
                continue
            self.index += 1
            if char == '"':
                break
        else:
            raise SpanUnavailable("json_string_unterminated")
        raw = self.text[start:self.index]
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise SpanUnavailable("json_string_invalid") from exc
        coords = json_char_coords(self.text, start, self.index, self.lines)
        if len(coords) != len(value) + 1:
            raise SpanUnavailable("json_scalar_coords_unavailable")
        return self._node("scalar", start, value=value, char_coords=coords)

    def _literal(self):
        start = self.index
        while (self.index < len(self.text)
               and self.text[self.index] not in " \t\r\n,]}"):
            self.index += 1
        raw = self.text[start:self.index]
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise SpanUnavailable("json_literal_invalid") from exc
        return self._node("scalar", start, value=value)


def parse_json(text):
    """Recursive-descent JSON node parser with exact source ranges."""
    return _JsonParser(text).parse()


def parse_yaml(text):
    """Compose the already-validated YAML document into span nodes."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SpanUnavailable("yaml_parser_unavailable") from exc
    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise SpanUnavailable("invalid_yaml") from exc
    if node is None:
        return SpanNode("scalar", 0, 0, 1, 1, 1, 1, None)
    lines = Lines(text)
    # YAML's loader counts CRLF as one break and CR, NEL, LS and PS as breaks;
    # the document index must use the same numbering or node marks and scalar
    # character maps disagree about which physical line a leaf is on.
    lines.configure(True)

    def check_bounds(line, column, end_line, end_column, label):
        if not lines.bounds(line, column, end_line, end_column):
            raise SpanUnavailable(f"{label}_out_of_bounds")

    def convert(item, parent_indent=0):
        start, end = item.start_mark.index, item.end_mark.index
        line, column = item.start_mark.line + 1, item.start_mark.column + 1
        end_line = item.end_mark.line + 1
        end_column = max(1, item.end_mark.column + 1)
        check_bounds(line, column, end_line, end_column, "yaml_node")
        if isinstance(item, yaml.MappingNode):
            members = tuple((convert(key, item.start_mark.column),
                             convert(value, item.start_mark.column))
                            for key, value in item.value)
            return SpanNode("object", start, end, line, column, end_line,
                            end_column, members=members)
        if isinstance(item, yaml.SequenceNode):
            items = tuple(convert(value, item.start_mark.column)
                          for value in item.value)
            return SpanNode("array", start, end, line, column, end_line,
                            end_column, items=items)
        value = getattr(item, "value", None)
        coords = ()
        if isinstance(value, str):
            coords = yaml_scalar_coords(text, item, value, lines, parent_indent)
            if len(coords) != len(value) + 1:
                raise SpanUnavailable(
                    f"yaml_scalar_coords_unavailable:{getattr(item, 'style', None)}")
            for entry in coords:
                check_bounds(entry[0], entry[1], entry[3], entry[4],
                             "yaml_char")
        return SpanNode("scalar", start, end, line, column, end_line,
                        end_column, value, char_coords=coords)

    return convert(node)


def document_root(text, format):
    if format == "json":
        return parse_json(text)
    if format == "yaml":
        return parse_yaml(text)
    raise SpanUnavailable(f"structured_format_unsupported:{format}")


def select_node(root, payload, index):
    """Selected container node, derived from the decoded payload's index."""
    if isinstance(payload, list):
        if root.kind == "array" and 0 <= index < len(root.items):
            return root.items[index]
        return root
    if isinstance(payload, Mapping) and "containers" in payload:
        containers = root.member("containers")
        if containers is not None and 0 <= index < len(containers.items):
            return containers.items[index]
    return root


def scalar_words(node, text, source):
    """One Word per scalar element, carrying that element's own node range."""
    if node is None or node.kind != "array":
        raise SpanUnavailable("structured_argv_array_required")
    words = []
    for child in node.items:
        if child.kind != "scalar" or not isinstance(child.value, str):
            raise SpanUnavailable("structured_argv_scalar_required")
        # ``raw_text`` is the decoded value so ``char_coords`` indexes it
        # directly; the node span still points at the raw source token.
        words.append(Word(child.value, span_of(child, source), True, 0, False,
                          (), child.value, True, child.char_coords))
    return tuple(words)


def map_shell_words(words, text, coords, source):
    """Compose shell tokens parsed in a decoded scalar to outer coordinates."""
    if not coords:
        return tuple(words)
    starts = line_starts(text)
    shifted = []
    for word in words:
        first = at(starts, word.span.line_start, word.span.column_start)
        last = at(starts, word.span.line_end, word.span.column_end)
        if first >= len(coords) or last >= len(coords):
            shifted.append(word)
            continue
        start_coord, end_coord = coords[first], coords[last]
        raw_start = word.offset
        raw_stop = raw_start + len(word.raw_text)
        raw_coords = coords[raw_start:raw_stop + 1]
        decoded, decoded_coords = unquote_word(
            word.raw_text, raw_coords, escaped_char=True)
        if decoded != word.text:
            decoded_coords = ()
        shifted.append(Word(
            word.text, SourceSpan(source, start_coord[0], start_coord[1],
                                  end_coord[3], end_coord[4]), word.quoted,
            word.offset, word.secret, word.segments, word.raw_text,
            word.json_escaped, decoded_coords))
    return tuple(shifted)


def _map_record_value_coords(record, coords):
    """Compose an assignment value's inner interval through the outer map.

    An inner ``CharCoord`` is half-open: ``[start_offset, end_offset)``.
    Composing only its start entry loses both a multi-character outer escape
    and the escaped character's full ``.end``.  Map the first inner character
    to the outer start and the last *contained* inner character to the outer
    end, preserving the whole consumed source interval.
    """
    if not record.char_coords:
        return ()
    mapped = []
    for entry in record.char_coords:
        start_offset, end_offset = entry[2], entry[5]
        if start_offset < 0 or start_offset >= len(coords) \
                or end_offset < start_offset or end_offset > len(coords):
            return ()
        if end_offset == start_offset:
            # The final map entry is a zero-width end boundary.
            mapped.append(coords[start_offset])
            continue
        first, last = coords[start_offset], coords[end_offset - 1]
        mapped.append((first[0], first[1], first[2],
                       last[3], last[4], last[5]))
    return tuple(mapped)


def map_shell_records(records, text, coords, source):
    """Shift inner-shell EnvAssignment spans into outer document coordinates."""
    if not coords:
        return tuple(records)
    starts = line_starts(text)
    mapped = []
    for record in records:
        first = at(starts, record.span.line_start, record.span.column_start)
        last = at(starts, record.span.line_end, record.span.column_end)
        if first >= len(coords) or last >= len(coords):
            mapped.append(record)
            continue
        start_coord, end_coord = coords[first], coords[last]
        mapped.append(type(record)(record.name, record.value, SourceSpan(
            source, start_coord[0], start_coord[1], end_coord[3], end_coord[4]),
            record.order, record.inline, record.exported, record.secret,
            _map_record_value_coords(record, coords),
            has_value=record.has_value))
    return tuple(mapped)


def mapping_key_spans(node, source):
    if node is None or node.kind != "object":
        return {}
    return {str(key.value): span_of(key, source) for key, _value in node.members}


def mapping_value_coords(node):
    """Decoded scalar value -> exact ``char_coords`` for one mapping node."""
    if node is None or node.kind != "object":
        return {}
    return {str(key.value): value.char_coords for key, value in node.members
            if value.kind == "scalar" and isinstance(value.value, str)
            and value.char_coords}


def map_scalar_words(node, words, text, source):
    """Compatibility wrapper: map words parsed in one structured scalar."""
    if (node is None or node.kind != "scalar"
            or not isinstance(node.value, str)):
        return tuple(words)
    value, coords = normalize_newlines(node.value, node.char_coords)
    return map_shell_words(words, value, coords, source)


def map_scalar_records(node, records, text, source):
    """Compatibility wrapper: map env records parsed in one scalar."""
    if (node is None or node.kind != "scalar"
            or not isinstance(node.value, str)):
        return tuple(records)
    value, coords = normalize_newlines(node.value, node.char_coords)
    return map_shell_records(records, value, coords, source)

__all__ = [
    "SpanNode", "SpanUnavailable", "STRUCTURAL_KEYS", "document_root",
    "duplicate_key_label", "map_scalar_records", "map_scalar_words",
    "map_shell_records", "map_shell_words", "mapping_key_spans",
    "mapping_value_coords",
    "normalize_newlines", "parse_json", "parse_yaml",
    "prescan_cli_secret_values", "prescan_document_secrets",
    "prescan_payload_secret_values", "scalar_words", "select_node", "span_of",
]
