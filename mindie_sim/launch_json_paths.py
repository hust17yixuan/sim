"""Structural JSON token/path discovery for exact leaf provenance.

[Role]
  Record every object member of a decoded JSON string at its exact JSON Pointer
  path with the source-token indices of its key and value.  `launch_actions`
  binds a JSON leaf to the member at the leaf's own path, never to the first
  equal key name found anywhere in the text.

[Invariants]
  * One forward cursor over the decoded text; every token is visited once, so
    discovery is O(N) in the text and allocates only per-member records.
  * JSON Pointer escaping is ``~0``/``~1`` and shared with path construction,
    so an unusual key cannot alias a different member.
  * Duplicate keys are retained in occurrence order; no key is looked up by
    string search and no source width is estimated.
  * Malformed input returns the members discovered so far; callers already
    validate the document and fail closed when a path is absent.
[Timeouts & Locks]
  Bounded in-memory scan; no locks, I/O or devices.
[Side-Effect Policy]
  Pure function over a string; no mutation of its input or global state.
"""
from __future__ import annotations

import json


def json_pointer_join(pointer, token):
    """Append one decoded object key to a JSON Pointer."""
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{pointer}/{escaped}" if pointer else f"/{escaped}"


def _space(text, index):
    size = len(text)
    while index < size and text[index] in " \t\r\n":
        index += 1
    return index


def _string_end(text, index):
    size = len(text)
    cursor = index + 1
    while cursor < size:
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == '"':
            return cursor + 1
        cursor += 1
    return None


def _scalar_end(text, index):
    size = len(text)
    cursor = index
    while cursor < size and text[cursor] not in " \t\r\n,]}":
        cursor += 1
    return cursor


def json_member_paths(text):
    """Map JSON Pointer -> occurrence-ordered member token index records."""
    members = {}
    if not isinstance(text, str) or not text:
        return members
    size = len(text)

    def value(index, pointer):
        index = _space(text, index)
        if index >= size:
            return size
        char = text[index]
        if char == "{":
            return object_(index, pointer)
        if char == "[":
            return array(index, pointer)
        if char == '"':
            return _string_end(text, index) or size
        return _scalar_end(text, index)

    def object_(index, pointer):
        cursor = index + 1
        while True:
            cursor = _space(text, cursor)
            if cursor >= size:
                return size
            if text[cursor] == "}":
                return cursor + 1
            if text[cursor] != '"':
                return cursor
            key_start = cursor
            key_end = _string_end(text, key_start)
            if key_end is None:
                return size
            try:
                key = json.loads(text[key_start:key_end])
            except ValueError:
                return key_end
            child = json_pointer_join(pointer, key)
            cursor = _space(text, key_end)
            if cursor >= size or text[cursor] != ":":
                return cursor
            cursor = _space(text, cursor + 1)
            value_start = cursor
            cursor = value(cursor, child)
            members.setdefault(child, []).append(
                (key_start, key_end, value_start, cursor))
            cursor = _space(text, cursor)
            if cursor >= size:
                return size
            if text[cursor] == ",":
                cursor += 1
                continue
            if text[cursor] == "}":
                return cursor + 1
            return cursor

    def array(index, pointer):
        cursor = index + 1
        item = 0
        while True:
            cursor = _space(text, cursor)
            if cursor >= size:
                return size
            if text[cursor] == "]":
                return cursor + 1
            cursor = value(cursor, json_pointer_join(pointer, str(item)))
            item += 1
            cursor = _space(text, cursor)
            if cursor >= size:
                return size
            if text[cursor] == ",":
                cursor += 1
                continue
            if text[cursor] == "]":
                return cursor + 1
            return cursor

    try:
        value(0, "")
    except (RecursionError, ValueError, IndexError):
        return members
    return members


__all__ = ["json_member_paths", "json_pointer_join"]
