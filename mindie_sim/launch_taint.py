"""Decoded credential taint for raw launch documents and typed redaction aids.

[Role]
  Collect literal credential values from JSON/YAML string tokens, shell
  assignments and CLI flags *before* syntax or duplicate-key rejection can
  build a diagnostic.  A decoded key/name decides sensitivity; the value's
  character class never does.

[Invariants]
  * JSON/YAML string tokens are decoded in one bounded pass; an escaped
    credential name (``API_\\u004bEY``) taints the decoded value and its raw
    unescaped source spelling.
  * Global discovery resolves static ``$NAME``/``${NAME}`` from the declaring
    mapping's completed literal ``env`` map, standalone/late command fields and
    statement-local inline assignments; host environment is never read. Docker and
    ``sh -c`` wrappers are unpacked without a shell.  Unsupported trailing
    shell syntax still yields the words around each delimiter.
  * A plain YAML 1.1 boolean under a secret name contributes the adapter's
    normalized ``true``/``false`` spelling as well as its raw spelling.
  * Any non-empty scalar bound to a credential name is tainted, including pure
    digits, short strings, hex and base64; structural delimiters are not.
  * Duplicate-key diagnostics never echo an untrusted key: a secret-named,
    tainted or non-schema key becomes ``[REDACTED]``; only the fixed launch
    schema allowlist may be echoed.
  * Redaction itself is typed: dictionaries/lists/string leaves are rewritten,
    integer/float/bool/None JSON primitives keep their type so JSON stays valid.
[Timeouts & Locks]
  Linear in document characters; optional YAML decoding is pure Python.
[Side-Effect Policy]
  Pure functions returning tuples/sets; never mutates an input document.
"""
from __future__ import annotations

import json
import math
import re

from . import launch_env
from .launch_coords import raw_word_coords, unquote_word
from .launch_env import EnvMap, ShellText, _ShellScope, _ShellWord, _tolerant_words, normalize_env
from .launch_parser import (
    LaunchSyntaxError, MAX_INTEGER_BITS, WordToken, _resolve, is_secret_name, scan,
)
from .launch_yaml_taint import yaml_document_taint

STRUCTURAL_KEYS = frozenset({
    "command", "args", "env", "env_file", "image", "entrypoint", "workdir",
    "cwd", "notes", "version", "services",
})
_COMMAND_PAIRS = (("command", "args"), ("entrypoint", "cmd"),
                  ("command", "cmd"), ("entrypoint", "args"))
_COMMAND_FIELDS = frozenset(key for pair in _COMMAND_PAIRS for key in pair)
_ASSIGNMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_QUOTED_VALUE = launch_env._QUOTED_VALUE
_RAW_FLAG_VALUE = launch_env._RAW_FLAG_VALUE
_SECRET_PAIR = re.compile(
    r"""(?:"|')?([A-Za-z_][A-Za-z0-9_.-]{0,127})(?:"|')?[ \t]*"""
    r"(?:=[ \t]*" + _RAW_FLAG_VALUE + r"|:[ \t]*(" + _QUOTED_VALUE +
    r"""|[^\s,;{}\[\]"':\\]+)(?=$|[\s,;{}\[\]:]))""")
_SECRET_FLAG = re.compile(
    r"""--([A-Za-z][A-Za-z0-9_-]{0,127})=""" + _RAW_FLAG_VALUE)
_FLAG_VALUE = re.compile(_RAW_FLAG_VALUE)


def _raw_credential(raw, found):
    """Retain a whole single-line spelling; unwrap only one enclosing pair."""
    if "\n" not in raw and "\r" not in raw and _FLAG_VALUE.fullmatch(raw):
        inner = raw[1:-1] if re.fullmatch(_QUOTED_VALUE, raw) else raw
        if taintable_literal(inner):
            found.add(inner)


def taintable_literal(value):
    """True for a non-empty scalar that is not structural punctuation."""
    text = str(value or "")
    return bool(text) and text not in {"|", ">", "{", "[", "]", "}"}


def duplicate_key_label(key, secrets=()):
    """A safe duplicate-key label; unknown/untrusted keys never echo."""
    text = str(key)
    if (is_secret_name(text) or text in {str(item) for item in secrets}
            or text not in STRUCTURAL_KEYS):
        return "[REDACTED]"
    return text


_JSON_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                 "n": "\n", "r": "\r", "t": "\t"}


def _hex4(raw, start):
    """Four hexadecimal digits at ``start`` or ``None`` without raising."""
    digits = raw[start:start + 4]
    if len(digits) != 4:
        return None
    try:
        return int(digits, 16)
    except ValueError:
        return None


def _manual_json_string(raw):
    """Total, allocation-bounded JSON string decoder for malformed input.

    Syntax admission still rejects the document; this decoder only exists so
    credential discovery can run *before* that rejection.  Unknown escapes are
    preserved literally and a truncated escape yields ``None`` instead of
    raising, so a malformed token can never abort the prescan.
    """
    try:
        return json.loads('"' + raw + '"')
    except ValueError:
        pass
    out, index, size = [], 0, len(raw)
    while index < size:
        char = raw[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        if index + 1 >= size:
            return None
        nxt = raw[index + 1]
        if nxt in _JSON_ESCAPES:
            out.append(_JSON_ESCAPES[nxt])
            index += 2
            continue
        if nxt == "u":
            code = _hex4(raw, index + 2)
            if code is None:
                # Preserve the malformed escape as literal text and keep
                # scanning: a later credential token must still be visible.
                out.append("u")
                index += 2
                continue
            if 0xD800 <= code <= 0xDBFF and index + 12 <= size \
                    and raw[index + 6:index + 8] == "\\u":
                low = _hex4(raw, index + 8)
                if low is not None and 0xDC00 <= low <= 0xDFFF:
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    index += 6
            out.append(chr(code))
            index += 6
            continue
        # A non-JSON escape is invalid but still carries its literal tail.
        out.append(nxt)
        index += 2
    return "".join(out)


def _json_tokens(text):
    tokens, index, size = [], 0, len(text)
    while index < size:
        char = text[index]
        if char == '"':
            start, index = index, index + 1
            while index < size:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == '"':
                    index += 1
                    break
                index += 1
            raw = text[start:index]
            if len(raw) >= 2 and raw.endswith('"'):
                inner = raw[1:-1]
                decoded = _manual_json_string(inner)
            else:
                inner, decoded = raw[1:], None
            if decoded is not None:
                cursor = index
                while cursor < size and text[cursor] in " \t\r\n":
                    cursor += 1
                is_key = cursor < size and text[cursor] == ":"
                tokens.append(("str", decoded, inner, is_key))
        elif char in "{}[],:":
            tokens.append((char, None, None, False))
            index += 1
        else:
            start = index
            while index < size and text[index] not in " \t\r\n{}[],:\"":
                index += 1
            if index == start:
                index += 1
                continue
            raw = text[start:index]
            tokens.append(("scalar", raw, raw, False))
    return tokens


def _number_spelling(raw):
    """Loader-normalized spelling for a finite JSON numeric env primitive."""
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            if value.bit_length() > MAX_INTEGER_BITS:
                return None
        elif not isinstance(value, float) or not math.isfinite(value):
            return None
        return str(value)
    except (ValueError, OverflowError, RecursionError):
        return None


def _compose_fields(fields, found, env=None):
    """Join sibling command/args token lists inside one mapping value."""
    for left, right in _COMMAND_PAIRS:
        first, second = fields.get(left), fields.get(right)
        if first and second:
            # Scalar command heads carry ShellText; argv tails stay delimited.
            _shell_taint(tuple((*first, *second)), found, env)


def _json_taint(text, found):
    stack = []

    def push(kind, secret, owner_key=None, is_env=False):
        stack.append({"kind": kind, "secret": bool(secret), "key_secret": False,
                      "expect_key": kind == "{", "key": None, "items": [],
                      "fields": {}, "pairs": {}, "dirty": False,
                      "owner_key": owner_key, "is_env": is_env,
                      "env": EnvMap(), "surfaces": [], "pending_name": None})

    def drain(frame):
        env = frame["env"] or None
        for surface in frame["surfaces"]:
            _shell_taint(surface, found, env if "$" in surface else None)
        if frame["kind"] == "[":
            if frame["items"]:
                _shell_taint(tuple(frame["items"]), found, env)
        elif frame["dirty"]:
            # Re-resolve every buffered command/args surface, standalone or
            # paired, against this mapping frame's completed literal env.
            for key in _COMMAND_FIELDS:
                tokens = frame["fields"].get(key)
                if tokens:
                    _shell_taint(tuple(tokens), found, env)
            _compose_fields(frame["fields"], found, env)
            frame["dirty"] = False

    def close_frame(frame):
        """Attach one frame to its owner, then finalize exactly once."""
        if frame["kind"] == "[":
            owner = frame.get("owner_key")
            parent = stack[-1] if stack else None
            if owner is not None and parent is not None \
                    and parent["kind"] == "{":
                parent["fields"].setdefault(owner, []).extend(frame["items"])
                if owner in _COMMAND_FIELDS:
                    parent["dirty"] = True
        elif frame["is_env"] and stack and stack[-1]["kind"] == "{":
            stack[-1]["env"].update(normalize_env(frame["pairs"]))
        drain(frame)

    def value_token(parent, decoded, raw):
        if parent is None:
            return
        if parent["secret"] or parent["key_secret"]:
            if taintable_literal(decoded):
                found.add(decoded)
            if raw and raw != decoded and taintable_literal(raw):
                found.add(raw)
        if parent["kind"] == "[":
            parent["items"].append(decoded)
        elif parent["kind"] == "{":
            key = parent.get("key")
            if key is not None:
                parent["fields"].setdefault(key, []).append(ShellText(decoded))
                parent["pairs"][key] = decoded
                if key in _COMMAND_FIELDS:
                    parent["dirty"] = True
            parent["expect_key"] = True
            parent["key"] = None

    for kind, decoded, raw, is_key in _json_tokens(text):
        if kind == "str":
            parent = stack[-1] if stack else None
            # Mapping scalars may be scripts; list elements retain boundaries.
            if parent is not None and parent["kind"] == "{":
                parent["surfaces"].append(decoded)
            if is_key:
                if parent is not None and parent["kind"] == "{":
                    parent["key_secret"] = bool(
                        parent["secret"] or is_secret_name(decoded))
                    parent["key"] = decoded
                    parent["expect_key"] = False
                    if parent["is_env"]:
                        parent["pending_name"] = decoded
            elif parent is not None:
                value_token(parent, decoded, raw)
                if parent["is_env"] and parent["pending_name"] is not None:
                    parent["pairs"][parent["pending_name"]] = decoded
                    parent["pending_name"] = None
        elif kind == "scalar":
            parent = stack[-1] if stack else None
            if parent is not None and (parent["secret"] or parent["key_secret"]):
                spelling = _number_spelling(decoded)
                text_value = spelling or (
                    decoded if decoded in {"true", "false"} else None)
                if text_value:
                    found.add(text_value)
                if parent["kind"] == "{":
                    key = parent.get("key")
                    if key is not None and spelling:
                        parent["fields"].setdefault(key, []).append(spelling)
                        if key in _COMMAND_FIELDS:
                            parent["dirty"] = True
            if parent is not None and parent["kind"] == "{" \
                    and parent["is_env"] and parent["pending_name"] is not None:
                normalized = _number_spelling(decoded)
                if normalized is None and decoded in {"true", "false"}:
                    normalized = decoded
                parent["pairs"][parent["pending_name"]] = (
                    normalized if normalized is not None else decoded)
                parent["pending_name"] = None
            if parent is not None and parent["kind"] == "{":
                parent["expect_key"] = True
                parent["key"] = None
        elif kind in "[{":
            parent = stack[-1] if stack else None
            inherited = bool(parent and (parent["secret"] or parent["key_secret"]))
            is_env = bool(parent is not None and parent["kind"] == "{"
                          and parent["key"] == "env"
                          and not parent["expect_key"])
            owner = parent.get("key") if parent is not None and parent["kind"] == "{" else None
            push(kind, inherited, owner, is_env)
        elif kind in "]}":
            if not stack:
                continue
            frame = stack.pop()
            close_frame(frame)
            if stack and stack[-1]["kind"] == "{":
                stack[-1]["expect_key"] = True
                stack[-1]["key"] = None
        elif kind == "," and stack and stack[-1]["kind"] == "{":
            stack[-1]["expect_key"] = True
            stack[-1]["key"] = None
    # Unexpected EOF: unwind bottom-up with the normal attach/finalize path so
    # an open child still reaches its owning mapping before composition.
    while stack:
        close_frame(stack.pop())


def _static_expansion(token, env):
    """Keep a known literal when admission alone requires splitting/globbing.

    Every variable must still be known and nonempty when unquoted. Joining
    scanned segments neither splits the value nor interprets its characters.
    """
    parts = []
    for segment in token.segments:
        if segment.kind == "literal":
            parts.append(segment.text)
            continue
        value = env.get(segment.text)
        if value is None or (not segment.quoted and not value[0]):
            return None
        parts.append(value[0])
    return "".join(parts)


def _decoded_shell_words(text, env=None):
    """Decode shell words, preserving identity-only statement boundaries."""
    from .launch_docker import _STATEMENT_END
    mapping = normalize_env(env)
    try:
        tokens = scan(text)
    except (LaunchSyntaxError, RecursionError, ValueError):
        return _tolerant_words(text, mapping)
    words, scope = [], _ShellScope(mapping)
    for token in tokens:
        if not isinstance(token, WordToken):
            words.append(_STATEMENT_END)
            scope.end_statement()
            continue
        raw = token.raw_text
        decoded = None
        if any(segment.kind == "var" for segment in token.segments):
            # The zero-copy overlay appears only for a real local assignment.
            try:
                decoded = _resolve(token, scope.env).text
            except LaunchSyntaxError as exc:
                if exc.message.startswith("word_splitting_or_globbing_required:"):
                    decoded = _static_expansion(token, scope.env)
            except (RecursionError, ValueError):
                decoded = None
        if decoded is None:
            decoded, _mapped = unquote_word(
                raw, raw_word_coords(raw, 1, 1, 0))
        word = _ShellWord(decoded, token.assignment_name, unquoted_ident=bool(
            _ASSIGNMENT_NAME.fullmatch(raw.replace("\\\n", ""))), raw_text=raw)
        words.append(word)
        scope.word(word, word.assignment_name, word.unquoted_ident)
    return tuple(words)


def _literal_option(word):
    """Distinguish literal CLI options from decoded credential operands.

    Quotes, escapes and expansion can produce a leading dash in a value.
    Only line continuations disappear without changing lexical identity.
    Literal long options and short-option groups cannot supply another flag's
    value; negative numbers and other single-dash data remain operands.
    """
    if not word.startswith("-"):
        return False
    raw = getattr(word, "raw_text", word)
    spelling = re.sub(r"\\(?:\r\n?|\n)", "", raw)
    return spelling == word and (word.startswith("--") or bool(
        re.fullmatch(r"-[A-Za-z][A-Za-z0-9]*(?:=.*)?", word)))


def _static_argv_operand(text, env, found=None):
    """Decode one delimited operand; whitespace/operators never split it.

    Unknown references or unfinished quotes/escapes have no static candidate.
    Known empty segments may join a nonempty composite. Values are appended
    once, never rescanned; only an entirely unquoted empty result is rejected.
    Publish contributing values only after the whole credential is known.
    """
    mapping = normalize_env(env)
    parts, resolved, index, quote, quoted = [], [], 0, None, False
    while index < len(text):
        char = text[index]
        if quote == "'":
            if char == quote:
                quote = None
            else:
                parts.append(char)
            index += 1
            continue
        if char in "'\"" and (quote is None or char == quote):
            quote = None if quote else char
            quoted = True
            index += 1
            continue
        if char == "\\":
            skipped = launch_env.continuation_length(text, index)
            if skipped:
                index += skipped
                continue
            if index + 1 == len(text):
                return None
            if quote is None or text[index + 1] in '$`"\\':
                parts.append(text[index + 1])
                index += 2
                continue
        if char == "$":
            kind, name, index = launch_env._read_variable(text, index)
            if kind == "invalid":
                return None
            if kind == "var":
                value = mapping.get(name)
                if value is None:
                    return None
                parts.append(value[0])
                resolved.append(value[0])
            else:
                parts.append("$")
            continue
        parts.append(char)
        index += 1
    value = "".join(parts)
    if quote is not None or (not value and not quoted):
        return None
    if found is not None and value:
        found.update(item for item in resolved if taintable_literal(item))
    return value


def _add_credential(value, found, env=None, name="--api-key"):
    """Discover named or JSON credentials in one complete eligible operand."""
    if not taintable_literal(value):
        return
    found.update(prescan_cli_secret_values(name, value))
    if env and "$" in value and not isinstance(value, _ShellWord):
        # Pairing and literal-option exclusions have already been checked.
        # Decode only this operand: no argv element can create new boundaries
        # or consume its neighbors. Keep its literal spelling for strict argv.
        decoded = _static_argv_operand(value, env, found if is_secret_name(name) else None)
        if decoded is not None and taintable_literal(decoded):
            found.update(prescan_cli_secret_values(name, decoded))


def _apply_words(words, found, *, raw_values=False, env=None):
    """Inspect complete assignment and option operands without joining argv."""
    # One reverse pass keeps flag/value pairing linear across redirect runs.
    next_values, following = [None] * len(words), None
    for index in range(len(words) - 1, -1, -1):
        next_values[index] = following
        if not getattr(words[index], "is_redirect", False):
            following = words[index]
    for index, word in enumerate(words):
        if not isinstance(word, str):
            continue
        value_env = None if isinstance(word, _ShellWord) else env
        name, equals, inline = word.partition("=")
        # Named inline credentials remain candidates on redirect targets;
        # ordinary targets cannot supply JSON option operands.
        if getattr(word, "is_redirect", False) and not (equals and is_secret_name(name)):
            continue
        assignment = equals and _ASSIGNMENT_NAME.fullmatch(name)
        if not assignment and not (name.startswith("--") or re.fullmatch(
                r"-[A-Za-z][A-Za-z0-9]*", name)):
            continue
        if equals:
            value, raw = inline, getattr(word, "raw_text", word).partition("=")[2]
        else:
            value = next_values[index]
            if not isinstance(value, str) or _literal_option(value):
                continue
            value_env = env
            raw = getattr(value, "raw_text", value)
        _add_credential(value, found, value_env, name)
        # Only an admitted credential binding retains a raw shell spelling;
        # ordinary JSON options publish only their sensitive descendants.
        if raw_values and is_secret_name(name) and isinstance(
                word if equals else value, _ShellWord):
            _raw_credential(raw, found)


def _wrapper_taint(words, found, env=None, depth=0):
    """Scan each admitted statement's wrapper with its own inline scope."""
    if depth >= 3:
        return
    # Imported lazily: launch_docker imports launch_spans, which imports this
    # discovery module; the deferred name avoids that import cycle.
    from .launch_docker import iter_discovery_wrappers
    for script, inner_env in iter_discovery_wrappers(words, env):
        _shell_taint(script, found, inner_env, _depth=depth + 1)


def _apply_sequence(words, found, env=None, depth=0):
    """Apply extraction to one argv surface, tokenizing a command string."""
    if words and isinstance(words[0], ShellText):
        # A structured string command is stored as one field token; split it
        # so a trailing flag can pair with the first literal args element.
        # Literal args append to the selected command's argv, including when
        # YAML block chomping leaves trailing line breaks in the command.
        head = words[0].rstrip(" \t\r\n")
        words = (*_decoded_shell_words(head, env if "$" in head else None),
                 *words[1:])
    _apply_words(words, found, raw_values=True, env=normalize_env(env))
    _wrapper_taint(words, found, env, depth)


def _shell_taint(text, found, env=None, _depth=0):
    if isinstance(text, (tuple, list)):
        # One already-decoded argv token per element: keep each element's own
        # spaces instead of re-tokenizing the joined spelling.
        _apply_sequence(tuple(item if isinstance(item, str) else str(item)
                              for item in text), found, env, _depth)
        return
    if text.lstrip().startswith(("{", "[")):
        found.update(prescan_cli_secret_values("", text))
    # Conservative raw spellings remain visible even when the launch grammar
    # rejects the surrounding text (malformed shell, unsupported operators).
    for match in _SECRET_PAIR.finditer(text):
        if not is_secret_name(match.group(1)):
            continue
        raw = next((item for item in match.groups()[1:] if item is not None), "")
        _raw_credential(raw, found)
        if match.group(3) is not None and re.fullmatch(_QUOTED_VALUE, raw):
            try:
                decoded = json.loads(raw)
            except ValueError:
                decoded = None
            if decoded and taintable_literal(decoded):
                found.add(decoded)
    # Raw flags contribute inline values only. Separate values, including
    # malformed-shell fallbacks and both YAML walkers, share _apply_words.
    for match in _SECRET_FLAG.finditer(text):
        if not is_secret_name(match.group(1)):
            continue
        _raw_credential(match.group(2), found)
    # Both decoders preserve statement boundaries and explicit env resolution;
    # unsupported operators do not discard the surrounding credential words.
    words = _decoded_shell_words(text, env if "$" in text else None)
    _apply_words(words, found, raw_values=True)
    # An unmatched quote must not hide a literal credential later in its body.
    # Recover only that suffix, without an enclosing env or redirect target.
    # There are two quote kinds: each unmatched body excludes its own closing
    # delimiter, so two bounded passes suffice and work remains linear.
    recovery = words
    for _ in range(2):
        last = recovery[-1] if recovery else None
        body = getattr(last, "unclosed_text", None)
        if body is None or getattr(last, "is_redirect", False):
            break
        recovery = _tolerant_words(body, None)
        _apply_words(recovery, found, raw_values=True)
    _wrapper_taint(words, found, env, _depth)


def prescan_document_secrets(text, format="shell"):
    """Credential values `raw` already spells, decoded before any rejection."""
    if not isinstance(text, str) or not text:
        return ()
    found = set()
    if format == "json" or (format == "shell"
                            and text.lstrip()[:1] in {"{", "["}):
        _json_taint(text, found)
    elif format == "yaml":
        yaml_document_taint(text, found, shell_scan=_shell_taint)
    else:
        _shell_taint(text, found)
    return tuple(sorted(item for item in found if taintable_literal(item)))


def prescan_payload_secret_values(payload):
    """Credential values under a secret-named JSON/YAML key, at any depth."""
    found = set()

    def walk(value, secret):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, secret or is_secret_name(key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, secret)
        elif secret and isinstance(value, str):
            if taintable_literal(value):
                found.add(value)
        elif secret and isinstance(value, bool):
            # The structured YAML/JSON adapter renders booleans as lowercase
            # strings, so that is the credential value the env map receives.
            found.add("true" if value else "false")
        elif secret and isinstance(value, (int, float)):
            # Integer limits and float finiteness precede string conversion.
            spelled = _number_spelling(value)
            if spelled:
                found.add(spelled)

    try:
        walk(payload, False)
    except (RecursionError, ValueError, OverflowError):
        pass  # Advisory traversal retains every value already discovered.
    return tuple(sorted(found))


def prescan_cli_secret_values(name, raw):
    """Literal credential CLI values and JSON-flag credentials for one flag."""
    if not isinstance(raw, str) or not raw:
        return ()
    found = set()
    if is_secret_name(name) and taintable_literal(raw):
        found.add(raw)
    if raw.lstrip().startswith(("{", "[")):
        _json_taint(raw, found)
        try:
            payload = json.loads(raw)
            found.update(prescan_payload_secret_values(payload))
        except (ValueError, RecursionError, OverflowError):
            pass  # Typed discovery cannot invalidate the lexical inventory.
    return tuple(sorted(found))


__all__ = [
    "STRUCTURAL_KEYS", "duplicate_key_label", "prescan_cli_secret_values",
    "prescan_document_secrets", "prescan_payload_secret_values",
    "taintable_literal",
]
