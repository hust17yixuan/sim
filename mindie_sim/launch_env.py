"""Shared launch value types, secret-name policy and bounded pre-scans.

[Role]
  Hold the occurrence types and conservative credential helpers shared by the
  shell parser, document adapters and span composers.  Keeping them here lets
  `launch_parser` stay a pure grammar module and keeps every importer using one
  `SourceSpan`/`EnvAssignment` definition.

[Invariants]
  * A secret name is decided by the decoded setting name only, never by the
    value's character class.
  * The lexical assignment pre-scan is total and conservative: it may over-
    taint a malformed document but never drops an already-declared credential.
  * ``EnvMap`` holds an already-normalized ``name -> (value, secret)`` view;
    document frames build it once per env mutation instead of copying a raw
    mapping for every scalar surface.
  * ``_EnvOverlay`` is a zero-copy view over a normalized base plus one
    surface's literal assignments: lookups check the overrides first and fall
    back to the base in O(1), never enumerating or copying the base.
  * Diagnostics carrying secrets are scrubbed in place before unwinding.
[Timeouts & Locks]
  Linear in the input text; no locks, I/O or environment lookup.
[Side-Effect Policy]
  Pure functions and frozen value types; no subprocess or global mutation.
"""
from __future__ import annotations


class ShellText(str):
    """A scalar command field, distinct from a delimited argv element."""

from collections.abc import Mapping
from dataclasses import dataclass
import re

_ASSIGNMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_QUOTED_VALUE = r"""(?:"(?:\\[\s\S]|[^"\\])*"|'[^']*')"""
# A word may concatenate quoted, escaped and bare segments. Only a shell
# delimiter can end it; JSON member recovery uses its own delimiter rules.
_RAW_FLAG_VALUE = (r"((?:" + _QUOTED_VALUE +
                   r"""|\\(?:\r\n|[\s\S])|[^ \t\r\n"'\\;|&<>()`!])+)(?=$|[ \t\r\n;|&<>()`!])""")
_SECRET_ASSIGNMENT = re.compile(
    r"(?:^|[; \t\r\n])[ \t]*(?:export[ \t]+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*" + _RAW_FLAG_VALUE, re.M)
_SECRET_WORDS = frozenset({"token", "secret", "password", "credential",
                           "authorization", "bearer"})


@dataclass(frozen=True)
class SourceSpan:
    source: str
    line_start: int
    column_start: int
    line_end: int
    column_end: int


@dataclass(frozen=True)
class EnvAssignment:
    """One literal assignment occurrence with its exact source position."""
    name: str
    value: str
    span: SourceSpan
    order: int
    inline: bool = False
    exported: bool = False
    secret: bool = False
    char_coords: tuple[tuple[int, ...], ...] = ()  # exact value source map
    #: False for a visibility-only ``export NAME``: it changes the export
    #: attribute but creates no new value origin and must not clear an earlier
    #: document/literal provenance map.
    has_value: bool = True


def is_secret_name(name) -> bool:
    """True when a setting/environment name denotes credential material."""
    text = str(name).lower().replace("-", "_")
    if "api_key" in text or "apikey" in text or "access_key" in text:
        return True
    return any(word in _SECRET_WORDS for word in re.split(r"[^a-z0-9]+", text))


class EnvMap(dict):
    """Already-normalized literal env: ``name -> (value, secret)``.

    Document frames build one instance per completed ``env`` mapping and pass
    it to every buffered surface, so a surface scan never re-enumerates or
    copies the frame's whole environment.  A raw mapping passed by an external
    caller is normalized once by ``normalize_env``.
    """
    __slots__ = ()


class _EnvOverlay(Mapping):
    """Zero-copy command-local view over an already-normalized base map.

    ``overrides`` holds only the few literal assignments declared on one shell
    surface (or prefix).  A lookup checks the overrides first and falls back to
    the base in O(1); constructing the view never copies or enumerates the
    base.  ``assign`` grows the override map one assignment at a time, so the
    total work stays proportional to the assignments, not to the base size.
    """

    __slots__ = ("_base", "_overrides")

    def __init__(self, base=None, overrides=None):
        self._base = base if base is not None else {}
        self._overrides = {} if overrides is None else overrides

    def __getitem__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return self._base[name]

    def __contains__(self, name):
        return name in self._overrides or name in self._base

    def __iter__(self):
        seen = set()
        for name in self._overrides:
            seen.add(name)
            yield name
        for name in self._base:
            if name not in seen:
                yield name

    def __len__(self):
        # ``len(base)`` is O(1) for the normalized dict/overlay base; only the
        # small override map is inspected here, never the base entries.
        return len(self._base) + sum(
            1 for name in self._overrides if name not in self._base)

    def __bool__(self):
        return bool(self._overrides) or bool(self._base)

    def assign(self, name, value):
        """Install one command-local value in the overrides only."""
        self._overrides[name] = value
        return self


def normalize_env(env):
    """Return ``env`` as an ``EnvMap`` without copying a normalized view."""
    if isinstance(env, (EnvMap, _EnvOverlay)):
        return env
    if not env:
        return EnvMap()
    return EnvMap((str(name), (str(value), is_secret_name(str(name))))
                  for name, value in dict(env).items())


class _ShellWord(str):
    """Decoded string retaining lexical prefix and redirection eligibility."""

    __slots__ = ("assignment_name", "is_redirect", "unquoted_ident", "raw_text",
                 "unclosed_text")

    def __new__(cls, value, assignment_name=None, is_redirect=False,
                unquoted_ident=False, raw_text=None):
        word = super().__new__(cls, value)
        word.assignment_name = (assignment_name if assignment_name is not None
                                and _ASSIGNMENT_NAME.fullmatch(assignment_name)
                                else None)
        word.is_redirect = is_redirect
        word.unquoted_ident = unquoted_ident
        word.raw_text = value if raw_text is None else raw_text
        word.unclosed_text = None
        return word


class _ShellScope:
    """Decode sequential prefixes without exporting executable-local values.

    Persistent shell assignments remain available for outer word expansion.
    Only prefix/export assignments affect expansion; command arguments do not.
    A statement boundary commits a prologue or discards an executable's local
    overlay. Neither path enumerates the enclosing environment.
    """

    def __init__(self, base):
        self.persistent = _EnvOverlay(base)
        self.local = None
        self.start = True
        self.exporting = self.seen_env = self.command = False

    @property
    def env(self):
        return self.local if self.local is not None else self.persistent

    def word(self, value, name=None, unquoted_ident=False):
        if self.start and value == "export":
            self.exporting = True
        elif not self.command:
            if name is not None and _ASSIGNMENT_NAME.fullmatch(name):
                if self.local is None:
                    self.local = _EnvOverlay(self.persistent)
                self.local.assign(name, (value.partition("=")[2], is_secret_name(name)))
            elif self.exporting:
                if not unquoted_ident or not _ASSIGNMENT_NAME.fullmatch(value):
                    self.command = True
            else:
                if value == "env" and not self.seen_env:
                    self.seen_env = True
                else:
                    self.command = True
        self.start = False

    def end_statement(self):
        if self.local is not None and not self.command:
            for name, value in self.local._overrides.items():
                self.persistent.assign(name, value)
        self.local = None
        self.start = True
        self.exporting = self.seen_env = self.command = False


def continuation_length(text, index):
    """Length of an escaped line continuation at ``index``, or 0."""
    if text[index:index + 1] != "\\":
        return 0
    nxt = text[index + 1:index + 2]
    if nxt == "\n":
        return 2
    if nxt == "\r":
        return 3 if text[index + 2:index + 3] == "\n" else 2
    return 0


def _skip_continuations(text, cursor):
    """Advance an unquoted cursor; callers consume the returned position."""
    # Do not probe the terminal non-continuation: its caller owns that cursor.
    while text[cursor:cursor + 2] in ("\\\n", "\\\r"):
        cursor += continuation_length(text, cursor)
    return cursor


def _read_variable(text, index):
    """Read one logical reference, committing every continuation exactly once.

    Return (kind, name, cursor); even a literal dollar consumes its following
    continuations. Values and quoting belong to the caller, not this reader.
    """
    cursor = _skip_continuations(text, index + 1)
    char = text[cursor:cursor + 1]
    name = []
    if char == "{":
        cursor = _skip_continuations(text, cursor + 1)
        char = text[cursor:cursor + 1]
        while char and char.isascii() and (char.isalnum() or char == "_"):
            name.append(char)
            cursor = _skip_continuations(text, cursor + 1)
            char = text[cursor:cursor + 1]
        value = "".join(name)
        if text[cursor:cursor + 1] == "}":
            return ("var", value, cursor + 1) if _ASSIGNMENT_NAME.fullmatch(
                value) else ("invalid", None, cursor + 1)
        return "invalid", None, cursor
    if char and char.isascii() and (char.isalpha() or char == "_"):
        while char and char.isascii() and (char.isalnum() or char == "_"):
            name.append(char)
            cursor = _skip_continuations(text, cursor + 1)
            char = text[cursor:cursor + 1]
        return "var", "".join(name), cursor
    if char in tuple("(?!#$@*0123456789-"):
        return "invalid", None, cursor
    return "literal", "$", cursor


def candidate_value(raw):
    if re.fullmatch(_QUOTED_VALUE, raw):
        return raw[1:-1]
    return raw


def prescan_secret_values(text):
    """Candidate secret values visible before a lexical scan can fail.

    A malformed expansion on a later line must not erase a credential that
    line one already declared. Whole lexical words retain their spelling;
    adjacent quoted or escaped fragments never become independent candidates.
    """
    found = []
    for match in _SECRET_ASSIGNMENT.finditer(text):
        name = match.group(1)
        if not is_secret_name(name):
            continue
        value = candidate_value(match.group(2))
        if value:
            found.append(value)
    return tuple(dict.fromkeys(found))


def scrub_error(exc, secrets):
    exc.secrets = tuple(dict.fromkeys(
        (*getattr(exc, "secrets", ()), *[item for item in secrets if item])))
    for secret in sorted((value for value in exc.secrets if value),
                         key=len, reverse=True):
        exc.message = exc.message.replace(secret, "[REDACTED]")
    exc.args = (f"{exc.code}:{exc.message}",)


def env_pairs(mapping):
    return tuple(sorted((name, value[0]) for name, value in mapping.items()))


def assignment_name(token):
    """Lexical assignment name only from an unquoted, unescaped prefix."""
    return getattr(token, "assignment_name", None)


def _tolerant_words(text, env):
    """Decode words from a surface with unsupported trailing shell syntax.

    Admission still rejects unsupported syntax. Quotes/escapes keep literal
    delimiters; word-boundary comments retain their terminating line break.
    Unquoted separators/group delimiters end lexical statements; redirection
    operators do not. This is discovery, not POSIX execution or subshell scope.
    Assignments use an overlay; variables use only the declarative environment.
    """
    from .launch_docker import _STATEMENT_END
    words, buffer = [], []
    index, size, quote, quote_start = 0, len(text), None, 0
    operators = "|&;<>()`!{}"
    scope = _ShellScope(env if env is not None else {})
    state = {"open": True, "name": [], "pending": None, "start": None}
    expect_redirect_target = False

    def flush(io_number=False):
        nonlocal expect_redirect_target
        if buffer or not state["open"]:
            value = "".join(buffer)
            redirect = expect_redirect_target or (
                io_number and state["open"] and value.isascii() and value.isdigit())
            word = _ShellWord(value, state["pending"], redirect,
                              bool(state["open"] and _ASSIGNMENT_NAME.fullmatch(value)),
                              raw_text=text[state["start"]:index])
            words.append(word)
            if not word.is_redirect:
                scope.word(word, word.assignment_name, word.unquoted_ident)
            expect_redirect_target = False
            buffer.clear()
        state.update(open=True, name=[], pending=None, start=None)

    def expand(position):
        kind, name, cursor = _read_variable(text, position)
        value = scope.env.get(name) if kind == "var" else None
        if value is not None:
            buffer.append(value[0])
        elif kind == "literal":
            buffer.append("$")
        else:
            buffer.append(text[position:cursor])
        return cursor

    while index < size:
        char = text[index]
        if quote is None:
            # An empty quote/expansion or escaped character still starts a word;
            # only a continuation leaves this lexical boundary unchanged.
            if char == "#" and not buffer and state["open"]:
                while index < size and text[index] not in "\r\n":
                    index += 1
                continue
            if char in "<>&|":
                # Continuations disappear before operator recognition. Commit
                # every lookahead cursor, even on a non-match, so no run of
                # continuations is rescanned and embedded &/| cannot end scope.
                flush(io_number=char in "<>")
                index = _skip_continuations(text, index + 1)
                pair = char + text[index:index + 1]
                if pair in {"<&", ">&", "&>", "<<", ">>", "<>", ">|"}:
                    index = _skip_continuations(text, index + 1)
                    third = text[index:index + 1]
                    if (pair == "&>" and third == ">") or (
                            pair == "<<" and third in {"<", "-"}):
                        index = _skip_continuations(text, index + 1)
                    expect_redirect_target = True
                elif char in "<>":
                    expect_redirect_target = True
                elif char in "&|":
                    if pair in {"&&", "||"}:
                        index = _skip_continuations(text, index + 1)
                    words.append(_STATEMENT_END)
                    scope.end_statement()
                    expect_redirect_target = False
                continue
            if char in " \t\r\n" or char in operators:
                flush()
                if char in "\r\n;(){}":
                    words.append(_STATEMENT_END)
                    scope.end_statement()
                    expect_redirect_target = False
                index += 2 if text[index:index + 2] == "\r\n" else 1
                continue
            if state["start"] is None:
                state["start"] = index
            if char == "'":
                state["open"] = False
                quote = "'"
                quote_start = index
                index += 1
                continue
            if char == '"':
                state["open"] = False
                quote = '"'
                quote_start = index
                index += 1
                continue
            if char == "\\":
                skipped = continuation_length(text, index)
                if skipped:
                    index += skipped
                    continue
                state["open"] = False
                if index + 1 < size:
                    buffer.append(text[index + 1])
                    index += 2
                else:
                    index += 1
                continue
            if char == "$":
                state["open"] = False
                index = expand(index)
                continue
            if char == "=" and state["open"] and state["name"]:
                state["pending"] = "".join(state["name"])
                state["open"] = False
                buffer.append(char)
                index += 1
                continue
            if state["open"] and (char.isalnum() or char == "_"):
                state["name"].append(char)
            else:
                state["open"] = False
            buffer.append(char)
            index += 1
            continue
        if quote == "'":
            if char == "'":
                quote = None
                index += 1
                continue
            buffer.append(char)
            index += 1
            continue
        if char == '"':
            quote = None
            index += 1
            continue
        if char == "\\":
            skipped = continuation_length(text, index)
            if skipped:
                index += skipped
                continue
            if index + 1 < size:
                nxt = text[index + 1]
                if nxt in '$`"\\':
                    buffer.append(nxt)
                    index += 2
                    continue
                buffer.append("\\")
                index += 1
                continue
            index += 1
            continue
        if char == "$":
            index = expand(index)
            continue
        buffer.append(char)
        index += 1
    flush()
    if quote is not None and words:
        words[-1].unclosed_text = text[quote_start + 1:]
    return tuple(words)


__all__ = [
    "EnvAssignment", "EnvMap", "SourceSpan", "_EnvOverlay", "assignment_name",
    "candidate_value", "continuation_length", "env_pairs", "is_secret_name",
    "normalize_env", "prescan_secret_values", "scrub_error",
]
