"""Safe static shell/argv grammar for launch declarations.

[Role]
  Turn one declarative launch script (or an already-tokenized argv) into
  literal commands with original source spans and a scoped environment
  occurrence ledger.  It is data processing: no shell is sourced, no
  substitution executes, no Docker runs, and no host environment is read as
  a deployment default.

[Invariants]
  * CRLF normalizes to LF; backslash-newline joins only outside single quotes.
  * A quote-aware scanner records exact spans, lexical assignment position and
    per-segment quoting; double quotes preserve ordinary backslashes and only
    ``$``, backtick, ``"``, ``\\`` and newline are escapes.
  * ``$NAME``/``${NAME}`` resolve only from an env established earlier in the
    same shell scope or passed explicitly, and only outside single quotes.
    Secret provenance travels with the resolved value.
  * Unquoted substitution that would require word splitting/globbing, unknown
    variables, command/arithmetic/process substitution, backticks, pipes,
    loops, conditionals, functions, source/eval, heredocs, redirections,
    globs, brace/tilde expansion and control operators reject the whole input
    with ``unsupported_shell_syntax``; no partial launch is returned.
  * ``export KEY=value``, ``export KEY``, command-local assignments and
    literal ``env`` prefixes are the only assignment forms; same-line
    assignments resolve sequentially (``A=2 B=$A``).  ``set -e/-u/-o
    pipefail`` is operational syntax and authorizes no command.
  * Multiple commands are retained; selecting one requires ``command_index``.
[Timeouts & Locks]
  Linear scan over at most 8 MiB/100,000 tokens; no locks.
[Side-Effect Policy]
  No I/O, environment lookup, subprocess, eval or mutable globals.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .launch_coords import raw_word_coords, shell_word_coords
from .launch_env import (
    EnvAssignment, SourceSpan, assignment_name as _assignment_name,
    env_pairs as _env_pairs, is_secret_name, _read_variable,
    prescan_secret_values as _prescan_secret_values,
    scrub_error as _scrub_error,
)

MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_TOKENS = 100_000
MAX_INTEGER_BITS = 4096
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)\Z", re.S)
_OPERATORS = frozenset("|&;<>()`!{}")
_GLOB = frozenset("*?[]")

class LaunchSyntaxError(ValueError):
    """Structured rejection with a stable code and an optional source span."""

    def __init__(self, message, *, code="unsupported_shell_syntax", span=None,
                 secrets=(), required_inputs=()):
        self.code, self.message, self.span = code, message, span
        #: Every secret value already resolved when this diagnostic was raised.
        #: The adapter cannot recover shell locals after an early failure, so
        #: the raising frame attaches them before unwinding.
        self.secrets = tuple(dict.fromkeys(str(item) for item in secrets if item))
        #: Independent requirements discovered before the rejection (for
        #: example ``-e MISSING`` ahead of a terminal ``-e``).  The adapter
        #: merges these even though the document did not parse.
        self.required_inputs = tuple(dict.fromkeys(
            str(item) for item in required_inputs if item))
        super().__init__(f"{code}:{message}")

@dataclass(frozen=True)
class Segment:
    kind: str  # literal | var
    text: str
    quoted: bool


@dataclass(frozen=True)
class WordToken:
    segments: tuple[Segment, ...]
    span: SourceSpan
    quoted: bool
    offset: int = 0
    raw_text: str = ""
    assignment_name: str | None = None
    json_escaped: bool = False


@dataclass(frozen=True)
class Newline:
    span: SourceSpan
    offset: int = 0


@dataclass(frozen=True)
class Word:
    text: str
    span: SourceSpan
    quoted: bool
    offset: int = 0
    secret: bool = False
    segments: tuple[Segment, ...] = ()
    raw_text: str = ""
    json_escaped: bool = False
    #: Per-character inner-text -> exact outer-document source interval
    #: ``(start_line, start_col, start_offset, end_line, end_col, end_offset)``.
    char_coords: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class LaunchCommand:
    argv: tuple[str, ...]
    words: tuple[Word, ...]
    env: tuple[tuple[str, str], ...]
    inline_env: tuple[tuple[str, str], ...]
    span: SourceSpan
    env_occurrences: tuple[EnvAssignment, ...] = ()
    secrets: tuple[str, ...] = ()

    @property
    def secret_values(self):
        return tuple(value for name, value in self.env if is_secret_name(name))


@dataclass(frozen=True)
class ShellProgram:
    source: str
    commands: tuple[LaunchCommand, ...]
    env: tuple[tuple[str, str], ...]
    operational: tuple[str, ...]
    command_index: int | None
    assignments: tuple[EnvAssignment, ...] = ()
    source_text: str = ""
    #: Every resolved assignment value whose *name* was secret-named, including
    #: shell-local variables that never reach a child process.  The parser is a
    #: diagnostic surface too, so these values are scrubbed from every report.
    secret_values: tuple[str, ...] = ()


def _span(source, line, col, end_line, end_col):
    return SourceSpan(source, line, col, end_line, end_col)


def _check_text_size(text):
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise LaunchSyntaxError("input_exceeds_8_mib", code="input_too_large")
    if len(text) > MAX_TEXT_BYTES:
        raise LaunchSyntaxError("input_exceeds_8_mib", code="input_too_large")


class _Scanner:
    def __init__(self, text, source):
        _check_text_size(text)
        self.text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.source, self.i, self.line, self.col = source, 0, 1, 1
        self.last_line, self.last_col = 1, 1
        self.tokens = []

    def _advance(self, count=1):
        for _ in range(count):
            if self.i >= len(self.text):
                return
            self.last_line, self.last_col = self.line, self.col
            if self.text[self.i] == "\n":
                self.line, self.col = self.line + 1, 1
            else:
                self.col += 1
            self.i += 1

    def _continuation(self):
        return self.text.startswith("\\\n", self.i)

    def _error(self, message, line=None, col=None):
        raise LaunchSyntaxError(message, span=_span(
            self.source, line or self.line, col or self.col,
            line or self.line, col or self.col))

    def _check_token_cap(self):
        if len(self.tokens) > MAX_TOKENS:
            raise LaunchSyntaxError("input_exceeds_100000_tokens",
                                    code="input_too_large")

    def run(self):
        while self.i < len(self.text):
            char = self.text[self.i]
            if char in " \t":
                self._advance(); continue
            if self._continuation():
                self._advance(2); continue
            if char == "\n":
                span = _span(self.source, self.line, self.col, self.line, self.col)
                self.tokens.append(Newline(span, self.i))
                self._check_token_cap()
                self._advance(); continue
            if char == "#":
                while self.i < len(self.text) and self.text[self.i] != "\n":
                    self._advance()
                continue
            if char == "`":
                self._error("command_substitution_not_supported")
            if char in _OPERATORS or char in _GLOB or char in "~":
                self._error(f"operator_or_expansion_not_supported:{char}")
            self._word()
            self._check_token_cap()
        return tuple(self.tokens)

    def _word(self):
        start_line, start_col, start_offset = self.line, self.col, self.i
        segments, literal, quoted = [], [], False
        # Lexical assignment name: only an *unquoted, unescaped* name prefix
        # before the first ``=`` forms ``NAME=value``.  ``VLLM\_X=1`` is a
        # command word, not an assignment.
        name_chars, name_open, assignment_name = [], True, None

        def close_name():
            nonlocal name_open
            name_open = False

        def flush(segment_quoted):
            if literal:
                segments.append(Segment("literal", "".join(literal), segment_quoted))
                literal.clear()

        while self.i < len(self.text):
            char = self.text[self.i]
            if char in " \t\n":
                break
            if self._continuation():
                self._advance(2); continue
            if char == "\\":
                close_name()
                self._advance()
                if self.i >= len(self.text):
                    self._error("dangling_backslash")
                literal.append(self.text[self.i]); self._advance()
            elif char == "'":
                quoted = True; close_name(); flush(False); self._advance()
                opening = (self.line, self.col)
                before = len(literal)
                while self.i < len(self.text) and self.text[self.i] != "'":
                    literal.append(self.text[self.i]); self._advance()
                if self.i >= len(self.text):
                    self._error("unterminated_single_quote", *opening)
                if len(literal) == before:
                    literal.append("")
                self._advance(); flush(True)
            elif char == '"':
                quoted = True; close_name(); flush(False); self._advance()
                opening = (self.line, self.col)
                before = len(literal)
                while self.i < len(self.text) and self.text[self.i] != '"':
                    current = self.text[self.i]
                    if current == "`":
                        self._error("command_substitution_not_supported")
                    if current == "\\" and self.i + 1 < len(self.text):
                        nxt = self.text[self.i + 1]
                        if nxt == "\n":
                            self._advance(2); continue
                        if nxt in '$`"\\':
                            literal.append(nxt)
                        else:
                            # POSIX double quotes preserve other backslashes.
                            literal.append("\\"); literal.append(nxt)
                        self._advance(2); continue
                    if current == "$":
                        flush(True); segments.append(self._variable(True)); continue
                    literal.append(current); self._advance()
                if self.i >= len(self.text):
                    self._error("unterminated_double_quote", *opening)
                if len(literal) == before and not segments:
                    literal.append("")
                self._advance(); flush(True)
            elif char == "$":
                close_name(); flush(False); segments.append(self._variable(False))
            elif char == "`":
                self._error("command_substitution_not_supported")
            elif char in _OPERATORS:
                self._error(f"control_operator_not_supported:{char}")
            elif char in _GLOB:
                self._error(f"glob_expansion_not_supported:{char}")
            elif char == "~":
                self._error("tilde_expansion_not_supported")
            elif char == "!" and not literal and not segments:
                self._error("history_expansion_not_supported")
            else:
                if name_open:
                    if char == "=" and name_chars:
                        assignment_name = "".join(name_chars)
                        name_open = False
                    elif (char.isalnum() or char == "_") and (
                            name_chars or char.isalpha() or char == "_"):
                        if assignment_name is None:
                            name_chars.append(char)
                    else:
                        name_open = False
                literal.append(char); self._advance()
        close_name()
        flush(False)
        if not segments:
            self._error("empty_word")
        self.tokens.append(WordToken(
            tuple(segments),
            _span(self.source, start_line, start_col, self.last_line, self.last_col),
            quoted, start_offset, self.text[start_offset:self.i], assignment_name))

    def _variable(self, quoted):
        start = self.i
        kind, name, cursor = _read_variable(self.text, self.i)
        self._advance(cursor - self.i)
        if kind == "var":
            return Segment("var", name, quoted)
        char = self.text[self.i:self.i + 1]
        if kind == "invalid":
            # Classify the consumed spelling without restarting the reader.
            if self.text.find("{", start + 1, cursor) >= 0:
                if self.i == len(self.text) and self.text[self.i - 1] != "}":
                    self._error("unterminated_parameter_expansion")
                self._error("parameter_expansion_not_supported")
            if char.isdigit():
                self._error("invalid_variable_name")
            if char in tuple("?!#$@*-"):
                self._error(f"special_parameter_not_supported:{char}")
            if char != "(":
                self._error("parameter_expansion_not_supported")
        if char == "(":
            self._error("command_or_arithmetic_substitution_not_supported")
        if char == "'":
            # Inside double quotes ``$'`` is an ordinary dollar followed by a
            # literal single quote; ANSI-C quoting only starts unquoted.
            if quoted:
                return Segment("literal", "$", quoted)
            self._error("ansi_c_quoting_not_supported")
        if char == '"':
            # Likewise ``$"`` is only locale quoting outside double quotes.
            if quoted:
                return Segment("literal", "$", quoted)
            self._error("locale_quoting_not_supported")
        if char and char.isalnum():
            self._error("invalid_variable_name")
        return Segment("literal", "$", quoted)


def scan(text, source="<launch>"):
    """Quote-aware tokenizer preserving spans; reject unsafe grammar."""
    return _Scanner(text, source).run()


def _resolve(token, env):
    parts, secret = [], False
    for segment in token.segments:
        if segment.kind == "literal":
            parts.append(segment.text); continue
        value = env.get(segment.text)
        if value is None:
            raise LaunchSyntaxError(f"unknown_variable:{segment.text}", span=token.span)
        text, item_secret = value[0], value[1]
        secret = secret or item_secret or is_secret_name(segment.text)
        if not segment.quoted:
            if not text:
                raise LaunchSyntaxError(
                    f"empty_unquoted_variable:{segment.text}", span=token.span)
            if any(char.isspace() or char in _GLOB for char in text):
                raise LaunchSyntaxError(
                    f"word_splitting_or_globbing_required:{segment.text}",
                    span=token.span)
        parts.append(text)
    text = "".join(parts)
    if not text and not token.quoted:
        raise LaunchSyntaxError("empty_unquoted_value", span=token.span)
    char_coords = (shell_word_coords(
        token.raw_text, text, token.span.line_start, token.span.column_start,
        token.offset, token.json_escaped) if token.raw_text else ())
    return Word(text, token.span, token.quoted, token.offset, secret,
                token.segments, token.raw_text, token.json_escaped, char_coords)


def resolve_token(token, env):
    """Resolve one scanned word against an explicit literal environment."""
    mapping = {name: (str(value), is_secret_name(name))
               for name, value in dict(env).items()}
    return _resolve(token, mapping)


def parse_argv(argv, source="<argv>", *, validate=True):
    """Already-tokenized literal argv; no expansion or runtime lookup."""
    if not isinstance(argv, (list, tuple)):
        raise LaunchSyntaxError("argv_must_be_a_list", code="invalid_argv")
    if not argv:
        raise LaunchSyntaxError("empty_argv", code="ambiguous_launch")
    if len(argv) > MAX_TOKENS:
        raise LaunchSyntaxError("input_exceeds_100000_tokens", code="input_too_large")
    words, column, total = [], 1, 0
    for value in argv:
        if not isinstance(value, str):
            raise LaunchSyntaxError("argv_member_not_text", code="invalid_argv")
        _check_text_size(value)
        total += len(value.encode("utf-8"))
        if total > MAX_TEXT_BYTES:
            raise LaunchSyntaxError("input_exceeds_8_mib", code="input_too_large")
        words.append(Word(value, _span(source, 1, column, 1,
                                       column + max(0, len(value) - 1)),
                          True, 0, False, (), value,
                          char_coords=raw_word_coords(value, 1, column, 0)))
        column += len(value) + 1
    if validate:
        _validate_command([item.text for item in words])
    return ShellProgram(source, (LaunchCommand(
        tuple(item.text for item in words), tuple(words), (), (),
        words[0].span),), (), (), 0, (), "")


def _validate_command(argv):
    if not argv:
        raise LaunchSyntaxError("empty_command", code="ambiguous_launch")
    if argv[0] == "vllm":
        if len(argv) < 2 or argv[1] != "serve":
            raise LaunchSyntaxError("vllm_requires_serve_subcommand")
        return "vllm"
    if argv[0] in {"python", "python3"}:
        if len(argv) < 3 or argv[1] != "-m" or argv[2] not in {
                "vllm.entrypoints.openai.api_server", "vllm.entrypoints.cli.main"}:
            raise LaunchSyntaxError("python_requires_supported_vllm_module")
        return "python"
    if argv[0] == "docker":
        if len(argv) < 2 or argv[1] != "run":
            raise LaunchSyntaxError("docker_requires_run_subcommand")
        return "docker"
    raise LaunchSyntaxError(f"unsupported_launch_command:{argv[0]}")


def command_kind(argv):
    """Classify a validated command without importing model code."""
    return _validate_command(tuple(argv))


def _resolve_assignment(token, env, order, *, inline, exported):
    word = _resolve(token, env)
    match = _ASSIGNMENT.match(word.text)
    if not match:
        raise LaunchSyntaxError("invalid_assignment", span=token.span)
    name, value = match.group(1), match.group(2)
    secret = word.secret or is_secret_name(name)
    split = word.text.find("=")
    coords = (word.char_coords[split + 1:]
              if split >= 0 and word.char_coords else ())
    if len(coords) != len(value) + 1:
        coords = ()
    record = EnvAssignment(name, value, token.span, order, inline, exported,
                           secret, coords)
    return name, (value, secret, record), record


def parse_shell(text, source="<launch>", *, env=None, command_index=None,
                validate=True):
    """Parse one declarative shell input; multiple commands stay unmerged."""
    _check_text_size(text)
    values = {str(name): (str(value), is_secret_name(name), None)
              for name, value in dict(env or {}).items()}
    exported = set(values)
    history, commands, operational = [], [], []
    known_secrets = [value for value, secret, _record in values.values() if secret]
    # Pre-extract secret-named assignments before the full scan.  A scanner
    # failure on a later line must still own the provenance declared earlier.
    known_secrets.extend(_prescan_secret_values(text))
    try:
        tokens = scan(text, source)
    except LaunchSyntaxError as exc:
        _scrub_error(exc, known_secrets)
        raise
    order = [0]

    def next_order():
        order[0] += 1
        return order[0]

    def note(record):
        if record.secret and record.value:
            known_secrets.append(record.value)

    pending = []
    try:
        for token in (*tokens, Newline(_span(source, 0, 0, 0, 0))):
            if isinstance(token, WordToken):
                pending.append(token); continue
            if pending:
                command = _line(pending, values, history, commands, operational,
                                source, validate, next_order, exported, note)
                if command is not None:
                    commands.append(command)
            pending = []
        if not commands:
            raise LaunchSyntaxError("no_launch_command_found", code="ambiguous_launch")
        selected = command_index if command_index is not None else (
            0 if len(commands) == 1 else None)
        if selected is None:
            raise LaunchSyntaxError("multiple_launch_commands_require_command_index",
                                    code="ambiguous_launch")
        if selected < 0 or selected >= len(commands):
            raise LaunchSyntaxError("command_index_out_of_range", code="ambiguous_launch")
    except LaunchSyntaxError as exc:
        # A shell statement (``export $API_KEY``, ``set``, a malformed child
        # command) can fail after a secret assignment was already resolved.
        # The adapter only sees the exception, so carry every collected value
        # on this diagnostic before unwinding.
        _scrub_error(exc, known_secrets)
        raise
    return ShellProgram(source, tuple(commands), _env_pairs(values),
                        tuple(operational), selected, tuple(history),
                        text.replace("\r\n", "\n").replace("\r", "\n"),
                        tuple(dict.fromkeys(known_secrets)))


def _line(tokens, env, history, commands, operational, source, validate, next_order,
          exported, note):
    first = _resolve(tokens[0], env)
    first_name = first.text
    if first_name == "export":
        for token in tokens[1:]:
            name = _assignment_name(token)
            if name is not None:
                name, value, record = _resolve_assignment(
                    token, env, next_order(), inline=False, exported=True)
                env[name] = value; history.append(record); exported.add(name); note(record)
            else:
                word = _resolve(token, env)
                if word.text not in env:
                    raise LaunchSyntaxError(
                        f"export_unknown_variable:{word.text}", span=word.span)
                previous = env[word.text]
                # The value came from an earlier assignment or an explicit
                # caller value.  A bare export only changes visibility, so
                # keep the earlier literal's exact value map when one exists.
                origin = previous[2] if len(previous) > 2 else None
                coords = (origin.char_coords
                          if origin is not None and origin.has_value else ())
                record = EnvAssignment(word.text, previous[0], word.span,
                                       next_order(), False, True, previous[1],
                                       coords, has_value=False)
                history.append(record); exported.add(word.text); note(record)
        return None
    if first_name == "set":
        options = [_resolve(token, env).text for token in tokens[1:]]
        allowed = {"-e", "-u", "-o", "pipefail", "-eu", "-euo",
                   "-o pipefail", "-eu pipefail"}
        if not options or any(item not in allowed for item in options):
            raise LaunchSyntaxError("unsupported_set_prologue", span=first.span)
        operational.append(" ".join([first_name, *options]))
        return None
    local = dict(env)
    inline, records, index = [], [], 0
    while index < len(tokens):
        token = tokens[index]
        name = _assignment_name(token)
        if name is None:
            break
        name, value, record = _resolve_assignment(
            token, local, next_order(), inline=True, exported=False)
        local[name] = value; inline.append((name, value[0])); records.append(record)
        note(record)
        index += 1
    if index < len(tokens) and _resolve(tokens[index], local).text == "env":
        index += 1
        while index < len(tokens):
            token = tokens[index]
            name = _assignment_name(token)
            if name is None:
                break
            name, value, record = _resolve_assignment(
                token, local, next_order(), inline=True, exported=False)
            local[name] = value; inline.append((name, value[0])); records.append(record)
            note(record)
            index += 1
    words = [_resolve(token, local) for token in tokens[index:]]
    if not words:
        if index >= len(tokens):
            for record in records:
                env[record.name] = (record.value, record.secret, record)
                if record.name in exported:
                    # POSIX: an assignment to an already-exported variable
                    # keeps the export attribute (``export A=1; A=2``).
                    history.append(replace(record, exported=True))
            return None
        raise LaunchSyntaxError("env_requires_command", span=tokens[0].span)
    argv = [word.text for word in words]
    if validate:
        try:
            _validate_command(argv)
        except LaunchSyntaxError as exc:
            # A failed diagnostic still owns every secret value already in
            # scope, including shell-local declarations that never reach the
            # child environment.
            exc.secrets = tuple(dict.fromkeys(
                value for value, secret, _ in local.values() if secret))
            raise
    # Only the process environment reaches the child: explicitly exported
    # shell variables plus command-local assignments.  A plain ``FOO=bar``
    # line remains a shell variable and is not inherited by later commands.
    scoped = {name: value for name, value in local.items() if name in exported}
    for name, _value in inline:
        scoped[name] = local[name]
    command_env = _env_pairs(scoped)
    effective = history + records
    last, value_coords = {}, {}
    for record in effective:
        last.pop(record.name, None)
        last[record.name] = record
        if record.has_value:
            value_coords[record.name] = record.char_coords
    keep = []
    for record in last.values():
        if not record.has_value and not record.char_coords:
            # A visibility-only ``export NAME`` keeps the most recent value
            # origin's exact map.  It is not a new assignment and must not
            # erase document/literal provenance.
            coords = value_coords.get(record.name)
            if coords:
                record = replace(record, char_coords=coords)
        keep.append(record)
    secrets = tuple(value for value, secret, _ in scoped.values() if secret)
    return LaunchCommand(
        tuple(argv), tuple(words), command_env, tuple(inline),
        _span(source, words[0].span.line_start, words[0].span.column_start,
              words[-1].span.line_end, words[-1].span.column_end),
        tuple(effective), secrets)


def selected_command(program, command_index=None):
    """Return the one selected command; never merge separate processes."""
    index = command_index if command_index is not None else program.command_index
    if index is None or not 0 <= index < len(program.commands):
        raise LaunchSyntaxError("multiple_launch_commands_require_command_index",
                                code="ambiguous_launch")
    return program.commands[index]


__all__ = [
    "EnvAssignment", "LaunchCommand", "LaunchSyntaxError", "MAX_INTEGER_BITS",
    "MAX_TEXT_BYTES", "MAX_TOKENS", "SourceSpan", "Word", "WordToken",
    "ShellProgram", "command_kind", "is_secret_name", "parse_argv",
    "parse_shell", "resolve_token", "scan", "selected_command",
]
