"""Docker envelope unwrapping for one literal launch declaration.

[Role]
  Recognize only the Docker ``run`` envelope, literal ``-e/--env`` forwardings,
  visibility and operational options, and one literal ``sh -c`` unwrap.  The
  inner script's exact structured ``char_coords`` are composed through shell
  parsing, including a document-defined environment variable spliced into the
  script by ``$NAME`` expansion.

[Invariants]
  * No Docker execution, host env lookup or ``--env-file`` read.
  * Credential discovery shares the same bounded ``-e/--env`` forwarding and
    single ``sh -c`` unwrap through ``discovery_wrapper``; it never executes
    a shell and returns ``None`` for unsupported option shapes.  Leading
    prologues persist in zero-copy overlays; executable-local assignments are
    discarded before visiting the next admitted launch statement's wrapper.
  * An exact ``$NAME``/``${NAME}`` script word resolves from the explicit
    container/forwarded environment before the inner scan, with admission's
    bounded rule and no value search.
  * Unknown options, a second nested ``sh -c`` and ``--entrypoint`` reject.
  * A structured script token passes its per-character map directly to the
    inner shell words.  An expanded script carries the exact per-variable map
    of the occurrence that produced it; a forwarded or external expansion
    without a map raises ``unmapped_shell_expansion_provenance`` instead of
    searching another variable with an equal value.
  * Container environment starts empty; a forwarded name is resolved only
    from the outer literal snapshot supplied by the caller.
[Timeouts & Locks]
  Linear in the finite argv; no locks, I/O or devices.
[Side-Effect Policy]
  No subprocess, environment lookup, file read or network.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .launch_coords import normalize_newlines
from .launch_env import EnvMap, _EnvOverlay, _ShellWord, normalize_env
from .launch_parser import (
    EnvAssignment, LaunchSyntaxError, SourceSpan, is_secret_name,
    command_kind, parse_shell, selected_command,
)
from .launch_spans import map_shell_records, map_shell_words

_VALUE_OPTIONS = frozenset({"-v", "--volume", "--device", "--cpus", "--memory",
                            "--network", "--name", "-w", "--workdir", "--ipc",
                            "--shm-size", "--gpus"})
_FLAG_OPTIONS = frozenset({"--rm", "-it", "-i", "-t", "--privileged", "--init",
                           "-d", "--detach"})
_DISCOVERY_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_DISCOVERY_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHELL_NAMES = frozenset({"sh", "bash", "/bin/sh", "/bin/bash"})
_SET_OPTIONS = frozenset({"-e", "-u", "-o", "pipefail", "-eu", "-euo",
                          "-o pipefail", "-eu pipefail"})
# Identity-only: quoted/escaped delimiter strings remain ordinary argv values.
_STATEMENT_END = object()


@dataclass(frozen=True)
class WrapperOccurrence:
    text: str
    span: SourceSpan


@dataclass(frozen=True)
class DocumentInputError(ValueError):
    """Structured adapter failure that may become a required-input diagnostic."""
    code: str
    message: str
    name: str | None = None
    secrets: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()


def _strip_quotes(text):
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1]
    return text


def _expansion_name(text, word):
    """``(name, literal_reference)`` for an exact ``$NAME`` script word."""
    segments = getattr(word, "segments", ())
    if len(segments) == 1 and getattr(segments[0], "kind", None) == "var":
        name = getattr(segments[0], "text", "")
        return (name, False) if name.isidentifier() else (None, False)
    stripped = _strip_quotes(text).strip()
    if stripped.startswith("$"):
        name = (stripped[2:-1] if stripped.startswith("${")
                and stripped.endswith("}") else stripped[1:])
        return (name, True) if name.isidentifier() else (None, True)
    return None, False


def _expand_env_script(text, word, container_env, container_coords,
                       outer_env, outer_coords):
    """Resolve a literal ``$NAME``/``"$NAME"`` script argument before parsing.

    A shell-source Docker command already expanded ``$SCRIPT`` lexically, so
    the variable identity comes from the parsed word's segment table and the
    substituted text must equal that occurrence's value.  A literal reference
    resolves against the container snapshot; an already-expanded reference
    resolves against the outer shell snapshot that performed the expansion.
    The exact per-name source map is carried on both paths; a forwarded or
    external variable without a document map returns ``None`` coordinates and
    the caller fails closed instead of searching for an equal value.
    """
    name, literal_reference = _expansion_name(text, word)
    source_env = container_env if literal_reference else outer_env
    source_coords = container_coords if literal_reference else outer_coords
    if name is None or name not in source_env:
        return None
    raw = source_env[name]
    value = raw[0] if isinstance(raw, (tuple, list)) else raw
    normalized, _mapped = normalize_newlines(str(value), tuple(
        source_coords.get(name) or ()))
    if not literal_reference and text not in {str(value), normalized}:
        return None
    coords = source_coords.get(name)
    return str(value), (tuple(coords) if coords else None)


def _device_target(value):
    """Container target of one Docker ``--device`` mapping."""
    parts = str(value).split(":")
    if len(parts) >= 2 and parts[1].startswith("/"):
        return parts[1]
    return parts[0]


def _canonical_rank(token):
    text = str(token).strip()
    if not text or not text.isdigit():
        return None
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return None


def _device_rank(path):
    import re
    match = re.search(r"(?:davinci|npu)(\d+)\s*$", str(path))
    return _canonical_rank(match.group(1)) if match else None


def visibility_ranks(container_env, unidentified):
    """Resolve the runtime mask and whether visibility was explicitly declared."""
    raw = dict(container_env).get("ASCEND_RT_VISIBLE_DEVICES", None)
    if raw is None:
        return set(), False
    value = raw[0] if isinstance(raw, tuple) else raw
    masked = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        rank = _canonical_rank(part)
        if rank is None:
            unidentified.append(f"ASCEND_RT_VISIBLE_DEVICES={part}")
        else:
            masked.add(rank)
    return masked, True


def _prefix_scope(words, base):
    """Consume operational ``set`` and scoped env/export/assignment prefixes.

    A command-local ``NAME=literal`` wins over the enclosing frame through a
    zero-copy overlay installed only in the override map.  A bare
    ``export`` retains mixed assignments/names, including wrapper names, until
    ``_STATEMENT_END``.  Leading assignment/export statements accumulate local
    overrides; the first executable's arguments stay inside its own statement.
    Keywords use decoded identity; assignments and bare export names retain
    lexical eligibility. Invalid export statements discard all local values.
    """
    local = None
    index = 0
    exporting = False
    seen_env = False
    start = True
    while index < len(words):
        word = words[index]
        if word is _STATEMENT_END:
            exporting = seen_env = False
            start = True
            index += 1
            continue
        if getattr(word, "is_redirect", False):
            index += 1
            continue
        unquoted = not isinstance(word, _ShellWord) or word.unquoted_ident
        if word == "set" and start:
            end = index + 1
            while end < len(words) and words[end] is not _STATEMENT_END:
                if not getattr(words[end], "is_redirect", False) and words[end] not in _SET_OPTIONS:
                    break
                end += 1
            if end > index + 1 and (end == len(words)
                                    or words[end] is _STATEMENT_END):
                index = end
                continue
            break
        if word == "export" and start:
            # Validate the whole statement before applying any of its values.
            # An invalid argument is not an executable wrapper after export.
            end = index + 1
            while end < len(words) and words[end] is not _STATEMENT_END:
                arg = words[end]
                if not getattr(arg, "is_redirect", False):
                    assign = (arg.assignment_name if isinstance(arg, _ShellWord)
                              else _DISCOVERY_ASSIGNMENT.match(arg))
                    bare = not isinstance(arg, _ShellWord) or arg.unquoted_ident
                    if not assign and not (bare and _DISCOVERY_NAME.fullmatch(arg)):
                        break
                end += 1
            if end < len(words) and words[end] is not _STATEMENT_END:
                break
            exporting = True
            start = False
            index += 1
            continue
        if (word.assignment_name is not None if isinstance(word, _ShellWord)
                else _DISCOVERY_ASSIGNMENT.match(word)):
            name, _equals, value = word.partition("=")
            if local is None:
                local = _EnvOverlay(base)
            local.assign(name, (value, is_secret_name(name)))
            start = False
            index += 1
            continue
        if exporting and unquoted and _DISCOVERY_NAME.fullmatch(word):
            index += 1
            continue
        if not exporting and not seen_env and word == "env":
            seen_env = True
            start = False
            index += 1
            continue
        break
    return index, (local if local is not None else base)


def _exact_script(word, env):
    """Resolve one exact ``$NAME``/``${NAME}`` script argument from ``env``.

    This mirrors the bounded admission-side literal reference: only a whole
    word naming one variable is resolved, only from the explicit environment
    the caller supplied, and an absent variable leaves the spelling untouched.
    """
    text = str(word).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    if not text.startswith("$"):
        return str(word)
    name = (text[2:-1] if text.startswith("${") and text.endswith("}")
            else text[1:])
    if not name.isidentifier():
        return str(word)
    raw = env.get(name) if env is not None else None
    if raw is None:
        return str(word)
    return str(raw[0] if isinstance(raw, (tuple, list)) else raw)


def discovery_wrapper(words, outer_env=None):
    """Bounded shell-free unwrap shared with global credential discovery.

    Mirrors ``docker``'s literal ``-e/--env`` forwarding and one
    ``sh -c``/``bash -c`` script for a supported wrapper.  Leading literal
    assignments and supported export prefixes scope the forwarded values, and
    an exact ``$NAME`` script argument is resolved before the inner scan.
    Returns ``(inner_script, env)`` or ``None`` when the word list is not a
    recognized wrapper; no shell is executed and no host environment is read.
    """
    words = tuple(word if word is _STATEMENT_END or isinstance(word, _ShellWord)
                  else str(word) for word in words or ()
                  if not getattr(word, "is_redirect", False))
    index, effective = _prefix_scope(words, normalize_env(outer_env))
    end = index
    while end < len(words) and words[end] is not _STATEMENT_END:
        end += 1
    words = words[index:end]
    unwrapped = _discovery_inner(words, effective)
    if unwrapped is not None:
        inner, inner_env = unwrapped
        if len(inner) >= 3 and inner[0] in _SHELL_NAMES and inner[1] == "-c":
            return _exact_script(inner[2], inner_env), inner_env
    return None


def _discovery_inner(words, effective):
    """Unwrap one complete statement's Docker options without crossing lines."""
    if not words:
        return None
    if words[0] in _SHELL_NAMES:
        return words, effective
    if len(words) < 2 or words[0] != "docker" or words[1] != "run":
        return None
    outer, container_env = effective, EnvMap()
    position = 2
    while position < len(words):
        item = words[position]
        if item == "--":
            position += 1
            break
        if not item.startswith("-"):
            break
        key, equals, attached = item.partition("=")
        if key in {"-e", "--env"}:
            if equals:
                value = attached
            else:
                position += 1
                if position >= len(words):
                    return None
                value = words[position]
            if "=" in value:
                name, literal = value.split("=", 1)
                container_env[str(name)] = (literal, is_secret_name(name))
            elif value in outer:
                container_env[str(value)] = outer[value]
            position += 1
            continue
        if key in _VALUE_OPTIONS:
            position += 1 if equals else 2
            continue
        if item in _FLAG_OPTIONS:
            position += 1
            continue
        return None
    if position >= len(words):
        return None
    return words[position + 1:], container_env


def iter_discovery_wrappers(words, outer_env=None):
    """Visit wrappers after prologues and admitted launch commands only.

    Assignment-only/export/standalone-env statements retain discovery's
    conservative shell overlay. Executable statements receive a disposable
    inline overlay; neither their assignments nor Docker env options persist.
    Merge only prologue overrides, keeping overlay depth and work bounded.
    """
    persistent = _EnvOverlay(normalize_env(outer_env))
    statement = []
    for word in (*words, _STATEMENT_END):
        if word is not _STATEMENT_END:
            if not getattr(word, "is_redirect", False):
                statement.append(word if isinstance(word, _ShellWord) else str(word))
            continue
        index, effective = _prefix_scope(statement, persistent)
        command = statement[index:]
        statement = []
        if not command:
            if effective is not persistent:
                for name, value in effective._overrides.items():
                    persistent.assign(name, value)
            continue
        if command[0] == "eval":
            yield " ".join(command[1:]), effective
            return  # Only the actual eval payload has shell-string semantics.
        if command[0] == "docker" or command[0] in _SHELL_NAMES:
            unwrapped = _discovery_inner(command, effective)
            if unwrapped is None:
                return
            command, inner_env = unwrapped
            if len(command) >= 3 and command[0] in _SHELL_NAMES and command[1] == "-c":
                yield _exact_script(command[2], inner_env), inner_env
                if len(command) != 3:
                    return  # Tolerate a tail for discovery, not for advancing.
                continue
        try:
            if command_kind(command) == "docker":
                return  # A partial or nested Docker envelope is not a launch.
        except LaunchSyntaxError:
            return


def docker(argv, words, outer_env, *, source, command_index, env_files,
           secret_sink=None, requirement_sink=None, env_coords=None):
    if len(argv) < 3 or argv[0] != "docker" or argv[1] != "run":
        raise LaunchSyntaxError("docker_requires_run_envelope")
    index, container_env, operational, devices, image = 2, {}, [], [], None
    records, missing_env, missing_files = [], [], []
    forwarded, unidentified = [], []
    env_files = dict(env_files or {})
    outer_coords = {
        str(name): (tuple(coords) if coords else None)
        for name, coords in dict(env_coords or {}).items()}
    container_env_coords = dict(outer_coords)

    def need(name):
        if requirement_sink is not None:
            requirement_sink.add(f"env:{name}")

    while index < len(argv):
        item, word = argv[index], words[index]
        if item == "--":
            index += 1
            if index >= len(argv):
                raise LaunchSyntaxError("docker_run_requires_image")
            image, index = argv[index], index + 1
            break
        if not item.startswith("-"):
            image, index = item, index + 1
            break
        key, equals, attached = item.partition("=")
        value_word = (words[index] if not equals else word)
        if key in {"-e", "--env"}:
            if equals:
                value = attached
            else:
                index += 1
                if index >= len(argv):
                    raise DocumentInputError("needs_env_value",
                                             "container_env_value_required", key)
                value, value_word = argv[index], words[index]
            if "=" in value:
                name, literal = value.split("=", 1)
                container_env[name] = (literal, is_secret_name(name))
                position = value_word.text.find(value)
                split = position + len(name) if position >= 0 else -1
                candidate = (value_word.char_coords[split + 1:]
                             if split >= 0 and value_word.char_coords
                             and value_word.text[split:split + 1] == "="
                             else ())
                container_env_coords[name] = (
                    candidate if len(candidate) == len(literal) + 1 else None)
                records.append(EnvAssignment(name, literal, value_word.span,
                                             len(records) + 1, False, True,
                                             is_secret_name(name)))
                if is_secret_name(name) and secret_sink is not None:
                    secret_sink.add(literal)
                forwarded.append(name)
            elif value in outer_env:
                raw = outer_env[value]
                literal, secret = ((raw[0], raw[1]) if isinstance(raw, tuple)
                                   else (raw, is_secret_name(value)))
                container_env[value] = (literal, secret)
                container_env_coords[value] = dict(env_coords or {}).get(
                    str(value))
                records.append(EnvAssignment(value, literal,
                                             value_word.span, len(records) + 1,
                                             False, True, secret))
                if secret and secret_sink is not None:
                    secret_sink.add(literal)
                forwarded.append(value)
            else:
                missing_env.append(value)
                need(value)
        elif key == "--env-file":
            if equals:
                path = attached
            else:
                index += 1
                if index >= len(argv):
                    raise DocumentInputError("needs_env_file",
                                             "env_file_contents_required", key)
                path = argv[index]
            if path not in env_files:
                missing_files.append(path)
                need(path)
                index += 1
                continue
            for name, value in dict(env_files[path]).items():
                container_env[str(name)] = (str(value), is_secret_name(str(name)))
                container_env_coords[str(name)] = None
                records.append(EnvAssignment(str(name), str(value), value_word.span,
                                             len(records) + 1, False, True,
                                             is_secret_name(str(name))))
                if is_secret_name(str(name)) and secret_sink is not None:
                    secret_sink.add(str(value))
                forwarded.append(str(name))
        elif key in _VALUE_OPTIONS:
            if equals:
                value = attached
            else:
                index += 1
                if index >= len(argv):
                    raise LaunchSyntaxError(f"docker_option_requires_value:{key}")
                value = argv[index]
            operational.append(WrapperOccurrence(item, value_word.span))
            if key == "--device":
                devices.append(_device_target(value))
        elif key == "--entrypoint":
            raise LaunchSyntaxError("docker_entrypoint_changes_executed_program")
        elif item in _FLAG_OPTIONS:
            operational.append(WrapperOccurrence(item, word.span))
        else:
            raise LaunchSyntaxError(f"unknown_docker_option:{item}")
        index += 1
    if image is None:
        raise LaunchSyntaxError("docker_run_requires_image")
    inner, inner_words = list(argv[index:]), list(words[index:])
    if not inner:
        raise LaunchSyntaxError("docker_run_requires_supported_command")
    if inner[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}:
        if len(inner) != 3 or inner[1] != "-c":
            raise LaunchSyntaxError("sh_wrapper_requires_single_literal_c")
        quoted = inner_words[2]
        expanded = _expand_env_script(
            inner[2], quoted, container_env, container_env_coords,
            outer_env, outer_coords)
        if expanded is not None:
            script_text, source_map = expanded
            if not source_map:
                # The variable expanded into this script but its origin has no
                # document coordinates.  Never search another variable with an
                # equal value: fail closed on the exact occurrence.
                raise LaunchSyntaxError(
                    "unmapped_shell_expansion_provenance",
                    code="unmapped_shell_expansion_provenance")
        else:
            script_text, source_map = inner[2], quoted.char_coords
        script, script_coords = normalize_newlines(script_text, source_map)
        if not script_coords:
            raise LaunchSyntaxError(
                "unmapped_shell_expansion_provenance",
                code="unmapped_shell_expansion_provenance")
        program = parse_shell(script, source,
                              env={k: v[0] for k, v in container_env.items()},
                              command_index=command_index)
        command = selected_command(program)
        inner = list(command.argv)
        inner_words = list(map_shell_words(
            command.words, script, script_coords, source))
        inner_records = list(map_shell_records(
            command.env_occurrences, script, script_coords, source))
        container_env = {**container_env, **{
            name: (value, is_secret_name(name)) for name, value in command.env}}
        records = records + inner_records
        if secret_sink is not None:
            secret_sink.update(program.secret_values)
            secret_sink.update(command.secrets)
            secret_sink.update(value for name, value in command.env
                               if is_secret_name(name) and value)
        if inner[:2] in (["sh", "-c"], ["bash", "-c"]):
            raise LaunchSyntaxError("nested_shell_wrapper_unsupported")
    ranks = set()
    for device in devices:
        if "davinci" not in device and "npu" not in device:
            continue
        rank = _device_rank(device)
        if rank is None:
            unidentified.append(device)
        else:
            ranks.add(rank)
    masked, has_mask = visibility_ranks(container_env, unidentified)
    if has_mask:
        selected = ranks & masked if ranks else masked
    else:
        selected = ranks
    visible = tuple(sorted(selected, key=int))
    return (tuple(inner), tuple(inner_words), container_env, tuple(operational),
            visible, image, tuple(records),
            tuple(dict.fromkeys(missing_env)),
            tuple(dict.fromkeys(missing_files)),
            tuple(dict.fromkeys(forwarded)),
            tuple(dict.fromkeys(unidentified)),
            bool(devices) or has_mask)


__all__ = ["DocumentInputError", "WrapperOccurrence", "discovery_wrapper",
           "docker", "iter_discovery_wrappers", "visibility_ranks"]
