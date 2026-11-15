"""Launch document adapters: Docker envelope, argv, JSON and optional YAML.

[Role]
  Extract one supported command from a declarative document and hand literal
  argv/env plus original word spans to `launch_parser`.  Recognize only the
  Docker `run` envelope, literal `-e/--env`, visibility and operational
  options, and one literal `sh -c` unwrap.

[Invariants]
  * No host env lookup, `--env-file` read, Docker execution or model import.
  * Container environment starts empty; `-e KEY=VAL` sets a literal and
    `-e KEY` is resolved only from the command's outer shell snapshot.
    `--env-file` requires supplied contents.  Outer exports never leak into a
    container implicitly.
  * Duplicate JSON/YAML keys, non-finite numbers, >16 levels, oversized
    text/token counts and an unselected multi-container document are errors.
    A duplicate-key diagnostic never echoes an untrusted key.
  * YAML uses `yaml.safe_load` only with the optional extra; anchors, aliases,
    merge keys and tags are rejected and a post-load depth check runs.
  * Structured `command` is the base argv and `args` are appended literal
    arguments; a top-level string list is one argv, a mapping list is a
    container document.  Container selection is a public parameter.
  * A structured string command is newline-normalized together with its exact
    character map before shell parsing; env records and words keep outer spans.
  * Unknown Docker options, a second nested `sh -c` wrapper, and a `--` form
    that loses the image reject explicitly.

[Timeouts & Locks]
  Bounded in-memory parsing; no locks, I/O or devices.
[Side-Effect Policy]
  No subprocess, file read, environment mutation or network.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Mapping

from .launch_coords import normalize_newlines
from .launch_docker import (
    DocumentInputError, WrapperOccurrence, docker, visibility_ranks,
)
from .launch_parser import (
    EnvAssignment, LaunchCommand, LaunchSyntaxError, MAX_INTEGER_BITS,
    MAX_TEXT_BYTES, MAX_TOKENS, SourceSpan, Word, is_secret_name, parse_argv,
    parse_shell, selected_command,
)
from .launch_spans import (
    SpanUnavailable, document_root, map_shell_records, map_shell_words,
    mapping_key_spans, mapping_value_coords, prescan_cli_secret_values,
    prescan_document_secrets, scalar_words, select_node,
)
from .launch_taint import duplicate_key_label
from .launch_yaml_numbers import _sexagesimal

_YAML_UNAVAILABLE = "yaml_parser_unavailable"
_STRING_TOKEN = None


@dataclass(frozen=True)
class LaunchDocument:
    """One selected launch process with its literal argv and environment."""
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    source: str
    input_format: str
    image: str | None = None
    visible_devices: tuple[str, ...] = ()
    operational: tuple[WrapperOccurrence, ...] = ()
    command_index: int = 0
    words: tuple[Word, ...] = ()
    env_occurrences: tuple[EnvAssignment, ...] = ()
    command_span: SourceSpan | None = None
    source_text: str = ""
    outer_env_occurrences: tuple[EnvAssignment, ...] = ()
    missing_env: tuple[str, ...] = ()
    missing_env_files: tuple[str, ...] = ()
    #: Container-visible NPU ranks: mounted container targets intersected with
    #: ``ASCEND_RT_VISIBLE_DEVICES`` when both exist.  A host path and its
    #: container target are one physical exposure, never two devices.
    forwarded_env: tuple[str, ...] = ()
    unidentified_devices: tuple[str, ...] = ()
    #: Explicit visibility declaration, independent of the resolved cardinality.
    device_visibility_declared: bool = False
    secret_values: tuple[str, ...] = ()
    #: Document-defined env name -> exact decoded ``char_coords``.  ``None``
    #: records a caller-supplied or file-sourced value with no document map.
    env_coords: tuple[tuple[str, tuple[tuple[int, ...], ...] | None], ...] = ()


def _check_text(text, source="<document>"):
    if not isinstance(text, str):
        raise DocumentInputError("document_not_text", "text_required")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES or len(text) > MAX_TEXT_BYTES:
        raise DocumentInputError("input_exceeds_8_mib", "input_too_large")


def _count_nodes(value, limit=MAX_TOKENS):
    """Structural JSON/YAML node count, excluding only the document root."""
    count = -1
    stack = [value]
    while stack:
        item = stack.pop()
        count += 1
        if count > limit:
            raise DocumentInputError("input_exceeds_100000_tokens", "input_too_large")
        if isinstance(item, Mapping):
            for key, child in item.items():
                stack.append(key)
                stack.append(child)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return count


def _check_depth_text(text):
    """A linear bracket-depth guard before the recursive JSON decoder."""
    depth, index, quote, escaped = 0, 0, None, False
    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in '"\'':
            quote = char
        elif char in "[{":
            depth += 1
            if depth > 16:
                raise DocumentInputError("json_nesting_exceeds_16", "nesting_too_deep")
        elif char in "]}":
            depth = max(0, depth - 1)
        index += 1


def _check_value(value, depth=0):
    if depth > 16:
        raise DocumentInputError("json_nesting_exceeds_16", "nesting_too_deep")
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise DocumentInputError("non_finite_json_number", "non_finite_number")
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > MAX_INTEGER_BITS:
            raise DocumentInputError("integer_too_large", "integer_exceeds_4096_bits")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_value(key, depth + 1)
            _check_value(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_value(item, depth + 1)


def _attach_prescan(exc, values):
    """Carry pre-scanned credentials and scrub the same diagnostic surface."""
    secrets = tuple(dict.fromkeys((*getattr(exc, "secrets", ()), *values)))
    message = exc.message
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    return replace(exc, message=message, secrets=secrets)


def finite_json_loads(text, source="<json>", prescanned=None):
    """Strict JSON: duplicate keys, non-finite numbers and deep nesting fail."""
    _check_text(text, source)
    if prescanned is None:
        prescanned = prescan_document_secrets(text, "json")
    try:
        return _finite_json_loads_core(text, source)
    except DocumentInputError as exc:
        raise _attach_prescan(exc, prescanned) from exc


def _finite_json_loads_core(text, source="<json>"):
    _check_depth_text(text)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DocumentInputError(
                    "duplicate_json_key",
                    f"duplicate_key:{duplicate_key_label(key)}")
            result[key] = value
        return result

    def constant(value):
        raise DocumentInputError("non_finite_json_number", f"constant:{value}")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as exc:
        raise DocumentInputError("invalid_json",
                                 f"{source}:{exc.lineno}:{exc.colno}") from exc
    except RecursionError as exc:
        raise DocumentInputError("json_nesting_exceeds_16", "nesting_too_deep") from exc
    except DocumentInputError:
        raise
    except ValueError as exc:
        raise DocumentInputError("integer_too_large",
                                 "integer_exceeds_platform_digit_limit") from exc
    _check_value(value)
    _count_nodes(value)
    return value


def _yaml_load(text, source="<yaml>", prescanned=None):
    _check_text(text, source)
    if prescanned is None:
        prescanned = prescan_document_secrets(text, "yaml")
    try:
        return _yaml_load_core(text, source)
    except DocumentInputError as exc:
        raise _attach_prescan(exc, prescanned) from exc


def _yaml_load_core(text, source="<yaml>"):
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DocumentInputError(_YAML_UNAVAILABLE,
                                 "optional_launch_yaml_extra_missing") from exc

    class StrictLoader(yaml.SafeLoader):
        def construct_yaml_float(self, node):
            value = self.construct_scalar(node).replace("_", "").lower()
            if ":" not in value:
                return super().construct_yaml_float(node)
            sign = -1 if value.startswith("-") else 1
            number = sign * _sexagesimal(value.lstrip("+-"), float)
            _check_value(number)
            return number

        def compose_node(self, parent, index):
            event = self.peek_event()
            if isinstance(event, yaml.events.AliasEvent):
                raise DocumentInputError("yaml_anchor_or_alias",
                                         "aliases_are_not_static")
            if getattr(event, "anchor", None):
                raise DocumentInputError("yaml_anchor_or_alias",
                                         "anchors_are_not_static")
            if getattr(event, "tag", None):
                raise DocumentInputError("yaml_explicit_tag",
                                         "tags_are_not_static")
            return super().compose_node(parent, index)

        def construct_mapping(self, node, deep=False):
            if any(getattr(key, "tag", "") == "tag:yaml.org,2002:merge"
                   for key, _ in node.value):
                raise DocumentInputError("yaml_merge_key",
                                         "merge_keys_are_not_static")
            keys = [self.construct_object(key, deep=True) for key, _ in node.value]
            try:
                distinct = len(set(keys))
            except TypeError as exc:
                raise DocumentInputError(
                    "yaml_unhashable_key",
                    "mapping_key_requires_hashable_scalar") from exc
            if len(keys) != distinct:
                seen, duplicate = set(), None
                for key in keys:
                    if key in seen:
                        duplicate = key
                        break
                    seen.add(key)
                raise DocumentInputError(
                    "yaml_duplicate_key",
                    f"duplicate_key:{duplicate_key_label(duplicate)}")
            return super().construct_mapping(node, deep=deep)

    StrictLoader.add_constructor("tag:yaml.org,2002:float",
                                 StrictLoader.construct_yaml_float)
    try:
        value = yaml.load(text, Loader=StrictLoader)
    except TimeoutError:
        raise
    except DocumentInputError:
        raise
    except RecursionError as exc:
        raise DocumentInputError("yaml_nesting_exceeds_16", "nesting_too_deep") from exc
    except OverflowError as exc:
        raise DocumentInputError("non_finite_json_number", "non_finite_number") from exc
    except yaml.YAMLError as exc:
        raise DocumentInputError("invalid_yaml",
                                 str(exc).splitlines()[0][:200]) from exc
    except TypeError as exc:
        raise DocumentInputError("yaml_unhashable_key",
                                 "mapping_key_requires_hashable_scalar") from exc
    except ValueError as exc:
        raise DocumentInputError("integer_too_large",
                                 "integer_exceeds_platform_digit_limit") from exc
    _check_value(value)
    _count_nodes(value)
    return value


def _structured(payload, *, source, text, format="json", container=None,
                prescanned=(), outer_env=None):
    """Read the selected container's own command/args/env AST nodes.

    Selection is by decoded payload index, then every span comes from that
    node's structural children.  Field order, an equal decoy string elsewhere
    and an earlier container therefore cannot move the attribution cursor.
    The document credential inventory is supplied by the caller and is never
    recomputed here.
    """
    secrets = set(prescanned)

    def add_secret(value):
        if value:
            secrets.add(str(value))

    try:
        root = document_root(text, format)
        selected, index = payload, 0
        if isinstance(payload, list):
            if not payload:
                raise DocumentInputError("structured_document_invalid", "empty_list")
            if all(isinstance(item, str) for item in payload):
                return (list(payload), {}, "argv",
                        scalar_words(root, text, source), (),
                        tuple(sorted(secrets)), ())
            selected, index = _select(payload, container)
        elif isinstance(payload, Mapping) and "containers" in payload:
            items = payload.get("containers")
            if not isinstance(items, list) or not items:
                raise DocumentInputError("containers_invalid",
                                         "container_list_required")
            selected, index = _select(items, container)
        if not isinstance(selected, Mapping):
            raise DocumentInputError("structured_document_invalid",
                                     "mapping_required")
        node = select_node(root, payload, index)
        env_node = node.member("env") if node.kind == "object" else None
        env_spans = mapping_key_spans(env_node, source)
        env_coords = mapping_value_coords(env_node)
        raw_env = selected.get("env") or {}
        if not isinstance(raw_env, Mapping) or not all(
                isinstance(value, (str, int, float, bool))
                for value in raw_env.values()):
            raise DocumentInputError("structured_env_invalid",
                                     "literal_env_mapping_required")
        env, records = {}, []
        outer = {str(name): str(value)
                 for name, value in dict(outer_env or {}).items()}
        for key, value in raw_env.items():
            name = str(key)
            if name in outer:
                # A caller-supplied external value has no document source
                # coordinates and supersedes the document declaration.
                text_value = outer[name]
                env_coords.pop(name, None)
            else:
                text_value = (("true" if value else "false")
                              if isinstance(value, bool) else str(value))
            env[name] = (text_value, is_secret_name(name))
            if is_secret_name(name):
                add_secret(text_value)
            span = env_spans.get(name) or SourceSpan(source, 1, 1, 1, 1)
            records.append(EnvAssignment(name, text_value, span, len(records) + 1,
                                         False, True, is_secret_name(name)))
        command, args = selected.get("command"), selected.get("args")
        if args is not None and (not isinstance(args, list)
                                 or not all(isinstance(item, str) for item in args)):
            raise DocumentInputError("structured_args_invalid",
                                     "argv_list_required")
        command_node, args_node = node.member("command"), node.member("args")
        if isinstance(command, str):
            coords = (command_node.char_coords
                      if command_node is not None and command_node.kind == "scalar"
                      else ())
            command_text, command_coords = normalize_newlines(command, coords)
            shell_env = {**dict(outer_env or {}),
                         **{k: v[0] for k, v in env.items()}}
            program = parse_shell(command_text, source, env=shell_env,
                                  validate=False)
            secrets.update(program.secret_values)
            inner = selected_command(program)
            argv = list(inner.argv) + list(args or ())
            words = list(map_shell_words(
                inner.words, command_text, command_coords, source))
            if args:
                words.extend(scalar_words(args_node, text, source))
            secrets.update(value for name, value in inner.env
                           if is_secret_name(name))
            scoped = {**env, **dict(inner.env)}
            inner_records = map_shell_records(
                inner.env_occurrences, command_text, command_coords, source)
            all_env_coords = dict(env_coords)
            for record in inner_records:
                # A bare ``export NAME`` only changes visibility: keep the
                # document or earlier literal map for that exact name.  A
                # visibility occurrence may itself carry the earlier literal
                # value's map (``SCRIPT=...`` followed by ``export SCRIPT``);
                # install it when present.  An actual assignment (including
                # ``export NAME=value``) is a new value origin, so its own
                # map -- or ``None`` when unavailable -- replaces any older
                # provenance.
                if record.has_value:
                    all_env_coords[record.name] = record.char_coords or None
                elif record.char_coords:
                    all_env_coords[record.name] = record.char_coords
            return (argv, scoped, "shell", tuple(words),
                    tuple(records) + tuple(inner_records),
                    tuple(sorted(secrets)),
                    tuple(sorted(all_env_coords.items())))
        if isinstance(command, list):
            if not all(isinstance(item, str) for item in command):
                raise DocumentInputError("structured_command_invalid",
                                         "argv_list_required")
            argv = list(command) + list(args or ())
            words = list(scalar_words(command_node, text, source))
            if args:
                words.extend(scalar_words(args_node, text, source))
            return (argv, env, "argv", tuple(words), tuple(records),
                    tuple(sorted(secrets)), tuple(sorted(env_coords.items())))
        if command is None and args is not None:
            return (list(args), env, "argv", scalar_words(args_node, text, source),
                    tuple(records), tuple(sorted(secrets)),
                    tuple(sorted(env_coords.items())))
        raise DocumentInputError("structured_command_required",
                                 "command_or_args_required")
    except DocumentInputError as exc:
        raise replace(exc, secrets=tuple(dict.fromkeys(
            (*getattr(exc, "secrets", ()), *sorted(secrets))))) from exc
    except SpanUnavailable as exc:
        raise DocumentInputError(
            "source_coordinates_unavailable",
            f"exact_source_coordinates_required:{str(exc)[:120]}",
            secrets=tuple(sorted(secrets))) from exc
    except LaunchSyntaxError as exc:
        exc.secrets = tuple(dict.fromkeys(
            (*getattr(exc, "secrets", ()), *sorted(secrets))))
        raise


def _select(items, container):
    if not items or not all(isinstance(item, Mapping) for item in items):
        raise DocumentInputError("containers_invalid", "container_mappings_required")
    if container is None:
        if len(items) != 1:
            raise DocumentInputError("container_selection_required",
                                     "select_one_container_explicitly")
        return items[0], 0
    for index, item in enumerate(items):
        if str(item.get("name", "")) == str(container):
            return item, index
    # Additive index selection for unnamed container lists: an explicit
    # integer entry (or digit-only selector that matched no name) addresses
    # the container by its declared position.
    if isinstance(container, int) and not isinstance(container, bool):
        if 0 <= container < len(items):
            return items[container], container
    elif isinstance(container, str) and container.isdigit():
        position = int(container)
        if 0 <= position < len(items):
            return items[position], position
    raise DocumentInputError("container_not_found", f"container:{container}")


def parse_document(text, *, format="shell", source="<launch>", env=None,
                   container=None, command_index=None, env_files=None):
    """Parse one shell/argv/JSON/YAML document into a selected launch process."""
    explicit = {str(key): (str(value), is_secret_name(str(key)))
                for key, value in dict(env or {}).items()}
    collected = {value for name, (value, secret) in explicit.items()
                 if secret and value}
    for mapping in (env or {}, *(env_files or {}).values()):
        if isinstance(mapping, Mapping):
            for name, value in mapping.items():
                collected.update(prescan_cli_secret_values(str(name), str(value)))
    # One document credential prescan per public parse entry.  Loader,
    # admission, container selection and diagnostics share this inventory.
    inventory_format = (format if format in {"json", "yaml"}
                        else "json" if format == "argv" else "shell")
    inventory = (prescan_document_secrets(text, inventory_format)
                 if isinstance(text, str) else ())
    collected.update(inventory)
    if format == "argv" and isinstance(text, (list, tuple)):
        from .launch_taint import _shell_taint
        _shell_taint(text, collected, explicit)
    requirements = set()
    try:
        return _parse_document(
            text, format=format, source=source, explicit=explicit,
            container=container, command_index=command_index,
            env_files=env_files, collected=collected,
            requirements=requirements, prescanned=inventory)
    except DocumentInputError as exc:
        secrets = tuple(sorted(collected | set(getattr(exc, "secrets", ()))))
        required = tuple(sorted(requirements | set(
            getattr(exc, "required_inputs", ()))))
        message = exc.message
        for secret in sorted((item for item in secrets if item),
                             key=len, reverse=True):
            message = message.replace(secret, "[REDACTED]")
        raise replace(exc, message=message, secrets=secrets,
                      required_inputs=required) from exc
    except LaunchSyntaxError as exc:
        exc.secrets = tuple(sorted(
            collected | set(getattr(exc, "secrets", ()))))
        exc.required_inputs = tuple(sorted(requirements | set(
            getattr(exc, "required_inputs", ()))))
        for secret in sorted((item for item in exc.secrets if item),
                             key=len, reverse=True):
            exc.message = exc.message.replace(secret, "[REDACTED]")
        exc.args = (f"{exc.code}:{exc.message}",)
        raise


def _parse_document(text, *, format, source, explicit, container, command_index,
                    env_files, collected, requirements, prescanned=()):
    def collect(records):
        for record in records or ():
            if (record.secret or is_secret_name(record.name)) and record.value:
                collected.add(record.value)

    image, operational, visible, records = None, (), (), ()
    outer_records, missing_env, missing_files = (), (), ()
    forwarded, unidentified = (), ()
    visibility_declared = False
    env_coords = {}
    if format == "shell":
        _check_text(text, source)
        program = parse_shell(text, source, env={k: v[0] for k, v in explicit.items()},
                              validate=False, command_index=command_index)
        command = selected_command(program)
        argv, merged, words = list(command.argv), dict(command.env), list(command.words)
        records, outer_records = command.env_occurrences, command.env_occurrences
        env_coords = {record.name: (record.char_coords or None)
                      for record in command.env_occurrences}
        for name in explicit:
            env_coords.setdefault(name, None)
        collect(records)
        collected.update(program.secret_values)
        collected.update(value for name, value in command.env
                         if is_secret_name(name) and value)
        operational, source_text = tuple(WrapperOccurrence(
            item, command.span) for item in program.operational), program.source_text
    elif format == "argv":
        if isinstance(text, str):
            payload = finite_json_loads(text, source, prescanned=prescanned)
        else:
            payload = text
        if not isinstance(payload, list):
            raise DocumentInputError("invalid_argv", "argv_list_required")
        program = parse_argv(payload, source, validate=False)
        command = program.commands[0]
        argv, merged, words = list(command.argv), dict(explicit), list(command.words)
        env_coords = {name: None for name in explicit}
        operational, source_text = (), ""
    elif format in {"json", "yaml"}:
        payload = (finite_json_loads(text, source, prescanned=prescanned)
                   if format == "json"
                   else _yaml_load(text, source, prescanned=prescanned))
        (argv, structured_env, kind, words, records, structured_secrets,
         structured_env_coords) = _structured(
            payload, source=source, text=text, format=format, container=container,
            prescanned=prescanned,
            outer_env={name: value for name, (value, _secret)
                       in explicit.items()})
        collected.update(structured_secrets)
        collect(records)
        merged = dict(structured_env)
        for name, pair in explicit.items():
            merged.setdefault(name, pair)
        env_coords = {name: None for name in explicit}
        env_coords.update(structured_env_coords)
        operational = (WrapperOccurrence(f"structured_format={kind}",
                                         SourceSpan(source, 1, 1, 1, 1)),)
        if format == "yaml":
            operational += (WrapperOccurrence("yaml_safe_load",
                                              SourceSpan(source, 1, 1, 1, 1)),)
        source_text = text
    else:
        raise LaunchSyntaxError(f"launch_format_unsupported:{format}")
    if argv and argv[0] == "docker":
        (argv, words, merged, docker_ops, visible, image,
         records, missing_env, missing_files, forwarded,
         unidentified, visibility_declared) = docker(
            argv, words, merged, source=source, command_index=command_index,
            env_files=env_files, secret_sink=collected,
            requirement_sink=requirements, env_coords=env_coords)
        collect(records)
        operational = tuple(operational) + tuple(docker_ops)
        if argv and argv[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}:
            raise LaunchSyntaxError("nested_shell_wrapper_unsupported")
    else:
        extra = []
        masked, has_mask = visibility_ranks(merged, extra)
        if has_mask:
            visible = tuple(sorted(masked, key=int))
            visibility_declared = True
        if extra:
            unidentified = tuple(dict.fromkeys((*unidentified, *extra)))
    for index, token in enumerate(argv):
        name, equals, inline = token.partition("=")
        raw = inline if equals else (
            argv[index + 1] if index + 1 < len(argv)
            and not argv[index + 1].startswith("-") else "")
        collected.update(prescan_cli_secret_values(name, raw))
    from .launch_parser import _validate_command
    plain_env = {str(name): (value[0] if isinstance(value, (tuple, list)) else value)
                 for name, value in merged.items()}
    try:
        _validate_command(list(argv))
    except LaunchSyntaxError as exc:
        exc.secrets = tuple(sorted(collected | set(getattr(exc, "secrets", ()))))
        exc.required_inputs = tuple(sorted(requirements | set(
            getattr(exc, "required_inputs", ()))))
        raise
    command_span = words[0].span if words else None
    return LaunchDocument(
        tuple(argv), tuple(sorted(plain_env.items())), source, format,
        image, visible, tuple(operational), command_index or 0, tuple(words),
        tuple(records), command_span, source_text, tuple(outer_records),
        missing_env, missing_files, tuple(forwarded), tuple(unidentified),
        visibility_declared, tuple(sorted(collected)),
        env_coords=tuple(sorted(env_coords.items())))


__all__ = ["DocumentInputError", "LaunchDocument", "WrapperOccurrence",
           "finite_json_loads", "parse_document"]
