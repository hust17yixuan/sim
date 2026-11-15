"""Linear raw YAML credential scanning and loader-compatible scalar folding.

[Role]
  Provide the total, PyYAML-independent document scan used when the optional
  loader is absent or the document is malformed.  It discovers decoded scalar
  values, literal/folded block values, flow collections and composed
  ``command``/``args`` (or ``entrypoint``/``cmd``) token sequences before any
  admission error can build a diagnostic.

[Invariants]
  * One character cursor; whitespace and YAML break runs are consumed once, so
    work stays linear in document characters.
  * CRLF, CR, LF and U+0085 normalize to LF; U+2028/U+2029 stay themselves.
  * Block scalars are decoded by the same exact stream used for coordinates:
    more-indented whitespace is content, blank lines are only those at or below
    the content indent, and strip/clip/keep follow the final content break.
  * Sibling command/args lists in one mapping are concatenated before shell
    extraction; composition runs only when a configured field changes.
  * Every block mapping owns a fresh env/field frame: sibling mappings at one
    indentation column stay isolated, and a late env rescans only that frame's
    own fields.  A sibling-field surface resolves static ``$NAME``/``${NAME}``
    from the same frame's literal env.  Each parsed env scalar is installed
    already normalized, so no growing prefix is re-normalized.  No host env
    is read and no shell is executed.
  * Every plain numeric scalar contributes the YAML 1.1 loader spelling
    (binary, octal, hex, sexagesimal and decimal integers; YAML float forms),
    and every plain boolean contributes the adapter's ``true``/``false``.
[Timeouts & Locks]
  Linear in document characters; no locks, I/O, devices or env lookup.
[Side-Effect Policy]
  Pure functions over text; never mutates the document, files or globals.
"""
from __future__ import annotations

from .launch_env import EnvMap, ShellText, normalize_env
from .launch_parser import is_secret_name
from .launch_yaml_blocks import (
    block_scalar, indent_at, line_end, next_line, plain_line_span,
    plain_run_end, plain_spaces,
)
from .launch_yaml_coords import (
    _fold_chars, _folds, is_break, yaml_break, yaml_double_value,
    yaml_single_value,
)
from .launch_yaml_numbers import bool_spelling, numeric_spelling

_FLOW_STOP = ",}]"
_STRUCTURAL = frozenset({"|", ">", "{", "[", "]", "}", ""})
_COMMAND_PAIRS = (("command", "args"), ("entrypoint", "cmd"),
                  ("command", "cmd"), ("entrypoint", "args"))
_FIELD_KEYS = frozenset(key for pair in _COMMAND_PAIRS for key in pair)


def taintable(value):
    text = str(value or "")
    return bool(text) and text not in _STRUCTURAL


def add(found, value):
    if taintable(value):
        found.add(str(value))


def add_folded(found, value):
    """Taint a folded scalar and its constituent tokens (conservative)."""
    add(found, value)
    if isinstance(value, str):
        for part in value.replace("\n", " ").split():
            add(found, part)


def env_text(value, plain=True):
    """The adapter's literal environment spelling for one decoded scalar."""
    text = str(value)
    if plain:
        spelled = bool_spelling(text) or numeric_spelling(text)
        if spelled:
            return spelled
    return text


def shell(scan, value, found, env=None):
    """Apply the shared shell/argv credential rules to one decoded surface."""
    if scan is None or not isinstance(value, (str, tuple, list)) or not value:
        return
    if env:
        try:
            scan(value, found, env)
            return
        except TypeError:  # compatibility with a two-argument test hook
            pass
    scan(value, found)


def compose_fields(fields, found, shell_scan=None, env=None):
    """Concatenate sibling command/args token sequences in one mapping."""
    for left, right in _COMMAND_PAIRS:
        first, second = fields.get(left), fields.get(right)
        if first and second:
            tokens = (*first, *second)
            # Scalar command heads retain their type; argv stays delimited.
            shell(shell_scan, tuple(tokens), found, env)


def add_scalar(found, value, plain):
    """Add a plain scalar and the loader-normalized scalar it resolves to."""
    add(found, value)
    if plain:
        spelling = numeric_spelling(value)
        if spelling:
            add(found, spelling)
        boolean = bool_spelling(value)
        if boolean:
            add(found, boolean)


def _skip_space(text, index):
    size = len(text)
    while index < size and text[index] in " \t":
        index += 1
    return index


def _quoted_token(text, index):
    """Return ``(decoded, raw_inner, next_index)`` for one YAML quoted token."""
    quote, size = text[index], len(text)
    cursor = index + 1
    while cursor < size:
        char = text[cursor]
        if quote == '"' and char == "\\":
            cursor += 2
            continue
        if quote == "'" and char == "'":
            if cursor + 1 < size and text[cursor + 1] == "'":
                cursor += 2
                continue
            cursor += 1
            break
        if char == quote:
            cursor += 1
            break
        cursor += 1
    raw = (text[index + 1:cursor - 1]
           if cursor <= size and text[cursor - 1:cursor] == quote
           else text[index + 1:cursor])
    decoded = yaml_double_value(raw) if quote == '"' else yaml_single_value(raw)
    return decoded, raw, cursor


def _plain_token(text, index, key):
    """Consume one plain YAML scalar; keys stop at ``:``, values at a break."""
    size, cursor = len(text), index
    while cursor < size:
        char = text[cursor]
        if is_break(char):
            break
        if key and char == ":":
            break
        if char == "#" and (cursor == index or text[cursor - 1] in " \t"):
            break
        if not key and char in _FLOW_STOP:
            break
        cursor += 1
    return text[index:cursor].strip(), cursor


def _plain_flow_token(text, index):
    """Decode one multiline plain scalar inside a flow collection."""
    size, cursor = len(text), index
    out = []
    while cursor < size:
        char = text[cursor]
        if char in _FLOW_STOP:
            break
        if char == ":" and (cursor + 1 >= size
                            or text[cursor + 1] in " \t\r\n\x85\u2028\u2029,}]"):
            break
        if char == "#" and (cursor == index or text[cursor - 1] in " \t"):
            break
        if char in " \t" or char in "\r\n\x85\u2028\u2029":
            breaks, run_end = _folds(text, cursor, size)
            if breaks is None:
                out.append(text[cursor:run_end])
            else:
                out.extend(item for item, _first, _last in _fold_chars(breaks))
            cursor = run_end
            continue
        out.append(char)
        cursor += 1
    return "".join(out).rstrip(" \t"), cursor


def _scan_flow(text, index, inherited, found, shell_scan=None, values=None,
               pairs=None, env=None):
    """Single-pass flow mapping/sequence scan; returns the index after close."""
    size = len(text)
    stack = []

    def push(kind, secret, role="value", owner_key=None, frame_env=None,
             is_env=False):
        stack.append({"kind": kind, "secret": bool(secret), "role": role,
                      "expect_key": kind == "map", "key_secret": False,
                      "key": None, "scalars": [], "fields": {}, "pairs": {},
                      "dirty": False, "owner_key": owner_key, "is_env": is_env,
                      "env": frame_env if frame_env is not None else EnvMap()})

    def drain(frame):
        env_map = frame["env"] or None
        if frame["kind"] == "seq":
            if frame["scalars"]:
                shell(shell_scan, tuple(frame["scalars"]), found, env_map)
        elif frame["dirty"]:
            for key in _FIELD_KEYS:
                tokens = frame["fields"].get(key)
                if tokens:
                    shell(shell_scan, tuple(tokens), found, env_map)
            compose_fields(frame["fields"], found, shell_scan, env_map)
            frame["dirty"] = False

    def close(frame):
        """Install an env value, attach a child, then finalize once."""
        if frame["is_env"] and stack and stack[-1]["kind"] == "map":
            stack[-1]["env"].update(normalize_env(frame["pairs"]))
        drain(frame)
        owner = frame.get("owner_key")
        if (owner is not None and frame["kind"] == "seq"
                and stack and stack[-1]["kind"] == "map"):
            stack[-1]["fields"].setdefault(owner, []).extend(frame["scalars"])
            if owner in _FIELD_KEYS:
                stack[-1]["dirty"] = True
        if not stack:
            if values is not None and frame["kind"] == "seq":
                values.extend(frame["scalars"])
            if pairs is not None and frame["kind"] == "map":
                pairs.update(frame["pairs"])

    root = "map" if text[index] == "{" else "seq"
    push(root, inherited,
         frame_env=(normalize_env(env) if root == "seq" else EnvMap()))
    cursor = index + 1
    while cursor < size and stack:
        char = text[cursor]
        if char in " \t" or is_break(char):
            cursor += 1
            continue
        if char == "#":
            cursor = line_end(text, cursor)
            continue
        frame = stack[-1]
        if char in "}]":
            expected = "}" if frame["kind"] == "map" else "]"
            if char == expected:
                close(stack.pop())
                if stack and stack[-1]["kind"] == "map" \
                        and not stack[-1]["expect_key"]:
                    stack[-1]["expect_key"] = True
                    stack[-1]["key"] = None
            cursor += 1
            continue
        if char == ",":
            if frame["kind"] == "map" and not frame["expect_key"]:
                frame["expect_key"] = True
                frame["key"] = None
            cursor += 1
            continue
        if frame["kind"] == "map" and frame["expect_key"]:
            if char in "{[":
                push("map" if char == "{" else "seq", frame["secret"], "key",
                     frame_env=(frame["env"] if char == "[" else None))
                cursor += 1
                continue
            if char in "'\"":
                decoded, _raw, end = _quoted_token(text, cursor)
            else:
                decoded, end = _plain_token(text, cursor, True)
            shell(shell_scan, decoded, found, frame["env"] or None)
            frame["key_secret"] = bool(frame["secret"] or is_secret_name(decoded))
            frame["key"] = decoded
            frame["expect_key"] = False
            cursor = _skip_space(text, end)
            if cursor < size and text[cursor] == ":":
                cursor += 1
            continue
        if char in "{[":
            secret = frame["secret"] or frame.get("key_secret", False)
            owner = frame["key"] if frame["kind"] == "map" else None
            is_env = bool(char == "{" and frame["kind"] == "map"
                          and frame["key"] == "env"
                          and not frame["expect_key"])
            push("map" if char == "{" else "seq", secret, owner_key=owner,
                 frame_env=(frame["env"] if char == "[" else None),
                 is_env=is_env)
            cursor += 1
            continue
        if char in "'\"":
            decoded, raw, end = _quoted_token(text, cursor)
            plain = False
        else:
            decoded, end = _plain_flow_token(text, cursor)
            raw, plain = decoded, True
        shell(shell_scan, (decoded,) if frame["kind"] == "seq" else decoded,
              found, frame["env"] or None)
        secret = frame["secret"] or frame.get("key_secret", False)
        if secret:
            add_folded(found, decoded)
            add_scalar(found, decoded, plain)
            if raw != decoded:
                add(found, raw)
        if frame["kind"] == "seq":
            frame["scalars"].append(decoded)
        else:
            if frame["key"] is not None:
                frame["fields"].setdefault(frame["key"], []).append(ShellText(decoded))
                frame["pairs"][frame["key"]] = env_text(decoded, plain)
                if frame["key"] in _FIELD_KEYS:
                    frame["dirty"] = True
            frame["expect_key"] = True
            frame["key"] = None
        cursor = end if end > cursor else cursor + 1
    while stack:
        close(stack.pop())
    return cursor


def _block_taint(value, lines, secret, found, shell_scan=None, env=None, delimited=False):
    """Taint one block scalar's exact and conservative source spellings."""
    shell(shell_scan, (value,) if delimited else value, found, env)
    # Remove only YAML indentation; content spaces and blank rows are semantic.
    raw = "\n".join(lines)
    shell(shell_scan, (raw,) if delimited else raw, found, env)
    if secret:
        stripped = [line.strip() for line in lines if line.strip()]
        for line in stripped:
            add(found, line)
        found.add(value)
        add(found, " ".join(part for part in stripped if part))


def _scan_mapping_value(text, cursor, key_indent, key_secret, found,
                        shell_scan=None, env=None, pairs=None):
    """Parse one block mapping value after the colon; return ``(end, values)``."""
    size = len(text)
    if cursor >= size or text[cursor] in "\r\n\x85\u2028\u2029#":
        return None
    char = text[cursor]
    if char in "{[":
        values = []
        end = _scan_flow(text, cursor, key_secret, found, shell_scan, values,
                         pairs, env)
        return end, values
    if char in "|>":
        value, end, stripped = block_scalar(text, cursor, key_indent)
        _block_taint(value, stripped, key_secret, found, shell_scan, env)
        return end, [ShellText(value)]
    if char in "'\"":
        decoded, raw, end = _quoted_token(text, cursor)
        shell(shell_scan, decoded, found, env)
        if key_secret:
            add(found, decoded)
            if raw != decoded:
                add(found, raw)
        return end, [ShellText(decoded)]
    decoded, end = _plain_token(text, cursor, False)
    shell(shell_scan, decoded, found, env)
    chunks, probe = [decoded], end
    while probe < size:
        chunk, probe, column = plain_spaces(text, probe)
        if not chunk:
            break
        if probe >= size or text[probe] == "#" or column < key_indent + 1:
            break
        line, probe = plain_line_span(text, probe)
        if not line:
            break
        chunks.append(chunk)
        chunks.append(line)
    if len(chunks) > 1:
        joined = "".join(chunks)
        shell(shell_scan, joined, found, env)
        if key_secret:
            add_folded(found, joined)
            add_scalar(found, joined, True)
        return end, [ShellText(joined)]
    elif key_secret:
        add_scalar(found, decoded, True)
    return end, [ShellText(decoded)]


def _sequence_value(text, position, stop, indent):
    """Decode one sequence item's scalar value for command composition."""
    if text[position:position + 1] in "'\"":
        decoded, _raw, _end = _quoted_token(text, position)
        return decoded
    if text[position:position + 1] in "|>":
        value, _end, _stripped = block_scalar(text, position, indent)
        return value
    size = len(text)
    chunks, probe = [text[position:stop].strip()], stop
    while probe < size:
        chunk, probe, column = plain_spaces(text, probe)
        if not chunk:
            break
        if probe >= size or text[probe] == "#" or column < indent + 1:
            break
        line, probe = plain_line_span(text, probe)
        if not line:
            break
        chunks.append(chunk)
        chunks.append(line)
    return "".join(chunks)


def _key_colon(text, probe):
    if probe >= len(text) or text[probe] != ":":
        return False
    return probe + 1 >= len(text) \
        or text[probe + 1] in " \t\r\n\x85\u2028\u2029"


def _item_is_key(text, position):
    """True when a sequence item begins a mapping (a new container item)."""
    if position >= len(text):
        return False
    if text[position] == "{":
        return True
    if text[position] in "[":
        return False
    if text[position] in "'\"":
        _decoded, _raw, end = _quoted_token(text, position)
        return _key_colon(text, _skip_space(text, end))
    plain, end = _plain_token(text, position, True)
    return bool(plain) and _key_colon(text, _skip_space(text, end))


def _scan_raw(text, found, shell_scan=None):
    """One-pass block/flow scanner with per-mapping frame ownership."""
    size = len(text)
    scopes = []  # (key indent, inherited secret) for following child blocks
    frames = []  # open block mapping frames, innermost last
    pending = None  # [owning frame, key, tokens, key column] while it grows
    sequence_values = []
    env_scope = None  # [env key column, EnvMap, owning frame]
    cursor = 0

    def rescan(frame):
        """Re-resolve only this frame's own fields after its late ``env``."""
        env = frame["env"] or None
        if not frame["fields"] or env is None:
            return
        for tokens in frame["fields"].values():
            shell(shell_scan, tuple(tokens), found, env)
        compose_fields(frame["fields"], found, shell_scan, env)

    def commit_env():
        nonlocal env_scope
        if env_scope is not None:
            _column, values, frame = env_scope
            frame["env"].update(normalize_env(values))
            env_scope = None
            rescan(frame)

    def flush_pending():
        nonlocal pending
        if pending is None:
            return
        frame, key, tokens = pending[0], pending[1], pending[2]
        pending = None
        if not tokens or key not in _FIELD_KEYS:
            return
        frame["fields"][key] = list(tokens)
        env = frame["env"] or None
        shell(shell_scan, tuple(tokens), found, env)
        compose_fields(frame["fields"], found, shell_scan, env)

    def close_frame(frame):
        if env_scope is not None and env_scope[2] is frame:
            commit_env()
        if pending is not None and pending[0] is frame:
            flush_pending()

    def close_deeper(column):
        while frames and frames[-1]["column"] > column:
            close_frame(frames.pop())

    def frame_at(column):
        close_deeper(column)
        if not frames or frames[-1]["column"] != column:
            frames.append({"column": column, "fields": {}, "env": EnvMap()})
        return frames[-1]

    def env_at(column):
        return next((frame["env"] or None for frame in reversed(frames)
                     if frame["column"] == column), None)

    while cursor < size:
        line_start = cursor
        indent, content = indent_at(text, cursor)
        if content >= size:
            break
        char = text[content]
        if is_break(char):
            cursor = next_line(text, content)
            continue
        if char == "#":
            newline = line_end(text, content)
            cursor = next_line(text, newline)
            continue
        if env_scope is not None and indent <= env_scope[0]:
            commit_env()
        sequence_line = text[content] == "-" and (
            content + 1 >= size or text[content + 1] in " \t\r\n\x85\u2028\u2029")
        if sequence_line:
            # Close the previous item's mapping frames before a new item; the
            # owning env must settle only within that item's own fields.
            if pending is not None and indent < pending[3]:
                flush_pending()
            close_deeper(indent)
        while scopes and (scopes[-1][0] > indent
                          or (scopes[-1][0] == indent and not sequence_line)):
            scopes.pop()
        inherited = bool(scopes and scopes[-1][1])
        position = content
        while position < size and text[position] == "-" and (
                position + 1 >= size
                or text[position + 1] in " \t\r\n\x85\u2028\u2029"):
            position = _skip_space(text, position + 1)
        if position >= size:
            cursor = size
            continue
        if is_break(text[position]):
            cursor = next_line(text, position)
            continue
        if text[position] == "#":
            newline = line_end(text, position)
            cursor = next_line(text, newline)
            continue
        block = None
        if text[position] in "|>":
            block = block_scalar(text, position, indent)
        if sequence_line:
            stop = line_end(text, position)
            token = block[0] if block is not None else _sequence_value(
                text, position, stop, indent)
            sequence_values.append(token if block is not None
                                   else text[position:stop].strip())
            item_is_key = _item_is_key(text, position)
            if pending is not None and (indent < pending[3] or (
                    indent == pending[3] and item_is_key)):
                flush_pending()
            if pending is not None and indent >= pending[3] and not item_is_key:
                if token:
                    pending[2].append(token)
            if item_is_key:
                item_column = position - line_start
                while frames and frames[-1]["column"] >= item_column:
                    close_frame(frames.pop())
        key_start = position
        key = None
        end = position
        if text[position] not in "{[":
            if text[position] in "'\"":
                key, _raw, end = _quoted_token(text, position)
                probe = _skip_space(text, end)
                if probe < size and text[probe] == ":":
                    end = probe
                else:
                    key = None
            else:
                plain, plain_end = _plain_token(text, position, True)
                probe = _skip_space(text, plain_end)
                if probe < size and text[probe] == ":" and plain:
                    key, end = plain, probe
        if key is not None:
            key_column = key_start - line_start
            if pending is not None and key_column <= pending[3]:
                flush_pending()
            frame = frame_at(key_column)
            env = frame["env"] or None
            if env_scope is not None and key_column > env_scope[0]:
                env = env_scope[1]
            shell(shell_scan, key, found, env)
            key_secret = bool(inherited or is_secret_name(key))
            cursor = _skip_space(text, end + 1)
            pairs = {} if key == "env" else None
            result = _scan_mapping_value(text, cursor, key_column, key_secret,
                                         found, shell_scan, env, pairs)
            if env_scope is not None and key_column > env_scope[0] \
                    and result is not None and result[1]:
                plain_value = text[cursor:cursor + 1] not in "'\"|>{["
                env_scope[1].update(normalize_env(
                    {key: env_text(result[1][0], plain_value)}))
            if result is None:
                if key in _FIELD_KEYS:
                    pending = [frame, key, [], key_column]
                if key == "env":
                    commit_env()
                    env_scope = [key_column, EnvMap(), frame]
                scopes.append((key_column, key_secret))
                newline = line_end(text, end)
                cursor = next_line(text, newline)
            else:
                result_end, values = result
                if key == "env" and pairs:
                    frame["env"] = normalize_env(pairs)
                    rescan(frame)
                if key in _FIELD_KEYS and values:
                    frame["fields"][key] = list(values)
                    shell(shell_scan, tuple(values), found, env)
                    compose_fields(frame["fields"], found, shell_scan, env)
                cursor = result_end
            continue
        if text[position] in "{[":
            extra = env_at(indent)
            cursor = _scan_flow(text, position, inherited, found, shell_scan,
                                None, None, extra)
        elif text[position] in "|>":
            if block is None:
                block = block_scalar(text, position, indent)
            value, value_end, stripped = block
            _block_taint(value, stripped, inherited, found, shell_scan,
                         env_at(indent), delimited=sequence_line)
            cursor = value_end
            continue
        elif text[position] in "'\"":
            decoded, raw, cursor = _quoted_token(text, position)
            shell(shell_scan, (decoded,) if sequence_line else decoded, found, env_at(indent))
            if inherited:
                add(found, decoded)
                if raw != decoded:
                    add(found, raw)
        else:
            decoded, cursor = _plain_token(text, position, False)
            shell(shell_scan, (decoded,) if sequence_line else decoded, found, env_at(indent))
            if cursor <= position:
                cursor = position + 1
            if inherited:
                add_scalar(found, decoded, True)
                if sequence_line:
                    continuation = []
                    probe = next_line(text, cursor)
                    while probe < size:
                        findent, fcontent = indent_at(text, probe)
                        fstop = line_end(text, probe)
                        fline = text[fcontent:fstop].strip()
                        if not fline or findent <= indent \
                                or fline.startswith("- ") or ":" in fline:
                            break
                        continuation.append(fline)
                        probe = next_line(text, fstop)
                    if continuation:
                        add_scalar(
                            found, " ".join([decoded, *continuation]), True)
        cursor = next_line(text, cursor)
    commit_env()
    flush_pending()
    while frames:
        close_frame(frames.pop())
    if sequence_values:
        shell(shell_scan, tuple(sequence_values), found)


def raw_document_taint(text, found, shell_scan=None):
    """Add every raw-scanner credential candidate to ``found``."""
    try:
        _scan_raw(text, found, shell_scan)
    except TimeoutError:
        raise
    except Exception:  # the prescan is total even for adversarial text
        return


__all__ = ["add", "add_folded", "add_scalar", "compose_fields", "env_text",
           "numeric_spelling", "raw_document_taint", "shell", "taintable"]
