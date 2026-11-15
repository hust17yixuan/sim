"""Linear decoded YAML credential discovery with inherited secret scope.

[Role]
  Discover decoded credential values from a raw YAML document before syntax,
  duplicate-key, anchor or size admission can reject it.  A PyYAML token/event
  walk covers decodable documents; a single-pass raw scanner covers malformed
  documents and PyYAML-absent environments.  Both modes carry an explicit
  container stack so a secret-named mapping key taints every descendant scalar
  leaf in block and flow mappings/sequences anywhere in the document, and both
  compose sibling ``command``/``args`` (or ``entrypoint``/``cmd``) tokens.

[Invariants]
  * Sensitivity comes from the decoded key name only; every non-empty string
    leaf under an inherited secret scope (digits included) is tainted, together
    with its raw spelling when one exists.
  * In PyYAML-enabled mode a plain scalar under a secret key also contributes
    the loader's own resolved spelling (octal, binary, hex and sexagesimal
    integers; exact YAML 1.1 float forms), so the prescan cannot disagree with
    ``yaml.safe_load``.
  * Both modes agree structurally: escaped double-quoted keys, plain flow
    scalars, nested flow/block containers, sequence items and unselected
    containers are all traversed.
  * Work is O(N): one character cursor, integer line/indent indices, no
    remaining-suffix list copies and no per-key rescan.
  * Malformed YAML errors fall back to the total raw scan; TimeoutError
    propagates instead of being treated as malformed input.
[Timeouts & Locks]
  Linear in document characters; no locks, I/O, devices or environment lookup.
[Side-Effect Policy]
  Pure functions over text; never mutates the document, files or globals.
"""
from __future__ import annotations

from .launch_env import EnvMap, ShellText, normalize_env
from .launch_parser import is_secret_name
from .launch_yaml_raw import (
    env_text, add, add_scalar, compose_fields, raw_document_taint, shell,
)

_NUMERIC_TAGS = frozenset({"tag:yaml.org,2002:int", "tag:yaml.org,2002:float"})
_EVENT_FIELD_KEYS = frozenset({"command", "args", "entrypoint", "cmd"})


def _loader_spelling(value):
    """PyYAML's own resolved spelling for one plain numeric scalar."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        loader = yaml.SafeLoader("")
        node_cls = yaml.nodes.ScalarNode
        tag = yaml.SafeLoader.resolve(loader, node_cls, value, (True, False))
        if tag not in _NUMERIC_TAGS:
            return None
        loaded = loader.construct_object(node_cls(tag, value))
        if isinstance(loaded, bool):
            return None
        text = str(loaded)
    except TimeoutError:
        raise
    except Exception:  # resolution is advisory taint only
        return None
    return text if text and text != value else None


def _event_taint(text, found, shell_scan=None):
    """Walk PyYAML's event stream with an explicit container/inheritance stack."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return
    stack = []

    def install_env(frame):
        if frame.get("is_env") and stack and stack[-1]["kind"] == "map":
            stack[-1]["env"].update(normalize_env(frame["pairs"]))

    def drain(frame):
        env = frame["env"] or None
        if frame["kind"] == "map":
            for surface in frame["surfaces"]:
                shell(shell_scan, surface, found, env)
            if frame["dirty"]:
                for key in _EVENT_FIELD_KEYS:
                    tokens = frame["fields"].get(key)
                    if tokens:
                        shell(shell_scan, tuple(tokens), found, env)
                compose_fields(frame["fields"], found, shell_scan, env)
                frame["dirty"] = False
        else:
            if frame["scalars"]:
                shell(shell_scan, tuple(frame["scalars"]), found, env)

    def close_frame(frame):
        """Attach an open child to its owner, then finalize exactly once."""
        if frame["kind"] == "map":
            install_env(frame)
            drain(frame)
        else:
            drain(frame)
            owner = frame.get("owner_key")
            if owner is not None and stack and stack[-1]["kind"] == "map":
                stack[-1]["fields"].setdefault(owner, []).extend(
                    frame["scalars"])
                if owner in _EVENT_FIELD_KEYS:
                    stack[-1]["dirty"] = True
        _finish_frame(stack, frame)

    try:
        events = yaml.parse(text, Loader=yaml.SafeLoader)
        for event in events:
            if isinstance(event, yaml.MappingStartEvent):
                secret, role = _start_scope(stack, mapping=True)
                parent = stack[-1] if stack else None
                is_env = bool(parent is not None and parent["kind"] == "map"
                              and parent.get("key") == "env"
                              and not parent["expect_key"])
                stack.append({"kind": "map", "secret": secret, "role": role,
                              "expect_key": True, "key_secret": False,
                              "key": None, "scalars": [], "fields": {},
                              "pairs": {}, "dirty": False, "is_env": is_env,
                              "env": EnvMap(), "surfaces": [],
                              "pending_name": None})
            elif isinstance(event, yaml.SequenceStartEvent):
                secret, role = _start_scope(stack, mapping=False)
                parent = stack[-1] if stack else None
                owner = (parent.get("key") if parent is not None
                         and parent["kind"] == "map" and not parent["expect_key"]
                         else None)
                stack.append({"kind": "seq", "secret": secret, "role": role,
                              "expect_key": False, "key_secret": False,
                              "key": None, "scalars": [], "fields": {},
                              "owner_key": owner, "surfaces": [],
                              "env": parent["env"] if parent is not None
                              else EnvMap(), "dirty": False})
            elif isinstance(event, yaml.MappingEndEvent):
                if not stack:
                    continue
                close_frame(stack.pop())
            elif isinstance(event, yaml.SequenceEndEvent):
                if not stack:
                    continue
                close_frame(stack.pop())
            elif isinstance(event, yaml.ScalarEvent):
                _event_scalar(stack, event.value, found,
                              event.style is None, shell_scan)
    except TimeoutError:
        raise
    except Exception:  # malformed YAML must not abort credential preservation
        pass
    finally:
        # A malformed tail must not hide credentials already decoded into an
        # open frame; unwind bottom-up through the same attach/finalize path.
        while stack:
            try:
                close_frame(stack.pop())
            except TimeoutError:
                raise
            except Exception:
                pass


def _start_scope(stack, mapping):
    if not stack:
        return False, "root"
    parent = stack[-1]
    if parent["kind"] == "seq":
        return bool(parent["secret"]), "value"
    if parent["expect_key"]:
        return bool(parent["secret"]), "key"
    return bool(parent["secret"] or parent["key_secret"]), "value"


def _finish_frame(stack, frame):
    if not stack or stack[-1]["kind"] != "map":
        return
    if frame.get("role") == "key":
        stack[-1]["expect_key"] = False
    else:
        stack[-1]["expect_key"] = True
        stack[-1]["key"] = None


def _event_scalar(stack, value, found, plain=True, shell_scan=None):
    if not stack:
        return
    text = value if isinstance(value, str) else str(value)
    parent = stack[-1]
    if parent["kind"] == "seq":
        if isinstance(value, str):
            parent["scalars"].append(value)
        parent["surfaces"].append(text)
        if parent["secret"]:
            add_scalar(found, value, plain)
            if plain:
                spelled = _loader_spelling(value)
                if spelled:
                    add(found, spelled)
        return
    parent["surfaces"].append(text)
    if parent["expect_key"]:
        parent["key_secret"] = bool(parent["secret"] or is_secret_name(text))
        parent["key"] = text
        parent["expect_key"] = False
        if parent.get("is_env"):
            parent["pending_name"] = text
        return
    if (parent["secret"] or parent["key_secret"]) and isinstance(value, str):
        add_scalar(found, value, plain)
        if plain:
            spelled = _loader_spelling(value)
            if spelled:
                add(found, spelled)
        if value == "":
            # The loader value of an empty block scalar still has to match the
            # raw scan's exact value set.
            found.add("")
    if parent.get("is_env") and parent["pending_name"] is not None:
        normalized = env_text(text, plain)
        parent["pairs"][parent["pending_name"]] = normalized
        parent["pending_name"] = None
    key = parent.get("key")
    if key is not None:
        parent["fields"].setdefault(key, []).append(ShellText(text))
        if key in _EVENT_FIELD_KEYS:
            parent["dirty"] = True
    parent["expect_key"] = True
    parent["key"] = None


def yaml_document_taint(text, found, shell_scan=None):
    """Add every credential candidate in raw YAML source to ``found``."""
    if not isinstance(text, str) or not text:
        return
    if shell_scan is None:
        _event_taint(text, found)
    else:
        try:
            _event_taint(text, found, shell_scan)
        except TypeError:
            # Compatibility with a two-argument test/instrumentation hook.
            _event_taint(text, found)
    raw_document_taint(text, found, shell_scan)


__all__ = ["yaml_document_taint"]
