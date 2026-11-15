"""Optional software-stack selectors for captured native operator sources.

[Role]
  One canonical place for the captured container/source identities
  (``r31``/``r35``/``r64``) and for the *optional* version gate an independent
  ACLNN caller passes.  A caller that declares a container SHA selects that
  software stack explicitly; a caller that declares none is a general shape
  and must not be rejected by an artificial image wall.

[Invariants]
  * An absent declaration is ``None``: it selects no stack and rejects no
    request.  Only an explicit declaration that contradicts the requested
    stack is a mismatch.
  * Digests are normalized once (``sha256:`` prefix, lowercase).  A version
    string is never interpreted as a duration, model name or shape.
  * The declared constants come from ``mx_source_provenance``; this module
    adds no new identity and no new measurement.

[Timeouts & Locks]
  Pure string work; no waits or locks.

[Side-Effect Policy]
  Read-only; returns normalized declarations.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.prediction.mx_source_provenance import (
    R31_IMAGE_SHA, R35_IMAGE_SHA, R64_IMAGE_SHA, image_digest,
    source_revision_from_oci,
)

#: Declared CANN release of the retained A5 captures.
CANN_VERSION = "cann-9.2.0"

#: Short prefixes used by the captured feature metadata.
R31_IMAGE_PREFIX = R31_IMAGE_SHA[:11]
R35_IMAGE_PREFIX = R35_IMAGE_SHA[:11]
R64_IMAGE_PREFIX = R64_IMAGE_SHA[:11]

STACK_IMAGES = {"r31": R31_IMAGE_SHA, "r35": R35_IMAGE_SHA,
                "r64": R64_IMAGE_SHA}

#: Revision markers that mean "no explicit stack declared".
UNPINNED_REVISIONS = frozenset({"", "unpinned", "unknown", "unresolved",
                                "not_retained", "none"})


def declared_version(version: Any) -> dict[str, str]:
    """Normalized declared version fields (absent fields stay absent)."""
    values: dict[str, str] = {}
    if isinstance(version, str):
        values["api"] = version
        return values
    if not isinstance(version, Mapping):
        return values
    for key in ("api", "api_version", "aclnn", "cann", "source",
                "source_revision", "revision", "image_id", "image",
                "deployment_image", "oci_image"):
        value = version.get(key)
        if value is not None:
            values[key] = str(value)
    return values


def _digest_match(digest: str, declared: str) -> bool:
    """A declared constant may be a full digest or its retained prefix."""
    if not digest or not declared:
        return False
    return digest == declared or digest.startswith(declared) \
        or declared.startswith(digest)


def declared_stack(values: Mapping[str, str]) -> str | None:
    """The stack an explicit version declaration selects, else ``None``.

    A declared source revision wins over a declared image digest.  An image
    that matches no declared stack returns ``"other"`` (an explicit
    contradiction); an empty declaration returns ``None``.
    """
    source = (values.get("source") or values.get("source_revision")
              or values.get("revision") or "")
    if source and source not in UNPINNED_REVISIONS:
        for name in STACK_IMAGES:
            if source == name or source.startswith(f"{name}/"):
                return name
        return "other"
    image = image_digest(values.get("image_id") or values.get("image") or "")
    if image:
        for name, digest in STACK_IMAGES.items():
            if _digest_match(image, digest):
                return name
        return "other"
    return None


def _stack_from_text(text: str) -> str | None:
    """Stack named by an image tag/revision text, else ``None``."""
    tokens = [token for token in str(text or "").lower()
              .replace(":", "-").replace("/", "-").replace("_", "-").split("-")
              if token]
    for token in tokens:
        if token in STACK_IMAGES:
            return token
    return None


def stack_of_metadata(meta: Mapping[str, Any] | None) -> str | None:
    """Stack selected by captured feature metadata, or ``None`` when absent.

    A named image tag and a matching digest select the same stack; naming one
    stack in the tag while pinning another stack's digest is an explicit
    contradiction (``"other"``), not a silent preference.
    """
    meta = dict(meta or {})
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    text_stack = _stack_from_text(meta.get("image")
                                  or meta.get("deployment_image")
                                  or meta.get("oci_image") or "")
    digest = image_digest(image_id)
    digest_stack = None
    for name, value in STACK_IMAGES.items():
        if digest and _digest_match(digest, value):
            digest_stack = name
            break
    if text_stack and digest_stack and text_stack != digest_stack:
        return "other"
    if digest_stack:
        return digest_stack
    if text_stack:
        return text_stack
    revision = source_revision_from_oci(
        str(meta.get("image") or meta.get("deployment_image")
            or meta.get("oci_image") or ""), image_id)
    if revision in STACK_IMAGES:
        return revision
    if revision and revision not in UNPINNED_REVISIONS:
        return "other"
    return None


def optional_version_gate(values: Mapping[str, str], *,
                          stack: str = "r31") -> list[str]:
    """Reasons an explicitly declared version contradicts ``stack``.

    An absent api/source/image yields no reason: a general shape is admitted
    with an unresolved image identity.  Only an explicit declaration against
    another stack or another CANN release is a mismatch.
    """
    reasons: list[str] = []
    api = values.get("api") or values.get("api_version")
    if api is not None and not api.startswith(CANN_VERSION):
        reasons.append(f"version:api_not_{CANN_VERSION}")
    selected = declared_stack(values)
    if selected is not None and selected != stack:
        source = values.get("source") or values.get("source_revision") \
            or values.get("revision")
        reasons.append(f"version:source_not_{stack}" if source
                       else f"version:image_not_{stack}")
    return reasons
