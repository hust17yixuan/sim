"""Ordered ACLNN invocation normalization: API, tensors, attrs, version.

[Role] Turn one declared invocation into the typed record the ABI catalogue
  binds: entry/version aliasing, ordered slot presence, dtype/layout/shape
  checks, declared conditions and the replayable minimal input for a gap.
[Invariants] Absent optional slots stay absent and are never guessed.  A
  version pin is a declared selector, not a rejection: only an explicit
  contradiction with a declared software stack fails.  No duration, profiler
  row or observation enters normalization.
[Timeouts & Locks] Pure bounded parsing; no waits or locks.
[Side-Effect Policy] Read-only; returns records, never mutates the catalogue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import prod
import importlib
import json
import math
from types import SimpleNamespace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_native_contracts import (
    add_rmsnorm_abi_reasons, add_rmsnorm_abi_region,
)
from ..native_binding import (
    NativeAbiRegion, NativeCallBinding, bind_abi_call, join_profiler_rows,
)
from ..runtime.contracts import (
    AllocationBinding, AllocationKey, AllocationSpec, BackingKey, BindingFrame,
    CatalogGeneration, EdgeSpec, OccurrenceKey, PoolKey, PortBinding, PortSpec,
    ProgramCatalog, StageSpec, WorkTerm,
)
from .compiler import ImplementationPolicy, PipeSupply
from .native_add_rms_norm import (
    NativeVectorExecutionUnavailable, prepare_add_rms_norm,
    predict_add_rms_norm,
)
from .native_calls import (
    NativeAbiCallSpec, NativeAbiTensorSpec, a5_native_abi_specs,
    native_abi_inventory, native_abi_spec, validate_native_occurrence,
)
from .native_qbmm import QBMM_SOURCES
from .native_catalog_builder import (
    KernelSupply, StageDraft, catalog_from_drafts, drafts_for,
    drafts_from_work,
)
from .native_occupancy import cacheline_bytes, vector_segments

# Selected cross-module records and helpers (no cycles: see module DAG).
from .native_abi_common import (  # noqa: E402,F401
    ABI_REPLAY_SCHEMA, ABI_REQUEST_SCHEMA, ABI_RESULT_SCHEMA,
    NativeAbiIncomplete, NativeAbiInvocation, NativeAbiMismatch,
    NativeAbiPrediction, NativeAbiTensor, NativeAbiVersionMismatch,
    _ALLOWED_API, _ALLOWED_CONDITIONS, _ALLOWED_REQUEST, _ALLOWED_TENSOR,
    _ALLOWED_VERSION, _DTYPE_ALIASES, _canonical_dtype, _digest_text,
    _execution_machine, _execution_resource_bindings, _full_typed,
    _hardware_execution, _invocation_ports, _invocation_replay,
    _native_cube_rate, _native_pool, _plain, _result, _stage_context, _strict,
)
def _request_entry(request: Any) -> str | None:
    if isinstance(request, str):
        return request
    if not isinstance(request, Mapping):
        return None
    api = request.get("api")
    if isinstance(api, str):
        return api
    if isinstance(api, Mapping):
        for key in ("entry", "symbol", "aclnn_symbol"):
            value = api.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _unknown_replay(request: Any) -> dict[str, Any]:
    entries = [spec.entry for spec in a5_native_abi_specs()]
    payload = dict(request) if isinstance(request, Mapping) else {
        "api": request if isinstance(request, str) else None}
    return {
        "schema": ABI_REPLAY_SCHEMA,
        "request": payload,
        "declared_entries": entries,
        "missing": ["api.entry"],
        "replayable": False,
        "reason": "native_abi_spec_not_declared",
    }

def _norm_version(raw: Any, api_version: Any, spec: NativeAbiCallSpec) -> dict:
    values: dict[str, Any] = {}
    if raw is not None:
        if isinstance(raw, str):
            values["api"] = raw
        elif isinstance(raw, Mapping):
            unknown = set(raw) - _ALLOWED_VERSION
            if unknown:
                raise ModelingError(
                    "native_abi_version_unknown_fields:" + ",".join(sorted(unknown)))
            for key in ("api", "api_version", "aclnn", "cann"):
                if raw.get(key) is None:
                    continue
                if values.get("api") not in (None, raw[key]):
                    raise ModelingError("native_abi_version_alias_conflict")
                values["api"] = raw[key]
            for key in ("source", "source_revision", "revision"):
                if raw.get(key) is None:
                    continue
                if values.get("source") not in (None, raw[key]):
                    raise ModelingError("native_abi_version_alias_conflict")
                values["source"] = raw[key]
            for key in ("image_id", "image"):
                if raw.get(key) is None:
                    continue
                if values.get("image_id") not in (None, raw[key]):
                    raise ModelingError("native_abi_version_alias_conflict")
                values["image_id"] = raw[key]
        else:
            raise ModelingError("native_abi_version_requires_string_or_mapping")
    if api_version is not None:
        if values.get("api") not in (None, api_version):
            raise NativeAbiMismatch("native_abi_api_version_conflict")
        values["api"] = api_version
    if spec.api_version:
        if values.get("api") is None:
            raise NativeAbiIncomplete(
                "native_abi_version_required", ("version.api",))
        if str(values["api"]) != spec.api_version:
            raise NativeAbiVersionMismatch(
                f"native_abi_version_mismatch:{values['api']}!={spec.api_version}")
    if spec.source_revision and values.get("source") is not None:
        if str(values["source"]) != spec.source_revision:
            raise NativeAbiVersionMismatch(
                "native_abi_source_revision_mismatch:"
                f"{values['source']}!={spec.source_revision}")
    if spec.image_id and values.get("image_id") is not None:
        if _digest_text(values["image_id"]) != _digest_text(spec.image_id):
            raise NativeAbiVersionMismatch("native_abi_image_mismatch")
    return {key: value for key, value in values.items() if value is not None}


def _norm_api(raw: Any, spec: NativeAbiCallSpec) -> dict:
    if isinstance(raw, str):
        entry, namespace, symbol, version = raw, "aclnn", raw, None
        if raw == spec.aclnn_symbol:
            namespace = spec.namespace
    elif isinstance(raw, Mapping):
        unknown = set(raw) - _ALLOWED_API
        if unknown:
            raise ModelingError(
                "native_abi_api_unknown_fields:" + ",".join(sorted(unknown)))
        namespace = str(raw.get("namespace") or spec.namespace)
        entry = raw.get("entry") or raw.get("symbol")
        symbol = raw.get("aclnn_symbol") or raw.get("symbol")
        if not entry and isinstance(symbol, str):
            entry = symbol
        if not symbol and namespace == "aclnn":
            symbol = entry
        version = raw.get("version", raw.get("api_version"))
    else:
        raise ModelingError("native_abi_api_requires_string_or_mapping")
    if not isinstance(entry, str) or not entry:
        raise NativeAbiIncomplete("native_abi_api_entry_required", ("api.entry",))
    # ``torch_npu`` is the Python binding name for the same torch dispatch
    # entry (``torch_npu.npu_quant_matmul`` / fused-infer-attention-score).
    # Bind it to the declared torch namespace; the original spelling remains
    # in the invocation's request facts.
    namespace = {"torch_npu": "torch", "torch-npu": "torch"}.get(
        namespace, namespace)
    if namespace != spec.namespace:
        raise NativeAbiMismatch(
            f"native_abi_api_namespace_mismatch:{namespace}!={spec.namespace}")
    if entry not in {spec.entry, spec.aclnn_symbol}:
        raise NativeAbiMismatch(f"native_abi_api_entry_mismatch:{entry}")
    if symbol is not None and spec.aclnn_symbol is not None and symbol != spec.aclnn_symbol:
        raise NativeAbiMismatch(f"native_abi_aclnn_symbol_mismatch:{symbol}")
    return {"namespace": namespace, "entry": entry,
            "aclnn_symbol": symbol, "version": version}

def _raw_tensors(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(name): value for name, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        result = {}
        for item in raw:
            if not isinstance(item, Mapping) or not item.get("name"):
                raise ModelingError("native_abi_tensor_requires_name")
            name = str(item["name"])
            if name in result:
                raise ModelingError(f"native_abi_tensor_duplicate:{name}")
            result[name] = item
        return result
    raise ModelingError("native_abi_tensors_requires_mapping_or_list")

def _norm_tensors(raw: Any, spec: NativeAbiCallSpec) -> tuple[NativeAbiTensor, ...]:
    provided = _raw_tensors(raw)
    declared = {tensor.name: tensor for tensor in spec.tensors}
    unknown = set(provided) - set(declared)
    if unknown:
        raise NativeAbiMismatch(
            "native_abi_unknown_tensors:" + ",".join(sorted(unknown)))
    tensors: list[NativeAbiTensor] = []
    missing: list[str] = []
    for tensor_spec in spec.tensors:
        item = provided.get(tensor_spec.name)
        if item is None:
            if tensor_spec.optional:
                tensors.append(NativeAbiTensor(
                    tensor_spec.name, tensor_spec.role, (), "", "ND",
                    optional=True, present=False))
                continue
            missing.append(f"tensors.{tensor_spec.name}")
            continue
        if not isinstance(item, Mapping):
            raise ModelingError(
                f"native_abi_tensor_requires_mapping:{tensor_spec.name}")
        unknown_fields = set(item) - _ALLOWED_TENSOR
        if unknown_fields and not spec.preserve_tensor_fields:
            raise ModelingError(
                "native_abi_tensor_unknown_fields:" + tensor_spec.name + ":"
                + ",".join(sorted(unknown_fields)))
        present = item.get("present", True) is not False
        # A capture may retain an absent optional position as an explicit
        # ``presence: absent`` slot rather than ``present: false``.  After the
        # ordered binder copies the slot, it carries no shape/dtype/layout
        # descriptor at all; an optional position like that is absent, not a
        # malformed tensor.  Required positions still report the same gaps.
        if (present and tensor_spec.optional
                and item.get("shape") in (None, ())
                and not item.get("dtype")
                and item.get("layout") in (None, "", "ND")):
            present = False
        if not present:
            if tensor_spec.optional:
                tensors.append(NativeAbiTensor(
                    tensor_spec.name, tensor_spec.role, (), "", "ND",
                    optional=True, present=False))
                continue
            missing.append(f"tensors.{tensor_spec.name}")
            continue
        shape = item.get("shape")
        if not isinstance(shape, (list, tuple)):
            missing.append(f"tensors.{tensor_spec.name}.shape")
            continue
        # An empty shape is a declared scalar operand (for example the scalar
        # addend of ``aclnnAdds``), not a missing descriptor: it stays present
        # with zero dimensions so the leaf can charge one broadcast read.
        values = []
        invalid = False
        for value in shape:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                invalid = True
                break
            values.append(value)
        if invalid:
            missing.append(f"tensors.{tensor_spec.name}.shape")
            continue
        dtype = _canonical_dtype(item.get("dtype") or tensor_spec.dtype)
        if not dtype:
            missing.append(f"tensors.{tensor_spec.name}.dtype")
            continue
        if (tensor_spec.dtype and dtype != tensor_spec.dtype
                and not spec.preserve_tensor_fields):
            raise NativeAbiMismatch(
                f"native_abi_tensor_dtype_mismatch:{tensor_spec.name}:"
                f"{dtype}!={tensor_spec.dtype}")
        layout = str(item.get("layout") or "ND").upper()
        if layout != "ND" and not spec.preserve_tensor_fields:
            raise NativeAbiMismatch(
                f"native_abi_tensor_layout_mismatch:{tensor_spec.name}:{layout}")
        role = str(item.get("role") or tensor_spec.role)
        if role != tensor_spec.role:
            raise NativeAbiMismatch(
                f"native_abi_tensor_role_mismatch:{tensor_spec.name}:"
                f"{role}!={tensor_spec.role}")
        fields = tuple(sorted(
            (str(key), item[key]) for key in item
            if key in _ALLOWED_TENSOR or spec.preserve_tensor_fields))
        tensors.append(NativeAbiTensor(
            tensor_spec.name, role, tuple(values), dtype, layout,
            optional=bool(tensor_spec.optional or item.get("optional")),
            present=True, fields=fields))
    if missing:
        raise NativeAbiIncomplete(
            "native_abi_incomplete_typed_arguments", tuple(missing))
    return tuple(tensors)


def _norm_attrs(raw: Any, spec: NativeAbiCallSpec) -> dict:
    if raw is None:
        attrs = {}
    elif isinstance(raw, Mapping):
        attrs = dict(raw)
    else:
        raise ModelingError("native_abi_attrs_requires_mapping")
    allowed = set(spec.attrs) | set(spec.optional_attrs)
    unknown = set(attrs) - allowed
    if unknown and not spec.preserve_extra_attrs:
        raise NativeAbiMismatch(
            "native_abi_unknown_attrs:" + ",".join(sorted(unknown)))
    missing = [f"attrs.{name}" for name in spec.attrs if name not in attrs]
    if missing:
        raise NativeAbiIncomplete(
            "native_abi_incomplete_typed_arguments", tuple(missing))
    result = {}
    for name, value in attrs.items():
        if isinstance(value, (dict, set)):
            raise ModelingError(f"native_abi_attr_not_scalar:{name}")
        if isinstance(value, (list, tuple)):
            result[name] = [item for item in value]
        else:
            result[name] = value
    return result

def _norm_conditions(raw: Any, spec: NativeAbiCallSpec) -> dict:
    if raw is None:
        conditions = {}
    elif isinstance(raw, Mapping):
        conditions = dict(raw)
    else:
        raise ModelingError("native_abi_conditions_requires_mapping")
    unknown = set(conditions) - _ALLOWED_CONDITIONS
    if unknown and not spec.preserve_extra_conditions:
        raise ModelingError(
            "native_abi_conditions_unknown_fields:" + ",".join(sorted(unknown)))
    raw_variant = conditions.get("variant")
    raw_owner = conditions.get("owner")
    raw_members = conditions.get("members")
    if raw_members is not None:
        if isinstance(raw_members, str) or not isinstance(
                raw_members, (list, tuple)):
            raise ModelingError("native_abi_members_requires_list")
        raw_members = tuple(str(item) for item in raw_members)
    layer = conditions.get("layer")
    if layer is not None and (isinstance(layer, bool)
                              or not isinstance(layer, int) or layer < 0):
        raise ModelingError("native_abi_layer_invalid")

    def by_name(name):
        return next((item for item in spec.variants if item.name == name), None)

    caller_present = any(
        key in conditions for key in ("variant", "owner", "members"))
    if not caller_present and spec.preserve_extra_conditions:
        match = None
    elif raw_variant is not None:
        match = by_name(str(raw_variant))
        if match is None:
            raise NativeAbiMismatch(
                f"native_abi_variant_not_declared:{raw_variant}")
    elif raw_owner is not None:
        candidates = [item for item in spec.variants
                      if item.owner == raw_owner]
        if not candidates:
            raise NativeAbiMismatch(
                f"native_abi_owner_not_declared:{raw_owner}")
        if len(candidates) != 1:
            raise NativeAbiMismatch(
                f"native_abi_owner_ambiguous:{raw_owner}")
        match = candidates[0]
    elif raw_members is not None:
        if raw_members:
            match = next((item for item in spec.variants
                          if tuple(item.members) == raw_members), None)
        else:
            match = by_name("native_call")
        if match is None:
            raise NativeAbiMismatch(
                "native_abi_members_not_declared:"
                + ",".join(raw_members))
    else:
        # A bare native call has no model/caller attribution.  Never default
        # to the fused N0 multi-member plan merely because it is declared first.
        match = by_name("native_call")
        if match is None and spec.variants:
            match = spec.variants[0]
    if match is None and caller_present:
        raise NativeAbiMismatch("native_abi_variant_unavailable")
    if match is not None:
        if raw_owner is not None and match.owner != raw_owner:
            raise NativeAbiMismatch(
                f"native_abi_owner_conflicts_variant:{raw_owner}!={match.owner}")
        if (raw_members is not None
                and tuple(match.members) != raw_members):
            raise NativeAbiMismatch(
                "native_abi_members_conflict:"
                + ",".join(raw_members) + "!=" + ",".join(match.members))
    sequence_rows = conditions.get("sequence_rows", "")
    if sequence_rows not in {"", "shard", "selected"}:
        raise ModelingError("native_abi_sequence_rows_invalid")
    sequence_ranks = conditions.get("sequence_ranks", 1)
    if (isinstance(sequence_ranks, bool)
            or not isinstance(sequence_ranks, int) or sequence_ranks < 1):
        raise ModelingError("native_abi_sequence_ranks_invalid")
    phase = conditions.get("phase")
    if phase not in (None, "prefill", "decode"):
        raise ModelingError("native_abi_phase_invalid")
    result = {
        "sequence_rows": sequence_rows,
        "sequence_ranks": sequence_ranks,
    }
    if match is not None:
        result.update({
            "variant": match.name,
            "owner": match.owner,
            "members": tuple(match.members),
        })
    if phase:
        result["phase"] = phase
    if layer is not None:
        result["layer"] = layer
    if spec.preserve_extra_conditions:
        result.update({
            str(key): value for key, value in conditions.items()
            if key not in _ALLOWED_CONDITIONS
        })
    return result


def normalize_native_abi(request: Any, spec: NativeAbiCallSpec) -> NativeAbiInvocation:
    """Normalize API + tensor + attr + version once, or raise a typed gap."""
    if isinstance(request, str):
        request = {"api": request}
    if not isinstance(request, Mapping):
        raise ModelingError("native_abi_request_requires_mapping")
    unknown = set(request) - _ALLOWED_REQUEST
    if unknown:
        raise ModelingError(
            "native_abi_request_unknown_fields:" + ",".join(sorted(unknown)))
    api_raw = request.get("api")
    api = _norm_api(api_raw, spec)
    raw_api_version = api.pop("version", None)
    raw_api_version = api.pop("api_version", raw_api_version)
    version = _norm_version(request.get("version"), raw_api_version, spec)
    # A capture names the same physical slot by its profiler spelling or its
    # ACLNN spelling (``key_cache``/``key``).  Normalize the names before the
    # ordered positional binding; absent positions are still preserved.
    from .native_binding import (
        clean_conditions, normalize_attr_aliases, normalize_tensor_aliases,
    )
    declared_tensors = {tensor.name for tensor in spec.tensors}
    declared_attrs = set(spec.attrs) | set(spec.optional_attrs)
    tensors = _norm_tensors(
        normalize_tensor_aliases(request.get("tensors"), declared_tensors),
        spec)
    attrs = _norm_attrs(
        normalize_attr_aliases(request.get("attrs"), declared_attrs), spec)
    conditions = _norm_conditions(
        clean_conditions(request.get("conditions")), spec)
    payload = {
        "api": api, "version": version,
        "tensors": [tensor.to_dict() for tensor in tensors
                    if tensor.present],
        "attrs": attrs,
    }
    abi_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    return NativeAbiInvocation(
        api=api, version=version, tensors=tensors, attrs=attrs,
        conditions=conditions, abi_hash=abi_hash)


def _replay_for(spec: NativeAbiCallSpec, request: Any, missing=(),
                reason: str = "insufficient_fields") -> dict:
    """Replay only the facts the caller supplied; the catalogue is not evidence.

    This is the single construction for failed/incomplete requests.  It never
    fills API/source/image from ``spec`` and never selects the catalogue's first
    variant.  Missing fields stay in ``missing``; the caller may add them and
    resubmit without changing the original deployment/caller facts.
    """
    raw = request if isinstance(request, Mapping) else {}
    provided = _raw_tensors(raw.get("tensors"))
    tensors = {
        str(name): (dict(item) if isinstance(item, Mapping) else item)
        for name, item in provided.items()
    }
    raw_attrs = raw.get("attrs")
    attrs = dict(raw_attrs) if isinstance(raw_attrs, Mapping) else raw_attrs
    raw_conditions = raw.get("conditions")
    conditions = (dict(raw_conditions) if isinstance(raw_conditions, Mapping)
                  else raw_conditions)
    raw_api = raw.get("api")
    api = dict(raw_api) if isinstance(raw_api, Mapping) else raw_api
    raw_version = raw.get("version")
    version = (dict(raw_version) if isinstance(raw_version, Mapping)
               else raw_version)
    return {
        "schema": ABI_REPLAY_SCHEMA,
        "api": api if api is not None else {},
        "version": version if version is not None else {},
        "tensors": tensors,
        "attrs": attrs if attrs is not None else {},
        "conditions": conditions if conditions is not None else {},
        "missing": list(missing),
        "replayable": not missing,
        "reason": reason,
    }


def _missing_fields(request: Any, spec: NativeAbiCallSpec) -> tuple[str, ...]:
    """All typed fields still needed for a replay, not just the first gap."""
    missing: list[str] = []
    raw = request if isinstance(request, Mapping) else {}
    provided = _raw_tensors(raw.get("tensors"))
    for tensor in spec.tensors:
        item = provided.get(tensor.name)
        if item is None or (isinstance(item, Mapping)
                            and item.get("present", True) is False):
            if not tensor.optional:
                missing.append(f"tensors.{tensor.name}")
            continue
        if not isinstance(item, Mapping):
            missing.append(f"tensors.{tensor.name}")
            continue
        shape = item.get("shape")
        if (not isinstance(shape, (list, tuple)) or not shape
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in shape)):
            missing.append(f"tensors.{tensor.name}.shape")
        if not _canonical_dtype(item.get("dtype") or tensor.dtype):
            missing.append(f"tensors.{tensor.name}.dtype")
    attrs = raw.get("attrs") if isinstance(raw.get("attrs"), Mapping) else {}
    for name in spec.attrs:
        if name not in attrs or attrs.get(name) is None:
            missing.append(f"attrs.{name}")
    return tuple(dict.fromkeys(missing))
