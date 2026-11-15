"""Shared pure helpers for native matrix ABI leaf adapters.

[Role] Normalize typed tensor/attr/version records and assemble the common
  work/geometry/selected-call record used by the matrix ABI leaves.
[Invariants] Pure CPU metadata; no device, weights, observations, sample-shape
  lookup or duration.  Missing or unsupported ABI fields stay explicit gaps;
  no alternative algorithm or envelope is substituted.
[Timeouts & Locks] Bounded dict/list arithmetic; no I/O or locks.
[Side-Effect Policy] Returns fresh plain records; imports only the standard
  library so a leaf can be used without the profiler or a device runtime.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping

ABI_SCHEMA = "mindie-sim/native-matrix-abi/1"
NON_CLAIMS = (
    "static_cpu_work_and_geometry_only",
    "no_device_dispatch",
    "no_measured_duration",
    "no_target_duration_injected",
    "no_algorithm_or_envelope_fallback",
)


def gaps(*values: str) -> list[str]:
    """Deduplicate gap reasons while preserving first-seen order."""
    return list(dict.fromkeys(str(value) for value in values if value))


def int_value(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def align(value: int, quantum: int) -> int:
    return (int(value) + int(quantum) - 1) // int(quantum) * int(quantum)


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def digest(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def status(reasons: list[str]) -> str:
    """Map typed gaps to the one public status; no fallback prediction."""
    reasons = gaps(*reasons)
    if any(reason.startswith("missing:") for reason in reasons):
        return "insufficient_fields"
    if any(reason.startswith(("mismatch:", "invalid:", "version:"))
           for reason in reasons):
        return "no_match"
    if any(reason.startswith("unmodeled:") for reason in reasons):
        return "unmodeled_abi"
    return "predicted"


def _abi_shape(value: Any, *, minimum: int = 1) -> list[int] | None:
    if (not isinstance(value, (list, tuple)) or not value
            or any(isinstance(item, bool) or not isinstance(item, int)
                   or item < minimum for item in value)):
        return None
    return [int(item) for item in value]


_FP4_ALIASES = frozenset({"fp4", "fp4_e2m1", "mxfp4", "mx_fp4",
                          "float4_e2m1", "float4_e2m1fn_x2"})


def _abi_bytes(dtype: str | None) -> int | None:
    text = str(dtype or "").lower()
    if text in {"fp32", "float32", "int32", "uint32"}:
        return 4
    if text in {"bf16", "bfloat16", "fp16", "float16"}:
        return 2
    if text in {"int8", "uint8", "fp8", "fp8_e4m3", "fp8_e5m2",
                "fp8_e8m0", "e8m0"}:
        return 1
    return None


def _abi_view(raw: Mapping[str, Any], kind: str,
              fallback: Mapping[str, Any] | None = None
              ) -> tuple[dict[str, Any] | None, list[str]]:
    """Merge flat and nested declarations of one physical view.

    Flat ``storage_shape/storage_dtype/storage_layout`` and nested
    ``views.storage.shape/dtype/layout`` are aliases.  When both are present
    they must agree; the helper never silently prefers one and drops the
    other.  A declared view is preserved, not reconstructed from the logical
    shape.
    """
    nested_all = raw.get("views")
    unknown = []
    if isinstance(nested_all, Mapping):
        unknown = [str(key) for key in nested_all
                   if key not in {"logical", "storage", "device",
                                  "interpretation"}]
    nested = (nested_all.get(kind)
              if isinstance(nested_all, Mapping)
              and isinstance(nested_all.get(kind), Mapping) else {})
    prefix = "interpret" if kind == "interpretation" else kind
    base = dict(fallback or {})

    def choose(field: str, nested_field: str = "dtype") -> tuple[Any, bool]:
        flat = raw.get(f"{prefix}_{field}")
        nested_value = nested.get(nested_field)
        if flat is not None and nested_value is not None:
            if isinstance(flat, (list, tuple)) or isinstance(
                    nested_value, (list, tuple)):
                same = tuple(flat) == tuple(nested_value)
            else:
                same = str(flat).lower() == str(nested_value).lower()
            if not same:
                return None, True
        return (flat if flat is not None else nested_value), False

    shape_value, conflict = choose("shape", "shape")
    dtype_value, dtype_conflict = choose("dtype", "dtype")
    layout_value, layout_conflict = choose("layout", "layout")
    conflict = conflict or dtype_conflict or layout_conflict
    if not any((shape_value is not None, dtype_value is not None,
                layout_value is not None, bool(nested))):
        return None, ([f"unmodeled:views:{','.join(unknown)}"] if unknown else [])
    if conflict:
        return None, ([f"mismatch:views.{kind}"]
                      + ([f"unmodeled:views:{','.join(unknown)}"]
                         if unknown else []))
    shape = _abi_shape(shape_value)
    dtype = str(dtype_value or base.get("dtype") or "").lower()
    layout = str(layout_value or base.get("layout") or "").upper()
    if kind == "interpretation":
        if not dtype:
            return None, [f"invalid:views.{kind}"]
        row: dict[str, Any] = {"dtype": dtype,
                               "layout": layout or None}
        direction = nested.get("direction") if isinstance(nested, Mapping) else None
        if direction is not None:
            row["direction"] = str(direction).lower()
        return row, ([f"unmodeled:views:{','.join(unknown)}"] if unknown else [])
    if shape is None or not dtype or not layout:
        return None, [f"invalid:views.{kind}"]
    row: dict[str, Any] = {"shape": shape, "dtype": dtype, "layout": layout}
    nested_bytes = nested.get("bytes") if isinstance(nested, Mapping) else None
    flat_bytes = raw.get(f"{prefix}_bytes")
    if flat_bytes is not None and nested_bytes is not None \
            and int(flat_bytes) != int(nested_bytes):
        return None, [f"mismatch:views.{kind}.bytes"]
    declared = flat_bytes if flat_bytes is not None else nested_bytes
    width = _abi_bytes(dtype)
    if declared is not None:
        if (width is None or isinstance(declared, bool)
                or not isinstance(declared, int) or declared < 1):
            return None, [f"invalid:views.{kind}.bytes"]
        if declared != int(math.prod(shape)) * width:
            return None, [f"mismatch:views.{kind}.bytes"]
        row["bytes"] = declared
    elif width is not None:
        row["bytes"] = int(math.prod(shape)) * width
    direction = nested.get("direction") if isinstance(nested, Mapping) else None
    if kind == "logical":
        for key in ("logical_direction", "direction", "orientation"):
            if raw.get(key) is not None:
                if direction is not None and str(raw[key]).lower() != str(
                        direction).lower():
                    return None, [f"mismatch:views.{kind}.direction"]
                direction = raw[key]
    if direction is not None:
        row["direction"] = str(direction).lower()
    return row, ([f"unmodeled:views:{','.join(unknown)}"] if unknown else [])


_PROVENANCE_KEYS = (
    "origin", "producer", "completion", "input", "output",
    "preparation_id", "preparation",
)


def preparation_record(*, kind: str, producer: str,
                       input_slot: str, input_shape: list[int],
                       output_slot: str, output_shape: list[int],
                       input_dtype: str = "int8",
                       output_dtype: str = "int8",
                       source_path: str | None = None,
                       source_sha256: str | None = None,
                       revision: str | None = None,
                       completion: str = "before_native_call"
                       ) -> dict[str, Any]:
    """One declared producer result and its exact read/write traffic."""
    width_in = 1 if input_dtype in {"int8", "uint8", "fp8_e4m3",
                                    "fp8_e8m0"} else 2
    width_out = 1 if output_dtype in {"int8", "uint8", "fp8_e4m3",
                                      "fp8_e8m0"} else 2
    read_bytes = int(math.prod(input_shape)) * width_in
    write_bytes = int(math.prod(output_shape)) * width_out
    body = {
        "kind": str(kind), "producer": str(producer),
        "input": {"slot": str(input_slot), "shape": list(input_shape),
                  "dtype": input_dtype, "layout": "ND"},
        "output": {"slot": str(output_slot), "shape": list(output_shape),
                   "dtype": output_dtype, "layout": "FRACTAL_NZ"},
        "read_bytes": read_bytes, "write_bytes": write_bytes,
        "traffic_bytes": read_bytes + write_bytes,
        "bytes": read_bytes + write_bytes,
        "included_in_call_footprint": False,
        "completion": str(completion),
    }
    if source_path is not None:
        body["source_path"] = str(source_path)
    if source_sha256 is not None:
        body["source_sha256"] = str(source_sha256)
    if revision is not None:
        body["revision"] = str(revision)
    body["preparation_id"] = "sha256:" + sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode(
            "utf-8")).hexdigest()
    return body


def provenance(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the normalized producer/output provenance of one slot."""
    value = (row or {}).get("provenance")
    return dict(value) if isinstance(value, Mapping) else {}


def slot_view_reasons(row: Mapping[str, Any] | None, name: str, *,
                      logical: Mapping[str, Any] | None = None,
                      storage: Mapping[str, Any] | None = None,
                      device: Mapping[str, Any] | None = None,
                      interpretation: Mapping[str, Any] | None = None
                      ) -> list[str]:
    """Validate every declared view of one slot against the source constraint.

    A missing view is optional; when a view is declared, its shape, dtype,
    layout and direction must all agree with the expected physical form.
    """
    if row is None:
        return []
    reasons: list[str] = []
    expected = {"logical": logical, "storage": storage, "device": device,
                "interpretation": interpretation}
    views = row.get("views") or {}
    for kind in ("logical", "storage", "device", "interpretation"):
        want = expected.get(kind)
        actual = views.get(kind)
        if want is None or actual is None:
            continue
        if want.get("shape") is not None and (
                list(actual.get("shape") or ()) != list(want["shape"])):
            reasons.append(f"mismatch:tensors.{name}.{kind}_shape")
        if want.get("dtype") is not None and str(
                actual.get("dtype") or "").lower() != str(
                want["dtype"]).lower():
            reasons.append(f"mismatch:tensors.{name}.{kind}_dtype")
        if want.get("layout") is not None and str(
                actual.get("layout") or "").upper() != str(
                want["layout"]).upper():
            reasons.append(f"mismatch:tensors.{name}.{kind}_layout")
        if (want.get("direction") is not None
                and actual.get("direction") is not None
                and str(actual.get("direction")).lower() != str(
                    want["direction"]).lower()):
            reasons.append(f"mismatch:tensors.{name}.{kind}_direction")
    return list(dict.fromkeys(reasons))


def tensor(tensors: Mapping[str, Any] | None, name: str, *, role: str,
           dtype: str | None = None, rank: int | None = None,
           optional: bool = False, layouts: tuple[str, ...] = ("ND",),
           dtypes: tuple[str, ...] | None = None
           ) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize one declared tensor slot and all of its viable views.

    ``shape`` is the ABI tensor shape.  ``views.logical`` may be a transposed
    or tiled logical view with the same element count.  ``views.storage`` and
    ``views.device`` are explicit physical declarations; they are never
    derived from the logical element count.  ``views.interpretation`` is the
    element interpretation (for example packed FP4 or E8M0).  Flat and nested
    aliases for the same field are merged and must agree.
    """
    raw = (tensors or {}).get(name)
    if raw is None or (isinstance(raw, Mapping)
                       and raw.get("present", True) is False):
        if optional:
            return None, []
        return None, [f"missing:tensors.{name}"]
    if not isinstance(raw, Mapping):
        return None, [f"invalid:tensors.{name}"]
    shape = _abi_shape(raw.get("shape"))
    if shape is None or (rank is not None and len(shape) != rank):
        return None, [f"invalid:tensors.{name}.shape"]
    actual_dtype = str(raw.get("dtype") or dtype or "").lower()
    allowed = tuple(dtypes or ((dtype,) if dtype else ()))
    if not actual_dtype:
        return None, [f"missing:tensors.{name}.dtype"]
    if allowed and actual_dtype not in allowed:
        return None, [f"mismatch:tensors.{name}.dtype:{actual_dtype}"]
    layout = str(raw.get("layout") or layouts[0]).upper()
    if layout not in layouts:
        return None, [f"mismatch:tensors.{name}.layout:{layout}"]
    actual_role = str(raw.get("role") or role)
    if actual_role != role:
        return None, [f"mismatch:tensors.{name}.role:{actual_role}"]
    reasons: list[str] = []
    logical, logical_gaps = _abi_view(
        raw, "logical", {"dtype": actual_dtype, "layout": layout})
    reasons.extend(logical_gaps)
    if logical is None:
        logical = {"shape": list(shape), "dtype": actual_dtype,
                   "layout": layout}
        for key in ("logical_direction", "direction", "orientation"):
            if raw.get(key) is not None:
                logical["direction"] = str(raw[key]).lower()
                break
    if math.prod(logical["shape"]) != math.prod(shape):
        reasons.append(f"mismatch:tensors.{name}.logical_shape")
    storage, storage_gaps = _abi_view(
        raw, "storage", {"dtype": actual_dtype, "layout": layout})
    device, device_gaps = _abi_view(
        raw, "device", {"dtype": actual_dtype, "layout": layout})
    interpretation, interpret_gaps = _abi_view(
        raw, "interpretation", {"dtype": actual_dtype, "layout": layout})
    reasons.extend(storage_gaps)
    reasons.extend(device_gaps)
    reasons.extend(interpret_gaps)
    if (interpretation is not None
            and _abi_bytes(interpretation["dtype"]) is None
            and interpretation["dtype"] not in _FP4_ALIASES):
        reasons.append(f"invalid:tensors.{name}.interpret_dtype")
    if reasons:
        return None, list(dict.fromkeys(reasons))
    row = {"name": name, "role": role, "dtype": actual_dtype,
           "shape": shape, "layout": layout,
           "optional": bool(optional), "present": True,
           "direction": logical.get("direction"),
           "views": {
               "logical": logical,
               "storage": storage,
               "device": device,
               "interpretation": interpretation,
           }}
    nested_prov = raw.get("provenance")
    if isinstance(nested_prov, Mapping):
        prov = dict(nested_prov)
    else:
        prov = {key: raw[key] for key in _PROVENANCE_KEYS
                if raw.get(key) is not None}
    if prov:
        row["provenance"] = prov
    return row, []


def view_value(row: Mapping[str, Any], kind: str, key: str) -> Any:
    view = (row.get("views") or {}).get(kind)
    return view.get(key) if isinstance(view, Mapping) else None


def interpretation_reasons(row: Mapping[str, Any], name: str,
                           allowed: tuple[str, ...]) -> list[str]:
    actual = view_value(row, "interpretation", "dtype")
    if actual is None:
        return []
    if str(actual).lower() not in allowed:
        return [f"mismatch:tensors.{name}.interpret_dtype:{actual}"]
    return []


def direction_reasons(row: Mapping[str, Any], name: str,
                      allowed: tuple[str, ...]) -> list[str]:
    actual = row.get("direction") or view_value(row, "logical", "direction")
    if actual is None:
        return []
    if str(actual).lower() not in allowed:
        return [f"mismatch:tensors.{name}.direction:{actual}"]
    return []


def row_column_scale_epilogue(*, m: int, n: int,
                              row_slot: str = "pertoken_scale",
                              column_slot: str = "weight_scale",
                              groups: int = 1,
                              output_dtype: str = "bf16",
                              column_dtype: str = "bf16") -> dict[str, Any]:
    """Describe the existing final-K row/column scale epilogue.

    The accumulator is INT32; the output path converts to FP32, applies the
    per-token FP32 row scale and per-channel BF16/FP32 column scale, then
    converts to BF16.  That is 4 semantic vector operations per output.  The
    actual Vector/Fixpipe ownership and fusion stay with the selected matrix
    program; this record only makes the numerical dependency explicit.
    """
    ops = 4 if str(output_dtype).lower() in {"bf16", "bfloat16"} else 3
    col_fmt = ("fp32" if str(column_dtype).lower() in {"fp32", "float32"}
               else "bf16")
    col_width = 4 if col_fmt == "fp32" else 2
    row_bytes = int(m) * 4
    column_bytes = int(groups) * int(n) * col_width
    return {
        "kind": "row_column_scale",
        "row_scale_bytes": row_bytes,
        "column_scale_bytes": column_bytes,
        "operations_per_output": ops,
        "vector_ops": ops * int(m) * int(n),
        "output_dtype": str(output_dtype).lower(),
        "row_scale": {"slot": row_slot, "dtype": "fp32",
                      "shape": [int(m)], "bytes": row_bytes},
        "column_scale": {"slot": column_slot, "dtype": col_fmt,
                         "shape": [int(groups) * int(n)],
                         "bytes": column_bytes},
        "dependency": ["k_reduction_complete", "row_scale_ready",
                       "column_scale_ready", "epilogue_once",
                       "output_store"],
        "owner": "matrix_epilogue",
        "applied_after": "complete_k_reduction",
    }


def group_rows(raw: Mapping[str, Any], experts: int,
               group_list_type: Any, interface_rows: int,
               valid_hint: int | None = None
               ) -> tuple[list[int] | None, list[int] | None,
                          int | None, list[str]]:
    """Parse one grouped call's real counts/cumulative encoding.

    The row sum is checked later against the declared valid extent, not
    silently against the interface row count.  Ambiguous encodings require
    the explicit type.
    """
    values = raw.get("values")
    if values is None:
        return None, None, None, ["missing:tensors.group_list.values"]
    if (not isinstance(values, (list, tuple)) or len(values) != int(experts)
            or any(isinstance(item, bool) or not isinstance(item, int)
                   or item < 0 for item in values)):
        return None, None, None, ["invalid:tensors.group_list.values"]
    parsed = [int(item) for item in values]
    cumulative_ok = all(left <= right for left, right in zip(parsed, parsed[1:]))
    declared = int_value(group_list_type)
    if declared is None:
        counts_ok = (sum(parsed) == (valid_hint if valid_hint is not None
                                     else interface_rows))
        ends_at_interface = cumulative_ok and parsed[-1] == interface_rows
        if counts_ok and not ends_at_interface:
            declared = 1
        elif ends_at_interface and not counts_ok:
            declared = 0
        elif counts_ok and ends_at_interface:
            return None, None, None, ["missing:attrs.group_list_type"]
        else:
            return None, None, None, ["mismatch:tensors.group_list.sum"]
    if declared == 1:
        cumulative: list[int] = []
        running = 0
        for count in parsed:
            running += count
            cumulative.append(running)
        return parsed, cumulative, 1, []
    if declared == 0:
        if not cumulative_ok:
            return (None, None, 0,
                    ["mismatch:tensors.group_list.cumulative"])
        previous = 0
        counts: list[int] = []
        for value in parsed:
            counts.append(value - previous)
            previous = value
        return counts, parsed, 0, []
    return None, None, declared, ["mismatch:attrs.group_list_type"]


def row_extents(conditions: Mapping[str, Any], interface_rows: int,
                counts_total: int) -> tuple[dict[str, int] | None,
                                            list[str]]:
    """Separate valid, received/interface and capacity rows.

    A count sum different from the interface rows is legal only when the
    caller explicitly declares the extents; without that declaration the
    list is a mismatch.  Capacity never changes issued work by itself.
    """
    explicit = any(key in conditions for key in (
        "valid_rows", "received_rows", "interface_rows", "capacity_rows"))
    if not explicit:
        if counts_total != interface_rows:
            return None, ["mismatch:tensors.group_list.sum"]
        return {"valid_rows": interface_rows,
                "received_rows": interface_rows,
                "capacity_rows": interface_rows, "explicit": 0}, []
    valid = int_value(conditions.get("valid_rows", counts_total))
    received = int_value(conditions.get("received_rows",
                                        conditions.get("interface_rows",
                                                       interface_rows)))
    capacity = int_value(conditions.get("capacity_rows",
                                        max(received or 0, valid or 0)))
    if valid is None or received is None or capacity is None \
            or min(valid, received, capacity) < 0:
        return None, ["invalid:conditions.row_extents"]
    reasons: list[str] = []
    if received != interface_rows:
        reasons.append("mismatch:conditions.received_rows")
    if valid > received:
        reasons.append("mismatch:conditions.valid_rows_gt_received")
    if capacity < received:
        reasons.append("mismatch:conditions.capacity_rows_lt_received")
    if counts_total != valid:
        reasons.append("mismatch:conditions.valid_rows")
    if reasons:
        return None, reasons
    return {"valid_rows": valid, "received_rows": received,
            "capacity_rows": capacity, "explicit": int(explicit)}, []


def with_views(row: dict[str, Any], *, storage_shape: list[int] | None = None,
               storage_dtype: str | None = None,
               storage_layout: str | None = None,
               device_shape: list[int] | None = None,
               device_dtype: str | None = None,
               device_layout: str | None = None,
               interpret_dtype: str | None = None,
               interpret_layout: str | None = None) -> dict[str, Any]:
    """Fill missing physical views without replacing a caller-declared one."""
    result = dict(row)
    views = dict(result.get("views") or {})
    if views.get("storage") is None and storage_shape is not None:
        dtype = str(storage_dtype or result["dtype"])
        views["storage"] = {"shape": [int(item) for item in storage_shape],
                            "dtype": dtype,
                            "layout": str(storage_layout or "ND").upper()}
        width = _abi_bytes(dtype)
        if width is not None:
            views["storage"]["bytes"] = (
                int(math.prod(storage_shape)) * width)
    if views.get("device") is None and device_shape is not None:
        dtype = str(device_dtype or result["dtype"])
        views["device"] = {"shape": [int(item) for item in device_shape],
                           "dtype": dtype,
                           "layout": str(device_layout or "ND").upper()}
        width = _abi_bytes(dtype)
        if width is not None:
            views["device"]["bytes"] = int(math.prod(device_shape)) * width
    if views.get("interpretation") is None and (
            interpret_dtype or interpret_layout):
        views["interpretation"] = {
            "dtype": (str(interpret_dtype).lower()
                      if interpret_dtype else None),
            "layout": str(interpret_layout or result["layout"]).upper()}
    result["views"] = views
    return result


def view_reasons(row: Mapping[str, Any], name: str, kind: str, *,
                 shape: list[int] | None = None,
                 dtype: str | None = None,
                 layout: str | None = None) -> list[str]:
    """Reject a caller-declared physical view that the selected call cannot use."""
    actual = ((row.get("views") or {}).get(kind))
    if actual is None:
        return []
    reasons: list[str] = []
    if shape is not None and list(actual.get("shape") or ()) != list(shape):
        reasons.append(f"mismatch:tensors.{name}.{kind}_shape")
    if dtype is not None and str(actual.get("dtype") or "").lower() != dtype.lower():
        reasons.append(f"mismatch:tensors.{name}.{kind}_dtype")
    if layout is not None and str(actual.get("layout") or "").upper() != layout.upper():
        reasons.append(f"mismatch:tensors.{name}.{kind}_layout")
    return reasons


def selected(*, entry: str, namespace: str, aclnn_symbol: str | None,
             api_version: str | None, source_revision: str | None,
             image_id: str | None, source: Mapping[str, Any] | None,
             kernels: list[dict[str, Any]] | tuple[dict[str, Any], ...],
             execution_core: str | None, relation: str,
             members: tuple[str, ...] | list[str],
             tensors: list[dict[str, Any]], attrs: Mapping[str, Any],
             regions: list[str] | tuple[str, ...],
             variant: str | None = None) -> dict[str, Any]:
    """One stable selected-call identity derived only from the ABI record."""
    body: dict[str, Any] = {
        "entry": str(entry),
        "namespace": str(namespace),
        "aclnn_symbol": aclnn_symbol,
        "api_version": api_version,
        "source_revision": source_revision,
        "image_id": image_id,
        "source": dict(source or {}),
        "kernels": [dict(kernel) for kernel in kernels],
        "execution_core": execution_core,
        "relation": str(relation),
        "variant": variant,
        "members": [str(member) for member in members],
        "regions": [str(region) for region in regions],
        "tensors": {
            item["name"]: {
                "role": item["role"], "dtype": item["dtype"],
                "rank": len(item["shape"]), "layout": item["layout"],
                "views": {
                    key: (None if value is None else
                          {"dtype": value["dtype"],
                           "layout": value["layout"]})
                    for key, value in (item.get("views") or {}).items()
                },
                "provenance": item.get("provenance"),
            }
            for item in tensors
        },
        "attrs": dict(attrs),
    }
    body["signature_hash"] = sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()
    return body


def result(*, status_value: str, entry: str, reasons: list[str],
           invocation: Mapping[str, Any],
           selected_call: dict[str, Any] | None,
           geometry: dict[str, Any] | None,
           work: dict[str, Any] | None,
           tensors: Mapping[str, Any],
           scale: Mapping[str, Any],
           metadata: Mapping[str, Any],
           accumulator: Mapping[str, Any] | None,
           regions: list[dict[str, Any]],
           schema: str = ABI_SCHEMA) -> dict[str, Any]:
    reasons = gaps(*reasons)
    return {
        "schema": schema,
        "status": str(status_value),
        "entry": str(entry),
        "native_call_ready": status_value == "predicted",
        "fallback_performed": False,
        "selected_call": selected_call,
        "geometry": geometry,
        "work": work,
        "tensors": dict(tensors),
        "scale": dict(scale),
        "metadata": dict(metadata),
        "accumulator": dict(accumulator) if accumulator is not None else None,
        "regions": [dict(region) for region in regions],
        "invocation": dict(invocation),
        "reasons": reasons,
        "gap": reasons,
        "non_claims": list(NON_CLAIMS),
    }


def matrix_builder_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """The exact typed values the generic matrix family consumes."""
    if record.get("status") != "predicted":
        return {"status": record.get("status")}
    tensors = record.get("tensors") or {}
    geometry = record.get("geometry") or {}
    selected = record.get("selected_call") or {}
    attrs = selected.get("attrs") or {}
    runtime = {}
    for name, row in tensors.items():
        if not isinstance(row, Mapping):
            continue
        views = row.get("views") or {}
        storage = views.get("storage")
        device = views.get("device")
        runtime[name] = {
            "shape": list(row.get("shape") or ()),
            "dtype": row.get("dtype"), "layout": row.get("layout"),
            "storage": (None if not isinstance(storage, Mapping) else {
                "shape": storage.get("shape"), "dtype": storage.get("dtype"),
                "layout": storage.get("layout"), "bytes": storage.get("bytes")}),
            "device": (None if not isinstance(device, Mapping) else {
                "shape": device.get("shape"), "dtype": device.get("dtype"),
                "layout": device.get("layout"), "bytes": device.get("bytes")}),
        }
    prep = (record.get("work") or {}).get("preparation")
    return {
        "status": "predicted",
        "m": geometry.get("m"), "n": geometry.get("n"), "k": geometry.get("k"),
        "padded_m": geometry.get("padded_m"),
        "padded_n": geometry.get("padded_n"),
        "padded_k": geometry.get("padded_k"),
        "entry": record.get("entry"), "variant": selected.get("variant"),
        "input_dtypes": attrs.get("input_dtypes"),
        "input_slots": attrs.get("input_slots"),
        "offset": attrs.get("offset"),
        "transpose": {"x1": attrs.get("transpose_x1"),
                      "x2": attrs.get("transpose_x2")},
        "quant": {"quantized": bool(
            str(record.get("entry")) != "aclnnMatmul"),
            "weight_path": (geometry.get("weight_path")
                            or (record.get("metadata") or {}).get(
                                "weight_path")),
            "weight_layout": runtime.get("weight", {}).get("layout"),
            "weight_nz_layout": runtime.get("weight_nz", {}).get("layout")},
        "scales": record.get("scale") or {},
        "runtime_views": runtime,
        "preparation": prep,
        "footprint_bytes": (record.get("work") or {}).get("bytes", {}).get(
            "total"),
    }


def _execution_inputs(record: Mapping[str, Any], *, tile=(16, 32, 16),
                      slots: int = 2):
    """Create the one finite matrix op/recipe described by a passed record."""
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.quant import QuantSpec, quant_preset
    from mindie_sim.prediction.matrix import TiledMatmul

    entry = str(record.get("entry") or "")
    geom = record.get("geometry") or {}
    m, n, k = int(geom["m"]), int(geom["n"]), int(geom["k"])
    variant = str((record.get("selected_call") or {}).get("variant") or "")
    mxfp8 = variant == "mxfp8_group32"
    int8 = entry != "aclnnMatmul" and not mxfp8
    dense_dtype = str((record.get("metadata") or {}).get("input_dtype")
                      or "bf16")
    if dense_dtype not in {"bf16", "fp16", "fp32"}:
        dense_dtype = "bf16"
    quant = (quant_preset("w8a8_mxfp8") if mxfp8 else
             quant_preset("w8a8_int8") if int8 else QuantSpec(dense_dtype))
    dtype = "fp8_e4m3" if mxfp8 else ("int8" if int8 else dense_dtype)
    op = gemm_op(op_id="native_matrix:" + entry, deps=(), stage=0, layer=0,
                 requests=(), m=m, n=n, k=k, quant=quant,
                 category="dense_mlp", phase="prefill")
    bytes_ = record.get("work", {}).get("bytes") or {}
    meta = dict(op.metadata)
    meta.update({key: value for key, value in
                 (record.get("metadata") or {}).items()
                 if key not in {"preparation", "preparation_id"}})
    meta["matrix_builder"] = matrix_builder_inputs(record)
    weight = int(bytes_.get("weight") or 0)
    weight_scale = int(bytes_.get("weight_scale") or 0)
    row_scale = int(bytes_.get("pertoken_scale") or 0)
    meta["packed_weight_bytes"] = float(weight + weight_scale)
    meta["matrix_groups"] = 1
    meta["weight_scale_bytes"] = 2.0
    meta["activation_scale_bytes"] = float(row_scale)
    meta["activation_scale_shape"] = [m]
    meta["weight_scale_shape"] = [n]
    meta["activation_scale_format"] = "fp32"
    meta["weight_scale_format"] = "bf16"
    meta["int8_weight_metadata"] = "scale_only"
    epilogue = (record.get("metadata") or {}).get("matrix_epilogue")
    if epilogue is not None:
        meta["matrix_epilogue"] = dict(epilogue)
    else:
        meta.pop("matrix_epilogue", None)
    mte2 = int(bytes_.get("input") or 0) + weight + weight_scale + row_scale
    vector = int((record.get("work", {}).get("vector") or {}).get("ops") or 0)
    op = __import__("dataclasses").replace(
        op, local_shape=(m, n, k), semantic_shape=(m, n, k),
        padded_shape=(align(m, 16), align(n, 32), align(k, 16)),
        launch_shape=(m, n, k), dtype=dtype,
        flops=float(2 * m * n * k), vector_ops=float(vector),
        hbm_bytes=float(bytes_.get("total") or 0), mte2_bytes=float(mte2),
        mte3_bytes=float(bytes_.get("output") or 0),
        matrix_flops=(((dtype, float(2 * m * n * k)),)),
        metadata=tuple(sorted(meta.items())))
    recipe = TiledMatmul(*tile, slots=slots, template_execution=False)
    return op, recipe


def operator_plan_descriptor(record: Mapping[str, Any], *, tile=(16, 32, 16),
                             slots: int = 2) -> dict[str, Any]:
    """The selected finite program, output handles and kernel ownership."""
    if record.get("status") != "predicted":
        return {"status": record.get("status"), "reasons": list(
            record.get("reasons") or ())}
    geom = dict(record.get("geometry") or {})
    output = (record.get("tensors") or {}).get("output") or {}
    int8 = str(record.get("entry")) != "aclnnMatmul"
    kernels = [{"stage": "load", "owner": "mte2"},
               {"stage": "compute", "owner": "cube"}]
    if int8:
        kernels.extend([{"stage": "scales", "owner": "mte2"},
                        {"stage": "epilogue", "owner": "vector"}])
    kernels.append({"stage": "store", "owner": "mte3"})
    return {
        "status": "predicted",
        "kind": "tiled_matmul",
        "entry": record.get("entry"),
        "variant": (record.get("selected_call") or {}).get("variant"),
        "tile": [int(tile[0]), int(tile[1]), int(tile[2])],
        "slots": int(slots),
        "m": geom.get("m"), "n": geom.get("n"), "k": geom.get("k"),
        "issued": {
            "macs": (record.get("work", {}).get("cube") or {}).get("macs"),
            "flops": (record.get("work", {}).get("cube") or {}).get("flops"),
            "declared_padded_macs": (record.get("work", {}).get("cube") or {}).get(
                "issued_macs"),
            "declared_padded_basis": (record.get("work", {}).get("cube") or {}).get(
                "issued_basis"),
            "program_macs": ((record.get("work", {}).get("cube") or {}).get(
                "macs")),
            "program_basis": "tail_exact_16x32x16_tiled_matmul",
            "mte2_bytes": int((record.get("work", {}).get("bytes") or {}).get("input", 0))
            + int((record.get("work", {}).get("bytes") or {}).get("weight", 0))
            + int((record.get("work", {}).get("bytes") or {}).get("weight_scale", 0))
            + int((record.get("work", {}).get("bytes") or {}).get("pertoken_scale", 0)),
            "mte3_bytes": (record.get("work", {}).get("bytes") or {}).get("output"),
            "hbm_bytes": (record.get("work", {}).get("bytes") or {}).get("total"),
        },
        "output_handles": [{
            "name": "output", "shape": list(output.get("shape") or ()),
            "dtype": output.get("dtype"), "layout": output.get("layout", "ND"),
            "role": "output",
        }],
        "kernels": kernels,
        "matrix_builder": matrix_builder_inputs(record),
        "preparation": (record.get("work", {}).get("preparation")
                        or None),
        "preparation_owner": "producer_or_upstream" if
        (record.get("work", {}).get("preparation")) else None,
    }


def build_operator_plan(record: Mapping[str, Any], *, tile=(16, 32, 16),
                        slots: int = 2):
    """Return the one real ``OperatorPlan`` for a passed matrix ABI record."""
    from mindie_sim.prediction.mechanism import OperatorPlan, Work
    descriptor = operator_plan_descriptor(record, tile=tile, slots=slots)
    if descriptor.get("status") != "predicted":
        raise ValueError("native_matrix_plan_requires_predicted_record")
    op, recipe = _execution_inputs(record, tile=tile, slots=slots)
    bytes_ = record.get("work", {}).get("bytes") or {}
    resources = {
        "hbm": float(bytes_.get("total") or 0),
        "mte2": float(descriptor["issued"]["mte2_bytes"]),
        "mte3": float(descriptor["issued"]["mte3_bytes"]),
    }
    vector = int((record.get("work", {}).get("vector") or {}).get("ops") or 0)
    if vector:
        resources["vector"] = float(vector)
    variant = str((record.get("selected_call") or {}).get("variant") or "")
    dense_dtype = str((record.get("metadata") or {}).get("input_dtype")
                      or "bf16")
    work_key = ("fp8_e4m3" if variant == "mxfp8_group32" else
                (dense_dtype if str(record.get("entry")) == "aclnnMatmul"
                 else "int8"))
    work = Work({work_key: float(op.flops)}, resources)
    metadata = {
        "native_call": dict(record.get("selected_call") or {}),
        "operator_plan": descriptor,
        "output_handles": descriptor["output_handles"],
        "kernels": descriptor["kernels"],
        "preparation": descriptor["preparation"],
    }
    return OperatorPlan(work, lambda context: recipe.lower(context),
                        metadata=metadata, native=True)


def execute_operator_plan(record: Mapping[str, Any], *, tile=(16, 32, 16),
                          slots: int = 2, cores: int = 1,
                          rates: Mapping[str, float] | None = None
                          ) -> dict[str, Any]:
    """Run the selected real KernelProgram on the existing Engine.

    This is the CPU mechanism check: it submits the same finite stages,
    allocations and dependencies that ``build_operator_plan`` exposes to T12;
    it does not execute a dict projection or re-price a different program.
    """
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.program import RecipeContext

    plan = build_operator_plan(record, tile=tile, slots=slots)
    op, _ = _execution_inputs(record, tile=tile, slots=slots)
    vector = int((record.get("work", {}).get("vector") or {}).get("ops") or 0)
    rate_map = {name: float(rate) for name, rate in (rates or {
        "cube": 1e9, "vector": 1e9, "hbm": 1e9, "mte1": 1e9,
        "mte2": 1e9, "mte3": 1e9}).items()}
    names = {name: name for name in rate_map}
    units = {name: (1 if name == "hbm" else cores) for name in rate_map}
    pools = {"l1": 1e9, "l0c": 1e9}
    engine = Engine([Resource(name, rate, units=units[name])
                     for name, rate in rate_map.items()],
                    pools=pools, pool_units={name: cores for name in pools})
    context = RecipeContext(
        op, {"cube": op.flops, **({"vector": vector} if vector else {})},
        names, rate_map, cores,
        resource_units=units,
        pool_names={"l1": "l1", "l0c": "l0c"},
        pool_capacities=pools,
        pool_units={name: cores for name in pools},
    )
    program = plan.program(context)
    done = program.start(engine, "native_matrix", engine.submit("ready"))
    engine.advance(until=(done,))
    return {
        "status": "executed",
        "plan": plan,
        "program": program,
        "context": context,
        "engine": engine,
        "completion_ms": done.time,
        "serviced": dict(engine.serviced),
        "output_handles": plan.metadata["output_handles"],
        "kernels": plan.metadata["kernels"],
        "preparation": plan.metadata["preparation"],
        "footprint_bytes": (record.get("work", {}).get("bytes") or {}).get(
            "total"),
    }
