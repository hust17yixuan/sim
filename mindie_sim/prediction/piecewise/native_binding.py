"""Multi-to-many binding of captured ACLNN tensor slots to native programs.

[Role] Declare the real ordered positional slot schema of an ACLNN operator,
  bind a capture's present/absent slots to the selected native leaf by role and
  position (never by a shape table), and provide the generic BF16 MatMulV3
  physical mapping that does not carry an R31-only source pin.
[Invariants] A slot position is never deleted or reordered; an absent optional
  slot stays absent.  Declared tensor names may have capture-side aliases
  (``key_cache``/``key``, ``sequence_lengths``/``actual_seq_lengths_kv``);
  aliases are normalized before the ordered binding is checked.  Geometry is
  derived from the declared shapes, dtypes, attributes and the target
  hardware's physical rates; no per-shape duration, lookup table or default
  transpose is invented.
[Timeouts & Locks] Bounded metadata arithmetic over the ordered slots.
[Side-Effect Policy] Pure records; the catalogue installer only replaces the
  in-process declared spec objects and never mutates a capture or a device.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .native_matrix_abi import (
    ABI_SCHEMA, NON_CLAIMS, align, gaps, interpretation_reasons, result,
    selected, slot_view_reasons, status, with_views,
)

#: Capture-side aliases for canonical declared tensor slots.  A request may
#: name the same physical slot by its profiler name or by its ACLNN name; the
#: ordered positional binding uses the canonical name.
TENSOR_ALIASES: dict[str, str] = {
    "key_cache": "key",
    "value_cache": "value",
    "sequence_lengths": "actual_seq_lengths_kv",
    "seq_lengths": "actual_seq_lengths_kv",
    "kv_lengths": "actual_seq_lengths_kv",
    "attention_out": "output",
    "attn_out": "output",
    "lse": "softmax_lse",
    "softmax_lse_out": "softmax_lse",
    # A shared torch QBMM slot may carry the pre-materialized NZ weight; the
    # leaf selects its NZ branch from that slot's actual rank/layout.
    "weight_nz": "weight",
}

#: Capture-side attribute aliases.  ``None`` marks an attribute that only
#: spells an explicitly absent optional argument and is dropped.
ATTR_ALIASES: dict[str, str | None] = {
    "transpose_a": "transpose_x1",
    "transpose_b": "transpose_x2",
    "trans_a": "transpose_x1",
    "trans_b": "transpose_x2",
    "x2_format": "weight_layout",
    "num_key_value_heads": "num_kv_heads",
    "softmax_scale": "scale",
    "return_lse": "return_softmax_lse",
}

_ABSENT_ATTR_VALUES = {"absent", "none", "null", "not_present", "np"}


@dataclass(frozen=True)
class NativeAbiSlot:
    """One ordered ACLNN tensor slot declaration."""

    name: str
    role: str
    dtype: str | None = None
    layout: str = "ND"
    dims: str = ""
    optional: bool = True
    side: str = "input"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "dtype": self.dtype,
                "dims": self.dims, "optional": self.optional,
                "layout": self.layout, "side": self.side}


@dataclass(frozen=True)
class NativeAbiSchema:
    """One captured native operator's ordered ABI schema."""

    operator: str
    entry: str
    namespace: str
    aclnn_symbol: str | None
    slots: tuple[NativeAbiSlot, ...]
    attrs: tuple[str, ...]
    optional_attrs: tuple[str, ...]
    kernels: tuple[dict[str, Any], ...]
    leaf: str
    dispatch: str
    execution_core: str
    aliases: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    source: Mapping[str, Any] | None = None
    preserve_attrs: bool = True
    preserve_conditions: bool = True
    variant_name: str = "native_call"
    relation: str = "one_to_one"


# ---------------------------------------------------------------------------
# Declared ordered ACLNN slot schemas
# ---------------------------------------------------------------------------

_MM_SLOTS = (
    NativeAbiSlot("x1", "input", "bf16", "ND", "M,K", optional=False),
    NativeAbiSlot("x2", "input", "bf16", "ND", "K,N or N,K", optional=False),
    NativeAbiSlot("bias", "input", "bf16", "ND", "N", optional=True),
    NativeAbiSlot("output", "output", "bf16", "ND", "M,N", optional=False,
                  side="output"),
)

_QBMM_SLOTS = (
    NativeAbiSlot("x", "input", "int8", "ND", "M,K", optional=False),
    NativeAbiSlot("weight", "weight", "int8", "ND", "K,N or N,K",
                  optional=False),
    NativeAbiSlot("weight_scale", "scale", None, "ND", "N", optional=False),
    NativeAbiSlot("pertoken_scale", "scale", None, "ND", "M",
                  optional=False),
    NativeAbiSlot("bias", "input", None, "ND", "N", optional=True),
    NativeAbiSlot("output", "output", "bf16", "ND", "M,N", optional=False,
                  side="output"),
)

#: CANN FusedInferAttentionScore positional inputs.  The capture retains all
#: 31 ordered positions; every position that is not part of the ordinary
#: paged-MHA/GQA path stays an explicit absent slot.
_FIA_INPUT_SLOTS = (
    ("query", "input", None, "ND", "batch,heads,q,head_dim", False),
    ("key", "input", None, "ND", "blocks,kv_heads,block_size,head_dim", False),
    ("value", "input", None, "ND", "blocks,kv_heads,block_size,head_dim",
     False),
    ("pse_shift", "input", None, "ND", "", True),
    ("atten_mask", "input", None, "ND", "", True),
    ("actual_seq_lengths", "input", "int64", "ND", "batch", True),
    ("actual_seq_lengths_kv", "input", "int64", "ND", "batch", False),
    ("dequant_scale1", "input", None, "ND", "", True),
    ("quant_scale1", "input", None, "ND", "", True),
    ("dequant_scale2", "input", None, "ND", "", True),
    ("quant_scale2", "input", None, "ND", "", True),
    ("quant_offset2", "input", None, "ND", "", True),
    ("antiquant_scale", "input", None, "ND", "", True),
    ("antiquant_offset", "input", None, "ND", "", True),
    ("block_table", "input", "int32", "ND", "batch,max_blocks", False),
    ("query_padding_size", "input", None, "ND", "", True),
    ("kv_padding_size", "input", None, "ND", "", True),
    ("key_antiquant_scale", "input", None, "ND", "", True),
    ("key_antiquant_offset", "input", None, "ND", "", True),
    ("value_antiquant_scale", "input", None, "ND", "", True),
    ("value_antiquant_offset", "input", None, "ND", "", True),
    ("key_shared_prefix", "input", None, "ND", "", True),
    ("value_shared_prefix", "input", None, "ND", "", True),
    ("actual_shared_prefix_len", "input", None, "ND", "", True),
    ("query_rope", "input", None, "ND", "", True),
    ("key_rope", "input", None, "ND", "", True),
    ("key_rope_antiquant_scale", "input", None, "ND", "", True),
    ("dequant_scale_key", "input", None, "ND", "", True),
    ("dequant_scale_query", "input", None, "ND", "", True),
    ("key_sink", "input", None, "ND", "", True),
    ("value_sink", "input", None, "ND", "", True),
)

_FIA_SLOTS = tuple(
    NativeAbiSlot(name, role, dtype, layout, dims, optional=optional)
    for name, role, dtype, layout, dims, optional in _FIA_INPUT_SLOTS
) + (
    NativeAbiSlot("output", "output", "bf16", "ND",
                  "batch,heads,q,head_dim", optional=False, side="output"),
    NativeAbiSlot("softmax_lse", "output", "fp32", "ND",
                  "batch,heads,q", optional=True, side="output"),
)

_FIA_ATTRS = (
    "input_layout", "num_heads", "num_kv_heads", "block_size", "scale",
    "sparse_mode", "pre_tokens", "next_tokens", "actual_seq_lengths_kv",
    "causal", "mask",
)

_MM_SCHEMA = NativeAbiSchema(
    operator="MatMulV3",
    entry="torch.matmul",
    namespace="torch",
    aclnn_symbol="aclnnMatmul",
    slots=_MM_SLOTS,
    attrs=(),
    optional_attrs=("transpose_x1", "transpose_x2", "bias", "cube_math_type",
                    "output_dtype"),
    kernels=({"name": "MatMulV3", "role": "cube", "host_op": "MatMulV3Common",
              "evidence": "captured_native_operator"},),
    leaf="native_binding",
    dispatch="generic_matrix_abi",
    execution_core="aic",
    aliases=("MatMulV3", "aclnnMatmul_MatMulV3Common_MatMulV3"),
    unresolved=("a5_bf16_matmul_kernel_binary_not_retained",
                "a5_bf16_matmul_cube_tile_schedule_declared_by_geometry"),
)

_QBMM_SCHEMA = NativeAbiSchema(
    operator="QuantBatchMatmulV3",
    entry="torch_npu.npu_quant_matmul",
    namespace="torch",
    aclnn_symbol=None,
    slots=_QBMM_SLOTS,
    attrs=(),
    optional_attrs=("group_size", "output_dtype", "scale_dtype",
                    "pertoken_scale_dtype", "weight_scale_storage_format",
                    "weight_scale_storage_bytes", "scale_group",
                    "weight_layout", "bias", "offset", "x2_format"),
    kernels=({"name": "QuantBatchMatmulV3", "role": "cube",
              "evidence": "captured_native_operator"},),
    leaf="native_qbmm",
    dispatch="matrix_predict_abi",
    execution_core="mix",
    aliases=("QuantBatchMatmulV3",
             "aclnnQuantMatmulV5_QuantBatchMatmulV3_QuantBatchMatmulV3"),
    unresolved=("quant_batch_matmul_cann_symbol_not_retained",
                "quant_batch_matmul_kernel_selection_unresolved"),
)

_FIA_SCHEMA = NativeAbiSchema(
    operator="FusedInferAttentionScore",
    entry="torch_npu.npu_fused_infer_attention_score",
    namespace="torch",
    aclnn_symbol="aclnnFusedInferAttentionScore",
    slots=_FIA_SLOTS,
    attrs=(),
    optional_attrs=_FIA_ATTRS,
    kernels=({"name": "FusedInferAttentionScore", "role": "mix",
              "evidence": "captured_native_operator"},),
    leaf="attention_decode",
    dispatch="paged_fia_build",
    execution_core="mix",
    aliases=("FusedInferAttentionScore",),
    unresolved=("fia_cann_symbol_not_retained",
                "fia_kernel_binary_selection_unresolved"),
    variant_name="ordinary_paged_fia",
)

def _elementwise_schema(operator: str, entry: str, kernel: str,
                        slots: tuple[NativeAbiSlot, ...],
                        aliases: tuple[str, ...] = ()) -> NativeAbiSchema:
    """One declared standalone row-elementwise ACLNN operator."""
    return NativeAbiSchema(
        operator=operator, entry=entry, namespace="aclnn",
        aclnn_symbol=entry, slots=slots, attrs=(),
        optional_attrs=("dtype", "alpha", "beta", "round_mode", "scalar"),
        kernels=({"name": kernel, "role": "vector",
                  "evidence": "captured_native_operator"},),
        leaf="native_elementwise", dispatch="vector_prepare_predict",
        execution_core="aiv", aliases=(kernel, *aliases),
        unresolved=("elementwise_kernel_binary_selection_unresolved",))


_ELEMENTWISE_SCHEMAS = (
    _elementwise_schema(
        "Add", "aclnnAdd", "Add",
        (NativeAbiSlot("x1", "input", None, "ND", "rows,width", False),
         NativeAbiSlot("x2", "input", None, "ND", "rows,width or scalar", False),
         NativeAbiSlot("output", "output", None, "ND", "rows,width", False,
                       side="output"))),
    _elementwise_schema(
        "Mul", "aclnnMul", "Mul",
        (NativeAbiSlot("x1", "input", None, "ND", "rows,width", False),
         NativeAbiSlot("x2", "input", None, "ND", "rows,width or scalar", False),
         NativeAbiSlot("output", "output", None, "ND", "rows,width", False,
                       side="output"))),
    _elementwise_schema(
        "Cast", "aclnnCast", "Cast",
        (NativeAbiSlot("input", "input", None, "ND", "rows,width", False),
         NativeAbiSlot("output", "output", None, "ND", "rows,width", False,
                       side="output"))),
    _elementwise_schema(
        "Gelu", "aclnnGelu", "Gelu",
        (NativeAbiSlot("input", "input", None, "ND", "rows,width", False),
         NativeAbiSlot("output", "output", None, "ND", "rows,width", False,
                       side="output"))),
    _elementwise_schema(
        "Muls", "aclnnMuls", "Muls",
        (NativeAbiSlot("input", "input", None, "ND", "rows,width", False),
         NativeAbiSlot("output", "output", None, "ND", "rows,width", False,
                       side="output"))),
    # A standalone SiLU/Swish activation row is one elementwise AIV launch; the
    # captured kernel is named ``Swish`` and has the same row ABI as ``Gelu``.
    _elementwise_schema(
        "Swish", "aclnnSwish", "Swish",
        (NativeAbiSlot("input", "input", None, "ND", "rows,width", False),
         NativeAbiSlot("output", "output", None, "ND", "rows,width", False,
                       side="output")),
        aliases=("aclnnSilu_SiluAiCore_Swish", "Silu", "SiLU")),
)

_MX_QUANT_SCHEMA = NativeAbiSchema(
    operator="DynamicMxQuant", entry="aclnnDynamicMxQuant",
    namespace="aclnn", aclnn_symbol="aclnnDynamicMxQuant",
    slots=(NativeAbiSlot("x", "input", None, "ND", "rows,width", False),
           NativeAbiSlot("y", "output", None, "ND", "rows,width", False,
                         side="output"),
           NativeAbiSlot("scale", "output", "fp8_e8m0", "ND",
                         "rows,ceil(width/64),2", False, side="output")),
    attrs=(), optional_attrs=("group_size", "scale_group", "dst_dtype"),
    kernels=({"name": "DynamicMxQuant", "role": "vector",
              "evidence": "captured_native_operator"},),
    leaf="native_mx_quant", dispatch="vector_prepare_predict",
    execution_core="aiv", aliases=("DynamicMxQuant",),
    unresolved=("mx_quant_kernel_binary_selection_unresolved",))

_SCHEMAS = (_MM_SCHEMA, _QBMM_SCHEMA, _FIA_SCHEMA, *_ELEMENTWISE_SCHEMAS,
            _MX_QUANT_SCHEMA)


# ---------------------------------------------------------------------------
# Slot/attribute alias normalization
# ---------------------------------------------------------------------------

def normalize_tensor_aliases(tensors: Any, declared: Any = None) -> Any:
    """Return the same ordered slots under their canonical declared names.

    ``declared`` restricts a rename to a slot the selected schema actually
    declares, so an unrelated ABI that happens to use the word
    ``sequence_lengths`` keeps its own name.
    """
    if not isinstance(tensors, Mapping):
        return tensors
    allowed = None if declared is None else {str(name) for name in declared}
    renamed: dict[str, Any] = {}
    for name, item in tensors.items():
        source = str(name)
        canonical = TENSOR_ALIASES.get(source, source)
        if allowed is not None:
            # An explicitly declared slot keeps its own name even when it is
            # also an alias target elsewhere (e.g. WeightNz declares both
            # ``weight`` and ``weight_nz``).
            if source in allowed:
                canonical = source
            elif canonical not in allowed:
                canonical = source
        if canonical in renamed and canonical != str(name):
            # The canonical name was supplied explicitly; never let an alias
            # overwrite it (the ordered slot keeps the declared position).
            continue
        renamed[canonical] = item
    return renamed


def normalize_attr_aliases(attrs: Any, declared: Any = None) -> Any:
    """Fold capture-side attribute aliases without inventing a value."""
    if not isinstance(attrs, Mapping):
        return attrs
    allowed = None if declared is None else {str(name) for name in declared}
    normalized: dict[str, Any] = {}
    for name, value in attrs.items():
        key = str(name)
        target = ATTR_ALIASES.get(key, key)
        if target is None:
            continue
        if allowed is not None and target not in allowed:
            target = key
        if isinstance(value, str) and value.strip().lower() in _ABSENT_ATTR_VALUES:
            continue
        if target in normalized:
            continue
        normalized[target] = value
    return normalized


def clean_conditions(conditions: Any) -> Any:
    """Keep declared conditions but drop capture-phase spelling.

    ``operator_only`` names the capture boundary, not the attention/recurrent
    phase the physical plan must select.  It is removed only when no real
    prefill/decode phase is present; the original request facts remain in the
    prediction's ``request_conditions``.
    """
    if not isinstance(conditions, Mapping):
        return conditions
    result = dict(conditions)
    if str(result.get("phase") or "") not in {"prefill", "decode"}:
        result.pop("phase", None)
    return result


# ---------------------------------------------------------------------------
# Declared ABI spec construction and catalogue installation
# ---------------------------------------------------------------------------

def build_native_abi_spec(schema: NativeAbiSchema):
    """Construct one ``NativeAbiCallSpec`` with the ordered slot objects."""
    from .native_calls import NativeAbiCallSpec, NativeAbiVariant
    variant = NativeAbiVariant(
        schema.variant_name, "native:" + schema.entry,
        "native:" + schema.entry, (), "call", schema.relation)
    return NativeAbiCallSpec(
        entry=schema.entry, namespace=schema.namespace,
        aclnn_symbol=schema.aclnn_symbol, api_version=None,
        source=dict(schema.source or {}), source_revision=None, image_id=None,
        kernels=tuple(dict(kernel) for kernel in schema.kernels),
        tensors=tuple(schema.slots), attrs=tuple(schema.attrs),
        optional_attrs=tuple(schema.optional_attrs), variants=(variant,),
        execution_core=schema.execution_core, arch="A5", implemented=True,
        gap=(), unresolved=tuple(schema.unresolved), leaf=schema.leaf,
        dispatch=schema.dispatch, preserve_tensor_fields=True,
        preserve_extra_attrs=schema.preserve_attrs,
        preserve_extra_conditions=schema.preserve_conditions)


def declared_specs() -> dict[str, Any]:
    """The three ordered schemas keyed by canonical API entry."""
    return {schema.entry: build_native_abi_spec(schema) for schema in _SCHEMAS}


def operator_aliases() -> dict[str, str]:
    """Capture operator/API spellings mapped to their canonical entry."""
    aliases: dict[str, str] = {}
    for schema in _SCHEMAS:
        for alias in (schema.entry, schema.operator, *schema.aliases):
            aliases[str(alias)] = schema.entry
    # A kernel_details.csv row names the physical kernel, not the Python or
    # ACLNN caller.  These spellings are the captured kernel names observed in
    # the 4,240-record corpus; each maps onto an already-declared ordered ABI.
    aliases.update({
        "RmsNorm": "torch_npu.npu_rms_norm",
        "AddRmsNorm": "aclnnAddRmsNorm",
        "DynamicQuant": "torch_npu.npu_dynamic_quant",
        "DynamicQuantV2": "DynamicQuantV2",
        "DynamicMxQuant": "aclnnDynamicMxQuant",
        "Slice": "Slice",
        # The captured 9-slot ordered row (x, weight, bias, scale, offset,
        # antiquant_scale, antiquant_offset, group_list, per_token_scale) is
        # its own declared entry; the W4A4 source adapter keeps the V4 name.
        "GroupedMatmul": "GroupedMatmul",
        "GroupedMatmulV4": "torch_npu.npu_grouped_matmul",
        "SwiGlu": "torch_npu.npu_swiglu",
        "TransposeBatchMatMul": "torch_npu.npu_transpose_batchmatmul",
        "TransposeBatchMatmul": "torch_npu.npu_transpose_batchmatmul",
        "ScatterPaKvCache": "ScatterPaKvCache",
        "MatMulV2": "torch.matmul",
        "FusedInferAttentionScore":
            "torch_npu.npu_fused_infer_attention_score",
    })
    return aliases


def install_catalogue() -> dict[str, Any]:
    """Install the ordered slot schemas into the in-process ABI catalogue.

    The canonical declared entry set stays exactly the catalogue's existing
    entries: QBMM and paged FIA are *replaced* in place by their ordered-slot
    declaration, while the generic MatMulV3 aliases resolve to an additional
    declared spec that is intentionally not injected into the fixed inventory
    tuple (so an inventory count is a declaration count, not an alias count).
    Re-running the installer is idempotent.
    """
    from . import native_calls
    if getattr(native_calls, "_NATIVE_ORDERED_SLOT_CATALOGUE", False):
        return {
            "installed": True,
            "entries": sorted(declared_specs()),
            "aliases": sorted(operator_aliases()),
        }
    replacements = declared_specs()
    base = native_calls.a5_native_abi_specs()
    merged = tuple(
        replacements.get(spec.entry, spec) for spec in base)
    merged_by_entry = {spec.entry: spec for spec in merged}
    merged_by_entry.update({
        entry: spec for entry, spec in replacements.items()
        if entry not in merged_by_entry})
    aliases = operator_aliases()
    original_spec = native_calls.native_abi_spec

    def ordered_native_abi_spec(name: Any):
        if isinstance(name, str):
            entry = aliases.get(name)
            if entry is not None:
                return merged_by_entry.get(entry)
            for spec in merged:
                if name in {spec.entry, spec.aclnn_symbol}:
                    return spec
            return None
        return original_spec(name)

    native_calls.a5_native_abi_specs = lambda: merged
    native_calls.native_abi_spec = ordered_native_abi_spec
    native_calls._NATIVE_ORDERED_SLOT_CATALOGUE = True
    return {
        "installed": True,
        "entries": sorted(replacements),
        "aliases": sorted(aliases),
        "canonical_entries": [spec.entry for spec in merged],
    }


# ---------------------------------------------------------------------------
# Generic BF16 MatMulV3 physical mapping (no R31-only source pin)
# ---------------------------------------------------------------------------

def _version_view(version: Any) -> tuple[str | None, str | None, str | None]:
    """Return declared (api, source revision, image) without inventing one."""
    if isinstance(version, str):
        return version, None, None
    if not isinstance(version, Mapping):
        return None, None, None
    api = version.get("api") or version.get("api_version")
    if api is None and version.get("cann") is not None:
        api = str(version["cann"])
    source = (version.get("source") or version.get("source_revision")
              or version.get("revision"))
    image = version.get("image_id") or version.get("image")
    return (None if api is None else str(api),
            None if source is None else str(source),
            None if image is None else str(image))


def _orientation(x1: Mapping[str, Any], x2: Mapping[str, Any],
                 output: Mapping[str, Any], attrs: Mapping[str, Any]
                 ) -> tuple[int | None, int | None, int | None,
                            bool | None, bool | None, list[str]]:
    """Resolve M/N/K from declared shapes and the real transpose attributes."""
    m, n = output["shape"]
    first, second = x1["shape"], x2["shape"]
    declared_x1 = attrs.get("transpose_x1")
    declared_x2 = attrs.get("transpose_x2")
    if declared_x1 is None:
        x1_m = first[0] == m
        x1_k = first[1] == m
        if x1_m == x1_k:
            return None, None, None, None, None, [
                "missing:attrs.transpose_x1"]
        transpose_x1 = x1_k
    else:
        transpose_x1 = bool(declared_x1)
        if not (first[1] == m if transpose_x1 else first[0] == m):
            return None, None, None, None, None, ["mismatch:tensors.x1.shape"]
    k = first[0] if transpose_x1 else first[1]
    if declared_x2 is None:
        x2_k = second[0] == k
        x2_n = second[1] == k
        if x2_k == x2_n:
            return None, None, None, None, None, [
                "missing:attrs.transpose_x2"]
        transpose_x2 = x2_n
    else:
        transpose_x2 = bool(declared_x2)
    if transpose_x2:
        if second[1] != k or second[0] != n:
            return None, None, None, None, None, ["mismatch:tensors.x2.shape"]
    elif second[0] != k or second[1] != n:
        return None, None, None, None, None, ["mismatch:tensors.x2.shape"]
    return m, n, k, transpose_x1, transpose_x2, []


def _tensor(tensors: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    raw = tensors.get(name)
    if not isinstance(raw, Mapping) or raw.get("present", True) is False:
        return None
    shape = raw.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    return {"name": name, "role": str(raw.get("role") or "input"),
            "shape": [int(value) for value in shape],
            "dtype": str(raw.get("dtype") or ""),
            "layout": str(raw.get("layout") or "ND")}


def predict_bf16_matmul(tensors: Mapping[str, Any], *,
                        attrs: Mapping[str, Any] | None = None,
                        version: Any = None,
                        conditions: Mapping[str, Any] | None = None
                        ) -> dict[str, Any]:
    """Map one A5 BF16 MatMulV3 ABI to geometry, physical work and a call.

    The Cube tile is resolved from the declared M/N/K and the target tile
    schedule, never from a captured duration or a sample-shape table.  The
    padded K wave is explicit so an unaligned contraction (K=257) is priced as
    whole Cube tiles rather than a fractional one.
    """
    attrs = dict(attrs or {})
    conditions = dict(conditions or {})
    reasons: list[str] = []
    api, source_revision, image_id = _version_view(version)
    x1 = _tensor(tensors, "x1")
    x2 = _tensor(tensors, "x2")
    output = _tensor(tensors, "output")
    _canonical_dtype = {"bf16": "bf16", "bfloat16": "bf16",
                        "fp16": "fp16", "float16": "fp16",
                        "fp32": "fp32", "float32": "fp32"}
    for name, row in (("x1", x1), ("x2", x2), ("output", output)):
        if row is None:
            reasons.append(f"missing:tensors.{name}")
            continue
        if len(row["shape"]) != 2:
            reasons.append(f"mismatch:tensors.{name}.rank")
        if row["dtype"].lower() not in _canonical_dtype:
            reasons.append(f"mismatch:tensors.{name}.dtype")
    if (x1 is not None and x2 is not None and output is not None
            and x1["dtype"].lower() in _canonical_dtype
            and x2["dtype"].lower() in _canonical_dtype
            and output["dtype"].lower() in _canonical_dtype):
        act_dtype = _canonical_dtype[x1["dtype"].lower()]
        weight_dtype = _canonical_dtype[x2["dtype"].lower()]
        out_dtype = _canonical_dtype[output["dtype"].lower()]
        if act_dtype != weight_dtype:
            reasons.append("unmodeled:mixed_activation_weight_dtype")
    else:
        act_dtype, weight_dtype, out_dtype = "bf16", "bf16", "bf16"
    dtype_bytes = {"bf16": 2, "fp16": 2, "fp32": 4}[act_dtype]
    out_bytes = {"bf16": 2, "fp16": 2, "fp32": 4}[out_dtype]
    if attrs.get("bias") is not None:
        reasons.append("unmodeled:attrs.bias")
    # A present bias is a distinct epilogue operand, not a silently ignored
    # optional slot; an absent bias declaration stays absent.
    raw_bias = (tensors or {}).get("bias")
    if (isinstance(raw_bias, Mapping)
            and raw_bias.get("present", True) is not False):
        reasons.append("unmodeled:tensors.bias")
    if attrs.get("cube_math_type") is not None:
        reasons.append("unmodeled:attrs.cube_math_type")
    current = status(reasons)
    invocation = {"tensors": dict(tensors), "attrs": attrs,
                  "version": {"api": api, "source": source_revision,
                              "image_id": image_id},
                  "conditions": conditions}
    if current != "predicted":
        return result(
            status_value=current, entry="aclnnMatmul", reasons=reasons,
            invocation=invocation, selected_call=None, geometry=None,
            work=None, tensors={}, scale={}, metadata={}, accumulator=None,
            regions=[])
    m, n, k, transpose_x1, transpose_x2, orient_gaps = _orientation(
        x1, x2, output, attrs)
    reasons.extend(orient_gaps)
    if m is None:
        return result(
            status_value=status(reasons), entry="aclnnMatmul",
            reasons=reasons, invocation=invocation, selected_call=None,
            geometry=None, work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    if m < 1 or n < 1 or k < 1:
        reasons.append("mismatch:tensors.geometry_nonpositive")
        return result(
            status_value=status(reasons), entry="aclnnMatmul",
            reasons=reasons, invocation=invocation, selected_call=None,
            geometry=None, work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    for name, row, logical, storage, view_dtype in (
            ("x1", x1, {"shape": [m, k]},
             {"shape": ([k, m] if transpose_x1 else [m, k])}, act_dtype),
            ("x2", x2, None,
             {"shape": ([n, k] if transpose_x2 else [k, n])}, weight_dtype),
            ("output", output, {"shape": [m, n]}, {"shape": [m, n]},
             out_dtype)):
        reasons.extend(slot_view_reasons(
            row, name,
            logical=(None if logical is None else {
                **logical, "dtype": view_dtype, "layout": "ND"}),
            storage={**storage, "dtype": view_dtype, "layout": "ND"},
            device={**storage, "dtype": view_dtype, "layout": "ND"}))
    if reasons:
        return result(
            status_value=status(reasons), entry="aclnnMatmul",
            reasons=reasons, invocation=invocation, selected_call=None,
            geometry=None, work=None, tensors={}, scale={}, metadata={},
            accumulator=None, regions=[])
    tile = dict(conditions.get("matrix_tile") or {})
    tile_m = int(tile.get("m") or 16)
    tile_n = int(tile.get("n") or 32)
    tile_k = int(tile.get("k") or 16)
    waves = (math.ceil(m / tile_m) * math.ceil(n / tile_n)
             * math.ceil(k / tile_k))
    issued_macs = waves * tile_m * tile_n * tile_k
    macs = m * n * k
    x1_bytes = m * k * dtype_bytes
    x2_bytes = n * k * dtype_bytes
    output_bytes = m * n * out_bytes
    total_bytes = x1_bytes + x2_bytes + output_bytes
    x1 = with_views(x1, storage_shape=list(x1["shape"]),
                    storage_dtype=act_dtype, storage_layout="ND",
                    interpret_dtype=act_dtype)
    x1["views"]["logical"] = {"shape": [m, k], "dtype": act_dtype,
                              "layout": "ND"}
    x2 = with_views(x2, storage_shape=[n, k] if transpose_x2 else [k, n],
                    storage_dtype=weight_dtype, storage_layout="ND",
                    interpret_dtype=weight_dtype)
    x2["views"]["logical"] = {"shape": [k, n], "dtype": weight_dtype,
                              "layout": "ND"}
    output = with_views(output, storage_shape=[m, n],
                        storage_dtype=out_dtype, storage_layout="ND",
                        interpret_dtype=out_dtype)
    geometry = {
        "m": m, "n": n, "k": k,
        "padded_m": align(m, tile_m), "padded_n": align(n, tile_n),
        "padded_k": align(k, tile_k),
        "tile": [tile_m, tile_n, tile_k],
        "cube_waves": waves,
        "issued_macs": issued_macs,
        "transpose_x1": bool(transpose_x1),
        "transpose_x2": bool(transpose_x2),
        "weight_storage_shape": [n, k] if transpose_x2 else [k, n],
        "basis": "declared_shapes_attributes_and_generic_cube_tile",
    }
    work = {
        "cube": {"macs": macs, "flops": 2 * macs,
                 "issued_macs": issued_macs,
                 "issued_flops": 2 * issued_macs,
                 "issued_basis": "declared_mnk_tiled_cube_waves",
                 "tile": [tile_m, tile_n, tile_k],
                 "waves": waves},
        "vector": {"ops": 0, "basis": "no_separate_vector_epilogue"},
        "bytes": {"input": x1_bytes, "weight": x2_bytes,
                  "output": output_bytes, "total": total_bytes,
                  "basis": "call_boundary_tensor_footprint"},
    }
    members = tuple(conditions.get("members") or ())
    relation = "many_to_one_fused" if len(members) > 1 else "one_to_one"
    call = selected(
        entry="aclnnMatmul", namespace="aclnn",
        aclnn_symbol="aclnnMatmul", api_version=api,
        source_revision=source_revision, image_id=image_id, source=None,
        kernels=[{"name": "MatMulV3", "role": "cube",
                  "host_op": "MatMulV3Common", "acl_api": "aclnnMatmul",
                  "evidence": "captured_native_operator"}],
        execution_core=None, relation=relation, members=members,
        tensors=[x1, x2, output],
        attrs={"transpose_x1": bool(transpose_x1),
               "transpose_x2": bool(transpose_x2),
               "weight_storage_shape": ([n, k] if transpose_x2 else [k, n]),
               "input_dtypes": [act_dtype, weight_dtype, out_dtype],
               "input_slots": ["x1", "x2", "output"],
               "offset": "not_present"},
        regions=["projection"], variant="dense_" + act_dtype + "_matmul_generic")
    call["unresolved"] = [
        "a5_bf16_matmul_kernel_binary_not_retained",
        "a5_bf16_matmul_cube_tile_declared_from_geometry",
    ]
    metadata = {
        "transpose_x1": bool(transpose_x1),
        "transpose_x2": bool(transpose_x2),
        "weight_storage_shape": [n, k] if transpose_x2 else [k, n],
        "weight_logical_shape": [k, n],
        "host_op": "MatMulV3Common",
        "profiler_name": "aclnnMatmul_MatMulV3Common_MatMulV3",
        "execution_core_status": "not_retained",
        "input_dtypes": [act_dtype, weight_dtype, out_dtype],
        "input_dtype": act_dtype, "weight_dtype": weight_dtype,
        "output_dtype": out_dtype, "dtype_bytes": dtype_bytes,
        "output_dtype_bytes": out_bytes,
        "tile": [tile_m, tile_n, tile_k],
        "cube_waves": waves,
        "issued_macs": issued_macs,
    }
    accumulator = {
        "dtype": "fp32", "shape": [m, n], "elements": m * n,
        "bytes": m * n * 4, "abi_tensor": False, "on_chip": True,
        "basis": "dense_matmul_internal_accumulator",
        "residency_status": "requires_selected_matrix_program",
        "source_status": "mechanism_inferred_not_source_declared",
    }
    regions = [{
        "id": "projection", "relation": relation, "rows": m, "cols": n,
        "inner": k, "macs": macs, "outputs": ["output"],
        "members": list(members),
    }]
    return result(
        status_value="predicted", entry="aclnnMatmul", reasons=[],
        invocation=invocation, selected_call=call, geometry=geometry,
        work=work, tensors={"x1": x1, "x2": x2, "output": output},
        scale={}, metadata=metadata, accumulator=accumulator,
        regions=regions)


def operator_plan(record, *, tile=(16, 32, 16), slots=2):
    """The generic matrix leaf's selected finite operator plan."""
    from .native_matrix_abi import build_operator_plan
    return build_operator_plan(record, tile=tile, slots=slots)


# The schema objects are data declarations.  Importing this module installs
# them once so a caller that asks the shared catalogue for a captured operator
# (``torch.matmul``/``MatMulV3``) sees the ordered-slot spec without having to
# know the installer exists; re-import/re-install is idempotent.
__all__ = [
    "ATTR_ALIASES", "TENSOR_ALIASES", "NativeAbiSchema", "NativeAbiSlot",
    "build_native_abi_spec", "declared_specs", "install_catalogue",
    "clean_conditions", "normalize_attr_aliases",
    "normalize_tensor_aliases", "operator_plan",
    "predict_bf16_matmul",
]

AUTO_INSTALL = install_catalogue()
