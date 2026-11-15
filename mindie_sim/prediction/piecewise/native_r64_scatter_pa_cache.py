"""Prepare-only R64 ScatterPaCache indexer cache write.

[Role] Expand the existing R64 A5 single-stream indexer cache write at
  ``vllm_ascend/device/device_op.py:1537``
  (``A5DeviceAdaptor.indexer_quant_scatter``) into the real fused
  ``torch.ops._C_ascend.indexer_compress_epilog_v2`` call and the two
  captured ``ScatterPaCache`` kernel occurrences (FP8 index cache and FP32
  scale cache) for the evaluated token geometry.  The compiled initiator is
  the existing ``I0:kv_access:index_cache_store`` operator; the census role
  label is "I1 quantized cache write".
[Invariants] The source identity is the exact R64 image digest
  ``8d422528...`` and the retained R64 ``device_op.py`` blob.  The selected
  branch is the single-stream ``indexer_quant_scatter`` call; the
  ``indexer_quant_scatter_part1`` multi-stream branch is explicitly rejected
  because this deployment does not declare ``multistream_attention_indexer``.
  Head/width identity, dtype and source lines stay in the binding; the token
  lane and the two cache-write occurrences live only in the occurrence.  The
  indexer read (``QuantLightningIndexer``) is a separate signed call and is
  never charged here.  The fused op's C++ source and ACLNN symbol are not
  retained, so the wrapper-to-kernel mapping, packed cache row bytes and
  cache capacity stay explicit unresolved fields.  No tensor is allocated, no
  pointer or symbol is fabricated, no device launch is claimed and no
  measured clock prices or selects work.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal workload geometry.
"""
from __future__ import annotations

from typing import Any

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
REVISION = "r64/8d422528"
ENTRY = "torch.ops._C_ascend.indexer_compress_epilog_v2"
KERNEL_NAME = "ScatterPaCache"
SOURCE_SYMBOL = "A5DeviceAdaptor.indexer_quant_scatter"
REJECTED_SYMBOL = "A5DeviceAdaptor.indexer_quant_scatter_part1"
SOURCE_LINES = [1537, 1552]
CALL_LINES = [1547, 1552]
REJECTED_LINES = [1556, 1569]
LAYOUT = 2
SELECTED_BRANCH = "single_stream_indexer_quant_scatter"
REJECTED_BRANCH = "multistream_indexer_quant_scatter_part1"
CAPTURE_CASE = "glm51_dp16tp1ep16_pf_1k_c1"
CAPTURE_CSV = (
    "/Volumes/data/mindie-agent-workstreams/"
    "a5-multinode-fleet-rack06-20261107/out/glm51_dp16tp1ep16/"
    "pf_1k_c1/177.25.21.1/dp2_pp0_tp0_dcp0_ep2_rank0_1467_"
    "20261107094934940_ascend_pt/kernel_details.csv.gz"
)
CAPTURE_CSV_SHA_PREFIX = "0223d793"
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": R64_DEVICE_OP_SHA,
    "symbol": SOURCE_SYMBOL,
    "revision": REVISION,
    "state": "task_artifact",
}
# Only immutable binding facts are shared across calls.  Actual attrs, owner,
# state and prepared arguments are rebuilt per occurrence so two layers that
# reuse the same (call_id, tokens) cannot receive the first layer's owner.
_STATIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _descriptor(shape: Any, dtype: Any, layout: Any, role: str, *,
                allocation: str = "external", **extra: Any) -> dict:
    result = {
        "shape": (None if shape is None else
                  [None if item is None else int(item) for item in shape]),
        "dtype": dtype, "layout": layout, "role": role, "pointer": None,
        "allocation": allocation,
    }
    result.update({key: value for key, value in extra.items()
                   if value is not None})
    return result


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The exact witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image") or "")
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    if source_revision_from_oci(image, image_id) != "r64":
        return {}
    return meta


def _cache_dims(compiled: Any, meta: dict[str, Any]) -> dict[str, int] | None:
    """The source-declared index-cache row identity, or ``None``.

    The row width must be the declared index width at one byte (FP8).  A
    different width, a pooled producer or a fixed valid-row declaration is a
    different invocation and stays unbound; a shared-cache consumer does not
    emit this store operator in the existing I0 builder.
    """
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    theta = operator.theta
    extras = dict(theta.extras)
    dim = _int(extras.get("dim"), 0)
    heads = _int(extras.get("heads"), 0)
    kpool = _int(extras.get("kpool"), 0)
    row_bytes = float(theta.find("row_bytes", 0) or 0)
    valid_rows_mode = str(theta.find("valid_rows_mode", "") or "")
    cache_format = str(theta.find("cache_format", "") or "")
    block_size = _int(meta.get("kv_block_size"), 0)
    if dim < 1 or heads < 1 or block_size < 1:
        return None
    if kpool != 1:
        return None
    if valid_rows_mode != "dynamic":
        return None
    if cache_format != "index_dtype":
        return None
    if row_bytes != float(dim):
        return None
    return {"dim": dim, "heads": heads, "row_bytes": int(row_bytes),
            "block_size": block_size, "kpool": kpool}


def _phase(compiled: Any, phase: str) -> str:
    return str(phase or getattr(compiled, "phase", "") or "").lower()


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 indexer ``ScatterPaCache`` cache-write identity or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain.  Non-R64 images, the multi-stream part1 branch, a BF16/absent
    index cache, the separate ``I1:indexer:index_score`` read and every other
    operator return ``None`` so ordinary prediction stays unchanged.
    """
    if compiled is None or str(getattr(compiled, "kind", "") or "") != "indexer_prolog":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if str(getattr(operator, "name", "") or "") != "index_cache_store":
        return None
    if str(getattr(operator, "family", "") or "") != "kv_access":
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    key_dtype = str(meta.get("indexer_kv_dtype") or "").lower()
    if key_dtype == "fp8":
        key_dtype = "fp8_e4m3"
    if key_dtype != "fp8_e4m3":
        return None
    if meta.get("indexer_query_scale_present") is not True:
        return None
    if str(meta.get("indexer_query_scale_dtype") or "").lower() != "fp32":
        return None
    if bool(meta.get("multistream_attention_indexer")):
        return None
    phase_name = _phase(compiled, phase)
    if phase_name and phase_name not in {"prefill", "pf"}:
        return None
    dims = _cache_dims(compiled, meta)
    if dims is None:
        return None
    dim, heads = dims["dim"], dims["heads"]
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        profiler_name=KERNEL_NAME,
        kernel_source_status="kernel_source_not_retained",
        kernel_mapping_status="captured_not_source_linked",
        captured_roles=["indexer_k_cache", "indexer_scale_cache"],
        captured_input_dtypes=["fp8_e4m3", "fp8_e4m3", "int32"],
        captured_scale_input_dtypes=["fp32", "fp32", "int32"],
        captured_output_dtypes=["fp8_e4m3", "fp32"],
        captured_input_slots=6,
        captured_defined_input_slots=3,
        captured_optional_input_state="DT_UNDEFINED",
        captured_case=CAPTURE_CASE,
    )
    typed = {
        "dtype": "fp8_e4m3",
        "layout": "PA_BSND",
        "phase": "prefill",
        "shapes": {
            "update": [1, dim],
            "cache": [1, 1, dim],
            "scale_update": [1],
            "scale_cache": [1, 1, 1],
            "output": [1, 1, dim],
            "scale_output": [1, 1, 1],
        },
        "attrs": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "profiler_name": KERNEL_NAME,
            "entry": ENTRY,
            "layout_value": LAYOUT,
            "cache_format": "index_dtype",
            "indexer_kv_dtype": "fp8_e4m3",
            "scale_cache_present": True,
            "scale_dtype": "fp32",
            "index_heads": heads,
            "index_head_dim": dim,
            "block_size": dims["block_size"],
            "kpool": dims["kpool"],
            "valid_rows_mode": "dynamic",
            "source_family": SOURCE_SYMBOL,
            "selected_branch": SELECTED_BRANCH,
            "rejected_branch": REJECTED_BRANCH,
            "rejected_reason": "multistream_attention_indexer_not_selected",
            "source_lines": list(SOURCE_LINES),
            "call_lines": list(CALL_LINES),
            "rejected_source_lines": list(REJECTED_LINES),
            "deployment_image_id": R64_IMAGE_SHA,
            "deployment_source_revision": "r64",
            "source_commit": R64_SOURCE_COMMIT,
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "kernel_source_status": "kernel_source_not_retained",
            "kernel_mapping_status": "captured_not_source_linked",
            "indexer_read_owner": "I1:indexer:index_score",
            "physical_owner": "I0:kv_access:index_cache_store",
            "input_contract": ["indexer_compress_cache", "x",
                               "slot_mapping"],
            "kernel_argument_order": ["update", "cache", "indices"],
        },
        "rank_width": 4,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, version=REVISION), source, typed,
        (kernel,), (), "I0:kv_access:index_cache_store",
        execution_core="aiv",
        unresolved=(
            "aclnn_symbol_unresolved",
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "packed_indexer_cache_row_bytes_unresolved",
            "indexer_cache_capacity_not_declared",
            "scale_cache_write_not_separately_priced",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only occurrence for the selected indexer cache write.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The primary owner is the existing
    ``I0:kv_access:index_cache_store`` row; the two captured ``ScatterPaCache``
    kernel writes and the fused wrapper's argument bindings are attribution
    records.  Cache capacity and packed row bytes are not declared by the
    normal case and stay ``None``; they are never replaced by the captured
    sample.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("native_type") != KERNEL_NAME:
        return None
    if attrs.get("selected_branch") != SELECTED_BRANCH:
        return None
    if attrs.get("deployment_source_revision") != "r64":
        return None
    if str(getattr(geometry, "phase_name", "") or "").lower() not in {
            "prefill", "pf"}:
        return None
    tokens = _int(getattr(geometry, "effective_tokens", 0), 0)
    if tokens < 1:
        return None
    dim = _int(attrs.get("index_head_dim"), 0)
    heads = _int(attrs.get("index_heads"), 0)
    block_size = _int(attrs.get("block_size"), 0)
    if dim < 1 or heads < 1 or block_size < 1:
        return None
    cache_key = (binding.call_id, tokens)
    static = _STATIC_CACHE.get(cache_key)
    if static is None:
        static = {
            "source": dict(binding.source or {}),
            "kernel": (dict(binding.kernels[0]) if binding.kernels else {}),
        }
        if len(_STATIC_CACHE) >= 32:
            _STATIC_CACHE.clear()
        _STATIC_CACHE[cache_key] = static
    requests = tuple(getattr(geometry, "requests", ()) or ())
    batch = max(1, len(requests))
    operator = getattr(compiled, "operator", None)
    theta_extras = dict(getattr(getattr(operator, "theta", None),
                                "extras", ()) or ())
    actual_shapes = {
        "update": [tokens, 1, dim],
        "cache": [None, 1, 1, dim],
        "scale_update": [tokens, 1, 1],
        "scale_cache": [None, 1, 1, 1],
        "output": [None, 1, 1, dim],
        "scale_output": [None, 1, 1, 1],
    }
    owner_layer = _int(theta_extras.get("owner_layer", -1), -1)
    actual_attrs = {
        **attrs,
        "m": tokens,
        "tokens": tokens,
        "batch": batch,
        "cache_rows": None,
        "scale_cache_rows": None,
        "cache_capacity_status": "not_declared",
        "owner_layer": owner_layer,
        "identity_rebound_per_call": True,
    }
    actual = {"dtype": "fp8_e4m3", "layout": "PA_BSND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "update": _descriptor([tokens, 1, dim], "fp8_e4m3", "PA_BSND",
                              "input"),
        "cache": _descriptor([None, 1, 1, dim], "fp8_e4m3", "PA_BSND",
                             "input_output"),
        "slot_mapping": _descriptor([tokens], "int32", "ND", "input"),
        "scale_update": _descriptor([tokens, 1, 1], "fp32", "PA_BSND",
                                    "input"),
        "scale_cache": _descriptor([None, 1, 1, 1], "fp32", "PA_BSND",
                                   "input_output"),
        "output": _descriptor([None, 1, 1, dim], "fp8_e4m3", "PA_BSND",
                              "output", alias_of="cache",
                              aliasing="in_place"),
        "scale_output": _descriptor([None, 1, 1, 1], "fp32", "PA_BSND",
                                    "output", alias_of="scale_cache",
                                    aliasing="in_place"),
    }
    wrapper_descriptors = {
        "indexer_compress_cache": _descriptor(
            [None, 1, 1, None], "uint8", "packed_indexer_cache",
            "input_output", packed_row_bytes=None,
            evidence="source_view_uint8",
            interpretation="packed k FP8 + scale FP32; split not declared"),
        "x": _descriptor(
            [tokens, 1, dim], None, "ND", "input",
            evidence="captured_kernel_update_shape",
            wrapper_dtype_status="not_retained"),
        "slot_mapping": _descriptor(
            [tokens], "int32", "ND", "input",
            evidence="source_argument_and_captured_kernel_index_dtype"),
    }
    wrapper_descriptors["indexer_compress_cache"]["packed_row_bytes"] = None
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({
        "captured_roles": ["indexer_k_cache", "indexer_scale_cache"],
        "captured_input_dtypes": ["fp8_e4m3", "fp8_e4m3", "int32"],
        "captured_output_dtypes": ["fp8_e4m3", "fp32"],
        "captured_case": CAPTURE_CASE,
        "mapping_status": "captured_not_source_linked",
    })
    source = dict(binding.source or {})
    prepare = {
        "schema": "mindie-sim/scatter-pa-cache-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_commit": R64_SOURCE_COMMIT,
        "source_commit_status": "inherited_same_device_op_blob_record",
        "source_lines": list(SOURCE_LINES),
        "call_lines": list(CALL_LINES),
        "selected_branch": SELECTED_BRANCH,
        "rejected_branch": REJECTED_BRANCH,
        "rejected_reason": "multistream_attention_indexer_not_selected",
        "rejected_source_lines": list(REJECTED_LINES),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "wrapper_arguments": {
            "indexer_compress_cache": "cache",
            "x": "update",
            "slot_mapping": "slot_mapping",
            "layout": LAYOUT,
        },
        "wrapper_argument_order": ["indexer_compress_cache", "x",
                                   "slot_mapping", "layout"],
        "wrapper_keywords": {
            "indexer_compress_cache": "cache",
            "x": "update",
            "slot_mapping": "slot_mapping",
            "layout": LAYOUT,
        },
        "kernel_arguments": {
            "update": "update", "cache": "cache", "indices": "slot_mapping",
        },
        "kernel_argument_order": ["update", "cache", "indices"],
        "kernel_occurrences": {
            "indexer_k_cache": {
                "kernel": KERNEL_NAME,
                "arguments": ["update", "cache", "indices"],
                "input_shapes": {
                    "update": [tokens, 1, dim],
                    "cache": [None, 1, 1, dim],
                    "indices": [tokens],
                },
                "input_dtypes": ["fp8_e4m3", "fp8_e4m3", "int32"],
                "input_formats": ["ND", "ND", "ND"],
                "optional_input_slots": 3,
                "optional_input_state": "DT_UNDEFINED",
                "output_shape": [None, 1, 1, dim],
                "output_dtype": "fp8_e4m3",
                "output_format": "ND",
                "aliasing": "cache_in_place",
                "evidence": "captured_original_profiler_row",
            },
            "indexer_scale_cache": {
                "kernel": KERNEL_NAME,
                "arguments": ["update", "cache", "indices"],
                "input_shapes": {
                    "update": [tokens, 1, 1],
                    "cache": [None, 1, 1, 1],
                    "indices": [tokens],
                },
                "input_dtypes": ["fp32", "fp32", "int32"],
                "input_formats": ["ND", "ND", "ND"],
                "optional_input_slots": 3,
                "optional_input_state": "DT_UNDEFINED",
                "output_shape": [None, 1, 1, 1],
                "output_dtype": "fp32",
                "output_format": "ND",
                "aliasing": "scale_cache_in_place",
                "evidence": "captured_original_profiler_row",
            },
        },
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "wrapper_descriptors": wrapper_descriptors,
        "captured_example": {
            "case": CAPTURE_CASE,
            "source_csv": CAPTURE_CSV,
            "retained_orig_sha_prefix": CAPTURE_CSV_SHA_PREFIX,
            "retained_orig_sha_note": (
                "retained record prefix, not a full file digest"),
            "optional_input_slots": "three DT_UNDEFINED/NULL slots",
            "prefill_rows_per_cache": 78,
            "k_update_shape": [1024, 1, 128],
            "k_cache_shape": [522240, 1, 1, 128],
            "k_slot_shape": [1024],
            "k_dtype": "fp8_e4m3",
            "scale_update_shape": [1024, 1, 1],
            "scale_cache_shape": [522240, 1, 1, 1],
            "scale_slot_shape": [1024],
            "scale_dtype": "fp32",
            "captured_cache_rows": 522240,
            "note": "captured_sample_evidence_not_a_predictive_dimension",
        },
        "mapping": {
            "status": "captured_not_source_linked",
            "wrapper_source_retained": True,
            "fused_cpp_source_retained": False,
            "kernel_source_retained": False,
            "kernel_name_evidence": "original_profiler_row",
            "reason": ("indexer_compress_epilog_v2 C++ source and ACLNN "
                       "symbol are not retained; the captured ScatterPaCache "
                       "row is not promoted to a source-declared child"),
        },
        "deployment": {
            "image_id": R64_IMAGE_SHA,
            "revision": REVISION,
            "selection": ("exact R64 image digest + explicit "
                          "indexer_kv_dtype=fp8_e4m3 + present fp32 "
                          "query scale + multistream_attention_indexer "
                          "false"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": R64_SOURCE_COMMIT,
            "source_commit_status": "inherited_same_device_op_blob_record",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "selected_branch": SELECTED_BRANCH,
            "rejected_branch": REJECTED_BRANCH,
        },
        "match": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "profiler_name": KERNEL_NAME,
            "entry": ENTRY,
            "call_id": binding.call_id,
            "source_revision": REVISION,
            "source_sha256": R64_DEVICE_OP_SHA,
            "source_lines": list(SOURCE_LINES),
            "call_lines": list(CALL_LINES),
            "selected_branch": SELECTED_BRANCH,
            "rejected_branch": REJECTED_BRANCH,
            "m": tokens,
            "index_heads": heads,
            "index_head_dim": dim,
            "block_size": block_size,
            "cache_rows": None,
            "cache_dtype": "fp8_e4m3",
            "scale_cache_present": True,
            "scale_dtype": "fp32",
            "layout": "PA_BSND",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "complete": False,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": ("fused_custom_op_cpp_source_and_acl_nn_symbol_"
                       "not_retained"),
            "owner_kernel": KERNEL_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "packed_indexer_cache_row_bytes_unresolved",
            "indexer_cache_capacity_not_declared",
            "scale_cache_write_not_separately_priced",
        ],
        "unresolved": [
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "packed_indexer_cache_row_bytes_unresolved",
            "indexer_cache_capacity_not_declared",
            "scale_cache_write_not_separately_priced",
            "execution_rank_not_selected",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_complete_native_execution",
        ],
    }
    from .native_gdn import state_effect
    prepare["state_slots"] = list(ABI_STATE_SLOTS)
    prepare["state_effects"] = [
        state_effect(
            order=1, direction="write", slot=ABI_STATE_SLOTS[0],
            tensor="cache", rows=tokens, dtype="fp8_e4m3",
            role="index_k_cache_write", cache_rows=None,
            index_dtype="int32", index_tensor="slot_mapping",
            row_bytes=float(dim), kernel_occurrence="indexer_k_cache"),
        state_effect(
            order=2, direction="write", slot=ABI_STATE_SLOTS[1],
            tensor="scale_cache", rows=tokens, dtype="fp32",
            role="index_scale_cache_write", cache_rows=None,
            index_dtype="int32", index_tensor="slot_mapping",
            row_bytes=4.0, kernel_occurrence="indexer_scale_cache"),
    ]
    prepare["representation"] = {
        "cache_row_bytes": None,
        "cache_row_bytes_status": "packed_row_bytes_unresolved",
        "slot_values": "runtime_output_position_indices",
        "slot_values_are_capacity": False,
        "k_and_scale_caches_are_independent_values": True,
    }
    prepare["boundaries"] = {
        "parent_api": {
            "entry": ENTRY,
            "packed_inputs": ["indexer_compress_cache", "x",
                              "slot_mapping", "layout"],
            "complete": False,
            "reason": ["packed_row_bytes_unresolved",
                       "parent_child_input_mapping_not_retained",
                       "scale_split_not_retained"],
            "child_tensors_are_not_parent_inputs": True,
        },
        "child_kernels": {
            "names": ["indexer_k_cache", "indexer_scale_cache"],
            "kernel": KERNEL_NAME,
            "count": 2,
            "input_reads_modelled_separately": True,
            "scope": "child_kernel",
        },
        "predicted_scope": "prepare_only_metadata",
    }
    prepare["identity_binding"] = {
        "owner_layer": owner_layer,
        "tokens": tokens,
        "cache_rows": None,
        "rebound_per_call": True,
        "static_reused": ["source", "kernel"],
        "no_cross_owner_replay": True,
    }
    return {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": "fp8_e4m3",
            "layout": "PA_BSND",
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }


# ---------------------------------------------------------------------------
# Independent native-input slice (A5 attention/index state leaf).
#
# The fused ``indexer_compress_epilog_v2`` wrapper is invoked directly with the
# two real cache tensors, the packed slot mapping and the K/scale updates.  The
# slice maps that input to the shared ``kv_access.index_store`` program, prices
# the K FP8 traffic plus the FP32 scale traffic, and emits the two cache writes
# as ordered state effects.  Cache capacity is the real tensor row axis; the
# captured 522240-row sample is never substituted for it.
# ---------------------------------------------------------------------------

ABI_ROLE = "index_cache_write_pair"
CHILD_ENTRY = KERNEL_NAME
PARENT_ENTRY = ENTRY
ABI_STATE_SLOTS = ("indexer_k_cache", "indexer_scale_cache")
PARENT_GAP = (
    "packed_row_bytes_unresolved",
    "parent_child_input_mapping_not_retained",
    "scale_split_not_retained",
    "indexer_compress_epilog_v2_cpp_source_not_retained",
)


def _child_source() -> dict:
    source = dict(SOURCE)
    source.update({
        "source_commit": R64_SOURCE_COMMIT,
        "selected_branch": SELECTED_BRANCH,
        "rejected_branch": REJECTED_BRANCH,
        "kernel_identity": "captured_profiler_row_only",
    })
    return source


def abi_spec() -> Any:
    """Declared independent ABI of the two known ScatterPaCache child kernels.

    This is deliberately the child-kernel boundary, not the fused parent API:
    the request carries the already-split FP8 K update/cache and FP32 scale
    update/cache plus the slot indices.  The parent packed API is declared by
    :func:`parent_abi_spec` and stays an explicit gap.
    """
    from .native_calls import (
        NativeAbiCallSpec, NativeAbiTensorSpec, NativeAbiVariant)
    kernel = {
        "name": KERNEL_NAME, "kind": "vector",
        "path": SOURCE["path"], "sha256": SOURCE["sha256"],
        "evidence": "measured", "profiler_name": KERNEL_NAME,
        "kernel_source_status": "kernel_source_not_retained",
        "kernel_mapping_status": "captured_not_source_linked",
        "captured_case": CAPTURE_CASE,
        "boundary": "child_kernel",
    }
    return NativeAbiCallSpec(
        entry=CHILD_ENTRY, namespace="cann_kernel", aclnn_symbol=None,
        api_version=REVISION, source=_child_source(),
        source_revision=REVISION, image_id=R64_IMAGE_SHA,
        kernels=(kernel,),
        tensors=(
            NativeAbiTensorSpec("update", "input", None,
                                "tokens,1,index_head_dim"),
            NativeAbiTensorSpec("cache", "inout", None,
                                "cache_rows,1,1,index_head_dim"),
            NativeAbiTensorSpec("indices", "input", None, "tokens"),
            NativeAbiTensorSpec("scale_update", "input", None,
                                "tokens,1,1"),
            NativeAbiTensorSpec("scale_cache", "inout", None,
                                "cache_rows,1,1,1"),
        ),
        attrs=("index_head_dim", "index_heads", "layout", "block_size",
               "kpool", "valid_rows_mode", "storage_dtype",
               "scale_storage_dtype", "index_dtype"),
        optional_attrs=(),
        variants=(NativeAbiVariant(
            ABI_ROLE, "I0:kv_access:index_cache_store",
            "I0:kv_access:index_cache_store", ("index_cache_store",),
            "kernel", "one_to_many"),),
        execution_core="aiv", arch="A5", implemented=True,
        unresolved=(
            "aclnn_symbol_unresolved",
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "parent_child_input_mapping_not_retained",
        ))


def parent_abi_spec() -> Any:
    """The fused parent API with its real packed inputs and mapping gap.

    The parent consumes a packed ``indexer_compress_cache`` view and produces
    both child caches internally; the retained source does not declare the
    packed row layout or how K/scale are split.  It is therefore declared as
    a known, unimplemented boundary rather than predicted from child tensors.
    """
    from .native_calls import (
        NativeAbiCallSpec, NativeAbiTensorSpec, NativeAbiVariant)
    kernel = {
        "name": KERNEL_NAME, "kind": "vector",
        "path": SOURCE["path"], "sha256": SOURCE["sha256"],
        "evidence": "measured", "profiler_name": KERNEL_NAME,
        "kernel_source_status": "kernel_source_not_retained",
        "kernel_mapping_status": "captured_not_source_linked",
        "boundary": "parent_api",
    }
    return NativeAbiCallSpec(
        entry=PARENT_ENTRY, namespace="torch", aclnn_symbol=None,
        api_version=REVISION, source=_child_source(),
        source_revision=REVISION, image_id=R64_IMAGE_SHA,
        kernels=(kernel,),
        tensors=(
            NativeAbiTensorSpec("indexer_compress_cache", "inout", "uint8",
                                "packed_rows,1,1,packed_row_bytes"),
            NativeAbiTensorSpec("x", "input", None, "tokens,1,index_head_dim"),
            NativeAbiTensorSpec("slot_mapping", "input", None, "tokens"),
        ),
        attrs=("layout", "index_head_dim", "index_heads", "block_size",
               "kpool"),
        optional_attrs=("packed_row_bytes",),
        variants=(NativeAbiVariant(
            ABI_ROLE, "I0:kv_access:index_cache_store",
            "I0:kv_access:index_cache_store", ("index_cache_store",),
            "call", "one_to_many"),),
        execution_core="aiv", arch="A5", implemented=False,
        gap=PARENT_GAP,
        unresolved=(
            "aclnn_symbol_unresolved",
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            *PARENT_GAP))


def _index_bytes(index_dtype: str) -> int:
    return 8 if index_dtype == "int64" else 4


def _write_effects(*, tokens: int, cache_rows: int, dim: int,
                   storage_dtype: str, scale_dtype: str,
                   index_dtype: str) -> tuple:
    from .native_gdn import state_effect
    return (
        state_effect(
            order=1, direction="write", slot=ABI_STATE_SLOTS[0],
            tensor="cache", rows=tokens, dtype=storage_dtype,
            role="index_k_cache_write", cache_rows=cache_rows,
            index_dtype=index_dtype, index_tensor="indices",
            row_bytes=float(dim), kernel_occurrence="indexer_k_cache",
            in_place_cache_update=True),
        state_effect(
            order=2, direction="write", slot=ABI_STATE_SLOTS[1],
            tensor="scale_cache", rows=tokens, dtype=scale_dtype,
            role="index_scale_cache_write", cache_rows=cache_rows,
            index_dtype=index_dtype, index_tensor="indices",
            row_bytes=4.0, kernel_occurrence="indexer_scale_cache",
            in_place_cache_update=True),
    )


def _abi_program(inv: Any, supply: Any, policy: Any,
                 phase: str, machine: Any = None,
                 trace_limit: Any = None,
                 execution: Any = None) -> dict[str, Any] | None:
    from fractions import Fraction
    from mindie_sim.prediction.native_binding import NativeAbiRegion
    from mindie_sim.prediction.piecewise.compiler import CompiledOperator
    from mindie_sim.prediction.piecewise.geometry import (
        RequestGeometry, WorkloadGeometry)
    from mindie_sim.prediction.piecewise.native_gdn import (
        LeafAbiUnsupported, leaf_attr, leaf_dtype,
        leaf_int, leaf_launch_attr, leaf_require_attr_matches_tensor_dtype,
        leaf_shape)
    from mindie_sim.schema.subgraphs import (
        BoundOperator, OperatorTemplate, StaticSignature)

    if phase not in {"", "prefill"}:
        raise LeafAbiUnsupported(f"scatter_pa_abi_phase_not_prefill:{phase}")
    dim = leaf_int(inv, "index_head_dim")
    heads = leaf_int(inv, "index_heads")
    layout = leaf_int(inv, "layout")
    block_size = leaf_int(inv, "block_size")
    kpool = leaf_int(inv, "kpool")
    if dim < 1 or heads < 1 or block_size < 1:
        raise LeafAbiUnsupported("scatter_pa_abi_cache_dims_invalid")
    if layout != LAYOUT:
        raise LeafAbiUnsupported(f"scatter_pa_abi_layout_unknown:{layout}")
    if kpool != 1:
        raise LeafAbiUnsupported(f"scatter_pa_abi_kpool_unsupported:{kpool}")
    if str(leaf_attr(inv, "valid_rows_mode")) != "dynamic":
        raise LeafAbiUnsupported("scatter_pa_abi_valid_rows_mode_not_dynamic")
    storage_dtype = leaf_require_attr_matches_tensor_dtype(
        inv, "storage_dtype", ("update", "cache"))
    scale_dtype = leaf_require_attr_matches_tensor_dtype(
        inv, "scale_storage_dtype", ("scale_update", "scale_cache"))
    index_dtype = str(leaf_attr(inv, "index_dtype")).lower()
    if storage_dtype != "fp8_e4m3":
        raise LeafAbiUnsupported(
            f"scatter_pa_abi_storage_dtype_unsupported:{storage_dtype}")
    if scale_dtype != "fp32":
        raise LeafAbiUnsupported(
            f"scatter_pa_abi_scale_dtype_unsupported:{scale_dtype}")
    if index_dtype not in {"int32", "int64"}:
        raise LeafAbiUnsupported(
            f"scatter_pa_abi_index_dtype_unsupported:{index_dtype}")
    if leaf_dtype(inv, "indices") != index_dtype:
        raise LeafAbiUnsupported(
            "scatter_pa_abi_index_dtype_mismatch:"
            f"{leaf_dtype(inv, 'indices')}!={index_dtype}")
    update_shape = leaf_shape(inv, "update")
    cache_shape = leaf_shape(inv, "cache")
    indices_shape = leaf_shape(inv, "indices")
    scale_update_shape = leaf_shape(inv, "scale_update")
    scale_cache_shape = leaf_shape(inv, "scale_cache")
    if (len(update_shape) != 3 or update_shape[1] != 1
            or update_shape[2] != dim):
        raise LeafAbiUnsupported("scatter_pa_abi_update_shape_mismatch")
    tokens = update_shape[0]
    if indices_shape != [tokens]:
        raise LeafAbiUnsupported("scatter_pa_abi_indices_shape_mismatch")
    if scale_update_shape != [tokens, 1, 1]:
        raise LeafAbiUnsupported("scatter_pa_abi_scale_update_shape_mismatch")
    if (len(cache_shape) != 4 or cache_shape[1:] != [1, 1, dim]
            or len(scale_cache_shape) != 4
            or scale_cache_shape[1:] != [1, 1, 1]
            or cache_shape[0] != scale_cache_shape[0]):
        raise LeafAbiUnsupported("scatter_pa_abi_cache_shape_mismatch")
    cache_rows = cache_shape[0]

    def _child(kind: str, *, elems: int, dtype: str,
               update_key: str, cache_key: str,
               static_update: list, static_cache: list,
               update_shape: list, cache_shape: list,
               row_bytes: float) -> dict:
        theta = StaticSignature(
            "state_scatter", matrix=(("value_heads", 1),),
            extras=(("dk", elems), ("dv", 1), ("state_dtype", dtype),
                    ("index_dtype", index_dtype),
                    ("source_kernel", KERNEL_NAME),
                    ("scatter_branch", "captured_child_kernel"),
                    ("state_layout", "V,K"),
                    ("block_size", block_size), ("num_warps", 8)))
        bound = BoundOperator(
            OperatorTemplate("scatter_pa_cache", "state_scatter",
                             "state_scatter"), theta)
        geometry = WorkloadGeometry(
            tuple(RequestGeometry(1, 0, phase="prefill")
                  for _ in range(tokens)), phase="prefill")
        compiled = CompiledOperator(bound, supply, policy, kind="indexer_prolog")
        cost = compiled.evaluate(geometry)
        attrs = {
            "kernel_dtype": dtype, "index_dtype": index_dtype,
            "index_head_dim": dim, "index_heads": heads,
            "layout_value": layout, "block_size": block_size,
            "kpool": kpool, "valid_rows_mode": "dynamic",
            "row_bytes": row_bytes, "kernel": KERNEL_NAME,
            "kernel_occurrence": kind,
            "boundary": "child_kernel",
        }
        typed = {
            "dtype": dtype, "layout": "ND", "phase": "prefill",
            "shapes": {update_key: static_update,
                       cache_key: static_cache},
            "attrs": attrs,
            "occurrence_attrs": {"tokens": "positive_int",
                                 "cache_rows": "positive_int"},
            "rank_width": 4,
        }
        occurrence = {
            "dtype": dtype, "layout": "ND", "phase": "prefill",
            "shapes": {update_key: list(update_shape),
                       cache_key: list(cache_shape),
                       "indices": [tokens]},
            "attrs": {**attrs, "tokens": tokens,
                      "cache_rows": cache_rows},
        }
        return {
            "cost": cost, "compiled": compiled, "geometry": geometry,
            "region": NativeAbiRegion(
                op_id="I0:kv_access:index_cache_store", family="state_scatter",
                recipe="state_scatter", source_boundary="kernel",
                dtype=dtype, layout="ND",
                metadata=tuple(sorted(attrs.items()))),
            "typed": typed, "occurrence": occurrence,
            "effects": (), "mode": kind,
            "op_id": "I0:kv_access:index_cache_store",
            "kind": "indexer_prolog", "phase": "prefill", "charge": "call",
            "source_boundary": "kernel",
            "owner": "I0:kv_access:index_cache_store",
            "unresolved": ("scatter_pa_cache_kernel_source_not_retained",),
        }

    k_part = _child(
        "indexer_k_cache", elems=dim, dtype=storage_dtype,
        update_key="update", cache_key="cache",
        static_update=[1, dim], static_cache=[1, 1, dim],
        update_shape=update_shape, cache_shape=cache_shape,
        row_bytes=float(dim))
    scale_part = _child(
        "indexer_scale_cache", elems=1, dtype=scale_dtype,
        update_key="scale_update", cache_key="scale_cache",
        static_update=[1, 1], static_cache=[1, 1, 1],
        update_shape=scale_update_shape, cache_shape=scale_cache_shape,
        row_bytes=4.0)
    effects = _write_effects(
        tokens=tokens, cache_rows=cache_rows, dim=dim,
        storage_dtype=storage_dtype, scale_dtype=scale_dtype,
        index_dtype=index_dtype)
    k_part["effects"] = (effects[0],)
    scale_part["effects"] = (effects[1],)
    idx_bytes = _index_bytes(index_dtype)
    k_read = tokens * dim + tokens * idx_bytes
    scale_read = tokens * 4 + tokens * idx_bytes
    k_write = tokens * dim
    scale_write = tokens * 4
    aggregate_derived = (
        ("indexer_k_read_bytes", Fraction(k_read)),
        ("indexer_k_index_bytes", Fraction(tokens * idx_bytes)),
        ("indexer_k_write_bytes", Fraction(k_write)),
        ("indexer_scale_read_bytes", Fraction(scale_read)),
        ("indexer_scale_index_bytes", Fraction(tokens * idx_bytes)),
        ("indexer_scale_write_bytes", Fraction(scale_write)),
        ("total_read_bytes", Fraction(k_read + scale_read)),
        ("total_write_bytes", Fraction(k_write + scale_write)),
        ("cache_capacity_rows", Fraction(cache_rows)),
        ("cache_allocated_bytes", Fraction(0)),
    )
    cost = None
    scale_bytes = float(scale_write)
    scale_bytes = float(scale_write)
    access = {
        "boundary": "child_kernel",
        "k_read_bytes": k_read,
        "k_index_bytes": tokens * idx_bytes,
        "k_write_bytes": k_write,
        "scale_read_bytes": scale_read,
        "scale_index_bytes": tokens * idx_bytes,
        "scale_write_bytes": scale_write,
        "total_read_bytes": k_read + scale_read,
        "total_write_bytes": k_write + scale_write,
        "cache_capacity_rows": cache_rows,
        "cache_row_payload_bytes": dim,
        "cache_reallocated": False,
        "in_place_cache_update": True,
        "index_dtype": index_dtype,
        "parent_packed_inputs_excluded": True,
    }
    typed = {
        "dtype": storage_dtype, "layout": "ND", "phase": "prefill",
        "shapes": {"update": update_shape, "cache": cache_shape,
                   "scale_update": scale_update_shape,
                   "scale_cache": scale_cache_shape},
        "attrs": {
            "index_head_dim": dim, "index_heads": heads,
            "layout_value": layout, "block_size": block_size,
            "kpool": kpool, "valid_rows_mode": "dynamic",
            "storage_dtype": storage_dtype,
            "scale_storage_dtype": scale_dtype,
            "index_dtype": index_dtype,
            "boundary": "child_kernel",
        },
        "occurrence_attrs": {"tokens": "positive_int",
                             "cache_rows": "positive_int"},
        "rank_width": 4,
    }
    return {
        "phase": "prefill", "kind": "indexer_prolog",
        "aggregate_name": "ScatterPaCache.pair",
        "aggregate_derived": aggregate_derived,
        "aggregate_materialization": (
            ("indexer_k_cache", k_write),
            ("indexer_scale_cache", scale_write),
            ("cache_allocated", 0),
        ),
        "environment": {"tokens": tokens, "cache_rows": cache_rows,
                        "index_head_dim": dim, "kernel_rows": tokens * 2},
        "region": NativeAbiRegion(
            op_id="I0:kv_access:index_cache_store", family="state_scatter",
            recipe="state_scatter", source_boundary="kernel",
            dtype=storage_dtype, layout="ND",
            metadata=tuple(sorted({
                "index_head_dim": dim, "index_heads": heads,
                "layout_value": layout, "block_size": block_size,
                "row_bytes": float(dim),
                "boundary": "child_kernel",
            }.items()))),
        "typed": typed,
        "occurrence": {
            "dtype": storage_dtype, "layout": "ND", "phase": "prefill",
            "shapes": dict(typed["shapes"]),
            "attrs": {**typed["attrs"], "tokens": tokens,
                      "cache_rows": cache_rows},
        },
        "effects": effects,
        "parts": (k_part, scale_part),
        "unresolved": (
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
        ),
        "program": {
            "mode": ABI_ROLE, "family": "state_scatter",
            "common_program": "state_scatter.sequence_rows",
            "members": ["index_cache_store"], "phase": "prefill",
            "boundary": "child_kernel",
            "state_slots": list(ABI_STATE_SLOTS),
            "state_effects": list(effects),
            "kernel_occurrences": ["indexer_k_cache",
                                   "indexer_scale_cache"],
            "access": access,
            "materialization": {
                "indexer_k_cache_bytes": k_write,
                "indexer_scale_cache_bytes": scale_write,
                "cache_allocated_bytes": 0,
                "note": ("in-place cache word updates; capacity comes from "
                         "the supplied cache tensor, not the captured sample"),
            },
            "parent_api": {
                "entry": PARENT_ENTRY,
                "status": "declared_unimplemented",
                "reason": list(PARENT_GAP),
                "packed_inputs": ["indexer_compress_cache", "x",
                                  "slot_mapping", "layout"],
                "child_tensors_are_not_parent_inputs": True,
            },
            "formula": " ; ".join(
                f"{part['mode']}: {part['cost'].formula}"
                for part in (k_part, scale_part)),
            "work": {
                key: str(sum((Fraction(dict(part["cost"].work).get(key, 0))
                              for part in (k_part, scale_part)),
                             Fraction(0)))
                for key in {key for part in (k_part, scale_part)
                            for key, _ in part["cost"].work}
            },
            "derived": {key: str(value)
                        for key, value in aggregate_derived},
        },
    }


def simulate_independent(request: Any, **kwargs: Any) -> dict:
    """Predict the two child cache writes from split native inputs."""
    from .native_gdn import simulate_leaf
    return simulate_leaf(request, abi_spec(), _abi_program, **kwargs)


def simulate_parent(request: Any, **kwargs: Any) -> dict:
    """Report the fused parent API's known mapping gap, not a prediction."""
    from .native_gdn import simulate_leaf
    return simulate_leaf(request, parent_abi_spec(), lambda *args: None,
                         **kwargs)
