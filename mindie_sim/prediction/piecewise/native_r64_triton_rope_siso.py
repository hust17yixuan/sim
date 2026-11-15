"""Prepare-only R64 SFA indexer query/key ``_triton_rope_siso`` invocation.

[Role] Expand the source-selected R64 ``AscendSFAIndexerBackend`` rotary step
  into the real in-place ``_triton_rope_siso`` launch for one evaluated prefill
  token geometry.  The direct caller is
  ``vllm_ascend/attention/indexer.py`` (``forward_k`` for the shared key and
  ``forward`` for the query); the normal SFA invoker is
  ``vllm_ascend/attention/sfa_v1.py``.  The host wrapper is
  ``rope_forward_triton_siso`` in ``vllm_ascend/ops/triton/rope.py``.
[Invariants] Only the exact R64 image digest, the exact retained
  ``indexer.py``/``rope.py`` blobs and the producer-declared
  ``rope_implementation="triton_siso"`` callable are admitted.  Head count,
  head width, rotary width, Neox/GPT-J style, tensor dtype and the applicable
  phase are read from the evaluated callable/config, never from the enrolled
  32/128/64 sample or a blanket phase.  Token rows live only in the occurrence;
  the static rotary geometry and in-place output alias live in the binding.
  The kernel receives already-selected ``cos``/``sin`` rows, never a position
  cache.  No tensor is allocated, no pointer is fabricated, no device launch is
  claimed, no compiled cubin is claimed and no measurement selects the recipe.
  The separate query and key launches are distinct native calls; each row is
  charged once by its own existing operator plan.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal geometry; unknown source/ABI/config stays absent or unresolved.
"""
from __future__ import annotations

from typing import Any

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_IMAGE_TAG = (
    "swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:"
    "nightly-ci-releases-v0.32.0rc-a5-openeuler-cann9.2.1-20261101-r64")
R64_SOURCE_COMMIT = "4926e0a22613e66c9bddf4141951845b6397eca7"
REVISION = "r64/8d422528"
ENTRY = "rope_forward_triton_siso"
KERNEL_NAME = "_triton_rope_siso"
SOURCE_IMPLEMENTATION = "triton_siso"
APPLICABLE_PHASES = frozenset({"prefill", "decode"})
SOURCE_DTYPES = frozenset({"bf16", "fp16", "fp32"})
ROLE_QUERY = "query"
ROLE_KEY = "key"

# Retained R64 source blobs.  The direct caller, wrapper and SFA invoker were
# independently fetched from the deployed image; the exact sha256 set resolves
# the local source commit above.  The 32/128/64 prefill call is an enrolled
# observation, not a production geometry gate.
CALLER_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/attention/indexer.py",
    "sha256": "99b7e507c2ba6f8f47d04b051e647f51e6b18b23ecadfc839cc351d81b785942",
    "symbol": "AscendSFAIndexerBackend.forward/forward_k",
    "revision": REVISION,
    "state": "task_artifact",
}
WRAPPER_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/triton/rope.py",
    "sha256": "82422c46f93c0962673ad51bc77fdd5caef412f45f2ff8b20bbeb20a4407f64b",
    "symbol": "rope_forward_triton_siso",
    "revision": REVISION,
    "state": "task_artifact",
}
INVOKER_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/attention/sfa_v1.py",
    "sha256": "82eeef0272977cc2bf0c870699eba005ae71236b1eef7c4b71fe67303a621e68",
    "symbol": "AscendSFAImpl.forward",
    "revision": REVISION,
    "state": "task_artifact",
}
SOURCE_ANCHORS = {
    "vllm_ascend/attention/indexer.py":
        CALLER_SOURCE["sha256"],
    "vllm_ascend/ops/triton/rope.py":
        WRAPPER_SOURCE["sha256"],
    "vllm_ascend/attention/sfa_v1.py":
        INVOKER_SOURCE["sha256"],
    "vllm_ascend/attention/mla_v1.py":
        "55cdabc1bcb85a16905094b03d3eeabeebb108f26e16f6f55e22ad71934cbf45",
    "vllm_ascend/device/device_op.py":
        "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d",
    "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py":
        "38de2275cf33ced653b8398e98a07f483489dc8e7353b2b3c238a5ff00782f79",
}
CALLER_LINES = {
    ROLE_QUERY: [485, 488],
    ROLE_KEY: [338, 343],
}
KERNEL_LINES = [327, 400]
WRAPPER_LINES = [663, 738]
LAUNCH_LINES = [713, 732]
INVOKER_LINES = [1987, 2004]
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _image_digest(meta: dict[str, Any]) -> str:
    value = str(meta.get("image_id") or meta.get("deployment_image_id")
                or meta.get("oci_image_digest") or "")
    if value.lower().startswith("sha256:"):
        value = value.split(":", 1)[1]
    return value.lower()


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The exact R64 image only; geometry and phase are derived downstream."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    if _image_digest(meta) != R64_IMAGE_SHA:
        return {}
    return meta


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _sibling_index_geometry(compiled: Any) -> tuple[int, int] | None:
    """Read the evaluated indexer head/width from the existing score sibling."""
    for sibling in getattr(compiled, "siblings", ()) or ():
        template = getattr(sibling, "template", None)
        name = str(getattr(sibling, "name", "")
                   or getattr(template, "name", "") or "")
        if name != "index_score":
            continue
        theta = getattr(sibling, "theta", None)
        heads = dict(getattr(theta, "heads", ()) or ())
        count = _int(heads.get("heads"))
        dim = _int(heads.get("dim"))
        if count > 0 and dim > 0:
            return count, dim
    return None


def _source_geometry(compiled: Any, operator: Any, name: str,
                     meta: dict[str, Any]) -> dict[str, Any] | None:
    """Derive the evaluated callable's rotary geometry, not the sample values.

    The producer attaches the actual ``indexer.py`` callable facts:
    implementation, head count/width, ``qk_rope_head_dim``, style and dtype.
    The enrolled 32/128/64 prefill observation is deliberately not a default or
    a gate here.
    """
    extras = dict(getattr(operator.theta, "extras", ()) or ())
    matrix = dict(getattr(operator.theta, "matrix", ()) or ())
    role = str(extras.get("index_rope_role") or
               (ROLE_QUERY if name == "index_rope" else ROLE_KEY))
    if role not in {ROLE_QUERY, ROLE_KEY}:
        return None
    if name == "index_rope" and role != ROLE_QUERY:
        return None
    if name == "index_k_rope" and role != ROLE_KEY:
        return None
    if str(extras.get("rope_implementation") or "") != SOURCE_IMPLEMENTATION:
        return None

    dtype = str(extras.get("rope_dtype") or "")
    dim = _int(extras.get("index_dim"), 0)
    rope = _int(extras.get("qk_rope_head_dim"), 0)
    style = _bool(extras.get("is_neox_style"))
    if (dtype not in SOURCE_DTYPES or dim < 1 or rope < 2
            or rope > dim or rope % 2 or style is None):
        return None

    sibling = _sibling_index_geometry(compiled)
    if sibling is not None and sibling[1] != dim:
        return None

    elements = _int(matrix.get("elements"))
    heads = _int(extras.get("index_heads"), 0)
    if role == ROLE_QUERY:
        if heads < 1 and sibling is not None:
            heads = sibling[0]
        if heads < 1 and elements and dim > 0 and elements % dim == 0:
            heads = elements // dim
        if heads < 1 or (elements and elements != heads * dim):
            return None
    else:
        # ``indexer.forward_k`` builds ``k_li.view(-1, 1, head_dim)``; the
        # producer normally declares that one-head fact, and this fallback is
        # the source callable, not the enrolled query sample.
        if heads < 1:
            heads = 1
        if elements and elements != dim:
            return None

    query_heads = _int(extras.get("index_query_heads"), 0)
    key_heads = _int(extras.get("index_key_heads"), 0)
    if role == ROLE_QUERY:
        if query_heads < 1:
            query_heads = heads
        if key_heads < 1:
            key_heads = 1
        if query_heads != heads:
            return None
    else:
        if key_heads < 1:
            key_heads = heads
        if key_heads != heads or query_heads < 1:
            return None
    return {"role": role, "heads": heads, "dim": dim, "rope_dim": rope,
            "is_neox_style": style, "dtype": dtype,
            "query_heads": query_heads, "key_heads": key_heads}


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 indexer rotary call identity, or ``None``.

    The existing normal row is ``I0:elementwise:index_rope`` (the query launch).
    The key launch is the separate ``index_k_rope`` row added by the narrow
    builder attachment.  The census label ``I1 query/key rotary caller`` names
    the indexer caller module, not the product's I1 score/top-k subgraph.
    Both prefill and decode are applicable phases for this callable.  Non-R64
    images, absent producer geometry/implementation attrs, other operator rows
    and unsupported tensor dtypes return ``None`` so ordinary prediction is
    unchanged.
    """
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    name = str(getattr(operator, "name", "") or "")
    if name not in {"index_rope", "index_k_rope"}:
        return None
    if str(getattr(operator, "family", "") or "") != "elementwise":
        return None
    if str(getattr(compiled, "kind", "") or "") not in {"indexer_prolog", "lightning_indexer"}:
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    phase_name = str(phase or getattr(compiled, "phase", "")
                     or meta.get("phase") or "")
    if phase_name not in APPLICABLE_PHASES:
        return None
    geometry = _source_geometry(compiled, operator, name, meta)
    if geometry is None:
        return None

    from . import native_calls
    role = geometry["role"]
    heads = geometry["heads"]
    dim = geometry["dim"]
    rope_dim = geometry["rope_dim"]
    style = geometry["is_neox_style"]
    dtype = geometry["dtype"]
    query_heads = geometry["query_heads"]
    key_heads = geometry["key_heads"]
    caller_lines = list(CALLER_LINES[role])
    caller_symbol = ("AscendSFAIndexerBackend.forward_k" if role == ROLE_KEY
                     else "AscendSFAIndexerBackend.forward")
    source = native_calls._source(
        CALLER_SOURCE["repository"], CALLER_SOURCE["path"],
        sha256=CALLER_SOURCE["sha256"], symbol=caller_symbol,
        revision=REVISION, state="task_artifact")
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", path=WRAPPER_SOURCE["path"],
        sha256=WRAPPER_SOURCE["sha256"], evidence="source_declared",
        wrapper=ENTRY, wrapper_lines=list(WRAPPER_LINES),
        kernel_lines=list(KERNEL_LINES), launch_lines=list(LAUNCH_LINES),
        caller_path=CALLER_SOURCE["path"],
        caller_sha256=CALLER_SOURCE["sha256"],
        caller_lines=caller_lines,
        invoker_path=INVOKER_SOURCE["path"],
        invoker_sha256=INVOKER_SOURCE["sha256"],
        invoker_lines=list(INVOKER_LINES),
        grid="min(num_tokens, get_vectorcore_num())",
        profiler_name=KERNEL_NAME, execution_core="aiv",
        source_status="triton_jit_source_retained")
    attrs = {
        "index_rope_role": role,
        "n_heads": heads,
        "head_dim": dim,
        "rope_dim": rope_dim,
        "is_neox_style": style,
        "rope_implementation": SOURCE_IMPLEMENTATION,
        "rope_dtype": dtype,
        "index_query_heads": query_heads,
        "index_key_heads": key_heads,
        "phase": phase_name,
        "cos_sin_mode": "selected_cos_sin",
        "use_cos_sin": False,
        "in_place": True,
        "kernel_name": KERNEL_NAME,
        "source_commit": R64_SOURCE_COMMIT,
        "deployment_source_revision": "r64",
        "deployment_image_id": str(meta.get("image_id") or R64_IMAGE_SHA),
        "caller_symbol": caller_symbol,
        "caller_lines": caller_lines,
    }
    typed = {
        "dtype": dtype,
        "layout": "ND",
        "phase": phase_name,
        "shapes": {
            "qk": [heads, dim],
            "cos": [rope_dim],
            "sin": [rope_dim],
            "output": [heads, dim],
        },
        "attrs": attrs,
        "rank_width": 3,
    }
    return native_calls._call_spec(
        native_calls._api("triton", ENTRY, version=REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=("aclnn_symbol_unresolved",
                    "triton_compiled_cubin_not_retained",
                    "vectorcore_grid_count_not_retained"))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for one query or key rotary launch.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The physical owner is this row's single
    kernel launch; the SFA caller and the other tensor's launch are attribution
    only and are not charged here.
    """
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    role = str(attrs.get("index_rope_role") or "")
    if role not in {ROLE_QUERY, ROLE_KEY}:
        return None
    if attrs.get("deployment_source_revision") != "r64":
        return None
    if attrs.get("kernel_name") != KERNEL_NAME:
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    if phase not in APPLICABLE_PHASES:
        return None
    if str(attrs.get("phase") or "") != phase:
        return None
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    heads = _int(attrs.get("n_heads"))
    dim = _int(attrs.get("head_dim"))
    rope_dim = _int(attrs.get("rope_dim"))
    style = _bool(attrs.get("is_neox_style"))
    dtype = str(attrs.get("rope_dtype")
                or binding.typed.get("dtype") or "")
    query_heads = _int(attrs.get("index_query_heads"))
    key_heads = _int(attrs.get("index_key_heads"))
    if (dtype not in SOURCE_DTYPES or tokens < 1 or heads < 1 or dim < 1
            or rope_dim < 2 or rope_dim > dim or rope_dim % 2
            or style is None or query_heads < 1 or key_heads < 1):
        return None
    if role == ROLE_QUERY and heads != query_heads:
        return None
    if role == ROLE_KEY and heads != key_heads:
        return None

    cache_key = (binding.call_id, tokens, role, heads, dim, rope_dim, style,
                 dtype, phase, query_heads, key_heads)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    qk_shape = [tokens, heads, dim]
    cos_shape = [tokens, rope_dim]
    sin_shape = [tokens, rope_dim]
    actual_shapes = {
        "qk": qk_shape,
        "cos": cos_shape,
        "sin": sin_shape,
        "output": list(qk_shape),
    }
    actual_attrs = {
        **attrs,
        "tokens": tokens,
        "num_tokens": tokens,
        "qk_row_stride": heads * dim,
        "cos_row_stride": rope_dim,
        "sin_row_stride": rope_dim,
        "pad_n_h": _next_power_of_2(heads),
        "pad_rope_dim": _next_power_of_2(rope_dim),
        "block_size": _next_power_of_2(heads),
        "grid": "min(num_tokens, get_vectorcore_num())",
        "vectorcore_count": None,
        "vectorcore_count_resolved": False,
        "launch_mode": "selected_cos_sin",
        "triton_jit": True,
    }
    actual = {"dtype": dtype, "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)

    source = dict(binding.source or {})
    source["caller_lines"] = list(attrs.get("caller_lines") or ())
    source["invoker"] = dict(INVOKER_SOURCE)
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({
        "wrapper": ENTRY,
        "wrapper_lines": list(WRAPPER_LINES),
        "kernel_lines": list(KERNEL_LINES),
        "launch_lines": list(LAUNCH_LINES),
        "grid": actual_attrs["grid"],
        "api": {"namespace": "triton", "entry": ENTRY},
        "source_status": "triton_jit_source_retained",
    })
    qk = _descriptor(qk_shape, dtype, "ND", "input_output")
    qk["in_place"] = True
    cos = _descriptor(cos_shape, dtype, "ND", "input")
    sin = _descriptor(sin_shape, dtype, "ND", "input")
    output = _descriptor(qk_shape, dtype, "ND", "output")
    output.update({"alias_of": "qk", "in_place": True})
    descriptors = {
        "qk": qk,
        "cos": cos,
        "sin": sin,
        "output": output,
        "num_tokens": {
            "shape": [], "dtype": "int32", "layout": "ND",
            "role": "host_scalar", "value": tokens, "pointer": None,
            "allocation": "caller",
        },
        "qk_row_stride": {
            "shape": [], "dtype": "int32", "layout": "ND",
            "role": "host_scalar", "value": heads * dim,
            "expression": "n_heads*head_dim", "pointer": None,
            "allocation": "caller",
        },
        "cos_row_stride": {
            "shape": [], "dtype": "int32", "layout": "ND",
            "role": "host_scalar", "value": rope_dim,
            "expression": "rope_dim", "pointer": None,
            "allocation": "caller",
        },
        "sin_row_stride": {
            "shape": [], "dtype": "int32", "layout": "ND",
            "role": "host_scalar", "value": rope_dim,
            "expression": "rope_dim", "pointer": None,
            "allocation": "caller",
        },
    }
    arguments = {
        "qk": "qk",
        "cos": "cos",
        "sin": "sin",
        "rope_dim": rope_dim,
        "is_neox_style": style,
    }
    kernel_arguments = [
        {"name": "num_tokens", "value": tokens},
        {"name": "n_h", "value": heads},
        {"name": "hd", "value": dim},
        {"name": "rope_dim", "value": rope_dim},
        {"name": "pad_n_h", "value": actual_attrs["pad_n_h"]},
        {"name": "pad_rope_dim", "value": actual_attrs["pad_rope_dim"]},
        {"name": "BLOCK_SIZE", "value": actual_attrs["block_size"]},
        {"name": "IS_NEOX_STYLE", "value": style},
        {"name": "USE_COS_SIN", "value": False},
    ]
    prepare = {
        "schema": "mindie-sim/r64-triton-rope-siso-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_commit": R64_SOURCE_COMMIT,
        "source_lines": list(attrs.get("caller_lines") or ()),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": ["qk", "cos", "sin", "rope_dim", "is_neox_style"],
        "kernel_arguments": kernel_arguments,
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "optional_slots": {
            "cos_sin_cache": "absent",
            "positions": "absent",
            "cos_sin_mode": "selected_cos_sin",
        },
        "indexer_rope_pair": {
            "mechanism": (
                "query and key are two separate launches of the same kernel; "
                "each attached row charges exactly one launch"),
            "query": {
                "shape": [tokens, query_heads, dim],
                "n_heads": query_heads,
                "source_lines": list(CALLER_LINES[ROLE_QUERY]),
                "this_row": role == ROLE_QUERY,
            },
            "key": {
                "shape": [tokens, key_heads, dim],
                "n_heads": key_heads,
                "source_lines": list(CALLER_LINES[ROLE_KEY]),
                "this_row": role == ROLE_KEY,
            },
        },
        "match": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "entry": ENTRY,
            "api_namespace": "triton",
            "call_id": binding.call_id,
            "tokens": tokens,
            "n_heads": heads,
            "head_dim": dim,
            "rope_dim": rope_dim,
            "is_neox_style": style,
            "phase": phase,
            "rope_implementation": SOURCE_IMPLEMENTATION,
            "dtype": dtype,
            "source_sha256": source.get("sha256"),
            "source_lines": list(attrs.get("caller_lines") or ()),
            "kernel_source_sha256": kernel.get("source_sha256"),
            "kernel_lines": list(KERNEL_LINES),
            "launch_lines": list(LAUNCH_LINES),
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "source_commit": R64_SOURCE_COMMIT,
            "tensor_role": role,
            "in_place": True,
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "image": R64_IMAGE_TAG,
            "revision": REVISION,
            "source_commit": R64_SOURCE_COMMIT,
            "selection": ("exact R64 image digest + exact retained "
                          "indexer.py/rope.py blob hashes at the resolved "
                          "source commit"),
            "source_match": {
                "method": "sha256_of_retained_r64_source_blobs",
                "anchors": dict(SOURCE_ANCHORS),
            },
            "caller": source,
            "invoker": dict(INVOKER_SOURCE),
            "kernel_source": {
                "repository": WRAPPER_SOURCE["repository"],
                "path": WRAPPER_SOURCE["path"],
                "sha256": WRAPPER_SOURCE["sha256"],
                "symbol": KERNEL_NAME,
            },
        },
        "binary_selection": {
            "status": "triton_jit_from_retained_source",
            "reason": "compiled_cubin_not_retained",
            "compiler": "triton",
            "kernel": KERNEL_NAME,
            "entry": ENTRY,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "prepare_only_no_runtime_execution",
            "compiled_triton_cubin_not_retained",
            "vectorcore_grid_count_not_retained",
        ],
        "unresolved": [
            "aclnn_symbol_unresolved",
            "triton_compiled_cubin_not_retained",
            "vectorcore_grid_count_not_retained",
            "execution_rank_not_selected",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_compiled_cubin_claim",
            "no_tensor_pointer_fabrication",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": dtype,
            "layout": "ND",
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Independent native-input slice (A5 index rotary leaf).
#
# The selected cos/sin rows and the qk block are supplied directly; the slice
# maps them to the same elementwise program the model path uses, keeps the
# rotary dims/style/dtype and the in-place output alias explicit, and returns
# the exact implementation revision.  Query and key remain two separate
# launches; the caller must select one, never both by default.
# ---------------------------------------------------------------------------

ABI_ROLES = (ROLE_QUERY, ROLE_KEY)


def abi_spec() -> Any:
    """The declared independent ABI of the R64 index rotary launch."""
    from .native_calls import (
        NativeAbiCallSpec, NativeAbiTensorSpec, NativeAbiVariant)
    source = dict(CALLER_SOURCE)
    source.update({"source_commit": R64_SOURCE_COMMIT,
                   "wrapper": ENTRY,
                   "wrapper_path": WRAPPER_SOURCE["path"],
                   "wrapper_sha256": WRAPPER_SOURCE["sha256"],
                   "invoker_path": INVOKER_SOURCE["path"],
                   "invoker_sha256": INVOKER_SOURCE["sha256"]})
    kernel = {
        "name": KERNEL_NAME, "kind": "vector",
        "path": WRAPPER_SOURCE["path"],
        "sha256": WRAPPER_SOURCE["sha256"],
        "evidence": "source_declared",
        "wrapper": ENTRY,
        "wrapper_lines": list(WRAPPER_LINES),
        "kernel_lines": list(KERNEL_LINES),
        "launch_lines": list(LAUNCH_LINES),
        "source_commit": R64_SOURCE_COMMIT,
        "source_status": "triton_jit_source_retained",
        "launch": "grid=min(num_tokens, vectorcore_count)",
    }
    variants = tuple(
        NativeAbiVariant(role, f"I0:elementwise:index_{role}_rope",
                         f"I0:elementwise:index_{role}_rope",
                         (f"index_{role}_rope",), "kernel", "one_to_one")
        for role in ABI_ROLES)
    return NativeAbiCallSpec(
        entry=ENTRY, namespace="triton", aclnn_symbol=None,
        api_version=REVISION, source=source,
        source_revision=REVISION, image_id=R64_IMAGE_SHA,
        kernels=(kernel,),
        tensors=(
            NativeAbiTensorSpec("qk", "inout", None,
                                "tokens,n_heads,head_dim"),
            NativeAbiTensorSpec("cos", "input", None, "tokens,rope_dim"),
            NativeAbiTensorSpec("sin", "input", None, "tokens,rope_dim"),
            NativeAbiTensorSpec("output", "output", None,
                                "tokens,n_heads,head_dim"),
        ),
        attrs=("index_rope_role", "n_heads", "head_dim", "rope_dim",
               "is_neox_style", "rope_dtype"),
        optional_attrs=("phase", "cos_stride0", "sin_stride0"),
        variants=variants,
        execution_core="aiv", arch="A5", implemented=True,
        unresolved=("aclnn_symbol_unresolved",
                    "triton_compiled_cubin_not_retained",
                    "vectorcore_grid_count_not_retained"))


def _abi_program(inv: Any, supply: Any, policy: Any,
                 phase: str, machine: Any = None,
                 trace_limit: Any = None,
                 execution: Any = None) -> dict[str, Any] | None:
    from mindie_sim.prediction.native_binding import NativeAbiRegion
    from mindie_sim.prediction.piecewise.compiler import CompiledOperator
    from mindie_sim.prediction.piecewise.geometry import (
        RequestGeometry, WorkloadGeometry)
    from mindie_sim.prediction.piecewise.native_gdn import (
        LeafAbiUnsupported, leaf_attr, leaf_dtype, leaf_int,
        leaf_require_alias, leaf_shape, leaf_stride)
    from mindie_sim.schema.common import dtype_bytes
    from mindie_sim.schema.subgraphs import (
        BoundOperator, OperatorTemplate, StaticSignature)

    role = str(inv.conditions.get("variant") or "")
    if role not in ABI_ROLES:
        raise LeafAbiUnsupported(f"rope_abi_role_unknown:{role}")
    if str(leaf_attr(inv, "index_rope_role") or "") != role:
        raise LeafAbiUnsupported("rope_abi_role_attr_mismatch")
    dtype = str(leaf_attr(inv, "rope_dtype") or "")
    if dtype not in SOURCE_DTYPES:
        raise LeafAbiUnsupported(f"rope_abi_dtype_unsupported:{dtype}")
    heads = leaf_int(inv, "n_heads")
    dim = leaf_int(inv, "head_dim")
    rope_dim = leaf_int(inv, "rope_dim")
    style = _bool(leaf_attr(inv, "is_neox_style"))
    if heads < 1 or dim < 1 or rope_dim < 2 or rope_dim > dim or rope_dim % 2:
        raise LeafAbiUnsupported("rope_abi_rotary_dims_invalid")
    if style is None:
        raise LeafAbiUnsupported("rope_abi_style_unknown")
    runtime_phase = str(
        phase or inv.conditions.get("phase") or leaf_attr(inv, "phase", "")
        or "prefill")
    if runtime_phase not in APPLICABLE_PHASES:
        raise LeafAbiUnsupported(f"rope_abi_phase_unsupported:{runtime_phase}")
    qk_shape = leaf_shape(inv, "qk")
    output_shape = leaf_shape(inv, "output")
    cos_shape = leaf_shape(inv, "cos")
    sin_shape = leaf_shape(inv, "sin")
    if len(qk_shape) != 3 or qk_shape[1:] != [heads, dim]:
        raise LeafAbiUnsupported("rope_abi_qk_shape_mismatch")
    tokens = qk_shape[0]
    if output_shape != qk_shape:
        raise LeafAbiUnsupported("rope_abi_output_shape_mismatch")
    if len(cos_shape) != 2 or len(sin_shape) != 2:
        raise LeafAbiUnsupported("rope_abi_cos_sin_rank_mismatch")
    half = rope_dim // 2
    if (cos_shape[0] != tokens or sin_shape[0] != tokens
            or cos_shape[1] < half or sin_shape[1] < half):
        raise LeafAbiUnsupported("rope_abi_cos_sin_shape_mismatch")
    if leaf_dtype(inv, "qk") != dtype:
        raise LeafAbiUnsupported("rope_abi_qk_dtype_mismatch")
    for name in ("cos", "sin"):
        if leaf_dtype(inv, name) not in SOURCE_DTYPES:
            raise LeafAbiUnsupported(f"rope_abi_{name}_dtype_unsupported")
    leaf_require_alias(inv, "output", "qk")
    qk_bytes = float(dtype_bytes(dtype))
    cos_bytes = float(dtype_bytes(leaf_dtype(inv, "cos")))
    sin_bytes = float(dtype_bytes(leaf_dtype(inv, "sin")))
    cos_stride0 = leaf_stride(inv, "cos_stride0", half)
    sin_stride0 = leaf_stride(inv, "sin_stride0", half)
    cos_pitch = cos_stride0 if cos_stride0 is not None else cos_shape[1]
    sin_pitch = sin_stride0 if sin_stride0 is not None else sin_shape[1]
    # Source-level effective access of _triton_rope_siso: two R/2 halves of
    # qk per head, cos and sin R/2 each, and R elements written per head.  The
    # rest of each head is untouched (in-place) and is neither read nor
    # written by this kernel.
    qk_read = heads * rope_dim * qk_bytes
    cos_read = half * cos_bytes
    sin_read = half * sin_bytes
    read_bytes = qk_read + cos_read + sin_read
    write_bytes = heads * rope_dim * qk_bytes
    # One common family program consumes qk/cos/sin as separate values: the
    # native SISO branch derives R-based access and math from each value's
    # own dtype.  No aggregate mixed elementwise envelope is selected.
    cos_dtype = leaf_dtype(inv, "cos")
    sin_dtype = leaf_dtype(inv, "sin")
    conversion_values = (
        ("qk_load", dtype, "fp32", tokens * heads * rope_dim
         if dtype != "fp32" else 0),
        ("cos_load", cos_dtype, "fp32", tokens * half
         if cos_dtype != "fp32" else 0),
        ("sin_load", sin_dtype, "fp32", tokens * half
         if sin_dtype != "fp32" else 0),
        ("qk_store", "fp32", dtype, tokens * heads * rope_dim
         if dtype != "fp32" else 0),
    )
    conversion_total = sum(item[3] for item in conversion_values)
    theta = StaticSignature(
        "index_rope", matrix=(("elements", heads * dim),),
        quant=(("act_bytes", qk_bytes),),
        extras=(("qk_rope_head_dim", rope_dim),
                ("index_heads", heads), ("heads", heads),
                ("rope_dtype", dtype),
                ("cos_dtype", cos_dtype), ("sin_dtype", sin_dtype),
                ("index_rope_role", role),
                ("rope_implementation", SOURCE_IMPLEMENTATION),
                ("is_neox_style", style),
                ("cos_stride0", cos_pitch),
                ("sin_stride0", sin_pitch),
                ("source_kernel", KERNEL_NAME)))
    bound = BoundOperator(OperatorTemplate("index_rope", "elementwise",
                                           "rope"), theta)
    geometry = WorkloadGeometry(
        (RequestGeometry(tokens, 0, phase=runtime_phase),),
        phase=runtime_phase)
    compiled = CompiledOperator(bound, supply, policy, kind="indexer_prolog")
    cost = compiled.evaluate(geometry)
    pad_n_h = _next_power_of_2(heads)
    pad_rope_dim = _next_power_of_2(rope_dim)
    access = {
        "qk_read_bytes": qk_read * tokens,
        "cos_read_bytes": cos_read * tokens,
        "sin_read_bytes": sin_read * tokens,
        "qk_write_bytes": write_bytes * tokens,
        "untouched_head_elements": heads * (dim - rope_dim),
        "effective_read_bytes": read_bytes * tokens,
        "effective_write_bytes": write_bytes * tokens,
        "padded_qk_read_bytes": (
            pad_n_h * pad_rope_dim * qk_bytes * tokens),
        "padded_qk_write_bytes": (
            pad_n_h * pad_rope_dim * qk_bytes * tokens),
        "cos_row_stride_bytes": cos_pitch * cos_bytes,
        "sin_row_stride_bytes": sin_pitch * sin_bytes,
        "issued_basis": "effective_lane_access_not_measured_hbm",
        "per_token": {
            "qk_read_bytes": qk_read,
            "cos_read_bytes": cos_read,
            "sin_read_bytes": sin_read,
            "qk_write_bytes": write_bytes,
            "math_ops": 3 * heads * rope_dim,
            "conversion_elements": conversion_total // max(1, tokens),
        },
        "conversion_elements": conversion_total,
    }
    typed = {
        "dtype": dtype, "layout": "ND", "phase": runtime_phase,
        "shapes": {"qk": [heads, dim], "cos": [cos_shape[1]],
                   "sin": [sin_shape[1]], "output": [heads, dim]},
        "attrs": {
            "index_rope_role": role, "n_heads": heads, "head_dim": dim,
            "rope_dim": rope_dim, "is_neox_style": style,
            "rope_implementation": SOURCE_IMPLEMENTATION,
            "rope_dtype": dtype, "source_kernel": KERNEL_NAME,
            "cos_sin_mode": "selected_cos_sin",
            "cos_dtype": cos_dtype, "sin_dtype": sin_dtype,
            "output_dtype": dtype,
            "conversion_elements": conversion_total,
            "cos_stride0": cos_stride0, "sin_stride0": sin_stride0,
            "in_place": True,
            "source_commit": R64_SOURCE_COMMIT,
            "caller_lines": list(CALLER_LINES[role]),
        },
        "occurrence_attrs": {"tokens": "positive_int",
                             "num_tokens": "positive_int"},
        "rank_width": 3,
    }
    return {
        "cost": cost, "compiled": compiled, "geometry": geometry,
        "phase": runtime_phase, "kind": "indexer_prolog",
        "environment": {"tokens": tokens, "n_heads": heads,
                        "head_dim": dim, "rope_dim": rope_dim,
                        "is_neox_style": style, "dtype": dtype,
                        "pad_n_h": pad_n_h, "pad_rope_dim": pad_rope_dim},
        "region": NativeAbiRegion(
            op_id=f"I0:elementwise:index_{role}_rope",
            family="elementwise", recipe="rope",
            source_boundary="kernel", dtype=dtype, layout="ND",
            metadata=tuple(sorted({
                "n_heads": heads, "head_dim": dim, "rope_dim": rope_dim,
                "is_neox_style": style,
                "rope_implementation": SOURCE_IMPLEMENTATION,
                "cos_stride0": cos_pitch, "sin_stride0": sin_pitch,
            }.items()))),
        "typed": typed,
        "occurrence": {
            "dtype": dtype, "layout": "ND", "phase": runtime_phase,
            "shapes": {"qk": [tokens, heads, dim],
                       "cos": list(cos_shape), "sin": list(sin_shape),
                       "output": [tokens, heads, dim]},
            "attrs": {**typed["attrs"], "tokens": tokens,
                      "num_tokens": tokens,
                      "qk_row_stride": heads * dim,
                      "cos_row_stride": cos_shape[1],
                      "sin_row_stride": sin_shape[1],
                      "pad_n_h": pad_n_h, "pad_rope_dim": pad_rope_dim},
        },
        "effects": (),
        "unresolved": ("triton_compiled_cubin_not_retained",
                       "vectorcore_grid_count_not_retained"),
        "program": {
            "mode": f"index_{role}_rope", "family": "elementwise",
            "common_program": "elementwise.rope.triton_siso",
            "members": [f"index_{role}_rope"], "phase": runtime_phase,
            "rotary": {"dim": rope_dim, "style": ("neox" if style else "gptj"),
                       "dtype": dtype, "head_dim": dim, "heads": heads,
                       "modified_range": [0, rope_dim],
                       "untouched_range": [rope_dim, dim],
                       "math_ops_per_token": 3 * heads * rope_dim},
            "cos_sin": {
                "cos": {"shape": list(cos_shape), "dtype": cos_dtype,
                        "row_stride": cos_shape[1],
                        "effective_read_elements_per_token": half},
                "sin": {"shape": list(sin_shape), "dtype": sin_dtype,
                        "row_stride": sin_shape[1],
                        "effective_read_elements_per_token": half},
                "compute_dtype": "fp32",
            },
            "position_representation": "selected_cos_sin_rows",
            "output_alias": {"output": "qk", "in_place": True},
            "access": access,
            "cost_basis": "family_native_rope_rdim_per_value",
            "representation": {
                "qk_dtype": dtype, "output_dtype": dtype,
                "cos_dtype": cos_dtype, "sin_dtype": sin_dtype,
                "alias": "qk_in_place",
                "conversion_separate": True,
            },
            "conversion": {
                "compute_dtype": "fp32",
                "per_value": [
                    {"value": name, "source_dtype": source,
                     "target_dtype": target, "elements": elements}
                    for name, source, target, elements in conversion_values
                ],
                "total_elements": conversion_total,
                "per_token_elements": conversion_total // max(1, tokens),
            },
            "launch": {"grid": "min(tokens, vectorcore_count)",
                       "pad_n_h": pad_n_h, "pad_rope_dim": pad_rope_dim,
                       "vectorcore_count_unresolved": True},
            "formula": cost.formula,
            "work": {key: str(value) for key, value in cost.work},
            "derived": {key: str(value) for key, value in cost.derived},
            "execution": None,
        },
    }


def simulate_independent(request: Any, **kwargs: Any) -> dict:
    """Predict one index query/key rotary launch from its native input."""
    from .native_gdn import simulate_leaf
    return simulate_leaf(request, abi_spec(), _abi_program,
                         require_explicit_variant=True, **kwargs)
