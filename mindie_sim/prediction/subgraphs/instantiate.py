"""Deterministic instantiation of canonical subgraphs and static ``Theta``.

[Role]
  Project ``(ModelSpec, LayerPlan, FeatureConfig)`` into immutable, typed
  ``SubgraphInstance`` records.  Each operator receives a fully fixed
  ``StaticSignature``: weight dimensions, local head/expert geometry,
  quantisation role bytes, parallel axes and declared tiles.  Only sequence
  geometry and explicit state ports remain dynamic.

[Invariants]
  * Instantiation is a pure function of model/layer/features and declared
    implementation tiles; no measurement, model-name branch or request id is
    consulted.
  * Semantic macros reference child subgraphs instead of duplicating their
    work.  A child kind present in ``SubgraphTemplate.children`` contributes no
    second operator list here.
  * Quantisation is resolved per role; an override on one role cannot rewrite
    another role's weight payload, scale or zero-point accounting.
  * Repeated layers intern the same template object; instance identity and
    layer ids stay outside the template and performance key.
  * Unsupported mechanisms remain ``CoverageHole`` records, never fabricated
    operators or measured latency.

[Timeouts & Locks]
  Bounded per-layer construction; no iteration history, waits or locks.

[Side-Effect Policy]
  Returns new immutable records and mutates only the caller-supplied bounded
  template cache.  No I/O, device access or global model state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.dynamic_quant_schedule import dynamic_quant_traffic
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.schema.subgraphs import (
    ALL_ATTENTION_KINDS, BoundOperator, CoverageHole, CoverageReport,
    FusionContract, OperatorTemplate, StatePortSpec, StaticSignature,
    SubgraphInstance, SubgraphTemplate,
)
from .kinds import STANDARD_VARIANTS, attention_kind as resolve_attention_kind
from .registry import (
    KIND_BUILDERS, TemplateRegistry, build, kind_spec, port, register_builder,
    template, validate_combination, validate_kind, vocabulary,
)

_DEFAULT_MT = 128
_DEFAULT_NT = 128
_DEFAULT_KV_TILE = 128
_DEFAULT_CORES = 24
_DEFAULT_MOE_ROWS = 128
_DEFAULT_SPLIT_TARGET = 4096
_DEFAULT_MAX_SPLITS = 8

def _declared_kda_use_beta_sigmoid(runtime: Any) -> int | None:
    """Return the declared KDA option, or None when the case does not set it.

    Native ``recurrent.linear`` removes the separate ``linear_beta_sigmoid``
    producer only when this option is true on a decode source bind. The
    piecewise walk reads the same declaration from feature metadata.
    """
    kernels = getattr(runtime, "kernels", None)
    if not isinstance(kernels, Mapping):
        return None
    lowerings = kernels.get("lowerings")
    if not isinstance(lowerings, Mapping):
        return None
    binding = lowerings.get("kda")
    if not isinstance(binding, Mapping):
        return None
    options = binding.get("options")
    if not isinstance(options, Mapping) or "use_beta_sigmoid" not in options:
        return None
    value = options["use_beta_sigmoid"]
    if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
        return int(value)
    raise ModelingError("kda_use_beta_sigmoid_invalid")


def _pool_indexer_lowering(runtime: Any) -> str:
    """Implementation name of the pool-key lowering, or an empty string."""
    kernels = getattr(runtime, "kernels", None) or {}
    if not isinstance(kernels, dict):
        return ""
    lowerings = kernels.get("lowerings") or {}
    if not isinstance(lowerings, dict):
        return ""
    binding = lowerings.get("pool_key_indexer")
    if isinstance(binding, str):
        return binding
    if isinstance(binding, dict):
        return str(binding.get("implementation") or "")
    return ""


def _identity_metadata(machine: Any, model: Any) -> dict[str, Any]:
    """Actual deployment identity for compiled feature records.

    This is derived from the unaltered case machine/model, never from a
    measured supply record or a model/case label.  The keys are metadata only;
    a consumer that has no source-backed use for them must leave selection
    unchanged.
    """
    if machine is None:
        return {}
    key = str(getattr(machine, "hardware_key", "") or
              getattr(machine, "sku", "") or getattr(machine, "name", "") or "")
    sku = str(getattr(machine, "sku", "") or "")
    name = str(getattr(machine, "name", "") or "")
    arch = str(getattr(machine, "arch_family", "") or "")
    device = sku or key or name
    model_meta = dict(getattr(model, "metadata", ()) or ()) if model is not None else {}
    model_name = str(getattr(model, "name", "") or "") if model is not None else ""
    model_id = str(model_meta.get("catalog_key") or
                   model_meta.get("model_catalog_key") or model_name or "")
    result = {}
    for field, value in (("hardware_key", key), ("hardware_sku", sku),
                         ("hardware_name", name), ("arch_family", arch),
                         ("device", device), ("model_identifier", model_id)):
        if value:
            result[field] = value
    return result


def _effective_graph_buckets(runtime: Any) -> tuple[int, ...]:
    """Capture sizes selected by the same declared mode as GraphPolicy.

    Full, uniform decode may use full-capture keys; piecewise/eager modes use
    their own keys.  The feature record is not a capture policy fork: only the
    declared runtime fields choose, with the historical graph_buckets fallback
    retained when a legacy case has no explicit capture key list.
    """
    mode = str(getattr(runtime, "graph_mode", "") or "").lower()
    if mode in {"none", "off", "eager", "disabled"} or bool(
            getattr(runtime, "deepep_high_throughput", False)):
        return ()
    capture_mode = str(getattr(runtime, "graph_capture_mode", "") or "").lower()
    keys = getattr(runtime, "graph_capture_keys", ()) or ()
    piecewise = getattr(runtime, "piecewise_capture_keys", ()) or ()
    buckets = getattr(runtime, "graph_buckets", ()) or ()
    if capture_mode == "full" and not bool(getattr(runtime, "invalid_full_modes", False)):
        return tuple(int(value) for value in (keys or buckets))
    if capture_mode == "piecewise" or capture_mode == "mixed":
        return tuple(int(value) for value in piecewise)
    return tuple(int(value) for value in buckets)


def _model_source_metadata(model: Any) -> dict[str, Any]:
    """Promote only explicitly parsed HF source facts into feature metadata.

    No defaults, width guesses or boolean aliases.  The nested
    ``unconsumed_config_fields`` record is the parsed-source carrier for the
    native config; top-level model metadata is accepted only when the parser
    already put the exact key there.
    """
    if model is None:
        return {}
    model_meta = dict(getattr(model, "metadata", ()) or ())
    unconsumed = model_meta.get("unconsumed_config_fields")
    sources = [model_meta]
    if isinstance(unconsumed, Mapping):
        sources.append(unconsumed)
    meta: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        rope = source.get("rope_parameters")
        if isinstance(rope, Mapping):
            rope_copy = dict(rope)
            if rope_copy.get("mrope_section") is not None:
                try:
                    rope_copy["mrope_section"] = tuple(int(item) for item in rope_copy["mrope_section"])
                except (TypeError, ValueError):
                    pass
            meta["rope_parameters"] = rope_copy
        for key in ("mrope_section", "mrope_interleaved",
                    "partial_rotary_factor", "rope_theta", "rope_type"):
            value = source.get(key, rope.get(key) if isinstance(rope, Mapping) else None)
            if value is not None:
                if key == "mrope_section":
                    try:
                        value = tuple(int(item) for item in value)
                    except (TypeError, ValueError):
                        continue
                meta[key] = value
        value = source.get("routed_scaling_factor")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) > 0):
            meta["routed_scaling_factor"] = float(value)
        source_block = source.get("glm51_source")
        if isinstance(source_block, Mapping):
            meta["glm51_source"] = dict(source_block)
    eps = getattr(model, "norm_eps", None)
    if isinstance(eps, (int, float)) and not isinstance(eps, bool) and eps > 0:
        meta["norm_eps"] = float(eps)
    return meta


_NORMAL_SUMMARY_CACHE: dict[tuple[str, int, int, str, int, int], dict[str, Any]] = {}


def _source_stat(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except TimeoutError:
        raise
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _normal_summary(source_path: Path) -> dict[str, Any]:
    """Parse one real source once and cache only its normalized small summary."""
    quant_path = source_path.parent / "quant_model_description.json"
    index_path = source_path.parent / "quant_model_weights.safetensors.index.json"
    qstat = _source_stat(quant_path)
    istat = _source_stat(index_path)
    key = (str(quant_path), *(qstat or (-1, -1)), str(index_path),
           *(istat or (-1, -1)))
    cached = _NORMAL_SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached
    quant: Any = None
    index: Any = None
    q_status = "ok"
    idx_status = "ok"
    if qstat is None:
        q_status = "unavailable"
    else:
        try:
            quant = json.loads(quant_path.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, ValueError, TypeError):
            q_status = "unreadable_or_malformed"
    if istat is None:
        idx_status = "unavailable"
    else:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, ValueError, TypeError):
            idx_status = "unreadable_or_malformed"
    summary: dict[str, Any] = {
        "quant_source_status": q_status,
        "weight_index_status": idx_status,
        "quant_manifest_path": str(quant_path) if qstat is not None else None,
        "weight_index_path": str(index_path) if istat is not None else None,
        "quant_key_count": None,
        "weight_key_count": None,
        "weight_index_complete": False,
        "constructor_wants_bias": None,
        "role_bias": {},
        "trained_layernorm_bias": False,
    }
    if q_status == "ok" and isinstance(quant, Mapping):
        quant_keys = [str(key) for key in quant]
        summary["quant_key_count"] = len(quant_keys)
        summary["constructor_wants_bias"] = any(
            "norm.bias" in key for key in quant_keys)
        layers: list[int] = []
        for qkey in quant_keys:
            match = re.match(r"model\.layers\.(\d+)\.", qkey)
            if match:
                layers.append(int(match.group(1)))
        layers = sorted(set(layers))
        weight_map: dict[str, Any] = {}
        if idx_status == "ok" and isinstance(index, Mapping):
            raw_map = index.get("weight_map")
            if isinstance(raw_map, Mapping):
                weight_map = {str(key): value for key, value in raw_map.items()}
        summary["weight_key_count"] = len(weight_map)
        index_complete = bool(weight_map) and bool(layers) and all(
            f"model.layers.{layer}.input_layernorm.weight" in weight_map
            and f"model.layers.{layer}.post_attention_layernorm.weight" in weight_map
            for layer in layers) and "model.norm.weight" in weight_map
        summary["weight_index_complete"] = index_complete
        roles: dict[str, dict[str, Any]] = {}
        specs = {
            "input_layernorm": [f"model.layers.{layer}.input_layernorm.bias"
                                for layer in layers],
            "post_attention_layernorm": [
                f"model.layers.{layer}.post_attention_layernorm.bias"
                for layer in layers],
            "final_norm": ["model.norm.bias"],
            "indexer_k_norm": [
                f"model.layers.{layer}.self_attn.indexer.k_norm.bias"
                for layer in layers],
        }
        for role_name, expected in specs.items():
            loaded = [key for key in expected if key in weight_map]
            if index_complete:
                absent = [key for key in expected if key not in weight_map]
                unobserved: list[str] = []
            else:
                absent = []
                unobserved = [key for key in expected if key not in weight_map]
            if loaded:
                state = "loaded"
            elif index_complete:
                state = "absent"
            else:
                state = "unobserved"
            roles[role_name] = {
                "state": state,
                "loaded": loaded,
                "absent": absent,
                "unobserved": unobserved,
            }
        summary["role_bias"] = roles
        summary["trained_layernorm_bias"] = any(
            roles.get(name, {}).get("state") == "loaded"
            for name in ("input_layernorm", "post_attention_layernorm"))
    if len(_NORMAL_SUMMARY_CACHE) >= 32:
        _NORMAL_SUMMARY_CACHE.clear()
    _NORMAL_SUMMARY_CACHE[key] = summary
    del quant, index
    return summary


def _normal_source_metadata(model: Any, runtime: Any = None) -> dict[str, Any]:
    """Normalize declared normalization semantics without synthesizing identity."""
    if model is None:
        return {}
    meta = dict(getattr(model, "metadata", ()) or ())
    if runtime is not None:
        meta = {**dict(getattr(runtime, "metadata", ()) or ()), **meta}
    source = dict(meta.get("glm51_source") or {}) if isinstance(
        meta.get("glm51_source"), Mapping) else {}
    model_type = str(meta.get("source_model_type") or "")
    classes = {
        "qwen3_5_text": ("AscendGemmaRMSNorm", 1.0, "input_layernorm.initial"),
        "glm_moe_dsa": ("AscendRMSNorm", 0.0, "fused_residual"),
    }
    resolved = classes.get(model_type, (None, None, None))
    explicit_class = str(meta.get("rms_norm_source_class") or "") or None
    explicit_offset = meta.get("rms_norm_source_weight_offset")
    explicit_role = str(meta.get("rms_norm_source_role") or "") or None
    source_class = explicit_class or resolved[0]
    offset = explicit_offset if explicit_offset is not None else resolved[1]
    role = explicit_role or resolved[2]
    conflict = bool(explicit_class and resolved[0] and explicit_class != resolved[0])
    normal: dict[str, Any] = {
        "source_model_type": model_type or None,
        "source_class": source_class,
        "source_weight_offset": offset,
        "source_role": role,
        "source_conflict": conflict,
        "package": {},
        "quant_manifest_sha256": str(source.get("quant_manifest_sha256") or "") or None,
    }
    for key in ("vllm_ascend_commit", "vllm_ascend_package",
                "vllm_ascend_layernorm_sha256", "vllm_ascend_device_op_sha256",
                "vllm_commit", "vllm_qwen3_5_sha256"):
        if meta.get(key):
            normal[key] = str(meta[key])
    if explicit_class:
        normal["explicit_source_class"] = explicit_class
    if explicit_offset is not None:
        normal["explicit_source_weight_offset"] = explicit_offset
    if explicit_role:
        normal["explicit_source_role"] = explicit_role
    # Only a nonempty declared semantic source path; never Path('')/cwd.
    raw_source_path = str(source.get("model_config") or "")
    source_path = Path(raw_source_path) if raw_source_path else None
    if source_path is None:
        fallback = str(meta.get("source_model_path") or "")
        if fallback and Path(fallback).is_dir():
            source_path = Path(fallback) / "config.json"
    if source_path is None or not source_path.is_file():
        normal["quant_source_status"] = (
            "no_declared_semantic_source" if source_path is None
            else "semantic_source_path_unavailable")
        normal["weight_index_status"] = normal["quant_source_status"]
        normal["constructor_wants_bias"] = None
        normal["role_bias"] = {}
        return {"normal_source": normal}
    normal.update(_normal_summary(source_path))
    return {"normal_source": normal}


def _runtime_metadata(runtime: Any, machine: Any = None, model: Any = None) -> tuple:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    meta.update(_identity_metadata(machine, model))
    meta.update(_model_source_metadata(model))
    _normal = _normal_source_metadata(model, runtime)
    if _normal.get("normal_source"):
        _ns = dict(_normal["normal_source"])
        meta["normal_source"] = _ns
        if _ns.get("source_model_type"):
            meta["source_model_type"] = _ns["source_model_type"]
        if _ns.get("source_class"):
            meta["rms_norm_source_class"] = _ns["source_class"]
        if _ns.get("source_weight_offset") is not None:
            meta["rms_norm_source_weight_offset"] = _ns["source_weight_offset"]
        if _ns.get("source_role"):
            meta["rms_norm_source_role"] = _ns["source_role"]
        for key, value in (_ns.get("package") or {}).items():
            meta.setdefault(key, value)
        if _ns.get("native_norm"):
            meta["native_norm"] = dict(_ns["native_norm"])
    for flag in ("multistream_kda_proj", "multistream_attention_indexer",
                 "multistream_window_compressed"):
        if bool(getattr(runtime, flag, False)):
            meta[flag] = True
    declared = _declared_kda_use_beta_sigmoid(runtime)
    if declared is not None:
        meta["kda_use_beta_sigmoid"] = declared
    # The analytical producer and the CANN KeyPool caller are different
    # contracts. Record only the CANN selection so a missing lowering stays
    # on the analytical default.
    if _pool_indexer_lowering(runtime) == "vllm_ascend_cann_keypool_caller":
        meta["pool_key_indexer"] = "vllm_ascend_cann_keypool_caller"
    # Deployment block size.  Attention uses it only for incremental q=1
    # page-table index traffic.  A missing or zero value stays off that path.
    block = getattr(runtime, "kv_block_size", 0) or 0
    if isinstance(block, bool) or not isinstance(block, int) or block < 0:
        raise ModelingError("kv_block_size_invalid")
    if block > 0:
        meta["kv_block_size"] = block
    if "cp_kv_cache_interleave_size" not in meta and block > 0:
        meta["cp_kv_cache_interleave_size"] = block
    max_model_len = getattr(runtime, "max_model_len", 0) or 0
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 0:
        raise ModelingError("max_model_len_invalid")
    if max_model_len > 0:
        meta["max_model_len"] = max_model_len
    # Source MC2 capacity reads the configured scheduler budgets directly.
    # These are runtime declarations, not timing inputs; a missing value keeps
    # the historical static recipe and the selector's own branch.
    for key in ("max_num_batched_tokens", "max_num_seqs"):
        value = getattr(runtime, key, 0) or 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelingError(f"runtime_{key}_invalid")
        if value > 0:
            meta[key] = value
    kernels = getattr(runtime, "kernels", None)
    if isinstance(kernels, Mapping) and "sequence_parallel" in kernels:
        decl = kernels.get("sequence_parallel")
        if not isinstance(decl, Mapping):
            raise ModelingError("runtime_sequence_parallel_invalid")
        meta["sequence_parallel"] = dict(decl)
    if isinstance(kernels, Mapping) and "lowerings" in kernels:
        lowerings = kernels.get("lowerings") or {}
        if not isinstance(lowerings, Mapping):
            raise ModelingError("runtime_lowerings_invalid")
        meta["source_lowerings"] = tuple(sorted(str(key) for key in lowerings))
        fusion = lowerings.get("native_fusion")
        if isinstance(fusion, Mapping):
            options = fusion.get("options")
            if isinstance(options, Mapping):
                roles = options.get("roles")
                if roles:
                    meta["native_fusion_roles"] = tuple(
                        str(role) for role in roles)
                profiles = options.get("program_profiles")
                if isinstance(profiles, Mapping):
                    meta["native_fusion_profiles"] = tuple(sorted(
                        (str(key), str(value)) for key, value in profiles.items()))
    from mindie_sim.prediction.mx_source_provenance import source_metadata
    meta.update(dict(source_metadata(meta)))
    return tuple(sorted(meta.items()))


def source_lowering_selected(ctx: Any, name: str) -> bool:
    """True only when the deployment/source declares this native lowering.

    A legacy feature record without a lowering table keeps the existing
    template contract; once a real runtime table is present, absence means the
    causal template stays unfused.  The template alone is not selection.
    """
    keys = dict(getattr(ctx.features, "metadata", ()) or ()).get("source_lowerings")
    if keys is None:
        return True
    declared = {str(value) for value in keys}
    return str(name) in declared

@dataclass(frozen=True)
class FeatureConfig:
    """Normalised projection of Runtime/ParallelSpec/QuantSpec.

    This is deliberately not a second user-facing schema: it is constructed
    from the existing runtime/model records and only adds implementation
    tiles that the static contract must fix.
    """

    tp: int = 1
    ep: int = 1
    etp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    pcp: int = 1
    cp_plan: str = "none"
    activation_dtype: str = "bf16"
    quant_activation_format: str = ""
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    index_dtype: str = "int8"
    quant: QuantSpec | None = None
    graph_buckets: tuple[int, ...] = ()
    multistream_shared_expert: bool = False
    multistream_kda_proj: bool = False
    multistream_attention_indexer: bool = False
    multistream_window_compressed: bool = False
    moe_recipe: str = "alltoall"
    mc2_chunks: int = 4
    lowering_passes: tuple[str, ...] = ("activation_quantization",)
    implementation: str = "analytical"
    final_norm_scope: str = "all"
    token_tile: int = _DEFAULT_MT
    kv_tile: int = _DEFAULT_KV_TILE
    cores: int = _DEFAULT_CORES
    moe_row_tile: int = _DEFAULT_MOE_ROWS
    split_target: int = _DEFAULT_SPLIT_TARGET
    max_splits: int = _DEFAULT_MAX_SPLITS
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("tp", "ep", "etp", "pp", "dp", "cp", "pcp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_parallel_invalid:{name}")
        for name in ("token_tile", "kv_tile", "cores", "moe_row_tile",
                     "split_target", "max_splits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"feature_tile_invalid:{name}")
        for name in ("activation_dtype", "weight_dtype", "kv_dtype", "index_dtype"):
            dtype_bytes(getattr(self, name))
        if self.quant_activation_format:
            dtype_bytes(self.quant_activation_format)
        if self.final_norm_scope not in {"all", "selected"}:
            raise ModelingError("feature_final_norm_scope_invalid")
        if self.cp_plan not in {"none", "dcp", "dsa_cp", "pcp", "pcp_dcp"}:
            raise ModelingError(f"feature_cp_plan_invalid:{self.cp_plan}")
        if self.moe_recipe not in {"alltoall", "mc2", "tp_join_reduce",
                                   "local_expert_reduce"}:
            raise ModelingError(f"feature_moe_recipe_invalid:{self.moe_recipe}")
        if self.mc2_chunks < 1:
            raise ModelingError("feature_mc2_chunks_invalid")
        object.__setattr__(self, "graph_buckets", tuple(sorted(set(self.graph_buckets))))
        object.__setattr__(self, "lowering_passes", tuple(self.lowering_passes))
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", tuple(sorted(self.metadata.items())))
        else:
            object.__setattr__(self, "metadata", tuple(self.metadata))

    @property
    def activation_format(self) -> str:
        return self.quant_activation_format or self.activation_dtype or "bf16"

    @property
    def world_size(self) -> int:
        return self.dp * self.pp * self.tp * self.pcp

    @property
    def etp_width(self) -> int: return max(1, self.etp)

    @property
    def max_bucket(self) -> int: return max(self.graph_buckets) if self.graph_buckets else 0

    def project(self, *, tp: int = 0, dp: int = 0, ep: int = 0, pp: int = 0,
                cores: int = 0, graph_buckets: Iterable[int] | None = None,
                kv_dtype: str = "", kv_bf16_override: bool = False) -> "FeatureConfig":
        tp_v, dp_v = max(1, tp or self.tp), max(1, dp or self.dp)
        ep_v = max(1, int(ep or self.ep))
        # ETP is the remaining TP shard of each expert, not DP*TP/EP.  EP
        # groups already live in the DP*TP domain; EP beyond TP leaves ETP=1.
        etp_v = (tp_v if ep_v <= 1 else
                 (tp_v // ep_v if ep_v <= tp_v and tp_v % ep_v == 0 else 1))
        meta = dict(self.metadata)
        if kv_bf16_override: meta["kv_bf16_override"] = 1
        updates: dict[str, Any] = dict(tp=tp_v, dp=dp_v, ep=ep_v, etp=etp_v,
                                       pp=max(1, pp or self.pp),
                                       graph_buckets=(tuple(self.graph_buckets)
                                                      if graph_buckets is None or not tuple(graph_buckets)
                                                      else tuple(graph_buckets)),
                                       metadata=tuple(sorted(meta.items())))
        if cores > 0: updates["cores"] = int(cores)
        if kv_dtype: updates["kv_dtype"] = kv_dtype
        return replace(self, **updates)

    @classmethod
    def from_runtime(cls, runtime: Any, model: ModelSpec | None = None,
                     *, machine: Any = None) -> "FeatureConfig":
        parallel = getattr(runtime, "parallel", ParallelSpec()) if runtime is not None else ParallelSpec()
        if not isinstance(parallel, ParallelSpec):
            raise ModelingError("feature_parallel_record_invalid")
        quant = getattr(model, "quant", None) if model is not None else None
        declared_etp = int(getattr(parallel, "etp", 0) or 0)
        if declared_etp < 1:
            declared_etp = (parallel.tp if parallel.ep <= 1 else
                            (parallel.tp // parallel.ep
                             if parallel.ep <= parallel.tp and parallel.tp % parallel.ep == 0
                             else 1))
        return cls(
            tp=parallel.tp, ep=parallel.ep, etp=declared_etp,
            pp=parallel.pp, dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp,
            cp_plan=str(getattr(parallel, "cp_plan", "none") or "none"),
            activation_dtype=getattr(model, "activation_dtype", "bf16") if model else "bf16",
            quant_activation_format=(str(quant.activation_format)
                                     if quant is not None and quant.activation_format else ""),
            weight_dtype=getattr(model, "weight_dtype", "bf16") if model else "bf16",
            kv_dtype=((str(getattr(quant, "kv_dtype", "") or "") if quant else "") or
                      (getattr(model, "kv_dtype", "bf16") if model else "bf16")),
            index_dtype=(str(getattr(quant, "index_dtype", "") or "") if quant else "") or "int8",
            quant=quant,
            graph_buckets=_effective_graph_buckets(runtime),
            multistream_shared_expert=bool(getattr(runtime, "multistream_overlap_shared_expert", False)),
            multistream_kda_proj=bool(getattr(runtime, "multistream_kda_proj", False)),
            multistream_attention_indexer=bool(getattr(runtime, "multistream_attention_indexer", False)),
            multistream_window_compressed=bool(getattr(runtime, "multistream_window_compressed", False)),
            moe_recipe=str(getattr(runtime, "moe_recipe", "alltoall")),
            mc2_chunks=int(getattr(runtime, "mc2_chunks", 4)),
            lowering_passes=tuple(getattr(runtime, "lowering_passes", ("activation_quantization",))),
            final_norm_scope=str(getattr(runtime, "final_norm_scope", "all")),
            metadata=_runtime_metadata(runtime, machine, model),
        )

    @classmethod
    def from_model(cls, model: ModelSpec, *, machine: Any = None,
                   **overrides: Any) -> "FeatureConfig":
        quant = getattr(model, "quant", None)
        base = dict(
            activation_dtype=model.activation_dtype or "bf16",
            quant_activation_format=(str(quant.activation_format)
                                     if quant is not None and quant.activation_format else ""),
            weight_dtype=model.weight_dtype or "bf16",
            kv_dtype=((getattr(quant, "kv_dtype", "") if quant else "")
                      or model.kv_dtype or "bf16"),
            index_dtype=((getattr(quant, "index_dtype", "") if quant else "")
                         or "int8"),
            quant=quant,
            cp_plan=(("dcp" if int(overrides.get("cp", 1) or 1) > 1 else "none")
                     if "cp_plan" not in overrides else str(overrides["cp_plan"])),
            metadata=tuple(sorted({**_identity_metadata(machine, model),
                                  **_model_source_metadata(model)}.items())),
        )
        if "etp" not in overrides:
            tp = int(overrides.get("tp", 1) or 1)
            dp = int(overrides.get("dp", 1) or 1)
            ep = int(overrides.get("ep", 1) or 1)
            base["etp"] = ((tp if ep <= 1 else
                            (tp // ep if ep <= tp and tp % ep == 0 else 1))
                           if tp > 0 else 1)
        base.update(overrides)
        return cls(**base)

    def with_parallel(self, parallel: ParallelSpec) -> "FeatureConfig":
        return replace(self, tp=parallel.tp, ep=parallel.ep, pp=parallel.pp,
                       dp=parallel.dp, cp=parallel.cp, pcp=parallel.pcp,
                       cp_plan=str(getattr(parallel, "cp_plan", self.cp_plan)))

@dataclass(frozen=True)
class _Ctx:
    model: ModelSpec
    layer: LayerPlan
    features: FeatureConfig
    stage: int
    role: str = "body"
    variant: str = ""
    layers: int = 0
    #: Kind explicitly requested by the standalone entry.  The model assembly
    #: leaves it empty and always derives the kind from the layer plan.  A
    #: request only selects among kinds that share one builder macro; it never
    #: relabels a different mechanism.
    requested_kind: str = ""

    def __post_init__(self) -> None:
        if self.layer.index < 0:
            raise ModelingError("layer_index_invalid")
        if self.stage < 0 or self.stage >= self.features.pp:
            raise ModelingError(f"layer_stage_invalid:{self.stage}")
        if self.layers and self.layer.index >= self.layers:
            raise ModelingError(f"layer_index_exceeds_model:{self.layer.index}")

    @property
    def hidden(self) -> int: return int(self.model.hidden_size)
    @property
    def tp(self) -> int: return self.features.tp
    @property
    def ep(self) -> int: return self.features.ep
    @property
    def etp(self) -> int: return self.features.etp
    @property
    def heads(self) -> int: return int(self.layer.num_heads or self.model.num_heads)
    @property
    def local_heads(self) -> int: return max(1, self.heads // self.tp)
    @property
    def kv_heads(self) -> int: return int(self.layer.num_kv_heads or self.model.num_kv_heads or self.heads)
    @property
    def local_kv_heads(self) -> int: return local_kv_heads(self.kv_heads, self.tp)
    @property
    def group_heads(self) -> int: return max(1, self.local_heads // self.local_kv_heads)
    @property
    def width(self) -> int: return int(self.layer.head_dim or self.model.head_dim)
    @property
    def intermediate(self) -> int: return int(self.layer.intermediate_size or self.model.intermediate_size)
    @property
    def local_intermediate(self) -> int: return max(1, math.ceil(self.intermediate / self.tp))
    @property
    def local_experts(self) -> int:
        experts = int(self.layer.num_experts or self.model.num_experts or 1)
        return max(1, math.ceil(experts / self.ep))

    @property
    def expert_width(self) -> int:
        width = int(self.layer.moe_intermediate_size or self.model.moe_intermediate_size)
        return max(1, math.ceil(width / max(1, self.etp)))

    @property
    def latent(self) -> int:
        return int(self.layer.expert_latent_size or self.hidden)

    @property
    def shared_width(self) -> int:
        width = int(self.layer.shared_intermediate_size
                    or self.layer.moe_intermediate_size or self.model.moe_intermediate_size)
        count = int(self.layer.num_shared_experts or self.model.num_shared_experts or 1)
        return width * max(1, count)

    @property
    def top_k(self) -> int:
        return int(self.layer.top_k or self.model.top_k or 1)

    @property
    def quant(self) -> QuantSpec | None:
        return self.features.quant

    @property
    def kv_dtype(self) -> str:
        return self.features.kv_dtype or self.model.kv_dtype

    @property
    def kv_row_elements(self) -> int:
        return int(self.layer.cache_row_elements)

    @property
    def kv_row_bytes(self) -> float:
        # Ordinary GQA/MQA owns one local KV row per rank (replicated when the
        # global KV head count is below TP); using the global cache row would
        # charge every rank for all logical KV heads.  MLA and compressed
        # representations keep their declared cache-row layout.
        if (self.layer.attention in {"gqa", "mha", "mqa"}
                and self.layer.cache_kind != "window_compressed"):
            elements = self.local_kv_heads * (self.width + self.width)
        else:
            elements = self.kv_row_elements
        return elements * dtype_bytes(self.kv_dtype)

    @property
    def shared_consumer(self) -> bool:
        owner = int(self.layer.kv_owner_layer_id)
        return 0 <= owner < self.layer.index and not self.layer.owns_long_kv

    @property
    def attention_kind(self) -> str:
        """The canonical kind of this layer's attention core."""
        return resolve_attention_kind(self.layer)

def _sig(role: str, **groups: Any) -> StaticSignature:
    return StaticSignature(role, **groups)

def _parallel(ctx: _Ctx) -> tuple[tuple[str, Any], ...]:
    return (("tp", ctx.tp), ("ep", ctx.ep), ("etp", ctx.etp),
            ("pp", ctx.features.pp), ("dp", ctx.features.dp),
            ("cp", ctx.features.cp), ("pcp", ctx.features.pcp),
            ("cp_plan", getattr(ctx.features, "cp_plan", ())),
            ("world", ctx.features.world_size), ("ranks_per_node", int(dict(ctx.features.metadata).get("ranks_per_node", 8) or 8)))

def _tiles(ctx: _Ctx) -> tuple[tuple[str, Any], ...]:
    features = ctx.features
    return (("mt", features.token_tile), ("nt", features.kv_tile),
            ("cores", features.cores), ("moe_rows", features.moe_row_tile),
            ("split_target", features.split_target),
            ("max_splits", features.max_splits),
            ("graph_buckets", features.graph_buckets))

_LOW_ACT = frozenset({"mxfp8", "mxfp4", "fp8_e4m3", "int8"})
_HIGH_ACT = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})

def _matrix_act_format(ctx: _Ctx, category: str) -> str:
    stream = ctx.features.activation_dtype or "bf16"
    quant = ctx.quant
    if quant is None:
        declared = ctx.features.quant_activation_format
        return declared if declared in _LOW_ACT else stream
    resolved = quant.for_category(category)
    fmt = str(resolved.format_for(category))
    # compute_dtype_for returns fp32 activations for an fp32 category.  The
    # model stream dtype is the pre-cast value, not the matrix operand.
    if fmt in {"fp32", "float32"}:
        return "fp32"
    if fmt in _HIGH_ACT:
        return stream
    act = str(resolved.activation_format or ctx.features.quant_activation_format or "")
    return act if act in _LOW_ACT else stream


def _quant_fields(ctx: _Ctx, category: str, rows: int, cols: int) -> tuple[tuple[str, Any], ...]:
    rows, cols = int(rows), int(cols)
    quant = ctx.quant
    act_format = _matrix_act_format(ctx, category)
    fallback = dtype_bytes(act_format)
    if rows <= 0 or cols <= 0:
        return (("weight_format", ctx.features.weight_dtype), ("act_format", act_format),
                ("weight_payload", 0.0), ("weight_scale", 0.0), ("weight_zero", 0.0),
                ("weight_total", 0.0), ("residency", 0.0), ("scale_group", 0),
                ("act_bytes", fallback), ("quantized", 0))
    if quant is None:
        weight_bytes = float(rows * cols) * dtype_bytes(ctx.features.weight_dtype)
        return (("weight_format", ctx.features.weight_dtype),
                ("act_format", act_format),
                ("weight_payload", weight_bytes), ("weight_scale", 0.0),
                ("weight_zero", 0.0), ("weight_total", weight_bytes),
                ("residency", weight_bytes), ("scale_group", 0),
                ("act_bytes", fallback), ("quantized", 0))
    resolved = quant.for_category(category)
    payload = resolved.matrix_payload_bytes(rows, cols, category)
    total = resolved.matrix_bytes(rows, cols, category)
    zero = 0.0
    fmt = resolved.format_for(category)
    runtime_meta = resolved.execution_int8_weight_metadata()
    runtime_scale_bytes = resolved.execution_weight_scale_bytes()
    if fmt in {"int8", "w8a8"} and runtime_meta == "scale_offset":
        zero = float(rows) * float(runtime_scale_bytes)
    scale = max(0.0, total - payload - zero)
    group = 0
    if fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        group = int(resolved.weight_block[1] if resolved.weight_block else 0)
        if not group:
            group = int(resolved.expert_block or 0)
    fields = (("weight_format", fmt), ("act_format", act_format),
              ("weight_payload", payload), ("weight_scale", scale),
              ("weight_zero", zero), ("weight_total", total),
              ("residency", total), ("scale_group", group),
              ("act_bytes", dtype_bytes(act_format)),
              ("quantized", int(fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})))
    method = resolved.linear_method(category)
    prequant = str(getattr(ctx.features, "activation_dtype", "") or
                   getattr(ctx.features, "activation_format", "") or "bf16")
    declared_output = resolved.output_format_for(
        category, prequant=prequant)
    fields += (("linear_method", method), ("prequant_format", prequant),
               ("category", str(category)))
    if method == "mx_fp8":
        # The consumer's second input is the producer's E8M0 activation scale,
        # M x (K/64) x 2.  Keep it as a separate tensor; never fold it into
        # the E4M3 payload width.
        blocks = (int(cols) + 31) // 32
        padded = (blocks + 1) // 2 * 2
        fields += (("activation_payload_bytes", float(int(cols))),
                   ("activation_scale_bytes", float(padded)),
                   ("activation_scale_shape", (padded // 2, 2)),
                   ("activation_scale_format", "fp8_e8m0"))
    if declared_output:
        fields += (("output_format", declared_output),)
    if fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        fields += (("weight_scale_format", resolved.execution_weight_scale_format()),
                   ("weight_scale_bytes", float(runtime_scale_bytes)),
                   ("weight_scale_storage_format", resolved.scale_format),
                   ("weight_scale_storage_bytes", float(resolved.scale_bytes)))
    return fields


def _declared_compute_quant(rows: int, cols: int, dtype: str) -> tuple[tuple[str, Any], ...]:
    """Weight and activation of a compute dtype declared beside the quant plan.

    Native router construction replaces the model quant with
    ``QuantSpec(weight_format=router_dtype, activation_format=router_dtype)``
    and category ``router_runtime``.  The ``router`` exclusion is BF16 and
    does not carry that declaration.
    """
    if dtype not in {"fp32", "bf16", "fp16"}:
        raise ModelingError(f"unsupported_declared_compute_dtype:{dtype}")
    width = float(dtype_bytes(dtype))
    payload = float(int(rows)) * float(int(cols)) * width
    return (
        ("weight_format", dtype), ("act_format", dtype),
        ("weight_payload", payload), ("weight_scale", 0.0),
        ("weight_zero", 0.0), ("weight_total", payload),
        ("residency", payload), ("scale_group", 0),
        ("act_bytes", width), ("quantized", 0),
    )

def _result_bytes(ctx: _Ctx, category: str) -> float:
    """Materialized consumer bytes for a role, with the prior fallback."""
    prequant = str(getattr(ctx.features, "activation_dtype", "") or
                   getattr(ctx.features, "activation_format", "") or "bf16")
    fmt = (ctx.quant.output_format_for(category, prequant=prequant)
           if ctx.quant is not None else "")
    return dtype_bytes(fmt or ctx.features.activation_format)


def _matrix(role: str, k: int, n: int, ctx: _Ctx, category: str,
            *, extra: tuple[tuple[str, Any], ...] = (),
            slots: tuple[str, ...] = ("weight_total",)) -> StaticSignature:
    return _sig(role, matrix=(("K", int(k)), ("N", int(n))),
                quant=_quant_fields(ctx, category, n, k),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _norm_gamma(ctx: _Ctx, width: int, role: str) -> tuple[tuple[str, Any], ...]:
    """Declare one normalization gamma read in its actual loaded format."""
    quant = ctx.quant
    if quant is not None:
        runtime = quant.norm_gamma_bytes(width, role)
        storage = quant.norm_gamma_bytes(width, role, source=True)
        runtime_format = quant.norm_gamma_format(role)
        storage_format = quant.norm_gamma_format(role, source=True)
    else:
        runtime = storage = width * dtype_bytes(ctx.features.activation_dtype or "bf16")
        runtime_format = storage_format = ctx.features.activation_dtype or "bf16"
    return (
        ("gamma_bytes", runtime),
        ("gamma_storage_bytes", storage),
        ("gamma_format", runtime_format),
        ("gamma_storage_format", storage_format),
        ("gamma_role", role),
    )


def _elementwise(role: str, ctx: _Ctx, *, elements: int, ops: int,
                 read_bytes: float, write_bytes: float,
                 extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("elements", int(elements)),),
                quant=(("act_bytes", dtype_bytes(ctx.features.activation_format)),),
                parallel=_parallel(ctx), tiles=_tiles(ctx),
                extras=(("ops_per_element", ops), ("read_bytes", read_bytes),
                        ("write_bytes", write_bytes), *extra))

def _copy(role: str, ctx: _Ctx, *, elements: int, bytes_per_element: float,
          extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("elements", int(elements)),),
                quant=(("act_bytes", bytes_per_element),),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _collective(role: str, ctx: _Ctx, *, axis: str, members: int,
                bytes_per_token: float, stages: int = 2,
                extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, parallel=_parallel(ctx), tiles=_tiles(ctx),
                quant=(("wire_bytes", float(bytes_per_token)),),
                extras=(("group_axis", axis), ("members", int(members)),
                        ("stages", int(stages)), *extra))


def _dense_flashcomm(ctx: _Ctx) -> bool:
    """v0.30 dense FlashComm sequence parallelism. Not a timed branch."""
    from mindie_sim.prediction.piecewise.flashcomm_dense import dense_flashcomm_enabled
    return dense_flashcomm_enabled(ctx.model, ctx.features)


def _activation_hidden_bytes(ctx: _Ctx) -> float:
    """Residual payload. The collective moves the activation stream, not the quant stream."""
    return float(ctx.hidden) * float(dtype_bytes(ctx.features.activation_dtype or "bf16"))


def _native_group2_declared(ctx: _Ctx) -> bool:
    """Only an explicit structured AICPU/BF16 AG-RS opt-in admits the curve."""
    meta = dict(ctx.features.metadata or {})
    decl = meta.get("a5_native_group2")
    if not isinstance(decl, dict):
        return False
    primitives = decl.get("primitives")
    if not isinstance(primitives, (list, tuple)):
        return False
    return bool(
        ctx.tp == 2
        and str(ctx.features.activation_dtype or "").lower() == "bf16"
        and str(decl.get("schema") or "") == "mindie-sim/a5-native-group2/1"
        and int(decl.get("group_size") or 0) == 2
        and str(decl.get("dtype") or "").lower() == "bf16"
        and str(decl.get("execution_core") or "").lower() == "aicpu"
        and str(decl.get("stack") or "").lower() == "rack08"
        and str(decl.get("cann") or "") == "9.2.0"
        and bool(decl.get("source_path"))
        and str(decl.get("deployment_image_id") or "") == str(meta.get("image_id") or "")
        and str(decl.get("deployment_cann") or "") in str(meta.get("image") or "")
        and {"all_gather", "reduce_scatter"} <= {str(v) for v in primitives}
        and "cann9.2.0" in str(meta.get("image") or ""))


def _sp_collective(ctx: _Ctx, kind: str) -> StaticSignature:
    rows = "shard" if kind == "all_gather" else "padded"
    name = "tp_all_gather" if kind == "all_gather" else "tp_reduce_scatter"
    extra = (("collective", kind), ("sp_rows", rows),
             ("payload_dtype", str(ctx.features.activation_dtype or "bf16")))
    if _native_group2_declared(ctx):
        decl = dict(ctx.features.metadata or {}).get("a5_native_group2")
        extra += (("a5_native_group2", "native-20261107"),
                  ("a5_native_stack", str(decl.get("stack") or "rack08")),
                  ("a5_native_cann", str(decl.get("cann") or "9.2.0")),
                  ("a5_native_source", str(decl.get("source_id")
                                            or "a5-hccl-native-20261107")),
                  ("a5_native_source_path", str(decl.get("source_path") or "")),
                  ("a5_native_execution_core", str(decl.get("execution_core") or "")),
                  ("a5_native_placement", str(decl.get("placement") or "")),
                  ("a5_native_deployment_image_id",
                   str(decl.get("deployment_image_id") or "")))
    return _collective(
        name, ctx, axis="tp", members=ctx.tp,
        bytes_per_token=_activation_hidden_bytes(ctx), extra=extra)


def _dense_sp_block(specs: list, signatures: dict, ctx: _Ctx) -> None:
    """Gather the sequence shard, then reduce-scatter partial sums.

    This replaces one TP allreduce. The gather is first because the shard
    norm is the preceding subgraph. Do not also append ``tp_allreduce``.
    """
    specs.insert(0, ("tp_all_gather", "collective", "tp_all_gather"))
    signatures["tp_all_gather"] = _sp_collective(ctx, "all_gather")
    specs.append(("tp_reduce_scatter", "collective", "tp_reduce_scatter"))
    signatures["tp_reduce_scatter"] = _sp_collective(ctx, "reduce_scatter")


def _dense_sp_exit(specs: list, signatures: dict, ctx: _Ctx) -> None:
    """One hidden all-gather after the final norm and before the LM head."""
    specs.append(("tp_all_gather", "collective", "tp_all_gather"))
    signatures["tp_all_gather"] = _sp_collective(ctx, "all_gather")

def _state(role: str, ctx: _Ctx, *, row_bytes: float, rows: int = 1,
           extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(role, matrix=(("rows", int(rows)), ("row_bytes", float(row_bytes))),
                parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra)

def _attention_sig(role: str, ctx: _Ctx, *, local_heads: int, local_kv_heads: int,
                   group_heads: int, width: int, value_width: int,
                   kv_row_bytes: float, extra: tuple[tuple[str, Any], ...] = ()) -> StaticSignature:
    return _sig(
        role,
        heads=(("q", int(local_heads)), ("kv", int(local_kv_heads)),
               ("g", int(group_heads)), ("dk", int(width)), ("dv", int(value_width))),
        quant=(("kv_bytes", float(kv_row_bytes)),
               ("act_bytes", dtype_bytes(ctx.features.activation_format))),
        parallel=_parallel(ctx), tiles=_tiles(ctx),
        extras=(("window", int(ctx.layer.window)),
                ("causal", int(not ctx.layer.noncausal_attention)),
                ("noncausal", int(ctx.layer.noncausal_attention)),
                ("cp", int(ctx.features.cp)),
                ("cp_plan", str(ctx.features.cp_plan)),
                ("cp_interleave", int(dict(ctx.features.metadata).get(
                    "cp_kv_cache_interleave_size", 128) or 128)),
                *extra),
    )

# ---------------------------------------------------------------------------
# Reusable ports

def _in(name: str, representation: str, **kwargs: Any) -> StatePortSpec:
    return port(name, "in", representation, **kwargs)

def _out(name: str, representation: str, **kwargs: Any) -> StatePortSpec:
    return port(name, "out", representation, **kwargs)

_HIDDEN_IN = _in("hidden", "activation")
_HIDDEN_OUT = _out("hidden", "activation")
_RESIDUAL_IN = _in("residual", "activation")
_RESIDUAL_OUT = _out("residual", "activation")
_KV_IN = _in("kv_cache", "kv_cache")
_KV_OUT = _out("kv_cache", "kv_cache")
_KV_READY_OUT = _out("kv_ready", "readiness", scope="stage")

@dataclass(frozen=True)
class _Built:
    template: SubgraphTemplate
    theta: tuple[tuple[str, StaticSignature], ...]

    def signature(self, name: str) -> StaticSignature:
        for key, value in self.theta:
            if key == name:
                return value
        raise ModelingError(f"builder_signature_missing:{self.template.kind}:{name}")

def _cross_facts(ctx: _Ctx) -> dict[str, Any]:
    routed = str(dict(_quant_fields(ctx, "routed_expert", 1, 1))["weight_format"]) if int(ctx.model.num_experts or 0) else ""
    kv_bf16 = int(bool(dict(ctx.features.metadata).get("kv_bf16_override")))
    return {"kv_dtype": ctx.kv_dtype, "kv_bf16": kv_bf16, "routed_weight_format": routed, "max_bucket": ctx.features.max_bucket}

#: Existing compiler output names for values whose names are not the default
#: ``<operator>_output``.  Kept here so the independent builder and the common
#: program name the same value; PW-CORE remains the consumer/owner of the
#: compiled output projection.
_OPERATOR_OUTPUT_ALIASES = {
    "qkv_projection": "qkv_activation", "q_down": "q_lora_activation",
    "q_up": "q_activation", "q_direct": "q_activation",
    "gate_up": "ffn_activation", "shared_gate_up": "shared_ffn_activation",
    "activation": "activation_intermediate",
    "shared_activation": "shared_activation_intermediate",
    "lm_head": "logits", "final_norm": "hidden_out",
}


def _operator_output_name(name: str) -> str:
    return _OPERATOR_OUTPUT_ALIASES.get(name, name + "_output")


def _port_representation(family: str) -> str:
    if family == "recurrent" or family in {"state_gather", "state_scatter"}:
        return "recurrent"
    if family == "kv_access":
        return "kv_cache"
    if family == "router":
        return "routing"
    if family == "transport" or family == "mc2_transport":
        return "traffic"
    if family == "attention" or family == "mla":
        return "attention"
    return "activation"


def _built(kind: str, ctx: _Ctx, variant: str, title: str,
           specs: Iterable[tuple[str, str, str]], signatures: Mapping[str, StaticSignature],
           *, children: tuple[str, ...] = (), fusions: tuple[FusionContract, ...] = (),
           state_in: tuple[StatePortSpec, ...] = (), state_out: tuple[StatePortSpec, ...] = (),
           coverage: str = "structural", contract: tuple[tuple[str, str], ...] = (),
           streams: Mapping[str, str] | None = None,
           operator_inputs: Mapping[str, tuple[StatePortSpec, ...]] | None = None,
           operator_outputs: Mapping[str, tuple[StatePortSpec, ...]] | None = None) -> _Built:
    # A builder-selected variant is part of the declared capability surface.
    # An undeclared value must fail here instead of silently naming a template
    # that the vocabulary cannot explain or a later stage cannot invalidate.
    if variant not in kind_spec(kind).variants:
        raise ModelingError(f"subgraph_variant_undeclared:{kind}:{variant}")
    operators, theta, facts, smap = [], [], _cross_facts(ctx), dict(streams or ())
    declared_inputs = dict(operator_inputs or ())
    declared_outputs = dict(operator_outputs or ())
    for name, family, role in specs:
        if name not in signatures:
            raise ModelingError(f"builder_signature_missing:{kind}:{name}")
        signature = signatures[name]
        inputs = list(declared_inputs.get(name, ()))
        # A quantized matrix consumes the producer's payload and scale as two
        # independent values.  They are explicit in the same signature that
        # carries the producer name; never fold the scale into the payload.
        for key, representation in (("activation_payload_input", "activation"),
                                    ("activation_scale_input", "scale")):
            value = signature.find(key, None)
            if value and not any(port.name == str(value) for port in inputs):
                inputs.append(_in(str(value), representation))
        outputs = tuple(declared_outputs.get(name, ()))
        if not outputs:
            scale_bytes = int(signature.find("activation_scale_bytes", 0) or 0)
            if (str(signature.find("native_kind", "")) == "DynamicMxQuant"
                    and scale_bytes > 0):
                # The native producer publishes payload and E8M0 scale as two
                # separately addressable outputs.
                outputs = (
                    _out(f"{name}_output_payload", "activation"),
                    _out(f"{name}_output_scale", "scale"),
                )
            else:
                outputs = (_out(_operator_output_name(name),
                                _port_representation(family)),)
        if len({port.name for port in outputs}) != len(outputs):
            raise ModelingError(f"operator_output_duplicate:{kind}:{name}")
        operators.append(OperatorTemplate(
            name=name, family=family, role=role, stream=smap.get(name, "main"),
            inputs=tuple(inputs), outputs=tuple(outputs)))
        theta.append((name, signature.relocated(**facts)))
    return _Built(SubgraphTemplate(kind=kind, variant=variant, title=title, operators=tuple(operators),
                  state_in=state_in, state_out=state_out, children=children, fusions=fusions,
                  coverage=coverage, contract=contract), tuple(theta))

def _variant(ctx: _Ctx, default: str) -> str:
    return ctx.variant or default

def _guard_low_act(specs: list, signatures: dict, ctx: _Ctx, name: str, *,
                   elements: int, category: str, dynamic_only: bool = False) -> None:
    if not name:
        return
    fmt = _matrix_act_format(ctx, category)
    if fmt not in _LOW_ACT or (specs and specs[-1][1] == "quantize"):
        return
    count = int(elements)
    if fmt in {"int8", "int4"}:
        # Per-token npu_dynamic_quant. Static int8 and FP8/MX do not qualify
        # and must not fall through to a one-pass stand-in.
        traffic = dynamic_quant_traffic(
            ctx.quant, category, count, fmt,
            str(ctx.features.activation_dtype or "bf16"))
        if traffic is None:
            return
        read_bytes, write_bytes, extra = traffic
        qname = f"{name}_QuantV2"
        specs.append((qname, "quantize", "quantize"))
        signatures[qname] = _elementwise(
            "quantize", ctx, elements=count, ops=2,
            read_bytes=read_bytes, write_bytes=write_bytes, extra=extra)
        return
    # New dense/GDN/qkv sites ask only for the dynamic int producer. FP8 and
    # MX stay on the call sites that already named them.
    if dynamic_only:
        return
    # MX is a resolved runtime method, not a bare E4M3 label.  The header
    # normalizes the stored weight dtype to fp8_e4m3; only linear_method
    # distinguishes the group-32 E8M0 DynamicMxQuant producer from regular
    # FP8 or INT8 quant rows.
    method = (ctx.quant.linear_method(category)
              if ctx.quant is not None else "dense")
    if method == "mx_fp8":
        from mindie_sim.prediction.mx_dynamic_quant import producer_contract
        stream_dtype = str(ctx.features.activation_dtype or "bf16")
        source_revision = str(dict(ctx.features.metadata).get(
            "mx_quant_source_revision", "unpinned"))
        contract = producer_contract(count, input_dtype=stream_dtype,
                                     source_revision=source_revision)
        # A1.o keeps the pre-existing QuantV2 graph slot/name so downstream
        # state/fusion identity is stable; other MX roles use the native name.
        # In both cases the program and ABI are DynamicMxQuant.
        qname = (f"{name}_QuantV2"
                 if category in {"attn.o", "attn.qkv"} else
                 f"{name}_DynamicMxQuant")
        specs.append((qname, "quantize", "quantize"))
        signatures[qname] = _elementwise(
            "quantize", ctx, elements=count, ops=1,
            read_bytes=float(contract["input_bytes"]),
            write_bytes=float(contract["write_bytes"]),
            extra=(("quant_op", "DynamicMxQuant"),
                   ("native_kind", "DynamicMxQuant"),
                   ("out_format", "fp8_e4m3"),
                   ("output_format", "fp8_e4m3"),
                   ("input_format", str(contract["input_format"])),
                   ("tiling_key", int(contract["tiling_key"])),
                   ("source_branch", str(contract["branch"])),
                   ("source_revision", str(contract["source_revision"])),
                   ("source_certified", int(bool(contract.get("source_certified")))),
                   ("activation_scale_format", "fp8_e8m0"),
                   ("activation_scale_shape", tuple(contract["scale_shape"])),
                   ("activation_payload_bytes", int(contract["payload_bytes"])),
                   ("activation_scale_bytes", int(contract["scale_bytes"])),
                   ("row_scale_bytes", int(contract["scale_bytes"])),
                   ("mx_scale_row_bytes", int(contract["scale_bytes"])),
                   ("mx_groups_per_row", int(contract["scale_bytes"])),
                   ("mx_vector_ops_per_row", int(contract["vector_ops"])),
                   ("mx_register_calls", int(contract["work"]["register_calls"]["total"])),
                   ("mx_work_kind", str(contract["work"]["work_kind"])),
                   ("mx_issue_cycles_known", int(bool(contract["work"]["issue_cycles_known"]))),
                   ("mx_algorithm", str(contract["work"]["algorithm"])),
                   ("mx_source", f"{contract['source_revision']}:{contract['source'].get('kernel', 'unpinned')}:{contract['source'].get('kernel_sha256', contract['source'].get('sha256', ''))}"),
                   ("category", category)))
        if name in signatures:
            signatures[name] = signatures[name].relocated(
                scale_producer=qname,
                activation_payload_input=f"{qname}_output_payload",
                activation_scale_input=f"{qname}_output_scale")
        return
    # No raw FP4 DynamicMxQuant producer is claimed here; only the resolved
    # mx_fp8 branch above has the source ABI and group-32 E8M0 contract.
    op = "QuantV2"
    qname = f"{name}_{op}"
    specs.append((qname, "quantize", "quantize"))
    signatures[qname] = _elementwise(
        "quantize", ctx, elements=count, ops=2,
        read_bytes=count * dtype_bytes(ctx.features.activation_dtype or "bf16"),
        write_bytes=count * dtype_bytes(fmt),
        extra=(("quant_op", op), ("out_format", fmt), ("category", category)))


def _precede(specs: list, signatures: dict, ctx: _Ctx, name: str,
             elements: int, category: str, *, dynamic_only: bool = False) -> None:
    """Insert the quant guard immediately before ``name`` when that operator exists."""
    if not name:
        return
    index = next((i for i, item in enumerate(specs) if item[0] == name), None)
    if index is None:
        return
    head = list(specs[:index])
    _guard_low_act(head, signatures, ctx, name, elements=elements, category=category,
                   dynamic_only=dynamic_only)
    if len(head) > index:
        specs[index:index] = head[index:]


from .builders_attention import BUILDERS as _ATTENTION_BUILDERS
from .builders_ffn import BUILDERS as _FFN_BUILDERS

_BUILDERS = {**_ATTENTION_BUILDERS, **_FFN_BUILDERS}

for _kind, _builder in _BUILDERS.items():
    register_builder(_kind, _builder)

@dataclass(frozen=True)
class LayerSlot:
    kind: str
    variant: str
    role: str

@dataclass(frozen=True)
class LayerInstances:
    layer: int
    stage: int
    subgraphs: tuple[SubgraphInstance, ...]
    slots: tuple[LayerSlot, ...]
    holes: tuple[CoverageHole, ...] = ()

    def find(self, kind: str, role: str | None = None) -> SubgraphInstance:
        for instance in self.subgraphs:
            if instance.kind != kind:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role or (role in {"pre", "post"} and inst_role.endswith("_" + role)):
                return instance
        raise ModelingError(f"subgraph_instance_missing:{kind}:{self.layer}:{role}")

@dataclass(frozen=True)
class InstantiatedModel:
    model: ModelSpec
    features: FeatureConfig
    layers: tuple[LayerInstances, ...]
    globals: tuple[SubgraphInstance, ...]
    coverage: CoverageReport
    stage_of: tuple[int, ...]

    @property
    def subgraphs(self) -> tuple[SubgraphInstance, ...]:
        return tuple(instance for layer in self.layers for instance in layer.subgraphs) + self.globals

    def find(self, kind: str, *, layer: int | None = None, role: str | None = None) -> SubgraphInstance:
        candidates = self.globals if layer is None else self.layers[layer].subgraphs
        for instance in candidates:
            if instance.kind != kind:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role or (role in {"pre", "post"} and inst_role.endswith("_" + role)):
                return instance
        raise ModelingError(f"subgraph_instance_missing:{kind}:{layer}:{role}")

    def find_attention(self, layer: int, role: str | None = None) -> SubgraphInstance:
        """The declared attention macro of one layer, whichever family it is.

        Callers that reason about the attention core (rather than one named
        family) should not re-derive the kind: MHA/GQA/MQA, MLA/DSA and
        SWA/CSA/CSA2/HCA are all valid answers for a declared layer.
        """
        for instance in self.layers[layer].subgraphs:
            if instance.kind not in ALL_ATTENTION_KINDS:
                continue
            inst_role = str(dict(instance.metadata).get("role", ""))
            if role is None or inst_role == role:
                return instance
        raise ModelingError(
            f"subgraph_attention_instance_missing:{layer}:{role}")

    def performance_keys(self) -> tuple[tuple, ...]:
        return tuple(instance.performance_key() for instance in self.subgraphs)

    def fingerprints(self) -> tuple[str, ...]:
        return tuple(instance.fingerprint() for instance in self.subgraphs)

def _stage_for(layer: int, layers: int, pp: int) -> int:
    if pp <= 1 or layers <= 1:
        return 0
    return min(pp - 1, layer * pp // layers)

def _validate_parallel(model: ModelSpec, features: FeatureConfig) -> None:
    """Reject layouts whose local ownership would silently drop heads."""
    tp = max(1, features.tp)
    if tp > 1:
        heads = int(model.num_heads)
        if heads % tp:
            raise ModelingError(f"heads_not_divisible_by_tp:{heads}%{tp}")
        for layer in model.layer_plans:
            local = int(layer.num_heads or heads)
            if local % tp:
                raise ModelingError(
                    f"layer_heads_not_divisible_by_tp:{layer.index}:{local}%{tp}")
    ep = max(1, features.ep)
    if ep > 1:
        declared = [int(model.num_experts or 0)]
        declared.extend(int(layer.num_experts or 0) for layer in model.layer_plans)
        for count in declared:
            if count and count % ep:
                raise ModelingError(f"experts_not_divisible_by_ep:{count}%{ep}")

def _gemma_facts(model: ModelSpec) -> dict:
    return dict(model.catalog_facts)

def _layer_has_ple(model: ModelSpec, layer: LayerPlan) -> bool:
    facts = _gemma_facts(model)
    layers = facts.get("gemma4_ple_layers")
    if layers is not None:
        return layer.index in {int(value) for value in layers}
    if facts.get("ple") is not None:
        return bool(facts.get("ple"))
    return bool(layer.engram and facts.get("gemma4"))

def _layer_slots(model: ModelSpec, layer: LayerPlan,
                 features: FeatureConfig) -> tuple[LayerSlot, ...]:
    slots: list[LayerSlot] = []
    hc = bool(layer.hyper_connections)

    def boundary(name: str) -> None:
        if hc:
            slots.append(LayerSlot("hyper_connection", name, name))

    if layer.residual_block_size:
        slots.append(LayerSlot("attn_residual", "block_stack", "residual"))
    slots.append(LayerSlot("layernorm", "rmsnorm", "attn_norm"))
    boundary("attn_pre")
    attention_kind = resolve_attention_kind(layer)
    window_compressed = str(layer.cache_kind or "") == "window_compressed"
    latent_attention = str(layer.attention or "") == "mla" and not window_compressed
    shared_consumer = (0 <= layer.kv_owner_layer_id < layer.index
                       and not layer.owns_long_kv)
    kv_slot = LayerSlot("kv_cache", "read" if shared_consumer else "store", "kv")

    def compression_slots() -> None:
        if layer.compress_ratio or window_compressed:
            slots.append(LayerSlot("kv_compressor",
                                   str(layer.compressor_kind or "weighted_pool"),
                                   "compress"))
        if layer.indexer:
            slots.append(LayerSlot("indexer_prolog",
                                   str(layer.indexer_type or "full"),
                                   "index_produce"))
            slots.append(LayerSlot("lightning_indexer",
                                   "candidate" if layer.candidate_filter else "topk",
                                   "index_select"))

    if window_compressed:
        # The window/compressed macro owns its own KV prolog, so the standalone
        # kv_cache slot only materialises the declared layout.
        slots.append(LayerSlot(attention_kind, "window_compressed",
                               "attention_prolog"))
        slots.append(LayerSlot("kv_cache", "layout", "kv"))
        compression_slots()
        slots.append(LayerSlot(attention_kind, "", "attention"))
    elif latent_attention:
        compression_slots()
        slots.append(LayerSlot(attention_kind,
                               "latent_mqa" if layer.mla_variant == "latent_mqa"
                               else "latent_mla", "attention"))
        if shared_consumer:
            slots.append(kv_slot)
    elif attention_kind in {"gdn_linear_attn", "kda_linear_attn"}:
        slots.append(LayerSlot(attention_kind, "", "attention"))
        compression_slots()
    else:
        variant = ("shared_consumer" if shared_consumer
                   else STANDARD_VARIANTS[attention_kind])
        slots.append(LayerSlot(attention_kind, variant, "attention"))
        slots.append(kv_slot)
        compression_slots()
    boundary("attn_post")
    # The post-attention residual is folded into the pre-FFN AddRmsNorm;
    # there is no third standalone residual N0 per layer.
    slots.append(LayerSlot("layernorm", "rmsnorm", "ffn_norm"))
    boundary("ffn_pre")
    if layer.ffn == "moe":
        slots.append(LayerSlot("moe_router", str(layer.router_mode or "softmax"), "router"))
        slots.append(LayerSlot("moe_token_dispatcher", features.moe_recipe, "transport"))
        slots.append(LayerSlot("routed_experts", "latent" if layer.expert_latent_size
                               else "packed", "routed_ffn"))
        slots.append(LayerSlot("moe_token_dispatcher", features.moe_recipe, "transport_combine"))
        if layer.num_shared_experts:
            slots.append(LayerSlot("shared_experts",
                               "gated" if layer.shared_expert_gate else "plain",
                               "shared_ffn"))
            slots.append(LayerSlot("shared_experts", "join", "join"))
    else:
        activation = str(layer.ffn_activation or "silu")
        if activation not in {"silu", "gelu", "gelu_tanh", "situ", "ungated"}:
            activation = "silu"
        slots.append(LayerSlot("mlp", activation, "dense_ffn"))
    boundary("ffn_post")
    if layer.engram:
        slots.append(LayerSlot("per_layer_embedding", "engram", "engram"))
    elif _layer_has_ple(model, layer):
        slots.append(LayerSlot("per_layer_embedding", "ple", "ple"))
    return tuple(slots)

def bind_instance_state_ports(template: SubgraphTemplate,
                             operators: Iterable[BoundOperator], *,
                             layer: int,
                             owner_layer: int | None = None
                             ) -> tuple[tuple[StatePortSpec, ...], tuple[StatePortSpec, ...]]:
    """Bind declared state ports to the actual instance owner/version.

    A template port is structural; ``owner`` defaults to ``self``.  A shared
    KV/index consumer must publish the owner it actually reads so the model
    and standalone paths, cache reuse and serialization all expose layer:0/1
    instead of ``self``.  Hidden/unrelated ports keep their declared owner.
    """
    operators = tuple(operators)
    kv_read_owners: list[int] = []
    kv_writers = False
    index_read_owners: list[int] = []
    index_writers = False
    selection_owners: list[int] = []
    compression_owners: list[int] = []
    for operator in operators:
        theta = operator.theta
        # Declared requirements are the source, independent of family/name:
        # A0 attention, A2 compressed attention and A1 MLA all expose
        # require_kv/require_index/require_selection on their actual operator.
        raw = theta.find("require_kv", None)
        if raw is not None:
            kv_read_owners.append(int(raw))
        raw = theta.find("require_index", None)
        if raw is not None:
            index_read_owners.append(int(raw))
        raw = theta.find("require_selection", None)
        if raw is not None:
            selection_owners.append(int(raw))
        raw = theta.find("compression_owner", None)
        if raw is not None:
            compression_owners.append(int(raw))
        if operator.name == "kv_store" or operator.role == "produce":
            if operator.family == "kv_access":
                kv_writers = True
        if operator.family == "indexer":
            raw = theta.find("owner_layer", None)
            if raw is not None and operator.name == "topk":
                selection_owners.append(int(raw))
        if operator.name == "index_cache_store":
            index_writers = True
        # layout_convert/copy only materialises a local view; it does not
        # publish a canonical persistent KV version.

    def unique(values: list[int], port: str) -> int | None:
        distinct = sorted(set(values))
        if len(distinct) > 1:
            raise ModelingError(
                f"instance_state_port_owner_ambiguous:{port}:{distinct}")
        return distinct[0] if distinct else None

    kv_read_owner = unique(kv_read_owners, "kv_cache")
    index_read_owner = unique(index_read_owners, "index_cache")
    selection_owner = unique(selection_owners, "index_selection")
    compression_owner = unique(compression_owners, "compress_state")
    if kv_read_owner is None and owner_layer is not None:
        kv_read_owner = int(owner_layer)

    state_in: list[StatePortSpec] = []
    state_out: list[StatePortSpec] = []
    for port in template.state_in:
        owner = None
        if port.name == "kv_cache" and kv_read_owner is not None:
            owner = f"layer:{kv_read_owner}"
        elif port.name == "index_cache" and index_read_owner is not None:
            owner = f"layer:{index_read_owner}"
        elif port.name == "index_selection" and selection_owner is not None:
            owner = f"layer:{selection_owner}"
        elif port.name in {"compress_state", "compressed_kv"} and compression_owner is not None:
            owner = f"layer:{compression_owner}"
        state_in.append(replace(port, owner=owner) if owner else port)
    for port in template.state_out:
        owner = None
        if port.name == "kv_cache":
            if kv_writers:
                owner = f"layer:{layer}"
            elif kv_read_owner is not None:
                owner = f"layer:{kv_read_owner}"
        elif port.name == "kv_ready":
            value = layer if kv_writers else kv_read_owner
            if value is not None:
                owner = f"layer:{value}"
        elif port.name == "cross_stage_ready" and port.owner in {"", "self"}:
            value = layer if kv_writers else kv_read_owner
            if value is not None:
                owner = f"layer:{value}"
        elif port.name == "index_cache":
            if index_writers:
                owner = f"layer:{layer}"
            elif index_read_owner is not None:
                owner = f"layer:{index_read_owner}"
        elif port.name == "index_ready":
            value = layer if index_writers else index_read_owner
            if value is not None:
                owner = f"layer:{value}"
        elif port.name == "index_selection" and selection_owner is not None:
            owner = f"layer:{selection_owner}"
        elif port.name in {"compress_state", "compressed_kv"}:
            value = layer if compression_owner is None else compression_owner
            owner = f"layer:{value}"
        state_out.append(replace(port, owner=owner) if owner else port)
    return tuple(state_in), tuple(state_out)


def instantiate_layer(model: ModelSpec, layer: LayerPlan,
                      features: FeatureConfig, *,
                      registry: TemplateRegistry | None = None,
                      layers: int | None = None,
                      stage: int | None = None) -> LayerInstances:
    total = int(layers if layers is not None else len(model.layer_plans) or model.layers)
    _validate_parallel(model, features)
    stage_value = _stage_for(layer.index, total, features.pp) if stage is None else int(stage)
    registry = registry if registry is not None else _DEFAULT_REGISTRY
    slots = _layer_slots(model, layer, features)
    validate_combination(slot.kind for slot in slots)
    subgraphs: list[SubgraphInstance] = []
    holes: list[CoverageHole] = []
    for slot in slots:
        ctx = _Ctx(model=model, layer=layer, features=features, stage=stage_value,
                   role=slot.role, variant=slot.variant, layers=total)
        built = build(slot.kind, ctx)
        template = registry.intern(built.template)
        bound = tuple(BoundOperator(op, built.signature(op.name))
                      for op in template.operators)
        owner_layer = (int(layer.kv_owner_layer_id)
                       if 0 <= layer.kv_owner_layer_id < layer.index else None)
        state_in, state_out = bind_instance_state_ports(
            template, bound, layer=layer.index, owner_layer=owner_layer)
        instance = SubgraphInstance(
            instance_id=f"{slot.kind}:{template.variant}:L{layer.index}:{slot.role}",
            template=template, layer=layer.index, stage=stage_value,
            operators=bound, state_in=state_in,
            state_out=state_out,
            owner_layer=owner_layer,
            metadata=(("role", slot.role), ("variant", slot.variant),
                      ("attention_kind", resolve_attention_kind(layer))),
        )
        subgraphs.append(instance)
        if template.coverage != "structural":
            holes.append(CoverageHole(slot.kind, "declared_partial", template.title))
    return LayerInstances(layer.index, stage_value, tuple(subgraphs), slots, tuple(holes))

_DEFAULT_REGISTRY = TemplateRegistry()

def _has_ple(model: ModelSpec) -> bool:
    facts = dict(model.catalog_facts)
    metadata = dict(model.metadata)
    return bool(facts.get("ple") or metadata.get("ple") or facts.get("gemma_ple"))

def instantiate_model(model: ModelSpec, features: FeatureConfig, *,
                      registry: TemplateRegistry | None = None) -> InstantiatedModel:
    if not model.layer_plans:
        raise ModelingError("instantiate_requires_explicit_layer_plans")
    registry = registry if registry is not None else _DEFAULT_REGISTRY
    total = len(model.layer_plans)
    layers = tuple(
        instantiate_layer(model, layer, features, registry=registry,
                          layers=total, stage=_stage_for(layer.index, total, features.pp))
        for layer in model.layer_plans)
    globals_: list[SubgraphInstance] = []
    holes: list[CoverageHole] = []

    def add_global(kind: str, variant: str, role: str, stage: int) -> None:
        ctx = _Ctx(model=model, layer=model.layer_plans[0], features=features,
                   stage=min(features.pp - 1, max(0, stage)), role=role,
                   variant=variant, layers=total)
        built = build(kind, ctx)
        template = registry.intern(built.template)
        bound = tuple(BoundOperator(op, built.signature(op.name))
                      for op in template.operators)
        state_in, state_out = bind_instance_state_ports(
            template, bound, layer=0, owner_layer=None)
        globals_.append(SubgraphInstance(
            instance_id=f"{kind}:{template.variant}:G:{role}", template=template,
            layer=0, stage=min(features.pp - 1, max(0, stage)), operators=bound,
            state_in=state_in, state_out=state_out,
            metadata=(("role", role), ("global", True))))
        if template.coverage != "structural":
            holes.append(CoverageHole(kind, "declared_partial", template.title))

    add_global("vocab_parallel_embedding",
               "media" if model.vision is not None else "token", "input", 0)
    if model.vision is not None:
        add_global("mm_encoder_projector", "vision", "media", 0)
    if model.has_optional_draft or model.is_standalone_draft:
        draft_var = ("dspark" if model.auxiliary and model.auxiliary.dspark and not model.auxiliary.mtp
                     else ("mtp" if model.has_optional_draft else "standalone"))
        add_global("spec_decode", draft_var, "draft", features.pp - 1)
    add_global("lm_head_sampler", features.final_norm_scope, "output",
               features.pp - 1)
    entries = []
    for layer in layers:
        for instance in layer.subgraphs:
            entries.append((instance.instance_id, f"{instance.kind}:{instance.variant}"))
    for instance in globals_:
        entries.append((instance.instance_id, f"{instance.kind}:{instance.variant}"))
    holes.extend(layer_hole for layer in layers for layer_hole in layer.holes)
    if _has_ple(model) and not any(
            layer_slot.kind == "per_layer_embedding" for layer in layers
            for layer_slot in layer.slots):
        if not dict(model.catalog_facts).get("gemma4"):
            holes.append(CoverageHole("per_layer_embedding", "ple_not_instantiated",
                                      "PLE declared without an explicit per-layer table record"))
    coverage = CoverageReport(tuple(entries), tuple(holes))
    return InstantiatedModel(
        model=model, features=features, layers=layers, globals=tuple(globals_),
        coverage=coverage, stage_of=tuple(layer.stage for layer in layers))

def instantiate(model: ModelSpec, features: FeatureConfig | None = None, *,
                runtime: Any = None, registry: TemplateRegistry | None = None,
                machine: Any | None = None) -> InstantiatedModel:
    return instantiate_model(
        model, features or FeatureConfig.from_runtime(runtime, model, machine=machine),
        registry=registry)


# ---------------------------------------------------------------------------
# Declared-kind disposition / standalone level entry

def subgraph_dispositions(instantiated: InstantiatedModel) -> dict[str, dict[str, Any]]:
    """Account for every declared kind, not only the kinds this model selected.

    The coverage report can only list instances a model actually produced, so
    a model that uses no multimodal projector leaves ``V0`` absent rather than
    explained.  This projection keeps the finite vocabulary authoritative:
    each declared kind receives exactly one disposition, derived from the
    instantiated structure and the registered builders, never from a model
    label or measurement.  ``not_selected`` is a structural statement about
    this model, not a claim that the mechanism is unsupported.
    """
    selected: dict[str, list[SubgraphInstance]] = {}
    for instance in instantiated.subgraphs:
        selected.setdefault(instance.kind, []).append(instance)
    result: dict[str, dict[str, Any]] = {}
    for spec in vocabulary():
        instances = tuple(selected.get(spec.kind, ()))
        if not instances:
            has_builder = spec.kind in KIND_BUILDERS
            result[spec.kind] = {
                "status": "not_selected" if has_builder else "missing_builder",
                "variants": (),
                "instances": 0,
                "layers": (),
                "coverage": (),
                "reason": ("model_structure_selects_no_instance" if has_builder
                           else "builder_not_registered"),
            }
            continue
        coverage = tuple(sorted({instance.template.coverage for instance in instances}))
        partial = any(value != "structural" for value in coverage)
        result[spec.kind] = {
            "status": "partial" if partial else "instantiated",
            "variants": tuple(sorted({instance.variant for instance in instances})),
            "instances": len(instances),
            "layers": tuple(sorted({instance.layer for instance in instances})),
            "coverage": coverage,
            "reason": ("declared_partial" if partial else ""),
        }
    return result



def _instance_feature_metadata(instance: SubgraphInstance) -> tuple[tuple[str, Any], ...]:
    """Return the source/dispatch metadata bound at standalone construction."""
    stored = dict(instance.metadata).get("feature_metadata")
    return tuple(stored) if stored else ()


def _built_structure(built: _Built) -> tuple:
    """Structural work identity, excluding the variant label and title.

    Used by the standalone entry to reject a request that only renames the
    natural template: if operators, static signatures, ports, fusions and
    declared contract are all unchanged, the requested variant has no
    corresponding physical construction.
    """
    template = built.template
    return (
        tuple(operator.key() for operator in template.operators),
        tuple((name, signature.key()) for name, signature in built.theta),
        tuple(tuple(port.to_dict().items()) for port in template.state_in),
        tuple(tuple(port.to_dict().items()) for port in template.state_out),
        template.children,
        template.fusions,
        template.coverage,
        template.contract,
    )


def _production_slot_variant(model: ModelSpec, plan: LayerPlan,
                            features: FeatureConfig, kind: str,
                            role: str) -> str | None:
    """The variant the model assembly would select for this kind/role.

    A production slot may intentionally use a label that aliases the builder's
    generic default (for example a one-KV-head GQA slot labelled ``gqa`` while
    the builder's default is ``mqa``).  Such a label is not a caller rename:
    it is the selection the real model path makes, so the standalone entry
    must accept it and reproduce the same structure.
    """
    if kind == "vocab_parallel_embedding":
        return "media" if model.vision is not None else "token"
    if kind == "mm_encoder_projector":
        return "vision" if model.vision is not None else "unified"
    if kind == "lm_head_sampler":
        return features.final_norm_scope
    if kind == "spec_decode":
        if not (model.has_optional_draft or model.is_standalone_draft):
            return None
        return ("dspark" if (model.auxiliary and model.auxiliary.dspark
                             and not model.auxiliary.mtp)
                else ("mtp" if model.has_optional_draft else "standalone"))
    try:
        slots = _layer_slots(model, plan, features)
    except ModelingError:
        return None
    for slot in slots:
        if slot.kind == kind and slot.role == role:
            return slot.variant
    return None


def build_standalone(kind: str, *, variant: str | None = None,
                     role: str = "standalone",
                     model: ModelSpec | Mapping[str, Any] | None = None,
                     layer: LayerPlan | Mapping[str, Any] | None = None,
                     features: FeatureConfig | Mapping[str, Any] | None = None,
                     stage: int | None = None,
                     registry: TemplateRegistry | None = None) -> SubgraphInstance:
    """Build one canonical subgraph from explicit structural facts.

    This is the independent subgraph level: the caller supplies model geometry
    (hidden width, heads, experts, MLA/media/draft records) and the structural
    position (``role``/``variant``/``stage``), not a model tree with a fake
    layer.  ``layer`` is an optional structural record; when omitted the
    model's first declared plan supplies the defaults.  The returned template
    carries no layer or request identity, so a repeated use interns the same
    object; the returned *instance* keeps the real layer/stage/owner identity
    because index/KV state effects are published on that namespace.

    ``variant`` is a request, not a label: an undeclared name is rejected, and
    a builder that cannot construct the requested structure must return an
    explicit conflict/gap instead of silently relabeling its natural variant.
    ``evaluate_standalone``/``evaluate_standalone_graph``/``standalone_span``
    submit the instance to the existing piecewise program; this function
    performs no I/O, device access or measurement.
    """
    validate_kind(kind)
    if model is None:
        raise ModelingError("standalone_model_facts_required")
    requested = None if variant is None else str(variant)
    if requested is not None and requested not in kind_spec(kind).variants:
        raise ModelingError(f"subgraph_variant_undeclared:{kind}:{requested}")
    if isinstance(model, Mapping):
        raw = dict(model)
        raw.setdefault("name", "standalone-subgraph")
        raw.setdefault("layers", 1)
        if "layer_plans" not in raw:
            plan_raw = dict(layer) if isinstance(layer, Mapping) else {"index": 0}
            plan_raw.setdefault("index", 0)
            raw["layer_plans"] = [plan_raw]
        model = ModelSpec.from_mapping(raw)
    if not isinstance(model, ModelSpec):
        raise ModelingError("standalone_model_facts_invalid")
    base = model.layer_plans[0] if model.layer_plans else LayerPlan(index=0)
    if layer is None:
        plan = base
    elif isinstance(layer, Mapping):
        plan = replace(base, **{str(key): value for key, value in layer.items()})
    elif isinstance(layer, LayerPlan):
        plan = layer
    else:
        raise ModelingError("standalone_layer_plan_invalid")
    if plan.index < 0:
        raise ModelingError("standalone_layer_index_invalid")
    if features is None:
        resolved_features = FeatureConfig.from_model(model)
    elif isinstance(features, Mapping):
        resolved_features = FeatureConfig.from_model(model, **dict(features))
    elif isinstance(features, FeatureConfig):
        resolved_features = features
    else:
        raise ModelingError("standalone_features_invalid")
    total_layers = max(1, len(model.layer_plans), plan.index + 1)
    if stage is None:
        stage_value = _stage_for(plan.index, total_layers, resolved_features.pp)
    else:
        stage_value = int(stage)
        if stage_value < 0 or stage_value >= resolved_features.pp:
            raise ModelingError(f"standalone_stage_invalid:{stage_value}")
    ctx = _Ctx(model=model, layer=plan, features=resolved_features,
               stage=stage_value, role=str(role),
               variant=requested or "", layers=total_layers,
               requested_kind=kind)
    slot_variant = _production_slot_variant(
        model, plan, resolved_features, kind, str(role))
    built = build(kind, ctx)
    if built.template.kind != kind:
        # A request that names a kind of another macro family is a missing
        # construction, not a silent relabel.
        raise ModelingError(
            f"subgraph_kind_not_constructed:{kind}:{built.template.kind}")
    if requested is not None:
        if built.template.variant != requested:
            raise ModelingError(
                f"subgraph_variant_not_constructed:{kind}:{requested}:"
                f"{built.template.variant}")
        # A requested variant must change the physical construction, not just
        # the label.  Build the natural variant of the same plan and compare
        # operators/static signatures/ports/fusions; an identical body is a
        # renaming, not a modeled alternative.  When the natural build cannot
        # be formed, there is no positive evidence of a rename and the
        # requested builder's own explicit error (if any) is authoritative.
        try:
            natural = build(kind, replace(ctx, variant=""))
        except ModelingError:
            natural = None
        if natural is not None and natural.template.variant != requested:
            if (_built_structure(built) == _built_structure(natural)
                    and requested != slot_variant):
                raise ModelingError(
                    f"subgraph_variant_label_only:{kind}:{requested}:"
                    f"{natural.template.variant}")
    target = registry if registry is not None else _DEFAULT_REGISTRY
    template = target.intern(built.template)
    bound = tuple(BoundOperator(operator, built.signature(operator.name))
                  for operator in template.operators)
    owner_raw = int(plan.kv_owner_layer_id)
    owner_layer = owner_raw if 0 <= owner_raw < plan.index else None
    state_in, state_out = bind_instance_state_ports(
        template, bound, layer=plan.index, owner_layer=owner_layer)
    return SubgraphInstance(
        instance_id=(f"{kind}:{template.variant}:standalone:"
                     f"L{plan.index}:S{stage_value}:{role}"),
        template=template, layer=plan.index, stage=stage_value, operators=bound,
        state_in=state_in, state_out=state_out,
        owner_layer=owner_layer,
        metadata=(("role", str(role)), ("standalone", True),
                  ("variant", template.variant),
                  ("feature_metadata", tuple(resolved_features.metadata)),
                  ("owner_layer", owner_layer)))


def occurrence_contract(instance: SubgraphInstance) -> dict[str, Any]:
    """Producer/consumer and stream-order contract for the common assembly.

    This is a projection of the existing ``OperatorTemplate`` inputs/outputs,
    the per-operator theta and the declared stream order, not a second IR.
    Outputs carry the selected dtype/layout/output slot when theta declares
    them; an input with no producer in this instance is reported as unresolved
    instead of being silently dropped.  ``PW-CORE`` consumes the same contract
    to put every occurrence on its stream and add cross-stream completion
    edges.
    """
    streams = {operator.name: operator.stream for operator in instance.operators}
    contract = dict(instance.template.contract)
    producer_by_value: dict[str, str] = {}
    contract_views = tuple(contract.get("qkv_views", ()) or ())

    def _view_input(view: tuple, name: str) -> dict[str, Any]:
        token = view[9]
        layout = TensorLayout(
            shape=tuple(token[0]), strides=tuple(token[1]),
            offset=int(token[2]), storage_shape=tuple(token[3]))
        storage = str(view[10]) if len(view) > 10 else ""
        tensor = TensorValue(
            producer=str(view[2]), dtype=str(view[5]), layout=layout,
            output=str(view[3]), alias_of=str(view[7]),
            storage=storage, state_version=None)
        value = {
            "name": name, "producer": str(view[2]), "dtype": str(view[5]),
            "layout": layout.to_dict(), "output": str(view[3]),
            "tensor": tensor.to_dict(), "storage": storage,
            "alias_of": str(view[7]), "phase": str(view[6]),
        }
        return {"name": name, "producer": str(view[2]), "value": value,
                "source": "contract_view"}

    attention_view_inputs: dict[str, dict[str, Any]] = {}
    for view in contract_views:
        target = {"query": "query", "key": "key", "value": "value",
                  "cache_key": "kv_cache_k",
                  "cache_value": "kv_cache_v"}.get(view[0])
        if target is not None:
            attention_view_inputs[target] = _view_input(view, target)
    occurrence_rows: list[dict[str, Any]] = []
    output_values: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for index, operator in enumerate(instance.operators):
        theta = operator.theta
        outputs = []
        for port in operator.template.outputs:
            declared_dtypes = dict(theta.find("output_dtypes", ()) or ())
            dtype = str(declared_dtypes.get(port.name, "")
                        or theta.find(f"{port.name}_dtype", None)
                        or theta.find("output_format", None)
                        or theta.find("act_format", None)
                        or theta.find("out_format", None)
                        or "")
            scale_shape: tuple[int, ...] = ()
            if port.name.endswith("_output_scale"):
                dtype = str(theta.find("activation_scale_format", None)
                            or "fp8_e8m0")
                scale_shape = tuple(int(value) for value in (
                    theta.find("activation_scale_shape", ()) or ()))
                row_bytes_value = theta.find("activation_scale_bytes", None)
                shape = (0,) + scale_shape
            elif port.name.endswith("_output_payload"):
                dtype = str(theta.find("out_format", None) or dtype)
                row_bytes_value = theta.find("activation_payload_bytes", None)
                try:
                    shape = (0, int(row_bytes_value))
                except (TypeError, ValueError):
                    shape = (0,)
            elif port.name in {"gdn_decay_fp32", "gdn_beta_bf16"}:
                heads = int(theta.find("value_heads", 0) or 0)
                if not dtype:
                    dtype = "fp32" if port.name == "gdn_decay_fp32" else "bf16"
                shape = (0, heads)
                row_bytes_value = heads * dtype_bytes(dtype)
            else:
                n = theta.find("N", None)
                elements = theta.find("elements", None)
                if n is not None:
                    try:
                        shape = (0, int(n))
                    except (TypeError, ValueError):
                        shape = (0,)
                elif elements is not None:
                    try:
                        shape = (0, int(elements))
                    except (TypeError, ValueError):
                        shape = (0,)
                else:
                    shape = (0,)
                row_bytes_value = theta.find("row_bytes", None)
            if not dtype:
                act_bytes = theta.find("act_bytes", None)
                try:
                    dtype = {1: "int8", 2: "bf16", 4: "fp32"}.get(
                        int(act_bytes), "")
                except (TypeError, ValueError):
                    dtype = ""
            strides = []
            stride = 1
            for size in reversed(shape):
                strides.append(stride)
                stride *= max(1, size)
            layout = TensorLayout(tuple(shape), tuple(reversed(strides)),
                                  storage_shape=tuple(shape))
            value = {
                "name": port.name,
                "producer": operator.name,
                "dtype": dtype,
                "layout": layout.to_dict(),
                "row_bytes": (float(row_bytes_value)
                              if row_bytes_value is not None else None),
                "rows": 1,
                "output": port.name,
                "quant": str(theta.find("quant_op", None)
                             or theta.find("linear_method", None) or ""),
                "storage": str(theta.find("storage", None) or ""),
                "alias_of": str(theta.find("alias_of", None) or ""),
                "state_version": (theta.find("state_version", None)
                                  if isinstance(theta.find("state_version", None), int)
                                  else None),
                "index": (theta.find("require_index", None)
                          if theta.find("require_index", None) is not None
                          else theta.find("owner_layer", None)),
                "attrs": tuple(
                    (key, theta.find(key, None))
                    for key in ("tiling_key", "input_format", "source_branch",
                                "source_revision", "scale_producer",
                                "activation_payload_input",
                                "activation_scale_input", "category",
                                "owner_layer", "require_index",
                                "require_selection", "compression_owner")
                    if theta.find(key, None) is not None),
                "scale": (
                    {"dtype": dtype, "shape": scale_shape}
                    if port.name.endswith("_output_scale") else None),
            }
            if dtype:
                value["tensor"] = TensorValue(
                    producer=operator.name, dtype=dtype, layout=layout,
                    output=port.name,
                    quant=str(theta.find("quant_op", None)
                              or theta.find("linear_method", None) or ""),
                    storage=str(theta.find("storage", None) or ""),
                    alias_of=str(theta.find("alias_of", None) or ""),
                    state_version=value["state_version"]).to_dict()
            base_tensor = contract.get("qkv_tensor")
            if (isinstance(base_tensor, TensorValue)
                    and port.name == contract.get("qkv_output")):
                value["tensor"] = base_tensor.to_dict()
                value["dtype"] = base_tensor.dtype
                value["layout"] = base_tensor.layout.to_dict()
                value["output"] = port.name
            output_values[port.name] = value
            producer_by_value.setdefault(port.name, operator.name)
            outputs.append(value)
        inputs = []
        for port in operator.template.inputs:
            if operator.name == "attention" and port.name in attention_view_inputs:
                inputs.append(attention_view_inputs[port.name])
                continue
            producer = producer_by_value.get(port.name)
            if producer is None:
                item = {"name": port.name, "producer": None,
                        "reason": "no_producer_in_instance"}
                unresolved.append({"consumer": operator.name,
                                   "value": port.name,
                                   "reason": "no_producer_in_instance"})
            else:
                item = {"name": port.name, "producer": producer,
                        "value": output_values.get(port.name)}
            inputs.append(item)
        occurrence_rows.append({
            "name": operator.name,
            "stream": operator.stream,
            "index": index,
            "inputs": tuple(inputs),
            "outputs": tuple(outputs),
        })
    edges: list[dict[str, Any]] = []
    for operator in instance.operators:
        for port in operator.template.inputs:
            value_name = port.name
            producer = producer_by_value.get(port.name)
            if operator.name == "attention" and port.name in attention_view_inputs:
                item = attention_view_inputs[port.name]
                producer = item["producer"]
                value_name = item["value"]["output"]
            if producer is None or producer == operator.name:
                continue
            producer_stream = streams.get(producer, "external")
            consumer_stream = streams[operator.name]
            edges.append({
                "producer": producer,
                "consumer": operator.name,
                "value": value_name,
                "producer_stream": producer_stream,
                "consumer_stream": consumer_stream,
                "cross_stream": (producer_stream != "external"
                                 and producer_stream != consumer_stream),
            })
    ordered: dict[str, list[str]] = {}
    for operator in instance.operators:
        ordered.setdefault(operator.stream, []).append(operator.name)
    # State access identity comes from actual read/write operators and their
    # declared ranges, never from a generic state_in/out name sweep.  A port
    # with no physical writer stays unresolved instead of becoming a 0 B
    # persistent write.
    state_accesses: list[dict[str, Any]] = []
    resolved_state_ports: set[str] = set()

    def access_version(operator, write: bool):
        keys = ("state_version", "state_commit_versions") if write else (
            "state_version", "state_read_versions")
        for key in keys:
            raw = operator.theta.find(key, None)
            if isinstance(raw, int):
                return raw
            if isinstance(raw, (tuple, list)) and raw:
                values = [item for item in raw if isinstance(item, int)]
                if values:
                    return values[0]
        return None

    def add_access(operator, direction: str, representation: str,
                   output: str, producer: str, owner: str, storage: str,
                   position: str, row_bytes: Any, *, rows: int = 1,
                   stage: int | None = None, persistent: bool = True,
                   temporary: bool = False, source: str = "",
                   reader: str | None = None) -> None:
        try:
            per_row = float(row_bytes) if row_bytes is not None else 0.0
        except (TypeError, ValueError):
            per_row = 0.0
        state_accesses.append({
            "occurrence": f"{instance.instance_id}:{operator.name}",
            "producer": producer,
            "request": None,
            "stage": instance.stage if stage is None else stage,
            "layer": instance.layer,
            "direction": direction,
            "position": position,
            "output": output,
            "storage": storage,
            "owner": owner,
            "representation": representation,
            "persistent": persistent,
            "temporary": temporary,
            "state_version": access_version(operator, direction == "write"),
            "span": ((0, int(rows)),),
            "row_bytes": per_row,
            "rows": int(rows),
            "rank": None,
            "scope": "target",
            "bytes": per_row * int(rows),
            "source": source or owner,
            "reader": reader,
            "grant": None,
            "decision": None,
            "last_use": None,
        })

    has_gather_scatter = any(
        operator.name in {"gather_recurrent_state", "scatter_recurrent_state"}
        for operator in instance.operators)
    for operator in instance.operators:
        theta = operator.theta
        name = operator.name
        rows = int(theta.find("rows", 1) or 1)
        if name == "kv_store":
            owner = f"layer:{instance.layer}"
            add_access(operator, "write", "kv_cache", "kv_cache", name, owner,
                       f"kv_cache:{owner}", "cache",
                       theta.find("row_bytes", None), rows=rows,
                       persistent=True)
            resolved_state_ports.add("kv_cache")
        elif name == "layout_convert":
            layout_bytes = theta.find("write_bytes", None)
            if layout_bytes is None:
                layout_bytes = theta.find("row_bytes", None)
            if layout_bytes is None:
                elements = theta.find("elements", None)
                element_bytes = theta.find("act_bytes", None)
                if elements is not None and element_bytes is not None:
                    try:
                        layout_bytes = float(elements) * float(element_bytes)
                    except (TypeError, ValueError):
                        layout_bytes = 0.0
            add_access(operator, "write", "layout", "layout_convert_output",
                       name, f"layout:temp:{instance.instance_id}",
                       f"layout:temp:{instance.instance_id}", "temp",
                       layout_bytes, rows=1,
                       persistent=False, temporary=True)
        elif name == "kv_read":
            raw_owner = theta.find("owner_layer", theta.find("require_kv", None))
            owner = (f"layer:{raw_owner}" if raw_owner is not None
                     else f"layer:{instance.layer}")
            add_access(operator, "read", "kv_cache", "kv_cache", name, owner,
                       f"kv_cache:{owner}", "cache",
                       theta.find("row_bytes", None), rows=rows,
                       persistent=True, source=owner)
            resolved_state_ports.add("kv_cache")
        elif name == "stage_transfer":
            raw_owner = theta.find("owner_layer", instance.layer)
            owner = f"layer:{raw_owner}"
            destinations = tuple(theta.find("destination_stages", ()) or ())
            if not destinations:
                destinations = (int(theta.find("destination_stage", instance.stage)),)
            for destination in destinations:
                add_access(operator, "write", "kv_cache", "kv_cache", name,
                           owner, f"kv_cache:{owner}/stage:{destination}",
                           "cache", theta.find("row_bytes", None), rows=rows,
                           stage=int(destination), persistent=True,
                           source=owner)
        elif name in {"gather_recurrent_state", "scatter_recurrent_state"}:
            direction = "read" if name.startswith("gather") else "write"
            add_access(operator, direction, "recurrent", "recurrent_state",
                       name, f"layer:{instance.layer}",
                       f"recurrent:layer:{instance.layer}",
                       "slot", theta.find("row_bytes", None), rows=rows,
                       source=f"layer:{instance.layer}")
            resolved_state_ports.add("recurrent_state")
        elif name in {"conv"} and operator.family == "recurrent":
            owner = f"layer:{instance.layer}"
            state_row_bytes = theta.find(
                "state_row_bytes", theta.find("row_bytes", None))
            for direction in ("read", "write"):
                add_access(operator, direction, "conv", "conv_state", name,
                           owner, f"conv:{owner}", "slot",
                           state_row_bytes, rows=1, source=owner)
            resolved_state_ports.add("conv_state")
        elif name in {"gated_delta", "kda_recurrent"}:
            owner = f"layer:{instance.layer}"
            local = has_gather_scatter
            for direction in ("read", "write"):
                add_access(
                    operator, direction, "recurrent",
                    "local_recurrent_state" if local else "recurrent_state",
                    name, owner,
                    f"recurrent-local:{owner}" if local
                    else f"recurrent:{owner}",
                    "local" if local else "slot",
                    theta.find("row_bytes", None), rows=rows,
                    persistent=not local, temporary=local, source=owner)
            if not local:
                resolved_state_ports.add("recurrent_state")
        elif name == "attention" and theta.find("require_kv", None) is not None:
            owner = f"layer:{theta.find('require_kv')}"
            add_access(operator, "read", "kv_cache", "kv_cache", name, owner,
                       f"kv_cache:{owner}", "cache",
                       theta.find("kv_bytes", theta.find("kv_row_bytes", None)), rows=1,
                       source=owner, reader=instance.instance_id)
            resolved_state_ports.add("kv_cache")
        elif name == "index_cache_store":
            add_access(operator, "write", "index", "index_cache", name,
                       f"layer:{instance.layer}",
                       f"index:layer:{instance.layer}", "cache",
                       theta.find("row_bytes", None), rows=rows)
            resolved_state_ports.add("index_cache")
        elif name == "index_score":
            raw_owner = theta.find("require_index", None)
            if raw_owner is not None:
                owner = f"layer:{raw_owner}"
                add_access(operator, "read", "index", "index_cache", name,
                           owner, f"index:{owner}", "cache",
                           theta.find("index_bytes", None), rows=rows,
                           source=owner)
                resolved_state_ports.add("index_cache")
        elif name == "topk":
            raw_owner = theta.find("owner_layer", None)
            if raw_owner is not None:
                owner = f"layer:{raw_owner}"
                add_access(operator, "write", "selection", "index_selection",
                           name, owner, f"selection:{owner}", "cache",
                           theta.find("index_bytes", None), rows=rows)
                resolved_state_ports.add("index_selection")
        elif name in {"compress", "compression"}:
            add_access(operator, "write", "compress_state", "compress_state",
                       name, f"layer:{instance.layer}",
                       f"compress:layer:{instance.layer}", "slot",
                       theta.find("row_bytes", None), rows=rows)
            resolved_state_ports.add("compress_state")

    unresolved_state_outputs = tuple(
        port.name for port in instance.state_out
        if port.name not in resolved_state_ports
        and port.representation in {"kv_cache", "recurrent", "conv", "index",
                                    "selection", "compress_state"})
    return {
        "instance_id": instance.instance_id,
        "kind": instance.kind,
        "variant": instance.variant,
        "occurrences": tuple(occurrence_rows),
        "streams": tuple((stream, tuple(names))
                         for stream, names in ordered.items()),
        "edges": tuple(edges),
        "cross_stream_edges": tuple(edge for edge in edges
                                    if edge["cross_stream"]),
        "unresolved_inputs": tuple(unresolved),
        "unresolved_state_outputs": unresolved_state_outputs,
        "state_accesses": tuple(state_accesses),
        "qkv_tensor": (contract.get("qkv_tensor").to_dict()
                       if isinstance(contract.get("qkv_tensor"), TensorValue)
                       else None),
        "qkv_producer": contract.get("qkv_producer"),
        "qkv_output": contract.get("qkv_output"),
        "qkv_slices": contract.get("qkv_slices"),
        "views": tuple(contract.get("qkv_views", ()) or ()),
        "view_tensors": tuple(
            {"role": view[0], "phase": view[6],
             "tensor": TensorValue(
                 producer=str(view[2]), dtype=str(view[5]),
                 layout=TensorLayout(
                     shape=tuple(view[4][0]), strides=tuple(view[4][1]),
                     offset=int(view[4][2]),
                     storage_shape=tuple(view[4][3])),
                 output=str(view[3]), alias_of=str(view[7]),
                 storage=(str(view[10]) if len(view) > 10 else ""),
                 state_version=None).to_dict(),
             "token_layout": view[9]}
            for view in (contract.get("qkv_views", ()) or ())
            if len(view) >= 8 and view[5]),
    }


def _binding_rows(geometry: Any) -> int:
    if geometry is None:
        return 1
    if isinstance(geometry, Mapping):
        for key in ("tokens", "effective_tokens", "rows"):
            if geometry.get(key) is not None:
                return max(1, int(geometry[key]))
        return 1
    for name in ("tokens", "effective_tokens", "rows"):
        value = getattr(geometry, name, None)
        if value is not None:
            return max(1, int(value))
    return 1


def _binding_cached_rows(geometry: Any) -> int:
    if geometry is None:
        return 0
    if isinstance(geometry, Mapping):
        for key in ("cached_tokens", "kv_tokens", "context_tokens", "context"):
            if geometry.get(key) is not None:
                return max(0, int(geometry[key]))
        return 0
    for name in ("cached_tokens", "kv_tokens", "context_tokens", "context"):
        value = getattr(geometry, name, None)
        if value is not None:
            return max(0, int(value))
    return 0


def _binding_key(occurrence: Occurrence) -> str:
    return (f"{occurrence.name}:{occurrence.serial}:"
            f"{occurrence.rank if occurrence.rank is not None else 'n'}")


def _binding_request_rows(geometry: Any, occurrence: Occurrence,
                         rows: int) -> tuple[tuple[str, int, int, int], ...]:
    """Original lane -> (request, row_start, row_end, request_slot)."""
    raw = None
    if isinstance(geometry, Mapping):
        raw = geometry.get("request_rows") or geometry.get("request_lanes")
    if raw:
        out = []
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                rid = str(item.get("request") or "")
                slot = int(item.get("slot", index))
                start = int(item.get("row_start", item.get("start", 0)))
                end = int(item.get("row_end", item.get("end", start + 1)))
            else:
                rid = str(item[0])
                start, end = int(item[1]), int(item[2])
                slot = int(item[3]) if len(item) > 3 else index
            out.append((rid, start, end, slot))
        return tuple(out)
    if len(occurrence.request_ids) == 1:
        return ((occurrence.request_ids[0], 0, rows, 0),)
    return ()


def _binding_padding_rows(geometry: Any) -> frozenset[int]:
    if geometry is None:
        return frozenset()
    raw = None
    if isinstance(geometry, Mapping):
        raw = geometry.get("padding_rows") or geometry.get("padding_lanes")
    else:
        raw = getattr(geometry, "padding_rows", None)
    return frozenset(int(item) for item in (raw or ()) if int(item) >= 0)


def _padding_runs(start: int, end: int,
                  padding: frozenset[int]) -> tuple[tuple[int, int, bool], ...]:
    if not padding or end <= start:
        return ((int(start), int(end), False),)
    runs = []
    cursor = int(start)
    while cursor < end:
        is_padding = cursor in padding
        stop = cursor + 1
        while stop < end and ((stop in padding) == is_padding):
            stop += 1
        runs.append((cursor, stop, is_padding))
        cursor = stop
    return tuple(runs)


def _binding_cache_rows(geometry: Any, request_rows, cached_rows: int
                        ) -> tuple[dict[str, Any], ...] | None:
    """Per-request cache segments supplied by the actual producer.

    ``None`` means the field is absent and must not be replaced by one aggregate
    range.  An empty tuple is an explicit zero.  Each entry keeps the original
    request, slot, row segment and, when given, its own storage/version.
    """
    raw = None
    if isinstance(geometry, Mapping):
        for key in ("request_cached_rows", "request_cache_rows", "cache_rows"):
            if geometry.get(key) is not None:
                raw = geometry[key]
                break
    elif geometry is not None:
        for key in ("request_cached_rows", "request_cache_rows", "cache_rows"):
            value = getattr(geometry, key, None)
            if value is not None:
                raw = value
                break
    if raw is None:
        return None
    known = {"request", "start", "end", "rows", "slot", "storage",
             "state_version", "version", "padding"}
    items: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        if known & set(str(key) for key in raw):
            items = [dict(raw)]
        else:
            for key, value in raw.items():
                if isinstance(value, Mapping):
                    entry = dict(value)
                    entry.setdefault("request", str(key))
                elif isinstance(value, (tuple, list)):
                    pair = tuple(value)
                    entry = {"request": str(key), "start": int(pair[0]),
                             "end": int(pair[1])}
                    if len(pair) > 2:
                        entry["slot"] = int(pair[2])
                else:
                    entry = {"request": str(key), "start": 0,
                             "end": int(value)}
                items.append(entry)
    else:
        for value in raw:
            if isinstance(value, Mapping):
                items.append(dict(value))
            elif isinstance(value, (tuple, list)):
                pair = tuple(value)
                if pair and isinstance(pair[0], str):
                    entry = {"request": pair[0], "start": int(pair[1]),
                             "end": int(pair[2])}
                    if len(pair) > 3:
                        entry["slot"] = int(pair[3])
                else:
                    entry = {"start": int(pair[0]), "end": int(pair[1])}
                    if len(pair) > 2:
                        entry["slot"] = int(pair[2])
                items.append(entry)
            else:
                items.append({"start": 0, "end": int(value)})
    lanes = tuple(request_rows or ())
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(items):
        request = str(entry.get("request") or "")
        lane = next((item for item in lanes if item[0] == request), None)
        if lane is None and not request and index < len(lanes):
            lane = lanes[index]
        start = entry.get("start")
        end = entry.get("end")
        explicit_start = start is not None
        if start is None:
            start = 0
        if end is None:
            count = entry.get("rows")
            if count is None:
                raise ModelingError("binding_cache_rows_require_end_or_rows")
            end = int(start) + int(count)
        start, end = int(start), int(end)
        if start < 0 or end < start:
            raise ModelingError(
                f"binding_cache_rows_invalid:{start}:{end}")
        slot = entry.get("slot")
        if slot is None:
            slot = lane[3] if lane is not None else index
        entries.append({
            "request": request or (lane[0] if lane is not None else ""),
            "start": start,
            "end": end,
            "slot": int(slot),
            "storage": entry.get("storage"),
            "state_version": entry.get(
                "state_version", entry.get("version")),
            "padding": bool(entry.get("padding", False)),
            "explicit_start": explicit_start,
        })
    storage_seen: dict[Any, int] = {}
    for entry in entries:
        if entry["end"] <= entry["start"]:
            continue
        raw_storage = entry["storage"]
        storage_key = (tuple(sorted(raw_storage.items()))
                       if isinstance(raw_storage, Mapping) else raw_storage)
        storage_seen[storage_key] = storage_seen.get(storage_key, 0) + 1
    for entry in entries:
        raw_storage = entry["storage"]
        storage_key = (tuple(sorted(raw_storage.items()))
                       if isinstance(raw_storage, Mapping) else raw_storage)
        entry["offset_unresolved"] = (
            not entry["explicit_start"]
            and entry["end"] > entry["start"]
            and storage_seen.get(storage_key, 0) > 1)
    return tuple(entries)


def _plan_lookup(entry: Mapping[str, Any], key: str, role: str | None,
                 fallback: Any = None) -> Any:
    """Resolve a per-request plan field, optionally per cached K/V value."""
    raw = entry.get(key)
    aliases = tuple(
        item for item in (
            role, "k" if role == "cache_key" else "v" if role == "cache_value"
            else None,
            "key" if role == "cache_key" else "value" if role == "cache_value"
            else None,
        ) if item)
    role_key = (f"{role}_{key}" if role else "")
    short_key = ("k_" if role == "cache_key"
                 else "v_" if role == "cache_value" else "")
    for name in (role_key, f"{short_key}{key}"):
        if name and entry.get(name) is not None:
            return entry[name]
    if isinstance(raw, Mapping):
        for alias in aliases:
            if alias in raw:
                return raw[alias]
        return fallback
    return fallback if raw is None else raw


def _layout_geometry(layout: Mapping[str, Any], dtype: str
                     ) -> tuple[int, float, float, float]:
    """Return ``(rows, offset, row_pitch, row_width)`` in bytes.

    ``row_pitch`` is the address distance between consecutive first-dimension
    indices; ``row_width`` is the actually covered span inside one row.  They
    differ for packed/padded views such as the Q/K rotary output, so neither
    can stand in for the other.  Addresses are element offsets in the
    ``TensorLayout``; the element width comes from the value's actual dtype.
    """
    shape = tuple(int(item) for item in layout.get("shape", ()))
    strides = tuple(int(item) for item in layout.get("strides", ()))
    if not shape or not strides or len(shape) != len(strides):
        return 0, 0.0, 0.0, 0.0
    elem = float(dtype_bytes(dtype or "bf16"))
    offset = float(int(layout.get("offset", 0) or 0)) * elem
    rows = max(0, shape[0])
    if len(shape) == 1:
        return rows, offset, elem, elem
    if any(size <= 0 for size in shape[1:]):
        return rows, offset, 0.0, 0.0
    width_elems = 1 + sum((size - 1) * stride
                          for size, stride in zip(shape[1:], strides[1:]))
    width_elems = max(1, int(width_elems))
    return (rows, offset, float(strides[0]) * elem,
            float(width_elems) * elem)


def _row_byte_ranges(start: int, end: int, *, offset: float,
                     pitch: float, width: float) -> tuple[tuple[float, float], ...]:
    """Physical byte intervals for rows ``[start, end)`` of one view.

    Adjacent rows merge only when the pitch equals the covered row width.
    Otherwise every row is its own interval, which preserves the gaps of a
    packed view instead of reporting one contiguous range over the holes.
    """
    start, end = int(start), int(end)
    if end <= start or width <= 0:
        return ()
    if abs(pitch - width) <= 1e-9:
        return ((offset + start * pitch, offset + end * pitch),)
    return tuple((offset + row * pitch, offset + row * pitch + width)
                 for row in range(start, end))


def _access_action(value: Mapping[str, Any], direction: str, *,
                   reader: str | None, port: str | None) -> str:
    output = str(value.get("output", ""))
    if direction == "read":
        return f"{reader or value.get('producer', '')}:read:{port or output}"
    return f"{value.get('producer', '')}:write:{output}"


def _tensor_accesses(value: Mapping[str, Any], request_rows, *, rank,
                     stage: int, scope: str, direction: str,
                     position: str, purpose: str,
                     padding_rows: frozenset[int] = frozenset(),
                     reader: str | None = None,
                     port: str | None = None,
                     row_plan: tuple[dict[str, Any], ...] | None = None,
                     value_role: str | None = None,
                     ) -> tuple[dict[str, Any], ...]:
    """Project one actual TensorValue into its physical read/write intervals.

    The ranges come from the selected value's own shape, strides, offset and
    dtype (not from a template row estimate).  ``row_span`` stays in rows and
    ``byte_span`` in bytes; the unit of each is named independently.  A plan
    entry may carry its own storage/version/request identity for cached state
    whose per-request segment is known; no averaged aggregate range is formed.
    """
    layout = value.get("layout", {})
    dtype = str(value.get("dtype") or "bf16")
    storage = str(value.get("storage") or "")
    version = value.get("state_version")
    rows, offset, pitch, width = _layout_geometry(layout, dtype)
    output = str(value.get("output", ""))
    reader_port = port or output
    action = _access_action(value, direction, reader=reader, port=reader_port)
    plan: list[dict[str, Any]] = []
    if row_plan is not None:
        plan = [dict(item) for item in row_plan]
    elif request_rows:
        for request, start, end, slot in request_rows:
            plan.extend({
                "request": request, "start": start, "end": end, "slot": slot,
                "padding": is_padding,
            } for start, end, is_padding in _padding_runs(
                start, end, padding_rows))
    else:
        plan = [{"request": "", "start": 0, "end": rows, "slot": 0,
                 "padding": False}]
    result = []
    for entry in plan:
        start = int(entry.get("start", 0))
        end = int(entry.get("end", 0))
        is_padding = bool(entry.get("padding", False))
        entry_storage = str(_plan_lookup(
            entry, "storage", value_role, storage) or "")
        entry_version = _plan_lookup(
            entry, "state_version", value_role, version)
        ranges = (() if is_padding
                  else _row_byte_ranges(start, end, offset=offset,
                                        pitch=pitch, width=width))
        if not ranges:
            anchor = offset + start * pitch if pitch else offset
            ranges = ((anchor, anchor),)
        per_row = len(ranges) == (end - start) and (end - start) > 1
        for index, (begin, stop) in enumerate(ranges):
            row_start = start + index if per_row else start
            row_stop = row_start + 1 if per_row else end
            result.append({
                "request": entry.get("request") or None,
                "request_slot": entry.get("slot"),
                "rank": rank,
                "stage": stage,
                "storage": entry_storage,
                "state_version": entry_version,
                "row_span": (int(row_start), int(row_stop)),
                "byte_span": (float(begin), float(stop)),
                "unit": "rows",
                "byte_unit": "bytes",
                "row_stride": pitch,
                "row_width": width,
                "scope": scope,
                "direction": direction,
                "position": position,
                "purpose": "padding_zero" if is_padding else purpose,
                "padding": is_padding,
                "bytes": float(stop - begin),
                "action": action,
                "port": reader_port,
                "output": output,
                "owner": storage,
                "reader": reader,
            })
    return tuple(result)


def _resize_tensor_layout(layout: Mapping[str, Any], rows: int, *,
                          scale: bool = False) -> dict[str, Any]:
    shape = list(layout.get("shape", ()))
    storage = list(layout.get("storage_shape", ()))
    strides = list(layout.get("strides", ()))
    if shape and shape[0] == 0:
        # A quantisation scale keeps its declared token axis.  Collapsing the
        # three-dimensional (token, group, group) port into two dimensions for
        # a single row changed both its stride and the physical bytes covered
        # by the scale access (four E8M0 bytes in the MX control).
        shape[0] = rows
        if storage and storage[0] == 0:
            storage[0] = rows
    return {
        "shape": tuple(shape), "strides": tuple(strides),
        "offset": int(layout.get("offset", 0)),
        "storage_shape": tuple(storage),
    }


def bind_occurrence(instance: SubgraphInstance, occurrence: Occurrence, *,
                    geometry: Any = None,
                    external_values: Mapping[str, Any] | None = None,
                    fusion: Any = None,
                    state_accesses: Any = None,
                    state_versions: Mapping[str, int] | None = None,
                    scope: str = "target"
                    ) -> dict[str, Any]:
    """Bind one actual invocation to the instance's typed values.

    Static templates carry formulas; this entry substitutes the current row
    geometry, external inputs, selected fusion and occurrence identity.  The
    same bound QKV ``TensorValue`` is used by first execution and cache hits.
    State versions come only from actual state reads/writes, never from the
    call generation.
    """
    if not isinstance(occurrence, Occurrence):
        raise ModelingError("occurrence_binding_requires_occurrence")
    contract = occurrence_contract(instance)
    rows = _binding_rows(geometry)
    cached_rows = _binding_cached_rows(geometry)
    request_rows = _binding_request_rows(geometry, occurrence, rows)
    padding_rows = _binding_padding_rows(geometry)
    cache_plan = _binding_cache_rows(geometry, request_rows, cached_rows)
    unresolved_state_ranges: list[dict[str, Any]] = []
    if len(occurrence.request_ids) > 1 and not request_rows:
        unresolved_state_ranges.append({
            "field": "request_rows",
            "producer": "geometry",
            "requests": occurrence.request_ids,
            "reason": "per_request_rows_required",
        })

    def _request_padding(start: int, end: int) -> bool:
        return (end > start
                and all(row in padding_rows for row in range(start, end)))

    def _cache_row_plan(producer: str) -> tuple[dict[str, Any], ...] | None:
        """Actual per-request cache segment plan, or None when unresolved."""
        if cache_plan is not None:
            if any(entry.get("offset_unresolved")
                   for entry in cache_plan if entry["end"] > entry["start"]):
                unresolved_state_ranges.append({
                    "field": "request_cached_rows",
                    "producer": producer,
                    "requests": tuple(entry["request"] for entry in cache_plan),
                    "reason": "per_request_cache_offsets_required",
                })
                return None
            return cache_plan
        if cached_rows > 0:
            lanes = request_rows or (("", 0, rows, 0),)
            if len(lanes) > 1:
                unresolved_state_ranges.append({
                    "field": "request_cached_rows",
                    "producer": producer,
                    "requests": tuple(lane[0] for lane in lanes),
                    "reason": "per_request_cache_rows_required",
                })
                return None
            lane = lanes[0]
            return ({"request": lane[0], "start": 0, "end": cached_rows,
                     "slot": lane[3], "storage": None,
                     "state_version": None, "padding": False},)
        lanes = request_rows or (("", 0, rows, 0),)
        plan = []
        for request_id, start, end, slot in lanes:
            for run_start, run_end, is_padding in _padding_runs(
                    start, end, padding_rows):
                plan.append({
                    "request": request_id, "start": run_start,
                    "end": run_end, "slot": slot, "storage": None,
                    "state_version": None, "padding": is_padding,
                })
        return tuple(plan)
    key = _binding_key(occurrence)
    actual_owner = (f"layer:{instance.owner_layer}"
                    if instance.owner_layer is not None else "")

    def bind_owner(text: Any) -> Any:
        if actual_owner and isinstance(text, str):
            return text.replace("kv_owner", actual_owner)
        return text
    request = occurrence.request_ids[0] if len(occurrence.request_ids) == 1 else None
    external = dict(external_values or ())
    versions = dict(state_versions or ())
    outputs: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for row in contract["occurrences"]:
        for value in row["outputs"]:
            tensor = value.get("tensor")
            if not tensor:
                continue
            bound = dict(tensor)
            bound["layout"] = _resize_tensor_layout(
                tensor["layout"], rows, scale=value.get("scale") is not None)
            bound["output"] = value["output"]
            bound["binding"] = "event" if occurrence.request_ids else "static"
            bound["binding_name"] = key
            bound["storage"] = bind_owner(bound.get("storage", ""))
            if value["name"] in versions:
                bound["state_version"] = int(versions[value["name"]])
            bound["occurrence"] = occurrence.to_dict()
            bound["occurrence_key"] = key
            position = ("cache" if str(bound.get("storage", "")).startswith("kv:")
                        else "local")
            bound["accesses"] = _tensor_accesses(
                bound, request_rows, rank=occurrence.rank,
                stage=instance.stage, scope=scope, direction="write",
                position=position, purpose="activation_write",
                padding_rows=padding_rows)
            outputs.append(bound)
            by_name[value["name"]] = bound
    qkv = contract.get("qkv_tensor")
    if qkv is not None:
        qkv = dict(qkv)
        qkv["layout"] = _resize_tensor_layout(qkv["layout"], rows)
        qkv["binding"] = "event" if occurrence.request_ids else "static"
        qkv["binding_name"] = key
        qkv["occurrence"] = occurrence.to_dict()
        qkv["occurrence_key"] = key
        qkv["accesses"] = _tensor_accesses(
            qkv, request_rows, rank=occurrence.rank, stage=instance.stage,
            scope=scope, direction="write", position="local",
            purpose="activation_write", padding_rows=padding_rows)
    views = []
    q_slice = k_elems = 0
    if contract.get("qkv_slices"):
        try:
            q_slice, _gate, k_elems, _v, _total, _bytes = (
                int(part) for part in str(contract["qkv_slices"]).split(",")[:6])
        except (TypeError, ValueError):
            q_slice = k_elems = 0
    post_pitch = q_slice + k_elems
    shared_projection = contract.get("qkv_producer") == "q_projection"
    for view in contract["views"]:
        source, producer, output = (bind_owner(view[1]), view[2],
                                    bind_owner(view[3]))
        dtype, phase, alias, version = view[5], view[6], view[7], view[8]
        if fusion is False and view[0] in {"query", "key"} and source == "fused":
            source, producer, output = "operator", "rotary", "rotary_output"
        token = list(view[9])
        view_rows = rows
        if view[0] in {"cache_key", "cache_value"}:
            view_rows = cached_rows or rows
        elif shared_projection and view[0] == "value":
            view_rows = cached_rows or rows
        if token and token[0][0] == 0:
            token[0] = (view_rows,) + tuple(token[0][1:])
            token[3] = ((view_rows,) + tuple(token[3][1:])
                        if token[3] and token[3][0] == 0 else token[3])
        row_layout = view[4]
        storage = bind_owner(view[10]) if len(view) > 10 else ""
        if (fusion is False and view[0] in {"query", "key"}
                and not shared_projection and post_pitch > 0):
            source, producer, output = "operator", "rotary", "rotary_output"
            offset = 0 if view[0] == "query" else q_slice
            heads = view[4][0][0]
            width = view[4][0][1]
            token = ((rows, heads, width),
                     (post_pitch, width, 1), offset,
                     (rows, post_pitch))
            row_layout = ((heads, width), (width, 1), offset, (post_pitch,))
            storage = "post:rotary"
        bound_view = (view[0], source, producer, output, row_layout, dtype,
                      phase, alias, version, tuple(token), storage)
        views.append(bound_view)
    view_tensors = []
    cache_view_plan = _cache_row_plan("attention")
    for view in views:
        storage = bind_owner(view[10]) if len(view) > 10 else ""
        tensor = TensorValue(
            producer=str(view[2]), dtype=str(view[5]),
            layout=TensorLayout(
                shape=tuple(view[9][0]), strides=tuple(view[9][1]),
                offset=int(view[9][2]), storage_shape=tuple(view[9][3])),
            output=str(bind_owner(view[3])), alias_of=str(view[7]),
            storage=storage, state_version=versions.get(storage))
        tensor_dict = tensor.to_dict()
        position = ("cache" if str(tensor_dict.get("storage", "")).startswith("kv:")
                    else "local")
        is_cache = position == "cache" or view[6] == "current"
        if is_cache and cache_view_plan is None:
            tensor_dict["accesses"] = ()
        else:
            tensor_dict["accesses"] = _tensor_accesses(
                tensor_dict, request_rows, rank=occurrence.rank,
                stage=instance.stage, scope=scope, direction="read",
                position="cache" if is_cache else position,
                purpose="state_read" if is_cache else "activation_read",
                padding_rows=padding_rows,
                reader="attention", port=view[0],
                row_plan=(cache_view_plan if is_cache else None),
                value_role=view[0])
        view_tensors.append({"role": view[0], "phase": view[6],
                             "tensor": tensor_dict,
                             "occurrence": occurrence.to_dict()})
    view_tensor_by_role = {item["role"]: dict(item["tensor"])
                           for item in view_tensors}
    signal_inputs = {"query": "query",
                     "key": "cache_key" if shared_projection else "key",
                     "value": "value",
                     "kv_cache_k": "cache_key",
                     "kv_cache_v": "cache_value"}
    bound_rows = []
    for row in contract["occurrences"]:
        bound_inputs = []
        for item in row["inputs"]:
            name = item["name"]
            input_role = (signal_inputs.get(name)
                          if row["name"] == "attention" else None)
            value = external.get(name)
            if (row["name"] == "attention" and name in signal_inputs
                    and signal_inputs[name] in view_tensor_by_role):
                tensor = view_tensor_by_role[signal_inputs[name]]
                value = {
                    "name": name, "producer": tensor["producer"],
                    "dtype": tensor["dtype"], "layout": tensor["layout"],
                    "output": tensor["output"], "tensor": tensor,
                    "storage": tensor["storage"],
                    "state_version": tensor.get("state_version"),
                }
            if value is None and name in by_name:
                value = by_name[name]
            if value is None and item.get("value") is not None:
                value = item["value"]
            if value is None:
                bound_inputs.append({**item, "value": None,
                                     "external": False})
                continue
            if isinstance(value, TensorValue):
                value = value.to_dict()
            value = dict(value)
            value["storage"] = bind_owner(value.get("storage", ""))
            value["output"] = bind_owner(value.get("output", ""))
            if value["storage"] in versions:
                value["state_version"] = int(versions[value["storage"]])
            if (fusion is False and name in {"query", "key"}
                    and str(value.get("producer", "")).startswith("fused:")):
                value["producer"] = "rotary"
                value["output"] = "rotary_output"
            if value.get("layout"):
                value["layout"] = _resize_tensor_layout(value["layout"], rows)
            value["binding_name"] = key
            value["occurrence"] = occurrence.to_dict()
            position = ("cache" if str(value.get("storage", "")).startswith("kv:")
                        else "local")
            is_cache = position == "cache"
            if is_cache and cache_view_plan is None:
                value["accesses"] = ()
            else:
                value["accesses"] = _tensor_accesses(
                    value, request_rows, rank=occurrence.rank,
                    stage=instance.stage, scope=scope, direction="read",
                    position=position,
                    purpose=("state_read" if is_cache
                             else "activation_read"),
                    padding_rows=padding_rows,
                    reader=row["name"], port=name,
                    row_plan=(cache_view_plan if is_cache else None),
                    value_role=input_role)
            bound_inputs.append({**item, "value": value,
                                 "external": name in external})
        bound_outputs = tuple(
            dict(by_name[value["name"]])
            if value["name"] in by_name else value
            for value in row["outputs"])
        bound_rows.append({**row, "outputs": bound_outputs,
                           "inputs": tuple(bound_inputs)})
    memory_accesses: list[dict[str, Any]] = []
    for value in outputs:
        memory_accesses.extend(value.get("accesses", ()))
    for row in bound_rows:
        for item in row["inputs"]:
            value = item.get("value")
            if value:
                memory_accesses.extend(value.get("accesses", ()))

    def _state_entry(item: Mapping[str, Any], *, request_id: str, slot: int,
                     row_start: int, row_stop: int, storage: str,
                     version: Any, padding: bool, reader: str | None = None
                     ) -> dict[str, Any]:
        row_bytes = float(item.get("row_bytes", 0.0) or 0.0)
        direction = str(item.get("direction") or "read")
        producer = str(item.get("producer") or "")
        output = str(item.get("output") or "")
        temporary = bool(item.get("temporary"))
        if padding:
            rows_value, bytes_value = 0, 0.0
            byte_span = (row_start * row_bytes, row_start * row_bytes)
        elif row_bytes <= 0:
            rows_value, bytes_value = 0, 0.0
            byte_span = (0.0, 0.0)
        else:
            rows_value = int(row_stop - row_start)
            bytes_value = float(rows_value) * row_bytes
            byte_span = (row_start * row_bytes, row_stop * row_bytes)
        if padding:
            purpose = "padding_zero"
        elif temporary:
            purpose = "layout_temp"
        elif row_bytes <= 0:
            purpose = "readiness"
        elif producer == "stage_transfer":
            purpose = "transfer_write"
        elif item.get("persistent") and direction == "write":
            purpose = "state_write"
        else:
            purpose = f"state_{direction}"
        return {
            "occurrence": f"{instance.instance_id}:{key}",
            "producer": producer,
            "request": request_id or None,
            "request_slot": int(slot),
            "rank": occurrence.rank,
            "stage": item.get("stage", instance.stage),
            "layer": instance.layer,
            "direction": direction,
            "position": str(item.get("position") or "local"),
            "output": output,
            "storage": storage,
            "owner": str(item.get("owner", "")),
            "representation": str(item.get("representation") or ""),
            "persistent": bool(item.get("persistent")),
            "temporary": temporary,
            "state_version": version,
            "span": ((int(row_start), int(row_stop)),),
            "unit": "rows",
            "row_span": (int(row_start), int(row_stop)),
            "byte_span": byte_span,
            "byte_unit": "bytes",
            "row_stride": row_bytes,
            "row_width": row_bytes,
            "rows": rows_value,
            "bytes": bytes_value,
            "scope": scope,
            "padding": padding,
            "purpose": purpose,
            "action": f"{producer}:{direction}:{output}",
            "port": output,
            "reader": reader if reader is not None else item.get("reader"),
            "source": str(item.get("source") or item.get("owner", "")),
            "grant": None,
            "decision": None,
            "last_use": None,
        }

    def _expand_state_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        representation = str(item.get("representation") or "")
        producer = str(item.get("producer") or "")
        storage = str(item.get("storage") or "")
        if len(occurrence.request_ids) > 1 and not request_rows:
            # A multi-request invocation without its per-request row map has
            # no physical lane decomposition; the binding reports the missing
            # field instead of submitting one aggregated state range.
            return []
        base_version = item.get("state_version")
        if base_version is None and storage in versions:
            base_version = int(versions[storage])
        attention_cache = (producer == "attention"
                           and representation == "kv_cache")
        rowwise = (representation in {"kv_cache", "layout"}
                   and producer in {"kv_store", "kv_read", "layout_convert",
                                    "stage_transfer"})
        expanded_items: list[dict[str, Any]] = []
        if attention_cache:
            lane_plan = cache_view_plan
            if lane_plan is None:
                return expanded_items
            value_role = str(item.get("value_role") or "")
            for entry in lane_plan:
                entry_storage = str(_plan_lookup(
                    entry, "storage", value_role, storage) or "")
                version = _plan_lookup(
                    entry, "state_version", value_role, None)
                if version is None:
                    version = versions.get(entry_storage, base_version)
                padding = bool(entry.get("padding")) or _request_padding(
                    entry["start"], entry["end"])
                expanded_items.append(_state_entry(
                    item, request_id=entry["request"],
                    slot=entry["slot"], row_start=entry["start"],
                    row_stop=entry["end"], storage=entry_storage,
                    version=version, padding=padding,
                    reader=instance.instance_id))
            return expanded_items
        lanes = request_rows or (("", 0, rows, 0),)
        if rowwise:
            for request_id, start, end, slot in lanes:
                for run_start, run_end, is_padding in _padding_runs(
                        start, end, padding_rows):
                    lane_storage = storage.replace("kv_owner", actual_owner)
                    version = (versions.get(lane_storage, base_version)
                               if lane_storage else base_version)
                    expanded_items.append(_state_entry(
                        item, request_id=str(request_id), slot=slot,
                        row_start=run_start, row_stop=run_end,
                        storage=lane_storage, version=version,
                        padding=is_padding))
            return expanded_items
        # Slot/recurrent/conv/index state is one logical row segment per
        # request lane; query-token rows must not multiply it.
        base_rows = max(0, int(item.get("rows", 1) or 1))
        for request_id, start, end, slot in lanes:
            is_padding = _request_padding(int(start), int(end))
            lane_storage = storage.replace("kv_owner", actual_owner)
            version = (versions.get(lane_storage, base_version)
                       if lane_storage else base_version)
            expanded_items.append(_state_entry(
                item, request_id=str(request_id), slot=slot,
                row_start=0, row_stop=(0 if is_padding else base_rows),
                storage=lane_storage, version=version,
                padding=is_padding))
        return expanded_items

    expanded: list[dict[str, Any]] = []
    if state_accesses is not None:
        # An explicit producer-owned access tuple is authoritative, including
        # explicit empty and target/draft pairs; it is never re-split.
        expanded = [dict(item) for item in state_accesses]
    else:
        for item in contract["state_accesses"]:
            sub_items: list[dict[str, Any]] = []
            if (item.get("producer") == "attention"
                    and item.get("representation") == "kv_cache"
                    and not item.get("temporary")):
                # The owner cache holds independent K and V values with their
                # own storage/version; one combined read would hide which
                # value physically carries the 256 B range.
                for role in ("cache_key", "cache_value"):
                    tensor = view_tensor_by_role.get(role)
                    if not tensor:
                        continue
                    sub = dict(item)
                    _value_rows, _value_offset, _value_pitch, value_width = (
                        _layout_geometry(tensor["layout"], tensor["dtype"]))
                    sub["storage"] = tensor.get("storage") or item.get("storage")
                    if value_width:
                        sub["row_bytes"] = value_width
                    sub["state_version"] = tensor.get("state_version",
                                                     item.get("state_version"))
                    sub["output"] = tensor.get("output") or item.get("output")
                    sub["reader"] = instance.instance_id
                    sub["value_role"] = role
                    sub_items.append(sub)
            if not sub_items:
                sub_items = [dict(item)]
            for sub in sub_items:
                expanded.extend(_expand_state_item(sub))
    memory_accesses.extend(expanded)
    return {
        "instance_id": instance.instance_id,
        "occurrence": occurrence.to_dict(),
        "occurrence_key": key,
        "rows": rows,
        "request_rows": request_rows,
        "outputs": tuple(outputs),
        "inputs": tuple(bound_rows),
        "qkv_tensor": qkv,
        "views": tuple(views),
        "view_tensors": tuple(view_tensors),
        "state_accesses": tuple(expanded),
        "memory_accesses": tuple(memory_accesses),
        "unresolved_inputs": contract["unresolved_inputs"],
        "unresolved_state_ranges": tuple(unresolved_state_ranges),
    }


def compile_standalone(instance: SubgraphInstance, supply: Any, *,
                      policy: Any = None,
                      feature_metadata: Any = None) -> Any:
    """Compile one standalone subgraph with the existing piecewise compiler.

    The source/dispatch metadata recorded at construction is forwarded so the
    independent entry selects the same implementation/source conditions as the
    model path.  This returns the same ``CompiledSubgraph`` type used there;
    it does not create a second compiler or price model.
    """
    from mindie_sim.prediction.piecewise.compiler import OperatorPiecewiseCompiler
    metadata = (tuple(feature_metadata) if feature_metadata is not None
                else _instance_feature_metadata(instance))
    return OperatorPiecewiseCompiler(supply, policy).compile(
        instance, feature_metadata=metadata)


def evaluate_standalone(instance: SubgraphInstance, geometry: Any, supply: Any,
                        *, state: Any = None, policy: Any = None,
                        feature_metadata: Any = None) -> Any:
    """Evaluate one standalone subgraph's isolated operator work.

    Returns the existing ``SubgraphEvaluation`` (per-operator work/state
    effects and resulting ``StateView``).  This is the isolated operator sum,
    not a graph span; use ``evaluate_standalone_graph``/``standalone_span``
    for the declared-stream program.  The source/dispatch metadata recorded at
    construction is forwarded to the same compiler the model path uses.
    """
    compiled = compile_standalone(instance, supply, policy=policy,
                                  feature_metadata=feature_metadata)
    if state is None:
        # No supplied state means isolated operator work without readiness
        # admission, preserving the historical isolated-entry semantics.
        return compiled.evaluate(geometry, None)
    # With a real state, consume the same declared-effect program as the graph
    # entry (including transfer publication), rather than applying only the
    # family effects and returning a different state.
    from mindie_sim.prediction.piecewise.compiler import SubgraphEvaluation
    from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
    result = evaluate_serial_split(compiled, geometry, state, fusion=())
    # This entry is the isolated-operator view: its total is always the sum of
    # the executed costs, even when the same graph also has a makespan/span
    # field.  Supplying a state may change work/admission and the resulting
    # state effects, but not the accounting identity of this result.
    isolated = sum((cost.duration_ms for cost in result.costs), 0)
    return SubgraphEvaluation(instance.instance_id, instance.kind,
                              result.costs, isolated, result.state)


def evaluate_standalone_graph(instance: SubgraphInstance, geometry: Any,
                              supply: Any, *, state: Any = None,
                              policy: Any = None, fusion: bool = True,
                              feature_metadata: Any = None) -> Any:
    """Evaluate one standalone subgraph through the existing graph program.

    Uses ``evaluate_serial_split`` — the same function ``ModelPipeline._run``
    calls — so declared main/aux streams, per-kernel startup/dependency,
    native stages/programs and shared-resource contention stay in the existing
    solver.  Auxiliary operators outside the program's declared fork set and
    undeclared streams are explicit unsupported-span errors, never silently
    flattened into one serial chain.
    """
    from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
    contract = occurrence_contract(instance)
    cross = contract["cross_stream_edges"]
    if cross:
        detail = ",".join(
            f"{edge['producer']}->{edge['consumer']}:{edge['value']}"
            for edge in cross)
        # The current common solver selects a fixed subset by name and does
        # not yet consume named cross-stream producer edges.  Returning a
        # total here would be a false span; expose the exact dependency that
        # PW-CORE must assemble instead.
        raise ModelingError(
            f"standalone_span_dependency_unexpressed:{detail}")
    non_main = tuple(
        (stream, names) for stream, names in contract["streams"]
        if stream != "main")
    if non_main:
        detail = ",".join(f"{stream}:{','.join(names)}"
                          for stream, names in non_main)
        raise ModelingError(
            f"standalone_span_stream_order_unexpressed:{detail}")
    compiled = compile_standalone(instance, supply, policy=policy,
                                  feature_metadata=feature_metadata)
    names = (tuple(contract.name for contract in instance.template.fusions)
             if fusion else ())
    return evaluate_serial_split(compiled, geometry, state, fusion=names)


def standalone_span(instance: SubgraphInstance, geometry: Any, supply: Any,
                    *, state: Any = None, policy: Any = None,
                    fusion: bool = True,
                    feature_metadata: Any = None) -> dict[str, Any]:
    """Return the standalone graph span from the existing stream solver.

    ``span_scope`` states exactly what was solved: ``single_chain`` for a
    one-stream subgraph, ``declared_streams`` when the existing fork/contention
    path was used.  ``span_ms`` is the existing program's completion under the
    caller's supply, not a re-summed serial total; ``isolated_ms`` is reported
    separately as the sum of operator durations.
    """
    result = evaluate_standalone_graph(
        instance, geometry, supply, state=state, policy=policy,
        fusion=fusion, feature_metadata=feature_metadata)
    contention = result.contention
    payload: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "kind": instance.kind,
        "variant": instance.variant,
        "span_scope": ("declared_streams" if contention is not None
                       else "single_chain"),
        "isolated_ms": sum((cost.duration_ms for cost in result.costs), 0),
        "serial_ms": result.total_ms,
        "readiness_gap_ms": result.gap_ms,
        "span_ms": result.total_ms,
        "fusion": result.fused,
        "contention": None,
        "span_limits": (),
    }
    # State explicitly which parts of an operator's internal execution are
    # carried by its published duration rather than replayed on the Engine.
    # The model path computes a subgraph contribution the same way; treating
    # it as a full fragment execution would overstate the span.
    limits: list[str] = []
    native = sorted(cost.name for cost in result.costs
                    if cost.native_program is not None
                    or cost.native_schedule is not None)
    if native:
        limits.append("native_program_elapsed_not_resubmitted:"
                      + ",".join(native))
    multi_stage = sorted(cost.name for cost in result.costs
                         if len(cost.stage_work) > 1)
    if multi_stage:
        limits.append("multi_stage_elapsed_not_replayed:"
                      + ",".join(multi_stage))
    payload["span_limits"] = tuple(limits)
    if contention is not None:
        payload["contention"] = {
            "makespan_ms": contention.makespan_ms,
            "isolated_max_ms": contention.isolated_max_ms,
            "isolated_sum_ms": contention.isolated_sum_ms,
            "additivity_gap_ms": contention.additivity_gap_ms,
            "resource_work": dict(contention.resource_work),
            "shared_pipes": contention.shared_pipes,
            "disjoint_pipes": contention.disjoint_pipes,
            "branches": dict(contention.branches),
            "completions": dict(contention.completions),
        }
    return payload
