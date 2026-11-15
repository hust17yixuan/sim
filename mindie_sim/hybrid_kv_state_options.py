"""Declared hybrid KV/state options and deterministic input projections.

[Role]
  Define the single immutable option record of ``evaluate_hybrid_kv_state`` and
  the pure model/machine projections its evaluation needs, including the
  model-derived sparse/MLA/MLAPO facts, the declared PP layer partition and
  the PD producer/consumer role.

[Invariants]
  * Every option is validated before use; booleans, strings, NaN, negative
    dimensions and non-integral counts are typed rejections.  A passed
    ``options=False`` is an invalid option record, never a default.
  * Defaults are resolved from the declared Case only; no host probe or
    environment value is consulted.
  * Model facts (sparse SFA, kpool indexer, MLA vs GQA, MLAPO eligibility) are
    derived from catalog structure, and explicit overrides are validated
    against that ground truth.
  * PP ownership always comes from ``runtime.pp_layer_partition``; no
    floor-division partition is invented.
  * ``_json_safe`` rejects non-finite floats, it does not coerce them.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
import hashlib
import json
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.linear_state_physics import LinearLayer
from mindie_sim.nr_s7_validate import (
    positive_rate, strict_bool, strict_int, strict_number, strict_text, typed_sequence,
    _validate_nested, stage_of_layer,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec as CatalogModel, LayerPlan
from mindie_sim.schema.case import Case, Software
from mindie_sim.catalog.types import DraftSpec


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="hybrid_kv")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="hybrid_kv")


def _num(name: str, value: Any, *, minimum: float = 0.0,
         allow_none: bool = False) -> float | None:
    return strict_number(name, value, minimum=minimum, allow_none=allow_none,
                         reason="hybrid_kv")


@typed_inputs
@dataclass(frozen=True)
class HybridKvStateOptions:
    """Declared switches for one deterministic hybrid evaluation."""

    batch: int | None = None
    sequence_tokens: int | None = None
    enable_kv_nz: bool = False
    pd_disaggregated: bool = False
    is_kv_consumer: bool = False
    is_sparse_sfa: bool | None = None
    mlapo_eligible: bool | None = None
    enable_mlapo: bool = True
    kv_connector: str | None = None
    kv_connector_module_path: str | None = None
    kv_role: str | None = None
    use_replayssm: bool = False
    num_spec: int = 0
    enable_prefix_caching: bool = False
    mamba_cache_mode: str = "none"
    mamba_block_size: int = 0
    enable_stochastic_rounding: bool = False
    prefix_hit_tokens: int = 0
    prefix_available_hashes: int | None = None
    swa_bounded_replay: bool = False
    extra_retained_tokens: int = 0
    dspark_active: bool | None = None
    dspark_block_size: int = 0
    dspark_aux_layer_ids: tuple[int, ...] = ()
    dspark_target_page_size_bytes: int | None = None
    dspark_draft_page_size_bytes: int | None = None
    state_page_stride_bytes: int | None = None
    page_size_padded_override: int | None = None
    indexes_kv_by_block_stride: bool | None = None
    conv_dim_stride: int = 1
    hbm_bytes_per_s: float | None = None
    compute_flops_per_s: float | None = None
    kernel_launch_ms: float | None = None
    accept_rate: float = 0.0
    accepted_tokens: int | None = None
    descriptor_setup_us: float = 0.0
    pd_page_alignment: int = 0
    align_layerwise_copy: bool = True
    align_copy_overlap: float = 0.0
    program_dispatch_us: float = 0.0
    graph_analytical_bound: bool = False
    strict: bool = False

    def __post_init__(self) -> None:
        for name in ("enable_kv_nz", "pd_disaggregated", "is_kv_consumer",
                     "use_replayssm", "enable_prefix_caching",
                     "enable_stochastic_rounding", "swa_bounded_replay",
                     "align_layerwise_copy", "graph_analytical_bound",
                     "enable_mlapo", "strict"):
            _bool(name, getattr(self, name))
        for name in ("is_sparse_sfa", "mlapo_eligible", "dspark_active"):
            value = getattr(self, name)
            if value is not None:
                _bool(name, value)
        for name in ("batch", "sequence_tokens", "dspark_target_page_size_bytes",
                     "dspark_draft_page_size_bytes", "state_page_stride_bytes",
                     "page_size_padded_override", "accepted_tokens",
                     "prefix_available_hashes"):
            value = getattr(self, name)
            if value is not None:
                _int(name, value, minimum=1)
        for name in ("num_spec", "extra_retained_tokens", "dspark_block_size",
                     "pd_page_alignment"):
            _int(name, getattr(self, name), minimum=0)
        _int("prefix_hit_tokens", self.prefix_hit_tokens, minimum=0)
        _int("mamba_block_size", self.mamba_block_size, minimum=0)
        _int("conv_dim_stride", self.conv_dim_stride, minimum=1)
        strict_text("mamba_cache_mode", self.mamba_cache_mode, reason="hybrid_kv")
        if self.mamba_cache_mode not in {"none", "align"}:
            raise ModelingError(
                f"hybrid_kv_mamba_cache_mode_invalid:{self.mamba_cache_mode}")
        for name in ("kv_connector", "kv_connector_module_path", "kv_role"):
            value = getattr(self, name)
            if value is not None:
                strict_text(name, value, reason="hybrid_kv")
        if self.kv_role is not None and self.kv_role not in {
                "kv_consumer", "kv_producer"}:
            raise ModelingError(f"hybrid_kv_kv_role_invalid:{self.kv_role}")
        for name in ("hbm_bytes_per_s", "compute_flops_per_s",
                     "kernel_launch_ms"):
            value = getattr(self, name)
            if value is not None:
                validator = strict_number if name == "kernel_launch_ms" else positive_rate
                validator(name, value, reason="hybrid_kv")
        for name in ("descriptor_setup_us", "program_dispatch_us"):
            _num(name, getattr(self, name))
        _num("accept_rate", self.accept_rate)
        _num("align_copy_overlap", self.align_copy_overlap)
        if self.align_copy_overlap > 1.0:
            raise ModelingError(
                f"hybrid_kv_align_copy_overlap_invalid:{self.align_copy_overlap}")
        if self.accept_rate > 1.0:
            raise ModelingError(
                f"hybrid_kv_accept_rate_invalid:{self.accept_rate}")
        if self.indexes_kv_by_block_stride is not None:
            _bool("indexes_kv_by_block_stride",
                  self.indexes_kv_by_block_stride)
        if not isinstance(self.dspark_aux_layer_ids, tuple) or not all(
                type(value) is int and value >= 0
                for value in self.dspark_aux_layer_ids):
            raise ModelingError("hybrid_kv_dspark_aux_layer_ids_invalid")
        if list(self.dspark_aux_layer_ids) != sorted(
                set(self.dspark_aux_layer_ids)):
            raise ModelingError("hybrid_kv_dspark_aux_layer_ids_not_canonical")
        if self.pd_disaggregated and self.kv_connector is None:
            raise ModelingError("hybrid_kv_pd_requires_connector")
        if self.is_kv_consumer and self.kv_role == "kv_producer":
            raise ModelingError("hybrid_kv_pd_role_mismatch")
        if not self.pd_disaggregated and self.kv_role is not None:
            raise ModelingError("hybrid_kv_pd_role_without_pd")

    def with_dspark_aux(self, values: Sequence[int]) -> "HybridKvStateOptions":
        values = typed_sequence("dspark_aux_layer_ids", values, int)
        for value in values:
            _int("dspark_aux_layer_ids", value)
        if list(values) != sorted(set(values)):
            raise ModelingError("hybrid_kv_dspark_aux_layer_ids_not_canonical")
        if self.dspark_aux_layer_ids:
            return self
        return HybridKvStateOptions(**{
            **{item.name: getattr(self, item.name) for item in fields(self)},
            "dspark_aux_layer_ids": tuple(values)})

    @property
    def producer(self) -> bool:
        return self.pd_disaggregated and self.kv_role == "kv_producer"

    @property
    def consumer(self) -> bool:
        return self.pd_disaggregated and self.is_kv_consumer


@typed_inputs
def pd_role(options: HybridKvStateOptions) -> tuple[bool, bool]:
    """The one PD consumer/producer normalization every consumer shares.

    The source accepts either the boolean consumer flag or the equivalent
    ``kv_role="kv_consumer"`` string; the two spellings must produce identical
    eligibility, NZ validation and MLAPO baselines for the same deployment.
    """
    if not isinstance(options, HybridKvStateOptions):
        raise ModelingError("hybrid_kv_options_invalid")
    consumer = bool(options.is_kv_consumer or options.kv_role == "kv_consumer")
    producer = bool(options.kv_role == "kv_producer")
    if consumer and producer:
        raise ModelingError("hybrid_kv_pd_role_ambiguous")
    return consumer, producer


def _facts(model: Any) -> dict[str, Any]:
    value = getattr(model, "catalog_facts", ())
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = value
    else:
        raise ModelingError("hybrid_kv_catalog_facts_invalid")
    result = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ModelingError("hybrid_kv_catalog_fact_invalid")
        key, value = item
        strict_text("catalog_fact_key", key)
        result[key] = value
    return result


@typed_inputs
def _model_type(model: CatalogModel) -> str:
    facts = _facts(model)
    return strict_text("model_type", facts["model_type"], reason="hybrid_kv") if "model_type" in facts else ""


@typed_inputs
def _architecture(model: CatalogModel) -> str:
    values = typed_sequence("architectures", _facts(model).get("architectures", ()), str)
    return strict_text("architecture", values[0]) if values else ""


@typed_inputs
def _draft_spec(model: CatalogModel) -> Any:
    auxiliary = getattr(model, "auxiliary", None)
    if auxiliary is None:
        return None
    draft = getattr(auxiliary, "dspark", None)
    if draft is not None:
        return draft
    candidate = getattr(auxiliary, "mtp", None)
    if candidate is not None and getattr(candidate, "kind", "") in {
            "dspark", "mtp_dspark"}:
        return candidate
    return None


@typed_inputs
def _attention_row(model: CatalogModel) -> tuple[int, str]:
    for plan in typed_sequence("layer_plans", model.layer_plans, LayerPlan):
        if plan.attention in {"mla", "gqa"}:
            return _int("cache_row_elements", plan.cache_row_elements,
                        minimum=1), str(plan.attention)
    if getattr(model, "mla", None) is not None:
        return _int("mla_row", model.mla.kv_lora_rank
                    + model.mla.qk_rope_head_dim, minimum=1), "mla"
    return _int("gqa_row", 2 * model.num_kv_heads * model.head_dim,
                minimum=1), "gqa"


@typed_inputs
def _attention_layer_count(model: CatalogModel) -> int:
    count = sum(1 for plan in typed_sequence("layer_plans", model.layer_plans, LayerPlan)
                if plan.attention in {"mla", "gqa"})
    return max(1, count)


@typed_inputs
def model_is_sparse_sfa(model: Any) -> bool:
    """Derive ``model_uses_sfa_sparse`` from explicit presence facts.

    Source semantics are attribute presence, not a positive value:
    ``hasattr(index_topk)`` is true even for ``0``, while a kpool indexer or a
    ``compress_ratios`` attribute excludes the model.  Catalog dataclasses
    default every field, so their presence evidence comes from the preserved
    catalog facts; a source-shaped configuration double is inspected directly.
    """
    mla = getattr(model, "mla", None)
    if isinstance(model, CatalogModel) and not (
            hasattr(model, "hf_text_config") or hasattr(model, "hf_config")):
        _validate_nested(model, "model")
        facts = _facts(model)
        # Numeric defaults in normalized plans are not attribute-presence
        # evidence. Preserve an explicit index_kpool even when its value is 1.
        if "index_kpool" in facts or "compress_ratios" in facts:
            return False
        if mla is None:
            return False
        index_topk_present = ("index_topk" in facts
                              or _int("index_topk",
                                      getattr(mla, "index_topk", 0)) > 0)
        return bool(index_topk_present)
    # Source-shaped doubles: use attribute presence exactly as the reference.
    text = getattr(model, "hf_text_config", None)
    if text is None:
        return False
    sources = (text, getattr(model, "hf_config", None))
    if any(item is not None and hasattr(item, "compress_ratios")
           for item in sources):
        return False
    if any(item is not None and hasattr(item, "index_kpool")
           for item in sources):
        return False
    return hasattr(text, "index_topk")


@typed_inputs
def projection_quant_class(model: CatalogModel) -> str:
    """Projection-weight recipe class used by the MLAPO source gate.

    Static W8A8, Ascend W8A8-MXFP8 and the distinct dynamic INT8 linear method
    are separate source classes; collapsing them loses the dynamic INT8
    ``not support_fp8_attention`` condition.
    """
    facts = _facts(model)
    quantization = facts.get("quantization", {})
    if not isinstance(quantization, Mapping):
        raise ModelingError("hybrid_kv_quantization_invalid")
    name, weight = (quantization.get(k, "") for k in ("name", "weight_format"))
    if type(name) is not str or type(weight) is not str:
        raise ModelingError("hybrid_kv_quantization_text_invalid")
    name, weight = name.strip().lower(), weight.strip().lower()
    if not name or name == "none" or "bf16" in name or weight.startswith("bf16"):
        return "native"
    if "mxfp8" in name or "mxfp8" in weight:
        return "w8a8_mxfp8"
    if "int8" in name or weight == "int8":
        return ("w8a8_dynamic_int8"
                if "dynamic" in name or "dynamic" in weight
                else "w8a8_int8")
    if weight.startswith("fp8") or "fp8" in name:
        return "fp8"
    return "unknown"


@typed_inputs
def hardware_supports_fp8_attention(hardware_arch: str) -> bool:
    """Whether the declared device advertises ``FP8_ATTENTION``.

    Only the A5/DAV_3510 family does; A2/A3 use CANN's INT8 mode 2 for the
    dynamic W8A8 linear method instead.
    """
    arch = strict_text("hardware_arch", hardware_arch).strip().upper()
    return arch in {"A5", "DAV_3510"}


@typed_inputs
def model_mlapo_eligible(model: CatalogModel, *, hardware_arch: str = "A3",
                          enable_mlapo: bool = True,
                          decode_instance: bool = False,
                          is_draft: bool = False) -> bool:
    """The complete source MLAPO predicate, not a "has MLA" heuristic.

    ``attention/mla_v1.py`` requires a non-draft MLA model, the configured
    enable flag, a decode instance unless the hardware grants
    ``UNRESTRICTED_MLAPO`` (A5 only), no fused QKV-A output gate (Kimi K3), and
    a supported projection-weight class: static W8A8/MXFP8 everywhere,
    dynamic INT8 only where ``not support_fp8_attention`` (A2/A3), or native
    floating-point weights on hardware exposing ``MLAPO_NATIVE_WEIGHTS`` (A5).
    """
    for name, value in (("enable_mlapo", enable_mlapo),
                        ("decode_instance", decode_instance), ("is_draft", is_draft)):
        _bool(name, value)
    arch = strict_text("hardware_arch", hardware_arch).strip().upper()
    if is_draft:
        return False
    mla = getattr(model, "mla", None)
    if mla is None:
        return False
    if model_is_sparse_sfa(model):
        return False
    if not enable_mlapo:
        return False
    unrestricted = arch in {"A5", "DAV_3510"}
    if not unrestricted and not decode_instance:
        return False
    plans = typed_sequence("layer_plans", model.layer_plans, LayerPlan)
    gates = [_bool("mla_qkv_a_gate_fused", plan.mla_qkv_a_gate_fused)
             for plan in plans]
    if any(gates):
        return False
    quant = projection_quant_class(model)
    if quant == "w8a8_mxfp8":
        return True
    if quant == "w8a8_int8":
        return arch in {"A2", "A3", "A5", "DAV_3510"}
    if quant == "w8a8_dynamic_int8":
        return not hardware_supports_fp8_attention(arch)
    if quant == "native":
        return unrestricted
    return False


@typed_inputs
def _draft_row_elements(draft: DraftSpec) -> int:
    """Full-model draft row elements including both K and V planes."""
    if getattr(draft, "attention_kind", "") == "mla":
        value = (_int("kv_lora_rank", draft.kv_lora_rank)
                 + _int("qk_rope_head_dim", draft.qk_rope_head_dim))
        if value > 0:
            return value
    value = 2 * _int("num_kv_heads", draft.num_kv_heads) * _int(
        "head_dim", draft.head_dim)
    return value if value > 0 else _int("kv_width", draft.kv_width, minimum=1)


@typed_inputs
def _draft_local_kv_heads(draft: DraftSpec, tp: int) -> int:
    heads = _int("num_kv_heads", draft.num_kv_heads, minimum=1)
    _int("tp", tp, minimum=1)
    if getattr(draft, "attention_kind", "") == "mla":
        # Latent MLA KV is a single replicated head, never TP-partitioned.
        return heads
    if heads % tp and tp % heads:
        raise ModelingError(
            f"hybrid_kv_draft_kv_heads_not_divisible_by_tp:{heads}%{tp}")
    return max(1, heads // tp)


def _machine_rate(name: str, value: Any) -> float:
    """Reject strings, booleans, NaN/Inf and non-positive rates before ``float``."""
    return positive_rate(name, value, reason="hybrid_kv_machine")


def _machine_non_negative(name: str, value: Any) -> float:
    number = strict_number(name, value, minimum=0.0, reason="hybrid_kv_machine")
    assert number is not None
    return number


@typed_inputs
def _machine_limits(case: Case) -> tuple[float, float, float, float]:
    """Return ``(hbm_bytes_per_s, fp32_flops_per_s, launch_ms, collective_bps)``."""
    dies = tuple(case.machine.dies)
    hbm: list[float] = []
    for device in dies:
        rates = dict(device.rates)
        if "hbm" in rates:
            hbm.append(_machine_rate("hbm", rates["hbm"]))
    hbm_bytes = min(hbm) if hbm else 1.0e12
    peaks: list[float] = []
    for device in dies:
        peaks_dict = dict(device.compute_peaks)
        if "fp32" in peaks_dict:
            # A declared FP32 peak of zero is not "no compute term"; it is an
            # invalid machine rate and must reject before any division.
            peaks.append(_machine_rate("fp32_flops", peaks_dict["fp32"]))
    flops = min(peaks) if peaks else 0.0
    launch_ms = max(_machine_non_negative("launch_us", device.launch_us)
                    for device in dies) / 1000.0
    links = [_machine_rate("d2d_capacity_bytes_per_ms",
                           link.capacity_bytes_per_ms) * 1000.0
             for link in getattr(case.machine, "links", ())
             if "d2d" in str(getattr(link, "channel", ""))]
    collective = min(links) if links else hbm_bytes
    return (hbm_bytes, flops, launch_ms, collective)


@typed_inputs
def _stage_partition(runtime: Software, model: CatalogModel) -> tuple[int, ...]:
    partition = typed_sequence("pp_layer_partition", runtime.pp_layer_partition, int)
    pp = _int("pp", runtime.parallel.pp, minimum=1)
    if not partition:
        layers = _int("layers", model.layers, minimum=1)
        partition = tuple(layers // pp + int(stage < layers % pp)
                          for stage in range(pp))
    total = 0
    for count in partition:
        total += _int("partition_count", count, minimum=0)
    if len(partition) != pp:
        raise ModelingError("hybrid_kv_pp_partition_stage_count_invalid")
    if total != _int("layers", model.layers, minimum=1):
        raise ModelingError(
            f"hybrid_kv_pp_partition_total_mismatch:{total}!={model.layers}")
    return partition


@typed_inputs
def _layer_stage(partition: tuple[int, ...], index: int) -> int:
    return stage_of_layer(partition, index)


@typed_inputs
def _linear_layers_for_stage(model: CatalogModel, layers: Sequence[LinearLayer],
                             stage: int, partition: tuple[int, ...]) -> list[LinearLayer]:
    _int("stage", stage, minimum=0)
    for count in partition:
        _int("partition_count", count)
    if stage >= len(partition):
        raise ModelingError(f"hybrid_kv_stage_out_of_range:{stage}")
    return [layer for layer in layers
            if _layer_stage(partition, layer.index) == stage]


@typed_inputs
def _attention_layers_for_stage(model: CatalogModel, stage: int,
                                partition: tuple[int, ...]) -> list[Any]:
    _int("stage", stage, minimum=0)
    for count in partition:
        _int("partition_count", count)
    if stage >= len(partition):
        raise ModelingError(f"hybrid_kv_stage_out_of_range:{stage}")
    return [plan for plan in typed_sequence("layer_plans", model.layer_plans, LayerPlan)
            if plan.attention in {"mla", "gqa"}
            and _layer_stage(partition, plan.index) == stage]


@typed_inputs
def _digest(*values: Any) -> str:
    try:
        payload = json.dumps(values, sort_keys=True, default=str)
    except (TypeError, ValueError) as error:
        raise ModelingError("hybrid_kv_digest_value_invalid") from error
    return hashlib.sha256(payload.encode()).hexdigest()


@typed_inputs
def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ModelingError("hybrid_kv_json_value_invalid") from error


__all__ = ["HybridKvStateOptions", "_architecture", "_attention_layer_count",
           "_attention_layers_for_stage", "_attention_row", "_digest",
           "_draft_local_kv_heads", "_draft_row_elements", "_draft_spec",
           "_json_safe", "_layer_stage", "_linear_layers_for_stage",
           "_machine_limits", "_model_type", "_stage_partition",
           "hardware_supports_fp8_attention", "model_is_sparse_sfa",
           "model_mlapo_eligible", "pd_role", "projection_quant_class"]
