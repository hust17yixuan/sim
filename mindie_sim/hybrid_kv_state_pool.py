"""Baseline cache-component pricing and per-stage pool reconciliation.

[Role]
  Read the actual ``runtime.memory.cache_page_groups`` declarations and the
  catalog layer geometry to compute what the baseline ledger already charges
  for the attention/state components the hybrid model replaces, per PP stage.
  This is the "already priced" side of a net-once reconciliation.

[Invariants]
  * A declared ``CachePageGroup`` replaces the per-layer component pricing for
    the components it owns; its page price is charged once, never once per
    component.
  * Undeclared layers use the same ``layer_cache_components`` call the
    baseline ``Memory`` uses, so TP/PCP/DCP local row ownership is identical.
  * Draft KV is priced through ``stage_cache_components`` with the baseline's
    own ``draft_kind`` and is attributed to the final PP stage only.
  * All arithmetic is finite and non-negative; malformed declarations reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.catalog.cache import (layer_cache_components,
                                     stage_cache_components)
from mindie_sim.catalog.cache_pages import CachePageGroup
from mindie_sim.hybrid_kv_state_options import _layer_stage, _stage_partition
from mindie_sim.nr_s7_validate import strict_bool, strict_int, strict_number
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.case import Case

#: Components the hybrid page-class model replaces.
REPLACED_COMPONENTS = frozenset({
    "mla_kv", "gqa_kv", "window", "compressed_kv", "recurrent_state",
    "conv_state"})
DRAFT_COMPONENT = "draft_kv"
SMALL_CLASS_COMPONENTS = frozenset({"index_k", "index_pool_tail"})


@typed_inputs
@dataclass(frozen=True)
class BaselinePricing:
    """Per-stage bytes already charged for the replaced components."""

    stages: int
    replaced_per_stage: tuple[float, ...]
    draft_kv_per_stage: tuple[float, ...]
    declared_group_bytes: float
    declared_layer_bytes: float
    undeclared_bytes: float

    def __post_init__(self) -> None:
        stages = strict_int("stages", self.stages, minimum=1,
                            reason="baseline_pricing")
        # Validate the raw values before they reach subtraction: booleans,
        # fractions, strings, NaN and infinities are not byte counts.
        for name, values in (("replaced_per_stage", self.replaced_per_stage),
                             ("draft_kv_per_stage", self.draft_kv_per_stage)):
            if not isinstance(values, tuple):
                raise ModelingError(f"baseline_pricing_{name}_invalid")
            if len(values) != stages:
                raise ModelingError(f"baseline_pricing_{name}_stage_shape_invalid")
            for value in values:
                strict_number(name, value, minimum=0.0,
                              reason="baseline_pricing")
        for name in ("declared_group_bytes", "declared_layer_bytes",
                     "undeclared_bytes"):
            strict_number(name, getattr(self, name), minimum=0.0,
                          reason="baseline_pricing")

    def to_dict(self) -> dict[str, Any]:
        return {"replaced_per_stage": list(self.replaced_per_stage),
                "draft_kv_per_stage": list(self.draft_kv_per_stage),
                "declared_group_bytes": self.declared_group_bytes,
                "declared_layer_bytes": self.declared_layer_bytes,
                "undeclared_bytes": self.undeclared_bytes}


def _group_components(group: CachePageGroup, layer: int,
                      available: set[str]) -> set[str]:
    if group.members:
        return {component for owner, component in group.members
                if owner == layer}
    if group.components:
        return set(group.components)
    return available


def _declared_prices(parsed: Any, tokens: int,
                     stage_of_layer: dict[int, int], selected: frozenset[str],
                     active_sequences: int) -> tuple[
                         dict[int, float], dict[int, float], set[tuple[int, str]]]:
    """Return (layer prices, group prices, covered (layer, component)) pairs.

    Declared page groups are per-sequence prices, so they are multiplied by
    ``active_sequences`` on the same basis as undeclared components.  Only the
    components that actually belong to the replaced target/state set are
    covered or priced; a group outside that set contributes nothing.
    """
    runtime = parsed.runtime
    raw_groups = tuple(dict(runtime.memory).get("cache_page_groups", ()) or ())
    available = {plan.index: set(_component_price(parsed, plan, tokens))
                 for plan in parsed.model.layer_plans} if raw_groups else {}
    layer_price: dict[int, float] = {}
    group_price: dict[int, float] = {}
    covered: set[tuple[int, str]] = set()
    for raw in raw_groups:
        group = CachePageGroup.from_mapping(raw)
        pages = group.pages(tokens, int(runtime.max_model_len))
        components_by_layer = {
            layer: _group_components(group, layer, available.get(layer, set())) & selected
            for layer in group.layers}
        if not any(components_by_layer.values()):
            continue
        if group.bytes_per_group_page:
            stage = stage_of_layer[group.layers[0]]
            group_price[stage] = group_price.get(stage, 0.0) + float(
                pages * group.bytes_per_group_page) * active_sequences
            for layer, components in components_by_layer.items():
                for component in components:
                    covered.add((layer, component))
        else:
            for layer, components in components_by_layer.items():
                if not components:
                    continue
                covered.update((layer, component) for component in components)
                stage = stage_of_layer[layer]
                layer_price[stage] = layer_price.get(stage, 0.0) + float(
                    pages * group.bytes_per_layer_page) * active_sequences
    return layer_price, group_price, covered


def _component_price(parsed: Any, plan: Any, tokens: int) -> dict[str, float]:
    return dict(layer_cache_components(
        parsed.model, plan, parsed.runtime.parallel, tokens,
        capacity_padded=True))


@typed_inputs
def baseline_pricing(parsed: Case, *, tokens: int, active_sequences: int,
                     components: Sequence[str] = tuple(
                         sorted(REPLACED_COMPONENTS)),
                     include_draft: bool = True) -> BaselinePricing:
    """Estimate the baseline's already-priced replaced components by stage."""
    if isinstance(tokens, bool) or type(tokens) is not int or tokens < 1:
        raise ModelingError(f"baseline_pricing_tokens_invalid:{tokens!r}")
    if (isinstance(active_sequences, bool)
            or type(active_sequences) is not int or active_sequences < 1):
        raise ModelingError(
            f"baseline_pricing_sequences_invalid:{active_sequences!r}")
    strict_bool("include_draft", include_draft, reason="baseline_pricing")
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) \
            or not all(isinstance(value, str) for value in components):
        raise ModelingError("baseline_pricing_components_invalid")
    selected = frozenset(components)
    target_components = REPLACED_COMPONENTS | SMALL_CLASS_COMPONENTS
    if not selected <= target_components | {DRAFT_COMPONENT}:
        raise ModelingError("baseline_pricing_component_unknown")
    model, runtime = parsed.model, parsed.runtime
    partition = _stage_partition(runtime, model)
    stage_of_layer = {plan.index: _layer_stage(partition, plan.index)
                      for plan in model.layer_plans}
    stages = len(partition)
    layer_price, group_price, covered = _declared_prices(
        parsed, tokens, stage_of_layer, selected, active_sequences)
    replaced = [0.0] * stages
    undeclared = 0.0
    for plan in model.layer_plans:
        price = _component_price(parsed, plan, tokens)
        for component in selected & target_components:
            if (plan.index, component) in covered:
                continue
            value = float(price.get(component, 0.0)) * active_sequences
            if value:
                replaced[stage_of_layer[plan.index]] += value
                undeclared += value
    for stage, value in group_price.items():
        replaced[stage] += value
    for stage, value in layer_price.items():
        replaced[stage] += value
    draft = [0.0] * stages
    if include_draft:
        draft_kind = _baseline_draft_kind(parsed)
        if draft_kind and getattr(model, "auxiliary", None) is not None:
            values = stage_cache_components(
                model, runtime.parallel, tokens, capacity_padded=True,
                draft_kind=draft_kind)
            for stage, components_map in enumerate(values):
                draft[stage] += float(components_map.get(DRAFT_COMPONENT, 0.0)) \
                    * active_sequences
    return BaselinePricing(
        stages=stages,
        replaced_per_stage=tuple(replaced),
        draft_kv_per_stage=tuple(draft),
        declared_group_bytes=float(sum(group_price.values())),
        declared_layer_bytes=float(sum(layer_price.values())),
        undeclared_bytes=float(undeclared))


def _baseline_draft_kind(parsed: Any) -> str:
    runtime = parsed.runtime
    metadata = dict(runtime.metadata)
    declared = metadata.get("draft_mode")
    if declared:
        return str(declared)
    if runtime.dspark:
        return "dspark"
    if (runtime.mtp_acceptance_length > 1
            or any(value > 1
                   for _, value in runtime.mtp_acceptance_by_output_tokens)
            or metadata.get("speculative")):
        return "mtp"
    return ""


@typed_inputs(errors={"modeled": "baseline_pricing_modeled_bytes_invalid",
                      "replaced": "baseline_pricing_replaced_bytes_invalid"})
def stage_deltas(*, modeled: Sequence[float], replaced: Sequence[float],
                 conservative: bool = True) -> tuple[float, ...]:
    """Per-stage net delta; floor at zero for a conservative baseline."""
    strict_bool("conservative", conservative, reason="baseline_pricing")
    for name, values in (("modeled", modeled), ("replaced", replaced)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) \
                or not values:
            raise ModelingError(f"baseline_pricing_{name}_stage_shape_invalid")
    if len(modeled) != len(replaced):
        raise ModelingError("baseline_pricing_delta_stage_mismatch")
    result = []
    for left, right in zip(modeled, replaced):
        left_value = strict_number("modeled_bytes", left, minimum=0.0,
                                   reason="baseline_pricing")
        right_value = strict_number("replaced_bytes", right, minimum=0.0,
                                    reason="baseline_pricing")
        assert left_value is not None and right_value is not None
        delta = left_value - right_value
        result.append(max(0.0, delta) if conservative else delta)
    return tuple(result)


__all__ = ["BaselinePricing", "DRAFT_COMPONENT", "REPLACED_COMPONENTS",
           "SMALL_CLASS_COMPONENTS", "baseline_pricing", "stage_deltas"]
