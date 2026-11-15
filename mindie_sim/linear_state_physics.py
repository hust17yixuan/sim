"""KDA/GDN recurrent-state rollback, RecoverSSM pages and align-mode replay.

[Role]
  Price the recurrent-state storage and the two-stage RecoverSSM verify/commit
  path of hybrid KDA/GDN/Mamba models, the standard ``(K + 1)``-slot
  speculative rollback, block-aligned ``mamba_cache_mode="align"`` checkpoints
  and the prefix tail replay a checkpoint forces.

[Invariants]
  * One canonical page definition drives memory, stride, transfer and replay:
    ``state + conv(history + K rows) + (K + 1) * (correction + key/gate)``.
    Both record caches carry the physical page's leading ``num_blocks`` axis,
    so they are persistent page payload, never a per-request transient.
  * ``correction_cache`` is FP32; ``kg_cache`` matches the activation dtype
    (BF16 for GLM KDA) and is independently overridable.  At symmetric
    ``Dk = Dv = 128`` this gives an exact ``64x`` matrix-to-record ratio.
  * The standard path keeps ``K + 1`` full state slots and ``K + 1`` conv
    slots, each conv slot storing ``history + K`` rows (``num_spec=K`` is
    passed to the convolution shape calculator in both modes).
  * ``validate_glm_recoverssm`` reproduces the source preconditions; the GLM
    KDA execution contract (BF16, Dk = Dv = 128, ``B >= K + 1`` in align mode,
    speculative cap) is a separate, explicit validation.
  * ``E[m] = 1 + alpha + ... + alpha**K`` is kept exact; the accepted-length
    PMF is exposed so nonlinear boundary work can be integrated exactly.
  * ``cache_mode="none"`` keeps one page per sequence; ``"align"`` keeps
    ``ceil(L / B_mamba)`` pages.  A page stride may only widen a page.
  * All dimensions are strict integers; booleans, NaN and negatives reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    dtype_elem_bytes, strict_bool, strict_dtype, strict_int, strict_number,
    strict_text, typed_sequence, _validate_nested,
)
from mindie_sim.recoverssm_config import (
    CACHE_MODES, GLM_KDA_ACTIVATION_DTYPE, GLM_KDA_HEAD_DIM, GLM_MODEL_TYPES,
    GlmKdaExecution, KDA_MAX_RECURRENT_TOKENS, RECOVERSSM_KV_CONNECTORS,
    RecoverSsmConfig, RecoverSsmValidation, validate_glm_kda_execution,
    validate_recoverssm,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="linear_state")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="linear_state")


def _num(name: str, value: Any, *, minimum: float = 0.0,
         allow_none: bool = False) -> float | None:
    return strict_number(name, value, minimum=minimum, allow_none=allow_none,
                         reason="linear_state")


LINEAR_STATE_KINDS = frozenset({
    "kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet",
})


@typed_inputs(errors={"index": "linear_state_layer_index_invalid"})
@dataclass(frozen=True)
class LinearLayer:
    """One recurrent layer's declared state geometry (full-model widths)."""

    name: str
    kind: str = "kda"
    index: int = -1
    num_heads: int = 1
    key_dim: int = 1
    value_dim: int = 1
    conv_kernel: int = 4
    conv_width: int = 0
    recurrent_dtype: str = "fp32"
    conv_dtype: str = "bf16"
    activation_dtype: str = "bf16"
    record_kg_dtype: str | None = None
    recurrent_elements: int | None = None
    conv_history_elements: int | None = None
    kda_gate_lower_bound: float = -5.0

    def __post_init__(self) -> None:
        strict_text("layer_name", self.name, reason="linear_state")
        if not isinstance(self.kind, str) or self.kind not in LINEAR_STATE_KINDS:
            raise ModelingError(f"linear_state_layer_kind_invalid:{self.kind}")
        # A raw boolean or fraction is not a layer index; validate before any
        # arithmetic consumes the value.
        _int("layer_index", self.index, minimum=0)
        for name in ("num_heads", "key_dim", "value_dim", "conv_kernel"):
            _int(name, getattr(self, name), minimum=1)
        _int("conv_width", self.conv_width, minimum=1)
        strict_dtype("recurrent_dtype", self.recurrent_dtype,
                     reason="linear_state")
        strict_dtype("conv_dtype", self.conv_dtype, reason="linear_state")
        strict_dtype("activation_dtype", self.activation_dtype,
                     reason="linear_state")
        if self.record_kg_dtype is not None:
            strict_dtype("record_kg_dtype", self.record_kg_dtype,
                         reason="linear_state")
        if self.recurrent_elements is not None:
            _int("recurrent_elements", self.recurrent_elements, minimum=0)
            expected = self.num_heads * self.key_dim * self.value_dim
            if self.recurrent_elements != expected:
                raise ModelingError(
                    "linear_state_recurrent_elements_mismatch:"
                    f"{self.recurrent_elements}!={expected}")
        lower = strict_number("kda_gate_lower_bound",
                              self.kda_gate_lower_bound, minimum=-math.inf,
                              reason="linear_state")
        assert lower is not None
        if self.conv_history_elements is not None:
            _int("conv_history_elements", self.conv_history_elements,
                 minimum=0)
            expected = self.conv_width * (self.conv_kernel - 1)
            if self.conv_history_elements != expected:
                raise ModelingError(
                    "linear_state_conv_elements_mismatch:"
                    f"{self.conv_history_elements}!={expected}")

    @property
    def history_rows(self) -> int:
        return self.conv_kernel - 1

    @property
    def kg_element_bytes(self) -> float:
        dtype = self.record_kg_dtype or self.activation_dtype
        return dtype_elem_bytes("record_kg_dtype", dtype, reason="linear_state")

    def local_heads(self, tp: int) -> int:
        _int("tp", tp, minimum=1)
        if self.num_heads % tp:
            raise ModelingError(
                "linear_state_heads_not_divisible_by_tp:"
                f"{self.num_heads}%{tp}")
        return self.num_heads // tp

    def state_elements(self, tp: int) -> int:
        return self.local_heads(tp) * self.key_dim * self.value_dim

    def state_bytes(self, tp: int) -> float:
        return (float(self.state_elements(tp))
                * dtype_bytes(self.recurrent_dtype))

    def conv_rows(self, spec_query_len: int) -> int:
        """``conv_state_len = conv_history_len + spec_query_len - 1``."""
        _int("spec_query_len", spec_query_len, minimum=1)
        return self.history_rows + spec_query_len - 1

    def conv_bytes(self, tp: int, *, spec_query_len: int = 1) -> float:
        _int("tp", tp, minimum=1)
        if self.conv_width % tp:
            raise ModelingError(
                "linear_state_conv_width_not_divisible_by_tp:"
                f"{self.conv_width}%{tp}")
        rows = self.conv_rows(spec_query_len)
        return (float(self.conv_width * rows) / tp
                * dtype_bytes(self.conv_dtype))

    def correction_bytes(self, tp: int, *, tokens: int = 1) -> float:
        _int("tokens", tokens, minimum=0)
        return float(self.local_heads(tp) * self.value_dim * tokens) * 4.0

    def kg_bytes(self, tp: int, *, tokens: int = 1) -> float:
        _int("tokens", tokens, minimum=0)
        return (float(self.local_heads(tp) * 2 * self.key_dim * tokens)
                * self.kg_element_bytes)

    def record_bytes_per_token(self, tp: int) -> float:
        return self.correction_bytes(tp) + self.kg_bytes(tp)

    def page_geometry(self, tp: int, spec_query_len: int) -> "LayerPageGeometry":
        return LayerPageGeometry(
            layer=self, tp=tp, spec_query_len=spec_query_len)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "index": self.index,
                "num_heads": self.num_heads, "key_dim": self.key_dim,
                "value_dim": self.value_dim, "conv_kernel": self.conv_kernel,
                "conv_width": self.conv_width,
                "recurrent_dtype": self.recurrent_dtype,
                "conv_dtype": self.conv_dtype,
                "activation_dtype": self.activation_dtype,
                "record_kg_dtype": self.record_kg_dtype or self.activation_dtype,
                "kda_gate_lower_bound": self.kda_gate_lower_bound}


@typed_inputs
@dataclass(frozen=True)
class LayerPageGeometry:
    """One layer's canonical RecoverSSM physical page."""

    layer: LinearLayer
    tp: int
    spec_query_len: int

    def __post_init__(self) -> None:
        if not isinstance(self.layer, LinearLayer):
            raise ModelingError("linear_state_page_layer_invalid")
        _int("tp", self.tp, minimum=1)
        _int("spec_query_len", self.spec_query_len, minimum=1)

    @property
    def conv_rows(self) -> int:
        return self.layer.conv_rows(self.spec_query_len)

    @property
    def recurrent_state_bytes(self) -> float:
        return self.layer.state_bytes(self.tp)

    @property
    def conv_state_bytes(self) -> float:
        return self.layer.conv_bytes(self.tp,
                                     spec_query_len=self.spec_query_len)

    @property
    def state_conv_bytes(self) -> float:
        return self.recurrent_state_bytes + self.conv_state_bytes

    @property
    def record_bytes_per_token(self) -> float:
        return self.layer.record_bytes_per_token(self.tp)

    @property
    def packed_record_bytes(self) -> float:
        return float(self.spec_query_len) * self.record_bytes_per_token

    @property
    def page_bytes(self) -> float:
        return self.state_conv_bytes + self.packed_record_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer.name, "tp": self.tp,
                "spec_query_len": self.spec_query_len,
                "conv_rows": self.conv_rows,
                "recurrent_state_bytes": self.recurrent_state_bytes,
                "conv_state_bytes": self.conv_state_bytes,
                "packed_record_bytes": self.packed_record_bytes,
                "page_bytes": self.page_bytes}


@typed_inputs
@dataclass(frozen=True)
class LinearStateMemoryPlan:
    """Per-request committed pages and rollback slots from one page contract."""

    config: RecoverSsmConfig
    layers: tuple[LinearLayer, ...]
    sequence_tokens: int
    active_sequences: int = 1
    page_stride_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, RecoverSsmConfig):
            raise ModelingError("linear_state_plan_config_invalid")
        if (not isinstance(self.layers, tuple) or not self.layers
                or not all(isinstance(layer, LinearLayer) for layer in self.layers)):
            raise ModelingError("linear_state_plan_layers_invalid")
        _int("sequence_tokens", self.sequence_tokens, minimum=1)
        _int("active_sequences", self.active_sequences, minimum=1)
        _ = validate_recoverssm(self.config, strict=True)
        if self.page_stride_bytes is not None:
            _int("page_stride_bytes", self.page_stride_bytes, minimum=1)
            if not self.config.use_replayssm:
                raise ModelingError(
                    "linear_state_page_stride_requires_replayssm")
            for page in self.page_geometries:
                if self.page_stride_bytes < page.page_bytes:
                    raise ModelingError(
                        "linear_state_page_stride_smaller_than_page:"
                        f"{self.page_stride_bytes}<{page.page_bytes}")

    @property
    def mode(self) -> str:
        return "recoverssm" if self.config.use_replayssm else "standard"

    @property
    def tp(self) -> int:
        return self.config.kda_head_shard

    @property
    def spec_query_len(self) -> int:
        return self.config.spec_query_len

    @property
    def page_geometries(self) -> tuple[LayerPageGeometry, ...]:
        return tuple(layer.page_geometry(self.tp, self.spec_query_len)
                     for layer in self.layers)

    @property
    def prefix_cache_enabled_linear(self) -> bool:
        return self.config.mamba_cache_mode == "align"

    @property
    def effective_prefix_hit_linear(self) -> int:
        return 1 if self.prefix_cache_enabled_linear else 0

    @property
    def pages_per_sequence(self) -> int:
        if not self.prefix_cache_enabled_linear:
            return 1
        return math.ceil(self.sequence_tokens / self.config.block_size)

    @property
    def state_slots_per_sequence(self) -> int:
        if self.config.use_replayssm:
            return self.pages_per_sequence
        return self.spec_query_len

    @property
    def conv_slots_per_sequence(self) -> int:
        return self.state_slots_per_sequence

    @property
    def stride_per_page(self) -> float:
        return (float(self.page_stride_bytes) if self.page_stride_bytes
                else 0.0)

    def _padded_page(self, page: LayerPageGeometry) -> float:
        return max(page.page_bytes, self.stride_per_page)

    @property
    def state_bytes_per_sequence(self) -> float:
        if self.config.use_replayssm:
            return self.pages_per_sequence * sum(
                self._padded_page(page) for page in self.page_geometries)
        return self.spec_query_len * sum(
            page.state_conv_bytes for page in self.page_geometries)

    @property
    def packed_record_bytes_per_sequence(self) -> float:
        if not self.config.use_replayssm:
            return 0.0
        return self.pages_per_sequence * sum(
            page.packed_record_bytes for page in self.page_geometries)

    @property
    def transient_verify_bytes_per_sequence(self) -> float:
        # The records live in every physical page; no per-request transient
        # record buffer exists in the Ascend page layout.
        return 0.0

    @property
    def workspace_bytes_per_sequence(self) -> float:
        return 0.0

    @property
    def total_bytes_per_sequence(self) -> float:
        return self.state_bytes_per_sequence

    @property
    def total_bytes(self) -> float:
        return self.total_bytes_per_sequence * self.active_sequences

    @property
    def padding_bytes_per_sequence(self) -> float:
        return self.pages_per_sequence * sum(
            self._padded_page(page) - page.page_bytes
            for page in self.page_geometries)

    @property
    def checkpoint_page_bytes(self) -> float:
        return sum(self._padded_page(page) for page in self.page_geometries)

    @property
    def state_conv_page_bytes(self) -> float:
        return sum(page.state_conv_bytes for page in self.page_geometries)

    @property
    def matrix_to_record_ratio(self) -> float:
        layer = self.layers[0]
        record = layer.record_bytes_per_token(self.tp)
        return layer.state_bytes(self.tp) / record if record else 0.0

    @property
    def compact_record_bytes_per_draft_token(self) -> float:
        return sum(layer.record_bytes_per_token(self.tp)
                   for layer in self.layers)

    @property
    def full_state_bytes_per_draft_token(self) -> float:
        return sum(layer.state_bytes(self.tp) for layer in self.layers)

    @property
    def compression_ratio_per_draft_token(self) -> float:
        compact = self.compact_record_bytes_per_draft_token
        return self.full_state_bytes_per_draft_token / compact if compact else 0.0

    def capacity_sequences(self, capacity_bytes: float) -> int:
        capacity = _num("capacity_bytes", capacity_bytes)
        per = self.total_bytes_per_sequence
        return int(capacity // per) if per > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "sequence_tokens": self.sequence_tokens,
                "active_sequences": self.active_sequences,
                "pages_per_sequence": self.pages_per_sequence,
                "state_slots_per_sequence": self.state_slots_per_sequence,
                "conv_slots_per_sequence": self.conv_slots_per_sequence,
                "conv_rows_per_slot": self.layers[0].conv_rows(
                    self.spec_query_len),
                "state_bytes_per_sequence": self.state_bytes_per_sequence,
                "packed_record_bytes_per_sequence":
                    self.packed_record_bytes_per_sequence,
                "transient_verify_bytes_per_sequence":
                    self.transient_verify_bytes_per_sequence,
                "total_bytes_per_sequence": self.total_bytes_per_sequence,
                "total_bytes": self.total_bytes,
                "padding_bytes_per_sequence": self.padding_bytes_per_sequence,
                "checkpoint_page_bytes": self.checkpoint_page_bytes,
                "state_conv_page_bytes": self.state_conv_page_bytes,
                "matrix_to_record_ratio": self.matrix_to_record_ratio,
                "compression_ratio_per_draft_token":
                    self.compression_ratio_per_draft_token,
                "prefix_cache_enabled_linear": self.prefix_cache_enabled_linear,
                "effective_prefix_hit_linear": self.effective_prefix_hit_linear,
                "page_stride_bytes": self.page_stride_bytes,
                "layer_pages": [page.to_dict()
                                for page in self.page_geometries],
                "layers": len(self.layers)}


@typed_inputs
def linear_state_memory(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                        sequence_tokens: int, active_sequences: int = 1,
                        page_stride_bytes: int | None = None) -> LinearStateMemoryPlan:
    """Validate and construct one recurrent-state memory plan."""
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise ModelingError("linear_state_plan_layers_invalid")
    return LinearStateMemoryPlan(
        config=config, layers=tuple(layers), sequence_tokens=sequence_tokens,
        active_sequences=active_sequences, page_stride_bytes=page_stride_bytes)


@typed_inputs
def compare_linear_state_memory(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                                sequence_tokens: int, active_sequences: int = 1,
                                page_stride_bytes: int | None = None,
                                allow_generic_model: bool = False) -> dict[str, Any]:
    """Standard vs RecoverSSM capacity; a stride applies only to RecoverSSM."""
    strict_bool("allow_generic_model", allow_generic_model, reason="linear_state")
    if page_stride_bytes is not None:
        _int("page_stride_bytes", page_stride_bytes, minimum=1)
    validate_recoverssm(config, strict=False, allow_generic_model=allow_generic_model)
    standard = linear_state_memory(
        config.with_replayssm(False), layers,
        sequence_tokens=sequence_tokens, active_sequences=active_sequences,
        page_stride_bytes=None)
    # Check the model before constructing a counterfactual that enables GLM's
    # seven-token kernel cap. An unavailable backend cannot reject standard.
    generic_ok = bool(config.generic_backend or allow_generic_model)
    if config.model_type not in {"glm5_next", "glm5_next_text"} and not generic_ok:
        codes = ["recoverssm_requires_glm5_next_model"]
    else:
        replay_config = replace(config.with_replayssm(True), generic_backend=True) if generic_ok else config.with_replayssm(True)
        codes = list(validate_recoverssm(
            replay_config, strict=False, allow_generic_model=generic_ok).codes)
    if codes:
        return {"standard": standard.to_dict(), "recoverssm": None,
                "applicable": False,
                "saved_bytes": 0.0, "saved_full_state_slots": 0,
                "ratio": 1.0, "reason": "recoverssm_inapplicable",
                "state_page_stride_bytes_applied_to": None,
                "codes": codes}
    replayssm = linear_state_memory(
        replay_config, layers, sequence_tokens=sequence_tokens,
        active_sequences=active_sequences, page_stride_bytes=page_stride_bytes)
    saved = standard.total_bytes - replayssm.total_bytes
    return {"standard": standard.to_dict(), "recoverssm": replayssm.to_dict(),
            "applicable": True,
            "saved_bytes": saved,
            "saved_full_state_slots": (
                standard.state_slots_per_sequence
                - replayssm.state_slots_per_sequence),
            "ratio": (standard.total_bytes / replayssm.total_bytes
                      if replayssm.total_bytes else 0.0),
            "state_page_stride_bytes_applied_to": (
                "recoverssm" if page_stride_bytes is not None else None),
            "reason": None, "codes": []}


@typed_inputs
def accepted_token_pmf(speculative_k: int, accept_rate: float) -> tuple[tuple[int, float], ...]:
    """Exact ``P(M = m)`` for ``m`` in ``1..K+1`` under the declared law.

    ``P(m) = (1 - alpha) * alpha**(m-1)`` for ``m <= K`` and the remaining
    mass ``alpha**K`` at ``m = K + 1``.  The tuple always sums to one, so
    nonlinear (boundary) work integrates exactly instead of using ``E[m]``.
    """
    k = _int("speculative_k", speculative_k, minimum=0)
    alpha = _num("accept_rate", accept_rate, minimum=0.0)
    if alpha > 1.0:
        raise ModelingError(f"linear_state_accept_rate_invalid:{alpha}")
    pmf = [(m, (1.0 - alpha) * alpha ** (m - 1))
           for m in range(1, k + 1)]
    pmf.append((k + 1, alpha ** k))
    return tuple(pmf)


@typed_inputs
def expected_accepted_tokens(speculative_k: int, accept_rate: float) -> float:
    """``E[m] = 1 + alpha + ... + alpha**K`` for the bonus-token law."""
    return sum(probability * m for m, probability
               in accepted_token_pmf(speculative_k, accept_rate))


@typed_inputs
def boundary_crossings(sequence_start: int, sequence_end: int,
                       block_size: int) -> int:
    """Checkpoints in ``(start, end]``; ``end`` is the post-step token count."""
    start = _int("sequence_start", sequence_start, minimum=0)
    end = _int("sequence_end", sequence_end, minimum=0)
    block = _int("block_size", block_size, minimum=1)
    if end < start:
        raise ModelingError(f"linear_state_end_before_start:{end}<{start}")
    return end // block - start // block


def _catalog_int(name: str, value: Any, *, minimum: int = 0,
                 default: int | None = None) -> int:
    if value is None or (type(value) is int and value == 0 and default is not None):
        if default is None:
            raise ModelingError(f"linear_state_catalog_{name}_missing")
        return default
    return strict_int(name, value, minimum=minimum,
                      reason="linear_state_catalog")


def _catalog_optional_int(name: str, value: Any) -> int | None:
    if value is None or (type(value) is int and value == 0):
        return None
    return strict_int(name, value, minimum=0,
                      reason="linear_state_catalog")


@typed_inputs
def linear_layers_from_model(model: Any, *, tp: int = 1) -> tuple[LinearLayer, ...]:
    """Extract recurrent layer geometry without coercing catalog values."""
    _int("tp", tp, minimum=1)
    _validate_nested(model, "model")
    plans = typed_sequence("layer_plans", getattr(model, "layer_plans", None), LayerPlan)
    _model_activation(model)
    layers: list[LinearLayer] = []
    for plan in plans:
        kind = strict_text("attention", getattr(plan, "attention", None))
        if kind not in LINEAR_STATE_KINDS:
            continue
        index = strict_int("plan_index", getattr(plan, "index", -1),
                           minimum=0, reason="linear_state_catalog")
        heads = _catalog_int("linear_heads",
                             getattr(plan, "linear_heads", None))
        key_dim = _catalog_int("linear_key_head_dim",
                               getattr(plan, "linear_key_head_dim", None))
        value_dim = _catalog_int("linear_head_dim",
                                 getattr(plan, "linear_head_dim", None))
        conv_kernel = _catalog_int("conv_kernel",
                                   getattr(plan, "conv_kernel", None))
        conv_width = _catalog_int("conv_width",
                                  getattr(plan, "conv_width", None))
        conv_elements = _catalog_optional_int(
            "conv_state_elements", getattr(plan, "conv_state_elements", None))
        dtype = strict_dtype("recurrent_state_dtype", getattr(plan, "recurrent_state_dtype", "fp32"))
        conv_dtype = strict_dtype("conv_state_dtype", getattr(plan, "conv_state_dtype", "bf16"))
        activation = _model_activation(model)
        declared_lower = getattr(plan, "kda_gate_lower_bound", None)
        lower = (-5.0 if declared_lower is None else
                 strict_number("kda_gate_lower_bound", declared_lower,
                               minimum=-math.inf,
                               reason="linear_state_catalog"))
        assert lower is not None
        layers.append(LinearLayer(
            name=f"layer{index}", index=index, kind=kind, num_heads=heads,
            key_dim=key_dim, value_dim=value_dim, conv_kernel=conv_kernel,
            conv_width=conv_width, recurrent_dtype=dtype,
            conv_dtype=conv_dtype, activation_dtype=activation,
            recurrent_elements=_catalog_optional_int(
                "recurrent_state_elements",
                getattr(plan, "recurrent_state_elements", None)),
            conv_history_elements=conv_elements,
            kda_gate_lower_bound=lower))
    for layer in layers:
        layer.local_heads(tp)
    return tuple(layers)


def _model_activation(model: Any) -> str:
    for attr in ("activation_dtype", "dtype"):
        if hasattr(model, attr):
            return strict_dtype(attr, getattr(model, attr), reason="linear_state_catalog")
    return "bf16"


validate_glm_recoverssm = validate_recoverssm

_REPLAY_EXPORTS = frozenset({
    "CommitLaunchGeometry", "PrecopyCost", "PrefixReplayPlan",
    "VerifyCommitCost", "precopy_cost", "prefix_replay_cost",
    "prefix_replay_plan", "recoverssm_crossover", "verify_commit_cost"})


def __getattr__(name: str) -> Any:
    """Lazily re-export the companion replay module's public records."""
    if name in _REPLAY_EXPORTS:
        from mindie_sim import linear_state_replay
        return getattr(linear_state_replay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CACHE_MODES", "GLM_KDA_ACTIVATION_DTYPE", "GLM_KDA_HEAD_DIM",
           "GLM_MODEL_TYPES", "GlmKdaExecution", "KDA_MAX_RECURRENT_TOKENS",
           "LINEAR_STATE_KINDS", "LayerPageGeometry", "LinearLayer",
           "LinearStateMemoryPlan", "RECOVERSSM_KV_CONNECTORS",
           "RecoverSsmConfig", "RecoverSsmValidation", "accepted_token_pmf",
           "boundary_crossings", "compare_linear_state_memory",
           "expected_accepted_tokens", "linear_layers_from_model",
           "linear_state_memory", "validate_glm_kda_execution",
           "validate_glm_recoverssm", "validate_recoverssm"]
