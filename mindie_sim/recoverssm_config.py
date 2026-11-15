"""RecoverSSM configuration, precondition codes and the GLM KDA contract.

[Role]
  Declare the RecoverSSM cache/speculative topology, reproduce
  ``validate_glm_recoverssm`` as explicit diagnostic codes, and keep the GLM
  AscendC KDA backend execution contract separate from generic state math.

[Invariants]
  * Every field is validated before use; booleans, NaN, negative dimensions
    and unknown cache modes are typed ``ModelingError`` rejections.
  * Validation reports every violated precondition, never only the first.
  * The GLM kernel contract (BF16 activation, ``Dk = Dv = 128``, conv kernel
    2..4, ``B >= K + 1`` in align mode, ``K + 1 <= 7``) is validated
    separately from the generic mathematical layer model.
  * No source file, host value or environment fact enters a decision.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.nr_s7_validate import (
    optional_text, strict_bool, strict_dtype, strict_int, strict_number,
    strict_text,
)
from mindie_sim.schema.common import ModelingError

#: Ascend KDA permits at most seven speculative tokens (K + 1 <= 7).
KDA_MAX_RECURRENT_TOKENS = 7
#: Supported KV connectors for a RecoverSSM PD deployment.
RECOVERSSM_KV_CONNECTORS = frozenset(
    {"MooncakeConnectorV2", "MooncakePullConnector"})
GLM_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})
CACHE_MODES = frozenset({"none", "align"})
#: GLM KDA kernel contract.
GLM_KDA_HEAD_DIM = 128
GLM_KDA_ACTIVATION_DTYPE = "bf16"


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="linear_state")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="linear_state")


@typed_inputs
@dataclass(frozen=True)
class RecoverSsmConfig:
    """Declared cache/speculative topology; validated by ``validate``."""

    use_replayssm: bool
    speculative_k: int
    enable_prefix_caching: bool
    mamba_cache_mode: str
    enable_stochastic_rounding: bool = False
    pp: int = 1
    dcp: int = 1
    pcp: int = 1
    tp: int = 1
    kv_connector: str | None = None
    kv_role: str | None = None
    model_type: str = "glm5_next"
    mamba_block_size: int = 0
    kv_block_size: int = 128
    swa_bounded_replay: bool = False

    def __post_init__(self) -> None:
        for name in ("use_replayssm", "enable_prefix_caching",
                     "enable_stochastic_rounding", "swa_bounded_replay"):
            strict_bool(name, getattr(self, name))
        for name in ("speculative_k", "pp", "dcp", "pcp", "tp",
                     "mamba_block_size", "kv_block_size"):
            minimum = 0 if name in {"speculative_k", "mamba_block_size"} else 1
            _int(name, getattr(self, name), minimum=minimum)
        strict_text("mamba_cache_mode", self.mamba_cache_mode, reason="linear_state")
        if self.mamba_cache_mode not in CACHE_MODES:
            raise ModelingError(
                f"linear_state_mamba_cache_mode_invalid:{self.mamba_cache_mode}")
        strict_text("model_type", self.model_type, reason="linear_state")
        optional_text("kv_connector", self.kv_connector,
                      reason="linear_state")
        optional_text("kv_role", self.kv_role, reason="linear_state")
        if self.kv_role is not None and self.kv_role not in {
                "kv_consumer", "kv_producer"}:
            raise ModelingError(f"linear_state_kv_role_invalid:{self.kv_role}")
        if self.speculative_k + 1 > KDA_MAX_RECURRENT_TOKENS \
                and self.glm_kda_backend:
            raise ModelingError(
                "linear_state_spec_tokens_exceed_kda_limit:"
                f"{self.speculative_k}")
        if self.enable_prefix_caching and self.mamba_cache_mode == "align":
            if self.mamba_block_size not in (0, self.kv_block_size):
                raise ModelingError(
                    "linear_state_mamba_block_size_must_match_kv_block:"
                    f"{self.mamba_block_size}!={self.kv_block_size}")
            if self.use_replayssm and self.block_size < self.spec_query_len:
                raise ModelingError(
                    "linear_state_align_block_smaller_than_spec_window:"
                    f"{self.block_size}<{self.spec_query_len}")

    @property
    def spec_query_len(self) -> int:
        return self.speculative_k + 1

    @property
    def glm_kda_backend(self) -> bool:
        """Whether this configuration executes the GLM AscendC KDA kernel.

        The ``K + 1 <= 7`` token cap lives in the GLM KDA constructor
        (``models/glm5next/kda.py:266-276``), not in generic standard
        linear-state mathematics.  It therefore applies to RecoverSSM
        execution or a GLM model type, never to a generic standard path.
        """
        return bool(self.use_replayssm or self.model_type in GLM_MODEL_TYPES)

    @property
    def speculative_window(self) -> int:
        return self.speculative_k

    @property
    def kda_head_shard(self) -> int:
        """KDA head-shard world: ``get_kda_parallel_size`` (PCP only at TP=1)."""
        return self.tp if self.tp != 1 else self.pcp

    @property
    def block_size(self) -> int:
        return self.mamba_block_size or self.kv_block_size

    @property
    def expected_cache_mode(self) -> str:
        return "align" if self.enable_prefix_caching else "none"

    def with_replayssm(self, value: bool) -> "RecoverSsmConfig":
        return replace(self, use_replayssm=strict_bool("use_replayssm", value))


@typed_inputs
@dataclass(frozen=True)
class RecoverSsmValidation:
    """All precondition violations, never only the first one."""

    codes: tuple[str, ...]
    expected_cache_mode: str
    model_is_glm: bool
    recoverssm_active: bool

    @property
    def ok(self) -> bool:
        return not self.codes

    def raise_if_invalid(self) -> "RecoverSsmValidation":
        if self.codes:
            raise ModelingError("recoverssm_invalid:" + ",".join(self.codes))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "codes": list(self.codes),
                "expected_cache_mode": self.expected_cache_mode,
                "model_is_glm": self.model_is_glm,
                "recoverssm_active": self.recoverssm_active}


@typed_inputs
def validate_recoverssm(config: RecoverSsmConfig, *,
                        strict: bool = False) -> RecoverSsmValidation:
    """Replicate ``validate_glm_recoverssm`` with explicit diagnostic codes."""
    if not isinstance(config, RecoverSsmConfig):
        raise ModelingError("linear_state_recoverssm_config_invalid")
    strict_bool("strict", strict, reason="linear_state")
    expected = config.expected_cache_mode
    model_is_glm = config.model_type in GLM_MODEL_TYPES
    active = config.use_replayssm
    codes: list[str] = []
    if active:
        if not model_is_glm:
            codes.append("recoverssm_requires_glm5_next_model")
        if config.speculative_k < 1:
            codes.append("recoverssm_requires_speculative_decoding")
        if config.enable_stochastic_rounding:
            codes.append("recoverssm_rejects_stochastic_rounding")
        if config.mamba_cache_mode != expected:
            codes.append("recoverssm_cache_mode_mismatch")
        if (config.mamba_cache_mode == "align"
                and config.block_size < config.spec_query_len):
            codes.append("recoverssm_align_block_smaller_than_spec_window")
        if config.pp != 1:
            codes.append("recoverssm_requires_pp1")
        if config.dcp != 1:
            codes.append("recoverssm_requires_dcp1")
        if (config.kv_connector is not None
                and config.kv_connector not in RECOVERSSM_KV_CONNECTORS):
            codes.append("recoverssm_kv_connector_unsupported")
        if config.pcp > 1:
            if config.tp != 1:
                codes.append("recoverssm_pcp_requires_tp1")
            if config.kv_connector is None:
                codes.append("recoverssm_pcp_requires_mooncake_producer")
            elif config.kv_role != "kv_producer":
                codes.append("recoverssm_pcp_requires_kv_producer")
    result = RecoverSsmValidation(tuple(codes), expected, model_is_glm, active)
    if strict:
        result.raise_if_invalid()
    return result


@typed_inputs
@dataclass(frozen=True)
class GlmKdaExecution:
    """GLM KDA backend execution constraints, kept apart from generic math."""

    kind: str
    activation_dtype: str
    key_dim: int
    value_dim: int
    conv_kernel: int
    speculative_k: int
    cache_mode: str
    block_size: int
    kda_gate_lower_bound: float
    mamba_block_size: int
    codes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.codes

    def raise_if_invalid(self) -> "GlmKdaExecution":
        if self.codes:
            raise ModelingError("glm_kda_execution_invalid:" + ",".join(self.codes))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "codes": list(self.codes),
                "kind": self.kind, "activation_dtype": self.activation_dtype,
                "key_dim": self.key_dim, "value_dim": self.value_dim,
                "conv_kernel": self.conv_kernel,
                "speculative_k": self.speculative_k,
                "cache_mode": self.cache_mode, "block_size": self.block_size,
                "kda_gate_lower_bound": self.kda_gate_lower_bound,
                "mamba_block_size": self.mamba_block_size}


@typed_inputs
def validate_glm_kda_execution(*, kind: str, activation_dtype: str,
                               key_dim: int, value_dim: int, conv_kernel: int,
                               speculative_k: int, cache_mode: str,
                               block_size: int,
                               kda_gate_lower_bound: float = -5.0,
                               mamba_block_size: int | None = None) -> GlmKdaExecution:
    """Validate one GLM AscendC KDA layer, not the generic formulas.

    Mirrors ``models/glm5next/kda.py``: BF16 activation, ``Dk = Dv = 128``,
    the ``K + 1 <= 7`` token cap, a conv kernel in ``[2, 4]`` (exactly 4 under
    speculation) and ``linear_lower_bound`` in ``[-5, 0)``.  In ``align`` mode
    the Mamba block must hold at least the whole ``K + 1`` speculative window.
    """
    strict_text("kind", kind, reason="glm_kda_execution")
    strict_dtype("activation_dtype", activation_dtype,
                 reason="glm_kda_execution")
    _int("key_dim", key_dim, minimum=1)
    _int("value_dim", value_dim, minimum=1)
    _int("conv_kernel", conv_kernel, minimum=1)
    _int("speculative_k", speculative_k, minimum=0)
    strict_text("cache_mode", cache_mode, reason="glm_kda_execution")
    if cache_mode not in CACHE_MODES:
        raise ModelingError(f"glm_kda_execution_cache_mode_invalid:{cache_mode}")
    _int("block_size", block_size, minimum=1)
    lower = strict_number("kda_gate_lower_bound", kda_gate_lower_bound,
                          minimum=-math.inf, reason="glm_kda_execution")
    assert lower is not None
    if mamba_block_size is None:
        mamba_block_size = block_size
    _int("mamba_block_size", mamba_block_size, minimum=1)
    codes: list[str] = []
    if kind != "kda":
        codes.append("glm_kda_requires_kda_layer")
    if activation_dtype != GLM_KDA_ACTIVATION_DTYPE:
        codes.append("glm_kda_requires_bf16_activation")
    if key_dim != GLM_KDA_HEAD_DIM or value_dim != GLM_KDA_HEAD_DIM:
        codes.append("glm_kda_requires_head_dim_128")
    if not 2 <= conv_kernel <= 4:
        codes.append("glm_kda_conv_kernel_out_of_range")
    if speculative_k and conv_kernel != 4:
        codes.append("glm_kda_speculation_requires_conv_kernel_4")
    if not -5.0 <= lower < 0.0:
        codes.append("glm_kda_gate_lower_bound_out_of_range")
    if speculative_k + 1 > KDA_MAX_RECURRENT_TOKENS:
        codes.append("glm_kda_spec_tokens_exceed_limit")
    if cache_mode == "align" and mamba_block_size < speculative_k + 1:
        codes.append("glm_kda_align_block_smaller_than_spec_window")
    return GlmKdaExecution(kind, activation_dtype, key_dim, value_dim,
                           conv_kernel, speculative_k, cache_mode, block_size,
                           lower, mamba_block_size, tuple(codes))

