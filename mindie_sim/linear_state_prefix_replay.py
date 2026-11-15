"""Checkpoint-aligned prefix tail replay for linear-attention layers.

[Role]
  Plan the nearest saved block-aligned checkpoint after a prefix hit and price
  the one-time checkpoint load plus the unaligned tail replay, separately from
  steady-state decode.  DeepSeek-V4.1 target SWA layers replay at most one
  window; the DSpark draft SWA cache never replays.

[Invariants]
  * ``L_ckpt = (L_hit // B) * B`` and ``L_replay = L_hit - L_ckpt``; an exact
    block multiple replays zero and a hit at zero loads no checkpoint.
  * ``cache_mode="none"`` forces an effective linear hit of zero.
  * The checkpoint read is derived from the same page contract as memory:
    RecoverSSM includes the widened conv window and packed records.
  * Replay latency is reported as ``replay_ms`` (one-time prefix restoration),
    never folded into a steady-state per-step decode overhead.
  * All dimensions are strict integers; booleans, NaN and negatives reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.linear_state_physics import (
    LinearLayer, RecoverSsmConfig, validate_recoverssm,
)
from mindie_sim.nr_s7_validate import (
    positive_rate, strict_bool, strict_int, strict_number, typed_sequence,
)
from mindie_sim.schema.common import ModelingError


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="linear_state")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="linear_state")


def _rate(name: str, value: Any) -> float:
    return positive_rate(name, value, reason="linear_state")


def _num(name: str, value: Any, *, allow_none: bool = False):
    return strict_number(name, value, allow_none=allow_none,
                         reason="linear_state")


def _layers(value: Any) -> tuple[LinearLayer, ...]:
    if (not isinstance(value, tuple) or not value
            or not all(isinstance(layer, LinearLayer) for layer in value)):
        raise ModelingError("linear_state_replay_layers_invalid")
    return value


@typed_inputs
@dataclass(frozen=True)
class PrefixReplayPlan:
    """Checkpoint-aligned prefix replay for linear layers and SWA targets."""

    config: RecoverSsmConfig
    layers: tuple[LinearLayer, ...]
    isl: int
    prefix_hit: int
    swa_window: int = 0
    swa_bounded_replay: bool = False
    swa_architecture_supported: bool = True
    draft_swa: bool = False
    draft_cache_class: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, RecoverSsmConfig):
            raise ModelingError("linear_state_replay_config_invalid")
        object.__setattr__(self, "layers", _layers(self.layers))
        _int("isl", self.isl, minimum=1)
        _int("prefix_hit", self.prefix_hit, minimum=0)
        _int("swa_window", self.swa_window, minimum=0)
        _bool("swa_bounded_replay", self.swa_bounded_replay)
        _bool("swa_architecture_supported", self.swa_architecture_supported)
        _bool("draft_swa", self.draft_swa)
        if self.draft_cache_class is not None and type(self.draft_cache_class) is not str:
            raise ModelingError("linear_state_draft_cache_class_invalid")
        if self.prefix_hit > self.isl:
            raise ModelingError(
                f"linear_state_prefix_hit_exceeds_isl:{self.prefix_hit}>{self.isl}")
        if self.swa_bounded_replay and self.swa_window <= 0:
            raise ModelingError("linear_state_swa_bounded_replay_requires_window")

    @property
    def draft_excluded(self) -> bool:
        return bool(self.draft_swa
                    or self.draft_cache_class == "DeepseekV41DSparkSWACache")

    @property
    def effective_prefix_hit(self) -> int:
        if (not self.config.enable_prefix_caching
                or self.config.mamba_cache_mode != "align"):
            return 0
        return self.prefix_hit

    @property
    def checkpoint_tokens(self) -> int:
        hit = self.effective_prefix_hit
        return (hit // self.config.block_size) * self.config.block_size

    @property
    def replay_tokens(self) -> int:
        return self.effective_prefix_hit - self.checkpoint_tokens

    @property
    def novel_tokens(self) -> int:
        return self.isl - self.effective_prefix_hit

    @property
    def linear_layer_tokens(self) -> int:
        return self.replay_tokens + self.novel_tokens

    @property
    def checkpoint_page(self) -> int | None:
        if self.checkpoint_tokens <= 0:
            return None
        return self.checkpoint_tokens // self.config.block_size - 1

    @property
    def swa_replay_tokens(self) -> int:
        if self.draft_excluded or not self.swa_bounded_replay:
            return 0
        if not self.swa_architecture_supported:
            return 0
        return min(self.effective_prefix_hit, self.swa_window)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        codes: list[str] = []
        if self.draft_excluded and self.swa_bounded_replay:
            codes.append("dspark_swa_bounded_replay_excluded")
        if self.swa_bounded_replay and not self.swa_architecture_supported:
            codes.append("swa_bounded_replay_unsupported_architecture")
        return tuple(codes)

    def to_dict(self) -> dict[str, Any]:
        return {"effective_prefix_hit": self.effective_prefix_hit,
                "checkpoint_tokens": self.checkpoint_tokens,
                "checkpoint_page": self.checkpoint_page,
                "replay_tokens": self.replay_tokens,
                "novel_tokens": self.novel_tokens,
                "linear_layer_tokens": self.linear_layer_tokens,
                "swa_replay_tokens": self.swa_replay_tokens,
                "draft_excluded": self.draft_excluded,
                "diagnostics": list(self.diagnostics)}


@typed_inputs
def prefix_replay_plan(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                       isl: int, prefix_hit: int, swa_window: int = 0,
                       swa_bounded_replay: bool = False,
                       swa_architecture_supported: bool = True,
                       draft_swa: bool = False,
                       draft_cache_class: str | None = None) -> dict[str, Any]:
    """Plan the checkpoint load and unaligned tail replay after a prefix hit."""
    _ = validate_recoverssm(config, strict=True)
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise ModelingError("linear_state_replay_layers_invalid")
    plan = PrefixReplayPlan(
        config=config, layers=tuple(layers), isl=isl, prefix_hit=prefix_hit,
        swa_window=swa_window, swa_bounded_replay=swa_bounded_replay,
        swa_architecture_supported=swa_architecture_supported,
        draft_swa=draft_swa, draft_cache_class=draft_cache_class)
    return plan.to_dict()


@typed_inputs
def prefix_replay_cost(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                       plan: dict[str, Any], hbm_bytes_per_s: float,
                       compute_flops_per_s: float | None = None) -> dict[str, Any]:
    """One-time checkpoint load + tail replay, reported apart from decode."""
    _rate("hbm_bytes_per_s", hbm_bytes_per_s)
    if compute_flops_per_s is not None:
        _rate("compute_flops_per_s", compute_flops_per_s)
    if not isinstance(plan, dict) or not all(key in plan for key in (
            "replay_tokens", "linear_layer_tokens", "checkpoint_tokens")):
        raise ModelingError("linear_state_replay_plan_invalid")
    replay = _int("replay_tokens", plan["replay_tokens"], minimum=0)
    linear_tokens = _int("linear_layer_tokens", plan["linear_layer_tokens"])
    checkpoint_tokens = _int("checkpoint_tokens", plan["checkpoint_tokens"],
                             minimum=0)
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise ModelingError("linear_state_replay_layers_invalid")
    records = typed_sequence("replay_layers", layers, LinearLayer, nonempty=True)
    spec = config.spec_query_len
    if config.use_replayssm:
        checkpoint_bytes = sum(layer.page_geometry(config.kda_head_shard, spec).page_bytes
                               for layer in records)
    else:
        checkpoint_bytes = sum(
            layer.state_bytes(config.kda_head_shard)
            + layer.conv_bytes(config.kda_head_shard, spec_query_len=spec)
            for layer in records)
    per_token_flops = sum(4.0 * layer.state_elements(config.kda_head_shard)
                          for layer in records)
    flops = float(linear_tokens) * per_token_flops
    load_ms = (float(checkpoint_bytes) / hbm_bytes_per_s * 1000.0
               if checkpoint_tokens > 0 else 0.0)
    compute_ms = (flops / compute_flops_per_s * 1000.0
                  if compute_flops_per_s else 0.0)
    replay_flops = float(replay) * per_token_flops
    replay_compute_ms = (replay_flops / compute_flops_per_s * 1000.0
                         if compute_flops_per_s else 0.0)
    return {"checkpoint_bytes": float(checkpoint_bytes)
            if checkpoint_tokens > 0 else 0.0,
            "replay_tokens": replay, "linear_layer_tokens": linear_tokens,
            "replay_extra_tokens": replay, "flops": flops,
            "replay_ms": load_ms + replay_compute_ms,
            "prompt_ms": load_ms + compute_ms,
            "load_ms": load_ms, "compute_ms": compute_ms,
            "replay_compute_ms": replay_compute_ms}



__all__ = ["PrefixReplayPlan", "prefix_replay_cost", "prefix_replay_plan"]
