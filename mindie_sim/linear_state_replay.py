"""RecoverSSM verify/commit, align pre-copy and prefix tail replay costs.

[Role]
  Price the two-stage RecoverSSM verify and post-sampler commit, the standard
  ``(K + 1)`` full-state write path, the align-mode cross-block pre-copy and
  the checkpoint-aligned prefix tail replay (including bounded SWA targets).

[Invariants]
  * Linear expected work uses the exact ``E[m]``; nonlinear align-boundary work
    is integrated over the exact accepted-length PMF ``P(M = m)`` for
    ``m in {1..K+1}``.  ``E[m]`` is never rounded before cost evaluation.
  * Stage 1 reads one committed state/conv page and writes only compact
    records plus ``spec_output``.  Stage 2 reads the page, the value-tiled
    records and, for a crossing request only, one extra conv history window;
    it writes the committed page once and the boundary page for crossings.
    Each checkpoint write is counted once in ``write_bytes`` and
    ``hbm_bytes``; ``full_state_writes = 1 + boundary``.
  * One per-branch record feeds reads, writes, FLOPs and Stage-2 timing, so
    the HBM component of ``total_ms`` always matches ``hbm_bytes``.
  * Stage-2 tiles and program grids follow ``recoverssm_launch_geometry``;
    fixed launch counts are plan=1, conv=1, commit=1 and align-boundary=1.
  * ``precopy_cost`` is zero outside ``mamba_cache_mode="align"``; inside
    align it copies state and the source's conv window and still charges the
    fast-exit launches on non-crossing steps.
  * Prefix replay is a one-time restoration/prefix quantity, reported apart
    from steady-state per-step decode overhead.
  * All dimensions are strict integers; booleans, NaN and negatives reject.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller inputs are not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.linear_state_physics import (
    LinearLayer, RecoverSsmConfig, accepted_token_pmf, boundary_crossings,
    expected_accepted_tokens, validate_recoverssm,
)
from mindie_sim.nr_s7_validate import (
    positive_rate, strict_bool, strict_dtype, strict_int, strict_number, typed_sequence,
)
from mindie_sim.recoverssm_launch_geometry import (
    DEFAULT_UB_BYTES, RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES,
    RECOVERSSM_COMMIT_UB_MATRICES, CommitLaunchGeometry,
    commit_launch_geometry,
)
from mindie_sim.recoverssm_commit_work import commit_record_bytes, commit_trace_work
from mindie_sim.schema.common import ModelingError, dtype_bytes


def _int(name: str, value: Any, *, minimum: int = 0) -> int:
    return strict_int(name, value, minimum=minimum, reason="linear_state")


def _bool(name: str, value: Any) -> bool:
    return strict_bool(name, value, reason="linear_state")


def _num(name: str, value: Any, *, minimum: float = 0.0,
         allow_none: bool = False) -> float | None:
    return strict_number(name, value, minimum=minimum, allow_none=allow_none,
                         reason="linear_state")


def _rate(name: str, value: Any) -> float:
    return positive_rate(name, value, reason="linear_state")


def _optional_rate(name: str, value: Any) -> float | None:
    """A declared rate must be finite and strictly positive, never coerced."""
    if value is None:
        return None
    return positive_rate(name, value, reason="linear_state")


def _layers(value: Any) -> tuple[LinearLayer, ...]:
    if (not isinstance(value, tuple) or not value
            or not all(isinstance(layer, LinearLayer) for layer in value)):
        raise ModelingError("linear_state_cost_layers_invalid")
    return value


@typed_inputs
@dataclass(frozen=True)
class VerifyCommitCost:
    """One speculative verify plus the post-sampler commit."""

    config: RecoverSsmConfig
    layers: tuple[LinearLayer, ...]
    batch: int
    sequence_tokens: int
    hbm_bytes_per_s: float
    compute_flops_per_s: float | None
    kernel_launch_ms: float
    accepted_tokens: int | None = None
    accept_rate: float = 0.0
    ub_bytes: int = DEFAULT_UB_BYTES
    output_dtype: str = "bf16"
    is_a5: bool = False
    program_dispatch_us: float = 0.0
    conv_dim_stride: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.config, RecoverSsmConfig):
            raise ModelingError("linear_state_cost_config_invalid")
        object.__setattr__(self, "layers", _layers(self.layers))
        _int("batch", self.batch, minimum=1)
        _int("sequence_tokens", self.sequence_tokens, minimum=1)
        _rate("hbm_bytes_per_s", self.hbm_bytes_per_s)
        _optional_rate("compute_flops_per_s", self.compute_flops_per_s)
        _num("kernel_launch_ms", self.kernel_launch_ms)
        _int("ub_bytes", self.ub_bytes, minimum=1)
        strict_dtype("output_dtype", self.output_dtype,
                     reason="linear_state")
        _bool("is_a5", self.is_a5)
        _int("conv_dim_stride", self.conv_dim_stride, minimum=1)
        _num("program_dispatch_us", self.program_dispatch_us)
        if self.accepted_tokens is not None:
            _int("accepted_tokens", self.accepted_tokens, minimum=1)
            if self.accepted_tokens > self.config.spec_query_len:
                raise ModelingError(
                    "linear_state_accepted_exceeds_spec_window:"
                    f"{self.accepted_tokens}>{self.config.spec_query_len}")
        _num("accept_rate", self.accept_rate)
        if self.accept_rate > 1.0:
            raise ModelingError(
                f"linear_state_accept_rate_invalid:{self.accept_rate}")

    @property
    def tp(self) -> int:
        return self.config.kda_head_shard

    @property
    def pmf(self) -> tuple[tuple[int, float], ...]:
        return accepted_token_pmf(self.config.speculative_k, self.accept_rate)

    @property
    def commit_len(self) -> float:
        """Exact expected accepted length (or the explicit trace value)."""
        if self.accepted_tokens is not None:
            return float(self.accepted_tokens)
        return expected_accepted_tokens(self.config.speculative_k,
                                        self.accept_rate)

    @property
    def _boundary_threshold(self) -> int:
        start = self.sequence_tokens
        block = self.config.block_size
        return ((start // block) + 1) * block - start

    def _crosses(self, accepted: int) -> bool:
        return boundary_crossings(
            self.sequence_tokens, self.sequence_tokens + accepted,
            self.config.block_size) > 0

    @property
    def boundary_probability(self) -> float:
        if (not self.config.use_replayssm
                or self.config.mamba_cache_mode != "align"):
            return 0.0
        if self.accepted_tokens is not None:
            return 1.0 if self._crosses(self.accepted_tokens) else 0.0
        threshold = self._boundary_threshold
        return sum(probability for m, probability in self.pmf
                   if m >= threshold)

    @property
    def crossing_commit_len(self) -> float:
        """``E[m * 1{crossing}]``, the extra boundary-branch token work.

        Boundary FLOPs scale with the accepted length of the crossing traces,
        not with ``E[m] * P(crossing)``; using the product would break the
        exact per-trace expectation the module advertises.
        """
        if (not self.config.use_replayssm
                or self.config.mamba_cache_mode != "align"):
            return 0.0
        if self.accepted_tokens is not None:
            return (float(self.accepted_tokens)
                    if self._crosses(self.accepted_tokens) else 0.0)
        threshold = self._boundary_threshold
        return sum(m * probability for m, probability in self.pmf
                   if m >= threshold)

    @property
    def boundary_writes(self) -> float:
        return float(self.batch) * self.boundary_probability

    @property
    def boundary_commits(self) -> float:
        return self.boundary_writes

    @property
    def page_state_conv_bytes(self) -> float:
        """Allocated page payload, including the widened speculative conv rows."""
        return sum(layer.state_bytes(self.tp)
                   + layer.conv_bytes(self.tp,
                                      spec_query_len=self.config.spec_query_len)
                   for layer in self.layers)

    @property
    def committed_conv_bytes(self) -> float:
        """Committed conv payload: ``conv_history_len`` rows, never ``history + K``."""
        return sum(float(layer.conv_width * layer.history_rows) / self.tp
                   * dtype_bytes(layer.conv_dtype)
                   for layer in self.layers)

    @property
    def committed_page_bytes(self) -> float:
        """State plus the committed conv history the compaction reads/writes."""
        return sum(layer.state_bytes(self.tp) for layer in self.layers) \
            + self.committed_conv_bytes

    @property
    def record_bytes_per_token(self) -> float:
        return sum(layer.record_bytes_per_token(self.tp)
                   for layer in self.layers)

    @property
    def record_read_bytes_per_token(self) -> float:
        """Normal-branch record reads under the commit value tiling.

        Each value-tile program loads the full ``2 * Dk`` key/gate record and
        only its own ``Dv`` correction slice, so the key/gate plane is read
        ``ceil(Dv / block_v)`` times.  A single record width would understate
        the source's read traffic whenever more than one value tile is live.
        """
        return commit_record_bytes(self.layers, self.tp, self.commit_block_v)

    @property
    def boundary_record_read_bytes_per_token(self) -> float:
        return commit_record_bytes(self.layers, self.tp, self.boundary_block_v)

    def _commit_work(self) -> dict[str, float]:
        """Select and price each source branch before taking its expectation."""
        traces = (((self.accepted_tokens, 1.0),) if self.accepted_tokens is not None
                  else self.pmf)
        expected: dict[str, float] = {}
        for m, probability in traces:
            crossing = self.config.mamba_cache_mode == "align" and self._crosses(m)
            tile = self.boundary_block_v if crossing else self.commit_block_v
            work = commit_trace_work(self.layers, tp=self.tp, tile=tile,
                                     accepted=m, crossing=crossing)
            for name, value in work.items():
                expected[name] = expected.get(name, 0.0) + self.batch * probability * value
        return expected

    @property
    def committed_conv_history_bytes(self) -> float:
        """One crossing request's extra boundary history-window load."""
        return self.committed_conv_bytes

    @property
    def boundary_write_bytes(self) -> float:
        return self.boundary_writes * self.committed_page_bytes

    @property
    def boundary_read_bytes(self) -> float:
        """Extra boundary conv window only; ``initial_state`` is loaded once."""
        return self.boundary_writes * self.committed_conv_history_bytes

    @property
    def boundary_extra_bytes(self) -> float:
        return self.boundary_read_bytes + self.boundary_write_bytes

    @property
    def launch_geometry(self) -> CommitLaunchGeometry:
        return commit_launch_geometry(
            ub_bytes=self.ub_bytes,
            key_dim=max(layer.key_dim for layer in self.layers),
            value_dim=max(layer.value_dim for layer in self.layers),
            num_heads=max(layer.local_heads(self.tp) for layer in self.layers),
            batch=self.batch, layers=len(self.layers),
            align_mode=(self.config.use_replayssm
                        and self.config.mamba_cache_mode == "align"),
            is_a5=self.is_a5,
            recurrent_dtype=self.layers[0].recurrent_dtype,
            program_dispatch_us=self.program_dispatch_us,
            conv_dim=max(layer.conv_width // self.tp for layer in self.layers),
            conv_dim_stride=self.conv_dim_stride)

    @property
    def commit_block_v(self) -> int:
        return self.launch_geometry.commit_block_v

    @property
    def boundary_block_v(self) -> int:
        return self.launch_geometry.boundary_block_v

    @property
    def ub_tile_rows(self) -> tuple[int, int]:
        return self.launch_geometry.ub_capacity_rows

    @property
    def boundary_tile_slowdown(self) -> float:
        return self.launch_geometry.program_grid_ratio

    def _flops_per_token(self) -> float:
        return sum(4.0 * layer.state_elements(self.tp) for layer in self.layers)

    def _timing(self, read: float, write: float, flops: float,
                launches: int, programs: int) -> dict[str, float]:
        hbm_ms = (read + write) / self.hbm_bytes_per_s * 1000.0
        compute = self.compute_flops_per_s
        compute_ms = (flops / compute * 1000.0
                      if compute is not None and compute > 0 else 0.0)
        launch_ms = launches * self.kernel_launch_ms
        grid_ms = programs * self.program_dispatch_us / 1000.0
        return {"hbm_ms": hbm_ms, "compute_ms": compute_ms,
                "launch_ms": launch_ms, "program_grid_ms": grid_ms,
                "roofline_ms": max(hbm_ms, compute_ms),
                "total_ms": hbm_ms + compute_ms + launch_ms + grid_ms}

    def _standard(self) -> dict[str, Any]:
        page = self.page_state_conv_bytes
        read = float(self.batch) * page
        writes = self.config.spec_query_len
        write = float(self.batch * writes) * page
        flops = float(self.batch * writes) * self._flops_per_token()
        timing = self._timing(read, write, flops, len(self.layers), 0)
        return {"schema": "mindie-sim.verify-commit/2", "mode": "standard",
                "timing_model": "serial_estimate",
                "full_state_writes": float(writes),
                "stage1_full_state_writes": float(writes),
                "stage2_full_state_writes": 0.0,
                "stage1_ms": timing["total_ms"], "stage2_ms": 0.0,
                "total_ms": timing["total_ms"],
                "read_bytes": read, "write_bytes": write,
                "hbm_bytes": read + write, "flops": flops,
                "conv_state_bytes": float(self.batch) * sum(
                    layer.conv_bytes(self.tp,
                                     spec_query_len=self.config.spec_query_len)
                    for layer in self.layers),
                "compact_record_bytes": 0.0,
                "boundary_commits": 0.0, "boundary_extra_bytes": 0.0,
                "boundary_probability": 0.0,
                "ub_tile_matrices": RECOVERSSM_COMMIT_UB_MATRICES,
                "ub_tile_rows": self.ub_tile_rows,
                "commit_len": None, "accept_rate": None,
                "saved_write_bytes": 0.0, "stage2_read_bytes": 0.0,
                "kernel_launches": len(self.layers),
                "commit_block_v": 0, "boundary_block_v": 0,
                "stage2_program_count": 0, **timing}

    def _recoverssm(self) -> dict[str, Any]:
        batch, spec = self.batch, self.config.spec_query_len
        page = self.committed_page_bytes
        record = self.record_bytes_per_token
        record_read = self.record_read_bytes_per_token
        expected_m = self.commit_len
        boundary_writes = self.boundary_writes
        stage1_read = float(batch) * page
        output_width = sum(layer.local_heads(self.tp) * layer.value_dim
                           for layer in self.layers)
        spec_output = float(batch * spec * output_width) * dtype_bytes(
            self.output_dtype)
        stage1_write = float(batch * spec) * record + spec_output
        stage1_flops = float(batch * spec) * self._flops_per_token()
        # One branch record per request: a crossing request pays the extra
        # boundary conv-history load, the boundary state/conv write and the
        # second accumulator/update.  The initial state is loaded once.
        boundary_conv_read = (boundary_writes
                              * self.committed_conv_history_bytes)
        work = self._commit_work()
        stage2_read = (float(batch) * page
                       + work["record_read_bytes"]
                       + boundary_conv_read)
        stage2_write = float(batch) * page + self.boundary_write_bytes
        stage2_flops = work["matrix_flops"] + work["vector_flops"]
        geometry = self.launch_geometry
        # The model invokes ``kda_recoverssm_verify`` once per KDA layer; the
        # batched Stage-2 commit stays a fixed handful of launches.
        stage1 = self._timing(stage1_read, stage1_write, stage1_flops,
                              len(self.layers), 0)
        stage2 = self._timing(stage2_read, stage2_write, stage2_flops,
                              geometry.kernel_launches, geometry.program_count)
        standard_write = float(batch * spec) * page
        return {
            "schema": "mindie-sim.verify-commit/2", "mode": "recoverssm",
            "timing_model": "serial_estimate",
            "full_state_writes": 1.0 + boundary_writes,
            "stage1_full_state_writes": 0.0,
            "stage2_full_state_writes": 1.0 + boundary_writes,
            "stage1_ms": stage1["total_ms"], "stage2_ms": stage2["total_ms"],
            "total_ms": stage1["total_ms"] + stage2["total_ms"],
            "read_bytes": stage1_read + stage2_read,
            "write_bytes": stage1_write + stage2_write,
            "hbm_bytes": (stage1_read + stage1_write
                          + stage2_read + stage2_write),
            "flops": stage1_flops + stage2_flops,
            "compact_record_bytes": float(batch * spec) * record,
            "record_read_bytes_per_token": work["record_read_bytes"] / (batch * expected_m),
            "normal_record_read_bytes_per_token": record_read,
            "boundary_record_read_bytes_per_token": self.boundary_record_read_bytes_per_token,
            "stage2_record_read_bytes": work["record_read_bytes"],
            "stage2_matrix_flops": work["matrix_flops"],
            "boundary_matrix_flops": work["boundary_matrix_flops"],
            "stage2_vector_flops": work["vector_flops"],
            "stage2_vector_special_ops": {name: work[name] for name in (
                "vector_exp_ops", "vector_sigmoid_ops", "vector_rsqrt_ops")},
            "compute_scope": "arithmetic_flops; special_vector_ops_reported_without_rate",
            "spec_output_bytes": spec_output,
            "conv_state_bytes": float(batch) * sum(
                layer.conv_bytes(self.tp, spec_query_len=spec)
                for layer in self.layers),
            "committed_conv_bytes": float(batch) * self.committed_conv_bytes,
            "stage1_read_bytes": stage1_read,
            "stage1_write_bytes": stage1_write,
            "stage2_read_bytes": stage2_read,
            "stage2_write_bytes": stage2_write,
            "commit_len": expected_m,
            "accept_rate": self.accept_rate,
            "boundary_commits": boundary_writes,
            "boundary_probability": self.boundary_probability,
            "boundary_extra_bytes": self.boundary_extra_bytes,
            "boundary_write_bytes": self.boundary_write_bytes,
            "boundary_read_bytes": self.boundary_read_bytes,
            "ub_tile_matrices": (
                RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES
                if self.boundary_probability else RECOVERSSM_COMMIT_UB_MATRICES),
            "ub_tile_rows": self.ub_tile_rows,
            "active_ub_tile_rows": (
                self.ub_tile_rows[1] if self.boundary_probability
                else self.ub_tile_rows[0]),
            "commit_block_v": geometry.commit_block_v,
            "boundary_block_v": geometry.boundary_block_v,
            "boundary_tile_slowdown": geometry.program_grid_ratio,
            "stage2_program_count": geometry.program_count,
            "kernel_launches": len(self.layers) + geometry.kernel_launches,
            "saved_write_bytes": max(0.0, standard_write - stage2_write),
            "stage1_timing": stage1, "stage2_timing": stage2,
            "launch_geometry": geometry.to_dict()}

    def to_dict(self) -> dict[str, Any]:
        record = self._standard() if not self.config.use_replayssm else self._recoverssm()
        return record


@typed_inputs
def verify_commit_cost(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                       batch: int, sequence_tokens: int,
                       hbm_bytes_per_s: float,
                       compute_flops_per_s: float | None = None,
                       kernel_launch_ms: float = 0.0,
                       accepted_tokens: int | None = None,
                       accept_rate: float = 0.0,
                       ub_bytes: int = DEFAULT_UB_BYTES,
                       output_dtype: str = "bf16",
                       is_a5: bool = False,
                       program_dispatch_us: float = 0.0,
                       conv_dim_stride: int = 1) -> dict[str, Any]:
    """Price the standard ``recurrent_kda`` write or the two-stage path."""
    _ = validate_recoverssm(config, strict=True)
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise ModelingError("linear_state_cost_layers_invalid")
    return VerifyCommitCost(
        config=config, layers=tuple(layers), batch=batch,
        sequence_tokens=sequence_tokens, hbm_bytes_per_s=hbm_bytes_per_s,
        compute_flops_per_s=compute_flops_per_s,
        kernel_launch_ms=kernel_launch_ms,
        accepted_tokens=accepted_tokens, accept_rate=accept_rate,
        ub_bytes=ub_bytes, output_dtype=output_dtype, is_a5=is_a5,
        program_dispatch_us=program_dispatch_us,
        conv_dim_stride=conv_dim_stride).to_dict()


@typed_inputs
def recoverssm_crossover(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                         hbm_bytes_per_s: float,
                         compute_flops_per_s: float | None = None,
                         kernel_launch_ms: float = 0.0,
                         sequence_tokens: int = 2048,
                         max_batch: int = 256,
                         accept_rate: float = 0.8,
                         accepted_tokens: int | None = None,
                         program_dispatch_us: float = 0.0) -> dict[str, Any]:
    """Smallest batch where two-stage RecoverSSM beats the standard writes."""
    _int("max_batch", max_batch, minimum=1)
    _ = validate_recoverssm(config, strict=True)
    standard = config.with_replayssm(False)
    replayssm = config.with_replayssm(True)

    def _cost(cfg: RecoverSsmConfig, batch: int) -> float:
        return verify_commit_cost(
            cfg, layers, batch=batch, sequence_tokens=sequence_tokens,
            hbm_bytes_per_s=hbm_bytes_per_s,
            compute_flops_per_s=compute_flops_per_s,
            kernel_launch_ms=kernel_launch_ms,
            accepted_tokens=accepted_tokens, accept_rate=accept_rate,
            program_dispatch_us=program_dispatch_us)["total_ms"]

    standard_one, standard_two = _cost(standard, 1), _cost(standard, 2)
    replay_one, replay_two = _cost(replayssm, 1), _cost(replayssm, 2)
    slope_standard = standard_two - standard_one
    slope_replay = replay_two - replay_one
    fixed_standard = standard_one - slope_standard
    fixed_replay = replay_one - slope_replay
    crossover: int | None = None
    reason: str | None = None
    if slope_replay >= slope_standard - 1e-15:
        reason = "recoverssm_stage2_launch_never_amortized"
    else:
        crossing = (fixed_replay - fixed_standard) / (
            slope_standard - slope_replay)
        candidate = max(1, math.ceil(crossing))
        if candidate <= max_batch:
            crossover = candidate
        else:
            reason = "recoverssm_crossover_exceeds_max_batch"
    return {"crossover_batch": crossover,
            "standard_batch1_ms": standard_one,
            "recoverssm_batch1_ms": replay_one,
            "per_batch_slope_standard_ms": slope_standard,
            "per_batch_slope_recoverssm_ms": slope_replay,
            "reason": reason, "max_batch": max_batch,
            "timing_model": "serial_estimate"}


@typed_inputs
@dataclass(frozen=True)
class PrecopyCost:
    """Cross-block state copy for ``mamba_cache_mode="align"``."""

    mode: str
    layers: int
    state_slot_bytes: float
    conv_copy_bytes: float
    crossings: int
    batch: int
    hbm_bytes_per_s: float
    kernel_launch_ms: float
    layerwise: bool
    overlap_fraction: float = 0.0
    step_tokens: int = 0
    block_size: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"none", "align"}:
            raise ModelingError(f"linear_state_precopy_mode_invalid:{self.mode}")
        _int("layers", self.layers, minimum=1)
        _num("state_slot_bytes", self.state_slot_bytes)
        _num("conv_copy_bytes", self.conv_copy_bytes)
        _int("crossings", self.crossings, minimum=0)
        _int("batch", self.batch, minimum=1)
        _rate("hbm_bytes_per_s", self.hbm_bytes_per_s)
        _num("kernel_launch_ms", self.kernel_launch_ms)
        _bool("layerwise", self.layerwise)
        _num("overlap_fraction", self.overlap_fraction)
        _int("step_tokens", self.step_tokens, minimum=0)
        _int("block_size", self.block_size, minimum=1)
        if self.overlap_fraction > 1.0:
            raise ModelingError(
                f"linear_state_overlap_fraction_invalid:{self.overlap_fraction}")

    @property
    def applies(self) -> bool:
        return self.mode == "align"

    @property
    def copy_bytes(self) -> float:
        if not self.applies:
            return 0.0
        per_layer = self.state_slot_bytes + self.conv_copy_bytes
        return float(self.batch * self.crossings) * 2.0 * per_layer

    @property
    def copy_ms(self) -> float:
        return self.copy_bytes / self.hbm_bytes_per_s * 1000.0

    @property
    def kernel_launches(self) -> int:
        if not self.applies:
            return 0
        return 2 + (self.layers - 1 if self.layerwise else 0)

    @property
    def bulk_ms(self) -> float:
        if not self.applies:
            return 0.0
        return self.copy_ms + 2.0 * self.kernel_launch_ms

    @property
    def layerwise_serial_ms(self) -> float:
        if not self.applies:
            return 0.0
        return self.copy_ms + (self.layers + 1) * self.kernel_launch_ms

    @property
    def layerwise_overlapped_ms(self) -> float:
        if not self.applies:
            return 0.0
        serial = max(self.copy_ms, self.layers * self.kernel_launch_ms)
        return self.kernel_launch_ms + serial * (1.0 - self.overlap_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {"applies": self.applies, "mode": self.mode,
                "copy_bytes": self.copy_bytes, "copy_ms": self.copy_ms,
                "conv_copy_bytes": float(self.batch * self.crossings)
                * self.conv_copy_bytes,
                "bulk_ms": self.bulk_ms,
                "layerwise_serial_ms": self.layerwise_serial_ms,
                "layerwise_overlapped_ms": self.layerwise_overlapped_ms,
                "kernel_launches": self.kernel_launches,
                "crossings": self.crossings,
                "steady_state_probability": (
                    min(1.0, self.step_tokens / self.block_size)
                    if self.applies else 0.0)}


@typed_inputs
def precopy_cost(config: RecoverSsmConfig, layers: Sequence[LinearLayer], *,
                 sequence_start: int, sequence_end: int, batch: int,
                 hbm_bytes_per_s: float, kernel_launch_ms: float = 0.0,
                 layerwise: bool = False,
                 overlap_fraction: float = 0.0) -> dict[str, Any]:
    """Charge the align pre-copy (state + conv window) over the step interval."""
    _ = validate_recoverssm(config, strict=True)
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise ModelingError("linear_state_precopy_layers_invalid")
    # Validate the interval before the non-align short circuit: a disabled mode
    # must not turn a malformed interval into a valid zero-cost record.
    start = _int("sequence_start", sequence_start, minimum=0)
    end = _int("sequence_end", sequence_end, minimum=0)
    if end < start:
        raise ModelingError(f"linear_state_end_before_start:{end}<{start}")
    records = typed_sequence("precopy_layers", layers, LinearLayer, nonempty=True)
    mode = config.mamba_cache_mode
    crossings = (boundary_crossings(start, end, config.block_size)
                 if mode == "align" else 0)
    head_shard = config.kda_head_shard
    state_slot = sum(layer.state_bytes(head_shard) for layer in records)
    conv_slot = sum(layer.conv_bytes(head_shard,
                                    spec_query_len=config.spec_query_len)
                    for layer in records)
    step_tokens = max(0, end - start)
    # The copy kernel transfers the committed window only; the speculative
    # conv rows are excluded by ``token_bias = spec_query_len - 1``.
    bias_rows = config.spec_query_len - 1
    conv_copy = sum(
        float(layer.conv_width * max(0, layer.conv_rows(config.spec_query_len)
                                     - bias_rows))
        / head_shard * dtype_bytes(layer.conv_dtype)
        for layer in records)
    cost = PrecopyCost(
        mode=mode, layers=len(records), state_slot_bytes=state_slot,
        conv_copy_bytes=conv_copy, crossings=crossings, batch=batch,
        hbm_bytes_per_s=hbm_bytes_per_s, kernel_launch_ms=kernel_launch_ms,
        layerwise=layerwise, overlap_fraction=overlap_fraction,
        step_tokens=step_tokens, block_size=config.block_size)
    result = cost.to_dict()
    result["mamba_block_size"] = config.block_size
    result["conv_slot_bytes"] = conv_slot
    return result


__all__ = ["PrecopyCost", "PrefixReplayPlan", "VerifyCommitCost",
           "precopy_cost", "prefix_replay_cost", "prefix_replay_plan",
           "recoverssm_crossover", "verify_commit_cost"]


_PREFIX_EXPORTS = frozenset({"PrefixReplayPlan", "prefix_replay_cost",
                             "prefix_replay_plan"})


def __getattr__(name: str) -> Any:
    if name in _PREFIX_EXPORTS:
        from mindie_sim import linear_state_prefix_replay
        return getattr(linear_state_prefix_replay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
