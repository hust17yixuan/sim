"""Per-trace RecoverSSM record traffic and commit arithmetic.

[Role]
  Count the source commit loop before accepted-length PMF integration.
  The normal and boundary kernels use different value tiles and accumulator
  counts (recoverssm.py:577-614,787-796,1527 at the pinned reference).

[Invariants]
  * Each value program reads the full key/gate vectors and its correction slice.
  * Each accumulator performs m matrix FMAs and one initial-state multiply/add:
    2*N*(m+1) FLOPs. Boundary selects still execute the second FMA at every m.
  * Vector arithmetic is separate: norm 3*K-1, lower-bound gate 3*K,
    weighted key/decay 2*K per accumulator per token, per head/value program.
  * Exp, sigmoid and rsqrt counts are separate operation counts, not FLOPs.
    Comparisons, selects, address/metadata work and conversions are excluded.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Returns fresh dictionaries; caller records are immutable.
"""
from __future__ import annotations

from typing import Sequence

from mindie_sim.nr_s7_validate import typed_inputs
from mindie_sim.linear_state_physics import LinearLayer
from mindie_sim.nr_s7_validate import strict_bool, strict_int
from mindie_sim.schema.common import ModelingError


@typed_inputs
def commit_record_bytes(layers: Sequence[LinearLayer], tp: int, tile: int) -> float:
    """Payload bytes per accepted token for one request on this branch."""
    _validate(layers, tp, tile)
    return sum(layer.local_heads(tp) * (
        layer.value_dim * 4 + 2 * layer.key_dim * layer.kg_element_bytes
        * (-(-layer.value_dim // tile))) for layer in layers)


def _validate(layers: Sequence[LinearLayer], tp: int, tile: int) -> None:
    strict_int("tp", tp, minimum=1, reason="linear_state")
    strict_int("value_tile", tile, minimum=1, reason="linear_state")
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) \
            or not layers or not all(isinstance(x, LinearLayer) for x in layers):
        raise ModelingError("linear_state_commit_layers_invalid")


@typed_inputs
def commit_trace_work(layers: Sequence[LinearLayer], *, tp: int, tile: int,
                      accepted: int, crossing: bool) -> dict[str, float]:
    """One request's executed branch, before multiplying by batch/probability."""
    _validate(layers, tp, tile)
    strict_int("accepted", accepted, minimum=1, reason="linear_state")
    strict_bool("crossing", crossing, reason="linear_state")
    accumulators = 1 + int(crossing)
    matrix = 2 * sum(layer.state_elements(tp) for layer in layers) * (accepted + 1)
    vector_flops = exp_ops = sigmoid_ops = rsqrt_ops = 0.0
    for layer in layers:
        programs = layer.local_heads(tp) * (-(-layer.value_dim // tile))
        k = layer.key_dim
        vector_flops += programs * accepted * ((3 * k - 1) + 3 * k + 2 * k * accumulators)
        exp_ops += programs * (1 + accepted * k)  # A=exp(a_log), then decay.
        sigmoid_ops += programs * accepted * k
        rsqrt_ops += programs * accepted
    return {"record_read_bytes": accepted * commit_record_bytes(layers, tp, tile),
            "matrix_flops": matrix * accumulators,
            "boundary_matrix_flops": matrix * int(crossing),
            "vector_flops": vector_flops, "vector_exp_ops": exp_ops,
            "vector_sigmoid_ops": sigmoid_ops, "vector_rsqrt_ops": rsqrt_ops}


__all__ = ["commit_record_bytes", "commit_trace_work"]
