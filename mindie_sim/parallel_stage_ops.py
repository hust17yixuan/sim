"""Canonical stage slices for all layer-dependent collectives.

[Role]
  Resolve each communicator to its owning PP stage and each layer to the
  attention, dense FFN and MoE operations it executes.
[Invariants]
  Every target layer occurs once in the canonical PP partition. Expert axes
  run on MoE layers; context axes run on attention layers. A draft's own
  layer count belongs to the final stage, matching catalog draft weights/KV.
[Timeouts & Locks] Bounded pure arithmetic; no locks or I/O.
[Side-Effect Policy] Fresh tuples and dictionaries only.
"""
from __future__ import annotations

from mindie_sim.parallel_schedule import layer_partition
from mindie_sim.schema.common import ModelingError


def stage_groups(plan, stage, axis):
    """Local communicator members, restricted to exactly one PP stage."""
    result = []
    for group in plan.get(axis + "_groups", ()):
        stages = {plan["rank_map"][rank][0] for rank in group}
        if len(stages) > 1:
            raise ModelingError(f"collective_crosses_pp_stage:{axis}")
        if stages == {stage}:
            result.append(tuple(group))
    return tuple(result)


def stage_layer_slices(case, *, draft=None):
    """Return (partition, per-stage layer indices, per-layer operation masks)."""
    p, model = case.runtime.parallel, case.model
    if draft is None:
        partition = layer_partition(model.layers, p.pp, case.runtime.pp_layer_partition or None)
        layers = tuple(model.layer_plans) or (model,) * model.layers
        masks = tuple((str(layer.ffn) == "moe",
                       str(layer.attention) not in {"none", "unmodeled", "kda", "gdn"},
                       str(layer.ffn) != "moe") for layer in layers)
    else:
        # catalog.weights.stage_weight_terms and stage_cache_components place
        # auxiliary draft layers on the final stage; earlier stages only relay.
        partition = (0,) * (p.pp - 1) + (draft.layers,)
        masks = ((draft.num_experts > 0, True, draft.num_experts == 0),) * draft.layers
    offset, stages = 0, []
    for count in partition:
        stages.append(tuple(range(offset, offset + count)))
        offset += count
    return partition, tuple(stages), masks


def axis_layer_indices(indices, masks, axis):
    if axis in {"ep", "etp", "edp"}:
        return tuple(i for i in indices if masks[i][0])
    if axis in {"pcp", "dcp"}:
        return tuple(i for i in indices if masks[i][1])
    return tuple(i for i in indices if masks[i][1] or masks[i][2])
