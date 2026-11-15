"""Source recurrence with resident QKV and finite state-output queues.

[Role] Lower the arch35 FP32-state/BF16-input RecurrentGatedDeltaRule branch.
[Invariants] Whole-sequence QKV is loaded once per value head. State writes
  begin before the query dot; a single output slot prevents reuse until DMA
  completes. Acceptance selects the initial address, never fewer query rows.
  The consumed version is the committed token count; candidate checkpoints
  exist before token commit, and only the accepted offset becomes visible.
  Prefill effects distinguish cache gather, the local H chunk bank and the
  post-H scatter publish; conv effects are tail writebacks, not chunk commits.
  Padding graph slots own no state and retain no candidate.
[Timeouts & Locks] O(sequences + owners) plan, bounded head/value/token windows.
[Side-Effect Policy] Pure physical work and virtual Engine tasks; no I/O.
"""
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import StateEffect
from .engine import Allocation
from .mechanism import OperatorPlan, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

IMPLEMENTATION = "ascend_recurrent_gdn_fp32_state"
STATE_CONTRACT_KEY = "gdn_state_contract"
STATE_EFFECTS_KEY = "gdn_state_effects"
SOURCE = {
    "repository": "Ascend/ops-transformer",
    "path": "attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule_arch35.h",
    "sha256": "5590f9a236f83660df7b216af2d535c4cb07a8ccd96fa07ebc6be3f78c9a03c2",
    "ranges": ((135, 158), (188, 249), (285, 399), (405, 475)),
    "host": {
        "path": "attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp",
        "sha256": "785101bff3cc4454ffdf731661b940b69254850c5eb5fab7232bb118085cb9ed",
        "block_count_lines": (327, 329),
    },
}
UB_BYTES = 247808


def _state_effect(target, action, version, condition, arbiter="", decision="",
                  **extra):
    data = StateEffect(target, action, version, condition, arbiter).to_dict()
    data["decision"] = decision
    for key, value in extra.items():
        if value is not None:
            data[key] = value
    return data


STATE_DIRECTIONS = frozenset({"token", "conv", "gather", "h", "scatter", "call", "mixed"})
STATE_TARGET_KINDS = frozenset({"recurrent", "conv", "local", "call"})
_TARGET_FIELD = {
    "recurrent": "recurrent_state",
    "conv": "conv_state",
    "local": "chunk_bank",
    "call": "final_state",
}

_SLOT_SCOPES = {"token": "cache", "conv": "cache", "gather": "cache_read",
                "h": "local", "scatter": "cache_publish", "call": "call"}
_SLOT_PUBLISH = {"token": "token_commit", "conv": "sequence_tail",
                 "gather": "before_h", "h": "scatter_complete",
                 "scatter": "scatter_complete", "call": "call_return"}


def _state_target(prefix, kind, request):
    return f"{prefix}.{_TARGET_FIELD[kind]}[{request}]"


def tensor_port(value, *, ready="", role="", owner="", storage="",
                generation=None):
    """Complete typed value port, including the physical layout view.

    A port is the serializable form of a :class:`TensorValue` plus the actual
    completion edge.  It keeps producer/output, storage/alias, the strided
    offset and the state generation, so a stage cannot fall back to a bare
    producer string and silently lose a narrowed subview.
    """
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        data = dict(value.to_dict())
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise ModelingError("gdn_state_port_requires_tensor_value")
    layout = dict(data.get("layout") or {})
    data.setdefault("dtype", "")
    data.setdefault("output", 0)
    data.setdefault("alias_of", "")
    data.setdefault("state_version", None)
    data.setdefault("binding", "static")
    data.setdefault("binding_name", "")
    data["offset"] = int(layout.get("offset", data.get("offset", 0)) or 0)
    data["shape"] = tuple(layout.get("shape") or data.get("shape") or ())
    data["strides"] = tuple(layout.get("strides") or data.get("strides") or ())
    data["storage_shape"] = tuple(layout.get("storage_shape")
                                  or data.get("storage_shape") or ())
    data["ready"] = str(ready or data.get("ready") or "")
    data["role"] = str(role or data.get("role") or "")
    data["owner"] = str(owner or data.get("owner") or "")
    data["storage"] = str(storage or data.get("storage") or "")
    if generation is not None:
        data["generation"] = int(generation)
    elif data.get("state_version") is not None:
        data["generation"] = int(data["state_version"])
    else:
        data["generation"] = None
    return data


def _slot_ports(values, slots):
    if values is None:
        return (None,) * slots
    if hasattr(values, "to_dict") or isinstance(values, Mapping):
        return (values,) + (None,) * max(0, slots - 1)
    values = tuple(values)
    if len(values) < slots:
        values = values + (None,) * (slots - len(values))
    return values[:slots]


def state_contract(*, views, phase, state_dtype, conv_state_dtype, conv_kernel,
                   target_prefix="gdn", zero_slots=0, chunk=64,
                   index_row_stride=1, index_slots=None, state_layout="V,K",
                   initial_state=None, direction="token",
                   target_kind="recurrent", global_lengths=None,
                   slot_indices=None, state_capacity=None, slot_source="",
                   remote_publish=False, directions=None, decisions=None,
                   value_inputs=None, value_outputs=None, remote_input=None,
                   heads=0, head_shards=None):
    """Declare one state boundary at its real scope and occurrence.

    ``directions`` selects the actual event per slot.  The selected native
    mixed caller uses ``token`` for its speculative decode rows and ``conv``
    for its prefill rows in one physical convolution launch; the top-level
    direction becomes ``mixed`` but every slot keeps its own event, version
    and decision.  ``decisions`` supplies the actual request acceptance
    (accepted/advanced/pending/ready) instead of inferring it from a batch
    summary.  ``value_inputs``/``value_outputs`` must carry complete
    ``TensorValue`` descriptors (layout offsets included), never a producer
    label alone.
    """
    if phase not in {"decode", "prefill", "mixed"}:
        raise ModelingError(f"gdn_state_phase_unknown:{phase}")
    views = tuple(views)
    slots = len(views) + zero_slots
    if directions is None:
        slot_directions = (direction,) * slots
    else:
        slot_directions = tuple(directions)
        if len(slot_directions) != len(views):
            raise ModelingError("gdn_state_directions_must_match_slots")
        if any(value not in STATE_DIRECTIONS for value in slot_directions):
            raise ModelingError("gdn_state_direction_unknown")
        slot_directions = slot_directions + (slot_directions[0] if slot_directions
                                             else direction,) * zero_slots
    if any(value not in STATE_DIRECTIONS for value in slot_directions):
        raise ModelingError(f"gdn_state_direction_unknown:{direction}")
    uniform = len(set(slot_directions)) <= 1
    contract_direction = slot_directions[0] if uniform else "mixed"
    if target_kind not in STATE_TARGET_KINDS:
        raise ModelingError(f"gdn_state_target_kind_unknown:{target_kind}")
    if state_dtype is not None:
        dtype_bytes(state_dtype)
    if conv_state_dtype is not None:
        dtype_bytes(conv_state_dtype)
    if not isinstance(conv_kernel, int) or isinstance(conv_kernel, bool) or conv_kernel < 0:
        raise ModelingError(f"gdn_state_kernel_width_invalid:{conv_kernel}")
    if not isinstance(index_row_stride, int) or isinstance(index_row_stride, bool) \
            or index_row_stride < 1:
        raise ModelingError("gdn_state_index_row_stride_invalid")
    if not isinstance(zero_slots, int) or isinstance(zero_slots, bool) or zero_slots < 0:
        raise ModelingError("gdn_state_zero_slots_invalid")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ModelingError("gdn_state_chunk_invalid")
    if index_slots is None:
        index_slots = slots
    if not isinstance(index_slots, int) or isinstance(index_slots, bool) or index_slots < slots:
        raise ModelingError(f"gdn_state_index_slots_invalid:{index_slots}:{slots}")
    if initial_state is not None:
        initial_state = tuple(bool(value) for value in initial_state)
        if len(initial_state) != len(views):
            raise ModelingError("gdn_state_initial_flags_must_match_slots")
    if global_lengths is not None:
        global_lengths = tuple(int(value) for value in global_lengths)
        if len(global_lengths) != len(views):
            raise ModelingError("gdn_state_global_lengths_must_match_slots")
    if slot_indices is not None:
        slot_indices = tuple(None if value is None else int(value)
                             for value in slot_indices)
        if len(slot_indices) != slots:
            raise ModelingError("gdn_state_slot_indices_must_match_slots")
        if any(value is not None and value < 0 for value in slot_indices):
            raise ModelingError("gdn_state_slot_indices_must_be_nonnegative")
    known_slots = tuple(value for value in (slot_indices or ())
                        if value is not None and value != 0)
    capacity_lower_bound = max(known_slots, default=0) + 1 if known_slots else None
    if state_capacity is not None:
        if isinstance(state_capacity, bool) or not isinstance(state_capacity, int) \
                or state_capacity < 1:
            raise ModelingError("gdn_state_capacity_invalid")
        if capacity_lower_bound is not None and capacity_lower_bound > state_capacity:
            raise ModelingError("gdn_state_slot_outside_capacity")
    if isinstance(remote_publish, (list, tuple)):
        remote_flags = tuple(bool(value) for value in remote_publish)
        if len(remote_flags) != len(views):
            raise ModelingError("gdn_state_remote_publish_must_match_slots")
    else:
        remote_flags = (bool(remote_publish),) * len(views)
    if not remote_flags:
        remote_flags = (bool(remote_publish),)
    value_inputs = _slot_ports(value_inputs, slots)
    value_outputs = _slot_ports(value_outputs, slots)
    remote_input = _slot_ports(remote_input, slots)
    decision_map = dict(decisions or {})
    head_shards = int(head_shards) if isinstance(head_shards, int) else None
    seen: set[str] = set()
    padding_slots, active_slots, skipped, fresh = [], [], [], []
    queries, global_queries, contexts, query_ranges = [], [], [], []
    accepted_offsets, advances = [], []
    checkpoints, read_versions = [], []
    commit_versions, publish_versions, local_bank_versions = [], [], []
    local_chunk_versions, local_checkpoint_versions, targets, slot_requests = [], [], [], []
    local_commit_versions = []
    slot_commit_policy, pending_slots = [], []
    slot_phases, slot_scopes, slot_publish_points = [], [], []
    candidates, decision_inputs, decision_ready = [], [], []
    input_ports, output_ports, remote_ports = [], [], []
    pending = False
    publish_point = "per_slot" if not uniform else _SLOT_PUBLISH[slot_directions[0]]
    scope = "per_slot" if not uniform else _SLOT_SCOPES[slot_directions[0]]
    effects: list[dict] = []

    for slot, view in enumerate(views):
        view_phase = (str(getattr(view, "phase", phase))
                      if phase == "mixed" else phase)
        if view_phase not in {"decode", "prefill"}:
            raise ModelingError(f"gdn_state_slot_phase_unknown:{view_phase}")
        slot_phase = view_phase
        slot_direction = slot_directions[slot]
        if slot_direction == "token" and slot_phase != "decode":
            raise ModelingError(f"gdn_state_token_direction_requires_decode:{view_phase}")
        if slot_direction in {"gather", "h", "scatter"} and slot_phase != "prefill":
            raise ModelingError(f"gdn_state_prefill_direction_requires_prefill:{slot_direction}")
        slot_phases.append(slot_phase)
        request = str(getattr(view, "request_id", slot))
        if not request or request in seen:
            raise ModelingError(f"gdn_state_request_id_invalid:{request}")
        seen.add(request)
        q = int(getattr(view, "q_len", 0))
        context = int(getattr(view, "context", 0))
        if q < 0 or context < 0:
            raise ModelingError(f"gdn_state_geometry_invalid:{request}")
        if global_lengths is not None:
            global_q = int(global_lengths[slot])
        else:
            declared = int(getattr(view, "global_q_len", 0) or 0)
            global_q = declared if declared else q
        if global_q < q:
            raise ModelingError(f"gdn_state_global_query_below_local:{request}")
        ranges = tuple(tuple(int(value) for value in item)
                       for item in (getattr(view, "query_ranges", ()) or ()))
        if any(len(item) != 2 or item[0] < 0 or item[1] <= item[0]
               or item[1] > global_q for item in ranges):
            raise ModelingError(f"gdn_state_query_ranges_invalid:{request}")
        padding = bool(getattr(view, "padding", False))
        queries.append(q)
        global_queries.append(global_q)
        contexts.append(context)
        query_ranges.append(ranges)
        accepted_offsets.append(None)
        advances.append(0)
        targets.append(_state_target(target_prefix, target_kind, request))
        decision = dict(decision_map.get(request) or {})
        decision_input = (f"acceptance[{request}]" if slot_direction == "token"
                          and not padding and q > 0 else None)
        decision_inputs.append(decision_input)
        decision_ready.append(str(decision.get("ready")
                                  or (decision_input or "")))
        input_ports.append(tensor_port(value_inputs[slot], role="state_input"))
        if padding:
            padding_slots.append(slot)
            checkpoints.append(0)
            read_versions.append(None)
            commit_versions.append(None)
            publish_versions.append(None)
            local_bank_versions.append(None)
            local_chunk_versions.append(())
            local_checkpoint_versions.append(())
            local_commit_versions.append(None)
            slot_commit_policy.append("none")
            slot_requests.append(request)
            candidates.append(())
            output_ports.append(None)
            remote_ports.append(None)
            continue
        if q == 0:
            skipped.append(slot)
            checkpoints.append(0)
            read_versions.append(None)
            commit_versions.append(None)
            publish_versions.append(None)
            local_bank_versions.append(None)
            local_chunk_versions.append(())
            local_checkpoint_versions.append(())
            local_commit_versions.append(None)
            slot_commit_policy.append("none")
            slot_requests.append(request)
            candidates.append(())
            output_ports.append(None)
            remote_ports.append(None)
            continue
        active_slots.append(slot)
        has_initial = (bool(initial_state[slot]) if initial_state is not None
                       else context > 0)
        view_ranges = tuple(getattr(view, "query_ranges", ()) or ())
        range_start = int(view_ranges[0][0]) if view_ranges else 0
        range_end = int(view_ranges[-1][1]) if view_ranges else q
        read_context = (context + range_start
                        if slot_direction in {"scatter", "conv"} and range_start > 0
                        else context)
        if not has_initial:
            fresh.append(slot)
        target = targets[-1]
        read_versions.append(read_context if has_initial else None)
        slot_candidates = ()
        local_commit = None
        local_final = None
        local_chunks = ()
        local_checkpoints = ()
        commit = None
        if slot_direction == "token":
            checkpoint_count = q
            slot_pending = False
            if decision:
                if bool(decision.get("pending")):
                    accepted, advance, slot_pending = 0, 0, True
                else:
                    advance = int(decision.get("advanced", 0) or 0)
                    accepted = int(decision.get("accepted", 0) or 0)
                    if advance <= 0:
                        accepted, advance, slot_pending = 0, 0, True
                    elif not 1 <= advance <= q:
                        raise ModelingError(f"gdn_state_advance_outside_query:{request}")
                    elif q == 1:
                        if advance != 1 or accepted != 0:
                            raise ModelingError(
                                f"gdn_state_single_row_acceptance_invalid:{request}")
                    elif accepted not in (0, advance - 1):
                        raise ModelingError(
                            f"gdn_state_accepted_advance_mismatch:{request}")
            elif q == 1:
                accepted, advance = 0, 1
            else:
                accepted_raw = int(getattr(view, "draft_accepted_tokens", 0) or 0)
                advance_raw = int(getattr(view, "draft_advanced_tokens", 0) or 0)
                if advance_raw <= 0:
                    # Verification candidates exist but the acceptance event has
                    # not been declared.  Do not infer that every row commits.
                    accepted, advance, slot_pending = 0, 0, True
                    pending = True
                else:
                    if not 1 <= advance_raw <= q:
                        raise ModelingError(f"gdn_state_advance_outside_query:{request}")
                    accepted = advance_raw - 1
                    if accepted_raw not in (0, accepted):
                        raise ModelingError(f"gdn_state_accepted_advance_mismatch:{request}")
                    advance = advance_raw
            accepted_offsets[-1] = accepted
            advances[-1] = advance
            if has_initial:
                effects.append(_state_effect(target, "read", context, ""))
            # Every query row owns a concrete candidate version; the accepted
            # commitment is one of these live versions, not an unbound future.
            for offset in range(checkpoint_count):
                effects.append(_state_effect(target, "write", context + offset + 1, ""))
            if slot_pending:
                commit = None
            else:
                commit = context + advance
                effects.append(_state_effect(
                    target, "commit", commit, "", "acceptance_trace",
                    f"acceptance[{request}]",
                    decision_input=decision_input,
                    decision_ready=decision_ready[-1]))
                for offset in range(advance, q):
                    effects.append(_state_effect(
                        target, "rollback", context + offset + 1, "",
                        "acceptance_trace", f"acceptance[{request}]",
                        decision_input=decision_input,
                        decision_ready=decision_ready[-1]))
            # Candidate storage remains addressable per query row until the
            # request decision and the last asynchronous reader finish.
            slot_candidates = tuple({
                "version": context + offset + 1,
                "target": target,
                "request": request,
                "heads": int(heads or 0),
                "head_shards": int(head_shards or 1),
                "ready": "",
                "storage": f"{target}@candidate",
                "decision": decision_input,
            } for offset in range(checkpoint_count))
            slot_publish = "per_slot" if not uniform else "token_commit"
            slot_scope = "per_slot" if not uniform else "cache"
        else:
            checkpoint_count = 1
            final = context + global_q
            if slot_direction == "h":
                local_final = q
                local_chunks = tuple(min(chunk * step, q)
                                     for step in range(1, (q + chunk - 1) // chunk + 1))
            else:
                local_final = None
                local_chunks = ()
            if slot_direction in {"h", "conv", "scatter"}:
                local_base = (read_context if slot_direction in {"conv", "scatter"}
                              else context)
                local_checkpoints = tuple(local_base + offset + 1
                                          for offset in range(q))
            slot_publish = _SLOT_PUBLISH[slot_direction]
            slot_scope = _SLOT_SCOPES[slot_direction]
            remote = bool(remote_flags[slot] and slot_direction in {"conv", "scatter"})
            local_commit = (context + range_end
                            if slot_direction in {"conv", "scatter"}
                            else (q if slot_direction == "h" else None))
            commit = final if (slot_direction in {"conv", "scatter"} and not remote) else None
            if remote:
                slot_publish = "remote_complete"
            if slot_direction == "gather":
                if has_initial:
                    effects.append(_state_effect(
                        target, "read", read_context, ""))
            elif slot_direction == "h":
                if has_initial:
                    effects.append(_state_effect(
                        target, "read", context, ""))
                effects.append(_state_effect(
                    target, "write", q, ""))
            elif slot_direction == "scatter":
                if has_initial:
                    effects.append(_state_effect(
                        target, "read", read_context, ""))
                effects.append(_state_effect(
                    target, "write", local_commit if remote else final, ""))
                if not remote:
                    effects.append(_state_effect(
                        target, "commit", final, ""))
            elif slot_direction == "call":
                if has_initial:
                    effects.append(_state_effect(
                        target, "read", context, ""))
                effects.append(_state_effect(
                    target, "write", final, ""))
            else:  # conv
                if has_initial:
                    effects.append(_state_effect(
                        target, "read", read_context, ""))
                effects.append(_state_effect(
                    target, "write", local_commit if remote else final, ""))
                if not remote:
                    effects.append(_state_effect(
                        target, "commit", final, ""))
        slot_scopes.append(slot_scope)
        slot_publish_points.append(slot_publish)
        local_commit_versions.append(local_commit)
        slot_requests.append(request)
        if slot_direction == "token":
            slot_policy = ("pending_acceptance_event" if slot_pending
                           else "token_commit")
        else:
            slot_policy = ({
                "conv": "remote_pending", "scatter": "remote_pending"}[slot_direction]
                if remote_flags[slot] and slot_direction in {"conv", "scatter"} else
                {"conv": "sequence_tail", "scatter": "scatter_complete",
                 "h": "h_return_no_persistent_commit", "gather": "read_only",
                 "call": "call_return_no_cache_commit"}[slot_direction])
        slot_commit_policy.append(slot_policy)
        if slot_policy == "pending_acceptance_event":
            pending_slots.append(slot)
        checkpoints.append(checkpoint_count)
        # The per-slot normalized commit is the only authority.  A pending
        # request never rewrites a different request's commit version.
        commit_versions.append(commit)
        publish_versions.append(
            (None if slot_direction in {"conv", "scatter"} and remote_flags[slot]
             else (context + global_q) if slot_direction in {"conv", "scatter"} else
             (commit if slot_direction == "token" else None)))
        local_bank_versions.append(local_final)
        local_chunk_versions.append(local_chunks)
        local_checkpoint_versions.append(local_checkpoints)
        candidates.append(slot_candidates)
        remote_ports.append(tensor_port(remote_input[slot], role="remote_input"))
        output_version = (publish_versions[-1] if publish_versions[-1] is not None
                          else (local_final if local_final is not None else commit))
        output_ports.append(tensor_port(value_outputs[slot], role="state_output",
                                        generation=output_version))
    for slot in range(len(views), slots):
        slot_phases.append("")
        skipped.append(slot)
        queries.append(0)
        global_queries.append(0)
        contexts.append(0)
        query_ranges.append(())
        accepted_offsets.append(None)
        advances.append(0)
        checkpoints.append(0)
        read_versions.append(None)
        commit_versions.append(None)
        publish_versions.append(None)
        local_bank_versions.append(None)
        local_chunk_versions.append(())
        local_checkpoint_versions.append(())
        local_commit_versions.append(None)
        slot_commit_policy.append("none")
        slot_requests.append("")
        candidates.append(())
        decision_inputs.append(None)
        decision_ready.append("")
        input_ports.append(None)
        output_ports.append(None)
        remote_ports.append(None)
        slot_scopes.append("")
        slot_publish_points.append("")
        targets.append("")
    if pending:
        commit_policy = "pending_acceptance_event"
    elif not uniform:
        commit_policy = "per_slot"
    elif slot_directions and slot_directions[0] == "token":
        commit_policy = "token_commit"
    elif slot_directions and slot_directions[0] == "conv":
        commit_policy = ("remote_pending"
                         if remote_flags and remote_flags[0] else "sequence_tail")
    elif slot_directions and slot_directions[0] == "scatter":
        commit_policy = ("remote_pending"
                         if remote_flags and remote_flags[0] else "scatter_complete")
    elif slot_directions and slot_directions[0] == "h":
        commit_policy = "h_return_no_persistent_commit"
    elif slot_directions and slot_directions[0] == "gather":
        commit_policy = "read_only"
    else:
        commit_policy = "call_return_no_cache_commit"
    copy_required = bool(index_row_stride > 1 and index_slots > 1)
    boundaries = ("cache_index_single_column_view",)
    if copy_required:
        boundaries += ("cache_index_normalized_copy",)
    if state_dtype is not None:
        boundaries += (f"recurrent_state_{state_dtype}_no_cast",)
    if conv_state_dtype is not None:
        boundaries += (f"conv_history_{conv_state_dtype}",)
    contract = {
        "phase": phase,
        "direction": contract_direction,
        "target_kind": target_kind,
        "scope": scope,
        "publish_point": publish_point,
        "slot_phases": tuple(slot_phases),
        "slot_directions": tuple(slot_directions),
        "slot_scopes": tuple(slot_scopes),
        "slot_publish_points": tuple(slot_publish_points),
        "state_dtype": state_dtype or "",
        "conv_state_dtype": conv_state_dtype or "",
        "conv_state_layout": "slots_history_channels",
        "conv_kernel": conv_kernel,
        "state_layout": state_layout,
        "index_row_stride": index_row_stride,
        "index_slots": index_slots,
        "index_copy_required": copy_required,
        "slots": slots,
        "slot_indices": slot_indices,
        "state_capacity": state_capacity,
        "state_capacity_lower_bound": capacity_lower_bound,
        "slot_source": slot_source,
        "padding_slots": tuple(padding_slots),
        "active_slots": tuple(active_slots),
        "fresh_slots": tuple(fresh),
        "no_state_io_slots": tuple(sorted(padding_slots + skipped)),
        "request_ids": tuple(slot_requests),
        "queries": tuple(queries),
        "global_queries": tuple(global_queries),
        "query_ranges": tuple(query_ranges),
        "contexts": tuple(contexts),
        "accepted_offsets": tuple(accepted_offsets),
        "advances": tuple(advances),
        "checkpoints": tuple(checkpoints),
        "read_versions": tuple(read_versions),
        "commit_versions": tuple(commit_versions),
        "publish_versions": tuple(publish_versions),
        "local_commit_versions": tuple(local_commit_versions),
        "global_versions": tuple(
            ((contexts[slot] + global_queries[slot])
             if slot_directions[slot] in {"conv", "scatter"} and remote_flags[slot] and
             slot in active_slots else None)
            for slot in range(slots)),
        "publish_pending": ("remote_producer"
                            if any(remote_flags[slot] for slot in active_slots) and any(
                                slot_directions[slot] in {"conv", "scatter"}
                                for slot in active_slots)
                            else ""),
        "local_bank_versions": tuple(local_bank_versions),
        "local_chunk_versions": tuple(local_chunk_versions),
        "local_checkpoint_versions": tuple(local_checkpoint_versions),
        "internal_commit_versions": (),
        "targets": tuple(targets),
        "commit_policy": commit_policy,
        "slot_commit_policy": tuple(slot_commit_policy),
        "pending_slots": tuple(pending_slots),
        "decision_inputs": tuple(decision_inputs),
        "decision_ready": tuple(decision_ready),
        "candidate_stores": tuple(candidates),
        "candidate_write_point": "device_complete_before_token_commit",
        "accepted_checkpoint": ("per_slot" if not uniform else
                                "accepted_offset" if slot_directions[0] == "token"
                                else "final"),
        "rejected_checkpoints": "not_published",
        "value_inputs": tuple(input_ports),
        "value_outputs": tuple(output_ports),
        "remote_inputs": tuple(remote_ports),
        "head_shards": int(head_shards or 1),
        "heads": int(heads or 0),
        "copy_boundaries": boundaries,
    }
    return contract, tuple(effects)


def bind_state_ports(contract, op_id, *, heads=0, head_shards=None,
                     value_outputs=None, remote_input=None, remote_publish=None,
                     decision_ready=None):
    """Attach an actual stage completion and its complete value ports.

    Called after the physical op exists.  Candidate readiness is the whole
    launched head/shard store set, not an early per-head completion.  A
    missing explicit output port is derived from the same contract version
    that the effects publish.
    """
    if contract is None:
        return None, ()
    data = dict(contract)
    data["op_id"] = str(op_id)
    data["ready"] = str(op_id)
    if heads:
        data["heads"] = int(heads)
    if head_shards is not None:
        data["head_shards"] = int(head_shards)
    stores = []
    for slot, slot_stores in enumerate(contract.get("candidate_stores") or ()):
        row = []
        for store in slot_stores or ():
            item = dict(store)
            item.setdefault("heads", int(heads or 0))
            item["head_shards"] = int(item.get("head_shards")
                                      or data.get("head_shards") or 1)
            item["ready"] = str(op_id)
            item["completion"] = "all_head_shard_stores"
            row.append(item)
        stores.append(tuple(row))
    data["candidate_stores"] = tuple(stores)
    if decision_ready is not None:
        data["decision_ready"] = tuple(decision_ready)
    outputs = list(contract.get("value_outputs") or ())
    while len(outputs) < len(data.get("targets") or ()):
        outputs.append(None)
    versions = tuple(contract.get("publish_versions") or ())
    locals_ = tuple(contract.get("local_bank_versions") or ())
    commits = tuple(contract.get("commit_versions") or ())
    read_versions = tuple(contract.get("read_versions") or ())
    if value_outputs is not None:
        explicit = _slot_ports(value_outputs, len(outputs))
    else:
        explicit = (None,) * len(outputs)
    for slot in range(len(outputs)):
        if explicit[slot] is not None:
            outputs[slot] = tensor_port(explicit[slot], role="state_output",
                                        generation=(versions[slot]
                                                    if slot < len(versions) else None))
            continue
        if outputs[slot] is not None:
            item = dict(outputs[slot])
            item["ready"] = str(op_id)
            item["owner"] = str(op_id)
            outputs[slot] = item
            continue
        if slot >= len(data.get("targets") or ()) or not data["targets"][slot]:
            continue
        slot_dirs = tuple(contract.get("slot_directions") or ())
        slot_direction = (slot_dirs[slot] if slot < len(slot_dirs)
                          else contract.get("direction", ""))
        version = (versions[slot] if slot < len(versions) else None)
        local_version = (locals_[slot] if slot < len(locals_) else None)
        if slot_direction in {"h", "gather"}:
            # H returns a local chunk bank; the persistent cache generation is
            # still the version H read.  Gather materializes that same initial
            # generation for H.  Keep both identities on the port.
            version = (read_versions[slot]
                       if slot < len(read_versions)
                       and read_versions[slot] is not None else local_version)
        elif version is None:
            version = (local_version if local_version is not None else
                       (commits[slot] if slot < len(commits) else None))
        outputs[slot] = tensor_port({
            "producer": str(op_id), "dtype": contract.get("state_dtype") or "fp32",
            "output": data["targets"][slot], "alias_of": "",
            "state_version": version, "binding": "static", "binding_name": "",
            "local_version": local_version,
            "layout": {"shape": (), "strides": (), "offset": 0,
                       "storage_shape": ()},
        }, ready=str(op_id), role="state_output", owner=str(op_id),
            storage=data["targets"][slot], generation=version)
    data["value_outputs"] = tuple(outputs)
    if remote_input is not None:
        data["remote_inputs"] = _slot_ports(remote_input, len(data["targets"]))
    if remote_publish is not None:
        data["remote_publish"] = remote_publish
    return data, tuple(item for item in outputs if item is not None)


def head_ownership(lengths, heads, blocks):
    """Source IsCurrentBlock accumulates whole-head lengths before advancing."""
    target = max(math.ceil(sum(lengths) * heads / blocks), max(lengths))
    owners = [[] for _ in range(blocks)]
    owner, load = 0, 0
    for seq, q in enumerate(lengths):
        remaining = heads
        while remaining:
            count = min(remaining, math.ceil((target - load) / q))
            owners[owner].append((seq, count))
            remaining -= count
            load += count * q
            if load >= target:
                owner += 1
                load = 0
    return tuple(tuple(segments) for segments in owners if segments)


def head_runs(lengths, segments, *, compact=True):
    """Consecutive equal-q heads retain their owner's full repeated work."""
    current, count = None, 0
    for sequence, heads in segments:
        q = lengths[sequence]
        if not compact:
            for _ in range(heads):
                yield q, 1
        elif q == current:
            count += heads
        else:
            if count:
                yield current, count
            current, count = q, heads
    if compact and count:
        yield current, count


def merge_contracts(contracts, *, allow_mixed_directions=False):
    """Concatenate per-request contracts without copying the first request.

    A fused launch keeps one physical op but must retain every request's
    targets, versions, masks, slot mapping and publish policy.
    """
    contracts = tuple(contracts)
    if not contracts:
        return None
    if len(contracts) == 1:
        return contracts[0]
    first = contracts[0]
    for contract in contracts[1:]:
        for key in ("phase", "direction", "target_kind", "scope",
                    "publish_point", "state_dtype", "conv_state_dtype",
                    "conv_state_layout", "conv_kernel", "state_layout",
                    "index_row_stride", "candidate_write_point",
                    "accepted_checkpoint", "rejected_checkpoints",
                    "commit_policy"):
            if contract.get(key) != first.get(key):
                if key == "commit_policy" and (
                        "pending_acceptance_event" in {
                            contract.get(key), first.get(key)}
                        or allow_mixed_directions):
                    continue
                if (key == "direction" and allow_mixed_directions
                        and {contract.get(key), first.get(key)}
                        <= {"token", "conv"}):
                    continue
                if (key == "phase" and allow_mixed_directions
                        and {contract.get(key), first.get(key)}
                        <= {"decode", "prefill", "mixed"}):
                    continue
                if key in {"publish_point", "scope", "accepted_checkpoint",
                           "rejected_checkpoints"} and allow_mixed_directions:
                    continue
                raise ModelingError(f"gdn_state_contract_merge_mismatch:{key}")
    def concat(key):
        return tuple(item for contract in contracts
                     for item in (contract.get(key) or ()))
    def offset(key):
        result, cursor = [], 0
        for contract in contracts:
            result.extend(value + cursor for value in (contract.get(key) or ()))
            cursor += int(contract.get("slots", 0))
        return tuple(result)
    slot_indices = [contract.get("slot_indices") for contract in contracts]
    merged = dict(first)
    merged.update({
        "slots": sum(int(contract.get("slots", 0)) for contract in contracts),
        "index_slots": sum(int(contract.get("index_slots", 0))
                           for contract in contracts),
        "index_copy_required": any(bool(contract.get("index_copy_required"))
                                   for contract in contracts),
        "padding_slots": offset("padding_slots"),
        "active_slots": offset("active_slots"),
        "fresh_slots": offset("fresh_slots"),
        "no_state_io_slots": offset("no_state_io_slots"),
        "pending_slots": offset("pending_slots"),
        "slot_phases": concat("slot_phases"),
        "slot_directions": concat("slot_directions"),
        "slot_publish_points": concat("slot_publish_points"),
        "mixed_directions": any(contract.get("direction") != first.get("direction")
                                or tuple(contract.get("slot_directions") or ())
                                != tuple(first.get("slot_directions") or ())
                                for contract in contracts),
        "direction": ("mixed"
                      if any(contract.get("direction") != first.get("direction")
                             or tuple(contract.get("slot_directions") or ())
                             != tuple(first.get("slot_directions") or ())
                             for contract in contracts)
                      else first.get("direction")),
        "phase": ("mixed" if any(contract.get("phase") != first.get("phase")
                                  for contract in contracts) else first.get("phase")),
        "scope": ("per_slot"
                  if any(contract.get("scope") != first.get("scope")
                         for contract in contracts) else first.get("scope")),
        "publish_point": ("per_slot"
                          if any(contract.get("publish_point")
                                 != first.get("publish_point")
                                 for contract in contracts)
                          else first.get("publish_point")),
        "slot_scopes": concat("slot_scopes"),
        "accepted_checkpoint": ("per_slot"
                                if any(contract.get("accepted_checkpoint")
                                       != first.get("accepted_checkpoint")
                                       for contract in contracts)
                                else first.get("accepted_checkpoint")),
        "request_ids": concat("request_ids"),
        "queries": concat("queries"),
        "global_queries": concat("global_queries"),
        "query_ranges": concat("query_ranges"),
        "contexts": concat("contexts"),
        "accepted_offsets": concat("accepted_offsets"),
        "advances": concat("advances"),
        "checkpoints": concat("checkpoints"),
        "read_versions": concat("read_versions"),
        "commit_versions": concat("commit_versions"),
        "publish_versions": concat("publish_versions"),
        "local_bank_versions": concat("local_bank_versions"),
        "local_chunk_versions": concat("local_chunk_versions"),
        "local_checkpoint_versions": concat("local_checkpoint_versions"),
        "local_commit_versions": concat("local_commit_versions"),
        "global_versions": concat("global_versions"),
        "internal_commit_versions": concat("internal_commit_versions"),
        "slot_commit_policy": concat("slot_commit_policy"),
        "decision_inputs": concat("decision_inputs"),
        "decision_ready": concat("decision_ready"),
        "candidate_stores": concat("candidate_stores"),
        "value_inputs": concat("value_inputs"),
        "value_outputs": concat("value_outputs"),
        "remote_inputs": concat("remote_inputs"),
        "targets": concat("targets"),
        "heads": max((int(contract.get("heads") or 0)
                      for contract in contracts), default=0),
        "head_shards": max((int(contract.get("head_shards") or 1)
                            for contract in contracts), default=1),
        "slot_indices": (None if any(values is None for values in slot_indices)
                         else tuple(item for values in slot_indices
                                    for item in values)),
        "state_capacity": (first.get("state_capacity")
                           if all(contract.get("state_capacity")
                                  == first.get("state_capacity")
                                  for contract in contracts) else
                           (None if any(contract.get("state_capacity") is None
                                        for contract in contracts) else
                            max(contract.get("state_capacity") or 0
                                for contract in contracts))),
        "state_capacity_lower_bound": max(
            (contract.get("state_capacity_lower_bound") or 0
             for contract in contracts), default=0) or None,
        "copy_boundaries": tuple(dict.fromkeys(
            item for contract in contracts
            for item in (contract.get("copy_boundaries") or ()))),
        "publish_pending": (first.get("publish_pending", "")
                            if all(contract.get("publish_pending")
                                   == first.get("publish_pending")
                                   for contract in contracts) else
                            "remote_producer"),
        "commit_policy": ("per_slot"
                          if any(contract.get("commit_policy")
                                 != first.get("commit_policy")
                                 for contract in contracts)
                          else first.get("commit_policy")),
    })
    return merged


def _state_plan_meta(meta):
    """Program-level mirror of the lowering's typed state effects."""
    result = {}
    if "gdn_state_contract" in meta:
        result["state_contract"] = meta["gdn_state_contract"]
    if "gdn_state_effects" in meta:
        result["state_effects"] = meta["gdn_state_effects"]
    return result


def build(op, geometry, *, compact=True):
    meta = dict(op.metadata)
    lengths = tuple(meta["sequence_tokens"])
    if not any(length > 0 for length in lengths):
        # A graph-padded step can own physical rows without an active sequence.
        # Its host grid and buffers remain, but the source recurrence performs
        # no state work; keep an explicit empty program rather than inventing a
        # token or falling back to a different implementation.
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"empty_active_sequences": True,
                                      **_state_plan_meta(meta)})
    heads, dk, dv = (int(meta[key]) for key in ("heads", "dk", "dv"))
    keys = int(meta["key_heads"])
    if not (1 <= dk <= 512 and 1 <= dv <= 512 and 1 <= heads <= 256
            and keys > 0 and heads % keys == 0):
        raise ValueError("source_recurrent_gdn_head_dimensions_outside_host_domain")
    ak, av = math.ceil(dk / 16) * 16, math.ceil(dv / 16) * 16
    if op.dtype != "bf16" or meta["state_dtype"] != "fp32" or max(lengths) > 8:
        raise ValueError("source_recurrent_gdn_requires_bf16_fp32_state_q_le8")
    if max(lengths) * max(ak, av) * 4 > 16384:
        raise ValueError("source_recurrent_gdn_resident_qkv_exceeds_16k_buffer")
    value_step = 8192 // ak
    value_step = (255 if value_step > 256 else value_step) // 16 * 16
    if not value_step:
        raise ValueError("source_recurrent_gdn_state_tile_exceeds_32k_buffer")
    parts = math.ceil(dv / value_step)
    units = geometry.resource_units.get("vector", geometry.compute_units)
    blocks = min(units, len(lengths) * heads)
    ownership = head_ownership(lengths, heads, blocks)
    run_counts = tuple(sum(1 for _ in head_runs(lengths, segments, compact=compact))
                       for segments in ownership)
    read = read_role(geometry.resource_units, aiv=0)
    total = sum(lengths) * heads
    tensor_reads = len(lengths) * heads * dk * dv * 4 + total * (2 * dk + dv) * 2
    scalar_payload = total * (4 + 2 * parts)
    writes = total * (dk * dv * 4 + dv * 2)
    vector = len(lengths) * heads * ak * dv + total * (8 * ak * dv + dv + 6 * ak + av)
    work = Work(resources={"hbm": tensor_reads + scalar_payload + writes,
                           read: tensor_reads, "mte3": writes, "vector": vector})

    def program(context):
        def owner_program(owner, segments):
            owned = context.owned(owner, units, ("vector", read, "mte3"))
            def head(q, repeat):
                # This scales real repeated work, including each distinct
                # head's QKV/state traffic; it does not scale a duration or
                # keep tensors from different heads simultaneously resident.
                def v(amount):
                    return context.bind(Work(resources={"vector": amount}).scaled(repeat))
                def memory(amount, role):
                    return context.bind(Work(resources={"hbm": amount, role: amount}).scaled(repeat))
                def part(part_index):
                    n = min(value_step, dv - part_index * value_step)
                    def token(t):
                        first = part_index == 0 and t == 0
                        # Scalar GetValue payload is not an MTE2 instruction.
                        # Its issue/cache latency needs a separate hardware
                        # declaration; no PMU counter becomes a stage delay.
                        scalar = 2 + (4 if part_index == 0 else 0)
                        stages = [
                            Stage("scalar_inputs", context.bind(Work(resources={"hbm": scalar}).scaled(repeat)),
                                  previous=("attention_cast",)),
                        ]
                        prev = "scalar_inputs"
                        if part_index == 0:
                            stages.append(Stage("gamma", v(2 * ak), after=(prev,), placement=owned))
                            prev = "gamma"
                        stages.append(Stage("decay", v(ak * n), after=(prev,), placement=owned))
                        prev = "decay"
                        if first:
                            stages.append(Stage("qk_load", memory(q * 2 * dk * 2, read),
                                                after=(prev,), placement=owned))
                            stages.append(Stage("qk_cast_scale", v(q * 3 * ak),
                                                after=("qk_load",), placement=owned))
                            prev = "qk_cast_scale"
                        stages.append(Stage("key_dot", v((2 * ak - 1) * n), after=(prev,), placement=owned))
                        prev = "key_dot"
                        if first:
                            stages.append(Stage("v_load", memory(q * dv * 2, read),
                                                after=(prev,), placement=owned))
                            stages.append(Stage("v_cast", v(q * av),
                                                after=("v_load",), placement=owned))
                            prev = "v_cast"
                        stages.extend((
                            Stage("delta", v(2 * n), after=(prev,), placement=owned),
                            Stage("rank_one_update", v(2 * ak * n), after=("delta",), placement=owned),
                            Stage("state_copy", v(ak * n), after=("rank_one_update",),
                                  previous=("state_write",), placement=owned),
                            Stage("state_write", memory(dk * n * 4, "mte3"), after=("state_copy",),
                                  previous=("attention_write",), placement=owned),
                            Stage("query_dot", v((2 * ak - 1) * n), after=("state_copy",), placement=owned),
                            Stage("attention_cast", v(n), after=("query_dot",),
                                  previous=("attention_write",), placement=owned),
                            Stage("attention_write", memory(n * 2, "mte3"),
                                  after=("attention_cast", "state_write"), placement=owned),
                        ))
                        return tuple(stages)
                    return (Stage("state_load", memory(dk * n * 4, read), placement=owned),
                            Stage("state_resident_copy", v(ak * n), after=("state_load",), placement=owned),
                            Stage("tokens", after=("state_resident_copy",), program=PipelineProgram(q, 2, token)))
                return (Stage("gamma_zero", v(q * ak), placement=owned),
                        Stage("value_tiles", after=("gamma_zero",), program=PipelineProgram(parts, 1, part)))
            pending = tuple(head_runs(lengths, segments, compact=compact))
            heads_program = PipelineProgram(run_counts[owner], 1,
                                            lambda index, pending=pending: head(*pending[index]))
            pool = context.pool_names.get("ub")
            if pool is None:
                return heads_program
            allocation = (Allocation("recurrent_buffers", pool, UB_BYTES,
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("heads", after=("buffers",), program=heads_program),
                                  Stage("release", after=("heads",), free=allocation)))
        # Owners stay separate programs: ``tests/test_gdn_recurrent_compaction``
        # requires every demand-carrying task of this operator to keep its own
        # single physical unit, which a summed-work cohort placement would
        # widen.  Only the boundary encoding is collapsed (see program.py).
        return ParallelProgram(tuple(owner_program(owner, segments)
                                     for owner, segments in enumerate(ownership)))
    return OperatorPlan(work, program, metadata={**_state_plan_meta(meta),
        "recurrent_head_schedule": {
        "compact": compact, "logical_heads": len(lengths) * heads,
        "resource_cycles": sum(run_counts),
        "owner_heads": tuple(sum(count for _, count in segments) for segments in ownership),
        "owner_cycles": run_counts,
        "handoff": "conservative_serial_head_and_DV_tile_retirement",
        "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
    }})
