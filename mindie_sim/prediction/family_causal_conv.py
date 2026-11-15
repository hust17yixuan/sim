"""Explicit stateful causal-convolution update on the shared native program.

[Role] Connect typed KDA producers to one native decode/verify call.
[Invariants] Graph/SP padding rows execute without owning a state slot.
  Speculative verification writes an accepted prefix and rejects later
  candidates; every candidate still performs its real query/history work.
  Prepared weights coexist with original checkpoint weights. Input aliases
  stay views, while separate producers require a real row pack. A state
  history capacity smaller than the source ABI is an error, never a silent
  non-speculative fallback.
[Timeouts & Locks] O(sequence/projection descriptors); no locks or external I/O.
[Side-Effect Policy] Creates model metadata, weight terms and graph operators.
"""
from collections.abc import Mapping
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import ProgramFragment, SemanticAttr, StateEffect
from mindie_sim.schema.subgraphs import PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .gdn_decode_programs import CONV, CONV_SOURCE, conv_plan
from .mechanism import OperatorPlan, PlannedRecipe
from .operators.kda_state_selection import (
    bind_state_value, binding_record, binding_value, decision_ports,
    state_ports, state_records, state_value,
)
from .program import KernelProgram, Stage

IMPLEMENTATION = CONV
SOURCE = {
    "caller": {"path": "vllm_ascend/ops/kimi_kda.py",
               "sha256": "35365b3b353fc492b4dc5d75b88916431f6802c21bf51b61b4893d39d54f11f3",
               "lines": "233-255,386-430,544-555,635-648"},
    "kernel": CONV_SOURCE,
    "scope": "source_compatible_update_program; capture_binary_identity_unverified",
    "autocontiguous": "csrc/moe/causal_conv1d/op_host/causal_conv1d_def.cpp:24-28",
    "upstream_projection": {
        "path": "vllm/models/kimi_k3/nvidia/kda.py", "lines": "470-499",
        "sha256": "3663c3b273abf41e9727e2b1724e87628b64b6a9dc048a2c48d002b1f9085420"},
}


def recipes():
    return {CONV: PlannedRecipe(_planned)}


def prepare_model(model, runtime):
    prepared = {}
    for layer in model.layer_plans:
        if layer.attention != "kda":
            continue
        heads = (layer.linear_key_heads or layer.linear_heads,
                 layer.linear_value_heads or layer.linear_heads)
        if any(count <= 0 or count % runtime.parallel.tp for count in heads):
            # The retained KDA producer asserts integral head shards. A
            # channel-wise ceil would undercount replicated whole heads.
            raise ModelingError("causal_conv_update_source_requires_integral_local_heads")
        if model.activation_dtype not in {"bf16", "fp16"}:
            raise ModelingError("causal_conv_update_requires_16bit_activation")
        if layer.conv_state_dtype != model.activation_dtype:
            raise ModelingError("causal_conv_update_state_dtype_must_match_activation")
        if layer.conv_kernel not in {2, 3, 4}:
            raise ModelingError("causal_conv_update_requires_width2to4")
        conv_width = int(getattr(layer, "conv_width", 0) or 0)
        state_elements = int(getattr(layer, "conv_state_elements", 0) or 0)
        if conv_width < 1 or state_elements < 0 or state_elements % conv_width:
            raise ModelingError(
                "causal_conv_update_state_capacity_not_a_whole_history_row")
        history_slots = state_elements // conv_width
        if history_slots < layer.conv_kernel - 1:
            raise ModelingError("causal_conv_update_state_history_below_kernel_width")
        prepared[str(layer.index)] = {
            "dtype": model.activation_dtype,
            "prepared_weight_policy": "retain_original_fp32_checkpoint_weight",
            "state_layout": "slots_history_channels",
            "state_history_slots": history_slots,
            "conv_width": conv_width,
        }
        declared_history = dict(model.metadata).get(
            "causal_conv_history_acceptance", {})
        if str(layer.index) in declared_history:
            prepared[str(layer.index)]["history_acceptance"] = tuple(
                declared_history[str(layer.index)])
    metadata = {**dict(model.metadata), "causal_conv_update": {
        "implementation": IMPLEMENTATION, "layers": prepared, "source": SOURCE}}
    return replace(model, metadata=tuple(sorted(metadata.items())))


def _history_slots(model, plan):
    """Declared whole history rows for this rank's state allocation."""
    prepared = dict(dict(model.metadata).get("causal_conv_update", {}))
    entry = dict(prepared.get("layers", {})).get(str(plan.index), {})
    slots = entry.get("state_history_slots")
    if slots is None:
        width = int(getattr(plan, "conv_width", 0) or 0)
        elements = int(getattr(plan, "conv_state_elements", 0) or 0)
        if width < 1 or elements < 0 or elements % width:
            raise ModelingError(
                "causal_conv_update_state_capacity_not_a_whole_history_row")
        slots = elements // width
    if type(slots) is not int or slots < plan.conv_kernel - 1:
        raise ModelingError("causal_conv_update_state_history_below_kernel_width")
    return slots


def _per_sequence(value, count, default=None):
    if value is None or value == () or value == []:
        return (default,) * count
    if isinstance(value, Mapping):
        return tuple(value.get(index, default) for index in range(count))
    value = tuple(value)
    if len(value) == 1 and count > 1:
        return value * count
    if len(value) != count:
        raise ModelingError("causal_conv_update_values_must_match_sequences")
    return value


def _slot_storage(storage, slot):
    base = str(storage)
    marker = ":slot:"
    if marker in base:
        base = base.rsplit(marker, 1)[0]
    return base if slot is None else f"{base}{marker}{slot}"


def _narrow_window(value, offset, rows):
    shape = tuple(value.layout.shape)
    if len(shape) == 2:
        return value.narrow(0, offset, rows)
    if len(shape) == 3 and shape[0] == 1:
        return value.narrow(1, offset, rows)
    raise ModelingError(
        f"causal_conv_state_requires_history_axis:{shape}")


def _conv_bindings(*, views, active, targets, contexts, history_slots, width,
                   dtype, values, slots, owner, ready, grants):
    count = len(views)
    slots = _per_sequence(slots, count)
    ready = _per_sequence(ready, count)
    grants = _per_sequence(grants, count)
    shape = (history_slots, width)
    if values is None:
        produced = [None] * count
    elif isinstance(values, TensorValue):
        layout = tuple(values.layout.shape)
        if layout == shape and count == 1:
            produced = [values]
        elif len(layout) == 3 and layout[0] == count and layout[1:] == shape:
            produced = [values.narrow(0, index, 1) for index in range(count)]
        elif (len(layout) == 3 and layout[1:] == shape
              and all(slot is not None and slot < layout[0] for slot in slots
                      if slot is not None)):
            produced = [_narrow_window(values, slot, 1)
                        for slot in slots]
        else:
            raise ModelingError(
                f"causal_conv_state_pool_shape_mismatch:{layout}:{shape}")
    elif isinstance(values, Mapping):
        produced = [values.get(target, values.get(index))
                    for index, target in enumerate(targets)]
    elif isinstance(values, (tuple, list)):
        produced = list(values)
        if len(produced) != count:
            raise ModelingError(
                "causal_conv_update_values_must_match_sequences")
    else:
        raise ModelingError(
            "causal_conv_update_values_require_tensor_or_sequence")
    bindings = []
    for index, view in enumerate(views):
        if index not in set(active):
            bindings.append(None)
            continue
        target, context, slot = targets[index], contexts[index], slots[index]
        provided = produced[index]
        if provided is None:
            storage = _slot_storage(f"{target}.state", slot)
            provided = TensorValue(
                "external", dtype, TensorLayout.contiguous(shape),
                output=(slot if slot is not None else index),
                storage=storage, binding="external",
                binding_name=ready[index] or f"kda.conv_state[{target}]",
                state_version=context)
        else:
            provided = state_value(provided)
            if tuple(provided.layout.shape[-2:]) != shape:
                raise ModelingError(
                    f"causal_conv_state_value_shape_mismatch:"
                    f"{provided.layout.shape}:{shape}")
            if provided.dtype != dtype:
                raise ModelingError(
                    f"causal_conv_state_value_dtype_mismatch:"
                    f"{provided.dtype}:{dtype}")
        bindings.append(bind_state_value(
            provided, target=target, owner=owner, slot=slot,
            version=context, ready=ready[index], grant=grants[index],
            generation=context, required=True,
            representation="history", span=((0, history_slots), (0, width)),
            readiness=("previous_iteration" if context else "external")))
    return tuple(bindings)


def _state_contract(*, views, q_lens, padding_slots, width, byte,
                    kernel_width, history_slots, speculative, layer,
                    bindings=None, history_acceptance=None, reader_holds=None,
                    slots=None, targets=None, rebase_outputs=None,
                    state_decisions=()):
    """Declare the actual read/write versions for this sequence partition.

    Historical selection (which committed window is copied), this call's
    geometry (query rows, slots, padding) and the current acceptance decision
    stay separate fields.  Every access/effect/port is projected from the same
    caller value binding; a rebase replaces the read producer with the copy's
    actual output, not a second name for the old state.
    """
    active_slots = tuple(index for index in range(len(views))
                         if index not in set(padding_slots))
    layer_target = f"kda.layer[{layer}]" if layer >= 0 else "kda.call"
    if targets is None:
        targets = [None] * len(views)
        for index in active_slots:
            request = str(getattr(views[index], "request_id", index))
            if not request:
                raise ModelingError("causal_conv_update_request_id_invalid")
            targets[index] = f"{layer_target}.conv_state[{request}]"
    targets = tuple(targets)
    active_targets = tuple(targets[index] for index in active_slots)
    if len(set(active_targets)) != len(active_targets):
        raise ModelingError("causal_conv_update_request_id_duplicate")
    if not active_slots:
        empty = {
            "phase": "decode", "queries": q_lens, "padding_slots": padding_slots,
            "active_slots": (), "targets": (None,) * len(views),
            "read_versions": (None,) * len(views),
            "commit_versions": (None,) * len(views),
            "checkpoint_versions": ((),) * len(views),
            "accepted_offsets": (None,) * len(views),
            "advances": (0,) * len(views),
            "rejected_versions": ((),) * len(views),
            "read_records": (), "write_records": (), "commit_records": (),
            "rollback_records": (), "retained_versions": ((),) * len(views),
            "overwritten_versions": ((),) * len(views),
            "pending_acceptance": False, "commit_policy": "no_state_io",
            "candidate_write_point": "device_complete_before_token_commit",
            "state_read_bytes": 0.0, "state_write_bytes": 0.0,
            "version_basis": "invocation_local",
            "history_selection": (), "current_geometry": {},
            "acceptance": {}, "read_bindings": (), "write_bindings": (),
        }
        return empty, (), 0.0, 0.0
    contexts = tuple(int(getattr(views[index], "context", 0))
                     for index in active_slots)
    acceptance = tuple(
        (int(getattr(views[index], "draft_accepted_tokens", 0) or 0),
         int(getattr(views[index], "draft_advanced_tokens", 0) or 0))
        for index in active_slots)
    records = state_records(
        targets=active_targets, lengths=tuple(q_lens[index]
                                              for index in active_slots),
        contexts=contexts, acceptance=acceptance)
    read_bytes = write_bytes = 0.0
    for q in (q_lens[index] for index in active_slots):
        state_read = kernel_width - 1 + (2 + q if speculative else 0)
        state_write = 2 + q if speculative else kernel_width - 1
        read_bytes += state_read * width * byte
        write_bytes += state_write * width * byte
    target_by_view = [None] * len(views)
    read_versions = [None] * len(views)
    commit_versions = [None] * len(views)
    checkpoint_versions = [()] * len(views)
    accepted_offsets = [None] * len(views)
    advances = [0] * len(views)
    retained_versions = [()] * len(views)
    overwritten_versions = [()] * len(views)
    for position, index in enumerate(active_slots):
        target_by_view[index] = active_targets[position]
        read_versions[index] = records["read_versions"][position]
        commit_versions[index] = records["commit_versions"][position]
        checkpoint_versions[index] = records["checkpoint_versions"][position]
        accepted_offsets[index] = records["accepted_offsets"][position]
        advances[index] = records["advances"][position]
        retained_versions[index] = records["retained_versions"][position]
        overwritten_versions[index] = records["overwritten_versions"][position]
    effects = []
    for target, version in records["read_records"]:
        effects.append(StateEffect(target, "read", version).to_dict())
    for target, version, _slot in records["write_records"]:
        effects.append(StateEffect(target, "write", version).to_dict())
    for target, version in records["commit_records"]:
        effects.append(StateEffect(
            target, "commit", version, "", "acceptance_trace").to_dict())
    for target, version in records["rollback_records"]:
        effects.append(StateEffect(
            target, "rollback", version, "", "acceptance_trace").to_dict())
    if bindings is None:
        raise ModelingError("causal_conv_update_requires_value_bindings")
    bindings = tuple(bindings)
    if len(bindings) != len(views):
        raise ModelingError("causal_conv_update_bindings_must_match_views")
    rebase_outputs = dict(rebase_outputs or {})
    read_bindings = []
    history_selection = []
    for position, index in enumerate(active_slots):
        binding = bindings[index]
        if binding is None:
            raise ModelingError("causal_conv_update_active_binding_missing")
        accepted = (0 if history_acceptance is None
                    else int(history_acceptance[index]))
        source_offset = max(0, accepted - 1)
        history_window = (source_offset, source_offset + kernel_width - 1)
        if index in rebase_outputs:
            binding = rebase_outputs[index]
        read_bindings.append(binding)
        history_selection.append({
            "sequence": index,
            "request": str(getattr(views[index], "request_id", index)),
            "accepted_tokens": accepted,
            "source_offset": source_offset,
            "source_context": contexts[position],
            "history_window": history_window,
            "current_window": ((0, 2 + int(q_lens[index]))
                               if speculative else (0, int(q_lens[index]))),
            "slot": (None if slots is None else slots[index]),
            "rebase": index in rebase_outputs,
            "reader_hold": (None if reader_holds is None
                            else reader_holds[index]),
        })
    write_bindings = []
    by_target = {binding["target"]: binding for binding in read_bindings}
    for target, version, record_slot in records["write_records"]:
        source = by_target[target]
        sequence = active_targets.index(target)
        view_index = active_slots[sequence]
        slot = (record_slot if slots is None else slots[view_index])
        value = binding_value(source)
        write_value = replace(
            value, producer=f"{layer_target}:conv_candidate",
            output=f"checkpoint:{version}",
            storage=_slot_storage(source["storage"], slot),
            state_version=version, binding="static", binding_name="")
        write_rows = (2 + int(q_lens[view_index]) if speculative
                      else kernel_width - 1)
        write_bindings.append(bind_state_value(
            write_value, target=target, owner=source.get("owner", layer_target),
            slot=slot, version=version, ready=source.get("ready"),
            grant=source.get("grant"), generation=version,
            decision=("token_commit" if version in records["commit_versions"]
                      else None),
            last_use=("next_conv_read" if version in records["commit_versions"]
                      else "last_reader"),
            representation="history", required=True,
            span=((0, write_rows), (0, width))))
    contract = {
        **records,
        "phase": "decode",
        "queries": q_lens,
        "padding_slots": padding_slots,
        "active_slots": active_slots,
        "targets": tuple(target_by_view),
        "read_versions": tuple(read_versions),
        "commit_versions": tuple(commit_versions),
        "checkpoint_versions": tuple(checkpoint_versions),
        "accepted_offsets": tuple(accepted_offsets),
        "advances": tuple(advances),
        "rejected_versions": tuple(
            () if index in set(padding_slots)
            else records["rejected_versions"][active_slots.index(index)]
            for index in range(len(views))),
        "retained_versions": tuple(retained_versions),
        "overwritten_versions": tuple(overwritten_versions),
        "pending_acceptance": records["pending_acceptance"],
        "commit_policy": records["commit_policy"],
        "candidate_write_point": records["candidate_write_point"],
        "version_basis": records["version_basis"],
        "state_read_bytes": read_bytes,
        "state_write_bytes": write_bytes,
        "kernel_width": kernel_width,
        "speculative": speculative,
        "state_dtype": None,
        "state_layout": "slots_history_channels",
        "history_slots": history_slots,
        "read_bindings": tuple(binding_record(b) for b in read_bindings),
        "write_bindings": tuple(binding_record(b) for b in write_bindings),
        "history_selection": tuple(history_selection),
        "current_geometry": {
            "queries": tuple(q_lens),
            "active_slots": active_slots,
            "padding_slots": tuple(padding_slots),
            "kernel_width": kernel_width,
            "history_slots": history_slots,
            "width": width,
            "slots": (None if slots is None else tuple(slots)),
        },
        "acceptance": {
            "accepted_offsets": tuple(accepted_offsets),
            "advances": tuple(advances),
            "commit_versions": tuple(commit_versions),
            "rejected_versions": tuple(
                () if index in set(padding_slots)
                else records["rejected_versions"][active_slots.index(index)]
                for index in range(len(views))),
            "pending": records["pending_acceptance"],
            "commit_policy": records["commit_policy"],
        },
        "state_decisions": tuple(state_decisions),
    }
    return contract, tuple(effects), read_bytes, write_bytes
def _fragment(name, effects, ports=None):
    return ProgramFragment(name, state_effects=tuple(effects),
                           ports=PortSet() if ports is None else ports)


def state_fragments(op):
    """Explicit read/write/commit/rollback fragments for one conv op."""
    contract = dict(op.metadata).get("conv_state_contract")
    if contract is None:
        return {"read": None, "write": None, "commits": (), "rollbacks": ()}
    owner = f"kda.layer[{op.layer}]"
    read_bindings = tuple(contract.get("read_bindings", ()))
    write_bindings = tuple(contract.get("write_bindings", ()))
    read = ProgramFragment(
        f"{op.op_id}:state_read",
        state_effects=tuple(
            StateEffect(target, "read", version)
            for target, version in contract["read_records"]),
        ports=state_ports(bindings=read_bindings, owner=owner, direction="in",
                          readiness="previous_iteration", optional=False),
        attrs=(SemanticAttr("state_bindings", tuple(
            binding_record(binding) for binding in read_bindings)),)
        if read_bindings else ())
    write = ProgramFragment(
        f"{op.op_id}:state_write",
        state_effects=tuple(
            StateEffect(target, "write", version)
            for target, version, _slot in contract["write_records"]),
        ports=state_ports(bindings=write_bindings, owner=owner, direction="out",
                          readiness="producer_complete", optional=False),
        attrs=(SemanticAttr("state_bindings", tuple(
            binding_record(binding) for binding in write_bindings)),)
        if write_bindings else ())
    commits = []
    for index, (target, version) in enumerate(contract["commit_records"]):
        commits.append(ProgramFragment(
            f"{op.op_id}:state_commit:{index}",
            state_effects=(StateEffect(target, "commit", version, "",
                                       "acceptance_trace"),),
            ports=decision_ports(target=target, version=version, owner=owner),
            attrs=(SemanticAttr("decision", {
                "action": "commit", "target": target, "version": version,
                "event": _decision_name(contract, index)}),)))
    rollbacks = []
    for index, (target, version) in enumerate(contract["rollback_records"]):
        rollbacks.append(ProgramFragment(
            f"{op.op_id}:state_rollback:{index}",
            state_effects=(StateEffect(target, "rollback", version, "",
                                       "acceptance_trace"),),
            ports=decision_ports(target=target, version=version, owner=owner,
                                action="rollback"),
            attrs=(SemanticAttr("decision", {
                "action": "rollback", "target": target, "version": version,
                "event": _decision_name(contract, index)}),)))
    return {"read": read, "write": write,
            "commits": tuple(commits), "rollbacks": tuple(rollbacks)}


def _decision_name(contract, index):
    decisions = tuple(contract.get("state_decisions", ()))
    return decisions[index] if index < len(decisions) else None
def state_accesses(op):
    """Project conv UPDATE state reads/writes into SERVICE access records.

    The history window and the speculative reload window are separate ranges;
    a continuous ``kernel_width-1 + 2 + q`` span never exceeds the declared
    storage.  A rebase is a real predecessor: UPDATE reads its output at the
    selected destination window.
    """
    contract = dict(op.metadata).get("conv_state_contract")
    if contract is None:
        return ()
    width = int(contract.get("conv_width", 0) or 0)
    byte = {"bf16": 2, "fp16": 2, "fp32": 4}.get(
        str(contract.get("state_dtype", "bf16")), 2)
    rebase = dict(op.metadata).get("history_rebase_op")
    ready = tuple(contract.get("state_ready", ()))
    grants = tuple(contract.get("state_grants", ()))
    read_bindings = tuple(contract.get("read_bindings", ()))
    targets = tuple(contract["targets"])
    target_index = {target: index for index, target in enumerate(targets)
                    if target is not None}
    selection = {item["sequence"]: item
                 for item in contract.get("history_selection", ())}
    result = []
    for binding in read_bindings:
        index = target_index[binding["target"]]
        request = str(getattr(op, "requests", ())[index]) if index < len(
            getattr(op, "requests", ())) else str(index)
        q = int(contract["queries"][index])
        history = selection[index]
        source = binding.get("ready") or binding["producer"]
        base = {
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer, "direction": "read",
            "position": "state", "output": binding["target"],
            "storage": binding["storage"],
            "owner": binding.get("owner", f"kda.layer[{op.layer}]"),
            "representation": "history", "persistent": True,
            "temporary": False, "source": source, "reader": "",
            "grant": binding.get("grant") or (
                grants[index] if index < len(grants) else ""),
            "decision": None, "rank": None, "scope": "target",
            "generation": binding.get("generation"),
            "slot": binding.get("slot"), "offset": binding.get("offset"),
            "physical_key": binding["storage"],
            "request_slot": binding.get("slot"), "rows": 1,
            "row_stride": width, "tier": binding.get("tier", "hbm"),
            "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"),
            "last_use_event": "before_short_convolution",
        }
        if rebase:
            rows = int(contract["kernel_width"]) - 1
            start_row = 0
            result.append({
                **base,
                "occurrence": f"{op.op_id}:read:{index}:rebase_output",
                "state_version": binding["state_version"],
                "span": ((start_row, start_row + rows), (0, width)),
                "bytes": float(rows * width * byte),
                "last_use": "before_short_convolution",
                "source_ready": binding.get("ready"),
                "decision_event": None,
                "window": (start_row, start_row + rows),
            })
            continue
        history_rows = int(contract["kernel_width"]) - 1
        source_offset = int(history["source_offset"])
        result.append({
            **base,
            "occurrence": f"{op.op_id}:read:{index}:history",
            "state_version": binding["state_version"],
            "span": ((source_offset, source_offset + history_rows),
                     (0, width)),
            "bytes": float(history_rows * width * byte),
            "last_use": "before_short_convolution",
            "window": (source_offset, source_offset + history_rows),
            "decision_event": None,
        })
        if bool(contract["speculative"]):
            reload_rows = 2 + q
            result.append({
                **base,
                "occurrence": f"{op.op_id}:read:{index}:reload",
                "state_version": binding["state_version"],
                "span": ((0, reload_rows), (0, width)),
                "bytes": float(reload_rows * width * byte),
                "last_use": "before_short_convolution",
                "window": (0, reload_rows),
                "decision_event": None,
            })
    committed = set(contract.get("commit_versions", ()))
    decisions = tuple(contract.get("state_decisions", ()))
    for binding in tuple(contract.get("write_bindings", ())):
        index = target_index[binding["target"]]
        request = str(getattr(op, "requests", ())[index]) if index < len(
            getattr(op, "requests", ())) else str(index)
        q = int(contract["queries"][index])
        version = binding["state_version"]
        rows = 2 + q if bool(contract["speculative"]) else (
            int(contract["kernel_width"]) - 1)
        slot = binding.get("slot")
        commit = version in committed
        decision_event = decisions[index] if index < len(decisions) else None
        result.append({
            "occurrence": f"{op.op_id}:write:{index}:{version}",
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer, "direction": "write",
            "position": "state", "output": binding["target"],
            "storage": binding["storage"],
            "owner": binding.get("owner", f"kda.layer[{op.layer}]"),
            "representation": "history", "persistent": True,
            "temporary": False, "state_version": version,
            "span": ((0, rows), (0, width)),
            "bytes": float(rows * width * byte),
            "source": binding["producer"], "reader": "",
            "grant": binding.get("grant") or "",
            "decision": ("token_commit" if commit else "candidate_hold"),
            "last_use": ("next_conv_read" if commit else "last_reader"),
            "rank": None, "scope": "target", "generation": version,
            "slot": slot, "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:slot:{slot}",
            "request_slot": slot, "rows": rows, "row_stride": width,
            "tier": binding.get("tier", "hbm"),
            "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"),
            "decision_event": decision_event,
            "window": (0, rows),
            "last_use_event": ("next_conv_read" if commit else "last_reader"),
        })
    return tuple(result)
def _planned(op, geometry, *, compact=True):
    """Turn the shared conv program into a plan that carries state semantics.

    The physical program remains ``gdn_decode_programs.conv_plan``.  Only
    zero-work read/write boundary stages and plan metadata are added; no
    convolution or state work is duplicated.
    """
    plan = conv_plan(op, geometry, compact=compact)
    contract = dict(op.metadata).get("conv_state_contract")
    if contract is None:
        return plan
    fragments = state_fragments(op)

    def program(context):
        inner = plan.program(context)
        stages = []
        if fragments["read"] is not None and fragments["read"].state_effects:
            stages.append(Stage("state_read", fragment=fragments["read"]))
            stages.append(Stage("conv_owners", after=("state_read",),
                                program=inner))
        else:
            stages.append(Stage("conv_owners", program=inner))
        if fragments["write"] is not None and fragments["write"].state_effects:
            stages.append(Stage("state_publish", after=("conv_owners",),
                                fragment=fragments["write"]))
        return KernelProgram(tuple(stages))

    metadata = dict(plan.metadata)
    metadata.update({
        "state_contract": contract,
        "state_accesses": state_accesses(op),
        "state_effects": dict(op.metadata).get("conv_state_effects", ()),
        "conv_state_contract": contract,
        "conv_state_effects": dict(op.metadata).get("conv_state_effects", ()),
        "state_fragment_counts": {
            "read": 1 if fragments["read"] is not None else 0,
            "write": 1 if fragments["write"] is not None else 0,
            "commits": len(fragments["commits"]),
            "rollbacks": len(fragments["rollbacks"]),
        },
    })
    return OperatorPlan(plan.work, program, metadata=metadata)


def lower(b, tensors, *, plan, views, tokens, sources, key_width, value_width,
          history_acceptance=None, state_ready=(), state_values=None,
          state_slots=None, state_grants=(), state_reader_holds=None,
          state_targets=None, state_staging=False, state_decisions=()):
    """Bind one native update; the selected recipe owns the state writeback."""
    if not views:
        raise ModelingError("causal_conv_update_requires_decode_views")
    if any(view.phase != "decode" for view in views):
        raise ModelingError("causal_conv_update_requires_decode_only_views")
    q_lens = tuple(int(view.q_len) for view in views)
    if any(q < 1 for q in q_lens):
        raise ModelingError("causal_conv_update_requires_positive_query_rows")
    rows = sum(q_lens)
    if tokens < rows:
        raise ModelingError(
            "causal_conv_update_tokens_must_equal_sequence_partition_rows")
    width = 2 * key_width + value_width
    dtype = b.model.activation_dtype
    byte = dtype_bytes(dtype)
    history_slots = _history_slots(b.model, plan)
    padding_slots = tuple(index for index, view in enumerate(views)
                          if getattr(view, "padding", False))
    active = tuple(index for index in range(len(views))
                   if index not in set(padding_slots))
    if any(q_lens[index] == 1 and (
            int(getattr(views[index], "draft_proposal_tokens", 0) or 0)
            or int(getattr(views[index], "draft_accepted_tokens", 0) or 0)
            or int(getattr(views[index], "draft_advanced_tokens", 0) or 0) > 1)
           for index in active):
        raise ModelingError("causal_conv_update_requires_q1_non_speculative_decode")
    for index in active:
        q = q_lens[index]
        if q <= 1:
            continue
        accepted = int(getattr(views[index], "draft_accepted_tokens", 0) or 0)
        advance = int(getattr(views[index], "draft_advanced_tokens", 0) or 0)
        if advance > 0:
            if not 1 <= advance <= q:
                raise ModelingError("causal_conv_update_advance_outside_query")
            if accepted not in (0, advance - 1):
                raise ModelingError("causal_conv_update_accepted_advance_mismatch")
    if history_acceptance is None:
        prepared = dict(dict(b.model.metadata).get(
            "causal_conv_update", {})).get("layers", {})
        history_acceptance = dict(prepared.get(str(plan.index), {})).get(
            "history_acceptance")
    if history_acceptance is None:
        history_acceptance = dict(
            dict(b.model.metadata).get("causal_conv_history_acceptance", {})
        ).get(str(plan.index))
    if history_acceptance is not None:
        history_acceptance = tuple(int(value) for value in history_acceptance)
        if len(history_acceptance) != len(views) or any(
                value < 0 for value in history_acceptance):
            raise ModelingError(
                "causal_conv_update_history_acceptance_invalid")
    speculative = any(
        q_lens[index] > 1
        or int(getattr(views[index], "draft_proposal_tokens", 0) or 0) > 0
        or int(getattr(views[index], "draft_advanced_tokens", 0) or 0) > 1
        for index in active)
    if speculative:
        if plan.conv_kernel != 4:
            raise ModelingError(
                "causal_conv_update_speculative_requires_width4_history")
        max_query = max((q_lens[index] for index in active), default=1)
        if history_slots < 2 + max_query:
            raise ModelingError(
                "causal_conv_update_speculative_requires_extended_history:"
                f"{history_slots}<{2 + max_query}")
        if history_acceptance is None:
            raise ModelingError(
                "causal_conv_update_history_acceptance_required")
    state_ready = ((state_ready,) if isinstance(state_ready, str)
                   else tuple(state_ready))
    state_grants = tuple(state_grants) if state_grants else ()
    contexts = tuple(int(getattr(view, "context", 0)) for view in views)
    layer_target = (f"kda.layer[{plan.index}]" if plan.index >= 0
                    else "kda.call")
    if state_targets is None:
        targets_by_view = tuple(
            None if index in set(padding_slots)
            else f"{layer_target}.conv_state["
                 f"{getattr(views[index], 'request_id', index)}]"
            for index in range(len(views)))
    else:
        state_targets = tuple(state_targets)
        if len(state_targets) != len(views):
            raise ModelingError("causal_conv_update_targets_must_match_views")
        targets_by_view = tuple(
            None if index in set(padding_slots) else state_targets[index]
            for index in range(len(views)))
    slots = _per_sequence(state_slots, len(views))
    if all(slot is None for slot in slots):
        slots = tuple(index for index in range(len(views)))
    bindings = _conv_bindings(
        views=views, active=active, targets=targets_by_view,
        contexts=contexts, history_slots=history_slots, width=width,
        dtype=plan.conv_state_dtype, values=state_values, slots=slots,
        owner=layer_target, ready=state_ready, grants=state_grants)
    # A q1 following a speculative q4 has no source numAcceptedTokens branch
    # for its own one-row query.  The committed window is normalized by an
    # ordered history copy before the ordinary UPDATE.  In a mixed speculative
    # call the native branch is selected once for the whole call; every
    # sequence then applies its own accepted-1 offset exactly once and no
    # sequence may additionally rebase.
    rebase = tuple(
        index for index in active
        if not speculative and q_lens[index] == 1
        and history_acceptance is not None
        and history_acceptance[index] > 0)
    rebase_records = []
    rebase_outputs = {}
    rebase_ids = ()
    reader_holds = _per_sequence(state_reader_holds, len(views))
    for index in rebase:
        accepted = history_acceptance[index]
        if plan.conv_kernel != 4:
            raise ModelingError(
                "causal_conv_update_history_rebase_requires_width4")
        if accepted - 1 + plan.conv_kernel - 1 > history_slots:
            raise ModelingError(
                "causal_conv_update_history_rebase_exceeds_state")
        binding = bindings[index]
        source_value = binding_value(binding)
        source_offset = accepted - 1
        copy_rows = plan.conv_kernel - 1
        source_view = _narrow_window(source_value, source_offset, copy_rows)
        dest_view = _narrow_window(source_value, 0, copy_rows)
        hold = reader_holds[index]
        hold_window = None
        hold_event = None
        hold_grant = None
        if isinstance(hold, Mapping):
            window = hold.get("window", hold.get("rows"))
            if window is not None:
                hold_window = tuple(int(item) for item in window)
            hold_event = hold.get("event", hold.get("last_use"))
            hold_grant = hold.get("grant")
        protected = ()
        if hold_window is not None:
            start = max(hold_window[0], 0)
            end = min(hold_window[1], copy_rows)
            if end > start:
                protected = (start, end)
        staged = bool(state_staging or (protected and not hold_event))
        copy_bytes = copy_rows * width * byte
        copy_deps = tuple(state_ready)
        if hold_event:
            copy_deps = tuple(dict.fromkeys((*copy_deps, hold_event)))
        staging_id = None
        staging_value = None
        staging_grant = hold_grant or f"{b.prefix}.rebase_grant"
        if staged:
            staging_id = b.work(
                "conv_history_staging", copy_deps, (copy_rows, width),
                read=copy_bytes, write=copy_bytes, family="tensor_copy",
                dtype=dtype,
                metadata=(("materialization_reason",
                           "independent_overlap_staging"),
                          ("source_offset_rows", (source_offset,)),
                          ("dest_offset_rows", (0,)),
                          ("history_rows", copy_rows),
                          ("allocation_bytes", copy_bytes),
                          ("grant", staging_grant),
                          ("temporary", True),
                          ("caller_boundary", "old_reader_may_overlap_destination")))
            staging_value = replace(
                source_view, producer=staging_id, output=0,
                storage=f"{binding['storage']}:staging",
                state_version=contexts[index], binding="static",
                binding_name="")
            copy_deps = (staging_id,)
        rebase_id = b.work(
            "conv_history_rebase", copy_deps, (copy_rows, width),
            read=copy_bytes, write=copy_bytes, family="tensor_copy", dtype=dtype,
            metadata=(("materialization_reason",
                       "committed_history_window_normalize"),
                      ("source_offset_rows", (source_offset,)),
                      ("dest_offset_rows", (0,)),
                      ("history_rows", copy_rows),
                      ("copy_order", ("staged_then_write" if staged
                                      else "descending_overlap_in_place")),
                      ("state_alias", "same_conv_state_buffer"),
                      ("allocation", ("independent_staging" if staged
                                      else "in_place_no_new_slot")),
                      ("staging_bytes", copy_bytes if staged else 0),
                      ("last_use", "before_short_convolution"),
                      ("state_layout", "slots_history_channels"),
                      ("state_dtype", plan.conv_state_dtype),
                      ("caller_boundary", "before_ordinary_q1_update"),
                      ("invocation_class", "vector")))
        copy_value = TensorValue(
            rebase_id, plan.conv_state_dtype, dest_view.layout, output=0,
            storage=binding["storage"], alias_of=(source_value.alias_of
                                                   or binding["storage"]),
            state_version=contexts[index], binding="static", binding_name="")
        copy_binding = bind_state_value(
            copy_value, target=targets_by_view[index], owner=layer_target,
            slot=binding.get("slot"), version=contexts[index],
            ready=rebase_id, grant=hold_grant, generation=contexts[index],
            required=True, representation="history",
            window=(0, copy_rows),
            span=((0, copy_rows), (0, width)),
            readiness="producer_complete",
            last_use="before_short_convolution")
        rebase_outputs[index] = copy_binding
        source_producer = binding["producer"]
        source_ready = binding.get("ready") or source_producer
        staging_accesses = ()
        if staged:
            staging_read = {
                "occurrence": f"{staging_id}:read:{index}",
                "producer": source_producer, "request": str(
                    getattr(views[index], "request_id", index)),
                "stage": b.common["stage"], "layer": plan.index,
                "direction": "read", "position": "state",
                "output": targets_by_view[index],
                "storage": binding["storage"],
                "owner": layer_target, "representation": "history",
                "persistent": False, "temporary": True,
                "state_version": contexts[index],
                "span": ((source_offset, source_offset + copy_rows),
                         (0, width)), "bytes": float(copy_bytes),
                "source": source_ready, "reader": "",
                "grant": staging_grant,
                "decision": None, "last_use": "before_staging_write",
                "rank": None, "scope": "target",
                "generation": contexts[index], "slot": binding.get("slot"),
                "offset": binding.get("offset"),
                "physical_key": f"{binding['storage']}:staging",
                "request_slot": binding.get("slot"), "rows": copy_rows,
                "row_stride": width, "tier": "hbm",
                "source_ready": source_ready,
                "grant_ready": staging_grant,
                "decision_event": None,
                "last_use_event": "before_staging_write",
            }
            staging_write = {
                **staging_read,
                "occurrence": f"{staging_id}:write:{index}",
                "producer": staging_id, "direction": "write",
                "position": "state", "temporary": True,
                "storage": f"{binding['storage']}:staging",
                "source": staging_id, "decision": "staging_complete",
                "last_use": "before_rebase_write",
            }
            staging_accesses = (staging_read, staging_write)
            for position, staged_op in enumerate(b.ops):
                if staged_op.op_id == staging_id:
                    b.ops[position] = replace(
                        staged_op,
                        metadata=staged_op.metadata + (
                            ("state_accesses", staging_accesses),))
                    break
        rebase_read = {
            "occurrence": f"{rebase_id}:read:{index}",
            "producer": (staging_id if staged else source_producer),
            "request": str(getattr(views[index], "request_id", index)),
            "stage": b.common["stage"], "layer": plan.index,
            "direction": "read", "position": "state",
            "output": targets_by_view[index],
            "storage": (f"{binding['storage']}:staging" if staged
                        else binding["storage"]),
            "owner": layer_target, "representation": "history",
            "persistent": True, "temporary": False,
            "state_version": contexts[index],
            "span": ((source_offset, source_offset + copy_rows), (0, width)),
            "bytes": float(copy_bytes),
            "source": (staging_id if staged else source_ready),
            "reader": "", "grant": hold_grant, "decision": None,
            "last_use": "before_short_convolution", "rank": None,
            "scope": "target", "generation": contexts[index],
            "slot": binding.get("slot"), "offset": binding.get("offset"),
            "physical_key": binding["storage"],
            "request_slot": binding.get("slot"), "rows": copy_rows,
            "row_stride": width, "tier": "hbm",
            "source_ready": (staging_id if staged else source_ready),
            "grant_ready": hold_grant, "decision_event": None,
            "last_use_event": "before_short_convolution",
        }
        rebase_write = {
            **rebase_read,
            "occurrence": f"{rebase_id}:write:{index}",
            "producer": rebase_id, "direction": "write",
            "storage": binding["storage"], "source": rebase_id,
            "span": ((0, copy_rows), (0, width)),
            "window": (0, copy_rows),
            "decision": "copy_complete",
            "last_use": "before_short_convolution",
        }
        b.ops[-1] = replace(b.ops[-1], metadata=b.ops[-1].metadata + (
            ("state_accesses", (*staging_accesses, rebase_read)),
            ("state_accesses_write", (rebase_write,)),
            ("state_ready", tuple(state_ready)),
            ("reader_hold", hold_event),
            ("grant", hold_grant),
            ("protected_rows", protected),
            ("copy_source_value", source_view.to_dict()),
            ("copy_dest_value", dest_view.to_dict()),
        ))
        rebase_records.append({
            "sequence": index,
            "request": str(getattr(views[index], "request_id", index)),
            "accepted_tokens": accepted,
            "source_context": contexts[index],
            "source_version": contexts[index],
            "dest_version": contexts[index],
            "source_offset": source_offset,
            "dest_offset": 0,
            "rows": copy_rows,
            "bytes": copy_bytes,
            "source_window": (source_offset, source_offset + copy_rows),
            "dest_window": (0, copy_rows),
            "storage": binding["storage"],
            "slot": binding.get("slot"),
            "copy_op": rebase_id,
            "copy_output": copy_value.to_dict(),
            "staging_op": staging_id,
            "staging_bytes": copy_bytes if staged else 0,
            "order": ("staged_then_write" if staged
                      else "descending_overlap_in_place"),
            "allocation": ("independent_staging" if staged
                           else "in_place_no_new_slot"),
            "last_use": "before_short_convolution",
            "reader_hold": hold_event,
            "grant": hold_grant,
            "held_rows": hold_window,
            "protected_rows": protected,
        })
        rebase_ids = (*rebase_ids, rebase_id)
    contract, effects, state_read, state_write = _state_contract(
        views=views, q_lens=q_lens, padding_slots=padding_slots, width=width,
        byte=byte, kernel_width=plan.conv_kernel, history_slots=history_slots,
        speculative=speculative, layer=plan.index, bindings=bindings,
        history_acceptance=history_acceptance, reader_holds=reader_holds,
        slots=slots, targets=targets_by_view, rebase_outputs=rebase_outputs,
        state_decisions=state_decisions)
    contract = {**contract, "state_dtype": plan.conv_state_dtype,
                "conv_width": width, "state_ready": tuple(state_ready),
                "state_grants": tuple(state_grants),
                "slots": tuple(slots),
                "reader_holds": tuple(reader_holds)}
    by_id = {op.op_id: op for op in b.ops}
    producers = tuple(dict.fromkeys(sources))
    if any(dict(by_id[name].metadata).get("output_format", dtype) != dtype
           for name in producers):
        raise ModelingError("causal_conv_update_projection_dtype_mismatch")
    if len(producers) == 1:
        projection = by_id[producers[0]]
        slices = dict(projection.metadata).get("packed_projection_slices", ())
        qkv = [item for item in slices if item["role"] in {"q", "k", "v"}]
        if (tuple(item["role"] for item in qkv) != ("q", "k", "v")
                or tuple(item["offset_elements"] for item in qkv) != (0, key_width, 2 * key_width)):
            raise ModelingError("causal_conv_update_requires_declared_contiguous_qkv_slice")
        value = tensors.result(producers[0], (tokens, projection.local_shape[1]), dtype)
        value = value.narrow(0, 0, rows).narrow(1, 0, width)
        incoming = tensors.contiguous(value, "convolution_input_contiguous",
                                      reason="CausalConv1d.x.AutoContiguous").producer
        adaptation = "packed_projection_qkv_slice; AutoContiguous_if_strided"
    else:
        if len(sources) != 3 or len(producers) != 3:
            raise ModelingError("causal_conv_update_requires_three_qkv_producers_or_one_packed")
        incoming = b.work(
            "convolution_input_pack", producers, (rows, width),
            read=rows * width * byte, write=rows * width * byte,
            family="tensor_copy", dtype=dtype,
            metadata=(("materialization_reason", "rowwise_qkv_concatenation"),
                      ("source_row_widths", (key_width, key_width, value_width)),
                      ("caller_boundary", "declared_separate_producer_adapter_not_native_projection_claim"),
                      ("invocation_class", "vector")))
        adaptation = "explicit_row_pack_of_separate_semantic_producers"
    num_accepted_values = history_acceptance if speculative else None
    metadata = (
        ("sequence_tokens", q_lens), ("physical_sequence_tokens", q_lens),
        ("padding_slots", padding_slots),
        ("kernel_width", plan.conv_kernel),
        ("speculative", speculative), ("state_history_slots", history_slots),
        ("state_dtype", plan.conv_state_dtype), ("state_layout", "slots_history_channels"),
        ("state_mutation", "inplace_history_writeback"),
        ("state_read_bytes", state_read), ("state_write_bytes", state_write),
        ("state_read_versions", contract["read_versions"]),
        ("state_commit_versions", contract["commit_versions"]),
        ("state_checkpoint_versions", contract["checkpoint_versions"]),
        ("conv_state_contract", contract), ("conv_state_effects", effects),
        ("state_bindings", tuple(
            binding_record(binding) for binding in bindings if binding)),
        ("num_accepted_tokens_present", speculative),
        ("num_accepted_tokens_dtype", "int32"),
        ("num_accepted_tokens_shape", (len(views),)),
        ("num_accepted_tokens_values", tuple(num_accepted_values or ())),
        ("state_token_offsets", tuple(
            max(0, value - 1) for value in (num_accepted_values or ()))),
        ("history_acceptance_basis",
         "caller_declared" if history_acceptance is not None else "absent"),
        ("has_initial_state", True), ("has_conv_states", bool(active)),
        ("cache_index_dtype", "int32"), ("cache_indices", "one_valid_slot_per_active_sequence"),
        ("query_start_loc_dtype", "int32"), ("bias", False),
        ("activation", "silu"), ("run_mode", 1), ("source", SOURCE),
        ("speculative_source_selector",
         "num_accepted_tokens_presence_width4" if speculative else "none"),
        ("history_rebase", tuple(rebase_records)),
        ("history_rebase_bytes", sum(record["bytes"]
                                     for record in rebase_records)),
        ("history_rebase_op", rebase_ids[0] if rebase_ids else None),
        ("state_ready", tuple(state_ready)),
        ("state_slots", tuple(slots)),
        ("state_decisions", tuple(state_decisions)),
        ("state_grants", tuple(state_grants)),
        ("reader_holds", tuple(reader_holds)),
        ("input_adaptation", adaptation), ("prepared_weight_dtype", dtype),
        ("prepared_weight_layout", "kernel_width_channels"),
        ("native_kind", "CausalConv1d"), ("invocation_class", "vector"),
        ("implementation", IMPLEMENTATION),
        ("limitations", ("scalar_metadata_issue_and_cache_latency_unmodeled",
                          "source_update_kernel_revision_not_capture_binary_identity",
                          "same_shape_owner_runs_use_existing_fluid_compaction",
                          "speculative_history_requires_declared_state_capacity")),
    )
    op_id = b.work("short_convolution",
                   (incoming, *state_ready, *rebase_ids), (rows, width),
                   family="short_convolution", dtype=dtype, metadata=metadata)
    op = b.ops[-1]
    b.ops[-1] = replace(
        op, recipe=CONV, source_boundary="kernel",
        metadata=op.metadata + (("state_accesses", state_accesses(op)),))
    return op_id
