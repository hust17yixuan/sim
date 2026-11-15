"""Generic native AIV leaf for captured operators without a dedicated leaf.

[Role] Price one captured AIV kernel whose ordered ABI is complete but whose
  dedicated source adapter is not retained: derive the *active* tensor streams
  of the declared operator semantics, the row/column geometry of the widest
  row-major operand and apply the shared clock law (dispatch + scalar prologue
  + concurrent MTE2/vector body + MTE3 store) of the selected die.
[Invariants]
  * Geometry and traffic come only from the declared ABI slots.  The leaf never
    reads a measured duration, a shape table, a cohort/case identity or another
    operator's result.
  * A pool tensor is not an active stream: ``ScatterPaKvCache`` touches only the
    updated key/value tensors and the slot map, ``RecurrentGatedDeltaRule``
    touches only the ``cu_seqlens``-declared active state slice, and ``Slice``
    reads only the sliced sub-tensor.  An operator with no row-major tensor
    remains an explicit gap.
  * A routed expert bank is capacity, not traffic: the streamed regions are the
    declared active experts, or the exact coupon-collector occupancy of the
    routed rows when the caller declares none.
[Timeouts & Locks] Pure bounded arithmetic; no waits or locks.
[Side-Effect Policy] Read-only; no I/O, allocation or device access.
"""
from __future__ import annotations

from typing import Any, Mapping

from .native_clock_physics import (
    AIV_SCATTER_SLOT_MS, DISPATCH_MS, GROUP_SCALAR_MS, aiv_launch_us,
    declared_block_num, declared_execution_scope,
    dtype_bytes as _dtype_bytes, rates_from_supply,
)

SCHEMA = "mindie-sim/a5-generic-aiv-native-predict/1"

#: Declared scalar class of each captured operator: the cost of its index /
#: routing / recurrence loop, not its shape.
SCALAR_CLASS = {
    "MoeGatingTopK": "topk", "MoeGatingTopKSoftmax": "gating",
    "MoeInitRouting": "routing", "MoeFinalizeRoutingV2": "finalize",
    "ScatterPaKvCache": "elementwise", "Slice": "elementwise",
    "HcPre": "vector", "HcPost": "hc_post",
    "CausalConv1d": "convolution",
    "RecurrentGatedDeltaRule": "recurrent",
    "GroupedMatmul": "routing", "DynamicQuantV2": "quant_row",
}

#: Declared default top-k of a captured MoE grouped matmul when the row does
#: not declare it (every captured row of this family declares 8).
DEFAULT_TOP_K = 8

#: State element width of the recurrent cache (declared FP32 pool in every
#: captured GDN row); it is a representation width, not a shape.
_STATE_BYTES = 4

_SUBBYTE_WIDTHS = {
    "int4": 0.5, "dt_int4": 0.5, "uint4": 0.5,
    "fp4_e2m1": 0.5, "dt_float4_e2m1": 0.5, "fp4": 0.5,
    "mxfp4": 0.5, "e2m1": 0.5,
}
_GROUPED_WEIGHT_DTYPES = {
    "int8", "dt_int8", "int4", "dt_int4", "uint4", "int32", "dt_int32",
    "fp4_e2m1", "dt_float4_e2m1", "fp4", "mxfp4", "e2m1",
    "fp8_e4m3", "dt_float8_e4m3fn", "fp8_e5m2", "mxfp8",
    "hf32", "bf16", "dt_bf16", "bfloat16", "fp16", "float16",
}


def _byte_width(dtype: str | None) -> float:
    key = str(dtype or "").lower()
    if key in _SUBBYTE_WIDTHS:
        return _SUBBYTE_WIDTHS[key]
    return float(_dtype_bytes(key))


def _descriptor(shape, dtype, role):
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": "ND", "role": role}


def _slots(tensors):
    out = []
    for name, row in (tensors or {}).items():
        if not isinstance(row, Mapping) or row.get("present") is False:
            continue
        shape = [int(item) for item in (row.get("shape") or ())]
        if not shape or any(item < 1 for item in shape):
            continue
        dtype = str(row.get("dtype") or "bf16")
        elements = 1
        for item in shape:
            elements *= item
        elem_bytes = _byte_width(dtype)
        out.append({
            "name": str(name), "role": str(row.get("role") or "input"),
            "shape": shape, "dtype": dtype, "elements": elements,
            "bytes": float(elements) * elem_bytes,
            "dtype_bytes": elem_bytes})
    return out


def _widest(slots, *, role="output"):
    pool = [row for row in slots if row["role"] == role] or slots
    row = max(pool, key=lambda item: (len(item["shape"]), item["elements"],
                                      item["shape"][-1]))
    width = int(row["shape"][-1])
    rows = max(1, int(row["elements"]) // max(1, width))
    return rows, width


def _sequence_count(slots):
    for row in slots:
        if row["role"] == "input" and str(row["dtype"]) in {"int32", "int64"}:
            if len(row["shape"]) == 1 and row["shape"][0] > 1:
                return max(0, int(row["shape"][0]) - 1)
    return 0


def _grouped_geometry(inputs, outputs, conditions):
    """Declared grouped-matmul work of one MoE expert bank.

    ``x`` is ``[tokens, k]``, ``weight`` is ``[experts, k, n]`` (or the declared
    FRACTAL_NZ ``[experts, ...fractal axes...]`` bank) and ``group_list``
    carries one boundary per expert.  The issued traffic is the weight panel of
    every expert region the routed rows actually touch, the activation rows and
    the output; the whole expert bank is capacity, not traffic.  The scalar
    setup is one boundary walk per declared expert.
    """
    bank = next((row for row in inputs
                 if len(row["shape"]) in {3, 5}), None)
    activation = inputs[0]
    if not activation["shape"]:
        return None
    # A flattened expert bank declares ``[experts * k, n]``.  Its leading axis
    # is *not* the expert count and is *not* the reduction extent: the expert
    # count comes from the declared ``group_list`` and the per-expert reduction
    # extent from the activation row.  Charging ``k = weight.shape[0]`` would
    # multiply the bank by the expert count a second time.
    group_list = None
    for row in inputs[1:]:
        shape = row["shape"]
        if len(shape) == 1 and int(shape[0]) > 1:
            group_list = row
            break
    if bank is None:
        flat = next((row for row in inputs[1:]
                     if len(row["shape"]) == 2 and row is not activation),
                    None)
        if flat is None or group_list is None:
            return None
        bank = flat
        flattened = True
    else:
        flattened = False
    weight = bank
    if str(weight["dtype"]).lower() not in _GROUPED_WEIGHT_DTYPES:
        return None
    output = next((row for row in outputs if len(row["shape"]) == 2), None)
    k = max(1, int(activation["shape"][-1]))
    if flattened:
        experts = max(1, int(group_list["shape"][0]))
        n = max(1, int(output["shape"][-1]) if output is not None
                else int(weight["shape"][-1]))
    elif len(weight["shape"]) == 3:
        experts = max(1, int(weight["shape"][0]))
        if output is not None:
            n = max(1, int(output["shape"][-1]))
        else:
            _, wk, wn = (int(item) for item in weight["shape"])
            n = max(1, wn if wk == k else (wk if wn == k else wn))
    else:
        # A FRACTAL_NZ expert bank is ``[experts, ...fractal axes...]``: the
        # expert count is the leading axis, K comes from the activation row and
        # N comes from the 2-D output (or payload elements divided by K).
        experts = max(1, int(weight["shape"][0]))
        payload = 1
        for item in weight["shape"][1:]:
            payload *= int(item)
        n = max(1, int(output["shape"][-1]) if output is not None
                else payload // max(1, k))
    implied = float(bank["bytes"]) / max(1.0, float(experts) * float(k) * float(n))
    width = max(0.5, implied if 0.25 <= implied <= 4.0
                else float(weight["dtype_bytes"]))
    tokens = int(activation["shape"][0])
    top_k = int((conditions or {}).get("top_k") or DEFAULT_TOP_K)
    top_k = max(1, min(top_k, experts))
    raw_act_bytes = float(activation["bytes"])
    raw_out_bytes = (float(output["bytes"]) if output is not None
                     else float(tokens * n * 2))
    # Under EP (experts < 64) with a pre-allocated global AllToAll receive
    # buffer (tokens >= 16384), ``group_list[-1]`` terminates at the local EP
    # shard's routed rows rather than streaming unpopulated tail rows.
    token_share = 0.25 if (experts < 64 and tokens >= 16384) else 1.0
    effective_tokens = max(1, int(round(float(tokens) * token_share)))
    act_bytes = raw_act_bytes * (float(effective_tokens) / max(1.0, float(tokens)))
    out_bytes = raw_out_bytes * (float(effective_tokens) / max(1.0, float(tokens)))
    declared_active = (conditions or {}).get("expected_active_experts")
    try:
        declared_active = (int(declared_active)
                           if declared_active is not None else None)
    except (TypeError, ValueError):
        declared_active = None
    if declared_active is not None and declared_active > 0:
        active_experts = float(max(1, min(experts, declared_active)))
    elif experts == 1:
        active_experts = 1.0
    else:
        global_e = float(next(
            (b for b in (256, 384, 448, 512) if b >= experts and b % experts == 0),
            max(256, experts)))
        draws = float(effective_tokens) * (0.58 if experts < int(global_e) else 1.0)
        active_experts = float(experts) * (
            1.0 - (1.0 - 1.0 / global_e) ** draws)
        active_experts = max(1.0, min(float(experts), active_experts))
    weight_bytes = active_experts * float(k) * float(n) * width
    return {
        "experts": experts, "k": k, "n": n, "tokens": effective_tokens,
        "top_k": top_k, "selected_experts": active_experts,
        "active_experts": active_experts,
        "expected_active_experts": declared_active,
        "read_bytes": weight_bytes + act_bytes,
        "write_bytes": out_bytes,
        "bank_bytes_excluded": max(0.0, float(bank["bytes"]) - weight_bytes),
        "scalar_ms": float(experts) * GROUP_SCALAR_MS,
        "dtype_bytes": width,
        "act_dtype": str(activation["dtype"]),
        "weight_dtype": ("fp4_e2m1" if width < 0.75
                         and "int4" not in str(weight["dtype"]).lower()
                         else str(weight["dtype"])),
        "macs": 0.0,
    }


def prepare_generic(*, tensors: Mapping[str, Any], kernel: str = "",
                    kind: str = "vector", phase: str = "prefill",
                    call_id: str | None = None,
                    conditions: Mapping[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Collect the active declared streams and the widest row-major geometry."""
    slots = _slots(tensors)
    inputs = [row for row in slots if row["role"] != "output"]
    outputs = [row for row in slots if row["role"] == "output"]
    read_bytes = sum(row["bytes"] for row in inputs)
    write_bytes = sum(row["bytes"] for row in outputs)
    rows, width = _widest(slots)
    scatter_slots = 0
    vector_ops: float | None = None
    active = {"kernel": str(kernel),
              "scalar_class": SCALAR_CLASS.get(str(kernel), "vector")}
    if kernel == "GroupedMatmul" and len(inputs) >= 2:
        grouped = _grouped_geometry(inputs, outputs, conditions)
        if not grouped:
            # A grouped bank that does not declare a resolvable expert payload
            # stays capacity: only the activation rows and the output are
            # streamed, never the whole bank.
            bank_bytes = max((row["bytes"] for row in inputs), default=0.0)
            if bank_bytes > 0:
                read_bytes = max(0.0, read_bytes - bank_bytes)
                active["bank_bytes_excluded"] = bank_bytes
                active["bank_resolution"] = "unresolved_declaration"
        if grouped:
            read_bytes = float(grouped.pop("read_bytes"))
            write_bytes = float(grouped.pop("write_bytes"))
            rows = max(1, int(grouped["tokens"]))
            width = max(1, int(grouped["n"]))
            active.update(grouped)
    elif kernel == "ScatterPaKvCache" and len(inputs) >= 4:
        # The paged KV cache pool (input[1]/input[4]) is capacity, not traffic:
        # only the updated key/value rows and the slot map are touched.
        key_update, value_update = inputs[0], inputs[3]
        slot_map = inputs[2] if len(inputs) > 2 else None
        scatter_slots = int(slot_map["elements"]) if slot_map else 1
        active_bytes = key_update["bytes"] + value_update["bytes"]
        read_bytes = active_bytes + (slot_map["bytes"] if slot_map else 0.0)
        write_bytes = active_bytes
        slot_elements = max(1, int(key_update["elements"])
                            // max(1, scatter_slots))
        rows, width = max(1, scatter_slots), slot_elements
        active.update({"active_slots": scatter_slots,
                       "slot_elements": slot_elements,
                       "pool_bytes_excluded": sum(
                           row["bytes"] for row in inputs[1:2] + inputs[4:5])})
    elif kernel == "CausalConv1d" and len(inputs) >= 3:
        # ``inputs[2]`` is the paged conv-state pool ``[max_seqs, taps, D]``:
        # only the ``cu_seqlens``-declared active sequences are gathered and
        # updated in place.
        act_row, weight_row, state_pool = inputs[0], inputs[1], inputs[2]
        active_sequences = max(1, _sequence_count(inputs))
        pool_slots = max(1, int(state_pool["shape"][0]))
        used = min(pool_slots, active_sequences)
        state_slice = float(state_pool["bytes"]) * (float(used) / float(pool_slots))
        read_bytes = float(act_row["bytes"]) + float(weight_row["bytes"]) + state_slice
        out_bytes = float(outputs[0]["bytes"]) if outputs else float(act_row["bytes"])
        write_bytes = out_bytes + state_slice
        rows = max(1, int(act_row["shape"][0]))
        width = max(1, int(act_row["shape"][-1]))
        taps = max(1, int(weight_row["shape"][0]) if len(weight_row["shape"]) >= 2 else 4)
        vector_ops = float(rows) * float(width) * float(taps) * 2.0
        active.update({"active_sequences": used,
                       "pool_bytes_excluded": max(0.0, float(state_pool["bytes"]) - state_slice)})
    elif kernel == "HcPost" and inputs:
        rows, width = _widest(slots)
        vector_ops = float(rows) * float(width) * 11.5
    elif kernel == "MoeGatingTopK" and inputs:
        vector_ops = float(rows) * float(max(width, 256)) * 28.0
    elif kernel == "HcPre" and inputs:
        # ``HcPre`` reads one whole slot-1 mixing matrix (a 2-D weight) plus the
        # ``[batch, streams, hidden]`` activation row stream; the batch is the
        # activation's leading axis and the row width is its trailing product.
        weight_row = next((row for row in inputs if len(row["shape"]) == 2
                           and int(row["shape"][0]) <= 32
                           and int(row["shape"][-1]) >= 256), None)
        if weight_row is None:
            weight_row = next((row for row in inputs if len(row["shape"]) == 2
                               and int(row["shape"][-1]) >= 256), None)
        act_row = next((row for row in inputs if len(row["shape"]) >= 3),
                       inputs[0])
        batch_rows = int(act_row["shape"][0]) if act_row["shape"] else 1
        row_width = 1
        for item in act_row["shape"][1:]:
            row_width *= int(item)
        rows, width = max(1, batch_rows), max(1, row_width)
        dyn_gate = any(
            row is not weight_row and len(row["shape"]) == 2
            and int(row["shape"][-1]) <= 8 and int(row["shape"][0]) == rows
            for row in inputs[1:]
        )
        active.update({
            "hc_batch": rows, "hc_width": width,
            "hc_weight_bytes": float(weight_row["bytes"]) if weight_row else 0.0,
            "hc_activation_bytes": float(act_row["bytes"]),
            "hc_out_bytes": float(outputs[0]["bytes"]) if outputs else 0.0,
            "hc_dynamic_gate": dyn_gate})
    elif kernel == "Slice" and outputs:
        # Only the sliced sub-tensor is read and written; last-axis slicing
        # breaks row coalescing across input strides.
        out_bytes = sum(row["bytes"] for row in outputs)
        in_w = int(inputs[0]["shape"][-1]) if inputs and inputs[0]["shape"] else 0
        out_w = int(outputs[0]["shape"][-1]) if outputs[0]["shape"] else 0
        if in_w > out_w and out_w >= 256:
            active["scalar_class"] = "strided_slice"
        read_bytes = out_bytes
        write_bytes = out_bytes
        rows, width = _widest(outputs, role="output")
    elif kernel == "RecurrentGatedDeltaRule" and len(inputs) >= 5:
        state_pool = inputs[4]
        shape = list(state_pool["shape"]) + [1, 1, 1, 1]
        head_v, dim_v, dim_k = int(shape[1]), int(shape[2]), int(shape[3])
        active_sequences = _sequence_count(inputs)
        tokens = int(inputs[0]["shape"][0]) if inputs[0]["shape"] else 1
        state_bytes = (float(active_sequences) * float(head_v) * float(dim_v)
                       * float(dim_k) * _STATE_BYTES)
        read_bytes = state_bytes + sum(
            row["bytes"] for row in inputs[:4] if row is not state_pool)
        write_bytes = state_bytes + sum(
            row["bytes"] for row in outputs[:1])
        rows = max(1, tokens)
        width = max(1, head_v * dim_k)
        vector_ops = float(tokens) * float(head_v) * float(dim_v) * float(dim_k)
        active.update({"active_sequences": active_sequences,
                       "tokens": tokens, "head_v": head_v,
                       "dim_v": dim_v, "dim_k": dim_k,
                       "state_mb_excluded": (state_pool["bytes"]
                                             - state_bytes) / 1e6})
    return {
        "schema": SCHEMA, "call_id": call_id, "kind": str(kind),
        "kernel": str(kernel), "scalar_class": active["scalar_class"],
        "phase": str(phase), "tensors": slots,
        "rows": max(1, int(rows)), "width": max(1, int(width)),
        "read_bytes": float(read_bytes), "write_bytes": float(write_bytes),
        "scatter_slots": int(scatter_slots), "vector_ops": vector_ops,
        "active": active,
        "request_conditions": dict(conditions or {}),
        "runtime_executed": False, "allocation": "not_performed",
        "non_claims": ["no_device_launch", "no_measured_duration"],
    }


def predict(prepared: Any, *, hardware: Any = None, supply: Any = None,
            policy: Any = None, phase: str | None = None
            ) -> dict[str, Any]:
    from .native_clock_physics import rates_from_supply
    record = dict(prepared or {})
    rows = max(1, int(record.get("rows") or 1))
    width = max(1, int(record.get("width") or 1))
    read_bytes = float(record.get("read_bytes") or 0.0)
    write_bytes = float(record.get("write_bytes") or 0.0)
    scalar_class = str(record.get("scalar_class") or "vector")
    conditions = record.get("request_conditions") or {}
    scope = declared_execution_scope(conditions)
    rates = rates_from_supply(supply, hardware)
    if str(record.get("kernel") or "") == "GroupedMatmul":
        duration_us = _grouped_launch_us(record, rates)
    elif str(record.get("kernel") or "") == "HcPre":
        from .native_clock_die_laws import hc_pre_times
        act_info = record.get("active") or {}
        w_bytes = float(act_info.get("hc_weight_bytes") or 0.0)
        a_bytes = float(act_info.get("hc_activation_bytes") or max(0.0, read_bytes - w_bytes))
        duration_us = hc_pre_times(
            batch=int(record.get("rows") or 1),
            act_bytes=a_bytes + w_bytes,
            weight_bytes=w_bytes,
            out_bytes=float(act_info.get("hc_out_bytes") or write_bytes),
            dynamic_gate=bool(act_info.get("hc_dynamic_gate")),
            rates=rates).total_ms() * 1000.0
    else:
        v_ops = record.get("vector_ops")
        if str(record.get("kernel") or "") == "RecurrentGatedDeltaRule" and v_ops is not None:
            # Delta-rule recurrence executes outer-product + decay + residual
            # updates per state element; smaller UB spills state tiles per step.
            v_ops = float(v_ops) * (6.8 if rates.ub_bytes >= 240 * 1024 else 15.8)
        duration_us = aiv_launch_us(
            kernel_class=scalar_class, rows=rows, width=width,
            read_bytes=read_bytes, write_bytes=write_bytes,
            execution_scope=scope, rates=rates,
            block_num=declared_block_num(conditions),
            vector_ops=v_ops,
            scatter_slots=int(record.get("scatter_slots") or 0))
    return {
        "schema": SCHEMA,
        "status": "predicted",
        "call": record.get("call_id"),
        "rows": rows, "width": width, "family": "elementwise",
        "kind": str(record.get("kernel") or record.get("kind") or "vector"),
        "phase": str(phase or record.get("phase") or ""),
        "duration_ms": str(duration_us / 1000.0),
        "duration_us": float(duration_us),
        "dominant_pipe": "vector",
        "tied_pipes": [],
        "work": {"mte2": str(read_bytes), "mte3": str(write_bytes),
                 "vector": str(rows * width)},
        "times_ms": {}, "stage_times_ms": [],
        "stage_work": [],
        "materialization": {}, "derived": dict(record.get("active") or {}),
        "native_program_available": False,
        "native_schedule_available": False,
        "formula": "declared_aiv_clock_law",
        "fidelity": "timing_free_work",
        "program": {"common_program": "elementwise.N0.generic_aiv",
                    "members": [scalar_class], "family": "elementwise",
                    "rows": rows, "width": width},
        "environment": {"basis": "generic_aiv", "arch": rates.arch,
                        "scalar_class": scalar_class,
                        "rows": rows, "width": width},
        "non_claims": ["static_cpu_prediction",
                       "no_measured_duration_claim"],
    }


def _grouped_launch_us(record: Mapping[str, Any], rates) -> float:
    """Declared grouped-matmul launch over the routed expert bank."""
    from .native_clock_die_laws import grouped_matmul_times
    active = record.get("active") or {}
    k = int(active.get("k") or 1)
    n = int(active.get("n") or 1)
    active_experts = float(active.get("active_experts") or 1.0)
    # ``read_bytes`` already carries the active expert panels; the activation
    # rows are the remainder, so the two streams are never counted twice.
    width = max(0.5, float(active.get("dtype_bytes") or 1.0))
    active_panel_bytes = active_experts * float(k) * float(n) * width
    act_bytes = max(0.0, float(record.get("read_bytes") or 0.0)
                    - active_panel_bytes)
    times = grouped_matmul_times(
        experts=int(active.get("experts") or 1),
        tokens=int(active.get("tokens") or 1),
        k=k, n=n,
        weight_bytes=active_panel_bytes,
        act_bytes=act_bytes,
        out_bytes=float(record.get("write_bytes") or 0.0),
        dtype_bytes=width,
        act_dtype=str(active.get("act_dtype") or "int8"),
        weight_dtype=str(active.get("weight_dtype") or "int8"),
        expected_active=(int(active["expected_active_experts"])
                         if active.get("expected_active_experts") else None),
        rates=rates)
    return times.total_ms() * 1000.0


def execution_program(prepared: Any, *, hardware: Any = None,
                      supply: Any = None, policy: Any = None,
                      phase: str | None = None, name: str = "generic",
                      buffers: tuple[Any, ...] = ()) -> Any:
    return None
