"""Selected v4 count routes used by the legacy shape-oriented flow API.

[Role]
Bind explicit selected-route assumptions to the exact count algebra. Timing
remains an uncalibrated service estimate, distinct from a source certification.
[Invariants]
Semantic work, executed traffic, unique payload, and physical allocation are
different counters. Irregular schedules retain their task/sequence assumptions.
FP16 replay U must be promoted explicitly before the FP32 commit ABI.
[Timeouts & Locks]
Bounded arithmetic and packaged read-only math records; no waits or locks.
[Side-Effect Policy]
Construct immutable records only; no device or repository writes.
"""

from .catalog_common import check_kwargs, pick_int
from .exact_math import ceildiv
from .flow_schema import (AuthoringBackend, BufferPool, BufferSpec,
                          ExecutionArchetype, OpFlowTemplate, OwnerWork, PipeWork)
from .v4_math import evaluate_math, operator_record


def _vector_flow(name, bindings, params, *, items, tile, steps=1, read_key="actual_read_bytes",
                 work_key="ops_vec", owner_work=()):
    result = evaluate_math(name, bindings)
    counts = result["counts"]
    buffers = tuple(BufferSpec(b["name"], BufferPool.UB_LANE0 if declared["pool"] == "UB"
                               else BufferPool(declared["pool"]), b["raw_bytes"], slots=b["slots"])
                    for b, declared in zip(result["buffers"], operator_record(name)["sram_buffers"]))
    params = dict(params, exact_counts=counts, count_bindings=bindings,
                  counter_scope="whole_operator", timing_confidence="analytic_prior")
    return OpFlowTemplate(
        name=name, archetype=ExecutionArchetype.PURE_VECTOR,
        total_units=items, base_tile_units=tile, live_bytes_per_unit=0,
        pipe_work=(PipeWork("VEC", counts[work_key]),),
        gm_read_bytes=counts[read_key], gm_write_bytes=counts["actual_write_bytes"],
        unique_gm_bytes=counts["logical_read_bytes"] + counts["logical_write_bytes"],
        buffers=buffers, params=params, recurrence_steps=steps, owner_work=owner_work,
        authoring_backend=AuthoringBackend.CANNBOT_DSL,
        dtype="fp32", no_launch=items == 0, estimate_kind="analytic_prior",
        notes="v4 selected count route; tokens=0 AttnRes is no-launch; semantic Vector ops use an uncalibrated scalar rate",
    )


def attnres(**kw):
    check_kwargs(kw, {"tile", "active_owners", "row_tile", "depth"})
    t = pick_int(kw, ("tokens", "rows"), 128, minimum=0)
    d = pick_int(kw, ("hidden", "cols"), 7168)
    u = pick_int(kw, ("active_owners",), min(t, 64), minimum=0)
    b = dict(T=t, D=d, U=u, t=pick_int(kw, ("row_tile",), 1),
             depth=pick_int(kw, ("depth",), 2))
    return _vector_flow("block_attn_res_update_rms_norm", b,
                        {"tokens": t, "hidden": d, "active_owners": u}, items=t, tile=1)


def engram(**kw):
    check_kwargs(kw, {"hc_mult", "tile", "has_mask", "unmasked", "weight_loads"})
    t, d, h = (pick_int(kw, names, default) for names, default in (
        (("tokens", "rows"), 32), (("hidden", "cols"), 5120), (("hc_mult",), 4)))
    mask = kw.get("has_mask", False)
    if not isinstance(mask, bool):
        raise ValueError("has_mask must be bool")
    if mask and "unmasked" not in kw:
        raise ValueError("masked Engram requires the unmasked token count")
    au = pick_int(kw, ("unmasked",), t, minimum=0)
    streaming = ceildiv(d, 64) * 64 > 8448
    b = dict(T=t, h=h, D=d, Au=au,
             weight_loads=pick_int(kw, ("weight_loads",), au * h, minimum=0),
             mask_service=t * h if mask else 0,
             mask_service_streaming=t * h if mask else 0, chunk=min(4096, ceildiv(d, 64) * 64))
    flow = _vector_flow("engram_gate", b,
                        {"tokens": t, "hidden": d, "hc_mult": h,
                         "passes": 2 if streaming else 1, "has_mask": mask},
                        items=t * h, tile=1,
                        read_key="actual_read_bytes_streaming" if streaming else "actual_read_bytes")
    # The math record exposes both alternatives; only one is simultaneously live.
    from dataclasses import replace
    wanted = "stream" if streaming else "resident"
    return replace(flow, buffers=tuple(x for x in flow.buffers if wanted in x.name))


def replay(**kw):
    check_kwargs(kw, {"batch", "seq", "partitions", "active_owners", "control_bytes"})
    b = pick_int(kw, ("batch", "tokens", "rows"), 2)
    s = pick_int(kw, ("seq",), 8)
    n, d = pick_int(kw, ("heads",), 6), pick_int(kw, ("dim", "d"), 128)
    p = pick_int(kw, ("partitions",), 1)
    if p not in (1, 2, 4):
        raise ValueError("ReplaySSM partitions must be 1, 2, or 4")
    u = pick_int(kw, ("active_owners",), min(b * n * p, 64))
    bindings = dict(B=b, N=n, D=d, L=b*s, PLH=p*n*b*s, U=u, r=d//p, v=max(d//p, 64),
                    control=pick_int(kw, ("control_bytes",), 4*b*n*p, minimum=0), metadata=4*(b+1))
    return _vector_flow("fused_recurrent_kda_replayssm", bindings,
                        {"batch": b, "seq": s, "heads": n, "dim": d,
                         "partitions": p, "U_dtype": "fp16", "checkpoint_read_only": True},
                        items=b*n*p, tile=1, steps=s)


def commit(**kw):
    check_kwargs(kw, {"batch", "blocks", "accepted_tokens", "accepted", "partitions", "u_dtype"})
    b, n, d = (pick_int(kw, names, default) for names, default in (
        (("batch", "blocks"), 2), (("heads",), 6), (("dim",), 128)))
    accepted = kw.get("accepted_tokens", kw.get("accepted", 4))
    values = tuple(accepted) if isinstance(accepted, (tuple, list)) else (accepted,) * b
    if len(values) != b or any(type(x) is not int or x < 0 for x in values):
        raise ValueError("accepted tokens must be a nonnegative integer per batch")
    if kw.get("u_dtype", "fp32") != "fp32":
        raise ValueError("commit U requires FP32; explicitly promote replay FP16 U first")
    p = pick_int(kw, ("partitions",), 1)
    if p not in (1, 2, 4):
        raise ValueError("commit partitions must be 1, 2, or 4")
    a, positive = sum(values), sum(x > 0 for x in values)
    bindings = dict(B=b, Bpos=positive, N=n, D=d, A=a, PAH=p*n*a,
                    tasks=b*n*p, active_items=positive*n*p)
    owners = tuple(OwnerWork(3*x*d*d//p,
                             (4*d*d//p + 4*d*x//p + 6*d*x + 4 + 12*x if x else 0) + 4,
                             4*d*d//p if x else 0, x, n*p) for x in values)
    return _vector_flow("commit_recurrent_kda", bindings,
                        {"batch": b, "heads": n, "dim": d, "accepted_tokens": list(values),
                         "max_steps": max(values), "U_dtype": "fp32",
                         "timing_schedule": "max_sequence_service_bound"},
                        items=b*n*p, tile=1, steps=max(1, max(values)), owner_work=owners)


def add_rms(**kw):
    check_kwargs(kw, {"tile", "row_mode", "weight_loads"})
    r, d = pick_int(kw, ("tokens", "rows", "seq_len"), 32), pick_int(kw, ("hidden", "cols", "n"), 1536)
    mode = kw.get("row_mode", "auto")
    if mode not in ("auto", "row_resident", "row_split"):
        raise ValueError("unknown row mode")
    split = mode == "row_split" or (mode == "auto" and 14*d > 253952)
    b = dict(R=r, D=d, b=2, bw=2, bo=2, res_out=1,
             weight_loads=pick_int(kw, ("weight_loads",), r), c=min(d, 4096))
    flow = _vector_flow("add_rms_norm", b, {"tokens": r, "hidden": d,
                        "row_mode": "row_split" if split else "row_resident"},
                        items=r*d, tile=pick_int(kw, ("tile",), 8192),
                        read_key="actual_read_bytes_row_split" if split else "actual_read_bytes",
                        work_key="ops_vec_row_split" if split else "ops_vec")
    from dataclasses import replace
    return replace(flow, live_bytes_per_unit=6,
                   buffers=tuple(x for x in flow.buffers if ("split" in x.name) == split))


def hcpost(**kw):
    check_kwargs(kw, {"hc_mult", "tile", "dim_tile"})
    t, d, h = (pick_int(kw, names, default) for names, default in (
        (("tokens", "rows"), 32), (("hidden", "cols"), 4096), (("hc_mult",), 4)))
    dt = pick_int(kw, ("dim_tile",), 128)
    return _vector_flow("mhc_hcpost", dict(T=t, h=h, D=d, dt=dt, U=t*ceildiv(d, dt)),
                        {"tokens": t, "hidden": d, "hc_mult": h}, items=t*ceildiv(d, dt), tile=1)


def welford(**kw):
    check_kwargs(kw, {"tile", "two_pass", "partitions"})
    r, d = pick_int(kw, ("tokens", "rows"), 32), pick_int(kw, ("hidden", "cols"), 4096)
    two = kw.get("two_pass", 12*d > 253952)
    if not isinstance(two, bool):
        raise ValueError("two_pass must be bool")
    b = dict(R=r, D=d, P=pick_int(kw, ("partitions",), 1), b=4, bg=4, bb=4,
             bo=4, resident=int(not two), correction=0)
    return _vector_flow("layernorm_welford", b, {"tokens": r, "hidden": d, "two_pass": two},
                        items=r*d, tile=pick_int(kw, ("tile",), 4096))
