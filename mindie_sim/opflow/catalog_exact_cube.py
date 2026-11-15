"""Guarded v4 matmul reference and MXA8W4 selected-route counters.

[Role]
Keep semantic FLOPs separate from issued padded work, output payload from FIX
FP32 input, Stream-K payload from its full workspace, and FP4 storage from MXFP8
Cube execution after per-lane unpacking.
[Invariants]
Independent physical allocations are summed. Implicit MX views share L0 backing
and do not consume SSBUF. Bias has its own 4096-byte pool. Timing is a prior.
[Timeouts & Locks]
Bounded arithmetic over a packaged read-only model; no waits or locks.
[Side-Effect Policy]
Only immutable in-memory flow construction.
"""

from .catalog_common import check_kwargs, pick_int
from .exact_math import align, ceildiv
from .flow_schema import (AuthoringBackend, BufferPool, BufferSpec,
                          ExecutionArchetype, OpFlowTemplate, PipeWork)
from .v4_math import evaluate_math, operator_record


def _buffers(name, result):
    buffers = []
    for value, decl in zip(result["buffers"], operator_record(name)["sram_buffers"]):
        pool = decl["pool"]
        if pool in ("L0A_MX", "L0B_MX"):
            continue  # fixed L0 views, not independent physical allocations
        pools = (BufferPool.UB_LANE0, BufferPool.UB_LANE1) if pool == "UB" and "quant_" in name else (
            BufferPool.UB_LANE0 if pool == "UB" else BufferPool(pool),)
        buffers.extend(BufferSpec(value["name"], p, value["raw_bytes"], slots=value["slots"])
                       for p in pools)
    return tuple(buffers)


def matmul_reference(name, **kw):
    check_kwargs(kw, {"dtype", "depth", "partitions", "tail_tiles", "tail_valid_elems", "tail_write_elems"})
    if kw.get("dtype", "bf16") != "bf16":
        raise ValueError("v4 matmul reference requires BF16")
    m, n, k, tm, tn, tk = (pick_int(kw, names, default) for names, default in (
        (("m", "tokens", "rows"), 2048), (("n", "hidden", "cols"), 2048), (("k",), 2048),
        (("base_m",), 128), (("base_n",), 256), (("base_k",), 64)))
    bindings = dict(M=m, N=n, K=k, tm=tm, tn=tn, tk=tk)
    tiles = ceildiv(m, tm)*ceildiv(n, tn)
    if name == "matmul_streamk":
        p = pick_int(kw, ("partitions",), 1)
        if p > 1 and not {"tail_tiles", "tail_valid_elems", "tail_write_elems"} <= kw.keys():
            raise ValueError("Stream-K partitions require an explicit tail ledger")
        bindings.update(P=p, all_tiles=tiles,
                        **{x: pick_int(kw, (x,), 0, minimum=0)
                           for x in ("tail_tiles", "tail_valid_elems", "tail_write_elems")})
    else:
        bindings["depth"] = pick_int(kw, ("depth",), 2)
    result = evaluate_math(name, bindings)
    c = result["counts"]
    issued = 2*align(m,16)*align(n,16)*align(k,16)
    works = [PipeWork("CUBE", issued//2, "macs"),
             PipeWork("MTE1", c.get("mte1_padded_operand_bytes", c["actual_read_bytes"]), "bytes"),
             PipeWork("FIX", 4*m*n, "bytes")]
    if c["ops_vec"]:
        works.append(PipeWork("VEC", c["issued_ops_vec"]))
    return OpFlowTemplate(name=name, archetype=(ExecutionArchetype.CV_COUPLED_1C2V
                          if c["ops_vec"] else ExecutionArchetype.PURE_CUBE),
                          total_units=tiles, base_tile_units=1, live_bytes_per_unit=0,
                          pipe_work=tuple(works), buffers=_buffers(name, result),
                          gm_read_bytes=c["actual_read_bytes"], gm_write_bytes=c["actual_write_bytes"],
                          unique_gm_bytes=c["logical_read_bytes"] + c["logical_write_bytes"],
                          spill_workspace_bytes=c.get("partial_workspace_capacity", 0),
                          tiling_pool=BufferPool.L1, dtype="bf16", estimate_kind="reference_algorithm",
                          params={"m": m, "n": n, "k": k, "base_m": tm, "base_n": tn, "base_k": tk,
                                  "exact_counts": c, "count_bindings": bindings,
                                  "streamk_workspace_bytes": c.get("partial_workspace_capacity", 0)},
                          notes="v4 selected reference plan; no native schedule certification")


def mxa8w4(**kw):
    check_kwargs(kw, {"dtype", "kl1", "depth_a", "depth_b", "depth_c", "bias"})
    if kw.get("dtype", "mxfp8") != "mxfp8":
        raise ValueError("MXA8W4 storage FP4 is unpacked to MXFP8 before Cube")
    m, n, k, tm, tn, kl0 = (pick_int(kw, names, default) for names, default in (
        (("m", "tokens", "rows"), 2048), (("n", "hidden", "cols"), 2048), (("k",), 2048),
        (("base_m",), 64), (("base_n",), 64), (("base_k",), 256)))
    b = dict(M=m, N=n, K=k, tm=tm, tn=tn, kl0=kl0,
             kl1=pick_int(kw, ("kl1",), kl0), da=pick_int(kw, ("depth_a",), 4),
             db=pick_int(kw, ("depth_b",), 4), dc=pick_int(kw, ("depth_c",), 2),
             bias=pick_int(kw, ("bias",), 1, minimum=0))
    if b["bias"] not in (0,1):
        raise ValueError("bias must be 0 or 1")
    name = "quant_batch_matmul_mxa8w4"
    result = evaluate_math(name, b)
    c = result["counts"]
    vec = c["ops_vec"] + c["bit_ops"]
    return OpFlowTemplate(name=name, archetype=ExecutionArchetype.CV_COUPLED_1C2V,
                          total_units=ceildiv(m,tm)*ceildiv(n,tn), base_tile_units=1,
                          live_bytes_per_unit=0, pipe_work=(
                              PipeWork("CUBE", c["issued_flops_cube"]//2, "macs"), PipeWork("VEC", vec)),
                          gm_read_bytes=c["actual_read_bytes"], gm_write_bytes=c["actual_write_bytes"],
                          unique_gm_bytes=c["logical_read_bytes"] + c["logical_write_bytes"],
                          buffers=_buffers(name, result), tiling_pool=BufferPool.L1,
                          sync_pairs=2, vector_ops_by_lane=(vec//2, vec-vec//2), dtype="mxfp8",
                          estimate_kind="analytic_prior", authoring_backend=AuthoringBackend.CANNBOT_DSL,
                          params={"m":m,"n":n,"k":k,"dtype":"mxfp8", "storage_bits":4,
                                  "exact_counts":c,"count_bindings":b,
                                  "rate_assumption":"bit unpack and arithmetic priced by one Vector prior"},
                          notes="two lanes each load packed weights; wrapper zero-fill is a separate launch")
