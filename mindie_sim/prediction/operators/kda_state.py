"""KDA prefill FWD_H state phase on the shared mixed-core resource engine.

[Role]
  Bind the retained vLLM-Ascend KDA FWD_H source phase to a bounded per-stream
  chunk pipeline.  The semantic KDA boundary and its external gate/QK
  normalization stay with the caller; this module owns only recurrent state.

[Invariants]
  * Source domain is BF16 K/V/W/U, FP32 key-wise gates and FP32 state,
    ``dk == dv == 128``, ``chunk == 64``.  Unsupported branches are errors.
  * Each ``(sequence, value_head)`` is an independent state stream.  Chunks are
    serial: C1 ``W @ H`` -> vector ``v_new = U - V_work`` -> C2
    ``K_decay^T @ v_new`` -> vector ``H = gate_last * H + H_work``.  Every
    chunk writes its FP32 state checkpoint; earlier chunk boundaries are
    internal commits and the final boundary publishes the request state.
  * The internal state is ``[K, V]`` FP32.  The caller's public ``[V, K]``
    transpose is outside this selected phase.
  * A core owns at most two live source streams (ping-pong slots), assigned by
    the source task stride.  Other cores or AIV pairs are never added as supply.
    Vector phases use aggregate pair supply; odd V1 tails are not lane-exact.
  * Work is interval arithmetic over real chunk tails; programs are lazy and
    retain only the active stream/chunk stages.

[Timeouts & Locks]
  Pure metadata and arithmetic.  No I/O, locks, device contexts or threads.

[Side-Effect Policy]
  Returns immutable plans and submits only to the caller's virtual engine.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Any, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import (
    OpSpec, ProgramFragment, SemanticAttr, StateEffect,
)
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .kda_state_selection import (
    bind_state_value, binding_record, prefill_records, state_ports, state_value,
)
from .mixed_core import placement, require_pair, split_reads

RECIPE = "vllm_ascend_kda_fwd_h"
IMPLEMENTATION = "vllm_ascend_kda_chunk_state"

# Arch35 FWD_H tile shape and KDA wrapper constants.
_K_DIM = 128
_V_DIM = 128
_CHUNK = 64
_BF16_BYTES = 2
_FP32_BYTES = 4
_STREAMS_PER_CORE = 2
_IMPLEMENTATION = "chunk_kda_fwd/arch35/FWD_H/k_gated/scalar_off/exp2/state_fp32"

_PARAM_KEYS = frozenset({
    "lengths", "heads", "key_heads", "dk", "dv", "chunk", "dtype",
    "gate_dtype", "state_dtype", "use_initial_state", "output_final_state",
    "state_v_first", "use_exp2", "save_new_value",
})


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _positive_int(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(reason)
    return value


def _flag(params: Mapping[str, Any], name: str, default: bool) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ModelingError(f"kda_state_parameter_must_be_boolean:{name}")


def _dtype(value: Any, allowed: Mapping[str, str], reason: str) -> str:
    key = str(value).lower()
    if key not in allowed:
        raise ModelingError(reason)
    return allowed[key]


@dataclass(frozen=True)
class Params:
    lengths: tuple[int, ...]
    heads: int
    key_heads: int
    dk: int
    dv: int
    chunk: int

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def chunks_per_sequence(self) -> tuple[int, ...]:
        return tuple(_ceil_div(length, self.chunk) for length in self.lengths)

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_per_sequence)

    @property
    def head_tasks(self) -> int:
        return len(self.lengths) * self.heads

    def metadata(self) -> dict[str, Any]:
        per_core_workspace = {
            "v_work": 2 * self.chunk * self.dv * _FP32_BYTES,
            "v_update": 2 * self.chunk * self.dv * _FP32_BYTES,
            "k_decay": 2 * self.chunk * self.dk * _FP32_BYTES,
            "h_work": 2 * self.dk * self.dv * _FP32_BYTES,
        }
        return {
            "lengths": self.lengths,
            "heads": self.heads,
            "key_heads": self.key_heads,
            "group_size": self.heads // self.key_heads,
            "dk": self.dk,
            "dv": self.dv,
            "chunk": self.chunk,
            "dtype": "bf16",
            "gate_dtype": "fp32",
            "state_dtype": "fp32",
            "use_initial_state": True,
            "output_final_state": True,
            "state_v_first": True,
            "internal_state_layout": "K,V",
            "public_state_layout": "V,K",
            "caller_state_transpose": "outside_selected_fwd_h_phase",
            "use_exp2": True,
            "save_new_value": True,
            "implementation": _IMPLEMENTATION,
            "source_phase": "FWD_H",
            "native_kind": "ChunkKdaFwd/FWD_H",
            "gate_mode": "key_wise_fp32",
            "scalar_gate": False,
            "k_gated": True,
            "tile_shapes": "GDNFwdHTileShapes128",
            "streams_per_core": _STREAMS_PER_CORE,
            "workspace_bytes_per_core": sum(per_core_workspace.values()),
            "workspace_per_core": per_core_workspace,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "chunks_per_sequence": self.chunks_per_sequence,
            "head_tasks": self.head_tasks,
            "state_checkpoints_per_stream": "one_fp32_write_per_chunk",
            "h_snapshot_writes_per_stream": "one_per_chunk_including_initial_h0",
            "tail_local_l1_zero_fill_bytes": (
                sum(length % self.chunk != 0 for length in self.lengths)
                * self.heads * 4 * 128 * 128 * _BF16_BYTES),
            "tail_local_l1_zero_fill_service": "not_timed_no_declared_fill_resource",
            "schedule_scope": "owner_phase_order_with_global_h0_barrier",
            "vector_lane_scope": "aggregate_aiv_pair_odd_v1_tail_not_lane_exact",
            "dispatch": {
                "implementation": _IMPLEMENTATION,
                "source_phase": "FWD_H",
                "tile_shapes": "GDNFwdHTileShapes128",
                "k_gated": True,
                "scalar_gated": False,
                "use_exp2": True,
                "state_dtype": "fp32",
                "workspace_dtype": "fp32",
                "use_initial_state": True,
                "output_final_state": True,
            },
        }


def _parse_params(params: Mapping[str, Any]) -> Params:
    if not isinstance(params, Mapping):
        raise ModelingError("kda_state_params_must_be_a_mapping")
    unknown = set(params) - _PARAM_KEYS
    if unknown:
        raise ModelingError("kda_state_unknown_parameter:" + ",".join(sorted(unknown)))

    raw_lengths = params.get("lengths")
    if (isinstance(raw_lengths, (str, bytes))
            or not isinstance(raw_lengths, (list, tuple)) or not raw_lengths):
        raise ModelingError("kda_state_lengths_must_be_a_nonempty_integer_list")
    lengths = tuple(_positive_int(value, "kda_state_lengths_must_be_positive")
                    for value in raw_lengths)

    heads = _positive_int(params.get("heads"), "kda_state_heads_must_be_positive")
    key_heads = _positive_int(params.get("key_heads"),
                              "kda_state_key_heads_must_be_positive")
    if key_heads != heads:
        raise ModelingError("kda_state_requires_equal_key_and_value_heads")
    dk = _positive_int(params.get("dk"), "kda_state_dk_must_be_positive")
    dv = _positive_int(params.get("dv"), "kda_state_dv_must_be_positive")
    chunk = _positive_int(params.get("chunk"), "kda_state_chunk_must_be_positive")
    if dk != _K_DIM:
        raise ModelingError(f"kda_state_requires_k_head_dim:{_K_DIM}")
    if dv != _V_DIM:
        raise ModelingError(f"kda_state_requires_v_head_dim:{_V_DIM}")
    if chunk != _CHUNK:
        raise ModelingError(f"kda_state_requires_chunk_size:{_CHUNK}")

    _dtype(params.get("dtype", "bf16"),
           {"bf16": "bf16", "bfloat16": "bf16"}, "kda_state_requires_bf16_inputs")
    _dtype(params.get("gate_dtype", "fp32"),
           {"fp32": "fp32", "float32": "fp32"}, "kda_state_requires_fp32_gates")
    _dtype(params.get("state_dtype", "fp32"),
           {"fp32": "fp32", "float32": "fp32"}, "kda_state_requires_fp32_state")

    flags = {
        "use_initial_state": True,
        "output_final_state": True,
        "state_v_first": True,
        "use_exp2": True,
        "save_new_value": True,
    }
    for name, required in flags.items():
        if not _flag(params, name, required):
            raise ModelingError(f"kda_state_requires_flag:{name}")
    return Params(lengths, heads, key_heads, dk, dv, chunk)


def _params_from_metadata(metadata: Mapping[str, Any]) -> Params:
    return _parse_params({key: metadata[key] for key in _PARAM_KEYS if key in metadata})


def state_offset(sequence, head, key, value, heads):
    """Internal FP32 state address: sequence-major, then head, then [K, V]."""
    return ((sequence * heads + head) * _K_DIM + key) * _V_DIM + value


def h_snapshot_offset(head, total_chunks, chunk_prefix, chunk, key, value):
    """BF16 H snapshot address in the source varlen head/chunk workspace."""
    return (((head * total_chunks + chunk_prefix + chunk) * _K_DIM + key)
            * _V_DIM + value)


# ---------------------------------------------------------------------------
# Source phase work.  Every byte and operation below is derived from the
# kernel's C1/V1/C2/V2 tensor intervals; no rate, floor or measured span enters.
# ---------------------------------------------------------------------------

def _work(*, matrix: float = 0.0, read: float = 0.0, write: float = 0.0,
          vector: float = 0.0, mte1: float = 0.0,
          fixpipe: float = 0.0) -> Work:
    resources = {
        "hbm": read + write + fixpipe,
        "mte2": read,
        "mte3": write,
        "vector": vector,
        "mte1": mte1,
        "fixpipe": fixpipe,
    }
    return Work({"bf16": matrix} if matrix else {},
                {key: value for key, value in resources.items() if value})


@dataclass(frozen=True)
class ChunkWork:
    """Source phases for one stream chunk; ``init`` exists only on chunk 0."""

    init: Work
    c1: Work
    delta: Work
    c2: Work
    state: Work

    def total(self) -> Work:
        return self.init + self.c1 + self.delta + self.c2 + self.state


def chunk_work(params: Params, valid: int, *, first: bool, last: bool) -> ChunkWork:
    """Physical work for one ``(sequence, head, chunk)`` source task."""
    k, v, chunk = params.dk, params.dv, params.chunk
    h_snapshot = k * v * _BF16_BYTES
    state = k * v * _FP32_BYTES
    v_work = valid * v * _FP32_BYTES
    k_work = valid * k * _BF16_BYTES
    h_work = k * v * _FP32_BYTES
    gate_last = k * _FP32_BYTES

    init = _work(
        read=state, write=h_snapshot, vector=k * v,
    ) if first else _work()

    # C1: V_work = W @ H. GM payload and MMAD arguments use real valid rows.
    # Tail L1 zero-fill is a separate local operation recorded in metadata;
    # it does not create padded GM reads or proven matrix instructions.
    c1 = _work(
        matrix=2 * valid * k * v,
        read=valid * k * _BF16_BYTES + h_snapshot,
        vector=0,
        mte1=2 * (valid * k + k * v),
        fixpipe=v_work,
    )

    # V1: V_new = U - V_work; K is copied to the decay workspace, and the
    # BF16 V_update operand is published for C2.
    delta = _work(
        read=valid * v * _BF16_BYTES + v_work + k_work,
        write=2 * valid * v * _BF16_BYTES + k_work,
        vector=4 * valid * v,
    )

    # C2: H_work = K_decay^T @ V_update.
    c2 = _work(
        matrix=2 * k * valid * v,
        read=k_work + valid * v * _BF16_BYTES,
        mte1=2 * (valid * k + valid * v),
        fixpipe=h_work,
    )

    # V2: H = exp2(g_last) * H + H_work.  The state is read/written through GM
    # on every chunk; non-final chunks also publish the BF16 snapshot C1 reads.
    # The initial H0 conversion is the separate ``init`` phase above.
    state_read = h_work + gate_last + state
    state_write = state + (h_snapshot if not last else 0)
    gate_ops = 2 * k
    state_vector = gate_ops + k * v * (2 + (0 if last else 1))
    state_phase = _work(
        read=state_read, write=state_write, vector=state_vector,
    )
    return ChunkWork(init, c1, delta, c2, state_phase)


# ---------------------------------------------------------------------------
# Source stream ownership.  The scheduler starts each active core at
# ``core * slots + slot`` and advances by ``core_count * slots``.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamTask:
    core: int
    slot: int
    sequence: int
    head: int
    length: int
    chunks: int
    chunk_offset: int


@dataclass(frozen=True)
class CoreTasks:
    core: int
    rounds: tuple[tuple[StreamTask, ...], ...]

    @property
    def streams(self) -> tuple[StreamTask, ...]:
        return tuple(task for round in self.rounds for task in round)


def head_ownership(lengths, heads: int, core_count: int, *,
                   chunk: int = _CHUNK) -> tuple[CoreTasks, ...]:
    """Resolve the source task queue into at most two live streams per core."""
    lengths = tuple(lengths)
    if not lengths or any(isinstance(value, bool) or not isinstance(value, int)
                          or value < 1 for value in lengths):
        raise ModelingError("kda_state_lengths_must_be_positive")
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ModelingError("kda_state_heads_must_be_positive")
    if isinstance(core_count, bool) or not isinstance(core_count, int) or core_count < 1:
        raise ModelingError("kda_state_requires_positive_compute_units")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ModelingError("kda_state_chunk_must_be_positive")

    total = len(lengths) * heads
    slots = _STREAMS_PER_CORE if total > core_count else 1
    stride = core_count * slots
    prefixes = [0]
    for length in lengths:
        prefixes.append(prefixes[-1] + _ceil_div(length, chunk))

    result = []
    for core in range(core_count):
        rounds = []
        base = core * slots
        while base < total:
            tasks = []
            for slot in range(slots):
                index = base + slot
                if index >= total:
                    continue
                sequence, head = divmod(index, heads)
                tasks.append(StreamTask(
                    core=core, slot=slot, sequence=sequence, head=head,
                    length=lengths[sequence],
                    chunks=_ceil_div(lengths[sequence], chunk),
                    chunk_offset=prefixes[sequence],
                ))
            if tasks:
                rounds.append(tuple(tasks))
            base += stride
        if rounds:
            result.append(CoreTasks(core, tuple(rounds)))
    return tuple(result)


def stream_work(params: Params, task: StreamTask) -> Work:
    """Sum the real chunk tails of one independent source state stream."""
    total = Work()
    for index in range(task.chunks):
        valid = min(params.chunk, task.length - index * params.chunk)
        total += chunk_work(
            params, valid, first=index == 0, last=index == task.chunks - 1).total()
    return total


# ---------------------------------------------------------------------------
# Bounded mixed-core program.
# ---------------------------------------------------------------------------

def _phases(name, work: Work, context: RecipeContext, core: int, *,
            aiv_pair: bool = False, after=(), previous=()):
    """Expose one source phase as read / inner-copy / compute / publish / write.

    The copy/vector side is pinned to both AIVs of the owning AIC.  The matrix
    side is pinned to that AIC.  No other core or AIV is admitted as supply.
    """
    resources = dict(work.resources)
    read = resources.pop("mte2", 0)
    write = resources.pop("mte3", 0)
    memory = resources.pop("hbm", 0)
    mte1 = resources.pop("mte1", 0)
    fixpipe = resources.pop("fixpipe", 0)
    role = read_role(context.resource_names, 0 if aiv_pair else None)
    if aiv_pair:
        owned = context.owned(core, context.core_count,
                              ("vector", role, "mte3"))
    else:
        owned = placement(context, core)

    gm_read = min(read, memory)
    gm_write = memory - gm_read
    prefix = name + ":"
    load = Stage(prefix + "load",
                 context.bind(Work(resources={"hbm": gm_read, role: read})),
                 after=after, previous=previous, placement=owned)
    stages = [load]
    tail = load
    if mte1:
        tail = Stage(prefix + "l0",
                     context.bind(Work(resources={"mte1": mte1})),
                     after=(load.name,), placement=owned)
        stages.append(tail)
    compute = Stage(prefix + "compute",
                    context.bind(Work(work.matrix, resources)),
                    after=(tail.name,), placement=owned)
    stages.append(compute)
    tail = compute
    if fixpipe:
        fix_gm = gm_write if not write else 0
        gm_write -= fix_gm
        publish = Stage(prefix + "publish",
                        context.bind(Work(resources={"fixpipe": fixpipe,
                                                     "hbm": fix_gm})),
                        after=(tail.name,), placement=owned)
        stages.append(publish)
        tail = publish
    stages.append(Stage(
        prefix + "store",
        context.bind(Work(resources={"hbm": gm_write, "mte3": write})),
        after=(tail.name,), placement=owned))
    stages.append(Stage(name, after=(prefix + "store",)))
    return tuple(stages)


@dataclass(frozen=True)
class SlotQueue:
    """Finite stream descriptors; chunk indexes are resolved without expansion."""

    tasks: tuple[StreamTask, ...]
    ends: tuple[int, ...]

    def at(self, iteration: int) -> tuple[StreamTask, int] | None:
        index = bisect_right(self.ends, iteration)
        if index == len(self.tasks):
            return None
        start = self.ends[index - 1] if index else 0
        return self.tasks[index], iteration - start


def slot_queues(group: CoreTasks) -> tuple[SlotQueue, ...]:
    """Source InitTasks advances each slot after its own final chunk."""
    queues = []
    for slot in range(_STREAMS_PER_CORE):
        tasks = tuple(task for task in group.streams if task.slot == slot)
        ends, count = [], 0
        for task in tasks:
            count += task.chunks
            ends.append(count)
        if tasks:
            queues.append(SlotQueue(tasks, tuple(ends)))
    return tuple(queues)


def _core_program(group: CoreTasks, params: Params,
                  context: RecipeContext, fragment_map=None) -> PipelineProgram:
    queues = slot_queues(group)

    def tile(index):
        chunks = []
        for queue in queues:
            entry = queue.at(index)
            if entry is None:
                continue
            task, chunk = entry
            valid = min(params.chunk, task.length - chunk * params.chunk)
            work = chunk_work(params, valid, first=False,
                              last=chunk == task.chunks - 1)
            fragment = (None if fragment_map is None
                        else fragment_map.get((task.sequence, task.head, chunk)))
            chunks.append((task.slot, fragment, work))
        stages = []
        # The source AIC issues C1 for both slots before C2 for both slots;
        # the AIV pair does the analogous V1/V2 ordering. Different processors
        # may overlap, but a slot cannot overwrite its prior state or scratch.
        for phase, aiv in (("c1", False), ("delta", True),
                           ("c2", False), ("state", True)):
            phase_tail = None
            for slot, fragment, work in chunks:
                name = f"{phase}s{slot}"
                after = (() if phase_tail is None else (phase_tail,))
                previous = ()
                if phase == "c1":
                    previous = (f"states{slot}",)
                    if phase_tail is None:
                        previous += ("cube_done",)
                elif phase == "delta":
                    after += (f"c1s{slot}",)
                    if phase_tail is None:
                        previous = ("vector_done",)
                elif phase == "c2":
                    after += (f"deltas{slot}", f"c1s{chunks[-1][0]}")
                else:
                    after += (f"c2s{slot}", f"deltas{chunks[-1][0]}")
                phase_stages = _phases(
                    name, getattr(work, phase), context, group.core,
                    aiv_pair=aiv, after=after, previous=previous)
                if phase == "state" and fragment is not None:
                    phase_stages = (*phase_stages[:-1],
                                    replace(phase_stages[-1], fragment=fragment))
                stages.extend(phase_stages)
                phase_tail = name
        stages.append(Stage("cube_done", after=(f"c2s{chunks[-1][0]}",)))
        stages.append(Stage("vector_done", after=(f"states{chunks[-1][0]}",)))
        return tuple(stages)

    return PipelineProgram(max(queue.ends[-1] for queue in queues), 2, tile)


def _program(params: Params, groups: tuple[CoreTasks, ...],
             context: RecipeContext, core_count: int,
             fragment_map=None, read_fragment=None):
    if context.resource_units.get("vector", core_count) != 2 * core_count:
        raise ModelingError("kda_state_requires_two_vector_subblocks_per_matrix_core")
    if int(context.core_count) != core_count:
        raise ModelingError("kda_state_geometry_changed_after_build")
    init_programs = []
    for group in groups:
        # H0 for every stream is materialized before the source SyncAll. The
        # initialization is a per-owner resource phase, not individual casts.
        work = _work(read=params.dk * params.dv * _FP32_BYTES,
                     write=params.dk * params.dv * _BF16_BYTES,
                     vector=params.dk * params.dv).scaled(len(group.streams))
        init_programs.append(KernelProgram(_phases(
            "init", work, context, group.core, aiv_pair=True)))
    initialize = Stage("initialize", program=ParallelProgram(tuple(init_programs)))
    if read_fragment is not None:
        initialize = replace(initialize, fragment=read_fragment)
    return KernelProgram((
        initialize,
        Stage("scan", after=("initialize",), program=ParallelProgram(tuple(
            _core_program(group, params, context, fragment_map)
            for group in groups))),
    ))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower the selected FWD_H phase onto one shared resource engine."""
    params = _params_from_metadata(dict(op.metadata))
    require_pair(geometry)
    core_count = int(geometry.compute_units)
    if core_count < 1:
        raise ModelingError("kda_state_requires_positive_compute_units")
    groups = head_ownership(params.lengths, params.heads, core_count,
                            chunk=params.chunk)

    total = Work()
    aic_mte2 = 0
    for group in groups:
        for task in group.streams:
            for index in range(task.chunks):
                valid = min(params.chunk, task.length - index * params.chunk)
                work = chunk_work(
                    params, valid, first=index == 0,
                    last=index == task.chunks - 1)
                total += work.total()
                aic_mte2 += (work.c1.resources.get("mte2", 0)
                             + work.c2.resources.get("mte2", 0))
    contract = dict(op.metadata).get("kda_state_contract")
    layout_bytes = 0.0
    if contract is not None:
        layout_bytes = float(contract["layout_bytes"])
    total = split_reads(total, aic_mte2, geometry.resource_units)

    fragments = state_fragments(op)

    def program(context: RecipeContext):
        return _program(params, groups, context, core_count,
                        fragment_map=fragments["internal"],
                        read_fragment=fragments["initialize"])

    metadata = ({} if contract is None else {
        "state_contract": contract,
        "state_effects": dict(op.metadata).get("kda_state_effects", ()),
        "state_fragment_counts": {
            "initialize": 1 if fragments["initialize"] is not None else 0,
            "internal": len(fragments["internal"]),
            "publication": len(fragments["publication"]),
        },
        "state_accesses": state_accesses(op),
        "layout_bytes": layout_bytes,
        "publication_producer": contract["layout_producer"],
        "state_bindings": tuple(
            binding_record(binding)
            for binding in contract["publication_bindings"]),
    })
    return OperatorPlan(total, program, metadata=metadata)


def _effect(target, action, version, arbiter=""):
    return StateEffect(target, action, version, "", arbiter)


def _state_effect(target, action, version, arbiter=""):
    return _effect(target, action, version, arbiter).to_dict()


def _per_sequence(value, count, default=None):
    if value is None or value == () or value == []:
        return (default,) * count
    if isinstance(value, Mapping):
        return tuple(value.get(index, default) for index in range(count))
    value = tuple(value)
    if len(value) != count:
        raise ModelingError("kda_prefill_values_must_match_sequences")
    return value


def _sequence_values(value, count, shape):
    """Split one pooled TensorValue on its leading sequence axis."""
    if isinstance(value, TensorValue):
        layout = tuple(value.layout.shape)
        if len(layout) == len(shape) + 1 and layout[0] == count \
                and layout[1:] == tuple(shape):
            return tuple(value.narrow(0, index, 1) for index in range(count))
        if count == 1 and layout == tuple(shape):
            return (value,)
    return _per_sequence(value, count)


def _public_value(value, *, target, shape, dtype, context, ready):
    if value is None:
        return TensorValue(
            "external", dtype, TensorLayout.contiguous(shape), output=0,
            storage=f"public_state:{target}", binding="external",
            binding_name=ready or f"kda.public_state[{target}]",
            state_version=context)
    value = state_value(value)
    if tuple(value.layout.shape[-len(shape):]) != shape:
        raise ModelingError(
            f"kda_prefill_public_value_shape_mismatch:{value.layout.shape}:{shape}")
    if value.dtype != dtype:
        raise ModelingError(
            f"kda_prefill_public_value_dtype_mismatch:{value.dtype}:{dtype}")
    if value.state_version is not None and value.state_version != context:
        raise ModelingError(
            f"kda_prefill_public_value_version_mismatch:"
            f"{value.state_version}:{context}")
    return value


def _head_view(value, head, *, heads):
    shape = tuple(value.layout.shape)
    if len(shape) == 3:
        return value.permute((0, 2, 1)).narrow(0, head, 1)
    if len(shape) == 4 and shape[0] == 1:
        return value.permute((0, 1, 3, 2)).narrow(1, head, 1)
    raise ModelingError(
        f"kda_prefill_public_value_requires_head_axis:{shape}:{heads}")


def prefill_state_contract(*, lengths, requests=(), chunk=64, contexts=None,
                           targets=None, heads=1, dk=128, dv=128,
                           state_dtype="fp32", op_id="kda_fwd_h",
                           state_values=None, layout_values=None,
                           state_ready=(), state_grants=(), owner=None):
    """Declare per-head FWD_H stores from real public subviews.

    The initialize stage reads a caller ``TensorValue`` subview of the public
    ``[V,K]`` state (or an explicit standalone external port).  Chunk stores,
    internal reads, the final layout producer and the public commit all derive
    from that same binding.  No internal target or public version is prefilled.
    """
    lengths = tuple(lengths)
    if targets is None:
        requests = tuple(requests)
        targets = tuple(
            f"kda.layer.recurrent_state["
            f"{requests[index] if index < len(requests) else index}]"
            for index in range(len(lengths)))
    targets = tuple(targets)
    records = prefill_records(
        targets=targets, lengths=lengths, contexts=contexts, chunk=chunk,
        heads=heads, dk=dk, dv=dv, state_dtype=state_dtype, op_id=op_id)
    contexts = records["contexts"]
    count = len(targets)
    ready = _per_sequence(state_ready, count, "")
    grants = _per_sequence(state_grants, count, None)
    shape = (heads, dv, dk)
    values = _sequence_values(state_values, count, shape)
    layouts = _sequence_values(layout_values, count, shape)
    owner = owner or "kda.layer"
    read_bindings, initialize_bindings = [], []
    internal_bindings = {}
    for sequence, (target, context) in enumerate(zip(targets, contexts)):
        public = _public_value(
            values[sequence], target=target, shape=shape, dtype=state_dtype,
            context=context, ready=ready[sequence])
        read_bindings.append(bind_state_value(
            public, target=target, owner=owner, version=context,
            ready=ready[sequence] or public.producer, grant=grants[sequence],
            generation=context, required=True, representation="state",
            span=((0, heads), (0, dv), (0, dk)),
            readiness=("external" if public.binding == "external"
                       else "producer_complete")))
        for head in range(heads):
            head_value = _head_view(public, head, heads=heads)
            internal = records["internal_targets"][sequence][head]
            initial = replace(
                head_value, producer=f"{op_id}:initialize",
                output=f"{sequence}:{head}",
                storage=f"state:{sequence}:head:{head}", binding="static",
                binding_name="", state_version=context)
            initialize_bindings.append(bind_state_value(
                initial, target=internal, owner=f"{owner}.head[{head}]",
                version=context, generation=context, required=True,
                representation=state_dtype, span=((0, dk), (0, dv))))
            previous = initial
            for chunk_index, version in enumerate(
                    records["checkpoint_versions"][sequence]):
                chunk_value = replace(
                    previous, producer=f"{op_id}:state_chunk:"
                                      f"{sequence}:{head}:{chunk_index}",
                    output=chunk_index, binding="static", binding_name="",
                    state_version=version)
                internal_bindings[(sequence, head, chunk_index)] = (
                    bind_state_value(
                        chunk_value, target=internal,
                        owner=f"{owner}.head[{head}]", version=version,
                        generation=version, required=True,
                        representation=state_dtype,
                        span=((0, dk), (0, dv))))
                previous = chunk_value
    publication_bindings = []
    for sequence, (target, context) in enumerate(zip(targets, contexts)):
        final = records["publication_write_records"][sequence]["version"]
        layout = layouts[sequence]
        if layout is None:
            layout = TensorValue(
                records["layout_producer"], state_dtype,
                TensorLayout.contiguous(shape), output=sequence,
                storage=f"public_state:{target}", binding="static",
                binding_name="", state_version=final)
        else:
            layout = state_value(layout)
            if tuple(layout.layout.shape[-3:]) != shape:
                raise ModelingError(
                    f"kda_prefill_layout_shape_mismatch:"
                    f"{layout.layout.shape}:{shape}")
            if layout.dtype != state_dtype:
                raise ModelingError(
                    f"kda_prefill_layout_dtype_mismatch:"
                    f"{layout.dtype}:{state_dtype}")
            if layout.state_version is not None and layout.state_version != final:
                raise ModelingError(
                    f"kda_prefill_layout_version_mismatch:"
                    f"{layout.state_version}:{final}")
        publication_bindings.append(bind_state_value(
            layout, target=target, owner=owner, version=final,
            ready=layout.producer, generation=final, required=True,
            representation="layout",
            span=((0, heads), (0, dv), (0, dk)),
            readiness="producer_complete",
            last_use="after_layout_publication"))
    publication_effects = []
    for record in records["publication_write_records"]:
        publication_effects.append(_state_effect(
            record["target"], "write", record["version"],
            "layout_producer_complete"))
    for record in records["publication_records"]:
        publication_effects.append(_state_effect(
            record["target"], "commit", record["version"],
            "layout_publication"))
    contract = {
        **{key: records[key] for key in (
            "phase", "queries", "contexts", "version_basis", "targets",
            "heads", "dk", "dv", "state_dtype", "internal_targets",
            "layout_producer", "layout_bytes",
            "checkpoint_versions", "internal_checkpoint_versions",
            "internal_commit_versions", "commit_versions", "commit_policy",
            "candidate_write_point", "publication_point")},
        "chunk": chunk,
        "request_ids": tuple(requests),
        "owner": owner,
        "state_ready": tuple(ready),
        "state_grants": tuple(grants),
        "read_versions": tuple(records["contexts"]),
        "read_records": tuple(records["read_records"]),
        "initialize_write_records": tuple(records["initialize_write_records"]),
        "internal_read_records": tuple(records["internal_read_records"]),
        "internal_write_records": tuple(records["internal_write_records"]),
        "internal_commit_records": tuple(records["internal_commit_records"]),
        "publication_write_records": tuple(records["publication_write_records"]),
        "publication_records": tuple(records["publication_records"]),
        "publication_write_records": tuple(records["publication_write_records"]),
        "read_bindings": tuple(binding_record(b) for b in read_bindings),
        "initialize_bindings": tuple(
            binding_record(b) for b in initialize_bindings),
        "internal_bindings": tuple(
            (key, binding_record(b))
            for key, b in sorted(internal_bindings.items())),
        "publication_bindings": tuple(
            binding_record(b) for b in publication_bindings),
        "history_selection": {
            "contexts": tuple(records["contexts"]),
            "version_basis": records["version_basis"],
            "public_subviews": tuple(
                (sequence, context) for sequence, context in
                enumerate(records["contexts"])),
        },
        "current_geometry": {
            "queries": records["queries"],
            "heads": heads, "dk": dk, "dv": dv, "chunk": chunk,
            "targets": targets, "request_ids": tuple(requests),
        },
        "internal_record_count": len(records["internal_commit_records"]),
    }
    return contract, tuple(publication_effects)
def _fragment(name, effects, ports=None):
    return ProgramFragment(name, state_effects=tuple(effects),
                           ports=PortSet() if ports is None else ports)


def _prefill_publication_fragments(op, contract):
    result = []
    for index, (write, commit) in enumerate(zip(
            contract["publication_write_records"],
            contract["publication_records"])):
        result.append(_fragment(
            f"{op.op_id}:state_publication:{index}",
            (_effect(write["target"], "write", write["version"]),
             _effect(commit["target"], "commit", commit["version"],
                     "caller_layout_publication"))))
    return tuple(result)


def _binding_attr(bindings):
    if not bindings:
        return ()
    return (SemanticAttr(
        "state_bindings",
        tuple(binding_record(binding) for binding in bindings)),)


def _binding_ports(bindings, *, direction, owner, readiness,
                   optional=False):
    return state_ports(
        bindings=bindings, owner=owner, direction=direction,
        readiness=readiness, optional=optional)


def state_fragments(op):
    """Return per-head FWD_H stores and the final layout publication.

    ``initialize`` reads the public subview and writes the first internal
    values; ``internal`` is keyed by ``(sequence, head, chunk)`` and reads the
    previous internal version before its own write/commit; ``publication`` is
    one fragment per sequence whose write/commit follows the layout producer.
    """
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return {"read": None, "initialize": None, "internal": {},
                "publication": ()}
    owner = contract.get("owner", f"kda.layer[{op.layer}]")
    read_bindings = tuple(contract.get("read_bindings", ()))
    read_effects = tuple(
        _effect(record["target"], "read", record["version"])
        for record in contract["read_records"])
    read = ProgramFragment(
        f"{op.op_id}:state_read", state_effects=read_effects,
        ports=_binding_ports(read_bindings, direction="in", owner=owner,
                             readiness="previous_iteration", optional=False),
        attrs=_binding_attr(read_bindings))
    initialize_bindings = tuple(contract.get("initialize_bindings", ()))
    initialize_effects = read_effects + tuple(
        _effect(record["target"], "write", record["version"])
        for record in contract["initialize_write_records"])
    initialize_ports = PortSet(
        tensor_in=_binding_ports(
            read_bindings, direction="in", owner=owner,
            readiness="previous_iteration", optional=False).tensor_in,
        tensor_out=_binding_ports(
            initialize_bindings, direction="out", owner=owner,
            readiness="producer_complete", optional=False).tensor_out)
    initialize = ProgramFragment(
        f"{op.op_id}:state_initialize", state_effects=initialize_effects,
        ports=initialize_ports,
        attrs=_binding_attr((*read_bindings, *initialize_bindings)))
    internal = {}
    by_key = dict(contract.get("internal_bindings", ()))
    reads = {(record["sequence"], record["head"], record["chunk"]):
             record for record in contract.get("internal_read_records", ())}
    writes = {(record["sequence"], record["head"], record["chunk"]):
              record for record in contract.get("internal_write_records", ())}
    commits = {(record["sequence"], record["head"], record["chunk"]):
               record for record in contract.get("internal_commit_records", ())}
    for key, binding in by_key.items():
        read_record = reads[key]
        write_record = writes[key]
        commit_record = commits[key]
        effects = (
            _effect(read_record["target"], "read", read_record["version"]),
            _effect(write_record["target"], "write", write_record["version"]),
            _effect(commit_record["target"], "commit", commit_record["version"]),
        )
        internal[key] = ProgramFragment(
            f"{op.op_id}:state_chunk:{key[0]}:{key[1]}:{key[2]}",
            state_effects=effects,
            ports=_binding_ports(
                (binding,), direction="inout",
                owner=binding.get("owner", owner),
                readiness="producer_complete", optional=False),
            attrs=_binding_attr((binding,)))
    publication = []
    publication_bindings = tuple(contract.get("publication_bindings", ()))
    for sequence, binding in enumerate(publication_bindings):
        final_internal = []
        for head in range(int(contract["heads"])):
            chunk = len(contract["internal_checkpoint_versions"][sequence][head]) - 1
            final_internal.append(by_key[(sequence, head, chunk)])
        effects = []
        for internal_binding in final_internal:
            effects.append(_effect(
                internal_binding["target"], "read",
                internal_binding["state_version"]))
        write = contract["publication_write_records"][sequence]
        commit = contract["publication_records"][sequence]
        effects.append(_effect(write["target"], "write", write["version"]))
        effects.append(_effect(
            commit["target"], "commit", commit["version"], "layout_publication"))
        ports = PortSet(
            tensor_in=_binding_ports(
                tuple(final_internal), direction="in", owner=owner,
                readiness="producer_complete", optional=False).tensor_in,
            tensor_out=_binding_ports(
                (binding,), direction="out", owner=owner,
                readiness="producer_complete", optional=False).tensor_out,
            event_in=(EventPortSpec(
                name=(f"layout:{sequence}:{binding['producer']}:"
                      f"{binding['state_version']}"),
                direction="in", event="complete", scope="request",
                owner=binding["producer"], readiness="producer_complete",
                generation=binding["state_version"], optional=False),))
        publication.append(ProgramFragment(
            f"{op.op_id}:state_publication:{sequence}",
            state_effects=tuple(effects), ports=ports,
            attrs=_binding_attr((*final_internal, binding))))
    return {"read": read, "initialize": initialize, "internal": internal,
            "publication": tuple(publication)}


def layout_fragment(op):
    """Combine per-sequence publication fragments into one real layout stage."""
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return None
    fragments = state_fragments(op)["publication"]
    if not fragments:
        return None
    effects = tuple(
        effect for fragment in fragments for effect in fragment.state_effects)
    tensor_in = tuple(
        port for fragment in fragments for port in fragment.ports.tensor_in)
    tensor_out = tuple(
        port for fragment in fragments for port in fragment.ports.tensor_out)
    event_in = tuple(
        port for fragment in fragments for port in fragment.ports.event_in)
    bindings = tuple(contract.get("publication_bindings", ()))
    attrs = ((SemanticAttr("state_bindings", bindings),) if bindings else ())
    return ProgramFragment(
        f"{op.op_id}:state_layout", state_effects=effects,
        ports=PortSet(tensor_in=tensor_in, tensor_out=tensor_out,
                      event_in=event_in),
        attrs=attrs)
def state_accesses(op):
    """Project FWD_H physical reads/writes into SERVICE access records.

    Reads are the actual public ``[V,K]`` subviews consumed by initialize;
    initialize writes materialize the first internal per-head values; chunk
    accesses read the previous internal version and write/commit the next; the
    public write comes from the final layout producer, not the FWD_H name.
    """
    contract = dict(op.metadata).get("kda_state_contract")
    if contract is None:
        return ()
    owner = contract.get("owner", f"kda.layer[{op.layer}]")
    request_ids = tuple(contract.get("request_ids", ()))
    read_bindings = tuple(contract.get("read_bindings", ()))
    initialize_bindings = tuple(contract.get("initialize_bindings", ()))
    publication_bindings = tuple(contract.get("publication_bindings", ()))
    result = []
    for record in contract["read_records"]:
        sequence, head = record["sequence"], record["head"]
        binding = read_bindings[sequence]
        request = (request_ids[sequence] if sequence < len(request_ids)
                   else str(sequence))
        source = binding.get("ready") or binding["producer"]
        result.append({
            "occurrence": f"{op.op_id}:read:{sequence}:{head}",
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer,
            "direction": "read", "position": "state",
            "output": record["target"], "storage": binding["storage"],
            "owner": f"{owner}.head[{head}]",
            "representation": record["dtype"], "persistent": True,
            "temporary": False, "state_version": record["version"],
            "span": record["span"], "bytes": float(record["bytes"]),
            "source": source, "reader": "",
            "grant": binding.get("grant") or "", "decision": None,
            "last_use": "before_head_initialize", "rank": None,
            "scope": "target", "generation": record["version"],
            "slot": None, "window": None, "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:head:{head}",
            "request_slot": sequence, "rows": 1,
            "row_stride": int(contract["dk"]) * int(contract["dv"]),
            "tier": "hbm", "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"),
            "decision_event": None,
            "last_use_event": "before_head_initialize",
        })
    for record, binding in zip(
            contract["initialize_write_records"], initialize_bindings):
        sequence, head = record["sequence"], record["head"]
        request = (request_ids[sequence] if sequence < len(request_ids)
                   else str(sequence))
        result.append({
            "occurrence": (f"{op.op_id}:initialize:"
                           f"{sequence}:{head}"),
            "producer": record["producer"], "request": request,
            "stage": op.stage, "layer": op.layer,
            "direction": "write", "position": "state",
            "output": record["target"], "storage": binding["storage"],
            "owner": f"{owner}.head[{head}]",
            "representation": record["dtype"], "persistent": True,
            "temporary": False, "state_version": record["version"],
            "span": record["span"], "bytes": float(record["bytes"]),
            "source": record["producer"], "reader": "", "grant": "",
            "decision": "initial_materialization",
            "last_use": "next_head_chunk_read", "rank": None,
            "scope": "target", "generation": record["version"],
            "slot": None, "window": None, "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:head:{head}",
            "request_slot": sequence, "rows": 1,
            "row_stride": int(contract["dk"]) * int(contract["dv"]),
            "tier": "hbm", "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"),
            "decision_event": None,
            "last_use_event": "next_head_chunk_read",
        })
    internal_reads = {
        (record["sequence"], record["head"], record["chunk"]): record
        for record in contract["internal_read_records"]}
    internal_writes = {
        (record["sequence"], record["head"], record["chunk"]): record
        for record in contract["internal_write_records"]}
    for key, binding in contract.get("internal_bindings", ()):
        sequence, head, chunk = key
        request = (request_ids[sequence] if sequence < len(request_ids)
                   else str(sequence))
        read_record = internal_reads[key]
        write_record = internal_writes[key]
        result.append({
            "occurrence": f"{op.op_id}:chunk_read:{sequence}:{head}:{chunk}",
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer,
            "direction": "read", "position": "state",
            "output": read_record["target"], "storage": binding["storage"],
            "owner": f"{owner}.head[{head}]",
            "representation": read_record["dtype"], "persistent": True,
            "temporary": False, "state_version": read_record["version"],
            "span": read_record["span"], "bytes": float(read_record["bytes"]),
            "source": binding["producer"], "reader": "", "grant": "",
            "decision": None, "last_use": "before_chunk_store",
            "rank": None, "scope": "target",
            "generation": read_record["version"], "slot": None, "window": None,
            "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:head:{head}",
            "request_slot": sequence, "rows": 1,
            "row_stride": int(contract["dk"]) * int(contract["dv"]),
            "tier": "hbm", "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"), "decision_event": None,
            "last_use_event": "before_chunk_store",
        })
        result.append({
            "occurrence": f"{op.op_id}:write:{sequence}:{head}:{chunk}",
            "producer": write_record["producer"], "request": request,
            "stage": op.stage, "layer": op.layer,
            "direction": "write", "position": "state",
            "output": write_record["target"],
            "storage": binding["storage"],
            "owner": f"{owner}.head[{head}]",
            "representation": write_record["dtype"], "persistent": True,
            "temporary": False, "state_version": write_record["version"],
            "span": write_record["span"], "bytes": float(write_record["bytes"]),
            "source": write_record["producer"], "reader": "", "grant": "",
            "decision": "internal_chunk_commit",
            "last_use": "next_head_chunk_read", "rank": None,
            "scope": "target", "generation": write_record["generation"],
            "slot": None, "window": None, "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:head:{head}",
            "request_slot": sequence, "rows": 1,
            "row_stride": int(contract["dk"]) * int(contract["dv"]),
            "tier": "hbm", "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"), "decision_event": None,
            "last_use_event": "next_head_chunk_read",
        })
    for sequence, binding in enumerate(publication_bindings):
        record = contract["publication_write_records"][sequence]
        request = (request_ids[sequence] if sequence < len(request_ids)
                   else str(sequence))
        result.append({
            "occurrence": f"{op.op_id}:public:{sequence}",
            "producer": binding["producer"], "request": request,
            "stage": op.stage, "layer": op.layer,
            "direction": "write", "position": "layout",
            "output": record["target"], "storage": binding["storage"],
            "owner": owner, "representation": record["dtype"],
            "persistent": True, "temporary": False,
            "state_version": record["version"], "span": record["span"],
            "bytes": float(record["bytes"]), "source": binding["producer"],
            "reader": "", "grant": "", "decision": "layout_complete",
            "last_use": "after_layout_publication", "rank": None,
            "scope": "target", "generation": record["version"],
            "slot": None, "window": None, "offset": binding.get("offset"),
            "physical_key": f"{binding['storage']}:layout",
            "request_slot": sequence, "rows": int(contract["heads"]),
            "row_stride": int(contract["dk"]) * int(contract["dv"]),
            "tier": "hbm", "source_ready": binding.get("ready"),
            "grant_ready": binding.get("grant"),
            "decision_event": binding.get("ready"),
            "last_use_event": "after_layout_publication",
        })
    return tuple(result)
def make_op(params: Mapping[str, Any], *, op_id: str = "kda_fwd_h",
            deps=(), requests=(), stage: int = 0, layer: int = 0,
            phase: str = "prefill", stream: str = "",
            notes: tuple[str, ...] = (), state_contexts=None,
            state_targets=None, state_values=None, layout_values=None,
            state_ready=(), state_grants=()) -> OpSpec:
    """Create the semantic record for one actual FWD_H kernel launch."""
    parsed = _parse_params(params)
    meta = parsed.metadata()
    # Aggregate source work is independent of the core count; per-core
    # partition and residency are resolved by ``build``.
    total = Work()
    chunk_offset = 0
    for sequence, length in enumerate(parsed.lengths):
        chunks = parsed.chunks_per_sequence[sequence]
        for head in range(parsed.heads):
            task = StreamTask(
                core=0, slot=0, sequence=sequence, head=head, length=length,
                chunks=chunks, chunk_offset=chunk_offset)
            total += stream_work(parsed, task)
        chunk_offset += chunks
    chunk_tokens = sum(count * parsed.chunk
                       for count in parsed.chunks_per_sequence)
    semantic = (parsed.total_tokens, parsed.heads, parsed.dk, parsed.dv)
    if state_targets is None:
        request_ids = tuple(requests)
        layer_target = f"kda.layer[{layer}]" if layer >= 0 else "kda.call"
        state_targets = tuple(
            f"{layer_target}.recurrent_state["
            f"{request_ids[index] if index < len(request_ids) else index}]"
            for index in range(len(parsed.lengths)))
    contract, effects = prefill_state_contract(
        lengths=parsed.lengths, requests=requests, chunk=parsed.chunk,
        contexts=state_contexts, targets=state_targets, heads=parsed.heads,
        dk=parsed.dk, dv=parsed.dv, state_dtype="fp32", op_id=op_id,
        state_values=state_values, layout_values=layout_values,
        state_ready=state_ready, state_grants=state_grants,
        owner=(f"kda.layer[{layer}]" if layer >= 0 else "kda.call"))
    meta.update(
        kda_state_contract=contract, kda_state_effects=effects,
        state_targets=tuple(state_targets),
        state_checkpoint_versions=contract["checkpoint_versions"],
        state_read_versions=contract["read_versions"],
        state_commit_versions=contract["commit_versions"])
    op = OpSpec(
        op_id=op_id,
        family="recurrent_scan",
        phase=phase,
        stage=stage,
        layer=layer,
        requests=tuple(requests),
        deps=tuple(deps),
        semantic_shape=semantic,
        local_shape=semantic,
        padded_shape=(1, parsed.heads, chunk_tokens, parsed.dv),
        launch_shape=semantic,
        dtype="bf16",
        layout="TND",
        recipe=RECIPE,
        source_boundary="kernel",
        coverage="source_selected_fwd_h_phase",
        flops=sum(total.matrix.values()),
        matrix_flops=tuple(sorted(total.matrix.items())),
        hbm_bytes=total.resources.get("hbm", 0),
        mte2_bytes=total.resources.get("mte2", 0),
        mte3_bytes=total.resources.get("mte3", 0),
        vector_ops=total.resources.get("vector", 0),
        metadata=tuple(sorted(meta.items())),
        notes=(
            "one varlen FWD_H launch for all prefill sequences",
            "C1 -> V1 -> C2 -> V2 serial chunks per (sequence, head)",
            "internal FP32 state [K,V]; caller [V,K] transpose is outside phase",
        ) + tuple(notes),
        stream=stream,
    )
    return replace(op, metadata=op.metadata + (
        ("state_accesses", state_accesses(op)),))


__all__ = [
    "ChunkWork", "CoreTasks", "IMPLEMENTATION", "Params", "RECIPE", "StreamTask",
    "build", "chunk_work", "h_snapshot_offset", "head_ownership", "make_op",
    "slot_queues", "state_accesses", "state_fragments", "state_offset",
    "stream_work",
]
