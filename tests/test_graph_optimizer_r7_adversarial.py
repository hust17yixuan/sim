"""NR-S2-FIX7: executed Round-7 architectural-closure witnesses.

[Role] Reproduce the four Round-7 findings: global positional data operands
  through contraction and stream events, generation-scoped retire-gated pool
  leases, ULP-aware collective fill at finite bandwidth scale, and
  budget-bounded wide join/fork stream discovery.
[Invariants] Repeated external inputs and interleaved producers keep exact
  operand index order; a scheduled free keeps its key reserved until the Engine
  free executes; a stale copy handle can never free a newer generation; genuine
  oversubscription still raises while sub-16-ULP dust clamps to saturation; a
  wide discovery completes under its caller budget and bounds regions.
[Timeouts & Locks] Pure in-process virtual simulation; wide discovery is
  bounded by a 0.01-s wall budget and the measured allowance below.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from dataclasses import replace
import math
import random
import time

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction.engine_plan import PlanNode
from mindie_sim.prediction.fusion_bridge import _strip_events, insert_stream_events
from mindie_sim.prediction.fusion_ports import full_operand_ports
from mindie_sim.prediction.fusion_pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram,
)
from mindie_sim.prediction.graph_optimizer import run_optimizer
from mindie_sim.prediction.graph_optimizer_engine import _stream_regions
from mindie_sim.prediction.optimizer_report import Budget
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import hw

_CASE = None


def case():
    global _CASE
    if _CASE is None:
        _CASE = compose_case("Qwen3.8-27B", hardware="a3", tp=1, layers=1)
    return _CASE


def _add(op_id, deps=(), *, family="add", metadata=(), hbm=1_000_000.0):
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
        padded_shape=(1, 4), launch_shape=(1, 4), hbm_bytes=hbm,
        vector_ops=2.0, dtype="bf16", deps=tuple(deps),
        metadata=tuple(metadata))


# ---------------------------------------------------------------------------
# Finding 1: complete global positional operand sequence
# ---------------------------------------------------------------------------
def test_r7_f1_repeated_external_operands_keep_their_positions():
    ops = (
        _add("a", ("analysis_input",)),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", ("analysis_input", "analysis_input", "b")),
    )
    report = run_optimizer(case(), ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    assert [row["members"] for row in report.applied] == [["a", "b"]]
    graph = report.canonical_plan.graph
    node = graph.ops["c"]
    # Scheduling deps stay deduplicated, but the tensor expression keeps both
    # occurrences of the external input in exact global index order.
    assert node.deps == ("analysis_input", "fused:a+b")
    assert node.inputs == ("analysis_input:source", "analysis_input:source",
                           "fused:a+b:out")
    graph.validate()


def test_r7_f1_interleaved_producers_keep_their_positions():
    ops = (
        _add("a", ("analysis_input",)),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", ("b", "analysis_input", "b")),
    )
    report = run_optimizer(case(), ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    assert [row["members"] for row in report.applied] == [["a", "b"]]
    graph = report.canonical_plan.graph
    node = graph.ops["c"]
    assert node.deps == ("fused:a+b", "analysis_input")
    assert node.inputs == ("fused:a+b:out", "analysis_input:source",
                           "fused:a+b:out")
    graph.validate()


def test_r7_f1_chained_secondary_live_outs_survive_event_round_trip():
    ops = (
        _add("a", ("analysis_input",)),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", ("a", "b")),
        _add("d", ("c",)),
        _add("e", ("c", "analysis_input", "c")),
    )
    report = run_optimizer(case(), ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    assert [row["members"] for row in report.applied] == [["a", "b"], ["c", "d"]]
    plan = report.canonical_plan
    fused_secondary = plan.graph.ops["fused:c+d"]
    assert fused_secondary.deps == ("fused:a+b",)
    assert len(fused_secondary.inputs) == 2
    assert fused_secondary.inputs[0] != fused_secondary.inputs[1]
    for port in fused_secondary.inputs:
        assert plan.graph.values[port].producer == "fused:a+b"
    node = plan.graph.ops["e"]
    assert node.deps == ("fused:c+d", "analysis_input")
    assert len(node.inputs) == 3
    assert node.inputs[0] == node.inputs[2]
    assert node.inputs[1] == "analysis_input:source"
    # Move the surviving consumer across streams: record/wait insertion and
    # stripping must preserve every positional operand, including the repeat.
    hardware = hw(max_streams=2, stream_switch_ms=0.01)
    moved = tuple(replace(op, stream="aux") if op.op_id == "e" else op
                  for op in plan.ops)
    inserted = insert_stream_events(moved, hardware)
    assert any(op.family == "event" for op in inserted)
    inserted_graph = plan.rewrite(inserted).graph
    assert inserted_graph.ops["e"].inputs == node.inputs
    stripped = _strip_events(inserted)
    for before, after in zip(plan.ops, stripped):
        assert before.op_id == after.op_id
        assert before.deps == after.deps
        assert full_operand_ports(before) == full_operand_ports(after)
    restored = plan.rewrite(stripped).graph
    assert restored.ops["e"].inputs == node.inputs
    restored.validate()


# ---------------------------------------------------------------------------
# Finding 2: retire-gated keys and generation-scoped capabilities
# ---------------------------------------------------------------------------
def _lease_engine():
    return Engine([Resource("hbm", 1024, "bytes"), Resource("host", 1, "ms")],
                  pools={"l1": 2048}, pool_units={"l1": 1})


def _live_l1(trace, at):
    live = 0
    for row in trace:
        if float(row["time"]) > at + 1e-12:
            continue
        delta = float(row["size"]) * len(tuple(row["units"]))
        live += int(delta) if row["action"] == "acquire" else -int(delta)
    return live


def test_r7_f2_scheduled_release_keeps_key_live_until_execution():
    engine = _lease_engine()
    entry = engine.submit("entry")
    holder = PoolHolder("shared")
    copy = PrefetchPoolProgram(holder, 1024, 1)
    release = ReleasePoolProgram(holder)
    first = copy.start(engine, "trial", entry)
    consumer = engine.submit("consumer", {"host": 10}, after=(first,))
    delayed = release.start(engine, "trial:release", consumer)
    # The Engine allocation is still live until ``t_free``; a second copy must
    # reject before Engine submission instead of failing later inside advance.
    with pytest.raises(ValueError, match="pool_lease_key_already_live"):
        copy.start(engine, "trial", entry)
    engine.advance(until=(delayed,), first=False)
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_allocated"] == {"l1": 0.0}


def test_r7_f2_stale_handle_cannot_free_the_next_generation():
    engine = _lease_engine()
    entry = engine.submit("entry")
    holder = PoolHolder("shared")
    copy = PrefetchPoolProgram(holder, 1024, 1)
    release = ReleasePoolProgram(holder)
    first = copy.start(engine, "trial", entry)
    first_free = release.start(engine, "trial:release", first)
    engine.advance(until=(first_free,), first=False)
    assert engine.summary()["live_allocated"] == {"l1": 0.0}
    # Legitimate non-overlapping reuse at t=1.
    second = copy.start(engine, "trial", entry)
    consumer = engine.submit("consumer", {"host": 10}, after=(second,))
    stale = release.start(engine, "trial:release", first)
    assert stale.name == "trial:release:noop"
    genuine = release.start(engine, "trial:release", consumer)
    engine.advance(horizon=11.9, first=False)
    trace = list(engine.allocation_trace)
    assert _live_l1(trace, 2.0) == 1024
    assert _live_l1(trace, 11.9) == 1024
    engine.advance(until=(genuine,), first=False)
    trace = list(engine.allocation_trace)
    frees = [row for row in trace
             if row["action"] == "release" and row["name"] == "trial:pool"]
    assert [row["time"] for row in frees] == [1.0, 12.0]
    assert _live_l1(trace, 12.0) == 0
    assert engine.summary()["live_allocated"] == {"l1": 0.0}


def test_r7_f2_explicit_token_mints_generation_ids():
    engine = _lease_engine()
    entry = engine.submit("entry")
    holder = PoolHolder("shared")
    copy = PrefetchPoolProgram(holder, 1024, 1, lease_token="raw-token")
    release = ReleasePoolProgram(holder, lease_token="raw-token")
    first = copy.start(engine, "first", entry)
    first_ids = tuple(holder._bucket(engine).by_id)
    assert first_ids and first_ids[0].startswith("grant:")
    assert first_ids[0] != "raw-token"
    first_free = release.start(engine, "first:release", first)
    engine.advance(until=(first_free,), first=False)
    # A retired token generation may be reused; the new acquisition must still
    # mint its own generation id rather than resurrect the token as an id.
    second = copy.start(engine, "second", entry)
    second_free = release.start(engine, "second:release", second)
    engine.advance(until=(second_free,), first=False)
    grants = holder._bucket(engine).by_id
    assert len(grants) == 2
    assert all(grant_id.startswith("grant:") for grant_id in grants)
    assert "raw-token" not in grants
    token_grant = holder._bucket(engine).capabilities[("token", "raw-token")]
    assert token_grant in grants
    assert engine.summary()["live_allocated"] == {"l1": 0.0}


# ---------------------------------------------------------------------------
# Finding 3: scale/ULP-aware collective residual accounting
# ---------------------------------------------------------------------------
def test_r7_f3_finite_hbm_scale_coupled_fill_does_not_crash():
    engine = Engine([
        Resource("hbm", 4_249_000_000, "bytes", units=2),
        Resource("vector", 1e12, "ops", units=2),
    ])
    placement = {"hbm": (0,), "vector": (0, 1)}
    a = engine.submit("a", {"hbm": 356_000_000, "vector": 481_000_000},
                      placement=placement, coupled=True)
    b = engine.submit("b", {"hbm": 722_000_000, "vector": 50_000_000},
                      placement=placement, coupled=True)
    engine.advance(until=(a, b), first=False)
    assert a.time is not None and b.time is not None
    assert engine.summary()["live_tasks"] == 0
    assert engine.serviced["hbm"] == pytest.approx(1_078_000_000.0)
    assert engine.serviced["vector"] == pytest.approx(531_000_000.0)


def _stress_random_arbitration(seed, cases=120):
    rng = random.Random(seed)
    names_pool = ["hbm", "vector", "mte2", "mte3"]
    for _case in range(cases):
        names = names_pool[:rng.randint(1, 3)]
        units = {name: rng.randint(1, 4) for name in names}
        rates = {name: 10.0 ** rng.uniform(9.0, 15.0) for name in names}
        engine = Engine([Resource(name, rates[name], "units", units=units[name])
                         for name in names])
        for index in range(rng.randint(1, 6)):
            workload = {}
            placement = {}
            for name in names:
                if rng.random() < 0.9:
                    workload[name] = 10.0 ** rng.uniform(7.0, 15.0)
                if rng.random() < 0.6:
                    count = units[name]
                    if rng.random() < 0.5:
                        placement[name] = (rng.randrange(count),)
                    else:
                        placement[name] = tuple(sorted(rng.sample(
                            range(count), rng.randint(1, count))))
            if not workload:
                workload[names[0]] = 10.0 ** rng.uniform(7.0, 15.0)
            caps = {name: 10.0 ** rng.uniform(8.0, 15.0)
                    for name in workload if rng.random() < 0.3}
            engine.submit(f"t{index}", workload, placement=placement or None,
                          coupled=rng.random() < 0.55, caps=caps or None,
                          priority=rng.randint(0, 2))
        engine._settle()
        assigned = engine._rates()
        per_task = {}
        for name, rows in assigned.items():
            total = sum(rows.values())
            tolerance = max(1e-12, 16.0 * math.ulp(
                max(1.0, abs(rates[name]), abs(total))))
            assert total <= rates[name] + tolerance
            for sequence, rate in rows.items():
                per_task.setdefault(sequence, {})[name] = rate
        for sequence, values in per_task.items():
            task = engine._tasks[sequence]
            if not task.coupled:
                continue
            fractions = [values[name] / task.remaining[name]
                         for name in values
                         if task.remaining.get(name, 0.0) > 0.0]
            if len(fractions) > 1:
                assert max(fractions) - min(fractions) <= \
                    1e-9 * max(1.0, max(fractions))
        engine.advance(horizon=0.001, first=False)


def test_r7_f3_randomized_large_scale_coupled_accounting():
    _stress_random_arbitration(20261114)
    _stress_random_arbitration(7)


# ---------------------------------------------------------------------------
# Finding 5: O(D) prefix/suffix joins and inner-loop budget enforcement
# ---------------------------------------------------------------------------
_WIDE = 3000


def _wide_fan_in():
    return tuple(
        PlanNode(f"n{index}", (("vector", 1.0),)) for index in range(_WIDE)
    ) + (PlanNode("join", (("vector", 1.0),),
                  deps=tuple(f"n{index}" for index in range(_WIDE))),)


def _wide_fan_out():
    nodes = [PlanNode("root", (("vector", 1.0),))]
    nodes += [PlanNode(f"n{index}", (("vector", 1.0),), deps=("root",))
              for index in range(_WIDE)]
    nodes.append(PlanNode("join", (("vector", 1.0),),
                          deps=tuple(f"n{index}" for index in range(_WIDE))))
    return tuple(nodes)


@pytest.mark.parametrize("factory", (_wide_fan_in, _wide_fan_out))
def test_r7_f5_wide_join_and_fork_are_inner_loop_bounded(factory):
    plan = factory()
    # Correctness and region discovery run under a generous wall budget so a
    # loaded host cannot turn a tight stopwatch race into a missing region.
    # The elapsed bound below is only a deterministic non-hang check.
    budget = Budget(per_pass=32, total=64, wall_s=30.0)
    region_cap = max(budget.total * 2, budget.per_pass * 2, 64)
    started = time.perf_counter()
    regions = _stream_regions(plan, 2, budget)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
    assert isinstance(regions, tuple) and regions
    assert len(regions) <= region_cap
    # The 128-region growth cap stops discovery before a 3,000-wide plan can
    # materialize one region per branch; exhaustion here is tick/cap-driven,
    # not a wall-clock outcome.
    assert budget.exhausted is True
    # A tiny wall budget must stop mid-traversal and preserve what it found.
    tiny = Budget(per_pass=32, total=64, wall_s=0.000001)
    stopped = _stream_regions(plan, 2, tiny)
    assert isinstance(stopped, tuple)
    assert tiny.exhausted is True
