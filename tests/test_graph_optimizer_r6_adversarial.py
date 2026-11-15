"""NR-S2-FIX6: executed Round-6 closure witnesses.

[Role] Reproduce the five Round-6 counterexamples: successive retained-port
  contraction, colliding lease invocations, per-owner coupled oversubscription,
  stochastic Cast/KV Cast admission and recursive typed stream discovery.
[Invariants] A fused node exposes every occurrence-ordered boundary tensor and
  every earlier retained live-out; a release frees only its own invocation; a
  coupled tier never exceeds any physical unit residual; stochastic rounding
  needs an explicit caller opt-in; stream discovery is iterative, budget-aware
  and never raises ``RecursionError``.
[Timeouts & Locks] Pure in-process virtual simulation; deep-DAG discovery is
  bounded by the caller's wall budget and the public optimizer's budgets.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction.engine_plan import PlanNode
from mindie_sim.prediction.fusion_bridge import (
    apply_rewrite, canonical_plan, rewrite_fusion,
)
from mindie_sim.prediction.fusion_graph import TypedGraph
from mindie_sim.prediction.fusion_pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram,
)
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_vector,
)
from mindie_sim.prediction.graph_optimizer import run_optimizer
from mindie_sim.prediction.graph_optimizer_engine import _stream_regions
from mindie_sim.prediction.optimizer_report import Budget
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import cast_chain_graph, hw, op, val

_CASE = None


def case():
    global _CASE
    if _CASE is None:
        _CASE = compose_case("Qwen3.8-27B", hardware="a3", tp=1, layers=1)
    return _CASE


def _add(op_id, deps=(), *, family="add", metadata=(), hbm=1_000_000.0,
         dtype="bf16"):
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
        padded_shape=(1, 4), launch_shape=(1, 4), hbm_bytes=hbm,
        vector_ops=2.0, dtype=dtype, deps=tuple(deps),
        metadata=tuple(metadata))


def _cast(op_id, dep, target, rounding, family="cast"):
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
        padded_shape=(1, 4), launch_shape=(1, 4), hbm_bytes=8.0,
        vector_ops=2.0, dtype=target, deps=(dep,),
        metadata=(("rounding", rounding),))


# ---------------------------------------------------------------------------
# Finding 1: successive public contractions keep both boundary operands
# ---------------------------------------------------------------------------
def test_r6_f1_public_fusion_composes_both_retained_operand_ports():
    ops = (
        _add("a", ("analysis_input",)),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", ("a", "b")),
        _add("d", ("c",)),
    )
    report = run_optimizer(case(), ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    assert [row["members"] for row in report.applied] == [
        ["a", "b"], ["c", "d"]]
    graph = report.canonical_plan.graph
    assert graph.ops["fused:c+d"].inputs == (
        "fused:a+b:a_out", "fused:a+b:out")
    assert graph.ops["fused:c+d"].deps == ("fused:a+b",)
    graph.validate()


def test_r6_f1_three_successive_public_contractions_validate():
    ops = tuple(
        item for first in ("a", "b", "c")
        for item in (_add(first + "0", ("analysis_input",)),
                     _add(first + "1", (first + "0",))))
    report = run_optimizer(case(), ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    assert [row["members"] for row in report.applied] == [
        ["a0", "a1"], ["b0", "b1"], ["c0", "c1"]]
    graph = report.canonical_plan.graph
    for name in ("a", "b", "c"):
        node = graph.ops["fused:" + name + "0+" + name + "1"]
        assert node.inputs == ("analysis_input:source",)
    graph.validate()


def test_r6_f1_retained_ports_compose_through_a_fused_producer():
    fused = _add("F", ("analysis_input",), family="native_fusion",
                 metadata=(
                     ("retained_ports", (
                         ("F:out2", "bf16", (1, 4), "contiguous", None, False),
                         ("F:out1", "bf16", (1, 4), "contiguous", None, False))),
                     ("retained_port_map", (("F:out2", "F:out2"),
                                            ("F:out1", "F:out1")))))
    d = _add("d", ("F",), metadata=(
        ("fusion_input_ports", (("F", "F:out1"),)),))
    e = _add("e", ("F", "F"), metadata=(
        ("fusion_input_ports", (("F", "F:out1"), ("F", "F:out2"))),))
    values = {
        "src": val("src", "bf16", (1, 4), storage=8),
        "F:out1": val("F:out1", "bf16", (1, 4), producer="F", storage=8),
        "F:out2": val("F:out2", "bf16", (1, 4), producer="F", storage=8),
        "d:out": val("d:out", "bf16", (1, 4), producer="d", storage=8),
        "e:out": val("e:out", "bf16", (1, 4), producer="e", storage=8),
    }
    graph = TypedGraph(1, values, {
        "F": op("F", "elementwise", "vector", ("src",),
                ("F:out1", "F:out2"), work=(("vector", 2.0), ("hbm", 8.0))),
        "d": op("d", "add", "vector", ("F:out1",), ("d:out",),
                work=(("vector", 2.0), ("hbm", 8.0))),
        "e": op("e", "add", "vector", ("F:out1", "F:out2"), ("e:out",),
                work=(("vector", 2.0), ("hbm", 8.0))),
    }, ("e:out", "d:out"))
    base = canonical_plan(case(), (fused, d, e))
    plan = replace(base, ops=(fused, d, e), graph=graph)
    candidate = synthesize_vector(
        graph, ("F", "d"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    rewrite = rewrite_fusion(plan, candidate)
    ports = dict(dict(rewrite.fused.metadata)["retained_port_map"])
    assert set(ports) == {"F:out1", "F:out2", "d:out"}
    consumer = next(item for item in rewrite.ops if item.op_id == "e")
    assert consumer.deps == (rewrite.fused.op_id,)
    assert dict(consumer.metadata)["fusion_input_ports"] == (
        (rewrite.fused.op_id, ports["F:out1"]),
        (rewrite.fused.op_id, ports["F:out2"]))
    updated = apply_rewrite(plan, rewrite)
    assert updated.graph.ops["e"].inputs == (
        ports["F:out1"], ports["F:out2"])
    updated.graph.validate()


# ---------------------------------------------------------------------------
# Finding 2: exact invocation pairing and no silent FIFO collision
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


def test_r6_f2_injective_release_pairing_and_overlap_rejection():
    engine = _lease_engine()
    entry = engine.submit("entry")
    holder = PoolHolder("shared")
    copy = PrefetchPoolProgram(holder, 1024, 1)
    first = copy.start(engine, "trial", entry)
    second = copy.start(engine, "trial:copy", entry)
    with pytest.raises(ValueError, match="pool_lease_key_already_live"):
        copy.start(engine, "trial", entry)
    with pytest.raises(ValueError, match="prefetch_invocation_name_reserved"):
        copy.start(engine, "bad:release", entry)
    consumer = engine.submit("long", {"host": 10}, after=(first,))
    release = ReleasePoolProgram(holder)
    second_free = release.start(engine, "trial:copy:release", second)
    first_free = release.start(engine, "trial:release", consumer)
    engine.advance(until=(first_free, second_free), first=False)
    trace = list(engine.allocation_trace)
    freed = {row["task"]: row for row in trace if row["action"] == "release"}
    assert freed["trial:copy:release:free"]["name"] == "trial:copy:pool"
    assert freed["trial:release:free"]["name"] == "trial:pool"
    assert freed["trial:copy:release:free"]["time"] == pytest.approx(2.0)
    assert freed["trial:release:free"]["time"] == pytest.approx(12.0)
    assert _live_l1(trace, 2.0) == 1024
    assert _live_l1(trace, 11.999) == 1024
    assert _live_l1(trace, 12.0) == 0
    # A released key may be reused by a later non-overlapping iteration.
    third = copy.start(engine, "trial", entry)
    third_free = release.start(engine, "trial:release", third)
    engine.advance(until=(third_free,), first=False)
    assert engine.summary()["live_allocated"] == {"l1": 0.0}


# ---------------------------------------------------------------------------
# Finding 3: collective per-unit coupled tier arbitration
# ---------------------------------------------------------------------------
def test_r6_f3_coupled_tier_never_oversubscribes_a_physical_owner():
    engine = Engine([Resource("hbm", 1024, "bytes", units=2),
                     Resource("vector", 10000, "ops")])
    top = engine.submit("top", {"hbm": 10000}, priority=0,
                        placement={"hbm": (0,)}, caps={"hbm": 256})
    lows = [engine.submit(name, {"hbm": 10000, "vector": 10000},
                          priority=1, placement={"hbm": (0,)}, coupled=True)
            for name in ("lo1", "lo2")]
    engine.advance(horizon=1.0, first=False)
    assert engine.service_of(top)["hbm"] == pytest.approx(256.0)
    for low in lows:
        assert engine.service_of(low) == pytest.approx(
            {"hbm": 128.0, "vector": 128.0})
    assert 256.0 + 128.0 + 128.0 == pytest.approx(512.0)

    asym = Engine([Resource("hbm", 1024, "bytes", units=2),
                   Resource("vector", 10000, "ops", units=2)])
    top0 = asym.submit("top0", {"hbm": 10000}, priority=0,
                       placement={"hbm": (0,)}, caps={"hbm": 300})
    top1 = asym.submit("top1", {"hbm": 10000}, priority=0,
                       placement={"hbm": (1,)}, caps={"hbm": 100})
    lo1 = asym.submit("lo1", {"hbm": 10000, "vector": 10000}, priority=1,
                      placement={"hbm": (0, 1), "vector": (0, 1)},
                      coupled=True)
    lo2 = asym.submit("lo2", {"hbm": 6000, "vector": 12000}, priority=1,
                      placement={"hbm": (0,), "vector": (0,)}, coupled=True)
    asym._settle()
    rates = asym._rates()
    h1, v1 = rates["hbm"][lo1.sequence], rates["vector"][lo1.sequence]
    h2, v2 = rates["hbm"][lo2.sequence], rates["vector"][lo2.sequence]
    assert h1 / 10000 == pytest.approx(v1 / 10000)
    assert h2 / 6000 == pytest.approx(v2 / 12000)
    assert rates["hbm"][top0.sequence] + h1 / 2 + h2 <= 512.0 + 1e-9
    assert rates["hbm"][top1.sequence] + h1 / 2 <= 512.0 + 1e-9
    assert v1 / 2 + v2 <= 10000.0 + 1e-9
    assert rates["vector"][lo2.sequence] == pytest.approx(v2)

    fifo = Engine([Resource("a", 100, "work", policy="fifo"),
                   Resource("b", 100, "work", policy="fair")])
    early = fifo.submit("early", {"a": 100, "b": 100}, coupled=True,
                        priority=10)
    late = fifo.submit("late", {"a": 10, "b": 10}, coupled=True,
                       priority=0, release=0.1)
    fifo.advance(until=(early, late), first=False)
    assert early.time == pytest.approx(1.0)
    assert late.time == pytest.approx(1.1)

    tiers = Engine([Resource("hbm", 1024, "bytes", units=2)])
    tier0 = tiers.submit("tier0", {"hbm": 10000}, priority=0,
                         placement={"hbm": (0,)}, caps={"hbm": 256})
    tier1 = tiers.submit("tier1", {"hbm": 10000}, priority=1,
                         placement={"hbm": (1,)}, caps={"hbm": 256})
    tier2 = tiers.submit("tier2", {"hbm": 10000}, priority=2,
                         placement={"hbm": (1,)})
    tiers.advance(horizon=1.0, first=False)
    for tier in (tier0, tier1, tier2):
        assert tiers.service_of(tier)["hbm"] == pytest.approx(256.0)


# ---------------------------------------------------------------------------
# Finding 4: stochastic Cast/KV Cast requires explicit relaxed numerics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ("cast", "kv_cast"))
def test_r6_f4_stochastic_cast_needs_explicit_opt_in(family):
    ops = (_cast("a", "analysis_input", "fp16", "stochastic", family),
           _cast("b", "a", "bf16", "stochastic", family))
    plan = canonical_plan(case(), ops)
    denied = synthesize_vector(
        plan.graph, ("a", "b"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert isinstance(denied, RejectedFusion)
    assert "relaxed_numerics_not_allowed" in denied.reasons
    allowed = synthesize_vector(
        plan.graph, ("a", "b"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv,
        allow_relaxed_numerics=True)
    assert not isinstance(allowed, RejectedFusion)
    assert allowed.numerical_policy == "relaxed"
    default = run_optimizer(case(), ops, evaluation_budget=16,
                            max_candidates_per_pass=8)
    assert not default.applied
    assert any("relaxed_numerics_not_allowed" in row.reasons
               for row in default.rejections)
    relaxed = run_optimizer(case(), ops, evaluation_budget=16,
                            max_candidates_per_pass=8,
                            allow_relaxed_numerics=True)
    assert any(row["members"] == ["a", "b"] for row in relaxed.applied)

    control = (_cast("a", "analysis_input", "fp16", "nearest", family),
               _cast("b", "a", "bf16", "nearest", family))
    control_plan = canonical_plan(case(), control)
    exact = synthesize_vector(
        control_plan.graph, ("a", "b"), tile_rows=1,
        capacity_per_owner=control_plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(exact, RejectedFusion)
    assert exact.numerical_policy in {"exact", "strict"}
    assert run_optimizer(case(), control, evaluation_budget=16,
                         max_candidates_per_pass=8).applied


@pytest.mark.parametrize("kind", ("cast", "kv_cast"))
def test_r6_f4_typed_admission_rejects_stochastic_rounding(kind):
    values, ops = cast_chain_graph("r6")
    local_values = {
        "r6x": replace(values["r6x"], dtype="bf16"),
        "r6y": replace(values["r6y"], dtype="fp16"),
        "r6z": replace(values["r6z"], dtype="bf16"),
    }
    a = replace(ops["r6a"], kind=kind,
                params=(("target_dtype", "fp16"),
                        ("rounding", "stochastic")))
    b = replace(ops["r6b"], kind=kind,
                params=(("target_dtype", "bf16"),
                        ("rounding", "stochastic")))
    graph = TypedGraph(1, local_values, {"r6a": a, "r6b": b}, ("r6z",))
    denied = run_optimizer(graph, hw(), evaluation_budget=16,
                           max_candidates_per_pass=8)
    assert not denied.applied
    assert any("relaxed_numerics_not_allowed" in row.reasons
               for row in denied.rejections)
    allowed = run_optimizer(graph, hw(), evaluation_budget=16,
                            max_candidates_per_pass=8,
                            allow_relaxed_numerics=True)
    assert any(row["members"] == ["r6a", "r6b"] for row in allowed.applied)


# ---------------------------------------------------------------------------
# Finding 5: iterative, budget-bounded stream discovery on deep valid DAGs
# ---------------------------------------------------------------------------
def _deep_plan(count):
    nodes = [PlanNode(f"n{index}", (("vector", 1.0),),
                      deps=(() if index == 0 else (f"n{index - 1}",)))
             for index in range(count)]
    nodes.append(PlanNode("side", (("vector", 1.0),)))
    nodes.append(PlanNode("join", (("vector", 1.0),),
                          deps=(f"n{count - 1}", "side")))
    return tuple(nodes)


def _deep_graph(count):
    values, ops, previous = {}, {}, None
    values["x"] = val("x", "bf16", (1,), storage=2)
    for index in range(count):
        source = "x" if previous is None else previous
        output = f"n{index}:out"
        values[output] = val(output, "bf16", (1,), producer=f"n{index}",
                             storage=2)
        ops[f"n{index}"] = op(
            f"n{index}", "elementwise", "vector", (source,), (output,),
            work=(("vector", 1.0),))
        previous = output
    values["side:out"] = val("side:out", "bf16", (1,), producer="side",
                             storage=2)
    ops["side"] = op("side", "elementwise", "vector", ("x",), ("side:out",),
                     work=(("vector", 1.0),))
    values["join:out"] = val("join:out", "bf16", (1,), producer="join",
                             storage=2)
    ops["join"] = op("join", "elementwise", "vector",
                     (previous, "side:out"), ("join:out",),
                     work=(("vector", 1.0),))
    return TypedGraph(1, values, ops, ("join:out",))


@pytest.mark.parametrize("count", (1202, 3000))
def test_r6_f5_stream_discovery_is_iterative_and_budget_bounded(count):
    plan = _deep_plan(count)
    budget = Budget(per_pass=32, total=64, wall_s=0.5)
    regions = _stream_regions(plan, 2, budget)
    assert isinstance(regions, tuple) and regions
    assert budget.exhausted is False
    tiny = Budget(per_pass=32, total=64, wall_s=0.000001)
    exhausted_regions = _stream_regions(plan, 2, tiny)
    assert isinstance(exhausted_regions, tuple)
    assert tiny.exhausted is True


def test_r6_f5_public_typed_optimizer_completes_on_deep_dags():
    for count, evaluations in ((1202, 64), (3000, 2)):
        report = run_optimizer(
            _deep_graph(count), hw(), evaluation_budget=evaluations,
            max_candidates_per_pass=32 if count == 1202 else 1,
            wall_budget_s=30.0)
        assert report.graph_operators == count + 2
        assert isinstance(report.unevaluated, tuple)
    # A caller wall budget that expires during discovery is recorded, never
    # raised as a recursion failure.
    tiny = run_optimizer(_deep_graph(1202), hw(), evaluation_budget=64,
                         max_candidates_per_pass=32, wall_budget_s=0.000001)
    assert tiny.budget_exhausted is True
    assert any(row["candidate_id"] == "stream:discovery"
               and row["reason"] == "discovery_budget_exhausted"
               for row in tiny.unevaluated)
