"""NR-S2-FIX11: Round-11 contraction closure for control-only boundaries.

[Role] Reproduce the three Round-11 contraction defects: a control-only
  consumer losing genuine positional data operands, an explicitly bound caller
  event/barrier operand being dropped as if it were an unbound scheduling
  marker, and the fused node rejecting a legal slot when a later member's
  predecessor sits between members or after a generated event wait.
[Invariants] A surviving consumer keeps its complete original positional data
  sequence across scheduling deduplication; an explicit ``fusion_input_ports``
  binding on a caller marker is retargeted through the retained-port map while
  an unbound marker keeps its control-only stripping; the fused node is placed
  after every incoming predecessor and before every outgoing consumer, with a
  deterministic stable ordering only when no single survivor slot is legal.  A
  genuine dependency cycle still raises ``fusion_boundary_order_invalid``.
  Explicit data survives empty scheduling deps; invalid producers fail closed.
[Timeouts & Locks] Pure in-process simulation; no clocks, sockets or locks.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from collections import UserDict
from dataclasses import replace
from types import MappingProxyType

import pytest

from mindie_sim.prediction.fusion_bridge import (
    _strip_events, add_prefetch, apply_prefetch_rewrite, apply_rewrite, canonical_plan,
    canonical_stream_move, insert_stream_events, rewrite_fusion, settle_prefetch_rewrite,
)
from mindie_sim.prediction.fusion_graph import FeasibilityError
from mindie_sim.prediction.fusion_ports import (
    CONTROL_METADATA_KEYS, full_operand_ports, input_port_entries,
)
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_vector,
)
from mindie_sim.prediction.fusion_projection import project_graph
from mindie_sim.prediction.weight_prefetch import PrefetchDecision

from graph_opt_fixtures import hw
from test_graph_optimizer_r8_adversarial import _add, case

_HARDWARE = hw(max_streams=2, stream_switch_ms=0.01)
_POSITIONS = (("x", "x:out"), ("y", "y:out"), ("x", "x:out"))


def _contract(plan, members):
    candidate = synthesize_vector(
        plan.graph, members, tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    rewrite = rewrite_fusion(plan, candidate)
    result = apply_rewrite(plan, rewrite)
    result.graph.validate()
    return result, rewrite


def _assert_round_trips(result, op_id, expected):
    variants = (
        _strip_events(result.ops),
        insert_stream_events(result.ops, _HARDWARE),
        insert_stream_events(insert_stream_events(result.ops, _HARDWARE),
                             _HARDWARE),
    )
    for ops in variants:
        fresh = result.rewrite(ops)
        fresh.graph.validate()
        assert fresh.graph.ops[op_id].inputs == expected
        assert fresh.evaluate()["duration_ms"] > 0


@pytest.mark.parametrize("key", CONTROL_METADATA_KEYS)
@pytest.mark.parametrize("bindings", ("absent", "partial"))
def test_r11_control_only_successor_keeps_repeated_data_positions(
        key, bindings):
    metadata = ((key, ("a", "a")),)
    if bindings == "partial":
        metadata += (("fusion_input_ports", _POSITIONS[:1]),)
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("a",)),
        _add("x", ("analysis_input",)),
        _add("y", ("analysis_input",)),
        _add("c", ("x", "y", "x", "a"), metadata=metadata),
    ))
    expected = ("x:out", "y:out", "x:out")
    assert plan.graph.ops["c"].inputs == expected
    after, rewrite = _contract(plan, ("a", "b"))
    consumer = next(op for op in after.ops if op.op_id == "c")
    assert after.graph.ops["c"].inputs == expected
    assert dict(consumer.metadata)[key] == (rewrite.fused.op_id,)
    assert consumer.deps == ("x", "y", rewrite.fused.op_id)
    assert "a" not in after.graph.external_inputs
    _assert_round_trips(after, "c", expected)


@pytest.mark.parametrize("key", CONTROL_METADATA_KEYS)
def test_r11_repeated_member_operand_keeps_multiplicity(key):
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("a",)),
        _add("x", ("analysis_input",)),
        _add("c", ("a", "x", "a"), metadata=((key, ("b",)),)),
    ))
    expected = ("a:out", "x:out", "a:out")
    assert plan.graph.ops["c"].inputs == expected
    after, rewrite = _contract(plan, ("a", "b"))
    retained = dict(dict(rewrite.fused.metadata)["retained_port_map"])["a:out"]
    expected = (retained, "x:out", retained)
    assert after.graph.ops["c"].inputs == expected
    consumer = next(op for op in after.ops if op.op_id == "c")
    assert dict(consumer.metadata)[key] == (rewrite.fused.op_id,)
    assert consumer.deps == (rewrite.fused.op_id, "x")
    assert "a" not in after.graph.external_inputs
    assert "b" not in after.graph.external_inputs
    _assert_round_trips(after, "c", expected)


@pytest.mark.parametrize("name,family", (
    ("gate", "event"),
    ("gate", "barrier"),
    ("event:gate", "add"),
    ("barrier:gate", "add"),
))
def test_r11_explicit_caller_marker_binding_is_retargeted(name, family):
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("a",)),
        _add(name, ("a",), family=family,
             metadata=(("fusion_input_ports", (("a", "a:out"),)),)),
        _add("c", (name,),
             metadata=(("fusion_input_ports", ((name, f"{name}:out"),)),)),
    ))
    assert plan.graph.ops[name].inputs == ("a:out",)
    after, rewrite = _contract(plan, ("a", "b"))
    retained = dict(dict(rewrite.fused.metadata)["retained_port_map"])["a:out"]
    marker = next(op for op in after.ops if op.op_id == name)
    assert after.graph.ops[name].inputs == (retained,)
    assert dict(marker.metadata)["fusion_input_ports"] == (
        (rewrite.fused.op_id, retained),)
    assert after.evaluate()["duration_ms"] > 0
    assert "a" not in after.graph.external_inputs
    moved = tuple(replace(op, stream="aux") if op.op_id in {name, "c"} else op
                  for op in after.ops)
    for ops in (insert_stream_events(moved, _HARDWARE),
                _strip_events(insert_stream_events(moved, _HARDWARE))):
        fresh = after.rewrite(ops)
        fresh.graph.validate()
        assert fresh.graph.ops[name].inputs == (retained,)
        assert "a" not in fresh.graph.external_inputs
        assert fresh.evaluate()["duration_ms"] > 0


def test_r11_explicit_marker_binding_beats_same_name_control():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("a",)),
        _add("gate", ("a",), family="event",
             metadata=(("fusion_input_ports", (("a", "a:out"),)),
                       ("control_deps", ("a",)))),
        _add("c", ("gate",),
             metadata=(("fusion_input_ports", (("gate", "gate:out"),)),)),
    ))
    after, rewrite = _contract(plan, ("a", "b"))
    retained = dict(dict(rewrite.fused.metadata)["retained_port_map"])["a:out"]
    gate = next(op for op in after.ops if op.op_id == "gate")
    assert after.graph.ops["gate"].inputs == (retained,)
    assert dict(gate.metadata)["fusion_input_ports"] == (
        (rewrite.fused.op_id, retained),)
    assert "a" not in after.graph.external_inputs
    assert after.evaluate()["duration_ms"] > 0
    moved = tuple(replace(op, stream="aux") if op.op_id in {"gate", "c"} else op
                  for op in after.ops)
    for ops in (insert_stream_events(moved, _HARDWARE),
                _strip_events(insert_stream_events(moved, _HARDWARE))):
        fresh = after.rewrite(ops)
        fresh.graph.validate()
        marker = next(op for op in fresh.ops if op.op_id == "gate")
        assert fresh.graph.ops["gate"].inputs == (retained,)
        assert dict(marker.metadata)["fusion_input_ports"] == (
            (rewrite.fused.op_id, retained),)
        assert "a" not in fresh.graph.external_inputs
        assert fresh.evaluate()["duration_ms"] > 0


@pytest.mark.parametrize("key", CONTROL_METADATA_KEYS)
@pytest.mark.parametrize("count", (2, 3))
def test_r11_interleaved_predecessor_gets_a_legal_slot(key, count):
    members = ("a", "b") if count == 2 else ("a", "b", "c")
    ops = (
        _add("a", ("analysis_input",)),
        _add("gate", ("analysis_input",)),
        _add("b", ("a", "gate"), metadata=((key, ("gate",)),)),
    )
    if count == 3:
        ops += (_add("c", ("b",)),)
    ops += (_add("z", (members[-1],)),)
    plan = canonical_plan(case(), ops)
    assert plan.graph.validate()
    assert plan.evaluate()["duration_ms"] > 0
    after, rewrite = _contract(plan, members)
    order = [op.op_id for op in after.ops]
    assert order.index("gate") < order.index(rewrite.fused.op_id)
    assert order.index(rewrite.fused.op_id) < order.index("z")
    assert after.graph.ops[rewrite.fused.op_id].inputs == (
        "analysis_input:source",)
    assert after.evaluate()["duration_ms"] > 0


@pytest.mark.parametrize("key", CONTROL_METADATA_KEYS)
@pytest.mark.parametrize("dual", (False, True))
def test_r11_generated_wait_after_first_member_contracts(key, dual):
    data = ("c", "a") if dual else ("c",)
    metadata = ((key, ("a",)),)
    if dual:
        # A genuine data operand declared beside a same-name control: the
        # explicit binding keeps ``a`` in the data sequence and creates the
        # later ``event:wait:a->d`` cross-stream wait.
        metadata += (("fusion_input_ports",
                      (("c", "c:out"), ("a", "a:out"))),)
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("a",)),
        _add("c", ("b",)),
        _add("d", data, metadata=metadata),
    ))
    expected = ("c:out", "a:out") if dual else ("c:out",)
    assert plan.graph.ops["d"].inputs == expected
    moved = canonical_stream_move(plan, ("c", "d"), _HARDWARE)
    moved = insert_stream_events(moved, _HARDWARE)
    event_ids = {op.op_id for op in moved}
    assert ("event:wait:a->d" in event_ids) is dual
    plan = plan.rewrite(moved)
    plan.graph.validate()
    after, rewrite = _contract(plan, ("c", "d"))
    expected = ("b:out", "a:out") if dual else ("b:out",)
    assert after.graph.ops[rewrite.fused.op_id].inputs == expected
    assert after.evaluate()["duration_ms"] > 0
    clean = after.rewrite(_strip_events(after.ops))
    clean.graph.validate()
    assert clean.graph.ops[rewrite.fused.op_id].inputs == expected
    for op in clean.ops:
        assert op.op_id != "c" and op.op_id != "d"
        assert "c" not in op.deps and "d" not in op.deps


def test_r11_control_consumer_before_interleaved_dep_is_topo_placed():
    plan = canonical_plan(case(), (
        _add("early", (), metadata=(("control_deps", ("a",)),)),
        _add("a", ("analysis_input",)),
        _add("gate", ("analysis_input",)),
        _add("b", ("a", "gate"), metadata=(("control_deps", ("gate",)),)),
        _add("z", ("b",)),
    ))
    assert plan.graph.ops["early"].deps == ()
    after, rewrite = _contract(plan, ("a", "b"))
    order = [op.op_id for op in after.ops]
    assert order.index("gate") < order.index(rewrite.fused.op_id)
    assert order.index(rewrite.fused.op_id) < order.index("early")
    assert order.index(rewrite.fused.op_id) < order.index("z")
    assert after.evaluate()["duration_ms"] > 0


def test_r11_genuine_dependency_cycle_is_still_rejected():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("x", ("analysis_input",),
             metadata=(("control_deps", ("a",)),)),
        _add("b", ("a", "x")),
    ))
    candidate = synthesize_vector(
        plan.graph, ("a", "b"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    with pytest.raises(ValueError, match="fusion_boundary_order_invalid"):
        rewrite_fusion(plan, candidate)


_R15_CONTAINERS = (tuple, list, dict, UserDict, MappingProxyType)


def _r15_metadata(container, rows):
    return container(rows if container in (tuple, list) else dict(rows))


@pytest.mark.parametrize("container", _R15_CONTAINERS)
@pytest.mark.parametrize("key", CONTROL_METADATA_KEYS)
@pytest.mark.parametrize("deps", ((), ("a",)))
def test_r15_empty_schedule_explicit_ports_lifecycle(container, key, deps):
    ports = (("event:g", "event:g:out"), ("x", "x:out"), ("event:g", "event:g:out"))
    metadata = _r15_metadata(container, ((key, ("a", "event:g", "external_ticket")),
                                       ("fusion_input_ports", ports)))
    consumer = replace(_add("c", deps), metadata=metadata)
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)), _add("b", ("a",)),
        _add("event:g", ("analysis_input",), family="event"),
        _add("x", ("analysis_input",)), consumer,
        replace(_add("tail", ("c",)), stream="aux")))
    expected = tuple(port for _, port in ports)
    assert full_operand_ports(consumer) == ports
    after, rw = _contract(plan, ("a", "b"))
    touched = next(op for op in after.ops if op.op_id == "c")
    assert touched is not consumer and rw.fused.op_id in touched.deps
    assert input_port_entries(touched.metadata) == ports
    decision = PrefetchDecision("c", (), 256, 256, 0, 256, 0,
                                1.0, 1e6, 0.0, True, "r15", ())
    for base in (plan, after):
        inserted = insert_stream_events(base.ops, _HARDWARE)
        assert inserted != base.ops and insert_stream_events(inserted, _HARDWARE) == inserted
        for ops in (base.ops, inserted, _strip_events(inserted)):
            stage = base.rewrite(ops)
            assert stage.graph.validate() and stage.evaluate()["duration_ms"] > 0
            assert stage.graph.ops["c"].inputs == expected
            pending, pid = add_prefetch(stage, decision, _HARDWARE)
            rewrite = settle_prefetch_rewrite(pending, decision, pid, 256, _HARDWARE)
            assert input_port_entries(rewrite.consumer.metadata) == (*ports, (pid, pid + ":out"))
            settled = apply_prefetch_rewrite(pending, rewrite, pid)
            events = insert_stream_events(settled.ops, _HARDWARE)
            assert insert_stream_events(events, _HARDWARE) == events
            for final_ops in (settled.ops, events, _strip_events(events)):
                final = settled.rewrite(final_ops)
                assert final.graph.validate() and final.evaluate()["duration_ms"] > 0
                assert final.graph.ops["c"].inputs == (*expected, pid + ":out")
    assert consumer.deps == deps and input_port_entries(consumer.metadata) == ports


@pytest.mark.parametrize("container", _R15_CONTAINERS)
def test_r15_genuinely_inputless_operator_stays_untouched(container):
    c = replace(_add("c", ()), metadata=_r15_metadata(container, ()))
    plan = canonical_plan(case(), (_add("a", ("analysis_input",)), _add("b", ("a",)), c))
    after, _ = _contract(plan, ("a", "b"))
    assert full_operand_ports(c) == () and plan.graph.ops["c"].inputs == ()
    assert next(op for op in after.ops if op.op_id == "c") is c
    assert insert_stream_events((c,), _HARDWARE) == _strip_events((c,)) == (c,)
    assert after.graph.ops["c"].inputs == () and after.graph.validate()


@pytest.mark.parametrize("container", _R15_CONTAINERS)
@pytest.mark.parametrize("deps", ((), ("a",)))
@pytest.mark.parametrize("kind", ("missing", "future", "port", "owner", "unordered"))
def test_r15_explicit_ports_require_valid_causal_producers(container, deps, kind):
    port = "x:missing" if kind == "port" else "a:out" if kind == "owner" else "x:out"
    metadata = _r15_metadata(container, (("control_deps", ("a", "x")),
                                       ("fusion_input_ports", (("x", port),))))
    c = replace(_add("c", deps), metadata=metadata,
                stream="aux" if kind == "unordered" else "main")
    a, x = _add("a", ("analysis_input",)), _add("x", ("analysis_input",))
    ops = (a, c) if kind == "missing" else (a, c, x) if kind == "future" else (a, x, c)
    assert full_operand_ports(c) == (("x", port),)
    reason = "graph_input_not_ordered" if kind == "unordered" else "graph_input_port_unknown"
    with pytest.raises(FeasibilityError, match=f"^{reason}:c:"):
        canonical_plan(case(), ops)


@pytest.mark.parametrize("deps", (("a",), ("x",)))
def test_r15_transitive_cross_stream_schedule_keeps_explicit_ports(deps):
    x = replace(_add("x", ("analysis_input",)), hbm_bytes=1e9)
    a = replace(_add("a", ("x",)), stream="aux")
    c = replace(_add("c", deps, metadata=(("control_deps", ("a",)),
                 ("fusion_input_ports", (("x", "x:out"), ("x", "x:out"))))), stream="aux")
    plan = canonical_plan(case(), (x, a, c))
    for ops in (plan.ops, insert_stream_events(plan.ops, _HARDWARE),
                _strip_events(insert_stream_events(plan.ops, _HARDWARE))):
        stage = plan.rewrite(ops)
        assert stage.graph.validate() and stage.graph.ops["c"].inputs == ("x:out", "x:out")
        rows = {row["op_id"]: row for row in stage.evaluate()["timeline"]}
        assert rows["c"]["ready_ms"] >= rows["x"]["complete_ms"]


def test_r15_explicit_completeness_requires_data_producer_coverage():
    ports = (("x", "x:out"), ("z", "z:out"))
    c = _add("c", ("x", "y"), metadata=(("fusion_input_ports", ports),))
    assert full_operand_ports(c) == (("x", "x:out"), ("y", "y:out"))


@pytest.mark.parametrize("source", ("analysis_input", "declared_input"))
def test_r15_default_projection_keeps_scheduled_external_source_bindings(source):
    x = _add("x", (source,))
    c = _add("c", (), metadata=(("fusion_input_ports", ((source, source + ":out"),)),))
    graph = project_graph(case(), (x, c))
    assert graph.validate() and graph.ops["c"].inputs == (source + ":source",)
    missing = replace(c, metadata=(("fusion_input_ports", (("missing", "missing:out"),)),))
    with pytest.raises(FeasibilityError, match="^graph_input_port_unknown:c:"):
        project_graph(case(), (x, missing))
