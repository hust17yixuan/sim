"""Behavioral checks for the A5-T01 common contract.

The fixtures below fix independent expectations for typed values, ports,
unit-bearing work, state/allocation effects and the existing Engine mapping.
They do not load weights, initialize devices or claim measured accuracy.
Engine time is the existing millisecond clock; the 4/5/6 fixture is the
published us example scaled 1:1 into that clock and labeled as a fixture.
"""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import KernelProgram, stage_from_fragment
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (
    AllocationEffect, FieldDisposition, ProgramFragment, ResourcePolicy,
    SemanticAttr, SemanticRef, StateEffect, WorkItem, WorkSet,
)
from mindie_sim.schema.subgraphs import (
    EventPortSpec, PortSet, StatePortSpec, TensorPortSpec,
)
from mindie_sim.schema.tensors import TensorLayout, TensorValue


def _fragment(name, resource, amount, unit="bytes", **kwargs):
    return ProgramFragment(
        name, work=WorkSet((WorkItem(resource, amount, unit),)), **kwargs)


def test_tensor_view_and_rebinding_preserve_typed_representation():
    value = TensorValue(
        "producer", "bf16", TensorLayout.contiguous((2, 4)), output="y",
        quant="w8a8", layout_kind="nz", storage="buf0", alias_of="base",
        state_version=3, binding="event", binding_name="kv_ready",
    )
    narrowed = value.narrow(1, 1, 2)
    assert narrowed.layout.shape == (2, 2)
    for name in ("dtype", "quant", "layout_kind", "storage", "alias_of",
                 "state_version", "binding", "binding_name"):
        assert getattr(narrowed, name) == getattr(value, name)

    rebound = value.with_binding("static")
    assert (rebound.binding, rebound.binding_name) == ("static", "")
    assert (rebound.dtype, rebound.quant, rebound.storage) == ("bf16", "w8a8", "buf0")
    assert rebound.state_version == 3
    assert value.to_dict()["layout_kind"] == "nz"

    with pytest.raises(ValueError, match="binding_unknown"):
        TensorValue("p", "bf16", TensorLayout.contiguous((1,)), binding="guessed")
    with pytest.raises(ValueError, match="binding_name_required"):
        TensorValue("p", "bf16", TensorLayout.contiguous((1,)), binding="symbolic")


def test_work_set_charges_only_issued_work_and_rejects_ambiguous_units():
    work = WorkSet((
        WorkItem("hbm", 128, "bytes"),
        WorkItem("vector", 64, "ops"),
        WorkItem("hbm", 4096, "bytes", kind="useful"),
    ))
    assert work.service_work() == {"hbm": 128.0, "vector": 64.0}
    assert work.by_resource("useful") == {"hbm": 4096.0}
    assert not WorkSet().service_work()

    with pytest.raises(ModelingError, match="work_set_duplicate"):
        WorkSet((WorkItem("hbm", 1, "bytes"), WorkItem("hbm", 2, "bytes")))
    with pytest.raises(ModelingError, match="work_set_unit_conflict"):
        WorkSet((WorkItem("hbm", 1, "bytes"),
                 WorkItem("hbm", 1, "transactions", kind="useful")))
    with pytest.raises(ModelingError, match="work_item_amount_invalid"):
        WorkItem("hbm", -1, "bytes")


def test_ports_and_occurrence_preserve_typed_scope_without_guessing():
    occurrence = Occurrence(
        "kernel", "native.call0", invocation="inv1", phase="decode", serial=2,
        request_ids=("r1", "r2"), batch_ids=("b1",), rank=3, stream="aux",
        generation=4,
    )
    ports = PortSet(
        tensor_in=(TensorPortSpec(
            "hidden", "in", "activation", dtype="bf16", layout_kind="nz",
            scope="operator"),),
        state_out=(StatePortSpec(
            "kv", "out", "paged_kv", scope="layer", unit="bytes"),),
        event_in=(EventPortSpec(
            "kv_ready", "in", "ready", generation=4),),
    )
    fragment = ProgramFragment(
        "fused_call", occurrence=occurrence, semantic_refs=None, ports=ports,
        attrs=(SemanticAttr("epsilon", 1e-6, source="declared"),),
    )
    payload = fragment.to_dict()
    assert fragment.semantic_refs == ()
    assert payload["occurrence"]["rank"] == 3
    assert payload["ports"]["tensor_in"][0]["dtype"] == "bf16"
    assert payload["ports"]["state_out"][0]["representation"] == "paged_kv"
    assert payload["ports"]["event_in"][0]["generation"] == 4
    assert payload["attrs"][0]["value"] == 1e-6

    with pytest.raises(ModelingError, match="port_set_name_duplicate"):
        PortSet(
            tensor_in=(TensorPortSpec("same", "in", "activation"),),
            state_out=(StatePortSpec("same", "out", "cache"),),
        )
    with pytest.raises(ModelingError, match="event_port_kind_unknown"):
        EventPortSpec("bad", "in", "guessed_event")
    with pytest.raises(ValueError, match="occurrence_request_ids_duplicate"):
        Occurrence("request", "r", request_ids=("r1", "r1"))
    with pytest.raises(TypeError, match="scope_ids_require_sequence"):
        Occurrence("request", "r", request_ids="r1")


def test_fragment_requires_explicit_state_conflict_and_field_disposition():
    with pytest.raises(ModelingError, match="commit_version_required"):
        StateEffect("state", "commit")
    with pytest.raises(ModelingError, match="field_disposition_source_required"):
        FieldDisposition("activation", "rewrite")

    conflict = (
        StateEffect("cache", "commit", version=8),
        StateEffect("cache", "commit", version=9),
    )
    with pytest.raises(ModelingError, match="state_effect_conflict_requires_arbiter"):
        ProgramFragment("bad", state_effects=conflict)
    accepted = ProgramFragment(
        "ok", state_effects=(
            StateEffect("cache", "commit", version=8, arbiter="winner:A"),
            StateEffect("cache", "commit", version=9, arbiter="winner:B"),
        ),
    )
    assert len(accepted.state_effects) == 2
    assert all(effect.arbiter for effect in accepted.state_effects)


def test_fragment_policies_are_explicit_and_non_preemptive():
    with pytest.raises(ModelingError, match="resource_preemption_unsupported"):
        ResourcePolicy(sharing="fifo", preemptive=True)
    with pytest.raises(ModelingError, match="resource_sharing_unknown"):
        ResourcePolicy(sharing="round_robin")
    with pytest.raises(ModelingError, match="policy_resource_unused"):
        ProgramFragment(
            "bad", work=WorkSet((WorkItem("hbm", 1, "bytes"),)),
            policies={"unused": ResourcePolicy()},
        )
    fragment = ProgramFragment(
        "ok", work=WorkSet((WorkItem("hbm", 1, "bytes"),)),
        policies={"hbm": ResourcePolicy(sharing="fifo", priority=2)},
    )
    assert fragment.policy("hbm").engine_kwargs() == {"policy": "fifo"}
    assert fragment.policy("other").sharing == "fair"


def test_fragment_lowers_to_existing_stage_without_executing():
    assert ProgramFragment("twice", repeats=2).repeats == 2
    with pytest.raises(ModelingError, match="repeats_invalid"):
        ProgramFragment("bad", repeats=0)
    fragment = ProgramFragment(
        "transfer",
        work=WorkSet((WorkItem("hbm", 128, "bytes"),
                      WorkItem("hbm", 4096, "bytes", kind="useful"))),
        allocations=(
            AllocationEffect.acquire("staging", "l2", 128),
            AllocationEffect.release("staging", "l2", 128),
        ),
    )
    stage = stage_from_fragment(fragment)
    assert stage.work == {"hbm": 128.0}
    assert stage.work_units == {"hbm": "bytes"}
    assert [item.name for item in stage.acquire] == ["staging"]
    assert [item.name for item in stage.free] == ["staging"]
    assert stage.fragment is fragment

    engine = Engine([Resource("hbm", 64, unit="bytes", policy="fair")],
                    pools={"l2": 1024})
    ready = engine.submit("ready")
    done = KernelProgram((stage,)).start(engine, "transfer", ready)
    engine.advance(until=(done,), first=False)
    assert done.time == 2
    assert engine.serviced["hbm"] == 128
    assert not engine.allocations


def test_stage_projection_rejects_semantics_it_cannot_execute():
    work = WorkSet((WorkItem("hbm", 64, "bytes"),))
    # A repeats=2 fragment would otherwise execute one 64 B stage and
    # silently under-count the second repetition.  Reject the projection.
    repeated = ProgramFragment("repeat", work=work, repeats=2)
    assert repeated.stage_projection_blockers == ("repeats",)
    assert not repeated.supports_stage_projection
    with pytest.raises(ModelingError, match="stage_from_fragment_unsupported:repeats"):
        stage_from_fragment(repeated)

    conditional = ProgramFragment(
        "conditional", work=work,
        allocations=(AllocationEffect.acquire(
            "buf", "l2", 64, condition="if input.ready"),))
    with pytest.raises(
            ModelingError,
            match="stage_from_fragment_unsupported:conditional_allocation"):
        stage_from_fragment(conditional)

    program_scoped = ProgramFragment(
        "program", work=work,
        allocations=(AllocationEffect.acquire(
            "buf", "l2", 64, lifetime="program"),))
    with pytest.raises(
            ModelingError,
            match="stage_from_fragment_unsupported:allocation_lifetime"):
        stage_from_fragment(program_scoped)

    stateful = ProgramFragment(
        "stateful", work=work,
        state_effects=(StateEffect("cache", "read"),))
    with pytest.raises(
            ModelingError, match="stage_from_fragment_unsupported:state_effects"):
        stage_from_fragment(stateful)

    policy = ProgramFragment(
        "policy", work=work,
        policies={"hbm": ResourcePolicy(sharing="fifo")})
    with pytest.raises(
            ModelingError, match="stage_from_fragment_unsupported:policies"):
        stage_from_fragment(policy)

    semantic = ProgramFragment(
        "semantic", work=work,
        attrs=(SemanticAttr("epsilon", 1e-6),))
    with pytest.raises(
            ModelingError, match="stage_from_fragment_unsupported:attrs"):
        stage_from_fragment(semantic)

    # Ports, occurrence, semantic refs and notes are identity/assembly
    # metadata; they do not change the single static stage's service, and the
    # complete fragment remains available on Stage.fragment.
    supported = ProgramFragment(
        "supported", work=work,
        occurrence=Occurrence("stage", "supported"),
        ports=PortSet(event_out=(EventPortSpec("done", "out", "complete"),)),
    )
    assert supported.supports_stage_projection
    stage = stage_from_fragment(supported)
    assert stage.fragment is supported
    assert stage.work == {"hbm": 64.0}
    assert stage.work_units == {"hbm": "bytes"}


def test_static_projection_rejects_unbound_event_ports():
    work = WorkSet((WorkItem("hbm", 64, "bytes"),))
    waiting = ProgramFragment(
        "wait_generation", work=work,
        ports=PortSet(event_in=(EventPortSpec(
            "kv_ready", "in", "ready", generation=2),)),
    )
    assert waiting.stage_projection_blockers == ("event_in",)
    assert not waiting.supports_stage_projection
    with pytest.raises(
            ModelingError, match="stage_from_fragment_unsupported:event_in"):
        stage_from_fragment(waiting)

    commit = ProgramFragment(
        "commit", work=work,
        ports=PortSet(event_out=(EventPortSpec(
            "commit", "out", "commit"),)),
    )
    assert commit.stage_projection_blockers == ("event_out",)
    with pytest.raises(
            ModelingError, match="stage_from_fragment_unsupported:event_out"):
        stage_from_fragment(commit)

    double_complete = ProgramFragment(
        "double_complete", work=work,
        ports=PortSet(event_out=(
            EventPortSpec("done_a", "out", "complete"),
            EventPortSpec("done_b", "out", "complete"),
        )),
    )
    assert double_complete.stage_projection_blockers == ("event_out",)

    # A default already-ready single fragment remains usable: the caller's
    # ready event is the one readiness for data ports, and one completion
    # output is represented by the returned Stage event.
    supported = ProgramFragment(
        "ready_single", work=work,
        ports=PortSet(
            tensor_in=(TensorPortSpec("hidden", "in", "activation", dtype="bf16"),),
            event_out=(EventPortSpec("done", "out", "complete"),),
        ),
    )
    assert supported.supports_stage_projection
    stage = stage_from_fragment(supported)
    assert stage.fragment is supported
    assert stage.work == {"hbm": 64.0}


def test_static_projection_passes_priority_to_existing_engine():
    engine = Engine(
        [Resource("H", 64, unit="bytes", policy="fifo")], trace_limit=10)
    ready = engine.submit("ready")
    low = ProgramFragment(
        "low", work=WorkSet((WorkItem("H", 128, "bytes"),)), priority=1)
    high = ProgramFragment(
        "high", work=WorkSet((WorkItem("H", 64, "bytes"),)), priority=0)
    low_done = KernelProgram((stage_from_fragment(low),)).start(
        engine, "low", ready)
    high_done = KernelProgram((stage_from_fragment(high),)).start(
        engine, "high", ready)
    engine.advance(until=(low_done, high_done), first=False)
    # priority=0 is admitted before priority=1 on the same FIFO resource;
    # without priority pass-through the submission order would win instead.
    assert (high_done.time, low_done.time) == (1, 3)
    assert engine.serviced["H"] == 192


def _run_policy(order, sharing):
    """Run the published 4/5/6us counterexample on the existing Engine."""
    engine = Engine([
        Resource("H", 64, unit="bytes",
                 **ResourcePolicy(sharing=sharing).engine_kwargs()),
        Resource("C", 1, unit="service",
                 **ResourcePolicy(sharing=sharing).engine_kwargs()),
    ])
    ready = engine.submit("ready")
    fragments = {"A": _fragment("A", "H", 64),
                 "B": _fragment("B", "H", 128),
                 "C": _fragment("C", "C", 3, unit="service")}
    events = {}
    for name in order:
        events[name] = KernelProgram((stage_from_fragment(fragments[name]),)).start(
            engine, name, ready)
    events["C"] = KernelProgram((stage_from_fragment(fragments["C"]),)).start(
        engine, "C", events["A"])
    engine.advance(until=tuple(events.values()), first=False)
    return ({name: event.time for name, event in events.items()},
            dict(engine.serviced))


def test_resource_sharing_counterexample_is_4_5_6_with_conserved_work():
    fifo_a, fifo_a_work = _run_policy(("A", "B"), "fifo")
    fluid, fluid_work = _run_policy(("A", "B"), "fair")
    fifo_b, fifo_b_work = _run_policy(("B", "A"), "fifo")

    assert (fifo_a["A"], fifo_a["B"], fifo_a["C"]) == (1, 3, 4)
    assert (fluid["A"], fluid["B"], fluid["C"]) == (2, 3, 5)
    assert (fifo_b["A"], fifo_b["B"], fifo_b["C"]) == (3, 2, 6)
    for work in (fifo_a_work, fluid_work, fifo_b_work):
        assert work["H"] == 192
        assert work["C"] == 3


def test_waiting_task_holds_its_slot_without_inventing_work():
    engine = Engine([Resource("C", 64, unit="credits")], pools={"S": 1},
                    trace_limit=100)
    ready = engine.submit("ready")
    a_fragment = ProgramFragment(
        "A", work=WorkSet((WorkItem("C", 128, "credits"),)),
        allocations=(
            AllocationEffect.acquire("A_slot", "S", 1, (0,)),
            AllocationEffect.release("A_slot", "S", 1, (0,)),
        ),
    )
    b_fragment = ProgramFragment(
        "B", work=WorkSet((WorkItem("C", 64, "credits"),)),
        allocations=(
            AllocationEffect.acquire("B_slot", "S", 1, (0,)),
            AllocationEffect.release("B_slot", "S", 1, (0,)),
        ),
    )
    a = KernelProgram((stage_from_fragment(a_fragment, latency=5),)).start(
        engine, "A", ready)
    b_ready = engine.submit("B_ready", release=1)
    b = KernelProgram((stage_from_fragment(b_fragment),)).start(
        engine, "B", b_ready)
    engine.advance(until=(a, b), first=False)

    assert (a.time, b.time) == (5, 6)
    assert engine.serviced["C"] == 192
    assert engine.busy_ms["C"] == 3
    rows = {row["task"]: row for row in engine.trace}
    assert rows["A:A"]["start_ms"] == 0
    assert rows["A:A"]["end_ms"] == 5
    assert rows["B:B"]["start_ms"] == 5
    assert rows["B:B"]["end_ms"] == 6
    assert not engine.allocations


def test_same_time_events_follow_explicit_submission_order():
    engine = Engine([], trace_limit=10)
    first = engine.submit("first")
    second = engine.submit("second")
    joined = engine.submit("joined", after=(second, first))
    engine.advance(until=(joined,), first=False)
    assert (first.time, second.time, joined.time) == (0, 0, 0)
    assert [row["task"] for row in engine.trace] == ["first", "second", "joined"]

    reverse = Engine([], trace_limit=10)
    later = reverse.submit("later")
    earlier = reverse.submit("earlier")
    reverse.submit("joined", after=(later, earlier))
    reverse.advance(until=(), horizon=0)
    assert [row["task"] for row in reverse.trace] == [
        "later", "earlier", "joined"]


def test_coupled_stage_advances_both_resources_in_declared_proportion():
    engine = Engine([Resource("r1", 10), Resource("r2", 10)], trace_limit=10)
    ready = engine.submit("ready")
    fragment = ProgramFragment(
        "coupled",
        work=WorkSet((WorkItem("r1", 30, "ops"), WorkItem("r2", 10, "ops"))),
        coupled=True,
    )
    done = KernelProgram((stage_from_fragment(fragment),)).start(
        engine, "x", ready)
    engine.advance(until=(done,), first=False)

    assert done.time == 3
    assert engine.serviced == {"r1": 30, "r2": 10}
    assert engine.busy_ms == {"r1": 3, "r2": 3}
    row = next(row for row in engine.trace if row["task"] == "x:coupled")
    assert row["coupled"] is True

    from mindie_sim.prediction.program import Stage
    independent = Engine([Resource("r1", 10), Resource("r2", 10)])
    ready = independent.submit("ready")
    done = KernelProgram((Stage("split", {"r1": 30, "r2": 10}),)).start(
        independent, "x", ready)
    independent.advance(until=(done,), first=False)
    assert independent.busy_ms == {"r1": 3, "r2": 1}
